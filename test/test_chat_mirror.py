"""Tests for the channel-neutral mirror-link / mirror-unlink endpoints (C3)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_state

from kiro_crew.messaging.link import ChannelLink
from kiro_crew.messaging.transport import ConfiguredChannelTarget


def _make_mirror_app(state):
    from kiro_crew.dashboard.chat_mirror import (
        api_channel_targets,
        api_chat_slot_mirror_link,
        api_chat_slot_mirror_unlink,
    )

    app = web.Application()
    app["state"] = state
    app.router.add_post("/api/chat/slots/{name}/mirror-link", api_chat_slot_mirror_link)
    app.router.add_post("/api/chat/slots/{name}/mirror-unlink", api_chat_slot_mirror_unlink)
    app.router.add_get("/api/chat/channel-targets", api_channel_targets)
    return app


def _fake_transport(channel_type="telegram", proactive=True, max_message_chars=4096):
    return SimpleNamespace(
        channel_type=channel_type,
        capabilities=SimpleNamespace(
            supports_proactive_send=proactive,
            # The real TransportCapabilities always carries this; the mirror
            # backfill chunks to it instead of truncating, so the fake needs it
            # to exercise that path rather than the getattr fallback.
            max_message_chars=max_message_chars,
        ),
        send_message=AsyncMock(return_value="mid-1"),
        configured_targets=MagicMock(
            return_value=[ConfiguredChannelTarget("user:123", f"{channel_type.title()} DM · 123")]
        ),
        resolve_configured_target=AsyncMock(return_value=("123", None)),
    )


def _prep(tmp_path, monkeypatch):
    monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
    state = _make_state(tmp_path)
    state.sessions.get_mirror_link = MagicMock(return_value=None)
    state.sessions.get_slack_link = MagicMock(return_value=(None, None))
    state.get_or_create_slot("s1")
    state.push_slots_update = MagicMock()
    return state


class TestMirrorLink:
    @pytest.mark.asyncio
    async def test_configured_targets_are_listed(self, tmp_path, monkeypatch):
        state = _prep(tmp_path, monkeypatch)
        state.register_channel_transport(_fake_transport("telegram"))
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.get("/api/chat/channel-targets")
            assert resp.status == 200
            assert await resp.json() == [
                {
                    "channel_type": "telegram",
                    "target_id": "user:123",
                    "label": "Telegram DM · 123",
                    "available": True,
                    "unavailable_reason": "",
                }
            ]

    @pytest.mark.asyncio
    async def test_configured_target_is_resolved_server_side(self, tmp_path, monkeypatch):
        state = _prep(tmp_path, monkeypatch)
        transport = _fake_transport("telegram")
        state.register_channel_transport(transport)
        state.sessions.set_mirror_link = MagicMock()
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/mirror-link",
                json={"channel_type": "telegram", "target_id": "user:123"},
            )
            assert resp.status == 200
        transport.resolve_configured_target.assert_awaited_once_with("user:123")
        link = state.sessions.set_mirror_link.call_args.args[1]
        assert link == ChannelLink("telegram", channel_id="123", thread_id=None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body",
        [
            {"channel_type": "telegram", "target_id": "user:123"},
        ],
    )
    async def test_governance_deny_blocks_target_resolution_and_send(
        self, tmp_path, monkeypatch, body
    ):
        monkeypatch.setattr(
            "kiro_crew.platform.governance_profiles.governance_permits",
            lambda *args, **kwargs: SimpleNamespace(permitted=False),
        )
        state = _prep(tmp_path, monkeypatch)
        transport = _fake_transport("telegram")
        state.register_channel_transport(transport)

        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/mirror-link", json=body)
            assert resp.status == 403
            assert (await resp.json())["error"] == "channel is not permitted"

        transport.resolve_configured_target.assert_not_awaited()
        transport.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_governance_narrowing_mid_delivery_fails_closed(self, tmp_path, monkeypatch):
        # Permit the initial link + the announcement, then deny once the
        # historical context-delivery loop starts. The endpoint must fail closed:
        # return 403 and NOT persist the mirror link (regression for a denial
        # that only broke the loop and still persisted + returned 200).
        transport = _fake_transport("telegram")

        def _permits(*args, **kwargs):
            # A PREDICATE, not a call counter. The old version denied on the
            # third governance consult, which silently depended on exactly two
            # consults happening before the loop — change how many messages the
            # backfill selects, or add a pre-loop check, and the denial lands on
            # a pre-loop gate while every assertion below still passes, so the
            # test stops guarding the path it was written for.
            #
            # Keying on "has the transport already delivered?" pins the denial
            # to the first in-loop unit regardless of selection size: the
            # announcement is the only send before the loop.
            return SimpleNamespace(
                permitted=not transport.send_message.await_args_list,
                rule="",
                layer="",
                reason="",
            )

        monkeypatch.setattr("kiro_crew.platform.governance_profiles.governance_permits", _permits)
        state = _prep(tmp_path, monkeypatch)
        state.register_channel_transport(transport)
        state.sessions.set_mirror_link = MagicMock()
        # Give the slot history so the context-delivery loop iterates.
        slot = state.get_or_create_slot("s1")
        slot.messages.extend(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        )

        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/mirror-link",
                json={"channel_type": "telegram", "target_id": "user:123"},
            )
            assert resp.status == 403
            assert (await resp.json())["error"] == "channel is not permitted"

        # Non-vacuity: the announcement is sent only AFTER its own governance
        # check passes, so having sent it proves the denial came later than that
        # check — i.e. inside the context-delivery loop, which is the path under
        # test. Without this, a denial at the very first gate would produce the
        # same 403 and the same unpersisted link.
        assert transport.send_message.await_count >= 1
        state.sessions.set_mirror_link.assert_not_called()
        state = _prep(tmp_path, monkeypatch)
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/nope/mirror-link",
                json={"channel_type": "telegram", "conversation_id": "1"},
            )
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_missing_channel_type(self, tmp_path, monkeypatch):
        state = _prep(tmp_path, monkeypatch)
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/mirror-link", json={})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_slack_rejected(self, tmp_path, monkeypatch):
        state = _prep(tmp_path, monkeypatch)
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/mirror-link",
                json={"channel_type": "slack", "conversation_id": "C1"},
            )
            assert resp.status == 400
            assert "slack-link" in (await resp.json())["error"]

    @pytest.mark.asyncio
    async def test_missing_target_id(self, tmp_path, monkeypatch):
        state = _prep(tmp_path, monkeypatch)
        state.register_channel_transport(_fake_transport("telegram"))
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/mirror-link", json={"channel_type": "telegram"}
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_channel_not_connected(self, tmp_path, monkeypatch):
        state = _prep(tmp_path, monkeypatch)  # no transport registered
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/mirror-link",
                json={"channel_type": "telegram", "target_id": "user:1"},
            )
            assert resp.status == 503

    @pytest.mark.asyncio
    async def test_non_proactive_channel_rejected(self, tmp_path, monkeypatch):
        state = _prep(tmp_path, monkeypatch)
        state.register_channel_transport(_fake_transport("wecom", proactive=False))
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/mirror-link",
                json={"channel_type": "wecom", "target_id": "user:u1"},
            )
            assert resp.status == 400
            assert "proactive" in (await resp.json())["error"]

    @pytest.mark.asyncio
    async def test_link_success(self, tmp_path, monkeypatch):
        state = _prep(tmp_path, monkeypatch)
        state.register_channel_transport(_fake_transport("telegram"))
        state.sessions.set_mirror_link = MagicMock()
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/mirror-link",
                json={"channel_type": "telegram", "target_id": "user:123"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["ok"] is True and data["conversation_id"] == "123"
        state.sessions.set_mirror_link.assert_called_once()
        link = state.sessions.set_mirror_link.call_args.args[1]
        assert link == ChannelLink("telegram", channel_id="123", thread_id=None)

    @pytest.mark.asyncio
    async def test_link_passes_thread_id(self, tmp_path, monkeypatch):
        state = _prep(tmp_path, monkeypatch)
        transport = _fake_transport("telegram")
        # thread_id now flows from the configured-target resolution, not the body.
        transport.resolve_configured_target = AsyncMock(return_value=("C", "T"))
        state.register_channel_transport(transport)
        state.sessions.set_mirror_link = MagicMock()
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/mirror-link",
                json={"channel_type": "telegram", "target_id": "user:C"},
            )
            assert resp.status == 200
        link = state.sessions.set_mirror_link.call_args.args[1]
        assert link.thread_id == "T"


class TestMirrorUnlink:
    @pytest.mark.asyncio
    async def test_slot_not_found(self, tmp_path, monkeypatch):
        state = _prep(tmp_path, monkeypatch)
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post("/api/chat/slots/nope/mirror-unlink")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_unlink_success(self, tmp_path, monkeypatch):
        state = _prep(tmp_path, monkeypatch)
        state.sessions.clear_mirror_link = MagicMock(return_value=True)
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/mirror-unlink")
            assert resp.status == 200
            assert (await resp.json())["was_linked"] is True

    @pytest.mark.asyncio
    async def test_unlink_noop(self, tmp_path, monkeypatch):
        state = _prep(tmp_path, monkeypatch)
        state.sessions.clear_mirror_link = MagicMock(return_value=False)
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/mirror-unlink")
            assert resp.status == 200
            assert (await resp.json())["was_linked"] is False

    @pytest.mark.asyncio
    async def test_unlink_is_scoped_to_the_channel_the_caller_names(
        self, tmp_path, monkeypatch
    ):
        """The header chip names ONE channel; the unnamed clear drops every binding.

        Releasing the Discord the chip labels must not also delete a Telegram
        binding the user never mentioned — that would strip its `accepts_inbound`
        and fork the next Telegram message into a fresh, historyless session.
        """
        state = _prep(tmp_path, monkeypatch)
        state.sessions.clear_mirror_link = MagicMock(return_value=True)
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/mirror-unlink", json={"channel_type": "discord"}
            )
            assert resp.status == 200
        _key, channel_type = state.sessions.clear_mirror_link.call_args[0]
        assert channel_type == "discord"

    @pytest.mark.asyncio
    async def test_unlink_reads_a_chunked_body_rather_than_content_length(
        self, tmp_path, monkeypatch
    ):
        """A chunked POST has `content_length is None`.

        Branching on Content-Length would read no `channel_type` and silently
        widen a scoped release into "clear everything" — the same defect the link
        handler documents.
        """
        state = _prep(tmp_path, monkeypatch)
        state.sessions.clear_mirror_link = MagicMock(return_value=True)

        async def chunked():
            yield b'{"channel_type": "discord"}'

        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/mirror-unlink", data=chunked())
            assert resp.status == 200
        _key, channel_type = state.sessions.clear_mirror_link.call_args[0]
        assert channel_type == "discord"

    @pytest.mark.asyncio
    async def test_unlink_rejects_a_malformed_body_with_400_not_500(
        self, tmp_path, monkeypatch
    ):
        state = _prep(tmp_path, monkeypatch)
        state.sessions.clear_mirror_link = MagicMock(return_value=True)
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/mirror-unlink",
                data=b"{not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
            assert (await resp.json())["code"] == "body_not_json"
        state.sessions.clear_mirror_link.assert_not_called()


class TestTakeoverConsentIsStrict:
    """Consent is a boolean or it is absent.

    A body carrying `{"confirm": "false"}` — or any non-empty string, or 0/1 from a
    sloppy client — read as truthy and evicted another session's binding without
    the user ever seeing the prompt.
    """

    @staticmethod
    def _prepped(tmp_path, monkeypatch):
        monkeypatch.setattr(
            "kiro_crew.platform.governance_profiles.governance_permits",
            lambda *args, **kwargs: SimpleNamespace(permitted=True),
        )
        state = _prep(tmp_path, monkeypatch)
        state.register_channel_transport(_fake_transport("discord"))
        state.sessions.find_mirror_sessions = MagicMock(return_value=["dashboard:other"])
        state.sessions.clear_mirror_links_at = MagicMock(return_value=["dashboard:other"])
        state.sessions.set_mirror_link = MagicMock()
        return state

    @staticmethod
    async def _connect(state, confirm):
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/mirror-link",
                json={
                    "channel_type": "discord", "target_id": "user:123", "confirm": confirm,
                },
            )
            return resp.status, await resp.json()

    @pytest.mark.asyncio
    async def test_the_string_false_does_not_count_as_confirmation(
        self, tmp_path, monkeypatch
    ):
        state = self._prepped(tmp_path, monkeypatch)

        status, body = await self._connect(state, "false")

        assert status == 409
        assert body["code"] == "conversation_occupied"
        state.sessions.set_mirror_link.assert_not_called()
        state.sessions.clear_mirror_links_at.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_truthy_non_boolean_does_not_count_either(self, tmp_path, monkeypatch):
        state = self._prepped(tmp_path, monkeypatch)

        status, _body = await self._connect(state, 1)

        assert status == 409
        state.sessions.set_mirror_link.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_real_boolean_true_still_takes_the_conversation(
        self, tmp_path, monkeypatch
    ):
        """Non-vacuity: the strict check must not break actual consent."""
        state = self._prepped(tmp_path, monkeypatch)

        status, _body = await self._connect(state, True)

        assert status == 200
        state.sessions.set_mirror_link.assert_called_once()
        state.sessions.clear_mirror_links_at.assert_called_once()


class TestTheOccupancyCheckIsRecheckedAtTheCommit:
    """The first check is a stale snapshot by the time the binding is written.

    Three awaited sends sit between them (target resolution, the link notice, the
    catch-up), so two concurrent connects could both pass the first check and both
    persist an inbound binding — after which the resolver refuses to route and the
    conversation reaches NOBODY.
    """

    @staticmethod
    def _prepped(tmp_path, monkeypatch, *, occupants_over_time):
        monkeypatch.setattr(
            "kiro_crew.platform.governance_profiles.governance_permits",
            lambda *args, **kwargs: SimpleNamespace(permitted=True),
        )
        state = _prep(tmp_path, monkeypatch)
        state.register_channel_transport(_fake_transport("discord"))
        # Each call returns the next snapshot: empty at the first check, occupied
        # by the time we commit — a rival connect that landed mid-delivery.
        state.sessions.find_mirror_sessions = MagicMock(side_effect=occupants_over_time)
        state.sessions.clear_mirror_links_at = MagicMock(return_value=["dashboard:rival"])
        state.sessions.set_mirror_link = MagicMock()
        return state

    @staticmethod
    async def _connect(state, **extra):
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/mirror-link",
                json={"channel_type": "discord", "target_id": "user:123", **extra},
            )
            return resp.status, await resp.json()

    @pytest.mark.asyncio
    async def test_a_rival_that_lands_mid_delivery_is_not_silently_evicted(
        self, tmp_path, monkeypatch
    ):
        """The user confirmed nothing about THIS occupant, so it must ask again."""
        state = self._prepped(
            tmp_path, monkeypatch,
            occupants_over_time=[[], ["dashboard:rival"]],
        )

        status, body = await self._connect(state)

        assert status == 409
        assert body["code"] == "conversation_occupied"
        # Neither binding was touched: no duplicate owners, no silent eviction.
        state.sessions.set_mirror_link.assert_not_called()
        state.sessions.clear_mirror_links_at.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_confirmed_takeover_still_evicts_a_late_arrival(
        self, tmp_path, monkeypatch
    ):
        """`confirm` means "take it" — including from whoever holds it at commit."""
        state = self._prepped(
            tmp_path, monkeypatch,
            occupants_over_time=[["dashboard:first"], ["dashboard:rival"]],
        )

        status, _body = await self._connect(state, confirm=True)

        assert status == 200
        state.sessions.clear_mirror_links_at.assert_called_once()
        state.sessions.set_mirror_link.assert_called_once()

    @pytest.mark.asyncio
    async def test_an_unoccupied_conversation_commits_without_a_second_prompt(
        self, tmp_path, monkeypatch
    ):
        """Non-vacuity: the re-check must not add a prompt to the ordinary path."""
        state = self._prepped(tmp_path, monkeypatch, occupants_over_time=[[], []])

        status, _body = await self._connect(state)

        assert status == 200
        state.sessions.set_mirror_link.assert_called_once()
        state.sessions.clear_mirror_links_at.assert_not_called()


class TestOneSessionPerConversation:
    """A conversation hosts exactly one session, and taking it is confirmed.

    A Discord DM cannot hold threads, so there is nothing to scope two bindings
    to: with two, the inbound resolver refuses to pick and a message in that
    conversation reaches NOBODY. Eviction is what keeps "replying reconnects it"
    unambiguous — the same last-writer-wins guarantee Slack gets for free from its
    single-valued thread index.
    """

    @staticmethod
    def _prepped(tmp_path, monkeypatch, *, occupied_by=()):
        monkeypatch.setattr(
            "kiro_crew.platform.governance_profiles.governance_permits",
            lambda *args, **kwargs: SimpleNamespace(permitted=True),
        )
        state = _prep(tmp_path, monkeypatch)
        transport = _fake_transport("discord")
        state.register_channel_transport(transport)
        state.sessions.find_mirror_sessions = MagicMock(return_value=list(occupied_by))
        state.sessions.clear_mirror_links_at = MagicMock(return_value=list(occupied_by))
        state.sessions.set_mirror_link = MagicMock()
        return state, transport

    @staticmethod
    async def _connect(state, **extra):
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/mirror-link",
                json={"channel_type": "discord", "target_id": "user:123", **extra},
            )
            return resp.status, await resp.json()

    @pytest.mark.asyncio
    async def test_connecting_an_occupied_conversation_asks_first(self, tmp_path, monkeypatch):
        state, transport = self._prepped(tmp_path, monkeypatch, occupied_by=["dashboard:other"])

        status, body = await self._connect(state)

        assert status == 409
        assert body["code"] == "conversation_occupied"
        assert body["requires_confirm"] is True
        # Refused BEFORE any side effect: nothing announced, nothing evicted,
        # nothing bound. A confirm the user has not given cannot cost them a
        # session's connection.
        transport.send_message.assert_not_awaited()
        state.sessions.clear_mirror_links_at.assert_not_called()
        state.sessions.set_mirror_link.assert_not_called()

    @pytest.mark.asyncio
    async def test_confirmed_takeover_evicts_and_tells_the_conversation(
        self, tmp_path, monkeypatch
    ):
        state, transport = self._prepped(tmp_path, monkeypatch, occupied_by=["dashboard:other"])

        status, _ = await self._connect(state, confirm=True)

        assert status == 200
        state.sessions.clear_mirror_links_at.assert_called_once()
        # Whoever is reading that conversation is told which session they are
        # talking to now — the eviction is not silent on the channel side.
        sent = [call.args[1] for call in transport.send_message.await_args_list]
        assert any("different session is connected here now" in text for text in sent)

    @pytest.mark.asyncio
    async def test_a_free_conversation_needs_no_confirm(self, tmp_path, monkeypatch):
        state, transport = self._prepped(tmp_path, monkeypatch)

        status, _ = await self._connect(state)

        assert status == 200
        state.sessions.clear_mirror_links_at.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_makes_replies_route_back_to_this_session(
        self, tmp_path, monkeypatch
    ):
        """The reported defect: connect a session, reply in Discord, and the reply
        landed in a brand-new channel-born tab.

        The inbound resolver only counts bindings flagged ``accepts_inbound``, and
        the dashboard's connect never set it — so the resolver found no owner and
        fell through to the conversation's own session key.
        """
        state, _ = self._prepped(tmp_path, monkeypatch)

        status, _ = await self._connect(state)

        assert status == 200
        _, kwargs = state.sessions.set_mirror_link.call_args
        assert kwargs["accepts_inbound"] is True


class TestMirrorReminder:
    @pytest.mark.asyncio
    async def test_existing_live_mirror_is_a_silent_no_op(self, tmp_path, monkeypatch):
        """Connecting an already-connected channel does nothing, and says nothing.

        This used to post "Session linked from dashboard — continuing here." for
        the "Post reminder in <channel>" menu item. That row is gone, so the only
        ways to arrive here are a stale dashboard tab racing a connected one or a
        direct API call — and in both cases a stray message in the conversation
        explains nothing to whoever reads it.
        """
        monkeypatch.setattr(
            "kiro_crew.platform.governance_profiles.governance_permits",
            lambda *args, **kwargs: SimpleNamespace(permitted=True),
        )
        state = _prep(tmp_path, monkeypatch)
        transport = _fake_transport("discord")
        state.register_channel_transport(transport)
        state.sessions.get_mirror_link = MagicMock(
            return_value=ChannelLink("discord", channel_id="356163505868767244")
        )
        state.sessions.is_mirror_paused = MagicMock(return_value=False)

        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/mirror-link")
            assert resp.status == 200
            assert await resp.json() == {
                "ok": True,
                "already_linked": True,
                "channel_type": "discord",
            }

        transport.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_muted_mirror_reconnects_and_catches_the_conversation_up(
        self, tmp_path, monkeypatch
    ):
        """The same empty-body call on a MUTED link is the reconnect.

        It lifts the mute through ``set_mirror_link`` — the rebind is what clears
        the flag — and seeds the conversation with the history it missed, because
        the gap in it is there precisely because delivery was off.
        """
        monkeypatch.setattr(
            "kiro_crew.platform.governance_profiles.governance_permits",
            lambda *args, **kwargs: SimpleNamespace(permitted=True),
        )
        state = _prep(tmp_path, monkeypatch)
        transport = _fake_transport("discord")
        state.register_channel_transport(transport)
        link = ChannelLink("discord", channel_id="356163505868767244")
        state.sessions.get_mirror_link = MagicMock(return_value=link)
        state.sessions.is_mirror_paused = MagicMock(return_value=True)
        state.sessions.set_mirror_link = MagicMock()
        # History for the catch-up to carry. Without it there is nothing to send
        # and the test would pass on an empty delivery.
        slot = state.get_or_create_slot("s1")
        slot.append("user", "what changed while I was away")
        slot.append("assistant", "the lint rule moved")
        slot.drain()

        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/mirror-link")
            assert resp.status == 200
            body = await resp.json()

        assert body["reconnected"] is True
        assert body["conversation_id"] == "356163505868767244"
        # The rebind is the un-mute, and it must land AFTER a successful catch-up.
        # accepts_inbound is re-asserted so a reply in that conversation resumes
        # THIS session rather than starting a channel-born one.
        state.sessions.set_mirror_link.assert_called_once_with(
            "dashboard:s1", link, accepts_inbound=True
        )
        # Catch-up delivered the missed history: this is the difference between a
        # reconnect and the silent no-op above.
        sent = "\n".join(call.args[1] for call in transport.send_message.await_args_list)
        assert "what changed while I was away" in sent
        assert "the lint rule moved" in sent

    @pytest.mark.asyncio
    async def test_partial_body_validates_instead_of_posting(self, tmp_path, monkeypatch):
        """A non-empty partial payload must hit field validation, not send.

        ``{"thread_id": ...}`` carries neither channel_type nor conversation_id,
        so gating reminder mode on those two fields being absent would post an
        unsolicited message to the persisted channel instead of rejecting a
        malformed link attempt.
        """
        monkeypatch.setattr(
            "kiro_crew.platform.governance_profiles.governance_permits",
            lambda *args, **kwargs: SimpleNamespace(permitted=True),
        )
        state = _prep(tmp_path, monkeypatch)
        transport = _fake_transport("discord")
        state.register_channel_transport(transport)
        state.sessions.get_mirror_link = MagicMock(
            return_value=ChannelLink("discord", channel_id="356163505868767244")
        )

        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/mirror-link", json={"thread_id": "unexpected"}
            )
            assert resp.status == 400
            assert (await resp.json())["error"] == "channel_type required"

        transport.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_object_body_is_rejected(self, tmp_path, monkeypatch):
        state = _prep(tmp_path, monkeypatch)
        transport = _fake_transport("discord")
        state.register_channel_transport(transport)

        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/mirror-link", json=["nope"])
            assert resp.status == 400

        transport.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_utf8_body_is_400_not_500(self, tmp_path, monkeypatch):
        """A body that cannot be decoded is a client error, not a traceback."""
        state = _prep(tmp_path, monkeypatch)
        transport = _fake_transport("discord")
        state.register_channel_transport(transport)

        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/mirror-link",
                data=b"\xff\xfe\x00bad",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400

        transport.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_charset_is_400_not_500(self, tmp_path, monkeypatch):
        state = _prep(tmp_path, monkeypatch)
        transport = _fake_transport("discord")
        state.register_channel_transport(transport)

        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/mirror-link",
                data=b'{"channel_type":"discord"}',
                headers={"Content-Type": "application/json; charset=nosuchcharset"},
            )
            assert resp.status == 400

        transport.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_chunked_partial_body_validates_instead_of_posting(self, tmp_path, monkeypatch):
        """A CHUNKED partial payload must not read as an empty body.

        A chunked request has ``content_length is None``, so branching on
        Content-Length to decide whether to read JSON treats a real body as
        empty and falls into reminder mode — posting an unsolicited message.
        """
        monkeypatch.setattr(
            "kiro_crew.platform.governance_profiles.governance_permits",
            lambda *args, **kwargs: SimpleNamespace(permitted=True),
        )
        state = _prep(tmp_path, monkeypatch)
        transport = _fake_transport("discord")
        state.register_channel_transport(transport)
        state.sessions.get_mirror_link = MagicMock(
            return_value=ChannelLink("discord", channel_id="356163505868767244")
        )

        async def _chunked():
            yield b'{"thread_id": "unexpected"}'

        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/mirror-link",
                data=_chunked(),
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
            assert (await resp.json())["error"] == "channel_type required"

        transport.send_message.assert_not_awaited()


class TestMirrorBackfillFidelity:
    """The non-Slack mirror seeds the same turn-aware history, chunked not cut.

    Deliberately asymmetric with the Slack path: this delivery stays INLINE
    because its per-message governance re-check has to be able to fail the
    request closed with 403, which a backgrounded drain could not do after the
    handler had already returned 200 and persisted the link.
    """

    # A LITERAL ceiling, deliberately NOT chat_mirror._MAX_INLINE_BACKFILL_UNITS:
    # asserting against the module constant would move with it, so raising the
    # cap — or deleting it — would still pass. This leaves headroom for a
    # deliberate tuning change while still failing an unbounded loop.
    _BOUND_CEILING = 16

    def _linked(self, tmp_path, monkeypatch, max_message_chars=4096):
        state = _prep(tmp_path, monkeypatch)
        transport = _fake_transport("telegram", max_message_chars=max_message_chars)
        state.register_channel_transport(transport)
        state.sessions.set_mirror_link = MagicMock()
        return state, transport

    async def _link(self, state):
        async with TestClient(TestServer(_make_mirror_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/mirror-link",
                json={"channel_type": "telegram", "target_id": "user:123"},
            )
            assert resp.status == 200

    def _sent(self, transport):
        """Delivered bodies, excluding the link announcement."""
        texts = [call.args[1] for call in transport.send_message.await_args_list]
        return [t for t in texts if "Session linked from dashboard" not in t]

    @pytest.mark.asyncio
    async def test_filter_runs_before_slice(self, tmp_path, monkeypatch):
        state, transport = self._linked(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        for role, content in [
            ("user", "why is the build red"),
            ("assistant", "a lint rule changed"),
            ("tool", "grep ..."),
            ("tool", "cat ..."),
            ("tool", "pytest ..."),
        ]:
            slot.append(role, content)
        slot.drain()

        await self._link(state)
        body = "\n".join(self._sent(transport))
        assert "why is the build red" in body
        assert "a lint rule changed" in body
        assert "grep" not in body and "pytest" not in body

    @pytest.mark.asyncio
    async def test_long_message_is_chunked_not_truncated(self, tmp_path, monkeypatch):
        state, transport = self._linked(tmp_path, monkeypatch, max_message_chars=500)
        slot = state.get_or_create_slot("s1")
        long_answer = "".join(f"[{i:04d}]" for i in range(600))  # 3600 chars
        slot.append("user", "explain")
        slot.append("assistant", long_answer)
        slot.drain()

        await self._link(state)
        sent = self._sent(transport)
        assert all(len(text) <= 500 for text in sent), "a chunk exceeded the transport limit"
        body = "".join(sent)
        for i in (0, 300, 599):
            assert f"[{i:04d}]" in body, f"marker {i} lost — content was truncated"

    @pytest.mark.asyncio
    async def test_first_turn_and_gap_marker(self, tmp_path, monkeypatch):
        state, transport = self._linked(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        for i in range(1, 11):
            slot.append("user", f"question {i}")
            slot.append("assistant", f"answer {i}")
        slot.drain()

        await self._link(state)
        sent = self._sent(transport)
        body = "\n".join(sent)
        assert "question 1" in body
        assert "question 10" in body
        assert "question 3" not in body
        markers = [t for t in sent if "earlier turn" in t]
        assert len(markers) == 1
        # Slack would report 4 skipped (10 turns, 5 recent). The inline path is
        # additionally under a delivery budget, so the oldest recent turn is
        # folded into the marker instead of being sent -- 5, not 4. That fold is
        # the point of the budget: the marker absorbs the overflow.
        assert "5 earlier turns" in markers[0]

    @pytest.mark.asyncio
    async def test_history_that_fits_exactly_is_not_trimmed(self, tmp_path, monkeypatch):
        """No gap marker when there is no gap.

        Six two-message turns is exactly the 12-unit budget. An earlier version
        reserved the marker's slot unconditionally, so the reservation pushed the
        oldest turn out and then spent that slot announcing the omission it had
        itself caused — a false gap on history that fit.
        """
        state, transport = self._linked(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        for i in range(1, 7):
            slot.append("user", f"q{i}")
            slot.append("assistant", f"a{i}")
        slot.drain()

        await self._link(state)
        sent = self._sent(transport)
        body = "\n".join(sent)
        assert not any("earlier turn" in t for t in sent), f"false gap marker: {sent}"
        for i in range(1, 7):
            assert f"q{i}" in body, f"turn {i} was trimmed even though it fit"
        assert len(sent) == 12, f"expected all 12 units, got {len(sent)}"

    @pytest.mark.asyncio
    async def test_inline_delivery_is_bounded(self, tmp_path, monkeypatch):
        """The request cannot grow without limit just because history did.

        This path is inline (its governance re-check must be able to 403), so
        every extra unit is another governance hop plus a send on a channel that
        may accept ~1 msg/s. Long history must not push the request past a
        browser fetch timeout.
        """

        state, transport = self._linked(tmp_path, monkeypatch, max_message_chars=200)
        slot = state.get_or_create_slot("s1")
        for i in range(1, 9):
            slot.append("user", f"question {i}")
            slot.append("assistant", f"answer {i} " + "y" * 900)  # ~5 units each
        slot.drain()

        await self._link(state)
        sent = self._sent(transport)
        assert len(sent) <= self._BOUND_CEILING, (
            f"inline delivery sent {len(sent)} units, over the budget"
        )
        # Priority order: the newest turn is irreducible, then the marker, then
        # the opening turn, then older turns. Here each turn costs ~6 units, so
        # the opening turn cannot be afforded and is folded into the count.
        body = "\n".join(sent)
        assert "question 8" in body, "newest turn was trimmed away"
        assert any("earlier turn" in t for t in sent), "trim happened with no marker"

    @pytest.mark.asyncio
    async def test_delivery_scales_with_the_budget_not_with_history(
        self, tmp_path, monkeypatch
    ):
        """Ten times the history must not mean ten times the request duration."""

        counts = []
        for turn_count in (8, 80):
            state, transport = self._linked(tmp_path, monkeypatch)
            slot = state.get_or_create_slot("s1")
            for i in range(1, turn_count + 1):
                slot.append("user", f"q{i}")
                slot.append("assistant", f"a{i}")
            slot.drain()
            await self._link(state)
            counts.append(len(self._sent(transport)))

        assert all(c <= self._BOUND_CEILING for c in counts), counts
        assert counts[0] == counts[1], (
            f"unit count tracked history length ({counts}) instead of the budget"
        )

    @pytest.mark.asyncio
    async def test_no_slack_mrkdwn_conversion_on_a_non_slack_channel(self, tmp_path, monkeypatch):
        """Telegram is not Slack: markdown must pass through unconverted."""
        state, transport = self._linked(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "doc it")
        slot.append("assistant", "## Heading\n\n**bold** text")
        slot.drain()

        await self._link(state)
        body = "\n".join(self._sent(transport))
        assert "## Heading" in body
        assert "**bold**" in body

    @pytest.mark.asyncio
    async def test_credentials_are_redacted(self, tmp_path, monkeypatch):
        state, transport = self._linked(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        secret = "AKIAIOSFODNN7EXAMPLE"
        slot.append("user", "creds")
        slot.append("assistant", f"key is {secret}")
        slot.drain()

        await self._link(state)
        body = "\n".join(self._sent(transport))
        assert secret not in body

    @pytest.mark.asyncio
    async def test_compaction_rows_are_excluded(self, tmp_path, monkeypatch):
        state, transport = self._linked(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "real question")
        slot.append("assistant", "real answer")
        slot.append("assistant", "context compacted", meta={"kind": "compaction"})
        slot.drain()

        await self._link(state)
        body = "\n".join(self._sent(transport))
        assert "real question" in body and "real answer" in body
        assert "context compacted" not in body

    @pytest.mark.asyncio
    async def test_delivery_stays_inline_so_the_link_persists_after_seeding(
        self, tmp_path, monkeypatch
    ):
        """The 200 must not be returned before the seeding is delivered.

        This is the property that forbids backgrounding this path: the mirror
        link is persisted only after every unit has cleared governance.
        """
        state, transport = self._linked(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "one")
        slot.append("assistant", "two")
        slot.drain()

        order: list[str] = []
        original_send = transport.send_message

        async def _tracked_send(*args, **kwargs):
            order.append("send")
            return await original_send(*args, **kwargs)

        transport.send_message = _tracked_send
        state.sessions.set_mirror_link = MagicMock(side_effect=lambda *a, **k: order.append("persist"))

        await self._link(state)
        assert "persist" in order, "link was never persisted"
        assert order.index("persist") == len(order) - 1, (
            "the link was persisted before delivery finished"
        )
        assert order.count("send") >= 3, "announcement + both messages should have been sent"
