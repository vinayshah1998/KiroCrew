"""Tests for C2: channel-neutral cross-surface reply delivery (dashboard->channel).

Covers the DashboardState transport-registry seam and
``_deliver_cross_surface_reply``: it pushes a completed dashboard reply to a
linked non-Slack proactive channel via ``Transport.send_message``,
capability-gated, and is a silent no-op for Slack (its own streaming mirror),
WeCom (no proactive send), unregistered transports, unlinked sessions, and
empty replies.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from chat_test_helpers import _make_state

from kiro_crew.dashboard.chat_runner import (
    _deliver_cross_surface_reply,
    _deliver_cross_surface_user_message,
)
from kiro_crew.messaging.link import ChannelLink
from kiro_crew.platform import redact_via_context
from kiro_crew.security import (
    redact_credentials,
    redact_exfiltration_urls,
)


def _fake_transport(channel_type: str = "telegram", proactive: bool = True):
    return SimpleNamespace(
        channel_type=channel_type,
        capabilities=SimpleNamespace(supports_proactive_send=proactive, max_message_chars=4096),
        send_message=AsyncMock(return_value="mid-1"),
    )


def _bind(state, *links):
    """Stub BOTH mirror accessors so the double matches the real interface.

    Outbound delivery reads ``get_mirror_links`` (a session can hold several
    bindings); callers that know they mean one still read ``get_mirror_link``,
    which returns None rather than an arbitrary sibling when several exist.
    """
    state.sessions.get_mirror_links = MagicMock(return_value=list(links))
    state.sessions.get_mirror_link = MagicMock(
        return_value=links[0] if len(links) == 1 else None
    )


class TestSeveralChannelsAtOnce:
    """A session can mirror to several channels, and each stands on its own.

    Three independent properties, all of which a single-target implementation
    would have silently broken: delivery fans out, one channel's failure does not
    cost the others their message, and a per-binding mute silences only its own.
    """

    @staticmethod
    def _two(tmp_path, *, discord_paused=False, telegram_paused=False):
        state = _make_state(tmp_path)
        discord = _fake_transport("discord")
        telegram = _fake_transport("telegram")
        state.register_channel_transport(discord)
        state.register_channel_transport(telegram)
        links = [
            ChannelLink("discord", channel_id="D1"),
            ChannelLink("telegram", channel_id="T1"),
        ]
        state.sessions.get_mirror_links = MagicMock(return_value=links)
        state.sessions.is_mirror_paused = MagicMock(
            side_effect=lambda _key, channel_type="": (
                discord_paused if channel_type == "discord" else telegram_paused
            )
        )
        return state, discord, telegram

    @pytest.mark.asyncio
    async def test_the_reply_reaches_every_connected_channel(self, tmp_path):
        state, discord, telegram = self._two(tmp_path)
        await _deliver_cross_surface_reply(state, "dashboard:chat-1", "the answer")
        discord.send_message.assert_awaited_once_with("D1", "the answer", thread_id=None)
        telegram.send_message.assert_awaited_once_with("T1", "the answer", thread_id=None)

    @pytest.mark.asyncio
    async def test_the_user_echo_reaches_every_connected_channel(self, tmp_path):
        state, discord, telegram = self._two(tmp_path)
        await _deliver_cross_surface_user_message(state, "dashboard:chat-1", "my question")
        assert discord.send_message.await_count == 1
        assert telegram.send_message.await_count == 1

    @pytest.mark.asyncio
    async def test_muting_one_channel_leaves_the_other_delivering(self, tmp_path):
        state, discord, telegram = self._two(tmp_path, discord_paused=True)
        await _deliver_cross_surface_reply(state, "dashboard:chat-1", "the answer")
        discord.send_message.assert_not_awaited()
        telegram.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_one_channel_failing_does_not_cost_the_other_its_message(self, tmp_path):
        state, discord, telegram = self._two(tmp_path)
        discord.send_message = AsyncMock(side_effect=RuntimeError("discord down"))
        await _deliver_cross_surface_reply(state, "dashboard:chat-1", "the answer")
        telegram.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_each_channel_is_split_at_its_own_length_limit(self, tmp_path):
        """One shared split would cut every channel at the strictest limit."""
        state = _make_state(tmp_path)
        roomy = _fake_transport("discord")
        roomy.capabilities.max_message_chars = 4000
        tight = _fake_transport("telegram")
        tight.capabilities.max_message_chars = 100
        state.register_channel_transport(roomy)
        state.register_channel_transport(tight)
        state.sessions.get_mirror_links = MagicMock(
            return_value=[
                ChannelLink("discord", channel_id="D1"),
                ChannelLink("telegram", channel_id="T1"),
            ]
        )
        state.sessions.is_mirror_paused = MagicMock(return_value=False)

        await _deliver_cross_surface_reply(state, "dashboard:chat-1", "x" * 350)

        assert roomy.send_message.await_count == 1
        assert tight.send_message.await_count > 1


class TestGovernanceDegradationFailsClosed:
    """A degraded governance evaluation must DENY the mirror egress, not permit it.

    ``governance_permits`` catches its own internal errors and, by default,
    returns a permissive "no opinion" Decision — its own docstring notes that a
    caller wrapping it in ``except`` can never observe the failure, so the DENY
    has to be produced at the call site via ``fail_closed=True``. Without that,
    a governance outage silently becomes permission to send to an external
    channel. These tests pin both halves of the gate.
    """

    @pytest.mark.asyncio
    async def test_degraded_evaluation_blocks_delivery(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "kiro_crew.platform.governance_profiles.resolve_active_scope",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("profile store down")),
        )
        state = _make_state(tmp_path)
        tp = _fake_transport("telegram")
        state.register_channel_transport(tp)
        _bind(state, ChannelLink("telegram", channel_id="123", thread_id=None))

        await _deliver_cross_surface_reply(state, "dashboard:chat-1", "hi there")

        tp.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_decision_without_permitted_attr_blocks_delivery(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "kiro_crew.platform.governance_profiles.governance_permits",
            lambda *a, **k: SimpleNamespace(),
        )
        state = _make_state(tmp_path)
        tp = _fake_transport("telegram")
        state.register_channel_transport(tp)
        _bind(state, ChannelLink("telegram", channel_id="123", thread_id=None))

        await _deliver_cross_surface_reply(state, "dashboard:chat-1", "hi there")

        tp.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_composition_error_propagates(self, tmp_path, monkeypatch):
        """A broken governance ceiling must NOT read as an ordinary skip.

        ``governance_permits`` deliberately re-raises PlatformCompositionError
        instead of degrading, so the resolver's generic fail-closed handler must
        let it through rather than swallowing it into a silent no-mirror.
        """
        from kiro_crew.platform.context import PlatformCompositionError

        def _boom(*a, **k):
            raise PlatformCompositionError("ceiling weakened")

        monkeypatch.setattr("kiro_crew.platform.governance_profiles.governance_permits", _boom)
        state = _make_state(tmp_path)
        tp = _fake_transport("telegram")
        state.register_channel_transport(tp)
        _bind(state, ChannelLink("telegram", channel_id="123", thread_id=None))

        with pytest.raises(PlatformCompositionError):
            await _deliver_cross_surface_reply(state, "dashboard:chat-1", "hi there")

        tp.send_message.assert_not_awaited()


class TestRegistrySeam:
    def test_register_and_get(self, tmp_path):
        state = _make_state(tmp_path)
        tp = _fake_transport("telegram")
        state.register_channel_transport(tp)
        assert state.get_channel_transport("telegram") is tp

    def test_register_attaches_dashboard_state_to_the_dispatcher(self, tmp_path):
        state = _make_state(tmp_path)
        dispatcher = SimpleNamespace()
        tp = _fake_transport("telegram")
        tp.dispatcher = dispatcher

        state.register_channel_transport(tp)

        assert dispatcher.dashboard_state is state

    def test_get_missing_returns_none(self, tmp_path):
        state = _make_state(tmp_path)
        assert state.get_channel_transport("telegram") is None

    def test_register_ignores_blank_channel_type(self, tmp_path):
        state = _make_state(tmp_path)
        state.register_channel_transport(SimpleNamespace(channel_type=""))
        assert state.channel_transports == {}


class TestDeliverCrossSurfaceReply:
    @pytest.mark.asyncio
    async def test_delivers_to_telegram(self, tmp_path):
        state = _make_state(tmp_path)
        tp = _fake_transport("telegram")
        state.register_channel_transport(tp)
        _bind(state, ChannelLink("telegram", channel_id="123", thread_id=None))
        await _deliver_cross_surface_reply(state, "dashboard:chat-1", "hi there")
        tp.send_message.assert_awaited_once_with("123", "hi there", thread_id=None)

    @pytest.mark.asyncio
    async def test_passes_thread_id(self, tmp_path):
        state = _make_state(tmp_path)
        tp = _fake_transport("telegram")
        state.register_channel_transport(tp)
        _bind(state, ChannelLink("telegram", channel_id="C", thread_id="T"))
        await _deliver_cross_surface_reply(state, "k", "x")
        tp.send_message.assert_awaited_once_with("C", "x", thread_id="T")

    @pytest.mark.asyncio
    async def test_skips_slack(self, tmp_path):
        state = _make_state(tmp_path)
        tp = _fake_transport("slack")
        state.register_channel_transport(tp)
        _bind(state, ChannelLink("slack", channel_id="C1", thread_id="ts"))
        await _deliver_cross_surface_reply(state, "k", "hi")
        tp.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_no_link(self, tmp_path):
        state = _make_state(tmp_path)
        _bind(state)
        await _deliver_cross_surface_reply(state, "k", "hi")  # must not raise

    @pytest.mark.asyncio
    async def test_skips_when_transport_unregistered(self, tmp_path):
        state = _make_state(tmp_path)
        _bind(state, ChannelLink("telegram", channel_id="123"))
        await _deliver_cross_surface_reply(state, "k", "hi")  # telegram not registered

    @pytest.mark.asyncio
    async def test_skips_when_not_proactive(self, tmp_path):
        state = _make_state(tmp_path)
        tp = _fake_transport("wecom", proactive=False)
        state.register_channel_transport(tp)
        _bind(state, ChannelLink("wecom", channel_id="u1"))
        await _deliver_cross_surface_reply(state, "k", "hi")
        tp.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_empty_text(self, tmp_path):
        state = _make_state(tmp_path)
        tp = _fake_transport("telegram")
        state.register_channel_transport(tp)
        _bind(state, ChannelLink("telegram", channel_id="123"))
        await _deliver_cross_surface_reply(state, "k", "")
        tp.send_message.assert_not_awaited()
        state.sessions.get_mirror_link.assert_not_called()

    @pytest.mark.asyncio
    async def test_applies_redaction_pipeline(self, tmp_path):
        state = _make_state(tmp_path)
        tp = _fake_transport("telegram")
        state.register_channel_transport(tp)
        _bind(state, ChannelLink("telegram", channel_id="123"))
        raw = "see https://evil.example/exfil?q=1 and AKIAIOSFODNN7EXAMPLE"
        expected = redact_credentials(redact_exfiltration_urls(raw)[0])[0]
        await _deliver_cross_surface_reply(state, "k", raw)
        assert tp.send_message.await_args.args[1] == expected

    @pytest.mark.asyncio
    async def test_send_failure_is_swallowed(self, tmp_path):
        state = _make_state(tmp_path)
        tp = _fake_transport("telegram")
        tp.send_message = AsyncMock(side_effect=RuntimeError("boom"))
        state.register_channel_transport(tp)
        _bind(state, ChannelLink("telegram", channel_id="123"))
        await _deliver_cross_surface_reply(state, "k", "hi")  # must not raise

    @pytest.mark.asyncio
    async def test_long_reply_is_chunked(self, tmp_path):
        state = _make_state(tmp_path)
        tp = _fake_transport("telegram")
        tp.capabilities.max_message_chars = 100
        state.register_channel_transport(tp)
        _bind(state, ChannelLink("telegram", channel_id="123"))
        long_text = "x" * 250
        await _deliver_cross_surface_reply(state, "k", long_text)
        # 250 chars / 100 per chunk = 3 sends; content preserved end-to-end and
        # each part stays within the channel's max_message_chars.
        assert tp.send_message.await_count == 3
        sent = "".join(c.args[1] for c in tp.send_message.await_args_list)
        assert sent == long_text
        for c in tp.send_message.await_args_list:
            assert len(c.args[1]) <= 100


class TestDeliverCrossSurfaceUserMessage:
    @pytest.mark.asyncio
    async def test_delivers_with_prefix(self, tmp_path):
        state = _make_state(tmp_path)
        tp = _fake_transport("telegram")
        state.register_channel_transport(tp)
        _bind(state, ChannelLink("telegram", channel_id="123", thread_id=None))
        await _deliver_cross_surface_user_message(state, "k", "hello there")
        tp.send_message.assert_awaited_once_with("123", "💬 hello there", thread_id=None)

    @pytest.mark.asyncio
    async def test_passes_thread_id(self, tmp_path):
        state = _make_state(tmp_path)
        tp = _fake_transport("telegram")
        state.register_channel_transport(tp)
        _bind(state, ChannelLink("telegram", channel_id="C", thread_id="T"))
        await _deliver_cross_surface_user_message(state, "k", "x")
        tp.send_message.assert_awaited_once_with("C", "💬 x", thread_id="T")

    @pytest.mark.asyncio
    async def test_skips_slack(self, tmp_path):
        state = _make_state(tmp_path)
        tp = _fake_transport("slack")
        state.register_channel_transport(tp)
        _bind(state, ChannelLink("slack", channel_id="C1", thread_id="ts"))
        await _deliver_cross_surface_user_message(state, "k", "hi")
        tp.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_no_link(self, tmp_path):
        state = _make_state(tmp_path)
        _bind(state)
        await _deliver_cross_surface_user_message(state, "k", "hi")  # must not raise

    @pytest.mark.asyncio
    async def test_skips_when_transport_unregistered(self, tmp_path):
        state = _make_state(tmp_path)
        _bind(state, ChannelLink("telegram", channel_id="123"))
        await _deliver_cross_surface_user_message(state, "k", "hi")

    @pytest.mark.asyncio
    async def test_skips_when_not_proactive(self, tmp_path):
        state = _make_state(tmp_path)
        tp = _fake_transport("wecom", proactive=False)
        state.register_channel_transport(tp)
        _bind(state, ChannelLink("wecom", channel_id="u1"))
        await _deliver_cross_surface_user_message(state, "k", "hi")
        tp.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_empty_message(self, tmp_path):
        state = _make_state(tmp_path)
        tp = _fake_transport("telegram")
        state.register_channel_transport(tp)
        _bind(state, ChannelLink("telegram", channel_id="123"))
        await _deliver_cross_surface_user_message(state, "k", "")
        tp.send_message.assert_not_awaited()
        state.sessions.get_mirror_link.assert_not_called()

    @pytest.mark.asyncio
    async def test_truncates_and_redacts(self, tmp_path):
        state = _make_state(tmp_path)
        tp = _fake_transport("telegram")
        state.register_channel_transport(tp)
        _bind(state, ChannelLink("telegram", channel_id="123"))
        raw = "tok AKIAIOSFODNN7EXAMPLE " + "x" * 800
        await _deliver_cross_surface_user_message(state, "k", raw)
        sent = tp.send_message.await_args.args[1]
        # _prepare_mirror_msg truncates to 500 THEN redacts (redact_via_context),
        # matching the Slack echo. Distinct from security.redact_and_truncate,
        # which redacts-then-truncates (security-review e27617c6) — the mirror echo keeps
        # the truncate-first order so the 500-char budget is measured pre-redaction.
        assert sent == "💬 " + redact_via_context(raw[:500])

    @pytest.mark.asyncio
    async def test_send_failure_is_swallowed(self, tmp_path):
        state = _make_state(tmp_path)
        tp = _fake_transport("telegram")
        tp.send_message = AsyncMock(side_effect=RuntimeError("boom"))
        state.register_channel_transport(tp)
        _bind(state, ChannelLink("telegram", channel_id="123"))
        await _deliver_cross_surface_user_message(state, "k", "hi")  # must not raise
