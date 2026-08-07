"""Pause / reconnect semantics for a Slack-linked session.

Pause is not unlink. The thread<->session binding, both coordinate fields and the
``_thread_to_session`` reverse index all survive a pause, so inbound routing is
untouched and a reply resumes the SAME session. Only outbound turn mirroring
stops.

Both halves are pinned here because either one alone is a shipped bug: mirroring
that does not stop is a pause that does nothing, and a binding that does not
survive is an unlink wearing a pause label -- the reply then mints a fresh
``slack:<ts>`` session and steals the thread, which is the fork this feature
exists to prevent.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_state
from test_slack_mirror_unlink import (
    _fake_provider,
    _make_slack_client,
)
from test_slack_mirror_unlink import _make_state as _make_runner_state

from kiro_crew.messaging.link import canonical_key
from kiro_crew.session_map import SessionMap

SLACK_TS = "1785370133.085469"
SLACK_KEY = f"slack:{SLACK_TS}"
SLACK_STEM = f"slack_{SLACK_TS}"


@pytest.fixture
def session_map(tmp_path):
    with patch("kiro_crew.session_map.config_dir", return_value=tmp_path):
        return SessionMap()


def _pause_app(state):
    """Just the pause + link + unlink routes, on the {slot} spelling."""
    from kiro_crew.dashboard.chat_slack import (
        api_chat_slot_slack_link,
        api_chat_slot_slack_pause,
        api_chat_slot_slack_unlink,
    )

    app = web.Application()
    app["state"] = state
    app.router.add_post("/api/chat/slots/{slot}/slack-link", api_chat_slot_slack_link)
    app.router.add_post("/api/chat/slots/{slot}/slack-unlink", api_chat_slot_slack_unlink)
    app.router.add_post("/api/chat/slots/{slot}/slack-pause", api_chat_slot_slack_pause)
    return app


class TestSessionMapPause:
    """Storage: a presence flag that survives nothing it should not."""

    def test_absent_reads_as_active(self, session_map):
        session_map.set("dash:1", "sid-abc")
        assert session_map.is_slack_paused("dash:1") is False

    def test_pause_then_read(self, session_map):
        session_map.set("dash:1", "sid-abc")
        session_map.set_slack_link("dash:1", SLACK_TS, "C-1")
        assert session_map.set_slack_paused("dash:1", True) is False
        assert session_map.is_slack_paused("dash:1") is True

    def test_resume_removes_the_key_rather_than_storing_false(self, session_map):
        """A resumed session must leave nothing behind, per the house presence-flag style."""
        session_map.set("dash:1", "sid-abc")
        session_map.set_slack_link("dash:1", SLACK_TS, "C-1")
        session_map.set_slack_paused("dash:1", True)
        assert session_map.set_slack_paused("dash:1", False) is True
        assert "slack_paused" not in session_map._data["dash:1"]

    def test_pause_is_idempotent_and_reports_the_prior_state(self, session_map):
        session_map.set("dash:1", "sid-abc")
        session_map.set_slack_link("dash:1", SLACK_TS, "C-1")
        assert session_map.set_slack_paused("dash:1", True) is False
        assert session_map.set_slack_paused("dash:1", True) is True

    def test_pause_retains_the_link_and_the_reverse_index(self, session_map):
        """The whole point: inbound routing must be untouched.

        ``transport_dispatch._resolve_thread_owner`` reroutes a reply on exactly
        this lookup, and its unclaimed-thread guard only self-claims a thread when
        the lookup comes back empty. If pause evicted the index, a reply would
        fork into a new ``slack:<ts>`` session and claim the thread permanently.
        """
        session_map.set("dashboard:chat-1", "sid-abc")
        session_map.set_slack_link("dashboard:chat-1", SLACK_TS, "C-1")
        session_map.set_slack_paused("dashboard:chat-1", True)

        assert session_map.get_slack_link("dashboard:chat-1") == (SLACK_TS, "C-1")
        assert session_map.get_session_for_thread(SLACK_TS) == "dashboard:chat-1"

    def test_pause_covers_both_key_spellings(self, session_map):
        """chat_runner copies a link bare -> "dashboard:"-prefixed when a turn runs.

        A flag written to one spelling and read from the other would silently
        resume a muted thread on the next turn.
        """
        session_map.set("dashboard:chat-1", "sid-a")
        session_map.set("chat-1", "sid-b")
        session_map.set_slack_link("dashboard:chat-1", SLACK_TS, "C-1")
        session_map.set_slack_link("chat-1", SLACK_TS, "C-1")

        session_map.set_slack_paused("dashboard:chat-1", True)

        assert session_map.is_slack_paused("dashboard:chat-1") is True
        assert session_map.is_slack_paused("chat-1") is True, (
            "the bare twin stayed live, so the next turn re-reads it and resumes"
        )

    def test_channel_key_has_no_twin(self, session_map):
        session_map.set(SLACK_KEY, "sid-abc")
        session_map.set_slack_link(SLACK_KEY, SLACK_TS, "C-1")
        session_map.set_slack_paused(SLACK_KEY, True)
        assert session_map.is_slack_paused(SLACK_KEY) is True
        assert session_map.is_slack_paused(canonical_key(SLACK_TS)) is True

    def test_unlink_pops_the_pause_flag(self, session_map):
        """A pause must not outlive the link it describes."""
        session_map.set("dash:1", "sid-abc")
        session_map.set_slack_link("dash:1", SLACK_TS, "C-1")
        session_map.set_slack_paused("dash:1", True)

        assert session_map.clear_slack_link("dash:1") is True

        assert "slack_paused" not in session_map._data["dash:1"]
        assert session_map.is_slack_paused("dash:1") is False

    def test_relink_after_unlink_is_not_silently_paused(self, session_map):
        """The regression the pop above prevents."""
        session_map.set("dash:1", "sid-abc")
        session_map.set_slack_link("dash:1", SLACK_TS, "C-1")
        session_map.set_slack_paused("dash:1", True)
        session_map.clear_slack_link("dash:1")

        session_map.set_slack_link("dash:1", "1785999999.000001", "C-2")

        assert session_map.is_slack_paused("dash:1") is False

    def test_pause_survives_a_reload_from_disk(self, session_map, tmp_path):
        session_map.set("dash:1", "sid-abc")
        session_map.set_slack_link("dash:1", SLACK_TS, "C-1")
        session_map.set_slack_paused("dash:1", True)

        with patch("kiro_crew.session_map.config_dir", return_value=tmp_path):
            reloaded = SessionMap()

        assert reloaded.is_slack_paused("dash:1") is True
        assert reloaded.get_session_for_thread(SLACK_TS) == "dash:1"


class TestPauseStopsOutboundMirroring:
    """Turn level, driven through the real mirror gate with a real SessionMap."""

    @pytest.mark.asyncio
    async def test_linked_turn_mirrors_then_paused_turn_does_not(self, tmp_path, monkeypatch):
        from kiro_crew.dashboard.chat import _history_key_for, _run_chat

        monkeypatch.setattr("kiro_crew.dashboard.chat.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.dashboard.chat.sel", lambda: MagicMock())

        with patch("kiro_crew.session_map.config_dir", return_value=tmp_path):
            smap = SessionMap()

        state = _make_runner_state(tmp_path, smap)
        # The pause accessors are what the gate consults; route them at the real map.
        state.sessions.set_slack_paused = smap.set_slack_paused
        state.sessions.is_slack_paused = smap.is_slack_paused
        state.slack_client = _make_slack_client()
        state.sessions.get_or_create = AsyncMock(
            return_value=(_fake_provider(), False, False)
        )

        slot = state.get_or_create_slot("s1")
        session_key = _history_key_for(slot.key)
        state.sessions.set_slack_link(session_key, "thread-1", "C-1")
        slot._slack_linked = True
        slot._slack_channel = "C-1"
        slot._slack_thread_ts = "thread-1"

        # Live turn: mirrors (status quo, and proves the fixture reaches the gate).
        await _run_chat(state, slot, "first message")
        assert state.slack_client.post_message.await_count >= 1
        assert state.slack_client.start_stream.await_count == 1

        state.sessions.set_slack_paused(session_key, True)
        state.slack_client.post_message.reset_mock()
        state.slack_client.start_stream.reset_mock()

        await _run_chat(state, slot, "second message")
        assert state.slack_client.post_message.await_count == 0, (
            "a paused thread still received the user-message echo"
        )
        assert state.slack_client.start_stream.await_count == 0, (
            "a paused thread still opened a tool stream"
        )

    @pytest.mark.asyncio
    async def test_pause_does_not_evict_the_slot_from_the_thread_index(
        self, tmp_path, monkeypatch
    ):
        """``get_linked_slot`` drops the index entry when ``_slack_linked`` is false.

        Expressing pause by clearing that boolean would make the slot self-purge,
        so a resumed turn would never render in the open dashboard tab.
        """
        from kiro_crew.dashboard.chat import _history_key_for

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)

        with patch("kiro_crew.session_map.config_dir", return_value=tmp_path):
            smap = SessionMap()

        state = _make_state(tmp_path)
        state.sessions.set_slack_paused = smap.set_slack_paused
        state.sessions.is_slack_paused = smap.is_slack_paused
        state.sessions.get_slack_link = smap.get_slack_link
        state.sessions.set_slack_link = smap.set_slack_link
        state.push_slots_update = MagicMock()

        slot = state.get_or_create_slot("s1")
        session_key = _history_key_for(slot.key)
        smap.set(session_key, "sid-abc")
        state.link_slack(slot.key, "thread-1", "C-1")

        smap.set_slack_paused(session_key, True)

        assert state.get_linked_slot("thread-1") is slot
        assert slot._slack_linked is True
        assert slot._slack_channel == "C-1"
        assert slot._slack_thread_ts == "thread-1"


class TestReplyResumes:
    @pytest.mark.asyncio
    async def test_resume_helper_lifts_the_flag(self, session_map):
        from kiro_crew.slack.handler import _resume_if_paused

        session_map.set("dashboard:chat-1", "sid-abc")
        session_map.set_slack_link("dashboard:chat-1", SLACK_TS, "C-1")
        session_map.set_slack_paused("dashboard:chat-1", True)

        await _resume_if_paused(session_map, "dashboard:chat-1", SLACK_TS)

        assert session_map.is_slack_paused("dashboard:chat-1") is False

    @pytest.mark.asyncio
    async def test_resume_is_a_noop_on_a_live_link(self, session_map):
        """Guarded on the read so the persisting write runs only on the transition."""
        from kiro_crew.slack.handler import _resume_if_paused

        session_map.set("dashboard:chat-1", "sid-abc")
        session_map.set_slack_link("dashboard:chat-1", SLACK_TS, "C-1")
        session_map.set_slack_paused = MagicMock()  # type: ignore[method-assign]

        await _resume_if_paused(session_map, "dashboard:chat-1", SLACK_TS)

        session_map.set_slack_paused.assert_not_called()

    @pytest.mark.asyncio
    async def test_resume_swallows_a_session_manager_without_the_accessor(self, session_map):
        from kiro_crew.slack.handler import _resume_if_paused

        await _resume_if_paused(object(), "dashboard:chat-1", SLACK_TS)

    def test_both_inbound_paths_call_the_resume_helper(self):
        """The transport path is live by default; the native path is the fallback.

        Wiring only one leaves paused threads permanently muted under the other,
        so pin that both reference the helper.
        """
        import inspect

        from kiro_crew.slack import handler, transport_dispatch

        assert "_resume_if_paused" in inspect.getsource(
            transport_dispatch.handle_message_transport
        )
        assert "_resume_if_paused" in inspect.getsource(handler.handle_message)


class TestPauseEndpoint:
    @pytest.mark.asyncio
    async def test_pause_posts_a_courtesy_note_once(self, tmp_path, monkeypatch):
        """The note goes INTO the Slack thread, and only on the transition.

        It is the only thing marking why the thread stopped, so a watcher can tell
        a disconnected conversation from a stalled one. It states the fact and
        nothing else: no mention of pausing (a verb no surface uses) and no
        coaching to reply, since resuming that way is treated as a given.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.dashboard.chat_slack.sel", lambda: MagicMock())
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        state.slack_client = _make_slack_client()
        state.sessions.get_slack_link = MagicMock(return_value=("ts123", "C123"))
        state.sessions.set_slack_paused = MagicMock(side_effect=[False, True])
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(_pause_app(state))) as client:
            first = await client.post("/api/chat/slots/s1/slack-pause")
            assert first.status == 200
            assert (await first.json()) == {"ok": True, "was_paused": False}

            second = await client.post("/api/chat/slots/s1/slack-pause")
            assert second.status == 200
            assert (await second.json()) == {"ok": True, "was_paused": True}

        # Exactly one note across two calls: idempotent re-disconnect stays silent.
        state.slack_client.post_message.assert_awaited_once()
        args = state.slack_client.post_message.await_args.args
        assert args[0] == "C123"
        assert args[2] == "ts123"
        assert "Disconnected" in args[1]
        assert "paus" not in args[1].lower()
        assert "repl" not in args[1].lower()
        assert slot is state.get_or_create_slot("s1")

    @pytest.mark.asyncio
    async def test_pause_on_an_unlinked_session_is_a_conflict(self, tmp_path, monkeypatch):
        """Nothing to mute, and returning ok would leave the UI showing Reconnect."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.dashboard.chat_slack.sel", lambda: MagicMock())
        state = _make_state(tmp_path)
        state.get_or_create_slot("s1")
        state.slack_client = _make_slack_client()
        state.sessions.get_slack_link = MagicMock(return_value=(None, None))
        state.sessions.set_slack_paused = MagicMock()
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(_pause_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/slack-pause")
            assert resp.status == 409

        state.sessions.set_slack_paused.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_slot_is_404(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        async with TestClient(TestServer(_pause_app(state))) as client:
            resp = await client.post("/api/chat/slots/nope/slack-pause")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_channel_born_slot_pauses_its_own_session(self, tmp_path, monkeypatch):
        """The key must come from the slot, not its name.

        ``_history_key_for`` would build ``dashboard:slack:<ts>`` here -- a session
        that does not exist -- so the pause would land nowhere and the real thread
        would keep mirroring.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.dashboard.chat_slack.sel", lambda: MagicMock())
        state = _make_state(tmp_path)
        state.slack_client = _make_slack_client()
        state.sessions.get_slack_link = MagicMock(return_value=(SLACK_TS, "C777"))
        state.sessions.channel_key_for_stem = lambda stem: (
            SLACK_KEY if stem == SLACK_STEM else ""
        )
        state.sessions.set_slack_paused = MagicMock(return_value=False)
        state.push_slots_update = MagicMock()
        state.get_or_create_slot(SLACK_STEM, linked_session_key=SLACK_KEY)

        async with TestClient(TestServer(_pause_app(state))) as client:
            resp = await client.post(f"/api/chat/slots/{SLACK_STEM}/slack-pause")
            assert resp.status == 200

        state.sessions.set_slack_paused.assert_called_once_with(SLACK_KEY, True)


class TestReconnect:
    @pytest.mark.asyncio
    async def test_reconnect_reseeds_the_existing_thread(self, tmp_path, monkeypatch):
        """No new thread, pause lifted, backfill fired at the thread we already had."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.dashboard.chat_slack.sel", lambda: MagicMock())
        spawned: list[tuple] = []
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_slack._spawn_slack_backfill",
            lambda state, slot, channel, thread_ts: spawned.append((channel, thread_ts)),
        )
        state = _make_state(tmp_path)
        state.owner_id = "U1"
        state.slack_client = _make_slack_client()
        state.sessions.get_slack_link = MagicMock(return_value=("ts123", "C123"))
        state.sessions.is_slack_paused = MagicMock(return_value=True)
        state.sessions.set_slack_paused = MagicMock(return_value=True)
        state.sessions.set_slack_link = MagicMock()
        state.push_slots_update = MagicMock()
        state.get_or_create_slot("s1")

        async with TestClient(TestServer(_pause_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/slack-link")
            assert resp.status == 200
            body = await resp.json()

        assert body["reconnected"] is True
        assert (body["channel"], body["thread_ts"]) == ("C123", "ts123")
        state.sessions.set_slack_paused.assert_called_once_with("dashboard:s1", False)
        assert spawned == [("C123", "ts123")], "reconnect must re-seed the SAME thread"
        # No anchor message: a new thread was never created.
        state.slack_client.post_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_link_on_a_live_link_is_a_silent_no_op(self, tmp_path, monkeypatch):
        """Not disconnected -> nothing happens, and nothing is said in the thread.

        The note this used to post existed for the "Post reminder in Slack" menu
        item, which called this endpoint on a live link just to ping the thread.
        That item is gone, so the note would only ever appear from a stale tab or a
        direct API call — a stray message that explains nothing to whoever reads it.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.dashboard.chat_slack.sel", lambda: MagicMock())
        spawned: list[tuple] = []
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_slack._spawn_slack_backfill",
            lambda state, slot, channel, thread_ts: spawned.append((channel, thread_ts)),
        )
        state = _make_state(tmp_path)
        state.owner_id = "U1"
        state.slack_client = _make_slack_client()
        state.sessions.get_slack_link = MagicMock(return_value=("ts123", "C123"))
        state.sessions.is_slack_paused = MagicMock(return_value=False)
        state.push_slots_update = MagicMock()
        state.get_or_create_slot("s1")

        async with TestClient(TestServer(_pause_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/slack-link")
            body = await resp.json()

        assert body["already_linked"] is True
        assert "reconnected" not in body
        assert spawned == []
        state.slack_client.post_message.assert_not_awaited()


class TestSerialization:
    def test_channel_born_slot_carries_a_paused_origin_link(self, tmp_path, monkeypatch):
        """The defect this fixes: no Slack row at all, so the UI offered "Send to Slack"."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.sessions.get_slack_link = MagicMock(return_value=(SLACK_TS, "C777"))
        state.sessions.get_mirror_link = MagicMock(return_value=None)
        state.sessions.mirror_accepts_inbound = MagicMock(return_value=False)
        state.sessions.is_slack_paused = MagicMock(return_value=True)
        state.sessions.channel_key_for_stem = lambda stem: (
            SLACK_KEY if stem == SLACK_STEM else ""
        )
        slot = state.get_or_create_slot(SLACK_STEM, linked_session_key=SLACK_KEY)

        payload = state.serialize_slot(slot)

        slack_rows = [x for x in payload["links"] if x["channel"] == "slack"]
        assert len(slack_rows) == 1
        assert slack_rows[0]["direction"] == "origin"
        assert slack_rows[0]["paused"] is True
        # Still false: the frontend rebuilds a phantom mirror row from this flag
        # whenever no Slack wire link is present, and a real row plus a true flag
        # would badge the same conversation twice.
        assert payload["slack_linked"] is False

    def test_a_live_channel_born_slot_reports_not_paused(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.sessions.get_slack_link = MagicMock(return_value=(SLACK_TS, "C777"))
        state.sessions.get_mirror_link = MagicMock(return_value=None)
        state.sessions.mirror_accepts_inbound = MagicMock(return_value=False)
        state.sessions.is_slack_paused = MagicMock(return_value=False)
        state.sessions.channel_key_for_stem = lambda stem: (
            SLACK_KEY if stem == SLACK_STEM else ""
        )
        slot = state.get_or_create_slot(SLACK_STEM, linked_session_key=SLACK_KEY)

        slack_rows = [
            x for x in state.serialize_slot(slot)["links"] if x["channel"] == "slack"
        ]
        assert [x["paused"] for x in slack_rows] == [False]

    def test_a_stubbed_session_manager_never_reads_as_paused(self, tmp_path, monkeypatch):
        """``sessions`` is a bare MagicMock across much of the suite.

        Truthiness on an unstubbed accessor would report every link paused and
        silence mirroring everywhere, so the gate demands identity with ``True``.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.sessions.get_slack_link = MagicMock(return_value=(SLACK_TS, "C777"))
        state.sessions.get_mirror_link = MagicMock(return_value=None)
        state.sessions.mirror_accepts_inbound = MagicMock(return_value=False)
        state.sessions.channel_key_for_stem = lambda stem: (
            SLACK_KEY if stem == SLACK_STEM else ""
        )
        slot = state.get_or_create_slot(SLACK_STEM, linked_session_key=SLACK_KEY)

        slack_rows = [
            x for x in state.serialize_slot(slot)["links"] if x["channel"] == "slack"
        ]
        assert [x["paused"] for x in slack_rows] == [False]


class TestResumeNeverPrecedesTheGovernanceGate:
    """A denied inbound message must not leave the channel connected.

    Lifting the pause PERSISTS, so doing it during owner resolution — before
    `channel_inbound_permitted` has had its say — means a message governance goes
    on to drop has already reconnected the link. The deny is silent and the link is
    live, which is the opposite of what a denial means.

    Asserted STRUCTURALLY, on source call order, rather than by driving the whole
    dispatcher: the defect IS the ordering, and an end-to-end harness for this path
    needs the full Slack event envelope, a conversation log and a governance policy
    — at which point the test asserts against its own stubs instead of the code.
    """

    @staticmethod
    def _source(module) -> str:
        import inspect

        return inspect.getsource(module)

    def test_the_transport_dispatch_resumes_after_the_gate(self):
        from kiro_crew.slack import transport_dispatch

        src = self._source(transport_dispatch)
        gate = src.find('channel_inbound_permitted("slack")')
        # The CALL, not the import line, which also contains the bare name.
        resume = src.find("await _resume_if_paused(")
        assert gate != -1, "governance gate not found — did it move or get renamed?"
        assert resume != -1, "resume call not found — did it move or get renamed?"
        assert resume > gate, (
            "the Slack transport path resumes a paused link BEFORE the inbound "
            "governance gate; a denied message would leave the channel connected"
        )

    def test_the_native_handler_resumes_after_the_gate(self):
        from kiro_crew.slack import handler

        src = self._source(handler)
        gate = src.find('channel_inbound_permitted("slack")')
        resume = src.find("await _resume_if_paused(")
        assert gate != -1 and resume != -1
        assert resume > gate, (
            "the native Slack handler resumes a paused link BEFORE the inbound "
            "governance gate"
        )

    def test_the_discord_dispatch_resumes_after_the_gate(self):
        from kiro_crew.discord import transport_dispatch

        src = self._source(transport_dispatch)
        gate = src.find('channel_inbound_permitted("discord")')
        resume = src.find("resume_if_muted(")
        assert gate != -1 and resume != -1
        assert resume > gate, (
            "the Discord path resumes a muted binding BEFORE the inbound "
            "governance gate"
        )
