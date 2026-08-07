"""/side steer + queue invariants — the main chat's busy-send semantics, applied
to the sidecar.

1. Steer: a message marked ``steer`` is injected into the RUNNING side turn and
   does not start a second run.
2. Fall-through: an unavailable/refused steer lands on the queue instead of
   being dropped.
3. Queue mode: without the ``steer`` flag an in-flight submit always queues.
4. Drain: the queue is drained FIFO, one entry per turn.
5. Cancel/edit: a queued entry can be removed (content echoed back for the
   composer) or rewritten in place with order preserved.
6. Bound: the queue refuses past ``MAX_SIDE_QUEUE`` rather than growing without
   limit.
7. No stranding: an entry queued after the turn already finished is drained
   immediately instead of waiting for a finally that already ran.
8. Close wins: closing the side drops the queue, and a late drain from the
   finished turn cannot resurrect it.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_state

from kiro_crew.dashboard.handlers.side import (
    api_side_close,
    api_side_open,
    api_side_queue_cancel,
    api_side_queue_edit,
    api_side_turn,
)
from kiro_crew.dashboard.side_state import MAX_SIDE_QUEUE, SideState
from kiro_crew.kiro_prerequisite import KiroPrerequisiteService


class _ReadyKiroPrerequisiteService(KiroPrerequisiteService):
    async def session_ready(self) -> bool:
        return True


_READY_KIRO_PREREQUISITE = object.__new__(_ReadyKiroPrerequisiteService)


def _make_side_app(state) -> web.Application:
    app = web.Application()
    app["state"] = state
    app["kiro_prerequisite_service"] = _READY_KIRO_PREREQUISITE
    app.router.add_post("/api/chat/slots/{slot}/side/open", api_side_open)
    app.router.add_post("/api/chat/slots/{slot}/side/turn", api_side_turn)
    app.router.add_post("/api/chat/slots/{slot}/side/close", api_side_close)
    app.router.add_delete(
        "/api/chat/slots/{slot}/side/queue/{queue_id}", api_side_queue_cancel
    )
    app.router.add_patch(
        "/api/chat/slots/{slot}/side/queue/{queue_id}", api_side_queue_edit
    )
    return app


def _capture_broadcasts(state) -> list[tuple[str, Any]]:
    events: list[tuple[str, Any]] = []
    state.broadcast_ws = lambda msg_type, data: events.append((msg_type, data))
    return events


def _install_gated_stream(
    state, monkeypatch, gate: asyncio.Event, *, echo_steers: bool = True
) -> list[str]:
    """Run the REAL ``_run_side_turn`` against a fake stream whose first call
    blocks on *gate*, so a turn stays genuinely in flight while the test posts.

    ``echo_steers`` mirrors a healthy backend: before the turn ends it echoes
    ``steering_consumed`` for whatever is pending, which is what settles a steer.
    Pass False to model a backend that took the write but never injected it.

    Returns the list of prompts the stream saw, in dispatch order — the drain's
    FIFO ordering is asserted against it.
    """
    seen: list[str] = []

    async def _fake_get_or_create(key, **kwargs):
        return MagicMock(), True, False

    state.sessions.get_or_create = _fake_get_or_create
    state.sessions.release = MagicMock()
    state.sessions.destroy = AsyncMock()

    async def _fake_stream(
        provider, message, *, on_chunk=None, on_steer_consumed=None, **kwargs
    ):
        seen.append(message)
        if len(seen) == 1:
            await gate.wait()
        if echo_steers and on_steer_consumed is not None:
            slot = next(iter(state._slots.values()))
            pending = list(getattr(slot._side, "pending_steers", []) or [])
            if pending:
                on_steer_consumed(
                    "".join(f"<user_message>\n{m}\n</user_message>" for m in pending)
                )
        return "answer"

    monkeypatch.setattr(
        "kiro_crew.dashboard.handlers.side.stream_and_collect", _fake_stream
    )
    return seen


def _steerable_provider(state, *, supports: bool = True, live: bool = True) -> list[str]:
    """Register a fake isolated session provider and return the steer sink."""
    steered: list[str] = []

    async def _steer(text: str) -> bool:
        steered.append(text)
        return True

    provider = MagicMock()
    provider.supports_steer = supports
    provider.has_active_turn = MagicMock(return_value=live)
    provider.steer = _steer
    state.sessions.get_provider = MagicMock(return_value=provider)
    return steered


async def _settle(state, timeout: float = 5.0) -> None:
    """Wait for every side background task (including drained ones) to finish."""
    deadline = asyncio.get_running_loop().time() + timeout
    while state._background_tasks:
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"side tasks did not settle: {state._background_tasks}")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_in_flight_steer_injects_into_the_running_turn(tmp_path, monkeypatch):
    """``steer`` reaches the live turn and does NOT start a second run."""
    state = _make_state(tmp_path)
    events = _capture_broadcasts(state)
    parent = state.get_or_create_slot("parent")
    gate = asyncio.Event()
    _install_gated_stream(state, monkeypatch, gate)
    steered = _steerable_provider(state)

    app = _make_side_app(state)
    async with TestClient(TestServer(app)) as client:
        await client.post("/api/chat/slots/parent/side/open", json={})
        first = await client.post(
            "/api/chat/slots/parent/side/turn", json={"question": "q1"}
        )
        run_id = (await first.json())["run_id"]

        resp = await client.post(
            "/api/chat/slots/parent/side/turn",
            json={"question": "actually use QUIC", "steer": True},
        )
        body = await resp.json()
        assert resp.status == 200, body
        assert body["steered"] is True
        assert body["run_id"] == run_id, "steer must not mint a new run"
        assert steered == ["actually use QUIC"]
        assert parent._side.queue == [], "a successful steer must not also queue"

        gate.set()
        await _settle(state)

    user_frames = [
        d for t, d in events if t == "chat.side_result" and d.get("role") == "user"
    ]
    assert user_frames[-1]["steer"] is True
    assert user_frames[-1]["run_id"] == run_id
    stored = [m for m in parent._side.messages if m["role"] == "user"]
    assert stored[-1].get("steer") is True


@pytest.mark.asyncio
async def test_steer_unavailable_falls_through_to_the_queue(tmp_path, monkeypatch):
    """A backend with no steer support must queue, never drop the text."""
    state = _make_state(tmp_path)
    _capture_broadcasts(state)
    parent = state.get_or_create_slot("parent")
    gate = asyncio.Event()
    seen = _install_gated_stream(state, monkeypatch, gate)
    _steerable_provider(state, supports=False)

    app = _make_side_app(state)
    async with TestClient(TestServer(app)) as client:
        await client.post("/api/chat/slots/parent/side/open", json={})
        await client.post("/api/chat/slots/parent/side/turn", json={"question": "q1"})
        resp = await client.post(
            "/api/chat/slots/parent/side/turn",
            json={"question": "fallback me", "steer": True},
        )
        body = await resp.json()
        assert resp.status == 200, body
        assert body["queued"] is True
        assert body["depth"] == 1
        assert [e["content"] for e in parent._side.queue] == ["fallback me"]

        gate.set()
        await _settle(state)

    assert "fallback me" in seen, "queued fallback never reached a turn"


@pytest.mark.asyncio
async def test_queue_mode_defers_even_when_steer_is_available(tmp_path, monkeypatch):
    """Without the flag, an in-flight submit queues — the steerable session is
    left untouched (this is the split button's Queue side)."""
    state = _make_state(tmp_path)
    _capture_broadcasts(state)
    parent = state.get_or_create_slot("parent")
    gate = asyncio.Event()
    _install_gated_stream(state, monkeypatch, gate)
    steered = _steerable_provider(state)

    app = _make_side_app(state)
    async with TestClient(TestServer(app)) as client:
        await client.post("/api/chat/slots/parent/side/open", json={})
        await client.post("/api/chat/slots/parent/side/turn", json={"question": "q1"})
        resp = await client.post(
            "/api/chat/slots/parent/side/turn", json={"question": "later please"}
        )
        assert (await resp.json())["queued"] is True
        assert steered == [], "queue mode must not steer"
        assert [e["content"] for e in parent._side.queue] == ["later please"]

        gate.set()
        await _settle(state)


@pytest.mark.asyncio
async def test_queue_drains_fifo_one_entry_per_turn(tmp_path, monkeypatch):
    """Queued questions run in submission order after the turn completes."""
    state = _make_state(tmp_path)
    events = _capture_broadcasts(state)
    parent = state.get_or_create_slot("parent")
    gate = asyncio.Event()
    seen = _install_gated_stream(state, monkeypatch, gate)

    app = _make_side_app(state)
    async with TestClient(TestServer(app)) as client:
        await client.post("/api/chat/slots/parent/side/open", json={})
        await client.post("/api/chat/slots/parent/side/turn", json={"question": "q1"})
        await client.post("/api/chat/slots/parent/side/turn", json={"question": "q2"})
        await client.post("/api/chat/slots/parent/side/turn", json={"question": "q3"})
        assert [e["content"] for e in parent._side.queue] == ["q2", "q3"]

        gate.set()
        await _settle(state)

    # First prompt carries the parent-snapshot envelope; later turns reuse the
    # session and go out bare, so they are byte-comparable.
    assert seen[1:] == ["q2", "q3"], seen
    assert parent._side.queue == []
    drained = [
        d
        for t, d in events
        if t == "chat.side_queue" and d.get("action") == "drain"
    ]
    assert [d["depth"] for d in drained] == [1, 0]


@pytest.mark.asyncio
async def test_queue_cancel_removes_the_entry_and_echoes_content(tmp_path, monkeypatch):
    """Cancel drops the entry and returns its text so the composer can restore it."""
    state = _make_state(tmp_path)
    events = _capture_broadcasts(state)
    parent = state.get_or_create_slot("parent")
    gate = asyncio.Event()
    seen = _install_gated_stream(state, monkeypatch, gate)

    app = _make_side_app(state)
    async with TestClient(TestServer(app)) as client:
        await client.post("/api/chat/slots/parent/side/open", json={})
        await client.post("/api/chat/slots/parent/side/turn", json={"question": "q1"})
        queued = await client.post(
            "/api/chat/slots/parent/side/turn", json={"question": "drop me"}
        )
        qid = (await queued.json())["queue_id"]

        resp = await client.delete(f"/api/chat/slots/parent/side/queue/{qid}")
        body = await resp.json()
        assert resp.status == 200, body
        assert body["content"] == "drop me"
        assert body["depth"] == 0
        assert parent._side.queue == []

        missing = await client.delete(f"/api/chat/slots/parent/side/queue/{qid}")
        assert missing.status == 404

        gate.set()
        await _settle(state)

    assert "drop me" not in seen, "cancelled entry still ran"
    cancels = [
        d for t, d in events if t == "chat.side_queue" and d.get("action") == "cancel"
    ]
    assert cancels and cancels[-1]["queue_id"] == qid


@pytest.mark.asyncio
async def test_queue_edit_rewrites_in_place_preserving_order(tmp_path, monkeypatch):
    """Editing the second of two entries changes only its content."""
    state = _make_state(tmp_path)
    _capture_broadcasts(state)
    parent = state.get_or_create_slot("parent")
    gate = asyncio.Event()
    seen = _install_gated_stream(state, monkeypatch, gate)

    app = _make_side_app(state)
    async with TestClient(TestServer(app)) as client:
        await client.post("/api/chat/slots/parent/side/open", json={})
        await client.post("/api/chat/slots/parent/side/turn", json={"question": "q1"})
        await client.post("/api/chat/slots/parent/side/turn", json={"question": "keep"})
        second = await client.post(
            "/api/chat/slots/parent/side/turn", json={"question": "old"}
        )
        qid = (await second.json())["queue_id"]

        resp = await client.patch(
            f"/api/chat/slots/parent/side/queue/{qid}", json={"content": "new"}
        )
        assert resp.status == 200, await resp.json()
        assert [e["content"] for e in parent._side.queue] == ["keep", "new"]

        blank = await client.patch(
            f"/api/chat/slots/parent/side/queue/{qid}", json={"content": "   "}
        )
        assert blank.status == 400

        gate.set()
        await _settle(state)

    assert seen[1:] == ["keep", "new"], seen


@pytest.mark.asyncio
async def test_queue_refuses_past_its_bound(tmp_path, monkeypatch):
    """The sidecar lives in memory on the parent slot: the queue must be bounded."""
    state = _make_state(tmp_path)
    _capture_broadcasts(state)
    parent = state.get_or_create_slot("parent")
    gate = asyncio.Event()
    _install_gated_stream(state, monkeypatch, gate)

    app = _make_side_app(state)
    async with TestClient(TestServer(app)) as client:
        await client.post("/api/chat/slots/parent/side/open", json={})
        await client.post("/api/chat/slots/parent/side/turn", json={"question": "q1"})
        for i in range(MAX_SIDE_QUEUE):
            resp = await client.post(
                "/api/chat/slots/parent/side/turn", json={"question": f"q{i}"}
            )
            assert resp.status == 200, await resp.json()

        overflow = await client.post(
            "/api/chat/slots/parent/side/turn", json={"question": "too much"}
        )
        assert overflow.status == 429
        body = await overflow.json()
        assert str(MAX_SIDE_QUEUE) in body["error"]
        assert len(parent._side.queue) == MAX_SIDE_QUEUE

        gate.set()
        await _settle(state)


@pytest.mark.asyncio
async def test_entry_queued_after_completion_is_not_stranded(tmp_path, monkeypatch):
    """A steer that loses the race to the turn's end must still get answered.

    The drain hook only runs inside a turn's ``finally``. If the turn ends while
    the steer RPC is awaiting, the fallback entry lands on a queue nothing will
    ever visit — so the queue path kicks the drain itself.
    """
    state = _make_state(tmp_path)
    _capture_broadcasts(state)
    parent = state.get_or_create_slot("parent")
    gate = asyncio.Event()
    seen = _install_gated_stream(state, monkeypatch, gate, echo_steers=False)

    async def _steer_after_turn_ends(text: str) -> bool:
        # Let the in-flight turn finish while this RPC is suspended, exactly as a
        # slow stdin.drain() would.
        gate.set()
        await _settle(state)
        return False

    provider = MagicMock()
    provider.supports_steer = True
    provider.has_active_turn = MagicMock(return_value=True)
    provider.steer = _steer_after_turn_ends
    state.sessions.get_provider = MagicMock(return_value=provider)

    app = _make_side_app(state)
    async with TestClient(TestServer(app)) as client:
        await client.post("/api/chat/slots/parent/side/open", json={})
        await client.post("/api/chat/slots/parent/side/turn", json={"question": "q1"})
        resp = await client.post(
            "/api/chat/slots/parent/side/turn",
            json={"question": "late question", "steer": True},
        )
        assert (await resp.json())["queued"] is True
        await _settle(state)

    assert "late question" in seen, "entry queued past the finally was stranded"
    assert parent._side.queue == []


@pytest.mark.asyncio
async def test_close_drops_the_queue_and_a_late_drain_cannot_resurrect_it(
    tmp_path, monkeypatch
):
    """Closing mid-turn discards queued text; the finished turn's drain no-ops."""
    state = _make_state(tmp_path)
    _capture_broadcasts(state)
    parent = state.get_or_create_slot("parent")
    gate = asyncio.Event()
    seen = _install_gated_stream(state, monkeypatch, gate)

    app = _make_side_app(state)
    async with TestClient(TestServer(app)) as client:
        await client.post("/api/chat/slots/parent/side/open", json={})
        await client.post("/api/chat/slots/parent/side/turn", json={"question": "q1"})
        await client.post(
            "/api/chat/slots/parent/side/turn", json={"question": "orphan"}
        )
        await client.post("/api/chat/slots/parent/side/close", json={})
        assert parent._side is None

        gate.set()
        await _settle(state)

    assert "orphan" not in seen, "close did not discard the queued entry"
    assert parent._side is None


@pytest.mark.asyncio
async def test_a_stale_task_cannot_drain_a_newer_sides_queue(tmp_path, monkeypatch):
    """The drain is identity-checked on run_id, not just on busy-ness.

    A task belonging to a superseded run can reach its ``finally`` long after a
    close/reopen replaced the sidecar. Matching only on ``is_complete`` would let
    it pop the NEW side's queue and dispatch a turn the new run never asked for.
    """
    from kiro_crew.dashboard.handlers.side import _drain_side_queue

    state = _make_state(tmp_path)
    _capture_broadcasts(state)
    parent = state.get_or_create_slot("parent")
    parent._side = SideState(open=True)
    parent._side.last_run_id = "run-new"
    parent._side.is_complete = True
    qid = parent._side.queue_append("belongs to the new side")

    dispatched: list[str] = []
    monkeypatch.setattr(
        "kiro_crew.dashboard.handlers.side._dispatch_side_turn",
        lambda state, slot, question: dispatched.append(question) or "run-x",
    )

    _drain_side_queue(state, parent, "run-stale")

    assert dispatched == [], "a stale run drained the current side's queue"
    assert [e["id"] for e in parent._side.queue] == [qid]

    _drain_side_queue(state, parent, "run-new")
    assert dispatched == ["belongs to the new side"]


@pytest.mark.asyncio
async def test_a_steer_that_lands_after_its_run_ended_is_queued_not_claimed(
    tmp_path, monkeypatch
):
    """A steer whose RPC outlives its own turn must not be reported as delivered.

    kiro-cli accepts the write but no live generation consumes it, so claiming
    ``steered`` against whatever run is current now would leave the question
    unanswered — and attribute the bubble to a turn that never saw it.
    """
    state = _make_state(tmp_path)
    _capture_broadcasts(state)
    parent = state.get_or_create_slot("parent")
    gate = asyncio.Event()
    seen = _install_gated_stream(state, monkeypatch, gate, echo_steers=False)

    async def _steer_after_the_run_ends(text: str) -> bool:
        # Let the in-flight turn finish while this RPC is suspended, then report
        # the write as having succeeded.
        gate.set()
        await _settle(state)
        return True

    provider = MagicMock()
    provider.supports_steer = True
    provider.has_active_turn = MagicMock(return_value=True)
    provider.steer = _steer_after_the_run_ends
    state.sessions.get_provider = MagicMock(return_value=provider)

    app = _make_side_app(state)
    async with TestClient(TestServer(app)) as client:
        await client.post("/api/chat/slots/parent/side/open", json={})
        await client.post("/api/chat/slots/parent/side/turn", json={"question": "q1"})
        resp = await client.post(
            "/api/chat/slots/parent/side/turn",
            json={"question": "orphan steer", "steer": True},
        )
        body = await resp.json()
        assert body.get("steered") is not True, body
        assert body["queued"] is True
        await _settle(state)

    assert "orphan steer" in seen, "the swallowed steer was never re-delivered"
    assert not any(m.get("steer") for m in parent._side.messages)


@pytest.mark.asyncio
async def test_a_steer_cannot_land_on_a_sidecar_that_was_replaced(tmp_path, monkeypatch):
    """close + reopen during the steer RPC yields a fresh, also-``open`` sidecar.

    Checking only ``open`` would attach the text to a side conversation that
    never asked for it, so the commit is bound to the sidecar OBJECT.
    """
    state = _make_state(tmp_path)
    _capture_broadcasts(state)
    parent = state.get_or_create_slot("parent")
    gate = asyncio.Event()
    _install_gated_stream(state, monkeypatch, gate)

    async def _steer_then_replace_the_sidecar(text: str) -> bool:
        parent._side = SideState(open=True)
        return True

    provider = MagicMock()
    provider.supports_steer = True
    provider.has_active_turn = MagicMock(return_value=True)
    provider.steer = _steer_then_replace_the_sidecar
    state.sessions.get_provider = MagicMock(return_value=provider)

    app = _make_side_app(state)
    async with TestClient(TestServer(app)) as client:
        await client.post("/api/chat/slots/parent/side/open", json={})
        await client.post("/api/chat/slots/parent/side/turn", json={"question": "q1"})
        resp = await client.post(
            "/api/chat/slots/parent/side/turn",
            json={"question": "wrong side", "steer": True},
        )
        assert resp.status == 409, await resp.json()
        assert parent._side.messages == [], "steer polluted the replacement sidecar"
        assert parent._side.queue == []

        gate.set()
        await _settle(state)


@pytest.mark.asyncio
async def test_an_unconsumed_steer_becomes_a_queue_card_instead_of_vanishing(
    tmp_path, monkeypatch
):
    """``steer()`` returning True only proves the bytes left the process.

    The backend's ``steering_consumed`` echo is the authoritative signal. A steer
    that reaches the process but no generation — the turn ends first — would
    otherwise be reported delivered and then silently lost, so it is requeued as
    an ordinary, cancellable card.
    """
    state = _make_state(tmp_path)
    events = _capture_broadcasts(state)
    parent = state.get_or_create_slot("parent")
    gate = asyncio.Event()
    seen = _install_gated_stream(state, monkeypatch, gate, echo_steers=False)
    _steerable_provider(state)

    app = _make_side_app(state)
    async with TestClient(TestServer(app)) as client:
        await client.post("/api/chat/slots/parent/side/open", json={})
        await client.post("/api/chat/slots/parent/side/turn", json={"question": "q1"})
        resp = await client.post(
            "/api/chat/slots/parent/side/turn",
            json={"question": "never consumed", "steer": True},
        )
        assert (await resp.json())["steered"] is True
        assert parent._side.pending_steers == ["never consumed"]

        # The turn ends without the backend ever echoing steering_consumed.
        gate.set()
        await _settle(state)

    assert "never consumed" in seen, "the unconsumed steer was lost"
    assert parent._side.pending_steers == []
    pushes = [
        d for t, d in events if t == "chat.side_queue" and d.get("action") == "push"
    ]
    assert pushes, "no queue card was broadcast for the orphaned steer"


@pytest.mark.asyncio
async def test_a_consumed_steer_is_settled_and_not_requeued(tmp_path, monkeypatch):
    """The echo proves injection, so that steer must NOT come back as a card.

    Without settling, every steer would be asked a second time.
    """
    state = _make_state(tmp_path)
    _capture_broadcasts(state)
    parent = state.get_or_create_slot("parent")

    async def _fake_get_or_create(key, **kwargs):
        return MagicMock(), True, False

    state.sessions.get_or_create = _fake_get_or_create
    state.sessions.release = MagicMock()
    state.sessions.destroy = AsyncMock()

    async def _fake_stream(provider, message, *, on_chunk=None, on_steer_consumed=None, **kw):
        # The handler registers the steer, then the backend echoes it back
        # ``<user_message>``-wrapped exactly as kiro-cli does.
        parent._side.pending_steers.append("consumed steer")
        if on_steer_consumed:
            on_steer_consumed("<user_message>\nconsumed steer\n</user_message>")
        return "answer"

    monkeypatch.setattr(
        "kiro_crew.dashboard.handlers.side.stream_and_collect", _fake_stream
    )

    app = _make_side_app(state)
    async with TestClient(TestServer(app)) as client:
        await client.post("/api/chat/slots/parent/side/open", json={})
        await client.post("/api/chat/slots/parent/side/turn", json={"question": "q1"})
        await _settle(state)

    assert parent._side.pending_steers == []
    assert parent._side.queue == [], "a consumed steer was requeued as a duplicate"


@pytest.mark.asyncio
async def test_a_failed_drain_puts_the_entry_back_instead_of_dropping_it(
    tmp_path, monkeypatch
):
    """The card is already retired on the client when the drain pops it, so a
    dispatch failure must return the text to the queue, not just report."""
    from kiro_crew.dashboard.handlers.side import _drain_side_queue

    state = _make_state(tmp_path)
    events = _capture_broadcasts(state)
    parent = state.get_or_create_slot("parent")
    parent._side = SideState(open=True)
    parent._side.last_run_id = "run-1"
    parent._side.is_complete = True
    parent._side.queue_append("must survive")

    def _boom(*_a, **_k):
        raise RuntimeError("dispatch exploded")

    monkeypatch.setattr(
        "kiro_crew.dashboard.handlers.side._dispatch_side_turn", _boom
    )

    _drain_side_queue(state, parent, "run-1")

    assert [e["content"] for e in parent._side.queue] == ["must survive"]
    errors = [d for t, d in events if d.get("is_error")]
    assert errors and "back in the queue" in errors[-1]["content"]


@pytest.mark.asyncio
async def test_a_FAILED_steer_cannot_queue_onto_a_replaced_sidecar(tmp_path, monkeypatch):
    """The identity check has to cover the failure path too.

    A steer that returns False still falls through to the queue, and if a
    close+reopen landed during the RPC that queue belongs to a different side
    conversation — one that never asked for this question.
    """
    state = _make_state(tmp_path)
    _capture_broadcasts(state)
    parent = state.get_or_create_slot("parent")
    gate = asyncio.Event()
    _install_gated_stream(state, monkeypatch, gate)

    async def _steer_then_replace_the_sidecar(text: str) -> bool:
        parent._side = SideState(open=True)
        return False

    provider = MagicMock()
    provider.supports_steer = True
    provider.has_active_turn = MagicMock(return_value=True)
    provider.steer = _steer_then_replace_the_sidecar
    state.sessions.get_provider = MagicMock(return_value=provider)

    app = _make_side_app(state)
    async with TestClient(TestServer(app)) as client:
        await client.post("/api/chat/slots/parent/side/open", json={})
        await client.post("/api/chat/slots/parent/side/turn", json={"question": "q1"})
        resp = await client.post(
            "/api/chat/slots/parent/side/turn",
            json={"question": "wrong side", "steer": True},
        )
        assert resp.status == 409, await resp.json()
        assert parent._side.queue == [], "queued onto the replacement sidecar"
        assert parent._side.messages == []

        gate.set()
        await _settle(state)


@pytest.mark.asyncio
async def test_a_stale_consumption_echo_cannot_settle_a_replacement_sidecar(
    tmp_path, monkeypatch
):
    """An echo belongs to the turn that produced it.

    If a close+reopen swapped the sidecar mid-turn, letting the old turn's echo
    settle the NEW sidecar's pending steer would mean that steer is never
    requeued — the question disappears with no card and no answer.
    """
    from kiro_crew.dashboard.handlers.side import _run_side_turn

    state = _make_state(tmp_path)
    _capture_broadcasts(state)
    parent = state.get_or_create_slot("parent")
    original = SideState(open=True)
    original.last_run_id = "run-old"
    original.is_complete = False
    original.pending_steers.append("belongs to the old turn")
    parent._side = original

    replacement = SideState(open=True)
    replacement.pending_steers.append("belongs to the NEW side")

    async def _fake_get_or_create(key, **kwargs):
        return MagicMock(), True, False

    state.sessions.get_or_create = _fake_get_or_create
    state.sessions.release = MagicMock()
    state.sessions.destroy = AsyncMock()

    async def _fake_stream(provider, message, *, on_chunk=None, on_steer_consumed=None, **kw):
        # The sidecar is replaced part-way through, then the OLD turn's backend
        # echoes a steer naming the replacement's text.
        parent._side = replacement
        if on_steer_consumed:
            on_steer_consumed("<user_message>\nbelongs to the NEW side\n</user_message>")
        return "answer"

    monkeypatch.setattr(
        "kiro_crew.dashboard.handlers.side.stream_and_collect", _fake_stream
    )

    await _run_side_turn(state, parent, "run-old", "q", is_first_turn=True)

    assert replacement.pending_steers == ["belongs to the NEW side"], (
        "a stale echo settled the replacement sidecar's steer"
    )
    assert replacement.queue == [], "the old turn requeued onto the replacement"


@pytest.mark.asyncio
async def test_queue_mutations_require_an_open_side(tmp_path):
    """Cancel/edit on a closed side is a 409, not a silent success."""
    state = _make_state(tmp_path)
    _capture_broadcasts(state)
    state.get_or_create_slot("parent")

    app = _make_side_app(state)
    async with TestClient(TestServer(app)) as client:
        cancel = await client.delete("/api/chat/slots/parent/side/queue/abc")
        assert cancel.status == 409
        edit = await client.patch(
            "/api/chat/slots/parent/side/queue/abc", json={"content": "x"}
        )
        assert edit.status == 409
