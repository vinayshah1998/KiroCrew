"""SideState: sidecar buffer attached to a parent ChatSlot.

Side messages live only on ``slot._side``; they are never persisted to
JSONL, memory.db, lessons, or preferences. Lifecycle: open → turn(s) → close.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: FIFO ceiling on messages held behind an in-flight side turn. The sidecar is
#: ephemeral and lives entirely in memory on the parent slot, so an unbounded
#: queue is a client-driven memory sink; refusing past this depth keeps the
#: pressure visible to the user instead of silently growing the process.
MAX_SIDE_QUEUE = 20


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SideState:
    """One side conversation attached to a parent slot.

    ``is_complete`` is False while a turn is in flight; flipped True in the
    ``_run_side_turn`` finally block.

    ``queue`` holds messages submitted while a turn is in flight, as
    ``{"id", "content", "ts"}`` dicts. It is drained one entry per turn by
    ``_run_side_turn``'s finally block.
    """

    open: bool = False
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_run_id: str = ""
    is_complete: bool = True
    created_at: str = field(default_factory=_now_iso)
    queue: list[dict[str, str]] = field(default_factory=list)
    #: Steers handed to the backend that it has not yet echoed as consumed. A
    #: steer is a fire-and-forget write, so an entry here is a question whose
    #: delivery is still UNPROVEN; whatever is left when the turn ends never
    #: reached a generation and is requeued rather than lost.
    pending_steers: list[str] = field(default_factory=list)

    def append_user(self, content: str, ts: str = "", *, steer: bool = False) -> None:
        """Append a user turn. ``steer`` marks it as injected mid-turn, which the
        panel renders as a distinct bubble rather than a normal question."""
        entry: dict[str, Any] = {
            "role": "user",
            "content": content,
            "ts": ts or _now_iso(),
        }
        if steer:
            entry["steer"] = True
        self.messages.append(entry)

    def append_assistant(self, content: str, ts: str = "") -> None:
        self.messages.append(
            {
                "role": "assistant",
                "content": content,
                "ts": ts or _now_iso(),
            }
        )

    def clear(self) -> None:
        self.messages.clear()
        self.last_run_id = ""
        self.is_complete = True
        self.queue.clear()
        self.pending_steers.clear()

    # ── Queue helpers ──

    def queue_append(self, content: str) -> str | None:
        """Append to the pending queue. Returns the queue ID, or None when the
        queue is already at :data:`MAX_SIDE_QUEUE` (caller surfaces a 429)."""
        if len(self.queue) >= MAX_SIDE_QUEUE:
            return None
        qid = uuid.uuid4().hex[:12]
        self.queue.append({"id": qid, "content": content, "ts": _now_iso()})
        return qid

    def queue_insert_front(self, content: str) -> str:
        """Put *content* at the HEAD of the queue, ahead of anything waiting.

        For text that was meant to reach the CURRENT turn (an unconsumed steer,
        or an entry whose dispatch failed) — it should run before entries the
        user submitted after it. Unbounded on purpose: refusing here would drop
        text the user already believes was accepted, which is the exact loss the
        bound exists to make visible rather than to cause.
        """
        qid = uuid.uuid4().hex[:12]
        self.queue.insert(0, {"id": qid, "content": content, "ts": _now_iso()})
        return qid

    def queue_pop(self) -> dict[str, str] | None:
        """Pop the oldest queued entry, or None when the queue is empty."""
        if not self.queue:
            return None
        return self.queue.pop(0)

    def queue_remove(self, queue_id: str) -> str | None:
        """Remove one entry by ID. Returns its content, or None if not found."""
        for i, item in enumerate(self.queue):
            if item["id"] == queue_id:
                del self.queue[i]
                return item["content"]
        return None

    def queue_edit(self, queue_id: str, content: str) -> bool:
        """Replace one entry's content by ID, preserving order. True if found."""
        for item in self.queue:
            if item["id"] == queue_id:
                item["content"] = content
                return True
        return False
