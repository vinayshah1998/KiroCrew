"""Settlement of pending steers against a ``steering_consumed`` echo.

The wrong answer here loses a user's question silently, so each rule gets a test:
equality (not containment), count-awareness, and settle-all on an unusable echo.
Shared by the main chat and the /side sidecar.
"""

from __future__ import annotations

from kiro_crew.dashboard.steer_settle import settle_consumed_steers


def _echo(*messages: str) -> str:
    return "".join(f"<user_message>\n{m}\n</user_message>" for m in messages)


def test_a_consumed_steer_is_settled():
    assert settle_consumed_steers(["use QUIC"], _echo("use QUIC")) == []


def test_a_steer_registered_after_the_snapshot_stays_pending():
    remaining = settle_consumed_steers(["first", "second"], _echo("first"))
    assert remaining == ["second"]


def test_settling_matches_by_equality_not_containment():
    """A short steer must not be settled by a longer one that contains it —
    a falsely-settled steer is never requeued, so the question is lost."""
    remaining = settle_consumed_steers(["ls"], _echo("please run ls in /tmp"))
    assert remaining == ["ls"]


def test_settling_is_count_aware():
    """One echoed block settles exactly one pending entry, so a duplicate
    submitted after the snapshot survives instead of being swept."""
    remaining = settle_consumed_steers(["retry", "retry"], _echo("retry"))
    assert remaining == ["retry"]


def test_whitespace_does_not_cause_a_false_non_match():
    """The RPC wraps ``message.strip()`` while pending holds the raw text."""
    assert settle_consumed_steers(["  spaced  "], _echo("spaced")) == []


def test_an_unusable_echo_settles_everything():
    """An empty/redacted echo means the backend gave no usable text. Settling all
    risks a visible, cancellable duplicate; keeping all pending would requeue
    steers that WERE injected, asking every one of them twice."""
    assert settle_consumed_steers(["a", "b"], "") == []
    assert settle_consumed_steers(["a", "b"], "   ") == []


def test_an_echo_without_recognisable_blocks_keeps_entries_pending():
    """Text that is present but carries no envelope settles nothing — the safe
    direction is a duplicate card, never a silent loss."""
    assert settle_consumed_steers(["a"], "some unrelated prose") == ["a"]
