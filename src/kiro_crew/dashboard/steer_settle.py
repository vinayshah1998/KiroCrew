"""Settlement of pending mid-turn steers against a ``steering_consumed`` echo.

Pure and shared: the main chat and the ``/side`` sidecar both hand kiro-cli
fire-and-forget steers, and both need the same answer to "which of these did the
backend actually inject?". One subtle parser, two callers — a second copy would
drift, and the failure mode of a wrong answer is a silently lost question.
"""

from __future__ import annotations

import re

#: kiro-cli wraps each injected steer in this envelope inside the echo text.
_BLOCK_RE = re.compile(r"<user_message>\n(.*?)\n</user_message>", re.DOTALL)


def settle_consumed_steers(pending: list[str], snapshot: str) -> list[str]:
    """Return the entries of *pending* that ``snapshot`` did NOT account for.

    kiro-cli injects the CONCATENATION of every steer queued since the last
    consumption, and the echo carries each one ``<user_message>``-wrapped. Parse
    the snapshot into blocks and settle by EQUALITY: substring containment would
    false-positive a short steer against a longer one or against the wrapper
    text itself, and a falsely-settled steer is silently lost when the turn ends.

    Settling is COUNT-AWARE — each block settles at most one pending entry, so a
    duplicate identical steer registered after the snapshot stays pending
    instead of being swept by set membership. An entry registered after kiro-cli
    took its snapshot is simply not among the blocks and stays pending.

    When the echo carries no usable text (older backend, redacted echo), settle
    everything: the worst case is then a visible, cancellable duplicate rather
    than a silent loss.
    """
    if not snapshot.strip():
        return []
    counts: dict[str, int] = {}
    for block in _BLOCK_RE.findall(snapshot):
        counts[block] = counts.get(block, 0) + 1
    remaining: list[str] = []
    for message in pending:
        # The steer RPC wraps message.strip(); pending stores the raw message.
        # Strip for parity so whitespace never causes a false NON-match.
        key = message.strip()
        if counts.get(key, 0) > 0:
            counts[key] -= 1
        else:
            remaining.append(message)
    return remaining
