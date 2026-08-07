"""Persistent session-to-kiro-cli mapping.

Stores ``session_map.json`` mapping session keys to kiro-cli session IDs,
with Slack thread linkage for bidirectional sync.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from kiro_crew.config.paths import config_dir, kiro_sessions_dir
from kiro_crew.messaging.link import (
    SLACK_NAMESPACE,
    ChannelLink,
    canonical_key,
    is_channel_session_key,
    legacy_dashboard_mirror_key,
)

logger = logging.getLogger(__name__)

_SESSION_MAP_FILE = "session_map.json"

# Resolved per call, never captured at import: an import-time binding freezes
# the data home and defeats pod isolation, the lazy legacy-home migration and
# test isolation. The name below is an opt-in override (None = live home) so
# existing monkeypatch call sites keep working. See config.md "Data Home" and
# issue #874; dashboard/handlers/usage.py is the reference implementation.
_KIRO_SESSIONS_DIR: Path | None = None


def _kiro_sessions_dir() -> Path:
    """kiro-cli sessions directory, resolved against the live data home."""
    return _KIRO_SESSIONS_DIR if _KIRO_SESSIONS_DIR is not None else kiro_sessions_dir()


class SessionMap:
    """Persistent mapping of session_key → kiro-cli session ID.

    Stored as ``~/.kiro/crew/session_map.json``. Atomic write via tmp+rename.
    Only used for long-lived conversational sessions (Slack DM, dashboard).
    Stateless sessions (cron, subagent, taskrunner) are excluded.

    Each entry is a dict with keys: ``sid``, ``slack_thread_ts``, ``slack_channel_id``.
    A reverse index ``_thread_to_session`` maps Slack thread_ts → session_key
    for bidirectional sync lookups.
    """

    def __init__(self) -> None:
        self._path = config_dir() / _SESSION_MAP_FILE
        self._data: dict[str, dict] = {}  # key → {"sid", "slack_thread_ts", "slack_channel_id"}
        self._thread_to_session: dict[str, str] = {}  # slack_thread_ts → session_key
        self._load()

    def _load(self) -> None:
        self._thread_to_session.clear()
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                self._data = {}
                return
            if not isinstance(raw, dict):
                self._data = {}
                return
            migrated = False
            new_data: dict[str, dict] = {}
            for key, val in raw.items():
                if isinstance(val, str):
                    # Backward compat: plain string → new dict format
                    entry: dict = {
                        "sid": val,
                        "slack_thread_ts": None,
                        "slack_channel_id": None,
                    }
                    migrated = True
                elif isinstance(val, dict) and "sid" in val:
                    entry = val
                else:
                    continue  # skip corrupt entries
                # v1c: namespace bare Slack thread_ts keys → slack:<thread>,
                # preserving sid. Keep the raw thread_ts inside the entry so the
                # reverse index + challenge-redirect resume path are unaffected,
                # and populate the Layer-3 own-channel link.
                canon = canonical_key(key)
                # Scrub a pre-release corruption signature: a ``dashboard:`` key
                # carrying no ``slack_thread_ts`` but a ``discord:``
                # ``slack_channel_id`` is an impossible pair. It is left by an
                # early Discord session-resume build that ran the legacy
                # Slack-only ``set_channel`` path on a cold resumed dashboard
                # session; ``!unlink`` removes the Discord mirror but cannot
                # clear these separate legacy fields, so a later resume attempt
                # sees a phantom Slack binding. A genuine Slack link always has a
                # thread timestamp, so scrub only this exact signature.
                legacy_channel = entry.get("slack_channel_id")
                if (
                    canon.startswith("dashboard:")
                    and not entry.get("slack_thread_ts")
                    and isinstance(legacy_channel, str)
                    and legacy_channel.startswith("discord:")
                ):
                    entry.pop("slack_thread_ts", None)
                    entry.pop("slack_channel_id", None)
                    migrated = True
                if canon != key:
                    migrated = True
                    if not entry.get("slack_thread_ts"):
                        entry["slack_thread_ts"] = key
                    if "link" not in entry:
                        entry["link"] = ChannelLink(
                            channel_type="slack",
                            channel_id=entry.get("slack_channel_id"),
                            thread_id=key,
                        ).to_dict()
                existing = new_data.get(canon)
                if existing is not None:
                    # Collision (e.g. partially-migrated file): never clobber a
                    # live session. Overwrite ONLY when the existing entry has no
                    # sid and the incoming one does; otherwise keep existing.
                    # Order-independent — if both have a sid, the first-seen wins
                    # deterministically rather than depending on dict iteration.
                    if not (entry.get("sid") and not existing.get("sid")):
                        continue
                new_data[canon] = entry
            self._data = new_data
            self._rebuild_thread_index()
            if migrated:
                self._save()
        else:
            self._data = {}

    def _rebuild_thread_index(self) -> None:
        """Rebuild _thread_to_session from current _data.

        Two entries can claim the same ``slack_thread_ts``: a dashboard session
        that created the thread via send-to-Slack, and a ``slack:<ts>`` session
        forked by an inbound reply that ignored the existing binding. A plain
        last-write-wins pass resolves the thread by dict order, which is file
        order -- so the fork usually wins and the thread keeps routing to the
        wrong session even after the fork bug is fixed.

        Break that tie in favour of the session that does NOT derive its key from
        this thread. A ``slack:<ts>`` key whose ts IS the thread is the fork (or
        a self-link, which is a no-op rewrite); any other key holds the real
        conversation. This heals maps corrupted before the fix, on load, with no
        migration pass.
        """
        self._thread_to_session.clear()
        derived: dict[str, str] = {}
        for key, entry in self._data.items():
            ts = entry.get("slack_thread_ts")
            if not ts or not isinstance(ts, str):
                # A hand-edited or legacy file can hold a non-string ts. The old
                # index only ever used it as a dict key, so it survived; the
                # tie-break below calls str.endswith, which would raise
                # TypeError here and take gateway startup down with it.
                continue
            if is_channel_session_key(key) and key.endswith(ts):
                # Self-derived: only usable if nothing else claims the thread.
                derived.setdefault(ts, key)
                continue
            self._thread_to_session[ts] = key
        for ts, key in derived.items():
            self._thread_to_session.setdefault(ts, key)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
            os.replace(tmp_path, str(self._path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def get(self, key: str) -> str | None:
        """Return kiro-cli session ID if mapping exists and .json file is present.

        Handles the dashboard history key round-trip: the original session key
        ``dashboard:chat-1-xxx`` becomes ``dashboard_chat-1-xxx`` on disk (via
        ``_safe_key``), and when resumed from history the slot name becomes
        ``dashboard_chat-1-xxx``, producing session key
        ``dashboard:dashboard_chat-1-xxx``.  We try the canonical form too.
        """
        entry = self._data.get(key)
        # Bidirectional bare <-> slack: shim: a not-yet-updated caller may pass
        # a bare thread_ts; resolve it to the namespaced entry written at load.
        if not entry:
            canon = canonical_key(key)
            if canon != key:
                entry = self._data.get(canon)
                if entry:
                    key = canon
        # Fallback: dashboard history round-trip (dashboard:dashboard_X → dashboard:X)
        matched_key = key
        if not entry and key.startswith("dashboard:dashboard_"):
            canonical = "dashboard:" + key[len("dashboard:dashboard_"):]
            entry = self._data.get(canonical)
            if entry:
                matched_key = canonical
        if not entry:
            return None
        sid = entry["sid"]
        if entry.get("provider") == "claude_code":
            return sid
        sessions_dir = _kiro_sessions_dir()
        if sid and (sessions_dir / f"{sid}.json").exists():
            jsonl = sessions_dir / f"{sid}.jsonl"
            try:
                jsonl_size = jsonl.stat().st_size
            except FileNotFoundError:
                jsonl_size = 0
            if jsonl_size < 10:
                logger.info("Session %s has empty JSONL — pruning stale entry for %s", sid, key)
                self._remove_entry(matched_key)
                return None
            return sid
        if sid:
            self._remove_entry(matched_key)
        return None

    def _remove_entry(self, key: str) -> None:
        """Remove an entry and update reverse index."""
        entry = self._data.pop(key, None)
        if entry:
            ts = entry.get("slack_thread_ts")
            if ts and self._thread_to_session.get(ts) == key:
                del self._thread_to_session[ts]
            self._save()

    def set(self, key: str, sid: str, *, provider: str = "", cwd: str = "") -> None:
        """Save mapping and persist to disk, preserving existing slack fields."""
        key = canonical_key(key)
        existing = self._data.get(key)
        if existing:
            existing["sid"] = sid
            if provider:
                existing["provider"] = provider
            if cwd:
                existing["cwd"] = cwd
        else:
            entry: dict = {"sid": sid, "slack_thread_ts": None, "slack_channel_id": None}
            if provider:
                entry["provider"] = provider
            if cwd:
                entry["cwd"] = cwd
            self._data[key] = entry
        self._save()

    def get_cwd(self, key: str) -> str:
        """Return the stored CWD for *key*, or '' if not set."""
        entry = self._data.get(canonical_key(key))
        if not entry:
            return ""
        return entry.get("cwd", "")

    def get_provider(self, key: str) -> str:
        """Return the stored provider for *key* (e.g. 'acp', 'claude_code'), or ''."""
        entry = self._data.get(canonical_key(key))
        if not entry:
            return ""
        return entry.get("provider", "")

    def clear_sid(self, key: str) -> None:
        """Clear the stored session ID without removing the entry.

        Used on provider switch: the SID is incompatible with the new
        provider, but we keep the entry so Slack link and CWD persist.
        """
        entry = self._data.get(canonical_key(key))
        if entry and entry.get("sid"):
            entry["sid"] = ""
            self._save()

    def delete(self, key: str) -> None:
        """Remove mapping and persist."""
        self._remove_entry(canonical_key(key))

    def prune(self) -> int:
        """Remove entries whose session files no longer exist."""
        sessions_dir = _kiro_sessions_dir()
        stale = [
            k
            for k, entry in self._data.items()
            if entry.get("provider") != "claude_code"
            and (
                (entry.get("sid") and not (sessions_dir / f"{entry['sid']}.json").exists())
                or (
                    not entry.get("sid")
                    and not entry.get("slack_thread_ts")
                    and not entry.get("mirror")
                    and not entry.get("mirrors")
                )
            )
        ]
        for k in stale:
            del self._data[k]
        if stale:
            self._rebuild_thread_index()
            self._save()
            logger.info("Pruned %d stale session map entries", len(stale))
        return len(stale)

    def set_slack_link(self, key: str, thread_ts: str, channel_id: str | None) -> None:
        """Link a session to a Slack thread. Creates entry if needed.

        Establishing a link DROPS any ``slack_paused`` marker: a bind replaces the
        link, so it must not inherit the previous one's mute. Without this a
        handoff release (``set_slack_link(key, "", "")``, which unlike
        ``clear_slack_link`` leaves the flag) would leave the next thread bound to
        this row born muted — no echo, no reply, no tool stream. Same rule the
        channel bindings follow in ``set_mirror_link``.
        """
        key = canonical_key(key)
        entry = self._data.get(key)
        if entry:
            if (
                entry.get("slack_thread_ts") == thread_ts
                and entry.get("slack_channel_id") == channel_id
            ):
                self._thread_to_session.setdefault(thread_ts, key)
                return
            old_ts = entry.get("slack_thread_ts")
            if old_ts and old_ts != thread_ts:
                self._thread_to_session.pop(old_ts, None)
            entry["slack_thread_ts"] = thread_ts
            entry["slack_channel_id"] = channel_id
            # Only on a real bind. The empty-string form is a RELEASE, and popping
            # here would make it indistinguishable from `clear_slack_link`.
            if thread_ts:
                entry.pop("slack_paused", None)
        else:
            self._data[key] = {
                "sid": "",
                "slack_thread_ts": thread_ts,
                "slack_channel_id": channel_id,
            }
        self._thread_to_session[thread_ts] = key
        self._save()

    def get_slack_link(self, key: str) -> tuple[str | None, str | None]:
        """Return (thread_ts, channel_id) for a session."""
        entry = self._data.get(canonical_key(key))
        if not entry:
            return None, None
        return entry.get("slack_thread_ts"), entry.get("slack_channel_id")

    def clear_slack_link(self, key: str) -> bool:
        """Remove the Slack link from a session, keeping the session itself.

        Clears the two coordinate fields plus any ``slack_paused`` marker
        (preserves ``sid`` and the entry) and evicts the ``_thread_to_session``
        reverse index so a later Slack reply in the old thread does not re-route
        to this session and silently re-engage mirroring. Returns True iff a link
        was present (only then is ``_save()`` called).
        """
        entry = self._data.get(canonical_key(key))
        if not entry:
            return False
        old_ts = entry.get("slack_thread_ts")
        had_link = bool(old_ts or entry.get("slack_channel_id"))
        if old_ts and self._thread_to_session.get(old_ts) == key:
            del self._thread_to_session[old_ts]
        entry.pop("slack_thread_ts", None)
        entry.pop("slack_channel_id", None)
        # A pause describes a link, so it must not outlive one: left behind, it
        # would re-pause a future link to this session that nobody paused.
        # ``had_link`` deliberately still keys on the two coordinate fields --
        # pause is only ever set on an already-linked entry, so a paused entry
        # always has a link to report, and a lone flag must not make an
        # unlinked session report that a link was cleared.
        entry.pop("slack_paused", None)
        if had_link:
            self._save()
        return had_link

    def _pause_keys(self, key: str) -> list[str]:
        """Every spelling a dashboard session's link can be stored under.

        ``chat_runner`` copies a dashboard session's link from the bare key onto
        the ``dashboard:``-prefixed one when a turn runs, so a pause written to
        one spelling alone would be read back from the other and silently lost.
        Owning that here means no call site has to remember it. A channel key
        (``slack:<ts>``) has no twin, so it yields a single spelling.
        """
        canonical = canonical_key(key)
        keys = [canonical]
        if canonical.startswith("dashboard:"):
            keys.append(canonical[len("dashboard:") :])
        return keys

    def set_slack_paused(self, key: str, paused: bool) -> bool:
        """Pause or resume outbound mirroring, keeping the thread binding.

        Returns the PREVIOUS state, so an idempotent endpoint can report whether
        it changed anything without a second read.

        Stored as a presence flag -- ``True`` or absent, never ``False`` --
        matching ``mirror_accepts_inbound`` above, so resuming leaves nothing
        behind on disk rather than accreting a false-valued key. ``_save`` runs
        only on a real transition, so a steady stream of replies to an already
        resumed thread costs no disk writes.
        """
        was_paused = False
        changed = False
        for candidate in self._pause_keys(key):
            entry = self._data.get(candidate)
            if entry is None:
                continue
            if entry.get("slack_paused") is True:
                was_paused = True
                if not paused:
                    del entry["slack_paused"]
                    changed = True
            elif paused:
                entry["slack_paused"] = True
                changed = True
        if changed:
            self._save()
        return was_paused

    def is_slack_paused(self, key: str) -> bool:
        """True when this session's outbound Slack mirroring is paused.

        The thread binding itself is untouched by a pause, so this is the ONLY
        signal that distinguishes a muted link from a live one. Inbound routing
        deliberately does not consult it: a reply must still reach the session
        that owns the thread, which is what makes reply-to-resume land in the
        original conversation instead of forking a new one.

        A live link is REQUIRED on the row carrying the flag. ``clear_slack_link``
        pops the flag, but ``set_slack_link(key, "", "")`` — how a Slack-side
        handoff releases the previous owner — empties the coordinates and leaves it
        behind. Without this check the flag outlives its link, and the next thread
        bound to that row is born muted: no echo, no reply, no tool stream, and the
        row snaps back to "Connect to Slack" the moment after the user connected
        it. Same invariant ``set_slack_paused`` states — a pause must not outlive
        the link it describes.
        """
        for candidate in self._pause_keys(key):
            entry = self._data.get(candidate)
            if (
                entry is not None
                and entry.get("slack_paused") is True
                and entry.get("slack_thread_ts")
            ):
                return True
        return False

    def get_session_for_thread(self, thread_ts: str) -> str | None:
        """Return the session key linked to a Slack thread_ts, or None."""
        return self._thread_to_session.get(thread_ts)

    # ── Channel-neutral outbound mirror binding (generalizes Slack linking) ──
    # ``set/get/clear_slack_link`` above are the Slack-specific backend of this
    # API: they own the dedicated ``slack_thread_ts`` / ``slack_channel_id``
    # fields and the ``_thread_to_session`` reverse index that powers Slack's
    # inbound leg. The trio below exposes the SAME binding channel-neutrally as
    # a ``ChannelLink`` so the dashboard turn path can deliver a reply to any
    # proactive-capable channel via ``Transport.send_message`` without
    # special-casing Slack. Slack routes back through the dedicated fields;
    # every other channel stores a ``ChannelLink`` under ``mirrors``.

    @staticmethod
    def _mirrors(entry: dict | None) -> dict[str, dict]:
        """Read an entry's non-Slack bindings as ``{channel_type: binding}``.

        One binding per channel type per session: a session may mirror to Discord
        AND Telegram at once, but never to two Discord conversations — which is
        the product rule (a conversation hosts one session, so a session holding
        two of the same channel could not be addressed unambiguously either).

        Read-compat, never migrating on read: an entry written before multi-bind
        carries a single ``mirror`` dict plus entry-level ``mirror_accepts_inbound``
        / ``mirror_paused`` flags. Those are folded into the same shape here, so
        every reader sees one format and no file is rewritten just by being read.
        """
        if not entry:
            return {}
        mirrors = entry.get("mirrors")
        if isinstance(mirrors, dict):
            return {k: v for k, v in mirrors.items() if isinstance(v, dict)}
        legacy = entry.get("mirror")
        if not isinstance(legacy, dict):
            return {}
        channel_type = str(legacy.get("channel_type") or "")
        if not channel_type:
            return {}
        folded = dict(legacy)
        if entry.get("mirror_accepts_inbound"):
            folded["accepts_inbound"] = True
        if entry.get("mirror_paused"):
            folded["paused"] = True
        return {channel_type: folded}

    def _write_mirrors(self, entry: dict, mirrors: dict[str, dict]) -> None:
        """Persist the binding map, retiring the legacy single-binding keys.

        Writing is the migration point: once a session's bindings are written in
        the new shape the legacy keys are dropped, so a row never carries both
        and no reader has to decide which one wins.
        """
        if mirrors:
            entry["mirrors"] = mirrors
        else:
            entry.pop("mirrors", None)
        entry.pop("mirror", None)
        entry.pop("mirror_accepts_inbound", None)
        entry.pop("mirror_paused", None)

    def set_mirror_link(
        self,
        key: str,
        link: ChannelLink | None,
        *,
        accepts_inbound: bool = False,
    ) -> None:
        """Bind (or clear, when *link* is None) a session's mirror target.

        Scoped to the link's OWN channel type: binding Discord leaves a Telegram
        binding on the same session untouched, and re-binding Discord replaces
        only the Discord one. Passing None clears every non-Slack binding, which
        is what the historical single-binding contract meant.

        ``accepts_inbound`` marks the binding as a session-resume target:
        messages arriving from that exact channel location may be routed back to
        *key*. Slack keeps its dedicated reverse index and ignores this flag.
        """
        if link is None:
            self.clear_mirror_link(key)
            return
        if link.channel_type == SLACK_NAMESPACE:
            self.set_slack_link(key, link.thread_id or "", link.channel_id)
            return
        key = canonical_key(key)
        # Bindings may live on the legacy `dashboard:`-spelled row. Read them from
        # wherever they actually are and write the whole set onto the canonical
        # row, then empty the legacy one — otherwise adding a SECOND channel makes
        # the canonical row win `_mirror_key` and the legacy row's binding vanishes
        # from delivery and from the UI while still occupying its location. Under
        # single-binding that supersession was correct (the new binding replaced
        # the old one); with several it would silently orphan a sibling.
        active = self._mirror_key(key)
        entry = self._ensure_entry(key)
        mirrors = self._mirrors(self._data.get(active))
        binding = link.to_dict()
        # A rebind replaces the binding, so it must not inherit the previous
        # one's mute: this writer is how a reconnect re-establishes a link, and a
        # stale paused flag would leave the new binding silently muted.
        if accepts_inbound:
            binding["accepts_inbound"] = True
        mirrors[link.channel_type] = binding
        self._write_mirrors(entry, mirrors)
        if active != key:
            legacy = self._data.get(active)
            if legacy is not None:
                # The bindings moved to the canonical row; leaving copies behind
                # would double-count this session in every by-location scan.
                self._write_mirrors(legacy, {})
        self._save()

    def _mirror_key(self, key: str) -> str:
        """The key a session's mirror binding is actually stored under.

        A channel conversation's binding belongs on its own session key — the key
        its dashboard turns run under. Bindings written before that unification
        sit on the sanitized ``dashboard:`` spelling
        (:func:`~kiro_crew.messaging.link.legacy_dashboard_mirror_key`); when
        that row holds the only binding it is still the live one, so reads and
        clears resolve to it instead of silently dropping a link the user set. A
        binding on the canonical key always wins, so a rebind supersedes the
        legacy row rather than being shadowed by it.
        """
        canon = canonical_key(key)
        entry = self._data.get(canon)
        if self._mirrors(entry):
            return canon
        if is_channel_session_key(canon):
            legacy = self._data.get(legacy_dashboard_mirror_key(canon))
            if self._mirrors(legacy):
                return legacy_dashboard_mirror_key(canon)
        return canon

    def set_mirror_paused(self, key: str, paused: bool, channel_type: str = "") -> bool:
        """Mute or unmute one binding, keeping it bound. Returns the PREVIOUS state.

        Per binding, not per session: muting Discord leaves a Telegram binding on
        the same session delivering. With no ``channel_type`` it applies to every
        non-Slack binding the session holds, and reports whether they were ALL
        already in that state — which keeps the historical single-binding
        contract (and its idempotent ``was_paused``) exactly as it was.

        It REFUSES to create an entry or a binding: a mute that outlives the link
        it describes would silently mute a future rebind, so a session with no
        matching binding is a no-op.
        """
        mkey = self._mirror_key(key)
        entry = self._data.get(mkey)
        mirrors = self._mirrors(entry)
        targets = [channel_type] if channel_type else list(mirrors)
        present = [t for t in targets if t in mirrors]
        if entry is None or not present:
            return False
        was_paused = all(mirrors[t].get("paused") is True for t in present)
        changed = False
        for target in present:
            binding = mirrors[target]
            if paused and binding.get("paused") is not True:
                binding["paused"] = True
                changed = True
            elif not paused and binding.pop("paused", None) is not None:
                changed = True
        if changed:
            self._write_mirrors(entry, mirrors)
            self._save()
        return was_paused

    def is_mirror_paused(self, key: str, channel_type: str = "") -> bool:
        """True when the named binding is muted (or, with no name, when EVERY
        non-Slack binding is).

        The binding itself is untouched by a mute, so this is the ONLY thing
        separating "muted" from "not linked" — every inbound resolver, the
        resume-conflict check and both clear paths must still see the link.

        The all-must-be-muted reading is what makes the outbound gate safe under
        multi-binding: one muted channel must never silence a sibling that is
        still connected.
        """
        mirrors = self._mirrors(self._data.get(self._mirror_key(key)))
        if channel_type:
            binding = mirrors.get(channel_type)
            return bool(binding and binding.get("paused"))
        return bool(mirrors) and all(b.get("paused") for b in mirrors.values())

    def get_mirror_links(self, key: str) -> list[ChannelLink]:
        """Every non-Slack binding this session holds, in channel-type order.

        The list form is what outbound delivery and the dashboard's link
        projection need: a session may mirror to several channels at once and
        each has to be resolved, governed and rendered on its own.
        """
        mirrors = self._mirrors(self._data.get(self._mirror_key(key)))
        links: list[ChannelLink] = []
        for channel_type in sorted(mirrors):
            raw = dict(mirrors[channel_type])
            raw.pop("accepts_inbound", None)
            raw.pop("paused", None)
            raw.setdefault("channel_type", channel_type)
            try:
                links.append(ChannelLink.from_dict(raw))
            except (TypeError, ValueError):
                continue
        return links

    def get_mirror_link(self, key: str, channel_type: str = "") -> ChannelLink | None:
        """One binding, by channel type — or the session's only one when unnamed.

        Kept single-valued for the many callers that already know which channel
        they mean (an in-channel ``!link``, an occupancy check, a reconnect for
        one row). With no ``channel_type`` it returns the sole binding, or None
        when the session holds several, so a caller that assumes one binding can
        never silently act on an arbitrary sibling.

        For a legacy Slack session carrying only ``slack_thread_ts`` /
        ``slack_channel_id`` it synthesizes the equivalent Slack ``ChannelLink``
        so callers never have to special-case Slack.
        """
        links = self.get_mirror_links(key)
        if channel_type:
            return next((link for link in links if link.channel_type == channel_type), None)
        if len(links) == 1:
            return links[0]
        if links:
            return None
        entry = self._data.get(self._mirror_key(key))
        if not entry:
            return None
        ts = entry.get("slack_thread_ts")
        ch = entry.get("slack_channel_id")
        if ts or ch:
            return ChannelLink(channel_type=SLACK_NAMESPACE, channel_id=ch, thread_id=ts)
        return None

    def mirror_accepts_inbound(self, key: str, channel_type: str = "") -> bool:
        """True iff a binding is a session-RESUME target (messages route back).

        The read counterpart of ``set_mirror_link(accepts_inbound=True)``.
        ``ChannelLink`` cannot carry the flag, so a caller that needs to tell a
        two-way resume from an outbound-only mirror asks here. With no
        ``channel_type`` it answers for ANY binding, matching the historical
        single-binding contract. Slack is excluded: it routes inbound through its
        own ``_thread_to_session`` index and never sets this marker.
        """
        # Resolves through `_mirror_key`, like every other binding reader: a
        # binding still held on the legacy `dashboard:`-spelled row would otherwise
        # read as outbound-only and lose its inbound affordance.
        mirrors = self._mirrors(self._data.get(self._mirror_key(key)))
        if channel_type:
            binding = mirrors.get(channel_type)
            return bool(binding and binding.get("accepts_inbound"))
        return any(b.get("accepts_inbound") for b in mirrors.values())

    def find_mirror_sessions(
        self,
        link: ChannelLink,
        *,
        inbound_only: bool = False,
    ) -> list[str]:
        """Return sessions bound to an exact non-Slack mirror location.

        The list form makes duplicate/corrupt bindings explicit so callers can
        fail closed rather than routing an inbound message to an arbitrary
        session. With ``inbound_only=True``, ordinary outbound-only dashboard
        mirrors are excluded.
        """
        matches: list[str] = []
        for key, entry in self._data.items():
            binding = self._mirrors(entry).get(link.channel_type)
            if not binding:
                continue
            if inbound_only and not binding.get("accepts_inbound"):
                continue
            raw = dict(binding)
            raw.pop("accepts_inbound", None)
            # A MUTED binding still occupies the location. Skipping it here would
            # make a muted conversation look free: a second session could claim
            # it, and the in-channel unlink the code tells the user to run would
            # report nothing to clear.
            raw.pop("paused", None)
            raw.setdefault("channel_type", link.channel_type)
            try:
                candidate = ChannelLink.from_dict(raw)
            except (TypeError, ValueError):
                continue
            if candidate == link:
                matches.append(key)
        return matches

    def clear_mirror_links_at(self, link: ChannelLink) -> list[str]:
        """Clear EVERY session whose mirror targets an exact non-Slack location.

        The write counterpart of :meth:`find_mirror_sessions`. An in-channel
        unlink means "nothing mirrors into this conversation anymore", and the
        bindings that occupy a location are matched by VALUE there — so a row
        stranded under a key spelling the conversation no longer uses (a rotated
        DM generation, a pre-unification ``dashboard:`` row) or a dashboard
        session mirroring into the conversation still blocks it while being
        unreachable by any key-addressed :meth:`clear_mirror_link`. Clearing by
        location closes that gap and doubles as the repair path for duplicate
        bindings, which the inbound resolver deliberately refuses to pick from.

        Returns the cleared session keys (empty when the location was free).
        Slack mirrors live in their own reverse index and are out of scope,
        exactly as in :meth:`find_mirror_sessions`.
        """
        cleared: list[str] = []
        for key in self.find_mirror_sessions(link):
            entry = self._data.get(key)
            if entry is None:  # pragma: no cover - keys come from _data itself
                continue
            mirrors = self._mirrors(entry)
            # Only the binding at THIS location: a session also mirroring to
            # Telegram keeps that one. Popping the binding takes its mute with it,
            # which is required — a mute describes a binding, and left behind it
            # would silently mute whatever is bound here next.
            mirrors.pop(link.channel_type, None)
            self._write_mirrors(entry, mirrors)
            cleared.append(key)
        if cleared:
            self._save()
        return cleared

    def clear_mirror_link(self, key: str, channel_type: str = "") -> bool:
        """Remove a session's binding(s); return True iff one existed.

        With a ``channel_type`` it removes only that binding, leaving siblings on
        other channels alone; naming Slack clears the Slack link so its reverse
        index is evicted too. With none it removes EVERY non-Slack binding (and
        falls back to the Slack link), which is what the historical single-binding
        contract meant. Resolves through :meth:`_mirror_key` so an unlink reaches a
        binding still held under the legacy spelling — otherwise a mirror that
        reads as live could not be turned off.
        """
        mkey = self._mirror_key(key)
        entry = self._data.get(mkey)
        if not entry:
            return False
        mirrors = self._mirrors(entry)
        if channel_type:
            if channel_type == SLACK_NAMESPACE:
                return self.clear_slack_link(mkey)
            removed = mirrors.pop(channel_type, None) is not None
        else:
            removed = bool(mirrors)
            mirrors.clear()
        if removed:
            # Popping the binding takes its mute with it: same reason as
            # clear_mirror_links_at, a mute must never outlive what it describes.
            self._write_mirrors(entry, mirrors)
            self._save()
            return True
        # The Slack fall-through is for the UNNAMED clear only, which means "every
        # binding". Naming a channel names the one you mean: if it holds nothing,
        # the answer is False, not "clear Slack instead" — otherwise an in-channel
        # `!unlink` on a channel with no binding would disconnect the session's
        # Slack thread, which nobody asked about.
        if not channel_type and (
            entry.get("slack_thread_ts") or entry.get("slack_channel_id")
        ):
            return self.clear_slack_link(mkey)
        return False

    def max_generation(self, bucket: str) -> int:
        """Return the highest persisted DM generation for a session *bucket*.

        The bucket is the generation-0 key (e.g.
        ``telegram:<agent>:direct:<user>``); generations persist as ``{bucket}``
        (gen 0) and ``{bucket}:gen{N}``. Returns the max ``N`` with a persisted
        entry, or -1 when the bucket has none. Channels seed their in-memory
        generation counter from this so ``/new`` and idle/daily reset advance
        past any generation left on disk (restart-safe) instead of colliding
        with a stale session and resuming it.
        """
        bucket = canonical_key(bucket)
        best = 0 if bucket in self._data else -1
        prefix = f"{bucket}:gen"
        for key in self._data:
            if key.startswith(prefix):
                suffix = key[len(prefix):]
                if suffix.isdigit():
                    best = max(best, int(suffix))
        return best

    def find_key_by_sid(self, session_id: str) -> str | None:
        """Find the session map key for a given kiro-cli session ID."""
        for k, entry in self._data.items():
            sid = entry.get("sid") if isinstance(entry, dict) else entry
            if sid == session_id:
                return k
        return None

    def channel_key_for_stem(self, stem: str) -> str:
        """The real channel session key whose transcript filename is *stem*.

        ``history._safe_key`` folds every ``:`` in a session key to ``_`` to
        build the filename, and that fold is NOT reversible: given
        ``discord_kirocrew_direct_123`` there is no way to tell which
        underscores were colons, and an agent name may legitimately contain
        one. This map holds the unfolded keys, so it is the only authority.

        Returns ``""`` when no mapped session matches, which callers must treat
        as "leave it unbound" rather than guessing — binding a tab to a
        wrongly-spelled key would answer the user from a session the channel
        never sees.
        """
        if not stem:
            return ""
        from kiro_crew.history import _safe_key

        for k in self._data:
            if is_channel_session_key(k) and _safe_key(k) == stem:
                return k
        return ""

    def get_link(self, key: str) -> ChannelLink | None:
        """Return the session's OWN inbound-channel link, or None.

        Distinct from the dashboard->Slack *mirror* binding, which stays
        behind ``get/set_slack_link`` (guardrail G3).
        """
        entry = self._data.get(canonical_key(key))
        if not entry:
            return None
        raw = entry.get("link")
        return ChannelLink.from_dict(raw) if raw else None

    def set_link(self, key: str, link: ChannelLink) -> None:
        """Set the session's OWN inbound-channel link. Creates entry if needed."""
        key = canonical_key(key)
        entry = self._data.get(key)
        if entry:
            entry["link"] = link.to_dict()
        else:
            self._data[key] = {
                "sid": "",
                "slack_thread_ts": None,
                "slack_channel_id": None,
                "link": link.to_dict(),
            }
        self._save()

    # --- v1c-B: per-conversation state on the session entry ---------------
    # Durable backing for per-thread state (``temporary``, ``incognito``,
    # ``agents``, ``projects``). Storing it on the session entry makes it
    # survive gateway restarts and ties its lifetime to the session (pruned
    # with the entry) rather than an ad-hoc bounded LRU in module-global dicts.

    def _ensure_entry(self, key: str) -> dict:
        """Return the entry for *key*, creating a blank one if absent."""
        entry = self._data.get(key)
        if entry is None:
            entry = {"sid": "", "slack_thread_ts": None, "slack_channel_id": None}
            self._data[key] = entry
        return entry

    def set_flag(self, key: str, flag: str, value: bool) -> None:
        """Set or clear a boolean per-conversation flag (e.g. ``temporary``).

        Flags are stored under an ``flags`` sub-dict on the entry. Clearing the
        last flag removes the sub-dict so empty state does not accrete on disk.
        Idempotent: writing an unchanged value still persists (cheap) so callers
        need not pre-check.
        """
        key = canonical_key(key)
        if value:
            entry = self._ensure_entry(key)
        else:
            # Clearing a flag on a key that was never stored is a no-op — don't
            # materialize a blank entry (would accrete empty state on disk).
            existing = self._data.get(key)
            if not existing:
                return
            entry = existing
        flags = entry.get("flags") or {}
        if value:
            flags[flag] = True
        else:
            flags.pop(flag, None)
        if flags:
            entry["flags"] = flags
        else:
            entry.pop("flags", None)
        self._save()

    def get_flag(self, key: str, flag: str) -> bool:
        """Return the value of a per-conversation boolean *flag* (default False)."""
        entry = self._data.get(canonical_key(key))
        if not entry:
            return False
        flags = entry.get("flags")
        return bool(flags and flags.get(flag))

    def set_agent_override(self, key: str, agent: str | None) -> None:
        """Set (or clear, when *agent* is falsy) the per-thread agent override."""
        key = canonical_key(key)
        if agent:
            self._ensure_entry(key)["agent_override"] = agent
        else:
            entry = self._data.get(key)
            if not entry or "agent_override" not in entry:
                return
            entry.pop("agent_override", None)
        self._save()

    def get_agent_override(self, key: str) -> str | None:
        """Return the per-thread agent override for *key*, or None."""
        entry = self._data.get(canonical_key(key))
        return entry.get("agent_override") if entry else None

    def set_project_override(self, key: str, project: str | None) -> None:
        """Set (or clear, when *project* is falsy) the per-thread project dir."""
        key = canonical_key(key)
        if project:
            self._ensure_entry(key)["project_override"] = project
        else:
            entry = self._data.get(key)
            if not entry or "project_override" not in entry:
                return
            entry.pop("project_override", None)
        self._save()

    def get_project_override(self, key: str) -> str | None:
        """Return the per-thread project-dir override for *key*, or None."""
        entry = self._data.get(canonical_key(key))
        return entry.get("project_override") if entry else None
