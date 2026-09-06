"""What a delegated run did DURING one ``delegate_wait`` window, and how it reaches
the human while the model is still waiting.

Extracted from ``ouroboros/tools/delegate.py`` for the same reason ``delegate_output``
was (size gate), and it is one coherent concern: the wait used to hand control back to
the model on the FIRST journal advance, so a healthy streaming run woke a full-context
nanny round every poll interval — measured, 18 rounds and 861k prompt tokens spent
narrating a run that was doing fine. The fix is the TIMER, not the stream: every
advance is still observed and still pushed to the live progress surface the instant it
is seen, but it is RECORDED here instead of returned, and the model is woken once, at
the window's expiry, carrying the whole sequence.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ouroboros.config import DELEGATE_WAIT_CEILING_SEC
from ouroboros.utils import truncate_review_artifact

log = logging.getLogger(__name__)

_TIMELINE_TAIL = 12
_TIMELINE_LABEL_CHARS = 300
_TEXT_KINDS = ("thinking", "message")
# The closing `note` is appended AFTER the advance list is sized, so its cost is
# reserved rather than measured; it is a fixed string of this module's own authorship.
_NOTE_RESERVE_CHARS = 700
# The floor the advance list ASKS for when the rest of the payload has eaten the budget:
# one advance plus its marker still tells the model the run is moving. It is a floor on
# the request, never on the result — a residual too small to hold it drops the list to
# its marker rather than shipping the floor over the limit (see `window_payload`).
_MIN_ADVANCE_BUDGET_CHARS = 400
# Re-measure passes for the fit loop: the first correction is exact for a uniform list,
# the rest absorb the per-row indent rounding. A payload still over after these is one
# whose harness-authored parts alone overflow, which only the truncator can answer.
_BUDGET_FIT_ATTEMPTS = 4


def _rows(detail: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [row for row in (detail.get("timeline") or []) if isinstance(row, dict)]


def _bounded(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The last few session events, each label bounded.

    The same delivery class as the terminal payload, one surface over: a harness-supplied
    title is unbounded, and twelve long ones push the progress payload past the tool-result
    cap, where head-truncation severs the JSON. Bounded through the shared disclosed
    contract (OMISSION NOTE + original length, with its anti-waste floor) rather than a
    hand-rolled slice+ellipsis — DEVELOPMENT.md forbids new hand-rolled sites (P34R.5).

    A DISPLAY tail: it drops rows off the head and says nothing about them, which is the
    right shape for the STANDING timeline (whose head is old news the model has seen) and
    the wrong shape on its own for a BATCH (whose head is news that arrived one poll ago).
    A caller sizing a batch with it owes the reader the count — see `record`.
    """
    def _label(value: Any) -> Any:
        return truncate_review_artifact(value, _TIMELINE_LABEL_CHARS) \
            if isinstance(value, str) else value

    out = []
    for row in rows[-_TIMELINE_TAIL:]:
        item = {"type": _label(row.get("type")), "title": _label(row.get("title")),
                "severity": _label(row.get("severity"))}
        for key in ("attemptId", "harnessId"):
            if isinstance(row.get(key), str) and row[key]:
                item[key] = _label(row[key])
        if row.get("textKind") in _TEXT_KINDS and isinstance(row.get("detail"), str):
            # The typed body preserves stream whitespace; the engine's title is
            # only a preview. Keep the same disclosed bound, without a second copy.
            item.update(title=_label(row["detail"]), textKind=row["textKind"],
                        textDelta=row.get("textDelta") is True)
        out.append(item)
    return out


def timeline_tail(detail: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _bounded(_rows(detail))


def _omission_marker(through_seq: int, count: int, *, whole: bool = False) -> Dict[str, Any]:
    """The row that stands where dropped advances were.

    ONE vocabulary for both places a drop can happen — the head shedding inside the list,
    and the whole list yielding when the rest of the payload leaves no room for even the
    floor — so a reader learns the same fact from the same keys either way.

    What the note may CLAIM is bounded by what is actually recoverable. It used to say
    every omitted advance "was streamed live and is in the event log", and both halves
    were wrong: a batch bigger than the display tail never reached the live line at all
    (the defect `record` now counts as `events_omitted`), and nothing on this side keeps
    the rows — Ouroboros persists no timeline anywhere, the daemon's own run directory
    does. So the note says where they really are.
    """
    return {
        "advances_omitted": count,
        "omitted_through_seq": through_seq,
        "note": ("every advance of this window is" if whole else
                 "older advances of this window are")
        + " omitted from THIS payload only; the rows live on the RUN, in Claudexor's own "
          "timeline — poll it again for the standing tail, nothing here keeps a copy",
    }


@dataclass
class _Advance:
    """One movement of the run's journal cursor, and what arrived with it."""

    seq: int
    at_sec: int
    events: List[Dict[str, Any]]
    # What the batch published BEYOND the display tail `record` sizes it with. Folded
    # into the same `events_omitted` the budget shedding writes, so a reader gets ONE
    # honest number per advance instead of two vocabularies for the same kind of cut.
    events_omitted: int = 0


@dataclass
class WindowObservations:
    """Every advance seen inside ONE wait window, in order."""

    advances: List[_Advance] = field(default_factory=list)
    # The last timeline this window actually saw, kept whole: the only thing that can
    # say which rows of the NEXT one are new. See `_fresh`.
    _prev: List[Dict[str, Any]] = field(default_factory=list)

    def observe_baseline(self, detail: Dict[str, Any], seq: int) -> None:
        """Adopt what was ALREADY on the timeline as history, not as this window's news.

        Without it the first real advance re-announced the whole standing tail as "new
        session events" — to the human's live stream and to the model alike — because the
        comparison in `_fresh` started from nothing while the run had been talking for a
        while.

        Only for a caller that is CAUGHT UP, though. `seq` is where the caller says it has
        read to, and one BEHIND the daemon's cursor (a `since_seq` under `lastSeq`) is
        asking for exactly the rows already standing there; adopting them would drop the
        very batch it called for. Rows carry no cursor of their own, so the choice is
        between the whole standing tail and none of it, and only a caught-up caller can
        be told nothing.
        """
        if int(seq or 0) >= int(detail.get("lastSeq") or 0):
            self._prev = _rows(detail)

    def record(self, detail: Dict[str, Any], seq: int, at_sec: int) -> _Advance:
        """Observe ONE journal advance, and return what arrived with it.

        The batch is still BOUNDED — an unbounded one is not free, a busy daemon can
        publish hundreds of rows between two three-second polls — but by a bound that
        COUNTS what it cut. It used to cut in silence, with the display tail written for
        the standing timeline: measured against a harness emitting sixteen rows per
        cursor step, the first four of every batch were gone from the live line AND from
        `advances`, with no `events_omitted` and no other key naming them. The count
        rides the advance to both surfaces (`rows`, `live_line`).
        """
        rows = _rows(detail)
        fresh = self._fresh(rows)
        advance = _Advance(seq=seq, at_sec=at_sec, events=_bounded(fresh),
                           events_omitted=max(0, len(fresh) - _TIMELINE_TAIL))
        if rows:
            self._prev = rows
        self.advances.append(advance)
        return advance

    def _fresh(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """The rows of this timeline that the previous observation did not already hold.

        Read off the DATA, because nothing the daemon reports says how big a batch is.
        Not the CURSOR: it moves by one per FOUR rows against a harness that emits four
        timeline rows per `lastSeq` step (reading the batch off it there dropped three of
        every four rows from the live stream AND from the record, silently, which is the
        one failure this path exists to prevent), and by MORE than the rows added when
        journal entries never become timeline rows (reading the batch off it there
        re-emitted rows already reported as new). Not the LENGTH either, once the
        timeline is bounded and rolling and its length stops growing.

        So the tails are compared: the longest overlap between the END of the previous
        one and the START of this one is what survived the roll, and everything past it
        is new. A pure append overlaps at the previous tail's whole length, which is the
        answer `rows[seen:]` gave while that held; a timeline replaced outright overlaps
        nowhere, and then all of it is new. Rows compare by value — identical rows are
        indistinguishable by construction, so a roll that shifts an unchanging row out
        and an identical one in reads as no news, which is also all it is.
        """
        if not rows:
            return []          # nothing observed; the last real tail stays the reference
        for overlap in range(min(len(self._prev), len(rows)), 0, -1):
            if self._prev[-overlap:] == rows[:overlap]:
                return rows[overlap:]
        return rows

    def rows(self, budget: int) -> List[Dict[str, Any]]:
        """The advance sequence, bounded to ``budget`` chars of JSON.

        A LIST, never a count: there is one row per observed advance carrying its `seq`
        and `at_sec`, and under pressure rows shed their event LABELS oldest-first with
        the shedding disclosed ON the row. Only if the bare spine still overflows does
        the HEAD shed, and it says so — a truncation that pretended the run started later
        than it did is the one shape forbidden here.

        `events_omitted` is ONE number per row and covers BOTH cuts a row can carry: the
        labels this payload shed for budget, and the rows the batch itself lost to the
        display tail when it was observed (`record`). They differ in what a reader can go
        find — a shed label reached the human's live line at observation time, a batch-cut
        row never did — and NEITHER is recoverable on this side: no timeline is persisted
        here, only the run's own timeline in Claudexor's run directory holds the rows. So
        the count is stated rather than the loss being waved away as immaterial.
        """
        out: List[Dict[str, Any]] = []
        for a in self.advances:
            row: Dict[str, Any] = {"seq": a.seq, "at_sec": a.at_sec, "events": list(a.events)}
            if a.events_omitted:
                row["events_omitted"] = a.events_omitted
            out.append(row)

        def _size(rows: List[Dict[str, Any]]) -> int:
            # MEASURED, never estimated, and rendered the way the caller renders it
            # (indent=2): an estimate that assumed a fixed cost per row overflowed the
            # tool-result limit on a long busy window, and the generic truncator then
            # cut the JSON mid-structure — the model got a payload it could not parse
            # at all, which is worse than any amount of honest shedding.
            return len(json.dumps(rows, ensure_ascii=False, indent=2))

        # Label shedding, oldest-first. The running total is kept by SUBTRACTING what
        # each row actually gave up (measured on that row, not assumed), and a real
        # whole-list measure confirms before returning — the arithmetic only decides
        # when to stop shedding, never what the payload is.
        running = _size(out)
        for row in out:
            if running <= budget and _size(out) <= budget:
                return out
            if not row["events"]:
                continue
            before = len(json.dumps(row, ensure_ascii=False, indent=2))
            # ACCUMULATED, never overwritten: a row whose batch was already cut at
            # observation would otherwise report only what THIS payload shed and quietly
            # forget the rest — two cuts, one of them invisible again.
            row["events_omitted"] = int(row.get("events_omitted") or 0) + len(row["events"])
            row["events"] = []
            running -= before - len(json.dumps(row, ensure_ascii=False, indent=2))
        if _size(out) <= budget:
            return out
        # The bare spine still overflows: shed from the HEAD. Shedding one row per
        # re-measure is O(n²) and cost ~7s of CPU at a full 1800s window (600 advances
        # at the 3s poll), so the drop count is ESTIMATED from the measured average row
        # and then corrected by measurement — the estimate only picks where to start,
        # the loop below still decides on the real rendered size.
        spine, kept = out, list(out)
        over = _size(kept) - budget
        if over > 0 and kept:
            per_row = max(1, _size(kept) // len(kept))
            guess = min(len(kept) - 1, max(1, over // per_row + 1))
            kept = spine[guess:]
        while kept and _size([_omission_marker(spine[len(spine) - len(kept) - 1]["seq"],
                                              len(spine) - len(kept))] + kept) > budget:
            kept = kept[1:]
        if len(kept) == len(spine):
            return out
        dropped = len(spine) - len(kept)
        if not kept:                      # even one row plus its marker overflows
            kept, dropped = spine[-1:], len(spine) - 1
        return [_omission_marker(spine[dropped - 1]["seq"], dropped)] + kept


# Typed external-wait lease (poltergeist phase B, B3). While `delegate_wait`
# holds its bounded window over a live delegated run, the supervisor's IDLE rail
# is spared — and ONLY the idle rail: the explicit deadline, the absolute
# ceiling, every budget fence, and cancel are untouched. The lease is pure
# metadata with a hard expiry (no background thread holds anything): the worker
# grants it for one window, bounded by min(task deadline, the run's own
# maxSeconds horizon, this absolute ceiling), and releases it when the wait
# returns. The ceiling exceeds the delegate_wait ToolEntry timeout (2100s, which
# DELEGATE_WAIT_CEILING_SEC equals) by +300 headroom so a full window plus its
# own teardown can never be idle-killed mid-hold.
DELEGATE_WAIT_LEASE_GRACE_SEC = 120
EXTERNAL_WAIT_LEASE_CEILING_SEC = DELEGATE_WAIT_CEILING_SEC + 300


def external_wait_lease_until(ctx: Any, window: int,
                              run_started_at: Any, run_max_seconds: int) -> float:
    """When the idle-rail lease for ONE wait window must expire (epoch seconds).

    Bounded by construction, never open-ended: the window itself plus a teardown
    grace, clamped under the task's own explicit deadline, under the run's own
    ``maxSeconds`` horizon (a run past its cap is not a healthy wait target), and
    under the absolute lease ceiling. Unknown facts simply do not narrow — they
    never widen.
    """
    import time as _time

    from ouroboros.deadline_utils import parse_deadline_ts

    now = _time.time()
    grace = float(DELEGATE_WAIT_LEASE_GRACE_SEC)
    candidates = [now + float(window) + grace,
                  now + float(EXTERNAL_WAIT_LEASE_CEILING_SEC)]
    meta = getattr(ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    deadline = parse_deadline_ts(meta.get("deadline_at"))
    if deadline is not None:
        candidates.append(deadline.timestamp())
    if run_started_at is not None and run_max_seconds:
        candidates.append(run_started_at.timestamp() + float(run_max_seconds) + grace)
    return min(candidates)


def emit_external_wait_lease(ctx: Any, run_id: str, until_ts: float,
                             lease_id: str = "") -> None:
    """Grant (``until_ts`` in the future) or release (``0.0``) the idle-rail lease.

    A typed supervisor event, not synthetic progress: the idle enforcer keeps
    judging REAL progress, and this lease only tells it that a bounded host-side
    hold over a delegated run is a legitimate silence. It spares nothing else —
    deadline, absolute ceiling, budget fences, and cancel all cut through it.
    Best-effort by design: a lost lease costs at worst one idle-timeout episode,
    never correctness, and no wait may fail because its lease could not be spoken.

    ``lease_id`` is the grant's identity (F5b): each wait window mints one, and
    its release names the same id, so the supervisor can refuse a release from
    an abandoned, timed-out wait thread that would otherwise blank a NEWER
    grant made by the task's next wait.
    """
    q = getattr(ctx, "event_queue", None)
    if q is None:
        return
    try:
        q.put_nowait({
            "type": "external_wait_lease",
            "task_id": str(getattr(ctx, "task_id", "") or ""),
            "run_id": str(run_id or ""),
            "until_ts": float(until_ts),
            "lease_id": str(lease_id or ""),
        })
    except Exception:
        log.debug("external-wait lease emission failed", exc_info=True)


def poll_bound(seconds_left: float, *, strict: bool = False) -> float:
    """What one call inside a clamped window may ASK the transport for.

    NARROWING in both directions, which is the only thing a bound is for. Never more
    than the window has left — that is the whole point — and never more than the read
    default the client would have used anyway: floored alone, the arithmetic RAISED the
    ask above that default whenever the window had over a minute left (measured: 1797.0,
    1000.0, 61.0), so a hung daemon stopped failing at sixty seconds and held the whole
    window instead, to be reported afterwards as a wait that simply saw nothing. And
    never less than the floor a bound is useful at: a nearly spent window that asked for
    its own 0.2s would turn every daemon into a timeout, and the honest answer there is
    one short read, then expiry.

    ONE place computes it, for both entry points below, so the two cannot drift.
    """
    from ouroboros.gateways.claudexor import _READ_TIMEOUT_SEC, SHORT_POLL_TIMEOUT_SEC

    if strict:
        return min(_READ_TIMEOUT_SEC, max(0.001, float(seconds_left)))
    return min(_READ_TIMEOUT_SEC, max(SHORT_POLL_TIMEOUT_SEC, float(seconds_left)))


def _strict_poll(gateway: Any, run_id: str, timeout: float) -> Dict[str, Any]:
    """Give each HTTP phase the full remainder, while bounding total wall time."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    holder: Dict[str, Any] = {}
    done = threading.Event()

    def call() -> None:
        try:
            holder["detail"] = gateway.get_run(run_id, timeout_sec=timeout)
        except BaseException as exc:
            holder["error"] = exc
        finally:
            done.set()

    threading.Thread(target=call, name=f"review-poll-{run_id}", daemon=True).start()
    if not done.wait(timeout=timeout):
        raise ClaudexorUnavailable(
            "poll_wall_timeout", f"Claudexor poll exceeded {timeout:g}s wall-clock bound",
        )
    if "error" in holder:
        raise holder["error"]
    return holder.get("detail") or {}


def is_transient_git_object_race(exc: Exception) -> bool:
    """Recognize ONLY Git's disappearing atomic-object scratch path.

    The engine's run reader loses a race against Git renaming ``.git/objects/…/
    tmp_obj_*`` into place and answers one poll with an ENOENT naming that scratch
    path — a fact about a file that existed a moment ago and exists again a moment
    later. The CI platform gate learned to retry exactly this shape (938094a9) while
    the production poll kept propagating it, so CI could pass on an engine whose
    live ``delegate_wait`` still failed. The matcher is deliberately this narrow:
    any other ENOENT, or the same errno on any other path, stays a real failure.
    """
    detail = str(exc).replace("\\", "/").lower()
    code = str(getattr(exc, "code", "") or "").upper()
    return (
        (code == "ENOENT" or "enoent" in detail)
        and "/.git/objects/" in detail
        and "/tmp_obj_" in detail
    )


def bounded_poll(
    gateway: Any, run_id: str, seconds_left: float, *, strict: bool = False,
) -> Dict[str, Any]:
    """One poll that may not ask for longer than the window has left.

    The client's read default is minutes-scale, which is right for a call with all the
    time it needs and wrong for one inside a clamped window: a poll started a moment
    before expiry could answer a minute after it, so the clamp bounded the sleeping and
    not the waiting. Strict review polls add a total wall-clock guard; ordinary
    delegate waits retain their intentional short-poll floor and phase-local timeout.

    A failure PROPAGATES, as the typed refusal the gateway already raised. There is time
    left on the window here, so there is no expiry to report and nothing true to say
    about a daemon that has stopped answering: swallowing it handed the model a
    completed, uneventful wait of a duration nobody waited (measured — a
    ``daemon_unreachable`` three seconds into a 1800s window came back as
    ``status: progress``, ``waited_sec: 1800``, "The run advanced 1 time(s) during the
    1800s you asked to wait", after 3.0s of wall; before any advance, a 503 came back as
    ``no_progress`` over a full window with a note inviting a `delegate_cancel` of a live
    run because the transport had blipped). Only the LAST poll may expire instead.

    One exception to the propagation rule, matched as narrowly as it occurs: the
    engine's transient Git atomic-object race (see ``is_transient_git_object_race``)
    gets ONE immediate re-read while the window still has time — it is a lie about a
    file mid-rename, not a daemon that stopped answering.
    """
    deadline = time.monotonic() + max(0.0, float(seconds_left))

    def poll(remaining: float) -> Dict[str, Any]:
        bound = poll_bound(remaining, strict=strict)
        return _strict_poll(gateway, run_id, bound) if strict else gateway.get_run(run_id, timeout_sec=bound)

    try:
        return poll(seconds_left)
    except Exception as exc:
        remaining = max(0.0, deadline - time.monotonic())
        if remaining > 0 and is_transient_git_object_race(exc):
            log.debug("transient Git object race on poll of %s; one re-read", run_id)
            return poll(remaining)
        raise


def expiring_poll(
    gateway: Any, run_id: str, *, strict: bool = False,
) -> Optional[Dict[str, Any]]:
    """The poll of a SPENT window. ``None`` when the daemon did not answer inside it.

    A spent window still takes its poll rather than skipping it. The ordinary path uses
    the short-poll floor; strict owner-deadline callers use their 1 ms total wall bound.
    Skipping looked like the cheap trade and cost more, because terminal state and containment
    breach were then judged on data read BEFORE the last sleep — a run that finished
    during that sleep came back as ``progress``/``running`` with no settlement, and the
    model paid another full-context round for a run already done.

    This is the ONE poll whose silence is not a failure, and the reason it is a separate
    entry point: the window is already over, so a daemon too slow to answer inside the
    bound is this window's EXPIRY, which the caller knows how to report, and never a
    transport failure raised out of a tool holding a live overpowered run.
    """
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    try:
        return bounded_poll(gateway, run_id, 0.0, strict=strict)
    except ClaudexorUnavailable:
        log.debug("the last poll of a spent delegate_wait window went unanswered", exc_info=True)
        return None


def rendered_window(**kwargs: Any) -> str:
    """``window_payload`` rendered the way the wait returns it (and measures it)."""
    return json.dumps(window_payload(**kwargs), ensure_ascii=False, indent=2)


def waiting_expiry_clause(pending: Optional[List[Dict[str, Any]]]) -> str:
    """The honest expiry clause for a waiting note, keyed on the rows' own
    ``timeout_at`` (R2-7e). The engine benign-declines an unanswered question
    only when the row carries a ``timeout_at``; a null one means NO automatic
    expiry, and a note promising a benign decline there teaches the nanny to
    wait for a timeout that never comes. Claimed only when EVERY visible row
    carries one; rows absent entirely (a bare ``waitingOnUser`` flag) prove no
    expiry either."""
    rows = [row for row in (pending or []) if isinstance(row, dict)]
    if rows and all(str(row.get("timeout_at") or "").strip() for row in rows):
        return ("the engine timeout benign-declines it (each row's timeout_at; "
                "the run then continues on stated assumptions)")
    return ("timeout_at is null here, so there is NO automatic expiry — the "
            "run waits until answered")


def _fitted_pending(payload: Dict[str, Any], pending: List[Dict[str, Any]],
                    budget: int) -> None:
    """Install a MEASURED pending-interactions projection onto ``payload`` (F2).

    The same shed discipline as the immediate ``waiting_on_user`` payload: rows
    shed from the TAIL with the cut COUNTED (``interactions_omitted``), measured
    on the rendered payload rather than estimated — two probes put a
    bounded-per-field projection at 24 459 and 51 719 chars against a 15 000
    budget, and the generic truncator then cut the JSON mid-structure. When even
    ONE row cannot fit, the rows yield entirely to the counted marker plus a
    pointer at the recoverable copy: the full set was already returned by the
    immediate ``waiting_on_user`` payload (staged whole to the task drive with a
    sha256 receipt when it spilled), and the run itself still holds it — an
    unparseable payload recovers nothing.
    """
    if not pending:
        return
    rows = list(pending)
    while rows:
        payload["pending_interactions"] = rows
        omitted = len(pending) - len(rows)
        if omitted:
            payload["interactions_omitted"] = omitted
        else:
            payload.pop("interactions_omitted", None)
        if len(json.dumps(payload, ensure_ascii=False, indent=2)) <= budget:
            return
        rows = rows[:-1]
    payload.pop("pending_interactions", None)
    payload["interactions_omitted"] = len(pending)
    payload["interactions_note"] = (
        "the pending question set could not fit this payload even one row at a "
        "time; the run is still PAUSED on it — the full set was delivered by the "
        "immediate waiting_on_user return (staged to the task drive when it "
        "spilled), or re-read it with delegate_wait"
    )


def live_line(run_id: str, advance: _Advance) -> str:
    parts: List[str] = []
    previous_stream = None
    for row in advance.events:
        text_kind = row.get("textKind")
        is_text = text_kind in _TEXT_KINDS and isinstance(row.get("title"), str)
        text = row["title"] if is_text else str(row.get("title") or row.get("type") or "")
        if not is_text:
            actor = "/".join(str(row[key]) for key in ("harnessId", "attemptId") if row.get(key))
            if actor:
                text = f"[{actor}] {text}"
        stream = (text_kind, row.get("attemptId"), row.get("harnessId")) \
            if is_text and row.get("textDelta") is True else None
        if not text:
            if stream != previous_stream:
                previous_stream = None
            continue
        # Native fragments may split a word or contain only whitespace. Only the
        # engine's explicit delta fact authorizes concatenation, never the prose.
        if parts and stream is not None and stream == previous_stream:
            parts[-1] += text
        else:
            parts.append(text)
        previous_stream = stream
    titles = " · ".join(parts)
    # A batch larger than the display tail shows its LAST rows. Saying how many it is not
    # showing is the difference between an honest subset and a line that reads like the
    # whole batch — the human's stream is the one surface that had no marker at all. It
    # says where they are, not that they are somewhere: these rows never reached this
    # line, and nothing on this side stores them — the run's own timeline does.
    more = (f" (+{advance.events_omitted} earlier in this batch, on the run's timeline)"
            if advance.events_omitted else "")
    return f"🛰 delegated run {run_id} @seq {advance.seq}: {titles or '(new session events)'}{more}"


def emit(ctx: Any, run_id: str, advance: _Advance) -> None:
    """Push one advance to the LIVE progress surface, from inside the wait.

    `ctx.emit_progress_fn` is immediate (it puts the frame straight on the event queue),
    unlike `ctx.pending_events`, which is drained only when the task ends — a whole
    window's narration held until the nanny's turn was over is the collapse this fix
    exists to avoid. The frame is also what stamps the supervisor's `last_progress_at`,
    which a silently blocking wait would otherwise starve. A broken progress channel
    must never abort a wait that is holding a live overpowered run.
    """
    fn = getattr(ctx, "emit_progress_fn", None)
    if not callable(fn):
        return
    try:
        fn(live_line(run_id, advance))
    except Exception:
        log.debug("delegated progress emit failed", exc_info=True)


def window_payload(
    *,
    run_id: str,
    state: str,
    last_seq: int,
    window: int,
    elapsed_seconds: Optional[int],
    max_seconds: Optional[int],
    waiting_on_user: bool,
    detail: Dict[str, Any],
    seen: WindowObservations,
    budget: int,
    pending_interactions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """What the model is handed when the window expires without a terminal state.

    ``pending_interactions`` is the caller's BOUNDED question projection (B4): a
    known question the model chose not to answer keeps riding every expiry payload
    beside the ``waiting_on_user`` boolean, so a nanny that escalated up the hierarchy
    is re-shown what the run is paused on instead of a bare flag. MEASURED into
    the payload in BOTH branches (F2) — through ``_fitted_pending``'s shed
    discipline, before the advance list is sized — so it can never push the
    result past the budget: per-field bounds alone left a many-question set at
    24k/51k chars against a 15k budget, exactly on the early ``no_progress``
    return that used to skip measurement entirely.
    """
    payload: Dict[str, Any] = {
        "status": "progress" if seen.advances else "no_progress",
        "run_id": run_id,
        "state": state,
        "last_seq": last_seq,
        "waited_sec": window,
        "elapsed_seconds": elapsed_seconds,
        "max_seconds": max_seconds,
        "waiting_on_user": waiting_on_user,
    }
    if not seen.advances:
        payload["reason"] = "non_terminal_and_no_new_session_events_within_wait_window"
        # A run PAUSED on its own question is not "stuck" (owner 7=A / F13): the
        # generic delegate_cancel hint invited cancelling a run that is simply
        # waiting to be answered. The waiting state gets its own note.
        payload["note"] = (
            ("The run is alive and PAUSED on the question(s) it already asked "
             "(waiting_on_user; see pending_interactions). Decide: answer with "
             "delegate_answer, escalate an above-authority question with the "
             "escalate verb (parent-first; the reply reaches your mailbox on a "
             "later round), or "
             f"keep waiting (call again) — {waiting_expiry_clause(pending_interactions)}. "
             "Do not cancel a run merely because it asked a question.")
            if waiting_on_user else
            ("The run is alive but silent. Decide: keep waiting (call again), "
             "or delegate_cancel if it is stuck."))
        _fitted_pending(payload, list(pending_interactions or []), budget)
        return payload
    payload["timeline_tail"] = timeline_tail(detail)
    # The pending questions are measured in FIRST, against a budget that reserves
    # the closing note and the advance list's floor marker, so an oversized
    # question set can never push the rendered result past the limit (F2) — the
    # advance-fit loop below then sizes itself against what is actually left.
    _fitted_pending(payload, list(pending_interactions or []),
                    budget - _NOTE_RESERVE_CHARS - _MIN_ADVANCE_BUDGET_CHARS)
    # The advance list gets what is LEFT, measured — not a fixed share. A fixed share
    # bounded only itself: `timeline_tail` carries harness-authored text (three fields,
    # each capped at 300 chars, twelve rows) and can reach most of the limit on its own,
    # so the whole result overflowed and the generic truncator cut the JSON mid-structure
    # — the very failure this list is bounded to avoid, and one the base did not have.
    # `note` is added after this line, so its own cost is reserved here too. The list is
    # then sized against the WHOLE rendered payload rather than against itself: nested a
    # level deeper it costs more than it measures standalone (two spaces of indent per
    # line, plus its key), so a sub-block that fits its share can still overflow the
    # result. The loop re-measures what is actually shipped and hands back the overflow.
    _room = max(_MIN_ADVANCE_BUDGET_CHARS,
                budget - len(json.dumps(payload, ensure_ascii=False, indent=2))
                - _NOTE_RESERVE_CHARS)
    for _ in range(_BUDGET_FIT_ATTEMPTS):
        payload["advances"] = seen.rows(_room)
        _over = len(json.dumps(payload, ensure_ascii=False, indent=2)) + _NOTE_RESERVE_CHARS - budget
        if _over <= 0:
            break
        if _room <= _MIN_ADVANCE_BUDGET_CHARS:
            # The residual cannot hold even the FLOOR, and the floor is what overflowed:
            # measured, twelve tail rows of three 20 000-char harness fields leave ~400
            # chars once the `note` is reserved, and a floor-sized list shipped 422 on
            # top — a 15 018-char result against a 15 000 limit, which the generic
            # truncator then cut mid-structure. That is the round-1 defect from the other
            # end, and it is this module's own floor that caused it, not the harness text.
            # So the floor is an aspiration and the MEASUREMENT is the rule: the list
            # yields entirely rather than the payload crossing the limit, and it says so
            # where it stood, in the same vocabulary a head-shed uses.
            payload["advances"] = [_omission_marker(
                seen.advances[-1].seq, len(seen.advances), whole=True)]
            break
        _room = max(_MIN_ADVANCE_BUDGET_CHARS, _room - _over)
    payload["quiet_for_sec"] = max(0, window - seen.advances[-1].at_sec)
    payload["note"] = (
        f"The run advanced {len(seen.advances)} time(s) during the {window}s you asked to "
        "wait; each advance was offered to your human's live stream as it happened "
        "(delivery is best-effort) — this call held the window instead of waking you "
        "per event. `advances` is that sequence. "
        "Call delegate_wait again with since_seq=last_seq to keep watching; "
        "`quiet_for_sec` is how long it has been silent at the end of the window."
    )
    if waiting_on_user:
        # F13: the paused state earns its own instruction here too — never the
        # generic keep-watching line alone while a question sits unanswered.
        # The expiry claim is keyed on the rows' own timeout_at (R2-7e).
        payload["note"] += (
            " The run is PAUSED on a question: answer it (delegate_answer), "
            "raise it with the escalate verb (parent-first) if it is above "
            f"your authority, or keep waiting — "
            f"{waiting_expiry_clause(pending_interactions)}; do not cancel over it."
        )
    return payload
