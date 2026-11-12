"""What a delegated run's journal and timeline say, and how it reaches the human.

Extracted from ``ouroboros/tools/delegate.py`` (size gate). The body follows each held run's
speech (``delegate_feed``) and publishes it here as typed records; an engine without the run's
journal keeps the timeline window view (``WindowObservations``, ``emit``). A ``delegate_wait``
wake reads its run's detail once (``tools/delegate._read_run``) and renders its present state
(``present_payload``); every read the host makes of a run is bounded (``bounded_poll``).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ouroboros.utils import truncate_review_artifact

log = logging.getLogger(__name__)

_TIMELINE_TAIL = 12
_TIMELINE_LABEL_CHARS = 300
_TEXT_KINDS = ("thinking", "message")


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
        # ``messageId``/``outcome`` ride the engine's ``message.*`` rows: the
        # receipts a recovered nanny reconciles a delegate_message against (A26).
        for key in ("attemptId", "harnessId", "messageId", "outcome"):
            if isinstance(row.get(key), str) and row[key]:
                item[key] = _label(row[key])
        # The engine's opaque utterance key is compared whole, never cut.
        for key in ("speechKey", "speechOp"):
            if isinstance(row.get(key), str) and row[key]:
                item[key] = row[key]
        if row.get("textKind") in _TEXT_KINDS and isinstance(row.get("detail"), str):
            # The typed body preserves stream whitespace; the engine's title is
            # only a preview. Keep the same disclosed bound, without a second copy.
            item.update(title=_label(row["detail"]), textKind=row["textKind"],
                        textDelta=row.get("textDelta") is True)
            if row.get("textFinal") is True:
                item["textFinal"] = True
        out.append(item)
    return out


def timeline_tail(detail: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _bounded(_rows(detail))


@dataclass
class _Advance:
    """One movement of the run's journal cursor, and what arrived with it."""

    seq: int
    events: List[Dict[str, Any]]
    # What the batch published BEYOND the display tail `record` sizes it with, so a
    # reader is told how many rows the bounded batch does not show.
    events_omitted: int = 0


@dataclass
class WindowObservations:
    """What one run's timeline added between the details read of it (the window view of an
    engine without the run's journal, ``delegate_feed.Feed.window``)."""

    # The last timeline actually seen, kept whole: the only thing that can say which rows
    # of the NEXT one are new. See `_fresh`.
    _prev: List[Dict[str, Any]] = field(default_factory=list)

    def observe_baseline(self, detail: Dict[str, Any], seq: int) -> None:
        """Adopt what was ALREADY on the timeline as history, not as news.

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

    def record(self, detail: Dict[str, Any], seq: int) -> _Advance:
        """Observe ONE journal advance, and return what arrived with it.

        The batch is BOUNDED — a busy daemon can publish hundreds of rows between two
        reads — by a bound that COUNTS what it cut (``events_omitted``, shown by
        ``delegate_activity.progress_text``): measured against a harness emitting sixteen rows per cursor step,
        a silent cut lost the first four of every batch from the live line.
        """
        rows = _rows(detail)
        fresh = self._fresh(rows)
        advance = _Advance(seq=seq, events=_bounded(fresh), events_omitted=max(0, len(fresh) - _TIMELINE_TAIL))
        if rows:
            self._prev = rows
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


def poll_bound(seconds_left: float, *, strict: bool = False) -> float:
    """What one call with ``seconds_left`` of its caller's time may ASK the transport for.

    NARROWING in both directions, which is the only thing a bound is for. Never more
    than the caller has left, and never more than the read default the client would have
    used anyway: floored alone, the arithmetic RAISED the ask above that default whenever
    over a minute was left (measured: 1797.0, 1000.0, 61.0), so a hung daemon stopped
    failing at sixty seconds. And never less than the floor a bound is useful at: a
    nearly spent remainder that asked for its own 0.2s would turn every daemon into a
    timeout. ONE place computes it, for every caller, so they cannot drift.
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
    """One poll that may not ask for longer than its caller has left (``poll_bound``).

    The client's read default is minutes-scale, which is wrong for a read bounded by a
    deadline: started a moment before it, the read could answer a minute after it.
    Strict review polls add a total wall-clock guard; ordinary reads keep the short-poll
    floor and the phase-local timeout.

    A failure PROPAGATES, as the typed refusal the gateway already raised: swallowing it
    once handed the model a completed, uneventful wait nobody waited. One exception,
    matched as narrowly as it occurs: the engine's transient Git atomic-object race
    (``is_transient_git_object_race``) gets ONE immediate re-read while time is left —
    it is a lie about a file mid-rename, not a daemon that stopped answering.
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


def waiting_expiry_clause(pending: Optional[List[Dict[str, Any]]], async_only: bool = False) -> str:
    """The honest expiry clause for a waiting note, keyed on the rows' own
    ``timeout_at`` (R2-7e). The engine benign-declines an unanswered question
    only when the row carries a ``timeout_at``; a null one means NO automatic
    expiry, and a note promising a benign decline there teaches the nanny to
    wait for a timeout that never comes. Claimed only when EVERY visible row
    carries one; rows absent entirely (a bare ``waitingOnUser`` flag) prove no
    expiry either. ``async_only`` is the caller's whole-set fact: such a question
    closes at its timeout with nothing sent and resumes nothing, since nothing
    was held. Otherwise only the rows that may hold the run speak for the run."""
    rows = [row for row in (pending or []) if isinstance(row, dict)]
    if not async_only:
        rows = [row for row in rows if row.get("delivery") != "async"]
    if not rows or any("timeout_at" not in row for row in rows):
        return "expiry details are unavailable in this observation; no deadline or absence of one is established"
    if async_only:
        if rows and all(str(row.get("timeout_at") or "").strip() for row in rows):
            return ("at each row's timeout_at an unanswered async question closes and nothing "
                    "is sent; the agent keeps working without an answer")
        return ("timeout_at is null here, so the question stays open while the agent keeps "
                "working; silence answers nothing")
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
        + ("time; the questions are still pending and the run keeps working"
           if payload.get("async_questions_only") is True else
           "time; the run is still PAUSED on it")
        + " — the full set was delivered by the "
        "immediate waiting_on_user return (staged to the task drive when it "
        "spilled), or re-read it with delegate_wait"
    )


def executor_observation(
    ctx: Any, run_id: str, advance: _Advance, detail: Dict[str, Any], entry: Any,
) -> Dict[str, Any]:
    """Latest typed actor in this owned run snapshot, never a liveness verdict.

    Read the original timeline, not its truncated prose/display tail. The
    summary route's observedModel belongs to a final attempt, so it cannot
    identify a later live/reviewer attempt. Current live detail supplies no
    per-attempt observed model; the requested model stays labelled as such.
    """
    from ouroboros.delegate_custody_usage import names_another_run
    from ouroboros.subagent_messages import executor_observation_meta

    task_id = str(getattr(ctx, "task_id", "") or "")
    if (getattr(entry, "task_id", None) != task_id
            or getattr(entry, "run_id", None) != run_id
            or names_another_run(detail, run_id)
            or detail.get("lastSeq") != advance.seq):
        return {}
    actor = next((row for row in reversed(_rows(detail))
                  if all(isinstance(row.get(key), str) and row[key]
                         for key in ("harnessId", "attemptId", "type"))), None)
    if actor is None:
        return {}
    attempt = getattr(ctx, "task_attempt", None)
    observation = {
        "task_id": task_id, "task_attempt": str(attempt) if attempt is not None else "",
        "run_id": run_id, "attempt_id": actor["attemptId"],
        "harness_id": actor["harnessId"], "phase": actor["type"], "revision": advance.seq,
    }
    if getattr(entry, "model", "") and getattr(entry, "route_id", None) == actor["harnessId"]:
        observation.update(model=entry.model, model_source="requested")
    return executor_observation_meta(observation, task_id=task_id, task_attempt=attempt)


def _hand_over(ctx: Any, frames: Any, what: str, meta: Any = dict) -> None:
    """Hand each frame of ``frames`` (a ``delegate_activity`` reader, not started yet) to the task's progress
    channel with ``meta()``, committing it only once handed over. Without a channel nothing is read, so retained
    frames stay pending for a delivery that has one (the body's feed always binds it). A broken channel never
    aborts the caller or advances the cursor."""
    from ouroboros import delegate_activity

    fn = getattr(ctx, "emit_progress_fn", None)
    if not callable(fn):
        return
    try:
        metadata = meta()
        for activity, commit in frames:
            fn(delegate_activity.progress_text(activity), **metadata, delegated_activity=activity)
            commit()
    except Exception:
        log.debug("delegated %s emit failed", what, exc_info=True)


def emit(ctx: Any, run_id: str, advance: _Advance, *,
         detail: Dict[str, Any], entry: Any, after_seq: int) -> None:
    """Publish the body's window advance as typed activity and its latest executor fact.

    ``Feed.window`` already selected the fallback for an engine without a readable journal.
    The activity owner replays unhanded frames before the window.
    """
    from ouroboros import delegate_activity

    def observed() -> Dict[str, Any]:
        observation = executor_observation(ctx, run_id, advance, detail, entry)
        return {"executor_observation": observation} if observation else {}
    _hand_over(ctx, delegate_activity.window_observations(ctx, run_id, advance, after_seq=after_seq), "window",
               observed)


def emit_stream(ctx: Any, run_id: str, read: Any, *, after_seq: int) -> None:
    """Push the events of one journal read the body made (``delegate_feed``) to the live progress surface."""
    from ouroboros import delegate_activity

    _hand_over(ctx, delegate_activity.stream_observations(ctx, run_id, read, after=after_seq), "stream")


def emit_retained(ctx: Any, run_id: str) -> None:
    """Retry local activity delivery without delaying a ready result on transport."""
    from ouroboros import delegate_activity

    _hand_over(ctx, delegate_activity.retained_observations(ctx, run_id), "retained")


def paused_note(pending_interactions: Optional[List[Dict[str, Any]]], async_only: bool = False) -> str:
    """The note of a run PAUSED on its own question(s): never "stuck" (owner 7=A /
    F13). Shared by the per-window payload and the wake, which keeps it when it
    drops the read's own keep-watching/cancel note. ``async_only`` (the whole set's
    fact, judged before any preview) names a run that keeps working instead."""
    if async_only:
        return ("The run is alive and KEEPS WORKING; the question(s) it asked are attention "
                "only and hold nothing (async_questions_only; see pending_interactions). "
                "Answer with delegate_answer when an answer would help — it reaches the "
                "agent as its next user message in this session —, escalate an "
                "above-authority question with the escalate verb (parent-first), or keep "
                f"waiting (call again) — {waiting_expiry_clause(pending_interactions, True)}. "
                "Do not cancel a run merely because it asked a question.")
    return ("The run is alive and PAUSED on the question(s) it already asked "
            "(waiting_on_user; see pending_interactions). Decide: answer with "
            "delegate_answer, escalate an above-authority question with the "
            "escalate verb (parent-first; the reply reaches your mailbox on a "
            "later round), or "
            f"keep waiting (call again) — {waiting_expiry_clause(pending_interactions)}. "
            "Do not cancel a run merely because it asked a question.")


def message_journal_receipt(ctx: Any, result: Any) -> Any:
    """Attach peer journal rows at the ordinary receipt boundary, never a wake.

    Receipts may repeat rows by message_id. No read cursor or ACK advances
    here, so losing a receipt before transcript insertion loses no message.
    The existing wake renderer retains oversized receipts before projecting.
    """
    from ouroboros.delegate_message_store import journal
    from ouroboros.delegate_shared import delegate_payload, delegate_result
    from ouroboros.tool_capabilities import tool_result_limit

    payload = delegate_payload(result)
    run_id = str(payload.get("run_id") or "")
    if not run_id or payload.get("message_id") or payload.get("ok") is False:
        return result
    rows = []
    try:
        rows = journal(ctx, run_id)
    except Exception:
        payload["peer_messages_delivery"] = {"complete": False, "reason": "journal_unavailable"}
    else:
        if not rows:
            return result
        payload["peer_messages"] = rows
    if len(json.dumps(payload, ensure_ascii=False, indent=2)) > tool_result_limit("delegate_wait"):
        from ouroboros.delegate_supervision import _render_wake_payload

        payload = delegate_payload(_render_wake_payload(ctx, payload))
        if "wake_delivery" in payload and rows:
            payload["wake_delivery"]["note"] = (
                f"Full receipt and {len(rows)} peer messages at source. Read before relying on omitted detail.")
    return delegate_result(payload, meta=result.meta)


# What this task's own ending does to an open run (``delegate_custody_reconcile._owner_terminal_is_deliberate`` and
# the maintenance sweep's spared handles), said where the run's state is read, before its owner chooses to end
# (audit 11.11, item 18: a fact at its birth, not a return at the finish door).
OWNER_END = ("If this task ends while this run is open (finish, stop, a failure it declares, a cancel, or a wrap-up at "
             "a deadline, budget or round limit), the next custody pass cancels the run (owner_task_gone); a provider "
             "or worker death of this task, or a task kept waiting for its human's answer, spares it.")


def present_payload(
    *,
    run_id: str,
    state: str,
    last_seq: int,
    moved: bool,
    elapsed_seconds: Optional[int],
    max_seconds: Optional[int],
    waiting_on_user: bool,
    detail: Dict[str, Any],
    budget: int,
    pending_interactions: Optional[List[Dict[str, Any]]] = None,
    async_questions_only: Optional[bool] = None,
) -> Dict[str, Any]:
    """A live run's present state, as the one read of a ``delegate_wait`` wake found it.

    ``moved``: its journal went past the wake's cursor, so the run's timeline tail rides
    along (``progress``); otherwise ``no_progress``. A question the model already saw keeps
    riding beside ``waiting_on_user`` (``pending_interactions``, the caller's BOUNDED
    projection), so a nanny that escalated it is re-shown what the run is paused on. It is
    MEASURED into the payload (``_fitted_pending``): per-field bounds alone left a
    many-question set at 24k/51k chars against a 15k budget.

    ``async_questions_only`` is the fact about the WHOLE pending set; omitted, it is judged on
    the full ``detail`` (``claudexor_observation.question_attention``), never on
    ``pending_interactions``, a bounded preview that may shed the only blocking row. Present,
    the run keeps working: its questions hold nothing (the wake's note says so, ``paused_note``).
    """
    payload: Dict[str, Any] = {
        "status": "progress" if moved else "no_progress",
        "run_id": run_id,
        "state": state,
        "last_seq": last_seq,
        "elapsed_seconds": elapsed_seconds,
        "max_seconds": max_seconds,
        "waiting_on_user": waiting_on_user,
        "owner_end": OWNER_END,
    }
    if async_questions_only is None:
        from ouroboros.gateways.claudexor_observation import question_attention

        async_questions_only = waiting_on_user and question_attention(detail)[1]
    if waiting_on_user and async_questions_only:
        payload["async_questions_only"] = True  # absent keeps the old meaning: a question holds the run
    if waiting_on_user:
        # The flat route fact of the immediate waiting payload: an answer resumes THIS session.
        from ouroboros.delegate_interactions import SAME_SESSION_CONTINUATION

        payload["continuation"] = SAME_SESSION_CONTINUATION["continuation"]
    if moved:
        payload["timeline_tail"] = timeline_tail(detail)
    else:
        payload["reason"] = "non_terminal_and_no_new_session_events_since_cursor"
    _fitted_pending(payload, list(pending_interactions or []), budget)
    return payload
