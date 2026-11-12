"""Explicit continuation of a stopped session reviewer seat (owner, 2 November 2026).

A stopped seat continues only when its author names it: ``continue_seats`` beside the exact
``review_reference`` (Plan: beside nothing, the ask is found by the operation) lists
``{seat_id, operation_id}`` as the result's ``continuable_seats`` offer recorded them. The
choice is ONE compared transition of the existing review ledger record: a new operation of
the same seat in the same paid wave, whose start body is the seat's ORIGINAL retained request
with only its prompt, its remaining original clock and (one-shot runs) ``continueFrom``
replaced. Repeating a choice rejoins its recorded successor without a revision; a new choice
after a definite unsent refusal names the refused operation and continues its target run. A
choice is recorded only once the chosen operation's worker in this process has finished
publishing its settlement (``_await_publication``): the successor shares its custody key.

The START_REQUESTED row, the writable checking copy's holder, the attempt row or Plan
participant and this process's replay caches are idempotent projections of that decision
(``project_decisions``), repaired by the author's explicit collections (the same choice, a
plain reference collection, an identical reissue of the round, the result delivery's
collection). A failure after the choice is ``ContinuationUnsent``: recorded, nothing sent,
and the repeat repairs that very choice. A START records the instant the RECORDED window closes,
narrowed by the owner deadline and task window when the projection read it (``generation_end_at``),
asks the engine for no more than what then remains, and is written by the call that then carries
it into its roster row and sends it; collectors send only what a roster row carries, under its
recorded token, and an UNKNOWN send replays the same token and bytes. Settling unsent and carrying
are compare-and-swap transitions of that same row (``Roster.transition``): a successor no row
carried settles unsent once its window has closed, also when it closes before its START is
written, and a START some earlier call wrote but never carried is retired as a definite zero-send
and offered again, so a new explicit choice starts it within what remains instead of a later send
on stale bytes. A successor START is new generation: before it is decided or written it rides the
known-spend admission every review wave rides, and the executor's send callback admits its POST,
a replay included, on that same admission and on the recorded instant, then stops it generating
there (``review_session_custody``). Passive readers never project or send. The transport creates
no successor of its own. A read-only seat whose wave released its isolated checkout is refused before the choice
unless the record can rebuild it; its START row carries the exact mapping (``RESTORE_KEY``) from which a read-only
checkout, released or still present, is put back and re-owned before the START is written and again before every
POST, a crash replay included (#2796).
"""
from __future__ import annotations

import copy
import hashlib
import math
import pathlib
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

PREFIX = "review_continue_"
WINDOW_SPENT = "continuation_window_spent"
START_UNSENT = "continuation_start_unsent"
_UNSENT_CAUSES = {
    WINDOW_SPENT: "the original review window ended before this continuation was sent",
    START_UNSENT: "its start was recorded by a call that never sent it; choose this seat again to start it "
                  "within what remains of the original window",
}
# How often the publication wait rereads custody's in-memory live entries; it bounds nothing (``_await_publication``).
PUBLICATION_POLL_SEC = 0.05
_CLEARED_SEAT_KEYS = ("outcome", "outcome_id", "usage_ref", "failure_code", "reset_at", "http_status",
                      "transport_status", "reported_cause", "failure_details")
_CLEARED_ROW_KEYS = (*_CLEARED_SEAT_KEYS, "raw_text", "text", "error", "response_ref", "delegated_run_id",
                     "delegated_run_started", "provider_status_code", "physical_attempt_state", "coverage",
                     "parsed_items", "answers", "verdict", "semantic_verdict", "parse_status", "review_thread_id")
CONTINUE_SEATS_SCHEMA = {
    "type": "array", "minItems": 1,
    "items": {"type": "object", "additionalProperties": False, "required": ["seat_id", "operation_id"],
              "properties": {"seat_id": {"type": "string"}, "operation_id": {"type": "string"}}},
    "description": (
        "Continue stopped session reviewers of the referenced review, each named exactly as its continuable_seats "
        "offer gives it. Each choice starts one successor run of the same seat in the same wave from the seat's "
        "original request (model, effort, account choice, access, output contract) within the original "
        "remaining time and current limits; repeating it rejoins that successor. Running, unknown or answered "
        "seats are refused at $0; a fresh review stays a separate act."),
}


class ContinuationRefused(ValueError):
    """A typed $0 refusal of a continuation choice; ``offers`` is the record's current offer."""

    recorded: List[Dict[str, Any]] = []

    def __init__(self, reason: str, message: str, offers: Optional[List[Dict[str, Any]]] = None) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason, self.offers = reason, list(offers or [])

    def consequence(self) -> str:
        return "Nothing was decided or sent; the review is unchanged."

    def payload(self) -> Dict[str, Any]:
        return {"status": "refused", "reason": self.reason, "message": str(self), "consequence": self.consequence(),
                "continuable_seats": self.offers}


class ContinuationUnsent(ContinuationRefused):
    """The choice IS recorded (``recorded``: each seat's successor operation) but a projection after it failed,
    so nothing was sent; repeating the same call repairs that very choice (its operation, token and body), or,
    for a seat whose start was already recorded, settles it unsent and offers it again."""

    def __init__(self, reason: str, message: str, recorded: List[Dict[str, Any]]) -> None:
        super().__init__(reason, message)
        self.recorded = [dict(item) for item in recorded]

    def consequence(self) -> str:
        chosen = ", ".join(f"seat {item['seat_id']} as {item['operation_id']}" for item in self.recorded)
        return (f"The choice is recorded ({chosen}), but nothing was sent. Repeat the same call to send that same "
                "choice; nothing is decided again. A seat whose start was already recorded settles as not sent "
                "instead and is offered again, so a new choice starts it within the original window.")

    def payload(self) -> Dict[str, Any]:
        return {**super().payload(), "status": "decided_unsent", "recorded": self.recorded}


@dataclass
class Roster:
    """Where the chosen seats' recorded rows live: an attempt row, or a Plan ask's participants.

    ``transition(seat_id, decision, build)`` is the one compare-and-swap of a seat's row, under the roster's own
    lock: a row naming ``decision["supersedes"]`` becomes ``build(row)``; a row already naming the successor
    stays (another call carried it or settled it unsent first); any other row refuses. It returns the row now
    in place. Collectors send only what such a row carries, so this is where sending and settling unsent meet."""

    task_id: str
    retry_key: str
    rows: Dict[str, Tuple[str, Dict[str, Any]]]  # seat_id -> (surface, recorded row)
    transition: Callable[[str, Dict[str, Any], Callable[[Dict[str, Any]], Dict[str, Any]]], Dict[str, Any]]
    carried: Callable[[str], Dict[str, Any]]  # seat_id -> its row read now
    root_task_id: str = ""
    review_contract: str = ""


@dataclass
class _Choice:
    seat_id: str
    named: str
    rejoin: Optional[Dict[str, Any]] = None
    entry: Dict[str, Any] = field(default_factory=dict)
    body: Dict[str, Any] = field(default_factory=dict)
    prompt_payload: Dict[str, Any] = field(default_factory=dict)


def successor_operation_id(record_id: str, seat_id: str, named: str) -> str:
    digest = hashlib.sha256(f"{record_id}\0{seat_id}\0{named}".encode("utf-8")).hexdigest()[:24]
    return PREFIX + digest


def parse_selections(value: Any) -> List[Tuple[str, str]]:
    if not isinstance(value, list) or not value:
        raise ContinuationRefused("invalid_selection", "continue_seats is a non-empty list of {seat_id, operation_id}")
    chosen: List[Tuple[str, str]] = []
    for item in value:
        if (not isinstance(item, dict) or set(item) != {"seat_id", "operation_id"}
                or not all(isinstance(item[key], str) and item[key].strip() for key in item)):
            raise ContinuationRefused("invalid_selection", "each choice is exactly {seat_id, operation_id} from the offer")
        if any(seat == item["seat_id"].strip() for seat, _ in chosen):
            raise ContinuationRefused("invalid_selection", f"seat {item['seat_id']} is named twice")
        chosen.append((item["seat_id"].strip(), item["operation_id"].strip()))
    return chosen


def _session(seat: Dict[str, Any]) -> bool:
    from ouroboros.review_execution import ReviewRouteKind

    return str((seat.get("requested") or {}).get("route") or "") == ReviewRouteKind.AGENT_SESSION.value


def _last_decision(seat: Dict[str, Any]) -> Dict[str, Any]:
    decisions = [entry for entry in seat.get("continuations") or [] if isinstance(entry, dict)]
    return decisions[-1] if decisions and decisions[-1].get("operation_id") == seat.get("operation_id") else {}


def _offer(seat: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One seat's offer from recorded facts only; ``None`` when nothing explicit can continue it."""
    from ouroboros.review_ledger import ANSWERED_STATUSES, seat_open

    operation, decision = str(seat.get("operation_id") or ""), _last_decision(seat)
    state, status = str(seat.get("operation_state") or ""), str(seat.get("status") or "")
    if (not operation or not _session(seat) or seat_open(seat) or status in ANSWERED_STATUSES
            or state == "custody_lost" or (state == "not_dispatched" or status == "not_dispatched") and not decision
            or seat.get("failure_code") == WINDOW_SPENT):
        return None
    usage_ref = (seat.get("outcome") or {}).get("usage_ref") or seat.get("usage_ref") or {}
    own_run = str(usage_ref.get("delegated_run_id") or "")
    resumable = (seat.get("failure_details") or {}).get("resumable")
    if (not isinstance(resumable, dict) or not resumable) and decision and not own_run:
        resumable = decision.get("stop")  # a continuation refused before any run keeps its target's stop
    if not isinstance(resumable, dict) or not resumable:
        return None
    usd = [seat.get("usd"), *(prior.get("usd") for prior in seat.get("prior_operations") or [])]
    return {"seat_id": str(seat.get("seat_id") or ""), "operation_id": operation, "run_id": own_run,
            "continues_run": own_run or str(decision.get("continues_run") or ""),
            "cause": str(seat.get("reported_cause") or seat.get("failure_code") or ""),
            "engine_stop": {key: resumable.get(key) for key in ("cause", "resetsAt", "carriers", "limitWindow")
                            if resumable.get(key) is not None},
            "cost_so_far": {"usd": round(sum(float(v) for v in usd if isinstance(v, (int, float))), 6),
                            "unknown": any(not isinstance(v, (int, float)) for v in usd)}}


def continuable_seats(record: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [offer for seat in (record or {}).get("rows") or [] if isinstance(seat, dict) and (offer := _offer(seat))]


def offers_for(ctx: Any, record_id: Any) -> List[Dict[str, Any]]:
    """The offer of one ledger record by id, for a surface that holds only its id; unreadable offers nothing."""
    from ouroboros.review_ledger import ledger_root, load_record

    try:
        return continuable_seats(load_record(ledger_root(ctx), str(record_id))) if record_id else []
    except ValueError:
        return []


def _checking_home(custody: Any, task_id: str, operation: str) -> Tuple[Dict[str, Any], Any, str]:
    """``(operation marker, the home that physically keeps its copy, "thread"|"held"|"own")``; ``{}`` = no copy."""
    import json

    from ouroboros.review_checking import seat_dir

    try:
        marker = json.loads((seat_dir(custody, task_id, operation) / "checking.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, None, ""
    seat, held = str(marker.get("thread_seat") or ""), str(marker.get("held_home") or "")
    kind = "thread" if seat else "held" if held else "own"
    return marker, seat_dir(custody, task_id, f"thread:{seat}" if seat else held or operation), kind


def _holder(physical: Dict[str, Any], kind: str) -> str:
    return str(physical.get("operation_id") if kind == "thread" else physical.get("holder_operation_id")
               or physical.get("operation_id") or "")


def checking_copy_holder(custody: Any, task_id: str, operation: str) -> Tuple[bool, str]:
    """Whether ``operation`` still holds a writable checking copy its explicit successor can take over."""
    import json

    from ouroboros.review_checking import _checking_area_present
    from ouroboros.unindexed import unindexed

    try:
        marker, home, kind = _checking_home(custody, task_id, operation)
        if not marker:
            return False, "no checking copy was recorded for this operation"
        physical = marker if kind == "own" else json.loads((home / "checking.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"checking marker unreadable: {exc}"
    if _holder(physical, kind) != operation:
        return False, f"the copy belongs to operation {_holder(physical, kind)}"
    if not _checking_area_present(unindexed(home, "area", physical.get("area"))):
        return False, "the copy was already released; its saved receipt is evidence, not a restored copy"
    return True, ""


def transfer_checking_copy(custody: Any, task_id: str, source: str, successor: str) -> None:
    """Hand the stopped operation's writable copy to its successor at the SAME path (idempotent).

    Under the copy's home lock: the successor's marker is written first, then the holder flips, so a
    crash leaves the old holder in charge. A thread seat's home moves to the successor turn without
    rematerializing; a one-shot home records ``holder_operation_id`` and the successor names ``held_home``."""
    import json

    from ouroboros.review_checking import CheckingAreaUnavailable, _checking_area_present, _seat_locked, seat_dir
    from ouroboros.unindexed import unindexed
    from ouroboros.utils import atomic_write_json, utc_now_iso

    marker, home, kind = _checking_home(custody, task_id, source)
    if not marker:
        return  # a read-only seat has no copy
    target = seat_dir(custody, task_id, successor) / "checking.json"
    with _seat_locked(home):
        physical = marker if kind == "own" else json.loads((home / "checking.json").read_text(encoding="utf-8"))
        holder, area = _holder(physical, kind), unindexed(home, "area", physical.get("area"))
        if holder == successor and target.exists():
            return
        if holder not in (source, successor) or not _checking_area_present(area):
            raise CheckingAreaUnavailable(f"the checking copy of {source} is gone or held by {holder}")
        held_home = "" if kind == "thread" else str(marker.get("held_home") or source)
        own = {**physical, "operation_id": successor, "created_at": utc_now_iso()} if kind == "thread" else {
            "schema": 1, "created_at": utc_now_iso(), "task_id": task_id, "operation_id": successor,
            "source": physical.get("source") or {}, "area": str(area), "held_home": held_home,
            **({"git": physical["git"]} if physical.get("git") else {})}
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target, {key: value for key, value in own.items() if key != "capture_id"})
        if kind == "thread":
            atomic_write_json(home / "checking.json", {**physical, "operation_id": successor})
        else:
            atomic_write_json(home / "checking.json", {**{k: v for k, v in physical.items() if k != "held_for_continuation"},
                                                       "holder_operation_id": successor})


def _window_end(custody: Any, invocation: Dict[str, Any]) -> str:
    """When the seat's ORIGINAL window ends: its first run's start plus its cap; "" = sent with the clock off."""
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.delegate_custody import run_timing

    started, cap = run_timing(custody, str(invocation.get("run_id") or ""), window=True) if invocation.get("run_id") else ("", 0)
    body = invocation.get("request") or {}
    cap = int(cap or invocation.get("max_seconds") or body.get("maxSeconds") or 0)
    if cap <= 0:
        return ""
    start = parse_deadline_ts(started or invocation.get("requested_at"))
    if start is None:
        raise ContinuationRefused("window_unknown", "the original review window has no recorded start")
    return (start + timedelta(seconds=cap)).isoformat()


def _window_closes(ctx: Any, window_end: str) -> Optional[datetime]:
    """The instant the original window closes for this call: ``window_end`` narrowed by the current owner deadline
    (less the finalization reserve) and task execution window, read now; ``None`` = none. An instant, so the steps
    after this read take time from it instead of adding to it; an unreadable recorded end is a closed window."""
    from ouroboros import deadline_utils as clock
    from ouroboros.config import get_finalization_grace_sec

    meta, now = getattr(ctx, "task_metadata", None), clock.utc_now()
    ends = [clock.parse_deadline_ts(window_end) or now] if window_end else []
    owner = clock.dispatch_window_remaining_sec(
        deadline_at=(meta or {}).get("deadline_at") if isinstance(meta, dict) else None,
        deadline_ts=getattr(ctx, "deadline_ts", None), reserve_sec=get_finalization_grace_sec())
    ends += [now + timedelta(seconds=value) for value in (owner, clock.task_execution_window_sec()) if value is not None]
    return min(ends) if ends else None


def _seconds_left(closes: Optional[datetime]) -> Optional[int]:
    """Whole seconds before ``closes`` (``_window_closes``): ``<= 0`` once it has closed; ``None`` = no window."""
    from ouroboros.deadline_utils import utc_now

    return None if closes is None else int(math.floor((closes - utc_now()).total_seconds()))


def _remaining_seconds(ctx: Any, window_end: str) -> Optional[int]:
    return _seconds_left(_window_closes(ctx, window_end))


def continuation_prompt(seat_id: str, run_id: str, cause: str, stop: Dict[str, Any]) -> str:
    """Host facts only: which run stopped, the engine's own words, and that the assignment is unchanged."""
    why = "; ".join(part for part in (cause, f"engine stop cause: {stop.get('cause')}" if stop.get("cause") else "") if part)
    return (f"Your previous run{f' {run_id}' if run_id else ''} in review seat {seat_id} stopped before it finished"
            f"{f' ({why})' if why else ''}. Ouroboros chose to continue it: the same seat of the same review wave, "
            "with the same frozen subject, assignment and output contract. Continue the remaining work and give "
            "your complete final answer under the original output contract.")


def _invocation(custody: Any, slot_id: str, operation: str, row: Dict[str, Any]) -> Dict[str, Any]:
    from ouroboros import delegate_invocation
    from ouroboros.review_dispatch import recorded_invocation_tokens

    tokens = recorded_invocation_tokens(row)
    if len(tokens) == 1:
        token = next(iter(tokens))
        found = delegate_invocation.invocation_record(custody, token)
        return {**found, "invocation_id": token} if found else {}
    found = delegate_invocation.operation_record(custody, operation, slot_id)
    return dict(found or {})


def _stopped_run(custody: Any, seat: Dict[str, Any], invocation: Dict[str, Any]) -> str:
    """The run the operation last stopped in: its own, or the head of a legacy automatic chain under it."""
    from ouroboros.review_session_custody import review_continuation_descends_from

    own = str(invocation.get("run_id") or "")
    head = str(((seat.get("outcome") or {}).get("usage_ref") or {}).get("delegated_run_id") or "")
    if head and head != own:
        if not review_continuation_descends_from(custody, head, invocation):
            raise ContinuationRefused("run_lineage_unproven", f"run {head} is not proven to continue {own}")
        return head
    return own


RESTORE_KEY = "review_subject_restore"  # the START row's exact mapping of what a released checkout held (#2796)


def _prepare(ctx: Any, custody: Any, record: Dict[str, Any], roster: Roster, seat_id: str, named: str) -> _Choice:
    """Validate one choice and build its successor from retained sources; writes nothing."""
    from ouroboros.delegate_invocation import invocation_record
    from ouroboros.review_execution import _CLAUDEXOR_MAX_SECONDS
    from ouroboros.review_operation import recorded_review_assignment
    from ouroboros.deadline_utils import bounded_seconds

    offers = continuable_seats(record)

    def refuse(reason: str, message: str) -> ContinuationRefused:
        return ContinuationRefused(reason, message, offers)

    seat = next((row for row in record.get("rows") or [] if str(row.get("seat_id") or "") == seat_id), None)
    if seat is None:
        raise refuse("unknown_seat", f"the review has no seat {seat_id}")
    if not _session(seat):
        raise refuse("not_a_session_route", f"seat {seat_id} ran no engine session that could be continued")
    rejoin = next((entry for entry in seat.get("continuations") or [] if entry.get("supersedes") == named), None)
    if rejoin is not None:
        return _Choice(seat_id, named, rejoin=dict(rejoin))
    if str(seat.get("operation_id") or "") != named:
        raise refuse("not_current", f"seat {seat_id}'s current operation is {seat.get('operation_id')}, not {named}")
    offer = next((item for item in offers if item["seat_id"] == seat_id), None)
    if offer is None:
        raise refuse("not_continuable", f"seat {seat_id} is running, unknown, answered or reported no continuable work")
    if _fenced(ctx, roster.task_id):
        raise refuse("stopped_or_paused", "the owning task is stopped or entering a pause; nothing was decided")
    surface, row = roster.rows.get(seat_id) or ("", {})
    if str(row.get("operation_id") or "") != named:  # also a compacted attempt whose roster is gone
        raise refuse("roster_unavailable", f"the recorded roster row of seat {seat_id} is not operation {named}")
    try:
        request, slot = recorded_review_assignment(
            custody, task_id=roster.task_id, surface=surface, row=row, retry_key=roster.retry_key,
            root_task_id=roster.root_task_id, review_contract=roster.review_contract)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise refuse("assignment_unavailable", f"the seat's original assignment is unreadable: {exc}") from exc
    invocation, decision = _invocation(custody, seat_id, named, row), _last_decision(seat)
    own_run = ""
    if decision and invocation.get("state") != "started":  # refused or never sent before any run: its target stays
        target, stop = str(decision.get("continues_run") or ""), dict(decision.get("stop") or {})
        origin, window_end = str(decision.get("origin_invocation_id") or ""), str(decision.get("window_end") or "")
    else:
        if invocation.get("state") != "started":
            raise refuse("no_stopped_run", f"operation {named} has no recorded run to continue")
        target = own_run = _stopped_run(custody, seat, invocation)
        stop = dict((seat.get("failure_details") or {}).get("resumable") or {})
        origin = str(decision.get("origin_invocation_id") or invocation.get("invocation_id") or "")
        window_end = str(decision["window_end"]) if decision else _window_end(custody, invocation)
    original = (invocation_record(custody, origin) or {}).get("request") if origin else None
    if not isinstance(original, dict) or not isinstance(original.get("prompt"), str) or "continueFrom" in original:
        raise refuse("original_request_unavailable", "the seat's original start body is not retained")
    thread = str(original.get("_thread_id") or "")
    if surface == "plan_review" and not thread:
        raise refuse("thread_unavailable", "the plan seat's original thread is not recorded")
    seconds = _remaining_seconds(ctx, window_end)
    if seconds is not None and seconds <= 0:
        raise refuse("window_spent", "no original review time remains within the current deadline; start a fresh review instead")
    from ouroboros.tools.review_subject import readonly_scope_root

    if str(original.get("access") or "readonly") in {"workspace_write", "full"}:
        held, why = checking_copy_holder(custody, roster.task_id, named)
        if not held:
            raise refuse("checking_progress_unavailable", f"the reviewer's writable checking copy is not held: {why}")
    elif (read := readonly_scope_root(original)) and not pathlib.Path(read).exists():  # put back before each send
        from ouroboros.tools.review_subject import released_subject

        try:
            released_subject(custody, record, read, invocation_record(custody, origin) or {}, str(ctx.task_id))
        except ValueError as exc:
            raise refuse("subject_unavailable", f"the read-only reviewer's checkout was released and its reviewed "
                                                f"subject cannot be materialized again: {exc}") from exc
    successor = successor_operation_id(str(record.get("record_id") or ""), seat_id, named)
    from ouroboros.delegate_custody import idempotency_key

    prompt = continuation_prompt(seat_id, target, offer["cause"], stop)
    body = {**copy.deepcopy(original), "prompt": prompt}
    body.pop("maxSeconds", None)
    if seconds is not None:
        body["maxSeconds"] = bounded_seconds(seconds, default=1, maximum=_CLAUDEXOR_MAX_SECONDS)
    if not thread:  # a thread turn continues its thread; a run names the stopped run
        body["continueFrom"] = target
    entry = {"operation_id": successor, "supersedes": named, "continues_run": target, "predecessor_run": own_run,
             "thread_id": thread,
             "invocation_id": idempotency_key("review_continue_decision", record.get("record_id"), seat_id, successor),
             "origin_invocation_id": origin, "predecessor_invocation_id": str(invocation.get("invocation_id") or ""),
             "surface": surface, "max_seconds": body.get("maxSeconds"), "window_end": window_end, "stop": stop,
             "decided_by_task": str(getattr(ctx, "task_id", "") or "")}
    from ouroboros.review_dispatch import review_operation_binding

    prompt_payload = {"request": asdict(request), "slot": asdict(slot), "session_prompt": prompt,
                      "binding": review_operation_binding(request, slot, successor)}
    return _Choice(seat_id, named, entry=entry, body=body, prompt_payload=prompt_payload)


def _admit_new_generation(ctx: Any, starts: List[Tuple[str, str, str, Any, int]], offers: List[Dict[str, Any]]) -> None:
    """A successor START is new generation, so before one is decided or written it rides the known-spend admission
    every review wave rides (``review_wave_budget_gate``: the global, root and original-group fences, fail-open on
    an unknown); a fence already reached refuses at $0. ``starts``: (seat, surface, model, max tokens, prompt chars).
    A recorded start is admitted again where it is POSTed, a replay included (``continuation_send_admitted``);
    collecting a known run is never admitted."""
    from ouroboros.review_substrate import review_usage_category
    from ouroboros.tools.review_helpers import review_wave_binding_fence, review_wave_budget_gate

    if not starts:
        return
    admission = review_wave_budget_gate(
        ctx, surface=starts[0][1], models=[model for _seat, _surface, model, _tokens, _chars in starts],
        prompt_chars=[chars for *_rest, chars in starts],
        max_completion_tokens=[int(tokens or 65536) for _seat, _surface, _model, tokens, _chars in starts],
        categories=[review_usage_category(surface) for _seat, surface, *_rest in starts],
        slot_ids=[seat for seat, *_rest in starts], extra={"continuation": True})
    if admission is not None:
        fence, remedy = review_wave_binding_fence(admission)
        raise ContinuationRefused("budget_insufficient", f"known spend has reached the {fence}; no successor was "
                                  f"started. To continue, {remedy}.", offers)


def _retain_sources(custody: Any, task_id: str, choices: List[_Choice]) -> None:
    """Immutable inputs first: the successor body (content-addressed) and the clock-free ``{op}_prompt``."""
    from ouroboros.observability import persist_call, read_call_payload, write_blob

    for choice in choices:
        choice.entry["request_ref"] = write_blob(custody, choice.body, kind="json")
        call_id, payload = f"{choice.entry['operation_id']}_prompt", choice.prompt_payload
        try:
            read_call_payload(custody, task_id=task_id or "review", call_id=call_id)
        except FileNotFoundError:
            persist_call(custody, task_id=task_id or "review", call_id=call_id,
                         call_type=f"{payload['request'].get('call_type') or choice.entry['surface'] + '_review'}_prompt",
                         payload={key: payload[key] for key in ("request", "slot", "session_prompt")},
                         manifest={"surface": choice.entry["surface"], "slot_id": choice.seat_id,
                                   "model": payload["slot"].get("model", ""), "review_operation_binding": payload["binding"]})


def _reopen(seat: Dict[str, Any], entry: Dict[str, Any]) -> None:
    """Supersede the stopped operation in place: its facts, sources and money stay in the seat's history."""
    from ouroboros.review_ledger import ANSWER_NOT_ASKED, _blank_answer, _merge_evidence, answer_keys

    sources = [ref for ref in seat.get("source_refs") or [] if isinstance(ref, dict)]
    seat["prior_operations"] = [*(seat.get("prior_operations") or []), {
        "operation_id": seat.get("operation_id"), "status": seat.get("status"), "usd": seat.get("usd"),
        "operation_state": seat.get("operation_state"), "run_id": entry.get("predecessor_run", ""),
        **{key: seat[key] for key in _CLEARED_SEAT_KEYS if seat.get(key) not in (None, "", {}, [])}}]
    seat["superseded_operation_ids"] = [*(seat.get("superseded_operation_ids") or []), entry["supersedes"]]
    seat["continuations"] = [*(seat.get("continuations") or []), entry]
    seat["historical_source_refs"] = _merge_evidence(seat.get("historical_source_refs") or [],
                                                     [ref for ref in sources if ref.get("role") != "prompt"])
    seat["source_refs"] = [ref for ref in sources if ref.get("role") == "prompt"]
    for key in _CLEARED_SEAT_KEYS:
        seat.pop(key, None)
    parts = list(seat.get("parts") or [])
    seat.update(operation_id=entry["operation_id"], status="pending", operation_state="in_flight", usd=None,
                critical_count=0, parts_answered=[], answers={
                    part: _blank_answer(part, "pending" if part in parts else ANSWER_NOT_ASKED) for part in answer_keys(parts)})


def _record_choice(drive: Any, record: Dict[str, Any], choices: List[_Choice]) -> Dict[str, Any]:
    """The one canonical write: a locked compare of each chosen seat, no other lock, no external I/O."""
    from ouroboros import review_ledger as ledger
    from ouroboros.utils import utc_now_iso

    decided_at = utc_now_iso()

    def mutate(payload: Dict[str, Any]) -> Dict[str, Any]:
        rows, changed = [dict(row) for row in payload.get("rows") or []], False
        for choice in choices:
            seat = next((row for row in rows if str(row.get("seat_id") or "") == choice.seat_id), None)
            current = str((seat or {}).get("operation_id") or "")
            if current == choice.entry["operation_id"]:
                continue  # an identical concurrent choice landed first; its body, clock and paths stand
            if (seat is None or current != choice.named or ledger.seat_open(seat)
                    or str(seat.get("status") or "") in ledger.ANSWERED_STATUSES):
                raise ContinuationRefused("state_changed", f"seat {choice.seat_id} changed before the choice was recorded")
            _reopen(seat, {**choice.entry, "decided_at": decided_at})
            changed = True
        if not changed:
            raise ledger.RecordUnchanged()
        counted = [row for row in rows if not row.get("additional")]
        before = dict(payload.get("verdict") or {})
        verdict = ledger.reduce_verdict(counted, dispatch_refusal=payload.get("dispatch_refusal"), pending=True)
        verdict["per_row"].update({str(row.get("seat_id") or ""): ledger.row_verdict(row) for row in rows if row.get("additional")})
        verdict.update({key: list(before.get(key) or []) for key in (*ledger.HEAVY_VERDICT_FIELDS, "degraded_reasons", "wave_failures")})
        return {**payload, "rows": rows, "verdict": verdict, "state": ledger.STATE_PENDING, "cost": ledger.record_cost(rows),
                "panel": {**dict(payload.get("panel") or {}), "not_reached": ledger.panel_not_reached(rows)}}

    return ledger.revise_record(drive, str(record["record_id"]), mutate) or record


def current_decisions(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """seat_id -> the recorded decision whose successor is that seat's current, still open operation."""
    from ouroboros.review_ledger import seat_open

    return {str(row.get("seat_id") or ""): decision for row in record.get("rows") or []
            if seat_open(row) and (decision := _last_decision(row))}


def continued_operations(drive: Any, record_id: str) -> List[str]:
    """The current operations of a record's continued seats: a left wave's duty and late mark name them,
    so a continued seat's later settlement is never deduplicated into the earlier one's."""
    from ouroboros.review_ledger import load_record

    try:
        record = load_record(drive, record_id) if record_id else None
    except ValueError:
        return []
    return sorted(str(row.get("operation_id") or "") for row in (record or {}).get("rows") or []
                  if row.get("superseded_operation_ids"))


def unsent_decisions(drive: Any, record_id: str, seats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For an author that ended: each recorded choice no roster row ever carried was never sent, a typed
    ``not_dispatched`` seat for the final settlement (no maintenance sends it; its target run stays history)."""
    from ouroboros.review_ledger import load_record

    try:
        record = load_record(drive, record_id) if record_id else None
    except ValueError:
        return []
    carried = {(str(seat.get("slot_id") or ""), str(seat.get("operation_id") or "")) for seat in seats}
    return [{"slot_id": seat_id, "operation_id": decision["operation_id"], "operation_state": "not_dispatched",
             "status": "not_dispatched", "parts": list(row.get("parts") or []), "late_result_pending": False,
             "error": "continuation decided, never sent"}
            for seat_id, decision in current_decisions(record or {}).items()
            for row in (record or {}).get("rows") or [] if row.get("seat_id") == seat_id
            if (seat_id, str(decision["operation_id"])) not in carried]


def _fenced(ctx: Any, task_id: str) -> bool:
    """Stop (a durable cancellation), a budget pause or the owner's Pause of this task: nothing new is decided.
    The owner's fence is read as an author's drain reads it (``review_pause.author_paused``): unreadable is open."""
    from ouroboros.budget_pause import dispatch_fenced
    from ouroboros.owner_pause import _member_coordinates, fence_closed, read_fence
    from ouroboros.review_custody import review_retry_cancelled

    if review_retry_cancelled(ctx) or dispatch_fenced(task_id):
        return True
    root_drive, root_task_id, member = _member_coordinates(ctx)
    try:
        fence = read_fence(root_drive, root_task_id) if root_drive and root_task_id else {}
    except Exception:
        return False
    return bool(fence_closed(fence) and member not in (fence.get("selected_members") or {}))


def _await_publication(ctx: Any, task_id: str, choices: List[_Choice], offers: List[Dict[str, Any]]) -> None:
    """Before a choice is recorded, the worker of each chosen operation in THIS process has finished publishing
    its settlement. Custody keeps that worker's live entry (``review_custody._ACTIVE``) after its seat settled
    and its reply landed, until its notices are out; the successor shares that attempt key, where custody
    refuses a different live operation as lost. The wait ends with that publication; Stop, Pause or the end
    of the chosen operation's original window refuse at $0 (nothing decided), reread each drain slice
    (``review_pause.DETACH_POLL_SEC``)."""
    from ouroboros import review_pause
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK

    ends, read_at = {choice.named: str(choice.entry.get("window_end") or "") for choice in choices}, float("-inf")
    while True:
        with _ACTIVE_LOCK:
            live = sorted(set(ends) & {entry.operation_id for entry in _ACTIVE.values()})
        if not live:
            return
        if time.monotonic() - read_at >= review_pause.DETACH_POLL_SEC:
            read_at = time.monotonic()
            if _fenced(ctx, task_id):
                raise ContinuationRefused("stopped_or_paused", "the owning task is stopped or entering a pause while "
                                          f"{', '.join(live)} still publishes its settlement; nothing was decided", offers)
            if any((left := _remaining_seconds(ctx, ends[op])) is not None and left <= 0 for op in live):
                raise ContinuationRefused("window_spent", "the original review window ended while the stopped run "
                                          "still published its settlement; start a fresh review instead", offers)
        time.sleep(PUBLICATION_POLL_SEC)


def _drop_replays(ctx: Any, superseded: set) -> None:
    """This process's settled replays, pending tokens and no-resend marks of superseded operations go stale."""
    from ouroboros.review_custody import _ACTIVE_LOCK, _NO_RESEND

    with _ACTIVE_LOCK:
        for cache in (getattr(ctx, "_review_settled_attempts", None), getattr(ctx, "_review_pending_invocations", None)):
            if isinstance(cache, dict):
                for key in [k for k, v in cache.items() if str(getattr(v, "operation_id", "") or (
                        v.get("operation_id") if isinstance(v, dict) else "") or "") in superseded]:
                    cache.pop(key, None)
        for key in [k for k, operation in _NO_RESEND.items() if operation in superseded]:
            _NO_RESEND.pop(key, None)


def _start_requested(custody: Any, task_id: str, slot_id: str, decision: Dict[str, Any],
                     closes: Optional[datetime] = None, record: Optional[Dict[str, Any]] = None) -> Optional[bool]:
    """Write the successor's START_REQUESTED from the RECORDED body (first wins per token): True when written, False
    when an earlier call wrote it, ``None`` when ``closes`` (``_window_closes``, read before this call's steps) passed
    first, so nothing was written. The START records that instant (``generation_end_at``, where the executor admits
    its POST and stops it generating) and asks the engine for no more than what remains of it once written. A
    read-only seat's released checkout is resolved from ``record`` into the START row's exact mapping (``RESTORE_KEY``)
    and put back before the START is written; every POST puts it back again (``review_session_custody.prepare_continuation_send``). A failure
    here is the choice's typed unsent refusal."""
    from ouroboros import delegate_custody as custody_module
    from ouroboros._usage_rows import REVIEW_ATTRIBUTION_KEYS
    from ouroboros.deadline_utils import bounded_seconds
    from ouroboros.observability import read_blob_ref
    from ouroboros.review_execution import _CLAUDEXOR_MAX_SECONDS

    token = str(decision["invocation_id"])
    if custody_module.invocation_record(custody, token):
        return False
    body = read_blob_ref(custody, decision["request_ref"], expected_kind="json")
    source = (custody_module.invocation_record(custody, str(decision.get("predecessor_invocation_id") or ""))
              or custody_module.invocation_record(custody, str(decision.get("origin_invocation_id") or "")) or {})
    from ouroboros.tools.review_subject import readonly_scope_root, released_subject, restore_released_subject

    restore = {}
    if record is not None and (read := readonly_scope_root(body)):  # an existing checkout is owned and mapped too
        origin = custody_module.invocation_record(custody, str(decision.get("origin_invocation_id") or "")) or {}
        try:
            restore = released_subject(custody, record, read, origin, str(decision.get("decided_by_task") or task_id))
        except ValueError as exc:
            if not pathlib.Path(read).exists():
                raise ContinuationRefused("subject_unavailable", f"{exc}; nothing was sent") from exc
        try:  # a live root that is no review checkout, or an existing one with no mapping, is read as it is
            if restore:
                restore_released_subject(custody, restore)
        except ValueError as exc:
            raise ContinuationRefused("subject_unavailable", f"{exc}; nothing was sent") from exc
    project_id, owned, left = str(source.get("project_id") or ""), False, _seconds_left(closes)
    scope = body.get("scope") or {}
    if not decision.get("thread_id") and scope.get("kind") == "project" and not scope.get("ephemeral"):
        from ouroboros.claudexor_daemon import ensure_owned_gateway
        from ouroboros.delegate_registration_policy import resolve_registration
        from ouroboros.execution_domain import run_gateway

        gateway = run_gateway(source, ensure_owned_gateway)  # its engine; the stopped run's registration may be retired
        try:
            project_id, owned_id, _persistent, _ = resolve_registration(gateway, str(scope.get("root") or ""), "",
                                                                     str(body.get("access") or "readonly"))
            owned, left = bool(owned_id), _seconds_left(closes)  # what remains once it is registered
            if owned and left is not None and left <= 0:  # the window closed meanwhile: no START will name it
                gateway.remove_project(project_id)
        finally:
            gateway.close()
    if left is not None and left <= 0:
        return None
    if left is not None:
        now = bounded_seconds(left, default=1, maximum=_CLAUDEXOR_MAX_SECONDS)
        body["maxSeconds"] = min(int(body.get("maxSeconds") or now), now)
    facts = copy.deepcopy(source.get("work_order_source_request") or {})
    facts.pop("original_prompt_sha256", None)
    (facts.get("turn_sources") or {}).pop("prompt", None)
    # The successor continues its stopped run (or that run's thread) where that run was admitted: the start is for that
    # store, and its first send or exact replay goes nowhere else (``replay_store_unproven``). A predecessor bound to
    # no store (an older engine or receipt) leaves "" and the earlier replay rule; never the cached handshake.
    from ouroboros.delegate_custody_current import run as custody_run

    stopped = custody_run(custody, str(decision.get("continues_run") or "")) if decision.get("continues_run") else None
    written = custody_module.record_start_requested(
        custody, run_id="", task_id=task_id, idempotency_key=token, invocation_id=token,
        daemon_store_id=str(getattr(stopped, "daemon_store_id", "") or ""),
        operation_id=decision["operation_id"], max_seconds=int(body.get("maxSeconds") or 0), request=body,
        continuation_of="" if decision.get("thread_id") else str(decision.get("continues_run") or ""),
        surface=decision["surface"], slot_id=slot_id, route=str(source.get("route") or ""),
        project_id=project_id, project_owned=owned, review_source_delivery=facts, work_order_source_request=facts,
        prompt_cache_ttl=source.get("prompt_cache_ttl") or None, **({RESTORE_KEY: restore} if restore else {}),
        **({"engine_ref": source["engine_ref"]} if source.get("engine_ref") else {}),  # the predecessor's engine
        **({"generation_end_at": closes.isoformat()} if closes is not None else {}),
        **{key: str(source.get(key) or "") for key in ("root_task_id", "parent_task_id", "category", "source",
                                                       *REVIEW_ATTRIBUTION_KEYS, "authority_source", "execution_root", "target_root")})
    if not written:
        raise ContinuationRefused("start_request_unwritable", "the successor start request could not be recorded; nothing was sent")
    return True


def _settle_unsent(drive: Any, record: Dict[str, Any], unsent: Dict[str, Tuple[Dict[str, Any], str]]) -> Dict[str, Any]:
    """A successor no roster row carried is never sent: its seat settles as that successor's typed
    ``not_dispatched`` outcome through the ledger's own collected settlement (``WINDOW_SPENT``: offered no
    more; ``START_UNSENT``: offered again); the target run and every earlier fact stay history."""
    from ouroboros.review_ledger import settle_from_collected

    raws = [{"slot_id": seat_id, "operation_id": decision["operation_id"], "status": "not_dispatched",
             "operation_state": "not_dispatched", "failure_code": code, "reported_cause": _UNSENT_CAUSES[code]}
            for seat_id, (decision, code) in unsent.items()]
    return settle_from_collected(drive, str(record["record_id"]), raws) or record


def _retire_start(custody: Any, task_id: str, seat_id: str, decision: Dict[str, Any], code: str) -> None:
    """A START no roster row carries can never be sent: its definite refusal (the existing START_FAILED fact)
    keeps every collector and recovery from sending it later."""
    from ouroboros import delegate_custody

    token = str(decision["invocation_id"])
    if (delegate_custody.invocation_record(custody, token) or {}).get("state") != "pending":
        return
    if not delegate_custody.emit(custody, delegate_custody.START_FAILED, {
            "run_id": "", "invocation_id": token, "task_id": task_id, "reason": code, "definite": True,
            "surface": str(decision["surface"]), "slot_id": seat_id}):
        raise ContinuationRefused("start_retirement_unwritable", "the never-sent start could not be retired")


def unsent_row(row: Dict[str, Any], decision: Dict[str, Any], binding: Dict[str, Any], code: str) -> Dict[str, Any]:
    """The roster row of a successor settled as never sent: it names the successor and carries no token."""
    kept = {key: value for key, value in row.items() if key not in (*_CLEARED_ROW_KEYS, "pending_invocation_id", "usage")}
    return {**kept, "operation_id": decision["operation_id"], "operation_state": "not_dispatched", "status": "not_dispatched",
            "late_result_pending": False, "failure_code": code, "reported_cause": _UNSENT_CAUSES[code],
            "error": _UNSENT_CAUSES[code], "not_dispatched_reason": _UNSENT_CAUSES[code], "usage": {},
            "recovery_binding": {key: value for key, value in binding.items() if key != "pending_invocation_id"},
            "superseded_operation_ids": [*(row.get("superseded_operation_ids") or []), str(decision["supersedes"])]}


def project_decisions(ctx: Any, custody: Any, record: Dict[str, Any], roster: Roster,
                      decisions: Optional[Dict[str, Dict[str, Any]]] = None, *, drive: Any = None) -> Dict[str, Any]:
    """Bring every projection in line with the recorded decisions (idempotent; explicit callers only).

    Order: the checking copy's holder, START_REQUESTED, the roster row, then this process's replays,
    so a passive reader always finds either the stopped row or a successor with its start recorded.
    A successor no roster row carried yet meets today's Stop/pause and the RECORDED window: none of it
    left, or a START an earlier call wrote and never carried, settles the seat unsent through that same
    row's compare-and-swap, so a concurrent carry wins it instead (``Roster.transition``). Returns
    ``{"written": seats whose start this call recorded and carried, "spent" / "unsent": seats settled
    unsent (window ended / start never sent), "record": the record after}``; any failure is
    ``ContinuationUnsent`` (the choice stands)."""
    from ouroboros.review_ledger import ledger_root

    decisions = current_decisions(record) if decisions is None else decisions
    try:
        return _project(ctx, custody, record, roster, decisions, drive if drive is not None else ledger_root(ctx))
    except ContinuationUnsent:
        raise
    except Exception as exc:  # whatever failed after the choice, the choice is recorded and a repeat repairs it
        raise ContinuationUnsent(exc.reason if isinstance(exc, ContinuationRefused) else "projection_failed",
                                 str(exc) if isinstance(exc, ContinuationRefused) else f"{type(exc).__name__}: {exc}",
                                 [{"seat_id": seat_id, "operation_id": decision["operation_id"],
                                   "continues": decision["supersedes"]} for seat_id, decision in decisions.items()]) from exc


def _project(ctx: Any, custody: Any, record: Dict[str, Any], roster: Roster,
             decisions: Dict[str, Dict[str, Any]], drive: Any) -> Dict[str, Any]:
    from ouroboros.budget_pause import dispatch_fenced
    from ouroboros.delegate_custody import invocation_record
    from ouroboros.review_custody import review_retry_cancelled
    from ouroboros.review_dispatch import review_operation_binding
    from ouroboros.review_operation import recorded_review_assignment

    written, fresh, joined, bindings, unsent, slots = [], {}, [], {}, {}, {}
    for seat_id, decision in decisions.items():
        request, slots[seat_id] = recorded_review_assignment(
            custody, task_id=roster.task_id, surface=str(decision["surface"]),
            row={"operation_id": decision["operation_id"], "slot_id": seat_id}, retry_key=roster.retry_key,
            root_task_id=roster.root_task_id, review_contract=roster.review_contract)
        slot = slots[seat_id]
        bindings[seat_id] = {**review_operation_binding(request, slot, str(decision["operation_id"])),
                             "pending_invocation_id": str(decision["invocation_id"])}
        row = roster.carried(seat_id)
        operation = str(row.get("operation_id") or "")
        if operation == str(decision["operation_id"]):  # another call already moved this row
            if row.get("operation_state") == "not_dispatched":
                unsent[seat_id] = str(row.get("failure_code") or START_UNSENT)
            else:
                joined.append(seat_id)  # carried: possibly sent, so it is only ever joined or replayed exactly
            continue
        if operation != str(decision["supersedes"]):
            raise ContinuationRefused("roster_changed", f"seat {seat_id}'s roster row names {operation or 'nothing'}, "
                                      f"not {decision['supersedes']}")
        # Collectors send only what a roster row carries: a row still naming the stopped operation proves no POST.
        if review_retry_cancelled(ctx) or dispatch_fenced(roster.task_id):
            raise ContinuationRefused("stopped_or_paused", "the owning task is stopped or entering a pause")
        closes = _window_closes(ctx, str(decision.get("window_end") or ""))
        left = _seconds_left(closes)
        code = (WINDOW_SPENT if left is not None and left <= 0 else
                START_UNSENT if invocation_record(custody, str(decision["invocation_id"])) else "")
        if not code:
            fresh[seat_id] = closes  # the instant, carried into the START: the steps below take from it
            continue
        now = roster.transition(seat_id, decision, lambda current, d=decision, b=bindings[seat_id], c=code: unsent_row(current, d, b, c))
        if now.get("operation_state") == "not_dispatched":
            unsent[seat_id] = code
        else:
            joined.append(seat_id)  # another call carried it between the read and the swap: join it
    if fresh:  # the recorded successor body is what would be sent
        from ouroboros.observability import read_blob_ref

        bodies = {seat_id: read_blob_ref(custody, decisions[seat_id]["request_ref"], expected_kind="json") for seat_id in fresh}
        _admit_new_generation(ctx, [(seat_id, str(decisions[seat_id]["surface"]), str(slots[seat_id].model),
                                     slots[seat_id].max_tokens, len(str(bodies[seat_id].get("prompt") or "")))
                                    for seat_id in fresh], [])
    for seat_id, closes in fresh.items():
        decision = decisions[seat_id]
        transfer_checking_copy(custody, roster.task_id, str(decision["supersedes"]), str(decision["operation_id"]))
        recorded_now = _start_requested(custody, roster.task_id, seat_id, decision, closes, record)
        now = roster.transition(seat_id, decision, lambda current, d=decision, b=bindings[seat_id], ended=recorded_now is None: (
            unsent_row(current, d, b, WINDOW_SPENT) if ended else continued_row(current, d, b)))  # ended: no START written
        if now.get("operation_state") == "not_dispatched":  # its window closed first, or another call settled it unsent
            unsent[seat_id] = str(now.get("failure_code") or WINDOW_SPENT)
        elif recorded_now:
            written.append(seat_id)
        else:
            joined.append(seat_id)
    for seat_id, code in unsent.items():
        _retire_start(custody, roster.task_id, seat_id, decisions[seat_id], code)
    if unsent:  # a seat the record no longer holds open is left as it is
        record = _settle_unsent(drive, record, {seat_id: (decisions[seat_id], code) for seat_id, code in unsent.items()})
    if written or joined:
        superseded = {str(op) for row in record.get("rows") or [] for op in row.get("superseded_operation_ids") or []}
        _drop_replays(ctx, superseded)
    return {"written": written, "spent": sorted(seat for seat, code in unsent.items() if code == WINDOW_SPENT),
            "unsent": sorted(seat for seat, code in unsent.items() if code != WINDOW_SPENT), "record": record}


def continued_row(row: Dict[str, Any], decision: Dict[str, Any], binding: Dict[str, Any]) -> Dict[str, Any]:
    """The roster row of the successor: in flight under its recorded token; the stopped facts stay elsewhere."""
    kept = {key: value for key, value in row.items() if key not in _CLEARED_ROW_KEYS}
    return {**kept, "operation_id": decision["operation_id"], "operation_state": "in_flight",
            "pending_invocation_id": str(decision["invocation_id"]), "late_result_pending": True,
            "usage": {"pending_invocation_id": str(decision["invocation_id"])}, "recovery_binding": dict(binding),
            "superseded_operation_ids": [*(row.get("superseded_operation_ids") or []), str(decision["supersedes"])]}


def arm(ctx: Any, record: Dict[str, Any], roster: Roster, selections: Any, *, drive: Any, custody: Any) -> Dict[str, Any]:
    """Validate, retain, record and project one explicit choice; the caller's collector sends it."""
    chosen = parse_selections(selections)
    choices = [_prepare(ctx, custody, record, roster, seat_id, named) for seat_id, named in chosen]
    new = [choice for choice in choices if choice.rejoin is None]
    if new:
        _await_publication(ctx, roster.task_id, new, continuable_seats(record))
        _admit_new_generation(ctx, [(choice.seat_id, str(choice.entry["surface"]), str(choice.prompt_payload["slot"]["model"]),
                                     choice.prompt_payload["slot"].get("max_tokens"), len(choice.body["prompt"])) for choice in new],
                              continuable_seats(record))
        _retain_sources(custody, roster.task_id, new)
        record = _record_choice(drive, record, new)
    decided = {str(row.get("seat_id") or ""): row for row in record.get("rows") or []}
    decisions = {}
    for choice in choices:
        decision = next((entry for entry in decided[choice.seat_id].get("continuations") or []
                         if entry.get("supersedes") == choice.named), None)
        if decision is not None and decided[choice.seat_id].get("operation_id") == decision["operation_id"]:
            decisions[choice.seat_id] = decision
    projected = project_decisions(ctx, custody, record, roster, decisions, drive=drive)
    from ouroboros.delegate_custody import invocation_record

    def facts(choice: _Choice) -> Dict[str, Any]:
        decision = decisions.get(choice.seat_id) or choice.rejoin or {}
        start = invocation_record(custody, str(decision.get("invocation_id") or "")) if decision else None
        return {"seat_id": choice.seat_id, "operation_id": decision.get("operation_id", ""), "continues": choice.named,
                "rejoined": choice.rejoin is not None, "start_recorded_now": choice.seat_id in projected["written"],
                "max_seconds": (start or {}).get("max_seconds"), "window_spent": choice.seat_id in projected["spent"],
                "not_sent": choice.seat_id in projected["unsent"]}
    return {"record": projected["record"], "continued": [facts(choice) for choice in choices]}
