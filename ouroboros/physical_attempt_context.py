"""The measured context carried from Main fit to a settled physical attempt.

This data record is independent of the usage ledger and of context assembly.
"""
from __future__ import annotations

import contextlib
import contextvars
import datetime
import logging
from dataclasses import InitVar, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Optional, Tuple, TypedDict

if TYPE_CHECKING:
    from ouroboros.context_window import WindowRecord

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhysicalAttemptContext:
    profile: Literal["owner_max", "owner_low", "owner_nano", "task_local_low", "task_local_nano"]
    rendered_mode: Literal["max", "low", "nano"]
    measurement_basis: Literal["fresh_route_usage", "fresh_model_usage", "refused_input", "cold_estimate"]
    route_fp: str
    round_id: str
    target_total_tokens: Optional[int]
    capacity_total_tokens: Optional[int]
    context_target_miss: bool
    automatic_pass_used: bool
    # Main's calibration (real tokens per estimated token) of the measurement above;
    # the send finalizer sizes the reply with it. None on rows written before it existed.
    measurement_density: Optional[float] = None
    view_kind: str = ""  # the captured memory view; old producers stay unclassified
    response_reserve_tokens: int = 0
    window_witness_bound: bool = False
    window_observation_fp: str = ""  # Main account scope, separate from capability/density route_fp
    required_book_tokens: int = 0  # a directly required handbook inside this candidate (density applied), its note too
    preferred_mode: str = ""  # the owner's mode this task's plan holds at send (Settings may differ by now)
    # Why ``rendered_mode`` is lower ("" for agreement or an unknown cause): the start's Max room lost its whole books
    # to ``memory``, to what the ``request`` itself carries, or to the ``route``'s window; ``window`` the start's window
    # took it otherwise; ``after_start`` a lower mode the task took after its start (``context_measurement.mode_cause``).
    mode_cause: str = ""
    # Low's missed target at this task's start: the request with no memory at all, reply reserve included
    # (``memory_floor``'s ``floor.no_memory_tokens``); None when the target was met or none applied, and on older rows.
    no_memory_tokens: Optional[int] = None
    # What this request did to its route's cached start, read from the host's own earlier sends on that route
    # (``transcript_prefix.attempt_prefix_fact``): never the provider's verdict, which is the row's measured counts.
    # None on rows written before it existed and on sends without a Main candidate.
    prefix: Optional[dict] = field(default=None, compare=False)
    # The capture's prompt-file witness (``ContextCore.base_prompt_source``), unverified until the send seam matches
    # it against the leading system content (``model_send_seal``). Carried beside the record, never in it: it is no
    # field, so no ``asdict`` of a ledger row, receipt, capture or checkpoint holds it; ``replace`` keeps it.
    base_prompt_source: InitVar[Optional[Tuple[str, str, int, str]]] = None
    first_contact_scope: InitVar[str] = ""  # private until a comparable outcome reaches the evidence store
    first_contact_dispatch: InitVar[Optional[Tuple[int, str]]] = None
    # A fresh direct chat's plan (``ContextFitPlan.first_contact``): the scope its fit read and whether a start hint
    # shaped the view; the send seam signs ``first_contact_scope`` from the target it actually resolves.
    first_contact_plan: InitVar[Optional[Tuple[str, bool]]] = None
    window_boundary_tokens: InitVar[Optional[int]] = None  # stated boundary, not admission's reserve-adjusted total
    window_record: InitVar[Optional[WindowRecord]] = None  # live capacity terms; not an extra ledger projection
    dispatch_account: InitVar[str] = ""  # private route witness, not part of the priced or delivered context view
    # The native Ollama load this send's plan asked for (0: Auto), None for every other route. A record field, so the
    # ledger row and a restored receipt carry it; last, so a positional caller of the fields above is unchanged.
    requested_load_window: Optional[int] = None

    bible_source: InitVar[Optional[Tuple[str, str, int, str]]] = None
    assembly_witnesses: InitVar[tuple] = ()
    contribution_input: InitVar[Optional[tuple]] = None

    def __post_init__(self, base_prompt_source: Optional[Tuple[str, str, int, str]], first_contact_scope: str,
                      first_contact_dispatch: Optional[Tuple[int, str]],
                      first_contact_plan: Optional[Tuple[str, bool]], window_boundary_tokens: Optional[int],
                      window_record: Optional[WindowRecord], dispatch_account: str,
                      bible_source: Optional[Tuple[str, str, int, str]],
                      assembly_witnesses: tuple, contribution_input: Optional[tuple]) -> None:
        # Only a witness is kept on the instance: without one, ``vars`` holds just the record's fields, and ``replace``
        # reads the class's None back (it reads every InitVar with ``getattr``).
        if base_prompt_source is not None:
            object.__setattr__(self, "base_prompt_source", tuple(base_prompt_source))
        if first_contact_scope:
            object.__setattr__(self, "first_contact_scope", first_contact_scope)
            object.__setattr__(self, "first_contact_dispatch", first_contact_dispatch)
        if first_contact_plan:
            object.__setattr__(self, "first_contact_plan", tuple(first_contact_plan))
        if window_boundary_tokens is not None:
            object.__setattr__(self, "window_boundary_tokens", window_boundary_tokens)
        if window_record is not None:
            object.__setattr__(self, "window_record", window_record)
        if dispatch_account:
            object.__setattr__(self, "dispatch_account", dispatch_account)
        if bible_source is not None:
            object.__setattr__(self, "bible_source", tuple(bible_source))
        if assembly_witnesses:
            object.__setattr__(self, "assembly_witnesses", assembly_witnesses)
        if contribution_input is not None:
            object.__setattr__(self, "contribution_input", contribution_input)


def with_dispatch_account(context: Optional[PhysicalAttemptContext], account: str) -> Optional[PhysicalAttemptContext]:
    return replace(context, dispatch_account=account) if context is not None and account else context


_WIRE_ADMISSION = contextvars.ContextVar("physical_wire_admission", default=None)


@contextlib.contextmanager
def with_wire_admission(check):
    """One logical caller's extra admission over its final wire copy; never priced or serialized."""
    token = _WIRE_ADMISSION.set(check)
    try:
        yield
    finally:
        _WIRE_ADMISSION.reset(token)


def check_wire_admission(request, candidate, reservation) -> None:
    if check := _WIRE_ADMISSION.get():
        check(request, candidate, reservation)


def with_dispatch_stamp(request: Any, stamp: Tuple[int, str]) -> Any:
    """Bind the exact ledger dispatch order to a candidate prepared for the start preference."""
    context = request.physical_context
    return (replace(request, physical_context=replace(context, first_contact_dispatch=stamp))
            if context is not None and context.first_contact_scope else request)


OBSERVED = frozenset({"dispatched", "settled", "unresolved"})  # a send the ledger records as made


def shown_mode(rendered_mode: str) -> Tuple[str, str]:
    """``(mode, books)``: the mode a send is named in and what stands for the books in it, the one fact every surface
    reads. A start the window took from my human's Max renders the ``low`` frame and is named Low: it runs in Low,
    temporarily, for the window, while the owner's choice stays Max beside it (``preferred_at_send``, owner
    21.11.2026). Every mode and focus carries the resident Body Map (owner 25.11.2026), so ``books`` is ``body_map``
    on every new send; whether the window took the Map to its parts is the view's own receipt
    (``host_sections_addressed``). Rows written before that say ``whole`` or ``maps`` and stay readable."""
    return rendered_mode, "body_map"


class ContextAttemptObservation(TypedDict, total=False):
    """One attempt's context as its task's activity row carries it (the base of ``ContextAttemptFact``): the
    ``attempt_fact`` fields, then the call identity of the frame that delivered it (``supervisor.events``).
    ``dispatched`` is the host's commitment, never a provider's acceptance; ``released`` retracts that attempt.
    While this attempt is dispatched or released, ``previous`` may hold the next-newest attempt of this task attempt
    not proven unsent (one level, never a chain). The owner's current Settings mode is not here. All other keys are
    present on a published row."""
    attempt_id: str
    ledger_state: Literal["dispatched", "settled", "unresolved", "released"]
    round_id: str
    preferred_at_send: Literal["", "max", "low", "nano"]  # the task plan's owner mode when it sent
    applied: Literal["max", "low", "nano"]  # the mode the send is named in (``shown_mode``): Low for Low under Max
    books: Literal["body_map", "whole", "maps"]  # what stood for the books in this send (``shown_mode``)
    profile: Literal["owner_max", "owner_low", "owner_nano", "task_local_low", "task_local_nano"]
    cause: Literal["", "memory", "request", "route", "window", "after_start"]  # why applied is lower than preferred_at_send; "" also when unknown
    target_missed: bool  # this send exceeded the owner's mode target (``context_target_miss``)
    no_memory_tokens: Optional[int]  # Low's missed target: the start's request with no memory at all, else None
    observed_at: str
    task_attempt: int
    llm_call_id: str
    execution_id: str
    call_attempt: int


class ContextAttemptFact(ContextAttemptObservation, total=False):
    """A current observation with at most one nonrecursive earlier attempt not proven unsent."""
    previous: ContextAttemptObservation


def attempt_fact(capture: Any, round_id: str = "") -> Optional[ContextAttemptFact]:
    """One physical send's context for its task's activity record (``context_attempt``): a projection of the
    ledger row's ``physical_context`` (the SSOT), bound to the attempt id and to the ledger's state of it
    (``ledger_state``: dispatched -- the host committed this candidate to its transport, not a provider's acceptance,
    which the call's ``model_activity`` reports --, settled or unresolved). Released is an identity-bound retraction,
    not a sent mode. None for no capture, no Main context, a reserved candidate or another round's capture.
    Never raises: a projection failure leaves the send and its accounting as they are."""
    try:
        context = getattr(capture, "physical_context", None)
        if (not isinstance(context, PhysicalAttemptContext) or getattr(capture, "state", "") not in OBSERVED | {"released"}
                or (round_id and context.round_id != round_id)):
            return None
        applied, books = shown_mode(context.rendered_mode)
        # Task-local Low exists only under my human's Max: a row written before the owner's mode was kept says so too
        preferred = context.preferred_mode or ("max" if context.profile == "task_local_low" else "")
        return {"attempt_id": str(capture.attempt_id), "ledger_state": str(capture.state), "round_id": context.round_id,
                "preferred_at_send": preferred, "applied": applied, "books": books,
                "profile": context.profile, "cause": context.mode_cause,
                "target_missed": bool(context.context_target_miss), "no_memory_tokens": context.no_memory_tokens,
                "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    except Exception:  # a projection, never the send's authority
        return None


_PROGRESS = {"dispatched": 0, "unresolved": 1, "settled": 2}
_CALL = ("task_attempt", "llm_call_id", "execution_id", "round_id", "call_attempt")


def _advances(held: str, new: str) -> bool:
    """A ledger state only moves forward here: a release is terminal, and a settled receipt is never retracted."""
    if new == "released":
        return held in {"dispatched", "unresolved"}
    return held != "released" and new in _PROGRESS and _PROGRESS[new] >= _PROGRESS.get(held, -1)


def _still_answerable(entries: list) -> list:
    """Newest first, what can still become the observed fact. Only a sender's settled capture is a pruning barrier;
    unresolved accepts a late unsent proof in the ledger. Administrative abandoned settlements never produce these
    sender captures. A released attempt never counts:
    it stays only as the head or beside its own call, where it stops a stale frame of that call reviving it."""
    head, kept = entries[0], []
    for position, sent in enumerate(entries):
        if position and sent["ledger_state"] == "released" and any(sent.get(key) != head.get(key) for key in _CALL):
            continue
        kept.append(sent)
        if sent["ledger_state"] == "settled":
            break
    return kept


def fold_attempt(held: list, fact: dict, *, active: bool) -> Optional[list]:
    """One frame folded into a task attempt's private newest-first attempts; ``None`` when it changes nothing.
    Only the task's active call (``active``) opens a new attempt; any frame of an attempt still held under its exact
    call identity may advance that attempt, its call closed or not, so a late outcome lands on its own attempt.
    A send's time is its first observation; settlement cannot make an old send newer."""
    fact = {key: value for key, value in fact.items() if key != "previous"}
    entries = [sent for sent in held if sent.get("task_attempt") == fact.get("task_attempt")]
    index = next((at for at, sent in enumerate(entries) if sent.get("attempt_id") == fact.get("attempt_id")), None)
    if index is None:
        if not active:
            return None
        entries.insert(0, fact)
    else:
        sent = entries[index]
        if any(sent.get(key) != fact.get(key) for key in _CALL) or not _advances(sent["ledger_state"], fact["ledger_state"]):
            return None
        entries[index] = {**fact, "observed_at": sent["observed_at"]}
    return _still_answerable(entries)


def attempt_view(entries: list) -> ContextAttemptFact:
    """The public fact of ``fold_attempt``'s attempts: the newest one, with the next attempt not proven unsent as its one
    ``previous`` while its own send is still unproven or disproven. New objects: published rows are never mutated."""
    head, *older = entries
    view: ContextAttemptFact = dict(head)  # type: ignore[assignment]
    if head["ledger_state"] in {"dispatched", "released"}:
        earlier = next((sent for sent in older if sent["ledger_state"] != "released"), None)
        if earlier is not None:
            view["previous"] = dict(earlier)  # type: ignore[typeddict-item]
    return view


_DISPATCH_OBSERVER: contextvars.ContextVar[Tuple[Callable[[Any], None], ...]] = contextvars.ContextVar(
    "physical_dispatch_observer", default=())


@contextlib.contextmanager
def observe_dispatch(observer: Optional[Callable[[Any], None]]) -> Iterator[None]:
    """Add who hears this context's physical dispatches and identity-bound releases, including in-call repairs. An
    outer listener keeps hearing them beside an inner one (a round's start-hint seam beside each attempt's own)."""
    token = _DISPATCH_OBSERVER.set(_DISPATCH_OBSERVER.get() + ((observer,) if observer is not None else ()))
    try:
        yield
    finally:
        _DISPATCH_OBSERVER.reset(token)


def dispatched(capture: Any) -> None:
    """The ledger publishes dispatch before send, then a release if disproven, even before its own retry.
    Every bound observer hears that capture. A failure is logged, never raised: the observation cannot change
    the send, its accounting or whether it may be retried."""
    for observer in _DISPATCH_OBSERVER.get():
        try:
            observer(capture)
        except Exception:
            log.debug("Dispatch observer failed; the send goes on", exc_info=True)
