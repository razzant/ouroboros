"""The measured context carried from Main fit to a settled physical attempt.

This data record is independent of the usage ledger and of context assembly.
"""
from __future__ import annotations

import contextlib
import contextvars
import datetime
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal, Optional, TypedDict

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
    # Why ``rendered_mode`` is lower ("" for agreement or an unknown cause): ``memory`` crowded the start's Max room,
    # ``route`` the window cannot hold the books and core, ``window`` the start's window took it otherwise,
    # ``after_start`` a lower mode the task took after its start (``context_measurement.mode_cause``).
    mode_cause: str = ""


OBSERVED = frozenset({"dispatched", "settled", "unresolved"})  # a send the ledger records as made


class ContextAttemptObservation(TypedDict, total=False):
    """One attempt's context as its task's activity row carries it (the base of ``ContextAttemptFact``): the
    ``attempt_fact`` fields, then the call identity of the frame that delivered it (``supervisor.events``).
    ``dispatched`` is the host's commitment, never a provider's acceptance; ``released`` retracts that attempt.
    While dispatch is provisional, ``previous`` may hold one earlier send of this task attempt (never a chain).
    The owner's current Settings mode is not here. All other keys are present on a published row."""
    attempt_id: str
    ledger_state: Literal["dispatched", "settled", "unresolved", "released"]
    round_id: str
    preferred_at_send: Literal["", "max", "low", "nano"]  # the task plan's owner mode when it sent
    applied: Literal["max", "low", "nano"]  # the mode the sent candidate was rendered in
    profile: Literal["owner_max", "owner_low", "owner_nano", "task_local_low", "task_local_nano"]
    cause: Literal["", "memory", "route", "window", "after_start"]  # why lower; "" also when the original start is unknown
    observed_at: str
    task_attempt: int
    llm_call_id: str
    execution_id: str
    call_attempt: int


class ContextAttemptFact(ContextAttemptObservation, total=False):
    """A current observation with at most one nonrecursive earlier send during provisional dispatch."""
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
        return {"attempt_id": str(capture.attempt_id), "ledger_state": str(capture.state), "round_id": context.round_id,
                "preferred_at_send": context.preferred_mode, "applied": context.rendered_mode,
                "profile": context.profile, "cause": context.mode_cause,
                "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    except Exception:  # a projection, never the send's authority
        return None


_DISPATCH_OBSERVER: contextvars.ContextVar[Optional[Callable[[Any], None]]] = contextvars.ContextVar(
    "physical_dispatch_observer", default=None)


@contextlib.contextmanager
def observe_dispatch(observer: Optional[Callable[[Any], None]]) -> Iterator[None]:
    """Bind who hears this context's physical dispatches and identity-bound releases, including in-call repairs."""
    token = _DISPATCH_OBSERVER.set(observer)
    try:
        yield
    finally:
        _DISPATCH_OBSERVER.reset(token)


def dispatched(capture: Any) -> None:
    """The ledger publishes dispatch before send, then a release if disproven, even before its own retry.
    The bound observer hears that capture. Its failure is logged, never raised: the observation cannot change
    the send, its accounting or whether it may be retried."""
    observer = _DISPATCH_OBSERVER.get()
    if observer is not None:
        try:
            observer(capture)
        except Exception:
            log.debug("Dispatch observer failed; the send goes on", exc_info=True)
