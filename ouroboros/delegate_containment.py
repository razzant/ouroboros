"""Containment verification for ONE delegated run (extracted from tools/delegate.py
for the module-size gate — same logic, same names, no behavioural change).

The host asks the ENGINE's own artifacts what a delegated run actually ran under
— the applied access profile and the applied scoped HOME — and compares them with
what the host's authority entitled the child to. A recorded WIDER profile or an
unexcused operator-home landing is a containment breach the body cancels on
(``hold_to_grant``, every run it observes; the windowed wait keeps its own); an
absent fact stays absence ("unproven"), reported by the evidence reader instead of
enforced as a fault.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ouroboros import delegate_custody as custody
from ouroboros.utils import resolve_path_allow_missing

log = logging.getLogger(__name__)
_TERMINAL_STATES = custody.TERMINAL_STATES

# Ordering, not a set: honest verification needs to tell "narrower than asked" (fine)
# from "wider than asked" (a refusal). An unlisted profile ranks above everything
# known, so a profile added by a future engine is treated as widening, not ignored.
_ACCESS_RANK: Dict[str, int] = {
    "readonly": 0,
    "workspace_write": 1,
    "full": 2,
    "external_sandbox_full": 2,
    "inherit_native": 3,
}
_ACCESS_UNVERIFIED = "access_unverified"
_UNKNOWN_ACCESS_RANK = 99


def _resolved(path: Any) -> Optional[pathlib.Path]:
    try:
        return (
            resolve_path_allow_missing(pathlib.Path(str(path)))
            if str(path or "").strip()
            else None
        )
    except (OSError, ValueError, RuntimeError, TypeError):
        return None


def _widened_access(detail: Dict[str, Any], expected: str) -> str:
    """The effective profile when the engine ran WIDER than the host asked, else ''.

    Claudexor derives effective access itself, so the request is a request. Honest
    verification means reading back what was enforced instead of trusting the echo; a
    narrower effective profile is fine, a wider one is a containment breach.
    """
    summary = custody.summary_of(detail)
    # `access` is NOT an independent witness: the daemon computes it as
    # `effectiveAccess ?? the client's own parsed request`, so falling back to it compares
    # our request against itself and always passes. Only the derived field testifies.
    effective = str(summary.get("effectiveAccess") or "")
    state = str(summary.get("state") or "")
    # The journal cursor is the honest "has this run produced anything yet" signal: a
    # freshly dequeued run has not written its contract, so it cannot have disclosed.
    # It lives on the run DETAIL, not on the summary — reading it from the summary made
    # this whole branch unreachable against the real wire shape while its test, whose
    # fixture put it in the summary, went on asserting the gate worked.
    try:
        seq_seen = int(detail.get("lastSeq") or 0) > 0
    except (TypeError, ValueError):
        seq_seen = False
    if not effective:
        # An UNDISCLOSED profile is not a verified narrow one — but absence only means
        # "no evidence" while the run can still act. The daemon marks a run `running` at
        # DEQUEUE, before the orchestrator writes the contract that derives the profile,
        # and a run that failed or was cancelled before that write never has one at all.
        # Treating those as breaches cancelled healthy runs and reported a failed start
        # as a containment fault. Judge only a run that is admitted, past its first
        # disclosure, and not already over.
        if state in ("", "queued") or state in _TERMINAL_STATES:
            return ""
        if not seq_seen:
            return ""
        return _ACCESS_UNVERIFIED
    if _ACCESS_RANK.get(effective, _UNKNOWN_ACCESS_RANK) > _ACCESS_RANK.get(expected, 0):
        return effective
    return ""


@dataclass(frozen=True)
class _Breach:
    """A containment guarantee the ENGINE did not deliver. Typed, with its evidence."""

    code: str
    detail: str
    facts: Dict[str, Any]


def _home_isolation_breach(detail: Dict[str, Any]) -> Optional[_Breach]:
    """Did the scoped harness HOME the host ASKED for actually get APPLIED?

    Asking is not evidence: ``execution.delegated`` is a request, and a request that
    the engine accepted and then did not honour leaves the harness holding the
    operator's real ``$HOME`` — with ``~/.claudexor/v3/daemon/token`` in it, which
    grants the entire ``/v2`` control API. Claudexor records the APPLIED fact on each
    attempt (``harness_home_isolated`` / ``harness_home_dir``) and projects it onto no
    ``/v2`` response, so the artifact is the only witness there is.

    THE RULE IS TWO EXACT FACTS (phase A3, Poltergeist sprint — the simplified
    form): a breach is ONLY a recorded ``harness_home_isolated: false``, or an
    applied home EQUAL to the operator's own (the claim is the lie, whatever
    boundary sits beside it). Everything else is reported, not enforced:

    - A MISSING fact stays absence. Current Claudexor spreads the applied facts
      into ``attemptFailureRecord`` too, but ``harness_home_isolated`` is the one
      optional member, omitted when the attempt died before its home was decided
      — "a01 errored, a02 repaired it" is the ordinary converge loop, and reading
      absence as a fault cancelled healthy finished runs. Unproven is REPORTED by
      ``_containment_evidence``, never enforced.
    - A scoped home NESTED under ``$HOME`` — with or without a recorded OS
      boundary — is NOT a breach. The engine roots every scoped home under its
      runtime dir, which lives under ``$HOME`` on every host it supports, and on
      hosts with no boundary mechanism (every non-macOS host today) it CANNOT
      record one, so the old nested-without-mechanism rule cancelled every
      mutating Linux run post-factum (the colleague's issue-2 class). The nested
      shape flows to the EXISTING disclosed-unconfined path instead
      (``delegate_run_unconfined`` + the three-place disclosure): the child
      already holds a shell in this worktree, and the marginal step from "shell"
      to "``~``-relative token reachability" does not justify cutting the lane on
      every boundary-less host (AGENTS.md "Disclose instead of forbid";
      Proportionality). ``confinement_unavailable_reason`` from the same attempt
      artifact amplifies that disclosure — telemetry, never an admission token.
    """
    from ouroboros.gateways.claudexor import attempt_containment, operator_home

    run_dir = str(custody.summary_of(detail).get("runDir") or "")
    if not run_dir:
        return None
    attempts = attempt_containment(run_dir)
    if not attempts:
        return None
    real_home = _resolved(operator_home())
    for attempt in attempts:
        if attempt.home_isolated is None:
            continue
        if attempt.home_isolated is False:
            return _Breach(
                "home_isolation_not_applied",
                "The delegated run asked for a scoped harness HOME and the engine "
                "recorded that none was applied, leaving the harness in the "
                "operator's own home, where the Claudexor daemon token grants the "
                "entire control API.",
                {"attempt_id": attempt.attempt_id, "harness_home_dir": attempt.home_dir,
                 "harness_home_isolated": attempt.home_isolated},
            )
        applied = _resolved(attempt.home_dir) if attempt.home_dir else None
        # EQUALITY is never excused: an "isolated" home naming the operator's own
        # verbatim is the lie itself. Both sides resolved, so a symlink cannot
        # launder it. (Nested-under-$HOME is the engine's own layout — disclosed,
        # not refused; see the docstring.)
        if applied is not None and real_home is not None and applied == real_home:
            return _Breach(
                "home_isolation_not_applied",
                "The delegated run claims an isolated harness HOME that IS the "
                "operator's own home, where the Claudexor daemon token grants the "
                "entire control API.",
                {"attempt_id": attempt.attempt_id, "harness_home_dir": attempt.home_dir,
                 "harness_home_isolated": attempt.home_isolated},
            )
    return None


def halt_breached(drive_root: Any, gateway: Any, entry: Any, breach: _Breach) -> str:
    """Stop a run the engine did not contain as asked, through the ONE cancel path (its four typed
    outcomes) and the incident writer an unverified cancel uses, so the breach stays a CRITICAL health
    invariant until a terminal receipt resolves it. Returns the cancel outcome."""
    try:
        cancelled = custody.cancel_and_verify(drive_root, gateway, entry, breach.code)
    except Exception:
        log.warning("Failed to cancel an uncontained delegated run %s", entry.run_id, exc_info=True)
        cancelled = {"outcome": custody.CANCEL_CONTAINMENT_FAULT}
    custody.record_containment_fault(drive_root, entry, breach.code, breach.detail, fault=breach.code, **breach.facts)
    return str(cancelled.get("outcome") or custody.CANCEL_CONTAINMENT_FAULT)


def granted_shape(entry: Any) -> Any:
    """The GRANTED run shape, replayed from the run's start row (access, mode, isolation, the delegated
    HOME marker and the investigation authority): what every check of a held run holds it to."""
    from ouroboros.delegate_registration_policy import INVESTIGATION_AUTHORITY
    from ouroboros.subagents import DelegatedRunShape

    return DelegatedRunShape(access=entry.access, mode=entry.mode, isolation=entry.isolation,
                             delegated=entry.delegated,
                             investigation=entry.authority_source == INVESTIGATION_AUTHORITY)


def hold_to_grant(drive_root: Any, gateway: Any, entry: Any, detail: Dict[str, Any]) -> bool:
    """The rights floor the body keeps on every run it observes (pool-capacity D3): what the engine
    enforced, read from a detail, against the grant recorded on the run's start row. A breach stops the
    run once (its open incident names it); the nanny reads the breach with the terminal. A row without a
    recorded grant (written before the shape rode the start row) has nothing to hold the run to here: the
    waiting nanny's live derivation keeps it. Whether it stopped the run; a check that cannot be read is logged."""
    from ouroboros.tools.delegate_terminal_evidence import _containment_breach

    try:
        breach = _containment_breach(detail, granted_shape(entry)) if entry.access else None
        if breach is None or (entry.containment_fault or {}).get("reason") == breach.code:
            return False
        halt_breached(drive_root, gateway, entry, breach)
    except Exception:
        log.warning("Rights floor of delegated run %s unread this beat", entry.run_id, exc_info=True)
        return False
    return True


def home_nested_under_operator_home(detail: Dict[str, Any]) -> bool:
    """Did any attempt's applied scoped HOME land INSIDE the operator's own home?

    NOT a breach (see ``_home_isolation_breach``) — the engine roots every scoped
    home under its runtime dir, which lives under ``$HOME``. It is a REPORTING
    fact, and one the evidence reader must never drop: a nested home leaves
    ``~/.claudexor/v3/daemon/token`` reachable by absolute path, so a run that
    also recorded an OS boundary was being promoted to ``verified: true`` with a
    note claiming a home "outside the operator's own" — and, because the durable
    unconfined row is only emitted for runs with no boundary, the disclosure
    disappeared entirely. Absence of the fact stays absence: a run that recorded
    no home is unproven, not nested.
    """
    from ouroboros.gateways.claudexor import attempt_containment, operator_home

    run_dir = str(custody.summary_of(detail).get("runDir") or "")
    if not run_dir:
        return False
    real_home = _resolved(operator_home())
    if real_home is None:
        return False
    for attempt in attempt_containment(run_dir):
        applied = _resolved(attempt.home_dir) if attempt.home_dir else None
        if applied is not None and real_home in applied.parents:
            return True
    return False

