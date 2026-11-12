"""The custody side of one ``review_change`` wave.

``ouroboros/tools/review_change.py`` owns the request, the wave and its ledger
record; what the shared review state (``review_state``) knows about the wave lives
here: the reserved attempt row the per-task cycle ceiling counts once a physical
send stamps it paid, the delegated start token checkpointed onto its exact reserved
seat, the settled row carrying the wave's final roster — and the rejoin. Before a rerun pays again, the
open operation of the SAME logical round on the SAME task (identity c,
``review_subject.review_retry_key``) is collected through the commit gate's
reconcile-only path (``review_custody``: settled seats replay, the exact pending
delegated invocation is rejoined, nothing new is sent). A rejoin reads the retained
checkout at the deterministic path of this task's round
(``review_subject.checkout_token``), so its custody attempt key and every
operation's recovery binding are those of the operation it rejoins, in this process
or after a restart; another task's wave of the round has its own path. The exact repair of a
persisted attempt an earlier writer left held lives here too (``repair_settled_attempt``; for
the checkout pass, all held attempts in one write, ``repair_held_attempts``; for a collection that reads a settled
record, ``reconcile_collected_attempt``, #2779), and so does reading an author continuation's evicted free outcome
back from its settled record (``commit_reference_source``, #2830).
"""

from __future__ import annotations

import copy
import logging
import pathlib
from typing import Any, Dict, Optional

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

TOOL_NAME = "review_change"


def _tool_name(wave: Any) -> str:
    """The wave's own attempt-row tool name (``review_change.wave_tool_name``)."""
    from ouroboros.tools.review_change import wave_tool_name

    return wave_tool_name(getattr(wave, "request", None))


def paid_repo_key(ctx: Any, root: Any) -> str:
    """The host-bound source owns a copy's paid pool, never its review identity.

    Follow actual copy provenance only: neither a directory name nor a shared
    remote proves ownership. Pin this address on the paid row before copy GC.
    """
    from ouroboros.review_ledger import ledger_root
    from ouroboros.subagent_worktrees import list_worktrees
    from ouroboros.workspace_copies import copy_binding, same_directory

    bindings = [copy_binding(ctx), *(
        {"execution_root": row.get("path"), "source_root": row.get("repo_dir")}
        for row in list_worktrees(ledger_root(ctx)) if row.get("kind") != "delegated_exec")]
    current, seen = pathlib.Path(root).resolve(), set()
    while str(current) not in seen:
        seen.add(str(current))
        bound = next((item for item in bindings if same_directory(current, item.get("execution_root"))), {})
        source = str(bound.get("source_root") or "")
        if not source:
            return str(current)
        if not pathlib.Path(source).is_absolute():
            raise ValueError("review copy source binding must be absolute")
        current = pathlib.Path(source).resolve()
    raise ValueError("review copy source bindings contain a cycle")


def review_state_root(ctx: Any) -> pathlib.Path:
    """Canonical attempt custody; move this task's legacy rows off its child drive.

    A crash after the canonical save can leave the old row: the next move only
    removes that duplicate, never overwrites a newer canonical settlement. Both
    locks cover the move; new dispatch/checkpoint/settlement writes only canonical.
    """
    from ouroboros.review_ledger import ledger_root
    from ouroboros.review_state import _STATE_RELPATH, ReviewStateMutation, update_state
    from ouroboros.review_state_records import _attempt_identity_tuple

    canonical, old = ledger_root(ctx), pathlib.Path(ctx.drive_root).resolve()
    if old == canonical or not (old / _STATE_RELPATH).exists():
        return canonical
    task_id = str(getattr(ctx, "task_id", "") or "")

    def move(state: Any) -> ReviewStateMutation:
        rows = [row for row in state.attempts if row.tool_name == TOOL_NAME and row.task_id == task_id]
        if not rows:
            return ReviewStateMutation(None, changed=False)

        def accept(target: Any) -> None:
            existing = {_attempt_identity_tuple(row): row for row in target.attempts}
            for row in rows:
                prior = existing.get(_attempt_identity_tuple(row))
                if prior is not None:
                    if any(getattr(prior, key) != getattr(row, key) for key in (
                            "review_retry_key", "review_record_id", "root_task_id")):
                        raise ValueError("conflicting legacy review attempt; custody was not moved")
                    continue
                moved = copy.deepcopy(row)
                moved.review_budget_repo_key = row.review_budget_repo_key or paid_repo_key(ctx, row.repo_key)
                target.record_attempt(moved)

        update_state(canonical, accept)
        state.attempts = [row for row in state.attempts if row not in rows]
        return ReviewStateMutation(None, changed=True)

    update_state(old, move)
    return canonical


def attempt_record(wave: Any, ctx: Any, **fields: Any) -> Any:
    """One attempt row of this wave (the gate's ``CommitAttemptRecord``): a review only.
    A paid row is bound to the process that pays it (``stamp_paid_review_owner``, as the
    gate's rows are): should that process die mid-wave, the next server generation's
    startup reconciliation proves the owner dead and closes the tokenless row as an
    infra failure instead of leaving an open operation nobody can collect; a row whose
    seats hold durable delegated tokens stays recoverable by their exact rejoin."""
    from ouroboros.review_owner_custody import stamp_paid_review_owner
    from ouroboros.review_state import CommitAttemptRecord, make_repo_key

    attempt = CommitAttemptRecord(
        ts=utc_now_iso(), commit_message=wave.label, task_id=str(getattr(ctx, "task_id", "") or ""),
        root_task_id=wave.root_task_id, repo_key=make_repo_key(wave.root), tool_name=_tool_name(wave),
        review_budget_repo_key=(getattr(getattr(wave, "rejoin", None), "review_budget_repo_key", "")
                                or paid_repo_key(ctx, wave.root)),
        pre_review_fingerprint=str(wave.frozen.diff_sha), review_retry_key=wave.retry_key,
        rebuttal_sha256=wave.rebuttal_sha, review_contract_fingerprint=wave.contract_fp,
        review_record_id=wave.record_id, **fields)
    # A reserved roster row is owned by this process before it is paid (startup reconciliation).
    stamp_paid_review_owner(attempt, paid=bool(getattr(attempt, "paid", False)) or attempt.status == "reviewing")
    return attempt


def pending_round_attempt(ctx: Any, *, root: pathlib.Path, retry_key: str, tool_name: str = TOOL_NAME) -> Optional[Any]:
    """The open operation of THIS round on THIS task, or ``None``: the newest attempt
    row of (root, ``review_change``, task) under the wave's retry key whose custody is
    still active — a seat late, in flight or lost (``review_state_custody``). Found,
    the rerun collects it instead of paying for a second physical review; only a
    NEW paid wave meets the per-task cycle ceiling. An unreadable review state
    raises: it is not evidence that nothing is owed."""
    from ouroboros.review_state import _STATE_RELPATH, ReviewStateMutation, _utc_now, make_repo_key, update_state

    drive = review_state_root(ctx)
    if not (drive / _STATE_RELPATH).exists():
        return None  # no review state holds no open round; a refused call writes none
    repo_key, task_id = make_repo_key(root), str(getattr(ctx, "task_id", "") or "")

    def _active(state: Any) -> ReviewStateMutation:
        expired = state.expire_stale_attempts(now_ts=_utc_now())
        rows = [item for item in state.get_active_attempts(repo_key=repo_key)
                if item.tool_name == tool_name and item.task_id == task_id and item.review_retry_key == retry_key]
        return ReviewStateMutation(rows, changed=bool(expired))

    rows = update_state(drive, _active)
    return rows[-1] if rows else None


def pending_preflight_for_call(ctx: Any, *, goal: str, scope: str, review_rebuttal: str) -> Optional[Any]:
    """Probe the same public preflight identity before a commit's fresh-row check.

    No checkout or operation is created. A missing/unreadable exact match keeps
    the ordinary invalid-catalog refusal before staging.
    """
    from ouroboros.tools.review_change import _round_sha, _system_repo, parse_request
    from ouroboros.tools.review_subject import ReviewSubjectSpec, freeze_subject, review_retry_key

    try:
        root = _system_repo(ctx)
        request = parse_request(dict(subject="worktree", surface="preflight", reviewers=["recorded"],
                                     goal=goal, scope=scope, review_rebuttal=review_rebuttal))
        frozen = freeze_subject(ctx, ReviewSubjectSpec(root_kind="system_repo", root=str(root),
                                                       kind="worktree", surface="preflight"))
        return pending_round_attempt(ctx, root=root, retry_key=review_retry_key(
            frozen, round_sha=_round_sha(request, frozen)))
    except (OSError, ValueError, RuntimeError):
        return None


def arm_rejoin(ctx: Any, wave: Any) -> None:
    """Put the wave on the commit gate's reconcile-only path for the open operation it
    collects: ``parallel_review`` freezes the attempt's roster
    (``review_custody.prepare_frozen_review_reconciliation``), the custody layer
    replays each settled seat and rejoins the exact pending delegated invocation, and
    ``git_review_cycle._reconcile_and_clear_review_roster`` merges the roster back.
    Read inside the wave context, after its reset. A no-op for a new wave."""
    rejoin = getattr(wave, "rejoin", None)
    if rejoin is None:
        return
    ctx._review_reconcile_only = True
    ctx._pending_review_attempt = rejoin
    ctx._current_review_attempt_number = int(rejoin.attempt or 0)


def install_paid_stamp(ctx: Any, wave: Any) -> Dict[str, int]:
    """The wave's attempt row, keyed by the REVIEWED root so the shared per-task-tree
    ceiling of (this root, ``review_change``) counts it: the reservation writes the
    roster before any worker starts, unpaid; the first physical dispatch merges the
    write-ahead paid fact onto that row (``review_state_custody.mark_review_attempt_paid``);
    the delegated start token is checkpointed onto its reserved seat
    (``checkpoint_pending_review_invocation``), so a restart finds the exact invocation
    to rejoin. A rejoin writes onto the attempt row it collects, never a second row.
    A new wave first settles its round's completed earlier wave
    (``settle_completed_round``): an explicit repeat after a wave without a verdict
    is a new physical attempt, while open or unknown custody is still collected and a
    usable verdict was already reused by the ledger (``reuse_or_none``)."""
    from ouroboros.review_dispatch import ReviewPaidStamp
    from ouroboros.review_state import (
        checkpoint_pending_review_invocation, make_repo_key, mark_review_attempt_paid, settle_completed_round,
        update_state,
    )

    holder = {"attempt": int(getattr(getattr(wave, "rejoin", None), "attempt", 0) or 0)}
    drive, task_id = review_state_root(ctx), str(getattr(ctx, "task_id", "") or "")
    identity = {"repo_key": make_repo_key(wave.root), "tool_name": _tool_name(wave), "task_id": task_id}
    if getattr(wave, "rejoin", None) is None:
        settle_completed_round(ctx, drive, review_retry_key=wave.retry_key, **identity)

    def _record(*, paid: bool) -> None:
        reserved = getattr(ctx, "_review_reserved_roster", None)
        reserved = reserved if isinstance(reserved, dict) else {}
        triad = copy.deepcopy(list(reserved.get("multi_model_review") or []))

        def _mutate(state: Any) -> None:
            number = holder["attempt"] or state.next_attempt_number(
                identity["repo_key"], identity["tool_name"], task_id)
            state.record_attempt(attempt_record(
                wave, ctx, status="reviewing", phase="review", paid=paid, attempt=number,
                late_result_pending=not paid, triad_raw_results=triad))
            holder["attempt"] = number

        update_state(drive, _mutate)

    def _write() -> None:
        if holder["attempt"]:
            mark_review_attempt_paid(drive, attempt=holder["attempt"], review_retry_key=wave.retry_key, **identity)
        else:  # no reservation (a caller dispatching without one): the row and its paid fact at once
            _record(paid=True)

    def _checkpoint(*, surface: str, slot_id: str, operation_id: str, invocation_id: str) -> None:
        checkpoint_pending_review_invocation(
            drive, attempt=holder["attempt"], review_retry_key=wave.retry_key, surface=surface, slot_id=slot_id,
            operation_id=operation_id, invocation_id=invocation_id, **identity)

    ctx._review_paid_stamp = ReviewPaidStamp(_write, fail_closed=True, reserve=lambda: _record(paid=False))
    ctx._review_pending_invocation_checkpoint = _checkpoint
    return holder


def settle_attempt(ctx: Any, wave: Any, outcome: Dict[str, Any], payload: Dict[str, Any],
                   forensic: Dict[str, Any]) -> bool:
    """Close the attempt row this wave opened or rejoined (a review only; never a
    commit) with the wave's final roster: a seat still owed stays
    ``late_result_pending`` with its exact operation and start token, so a later
    rerun of the round — in this process or after a restart — finds what to collect.
    The row's paid fact is its dispatch stamp's alone: a wave that sent nothing stays unpaid."""
    number = (int((outcome.get("attempt") or {}).get("attempt") or 0)
              or int(getattr(getattr(wave, "rejoin", None), "attempt", 0) or 0))
    if number <= 0:
        return True
    from ouroboros.review_state import update_state

    verdict = dict(payload.get("verdict") or {})
    try:
        update_state(review_state_root(ctx), lambda state: state.record_attempt(attempt_record(
            wave, ctx, status="reviewed", phase="review", paid=False, attempt=number,
            late_result_pending=payload.get("state") == "pending",
            block_reason=str(outcome.get("block_reason") or ""),
            # The record's wave failure note in its safe words; the complete note stays on the record.
            block_details="\n".join(str(note.get("reported_cause") or "") for note in verdict.get("wave_failures") or []),
            triad_raw_results=copy.deepcopy([row for row in forensic.get("triad_raw") or [] if isinstance(row, dict)]),
            degraded_reasons=[str(item) for item in verdict.get("degraded_reasons") or []])))
        return True
    except Exception:
        log.warning("review_change attempt row was not settled", exc_info=True)

        return False

def _recorded_commit_identity(ctx, surface: str, row: dict, retry_key: str) -> dict:
    attempt = getattr(ctx, "_pending_review_attempt", None)
    return dict(
        task_id=str(getattr(attempt, "task_id", "") or getattr(ctx, "task_id", "") or ""),  # the wave's own custody
        surface=surface, row=row, retry_key=retry_key,
        root_task_id=str(getattr(attempt, "root_task_id", "") or ""),
        review_contract=str(getattr(attempt, "review_contract_fingerprint", "") or ""),
    )


def recorded_commit_assignment(ctx, surface: str, row: dict):
    from ouroboros.review_operation import recorded_review_assignment
    from ouroboros.review_custody import custody_root

    return recorded_review_assignment(custody_root(ctx), **_recorded_commit_identity(
        ctx, surface, row, str(getattr(ctx, "_current_review_retry_key", "") or "")))


def recorded_commit_plan(ctx) -> dict:
    """Restore the paid combined seats, including which questions count for quorum.

    The write-ahead roster owns composition before a result record exists. Older
    rows use their original ledger's composition, never the current catalog.
    Missing physical source remains a custody gap in the normal collector.
    """
    from ouroboros.review_ledger import ledger_root, load_record, seat_parts
    from ouroboros.review_records import ReviewSlot

    attempt = getattr(ctx, "_pending_review_attempt", None)
    recorded = load_record(ledger_root(ctx), str(getattr(attempt, "review_record_id", "") or "")) or {}
    prior = {str(row.get("seat_id") or ""): row for row in recorded.get("rows") or []}
    rows = list(((getattr(ctx, "_review_frozen_rows", {}) or {}).get("multi_model_review", {}) or {}).values())
    slots, composition = [], []
    for row in rows:
        try:
            _request, slot = recorded_commit_assignment(ctx, "multi_model_review", row)
        except (OSError, ValueError, KeyError, TypeError):
            slot = ReviewSlot(slot_id=str(row.get("slot_id") or ""),
                              model=str(row.get("model_id") or row.get("model") or "unknown"),
                              route=str(row.get("route") or "api_chat"))
        slots.append(slot)
        composition.append({**prior.get(slot.slot_id, {}), **dict(row.get("review_plan") or {})})
    plan = {key: [getattr(slot, attr) for slot in slots] for key, attr in (
        ("models", "model"), ("slot_ids", "slot_id"), ("routes", "route"), ("efforts", "effort"),
        ("session_targets", "session_target"), ("session_profiles", "session_profile"),
        ("subagent_ids", "subagent_id"), ("use_local", "use_local"), ("session_accesses", "session_access"))}
    plan.update(retrieves=[slot.retrieves for slot in slots],
                parts=[tuple(row.get("parts") or seat_parts(slot)) for slot, row in zip(slots, composition)],
                additional=[bool(row.get("additional")) for row in composition],
                brief_shas=[str(row.get("brief_sha") or "") for row in composition])
    # The record's declared per-item severity policy (a skill wave's checklist rule) rides the rejoin plan as
    # it rode the dispatching one: fresh, explicit-rejoin and background settlement read the same policy.
    policy = ((recorded.get("brief") or {}).get("checklist") or {}).get("severity_policy")
    if isinstance(policy, dict) and policy:
        plan["severity_policy"] = dict(policy)
    return plan


def recorded_change_panel(ctx, attempt):
    """The original pool composition before any current catalog lookup."""
    from types import SimpleNamespace
    from ouroboros.review_custody import prepare_frozen_review_reconciliation
    from ouroboros.review_ledger import ledger_root, load_record
    from ouroboros.reviewer_slot_config import PoolSeat
    from ouroboros.tools.review_change import ComposedPanel
    from ouroboros.review_records import ReviewSlot

    collector = SimpleNamespace(**vars(ctx))
    collector._pending_review_attempt = attempt
    collector._current_review_retry_key = attempt.review_retry_key
    prepare_frozen_review_reconciliation(collector, attempt)
    plan = recorded_commit_plan(collector)
    seats = []
    for index, slot_id in enumerate(plan["slot_ids"]):
        saved = collector._review_frozen_rows["multi_model_review"][slot_id]
        try:
            _request, slot = recorded_commit_assignment(collector, "multi_model_review", saved)
        except (OSError, ValueError, KeyError, TypeError):
            slot = ReviewSlot(slot_id=slot_id, model=plan["models"][index], route=plan["routes"][index])
        seats.append(PoolSeat(slot, plan["parts"][index], plan["additional"][index]))
    prior = load_record(ledger_root(ctx), str(attempt.review_record_id or "")) or {}
    return ComposedPanel(tuple(seats), True, dict(prior.get("panel") or {}))


def collect_recorded_commit_actor(ctx, surface: str, slot_id: str, retry_key: str) -> dict:
    """Join a live original seat; otherwise collect its recorded source/token."""
    from dataclasses import asdict
    from ouroboros.delegate_custody import invocation_record
    from ouroboros.review_live_join import preparing_live_review_assignment, recorded_live_review_attempt
    from ouroboros.review_operation import collect_recorded_review_actor, recorded_review_assignment, recover_review_producer
    from ouroboros.review_substrate import run_review_request
    from ouroboros.review_custody import custody_root

    row = ((getattr(ctx, "_review_frozen_rows", {}) or {}).get(surface, {}) or {}).get(slot_id, {})
    identity = _recorded_commit_identity(ctx, surface, row, retry_key)
    root = custody_root(ctx)
    try:
        request, slot = recorded_review_assignment(root, **identity)
    except FileNotFoundError:
        assignment = preparing_live_review_assignment(**identity)
        if assignment is None:
            return collect_recorded_review_actor(root, **identity, usage_ctx=ctx)
        request, slot = assignment
    except (OSError, ValueError, KeyError, TypeError):
        return collect_recorded_review_actor(root, **identity, usage_ctx=ctx)
    recovered = recover_review_producer(root, request, slot, row)
    if recovered is not None and recovered.operation_state != "in_flight":
        from ouroboros.review_seat_outcome import publish_seat_execution

        publish_seat_execution(request, slot, recovered, ctx)
        return asdict(recovered)
    token = str(row.get("pending_invocation_id") or (row.get("usage") or {}).get("pending_invocation_id")
                or (row.get("recovery_binding") or {}).get("pending_invocation_id") or "")
    invocation = invocation_record(root, token) if token else None
    live = recorded_live_review_attempt(request, slot, row)
    if live is None and (not invocation or invocation.get("state") != "pending"):
        return collect_recorded_review_actor(root, **identity, usage_ctx=ctx)
    # The existing custody drain joins a matching local worker, including API
    # work detached by Pause. Cold collection above remains nonblocking.
    # Only an unresolved original start can require a POST, with the existing
    # invocation token and its raw recorded request. No replacement is admitted.
    request.reconcile_only = True
    collector = ctx
    if live is not None:
        # Coordinator defaults must not fill original unknowns from today's author.
        collector = copy.copy(ctx)
        collector._review_live_join_only = True
        collector.task_attempt = request.task_attempt
        collector.task_metadata = {**dict(getattr(ctx, "task_metadata", {}) or {}),
                                   "deadline_at": request.deadline_at,
                                   "root_task_id": request.reconciliation_identity["root_task_id"]}
    result = run_review_request(request, slots=[slot], drive_root=root, usage_ctx=collector)
    if getattr(collector, "_review_custody_lost", False):
        ctx._review_custody_lost = True
    return (result.actors or [{}])[0]


def prepare_recorded_commit_review(ctx, *, target_repo, layer: str,
                                   frozen: Any = None, review_rebuttal: str = "") -> tuple:
    """Restore a paid packet from its roster without fresh source/catalog assembly.

    The roster and recorded plan own the questions owed by the existing physical
    operations. An empty roster has nothing to collect; no new seat is invented.
    The wave's saved dispute input (or its explicit gap) is restored first
    (``review_history.prepare_history``), so the collected record never names
    today's history heads as its predecessors.
    """
    from ouroboros.review_history import prepare_history

    prepare_history(ctx, frozen, review_rebuttal)
    rows = ((getattr(ctx, "_review_frozen_rows", {}) or {}).get("multi_model_review", {}) or {})
    if not rows:
        return None, None, True
    plan = recorded_commit_plan(ctx)
    ctx._last_triad_models = list(plan["models"])
    return {"prompt": "", "stable_prefix_len": 0, "models": plan["models"],
            "routes": plan["routes"], "row_plan": plan, "session_task": "",
            "target_repo": target_repo, "layer": layer}, None, False


# A settled attempt's lifecycle projection, per tool: a commit's outcome returned before Git
# (``commit_gate._return_commit_feedback``), a review's closed wave (``settle_attempt``).
_SETTLED_PROJECTION = {"commit_reviewed": ("reviewed", "review_only"), "vcs_commit_reviewed": ("reviewed", "review_only")}


def _merge_seat(rows, incoming):
    """One collected seat onto its slot's row. A terminal outcome of the row's own operation, with no start token
    left in its usage, closes the start token that operation left on the row (#2587): a merged token would keep a
    settled seat in custody forever. A pending, lost, other-operation or unidentified outcome keeps its custody."""
    from ouroboros.review_custody import _row_is_pending

    key = str(incoming.get("slot_id") or incoming.get("slot") or "")
    old = rows.get(key) or {}
    if old and not _row_is_pending(old) and _row_is_pending(incoming):
        return
    if old.get("operation_id") and incoming.get("operation_id") not in (None, "", old["operation_id"]):
        return
    if old.get("superseded_operation_ids") and incoming.get("operation_id") != old.get("operation_id"):
        return  # a seat reopened by an explicit continuation takes only its successor's outcome
    merged = {**old, **incoming}
    usage = incoming.get("usage") if isinstance(incoming.get("usage"), dict) else {}
    if (old.get("operation_id") and incoming.get("operation_id") == old["operation_id"]
            and not _row_is_pending(incoming) and not usage.get("pending_invocation_id")):
        merged.pop("pending_invocation_id", None)
    rows[key] = merged


def _settled_row(current, seats, record):
    """``current`` with each collected seat merged: a scope seat onto the scope roster, every other seat onto the
    triad (``_merge_seat``). Once every counted seat is terminal with no start token holding (a settled seat's
    leftover token holds nothing, ``review_state_custody._start_token_open``), a row still ``reviewing``
    takes its tool's settled projection (``_SETTLED_PROJECTION``): counted completion ends the review phase, while an
    additional seat still running keeps its own physical custody (the late flag). Pure, so a row it leaves equal
    needs no save."""
    from dataclasses import replace
    from ouroboros.review_custody import _row_is_pending
    from ouroboros.review_ledger import counted_pending
    from ouroboros.review_state import _review_roster_row_is_pending, _utc_now, counted_roster_rows

    rows = {str(row.get("slot_id") or ""): copy.deepcopy(row) for row in current.triad_raw_results}
    wrapper = copy.deepcopy(current.scope_raw_result) if isinstance(current.scope_raw_result, dict) else {}
    listed = isinstance(wrapper.get("raw_results"), list)
    scope = {str(row.get("slot_id") or row.get("slot") or ""): row for row in (
        wrapper["raw_results"] if listed else [wrapper] if wrapper else []) if isinstance(row, dict)}
    for incoming in seats:
        roster = scope if incoming.get("surface") == "scope_review" and scope else rows
        _merge_seat(roster, {key: value for key, value in incoming.items() if key != "surface"})
    if listed:
        wrapper["raw_results"] = list(scope.values())
    elif wrapper:  # a historical single scope row is its own wrapper
        wrapper = next(iter(scope.values()), wrapper)
    roster = [*rows.values(), *scope.values()]
    status, phase = ((_SETTLED_PROJECTION.get(current.tool_name, ("reviewed", "review")))
                     if roster and current.status == "reviewing" and not counted_pending(roster)
                     and not any(_review_roster_row_is_pending(row) for row in counted_roster_rows(roster))
                     else (current.status, current.phase))
    return replace(current, triad_raw_results=list(rows.values()), scope_raw_result=wrapper,
                   late_result_pending=any(_row_is_pending(row) for row in roster), status=status, phase=phase,
                   finished_ts=current.finished_ts or (_utc_now() if status != current.status else ""),
                   review_record_id=record["record_id"])


def settle_collected_attempt(root, attempt, seats, record):
    """Merge each collected seat against the current attempt under its state lock (``_settled_row``). The record's
    verdict and every recorded answer stay as they are, and a collection that changes nothing (a periodic replay of a
    wave still waiting, a seat already merged) writes nothing (``ReviewStateMutation``)."""
    from ouroboros.review_state import ReviewStateMutation, update_state

    def mutate(state):
        for index, current in enumerate(state.attempts):
            if (current.repo_key, current.tool_name, current.task_id, current.attempt) != (
                    attempt.repo_key, attempt.tool_name, attempt.task_id, attempt.attempt):
                continue
            state.attempts[index] = _settled_row(current, seats, record)
            # The saved bytes decide: the file keeps each seat's named evidence as a reference to its producer
            # payload, so a seat collected again with its full inventory differs in memory and not on disk.
            return ReviewStateMutation(None, changed=False)
        raise LookupError("the paid attempt disappeared before collection was saved")
    update_state(root, mutate)


def settled_record(ledger, attempt, record=None, *, rows_closed=True):
    """The attempt's bound review record, read without a lock from the ``ledger`` drive (or the ``record`` a caller
    already read by that id), once its counted seats positively settled, whatever their verdict (an additional seat
    may still run): ``None`` unless that exact record is this attempt's (its task, and for a commit its staged
    fingerprint), names at least one counted seat, each under its operation and none open, and an attempt that still
    keeps its rows holds those same counted operations closed. A missing, foreign, unreadable, empty or reopened
    record, or a counted seat still open or unbound on either side, proves nothing. ``rows_closed=False`` asks for
    the record's own proof alone: the caller proves each of the attempt's rows itself (``settled_seats``)."""
    from ouroboros.review_custody import _row_is_pending
    from ouroboros.review_ledger import counted_pending, load_record
    from ouroboros.review_state import _attempt_review_roster_rows, counted_roster_rows

    record_id = str(getattr(attempt, "review_record_id", "") or "")
    try:
        record = record if record is not None and record.get("record_id") == record_id else (
            load_record(ledger, record_id) if record_id else None)
    except (OSError, ValueError):
        return None
    seats = counted_roster_rows([row for row in (record or {}).get("rows") or [] if isinstance(row, dict)])
    if (not record or record.get("task_id") != attempt.task_id or not seats or counted_pending(record.get("rows") or [])
            or not all(seat.get("operation_id") for seat in seats)
            or (attempt.tool_name in _SETTLED_PROJECTION
                and (record.get("fingerprints") or {}).get("binding") != attempt.pre_review_fingerprint)):
        return None
    confirmed = {str(seat.get("seat_id") or ""): str(seat["operation_id"]) for seat in seats}
    own = counted_roster_rows(_attempt_review_roster_rows(attempt))
    if rows_closed and not attempt.raw_stripped and (not own or any(_row_is_pending(row) or confirmed.get(
            str(row.get("slot_id") or row.get("slot") or "")) != str(row.get("operation_id") or "") for row in own)):
        return None
    return record


def settled_seats(ledger, attempt, record=None):
    """A persisted attempt's own seats that its bound settled record (``settled_record``, ``record`` when the caller
    already read it) confirms closed, each under the operation its record seat names: ``(record, seats)``. A seat the
    row already closed stands as it is. One the row still holds open (a row write an earlier collection's busy
    shared-state lock deferred, #2779) stands only with its own operation's durable terminal outcome, never on the
    record's word alone: its producer's retained outcome (``_durable_outcome``), else, for a delegated seat whose
    outcome still names its start, that start's own custody record (``custody_closed_seats``). Every counted seat must
    stand; an additional seat that does not (a running adviser) keeps its custody and is left out. ``None`` when the
    record proves nothing, a counted seat does not stand, or the attempt keeps no rows of its own."""
    from ouroboros.review_custody import _row_is_pending
    from ouroboros.review_ledger import seat_open
    from ouroboros.review_state import counted_roster_rows

    record = settled_record(ledger, attempt, record, rows_closed=False)
    if record is None or attempt.raw_stripped:
        return None
    confirmed = {str(seat.get("seat_id") or ""): str(seat.get("operation_id") or "")
                 for seat in record.get("rows") or [] if not seat_open(seat)}
    wrapper = attempt.scope_raw_result if isinstance(attempt.scope_raw_result, dict) else {}
    scope = wrapper.get("raw_results") if isinstance(wrapper.get("raw_results"), list) else [wrapper] if wrapper else []
    rows = [{**row, "surface": surface} for surface, group in (
        ("multi_model_review", attempt.triad_raw_results or []), ("scope_review", scope)) for row in group
        if isinstance(row, dict)]

    def seat_key(row):
        return row["surface"], str(row.get("slot_id") or row.get("slot") or "")
    counted, seats, held = counted_roster_rows(rows), [], set()
    for row in rows:
        operation = str(row.get("operation_id") or "")
        if not operation or confirmed.get(seat_key(row)[1]) != operation:
            if row in counted:
                return None
            continue
        seat = row if not _row_is_pending(row) else _durable_outcome(ledger, attempt, row)
        if seat is None:
            held.add(seat_key(row))
        else:
            seats.append(seat)
    if held:
        seats += [seat for seat in custody_closed_seats(ledger, attempt, record, {seat_key(seat) for seat in seats})
                  if seat_key(seat) in held]
    standing = {seat_key(seat) for seat in seats}
    if not counted or any(seat_key(row) not in standing for row in counted):
        return None
    return record, seats


_DURABLY_TERMINAL = frozenset({"settled", "late_settled", "not_dispatched"})


def _durable_outcome(ledger, attempt, row):
    """The terminal outcome a seat its attempt row still holds open has on disk: the producer's own retained outcome
    under the seat's recorded assignment (``review_operation.recover_review_producer``, read as
    ``review_pause._left_roster_settled`` reads it), the very actor its wave's collection merges, terminal under the
    row's own operation. Local reads only, never a send, a join or a remote poll; ``None`` for a missing, unreadable,
    differently bound, lost or still running outcome."""
    from dataclasses import asdict

    from ouroboros.review_operation import recorded_review_assignment, recover_review_producer

    surface, own = row["surface"], {key: value for key, value in row.items() if key != "surface"}
    try:
        request, slot = recorded_review_assignment(
            ledger, task_id=str(attempt.task_id or ""), surface=surface, row=own,
            retry_key=str(attempt.review_retry_key or ""), root_task_id=str(attempt.root_task_id or ""),
            review_contract=str(attempt.review_contract_fingerprint or ""))
        actor = recover_review_producer(ledger, request, slot, own)
    except Exception:
        log.debug("review seat %s: its durable outcome is unreadable; the row stays held", own.get("slot_id"),
                  exc_info=True)
        return None
    if (actor is None or str(actor.operation_state or "") not in _DURABLY_TERMINAL
            or str(actor.operation_id or "") != str(own.get("operation_id") or "")):
        return None
    return {**asdict(actor), "surface": surface, "slot_id": str(own.get("slot_id") or own.get("slot") or "")}


def late_write_check(ctx, phase, record_id, triad, scope):
    """The test a ``late_wait`` write (``commit_gate._record_commit_attempt``) applies to its row under the state lock
    (``review_state_custody.late_write_outlived``). Its bound record is read here, once, outside that lock: for a row
    compacted out of the recent history, the operations that settled record confirms closed stand for its rows."""
    from ouroboros.review_ledger import ledger_root, load_record, seat_open
    from ouroboros.review_state import late_write_outlived

    if phase != "late_wait":
        return lambda existing: False
    try:
        record = load_record(ledger_root(ctx), str(record_id)) if record_id else None
    except (OSError, ValueError):
        record = None

    def outlived(existing):
        bound = record if record and existing is not None and existing.review_record_id == record.get("record_id") else None
        settled = settled_record(None, existing, bound) if bound and existing.raw_stripped else None
        operations = None if settled is None else {
            (str(seat.get("seat_id") or ""), str(seat.get("operation_id") or "")) for seat in settled.get("rows") or []
            if seat.get("operation_id") and not seat_open(seat)
            and "not_dispatched" not in (seat.get("status"), seat.get("operation_state"))}
        return late_write_outlived(existing, triad, scope, operations)
    return outlived


def repair_settled_attempt(root, ledger, attempt):
    """The exact repair of one persisted attempt a consumer would otherwise hold or refuse: a row an earlier
    collector settled without its lifecycle projection, or with a closed seat's start token still on it. The bound
    record is read here from ``ledger``, outside the shared-state lock (``settled_seats``); the writer then rechecks
    the current row in ``root`` at its own mutation. A row that already says what its record proves (a settled row
    beside a still-running adviser) is returned as read, with no lock and no save. The row as saved, or ``None``
    when nothing proves it settled."""
    from ouroboros.review_state import load_state

    found = settled_seats(ledger, attempt)
    if found is None:
        return None
    if _settled_row(attempt, found[1], found[0]) == attempt:
        return attempt
    try:
        settle_collected_attempt(root, attempt, found[1], found[0])
        return load_state(pathlib.Path(root)).latest_attempt_for(
            repo_key=attempt.repo_key, tool_name=attempt.tool_name, task_id=attempt.task_id, attempt=int(attempt.attempt or 0))
    except (OSError, LookupError, ValueError):
        log.warning("settled review attempt %s could not be repaired", attempt.attempt, exc_info=True)
        return None


def reconcile_collected_attempt(root, ledger, attempt, record):
    """What a collection owes the attempt row before it returns ``record`` settled (``review_collection.collect_wave``):
    the exact repair (``repair_settled_attempt``) of a row an earlier writer left open beside it, a row write a busy
    shared-state lock deferred included (#2779), from the seats its own operations' durable outcomes prove
    (``settled_seats``). Nothing proven, or a row that already says it, writes nothing and takes no lock; a lock still
    busy raises ``ReviewStateLockError`` for the caller to report."""
    found = settled_seats(ledger, attempt, record)
    if found is not None and _settled_row(attempt, found[1], found[0]) != attempt:
        settle_collected_attempt(root, attempt, found[1], found[0])


def _invocation_closed(root, attempt, row):
    """What the custody record of an in-flight row's own start token proves (``delegate_custody_current``):
    ``("not_dispatched", facts)`` for a definite refusal, ``("late_settled", facts)`` for a bound run custody holds
    terminal, else ``None`` (no token or operation, a pending, unbound, unknown or foreign invocation, a run still
    going, an unreadable store)."""
    from ouroboros import delegate_custody_current as current
    from ouroboros.delegate_custody import TERMINAL_STATES

    token, operation = str(row.get("pending_invocation_id") or "").strip(), str(row.get("operation_id") or "")
    if not token or not operation or str(row.get("operation_state") or "") != "in_flight":
        return None
    named = {"task_id": str(attempt.task_id or ""), "slot_id": str(row.get("slot_id") or row.get("slot") or ""),
             "operation_id": operation}
    try:
        fate = current.invocation(root, token, lambda _root, request: {key: str(request.get(key) or "") for key in named})
        if not fate or any(fate.get(key) != value for key, value in named.items()):
            return None
        if fate.get("state") == "failed_definite":
            return "not_dispatched", {"invocation_id": token, "state": "failed_definite"}
        entry = current.run(root, fate["run_id"]) if fate.get("state") == "started" and fate.get("run_id") else None
    except Exception:
        log.debug("review start %s: its custody record is unreadable; the row stays held", token, exc_info=True)
        return None
    if entry is None or str(entry.terminal_state or "") not in TERMINAL_STATES:
        return None
    return "late_settled", {"invocation_id": token, "run_id": str(fate["run_id"]), "run_state": entry.terminal_state}


def custody_closed_seats(root, attempt, record, covered=()):
    """The seats of ``attempt`` left in flight that their invocation's own custody record closes (#2710), never age or
    task end: a definite refusal closes ``not_dispatched`` (no outcome exists, as the collector reads it); a run custody
    holds terminal closes ``late_settled`` with the outcome its bound ``record`` holds for that seat. A run whose answer
    the record still awaits keeps its row for the wave's own $0 collection (a rejoin, a left review's settlement); so
    do a pending, unbound, foreign or running invocation. ``covered``: ``(surface, slot)`` pairs already proven."""
    from ouroboros.review_ledger import seat_open

    outcomes = {str(seat.get("seat_id") or ""): seat for seat in (record or {}).get("rows") or [] if isinstance(seat, dict)}
    wrapper = attempt.scope_raw_result if isinstance(attempt.scope_raw_result, dict) else {}
    scope = wrapper.get("raw_results") if isinstance(wrapper.get("raw_results"), list) else [wrapper] if wrapper else []
    rows = [("multi_model_review", row) for row in attempt.triad_raw_results or []] + [("scope_review", row) for row in scope]
    closed = []
    for surface, row in [(surface, row) for surface, row in rows if isinstance(row, dict)]:
        slot = str(row.get("slot_id") or row.get("slot") or "")
        proof = None if (surface, slot) in covered else _invocation_closed(root, attempt, row)
        if proof is None:
            continue
        state, facts = proof
        seat = {**row, "surface": surface, "late_result_pending": False, "custody_closure": facts, "usage": {
            key: value for key, value in (row.get("usage") or {}).items() if key != "pending_invocation_id"}}
        outcome = outcomes.get(slot) or {}
        if state == "not_dispatched":
            closed.append({**seat, "status": "not_dispatched", "operation_state": "not_dispatched", "failure_code": "",
                           "error": "The daemon definitively refused this reviewer's start."})
        elif outcome and not seat_open(outcome) and "not_dispatched" not in (outcome.get("status"), outcome.get("operation_state")):
            closed.append({**seat, "operation_state": "late_settled", **{key: outcome[key] for key in (
                "status", "failure_code", "transport_status", "reported_cause") if key in outcome}})
    return closed


def repair_held_attempts(root, state, ledger=None):
    """``repair_settled_attempt`` for every attempt ``state`` still holds active, in ONE locked write: the checkout
    pass's repair before it judges custody (``review_checking_prune._review_rounds``). Each bound record is read here,
    outside the shared-state lock (``settled_seats``, from ``ledger``, default ``root``), and a seat left in flight
    also by its invocation's own custody record (``custody_closed_seats``); inside the lock, an attempt that changed
    since it was read is left for the next pass, so a writer in between wins. Returns the state as saved, or ``state``
    itself when nothing is proven or the write failed (those rounds stay held this pass); a pass that proves nothing
    new writes nothing."""
    from ouroboros.review_ledger import load_record
    from ouroboros.review_state import ReviewStateMutation, update_state
    from ouroboros.review_state_records import _attempt_identity_tuple

    ledger, plans = root if ledger is None else ledger, []
    for attempt in state.get_active_attempts():
        try:
            record = load_record(ledger, attempt.review_record_id) if attempt.review_record_id else None
        except (OSError, ValueError):
            continue  # an unreadable record proves nothing
        found = settled_seats(ledger, attempt, record)
        seats = list(found[1]) if found is not None else []
        seats += custody_closed_seats(root, attempt, record, {
            (seat["surface"], str(seat.get("slot_id") or seat.get("slot") or "")) for seat in seats})
        bound = found[0] if found is not None else {"record_id": attempt.review_record_id}
        if seats and _settled_row(attempt, seats, bound) != attempt:
            plans.append((_attempt_identity_tuple(attempt), attempt, (bound, seats)))
    if not plans:
        return state

    def mutate(current):
        changed = False
        for identity, read, (record, seats) in plans:
            index = next((at for at, row in enumerate(current.attempts) if _attempt_identity_tuple(row) == identity), None)
            if index is not None and current.attempts[index] == read:
                current.attempts[index] = _settled_row(read, seats, record)
                changed = True
        return ReviewStateMutation(None, changed=changed)
    try:
        return update_state(pathlib.Path(root), mutate)
    except Exception:
        log.warning("held review attempts were not repaired this pass; they stay held", exc_info=True)
        return state


def repaired_commit_source(ctx, source):
    """``source`` after its exact repair (``repair_settled_attempt``), for a resolver about to refuse its phase."""
    from ouroboros.review_ledger import ledger_root

    return repair_settled_attempt(pathlib.Path(ctx.drive_root), ledger_root(ctx), source) or source


def commit_reference_source(ctx, reference, state, source, *, locked):
    """The attempt row whose evidence an author continuation's commit ``reference`` names, for
    ``commit_gate.resolve_commit_review_reference``: the row the reference names while the bounded hot history holds it,
    else the exact evidence the ledger keeps for it (``recorded_commit_source``; ``ReviewState._trim_attempt_history``
    keeps paid and verdict rows and evicts free outcomes, the author's own new rows included, #2830), where the paid
    wave a free replay collected stands under that wave's OWN identity while the reference keeps its own
    (``bound_reference``, ``continued_attempt``). Outside the state lock a row still ``reviewing`` first gets its exact
    repair (``repaired_commit_source``, a row a busy lock left open beside its settled record included, #2779); inside
    it nothing is read again, and a row it no longer holds stands only as the source this same call bound outside for
    this very reference. ``None`` when nothing exact remains."""
    if source is not None and reference.get("review_record_id") and source.review_record_id != reference["review_record_id"]:
        source = None  # the number names another wave now: the evicted outcome's number was allocated again
    if locked:
        return source if source is not None else (
            getattr(ctx, "_author_commit_source", None) if bound_reference(ctx, reference) else None)
    source = recorded_commit_source(ctx, reference, state) if source is None else source
    return repaired_commit_source(ctx, source) if source is not None and source.status == "reviewing" else source


_REFERENCE_IDENTITY = ("repo_key", "task_id", "tool_name", "attempt", "pre_review_fingerprint", "review_record_id")


def bound_reference(ctx, reference):
    """Whether ``reference`` is the very one this call resolved outside the state lock (``ctx._author_commit_reference``)
    to the evidence row it bound there (``ctx._author_commit_source``): the same returned identity, and that row the
    same review of the same staged bytes, under its own attempt number (a free replay's paid wave keeps its own)."""
    bound, source = getattr(ctx, "_author_commit_reference", None), getattr(ctx, "_author_commit_source", None)
    return (source is not None and isinstance(bound, dict) and isinstance(reference, dict)
            and all(bound.get(key) == reference.get(key) for key in _REFERENCE_IDENTITY)
            and (source.task_id, source.tool_name, source.pre_review_fingerprint, source.review_record_id)
            == tuple(reference.get(key) for key in ("task_id", "tool_name", "pre_review_fingerprint", "review_record_id")))


def continued_attempt(ctx, repo_key, tool_name, task_id):
    """The attempt number an author continuation's own new rows must exceed (``ReviewState.next_attempt_number``): those
    of the reference it continues and of the row whose evidence it reads, each when it is this task's. A number whose
    free row was evicted is otherwise allocated again, and the author's row would then carry the reference's very
    identity (a free replay's evicted row included, whose evidence is an older paid row)."""
    source = getattr(ctx, "_author_commit_source", None)
    if source is None:
        return 0
    reference = getattr(ctx, "_author_commit_reference", None)
    reference = reference if isinstance(reference, dict) else {}
    named = ((reference.get("repo_key"), reference.get("tool_name"), reference.get("task_id"), reference.get("attempt")),
             (source.repo_key, source.tool_name, source.task_id, source.attempt))
    return max((int(number or 0) for *identity, number in named if tuple(identity) == (repo_key, tool_name, task_id)),
               default=0)


def recorded_commit_source(ctx, reference, state):
    """The outcome an evicted free commit row returned, from the ledger record its reference names: a commit-gate record of
    the reference's task and staged binding, settled (a running or UNKNOWN wave is no terminal outcome). A paid wave keeps
    its attempt row, which is then the source under its own attempt number, unchanged and never copied; otherwise only
    a record that dispatched nothing (``NOT_DISPATCHED``, the ceiling's or another pre-send refusal) stands, as a
    seatless ``review_only`` outcome with the record's own verdict facts. Missing paid evidence is never read as a free
    refusal, and nothing here is written. ``None`` otherwise."""
    from ouroboros.review_ledger import VERDICT_NOT_DISPATCHED, ledger_root, load_record
    from ouroboros.review_state import CommitAttemptRecord, make_repo_key

    record_id = str(reference.get("review_record_id") or "")
    try:
        record = load_record(ledger_root(ctx), record_id) if record_id else None
    except (OSError, ValueError):
        return None
    root = str(((record or {}).get("subject") or {}).get("root") or "")
    if (not record or record.get("surface") != "commit_gate" or record.get("state") != "settled"
            or record.get("task_id") != reference.get("task_id") or (root and make_repo_key(pathlib.Path(root)) != reference.get("repo_key"))
            or (record.get("fingerprints") or {}).get("binding") != reference.get("pre_review_fingerprint")):
        return None
    paid = next((row for row in reversed(state.attempts) if row.paid and row.review_record_id == record_id
                 and row.task_id == record["task_id"] and row.tool_name == reference.get("tool_name")), None)
    verdict = record.get("verdict") or {}
    if paid is not None or verdict.get("aggregate") != VERDICT_NOT_DISPATCHED:
        return paid
    refusal = record.get("dispatch_refusal") if isinstance(record.get("dispatch_refusal"), dict) else {}
    return CommitAttemptRecord(
        ts=str(record.get("ts") or ""), commit_message="", status="reviewed", phase="review_only",  # no record keeps the message
        block_reason=str(refusal.get("kind") or VERDICT_NOT_DISPATCHED), task_id=str(record["task_id"]),
        root_task_id=str(record.get("root_task_id") or ""), repo_key=str(reference.get("repo_key") or ""),
        tool_name=str(reference.get("tool_name") or ""), attempt=int(reference.get("attempt") or 0),
        pre_review_fingerprint=str(reference.get("pre_review_fingerprint") or ""), review_record_id=record_id,
        critical_findings=list(verdict.get("critical_findings") or []),
        advisory_findings=list(verdict.get("advisory_findings") or []), finished_ts=str(record.get("ts") or ""))


def stripped_source_bound(ctx, source, *, locked):
    """Whether an attempt whose inline rows were compacted (``raw_stripped``) still names its exact review for an
    explicit author continuation: outside the state lock, its bound settled record (``settled_record``), which keeps
    every seat's retained source; inside the lock, only the source this same call bound outside it."""
    if locked:
        bound = getattr(ctx, "_author_commit_source", None)
        return bound is not None and (bound.repo_key, bound.tool_name, bound.task_id, bound.attempt) == (
            source.repo_key, source.tool_name, source.task_id, source.attempt)
    from ouroboros.review_ledger import ledger_root

    return settled_record(ledger_root(ctx), source) is not None


def repair_held_commit_attempts(ctx):
    """Before the overlap check reads them under its lock: this repository's active commit attempts with no counted
    seat open, the shape an earlier collector left when it settled the seats but not the row's projection or a
    closed seat's start token. Each gets its exact repair (``repair_settled_attempt``); anything unproven stays."""
    from ouroboros.review_ledger import counted_pending, ledger_root
    from ouroboros.review_state import _attempt_review_roster_rows, load_state, make_repo_key

    root = pathlib.Path(ctx.drive_root)
    try:
        held = [item for item in load_state(root).get_active_attempts(repo_key=make_repo_key(pathlib.Path(ctx.repo_dir)))
                if item.tool_name in _SETTLED_PROJECTION and item.review_record_id
                and _attempt_review_roster_rows(item) and not counted_pending(_attempt_review_roster_rows(item))]
    except Exception:
        log.warning("held commit attempts could not be read for their exact repair", exc_info=True)
        return
    for item in held:
        repair_settled_attempt(root, ledger_root(ctx), item)
