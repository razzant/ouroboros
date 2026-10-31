"""The custody side of one ``review_change`` wave.

``ouroboros/tools/review_change.py`` owns the request, the wave and its ledger
record; what the shared review state (``review_state``) knows about the wave lives
here: the write-ahead paid attempt row the per-task cycle ceiling counts, the
delegated start token checkpointed onto its exact reserved seat, the settled row
carrying the wave's final roster — and the rejoin. Before a rerun pays again, the
open operation of the SAME logical round on the SAME task (identity c,
``review_subject.review_retry_key``) is collected through the commit gate's
reconcile-only path (``review_custody``: settled seats replay, the exact pending
delegated invocation is rejoined, nothing new is sent). A rejoin reads the retained
checkout at the deterministic path of this task's round
(``review_subject.checkout_token``), so its custody attempt key and every
operation's recovery binding are those of the operation it rejoins, in this process
or after a restart; another task's wave of the round has its own path.
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
    stamp_paid_review_owner(attempt, paid=bool(getattr(attempt, "paid", False)))
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
    """The write-ahead paid fact at the wave's first physical dispatch, keyed by the
    REVIEWED root so the shared per-task-tree ceiling of (this root, ``review_change``)
    counts it; and the delegated start token's checkpoint onto its reserved seat
    (``review_state_custody.checkpoint_pending_review_invocation``), so a restart
    finds the exact invocation to rejoin. A rejoin writes onto the attempt row it
    collects, never a second paid row; its empty reservation leaves that row's
    roster as recorded."""
    from ouroboros.review_dispatch import ReviewPaidStamp
    from ouroboros.review_state import checkpoint_pending_review_invocation, make_repo_key, update_state

    holder = {"attempt": int(getattr(getattr(wave, "rejoin", None), "attempt", 0) or 0)}
    drive, repo_key = review_state_root(ctx), make_repo_key(wave.root)
    task_id = str(getattr(ctx, "task_id", "") or "")

    def _write() -> None:
        reserved = getattr(ctx, "_review_reserved_roster", None)
        reserved = reserved if isinstance(reserved, dict) else {}
        triad = copy.deepcopy(list(reserved.get("multi_model_review") or []))

        def _mutate(state: Any) -> None:
            number = holder["attempt"] or state.next_attempt_number(repo_key, _tool_name(wave), task_id)
            state.record_attempt(attempt_record(
                wave, ctx, status="reviewing", phase="review", paid=True, attempt=number,
                triad_raw_results=triad))
            holder["attempt"] = number

        update_state(drive, _mutate)

    def _checkpoint(*, surface: str, slot_id: str, operation_id: str, invocation_id: str) -> None:
        checkpoint_pending_review_invocation(
            drive, repo_key=repo_key, tool_name=_tool_name(wave), task_id=task_id, attempt=holder["attempt"],
            review_retry_key=wave.retry_key, surface=surface, slot_id=slot_id, operation_id=operation_id,
            invocation_id=invocation_id)

    ctx._review_paid_stamp = ReviewPaidStamp(_write, fail_closed=True)
    ctx._review_pending_invocation_checkpoint = _checkpoint
    return holder


def settle_attempt(ctx: Any, wave: Any, outcome: Dict[str, Any], payload: Dict[str, Any],
                   forensic: Dict[str, Any]) -> bool:
    """Close the attempt row this wave opened or rejoined (a review only; never a
    commit) with the wave's final roster: a seat still owed stays
    ``late_result_pending`` with its exact operation and start token, so a later
    rerun of the round — in this process or after a restart — finds what to collect."""
    number = (int((outcome.get("attempt") or {}).get("attempt") or 0)
              or int(getattr(getattr(wave, "rejoin", None), "attempt", 0) or 0))
    if number <= 0:
        return True
    from ouroboros.review_state import update_state

    verdict = dict(payload.get("verdict") or {})
    try:
        update_state(review_state_root(ctx), lambda state: state.record_attempt(attempt_record(
            wave, ctx, status="reviewed", phase="review", paid=True, attempt=number,
            late_result_pending=payload.get("state") == "pending",
            block_reason=str(outcome.get("block_reason") or ""),
            triad_raw_results=copy.deepcopy([row for row in forensic.get("triad_raw") or [] if isinstance(row, dict)]),
            degraded_reasons=[str(item) for item in verdict.get("degraded_reasons") or []])))
        return True
    except Exception:
        log.warning("review_change attempt row was not settled", exc_info=True)

        return False

def _recorded_commit_identity(ctx, surface: str, row: dict, retry_key: str) -> dict:
    attempt = getattr(ctx, "_pending_review_attempt", None)
    return dict(
        task_id=str(getattr(ctx, "task_id", "") or getattr(attempt, "task_id", "") or ""),
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


def prepare_recorded_commit_review(ctx, *, target_repo, blocking_review: bool, layer: str,
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
            "target_repo": target_repo, "blocking_review": blocking_review,
            "layer": layer}, None, False


def settle_collected_attempt(root, attempt, seats, record):
    """Merge each collected seat against the current attempt under its state lock."""
    from dataclasses import replace
    from ouroboros.review_custody import _row_is_pending
    from ouroboros.review_state import update_state

    def mutate(state):
        for index, current in enumerate(state.attempts):
            if (current.repo_key, current.tool_name, current.task_id, current.attempt) != (
                    attempt.repo_key, attempt.tool_name, attempt.task_id, attempt.attempt):
                continue
            rows = {str(row.get("slot_id") or ""): copy.deepcopy(row) for row in current.triad_raw_results}
            for incoming in seats:
                key = str(incoming.get("slot_id") or "")
                old = rows.get(key) or {}
                if old and not _row_is_pending(old) and _row_is_pending(incoming):
                    continue
                if old.get("operation_id") and incoming.get("operation_id") not in (None, "", old["operation_id"]):
                    continue
                rows[key] = {**old, **incoming}
            state.attempts[index] = replace(current, triad_raw_results=list(rows.values()),
                late_result_pending=any(_row_is_pending(row) for row in rows.values()),
                review_record_id=record["record_id"])
            return
        raise LookupError("the paid attempt disappeared before collection was saved")
    update_state(root, mutate)
