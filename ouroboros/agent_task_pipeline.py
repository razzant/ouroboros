"""Post-task result emission, root post-task work, and review context.

``emit_task_results`` derives the loop outcome once, emits the lifecycle and
usage events, registers the final answer in the durable outbox, stores the
result (``_store_task_result`` via ``write_task_result``, which stamps
``task_result_schema``) and dispatches the root-only post-task synthesis
(checkpointed by ``post_task_checkpoint``).

``total_rounds``/``prompt_tokens``/``completion_tokens`` on the stored result
come from the usage ledger (``reconstruct_task_cost``); the loop's own tally
rides ``loop_outcome.usage``. A capture the ledger could not read is ``None``
(unknown), never zero. A task exception is ``failure.kind = "runtime"``, not a
provider failure (``derive_loop_outcome``).
"""

from __future__ import annotations

import contextvars
import json
import copy
import functools
import logging
import pathlib
import threading
import time
from dataclasses import replace
from typing import Any, Callable, Dict, List

from ouroboros.finalization_timing import emit_post_answer_timing, finalization_child_scope
from ouroboros.observability import timed_phase
from ouroboros.context_timing import timed_section
from ouroboros.settings_integrity import copy_task_settings_context
from ouroboros.cost_projection import cost_projection, resolve_cost_pair
from ouroboros.task_results import (
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    lifecycle_named,
    load_task_result,
    write_task_result,
)
from ouroboros.artifacts import collect_task_artifact_records, merge_artifact_records
from ouroboros.outcomes import (
    EXECUTION_BEST_EFFORT,
    EXECUTION_FAILED,
    EXECUTION_INFRA_FAILED,
    EXECUTION_OK,
    REASON_OWNER_REQUESTED_FINALIZATION,
    WARN_DELEGATED_CUSTODY_UNRECONCILED,
    apply_receipt_absent_flag,
    artifact_bundle_from_result,
    build_verification_ledger,
    derive_loop_outcome,
    maybe_write_verification_artifact,
    normalize_outcome_axes,
)
from ouroboros.outcome_receipt_store import task_verification_receipts
from ouroboros.contracts.task_contract import build_task_contract
from ouroboros.subagents import effort_result_fields, envelope_from_task, substrate_result_fields
from ouroboros.subagent_messages import initiator_meta, subagent_message_meta
from ouroboros.utils import utc_now_iso, append_jsonl, truncate_review_artifact as _truncate_with_notice
from ouroboros.utils import in_worker_process
from ouroboros.post_task_checkpoint import (
    POST_TASK_SYNTHESIS_INFLIGHT as _POST_TASK_SYNTHESIS_INFLIGHT,
    POST_TASK_SYNTHESIS_LOCK as _POST_TASK_SYNTHESIS_LOCK,
    post_task_checkpoint_witness, begin_root_post_task, PostTaskCheckpointChanged,
    reserve_post_task_synthesis, release_post_task_synthesis, post_task_dispatch_evidence,
    is_root_post_task as _is_root_post_task,
    post_task_synthesis_is_open as _post_task_synthesis_is_open,
    post_task_synthesis_is_paused as _post_task_synthesis_is_paused,
    post_task_synthesis_is_terminal as _post_task_synthesis_is_terminal,
    root_checkpoint_roots as _root_checkpoint_roots,
    root_post_task_already_completed as _root_post_task_already_completed,
    set_root_post_task_checkpoint as _set_root_post_task_checkpoint,
    settle_terminal_projection as _settle_terminal_projection,
)
from ouroboros.skill_publish_result import apply_skill_publish_receipt_veto
from ouroboros.task_finalization import (
    TERMINAL_ORIGIN_HOST_NOTICE,
    build_sealed_final_package,
    set_terminal_host_notice,
    model_execution_projection,
    build_swarm_efficiency as _build_swarm_efficiency,  # moved (module ceiling); tests import it here
    deliver_final_message_live, prepare_terminal_send_event, register_final_answer_owed, stamp_root_final_phase,
    sealed_final_prompt_section, terminal_result_fields, terminal_notice_text,  # noqa: F401 -- the pipeline module keeps its historical import surface for the synthesis leaf
)
from ouroboros.dialogue_provenance import is_presence_task, presence_provenance_fields  # noqa: F401 -- the pipeline module keeps its historical import surface for the synthesis leaf
from ouroboros.presence_runner import build_presence_result_event

log = logging.getLogger(__name__)


# The task_summary cost fields live in `ouroboros/synthesis_cost_text.py`
# (extracted at this module's size ceiling); re-exported here because the
# tests name the helper on THIS surface.
from ouroboros.synthesis_cost_text import (  # noqa: F401,E402
    _summary_row_cost_fields,
)


def _publish_root_post_task_checkpoint(env, task, status, *, expected=None, stop_requested=None, **fields):
    """Retry only an already computed checkpoint, never a synthesis stage or provider call."""
    from ouroboros.store import StoreBusy, acquisition_budget
    from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
    from ouroboros.task_custody import fence_publication

    while True:
        fence_publication()
        if stop_requested and stop_requested():
            return None  # Current phase/paid receipts remain owed, never a false completion.
        try:
            with acquisition_budget(0):
                return _set_root_post_task_checkpoint(env, task, status, expected=expected, strict=True, **fields)
        except StoreBusy as exc:
            if exc.reason != "contention" or getattr(exc, "rollback_confirmed", None) is False:
                raise
            from ouroboros.model_wait import current_model_wait
            owner = current_model_wait()
            try:
                if owner is not None and owner.control_reason():
                    return None  # Stop/deadline ends waiting, not the already recorded paid work.
            except StoreBusy:
                pass  # A temporarily unreadable control is not a fabricated stop.
            # The setter has unwound its result/store/phase locks before this wait.
            time.sleep(ACQUISITION_SLICE_SEC)



def _run_post_task_processing_async(
    env: Any,
    task: Dict[str, Any],
    usage: Dict[str, Any],
    llm_trace: Dict[str, Any],
    drive_logs: pathlib.Path,
    *,
    blocking: bool = False,
    on_reflection: Callable[[Dict[str, Any] | None, Any], None] | None = None,
    sealed_final: Dict[str, Any] | None = None,
    event_queue: Any = None,
    expected_record: Dict[str, Any] | None = None,
    owner_liveness=None, claim_guard=None, stop_requested=None,
) -> None:
    """Run best-effort post-task work off the reply path.

    No stage authors a lesson or a knowledge note; that is the acting mind's
    ordinary ``knowledge_write``. ``on_reflection`` is a split Project
    root's canonical chooser entry; the promotion stage hands it the sealed final.
    An owner Pause ends the remaining stages at once, disclosed like a Stop;
    nothing is saved to resume.
    """
    task_snapshot = json.loads(json.dumps(task, ensure_ascii=False, default=str))
    trace_snapshot = json.loads(json.dumps(llm_trace, ensure_ascii=False, default=str))
    sealed_snapshot = (
        json.loads(json.dumps(sealed_final, ensure_ascii=False, default=str))
        if isinstance(sealed_final, dict) and sealed_final else None
    )

    from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

    base_scope = current_usage_scope()
    if base_scope is not None:
        post_scope = replace(
            base_scope, category="post_task", source="post_task_synthesis",
        )
    else:
        task_id = str(task_snapshot.get("id") or "")
        post_scope = UsageScope(
            drive_root=task_snapshot.get("budget_drive_root") or env.drive_root,
            task_id=task_id,
            root_task_id=str(task_snapshot.get("root_task_id") or task_id),
            parent_task_id=str(task_snapshot.get("parent_task_id") or ""),
            category="post_task",
            source="post_task_synthesis",
        )

    from ouroboros.model_wait import current_model_wait, task_model_wait_scope
    parent_wait = current_model_wait()
    role_overrides = copy.deepcopy(parent_wait.overrides) if parent_wait else {}
    # No reservation exists until all fallible, unpaid snapshot preparation is done.
    usage_snapshot = _pre_synthesis_usage_snapshot(env, task_snapshot, usage)
    post_task_key, phase_witness = None, None
    if _is_root_post_task(task_snapshot):
        post_task_key, phase_witness = begin_root_post_task(env, task_snapshot,
            expected_record=expected_record, owner_liveness=owner_liveness,
            claim_guard=claim_guard, stop_requested=stop_requested, event_queue=event_queue)
        if post_task_key is None:
            return None

    # The owner's will while the synthesis is ALREADY billing (Stop-now after
    # the loop returned): one local predicate, consulted before EACH paid
    # stage — the durable immediate cancel intent every stop ingress mints
    # and custody keeps open while this worker holds its in-flight key, or
    # the live task marker re-read (the entry snapshot cannot see a stop that
    # landed later). Stop intent stays separate from model-wait choices, whose
    # existing mailbox remains owned until this phase settles.
    intent_root = task_snapshot.get("budget_drive_root") or env.drive_root
    stage_task_id = str(task_snapshot.get("id") or task_snapshot.get("task_id") or "")
    def _release_owner() -> None:
        nonlocal post_task_key
        key, post_task_key = post_task_key, None
        release_post_task_synthesis(key)

    def _owner_stop_requested() -> bool:
        if stop_requested and stop_requested():
            return True
        if task.get("_skip_post_task_synthesis"):
            return True
        try:
            from ouroboros.cancel_intents import STOP_POLICY_IMMEDIATE, active_intent, stop_policy

            intent = active_intent(intent_root, stage_task_id)
        except Exception:
            log.debug("cancel-intent read failed for post-task stage gate", exc_info=True)
            return False
        return isinstance(intent, dict) and stop_policy(intent) == STOP_POLICY_IMMEDIATE

    def _run_scoped() -> None:
        checkpoint_status = "degraded"
        skipped: list[str] = []
        interrupted = current = ""
        try:
            # The free facts row precedes every paid stage, so neither Stop nor a
            # failed paid stage costs the card its facts; it is not a stage.
            _record_task_facts(env, task_snapshot, usage_snapshot, trace_snapshot, drive_logs)
            from ouroboros.llm import LLMClient

            llm_client = LLMClient()

            def _promotion() -> str:
                if is_presence_task(task_snapshot):
                    return ""
                failure = ""
                try:
                    from ouroboros.post_task_evolution import maybe_promote

                    maybe_promote(env, task_snapshot, sealed_snapshot, llm_client)
                except Exception as error:
                    propagate_paid_interruption(error)
                    # An ordinary chooser failure is this stage's own typed failure
                    # (degraded, nothing skipped); the Project root's canonical entry still runs.
                    failure = "promotion_failed"
                    log.warning("Post-task evolution promotion failed", exc_info=True)
                if on_reflection is not None:
                    on_reflection(sealed_snapshot, llm_client)
                return failure

            # All late model work belongs to this one scoped worker.  This keeps
            # the root checkpoint non-final until the fallback memory draft and
            # promotion have both stopped billing. No stage rewrites the
            # scratchpad: only a fold the mind accepts replaces its blocks (#918).
            stages: List[tuple[str, Callable[[], Any]]] = [
                ("memory_fallback_draft", lambda: _run_memory_fallback_draft(
                    env, task_snapshot, llm_client, drive_logs, trace_snapshot)),
                ("promotion", _promotion),
            ]
            from ouroboros.post_task_synthesis import (
                POST_TASK_INTERRUPT_KINDS, post_task_interruption, propagate_paid_interruption,
            )
            from ouroboros.usage_accounting import BudgetExceeded

            stage_errors = False
            for index, (name, run_stage) in enumerate(stages):
                if _owner_stop_requested():
                    interrupted = "owner_stopped"
                    skipped = [stage for stage, _run in stages[index:]]
                    break
                current = name
                try:
                    stage_reason = run_stage()
                    if isinstance(stage_reason, str) and stage_reason in POST_TASK_INTERRUPT_KINDS:
                        interrupted = stage_reason
                        skipped = [stage for stage, _run in stages[index + 1:]]
                        log.warning("Post-task paid stage %s interrupted for %s: %s",
                                    name, stage_task_id, interrupted)
                        break
                    if isinstance(stage_reason, str) and stage_reason:
                        # A returned ordinary failure is isolated to its stage: later
                        # stages still run, the checkpoint stays degraded (TZ-2 C3).
                        stage_errors = True
                        log.warning("Post-task stage %s failed for %s: %s", name, stage_task_id, stage_reason)
                except Exception as error:
                    # The adapters' own classifier: a control, the wallet and an
                    # unresolved attempt on any provider's chain stop later paid work.
                    try:
                        propagate_paid_interruption(error)
                    except BudgetExceeded:
                        interrupted = "budget_exhausted"
                    except Exception as control:
                        interrupted = post_task_interruption(control)
                    if interrupted:
                        skipped = [stage for stage, _run in stages[index + 1:]]
                        log.warning("Post-task paid stage %s interrupted for %s: %s",
                                    name, stage_task_id, interrupted)
                        break
                    stage_errors = True
                    log.warning("Post-task stage %s failed for %s", name, stage_task_id, exc_info=True)
            if not interrupted and not stage_errors:
                checkpoint_status = "completed"
        except Exception:
            log.warning("Post-task setup failed for %s", stage_task_id, exc_info=True)
        finally:
            if interrupted == "owner_pause":
                # The owner's Pause ends the upkeep at once, the stage it stopped
                # included, and nothing is saved to resume; Stop outranks it.
                interrupted = "owner_stopped" if _owner_stop_requested() else interrupted
                skipped = [current, *skipped]
            try:
                _publish_root_post_task_checkpoint(
                    env, task_snapshot, checkpoint_status, expected=phase_witness,
                    stop_requested=stop_requested,
                    stop_reason=f"{interrupted}:skipped={','.join(skipped)}" if interrupted else "",
                    event_queue=event_queue,
                )
            except Exception:
                log.warning("Post-task result publication remains owed for %s", stage_task_id, exc_info=True)
            _release_owner()  # The following owner-Pause census must see the phase's physical return.
            if not in_worker_process():
                # Pooled completion still owes child files and accepted inputs.
                # Its terminal owner cleans up after the first save attempt.
                from supervisor.terminal_delivery import cleanup_settled_owner_mailbox
                cleanup_settled_owner_mailbox(intent_root, stage_task_id, task_snapshot)
                from supervisor.owner_pause_control import late_phase_settled
                late_phase_settled(intent_root, stage_task_id)

    def _run_owned() -> None:
        with usage_scope(post_scope):
            if blocking:
                # Native post-task work retains its call owner after dialogue
                # admission closes; pooled owners stay in their worker process.
                if post_task_key is not None and parent_wait is not None and not parent_wait.worker_slot_held:
                    with _POST_TASK_SYNTHESIS_LOCK:
                        _POST_TASK_SYNTHESIS_INFLIGHT[post_task_key] = parent_wait
                # The task's optional absolute execution ceiling ends the solve
                # phase, not already-started post-work. Calendar deadlines and
                # logical call bounds remain checked before owner_control.
                prior_control = parent_wait.owner_control if parent_wait is not None else None
                if parent_wait is not None:
                    parent_wait.owner_control = lambda: "cancelled" if _owner_stop_requested() else None
                try:
                    _run_scoped()
                finally:
                    if parent_wait is not None:
                        parent_wait.owner_control = prior_control
            else:
                # A detached thread must not inherit its parent's closing scope.
                with task_model_wait_scope(task=task_snapshot, drive_root=env.drive_root,
                                           event_queue=event_queue, worker_slot_held=False,
                                           owner_control=lambda: "cancelled" if _owner_stop_requested() else None) as owner:
                    owner.overrides.update(role_overrides)
                    if post_task_key is not None:
                        with _POST_TASK_SYNTHESIS_LOCK:
                            _POST_TASK_SYNTHESIS_INFLIGHT[post_task_key] = owner
                    _run_scoped()

    def _run() -> None:
        try:
            _run_owned()
        finally:
            _release_owner()

    if blocking:
        _run()
        return None
    try:
        settings_context = contextvars.Context()
        copy_task_settings_context(settings_context)
        threading.Thread(target=settings_context.run, args=(_run,), daemon=True).start()
    except Exception:
        try:
            _publish_root_post_task_checkpoint(env, task_snapshot, "degraded", expected=phase_witness,
                event_queue=event_queue, stop_requested=stop_requested, stop_reason="post_task_worker_not_started")
        finally:
            _release_owner()
        raise
    return None


def recover_pending_root_post_task_synthesis(
    drive_root: Any, repo_dir: Any = None,
    *, exclude_task_ids: frozenset[str] = frozenset(),
    startup: bool = True, owner_liveness=None, claim_guard=None, stop_requested=None, event_queue=None,
) -> int:
    """Service existing synthesis debt; only proven absent/unsent work may dispatch.

    Startup retains its prior-worker proof. A steady-state caller supplies the
    queue's tri-state physical liveness and settlement interlock. A Pause an
    earlier version saved never runs: the steady-state pass ends it once
    (``end_saved_late_phase_pause``), startup leaves it for that pass, and a rest an
    earlier version left on its Resume is examined anyway. Paid or unknown legacy
    pending_once is indeterminate, just like a lost running phase. A member that
    waits on a recorded projection cause rests (``obligations.DutyRest``).
    """
    from types import SimpleNamespace
    from ouroboros.post_task_synthesis import end_saved_late_phase_pause
    from ouroboros.post_task_checkpoint import post_task_synthesis_in_flight
    from ouroboros.obligations import DutyRest, result_rows
    from ouroboros.install_identity import StoreForeign, require_task_roots, rest_foreign_member
    from ouroboros.terminal_projection import delivery_registry_stamp, rest_deferral, terminal_projection_owed

    root = pathlib.Path(drive_root).resolve(strict=False)
    if not startup and owner_liveness is None:
        return 0  # An empty/foreign process-local map cannot prove physical absence.
    recovered, rest = 0, DutyRest(root, "synthesis_recovery", name="synthesis", wake=("owner_resume",),
                                  inputs={"terminal_deliveries": delivery_registry_stamp(root)})
    for stored in result_rows(root, "synthesis", exclude=exclude_task_ids, ids=rest.examine):
        if stop_requested and stop_requested():
            break
        task_id = str(stored.get("task_id") or stored.get("id") or "")
        if not task_id:
            continue
        try:
            task = {**stored, "id": task_id, "root_task_id": str(stored.get("root_task_id") or task_id)}
            task.setdefault("budget_drive_root", str(root))
            require_task_roots(task)
            env = SimpleNamespace(repo_dir=pathlib.Path(repo_dir or task.get("repo_dir") or root.parent),
                                  drive_root=root, drive_path=lambda rel, _root=root: _root / rel)
            governing_root = _root_checkpoint_roots(env, task)[0]
            authority = (load_task_result(governing_root, task_id, strict=True) or {}
                         if governing_root != root else stored)
            here = rest if governing_root == root else None  # a rest keys on this root's row
            if post_task_synthesis_in_flight(governing_root, task_id):
                continue
            checkpoint = (authority or stored).get("root_phase_checkpoint") or {}
            phase = str(checkpoint.get("post_task_synthesis") or "")
            if not _post_task_synthesis_is_open(phase):
                waits = {} if terminal_projection_owed(task_id, authority) else {"on": "nothing_owed"}
                if not waits:
                    _settle_terminal_projection(governing_root, task_id, task=task, event_queue=event_queue, waits=waits)
                rest_deferral(here, task_id, waits)
                continue
            if owner_liveness is not None and owner_liveness(task_id) is not False:
                continue
            if _post_task_synthesis_is_paused(phase):
                if not startup and end_saved_late_phase_pause(
                        env, task, governing_root, authority, owner_liveness=owner_liveness,
                        claim_guard=claim_guard, stop_requested=stop_requested, event_queue=event_queue):
                    recovered += 1
                continue
            if not _is_root_post_task(task):
                rest_deferral(here, task_id, {"on": "not_root_post_task"})
                continue
            dispatched = phase == "running" or post_task_dispatch_evidence(governing_root, task_id, authority)
            if dispatched:
                key = reserve_post_task_synthesis(governing_root, task_id, owner_liveness=owner_liveness,
                    claim_guard=claim_guard, stop_requested=stop_requested)
                if key is None:
                    continue
                try:
                    updated = _publish_root_post_task_checkpoint(env, task, "degraded",
                        expected=post_task_checkpoint_witness(authority), event_queue=event_queue,
                        stop_requested=stop_requested, stop_reason=("restart_indeterminate_running" if startup
                            else "owner_ended_indeterminate_post_task"))
                    if updated and _post_task_synthesis_is_terminal(
                            (updated.get("root_phase_checkpoint") or {}).get("post_task_synthesis")):
                        recovered += 1
                finally:
                    release_post_task_synthesis(key)
                continue
            usage = {"cost": cost_projection(task)["accounted_upper_bound_usd"],
                     "rounds": ((task.get("loop_outcome") or {}).get("usage") or {}).get("total_rounds"),
                     "reason_code": str(task.get("reason_code") or ""), "outcome_axes": task.get("outcome_axes") or {}}
            trace = {"tool_calls": [], "reasoning_notes": [str(task.get("trace_summary") or "")],
                     "recovered_post_task_synthesis": True}
            _run_post_task_processing_async(
                env, task, usage, trace, root / "logs", blocking=False,
                sealed_final=build_sealed_final_package(task, str(task.get("result") or "")),
                expected_record=authority, owner_liveness=owner_liveness, claim_guard=claim_guard,
                stop_requested=stop_requested, event_queue=event_queue)
            fresh = load_task_result(governing_root, task_id, strict=True) or {}
            recovered += int((fresh.get("root_phase_checkpoint") or {}).get("post_task_synthesis") != phase)
        except StoreForeign as exc:
            rest_foreign_member(rest, task_id, stored, exc)
        except PostTaskCheckpointChanged:
            continue  # A newer Pause, completed result or task attempt owns the phase.
        except Exception:
            log.warning("Post-task recovery remains owed for %s", task_id, exc_info=True)
    rest.commit()
    return recovered


def _run_global_backlog_promotion_only(
    env: Any,
    task: Dict[str, Any],
    sealed_final: Dict[str, Any] | None,
    llm: Any,
) -> None:
    """A split Project root's entry to the canonical chooser (owner-enabled evolution).

    The project task's own chooser stays project-gated; this globalized view reaches
    the canonical drive with the run's sealed final and its provenance, never its
    project scope, initial text or memory, and that drive's opt-in, cadence counter,
    eligibility and authored backlog decide as for any root. Its cadence counter,
    observed call and any promotion request remain recorded. A failure
    raises to that stage's coordinator."""
    from ouroboros.consciousness_authority import consciousness_origin_metadata
    from ouroboros.post_task_evolution import maybe_promote

    global_task = {
        "id": str(task.get("id") or ""),
        "type": str(task.get("type") or "task"),
        "source": "project_scoped_global_improvement",
        # The origin survives the sanitized view: a campaign a consciousness tree
        # promotes stays inside the consciousness limits.
        "metadata": {"globalized_from_project_task": True, **consciousness_origin_metadata(task.get("metadata"))},
        # The eligibility probe reads the contract (disabled_tools), so the
        # globalized view keeps it: a level that may not evolve stays that way.
        **({"task_contract": dict(task["task_contract"])} if isinstance(task.get("task_contract"), dict) else {}),
    }
    maybe_promote(env, global_task, sealed_final if isinstance(sealed_final, dict) else None, llm)


def _attach_host_mutation_projection(
    env: Any,
    task: Dict[str, Any],
    llm_trace: Dict[str, Any],
) -> None:
    """Bind durable mutation evidence to the existing outcome trace seam."""
    from ouroboros.mutation_attribution import (
        attribution_task_id,
        load_mutation_evidence_projection,
        record_terminal_mutation_candidates,
    )

    task_id = str(task.get("id") or "").strip()
    root_task_id = str(task.get("root_task_id") or task_id).strip()
    task_ids = list(dict.fromkeys(item for item in (root_task_id, task_id) if item))
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    evidence_root = pathlib.Path(
        task.get("budget_drive_root")
        or metadata.get("budget_drive_root")
        or getattr(env, "budget_drive_root", None)
        or env.drive_root
    )
    # Only the owning root task refreshes its terminal candidate snapshot at
    # outcome derivation; a subagent never rewrites the root's evidence.
    snapshot_failed = False
    if task_id and task_id == root_task_id:
        try:
            if attribution_task_id(evidence_root, (task_id,)) == task_id:
                record_terminal_mutation_candidates(evidence_root, task_id)
        except Exception:
            snapshot_failed = True
            log.debug("terminal mutation snapshot failed for %s", task_id, exc_info=True)
    llm_trace.pop("mutation_attribution", None)
    for candidate in task_ids:
        projection = load_mutation_evidence_projection(evidence_root, candidate)
        if projection:
            if snapshot_failed:
                projection["uncommitted_remainder"] = {"state": "unknown", "count": None, "surfaces": []}
            llm_trace["mutation_attribution"] = projection
            return


def _derive_host_bound_loop_outcome(
    env: Any,
    task: Dict[str, Any],
    text: str,
    usage: Dict[str, Any],
    llm_trace: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive once from the current durable mutation-evidence binding.

    Delegated custody is not part of the outcome: an unreconciled run is a debt
    BESIDE the cause, kept on the row's own ``delegated_runs_unreconciled`` list and
    reconciliation envelope, which heal from the write side (#1583)."""
    _attach_host_mutation_projection(env, task, llm_trace)
    return apply_skill_publish_receipt_veto(derive_loop_outcome(text or "", usage, llm_trace), task, llm_trace)


def _stamp_project_room_pointer(task: Dict[str, Any], env: Any) -> None:
    """The project's last-result POINTER for a room's direct-chat root.

    A project ROOM's direct-chat turn writes a durable result carrying
    ``project_id`` but no letters home, so the pointer stayed empty and the
    room's own results were never offered as predecessors. Only the pointer is
    written here: no milestone, work-location row, digest or blocking
    post-processing - that exclusion is deliberate and stands.
    """
    from ouroboros.project_facts import resolve_project_id
    from ouroboros.tools.project_journal import record_project_last_result

    record_project_last_result(
        resolve_project_id(task), str(task.get("id") or ""),
        pathlib.Path(str(task.get("budget_drive_root") or env.drive_root)),
    )


def _custody_debt_event_fields(stored_result: Dict[str, Any]) -> Dict[str, Any]:
    """Carry the row's own custody debt list onto the live terminal event.

    The debt is that list, never the reason (#1583): the headline and the Reason
    line name it only while the record's own ``delegated_runs_unreconciled`` list is
    non-empty (docs/ARCHITECTURE.md, terminal composition), and the card reads this
    event rather than the durable row, so the event carries a copy of the stored
    list. A row holding no list states nothing about the debt: absence stays absence
    instead of becoming a second, guessed rule on one surface.
    """
    debt = stored_result.get("delegated_runs_unreconciled")
    return {"delegated_runs_unreconciled": list(debt)} if isinstance(debt, list) else {}


def _stamp_presence_terminal_facts(
    task: Dict[str, Any], usage: Dict[str, Any], llm_trace: Dict[str, Any], ctx: Any, reason_code: str,
) -> None:
    """Typed Presence facts the Host guard reads back from the durable row's metadata."""
    from ouroboros.presence_runner import presence_retry_proof, presence_unknown_outcome

    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    if reason_code == "resource_refusal_no_resend":
        proof = presence_retry_proof(task, usage, llm_trace, ctx)
        if proof:
            task["metadata"] = metadata = {**metadata, "presence_retry_proof": proof}
    # The loop's own no-resend predicate, stamped where the Host guard reads the row back: a
    # dispatched attempt left unresolved is not answered by the rail's text or a salvaged draft.
    unknown = presence_unknown_outcome(usage)
    if unknown:
        task["metadata"] = {**metadata, "presence_unknown_outcome": unknown}


def _wake_outcome_notice(text: str, loop_outcome: Dict[str, Any], custody_audit: Any = None) -> str:
    """The host notice a silent run's non-clean outcome publishes; ``""`` when it ran clean.

    Errors, forced or degraded outcomes and delegated custody still open in the current
    audit (``custody_audit.unreconciled``) are not hidden behind silence, yet the words the
    run selected as internal are never published for it: they stay the row's ``result``
    beside this notice (``terminal_host_notice``).
    """
    from ouroboros.project_dialogue import TASK_CAUSE_PHRASES

    execution = normalize_outcome_axes({"outcome_axes": loop_outcome.get("outcome_axes")}).get("execution") or {}
    status, reason = str(execution.get("status") or ""), str(loop_outcome.get("reason_code") or "")
    debt = custody_audit.get("unreconciled") if isinstance(custody_audit, dict) else None
    if status == EXECUTION_OK and not debt:
        return ""
    owed = f" {TASK_CAUSE_PHRASES[WARN_DELEGATED_CUSTODY_UNRECONCILED]}" if debt else ""
    kept = " Its internal result stays in the task details." if str(text or "").strip() else ""
    return (f"⚠️ This run chose to finish without a message, but it ended {status or 'unknown'} "
            f"({reason or 'unspecified'}).{owed}{kept}")


def _task_cost_fields(env: Any, task: Dict[str, Any]) -> Dict[str, Any]:
    """The terminal frames' cost authority (the usage ledger), never the loop's own tally."""
    try:
        from supervisor.state import reconstruct_task_cost

        task_cost_fields = reconstruct_task_cost(
            str(task.get("id") or ""), fields=True,
            drive_root=pathlib.Path(task.get("budget_drive_root") or env.drive_root),
            custody_root=env.drive_root,
        )
        from ouroboros.cost_projection import with_task_cost_presentation

        task_cost_fields = with_task_cost_presentation(task_cost_fields, task, env.drive_root)
    except Exception:
        log.error("Task cost authority unavailable at finalization", exc_info=True)
        task_cost_fields = {
            "cost_accounting_status": "unavailable", "cost_final": False,
            "cost_accounting_error": "ledger_unavailable",
            "accounted_upper_bound_usd": None, "total_rounds": None,
            "prompt_tokens": None, "completion_tokens": None,
            "reserved_usd": None, "unresolved_upper_bound_usd": None,
            "unknown_unmetered": None,
            "cost_presentation": None,
        }
    # SSOT cost naming (C2/ABI-3): the honest names on every terminal frame
    # this pipeline emits; the seam also strips any legacy alias spelling.
    from ouroboros.cost_projection import with_cost_aliases

    task_cost_fields = with_cost_aliases(task_cost_fields)
    if _is_root_post_task(task) and not _root_post_task_already_completed(env, task):
        task_cost_fields["cost_final"] = False
    return task_cost_fields


def _collect_review_evidence_bundle(
    env: Any, ctx: Any, task: Dict[str, Any], receipt_root: pathlib.Path, verification_receipts: Any,
) -> Dict[str, Any]:
    """Review evidence plus the captured task/historical inputs the durable result retains."""
    review_evidence: Dict[str, Any] = {}
    try:
        from ouroboros.review_evidence import collect_review_evidence

        review_evidence = collect_review_evidence(
            env.drive_root,
            task_id=str(task.get("id") or ""),
            repo_dir=getattr(env, "repo_dir", None),
            canonical_root=task.get("budget_drive_root"),
        )
    except Exception:
        log.debug("Failed to collect review evidence", exc_info=True)

    from ouroboros.post_task_synthesis import capture_task_inputs
    review_evidence["task_inputs"] = capture_task_inputs(
        ctx, task, task.get("budget_drive_root") or receipt_root, verification_receipts,
    )
    from ouroboros.context_input_selection import historical_inputs_exhibit
    review_evidence["historical_author_inputs"] = historical_inputs_exhibit(
        ctx, task.get("budget_drive_root") or receipt_root, str(task.get("id") or ""),
    )
    return review_evidence


def _root_post_task_lanes(env: Any, task: Dict[str, Any]) -> tuple:
    """``(budget_drive_root, split_drive, parent_env, parent_task)`` of a root's post-task.

    The CANONICAL data root is the parent/budget root for split children and the
    task's own drive for an ordinary root (whose ``budget_drive_root`` field is
    legitimately empty); a split child's post-task runs against that parent root.
    """
    budget_drive_root = str(task.get("budget_drive_root") or "").strip()
    split_drive = bool(
        budget_drive_root
        and str(pathlib.Path(budget_drive_root).resolve(strict=False)) != str(pathlib.Path(env.drive_root).resolve(strict=False))
    )
    parent_env = None
    parent_task = None
    if split_drive:
        from types import SimpleNamespace

        parent_env = SimpleNamespace(repo_dir=env.repo_dir, drive_root=pathlib.Path(budget_drive_root), drive_path=lambda rel: pathlib.Path(budget_drive_root) / rel)
        parent_task = {**task, "drive_root": budget_drive_root, "child_drive_root": str(env.drive_root)}
    return budget_drive_root, split_drive, parent_env, parent_task


def _root_post_task_blocking(
    task: Dict[str, Any], *, split_drive: bool, project_scoped: bool, project_task: bool,
    parent_env: Any, parent_task: Dict[str, Any] | None,
) -> bool:
    """Whether this root's post-task runs inline (the lane whose answer leaves on the live bus)."""
    split = split_drive and parent_env is not None and parent_task is not None
    split_non_project = split and not project_scoped
    blocking = in_worker_process() or bool(task.get("_is_direct_chat")) or split_non_project or (
        str(task.get("type") or "") == "evolution"
        or bool(str(task.get("workspace_root") or "").strip())
        or bool(str(task.get("workspace_mode") or "").strip())
        or project_task
    )
    if (is_presence_task(task) and task.get("_is_direct_chat")
            and not in_worker_process() and not split_drive):
        # The native adapter receives the durable result on return. Its request
        # must not wait for synthesis; pooled/forked workers retain their custody.
        blocking = False
    return blocking


def _deliver_answer_first(env: Any, task: Dict[str, Any], event_queue: Any, pending_events: List[Dict[str, Any]],
                          *, register: bool = False) -> bool:
    """Owner 1A(1): the owed answer leaves on the live bus BEFORE cost reconstruction,
    review-evidence/input capture and ``_store_task_result``; ``task_done`` stays
    buffered after the store. The lane is the one whose post-task would run inline
    (``_root_post_task_blocking``) — the nonblocking lane returns its buffered copy.
    ``register`` is for the lane that did not owe the frame itself (a presence root,
    whose result event is its own); every other root registered just before.
    True only when the live ``put`` actually succeeded (never a queue refusal)."""
    if event_queue is None or _root_post_task_already_completed(env, task):
        return False
    from ouroboros.project_facts import resolve_project_id

    budget_drive_root, split_drive, parent_env, parent_task = _root_post_task_lanes(env, task)
    project_scoped = bool(resolve_project_id(task))
    if not _root_post_task_blocking(
        task, split_drive=split_drive, project_scoped=project_scoped,
        project_task=project_scoped and not bool(task.get("_is_direct_chat")),
        parent_env=parent_env, parent_task=parent_task,
    ):
        return False
    # Owed just before (register_final_answer_owed): the live put registers nothing twice.
    return deliver_final_message_live(
        event_queue, pending_events, str(task.get("id") or ""),
        drive_root=budget_drive_root or env.drive_root, register=register,
    )


def _record_project_letters_home(env: Any, task: Dict[str, Any], outcome_axes: Dict[str, Any],
                                 pending_events: List[Dict[str, Any]]) -> None:
    """A pooled project task's durable letters home (journal milestone + digest event)."""
    from ouroboros.project_facts import resolve_project_id

    # Record the cycle in its project journal; project_digest only
    # touches recency. Unified memory already includes the project thread.
    _pid = resolve_project_id(task)
    # The journal bounds long objectives with a pointer to task_results,
    # which retains the full objective and outcome.
    _objective = str(task.get("objective") or task.get("description") or task.get("text") or "")
    _exec_status = str((outcome_axes.get("execution") or {}).get("status") or "unknown")
    try:
        # One fail-soft seam (project_journal.record_task_finalization) for
        # the durable letters home: the task-finished milestone, the Q8
        # off-registry work-location row, and — for the swarm ROOT — the
        # tree-ledger coordination mirror. Kind compares against the
        # canonical execution-axis constants (EXECUTION_OK is "ok"; a raw
        # "success" literal never matched — the C9.1 seed bug).
        from ouroboros.tools.project_journal import record_task_finalization

        record_task_finalization(
            _pid,
            task,
            objective=_objective,
            kind="done" if _exec_status in (EXECUTION_OK, EXECUTION_BEST_EFFORT) else "blocked",
            exec_status=_exec_status,
            # Registry lives on the canonical drive; stamps the durable
            # per-project last-result pointer.
            drive_root=pathlib.Path(str(task.get("budget_drive_root") or env.drive_root)),
        )
    except Exception:
        log.debug("project journal finalization entries failed", exc_info=True)
    try:
        pending_events.append({
            "type": "project_digest",
            "project_id": _pid,
            "task_id": str(task.get("id") or ""),
            "ts": utc_now_iso(),
        })
    except Exception:
        log.debug("project digest emission failed", exc_info=True)


@finalization_child_scope()
def emit_task_results(
    env: Any, memory: Any, llm: Any,
    pending_events: List[Dict[str, Any]],
    task: Dict[str, Any], text: str,
    usage: Dict[str, Any], llm_trace: Dict[str, Any],
    start_time: float, drive_logs: pathlib.Path,
    ctx: Any = None, event_queue: Any = None,
) -> None:
    """Emit all end-of-task events to supervisor and run post-task processing.

    Order (owner 1A(1), "answer first"): the final frame is built and stamped,
    owed in the durable outbox, and put on the live bus — THEN cost reconstruction,
    evidence capture and the result store run; ``task_done`` stays buffered after
    the store. Pre-store carriers state the pre-store observation; post-store
    carriers use the stored status (W10).
    """
    from ouroboros.subagent_bootstrap import actor_first_terminal_projection
    actor_fact, usage, llm_trace = actor_first_terminal_projection(ctx, task, usage, llm_trace, task.get("budget_drive_root") or getattr(env, "drive_root", None))
    from ouroboros.delegate_terminal import terminal_custody_disclosure, record_terminal_reconciliation

    custody_root = task.get("budget_drive_root") or env.drive_root
    with timed_phase("custody_audit", timing=usage.get("_finalization_timing") or {}):
        custody_audit = terminal_custody_disclosure(custody_root, str(task.get("id") or ""),
                                                   usage.get("delegate_terminal_reconciliation"))
    usage["delegate_terminal_reconciliation"] = custody_audit
    loop_outcome = _derive_host_bound_loop_outcome(env, task, text, usage, llm_trace)
    # A silent run's words stay internal. Clean, it publishes nothing (no chat row, owed outbox
    # row, live send or delivery subject); otherwise only a host notice of its outcome.
    notice = _wake_outcome_notice(text, loop_outcome, custody_audit) if usage.get("wake_silent") is True else ""
    if notice:
        set_terminal_host_notice(usage, str(usage.get("terminal_host_notice") or ""), notice)
    quiet, published = usage.get("wake_silent") is True and not notice, notice or text
    receipt_root = pathlib.Path(str(getattr(env, "drive_root", None) or "."))
    verification_receipts = task_verification_receipts(ctx, receipt_root, task)
    # Apply FR3 once so events and the durable result share the same flagged outcome.
    apply_receipt_absent_flag(
        loop_outcome, llm_trace, receipt_root, str(task.get("id") or ""),
        expected_output=str(task.get("expected_output") or ""), receipts=verification_receipts,
    )
    outcome_axes = normalize_outcome_axes({"outcome_axes": loop_outcome.get("outcome_axes")})
    loop_rounds = (loop_outcome.get("usage") or {}).get("total_rounds")
    execution_status = str((outcome_axes.get("execution") or {}).get("status") or "")
    reason_code = str(loop_outcome.get("reason_code") or "")
    if reason_code == REASON_OWNER_REQUESTED_FINALIZATION:
        # Which stop the owner pressed rides the terminal it ended: this frame, the stored result
        # (terminal_result_fields) and their readers. The loop's one reason covers both buttons.
        from supervisor.owner_stop import owner_stop_origin

        if stop_origin := owner_stop_origin(custody_root, str(task.get("id") or "")):
            usage["cancel_origin"] = stop_origin
    # Pre-store carriers state the status observed now, as the outbox stamp does (a later race may differ).
    terminal_status = _durable_terminal_status(env, task, execution_status, usage=usage, reason_code=reason_code)
    outcome_axes = lifecycle_named(outcome_axes, terminal_status)
    failed_or_forced = (
        execution_status in {EXECUTION_FAILED, EXECUTION_INFRA_FAILED, EXECUTION_BEST_EFFORT}
        or str(usage.get("execution_status") or usage.get("result_status") or "") in {EXECUTION_FAILED, EXECUTION_INFRA_FAILED}
    )
    if ctx is not None and failed_or_forced:
        ctx._presence_completion_accepted = False
    if is_presence_task(task):
        _stamp_presence_terminal_facts(task, usage, llm_trace, ctx, reason_code)
    # A Stop marker already on the task suppresses paid synthesis, not the durable outbox or free facts.
    _root_outbox = _is_root_post_task({k: v for k, v in task.items() if k != "_skip_post_task_synthesis"})
    if getattr(ctx, "_skip_post_task_synthesis", False):   # "Stop now": paid root predicates see it
        task["_skip_post_task_synthesis"] = True
    _presence = is_presence_task(task)
    _typed_routing_action = str(getattr(ctx, "_typed_routing_action_emitted", "") or "").strip()
    # The final frame's durable identity: the child lineage and the turn's
    # origin label (a wake-up's "consciousness"), so the chat row keeps both.
    _message_meta = {**subagent_message_meta(task, task_id=str(task.get("id") or "")), **initiator_meta(task)}
    from ouroboros.post_task_synthesis import task_tool_metrics
    tool_metrics = task_tool_metrics(llm_trace)
    send_event = {
        "type": "send_message", "chat_id": task["chat_id"],
        "text": published or "\u200b", "log_text": published or "",
        "format": "markdown",
        "task_id": task.get("id"), "ts": utc_now_iso(),
    }
    if _message_meta:
        # Final frames carry their own durable identity; replay must not depend
        # on a nearby progress row that may age out independently.
        send_event["progress_meta"] = dict(_message_meta)
    send_event = prepare_terminal_send_event(env.drive_root, task, published, usage, send_event,
                                             presence=_presence, custody_audit=custody_audit)
    if notice:  # the host's words, never the model's: a System row with its own delivery identity
        from supervisor.terminal_delivery import project_terminal_result_event

        send_event = project_terminal_result_event(env.drive_root, task, str(task.get("id") or ""), result_text=notice,
                                                   terminal_origin=TERMINAL_ORIGIN_HOST_NOTICE, base_event=send_event)
    if not quiet or _presence:
        pending_events.append(build_presence_result_event(
            task, text, ctx, terminal_origin=str(usage.get("terminal_origin") or ""),
            retain_scheduled_handoff=failed_or_forced,
        ) if _presence else send_event)
    # Every root registers its full answer payload in the durable outbox BEFORE
    # the live send and the result write, including the nonblocking post-task
    # lane. After a crash, boot replays the owed row without scanning
    # task_results; reversing these writes could strand a terminal result with
    # no delivery. Registration and deduplication share ouroboros/task_finalization.py.
    if _root_outbox and not _presence and not quiet:
        # The answer-time debt rides beside the cause it never replaces (#1583); replay
        # overlays the row's current list (gateway.history terminal truth).
        debt = list(custody_audit.get("unreconciled") or []) if isinstance(custody_audit, dict) else []
        send_event.setdefault("progress_meta", {}).update(
            outcome_axes=outcome_axes, reason_code=reason_code,
            **({"delegated_runs_unreconciled": debt} if debt else {}),
            **({"cancel_origin": usage["cancel_origin"]} if usage.get("cancel_origin") else {}))
        stamp_root_final_phase(  # shares the pre-store observation, not a later racing write
            send_event, task, terminal_status=terminal_status,
            post_task_open=not task.get("_skip_post_task_synthesis") and not _root_post_task_already_completed(env, task),
        )
        register_final_answer_owed(task, send_event, env_drive_root=env.drive_root)
    # Answer first (owner 1A(1)): the owner's answer is on the live bus before
    # the heavy tail below (cost ledger, evidence capture, the result store).
    _live_delivered = _root_outbox and not quiet and _deliver_answer_first(env, task, event_queue, pending_events, register=_presence)
    # A failed live put keeps its owed/buffered answer and skips optional annotation.
    # Children and silent runs retain their existing buffered/no-chat lifecycle.
    if _live_delivered or not _root_outbox or quiet:
        with timed_phase("custody_annotation", timing=usage.get("_finalization_timing") or {}):
            record_terminal_reconciliation(custody_root, str(task.get("id") or ""), custody_audit,
                                           addressed=True, emit_evidence=True)
    duration_sec = round(time.time() - start_time, 3)
    task_cost_fields = _task_cost_fields(env, task)
    try:
        append_jsonl(drive_logs / "events.jsonl", {
            "ts": utc_now_iso(), "type": "task_eval", "ok": execution_status not in {EXECUTION_FAILED, EXECUTION_INFRA_FAILED},
            "task_id": task.get("id"), "task_type": task.get("type"),
            "outcome_axes": outcome_axes,
            "reason_code": reason_code,
            "review_eligibility": str(loop_outcome.get("review_eligibility") or ""),
            "review_trigger": str(loop_outcome.get("review_trigger") or ""),
            "duration_sec": duration_sec,
            **tool_metrics,
            "response_len": len(text),
        })
    except Exception:
        log.warning("Failed to log task eval event", exc_info=True)
    pending_events.append({
        "type": "task_metrics",
        "task_id": task.get("id"), "task_type": task.get("type"),
        "rounds": loop_rounds,
        "outcome_axes": outcome_axes,
        "reason_code": reason_code,
        "duration_sec": duration_sec,
        **tool_metrics,
        **task_cost_fields,
        **({"resource_limit": dict(usage.get("resource_limit") or {})}
           if isinstance(usage.get("resource_limit"), dict) else {}),
        "ts": utc_now_iso(),
    })

    review_evidence = _collect_review_evidence_bundle(env, ctx, task, receipt_root, verification_receipts)

    # The durable result write happens AFTER the answer left (owner 1A(1)); the
    # outbox row registered above is the crash insurance for this window.
    with timed_phase("result_store", timing=usage.get("_finalization_timing") or {}):
        _store_task_result(env, task, text, usage, llm_trace, review_evidence=review_evidence,
                           loop_outcome=loop_outcome, cost_fields=task_cost_fields)
    if _live_delivered:
        # The live copy's timing snapshot was frozen at its enqueue; the store span
        # completed after it and is journaled as its own correlated row.
        emit_post_answer_timing(usage, drive_logs, task, str(send_event.get("delivery_id") or ""))
    stored_result = load_task_result(env.drive_root, str(task.get("id") or "")) or {}
    if stored_result.get("status"):  # post-store carriers state the stored row's word
        outcome_axes = lifecycle_named(outcome_axes, str(stored_result["status"]))
    if _root_outbox and task.get("_skip_post_task_synthesis"):
        # Stop before post-task dispatch forbids paid synthesis, not the free
        # factual row. Record it after durable result write; the structural
        # root predicate deliberately excludes stopped roots from recovery.
        fact_usage = {**usage, "rounds": loop_rounds, "outcome_axes": outcome_axes, "reason_code": reason_code}
        if _typed_routing_action:
            fact_usage["typed_routing_action"] = _typed_routing_action
        _record_task_facts(env, task, _pre_synthesis_usage_snapshot(env, task, fact_usage),
                           llm_trace, drive_logs)
    artifact_bundle = stored_result.get("artifact_bundle") if isinstance(stored_result.get("artifact_bundle"), dict) else {}
    review_projection = stored_result.get("review_projection") or {}
    pending_events.append({
        "type": "task_done", "task_id": task.get("id"), "task_type": task.get("type"),
        **{key: task.get(key) or stored_result[key] for key in ("root_task_id", "parent_task_id")
           if task.get(key) or stored_result.get(key)},  # #1578: producer-known lineage only
        # GR2-3c: the DURABLE status rides the event for honesty — the
        # supervisor validates every task_done against the durable row;
        # a stamped status makes the event self-describing.
        "status": str(stored_result.get("status") or ""),
        "_is_direct_chat": bool(task.get("_is_direct_chat")),
        "execution_owner": stored_result.get("execution_owner") or task.get("_execution_owner") or {},
        **({"typed_routing_action": _typed_routing_action} if _typed_routing_action else {}),
        **({"model_execution": send_event["progress_meta"]["model_execution"]}
           if "model_execution" in send_event.get("progress_meta", {}) else {}),
        # Carry the thread so the terminal card finalizes in its project panel
        # (per-thread fan-out), not just the main chat.
        "chat_id": int(task.get("chat_id") or 0),
        "outcome_axes": outcome_axes, "reason_code": reason_code,
        **_custody_debt_event_fields(stored_result),
        "artifact_status": artifact_bundle.get("status") or stored_result.get("artifact_status") or "",
        "artifact_bundle": artifact_bundle,
        "review_status": stored_result.get("review_status") if isinstance(stored_result.get("review_status"), dict) else {},
        **({"review_projection": review_projection} if review_projection.get("panels") or review_projection.get("gaps") or review_projection.get("review_records_omitted") else {}),
        **task_cost_fields,
        # v6.57.0 (P6b): recursive cost incl. children (from the stored rollup) so the
        # parent card / Logs can show the true subtree cost, not just this task's own.
        # ABI-3: honest name only; a legacy stored spelling still resolves.
        "accounted_upper_bound_usd_with_children": resolve_cost_pair(
            stored_result, "accounted_upper_bound_usd_with_children", "cost_usd_with_children")[1],
        "cost_with_children_partial": bool(stored_result.get("cost_with_children_partial")),
        **({"resource_limit": dict(usage.get("resource_limit") or {})}
           if isinstance(usage.get("resource_limit"), dict) else {}),
        "ts": utc_now_iso(),
    })
    # NOTE: task_done is NOT written to events.jsonl here.
    # It goes through EVENT_Q → supervisor _handle_task_done → append_jsonl.
    # This ensures causal ordering: send_message reaches the UI before task_done,
    # preventing the live card from collapsing before the assistant reply arrives.
    restart_reason = str(getattr(ctx, "pending_restart_reason", "") or "").strip()
    evolution_restart = bool(getattr(ctx, "pending_restart_is_evolution", False))
    if restart_reason:
        if not evolution_restart:
            pending_events.append({
                "type": "restart_request", "reason": restart_reason,
                "phase": "ready", **{key: value for key, value in
                    (getattr(ctx, "pending_restart_request", None) or {}).items()
                    if key in {"request_id", "task_id", "task_attempt"}},
                "evolution_restart": False, "ts": utc_now_iso(),
            })
        ctx.pending_restart_reason = None
        ctx.pending_restart_is_evolution = False

    if _is_root_post_task(task):
        post_usage = dict(usage or {})
        # A split root may dispatch on the parent drive before its actor result is copied back.
        post_usage["rounds"] = loop_rounds
        post_usage["outcome_axes"] = outcome_axes
        post_usage["reason_code"] = reason_code
        if _typed_routing_action:
            post_usage["typed_routing_action"] = _typed_routing_action
        from ouroboros.project_facts import resolve_project_id

        _project_scoped = bool(resolve_project_id(task))
        # A project THREAD conversation runs on the fast direct-chat lane. It is
        # project-scoped only for CONTEXT (it sees the project's knowledge/
        # journal), but it is NOT a pooled task completion: it must not block the
        # reply on LLM post-processing and must not write letters home (that
        # would turn every "как дела?" into a journal milestone + a consciousness
        # observation and stall the global chat lock). Only real pooled project
        # tasks get the letters-home + blocking treatment.
        _is_direct_chat = bool(task.get("_is_direct_chat"))
        _project_task = _project_scoped and not _is_direct_chat
        if _project_task:
            _record_project_letters_home(env, task, outcome_axes, pending_events)
        elif _project_scoped:
            _stamp_project_room_pointer(task, env)
        budget_drive_root, split_drive, parent_env, parent_task = _root_post_task_lanes(env, task)

        if not _root_post_task_already_completed(env, task):
            _dispatch_root_post_task(
                env, task, "" if quiet else str(send_event.get("text") or ""), event_queue, pending_events,
                post_usage, llm_trace, drive_logs,
                budget_drive_root=budget_drive_root, split_drive=split_drive,
                project_scoped=_project_scoped, project_task=_project_task,
                parent_env=parent_env, parent_task=parent_task, live_delivered=bool(_live_delivered), wake_silent=quiet,
            )


def _dispatch_root_post_task(
    env: Any, task: Dict[str, Any], text: str,
    event_queue: Any, pending_events: List[Dict[str, Any]],
    post_usage: Dict[str, Any], llm_trace: Dict[str, Any], drive_logs: pathlib.Path,
    *, budget_drive_root: str, split_drive: bool,
    project_scoped: bool, project_task: bool,
    parent_env: Any, parent_task: Dict[str, Any] | None,
    live_delivered: bool = False, wake_silent: bool = False,
) -> None:
    """Live final-answer delivery (unless already done) + sealed ground truth + post-task dispatch.

    The owner's answer normally went out on the live bus before the result store
    (``_deliver_answer_first``, same lane selection as the ``blocking`` below,
    ``live_delivered=True``); otherwise — a direct caller, or an early put the
    transport refused — it goes out here, BEFORE blocking post-task cognition.
    task_done stays LAST via the buffered return (early task_done would release
    the queue slot / start child-drive cleanup mid-post-task). Rationale and the
    never-lost/never-doubled delivery contract: ouroboros/task_finalization.py.
    """
    split = split_drive and parent_env is not None and parent_task is not None
    global_reflection_callback = None
    if split and project_scoped:
        global_reflection_callback = functools.partial(
            _run_global_backlog_promotion_only, parent_env, parent_task)
    split_non_project = split and not project_scoped
    blocking = _root_post_task_blocking(
        task, split_drive=split_drive, project_scoped=project_scoped, project_task=project_task,
        parent_env=parent_env, parent_task=parent_task,
    )
    # A silent run owes no final; the fallback scan would ship a deferred mid-turn send instead.
    if blocking and event_queue is not None and not live_delivered and not wake_silent:
        # The CANONICAL data root — what the supervisor's boot/tick outbox
        # replay reads (§8-A2): the parent/budget root for split children, the
        # task's own drive for an ordinary root (whose ``budget_drive_root``
        # field is legitimately empty).
        deliver_final_message_live(
            event_queue, pending_events, str(task.get("id") or ""),
            drive_root=budget_drive_root or env.drive_root,
        )
    # Sealed ground truth for the post-task phase: what the owner actually
    # received + the durable result's own artifact facts.
    sealed_final = build_sealed_final_package(
        load_task_result(env.drive_root, str(task.get("id") or "")), text,
    )
    if split_non_project:
        _run_post_task_processing_async(
            parent_env, parent_task, post_usage, llm_trace,
            pathlib.Path(budget_drive_root) / "logs",
            blocking=True, sealed_final=sealed_final, event_queue=event_queue,
        )
    else:
        _run_post_task_processing_async(
            env, task, post_usage, llm_trace, drive_logs,
            blocking=blocking,
            on_reflection=global_reflection_callback,
            sealed_final=sealed_final,
            event_queue=event_queue,
        )


def _durable_terminal_status(
    env: Any, task: Dict[str, Any], execution_status: str, *, existing: Dict[str, Any] | None = None,
    usage: Dict[str, Any] | None = None, reason_code: str = "",
) -> str:
    """The status the durable task row settles to (ONE rule for the row and its outbox stamp).

    A failed execution axis — including the owner's "Stop now", whose forced
    finalization stamps ``execution_status="failed"`` under
    ``owner_requested_finalization`` — and an already-failed row both settle
    ``failed``; everything else settles ``completed``. One exception: an inline
    Presence turn that Stop now ended (the rail's reason with the immediate ``cancel_origin``
    read from the still-open intent) settles ``cancelled``, also with a retained answer (a
    ``best_effort`` axis), kept and never sent. No cancel custody can write that row (no
    supervisor map holds the turn), so its runner's own terminal is the Stop's settlement
    (#2012); Wrap up keeps ``failed``/``completed`` and other in-process turns keep their reading.
    """
    if existing is None:
        existing = load_task_result(env.drive_root, str(task.get("id") or "")) or {}
    failed = (
        str(existing.get("status") or "") == STATUS_FAILED
        or execution_status in {EXECUTION_FAILED, EXECUTION_INFRA_FAILED}
    )
    origin = (usage or {}).get("cancel_origin")
    if (task.get("_presence_turn") and reason_code == REASON_OWNER_REQUESTED_FINALIZATION
            and isinstance(origin, dict) and origin.get("stop_policy") == "immediate"):
        return STATUS_CANCELLED
    return STATUS_FAILED if failed else STATUS_COMPLETED


def _store_task_result(env: Any, task: Dict[str, Any], text: str,
                       usage: Dict[str, Any], llm_trace: Dict[str, Any],
                       review_evidence: Dict[str, Any] | None = None,
                       loop_outcome: Dict[str, Any] | None = None,
                       cost_fields: Dict[str, Any] | None = None) -> None:
    """Store task result for parent task retrieval.

    ``loop_outcome``, when supplied by ``emit_task_results``, is the SINGLE already-
    derived, already-receipt_absent-flagged outcome that also fed the task_eval /
    task_metrics event stream — so the persisted axes match the events exactly and we
    do not derive/flag a second time. It is only re-derived here when called without one.
    """
    try:
        trace_summary = build_trace_summary(llm_trace)
        from ouroboros.cost_projection import with_cost_aliases

        cost_fields = with_cost_aliases(cost_fields or {
            "cost_accounting_status": "unavailable", "cost_final": False,
            "cost_accounting_error": "ledger_projection_missing",
            "cost_presentation": None,
            "accounted_upper_bound_usd": None, "total_rounds": None,
            "prompt_tokens": None, "completion_tokens": None,
            "reserved_usd": None, "unresolved_upper_bound_usd": None,
            "unknown_unmetered": None,
        })
        existing = load_task_result(env.drive_root, str(task.get("id") or "")) or {}
        if loop_outcome is None:
            loop_outcome = _derive_host_bound_loop_outcome(env, task, text, usage, llm_trace)
            # Apply FR3 before normalization so the persisted axes and ledger agree.
            apply_receipt_absent_flag(
                loop_outcome, llm_trace, env.drive_root, str(task.get("id") or ""),
                expected_output=str(task.get("expected_output") or ""),
                receipts=task_verification_receipts(None, env.drive_root, task),
            )
        outcome_axes = normalize_outcome_axes({"outcome_axes": loop_outcome.get("outcome_axes")})
        execution_status = str((outcome_axes.get("execution") or {}).get("status") or "")
        reason_code = str(loop_outcome.get("reason_code") or "")
        status = _durable_terminal_status(env, task, execution_status, existing=existing,
                                          usage=usage, reason_code=reason_code)
        from ouroboros.task_finalization import build_completion_observations

        observations = build_completion_observations(
            env.drive_root, {**task, "started_at": existing.get("started_at")}, llm_trace,
        )
        task_contract = build_task_contract(task)
        task = {**task, "task_contract": task_contract}
        artifact_bundle_for_ledger = artifact_bundle_from_result(existing)
        verification_ledger = build_verification_ledger(
            task=task,
            loop_outcome=loop_outcome,
            llm_trace=llm_trace,
            artifact_bundle=artifact_bundle_for_ledger,
            review_evidence=review_evidence or {},
        )
        verification_refs = maybe_write_verification_artifact(
            env.drive_root,
            str(task.get("id") or ""),
            verification_ledger,
        )
        artifacts = list(existing.get("artifacts") or []) if isinstance(existing.get("artifacts"), list) else []
        artifact_record = verification_refs.get("artifact")
        if artifact_record and artifact_record not in artifacts:
            artifacts.append(artifact_record)
        collected_artifacts = collect_task_artifact_records(env.drive_root, str(task.get("id") or ""))
        artifacts = merge_artifact_records(artifacts, collected_artifacts)
        provisional = {
            **existing,
            "artifacts": artifacts,
        }
        artifact_bundle = artifact_bundle_from_result(provisional)
        outcome_axes = dict(outcome_axes)
        existing_artifact_axis = (
            (existing.get("outcome_axes") or {}).get("artifacts")
            if isinstance(existing.get("outcome_axes"), dict)
            else {}
        )
        artifact_axis = dict(existing_artifact_axis) if isinstance(existing_artifact_axis, dict) else {}
        if isinstance(outcome_axes.get("artifacts"), dict):
            artifact_axis.update(outcome_axes.get("artifacts") or {})
        artifact_axis["status"] = str(artifact_bundle.get("status") or artifact_axis.get("status") or "not_applicable")
        outcome_axes["artifacts"] = artifact_axis
        # B1: swarm-efficiency rollup — observed fan-out, or the zero-fanout
        # block for a host-attested Swarm task; None (omitted) for plain tasks.
        swarm_efficiency = _build_swarm_efficiency(env, task)
        subagent_envelope = task.get("subagent_envelope") if isinstance(task.get("subagent_envelope"), dict) else {}
        if str(task.get("delegation_role") or "").lower() == "subagent":
            subagent_envelope = envelope_from_task(
                task, status=status, usage=usage,
                accounted_upper_bound_usd=cost_fields.get("accounted_upper_bound_usd"),
            )
            if cost_fields.get("cost_accounting_status") != "available":
                subagent_envelope.update({
                    "accounted_upper_bound_usd": None,
                    "cost_accounting_status": "unavailable",
                    "cost_final": False,
                })
        # Compatibility projection only. The physical-attempt ledger is the sole
        # monetary authority; the supervisor replaces the root projection with the
        # exact subtree total at terminal handling. Do not independently walk task
        # results here: that was a second accounting engine and could double-count.
        _own_cost = cost_fields.get("accounted_upper_bound_usd")
        _cost_with_children = _own_cost
        _cost_partial = True
        root_phase_checkpoint: Dict[str, Any] = {}
        if _is_root_post_task(task):
            incoming_checkpoint = llm_trace.get("root_phase_checkpoint")
            existing_checkpoint = existing.get("root_phase_checkpoint")
            if (
                isinstance(existing_checkpoint, dict)
                and (existing_checkpoint.get("post_task_synthesis") in {"running", "paused"}
                     or _post_task_synthesis_is_terminal(existing_checkpoint.get("post_task_synthesis")))
            ):
                root_phase_checkpoint = dict(existing_checkpoint)
            elif isinstance(incoming_checkpoint, dict):
                root_phase_checkpoint = dict(incoming_checkpoint)
            elif isinstance(existing_checkpoint, dict):
                root_phase_checkpoint = dict(existing_checkpoint)
            # A pre-fold row may also carry the task-acceptance phase fields (``phase``,
            # ``status``, ``pass_index``); nothing reads them and new rows carry none.
            root_phase_checkpoint.setdefault("post_task_synthesis", "pending_once")
        review_projection = _compact_review_projection(llm_trace, task, env.drive_root)
        model_execution = model_execution_projection(usage)
        write_task_result(
            env.drive_root,
            str(task.get("id") or ""),
            status,
            _terminal_observed=True,
            reason_code=reason_code,
            outcome_axes=outcome_axes,
            # Compatibility mirror consumed by the gateway and task_done event.
            # ``outcome_axes.review`` remains the canonical structured axis.
            review_status=dict(outcome_axes.get("review") or {}),
            task_contract=task_contract,
            _is_direct_chat=bool(task.get("_is_direct_chat")),
            loop_outcome=loop_outcome,
            project_id=str(task.get("project_id") or ""),
            parent_task_id=task.get("parent_task_id"),
            root_task_id=task.get("root_task_id"),
            session_id=task.get("session_id"),
            actor_id=task.get("actor_id"),
            delegation_role=task.get("delegation_role"),
            role=task.get("role"),
            description=task.get("description"),
            objective=task.get("objective") or task.get("description"),
            title=task.get("title"),
            expected_output=task.get("expected_output"),
            constraints=task.get("constraints"),
            context=task.get("context"),
            workspace_root=task.get("workspace_root"),
            workspace_mode=task.get("workspace_mode"),
            memory_mode=task.get("memory_mode"),
            drive_root=task.get("drive_root"),
            child_drive_root=task.get("child_drive_root") or task.get("drive_root"),
            budget_drive_root=task.get("budget_drive_root"),
            task_constraint=task.get("task_constraint"),
            model_lane=task.get("model_lane"),
            requested_model_lane=task.get("requested_model_lane"),
            parent_model_lane=task.get("parent_model_lane"),
            requested_executor=task.get("requested_executor"),
            effective_model_lane=task.get("effective_model_lane"),
            model=task.get("model"),
            use_local_model=task.get("use_local_model"),
            effective_executor=task.get("effective_executor"),
            executor_route=task.get("executor_route"),
            tool_profile=task.get("tool_profile"),
            capability_delta=subagent_envelope.get("capability_delta") or task.get("capability_delta"),  # Q1A: envelope copy carries the native_only amendment
            **substrate_result_fields(subagent_envelope),  # Q1A: substrate FACT + raw counts
            reasoning_effort=task.get("reasoning_effort"),
            **effort_result_fields(task),
            task_group_id=task.get("task_group_id"),
            task_group=task.get("task_group"),
            subagent_envelope=subagent_envelope,
            configured_subagent=task.get("configured_subagent"),
            parent_cognitive_route=task.get("parent_cognitive_route"),
            subagent_availability=task.get("subagent_availability"),
            metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
            result=text or "", **terminal_result_fields(usage, existing=existing, title=task.get("title")),
            final_answer=str(loop_outcome.get("final_answer") or ""),
            trace_summary=trace_summary,
            trace_refs=loop_outcome.get("trace_refs") or {},
            **({"model_execution": model_execution} if model_execution is not None else {}),
            **{**cost_fields, "accounted_upper_bound_usd_with_children": _cost_with_children,
               "cost_with_children_partial": _cost_partial},
            review_evidence=review_evidence or {},
            completion_observations=observations,
            **({"review_projection": review_projection} if review_projection.get("panels") or review_projection.get("gaps") or review_projection.get("review_records_omitted") else {}),
            verification_ledger=verification_refs.get("inline"),
            artifact_bundle=artifact_bundle,
            artifacts=artifacts,
            **({"root_phase_checkpoint": root_phase_checkpoint} if root_phase_checkpoint else {}),
            **({"swarm_efficiency": swarm_efficiency} if swarm_efficiency else {}),
            ts=utc_now_iso(),
        )
    except Exception as e:
        log.warning("Failed to store task result: %s", e)
        return
    try:
        from ouroboros.subagent_history import record_task_execution
        record_task_execution(task, usage, drive_root=task.get("budget_drive_root") or env.drive_root)
    except Exception as e:
        log.warning("Task result stored; subagent history unavailable: %s", e)


@timed_section("review_context")
def build_review_context(env: Any) -> str:
    """Build a compact review continuity section for the main reasoning context."""
    try:
        from ouroboros.context_sections import continuity_status_lines
        from ouroboros.review_state import (
            _LEGACY_CURRENT_REPO_KEY,
            format_status_section,
            load_state,
            make_repo_key,
            stale_marker_status,
        )
        from ouroboros.task_continuation import (
            list_review_continuations,
            retire_settled_continuations_for_context,
        )
        from ouroboros.task_results import load_task_result

        state = load_state(pathlib.Path(env.drive_root))
        retired = retire_settled_continuations_for_context(
            env.drive_root, state, lambda tid: load_task_result(env.drive_root, tid))
        continuations, corrupt = list_review_continuations(env.drive_root)
        repo_dir = pathlib.Path(env.repo_dir)
        repo_key = make_repo_key(repo_dir)
        open_obs = state.get_open_obligations(repo_key=repo_key)
        open_debts = state.get_open_commit_readiness_debts(repo_key=repo_key)
        if (
            not state.latest_attempt()
            and not continuations
            and not corrupt
            and not open_obs
            and not open_debts
        ):
            return ""

        # Owed work and the checkout's stale marker only: the retired advisory
        # gate projected no readiness here (decision 3A), and the commit gate is
        # the panel, tests, custody and binding, none of which this section decides.
        lines: List[str] = ["## Review Continuity", f"- repo_key={repo_key}"]

        marker_current, look_unreadable = stale_marker_status(state, pathlib.Path(env.drive_root), repo_key)
        if marker_current:
            lines.append(
                f"- stale_marker={state.last_stale_from_edit_ts[:19]}: "
                f"{_truncate_with_notice(state.last_stale_reason or 'worktree edit invalidated advisory freshness', 220)}"
            )
            lines.append(f"  invalidated_by={state.stale_marker_attribution_note()}")
            if look_unreadable:
                lines.append(f"  preflight_unreadable={_truncate_with_notice(look_unreadable, 220)} (marker unverified)")

        if open_debts:
            lines.append("- retry_anchor=commit_readiness_debt")
            lines.append(f"- commit_readiness_debt={len(open_debts)}")
            lines.append("\n### Commit-readiness debt")
            for debt in open_debts:
                summary = _truncate_with_notice(getattr(debt, "summary", ""), 180).replace("\n", " ")
                lines.append(
                    f"- [{getattr(debt, 'debt_id', '')}] status={getattr(debt, 'status', '')} "
                    f"category={getattr(debt, 'category', '')} source={getattr(debt, 'source', '')}"
                )
                lines.append(f"  summary={summary}")
                if getattr(debt, "source_obligation_ids", None):
                    lines.append(f"  obligation_ids={', '.join(list(debt.source_obligation_ids or []))}")
                for evidence in list(getattr(debt, "evidence", []) or []):
                    lines.append(f"  evidence={_truncate_with_notice(evidence, 180).replace(chr(10), ' ')}")
        else:
            lines.append("- commit_readiness_debt=0")

        if open_obs:
            lines.append(f"- open_obligations={len(open_obs)}")
            for ob in open_obs:
                reason = _truncate_with_notice(getattr(ob, "reason", ""), 120).replace("\n", " ")
                lines.append(
                    f"  [{getattr(ob, 'obligation_id', '')}] "
                    f"{getattr(ob, 'item', '')}: {reason}"
                )
        else:
            lines.append("- open_obligations=0")

        continuation_start = len(lines)
        if retired:
            lines.append(f"- {len(retired)} settled continuation(s) archived out of context "
                         "(durable under state/review_continuations/archived/): "
                         + ", ".join(retired[:5])
                         + (f" (+{len(retired) - 5} more)" if len(retired) > 5 else ""))
        scoped_continuations = [
            item for item in continuations
            if item.repo_key in ("", repo_key, _LEGACY_CURRENT_REPO_KEY)
        ]
        if scoped_continuations:
            lines.append("\n### Open review continuations")
            scoped_continuations.sort(key=lambda item: str(item.updated_ts or item.created_ts or ""), reverse=True)
            # Cap review context only with explicit OMISSION NOTEs; no silent slicing.
            _CONTINUATION_CAP = 5
            _PER_FINDING_CAP = 3
            shown_continuations = scoped_continuations[:_CONTINUATION_CAP]
            if len(scoped_continuations) > _CONTINUATION_CAP:
                lines.append(
                    f"⚠️ OMISSION NOTE: {len(scoped_continuations) - _CONTINUATION_CAP} "
                    f"older continuation(s) omitted (showing {_CONTINUATION_CAP} most recent)."
                )
            for item in shown_continuations:
                task_status = str((load_task_result(env.drive_root, item.task_id) or {}).get("status") or "missing")
                lines.append(
                    f"- task={item.task_id} status={task_status} source={item.source} "
                    f"stage={item.stage} tool={item.tool_name or 'commit_reviewed'} "
                    f"attempt={int(item.attempt or 0)}"
                )
                if item.block_reason:
                    lines.append(f"  block_reason={item.block_reason}")
                if item.readiness_warnings:
                    shown = list(item.readiness_warnings)[:_PER_FINDING_CAP]
                    for warn in shown:
                        warning = _truncate_with_notice(warn, 180).replace("\n", " ")
                        lines.append(f"  readiness_warning={warning}")
                    if len(item.readiness_warnings) > _PER_FINDING_CAP:
                        lines.append(
                            f"  ⚠️ OMISSION NOTE: {len(item.readiness_warnings) - _PER_FINDING_CAP} "
                            f"additional readiness_warning(s) omitted."
                        )
                if item.critical_findings:
                    shown = list(item.critical_findings)[:_PER_FINDING_CAP]
                    for top in shown:
                        label = str(top.get("item") or top.get("reason") or "critical finding")
                        reason = _truncate_with_notice(top.get("reason") or "", 140).replace("\n", " ")
                        lines.append(f"  critical_finding={label}: {reason}")
                    if len(item.critical_findings) > _PER_FINDING_CAP:
                        lines.append(
                            f"  ⚠️ OMISSION NOTE: {len(item.critical_findings) - _PER_FINDING_CAP} "
                            f"additional critical_finding(s) omitted."
                        )
                if item.advisory_findings:
                    shown = list(item.advisory_findings)[:_PER_FINDING_CAP]
                    for top in shown:
                        label = str(top.get("item") or top.get("reason") or "advisory finding")
                        reason = _truncate_with_notice(top.get("reason") or "", 140).replace("\n", " ")
                        lines.append(f"  advisory_finding={label}: {reason}")
                    if len(item.advisory_findings) > _PER_FINDING_CAP:
                        lines.append(
                            f"  ⚠️ OMISSION NOTE: {len(item.advisory_findings) - _PER_FINDING_CAP} "
                            f"additional advisory_finding(s) omitted."
                        )
                if item.obligation_ids:
                    lines.append(f"  obligation_ids={', '.join(item.obligation_ids)}")
        if corrupt:
            lines.append("\n### Corrupt review continuations")
            _CORRUPT_CAP = 3
            shown_corrupt = corrupt[:_CORRUPT_CAP]
            for item in shown_corrupt:
                lines.append(f"- {_truncate_with_notice(item, 220)}")
            if len(corrupt) > _CORRUPT_CAP:
                lines.append(
                    f"⚠️ OMISSION NOTE: {len(corrupt) - _CORRUPT_CAP} "
                    f"additional corrupt entry/entries omitted."
                )

        from ouroboros.section_floor import HostSection

        operative = '\n'.join(continuity_status_lines(state, repo_dir, (marker_current, look_unreadable)) + lines[continuation_start:]
                              + ['Historical review ledger: review_status().'])
        history = format_status_section(state, repo_dir=repo_dir)
        if history:
            history = history.replace("## Advisory Pre-Review Status", "### Historical review ledger")
            lines.append("\n" + history)

        return HostSection('review_continuity', "\n".join(lines), operative)
    except Exception:
        log.debug("Failed to build review continuity context", exc_info=True)
        return ""

# The v7 post-task-synthesis split: the members below moved into
# ouroboros/post_task_synthesis.py; this facade keeps their historical
# ouroboros.agent_task_pipeline bindings for consumers.
from ouroboros.post_task_synthesis import (  # noqa: E402, F401 -- intentional public re-exports
    build_trace_summary,
    _pre_synthesis_usage_snapshot,
    _compact_review_projection,
    _record_task_facts,
    _post_task_paid_interruption,
    _run_memory_fallback_draft,
)
