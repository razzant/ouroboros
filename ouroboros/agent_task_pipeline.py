"""Post-task result emission, memory work, reflections, and review context."""

from __future__ import annotations

import json
import functools
import logging
import pathlib
import threading
import time
from dataclasses import replace
from typing import Any, Callable, Dict, List

from ouroboros.cost_projection import cost_projection, resolve_cost_pair
from ouroboros.task_results import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    load_task_result,
    write_task_result,
)
from ouroboros.artifacts import collect_task_artifact_records, merge_artifact_records
from ouroboros.outcomes import (
    EXECUTION_BEST_EFFORT,
    EXECUTION_FAILED,
    EXECUTION_INFRA_FAILED,
    EXECUTION_OK,
    BEST_EFFORT_REASON_CODES,
    apply_receipt_absent_flag,
    artifact_bundle_from_result,
    build_verification_ledger,
    custody_debt_axes,
    derive_loop_outcome,
    maybe_write_verification_artifact,
    normalize_outcome_axes,
)
from ouroboros.outcome_receipt_store import task_verification_receipts
from ouroboros.contracts.task_contract import build_task_contract
from ouroboros.subagents import envelope_from_task, substrate_result_fields
from ouroboros.subagent_messages import subagent_message_meta
from ouroboros.utils import utc_now_iso, append_jsonl, truncate_review_artifact as _truncate_with_notice
from ouroboros.post_task_checkpoint import (
    POST_TASK_SYNTHESIS_INFLIGHT as _POST_TASK_SYNTHESIS_INFLIGHT,
    POST_TASK_SYNTHESIS_LOCK as _POST_TASK_SYNTHESIS_LOCK,
    is_root_post_task as _is_root_post_task,
    post_task_synthesis_is_open as _post_task_synthesis_is_open,
    post_task_synthesis_is_terminal as _post_task_synthesis_is_terminal,
    root_checkpoint_roots as _root_checkpoint_roots,
    root_post_task_already_completed as _root_post_task_already_completed,
    set_root_post_task_checkpoint as _set_root_post_task_checkpoint,
)
from ouroboros.skill_publish_result import apply_skill_publish_receipt_veto
from ouroboros.task_finalization import (
    build_sealed_final_package,
    build_swarm_efficiency as _build_swarm_efficiency,  # moved (module ceiling); tests import it here
    deliver_final_message_live, prepare_terminal_send_event, register_final_answer_owed, stamp_root_final_phase,
    sealed_final_prompt_section, terminal_result_fields,  # noqa: F401 -- the pipeline module keeps its historical import surface for the synthesis leaf
)
from ouroboros.dialogue_provenance import is_presence_task, presence_provenance_fields  # noqa: F401 -- the pipeline module keeps its historical import surface for the synthesis leaf
from ouroboros.presence_runner import build_presence_result_event

log = logging.getLogger(__name__)


# The synthesis cost/snapshot projections live in `ouroboros/synthesis_cost_text.py`
# (extracted at this module's size ceiling); re-exported here because the
# synthesis prompts, the tests and monkeypatch targets name them on THIS surface.
from ouroboros.synthesis_cost_text import (  # noqa: F401,E402
    _SYNTHESIS_USAGE_PROMPT_FIELDS,
    _summary_row_cost_fields,
    _synthesis_cost_text,
    _synthesis_cost_usd,
    _synthesis_usage_snapshot_text,
)


def _run_post_task_processing_async(
    env: Any,
    task: Dict[str, Any],
    usage: Dict[str, Any],
    llm_trace: Dict[str, Any],
    review_evidence: Dict[str, Any],
    drive_logs: pathlib.Path,
    *,
    blocking: bool = False,
    on_reflection: Callable[[Dict[str, Any] | None, Any], None] | None = None,
    sealed_final: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """Run best-effort LLM-heavy post-task memory work off the reply path."""
    task_snapshot = json.loads(json.dumps(task, ensure_ascii=False, default=str))
    trace_snapshot = json.loads(json.dumps(llm_trace, ensure_ascii=False, default=str))
    review_evidence_snapshot = json.loads(json.dumps(review_evidence, ensure_ascii=False, default=str))
    sealed_snapshot = (
        json.loads(json.dumps(sealed_final, ensure_ascii=False, default=str))
        if isinstance(sealed_final, dict) and sealed_final else None
    )

    result: Dict[str, Any] = {}
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

    post_task_key: tuple[str, str] | None = None
    if _is_root_post_task(task_snapshot):
        # The durable checkpoint owns paid idempotency; terminal roots never pay again.
        if _root_post_task_already_completed(env, task_snapshot):
            return None
        task_id = str(task_snapshot.get("id") or task_snapshot.get("task_id") or "")
        roots = _root_checkpoint_roots(env, task_snapshot)
        root_key = str(pathlib.Path(
            task_snapshot.get("budget_drive_root")
            or (roots[0] if roots else env.drive_root)
        ).resolve(strict=False))
        post_task_key = (root_key, task_id)
        with _POST_TASK_SYNTHESIS_LOCK:
            if post_task_key in _POST_TASK_SYNTHESIS_INFLIGHT:
                return None
            _POST_TASK_SYNTHESIS_INFLIGHT.add(post_task_key)
        _set_root_post_task_checkpoint(env, task_snapshot, "running")

    # Freeze one honest subtree view synchronously at the pre-synthesis
    # boundary. No scoped worker or consolidation/model call may start first;
    # summary and reflection then receive this same object, while the existing
    # terminal checkpoint remains the only final accounting authority.
    usage_snapshot = _pre_synthesis_usage_snapshot(env, task_snapshot, usage)

    # The owner's will while the synthesis is ALREADY billing (Stop-now after
    # the loop returned): one local predicate, consulted before EACH paid
    # stage — the durable immediate cancel intent every stop ingress mints
    # and custody keeps open while this worker holds its in-flight key, or
    # the live task marker re-read (the entry snapshot cannot see a stop that
    # landed later). The owner mailbox control is NOT the worker's currency:
    # it is unlinked at task_done and its only reader is the loop's drain.
    intent_root = task_snapshot.get("budget_drive_root") or env.drive_root
    stage_task_id = str(task_snapshot.get("id") or task_snapshot.get("task_id") or "")

    def _owner_stop_requested() -> bool:
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
        try:
            from ouroboros.llm import LLMClient
            from ouroboros.memory import Memory

            llm_client = LLMClient()
            task_memory = Memory(drive_root=env.drive_root, repo_dir=env.repo_dir)

            def _promotion() -> None:
                if is_presence_task(task_snapshot):
                    return
                from ouroboros.project_facts import resolve_project_id

                reflection_entry = result.get("reflection_entry")
                _pid = resolve_project_id(task_snapshot)
                # Project facts stay scoped; generic process lessons remain global.
                _update_improvement_backlog(env, reflection_entry)
                _apply_reflection_memory_actions(env, reflection_entry, project_id=_pid)
                try:
                    from ouroboros.post_task_evolution import maybe_promote

                    maybe_promote(env, task_snapshot, reflection_entry, llm_client)
                except Exception:
                    log.debug("Post-task evolution promotion failed", exc_info=True)
                if on_reflection is not None:
                    on_reflection(reflection_entry, llm_client)

            # All late model work belongs to this one scoped worker.  This keeps
            # the root checkpoint non-final until consolidation, summary,
            # reflection, and promotion have all stopped billing.  Summary
            # before reflection: chat.jsonl is more durable than best-effort
            # reflection/backlog.
            stages: List[tuple[str, Callable[[], Any]]] = [
                ("chat_consolidation", lambda: _run_chat_consolidation(
                    env, task_memory, llm_client, task_snapshot, drive_logs)),
                ("scratchpad_consolidation", lambda: _run_scratchpad_consolidation(
                    env, task_memory, llm_client)),
                ("summary", lambda: _run_task_summary(
                    env, llm_client, task_snapshot, usage_snapshot, trace_snapshot, drive_logs,
                    review_evidence=review_evidence_snapshot, sealed_final=sealed_snapshot)),
                ("reflection", lambda: result.__setitem__("reflection_entry", _run_reflection(
                    env, llm_client, task_snapshot, usage_snapshot, trace_snapshot,
                    review_evidence_snapshot, sealed_final=sealed_snapshot))),
                ("promotion", _promotion),
            ]
            for index, (_name, run_stage) in enumerate(stages):
                if _owner_stop_requested():
                    # Stop-now: the remaining paid stages are skipped and NAMED
                    # in the typed disclosure below; what already ran stays.
                    skipped = [name for name, _run in stages[index:]]
                    break
                run_stage()
            if not skipped:
                checkpoint_status = "completed"
        except Exception:
            log.warning("Async post-task processing failed", exc_info=True)
        finally:
            _set_root_post_task_checkpoint(
                env, task_snapshot, checkpoint_status,
                stop_reason=f"owner_stopped:skipped={','.join(skipped)}" if skipped else "",
            )
            if post_task_key is not None:
                with _POST_TASK_SYNTHESIS_LOCK:
                    _POST_TASK_SYNTHESIS_INFLIGHT.discard(post_task_key)

    def _run() -> None:
        with usage_scope(post_scope):
            _run_scoped()

    if blocking:
        _run()
        return result.get("reflection_entry")
    try:
        threading.Thread(target=_run, daemon=True).start()
    except Exception:
        _set_root_post_task_checkpoint(env, task_snapshot, "degraded")
        if post_task_key is not None:
            with _POST_TASK_SYNTHESIS_LOCK:
                _POST_TASK_SYNTHESIS_INFLIGHT.discard(post_task_key)
        raise
    return None


def recover_pending_root_post_task_synthesis(
    drive_root: Any, repo_dir: Any = None,
) -> int:
    """Resume an undispatched root synthesis; degrade an indeterminate one.

    The existing task-result checkpoint is the only durable authority.  The
    process-local in-flight set prevents the periodic reconciler from spawning a
    duplicate while the recovered thread is alive.  After restart only
    ``pending_once`` is replay-safe: ``running`` may have crossed a paid provider
    boundary, so it becomes terminal ``degraded`` instead of repeating calls.
    """
    from types import SimpleNamespace
    from ouroboros.task_results import list_task_results

    root = pathlib.Path(drive_root).resolve(strict=False)
    try:
        rows = list_task_results(root)
    except Exception:
        return 0
    recovered = 0
    for stored in rows:
        task_id = str(stored.get("task_id") or stored.get("id") or "")
        checkpoint = stored.get("root_phase_checkpoint")
        phase = str(checkpoint.get("post_task_synthesis") or "") if isinstance(checkpoint, dict) else ""
        if not task_id or not _post_task_synthesis_is_open(phase):
            continue
        task = {**stored, "id": task_id, "root_task_id": str(stored.get("root_task_id") or task_id)}
        task.setdefault("budget_drive_root", str(root))
        if not _is_root_post_task(task):
            continue
        env = SimpleNamespace(
            repo_dir=pathlib.Path(repo_dir or task.get("repo_dir") or root.parent),
            drive_root=root,
            drive_path=lambda rel, _root=root: _root / rel,
        )
        if phase == "running":
            stored = _set_root_post_task_checkpoint(
                env,
                task,
                "degraded",
                stop_reason="restart_indeterminate_running",
            )
            stored_checkpoint = (stored or {}).get("root_phase_checkpoint") or {}
            stored_phase = str(stored_checkpoint.get("post_task_synthesis") or "")
            if not _post_task_synthesis_is_terminal(stored_phase):
                raise RuntimeError(
                    "Root post-task synthesis recovery did not persist a terminal "
                    f"checkpoint for {task_id} (stored={stored_phase or 'unavailable'})"
                )
            recovered += 1
            continue
        usage = {
            # Null stays NULL (C2): `float(... or 0)` turned a task whose cost was
            # never accounted into a confident "$0.00" in the recovered synthesis —
            # a fabricated receipt for the one path (restart recovery) where the
            # amount is least likely to be known.
            "cost": cost_projection(task)["accounted_upper_bound_usd"],
            "rounds": int(task.get("total_rounds") or 0),
            "reason_code": str(task.get("reason_code") or ""),
            "outcome_axes": task.get("outcome_axes") or {},
        }
        trace = {
            "tool_calls": [],
            "reasoning_notes": [str(task.get("trace_summary") or "")],
            "recovered_post_task_synthesis": True,
        }
        _run_post_task_processing_async(
            env, task, usage, trace,
            task.get("review_evidence") if isinstance(task.get("review_evidence"), dict) else {},
            root / "logs", blocking=False,
            # Recovered synthesis gets the same sealed ground truth from the
            # durable record: delivered result text + its artifact facts.
            sealed_final=build_sealed_final_package(task, str(task.get("result") or "")),
        )
        recovered += 1
    return recovered


def _run_global_backlog_promotion_only(
    env: Any,
    task: Dict[str, Any],
    reflection_entry: Dict[str, Any] | None,
    llm: Any,
) -> None:
    """Feed canonical improvement backlog/promotion without leaking project memory."""

    if not reflection_entry:
        return
    try:
        candidates = [
            item for item in (reflection_entry.get("backlog_candidates") or [])
            if isinstance(item, dict) and str(item.get("summary") or "").strip()
        ]
        if not candidates:
            return
        sanitized_entry = {
            "reflection": "\n".join(f"- {str(item.get('summary') or '').strip()}" for item in candidates),
            "backlog_candidates": candidates,
            "memory_actions": [],
        }
        _update_improvement_backlog(env, sanitized_entry)
        from ouroboros.post_task_evolution import maybe_promote

        global_task = {
            "id": str(task.get("id") or ""),
            "type": str(task.get("type") or "task"),
            "source": "project_scoped_global_improvement",
            "metadata": {"globalized_from_project_task": True},
        }
        maybe_promote(env, global_task, sanitized_entry, llm)
    except Exception:
        log.debug("Canonical post-task promotion-only path failed", exc_info=True)


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
    if task_id and task_id == root_task_id:
        try:
            if attribution_task_id(evidence_root, (task_id,)) == task_id:
                record_terminal_mutation_candidates(evidence_root, task_id)
        except Exception:
            log.debug("terminal mutation snapshot failed for %s", task_id, exc_info=True)
    llm_trace.pop("mutation_attribution", None)
    for candidate in task_ids:
        projection = load_mutation_evidence_projection(evidence_root, candidate)
        if projection:
            llm_trace["mutation_attribution"] = projection
            return


def _derive_host_bound_loop_outcome(
    env: Any,
    task: Dict[str, Any],
    text: str,
    usage: Dict[str, Any],
    llm_trace: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive once from the current durable mutation-evidence binding."""
    _attach_host_mutation_projection(env, task, llm_trace)
    loop_outcome = apply_skill_publish_receipt_veto(derive_loop_outcome(text or "", usage, llm_trace), task, llm_trace)
    return _apply_terminal_custody_outcome(env, task, loop_outcome)


def _apply_terminal_custody_outcome(
    env: Any, task: Dict[str, Any], loop_outcome: Dict[str, Any],
) -> Dict[str, Any]:
    """Make the terminal custody audit authoritative for result and event axes."""

    drive_root = task.get("budget_drive_root") or getattr(env, "drive_root", None)
    if drive_root is None:
        return loop_outcome
    existing = load_task_result(drive_root, str(task.get("id") or "")) or {}
    unreconciled = existing.get("delegated_runs_unreconciled")
    if not isinstance(unreconciled, list) or not unreconciled:
        return loop_outcome
    overlaid = {
        **loop_outcome,
        "reason_code": "delegated_custody_unreconciled",
        "outcome_axes": custody_debt_axes(loop_outcome.get("outcome_axes")),
    }
    # A task truncated by the rails keeps ITS reason on the one Reason line; the
    # custody debt rides objective.warning(s) and the row's debt list instead.
    rail = str(loop_outcome.get("reason_code") or "")
    if rail in BEST_EFFORT_REASON_CODES:
        overlaid["reason_code"] = rail
    return overlaid

def emit_task_results(
    env: Any, memory: Any, llm: Any,
    pending_events: List[Dict[str, Any]],
    task: Dict[str, Any], text: str,
    usage: Dict[str, Any], llm_trace: Dict[str, Any],
    start_time: float, drive_logs: pathlib.Path,
    ctx: Any = None, event_queue: Any = None,
) -> None:
    """Emit all end-of-task events to supervisor and run post-task processing."""
    from ouroboros.subagent_bootstrap import actor_first_terminal_projection
    actor_fact, usage, llm_trace = actor_first_terminal_projection(ctx, task, usage, llm_trace, task.get("budget_drive_root") or getattr(env, "drive_root", None))
    loop_outcome = _derive_host_bound_loop_outcome(env, task, text, usage, llm_trace)
    receipt_root = pathlib.Path(str(getattr(env, "drive_root", None) or "."))
    receipt_rows = task_verification_receipts(ctx, receipt_root, task)
    # Apply FR3 once so events and the durable result share the same flagged outcome.
    apply_receipt_absent_flag(
        loop_outcome, llm_trace, receipt_root, str(task.get("id") or ""),
        expected_output=str(task.get("expected_output") or ""), receipts=receipt_rows,
    )
    outcome_axes = normalize_outcome_axes({"outcome_axes": loop_outcome.get("outcome_axes")})
    execution_status = str((outcome_axes.get("execution") or {}).get("status") or "")
    reason_code = str(loop_outcome.get("reason_code") or "")
    # CW3 (v6.34.0): a short same-route "turn=decision" turn (ephemeral, run while the
    # main agent is busy) DELIVERS its inline answer but must not leave a durable TASK
    # RECORD \u2014 no task_result file, no task_eval ledger row. The cognitive-memory writes
    # (reflection/consolidation/letters-home) are already gated further below; this
    # closes the remaining durable task-record writes. An inline answer + card
    # resolution and budget metrics still flow so the reply is visible.
    _ephemeral = bool(task.get("_ephemeral_turn"))
    _root_outbox = _is_root_post_task(task)   # durable outbox (no model call): pre-marker predicate
    if getattr(ctx, "_skip_post_task_synthesis", False):   # "Stop now": paid root predicates see it
        task["_skip_post_task_synthesis"] = True
    _presence = is_presence_task(task)
    _typed_routing_action = (
        str(getattr(ctx, "_typed_routing_action_emitted", "") or "").strip()
        if ctx is not None else ""
    )
    _has_typed_routing_action = bool(_ephemeral and _typed_routing_action)
    _message_meta = subagent_message_meta(task, task_id=str(task.get("id") or ""))
    if _ephemeral:
        _message_meta["ephemeral_decision"] = True
    send_event = {
        "type": "send_message", "chat_id": task["chat_id"],
        "text": text or "\u200b", "log_text": text or "",
        "format": "markdown",
        "task_id": task.get("id"), "ts": utc_now_iso(),
    }
    if _message_meta:
        # Final frames carry their own durable identity; replay must not depend
        # on a nearby progress row that may age out independently.
        send_event["progress_meta"] = dict(_message_meta)
    send_event = prepare_terminal_send_event(env.drive_root, task, text, usage, send_event, ephemeral=_ephemeral, presence=_presence)
    pending_events.append(build_presence_result_event(task, text, ctx) if _presence else send_event)
    duration_sec = round(time.time() - start_time, 3)
    n_tool_calls = len(llm_trace.get("tool_calls", []))
    n_tool_errors = sum(1 for tc in llm_trace.get("tool_calls", [])
                        if isinstance(tc, dict) and tc.get("is_error"))
    try:
        from supervisor.state import reconstruct_task_cost

        task_cost_fields = reconstruct_task_cost(
            str(task.get("id") or ""), fields=True,
            drive_root=pathlib.Path(task.get("budget_drive_root") or env.drive_root),
        )
    except Exception:
        log.error("Task cost authority unavailable at finalization", exc_info=True)
        task_cost_fields = {
            "cost_accounting_status": "unavailable", "cost_final": False,
            "cost_accounting_error": "ledger_unavailable",
            "accounted_upper_bound_usd": None, "total_rounds": None,
            "prompt_tokens": None, "completion_tokens": None,
            "reserved_usd": None, "unresolved_upper_bound_usd": None,
            "unknown_unmetered": None,
        }
    # SSOT cost naming (C2/ABI-3): the honest names on every terminal frame
    # this pipeline emits; the seam also strips any legacy alias spelling.
    from ouroboros.cost_projection import with_cost_aliases

    task_cost_fields = with_cost_aliases(task_cost_fields)
    if _is_root_post_task(task) and not _root_post_task_already_completed(env, task):
        task_cost_fields["cost_final"] = False
    if not _ephemeral:
        try:
            append_jsonl(drive_logs / "events.jsonl", {
                "ts": utc_now_iso(), "type": "task_eval", "ok": execution_status not in {EXECUTION_FAILED, EXECUTION_INFRA_FAILED},
                "task_id": task.get("id"), "task_type": task.get("type"),
                "outcome_axes": outcome_axes,
                "reason_code": reason_code,
                "review_eligibility": str(loop_outcome.get("review_eligibility") or ""),
                "review_trigger": str(loop_outcome.get("review_trigger") or ""),
                "duration_sec": duration_sec,
                "tool_calls": n_tool_calls,
                "tool_errors": n_tool_errors,
                "response_len": len(text),
            })
        except Exception:
            log.warning("Failed to log task eval event", exc_info=True)
            pass

    pending_events.append({
        "type": "task_metrics",
        "task_id": task.get("id"), "task_type": task.get("type"),
        "ephemeral_decision": _ephemeral,
        "outcome_axes": outcome_axes,
        "reason_code": reason_code,
        "duration_sec": duration_sec,
        "tool_calls": n_tool_calls, "tool_errors": n_tool_errors,
        **task_cost_fields,
        **({"resource_limit": dict(usage.get("resource_limit") or {})}
           if isinstance(usage.get("resource_limit"), dict) else {}),
        "ts": utc_now_iso(),
    })

    review_evidence: Dict[str, Any] = {}
    try:
        from ouroboros.review_evidence import collect_review_evidence

        review_evidence = collect_review_evidence(
            env.drive_root,
            task_id=str(task.get("id") or ""),
            repo_dir=getattr(env, "repo_dir", None),
        )
    except Exception:
        log.debug("Failed to collect review evidence", exc_info=True)

    if not _ephemeral:
        # GR2-5 (§8-A2, ONE outbox for EVERY root) + GR3-5 (ordering closes the
        # persist→register crash window): the final answer enters the durable
        # outbox — the owed row embeds the full payload — immediately BEFORE
        # the durable result write, regardless of the blocking/nonblocking
        # post-task split below. Registered-then-crashed leaves an owed row
        # boot replay delivers (projection-over-replay: no boot scan of
        # task_results is ever needed); the old stored-then-crashed order left
        # a terminal result nobody would ever deliver. The nonblocking lane
        # used to buffer the send with no delivery_id and no owed registration
        # at all. Seam + dedup: ouroboros/task_finalization.py.
        if _root_outbox and not _presence:
            stamp_root_final_phase(  # the stamp names the SAME word the durable row below settles to
                send_event, task, terminal_status=_durable_terminal_status(env, task, execution_status),
                post_task_open=not task.get("_skip_post_task_synthesis") and not _root_post_task_already_completed(env, task),
            )
            register_final_answer_owed(task, send_event, env_drive_root=env.drive_root)
        _store_task_result(
            env, task, text, usage, llm_trace, review_evidence=review_evidence,
            loop_outcome=loop_outcome, cost_fields=task_cost_fields,
        )
        stored_result = load_task_result(env.drive_root, str(task.get("id") or "")) or {}
    else:
        # No durable task_result file for a transient decision turn; the card still
        # resolves via task_done below (with empty artifact/review status).
        stored_result = {}
    artifact_bundle = stored_result.get("artifact_bundle") if isinstance(stored_result.get("artifact_bundle"), dict) else {}
    review_projection = stored_result.get("review_projection") or {}
    pending_events.append({
        "type": "task_done",
        "task_id": task.get("id"),
        "task_type": task.get("type"),
        # GR2-3c: the DURABLE status rides the event for honesty — the
        # supervisor validates every non-ephemeral task_done against the
        # durable row either way, but a stamped status makes the event
        # self-describing instead of a blank assertion. Ephemeral turns keep
        # a blank status (they have no durable lifecycle).
        "status": str(stored_result.get("status") or ""),
        # CW3: tells the supervisor's task_done handler to NOT synthesize a durable
        # missing-result task_result for a transient decision turn (which has none).
        "_ephemeral": _ephemeral,
        # Presentation marker only. The supervisor's typed routing event remains
        # the action/receipt authority; this keeps a transient decision task from
        # becoming a visible task card (and a bogus "Turn into project" target).
        "ephemeral_decision": _ephemeral,
        **({"typed_routing_action": _typed_routing_action} if _has_typed_routing_action else {}),
        # Carry the thread so the terminal card finalizes in its project panel
        # (per-thread fan-out), not just the main chat.
        "chat_id": int(task.get("chat_id") or 0),
        "outcome_axes": outcome_axes,
        "reason_code": reason_code,
        "artifact_status": stored_result.get("artifact_status") or artifact_bundle.get("status") or "",
        "artifact_bundle": artifact_bundle,
        "review_status": stored_result.get("review_status") if isinstance(stored_result.get("review_status"), dict) else {},
        **({"review_projection": review_projection} if review_projection.get("panels") else {}),
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
                "evolution_restart": False, "ts": utc_now_iso(),
            })
        ctx.pending_restart_reason = None
        ctx.pending_restart_is_evolution = False

    if _is_root_post_task(task):
        post_usage = dict(usage or {})
        post_usage["outcome_axes"] = outcome_axes
        post_usage["reason_code"] = reason_code
        # Ephemeral same-route turns (the "turn=decision" anti-freeze path while the
        # main agent is busy) are PROHIBITED from ALL durable memory: not only
        # reflection/evolution (below) but chat/scratchpad consolidation and project
        # letters-home too — the locked main path owns those (v6.33.0 WS10
        # idempotency contract; claudexor B5). ``_ephemeral`` is computed once near
        # the top of this function (it also gates the durable task-record writes).
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
        _project_task = _project_scoped and not _is_direct_chat and not _ephemeral
        if _project_task:
            # Letters home (v6.32.0): record the cycle in the project's own
            # journal and emit a concise completion digest for consciousness
            # (project_id + full objective + outcome). Full project awareness:
            # this is a crisp "task finished" summary, not an isolation boundary —
            # the one mind already sees the project thread in its unified memory;
            # only per-cycle RAW internal facts stay in the per-project store.
            _pid = resolve_project_id(task)
            # The full objective IS the meaning of the cycle — carry it whole into
            # the journal milestone and the consciousness digest (BIBLE P1: no
            # silent/lossy clip of cognitive text). Objectives are concise by
            # nature; the task and task_results remain the durable record.
            _objective = str(
                task.get("objective") or task.get("description") or task.get("text") or ""
            )
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
                    "objective": _objective,
                    "execution_status": _exec_status,
                    "objective_status": str((outcome_axes.get("objective") or {}).get("status") or "not_evaluated"),
                    "ts": utc_now_iso(),
                })
            except Exception:
                log.debug("project digest emission failed", exc_info=True)
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

        if not _ephemeral and not _root_post_task_already_completed(env, task):
            _dispatch_root_post_task(
                env, task, str(send_event.get("text") or ""), event_queue, pending_events,
                post_usage, llm_trace, review_evidence, drive_logs,
                budget_drive_root=budget_drive_root, split_drive=split_drive,
                project_scoped=_project_scoped, project_task=_project_task,
                parent_env=parent_env, parent_task=parent_task,
            )


def _dispatch_root_post_task(
    env: Any, task: Dict[str, Any], text: str,
    event_queue: Any, pending_events: List[Dict[str, Any]],
    post_usage: Dict[str, Any], llm_trace: Dict[str, Any],
    review_evidence: Dict[str, Any], drive_logs: pathlib.Path,
    *, budget_drive_root: str, split_drive: bool,
    project_scoped: bool, project_task: bool,
    parent_env: Any, parent_task: Dict[str, Any] | None,
) -> None:
    """Live final-answer delivery + sealed ground truth + post-task dispatch.

    The owner's answer goes out BEFORE blocking post-task cognition; task_done
    stays LAST via the buffered return (early task_done would release the
    queue slot / start child-drive cleanup mid-post-task). Rationale and the
    never-lost/never-doubled delivery contract: ouroboros/task_finalization.py.
    """
    split = split_drive and parent_env is not None and parent_task is not None
    global_reflection_callback = None
    if split and project_scoped:
        global_reflection_callback = functools.partial(
            _run_global_backlog_promotion_only, parent_env, parent_task)
    split_non_project = split and not project_scoped
    blocking = split_non_project or (
        str(task.get("type") or "") == "evolution"
        or bool(str(task.get("workspace_root") or "").strip())
        or bool(str(task.get("workspace_mode") or "").strip())
        or project_task
    )
    if blocking and event_queue is not None:
        # The CANONICAL data root — what the supervisor's boot/tick outbox
        # replay reads (§8-A2): the parent/budget root for split children, the
        # task's own drive for an ordinary root (whose ``budget_drive_root``
        # field is legitimately empty).
        deliver_final_message_live(
            event_queue, pending_events, str(task.get("id") or ""),
            drive_root=budget_drive_root or env.drive_root,
        )
    # Sealed ground truth for summary/reflection: what the owner actually
    # received + the durable result's own artifact facts (Q4A).
    sealed_final = build_sealed_final_package(
        load_task_result(env.drive_root, str(task.get("id") or "")), text,
    )
    if split_non_project:
        _run_post_task_processing_async(
            parent_env, parent_task, post_usage, llm_trace, review_evidence,
            pathlib.Path(budget_drive_root) / "logs",
            blocking=True, sealed_final=sealed_final,
        )
    else:
        _run_post_task_processing_async(
            env, task, post_usage, llm_trace, review_evidence, drive_logs,
            blocking=blocking,
            on_reflection=global_reflection_callback,
            sealed_final=sealed_final,
        )


def _durable_terminal_status(
    env: Any, task: Dict[str, Any], execution_status: str, *, existing: Dict[str, Any] | None = None,
) -> str:
    """The status the durable task row settles to (ONE rule for the row and its outbox stamp).

    A failed execution axis — including the owner's "Stop now", whose forced
    finalization stamps ``execution_status="failed"`` under
    ``owner_requested_finalization`` — and an already-failed row both settle
    ``failed``; everything else settles ``completed``.
    """
    if existing is None:
        existing = load_task_result(env.drive_root, str(task.get("id") or "")) or {}
    failed = (
        str(existing.get("status") or "") == STATUS_FAILED
        or execution_status in {EXECUTION_FAILED, EXECUTION_INFRA_FAILED}
    )
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
        from ouroboros.review_projection import publish_acceptance_checkpoint

        publish_acceptance_checkpoint(env, llm_trace, task_id=str(task.get("id") or ""),
                                      drive_root=env.drive_root, chat_id=task.get("chat_id"))
        trace_summary = build_trace_summary(llm_trace)
        from ouroboros.cost_projection import with_cost_aliases

        cost_fields = with_cost_aliases(cost_fields or {
            "cost_accounting_status": "unavailable", "cost_final": False,
            "cost_accounting_error": "ledger_projection_missing",
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
        loop_outcome = _apply_terminal_custody_outcome(env, task, loop_outcome)
        outcome_axes = normalize_outcome_axes({"outcome_axes": loop_outcome.get("outcome_axes")})
        execution_status = str((outcome_axes.get("execution") or {}).get("status") or "")
        reason_code = str(loop_outcome.get("reason_code") or "")
        status = _durable_terminal_status(env, task, execution_status, existing=existing)
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
                and _post_task_synthesis_is_terminal(
                    existing_checkpoint.get("post_task_synthesis")
                )
            ):
                root_phase_checkpoint = dict(existing_checkpoint)
            elif isinstance(incoming_checkpoint, dict):
                root_phase_checkpoint = dict(incoming_checkpoint)
            elif isinstance(existing_checkpoint, dict):
                root_phase_checkpoint = dict(existing_checkpoint)
            else:
                root_phase_checkpoint = {
                    "phase": "task_acceptance",
                    "status": "not_required",
                    "pass_index": 0,
                }
            root_phase_checkpoint.setdefault("post_task_synthesis", "pending_once")
        review_projection = _compact_review_projection(llm_trace)
        write_task_result(
            env.drive_root,
            str(task.get("id") or ""),
            status,
            reason_code=reason_code,
            outcome_axes=outcome_axes,
            # Compatibility mirror consumed by the gateway and task_done event.
            # ``outcome_axes.review`` remains the canonical structured axis.
            review_status=dict(outcome_axes.get("review") or {}),
            accounted_upper_bound_usd_with_children=_cost_with_children,
            cost_with_children_partial=_cost_partial,
            task_contract=task_contract,
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
            task_group_id=task.get("task_group_id"),
            task_group=task.get("task_group"),
            subagent_envelope=subagent_envelope,
            configured_subagent=task.get("configured_subagent"),
            parent_cognitive_route=task.get("parent_cognitive_route"),
            subagent_availability=task.get("subagent_availability"),
            metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
            result=text or "", **terminal_result_fields(usage),
            final_answer=str(loop_outcome.get("final_answer") or ""),
            trace_summary=trace_summary,
            trace_refs=loop_outcome.get("trace_refs") or {},
            **cost_fields,
            review_evidence=review_evidence or {},
            completion_observations=observations,
            **({"review_projection": review_projection} if review_projection.get("panels") else {}),
            verification_ledger=verification_refs.get("inline"),
            artifact_bundle=artifact_bundle,
            artifacts=artifacts,
            **({"root_phase_checkpoint": root_phase_checkpoint} if root_phase_checkpoint else {}),
            **({"swarm_efficiency": swarm_efficiency} if swarm_efficiency else {}),
            ts=utc_now_iso(),
        )
    except Exception as e:
        log.warning("Failed to store task result: %s", e)


def build_review_context(env: Any) -> str:
    """Build a compact review continuity section for the main reasoning context."""
    try:
        from ouroboros.review_state import (
            _LEGACY_CURRENT_REPO_KEY,
            advisory_commit_ready,
            compute_snapshot_hash,
            format_status_section,
            load_state,
            make_repo_key,
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
        snapshot_hash = compute_snapshot_hash(repo_dir)
        open_obs = state.get_open_obligations(repo_key=repo_key)
        open_debts = state.get_open_commit_readiness_debts(repo_key=repo_key)
        if (
            not state.advisory_runs
            and not state.latest_attempt()
            and not continuations
            and not corrupt
            and not open_obs
            and not open_debts
        ):
            return ""

        current_run = None
        for run in reversed(state.advisory_runs):
            if run.snapshot_hash != snapshot_hash:
                continue
            if run.repo_key not in ("", repo_key, _LEGACY_CURRENT_REPO_KEY):
                continue
            current_run = run
            break

        # H5 (capinv-447): honestly named — this is the ADVISORY readiness
        # projection, not the full commit gate (triad/scope/custody independent).
        lines: List[str] = ["## Review Continuity", "### Advisory readiness (not the full commit gate)"]
        live_status = str(getattr(current_run, "status", "") or "missing")
        repo_commit_ready = advisory_commit_ready(
            current_run is not None and current_run.status in ("fresh", "bypassed", "skipped"),
            open_obs, open_debts,
            matching_run=current_run if getattr(current_run, "repo_key", None) == repo_key else None,
        )
        lines.append(f"- repo_key={repo_key}")
        lines.append(f"- snapshot_hash={snapshot_hash[:12] or '(empty)'}")
        lines.append(f"- advisory_status={live_status}")
        lines.append(f"- repo_commit_ready={'yes' if repo_commit_ready else 'no'}")
        if current_run is not None:
            lines.append(f"- current_review_ts={str(current_run.ts or '')[:19]}")
            if current_run.bypass_reason:
                lines.append(f"- bypass_reason={_truncate_with_notice(current_run.bypass_reason, 220)}")
        else:
            lines.append("- no advisory run matches the current worktree snapshot")

        stale_matches_repo = not state.last_stale_repo_key or state.last_stale_repo_key == repo_key
        if state.last_stale_from_edit_ts and stale_matches_repo:
            lines.append(
                f"- stale_marker={state.last_stale_from_edit_ts[:19]}: "
                f"{_truncate_with_notice(state.last_stale_reason or 'worktree edit invalidated advisory freshness', 220)}"
            )

        if open_debts:
            lines.append("- retry_anchor=commit_readiness_debt")
            lines.append(f"- commit_readiness_debt={len(open_debts)}")
            lines.append("\n### Commit-readiness debt (start retry here)")
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

        history = format_status_section(state, repo_dir=repo_dir)
        if history:
            history = history.replace("## Advisory Pre-Review Status", "### Historical review ledger")
            lines.append("\n" + history)

        return "\n".join(lines)
    except Exception:
        log.debug("Failed to build review continuity context", exc_info=True)
        return ""

# The v7 post-task-synthesis split: the members below moved into
# ouroboros/post_task_synthesis.py; this facade keeps their historical
# ouroboros.agent_task_pipeline bindings for consumers.
from ouroboros.post_task_synthesis import (  # noqa: E402, F401 -- intentional public re-exports
    build_trace_summary,
    _update_improvement_backlog,
    _apply_reflection_memory_actions,
    _child_task_evidence,
    _pre_synthesis_usage_snapshot,
    _compact_review_projection,
    _run_task_summary,
    _run_chat_consolidation,
    _run_scratchpad_consolidation,
    _run_reflection,
    _TASK_SUMMARY_PROMPT,
)
