"""Post-task synthesis workers for the task pipeline (v7 L-C2 split).

The LLM-heavy best-effort memory work the post-task orchestrator
(``agent_task_pipeline._run_post_task_processing_async``) dispatches after a
task ends: the tool-trace summary, the free host facts row, chat/scratchpad
consolidation, the execution reflection with its child-task evidence, the
durable improvement backlog and reflection memory actions, plus the shared
pre-synthesis usage snapshot and the compact review projection those prompts
embed. Extracted from agent_task_pipeline.py; the pipeline re-exports every
name, so historical imports and monkeypatch targets keep working."""

from __future__ import annotations

import json
import logging
import pathlib

from dataclasses import replace
from typing import Any, Callable, Dict
from ouroboros.dialogue_provenance import presence_provenance_fields
from ouroboros.llm_claudexor import propagate_model_error
from ouroboros.outcomes import normalize_outcome_axes
from ouroboros.subagent_messages import initiator_meta
from ouroboros.synthesis_cost_text import _summary_row_cost_fields, _synthesis_cost_usd, _synthesis_usage_snapshot_text
from ouroboros.task_finalization import sealed_final_prompt_section
from ouroboros.tool_capabilities import routing_action_for_tool
from ouroboros.utils import append_jsonl, truncate_review_artifact as _truncate_with_notice, utc_now_iso


log = logging.getLogger("ouroboros.agent_task_pipeline")


def _atp():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from ouroboros import agent_task_pipeline

    return agent_task_pipeline


def task_tool_metrics(llm_trace: dict) -> dict:
    """Project recorded calls once; unknown names never become an empty census."""
    unavailable = bool(llm_trace.get("loop_evidence_unavailable"))
    calls = llm_trace.get("tool_calls") or []
    metrics = {
        "tool_calls": None if unavailable else len(calls),
        "tool_errors": None if unavailable else sum(
            1 for call in calls if isinstance(call, dict) and call.get("is_error")),
        # The addressing calls among them (tool_capabilities owns the family), so
        # a replayed block can tell a receipt-only turn from real work without
        # a client list of tool names.
        "routing_tool_calls": None if unavailable else sum(
            1 for call in calls if isinstance(call, dict) and routing_action_for_tool(call.get("tool"))),
        "tool_call_counts": None,
    }
    if unavailable or llm_trace.get("recovered_post_task_synthesis") or not isinstance(llm_trace.get("tool_calls"), list):
        return metrics
    counts: dict[str, int] = {}
    for call in calls:
        name = call.get("tool") if isinstance(call, dict) else None
        if not isinstance(name, str) or not name.strip():
            return metrics
        name = name.strip()
        counts[name] = counts.get(name, 0) + 1
    metrics["tool_call_counts"] = counts
    return metrics


def capture_task_inputs(ctx: Any, task: dict, drive_root: Any, receipts: list) -> dict:
    """Freeze the existing owner corpus and check receipts before actor cleanup."""
    from ouroboros._outcome_receipts import verification_receipt_ledger_row
    from ouroboros.observability import redact_projection
    from ouroboros.review_evidence_sections import (
        _accept_owner_directives, _accept_verification_summary,
    )

    task_id = str(task.get("id") or task.get("task_id") or "")
    result: dict = {
        "version": 1, "task_id": task_id,
        "source_ref": {"kind": "task_result", "task_id": task_id, "reader": "get_task_result"},
        "unavailable_sections": [],
    }
    try:
        # The run's provenance comes first: the synthesis reads who started the run
        # and whether the owner door stamped it before it reads the first text.
        from ouroboros.dialogue_provenance import run_origin

        metadata = getattr(ctx, "task_metadata", None)
        result["run_origin"] = run_origin({
            **task, "metadata": metadata if isinstance(metadata, dict) else task.get("metadata"),
        })
    except Exception:
        result["unavailable_sections"].append("run_origin")
        log.warning("Task run origin unavailable for synthesis: %s", task_id, exc_info=True)
    try:
        result["owner_requirements_and_decisions"] = _accept_owner_directives(ctx, drive_root, task_id)
    except Exception:
        result["unavailable_sections"].append("owner_requirements_and_decisions")
        log.warning("Task owner input unavailable for synthesis: %s", task_id, exc_info=True)
    try:
        result["verification_summary"] = _accept_verification_summary(receipts)
        result["verification_receipts"] = [
            {"ts": row.get("ts"), **verification_receipt_ledger_row(row)}
            for row in receipts if isinstance(row, dict)
        ]
    except Exception:
        result["unavailable_sections"].append("verification_receipts")
        log.warning("Task verification input unavailable for synthesis: %s", task_id, exc_info=True)
    # Copy once into the normal durable completion package. Prompt workers must
    # neither retain mutable actor objects nor re-read a later task's messages.
    return json.loads(json.dumps(redact_projection(result).value, ensure_ascii=False, default=str))



def _trace_round(tc: dict) -> int | None:
    """Ordinal of the model round that issued this call (``…:round:<n>``); None when unrecorded."""
    tail = str(tc.get("round_id") or "").rpartition(":round:")[2]
    return int(tail) if tail.isdigit() else None


def _fold_identical_calls(tool_calls: list) -> list[tuple[int, dict, int, int | None, int | None]]:
    """Run-length fold of consecutive IDENTICAL calls: (first index, row, count, first round, last round).

    Identity is equality of tool, RECORDED arguments, recorded status and delivered result —
    arithmetic, not vocabulary — so a refusal returned as a plain string, a sleep loop and a
    blind poll fold exactly like a typed error. Nothing is dropped: the count stays on the row.
    Recorded is the load-bearing word, because these arguments already passed the log
    sanitizer: an oversized string keeps its length and sha in the marker (two different
    large payloads still differ) and a truncated list keeps its remaining count (so tails
    of DIFFERENT length still differ), but two same-length tails with different content, a
    structure past depth 3, and two different secrets (both ``*** REDACTED ***``) collapse
    to a shape that compares equal and folds into one ``×2`` row. The returned row is the
    group's FIRST call, whose trace_ref addresses that call's exact recorded projection.
    """
    folded: list[list] = []
    previous = None
    for index, tc in enumerate(tool_calls):
        key = (tc.get("tool"), json.dumps(tc.get("args"), sort_keys=True, default=str, ensure_ascii=False),
               str(tc.get("status") or ""), bool(tc.get("is_error")), str(tc.get("result") or ""))
        if folded and key == previous:
            folded[-1][2] += 1
            folded[-1][4] = _trace_round(tc)
        else:
            folded.append([index, tc, 1, _trace_round(tc), _trace_round(tc)])
        previous = key
    return [tuple(item) for item in folded]


def build_trace_summary(llm_trace: dict, *, all_calls: bool = False) -> str:
    """Return a human-readable summary of tool calls and agent notes.

    The default is the BOUNDED PREVIEW that is stored and shown (task card, parents,
    children): two arguments per call, a positional window past thirty calls, a total
    cut. ``all_calls=True`` is the post-task reflection's listing: every call in order
    with every argument, identical consecutive calls folded into one ``×N`` row, the
    first line of a failed or repeated call's result, no window and no total cut — a
    decider must not adjudicate less of a call than the actor saw; its prompt is fitted
    by the consolidation seam. Values stay width-bounded, with a disclosed omission.
    """
    if llm_trace.get("loop_evidence_unavailable"):
        return "## Tool trace (call count unknown)\nThe failed loop supplied no verified execution trace."
    tool_calls = llm_trace.get("tool_calls", []) or []
    notes = llm_trace.get("reasoning_notes", []) or []

    n = len(tool_calls)
    # v6.57.0 — honest breakdown so a task that finished with a deliverable is not
    # mislabeled "43 errors" (the site/PB incidents): separate GENUINE unresolved
    # errors from POLICY denials, cosmetic non-zero exits, recovered errors, and
    # ignored read-only blocks. Self-learning (reflection reads this) must not be
    # poisoned by counting policy refusals or intentional probe exits as failures.
    from ouroboros.outcomes import _classify_tool_errors
    from ouroboros.reflection import _trace_call_errored  # the ONE reading of "this call went wrong"

    _buckets = _classify_tool_errors(llm_trace)
    _unresolved = len(_buckets.get("unresolved") or [])
    _policy = len(_buckets.get("policy_denials") or [])
    _cosmetic = len(_buckets.get("cosmetic") or [])
    _recovered = len(_buckets.get("recovered") or [])
    _ignored = len(_buckets.get("ignored") or [])
    _breakdown_bits = [f"{_unresolved} errors"]
    if _policy:
        _breakdown_bits.append(f"{_policy} policy-denied")
    if _recovered:
        _breakdown_bits.append(f"{_recovered} recovered")
    if _cosmetic:
        _breakdown_bits.append(f"{_cosmetic} cosmetic")
    if _ignored:
        _breakdown_bits.append(f"{_ignored} ignored")
    if all_calls:
        # One arithmetic fact the rows cannot show at a glance: rounds in which NO call returned ok.
        rounds: dict[int, bool] = {}
        for tc in tool_calls:
            number = _trace_round(tc)
            if number is not None:
                rounds[number] = rounds.get(number, True) and _trace_call_errored(tc)
        if any(rounds.values()):
            _breakdown_bits.append(f"{sum(rounds.values())} of {len(rounds)} rounds had only non-ok results")

    lines: list[str] = [f"## Tool trace ({n} calls, {', '.join(_breakdown_bits)})"]

    if not tool_calls:
        lines.append("No tool calls.")
    else:
        from ouroboros.observability import redact_projection

        def _fmt_call(first: int, tc: dict, count: int, round_a: int | None, round_b: int | None) -> str:
            name = tc.get("tool", "unknown")
            args = tc.get("args", {})
            if isinstance(args, dict):
                parts = []
                arg_items = list(args.items())
                for k, v in (arg_items if all_calls else arg_items[:2]):
                    v_str = str(v)
                    if len(v_str) > 200:
                        v_str = _truncate_with_notice(v_str, 200).replace("\n", " ")
                    parts.append(f"{k}={v_str!r}")
                if not all_calls and len(arg_items) > 2:
                    parts.append(f"⚠️ OMISSION NOTE: {len(arg_items) - 2} more args omitted")
                args_str = ", ".join(parts)
            else:
                args_str = repr(args)
                if len(args_str) > 200:
                    args_str = _truncate_with_notice(args_str, 200).replace("\n", " ")
            facts = []
            status = str(tc.get("status") or "").strip()
            if status:
                facts.append(f"status={status}")
            if tc.get("exit_code") is not None:
                facts.append(f"exit_code={tc.get('exit_code')}")
            if tc.get("signal"):
                facts.append(f"signal={tc.get('signal')}")
            fact_suffix = f" [{', '.join(facts)}]" if facts else ""
            suffix = " → ERROR" if tc.get("is_error") else ""
            index = f"{first + 1}" if count == 1 else f"{first + 1}–{first + count}"
            if count > 1:
                span = "" if round_a is None else f", rounds {round_a}–{round_b}" if round_b != round_a else f", round {round_a}"
                suffix += f" ×{count} identical{span}"
            if all_calls and (count > 1 or _trace_call_errored(tc)):
                # The answer is what a later reader needs to tell a refusal from progress.
                head = str(redact_projection(str(tc.get("result") or "")).value).strip().splitlines()[:1]
                if head:
                    suffix += " ← " + _truncate_with_notice(head[0], 200)
            return f"{index}. {name}({args_str}){fact_suffix}{suffix}"

        if all_calls:
            shown = [_fmt_call(*row) for row in _fold_identical_calls(tool_calls)]
        elif n > 30:
            shown = (
                [_fmt_call(i, tool_calls[i], 1, None, None) for i in range(15)]
                + [f"⚠️ OMISSION NOTE: {n - 30} middle tool calls omitted from trace summary."]
                + [_fmt_call(n - 15 + i, tool_calls[n - 15 + i], 1, None, None) for i in range(15)]
            )
        else:
            shown = [_fmt_call(i, tool_calls[i], 1, None, None) for i in range(n)]
        lines.extend(shown)

    if notes:
        lines.append("\n## Agent notes (supplementary, not source of truth)")
        lines.extend(f"- {note}" for note in notes)

    summary = "\n".join(lines)
    if not all_calls and len(summary) > 4000:
        summary = _truncate_with_notice(summary, 4000)
    return summary


def _update_improvement_backlog(
    env: Any,
    reflection_entry: Dict[str, Any] | None,
) -> int:
    """Persist LLM-nominated follow-up improvements into the durable backlog.

    Returns the number appended; 0 is a genuine no-op (nothing nominated), never a
    swallowed failure. An append or grooming failure raises to the promotion stage,
    which isolates an ordinary one and stops later paid work on an interruption.
    """
    from ouroboros.improvement_backlog import append_backlog_items, groom_backlog

    candidates = list((reflection_entry or {}).get("backlog_candidates") or [])
    if not candidates:
        return 0
    added = append_backlog_items(env.drive_root, candidates)
    groom_backlog(env.drive_root)  # size-triggered; no-op while small
    return added


def _apply_reflection_memory_actions(
    env: Any,
    reflection_entry: Dict[str, Any] | None,
    project_id: str = "",
) -> int:
    """Auto-apply LLM-nominated durable memory actions from the experience review.

    Runs against ``env.drive_root``; for forked/workspace tasks the finalizer
    also invokes post-task processing with the parent drive, so learnings land
    on the canonical drive rather than a discarded child drive.
    """
    try:
        actions = list((reflection_entry or {}).get("memory_actions") or [])
        if not actions:
            return 0
        from ouroboros.reflection import apply_memory_actions

        return apply_memory_actions(env, actions, project_id=project_id)
    except Exception:
        log.debug("Reflection memory action application failed", exc_info=True)
        return 0


def _child_failure_classes(rows: Any) -> list:
    """The sorted TYPED execution FAILURE classes among the children.

    Read off the same ``outcome_axes`` the evidence walk already normalized, so
    the root's reflection can carry what its subtree did without a second
    collector, a second walk, or children reflecting on their own.

    Only genuine failures count. "Not ok" is a much wider set: a child the parent
    cancelled in an ordinary cascade, one that soft-landed ``best_effort`` on a
    rail, a ``degraded`` one, and an ``interrupted`` one that is not even
    terminal all end non-ok without anything having gone wrong, and admitting
    them opened the Pattern Register - a paid rewrite of the register - on clean
    roots with nothing to learn."""
    from ouroboros.outcomes import EXECUTION_FAILED, EXECUTION_INFRA_FAILED

    failures = {EXECUTION_FAILED, EXECUTION_INFRA_FAILED}
    classes = set()
    for row in rows or []:
        axes = row.get("outcome_axes") if isinstance(row, dict) else None
        execution = axes.get("execution") if isinstance(axes, dict) else None
        status = str(execution.get("status") or "").strip() if isinstance(execution, dict) else ""
        if status in failures:
            classes.add(status)
    return sorted(classes)


def _child_engine_facts(item: Dict[str, Any]) -> Dict[str, Any]:
    """WHO ran a child and for how long, from the child's OWN stored record.

    Facts only: the frozen ``configured_subagent`` snapshot names the engine the
    way the catalog names a row, but from what actually ran (never the live
    roster, which would relabel the past), so a lowered access or a legacy twin
    can read differently from today's catalog. ``used_model`` is reported for an API child alone — a
    session child's ``model_execution`` describes its nanny's rounds, not the
    leaf. A duration needs both stamps: only the ordinary terminal write stamps
    ``ts``, so a row whose ``ts`` does not follow its start yields none.
    """
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.subagent_history import execution_identity, snapshot_handle

    facts: Dict[str, Any] = {}
    snapshot = item.get("configured_subagent")
    if isinstance(snapshot, dict) and isinstance(snapshot.get("route"), dict):
        identity = execution_identity(snapshot)
        facts["engine"] = {
            "subagent_id": snapshot_handle(snapshot), "kind": identity["kind"],
            "target": identity["target_id"],
            **{key: identity[key] for key in ("effort", "access") if identity.get(key)},
        }
        execution = item.get("model_execution")
        if identity["kind"] == "api_model" and isinstance(execution, dict) and execution.get("used_model"):
            facts["used_model"] = execution["used_model"]
    started, finished = parse_deadline_ts(item.get("started_at")), parse_deadline_ts(item.get("ts"))
    if started is not None:
        facts["started_at"] = item["started_at"]
        if finished is not None and finished > started:
            facts["duration_sec"] = round((finished - started).total_seconds(), 1)
    return facts


def _child_task_evidence(env: Any, task: Dict[str, Any], limit: int = 6000) -> tuple:
    """Compact evidence from child/subagent results for parent experience review.

    Returns the prompt text AND the rows it was rendered from: the caller needs
    the typed child outcomes, and one walk is the only walk (P7)."""
    task_id = str(task.get("id") or "")
    if not task_id:
        return "", []
    try:
        from ouroboros.cost_projection import resolve_cost_pair
        from ouroboros.task_results import list_task_results

        rows = []
        for item in list_task_results(env.drive_root):
            if not isinstance(item, dict):
                continue
            if str(item.get("task_id") or item.get("id") or "") == task_id:
                continue  # the persisted root names its own subtree too
            if str(item.get("parent_task_id") or "") != task_id and str(item.get("root_task_id") or "") != task_id:
                continue
            # ABI-3: resolve the stored pair (legacy read tolerance, deprecated
            # wins) but emit only the honest name into the evidence row.
            _, child_cost = resolve_cost_pair(
                item, "accounted_upper_bound_usd", "cost_usd")
            rows.append({
                "task_id": item.get("task_id") or item.get("id"),
                "status": item.get("status"),
                "role": item.get("role"),
                **_child_engine_facts(item),
                "outcome_axes": normalize_outcome_axes(item),
                "accounted_upper_bound_usd": child_cost,
                "trace_summary": _truncate_with_notice(item.get("trace_summary", ""), 800),
                "result": _truncate_with_notice(item.get("result", ""), 1600),
            })
        if not rows:
            return "", []
        # Verbose rows overflow the cap after about three children, so a compact
        # line per child leads: who ran what survives the truncation for ALL of them.
        overview = [
            {"task_id": row["task_id"], "role": row["role"], "status": row["status"],
             **({"engine": row["engine"]["subagent_id"]} if "engine" in row else {}),
             **({"duration_sec": row["duration_sec"]} if "duration_sec" in row else {}),
             "accounted_upper_bound_usd": row["accounted_upper_bound_usd"]}
            for row in rows
        ]
        text = json.dumps({"children_overview": overview, "children": rows}, ensure_ascii=False, indent=2)
        return _truncate_with_notice(text, limit), rows
    except Exception:
        log.debug("Failed to collect child task evidence", exc_info=True)
        return "", []


def _pre_synthesis_usage_snapshot(
    env: Any,
    task: Dict[str, Any],
    usage: Dict[str, Any],
) -> Dict[str, Any]:
    """Freeze one honest, non-final root/subtree cost view for synthesis.

    Summary and reflection share this loop-local dictionary.  The existing
    terminal checkpoint remains the sole final authority after their own model
    calls settle.
    """
    snapshot = json.loads(json.dumps(usage, ensure_ascii=False, default=str))
    if not _atp()._is_root_post_task(task):
        return snapshot

    task_id = str(task.get("id") or task.get("task_id") or "")
    budget_root = pathlib.Path(
        task.get("budget_drive_root") or getattr(env, "drive_root", ".")
    )
    snapshot.update({
        "cost_snapshot_at": utc_now_iso(),
        "cost_final": False,
        "cost_with_children_partial": True,
    })
    try:
        from ouroboros.usage_accounting import usage_breakdown
        from ouroboros.cost_projection import COST_SCOPE_ROOT_TREE, build_cost_presentation

        logical_root_id = str(task.get("root_task_id") or task_id)
        subtree = usage_breakdown(budget_root, root_task_id=logical_root_id)
        snapshot.update({
            "accounted_upper_bound_usd_with_children": round(float(subtree["accounted_usd"]), 6),
            "cost_presentation": build_cost_presentation(subtree, scope=COST_SCOPE_ROOT_TREE),
            "reserved_usd": round(float(subtree["reserved_usd"]), 6),
            "unresolved_upper_bound_usd": round(
                float(subtree["unresolved_upper_bound_usd"]), 6
            ),
            "unknown_unmetered": int(subtree["unknown_unmetered"]),
            "ledger_integrity": (
                "degraded" if bool(subtree.get("integrity_degraded")) else "ok"
            ),
            "cost_accounting_status": "available",
        })
    except Exception:
        log.warning(
            "Pre-synthesis subtree cost is unavailable for %s",
            task_id or "unknown",
            exc_info=True,
        )
        snapshot.update({
            "accounted_upper_bound_usd_with_children": None,
            "reserved_usd": None,
            "unresolved_upper_bound_usd": None,
            "unknown_unmetered": None,
            "ledger_integrity": "unavailable",
            "cost_accounting_status": "unavailable",
            "cost_presentation": None,
        })
    return snapshot


def _compact_review_projection(llm_trace: Dict[str, Any]) -> Dict[str, Any]:
    """Build the public review projection without copying raw actor output."""
    try:
        from ouroboros.review_substrate import compact_review_projection

        return compact_review_projection(llm_trace.get("review_runs") or [])
    except Exception:
        log.debug("Failed to build compact review projection", exc_info=True)
        return {"panels": []}


def _record_task_facts(env: Any, task: Dict[str, Any], usage: Dict[str, Any],
                       llm_trace: Dict[str, Any], drive_logs: pathlib.Path) -> None:
    """Append the task's free host facts row to chat.jsonl: no model call, no prose.

    The owner removed the paid task narrative (TZ-2 decision 2=A). This row keeps
    the facts its readers take from a ``task_summary`` row when the result file is
    gone: the direct-turn fact, origin label, typed routing action, tool metrics,
    rounds, cost and review projection. Its kind is ``host_task_facts``, never
    ``authored_root_summary``, and it persists no continuation narrative, so a
    Main continuation sees the typed ``continuation_narrative_unavailable`` gap.
    Text stays empty: existing Project and result references own navigation.
    """
    task_id = str(task.get("id") or "unknown")
    try:
        from ouroboros.project_dialogue import append_canonical_task_summary, completion_status_label, outcome_phase
        from ouroboros.task_finalization import artifact_store_roots, rescued_files_fact

        canonical_root = pathlib.Path(task.get("budget_drive_root") or drive_logs.parent)
        result_root = pathlib.Path(getattr(env, "drive_root", canonical_root))
        stored_result = _atp().load_task_result(result_root, task_id) or {}
        review_projection = _compact_review_projection(llm_trace)
        # TZ-2 C2: how many files the task rescued into its store(s) — positive, zero or
        # unknown — by stat alone; the fact discloses that no hash was computed. A split
        # non-Project root synthesizes on the canonical drive (parent env and task): its
        # actor store is then the row's recorded ``child_drive_root``, else this drive.
        canonical = result_root.resolve(strict=False) == canonical_root.resolve(strict=False)
        files_rescued = rescued_files_fact(task_id, artifact_store_roots(
            canonical_root, task_id, task=task, child_root=None if canonical else result_root))
        append_canonical_task_summary(canonical_root, {
            "ts": utc_now_iso(), "direction": "system", "type": "task_summary",
            "summary_kind": "host_task_facts", "summary_id": f"task-facts:{task_id}",
            "task_id": task_id, "parent_task_id": str(task.get("parent_task_id") or ""), "root_task_id": str(task.get("root_task_id") or task_id),
            "project_id": str(task.get("project_id") or ""), "chat_id": int(task.get("chat_id") or 0), "delegation_role": str(task.get("delegation_role") or ""), "role": str(task.get("role") or ""),
            "status": str(stored_result.get("status") or "completed"), "outcome": completion_status_label(stored_result, usage), "outcome_phase": outcome_phase(stored_result, usage),
            "outcome_final": False, "outcome_authority": "pre_finalization_host_facts",
            # The chat block reads its chrome, the addressing fact and the
            # origin label from this row when the task result has been pruned.
            "_is_direct_chat": bool(task.get("_is_direct_chat")), **initiator_meta(task),
            **({"typed_routing_action": str(usage["typed_routing_action"])} if usage.get("typed_routing_action") else {}),
            "text": "", **task_tool_metrics(llm_trace),
            "rounds": None if usage.get("loop_evidence_unavailable") else int(usage.get("rounds") or 0),
            "outcome_axes": normalize_outcome_axes(usage), "reason_code": str(usage.get("reason_code") or ""),
            "result_ref": {"kind": "task_result", "task_id": task_id, "reader": "get_task_result"},
            "files_rescued": files_rescued,
            **_summary_row_cost_fields(usage), **presence_provenance_fields(task),
            **({"review_projection": review_projection} if review_projection.get("panels") else {}),
        })
    except Exception:
        log.warning("Task facts row was not recorded for %s (non-critical)", task_id, exc_info=True)


POST_TASK_INTERRUPT_KINDS = frozenset({"budget_exhausted", "provider_outcome_unknown"})


def propagate_paid_interruption(error: BaseException) -> None:
    """Re-raise what must stop later paid post-work; return for an ordinary failure.

    ``propagate_model_error`` carries the control and typed unknown-provider facts;
    the wallet's ``BudgetExceeded`` and an unresolved attempt on ANY provider's
    exception chain — the consolidator's own classifier, never a broadened global
    one — are the others (TZ-2 C3). A stage adapter that only logged them let the
    coordinator run the next paid stage and write ``completed``. Anything else
    returns, so the caller isolates the failure to its own stage.
    """
    propagate_model_error(error)
    from ouroboros.transport_custody import outcome_unknown_on_chain
    from ouroboros.usage_accounting import BudgetExceeded

    if isinstance(error, BudgetExceeded) or outcome_unknown_on_chain(error):
        raise error


def post_task_interruption(control: BaseException) -> str:
    """The closed stop-reason word for a control that ended paid post-work.

    A model wait carries its ``control_reason``; a typed provider fact keeps its
    code; an unknown outcome (a dispatched attempt without a terminal provider
    fact) is always ``provider_outcome_unknown`` — the transport's own
    ``model_outcome_unknown`` spelling never reaches the checkpoint.
    """
    reason = str(getattr(control, "control_reason", "") or "")
    if reason:
        return reason
    code = str(getattr(control, "code", "") or "")
    from ouroboros.transport_custody import outcome_unknown_on_chain

    if not code or code == "model_outcome_unknown" or outcome_unknown_on_chain(control):
        return "provider_outcome_unknown"
    return code


def _post_task_paid_interruption(errors: Any) -> str:
    """The stage's typed outcome from returned error facts: '' when clean.

    Memory consolidation returns errors to keep completed chunks. Only this
    stage adapter interprets those existing facts: a budget or unknown-provider
    kind wins and stops later paid post-work; any other kind names the last
    UNRESOLVED ordinary failure, so a stage that lost a chunk reads ``degraded``
    like a stage that raised (TZ-2 C3: unfinished stages are never ``completed``).
    The history keeps every attempt; a refusal its producer answered (a split whose
    halves carry their own rows, ``resolution``) is not an unfinished stage.
    """
    rows = [row for row in (errors if isinstance(errors, list) else []) if isinstance(row, dict)]
    for row in rows:
        if row.get("kind") in POST_TASK_INTERRUPT_KINDS:
            return str(row["kind"])
    unresolved = [row for row in rows if not row.get("resolution")]
    return str((unresolved[-1].get("kind") or "stage_error")) if unresolved else ""


def _run_chat_consolidation(env, memory, llm, task, drive_logs):
    """Run dialogue-block consolidation inside the root post-task worker."""
    try:
        from ouroboros import consolidator as _c

        should_consolidate = _c.should_consolidate
        consolidate = _c.consolidate
        chat_path = drive_logs / "chat.jsonl"
        blocks_path = env.drive_path("memory") / "dialogue_blocks.json"
        meta_path = env.drive_path("memory") / "dialogue_meta.json"
        if should_consolidate(meta_path, chat_path):
            _id, _ident, _llm, _logs = task.get("id"), memory.load_identity(), llm, drive_logs
            from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

            base_scope = current_usage_scope()
            chat_scope = (
                replace(base_scope, category="consolidation", source="chat_consolidation")
                if base_scope is not None
                else UsageScope(
                    drive_root=task.get("budget_drive_root") or env.drive_root,
                    task_id=str(_id or ""),
                    root_task_id=str(task.get("root_task_id") or _id or ""),
                    category="consolidation",
                    source="chat_consolidation",
                )
            )

            with usage_scope(chat_scope):
                from ouroboros.tools.registry import ToolContext
                knowledge_context = ToolContext(
                    repo_dir=getattr(env, "repo_dir", env.drive_root),
                    drive_root=pathlib.Path(task.get("budget_drive_root") or env.drive_root),
                    budget_drive_root=str(task.get("budget_drive_root") or env.drive_root),
                    task_id=str(_id or ""), project_id=str(task.get("project_id") or ""))
                u = consolidate(chat_path=chat_path, blocks_path=blocks_path,
                                meta_path=meta_path, llm_client=_llm, identity_text=_ident,
                                knowledge_context=knowledge_context,
                                room_registry_root=knowledge_context.budget_drive_root)
            if u:
                # A run that produced no block and a run that never happened look the
                # same in this stream without a written count; last_error_kind names the
                # LAST error any attempt recorded (recovered splits keep theirs), which
                # is weaker than "the run failed" and is reported under that honest name.
                errors = u.get("_consolidation_errors") or []
                append_jsonl(_logs / "events.jsonl", {"ts": utc_now_iso(),
                    "type": "chat_block_consolidation", "task_id": _id,
                    "blocks_written": u.get("_blocks_written"),
                    "last_error_kind": (errors[-1] or {}).get("kind") if errors else None,
                    "cost_usd": (
                        round(float(u["cost"]), 6)
                        if u.get("cost") is not None
                        else None
                    )})
                if u.get("cost") or u.get("prompt_tokens"):
                    from supervisor.state import update_budget_from_usage
                    update_budget_from_usage(u)
                return _post_task_paid_interruption(errors)
    except Exception as error:
        propagate_paid_interruption(error)
        log.warning("Chat block consolidation setup failed", exc_info=True)
        return "stage_setup_failed"  # an ordinary failure isolated to this stage, never `completed`


def _run_scratchpad_consolidation(env: Any, memory: Any, llm: Any) -> None:
    """Run scratchpad consolidation inside the root post-task worker."""
    try:
        from ouroboros import consolidator as _c

        should_consolidate = _c.should_consolidate_scratchpad
        consolidate = _c.consolidate_scratchpad
        if should_consolidate(memory):
            kb_dir = env.drive_path("memory/knowledge")
            _identity = memory.load_identity()
            from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

            base_scope = current_usage_scope()
            scratch_scope = (
                replace(base_scope, category="consolidation", source="scratchpad_consolidation")
                if base_scope is not None
                else UsageScope(
                    drive_root=env.drive_root,
                    category="consolidation",
                    source="scratchpad_consolidation",
                )
            )

            with usage_scope(scratch_scope):
                u = consolidate(memory, kb_dir, llm, _identity)
            if u and (u.get("cost") or u.get("prompt_tokens")):
                from supervisor.state import update_budget_from_usage
                update_budget_from_usage(u)
            return _post_task_paid_interruption(u.get("_consolidation_errors") if isinstance(u, dict) else [])
    except Exception as error:
        propagate_paid_interruption(error)
        log.debug("Scratchpad consolidation setup failed", exc_info=True)
        return "stage_setup_failed"


def _run_reflection(env: Any, llm: Any, task: Dict[str, Any],
                    usage: Dict[str, Any], llm_trace: Dict[str, Any],
                    review_evidence: Dict[str, Any],
                    sealed_final: Dict[str, Any] | None = None,
                    publish: Callable[[Dict[str, Any]], Any] | None = None) -> Dict[str, Any] | None:
    """Run execution reflection synchronously (process memory, Bible P1).

    Returns the entry, or None only when there is nothing to reflect on; a
    failure raises to the post-task stage coordinator, which degrades the
    checkpoint and still runs the later stages (TZ-2 C3). ``publish`` receives
    the completed entry before its nested paid Pattern Register write, so an
    interruption there still leaves the coordinator its free memory actions.
    """
    from ouroboros.reflection import (
        should_generate_reflection, generate_reflection, append_reflection_routed,
    )
    synthesis_cost = _synthesis_cost_usd(usage)
    # The one walk happens BEFORE the decision, because a root whose only
    # failures are its children cannot be recognized without it: children do
    # not reflect, so their classes have to reach this gate to be learned
    # from at all. Still one walk, and its rows serve the prompt below.
    child_evidence, child_rows = _child_task_evidence(env, task)
    child_classes = _child_failure_classes(child_rows)
    if should_generate_reflection(
        llm_trace,
        task=task,
        rounds=int(usage.get("rounds", 0)),
        cost_usd=synthesis_cost,
        child_failure_classes=child_classes,
    ):
        trace_summary = build_trace_summary(llm_trace, all_calls=True)
        reflection_usage = dict(usage)
        # Reflection's legacy durable cost_usd field now records this
        # same subtree snapshot instead of silently reverting to own cost.
        reflection_usage["cost"] = synthesis_cost
        from ouroboros.tools.registry import ToolContext
        knowledge_context = ToolContext(
            repo_dir=getattr(env, "repo_dir", env.drive_root),
            drive_root=pathlib.Path(task.get("budget_drive_root") or env.drive_root),
            project_id=str(task.get("project_id") or ""),
            task_id=str(task.get("id") or ""))
        entry = generate_reflection(
            task, llm_trace, trace_summary,
            llm, reflection_usage,
            review_evidence=review_evidence,
            child_evidence=child_evidence,
            usage_snapshot_text=_synthesis_usage_snapshot_text(usage),
            sealed_final_text=sealed_final_prompt_section(sealed_final),
            child_failure_classes=child_classes,
            knowledge_context=knowledge_context,
        )
        entry = {**entry, **presence_provenance_fields(task)}
        if publish is not None:
            publish(entry)
        append_reflection_routed(env, task, entry)
        return entry
    return None
