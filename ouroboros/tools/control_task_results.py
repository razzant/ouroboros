"""Absorbing a child: reading one result, or waiting on a batch of them.

A parent takes a child's work back through two surfaces — the full single-child
read and the compact batch projection — and they must agree about what matters:
the outcome axes, the pinned result hash, the receipts, and any capability the
child did not actually have. The waits add the facts a blocking parent cannot
see for itself: an attention beacon raised mid-flight, siblings still running,
an id this tree never minted, and a prompt cache that expired while it waited.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

from ouroboros.outcomes import normalize_outcome_axes
from ouroboros.runtime_limits import NESTED_SETTLEMENT_MARGIN_SEC
from ouroboros.task_results import (
    STATUS_COMPLETED,
    STATUS_REJECTED_DUPLICATE,
    validate_task_id,
)
from ouroboros.task_status import (
    SETTLED_STATUSES,
    load_effective_task_result,
    wait_for_effective_tasks,
)
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result, completed_local_read
from ouroboros.utils import truncate_review_artifact, utc_now_iso


def disclosable_capability_delta(data: Dict[str, Any]) -> Dict[str, Any]:
    """The child's delta when it has something to SAY, else ``{}`` — ONE predicate.

    THE terminal parent-facing disclosure, and since v6.87.28 the only parent-facing
    one: the reduction is not known until the child is dispatched, so no scheduling
    result can carry it. It is a predicate rather than an inline test because the
    parent uses full `get_task_result` reads and bounded wait projections.
    Both retain the same capability facts, including batched absorption;
    omitting the delta from either surface conceals a child's actual limits.

    A delta that took nothing away and ignored nothing is noise in every payload.
    """
    delta = data.get("capability_delta") if isinstance(data.get("capability_delta"), dict) else {}
    return delta if (delta.get("reduced") or delta.get("legacy_note")) else {}


def disclosable_effort_fact(data: Dict[str, Any]) -> Dict[str, str]:
    """The child's effort decision when it has something to SAY — the parent's request was
    moved into the range or set aside by a pin or a model name (``effort_fact_says``) —
    else ``{}``. The same predicate on both parent surfaces (``[SUBTASK_OUTCOME]`` and the
    compact batch projection); a request that simply applied, or the plain default, is noise."""
    from ouroboros.settings_scales import effort_fact_says

    fact = {"requested": data.get("effort_requested"), "applied": data.get("effort_level"),
            "source": data.get("effort_source")}
    if not effort_fact_says(fact):
        return {}
    return {"level": str(fact["applied"] or ""), "requested": str(fact["requested"] or ""),
            "source": str(fact["source"] or "")}


def _subtask_outcome_summary(data: Dict[str, Any], receipts: list | None = None) -> str:
    ledger = data.get("verification_ledger") if isinstance(data.get("verification_ledger"), dict) else {}
    summary: Dict[str, Any] = {
        "outcome_axes": normalize_outcome_axes(data),
    }
    if isinstance(data.get("execution_observation"), dict):
        summary["execution_observation"] = dict(data["execution_observation"])
    # R5: CURRENT delegated-custody reconciliation state next to the historical
    # axes in the explicit full get_task_result handoff. Wait projections retain
    # their compact custody fields and an exact full-source reference.
    # The frozen delegated_runs_* counters are a historical snapshot (owner
    # Q2=B); the envelope's trigger + open_run_ids are the liveness surface a
    # parent can trust after a refresh/backfill healed the row. One bounded
    # entry: capped lists with an exact omitted count.
    unreconciled = data.get("delegated_runs_unreconciled")
    unreconciled = (
        [str(item) for item in unreconciled] if isinstance(unreconciled, list) else []
    )
    envelope = data.get("delegate_terminal_reconciliation")
    envelope = envelope if isinstance(envelope, dict) else {}
    if unreconciled or envelope:
        custody: Dict[str, Any] = {"unreconciled": unreconciled[:10]}
        omitted_any = False
        if len(unreconciled) > 10:
            custody["unreconciled_omitted"] = len(unreconciled) - 10
            omitted_any = True
        if envelope:
            custody["trigger"] = str(envelope.get("trigger") or "")
            custody["audit_status"] = str(envelope.get("audit_status") or "unknown")
            for field in ("open_run_ids", "pending_invocation_ids", "undisposed_patch_run_ids", "terminal_runs"):
                values = envelope.get(field)
                values = list(values) if isinstance(values, list) else []
                if field == "terminal_runs":  # the stored actor key is a durable join key, never a model-facing name
                    values = [{k: v for k, v in row.items() if k != "selected_subagent_id"}
                              if isinstance(row, dict) else row for row in values]
                custody[field] = values[:10]
                if len(values) > 10:
                    custody[field + "_omitted"] = len(values) - 10
                    omitted_any = True
        if omitted_any:
            # A bound must name a source the actor can resolve (BIBLE P1).
            # Retry lineage unions the ORIGINAL row's disclosure into this
            # projection, so the omitted identifiers may live on either row —
            # name both when the lineage exists.
            tid = str(data.get("task_id") or "")
            origin = str(
                data.get("original_task_id") or data.get("timeout_retry_from") or ""
            )
            paths = [f"task_results/{tid}.json"]
            if origin and origin != tid:
                paths.append(f"task_results/{origin}.json")
            custody["full_source"] = [
                f"read_file(root='runtime_data', path='{p}')" for p in paths
            ]
        summary["delegated_custody"] = custody
    if isinstance(data.get("task_contract"), dict):
        summary["task_contract"] = data.get("task_contract")
    _delta = disclosable_capability_delta(data)
    if _delta:
        summary["capability_delta"] = _delta
    if _effort := disclosable_effort_fact(data):
        summary["effort"] = _effort
    if isinstance(data.get("artifact_bundle"), dict):
        summary["artifact_bundle"] = data.get("artifact_bundle")
    if ledger:
        # An omitted-to-artifact stub carries no entries; its summary is the
        # count authority, and for a full ledger the two always agree.
        ledger_summary = ledger.get("summary") if isinstance(ledger.get("summary"), dict) else {}
        summary["verification_ledger"] = {
            "schema_version": ledger.get("schema_version"),
            "summary": ledger_summary,
            "entry_count": ledger_summary.get("entry_count", len(ledger.get("entries") or []) if isinstance(ledger.get("entries"), list) else 0),
        }
    if receipts:
        # W2: bounded per-receipt rows for the FULL single-child handoff ONLY
        # (get_task_result/wait_task — already uncapped surfaces): which checks
        # passed, not just counts, so a parent can absorb a child on receipt-level
        # green/red instead of prose. The wait_tasks BATCH projection deliberately
        # stays counts-compact (v6.17.0 birth shape + v6.71.2 measured compaction,
        # 694K->25K). Rows render through the SSOT identity projection + disclosed
        # bound (hard cap, exact omitted count).
        #
        # The bound is OUTSTANDING-FIRST, then newest: a plain newest-10 window let
        # a child that failed a check early and then produced ten greens hand the
        # parent an affirmatively all-green list, with the red only implied by a
        # count. The still-unreconciled SET is this repo's SSOT for exactly that
        # problem ("a newer red would let a latest-pointer erase an older still-red
        # one"), so every outstanding red / masked pass is carried first — tagged so
        # the parent sees WHY it is here — and the rest of the cap is filled with the
        # newest remaining receipts. The cap and its exact omitted count are unchanged.
        from ouroboros._outcome_receipts import (
            disclosed_list_projection,
            receipt_identity_projection,
            unreconciled_failed,
            unreconciled_masked,
        )

        rows = [r for r in receipts if isinstance(r, dict)]
        outstanding_kind: Dict[int, str] = {}
        for _receipt in unreconciled_failed(rows):
            outstanding_kind[id(_receipt)] = "unreconciled_failed"
        for _receipt in unreconciled_masked(rows):
            outstanding_kind.setdefault(id(_receipt), "unreconciled_masked_pass")
        ordered = [r for r in reversed(rows) if id(r) in outstanding_kind]
        ordered += [r for r in reversed(rows) if id(r) not in outstanding_kind]

        def _receipt_row(receipt: Any) -> Any:
            if not isinstance(receipt, dict):
                return truncate_review_artifact(str(receipt), limit=200)
            row = {"status": str(receipt.get("status") or "")}
            outstanding = outstanding_kind.get(id(receipt), "")
            if outstanding:
                row["outstanding"] = outstanding
            if "matched" in receipt:
                row["matched"] = receipt.get("matched")
            row.update(receipt_identity_projection(receipt, check_cap=200))
            return row

        summary.update(disclosed_list_projection(
            ordered, key="verification_receipts", limit=10, item=_receipt_row,
        ))
    return json.dumps(summary, ensure_ascii=False, indent=2, default=str)


def _unchanged_result_reference(task_id: str, current_hash: str, known_hash: Any) -> Dict[str, Any]:
    """Omit only an explicitly matched semantic body, never its current facts.

    This is a conditional read, not evidence that the caller still remembers or
    has accepted the result. The source request deliberately carries no condition.
    """
    if not isinstance(known_hash, str) or known_hash != current_hash:
        return {}
    return {
        "result_unchanged": True,
        "result_source": {"tool": "get_task_result", "arguments": {"task_id": task_id}},
    }


def get_task_result_entry() -> ToolEntry:
    """The result/source reader schema lives beside its scoped handler."""
    return ToolEntry("get_task_result", {
        "name": "get_task_result",
        "description": "Read the effective result or exact authority of a task, including one bounded canonical work-order source range when requested.",
        "parameters": {"type": "object", "required": ["task_id"], "properties": {
            "task_id": {"type": "string", "description": "Task ID returned by scheduling or exposed by the host routing manifest."},
            "known_result_sha256": {"type": "string", "description": "Optional child_result_sha256 from a previous read. An exact match omits only unchanged result/trace text, retaining current facts and a full-read reference. Omit for full text; explicit authority/source requests always return their requested view."},
            "include_authority": {"type": "boolean", "default": False, "description": "Return the exact selected result, task contract, origin, artifact references, and current plan-review authority."},
            "include_work_order_source": {"type": "boolean", "default": False, "description": "Return the canonical work-order source projection; provide both source_start_char and source_end_char for the exact bounded range."},
            "include_completion_source": {"type": "boolean", "default": False,
                                          "description": "Read the full stored completion observations for this task, including returns omitted from the summary. Omit bounds for source length/hash, then request explicit character ranges."},
            "include_focus_source": {"type": "boolean", "default": False, "description": "Read the exact bytes this task's focus source_ref answered when the focus was authored (the retained_source of an [INDEPENDENT_ROOTS] row); same bounds contract as include_completion_source."},
            "focus_source_sha256": {"type": "string", "default": "", "description": "With include_focus_source: select the retained source by the sha256 the roster row quoted, so a later focus of the same author cannot substitute its evidence."},
            "review_source_sha256": {"type": "string", "default": "", "description": "Root turns: read only the exact acceptance-review source named by a late-evidence digest, pinned to the physical task_id even after a retry. Works across forked/empty drives. Without a range returns complete_chars/hash; then use source_start_char/source_end_char to read exact text. Does not include authority."},
            "presence_reentry_sha256": {"type": "string", "default": "", "description": "Read the exact Presence reentry observation checkpoint named in a resumed-conversation note. Includes full observed messages and transport facts plus coverage gaps. Uses the existing task scope; omit ranges for complete_chars/hash, then page with source_start_char/source_end_char."},
            "presence_reentry_offset": {"type": "integer", "description": "With presence_reentry_sha256: scan the frozen canonical interval of that checkpoint's own conversation, including rows beyond its initial scan budget. Start at history_start and follow next_offset until interval_exhausted. Small pages return complete rows and explicit gaps. Oversized pages return complete_chars/hash; use source_start_char/source_end_char with the same offset to reconstruct that page's exact JSON before advancing."},
            "source_start_char": {"type": "integer", "description": "Inclusive character offset for the requested canonical source range."},
            "source_end_char": {"type": "integer", "description": "Exclusive character offset for the requested canonical source range. A range outside the source returns no text: the answer names complete_chars and the range received, and is an argument error."},
            "presence_scope": {"type": "string", "enum": ["own_binding"], "description": "Presence tasks only: read just independent work started from this Presence binding (any of its conversations) or this task's own tree."},
            "view_head_chars": {"type": "integer", "minimum": 0, "description": "Chars of this answer's head to show in this turn when the complete answer cannot be delivered whole (pair with view_tail_chars). The complete answer is kept as an exact, readable source either way; this only shapes the first view and never changes source_start_char/source_end_char selection."},
            "view_tail_chars": {"type": "integer", "minimum": 0, "description": "Chars of this answer's tail to show in this turn (see view_head_chars)."},
        }},
    }, _get_task_result)


@completed_local_read
def _get_task_result(
    ctx: ToolContext, task_id: str, include_authority: bool = False,
    include_work_order_source: bool = False, source_start_char: Any = None,
    source_end_char: Any = None, include_completion_source: bool = False,
    known_result_sha256: str = "", include_focus_source: bool = False, focus_source_sha256: str = "",
    presence_scope: str = "", review_source_sha256: str = "",
    presence_reentry_sha256: str = "", presence_reentry_offset: Any = None,
    view_head_chars: Any = None, view_tail_chars: Any = None,
) -> str:
    """Read a task result, or a bounded canonical work-order/completion source range.

    ``view_head_chars``/``view_tail_chars`` are accepted here so the declared
    parameters bind; the loop's delivery honors them when it shapes the first
    view of this answer. They select nothing in the source arithmetic below.
    """
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    status_drive_root = Path(str(metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    scoped = bool(str(presence_scope or "").strip())
    if scoped:
        from ouroboros.presence_authority import PRESENCE_OWN_WORK_SCOPE, presence_caller_binding, presence_work_refusal

        # The scoped read admits exactly the work a scoped page can list, plus this
        # task's own tree; everything else refuses before any record is projected.
        refusal = (
            "⚠️ PRESENCE_CAPABILITY_BLOCKED: presence_scope=own_binding needs a Presence task with a binding id."
            if str(presence_scope).strip() != PRESENCE_OWN_WORK_SCOPE or not presence_caller_binding(ctx)
            else presence_work_refusal(ctx, str(task_id or ""), drive_root=status_drive_root, same_tree=True)
        )
        if refusal:
            return _publish_tool_result(ctx, ToolResult(status="blocked", code="ACCESS_BLOCKED", text=refusal))
    data = load_effective_task_result(status_drive_root, task_id)
    if scoped:
        from ouroboros.presence_authority import presence_effective_refusal

        # The effective read may substitute a retry successor's record: it is judged too.
        refusal = presence_effective_refusal(ctx, str(task_id or ""), data, drive_root=status_drive_root,
                                             same_tree=True)
        if refusal:
            return _publish_tool_result(ctx, ToolResult(status="blocked", code="ACCESS_BLOCKED", text=refusal))
    from ouroboros.tools.recent_tasks import _restricted_actor

    restricted = _restricted_actor(ctx)
    if restricted:
        if bool(include_focus_source) or bool(review_source_sha256):
            # Children and Presence turns hold no cross-focus view (recent_tasks
            # strips focus, live_roots refuses); the retained SOURCE of a focus is
            # part of that view, not of the ordinary task result.
            # The identifier register records TOOL_FORBIDDEN as a typed policy
            # block (the same spelling project_journal and live_roots publish).
            return ("⚠️ TOOL_FORBIDDEN (get_task_result): restricted actors have no cross-focus "
                    "catalogue; focus/review source selectors are not available to them")
        if isinstance(data, dict) and "focus" in data:
            # The same ceiling on every projection of the record: the authority
            # view copies top-level fields, so focus leaves before it is built.
            data = {key: value for key, value in data.items() if key != "focus"}
    if not data:
        return _publish_tool_result(ctx, ToolResult(
            status="unavailable", code="LEGACY_UNAVAILABLE",
            text=f"Task {task_id}: unknown or not yet registered",
        ))
    from ouroboros.project_sources import CLONE_TIMEOUT_SEC
    from ouroboros.routing_wait import is_emitted_admission_stub

    if is_emitted_admission_stub(data):
        # #1160: a promote whose admission the supervisor has not confirmed yet is
        # PENDING, not unknown. Answering "unknown or not yet registered" read as
        # "your promote never happened" and invited the second promote that mints a
        # duplicate root; the id stays reserved and this read is the reconciliation.
        since = str((data.get("promotion_admission") or {}).get("emitted_at") or "")
        return _publish_tool_result(ctx, ToolResult(
            status="unavailable", code="LEGACY_UNAVAILABLE",
            text=(
                f"Task {task_id}: admission pending since {since} (now {utc_now_iso()}) - the promote "
                "was emitted and this row carries no supervisor receipt yet, neither scheduled nor "
                f"refused. Read get_task_result({task_id}) again before promoting the same work: a "
                f"promote that prepares a source can stay pending for up to {CLONE_TIMEOUT_SEC // 60} "
                "minutes. A new promote is a NEW task id; this row does not forbid one, and whether "
                "this one is lost is yours to judge from the two times above."
            ),
        ))
    if presence_reentry_sha256:
        from ouroboros.presence_continuation import reentry_source_projection

        source = reentry_source_projection(status_drive_root, str(task_id), presence_reentry_sha256,
                                            source_start_char, source_end_char, presence_reentry_offset)
        text = json.dumps({"presence_reentry_source": source}, ensure_ascii=False, sort_keys=True)
        if source.get("reason") == "source_range_invalid":
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=text))
        return text
    if review_source_sha256:
        from ouroboros.task_finalization import review_source_projection

        source = review_source_projection(status_drive_root, str(task_id), review_source_sha256,
                                          source_start_char, source_end_char)
        text = json.dumps({"review_source": source}, ensure_ascii=False, sort_keys=True)
        if source.get("reason") == "source_range_invalid":
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=text))
        return text
    if bool(include_authority) or bool(include_work_order_source) or bool(include_completion_source) or bool(include_focus_source):
        from ouroboros.agent_startup_checks import task_result_authority_projection

        authority = task_result_authority_projection(data, drive_root=status_drive_root)
        payload: Dict[str, Any] = {
            "status": "available", "authority": authority,
            "source": {"tool": "get_task_result", "task_id": str(task_id)},
        }
        payload["outcome"] = json.loads(_subtask_outcome_summary(data, receipts=_merged_task_receipts(status_drive_root, task_id, data)))
        if bool(include_work_order_source):
            from ouroboros.subagent_work_order import (
                _source_task_from_context,
                work_order_source_projection,
            )

            projection, reason = work_order_source_projection(
                _source_task_from_context(ctx, str(task_id)),
                source_start_char,
                source_end_char,
            )
            source = {
                "kind": "task_result",
                "task_id": str(task_id),
                "tool": "get_task_result",
                "arguments": {
                    "task_id": str(task_id),
                    "include_authority": True,
                    "include_work_order_source": True,
                },
                "projection": "canonical_work_order",
            }
            payload["source"] = source
            payload["work_order_source"] = projection or {
                "schema": 1, "kind": "canonical_work_order", "status": "unavailable",
            }
            if reason:
                payload["work_order_source"]["reason"] = reason
        if bool(include_completion_source):
            from ouroboros.task_finalization import completion_source_projection

            payload["completion_source"] = completion_source_projection(
                status_drive_root, str(task_id), data, source_start_char, source_end_char,
            )
        if bool(include_focus_source):
            from ouroboros.task_finalization import focus_source_projection
            from ouroboros.task_results import load_task_result

            # The PHYSICAL author's record: a retry supersedes the effective
            # result, but the roster names the task that retained the bytes.
            physical = load_task_result(status_drive_root, str(task_id))
            payload["focus_source"] = focus_source_projection(
                status_drive_root, str(task_id), physical if isinstance(physical, dict) else {},
                source_start_char, source_end_char, sha256=str(focus_source_sha256 or ""),
            )
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if any(isinstance(view, dict) and view.get("reason") == "source_range_invalid"
               for view in (payload.get("work_order_source"), payload.get("completion_source"), payload.get("focus_source"))):
            # The requested text was NOT returned: same JSON (it names complete_chars and
            # the range received), recorded as the argument fault it is, never as `ok`.
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=text))
        return text
    status = data.get("status", "unknown")
    result = data.get("result", "")
    trace = data.get("trace_summary", "")
    if str(status or "").lower() not in SETTLED_STATUSES:
        projection = _compact_child_projection(str(task_id), data, known_result_sha256)
        projection["delegated_runs"] = _delegated_run_facts(status_drive_root, str(task_id))
        return bounded_wait_response(ctx, {"tasks": {str(task_id): projection}, "nonterminal": True})
    receipts = _merged_task_receipts(status_drive_root, task_id, data)
    outcome_summary = _subtask_outcome_summary(data, receipts=receipts)
    debt = data.get("acceptance_debt")
    if isinstance(debt, dict) and debt.get("source_ref"):
        from ouroboros.acceptance_history import historical_source_reference

        summary = json.loads(outcome_summary)
        summary["acceptance_debt"] = {key: debt.get(key) for key in (
            "debt_id", "task_id", "accounting_root_task_id", "delivery_status")}
        summary["acceptance_debt"]["source_ref"] = historical_source_reference(
            status_drive_root, str(data.get("task_id") or task_id), debt["source_ref"], subject=True)
        outcome_summary = json.dumps(summary, ensure_ascii=False, indent=2)
    from ouroboros.tools.join_ledger import _child_result_sha256

    child_result_sha256 = _child_result_sha256(data)
    # SSOT cost projection (C2): unknown never renders as $0.00 (and a null in
    # the stored result no longer crashes the f-string with a TypeError).
    from ouroboros.cost_projection import cost_display

    unchanged = _unchanged_result_reference(str(task_id), child_result_sha256, known_result_sha256)
    if unchanged:
        # Accounting, receipts, authority and capability facts are deliberately
        # outside the join-ledger result identity; keep their current projection.
        if data.get("duplicate_of"):
            unchanged["duplicate_of"] = str(data["duplicate_of"])
        output = (
            f"Task {task_id} [{status}]: cost={cost_display(data)}\n"
            f"child_result_sha256={child_result_sha256}\n\n"
            f"[SUBTASK_OUTCOME]\n{outcome_summary}\n[/SUBTASK_OUTCOME]\n\n"
            f"{json.dumps(unchanged, ensure_ascii=False)}\n"
            "Result and trace are unchanged; omit known_result_sha256 to read them in full."
        )
    elif status == STATUS_COMPLETED:
        output = (
            f"Task {task_id} [{status}]: cost={cost_display(data)}\n"
            f"child_result_sha256={child_result_sha256}\n\n"
            f"[SUBTASK_OUTCOME]\n{outcome_summary}\n[/SUBTASK_OUTCOME]\n\n"
            f"[BEGIN_SUBTASK_OUTPUT]\n{result}\n[END_SUBTASK_OUTPUT]"
        )
    elif status == STATUS_REJECTED_DUPLICATE:
        duplicate_of = str(data.get("duplicate_of") or "?")
        output = (
            f"Task {task_id} [{status}]: duplicate_of={duplicate_of}\n"
            f"child_result_sha256={child_result_sha256}\n\n"
            f"[SUBTASK_OUTCOME]\n{outcome_summary}\n[/SUBTASK_OUTCOME]\n\n"
            f"{result or f'Task was rejected as a duplicate of {duplicate_of}.'}"
        )
    else:
        output = (
            f"Task {task_id} [{status}]\n"
            f"child_result_sha256={child_result_sha256}\n\n"
            f"[SUBTASK_OUTCOME]\n{outcome_summary}\n[/SUBTASK_OUTCOME]\n\n"
            f"{result or 'No details available.'}"
        )
    if isinstance(data.get("cancel_origin"), dict):
        output += f"\n\n[CANCELLED_BY] {json.dumps(data['cancel_origin'], ensure_ascii=False)}"
    if data.get("retired_tool_invocations"):
        output += ("\n\n[INTERRUPTED_TOOL_CALLS] Local execution ended; external effects remain unknown.\n"
                   + json.dumps(data["retired_tool_invocations"], ensure_ascii=False))
    from ouroboros.task_custody import unread_mail_notice

    if unread := unread_mail_notice(data.get("unread_mailbox")):
        output += f"\n\n{unread}"  # TZ-1 V10: mail written to this task that its model never read
    if trace and not unchanged:
        output += f"\n\n[SUBTASK_TRACE]\n{trace}\n[/SUBTASK_TRACE]"
    from ouroboros.task_finalization import provider_terminal_body, terminal_host_notice_text

    return provider_terminal_body(output, terminal_host_notice_text(data))


def _wait_attention_poll(
    ctx: ToolContext, after_ts: str, task_ids: List[str], *, consume: bool = True,
) -> Callable[..., Any]:
    """on_poll hook: break a sliced wait for the actor's mailbox or a child attention beacon
    (blocker/question/interface_contract/review_requested/delegation_constraint).

    The cursor is per child and retained by owner_wait.continuation_state: a beacon written before this
    particular tool call is still delivered, while a later wait in the same
    actor's warm or restored cold context does not replay it. Equal-timestamp rows use their stable
    content identity, so the five-row response bound cannot strand the rest.
    """
    # tree_note/tree_read live in ouroboros/tools/task_tree.py (extracted for module size).
    from ouroboros.owner_mailbox import OwnerMailboxPeek
    from ouroboros.tools.task_tree import tree_root_id

    rid = tree_root_id(ctx)
    mailbox_peek = OwnerMailboxPeek()

    cursor_store = getattr(ctx, "_wait_attention_cursors", None)
    if not consume:
        import copy
        cursor_store = copy.deepcopy(cursor_store) if isinstance(cursor_store, dict) else {}
    if not isinstance(cursor_store, dict):
        cursor_store = {}
        try:
            setattr(ctx, "_wait_attention_cursors", cursor_store)
        except Exception:
            # An exotic immutable context still gets correct delivery within
            # this hook instance; ordinary ToolContext objects retain it across
            # subsequent wait_task/wait_tasks calls.
            pass

    child_cursors: Dict[str, Dict[str, Any]] = {}
    for task_id in task_ids:
        key = f"{rid}:{task_id}"
        cursor = cursor_store.get(key)
        if not isinstance(cursor, dict):
            cursor = {"after_ts": str(after_ts or ""), "seen_ids": set()}
            cursor_store[key] = cursor
        if not isinstance(cursor.get("seen_ids"), set):
            cursor["seen_ids"] = {
                str(item) for item in (cursor.get("seen_ids") or []) if str(item)
            }
        child_cursors[str(task_id)] = cursor

    def _hook(_results: Dict[str, Any], _terminal: Dict[str, bool]) -> Any:
        # Reuse transport-wait's non-destructive peek. The ordinary round-top
        # drain still owns delivery and acknowledgement; child work stays live.
        if getattr(ctx, "drive_root", None) and getattr(ctx, "task_id", None) and mailbox_peek.pending(
            Path(ctx.drive_root), str(ctx.task_id),
            set(getattr(ctx, "_loop_mailbox_seen_ids", None) or ()),
            getattr(ctx, "task_attempt", None) or 1, actionable_only=True,
        ):
            return {"reason": "owner_mailbox_pending", "delivery": "pending_loop_drain"}
        if not rid:
            return None
        try:
            from ouroboros.task_tree_ledger import (
                tree_ledger_attention_after,
                tree_ledger_row_id,
            )

            attention = tree_ledger_attention_after(rid, "", task_ids=set(task_ids), data_root=_status_root(ctx))
        except Exception:
            return None
        pending: List[tuple[Dict[str, Any], str]] = []
        for row in attention:
            task_id = str(row.get("task_id") or "")
            cursor = child_cursors.get(task_id)
            if cursor is None:
                continue
            ts = str(row.get("ts") or "")
            cursor_ts = str(cursor.get("after_ts") or "")
            row_id = tree_ledger_row_id(row)
            if ts < cursor_ts:
                continue
            if ts == cursor_ts and row_id in cursor["seen_ids"]:
                continue
            pending.append((row, row_id))
        if not pending:
            return None

        delivered = pending[:5]
        for row, row_id in delivered:
            cursor = child_cursors[str(row.get("task_id") or "")]
            ts = str(row.get("ts") or "")
            cursor_ts = str(cursor.get("after_ts") or "")
            if ts > cursor_ts:
                cursor["after_ts"] = ts
                cursor["seen_ids"] = set()
            cursor["seen_ids"].add(row_id)
        return {
            "reason": "child_attention_beacon",
            "beacons": [row for row, _row_id in delivered],
            "beacons_remaining": len(pending) - len(delivered),
        }

    return _hook


def cache_horizon_note(ctx: Any, elapsed_sec: Any) -> str:
    """One factual line when a blocking wait outlived the APPLIED prompt-cache TTL.

    Reads the RECORDED fact of this task's latest send — ``_last_prompt_cache_ttl``
    in the loop's accumulated usage (published on the tool ctx), converted by
    ``llm.cache_ttl_seconds`` — never a route-level prediction (a second predictor
    can disagree with the payload after route-filter/promotion/cap). Empty string
    when the horizon is unknown or not yet elapsed. UNKNOWN covers three cases,
    all silent: no cached send recorded, a route that carries no markers at all,
    and a send whose markers were BARE (reported ``"default"``) — a bare marker
    names no tier, so its horizon is the provider's business and inventing one
    would mislead the agent into re-planning its waits around a number nobody
    established. Only the explicitly stamped ``5m``/``1h`` tiers speak here.
    Deliberately NO token-count predictions: the submarine forensics showed the
    fact ("the wait outlived the cache") is what changes the agent's next decision
    (batch waits, longer single windows), while "~X tokens will re-write" is a
    counterfactual — the next send may reroute, compact, or still hit a live cache.

    REACHABILITY: root waits feed their elapsed interval, a LOWER bound on
    cache age. Default warm sleep has no periodic timer and can cross either
    applied tier; explicit waits retain their separate 3600/7200s ceilings. ``delegate_wait``
    feeds the time since the task's last recorded model response, once per wake, so
    its line is reachable at ANY tier and no longer depends on the 3 s tick. Pinned by
    tests/test_cache_optimization.py::test_cache_horizon_reachability_matches_the_wait_clamps
    and the supervising-wake tests; the tier is an owner setting, not a constant, and
    a wait tool that silently could not disclose would be worse.
    """
    try:
        elapsed = float(elapsed_sec)
    except (TypeError, ValueError):
        return ""
    usage = getattr(ctx, "_accumulated_usage", None)
    if not isinstance(usage, dict):
        return ""
    applied_ttl = str(usage.get("_last_prompt_cache_ttl") or "").strip()
    from ouroboros.llm import cache_ttl_seconds

    horizon = cache_ttl_seconds(applied_ttl)
    if horizon is None or elapsed <= horizon:
        return ""
    return (
        f"⚠️ configured prompt-cache horizon ({applied_ttl}, {horizon}s) elapsed since the "
        f"last model response ({elapsed:.0f}s ago); the next model send may be cold."
    )


def _merged_task_receipts(status_drive_root, task_id, data):
    """Full explicit readers join canonical and not-yet-copied execution receipts."""
    from ouroboros.outcomes import read_verification_receipts_from_roots
    from ouroboros.task_status import _child_drive_candidates
    try:
        return read_verification_receipts_from_roots([*_child_drive_candidates(data), status_drive_root], task_id)
    except Exception:
        return []


def _wait_early_return_note(early) -> str:
    if not early:
        return ""
    if early.get("reason") == "owner_mailbox_pending":
        return "The ordinary loop will deliver and acknowledge the unread message. This does not stop the child."
    return "A child attention beacon interrupted this wait. Inspect early_return; the children keep running."


def _status_root(ctx: Any) -> Path:
    metadata = getattr(ctx, "task_metadata", {}) or {}
    return Path(metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", None) or ctx.drive_root)


def _event_wait_window(ctx: Any) -> int:
    """One finite transport envelope, not a periodic cognitive wake."""
    from ouroboros.config import get_task_abs_ceiling_sec
    from ouroboros.runtime_limits import operation_window_sec

    return max(0, int(operation_window_sec(get_task_abs_ceiling_sec())) - NESTED_SETTLEMENT_MARGIN_SEC)


WAIT_RESPONSE_CHARS = 15_000


def bounded_wait_response(ctx: Any, payload: Dict[str, Any], *, limit: int = WAIT_RESPONSE_CHARS) -> str:
    """Budget the WHOLE JSON response; retain exact bytes before shortening it.

    A preview is not a semantic summary or an ACK. If identities/metadata alone
    cannot fit, publish a source index, never a sliced JSON or apparent PASS.
    """
    rendered = json.dumps(payload, ensure_ascii=False, default=str)
    if len(rendered) <= limit:
        return rendered
    from ouroboros.artifacts import store_actor_source_bytes

    try:
        if not getattr(ctx, "task_id", None):
            raise ValueError("wait reader has no task source owner")
        source = store_actor_source_bytes(ctx.drive_root, str(ctx.task_id), category="tool_results",
            source_id="wait-handoff", data=rendered.encode("utf-8"), extension="json")
    except (OSError, ValueError) as exc:
        source = {"status": "unavailable", "reason": type(exc).__name__,
                  "full_read": {"tool": "get_task_result", "note": "explicitly read the named tasks in full"}}
    view = json.loads(rendered)
    view["complete_source"] = source
    view["complete_chars"] = len(rendered)
    view["preview_only"] = True
    rows = view.get("tasks") or {}
    if isinstance(rows, dict):
        budget = min(3000, max(0, 8000 // max(1, len(rows))))
        for row in rows.values():
            if not isinstance(row, dict):
                continue
            for key in ("result", "trace_summary"):
                text = row.get(key)
                if isinstance(text, str) and len(text) > budget:
                    row[key] = text[:budget]
                    row[key + "_omitted_chars"] = len(text) - budget
    preview = json.dumps(view, ensure_ascii=False)
    if len(preview) <= limit:
        return preview
    # Metadata can itself be arbitrarily large. Its exact meaning stays in the
    # retained source, not in a "compact" object with silently dropped warnings.
    selected = dict(list(rows.items())[:20]) if isinstance(rows, dict) else {}
    summary = {}
    for tid, row in selected.items():
        summary[tid] = {key: row[key] for key in ("task_id", "status", "child_result_sha256",
            "accounted_upper_bound_usd", "cost_final", "cancel_state", "result_unchanged") if key in row}
        notice = row.get("terminal_host_notice")
        if notice:
            summary[tid]["terminal_host_notice_preview"] = str(notice)[:160]
            summary[tid]["terminal_host_notice_chars"] = len(str(notice))
    return json.dumps({"preview_only": True, "complete_chars": len(rendered),
        "complete_source": source, "reason": "wait_metadata_requires_source_read",
        "all_terminal": payload.get("all_terminal"), "timed_out": payload.get("timed_out"),
        "tasks": summary, "tasks_omitted": len(rows) - len(selected),
        "task_count": len(rows)}, ensure_ascii=False)


def wait_mail_preview(ctx: Any) -> Dict[str, Any]:
    """Non-destructive informational exhibit; the next loop drain owns full delivery."""
    from ouroboros.owner_mailbox import drain_owner_entries, wait_message_requires_attention

    if not getattr(ctx, "task_id", None):
        return {}
    entries = drain_owner_entries(Path(ctx.drive_root), str(ctx.task_id),
        set(getattr(ctx, "_loop_mailbox_seen_ids", ()) or ()), getattr(ctx, "task_attempt", None) or 1)
    quiet = [entry for entry in entries if not wait_message_requires_attention(entry)]
    return {"informational_mail": [{"msg_id": entry["msg_id"], "source_task_id": entry.get("source_task_id"),
        "provenance": entry.get("provenance"),
        "complete_chars": len(entry["text"])} for entry in quiet[:20]],
        "mail_omitted": max(0, len(quiet) - 20), "mail_acknowledged": False,
        "mail_delivery": "full round-top drain in the same resumed request; checkpoint precedes ACK"}


def _wait_for_task(
    ctx: ToolContext, task_id: str, timeout_sec: int | None = None, known_result_sha256: str = "",
) -> str:
    """Named child plus an entry snapshot of this parent's other live children."""
    from ouroboros.task_status import find_child_tasks

    try:
        tid = validate_task_id(task_id)
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
            text=f"⚠️ TOOL_ARG_ERROR (wait_task): {exc}"))
    ids = [tid]
    # Do not expand waits on other roots (a plan/peer wait is not child absorption).
    named = load_effective_task_result(_status_root(ctx), tid, materialize_artifacts=False)
    parent_id = str(getattr(ctx, "task_id", "") or "")
    children = find_child_tasks(
        _status_root(ctx), parent_task_id=parent_id, scope="direct", materialize_artifacts=False,
    ) if parent_id else []
    if parent_id and (named.get("parent_task_id") == parent_id
                      or any(str(row.get("task_id") or "") == tid for row in children)):
        ids += [str(row["task_id"]) for row in children
                if row.get("task_id") and str(row.get("status") or "") not in SETTLED_STATUSES
                and str(row["task_id"]) != tid]
    return _wait_for_tasks(ctx, ids, timeout_sec=timeout_sec, mode="any_terminal",
                           known_result_sha256_by_task={tid: known_result_sha256} if known_result_sha256 else None,
                           _explicit_clamp=_WAIT_TASK_CLAMP_SEC)


def _age_sec(stamp: Any, now: float) -> Any:
    from datetime import datetime

    try:
        return round(now - datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp(), 1)
    except (TypeError, ValueError):
        return None


def _delegated_run_facts(drive_root: Path, child_id: str) -> Any:
    """The child's own open delegated runs, as DATED facts read once per result.

    Rows come from delegate custody (``open_runs`` + ``run_timing``; review-owned
    runs belong to their panel, as in ``prepare_handoff``). The child's supervision
    record (``state/delegate_supervision/<child>.json``) is read ONCE, after the
    wait, with no handle kept and no retry loop: a missing, unreadable or racing
    file (Windows ``os.replace``) is the fact ``unknown``. Its facts attach only to
    the run it names. Ages are facts; there is no threshold and no liveness verdict.
    """
    from ouroboros import delegate_custody as custody

    try:
        runs = [run for run in custody.open_runs(drive_root)
                if run.task_id == child_id and not run.review_owned]
    except Exception:
        return {"state": "unknown", "reason": "custody_unreadable"}
    if not runs:
        return []
    record: Any = None
    try:
        path = Path(drive_root) / "state" / "delegate_supervision" / f"{child_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        reason = "" if isinstance(record, dict) else "unreadable"
    except FileNotFoundError:
        reason = "no_supervision_record"
    except (OSError, ValueError):
        reason = "unreadable"
    now = time.time()
    rows: List[Dict[str, Any]] = []
    for run in runs:
        started_at, max_seconds = custody.run_timing(drive_root, run.run_id)
        row: Dict[str, Any] = {"run_id": run.run_id, "started_at": started_at or "unknown",
                               "age_sec": _age_sec(started_at, now), "max_seconds": max_seconds or None}
        if reason:
            row["supervision"] = {"state": "unknown", "reason": reason}
        elif str(record.get("run_id") or "") != run.run_id:
            row["supervision"] = {"state": "unknown", "reason": "record_names_another_run"}
        else:
            observation = record.get("observation") if isinstance(record.get("observation"), dict) else None
            entry = record.get("sleep_entry") if isinstance(record.get("sleep_entry"), dict) else {}
            last = str(record.get("last_answered_observation_at") or "")
            row["supervision"] = {
                "hold_entered_at": str(entry.get("entered_at") or "unknown"),
                "last_answered_observation_at": last or "unknown",
                "observation_age_sec": _age_sec(last, now) if last else None,
                "observed_run_state": str((observation or {}).get("run_state") or "unknown"),
                "observation_failure": (
                    "unknown" if observation is None else "" if observation.get("answered")
                    else str(observation.get("reason") or observation.get("status") or "unanswered")),
                "journal_cursor": record.get("journal_cursor"),
            }
        rows.append(row)
    return rows


_AWAIT_MESSAGES_POLL_SEC = 2.0
_AWAIT_MESSAGES_NOTE = (
    "Any pending message is delivered at the next round top, not by this tool; this wait "
    "holds the worker slot and releases nothing."
)


def _wait_window(
    ctx: Any, requested: int, *, clamp: int, minimum: int, margin: int,
) -> tuple[int, str]:
    """The wait window in seconds and the bound that set it — ONE ladder for
    ``wait_task``, ``wait_tasks`` and ``await_messages``.

    ``requested`` is clamped to [``minimum``, ``clamp``]. Under a task deadline the
    window also stops ``margin`` seconds short of the emit window (remaining minus
    the finalization reserve, subtracted ONCE) that ``_deadline_clamped_timeout``
    gives the executor's kill timer, so the tool returns and builds its result
    instead of being timed out mid-sleep; a spent emit window is a zero-second
    window (one peek, then return). Without a deadline the kill timer is at least
    the tool's entry timeout, which each caller keeps at ``clamp + margin`` or
    more. Inside the finalization reserve the executor's 1 s floor stays the
    bounded exit.

    The task waits pass minimum 0 and ``NESTED_SETTLEMENT_MARGIN_SEC``, so they end
    that far inside the executor's emission; ``await_messages`` passes minimum 1 and
    margin 1, and its last poll sleep is clipped to the remaining window.
    """
    from ouroboros.deadline_utils import deadline_remaining_sec, has_deadline
    from ouroboros.task_pacing import effective_finalization_reserve_sec

    if requested < minimum:
        window, bound = minimum, "minimum"
    elif requested > clamp:
        window, bound = clamp, "ceiling"
    else:
        window, bound = requested, "requested"
    if has_deadline(ctx):
        emit_window = deadline_remaining_sec(ctx) - effective_finalization_reserve_sec(ctx)
        deadline_window = max(0, int(emit_window) - margin)
        if deadline_window < window:
            window, bound = deadline_window, "deadline"
    return window, bound


def _await_messages(ctx: ToolContext, timeout_sec: int | None = None, mode: str = "warm", senders: Any = None,
                    tasks: Any = None, runs: Any = None, wake_at: Any = None, wake_after_sec: Any = None,
                    services: Any = None) -> str:
    """Default warm event sleep selects reply sources or live direct children;
    owner/control and typed attention always wake it. Explicit in_slot holds
    capacity for its bounded window. Neither mode delivers or acknowledges
    mail: the resumed round drains originals, then checkpoint persistence ACKs.

    Idle rail, honestly: the supervisor stamps ``last_progress_at`` on completed
    model rounds, on narration and, when a tool's typed lease closes, on the
    completed tool call (``supervisor/cognitive_operations``); the per-call
    ceiling alone would not keep a round that already spent long in earlier tools
    under the rail ``max(idle, ceiling + 120)``. What spares a task while this
    tool runs is the executor's typed cognitive-operation lease that EVERY tool
    call holds while physically in flight (``loop_tool_execution._emit_live_log``,
    kind ``tool``, bounded by the absolute ceiling and the deadline), and the
    close of that lease is the progress stamp the next model round starts from —
    a full idle window even after a wait that spent the whole ceiling, so the
    supervisor tick between the finished wait and the next model call never
    reaps the turn the wait was for. The in-slot path emits no separate lease
    and lends no capacity; warm mode uses the existing owner-wait capacity
    transfer. tests/test_await_messages.py exercises the in-slot lifecycle.
    """
    from ouroboros.owner_mailbox import OwnerMailboxPeek

    if mode not in ("in_slot", "warm", "cold"):
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
            text="⚠️ TOOL_ARG_ERROR (await_messages): mode must be in_slot, warm or cold."))
    if timeout_sec is not None:
        try:
            timeout_sec = int(timeout_sec)
        except (TypeError, ValueError):
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
                text="⚠️ TOOL_ARG_ERROR (await_messages): timeout_sec must be an integer."))
    if timeout_sec == 0 or (mode != "in_slot" and timeout_sec is not None and timeout_sec < 0):
        from ouroboros import model_sleep
        try:
            chosen = model_sleep.selectors(ctx, senders=senders, tasks=tasks, runs=runs,
                services=services, wake_at=wake_at, wake_after_sec=wake_after_sec, allow_empty=True)
            ready = model_sleep.wake_reason(ctx, chosen, consume_beacons=False)
        except (TypeError, ValueError) as exc:
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
                text=f"⚠️ TOOL_ARG_ERROR (await_messages): {exc}"))
        return bounded_wait_response(ctx, {"reason": "snapshot", "slept": False, "mode": mode,
            "ready": bool(ready), "woke_by": ready, "wake_beacons": chosen.get("wake_beacons"),
            "tasks": {tid: _compact_child_projection(tid, load_effective_task_result(
                _status_root(ctx), tid, materialize_artifacts=False), None) for tid in chosen["tasks"]},
            **wait_mail_preview(ctx)})
    if mode != "in_slot" or senders or tasks or runs or services or wake_at or wake_after_sec:
        if timeout_sec is not None:
            try:
                timeout_sec = int(timeout_sec)
            except (TypeError, ValueError):
                return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
                    text="⚠️ TOOL_ARG_ERROR (await_messages): timeout_sec must be an integer."))
            if wake_at or wake_after_sec:
                return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
                    text="⚠️ TOOL_ARG_ERROR (await_messages): give timeout_sec or a wake time, not both."))
            if timeout_sec <= 0:
                return json.dumps({"reason": "snapshot", "slept": False, "mode": mode})
            wake_after_sec = timeout_sec
        return _await_as_sleep(ctx, mode, senders=senders, tasks=tasks, runs=runs, services=services,
                               wake_at=wake_at, wake_after_sec=wake_after_sec)
    try:
        requested = int(timeout_sec) if timeout_sec is not None else _event_wait_window(ctx)
    except (TypeError, ValueError):
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR",
            text="⚠️ TOOL_ARG_ERROR (await_messages): timeout_sec must be an integer number of seconds.",
        ))
    from ouroboros.config import get_per_call_timeout_ceiling_sec

    window, bound = _wait_window(ctx, requested, clamp=int(get_per_call_timeout_ceiling_sec()),
                                 minimum=1, margin=1)
    peek = OwnerMailboxPeek()
    drive_root = getattr(ctx, "drive_root", None)
    task_id = str(getattr(ctx, "task_id", "") or "")
    seen = getattr(ctx, "_loop_mailbox_seen_ids", None)
    attempt = getattr(ctx, "task_attempt", None) or 1
    start = time.monotonic()
    while True:
        # The transport wait's non-destructive peek over a COPY of the seen
        # set; the mailbox and its acknowledgements are untouched.
        pending = bool(peek.pending(Path(drive_root), task_id, set(seen or ()), attempt, actionable_only=True))
        elapsed = time.monotonic() - start
        if pending or elapsed >= window:
            break
        time.sleep(min(_AWAIT_MESSAGES_POLL_SEC, max(0.0, window - elapsed)))
    if pending:
        reason = "owner_mailbox_pending"
    else:
        reason = "deadline" if bound == "deadline" else "timeout"
    out: Dict[str, Any] = {
        "reason": reason,
        "pending": pending,
        "elapsed_sec": round(float(elapsed), 3),
        "requested_sec": requested,
        "window_sec": window,
        "window_bound": bound,
        "slot": "held",
        "note": _AWAIT_MESSAGES_NOTE,
    }
    horizon_note = cache_horizon_note(ctx, elapsed)
    if horizon_note:
        out["cache_horizon"] = horizon_note
    return json.dumps(out, ensure_ascii=False)


def _await_as_sleep(ctx: ToolContext, mode: str, **chosen: Any) -> str:
    """The model's own warm/cold SLEEP (``ouroboros/model_sleep.py``): validate the
    selected sources, answer at once when one is already ready, else arm the park
    that follows this tool batch."""
    from ouroboros import model_sleep

    try:
        if mode not in (model_sleep.MODE_WARM, model_sleep.MODE_COLD):
            raise ValueError("senders/tasks/runs/services/wake_at/wake_after_sec select a sleep: give mode warm or cold")
        if not callable(getattr(ctx, "owner_wait_callback", None)):
            raise ValueError("this task has no continuation owner to sleep under")
        outcome = model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, **chosen), mode)
    except (TypeError, ValueError) as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR", text=f"⚠️ TOOL_ARG_ERROR (await_messages): {exc}",
            meta={"operation_outcome": "completed_no_effect"}))
    return json.dumps(outcome, ensure_ascii=False)


def await_messages_entry() -> ToolEntry:
    """The await_messages catalog entry, owned beside its handler; the kill timeout
    sits 60s above the largest window the tool can choose (the per-call ceiling)."""
    from ouroboros.config import get_per_call_timeout_ceiling_sec

    return ToolEntry("await_messages", {
        "name": "await_messages",
        "description": (
            "Sleep once until selected sender mail, task/run terminals, your service exit, a chosen time or "
            "actionable input. Default warm keeps your stack/browser and lends pooled capacity; with no "
            "selectors it snapshots your live direct children. Owner/control and typed escalations always "
            "wake; informational mail waits for full delivery in the same resumed request. Explicit "
            "in_slot holds capacity; cold requires settled writers. A missing continuation owner refuses "
            "warm sleep. Terminal means settled, not success; compact handoffs keep full sources."
        ),
        "parameters": {"type": "object", "properties": {
            "timeout_sec": {"type": "integer", "description":
                            "Explicit short wait in seconds, or 0 for a snapshot; omitted warm sleep has no periodic timer."},
            "mode": {"type": "string", "enum": ["in_slot", "warm", "cold"],
                     "default": "warm", "description": "warm (default) lends pooled capacity and keeps the stack; in_slot holds capacity; cold ends the process."},
            "senders": {"type": "array", "items": {"type": "string"},
                        "description": "Sleep: task ids whose mail wakes you (others' mail stays unread)."},
            "tasks": {"type": "array", "items": {"type": "string"},
                      "description": "Sleep: task ids whose terminal (any settled status) wakes you."},
            "runs": {"type": "array", "items": {"type": "string"},
                     "description": "Sleep: your delegated run ids whose terminal wakes you."},
            "services": {"type": "array", "items": {"type": "string"},
                         "description": "Warm sleep: names of your own running services whose exit wakes you."},
            "wake_at": {"type": "string", "description": "Sleep: an absolute ISO-8601 wake time (with timezone)."},
            "wake_after_sec": {"type": "integer", "description": "Sleep: wake after this many seconds."},
        }},
    }, _await_messages, timeout_sec=get_per_call_timeout_ceiling_sec() + 60)


def _count_live_sibling_children(ctx: ToolContext, status_drive_root: Path, *, exclude_task_id: str) -> int:
    """Count this parent's children still running/scheduled/requested (excluding the one
    just waited on). Advisory only — a failure returns 0 so it never breaks wait_task."""
    parent_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not parent_id:
        return 0
    try:
        from ouroboros.task_results import (
            STATUS_REQUESTED,
            STATUS_RUNNING,
            STATUS_SCHEDULED,
            list_task_results,
        )

        live = 0
        for item in list_task_results(status_drive_root, statuses=[STATUS_RUNNING, STATUS_SCHEDULED, STATUS_REQUESTED]):
            if str(item.get("task_id") or item.get("id") or "") == exclude_task_id:
                continue
            if str(item.get("parent_task_id") or "") == parent_id:
                live += 1
        return live
    except Exception:
        return 0


_UNMINTED_WAIT_GRACE_SEC = 30.0
# The wait ceilings, read by the tool schemas, the expiry disclosure and the one
# ``_wait_window`` ladder. Each ToolEntry kill timeout stays at least ceiling +
# NESTED_SETTLEMENT_MARGIN_SEC, so a full window ends inside its executor bound.
_WAIT_TASK_CLAMP_SEC = 3600
_WAIT_TASKS_CLAMP_SEC = 7200


def _unminted_wait_ids(ctx: ToolContext, status_drive_root: Path, task_ids: List[str],
                       require_waitable: bool = False) -> List[str]:
    """Ids with no trace on ANY surface this tree mints ids through: no task
    result, no queue-snapshot row, and no tree-ledger row naming them (v6.91).

    wave2's root blocked 900s slices on three hallucinated ids that wait_tasks
    silently polled as 'unknown' — while the real lead was missing from the wait
    set. The typed marker (plus the actual children roster) lets the parent
    repair its wait set instead of starving on phantoms. Fail-soft per probe: an
    unreadable surface treats the id as KNOWN — a real child must never be
    branded unknown on an I/O error. Event waits require a result/queue source:
    a ledger mention alone proves an ID was minted, not a wakeable task."""
    from ouroboros.task_status import _load_queue_snapshot, _queue_task_status

    try:
        snapshot = _load_queue_snapshot(status_drive_root)
    except Exception:
        snapshot = {"_snapshot_invalid": True}
    ledger_ids: set = set()
    try:
        from ouroboros.task_tree_ledger import tree_ledger_rows
        from ouroboros.tools.task_tree import tree_root_id

        for row in tree_ledger_rows(tree_root_id(ctx), data_root=status_drive_root):
            for key in ("task_id", "child_task_id", "parent_task_id"):
                value = str(row.get(key) or "").strip()
                if value:
                    ledger_ids.add(value)
    except Exception:
        pass
    unknown: List[str] = []
    for tid in task_ids:
        try:
            if load_effective_task_result(status_drive_root, tid):
                continue
            queue_status, _ = _queue_task_status(snapshot, tid)
            if queue_status:  # running/scheduled row, or "unknown" on a missing snapshot (fail-soft)
                continue
            if tid in ledger_ids and not require_waitable:
                continue
        except Exception:
            continue  # unreadable surface: treat as known
        unknown.append(tid)
    return unknown


def _children_roster_projection(
    ctx: ToolContext, status_drive_root: Path, *, limit: int = 30,
) -> Dict[str, Any]:
    """This parent's DIRECT children in the v6.71.2 compact field set (task_id/
    status/accounted_upper_bound_usd/sha/outcome_axes, ABI-3 honest name) —
    never result envelopes; missing
    accounting projects null, never a confirmed-looking $0. The bound is
    DISCLOSED through the shared ``disclosed_list_projection`` (BIBLE P1): the
    payload carries ``children_roster`` plus ``children_roster_omitted``, the
    exact count of real children the cap hid — a silent ``[:limit]`` here could
    hide the very replacement id this repair surface exists to show. Fail-soft:
    an empty roster with omitted=0."""
    from ouroboros._outcome_receipts import disclosed_list_projection
    from ouroboros.task_status import find_child_tasks
    from ouroboros.tools.join_ledger import _child_result_sha256

    empty = {"children_roster": [], "children_roster_omitted": 0}
    my_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not my_id:
        return empty
    try:
        rows = find_child_tasks(
            status_drive_root, parent_task_id=my_id, root_task_id="",
            exclude_task_id=my_id, scope="direct",
        )
    except Exception:
        return empty
    from ouroboros.cost_projection import cost_projection

    roster: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        _cost = cost_projection(row)
        roster.append({
            "task_id": str(row.get("task_id") or row.get("id") or ""),
            "status": row.get("status"),
            "accounted_upper_bound_usd": _cost["accounted_upper_bound_usd"],
            "child_result_sha256": _child_result_sha256(row),
            "outcome_axes": normalize_outcome_axes(row),
            **({"execution_observation": dict(row["execution_observation"])}
               if isinstance(row.get("execution_observation"), dict) else {}),
        })
    return disclosed_list_projection(
        roster, key="children_roster", limit=max(1, int(limit)), item=lambda entry: entry,
    )


def _compact_child_projection(tid: str, data: Dict[str, Any], known_hash: Any) -> Dict[str, Any]:
    """ONE compact per-child field list, shared by the batch wait and an unsettled
    single wait: the semantic handoff, never the forensics of the full envelope."""
    from ouroboros.cost_projection import cost_projection
    from ouroboros.tools.join_ledger import _child_result_sha256

    # SSOT cost projection (C2/ABI-3): honest null (never a
    # confirmed-looking $0), the honest name only, and finality
    # only when the child's own record claims it.
    _cost = cost_projection(data)
    projected: Dict[str, Any] = {
        "task_id": str(data.get("task_id") or data.get("id") or tid),
        "status": data.get("status"),
        "accounted_upper_bound_usd": _cost["accounted_upper_bound_usd"],
        "cost_final": _cost["cost_final"],
        "child_result_sha256": _child_result_sha256(data),
        "outcome_axes": normalize_outcome_axes(data),
        "result": data.get("result"),
        "trace_summary": data.get("trace_summary"),
    }
    projected["result_chars"] = len(str(data.get("result") or ""))
    projected["trace_summary_chars"] = len(str(data.get("trace_summary") or ""))
    projected["result_source"] = {"tool": "get_task_result", "arguments": {
        "task_id": str(tid), "include_authority": True}}
    if isinstance(data.get("verification_ledger"), dict):
        projected["verification_summary"] = data["verification_ledger"].get("summary") or {}
    if isinstance(data.get("execution_observation"), dict):
        projected["execution_observation"] = dict(data["execution_observation"])
    # The result hash binds this limitation too; keep its host authorship
    # separate from the unchanged model answer, including an empty answer.
    from ouroboros.task_finalization import terminal_host_notice_text

    notice = terminal_host_notice_text(data)
    if notice:
        projected["terminal_host_notice"] = notice
    if data.get("duplicate_of"):
        projected["duplicate_of"] = str(data.get("duplicate_of"))
    # A capability reduction is a SEMANTIC handoff fact, not forensics: it is
    # what decides how far to trust this answer, and this is the surface a
    # fan-out parent absorbs its children through. Same predicate as the
    # single-child read, so the batch and the singleton cannot disagree.
    _delta = disclosable_capability_delta(data)
    if _delta:
        projected["capability_delta"] = _delta
    if _effort := disclosable_effort_fact(data):
        projected["effort"] = _effort
    # Delegation honesty (Q1A, 2026-08-10 amendments): whether a
    # harness-dispatched child ACTUALLY delegated is a handoff fact the
    # fan-out parent absorbs here — the e9108a09 incident hid nine
    # native-only "harness" children behind this very projection.
    # Compact counts only; the full evidence stays in the envelope.
    _envelope = data.get("subagent_envelope") if isinstance(data.get("subagent_envelope"), dict) else {}
    _evidence = _envelope.get("execution_evidence") if isinstance(_envelope.get("execution_evidence"), dict) else {}
    if _evidence or str(data.get("effective_executor") or "") == "harness":
        _ee: Dict[str, Any] = {
            "dispatch_executor": str(data.get("effective_executor") or ""),
        }
        if _evidence.get("evidence_read_failed"):
            # Unreadable custody log (v6.94.0 landing-gate scope fix):
            # the counts are UNKNOWN — emitting them as 0 beside the
            # marker fabricated a "no runs" receipt for a log that was
            # never read. The compact projection carries ONLY the typed
            # marker; counts AND the substrate claim are omitted, the
            # same omission rule subagents.envelope_from_task applies.
            _ee["evidence_read_failed"] = True
        else:
            if _evidence:
                # Counts only when the envelope actually attested them:
                # a result with no evidence recorded (pre-6.94) gets NO
                # zero counts — absence means "no evidence yet", not
                # "no runs".
                _ee["delegated_runs_started"] = int(_evidence.get("delegated_runs_started") or 0)
                _ee["delegated_runs_settled"] = int(_evidence.get("delegated_runs_settled") or 0)
                _ee["delegated_runs_succeeded"] = int(_evidence.get("delegated_runs_succeeded") or 0)
                _ee["delegated_runs_failed"] = int(_evidence.get("delegated_runs_failed") or 0)
                _ee["delegated_runs_source_unresolved"] = int(
                    _evidence.get("delegated_runs_source_unresolved") or 0
                )
            # The substrate claim rides only when the envelope made one.
            _substrate = str(data.get("actual_substrate") or _envelope.get("actual_substrate") or "")
            if _substrate:
                _ee["actual_substrate"] = _substrate
                # C3: counters are delegated-run facts; the native
                # (metered) contribution beside them is unknown.
                _ee["native_contribution"] = "unknown"
        projected["execution_evidence"] = _ee
    unchanged = (_unchanged_result_reference(str(tid), projected["child_result_sha256"], known_hash)
                 if data else {})
    if isinstance(data.get("cancel_origin"), dict):
        projected["cancel_origin"] = data["cancel_origin"]
    if unchanged:
        projected.pop("result", None)
        projected.pop("trace_summary", None)
        projected.update(unchanged)
    return projected


def _wait_for_tasks(
    ctx: ToolContext,
    task_ids: List[str],
    timeout_sec: int | None = None,
    mode: str = "all_terminal",
    known_result_sha256_by_task: Dict[str, str] | None = None,
    _explicit_clamp: int = _WAIT_TASKS_CLAMP_SEC,
) -> str:
    """Wait for multiple subtasks and return a compact structural projection per child.

    Loop-owned default waits return an actionable snapshot for any unminted ID,
    never a bounded slot hold or a silently narrowed subscription. Standalone
    all-unminted sets retain the bounded registration grace."""
    if not isinstance(task_ids, list) or not task_ids:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR",
            text="⚠️ TOOL_ARG_ERROR (wait_tasks): task_ids must be a non-empty list.",
        ))
    from ouroboros.config import MAX_ACTIVE_SUBAGENTS_HARD_CAP

    if len(task_ids) > MAX_ACTIVE_SUBAGENTS_HARD_CAP:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR",
            text=(
                "⚠️ TOOL_ARG_ERROR (wait_tasks): task_ids is capped at "
                f"{MAX_ACTIVE_SUBAGENTS_HARD_CAP}."
            ),
        ))
    normalized_ids: List[str] = []
    for item in task_ids:
        try:
            tid = validate_task_id(item)
        except ValueError as exc:
            return _publish_tool_result(ctx, ToolResult(
                status="error", code="TOOL_ARG_ERROR",
                text=f"⚠️ TOOL_ARG_ERROR (wait_tasks): {exc}",
            ))
        if tid not in normalized_ids:
            normalized_ids.append(tid)
    try:
        # The normalized RAW request, kept before the clamp: an expiry that
        # reports the ceiling as the asked-for window hides the very fact the
        # model needs, that its request was cut down.
        requested_timeout = float(max(0, int(timeout_sec))) if timeout_sec is not None else float(_event_wait_window(ctx))
    except (TypeError, ValueError):
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
            text="⚠️ TOOL_ARG_ERROR (wait_tasks): timeout_sec must be an integer."))
    timeout, bound = _wait_window(ctx, int(requested_timeout),
                                  clamp=_explicit_clamp if timeout_sec is not None else _event_wait_window(ctx), minimum=0,
                                  margin=NESTED_SETTLEMENT_MARGIN_SEC)
    normalized_mode = str(mode or "all_terminal").strip().lower()
    if normalized_mode not in {"all_terminal", "any_terminal"}:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR",
            text="⚠️ TOOL_ARG_ERROR (wait_tasks): mode must be all_terminal or any_terminal.",
        ))
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    status_drive_root = Path(str(metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    # Typed unknown-id detection (v6.91): flagged ids KEEP polling — "not YET
    # registered" is a real state for a just-scheduled child — but a phantom id
    # is disclosed instead of silently starving the wait (wave2: three
    # hallucinated ids blocked 900s slices while the real lead went unwaited).
    require_waitable = bool(timeout_sec is None and callable(getattr(ctx, "owner_wait_callback", None)))
    entry_unknown_ids = (_unminted_wait_ids(ctx, status_drive_root, normalized_ids, True)
                         if require_waitable else _unminted_wait_ids(ctx, status_drive_root, normalized_ids))
    ready_attention = None
    unknown_repair = bool(timeout_sec is None and entry_unknown_ids
                          and callable(getattr(ctx, "owner_wait_callback", None)))
    if unknown_repair:
        # A phantom cannot satisfy all_terminal. Preserve the exact set and
        # disclose repair now; the corrected known-ID set parks warm normally.
        timeout = 0
    if timeout_sec is None and timeout > 0 and not entry_unknown_ids and callable(getattr(ctx, "owner_wait_callback", None)):
        from ouroboros import model_sleep

        try:
            chosen = model_sleep.selectors(ctx, tasks=normalized_ids,
                wake_after_sec=timeout if bound == "deadline" else None)
            chosen["terminal_mode"] = normalized_mode
            chosen["known_result_sha256_by_task"] = known_result_sha256_by_task or {}
            outcome = model_sleep.request_sleep(ctx, chosen, model_sleep.MODE_WARM)
        except ValueError as exc:
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
                text=f"⚠️ TOOL_ARG_ERROR (wait_tasks): {exc}"))
        if outcome.get("reason") == "sleep_armed":
            return bounded_wait_response(ctx, {**outcome, "task_ids": normalized_ids, **wait_mail_preview(ctx)})
        # Already ready: read the complete snapshot, including simultaneous
        # terminals and attention. No extra cognitive round is needed to fetch it.
        ready_attention = outcome.get("wake_beacons")
        timeout = 0
    # One beacon cursor for the whole wait, so a two-phase window cannot skip an
    # attention beacon emitted during its first phase.
    _wait_since = ""
    # A wait set in which EVERY id is unminted cannot be satisfied by waiting —
    # nothing was ever scheduled to terminate. Spend only the registration-race
    # grace on it (wave1's root blocked its whole window on three hallucinated
    # ids), then re-probe; the moment any id turns real this becomes an ordinary
    # wait and gets the rest of the requested window.
    _phantom_only = bool(entry_unknown_ids) and len(entry_unknown_ids) == len(normalized_ids)
    first_window = min(float(timeout), _UNMINTED_WAIT_GRACE_SEC) if _phantom_only else float(timeout)
    waited = wait_for_effective_tasks(
        status_drive_root, normalized_ids, timeout_sec=first_window, mode=normalized_mode,
        on_poll=_wait_attention_poll(ctx, _wait_since, normalized_ids), poll_interval_sec=2.0,
    )
    if _phantom_only and first_window < float(timeout) and waited.get("early_return") is None:
        entry_unknown_ids = _unminted_wait_ids(ctx, status_drive_root, normalized_ids)
        if len(entry_unknown_ids) < len(normalized_ids):
            elapsed = float(waited.get("elapsed_sec") or 0.0)
            resumed = wait_for_effective_tasks(
                status_drive_root, normalized_ids,
                timeout_sec=max(0.0, float(timeout) - elapsed), mode=normalized_mode,
                on_poll=_wait_attention_poll(ctx, _wait_since, normalized_ids), poll_interval_sec=2.0,
            )
            resumed["elapsed_sec"] = float(resumed.get("elapsed_sec") or 0.0) + elapsed
            resumed["timeout_sec"] = float(timeout)
            waited = resumed
        else:
            # Disclosed, not silent: the wait ended early and says why.
            waited["wait_short_circuited"] = {
                "reason": "all_task_ids_unminted",
                "requested_timeout_sec": float(timeout),
                "waited_sec": round(float(waited.get("elapsed_sec") or 0.0), 1),
                "note": (
                    "Every requested task_id was unminted at entry and still unminted after "
                    f"the {int(_UNMINTED_WAIT_GRACE_SEC)}s registration grace, so the wait "
                    "returned instead of blocking for the full timeout. Fix the wait set from "
                    "children_roster / your schedule_subagent results, then wait again."
                ),
            }
    tasks = waited.get("tasks")
    if unknown_repair:
        waited["wait_short_circuited"] = {
            "reason": "unknown_task_ids_require_repair",
            "requested_timeout_sec": requested_timeout,
            "waited_sec": float(waited.get("elapsed_sec") or 0),
            "note": "Repair unknown_task_ids from children_roster; no IDs were silently removed from the wait set.",
        }
    if ready_attention:
        waited["early_return"] = ready_attention
    if isinstance(tasks, dict):
        # Re-probe the entry-time unknowns once: an id minted mid-wait (queue
        # row or result appeared) is a real child, not a phantom.
        unknown_ids = [tid for tid in entry_unknown_ids if not tasks.get(tid)]
        if unknown_ids:
            unknown_ids = (_unminted_wait_ids(ctx, status_drive_root, unknown_ids, True)
                           if require_waitable else _unminted_wait_ids(ctx, status_drive_root, unknown_ids))

        # Compact STRUCTURAL projection (v6.71.2): the full public_task_result
        # envelope duplicated forensics (trace_refs, loop_outcome internals,
        # verification_ledger) into the parent context on every batch absorb.
        # The parent decision needs the semantic handoff only; the full envelope
        # stays on disk in task_results/<id>.json, addressable by
        # child_result_sha256 (the join-ledger SSOT hash), and is fetched with
        # get_task_result — a DISCLOSED omission (BIBLE P1), not silent
        # truncation. get_task_result is the explicit full terminal reader;
        # all wait handoffs are bounded with actor-readable complete sources.
        public_tasks: Dict[str, Any] = {}
        for tid, data in tasks.items():
            if str(tid) in unknown_ids:
                public_tasks[str(tid)] = {
                    "task_id": str(tid),
                    "status": None,
                    "unknown_task_id": True,
                    "note": (
                        "UNKNOWN_TASK_ID: no waitable result/queue source" if require_waitable else
                        "UNKNOWN_TASK_ID: not yet registered or never scheduled — no result, queue row or tree-ledger mention"
                    ) + (
                        ". Check it against your schedule_subagent results / the "
                        "children_roster below; an all_terminal wait cannot complete while "
                        "its result/queue source is unavailable."
                    ),
                }
                continue
            if not isinstance(data, dict):
                public_tasks[str(tid)] = data
                continue
            known = (known_result_sha256_by_task.get(str(tid))
                     if isinstance(known_result_sha256_by_task, dict) else None)
            public_tasks[str(tid)] = _compact_child_projection(str(tid), data, known)
            if str(data.get("status") or "").lower() not in SETTLED_STATUSES:
                public_tasks[str(tid)]["delegated_runs"] = _delegated_run_facts(status_drive_root, str(tid))
        waited["tasks"] = public_tasks
        waited["tasks_note"] = (
            "Compact per-child projection. The full result envelope (trace_refs, "
            "loop_outcome, verification_ledger) remains on disk in task_results/"
            "<task_id>.json, addressable by child_result_sha256; get_task_result "
            "returns the full result text plus trace/outcome summaries."
        )
        if unknown_ids:
            waited["unknown_task_ids"] = unknown_ids
            # The repair surface: the ACTUAL direct children, compact v6.71.2
            # field set only (never envelopes), so the parent can fix its wait
            # set instead of re-polling phantoms. Carries children_roster plus
            # the disclosed children_roster_omitted count (never a silent cap).
            waited.update(_children_roster_projection(ctx, status_drive_root))
    # Disclosed, not silent: the window ended while children were still live.
    # FACTS only, and deliberately no advisory note — how wide a window to ask
    # for next is the mind's call (unlike the short-circuit above, whose note
    # names a broken wait SET); the long-term orientation lives in the schema
    # description the model reads BEFORE it chooses a window. An id this tree
    # never minted is disclosed as unknown, never counted as a live child.
    projected = waited.get("tasks")
    if (not unknown_repair and waited.get("timed_out") and not waited.get("all_terminal")
            and isinstance(projected, dict) and projected):
        live_ids = [
            tid for tid in normalized_ids
            if not (projected.get(tid) or {}).get("unknown_task_id")
            and str((projected.get(tid) or {}).get("status") or "").strip().lower()
            not in SETTLED_STATUSES
        ]
        if live_ids:
            waited["wait_expired_with_live_children"] = {
                "reason": "timeout_expired_before_terminal",
                "requested_timeout_sec": requested_timeout,
                "max_timeout_sec": float(_explicit_clamp if timeout_sec is not None else _event_wait_window(ctx)),
                "live_task_ids": live_ids,
            }
    if bound != "requested":
        waited["window_sec"], waited["window_bound"] = float(timeout), bound
    if note := _wait_early_return_note(waited.get("early_return")):
        waited["early_return_note"] = note
    horizon_note = cache_horizon_note(ctx, waited.get("elapsed_sec"))
    if horizon_note:
        waited["cache_horizon_note"] = horizon_note
    waited.update(wait_mail_preview(ctx))
    return bounded_wait_response(ctx, waited)
