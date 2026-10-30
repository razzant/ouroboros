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

from ouroboros.model_sleep import HOLD_NOTE, IN_SLOT_HOLD_NOTE, hold_notice, in_slot_wait, unshown_holds
from ouroboros.outcomes import normalize_outcome_axes
from ouroboros.runtime_limits import NESTED_SETTLEMENT_MARGIN_SEC
from ouroboros.task_results import (
    STATUS_CANCELLED,
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
from ouroboros.utils import truncate_review_artifact
from ouroboros.work_stop import stop_for_reader, task_current_stop
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result, completed_local_read


def disclosable_capability_delta(data: Dict[str, Any]) -> Dict[str, Any]:
    """The child's delta when it has something to SAY, else ``{}`` — ONE predicate.

    THE terminal parent-facing disclosure, and since v6.87.28 the only parent-facing
    one: the reduction is not known until the child is dispatched, so no scheduling
    result can carry it. It is a predicate rather than an inline test because the
    parent absorbs a child through TWO surfaces — `get_task_result`/`wait_task` read
    one child in full, `wait_tasks` projects a batch compactly — and the batch one
    is the surface a fan-out parent actually uses. It had the test in neither place
    and the disclosure in one, so a parent that scheduled five children and absorbed
    them in a burst was told nothing about any of them.

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


def _subtask_outcome_summary(data: Dict[str, Any], receipts: list | None = None, reader: str = "") -> str:
    from ouroboros.delegate_output import project_task_output_readers

    data = project_task_output_readers(data)
    ledger = data.get("verification_ledger") if isinstance(data.get("verification_ledger"), dict) else {}
    summary: Dict[str, Any] = {
        "outcome_axes": normalize_outcome_axes(data),
    }
    if isinstance(data.get("execution_observation"), dict):
        summary["execution_observation"] = dict(data["execution_observation"])
    if stop := stop_for_reader(data.get("current_stop") or {}, data, reader):
        summary["current_stop"] = stop  # what holds it now, beside the historical axes
    if isinstance(data.get("swarm_receipt"), dict):
        summary["swarm_receipt"] = data["swarm_receipt"]
    # R5: current custody beside historical axes, on full get_task_result/wait_task
    # only; the compact wait_tasks contract stays pinned. Frozen delegated_runs_*
    # counters remain historical (Q2=B); the envelope's trigger/open_run_ids show
    # current liveness, with capped lists and exact omitted counts.
    unreconciled = data.get("delegated_runs_unreconciled")
    unreconciled = (
        [str(item) for item in unreconciled] if isinstance(unreconciled, list) else []
    )
    envelope = data.get("delegate_terminal_reconciliation")
    envelope = envelope if isinstance(envelope, dict) else {}
    # #1573: an effective retry view keeps each physical predecessor's retained outputs
    # on its lineage hop; their source_read names the task that publishes them.
    hops = [hop for hop in (data.get("retry_lineage") or []) if isinstance(hop, dict)]
    predecessor_runs = [{k: v for k, v in row.items() if k != "selected_subagent_id"}
                        for hop in hops for row in (hop.get("terminal_runs") or []) if isinstance(row, dict)]
    if unreconciled or envelope or predecessor_runs:
        custody: Dict[str, Any] = {"unreconciled": unreconciled[:10]}
        omitted_any = False
        if len(unreconciled) > 10:
            custody["unreconciled_omitted"] = len(unreconciled) - 10
            omitted_any = True
        if envelope:
            custody["trigger"] = str(envelope.get("trigger") or "")
            custody["audit_status"] = str(envelope.get("audit_status") or "unknown")
            if isinstance(envelope.get("owner_stop_children"), dict):
                receipt = envelope["owner_stop_children"]
                custody["owner_stop_children"] = {
                    "count": len(receipt.get("children") or []),
                    "full_source": receipt.get("full_source"),
                    **({"gap": receipt["gap"]} if receipt.get("gap") else {}),
                }
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
        if predecessor_runs:
            custody["predecessor_terminal_runs"] = predecessor_runs[:10]
            if len(predecessor_runs) > 10:
                custody["predecessor_terminal_runs_omitted"] = len(predecessor_runs) - 10
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
            ids = [tid]
            for other in (origin, *(str(hop.get("task_id") or "") for hop in hops)):
                if other and other not in ids:
                    ids.append(other)
            custody["full_source"] = [f"get_task_result(task_id='{task_id}', include_authority=True)" for task_id in ids]
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
        "description": "Read a task's selected answer, current outcome and evidence (view=answer/summary/evidence/full), or its exact authority/source ranges. Omitted/full returns the complete response; short views do not count as reading authority or sources.",
        "parameters": {"type": "object", "required": ["task_id"], "properties": {
            "task_id": {"type": "string", "description": "Task ID returned by scheduling or exposed by the host routing manifest."},
            "call_id": {"type": "string", "description": "With manifest_sha256: read one retained redacted tool call or script dispatch of this task. Uses the source range and presence scope parameters."},
            "manifest_sha256": {"type": "string", "description": "Exact retained manifest digest for call_id. Omit ranges for source length/hash; provide both bounds for text."},
            "known_result_sha256": {"type": "string", "description": "Conditional child_result_sha256 identity in every view; summary/evidence still refresh dynamic facts."},
            "view": {"type": "string", "enum": ["answer", "summary", "evidence", "full"], "description": "Optional semantic result view. Omitted/full preserves the legacy response exactly. Short views retain answer identity, terminal origin, outcome, host notice, and capability facts; summary adds current cost/custody/retry; evidence adds only authorized current-projection recipes."},
            "include_authority": {"type": "boolean", "default": False, "description": "Return the exact selected result, task contract, origin, artifact references, and current plan-review authority."},
            "authority_fields": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Explicit exact top-level keys of the same current authority projection; [] returns only its field index. Unknown names error. Selected values are whole; every omitted key has size/hash and a reread recipe. Not a full-authority source even with include_authority. Omit for legacy behavior."},
            "source_only": {"type": "boolean", "default": False, "description": "Explicitly omit the authority body from a source read. With authority_fields (or include_authority), return selection metadata, then use source_start_char/source_end_char over that selection's exact JSON. Existing source calls without this flag keep their legacy combined response."},
            "authority_projection_sha256": {"type": "string", "default": "", "description": "Expected current authority-projection digest from this reader's explicit fields/source-only response, NOT the continuation envelope's authority_sha256 or child_result_sha256. Drift returns authority_identity_mismatch without values/text; rereads are immutable=false."},
            "include_work_order_source": {"type": "boolean", "default": False, "description": "Return the canonical work-order source projection; provide both source_start_char and source_end_char for the exact bounded range."},
            "include_completion_source": {"type": "boolean", "default": False,
                                          "description": "Read the full stored completion observations for this task, including returns omitted from the summary. Omit bounds for source length/hash, then request explicit character ranges."},
            "include_focus_source": {"type": "boolean", "default": False, "description": "Read the exact bytes this task's focus source_ref answered when the focus was authored (the retained_source of an [INDEPENDENT_ROOTS] row); same bounds contract as include_completion_source."},
            "focus_source_sha256": {"type": "string", "default": "", "description": "With include_focus_source: select the retained source by the sha256 the roster row quoted, so a later focus of the same author cannot substitute its evidence."},
            "review_source_sha256": {"type": "string", "default": "", "description": "Root turns: read only the exact acceptance-review source named by a late-evidence digest, pinned to the physical task_id even after a retry. Works across forked/empty drives. Without a range returns complete_chars/hash; then use source_start_char/source_end_char to read exact text. Does not include authority."},
            "delegated_output_sha256": {"type": "string", "default": "", "description": "Read a retained full delegated output that this PHYSICAL task_id's own receipts publish, pinned even after a retry: call exactly the source_read of a staged_output/other_staged_outputs or terminal_observation/other_terminal_observations ref (unordered; each says complete; an observation is the exact engine read-back, not proof of full output) named in public summary delegated_custody.terminal_runs or predecessor_terminal_runs (it names the publishing task; the ref's task_id is only the staging store). Works across forked/empty drives and after drive cleanup. Omit ranges for complete_chars/hash, then use source_start_char/source_end_char. Never marks the run's result as read; does not include authority."},
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
    view_head_chars: Any = None, view_tail_chars: Any = None, delegated_output_sha256: str = "",
    view: str = "", call_id: str = "", manifest_sha256: str = "",
    authority_fields: Any = None, source_only: bool = False, authority_projection_sha256: str = "",
) -> str:
    """Read a task result, or a bounded canonical work-order/completion source range.

    ``view_head_chars``/``view_tail_chars`` are accepted here so the declared
    parameters bind; the loop's delivery honors them when it shapes the first
    view of this answer. They select nothing in the source arithmetic below.
    """
    from ouroboros.tools.task_result_views import validate_result_selectors

    try:
        selected_view, other_source = validate_result_selectors(locals())
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR", text=f"⚠️ TOOL_ARG_ERROR (get_task_result): {exc}",
        ))
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
    from ouroboros.promotion_intent import promotion_intent_notice

    notice = promotion_intent_notice(task_id, data)
    if notice:
        return _publish_tool_result(ctx, ToolResult(status="unavailable", code="LEGACY_UNAVAILABLE", text=notice))
    if authority_fields is not None or source_only and include_authority and not other_source:
        from ouroboros.tools.task_result_views import authority_selection_response

        return authority_selection_response(ctx, data, status_drive_root, locals())
    if call_id:
        from ouroboros.tools.task_result_views import retained_call_source

        try:
            source = retained_call_source(ctx, status_drive_root, data, str(task_id), call_id,
                                          manifest_sha256, source_start_char, source_end_char)
        except PermissionError as exc:
            return _publish_tool_result(ctx, ToolResult(status="blocked", code="ACCESS_BLOCKED", text=str(exc)))
        except FileNotFoundError:
            return _publish_tool_result(ctx, ToolResult(status="unavailable", code="LEGACY_UNAVAILABLE",
                text="Retained call has no available redacted projection."))
        except (OSError, ValueError, TypeError):
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ERROR",
                text="Retained call source is unreadable or failed verification."))
        text = json.dumps({"call_source": source}, ensure_ascii=False, sort_keys=True)
        if source.get("reason") == "source_range_invalid":
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=text))
        return text
    if presence_reentry_sha256:
        from ouroboros.presence_continuation import reentry_source_projection

        source = reentry_source_projection(status_drive_root, str(task_id), presence_reentry_sha256,
                                            source_start_char, source_end_char, presence_reentry_offset)
        text = json.dumps({"presence_reentry_source": source}, ensure_ascii=False, sort_keys=True)
        if source.get("reason") == "source_range_invalid":
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=text))
        return text
    if delegated_output_sha256:
        from ouroboros.delegate_output import delegated_output_source_projection
        from ouroboros.task_results import load_task_result

        # Only a source this PHYSICAL task's own receipts publish (#1573): the
        # staging task and path come from that ref, never from the caller.
        physical = load_task_result(status_drive_root, str(task_id))
        source = delegated_output_source_projection(
            status_drive_root, physical if isinstance(physical, dict) else {},
            delegated_output_sha256, source_start_char, source_end_char)
        text = json.dumps({"delegated_output_source": source}, ensure_ascii=False, sort_keys=True)
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

        authority = {} if source_only else task_result_authority_projection(data, drive_root=status_drive_root)
        payload: Dict[str, Any] = {
            "status": "available", **({} if source_only else {"authority": authority}),
            "source": {"tool": "get_task_result", "task_id": str(task_id)},
        }
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
        if source_only and isinstance(payload["source"].get("arguments"), dict):
            payload["source"]["arguments"]["source_only"] = True
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
    try:
        from ouroboros.outcomes import read_verification_receipts_from_roots
        from ouroboros.task_status import _child_drive_candidates

        # During the pre-copy-back window ordinary verification lives on the
        # isolated child drive while a zero-run lifecycle receipt is already on
        # the canonical root. Merge both replicas; a non-empty canonical file is
        # not evidence that the local one contains nothing new.
        receipts = read_verification_receipts_from_roots(
            [*_child_drive_candidates(data), status_drive_root], task_id,
        )
    except Exception:
        receipts = []
    reader = str(getattr(ctx, "task_id", "") or "")
    outcome_summary = _subtask_outcome_summary(data, receipts=receipts, reader=reader)
    from ouroboros.tools.join_ledger import _child_result_sha256

    child_result_sha256 = _child_result_sha256(data)
    # SSOT cost projection (C2): unknown never renders as $0.00 (and a null in
    # the stored result no longer crashes the f-string with a TypeError).
    from ouroboros.cost_projection import cost_display

    if selected_view and selected_view != "full":
        from ouroboros.tools.task_result_views import render_result_view

        return render_result_view(
            data, task_id=str(task_id), view=selected_view,
            known_result_sha256=known_result_sha256,
            child_result_sha256=child_result_sha256,
            outcome_summary=outcome_summary,
            capability_delta=disclosable_capability_delta(data),
            effort=disclosable_effort_fact(data), presence_scope=str(presence_scope or ""),
            current_stop=stop_for_reader(data.get("current_stop") or {}, data, reader),
        )

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
        origin = data["cancel_origin"]
        command = {"immediate": "stop_now", "finalize_then_cancel": "wrap_up"}.get(origin.get("stop_policy"), "unknown")
        marker = "[CANCELLED_BY]" if status == STATUS_CANCELLED else f"[OWNER_CONTROL] command={command}"
        output += f"\n\n{marker} {json.dumps(origin, ensure_ascii=False)}"
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
    ctx: ToolContext, after_ts: str, task_ids: List[str],
) -> Callable[..., Any]:
    """on_poll hook: break a sliced wait for the actor's mailbox, a waited task held by a current stop
    this actor has not seen (``model_sleep.unshown_holds``), or a child attention beacon
    (blocker/question/interface_contract/review_requested/delegation_constraint).

    The cursor is context-local and per child: a beacon written before this
    particular tool call is still delivered, while a later wait in the same
    actor context does not replay it.  Equal-timestamp rows use their stable
    content identity, so the five-row response bound cannot strand the rest.
    """
    # tree_note/tree_read live in ouroboros/tools/task_tree.py (extracted for module size).
    from ouroboros.tools.task_tree import tree_root_id
    from ouroboros.owner_mailbox import OwnerMailboxPeek

    rid = tree_root_id(ctx)
    mailbox_peek = OwnerMailboxPeek()

    cursor_store = getattr(ctx, "_wait_attention_cursors", None)
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
        from ouroboros.loop_transport import _owner_signal_pending

        # Reuse transport-wait's non-destructive peek. The ordinary round-top
        # drain still owns delivery and acknowledgement; child work stays live.
        if _owner_signal_pending(
            None, getattr(ctx, "drive_root", None), str(getattr(ctx, "task_id", "") or ""),
            getattr(ctx, "_loop_mailbox_seen_ids", None), getattr(ctx, "task_attempt", None) or 1,
            mailbox_peek,
        ):
            return {"reason": "owner_mailbox_pending", "delivery": "pending_loop_drain"}
        # A waited task a stop holds, at an identity this actor has not seen: shown once, then waited on.
        if held := unshown_holds(ctx, {tid: _results.get(tid) for tid in task_ids}):
            return {"reason": "dependency_held", "held": held}
        if not rid:
            return None
        try:
            from ouroboros.task_tree_ledger import (
                tree_ledger_attention_after,
                tree_ledger_row_id,
            )

            attention = tree_ledger_attention_after(rid, "", task_ids=set(task_ids))
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

    REACHABILITY, honestly: the root waits (``wait_task``, ``wait_tasks``) feed
    their own elapsed window, a LOWER bound on cache age, so at the shipped ``1h``
    tier only ``wait_tasks`` (7200s ceiling) can cross inside the window and
    ``wait_task`` sits on the 3600s horizon; at ``5m`` both cross. ``delegate_wait``
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


def _wait_early_return_note(early) -> str:
    if not early:
        return ""
    if early.get("reason") == "owner_mailbox_pending":
        return "The ordinary loop will deliver and acknowledge the unread message. This does not stop the child."
    if early.get("reason") == "dependency_held":
        return HOLD_NOTE
    return "A child attention beacon interrupted this wait. Inspect early_return; the children keep running."


def _terminal_probe(ctx: ToolContext, ids: list) -> Callable:
    """Adapt the existing status poll to the common, non-consuming readiness predicate."""
    from ouroboros.model_sleep import terminal_readiness

    def probe(results):
        ready = {row["id"] for row in terminal_readiness(ctx, {"tasks": ids}, task_results=results)}
        return {tid: f"task:{tid}" in ready for tid in ids}
    return probe


def _wait_for_task(
    ctx: ToolContext, task_id: str, timeout_sec: int = 180, known_result_sha256: str = "",
    view: str = "",
) -> str:
    """Wait for a subtask to reach a terminal status."""
    from ouroboros.tools.task_result_views import validate_result_view

    try:
        selected_view = validate_result_view(view)
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR", text=f"⚠️ TOOL_ARG_ERROR (wait_task): {exc}",
        ))
    try:
        tid = validate_task_id(task_id)
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR",
            text=f"⚠️ TOOL_ARG_ERROR (wait_task): {exc}",
        ))
    try:
        requested = int(timeout_sec)
    except (TypeError, ValueError):
        requested = 180
    timeout, bound = _wait_window(ctx, requested, clamp=_WAIT_TASK_CLAMP_SEC, minimum=0,
                                  margin=NESTED_SETTLEMENT_MARGIN_SEC)
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    status_drive_root = Path(str(metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    with in_slot_wait(ctx, timeout, "tasks", task_ids=[tid]):
        waited = wait_for_effective_tasks(
            status_drive_root, [tid], timeout_sec=timeout,
            terminal_probe=_terminal_probe(ctx, [tid]),
            on_poll=_wait_attention_poll(ctx, "", [tid]), poll_interval_sec=2.0,
        )
    early = waited.get("early_return")
    hold_meta, withdrawn = _held_return(ctx, early)
    if early and early.get("reason") == "owner_mailbox_pending":
        header = "Task wait interrupted by an unread message for this task"
        extra = "\n\n" + _wait_early_return_note(early)
    elif hold_meta:
        header = "Task wait returned to show a held dependency"
        extra = (f"\n\n[HELD_DEPENDENCY]\n{json.dumps(early['held'], ensure_ascii=False, indent=2)}\n"
                 f"[/HELD_DEPENDENCY]\n{_wait_early_return_note(early)}{withdrawn}")
    elif early:
        header = "Task wait interrupted by a child attention beacon"
        extra = f"\n\n[CHILD_BEACONS]\n{json.dumps(early, ensure_ascii=False, indent=2)}\n[/CHILD_BEACONS]"
    else:
        header = "Task wait completed" if waited.get("all_terminal") else "Task wait timed out"
        extra = ""
    # B2 advisory (never a gate): if ANY other child of THIS parent is still in flight
    # while we block on this one, point at wait_tasks(any_terminal) so the agent absorbs
    # whichever finishes first instead of blocking serially on one id at a time.
    held: List[str] = []
    other_live = _count_live_sibling_children(ctx, status_drive_root, exclude_task_id=tid, held=held)
    if other_live - len(held) >= 1:
        extra += (
            f"\n\n[ADVISORY] {other_live - len(held)} other child(ren) still running/scheduled — consider "
            "wait_tasks(any_terminal) to absorb whichever finishes first instead of waiting one at a time."
        )
    if held:
        extra += (f"\n\n[ADVISORY] {len(held)} other child(ren) held by a current stop ({', '.join(held)}): "
                  "none reaches a terminal state until its stop is released; get_task_result shows its current_stop.")
    horizon_note = cache_horizon_note(ctx, waited.get("elapsed_sec"))
    if horizon_note:
        extra += f"\n\n{horizon_note}"
    if bound != "requested":
        window = {"requested_sec": requested, "window_sec": timeout, "window_bound": bound}
        extra += f"\n\n[WAIT_WINDOW] {json.dumps(window)}"
    tasks = waited.get("tasks") if isinstance(waited.get("tasks"), dict) else {}
    result = _unsettled_wait_body(
        status_drive_root, tid, tasks.get(tid),
        known_result_sha256 if selected_view in {"", "full"} else "", reader=str(getattr(ctx, "task_id", "") or ""),
    )
    if not result:
        kwargs = {"view": selected_view} if selected_view else {}
        if known_result_sha256:
            kwargs["known_result_sha256"] = known_result_sha256
        result = _get_task_result(ctx, tid, **kwargs)
    return _with_held(ctx, f"{header} after {waited.get('elapsed_sec', 0):.1f}s.{extra}\n\n{result}", hold_meta)


def _unsettled_wait_body(drive_root: Path, tid: str, data: Any, known_result_sha256: str, *, reader: str = "") -> str:
    """The compact body of a wait_task that returned BEFORE the child settled
    (timeout, mailbox interrupt, attention beacon), or ``""`` when the full
    single-child handoff is due: a settled child, an unknown or admission-pending
    id, or a caller whose ``known_result_sha256`` no longer matches (the result
    changed, so it gets the full body). The ``Task <id> [<status>]`` header of
    the full read is kept; the full envelope stays one get_task_result away."""
    from ouroboros.routing_wait import is_emitted_admission_stub
    from ouroboros.tools.join_ledger import _child_result_sha256

    if not isinstance(data, dict) or not data or is_emitted_admission_stub(data):
        return ""
    status = data.get("status", "unknown")
    if str(status or "").strip().lower() in SETTLED_STATUSES:
        return ""
    sha = _child_result_sha256(data)
    if known_result_sha256 and known_result_sha256 != sha:
        return ""
    projection = _compact_child_projection(tid, data, known_result_sha256 or None)
    _reader_stop(projection, data, reader)
    projection["delegated_runs"] = _delegated_run_facts(drive_root, projection["task_id"])
    return (
        f"Task {tid} [{status}]\nchild_result_sha256={sha}\n\n[CHILD_PROJECTION]\n"
        f"{json.dumps(projection, ensure_ascii=False, indent=2, default=str)}\n[/CHILD_PROJECTION]\n"
        f"Compact projection of an unsettled child; get_task_result({tid}) returns the full envelope."
    )


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
        from ouroboros.delegate_supervision_state import normalize

        record = normalize(json.loads(path.read_text(encoding="utf-8")))
        reason = ""
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
        elif run.run_id not in record["runs"]:
            row["supervision"] = {"state": "unknown", "reason": "record_names_another_run"}
        else:
            state = record["runs"][run.run_id]
            observation = state.get("observation") if isinstance(state.get("observation"), dict) else None
            entry = state.get("sleep_entry") if isinstance(state.get("sleep_entry"), dict) else {}
            last = str(state.get("last_answered_observation_at") or "")
            row["supervision"] = {
                "hold_entered_at": str(entry.get("entered_at") or "unknown"),
                "last_answered_observation_at": last or "unknown",
                "observation_age_sec": _age_sec(last, now) if last else None,
                "observed_run_state": str((observation or {}).get("run_state") or "unknown"),
                "observation_failure": (
                    "unknown" if observation is None else "" if observation.get("answered")
                    else str(observation.get("reason") or observation.get("status") or "unanswered")),
                "journal_cursor": state.get("journal_cursor"),
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


def _await_messages(ctx: ToolContext, timeout_sec: int = 0, mode: str = "in_slot", senders: Any = None,
                    tasks: Any = None, runs: Any = None, wake_at: Any = None, wake_after_sec: Any = None,
                    services: Any = None) -> str:
    """Hold this task's worker slot until an unread mailbox entry exists or the
    window elapses. Delivers nothing: the round-top drain owns delivery and
    acknowledgement, exactly as after a wait_task early return. An owner Stop
    is a mailbox control, so it ends the wait like any message; a cancel kills
    the worker process; the window is bounded by ``_wait_window``.

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
    dates the turn the wait was for as silent. Its separate in_slot_wait mark identifies
    the wait for reply-preparation readers, adds no idle exemption and lends no
    slot; tests/test_await_messages.py drives that lifecycle.
    """
    from ouroboros.model_sleep import wake_reason

    if mode != "in_slot" or senders or tasks or runs or services or wake_at or wake_after_sec:
        return _await_as_sleep(ctx, mode, senders=senders, tasks=tasks, runs=runs, services=services,
                               wake_at=wake_at, wake_after_sec=wake_after_sec)
    try:
        requested = int(timeout_sec)
    except (TypeError, ValueError):
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR",
            text="⚠️ TOOL_ARG_ERROR (await_messages): timeout_sec must be an integer number of seconds.",
        ))
    if held := unshown_holds(ctx, {}):  # only what an earlier call of this batch returned: no read, no wait
        meta, withdrawn = hold_notice(ctx, held)
        return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", meta=meta, text=json.dumps(
            {"reason": "dependency_held", "elapsed_sec": 0.0, "held": held, "note": IN_SLOT_HOLD_NOTE + withdrawn},
            ensure_ascii=False)))
    from ouroboros.config import get_per_call_timeout_ceiling_sec

    window, bound = _wait_window(ctx, requested, clamp=int(get_per_call_timeout_ceiling_sec()),
                                 minimum=1, margin=1)
    start = time.monotonic()
    with in_slot_wait(ctx, window):
        while True:
            # The transport wait's non-destructive peek over a COPY of the seen
            # set; the mailbox and its acknowledgements are untouched.
            observed = wake_reason(ctx, {"any_mail": True})
            pending = bool(observed)
            elapsed = time.monotonic() - start
            if pending or elapsed >= window:
                break
            time.sleep(min(_AWAIT_MESSAGES_POLL_SEC, max(0.0, window - elapsed)))
    if pending:
        reason = observed if observed.startswith("run_question:") else "owner_mailbox_pending"
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
            raise ValueError("senders/tasks/runs/services/wake_at/wake_after_sec select a sleep: requires mode=warm or cold; for example await_messages(mode=\"warm\", wake_after_sec=60)")
        if not callable(getattr(ctx, "owner_wait_callback", None)):
            raise ValueError("this task has no continuation owner to sleep under")
        outcome = model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, **chosen), mode)
    except (TypeError, ValueError) as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR", text=f"⚠️ TOOL_ARG_ERROR (await_messages): {exc}",
            meta={"operation_outcome": "completed_no_effect"}))
    held, withdrawn = hold_notice(ctx, outcome["held"]) if outcome.get("held") else ({}, "")
    if withdrawn:
        outcome["note"] += withdrawn
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=json.dumps(outcome, ensure_ascii=False),
                                                meta={"sleep_question_notice": bool(outcome.get("questions")), **held}))


def await_messages_entry() -> ToolEntry:
    """The await_messages catalog entry, owned beside its handler; the kill timeout
    sits 60s above the largest window the tool can choose (the per-call ceiling)."""
    from ouroboros.config import get_per_call_timeout_ceiling_sec

    return ToolEntry("await_messages", {
        "name": "await_messages",
        "description": (
            "Wait without model rounds for this task's mail, or use mode=warm/cold for event-driven sleep. In-slot, the default, waits until an unread message for THIS task exists "
            "in your mailbox (an addressed contribution from a peer task, a parent's steering, "
            "an owner message, a child's escalation) or the window elapses. Use it when you have "
            "asked a peer or your parent for its next turn and have nothing useful to do until "
            "it arrives; you decide when waiting is worth it. The wait holds your worker slot "
            "and releases nothing; the window is clamped to the per-call timeout ceiling and, "
            "under a deadline, to the finalization emit window (the result names the bound); "
            "while it runs, the ordinary in-flight tool lease keeps the supervisor's silence attention "
            "(a notice, never a stop) off you and its completion counts as progress, so your next round "
            "starts inside a full silence window — call again to keep waiting. It delivers nothing itself: the message "
            "reaches you at the next round top, exactly as after a wait_task early return. The "
            "result says when the applied prompt-cache horizon elapsed since the last model response. "
            "mode=warm or mode=cold instead SLEEPS without holding your model slot, until what you select: "
            "mail from `senders`, the terminal of `tasks` you can read or the host's silence notice about one of "
            "them or its subtree, the terminal of delegated `runs` you "
            "own, the exit of your own `services` (warm only: the start each has now; a replaced, stopped "
            "or lost one wakes you as unknown, never as success), and/or `wake_at`/`wake_after_sec`; with "
            "nothing selected any addressed mail wakes you. The "
            "owner's messages, controls and typed descendant escalations always wake you; other unselected mail waits unread. Warm keeps your "
            "process and browser (a pooled slot is lent meanwhile). Cold saves your continuation and releases "
            "the process only after the body accepts custody of your bound external runs; local services "
            "still need warm sleep. A new question from any ordinary run you own also wakes you, even "
            "if not selected; reviewer runs stay with their panel. Question sources are delivered, not answered. "
            "Retaining an external writer retains its project lane. A source that is "
            "already ready answers at once; so does a selected task or sender held by a current stop you have "
            "not yet been shown (dependency_held, with its current_stop): the same request then sleeps. "
            "Sleep does not count as execution time; an explicit deadline "
            "still applies. A settled task is not a successful one: read its result when you wake."
        ),
        "parameters": {"type": "object", "properties": {
            "timeout_sec": {"type": "integer", "description":
                            "In-slot wait only: seconds to wait; clamped to the per-call timeout ceiling and to "
                            "the deadline emit window (the bound is reported in the result)."},
            "mode": {"type": "string", "enum": ["in_slot", "warm", "cold"],
                     "description": "in_slot (default) holds your slot for timeout_sec; warm/cold is a sleep."},
            "senders": {"type": "array", "items": {"type": "string"},
                        "description": "Requires mode=warm or cold. Sleep: task ids whose ordinary mail wakes you; owner input and descendant escalations bypass this filter."},
            "tasks": {"type": "array", "items": {"type": "string"},
                      "description": "Requires mode=warm or cold. Sleep: task ids whose terminal (any settled status), "
                                     "or a host silence notice about them or a descendant, wakes you."},
            "runs": {"type": "array", "items": {"type": "string"},
                     "description": "Requires mode=warm or cold. Sleep: your delegated run ids whose terminal wakes you."},
            "services": {"type": "array", "items": {"type": "string"},
                         "description": "Requires mode=warm or cold. Warm sleep: names of your own running services whose exit wakes you."},
            "wake_at": {"type": "string", "description": "Requires mode=warm or cold. Sleep: an absolute ISO-8601 wake time (with timezone)."},
            "wake_after_sec": {"type": "integer", "description": "Requires mode=warm or cold. Sleep: wake after this many seconds."},
        }},
    }, _await_messages, timeout_sec=get_per_call_timeout_ceiling_sec() + 60)


def _count_live_sibling_children(ctx: ToolContext, status_drive_root: Path, *, exclude_task_id: str,
                                 held: List[str] | None = None) -> int:
    """Count this parent's children still running/scheduled/requested (excluding the one
    just waited on). Advisory only — a failure returns 0 so it never breaks wait_task.

    Select indexed, stored row facts in any role, without decoding task bodies.
    The typed parent identity applies even when its body is padded or unreadable.
    ``held`` receives those a current stop holds (``work_stop.task_current_stop``: the
    queue snapshot and control facts).
    """
    parent_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not parent_id:
        return 0
    try:
        from ouroboros.task_results import (
            STATUS_REQUESTED,
            STATUS_RUNNING,
            STATUS_SCHEDULED,
        )
        from ouroboros.task_store import task_facts

        rows = task_facts(
            status_drive_root, where="parent_task_id=? AND task_id<>? AND status IN (?,?,?)",
            params=(parent_id, exclude_task_id, STATUS_RUNNING, STATUS_SCHEDULED, STATUS_REQUESTED))
        if held is not None and rows:
            from ouroboros.task_status import _load_queue_snapshot

            snapshot = _load_queue_snapshot(status_drive_root)
            held.extend(tid for tid in sorted(rows) if task_current_stop(
                status_drive_root, snapshot, tid).get("state") == "held")
        return len(rows)
    except Exception:
        return 0


_UNMINTED_WAIT_GRACE_SEC = 30.0
# The wait ceilings, read by the tool schemas, the expiry disclosure and the one
# ``_wait_window`` ladder. Each ToolEntry kill timeout stays at least ceiling +
# NESTED_SETTLEMENT_MARGIN_SEC, so a full window ends inside its executor bound.
_WAIT_TASK_CLAMP_SEC = 3600
_WAIT_TASKS_CLAMP_SEC = 7200


def _unminted_wait_ids(ctx: ToolContext, status_drive_root: Path, task_ids: List[str]) -> List[str]:
    """Ids with no trace on ANY surface this tree mints ids through: no task
    result, no queue-snapshot row, and no tree-ledger row naming them (v6.91).

    wave2's root blocked 900s slices on three hallucinated ids that wait_tasks
    silently polled as 'unknown' — while the real lead was missing from the wait
    set. The typed marker (plus the actual children roster) lets the parent
    repair its wait set instead of starving on phantoms. Fail-soft per probe: an
    unreadable surface treats the id as KNOWN — a real child must never be
    branded unknown on an I/O error."""
    from ouroboros.task_status import _load_queue_snapshot, _queue_task_status

    try:
        snapshot = _load_queue_snapshot(status_drive_root)
    except Exception:
        snapshot = {"_snapshot_invalid": True}
    ledger_ids: set = set()
    try:
        from ouroboros.task_tree_ledger import tree_ledger_rows
        from ouroboros.tools.task_tree import tree_root_id

        for row in tree_ledger_rows(tree_root_id(ctx)):
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
            if tid in ledger_ids:
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
        stop = stop_for_reader(row.get("current_stop") or {}, row, my_id)
        roster.append({
            "task_id": str(row.get("task_id") or row.get("id") or ""),
            "status": row.get("status"),
            "accounted_upper_bound_usd": _cost["accounted_upper_bound_usd"],
            "child_result_sha256": _child_result_sha256(row),
            "outcome_axes": normalize_outcome_axes(row),
            **({"execution_observation": dict(row["execution_observation"])}
               if isinstance(row.get("execution_observation"), dict) else {}),
            **({"current_stop": stop} if stop else {}),
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
    if isinstance(data.get("execution_observation"), dict):
        projected["execution_observation"] = dict(data["execution_observation"])
    if isinstance(data.get("current_stop"), dict) and data["current_stop"]:
        projected["current_stop"] = dict(data["current_stop"])  # a current fact: it outlives result suppression
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


def _reader_stop(projected: Dict[str, Any], data: Any, reader: str) -> None:
    """The projection's ``current_stop`` as this reader may act on it (``work_stop.stop_for_reader``)."""
    if projected.get("current_stop") and isinstance(data, dict):
        projected["current_stop"] = stop_for_reader(projected["current_stop"], data, reader)


def _held_return(ctx: ToolContext, early: Any) -> tuple:
    """A task wait that returned to show a held dependency: the identities its delivery records and the words for
    a sleep of this batch it withdrew (``model_sleep.hold_notice``); ``({}, "")`` for any other return."""
    if not isinstance(early, dict) or early.get("reason") != "dependency_held":
        return {}, ""
    return hold_notice(ctx, early["held"])


def _with_held(ctx: ToolContext, text: str, meta: Dict[str, Any]) -> str:
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=text, meta=meta)) if meta else text


def _wait_for_tasks(
    ctx: ToolContext,
    task_ids: List[str],
    timeout_sec: int = 600,
    mode: str = "all_terminal",
    known_result_sha256_by_task: Dict[str, str] | None = None,
) -> str:
    """Wait for multiple subtasks and return a compact structural projection per child.

    A wait set whose ids were ALL unminted at entry ends after the registration
    grace instead of the full requested window (disclosed as
    ``wait_short_circuited``); any id that turns real during the grace makes it
    an ordinary wait again, with the remaining window intact."""
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
        requested_timeout = float(max(0, int(timeout_sec)))
    except (TypeError, ValueError):
        requested_timeout = 600.0
    timeout, bound = _wait_window(ctx, int(requested_timeout), clamp=_WAIT_TASKS_CLAMP_SEC, minimum=0,
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
    entry_unknown_ids = _unminted_wait_ids(ctx, status_drive_root, normalized_ids)
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
    with in_slot_wait(ctx, first_window, "tasks", task_ids=normalized_ids):
        waited = wait_for_effective_tasks(
            status_drive_root, normalized_ids, timeout_sec=first_window, mode=normalized_mode,
            terminal_probe=_terminal_probe(ctx, normalized_ids),
            on_poll=_wait_attention_poll(ctx, _wait_since, normalized_ids), poll_interval_sec=2.0,
        )
    if _phantom_only and first_window < float(timeout) and waited.get("early_return") is None:
        entry_unknown_ids = _unminted_wait_ids(ctx, status_drive_root, normalized_ids)
        if len(entry_unknown_ids) < len(normalized_ids):
            elapsed = float(waited.get("elapsed_sec") or 0.0)
            with in_slot_wait(ctx, max(0.0, float(timeout) - elapsed), "tasks", task_ids=normalized_ids):
                resumed = wait_for_effective_tasks(
                    status_drive_root, normalized_ids,
                    terminal_probe=_terminal_probe(ctx, normalized_ids),
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
    if isinstance(tasks, dict):
        # Re-probe the entry-time unknowns once: an id minted mid-wait (queue
        # row or result appeared) is a real child, not a phantom.
        unknown_ids = [tid for tid in entry_unknown_ids if not tasks.get(tid)]
        if unknown_ids:
            unknown_ids = _unminted_wait_ids(ctx, status_drive_root, unknown_ids)

        # Compact STRUCTURAL projection (v6.71.2): the full public_task_result
        # envelope duplicated forensics (trace_refs, loop_outcome internals,
        # verification_ledger) into the parent context on every batch absorb.
        # The parent decision needs the semantic handoff only; the full envelope
        # stays in the current task store, addressable by
        # child_result_sha256 (the join-ledger SSOT hash), and is fetched with
        # get_task_result — a DISCLOSED omission (BIBLE P1), not silent
        # truncation. get_task_result and a SETTLED wait_task stay full; an
        # unsettled wait_task returns this same projection.
        public_tasks: Dict[str, Any] = {}
        for tid, data in tasks.items():
            if str(tid) in unknown_ids:
                public_tasks[str(tid)] = {
                    "task_id": str(tid),
                    "status": None,
                    "unknown_task_id": True,
                    "note": (
                        "UNKNOWN_TASK_ID: not yet registered or never scheduled — no task "
                        "result, no queue row, and no tree-ledger row names this id in this "
                        "tree. Check it against your schedule_subagent results / the "
                        "children_roster below; an all_terminal wait cannot complete while "
                        "it stays unscheduled."
                    ),
                }
                continue
            if not isinstance(data, dict):
                public_tasks[str(tid)] = data
                continue
            known = (known_result_sha256_by_task.get(str(tid))
                     if isinstance(known_result_sha256_by_task, dict) else None)
            public_tasks[str(tid)] = _compact_child_projection(str(tid), data, known)
            _reader_stop(public_tasks[str(tid)], data, str(getattr(ctx, "task_id", "") or ""))
        waited["tasks"] = public_tasks
        waited["tasks_note"] = (
            "Compact per-child projection. The full result envelope (trace_refs, "
            "loop_outcome, verification_ledger) remains in the task store, "
            "addressable by child_result_sha256; get_task_result "
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
    if (waited.get("timed_out") and not waited.get("all_terminal")
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
                "max_timeout_sec": float(_WAIT_TASKS_CLAMP_SEC),
                "live_task_ids": live_ids,
            }
    if bound != "requested":
        waited["window_sec"], waited["window_bound"] = float(timeout), bound
    hold_meta, withdrawn = _held_return(ctx, waited.get("early_return"))
    if note := _wait_early_return_note(waited.get("early_return")):
        waited["early_return_note"] = note + withdrawn
    horizon_note = cache_horizon_note(ctx, waited.get("elapsed_sec"))
    if horizon_note:
        waited["cache_horizon_note"] = horizon_note
    return _with_held(ctx, json.dumps(waited, ensure_ascii=False, indent=2), hold_meta)
