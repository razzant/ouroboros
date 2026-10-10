"""Historical acceptance inputs, independent of review execution.

Skip producers record eligibility; final-answer persistence retains exact values
and already captured evidence in the existing task source store. This is NOT a
workspace snapshot or delivery receipt. Missing historical bytes stay gaps.
The explicit tool reads this source and may amend its original root cap;
acceptance_late binds it to the existing paid operation and settlement owners.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
_EVIDENCE_SOURCE_REF_FIELDS = ("repo_diff_source_ref", "tool_trajectory_source_ref")


def historical_source_location_owned(location: str) -> bool:
    """Only host carrier locations authorize even a previously captured ref."""
    from ouroboros.observability import CALL_SOURCE_REF_FIELDS, TOOL_SOURCE_REF_FIELDS

    if location == "observations.source_ref" or location in {"evidence." + key for key in _EVIDENCE_SOURCE_REF_FIELDS}:
        return True
    if location in {f"historical_author_inputs.anchors[{index}].source_ref" for index in (0, 1)}:
        return True
    index, separator, field = location.removeprefix("trace.tool_calls[").partition("].")
    return bool(location.startswith("trace.tool_calls[") and separator and index.isdecimal()
                and field in (*TOOL_SOURCE_REF_FIELDS, *("trace_ref." + key for key in CALL_SOURCE_REF_FIELDS)))


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _zero_physical(trace: dict, root_row: dict, *, pending_panel: dict | None = None) -> bool:
    # A missing run or a pending label cannot release a paid write-ahead claim.
    from ouroboros.task_results import TASK_ACCEPTANCE_REVIEW_STATE_KEY, _validated_task_acceptance_review_state
    if TASK_ACCEPTANCE_REVIEW_STATE_KEY in root_row:
        wallet = _validated_task_acceptance_review_state(root_row[TASK_ACCEPTANCE_REVIEW_STATE_KEY], root_row["task_id"])
        if wallet["claims_by_binding"]:
            return False
    published = [panel for panel in (root_row.get("review_projection") or {}).get("panels", [])
                 if not isinstance(panel, dict) or panel.get("surface") == "task_acceptance"]
    if (root_row.get("review_status") or {}).get("run_count", 0) and not published:
        return False  # Legacy run count without physical records is unknown.
    for run in [*(trace.get("review_runs") or []), *published]:
        if not isinstance(run, dict):
            return False
        if run.get("authority") != "host_root":
            continue
        # The current live operation may publish its roster before its first
        # physical stamp. Only its exact pending actors are free at preclaim;
        # another/unknown panel and every durable paid claim still block.
        owned = (pending_panel or {}).get("operations") or {}
        same = bool(owned and run.get("binding_hash") == pending_panel.get("binding_hash"))
        actors = run.get("actors")
        if not actors or any(not isinstance(actor, dict)
                             or (actor.get("operation_state") != "not_dispatched" and not (
                                 same and actor.get("operation_state") == "pending_dispatch"
                                 and actor.get("operation_id")
                                 and owned.get(actor.get("slot_id")) == actor["operation_id"]))
                             for actor in actors):
            return False
    return True


def seed_acceptance_history(ctx: Any, trace: dict, cause: str) -> None:
    """Called only at an eligible skip; no evidence rebuild or model call."""
    from ouroboros.config import get_review_enforcement, get_task_review_mode
    from ouroboros.loop_acceptance_review import _acceptance_delivery_slots, _resolve_ctx_lineage
    from ouroboros.loop_delivery import _effective_delivery_criteria
    from ouroboros.task_results import load_task_result
    from ouroboros.tool_access import canonical_data_root

    try:
        lineage = _resolve_ctx_lineage(ctx, str(ctx.task_id or ""))
        if not lineage["is_root_task"] or (trace.get("review_decision") or {}).get("eligibility") != "eligible":
            return
        root_row = load_task_result(canonical_data_root(ctx), lineage["root_task_id"], strict=True)
        if root_row is None or not _zero_physical(trace, root_row):
            return
        cap = root_row.get("acceptance_original_root_cap") or {"state": "unknown", "source": "admission_scope_unavailable"}
        trace["acceptance_history_seed"] = _copy({
            "task_id": lineage["task_id"], "accounting_root_task_id": lineage["root_task_id"],
            "task_attempt": getattr(ctx, "task_attempt", None), "cause": cause,
            "original_root_cap": cap, "effective_criteria": _effective_delivery_criteria(ctx),
            "review_policy": {"mode": get_task_review_mode(), "enforcement": get_review_enforcement()},
            "reviewer_slots": [dataclasses.asdict(slot) for slot in _acceptance_delivery_slots()],
        })
    except Exception as exc:
        trace["acceptance_history_gap"] = type(exc).__name__
        log.warning("Acceptance history seed unavailable", exc_info=True)


def retain_acceptance_history(drive_root: Any, task: dict, text: str, trace: dict,
                              evidence: dict, observations: dict, artifacts: list,
                              delivery: dict | None) -> dict:
    """Pin final values and identities without reading any referenced source bytes.

    The ordinary copyback/GC owner promotes retained refs later. Artifact paths
    without an immutable source handle are manifests only, never permission to
    reconstruct historical evidence from whatever bytes happen to be there.
    """
    from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes
    from ouroboros.observability import CALL_SOURCE_REF_FIELDS, TOOL_SOURCE_REF_FIELDS
    from ouroboros.task_results import load_task_result
    from ouroboros.utils import utc_now_iso

    seed = trace.get("acceptance_history_seed")
    if not isinstance(seed, dict):
        return {}
    tid = str(task.get("id") or "")
    canonical = Path(task.get("budget_drive_root") or drive_root)
    try:
        root_row = load_task_result(canonical, seed["accounting_root_task_id"], strict=True)
        if root_row is None or not _zero_physical(trace, root_row):
            return {}
        delivery = delivery or {}
        answer = delivery.get("text")
        if seed["task_id"] != tid or not isinstance(answer, str):
            raise ValueError("final_delivery_body_unavailable")
        identity = {"task_id": tid, "task_attempt": seed["task_attempt"],
                    "delivery_id": str(delivery.get("delivery_id") or ""),
                    "chat_id": delivery.get("chat_id"),
                    "text_sha256": hashlib.sha256(answer.encode()).hexdigest()}
        debt_id = "acceptance-late:" + _digest(identity)
        for root in dict.fromkeys((canonical, Path(drive_root))):
            previous = (load_task_result(root, tid, strict=True) or {}).get("acceptance_debt") or {}
            if previous.get("debt_id") == debt_id:
                # Repeated tails never recapture newer evidence or repair a gap.
                return {"acceptance_debt": previous}
        gaps = [{"section": "workspace", "reason": "not_a_filesystem_snapshot"},
                {"section": "external_review_inputs", "reason": "unretained_external_sources_unavailable",
                 "detail": "Only retained HOST carrier bytes can be read later; uncaptured external inputs remain unavailable."}]
        if not identity["delivery_id"] or identity["chat_id"] is None:
            gaps.append({"section": "delivery", "reason": "final_delivery_identity_unavailable"})
        # These fields are assigned by build_task_acceptance_evidence,
        # process_tool_results/persist_call and build_completion_observations.
        # Their contents may include agent data; only their source carriers own
        # references. A nested shape, digest or provenance label grants nothing.
        carriers = [("evidence." + key, evidence.get(key)) for key in _EVIDENCE_SOURCE_REF_FIELDS]
        carriers.append(("observations.source_ref", observations.get("source_ref")))
        historical_inputs = evidence.get("historical_author_inputs") or {}
        carriers.extend((f"historical_author_inputs.anchors[{index}].source_ref", anchor.get("source_ref"))
                        for index, anchor in enumerate(historical_inputs.get("anchors", []))
                        if index < 2 and isinstance(anchor, dict))
        for index, call in enumerate(trace.get("tool_calls") or []):
            if not isinstance(call, dict):
                continue
            location = f"trace.tool_calls[{index}]."
            carriers.extend((location + key, call.get(key)) for key in TOOL_SOURCE_REF_FIELDS)
            call_refs = call.get("trace_ref")
            if isinstance(call_refs, dict):
                carriers.extend((location + "trace_ref." + key, call_refs.get(key))
                                for key in CALL_SOURCE_REF_FIELDS)
        sources = [{"location": location, "source_ref": _copy(ref)} for location, ref in carriers
                   if isinstance(ref, dict) and ref]
        manifests = [{key: item[key] for key in ("name", "kind", "size", "sha256", "immutable") if key in item}
                     for item in artifacts if isinstance(item, dict)]
        if manifests:
            gaps.append({"section": "artifact_manifests", "reason": "manifest_is_not_historical_bytes",
                         "count": len(manifests)})
        gaps.append({"section": "bulk_evidence", "reason": "only_already_retained_sources",
                     "detail": "Inline trace/evidence not in an immutable source is not recaptured here; no current-file reconstruction."})
        candidate = trace.get("delivery_candidate") or {}
        record = _copy({
            "schema_version": 1, "kind": "acceptance_historical_subject", "debt_id": debt_id,
            **seed, "delivery": identity, "captured_at": utc_now_iso(), "answer": answer,
            "result_text": text,
            "effective_criteria": candidate.get("effective_criteria", seed["effective_criteria"]),
            "task_contract": task.get("task_contract"),
            "owner_corpus": evidence.get("task_inputs"),
            "historical_author_inputs": historical_inputs,
            "sources": sources, "artifact_manifests": manifests,
            "trajectory_cutoff": {"tool_calls": len(trace.get("tool_calls") or []),
                                  "reasoning_notes": len(trace.get("reasoning_notes") or [])},
            "candidate_sha256": candidate.get("content_sha256"), "gaps": gaps,
        })
        if not evidence.get("task_inputs"):
            record["gaps"].append({"section": "owner_corpus", "reason": "not_captured"})
        if candidate.get("content_sha256") and candidate["content_sha256"] != identity["text_sha256"]:
            record["gaps"].append({"section": "candidate", "reason": "final_text_differs_from_candidate"})
        raw = json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2).encode()
        # Only this compact boundary record is written/read back. Bulk referenced
        # bytes remain with the existing lifetime owner until ordinary copyback.
        ref = store_actor_source_bytes(drive_root, tid, category="context_checkpoints",
                                      source_id="acceptance_historical", data=raw, extension="json")
        read_actor_source_bytes(drive_root, tid, ref)
        return {"acceptance_debt": {"schema_version": 1, "debt_id": debt_id, **seed,
                "delivery": identity, "source_ref": ref, "delivery_status": "unconfirmed",
                }}
    except Exception as exc:
        # Source preparation must never delay an urgent terminal or claim success
        # after failure. The original bypass remains independently visible.
        log.warning("Acceptance historical source unavailable: %s", exc)
        return {"acceptance_history_gap": {"reason": str(exc), "source_status": "unavailable"}}


def read_acceptance_history(root: Any, task_id: str, debt: dict) -> dict:
    """The tool's decision input is the frozen record, never today's result fields."""
    from ouroboros.artifacts import read_actor_source_bytes

    record = json.loads(read_actor_source_bytes(root, task_id, debt.get("source_ref")))
    if (record.get("schema_version") != 1 or debt.get("schema_version") != 1
            or record.get("kind") != "acceptance_historical_subject" or record.get("task_id") != task_id
            or debt.get("task_id") != task_id or record.get("task_attempt") != debt.get("task_attempt")
            or record.get("debt_id") != debt.get("debt_id") or record.get("delivery") != debt.get("delivery")
            or record["delivery"].get("task_id") != task_id
            or record["delivery"].get("task_attempt") != debt.get("task_attempt")
            or hashlib.sha256(record["answer"].encode()).hexdigest() != record["delivery"]["text_sha256"]):
        raise ValueError("historical_subject_identity_mismatch")
    return record


def preserve_acceptance_history(current: dict, incoming: dict) -> dict:
    """A same-answer replica cannot replace the pin or canonical amendments.

    A different, explicitly newer attempt is a different final subject; it does
    not inherit the old amendment. Mere mutable result text establishes nothing.
    """
    for key in ("acceptance_original_root_cap", "acceptance_root_cap_amendments"):
        if key in current:
            incoming = {**incoming, key: current[key]}
    old, new = current.get("acceptance_debt"), incoming.get("acceptance_debt")
    if not isinstance(old, dict) or "acceptance_debt" not in incoming:
        return incoming
    if not isinstance(new, dict):
        return {**incoming, "acceptance_debt": old}
    old_attempt, new_attempt = old.get("task_attempt"), new.get("task_attempt")
    if (new.get("debt_id") == old.get("debt_id")
            or not (type(old_attempt) is int and type(new_attempt) is int and new_attempt > old_attempt)):
        incoming = {**incoming, "acceptance_debt": old}
    return incoming


def historical_receipt_matches(delivery: dict, receipt: dict) -> bool:
    """One exact physical answer at its final routed destination, after send."""
    return bool(delivery.get("delivery_id") and delivery.get("chat_id") is not None
                and receipt.get("basis") == "send_handler_returned" and receipt.get("source_ref")
                and all(receipt.get(key) == delivery.get(key)
                        for key in ("task_id", "delivery_id", "chat_id", "text_sha256")))


def historical_review_controls(root: Any, row: dict, debt: dict, *, caller_task_id: str,
                               automatic: bool, historical_contract: dict | None = None,
                               check_paid: bool = True, check_pause: bool = True, pending_panel: dict | None = None) -> list[str]:
    """Read current controls for preparation; this is not dispatch admission.

    The operation owner must recheck controls, money and exact delivery at its
    actual dispatch boundary. No technical solve ceiling is reused as a ban.
    """
    from ouroboros.budget_pause import dispatch_fenced
    from ouroboros.cancel_intents import cancel_pending
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.deadline_utils import parse_deadline_ts, utc_now
    from ouroboros.owner_mailbox import drain_owner_entries, KIND_HURRY, KIND_FINALIZE_NOW
    from ouroboros.task_results import load_task_result

    blocked = []
    root = Path(root)
    ids = dict.fromkeys((row["task_id"], debt["accounting_root_task_id"], caller_task_id))
    deadlines = [("", (historical_contract if historical_contract is not None
                       else row.get("task_contract") or {}).get("deadline_at"))]
    try:
        for name in ("panic_stop.flag", "owner_restart_no_resume.flag"):
            if (root / "state" / name).exists():
                blocked.append(name)
        for tid in ids:
            if not tid:
                continue
            current = load_task_result(root, tid, strict=True) or {}
            if not current:
                blocked.append("control_authority_unavailable:" + tid)
            deadlines.extend((":" + tid, raw) for raw in (
                current.get("deadline_at"), (current.get("task_contract") or {}).get("deadline_at")))
            constraint = normalize_task_constraint(current.get("task_constraint"))
            if constraint is not None and not constraint.allow_review:
                blocked.append("review_prohibited:" + tid)
            if cancel_pending(root, tid, strict=True) or current.get("status") in {"cancelled", "cancel_requested"}:
                blocked.append("stop:" + tid)
            if automatic and (current.get("owner_hurry") or {}).get("reason") == "owner_hurry":
                blocked.append("owner_finalization:" + tid)
            if check_pause and (dispatch_fenced(tid) or (current.get("budget_pause") or {}).get("state") in {"pausing", "paused"}):
                blocked.append("pause:" + tid)
        path = root / "state" / "queue_snapshot.json"
        if path.exists():
            snapshot = json.loads(path.read_text())
            fences = snapshot.get("budget_root_fences", [])
            if not isinstance(fences, list) or any(not isinstance(f, dict) for f in fences):
                raise ValueError("invalid_budget_fences")
            for fence in fences if check_pause else ():
                if fence.get("root_task_id") != debt["accounting_root_task_id"] or fence.get("status") not in {"active", "paused"}:
                    continue
                if fence.get("cause") == "owner_pause":
                    from ouroboros.owner_pause import read_fence
                    current = read_fence(root, debt["accounting_root_task_id"])
                    # Resume opens durable authority before its queue projection is
                    # persisted. That stale latch cannot cancel newly resumed work.
                    if current.get("state") == "released" and current.get("fence_id") == fence.get("fence_id"):
                        continue
                blocked.append("root_budget_fence")
        from ouroboros.task_pacing import BudgetSnapshot, review_launch_allowed
        for suffix, raw in deadlines:
            deadline = parse_deadline_ts(raw)
            if raw and deadline is None:
                blocked.append("deadline_unknown" + suffix)
            elif deadline is not None:
                remaining = (deadline - utc_now()).total_seconds()
                if remaining <= 0:
                    blocked.append("owner_deadline" + suffix)
                elif automatic:
                    # Delivery is complete: no author-finalization reserve remains.
                    allowed, reason = review_launch_allowed(BudgetSnapshot(True, remaining_sec=remaining))
                    if not allowed:
                        blocked.append(reason)
        if row.get("status") not in {"completed", "failed"}:
            blocked.append("not_terminal")
        if automatic:
            from ouroboros.config import get_task_review_mode
            if get_task_review_mode() == "off":
                blocked.append("review_mode_off")
            if debt.get("cause") in {"owner_hurry", "acceptance_bypassed_owner_requested_finalization"}:
                blocked.append("owner_finalization")
            status = {}
            entries = drain_owner_entries(root, row["task_id"], include_acknowledged=True, _read_status=status)
            if not status.get("complete"):
                blocked.append("owner_controls_unknown")
            if any(e.get("kind") in {KIND_HURRY, KIND_FINALIZE_NOW} for e in entries):
                blocked.append("owner_finalization")
        accounting = load_task_result(root, debt["accounting_root_task_id"], strict=True)
        if check_paid and (accounting is None or not _zero_physical({}, accounting, pending_panel=pending_panel)):
            blocked.append("paid_or_unknown_panel")
    except Exception:
        blocked.append("control_authority_unavailable")
    return list(dict.fromkeys(blocked))


def prepare_owner_historical_review(ctx: Any, request: dict) -> dict:
    """Resolve the explicit owner request and optionally record an absolute cap.

    Main interprets the resolved words and names the selected task/debt/action/
    amount with a rationale. Source hashes alone, implicit origin inheritance,
    and task/Presence messages grant nothing. Preparation makes no paid claim.
    """
    from ouroboros.loop_acceptance_review import _resolve_ctx_lineage
    from ouroboros.owner_source import resolve_owner_source
    from ouroboros.presence_authority import presence_ceiling_from_context
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.task_results import load_task_result, validate_task_id, write_task_result
    from ouroboros.tool_access import active_tool_profile, canonical_data_root
    from ouroboros.utils import utc_now_iso

    refusal = lambda reason: {"status": "refused", "reason": reason, "dispatched": False}
    try:
        if (active_tool_profile(ctx) not in {"self_modification", "workspace_task", "external_workspace_task", "operator_control"}
                or not _resolve_ctx_lineage(ctx, str(ctx.task_id or ""))["is_root_task"]
                or presence_ceiling_from_context(ctx) is not None):
            return refusal("owner_caller_required")
        if not isinstance(request, dict) or not str(request.get("rationale") or "").strip():
            return refusal("explicit_owner_request_required")
        # Malformed intent is refused before any source or money write. A request
        # without the selector predates it: a named cap alone never buys review.
        action, amount = request.get("action"), request.get("new_original_root_cap_usd")
        if "action" not in request:
            action = "review" if amount is None else "amend_cap"
        if action not in ("review", "amend_cap"):
            return refusal("late_review_action_invalid")
        if action == "amend_cap" and amount is None:
            return refusal("absolute_root_cap_required")
        if amount is not None and (type(amount) not in (int, float) or not math.isfinite(amount) or amount <= 0):
            return refusal("absolute_root_cap_must_be_positive_finite")
        root = canonical_data_root(ctx)
        tid = validate_task_id(request.get("task_id"))
        row = load_task_result(root, tid, strict=True) or {}
        debt = row.get("acceptance_debt") or {}
        if not debt.get("debt_id") or debt["debt_id"] != request.get("debt_id"):
            return refusal("historical_debt_not_found")
        # The named target may belong to another Project. Source membership is
        # in the CALLER's conversation (or its host-bound relayed owner origin),
        # not the target's room. Never default to inherited origin authority.
        source = resolve_owner_source(ctx, request.get("owner_source"))
        if not source:
            return refusal("owner_source_unavailable")
        record = read_acceptance_history(root, tid, debt)
        # An inherited original request is evidence of the old operation, not
        # authority for a NEW operation after that answer was finalized.
        source_ts, frozen_ts = parse_deadline_ts(source.get("ts")), parse_deadline_ts(record.get("captured_at"))
        if source_ts is None or frozen_ts is None or source_ts <= frozen_ts:
            return refusal("new_owner_request_required")
        blocked = historical_review_controls(root, row, debt, caller_task_id=ctx.task_id, automatic=False,
                                             historical_contract=record.get("task_contract") or {})
        from ouroboros.artifacts import store_actor_source_bytes
        authority_ref = store_actor_source_bytes(root, tid, category="context_checkpoints",
            source_id="historical_owner_request", extension="json",
            data=json.dumps({"source": source, "request": request}, ensure_ascii=False, sort_keys=True, indent=2).encode())
        amendment = None
        accounting_id = debt["accounting_root_task_id"]
        accounting = load_task_result(root, accounting_id, strict=True)
        if amount is not None:
            amendment = {"source": source, "source_identity": _digest(source), "new_cap_usd": amount,
                         "accounting_root_task_id": debt["accounting_root_task_id"], "debt_id": debt["debt_id"],
                         "rationale": str(request["rationale"]), "source_ref": authority_ref,
                         "recorded_at": utc_now_iso()}

            def amend(current: dict, fields: dict) -> dict:
                nonlocal amendment, accounting
                latest = (current if accounting_id == tid else load_task_result(root, tid, strict=True) or {}).get("acceptance_debt") or {}
                if latest.get("debt_id") != debt["debt_id"] or latest.get("source_ref") != debt.get("source_ref"):
                    raise ValueError("historical_subject_changed")
                amendments = list(current.get("acceptance_root_cap_amendments") or [])
                prior = next((a for a in amendments if a["source_identity"] == amendment["source_identity"]), None)
                if prior:
                    if prior["new_cap_usd"] != amount:
                        raise ValueError("owner_source_amount_already_used")
                    if prior["debt_id"] != debt["debt_id"]:
                        raise ValueError("owner_source_target_already_used")
                    amendment = prior
                else:
                    cap = effective_original_root_cap(debt, current)
                    if cap["state"] == "finite" and amount < cap["usd"]:
                        raise ValueError("root_cap_amendment_cannot_reduce_cap")
                    amendment["previous_cap"] = cap
                    amendments.append(amendment)
                accounting = {**current, "acceptance_root_cap_amendments": amendments}
                return {"status": current["status"], "acceptance_root_cap_amendments": amendments}

            write_task_result(root, accounting_id, accounting["status"], strict_existing_dict=True, _field_projector=amend)
        return {"status": "prepared", "reason": "owner_historical_source_prepared", "action": action,
                "dispatched": False, "task_id": tid, "debt_id": debt["debt_id"],
                "source_ref": historical_source_reference(root, tid, debt["source_ref"], subject=True),
                "owner_source_ref": historical_source_reference(root, tid, authority_ref),
                "cap_amendment": ({key: amendment[key] for key in (
                    "source_identity", "new_cap_usd", "previous_cap", "accounting_root_task_id", "debt_id", "recorded_at")}
                    if amendment else None),
                "execution_blocked_by": blocked,
                "effective_original_root_cap": effective_original_root_cap(debt, accounting),
                "delivery_status": "unconfirmed"}
    except Exception as exc:
        return refusal(str(exc) if isinstance(exc, ValueError) else "historical_source_or_authority_unavailable")


def historical_source_reference(root: Any, task_id: str, ref: dict, *, subject: bool = False) -> dict:
    """Use the canonical acceptance selector for the pinned subject only.

    Other retained sources keep their existing file-reader authority. Neither
    pointer broadens a caller's filesystem roots.
    """
    if subject:
        from ouroboros.task_finalization import review_source_reader

        return {**ref, "reader": review_source_reader(task_id, ref)}
    from ouroboros.artifacts import task_artifact_dir_path

    return {**ref, "reader": {"tool": "read_file", "arguments": {
        "root": "runtime_data", "path": str(task_artifact_dir_path(root, task_id) / ref["path"]),
        "start_line": 1, "max_lines": 40, "start_char": 0}}}


def effective_original_root_cap(debt: dict, accounting: dict) -> dict:
    amendments = accounting.get("acceptance_root_cap_amendments") or []
    if amendments:
        return {"state": "finite", "usd": amendments[-1]["new_cap_usd"], "source": "owner_amendment"}
    return _copy(debt["original_root_cap"])
