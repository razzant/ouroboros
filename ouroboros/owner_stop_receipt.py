"""Wrap up's retained-work receipt (#1564), kept in the existing custody envelope.

``capture_receipt`` freezes it once per Wrap up capture from the addressed task and
custody rows; like the ordinary terminal audit, that Stop-path capture reads no saved
output. The owner's full copy (every row and every saved text) is written only when the
owner downloads it from the task's Files (``write_wrap_up_document``). Nothing here controls,
collects, pays for or continues work, and a missing source stays unknown. Bounds apply
only to presentation and every cut is counted.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import re
from typing import Any, Iterator, Mapping, Optional

log = logging.getLogger(__name__)
FIELD = "owner_stop_children"
# The receipt's own typed card row (``events_chat_delivery``): one Stop identity on
# every carrier, so a final answer, a cascade receipt and their replays send it once.
EVENT_FIELD = "terminal_retained_work"
PREVIEW_CHARS = 240
# The Files entry of the owner's copy (web ``result_files.js``); the card row names it.
SAVED_WORK_LABEL = "Work saved at Wrap up"


def result_read(task_id: str) -> dict:
    return {"tool": "get_task_result", "arguments": {
        "task_id": task_id, "include_authority": True}}


def row_id(receipt: Mapping[str, Any]) -> str:
    """``cascade:<root>:<request_id>:retained_work``, from the cascade intent the receipt recorded."""
    stop = receipt.get("stop") if isinstance(receipt.get("stop"), Mapping) else {}
    if stop.get("task_id") and stop.get("request_id"):
        return f"cascade:{stop['task_id']}:{stop['request_id']}:retained_work"
    return ""


def resolve_receipt_target(root: Any, task_id: str, stop: tuple = ()) -> tuple[str, list[str], set[str]]:
    """Return ``(physical_task_id, predecessor_root_ids, root_attempt_ids)`` for Wrap up's receipt."""
    from ouroboros.cancel_intents import _validated_single_cancel_target
    from ouroboros.task_results import load_task_result

    raw_tid, stop_tid = str(task_id or "").strip(), str(stop[0] if len(stop) == 2 and stop[0] else "").strip()
    # A failed lineage read is not evidence that the raw id is the physical owner.
    # Callers retain their existing unknown-receipt / owed-delivery paths.
    physical_id = _validated_single_cancel_target(root, raw_tid) if raw_tid else ""
    logical_id = stop_tid or raw_tid
    if logical_id == physical_id and physical_id:
        cand = str((load_task_result(root, physical_id) or {}).get("root_task_id") or "").strip()
        if cand and cand != physical_id and _validated_single_cancel_target(root, cand) == physical_id:
            logical_id = cand
    predecessors: list[str] = []
    if logical_id and physical_id and logical_id != physical_id:
        if _validated_single_cancel_target(root, logical_id) == physical_id:
            cur, seen = logical_id, set()
            while cur and cur != physical_id and cur not in seen:
                seen.add(cur)
                predecessors.append(cur)
                row = load_task_result(root, cur) or {}
                cur = str(row.get("superseded_by") or row.get("retry_task_id") or "").strip()
    return physical_id, predecessors, {t for t in (raw_tid, stop_tid, logical_id, physical_id, *predecessors) if t}


def _same_stop(old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    if not isinstance(old, Mapping) or not isinstance(new, Mapping):
        return False
    old_s = old.get("stop") if isinstance(old.get("stop"), Mapping) else {}
    new_s = new.get("stop") if isinstance(new.get("stop"), Mapping) else {}
    old_tid = str(old_s.get("task_id") or old.get("task_id") or "")
    new_tid = str(new_s.get("task_id") or new.get("task_id") or "")
    if old_tid and new_tid and old_tid != new_tid:
        return False
    old_req, new_req = str(old_s.get("request_id") or ""), str(new_s.get("request_id") or "")
    return old_req == new_req if (old_req and new_req) else True


def has_wrap_up_receipt(root: Any, task_id: str, stop: tuple = ()) -> bool:
    """Whether ``task_id`` (or its physical retry leaf) already stores a receipt for ``stop``."""
    from ouroboros.task_results import load_task_result

    probe = {"task_id": str(task_id or ""),
             **({"stop": {"task_id": stop[0], "request_id": stop[1]}} if len(stop) == 2 and all(stop) else {})}
    physical_id, _, _ = resolve_receipt_target(root, task_id, stop)
    for tid in (physical_id, str(task_id or "").strip()):
        try:
            audit = (load_task_result(root, tid) or {}).get("delegate_terminal_reconciliation") if tid else None
        except Exception:
            audit = None
        receipt = audit.get(FIELD) if isinstance(audit, Mapping) else None
        if isinstance(receipt, Mapping) and _same_stop(receipt, probe):
            return True
    return False


def _observed_text(detail: Mapping[str, Any]) -> tuple:
    """The engine report's own text, its completeness by the report's own facts, and its kind."""
    output = detail.get("primaryOutput")
    if isinstance(output, dict):
        text, size = str(output.get("text") or ""), output.get("bytes")
        cut = output.get("truncated") is True or (
            isinstance(size, int) and not isinstance(size, bool) and size > len(text.encode("utf-8")))
        completeness = "partial" if cut else "full" if output.get("truncated") is False else "unknown"
    else:
        text, completeness = (output if isinstance(output, str) else ""), "unknown"
    if text:
        return text, completeness, "output"
    summary = str(detail.get("finalSummary") or "")
    return summary, "unknown", "summary" if summary else ""


def retained_run_facts(root: Any, row: Any) -> dict:
    """One terminal run's facts: recorded custody plus ONE read of its retained observation.

    Only Wrap up's capture calls this, on the Stop path, so the retained output is never
    read, copied or hashed here: its canonical file's size is compared with the record. A
    different size is unavailable; a missing file is unverified, since a retained placement
    may still hold it. Its text is read only when the owner downloads it (``write_wrap_up_document``)."""
    from ouroboros import delegate_custody as custody
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.delegate_collection import capture_projection
    from ouroboros.delegate_continuation import terminal_continuity
    from ouroboros.delegate_output import engine_locator, staged_source_path

    facts: dict = {"owner_task_id": row.task_id, "invocation_id": row.invocation_id,
        "route_id": row.route_id, "reason": row.terminal_reason, "superseded_by": row.superseded_by,
        "accounting": {
        "settled": bool(row.settled), "ledger_recorded": bool(row.ledger_recorded),
        "cost_usd": row.cost_usd if row.spend_disclosed else None,
        "cost_final": bool(row.spend_disclosed and not row.spend_estimated and not row.cost_partial)},
        "patch": {"captured": bool(row.patch_captured),
                  "disposition": row.patch_disposed or "not_recorded",
                  "apply_pending": bool(row.patch_apply_pending)},
        "output_state": ("retained_complete" if row.output_complete else "retained_partial")
                        if row.output_source else "not_captured",
        "gaps": []}
    try:
        capture = capture_projection(root, row, reader_task_id="wrap_up_receipt")
        if capture:
            # The capture's note is advice for its original owner, not a grant
            # to the root reading this receipt. Keep the actual materials/state.
            facts["patch"]["capture"] = {k: v for k, v in capture.items() if k != "note"}
    except Exception:
        facts["gaps"].append("patch_capture_unavailable")
    if row.terminal_source:
        try:
            actor_tid = str(row.terminal_source.get("task_id") or row.task_id)
            detail = json.loads(read_actor_source_bytes(root, actor_tid, row.terminal_source))
            summary = custody.summary_of(detail)
            if not row.engine_locator:  # SETTLED records its own; an unsettled run has only this read-back
                facts["engine_locator"] = engine_locator(detail)
            continuity = terminal_continuity(row.run_id, summary)
            if continuity:
                facts["engine_continuity"] = {"scope": "engine_capability_only", **continuity}
            output = detail.get("primaryOutput")
            text = _observed_text(detail)[0]
            facts["answer_preview"] = text[:PREVIEW_CHARS]
            facts["answer_preview_omitted_chars"] = max(0, len(text) - PREVIEW_CHARS)
            if not row.output_source and text:
                facts["output_state"] = "observation_only"
            elif (not row.output_source and "primaryOutput" in detail and not text
                  and (output in (None, "") or isinstance(output, dict)
                       and output.get("bytes") == 0 and output.get("truncated") is False)):
                facts["output_state"] = "none_reported"
        except Exception:
            facts["gaps"].append("terminal_observation_unavailable")
    elif not row.output_source:
        facts["output_state"] = "unknown"
        facts["gaps"].append("terminal_observation_not_recorded")
    if row.output_source:
        facts["output_terminal_sha256"] = row.output_terminal_sha
        if row.terminal_source and row.output_terminal_sha != row.terminal_source.get("sha256"):
            facts["output_state"] = "retained_earlier_observation"
            facts["gaps"].append("current_output_not_collected")
        try:
            size = staged_source_path(root, row.output_source).stat().st_size
        except OSError:  # not at its canonical place; the verifying reader also tries retained placements
            facts["gaps"].append("retained_output_unverified")
        except ValueError:
            facts["gaps"].append("retained_output_unavailable")
        else:
            if size != row.output_source.get("size"):
                facts["gaps"].append("retained_output_unavailable")
    return facts


def _enriched(root: Any, rows: Any, state: Mapping[str, Any], *, owner_task_id: str = "") -> list:
    """Published receipt rows plus Wrap up's facts for each run the addressed snapshot holds."""
    enriched = []
    for run in rows if isinstance(rows, list) else []:
        if not isinstance(run, Mapping):
            continue
        entry = state.get(str(run.get("run_id") or ""))
        item = {**run, **retained_run_facts(root, entry)} if entry is not None else dict(run)
        if entry is None:
            item["gaps"] = [*item.get("gaps", []), "custody_snapshot_unavailable"]
        if owner_task_id and not item.get("owner_task_id"):
            item["owner_task_id"] = owner_task_id
        enriched.append(item)
    return enriched


def _merge_run_row(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict:
    """Keep published locators/digests and verified facts when a recapture degrades."""
    from ouroboros.delegate_terminal import _RECEIPT_LOCATOR_KEYS, _keep_published_outputs

    merged = _keep_published_outputs(
        {**new, **{key: old[key] for key in _RECEIPT_LOCATOR_KEYS if key in old and key not in new}}, old)
    # Locator preservation ignores malformed references; retain their diagnostics too.
    for key in ("other_staged_outputs", "other_terminal_observations"):
        kept = list(merged.get(key) or [])
        for source in (old, new):
            value = source.get(key)
            for ref in value if isinstance(value, list) else ([] if value is None else [value]):
                if not (isinstance(ref, Mapping) and _SHA256.fullmatch(str(ref.get("sha256") or ""))):
                    if ref not in kept:
                        kept.append(ref)
        if kept:
            merged[key] = kept
    if "owner_task_id" not in merged and old.get("owner_task_id"):
        merged["owner_task_id"] = old["owner_task_id"]
    gaps = list(merged.get("gaps") if isinstance(merged.get("gaps"), list) else [])
    if "custody_snapshot_unavailable" in gaps:
        # These facts were observed earlier, not reverified by the failed read.
        for key, value in old.items():
            merged.setdefault(key, value)
    if "terminal_observation_unavailable" in gaps:
        for key in ("answer_preview", "answer_preview_omitted_chars", "engine_locator",
                    "engine_continuity", "output_terminal_sha256"):
            if key in old and key not in merged:
                merged[key] = old[key]
        if merged.get("output_state") in (None, "unknown", "not_captured") and old.get("output_state"):
            merged["output_state"] = old["output_state"]
    if "patch_capture_unavailable" in gaps:
        old_cap = (old.get("patch") if isinstance(old.get("patch"), Mapping) else {}).get("capture")
        if isinstance(old_cap, Mapping):
            patch = dict(merged.get("patch") if isinstance(merged.get("patch"), Mapping) else {})
            if "capture" not in patch:
                patch["capture"] = dict(old_cap)
                merged["patch"] = patch
    return merged


def _merge_run_list(old_rows: Any, new_rows: Any, *, degraded: bool = False) -> list:
    prior = {str(r["run_id"]): r for r in _rows(old_rows) if r.get("run_id")}
    if not prior:
        return [dict(r) for r in _rows(new_rows)]
    incoming = {str(r["run_id"]): r for r in _rows(new_rows) if r.get("run_id")}
    # Dict insertion order preserves the old run order, then genuinely new runs.
    merged = {rid: dict(row) for rid, row in prior.items()}
    for rid, new_r in incoming.items():
        merged[rid] = _merge_run_row(prior[rid], new_r) if rid in prior else dict(new_r)
    if degraded:
        for rid in prior.keys() - incoming.keys():
            gaps = list(merged[rid].get("gaps") or [])
            if "terminal_observation_unavailable" not in gaps:
                merged[rid]["gaps"] = [*gaps, "terminal_observation_unavailable"]
    return list(merged.values())


def _merge_child_row(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict:
    merged = dict(new)
    degraded = bool(new.get("gap")) or new.get("result_state") in (None, "unknown")
    if degraded and old.get("result_state") not in (None, "unknown"):
        for key in ("result_preview", "result_preview_omitted_chars", "result_state",
                    "result_sha256", "terminal_host_notice", "terminal_provider_notice"):
            if key in old and (key not in merged or merged.get(key) in ("", 0, "unknown", None)):
                merged[key] = old[key]
        if merged.get("status") in (None, "", "unknown") and old.get("status"):
            merged["status"] = old["status"]
    old_custody = _custody(old)
    new_custody = _custody(new)
    if old_custody or new_custody:
        custody_degraded = not new_custody or new_custody.get("audit_status") != "ok"
        base_custody = dict(new_custody or old_custody)
        if not new_custody and old_custody:
            base_custody["audit_status"] = "failed"
        base_custody["terminal_runs"] = _merge_run_list(
            old_custody.get("terminal_runs"), new_custody.get("terminal_runs"), degraded=custody_degraded)
        merged["custody"] = base_custody
    return merged


def merge_stop_receipt(old: Any, new: Any) -> dict:
    """Preserve verified membership and material refs for the SAME Stop across degraded recaptures."""
    if not isinstance(new, Mapping) or not new:
        return dict(old) if isinstance(old, Mapping) else {}
    if not isinstance(old, Mapping) or not old or not _same_stop(old, new):
        return dict(new)
    merged = dict(new)
    old_stop = old.get("stop") if isinstance(old.get("stop"), Mapping) else {}
    new_stop = new.get("stop") if isinstance(new.get("stop"), Mapping) else {}
    if old_stop and not new_stop:
        merged["stop"] = dict(old_stop)
    merged["runs"] = _merge_run_list(old.get("runs"), new.get("runs"), degraded=bool(new.get("gap")))
    old_children = {str(c["task_id"]): c for c in _rows(old.get("children")) if c.get("task_id")}
    new_children = {str(c["task_id"]): c for c in _rows(new.get("children")) if c.get("task_id")}
    children = []
    missing_child = False
    for tid in sorted(set(old_children) | set(new_children)):
        if tid in old_children and tid in new_children:
            children.append(_merge_child_row(old_children[tid], new_children[tid]))
        elif tid in new_children:
            children.append(dict(new_children[tid]))
        else:
            missing_child = True
            kept = dict(old_children[tid])
            kept["gap"] = str(kept.get("gap") or new.get("gap") or "child_receipt_unavailable")
            children.append(kept)
    merged["children"] = children
    if new.get("gap"):
        merged["gap"] = new["gap"]
    elif missing_child:
        merged["gap"] = str(old.get("gap") or "descendant_enumeration_unavailable")
    else:
        merged.pop("gap", None)
    return merged


def capture_receipt(root: Any, task_id: str, descendants: Mapping[str, str], *, gap: str = "",
                    stop: tuple = ()) -> dict:
    """Freeze all selected descendants in the root's existing custody envelope.

    Each child's own receipt first publishes the exact locators its source_read names.
    Child debt stays nested here and never joins the root's open-run/patch lists.
    ``stop`` is the cascade intent's (task id, request id): the receipt row's identity.
    Each child's stored text keeps only a preview and its digest here, so the owner's
    download can say whether the text it quotes changed after this capture."""
    from ouroboros.delegate_terminal import (
        _merge_current_receipts,
        custody_audit_snapshot,
        record_terminal_reconciliation,
        terminal_custody_disclosure,
    )
    from ouroboros.task_results import load_task_result

    physical_id, predecessors, root_attempts = resolve_receipt_target(root, task_id, stop)

    def audited(tid: str) -> tuple:
        try:  # one addressed read serves the audit and the run facts
            snapshot = custody_audit_snapshot(root, task_id=tid)
        except Exception:  # Do not hide this failed snapshot behind a second, different read.
            return {"task_id": tid, "audit_status": "failed",
                    "unreconciled": ["delegated_run_state_unknown:snapshot_unavailable"]}, {}
        return terminal_custody_disclosure(root, tid, snapshot=snapshot), snapshot.get("state") or {}

    pred_runs = []
    for pred_id in predecessors:
        try:
            pred_stored = load_task_result(root, pred_id) or {}
            pred_audit, pred_state = audited(pred_id)
            pred_custody = _merge_current_receipts(pred_stored, pred_audit, addressed=True)
            pred_runs.extend(_enriched(root, pred_custody.get("terminal_runs"), pred_state, owner_task_id=pred_id))
            if pred_audit.get("audit_status") != "ok" or not pred_stored:
                gap = gap or "predecessor_receipt_unavailable"
            if pred_stored:
                record_terminal_reconciliation(root, pred_id, pred_audit)
        except Exception:
            gap = gap or "predecessor_receipt_unavailable"
    audit, state = audited(physical_id)
    if audit.get("audit_status") != "ok":
        gap = gap or "root_custody_unavailable"
    try:
        stored_root = load_task_result(root, physical_id) or {}
    except Exception:
        stored_root = {}
        gap = gap or "task_result_unavailable"
    merged_root_audit = _merge_current_receipts(stored_root, audit, addressed=True)
    own_runs = _enriched(root, merged_root_audit.get("terminal_runs"), state, owner_task_id=physical_id)
    seen_runs = {str(r.get("run_id") or "") for r in own_runs if isinstance(r, Mapping)}
    runs = [*(r for r in pred_runs if str(r.get("run_id") or "") not in seen_runs), *own_runs]
    children = []
    for tid, status in sorted(descendants.items()):
        if tid in root_attempts:
            continue
        child: dict = {"task_id": tid, "status": status or "unknown", "result_read": result_read(tid)}
        try:
            stored = load_task_result(root, tid) or {}
            child["status"] = str(stored.get("status") or status or "unknown")
            text = str(stored.get("result") or "")
            # Authorship is the recorded terminal origin, never inferred: a stopped task's
            # result is often host text that may quote its last unreviewed message.
            child.update(result_preview=text[:PREVIEW_CHARS],
                         result_preview_omitted_chars=max(0, len(text) - PREVIEW_CHARS),
                         result_state=("unknown" if not stored else "no_text_recorded" if not text else
                                       _TEXT_ORIGINS.get(str(stored.get("terminal_origin") or ""),
                                                         "unattributed_text")))
            if text:
                child["result_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            # API children may have no session custody. Their provider/host
            # failure receipts remain separate from any retained model bytes.
            for field in ("terminal_host_notice", "terminal_provider_notice"):
                if stored.get(field):
                    child[field] = str(stored[field])
            child_audit, child_state = audited(tid)
            # Publish fresh locators before advertising their source_read. This
            # changes no outcome and neither starts nor collects a run.
            if stored:
                record_terminal_reconciliation(root, tid, child_audit)
            published = (load_task_result(root, tid) or {}).get("delegate_terminal_reconciliation")
            custody = (published if isinstance(published, Mapping) and published
                       else _merge_current_receipts(stored, child_audit, addressed=True))
            child["custody"] = {**custody, "terminal_runs": _enriched(
                root, custody.get("terminal_runs"), child_state, owner_task_id=tid)}
            if child_audit.get("terminal_runs") and not published:  # readers advertised, publication unconfirmed
                child["gap"] = "receipt_publication_unconfirmed"
            if not stored:
                child["gap"] = "task_result_unavailable"
        except Exception:
            child["gap"] = "child_receipt_unavailable"
        children.append(child)
    previous = audit.get(FIELD) if isinstance(audit.get(FIELD), Mapping) else {}
    if not previous and isinstance(stored_root.get("delegate_terminal_reconciliation"), Mapping):
        prev_cand = stored_root["delegate_terminal_reconciliation"].get(FIELD)
        if isinstance(prev_cand, Mapping):
            previous = prev_cand
    if (not children and not runs and not audit.get("unreconciled")
            and audit.get("audit_status") == "ok" and not gap and not previous):
        return audit
    known = {"task_id": stop[0], "request_id": stop[1]} if len(stop) == 2 and all(stop) else previous.get("stop")
    receipt = merge_stop_receipt(previous, {
        "task_id": physical_id, "children": children, "runs": runs, "full_source": result_read(physical_id),
        **({"gap": gap} if gap else {}), **({"stop": dict(known)} if known else {})})
    audit[FIELD] = receipt
    record_terminal_reconciliation(root, physical_id, audit)
    try:
        published = load_task_result(root, physical_id) or {}
    except Exception:
        published = {}
    if (published.get("delegate_terminal_reconciliation") or {}).get(FIELD) != audit[FIELD]:
        audit[FIELD]["publication_gap"] = "receipt_publication_unconfirmed"
    return audit


def retained_work_fields(result: Mapping[str, Any]) -> dict:
    """The stored receipt as a terminal event's own typed row, in owner words."""
    audit = result.get("delegate_terminal_reconciliation")
    receipt = audit.get(FIELD) if isinstance(audit, Mapping) else None
    text = receipt_text(audit, owner_row=True) if isinstance(receipt, Mapping) else ""
    return {EVENT_FIELD: {"id": row_id(receipt), "text": text}} if text else {}


def terminal_event_fields(root: Any, task_id: str) -> dict:
    """The receipt row for a host-built terminal event (cascade, salvage, completion wins).

    An unreadable result claims Wrap up only when the task's own open Stop says it applies."""
    from ouroboros.task_results import load_task_result
    try:
        physical_id, _, _ = resolve_receipt_target(root, task_id)
        fields = retained_work_fields(load_task_result(root, physical_id) or {})
        if not fields and physical_id != task_id:
            fields = retained_work_fields(load_task_result(root, task_id) or {})
        return fields
    except Exception:
        log.warning("Wrap up receipt unavailable for %s", task_id, exc_info=True)
    try:
        from ouroboros.cancel_intents import (
            SCOPE_CASCADE, STOP_POLICY_FINALIZE, active_intent, resolve_owner_stop_intent, stop_policy,
        )

        intent_tid, intent = resolve_owner_stop_intent(root, task_id)
        if not intent:
            intent_tid, intent = task_id, active_intent(root, task_id) or {}
        if ((stop_policy(intent) != STOP_POLICY_FINALIZE and not intent.get("hardened_at"))
                or intent.get("scope") != SCOPE_CASCADE):
            return {}
    except Exception:
        return {}
    receipt = {"stop": {"task_id": intent_tid or task_id, "request_id": intent.get("request_id")}}
    return {EVENT_FIELD: {"id": row_id(receipt),
                          "text": "Work saved at Wrap up could not be read; what was kept is unknown."}}


def _rows(value: Any) -> list:
    """Only the mapping rows of a stored list: a malformed receipt renders less, never raises."""
    return [row for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def _custody(child: Mapping[str, Any]) -> Mapping[str, Any]:
    return child["custody"] if isinstance(child.get("custody"), Mapping) else {}


def _all_runs(receipt: Mapping[str, Any]) -> list:
    return [*_rows(receipt.get("runs")), *(run for child in _rows(receipt.get("children"))
                                           for run in _rows(_custody(child).get("terminal_runs")))]


def _run_line(row: dict) -> str:
    """The model/CLI fact line: exact recorded codes, nothing inferred."""
    patch = row.get("patch") or {}
    capture = patch.get("capture") or {}
    money = row.get("accounting") or {}
    amount = money.get("cost_usd")
    output = {"retained_complete": "saved full payload", "retained_partial": "saved partial payload",
              "observation_only": "saved terminal observation only", "none_reported": "none reported",
              "not_captured": "not captured", "retained_earlier_observation": "saved earlier payload"}.get(
                  row.get("output_state"), "unknown")
    decided = "disposition " + str(patch.get("disposition") or "unknown") + (
        ", result lost" + (f", {_kept(capture)}" if _kept(capture) else "") if capture.get("result_lost") else "") + (
        f", disposed by {capture['disposed_by']}" if capture.get("disposed_by") else "")
    parts = [f"run {row.get('run_id')}: {row.get('state') or 'unknown'}",
             f"output {output}",
             "patch " + str(capture.get("status") or ("captured" if patch.get("captured") else "not confirmed")),
             decided,
             "cost unknown" if amount is None else f"cost USD {amount}"]
    if money.get("settled") is False:
        parts.append("accounting not settled")
    if amount is not None and not money.get("cost_final"):
        parts.append("cost not final")
    if capture and not patch.get("captured") and not capture.get("result_lost"):
        parts.append("capture receipt unconfirmed")
    if patch.get("apply_pending"):
        parts.append("apply outcome unknown")
    resumable = (row.get("engine_continuity") or {}).get("resumable")
    if isinstance(resumable, dict) and isinstance(resumable.get("carriers"), list):
        carriers = ", ".join(str(value) for value in resumable["carriers"])
        parts.append("engine carriers (capability only): " + (carriers[:PREVIEW_CHARS] or "none reported")
                     + (f" … ({len(carriers) - PREVIEW_CHARS} characters omitted)"
                        if len(carriers) > PREVIEW_CHARS else ""))
    if isinstance(row.get("gaps"), list) and row["gaps"]:
        parts.append("gaps: " + ", ".join(map(str, row["gaps"])))
    return "; ".join(parts)


def _prompt_run(row: dict, reader_task_id: str = "", publisher_task_id: str = "") -> str:
    """Bound a run's presentation, naming exactly what the full source keeps."""
    from ouroboros.delegate_output import project_output_readers

    publisher = str(row.get("owner_task_id") or publisher_task_id or "")
    own = bool(reader_task_id and publisher == reader_task_id)
    row = project_output_readers(row, publisher)
    facts = {key: row[key] for key in ("staged_output", "other_staged_outputs", "terminal_observation", "other_terminal_observations",
                                     "engine_locator", "engine_continuity", "patch") if key in row}
    text = _run_line(row)
    if row.get("answer_preview"):
        text += "\n  Output excerpt (the run's own words): " + row["answer_preview"]
        if row.get("answer_preview_omitted_chars"):
            text += f" … ({row['answer_preview_omitted_chars']} preview characters omitted)"
    # The actionable reader precedes the potentially large fact block, so the
    # display bound cannot leave only an inaccessible staging-file recipe.
    for key in ("staged_output", "terminal_observation", "other_staged_outputs", "other_terminal_observations"):
        refs = row.get(key)
        for ref in refs if isinstance(refs, list) else [refs]:
            reader = ((ref.get("direct_read") if own else None) or ref.get("source_read")
                      if isinstance(ref, dict) else None)
            if isinstance(reader, dict):
                label = ("Read retained source to EOF with read_file: " if reader.get("tool") == "read_file" else
                         "Read retained source (metadata, then source_start_char/source_end_char): ")
                text = label + json.dumps(reader) + "\n  " + text
                break
        else:
            continue
        break
    if facts:
        text += "\n  Material facts: " + json.dumps(facts, ensure_ascii=False, sort_keys=True)
    limit = 1800
    return text if len(text) <= limit else text[:limit] + f" … ({len(text) - limit} receipt characters omitted)"


def _bounded(text: str) -> str:
    omitted = max(0, len(text) - PREVIEW_CHARS)
    return text[:PREVIEW_CHARS] + (f" … ({omitted} characters omitted)" if omitted else "")


def receipt_text(audit: Mapping[str, Any], *, prompt: bool = False, owner_row: bool = False,
                 max_rows: int = 20) -> str:
    """One bounded view: the forced final input (``prompt``), the owner's card row
    (``owner_row``), or the model/CLI custody text beside the root's other custody lines."""
    receipt = audit.get(FIELD)
    if not isinstance(receipt, Mapping):
        return ""
    if owner_row:
        return _owner_text(receipt)
    rows: list[str] = []
    if receipt.get("gap"):
        rows.append("Descendant membership unknown: " + receipt["gap"])
    if prompt:
        # The forced input carries no other custody text; elsewhere these lines precede the receipt.
        if audit.get("audit_status") != "ok":
            rows.append("Root delegated custody could not be verified.")
        rows.extend(f"{receipt.get('task_id')}: {label} {item}" for label, item in _unconfirmed(audit))
    for run in _rows(receipt.get("runs")):
        rows.append(_prompt_run(run, str(receipt.get("task_id") or ""), str(receipt.get("task_id") or ""))
                    if prompt else _run_line(run))
    for child in _rows(receipt.get("children")):
        tid = child.get("task_id")
        row = f"{tid} ({child.get('status') or 'unknown'}): "
        row += (_CHILD_EXCERPTS.get(child.get("result_state"), _CHILD_EXCERPTS["unattributed_text"]) + ": "
                + child["result_preview"] if child.get("result_preview") else str(child.get("result_state") or "unknown"))
        if child.get("result_preview_omitted_chars"):
            row += f" … ({child['result_preview_omitted_chars']} characters omitted)"
        if child.get("gap"):
            row += "; " + child["gap"]
        rows.append(row)
        for field in ("terminal_host_notice", "terminal_provider_notice"):
            if notice := child.get(field):
                rows.append(f"{tid}: host receipt: " + _bounded(str(notice)))
        custody = _custody(child)
        if custody.get("audit_status") != "ok":
            rows.append(f"{tid}: delegated custody unknown")
        rows.extend(f"{tid}: {label} {item}" for label, item in _unconfirmed(custody))
        for run in _rows(custody.get("terminal_runs")):
            rows.append(f"{tid} / " + (_prompt_run(run, str(receipt.get("task_id") or ""), str(tid or ""))
                                      if prompt else _run_line(run)))
    if not rows:
        return ""
    title = "[CHILD_RESULTS] Retained work at Wrap up:" if prompt else "Retained work at Wrap up:"
    lines = [title, *("- " + row for row in rows[:max_rows])]
    if len(rows) > max_rows:
        lines.append(f"{len(rows) - max_rows} more receipt rows omitted.")
    if receipt.get("publication_gap"):
        lines.append("Full receipt persistence is unconfirmed; material source access may be unavailable.")
    else:
        lines.append("Full facts and material readers are in task details: get_task_result(task_id="
                     + repr(str(receipt.get("task_id") or "")) + ", include_authority=True), "
                     "delegate_terminal_reconciliation.owner_stop_children (its runs and children).")
        lines.append(f"The owner downloads all rows and saved texts as “{SAVED_WORK_LABEL}” from this task's Files.")
    if any(run.get("engine_continuity") for run in _all_runs(receipt)):
        lines.append("Engine continuation facts describe capability only; they grant no session transfer "
                     "or new turn and do not promise engine retention.")
    return "\n".join(lines)


def _unconfirmed(audit: Mapping[str, Any]) -> list:
    terminal = {run.get("run_id") for run in _rows(audit.get("terminal_runs"))}
    opened, pending = audit.get("open_run_ids"), audit.get("pending_invocation_ids")
    return [*(("execution not confirmed terminal", rid) for rid in (opened if isinstance(opened, list) else [])
              if rid not in terminal),
            *(("pending invocation", item) for item in (pending if isinstance(pending, list) else []))]


# A child's stored text by its recorded terminal origin; an unstamped text has no known author.
_TEXT_ORIGINS = {"model_final": "answer", "host_salvage": "salvaged_text", "host_notice": "host_note"}
_CHILD_EXCERPTS = {"answer": "saved answer excerpt (its own words)",
                   "salvaged_text": "last unreviewed message excerpt (salvaged)",
                   "host_note": "host note excerpt (not its words)",
                   "unattributed_text": "saved text excerpt (author not recorded)"}

# Owner words (DESIGN "Chat authorship and System rows"): states, never codes or model calls.
_TASK_WORDS = {"completed": "Done", "cancelled": "Stopped", "failed": "Failed", "interrupted": "Interrupted",
               "running": "still running", "scheduled": "not started", "requested": "not started"}
_RUN_WORDS = {"succeeded": "finished", "cancelled": "stopped", "failed": "failed", "interrupted": "interrupted"}
_OUTPUT_WORDS = {"retained_complete": "full output saved", "retained_partial": "partial output saved",
                 "retained_earlier_observation": "an earlier output saved, the latest not collected",
                 "observation_only": "only the engine's last report saved", "none_reported": "no output reported",
                 "not_captured": "output not saved"}
_CAPTURE_WORDS = {"ready_with_changes": "file changes captured", "ready_no_changes": "no file changes",
                  "failed": "capturing file changes failed", "missing": "captured file changes are missing",
                  "pending": "file changes not captured yet"}
_DISPOSITION_WORDS = {"applied": "applied", "rejected": "rejected", "superseded": "continued by a later run"}
_LOST_WORDS = "file changes lost: the private copy was gone and nothing was saved"
_GAP_WORDS = {"terminal_observation_unavailable": "the engine's last report could not be read",
              "custody_snapshot_unavailable": "these saved facts could not be rechecked",
              "retained_output_unavailable": "the saved output was not found",
              "retained_output_unverified": "whether the saved output is still readable is unknown",
              "patch_capture_unavailable": "the change capture could not be read"}
_CHILD_WORDS = {"answer": "its answer is saved", "salvaged_text": "its last unreviewed message is saved",
                "host_note": "a host note is saved", "unattributed_text": "a saved text with no recorded author",
                "no_text_recorded": "no text saved"}
_CHILD_TEXT_LABELS = {"answer": "Its answer, in its own words:",
                      "salvaged_text": "Its last unreviewed message, kept by Ouroboros (not a final answer):",
                      "host_note": "A note Ouroboros wrote (not the subtask's words):",
                      "unattributed_text": "Its saved task text; the author is not recorded:"}
_RUN_TEXT_LABELS = {"full": "Full output", "partial": "Partial output",
                    "unknown": "Output (whether it is complete is unknown)"}
# (kind, role) of the ref a run's text was read from.
_RUN_TEXT_SOURCES = {("output", "current"): "its saved output",
                     ("output", "earlier"): "an earlier saved output",
                     ("output", "other"): "another retained saved output",
                     ("report", "current"): "the engine's last report",
                     ("report", "other"): "another retained engine report"}
_KEPT_COPY = "The full copy Ouroboros kept of its last unreviewed message (not a final answer):"
# The host salvage note's own words (``observability.salvaged_output_note``) name its private copy.
_SALVAGE_COPY_AT = re.compile(r"; full copy preserved at [^\n]+?\):\n")


def _kept(capture: Mapping[str, Any]) -> str:
    """The file changes a failed capture saved before a recorded loss (#1651), from its own count."""
    if capture.get("file_output_count") is None:
        return "saved file changes unknown"
    kept = int(capture.get("file_output_count") or 0)
    return f"{kept} saved file change{'s' * (kept != 1)} kept" if kept else ""


def _usd(amount: float) -> str:
    return f"${amount:.4f}" if 0 < amount < 0.01 else f"${amount:.2f}"


def _owner_run(run: Mapping[str, Any], prefix: str) -> str:
    patch = run.get("patch") or {}
    capture = patch.get("capture") or {}
    words = [_OUTPUT_WORDS.get(run.get("output_state"), "output unknown")]
    if capture or patch.get("captured"):
        status = capture.get("status")
        kept = _kept(capture)
        change = (f"file changes lost: the private copy was gone; {kept}" if kept else _LOST_WORDS) if capture.get(
            "result_lost") else _CAPTURE_WORDS.get(status, "file changes captured" if patch.get("captured") else
                                                   "file changes unknown")
        if status == "ready_with_changes" or not capture or capture.get("disposition") in _DISPOSITION_WORDS:
            by = str(capture.get("disposed_by") or "")  # a recorded decision by another task names it
            change += ", " + _DISPOSITION_WORDS.get(patch.get("disposition"), "not applied") + (
                f" by task {by}" if by not in ("", "not_recorded", str(run.get("owner_task_id") or "")) else "")
        words.append(change + (" (whether applying finished is unknown)" if patch.get("apply_pending") else ""))
    money = run.get("accounting") or {}
    amount = money.get("cost_usd")
    known = isinstance(amount, (int, float)) and not isinstance(amount, bool)
    words.append(f"cost {'' if money.get('cost_final') else 'about '}{_usd(amount)}" if known else "cost unknown")
    if money.get("settled") is False:
        words.append("accounting not settled")
    words.extend(_GAP_WORDS[gap] for gap in (run.get("gaps") if isinstance(run.get("gaps"), list) else [])
                 if gap in _GAP_WORDS)
    return f"{prefix} {run.get('run_id')} {_RUN_WORDS.get(run.get('state'), 'ended')}: " + "; ".join(words)


def _owner_rows(receipt: Mapping[str, Any], *, flat: bool = False) -> list:
    """Every (indent, line) of the owner's list: the card row shows flat bullets, the file nested."""
    rows: list[tuple[str, str]] = []
    if receipt.get("gap"):
        words = {"root_custody_unavailable": "This task's delegated runs could not be checked.",
                 "predecessor_receipt_unavailable": "Some earlier root-attempt facts could not be checked.",
                 "task_result_unavailable": "This task's previously saved facts could not be read."}
        rows.append(("", words.get(receipt["gap"],
                    "Which subtasks belonged to this work could not be read; this list may be incomplete.")))
    rows.extend(("", _owner_run(run, "This task's run")) for run in _rows(receipt.get("runs")))
    for child in _rows(receipt.get("children")):
        tid = child.get("task_id")
        custody = _custody(child)
        saved = _CHILD_WORDS.get(child.get("result_state"), "its result is unknown")
        rows.append(("", f"Subtask {tid} ({_TASK_WORDS.get(child.get('status'), 'status unknown')}): "
                         + saved + ("; its saved facts could not all be read" if child.get("gap") else "")))
        indent, prefix = ("", f"Subtask {tid}: ") if flat else ("  ", "")
        rows.extend((indent, f"{prefix}Host note: " + _bounded(" ".join(str(child[field]).split())))
                    for field in ("terminal_host_notice", "terminal_provider_notice") if child.get(field))
        if custody.get("audit_status") != "ok":
            rows.append((indent, f"{prefix}Its delegated runs could not be checked."))
        rows.extend((indent, f"{prefix}Run {item} is not confirmed finished." if label.startswith("execution")
                     else f"{prefix}A requested run ({item}) is not confirmed started.")
                    for label, item in _unconfirmed(custody))
        rows.extend((indent, f"{prefix}{_owner_run(run, 'Run')}") for run in _rows(custody.get("terminal_runs")))
    return rows


def _owner_text(receipt: Mapping[str, Any], max_rows: int = 10) -> str:
    """The owner's card row: what was saved and where it is, never a tool call or a raw code."""
    rows = _owner_rows(receipt, flat=True)
    if not rows:
        return ""
    lines = [f"{SAVED_WORK_LABEL}:", *(f"{indent}- {line}" for indent, line in rows[:max_rows])]
    # The Files entry exists wherever this receipt was stored (``wrap_up_document_view``); no link here,
    # since a relative link in a card row reads as words.
    stored = not receipt.get("publication_gap")
    tail = []  # Flat bullets also work in the compact card Markdown renderer.
    if len(rows) > max_rows:
        tail.append(f"{len(rows) - max_rows} more lines are not shown here"
                    + (f"; “{SAVED_WORK_LABEL}” in this task's Files has them all." if stored else "."))
    tail.append(f"Download “{SAVED_WORK_LABEL}” from this task's Files for the full list and every saved text."
                if stored else "Saving this list could not be confirmed.")
    changes = any(((run.get("patch") or {}).get("capture") or {}).get("status") == "ready_with_changes"
                  for run in _all_runs(receipt))
    if receipt.get("children"):
        tail.append("Each subtask's card shows what it saved"
                    + (", and its Files list captured file changes." if changes else "."))
    elif changes:
        tail.append("Captured file changes are listed under this task's Files.")
    return "\n".join([*lines, *(f"- {line}" for line in tail)])


# -- The owner's full copy: written when downloaded, never on the Stop path -----------------------

def wrap_up_document_view(result: Mapping[str, Any]) -> dict:
    """Task detail's ``wrap_up_document``: the download name and Stop request of the owner's full copy
    (``?wrap_up=<request_id>`` on the task-file route), from the stored receipt alone.
    Nothing is read, sized or hashed before the owner downloads it."""
    from ouroboros.task_results import validate_task_id

    audit = result.get("delegate_terminal_reconciliation") if isinstance(result, Mapping) else None
    receipt = audit.get(FIELD) if isinstance(audit, Mapping) else None
    if not isinstance(receipt, Mapping) or not _owner_rows(receipt):
        return {}
    try:
        name = f"work-saved-at-wrap-up-{validate_task_id(receipt.get('task_id'))}.md"
    except ValueError:
        return {}
    stop = receipt.get("stop") if isinstance(receipt.get("stop"), Mapping) else {}
    return {"name": name, "request_id": str(stop.get("request_id") or "")}


def _has_unknown_saved_text(receipt: Mapping[str, Any]) -> bool:
    return bool(
        receipt.get("gap") or receipt.get("publication_gap")
        or any(c.get("gap") or c.get("result_state") == "unknown"
               or _custody(c).get("audit_status") not in (None, "", "ok")
               for c in _rows(receipt.get("children")))
        or any(r.get("output_state") == "unknown" or (isinstance(r.get("gaps"), list) and r["gaps"])
               for r in _all_runs(receipt))
    )


def write_wrap_up_document(root: Any, result: Mapping[str, Any], sink: Any) -> None:
    """Write the owner's full copy (UTF-8 Markdown) to the binary ``sink``: every row as Wrap up
    recorded it, then each saved text whole, read now, with its author and whether it is
    complete. A text that cannot be read is named where it would be; one text is held at a time."""
    audit = result.get("delegate_terminal_reconciliation") if isinstance(result, Mapping) else None
    receipt = audit.get(FIELD) if isinstance(audit, Mapping) else None
    if not isinstance(receipt, Mapping):
        raise ValueError("this task stores no Wrap up receipt")
    stop = receipt.get("stop") if isinstance(receipt.get("stop"), Mapping) else {}
    stored: dict = {}  # task id -> its stored result, read once per download
    opened = False

    def emit(*lines: str) -> None:
        sink.write(("\n".join(lines) + "\n").encode("utf-8"))

    def section(title: str, blocks: Iterator[tuple]) -> None:
        nonlocal opened
        headed = False
        for label, text, note in blocks:
            if not opened:
                emit("", "## Saved texts")
                opened = True
            if not headed:
                emit("", f"### {title}")
                headed = True
            emit("", label, *(("", _quote(text)) if text else ()), *(("", note) if note else ()))

    emit(f"# {SAVED_WORK_LABEL}", "",
         f"Task {receipt.get('task_id')}" + (f", Stop request {stop['request_id']}" if stop.get("request_id") else "")
         + ". The list is what Wrap up recorded when it stopped this work. The saved texts after it are read "
         "where Ouroboros keeps them each time this file is downloaded: whole, unreviewed, and labelled with "
         "their author and whether they are complete.",
         "", "## What was saved", "", *(f"{indent}- {line}" for indent, line in _owner_rows(receipt)))
    for run in _rows(receipt.get("runs")):
        owner_tid = str(run.get("owner_task_id") or receipt.get("task_id") or "")
        section(f"This task's run {run.get('run_id')}",
                _run_texts(root, owner_tid, run, stored, receipt=receipt))
    for child in _rows(receipt.get("children")):
        tid = str(child.get("task_id") or "")
        section(f"Subtask {tid}", _child_texts(root, child, stored))
        for run in _rows(_custody(child).get("terminal_runs")):
            owner_tid = str(run.get("owner_task_id") or tid)
            section(f"Subtask {tid}, run {run.get('run_id')}",
                    _run_texts(root, owner_tid, run, stored, receipt=receipt))
    if not opened:
        emit("", "Whether this work kept saved text could not be verified."
             if _has_unknown_saved_text(receipt) else "This work kept no saved text.")


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in (text.splitlines() or [""]))


def _stored_result(root: Any, task_id: str, cache: dict) -> Any:
    """One task's stored result, read once per download; None when it cannot be read."""
    from ouroboros.task_results import load_task_result

    if task_id not in cache:
        try:
            cache[task_id] = load_task_result(root, task_id) or None
        except Exception:
            cache[task_id] = None
    return cache[task_id]


_SHA256 = re.compile(r"[0-9a-f]{64}")
# (receipt key, kind, role, a list of refs) for each text a run row can name; None: by output_state.
_RUN_TEXT_KEYS = (("staged_output", "output", None, False),
                  ("other_staged_outputs", "output", "other", True),
                  ("terminal_observation", "report", "current", False),
                  ("other_terminal_observations", "report", "other", True))


def _run_refs(row: Mapping[str, Any]) -> tuple:
    """(refs, malformed) of the texts a receipt row names, saved outputs before engine reports:
    (kind, ref, role) per ref with an exact sha256, and (kind, role, entry) per entry that
    names none (no record, or no exact digest), which is never opened, only disclosed."""
    refs, malformed = [], []
    for key, kind, role, many in _RUN_TEXT_KEYS:
        value = row.get(key)
        if value is None:
            continue
        source_role = ("earlier" if row.get("output_state") == "retained_earlier_observation"
                       else "current") if role is None else role
        if many and not isinstance(value, list):  # a list of refs that is no list names none of them
            malformed.append((kind, source_role, value))
            continue
        for entry in value if many else [value]:
            if isinstance(entry, Mapping) and _SHA256.fullmatch(str(entry.get("sha256") or "")):
                refs.append((kind, entry, source_role))
            else:
                malformed.append((kind, source_role, entry))
    return refs, malformed


def _after_snapshot_suffix(ref: Any, receipt: Optional[Mapping[str, Any]] = None) -> str:
    """Only genuine timestamps on both the receipt and the source prove later collection."""
    from ouroboros.deadline_utils import parse_deadline_ts

    stop = receipt.get("stop") if isinstance(receipt, Mapping) and isinstance(receipt.get("stop"), Mapping) else {}
    snap_raw = (receipt.get("captured_at") if isinstance(receipt, Mapping) else None) or stop.get("captured_at")
    ref_raw = (ref.get("captured_at") or ref.get("spilled_at") or ref.get("at")) if isinstance(ref, Mapping) else None
    snap_ts = parse_deadline_ts(snap_raw) if snap_raw else None
    ref_ts = parse_deadline_ts(ref_raw) if ref_raw else None
    if snap_ts is not None and ref_ts is not None and ref_ts > snap_ts:
        return ", collected after Wrap up"
    return ", not in the Wrap up snapshot"


def _read_gap(exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        return "it was not found where it is kept"
    return ("its bytes no longer match what was recorded" if isinstance(exc, ValueError)
            and "verification" in str(exc) else "it is not readable")


def _source_text(kind: str, data: Any, complete: bool) -> tuple:
    """(text, completeness, what, gap) of one verified source; text and gap may both be set. A
    report that recorded no output is legitimately empty; a record of another shape, or a report
    of output it does not hold, is a gap."""
    shape = ("", "", "", "its record has an unexpected shape")
    if not isinstance(data, dict):
        return shape
    if kind == "report":
        output = data.get("primaryOutput")
        if not isinstance(output, (dict, str, type(None))) or (
                isinstance(output, dict) and not isinstance(output.get("text") or "", str)):
            return shape
        text, completeness, what = _observed_text(data)
        if what == "summary" and not isinstance(data.get("finalSummary"), str):
            return shape
        size = output.get("bytes") if isinstance(output, dict) else None
        claimed = isinstance(output, dict) and (output.get("truncated") is True or (
            isinstance(size, int) and not isinstance(size, bool) and size > 0))
        return text, completeness, what, ("it reports output it does not hold" if claimed and what != "output" else "")
    primary = data.get("primary_output")
    text = primary.get("text") if isinstance(primary, dict) else primary
    if not isinstance(text, (str, type(None))):
        return shape
    if text:
        return text, "full" if complete else "partial", "output", ""
    summary = data.get("final_summary")
    if not isinstance(summary, (str, type(None))):
        return shape
    return (summary, "unknown", "summary", "") if summary else ("", "", "", "")


def _run_texts(root: Any, publisher: str, run: Mapping[str, Any], cache: dict, *,
               receipt: Optional[Mapping[str, Any]] = None) -> Iterator[tuple]:
    """(label, text, note) per text one run keeps. Only a ref its publishing task's own receipt
    publishes NOW is read (``delegate_output.published_output_ref``), verified from its staging
    store or a retained placement; a ref not in the Wrap up snapshot says so (or collected after
    Wrap up when timestamps prove it). An engine report a complete saved output was collected from
    is not repeated; an entry naming no exact ref is never opened and is said, like every source
    that cannot be read."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.delegate_output import published_output_ref

    current = _stored_result(root, publisher, cache)
    audit = current.get("delegate_terminal_reconciliation") if isinstance(current, dict) else None
    latest = next((row for row in _rows(audit.get("terminal_runs") if isinstance(audit, Mapping) else None)
                   if row.get("run_id") == run.get("run_id")), {})
    (frozen, broken), (newer, newer_broken) = _run_refs(run), _run_refs(latest)
    known = {str(ref["sha256"]) for _kind, ref, _role in frozen}
    frozen_current_kinds = {kind for kind, _ref, role in frozen if role == "current"}
    refs = [*((*item, False) for item in frozen),
            *((item[0], item[1], "other" if item[0] in frozen_current_kinds else item[2], True)
              for item in newer if str(item[1]["sha256"]) not in known)]
    said = {json.dumps(entry, sort_keys=True, default=str) for _kind, _role, entry in broken}
    broken = [*((*item, False) for item in broken),
              *((*item, True) for item in newer_broken if json.dumps(item[2], sort_keys=True, default=str) not in said)]
    output = run.get("staged_output") if isinstance(run.get("staged_output"), Mapping) else {}
    whole = str(output.get("sha256") or "") if output.get("complete") is True else ""
    covered = ""  # the report the complete saved output was collected from, once that output was read
    for kind, ref, role, later in sorted(refs, key=lambda item: item[0] == "report"):  # outputs first
        sha = str(ref["sha256"])
        if kind == "report" and sha == covered:
            continue
        source = _RUN_TEXT_SOURCES[kind, role] + (_after_snapshot_suffix(ref, receipt) if later else "")
        if not isinstance(current, dict):
            yield f"The text from {source} could not be read: the task that saved it is not readable.", "", ""
            continue
        own = published_output_ref(current, sha)[1]
        if own is None:
            yield f"The text from {source} is not published by the task that saved it, so it is not read.", "", ""
            continue
        try:
            data = json.loads(read_actor_source_bytes(root, str(own.get("task_id") or ""), {
                key: own.get(key) for key in ("kind", "root", "path", "size", "sha256")}))
        except Exception as exc:
            yield f"The text from {source} could not be read: {_read_gap(exc)}.", "", ""
            continue
        # A saved output's completeness is the publishing receipt's recorded fact.
        text, completeness, what, gap = _source_text(kind, data, own.get("complete") is True)
        if text:
            if sha == whole and what == "output" and not later:
                covered = str(run.get("output_terminal_sha256") or "")
            yield ("The engine's final summary of this run, not its output" if what == "summary" else
                   f"{_RUN_TEXT_LABELS.get(completeness, _RUN_TEXT_LABELS['unknown'])}, in the run's own words"
                   ) + f" (from {source}):", text, ""
        elif kind == "output" and not gap:  # a report that recorded no output adds nothing to the list
            yield f"{source[:1].upper()}{source[1:]} holds no text.", "", ""
        if gap:
            yield f"The text from {source} could not be read: {gap}.", "", ""
    for kind, role, _entry, later in broken:
        source = _RUN_TEXT_SOURCES[kind, role] + (_after_snapshot_suffix(_entry, receipt) if later else "")
        yield f"A record of {source} is malformed, so it is not opened.", "", ""


def _child_texts(root: Any, child: Mapping[str, Any], cache: dict) -> Iterator[tuple]:
    """A subtask's stored text by its recorded origin, the host notes the list cuts, then the full
    copy its salvage owner kept (``observability.preserved_salvage_path``, a pure read: found by
    task id, never by a path a record names, creating nothing). The host's salvage note names that
    copy's private path; here it points at the copy instead."""
    from ouroboros.observability import preserved_salvage_path

    tid = str(child.get("task_id") or "")
    stored = _stored_result(root, tid, cache)
    if stored is None:
        yield "Its stored result could not be read.", "", ""
    text = str((stored or {}).get("result") or "")
    if stored is not None and not text and child.get("result_sha256"):
        yield "Its recorded text is no longer present in the stored result.", "", ""
    origin = _TEXT_ORIGINS.get(str((stored or {}).get("terminal_origin") or ""), "unattributed_text")
    try:
        path = preserved_salvage_path(pathlib.Path(root), tid)
        kept, unread = (pathlib.Path(path).read_text(encoding="utf-8") if path else ""), ""
    except (OSError, ValueError):
        kept, unread = "", "The full copy Ouroboros kept of its last unreviewed message could not be read."
    if text:
        changed = child.get("result_sha256") not in (None, hashlib.sha256(text.encode("utf-8")).hexdigest())
        yield (_CHILD_TEXT_LABELS[origin], text if origin == "answer" else _SALVAGE_COPY_AT.sub(
            "; full copy below):\n" if kept and kept != text else "; full copy not readable here):\n", text),
            "This text changed after Wrap up; it is quoted as stored now." if changed else "")
    for field in ("terminal_host_notice", "terminal_provider_notice"):
        if len(note := str((stored or {}).get(field) or "")) > PREVIEW_CHARS:
            yield _CHILD_TEXT_LABELS["host_note"], note, ""
    if kept and kept != text:
        yield _KEPT_COPY, kept, ""
    elif unread:
        yield unread, "", ""
