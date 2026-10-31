"""Compact semantic views of an already-authorized effective task result."""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePath
from typing import Any, Dict, Mapping

RESULT_VIEWS = ("answer", "summary", "evidence", "full")
_EVIDENCE_ARTIFACT_LIMIT = 10


def validate_authority_selection(fields: Any, source_only: Any, digest: Any) -> None:
    """New explicit selectors never change the legacy None/omitted path."""
    if fields is not None and (not isinstance(fields, list)
            or any(not isinstance(key, str) for key in fields) or len(set(fields)) != len(fields)):
        raise ValueError("authority_fields must be an array of distinct exact top-level field names")
    if type(source_only) is not bool:
        raise ValueError("source_only must be a boolean")
    if not isinstance(digest, str) or digest and (len(digest) != 64 or digest.strip("0123456789abcdef")):
        raise ValueError("authority_projection_sha256 must be a lowercase 64-character projection digest")


def validate_result_selectors(selectors: Mapping[str, Any]) -> tuple[str, bool]:
    """Validate the handler's named source arguments; read no result or context.

    The mapping is its argument frame, not a second serialized contract. Selector
    compatibility lives beside field projection instead of growing the reader.
    """
    import re

    fields, source_only = selectors["authority_fields"], selectors["source_only"]
    digest, include = selectors["authority_projection_sha256"], selectors["include_authority"]
    validate_authority_selection(fields, source_only, digest)
    selected = fields is not None or bool(digest)
    call_id, manifest = selectors["call_id"], selectors["manifest_sha256"]
    sources = any(selectors[key] for key in (
        "include_work_order_source", "include_completion_source", "include_focus_source",
        "focus_source_sha256", "review_source_sha256", "delegated_output_sha256", "presence_reentry_sha256",
    )) or selectors["presence_reentry_offset"] is not None
    other_source = bool(sources or call_id or manifest)
    if selected and other_source:
        raise ValueError("authority_fields/authority_projection_sha256 cannot be combined with another source selector")
    if digest and fields is None and not (source_only and include):
        raise ValueError("authority_projection_sha256 requires authority_fields or source_only with include_authority")
    if source_only and (not (fields is not None or include or other_source)
            or selectors["focus_source_sha256"] and not selectors["include_focus_source"]
            or selectors["presence_reentry_offset"] is not None and not selectors["presence_reentry_sha256"]):
        raise ValueError("source_only requires an authority or complete source selector")
    if call_id or manifest:
        if (not isinstance(call_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", call_id)
                or call_id in {".", ".."} or not isinstance(manifest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", manifest)):
            raise ValueError("call_id and a lowercase 64-character manifest_sha256 are required together")
        if include or sources:
            raise ValueError("call_id cannot be combined with another authority/source selector")
    view = validate_result_view(selectors["view"])
    source_request = (include or other_source or selected or source_only
                      or selectors["source_start_char"] is not None or selectors["source_end_char"] is not None)
    if view not in {"", "full"} and source_request:
        raise ValueError(f"view={view} is incompatible with authority/source selectors; use view=full or omit view")
    return view, other_source


def authority_selection_response(ctx: Any, data: Dict[str, Any], drive_root: Any,
                                 selectors: Mapping[str, Any]) -> str:
    """Render exact fields only after the reader has admitted the selected task."""
    from ouroboros.agent_startup_checks import task_result_authority_projection
    from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

    try:
        payload = authority_selection(task_result_authority_projection(data, drive_root=drive_root),
            str(selectors["task_id"]), selectors["authority_fields"], source_only=selectors["source_only"],
            digest=selectors["authority_projection_sha256"], start=selectors["source_start_char"],
            end=selectors["source_end_char"], presence_scope=str(selectors["presence_scope"] or ""))
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=str(exc)))
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if payload.get("authority_source", {}).get("reason") == "source_range_invalid":
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=text))
    return text


def authority_selection(authority: Dict[str, Any], task_id: str, fields: Any, *, source_only: bool,
                        digest: str, start: Any, end: Any, presence_scope: str = "") -> Dict[str, Any]:
    """Exact mechanical view of the SAME current authority, never a new dossier/store.

    All digests/ranges describe sorted ensure_ascii=False JSON in Unicode code
    points, not bytes. The full projection identity is separate from the chosen
    fields' serialization and child_result_sha256; receipts/plans may change
    while that result identity stays fixed. Re-reads are current, not immutable.
    """
    from ouroboros.artifacts import text_source_range_projection

    def encode(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    full_text = encode(authority)
    projection_sha = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    keys = sorted(authority) if fields is None else fields
    missing = [key for key in keys if key not in authority]
    if missing:
        raise ValueError("unknown authority_fields: " + json.dumps(missing, ensure_ascii=False))
    arguments: Dict[str, Any] = {"task_id": task_id, "source_only": True,
                                 "authority_projection_sha256": projection_sha}
    arguments.update({"include_authority": True} if fields is None else {"authority_fields": list(keys)})
    if presence_scope:
        arguments["presence_scope"] = presence_scope
    source = {"kind": "current_reread_recipe", "immutable": False,
              "projection": "task_result_authority" if fields is None else "task_result_authority_fields",
              "tool": "get_task_result", "arguments": arguments}
    payload: Dict[str, Any] = {"status": "available", "immutable": False,
        "authority_projection_sha256": projection_sha, "selected_fields": list(keys),
        "partial": fields is not None, "source": source}
    if digest and digest != projection_sha:
        return {**payload, "status": "unavailable", "reason": "authority_identity_mismatch",
                "expected_authority_projection_sha256": digest}
    chosen = {key: authority[key] for key in keys}
    text = full_text if fields is None else encode(chosen)
    projection, reason = text_source_range_projection(text, source["projection"], start, end)
    payload["authority_source"] = {**(projection or {}), "immutable": False,
                                  **({"reason": reason} if reason else {})}
    if not source_only and start is None and end is None:
        payload["authority"] = chosen  # exact complete values, never previews
    omitted = []
    selected = set(keys)
    for key, value in authority.items():
        if key in selected:
            continue
        field_text = encode({key: value})
        omitted.append({"field": key, "complete_chars": len(field_text),
            "bytes": len(field_text.encode("utf-8")),
            "complete_sha256": hashlib.sha256(field_text.encode("utf-8")).hexdigest(),
            "source": {**source, "arguments": {**arguments, "authority_fields": [key]}}})
    payload["omitted_fields"] = sorted(omitted, key=lambda row: row["field"])
    return payload


def retained_call_source(ctx, root, data, task_id, call_id, digest, start, end):
    """One redacted source from the already-selected task result, or its dispatch."""
    import re

    from ouroboros.artifacts import text_source_range_projection
    from ouroboros.observability import call_manifest_path, read_blob_ref, read_call_manifest_ref
    from ouroboros.tools.query_data import _read_refusal

    def admit(path):
        if refusal := _read_refusal(ctx, path):
            raise PermissionError(refusal)

    dispatch = bool(re.fullmatch(r"tool_dispatch_[0-9a-f]{32}", call_id))
    if dispatch:
        ref = {"call_id": call_id, "sha256": digest,
               "path": str(call_manifest_path(root, task_id, call_id))}
    else:
        # The selected document may be a retry's: its published calls carry that task's identity.
        published = str(data.get("task_id") or task_id)
        rows = (data.get("trace_refs") or {}).get("tool_call_refs") or []
        ref = next((row["manifest_ref"] for row in rows if isinstance(row, dict)
                    and isinstance(row.get("manifest_ref"), dict)
                    and row["manifest_ref"].get("call_id") == call_id
                    and row["manifest_ref"].get("sha256") == digest), None)
        if ref is not None:
            task_id = published
        else:  # else only the selected task's own timed-out call that settled later (#1376)
            from ouroboros.late_tool_results import late_call_ref

            ref = late_call_ref(root, task_id, call_id, digest, admit=admit)
    manifest = read_call_manifest_ref(
        root, ref, task_id=task_id, admit=admit, max_bytes=1024 * 1024,
        call_types=("tool_dispatch",) if dispatch else ("tool_call", "tool_timeout"),
    )
    projection = manifest.get("redacted_projection_ref")
    if not isinstance(projection, dict) or not projection:
        projection = manifest.get("full_payload_ref") if manifest.get("full_payload_redacted") is True else None
    if not isinstance(projection, dict) or not projection:
        raise FileNotFoundError("retained call has no redacted projection")
    raw = read_blob_ref(root, projection, expected_kind=str(projection.get("kind") or "json"),
                        max_bytes=16 * 1024 * 1024, raw_bytes=True, admit=admit)
    source, reason = text_source_range_projection(
        raw.decode("utf-8"), kind="redacted_dispatch_projection" if dispatch else "redacted_call_projection",
        start_char=start, end_char=end,
    )
    source.update(task_id=task_id, call_id=call_id, manifest_sha256=digest,
                  call_type=manifest["call_type"], raw_payload_used=False)
    if reason:
        source["reason"] = reason
    return source


def validate_result_view(value: Any) -> str:
    view = str(value or "")
    if view and view not in RESULT_VIEWS:
        raise ValueError("view must be answer, summary, evidence, or full")
    return view


def _retry_projection(data: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for hop in data.get("retry_lineage") or []:
        if not isinstance(hop, Mapping):
            continue
        rows.append({key: hop.get(key) for key in (
            "task_id", "status", "original_task_id", "timeout_retry_from", "terminal_origin",
        ) if hop.get(key) not in (None, "")})
    projection: Dict[str, Any] = {
        key: data.get(key) for key in ("original_task_id", "timeout_retry_from", "duplicate_of")
        if data.get(key) not in (None, "")
    }
    if rows:
        projection["lineage"] = rows
    return projection


def _add_dynamic_summary(
    payload: Dict[str, Any], data: Mapping[str, Any], outcome_summary: str,
) -> Dict[str, Any]:
    from ouroboros.cost_projection import cost_projection

    try:
        summary = json.loads(outcome_summary)
    except (TypeError, ValueError):
        summary = {}
    if not isinstance(summary, dict):
        summary = {}
    payload["cost"] = cost_projection(data)
    payload["delegated_custody"] = summary.get("delegated_custody", {})
    payload["retry"] = _retry_projection(data)
    if "execution_observation" in summary:
        payload["execution_observation"] = summary["execution_observation"]
    return summary


def _compact_artifact_facts(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    artifacts = list(value.get("artifacts") or []) if isinstance(value.get("artifacts"), list) else []
    errors = value.get("errors") if isinstance(value.get("errors"), list) else []
    out: Dict[str, Any] = {
        **({"schema_version": value["schema_version"]} if "schema_version" in value else {}),
        "status": str(value.get("status") or "unknown"),
        "artifact_count": len(artifacts),
        "artifacts_omitted": max(0, len(artifacts) - _EVIDENCE_ARTIFACT_LIMIT),
        "error_count": len(errors),
        "artifacts": [],
    }
    for row in artifacts[:_EVIDENCE_ARTIFACT_LIMIT]:
        if not isinstance(row, Mapping):
            out["artifacts"].append({"status": "unprojectable"})
            continue
        path = str(row.get("path") or "")
        name = str(row.get("name") or (PurePath(path).name if path else ""))
        fact = {
            key: row[key] for key in ("kind", "status", "sha256", "size", "bytes")
            if row.get(key) not in (None, "")
        }
        if name:
            fact["name"] = name
        row_errors = row.get("errors") if isinstance(row.get("errors"), list) else []
        if row_errors:
            fact["error_count"] = len(row_errors)
        out["artifacts"].append(fact)
    return out


def _add_evidence_facts(payload: Dict[str, Any], summary: Mapping[str, Any]) -> None:
    ledger = summary.get("verification_ledger")
    if isinstance(ledger, Mapping):
        payload["verification_ledger"] = dict(ledger)
    receipts = summary.get("verification_receipts")
    if isinstance(receipts, list):
        payload["verification_receipts"] = list(receipts)
        if type(summary.get("verification_receipts_omitted")) is int:
            payload["verification_receipts_omitted"] = summary["verification_receipts_omitted"]
    artifact_facts = _compact_artifact_facts(summary.get("artifact_bundle"))
    if artifact_facts:
        payload["artifact_facts"] = artifact_facts


def render_result_view(
    data: Mapping[str, Any], *, task_id: str, view: str, known_result_sha256: Any,
    child_result_sha256: str, outcome_summary: str, capability_delta: Mapping[str, Any],
    effort: Mapping[str, str], presence_scope: str = "", current_stop: Mapping[str, Any] | None = None,
) -> str:
    """Render a short view without projecting the stored envelope itself.

    ``known_result_sha256`` is the child-result identity in every view.  Summary
    and evidence are always rebuilt because cost/custody/retry/authority facts
    can change while the authored answer stays byte-identical; every view keeps
    the reader's ``current_stop`` beside its status.
    """
    from ouroboros.outcomes import normalize_outcome_axes
    from ouroboros.task_finalization import terminal_host_notice_text

    answer = str(data.get("result") or "")
    answer_sha = hashlib.sha256(answer.encode("utf-8")).hexdigest()
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "view": view,
        "task_id": str(data.get("task_id") or data.get("id") or task_id),
        "status": str(data.get("status") or "unknown"),
        "answer_sha256": answer_sha,
        "answer_chars": len(answer),
        "child_result_sha256": child_result_sha256,
        "terminal_origin": str(data.get("terminal_origin") or "unknown"),
        "outcome_axes": normalize_outcome_axes(dict(data)),
        **({"current_stop": dict(current_stop)} if current_stop else {}),
    }
    notice = terminal_host_notice_text(dict(data))
    if notice:
        payload["nonclean_terminal_host_notice"] = notice
    if capability_delta:
        payload["capability_delta"] = dict(capability_delta)

    from ouroboros.task_custody import unread_mail_notice

    if isinstance(data.get("cancel_origin"), dict):
        payload["cancel_origin"] = data["cancel_origin"]
    if data.get("retired_tool_invocations"):
        payload["interrupted_tool_calls"] = {
            "warning": "Local execution ended; external effects remain unknown.",
            "invocations": data["retired_tool_invocations"],
        }
    if unread := unread_mail_notice(data.get("unread_mailbox")):
        payload["unread_mail_notice"] = unread
    if effort:
        payload["effort"] = dict(effort)

    reread_arguments = {"task_id": task_id, "view": "answer"}
    authority_arguments = {"task_id": task_id, "include_authority": True}
    if presence_scope:
        # A Presence grant is argument-scoped.  A recovery recipe that omitted
        # the same scope would not describe a currently authorized projection.
        reread_arguments["presence_scope"] = presence_scope
        authority_arguments["presence_scope"] = presence_scope
    current_answer = {
        "kind": "current_reread_recipe",
        "immutable": False,
        "projection": "answer",
        "tool": "get_task_result",
        "arguments": reread_arguments,
        "answer_sha256": answer_sha,
    }
    if view == "answer":
        if isinstance(known_result_sha256, str) and known_result_sha256 == child_result_sha256:
            payload["result_unchanged"] = True
            payload["result_source"] = current_answer
        else:
            payload["answer"] = answer
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    # Dynamic views deliberately do not take the unchanged-body branch.
    payload["known_result_matches"] = bool(
        isinstance(known_result_sha256, str) and known_result_sha256 == child_result_sha256
    )
    summary = _add_dynamic_summary(payload, data, outcome_summary)
    if view == "summary":
        payload["answer_source"] = current_answer
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    # Evidence layers current authorized recipes on the same dynamic summary;
    # no trace_refs, manifest paths, or copied stored rows.  Both recipes stay
    # within the get_task_result authority already exercised by this call. They
    # promise a current projection, never immutable bytes or EOF completeness.
    _add_evidence_facts(payload, summary)
    payload["projection_refs"] = [
        current_answer,
        {
            "kind": "current_reread_recipe",
            "immutable": False,
            "projection": "authority",
            "tool": "get_task_result",
            "arguments": authority_arguments,
            "coverage": "current_authorized_projection_only",
        },
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


__all__ = ["RESULT_VIEWS", "render_result_view", "validate_result_view"]
