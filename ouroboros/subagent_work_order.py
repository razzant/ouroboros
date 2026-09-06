"""One compiler for configured-session work orders and host assignment context."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Mapping

_WORK_ORDER_CHARS = 250_000
# Historical import name retained for tests/callers which only need the public
# wire budget. It is no longer a per-field truncation limit.
_FIELD_CHARS = _WORK_ORDER_CHARS


class WorkOrderBudgetExceeded(ValueError):
    """A complete work order cannot fit the one explicit wire budget."""

    def __init__(self, *, chars: int, sha256_hex: str) -> None:
        super().__init__(f"complete work order is {chars} characters (budget {_WORK_ORDER_CHARS})")
        self.chars = int(chars)
        self.sha256 = str(sha256_hex)
        self.limit = _WORK_ORDER_CHARS


def _source_selector(task: Mapping[str, Any]) -> dict[str, Any]:
    """Return stable, actor-resolvable pointers without copying the omitted brief.

    The selector is deliberately a small manifest.  The complete work order remains
    in the parent task's canonical authority; this prompt must never smuggle a prefix
    of it under a different name and make the child believe that the prefix is the
    contract.  The existing task-result reader exposes the same canonical projection
    the host validates, so the owner can answer a precise range request through the
    existing interaction channel.
    """

    task_id = str(task.get("id") or "")
    # Reuse the canonical actor-readable authority-ref shape. A bespoke nested
    # ``reader`` object looked descriptive but no existing materializer recognized
    # it, which would make the source pointer itself another unverifiable claim.
    selector: dict[str, Any] = {
        "kind": "task_result",
        "task_id": task_id,
        "tool": "get_task_result",
        "arguments": {
            "task_id": task_id,
            "include_authority": True,
            "include_work_order_source": True,
        },
        "projection": "canonical_work_order",
    }
    return selector


def build_work_order_source_request(
    task: Mapping[str, Any], exc: WorkOrderBudgetExceeded,
) -> tuple[str, dict[str, Any]]:
    """Build the bounded interaction lens for a complete brief over the wire cap.

    This is not a lossy work order.  It is an explicit partial-coverage envelope:
    the child receives the full-brief digest and a source it can ask its host to
    resolve, then requests only the exact range it needs.  The response is
    intentionally carried by the existing AskUserQuestion/delegate_answer seam; no
    second storage or retrieval subsystem is introduced here.
    """

    source = _source_selector(task)
    envelope: dict[str, Any] = {
        "schema": 1,
        "kind": "complete_work_order",
        "coverage": "partial",
        "complete_chars": int(exc.chars),
        "wire_budget_chars": int(exc.limit),
        "complete_sha256": str(exc.sha256),
        "source": source,
        "request": {
            "channel": "existing_interaction",
            "before_substantive_work": True,
            "ask_for": "one exact source character range at a time",
            "include": ["complete_sha256", "source", "start_char", "end_char", "reason"],
        },
        "response": {
            "include": [
                "complete_sha256", "source", "start_char", "end_char", "text",
            ],
            "rule": (
                "Answer delegate_answer with source_response={schema, kind, "
                "complete_sha256, source, start_char, end_char, text}. The host "
                "checks the exact canonical range and records only verified ranges. "
                "A missing, mismatched, or partial response remains cannot_verify; "
                "it never authorizes PASS, a destructive rewrite, or replacement "
                "of this complete work order."
            ),
        },
    }
    prompt = (
        "WORK ORDER SOURCE REQUEST\n"
        "The complete external work order exceeded the host wire budget. This message "
        "is an explicit partial-coverage lens, not the assignment and not a prefix of "
        "the assignment. Do not perform substantive work, claim completeness, accept "
        "a verdict, or replace the full contract from this message alone.\n\n"
        "Use the harness's native question channel to ask your host for the smallest "
        "exact source character range needed. Ask the nanny to answer with the "
        "typed source_response field of delegate_answer. After an answer, verify the "
        "returned source selector, range, digest, and completeness before relying on "
        "it; request another range when needed.\n\n"
        + json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2)
    )
    return prompt, envelope


def route_source_request_channel(gateway: Any, route_id: str) -> dict[str, Any]:
    """Read the generic live interaction capability from a Claudexor manifest.

    The route is opaque to Ouroboros.  Missing or failed capability evidence is
    therefore unverified rather than an invitation to start with a partial
    contract.  This mirrors the existing manifest reader used by review execution.
    """

    try:
        rows = gateway.harnesses()
    except Exception as exc:  # noqa: BLE001 - capability is unknown, not healthy
        return {
            "status": "unverified",
            "reason": "capability_read_failed",
            "detail": type(exc).__name__,
            "route": str(route_id or ""),
        }
    for row in rows or []:
        if not isinstance(row, dict) or str(row.get("id") or "") != str(route_id or ""):
            continue
        manifest = row.get("manifest") if isinstance(row.get("manifest"), dict) else {}
        capabilities = (
            manifest.get("capabilities")
            if isinstance(manifest.get("capabilities"), dict) else {}
        )
        if not isinstance(capabilities.get("interactive"), bool):
            return {
                "status": "unverified",
                "reason": "interactive_capability_missing",
                "route": str(route_id or ""),
            }
        return {
            "status": "available" if capabilities["interactive"] else "unavailable",
            "reason": "interactive" if capabilities["interactive"] else "interactive_unsupported",
            "route": str(route_id or ""),
        }
    return {
        "status": "unverified",
        "reason": "route_not_in_manifest",
        "route": str(route_id or ""),
    }


def _text(value: Any) -> str:
    if isinstance(value, list):
        value = "\n".join(f"- {item}" for item in value if str(item).strip())
    return str(value or "").strip()


def assignment_instructions(ctx: Any) -> str:
    """Host-authored complete normalized contract for every direct delegate start."""

    contract = getattr(ctx, "task_contract", None)
    if not isinstance(contract, dict) or not contract:
        meta = getattr(ctx, "task_metadata", {})
        raw = meta.get("task_contract") if isinstance(meta, dict) else None
        contract = raw if isinstance(raw, dict) else {}
    if contract:
        from ouroboros.contracts.task_contract import build_task_contract

        contract = build_task_contract({"task_contract": contract})
    parts: list[str] = []
    objective = _text(contract.get("objective"))
    expected = _text(contract.get("expected_output"))
    if objective:
        parts.append(
            "HOST TASK OBJECTIVE (immutable contract; the prompt is one assignment inside it): "
            + objective
        )
    if expected:
        parts.append("HOST EXPECTED OUTPUT: " + expected)
    if contract:
        parts.append(
            "HOST TASK CONTRACT AUTHORITY (complete normalized JSON; exact strings are authority):\n"
            + json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    return "\n\n".join(parts)


def _render_external_work_order(task: Mapping[str, Any]) -> str:
    contract = task.get("task_contract") if isinstance(task.get("task_contract"), dict) else {}
    assignment_context = "" if task.get("context") is None else str(task.get("context"))
    inherited_context = "" if contract.get("context") is None else str(contract.get("context"))
    context_sections = []
    if assignment_context:
        context_sections.append("DELEGATED ASSIGNMENT CONTEXT\n" + assignment_context)
    if inherited_context and inherited_context != assignment_context:
        context_sections.append("INHERITED CALLER AUTHORITY CONTEXT\n" + inherited_context)
    represented_keys = {
        "objective", "context", "expected_output", "constraints", "acceptance_claims",
        "attachment_manifest",
    }
    remaining_contract = {
        key: value for key, value in contract.items() if key not in represented_keys
    }
    sections: list[tuple[str, Any]] = [
        ("OBJECTIVE", task.get("objective") or contract.get("objective") or task.get("description")),
        ("PARENT CONTEXT / REFERENCES", "\n\n".join(context_sections)),
        ("EXPECTED OUTPUT", task.get("expected_output") or contract.get("expected_output")),
        ("CONSTRAINTS / NON-GOALS", task.get("constraints") or contract.get("constraints")),
        ("ACCEPTANCE CLAIMS", contract.get("acceptance_claims")),
        ("TASK CONTRACT AUTHORITY", remaining_contract),
        ("INHERITED TASK INPUTS", contract.get("attachment_manifest")),
    ]
    authority = {
        "task_id": str(task.get("id") or ""),
        "parent_task_id": str(task.get("parent_task_id") or ""),
        "root_task_id": str(task.get("root_task_id") or ""),
        "workspace_root": str(task.get("workspace_root") or ""),
        "workspace_mode": str(task.get("workspace_mode") or ""),
        "task_constraint": task.get("task_constraint") if isinstance(task.get("task_constraint"), dict) else {},
        "allowed_resources": contract.get("allowed_resources") if isinstance(contract.get("allowed_resources"), dict) else {},
        "deadline_at": str(contract.get("deadline_at") or ""),
        "origin_message_ref": task.get("origin_message_ref") if isinstance(task.get("origin_message_ref"), dict) else {},
    }
    rendered = []
    for title, value in sections:
        body = (
            str(value) if title == "PARENT CONTEXT / REFERENCES" and isinstance(value, str)
            else _text(value)
        )
        if body:
            rendered.append(f"{title}\n{body}")
    rendered.append(
        "HOST AUTHORITY BINDING (facts, not instructions to widen)\n"
        + _text(json.dumps(authority, ensure_ascii=False, sort_keys=True))
    )
    return "\n\n".join(rendered)


def work_order_source_projection(
    task: Mapping[str, Any], start_char: Any = None, end_char: Any = None,
) -> tuple[dict[str, Any] | None, str]:
    """Return one bounded, actor-readable slice of the canonical work order.

    ``get_task_result`` uses this projection, and source validation uses the same
    renderer, so the selector resolves to the exact bytes the host checks.  A caller
    must request both bounds; the reader never silently emits a large complete brief.
    """

    if not task.get("id") and task.get("task_id"):
        task = {**task, "id": task.get("task_id")}
    rendered = _render_external_work_order(task)
    from ouroboros.artifacts import text_source_range_projection

    return text_source_range_projection(rendered, "canonical_work_order", start_char, end_char)


def _source_task_from_context(ctx: Any, task_id: str) -> dict[str, Any]:
    """Use the durable task-result row, with the active context as pre-write fallback."""

    metadata = getattr(ctx, "task_metadata", {})
    task = dict(metadata) if isinstance(metadata, Mapping) else {}
    task["id"] = task_id
    contract = getattr(ctx, "task_contract", {})
    task["task_contract"] = dict(contract) if isinstance(contract, Mapping) else {}
    if not task.get("workspace_root"):
        workspace_root = getattr(ctx, "workspace_root", None)
        if workspace_root:
            task["workspace_root"] = str(workspace_root)
    if not task.get("workspace_mode"):
        workspace_mode = getattr(ctx, "workspace_mode", "")
        if workspace_mode:
            task["workspace_mode"] = str(workspace_mode)
    if "task_constraint" not in task:
        raw_constraint = getattr(ctx, "task_constraint", None)
        if isinstance(raw_constraint, Mapping):
            task["task_constraint"] = dict(raw_constraint)
    try:
        from pathlib import Path
        from ouroboros.task_status import load_effective_task_result

        root = Path(str(getattr(ctx, "budget_drive_root", "") or getattr(ctx, "drive_root", "")))
        stored = load_effective_task_result(root, task_id, materialize_artifacts=False)
    except Exception:
        stored = {}
    if isinstance(stored, Mapping) and stored:
        stored_nonempty = {
            key: value for key, value in stored.items() if value not in (None, "")
        }
        task = {**task, **stored_nonempty, "id": task_id}
        if not isinstance(task.get("task_contract"), Mapping):
            task["task_contract"] = dict(contract) if isinstance(contract, Mapping) else {}
    return task


def canonical_work_order_source(ctx: Any, request: Mapping[str, Any]) -> tuple[str, str]:
    """Materialize the exact complete brief behind one source request.

    The source-range answer is checked against the same deterministic renderer that
    created the over-budget fingerprint.  This is an existing authority read, not a
    second retrieval system: ``ToolContext`` already carries the immutable contract,
    lineage, origin, workspace and raw constraint mapping used at task admission.
    """

    source = request.get("source") if isinstance(request, Mapping) else None
    task_id = str(source.get("task_id") or "").strip() if isinstance(source, Mapping) else ""
    task_id = task_id or str(getattr(ctx, "task_id", "") or "").strip()
    if not task_id or not isinstance(source, Mapping):
        return "", "source_selector_missing"
    try:
        from ouroboros.agent_startup_checks import valid_task_result_authority_source

        arguments = source.get("arguments")
        if (
            not valid_task_result_authority_source(source, task_id)
            or source.get("projection") != "canonical_work_order"
            or not isinstance(arguments, Mapping)
            or arguments.get("include_work_order_source") is not True
        ):
            return "", "source_selector_invalid"
    except Exception:
        return "", "source_selector_unverifiable"
    task = _source_task_from_context(ctx, task_id)
    rendered = _render_external_work_order(task)
    expected_sha = str(request.get("complete_sha256") or "")
    if sha256(rendered.encode("utf-8")).hexdigest() != expected_sha:
        return "", "source_digest_mismatch"
    raw_chars = request.get("complete_chars")
    expected_chars = raw_chars if type(raw_chars) is int else -1
    if expected_chars != len(rendered):
        return "", "source_size_mismatch"
    return rendered, ""


def validate_work_order_source_response(
    ctx: Any, request: Mapping[str, Any], response: Any,
) -> tuple[dict[str, Any] | None, str]:
    """Verify one typed source range against canonical bytes before delivery.

    The model's ``complete`` claim is deliberately ignored.  Only an exact selector,
    digest, bounded interval and byte-for-byte range match earns a durable interval
    receipt; a host that cannot materialize the source answers with a typed refusal.
    """

    if not isinstance(response, Mapping):
        return None, "source_response_missing"
    if isinstance(response.get("schema"), bool) or response.get("schema") != 1 \
            or str(response.get("kind") or "") != "source_response":
        return None, "source_response_shape_invalid"
    request_source = request.get("source") if isinstance(request, Mapping) else None
    if not isinstance(request_source, Mapping) or response.get("source") != request_source:
        return None, "source_selector_mismatch"
    expected_sha = str(request.get("complete_sha256") or "")
    if str(response.get("complete_sha256") or "") != expected_sha:
        return None, "source_digest_mismatch"
    full_text, reason = canonical_work_order_source(ctx, request)
    if reason:
        return None, reason
    start, end = response.get("start_char"), response.get("end_char")
    if type(start) is not int or type(end) is not int:
        return None, "source_range_invalid"
    text = response.get("text")
    if not isinstance(text, str) or start < 0 or end <= start or end > len(full_text):
        return None, "source_range_invalid"
    if text != full_text[start:end]:
        return None, "source_range_mismatch"
    return {
        "start_char": start,
        "end_char": end,
        "complete_sha256": expected_sha,
        "source": dict(request_source),
        "text_chars": len(text),
        "text_sha256": sha256(text.encode("utf-8")).hexdigest(),
    }, ""


def compile_external_work_order(task: Mapping[str, Any]) -> str:
    """Compile one complete brief or refuse instead of sending a false prefix."""

    rendered = _render_external_work_order(task)
    if len(rendered) > _WORK_ORDER_CHARS:
        raise WorkOrderBudgetExceeded(
            chars=len(rendered), sha256_hex=sha256(rendered.encode("utf-8")).hexdigest(),
        )
    return rendered


def start_binding_fingerprints(ctx: Any, prompt: str) -> tuple[str, str]:
    """Digest the exact brief and the existing normalized task authority."""

    from ouroboros.delegate_recovery import authority_fingerprint_from_context

    return (
        sha256(str(prompt).encode("utf-8")).hexdigest(),
        authority_fingerprint_from_context(ctx),
    )


def work_order_fingerprint(task: Mapping[str, Any]) -> str:
    """Digest the complete canonical brief, including an over-budget one."""

    return sha256(_render_external_work_order(task).encode("utf-8")).hexdigest()


__all__ = [
    "WorkOrderBudgetExceeded", "assignment_instructions", "compile_external_work_order",
    "build_work_order_source_request", "canonical_work_order_source",
    "work_order_source_projection",
    "route_source_request_channel", "validate_work_order_source_response",
    "start_binding_fingerprints", "work_order_fingerprint",
]
