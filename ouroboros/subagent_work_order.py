"""One compiler for configured-session work orders and host assignment context."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Mapping


def input_source_selection_receipt(task: Mapping[str, Any]) -> dict[str, Any]:
    """Composition fact retained by existing work-order/model-send sources.

    This selects automatic inputs, never filesystem access or semantic purity.
    The selection persists for the task; first-position/mail ordering is the
    caller's ordinary collaboration procedure, not a second host controller.
    """
    from ouroboros.contracts.task_contract import task_input_sources

    if task_input_sources(task) != "declared":
        return {}
    actor = task.get("configured_subagent") or (task.get("metadata") or {}).get("configured_subagent") or {}
    external = isinstance(actor, Mapping) and (actor.get("route") or {}).get("kind") == "agent_session"
    governance = ("Complete normalized task authority and existing host instructions; SYSTEM/BIBLE govern the "
                  "supervising Ouroboros and are not claimed as native-session input bytes" if external else
                  "SYSTEM.md and BIBLE.md; existing reference-book projections")
    return {
        "input_sources": "declared",
        "included": [
            governance,
            "explicit assignment, question, evidence and normalized task authority",
            "runtime access, tools, workspace, clock, resource and budget facts",
            "this child's own retained progress, tool and event history",
        ],
        "omitted_automatic": [
            "shared autobiography: identity, WORLD, dialogue and scratchpad",
            "global/project knowledge, indexes, patterns, journal and workpad",
            "shared review history, health narratives, update letters, registry and installed-skill summaries",
            "parent context, notes, review_notes, predecessor narrative and inherited attachments",
            "task-tree blackboard, routing manifests and other-task summaries",
        ],
        "lifetime": "Entire task, including tools, retries, fallback, compaction and selected descendants.",
        "collaboration": (
            "Ordinary task messages remain available; input selection imposes no collaboration order. "
            "The host does not gate mailbox delivery or detect the first position."
        ),
        "later_inputs": (
            "Tools, messages, reviews and restored sources are additional inputs; reading outside "
            "the declared corpus changes the claim. Retained model-send/tool-source projections "
            "record the actual sequence."
        ),
        "limitations": (
            "No sandbox, no semantic filtering of declared facts or governance/authority, and no "
            "promise about learned priors. Vendor-side context is unobserved. This receipt records "
            "host composition, not blanket blindness or control over vendor-side native context."
        ),
    }


def _text(value: Any) -> str:
    if isinstance(value, list):
        value = "\n".join(f"- {item}" for item in value if str(item).strip())
    return str(value or "").strip()


def selected_input_sources(ctx: Any, requested: Any = None) -> str:
    from ouroboros.contracts.task_contract import normalize_input_sources, task_input_sources
    inherited = task_input_sources({"task_contract": getattr(ctx, "task_contract", {}),
                                    "metadata": getattr(ctx, "task_metadata", {})})
    selected = normalize_input_sources(requested) if requested is not None else inherited
    if inherited == "declared" and selected != "declared":
        raise ValueError("input_sources=shared cannot widen an inherited declared selection")
    return selected


def declared_parent_contract(contract: Mapping[str, Any], context: str = "") -> dict:
    """Keep full task authority while omitting inherited narrative carriers."""
    projected = {key: value for key, value in contract.items()
                 if key not in {"notes", "review_notes", "predecessor_authority", "attachment_manifest_ref"}}
    projected.update(context=context, attachment_manifest=[])
    predecessor = contract.get("predecessor_authority")
    if isinstance(predecessor, Mapping) and predecessor:
        keys = {"source", "task_id", "authority_sha256", "authority_chars", "digest_semantics"}
        reference = {key: value for key, value in predecessor.items() if key in keys}
        omitted = predecessor.get("omitted_fields")
        reference["omitted_fields"] = sorted(set(
            [str(key) for key in predecessor if key not in keys | {"omitted_fields"}]
            + (list(omitted) if isinstance(omitted, list) else [])))
        projected["predecessor_authority"] = reference
    return projected


def assignment_instructions(ctx: Any, input_sources: str | None = None) -> str:
    """Host-authored complete normalized contract for every direct delegate start."""

    contract = getattr(ctx, "task_contract", None)
    if not isinstance(contract, dict) or not contract:
        meta = getattr(ctx, "task_metadata", {})
        raw = meta.get("task_contract") if isinstance(meta, dict) else None
        contract = raw if isinstance(raw, dict) else {}
    if contract:
        from ouroboros.contracts.task_contract import build_task_contract

        contract = build_task_contract({"task_contract": contract})
    if not contract:
        return ""
    if input_sources == "declared":
        contract = {**declared_parent_contract(contract), "input_sources": "declared"}
    return (
        "HOST TASK CONTRACT AUTHORITY (complete normalized JSON; exact strings are authority):\n"
        + json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def memory_reference_text(reference: Any) -> str:
    """One plain reference projection, kept separate from task authority."""
    if not isinstance(reference, Mapping) or not reference.get("text"):
        return ""
    facts = {key: value for key, value in reference.items() if key != "text"}
    return ("MEMORY REFERENCE (selected context, not task authority)\n"
            "The following is attributed prior understanding, not a new instruction or permission.\n"
            + json.dumps(facts, ensure_ascii=False, sort_keys=True) + "\n" + str(reference["text"]))


def direct_start_selection(ctx, requested, retry_of):
    from ouroboros.delegate_shared import _fail
    try:
        selection = selected_input_sources(ctx, requested)
    except ValueError as exc:
        return "", _fail("delegate_start", "INPUT_SOURCE_SELECTION_INVALID", str(exc), definitely_unrun=True)
    if retry_of and requested is not None:
        return "", _fail("delegate_start", "retry_selector_conflict", "A retry replays its recorded input selection.")
    return selection, None


def direct_start_instructions(ctx, actor, selection, requested, source_bound=False):
    from ouroboros.tools.delegate import _assignment_instructions
    if actor.get("compiled_work_order") and requested is not None:
        raise ValueError("The scheduled work order already binds its input selection.")
    assignment = "" if actor.get("compiled_work_order") else _assignment_instructions(ctx, selection)
    reference = ""
    if not actor.get("compiled_work_order") and not source_bound and selection == "shared":
        from ouroboros.chronicle_view import helper_memory_reference
        reference = memory_reference_text(helper_memory_reference(ctx))
    if selection == "declared" and not actor.get("compiled_work_order"):
        receipt = input_source_selection_receipt({"input_sources": "declared",
            "configured_subagent": {"route": {"kind": "agent_session"}}})
        receipt["lifetime"] = "This external run and its immutable retries; later direct starts choose separately, subject to inherited selection."
        assignment += "\n\nINPUT SOURCE SELECTION\n" + json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    return assignment, reference


def chosen_request_fingerprint(request):
    """Direct requests retain authority/inert memory in instructions, prompt unchanged."""
    return sha256(json.dumps({key: request[key] for key in ("prompt", "instructions")},
                            ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


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
        ("INPUT SOURCE SELECTION", input_source_selection_receipt(task)),
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
    from ouroboros.contracts.task_contract import task_input_sources
    metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
    if task_input_sources(task) != "declared":
        memory = memory_reference_text(task.get("memory_reference") or metadata.get("memory_reference"))
        if memory:
            rendered.append(memory)
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
    """Compile the complete chosen brief; transport limits belong to the recipient."""

    return _render_external_work_order(task)


def start_binding_fingerprints(ctx: Any, prompt: str) -> tuple[str, str]:
    """Digest the exact brief and the existing normalized task authority."""

    from ouroboros.delegate_recovery import authority_fingerprint_from_context

    return (
        sha256(str(prompt).encode("utf-8")).hexdigest(),
        authority_fingerprint_from_context(ctx),
    )


def work_order_fingerprint(task: Mapping[str, Any]) -> str:
    """Digest the complete canonical brief."""

    return sha256(_render_external_work_order(task).encode("utf-8")).hexdigest()


__all__ = [
    "assignment_instructions", "compile_external_work_order", "canonical_work_order_source",
    "work_order_source_projection",
    "validate_work_order_source_response",
    "start_binding_fingerprints", "work_order_fingerprint",
]
