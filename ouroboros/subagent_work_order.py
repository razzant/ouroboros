"""One compiler for configured-session work orders and host assignment context."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Mapping


def input_source_selection_receipt(task: Mapping[str, Any]) -> dict[str, Any]:
    """Composition fact retained by existing work-order/model-send sources.

    This selects automatic inputs, never filesystem access or semantic purity;
    an API-model child and a configured-session child (nanny plus leaf) carry
    the same selection, each recipient named in ``limitations``.
    The selection persists for the task; first-position/mail ordering is the
    caller's ordinary collaboration procedure, not a second host controller.
    """
    from ouroboros.contracts.task_contract import task_input_sources

    if task_input_sources(task) != "declared":
        return {}
    return {
        "input_sources": "declared",
        "included": [
            "SYSTEM.md and BIBLE.md; existing reference-book projections",
            "explicit assignment, question, evidence and normalized task authority",
            "runtime access, tools, workspace, clock, resource and budget facts",
            "the inherited delegation_budget.intent_note: parent-authored advice carried in the task contract, never authority",
            "this child's own retained progress, tool and event history",
        ],
        "omitted_automatic": [
            "shared autobiography: identity, WORLD, the top level of the life account, the parent's room page "
            "(with the words that started its project) and the memory marks of that room and the global ones",
            "the owner's words that caused this work (shared children receive them verbatim; include them in "
            "context if this case needs them)",
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
            "host composition, not blanket blindness. A configured-session child has two recipients: "
            "the nanny receives this host-composed core and the session leaf receives the compiled work "
            "order; files the harness loads from its project and vendor-side memory are unobserved."
        ),
    }


def _text(value: Any) -> str:
    if isinstance(value, list):
        value = "\n".join(f"- {item}" for item in value if str(item).strip())
    return str(value or "").strip()


def assignment_instructions(ctx: Any) -> str:
    """Host-authored assignment with a predecessor brief for each direct start.

    The owner's words that caused the work follow the contract: a root's own
    corpus, a child's inherited words, or the host's absence marker; a declared
    contract keeps its parent's selection and carries none. Full predecessor
    reports are read through the launching task, never claimed to be inline.
    """

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
    from ouroboros.contracts.task_contract import task_input_sources
    from ouroboros.main_context_authority import project_helper_predecessor_authority
    from ouroboros.owner_words import owner_words_text

    # A declared run's receipt names the words among omitted inputs, for the sessions it starts too.
    declared = task_input_sources({"task_contract": contract}) == "declared"
    if contract.get("predecessor_authority"):
        contract["predecessor_authority"] = project_helper_predecessor_authority(
            contract["predecessor_authority"], declared=declared)
    words = "" if declared else owner_words_text(ctx, audience="session")
    return (
        "HOST TASK CONTRACT AUTHORITY (normalized JSON; predecessor is a brief):\n"
        + json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + ("\n\nThe predecessor brief omits evidence; its source names get_task_result. "
           "Ask the launching task (nanny or root) for the full report; it can reply through delegate_answer. "
           "Without a question channel, return input_required for a new run with the answers."
           if contract.get("predecessor_authority") else "")
        + (f"\n\n{words}" if words else "")
    )


def _owner_words_section(task: Mapping[str, Any]) -> str:
    """The owner's words carried into this task by value; "" for a task scheduled without them.

    Both the payload (nested ``metadata``) and the flat task-metadata/record shapes
    give the same bytes, so a source-range check never sees a digest mismatch. A
    declared task carries the field too, but its receipt names the words among the
    omitted inputs, so its work order holds none: it keeps its parent's selection and the
    words ride its task unread.
    """
    from ouroboros.contracts.task_contract import task_input_sources
    from ouroboros.owner_words import render_owner_words, task_governing_words

    if task_input_sources(task) == "declared":
        return ""
    return render_owner_words(*task_governing_words(task), audience="session",
                              root_task_id=str(task.get("root_task_id") or ""))


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
    owner_words = _owner_words_section(task)
    rendered = []
    for title, value in sections:
        body = (
            str(value) if title == "PARENT CONTEXT / REFERENCES" and isinstance(value, str)
            else _text(value)
        )
        if body:
            rendered.append(f"{title}\n{body}")
        if title == "PARENT CONTEXT / REFERENCES" and owner_words:
            rendered.append(owner_words)  # verbatim, its own heading; absent for a task scheduled before it
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
