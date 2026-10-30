"""One compiler for configured-session work orders and host assignment context."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Mapping

# Recorded beside the owner's words when a declared child is scheduled: its selection carries them
# (BIBLE P1). A declared task scheduled before this fact existed keeps the selection it started with,
# and with it the work-order bytes and fingerprint its recovery and source-range reads check.
DECLARED_WORDS_FACT = "declared_inputs_carry_owner_words"


def declared_words_fact(contract: Mapping[str, Any]) -> dict[str, bool]:
    """The fact a declared child is scheduled with, ``{}`` for a shared one (its words need none)."""
    from ouroboros.contracts.task_contract import task_input_sources

    return {DECLARED_WORDS_FACT: True} if task_input_sources({"task_contract": contract}) == "declared" else {}


def declared_inputs_carry_owner_words(task: Mapping[str, Any]) -> bool:
    """Whether the task was scheduled with that fact, read at the top level or under ``metadata``."""
    metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
    return task.get(DECLARED_WORDS_FACT) is True or metadata.get(DECLARED_WORDS_FACT) is True


def input_source_selection_receipt(task: Mapping[str, Any]) -> dict[str, Any]:
    """Composition fact retained by existing work-order/model-send sources.

    This selects automatic inputs, never filesystem access or semantic purity;
    an API-model child and a configured-session child (nanny plus leaf) carry
    the same selection, each recipient named in ``limitations``.
    The selection persists for the task; first-position/mail ordering is the
    caller's ordinary collaboration procedure, not a second host controller.
    The owner's words that caused the work are included when the task was
    scheduled with ``DECLARED_WORDS_FACT``; an older declared task names them
    among the omitted inputs, as its work order always did. This receipt describes
    case-input selection; standing owner rules are separate context/host governance,
    outside the canonical work order so its retained bytes and fingerprint stay stable.
    """
    from ouroboros.contracts.task_contract import task_input_sources

    if task_input_sources(task) != "declared":
        return {}
    selected = ["explicitly selected task documents (attachment_paths), copied as this task's inputs"
                ] if _selected_documents(task) else []
    with_words = declared_inputs_carry_owner_words(task)
    return {
        "input_sources": "declared",
        "included": [
            "SYSTEM.md and BIBLE.md; reference-book projections; own-body DESIGN navigation or source-gap disclosure",
            "explicit assignment, question, evidence and normalized task authority",
            *(["the owner's words that caused this work, verbatim as the host carries them down the task tree, "
               "or the host's marker that none are recorded"] if with_words else []), *selected,
            "runtime access, tools, workspace, clock, resource and budget facts",
            "the inherited delegation_budget.intent_note: parent-authored advice carried in the task contract, never authority",
            "this child's own retained progress, tool and event history",
        ],
        "omitted_automatic": [
            "shared autobiography: identity, WORLD, the top level of the life account, the parent's room page "
            "(with the words that started its project) and the memory marks of that room and the global ones",
            *([] if with_words else ["the owner's words that caused this work (shared children receive them verbatim; "
                                     "include them in context if this case needs them)"]),
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
            + (" When this task delegates through schedule_subagent or delegate_start, the host adds "
               "the knowledge index (the global shelf's and its room's) once, to the result of the "
               "first accepted call; knowledge_list reads it again." if with_words else "")
        ),
        "limitations": (
            "No sandbox, no semantic filtering of declared facts or governance/authority, and no "
            "promise about learned priors. Vendor-side context is unobserved. This receipt records "
            "host composition, not blanket blindness. A configured-session child has two recipients: "
            "the nanny receives this host-composed core and the session leaf receives the compiled work "
            "order; files the harness loads from its project and vendor-side memory are unobserved."
        ),
    }


def _selected_documents(task: Mapping[str, Any]) -> str:
    """The parent's ``attachment_paths`` disclosure; "" when the task selected none."""
    paths = task.get("attachment_paths")
    if not isinstance(paths, list) or not paths:
        return ""
    return ("The parent selected these retained inputs (attachment_paths). The host attaches them to the first "
            "start of this work order as immutable engine documents; a later start attaches only the documents "
            "its own request selects.\n" + "\n".join(f"- {path}" for path in paths))


def _text(value: Any) -> str:
    if isinstance(value, list):
        value = "\n".join(f"- {item}" for item in value if str(item).strip())
    return str(value or "").strip()


def assignment_instructions(ctx: Any) -> str:
    """Host-authored assignment with a predecessor brief for each direct start.

    The owner's words that caused the work follow the contract: a root's own
    corpus, a child's inherited words, or the host's absence marker; a declared
    contract scheduled before ``DECLARED_WORDS_FACT`` keeps the selection it
    started with and carries none. Full predecessor reports are read through the
    launching task, never claimed to be inline.
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

    # An older declared run's receipt names the words among omitted inputs, for the sessions it starts too.
    declared = task_input_sources({"task_contract": contract}) == "declared"
    if contract.get("predecessor_authority"):
        contract["predecessor_authority"] = project_helper_predecessor_authority(
            contract["predecessor_authority"], declared=declared)
    meta = getattr(ctx, "task_metadata", None)
    older = declared and not declared_inputs_carry_owner_words(meta if isinstance(meta, Mapping) else {})
    words = "" if older else owner_words_text(ctx, audience="session")
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
    declared task scheduled before ``DECLARED_WORDS_FACT`` carries the field too, but
    its receipt names the words among the omitted inputs, so its work order holds none
    and keeps the bytes and fingerprint it started with.
    """
    from ouroboros.contracts.task_contract import task_input_sources
    from ouroboros.owner_words import render_owner_words, task_governing_words

    if task_input_sources(task) == "declared" and not declared_inputs_carry_owner_words(task):
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
        ("SELECTED DOCUMENTS", _selected_documents(task)),
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


def _own_body_source(ctx: Any, target_root: str, source_root: str) -> tuple[Any, bool]:
    """``(root, prepared)``: where a session reads the author's own body, or ``(None, False)``.

    Identity comes from the selected task source; bytes come from the prepared copy
    (``prepared``), never a foreign target's file. Declared folderless body work
    reads the system docs, not its scratch."""
    from pathlib import Path

    from ouroboros.context_layout import task_design_source

    metadata = getattr(ctx, "task_metadata", {})
    task = dict(metadata) if isinstance(metadata, Mapping) else {}
    task["task_contract"] = getattr(ctx, "task_contract", {})
    selected = task_design_source(ctx, task)
    if selected is None:
        return None, False
    try:
        folder = Path(target_root).resolve().relative_to(selected.resolve())
    except ValueError:
        from ouroboros.tools.tool_resolution import active_repo_dir_for
        from ouroboros.workspace_copies import same_directory

        if not same_directory(target_root, active_repo_dir_for(ctx)):
            return None, False  # e.g. a payload session launched by an own-body task
        return selected, False
    source = Path(source_root).resolve()
    for _part in folder.parts:
        source = source.parent  # an investigation can start below the copied repo root
    return source, True


def session_design_navigation(ctx: Any, *, target_root: str, source_root: str) -> str:
    """Pin body navigation in the prepared session request, beside its work order.

    The canonical work order is reconstructed for source-range validation and
    cannot depend on mutable files. Session instructions instead ride the durable
    start request and its byte-identical retry (``_own_body_source`` selects the bytes).
    """
    from ouroboros.context_layout import design_navigation

    source, _prepared = _own_body_source(ctx, target_root, source_root)
    if source is None:
        return ""
    path = source / "docs/DESIGN.md"
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError):
        raw, text = None, None
    navigation = design_navigation(text, source=str(path))
    digest = f"\nSource bytes SHA-256: {sha256(raw).hexdigest()}." if raw is not None else ""
    return ("\n\nDESIGN NAVIGATION DELIVERY (host instructions, separate from canonical work order)\n"
            + navigation + digest)


def session_constitution_pointer(ctx: Any, *, target_root: str, source_root: str) -> str:
    """Point an own-body session at the constitution of the body it works on, beside DESIGN.

    The leaf is an executor, not Ouroboros, and starts before its nanny's first turn; a path
    and the pinned bytes' digest let its first proposals meet P1. The text is never inlined."""
    source, _prepared = _own_body_source(ctx, target_root, source_root)
    if source is None:
        return ""
    path = source / "BIBLE.md"
    head = "\n\nCONSTITUTION DELIVERY (host instructions, separate from canonical work order)\n"
    try:
        digest = sha256(path.read_bytes()).hexdigest()
    except OSError:
        return head + f"BIBLE.md source unavailable at {path}; no constitution delivered."
    return head + (f"This work concerns the body of Ouroboros, which {path} governs. Before your first proposal, "
                   "read its Principle 1 (continuity, memory and the cognitive horizon); read the rest as the "
                   f"work needs.\nSource bytes SHA-256: {digest}.")


def session_service_directories(ctx: Any, gateway: Any, route: Any, authority: Any, *, snapshot: Any,
                                payload: bool, execution_root: str, in_place: str = "") -> Any:
    """What ONE first start's private copy may write outside it, prepared before the binding, the host
    references and the request name it (``workspace_preflight.session_service_directories``).

    The delegating task's drive owns the short scratch, keyed by the snapshot. A same-tree
    continuation from another task can receive a different scratch. ``in_place`` is the invocation
    of an in-place run, which has no copy: at full access it gets its own scratch, named and never
    granted; a ``workspace_write`` one gets none, since that would be a new grant. A skill payload,
    an engine directory copy and any other session without a copy get none; a retry replays
    recorded old paths and never comes here, even if the original task's drive or the OS already
    retired them."""
    from ouroboros.tool_access_paths import canonical_data_root
    from ouroboros.workspace_preflight import ServiceDirectories, session_service_directories as prepare

    if payload or (snapshot is None and not (in_place and authority.access == "full")):
        return ServiceDirectories()
    return prepare(gateway, route.route_id, access=authority.access, mode=authority.mode,
                   data_root=canonical_data_root(ctx), drive_root=getattr(ctx, "drive_root", ""),
                   task_id=str(getattr(ctx, "task_id", "") or ""),
                   key=snapshot.snapshot_id if snapshot is not None else f"in-place-{in_place}",
                   execution_root=execution_root)


def in_place_scratch(ctx: Any, gateway: Any, route: Any, authority: Any, invocation_id: str, target_root: str) -> Any:
    """One in-place run's scratch preparation, deferred: ``delegate_directory.in_place_start`` calls it
    only after accepting the start (``session_service_directories`` with ``in_place``)."""
    from functools import partial

    return partial(session_service_directories, ctx, gateway, route, authority, snapshot=None, payload=False,
                   execution_root=target_root, in_place=invocation_id)


def session_host_references(ctx: Any, *, target_root: str, source_root: str,
                            private_copy: bool = False, payload_skill: str = "", service: Any = None) -> str:
    """Host references pinned in a session's start request, computed once at first start.

    Own-body DESIGN navigation and BIBLE path, the owner's standing rules for every session (declared
    inputs included: they govern the work, as BIBLE does), and the installed instruction-skill index for any workspace;
    a skill-payload builder also gets the serving host's skill contract, whatever its input
    selection, and any other private copy its environment facts: dependency files read at the
    copy's repository root, the environment placed where the session may write (``source_root``,
    possibly a folder below that root, or the scratch ``service`` grants); a read-only session
    has no copy, so it gets neither. A retry replays the recorded bytes."""
    from ouroboros.safety_config import owner_rules_section
    from ouroboros.skill_catalogue import session_skill_index, skill_contract_references  # lazy skill reads

    text = (session_design_navigation(ctx, target_root=target_root, source_root=source_root)
            + session_constitution_pointer(ctx, target_root=target_root, source_root=source_root)
            + owner_rules_section(session=True) + session_skill_index(ctx))
    if payload_skill:
        from ouroboros.body_candidate import serving_repo_dir_for

        return text + skill_contract_references(serving_repo_dir_for(ctx))
    if private_copy:
        from pathlib import Path

        from ouroboros.test_environment import git_ancestor
        from ouroboros.tool_access_paths import canonical_data_root
        from ouroboros.workspace_preflight import session_environment_facts

        start = Path(source_root).resolve()  # the session's execution root: its only write place in the copy
        root, body = _own_body_source(ctx, target_root, source_root)  # the copy's repository root
        root = root if body else git_ancestor(start) or start
        return text + session_environment_facts(root, data_root=canonical_data_root(ctx), body=body, write_root=start,
                                                **({"service": service} if service is not None else {}))
    return text


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
    "DECLARED_WORDS_FACT", "declared_inputs_carry_owner_words", "declared_words_fact",
    "assignment_instructions", "compile_external_work_order", "canonical_work_order_source",
    "work_order_source_projection",
    "validate_work_order_source_response",
    "start_binding_fingerprints", "work_order_fingerprint",
]
