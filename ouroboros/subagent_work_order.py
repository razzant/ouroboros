"""One compiler for configured-session work orders and host assignment context."""

from __future__ import annotations

import json
import logging
from hashlib import sha256
from typing import Any, Mapping, NamedTuple

log = logging.getLogger(__name__)

# Recorded beside the owner's words when a declared child is scheduled: its selection carries them
# (BIBLE P1). A declared task scheduled before this fact existed keeps the selection it started with,
# and with it the work-order bytes and fingerprint its recovery and source-range reads check.
DECLARED_WORDS_FACT = "declared_inputs_carry_owner_words"
# An immutable scheduling fact, not a migration: records without it must
# reconstruct the original canonical bytes for source reads and handoff recovery.
ORIGINAL_INPUTS_FACT = "work_order_input_files_original_only"


def declared_words_fact(contract: Mapping[str, Any]) -> dict[str, bool]:
    """The fact a declared child is scheduled with, ``{}`` for a shared one (its words need none)."""
    from ouroboros.contracts.task_contract import task_input_sources

    return {DECLARED_WORDS_FACT: True} if task_input_sources({"task_contract": contract}) == "declared" else {}


def declared_inputs_carry_owner_words(task: Mapping[str, Any]) -> bool:
    """Whether the task was scheduled with that fact, read at the top level or under ``metadata``."""
    metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
    return task.get(DECLARED_WORDS_FACT) is True or metadata.get(DECLARED_WORDS_FACT) is True


def input_source_selection_receipt(task: Mapping[str, Any], *, native: bool = False) -> dict[str, Any]:
    """Composition fact retained by existing work-order/model-send sources.

    This selects automatic inputs, never filesystem access or semantic purity;
    an API-model child and a configured-session child (nanny plus leaf) carry
    the same selection, each recipient named in ``limitations``.
    The selection persists for the task; first-position/mail ordering is the
    caller's ordinary collaboration procedure, not a second host controller.
    My human's words for the work are included when the task was scheduled with
    ``DECLARED_WORDS_FACT``; an older declared task names them among the omitted
    inputs, as its work order always did. A native request (``native``: the one
    context builder's) and a work order of a session task scheduled with its room
    rows (``work_order_room_words``) name the marks in force and the knowledge
    outline among the included inputs; an older session task's work order keeps
    the literals it was rendered with, so its retained bytes and fingerprint stay
    stable. This receipt describes case-input selection; standing owner rules are
    separate context/host governance, outside the canonical work order.
    """
    from ouroboros.contracts.task_contract import task_input_sources

    if task_input_sources(task) != "declared":
        return {}
    selected = (["explicitly selected task documents (attachment_paths), copied as this task's inputs"
                 ] if _selected_documents(task) else []) + (
        ["explicitly selected input files (input_files), retained by the host before this task was queued"]
        if _selected_input_files(task) else [])
    with_words = declared_inputs_carry_owner_words(task)
    current = native or work_order_room_words(task) is not None
    words_item = ("my human's words for this work, verbatim and dated as the host carries them down the task tree, "
                  "or the host's marker that none are recorded" if current else
                  "the owner's words that caused this work, verbatim as the host carries them down the task tree, "
                  "or the host's marker that none are recorded")
    no_words_item = ("my human's words for this work" if current else "the owner's words that caused this work"
                     ) + " (shared children receive them verbatim; include them in context if this case needs them)"
    return {
        "input_sources": "declared",
        "included": [
            "SYSTEM.md and BIBLE.md; reference-book projections; own-body DESIGN navigation or source-gap disclosure",
            "explicit assignment, question, evidence and normalized task authority",
            *([words_item] if with_words else []), *selected,
            *(["the memory marks in force (this room's and the global ones) and the knowledge outline "
               "(every note by address and title)"] if current else []),
            "runtime access, tools, workspace, clock, resource and budget facts",
            "the inherited delegation_budget.intent_note: parent-authored advice carried in the task contract, never authority",
            "this child's own retained progress, tool and event history",
        ],
        "omitted_automatic": [
            "shared autobiography: identity, WORLD, the top level of the life account, the parent's room page "
            + ("and its open conversation" if current else
               "(with the words that started its project) and the memory marks of that room and the global ones"),
            *([] if with_words else [no_words_item]),
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


def _selected_input_files(task: Mapping[str, Any]) -> str:
    """The parent's ``input_files`` disclosure; "" when the task selected none."""
    rows = task.get("prepared_input_files")
    if not isinstance(rows, list) or not rows:
        return ""
    metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
    if task.get(ORIGINAL_INPUTS_FACT, metadata.get(ORIGINAL_INPUTS_FACT)) is not True:
        return ("The parent selected these files (input_files); the host retained each one's exact bytes before queueing "
                "this task. The first start of this work order verifies those copies against these digests and its "
                "instructions name where this run reads them; a later start copies only what its own request selects.\n"
                + "\n".join(f"- {row.get('root')}:{row.get('path')} (sha256 {row.get('sha256')}, {row.get('size')} bytes)"
                            for row in rows if isinstance(row, dict)))
    from ouroboros.delegate_readonly_inputs import selection_label

    return ("Original schedule-time selection (input_files). Follow this start's SELECTED INPUT FILES instructions "
            "for effective copies. Without them there are no explicit copies for this run. A replacement, including [], "
            "supersedes these rows.\n"
            + "\n".join(f"- {selection_label(row)} (sha256 {row['sha256']}, {row['size']} bytes)" for row in rows))


def _text(value: Any) -> str:
    if isinstance(value, list):
        value = "\n".join(f"- {item}" for item in value if str(item).strip())
    return str(value or "").strip()


def assignment_instructions(ctx: Any) -> str:
    """Host-authored assignment with a predecessor brief for each direct start.

    My human's words for the work follow the contract: a root's own
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


def work_order_owner_rows(task: Mapping[str, Any]) -> tuple | None:
    """``(rows, absence marker, current)``: my human's rows a session work order carries, one per identity (the
    field's, then the bound room's), the same set ``_owner_words_section`` renders and the receipt counts; ``None``
    for a declared task scheduled before ``DECLARED_WORDS_FACT``, whose order holds none."""
    from ouroboros.contracts.task_contract import task_input_sources
    from ouroboros.owner_words import FIELD, _deduped, task_governing_words

    if task_input_sources(task) == "declared" and not declared_inputs_carry_owner_words(task):
        return None
    rows, absent = task_governing_words(task)
    room = work_order_room_words(task)
    if room is not None:
        rows = _deduped(rows + task_governing_words({FIELD: room})[0])
    return rows, absent, room is not None


def work_order_room_words(task: Mapping[str, Any]) -> list | None:
    """The bound room's rows a session task was scheduled with (``owner_words.ROOM_FIELD``, by value, at the top
    level or under ``metadata``); ``None`` for a task scheduled before the key existed, whose work order keeps the
    constants it was rendered with. The key is read here alone."""
    from ouroboros.owner_words import ROOM_FIELD

    metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
    rows = task.get(ROOM_FIELD, metadata.get(ROOM_FIELD))
    return list(rows) if isinstance(rows, list) else None


def _owner_words_section(task: Mapping[str, Any]) -> str:
    """The owner's words carried into this task by value; "" for a task scheduled without them.

    Both the payload (nested ``metadata``) and the flat task-metadata/record shapes
    give the same bytes, so a source-range check never sees a digest mismatch. A
    declared task scheduled before ``DECLARED_WORDS_FACT`` carries the field too, but
    its receipt names the words among the omitted inputs, so its work order holds none
    and keeps the bytes and fingerprint it started with. A session task's room rows
    (``work_order_room_words``) follow the field's rows, one row per identity; nothing
    is read from a store here, so the order is a function of the record. The key also
    tells the heading and frame: a record without it keeps the literals it launched with.
    The schedule text it was scheduled with (``owner_words.SCHEDULE_FIELD``) follows them
    in its own section; a record without that field renders as it always did.
    """
    from ouroboros.owner_words import render_owner_words, render_schedule_text, task_schedule_items

    carried = work_order_owner_rows(task)
    if carried is None:
        return ""
    rows, absent, current = carried
    words = render_owner_words(rows, absent, audience="session", root_task_id=str(task.get("root_task_id") or ""),
                               current=current)
    schedule = render_schedule_text(task_schedule_items(task), audience="session")
    return "\n\n".join(part for part in (words, schedule) if part)


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
        ("SELECTED INPUT FILES", _selected_input_files(task)),
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
    from ouroboros import work_domain
    from ouroboros.context_layout import task_design_source

    selected = task_design_source(ctx, _context_task(ctx))
    if selected is None:
        return None, False
    try:
        folder = work_domain.settle(target_root).relative_to(work_domain.settle(selected))
    except ValueError:
        from ouroboros.tools.tool_resolution import active_repo_dir_for
        from ouroboros.workspace_copies import same_directory

        if not same_directory(target_root, active_repo_dir_for(ctx)):
            return None, False  # e.g. a payload session launched by an own-body task
        return selected, False
    source = work_domain.settle(source_root)
    for _part in folder.parts:
        source = source.parent  # an investigation can start below the copied repo root
    return source, True


def _context_task(ctx: Any) -> dict[str, Any]:
    """The task facts a design source is selected by: the context's metadata and its contract."""
    metadata = getattr(ctx, "task_metadata", {})
    task = dict(metadata) if isinstance(metadata, Mapping) else {}
    task["task_contract"] = getattr(ctx, "task_contract", {})
    return task


def session_design_navigation(ctx: Any, *, target_root: str, source_root: str) -> str:
    """Pin the body's DESIGN in the prepared session request, beside its work order: the author's view, its
    design bible whole and the rest by headings (``context_layout.design_bible_view``), with the bytes' digest.

    The canonical work order is reconstructed for source-range validation and
    cannot depend on mutable files. Session instructions instead ride the durable
    start request and its byte-identical retry (``_own_body_source`` selects the bytes).
    A session on the task's own working folder that is not the body takes the author's wider choice (item 97, RV1):
    the installed book for a true ``context_requires_self_body_docs`` or an unknown body. The BIBLE pointer and the
    environment facts keep ``_own_body_source``.
    """
    from ouroboros.context_layout import author_design_source, design_bible_view, read_design_source
    from ouroboros.tools.tool_resolution import active_repo_dir_for
    from ouroboros.workspace_copies import same_directory

    source, prepared = _own_body_source(ctx, target_root, source_root)
    installed = False
    if not prepared and same_directory(target_root, active_repo_dir_for(ctx)):
        source, installed = author_design_source(ctx, _context_task(ctx))
    if source is None:
        return ""
    path = source / "docs/DESIGN.md"
    raw, text = read_design_source(path) or (None, None)
    view = design_bible_view(text, source=str(path), installed=installed)
    digest = f"\nSource bytes SHA-256: {sha256(raw).hexdigest()}." if raw is not None else ""
    return "\n\nDESIGN DELIVERY (host instructions, separate from canonical work order)\n" + view + digest


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
    granted, in a private child of the home where it qualifies (``private``); a ``workspace_write``
    one gets none, since that would be a new grant. A skill payload,
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
                   execution_root=execution_root, private=snapshot is None)


def in_place_scratch(ctx: Any, gateway: Any, route: Any, authority: Any, invocation_id: str, target_root: str) -> Any:
    """One in-place run's scratch preparation, deferred: ``delegate_directory.in_place_start`` calls it
    only after accepting the start (``session_service_directories`` with ``in_place``)."""
    from functools import partial

    return partial(session_service_directories, ctx, gateway, route, authority, snapshot=None, payload=False,
                   execution_root=target_root, in_place=invocation_id)


_ATTACHMENTS_HOLDER = (" An attachment named under a row is retained by the host: the task that started this "
                       "session holds it and can be asked for it.")
START_TAIL_FACT = ("[Host fact: these rows of my human's were not in the work order or instructions above; "
                   "the host carries them with this start, verbatim." + _ATTACHMENTS_HOLDER + "]")
CONTINUATION_TAIL_FACT = ("[Host fact: these rows of my human's stand in no earlier start of this session; "
                          "the host carries them with this continuation, verbatim." + _ATTACHMENTS_HOLDER + "]")


def unseen_words_section(ctx: Any, text: str, fact: str) -> str:
    """My human's rows of the starter's set at this handoff (``words_unseen_in``: the room read live) whose
    identity ``text`` does not hold, each with the reply target that follows it in the set (another author's
    words beside theirs, under its own stamp), under the session heading, the gap line of a store that could
    not be read (``work_words``) and ``fact`` as the host's closing lines; "" when ``text`` holds them all. A text
    with no stamp holds none: every row rides."""
    from ouroboros.owner_relay import words_unseen_in
    from ouroboros.owner_words import REPLY_TARGET, render_owner_words, work_words

    unseen = words_unseen_in(ctx, text)
    if not unseen:
        return ""
    rows, take = [], False
    rows_all, _marker, gaps = work_words(ctx)
    for row in rows_all:
        take = take if row.get("source") == REPLY_TARGET else row in unseen
        if take:
            rows.append(row)
    meta = getattr(ctx, "task_metadata", None)
    root = str((meta.get("root_task_id") if isinstance(meta, Mapping) else "") or getattr(ctx, "task_id", "") or "")
    return render_owner_words(rows, audience="session", root_task_id=root, gaps=[*gaps, fact])


NAMED_NOTES = "knowledge_notes"  # the notes the delegating mind names for a helper, by knowledge_read's address


def knowledge_notes_parameter(text: str) -> dict[str, Any]:
    """Shared address shape; each tool documents when its selection takes effect.

    Compact custom projection keeps the address shape without the parameter prose.
    """
    item = {"type": "object", "required": ["topic"], "additionalProperties": False,
            "properties": {"topic": {"type": "string"}, "scope": {"type": "string"}}}
    return {NAMED_NOTES: {"type": "array", "items": item,
                          "description": "Select knowledge notes by knowledge_read address ({topic, scope?}). " + text}}


class NamedNoteError(ValueError):
    """Named notes the caller cannot hand on, each problem with the call that works; nothing was queued or sent."""

    def __init__(self, problems: list[str], *, code: str = "knowledge_note_unresolved") -> None:
        super().__init__("; ".join(problems))
        self.code, self.detail = code, str(self)


def note_label(entry: Mapping[str, Any]) -> str:
    return f"{entry.get('topic')} ({entry.get('scope')})"


def _note_address(reader: Any, topic: str, scope: str, *, transfer: bool):
    """Check reader and path first; the range and revision may depend on the captured note."""
    from inspect import signature

    from ouroboros.delegate_readonly_inputs import admit_selected_path_policy, require_selected_tool
    from ouroboros.knowledge import shelf_spelling
    from ouroboros.tool_access import active_tool_profile, decide_tool_access, workspace_mode_block_reason
    from ouroboros.tool_access_types import ResolvedResourceBinding
    from ouroboros.tools.knowledge import _address, _knowledge_read, _range
    from ouroboros.tools.registry_core import PRESENCE_RESOURCE_BLOCKED, _presence_binding_allowed

    bound = require_selected_tool(reader, "knowledge_read", {"topic": topic, "scope": scope})
    if problem := workspace_mode_block_reason(reader):
        raise PermissionError(f"Workspace restriction ({problem}).")
    try:
        call = signature(_knowledge_read).bind(reader, **bound)
        call.apply_defaults()
        args = call.arguments
        start, end, version = args["start_char"], args["end_char"], args["version"]
        # A zero-length source is the necessary shape check: any valid whole
        # range starts at zero, and _range clamps a nonnegative end to EOF.
        _range(0, start, end)
    except (TypeError, ValueError):
        raise _not_whole() from None
    if (args["view"] not in ("note", None, "") or args["fingerprint"] not in (None, "")
            or (version is not None and not isinstance(version, str))):
        raise _not_whole()
    version = version if version and version.strip() else None
    if version is not None and (len(version) != 64 or version.strip("0123456789abcdef")):
        raise _not_whole()
    # A host binding may fill the default scope or name the same shelf entry, not select another one.
    effective = (args["topic"], args["scope"])
    requested = (topic, scope or args["scope"])
    address = _address(reader, *requested)
    # Project knowledge has a semantic reader and is deliberately not a generic
    # file-tool target. Check the positive ceiling and path policy on its exact canonical path.
    profile = active_tool_profile(reader)
    decision = decide_tool_access(profile=profile, root="runtime_data", operation="read")
    if not decision.allow:
        raise PermissionError(decision.reason)
    binding = ResolvedResourceBinding(
        profile=profile, root="runtime_data", operation="read", base_path=address.canonical_root,
        target_path=address.path.resolve(), source="runtime_data", skill_name="",
        state_drive_root=address.canonical_root, requested_path=address.path)
    if not _presence_binding_allowed(reader, binding):
        raise PermissionError(PRESENCE_RESOURCE_BLOCKED)
    admit_selected_path_policy(reader, binding, transfer=transfer)
    if effective != requested:
        bound_address = _address(reader, *effective)
        if bound_address != address and (bound_address.shelf != address.shelf
                                         or shelf_spelling(bound_address) != shelf_spelling(address)):
            raise PermissionError("The bound knowledge_read selects a different note.")
    return address, (end, version)


def _not_whole() -> PermissionError:
    return PermissionError("The bound knowledge_read does not select the whole current note.")


def named_note_arguments(value: Any, *, admitted: bool = False) -> tuple[list[dict] | None, list[str]]:
    """The shared shape pass for both launch commands. No reader or store is consulted."""
    from ouroboros.tools.arg_feedback import ignored_argument_note

    if value is None:
        return None, []
    code = "knowledge_note_unresolved" if admitted else "knowledge_notes_invalid"
    if not isinstance(value, list):
        raise NamedNoteError([f"{NAMED_NOTES}={value!r}: use an array of {{topic, scope?}} objects, as knowledge_read "
                              "takes them"], code=code)
    entries, problems, notes = [], [], []
    for index, item in enumerate(value, 1):
        name = f"{NAMED_NOTES} {index}"
        if isinstance(item, str):
            notes.append(f"{name}={item!r} read as {{'topic': {item!r}}}")
            item = {"topic": item}
        if (not isinstance(item, dict) or set(item) - {"topic", "scope"} or not isinstance(item.get("topic"), str)
                or not item["topic"].strip() or not isinstance(item.get("scope", ""), str)):
            problems.append(f"{name}={item!r}: an object {{topic, scope?}} with a nonempty topic, as knowledge_read "
                            "takes it")
            continue
        if "scope" in item and not item["scope"]:
            notes.append(ignored_argument_note(f"{name}.scope", item["scope"],
                                               "this task's project shelf, else global"))
        entries.append({"topic": item["topic"], "scope": item.get("scope", "")})
    if problems:
        raise NamedNoteError(problems, code=code)
    return entries, notes


def _note_readers(ctx, recipient, topic, scope, *, transfer, admitted):
    from ouroboros import knowledge as store

    address, limit = _note_address(ctx, topic, scope, transfer=transfer)
    limits = [limit]
    if admitted:
        if (address.topic, address.scope) != (topic, scope):
            raise PermissionError("The current knowledge_read does not select the admitted note address.")
    else:
        canonical = store.shelf_spelling(address)
        if canonical != address:
            address, limit = _note_address(ctx, canonical.topic, canonical.scope, transfer=transfer)
            if address != canonical:
                raise PermissionError("The bound knowledge_read does not select the canonical note address.")
            limits.append(limit)
    if recipient is not None:
        received, limit = _note_address(recipient, address.topic, address.scope, transfer=transfer)
        if received != address:
            raise PermissionError("The recipient's knowledge_read selects a different source.")
        limits.append(limit)
    return address, limits


def _whole_current(note, limits):
    from ouroboros.tools.knowledge import _range

    size = len(note.text) if any(end for end, _ in limits) else None
    for end, revision in limits:
        if ((revision is not None and revision != note.revision)
                or (end and _range(size, 0, end)[:2] != (0, size))):
            raise _not_whole()


class _SelectedNote(NamedTuple):
    entry: dict
    note: Any = None  # an immutable KnowledgeNote, held only by a start that will include its bytes
    gap: str = ""


def _selected_notes(ctx, entries, notes, *, recipient=None, transfer=False, capture=False, admitted=False):
    """Validate a schedule or take the one capture for a start. An admitted default may lose its source, not its gates."""
    from ouroboros import knowledge as store

    if entries is None:
        return None, notes
    found, problems = {}, []
    for index, entry in enumerate(entries, 1):
        name = f"{NAMED_NOTES} {index}"
        topic, scope = entry["topic"], entry["scope"]
        try:
            address, limits = _note_readers(ctx, recipient, topic, scope, transfer=transfer, admitted=admitted)
        except (PermissionError, ValueError) as exc:
            problems.append(f"{name} {topic!r}: {exc}")
            continue
        except OSError as exc:
            problems.append(f"{name} {topic!r}: the note address could not be checked ({type(exc).__name__})")
            continue
        identity = address.scope, address.topic
        entry = {"topic": address.topic, "scope": address.scope}
        if identity in found:
            notes.append(f"{name} names {note_label(entry)} again; it is carried once")
            continue
        try:
            note = store.read_knowledge_note(address)
        except (OSError, UnicodeDecodeError) as exc:
            if admitted:
                found[identity] = _SelectedNote(entry, gap=type(exc).__name__)
            elif any(end or revision for end, revision in limits):
                problems.append(f"{name} {topic!r}: {_not_whole()}")
            elif isinstance(exc, FileNotFoundError):
                elsewhere = _global_note_hint(ctx, recipient, address, transfer=transfer)
                problems.append(f"{name}: no note {address.topic!r} on the {address.scope} shelf; "
                                f"knowledge_list(scope={address.scope!r}) lists that shelf{elsewhere}")
            else:
                problems.append(f"{name} {topic!r}: the note could not be read ({type(exc).__name__})")
            continue
        try:
            _whole_current(note, limits)
        except PermissionError as exc:
            problems.append(f"{name} {topic!r}: {exc}")
            continue
        found[identity] = _SelectedNote(entry, note if capture else None)
    if problems:
        raise NamedNoteError(problems)
    return list(found.values()), notes


def _global_note_hint(ctx, recipient, address, *, transfer):
    if address.scope != "global":
        try:
            alternate, limits = _note_readers(ctx, recipient, address.topic, "global", transfer=transfer, admitted=True)
            if not any(end or revision for end, revision in limits) and alternate.path.is_file():
                return f"; a global note exists: {{'topic': {address.topic!r}, 'scope': 'global'}}"
        except (OSError, ValueError):
            pass  # A repair hint cannot reveal an unreadable alternate source.
    return ""


def resolve_named_notes(ctx, selection, *, recipient=None, session=False):
    """Schedule addresses, not bodies or claims about their eventual delivery."""
    rows, feedback = _selected_notes(ctx, *selection, recipient=recipient, transfer=session)
    return None if rows is None else [row.entry for row in rows], feedback


def prepare_named_notes(ctx, value, *, admitted=False):
    """Prepare one start from one read of each admitted note; rendering never reopens it."""
    rows, feedback = _selected_notes(ctx, *named_note_arguments(value, admitted=admitted),
                                     transfer=True, capture=True, admitted=admitted)
    return named_notes_section(rows), feedback


def named_notes_section(rows: list[_SelectedNote] | None) -> tuple[str, dict | None]:
    if not rows:
        return "", ({"sent": []} if rows == [] else None)
    parts = ["## Named knowledge notes\nThe notes named for this work (knowledge_notes), each whole as it stood when "
             "this start was prepared."]
    sent, gaps = [], []
    for entry, note, gap in rows:
        if gap:
            gaps.append({**entry, "reason": gap})
            parts.append(f"### {note_label(entry)}\n[Host: this named note could not be read at this start "
                         f"({gap}); its text is not here.]")
            continue
        topic, scope = entry["topic"], entry["scope"]
        text, revision = note.text, note.revision
        state = "metadata unavailable" if note.parse_error else f"state {note.state}"
        parts.append(f"### {note.title}\nNote: knowledge_read(topic={topic!r}, scope={scope!r})\n"
                     f"Revision: {revision} ({state}, {len(text)} chars)\n\n{text}")
        sent.append({**entry, "revision": revision, "chars": len(text)})
    return "\n\n".join(parts), {key: value for key, value in (("sent", sent), ("gaps", gaps)) if value}


def named_notes_block(entries: Any, *, session: bool) -> str:
    """A scheduled child's own reference to the notes its parent named, frozen into its first input: addresses only,
    as ``knowledge_read`` calls; "" without any, so a child scheduled without them keeps its first input."""
    if not entries:
        return ""
    calls = "\n".join(f"- knowledge_read(topic={entry['topic']!r}, scope={entry['scope']!r})" for entry in entries)
    where = ("These are the default for the initial session start only. Their text is not in this input." if session else
             "Their text is not in this input.")
    return ("[NAMED KNOWLEDGE NOTES]\nThe parent named these knowledge notes for this work (knowledge_notes). "
            f"{where}\n{calls}")


def session_memory_references(ctx: Any, named: str = "") -> str:
    """The marks in force for the session task's room (this room's and the global ones), the knowledge outline
    (the global shelf's and its room's: every note by address and title, people's summaries kept) and the heads of
    the notes that claim the situation names a helper of this work carries and of the notes their heads link
    (``held_notes``, each whole, its reader named unavailable; a declared task names their omission instead), as text
    in a session's start request: what a native helper's view holds, pinned outside the canonical work order and its
    fingerprint, so a session task scheduled before the update starts with the marks and the outline too (it carries
    no situation names, so no heads). The marks follow the room the heads come from: a Presence starter's own once the install records a claim
    (the gate of its helpers' carried names, ``held_notes.carried_situation``); with none, Main's outside a Project.
    The stores are read once as they stand, through the memory
    view's own capture and render (``memory_view._capture_marks``, ``render_room`` over a marks-only snapshot), the
    helper's outline (``context_sections.helper_navigation_sections``) and the heads resolved now, never the
    launcher's own capture. For a new session its already captured ``named`` selection follows, shared or declared.
    ``ctx._session_held_heads`` keeps the heads sent, by address and revision, for the start's custody; nothing else
    is written or activated here. A retry replays the recorded bytes."""
    import dataclasses
    from types import SimpleNamespace

    from ouroboros import held_notes, knowledge
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.context_sections import helper_navigation_sections
    from ouroboros.contracts.task_contract import task_input_sources
    from ouroboros.memory_inventory import membership_facts
    from ouroboros.memory_view import (MemoryViewSnapshot, ROLE_DEFAULTS, _capture_marks, render_room, room_labeler,
                                       view_role, view_room_id)
    from ouroboros.memory_view_events import helper_names
    from ouroboros.tool_access_paths import canonical_data_root

    head = ("\n\nMEMORY DELIVERY (host instructions, separate from canonical work order)\n"
            "What my memory holds for this work as this start was prepared, as a native helper's request holds it: "
            "the marks in force for this work's room and the global ones, then the knowledge outline (every note by "
            "address and title). The calls named beside them are the host's own readers, not yours.\n\n")
    meta = getattr(ctx, "task_metadata", None)
    task = {**(meta if isinstance(meta, Mapping) else {}), "id": str(getattr(ctx, "task_id", "") or "")}
    task["task_contract"] = getattr(ctx, "task_contract", None) or task.get("task_contract")
    ctx._session_held_heads = held = None
    helper = False  # the working-notes line closes held heads and named notes alike
    try:
        root = canonical_data_root(ctx)
        navigation = {"bodies": {}, "names": helper_names(ctx, root)}
        outline = "\n\n".join(helper_navigation_sections(root, str(getattr(ctx, "project_id", "") or ""),
                                                         global_capture=navigation))
        # The heads' inventory decides the gate; view_role reads Presence under metadata, as helper_names passes it.
        presence = (bool(knowledge.active_claims(root, inventory=navigation.get("inventory", ())))
                    and view_role({"metadata": task}) == "presence")
        room = view_room_id(task, ctx, root, project_chat_ids=membership_facts(root).project_chat_ids,
                            presence=presence)
        spec = dataclasses.replace(ROLE_DEFAULTS["child"], room_id=room)
        store = ChronicleStore(root)  # a chronicle never activated holds no mark; nothing activates it here
        marks = tuple(_capture_marks(store, spec, room_labeler(root))) if store.log_path.exists() else ()
        marks_text = render_room(MemoryViewSnapshot(spec=spec, store_status={"state": "active"}, frontier={}, marks=marks))
        if task_input_sources(task) == "declared":
            if topics := held_notes.claimed_topics(root, navigation["names"], navigation.get("inventory", ())):
                held = ("[Host: this start does not automatically add heads of room-claiming notes or of the notes "
                        "those heads link, because its input is declared: " + ", ".join(topics) + ".]")
        elif entries := held_notes.turn_sections(root, task, spec, navigation, ctx=SimpleNamespace(
                held_reader_unavailable=held_notes.EXTERNAL_READER)):  # a fresh namespace, never the launcher's capture
            held, helper = "\n\n".join(part.full_text for part in entries), True
            ctx._session_held_heads = held_notes.sent_heads(entries) or None
    except Exception as exc:  # an unreadable store is a named gap in the start, never a withheld start
        log.warning("session references: my memory could not be read at this start", exc_info=True)
        return head + "\n\n".join(part for part in (
            f"My memory could not be read here ({type(exc).__name__}); this start carries no marks and no outline.",
            named, named and held_notes.HELPER_NOTE) if part)
    return head + "\n\n".join(part for part in (
        marks_text or "No marks are in force for this work's room or globally.", outline, held, named,
        (helper or named) and held_notes.HELPER_NOTE) if part)


def session_host_references(ctx: Any, *, target_root: str, source_root: str,
                            private_copy: bool = False, payload_skill: str = "", service: Any = None,
                            start_body: str | None = None, named_notes: str = "") -> str:
    """Host references pinned in a session's start request, computed once at first start.

    Own-body DESIGN (bible whole, the rest by headings) and BIBLE path, the owner's standing rules for every session
    (declared included), marks, outline, held heads and any named notes for this start (``session_memory_references``,
    outside the work order's fingerprint), and the installed instruction-skill index for any workspace;
    a skill-payload builder also gets the serving host's skill contract, whatever its input
    selection, and any other private copy its environment facts: dependency files read at the
    copy's repository root, the environment placed where the session may write (``source_root``,
    possibly a folder below that root, or the scratch ``service`` grants); a read-only session
    has no copy, so it gets neither. A retry replays the recorded bytes. ``start_body`` (a start
    that continues no run: the prompt and instructions composed so far) closes the text with the
    start tail: my human's rows of the starter's set that body does not hold (``unseen_words_section``),
    so the interval between a session task's scheduling and its first start loses nothing; a
    continuation's rows ride its own prompt instead (``delegate_continuation``)."""
    from ouroboros.safety_config import owner_rules_section
    from ouroboros.skill_catalogue import session_skill_index, skill_contract_references  # lazy skill reads

    text = (session_design_navigation(ctx, target_root=target_root, source_root=source_root)
            + session_constitution_pointer(ctx, target_root=target_root, source_root=source_root)
            + owner_rules_section(session=True) + session_memory_references(ctx, named_notes)
            + session_skill_index(ctx))
    tail = unseen_words_section(ctx, start_body, START_TAIL_FACT) if start_body is not None else ""
    tail = f"\n\n{tail}" if tail else ""
    if payload_skill:
        from ouroboros.body_candidate import serving_repo_dir_for

        return text + skill_contract_references(serving_repo_dir_for(ctx)) + tail
    if private_copy:
        from pathlib import Path

        from ouroboros.test_environment import git_ancestor
        from ouroboros.tool_access_paths import canonical_data_root
        from ouroboros.workspace_preflight import session_environment_facts

        start = Path(source_root).resolve()  # the session's execution root: its only write place in the copy
        root, body = _own_body_source(ctx, target_root, source_root)  # the copy's repository root
        root = root if body else git_ancestor(start) or start
        return text + session_environment_facts(root, data_root=canonical_data_root(ctx), body=body, write_root=start,
                                                **({"service": service} if service is not None else {})) + tail
    return text + tail


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
    "start_binding_fingerprints", "work_order_fingerprint", "work_order_room_words", "unseen_words_section",
]
