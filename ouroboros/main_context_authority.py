"""Continuation views: the predecessor brief, the predecessor mind's notes, Main's copy.

Every consumer of a named predecessor — a Continue root, a delegated child, an
external work order — carries the same compact brief
(``project_helper_predecessor_authority``): identity and digest, status and cause,
the terminal notice, the owner's words verbatim and every omitted field with its
size, beside the exact reader of the complete record. The predecessor mind's
surviving authored notes travel separately, as one foldable conversation row after
the assignment (``predecessor_notes_message``), never inside the frozen Runtime
prefix. A fresh head an accepted late quiz answer started reads the asking task's
notes the same way, as information without authority (``late_answer_notes_message``).
No ancestor chain is walked: each successor reads its own predecessor through the
reader. Neither projection mutates canonical authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any, Dict, Optional

from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
from ouroboros.store import StoreError

# The typed kinds of the notes rows (``_host_context_kind``): plain host prose rows
# the mind folds like any dialogue unit; the key never reaches the wire.
PREDECESSOR_NOTES_NAME = "predecessor_notes"
LATE_ANSWER_NOTES_NAME = "late_answer_asker_notes"
#: Task-result field naming the mind's surviving authored notes after its latest applied fold.
WORKING_NOTES_FIELD = "working_notes"
#: When that fold named them: the host's clock at the write, kept while the same notes stay named.
WORKING_NOTES_AT_FIELD = "working_notes_at"
_SOURCE_ERRORS = (OSError, ValueError, KeyError, TypeError, StoreError)


def project_helper_predecessor_authority(
    authority: Any, *, declared: bool = False,
) -> Dict[str, Any]:
    """Pure, idempotent brief of an envelope, or its declared-input reference.

    The existing contract normalizer handles legacy bodies. Keep its envelope
    kind: a bounded preview's wrapper can exceed the per-field wire allowance,
    and relabelling it would make a rebuild collapse the preview and its digest
    again. Answers, contracts and review history stay behind the exact reader.
    Omission sizes count serialized characters (without a string's quotes);
    old name-only omissions have unknown sizes, represented by null.

    The successor's startup materializes it, a delegated child's contract applies
    it after the parent contract spread, and direct work-order sessions apply it
    to their own contract, so every consumer receives the same brief. The brief
    does not carry the omitted evidence; its source names the reader for the
    full result.
    """
    if not isinstance(authority, Mapping) or not authority:
        return {}
    from ouroboros.contracts.task_contract import build_task_contract

    envelope = build_task_contract({"predecessor_authority": authority})["predecessor_authority"]
    # A named plan/review source gap is part of the reference itself: declared briefs keep it too.
    keep = {"kind", "source", "task_id", "previous_task_id", "authority_sha256", "authority_chars",
            "digest_semantics", "plan_review_source_gap"}
    if not declared:
        # Keep how the work ended and the owner's words; history stays at its source.
        keep.update({"status", "execution_status", "reason_code", "terminal_origin", "terminal_host_notice",
                     "outcome_axes",
                     "origin_message_text", "origin_message_ref"})
    prior = envelope.get("omitted_fields")
    omitted = {}
    if isinstance(prior, Mapping):
        omitted = copy.deepcopy(dict(prior))
    elif isinstance(prior, list):
        omitted = {key: None for key in prior if isinstance(key, str)}
    for key, value in envelope.items():
        if key not in keep and key != "omitted_fields":
            omitted[key] = len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)) - (
                2 if isinstance(value, str) else 0)
    return {
        **{key: copy.deepcopy(value) for key, value in envelope.items() if key in keep},
        "omitted_fields": dict(sorted(omitted.items())),
    }


def _authority_identity(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    source = value.get("source")
    if isinstance(source, Mapping):
        source_id = str(source.get("task_id") or "").strip()
        if source_id:
            return source_id
        arguments = source.get("arguments")
        if isinstance(arguments, Mapping):
            source_id = str(arguments.get("task_id") or "").strip()
            if source_id:
                return source_id
    return str(value.get("task_id") or "").strip()


def _shown_elsewhere(text: str, representation: str, rule: str) -> Dict[str, Any]:
    return {"representation": representation, "chars": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "rule": rule}


def _assignment_form(value: Any) -> str:
    """The objective form ``build_task_contract`` stores (stripped), applied to both compared sides."""
    return str(value or "").strip()


def _assignment_reference(objective: str, task: Mapping[str, Any]) -> Dict[str, Any]:
    """The objective as a reference to the assignment as a request actually delivers it.

    The first user turn carries the assignment whole, or, when the whole input cannot fit a
    request, a pointer to its exact stored source: a wake's observation
    (``loop_model_call._project_wake_input``), an owner-Nano input
    (``context_input_selection.task_input_pointer``). This view is captured before either
    decision, so the rule names both deliveries, and a wake whose observation is already
    bound names that stored source itself (``consciousness_wake.bind_wake_observation``).
    """
    from ouroboros.consciousness_wake import WAKE_OBSERVATION_KEY

    rule = ("The objective is this task's assignment, the first user turn of this conversation; when the "
            "whole input cannot fit a request, that turn is a pointer to its exact stored source instead. "
            "It is not repeated here.")
    metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
    wake = metadata.get(WAKE_OBSERVATION_KEY)
    source = wake.get("source") if isinstance(wake, Mapping) else None
    if not isinstance(source, Mapping) or not isinstance(source.get("read"), Mapping):
        return _shown_elsewhere(objective, "assignment", rule)
    return {**_shown_elsewhere(objective, "assignment", rule + " This wake's complete observation is stored "
                               "at `source`, whichever way the turn delivers it."),
            "source": {key: copy.deepcopy(source[key]) for key in ("read", "sha256", "lines") if key in source}}


def project_main_task_authority(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the provider-only authority section without mutating ``task``.

    The materialized predecessor (the brief) appears once, at the top level; an
    equal copy nested in the task's own contract is dropped from this view. A
    contract objective equal to the assignment — the first user turn of the
    conversation — and a contract context byte-identical to the task block's
    own ``context`` are shown as typed references, so the work order and its
    context appear once per request. The objective comparison applies the
    contract's own normalization to both sides (``_assignment_form``): a task
    text ending in a line break (a wake's rendered template) is still the one
    assignment. The context needs none, since the contract keeps it verbatim.
    """
    projection: Dict[str, Any] = {}
    predecessor = task.get("predecessor_authority")
    contract = task.get("task_contract")
    if isinstance(predecessor, Mapping) and predecessor:
        projection["predecessor_authority"] = project_helper_predecessor_authority(predecessor)
    if isinstance(contract, Mapping):
        projected_contract = copy.deepcopy(dict(contract))
        nested = projected_contract.get("predecessor_authority")
        if (isinstance(nested, Mapping) and isinstance(predecessor, Mapping)
                and _authority_identity(nested) == _authority_identity(predecessor)):
            projected_contract.pop("predecessor_authority", None)
        elif isinstance(nested, Mapping):
            projected_contract["predecessor_authority"] = project_helper_predecessor_authority(nested)
        objective = projected_contract.get("objective")
        if (isinstance(objective, str) and _assignment_form(objective)
                and _assignment_form(objective) == _assignment_form(task.get("text"))):
            projected_contract["objective"] = _assignment_reference(objective, task)
        context = projected_contract.get("context")
        if isinstance(context, str) and context and context == task.get("context"):
            projected_contract["context"] = _shown_elsewhere(context, "task_context", (
                "The caller's context is the `context` field of the task block above; it is not repeated here."))
        projection["task_contract"] = projected_contract
    origin_ref = task.get("origin_message_ref")
    origin_text = task.get("origin_message_text")
    if isinstance(origin_ref, Mapping) and origin_ref:
        projection["task_authority_origin"] = {
            "ref": copy.deepcopy(dict(origin_ref)),
            **({"text": str(origin_text)} if isinstance(origin_text, str) and origin_text else {}),
        }
    if isinstance(task.get("authority_historical_gaps"), list):
        projection["authority_historical_gaps"] = copy.deepcopy(task["authority_historical_gaps"])
    return projection


def _materialized_predecessor(task: Mapping[str, Any]) -> str:
    """The predecessor a task's own start materialized (the exact host-issued pointer), else ``""``."""
    authority = task.get("predecessor_authority")
    source = authority.get("source") if isinstance(authority, Mapping) else None
    owner = str(source.get("task_id") or "") if isinstance(source, Mapping) else ""
    from ouroboros.agent_startup_checks import valid_task_result_authority_source

    return owner if valid_task_result_authority_source(source, owner) else ""


def _kept_notes(drive_root: Any, owner: str, saved: Mapping[str, Any], *, cleared_is_final: bool = False) -> Any:
    """``(texts, exact source)`` of the notes task ``owner``'s result ``saved`` names, ``None`` when
    it kept none, or the error that stopped the search of its chat log.

    The notes are the actor capsules still in its view after its latest applied fold
    (``WORKING_NOTES_FIELD``, ``publish_working_notes``) — every note its mind authored and kept,
    in order; on records written before that field, the capsules of its selected review-history
    view, else its authored narrative stored beside the result or in the chat log
    (``project_dialogue.resolve_legacy_continuation_narrative``). ``cleared_is_final``: a field
    that fold cleared stays cleared, so no older account returns in place of notes the mind
    retired. A named source that cannot be read raises.
    """
    from ouroboros import review_history_view as view
    from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path
    from ouroboros.project_dialogue import continuation_narrative_is_valid, resolve_legacy_continuation_narrative

    reader = f"get_task_result(task_id={owner!r}, include_authority=True)"

    def source(ref: Mapping[str, Any], field: str) -> str:
        # The immutable handle belongs to owner, not the receiving task's artifact_store.
        path = task_artifact_dir_path(drive_root, owner, create=False) / str(ref["path"])
        read = {"tool": "read_file", "arguments": {"root": "runtime_data", "path": str(path)}}
        return (f"{reader} -> {field}; read from this task with {json.dumps(read, sort_keys=True)}; "
                f"original task-local reference (its relative reader is scoped to {owner}): "
                f"{json.dumps(ref, ensure_ascii=False, sort_keys=True)}")

    working = saved.get(WORKING_NOTES_FIELD)
    if isinstance(working, Mapping) or (cleared_is_final and working is not None):
        texts = [str(text) for text in json.loads(read_actor_source_bytes(drive_root, owner, working))["notes"]]
        return texts, source(working, WORKING_NOTES_FIELD)
    if cleared_is_final and WORKING_NOTES_FIELD in saved:
        return None
    selection = saved.get(view.SELECTED_VIEW_FIELD)
    if selection:
        record = view.load_review_history_view(selection, lambda ref: read_actor_source_bytes(drive_root, owner, ref))
        return ([capsule["content"][0]["text"] for capsule in view.selected_actor_capsules(record)],
                source(selection["source_ref"], view.SELECTED_VIEW_FIELD))
    narrative = saved.get("continuation_narrative")
    if continuation_narrative_is_valid(narrative, owner):
        return [str(narrative["text"])], f"{reader} -> continuation_narrative"
    try:  # a record earlier versions wrote only to the chat log, read by its exact task identity
        legacy = resolve_legacy_continuation_narrative(
            drive_root, owner, {"kind": "task_result", "task_id": owner, "reader": "get_task_result"})
    except _SOURCE_ERRORS as exc:
        return exc
    if not continuation_narrative_is_valid(legacy, owner):
        return None
    return [str(legacy["text"])], f"the chat-log row {json.dumps(legacy['source'], ensure_ascii=False, sort_keys=True)}"


def predecessor_notes_message(task: Mapping[str, Any], *, drive_root: Any) -> Optional[Dict[str, Any]]:
    """The predecessor mind's surviving authored notes as ONE foldable row after the assignment.

    Only the task that materialized a named predecessor at its own start carries it
    (the successor); a child's contract spread carries the brief and its reader alone.
    The notes are ``_kept_notes``'s, its legacy sources included. The row is a plain
    host prose row (``_host_context_kind``): the mind folds it like any dialogue unit,
    the fold's checkpoint retains its exact text, and the header names the exact
    source. A pointer whose source cannot be read yields a typed gap row; a
    predecessor that kept no notes yields none (the brief's ``omitted_fields`` and
    reader remain).
    """
    owner = _materialized_predecessor(task)
    if not owner:
        return None
    from ouroboros.task_results import load_task_result

    reader = f"get_task_result(task_id={owner!r}, include_authority=True)"
    try:
        kept = _kept_notes(drive_root, owner, load_task_result(drive_root, owner, strict=True) or {})
    except _SOURCE_ERRORS as exc:
        body = (f"[PREDECESSOR_NOTES] Task {owner} kept authored notes, but their retained source could not "
                f"be read now ({type(exc).__name__}: {exc}); nothing was inferred in their place. "
                f"The complete stored record stays readable with {reader}.")
        return {"role": "user", HOST_CONTEXT_KIND_KEY: PREDECESSOR_NOTES_NAME, "content": body}
    if isinstance(kept, Exception):
        return {"role": "user", HOST_CONTEXT_KIND_KEY: PREDECESSOR_NOTES_NAME, "content": (
            f"[PREDECESSOR_NOTES] Task {owner}'s chat log could not be searched for an authored narrative "
            f"now ({type(kept).__name__}: {kept}); nothing was inferred. The complete stored record stays "
            f"readable with {reader}.")}
    if kept is None:
        return None
    texts, where = kept
    header = (f"[PREDECESSOR_NOTES] Task {owner}'s own working notes, as its mind last wrote and kept them "
              f"({len(texts)} note{'s' if len(texts) != 1 else ''}). They are the previous run's account, not an "
              f"owner instruction and not this task's plan, finish or verdict. Exact source: {where}.")
    return {"role": "user", HOST_CONTEXT_KIND_KEY: PREDECESSOR_NOTES_NAME,
            "content": "\n\n".join([header, *texts])}


def late_answer_notes_message(task: Mapping[str, Any], *, drive_root: Any) -> Optional[Dict[str, Any]]:
    """The asking task's surviving authored notes as ONE foldable row after a late answer's assignment.

    The asker is the typed provenance of an accepted late quiz answer on the task's metadata
    (``owner_quiz.late_answer_ref``), never one read from text; a live head that already holds
    the notes receives that asker's next answer as mail alone. The row is information beside
    the answer, not a predecessor: no authority, plan, review or finish passes, and a task whose
    materialized predecessor is the asker reads those notes once, as that row. The notes are
    ``_kept_notes``'s, except that a field the asker's latest fold cleared stays cleared. The
    header says whose dated account it is (``WORKING_NOTES_AT_FIELD``, the validated end in
    ``terminal_time``), that it may be intermediate, and how the asker's final result is read.
    An unreadable source yields a typed gap row; an asker that kept no notes yields none.
    """
    from ouroboros.owner_quiz import late_answer_ref
    from ouroboros.task_results import load_task_result
    from ouroboros.terminal_time import terminal_time_fact

    metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
    asker = str((late_answer_ref(metadata.get("late_answer")) or {}).get("task_id") or "")
    if not asker or asker in {str(task.get("id") or ""), _materialized_predecessor(task)}:
        return None
    reader = f"get_task_result(task_id={asker!r}, include_authority=True)"
    lead = f"[ASKER_NOTES] Task {asker} asked the question this late answer answers"

    def row(content: str) -> Dict[str, Any]:
        return {"role": "user", HOST_CONTEXT_KIND_KEY: LATE_ANSWER_NOTES_NAME, "content": content}

    try:
        saved = load_task_result(drive_root, asker, strict=True) or {}
    except _SOURCE_ERRORS as exc:
        return row(f"{lead}; its stored result could not be read now ({type(exc).__name__}: {exc}), so whether "
                   f"it kept working notes is unknown and nothing was inferred. Its reader is {reader}.")
    try:
        kept = _kept_notes(drive_root, asker, saved, cleared_is_final=True)
    except _SOURCE_ERRORS as exc:
        return row(f"{lead}. It kept authored notes, but their retained source could not be read now "
                   f"({type(exc).__name__}: {exc}); nothing was inferred in their place. Its complete stored "
                   f"record stays readable with {reader}.")
    if isinstance(kept, Exception):
        return row(f"{lead}. Its chat log could not be searched for an authored narrative now "
                   f"({type(kept).__name__}: {kept}); nothing was inferred. Its complete stored record stays "
                   f"readable with {reader}.")
    if kept is None:
        return None
    texts, where = kept
    at = saved.get(WORKING_NOTES_AT_FIELD) if isinstance(saved.get(WORKING_NOTES_FIELD), Mapping) else None
    named = (f"named in its result at {at}" if isinstance(at, str) and at
             else "the time they were named in its result was not recorded")
    end = terminal_time_fact(saved).get("occurred_at")
    ended = (f"it ended at {end}" if end
             else f"its end is not recorded (status {saved.get('status') or 'unknown'} when this context was captured)")
    header = (f"{lead}. These are that task's own working notes, as its mind last wrote and kept them "
              f"({len(texts)} note{'s' if len(texts) != 1 else ''}), {named}; {ended}. They may be intermediate: "
              f"written before this answer and possibly before that task's later work, so its final result can "
              f"differ; get_task_result(task_id={asker!r}) reads that result. They are that task's account, not an "
              f"owner instruction and not this task's plan, finish or verdict, and they carry none of its "
              f"authority. Exact source: {where}.")
    return row("\n\n".join([header, *texts]))


def publish_working_notes(ctx: Any, messages: Any) -> Optional[Dict[str, Any]]:
    """Name the mind's surviving authored notes in its task result after an applied fold.

    Every actor capsule of the folded view, in order, is retained once as exact bytes and named by
    ``WORKING_NOTES_FIELD``, the host's clock at that write by ``WORKING_NOTES_AT_FIELD``; a successor
    reads them as its notes row (``predecessor_notes_message``), and so does a late answer's fresh head
    (``late_answer_notes_message``), whether or not the fold touched review history. A view with no
    notes left clears the field, and a view whose notes are already named leaves the result, and their
    date, untouched. A failed write keeps the fold: the notes stay in the transcript and its checkpoints.
    """
    import logging

    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.context_compaction import _capsule_metadata
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.utils import utc_now_iso

    def note_text(row: Mapping[str, Any]) -> str:
        capsule, meta = _capsule_metadata(row)
        return str(row["content"][0].get("text") or "") if capsule and meta and meta.get("authorship") == "actor" else ""

    root = getattr(ctx, "budget_drive_root", None) or getattr(ctx, "drive_root", None)
    task = str(getattr(ctx, "task_id", "") or "")
    if not root or not task:
        return None  # no durable task result to name them in (a bare tool context)
    texts = [text for text in (note_text(row) for row in messages) if text.strip()]
    try:
        ref = None
        if texts:
            raw = json.dumps({"task_id": task, "notes": texts}, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ref = store_actor_source_bytes(root, task, category="context_checkpoints",
                                           source_id="working-notes-" + hashlib.sha256(raw).hexdigest(), data=raw,
                                           extension="json")
        if (load_task_result(root, task) or {}).get(WORKING_NOTES_FIELD) == ref:
            return ref  # the same notes are already named (a restore, a schema-only view)
        at = utc_now_iso()
        write_task_result(root, task, "running", strict_existing_dict=True, _field_projector=lambda existing, _new: {
            "status": existing.get("status") or "running", WORKING_NOTES_FIELD: ref, WORKING_NOTES_AT_FIELD: at})
        return ref
    except _SOURCE_ERRORS:
        logging.getLogger(__name__).warning("Working notes of %s were not named in its result", task, exc_info=True)
        return None


__all__ = ["LATE_ANSWER_NOTES_NAME", "PREDECESSOR_NOTES_NAME", "WORKING_NOTES_AT_FIELD", "WORKING_NOTES_FIELD",
           "late_answer_notes_message", "predecessor_notes_message", "publish_working_notes",
           "project_main_task_authority",
           "project_helper_predecessor_authority"]
