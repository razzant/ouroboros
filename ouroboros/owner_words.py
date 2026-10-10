"""The owner's words that caused a work tree: carried by value, resolved for older tasks, rendered verbatim.

A helper's floor is its assignment plus the words of the owner that caused the work
(BIBLE P1: provenance matters). ``governing_words_for_schedule`` computes them once,
when a parent schedules a helper, from what the parent already holds: a child passes
on exactly the words it inherited, so a grandchild sees the root's words unchanged;
any other run passes the owner-sourced rows of its own corpus (``ctx._owner_directives``)
— never ``initial_text``, which is not the owner's — and never reads the chat chain.
The words ride the existing tree-origin channel (``origin_metadata``) into every
descendant's ``task.metadata`` under ``FIELD``. A run with no owner words carries
``ABSENT_FIELD`` set to the raw ``run_origin`` marker (``initiator=consciousness``) or
``not_recorded``: no vocabulary of reasons of its own.

``task_owner_words`` resolves a task scheduled before the field existed, for readers
that run after the task (the fallback page writer's input, the measurement rig), never
on a helper's start path. Carriers are tried in order and the first that yields a row
is taken: the task's ingress origin text or ref, its Project binding, its routing
annotations, its mailbox, then the marker. An annotation counts only through the
inbound chat row with the same client message id: Ouroboros writes steer receipts of
its own, and nothing inbound joins to them.

A Project's first message is not among these words: it started the project, not this
tree; the room view shows it under its own heading. Nothing here calls a model.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Mapping, Tuple

log = logging.getLogger(__name__)

OWNER_SOURCES = frozenset({"initial_user", "owner_mailbox", "owner_quiz_answer", "origin_message",
                           "owner_corpus", "direct_incoming"})
FIELD, ABSENT_FIELD = "governing_owner_words", "governing_owner_words_absent"
NOT_RECORDED = "not_recorded"
ROW_KEYS = ("text", "source", "carrier", "task_id", "ts", "ref")
_RUN_ORIGIN_MARKERS = ("initiator", "source", "task_type")
_ANNOTATED_ACTS = frozenset({"promote_chat_to_task", "route_to_project", "steer_task", "mailbox_delivery"})
_ANNOTATED_STATUSES = frozenset({"scheduled", "delivered"})
_ORIGIN = " from task {root}"
_HEADINGS = {
    "child": "## Words of my human that caused this work (verbatim)",
    "session": ("OWNER WORDS THAT CAUSED THIS WORK (verbatim; your assignment is the task given to you, "
                "these words are why it exists)"),
    "reviewer": "## Words of my human that caused this work (verbatim, host-attested)",
    "plan": "## Words of my human that caused this work (verbatim, host-attested)",
    "writer": "## Words of my human that caused this work (verbatim)",
}
_FRAMES = {
    "child": ("Host fact, carried by value{origin}. Your assignment is the parent's message; "
              "these words are why it exists."),
    "session": "Host fact, carried by value{origin}.",
    "reviewer": "They state what was asked; the goal above is the author's account of this change.",
    "plan": "They state what was asked; the objective above is the author's account of this task.",
    "writer": "Host fact, read from what the host recorded{origin}; these words are why the work exists.",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _row(text: Any, *, source: Any, carrier: str, task_id: Any = "", ts: Any = "", ref: Any = "") -> Dict[str, str]:
    return {"text": str(text or ""), "source": str(source or ""), "carrier": carrier,
            "task_id": str(task_id or ""), "ts": str(ts or ""), "ref": str(ref or "")}


def _chat_ref(source: Mapping[str, Any]) -> str:
    client_id = str(source.get("client_message_id") or "")
    return f"chat {source.get('chat_id', 1)}" + (f" / {client_id}" if client_id else "")


def _deduped(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Drop empty rows and repeats of the same whitespace-normalized text (the ingress checksum)."""
    from ouroboros.project_dialogue import _text_sha256

    seen: set = set()
    kept = []
    for row in rows:
        key = _text_sha256(row["text"])
        if row["text"].strip() and key not in seen:
            seen.add(key)
            kept.append(row)
    return kept


def _absent_marker(record: Mapping[str, Any] | None) -> str:
    from ouroboros.dialogue_provenance import run_origin

    origin = run_origin(record)
    return next((f"{key}={origin[key]}" for key in _RUN_ORIGIN_MARKERS if origin.get(key)), NOT_RECORDED)


def _ctx_metadata(ctx: Any) -> Mapping[str, Any]:
    return _mapping(getattr(ctx, "task_metadata", None))


def directive_owner_rows(ctx: Any) -> List[Dict[str, str]]:
    """The owner-sourced rows of the run's retained corpus, verbatim (images by reference).

    The door-stamped first message is the door's own record of it (``origin_message_text``)
    when the door kept one: the host may have prefixed a notice of its own to that user
    turn (a Swarm initiative), never to the door's text.
    """
    from ouroboros.review_evidence_sections import _owner_content_projection

    origin_ref = _mapping(_ctx_metadata(ctx).get("origin_message_ref"))
    door_text = _ctx_metadata(ctx).get("origin_message_text")
    door_text = door_text if isinstance(door_text, str) and door_text.strip() else None
    task_id = str(getattr(ctx, "task_id", "") or "")
    rows = []
    for item in getattr(ctx, "_owner_directives", None) or []:
        if not isinstance(item, dict) or item.get("source") not in OWNER_SOURCES:
            continue
        stamped = item["source"] == "initial_user" and origin_ref
        msg_id = str(item.get("msg_id") or "")
        content = door_text if stamped and door_text is not None else item.get("content")
        rows.append(_row(_owner_content_projection(content), source=item["source"], carrier="ctx",
                         task_id=task_id, ts=origin_ref.get("ts") if stamped else "",
                         ref=_chat_ref(origin_ref) if stamped else (f"msg {msg_id}" if msg_id else "")))
    return _deduped(rows)


def task_governing_words(task: Mapping[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    """``(rows, absence marker)`` carried by value, read at the top level or under ``metadata``."""
    source = _mapping(task)
    nested = _mapping(source.get("metadata"))
    holder = source if (FIELD in source or ABSENT_FIELD in source) else nested
    raw = holder.get(FIELD)
    rows = [_row(item.get("text"), source=item.get("source"), carrier=str(item.get("carrier") or ""),
                 task_id=item.get("task_id"), ts=item.get("ts"), ref=item.get("ref"))
            for item in (raw if isinstance(raw, list) else []) if isinstance(item, Mapping)]
    return [row for row in rows if row["text"].strip()], str(holder.get(ABSENT_FIELD) or "")


def governing_words_for_schedule(ctx: Any) -> Dict[str, Any]:
    """``{FIELD: rows}`` or ``{ABSENT_FIELD: marker}`` for the helper this run is about to schedule."""
    metadata = _ctx_metadata(ctx)
    if str(metadata.get("delegation_role") or "") == "subagent":
        rows, absent = task_governing_words(metadata)
    else:
        rows, absent = directive_owner_rows(ctx), ""
    if rows:
        return {FIELD: rows}
    return {ABSENT_FIELD: absent or _absent_marker({"type": getattr(ctx, "current_task_type", None),
                                                    "metadata": dict(metadata)})}


def _inbound_rows(drive_root: Any, client_ids: Mapping[str, str]) -> List[Dict[str, Any]]:
    """Owner rows of the chat chain with these client message ids, in chain order."""
    from ouroboros.chat_chain import chat_chain_paths
    from ouroboros.utils import iter_jsonl_objects

    return [row for path in chat_chain_paths(drive_root) for row in iter_jsonl_objects(path)
            if str(row.get("client_message_id") or "") in client_ids and _owner_row(row)]


def _owner_row(row: Mapping[str, Any] | None) -> bool:
    """The owner-source cut: inbound, non-empty, not a system/presence/skill-repair injection."""
    return bool(row and row.get("direction") == "in" and str(row.get("text") or "").strip()
                and not (row.get("system_type") or row.get("presence") or row.get("source") == "skill_repair"))


def _message_rows(drive_root: Any, task_id: str, text: Any, ref: Any, carrier: str) -> List[Dict[str, str]]:
    ref = _mapping(ref)
    if isinstance(text, str) and text.strip():
        return [_row(text, source="origin_message", carrier=carrier, task_id=task_id, ts=ref.get("ts"),
                     ref=_chat_ref(ref) if ref else "")]
    if not ref:
        return []
    from ouroboros.project_dialogue import resolve_owner_message_source

    row = resolve_owner_message_source(drive_root, dict(ref))
    if not _owner_row(row):
        return []
    return [_row(row["text"], source="origin_message", carrier=carrier, task_id=task_id, ts=row.get("ts"),
                 ref=_chat_ref(row))]


def _origin_carrier(drive_root: Any, task_id: str, record: Mapping[str, Any]) -> List[Dict[str, str]]:
    nested = _mapping(record.get("metadata"))
    return _message_rows(drive_root, task_id,
                         record.get("origin_message_text") or nested.get("origin_message_text"),
                         record.get("origin_message_ref") or nested.get("origin_message_ref"), "origin_message")


def _binding_carrier(drive_root: Any, task_id: str, record: Mapping[str, Any]) -> List[Dict[str, str]]:
    from ouroboros.projects_registry import project_task_bindings

    binding = _mapping(project_task_bindings(drive_root).get(task_id))
    return _message_rows(drive_root, task_id, binding.get("source_text"), binding.get("source_ref"), "binding")


def _annotation_carrier(drive_root: Any, task_id: str, record: Mapping[str, Any]) -> List[Dict[str, str]]:
    from ouroboros.project_dialogue import _ANNOTATIONS_NAME, _latest_annotations_by_token

    acts: Dict[str, str] = {}
    for (client_id, _token), row in _latest_annotations_by_token(
            pathlib.Path(drive_root) / "logs" / _ANNOTATIONS_NAME).items():
        if (row.get("action") in _ANNOTATED_ACTS and row.get("status") in _ANNOTATED_STATUSES
                and str(row.get("target") or "") == task_id):
            acts.setdefault(client_id, str(row["action"]))
    return [_row(row["text"], source=acts[str(row["client_message_id"])], carrier="annotation", task_id=task_id,
                 ts=row.get("ts"), ref=_chat_ref(row)) for row in (_inbound_rows(drive_root, acts) if acts else [])]


def _mailbox_carrier(drive_root: Any, task_id: str, record: Mapping[str, Any]) -> List[Dict[str, str]]:
    from ouroboros.owner_mailbox import KIND_OWNER_TEXT, drain_owner_entries

    try:
        entries = drain_owner_entries(pathlib.Path(drive_root), task_id, include_acknowledged=True)
    except (OSError, ValueError):
        return []
    return [_row(entry.get("text"), source="owner_mailbox", carrier="mailbox", task_id=task_id, ts=entry.get("ts"),
                 ref=f"msg {entry.get('msg_id')}") for entry in entries if entry.get("kind") == KIND_OWNER_TEXT]


def _task_record(drive_root: Any, task_id: str) -> Dict[str, Any]:
    if not task_id:
        return {}
    from ouroboros.task_status import load_effective_task_result

    try:
        record = load_effective_task_result(drive_root, task_id, materialize_artifacts=False)
    except Exception:
        log.debug("owner words: task record unreadable for %s", task_id, exc_info=True)
        return {}
    return dict(record) if isinstance(record, dict) else {}


def task_owner_words(drive_root: Any, task_id: str, *,
                     task: Mapping[str, Any] | None = None) -> Tuple[List[Dict[str, str]], str]:
    """The words for a task: its carried field, else the first carrier of its root that yields a row."""
    record = dict(task) if isinstance(task, Mapping) else _task_record(drive_root, task_id)
    rows, absent = task_governing_words(record)
    if rows or absent:
        return rows, absent
    task_id = str(task_id or "")
    root_id = str(record.get("root_task_id") or _mapping(record.get("metadata")).get("root_task_id") or task_id)
    if root_id != task_id:
        task_id, record = root_id, (_task_record(drive_root, root_id) or record)
    for carrier in (_origin_carrier, _binding_carrier, _annotation_carrier, _mailbox_carrier):
        rows = _deduped(carrier(drive_root, task_id, record))
        if rows:
            return rows, ""
    return [], _absent_marker(record)


def render_owner_words(rows: Any, absent: str = "", *, audience: str, root_task_id: str = "", indent: str = "") -> str:
    """One verbatim section for a child, a session, a reviewer, a plan review or the page writer.

    ``indent`` prefixes every line of the words (the memory view's two spaces), so a line of
    the owner's own that starts with ``## `` never reads as a section of the request.
    """
    if audience not in _HEADINGS:
        raise ValueError(f"unknown owner-words audience {audience!r}; expected one of {sorted(_HEADINGS)}")
    rows = [row for row in (rows if isinstance(rows, list) else []) if isinstance(row, Mapping)
            and str(row.get("text") or "").strip()]
    if not rows:
        return f"No words of my human are recorded for this work (host marker: {absent})." if absent else ""
    origin = _ORIGIN.format(root=root_task_id) if root_task_id else ""
    blocks = []
    for row in rows:
        source, task_id = str(row.get("source") or ""), str(row.get("task_id") or "")
        stamp = [str(row.get("ts") or ""), "owner", f"{source} of task {task_id}" if task_id else source,
                 str(row.get("ref") or "")]
        words = "\n".join(indent + line if line else line for line in str(row["text"]).split("\n"))
        blocks.append("[" + " · ".join(part for part in stamp if part) + "]\n" + words)
    return f"{_HEADINGS[audience]}\n{_FRAMES[audience].format(origin=origin)}\n" + "\n\n".join(blocks)


def owner_words_text(ctx: Any, *, audience: str = "reviewer") -> str:
    """``render_owner_words`` over the words this run would hand a helper."""
    words = governing_words_for_schedule(ctx)
    root = str(_ctx_metadata(ctx).get("root_task_id") or getattr(ctx, "task_id", "") or "")
    return render_owner_words(words.get(FIELD) or [], str(words.get(ABSENT_FIELD) or ""),
                              audience=audience, root_task_id=root)
