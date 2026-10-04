"""Room evidence over the canonical history, progress and addressed mailbox.

This is a read projection, not another dialogue store. Immutable copies belong
in the existing task source-handle custody, exactly like operative plan specs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import pathlib
from typing import Any

from ouroboros.project_dialogue import (
    bound_room_chat, entry_matches_source_ref, project_origin_rows,
    room_membership, source_refs_for_project, latest_chat_annotations,
)
from ouroboros.projects_registry import all_task_bindings, list_reserved_projects
from ouroboros.utils import jsonl_archive_segments, jsonl_generation_signature, utc_now_iso

# Structural producer fields, not a semantic selection of what counts as an
# instruction. Binary media is deliberately represented by its delivered name.
_ROW_FACTS = (
    "ts", "chat_id", "direction", "type", "source", "task_id", "parent_task_id",
    "root_task_id", "client_message_id", "msg_id", "sender_label", "sender_session_id",
    "source_task_id", "relayed_from_task_id", "provenance", "client_surface",
    "status", "reason_code", "outcome", "outcome_axes", "terminal_host_notice", "task_terminal_status", "summary_kind",
    "lifecycle", "quiz", "actions", "title", "filename", "caption", "mime",
    "project_origin_projection", "presence_provenance", "attachment_gap",
)


def _chat_id(row: dict) -> int:
    try:
        return int(row.get("chat_id", 1))
    except (TypeError, ValueError):
        return 1  # The same legacy missing-address convention as chat_history.


def _row_projection(row: dict, stream: str, ordinal: int, root: Any = None, *, pos: int | None = None,
                    lineage: dict | None = None) -> dict:
    from ouroboros.dialogue_provenance import row_author

    result = {key: row[key] for key in _ROW_FACTS if key in row}
    result.update(stream=stream, source_ordinal=ordinal)
    result["text"] = str(row.get("content", row.get("text", "")) or "")
    # One attribution source, the row's own fields: a quiz answer is the Owner's, a
    # mailbox delivery keeps its own provenance, every other row is signed by its fields
    # and, for a chat row with a stream position, by the activation's lineage epoch.
    if stream == "mailbox" and row.get("type") != "quiz_answer":
        result["author"] = str(row.get("provenance") or "Owner")
    else:
        result["author"] = row_author(row, pos=pos, **(lineage if pos is not None and lineage else {}))["label"]
    # Payload bytes are never interpreted as dialogue. Existing attachment
    # manifests carry the owner-visible file names alongside custody handles.
    attachments = row.get("attachment_manifest")
    if row.get("attachment_manifest_ref") and root is not None:
        from ouroboros.artifacts import resolve_attachment_manifest
        try:
            attachments = resolve_attachment_manifest(root, str(row.get("task_id") or ""), row)
        except (OSError, ValueError) as exc:
            result["attachment_gap"] = type(exc).__name__
    if isinstance(attachments, list):
        result["attachments"] = [
            {key: item[key] for key in ("label", "name", "filename", "original_filename", "mime", "size", "sha256", "status", "reason") if key in item}
            for item in attachments if isinstance(item, dict)
        ]
    return result


def _progress_source(root: pathlib.Path, matches) -> tuple[list, dict]:
    from ouroboros.memory import Memory

    live = root / "logs" / "progress.jsonl"
    rows, gaps, generations = [], [], []
    try:
        paths = [*jsonl_archive_segments(live, strict=True), live]
        before = [jsonl_generation_signature(path) for path in paths]
        for path, signature in zip(paths, before):
            if not path.exists():
                continue
            entries, errors = Memory._read_chat_generation(path)
            gaps.extend(errors)
            generations.append({"path": str(path), **signature})
            rows.extend(row for row in entries if matches(_chat_id(row), row))
        after_paths = [*jsonl_archive_segments(live, strict=True), live]
        stable = paths == after_paths and before == [jsonl_generation_signature(path) for path in after_paths]
        if not stable:
            gaps.append({"kind": "progress_changed_during_capture"})
    except OSError as exc:
        stable = False
        gaps.append({"kind": "progress_unreadable", "error": type(exc).__name__})
    return rows, {"generations": generations, "gaps": gaps, "snapshot_stable": stable}


def read_room_source(drive_root: Any, chat_id: int, *, task_id: str = "",
                     mailbox_root: Any = None) -> dict | None:
    """Capture all retained room rows, independent of consolidation cursors.

    Every output line is one JSON record. Escaped newlines preserve verbatim
    message text while making line ranges stable within the captured bytes.
    Missing generations remain explicit coverage facts, never invented history.
    """
    from ouroboros.memory import Memory, _history_timestamp

    root, chat_id = pathlib.Path(drive_root), int(chat_id)
    projects = {int(p["chat_id"]): p for p in list_reserved_projects(root)}
    if chat_id < 0:
        return None
    bindings = all_task_bindings(root)
    refs = source_refs_for_project(root, chat_id) if chat_id in projects else []
    matches = room_membership(chat_id, set(projects), refs, bindings)
    rows, coverage = Memory(root).read_chat_generations(predicate=lambda row: matches(_chat_id(row), row))
    if chat_id not in projects and chat_id not in {0, 1} and not rows:
        return None
    annotations = latest_chat_annotations(root)
    from ouroboros.chat_chain import iter_room_rows, source_row_id  # D06->D15 is lazy-only
    from ouroboros.chronicle_import import row_lineage

    # An activated chronicle records the lineage epoch: an outgoing row before it is signed by
    # its task result, never assumed mine. That rule needs each room row's stream position.
    lineage, positions = row_lineage(root), {}
    if lineage:
        for address, _row, pos in iter_room_rows(root, chat_id):
            positions.setdefault(address["row_sha256"], pos)
    source_rows = []
    for index, row in enumerate(rows, 1):
        pos = positions.get(source_row_id(row)) if positions else None
        annotation = annotations.get(str(row.get("client_message_id") or ""), {})
        if (row.get("direction") == "in" and not row.get("attachment_manifest")
                and not row.get("attachment_manifest_ref") and isinstance(annotation.get("attachment_manifest"), list)):
            row = {**row, "attachment_manifest": annotation["attachment_manifest"]}
        source_rows.append(_row_projection(row, "chat", index, root, pos=pos, lineage=lineage))
    # Pre-existing accepted blocks are another retained projection of the same
    # quiz producer. Recover them where still available; never call an old ask
    # currently open merely because its lifecycle projection was evicted.
    from ouroboros.owner_quiz import quiz_states
    answered = {(str(row.get("task_id") or ""), str((row.get("quiz") or {}).get("quiz_id") or ""))
                for row in rows if row.get("type") == "quiz_answer"}
    states = {}
    quiz_gaps = []
    for row in rows:
        if row.get("type") != "quiz" or not isinstance(row.get("quiz"), dict):
            continue
        tid, qid = str(row.get("task_id") or ""), str(row["quiz"].get("quiz_id") or "")
        if not tid or (tid, qid) in answered:
            continue
        if tid not in states:
            states[tid] = quiz_states(root, tid)
        block = states[tid].get(qid)
        if not block:
            quiz_gaps.append({"kind": "quiz_lifecycle_unavailable", "task_id": tid, "quiz_id": qid})
        elif block.get("state") == "answered":
            source_rows.append(_row_projection({"task_id": tid, "quiz": block, "type": "quiz_answer",
                "direction": "system", "source": "owner_quiz_answer", "ts": block.get("answered_at"),
                "client_message_id": f"quiz_answer:{tid}:{qid}"}, "retained_quiz_projection", len(source_rows) + 1, root))
            answered.add((tid, qid))
    coverage["gaps"].extend(quiz_gaps)
    for origin in project_origin_rows(root, chat_id):
        if not any(entry_matches_source_ref(row, [origin["ref"]]) for row in rows):
            source_rows.append(_row_projection({**origin["ref"], "direction": "in", "text": origin["text"],
                                               "project_origin_projection": True}, "retained_origin", len(source_rows) + 1))
    progress, progress_coverage = _progress_source(root, matches)
    source_rows.extend(_row_projection(row, "progress", i, root) for i, row in enumerate(progress, 1))
    mailbox_coverage = {"included": bool(task_id), "complete": True}
    # steering.py constructs this exact delivery id from the canonical client
    # id and target. Link delivery provenance, never deduplicate by meaning.
    owner_deliveries = {f"{row['client_message_id']}:{task_id}": row for row in source_rows
                        if row.get("direction") == "in" and row.get("client_message_id")}
    if task_id:
        from ouroboros.owner_mailbox import (
            KIND_OWNER_TEXT, KIND_QUIZ_ANSWER, KIND_TASK_MESSAGE,
            deliver_task_message, drain_owner_entries,
        )
        entries = drain_owner_entries(pathlib.Path(mailbox_root or root), task_id, include_acknowledged=True,
                                      kinds={KIND_OWNER_TEXT, KIND_QUIZ_ANSWER, KIND_TASK_MESSAGE}, _read_status=mailbox_coverage)
        for i, entry in enumerate(entries, 1):
            row = {**entry, "task_id": task_id}
            if entry.get("kind") == KIND_QUIZ_ANSWER:
                # New canonical facts and the old mailbox use the same typed
                # quiz id; a duplicated physical delivery is one source answer.
                source_id = f"quiz_answer:{task_id}:" + str(entry.get("msg_id") or "").removeprefix("quiz_answer:")
                if any(r.get("client_message_id") == source_id for r in source_rows):
                    continue
            if entry.get("kind") == KIND_TASK_MESSAGE:
                rendered = []
                deliver_task_message(entry, task_id, None, rendered.append)
                row["text"] = "\n".join(rendered)
            projected = _row_projection(row, "mailbox", i, mailbox_root or root)
            original = owner_deliveries.get(str(entry.get("msg_id") or "")) if entry.get("kind") == KIND_OWNER_TEXT else None
            if original is not None:
                if projected["text"] == original["text"]:
                    projected.pop("text")  # Same proven source id and exact bytes.
                original.setdefault("mailbox_deliveries", []).append(projected)
            else:
                source_rows.append(projected)
    seen, unique = set(), []
    for row in source_rows:
        identity = str(row.get("client_message_id") or row.get("msg_id") or "")
        key = (str(row.get("task_id") or ""), identity) if identity else None
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        unique.append(row)

    def order(row):
        try:
            return _history_timestamp(row.get("ts"))
        except ValueError:
            return datetime.min.replace(tzinfo=timezone.utc)

    unique.sort(key=order)  # Stable ties preserve source generation/row order.
    source_coverage = {"chat": coverage, "progress": progress_coverage, "mailbox": mailbox_coverage}
    if not coverage.get("generations"):
        source_coverage["room_gap"] = "no_chat_generations"
    elif not unique:
        source_coverage["room_gap"] = "no_retained_room_rows"
    if any(not row.get("ts") for row in unique):
        source_coverage["ordering_gap"] = "rows_without_timestamps_keep_source_order"
    label = str(projects.get(chat_id, {}).get("name") or ("Main" if chat_id == 1 else f"Chat {chat_id}"))
    # Physical log growth/rotation and capture retries do not change this room.
    # Keep their audit facts outside the source bytes used by paid identity;
    # semantic gaps and stability still travel with every immutable snapshot.
    source_header_coverage = {
        name: {key: value for key, value in section.items()
               if key not in {"generations", "capture_attempts", "matched_rows"}}
        if isinstance(section, dict) else section
        for name, section in source_coverage.items()
    }
    header = {"chat_id": chat_id, "label": label, "rows": len(unique), "coverage": source_header_coverage}
    text = "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in [header, *unique]) + "\n"
    from ouroboros.tools.review_helpers import redact_prompt_secrets

    text, redacted = redact_prompt_secrets(text)
    return {**header, "coverage": source_coverage, "captured_at": utc_now_iso(), "rows": [json.loads(line) for line in text.split("\n")[1:-1]], "text": text, "secrets_redacted": redacted,
            "sha256": sha256(text.encode()).hexdigest(), "bytes": len(text.encode())}


def chat_evidence_reader(drive_root: Any):
    """Sibling of the task-result resolver; never an arbitrary file reader."""
    def read(chat: str):
        try:
            return read_room_source(drive_root, int(chat))
        except (TypeError, ValueError):
            return None
    return read


def task_room_record(drive_root: Any, task_id: str) -> dict:
    """Read-only addressing projection; never quarantine/migrate a legacy result."""
    from ouroboros.task_results import task_result_path
    from ouroboros.utils import read_json_dict

    return read_json_dict(task_result_path(drive_root, task_id, create=False)) or {}


def own_room_chat(ctx: Any, drive_root: Any) -> int | None:
    """Use task/current-chat/lineage facts without guessing Main for children."""
    task_id = str(getattr(ctx, "task_id", "") or "")
    stored = task_room_record(drive_root, task_id)
    meta = getattr(ctx, "task_metadata", {}) or {}
    task = {**stored, **meta, "task_id": task_id}
    bound = bound_room_chat(all_task_bindings(drive_root), task)
    if bound:
        return bound
    for value in (getattr(ctx, "current_chat_id", None), task.get("chat_id")):
        if value is not None and value != "":
            return int(value)
    for field in ("parent_task_id", "root_task_id"):
        ancestor = str(task.get(field) or "")
        if ancestor:
            source = task_room_record(drive_root, ancestor)
            if source.get("chat_id") is not None:
                return int(source["chat_id"])
    return None
