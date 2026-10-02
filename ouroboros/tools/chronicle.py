"""Thin, source-addressed memory tools; remembering choices remain with the mind."""
from __future__ import annotations

import json
from types import SimpleNamespace

from ouroboros.chronicle_store import ChronicleStore, source_row_id, source_time_span
from ouroboros.tools.registry import ToolEntry
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result


def _root(ctx):
    from ouroboros.tool_access import canonical_data_root
    return canonical_data_root(ctx)


def _room(ctx, room_id=""):
    metadata = getattr(ctx, "task_metadata", {}) or {}
    value = next((address for address in (room_id, metadata.get("chat_id"),
                  getattr(ctx, "current_chat_id", None), getattr(ctx, "chat_id", None))
                  if address is not None and address != ""), None)
    if value is None or value == "":
        raise ValueError("room_id is required when this task has no room address")
    return str(value)


def _author(ctx):
    from ouroboros.knowledge import observed_route_stamp
    return {"kind": "mind", "task_id": str(getattr(ctx, "task_id", "") or ""),
            "route": observed_route_stamp(getattr(ctx, "_accumulated_usage", {}) or {})}


def _result(ctx, payload, *, meta=None, error=False):
    return _publish_tool_result(ctx, ToolResult(status="error" if error else "ok",
        code="TOOL_ARG_ERROR" if error else "OK",
        text=json.dumps(payload, ensure_ascii=False, sort_keys=True), meta=meta or {}))


def _source(ctx, ref, *, start=None, end=None):
    from ouroboros.chronicle_sources import read_chronicle_source
    if not isinstance(ref, dict):
        raise ValueError("source_ref must be a retained source reference")
    return read_chronicle_source(_root(ctx), ref, start=start, end=end)


def _source_metadata(ctx, ref):
    raw = _source(ctx, ref)
    try:
        decoded = json.loads(raw)
    except ValueError:
        return {"coverage": "retained_text", "source_row_ids": []}
    rows = decoded.get("rows") if isinstance(decoded, dict) else decoded
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        return {"coverage": "retained_text", "source_row_ids": []}
    ids = [source_row_id(row) for row in rows]
    if isinstance(decoded, dict) and decoded.get("source_row_ids", ids) != ids:
        raise ValueError("retained source row identities do not match its exact rows")
    return {"coverage": "source_bound", "source_row_ids": ids,
            "task_ids": sorted({str(row[k]) for row in rows for k in ("task_id", "parent_task_id", "root_task_id") if row.get(k)}),
            "source_range": decoded.get("range") if isinstance(decoded, dict) else None,
            "source_span": source_time_span(row.get("ts") for row in rows),
            "source_coverage": decoded.get("coverage", {}) if isinstance(decoded, dict) else {}}


def _chronicle_write(ctx, text="", room_id="", node_id="", revision_id="", decision="", reason="", source_ref=None):
    try:
        store, author = ChronicleStore(_root(ctx)), _author(ctx)
        if revision_id:
            if decision not in {"accept", "reject"}:
                raise ValueError("revision_id requires decision accept or reject and a reason")
            record = store.decide_revision(revision_id, decision == "accept", author, reason)
        else:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must contain your episode or revision")
            metadata = (_source_metadata(ctx, source_ref) if source_ref else
                        {"coverage": "authored_without_source_range", "source_row_ids": []})
            metadata.setdefault("task_ids", [])
            if not source_ref:
                try:
                    current_room = _room(ctx)
                except ValueError:
                    current_room = None  # An explicit destination needs no current address.
                if author["task_id"] and (room_id in ("", None) or str(room_id) == current_room):
                    metadata["task_ids"] = [author["task_id"]]
            # Source task membership drives adoption; the writer stays in author.task_id.
            if source_ref:
                metadata["source_refs"] = [source_ref]
            if node_id:
                record = store.revise(node_id, text, author, metadata=metadata)
            else:
                record = store.append_episode(_room(ctx, room_id), text,
                    [source_ref] if source_ref else [], author, metadata=metadata)
        return _result(ctx, record, meta={"chronicle_record_id": record["id"], "memory_written": True})
    except (OSError, ValueError, TypeError) as exc:
        return _result(ctx, {"error": str(exc)}, error=True)


def _raw_room(ctx, room_id, start, end):
    from ouroboros.memory import Memory
    from ouroboros.projects_registry import all_task_bindings, reserved_project_chat_ids
    from ouroboros.project_dialogue import room_membership, source_refs_for_project, _row_chat_id
    from ouroboros.consolidator import retain_memory_source

    chat = int(room_id)
    root = _root(ctx)
    projects = reserved_project_chat_ids(root)
    matches = room_membership(chat, projects, source_refs_for_project(root, chat), all_task_bindings(root))
    rows, coverage = Memory(root).read_chat_generations(
        exclude_a2a=True, predicate=lambda row: matches(_row_chat_id(row), row))
    end = len(rows) if end is None else min(end, len(rows))
    if type(start) is not int or type(end) is not int or not 0 <= start <= end:
        raise ValueError("raw room range must satisfy 0 <= start <= end")
    page = rows[start:end]
    payload = {"room_id": room_id, "rows": page, "source_row_ids": [source_row_id(r) for r in page],
               "range": {"start": start, "end": end, "total": len(rows), "unit": "matching_rows"},
               "coverage": coverage, "page_complete": start == 0 and end == len(rows)}
    source = retain_memory_source(SimpleNamespace(drive_root=root, task_id=str(getattr(ctx, "task_id", "") or "memory-read")),
                                  "room_read", json.dumps(payload, ensure_ascii=False).encode("utf-8"), "json")
    payload["source_ref"] = source
    return payload


def _memory_read(ctx, node_id="", room_id="", after_seq=0, limit=None, raw_room=False, start=0, end=None, source_ref=None):
    try:
        store = ChronicleStore(_root(ctx))
        if type(after_seq) is not int or after_seq < 0 or (limit is not None and (type(limit) is not int or limit < 1)):
            raise ValueError("after_seq must be nonnegative and limit must be a positive integer")
        if source_ref:
            raw = _source(ctx, source_ref, start=start, end=end)
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = {"text": raw}
            if not isinstance(payload, dict):
                payload = {"source": payload}
            rows, span = payload.get("rows"), payload.get("range", {})
            if isinstance(rows, list) and ("row_locators" in payload or payload.get("missing")
                    or start or len(rows) != span.get("total", len(rows))):
                # Bind exactly this retrieved page, including gaps. The original
                # manifest does not credit unseen rows, nor need a whole-room copy.
                from ouroboros.consolidator import retain_memory_source
                payload.pop("row_locators", None)
                payload["parent_source_ref"] = source_ref
                source_ref = retain_memory_source(SimpleNamespace(drive_root=_root(ctx),
                    task_id=str(getattr(ctx, "task_id", "") or "memory-read")), "room_read",
                    json.dumps(payload, ensure_ascii=False).encode("utf-8"), "json")
            payload["source_ref"] = source_ref
        elif raw_room:
            if limit is not None:
                end = start + limit if end is None else min(end, start + limit)
            payload = _raw_room(ctx, _room(ctx, room_id), start, end)
        elif node_id:
            node = store.get(node_id)
            if node is None:
                raise ValueError("memory node not found")
            selected = next((row for row in store.room_records(node["room_id"], after_seq=node["sequence"] - 1, limit=1)
                             if row["id"] == node_id), None)
            payload = {"original": node, "current": selected or node}
        else:
            room = _room(ctx, room_id)
            rows = store.room_records(room, after_seq=after_seq, limit=None if limit is None else limit + 1)
            more = limit is not None and len(rows) > limit
            page = rows if limit is None else rows[:limit]
            payload = {"room_id": room, "records": page, "has_more": more,
                       "next_after_seq": page[-1]["sequence"] if page else after_seq,
                       "active_marks": store.active_marks(room), "range_complete": not more, "page_complete": after_seq == 0 and not more,
                       "range": {"after_seq": after_seq, "limit": limit, "unit": "chronicle_sequence"}}
        return _result(ctx, payload)
    except (OSError, ValueError, TypeError) as exc:
        return _result(ctx, {"error": str(exc)}, error=True)


def _memory_mark(ctx, text="", node_id="", source_ref=None, quote=None, room_id="", scope="room", release_id="", reason="", mark_id="", visibility=""):
    try:
        store, author = ChronicleStore(_root(ctx)), _author(ctx)
        if mark_id:
            record = store.set_mark_view(mark_id, visibility, author, reason)
        elif release_id:
            record = store.release_mark(release_id, author, reason)
        else:
            if scope not in {"room", "global"} or not isinstance(text, str) or not text.strip():
                raise ValueError("a mark needs a freeform text and scope room or global")
            if bool(node_id) == bool(source_ref):
                raise ValueError("choose exactly one node_id or source_ref")
            if node_id:
                node = store.get(node_id)
                if node is None:
                    raise ValueError("memory node not found")
                source_text = str(node.get("text", ""))
                target = {"kind": "chronicle", "id": node_id}
                room = str(room_id or node.get("room_id") or _room(ctx))
            else:
                source_text = _source(ctx, source_ref)
                target, room = source_ref, _room(ctx, room_id)
            quote_sources = [source_text]
            if source_ref:
                try:
                    parsed = json.loads(source_text)
                    rows = parsed.get("rows") if isinstance(parsed, dict) else parsed
                    if isinstance(rows, list):
                        quote_sources = [str(row.get("text", "")) for row in rows if isinstance(row, dict)]
                except ValueError:
                    pass
            if quote is not None and (not isinstance(quote, str) or not quote or not any(quote in value for value in quote_sources)):
                raise ValueError("quote must be an exact non-empty substring of the retained source")
            record = store.mark(target, text, author, room_id=room, scope=scope, quote=quote)
        # Compaction can retain the address and reread the active mark instead
        # of turning its freeform meaning into host policy.
        return _result(ctx, record, meta={"memory_mark_id": record.get("target_id", record["id"]),
                                        "memory_mark_operation": record["kind"]})
    except (OSError, ValueError, TypeError) as exc:
        return _result(ctx, {"error": str(exc)}, error=True)


def chronicle_tools():
    string = {"type": "string"}
    source = {"type": "object", "description": "Exact retained source_ref from memory_read or another source reader; keep task_id and hash."}
    definitions = [
        ("chronicle_write", _chronicle_write,
         "Record your own episode immediately, or revise an existing node. Helper corrections remain attributed and the original stays readable. Accept/reject a helper revision with revision_id, decision and reason. Sources bind exact rows; without a source the note claims no raw coverage. Use memory_read to inspect before revising.",
         {"text": string, "room_id": string, "node_id": string, "source_ref": source,
          "revision_id": string, "decision": {"type": "string", "enum": ["accept", "reject"]}, "reason": string}),
        ("memory_read", _memory_read,
         "Read originals, revisions and their sources, or page through a room. source_ref reads a retained snapshot, including its exact source chunks; start/end page its rows. room_id is the original numeric chat ID as text (for example 1), with the current addressed room as default. raw_room=true captures exact current room messages and returns a retained source_ref with row identities and explicit range/coverage. limit caps room records; with raw_room it caps rows from start without extending end. Omitting limit keeps the requested range unrestricted. Use it to investigate lost detail, verify a correction or recall an old decision; choose the depth yourself.",
         {"node_id": string, "room_id": string, "source_ref": source, "after_seq": {"type": "integer", "minimum": 0},
          "limit": {"type": "integer", "minimum": 1}, "raw_room": {"type": "boolean"},
          "start": {"type": "integer", "minimum": 0}, "end": {"type": "integer", "minimum": 0}}),
        ("memory_mark", _memory_mark,
         "Mark what matters in your own words: a source, decision, unresolved concern or memory correction. Choose room/global visibility. Optional quote is verified exactly against the retained target. Change only its visible detail with mark_id, visibility full/meaning and reason; the exact quote stays stored. Release explicitly with release_id and reason when you judge it superseded; history remains. No mark imposes a prescribed workflow. Active marks remain resident in shared memory views; declared-only inputs are not automatically augmented.",
         {"text": string, "node_id": string, "source_ref": source, "quote": string, "room_id": string,
          "scope": {"type": "string", "enum": ["room", "global"]}, "release_id": string, "reason": string,
          "mark_id": string, "visibility": {"type": "string", "enum": ["full", "meaning"]}}),
    ]
    schemas = [{"name": name, "description": description,
                "parameters": {"type": "object", "properties": properties, "additionalProperties": False}}
               for name, _handler, description, properties in definitions]
    return [ToolEntry("chronicle_write", schemas[0], _chronicle_write),
            ToolEntry("memory_read", schemas[1], _memory_read),
            ToolEntry("memory_mark", schemas[2], _memory_mark)]
