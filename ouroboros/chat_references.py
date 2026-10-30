"""Host-owned message addresses and Reply facts, shared by every owner ingress.

References describe what the sender pointed at. They do not select work or grant
authority. Hash the persisted row before adding any history/view fields. A row the
dialogue store cannot answer yet (its archive still importing, or unreadable) leaves
the reference unresolved; it never blocks the message that carries it.
"""
from __future__ import annotations

from ouroboros.chat_chain import format_address, resolve_row, row_address
from ouroboros.store import StoreError


def projection_fields(row: dict) -> dict:
    """Additive fields for live frames and history, from the same stored row.

    An outgoing chat row saved under its own name (a relay's ``relay:<id>:<task>``) carries that name as
    ``row_client_message_id``: on a live assistant/system frame ``client_message_id`` names the owner's
    submission the row answers, so the row's own name travels under a field of its own. A progress
    record is not named: it merges its frame's meta, owner correlation included.
    """
    fields = {key: row[key] for key in ("reply_to", "relay", "room_revisions") if isinstance(row.get(key), dict)}
    if row.get("reply_to_relay"):
        fields["reply_to_relay"] = row["reply_to_relay"]
    # A host-fact ingress row (a call's opening) is the host's own row named by that id, as history keys it.
    if (row.get("direction") != "in" or isinstance(row.get("host_fact"), dict)) and not row.get("is_progress") \
            and row.get("client_message_id"):
        fields["row_client_message_id"] = str(row["client_message_id"])
    fields["row_address"] = format_address(row_address(row))
    return fields


def reference_from_row(row: dict) -> dict:
    kind = str(row.get("type") or "message")
    # A lifecycle row carrying an authored answer is that answer in Ouroboros's
    # voice wherever Main shows it (project_answer.js::appendProjectRow); the
    # address still hashes the raw System row.
    answer = str(row.get("completion_answer") or "") if kind in {
        "project_started", "project_handoff", "project_completion_summary"} else ""
    kind = ("question" if kind == "quiz" else "completion" if kind in
            {"project_completion_summary", "task_summary"} else
            "system" if row.get("direction") == "system" else "message")
    role = "assistant" if answer else {"in": "user", "out": "assistant", "system": "system"}.get(
        row.get("direction"), "system")
    from ouroboros.room_membership import reply_target_home
    return {"address": format_address(row_address(row)), "chat_id": int(row.get("chat_id", 1)),
            "role": role, "author_label": (str(row.get("sender_label") or "You") if role == "user"
                                             else {"assistant": "Ouroboros", "system": "System"}[role]),
            "kind": kind, "snippet": " ".join(str(answer or row.get("text") or row.get("caption") or "").split())[:160],
            "ts": str(row.get("ts") or ""), "task_id": str(row.get("task_id") or ""),
            "target_home": reply_target_home(row)}


def resolve_reply(root, address: str) -> dict:
    """Resolve a transport-neutral Reply address once, at acceptance."""
    if not isinstance(address, str):
        raise ValueError("reply_to must be an address string")
    if not address:
        return {}
    if address.startswith("task:"):
        from ouroboros.task_status import load_effective_task_result
        task_id = address[5:]
        task = load_effective_task_result(root, task_id, materialize_artifacts=False) if task_id else None
        if task:
            return {"address": address, "chat_id": int(task.get("chat_id") or 1), "role": "assistant",
                    "author_label": "Ouroboros", "kind": "work_card",
                    "snippet": " ".join(str(task.get("title") or task.get("text") or "").split())[:160],
                    "ts": str(task.get("updated_at") or task.get("started_at") or ""), "task_id": task_id,
                    "target_home": "work"}
    else:
        try:
            row, _resolution = resolve_row(root, address)
            if row is None:
                row = resolve_progress_row(root, address)
        except (ValueError, StoreError):
            row = None
        if row:
            return reference_from_row(row)
    return {"address": address, "unresolved": True}


def ingress_reply(root, metadata: dict) -> dict:
    """Shared web/HTTP/skill seam. Raw input is resolved, never trusted as a preview."""
    return resolve_reply(root, metadata.get("reply_to", ""))


def origin_reply(root, task_id: str, metadata=None) -> dict:
    """The answer's host reference to the owner message that began its turn.

    Best-effort enrichment: a busy or unreadable store omits the reference, never the message."""
    if not task_id or not root:
        return {}
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.project_dialogue import resolve_owner_message_source
    try:
        task = metadata or load_effective_task_result(root, task_id, materialize_artifacts=False) or {}
        ref = task.get("origin_message_ref") or (task.get("metadata") or {}).get("origin_message_ref")
        row = resolve_owner_message_source(root, ref) if ref else None
    except StoreError:
        return {}
    return reference_from_row(row) if row else {}


def resolve_progress_row(root, address):
    """Progress uses the same persisted-row hash, found by its indexed chat and time.

    A missing or ambiguous row answers None, as a chat row that does not resolve does."""
    return resolve_row(root, address, stream="progress")[0]


def resolve_history_row(root, address):
    """Exact navigation reads BOTH typed sources; ingress's unresolved preview is not a verdict.

    A readable match survives the other source's import/failure. Only two complete readable
    misses prove absence; conflicting matches and an ambiguous prefix remain ambiguous. A
    same-time different row beside unreadable lines proves nothing here, though generic
    ``resolve_row`` callers keep that ``row_mismatch`` and its candidates.
    """
    from ouroboros import dialogue_positions as dp
    from ouroboros.chat_chain import source_row_id
    from ouroboros.store import StoreMigrating

    matches, states, errors = {}, set(), []
    for source in ("chat", "progress"):
        try:
            row, resolution = resolve_row(root, address, stream=source)
            gap = resolution["status"] == "row_mismatch" and dp.skipped_between(root, source)
            states.add("row_unreadable" if gap else resolution["status"])
            if row is not None:
                matches.setdefault(source_row_id(row), (row, {**resolution, "source": source}))
        except (OSError, StoreError) as exc:
            errors.append(exc)
    if len(matches) > 1 or "row_ambiguous" in states:
        return None, {"state": "ambiguous"}
    if matches:
        return next(iter(matches.values()))
    if errors or "row_unreadable" in states:
        if errors and all(isinstance(exc, StoreMigrating) for exc in errors) and "row_unreadable" not in states:
            raise StoreMigrating("Dialogue history is still importing")
        raise StoreError("Dialogue history source unavailable")
    return None, {"state": "absent"}


def history_row_destination(root, row, thread_id, project_ids, source_refs, rooms):
    """Prefer a real occurrence in this room; otherwise the row's home, without source-ref copies.

    Only row-derived candidates enter the shared membership rule. Retained origins, the quoting
    task and the reference's old chat label cannot supply a home. Unreadable authority raises.
    """
    from ouroboros.chat_chain import _row_chat_id
    from ouroboros.project_admission import reserved_project_for_chat
    from ouroboros.project_dialogue import bound_room_chat, room_membership

    written = _row_chat_id(row)
    def belongs(chat, refs):
        return room_membership(chat, project_ids, refs, rooms)(written, row)

    if thread_id in {1, *project_ids} and belongs(thread_id, source_refs):
        home = thread_id
    else:
        work = bound_room_chat(rooms, row)
        candidates = {1, written, work} & {1, *project_ids}
        homes = [chat for chat in candidates if belongs(chat, [])]
        if len(homes) > 1 and work in homes:  # work speech shown where it was written too: its work's room is home
            homes = [work]
        if len(homes) != 1:
            return {"state": "ambiguous" if homes else "unavailable"}
        home = homes[0]
    # Every room's identity, then only the home's routing: a malformed unrelated room blocks nothing.
    if home != 1 and reserved_project_for_chat(root, home).get("lifecycle") != "active":
        return {"state": "unavailable"}
    return {"state": "found" if home == thread_id else "moved", "chat_id": home}


def origin_projection(root, ref):
    """A projected origin keeps its real source address, never hashes view fields."""
    from ouroboros.project_dialogue import resolve_owner_message_source
    try:
        row = resolve_owner_message_source(root, ref)
    except StoreError:
        return {}
    return projection_fields(row) if row else {}
