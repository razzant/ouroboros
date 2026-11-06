"""Terminal links and covered rooms of authored pages; they never invent page coverage."""
from __future__ import annotations

from ouroboros import chat_chain, memory_inventory
from ouroboros.chronicle_store import DELEGATED_FOCUS_ROLES
from ouroboros.dialogue_provenance import recorded_finality


def _terminal_writer(record):
    author = record.get("author", {})
    task_id = str(author.get("task_id") or "")
    own_writer = author.get("kind") == "mind" or (author.get("kind") == "helper" and
        author.get("focus", {}).get("role") in DELEGATED_FOCUS_ROLES)
    return task_id if record.get("kind") == "page" and own_writer else ""


def _terminal_after(record):
    return int(record.get("metadata", {}).get("terminal_after_pos",
        (record.get("covers", {}).get("stream_span") or [0, 0])[-1]))


# The stored facts a linked row's finality is read from (``dialogue_provenance.recorded_finality``).
_FINALITY_FACTS = ("summary_kind", "outcome_final", "outcome_authority")


def capture_terminal_rows(root, records):
    """Group the writers' later terminal rows once, from the stored facts of the rows after the legacy frontier
    (``memory_inventory.terminal_rows``): a page's writer ran after the chronicle's activation, so its later
    terminal rows lie after the frontier. Each is ``(address, {"type", "status", <the finality facts it
    stored>}, pos)``."""
    return writer_terminal_rows(root, terminal_starts(records))


def terminal_starts(records):
    """``{(room, writer task): the stream position after which its terminal rows are later}`` of the pages."""
    starts = {}
    for record in records:
        task_id = _terminal_writer(record)
        if task_id:
            key = (str(record["room_id"]), task_id)
            starts[key] = min(starts.get(key, _terminal_after(record)), _terminal_after(record))
    return starts


def writer_terminal_rows(root, starts):
    """``capture_terminal_rows`` for the writers' ``terminal_starts``: all it reads besides the pages."""
    if not starts:
        return {}
    tasks, first = {task_id for _room, task_id in starts}, min(starts.values())
    chosen = [(address, meta, pos, status) for address, meta, pos, status in memory_inventory.terminal_rows(root)
              if pos > first and str(meta.get("task_id") or "") in tasks]
    facts, grouped = memory_inventory.membership_facts(root), {}
    # The terminal rows' task lineages in one read a generation, then their rooms.
    facts.rooms.warm({str(meta.get(key) or "") for _address, meta, _pos, _status in chosen
                      for key in ("task_id", "parent_task_id", "root_task_id")} - {""})
    for address, meta, pos, status in chosen:
        task_id = str(meta["task_id"])
        for room in memory_inventory.rooms_of_row(meta, facts):
            if (room, task_id) in starts and pos > starts[(room, task_id)]:
                finality = {key: meta[key] for key in _FINALITY_FACTS if key in meta}
                grouped.setdefault((room, task_id), []).append(
                    (address, {"type": meta.get("type"), "status": status, **finality}, pos))
    return grouped


def linked_rows(root, store, by_room, projects=()):
    """The ``row_sha256`` of every writer's later terminal row its page links below it (``terminal_links``), and of
    each closure's own notice among the open rows ``by_room`` (``closure_notices``, over the registry's ``projects``
    the capture holds): that page's or closed record's, never read as an open row of my other conversations
    (``memory_view_conversations``) or a debt."""
    pages = store.room_records(None, kinds=("page",))
    grouped = capture_terminal_rows(root, pages)
    return {address["row_sha256"] for rows in grouped.values() for address, _facts, _pos in rows} | closure_notices(
        pages, by_room, projects)


def _closures(pages, projects=()):
    """``{room: the closure deliveries its closing pages and its closed project name}`` (``metadata.closure_delivery``,
    ``closed.delivery``)."""
    named = {}
    for page in pages:
        if (page.get("metadata") or {}).get("closure_delivery"):
            named.setdefault(str(page["room_id"]), set()).add(page["metadata"]["closure_delivery"])
    for project in projects:
        if (project.get("closed") or {}).get("delivery"):
            named.setdefault(str(project["chat_id"]), set()).add(project["closed"]["delivery"])
    return named


def closure_notices(pages, by_room, projects=()):
    """``close_project`` names the delivery of the System row it posts in the room (``project_closure``) on the
    project's closed record and on the page it writes there from its closing note first, when anything is open: that
    one row, wherever it lands, is the closure's, linked below the page or held by the record, so a project closed with
    nothing open leaves nothing open. A reopening removes the record and posts a notice of its own delivery: the room
    is open again. No position or text decides it."""
    return {address["row_sha256"] for room, ids in _closures(pages, projects).items()
            for address, meta, _pos in by_room.get(room, ()) if meta.get("delivery_id") in ids}


def closure_link(root, record):
    """A closing page's own closure notice (``closure_notices``) as one line with its exact address and reader, or
    that it is not in the room yet; nothing for any other record."""
    delivery, project = ((record.get("metadata") or {}).get(key) for key in ("closure_delivery", "closes_project"))
    if not delivery or record.get("kind") != "page":
        return []
    try:  # read from the page's first row on; a first address that no longer resolves reads the room whole
        rows = chat_chain.iter_room_rows(root, record["room_id"], from_addr=(record.get("covers") or {}).get("first"))
    except (chat_chain.RowAddressError, ValueError):
        rows = chat_chain.iter_room_rows(root, record["room_id"])
    found = next((address for address, row, _pos in rows if row.get("delivery_id") == delivery), None)
    if found is None:
        return [f"- The closure notice of project {project} (delivery {delivery}) is not in the room yet."]
    at = chat_chain.format_address(found)
    return [f"- The closure notice of project {project}: memory_read(room_id='{record['room_id']}', rows=true, "
            f"from='{at}', to='{at}')"]


def covered_rooms(root, record):
    """A conversation page's rooms (``covers.rooms``) as ``(room, label, first, last)``, oldest first: the first and
    last address of the rows it sealed there, read from that room's rows inside the page's span; none for a page over
    one room."""
    from ouroboros.memory_view import room_labeler

    covers = record.get("covers") if isinstance(record.get("covers"), dict) else {}
    rooms = covers.get("rooms") if record.get("kind") == "page" and isinstance(covers.get("rooms"), dict) else {}
    if not rooms:  # a note, an account, a page over one room: no labeler to build (it reads two registries)
        return []
    label, found = room_labeler(root), []
    label.warm(list(rooms))
    for room, refs in rooms.items():
        try:
            rows = chat_chain.iter_room_rows(root, room, from_addr=covers.get("first"), to_addr=covers.get("last"))
        except (chat_chain.RowAddressError, ValueError):  # a span bound that no longer resolves reads the room whole
            rows = chat_chain.iter_room_rows(root, room)
        wanted = set(refs)
        sealed = [(pos, chat_chain.format_address(address)) for address, _row, pos in rows
                  if address["row_sha256"] in wanted]
        found += [(sealed[0][0], room, label(room), sealed[0][1], sealed[-1][1])] if sealed else []
    return [entry[1:] for entry in sorted(found)]


def terminal_links(root, record, cache=None):
    """This page writer's later typed terminal rows as one separately attributed line: each row's kind,
    status and recorded finality (its authority stays in the row), and one reader over their span."""
    task_id = _terminal_writer(record)
    if not task_id:
        return []
    end = _terminal_after(record)
    cache = capture_terminal_rows(root, (record,)) if cache is None else cache
    rows = [(address, row) for address, row, pos in cache.get((str(record["room_id"]), task_id), ()) if pos > end]
    if not rows:
        return []
    items = []
    for _address, row in rows:
        speaker = "host summary" if row.get("type") in ("task_summary", "project_completion_summary") else "task final report"
        finality = recorded_finality(row.get("outcome_final"), row.get("outcome_authority"),
                                     host_facts=row.get("summary_kind") == "host_task_facts").partition(";")[0]
        status = row.get("status") or row.get("task_terminal_status") or "status not recorded"
        items.append(f"{speaker} ({status}{', ' + finality if finality else ''})")
    first, last = (chat_chain.format_address(rows[index][0]) for index in (0, -1))
    return [f"- Later terminal rows of task {task_id} (host-linked, not page coverage): {'; '.join(items)}; "
            f"memory_read(room_id='{record['room_id']}', rows=true, from='{first}', to='{last}')"]
