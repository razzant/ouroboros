"""A room's considered point and its running work, from captured rows and indexed task facts; no history scan,
no sealing authority. What considers a room is an act of mine there (``memory_view_conversations``): my own
words, a quiet finish, a finish that folded what it read (``fold_read``), a note; never a time and never a wake's
bare boundary."""
from ouroboros.dialogue_provenance import row_class


def considered_through(entries, lineage):
    """Last mind-authored quiet finish in stored order, also when its own row is already sealed."""
    positions = []
    for _address, row, pos in entries:
        own = row_class(row, pos=pos, **lineage)["author"]["kind"] == "ouroboros"
        canonical = (row.get("summary_kind") == "terminal_root_projection"
                     and row.get("outcome_authority") == "canonical_task_result_after_finalization"
                     and row.get("terminal_origin") == "model_final"
                     and not row.get("subagent_task_id") and row.get("delegation_role") != "subagent")
        if row.get("quiet_finish") is True and (own or canonical):
            positions.append(pos)
    return max(positions, default=-1)


def running_by_room(root, by_room, notes, bindings, projects):
    """Current work from typed task columns; registered projects need no explicit binding or chat row."""
    from ouroboros.task_store import task_facts

    running = task_facts(root, where="status=?", params=("running",))
    project_rooms = {project["id"]: str(project["chat_id"]) for project in projects}
    addressed = {}
    for room, entries in by_room.items():
        for _address, row, _pos in entries:
            if row.get("task_id") in running:
                addressed.setdefault(row["task_id"], set()).add(room)
    for room, entries in notes.items():
        for note in entries:
            if note.get("task_id") in running:
                addressed.setdefault(note["task_id"], set()).add(room)
    result = {}
    for task, facts in running.items():
        room = bindings.get(task) or project_rooms.get(facts["project_id"])
        rooms = {str(room)} if room else addressed.get(task, set())
        for room in rooms:
            result.setdefault(room, []).append(task)
    return {room: sorted(tasks) for room, tasks in result.items()}
