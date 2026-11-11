"""Memory forms and their distance order, independent of a request's budget and fit.

Each element keeps an existing rung and source identity. Stages order those
elements; the clean-start guard walks only ordinary forms. Event counts reserve
their place without changing a silent room into an event.
"""
from ouroboros import memory_view as mv, memory_view_conversations as conversations, memory_view_legacy as legacy
from ouroboros.memory_view_work import works, work_line
from ouroboros.memory_form_facts import form_counts as form_counts, holding_room as holding_room

LADDER = ("F1", "F6w", "F6g", "F3", "F2", "F4", "F7p", "F5l", "F5", "F5a", "F7", "F7a", "F8", "F8a", "F6a")
# The owner's pending alternative swaps work_final and remote_last here.
STAGES = ("remote_events", "remote_told", "remote_unread", "remote_gists", "task_facts", "work_progress",
          "remote_mine", "remote_people", "remote_all", "work_final", "remote_last", "story_ordinary",
          "story_openings", "room_emergency", "task_only", "protected_replies", "remote_count", "story_count",
          "unread_count", "physical_rows", "owner")


def _catalogue(snapshot):
    """One capture's forms, source facts and stages; not a render or a persistent snapshot field."""
    room = snapshot.room or {}
    words = {w["id"]: w for g in snapshot.conversations for r in g["rooms"] for w in r["words"]}
    own = {w["address"]: w for w in room.get("lane1", ())}
    tasks = {t["first"]: t for t in room.get("lane2", ())}
    records = {r["id"]: r for r in [*snapshot.story, *room.get("notes", ())]}
    progress, finals = {}, {}
    for key, rows in works(room):
        progress["work:" + key + ":progress"] = rows
        finals["work:" + key + ":final"] = rows
    parents = {w["id"]: g["key"] for g in snapshot.conversations for r in g["rooms"] for w in r["words"]}
    return room, words, own, tasks, records, progress, finals, parents


def stage_of(snapshot, step, ident):
    cached = snapshot.__dict__.get("_form_catalogue")
    if cached is None:
        cached = _catalogue(snapshot)
        object.__setattr__(snapshot, "_form_catalogue", cached)
    room, words, own, tasks, records, progress, finals, _parents = cached
    if ident.startswith("bare:") and step in ("F2", "F6w", "F7p"):
        return "physical_rows"
    if step == "F1":
        task = tasks[ident]
        return "task_facts" if not task.get("gist") or any(
            w.get("task_id") == task.get("task") for w in own.values()) else "task_only"
    if step == "F2":
        if ident in progress:
            return "work_progress"
        if ident in finals:
            return "protected_replies" if any(w.get("protected_reply") for w in finals[ident]) else "work_final"
        return "protected_replies" if own[ident].get("protected_reply") else "room_emergency"
    if step == "F6w":
        word = words[ident]
        return ("remote_told" if conversations._told(word) else "remote_unread" if conversations._unread(word)
                else "remote_mine" if word["kind"] == "ouroboros" else "remote_people")
    if step == "F6g":
        return "remote_all" if ident.startswith("all:") else "remote_last" if ident.startswith("last:") else "remote_gists"
    if step == "F6a":
        return "unread_count" if ident == "unread" else "remote_count"
    if step in ("F5l", "F8"):
        return "story_ordinary" if "lead_rule" in records[ident] and records[ident].get("text", "").strip() else "story_openings"
    if step == "F4" and any(r["id"] == ident and r.get("standing") for r in room.get("under_parts", ())):
        return "story_ordinary"
    if step in ("F5a", "F8a"):
        return "story_count"
    if step == "F5":
        return "story_openings"
    return "owner" if step == "F7" else "room_emergency"


def ordinary_pairs(snapshot, elements=None):
    cached = snapshot.__dict__.get("_ordinary_forms")
    if elements is not None and cached and cached[0] is elements:
        return cached[1]
    ordinary = {"remote_events", "remote_told", "remote_unread", "remote_gists", "task_facts", "work_progress",
                "work_final", "story_ordinary"}
    pairs = {(step, ident) for step, ident, *_ in (list_elements(snapshot) if elements is None else elements) if stage_of(snapshot, step, ident) in ordinary}
    if elements is not None:
        object.__setattr__(snapshot, "_ordinary_forms", (elements, pairs))
    return pairs


def normalized(selected):
    """Superseded forms do not acquire a second receipt or a different price key."""
    chosen = set(selected)
    for step, ident in selected:
        if step == "F6w" and not ident.startswith("bare:"):
            continue
        if ident.startswith("bare:") and step in ("F2", "F6w", "F7p"):
            chosen.discard((step, ident[5:]))
        if step == "F5":
            chosen.discard(("F5l", ident))
        if step == "F2" and ident.startswith("work:") and ident.endswith(":final"):
            chosen.discard((step, ident.removesuffix(":final") + ":progress"))
        if step == "F6g" and ident.startswith(("all:", "last:")):
            key = ident.split(":", 1)[1]
            chosen.discard((step, key))
            if ident.startswith("last:"):
                chosen.discard((step, "all:" + key))
    return chosen


def swallowed(snapshot, step, ident, selected):
    """A grouped conversation owns its inner forms until it is returned to full text."""
    if step not in ("F6w", "F6g"):
        return False
    key = (snapshot.__dict__["_form_catalogue"][-1].get(ident.removeprefix("bare:")) if step == "F6w" else
           ident.split(":", 1)[1] if ident.startswith(("all:", "last:")) else ident)
    return (("F6a", key) in selected or step == "F6w" and any(
        ("F6g", name) in selected for name in (key, "all:" + key, "last:" + key))
        or step == "F6g" and ident == key and ("F6g", "all:" + key) in selected
        or step == "F6g" and ident != "last:" + key and ("F6g", "last:" + key) in selected)


def list_elements(snapshot):
    if not snapshot.active:
        return []
    room = snapshot.room or {}
    lane1, lane2 = room.get("lane1") or [], room.get("lane2") or []
    mine = sorted((r for r in lane1 if r["kind"] == "ouroboros" and not r.get("gist")), key=lambda r: r["pos"])
    block = conversations.floor_steps(snapshot.conversations, index=conversations.source_index(snapshot))
    work = []
    for key, rows in works(room):
        partial = [r for r in rows[:-1] if not r.get("protected_reply")]
        if partial:
            work.append(("work:" + key + ":progress", "\n".join(r["line"] for r in partial), work_line(room, partial)))
        work.append(("work:" + key + ":final", "\n".join(r["line"] for r in rows), work_line(room, rows)))
    story = sorted(snapshot.story, key=lambda r: tuple(r.get("floor_order", (0, 0))))
    steps = {
        "F1": [(r["first"], r["line"], "" if n else mv._facts_line(room, [r])) for n, r in enumerate(lane2)],
        **{step: block[step] for step in ("F6w", "F6g", "F6a")},
        "F2": work + [(r["address"], r["line"], mv._row_opening(r, "my reply")) for r in mine]
              + [("bare:" + r["address"], mv._row_opening(r, "my reply"), mv._row_pointer(r, "my reply")) for r in mine],
        "F3": [(rid, "\n".join(map(legacy.retold_record, group)), legacy.room_pointer(group))
               for rid, group in legacy.pointer_rooms(snapshot.story).items()],
        "F4": [(r["id"], mv._retold(r, r.get("standing", False)), mv._retold(r, True))
               for r in [*room.get("legacy", ()), *room.get("under_parts", ())]],
        "F7p": [(prefix + r["address"], mv._row_opening(r, "words") if prefix else r["line"], render(r, "words"))
                 for prefix, render in (("", mv._row_opening), ("bare:", mv._row_pointer)) for r in lane1
                 if r["kind"] != "ouroboros" and not r.get("owner_side", r["kind"] == "human")],
        "F7": [(r["address"], r["line"], r["short"]) for r in room.get("project_words") or () if r.get("address")]
              + block["F7"] + [(r["address"], r["line"], mv._row_pointer(r, "words")) for r in lane1
                              if r.get("owner_side", r["kind"] == "human") and not r.get("immune")],
        "F5l": [(r["id"], "\n".join(mv._page_lines(r)), mv._page_label(r)) for r in story
                 if r.get("kind") in ("page", "part") and r.get("text", "").strip() and not r.get("told_by")],
        "F5": [(r["id"], "\n".join(mv._page_lines(r)), mv._page_pointer(r)) for r in story
                if r.get("kind") not in ("legacy", "account") and not r.get("told_by")],
        "F7a": [(str(room['room_id']), '', '')] if lane1 else [],
        "F5a": [('story', '', '')] if snapshot.story else [],
        "F8": [(n['id'], '\n'.join(mv._note_lines([n])), mv._note_pointer(n)) for n in room.get('notes', ())],
        "F8a": [(str(room['room_id']), '', '')] if room.get('notes') else [],
    }
    listed = [(step, ident, whole, short) for step in LADDER for ident, whole, short in steps[step]
              if step in ("F1", "F5a", "F7a", "F8", "F8a", "F6a") or len(short) < len(whole)]
    return sorted(listed, key=lambda e: STAGES.index(stage_of(snapshot, e[0], e[1])))
