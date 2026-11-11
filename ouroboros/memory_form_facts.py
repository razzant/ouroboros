"""Receipts for the forms actually visible, and numeric prices of what holds the room.

These diagnostics share the renderers' source identities; cached values are only
counts and sizes, outside the persisted snapshot. No rejected full view is drawn.
"""
from ouroboros import memory_view as mv, memory_view_conversations as conversations, memory_view_legacy as legacy
from ouroboros.memory_view_work import works, folded_work_rows, task_line, task_gist_line, task_gists_before_work_lines


def form_counts(snapshot, level):
    cache = snapshot.__dict__.setdefault("_form_counts", {})
    if level.addressed not in cache:
        cache[level.addressed] = _count_forms(snapshot, level)
    return cache[level.addressed]


def _conversation_sources(snapshot):
    sources = snapshot.__dict__.get("_conversation_sources")
    if sources is None:
        sources = {}
        for data in conversations.source_index(snapshot).values():
            for rid, kinds in data["rooms"].items():
                sources[rid] = kinds["unread"], kinds["owner"], kinds["told"] + kinds["untold"] + kinds["fixed"]
        object.__setattr__(snapshot, "_conversation_sources", sources)
    return sources


def _count_forms(snapshot, level):
    """Visible replacement lines and their sources, after grouping; never sum swallowed rungs."""
    from ouroboros.memory_forms import stage_of
    gone = {step: set(ids) for step, ids in level.addressed}
    counts = {kind: {} for kind in ("retold", "host", "emergency", "window")}
    def add(kind, form, lines, sources):
        if sources:
            entry = counts[kind].setdefault(form, {"lines": 0, "sources": 0})
            entry["lines"] += lines
            entry["sources"] += sources
    room = snapshot.room or {}
    hidden, work_lines = folded_work_rows(room, gone)
    addressed = hidden | gone.get("F2", set()) | gone.get("F7", set()) | gone.get("F7p", set())
    addressed |= {i[5:] for i in addressed if i.startswith("bare:")}
    aggregate = [r for r in room.get("lane1", ()) if r["address"] in addressed and not r.get("immune")] if gone.get("F7a") else []
    grouped = {r["address"] for r in aggregate}
    folded_rows, separate = hidden - grouped, addressed - hidden - grouped
    bare = gone.get("F2", set()) | gone.get("F7p", set())
    add("emergency", "room ranges", bool(aggregate), len(aggregate))
    for key, rows in works(room):
        folded = [r for r in rows if r["address"] in folded_rows]
        add("retold", "works", int(bool(folded)), len(folded))
    for row in room.get("lane1", ()):
        if row["address"] in separate:
            add("emergency", "room addresses" if ("bare:" + row["address"] in bare or row["address"] in gone.get("F7", ())) else "room openings", 1, 1)
    tasks = [t for t in room.get("lane2", ()) if t["first"] in gone.get("F1", ())]
    sole = sum(stage_of(snapshot, "F1", t["first"]) == "task_only" for t in tasks)
    add("emergency", "sole task gists", int(bool(sole)), sole)
    add("host", "task facts", int(bool(tasks) and not sole), len(tasks) - sole)
    task_gists = ([t for t in room.get("lane2", ()) if t.get("gist") and t["first"] not in gone.get("F1", ())] if aggregate
                  else task_gists_before_work_lines(room, gone))
    add("retold", "task gists", len(task_gists), len(task_gists))
    for row in room.get("project_words") or ():
        if row.get("address") in gone.get("F7", ()):
            add("emergency", "sealed owner addresses", 1, 1)
    for entry in [*room.get("legacy", ()), *room.get("under_parts", ())]:
        if entry["id"] in gone.get("F4", ()):
            add("retold" if entry.get("standing") else "emergency", "room records", 1, 1)
    for step, entries, aggregate_step, label in (
            ("F5", [e for e in snapshot.story if not e.get("told_by")], "F5a", "story"),
            ("F8", room.get("notes", ()), "F8a", "notes")):
        aggregated = [e for e in entries if e["id"] in gone.get(step, ())] if gone.get(aggregate_step) else []
        grouped_ids = {e["id"] for e in aggregated}
        add("emergency", label + " lists", int(bool(aggregated)), len(aggregated))
        for entry in entries:
            if entry["id"] in grouped_ids:
                continue
            if step == "F5" and entry["id"] in gone.get(step, ()):
                add("emergency", label + " pointers", 1, 1)
            elif entry["id"] in gone.get("F5l" if step == "F5" else step, ()):
                lead = "lead_rule" in entry and bool(entry.get("text", "").strip())
                lines = mv._page_label(entry) if step == "F5" else mv._note_pointer(entry)
                add("retold" if lead else "emergency", label + (" leads" if lead else " openings"), len(lines.splitlines()), 1)
    for key, records in legacy.pointer_rooms(snapshot.story).items():
        if key in gone.get("F3", ()):
            add("emergency", "old story ranges", 1, len(records))
    _remote_counts(snapshot, gone, add)
    return counts


def _remote_counts(snapshot, gone, add):
    from ouroboros.memory_view_conversation_forms import local_key
    from ouroboros.memory_view_events import totals
    sources, index = _conversation_sources(snapshot), conversations.source_index(snapshot)
    cache = snapshot.__dict__.setdefault("_conversation_form_counts", {})
    lost = groups_counted = global_unread = event_arrivals = 0
    for group in snapshot.conversations:
        data = index[group["key"]]
        state = local_key(group, gone, data)
        count = state[3] == 4
        aggregate_unread = count and "unread" in gone.get("F6a", ())
        key = (*state, aggregate_unread)
        if key not in cache:
            cache[key] = _group_counts(group, data, sources, state, aggregate_unread)
        facts, missing, unread = cache[key]
        for kind, name, lines, n in facts:
            add(kind, name, lines, n)
        lost += missing
        if count:
            event_arrivals += totals(group["rooms"])[1]
        groups_counted += count
        global_unread += unread
    add("emergency", "conversation count", int(bool(groups_counted)), lost or groups_counted)
    add("host", "event counts", int(bool(event_arrivals) and not groups_counted), event_arrivals)
    add("window", "unread count", int(bool(global_unread)), global_unread)


def _group_counts(group, data, sources, state, aggregate_unread):
    """Only numbers for one local form; grouping another conversation cannot change them."""
    from ouroboros.memory_view_conversation_forms import _visible_gists
    from ouroboros.memory_view_events import totals
    form, words, owners, event_count = state[3:]
    count = form == 4
    facts, lost, global_unread = [], 0, 0
    def add(kind, name, lines, n):
        if n:
            facts.append((kind, name, lines, n))
    event_rooms, arrivals = totals(group["rooms"])
    if not count:
        add("host", "event counts" if event_count else "events", int(bool(event_rooms)) if event_count else event_rooms, arrivals)
    if form and not count:
        add("host", "conversation facts", 1, 1)
        gists = data["gists"]
        shown = gists[-1:] if form == 3 else gists
        add("retold", "work gists", len(shown), len(shown))
        if form >= 2:
            missing = data["untold"] + len(gists) - len(shown) + sum(len(r["notes"]) for r in group["rooms"])
            add("emergency", "all gists" if form == 2 else "last gist", 1, missing)
        no_gist = sum(not t.get("gist") for r in group["rooms"] for t in r["tasks"])
        add("host", "tasks without gist", int(bool(no_gist)), no_gist)
    elif not count and data["gists"]:
        n = len(_visible_gists(group, data, words))
        add("retold", "work gists", n, n)
    for remote in group["rooms"]:
        unread, owner, rest = sources[remote["room_id"]]
        if count:
            lost += len(rest) + len(remote["notes"]) + len(remote["tasks"]) + int((remote.get("folded") or {}).get("count", 0))
            lost += int(not remote["engaged"] and not remote.get("events"))
        elif form:
            if form == 1:
                add("retold", "note leads", len(remote["notes"]), len(remote["notes"]))
        else:
            unread = [w for w in unread if w["id"] in words]
        if aggregate_unread:
            global_unread += len(unread)
        elif not (form and remote.get("overheard")):
            add("host", "unread", int(bool(unread)), len(unread))
        if not count and (fold := remote.get("folded")):
            add("retold", "read and folded", 1, fold["count"])
        for word in owner:
            if word["id"] in owners:
                add("emergency", "other owner addresses", 1, 1)
        if not form:
            for word in rest:
                raw, bare = word["id"] in words, "bare:" + word["id"] in words
                if (raw or bare) and data["kinds"][word["id"]] == "untold":
                    add("emergency", "other addresses" if bare else "other openings", 1, 1)
    if form and not aggregate_unread:
        add("host", "overheard unread", int(bool(data["overheard"][1])), len(data["overheard"][1]))
    return tuple(facts), lost, global_unread


def holding_room(snapshot, level):
    cache = snapshot.__dict__.setdefault("_holding_room", {})
    if level.addressed not in cache:
        cache[level.addressed] = _holding_room(snapshot, level)
    return cache[level.addressed]


def _holding_room(snapshot, level):
    """Prices of the final sources, not of their old full forms; diagnoses may overlap."""
    gone = {step: set(ids) for step, ids in level.addressed}
    totals, ready = {}, []
    def hold(kind, text, reader="", unmarked=False):
        tokens = (mv.json_chars(text) + 5) // 4 if text else 0
        if tokens:
            totals[kind] = totals.get(kind, 0) + tokens
            if unmarked:
                totals["records without lead"] = totals.get("records without lead", 0) + tokens
            if reader:
                ready.append([tokens, reader])
    pages = [e for e in snapshot.story if not e.get("told_by") and e.get("kind") != "legacy"]
    aggregated = [e for e in pages if e["id"] in gone.get("F5", ())] if gone.get("F5a") else []
    grouped_ids = {e["id"] for e in aggregated}
    if aggregated:
        hold("story lists", mv._story_line(aggregated), unmarked=any("lead_rule" not in e for e in aggregated))
    for entry in pages:
        ident, kind = entry["id"], entry.get("kind")
        if entry["id"] in grouped_ids:
            continue
        text = (mv._page_pointer(entry) if ident in gone.get("F5", ()) else mv._page_label(entry)
                if ident in gone.get("F5l", ()) else "\n".join(mv._page_lines(entry)))
        hold({"account": "accounts", "part": "parts"}.get(kind, "pages without part"), text,
             f"memory_read(node_id='{ident}')", unmarked=kind != "account" and "lead_rule" not in entry)
    for key, records in legacy.pointer_rooms(snapshot.story).items():
        hold("old story", legacy.room_pointer(records) if key in gone.get("F3", ()) else
             "\n".join(legacy.retold_record(e) for e in records), f"memory_read(room_id='{key}')", unmarked=True)
    room = snapshot.room or {}
    for entry in [*room.get("legacy", ()), *room.get("under_parts", ())]:
        hold("room records", mv._retold(entry, entry["id"] in gone.get("F4", ()) or entry.get("standing", False)),
             f"memory_read(node_id='{entry['id']}')", unmarked=not entry.get("standing") and "lead_rule" not in entry)
    notes = room.get("notes", ())
    aggregate_notes = [e for e in notes if e["id"] in gone.get("F8", ())] if gone.get("F8a") else []
    grouped_notes = {e["id"] for e in aggregate_notes}
    if aggregate_notes:
        hold("notes", mv._notes_line(aggregate_notes), unmarked=any("lead_rule" not in e for e in aggregate_notes))
    for note in notes:
        if note["id"] not in grouped_notes:
            text = mv._note_pointer(note) if note["id"] in gone.get("F8", ()) else "\n".join(mv._note_lines([note]))
            hold("notes", text, f"memory_read(node_id='{note['id']}')", unmarked="lead_rule" not in note)
    hidden, work_lines = folded_work_rows(room, gone)
    addressed = hidden | {i.removeprefix("bare:") for step in ("F2", "F7p", "F7") for i in gone.get(step, ())}
    aggregate = [r for r in room.get("lane1", ()) if r["address"] in addressed and not r.get("immune")] if gone.get("F7a") else []
    if aggregate:
        hold("untold in this room", mv._rows_line(room, aggregate))
    else:
        for line in work_lines.values():
            hold("work gists", line)
    folded = {key for key, _rows in works(room) if any("work:" + key + suffix in gone.get("F2", ())
                                                     for suffix in (":progress", ":final"))} if not aggregate else set()
    only_gists = {t["first"] for t in task_gists_before_work_lines(room, gone)}
    for task in room.get("lane2", ()):
        if task.get("gist") and (task["first"] not in gone.get("F1", ()) or task["first"] in only_gists) and task.get("task") not in folded:
            gist = task["gist"]
            hold("task gists", task_gist_line(task) if task["first"] in only_gists else task_line(task), f"memory_read(room_id='{gist['room_id']}', rows=true, "
                 f"from='{gist['address']}', to='{gist['address']}')")
    grouped_rows = {r["address"] for r in aggregate}
    for row in room.get("lane1", ()):
        if row["address"] in grouped_rows or row.get("gist") or row["address"] in hidden:
            continue
        step = "F2" if row["kind"] == "ouroboros" else "F7" if row.get("owner_side", row["kind"] == "human") else "F7p"
        label = "my reply" if step == "F2" else "words"
        text = (mv._row_pointer(row, label) if "bare:" + row["address"] in gone.get(step, ()) or
                step == "F7" and row["address"] in gone.get(step, ()) else mv._row_opening(row, label)
                if row["address"] in gone.get(step, ()) else row["line"])
        hold("untold in this room", text)
    for item in [*room.get("origins", ()), *(room.get("project_words") or ())]:
        hold("untold in this room", item.get("short") if item.get("address") in gone.get("F7", ()) else item.get("line", item.get("text", "")))
    for group in snapshot.conversations:
        form, counted = conversations.group_form(group, gone), group["key"] in gone.get("F6a", ())
        for remote in group["rooms"]:
            for note in remote["notes"] if not counted and form < 2 else ():
                hold("notes", mv._note_pointer(note) if form else "\n".join(mv._note_lines([note])),
                     f"memory_read(node_id='{note['id']}')", unmarked="lead_rule" not in note)
    for _unread, owners, _rest in _conversation_sources(snapshot).values():
        for word in owners:
            hold("owner words in other rooms", mv._row_pointer(word, "words") if word["id"] in gone.get("F7", ()) else word["line"])
    return {"tokens": totals, "largest": sorted(ready, key=lambda entry: (-entry[0], entry[1]))[:3]}
