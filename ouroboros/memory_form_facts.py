"""Receipts for the forms actually visible, and numeric prices of what holds the room.

These diagnostics share the renderers' source identities; cached values are only
counts and sizes, outside the persisted snapshot. No rejected full view is drawn.
"""
from ouroboros import memory_view as mv, memory_view_conversations as conversations, memory_view_legacy as legacy
from ouroboros.memory_view_work import works, folded_work_rows


def form_counts(snapshot, level):
    cache = snapshot.__dict__.setdefault("_form_counts", {})
    if level.addressed not in cache:
        cache[level.addressed] = _count_forms(snapshot, level)
    return cache[level.addressed]


def _conversation_sources(snapshot):
    sources = snapshot.__dict__.get("_conversation_sources")
    if sources is None:
        sources = {}
        for group in snapshot.conversations:
            for room in group["rooms"]:
                unread, owner, rest = [], [], []
                for word in room["words"]:
                    (owner if conversations._whole_owner(word) else unread if conversations._unread(word) else rest).append(word)
                sources[room["room_id"]] = unread, owner, rest
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
    add("emergency", "room ranges", bool(aggregate), len(aggregate))
    for key, rows in works(room):
        folded = [r for r in rows if r["address"] in hidden - grouped]
        add("retold", "works", int(bool(folded)), len(folded))
    for row in room.get("lane1", ()):
        if row["address"] in addressed - hidden - grouped:
            add("emergency", "room addresses" if ("bare:" + row["address"] in gone.get("F2", set()) | gone.get("F7p", set()) or row["address"] in gone.get("F7", ())) else "room openings", 1, 1)
    tasks = [t for t in room.get("lane2", ()) if t["first"] in gone.get("F1", ())]
    sole = sum(stage_of(snapshot, "F1", t["first"]) == "task_only" for t in tasks)
    add("emergency", "sole task gists", int(bool(sole)), sole)
    add("host", "task facts", int(bool(tasks) and not sole), len(tasks) - sole)
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
        add("emergency", label + " lists", int(bool(aggregated)), len(aggregated))
        for entry in entries:
            if entry in aggregated:
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
    sources = _conversation_sources(snapshot)
    index = conversations.source_index(snapshot)
    counted = set(gone.get("F6a", ()))
    lost, groups_counted, global_unread = 0, 0, []
    for group in snapshot.conversations:
        count = group["key"] in counted
        form = conversations.group_form(group, gone)
        if count:
            groups_counted += 1
        elif form:
            add("host", "conversation facts", 1, 1)
            gists = index[group["key"]]["gists"]
            shown = gists[-1:] if form == 3 else gists
            add("retold", "work gists", len(shown), len(shown))
            untold = index[group["key"]]["untold"]
            if form >= 2:
                # The emergency line also names gists/notes no longer printed.
                missing = untold + len(gists) - len(shown) + sum(len(r["notes"]) for r in group["rooms"])
                add("emergency", "all gists" if form == 2 else "last gist", 1, max(1, missing))
            no_gist = sum(not t.get("gist") for r in group["rooms"] for t in r["tasks"])
            add("host", "tasks without gist", int(bool(no_gist)), no_gist)
        elif not count and index[group["key"]]["gists"]:
            tasks = {t.get("task") for r in group["rooms"] for t in r["tasks"] if t.get("gist")}
            folded = {w.get("work_id") or w.get("task_id") for r in group["rooms"] for w in r["words"]
                      if w["id"] in gone.get("F6w", ())}
            all_work = {w.get("work_id") or w.get("task_id") for r in group["rooms"] for w in r["words"]}
            gists = conversations._gist_lines(group, excluded=tasks | (all_work - folded))
            add("retold", "work gists", len(gists), len(gists))
        for remote in group["rooms"]:
            unread, owner, rest = sources[remote["room_id"]]
            if count:
                lost += len(rest) + len(remote["notes"]) + len(remote["tasks"]) + int((remote.get("folded") or {}).get("count", 0))
                lost += int(not remote["engaged"])
            elif form:
                if not remote["engaged"]:
                    add("host", "silent rooms", 1, 1)
                if form == 1:
                    add("retold", "note leads", len(remote["notes"]), len(remote["notes"]))
            if not (count or form):
                unread = [w for w in unread if w["id"] in gone.get("F6w", ())]
            if count and "unread" in counted:
                global_unread += unread
            else:
                add("host", "unread", int(bool(unread)), len(unread))
            if not count and (fold := remote.get("folded")):
                add("retold", "read and folded", 1, fold["count"])
            for word in owner:
                if word["id"] in gone.get("F7", ()):
                    add("emergency", "other owner addresses", 1, 1)
            if not (form or count):
                for word in rest:
                    raw, bare = word["id"] in gone.get("F6w", ()), "bare:" + word["id"] in gone.get("F6w", ())
                    if (raw or bare) and not conversations._told(word):
                        add("emergency", "other addresses" if bare else "other openings", 1, 1)
                    elif raw and (word.get("again") or word["kind"] == "place_fact"):
                        add("host", "place facts", 1, 1)
    add("emergency", "conversation count", int(bool(groups_counted)), lost or groups_counted)
    add("window", "unread count", int(bool(global_unread)), len(global_unread))


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
                ready.append({"tokens": tokens, "reader": reader})
    pages = [e for e in snapshot.story if not e.get("told_by") and e.get("kind") != "legacy"]
    aggregated = [e for e in pages if e["id"] in gone.get("F5", ())] if gone.get("F5a") else []
    if aggregated:
        hold("story lists", mv._story_line(aggregated), unmarked=any("lead_rule" not in e for e in aggregated))
    for entry in pages:
        ident, kind = entry["id"], entry.get("kind")
        if entry in aggregated:
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
    if aggregate_notes:
        hold("notes", mv._notes_line(aggregate_notes), unmarked=any("lead_rule" not in e for e in aggregate_notes))
    for note in notes:
        if note not in aggregate_notes:
            text = mv._note_pointer(note) if note["id"] in gone.get("F8", ()) else "\n".join(mv._note_lines([note]))
            hold("notes", text, f"memory_read(node_id='{note['id']}')", unmarked="lead_rule" not in note)
    hidden, _ = folded_work_rows(room, gone)
    addressed = hidden | {i.removeprefix("bare:") for step in ("F2", "F7p", "F7") for i in gone.get(step, ())}
    aggregate = [r for r in room.get("lane1", ()) if r["address"] in addressed and not r.get("immune")] if gone.get("F7a") else []
    if aggregate:
        hold("untold in this room", mv._rows_line(room, aggregate))
    for row in room.get("lane1", ()):
        if row in aggregate or row.get("gist") or row["address"] in hidden:
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
    return {"tokens": totals, "largest": sorted(ready, key=lambda entry: (-entry["tokens"], entry["reader"]))[:3]}
