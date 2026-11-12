"""Receipts for the forms actually visible, and numeric prices of what holds the room.

These diagnostics share the renderers' source identities; cached values are counts
and sizes, outside the persisted snapshot, and the holding prices are keyed by the
priced text itself, for the snapshot's life. No rejected full view is drawn.
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
    add("emergency", "story quotes", 0, len(set(gone.get("F7", ())) & mv.story_quote_rows(snapshot).keys()))
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
    lost = _remote_counts(snapshot, gone, add)
    snapshot.__dict__.setdefault("_telling_emergency", {})[level.addressed] = bool(
        lost or counts["emergency"].keys() - {"conversation count"})
    return counts


def _remote_counts(snapshot, gone, add):
    from ouroboros.memory_view_conversation_forms import local_key
    from ouroboros.memory_view_events import totals
    sources, index = _conversation_sources(snapshot), conversations.source_index(snapshot)
    cache = snapshot.__dict__.setdefault("_conversation_form_counts", {})
    lost = groups_counted = global_unread = event_arrivals = 0
    telling_lost = False
    for group in snapshot.conversations:
        data = index[group["key"]]
        state = local_key(group, gone, data)
        count = state[3] == 4
        aggregate_unread = count and "unread" in gone.get("F6a", ())
        key = (*state, aggregate_unread)
        if key not in cache:
            cache[key] = _group_counts(group, data, sources, state, aggregate_unread)
        facts, missing, unread, telling = cache[key]
        for kind, name, lines, n in facts:
            add(kind, name, lines, n)
        lost += missing
        telling_lost |= telling
        if count:
            event_arrivals += totals(group["rooms"])[1]
        groups_counted += count
        global_unread += unread
    add("emergency", "conversation count", int(bool(groups_counted)), lost or groups_counted)
    add("host", "event counts", int(bool(event_arrivals) and not groups_counted), event_arrivals)
    add("window", "unread count", int(bool(global_unread)), global_unread)
    return telling_lost


def _group_counts(group, data, sources, state, aggregate_unread):
    """Only numbers for one local form; grouping another conversation cannot change them."""
    from ouroboros.memory_view_conversation_forms import _visible_gists
    from ouroboros.memory_view_events import totals
    form, words, owners, event_count = state[3:]
    count = form == 4
    facts, lost, global_unread = [], 0, 0
    telling_lost = False
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
            add("emergency", "untold by count under gists" if form == 2 else "untold and older gists by count", 1, missing)
        no_gist = [sum(not t.get("gist") for t in r["tasks"]) for r in group["rooms"]]
        add("host", "tasks without gist", sum(map(bool, no_gist)), sum(no_gist))  # one line per room holding them
    elif not count and data["gists"]:
        n = len(_visible_gists(group, data, words))
        add("retold", "work gists", n, n)
        # Told words that yielded under their work's task line here: that line's sentence is their telling.
        lined = {t.get("task") for r in group["rooms"] for t in r["tasks"] if t.get("gist")}
        under = [w for w in data["words"] if w["id"] in words and data["kinds"][w["id"]] == "told"
                 and (w.get("work_id") or w.get("task_id")) in lined]
        add("retold", "task gists", len({w.get("work_id") or w.get("task_id") for w in under}), len(under))
    for remote in group["rooms"]:
        unread, owner, rest = sources[remote["room_id"]]
        if count:
            lost += len(rest) + len(remote["notes"]) + len(remote["tasks"]) + int((remote.get("folded") or {}).get("count", 0))
            lost += int(not remote["engaged"] and not remote.get("events"))
            kinds = data["rooms"][remote["room_id"]]
            telling_lost |= bool(kinds["untold"] or kinds["told"] or remote["notes"] or remote.get("folded")
                                 or any(t.get("gist") for t in remote["tasks"]))
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
        add("host", "unread in rooms without my human's word or my reply",
            int(bool(data["overheard"][1])), len(data["overheard"][1]))
    return tuple(facts), lost, global_unread, telling_lost


def holding_room(snapshot, level):
    cache = snapshot.__dict__.setdefault("_holding_room", {})
    if level.addressed not in cache:
        cache[level.addressed] = _holding_room(snapshot, level)
    return cache[level.addressed]


def _holding_room(snapshot, level):
    """Disjoint source kinds; untold remote words retain their whole-text price under every form."""
    gone = {step: set(ids) for step, ids in level.addressed}
    totals, ready, without_lead, counts, seen = {}, {}, {}, {}, set()
    prices = snapshot.__dict__.setdefault("_holding_text_prices", {})
    def price(text):
        if text not in prices:
            prices[text] = (mv.json_chars(text) + 5) // 4 if text else 0
        return prices[text]
    def hold(kind, text, reader, *, label="", unmarked=0, tokens=None):
        tokens = price(text) if tokens is None else tokens
        if tokens:
            totals[kind] = totals.get(kind, 0) + tokens
            if unmarked:
                without_lead[kind] = without_lead.get(kind, 0) + int(unmarked)
            unit = ready.setdefault(reader, [0, {}, {}])
            unit[0] += tokens
            unit[1][kind] = unit[1].get(kind, 0) + tokens
            unit[2][label] = None
    pages = [e for e in snapshot.story if not e.get("told_by") and e.get("kind") != "legacy"]
    aggregated = [e for e in pages if e["id"] in gone.get("F5", ())] if gone.get("F5a") else []
    grouped_ids = {e["id"] for e in aggregated}
    seen.update(grouped_ids)
    if aggregated:
        hold("story lists", "\n".join([mv._story_line(aggregated),
             *(line for e in aggregated for line in mv.page_quote_lines(e, gone.get("F7", ()), short=True))]),
             "memory_read(story=true, start=0)", label="older pages and parts",
             unmarked=sum("lead_rule" not in e for e in aggregated))
    for entry in pages:
        ident, kind = entry["id"], entry.get("kind")
        if entry["id"] in grouped_ids:
            continue
        seen.add(ident)
        text = (mv._page_short(entry, gone.get("F7", ())) if ident in gone.get("F5", ()) else
                mv._page_short(entry, gone.get("F7", ()), lead=True) if ident in gone.get("F5l", ()) else
                "\n".join(mv._page_lines(entry, gone.get("F7", ()))))
        hold({"account": "accounts", "part": "parts"}.get(kind, "pages without part"), text,
             f"memory_read(node_id='{ident}')", label=f"{entry.get('label', kind)} {ident}",
             unmarked=kind != "account" and "lead_rule" not in entry)
    for key, records in legacy.pointer_rooms(snapshot.story).items():
        hold("old story", legacy.room_pointer(records) if key in gone.get("F3", ()) else
             "\n".join(legacy.retold_record(e) for e in records), f"memory_read(room_id='{key}')",
             label=records[0].get("label", key), unmarked=len(records))
    room = snapshot.room or {}
    own_reader, own_label = _rows_reader(room.get("room_id"), room.get("lane1", ())), room.get("label", "this room")
    for entry in [*room.get("legacy", ()), *room.get("under_parts", ())]:
        if entry["id"] in seen:
            continue
        seen.add(entry["id"])
        hold("room records", mv._retold(entry, entry["id"] in gone.get("F4", ()) or entry.get("standing", False)),
             f"memory_read(node_id='{entry['id']}')", label=entry["id"],
             unmarked=not entry.get("standing") and "lead_rule" not in entry)
    notes = room.get("notes", ())
    aggregate_notes = [e for e in notes if e["id"] in gone.get("F8", ())] if gone.get("F8a") else []
    grouped_notes = {e["id"] for e in aggregate_notes}
    if aggregate_notes:
        hold("notes", mv._notes_line(aggregate_notes), f"memory_read(room_id='{room['room_id']}')",
             label=own_label, unmarked=sum("lead_rule" not in e for e in aggregate_notes))
    for note in notes:
        if note["id"] not in grouped_notes:
            text = mv._note_pointer(note) if note["id"] in gone.get("F8", ()) else "\n".join(mv._note_lines([note]))
            hold("notes", text, f"memory_read(node_id='{note['id']}')", label=note["id"], unmarked="lead_rule" not in note)
    hidden, work_lines = folded_work_rows(room, gone)
    addressed = hidden | {i.removeprefix("bare:") for step in ("F2", "F7p", "F7") for i in gone.get(step, ())}
    aggregate = [r for r in room.get("lane1", ()) if r["address"] in addressed and not r.get("immune")] if gone.get("F7a") else []
    if aggregate:
        hold("room ranges", mv._rows_line(room, aggregate), _rows_reader(room["room_id"], aggregate), label=own_label)
    else:
        by_address = {r["address"]: (key, rows) for key, rows in works(room) for r in rows}
        for address, line in work_lines.items():
            key, rows = by_address[address]
            hold("work gists", line, _rows_reader(room["room_id"], [r for r in rows if r["address"] in hidden]), label="work " + key)
    folded = {key for key, _rows in works(room) if any("work:" + key + suffix in gone.get("F2", ())
                                                     for suffix in (":progress", ":final"))} if not aggregate else set()
    only_gists = {t["first"] for t in task_gists_before_work_lines(room, gone)}
    for task in room.get("lane2", ()):
        if task.get("gist") and (task["first"] not in gone.get("F1", ()) or task["first"] in only_gists) and task.get("task") not in folded:
            gist = task["gist"]
            hold("task gists", task_gist_line(task) if task["first"] in only_gists else task_line(task), f"memory_read(room_id='{gist['room_id']}', rows=true, "
                 f"from='{gist['address']}', to='{gist['address']}')", label="work " + task["task"])
    grouped_rows = {r["address"] for r in aggregate}
    for row in room.get("lane1", ()):
        if row["address"] in grouped_rows or row["address"] in hidden:
            continue
        step = "F2" if row["kind"] == "ouroboros" else "F7" if row.get("owner_side", row["kind"] == "human") else "F7p"
        label = "my reply" if step == "F2" else "words"
        text = (mv._row_pointer(row, label) if "bare:" + row["address"] in gone.get(step, ()) or
                step == "F7" and row["address"] in gone.get(step, ()) else mv._row_opening(row, label)
                if row["address"] in gone.get(step, ()) else row["line"])
        hold("replies of completed works" if row.get("gist") else "untold in this room", text, own_reader, label=own_label)
    for item in room.get("origins", ()):
        hold("work origins", item.get("text", ""), "memory_read(owner_words=true)", label=own_label)
    for item in room.get("project_words") or ():
        hold("untold in this room", item.get("short") if item.get("address") in gone.get("F7", ()) else item.get("line", ""),
             _rows_reader(room.get("room_id"), [item]), label=own_label)
    _hold_conversations(snapshot, gone, hold, price, counts)
    units = []
    for reader, (tokens, kinds, labels) in sorted(ready.items(), key=lambda pair: (-pair[1][0], pair[0]))[:3]:
        names = "; ".join(sorted(kinds, key=lambda kind: (-kinds[kind], kind)))
        units.append([tokens, f"{names}, {' / '.join(labels)}: {reader}"])
    return {"tokens": dict(sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))),
            "without_lead": without_lead, "counts": counts,
            "largest": units}


def _rows_reader(room_id, rows):
    addresses = [r["address"] for r in rows if r.get("address")]
    span = f", from='{addresses[0]}', to='{addresses[-1]}'" if addresses else ""
    return f"memory_read(room_id='{room_id}', rows=true{span})"


def _holding_pieces(entries, kind, pieces, label):
    """Keep the joined form's price, including newline/rounding, while naming its source readers.

    A shared count or gist can cover several rooms. Its existing price is divided
    among those readers; the category total never grows from splitting the unit.
    """
    chars, priced = 0, 0
    for text, readers in pieces:
        if not text:
            continue
        chars += mv.json_chars(text) + (2 if chars else 0)
        total = (chars + 5) // 4
        tokens, priced = total - priced, total
        weight = sum(n for _reader, n in readers)
        used, through = 0, 0
        for reader, n in readers:
            through += n
            share = tokens * through // weight
            entries.append((kind, share - used, reader, label, 0))
            used = share


def _hold_conversations(snapshot, gone, hold, price, counts):
    from ouroboros import memory_view_conversation_forms as conv
    index = conv.source_index(snapshot)
    cache = snapshot.__dict__.setdefault("_holding_conversations", {})
    for group in snapshot.conversations:
        data = index[group["key"]]
        state = conv.local_key(group, gone, data)
        form, words, owners, _events = state[3:]
        aggregate_unread = form == 4 and "unread" in gone.get("F6a", ())
        key = (*state, aggregate_unread)
        if key not in cache:
            entries = []
            def add(kind, text, source, label=group["label"], unmarked=0):
                entries.append((kind, price(text), source, label, unmarked))
            def rows(kind, source_kind, render, include=lambda w: True):
                pieces = []
                for remote in group["rooms"]:
                    kept = [w for w in data["rooms"][remote["room_id"]][source_kind] if include(w)]
                    if kept:
                        pieces.append(("\n".join(render(w) for w in kept), [(_rows_reader(remote["room_id"], kept), 1)]))
                _holding_pieces(entries, kind, pieces, group["label"])
            rows("untold in other conversations", "untold", lambda w: w["line"])
            if not form:
                rows("replies of completed works", "told", lambda w: w["line"], lambda w: w["id"] not in words)
            rows("owner words in other rooms", "owner", lambda w: mv._row_pointer(w, "words") if w["id"] in owners else w["line"])
            for remote in group["rooms"]:
                for note in remote["notes"] if form < 2 else ():
                    add("notes", mv._note_pointer(note) if form else "\n".join(mv._note_lines([note])),
                        f"memory_read(node_id='{note['id']}')", note["id"], int("lead_rule" not in note))
                unread = data["rooms"][remote["room_id"]]["unread"]
                if not aggregate_unread:
                    if form and not remote.get("overheard") and unread:
                        add("unread host facts", conv._unread_line(unread), _rows_reader(remote["room_id"], unread))
                    elif not form:
                        whole = [w for w in unread if w["id"] not in words]
                        folded = [w for w in unread if w["id"] in words]
                        add("unread words in other conversations", "\n".join(w["line"] for w in whole),
                            _rows_reader(remote["room_id"], whole))
                        if folded:
                            add("unread host facts", conv._unread_line(folded), _rows_reader(remote["room_id"], folded))
                if form < 4 and (fold := remote.get("folded")):
                    add("read and folded", conv._fold_line(remote),
                        f"memory_read(room_id='{remote['room_id']}', rows=true, from='{fold['from']}', to='{fold['to']}')")
            if form and not aggregate_unread:
                readers = [(_rows_reader(r["room_id"], data["rooms"][r["room_id"]]["unread"]),
                            len(data["rooms"][r["room_id"]]["unread"])) for r in group["rooms"]
                           if r.get("overheard") and data["rooms"][r["room_id"]]["unread"]]
                _holding_pieces(entries, "unread host facts", [(conv._overheard_line(group, data), readers)], group["label"])
            if form < 4:
                gists = (data["gists"][-1:] if form == 3 else data["gists"]) if form else conv._visible_gists(group, data, words)
                lines = conv._gist_lines(group, gists=gists, replies=data["replies"] if form else None)
                pieces = []
                for (work, gist), line in zip(gists, lines):
                    spans = data["replies"].get(work, gist["address"]) if form else gist["address"]
                    readers = []
                    for span in spans.split(", "):
                        ends = span.split(" → ")
                        readers.append((_rows_reader(ends[0].partition("@")[0].removeprefix("row:"),
                                                     [{"address": ends[0]}, {"address": ends[-1]}]), 1))
                    pieces.append((line, readers))
                _holding_pieces(entries, "work gists", pieces, group["label"])
            if not form:
                for remote in group["rooms"]:
                    for task in remote["tasks"]:
                        if gist := task.get("gist"):
                            add("task gists", task_line(task), _rows_reader(gist["room_id"], [gist]))
            cache[key] = tuple(entries)
        for kind, tokens, reader, label, unmarked in cache[key]:
            hold(kind, "", reader, tokens=tokens, label=label, unmarked=unmarked)
        if data["untold"]:
            entry = counts.setdefault("untold in other conversations", {"conversations": 0, "counted_rows": 0})
            entry["conversations"] += 1
            entry["counted_rows"] += data["untold"] if form >= 2 else 0
        if data["owners"]:
            entry = counts.setdefault("owner words in other rooms", {"conversations": 0})
            entry["conversations"] += 1


def telling_emergency(snapshot, level):
    """An unread-only or event count stays an emergency form, but owes no telling."""
    form_counts(snapshot, level)
    return snapshot.__dict__["_telling_emergency"][level.addressed]


def physical_floor(snapshot, level, mode):
    """Nano, or a level retaining a stage from remote_count through owner."""
    if mode == "nano" or any(s in ("F6a", "F5a", "F8a", "F7", "F7a") for s, _ids in level.addressed):
        return True
    cache = snapshot.__dict__.setdefault("_physical_rungs", {})
    for step, ids in level.addressed:
        if step not in ("F2", "F6w", "F7p"):
            continue
        # Equal rungs from later frames reuse one numeric fact as well.
        physical = cache.get(ids)
        if physical is None:
            physical = cache[ids] = any(i.startswith("bare:") for i in ids)
        if physical:
            return True
    return False


def holding_lines(held, *, short=False):
    """Host facts only: kinds, their source counts and the largest addressable units."""
    kinds = []
    for kind, tokens in held["tokens"].items():
        # Remote untold words are priced whole, what folding them would have to tell, though most stand by count.
        whole = " at whole-text size" if kind == "untold in other conversations" else ""
        if short:
            kinds.append(f"{kind} {tokens:,}{whole}")
            continue
        detail = []
        if n := held["without_lead"].get(kind):
            detail.append(f"{n} records without lead")
        counts = held["counts"].get(kind, {})
        if "conversations" in counts:
            detail.append(f"{counts['conversations']} conversations")
        if counts.get("counted_rows"):  # a zero names no omission
            detail.append(f"{counts['counted_rows']} rows by count")
        if kind == "owner words in other rooms":
            detail.append("whole until I close their conversation with a page")
        kinds.append(f"{kind} {tokens:,}{whole}" + (" (" + "; ".join(detail) + ")" if detail else ""))
    line = "Holding the room (tokens): " + ", ".join(kinds) + "."
    return [line] if short else [line, *(
        f"{tokens:,} tokens, {description}" for tokens, description in held["largest"])]
