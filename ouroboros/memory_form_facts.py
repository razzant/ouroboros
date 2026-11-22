"""Receipts for the forms actually visible, and numeric prices of what holds the room.

These diagnostics share the renderers' source identities. Floor receipts cache only
counts and prices; an explicit inspection can draw a retained snapshot's ordinary
forms to account for every escaped character, without keeping its words in the map.
"""
from ouroboros import memory_view as mv, memory_view_conversations as conversations, memory_view_legacy as legacy
from ouroboros.memory_view_work import works, folded_work_rows, task_line, task_gist_line, task_gists_before_work_lines


def _aggregate_source(source, room_id):
    """Display endpoints need not enclose the selected rows in the stored stream."""
    fold = "from" in source
    task = source.get("task")
    result = {"selection": ("Open lane-1 rows folded through the finish, excluding owner speech, relays, "
                            "linked rows and rows also shown in this focus's room." if fold else
                            "Captured open lane-2 rows grouped by root_task_id or, if absent, task_id.")}
    if fold:
        result["gist_source"] = (_composition_source({"address": source["finish_address"]}, room_id)
                                 if source.get("finish_address") else {"read_gap": "Finish row address was not captured."})
    if room_id is not None and source.get("stored_bounds"):
        first, last = source["stored_bounds"]
        args = {"room_id": room_id, "rows": True, "from": first, "to": last}
        if task and not fold:
            args["task_id"] = task
        result.update(range_read={"tool": "memory_read", "arguments": args},
                      row_membership_gap="Inclusive stored bounds. The reader can return rows outside the captured selection.")
    else:
        result["row_membership_gap"] = "Stored bounds were not captured. Display endpoints are navigation only."
        if task and not fold:
            result["navigation"] = {"tool": "get_task_result", "arguments": {"task_id": task}}
    return result


def _composition_source(source, room_id=None, *, citation=False):
    """Only captured coordinates and facts; never a row's words, gist, quote or correction."""
    if "key" in source and "rooms" in source:
        return _composition_group(source)
    fields = ("id", "kind", "room_id", "address", "first", "last", "ts", "last_ts", "date", "period",
              "source_period", "period_basis", "revision", "status", "written", "selection", "told_by",
              "part", "task", "task_id", "work_id", "immune", "owner_side", "owner_quoted", "ref",
              "scope", "target", "accounts", "count", "span", "earliest", "latest", "running_tasks",
              "key", "pos", "last_pos", "from", "to", "engaged", "overheard", "newest",
              "transfer", "written_at", "stated_period", "byline", "source", "carrier", "supplier",
              "schedule_id", "field", "supplier_task_id", "supplied_at", "missing", "revision_known",
              "status_now", "current_revision", "later_corrections", "folded_into", "sha256", "again", "again_room")
    result = {key: source[key] for key in fields if key in source}
    if isinstance(source.get("author"), str):  # a row's speaker label; chronicle records already carry their byline
        result["author"] = source["author"]
    if isinstance(source.get("origin_ref"), dict):
        result["origin_ref"] = {key: source["origin_ref"][key] for key in
            ("chat_id", "client_message_id", "ts", "text_sha256", "row_sha256") if key in source["origin_ref"]}
    room_id = source.get("room_id", room_id)
    if room_id is not None:
        result["room_id"] = room_id
    address = source.get("address")
    if not address and "carrier" in source and str(source.get("ref") or "").startswith("row:"):
        address = result["address"] = source["ref"]  # e.g. a quoted target's own reference
    first, last = source.get("first"), source.get("last")
    if citation and (source.get("missing") or not source.get("revision") or not source.get("revision_known")):
        result["read_gap"] = ("The cited record was missing from this chronicle at capture." if source.get("missing") else
                              "The captured facts do not establish that this journal held the cited revision.")
    elif citation:
        result["read"] = {"tool": "memory_read", "arguments": {"node_id": source["id"], "revision": source["revision"]}}
    elif "from" in source or (source.get("task") and isinstance(first, str) and isinstance(last, str)):
        result.update(_aggregate_source(source, room_id))
    elif room_id is not None and (address or (isinstance(first, str) and first and first == last)):
        result["read"] = {"tool": "memory_read", "arguments": {"room_id": room_id, "rows": True,
                                                               "from": address or first, "to": address or first}}
    elif isinstance(address, str) and address.startswith("row:"):
        # A row's address names the chat it was written in; a moved or held row shows
        # elsewhere, so only a captured showing room makes an exact reader.
        result["read_gap"] = ("No showing room was captured; the address prefix is where the row was written "
                              "(navigation only), and a held row may live in another store.")
    elif source.get("id") and not address:
        args = {"node_id": source["id"]}
        reader = "read"
        if source.get("kind") == "mark":
            result["read_basis"] = "Mark record by ID; its visibility and release state are read as of now."
        elif source.get("revision"):
            args["revision"] = source["revision"]
            result["read_basis"] = "Record text at the captured revision; the reader also reports later relationships."
        else:
            reader = "navigation"
            result["read_gap"] = "The captured record revision is unknown."
        result[reader] = {"tool": "memory_read", "arguments": args}
    elif (source.get("task") or source.get("task_id")) and "carrier" not in source:
        # A carried owner row's task_id names its carrier, not the work that wrote it.
        result["navigation"] = {"tool": "get_task_result", "arguments": {"task_id": source.get("task") or source["task_id"]}}
        result["row_membership_gap"] = "Capture has no row-address bounds; the task reader reads the work, not this historical projection."
    else:
        result["read_gap"] = "No exact source reader in this captured metadata; current room/catalogue is navigation only."
    if source.get("again"):  # a repeat line names its earlier copy, not this row's words
        result["earlier"] = ({"read": {"tool": "memory_read", "arguments": {"room_id": source["again_room"], "rows": True,
                                                                            "from": source["again"], "to": source["again"]}}}
                             if source.get("again_room") else
                             {"read_gap": "Snapshot predates the earlier copy's room; the rendered line names it."})
    if result.get("running_tasks") is not None:
        result["running_work"] = [{"task_id": task, "navigation": {"tool": "get_task_result", "arguments": {"task_id": task}}}
                                  for task in result["running_tasks"]]
        if result["running_tasks"]:
            result["running_basis"] = "The work and its retries as they are now, not a source captured with this view."
    if source.get("gist"):
        result["gist_source"] = _composition_source({key: source["gist"][key] for key in
            ("room_id", "address", "ts") if key in source["gist"]})
    if "sources" in source:
        result["cited_sources"] = [_composition_source(s, citation=True) for s in source.get("sources") or ()]
    for key in ("words", "notes", "tasks"):
        if isinstance(source.get(key), (list, tuple)):
            result[key] = [_composition_source(item, room_id) for item in source[key]]
    if source.get("folded"):
        result["folded"] = _composition_source(source["folded"], room_id)
    if source.get("events"):
        event = source["events"]
        result["events"] = {key: event[key] for key in ("count", "unknown_count", "unknown_chars", "unknown_last") if key in event}
        result["events"]["samples"] = {key: _composition_source(event[key], room_id) for key in
                                      ("first", "last", "unknown_first") if event.get(key)}
        if "bounds" in event:
            bounds = event["bounds"]
            result["events"]["bounds"] = [_composition_source(bound, room_id) for bound in bounds]
            result["events"]["selection"] = "Inbound lane-1 arrivals in captured open and sealed entries; bounds are inclusive, not an enumeration."
            if bounds and room_id is not None:
                result["events"]["range_read"] = {"tool": "memory_read", "arguments": {"room_id": room_id, "rows": True,
                    "from": bounds[0]["address"], "to": bounds[-1]["address"]}}
        result["events"]["membership_gap"] = "Capture keeps samples and counts, not every arrival address."
    if "facts" in source:
        result["facts"] = {key: source["facts"][key] for key in
            ("rows", "earliest", "latest", "running_tasks", "pages", "last_covered", "gaps") if key in source["facts"]}
    return result


def _portable_source(ref):
    """A task-relative handle also needs its owner to travel through memory_read."""
    if (isinstance(ref, dict) and ref.get("kind") == "task_source" and ref.get("root") == "artifact_store"
            and ref.get("task_id") and ref.get("path") and ref.get("sha256") and ref.get("size") is not None):
        return {"tool": "memory_read", "arguments": {"source_ref": ref}}


def _refusal_source(refusal):
    """A helper's refused fold: the retold record that stayed open, and its answer's own reader when retained."""
    path = refusal.get("path")
    if reader := _portable_source(refusal.get("source_ref")):
        answer = {"read": reader}
    elif path:
        answer = {"canonical_path": path, "read_gap": "Canonical data path only; no portable source reference was captured."}
    else:
        answer = {"read_gap": "No portable reference or canonical path for the helper's answer was captured."}
    return {"kind": refusal.get("kind"), "retold_record": _composition_source(
                {key: refusal[key] for key in ("id", "revision") if key in refusal}),
            "answer": answer}


def _mark_source(mark):
    # The mark record has its own native node reader. Its target, including
    # the retained full review, is a distinct source.
    result = _composition_source({**mark, "kind": "mark"})
    if late := mark.get("late_review"):
        reader = _portable_source(late.get("source_ref"))
        result["review"] = {"task_id": late.get("task_id"), "sha256": late.get("sha256"),
                            **({"read": reader} if reader else
                               {"read_gap": "No portable reader for the full review was captured; the mark record identifies its target."})}
    return result


def _sealed_source(record):
    """A page another room published that seals rows here: read in its own room, never credited to the showing one."""
    result = _composition_source(record)
    if "room_id" not in record:
        result["publishing_room_gap"] = "Snapshot predates the sealing page's own room; the showing room did not publish it."
    return result


def _change(kind, change_id, target, gap="Snapshot predates the change record's id; only the changed record is named."):
    """A printed signed change: its immutable record at its own revision, and the record it changes (as of that
    correction; a decision binds no revision). Never its words."""
    result = {"kind": kind, "target": target}
    if change_id:
        result.update(id=change_id, read={"tool": "memory_read", "arguments": {"node_id": change_id, "revision": change_id}})
    else:
        result["read_gap"] = gap
    if kind == "correction" and change_id:
        result["target_read"] = {"tool": "memory_read", "arguments": {"node_id": target, "revision": change_id}}
    else:
        result["target_navigation"] = {"tool": "memory_read", "arguments": {"node_id": target}}
    return result


def _rendered_changes(kind, form, source):
    """Readers of the signed changes the drawn form prints, never their words: a record's ``fixes`` where
    ``_fix_lines`` runs, an account's ``told_fixes`` and its sources' later changes (``account_lines``).
    A nested change keeps both readers: the change's, and the frozen revision its citing edge names
    (``source_change_lines``); a directly cited draft's later acceptance names its decision where captured."""
    if kind == "story_record" and form != "address" or kind == "room_record" and form == "whole" and "stamp" in source:
        return [_change(fix["kind"], fix.get("id"), fix["target"]) for fix in source.get("fixes") or ()]
    if kind != "account":
        return []
    changes = [_change(fix["kind"], fix.get("id"), fix["target"]) for fix in source.get("told_fixes") or ()]
    for src in source.get("sources") or ():
        if src.get("missing"):
            continue
        ident, printed, ids = str(src.get("id")), len(src.get("later_fixes") or ()), src.get("later_corrections") or []
        changes += [_change("correction", ids[n] if len(ids) == printed else None, ident,
                            "Captured later-correction ids do not match the printed corrections; only the source is named.")
                    for n in range(printed)]
        if src.get("status_now") == "rejected" and src.get("status") != "rejected":
            changes.append(_change("rejection", src.get("rejection_id"), ident))
        elif src.get("status") == "draft" and src.get("status_now") == "accepted":
            changes.append(_change("acceptance", src.get("acceptance_id"), ident,
                                   "Snapshot predates the captured acceptance id; only the accepted source is named."))
        for nested in src.get("nested_changes") or ():
            event = nested.get("event") or {}
            name = "correction" if event.get("kind") == "correction" else "acceptance" if event.get("accepted") else "rejection"
            member = str(nested.get("source_id"))
            cited = ({"read": {"tool": "memory_read", "arguments": {"node_id": member, "revision": nested["revision"]}}}
                     if nested.get("revision") else
                     {"navigation": {"tool": "memory_read", "arguments": {"node_id": member}},
                      "read_gap": "Part member; the member revision the citing edge used was not recorded."})
            changes.append({**_change(name, event.get("id"), member), "via": list(nested.get("via") or ()),
                            "cited_source": cited})
    return changes


def _composition_group(group, *, identity_only=False):
    # ``running`` is rendered focus prose, never a metadata source. New captures
    # retain IDs by room separately; old snapshots must not reverse-parse prose.
    rooms = [{key: room[key] for key in ("room_id", "running_tasks") if key in room}
             for room in group["rooms"]] if identity_only else group["rooms"]
    result = {"key": group["key"], "rooms": [_composition_source(room) for room in rooms]}
    if identity_only and group.get("running"):  # the line or heading prints each running work's reader, its focus if authored
        result["running_focus_gap"] = ("A running work prints as its task reader, with authored focus text beside it "
                                       "only where the capture found one; that text was read at capture, no immutable "
                                       "reference to it was kept, and the task reader reads the work as it is now.")
    if "catalogue_start" in group:
        result["catalogue_start"] = group["catalogue_start"]
    if any("running_tasks" not in room for room in group["rooms"]):
        result["running_tasks_gap"] = "Snapshot predates structured running IDs by room; rendered focus is not parsed."
    return result


def _composition_key(key):
    """Piece identities contain sets; serialize their members deterministically, without reinterpreting them."""
    if isinstance(key, (set, frozenset)):
        return sorted(key)
    return [_composition_key(item) for item in key] if isinstance(key, (tuple, list)) else key


def memory_composition(snapshot, level):
    """Exact, disjoint metadata for the two rendered memory blocks, with no physical-floor note.

    The renderer calls ``part`` as it emits lines; ``_room_parts`` and conversation
    ``pieces`` supply section boundaries. No inventory of alternative rungs is priced here.
    Escaped characters add; tokens are rounded once per block, never per row.
    """
    from ouroboros.memory_floor import tokens_of

    gone, room = dict(level.addressed), snapshot.room or {}
    hidden, _ = folded_work_rows(room, gone)
    work_sources = {rows[-1]["address"]: [row for row in rows if row["address"] in hidden]
                    for _key, rows in works(room)}
    sections, fragments, scope = [], [], "story"

    def part(lines, kind="framing", sources=(), form="whole", room_id=None):
        if lines:
            sources = work_sources[sources[0]["address"]] if kind == "work" else list(sources)
            fragments.append({"kind": kind, "form": form,
                "sources": [(_composition_group(s, identity_only=True) if kind == "conversation_identity" else
                            _refusal_source(s) if kind == "refusal" else
                            _mark_source(s) if kind in ("mark", "late_review") else
                            _sealed_source(s) if kind == "sealed_by" else
                            _composition_source(s, room_id if room_id is not None else
                            room.get("room_id") if scope == "room" else None)) for s in sources],
                "text_json_chars": sum(mv.json_chars(line) for line in lines),
                "separator_before": "\n" if fragments else "", "line_joins": len(lines) - 1,
                "separator_json_chars": 2 * (len(lines) - 1 + bool(fragments))})
            if changes := [change for s in sources for change in _rendered_changes(kind, form, s)]:
                fragments[-1]["changes"] = changes
            if form == "without_gist":  # the line keeps the host's status and readers; the telling stands elsewhere
                for drawn in fragments[-1]["sources"]:
                    if "gist_source" in drawn:
                        drawn["gist_alternative"] = {**drawn.pop("gist_source"), "basis":
                            "This span's line renders without the telling; it stands on the span that shows it."}
            if form == "whole" and kind in ("story_record", "room_record"):
                # A whole page prints its writer's captured terminal rows; the node reader re-resolves them live.
                for drawn, entry in zip(fragments[-1]["sources"], sources):
                    if entry.get("terminal_links") and (kind == "story_record" or "stamp" in entry):
                        drawn["terminal_links_gap"] = (
                            "Host-linked terminal rows were resolved against live history at capture and print with "
                            "their own captured row reader; the node reader re-resolves them at read time, so rows "
                            "written after capture appear there and are no member of this span.")
        return lines

    def section(key, text):
        nonlocal fragments
        price = sum(p["text_json_chars"] + p["separator_json_chars"] for p in fragments)
        if price != mv.json_chars(text):
            raise ValueError("memory composition does not cover the renderer's section")
        sections.append((key, fragments, price))
        fragments = []
        return price

    story = mv._story_text(snapshot, dict(mv._story_key(level)), part=part)
    section(("story",), story)
    story_sections, sections = sections, []

    def piece(key, make):
        nonlocal scope
        scope = key[0]
        text = make()
        if text and not fragments:
            if key[0] == "owner_words":
                metadata = snapshot.owner_words_sources
                part([text], key[0], metadata.get("rows", ()))
                if metadata:
                    fragments[0]["capture"] = {name: metadata[name] for name in ("gaps", "absence", "schedules") if name in metadata}
                else:
                    fragments[0]["source_gap"] = "Captured governing block has no per-row coordinates; not a reconstructed reader."
            else:
                part([text], key[0])
        return section(key, text)

    mv._room_parts(snapshot, gone, piece, part=part)

    def block(parts):
        rows, identities, total = [], [], 0
        for key, entries, size in parts:
            if not size:
                continue
            if rows:  # _room_parts joins nonempty sections with a blank line
                rows.append({"kind": "separator", "form": "literal", "separator": "\n\n",
                             "text_json_chars": 0, "separator_json_chars": 4, "json_chars": 4,
                             "span": [total, total + 4]})
                total += 4
            identities.append(_composition_key(key))  # a room's selected IDs occur once, not in every row
            for row in entries:
                size = row["text_json_chars"] + row["separator_json_chars"]
                row.update(piece=len(identities) - 1, json_chars=size, span=[total, total + size])
                total += size
                rows.append(row)
        return {"json_chars": total, "estimated_tokens": tokens_of(total), "pieces": identities, "parts": rows}

    blocks = {"story": block(story_sections), "room": block(sections)}
    return {"blocks": blocks, "estimated_tokens": sum(b["estimated_tokens"] for b in blocks.values()),
            "represented_story_records": [_composition_source(e) for e in snapshot.story if e.get("told_by")],
            "units": "JSON-escaped characters, excluding enclosing quotes; spans are half-open within each block",
            "token_rule": "tokens_of(story.json_chars) + tokens_of(room.json_chars); two roundings, no floor note",
            "scope": "Ordinary memory only, not selected Low, a calibrated request price or possible savings. "
                     "Parts refer to the block's piece index; only parts contribute size. Sources may recur as readers. "
                     "Completion and permission to fold are not inferred."}


def ordinary_composition(snapshot):
    from ouroboros.memory_floor import FORMS_VERSION, ordinary_level

    level = ordinary_level(snapshot)
    return {"forms_version": FORMS_VERSION, "ordinary_addressed": _composition_key(level.addressed),
            **memory_composition(snapshot, level)}


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
        if key in gone.get("F3", ()):  # the helper's retelling and a room's transfers are each one line
            add("emergency", "old story ranges", len(legacy.room_pointer(records).split("\n")), len(records))
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
            hold("task gists", task_gist_line(task, room["room_id"]) if task["first"] in only_gists else task_line(task), f"memory_read(room_id='{gist['room_id']}', rows=true, "
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
                    # Each room showing its replies here reads its part; else the telling's own row reads in the room
                    # the capture found it in (the chat in its address only where a view recorded no room).
                    own = gist.get("room_id") or gist["address"].partition("@")[0].removeprefix("row:")
                    spans = (form and data["reply_rooms"].get(work)) or [(own, gist["address"], gist["address"])]
                    pieces.append((line, [(_rows_reader(room, [{"address": first}, {"address": last}]), 1)
                                          for room, first, last in spans]))
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
