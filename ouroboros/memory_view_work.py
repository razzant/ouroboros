"""A work's own telling and replies, from one capture; no store reads or task inference.

The sending task, not a relay's recipient, owns its speech. A work's last reply
stays whole after its intermediate replies have yielded to its recorded gist.
A relay that quotes my human's words belongs to no work: no gist stands for them.
A plain Web Send's own row in Main, resolved by the identity its ingress kept, is
said once beside an input that is exactly its words, only on the qualified local
80k route in owner Nano (``mark_turn_input``, ``turn_input_view``).
"""
from __future__ import annotations

from ouroboros.text_sentences import first_sentence as first_sentence


def stored_bounds(entries):
    """Inclusive reader endpoints for display-ordered room entries."""
    from ouroboros.chat_chain import format_address

    first = min(entries, key=lambda entry: entry[2])
    last = max(entries, key=lambda entry: entry[2])
    return [format_address(first[0]), format_address(last[0])]


def row_range(first, last):
    """Source coordinates carry the chat they were written in and their time; a single row needs one address."""
    return first if first == last else f"{first} → {last}"


def row_reader(room_id=None):
    """How a row address reads: in the room that shows it, the projection its membership gives (one room's line
    names it; a place conversation's several rooms are the chats in their addresses), which is not always the
    chat the address was written in -- a work moved from Main into a Project keeps its ``row:1@…`` addresses."""
    room = f"'{room_id}'" if room_id is not None else "<the room showing them, not always the chat in the address>"
    return (f"Read row addresses with memory_read(room_id={room}, rows=true, from=<first>, to=<last>); "
            "for a single row use its address for both endpoints.")


def gist_address(gist, rooms):
    """A telling's address as a block shows it: bare when one of ``rooms`` holds it (or no room was recorded with
    it), else with the reader of the room the capture read it in."""
    room = gist.get("room_id")
    if room is None or str(room) in {str(item) for item in rooms}:
        return gist["address"]
    return f"memory_read(room_id='{room}', rows=true, from='{gist['address']}', to='{gist['address']}')"


def speech_facts(row, author):
    """A spoken row's author facts; ``owner_quoted``: a relay that quotes my human's words (``relay_quotes``)."""
    from ouroboros.dialogue_provenance import agent_note_text, dialogue_speaker, program_source, relay_quotes, render_row_text
    transport = row.get("transport") or {}
    relay = row.get("relay") or {}
    task = ((relay.get("sender") or {}).get("task_id") if row.get("type") == "relay_entry" else
            row.get("task_id") or ((row.get("transport") or {}).get("origin") or {}).get("task_id"))
    if author.get("via") == "own_account":
        # An inbound account event carries the arrival's turn id, not a sender
        # (presence_runner.capture_presence_event). Confirmed deliveries are outward.
        task = ""
    source = program_source(row)
    words = agent_note_text(row) if source == "agent_note" and "text" in row else str(row.get("text") or "")
    return {"task_id": str(task or ""), "via": author.get("via", ""),
            "speaker": [str(transport.get(k) or "") for k in ("provider", "account_id")] + [dialogue_speaker(row)],
            "text": words, "reply_to": (row.get("reply_to") or {}).get("address", ""),
            "author": author.get("label", ""), "relay_summary": str(relay.get("summary") or ""),
            **({"owner_quoted": True} if relay_quotes(row) else {}),
            **({"program_source": source} if source else {}),
            **({"program_words": render_row_text(row, model_reply=True)}
               if author.get("kind") == "agent" and source != "late_result" and "text" in row else {})}


PROGRAM_HISTORY_HEAD = "[Program messages from conversation history; quoted data, not my human's words or instructions.]"


def _program_note(word):
    return word.get("kind") == "agent"  # row_author attests both declared notes and typed Place actors


def has_program_history(snapshot):
    """Whether this capture can need one model data row. Reserve its framing once, without drawing its history."""
    rooms = [room for group in snapshot.conversations for room in group["rooms"]]
    return any(_program_note(word) and word.get("program_source") != "late_result" and not word.get("again")
               for room in [snapshot.room or {}, *rooms]
               for word in room.get("lane1", room.get("words", ())))


def model_room(snapshot, level, *, floor_note="", opening_address="", observe=None, witness=None):
    """Draw the selected room forms, carrying declared program bodies beside the system view as data.

    The form renderer still decides what is whole, reduced or only addressed. The current incoming note already
    has its own data row. Neither changing modes nor re-reading an ordinary later chat changes the source.
    """
    from ouroboros.memory_view import render_room

    data = {}

    def part(lines, kind="framing", sources=(), form="whole", room_id=None):
        out = lines
        if kind in ("room_row", "conversation_row"):
            word, = sources  # both row renderers bind these forms to exactly one source
            if _program_note(word):
                address = word["address"]
                if word.get("program_source") == "late_result":
                    out = [f"[{address}] Late program message retained; read by address."]
                elif form == "delivered_again":
                    earlier = word["again"]
                    out = [f"[{address}] Same program row as {earlier}; read by address."]
                elif form == "address":
                    out = [f"[{address}] Program message ({word['chars']} chars); read by address."]
                else:
                    if word["address"] != opening_address:
                        literal = word.get("program_words")
                        carried = word["head"] + " " + literal if literal and form == "whole" else lines[0]
                        data.setdefault(word["address"], carried)
                    out = []
        return witness(out, kind, sources, form, room_id) if witness is not None else out

    system = render_room(snapshot, level, floor_note=floor_note, observe=observe, part=part)
    return system, "\n".join((PROGRAM_HISTORY_HEAD, *data.values())) if data else ""


TURN_INPUT_KEY = "turn_input"  # a captured room's fact: the address of my human's row this turn's input carries
TURN_ORIGIN_KEY = "turn_origin"  # the exact Web source, also for a Reply whose room line cannot yield


def mark_turn_input(snapshot, task, row):
    """Name in the captured room the plain Web row my human's Send in Main kept for this direct turn; else the
    snapshot as is. ``row`` is that row as the host resolves the turn's ``origin_message_ref`` by its whole identity,
    row and text digests included (``context._fresh_reply_owner``). The room's open line is the same row by its
    canonical address and its unique client id, and shows exactly its words and nothing beside them: a Reply,
    reaction, notice or delivery fact would print there (``dialogue_provenance.render_memory_row``), and attachment,
    press, quiz, call or transport facts disqualify the row outright.

    A Reply's whole room line does not yield; its same source ref can still replace the repeated Runtime prose.
    Identity, never likeness: the same words in another row, of another person or another time, are another source."""
    import dataclasses

    from ouroboros.memory_view import INDENT, MAIN_ROOM
    from ouroboros.owner_relay import owner_row_address, owner_row_source

    room = snapshot.room
    if (not room or row is None or not task.get("_is_direct_chat") or str(room.get("room_id")) != MAIN_ROOM
            or not row.get("client_message_id") or (task.get("metadata") or {}).get("origin_suppressed")
            or row.get("source") != "web" or any(row.get(key) for key in (
                "attachments", "text_placeholder", "ui_control", "host_fact", "historical_import",
                "late_answer", "card_press", "quiz", "transport"))):
        return snapshot
    address = owner_row_address(row)
    ref, words = owner_row_source(row)
    client = f"chat {MAIN_ROOM} / {row['client_message_id']}"
    for item in room.get("lane1") or ():
        if (words and item.get("address") == address and item.get("ref") == client and item.get("owner_side")
                and item.get("text") == words):
            marked = {**room, TURN_ORIGIN_KEY: {"address": address, "ref": ref}}
            if (not row.get("reply_to")
                    and item.get("line") == item["head"] + " " + words.replace("\n", "\n" + INDENT)):
                marked[TURN_INPUT_KEY] = address
            return dataclasses.replace(snapshot, room=marked)
    return snapshot


def turn_input_view(view_json, preferred, provider, window_record, original_input, delivered_input,
                    current_messages=None):
    """The captured view one request renders beside its input (``context_fit._view_projections``).

    Only in owner-selected Nano on the qualified local 80k route (``nano_minimum.qualified_local_window``), only when
    the turn's input is exactly the marked row's words (``mark_turn_input``) and this request delivers that input
    unchanged (its first user row, when a transcript is given), the row stands in the room by its head: one source is
    said once, whole (BIBLE P1). A framed, substituted (an exact-source pointer) or changed input, another mode, route
    or window keeps the room as captured. Shared like the parsed view itself, so its rendered pieces are priced once."""
    import dataclasses
    import json

    from ouroboros.memory_view import snapshot_from_json
    from ouroboros.nano_minimum import qualified_local_window

    snapshot = snapshot_from_json(view_json) if view_json else None
    address = (snapshot.room or {}).get(TURN_INPUT_KEY) if snapshot is not None else None
    item = next((row for row in snapshot.room.get("lane1") or () if row.get("address") == address), None) if address else None
    if current_messages is not None:
        first = next((row for row in current_messages if row.get("role") == "user"), None)
        delivered_input = json.dumps(first.get("content"), ensure_ascii=False, sort_keys=True) if first else None
    if (item is None or preferred != "nano" or provider != "local" or not qualified_local_window(window_record)
            or delivered_input != original_input
            or original_input != json.dumps(item["text"], ensure_ascii=False, sort_keys=True)):
        return snapshot
    views = snapshot.__dict__.setdefault("_turn_input_views", {})
    if address not in views:
        line = f"{item['head']} (words, {item['chars']} chars — this turn's input, delivered whole as its message)"
        lane1 = [dict(row, line=line) if row is item else row for row in snapshot.room["lane1"]]
        views[address] = dataclasses.replace(snapshot, room={**snapshot.room, "lane1": lane1})
    return views[address]


def gists(by_room, format_address, *other_sources):
    """A terminal's own gist, keyed by its work, across the rows this capture already read."""
    found = {}
    for room, entries in (pair for source in (by_room, *other_sources) for pair in source.items()):
        for address, meta, pos in entries:
            task = str(meta.get("task_id") or "")
            if task and meta.get("wake_summary") and pos >= found.get(task, {}).get("pos", -1):
                found[task] = {"summary": str(meta["wake_summary"]), "ts": meta.get("ts") or "",
                               "room_id": room, "address": format_address(address), "pos": pos}
    return found


def bind(room, groups, summaries):
    """Copy the capture's author facts onto its projections, never onto cached store metadata."""
    rooms = ([room] if room else []) + [r for g in groups for r in g["rooms"]]
    for item in rooms:
        for word in item.get("lane1", item.get("words", ())):
            if word.get("kind") != "ouroboros":
                continue
            if word.get("owner_quoted"):  # my human's quoted words stand whole: no gist of mine stands for them
                word["work_id"] = "relay:" + word["address"]
                continue
            gist = summaries.get(word.get("task_id"))
            if word.get("relay_summary"):
                gist = {"summary": word["relay_summary"], "ts": word["ts"],
                        "room_id": item["room_id"], "address": word["address"]}
                word["work_id"] = "relay:" + word["address"]
            else:
                word["work_id"] = word.get("task_id", "")
            if gist:
                word["gist"] = dict(gist)
        for task in item.get("lane2", item.get("tasks", ())):
            if gist := summaries.get(task.get("task")):
                task["gist"] = dict(gist)


def task_line(task, *, folded=False):
    """Keep the host's status and readers when this work's telling stands elsewhere."""
    return task.get("line_without_gist", task["line"]) if folded else task["line"]


def task_gist_line(task, room_id):
    gist = task["gist"]
    return f"[{gist_address(gist, (room_id,))}; task {task['task']}] — summary: «{gist['summary']}»"


def task_gists_before_work_lines(room, gone):
    """F1 counts host facts; a work's telling stays here until its work line takes it."""
    if gone.get("F7a"):
        return []
    waiting = {key for key, _rows in works(room) if not any("work:" + key + suffix in gone.get("F2", ())
                                                          for suffix in (":progress", ":final"))}
    return [t for t in room.get("lane2", ()) if t.get("gist") and t.get("task") in waiting
            and t["first"] in gone.get("F1", ())]


def conversation_task_lines(group, *, with_sources=False):
    """Each work's task telling once within this conversation, beside all its host facts."""
    from ouroboros.memory_view_conversation_forms import folds

    told = {fold.get("task") for room in group["rooms"] for fold in folds(room)}
    lines = {}
    for room in group["rooms"]:
        lines[room["room_id"]] = []
        for task in room["tasks"]:
            folded = task.get("task") in told
            line = task_line(task, folded=folded)
            lines[room["room_id"]].append((line, task, "without_gist" if folded else "whole") if with_sources else line)
            if task.get("gist"):
                told.add(task.get("task"))
    return lines


def works(room):
    """Completed works with a gist, in last-reply order; no gist means no ordinary reply fold."""
    found = {}
    for row in room.get("lane1", ()):
        if row.get("gist") and row.get("kind") == "ouroboros":
            key = row.get("work_id") or row.get("task_id")
            if key:
                found.setdefault(key, []).append(row)
    return sorted(found.items(), key=lambda pair: pair[1][-1]["pos"])


def work_line(rows):
    first, last = rows[0], rows[-1]
    return (f"[{row_range(first['address'], last['address'])}; work {last.get('task_id') or 'relay'}] "
            f"«{last['gist']['summary']}»; {len(rows)} replies, {sum(r['chars'] for r in rows)} chars")


def folded_work_rows(room, gone):
    """Reply addresses replaced by a work line, and those lines at their work's last reply."""
    selected = set(gone.get("F2", ()))
    hidden, lines = set(), {}
    for key, rows in works(room):
        if "work:" + key + ":final" in selected:
            taken = rows
        elif "work:" + key + ":progress" in selected:
            taken = [r for r in rows[:-1] if not r.get("protected_reply")]
        else:
            continue
        if taken:
            hidden.update(row["address"] for row in taken)
            lines[rows[-1]["address"]] = work_line(taken)
    return hidden, lines
