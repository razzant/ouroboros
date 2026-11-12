"""A work's own telling and replies, from one capture; no store reads or task inference.

The sending task, not a relay's recipient, owns its speech. A work's last reply
stays whole after its intermediate replies have yielded to its recorded gist.
A relay that quotes my human's words belongs to no work: no gist stands for them.
"""
from __future__ import annotations

from ouroboros.text_sentences import first_sentence as first_sentence


def row_range(first, last):
    """Source coordinates carry their own room and time; a single row needs one address."""
    return first if first == last else f"{first} → {last}"


def row_reader(room_id=None):
    room = f"'{room_id}'" if room_id is not None else "<room from address>"
    return (f"Read row addresses with memory_read(room_id={room}, rows=true, from=<first>, to=<last>); "
            "for a single row use its address for both endpoints.")


def speech_facts(row, author):
    """A spoken row's author facts; ``owner_quoted``: a relay that quotes my human's words (``relay_quotes``)."""
    from ouroboros.dialogue_provenance import dialogue_speaker, relay_quotes
    transport = row.get("transport") or {}
    relay = row.get("relay") or {}
    task = ((relay.get("sender") or {}).get("task_id") if row.get("type") == "relay_entry" else
            row.get("task_id") or ((row.get("transport") or {}).get("origin") or {}).get("task_id"))
    if author.get("via") == "own_account":
        # An inbound account event carries the arrival's turn id, not a sender
        # (presence_runner.capture_presence_event). Confirmed deliveries are outward.
        task = ""
    return {"task_id": str(task or ""), "via": author.get("via", ""),
            "speaker": [str(transport.get(k) or "") for k in ("provider", "account_id")] + [dialogue_speaker(row)],
            "text": str(row.get("text") or ""), "reply_to": (row.get("reply_to") or {}).get("address", ""),
            "author": author.get("label", ""), "relay_summary": str(relay.get("summary") or ""),
            **({"owner_quoted": True} if relay_quotes(row) else {})}


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


def task_gist_line(task):
    gist = task["gist"]
    return f"[{gist['address']}; task {task['task']}] — summary: «{gist['summary']}»"


def task_gists_before_work_lines(room, gone):
    """F1 counts host facts; a work's telling stays here until its work line takes it."""
    if gone.get("F7a"):
        return []
    waiting = {key for key, _rows in works(room) if not any("work:" + key + suffix in gone.get("F2", ())
                                                          for suffix in (":progress", ":final"))}
    return [t for t in room.get("lane2", ()) if t.get("gist") and t.get("task") in waiting
            and t["first"] in gone.get("F1", ())]


def conversation_task_lines(group):
    """Each work's task telling once within this conversation, beside all its host facts."""
    told = {(room.get("folded") or {}).get("task") for room in group["rooms"]}
    lines = {}
    for room in group["rooms"]:
        lines[room["room_id"]] = []
        for task in room["tasks"]:
            lines[room["room_id"]].append(task_line(task, folded=task.get("task") in told))
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
