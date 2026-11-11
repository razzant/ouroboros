"""A work's own telling and replies, from one capture; no store reads or task inference.

The sending task, not a relay's recipient, owns its speech. A work's last reply
stays whole after its intermediate replies have yielded to its recorded gist.
"""
from __future__ import annotations

import re


def first_sentence(text):
    return re.split(r'(?<=[.!?。！？])\s+', str(text).strip(), maxsplit=1)[0]


def speech_facts(row, author):
    relay = row.get("relay") or {}
    task = ((relay.get("sender") or {}).get("task_id") if row.get("type") == "relay_entry" else
            row.get("task_id") or ((row.get("transport") or {}).get("origin") or {}).get("task_id"))
    if author.get("via") == "own_account":
        task = ""
    return {"task_id": str(task or ""), "via": author.get("via", ""),
            "text": str(row.get("text") or ""), "reply_to": (row.get("reply_to") or {}).get("address", ""),
            "author": author.get("label", ""), "relay_summary": str(relay.get("summary") or "")}


def gists(by_room, format_address):
    """A terminal's own gist, keyed by its work, across the rows this capture already read."""
    found = {}
    for room, entries in by_room.items():
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
    # One telling site per work. Current replies represent their work here;
    # otherwise prefer the room holding the terminal, then its first projection.
    sites = {}
    for item in rooms:
        for row in [*item.get("lane1", item.get("words", ())), *item.get("lane2", item.get("tasks", ()))]:
            if row.get("gist"):
                key = row.get("work_id") or row.get("task_id") or row.get("task")
                priority = (0 if item is room else 1 if item["room_id"] == row["gist"]["room_id"] else 2)
                if key not in sites or priority < sites[key][0]:
                    sites[key] = priority, item["room_id"]
    for item in rooms:
        for row in [*item.get("lane1", item.get("words", ())), *item.get("lane2", item.get("tasks", ()))]:
            if row.get("gist"):
                key = row.get("work_id") or row.get("task_id") or row.get("task")
                row["telling_here"] = sites[key][1] == item["room_id"]


def task_line(task, *, folded=False):
    """Keep the host's status and readers when this work's telling stands elsewhere."""
    if task.get("gist") and (folded or not task.get("telling_here", True)):
        return task["line"].replace(" — summary: «" + task["gist"]["summary"] + "»", "", 1)
    return task["line"]


def works(room):
    """Completed works with a gist, in last-reply order; no gist means no ordinary reply fold."""
    found = {}
    for row in room.get("lane1", ()):
        if row.get("gist") and row.get("kind") == "ouroboros":
            key = row.get("work_id") or row.get("task_id")
            if key:
                found.setdefault(key, []).append(row)
    return sorted(found.items(), key=lambda pair: pair[1][-1]["pos"])


def work_line(room, rows):
    first, last = rows[0], rows[-1]
    return (f"[{first['ts']}; work {last.get('task_id') or 'relay'}] «{last['gist']['summary']}»; "
            f"{len(rows)} replies, {sum(r['chars'] for r in rows)} chars; "
            f"memory_read(room_id='{room['room_id']}', rows=true, from='{first['address']}', to='{last['address']}')")


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
            lines[rows[-1]["address"]] = work_line(room, taken)
    return hidden, lines
