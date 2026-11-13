"""Waiting classes and their closure acts; neutral measures of one captured memory view."""
from collections import Counter
from types import MappingProxyType

from ouroboros.chronicle_store import is_late_review_mark
from ouroboros.late_review_marks import SURFACE_NAMES, clean_pass, landing_words, mark_outcome, outcome_words


WAITING_CLASSES = MappingProxyType({
    "late_reviews": {"subject": "late reviews", "action": "memory_mark",
                     "parameter": ("release_ids",),
                     "reader": "memory_read(room_id='late_reviews', start=0)",
                     "act": "memory_mark(release_ids=[...], reason=...)",
                     "focus": "integrating", "moment": "when folding"},
    "notes": {"subject": "unsealed mind notes", "action": "chronicle_write",
              "parameter": ("covers", "note_ids"),
              "reader": "memory_read(room_id='notes', start=0)",
              "act": "chronicle_write(kind='page', room_id=..., covers={'note_ids': [...]})",
              "focus": "integrating", "moment": "when folding"},
    "projects": {"subject": "open projects", "action": "close_project",
                 "parameter": ("project_id",),
                 "reader": "list_projects(closed=false, oldest_first=true)",
                 "act": "close_project(project_id=..., note=...)",
                 "focus": "integrating", "moment": "when folding"},
    "conversations": {"subject": "conversations whose words no page holds", "action": "chronicle_write",
                      "parameter": ("covers", "conversation"),
                      "reader": "memory_read(room_id='conversations', start=0)",
                      "act": "chronicle_write(kind='page', room_id=..., covers={'to': ..., 'conversation': true})",
                      "focus": "integrating", "moment": "when folding"},
})


def late_reviews(marks):
    return [mark for mark in marks if is_late_review_mark(mark)]


def mind_notes(notes):
    return [(room, note) for room, items in notes.items() for note in items if note.get("author_kind") == "mind"]


def measure(marks, notes, projects, conversations=()):
    """Captured active marks, unsealed notes, registry view (None when its census is unavailable) and my other
    conversations (``memory_view_conversations.capture``'s groups: what of them no page holds, no extra read)."""
    from ouroboros.memory_view_conversations import owed

    reviews, unsealed, debts = late_reviews(marks), mind_notes(notes), owed(conversations)

    def facts(items, date):
        stamps = [str(date(item) or "") for item in items]
        return {"count": len(items), "oldest": min(stamps) if stamps and all(stamps) else "unknown" if stamps else "none"}

    rooms = Counter(room for room, _note in unsealed)
    outcomes = [mark_outcome(mark) for mark in reviews]  # the facts published with each mark: no read here
    return {"late_reviews": {**facts(reviews, lambda mark: mark.get("ts")),  # how many rooms hold them: a count of the
                             "in_rooms": len({str(mark.get("room_id")) for mark in reviews}),  # set, never a pick
                             "clean_pass": sum(1 for outcome in outcomes if clean_pass(outcome)),
                             "unrecorded": sum(1 for outcome in outcomes if outcome is None)},
            "notes": {**facts(unsealed, lambda item: item[1].get("date")),
                      "rooms": sorted(rooms.items(), key=lambda item: (-item[1], item[0]))[:3]},
            "projects": (facts(projects, lambda project: project.get("last_active_at")) if projects is not None
                         else {"count": "unknown", "oldest": "unknown"}),
            "conversations": {**facts(debts, lambda item: str(item["earliest"])[:10]),
                              "largest": [[item["group"]["label"], item["chars"]] for item in  # JSON-stable pairs
                                          sorted(debts, key=lambda item: -item["chars"])[:3]]}}


def text(waiting):
    if not waiting:
        return ""
    lines = []
    for name, cls in WAITING_CLASSES.items():
        facts = waiting.get(name)
        if facts is None:  # a view captured before this class existed
            continue
        rooms = ", ".join([*(f"room {room}: {count}" for room, count in facts.get("rooms", ())),
                           *(f"{label} ~{chars} chars" for label, chars in facts.get("largest", ()))])
        held = facts.get("in_rooms")  # absent in a view captured before the count existed
        where = f" in {held} room{'' if held == 1 else 's'}" if held else ""
        # Late reviews: how many say PASS with nothing to read and, while any remain, how many keep no outcome on
        # the mark (the list reads those records). Absent in a view captured before these counts; facts, never a
        # suggestion.
        told = (f", {facts['clean_pass']} PASS with no findings"
                + (f", {facts['unrecorded']} with no outcome kept on the mark (the list reads the record where one exists)"
                   if facts.get("unrecorded") else "")
                if facts.get("count") and "clean_pass" in facts else "")
        lines.append(f"{cls['subject']} {facts['count']}{where}, oldest {facts['oldest']}{told}"
                     + (f", most in {rooms}" if rooms else "") + f"; {cls['reader']}; act {cls['act']}")
    return "; waiting for me at the start: " + " | ".join(lines)


def listing(kind, marks, notes, label=None, outcome=mark_outcome, landing=None):
    """Exact ids and dates for the existing memory_read's paged list modes; no source prose. Late reviews are grouped
    by the room of their work, the room with the oldest first (``label`` names a room), each with the review it is
    (``late_review_marks.SURFACE_NAMES``), its work, its date, its outcome in words and its exact reader: what a
    wake once held as one pointer per mark is read here by choice. ``outcome`` reads one mark's outcome; the
    reader passes one that also reads the review record of a mark older than its own outcome facts, so the head
    can say how many of all listed are a PASS with no findings. ``landing`` (the reader's) says what is recorded
    of the change a commit review read; with it the head counts the PASS commit reviews with no commit under
    that review and no merged receipt of their task: approved work that may never have landed, by these two
    records alone."""
    if kind == "late_reviews":
        rows = []
        for m in late_reviews(marks):
            told, became = outcome(m), landing(m) if landing else None
            rows.append({"id": m["id"], "room": str(m.get("room_id")), "date": str(m.get("ts") or "unknown"),
                         "task": str((m.get("review") or {}).get("task_id") or "unknown"),
                         "review": SURFACE_NAMES.get(str((m.get("author") or {}).get("surface") or ""),
                                                     str((m.get("author") or {}).get("surface") or "review")),
                         "outcome": "; ".join(filter(None, (outcome_words(told), landing_words(became)))),
                         "clean": clean_pass(told),
                         "idle": isinstance(became, dict) and not (became.get("commit") or became.get("unread"))
                                 and not (became.get("pr") and (became.get("status") or "merged") == "merged")
                                 and isinstance(told, dict) and told.get("verdict") == "PASS"})
        rows.sort(key=lambda row: (row["date"], row["id"]))
        rooms = {}
        for row in rows:
            rooms.setdefault(row["room"], []).append(row)
        lines = []
        for room, items in sorted(rooms.items(), key=lambda item: (item[1][0]["date"], item[0])):
            lines.append(f"room {room}" + (f" ({label(room)})" if label else "") + f": {len(items)}")
            lines += [f"  - {row['date']}; {row['review']} of task {row['task']}; {row['outcome']}; "
                      f"memory_read(node_id='{row['id']}')" for row in items]
        return (f"{WAITING_CLASSES[kind]['subject']}: {len(rows)} in {len(rooms)} room{'' if len(rooms) == 1 else 's'}"
                f"; {sum(row['clean'] for row in rows)} PASS with no findings"
                + (f"; {sum(row['idle'] for row in rows)} PASS commit reviews with no commit under that review and no "
                   "merged receipt of their task" if landing else ""), "\n".join(lines))
    rows = [{"id": n["id"], "room": room, "date": str(n.get("date") or "unknown")} for room, n in mind_notes(notes)]
    rows.sort(key=lambda row: (row["date"], row["id"]))
    return (f"{WAITING_CLASSES[kind]['subject']}: {len(rows)}",
            "\n".join(f"{row['id']}; room {row['room']}; {row['date']}; memory_read(node_id='{row['id']}')" for row in rows))
