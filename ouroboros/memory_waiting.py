"""Waiting classes and their closure acts; neutral measures of one captured memory view."""
from collections import Counter
from types import MappingProxyType

from ouroboros.chronicle_store import is_late_review_mark


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
    return {"late_reviews": facts(reviews, lambda mark: mark.get("ts")),
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
        lines.append(f"{cls['subject']} {facts['count']}, oldest {facts['oldest']}"
                     + (f", most in {rooms}" if rooms else "") + f"; {cls['reader']}; act {cls['act']}")
    return "; waiting for me at the start: " + " | ".join(lines)


def listing(kind, marks, notes):
    """Exact ids and dates for the existing memory_read's paged list modes; no source prose."""
    rows = ([{"id": m["id"], "room": str(m.get("room_id")), "date": str(m.get("ts") or "unknown")}
             for m in late_reviews(marks)] if kind == "late_reviews" else
            [{"id": n["id"], "room": room, "date": str(n.get("date") or "unknown")} for room, n in mind_notes(notes)])
    rows.sort(key=lambda row: (row["date"], row["id"]))
    return (f"{WAITING_CLASSES[kind]['subject']}: {len(rows)}",
            "\n".join(f"{row['id']}; room {row['room']}; {row['date']}; memory_read(node_id='{row['id']}')" for row in rows))
