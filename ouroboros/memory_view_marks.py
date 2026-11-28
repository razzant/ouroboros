"""The marks a view keeps: which a role holds, how each is named, and how the block draws them.

``_capture_marks`` reads a role's acting marks once per capture (``mark_in_view``: its mark scope; this room's
late reviews whatever the scope); ``_marks_text`` and ``_late_reviews_text`` draw ``## Marks I keep in view`` and
``## Late review feedback (host)`` in the view's block C (``memory_view._room_parts``); ``mark_entry`` names a mark
the same way where a running focus meets a later change (``mark_anchors``). A late review's full source travels as
the reader this actor can resolve (``source_reader``). Nothing here writes a mark.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping

from ouroboros import chat_chain
from ouroboros.chronicle_store import DELEGATED_FOCUS_ROLES, ChronicleStore, is_late_review_mark
from ouroboros.mark_anchors import describe as describe_anchor
from ouroboros.memory_inventory import minute as _minute
from ouroboros.memory_view_legacy import INDENT, indented as _indented
from ouroboros.memory_view_parts import mapping as _mapping, rendered as _rendered

if TYPE_CHECKING:
    from ouroboros.memory_view import ViewSpec


def _mark_target(target: Any) -> str:
    target = _mapping(target)
    if target.get("kind") == "chronicle":
        return f"node {target.get('id')}"
    if target.get("kind") == "chat_row" and target.get("row_sha256"):
        return chat_chain.format_address(dict(target))
    if target.get("kind") == "task":
        return f"task {target.get('task_id')}"
    location = f" at {target['location']}" if target.get("location") else ""
    return f"source {target.get('path') or target.get('kind') or 'not recorded'}{location}"


def _portable_ref(target: Mapping[str, Any]) -> dict:
    return {key: target[key] for key in ("kind", "root", "path", "size", "sha256", "task_id") if key in target}


def source_reader(target: Mapping[str, Any], source_read: Any = None) -> str:
    """The reader a retained-source target names to every focus: the writer's explicit reader, else
    ``memory_read`` over the compact ref, verified on read. An author's own ``read`` recipe resolves in
    the author's drive only, so it never travels (BIBLE P1: a source this actor can actually resolve)."""
    if isinstance(source_read, Mapping):
        return f"{source_read.get('tool')}({json.dumps(source_read.get('arguments') or {}, ensure_ascii=False, sort_keys=True)})"
    ref = _portable_ref(target)
    return f"memory_read(source_ref={json.dumps(ref, ensure_ascii=False, sort_keys=True)})"


def _late_review_entry(mark: Mapping[str, Any], label: Callable[..., str]) -> Dict[str, Any]:
    """A host-kept late review of this room (``late_review_marks``): its words, when it settled, its full source and
    its release. Other rooms' late reviews are no entries: the waiting facts count them (``memory_waiting.measure``)
    and ``memory_read(room_id='late_reviews')`` lists them by room and work."""
    review, target = _mapping(mark.get("review")), _mapping(mark.get("target_ref"))
    return {"id": mark["id"], "scope": "room", "room": label(mark.get("room_id")),
            "text": str(mark.get("text") or ""),
            "by": "host", "date": _minute(mark.get("ts")).split(" ")[0], "target": _mark_target(target), "quote": "",
            "late_review": {"task_id": str(review.get("task_id") or ""),
                            **({"source_ref": _portable_ref(target)} if target.get("kind") == "task_source"
                               and not review.get("source_read") else {}),
                            "source": source_reader(target, review.get("source_read")),
                            "sha256": str(target.get("sha256") or "")}}


def mark_in_view(mark: Mapping[str, Any], spec: ViewSpec, *, running: bool = False) -> bool:
    """Whether a role keeps a mark in view by its scope: none, every room's, or this room's and the global ones.
    A focus already running meets later changes of its room's (``spec.room_id``: the room it is in now) and the
    global ones only, whatever its start view holds (``mark_anchors.append_mark_changes``)."""
    if spec.marks == "none":
        return False
    return (spec.marks == "all" and not running) or mark.get("scope") == "global" or str(mark.get("room_id")) == spec.room_id


def mark_entry(mark: Mapping[str, Any], label: Callable[..., str]) -> Dict[str, Any]:
    """A mark as every view names it: its words, who marked it and when, its target, its quote while visibility
    is full, and the action it is tied to."""
    author = _mapping(mark.get("author"))
    by = _mapping(author.get("focus")).get("role") or (
        "the old dialogue writer" if author.get("kind") == "legacy_helper" else author.get("kind") or "not recorded")
    return {"id": mark["id"], "scope": str(mark.get("scope") or "room"),
            "room": label(mark.get("room_id")), "text": str(mark.get("text") or ""), "by": str(by),
            "date": _minute(mark.get("ts")).split(" ")[0], "target": _mark_target(mark.get("target_ref")),
            "quote": str(mark.get("quote") or "") if mark.get("visibility", "full") == "full" else "",
            "anchor": describe_anchor(mark["anchor"]) if isinstance(mark.get("anchor"), dict) else ""}


def _capture_marks(store: ChronicleStore, spec: ViewSpec, label: Callable[..., str], marks=None) -> List[Dict[str, Any]]:
    """The acting marks a role keeps in view (``mark_in_view``).

    Late review feedback the host keeps in view belongs to the room of its work whatever the role's
    mark scope: that room holds its words, every role alike. Other rooms' late reviews are counted in the
    waiting facts and listed by room and work on request, never enumerated here: a wake once drew one
    pointer per foreign mark, hundreds of addresses on every start, beside the waiting line that already
    counted them (one source delivered twice; BIBLE P1).
    """
    if spec.marks == "none":
        return []
    marks = store.active_marks(None) if marks is None else marks
    entries = []
    for mark in (mark for mark in marks if mark_in_view(mark, spec)):
        if is_late_review_mark(mark):
            if str(mark.get("room_id")) == spec.room_id:
                entries.append(_late_review_entry(mark, label))
            continue
        entries.append(mark_entry(mark, label))
    return sorted(entries, key=lambda entry: (entry["scope"] == "global", "" if entry["scope"] == "global" else entry["room"]))


_LATE_REVIEW_INTRO = ("Reviewer feedback that settled after its work ended, kept here by the host for this room "
                      "until {who}. It is advice, not a decision or a veto; releasing it says it was dealt with, "
                      "not that it was agreed with.")


def _late_reviews_text(marks: List[Dict[str, Any]], role: str, part=_rendered) -> str:
    """This room's late reviews; a delegated helper reads who releases them (``chronicle_store``), not a call it cannot make."""
    helper = role in DELEGATED_FOCUS_ROLES
    who = ("the integrating mind releases it; as a delegated helper I report what matters in it" if helper
           else "I release it with my reason")
    lines = part(["## Late review feedback (host)", "", _LATE_REVIEW_INTRO.format(who=who)])
    for mark in marks:
        late = mark["late_review"]
        release = "" if helper else f"; memory_mark(release_ids=['{mark['id']}'], reason=…)"
        lines += part([f"- [{mark['room']}; kept since {mark['date']}] " + _indented(mark["text"]).lstrip(),
                  f"{INDENT}full source (sha256 {late['sha256'][:12]}): {late['source']}{release}"], "late_review", [mark])
    return "\n".join(lines)


def _marks_text(marks: List[Dict[str, Any]], part=_rendered) -> str:
    lines = part(["## Marks I keep in view", "Each mark ends with its id; I release marks with memory_mark(release_ids=['<id>', …], reason=…).", ""])
    for mark in marks:
        entry = [f"- [{mark['scope']}; {mark['room']}] " + _indented(mark["text"]).lstrip()
                     + f" — marked by {mark['by']} on {mark['date']}; target {mark['target']}; "
                     + (f"also shown after a task's first call matching {mark['anchor']}; " if mark.get("anchor") else "")
                     + f"id {mark['id']}"]
        if mark.get("quote"):
            entry.append(f"{INDENT}quote: " + _indented(mark["quote"]).lstrip())
        lines += part(entry, "mark", [mark])
    return "\n".join(lines)
