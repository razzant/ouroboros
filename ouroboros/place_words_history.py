"""Words a place change removes or replaces, kept before the change lands, and read back newest first.

``places._transact``, the one writer of ``state/places.json``, writes through ``keeping``: each change's records
before and after (``PlaceRecords``) go to ``retain``, and a note, a place's description or a circle's words that does
not stay exactly as it was (text, author, time) is appended to ``state/place_words_history.jsonl`` before the document
is replaced, with where it stood, the revisions, the act that removed it, who made the change and the focus that wrote
it. A row that cannot be written changes nothing (``place_history_unavailable``): the history row comes first, as a
knowledge note's does.

My human's words (``places.OWNER_VOICE``) leave only by their own act: a change they make or order from their own door,
or one made in a place focus their own words woke now (``presence_context.focus_owner_now``), in that place or in its
circle. A wake or another speaker is refused (``place_owner_words_owner_only``); my own words stay mine to change.
Disconnecting is an act on the place, never refused for its words, which are kept here like any other.

The journal has no remover: durable memory (BIBLE P3), kept with the data root. It enters no context by itself;
``configure_presence`` ``inspect`` reads ``earlier`` by place id even after disconnect, or by circle id even when
no place ever joined. An unreadable journal leaves the current view readable, with its exact error in ``earlier``.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ouroboros.places import AUTHOR_OWNER, OWNER_VOICE, PlaceError, PlaceRecords, _document, _records, note_identity
from ouroboros.utils import append_jsonl, utc_now_iso

JOURNAL_NAME = "place_words_history.jsonl"
_CIRCLE_WORDS = ("text", "text_author", "text_at")


def journal_path(drive_root: Any) -> Path:
    return Path(drive_root) / "state" / JOURNAL_NAME


def keeping(drive_root: Any, mutate: Any, author: str = "", by: Mapping[str, Any] | None = None) -> Any:
    """``mutate`` as the locked writer of the place document (``update_json_locked``): what the next document removes
    or replaces is kept first (``retain``, with ``by``'s facts of who acts); ``None`` writes nothing."""
    def write(current: dict) -> dict | None:
        before = _document(current)
        after = mutate(before)
        if after is not None:
            retain(drive_root, _records(before), _records(after), after["revision"], author=author, **(by or {}))
        return after
    return write


def removed_words(before: PlaceRecords, after: PlaceRecords) -> list[dict[str, Any]]:
    """Each word ``before`` holds that ``after`` does not keep exactly, with where it stood and the act that removed it
    (a place's words on ``disconnect``; a circle's with the places in it then)."""
    words = []
    for key, place in before.places.items():
        kept = after.places.get(key)
        stays = Counter(kept.notes if kept else ())
        where = {"place_id": key, "place_revision": place.revision}
        for index, note in enumerate(place.notes):
            if stays[note]:
                stays[note] -= 1
                continue
            words.append({"kind": "note", "action": "remove_note" if kept else "disconnect", **where,
                          "note_id": note_identity(key, index, note), "text": note.text, "author": note.author,
                          "at": note.at})
        described = before.description(key)
        if described and described != (after.description(key) if kept else None):
            words.append({"kind": "description", "action": "set_description" if kept else "disconnect", **where,
                          **described})
    for circle in before.circles:
        now = after.circle(circle["circle_id"])
        if circle["text"] and (now is None or any(now[field] != circle[field] for field in _CIRCLE_WORDS)):
            words.append({"kind": "circle_words", "action": "edit_circle" if now else "delete_circle",
                          "circle_id": circle["circle_id"], "circle_name": circle["name"],
                          "circle_revision": circle["revision"],
                          "places": [place.place_id for place in before.members(circle["circle_id"])],
                          "text": circle["text"], "author": circle["text_author"], "at": circle["text_at"]})
    return words


def _holders(word: Mapping[str, Any]) -> list:
    return word.get("places", [word.get("place_id")])


def retain(drive_root: Any, before: PlaceRecords, after: PlaceRecords, revision: int, *, author: str = "",
           owner_turn: bool = False, owner_speaks_in: str = "", writer: Mapping[str, Any] | None = None) -> None:
    """Keep what the change to document ``revision`` removes or replaces, inside the place lock, before it is written.

    ``author`` signs the change's new words (``places.AUTHOR_*``); ``owner_turn``: my human's own door ordered it,
    whoever composed its words; ``owner_speaks_in``: the place whose focus my human's words woke now; ``writer``: the
    focus that made it (``knowledge.focus_signature``), none from my human's app."""
    words = removed_words(before, after)
    if not words:
        return
    owner_act = author == AUTHOR_OWNER or owner_turn
    if not owner_act and any(word["author"] in OWNER_VOICE and word["action"] != "disconnect"
                             and owner_speaks_in not in _holders(word) for word in words):
        raise PlaceError("place_owner_words_owner_only", "author", "place_owner_words_owner_only: these are my "
                         "human's words; only their own act removes or replaces them.")
    row = {"changed_at": utc_now_iso(), "document_revision": revision,
           "changed_by": AUTHOR_OWNER if owner_act else author,
           **({"owner_speaks_in": owner_speaks_in} if owner_speaks_in else {}),
           **({"writer": dict(writer)} if writer else {}), "words": words}
    if not append_jsonl(journal_path(drive_root), row, ensure_record_boundary=True, require_lock=True):
        raise PlaceError("place_history_unavailable", "history", "place_history_unavailable: the words this change "
                         "removes could not be kept in their history, so nothing changed; the same call can be retried.")


def earlier(drive_root: Any, place_id: str = "", *, circle_id: str = "") -> list[dict[str, Any]]:
    """The words that left this place (including its circle then), or the named circle if no place id is given,
    newest change first (one change's words in the order they stood): each as it stood (kind, exact text, author,
    time, its note id or circle) with the act that
    removed or replaced it, when, who made the change and the focus that wrote it. No journal yet: none; an unreadable
    one is a ``place_history_unreadable`` error, never "none"."""
    from ouroboros.knowledge_versions import _journal_lines

    path = journal_path(drive_root)
    found = []
    try:
        if not path.exists():
            return []
        for _offset, line in _journal_lines(path, json.dumps(place_id or circle_id, ensure_ascii=False).encode("utf-8")):
            try:
                row = json.loads(line)
            except ValueError:  # a torn append: its change was refused, so its words still stand
                continue
            change = {key: row[key] for key in ("changed_at", "changed_by", "owner_speaks_in", "writer") if key in row}
            found += [{**{key: value for key, value in word.items() if key != "places"}, **change}
                      for word in row.get("words", [])
                      if (place_id in _holders(word) if place_id else word.get("circle_id") == circle_id)]
    except OSError as exc:
        raise PlaceError("place_history_unreadable", str(path), f"place_history_unreadable: {path}: {exc}") from exc
    return found
