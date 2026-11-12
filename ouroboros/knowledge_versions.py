"""One knowledge note's history, read from its shelf's journal: versions, outline, listing, changes.

The journal is ``address.shelf.parent / "knowledge_history.jsonl"``, the append-only record the one
writer (``knowledge.write_knowledge_note``) keeps for a shelf. A row belongs to a note by its
``topic`` and the scope its ``address`` names. Lines are pre-filtered by the note's JSON-encoded
topic before anything is parsed and are read newest first, in blocks from the journal's end, so a
large journal is never parsed whole; a line naming the note that is not JSON is counted as a gap,
never fatal. A version is named by its revision (the sha256 of its exact bytes) and its text is
returned only after it hashes to that revision. Nothing here writes: no index, no cache, no sidecar.

The outline is the one renderer the read, the receipt and the refusal share. Its character ranges
are the ``start_char``/``end_char`` of ``knowledge_read``; a section is the parser's
(``MarkdownHeading.section``), nested sections included, and a heading the parser gives no section
of its own (a setext heading) says so.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

from ouroboros import text_change
from ouroboros.utils import jsonl_window

JOURNAL_NAME = "knowledge_history.jsonl"
UNKNOWN = "unknown"  # knowledge.UNKNOWN_STAMP: a fact the row does not carry
_BLOCK = 8 * 1024 * 1024  # one backward read of the journal; a longer line grows across reads
_FIRST_LINE_CHARS = 120


class VersionUnavailable(ValueError):
    """No verified text for a revision: the message says why (not found, no text, hash mismatch)."""


def journal_path(address: Any) -> Path:
    return address.shelf.parent / JOURNAL_NAME


def _needles(address: Any) -> tuple[bytes, ...]:
    """The topic as a JSON string: the writer's own spelling, and the escaped one another serializer writes."""
    return tuple(dict.fromkeys(json.dumps(address.topic, ensure_ascii=ascii).encode("utf-8") for ascii in (False, True)))


def _belongs(row: Any, address: Any) -> bool:
    """A version row of this note: its topic, a version field, a capture (or an older unstamped row)."""
    if not isinstance(row, dict) or row.get("topic") != address.topic:
        return False
    if ("new_content" not in row and "new_sha256" not in row) or row.get("publication", "source_capture") != "source_capture":
        return False
    where = row.get("address")
    return not isinstance(where, dict) or where.get("scope", address.scope) == address.scope


def _matching_lines(region: bytes, base: int, needles: tuple[bytes, ...]) -> Iterator[tuple[int, bytes]]:
    """``(offset, line)`` of each line of ``region`` (whole lines from ``base``) holding a needle, newest first."""
    end = len(region)
    while end > 0:
        hit = max(region.rfind(needle, 0, end) for needle in needles)
        if hit < 0:
            return
        start = region.rfind(b"\n", 0, hit) + 1
        stop = region.find(b"\n", hit)
        yield base + start, region[start:len(region) if stop < 0 else stop + 1]
        end = start


def _journal_lines(journal: Path, needles: tuple[bytes, ...]) -> Iterator[tuple[int, bytes]]:
    """Every line of the journal holding a needle, newest first, read in blocks from its end."""
    with journal.open("rb") as handle:
        end, carry = os.fstat(handle.fileno()).st_size, b""
        while end > 0:
            start = max(0, end - _BLOCK)
            handle.seek(start)
            data = handle.read(end - start) + carry
            cut = data.find(b"\n") + 1 if start else 0
            if start and not cut:  # one line longer than everything read so far
                carry, end = data, start
                continue
            yield from _matching_lines(data[cut:], start + cut, needles)
            carry, end = data[:cut], start


def _parse(line: bytes) -> Any:
    try:
        return json.loads(line)
    except (ValueError, RecursionError):
        return None


def tail_rows(journal: Path, address: Any, tail_bytes: int) -> Iterator[tuple[int, dict]]:
    """This note's rows within the journal's last ``tail_bytes``, newest first, with their offsets.

    The writer reads it under the shelf lock (one read of the tail, as its own version lookup does);
    an unreadable journal or a line that is not JSON yields nothing."""
    try:
        with journal.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(0)
            tail = jsonl_window(handle, size, tail_bytes)
    except OSError:
        return
    for offset, line in _matching_lines(tail, size - len(tail), _needles(address)):
        row = _parse(line)
        if _belongs(row, address):
            yield offset, row


class NoteHistory:
    """This note's rows over the whole journal, newest first, with the gaps met on the way."""

    def __init__(self, address: Any, extra: bytes = b""):
        self.address, self.extra = address, extra
        self.malformed, self.first_malformed, self.error = 0, None, ""

    def __iter__(self) -> Iterator[tuple[int, dict]]:
        try:
            for offset, line in _journal_lines(journal_path(self.address), _needles(self.address)):
                if self.extra and self.extra not in line:
                    continue
                row = _parse(line)
                if row is None:
                    self.malformed += 1
                    self.first_malformed = offset  # newest first: the last one met is the first in the file
                elif _belongs(row, self.address):
                    yield offset, row
        except FileNotFoundError:
            return
        except OSError as exc:
            self.error = type(exc).__name__

    def gaps(self) -> list[str]:
        out = []
        if self.malformed:
            out.append(f"{self.malformed} journal line(s) naming this note are not JSON and were skipped "
                       f"(the first at byte offset {self.first_malformed})")
        if self.error:
            out.append(f"the journal could not be read to its beginning ({self.error}); older rows are not shown")
        return out


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _referenced_text(address: Any, ref: Any) -> str | None:
    """The ``new_content`` an ``old_content_ref`` names in this same journal, or ``None``."""
    if not isinstance(ref, dict) or ref.get("journal") != f"knowledge:{address.scope}" or type(ref.get("offset")) is not int:
        return None
    try:
        with journal_path(address).open("rb") as handle:
            handle.seek(ref["offset"])
            row = _parse(handle.readline())
    except OSError:
        return None
    text = row.get("new_content") if isinstance(row, dict) else None
    return text if isinstance(text, str) else None


def find_version(address: Any, version: Any) -> tuple[str, dict]:
    """The exact text of one version of this note and the row that wrote it, newest rows first.

    The text is a row's ``new_content`` when its ``new_sha256`` is the version, else a later row's
    inline ``old_content`` or the text its ``old_content_ref`` names when its ``old_sha256`` is; it
    is returned only when it hashes to the version. The row is the one that wrote the version, or
    ``{}`` when none did (an outside edit, or a write before history). ``VersionUnavailable`` says
    why there is no text."""
    if not isinstance(version, str) or len(version) != 64 or version.strip("0123456789abcdef"):
        raise VersionUnavailable("a version is a revision: the 64 lowercase hex characters of its sha256")
    why, held = "", None
    history = NoteHistory(address, version.encode("ascii"))
    for _offset, row in history:
        own = row.get("new_sha256") == version
        if not own and (held is not None or row.get("old_sha256") != version):
            continue
        text = (row.get("new_content") if own else
                row["old_content"] if "old_content" in row else _referenced_text(address, row.get("old_content_ref")))
        verified = isinstance(text, str) and _sha(text) == version  # an empty text is the version sha256(b"")
        if own and (verified or held is not None):
            return (text if verified else held), row
        if verified:
            held = text  # a later row's old side: keep looking for the row that wrote it
        elif isinstance(text, str) and text:
            why = "the stored text does not hash to the version"
        else:
            why = why or "the row that names it holds no text (digest-era)"
    if held is not None:
        return held, {}
    gaps = "; ".join(history.gaps())
    raise VersionUnavailable((why or f"no version {version} of this note in its history")
                             + (f" ({gaps})" if gaps else ""))


def row_facts(row: Mapping[str, Any]) -> dict[str, Any]:
    """Who wrote a version and how: ``{revision, ts, focus, mode, reason}``, ``unknown`` where absent."""
    return {"revision": row.get("new_sha256") or UNKNOWN, "ts": row.get("ts") or UNKNOWN,
            "focus": row.get("focus") or UNKNOWN, "mode": row.get("mode") or UNKNOWN,
            "reason": row.get("reason") or UNKNOWN}


def held_version(journal: Path, address: Any, revision: str, tail_bytes: int) -> dict[str, Any] | None:
    """The version a stale write holds, when this note's rows in the writer's tail have it.

    ``{"facts", "text", "since"}``: that row's facts, its verified text and the facts of the rows
    written after it, oldest first. ``None``: the tail holds no such row, or no text for it."""
    since = []
    for _offset, row in tail_rows(journal, address, tail_bytes):
        if row.get("new_sha256") == revision:
            text = row.get("new_content")
            if not isinstance(text, str) or _sha(text) != revision:
                return None
            return {"facts": row_facts(row), "text": text, "since": [row_facts(item) for item in reversed(since)]}
        since.append(row)
    return None


def focus_text(focus: Any) -> str:
    """A row's focus as one phrase: its role, then what else it carries."""
    if not isinstance(focus, Mapping):
        return str(focus or UNKNOWN)
    carried = [f"{key}={focus[key]}" for key in ("task_id", "root_task_id", "room_id", "chat_id", "place_id",
                                                  "conversation_key")
               if focus.get(key) not in (None, "") and not (key == "root_task_id" and focus[key] == focus.get("task_id"))]
    return " ".join([str(focus.get("role") or UNKNOWN), *carried])


def _own_section(heading: Any) -> Any:
    """The parser's section when it begins at this heading; a setext heading has none of its own."""
    return heading.section if heading.section.start_byte == heading.span.start_byte else None


def _char_offsets(raw: bytes, positions: set[int]) -> dict[int, int]:
    """Character offsets of byte positions on character boundaries, decoded once."""
    out, at, chars = {}, 0, 0
    for position in sorted(positions):
        chars += len(raw[at:position].decode("utf-8"))
        out[position], at = chars, position
    return out


def render_outline(note: Any, label: str = "") -> str:
    """Summary, revision, state, size and every heading with its line and section range."""
    summary = note.summary.strip()
    lines = [f"Outline of {note.address.topic!r} ({note.address.scope}){label}",
             "Summary: " + ("\n  ".join(summary.splitlines()) if summary else "no summary"),
             f"Revision: {note.revision}", f"State: {note.state}", f"Total: {len(note.text)} characters"]
    if note.source is None:
        lines.append("Headings: unknown (the metadata does not parse; the complete note is the default read)")
        return "\n".join(lines) + "\n"
    headings = note.source.headings
    if not headings:
        lines.append("Headings: none")
        return "\n".join(lines) + "\n"
    spans = [_own_section(heading) or heading.span for heading in headings]
    chars = _char_offsets(note.raw, {edge for span in spans for edge in (span.start_byte, span.end_byte)})
    lines.append("Headings (level, title, line, section range as start_char..end_char, size). A section runs "
                 "to the next heading of its level or higher, so a parent's range contains its children's:")
    for heading, span in zip(headings, spans):
        start, end = chars[span.start_byte], chars[span.end_byte]
        own = "" if _own_section(heading) else "; no section of its own (setext heading): the range is its heading"
        lines.append(f"{'#' * heading.level} {heading.title} — line {heading.span.start_line}, "
                     f"chars {start}..{end} ({end - start} chars{own})")
    return "\n".join(lines) + "\n"


def removed_sections(old: Any, new: Any) -> list[dict[str, Any]] | None:
    """Each heading that left, from the old text: ``{heading, chars, first_line}``.

    ``chars`` is its section's size in the old text (its heading alone when the parser gives it no
    section), ``first_line`` the first non-empty line under it, cut to 120 characters. Of a repeated
    heading the later instances are the ones counted as gone. ``None`` when a side does not parse."""
    if old is None:
        return []
    if old.source is None or new.source is None:
        return None
    left = (Counter((h.level, h.title) for h in old.source.headings)
            - Counter((h.level, h.title) for h in new.source.headings))
    headings, gone = old.source.headings, []
    for index in range(len(headings) - 1, -1, -1):
        key = (headings[index].level, headings[index].title)
        if left[key] > 0:
            left[key] -= 1
            gone.append(index)
    out = []
    for index in reversed(gone):
        heading = headings[index]
        own = _own_section(heading)
        span = own or heading.span
        end = (own.end_byte if own is not None else
               headings[index + 1].span.start_byte if index + 1 < len(headings) else len(old.raw))
        under = old.raw[heading.span.end_byte:max(end, heading.span.end_byte)].decode("utf-8")
        first = next((line.strip() for line in under.splitlines() if line.strip()), "")
        out.append({"heading": "#" * heading.level + " " + heading.title,
                    "chars": len(old.raw[span.start_byte:span.end_byte].decode("utf-8")),
                    "first_line": first[:_FIRST_LINE_CHARS]})
    return out


def _listed(row: Mapping[str, Any]) -> dict[str, Any]:
    """The facts a listing line shows, without the row's note texts (a long history is not held whole)."""
    text = row.get("new_content")
    stored = isinstance(text, str) and not row.get("content_digested") and bool(
        text or not row.get("new_sha256") or row.get("new_sha256") == _sha(""))
    return {**{key: row.get(key) for key in ("new_sha256", "old_sha256", "ts", "focus", "mode", "reason", "delta",
                                             "old_chars", "new_chars", "reverted_from")}, "text_stored": stored}


def _row_line(row: Mapping[str, Any], marks: list[str]) -> str:
    delta = row.get("delta") if isinstance(row.get("delta"), Mapping) else {}
    parts = [str(row.get("new_sha256") or "(no revision recorded)"), str(row.get("ts") or "time unknown"),
             "focus " + focus_text(row.get("focus")), str(row.get("mode") or "mode unknown"),
             "reason: " + (str(row["reason"]) if row.get("reason") else "not recorded")]
    if isinstance(delta.get("removed_chars"), int) and isinstance(delta.get("added_chars"), int):
        parts.append(f"{delta['removed_chars']} chars removed, {delta['added_chars']} added")
    elif isinstance(delta.get("change_chars"), int):
        parts.append(f"net change {delta['change_chars']:+d} chars")
    elif isinstance(row.get("old_chars"), int) and isinstance(row.get("new_chars"), int):
        parts.append(f"{row['old_chars']} → {row['new_chars']} chars")
    sections, headings = delta.get("removed_sections"), delta.get("removed_headings")
    if isinstance(sections, list) and sections:
        parts.append("sections removed: " + "; ".join(
            f"{item.get('heading')} ({item.get('chars')} chars)" for item in sections if isinstance(item, Mapping)))
    elif isinstance(headings, list) and headings:
        parts.append("headings removed: " + "; ".join(str(item) for item in headings))
    added = delta.get("added_headings")  # recorded at write time: no earlier version is parsed again here
    if isinstance(added, list) and added:
        parts.append("headings added: " + "; ".join(str(item) for item in added))
    if row.get("reverted_from"):
        parts.append(f"reverted_from {row['reverted_from']}")
    if not row["text_stored"]:
        marks.append("text not stored")
    return "- " + " · ".join(parts + marks)


def render_history(note: Any) -> str:
    """This note's versions, newest first, one line each, with the current one marked."""
    history = NoteHistory(note.address)
    rows = [_listed(row) for _offset, row in history]
    address = note.address
    lines = [f"History of {address.topic!r} ({address.scope}): {len(rows)} recorded version(s), newest first. "
             f"Current revision: {note.revision}."]
    lines += [f"Gap: {gap}." for gap in history.gaps()]
    current_seen = False
    for index, row in enumerate(rows):
        marks = []
        revision = row.get("new_sha256")
        if revision == note.revision:
            marks.append("same text as the current version" if current_seen else "current")
            current_seen = True
        newer = rows[index - 1] if index else None
        if newer is not None and revision != newer.get("old_sha256"):
            marks.append("not the base of the next write (it did not land, or the file changed outside)")
        elif newer is None and revision != note.revision:
            marks.append("not the current text (it did not land, or the file changed outside)")
        lines.append(_row_line(row, marks))
    if not rows:
        lines.append("(no rows: this note was written before the journal recorded it, or outside the writer)")
    arguments = f"topic={address.topic!r}, scope={address.scope!r}"
    lines.append(f"Read one: knowledge_read({arguments}, version=<revision>). Put one back: knowledge_write({arguments}, "
                 f"mode='revert', version=<revision>, expected_revision={note.revision!r}).")
    return "\n".join(lines) + "\n"


def _lines(raw: bytes, start: int, end: int, sign: str) -> tuple[int, list[str]]:
    """How many whole lines ``raw[start:end]`` holds, and each after ``sign``; a last line without a line end says so."""
    text = raw[start:end].decode("utf-8")
    pieces = text.split("\n") if text else []
    if text.endswith("\n"):
        pieces.pop()
    return len(pieces), [f"{sign} {piece}" for piece in pieces] + ([] if not text or text.endswith("\n")
                                                                   else ["\\ (no line end)"])


def render_changes(note: Any, held: Mapping[str, Any]) -> str:
    """Every change between the version a stale write holds and the current text, in whole lines.

    After the rows written since that version, each block is the whole lines it touches: ``- `` every
    line of the held version, then ``+ `` every line of the current one, with where each side's lines
    begin and the heading the block sits under in the current note; then the outline and the full
    read. Putting each block's ``+`` lines in place of its ``-`` lines in the held version gives the
    current note (``text_change.line_blocks``). Nothing here is cut: a reply that cannot carry all of
    it sends the complete current note instead (``tools/knowledge.py``)."""
    address, old_raw, new_raw = note.address, held["text"].encode("utf-8"), note.raw
    headings = note.source.headings if note.source else ()
    blocks = text_change.line_blocks(held["text"], note.text, text_change.exact_change(held["text"], note.text))
    body = []
    for old_start, old_end, new_start, new_end, removed, added in blocks:
        under = next((h for h in reversed(headings) if h.span.start_byte <= new_start), None)
        where = f"under {'#' * under.level} {under.title}" if under else "before the first heading"
        (old_count, old_lines), (new_count, new_lines) = (_lines(old_raw, old_start, old_end, "-"),
                                                          _lines(new_raw, new_start, new_end, "+"))
        old_line, new_line = old_raw.count(b"\n", 0, old_start) + 1, new_raw.count(b"\n", 0, new_start) + 1
        body.append(f"@ yours: {old_count} line(s) from line {old_line} → current: {new_count} line(s) from line "
                    f"{new_line}, {where}: {removed} chars removed, {added} added")
        body += old_lines + new_lines
    facts = held["facts"]
    head = [f"The revision you sent is an earlier version of this note, written {facts['ts']} by focus "
            f"{focus_text(facts['focus'])} ({facts['mode']}; reason: {reason_text(facts['reason'])}). "
            f"Current revision: {note.revision}.", "Written since that version, oldest first:"]
    head += [f"- {item['ts']} · focus {focus_text(item['focus'])} · {item['mode']} · reason: {reason_text(item['reason'])}"
             f" → {item['revision']}" for item in held["since"]] or ["- (no row: the file changed outside the writer)"]
    head.append(f"What changed from your version to the current text, in whole lines ({len(blocks)} block(s)): "
                "in your version, put each block's + lines in place of its - lines and you have the current note.")
    tail = ["Outline of the current note:", render_outline(note).rstrip("\n"),
            f"The complete current text is knowledge_read(topic={address.topic!r}, scope={address.scope!r})."]
    return "\n".join(head + body + tail) + "\n"


def reason_text(value: Any) -> str:
    """A row's reason as listings show it: ``not recorded`` where none was given."""
    return "not recorded" if value in (None, "", UNKNOWN) else str(value)
