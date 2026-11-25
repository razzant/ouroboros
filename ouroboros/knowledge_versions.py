"""Journal history for memory texts and the recorded claims of knowledge notes.

A note's journal is ``address.shelf.parent / "knowledge_history.jsonl"``, the append-only record the
one writer (``knowledge.write_knowledge_note``) keeps for a shelf; a row belongs to a note by its
``topic`` and the scope its ``address`` names. The same helpers read a plain memory text's journal
(``memory_texts.TextSource``: identity, the world map, a project's workpad): the source names its
``journal``, its rows' ``topic`` (none for a journal of one text, whose every row is its) and the
``noun`` the answers use. Lines are pre-filtered by the topic as the writer encodes it in JSON before
anything is parsed and are read newest first, in blocks from the journal's end, so a large journal is
never parsed whole; a line naming the text that is not JSON is counted as a gap, never fatal. A version is named by its revision (the sha256 of its exact bytes);
its text is read through ``knowledge_rows`` (stored whole, or rebuilt from the changes rows carry)
and returned only after it hashes to that revision. Readers write nothing. Claim changes commit
after the note and its index land, then project the committed names under the shelf lock.

The held-version store keeps a shown revision's full Markdown bytes when the journal cannot rebuild it, and a
25-byte journal proof where a fallback check succeeds. Writer-published revisions use the proof in their claim record
or, once the note has a folder here, the writer's proof, which replaces the folder's other proofs. A failed journal
check is remembered beside its body and rechecked when new history names that note. Capture replaces a body with a
proof, and the next write removes the body it replaced, after hash-verified reconstruction without the live file. Only
the journal can retire a body, so changing claims or ending a task cannot break an issued reader.

The outline is the one renderer the read, the receipt and the refusal share. It prints each heading
line as the note spells it; its character ranges are the ``start_char``/``end_char`` of
``knowledge_read``; a section is the parser's (``MarkdownHeading.section``), nested sections
included, and a heading the parser gives no section of its own (a setext heading) says so.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from collections import Counter
from dataclasses import dataclass
from heapq import merge
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping

from ouroboros import knowledge_rows, text_change, utils
from ouroboros.knowledge_edits import yaml_form
from ouroboros.utils import jsonl_window

JOURNAL_NAME = "knowledge_history.jsonl"
INDEX_SOURCE_TYPE = "knowledge_index_source"
UNKNOWN = "unknown"  # knowledge.UNKNOWN_STAMP: a fact the row does not carry
_BLOCK = 8 * 1024 * 1024  # one backward read of the journal; a longer line grows across reads
WRITE_TAIL_BYTES = 8 * 1024 * 1024  # a write keeps both texts whole on a miss
_FIRST_LINE_CHARS = 120


class VersionUnavailable(ValueError):
    """No verified text for a revision: the message says why (not found, no text, hash mismatch)."""


def journal_path(address: Any) -> Path:
    """A note's shelf journal, or the journal a plain source names."""
    own = getattr(address, "journal", None)
    return own if own is not None else address.shelf.parent / JOURNAL_NAME


def noun(address: Any) -> str:
    """What the answers call the text: ``note``, or the plain source's own noun."""
    return getattr(address, "noun", "note")


def _needle(address: Any) -> bytes:
    """The topic as a JSON string, spelled as every writer of this journal writes it (``ensure_ascii=False``).

    One spelling: a second one that no row carries would rescan the whole block for every line found. A
    source without a topic (a journal of one text) takes every row, and every row is an object: ``{``."""
    topic = getattr(address, "topic", None)
    return json.dumps(topic, ensure_ascii=False).encode("utf-8") if isinstance(topic, str) else b"{"


def is_claim_commit(row: Any) -> bool:
    """A ``claims_published`` row: the commit of a note's recorded names beside its write, never a write itself.
    The readers of the journal's writes skip these rows and no others."""
    return isinstance(row, dict) and row.get("publication") == _CLAIMS_PUBLICATION


def _belongs(row: Any, address: Any) -> bool:
    """A version row of this text: its topic (none for a journal of one text), a version field, a capture (or an
    older unstamped row) of its scope."""
    if not isinstance(row, dict) or row.get("topic") != getattr(address, "topic", None):
        return False
    if ("new_content" not in row and "new_sha256" not in row) or row.get("publication", "source_capture") != "source_capture":
        return False
    where, scope = row.get("address"), getattr(address, "scope", None)
    return not isinstance(where, dict) or where.get("scope", scope) == scope


def _matching_lines(region: bytes, base: int, needle: bytes) -> Iterator[tuple[int, bytes]]:
    """``(offset, line)`` of each line of ``region`` (whole lines from ``base``) holding ``needle``, newest first.

    Each search starts below the line last found, so the region is scanned once."""
    end = len(region)
    while end > 0:
        hit = region.rfind(needle, 0, end)
        if hit < 0:
            return
        start = region.rfind(b"\n", 0, hit) + 1
        stop = region.find(b"\n", hit)
        yield base + start, region[start:len(region) if stop < 0 else stop + 1]
        end = start


def _journal_lines(journal: Path, needle: bytes, *, after: int = 0) -> Iterator[tuple[int, bytes]]:
    """Lines holding ``needle`` after a record boundary, newest first in blocks."""
    with journal.open("rb") as handle:
        end, carry = os.fstat(handle.fileno()).st_size, b""
        if after:
            handle.seek(after - 1)
            if handle.read(1) != b"\n":
                raise ValueError("incomplete journal boundary")
        while end > after:
            start = max(after, end - _BLOCK)
            handle.seek(start)
            data = handle.read(end - start) + carry
            cut = data.find(b"\n") + 1 if start > after else 0
            if start > after and not cut:  # one line longer than everything read so far
                carry, end = data, start
                continue
            yield from _matching_lines(data[cut:], start + cut, needle)
            carry, end = data[:cut], start


def _parse(line: bytes) -> Any:
    try:
        return json.loads(line)
    except (ValueError, RecursionError):
        return None


def _journal_position(handle: Any, offset: int) -> dict:
    """Check the last 256 bytes at a complete record boundary, without replaying its prefix."""
    start = max(0, offset - 256)
    handle.seek(start)
    ending = handle.read(offset - start)
    if len(ending) != offset - start or (offset and not ending.endswith(b"\n")):
        raise ValueError("incomplete journal ending")
    return {"offset": offset, "check": hashlib.sha256(ending).hexdigest()}


def knowledge_changes(root: Any, boundary: Any, gaps: set) -> tuple[list, Any]:
    """Other foci's changed pages since accepted byte positions, in first-write order per journal.

    The first observation baselines existing journals; an accepted empty map sees new ones from zero.
    Only a fixed ending check precedes unread bytes; unchanged journals are never opened. Invalid
    complete rows are disclosed and skipped; missing/shortened journals drop/re-baseline their cursor.
    Baselines succeed per journal; an unfinished tail waits while its complete prefix advances.
    Other unreadable positions stay put. This reader writes nothing.
    """
    root, previous = Path(root), boundary or {}
    current, events = dict(previous), []
    journals = {"global": root / "memory" / JOURNAL_NAME}
    try:
        journals.update({f"project:{path.parent.name}": path for path in root.glob(f"projects/*/{JOURNAL_NAME}")})
    except OSError as exc:
        gaps.add(f"knowledge journals unreadable: {type(exc).__name__}; accepted positions retained")
        return [], boundary
    for scope in previous.keys() - journals.keys():
        journals[scope] = root / "projects" / scope.removeprefix("project:") / JOURNAL_NAME
    for scope, journal in sorted(journals.items()):
        prior = previous.get(scope, {"offset": 0, "check": hashlib.sha256(b"").hexdigest()})
        source, reset_gap = journal.relative_to(root), ""
        try:
            try:
                end = journal.stat().st_size
            except FileNotFoundError:
                if scope in current:
                    del current[scope]
                    gaps.add(f"knowledge journal missing ({scope}): {source}; cursor dropped")
                continue
            if end == prior["offset"]:
                continue
            if end < prior["offset"]:
                del current[scope]
                reset_gap = f"knowledge journal shortened ({scope}): {source}, {prior['offset']} → {end} bytes"
            with journal.open("rb") as handle:
                if boundary is None or reset_gap:
                    current[scope] = _journal_position(handle, end)
                    if reset_gap:
                        gaps.add(f"{reset_gap}; re-baselined at current end")
                    continue
                if _journal_position(handle, prior["offset"]) != prior:
                    raise ValueError("accepted journal ending changed")
                changes = {}
                while handle.tell() < end:
                    start = handle.tell()
                    line = handle.readline(end - handle.tell())
                    if not line.endswith(b"\n"):
                        gaps.add(f"knowledge changes unreadable ({scope}): {source}, bytes {start}..{handle.tell()} "
                                 f"(end exclusive); incomplete row pending; position stops at {start}")
                        end = start
                        break
                    row = _parse(line)
                    if not isinstance(row, dict):
                        gaps.add(f"knowledge journal gap ({scope}): {source}, bytes {start}..{handle.tell()} "
                                 "(end exclusive); invalid row skipped")
                        continue
                    topic, focus = row.get("topic"), row.get("focus")
                    if not _belongs(row, SimpleNamespace(topic=topic, scope=scope)):
                        continue
                    if isinstance(focus, Mapping) and focus.get("role") == "consciousness":
                        continue
                    facts = row_facts(row)
                    page = changes.setdefault(topic, {"count": 0, "first": facts["ts"], "writers": {}})
                    page["count"] += 1
                    page["writers"][focus_text(focus)] = None
                    page["latest"] = facts
                current[scope] = _journal_position(handle, end)
                for topic, page in changes.items():
                    latest, count = page["latest"], page["count"]
                    events.append(("knowledge_change", None,
                        f"- knowledge {scope}/{topic}: {count} write{'s' if count != 1 else ''}; "
                        f"by {', '.join(page['writers'])}; first {page['first']}; last {latest['ts']}; "
                        f"knowledge_read(topic={topic!r}, scope={scope!r}, version={latest['revision']!r})"))
        except (OSError, ValueError) as exc:
            where = reset_gap or f"knowledge changes unreadable ({scope}): {source}"
            action = "accepted position retained" if scope in current else "journal skipped"
            gaps.add(f"{where}: {exc}; {action}")
    return events, current


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
    for offset, line in _matching_lines(tail, size - len(tail), _needle(address)):
        row = _parse(line)
        if _belongs(row, address):
            yield offset, row


def write_tail(journal: Path, address: Any, revision: str | None) -> tuple[Any, Any]:
    """The newest authored row and the newest verified base for revision, from one tail read under the shelf lock."""
    rows = tail_rows(journal, address, WRITE_TAIL_BYTES)
    newest = next(rows, None)
    if newest is None or revision is None:
        return newest, None
    return newest, knowledge_rows.verified_base(journal, chain([newest], rows), revision)


class NoteHistory:
    """This text's rows over the whole journal, newest first, with the gaps met on the way."""

    def __init__(self, address: Any, extra: bytes = b""):
        self.address, self.extra = address, extra
        self.malformed, self.first_malformed, self.error = 0, None, ""

    def __iter__(self) -> Iterator[tuple[int, dict]]:
        try:
            for offset, line in _journal_lines(journal_path(self.address), _needle(self.address)):
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
            naming = f" naming this {noun(self.address)}" if isinstance(getattr(self.address, "topic", None), str) else ""
            out.append(f"{self.malformed} journal line(s){naming} are not JSON and were skipped "
                       f"(the first at byte offset {self.first_malformed})")
        if self.error:
            out.append(f"the journal could not be read to its beginning ({self.error}); older rows are not shown")
        return out


_CLAIMS_PUBLICATION = "claims_published"
_JOURNAL_STAMP = struct.Struct(">QQQ")
_HELD_PROOF = b"\xff"  # a flag outside the UTF-8 a Markdown body can contain here


def claim_names(value: Any) -> list[str]:
    return [name for name in value if isinstance(name, str)] if isinstance(value, list) else []


def claim_record_path(shelf: Path, topic: str) -> Path:
    # Topic paths may share prefixes with one another's filenames.
    key = hashlib.sha256(topic.encode("utf-8")).hexdigest()
    return shelf.parent / "knowledge_note_records" / f"{key}.json"


def _held_version_path(address: Any, revision: str) -> Path:
    topic_key = claim_record_path(address.shelf, address.topic).stem
    return address.shelf.parent / "knowledge_held_versions" / topic_key / f"{revision}.md"


def _journal_stamp(address: Any) -> bytes:
    try:
        with journal_path(address).open("rb", buffering=0) as journal:
            info = os.fstat(journal.fileno())
            journal.read(1)  # a stat alone cannot establish that a reader can open the journal
    except OSError:
        return b""
    changed = info.st_mtime_ns if os.name == "nt" else info.st_ctime_ns
    identity = hashlib.blake2b(f"{info.st_dev}:{info.st_ino}".encode(), digest_size=8).digest()
    return _JOURNAL_STAMP.pack(int.from_bytes(identity, "big"), info.st_size, changed)


def _covers_journal(stamped: bytes, observed: bytes) -> bool:
    """Witness the journal at the recorded byte boundary across ordinary appends."""
    if len(stamped) != _JOURNAL_STAMP.size or not observed:
        return False
    identity, end, _ = _JOURNAL_STAMP.unpack(stamped)
    current_identity, current_end, _ = _JOURNAL_STAMP.unpack(observed)
    return stamped == observed or (identity == current_identity and end < current_end)


def _check_still_current(address: Any, checked: bytes | None, stamp: bytes) -> bool:
    if not checked or not _covers_journal(checked, stamp):
        return False
    if checked == stamp:
        return True
    _, offset, _ = _JOURNAL_STAMP.unpack(checked)
    try:
        return not any(_belongs(_parse(line), address) for _, line in
                       _journal_lines(journal_path(address), _needle(address), after=offset))
    except (OSError, ValueError):
        return False


def _held_file(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _record_proof(path: Path, stamp: bytes) -> None:
    if stamp:
        try:
            utils.write_bytes_atomic(path, _HELD_PROOF + stamp, fsync=True)
            path.with_suffix(".checked").unlink(missing_ok=True)
        except OSError:
            pass  # the journal backs the reader even when its derived proof cannot land


def _journal_rebuilds(address: Any, revision: str) -> bool:
    try:
        find_version(address, revision, live=None, journal_only=True)
    except VersionUnavailable:
        return False
    return True


def _writer_proved(address: Any, revision: str, stamp: bytes) -> bool:
    record = utils.json_snapshot(claim_record_path(address.shelf, address.topic)) or {}
    if record.get("topic") != address.topic or record.get("revision") != revision:
        return False
    try:
        proof = bytes.fromhex(record.get("journal_proof", ""))
    except (ValueError, TypeError):
        return False
    return _covers_journal(proof, stamp)


def retain_held_version(note: Any, revision: str) -> None:
    """Prove the reader from history or put the exact source in custody before publishing it."""
    address = note.address
    path = _held_version_path(address, revision)
    stamp, captured = _journal_stamp(address), _held_file(path)
    if captured and captured.startswith(_HELD_PROOF) and _covers_journal(captured[1:], stamp):
        return
    if captured is None and _writer_proved(address, revision, stamp):
        return
    checked = path.with_suffix(".checked")
    previous = _held_file(checked)
    if stamp and not _check_still_current(address, previous, stamp) and _journal_rebuilds(address, revision):
        _record_proof(path, stamp)
        return
    if captured != note.raw:
        utils.write_bytes_atomic(path, note.raw, fsync=True)
    if stamp and previous != stamp:
        try:
            utils.write_bytes_atomic(checked, stamp)
        except OSError:
            pass  # the exact body is already in custody


def _retained_held_version(address: Any, revision: str) -> str | None:
    if getattr(address, "scope", None) != "global":
        return None
    raw = _held_file(_held_version_path(address, revision))
    if raw is not None and hashlib.sha256(raw).hexdigest() == revision:
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            pass
    return None


def held_write(address: Any, history: Mapping[str, Any]) -> None:
    """After a write's row lands, where this note was ever held (it has a folder here): the writer's proof of the new
    revision stands for it, so a note no claim record proves (a head a held head links) needs no copy of it. Every
    other proof leaves, and the replaced revision's body once this row rebuilds it: what stays is an only copy."""
    path, old = _held_version_path(address, history["new_sha256"]), history.get("old_sha256")
    stamp = _journal_stamp(address) if path.parent.is_dir() else b""
    if not stamp:
        return
    _record_proof(path, stamp)
    try:
        for other in path.parent.glob("*.md"):
            raw = _held_file(other)
            if other != path and raw is not None and (raw.startswith(_HELD_PROOF) or (
                    other.stem == old and _journal_rebuilds(address, old))):
                other.unlink()
                other.with_suffix(".checked").unlink(missing_ok=True)
    except OSError:
        pass  # a file left behind is a copy the journal backs: a later write removes a proof, no write a body


def inventory_claim_names(shelf: Path, rows: Iterator[dict[str, Any]]) -> Iterator[tuple[str, list[str]]]:
    """Intersect each inventory row with the record at its shelf address."""
    for row in rows:
        topic = row["topic"]
        record = utils.json_snapshot(claim_record_path(shelf, topic)) or {}
        written = set(row["claims"])
        yield topic, [name for name in claim_names(record.get("active_claims")) if name in written]


def _claim_publication(address: Any) -> dict | None:
    """The latest committed publication at this address, including before the projection landed."""
    try:
        for _offset, line in _journal_lines(journal_path(address), _needle(address)):
            if b'"claims_published"' not in line:
                continue
            row = _parse(line)
            if (not isinstance(row, dict) or row.get("publication") != _CLAIMS_PUBLICATION
                    or row.get("topic") != address.topic or not isinstance(row.get("new_sha256"), str)):
                continue
            where = row.get("address")
            if isinstance(where, dict) and where.get("scope", address.scope) != address.scope:
                continue
            return row
    except FileNotFoundError:
        pass
    return None


def _project_claim_record(path: Path, publication: dict | None) -> None:
    record = utils.json_snapshot(path)
    if not publication and record is None:
        return
    updated = {**(record or {}), "active_claims": claim_names((publication or {}).get("active_claims"))}
    if publication:
        updated.update(topic=publication["topic"], revision=publication["new_sha256"])
        if proof := publication.get("journal_proof"):
            updated["journal_proof"] = proof
        else:
            updated.pop("journal_proof", None)
    if updated != record:
        utils.atomic_write_json(path, updated)


def rebuild_claim_record(address: Any) -> None:
    """Rebuild one note's claims from completed writes. The caller holds the shelf lock."""
    _project_claim_record(claim_record_path(address.shelf, address.topic), _claim_publication(address))


@dataclass(frozen=True)
class ClaimWrite:
    """Proposed claim facts for a single revision-checked write under the shelf lock."""
    path: Path | None
    recorded: list[str]
    active: list[str]
    written: list[str]
    ignored: list[str]
    removed: bool
    report: bool
    publication: dict | None

    @property
    def changed(self) -> bool:
        return self.active != self.recorded

    def needs_publication(self, revision: str) -> bool:
        return self.changed or bool(self.active and revision != (self.publication or {}).get("new_sha256"))

    def publish(self, address: Any, history: Mapping[str, Any], revision: str) -> None:
        """Commit the names and their latest published revision after the source and index landed."""
        if self.path is None:
            return
        publication = self.publication
        if self.needs_publication(revision):
            row = {key: history[key] for key in (
                "task_id", "mode", "writer", "focus", "old_sha256", "reason",
            ) if key in history}
            row.update(ts=history.get("ts") or utils.utc_now_iso(), topic=address.topic, address=address.as_dict(),
                       new_sha256=revision, publication=_CLAIMS_PUBLICATION, active_claims=self.active)
            if self.active and history.get("new_sha256") == revision and (stamp := _journal_stamp(address)):
                row["journal_proof"] = stamp.hex()
            if not utils.append_jsonl(journal_path(address), row,
                                      ensure_record_boundary=True, require_lock=True):
                raise OSError("Claims history publication failed")
            publication = row
        _project_claim_record(self.path, publication)


def prepare_claim_write(address: Any, current: Any, updated: Any, *, explicit: bool) -> ClaimWrite | None:
    before = current.metadata if current is not None else {}
    after = updated.metadata
    path = claim_record_path(address.shelf, address.topic) if address.scope == "global" else None
    if "claims" not in before and "claims" not in after and (path is None or not path.is_file()):
        return None
    old, new = claim_names(before.get("claims")), claim_names(after.get("claims"))
    # This is a writer/recovery read. Starts read only the derived record.
    publication = _claim_publication(address) if path is not None else None
    recorded = claim_names((publication or {}).get("active_claims"))
    if path is None:
        active = []
    elif explicit:
        active = list(dict.fromkeys(new))
    else:
        keep = set(old) & set(recorded)
        active = list(dict.fromkeys(name for name in new if name in keep))
    value = after.get("claims")
    ignored = ([repr(item) for item in value if not isinstance(item, str)] if isinstance(value, list)
               else [repr(value)] if "claims" in after else [])
    removed = bool((set(old) - set(new)) or (set(recorded) - set(active)))
    fields = ({"claims": data["claims"]} if "claims" in data else {} for data in (before, after))
    before_claims, after_claims = map(yaml_form, fields)
    report = explicit or before_claims != after_claims or recorded != active
    return ClaimWrite(path, recorded, active, new, ignored, removed, report, publication)


_READ = object()  # find_version reads the live text itself


def find_version(address: Any, version: Any, live: Any = _READ, *, journal_only: bool = False) -> tuple[str, dict]:
    """The exact text of one version of this text and the row that wrote it, newest rows first.

    Every row that names the version, as the text it published (``new_sha256``) or replaced
    (``old_sha256``), is read through ``knowledge_rows.Texts``: stored whole, rebuilt through its
    base's changes, or, when that path is broken, from the other end of its segment (a whole text or
    the live note); the text is returned only when it hashes to the version (an empty text is the
    version sha256(b"")). The row is the one that wrote the version, or ``{}`` when none did (an
    outside edit, or a write before history). ``VersionUnavailable`` says why there is no text and
    names the gaps among the lines that name this note. ``live`` is the current text a writer already
    captured under its lock (the journal and the text it reads are then one state); by default the file
    is read here. A custody check passes ``live=None, journal_only=True`` to require independent
    reconstruction from the journal and skip the second pass for diagnostics on a miss."""
    if not isinstance(version, str) or len(version) != 64 or version.strip("0123456789abcdef"):
        raise VersionUnavailable("a version is a revision: the 64 lowercase hex characters of its sha256")
    if live is _READ:
        try:
            live = address.path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            live = None
    texts, why, held = knowledge_rows.Texts(journal_path(address), live), "", None
    history = NoteHistory(address, version.encode("ascii"))
    for offset, row in history:
        own = row.get("new_sha256") == version
        if not own and (held is not None or row.get("old_sha256") != version):
            continue
        texts.rows.setdefault(offset, row)
        try:
            text = texts.text(offset, "new" if own else "old")
        except knowledge_rows.TextUnavailable as exc:
            text, why = None, why or str(exc)
        if own and (text is not None or held is not None):
            return (held if text is None else text), row
        if text is not None:
            held = text  # a later row's old side: keep looking for the row that wrote it
    if held is not None:
        return held, {}
    if journal_only:
        raise VersionUnavailable(why or f"no version {version} of this {noun(address)} in its history")
    if (captured := _retained_held_version(address, version)) is not None:
        return captured, {}
    # A torn line may have lost the very version asked for: on a miss, count every line naming this note.
    unfiltered = NoteHistory(address)
    for _row in unfiltered:
        pass
    gaps = "; ".join(unfiltered.gaps())
    raise VersionUnavailable((why or f"no version {version} of this {noun(address)} in its history")
                             + (f" ({gaps})" if gaps else ""))


def find_note_version(address: Any, requested: Any, version: Any, live: Any = _READ) -> tuple[Any, str, dict]:
    """Return the journal address that verifies the revision, including legacy writes under a case alias."""
    try:
        text, row = find_version(address, version, live)
    except VersionUnavailable:
        if requested == address:
            raise
        address = requested
        text, row = find_version(address, version, live)
    return address, text, row


def row_facts(row: Mapping[str, Any]) -> dict[str, Any]:
    """Who wrote a version and how: ``{revision, ts, focus, mode, reason}``, ``unknown`` where absent."""
    return {"revision": row.get("new_sha256") or UNKNOWN, "ts": row.get("ts") or UNKNOWN,
            "focus": row.get("focus") or UNKNOWN, "mode": row.get("mode") or UNKNOWN,
            "reason": row.get("reason") or UNKNOWN}


def written_text(facts: Mapping[str, Any], outside: str = "an outside edit, or a write before history") -> str:
    """Who wrote a version, as a read, a receipt or a refusal says it: ``row_facts`` of the row that wrote it, or
    that no row did (``outside`` names what can change the text without one; it never names a cause as fact)."""
    if facts.get("ts") in (None, UNKNOWN):
        return f"no history row wrote it ({outside})"
    return (f"written {facts['ts']} by focus {focus_text(facts.get('focus'))} "
            f"({facts.get('mode') or UNKNOWN}; reason: {reason_text(facts.get('reason'))})")


def held_version(journal: Path, address: Any, revision: str, tail_bytes: int,
                 live: str | None = None) -> dict[str, Any] | None:
    """The version a stale write holds, when this text's rows in the writer's tail have it.

    ``{"facts", "text", "since"}``: the facts of the row that wrote it, its text read through
    ``knowledge_rows`` (``live``, the current text, closes the newest segment) and the facts of the rows
    written after it, oldest first. A version only a row's old side holds (a first captured pre-image,
    an outside edit) is read from that side, verified; its author is unknown unless an older row in the
    tail wrote it, and the writes since begin with the oldest row that replaced it. ``None``: the tail
    holds no verified text of it."""
    since, texts, held = [], knowledge_rows.Texts(journal, live), None
    for offset, row in tail_rows(journal, address, tail_bytes):
        texts.rows.setdefault(offset, row)
        if row.get("new_sha256") == revision:
            try:
                text = texts.text(offset, "new")
            except knowledge_rows.TextUnavailable:
                text = held[0] if held else None
            if text is None:
                return None
            return {"facts": row_facts(row), "text": text, "since": [row_facts(item) for item in reversed(since)]}
        since.append(row)
        if row.get("old_sha256") == revision:  # the version this row replaced, held only on its old side so far
            text = held[0] if held else None
            if text is None:
                try:
                    text = texts.text(offset, "old")
                except knowledge_rows.TextUnavailable:
                    text = None
            if text is not None:
                held = (text, list(since))
    if held is None:
        return None
    return {"facts": {**row_facts({}), "revision": revision}, "text": held[0],
            "since": [row_facts(item) for item in reversed(held[1])]}


def focus_text(focus: Any) -> str:
    """A row's focus as one phrase: its role, then what else it carries."""
    if not isinstance(focus, Mapping):
        return str(focus or UNKNOWN)
    carried = [f"{key}={focus[key]}" for key in ("task_id", "root_task_id", "room_id", "chat_id", "place_id",
                                                  "conversation_key")
               if focus.get(key) not in (None, "") and not (key == "root_task_id" and focus[key] == focus.get("task_id"))]
    return " ".join([str(focus.get("role") or UNKNOWN), *carried])


def _own_section(heading: Any) -> Any:
    """The parser's section when it begins at this heading; a setext heading has none of its own.

    An ATX heading's span is one line, a setext heading's two or more: a setext heading that opens the
    body starts the parser's implicit preamble section, which is not its own (it ends at the first ATX
    heading of any level)."""
    own = heading.section.start_byte == heading.span.start_byte and heading.span.start_line == heading.span.end_line
    return heading.section if own else None


def _first_line(text: str) -> str:
    """A text's first line without its LF or CRLF end: a setext heading's title may span lines."""
    return text.split("\n", 1)[0].removesuffix("\r")


def heading_line(raw: bytes, heading: Any) -> str:
    """The heading's first line as the note spells it (spacing, closing hashes, indentation), without its line end."""
    return _first_line(raw[heading.span.start_byte:heading.span.end_byte].decode("utf-8"))


def _char_offsets(raw: bytes, positions: set[int]) -> dict[int, int]:
    """Character offsets of byte positions on character boundaries, decoded once."""
    out, at, chars = {}, 0, 0
    for position in sorted(positions):
        chars += len(raw[at:position].decode("utf-8"))
        out[position], at = chars, position
    return out


def render_outline(note: Any, label: str = "", *, version: str | None = None) -> str:
    """Summary, revision, state, size and every heading with its line and section range.

    The ranges are of the text outlined: the note, or the earlier version a ``label`` marks, which
    is read with its version."""
    summary = note.summary.strip()
    lines = [f"Outline of {note.address.topic!r} ({note.address.scope}){label}",
             "Summary: " + ("\n  ".join(summary.splitlines()) if summary else "no summary"),
             f"Revision: {note.revision}", f"State: {note.state}", f"Total: {len(note.text)} characters"]
    if note.source is None:
        pin = f" with version={version}" if version is not None else ""
        lines.append(f"Headings: unknown (the metadata does not parse; the complete note is the default read{pin})")
        return "\n".join(lines) + "\n"
    headings = note.source.headings
    if not headings:
        lines.append("Headings: none")
        return "\n".join(lines) + "\n"
    spans = [_own_section(heading) or heading.span for heading in headings]
    chars = _char_offsets(note.raw, {edge for span in spans for edge in (span.start_byte, span.end_byte)})
    outlined = (f"this version itself, read with the default view and version={version}" if version is not None
                else "the note itself, read with the default view")
    lines.append("Headings (each heading line as written, its line, section range as start_char..end_char, size). "
                 f"The ranges are of {outlined}. "
                 "A section runs to the next heading of its level or higher, so a parent's range contains its children's:")
    for heading, span in zip(headings, spans):
        start, end = chars[span.start_byte], chars[span.end_byte]
        own = ("" if _own_section(heading) else
               f"; no section of its own (a level-{heading.level} setext heading): the range is its heading")
        lines.append(f"{heading_line(note.raw, heading)} — line {heading.span.start_line}, "
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


def body_chars(note: Any) -> int:
    """The same stored body the context loader carries, excluding parsed frontmatter."""
    if note is None:
        return 0
    return len(note.source.text_at(note.source.body_span) if note.source else note.text)


def note_delta(current: Any, updated: Any, change: list) -> dict[str, Any]:
    """The text and parsed-structure differences recorded with a note's publication."""
    old_text = current.text if current else ""
    before = Counter((h.level, h.title) for h in current.source.headings) if current and current.source else Counter()
    after = Counter((h.level, h.title) for h in updated.source.headings) if updated.source else Counter()
    known = (current is None or current.source) and updated.source
    removed_headings, added_headings = (
        sorted("#" * level + " " + title for (level, title) in (left - right).elements()) if known else None
        for left, right in ((before, after), (after, before)))
    # Body bytes and the resident summary are compared as published. An unparseable side leaves both unknown.
    old_body, old_summary = ((current.raw[current.source.body_span.start_byte:], current.summary)
                             if current and current.source else (b"", ""))
    removed_chars, added_chars = text_change.change_chars(change)
    return {"old_chars": len(old_text), "new_chars": len(updated.text),
            "old_body_chars": body_chars(current), "new_body_chars": body_chars(updated),
            "change_chars": len(updated.text) - len(old_text), "removed_chars": removed_chars,
            "added_chars": added_chars, "removed_headings": removed_headings, "added_headings": added_headings,
            "removed_sections": removed_sections(current, updated),
            "body_changed": old_body != updated.raw[updated.source.body_span.start_byte:] if known else None,
            "summary_changed": old_summary != updated.summary if known else None}


def _listed(row: Mapping[str, Any]) -> dict[str, Any]:
    """The facts a listing line shows, without the row's note texts (a long history is not held whole).

    A row has its text when it holds the whole ``new_content`` or a ``change`` against a ``base``, unless it is
    marked ``content_digested`` (the reader refuses such a row whatever it holds); no version is rebuilt to say so.
    The sizes a row recorded travel with it: an identity or map row written before rows carried a ``delta`` has only
    its ``old_len``/``new_len``, shown as they are (``delta_parts``), never turned into removed and added counts."""
    text = row.get("new_content")
    stored = not row.get("content_digested") and (
        (isinstance(text, str) and bool(text or not row.get("new_sha256")
                                         or row.get("new_sha256") == knowledge_rows.sha256_text("")))
        or (isinstance(row.get("change"), list) and isinstance(row.get("base"), Mapping)))
    return {**{key: row.get(key) for key in ("new_sha256", "old_sha256", "ts", "focus", "mode", "reason", "delta",
                                             "old_chars", "new_chars", "old_len", "new_len", "reverted_from",
                                             "old_not_text")},
            "text_stored": stored}


def delta_parts(row: Mapping[str, Any]) -> list[str]:
    """What a write changed, as its row (or a receipt's ``{"delta": ...}``) records it: characters, sections, headings.

    Recorded at write time: no earlier version is parsed again here."""
    delta = row.get("delta") if isinstance(row.get("delta"), Mapping) else {}
    parts = []
    if isinstance(delta.get("removed_chars"), int) and isinstance(delta.get("added_chars"), int):
        parts.append(f"{delta['removed_chars']} chars removed, {delta['added_chars']} added")
    elif isinstance(delta.get("change_chars"), int):
        parts.append(f"net change {delta['change_chars']:+d} chars")
    elif delta.get("old_not_text") and isinstance(delta.get("old_bytes"), int):  # the replaced bytes were no text
        parts.append(f"{delta['old_bytes']} bytes that were not UTF-8 text → {delta.get('new_chars')} chars")
    elif isinstance(row.get("old_chars"), int) and isinstance(row.get("new_chars"), int):
        parts.append(f"{row['old_chars']} → {row['new_chars']} chars")
    elif isinstance(row.get("old_len"), int) and isinstance(row.get("new_len"), int):  # a plain journal's row
        parts.append(f"{row['old_len']} → {row['new_len']} chars")
    # A stored heading is '#' * level + ' ' + title; a setext title that spans lines is named by its first line.
    sections, headings = delta.get("removed_sections"), delta.get("removed_headings")
    if isinstance(sections, list) and sections:
        parts.append("sections removed: " + "; ".join(f"{_first_line(str(item.get('heading')))} ({item.get('chars')} chars)"
                                                      for item in sections if isinstance(item, Mapping)))
    elif isinstance(headings, list) and headings:
        parts.append("headings removed: " + "; ".join(_first_line(str(item)) for item in headings))
    added = delta.get("added_headings")
    if isinstance(added, list) and added:
        parts.append("headings added: " + "; ".join(_first_line(str(item)) for item in added))
    return parts


def _replaced(row: Mapping[str, Any], current: str, revert: Callable[[str], str] | None,
              blocked: tuple[str, str], lifecycle: Callable[[str], str] | None = None) -> list[str]:
    """The version a write replaced, its actual undo target, and the call that puts it back over the current text
    (``revert``; ``None`` while no revert can run, ``blocked`` saying why and how the version is written instead). An
    empty ``old_sha256`` is a text that did not exist: no earlier version. A row that names no replaced version has no
    undo; never the line below's. A note's archive or restore row is undone by the opposite lifecycle call
    (``lifecycle(mode)``): the writer refuses a revert across archive states."""
    old = row.get("old_sha256")
    if old == "":
        return ["created it: no earlier version to put back"]
    if not isinstance(old, str) or not old:
        return ["replaced: not recorded, so no undo (the version it replaced is unknown)"]
    if row.get("old_not_text"):
        return [f"replaced {old}", "undo: none, those bytes were not UTF-8 text and history keeps only their sha256"]
    if old == current:
        undo = "undo: nothing to put back, that version is the current text"
    elif lifecycle is not None and row.get("mode") in ("archive", "restore"):
        undo = "undo: " + lifecycle("restore" if row["mode"] == "archive" else "archive")
    elif revert is None:
        undo = f"undo: {blocked[0]}, so read that version and write it with {blocked[1]}"
    else:
        undo = "undo: " + revert(old)
    return [f"replaced {old}", undo]


def _row_line(row: Mapping[str, Any], marks: list[str], replaced: list[str]) -> str:
    parts = [str(row.get("new_sha256") or "(no revision recorded)"), str(row.get("ts") or "time unknown"),
             "focus " + focus_text(row.get("focus")), str(row.get("mode") or "mode unknown"),
             "reason: " + (str(row["reason"]) if row.get("reason") else "not recorded"), *replaced]
    parts += delta_parts(row)
    if row.get("reverted_from"):
        parts.append(f"reverted_from {row['reverted_from']}")
    if not row["text_stored"]:
        marks.append("text not stored")
    return "- " + " · ".join(parts + marks)


def render_listing(source: Any, current: str, title: str, read: str, revert: Callable[[str], str] | None,
                   blocked: tuple[str, str] | None = None, lifecycle: Callable[[str], str] | None = None,
                   *, alias: Any = None) -> str:
    """A text's versions, newest first, one line each: who wrote it, when, in which mode, why, the version it replaced
    (its actual undo target, also where no line above wrote it: a first write, an outside change, a write that did not
    land) and the call that puts that version back, what it changed, with the current one marked; then the calls that
    read and put back a version. ``revert(version)`` is the exact call that puts ``version`` back over the current text
    (expected_revision, address and project included); ``None`` while no revert can run, ``blocked`` naming why and
    the write that brings a version back instead (default: the text does not exist, an overwrite creates it), so no
    line offers a call the writer refuses; ``lifecycle(mode)`` renders a note's archive or restore call, the undo of the
    opposite lifecycle row."""
    history = NoteHistory(source)
    legacy = NoteHistory(alias) if alias is not None else None
    entries = history if legacy is None else merge(history, legacy, key=lambda entry: entry[0], reverse=True)
    rows = [_listed(row) for _offset, row in entries]
    what = noun(source)
    blocked = blocked or (f"the {what} does not exist now", "mode=overwrite")
    lines = [f"{title}: {len(rows)} recorded version(s), newest first. Current revision: {current}."]
    lines += [f"Gap: {gap}." for gap in dict.fromkeys(history.gaps() + (legacy.gaps() if legacy else []))]
    current_seen = False
    for index, row in enumerate(rows):
        marks = []
        revision = row.get("new_sha256")
        if revision == current:
            marks.append("same text as the current version" if current_seen else "current")
            current_seen = True
        newer = rows[index - 1] if index else None
        if newer is not None and revision != newer.get("old_sha256"):
            marks.append("not the base of the next write (it did not land, or the file changed outside)")
        elif newer is None and revision != current:
            marks.append("not the current text (it did not land, or the file changed outside)")
        lines.append(_row_line(row, marks, _replaced(row, current, revert, blocked, lifecycle)))
    if not rows:
        lines.append(f"(no rows: this {what} was written before the journal recorded it, or outside the writer)")
    put_back = (revert("<revision>").replace("'<revision>'", "<revision>") if revert is not None else
                f"{blocked[0]}, so mode=revert cannot run: read a version and write it with {blocked[1]}")
    lifecycle_rows = lifecycle is not None and any(row.get("mode") in ("archive", "restore") for row in rows)
    lines.append(f"Read one: {read}. Put one back: {put_back}. Each line is the {what} after its write; its undo puts "
                 "back the version that write replaced, which is not always the line below" + (
                     "; an archive or restore line's undo is the opposite lifecycle mode, which changes the archive "
                     "field only" if lifecycle_rows else "") + ".")
    return "\n".join(lines) + "\n"


def render_history(note: Any, requested: Any = None) -> str:
    """This note's versions, newest first, one line each, with the current one marked (``render_listing``)."""
    address = note.address
    alias = requested if requested != address else None
    arguments = f"topic={address.topic!r}, scope={address.scope!r}"
    read = f"knowledge_read({arguments}, version=<revision>)"
    if alias is not None:
        arguments = f"topic={alias.topic!r}, scope={alias.scope!r}"
        read += f" or knowledge_read({arguments}, version=<revision>)"
    return render_listing(address, note.revision, f"History of {address.topic!r} ({address.scope})",
                          read,
                          lambda version: f"knowledge_write({arguments}, mode='revert', version={version!r}, "
                                          f"expected_revision={note.revision!r})",
                          lifecycle=lambda mode: f"knowledge_write({arguments}, mode={mode!r}, "
                                                 + ("reason=<why>, " if mode == "archive" else "")
                                                 + f"expected_revision={note.revision!r})", alias=alias)


def _lines(raw: bytes, start: int, end: int, sign: str) -> tuple[int, list[str]]:
    """How many whole lines ``raw[start:end]`` holds, and each after ``sign``; a last line without a line end says so."""
    text = raw[start:end].decode("utf-8")
    pieces = text.split("\n") if text else []
    if text.endswith("\n"):
        pieces.pop()
    return len(pieces), [f"{sign} {piece}" for piece in pieces] + ([] if not text or text.endswith("\n")
                                                                   else ["\\ (no line end)"])


def render_changes(note: Any, held: Mapping[str, Any]) -> str:
    """Every change between the version a stale write holds and the current note (``changes_since``), then its
    outline and the full read. Nothing here is cut: a reply that cannot carry all of it sends the complete current
    note instead (``tools/knowledge.py``)."""
    address = note.address
    return changes_since(note, held, ["Outline of the current note:", render_outline(note).rstrip("\n"),
                                      f"The complete current text is knowledge_read(topic={address.topic!r}, "
                                      f"scope={address.scope!r})."])


def changes_since(current: Any, held: Mapping[str, Any], tail: list[str]) -> str:
    """Every change between the version a stale write holds and ``current`` (its ``raw``, ``text``, parsed ``source``
    or ``None``, ``revision`` and ``address``), in whole lines, then the ``tail`` lines.

    After the rows written since that version, each block is the whole lines it touches: ``- `` every
    line of the held version, then ``+ `` every line of the current one, with where each side's lines
    begin and the heading the block sits under in the current text (its line as the outline prints it,
    so a header stays one line under a setext heading whose title spans lines). Putting each block's
    ``+`` lines in place of its ``-`` lines in the held version gives the current text
    (``text_change.line_blocks``). Nothing here is cut."""
    note, what = current, noun(current.address)
    old_raw, new_raw = held["text"].encode("utf-8"), note.raw
    headings = note.source.headings if note.source else ()
    blocks = text_change.line_blocks(held["text"], note.text, text_change.exact_change(held["text"], note.text))
    body = []
    for old_start, old_end, new_start, new_end, removed, added in blocks:
        under = next((h for h in reversed(headings) if h.span.start_byte <= new_start), None)
        where = f"under {heading_line(note.raw, under)}" if under else "before the first heading"
        (old_count, old_lines), (new_count, new_lines) = (_lines(old_raw, old_start, old_end, "-"),
                                                          _lines(new_raw, new_start, new_end, "+"))
        old_line, new_line = old_raw.count(b"\n", 0, old_start) + 1, new_raw.count(b"\n", 0, new_start) + 1
        body.append(f"@ yours: {old_count} line(s) from line {old_line} → current: {new_count} line(s) from line "
                    f"{new_line}, {where}: {removed} chars removed, {added} added")
        body += old_lines + new_lines
    facts = held["facts"]
    written = written_text(facts, getattr(note.address, "outside", "an outside edit, or a write before history"))
    head = [f"The revision you sent is an earlier version of this {what}{', ' if written.startswith('written') else '; '}"
            f"{written}. Current revision: {note.revision}.", "Written since that version, oldest first:"]
    head += [f"- {item['ts']} · focus {focus_text(item['focus'])} · {item['mode']} · reason: {reason_text(item['reason'])}"
             f" → {item['revision']}" for item in held["since"]] or ["- (no row: the file changed outside the writer)"]
    head.append(f"What changed from your version to the current text, in whole lines ({len(blocks)} block(s)): "
                f"in your version, put each block's + lines in place of its - lines and you have the current {what}.")
    return "\n".join(head + body + tail) + "\n"


def reason_text(value: Any) -> str:
    """A row's reason as listings show it: ``not recorded`` where none was given."""
    return "not recorded" if value in (None, "", UNKNOWN) else str(value)
