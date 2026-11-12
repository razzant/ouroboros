"""How a knowledge history row stores the texts of a note's versions, and the one reader that reads them back.

A note's history row (``publication: "source_capture"``, written by ``knowledge.write_knowledge_note``)
names the version it replaced (``old_sha256``) and the one it published (``new_sha256``), each the
sha256 of the note's exact bytes. It stores their texts in one of three forms:

- **by change**: ``base`` names the row whose new text is the replaced version, by that row's byte
  offset in this journal and the version's sha256, and ``change`` is the exact two-way change from that
  text (``text_change``: ``[[start byte in the old text, removed text, added text], ...]``). Every row
  with a base carries its change. It also stores the whole ``new_content`` when the change is not
  smaller than the new text, or when the change bytes since the nearest whole text on its base chain,
  this change included, would exceed the new text's bytes: two byte counts, no constant. A row without
  the whole text records that running sum as ``chain_bytes``, so the next writer need not walk the
  chain; it only decides when a text is stored whole, the hash decides what is true.
- **whole**: ``new_content`` and the replaced text inline as ``old_content``: a write without a
  verified base (no row in the writer's tail read holds the note's current version and reads back:
  an outside edit, a deleted note, a torn, damaged or unverified row) and every archive/restore row.
- **legacy**: ``old_content_ref``, a memory-reader entry address of an earlier row whose
  ``new_content`` is the replaced text. Rows written before changes were stored keep it; nothing
  writes it now.

``Texts`` reads the exact old or new text of any row, verified by sha256. Forward: through ``base``
links to the nearest stored text, applying changes. When that path breaks: from any stored text of
the note (a whole row, an inline old text, the live note) through the changes that connect it to the
version by hash, applying or unapplying them. So a landed version of a connected segment is reachable
from both of its ends, which are whole texts or the live note, and one damaged row loses no landed
version. The live note is a segment's newer end only while it still holds the segment's last version:
after an outside edit or a deleted note the next row has no verified base and keeps the outside text
inline, so that segment reads only forward from its whole start, and a damaged row in it loses the
versions after it (typed misses, never a wrong text). A capture that never landed lives only in its
own row (its version was never the note's, so it is nobody's base). A miss is ``TextUnavailable``,
naming where the path broke; a returned text always hashes to the version asked for. Rows stay
uncompressed; nothing here writes, indexes or keeps anything beyond one ``Texts``.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping

from ouroboros import text_change


class TextUnavailable(ValueError):
    """No verified text for a row's version: the message names where the path to it broke."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _revision(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and not value.strip("0123456789abcdef")


def _whole(row: Mapping[str, Any]) -> bool:
    """The row stores its new text whole; a digest-era row's empty ``new_content`` is no text."""
    return isinstance(row.get("new_content"), str) and not row.get("content_digested")


def _base(row: Mapping[str, Any], offset: int) -> tuple[int, str] | None:
    """``(offset, version)`` the row's ``base`` names, beside its change: always an earlier row."""
    base = row.get("base")
    if (isinstance(base, Mapping) and type(base.get("offset")) is int and 0 <= base["offset"] < offset
            and _revision(base.get("sha256")) and isinstance(row.get("change"), list)):
        return base["offset"], base["sha256"]
    return None


def _parse(line: bytes) -> Any:
    try:
        return json.loads(line)
    except (ValueError, RecursionError):
        return None


def change_bytes(change: Iterable) -> int:
    """The UTF-8 bytes a change carries: every removed and every added text."""
    return sum(len(removed.encode("utf-8")) + len(added.encode("utf-8")) for _start, removed, added in change)


def encode(old: str, new: str, base: tuple[int, Mapping[str, Any]] | None,
           change: Iterable | None = None) -> dict[str, Any]:
    """The text fields of a row that replaces ``old`` by ``new``.

    ``base`` is ``(offset, row)`` of the verified row whose new text is ``old`` (``verified_base``),
    ``change`` the exact change of the two texts when the caller already has it. Without a base, or
    when the change does not rebuild ``new`` from ``old`` and ``old`` from ``new``, both texts are
    stored whole. A base row whose running sum is unreadable starts the chain again: a whole text."""
    whole = {"old_content": old, "new_content": new}
    if base is None:
        return whole
    items = [list(item) for item in (text_change.exact_change(old, new) if change is None else change)]
    try:
        exact = text_change.apply_change(old, items) == new and text_change.unapply_change(new, items) == old
    except (ValueError, TypeError):
        exact = False
    if not exact:
        return whole
    offset, row = base
    before = 0 if _whole(row) else row.get("chain_bytes")
    size, limit = change_bytes(items), len(new.encode("utf-8"))
    fields = {"base": {"offset": offset, "sha256": sha256_text(old)}, "change": items}
    if type(before) is not int or before < 0 or size >= limit or before + size > limit:
        return {**fields, "new_content": new}
    return {**fields, "chain_bytes": before + size}


def verified_base(journal: Path, rows: Iterable[tuple[int, dict]], revision: str) -> tuple[int, dict] | None:
    """The newest of a note's ``rows`` (newest first) whose new text is ``revision`` and reads back forward.

    The writer passes its tail read, taken under the shelf lock. The text must read through the row's
    own base links to a stored whole text, verified at every step: the live note is no witness, so the
    version a new row replaces stays readable after the note changes, and a row whose chain is broken
    is no base, so a damaged row costs the next write one whole row instead of extending a segment that
    reads only from its other end. A capture that never landed holds another version than the note:
    never a base."""
    texts = Texts(journal)
    for offset, row in rows:
        texts.rows.setdefault(offset, row)
        if row.get("new_sha256") == revision and texts._new(offset, row, revision, []) is not None:
            return offset, row
    return None


class Texts:
    """The verified texts of one journal's rows. One instance serves one read and keeps the rows it parsed.

    ``live`` is the note's current text when the caller holds it: the version it hashes to needs no row."""

    def __init__(self, journal: Path, live: str | None = None):
        self.journal = journal
        self.rows: dict[int, dict] = {}
        self._live = {sha256_text(live): live} if isinstance(live, str) else {}
        self._graphs: dict[str, tuple[dict, dict]] = {}

    def row(self, offset: int) -> dict:
        """The JSON row whose line starts at ``offset``, or ``TextUnavailable`` saying why there is none."""
        if offset not in self.rows:
            try:
                with self.journal.open("rb") as handle:
                    if offset:
                        handle.seek(offset - 1)
                        if handle.read(1) != b"\n":
                            raise TextUnavailable(f"no journal line starts at byte offset {offset}")
                    line = handle.readline()
            except OSError as exc:
                raise TextUnavailable(f"the journal is unreadable at byte offset {offset} ({type(exc).__name__})") from None
            row = _parse(line)
            if not isinstance(row, dict):
                raise TextUnavailable(f"the line at byte offset {offset} is not a JSON row")
            self.rows[offset] = row
        return self.rows[offset]

    def text(self, offset: int, side: str) -> str:
        """The exact ``side`` (``"old"`` or ``"new"``) text of the row at ``offset``, verified by its sha256."""
        row = self.row(offset)
        want = row.get(f"{side}_sha256")
        if not _revision(want):
            raise TextUnavailable(f"the row at byte offset {offset} names no {side} version")
        if want in self._live:
            return self._live[want]
        if row.get("content_digested"):  # the digest compaction kept no text on either side
            raise TextUnavailable(f"the row at byte offset {offset} holds no text (digest-era)")
        keys = ("new_content", "change") if side == "new" else ("old_content", "old_content_ref", "change")
        if not any(key in row for key in keys):  # nothing stored for that side: no path to break, none to search from
            raise TextUnavailable(f"the row at byte offset {offset} holds no {side} text")
        broken: list[str] = []
        text = self._new(offset, row, want, broken) if side == "new" else self._old(offset, row, want, broken)
        if text is None:
            text = self._connected(row.get("topic"), want, broken)
        if text is None:
            raise TextUnavailable("; ".join(broken))
        return text

    def _base_row(self, offset: int, base: tuple[int, str], broken: list[str]) -> dict | None:
        """The row a base names, when it holds that version; a wrong address is caught by the hash."""
        try:
            row = self.row(base[0])
        except TextUnavailable as exc:
            broken.append(f"{exc} (the base of the row at byte offset {offset})")
            return None
        if row.get("new_sha256") != base[1]:
            broken.append(f"the base the row at byte offset {offset} names does not hash to its version: the row at "
                          f"byte offset {base[0]} holds {row.get('new_sha256') or 'no version'}, not {base[1]}")
            return None
        return row

    def _new(self, offset: int, row: dict, want: str, broken: list[str]) -> str | None:
        """The row's new text: stored whole, the live note, or its base's text with the changes applied."""
        chain = []
        while True:
            if want in self._live:
                text = self._live[want]
                break
            if row.get("content_digested"):  # the digest compaction kept no text, whatever else the row holds
                broken.append(f"the row at byte offset {offset} holds no text (digest-era)")
                return None
            whole, base = _whole(row), _base(row, offset)
            if whole and sha256_text(row["new_content"]) == want:
                text = row["new_content"]
                break
            if whole or base is None:
                broken.append(f"the stored text does not hash to the version (the row at byte offset {offset})" if whole
                              else f"the row at byte offset {offset} holds no text of version {want} and names no base")
            if base is None:
                return None
            chain.append((offset, row, want))
            row = self._base_row(offset, base, broken)
            if row is None:
                return None
            offset, want = base
        for at, item, version in reversed(chain):
            try:
                text = text_change.apply_change(text, item["change"])
            except (ValueError, TypeError):
                text = None
            if text is None or sha256_text(text) != version:
                broken.append(f"the change of the row at byte offset {at} does not rebuild its version {version}")
                return None
        return text

    def _old(self, offset: int, row: dict, want: str, broken: list[str]) -> str | None:
        """The text the row replaced: inline, an earlier row a legacy reference names, or its base's new text."""
        said, stored = len(broken), row.get("old_content")
        if isinstance(stored, str):
            if sha256_text(stored) == want:
                return stored
            broken.append(f"the stored text does not hash to the version (the row at byte offset {offset})")
        ref = row.get("old_content_ref")
        if isinstance(ref, Mapping):
            text = self._referenced(row, ref)
            if text is not None and sha256_text(text) == want:
                return text
            broken.append(f"the earlier row the row at byte offset {offset} names holds no text of version {want}")
        base = _base(row, offset)
        if base is None or base[1] != want:
            if len(broken) == said:
                broken.append(f"the row at byte offset {offset} names no base holding version {want}")
            return None
        base_row = self._base_row(offset, base, broken)
        return None if base_row is None else self._new(base[0], base_row, want, broken)

    def _referenced(self, row: Mapping[str, Any], ref: Mapping[str, Any]) -> str | None:
        """The ``new_content`` a legacy ``old_content_ref`` names in this journal (its row's scope), or ``None``."""
        where = row.get("address")
        scope = where.get("scope") if isinstance(where, Mapping) else None
        if ref.get("journal") != f"knowledge:{scope}" or type(ref.get("offset")) is not int:
            return None
        try:
            text = self.row(ref["offset"]).get("new_content")
        except TextUnavailable:
            return None
        return text if isinstance(text, str) else None

    def _connected(self, topic: Any, want: str, broken: list[str]) -> str | None:
        """``want`` rebuilt from the nearest stored text of this note through the changes that connect them by hash.

        A stored text or a change that fails its hash is set aside and the search runs again without it."""
        if not isinstance(topic, str):
            return None
        if topic not in self._graphs:
            self._graphs[topic] = self._graph(topic)
        stored, links = self._graphs[topic]
        dropped: set = set()
        while True:
            found = self._path(want, stored, links, dropped)
            if found is None:
                broken.append("no other stored text of this note connects to it through the changes in the journal")
                return None
            text = self._rebuilt(*found, stored, dropped)
            if text is not None:
                return text

    def _graph(self, topic: str) -> tuple[dict, dict]:
        """This note's rows over the whole journal: version -> ``(offset, side)`` of each row storing its text
        whole, and version -> ``(offset, other version)`` of each change that links the two."""
        needle = json.dumps(topic, ensure_ascii=False).encode("utf-8")  # as every writer of the journal spells it
        stored: dict[str, list] = {}
        links: dict[str, list] = {}
        try:
            with self.journal.open("rb") as handle:
                at = 0
                for line in handle:
                    offset, at = at, at + len(line)
                    row = _parse(line) if needle in line else None
                    if not isinstance(row, dict) or row.get("topic") != topic:
                        continue
                    new, old, base = row.get("new_sha256"), row.get("old_sha256"), _base(row, offset)
                    if _whole(row) and _revision(new):
                        stored.setdefault(new, []).append((offset, "new"))
                    if isinstance(row.get("old_content"), str) and _revision(old):
                        stored.setdefault(old, []).append((offset, "old"))
                    if base is not None and _revision(new) and base[1] != new:
                        links.setdefault(base[1], []).append((offset, new))
                        links.setdefault(new, []).append((offset, base[1]))
        except OSError:
            pass
        return stored, links

    def _path(self, want: str, stored: dict, links: dict, dropped: set) -> tuple[str, list] | None:
        """The nearest version with a stored text, by fewest changes, and the steps from it back to ``want``."""
        parent: dict[str, Any] = {want: None}
        queue = deque([want])
        while queue:
            version = queue.popleft()
            if version in self._live or any(where not in dropped for where in stored.get(version, ())):
                steps, at = [], version
                while parent[at] is not None:
                    nearer, offset = parent[at]
                    steps.append((at, nearer, offset))
                    at = nearer
                return version, steps
            for offset, other in links.get(version, ()):
                if offset not in dropped and other not in parent:
                    parent[other] = (version, offset)
                    queue.append(other)
        return None

    def _rebuilt(self, seed: str, steps: list, stored: dict, dropped: set) -> str | None:
        """The text of ``seed`` carried through ``steps`` (each a row's change, applied or unapplied), each step verified."""
        text = self._live.get(seed)
        for where in [] if text is not None else stored.get(seed, ()):
            if where in dropped:
                continue
            try:
                candidate = self.row(where[0]).get(f"{where[1]}_content")
            except TextUnavailable:
                candidate = None
            if isinstance(candidate, str) and sha256_text(candidate) == seed:
                text = candidate
                break
            dropped.add(where)
        if text is None:
            return None
        for source, target, offset in steps:
            try:
                row = self.row(offset)
                base = _base(row, offset)  # the change runs from the base version to the row's own
                rebuild = text_change.apply_change if base is not None and base[1] == source else text_change.unapply_change
                text = rebuild(text, row["change"])
            except (TextUnavailable, ValueError, TypeError, KeyError):
                text = None
            if text is None or sha256_text(text) != target:
                dropped.add(offset)
                return None
        return text
