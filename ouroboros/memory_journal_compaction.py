"""Digest-only compaction of old memory-journal snapshots (CPL4-C16, owner 4A).

``memory/identity_journal.jsonl``, ``memory/knowledge_history.jsonl`` and
``memory/knowledge/patterns_history.jsonl`` record the FULL old+new document
text on every write — O(doc×edits) growth, the worst byte offenders in the
memory plane. Owner decision 4A: entries younger than the unified GC
retention keep their full text; older entries become digest-only — the
content keys are replaced by their sha256 + length (existing hashes are
never overwritten) and the row is marked ``content_digested``.

Strictly fail-closed per line: an unparseable line, a row without a
readable ``ts``, a row with nothing to digest, or a row whose STORED digest
disagrees with the text it claims to describe is carried through
BYTE-IDENTICAL. The scratchpad journal (typed rows, its own eviction
contract) and the consciousness observation inbox (unacknowledged rows must
survive verbatim) are deliberately NOT in scope.

This is the only sweep that DESTROYS content rather than whole dead files,
so its three guards are load-bearing (audit #15-11):

* **Digest truth before deletion.** The digest becomes the only surviving
  record of the text, so a stored ``*_sha256``/``*_len`` that does not match
  the text is never published over it: the row keeps its full content and the
  mismatch is reported as a typed fact (``digest_mismatch``, surfaced on the
  ``memory_journal_compaction`` event).
* **A lock nobody can steal.** The append lock is taken ``owner_aware_stale``
  so elapsed time alone can never hand a second writer the same journal.
* **Publish only an unchanged source.** ``append_jsonl`` falls back to an
  UNLOCKED append after its own lock timeout, so a concurrent row can still
  land while this rewrite streams. The file is re-identified (size + inode)
  against the bytes actually consumed immediately before ``os.replace``; any
  delta aborts the publish and the journal stays as the appender left it.

The rewrite streams line by line into the temp sibling — the journals are the
worst byte offenders in the memory plane and must never be loaded whole.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
from typing import Any, Dict, Optional, Tuple

from ouroboros.deadline_utils import parse_deadline_ts
from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from ouroboros.utils import jsonl_append_lock_path

log = logging.getLogger(__name__)

_JOURNAL_RELS = (
    pathlib.Path("memory") / "identity_journal.jsonl",
    pathlib.Path("memory") / "knowledge_history.jsonl",
    pathlib.Path("memory") / "knowledge" / "patterns_history.jsonl",
)
_CONTENT_KEYS = ("old_content", "new_content")


def _digest_row(row: Dict[str, Any]) -> str:
    """Replace full-text keys with sha256+len.

    Returns ``"digested"`` when text was dropped, ``"mismatch"`` when a STORED
    digest or length contradicts the text it describes, ``""`` when there was
    nothing to digest. On a mismatch the row is left EXACTLY as found: the
    digest is about to become the only surviving record of that content, and
    publishing a digest already known to be false while deleting the last
    correct copy is unrecoverable.
    """
    dropped = []
    for key in _CONTENT_KEYS:
        value = row.get(key)
        if not isinstance(value, str):
            continue
        prefix = key[: -len("_content")]
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest() if value else ""
        stored_digest = row.get(f"{prefix}_sha256")
        stored_len = row.get(f"{prefix}_len")
        if isinstance(stored_digest, str) and stored_digest != digest:
            return "mismatch"
        if isinstance(stored_len, int) and not isinstance(stored_len, bool) and stored_len != len(value):
            return "mismatch"
        dropped.append((key, prefix, digest, len(value)))
    if not dropped:
        return ""
    for key, prefix, digest, length in dropped:
        row[f"{prefix}_sha256"] = digest
        row[f"{prefix}_len"] = length
        del row[key]
    row["content_digested"] = True
    return "digested"


def _digest_line(raw: bytes, cutoff: float) -> Tuple[bytes, str]:
    """One journal line, transformed or carried through byte-identical."""
    stripped = raw.strip()
    if not stripped:
        return raw, ""
    try:
        row = json.loads(stripped.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return raw, ""  # fail-closed: never rewrite what cannot be read
    if not isinstance(row, dict):
        return raw, ""
    parsed_ts = parse_deadline_ts(str(row.get("ts") or ""))
    if parsed_ts is None or parsed_ts.timestamp() >= cutoff:
        return raw, ""  # fresh, or age unknowable: keep full text
    outcome = _digest_row(row)
    if outcome != "digested":
        return raw, outcome
    return json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n", "digested"


def _publish_if_unchanged(
    path: pathlib.Path, tmp: pathlib.Path, expected: Tuple[int, int, int],
) -> bool:
    """Swap the rewritten journal in ONLY if the source is still what we read.

    ``append_jsonl`` appends WITHOUT the sidecar lock once its own acquisition
    times out, so a concurrent row can land while this rewrite streams. The
    identity is (bytes consumed, device, inode): a grown file means an append
    we did not carry over, a different inode means the journal was replaced
    outright. Either way the rewrite is dropped and the appender's file stands.
    """
    try:
        stat = path.stat()
    except OSError:
        return False
    if (int(stat.st_size), int(stat.st_dev), int(stat.st_ino)) != expected:
        return False
    os.replace(tmp, path)
    return True


def _compact_one(path: pathlib.Path, cutoff: float) -> Tuple[int, int, str]:
    """Digest one journal in place.

    Returns ``(digested, mismatched, error)``; a nonempty ``error``
    (``lock_unavailable`` / ``source_changed``) means nothing was published and
    the journal is byte-identical to what the appenders left.
    """
    lock_path = jsonl_append_lock_path(path)
    lock_fd = acquire_exclusive_file_lock(
        lock_path, timeout_sec=2.0, stale_sec=10.0, owner_aware_stale=True,
    )
    if lock_fd is None:
        return 0, 0, "lock_unavailable"
    tmp = path.with_name(path.name + ".compact.tmp")
    published = False
    try:
        digested = 0
        mismatched = 0
        consumed = 0
        with path.open("rb") as source:
            start = os.fstat(source.fileno())
            with tmp.open("wb") as sink:
                for raw in source:  # streaming: one line in flight, never the file
                    consumed += len(raw)
                    out, outcome = _digest_line(raw, cutoff)
                    sink.write(out)
                    if outcome == "digested":
                        digested += 1
                    elif outcome == "mismatch":
                        mismatched += 1
        if not digested:
            return 0, mismatched, ""
        published = _publish_if_unchanged(
            path, tmp, (consumed, int(start.st_dev), int(start.st_ino)),
        )
        if not published:
            return 0, 0, "source_changed"
        return digested, mismatched, ""
    finally:
        if not published:
            try:
                tmp.unlink()
            except OSError:
                log.debug("Failed to drop the journal compaction temp file", exc_info=True)
        release_exclusive_file_lock(lock_path, lock_fd)


def compact_memory_journal_snapshots(
    drive_root: Any,
    retention_days: Optional[int] = None,
    *,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Digest old full-text snapshots in the three memory journals."""
    from ouroboros.retention import age_cutoff, get_gc_retention_days

    if retention_days is None:
        retention_days = get_gc_retention_days()
    cutoff = age_cutoff(retention_days, now)
    report: Dict[str, Any] = {"digested": {}, "digest_mismatch": {}, "errors": []}
    root = pathlib.Path(drive_root)
    for rel in _JOURNAL_RELS:
        path = root / rel
        if not path.exists():
            continue
        try:
            digested, mismatched, error = _compact_one(path, cutoff)
        except OSError:
            report["errors"].append({"journal": rel.as_posix(), "error": "io_error"})
            continue
        if error:
            report["errors"].append({"journal": rel.as_posix(), "error": error})
        if digested:
            report["digested"][rel.as_posix()] = digested
        if mismatched:
            # Typed fact, not a silent skip: a stored digest that contradicts
            # its own text means one of the two is already corrupt, and the
            # content stays in full until a human looks.
            report["digest_mismatch"][rel.as_posix()] = mismatched
    return report


__all__ = ["compact_memory_journal_snapshots"]
