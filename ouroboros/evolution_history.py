"""Dashboard → Evolution: this repository's version history and this installation's two memory files.

``history`` is the one owner per server. ``projection`` answers at once from the saved snapshot and starts
at most one background refresh when the tags changed (or on ``force``); it never waits for a measurement.
A failed refresh keeps the previous snapshot and is retried only for new tags or on ``force``.

A version is a release tag (``vX.Y.Z`` or ``remote-vX.Y.Z``) bound to the commit it peels to: an alias joins
its version only when both name the same commit. Measurements are bound to that commit, so a measured
commit is never read again: one ``git ls-tree`` lists its tracked ``.py`` blobs and the two prompt files,
and one ``git cat-file --batch`` reads the blobs not yet counted plus ``README.md``; unchanged blobs are
counted once per refresh and the newest tree's counts are kept for the next one. The words of a version
are its own README "Version History" row at the tagged commit, else its complete tag annotation, else nothing.

The installation's two series come from its journals at request time, each in the unit its writer
records: identity.md's characters at every write (``new_len``) and the scratchpad's block-content
characters after every append (``content_len``, block headers excluded). Neither is the rendered file's
size and neither is read from the files; a date before the first record is unknown, never zero. Those
journals and the checkpoint journal behind the endpoint's older ``checkpoints`` are followed: a request
over an unchanged journal costs one stat, and an append reads only its new rows.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import logging
import os
import pathlib
import re
import threading
from collections import deque
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ouroboros.utils import atomic_write_json, iter_jsonl_objects, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)

SNAPSHOT_REL = pathlib.Path("state") / "evolution_metrics_cache.json"
SCHEMA = 3  # schema 2 could cache a missing README row when its description contained a literal pipe
_BIBLE, _SYSTEM = "BIBLE.md", "prompts/SYSTEM.md"
JOURNALS = {"identity": ("memory/identity_journal.jsonl", "new_len"),
            "scratchpad": ("memory/scratchpad_journal.jsonl", "content_len")}
CHECKPOINTS_SHOWN = 100  # the newest absorb checkpoints the endpoint still returns to older clients
# Seven fields per tag, each ended by a NUL, which no field can hold (git ends a field at one); git then writes a
# newline. An annotation's body keeps its own lines inside its field.
_REF_FORMAT = ("%(refname:short)%00%(objecttype)%00%(objectname)%00%(*objectname)%00%(creatordate:iso-strict)"
               "%00%(contents:subject)%00%(contents:body)%00")
_REF_FIELDS = 7
# A release row's description: an optional author short title, then the bold `type: heading`, then the body.
_TITLED = re.compile(r"^\*\*(?P<title>[^*]+?)\*\*\s+[—–-]\s+(?=\*\*)")
_HEAD = re.compile(r"^\*\*(?P<head>.+?)\*\*\s*(?P<body>.*)$", re.S)
_KIND = re.compile(r"^(?P<kind>[A-Za-z][\w.-]*(?:\([^)]*\))?!?):\s*(?P<heading>.+)$", re.S)


class HistoryUnavailable(RuntimeError):
    """Git could not list the tags at all: unknown, never an empty history."""


def _version_tag_re() -> "re.Pattern":
    # The repository's ONE version grammar lives with the writer of release rows (release_sync).
    from ouroboros.tools.release_sync import PRE_SUFFIX

    return re.compile(r"^(remote-)?v(\d+\.\d+\.\d+" + PRE_SUFFIX + r")$", re.IGNORECASE)


def _git(repo_dir: str, args: List[str], *, input_data: Optional[bytes] = None) -> Tuple[int, bytes, str]:
    """One bounded local Git process (tree-killed on timeout, Panic-tracked) under the update-read ceiling."""
    from supervisor.git_ops import _run_git_process_bounded
    from ouroboros.update_channels import get_managed_update_fetch_timeout_sec

    rc, out, err = _run_git_process_bounded(
        ["git", *args], timeout=float(get_managed_update_fetch_timeout_sec()), cwd=pathlib.Path(repo_dir),
        env={**os.environ, "LC_ALL": "C", "LANG": "C"}, text=False, input_data=input_data)
    return rc, out or b"", (err or b"").decode("utf-8", "replace").strip()


def _epoch(stamp: str) -> Optional[float]:
    try:
        moment = _dt.datetime.fromisoformat(str(stamp or "").strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_dt.timezone.utc)
    return moment.timestamp()


def version_level(version: str) -> str:
    """BIBLE P9's level of a version: ``major``, ``minor``, ``patch``, or ``pre`` for a pre-release."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
    if not match:
        return "pre"
    _major, minor, patch = (int(part) for part in match.groups())
    return "major" if minor == 0 and patch == 0 else "minor" if patch == 0 else "patch"


def list_versions(repo_dir: str) -> Tuple[str, List[Dict[str, Any]]]:
    """One ``for-each-ref``: the digest of every tag's identity and the versions they name, oldest first."""
    rc, out, err = _git(repo_dir, ["for-each-ref", "refs/tags", f"--format={_REF_FORMAT}"])
    if rc != 0:
        raise HistoryUnavailable(f"git for-each-ref failed (rc={rc}): {err[:200]}")
    fields = out.decode("utf-8", "replace").split("\0")
    if fields.pop().strip() or len(fields) % _REF_FIELDS:  # after the last terminator only git's newline remains
        raise HistoryUnavailable("git for-each-ref answered a listing that is not whole tag records")
    records = sorted((fields[at].lstrip("\n"), *fields[at + 1:at + _REF_FIELDS])
                     for at in range(0, len(fields), _REF_FIELDS))
    digest = hashlib.sha256("".join(f"{field}\0" for record in records for field in record).encode()).hexdigest()
    pattern = _version_tag_re()
    groups: Dict[Tuple[str, str], List[Tuple[bool, str, str, str]]] = {}
    for name, kind, obj, peeled, date, subject, body in records:
        match = pattern.match(name)
        if match:
            # An annotated tag's whole message: git's subject paragraph, then its body as written. A lightweight tag
            # (its ref names a commit) has none.
            annotation = ("\n\n".join(part for part in (subject.strip(), body.strip("\n").rstrip()) if part)
                          if kind == "tag" else "")
            groups.setdefault((match.group(2), peeled or obj), []).append((bool(match.group(1)), name, date, annotation))
    versions = []
    for (version, commit), names in groups.items():
        _remote, tag, date, annotation = sorted(names)[0]  # vX before remote-vX
        versions.append({"tag": tag, "aliases": sorted(name for _r, name, _d, _s in names if name != tag),
                         "version": version, "date": date, "commit": commit, "annotation": annotation})
    versions.sort(key=lambda row: (_epoch(row["date"]) or 0.0, row["version"]))
    return digest, versions


def _lines(data: bytes) -> int:
    return data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)


def _cat_file(repo_dir: str, names: List[str]) -> Dict[str, Optional[bytes]]:
    """Read every named object in ONE ``git cat-file --batch``; a missing object maps to ``None``."""
    if not names:
        return {}
    rc, out, err = _git(repo_dir, ["cat-file", "--batch"], input_data="".join(f"{n}\n" for n in names).encode())
    if rc != 0:
        raise RuntimeError(f"git cat-file failed (rc={rc}): {err[:200]}")
    found: Dict[str, Optional[bytes]] = {}
    cursor = 0
    for name in names:
        end = out.find(b"\n", cursor)
        header = out[cursor:end].split(b" ") if end >= 0 else []
        if len(header) == 2 and header[1] == b"missing":
            found[name], cursor = None, end + 1
            continue
        if len(header) != 3:
            raise RuntimeError(f"git cat-file answered a malformed header for {name}")
        start = end + 1
        size = int(header[2])
        found[name], cursor = out[start:start + size], start + size + 1
    return found


def _measure(repo_dir: str, commit: str, blob_lines: Dict[str, int]) -> Tuple[Dict[str, Any], List[str], Optional[str]]:
    """Measure one commit: lines of its tracked ``.py`` blobs, bytes of the two prompt files, its README."""
    rc, out, err = _git(repo_dir, ["ls-tree", "-r", "-z", commit])
    if rc != 0:
        raise RuntimeError(f"git ls-tree {commit[:12]} failed (rc={rc}): {err[:200]}")
    python: List[str] = []
    files: Dict[str, str] = {}
    for record in out.split(b"\0"):
        meta, _tab, raw_path = record.partition(b"\t")
        parts = meta.split(b" ")
        if len(parts) != 3 or parts[1] != b"blob":
            continue
        path, sha = raw_path.decode("utf-8", "replace"), parts[2].decode("ascii")
        if path.endswith(".py") and parts[0] in (b"100644", b"100755"):
            python.append(sha)
        elif path in (_BIBLE, _SYSTEM):
            files[path] = sha
    readme = f"{commit}:README.md"
    uncounted = sorted({sha for sha in python if sha not in blob_lines})
    blobs = _cat_file(repo_dir, [*uncounted, *sorted(set(files.values())), readme])
    absent = [sha for sha in uncounted if blobs.get(sha) is None]
    if absent:  # an unreadable blob is not an empty file: the commit stays unmeasured
        raise RuntimeError(f"{len(absent)} Python blob(s) of {commit[:12]} missing from the object store")
    for sha in uncounted:
        blob_lines[sha] = _lines(blobs[sha])

    def size(path: str) -> Optional[int]:
        data = blobs.get(files[path]) if path in files else None
        return len(data) if data is not None else None

    measured = {"code_lines": sum(blob_lines[sha] for sha in python),
                "bible_bytes": size(_BIBLE), "system_bytes": size(_SYSTEM)}
    text = blobs.get(readme)
    return measured, python, text.decode("utf-8", "replace") if text is not None else None


def release_row(readme: Optional[str], version: str) -> Optional[Dict[str, str]]:
    """The exact ``| version | date | text |`` row of README's Version History, or ``None``."""
    if not readme or "## Version History" not in readme:
        return None
    for line in readme.split("## Version History", 1)[1].splitlines():
        if line.startswith("## "):
            break
        row = line.strip()
        if not (row.startswith("|") and row.endswith("|")):
            continue
        # Only version and date delimit fields; historical descriptions contain literal pipes, including in code.
        cells = [cell.strip() for cell in row[1:-1].split("|", 2)]
        if len(cells) == 3 and cells[0] == version:
            return {"date": cells[1], "text": cells[2].replace("\\|", "|")}
    return None


def release_words(version: Dict[str, Any]) -> Dict[str, str]:
    """What a version says about itself: its README row split into title, type, heading and body (lossless
    apart from the bold markers), else its tag annotation (its subject the heading, the rest the body as
    written), else nothing (``source: ""``)."""
    row = version.get("row")
    if isinstance(row, dict) and row.get("text"):
        text, title = str(row["text"]).strip(), ""
        titled = _TITLED.match(text)
        if titled and not _KIND.match(titled.group("title")):  # a bold `type: …` opener is a heading, not a title
            title, text = titled.group("title").strip(), text[titled.end():]
        head = _HEAD.match(text)
        heading, body = (head.group("head").strip(), head.group("body").strip()) if head else ("", text)
        kind = _KIND.match(heading)
        return {"source": "readme", "title": title, "kind": kind.group("kind") if kind else "",
                "heading": kind.group("heading").strip() if kind else heading, "body": body}
    subject, _gap, body = str(version.get("annotation") or "").strip().partition("\n\n")
    # A subject that only repeats the product and version ("Ouroboros v7.7.1") heads nothing; its body still speaks.
    heading = re.sub(r"^(?:Ouroboros\s+)?v?" + re.escape(str(version.get("version") or "")) + r"\s*[:—–-]?\s*", "",
                     " ".join(subject.split()), flags=re.IGNORECASE)
    body = body.strip("\n").rstrip()
    if heading or body:
        return {"source": "tag", "title": "", "kind": "", "heading": heading, "body": body}
    return {"source": "", "title": "", "kind": "", "heading": "", "body": ""}


def read_snapshot(data_dir: str) -> Optional[Dict[str, Any]]:
    """The last complete refresh, or ``None`` (absent, unreadable or an older schema)."""
    saved = read_json_dict(pathlib.Path(data_dir) / SNAPSHOT_REL)
    if not saved or saved.get("schema") != SCHEMA:
        return None
    if not (isinstance(saved.get("versions"), list) and isinstance(saved.get("commits"), dict)
            and isinstance(saved.get("blob_lines"), dict)):
        return None
    return saved


def refresh(repo_dir: str, data_dir: str, digest: str, versions: List[Dict[str, Any]],
            previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Measure what the saved snapshot lacks and save the complete result atomically."""
    previous = previous or {}
    commits: Dict[str, Any] = dict(previous.get("commits") or {})
    blob_lines: Dict[str, int] = {str(k): int(v) for k, v in (previous.get("blob_lines") or {}).items()}
    known_rows = {(row.get("version"), row.get("commit")): row.get("row")
                  for row in previous.get("versions") or [] if isinstance(row, dict) and "row" in row}
    readmes: Dict[str, Optional[str]] = {}
    python_of: Dict[str, List[str]] = {}
    unmeasured, unmeasured_error = 0, ""
    for row in versions:
        commit = row["commit"]
        if commit not in commits:
            try:
                commits[commit], python_of[commit], readmes[commit] = _measure(repo_dir, commit, blob_lines)
            except (RuntimeError, ValueError) as exc:
                log.warning("Evolution history: version %s not measured: %s", row["tag"], exc)
                unmeasured += 1
                unmeasured_error = unmeasured_error or f"{row['tag']}: {exc}"[:300]
    newest_python = python_of.get(versions[-1]["commit"]) if versions else None
    missing_readmes = sorted({row["commit"] for row in versions if (row["version"], row["commit"]) not in known_rows
                              and row["commit"] in commits and row["commit"] not in readmes})
    for commit, text in _cat_file(repo_dir, [f"{c}:README.md" for c in missing_readmes]).items():
        readmes[commit.split(":", 1)[0]] = text.decode("utf-8", "replace") if text is not None else None
    saved_versions = []
    for row in versions:
        key = (row["version"], row["commit"])
        entry = dict(row)
        if key in known_rows:
            entry["row"] = known_rows[key]
        elif row["commit"] in readmes:
            entry["row"] = release_row(readmes[row["commit"]], row["version"])
        saved_versions.append(entry)
    current = {row["commit"] for row in versions}
    snapshot = {
        "schema": SCHEMA, "saved_at": utc_now_iso(), "refs": digest,
        # Versions whose commit could not be read: kept with unknown counts, retried by the next refresh (force).
        "unmeasured": unmeasured, "unmeasured_error": unmeasured_error,
        "versions": saved_versions,
        "commits": {commit: facts for commit, facts in commits.items() if commit in current},
        # Only the newest tree's counts help the next refresh; older blobs are dropped here.
        "blob_lines": ({sha: blob_lines[sha] for sha in newest_python} if newest_python is not None
                       else dict(previous.get("blob_lines") or {})),
    }
    atomic_write_json(pathlib.Path(data_dir) / SNAPSHOT_REL, snapshot)
    return snapshot


def _appended_rows(path: pathlib.Path, start: int) -> Tuple[Iterator[Dict[str, Any]], int]:
    """The object rows a journal holds after byte ``start`` and the byte offset they end at. A row still being written
    (no newline yet) waits for the next read; a malformed one is skipped. An unreadable journal raises ``OSError``."""
    with path.open("rb") as handle:
        handle.seek(start)
        data = handle.read()
    complete = data.rfind(b"\n") + 1
    return iter_jsonl_objects(path, _handle=io.BytesIO(data[:complete])), start + complete


def _journal_series(path: pathlib.Path, field: str, start: int = 0,
                    known: Optional[List[Tuple[float, int, str]]] = None) -> Tuple[List[Tuple[float, int, str]], int]:
    """``(epoch, characters, stamp)`` of every journal row that records ``field``, oldest first, and the byte offset
    read up to. From ``start`` only the appended rows are read and joined to ``known``."""
    rows, offset = _appended_rows(path, start)
    series = list(known or [])
    in_order = True
    for row in rows:
        value, moment = row.get(field), _epoch(row.get("ts", ""))
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0 and moment is not None:
            in_order = in_order and (not series or series[-1][0] <= moment)
            series.append((moment, value, str(row["ts"])))
    if not in_order:
        series.sort(key=lambda item: item[0])
    return series, offset


def _checkpoint_rows(path: pathlib.Path, start: int = 0,
                     known: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[Dict[str, Any]], int]:
    """The public form of the newest ``CHECKPOINTS_SHOWN`` absorb checkpoints, oldest first, and the byte offset read up
    to. ``cycle_outcome`` rows feed the promotion digest and are not shown. From ``start`` only the appended rows."""
    from ouroboros.outcomes import public_task_result

    rows, offset = _appended_rows(path, start)
    fresh = deque((row for row in rows if row.get("kind") != "cycle_outcome"), maxlen=CHECKPOINTS_SHOWN)
    return [*(known or []), *map(public_task_result, fresh)][-CHECKPOINTS_SHOWN:], offset


def legacy_points(versions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The `points` shape published before this history, kept for older clients: one point per tag name (aliases too), oldest
    first, with the same repository numbers as `versions` (KB = bytes / 1024) and null where nothing was measured. The
    three memory fields are null: their old values mixed journal characters with file bytes; see `install`."""
    def kib(value: Any) -> Optional[float]:
        return round(value / 1024, 2) if isinstance(value, int) else None

    return [{"tag": name, "date": row["date"], "code_lines": row["code_lines"], "bible_kb": kib(row["bible_bytes"]),
             "system_kb": kib(row["system_bytes"]), "identity_kb": None, "scratchpad_kb": None, "memory_kb": None}
            for row in versions for name in [row["tag"], *row["aliases"]]]


class EvolutionHistory:
    """The single-flight refresh and the request-time projection of one server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._status: Dict[str, Any] = {"state": "idle", "started_at": "", "finished_at": "", "error": ""}
        self._failed_refs = ""
        # Per journal path: (dev, inode, size, mtime), the byte offset read up to, and what was read from it.
        self._journals: Dict[str, Tuple[Tuple[int, int, int, int], int, Any]] = {}

    def join(self, timeout: Optional[float] = None) -> None:
        worker = self._worker
        if worker is not None:
            worker.join(timeout)

    def _run(self, repo_dir: str, data_dir: str, digest: str, versions: List[Dict[str, Any]],
             previous: Optional[Dict[str, Any]]) -> None:
        error = ""
        try:
            refresh(repo_dir, data_dir, digest, versions, previous)
        except Exception as exc:  # the saved snapshot stays; the failure is reported, never hidden
            log.warning("Evolution history refresh failed", exc_info=True)
            error = f"{type(exc).__name__}: {exc}"[:300]
        with self._lock:
            self._status = {"state": "failed" if error else "idle", "started_at": self._status["started_at"],
                            "finished_at": utc_now_iso(), "error": error}
            self._failed_refs = digest if error else ""

    def projection(self, repo_dir: str, data_dir: str, *, force: bool = False) -> Dict[str, Any]:
        """The saved history at once, the refresh status, and the installation's journals at version dates."""
        listing_error = ""
        with self._lock:
            running = self._worker is not None and self._worker.is_alive()
            status = dict(self._status)
        # Read after the status: a refresh ending in between leaves "running" beside the new copy, so the client asks once more.
        saved = read_snapshot(data_dir)
        if not running:
            try:
                digest, versions = list_versions(repo_dir)
            except HistoryUnavailable as exc:
                digest, versions, listing_error = "", [], str(exc)
            with self._lock:
                running = self._worker is not None and self._worker.is_alive()
                outdated = saved is None or saved.get("refs") != digest
                if digest and not running and (force or (outdated and digest != self._failed_refs)):
                    self._status = {"state": "running", "started_at": utc_now_iso(), "finished_at": "", "error": ""}
                    self._worker = threading.Thread(target=self._run, args=(repo_dir, data_dir, digest, versions, saved),
                                                    name="evolution-history-refresh", daemon=True)
                    self._worker.start()
                # Read in the same critical section: a refresh that ends at once still reports "running" here first.
                status = dict(self._status)
        if listing_error and status["state"] != "running":
            status.update(state="failed", error=listing_error)
        rows = sorted((row for row in (saved or {}).get("versions") or [] if isinstance(row, dict)),
                      key=lambda row: (_epoch(row.get("date", "")) or 0.0, str(row.get("version", ""))))
        commits = (saved or {}).get("commits") or {}
        out = []
        for row in rows:
            facts = commits.get(row.get("commit")) or {}
            words = release_words(row)
            level = version_level(str(row.get("version", "")))
            out.append({
                "tag": row.get("tag", ""), "aliases": list(row.get("aliases") or []), "version": row.get("version", ""),
                "date": row.get("date", ""), "commit": row.get("commit", ""), "level": level,
                # A milestone (owner 21.10, A): a MAJOR or MINOR release (P9), or any version its release author named.
                "milestone": bool(words["title"]) or level in ("major", "minor"),
                "code_lines": facts.get("code_lines"), "bible_bytes": facts.get("bible_bytes"),
                "system_bytes": facts.get("system_bytes"), "release": words,
            })
        status["unmeasured"] = int((saved or {}).get("unmeasured") or 0)
        status["unmeasured_error"] = str((saved or {}).get("unmeasured_error") or "")
        return {"versions": out, "saved_at": (saved or {}).get("saved_at", ""), "refresh": status,
                "install": self._install(data_dir, [row["date"] for row in out]),
                "points": legacy_points(out), "cached": saved is not None}

    def checkpoints(self, data_dir: str) -> List[Dict[str, Any]]:
        """The endpoint's older ``checkpoints``: the public form of the newest absorb checkpoints, oldest first."""
        from ouroboros.evolution_checkpoints import CHECKPOINTS_REL

        return list(self._follow(pathlib.Path(data_dir) / CHECKPOINTS_REL, _checkpoint_rows, []))

    def _follow(self, path: pathlib.Path, read: Callable[..., Tuple[Any, int]], empty: Any) -> Any:
        """``read`` over one append-only journal, again only when its (dev, inode, size, mtime) changed: an append reads
        just the new rows, a replaced or shorter file is read whole. Absent: nothing is recorded. Unreadable: nothing is
        known this time and nothing is remembered, so the next request reads it again."""
        try:
            stat = path.stat()
            key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        except OSError:
            key = (-1, -1, -1, -1)
        memo = self._journals.get(str(path))
        if memo is not None and memo[0] == key:
            return memo[2]
        found, offset = empty, 0
        if key[2] >= 0:
            grown = memo is not None and memo[0][:2] == key[:2] and key[2] >= memo[1] >= 0
            try:
                found, offset = read(path, memo[1], memo[2]) if grown else read(path)
            except Exception:
                log.warning("Evolution history: %s could not be read", path, exc_info=True)
                self._journals.pop(str(path), None)
                return empty
        self._journals[str(path)] = (key, offset, found)
        return found

    def _install(self, data_dir: str, dates: List[str]) -> Dict[str, Any]:
        install: Dict[str, Any] = {}
        for name, (relative, field) in JOURNALS.items():
            def read(path, start=0, known=None, field=field):
                return _journal_series(path, field, start, known)

            series = self._follow(pathlib.Path(data_dir) / relative, read, [])
            values, index, current = [], 0, None
            for moment in (_epoch(date) for date in dates):
                while moment is not None and index < len(series) and series[index][0] <= moment:
                    current, index = series[index][1], index + 1
                values.append(current if moment is not None else None)
            install[name] = {"at_versions": values, "first_at": series[0][2] if series else "",
                             "latest": {"at": series[-1][2], "chars": series[-1][1]} if series else None}
        return install


history = EvolutionHistory()

__all__ = ["EvolutionHistory", "HistoryUnavailable", "SNAPSHOT_REL", "history", "legacy_points", "list_versions",
           "read_snapshot", "refresh", "release_row", "release_words", "version_level"]
