"""Per-lane trace bundle: the lane server's own journals, published beside ``result.json``.

A lane's data root stays on the machine that ran it, so on a CI runner it is gone with the runner: the two SM1
deadline failures of issue #1501 could not be read for that reason. ``publish_lane_traces`` copies what explains a
run into ``lanes/<id>/traces/`` with the data-root layout kept: the journals, the advisory ledger, the review records,
the candidate and adoption records, the qualification receipts, the queue and campaign state and the observability call
manifests, of the lane root and of every headless task fork (where review evidence lands), plus the server's own
stdout and stderr (``server_stdio.log``, kept beside the data root because a server can fail before it opens a log).
Nothing else: never ``settings.json`` (it carries the key), never ``memory/``, the gzip payload blobs or any credential
store.

The SQLite stores (``SQLITE_STORES``: current facts, usage, dialogue) travel as data, never as files. Their committed
rows may still live in the WAL, and replacing a value inside database pages damages them. So each store is copied with
its WAL and rollback journal into private scratch (the source is never opened, no ``-shm`` or lock appears beside it),
recovered and quick-checked there, and written as ``<store>.logical.jsonl``: one header line (source sizes, schema,
column types, row counts) and one line per row of its user tables and virtual tables, never ``sqlite_*`` internals or a
virtual table's shadow pages. Every string it carries (names of tables, columns and types, the schema SQL, TEXT values)
is redacted before JSON encoding, whose ``\\uXXXX`` escapes would hide a non-ASCII secret from every byte scan after
it, and a BLOB on its raw bytes before base64 would do the same. Outside that guarantee: a value kept compressed or
otherwise encoded in any column (the writers of these three stores in this tree declare TEXT/INTEGER/REAL columns and
encode nothing; a later writer may), and a secret already stored JSON-escaped (one with a quote, a backslash or a
control character inside JSON text), which the byte redaction of the copied journals misses alike. An export is all or
nothing: a store that fails its check, its read or the residue scan of the finished export is withheld with a named
entry in ``gaps`` (reason and exported/total counts) and the rest of the bundle stays. Each export is consistent for
its own store only, not across stores or journals, and is no runnable backup.

The CI artifact of a public repository is downloadable by any signed-in user, so every credential value the lane
could have seen is replaced by ``<redacted:NAME sha256:...>`` (the stand's ``credential_fingerprint``) before a byte
is written, and a bundle in which a value still occurs afterwards is deleted rather than published:
``{"published": false, "reason": "secret_residue"}`` in ``result.json``. A bundle above ``BUNDLE_LIMIT_BYTES`` keeps
the newest tail of each journal, opened by a ``trace_truncated`` line and listed in the same fact; JSON files are
never cut.
"""
from __future__ import annotations

import base64
import json
import math
import pathlib
import shutil
import sqlite3
import tempfile
from typing import Any

from devtools.benchmarks.common.secrets import credential_fingerprint

TRACES_DIR = "traces"
SERVER_STDIO = "server_stdio.log"   # in the lane directory (``IsolatedServer.stdio_path``), at the bundle's top level
# Relative to a drive root: the lane's own and each headless task's forked one (FORK_ROOTS).
TRACE_GLOBS = ("logs/*.jsonl", "logs/*.log", "logs/*.log.[0-9]*", "state/advisory_review.json",
               "state/usage_attempts.jsonl", "state/queue_snapshot.json", "state/evolution_campaign.json",
               "state/review_ledger/*.json", "state/private_copies/*.json", "state/preflight_receipts/*.json",
               "state/body_adoption/handoff.json", "state/pending_restart_verify*.json",
               "observability/calls/*/*.json")
SQLITE_STORES = ("state/current.sqlite", "state/usage.sqlite", "state/dialog.sqlite")
LOGICAL_SUFFIX = ".logical.jsonl"
FORK_ROOTS = "state/headless_tasks/*/data"
# Line-oriented: the only files cut to a tail, with the numbered backups server.py's rotating handler leaves.
JOURNAL_SUFFIXES = (".jsonl", ".log")
# One attempt's bound (a bundle is written per attempt, not per concurrent lane). A stub SM1 lane leaves about 1.5 MB
# of these files and a paid lane's tool outputs run to tens of MB; 200 MiB keeps the CI run (one SM1 attempt) at
# 200 MiB and the full operator set (nine attempts) at 1.8 GiB, minutes of upload from a hosted runner, and a bundle
# that still exceeds it keeps the NEWEST part of every journal, the end where a deadline or stall shows.
BUNDLE_LIMIT_BYTES = 200 * 2**20
# Settings keys whose values are credentials (provider keys, tokens, passwords). Suffixes, not ``"TOKEN" in name``:
# ``*_MAX_TOKENS`` holds a number that must not be scrubbed out of every journal.
SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_CREDENTIALS", "_PASSWORD", "_SECRET")


def lane_secrets(settings_path: pathlib.Path, named: dict[str, str]) -> dict[str, str]:
    """``{value: name}`` of every credential the lane could have seen: the secret-shaped keys of its applied settings
    file (read here, never copied) and the values the caller names (the key under its ``--key-env`` name)."""
    try:
        cfg = json.loads(pathlib.Path(settings_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    found: dict[str, str] = {}
    for name, value in [*named.items(), *(cfg.items() if isinstance(cfg, dict) else ())]:
        text = value.strip() if isinstance(value, str) else ""
        if text and (name in named or str(name).upper().endswith(SECRET_SUFFIXES)):
            found.setdefault(text, str(name))
    return found


def publish_lane_traces(lane: pathlib.Path, data_root: pathlib.Path, secrets: dict[str, str], *,
                        limit: int = BUNDLE_LIMIT_BYTES) -> dict:
    """Write ``<lane>/traces/`` and return its typed fact for ``result.json``. Never raises: the lane's row is
    recorded after this call whatever happens here."""
    dest, replacements = pathlib.Path(lane) / TRACES_DIR, _replacements(secrets)
    try:
        if not pathlib.Path(data_root).is_dir():
            return {"published": False, "reason": "no_data_root"}
        shutil.rmtree(dest, ignore_errors=True)
        with tempfile.TemporaryDirectory(prefix=".traces_scratch_", dir=lane) as scratch:   # beside the raw data root, never uploaded
            exports, gaps = _logical_exports(pathlib.Path(data_root), pathlib.Path(scratch), replacements, secrets)
            facts = _write_bundle(_sources(pathlib.Path(lane), pathlib.Path(data_root)), dest, replacements, int(limit), exports)
        facts.update(sqlite_exports=[export["fact"] for export in exports], gaps=gaps)
        residue = _residue(dest, secrets)
        if residue:
            shutil.rmtree(dest, ignore_errors=True)
            return {"published": False, "reason": "secret_residue", "files_with_residue": residue}
        return {"published": True, **facts}
    except Exception as exc:  # noqa: BLE001 - a failed copy is a recorded fact, never a lost lane row
        shutil.rmtree(dest, ignore_errors=True)
        error = _redact(f"{type(exc).__name__}: {exc}".encode("utf-8"), replacements)[0]
        return {"published": False, "reason": "collect_error", "error": error.decode("utf-8", "replace")[:300]}


def _sources(lane: pathlib.Path, data_root: pathlib.Path) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """``(file, path in the bundle)`` for the server's stdio and every traced regular file of the lane root and its forks."""
    found: dict[pathlib.Path, pathlib.Path] = {}
    if (lane / SERVER_STDIO).is_file():
        found[lane / SERVER_STDIO] = pathlib.Path(SERVER_STDIO)
    for root in (data_root, *sorted(data_root.glob(FORK_ROOTS))):
        for pattern in TRACE_GLOBS:
            for path in root.glob(pattern):
                if path.is_file() and not path.is_symlink() and path.name != "settings.json":
                    found[path] = path.relative_to(data_root)
    return sorted(found.items(), key=lambda item: item[1].as_posix())


def _write_bundle(sources: list, dest: pathlib.Path, replacements: list, limit: int, exports: list) -> dict:
    sizes = {rel: src.stat().st_size for src, rel in sources}
    journals = {rel: size for rel, size in sizes.items() if _is_journal(rel)}
    whole = sum(export["fact"]["bytes"] for export in exports)   # like a JSON file, an export is never cut
    fixed = whole + sum(size for rel, size in sizes.items() if rel not in journals)
    keep = _tail_budgets(journals, limit - fixed) if sum(sizes.values()) + whole > limit else journals
    dest.mkdir(parents=True, exist_ok=True)
    total, redacted = whole, sum(export["fact"]["redacted"] for export in exports)
    for export in exports:
        target = dest / export["fact"]["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(export["path"]), str(target))
    truncated: list[dict] = []
    for src, rel in sources:
        size = sizes[rel]
        kept = keep.get(rel, size)
        with open(src, "rb") as fh:
            fh.seek(size - kept)
            data = fh.read(kept)
        if kept < size:   # drop the partial first line, then say so in the file itself
            data = data[data.find(b"\n") + 1:] if b"\n" in data else b""
            fact = {"path": rel.as_posix(), "original_bytes": size, "kept_tail_bytes": len(data)}
            truncated.append(fact)
            data = json.dumps({"trace_truncated": {**fact, "limit_bytes": limit}}).encode("utf-8") + b"\n" + data
        data, hits = _redact(data, replacements)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        total, redacted = total + len(data), redacted + hits
    return {"files": len(sources) + len(exports), "bytes": total, "redacted": redacted, "limit_bytes": limit, "truncated": truncated}


class _ExportGap(Exception):
    """Why one store's export is withheld; the rest of the bundle does not depend on it."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail)
        self.reason = reason


def _logical_exports(data_root: pathlib.Path, scratch: pathlib.Path, replacements: list, secrets: dict) -> tuple[list, list]:
    """``(exports, gaps)``: an export per store of the lane root and its forks that came through whole."""
    exports, gaps = [], []
    needles = [value.encode("utf-8") for value in secrets]
    for root in (data_root, *sorted(data_root.glob(FORK_ROOTS))):
        for relative in SQLITE_STORES:
            source = root / relative
            if not source.is_file() or source.is_symlink():
                continue
            rel = source.relative_to(data_root).as_posix()
            work = pathlib.Path(tempfile.mkdtemp(dir=scratch))
            counts = {"tables_total": 0, "tables_exported": 0, "rows_total": 0, "rows_exported": 0}
            try:
                exports.append(_export_store(source, rel, work, replacements, needles, counts))
            except _ExportGap as exc:
                gaps.append({"path": rel, "reason": exc.reason, **counts, **_gap_error(exc, replacements)})
            except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
                gaps.append({"path": rel, "reason": f"unreadable:{type(exc).__name__}", **counts, **_gap_error(exc, replacements)})
    return exports, gaps


def _gap_error(exc: Exception, replacements: list) -> dict:
    text = _redact(str(exc).encode("utf-8", "replace"), replacements)[0].decode("utf-8", "replace")
    return {"error": text[:300]} if text else {}


def _stamp(path: pathlib.Path) -> tuple | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def _private_copy(source: pathlib.Path, work: pathlib.Path) -> pathlib.Path:
    """The store with its WAL and rollback journal, stable across the copy or refused; its source is never opened."""
    paths = [source, *(source.with_name(source.name + suffix) for suffix in ("-wal", "-journal"))]
    for _attempt in range(3):
        before = [_stamp(path) for path in paths]
        for path in paths:
            (work / path.name).unlink(missing_ok=True)
            if path.exists():
                shutil.copyfile(path, work / path.name)
        if before == [_stamp(path) for path in paths]:
            return work / source.name
    raise _ExportGap("source_changing")


def _user_tables(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """``(name, table|virtual)``: a virtual table is read through its own SELECT, its shadow tables never are."""
    listed = conn.execute("PRAGMA table_list").fetchall()
    if not listed:   # the pragma answers nothing below SQLite 3.37: shadow pages could not be told apart here
        raise _ExportGap("sqlite_too_old", sqlite3.sqlite_version)
    return [(row[1], row[2]) for row in listed
            if row[0] == "main" and row[2] in ("table", "virtual") and not row[1].startswith("sqlite_")]


def _export_store(source: pathlib.Path, rel: str, work: pathlib.Path, replacements: list, needles: list, counts: dict) -> dict:
    private = _private_copy(source, work)
    hits = 0

    def text(value: str) -> str:
        nonlocal hits
        data, found = _redact(value.encode("utf-8", "surrogateescape"), replacements)
        hits += found
        return data.decode("utf-8", "surrogateescape")

    def logical(value: Any) -> Any:
        nonlocal hits
        if isinstance(value, str):
            return text(value)
        if isinstance(value, bytes):
            data, found = _redact(value, replacements)   # the raw bytes: base64 hides a secret from every later scan
            hits += found
            return {"$type": "blob", "bytes": len(data), "base64": base64.b64encode(data).decode("ascii")}
        if isinstance(value, float) and not math.isfinite(value):
            return {"$type": "real", "value": repr(value)}
        return value

    def line(payload: dict) -> bytes:
        nonlocal hits
        data, found = _redact(json.dumps(payload, ensure_ascii=True).encode("ascii") + b"\n", replacements)
        hits += found
        return data

    conn = sqlite3.connect(private)   # recovery runs on this copy
    try:
        conn.text_factory = lambda raw: raw.decode("utf-8", "surrogateescape")   # damaged text is data, not a crash
        checked = conn.execute("PRAGMA quick_check").fetchall()
        if checked != [("ok",)]:
            raise _ExportGap("quick_check_failed", text("; ".join(str(row[0]) for row in checked[:3])))
        schema = dict(conn.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table'").fetchall())
        tables = []
        for name, kind in _user_tables(conn):   # names are data too: redacted before ``ensure_ascii`` could escape them
            quoted = '"' + name.replace('"', '""') + '"'
            tables.append({"name": text(name), "kind": kind, "sql": text(str(schema.get(name) or "")), "quoted": quoted,
                           "columns": [{"name": text(row[1]), "type": text(row[2])} for row in conn.execute(f"PRAGMA table_info({quoted})")],
                           "rows": conn.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0]})
        counts.update(tables_total=len(tables), rows_total=sum(table["rows"] for table in tables))
        out = work / f"{source.name}{LOGICAL_SUFFIX}"
        with open(out, "wb") as sink:
            sink.write(line({"sqlite_logical_export": {
                "source": rel, "source_bytes": source.stat().st_size,
                "wal_bytes": (work / f"{source.name}-wal").stat().st_size if (work / f"{source.name}-wal").exists() else 0,
                "sqlite_version": sqlite3.sqlite_version, "quick_check": "ok",
                "consistency": "this store alone at its copy; not atomic with other stores or journals",
                "tables": [{key: value for key, value in table.items() if key != "quoted"} for table in tables]}}))
            for table in tables:
                exported = 0
                cursor = conn.execute(f"SELECT * FROM {table['quoted']}")
                columns = [column[0] for column in cursor.description]
                for values in cursor:
                    sink.write(line({"table": table["name"], "row": {text(c): logical(v) for c, v in zip(columns, values)}}))
                    exported += 1
                    counts["rows_exported"] += 1
                if exported != table["rows"]:
                    raise _ExportGap("row_count_mismatch", f"{table['name']}: {exported} of {table['rows']}")
                counts["tables_exported"] += 1
    finally:
        conn.close()
    data = out.read_bytes()
    if any(needle in data for needle in needles):
        raise _ExportGap("secret_residue")
    return {"path": out, "fact": {"path": rel, "file": rel + LOGICAL_SUFFIX, "tables": counts["tables_total"],
                                  "rows": counts["rows_exported"], "bytes": len(data), "redacted": hits}}


def _is_journal(rel: pathlib.Path) -> bool:
    """``*.jsonl``/``*.log``, or a rotated ``<name>.log.<n>`` backup; a JSON file never is."""
    rotated = rel.suffix[1:].isdigit() and rel.with_suffix("").suffix in JOURNAL_SUFFIXES
    return rel.suffix in JOURNAL_SUFFIXES or rotated


def _tail_budgets(sizes: dict, budget: int) -> dict:
    """Water-filling: a journal smaller than an even share of what is left stays whole; the rest split the remainder."""
    keep, left = {}, max(0, int(budget))
    pending = sorted(sizes.items(), key=lambda item: item[1])
    for index, (rel, size) in enumerate(pending):
        keep[rel] = min(size, left // (len(pending) - index))
        left -= keep[rel]
    return keep


def _replacements(secrets: dict[str, str]) -> list[tuple[bytes, bytes]]:
    """Longest value first, so a credential that contains another is replaced whole."""
    return [(value.encode("utf-8"), f"<redacted:{name} {credential_fingerprint(value)}>".encode("utf-8"))
            for value, name in sorted(secrets.items(), key=lambda item: -len(item[0]))]


def _redact(data: bytes, replacements: list) -> tuple[bytes, int]:
    hits = 0
    for value, marker in replacements:
        hits += data.count(value)
        data = data.replace(value, marker)
    return data, hits


def _residue(dest: pathlib.Path, secrets: dict[str, str]) -> list[str]:
    """Bundle files in which a credential value still occurs: any one withholds the whole bundle."""
    needles = [value.encode("utf-8") for value in secrets]
    hits = []
    for path in sorted(dest.rglob("*")) if needles else ():
        if path.is_file():
            data = path.read_bytes()
            if any(needle in data for needle in needles):
                hits.append(path.relative_to(dest).as_posix())
    return hits
