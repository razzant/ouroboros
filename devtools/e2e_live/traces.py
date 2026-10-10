"""Per-lane trace bundle: the lane server's own journals, published beside ``result.json``.

A lane's data root stays on the machine that ran it, so on a CI runner it is gone with the runner: the two SM1
deadline failures of issue #1501 could not be read for that reason. ``publish_lane_traces`` copies what explains a
run into ``lanes/<id>/traces/`` with the data-root layout kept: the journals, task results, the advisory ledger, the
money ledger, the queue and campaign state and the observability call manifests, of the lane root and of every
headless task fork (where review evidence lands). Nothing else: never ``settings.json`` (it carries the key), never
``memory/``, the gzip payload blobs or any credential store.

The CI artifact of a public repository is downloadable by any signed-in user, so every credential value the lane
could have seen is replaced by ``<redacted:NAME sha256:...>`` (the stand's ``credential_fingerprint``) before a byte
is written, and a bundle in which a value still occurs afterwards is deleted rather than published:
``{"published": false, "reason": "secret_residue"}`` in ``result.json``. A bundle above ``BUNDLE_LIMIT_BYTES`` keeps
the newest tail of each journal, opened by a ``trace_truncated`` line and listed in the same fact; JSON files are
never cut.
"""
from __future__ import annotations

import json
import pathlib
import shutil

from devtools.benchmarks.common.secrets import credential_fingerprint

TRACES_DIR = "traces"
# Relative to a drive root: the lane's own and each headless task's forked one (FORK_ROOTS).
TRACE_GLOBS = ("logs/*.jsonl", "logs/*.log", "logs/*.log.[0-9]*", "task_results/*.json", "state/advisory_review.json",
               "state/usage.sqlite", "state/usage_attempts.jsonl", "state/queue_snapshot.json", "state/evolution_campaign.json",
               "observability/calls/*/*.json")
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
        facts = _write_bundle(_sources(pathlib.Path(data_root)), dest, replacements, int(limit))
        residue = _residue(dest, secrets)
        if residue:
            shutil.rmtree(dest, ignore_errors=True)
            return {"published": False, "reason": "secret_residue", "files_with_residue": residue}
        return {"published": True, **facts}
    except Exception as exc:  # noqa: BLE001 - a failed copy is a recorded fact, never a lost lane row
        shutil.rmtree(dest, ignore_errors=True)
        error = _redact(f"{type(exc).__name__}: {exc}".encode("utf-8"), replacements)[0]
        return {"published": False, "reason": "collect_error", "error": error.decode("utf-8", "replace")[:300]}


def _sources(data_root: pathlib.Path) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """``(file, path relative to the data root)`` for every traced regular file of the lane root and its forks."""
    found: dict[pathlib.Path, pathlib.Path] = {}
    for root in (data_root, *sorted(data_root.glob(FORK_ROOTS))):
        for pattern in TRACE_GLOBS:
            for path in root.glob(pattern):
                if path.is_file() and not path.is_symlink() and path.name != "settings.json":
                    found[path] = path.relative_to(data_root)
    return sorted(found.items(), key=lambda item: item[1].as_posix())


def _write_bundle(sources: list, dest: pathlib.Path, replacements: list, limit: int) -> dict:
    sizes = {rel: src.stat().st_size for src, rel in sources}
    journals = {rel: size for rel, size in sizes.items() if _is_journal(rel)}
    fixed = sum(size for rel, size in sizes.items() if rel not in journals)
    keep = _tail_budgets(journals, limit - fixed) if sum(sizes.values()) > limit else journals
    dest.mkdir(parents=True, exist_ok=True)
    total = redacted = 0
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
    return {"files": len(sources), "bytes": total, "redacted": redacted, "limit_bytes": limit, "truncated": truncated}


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
