"""Directory artifact traversal and publication, including confirmed bind views.

The process-output owner validates every source and archive name. This packer
streams those members into the ordinary task artifact ledger and ZIP, whose
names derive from the members' digests; ``current_directory_capture`` proves a
retained pair still holds them, so an unchanged declaration needs no new pair.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import zipfile
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Union

from ouroboros.artifact_storage import temporary_sibling
from ouroboros.utils import atomic_write_json


def iter_artifact_tree(root: pathlib.Path) -> Iterable[pathlib.Path]:
    """Enumerate the full tree without following directory links or hiding I/O failures."""
    def failed(exc: OSError) -> None:
        raise exc
    for directory, dirs, files in os.walk(root, onerror=failed, followlinks=False):
        dirs.sort()
        for name in sorted([*dirs, *files]):
            path = pathlib.Path(directory) / name
            path.lstat()  # A disappearing or unreadable member is not an empty entry.
            yield path


def iter_artifact_entries(root: pathlib.Path, mounts: tuple[tuple[str, pathlib.Path], ...] = ()):
    """Yield visible (archive name, host source), pruning covered parent trees.

    Each nested bind is independently confirmed by the executor and admitted by
    output policy. A host decoy below a mountpoint is never visited, even for a
    baseline. Symlink members remain visible to policy but are never traversed.
    """
    def failed(exc: OSError) -> None:
        raise exc
    for prefix, base in (("", root), *mounts):
        if prefix:
            base.lstat()
            yield prefix, base
        if not base.is_dir() or base.is_symlink():
            continue
        covered = {name for name, _ in mounts if name != prefix}
        for directory, dirs, files in os.walk(base, onerror=failed, followlinks=False):
            relative = pathlib.Path(directory).relative_to(base)
            names = sorted([*dirs, *files])
            for name in names:
                rel = pathlib.PurePosixPath(prefix, relative.as_posix(), name).as_posix()
                if rel in covered:
                    if name in dirs:
                        dirs.remove(name)
                    continue
                path = pathlib.Path(directory) / name
                path.lstat()
                yield rel, path


def copy_directory_to_task_artifacts(
    ctx: Any,
    source_path: Union[pathlib.Path, str],
    *,
    kind: str = "process_output_directory",
    member_paths: Iterable[pathlib.Path] | None = None,
    member_sources: Iterable[tuple[str, pathlib.Path]] | None = None,
    artifact_dir: pathlib.Path | None = None,
) -> List[Dict[str, Any]]:
    """Package a generated directory as a manifest ledger plus zip artifact."""
    from ouroboros.artifacts import (
        _archive_previous_artifact_version,
        _register_task_artifact_records,
        artifact_record,
        stream_artifact_file,
        task_artifact_dir_path,
        task_id_for_artifacts,
    )

    source = pathlib.Path(source_path).expanduser().resolve(strict=False)
    if not source.is_dir():
        return []
    if artifact_dir is None:
        artifact_dir = task_artifact_dir_path(ctx.drive_root, task_id_for_artifacts(ctx), create=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    entries, excluded = _directory_entries(source, member_paths, member_sources)
    file_records: List[Dict[str, Any]] = []
    tmp_zip_path = temporary_sibling(artifact_dir / "directory.zip")
    tmp_ledger_path = temporary_sibling(artifact_dir / "directory.json")
    try:
        with zipfile.ZipFile(tmp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for rel, path in entries:
                path = _available_member(path, member_sources is not None)
                with archive.open(rel, "w", force_zip64=True) as member:
                    measured = stream_artifact_file(path, member)
                file_records.append({"path": rel, **measured})
        ledger_name, zip_name = _pair_names(source, file_records)
        ledger_path, zip_path = artifact_dir / ledger_name, artifact_dir / zip_name
        atomic_write_json(tmp_ledger_path, _ledger_document(kind, source, file_records, excluded, zip_name),
                          trailing_newline=True)
        if ctx is not None:
            # A repaired pair can replace bytes written directly by a process; keep their ordinary recovery history.
            task_id = task_id_for_artifacts(ctx)
            _archive_previous_artifact_version(ctx.drive_root, task_id, zip_path, tmp_zip_path)
            _archive_previous_artifact_version(ctx.drive_root, task_id, ledger_path, tmp_ledger_path)
        tmp_zip_path.replace(zip_path)
        tmp_ledger_path.replace(ledger_path)
    except Exception:
        for tmp_path in (tmp_zip_path, tmp_ledger_path):
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    records = [
        artifact_record(ledger_path, kind=f"{kind}_manifest", source_path=str(source)),
        artifact_record(zip_path, kind=kind, source_path=str(source)),
    ]
    _register_task_artifact_records(artifact_dir, records)
    return records


def _directory_entries(source: pathlib.Path, member_paths: Iterable[pathlib.Path] | None,
                       member_sources: Iterable[tuple[str, pathlib.Path]] | None) -> tuple[list, list]:
    """``(entries, excluded)``: validated (archive name, host path) members and the left-out WAL indexes."""
    if member_paths is not None and member_sources is not None:
        raise ValueError("choose member_paths or policy-checked member_sources")
    if member_sources is not None:
        entries = list(member_sources)
    elif member_paths is None:
        members = sorted(p for p in iter_artifact_tree(source) if p.is_file() and not p.is_symlink())
        entries = [(p.relative_to(source).as_posix(), p) for p in members]
    else:
        entries = []
        for p in member_paths:
            path = pathlib.Path(p)
            try:
                entries.append((path.resolve(strict=False).relative_to(source).as_posix(), path))
            except ValueError as exc:
                raise OSError(f"directory member escapes its source: {path}") from exc
    seen = set()
    for rel, path in entries:
        parts = pathlib.PurePosixPath(rel)
        if (not rel or parts.is_absolute() or ".." in parts.parts or rel in seen
                or os.name == "nt" and "\\" in rel):
            raise OSError(f"invalid or duplicate directory member name: {rel}")
        seen.add(rel)
    from ouroboros.store import is_wal_index  # shared memory no raw reader opens; a copy rebuilds its own

    excluded = [{"path": rel, "reason": "SQLite WAL index"} for rel, p in entries if is_wal_index(p)]
    return sorted((rel, p) for rel, p in entries if not is_wal_index(p)), excluded


def _available_member(path: pathlib.Path, policy_checked: bool) -> pathlib.Path:
    if path.is_symlink() or not path.is_file() or policy_checked and path.resolve(strict=True) != path.absolute():
        raise OSError(f"directory member is unavailable: {path}")
    return path


def _pair_names(source: pathlib.Path, files: List[Dict[str, Any]]) -> tuple[str, str]:
    """The ledger and ZIP names, derived from the source path and each member's name and digest."""
    tree = sha256(str(source).encode("utf-8", errors="replace") + b"\0")
    for row in files:
        tree.update(row["path"].encode("utf-8", errors="replace") + b"\0" + row["sha256"].encode("ascii") + b"\0")
    stem = (source.name.replace("/", "_").replace("\\", "_") or "directory") + "." + tree.hexdigest()[:8]
    return f"{stem}.manifest.json", f"{stem}.zip"


def _ledger_document(kind: str, source: pathlib.Path, files: list, excluded: list, zip_name: str) -> Dict[str, Any]:
    return {"schema_version": 1, "kind": kind, "source_path": str(source), "file_count": len(files), "files": files,
            **({"excluded": excluded} if excluded else {}), "zip_name": zip_name}


def current_directory_capture(
    ctx: Any,
    source_path: Union[pathlib.Path, str],
    *,
    kind: str = "process_output_directory",
    member_paths: Iterable[pathlib.Path] | None = None,
    member_sources: Iterable[tuple[str, pathlib.Path]] | None = None,
) -> List[Dict[str, Any]]:
    """The registered mutable ledger+ZIP rows already holding exactly these members' current bytes, or [].

    The members' digests name the pair; each retained file must still match its row and the ledger the
    digests. Nothing is repacked, so no claim rests on compressor output. Any doubt answers [] and the
    caller packs again."""
    from ouroboros.artifacts import (
        registered_task_artifact,
        stream_artifact_file,
        task_artifact_dir_path,
        task_id_for_artifacts,
    )

    source = pathlib.Path(source_path).expanduser().resolve(strict=False)
    try:
        entries, excluded = _directory_entries(source, member_paths, member_sources)
        files = [{"path": rel, **stream_artifact_file(_available_member(path, member_sources is not None))}
                 for rel, path in entries]
        drive, task_id = ctx.drive_root, task_id_for_artifacts(ctx)
        names, store = _pair_names(source, files), task_artifact_dir_path(drive, task_id)
        rows = [registered_task_artifact(drive, task_id, name) or {} for name in names]
        ledger = io.BytesIO()
        for row, name, row_kind, sink in zip(rows, names, (f"{kind}_manifest", kind), (ledger, None)):
            retained = store / name
            if (row.get("immutable") or not row.get("sha256") or row.get("kind") != row_kind
                    or row.get("source_path") != str(source) or retained.is_symlink()
                    or pathlib.Path(str(row.get("path") or "")).resolve(strict=False) != retained.resolve(strict=False)):
                return []
            stream_artifact_file(retained, sink, expected=row)
        if json.loads(ledger.getvalue()) != _ledger_document(kind, source, files, excluded, names[1]):
            return []
    except (OSError, ValueError, RuntimeError):
        return []
    return rows
