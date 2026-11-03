"""Directory artifact traversal and publication, including confirmed bind views.

The process-output owner validates every source and archive name. This packer
streams those members into the ordinary task artifact ledger and ZIP.
"""
from __future__ import annotations

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
    entries = [(rel, p) for rel, p in entries if not is_wal_index(p)]
    file_records: List[Dict[str, Any]] = []
    tree_hasher = sha256(str(source).encode("utf-8", errors="replace") + b"\0")
    tmp_zip_path = temporary_sibling(artifact_dir / "directory.zip")
    tmp_ledger_path = temporary_sibling(artifact_dir / "directory.json")
    try:
        with zipfile.ZipFile(tmp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for rel, path in sorted(entries):
                if (path.is_symlink() or not path.is_file()
                        or member_sources is not None and path.resolve(strict=True) != path.absolute()):
                    raise OSError(f"directory member is unavailable: {path}")
                with archive.open(rel, "w", force_zip64=True) as member:
                    measured = stream_artifact_file(path, member)
                tree_hasher.update(rel.encode("utf-8", errors="replace") + b"\0")
                tree_hasher.update(measured["sha256"].encode("ascii") + b"\0")
                file_records.append({"path": rel, **measured})
        safe_stem = source.name.replace("/", "_").replace("\\", "_") or "directory"
        tree_digest = tree_hasher.hexdigest()[:8]
        ledger_path = artifact_dir / f"{safe_stem}.{tree_digest}.manifest.json"
        zip_path = artifact_dir / f"{safe_stem}.{tree_digest}.zip"
        atomic_write_json(
            tmp_ledger_path,
            {
                "schema_version": 1,
                "kind": kind,
                "source_path": str(source),
                "file_count": len(file_records),
                "files": file_records,
                **({"excluded": excluded} if excluded else {}),
                "zip_name": zip_path.name,
            },
            trailing_newline=True,
        )
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
