"""Verified file results beside workspace.patch; temporary preparation owns no state."""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tempfile
from uuid import uuid4
import zipfile


def _path(root, relative):
    rel = PurePosixPath(relative)
    if not relative or rel.is_absolute() or ".." in rel.parts or "\\" in relative:
        raise ValueError(f"invalid file-result path: {relative!r}")
    path = Path(root) / relative
    if not path.parent.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"file-result path escapes target: {relative!r}")
    return path


def _side(path):
    from ouroboros.artifacts import stream_artifact_file
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(mode):
        link = os.readlink(path)
        data = os.fsencode(link)
        return {"kind": "symlink", "link_target": link, "size": len(data),
                "sha256": sha256(data).hexdigest(), "mode": stat.S_IMODE(mode)}
    if not stat.S_ISREG(mode):
        raise ValueError(f"file-result path is not a file or symlink: {path}")
    return {"kind": "file", **stream_artifact_file(path), "mode": stat.S_IMODE(mode)}


def _git_side(root, base_ref, relative):
    if base_ref == "(unborn)":
        return None
    entry = subprocess.run(["git", "ls-tree", "-z", base_ref, "--", f":(literal){relative}"],
                           cwd=root, capture_output=True, check=True).stdout
    if not entry:
        return None
    fields = entry.split(b"\t", 1)[0].split()
    mode, kind, oid = fields
    if kind != b"blob":
        raise ValueError(f"unsupported Git file-result baseline: {relative}")
    from ouroboros.artifacts import stream_artifact_file
    with tempfile.TemporaryFile() as blob:
        subprocess.run(["git", "cat-file", "blob", oid.decode()], cwd=root,
                       stdout=blob, stderr=subprocess.PIPE, check=True)
        result = {**stream_artifact_file(blob), "mode": stat.S_IMODE(int(mode, 8))}
        if mode == b"120000":
            blob.seek(0)
            result.update(kind="symlink", link_target=os.fsdecode(blob.read()))
        else:
            result["kind"] = "file"
        return result


def _matches(actual, expected, *, exact_mode=False):
    if actual is None or expected is None:
        return actual is expected
    keys = ("kind", "size", "sha256", "link_target")
    return (all(actual.get(key) == expected.get(key) for key in keys)
            and (actual["kind"] == "symlink" or (
                actual["mode"] == expected["mode"] if exact_mode else
                bool(actual["mode"] & 0o111) == bool(expected["mode"] & 0o111))))


def _validate_side(side):
    if side is None:
        return
    if (not isinstance(side, dict) or side.get("kind") not in {"file", "symlink"} or not isinstance(side.get("mode"), int)
            or not 0 <= side["mode"] <= 0o7777 or not isinstance(side.get("size"), int)
            or side["size"] < 0 or len(side.get("sha256", "")) != 64):
        raise ValueError("file-result side has no complete identity")
    if side["kind"] == "symlink":
        data = os.fsencode(side["link_target"])
        if sha256(data).hexdigest() != side["sha256"] or len(data) != side["size"]:
            raise ValueError("file-result symlink identity mismatch")


def _archive_files(records, cap_dir):
    """Resolve relocated artifact records by their capture-local names and identities."""
    from ouroboros.artifacts import stream_artifact_file
    files = {}
    archives = {}
    for record in records:
        name = record.get("name", "")
        if Path(name).name != name or not name:
            raise ValueError("file-result artifact has an invalid name")
        artifact = _path(cap_dir, name)
        if artifact.is_symlink() or not record.get("sha256") or "size" not in record:
            raise ValueError("file-result artifact has no exact capture identity")
        stream_artifact_file(artifact, expected=record)
        archives[name] = artifact
    for record in records:
        if record.get("kind") != "workspace_file_outputs_manifest":
            continue
        ledger = json.loads(archives[record["name"]].read_text(encoding="utf-8"))
        archive = archives.get(ledger["zip_name"])
        if archive is None:
            raise ValueError("file-result archive is missing")
        with zipfile.ZipFile(archive) as zipped:
            names = zipped.namelist()
            expected_names = [row["path"] for row in ledger["files"]]
            if len(set(names)) != len(names) or sorted(names) != sorted(expected_names):
                raise ValueError("file-result archive does not match its complete manifest")
        for row in ledger["files"]:
            relative = row["path"]
            _path(cap_dir, relative)
            if relative in files:
                raise ValueError(f"duplicate file-result path: {relative}")
            files[relative] = {**row, "archive_path": str(archive), "member": relative}
    return files


def changed_file_inputs(root, file_baseline):
    """Compare only materialized inputs, including files subsequently deleted."""
    changed = []
    for relative, before in file_baseline.items():
        _validate_side(before)
        if not _matches(_side(_path(root, relative)), before, exact_mode=True):
            changed.append(relative)
    return changed


def capture_file_output_changes(root, base_ref, paths, records, cap_dir, *, file_baseline=None):
    """Bind archived postimages and exact Git or copied-file preimages."""
    file_baseline = file_baseline or {}
    files = _archive_files(records, cap_dir)
    changes = []
    for relative in sorted(set(paths)):
        after = _side(_path(root, relative))
        if after and after["kind"] == "file":
            captured = files.get(relative)
            if not captured or any(after[key] != captured[key] for key in ("size", "sha256")):
                raise OSError(f"file-result changed after capture: {relative}")
        before = (file_baseline[relative] if relative in file_baseline
                  else _git_side(root, base_ref, relative))
        _validate_side(before)
        changes.append({"path": relative, "before": before, "after": after})
    if set(files) != {row["path"] for row in changes if row["after"] and row["after"]["kind"] == "file"}:
        raise OSError("file-result types changed after capture")
    return changes


def file_output_changes(manifest, cap_dir):
    """Load complete, verifiable rows; old captures without preimages need recapture."""
    files = _archive_files(manifest.get("file_outputs") or [], cap_dir)
    rows = manifest.get("file_output_changes")
    excluded = {row["path"] for row in manifest.get("tracked_excluded", [])
                if row.get("reason") == "captured as file-reference artifact"}
    if rows is None:
        if files or excluded:
            raise ValueError("file-result baseline unavailable; recapture the retained workspace")
        return []
    result = []
    seen = set()
    for row in rows:
        relative = row["path"]
        _path(cap_dir, relative)
        if relative in seen or "before" not in row or "after" not in row:
            raise ValueError(f"invalid file-result change: {relative}")
        seen.add(relative)
        _validate_side(row["before"])
        _validate_side(row["after"])
        after = row["after"]
        if after and after["kind"] == "file":
            captured = files.get(relative)
            if not captured or any(after[key] != captured[key] for key in ("size", "sha256")):
                raise ValueError(f"file-result postimage is missing or mismatched: {relative}")
            row = {**row, "archive_path": captured["archive_path"], "member": captured["member"]}
        result.append(dict(row))
    if not (set(files) | excluded).issubset(seen):
        raise ValueError("file-result changes omit captured paths")
    if set(files) != {row["path"] for row in result if row["after"] and row["after"]["kind"] == "file"}:
        raise ValueError("file-result manifest contains unaccounted files")
    return result


def verify_file_outputs(rows, target):
    """Verify already-applied shared-workspace results without preparing another apply."""
    return all(_matches(_side(_path(target, row["path"])), row["after"], exact_mode=True) for row in rows)


def file_output_states(rows, target):
    """Each file result's state now against its exact captured postimage (bytes, kind, mode).

    ``unchanged``, ``changed`` (another kind now included: a directory, FIFO, socket or device),
    ``removed`` (captured present, now absent), ``recreated`` (captured absent, now present) or
    ``unknown`` when the path cannot be read; never a postimage taken from the live tree.
    """
    states = {}
    for row in rows:
        path = None
        try:
            path = _path(target, row["path"])
            current = _side(path)
        except ValueError:  # not a path in the target (unknown), or no file or symlink now (a type change)
            states[row["path"]] = ("unknown" if path is None else "recreated" if row["after"] is None else "changed")
            continue
        except OSError:
            states[row["path"]] = "unknown"
            continue
        states[row["path"]] = ("unchanged" if _matches(current, row["after"], exact_mode=True) else
                               "recreated" if row["after"] is None else "removed" if current is None else "changed")
    return states


class PreparedFileOutputs:
    """Temporary verified bytes and exact preimages, owned by the caller's Git lock."""
    def __init__(self, rows, target, temporary, baseline_sha, file_baseline=None):
        from ouroboros.artifacts import copy_artifact_file, stream_artifact_file
        self.rows, self.target = rows, Path(target).resolve()
        file_baseline = file_baseline or {}
        self.paths = [row["path"] for row in rows]
        self.temporary, self.before, self.applied = Path(temporary), {}, []
        for index, row in enumerate(rows):
            relative = row["path"]
            target_path = _path(self.target, relative)
            observed = _side(target_path)
            if not _matches(observed, row["before"], exact_mode=relative in file_baseline):
                raise ValueError(f"file-result target changed from baseline: {relative}")
            if relative in file_baseline:
                _validate_side(file_baseline[relative])
                if not _matches(file_baseline[relative], row["before"], exact_mode=True):
                    raise ValueError(f"file-result baseline mismatch: {relative}")
            elif baseline_sha and not _matches(_git_side(self.target, baseline_sha, relative), row["before"]):
                raise ValueError(f"file-result baseline mismatch: {relative}")
            self.before[relative] = observed
            if observed and observed["kind"] == "file":
                copy_artifact_file(target_path, self.temporary / f"{index}.before", expected=observed)
            after = row["after"]
            if after and after["kind"] == "file":
                prepared = self.temporary / f"{index}.after"
                with zipfile.ZipFile(row["archive_path"]) as archive, archive.open(row["member"]) as source, prepared.open("xb") as sink:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        sink.write(chunk)
                stream_artifact_file(prepared, expected=after)
                prepared.chmod(after["mode"])

    def verify_applied(self):
        return verify_file_outputs(self.rows, self.target)

    def _write(self, index, side, suffix):
        from ouroboros.artifacts import copy_artifact_file
        path = _path(self.target, self.rows[index]["path"])
        if side is None:
            path.unlink(missing_ok=True)
        elif side["kind"] == "file":
            copy_artifact_file(self.temporary / f"{index}.{suffix}", path, expected=side)
        elif side["kind"] == "symlink":
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{uuid4().hex}.tmp")
            try:
                temporary.symlink_to(side["link_target"])
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            raise ValueError(f"unsupported file-result kind: {side['kind']}")

    def apply(self):
        # Recheck every path before the first mutation, even after preparation.
        for row in self.rows:
            if not _matches(_side(_path(self.target, row["path"])), self.before[row["path"]], exact_mode=True):
                raise ValueError(f"file-result target changed before apply: {row['path']}")
        for index, row in enumerate(self.rows):
            if not _matches(_side(_path(self.target, row["path"])), self.before[row["path"]], exact_mode=True):
                raise ValueError(f"file-result target changed during apply: {row['path']}")
            self._write(index, row["after"], "after")
            self.applied.append(index)

    def rollback(self):
        conflicts = []
        for index in reversed(self.applied):
            row = self.rows[index]
            if not _matches(_side(_path(self.target, row["path"])), row["after"], exact_mode=True):
                conflicts.append(row["path"])
                continue
            self._write(index, self.before[row["path"]], "before")
        self.applied.clear()
        if conflicts:
            raise ValueError(f"file-result rollback preserved concurrent changes: {conflicts}")


@contextmanager
def prepare_file_outputs(rows, target, *, baseline_sha="", file_baseline=None):
    """Prepare all bytes before apply. The caller owns Git staging and rollback decisions."""
    with tempfile.TemporaryDirectory(prefix="ouroboros-file-results-") as temporary:
        yield PreparedFileOutputs(rows, target, temporary, baseline_sha, file_baseline)


def _directory_direct_artifacts(
    root: Path, artifact_dir: Path, task: dict, existing: dict,
) -> tuple[list[dict], dict] | None:
    """Record direct folder effects without inventing a before-image or scanning the tree."""
    from ouroboros.headless_status import ARTIFACT_STATUS_MISSING, ARTIFACT_STATUS_READY
    from ouroboros.workspace_patch_capture import _acting_constraint_from_task, _preflight_head_from_task
    from ouroboros.utils import atomic_write_json, utc_now_iso
    from ouroboros.workspace_admission import has_git_metadata

    root = root.resolve(strict=False)
    meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    # Admitted while missing (#1799) and still absent: no tree to diff, and nothing remains there.
    absent = (not root.exists() and not root.is_symlink()
              and (meta.get("workspace_availability") or {}).get("state") == "missing")
    constraint = _acting_constraint_from_task(task)
    surface = str((task.get("task_constraint") or {}).get("surface") or "")
    if (not (root.is_dir() or absent) or surface in {"self_worktree", "genesis"}
            or (constraint and constraint.base_sha) or _preflight_head_from_task(task)
            or has_git_metadata(root)):
        return None
    from ouroboros.artifacts import (
        collect_task_artifact_records, merge_artifact_records, artifact_record, registered_task_artifact,
    )

    drive = artifact_dir.parents[2]
    captured = collect_task_artifact_records(drive, str(task["id"]))
    records = merge_artifact_records(existing.get("artifacts") or [], [
        registered_task_artifact(drive, str(task["id"]), item["name"]) or item for item in captured
    ])
    outputs = []
    for item in records:
        source = str(item.get("source_path") or "")
        if source and Path(source).resolve(strict=False).is_relative_to(root):
            outputs.append(dict(item))
    manifest = {
        "schema_version": 1, "workspace_root": str(root), "status": ARTIFACT_STATUS_READY,
        "capture_kind": "directory_direct", "apply_state": "already_applied",
        "before": "unknown", "complete": False, "evidence_extent": "registered_outputs_only",
        "registered_outputs": outputs, "created_at": utc_now_iso(),
        "note": "Direct effects remain in the selected folder. Registered outputs retain their captured "
                "bytes; other shell, GUI or external effects and the full changed-file set are unknown. "
                "No full rollback is available.",
    }
    if absent:
        manifest.update(status=ARTIFACT_STATUS_READY if outputs else ARTIFACT_STATUS_MISSING, before="absent",
                        note="The selected folder was missing when this work was admitted and is still absent, "
                             "so nothing remains there to capture. Registered outputs retain their captured "
                             "bytes; other shell, GUI or external effects are unknown.")
    path = artifact_dir / "workspace_patch.json"
    atomic_write_json(path, manifest, trailing_newline=True)
    return [*outputs, artifact_record(path, kind="workspace_patch_manifest")], manifest


def capture_known_workspace_outputs(ctx, workspace_root, paths, *, source_tool, include_preamble=True):
    """Retain known native file postimages in an ordinary folder, without inventorying it.

    Called only after successful writes. Capture cannot undo those effects, so a
    copy failure reports them as already applied and never requests a write retry.
    Git workspaces keep their existing patch capture; child artifacts use the
    child's own store and its existing result promotion to the canonical owner.
    """
    from ouroboros.artifacts import copy_file_to_task_artifacts, task_id_for_artifacts
    from ouroboros.tools.tool_resolution import active_repo_dir_for, system_repo_dir_for
    from ouroboros.workspace_admission import has_git_metadata

    root = Path(workspace_root).resolve(strict=False)
    if (not root.is_dir() or task_id_for_artifacts(ctx) == "interactive"
            or root == system_repo_dir_for(ctx).resolve(strict=False)
            or root != active_repo_dir_for(ctx).resolve(strict=False)
            or has_git_metadata(root)):
        return ""
    captured, unavailable, removed = [], [], []
    for relative in dict.fromkeys(paths):
        try:
            source = root / relative
            if not source.exists() and not source.is_symlink():
                removed.append(relative)
                continue
            record = copy_file_to_task_artifacts(ctx, source, kind="user_file")
            if not record:
                raise OSError("written file has no captured artifact")
            captured.append(f"{relative} (artifact_store:{record['name']}, {record['size']} bytes, sha256={record['sha256']})")
        except Exception as exc:
            unavailable.append(f"{relative}: {type(exc).__name__}: {exc}")
    notes = [f"{source_tool}: changes are already on disk in the selected folder; no separate patch apply is needed."] if include_preamble else []
    if captured:
        notes.append("Retained file outputs: " + "; ".join(captured))
    if removed:
        notes.append("Changed paths now absent have no file postimage to retain: " + ", ".join(removed))
    if unavailable:
        notes.append("⚠️ OUTPUT_CAPTURE_FAILED: " + "; ".join(unavailable)
                     + ". The writes remain applied; do not repeat them to retry artifact capture.")
    return "\n".join(notes)
