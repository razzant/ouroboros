"""Immutable named-evidence generations for ordinary delegated runs.

The custody row publishes the latest receipt only after a complete directory rename.
Earlier receipt paths (including legacy ``evidence/``) are never rewritten. The caller
holds the snapshot's disposition lock; only its unfinished staging directories may be
removed. A failed capture is a receipt too, keeping the source and its gaps visible.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import stat
import uuid

from ouroboros import delegate_custody as custody
from ouroboros.delegate_capture_attempt import POLICY, failure_retry
from ouroboros.delegate_start_instructions import EVIDENCE_DIRNAME
from ouroboros.utils import atomic_write_json


def linked(path):
    """Include Windows junction/reparse parents, which Path.is_symlink misses on 3.10."""
    info = pathlib.Path(path).lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def private_directory(path, anchor):
    """Create below an owned root without writing through an existing link or junction."""
    path, anchor = pathlib.Path(path), pathlib.Path(anchor)
    for relative in reversed((path.relative_to(anchor), *path.relative_to(anchor).parents)):
        if str(relative) == ".":
            continue
        directory = anchor / relative
        try:
            directory.mkdir()
        except FileExistsError:
            pass
        if linked(directory) or not directory.is_dir():
            raise OSError(f"evidence parent is not a private directory: {directory}")


def _raise(error):
    raise error  # a skipped subtree is not a complete inventory


def retain_files(area, paths, files_dir, *, preserve_mode=True):
    """Copy regular files byte-exact and whole into the evidence store."""
    from ouroboros import work_domain
    from ouroboros.review_checking import _open_regular, _sha

    if work_domain.is_work(area):
        return _retain_work_files(area, paths, files_dir, preserve_mode=preserve_mode)
    kept = []
    for relative in paths:
        source = area / relative
        if source.is_symlink() or not source.is_file():
            continue
        size = source.stat().st_size
        target = files_dir / relative
        private_directory(target.parent, files_dir.parent)
        # Fresh regular files: ordinary delegates keep writable copies; reviewers preserve modes. Streamed from
        # the descriptor that proved the file regular, and never a clone: the record compression refuses one.
        with _open_regular(source) as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        mode = stat.S_IMODE(source.stat().st_mode)
        if preserve_mode:
            target.chmod(mode)
        kept.append({"path": relative, "size": size, "sha256": _sha(target), "mode": mode})
    return kept


def _retain_work_files(area, paths, files_dir, *, preserve_mode=True):
    """``retain_files`` from a work copy: each regular file's bytes through the streamed confined intake
    (``work_domain.measure``), its mode from its own metadata, never a service open of its pathname."""
    from ouroboros import work_domain

    kept = []
    for relative in paths:
        source = area / relative
        if work_domain.kind(source) != "file":
            continue
        info = work_domain.lstat(source)
        target = files_dir / relative
        private_directory(target.parent, files_dir.parent)
        with target.open("xb") as dst:
            measured = work_domain.measure(source, dst)
        mode = stat.S_IMODE(info.st_mode)
        if preserve_mode:
            target.chmod(mode)
        kept.append({"path": relative, "size": measured["size"], "sha256": measured["sha256"], "mode": mode})
    return kept


def _work_evidence_rows(area, declared):
    """``(relative path, kind)`` of every entry under a work copy's evidence folder, by held descriptors; an
    unlistable folder raises, as the service walk's ``onerror``."""
    from ouroboros import work_domain

    for current, _dirs, _files in work_domain.walk(declared, strict=True):
        for name, kind in work_domain.entries(current):
            yield (pathlib.Path(current) / name).relative_to(area).as_posix(), kind


def named_evidence_witness(area):
    """Stat identity of the copy's evidence folder (paths, types, sizes, times); "" when it cannot be
    listed. A retained summary is reused only while it is unchanged. Not a byte proof: it assumes the
    run's processes stopped writing, so a same-size rewrite within one timestamp tick, or a write
    after the final check, goes unseen."""
    from ouroboros import work_domain

    rows = []
    if work_domain.is_work(area):  # the same witness from held descriptors
        try:
            declared = area / EVIDENCE_DIRNAME
            seen = work_domain.lstat(declared)
            rows.append(f"{EVIDENCE_DIRNAME}\0{seen.st_mode}\0{seen.st_size}\0{seen.st_mtime_ns}\0{seen.st_ctime_ns}")
            if not stat.S_ISDIR(seen.st_mode):
                return hashlib.sha256(rows[0].encode()).hexdigest()  # never walk a linked root
            for relative, _kind in _work_evidence_rows(area, declared):
                seen = work_domain.lstat(area / relative)
                rows.append(f"{relative}\0{seen.st_mode}\0{seen.st_size}\0{seen.st_mtime_ns}\0{seen.st_ctime_ns}")
        except OSError:
            return ""
        return hashlib.sha256("\n".join(sorted(rows)).encode("utf-8", "surrogateescape")).hexdigest()
    try:
        declared = area / EVIDENCE_DIRNAME
        seen = declared.lstat()
        rows.append(f"{EVIDENCE_DIRNAME}\0{seen.st_mode}\0{seen.st_size}\0{seen.st_mtime_ns}\0{seen.st_ctime_ns}")
        if declared.is_symlink() or not declared.is_dir():
            return hashlib.sha256(rows[0].encode()).hexdigest()  # never walk a linked root
        for current, dirs, files in os.walk(declared, onerror=_raise, followlinks=False):
            for name in sorted(dirs + files):
                seen = (pathlib.Path(current) / name).lstat()
                rows.append(f"{(pathlib.Path(current) / name).relative_to(area).as_posix()}\0{seen.st_mode}\0"
                            f"{seen.st_size}\0{seen.st_mtime_ns}\0{seen.st_ctime_ns}")
    except OSError:
        return ""
    return hashlib.sha256("\n".join(sorted(rows)).encode("utf-8", "surrogateescape")).hexdigest()


def retain_named_evidence(area, files_dir, gaps, *, product_paths=(), preserve_mode=True):
    """Retain a copy's ``.review-evidence/`` whole, independently of product-patch filters.

    Links are not dereferenced: a linked output is a named gap, never a receipt claiming
    its target's bytes were retained. So is an entry that is neither a file nor a folder;
    a folder that cannot be listed raises. Baseline-owned ``product_paths`` belong to
    capture, including symlinks; never retain them here.
    """
    from ouroboros import work_domain

    declared = area / EVIDENCE_DIRNAME
    if work_domain.is_work(area):  # held descriptors only: a link is a named gap, never followed
        what = work_domain.kind(declared)
        if EVIDENCE_DIRNAME in product_paths and what != "dir":
            return []
        if what == "link":
            gaps.append({"path": EVIDENCE_DIRNAME, "reason": "evidence_symlink"})
            return []
        if what not in ("dir", "missing"):
            gaps.append({"path": EVIDENCE_DIRNAME, "reason": "evidence_root_not_directory"})
            return []
        paths = []
        for relative, kind in (_work_evidence_rows(area, declared) if what == "dir" else ()):
            if relative in product_paths:
                continue
            if kind == "link":
                gaps.append({"path": relative, "reason": "evidence_symlink"})
            elif kind == "file":
                paths.append(relative)
            elif kind == "other":
                gaps.append({"path": relative, "reason": "unsupported_entry_type"})
        return retain_files(area, sorted(paths), files_dir, preserve_mode=preserve_mode)
    if EVIDENCE_DIRNAME in product_paths and (declared.is_symlink() or not declared.is_dir()):
        return []
    if declared.is_symlink():
        gaps.append({"path": EVIDENCE_DIRNAME, "reason": "evidence_symlink"})
        return []
    if declared.exists() and not declared.is_dir():
        gaps.append({"path": EVIDENCE_DIRNAME, "reason": "evidence_root_not_directory"})
        return []
    paths = []
    if declared.is_dir():
        for current, dirs, files in os.walk(declared, onerror=_raise, followlinks=False):
            for name in sorted(dirs + files):
                path = pathlib.Path(current) / name
                relative = path.relative_to(area).as_posix()
                if relative in product_paths:
                    continue
                if path.is_symlink():
                    gaps.append({"path": relative, "reason": "evidence_symlink"})
                elif path.is_file():
                    paths.append(relative)
                elif name in files:
                    gaps.append({"path": relative, "reason": "unsupported_entry_type"})
    return retain_files(area, sorted(paths), files_dir, preserve_mode=preserve_mode)


def retain_workspace_evidence(area, destination, *, product_paths=(), previous=None):
    """Retain a copy's evidence folder ONCE for a capture no run retains it for (a task's own workspace at
    finalization, #2711): ``destination/files/`` byte-exact through ``retain_named_evidence``, beside a
    ``summary.json`` carrying the folder's witness. ``previous`` is the last published summary of the same
    workspace: an unchanged folder (same witness, files still there) is not copied again and that summary
    stands, never a second copy; a changed one replaces this capture's own earlier generation whole. A
    raised copier leaves no partial files and a ``capture_failed`` summary: the caller then keeps the
    folder's files in its own capture, so the bytes are never zero copies either."""
    destination = pathlib.Path(destination)
    witness = named_evidence_witness(area)
    if (isinstance(previous, dict) and previous.get("status") == "retained" and witness
            and previous.get("witness") == witness and pathlib.Path(str(previous.get("evidence_dir") or "")).is_dir()):
        return previous
    staging = destination.parent / f".{destination.name}-{uuid.uuid4().hex}.pending"
    gaps = []
    try:
        private_directory(staging, destination.parent) if destination.parent.is_dir() else staging.mkdir(parents=True)
        files = retain_named_evidence(area, staging / "files", gaps, product_paths=product_paths)
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        return {"status": "capture_failed", "files": [], "gaps": gaps, "error": f"{type(exc).__name__}: {exc}",
                "witness": witness, "evidence_dir": str(destination)}
    summary = {"status": "retained", "files": files, "gaps": gaps, "witness": witness,
               "evidence_dir": str(destination), "summary_path": str(destination / "summary.json")}
    atomic_write_json(staging / "summary.json", summary)
    if destination.exists():  # only this capture's own earlier generation of the same workspace
        shutil.rmtree(destination, ignore_errors=True)
    staging.rename(destination)
    return summary


def published_workspace_evidence(result):
    """The last published summary of a task workspace's retained evidence folder (its ``named_evidence_manifest``
    artifact row), so a later finalization copies an unchanged folder no second time; None when there is none or
    it cannot be read."""
    row = next((item for item in reversed(result.get("artifacts") or [])
                if isinstance(item, dict) and item.get("kind") == "named_evidence_manifest" and item.get("path")), None)
    if row is None:
        return None
    try:
        saved = json.loads(pathlib.Path(str(row["path"])).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return saved if isinstance(saved, dict) else None


def saved_named_evidence(root, entry, *, observe=True):
    home = custody.delegated_capture_dir(root, entry.task_id, custody.capture_key(entry))
    name = str(entry.named_evidence.get("directory") or "evidence")
    # Only the writer's local directory name is accepted, never a stored absolute parent.
    valid_name = pathlib.Path(name).name == name and name.startswith("evidence")
    path = home / (name if valid_name else "evidence") / "summary.json"
    try:
        saved = json.loads(path.read_text(encoding="utf-8")) if valid_name else {}
    except (OSError, ValueError):
        saved = {}
    if not isinstance(saved, dict):
        saved = {}
    if observe and not saved and entry.named_evidence and entry.named_evidence.get("status") != "unreadable":
        # A consumer discovered a lost receipt, a new physical fact. It requests no
        # recapture itself; the body's ordinary due rule sees the changed custody.
        outcome = {**entry.named_evidence, "status": "unreadable"}
        custody.emit(root, custody.NAMED_EVIDENCE_RETAINED,
                     {"run_id": entry.run_id, "task_id": entry.task_id, "outcome": outcome})
        entry.named_evidence = outcome
    return path, saved


def prune_abandoned_generations(home):
    """Under the disposition lock, remove only unpublished staging; never receipt directories.

    Copies keep source modes in their manifests, not filesystem attributes, so Windows
    cleanup needs no chmod. A busy handle leaves staging for the next addressed capture.
    """
    for path in pathlib.Path(home).glob(".evidence-*.pending"):
        try:
            if linked(path):
                # A junction is a directory reparse point: unlink refuses it on Windows.
                if stat.S_ISDIR(path.lstat().st_mode):
                    path.rmdir()
                else:
                    path.unlink()
            else:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass  # A contended orphan never blocks publication of a fresh generation.


def _publish(root, entry, name, summary):
    outcome = {"directory": name, "status": summary["status"], "inputs": summary["inputs"],
               "retry": summary.get("retry", "")}
    if entry.named_evidence != outcome and not custody.emit(root, custody.NAMED_EVIDENCE_RETAINED,
            {"run_id": entry.run_id, "task_id": entry.task_id, "outcome": outcome}):
        raise OSError("named-evidence receipt publication failed")
    entry.named_evidence = outcome


def _selection(entry, area, whole):
    if not whole:
        return set()
    from ouroboros.subagent_worktrees import find_execution_snapshot
    from ouroboros.workspace_patch_capture import baseline_blob_sizes

    snapshot = find_execution_snapshot(entry.snapshot_id) or {}
    clone = pathlib.Path(str(snapshot.get("path") or area))
    prefix = pathlib.PurePosixPath(area.resolve().relative_to(clone.resolve()).as_posix())
    errors = []
    baseline = baseline_blob_sizes(clone, entry.baseline_sha, errors=errors)
    if errors:
        raise ValueError(f"evidence baseline could not be read: {errors}")
    paths = map(pathlib.PurePosixPath, set(baseline) | set(snapshot.get("file_baseline") or {}))
    return {path.relative_to(prefix).as_posix() for path in paths if path.is_relative_to(prefix)}


def retain_named(root, entry, *, whole=False, recapture=False, reread=False, publish=True):
    """Observe inputs on an addressed request; reuse partial as well as complete receipts.

    No BodyRuntime tick calls the witness walker. A new terminal observation, explicit
    recapture, or this foreground retention boundary is what observes changed sources.
    """
    root = pathlib.Path(root).resolve()
    home = custody.delegated_capture_dir(root, entry.task_id, custody.capture_key(entry))
    previous, known = saved_named_evidence(root, entry, observe=publish)
    area = pathlib.Path(str(entry.execution_root or ""))
    if reread or not entry.execution_root or not area.is_dir():
        return known or {"status": "copy_missing", "files": [], "gaps": []}
    source = {"policy": POLICY, "limit_bytes": None,
              "whole": whole, "baseline_sha": entry.baseline_sha,
              "terminal_sha": entry.terminal_source.get("sha256"), "invocation_id": entry.invocation_id,
              "snapshot_id": entry.snapshot_id, "capture_id": entry.capture_id,
              "witness": named_evidence_witness(area)}
    products, error = set(), None
    try:
        products = _selection(entry, area, whole)
        source["product_selection"] = hashlib.sha256(json.dumps(sorted(products)).encode()).hexdigest()
    except Exception as exc:
        error = exc
    if not recapture and error is None and known.get("inputs") == source:
        if publish:
            _publish(root, entry, previous.parent.name, known)
        return known
    private_directory(home, root)
    prune_abandoned_generations(home)
    token = uuid.uuid4().hex
    staging = home / f".evidence-{token}.pending"
    # Keep the first legacy-shaped address usable, but never reuse any existing destination.
    name = "evidence" if not os.path.lexists(home / "evidence") else f"evidence-{token}"
    final = home / name
    staging.mkdir()
    files, gaps = [], []
    try:
        if error is not None:
            raise error
        files = retain_named_evidence(area, staging / "files", gaps, product_paths=products, preserve_mode=False)
        # Links and special entries are recorded, never followed: nothing more can be
        # retained, so they do not hold the copy.
        summary = {"status": "retained", "files": files, "gaps": gaps}
    except Exception as exc:
        # A raised copier has no complete inventory. Keep the source, not unlisted
        # partial copies in every immutable retry generation.
        if (staging / "files").exists():
            shutil.rmtree(staging / "files", ignore_errors=True)
            if (staging / "files").exists():
                # A busy handle leaves bytes to the staging remover, never in a published receipt.
                staging = home / f".evidence-{uuid.uuid4().hex}.pending"
                staging.mkdir()
        summary = {"status": "capture_failed", "files": [], "gaps": gaps,
                   "error": f"{type(exc).__name__}: {exc}",
                   "retry": failure_retry(exc)}
    summary.update(evidence_dir=str(final), inputs=source, witness=source["witness"],
        summary_read={"root": "artifact_store", "path": f"delegated_runs/{home.name}/{name}/summary.json"})
    try:
        atomic_write_json(staging / "summary.json", summary)
        staging.rename(final)  # only a never-published, unique destination
        if publish:
            _publish(root, entry, name, summary)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def run_named_evidence(root, entry, *, recapture=False, publish=True):
    from ouroboros.delegate_directory import is_directory_run
    from ouroboros.delegate_registration_policy import candidate_apply_possible

    if (not (entry.snapshot_id and entry.execution_root) or is_directory_run(entry)
            or entry.authority_source == "skill_payload"):
        return {}
    return retain_named(root, entry, whole=candidate_apply_possible(entry), recapture=recapture,
                        reread=bool(entry.patch_disposed or entry.superseded_by), publish=publish)
