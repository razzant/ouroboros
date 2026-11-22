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
import threading
import uuid
from contextlib import contextmanager

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


@contextmanager
def _killed_after(seconds, process):
    deadline = threading.Timer(seconds, process.kill)
    deadline.start()
    try:
        yield
    finally:
        deadline.cancel()


def private_review_paths(area, trees, env):
    """Scan the immutable old and new sides of a review delta once per blob, through one Git reader."""
    import subprocess
    from ouroboros.review_checking import CheckingAreaUnavailable, _git
    from ouroboros.secret_masking import SecretScan, installation_secret_bytes

    secrets = installation_secret_bytes()
    if not secrets:
        return {}
    raw = _git(area, "diff", "--no-renames", "--raw", "--no-abbrev", "-z", *trees, env=env).stdout.split(b"\0")
    names = {}
    for change, path in zip(raw[::2], raw[1::2]):
        old_mode, new_mode, old, new, _ = change.lstrip(b":").split()
        for mode, oid in ((old_mode, old), (new_mode, new)):
            if mode not in {b"000000", b"160000"}:
                names.setdefault(oid, set()).add(os.fsdecode(path))
    gaps = {}
    if not names:
        return gaps
    # Request one object at a time. Neither pipe can fill while the parent is writing the list.
    # The reader keeps _git's 600 s bound: a stuck Git is killed, and its short read fails the capture.
    with subprocess.Popen(["git", "cat-file", "--batch"], cwd=area, env=env,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as git, _killed_after(600, git):
        for oid, paths in names.items():
            git.stdin.write(oid + b"\n")
            git.stdin.flush()
            header = git.stdout.readline().split()
            if len(header) != 3 or header[:2] != [oid, b"blob"]:
                raise CheckingAreaUnavailable("Git could not read an evidence blob")
            size = int(header[2])
            scan = SecretScan(secrets)
            while scan.size < size:
                chunk = git.stdout.read(min(1 << 20, size - scan.size))
                if not chunk:
                    raise CheckingAreaUnavailable("Git truncated an evidence blob")
                scan.write(chunk)
            if git.stdout.read(1) != b"\n":
                raise CheckingAreaUnavailable("Git did not finish an evidence blob")
            if scan.found:
                for path in paths:
                    gaps[path] = {"path": path, "reason": "installation_settings", "size": size}
        git.stdin.close()
        if git.wait():
            raise CheckingAreaUnavailable("Git failed to read the evidence blobs")
    return gaps


def _private_evidence_paths(root, paths):
    """Only named evidence returning through the patch fallback still needs the evidence floor."""
    from ouroboros import work_domain
    from ouroboros.workspace_patch_rules import _patch_exclude_reason
    from ouroboros.review_checking import _open_regular
    from ouroboros.secret_masking import SecretScan, installation_secret_bytes

    values = installation_secret_bytes()
    private = {}
    for rel in sorted(paths) if values else ():
        if _patch_exclude_reason(rel):
            continue
        path = root / rel
        scan = SecretScan(values)
        kind = work_domain.kind(path)
        if kind == "file":
            if work_domain.is_work(path):
                work_domain.measure(path, scan, strict=True)
            else:
                with _open_regular(path) as handle:
                    scan.read_from(handle)
        elif kind == "link":
            scan.write(os.fsencode(work_domain.readlink(path)))
        if scan.found:
            private[rel] = {"path": rel, "reason": "installation_settings", "size": scan.size}
    return private


def _held_named_evidence(root, host_evidence, paths, products, destination, previous):
    """Retain new named evidence. A failed capture returns public files to the patch."""
    prefix = pathlib.PurePosixPath(EVIDENCE_DIRNAME if destination is not None else (host_evidence or EVIDENCE_DIRNAME).strip("/"))
    held = {name for name in paths - products if pathlib.PurePosixPath(name).is_relative_to(prefix)}
    if not held or destination is None:
        return held, None
    summary = retain_workspace_evidence(root, destination, previous=previous, product_paths=products)
    if summary["status"] == "retained":
        return held, summary
    private = _private_evidence_paths(root, held)
    seen = {gap["path"] for gap in summary["gaps"]}
    summary["gaps"].extend(gap for name, gap in private.items() if name not in seen)
    return set(private), summary


def workspace_evidence_exclusions(root, host_evidence, paths, products, destination, previous):
    """Keep retained evidence out of the product patch and carry its exact exclusions into the receipt."""
    from ouroboros.workspace_patch_rules import NAMED_EVIDENCE_EXCLUSION

    held, summary = _held_named_evidence(root, host_evidence, paths, products,
                                        destination, previous)
    excluded = {name: {"path": name, "reason": NAMED_EVIDENCE_EXCLUSION} for name in held}
    excluded.update((gap["path"], gap) for gap in (summary or {}).get("gaps", []) if gap["reason"] == "installation_settings")
    return excluded, summary


def retain_files(area, paths, files_dir, *, gaps, preserve_mode=True):
    """Stream regular evidence and inspect the exact copy before publishing its receipt."""
    from ouroboros import work_domain
    from ouroboros.review_checking import _open_regular, _sha

    from ouroboros.secret_masking import SecretScan, installation_secret_bytes
    secrets = installation_secret_bytes()
    work = work_domain.is_work(area)
    kept = []
    for relative in paths:
        source, target = area / relative, files_dir / relative
        regular = work_domain.kind(source) == "file" if work else not source.is_symlink() and source.is_file()
        if not regular:
            continue
        info = work_domain.lstat(source) if work else source.stat()
        private_directory(target.parent, files_dir.parent)
        with target.open("xb") as dst:
            scan = SecretScan(secrets, dst)
            if work:
                measured = work_domain.measure(source, scan, strict=True)
            else:
                with _open_regular(source) as src:
                    shutil.copyfileobj(src, scan, 1 << 20)
        if scan.found:
            target.unlink()
            gaps.append({"path": relative, "reason": "installation_settings", "size": scan.size})
            continue
        if not work:
            measured = {"size": scan.size, "sha256": _sha(target)}
        mode = stat.S_IMODE(info.st_mode)
        if preserve_mode:
            target.chmod(mode)
        kept.append({"path": relative, **measured, "mode": mode})
    return kept


def _evidence_entries(area, declared, work):
    """Relative names and kinds below the evidence directory, without following links."""
    from ouroboros import work_domain

    if work:
        for folder, _, _ in work_domain.walk(declared, strict=True):
            for name, kind in work_domain.entries(folder):
                yield (pathlib.Path(folder) / name).relative_to(area).as_posix(), kind
    else:
        for folder, dirs, files in os.walk(declared, onerror=_raise, followlinks=False):
            for name in sorted(dirs + files):
                path = pathlib.Path(folder) / name
                kind = "link" if path.is_symlink() else "file" if path.is_file() else "dir" if name in dirs else "other"
                yield path.relative_to(area).as_posix(), kind


def named_evidence_witness(area):
    """Stat identity of the evidence tree after its writer stopped. An unreadable tree has no witness."""
    from ouroboros import work_domain

    declared = area / EVIDENCE_DIRNAME
    work = work_domain.is_work(area)
    observe = work_domain.lstat if work else os.lstat

    def stamp(name):
        seen = observe(area / name)
        return f"{name}\0{seen.st_mode}\0{seen.st_size}\0{seen.st_mtime_ns}\0{seen.st_ctime_ns}"

    try:
        rows = [stamp(EVIDENCE_DIRNAME)]
        if stat.S_ISDIR(observe(declared).st_mode):
            rows.extend(stamp(name) for name, _ in _evidence_entries(area, declared, work))
    except OSError:
        return ""
    return hashlib.sha256("\n".join(sorted(rows)).encode("utf-8", "surrogateescape")).hexdigest()


def retain_named_evidence(area, files_dir, gaps, *, product_paths=(), preserve_mode=True):
    """Keep the named regular files independently of product filters. A link or special file is a named gap."""
    from ouroboros import work_domain

    declared = area / EVIDENCE_DIRNAME
    work = work_domain.is_work(area)
    try:
        info = work_domain.lstat(declared) if work else declared.lstat()
    except FileNotFoundError:
        return []
    if not stat.S_ISDIR(info.st_mode):
        if EVIDENCE_DIRNAME not in product_paths:
            reason = "evidence_symlink" if stat.S_ISLNK(info.st_mode) else "evidence_root_not_directory"
            gaps.append({"path": EVIDENCE_DIRNAME, "reason": reason})
        return []
    paths = []
    for relative, kind in _evidence_entries(area, declared, work):
        if relative in product_paths:
            continue
        if kind == "file":
            paths.append(relative)
        elif kind in {"other", "link"}:
            gaps.append({"path": relative, "reason": "evidence_symlink" if kind == "link" else "unsupported_entry_type"})
    return retain_files(area, sorted(paths), files_dir, gaps=gaps, preserve_mode=preserve_mode)


def retain_workspace_evidence(area, destination, *, product_paths=(), previous=None):
    """Retain a copy's evidence folder ONCE for a capture no run retains it for (a task's own workspace at
    finalization, #2711): ``destination/files/`` byte-exact through ``retain_named_evidence``, beside a
    ``summary.json`` carrying the folder's witness (``named_evidence_witness``: its stat identity, so an
    unchanged generation is reused, never recopied). ``previous`` is the last published summary of the same
    workspace: an unchanged folder (same witness, files still there) is not copied again and that summary
    stands, never a second copy; a changed one replaces this capture's own earlier generation whole. A
    Storage release that protects the previous generation's bytes, or this destination, raises before any
    copy: already released bytes are never republished. A raised copier, or a folder that moved during the
    copy, leaves no partial files and a ``capture_failed`` summary: the caller then keeps the folder's files
    in its own capture, so the bytes are never zero copies either."""
    from ouroboros.payload_release_state import states_under

    destination = pathlib.Path(destination)
    witness = named_evidence_witness(area)
    if isinstance(previous, dict) and previous.get("evidence_dir"):
        summary = said_released(previous)
        if summary.get("reason"):
            raise OSError(summary["reason"])
        if (previous.get("status") == "retained" and witness and previous.get("witness") == witness
                and pathlib.Path(previous["evidence_dir"]).is_dir()):
            return summary
        if summary.get("released"):
            raise OSError("a Storage choice protects the previous evidence generation")
    if states_under(destination, include_retained=True):
        raise OSError("a Storage choice protects this evidence generation")
    staging = destination.parent / f".{destination.name}-{uuid.uuid4().hex}.pending"
    gaps = []
    try:
        private_directory(staging, destination.parent) if destination.parent.is_dir() else staging.mkdir(parents=True)
        files = retain_named_evidence(area, staging / "files", gaps, product_paths=product_paths)
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        return {"status": "capture_failed", "files": [], "gaps": gaps, "error": f"{type(exc).__name__}: {exc}",
                "witness": witness, "evidence_dir": str(destination)}
    summary = {"status": "retained", "files": files, "retained_bytes": sum(f["size"] for f in files), "gaps": gaps, "witness": witness,
               "evidence_dir": str(destination), "summary_path": str(destination / "summary.json")}
    atomic_write_json(staging / "summary.json", summary)
    try:
        # The task finalizer holds the canonical custody lock across this replacement.
        if destination.exists():
            shutil.rmtree(destination)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return summary


def said_released(summary, *, directory=None):
    """Project one retained generation from its addressed state. An uncertain read exposes no stale file
    refs and is never an input witness or a custody receipt. The immutable source summary stays intact."""
    from ouroboros.payload_release_state import KEY, PublicationReleases, states_under

    base = directory or summary.get("evidence_dir")
    if not base:
        return summary
    try:
        states = states_under(base)
        if any(receipt is None for receipt in states.values()):
            return {"status": "partial", "reason": "payload_release_pending", "files": []}
        if not states:
            return summary
        names = PublicationReleases({KEY: {"released": states}}, pathlib.Path(base))
        kept = [row for row in summary.get("files") or () if names.address(f"files/{_file_path(row)}") not in states]
    except Exception as exc:
        return {"status": "partial", "reason": "release_state_unavailable", "error": type(exc).__name__, "files": []}
    return {**summary, "files": kept, "released": sorted(states)}


def _file_path(row):
    return str(row.get("path") or row.get("relpath") or "") if isinstance(row, dict) else str(row or "")


def published_workspace_evidence(result):
    """The last published summary of a task workspace's retained evidence folder (its ``named_evidence_manifest``
    artifact row), so a later finalization reuses an unchanged folder. An unreadable summary still supplies
    its published directory for the Storage check. No manifest row yields None."""
    row = next((item for item in reversed(result.get("artifacts") or [])
                if isinstance(item, dict) and item.get("kind") == "named_evidence_manifest" and item.get("path")), None)
    if row is None:
        return None
    try:
        saved = json.loads(pathlib.Path(str(row["path"])).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = None
    # The published row still identifies the prior directory. Check its Storage choice before a recapture.
    return {**(saved if isinstance(saved, dict) else {}), "evidence_dir": str(pathlib.Path(row["path"]).parent)}


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
    view = said_released(saved, directory=path.parent)
    if view.get("reason") or (not saved and view.get("released")):
        return path, view if view.get("reason") else {"status": "partial", "reason": "released_summary_unavailable", "files": []}
    if observe and not saved and entry.named_evidence and entry.named_evidence.get("status") != "unreadable":
        # A consumer discovered a lost receipt, a new physical fact. It requests no
        # recapture itself; the body's ordinary due rule sees the changed custody.
        outcome = {**entry.named_evidence, "status": "unreadable"}
        custody.emit(root, custody.NAMED_EVIDENCE_RETAINED,
                     {"run_id": entry.run_id, "task_id": entry.task_id, "outcome": outcome})
        entry.named_evidence = outcome
    files = view.get("files")
    if isinstance(files, list) and all(isinstance(f, dict) and type(f.get("size")) is int for f in files):
        view["retained_bytes"] = sum(f["size"] for f in files)
    return path, view


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
    """The custody receipt of the generation NAME as it stands: ``released`` counts the files the owner released
    from it in Storage (``said_released``), so a reuse never re-asserts them retained."""
    outcome = {"directory": name, "status": summary["status"], "inputs": summary["inputs"],
               "retry": summary.get("retry", ""), **({"released": len(summary["released"])} if summary.get("released") else {})}
    if entry.named_evidence != outcome and not custody.emit(root, custody.NAMED_EVIDENCE_RETAINED,
            {"run_id": entry.run_id, "task_id": entry.task_id, "outcome": outcome}):
        raise OSError("named-evidence receipt publication failed")
    entry.named_evidence = outcome


def publish_saved(root, entry, summary, folder, *, custody_held=None):
    """Re-read the generation under task custody immediately before publishing its named refs."""
    from ouroboros.payload_release_state import publication_lock
    from ouroboros.task_custody import CustodyBusy

    try:
        with publication_lock(folder, custody_held=custody_held) as locked:
            if not locked:
                raise CustodyBusy("custody_busy")
            view = said_released(summary, directory=folder)
            if not view.get("reason"):
                _publish(root, entry, folder.name, view)
            return view
    except CustodyBusy as exc:
        return {"status": "partial", "reason": "release_state_unavailable", "error": str(exc), "files": []}


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


def retain_named(root, entry, *, whole=False, recapture=False, reread=False, publish=True, custody_held=None):
    """Observe inputs on an addressed request; reuse partial as well as complete receipts.

    No BodyRuntime tick calls the witness walker. A new terminal observation, explicit
    recapture, or this foreground retention boundary is what observes changed sources.
    """
    from ouroboros.payload_release_state import publication_lock
    from ouroboros.task_custody import CustodyBusy

    root = pathlib.Path(root).resolve()
    home = custody.delegated_capture_dir(root, entry.task_id, custody.capture_key(entry))
    previous, known = saved_named_evidence(root, entry, observe=publish)
    area = pathlib.Path(str(entry.execution_root or ""))
    if known.get("reason") or reread or not entry.execution_root or not area.is_dir():
        return known or {"status": "copy_missing", "files": [], "gaps": []}
    if known.get("released"):
        witness_now = named_evidence_witness(area)
        if not witness_now:
            return {**known, "status": "partial", "reason": "released_generation_unreadable"}
        if (known.get("inputs") or {}).get("witness") != witness_now:
            return {**known, "status": "partial", "reason": "released_generation_changed"}
        return publish_saved(root, entry, known, previous.parent, custody_held=custody_held) if publish else known
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
    if not recapture and error is None and source["witness"] and known.get("inputs") == source:
        return publish_saved(root, entry, known, previous.parent, custody_held=custody_held) if publish else known
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
        summary = {"status": "retained", "files": files, "retained_bytes": sum(f["size"] for f in files), "gaps": gaps}
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
        with publication_lock(final, custody_held=custody_held) as locked:
            if not locked:
                return {"status": "partial", "reason": "release_state_unavailable", "error": "custody_busy", "files": []}
            _, latest = saved_named_evidence(root, entry, observe=False)
            if latest.get("reason") or latest.get("released"):
                return latest
            staging.rename(final)  # only a never-published, unique destination
            if publish:
                _publish(root, entry, name, summary)
        return summary
    except CustodyBusy as exc:
        return {"status": "partial", "reason": "release_state_unavailable", "error": str(exc), "files": []}
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def run_named_evidence(root, entry, *, recapture=False, publish=True, custody_held=None):
    from ouroboros.delegate_directory import is_directory_run
    from ouroboros.delegate_registration_policy import candidate_apply_possible

    if (not (entry.snapshot_id and entry.execution_root) or is_directory_run(entry)
            or entry.authority_source == "skill_payload"):
        return {}
    return retain_named(root, entry, whole=candidate_apply_possible(entry), recapture=recapture,
                        reread=bool(entry.patch_disposed or entry.superseded_by), publish=publish,
                        custody_held=custody_held)
