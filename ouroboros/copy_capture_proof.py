"""May a finished plain copy go? Its work must already exist elsewhere: nothing beyond its baseline in
the checkout, or a saved capture holding each such path's exact bytes and mode: its patch, or the file
results beside it (binary and large files), read back whole. A path's name proves
nothing here: the patch rules that leave out ``.venv``, ``.cache`` or ``*.tmp`` decide transport, and no
producer declared those folders disposable, so they count as unique. No capture is made and no
ref is pinned: unproven work simply keeps its copy until the owner's explicit discard, which this module
judges and carries out through the record lock and keep gate of ``subagent_worktrees._retire_worktree``. Otherwise read-only over the checkout and its
repository (the check's index and objects live in a temporary folder)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import stat
import zipfile
import tempfile
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

CAPTURE_UNPROVEN = "capture_unproven"
_SYMLINK, _EXECUTABLE, _REGULAR = "120000", "100755", "100644"


def plain_copy_verdict(row: Dict[str, Any], data_dir: Optional[Any] = None) -> Dict[str, Any]:
    """``{"remove", "keep_branch", "reason"}``, the shape of ``body_candidate.retention_verdict``.
    Anything that could not be read counts as unique: unknown never means empty."""
    path = pathlib.Path(str(row.get("path") or ""))
    base = str(row.get("base_sha") or "")
    if not path.is_dir():
        # Nothing on disk to lose; only commits on its branch can be unique.
        return {"remove": True, "keep_branch": _branch_may_hold_commits(row, base), "reason": "checkout_missing"}
    keep = {"remove": False, "keep_branch": True, "reason": CAPTURE_UNPROVEN}
    try:
        from ouroboros.body_candidate import checkout_work

        work = checkout_work(path)
        if not base or work["tip"] != base:
            return keep  # a moved HEAD carries commits no patch of this copy holds
        unique = [*work["tracked"], *work["untracked"], *work["ignored"]]
        if not unique:
            return {"remove": True, "keep_branch": False, "reason": "no_unique_work"}
        if not _unsaved(path, row, unique, base, data_dir):
            return {"remove": True, "keep_branch": False, "reason": "capture_holds"}
    except Exception:
        log.debug("Work of copy %s could not be proven saved", path, exc_info=True)
    return keep


def _discard_refusal(row: Dict[str, Any], path: pathlib.Path, data_dir: Optional[Any], live: Any, root: pathlib.Path) -> str:
    """Why this finished copy's owner may not discard it now, ``""`` when it may go. Unknown refuses."""
    from ouroboros import body_candidate, owner_keep, owner_pause, subagent_worktrees as sw, worktree_removal
    from ouroboros.delegate_custody import custody_log_unreadable
    from ouroboros.delegate_custody_current import held_runs
    from ouroboros.task_results import load_task_result
    from ouroboros.test_environment import retention_markers

    task_id, data = str(row.get("task_id") or ""), sw._data_dir(data_dir)
    if row.get("kind") == sw._KIND_DELEGATED_EXEC:
        return "custody_held"  # custody owns a snapshot and its result
    if not path.is_dir() or not sw._deletable(path, root) or worktree_removal.unchanged(row):
        return "not_removable"
    if not sw._task_proven_over(task_id, data, live):
        return "not_proven_over"
    if owner_keep.is_kept(data, task_id):
        return owner_keep.OWNER_KEEP
    try:
        document = load_task_result(data, task_id, strict=True) or {}
        tree = {task_id, str(document.get("root_task_id") or (document.get("metadata") or {}).get("root_task_id") or task_id)}
        if any(owner_pause.fence_closed(owner_pause.read_fence(data, member)) for member in tree):
            return "owner_paused"
        if custody_log_unreadable(data):
            return "record_unreadable"
        for run in held_runs(data).values():
            places = [str(getattr(run, name, "") or "") for name in ("execution_root", "target_root")]
            if task_id in (run.task_id, getattr(run, "snapshot_task_id", "")) or any(
                    place and sw._is_within(pathlib.Path(place), path) for place in places):
                return "custody_held"
    except Exception:
        log.debug("Discard of %s refused: its task's facts are unreadable", path, exc_info=True)
        return "record_unreadable"
    if body_candidate._retained_scratch(row, classify=True, record_witness=False) or retention_markers(path):
        return "retained_process_environment"
    return ""


def discard_offer(row: Dict[str, Any], data_dir: Optional[Any], live: Any, root: pathlib.Path) -> Dict[str, Any]:
    """What the owner's explicit discard of this copy would lose, or why it may not go: ``{"refusal"}``, and
    with none ``unsaved`` (path, bytes; largest first), ``unsaved_bytes``, ``keep_branch`` and ``signature``.
    "Unsaved" here means not proven saved, never proven the only copy. A directory among the unsaved paths (a nested repository) has a history no list can show, and commits no
    ref that stays would hold are lost for good: neither is offered. The signature's stat covers ctime, so any
    write since the preview changes it without hashing a large folder twice."""
    from ouroboros import body_candidate, subagent_worktrees as sw

    path = pathlib.Path(str(row.get("path") or ""))
    refusal = _discard_refusal(row, path, data_dir, live, root)
    if refusal:
        return {"refusal": refusal}
    keep_branch, commits_alone = False, False
    try:
        work = body_candidate.checkout_work(path)
        if row.get("kind") == sw._KIND_BODY_CANDIDATE:
            unique = body_candidate.unique_work(row)
            unsaved = unique.get("excluded") or []  # the rest goes on its pin first (``retention_verdict``)
            keep_branch = body_candidate.has_unique_work(unique)  # that verdict's own rule: no other work, no pin, the branch goes
        else:
            base = str(row.get("base_sha") or "")
            unsaved = _unsaved_paths(path, row, work, base, data_dir)
            keep_branch = _keeps_commits(row, work["tip"], base)
            if keep_branch is None:
                return {"refusal": "commits_unprotected"}
            commits_alone = work["tip"] != base  # no path is lost and the branch that holds them stays
        if not unsaved and not commits_alone:
            return {"refusal": "nothing_unsaved"}
        sizes = {}
        for rel in unsaved:
            seen_now = _stat_of(path / rel)
            if seen_now is not None and stat.S_ISDIR(seen_now[0]):
                return {"refusal": "nested_repository"}
            sizes[rel] = seen_now[1] if seen_now is not None else 0  # a deleted tracked file: its deletion is what is lost
        index = sw._git(path, "rev-parse", "--path-format=absolute", "--git-path", "index").stdout.strip()
        signature = hashlib.sha256(json.dumps([work["tip"], _stat_of(path), _stat_of(pathlib.Path(index)), [
            (rel, _stat_of(path / rel)) for rel in sorted({*work["tracked"], *work["untracked"], *work["ignored"]},
                                                           key=os.fsencode)]], ensure_ascii=True).encode()).hexdigest()
    except Exception:
        log.debug("Discard of %s refused: its work could not be read", path, exc_info=True)
        return {"refusal": "record_unreadable"}
    ranked = sorted(sizes.items(), key=lambda pair: (-pair[1], pair[0]))
    return {"refusal": "", "unsaved": ranked, "unsaved_bytes": sum(sizes.values()), "keep_branch": keep_branch,
            "signature": signature}


def discard_unsaved_copy(preview: Dict[str, Any], signature: str, *, worktree_root: Optional[Any] = None,
                         data_dir: Optional[Any] = None, live: Optional[Any] = None,
                         reasons: Optional[Dict[str, str]] = None) -> bool:
    """The owner's explicit discard of one finished copy, after its unsaved paths were shown: never called by a
    pass. It goes only while ``discard_offer`` still answers with the previewed
    ``signature``, asked again under the copy's record lock, and through ``sw._retire_worktree``'s keep gate.
    A body candidate's work is captured on its pin first; a branch that may hold commits stays."""
    from ouroboros import body_candidate, subagent_worktrees as sw

    reasons = {} if reasons is None else reasons
    path, row, root = str(preview.get("path") or ""), dict(preview), sw._resolve_root(worktree_root)
    is_body = preview.get("kind") == sw._KIND_BODY_CANDIDATE
    if sw._record_at(path, data_dir, op="discard_unsaved") != preview:
        reasons[path] = "changed"
        return False
    verdict: Dict[str, Any] = {}
    if is_body:  # the capture and its witness are written outside the record lock, as the GC's are
        verdict = body_candidate.retention_verdict(row, expired=True, data_dir=data_dir, discard_excluded=True)
        if not verdict["remove"]:
            reasons[path] = verdict["reason"]
            return False

    def holds(match: Dict[str, Any], branch: str) -> Optional[str]:
        offer = discard_offer(match, data_dir, live, root)
        refusal = offer["refusal"] or ("" if offer["signature"] == signature else "changed")
        if not refusal and is_body and not body_candidate.verdict_holds(match, verdict, data_dir=data_dir):
            refusal = "changed"
        if refusal:
            reasons[path] = refusal
            return None
        return "" if (verdict.get("keep_branch") if is_body else offer["keep_branch"]) else branch

    return sw._retire_worktree(row, root, data_dir, str(row.get("branch") or ""), holds=holds, reasons=reasons)


def _stat_of(path: pathlib.Path) -> Optional[List[int]]:
    try:
        seen = path.lstat()
    except FileNotFoundError:
        return None
    return [seen.st_mode, seen.st_size, seen.st_dev, seen.st_ino, seen.st_mtime_ns, seen.st_ctime_ns]


def _keeps_commits(row: Dict[str, Any], tip: str, base: str) -> Optional[bool]:
    """Must the copy's branch stay for its commits? ``None`` when its HEAD holds commits no staying ref holds."""
    from ouroboros import subagent_worktrees as sw

    if base and tip == base:
        return _branch_may_hold_commits(row, base)
    branch = str(row.get("branch") or "")
    repo = pathlib.Path(str(row.get("git_dir") or row.get("repo_dir") or "."))
    held = branch and sw._git(repo, "merge-base", "--is-ancestor", tip, f"refs/heads/{branch}", check=False).returncode == 0
    return True if held else None


def plain_copy_branch(row: Dict[str, Any], branch: str, *, root: pathlib.Path, data_dir: Optional[Any], live: Any) -> Optional[str]:
    """The decision at the destructive boundary, under the copy's record lock (``holds`` of ``_retire_worktree``):
    the task still proven over and the work still proven saved. The branch to drop, or ``None`` to keep the copy.
    A record outside the worktree root deletes nothing, so its survivor stamp stays that path's answer."""
    from ouroboros import subagent_worktrees as sw

    if not sw._deletable(pathlib.Path(str(row.get("path") or "")), root):
        return branch
    if not sw._task_proven_over(str(row.get("task_id") or ""), data_dir, live):
        return None
    verdict = plain_copy_verdict(row, data_dir)
    return None if not verdict["remove"] else "" if verdict["keep_branch"] else branch


def _branch_may_hold_commits(row: Dict[str, Any], base: str) -> bool:
    from ouroboros import subagent_worktrees as sw

    branch = str(row.get("branch") or "")
    if not branch:
        return False
    repo = pathlib.Path(str(row.get("git_dir") or row.get("repo_dir") or "."))
    try:
        code = sw._git(repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}", check=False).returncode
        if code:  # 1 is Git's own "no such ref"; anything else could not read the branch
            return code != 1
        return not base or sw._git(repo, "rev-list", "--count", f"refs/heads/{branch}", "--not", base).stdout.strip() != "0"
    except Exception:
        return True


def _manifest(path: pathlib.Path, row: Dict[str, Any], base: str, data_dir: Optional[Any]) -> Tuple[Optional[Dict[str, Any]], pathlib.Path]:
    """The task's capture manifest when it speaks for exactly this checkout at this base, and its folder."""
    from ouroboros.headless import task_artifacts_dir
    from ouroboros.headless_status import ARTIFACT_STATUS_READY_WITH_CHANGES
    from ouroboros.subagent_worktrees import _data_dir

    artifacts = task_artifacts_dir(_data_dir(data_dir), str(row.get("task_id") or ""), create=False)
    manifest = json.loads((artifacts / "workspace_patch.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != ARTIFACT_STATUS_READY_WITH_CHANGES or manifest.get("errors")
            or pathlib.Path(str(manifest.get("workspace_root") or "")).resolve(strict=False) != path.resolve(strict=False)
            or manifest.get("base_head") != base or manifest.get("current_head") != base):
        return None, artifacts
    return manifest, artifacts


def _file_result_holds(path: pathlib.Path, change: Dict[str, Any]) -> bool:
    """The path as it stands is the captured postimage (kind, size, sha256, exact mode), and the archive
    gives those bytes back. A capture that recorded the path's deletion holds its absence."""
    from ouroboros.workspace_file_outputs import _matches, _path, _side

    after = change["after"]
    if not _matches(_side(_path(path, change["path"])), after, exact_mode=True):
        return False
    if not after or after["kind"] != "file":
        return True
    digest = hashlib.sha256()
    with zipfile.ZipFile(change["archive_path"]) as archive, archive.open(change["member"]) as member:
        for chunk in iter(lambda: member.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == after["sha256"]


def _unsaved_paths(path: pathlib.Path, row: Dict[str, Any], work: Dict[str, Any], base: str, data_dir: Optional[Any]) -> List[str]:
    """The checkout's unique paths no capture holds; every one of them when the capture cannot speak for it."""
    unique = [*work["tracked"], *work["untracked"], *work["ignored"]]
    if not base or work["tip"] != base:
        return unique
    try:
        return _unsaved(path, row, unique, base, data_dir)
    except Exception:
        log.debug("Capture of copy %s could not be read", path, exc_info=True)
        return unique


def _unsaved(path: pathlib.Path, row: Dict[str, Any], unique: List[str], base: str, data_dir: Optional[Any]) -> List[str]:
    """The unique paths whose bytes and mode equal neither their verified file result nor the saved patch
    replayed onto the base in a private index. A capture of file results alone has no patch to replay."""
    from ouroboros import subagent_worktrees as sw
    from ouroboros.artifacts import stream_artifact_file
    from ouroboros.workspace_file_outputs import file_output_changes

    manifest, artifacts = _manifest(path, row, base, data_dir)
    if manifest is None:
        return list(unique)
    changes = {change["path"]: change for change in file_output_changes(manifest, artifacts)}
    differing = [rel for rel in unique if rel in changes and not _file_result_holds(path, changes[rel])]
    unique = [rel for rel in unique if rel not in changes]
    if not unique:
        return differing
    patch: Optional[pathlib.Path] = None
    if manifest.get("patch_size"):
        patch = artifacts / str(manifest.get("patch_name") or "workspace.patch")
        measured = stream_artifact_file(patch)
        if (measured["sha256"], measured["size"]) != (manifest.get("sha256"), manifest.get("patch_size")):
            return [*differing, *unique]
    common = sw._git(path, "rev-parse", "--path-format=absolute", "--git-common-dir").stdout.strip()
    with tempfile.TemporaryDirectory(prefix="ob-copy-proof-") as scratch:
        env = {**sw._git_env_index(pathlib.Path(scratch) / "index"), "GIT_OPTIONAL_LOCKS": "0",
               "GIT_OBJECT_DIRECTORY": str(pathlib.Path(scratch) / "objects"),
               "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(pathlib.Path(common) / "objects")}
        (pathlib.Path(scratch) / "objects").mkdir()
        sw._git_env(path, "read-tree", base, env=env)
        if patch is not None:
            sw._git_env(path, "apply", "--cached", "--whitespace=nowarn", str(patch), env=env)
        saved = _index_entries(sw._git_env(path, "ls-files", "-s", "-z", env=env).stdout)
        return [*differing, *_paths_differing(path, unique, saved, env)]


def _index_entries(raw: bytes) -> Dict[str, Tuple[str, str]]:
    entries = {}
    for chunk in raw.split(b"\0"):
        if chunk:
            head, _, rel = chunk.partition(b"\t")
            mode, oid, _stage = head.decode().split()
            entries[rel.decode("utf-8", "surrogateescape")] = (mode, oid)
    return entries


def _paths_differing(path: pathlib.Path, unique: List[str], saved: Dict[str, Tuple[str, str]], env: Dict[str, str]) -> List[str]:
    """The paths that do not stand as their saved entry: gone where the patch holds none is equal; a folder or
    special file differs (no blob can hold it). Raw bytes, no clean filter: a file whose filesystem bytes differ
    from its Git form is not what the patch restores. A path the patch does not hold needs no hashing: it differs.
    One process hashes every regular file."""
    from ouroboros import subagent_worktrees as sw

    differing: List[str] = []
    regular: List[str] = []
    for rel in unique:
        target, entry = path / rel, saved.get(rel)
        try:
            info = target.lstat()
        except FileNotFoundError:
            if entry is not None:
                differing.append(rel)
            continue
        if entry is None:
            differing.append(rel)
        elif stat.S_ISLNK(info.st_mode):
            if entry != (_SYMLINK, sw._git_env(path, "hash-object", "--no-filters", "--stdin", env=env,
                                               input_bytes=os.fsencode(os.readlink(target))).stdout.decode().strip()):
                differing.append(rel)
        elif not stat.S_ISREG(info.st_mode) or entry[0] != (_EXECUTABLE if info.st_mode & stat.S_IXUSR else _REGULAR):
            differing.append(rel)
        else:
            regular.append(rel)
    if not regular:
        return differing
    hashed = sw._git_env(path, "hash-object", "--no-filters", "--stdin-paths", env=env,
                         input_bytes=b"".join(_quoted(rel) + b"\n" for rel in regular)).stdout.decode().split()
    if len(hashed) != len(regular):
        raise ValueError("git hashed another number of files than it was given")
    return [*differing, *(rel for rel, oid in zip(regular, hashed) if oid != saved[rel][1])]


def _quoted(rel: str) -> bytes:
    """One ``--stdin-paths`` line, C-quoted so a newline or a leading quote in a name stays that name."""
    raw = os.fsencode(rel).replace(b"\\", b"\\\\").replace(b'"', b'\\"').replace(b"\n", b"\\n")
    return b'"' + raw + b'"'
