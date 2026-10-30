"""Managed Update preparation outside the serving checkout (unified Update, owner decisions 1-3).

Preparation never changes the checkout that serves: its HEAD, branch, index, files,
imported modules, Python environment and writer admission stay as they are until the
owner activates. The work happens in ONE linked worktree of the serving repository,
recorded as a ``kind=update_candidate`` row of the existing copy registry
(``ouroboros.subagent_worktrees``) beside body candidates:

* The owner's local work is three independent layers, read without writing the real
  index (the temporary-index technique of ``body_candidate._preserve``): H, the HEAD
  commit; I, the tree of the real index copied entry by entry into a temporary index;
  W, that temporary index after ``add -A`` (tracked and untracked files, ignored ones
  excluded), checked against the disk with ``diff-files``. They are pinned as K1 = I on
  H and K2 = W on K1 under ``refs/ouroboros/update/<id>/``, never on a branch. Index
  states Git cannot carry this way (unmerged entries, skip-worktree, assume-unchanged,
  intent-to-add, submodules) and a volume without room for the untracked blobs plus the
  workspace refuse with their paths and numbers; the serving checkout is untouched.
* C merges the channel's pinned target into H inside the workspace (a fast-forward is
  the target itself); version carriers resolve and project exactly as the legacy planner
  does (``update_merge_plan``). C carries the whole target as its second parent.
* The layers ride onto C by three-way merges on an exact base (``_merge_tree``: objects only with
  ``git merge-tree --write-tree``, in the workspace on an older Git):
  I' = merge(H; C, K1) and W' = merge(K1; I' on C, K2). Only C enters the branch's history
  at activation; I' and W' stay staged and unstaged work. A conflict in any step stays
  in the workspace with its paths (``needs_adaptation``); the serving tree is unchanged.
  ``without_local_changes`` is the separate explicit choice I' = W' = C, which keeps K1
  and K2 on their private refs.
* ``ready`` means C, I' and W' exist and their workspace checks passed. Activation
  (``ouroboros.gateway.update_activation``) re-reads the base and the serving runtime and
  packages it was prepared on (``environment_moved``) after the writer fence and hands the
  candidate to ``body_adoption.authorize_update``.
* The boot's settled activation outcome lands on the candidate record and, in one retained
  record of this module's own (``ACTIVATION_REL``, never the supervisor's state, which a
  providerless generation lacks), outlives the record the GC retires until the next admitted
  preparation; the keep policy's healthy composition and start failure live there as well.
* A step records the process that works in it (``worker``). Every child it starts goes through
  the shell's tracked admission (``_run``), so a stop that has begun refuses the next one and the
  step fails typed (``stopped``). The boot fails as interrupted a candidate left preparing or
  checking by a process proven ended (``reconcile_interrupted``).

The record is installation-owned, not task-owned: it lives while the Update names it
(preparing through an open activation) and the worktree GC retires its checkout, its
sibling scratch, its refs and its generation environment only once it is discarded,
superseded or settled AND nothing still works in it: no step of this process and no
resolver task or descendant whose exit is unproven (``retention_verdict``). A discard
or supersession cancels the resolver through the task owner first. At most one
candidate exists at a time: a new preparation supersedes the previous one, so retained
data is one checkout of the body plus its scratch and, after a lock change, one
copy-on-write clone of the serving packages. Record transitions (supersession,
resolver admission, discard, activation, the boot's settlement) are serialized by
``lifecycle()``; a settled record is never written again, so a step that outlived its
candidate stops at its next write.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.runtime_venv import PRIVATE_MARKER, PRIVATE_NAME, interpreter  # the hook restates both literals
from ouroboros.utils import append_jsonl, atomic_write_json, read_json_dict, utc_now_iso, write_text_atomic

log = logging.getLogger(__name__)

KIND = "update_candidate"
BRANCH_PREFIX = "candidate/update-"
REF_PREFIX = "refs/ouroboros/update/"
# Settled states release the record: the GC may retire its checkout and scratch.
SETTLED = ("discarded", "superseded", "activated")
_IN_PROGRESS = ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-apply", "rebase-merge", "sequencer")
_GIT_TIMEOUT_SEC = 300
_IDENTITY = {"GIT_AUTHOR_NAME": "Ouroboros", "GIT_AUTHOR_EMAIL": "ouroboros@localhost",
             "GIT_COMMITTER_NAME": "Ouroboros", "GIT_COMMITTER_EMAIL": "ouroboros@localhost", "LC_ALL": "C"}
# The last activation's settled outcome, one record under the data root: never authority, replaced at the next
# admission. Not the supervisor's state.json: only its initializer creates that, and a providerless boot runs none.
ACTIVATION_REL = pathlib.Path("state") / "update_activation.json"
_WORKING_STATES = ("preparing", "checking")  # a step of some process works in the candidate until it moves on
_CARRIED_LIMIT = 200  # paths kept per carried layer; the counts stay exact
_running = threading.Lock()  # one preparation or adaptation step at a time in this process
_lifecycle = threading.RLock()  # record transitions; held briefly, never across merges or checks
_working: set = set()  # candidates a step of this process works in: their checkout outlives the step
_last_refusal: Dict[str, Any] = {}  # this process's latest refusal: it left no live candidate to say it


def lifecycle() -> "threading.RLock":
    """The lock every candidate state transition takes (activation and discard included)."""
    return _lifecycle


def _registry():
    """The copy registry, read lazily: the Update domain reaches delegation's owner only at call time."""
    from ouroboros import subagent_worktrees

    return subagent_worktrees


class PrepareRefused(Exception):
    """A typed refusal; the serving checkout was not changed."""

    def __init__(self, code: str, text: str, **facts: Any):
        super().__init__(text)
        self.code, self.text, self.facts = code, text, facts


# --------------------------------------------------------------------------- #
# Git, bound to one checkout
# --------------------------------------------------------------------------- #
def stopped() -> None:
    """Raise when this server's stop (or Emergency Stop) closed command admission: a step whose child was refused
    started nothing, and no refusal is ever read as that child's answer. The step fails; prepare again after it."""
    from ouroboros.tools.shell_process import admission_refusal

    refusal = admission_refusal()
    if refusal:
        raise PrepareRefused("UPDATE_PREPARE_STOPPED", f"{refusal}; this step started nothing further")


def _run(repo: Any, *args: str, env: Optional[Dict[str, str]] = None, data: Optional[bytes] = None,
         check: bool = True) -> subprocess.CompletedProcess:
    """One Git process in ``repo`` through the existing Git owner (``git_ops._run_git_process_bounded``: the
    shell's tracked admission and publication, its tree ended at the deadline), so a begun stop refuses it and
    the generation's stop ends one still running. A refused start raises (``stopped``), and so does one that ran
    out of time, whatever ``check`` says."""
    from supervisor.git_ops import FETCH_TIMEOUT_RC, _run_git_process_bounded

    cmd = ["git", *args]
    returncode, out, err = _run_git_process_bounded(
        cmd, timeout=_GIT_TIMEOUT_SEC, cwd=pathlib.Path(repo), text=False, input_data=data,
        env=env if env is not None else _registry()._host_env(GIT_OPTIONAL_LOCKS="0", **_IDENTITY))
    if returncode == 127:
        stopped()
    if returncode == FETCH_TIMEOUT_RC:
        raise subprocess.TimeoutExpired(cmd, _GIT_TIMEOUT_SEC, output=out, stderr=err)
    if check and returncode != 0:
        raise _registry().GitFailed(returncode, cmd, out, err)
    return subprocess.CompletedProcess(cmd, returncode, out, err)


def _out(repo: Any, *args: str, **kwargs: Any) -> str:
    return _run(repo, *args, **kwargs).stdout.decode("utf-8", "surrogateescape").strip()


def _ancestor(repo: Any, older: str, newer: str) -> bool:
    return _run(repo, "merge-base", "--is-ancestor", older, newer, check=False).returncode == 0


def _paths(raw: bytes) -> List[str]:
    return [chunk.decode("utf-8", "surrogateescape") for chunk in raw.split(b"\0") if chunk]


# --------------------------------------------------------------------------- #
# Local work as three layers
# --------------------------------------------------------------------------- #
def snapshot(repo: Any, branch_dev: str) -> Dict[str, Any]:
    """H, I and W of the serving checkout, plus every reason they cannot be carried.

    Reads only: the real index is copied once, byte for byte with its own time, and never refreshed or written.
    Every index fact comes from that one copy, whose stat data lets ``add -A`` hash only what changed (a listing
    carries none, so it rehashed the whole tree on every call) and whose time keeps Git checking by content an entry
    stamped in that second (racily clean: a copy stamped later trusts a same-size edit made then). A split index's
    shared part is read from the repository where Git keeps it, and the copy is written whole (``core.splitIndex``
    off), so nothing is written beside the real one. The copy lives outside the work tree and is deleted.
    ``refusals`` lists typed reasons with up to five paths each; an empty list means the layers are complete.
    """
    repo = pathlib.Path(repo)
    head = _out(repo, "rev-parse", "--verify", "HEAD^{commit}")
    branch = _out(repo, "symbolic-ref", "-q", "--short", "HEAD", check=False)
    git_dir = pathlib.Path(_out(repo, "rev-parse", "--absolute-git-dir"))
    real_index = repo / _out(repo, "rev-parse", "--git-path", "index")  # relative to the checkout, or absolute
    refusals: List[str] = []
    if branch != branch_dev:
        refusals.append(f"branch:{branch or 'detached'} (the update carries local work only on {branch_dev})")
    busy = [name for name in _IN_PROGRESS if (git_dir / name).exists()]
    if busy:
        refusals.append("operation_in_progress:" + ", ".join(busy))
    with tempfile.TemporaryDirectory(prefix="ouroboros-update-index-") as scratch:
        index = pathlib.Path(scratch) / "index"
        env = _registry()._host_env(GIT_INDEX_FILE=str(index), **_IDENTITY)

        def copy(*args: str, **kwargs: Any) -> subprocess.CompletedProcess:
            return _run(repo, "-c", "core.splitIndex=false", *args, env=env, **kwargs)

        try:
            with open(real_index, "rb") as source:  # one read of a file Git replaces whole: one coherent index
                index.write_bytes(source.read())  # and the time of that same file, mtime for both stamps
                os.utime(index, ns=(os.fstat(source.fileno()).st_mtime_ns,) * 2)
        except FileNotFoundError:
            copy("read-tree", "--empty")  # a repository that never wrote an index
        real = copy("ls-files", "-s", "-z").stdout
        flagged: Dict[str, List[str]] = {}
        for chunk in copy("ls-files", "-v", "-z").stdout.split(b"\0"):
            if len(chunk) > 2:
                tag, path = chunk[:1].decode(), chunk[2:].decode("utf-8", "surrogateescape")
                if tag in ("S", "s"):
                    flagged.setdefault("skip_worktree", []).append(path)
                elif tag.islower():
                    flagged.setdefault("assume_unchanged", []).append(path)
        for chunk in real.split(b"\0"):
            if chunk:
                meta, _tab, path = chunk.partition(b"\t")
                mode, _sha, stage = meta.decode().split(" ")
                if stage != "0":
                    flagged.setdefault("unmerged", []).append(path.decode("utf-8", "surrogateescape"))
                elif mode == "160000":
                    flagged.setdefault("submodule", []).append(path.decode("utf-8", "surrogateescape"))
        for chunk in copy("status", "--porcelain=v1", "-z", "--untracked-files=no").stdout.split(b"\0"):
            if chunk[:2] == b" A":
                flagged.setdefault("intent_to_add", []).append(chunk[3:].decode("utf-8", "surrogateescape"))
        refusals += [f"{name}:{', '.join(sorted(set(paths))[:5])}" for name, paths in sorted(flagged.items())]
        index_tree = copy("write-tree").stdout.decode().strip() if not flagged.get("unmerged") else ""
        copy("add", "-A", "--", ".")
        if copy("diff-files", "--quiet", check=False).returncode != 0:
            refusals.append("files_changed_while_read: the working files moved during the snapshot; try again")
        files_tree = copy("write-tree").stdout.decode().strip()
    untracked = 0
    for rel in _paths(_run(repo, "ls-files", "-o", "--exclude-standard", "-z").stdout):
        try:
            untracked += os.lstat(repo / rel).st_size
        except OSError:
            continue
    head_tree = _out(repo, "rev-parse", f"{head}^{{tree}}")
    return {"head": head, "branch": branch, "index_tree": index_tree, "files_tree": files_tree,
            "index_digest": hashlib.sha256(real).hexdigest(), "untracked_bytes": untracked,
            "dirty": index_tree != head_tree or files_tree != head_tree, "refusals": refusals}


def fingerprint(snap: Dict[str, Any]) -> Dict[str, str]:
    """What must still hold when the owner activates: any change means the composition moved."""
    return {key: str(snap.get(key) or "") for key in ("head", "branch", "index_digest", "index_tree", "files_tree")}


def _space(repo: pathlib.Path, root: pathlib.Path, target: str, untracked: int) -> str:
    """Why the untracked blobs plus one workspace checkout do not fit, with numbers; ``""`` when they do."""
    listing = _run(repo, "ls-tree", "-r", "-l", "-z", "--full-tree", target).stdout.split(b"\0")
    workspace = sum(int(meta.split()[3]) for meta, _tab, _path in (chunk.partition(b"\t") for chunk in listing if chunk)
                    if len(meta.split()) > 3 and meta.split()[3].isdigit())
    root.mkdir(parents=True, exist_ok=True)
    same = os.stat(repo).st_dev == os.stat(root).st_dev
    needs = [(repo, untracked + (workspace if same else 0))] + ([] if same else [(root, workspace)])
    for path, need in needs:
        free = shutil.disk_usage(path).free
        if free < need:
            return (f"space: {need} bytes needed on the volume of {path} (untracked files {untracked}, workspace "
                    f"{workspace}), {free} free")
    return ""


def _merge_tree(repo: Any, base: str, ours: str, theirs: str,
                workspace: Optional[pathlib.Path] = None) -> Tuple[str, List[str]]:
    """``(tree, conflicted paths)`` of the three-way merge of ``ours`` and ``theirs`` on exactly ``base``.

    Git takes a merge's base from history unless told (``--merge-base``, Git 2.40), so each side is
    first made a fresh commit of its tree on ``base`` alone: ``base`` is then their one merge base and
    any Git merges them on it. ``git merge-tree --write-tree`` (Git 2.38) writes the result as objects
    only; an older Git merges the same two commits in the candidate's own ``workspace`` (never the
    serving checkout) and puts that back as it was. A conflict keeps Git's markers in the tree.
    """
    ours, theirs = (_out(repo, "commit-tree", f"{side}^{{tree}}", "-p", base, "-m", "ouroboros update: layer side")
                    for side in (ours, theirs))
    options = ("merge-tree", "--write-tree", "-z", "--name-only")
    proc = _run(repo, *options, ours, theirs, check=False)
    if proc.returncode not in (0, 1):
        if workspace is not None and _run(repo, *options, base, base, check=False).returncode:
            return _workspace_merge(workspace, ours, theirs)  # this Git writes no merge tree as objects
        raise PrepareRefused("UPDATE_LAYER_MERGE_FAILED", "git merge-tree failed: "
                             + proc.stderr.decode("utf-8", "replace").strip()[-400:])
    head = proc.stdout.partition(b"\0\0")[0].strip(b"\0").split(b"\0")
    conflicts = [part.decode("utf-8", "surrogateescape") for part in head[1:] if part]
    if (proc.returncode == 1) != bool(conflicts):
        raise PrepareRefused("UPDATE_LAYER_MERGE_FAILED", "git merge-tree's result and conflict list disagree")
    return head[0].decode(), sorted(set(conflicts))


def _workspace_merge(workspace: pathlib.Path, ours: str, theirs: str) -> Tuple[str, List[str]]:
    """The same merge in the candidate's workspace, as ``_merge`` makes C there (rerere off): ``ours`` checked out,
    ``theirs`` merged without a commit, the conflicted paths staged with their markers, the tree written; then the
    workspace returns to the commit it held."""
    from supervisor import update_candidate as uc

    held = _out(workspace, "rev-parse", "--verify", "HEAD^{commit}")
    try:
        _run(workspace, "reset", "--hard", "--quiet", ours)
        rc, _merged, error = uc._git_run(["git", *uc._MERGE_NEUTRAL_FLAGS, "merge", "--no-commit", "--no-ff",
                                          "--no-edit", theirs], cwd=str(workspace))
        if rc not in (0, 1):
            stopped()
            raise PrepareRefused("UPDATE_LAYER_MERGE_FAILED", f"the layer merge could not run in the workspace: {error}")
        conflicts = sorted(set(_paths(_run(workspace, "diff", "--name-only", "-z", "--diff-filter=U").stdout)))
        if conflicts:
            _run(workspace, "add", "-A", "--", *conflicts)
        return _out(workspace, "write-tree"), conflicts
    finally:
        _run(workspace, "reset", "--hard", "--quiet", held)


def _layer_changes(repo: Any, old: str, new: str) -> List[List[str]]:
    """``[[status, path], ...]``: every change of one layer, in Git's order."""
    parts = _paths(_run(repo, "diff-tree", "-r", "-z", "--no-renames", "--name-status", old, new).stdout)
    return [[status, path] for status, path in zip(parts[0::2], parts[1::2])]


def carried_list_path(row: Dict[str, Any]) -> pathlib.Path:
    """The complete list of the work a candidate carries: a plain file beside its copy, which no scan of the copies
    takes for one, removed with it (``release_environment``)."""
    copy = pathlib.Path(str(row["path"]))
    return copy.with_name(copy.name + ".uncommitted.txt")


def _printed(path: str) -> str:
    return path if path.isprintable() and not path.startswith('"') else json.dumps(path)


def _listing(layers: Dict[str, List[List[str]]]) -> str:
    lines = ["Your uncommitted work as this update keeps it after the restart. Each line is one change as Git names "
             "it: A added, M modified, D deleted, T type changed. A path with control characters is quoted as in JSON."]
    for layer, title in (("index", "Staged"), ("files", "In the files")):
        lines += ["", f"{title}: {len(layers[layer])}"]
        lines += [f"{status}\t{_printed(path)}" for status, path in layers[layer]]
    return "\n".join(lines) + "\n"


def carried_layers(repo: Any, refs: str, listing: pathlib.Path) -> Dict[str, Any]:
    """The owner's work as it stays uncommitted on C after the restart, read from the candidate's pinned refs:
    staged (``c`` -> ``i_prime``) and in the files (``i_prime`` -> ``w_prime``). Every change is written once, to
    ``listing``; the record keeps each layer's exact count, its first paths and where that list is."""
    layers = {"index": _layer_changes(repo, refs + "c", refs + "i_prime"),
              "files": _layer_changes(repo, refs + "i_prime", refs + "w_prime")}
    carried: Dict[str, Any] = {layer: {"count": len(changes), "paths": changes[:_CARRIED_LIMIT]}
                               for layer, changes in layers.items()}
    if any(layers.values()):
        write_text_atomic(listing, _listing(layers))
        carried["list"] = str(listing)
    return carried


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #
def _rows(data_dir: Optional[Any] = None) -> List[Dict[str, Any]]:
    return [row for row in _registry()._records(data_dir, op="read_update_candidate") if row.get("kind") == KIND]


def current(data_dir: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """The installation's live candidate (newest unsettled), or ``None``."""
    live = [row for row in _rows(data_dir) if row.get("state") not in SETTLED]
    return max(live, key=lambda row: float(row.get("created_at") or 0)) if live else None


def find(candidate_id: str, data_dir: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    return next((row for row in _rows(data_dir) if row.get("candidate_id") == candidate_id), None)


def _update(row: Dict[str, Any], data_dir: Optional[Any] = None, **fields: Any) -> Dict[str, Any]:
    def change(entry: Dict[str, Any]) -> Dict[str, Any]:
        if entry.get("kind") != KIND:
            raise PrepareRefused("UPDATE_CANDIDATE_MISSING", "the update candidate's record changed kind")
        if entry.get("state") in SETTLED:
            # Discarded, superseded or activated: a step that outlived its candidate stops here.
            raise PrepareRefused("UPDATE_CANDIDATE_SETTLED", f"the update candidate is {entry.get('state')}")
        entry.update(fields, updated_at=utc_now_iso())
        return entry

    updated = _registry()._update_record(str(row["path"]), change, data_dir, op="update_update_candidate",
                                 task_id=str(row.get("candidate_id") or ""))
    if updated is None:
        raise PrepareRefused("UPDATE_CANDIDATE_MISSING", "the update candidate's record is gone")
    return dict(updated)


def _settle(row: Dict[str, Any], state: str, data_dir: Optional[Any] = None, **fields: Any) -> Dict[str, Any]:
    """Settle the record and drop its private refs; the GC takes the checkout, scratch and generation
    environment once nothing works in them (``retention_verdict``, ``release_environment``)."""
    settled = _update(row, data_dir, state=state, **fields)
    try:
        release_refs(settled)
    except Exception:
        log.warning("update candidate %s kept its refs", settled.get("candidate_id"), exc_info=True)
    return settled


def _event(kind: str, **fields: Any) -> None:
    try:
        append_jsonl(_registry()._data_dir(None) / "logs" / "supervisor.jsonl", {"ts": utc_now_iso(), "type": kind, **fields})
    except Exception:
        log.debug("update candidate event append failed", exc_info=True)


def _worker() -> Dict[str, Any]:
    """This process as the one working in a candidate: what ``local_custody_repair.owner_ended`` reads (the pid,
    its start and the custody proof whose kernel birth and session tell a reused pid or an exec'd image)."""
    from ouroboros.platform_layer import process_start_time
    from ouroboros.tool_custody import owner_proof

    return {"pid": os.getpid(), "birth": process_start_time(os.getpid()), "proof": owner_proof()}


def _provision(serving: pathlib.Path, target: Dict[str, str], snap: Dict[str, Any]) -> Dict[str, Any]:
    from supervisor.update_native import _tree_version

    root, data_root = _registry()._resolve_root(), _registry()._data_dir(None)
    sha = target["target_sha"]
    # Both versions as pinned now: a channel that moves on later never renames what was prepared.
    target = {**target, "version": _tree_version(serving, sha)}
    candidate_id = f"update_{sha[:12]}_{uuid.uuid4().hex[:6]}"
    path = _registry().copy_path(root, candidate_id).resolve()
    _registry()._assert_root_isolated(path, serving, data_root)
    common = pathlib.Path(_out(serving, "rev-parse", "--git-common-dir"))
    branch, suffix = f"{BRANCH_PREFIX}{sha[:12]}", 1
    while _run(serving, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}", check=False).returncode == 0:
        suffix += 1
        branch = f"{BRANCH_PREFIX}{sha[:12]}-{suffix}"
    now = time.time()
    row: Dict[str, Any] = {
        "kind": KIND, "candidate_id": candidate_id, "task_id": candidate_id, "owners": [], "path": str(path),
        "branch": branch, "base_sha": snap["head"], "repo_dir": str(serving),
        "git_dir": str((common if common.is_absolute() else serving / common).resolve()),
        "created_at": now, "used_at": now, "state": "preparing", "target": dict(target),
        "base": fingerprint(snap), "base_version": _tree_version(serving, snap["files_tree"]),
        "refs": f"{REF_PREFIX}{candidate_id}/", "ready": True, "worker": _worker(),
    }
    root.mkdir(parents=True, exist_ok=True)
    recorded = False
    try:
        with _registry()._record_lock(path, None, op="provision_update_candidate", task_id=candidate_id):
            if _registry()._record_at(path, None, op="provision_update_candidate") is not None or path.exists():
                raise PrepareRefused("UPDATE_CANDIDATE_PATH_OCCUPIED", f"{path} already holds unrecorded work")
            _registry()._put_record(row)
            recorded = True
            # A gone checkout's leftover registration at this path (``subagent_worktrees._forget_checkout``).
            _run(serving, "worktree", "remove", "--force", "--force", str(path), check=False)
            _run(serving, "worktree", "add", "--no-checkout", "-b", branch, str(path), snap["head"])
        _run(path, "reset", "--hard", "--quiet", "--no-recurse-submodules")
    except Exception as exc:  # not admitted: discarded at once like a copy the admission refuses; the live one stays
        if not recorded:
            raise  # no record yet: as every refusal before provisioning, nothing changed
        refused = exc if isinstance(exc, PrepareRefused) else PrepareRefused(  # a stop's own, else Git's words
            "UPDATE_WORKSPACE_FAILED", f"the update's copy could not be made: {type(exc).__name__}: {exc}")
        log.warning("update candidate %s discarded: %s", candidate_id, refused.text, exc_info=refused is not exc)
        _close_failed(row, f"{refused.code}: {refused.text}", "discarded")  # no private ref exists yet to release
        raise refused
    return row


def _merge(workspace: pathlib.Path, head: str, target_sha: str) -> Dict[str, Any]:
    """C in the workspace: H itself when it already contains the target (an app-only Update: no commit is
    made), the target itself when H is its ancestor, else a merge commit [H, target]."""
    from supervisor import update_candidate as uc
    from supervisor.update_carriers import resolve_carrier_conflicts, tracked_run_git
    from supervisor.update_merge_plan import _build_clean_merge_commit

    if _ancestor(workspace, target_sha, head):
        return {"kind": "current", "commit": head, "carrier_paths": []}
    if _ancestor(workspace, head, target_sha):
        _run(workspace, "reset", "--hard", "--quiet", target_sha)
        return {"kind": "fast_forward", "commit": target_sha, "carrier_paths": [], "local_commits": 0}
    # The commits of this installation the merge keeps in history (the updater's own earlier merges count too).
    local_commits = int(_out(workspace, "rev-list", "--count", f"{target_sha}..{head}") or 0)
    rc, _merged, error = uc._git_run(["git", *uc._MERGE_NEUTRAL_FLAGS, "merge", "--no-commit", "--no-ff", target_sha],
                                     cwd=str(workspace))
    if rc not in (0, 1):
        stopped()  # a refused start (rc 127) is no merge result
        raise PrepareRefused("UPDATE_MERGE_FAILED", f"the merge could not run in the workspace: {error}")
    unmerged = _paths(_run(workspace, "diff", "--name-only", "-z", "--diff-filter=U").stdout)
    # The resolver's Git through the tracked owner: once the stop began none starts, so that file is kept, and the
    # merge's next step, refused the same way, raises ``stopped``.
    carriers = list(resolve_carrier_conflicts(str(workspace), unmerged, prefer="theirs",
                                              run_git=tracked_run_git)["resolved"]) if unmerged else []
    remaining = sorted(set(unmerged) - set(carriers))
    if remaining:
        ok, _note, projection_error = uc.project_version_carriers(target_sha, cwd=str(workspace))
        m0, m0_error = uc.worktree_snapshot_tree(head, cwd=str(workspace)) if ok else ("", projection_error)
        if not m0:
            stopped()
        return {"kind": "conflicting", "commit": "", "conflicts": remaining, "carrier_paths": carriers,
                "m0_tree": m0, "m0_error": m0_error, "local_commits": local_commits}
    built, failure = _build_clean_merge_commit(str(workspace), head, target_sha, fast_forwardable=False,
                                               local_dirty_count=0)
    if failure is not None:
        stopped()
        raise PrepareRefused("UPDATE_MERGE_FAILED", str(failure.get("error") or failure))
    _run(workspace, "reset", "--hard", "--quiet", built)  # leaves the merge state; the branch is at C
    return {"kind": "clean", "commit": built, "carrier_paths": carriers, "local_commits": local_commits}


def environment_identity(repo: Any, files_tree: str, runtime: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """The interpreter, SQLite and the dependency input a generation with these files installs from:
    the file the launcher's ``install_deps`` selects and its exact bytes (``body_adoption.dependency_lock``,
    the same identity authorization compares). ``runtime``: a new app's measured interpreter
    (``update_native.staged_interpreter``) instead of this process's."""
    from ouroboros.body_adoption import dependency_lock
    from ouroboros.body_switch import runtime_identity

    return {**(runtime or runtime_identity()), **dependency_lock(repo, files_tree)}


# --------------------------------------------------------------------------- #
# Prepare
# --------------------------------------------------------------------------- #
def resolve_target(*, fetch: bool) -> Dict[str, Any]:
    """The channel's pinned target ``(channel, target_ref, target_sha)``: the existing resolver, no new feed."""
    from ouroboros.update_channels import get_update_channel
    from supervisor import git_ops as g
    from supervisor.update_merge import managed_update_constitution_present

    remote_name, remote_branch, branch_ref = g._managed_update_target()
    channel = get_update_channel()
    if not branch_ref:
        raise PrepareRefused("UPDATE_UNAVAILABLE", "no managed update remote is configured")
    if fetch and remote_name:
        ok, error = g.ensure_official_update_remote()
        if not ok:
            raise PrepareRefused("UPDATE_UNAVAILABLE", error or "could not configure the official update remote")
        rc, _out_text, error = g.git_fetch_bounded(remote_name)
        if rc != 0:
            raise PrepareRefused("UPDATE_UNAVAILABLE", error or f"git fetch {remote_name} failed")
    target_ref, target_sha, error = g._resolve_managed_update_target(remote_name, remote_branch, branch_ref, channel)
    if not target_ref or not target_sha:
        raise PrepareRefused("UPDATE_UNAVAILABLE", error or "could not resolve the update target")
    if not managed_update_constitution_present(target_sha):
        raise PrepareRefused("UPDATE_UNAVAILABLE", "the official update target does not preserve BIBLE.md")
    return {"channel": channel, "target_ref": target_ref, "target_sha": target_sha}


def prepare(*, fetch: bool = True, without_local_changes: bool = False, checks=None,
            expected: Optional[str] = None, admitted: bool = False) -> Dict[str, Any]:
    """Prepare (or replace) the installation's candidate for the pinned target; the serving tree is untouched.

    Returns the record. ``checks(row)`` (the workspace checks owner, injectable) returns ``{"ok": bool, ...}`` over
    the final composition. A refusal before provisioning raises ``PrepareRefused`` and leaves any previous candidate
    as it was; so does a copy whose checkout could not be made, discarded at once. ``expected`` names the candidate
    this preparation replaces (an apply without local changes asks for exactly one); a different live candidate at
    the admission refuses it with ``UPDATE_CANDIDATE_CHANGED`` and nothing of it remains. ``admitted``: the caller
    already holds this process's preparation slot (``admit``), which this call releases.
    """
    if not admitted and not _running.acquire(blocking=False):
        raise PrepareRefused("UPDATE_PREPARE_RUNNING", "an update is already being prepared")
    try:
        _last_refusal.clear()
        return _prepare(fetch=fetch, without_local_changes=without_local_changes, checks=checks, expected=expected)
    finally:
        _running.release()


def admit(expected: Optional[str] = None) -> None:
    """Take this process's one preparation slot for a request that starts ``prepare(admitted=True)`` on a thread.

    With ``expected`` (an apply without local changes), only while exactly that candidate is the
    live one and its state offers the action: a request from a screen that showed another one
    changes nothing. Raises ``PrepareRefused`` holding nothing; otherwise the caller owns the slot.
    """
    if not _running.acquire(blocking=False):
        raise PrepareRefused("UPDATE_PREPARE_RUNNING", "an update is already being prepared")
    try:
        if expected is not None:
            with _lifecycle:
                _expect_live(expected)
    except BaseException:
        _running.release()
        raise


def _expect_live(expected: str, *, besides: str = "") -> None:
    """Refuse unless ``expected`` is the newest unsettled candidate (other than ``besides``, the one being made)
    and its state offers the update without local changes. The caller holds ``lifecycle()``."""
    live = [entry for entry in _rows() if entry.get("state") not in SETTLED and entry.get("candidate_id") != besides]
    newest = max(live, key=lambda entry: float(entry.get("created_at") or 0)) if live else None
    if not expected or newest is None or newest.get("candidate_id") != expected \
            or "apply_without_local_changes" not in _actions(newest):
        raise PrepareRefused("UPDATE_CANDIDATE_CHANGED", "the prepared update changed; nothing was prepared",
                             expected=expected, live=str((newest or {}).get("candidate_id") or ""),
                             state=str((newest or {}).get("state") or ""))


def note_refusal(exc: PrepareRefused) -> None:
    """Keep a refusal, which left no live candidate of its own, for the panel (this process only)."""
    _last_refusal.clear()
    _last_refusal.update(code=exc.code, text=exc.text, at=utc_now_iso(), **{
        key: value for key, value in (exc.facts or {}).items() if key in ("refusals", "target_sha", "target_ref")})


def _prepare(*, fetch: bool, without_local_changes: bool, checks, expected: Optional[str] = None) -> Dict[str, Any]:
    from ouroboros import body_adoption
    from supervisor import git_ops as g
    from supervisor.update_merge import active_update_tx

    serving = pathlib.Path(g.REPO_DIR).resolve()
    if expected is not None:
        with _lifecycle:
            _expect_live(expected)  # before any copy exists; the admission below asks again
    if active_update_tx():
        raise PrepareRefused("UPDATE_ACTIVE", "a legacy managed update owns the checkout; let it settle first")
    if body_adoption.read(g.DRIVE_ROOT).get("phase") in ("authorized", "armed", "switching", "switched", "stuck"):
        raise PrepareRefused("UPDATE_ACTIVATION_PENDING", "an activation is still in progress")
    target = resolve_target(fetch=fetch)
    head = _out(serving, "rev-parse", "--verify", "HEAD^{commit}")
    snap = snapshot(serving, g.BRANCH_DEV)
    if _ancestor(serving, target["target_sha"], head):
        # The source is current; the app may still be unable to host it (an app-only Update, layers unchanged).
        from supervisor.update_native import app_need

        if app_need(serving, snap["files_tree"]).get("status") != "needed":
            raise PrepareRefused("UPDATE_CURRENT", "this checkout already contains the target", **target)
        if without_local_changes:
            raise PrepareRefused("UPDATE_CURRENT", "the source is current; only the app changes, with local work kept")
    if snap["refusals"]:
        raise PrepareRefused("UPDATE_LOCAL_STATE_UNSUPPORTED", "local work cannot be carried as it is: "
                             + "; ".join(snap["refusals"]), refusals=snap["refusals"])
    shortage = _space(serving, _registry()._resolve_root(), target["target_sha"], snap["untracked_bytes"])
    if shortage:
        raise PrepareRefused("UPDATE_SPACE", shortage)
    row = _provision(serving, target, snap)
    try:
        with _lifecycle:
            _working.add(row["candidate_id"])
            try:  # an admission that does not complete discards this copy at once: the live candidate stays current
                if body_adoption.read(g.DRIVE_ROOT).get("phase") in ("authorized", "armed", "switching", "switched",
                                                                      "stuck"):
                    raise PrepareRefused("UPDATE_ACTIVATION_PENDING", "an activation started meanwhile")
                if expected is not None:
                    _expect_live(expected, besides=row["candidate_id"])  # the admission's own answer
                for previous in [entry for entry in _rows() if entry.get("state") not in SETTLED
                                 and entry.get("candidate_id") != row["candidate_id"]]:
                    _stop_resolver(previous)
                    _settle(previous, "superseded", superseded_by=row["candidate_id"])
            except Exception as refused:  # also a live record held past the registry's wait; raised as it was
                _settle(row, "discarded", reason=f"{refused.code}: {refused.text}" if isinstance(refused, PrepareRefused)
                        else f"UPDATE_ADMISSION_FAILED: {type(refused).__name__}: {refused}")
                raise
            _forget_activation()  # this admitted attempt replaces the retained outcome of the last one
        return _layers_and_advance(serving, row, snap, without_local_changes, checks)
    except PrepareRefused as exc:
        if exc.code == "UPDATE_CANDIDATE_CHANGED":
            raise  # a stale request: its refusal is the answer, and its copy is already settled
        return _close_failed(row, f"{exc.code}: {exc.text}")
    finally:
        with _lifecycle:
            _working.discard(row["candidate_id"])


def _close_failed(row: Dict[str, Any], reason: str, state: str = "failed") -> Dict[str, Any]:
    """Record a step's failure on its candidate as ``state``; one settled meanwhile (discarded, superseded) stays."""
    try:
        return _update(row, state=state, reason=reason)
    except PrepareRefused as again:
        if again.code != "UPDATE_CANDIDATE_SETTLED":
            raise
        return find(row["candidate_id"]) or row


def _layers_and_advance(serving: pathlib.Path, row: Dict[str, Any], snap: Dict[str, Any],
                        without_local_changes: bool, checks) -> Dict[str, Any]:
    try:
        k1 = _out(serving, "commit-tree", snap["index_tree"], "-p", snap["head"], "-m", "ouroboros update: staged layer")
        k2 = _out(serving, "commit-tree", snap["files_tree"], "-p", k1, "-m", "ouroboros update: working layer")
        for name, sha in (("base", snap["head"]), ("k1", k1), ("k2", k2)):
            _run(serving, "update-ref", row["refs"] + name, sha)
        row = _update(row, layers_base={"head": snap["head"], "k1": k1, "k2": k2, "index_tree": snap["index_tree"],
                                        "files_tree": snap["files_tree"]},
                      dirty=bool(snap["dirty"]), without_local_changes=bool(without_local_changes))
        return _advance(row, checks)
    except PrepareRefused as exc:
        return _close_failed(row, f"{exc.code}: {exc.text}")
    except Exception as exc:
        log.warning("update preparation failed in its workspace", exc_info=True)
        return _close_failed(row, f"{type(exc).__name__}: {exc}")


def _advance(row: Dict[str, Any], checks=None) -> Dict[str, Any]:
    """Every step that is still open, from the recorded facts: C, then I', then W', then the checks.

    A conflict at any step hands the workspace to the adaptation resolver and returns
    (``update_adaptation``); its settlement records that step's result and calls this again.
    """
    from supervisor.update_adaptation import _adapt

    serving, workspace, refs = pathlib.Path(row["repo_dir"]), pathlib.Path(row["path"]), row["refs"]
    base, target = row["layers_base"], row["target"]["target_sha"]
    merge = dict(row.get("merge") or {})
    if not merge.get("commit"):
        merge = _merge(workspace, base["head"], target)
        if merge["kind"] == "conflicting":
            return _adapt(row, "merge", base["head"], merge["m0_tree"], merge["conflicts"], merge=merge)
        row = _update(row, merge=merge)
    _run(serving, "update-ref", refs + "c", merge["commit"])
    carry = dict(row.get("carry") or {})
    if row.get("without_local_changes"):
        c_tree = _out(serving, "rev-parse", f"{merge['commit']}^{{tree}}")
        carry = {"index_tree": c_tree, "index_commit": merge["commit"], "files_tree": c_tree}
    if not carry.get("index_tree"):
        carry["index_tree"], conflicts = _merge_tree(serving, base["head"], merge["commit"], base["k1"], workspace)
        if conflicts:
            return _adapt(row, "index", merge["commit"], carry["index_tree"], conflicts)
    if not carry.get("index_commit"):
        carry["index_commit"] = _out(serving, "commit-tree", carry["index_tree"], "-p", merge["commit"], "-m",
                                     "ouroboros update: carried staged layer")
    if not carry.get("files_tree"):
        carry["files_tree"], conflicts = _merge_tree(serving, base["k1"], carry["index_commit"], base["k2"],
                                                     workspace)
        if conflicts:
            return _adapt(row, "files", carry["index_commit"], carry["files_tree"], conflicts,
                          carry={key: value for key, value in carry.items() if key != "files_tree"})
    files_commit = carry["index_commit"] if carry["files_tree"] == carry["index_tree"] and row.get(
        "without_local_changes") else _out(serving, "commit-tree", carry["files_tree"], "-p", carry["index_commit"],
                                           "-m", "ouroboros update: carried working layer")
    added = [path for path in _paths(_run(serving, "diff-tree", "-r", "-z", "--name-only", "--diff-filter=A",
                                          base["files_tree"], carry["files_tree"]).stdout)
             if os.path.lexists(serving / path)]
    if added:
        return _update(row, state="failed", reason="ignored_path_collision:" + ", ".join(added[:5]), carry=carry)
    layers = {"old_commit": base["head"], "new_commit": merge["commit"], "old_index": base["index_tree"],
              "new_index": carry["index_tree"], "old_files": base["files_tree"], "new_files": carry["files_tree"]}
    _run(serving, "update-ref", refs + "i_prime", layers["new_index"])
    _run(serving, "update-ref", refs + "w_prime", layers["new_files"])
    row = _update(row, carry=carry, layers=layers, state="checking", composition_commit=files_commit, reason="",
                  carried=carried_layers(serving, refs, carried_list_path(row)))  # c, i_prime, w_prime pinned above
    from supervisor import update_native

    native = update_native.prepare(row)
    row = _update(row, native=native)
    if native["status"] == "manual":
        # The target needs an app this one cannot become automatically: the owner installs it, then prepares again.
        return _update(row, state="needs_decision", reason=f"native_app: {native.get('reason')}",
                       question=str(native.get("instruction") or ""))
    staged = native["status"] == "staged"
    environment = {"serving": environment_identity(serving, base["files_tree"]),
                   "candidate": environment_identity(serving, carry["files_tree"],
                                                     runtime=native.get("runtime") if staged else None)}
    environment["same"] = environment["serving"] == environment["candidate"]
    row = _update(row, environment=environment)
    if not environment["same"] or staged:
        # A new app runs its own bootstrap's pip before any hooked entry: its generation gets private packages
        # even when the lock is unchanged, prepared and checked by the interpreter that will run them (the clone
        # of a --runtime-venv of the app's Python is pointed at the staged app's until its checks are done).
        try:
            environment["generation"] = prepare_generation_environment(
                row, files_commit, python=native.get("python") if staged else None)
        except PrepareRefused as exc:
            return _update(row, state="failed", reason=f"{exc.code}: {exc.text}", environment=environment)
        row = _update(row, environment=environment)
    if checks is None:
        from supervisor.update_checks import workspace_checks as checks
    unpointed = ""
    try:
        outcome = checks(row)
    finally:
        if (environment.get("generation") or {}).get("links"):
            unpointed = _restore_venv(environment["generation"])
            try:  # the record no longer names a pointing to undo, also when the checks raised
                row = _update(row, environment=environment)
            except PrepareRefused:
                pass  # settled meanwhile: its own record stands
    if unpointed:
        return _update(row, state="failed", reason="UPDATE_ENVIRONMENT_FAILED: the prepared packages could not be "
                       f"pointed back at the installed app ({unpointed})")
    if not outcome.get("ok"):  # a run that did not complete verified nothing: never a failed test, never ready
        unverified = bool((outcome.get("tests") or {}).get("unverified"))
        return _update(row, state="failed", reason="checks_unverified" if unverified else "checks_failed",
                       checks=outcome)
    _event("update_candidate_ready", candidate_id=row["candidate_id"], target=target, merge=merge["kind"],
           dirty=bool(row.get("dirty")))
    return _update(row, state="ready", checks=outcome, reason="")


# --------------------------------------------------------------------------- #
# The generation environment: exactly the packages the new composition runs on
# --------------------------------------------------------------------------- #
def serving_user_base() -> pathlib.Path:
    """Where this generation's late-installed packages live: the packaged interpreter's user site."""
    import site

    return pathlib.Path(os.environ.get("PYTHONUSERBASE") or site.getuserbase()).resolve()


def private_environment_root(data_dir: Optional[Any] = None) -> pathlib.Path:
    """The fixed path of a bare interpreter's private packages; the package-init hook (``ouroboros/__init__.py``)
    resolves the same one before any import and runs that interpreter's entries there."""
    return _registry()._data_dir(data_dir) / "state" / PRIVATE_NAME


def serving_environment() -> Tuple[str, pathlib.Path]:
    """Where this generation's installed packages live, as a path the switch can move.

    ``("user_site", base)`` for the packaged interpreter (``launcher_bootstrap.embedded_python_env``),
    ``("venv", root)`` for a virtual environment, whose interpreter finds its packages from the
    ``pyvenv.cfg`` beside it; Windows renames both while their native modules, console image or venv
    redirector run (measured on NTFS), and a process holding one as its working directory refuses
    the rename, which the switch returns from (``body_switch._select_environment``). ``("private",
    root)`` for a bare interpreter, whose generations get a venv of their own at
    ``private_environment_root`` (absent until the first one is selected), and for that venv once it
    serves; under a launcher only one that hands itself over to new packages (``launcher_refusal``).
    ``("", prefix)`` where no such seam exists (``_no_seam``).
    """
    import sys

    from ouroboros.platform_layer import pip_install_target_args

    # The one owner of where installs go decides which seam this interpreter has.
    if pip_install_target_args(sys.executable) == ["--user"]:
        return "user_site", serving_user_base()
    private = private_environment_root()
    if sys.prefix != sys.base_prefix:
        return ("private", private) if os.path.realpath(sys.prefix) == os.path.realpath(private) else (
            "venv", pathlib.Path(sys.prefix))
    if launcher_refusal():
        return "", pathlib.Path(sys.prefix)
    return "private", private


# Where a package installer writes under an environment root: a venv's or a private environment's own site
# directory (POSIX, Windows) and a user base's (POSIX, the macOS framework's, Windows').
_SITE_DIRS = ("lib/python*/site-packages", "lib64/python*/site-packages", "lib/python/site-packages",
              "Lib/site-packages", "Python*/site-packages")
_INSTALLED = (".dist-info", ".egg-info", ".egg-link", ".pth")


def package_state(root: Any) -> str:
    """What a switch would replace of the packages at ``root``: one digest of every distribution and path file in
    its site directories, each by name and by its install record's size and modification time (a distribution's
    ``RECORD``). An install, upgrade, reinstall or removal there changes it; an import, a bytecode file or a read
    does not. ``""`` where nothing is installed (a bare interpreter's private root before its first selection)."""
    root, lines, seen = pathlib.Path(root), [], set()
    for pattern in _SITE_DIRS:
        for site_dir in sorted(root.glob(pattern)):
            real = os.path.realpath(site_dir)
            if real in seen or not site_dir.is_dir():
                continue
            seen.add(real)
            for name in sorted(os.listdir(site_dir)):
                if not name.endswith(_INSTALLED):
                    continue
                record = site_dir / name / "RECORD" if name.endswith(".dist-info") else site_dir / name
                try:
                    facts = os.stat(record)
                    stamp = f"{facts.st_size}\t{facts.st_mtime_ns}"
                except OSError:
                    stamp = "unreadable"
                lines.append(f"{site_dir.relative_to(root).as_posix()}/{name}\t{stamp}")
    return hashlib.sha256("\n".join(lines).encode("utf-8", "surrogateescape")).hexdigest() if lines else ""


# Set by a launcher (``launcher.start_agent``) that hands itself over to a fresh process when the packages it
# loaded are replaced (``launcher_bootstrap.loaded_packages_replaced``); never inherited by another launcher.
LAUNCHER_HANDOFF_ENV = "OUROBOROS_LAUNCHER_PACKAGE_HANDOFF"


def launcher_refusal() -> str:
    """Why the launcher this server runs under must not see its packages switched, or ``""``: one that predates
    handing itself over keeps the modules it imported over the new packages (and on a bare interpreter installs
    the lock into that interpreter's own packages at its restart)."""
    if os.environ.get("OUROBOROS_MANAGED_BY_LAUNCHER") == "1" and os.environ.get(LAUNCHER_HANDOFF_ENV) != "1":
        return ("the launcher running Ouroboros cannot restart itself onto new Python packages; quit Ouroboros, "
                "open it again, then restart to update")
    return ""


def _no_seam() -> str:
    """Why this interpreter's packages are not ones an Update can switch."""
    return launcher_refusal() or "this installation's packages cannot be switched; update by hand"


def _install_lock(target: pathlib.Path, lock: pathlib.Path, kind: str = "user_site",
                  python: Optional[str] = None) -> Dict[str, Any]:
    """The lock installed into ``target`` alone, bounded, through the shell's tracked process owner (its
    admission, Emergency Stop latch and reaping): ``pip install --user`` with that user base (by
    ``python``, a new app's own interpreter, when one comes), or the cloned venv's own interpreter
    installing into itself."""
    import sys

    from supervisor.update_checks import run_tracked

    env = _registry()._host_env(PYTHONDONTWRITEBYTECODE="1")
    for key in ("PYTHONUSERBASE", "PYTHONNOUSERSITE", "PIP_REQUIRE_VIRTUALENV", "PIP_TARGET", "PIP_PREFIX",
                "PIP_USER", "VIRTUAL_ENV", "PYTHONHOME"):
        env.pop(key, None)
    if kind in ("venv", "private"):
        env["PYTHONNOUSERSITE"] = "1"
        install = [interpreter(target), "-m", "pip", "install", "-q", "-r", str(lock)]
        if kind == "venv":
            return run_tracked(install, target.parent, 1800, env=env)
        # Every pin of the lock lands in the generation itself, the interpreter's own packages hidden while pip
        # decides what is installed; they stay visible underneath for what local functions import beside the lock.
        from ouroboros.runtime_venv import system_site

        system_site(target, False)
        try:
            return run_tracked(install, target.parent, 1800, env=env)
        finally:
            system_site(target, True)
    # Explicitly a user-site install into that user base (the platform layer chose --user for this
    # interpreter): the user site enabled (an empty value is unset for Python), no venv required.
    env.update(PYTHONUSERBASE=str(target), PYTHONNOUSERSITE="", PIP_REQUIRE_VIRTUALENV="false")
    return run_tracked([python or sys.executable, "-m", "pip", "install", "--user", "-q", "-r", str(lock)],
                       target.parent, 1800, env=env)


def _create_private(generation: pathlib.Path) -> str:
    """A bare interpreter's first private environment: a venv of exactly this interpreter with its own pip
    (``runtime_venv.create_with_pip``: ``ensurepip``, else this interpreter's own pip, else the lock's pip
    wheel). Where it cannot make one the preparation stops typed; the interpreter's own packages are never
    the fallback. Returns how the venv got its pip."""
    import sys

    from ouroboros.runtime_venv import RuntimeVenvRefused, create_with_pip
    from supervisor.update_checks import run_tracked

    env = _registry()._host_env(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    for key in ("PYTHONUSERBASE", "PIP_REQUIRE_VIRTUALENV", "PIP_TARGET", "PIP_PREFIX", "PIP_USER", "VIRTUAL_ENV",
                "PYTHONHOME"):
        env.pop(key, None)
    try:
        return create_with_pip(generation, sys.executable, lambda cmd: run_tracked(cmd, generation.parent, 600,
                                                                                   env=env))
    except RuntimeVenvRefused as exc:
        stopped()  # its steps were refused, not failed
        raise PrepareRefused("UPDATE_ENVIRONMENT_UNAVAILABLE", "the update changes the dependency lock, and this "
                             f"Python could not make a private environment for it: {str(exc)[-900:]}; let it reach "
                             "the package index (pip comes from there when it has none), or update by hand") from exc


def prepare_generation_environment(row: Dict[str, Any], composition: str,
                                   python: Optional[str] = None) -> Dict[str, Any]:
    """The new composition's packages beside the serving ones, never in them.

    Every generation runs with its packages at one path (``serving_environment``: the
    packaged user site, or a venv's root), the one every launcher starts the body with,
    so the new generation's copy is a sibling directory named for its candidate: a
    copy-on-write clone of the serving one where the volume offers one, then the new
    lock installed into the clone alone (with ``cp -c`` on APFS only the changed
    packages take space). Checks run it at that path, where everything pip wrote
    names it. The switch helper renames the two before the next generation imports
    anything and points a venv's console scripts at the path they then run from
    (``body_switch._select_environment``); a return renames them back. The serving
    copy stays beside as ``.previous-<adoption>`` until the update is adopted; the
    worktree GC removes an unused clone once its candidate is retired. A bare interpreter
    (``private``) gets a new venv of itself whose own pins are the whole lock, its own
    packages visible underneath, bound to this checkout; the package-init hook runs its
    entries there once it is selected. An interpreter without a seam refuses a lock
    change: nothing is ever prepared on the serving packages. ``python``: a new app's interpreter, which
    installs the lock into its user site and later runs it (``update_checks``). A venv's
    clone installs and runs with its own interpreter; when a new app is staged, the clone of a
    venv of the installed app's Python names the staged app's Python until its checks are done
    (``runtime_venv.point_base``; ``links`` keeps what ``_restore_venv`` writes back,
    ``staged_python`` what it ran), and one of the owner's own Python keeps it (``venv_python``).
    """
    import sys

    from ouroboros.body_adoption import dependency_lock
    from ouroboros.runtime_venv import base_interpreter

    kind, current = serving_environment()
    native = row.get("native") or {}
    # A new app replaces the launcher too; otherwise the running one must hand itself over to the new packages.
    refusal = "" if kind and native.get("status") == "staged" else (_no_seam() if not kind else launcher_refusal())
    if refusal:
        raise PrepareRefused("UPDATE_ENVIRONMENT_UNAVAILABLE", f"the update changes the dependency lock, and {refusal}")
    python = python if kind == "user_site" else None
    generation = current.with_name(f"{current.name}.update-{row['candidate_id']}")
    shutil.rmtree(generation, ignore_errors=True)
    cloned, pip = False, ""
    packages = package_state(current)  # what the switch replaces, read before the copy (activation reads it again)
    if kind == "private" and os.path.realpath(sys.prefix) != os.path.realpath(current):
        pip = _create_private(generation)  # a bare interpreter: the fixed path stays as it is until a selection
    else:
        current.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            from supervisor.update_checks import run_tracked

            cloned = run_tracked(["cp", "-c", "-R", str(current), str(generation)], current.parent, 600)[
                "returncode"] == 0
        if not cloned:
            stopped()  # a refused clone is no reason to copy instead
            shutil.rmtree(generation, ignore_errors=True)
            shutil.copytree(current, generation, symlinks=True)
        if package_state(current) != packages:  # a package writer ran during the copy: it may be torn
            shutil.rmtree(generation, ignore_errors=True)
            raise PrepareRefused("UPDATE_ENVIRONMENT_CHANGED", f"packages were installed or removed in {current} "
                                 "while it was copied; prepare the update again")
        if kind in ("venv", "private"):
            # Every console script of the copy, the unchanged ones too, runs the copy's own interpreter wherever
            # the checks start it; the selection points them at the fixed path (``body_switch._select_environment``).
            from ouroboros.body_switch import _retarget_scripts

            _retarget_scripts(str(generation), str(current), str(generation))
    links: Dict[str, str] = {}
    if kind == "venv" and native.get("status") == "staged" and not native.get("venv_python"):
        from ouroboros.runtime_venv import RuntimeVenvRefused, point_base

        try:
            links = point_base(generation, str(native["installed"]), str(native["staged_app"]))
        except (RuntimeVenvRefused, OSError) as exc:  # only the private clone was touched: it goes
            shutil.rmtree(generation, ignore_errors=True)
            raise PrepareRefused("UPDATE_ENVIRONMENT_UNLINKED" if isinstance(exc, RuntimeVenvRefused)
                                 else "UPDATE_ENVIRONMENT_FAILED", str(exc)) from exc
    workspace = pathlib.Path(row["path"])
    _run(workspace, "reset", "--hard", "--quiet", composition)
    selected = dependency_lock(row["repo_dir"], row["layers"]["new_files"])
    lock = workspace / selected["lock"] if selected["lock"] else None
    installed = _install_lock(generation, lock, kind, python) if lock else {"returncode": 0}
    if installed["returncode"] != 0:
        shutil.rmtree(generation, ignore_errors=True)
        stopped()  # pip was refused, not failed
        raise PrepareRefused("UPDATE_ENVIRONMENT_FAILED", "the new dependency lock did not install into the "
                             f"generation environment (rc={installed['returncode']}): "
                             + str(installed.get("stderr") or installed.get("stdout") or "")[-600:])
    if kind == "private":  # the checkout whose entries run in it once it is selected
        atomic_write_json(generation / PRIVATE_MARKER, {"schema_version": 1, "candidate_id": row["candidate_id"],
                                                        "repo_dir": os.path.realpath(row["repo_dir"])})
    return {"kind": kind, "user_base": str(generation), "current": str(current), "cloned": bool(cloned), **selected,
            "packages": packages, **({"python": python} if python else {}), **({"pip": pip} if pip else {}),
            **({"links": links, "staged_python": base_interpreter(generation)} if links else {})}


def _restore_venv(generation: Dict[str, Any]) -> str:
    """The clone names the installed app's Python again, exactly as before its checks (the exchange puts the
    staged app's bytes there): ``""``, or why it could not."""
    from ouroboros.runtime_venv import restore_base

    try:
        restore_base(str(generation.get("user_base") or ""), generation.get("links") or {})
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    generation.pop("links", None)
    return ""


# --------------------------------------------------------------------------- #
# Activation, discard and the boot side
# --------------------------------------------------------------------------- #
def environment_moved(row: Dict[str, Any]) -> str:
    """Why the interpreter or the packages this candidate was prepared on no longer serve, or ``""``.

    The serving runtime and dependency input are read again exactly as preparation recorded them
    (``environment_identity``). A candidate that switches packages in (``environment.generation``) is
    bound to where the serving ones live (``serving_environment``) and to their state when its copy was
    taken (``package_state``): a package installed there since, by the local model runtime's install
    or by a task, would otherwise be left behind at the switch. One that switches none keeps the
    serving packages as they are, so a later install there stays."""
    environment = row.get("environment") or {}
    recorded, base = environment.get("serving"), row.get("base") or {}
    if not isinstance(recorded, dict) or not base.get("files_tree"):
        return "the candidate recorded no serving environment"
    now = environment_identity(row["repo_dir"], base["files_tree"])
    changed = sorted(key for key in set(now) | set(recorded) if now.get(key) != recorded.get(key))
    if changed:
        return f"the Python that serves changed ({', '.join(changed)})"
    generation = environment.get("generation") or {}
    if not generation:
        return ""
    kind, current = serving_environment()
    if (kind, os.path.realpath(current)) != (generation.get("kind"), os.path.realpath(str(generation.get("current")))):
        return f"the packages serve from {current} ({kind or 'no seam'}) now"
    if package_state(current) != generation.get("packages"):
        return f"packages were installed or removed in {current} after the update was prepared"
    return ""


def activation_check(row: Dict[str, Any]) -> str:
    """Why this READY candidate may not be activated now (after the writer fence), or ``""``.

    The caller holds ``lifecycle()``. The base is read again: any change of HEAD, branch,
    index or files since the preparation makes the candidate ``stale:local_changed``.
    The owner's explicit change of the update channel makes it ``stale:channel_changed``;
    the channel merely moving forward does not (the candidate keeps its pinned target,
    and the panel says a newer one exists). A staged app recorded by an identity from before
    ``Contents`` was part of it (``body_switch.identity_stale``) makes it
    ``stale:native_identity_stale``: nothing proves those bundles any more. Another serving
    interpreter or dependency input, or packages installed where a prepared copy would replace
    them, make it ``stale:environment_changed`` (``environment_moved``).
    """
    from ouroboros.body_switch import identity_stale
    from ouroboros.update_channels import get_update_channel
    from supervisor import git_ops as g

    if row.get("state") != "ready":
        return f"the candidate is {row.get('state')}, not ready"
    native = row.get("native") or {}
    if (row.get("environment") or {}).get("generation") and native.get("status") != "staged" and launcher_refusal():
        return f"launcher_cannot_take_packages: {launcher_refusal()}"  # the candidate stays ready for that start
    if native.get("status") == "staged" and any(identity_stale(native.get(key))
                                                for key in ("previous_identity", "new_identity")):
        _update(row, state="stale", reason="native_identity_stale")
        return ("native_identity_stale: the update was prepared before the apps' contents were part of their "
                "identity; prepare it again")
    channel = get_update_channel()
    if str((row.get("target") or {}).get("channel") or "") != channel:
        _update(row, state="stale", reason="channel_changed")
        return f"channel_changed: the update channel is {channel} now; prepare the update again"
    snap = snapshot(row["repo_dir"], g.BRANCH_DEV)
    if snap["refusals"] or fingerprint(snap) != row.get("base"):
        _update(row, state="stale", reason="local_changed")
        return "local_changed: the checkout changed after the update was prepared; prepare it again"
    moved = environment_moved(row)
    if moved:
        _update(row, state="stale", reason=f"environment_changed: {moved}")
        return f"environment_changed: {moved}; prepare the update again"
    return ""


def mark(row: Dict[str, Any], **fields: Any) -> Dict[str, Any]:
    return _update(row, **fields)


def _stop_resolver(row: Dict[str, Any]) -> Dict[str, Any]:
    """Cancel the candidate's resolver through the task owner (``cancel_task_by_id``: the task and its live
    subtree, answered once torn down). ``confirmed`` False keeps the checkout for the GC to prove later."""
    task_id = str((row.get("resolver") or {}).get("task_id") or "")
    if not task_id:
        return {}
    try:
        from ouroboros.body_candidate import _recorded_terminal

        if _recorded_terminal(task_id, None):
            return {"task_id": task_id, "confirmed": True, "how": "already_ended"}
        from supervisor.task_lifecycle import cancel_task_by_id

        return {"task_id": task_id, "confirmed": bool(cancel_task_by_id(task_id)), "how": "cancelled"}
    except Exception as exc:
        log.warning("update resolver %s could not be cancelled", task_id, exc_info=True)
        return {"task_id": task_id, "confirmed": False, "how": f"{type(exc).__name__}: {exc}"}


def discard(row: Dict[str, Any]) -> Dict[str, Any]:
    """The owner's discard of exactly ``row``'s candidate, read again under ``lifecycle()``.

    Its resolver is cancelled first; the record settles; the GC retires checkout,
    scratch and generation environment once nothing works in them."""
    from ouroboros import body_adoption
    from supervisor import git_ops as g

    with _lifecycle:
        live = find(str(row.get("candidate_id") or ""))
        if live is None or live.get("state") in SETTLED:
            raise PrepareRefused("UPDATE_CANDIDATE_SETTLED", "this update candidate is already "
                                 + (str(live.get("state")) if live else "gone"))
        handoff = body_adoption.read(g.DRIVE_ROOT)
        if handoff.get("candidate_id") == live.get("candidate_id") and handoff.get("phase") in (
                "armed", "switching", "switched", "stuck"):
            raise PrepareRefused("UPDATE_ACTIVATION_PENDING", "this candidate is being activated")
        stopped = _stop_resolver(live)
        if body_adoption.close_update(g.DRIVE_ROOT, "candidate_discarded", str(live.get("candidate_id") or "")):
            raise PrepareRefused("UPDATE_ACTIVATION_PENDING", "this candidate is being activated")  # armed meanwhile
        return _settle(live, "discarded", reason="owner_discarded", **({"resolver_stop": stopped} if stopped else {}))


def record_activation(data_dir: Any, result: Dict[str, Any]) -> None:
    """The boot's settled activation outcome lands on its candidate (the panel's and the mind's record).

    A concluded outcome is also retained as the panel's Last attempt (``ACTIVATION_REL``): an
    activated record settles, and the GC then retires it with its checkout. ``unconfirmed`` (this
    start did not attest the switched tree; the handoff stays switched) is an observation, not an
    outcome: the record keeps activating, with what was observed beside its adoption id. A record
    names its latest attempt before that attempt's handoff exists (``body_adoption.authorize_update``)
    and keeps the name through its end, aborted or settled: a result of another handoff (an earlier
    one, settled late) changes nothing of it or of the Last attempt. A record or a result that names
    none takes the result as before.
    """
    with _lifecycle:
        row = find(str(result.get("candidate_id") or ""), data_dir)
        if row is None or row.get("state") in SETTLED:
            return
        named, settled = (str(value or "") for value in ((row.get("activation") or {}).get("adoption_id"),
                                                         result.get("adoption_id")))
        if named and settled and named != settled:
            log.info("update activation %s left %s as it is: that record is attempt %s's", settled,
                     row.get("candidate_id"), named)
            return
        outcome = str(result.get("outcome") or "")
        state = {"adopted": "activated", "returning": "activating", "unconfirmed": "activating"}.get(outcome, "failed")
        reason = f"activation:{outcome}"
        if outcome == "returned" and (result.get("decision") or {}).get("retry"):
            state = "ready"  # its packages could not be switched; everything returned, so Restart to update may retry
        if outcome == "not_applied":
            # Nothing was switched or exchanged: the candidate is offered again while what it was prepared on holds.
            moved = _prepared_moved(row)
            state, reason = ("stale", moved) if moved else ("ready", reason)
        activation = {key: result.get(key) for key in ("adoption_id", "outcome", "note", "reasons", "decision", "sha",
                                                       "previous", "why") if key in result}
        switched = result.get("environment") if isinstance(result.get("environment"), dict) else {}
        if outcome == "start_failed" and switched.get("state") == "selected" and switched.get("previous"):
            # The previous generation's own packages stay for a Restore while the Last attempt names them.
            activation["previous_environment"] = str(switched["previous"])
        if outcome == "unconfirmed":
            activation = {**(row.get("activation") or {}), **activation, "observed_at": utc_now_iso()}
        if state == "activated":
            _settle(row, "activated", data_dir, activation=activation, reason="")
            selected = (row.get("environment") or {}).get("generation") or {}
            if selected.get("kind") == "private":  # its entries need that venv from now on (dependency_python)
                retain_fact("private_environment", {"root": str(selected.get("current") or "")}, data_dir)
        else:
            _update(row, data_dir, state=state, activation=activation, reason=reason)
        if outcome not in ("returning", "unconfirmed"):  # a return under way, or a start yet to be confirmed, goes on
            _retain_activation(row, activation, data_dir)
    # The previous generation's packages, kept for a return the adoption has now closed, and any
    # other environment no record names any more: the next worktree GC pass reclaims them.


def _prepared_moved(row: Dict[str, Any]) -> str:
    """Why a candidate whose activation changed nothing cannot be activated as prepared any more, or ``""``: its
    workspace or prepared packages are gone, or its base or environment moved (``activation_check`` reads both
    again at the next activation)."""
    from supervisor import git_ops as g

    generation = (row.get("environment") or {}).get("generation") or {}
    if not pathlib.Path(str(row.get("path") or "")).is_dir():
        return "candidate_missing: its workspace is gone"
    if generation and not pathlib.Path(str(generation.get("user_base") or "")).is_dir():
        return "environment_missing: its prepared packages are gone"
    try:
        snap = snapshot(row["repo_dir"], g.BRANCH_DEV)
        if snap["refusals"] or fingerprint(snap) != row.get("base"):
            return "local_changed"
        moved = environment_moved(row)
    except Exception as exc:
        return f"local_unread: {type(exc).__name__}: {exc}"
    return f"environment_changed: {moved}" if moved else ""


def _retain_activation(row: Dict[str, Any], activation: Dict[str, Any], data_dir: Optional[Any] = None) -> None:
    """The activation's outcome with the candidate facts the Last attempt names: versions, the previous commit,
    what became of the local work and where work left out is kept. One value; never authority."""
    target, merge, native = row.get("target") or {}, row.get("merge") or {}, row.get("native") or {}
    layers = [(row.get("carried") or {}).get(layer) or {} for layer in ("index", "files")]
    # How many files stayed uncommitted, only where both layers were recorded in full; otherwise unknown (None).
    complete = bool(row.get("carried")) and all(
        isinstance(layer.get("count"), int) and layer["count"] == len(layer.get("paths") or []) for layer in layers)
    value = {
        "candidate_id": row.get("candidate_id"), "at": utc_now_iso(),
        **{key: activation[key] for key in ("outcome", "why", "reasons", "note", "sha", "previous",
                                            "previous_environment") if key in activation},
        "trigger": str((activation.get("decision") or {}).get("trigger") or ""),
        **({"detail": str(activation["decision"]["detail"])[:600]}
           if (activation.get("decision") or {}).get("detail") else {}),
        "target_sha": str(target.get("target_sha") or ""), "target_version": str(target.get("version") or ""),
        "base_version": str(row.get("base_version") or ""), "merge_kind": str(merge.get("kind") or ""),
        "local_commits": merge.get("local_commits"), "local_changes": bool(row.get("dirty")),
        "without_local_changes": bool(row.get("without_local_changes")),
        "carried_files": len({path for layer in layers for _status, path in layer.get("paths") or []})
        if complete else None,
        # The only copy of work an update without local changes left out: its refs outlive the record.
        **({"left_out": f"{row.get('refs') or ''}k2"} if row.get("without_local_changes") and row.get("dirty") else {}),
        **({"native": {key: native.get(key) for key in ("version", "previous_version", "backup_app")
                       if native.get(key)}} if native.get("status") == "staged" else {}),
    }
    try:
        with _retained_lock:
            path, saved = _retained(data_dir)
            # The one app backup kept for recovery (PERSISTENCE: at most one): a confirmed app adoption made its own,
            # so the one an earlier adoption left goes (``update_native.retire_backup``).
            kept = str(saved.get("backup_app") or "")
            made = str(native.get("backup_app") or "") if native.get("status") == "staged" else ""
            if made and activation.get("outcome") == "adopted":
                from supervisor.update_native import retire_backup

                if kept and kept != made and not retire_backup(kept, str(native.get("installed") or "")):
                    log.warning("the superseded app backup %s was not removed", kept)
                kept = made
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, {**value, **({"backup_app": kept} if kept else {}),
                                     **{name: saved[name] for name in KEEP_FACTS if name in saved}})
    except Exception:
        log.warning("the update activation outcome could not be retained for the panel", exc_info=True)


def _forget_activation(data_dir: Optional[Any] = None) -> None:
    """The next admitted attempt replaces the Last attempt; the app backup and the keep facts stay."""
    try:
        with _retained_lock:
            path, saved = _retained(data_dir)
            kept = {name: saved[name] for name in ("backup_app", *KEEP_FACTS) if saved.get(name)}
            if kept:
                atomic_write_json(path, kept)
            else:
                path.unlink(missing_ok=True)
    except Exception:
        log.warning("the retained update activation outcome could not be replaced", exc_info=True)


# The keep policy's boot facts (``git_ops_reset``), in the same record on every root: a providerless one has no
# supervisor state. ``_retained_lock``: one read-modify-write of ACTIVATION_REL at a time in this process.
KEEP_FACTS, _retained_lock = ("healthy_composition", "start_failure", "private_environment"), threading.Lock()


def _retained(data_dir: Optional[Any]) -> Tuple[pathlib.Path, Dict[str, Any]]:
    path = _registry()._data_dir(data_dir) / ACTIVATION_REL  # a rewrite's start, under ``_retained_lock``
    if (saved := read_json_dict(path)) is None and os.path.lexists(path):  # it may hold the backup or the selection
        raise OSError(f"{path} exists but cannot be read; it is kept as it is, never rewritten or deleted as empty")
    return path, saved or {}


def retained_fact(name: str, data_dir: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    value = (read_json_dict(_registry()._data_dir(data_dir) / ACTIVATION_REL) or {}).get(name)
    return value if isinstance(value, dict) else None


def retain_fact(name: str, value: Optional[Dict[str, Any]], data_dir: Optional[Any] = None) -> None:
    """Set one keep fact (``None`` clears it) beside the Last attempt; raises ``OSError`` when it cannot be written."""
    with _retained_lock:
        path, saved = _retained(data_dir)
        if saved.get(name) == value:
            return  # unchanged (clearing a fact that is not there writes nothing)
        saved = {**{key: item for key, item in saved.items() if key != name}, **({name: value} if value else {})}
        if saved:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, saved)
        else:
            path.unlink(missing_ok=True)


def reconcile_interrupted(data_dir: Optional[Any] = None) -> List[str]:
    """The boot's settlement of preparations their process left unfinished; returns the candidates it failed.

    A step runs on a thread of the serving process and names that process in its record (``worker``). A
    candidate still ``preparing`` or ``checking`` that no step of this process works in, whose worker provably
    ended (``local_custody_repair.owner_ended``: its pid gone, or another birth or custody session there), will
    never move on: it fails as interrupted, which offers preparing it again or discarding it. A live, unknown
    or unrecorded worker leaves the record as it is.
    """
    from ouroboros.local_custody_repair import owner_ended

    failed: List[str] = []
    with _lifecycle:
        try:
            rows = _rows(data_dir)
        except Exception:
            log.warning("interrupted update preparations were not reconciled: copy records unreadable", exc_info=True)
            return failed
        for row in rows:
            worker = row.get("worker") if isinstance(row.get("worker"), dict) else {}
            if row.get("state") not in _WORKING_STATES or row.get("candidate_id") in _working or not worker \
                    or not owner_ended(worker.get("pid"), worker.get("birth"), worker.get("proof")):
                continue
            try:
                _update(row, data_dir, state="failed", reason=(
                    f"UPDATE_PREPARE_INTERRUPTED: its preparation stopped with the process that ran it (pid "
                    f"{worker.get('pid')}) before it finished"))
            except PrepareRefused:
                continue  # settled or gone meanwhile: its own record stands
            failed.append(str(row.get("candidate_id") or ""))
    return failed


def _resolver_over(task_id: str, data_dir: Optional[Any], live) -> Optional[bool]:
    """True only on proof the resolver and its descendants ended; None while that is unknown."""
    from ouroboros.body_candidate import _recorded_terminal, live_lineage_processes

    if live is not None:
        answer = live(task_id)  # the supervisor's probe: True live, False proven absent, None unknown
        if answer is not False:
            return False if answer else None
    if not _recorded_terminal(task_id, data_dir):
        return None
    return not live_lineage_processes(data_dir, task_id)


def retention_verdict(row: Dict[str, Any], data_dir: Optional[Any] = None, live=None) -> Dict[str, Any]:
    """Whether the worktree GC may retire this candidate's checkout, scratch and generation environment.

    Only a settled record no open handoff names, that no step of this process works in,
    whose resolver (if any) is proven over: recorded terminal with no live lineage
    process and, where the GC supplies the supervisor's liveness probe, proven absent;
    and whose app installer, if it staged one, custody no longer holds alive. An unproven
    exit keeps the files; the next pass decides again.
    """
    from ouroboros import body_adoption

    if row.get("state") not in SETTLED:
        return {"remove": False, "reason": f"state:{row.get('state')}"}
    handoff = body_adoption.read(_registry()._data_dir(data_dir))
    if handoff.get("candidate_id") == row.get("candidate_id") and handoff.get("phase") not in ("", None, "closed"):
        return {"remove": False, "reason": f"handoff:{handoff.get('phase')}"}
    # No lifecycle lock here: the GC asks under this copy's record lock, which a transition takes after it.
    if row.get("candidate_id") in _working:  # one atomic read; only settled records are asked about
        return {"remove": False, "reason": "working"}
    task_id = str((row.get("resolver") or {}).get("task_id") or "")
    if task_id:
        over = _resolver_over(task_id, data_dir, live)
        if not over:
            return {"remove": False, "reason": f"resolver_{'live' if over is False else 'unconfirmed'}:{task_id}"}
    if (row.get("native") or {}).get("status") == "staged":  # its installer runs this checkout and staging
        from supervisor.update_native import installer_pending

        pending = installer_pending(_registry()._data_dir(data_dir))
        if pending:
            return {"remove": False, "reason": f"installer:{pending[0]}"}
    return {"remove": True, "reason": f"settled:{row.get('state')}"}


def release_refs(row: Dict[str, Any]) -> None:
    """Drop the settled candidate's private refs, except the layers an apply without local changes left
    behind; its checkout, scratch and generation environment stay for the GC."""
    repo = pathlib.Path(str(row.get("repo_dir") or ""))
    keep = ("k1", "k2") if row.get("without_local_changes") and row.get("state") == "activated" else ()
    for name in ("base", "k1", "k2", "c", "i_prime", "w_prime"):
        if name not in keep:
            _run(repo, "update-ref", "-d", f"{row.get('refs') or ''}{name}", check=False)


def _generation_path(row: Dict[str, Any]) -> Optional[pathlib.Path]:
    """The candidate's own generation environment, exactly as preparation named it, or ``None``."""
    generation = (row.get("environment") or {}).get("generation") or {}
    current, path = str(generation.get("current") or ""), str(generation.get("user_base") or "")
    if generation.get("kind") not in ("user_site", "venv", "private") or not current or not path:
        return None
    return pathlib.Path(path) if path == f"{current}.update-{row.get('candidate_id')}" else None


def release_environment(row: Dict[str, Any], data_dir: Optional[Any] = None) -> None:
    """After the GC retired a copy that is an update candidate: delete its package staging, the list of the work it
    carried and its generation environment, unless an open handoff selected or still names that environment (an
    adopted generation was renamed into place)."""
    from ouroboros import body_adoption
    from supervisor.update_native import release_staging

    if row.get("kind") != KIND:
        return

    release_staging(row.get("native") or {})
    try:
        if row.get("path"):
            carried_list_path(row).unlink(missing_ok=True)
    except OSError:
        log.warning("update candidate %s kept the list of its carried work", row.get("candidate_id"), exc_info=True)
    path = _generation_path(row)
    if path is None or not path.exists():
        return
    named = body_adoption.read(_registry()._data_dir(data_dir)).get("environment") or {}
    if str(path) in (str(named.get("generation") or ""), str(named.get("previous") or "")):
        return
    shutil.rmtree(path, ignore_errors=True)


def reclaim_environments(data_dir: Optional[Any] = None, rows: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """Delete package environments beside the serving one that nothing names any more.

    ``<serving>.update-<candidate>`` belongs to its candidate record while that exists;
    ``<serving>.previous-<adoption>`` to the open handoff that may return to it. A
    preparation that crashed, a candidate retired before its environment, a previous
    generation after its adoption closed or after a refused return: no record names
    them, and they are each a full package set (a clone shares unchanged files on APFS).
    ``rows`` (the caller's listing) only proposes; ownership is read again before a deletion.
    """
    try:
        return _reclaim_environments(data_dir, rows)
    except Exception:
        log.warning("update environment reclaim failed; the next pass retries", exc_info=True)
        return []


def _environment_owners(data_dir: Optional[Any], entries: List[Dict[str, Any]]) -> Tuple[set, set, set]:
    """``(named, currents, live)``: the paths the records and an open handoff name, the serving paths beside
    which generations live, and the candidates whose clone may exist before a record names it."""
    from ouroboros import body_adoption

    named, currents = set(), set()
    # A clone exists before its record names it (cloning and pip take minutes): its name carries its
    # candidate, so every unsettled candidate and every one a step of this process works in keeps it.
    live = set(_working)
    for entry in entries:
        generation = (entry.get("environment") or {}).get("generation") or {}
        named.add(str(generation.get("user_base") or ""))
        currents.add(str(generation.get("current") or ""))
        if entry.get("state") not in SETTLED:
            live.add(str(entry.get("candidate_id") or ""))
    handoff = body_adoption.read(_registry()._data_dir(data_dir))
    if handoff.get("phase") not in ("", None, "closed"):
        environment = handoff.get("environment") or {}
        named.update(str(environment.get(key) or "") for key in ("generation", "previous", "current"))
    # After a start that was not returned, the Last attempt names the previous generation's packages for a Restore
    # until the next admitted preparation replaces it (``_forget_activation``); an unreadable record raises: no pass.
    named.add(str(_retained(data_dir)[1].get("previous_environment") or ""))
    return named, currents, live


def _owned(sibling: pathlib.Path, base: str, named: set, live: set) -> bool:
    return str(sibling) in named or sibling.name in {f"{base}.update-{cid}" for cid in live if cid}


def _reclaim_environments(data_dir: Optional[Any], rows: Optional[List[Dict[str, Any]]]) -> List[str]:
    # ``rows``: the copy records the caller already listed (the GC pass keeps its listing budget).
    named, currents, live = _environment_owners(
        data_dir, _rows(data_dir) if rows is None else [row for row in rows if row.get("kind") == KIND])
    try:
        kind, serving = serving_environment()
        if kind:
            currents.add(str(serving))
    except Exception:
        log.debug("serving environment unreadable for reclaim", exc_info=True)
    doomed = []
    for current in sorted(path for path in currents if path):
        base = pathlib.Path(current)
        try:
            siblings = list(base.parent.iterdir())
        except OSError:
            continue
        doomed += [(sibling, base.name) for sibling in siblings
                   if sibling.name.startswith((base.name + ".update-", base.name + ".previous-"))
                   and not _owned(sibling, base.name, named, live) and sibling.is_dir() and not sibling.is_symlink()]
    if not doomed:
        return []
    # That view may predate a preparation that created, or even finished, its generation since. Each deletion is
    # decided again on the records, the handoff and this process's steps as they are now, under the lock every
    # record transition takes. An unowned generation regains no owner: a record exists before its clone, a new
    # candidate or adoption gets a fresh random id, and a settled record is never written again nor starts a step.
    # So the deletions themselves run outside the lock.
    with _lifecycle:
        named, _currents, live = _environment_owners(data_dir, _rows(data_dir))
        doomed = [(sibling, base) for sibling, base in doomed if not _owned(sibling, base, named, live)]
    for sibling, _base in doomed:
        shutil.rmtree(sibling, ignore_errors=True)  # Windows keeps a file a surviving process still maps
    return [str(sibling) for sibling, _base in doomed if not os.path.lexists(sibling)]  # the rest: the next pass


# --------------------------------------------------------------------------- #
# Status projection
# --------------------------------------------------------------------------- #
def _app_need_now() -> Dict[str, Any]:
    """The installed app's need for the serving commit (local facts only), for the panel; ``{}`` when none.

    Without it a current source under an app that cannot host it would read as up to date;
    preparation decides again from the composition itself.
    """
    try:
        from supervisor import git_ops as g
        from supervisor.update_native import app_need

        need = app_need(pathlib.Path(g.REPO_DIR), "HEAD")
    except Exception:
        log.debug("app need unreadable for the update panel", exc_info=True)
        return {}
    return {key: need.get(key) for key in ("status", "reason", "required", "installed_version")
            if need.get(key) is not None} if need.get("status") == "needed" else {}


def _actions(row: Dict[str, Any]) -> List[str]:
    """What the owner may do with this candidate in its state: the projection's offer and the admission's rule."""
    actions = {"preparing": [], "checking": [], "activating": [], "adapting": ["discard_candidate"],
               "ready": ["activate", "discard_candidate"],
               "needs_adaptation": ["apply_without_local_changes", "discard_candidate", "prepare"],
               "needs_decision": ["apply_without_local_changes", "discard_candidate", "prepare"],
               "paused_budget": ["prepare", "apply_without_local_changes", "discard_candidate"],
               "failed": ["prepare", "apply_without_local_changes", "discard_candidate"],
               "stale": ["prepare", "discard_candidate"]}.get(str(row.get("state") or ""), ["discard_candidate"])
    # Without local changes still needs the merge C, so it cannot answer a conflict of the merge itself.
    if row.get("without_local_changes") or not row.get("dirty") or str(row.get("reason") or "") == "merge_conflicts":
        actions = [action for action in actions if action != "apply_without_local_changes"]
    return list(actions)


# The candidate's app facts the panel names: the versions, where the app is and where the replaced one stays,
# what checked the package (its SHA-256 against the release's capsule, an official signature), the --runtime-venv
# the new app starts its body in, and the package manager that installed this app with the version it last did.
_NATIVE_FACTS = ("status", "reason", "version", "instruction", "previous_version", "installed", "backup_app",
                 "sha256", "origin", "runtime_venv", "manager", "manager_version")


def projection(latest_sha: str = "", data_dir: Optional[Any] = None) -> Dict[str, Any]:
    """The candidate as ``/api/update/status`` shows it: facts and the actions its state allows."""
    retained = read_json_dict(_registry()._data_dir(data_dir) / ACTIVATION_REL) or {}
    failure = retained.get("start_failure")  # this checkout did not import at its last start
    last = {key: value for key, value in retained.items() if key not in KEEP_FACTS}  # until the next admission
    recovery = {**({"start_failure": failure} if isinstance(failure, dict) else {}),
                **({"last_activation": last} if last.get("outcome") else {})}
    try:
        row = current(data_dir)
    except Exception as exc:
        return {"state": "unknown", "error": f"{type(exc).__name__}: {exc}", **recovery}
    if row is None:
        running = _running.locked()
        need = {} if running else _app_need_now()
        return {"state": "preparing" if running else "none",
                **({"refusal": dict(_last_refusal)} if _last_refusal and not running else {}),
                **({"app_needed": need} if need else {}),
                "actions": [] if running else ["prepare"], **recovery}
    target = dict(row.get("target") or {})
    merge = dict(row.get("merge") or {})
    resolver = row.get("resolver") or {}
    native = row.get("native") or {}
    # A conflict still open names its layer by its step: the merge (this installation's commits), the staged
    # changes or the uncommitted files. A reviewed resolution closes it.
    conflicts = list(resolver.get("conflicts") or merge.get("conflicts") or [])
    step = "" if resolver.get("resolution") else str(resolver.get("step") or ("merge" if merge.get("conflicts") else ""))
    return {
        "state": str(row.get("state") or "unknown"), "candidate_id": row.get("candidate_id"),
        "reason": str(row.get("reason") or ""),
        "target_sha": target.get("target_sha", ""), "target_ref": target.get("target_ref", ""),
        "target_version": str(target.get("version") or ""), "base_version": str(row.get("base_version") or ""),
        "channel": target.get("channel", ""), "base_sha": str(row.get("base_sha") or ""),
        "local_changes": bool(row.get("dirty")), "without_local_changes": bool(row.get("without_local_changes")),
        "merge_kind": merge.get("kind", ""), "local_commits": merge.get("local_commits"),
        "carried": row.get("carried") or {},
        "conflict_paths": conflicts[:20], "conflict_count": len(conflicts),
        "layer_conflicts": {step: conflicts[:20]} if step and conflicts else {},
        "environment_same": (row.get("environment") or {}).get("same"),
        "checks": row.get("checks") or {}, "activation": row.get("activation") or {},
        "newer_available": bool(latest_sha and target.get("target_sha") and latest_sha != target["target_sha"]),
        "adaptation": {key: resolver.get(key) for key in ("step", "task_id", "resolution") if resolver.get(key)},
        "native": {key: native.get(key) for key in _NATIVE_FACTS if native.get(key)},
        "question": str(row.get("question") or ""),
        "updated_at": row.get("updated_at") or "", "actions": _actions(row), **recovery,
    }
