"""Restart-bound adoption of ONE exact reviewed body-candidate commit.

Adoption is deliberate and never automatic: a plain restart, a crash or a boot adopts nothing, and a found candidate
authorizes nothing. The mind names the exact commit on ``request_restart``; this module carries that one transition.

* ``authorize`` (old generation, tree untouched) checks the candidate against the serving checkout, pins the commit,
  captures the stdlib switch helper OUTSIDE the tree and writes ``state/body_adoption/handoff.json``. Pins are set
  and removed in one ref transaction through the server's command admission (``_refs``): none once its stop began.
* ``bind_restart`` re-verifies under the existing update/restart serialization and marks the one restart that may arm
  it; a refused restart abandons it.
* ``arm`` (exit tail, after the existing stop owners) publishes the Git-dir pointer only on those owners' own evidence
  that the old generation's subprocess readers are gone: the worker pool's exit census (each worker process handle
  confirmed dead), the owned-work stop's completed outcome and no live child process handle. It probes nothing
  itself. Exit or exec then ends the in-process readers. Panic and the owner's Restart never arm and never wait.
  It holds the helper's lock, as every retirement of the record does (``_archive``) and every creation of one
  (``_new_helper``), whose folder and lock file outlive each handoff: one outcome, never a pointer to a missing helper.
* The switch itself is ``body_switch``, run by the package-init hook of the next cold entry before any other body
  module is imported.
* ``finalize_on_boot`` records ``adopted`` only for a generation that is actually ready and that verified, before its
  own imports, that it was loading exactly that commit (``body_switch._attest_loaded``), with native-host proof where
  one applies. A later read of HEAD is not that proof.
* ``settle_on_boot`` gives every outcome to the task that authorized it: its task result keeps ``body_adoptions`` and
  its card gets the row in the task's own room (a room it cannot name gets none); only a handoff that names no task
  stays an owner-wide runtime notice.

``authorize_update`` carries a prepared managed Update through the same transition (``kind=update``, schema 2): the
branch moves from the serving HEAD to the merge commit, while the index and the files take the carried layers of the
owner's staged and unstaged work, so one switch row names three sides. Its restart is the activation the owner asked
for; it arms on the same stop evidence, which the generic owner Restart never supplies. A successor that is not ready
asks ``body_switch.qualify_return``; only ``compatible`` evidence returns the previous generation, and the data root
is never rolled back. An Update names no task, and its every outcome goes to its candidate record
(``update_prepare.record_activation``), never to a task card or an owner-wide notice.
"""

from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import shutil
import subprocess
import uuid
from typing import Any, Dict, List, Optional

from ouroboros import body_switch
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

HOOK_MARKER = "_body_adoption_hook"
PIN_PREFIX = "refs/ouroboros/adoption/"
# Files Python parses whole before the hook can run; they must stay present and hooked.
_HOOK_FILE = "ouroboros/__init__.py"
_REQUIRED_FILES = ("server.py", _HOOK_FILE, "ouroboros/body_switch.py")
_DEPENDENCY_FILES = ("requirements-runtime.lock", "requirements.txt")
_ENGINE_PIN_FILE = "ouroboros/claudexor_runtime_pin.json"
_PENDING = ("authorized", "armed", "switching", "switched", "stuck")
_HELPERS = ("switch.py", "installation_stop.py", "app_install.py")  # what a creation copies beside its record
_NOT_ARMED = "not armed: "  # the exit tail's own record of the stop owners' blockers
UPDATE_REASON = "managed_update_activation"
# The new tree's import surface, probed after the switch with a scratch data root
# (the boot's own import test plus the pre-restart smoke's server surface).
UPDATE_PROBE_IMPORTS = ("import server, ouroboros.agent, ouroboros.gateway.router, supervisor.queue, "
                        "supervisor.events, ouroboros.tools.registry")
# Pins that keep every side of an Update switch reachable until the handoff closes.
_LAYER_PINS = {"index": "new_index", "files": "new_files", "old-index": "old_index", "old-files": "old_files"}
ADOPTIONS_KEY = "body_adoptions"  # the authorizing task's settled adoptions, every one, on its task result
_INTERVAL_KEPT = 64  # an adopted range names its newest commits; Git keeps the whole range


class AdoptionRefused(Exception):
    """A typed refusal; the serving tree and any earlier handoff are untouched."""

    def __init__(self, code: str, text: str):
        super().__init__(text)
        self.code, self.text = code, text


def helper_dir(data_dir: Any) -> pathlib.Path:
    return pathlib.Path(data_dir) / "state" / "body_adoption"


def read(data_dir: Any) -> Dict[str, Any]:
    """The current handoff, or ``{}`` when none exists or it cannot be read."""
    try:
        handoff = body_switch.read_handoff(str(helper_dir(data_dir)))
        return handoff if isinstance(handoff, dict) else {}
    except (OSError, ValueError):
        return {}


def _git(repo: Any, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, encoding="utf-8", check=check)


def _refs(repo: Any, *lines: str) -> None:
    """ONE ref transaction, through the command admission (``update_prepare._run``): none once the stop began."""
    from supervisor.update_prepare import PrepareRefused, _run
    try:
        _run(repo, "update-ref", "--stdin", data="".join(f"{line}\n" for line in lines).encode())
    except PrepareRefused as exc:
        raise AdoptionRefused("ADOPTION_STOPPED", exc.text) from None


def _log(data_dir: Any, kind: str, handoff: Dict[str, Any], **fields: Any) -> None:
    try:
        append_jsonl(pathlib.Path(data_dir) / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": kind, "adoption_id": handoff.get("id"),
            "candidate": handoff.get("cand"), "base": handoff.get("old"), **fields})
    except Exception:
        log.debug("body adoption event append failed", exc_info=True)


def _switch_rows(repo: pathlib.Path, old: str, cand: str) -> List[Dict[str, Any]]:
    raw = subprocess.run(["git", "diff-tree", "-r", "--no-renames", "-z", old, cand], cwd=str(repo),
                         capture_output=True, check=True).stdout.split(b"\0")
    rows: List[Dict[str, Any]] = []
    for meta, path in zip(raw[0::2], raw[1::2]):
        old_mode, new_mode, old_sha, new_sha, _status = meta.decode()[1:].split(" ")
        rows.append({
            "path": path.decode("utf-8", "surrogateescape"),
            "old_mode": old_mode, "new_mode": new_mode,
            "old_sha": None if not old_sha.strip("0") else old_sha,
            "new_sha": None if not new_sha.strip("0") else new_sha,
        })
    return rows


def _blob_has(repo: pathlib.Path, commit: str, path: str, needle: str = "") -> bool:
    shown = _git(repo, "show", f"{commit}:{path}", check=False)
    return shown.returncode == 0 and needle in shown.stdout


def _dependency_command(repo: pathlib.Path, rows: List[Dict[str, Any]]) -> Optional[List[str]]:
    """The owning sync's exact command, only when the switch changes what it installs."""
    if not {row["path"] for row in rows} & set(_DEPENDENCY_FILES):
        return None
    from supervisor.git_ops_reset import runtime_dependency_command

    return runtime_dependency_command(repo)


def authorize(ctx: Any, commit: str, *, reason: str) -> Dict[str, Any]:
    """Authorize adopting the bound candidate's exact ``commit`` at the restart named ``reason``."""
    from ouroboros import body_candidate

    if not body_candidate.is_bound(ctx):
        raise AdoptionRefused("ADOPTION_NO_CANDIDATE", "this task is not bound to a body candidate.")
    row = body_candidate.find(body_candidate.owner_id(ctx))
    if row is None:
        raise AdoptionRefused("ADOPTION_NO_CANDIDATE", "this task's candidate has no registry row.")
    from ouroboros.config import get_runtime_mode
    from ouroboros.consciousness_authority import effective_runtime_mode
    from ouroboros.tool_access_paths import canonical_data_root

    mode = effective_runtime_mode(get_runtime_mode(), getattr(ctx, "task_metadata", None))
    return authorize_row(canonical_data_root(ctx), row, commit, reason=reason, runtime_mode=mode,
                         task_id=str(getattr(ctx, "task_id", "") or ""), ctx=ctx)


def authorize_row(data_dir: pathlib.Path, row: Dict[str, Any], commit: str, *, reason: str,
                  runtime_mode: str = "advanced", task_id: str = "", ctx: Any = None) -> Dict[str, Any]:
    """The checks and the durable handoff; shared by the tool and the evolution supervisor path."""
    from ouroboros import body_candidate

    serving, candidate = pathlib.Path(str(row["repo_dir"])), pathlib.Path(str(row["path"]))
    resolved = _git(candidate, "rev-parse", "--verify", f"{str(commit).strip()}^{{commit}}", check=False)
    if resolved.returncode != 0 or not str(commit).strip():
        raise AdoptionRefused("ADOPTION_COMMIT_UNKNOWN", f"{commit!r} is not a commit of the candidate.")
    cand = resolved.stdout.strip()
    if _git(candidate, "merge-base", "--is-ancestor", cand, "HEAD", check=False).returncode != 0:
        raise AdoptionRefused("ADOPTION_COMMIT_UNKNOWN", "the commit is not on this candidate's branch.")
    old = _git(serving, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
    branch = _git(serving, "symbolic-ref", "-q", "--short", "HEAD", check=False).stdout.strip()
    if not branch:
        raise AdoptionRefused("ADOPTION_DETACHED", "the serving checkout has a detached HEAD; adoption moves a branch.")
    base = str(row.get("base_sha") or "")
    based = (old == base or _git(serving, "merge-base", "--is-ancestor", base, old, check=False).returncode == 0)
    if cand == old:
        raise AdoptionRefused("ADOPTION_NOTHING_TO_ADOPT", "the serving checkout is already at this commit.")
    if not based or _git(serving, "merge-base", "--is-ancestor", old, cand, check=False).returncode != 0:
        raise AdoptionRefused(
            "ADOPTION_BASE_MOVED",
            f"the serving checkout is at {old[:12]}, which this commit does not build on (candidate base "
            f"{base[:12]}). Merge or rebase the candidate onto the serving HEAD and review that result.")
    unreviewed = body_candidate.unreviewed_commits({**row, "base_sha": old}, cand)
    if unreviewed and runtime_mode != "cyber_pro":
        raise AdoptionRefused(
            "ADOPTION_UNREVIEWED",
            "these commits did not come through commit_reviewed under the configured enforcement: "
            + ", ".join(sha[:12] for sha in unreviewed))
    rows = _switch_rows(serving, old, cand)
    paths = [item["path"] for item in rows]
    if not rows:
        raise AdoptionRefused("ADOPTION_NOTHING_TO_ADOPT", "the commit changes no file relative to the serving tree.")
    if any("160000" in (item["old_mode"], item["new_mode"]) for item in rows):
        raise AdoptionRefused("ADOPTION_UNSUPPORTED", "the change moves a submodule link; adopt it through an update.")
    if not all(_blob_has(serving, cand, path) for path in _REQUIRED_FILES) \
            or not all(_blob_has(serving, sha, _HOOK_FILE, HOOK_MARKER) for sha in (old, cand)):
        raise AdoptionRefused(
            "ADOPTION_HOOK_MISSING",
            "both the serving body and the candidate must carry the package-init adoption hook and the "
            "entry files; a body without it receives the hook through an ordinary release first.")
    if "BIBLE.md" in paths and "VERSION" not in paths and runtime_mode != "cyber_pro":
        raise AdoptionRefused(
            "ADOPTION_CONSTITUTION_NEEDS_RELEASE",
            "a constitutional change takes effect only through an explicit reviewed release (BIBLE preamble): "
            "make it a numbered release, or deliver it as a contribution.")
    dirty = body_switch._dirty(str(serving), paths)  # the helper's own bounded check of the switch set
    if dirty:
        raise AdoptionRefused(
            "ADOPTION_SWITCH_SET_DIRTY",
            "the serving checkout has later work at paths this commit changes; it is kept as it is: "
            + dirty.splitlines()[0])
    from supervisor.update_merge import active_update_tx

    if active_update_tx():
        raise AdoptionRefused("ADOPTION_UPDATE_ACTIVE", "a managed update owns the checkout; adopt after it settles.")
    current = read(data_dir)
    if current.get("phase") in _PENDING and current.get("phase") != "authorized":
        raise AdoptionRefused(
            "ADOPTION_PENDING", f"adoption {current.get('id')} is {current.get('phase')}; it must settle first.")
    from ouroboros.preflight_qualification import adoption_test_evidence

    # Existing green evidence of exactly this commit; adoption never runs the battery itself. A
    # selected run may qualify only a reviewed delta (Cyber Pro's unreviewed commits keep full).
    try:  # an unreadable declared widening leaves full evidence only
        local = None if unreviewed else {"onto": old, "paths": paths,
                                         "widening": body_candidate.interval_widening(row, old, cand)}
    except (OSError, subprocess.SubprocessError):
        local = None
    try:
        qualification = adoption_test_evidence(data_dir, candidate, cand, ctx=ctx, local=local)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        raise AdoptionRefused("ADOPTION_TESTS_REQUIRED", str(exc)) from exc
    handoff = {
        "schema": 1, "id": uuid.uuid4().hex[:12], "phase": "authorized", "repo_dir": str(serving.resolve()),
        "data_dir": str(pathlib.Path(data_dir).resolve()), "branch": branch, "old": old, "cand": cand,
        "candidate_id": row.get("candidate_id"), "candidate_branch": row.get("branch"),
        "candidate_path": str(candidate), "task_id": task_id, "reason": str(reason or ""),
        "created_tag": str((row.get("reviewed_release_tags") or {}).get(cand) or ""),
        "switch": rows, "deps_command": _dependency_command(serving, rows), "authorized_at": utc_now_iso(),
        "unreviewed_commits": unreviewed, "attempts": 0, "events": [],
        "tests": qualification,
    }
    directory = helper_dir(data_dir)
    with _new_helper(data_dir, current):
        # The helper is captured from the body that is running, never from the candidate.
        shutil.copyfile(pathlib.Path(body_switch.__file__), directory / "switch.py")
        from ouroboros import installation_stop
        shutil.copyfile(pathlib.Path(installation_stop.__file__), directory / "installation_stop.py")
        _refs(serving, f"update {PIN_PREFIX}{handoff['id']} {cand}")
        body_switch.record(str(directory), handoff, "authorized", f"restart reason: {reason}")
    _log(data_dir, "body_adoption_authorized", handoff, branch=branch, paths=len(rows))
    return handoff


def _tree_map(repo: pathlib.Path, tree: str) -> Dict[str, tuple]:
    """``path -> (mode, object id)`` for every entry of ``tree`` (blobs, links and gitlinks)."""
    out = subprocess.run(["git", "ls-tree", "-r", "-z", "--full-tree", tree], cwd=str(repo),
                         capture_output=True, check=True).stdout
    entries: Dict[str, tuple] = {}
    for chunk in out.split(b"\0"):
        if chunk:
            meta, _tab, path = chunk.partition(b"\t")
            mode, _kind, sha = meta.decode().split(" ")
            entries[path.decode("utf-8", "surrogateescape")] = (mode, sha)
    return entries


def update_switch_rows(repo: Any, layers: Dict[str, str]) -> List[Dict[str, Any]]:
    """Schema 2 rows: every path any layer moves, with its commit, index and file sides.

    ``layers`` names six trees (or commits): ``old_commit``/``new_commit`` (H and C),
    ``old_index``/``new_index`` (I and I′), ``old_files``/``new_files`` (W and W′).
    The commit side keeps the schema 1 field names; an absent side is ``None``.
    """
    pairs = (("", "old_commit", "new_commit"), ("_index", "old_index", "new_index"),
             ("_file", "old_files", "new_files"))
    maps = {name: _tree_map(pathlib.Path(repo), layers[name]) for _suffix, old, new in pairs for name in (old, new)}
    paths = sorted({path for _suffix, old, new in pairs for path in maps[old].keys() | maps[new].keys()
                    if maps[old].get(path) != maps[new].get(path)})
    rows: List[Dict[str, Any]] = []
    for path in paths:
        row: Dict[str, Any] = {"path": path}
        for suffix, old, new in pairs:
            for side, name in (("old", old), ("new", new)):
                mode, sha = maps[name].get(path, ("000000", None))
                row[f"{side}{suffix}_sha"], row[f"{side}{suffix}_mode"] = sha, mode
        rows.append(row)
    return rows


def _side_changed(rows: List[Dict[str, Any]], paths) -> bool:
    return any(row["path"] in paths and (row["old_file_sha"], row["old_file_mode"]) != (
        row["new_file_sha"], row["new_file_mode"]) for row in rows)


def _served_web_root(serving: pathlib.Path) -> str:
    """The checkout-relative folder the gateway serves as its static web root, ``""`` when outside it."""
    from ouroboros.server_web import resolve_web_dir

    try:
        return resolve_web_dir(serving).resolve().relative_to(serving.resolve()).as_posix()
    except (OSError, ValueError):
        return ""


def dependency_lock(repo: Any, tree: str) -> Dict[str, str]:
    """The dependency input a checkout of ``tree`` installs from and its exact bytes (read by ``update_prepare._run``).

    The launcher installs the first of ``launcher_bootstrap.DEPENDENCY_INPUTS`` that exists; preparation (whether
    a generation environment is needed) and authorization (whether one must be switched in) compare exactly this,
    so an edit of a file the launcher does not install from changes neither.
    """
    import hashlib

    from ouroboros.launcher_bootstrap import DEPENDENCY_INPUTS
    from supervisor.update_prepare import _run

    for name in DEPENDENCY_INPUTS:
        shown = _run(repo, "cat-file", "blob", f"{tree}:{name}", check=False)
        if shown.returncode == 0:
            return {"lock": name, "lock_sha256": hashlib.sha256(shown.stdout).hexdigest()}
    return {"lock": "", "lock_sha256": hashlib.sha256(b"").hexdigest()}


def authorize_update(data_dir: Any, candidate: Dict[str, Any], *, reason: str = UPDATE_REASON) -> Dict[str, Any]:
    """Authorize switching the serving checkout to a READY Update candidate at its activation restart.

    ``candidate`` is the preparation's record (``supervisor.update_prepare``): the serving checkout and six pinned
    layers. The caller stopped the writers and re-read the base fingerprint; this re-verifies HEAD and branch and that
    every switch path still holds its recorded old file and index sides, so a late edit is never overwritten. The old
    generation does not write the tree here; the pins come first, so after this server's stop began nothing changes.
    The candidate's record names this attempt once what it replaces is gone and before its handoff exists, so no end of
    this process leaves a handoff its record does not name, or a name beside an earlier handoff still live.
    """
    serving = pathlib.Path(str(candidate.get("repo_dir") or ""))
    layers = dict(candidate.get("layers") or {})
    if any(not layers.get(name) for name in ("old_commit", "new_commit", "old_index", "new_index",
                                              "old_files", "new_files")):
        raise AdoptionRefused("ADOPTION_NOTHING_TO_ADOPT", "the update candidate has no complete prepared layers.")
    staged = dict(candidate.get("native") or {}) if (candidate.get("native") or {}).get("status") == "staged" else {}
    old = _git(serving, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
    branch = _git(serving, "symbolic-ref", "-q", "--short", "HEAD", check=False).stdout.strip()
    cand = layers["new_commit"]
    if not branch:
        raise AdoptionRefused("ADOPTION_DETACHED", "the serving checkout has a detached HEAD; an update moves a branch.")
    if old != layers["old_commit"] or _git(serving, "merge-base", "--is-ancestor", old, cand,
                                           check=False).returncode != 0:
        raise AdoptionRefused("ADOPTION_BASE_MOVED", f"the serving checkout is at {old[:12]}, not the prepared base "
                                                     f"{layers['old_commit'][:12]}; prepare the update again.")
    rows = update_switch_rows(serving, layers)
    if not rows and not staged:  # an app-only Update keeps H, I and W exactly and changes the app alone
        raise AdoptionRefused("ADOPTION_NOTHING_TO_ADOPT", "the prepared update changes nothing in this checkout.")
    if any("160000" in (row[f"{side}{suffix}_mode"] for side in ("old", "new")) for row in rows
           for suffix in ("", "_index", "_file")):
        raise AdoptionRefused("ADOPTION_UNSUPPORTED", "the update moves a submodule link; apply it by hand.")
    if not all(_blob_has(serving, layers["new_files"], path) for path in _REQUIRED_FILES) or not all(
            _blob_has(serving, layers[name], _HOOK_FILE, HOOK_MARKER) for name in ("old_files", "new_files")):
        raise AdoptionRefused("ADOPTION_HOOK_MISSING", "both the serving files and the prepared files must carry the "
                                                       "package-init adoption hook and the entry files.")
    tracked = body_switch._tracks_filemode(str(serving))
    files, index = body_switch._states(str(serving), rows, tracked), body_switch._index_states(str(serving), rows)
    moved = [row["path"] for row in rows if files[row["path"]] != body_switch._side(row, "old", tracked)
             or index[row["path"]] != body_switch._index_side(row, "old")]
    if moved:
        raise AdoptionRefused("ADOPTION_SWITCH_SET_DIRTY", "local work changed after the update was prepared, at "
                                                           f"{moved[0]}; it is kept as it is. Prepare the update again.")
    from supervisor.update_merge import active_update_tx

    if active_update_tx():
        raise AdoptionRefused("ADOPTION_UPDATE_ACTIVE", "a legacy managed update owns the checkout; let it settle first.")
    current = read(data_dir)
    if current.get("phase") in _PENDING and current.get("phase") != "authorized":
        raise AdoptionRefused(
            "ADOPTION_PENDING", f"adoption {current.get('id')} is {current.get('phase')}; it must settle first.")
    import sys

    # The launcher's effective dependency input, exactly as preparation compared it.
    dependencies_same = dependency_lock(serving, layers["old_files"]) == dependency_lock(serving, layers["new_files"])
    adoption_id = uuid.uuid4().hex[:12]
    generation = (candidate.get("environment") or {}).get("generation") or {}
    environment = None
    if not dependencies_same or staged:
        # The new lock, or the new app (whose frozen bootstrap installs packages before any hooked
        # entry), runs on its own prepared packages, switched in before it starts.
        # By the token preparation wrote into it (``body_switch.mark_generation``): a directory made again at that
        # path, or the generation emptied in place, may keep its device and inode, never the token; a reboot since
        # may renumber the device of the very generation, so its identity is read again here for the switch.
        prepared = generation.get("identity") or []
        if len(prepared) < 3 or not body_switch._is_prepared(str(generation.get("user_base") or ""), prepared):
            raise AdoptionRefused("ADOPTION_ENVIRONMENT_MISSING", "the update needs its own prepared package "
                                                                  "environment and it is gone; prepare it again.")
        environment = {"state": "prepared", "kind": generation.get("kind") or "", "current": generation["current"],
                       "generation": generation["user_base"],
                       "previous": f"{generation['current']}.previous-{adoption_id}",
                       # What is where now: the switch resumes and returns by these, never by names alone.
                       "identity": {"current": body_switch._path_identity(generation["current"]),
                                    "generation": body_switch._path_identity(generation["user_base"])
                                    + [prepared[2]]},
                       # A bare interpreter's private environment: the interpreter the previous generation runs on.
                       **({"python": sys.executable} if generation.get("kind") == "private" else {}),
                       # A venv of the installed app's Python: the staged app's, which it runs after the exchange.
                       **({"staged_python": generation["staged_python"]}
                          if staged and generation.get("staged_python") else {})}
    native = _native_record(staged, serving.resolve(), pathlib.Path(data_dir).resolve()) if staged else None
    _refs(serving, f"update {PIN_PREFIX}{adoption_id} {cand}",
          *(f"update {PIN_PREFIX}{adoption_id}-{name} {layers[layer]}" for name, layer in _LAYER_PINS.items()))
    try:
        from ouroboros.runtime_venv import interpreter

        # The new tree's import probe runs on the interpreter that will run it: the new app's, when one comes, or the
        # --runtime-venv's own path, which the new app keeps. Before the exchange that path still names the previous
        # app's Python, so the switch runs it on the environment's staged_python (body_switch._probe_command).
        venv = str((native or {}).get("runtime_venv") or "")
        python = (interpreter(venv) if venv else staged.get("python")) if staged else (
            None if getattr(sys, "frozen", False) else sys.executable)
        if environment and environment["kind"] == "private":  # a bare interpreter runs the new tree in its own packages
            python = interpreter(environment["current"])
        probe = [python, "-c", UPDATE_PROBE_IMPORTS + "; print('probe_ok')"] if python else None
        handoff = {
            "schema": 2, "kind": "update", "id": adoption_id, "phase": "authorized",
            "repo_dir": str(serving.resolve()), "data_dir": str(pathlib.Path(data_dir).resolve()), "branch": branch,
            "old": old, "cand": cand, "candidate_id": candidate.get("candidate_id"),
            "candidate_branch": candidate.get("branch"), "candidate_path": str(candidate.get("path") or ""),
            "task_id": "", "reason": str(reason or ""), "created_tag": "", "switch": rows, "layers": layers,
            "deps_command": None, "probe_command": probe, "environment": environment, "native": native,
            "authorized_at": utc_now_iso(), "unreviewed_commits": [], "attempts": 0, "events": [], "entries": [],
            "tests": dict(candidate.get("checks") or {}), "target": dict(candidate.get("target") or {}),
            "qualification": {"runtime": body_switch.runtime_identity(), "dependencies_same": dependencies_same,
                              "engine_pin_same": not _side_changed(rows, (_ENGINE_PIN_FILE,)),
                              # What a return after an entry needs (body_switch.qualify_return): the root the
                              # gateway serves stylesheets from, and whether this root held an owned engine at all.
                              "web_root": _served_web_root(serving),
                              "owned_engine": body_switch.owned_engine(str(pathlib.Path(data_dir).resolve()))},
        }
        directory = helper_dir(data_dir)
        with _new_helper(data_dir, current):
            shutil.copyfile(pathlib.Path(body_switch.__file__), directory / "switch.py")
            from ouroboros import installation_stop
            shutil.copyfile(pathlib.Path(installation_stop.__file__), directory / "installation_stop.py")
            if native:  # the installer the new app's signed entry runs: macOS only (it needs ``fcntl``): read only here
                from ouroboros import app_install

                shutil.copyfile(pathlib.Path(app_install.__file__), directory / "app_install.py")
            from supervisor import update_prepare

            update_prepare.mark(candidate, data_dir=data_dir, activation={"adoption_id": adoption_id})
            body_switch.record(str(directory), handoff, "authorized", f"restart reason: {reason}")
    except Exception:
        _release_pins(serving, {"id": adoption_id, "kind": "update"})  # no record names them: nothing else would
        raise
    _log(data_dir, "body_adoption_authorized", handoff, branch=branch, paths=len(rows), adoption_kind="update")
    return handoff


def _native_record(staged: Dict[str, Any], repo_dir: pathlib.Path, data_dir: pathlib.Path) -> Dict[str, Any]:
    """The app transition inside its activation record: what the installer stops, exchanges and opens.

    Both bundles are named by identity (device and inode, which a rename on one volume keeps), so
    an interrupted exchange is read from the disk; the generation that stops is named by pid and
    start time; the launch (``update_native.successor_launch``) carries this installation's
    configuration and launch inputs, never the stopping generation's process role, and the
    launcher's single-instance lock is named. The ``--runtime-venv`` the body runs in must be the
    one the packages were prepared beside (or none on both sides): the launch carries it, and the
    readback the venv's own Python when it is not the app's (``venv_python``).
    """
    from ouroboros import config
    from ouroboros.platform_layer import process_start_time_legacy
    from supervisor.update_native import launcher_pid, selected_runtime_venv, server_pid, successor_launch

    launcher = launcher_pid()
    if not launcher:
        raise AdoptionRefused("ADOPTION_NATIVE_UNMANAGED", "replacing the app needs the app's own launcher, and this "
                                                          "server was not started by it.")
    venv = selected_runtime_venv()
    if venv != str(staged.get("runtime_venv") or ""):
        raise AdoptionRefused("ADOPTION_NATIVE_CHANGED", "this installation was started with another --runtime-venv "
                                                        "than the update was prepared for; prepare it again.")
    installed, new = str(staged["installed"]), str(staged["staged_app"])
    previous_identity = body_switch.bundle_identity(installed)
    if not previous_identity or previous_identity != staged.get("previous_identity") or not staged.get(
            "new_identity") or body_switch.bundle_identity(new) != staged["new_identity"]:
        raise AdoptionRefused("ADOPTION_NATIVE_CHANGED", "the installed or the staged app changed after the update "
                                                        "was prepared; prepare it again.")
    return {"installed": installed, "staged": new, "backup": str(staged["backup_app"]),
            "version": str(staged["version"]), "previous_version": str(staged.get("previous_version") or ""),
            "previous_identity": previous_identity, "new_identity": staged["new_identity"],
            "python": str(staged["python"]), "runtime": dict(staged.get("runtime") or {}),
            **({"runtime_venv": venv} if venv else {}),
            **{key: staged[key] for key in ("manager", "manager_version", "venv_python") if staged.get(key)},
            "launch": successor_launch(data_dir, repo_dir), "pid_file": str(config.PID_FILE),
            "previous": {name: {"pid": pid, "start_time": process_start_time_legacy(pid)}
                         for name, pid in (("launcher", launcher), ("server", server_pid()))},
            "state": "prepared", "steps": []}


def bind_installer(data_dir: Any, adoption_id: str, pid: int) -> bool:
    """Name the app installer's exact process in its transition, once custody holds it.

    The installer acts only after it finds itself named here, and an entry or the holding
    server abandons the transition only once that process is provably gone.
    """
    from ouroboros.platform_layer import process_start_time_legacy

    directory = str(helper_dir(data_dir))
    with body_switch._locked(directory):
        handoff = read(data_dir)
        if handoff.get("id") != adoption_id or handoff.get("phase") != "authorized" or not handoff.get("native"):
            return False
        handoff["native"]["installer"] = {"pid": int(pid), "start_time": process_start_time_legacy(int(pid))}
        body_switch.record(directory, handoff, "authorized", f"app installer pid {pid} in custody")
    return True


def native_hold(data_dir: Any) -> str:
    """What the activating server does after its exit tail when its Update replaces the app.

    ``"hold"``: the transition is armed for a live app installer, so this server keeps the exit
    boundary until the old app's own Quit stops it through the launcher pipe: the launcher never
    recycles it (exit 42) into a generation the installer would race. ``"recycle"``: nothing is
    armed for an installer (another Update, a refused arm, an abandoned transition), so the ordinary
    restart exit follows; an installer that is provably gone is abandoned first, nothing switched.
    ``"panic"``: Panic is recorded.
    """
    handoff = read(data_dir)
    if handoff.get("kind") != "update" or not handoff.get("native"):
        return "recycle"
    if body_switch.installation_panic(data_dir):
        return "panic"
    if handoff.get("phase") != "armed":
        return "recycle"
    if not body_switch.process_gone(handoff["native"].get("installer")):
        return "hold"
    directory = str(helper_dir(data_dir))
    with body_switch._locked(directory):
        live = read(data_dir)
        if live.get("id") == handoff.get("id"):
            body_switch.native_abandon(str(live.get("repo_dir") or ""), directory, "installer_gone: nothing was switched")
    return "recycle"


def _native_readback(handoff: Dict[str, Any], repo_dir: Any) -> Dict[str, Any]:
    """What this ready generation actually runs under, against the app transition it settles.

    The installed path holds the new bundle; this interpreter is that app's, with the runtime
    preparation checked, or, when the launch carried a ``--runtime-venv``, that venv's, whose
    interpreter resolves into the new app, or to exactly the owner's own Python it ran on
    (``venv_python``); its launcher is this checkout's source launcher run by that app's signed
    entry; the generation environment is selected; the roots are the transition's, and so is every
    configuration and profile path its launch carried. A healthy HTTP answer proves none of this,
    and neither does a skipped external-host smoke.
    """
    import sys

    from ouroboros.config import resolve_data_dir, resolve_repo_dir
    from ouroboros.platform_layer import process_command
    from supervisor.update_native import configuration_differs, launch_configuration, selected_runtime_venv

    native = handoff.get("native") or {}
    installed = os.path.realpath(str(native.get("installed") or ""))
    failed = []
    if body_switch.identity_stale(native.get("new_identity")):
        failed.append("app_identity_stale")  # recorded before Contents was part of it: it proves no bundle
    elif body_switch.bundle_identity(installed) != native.get("new_identity"):
        failed.append("installed_app_is_not_the_new_one")
    # The path the interpreter was started by (the launcher's), walked, not resolved: an app's own Python. A venv's
    # interpreter is its own path, resolving to the app's Python where the venv points (the installed path), or to
    # the owner's own Python, by its real path, where the venv was made on that.
    executable, venv = os.path.abspath(sys.executable), str(native.get("runtime_venv") or "")
    own = str(native.get("venv_python") or "") if venv else ""
    in_app = tuple(path + os.sep for path in (str(native.get("installed") or ""), installed))
    if venv and os.path.realpath(selected_runtime_venv() or os.sep) != os.path.realpath(venv):
        failed.append("runtime_venv_not_selected")  # the new app started the body without the owner's venv
    elif own and os.path.realpath(executable) != own:
        failed.append("interpreter_is_not_the_venvs_own")  # the owner's Python is kept, never another one
    elif not own and not (os.path.realpath(executable) if venv else executable).startswith(in_app):
        failed.append("interpreter_outside_the_app")
    expected = {key: (native.get("runtime") or {}).get(key) for key in ("version", "sqlite")}
    if {key: body_switch.runtime_identity().get(key) for key in expected} != expected:
        failed.append("runtime_differs_from_the_prepared_one")
    managed = os.environ.get("OUROBOROS_MANAGED_BY_LAUNCHER") == "1"
    launcher = process_command(os.getppid()) if managed else ""
    entries = tuple(os.path.join(path, "Contents", "MacOS", "ouroboros-shell") + " "
                    for path in (str(native.get("installed") or ""), installed))
    sources = {os.path.join(str(path), "launcher.py") for path in (repo_dir, os.path.realpath(str(repo_dir)))}
    if not launcher.startswith(entries) or not any(source in launcher for source in sources):
        failed.append("launcher_is_not_the_signed_source_entry")
    if (handoff.get("environment") or {}).get("state") != "selected":
        failed.append("environment_not_selected")
    if (os.path.realpath(resolve_data_dir()), os.path.realpath(resolve_repo_dir())) != (
            os.path.realpath(str(handoff.get("data_dir") or "")), os.path.realpath(str(handoff.get("repo_dir") or ""))):
        failed.append("roots_differ")
    carried = (native.get("launch") or {}).get("env") or {}
    now = launch_configuration(str(handoff.get("data_dir") or ""), str(handoff.get("repo_dir") or "")) if carried else {}
    differs = configuration_differs(carried, now)
    if differs:
        failed.append("configuration_differs:" + ",".join(differs))
    return {"ok": not failed, "failed": failed, "launcher": launcher[:400], "executable": executable}


def bind_update_restart(data_dir: Any) -> bool:
    """The activation restart binds the authorized Update; a generic owner Restart never does."""
    handoff = read(data_dir)
    if handoff.get("kind") != "update" or handoff.get("phase") != "authorized":
        return False
    handoff["restart_bound"] = True
    body_switch.record(str(helper_dir(data_dir)), handoff, "authorized", "bound to the activation restart")
    return True


def authorized_base(data_dir: Any, candidate_sha: str, reason: str = "") -> str:
    """The serving commit an AUTHORIZED handoff for exactly ``candidate_sha`` expects, or ``""``."""
    handoff = read(data_dir)
    if (handoff.get("phase") == "authorized" and candidate_sha and handoff.get("cand") == candidate_sha
            and (not reason or handoff.get("reason") == reason)):
        return str(handoff.get("old") or "")
    return ""


def authorize_for_task(data_dir: Any, task_id: str, commit_sha: str, reason: str) -> bool:
    """Supervisor path (evolution's own restart): may a restart for ``commit_sha`` proceed?

    A commit made in the serving checkout needs no adoption. A commit on the
    task's candidate needs an authorized handoff; one the task already authorized
    for this restart is reused, otherwise it is authorized here.
    """
    try:
        from ouroboros import body_candidate

        row = body_candidate.find(str(task_id), data_dir)
        if row is None or authorized_base(data_dir, commit_sha, reason):
            return True
        if _git(row["repo_dir"], "rev-parse", "HEAD", check=False).stdout.strip() == commit_sha:
            return True
        from ouroboros.config import get_runtime_mode

        authorize_row(pathlib.Path(data_dir), row, commit_sha, reason=reason,
                      runtime_mode=get_runtime_mode(), task_id=str(task_id))
        return True
    except AdoptionRefused as exc:
        log.warning("Evolution restart not requested: adoption refused (%s: %s)", exc.code, exc.text)
    except Exception:
        log.warning("Evolution restart not requested: adoption could not be authorized", exc_info=True)
    return False


@contextlib.contextmanager
def _new_helper(data_dir: Any, current: Dict[str, Any]):
    """A new handoff's creation in the helper's folder, under its lock until the caller has written its helpers and
    its record: no retirement and no other creation acts in between. The lock file comes before any record (a full disk
    cannot then stop its retirement), and the folder and that file outlive every handoff (``_retire``). What ``current``
    named is retired first, unless it was armed since then: that record is never overwritten."""
    directory = helper_dir(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with body_switch._locked(str(directory)):
        if current:
            _retire(data_dir, current, f"replaced:{current.get('phase')}")
        armed = read(data_dir)
        if armed.get("phase") in _PENDING:
            raise AdoptionRefused("ADOPTION_PENDING",
                                  f"adoption {armed.get('id')} was armed meanwhile; let it settle first.")
        yield


def _release_pins(repo: Any, handoff: Dict[str, Any]) -> None:
    try:  # once this server's stop began the pins stay (they only keep objects); the record closes all the same
        _refs(repo, *(f"delete {PIN_PREFIX}{handoff['id']}{suffix}" for suffix in (
            "", *(f"-{name}" for name in _LAYER_PINS if handoff.get("kind") == "update"))))
    except Exception:
        log.warning("body adoption pins were kept", exc_info=True)


def _archive(data_dir: Any, handoff: Dict[str, Any], outcome: str) -> bool:
    """Retire the record the caller read: pointer and pin removed, the record kept under ``logs/``.

    Under the helper's lock, which ``arm`` holds while it decides and publishes and every creation while it writes
    (``_new_helper``): one armed since then stays, and nothing of a later handoff is touched. A lock that cannot be
    taken removes nothing and raises. Returns whether that record is closed (``_retire``).
    """
    with body_switch._locked(str(helper_dir(data_dir))):
        return _retire(data_dir, handoff, outcome)


def _retire(data_dir: Any, handoff: Dict[str, Any], outcome: str) -> bool:
    """``_archive`` with the helper's lock held. The record and this handoff's helpers go while it is held; the folder,
    its lock file and anything else in it stay, so no removal outlives the lock to reach a later handoff.

    True: the record the caller read is closed, here or by an earlier retirement (another record, or none, is there
    now). False: it moved on since (armed, say) or cannot be read, and stays as it is.
    """
    directory = helper_dir(data_dir)
    repo = pathlib.Path(str(handoff.get("repo_dir") or "")) if handoff else None
    pointer_gone = True
    live = read(data_dir)
    if (live.get("id"), live.get("phase")) != (handoff.get("id"), handoff.get("phase")):
        return live.get("id") != handoff.get("id") if live else not (directory / body_switch.HANDOFF_NAME).exists()
    if repo is not None and repo.is_dir():
        try:
            pathlib.Path(body_switch.pointer_path(str(repo))).unlink(missing_ok=True)
        except Exception:
            pointer_gone = False
            log.warning("body adoption pointer could not be removed", exc_info=True)
        _release_pins(repo, handoff)
    if handoff:
        _log(data_dir, "body_adoption_closed", handoff, outcome=outcome, phase=handoff.get("phase"),
             events=handoff.get("events"), interpreter_state=handoff.get("interpreter_state"),
             tests=handoff.get("tests"))
    if handoff and not pointer_gone:
        # The hook still finds the pointer: it must read a CLOSED record, never a missing helper
        # (which stops every boot).
        body_switch.record(str(directory), handoff, "closed", outcome)
        return True
    (directory / body_switch.HANDOFF_NAME).unlink(missing_ok=True)
    for name in _HELPERS:  # inert without the record; one that cannot go is overwritten by the next creation
        with contextlib.suppress(OSError):
            (directory / name).unlink(missing_ok=True)
    return True


def abandon(data_dir: Any, reason: str, *, task_id: str = "", candidate_id: str = "") -> None:
    """Close an unarmed handoff, scoped to the caller's identity when supplied. One that cannot be closed now stays for
    the next boot; ``close_update`` says what the record holds."""
    handoff = read(data_dir)
    if (handoff.get("phase") == "authorized"
            and (not task_id or handoff.get("task_id") == task_id)
            and (not candidate_id or handoff.get("candidate_id") == candidate_id)):
        try:
            _archive(data_dir, handoff, f"abandoned:{reason}")
        except OSError:
            log.warning("Body adoption %s could not be closed", handoff.get("id"), exc_info=True)


def close_update(data_dir: Any, reason: str, candidate_id: str) -> str:
    """Abandon this candidate's unarmed handoff, then read what the record holds: ``""`` once nothing of it can arm or
    switch; otherwise what still can (its phase, or why the record cannot be read), whatever the abandonment said."""
    try:
        abandon(data_dir, reason, candidate_id=candidate_id)
    except Exception:
        log.warning("Update handoff of %s could not be closed", candidate_id, exc_info=True)
    try:
        handoff = body_switch.read_handoff(str(helper_dir(data_dir)))
    except FileNotFoundError:
        return ""
    except (OSError, ValueError) as exc:
        return f"unreadable: {type(exc).__name__}: {exc}"
    if not isinstance(handoff, dict):
        return "unreadable: not a record"
    ours = handoff.get("candidate_id") == candidate_id and handoff.get("phase") in _PENDING
    return str(handoff["phase"]) if ours else ""


def bind_restart(safe_restart_fn, data_dir: Any, reason: str):
    """Wrap the restart's checkout step so the adoption is re-verified under the SAME serialization.

    Only the restart whose own receipt names this adoption binds it: the handoff's
    reason, and ``pending_restart_verify.json`` expecting exactly its candidate
    commit (a plain restart rewrites that receipt for the serving commit). A
    refused restart, or a serving tree that no longer matches, abandons it.
    """
    def run(**kwargs):
        ok, msg = safe_restart_fn(**kwargs)
        handoff = read(data_dir)
        from ouroboros.utils import read_json_dict

        receipt = read_json_dict(pathlib.Path(data_dir) / "state" / "pending_restart_verify.json") or {}
        if (handoff.get("phase") != "authorized" or handoff.get("reason") != reason
                or receipt.get("reason") != reason or receipt.get("expected_sha") != handoff.get("cand")):
            return ok, msg
        repo = pathlib.Path(str(handoff["repo_dir"]))
        if ok:
            head = _git(repo, "rev-parse", "HEAD", check=False).stdout.strip()
            branch = _git(repo, "symbolic-ref", "-q", "--short", "HEAD", check=False).stdout.strip()
            if head != handoff["old"] or branch != handoff["branch"]:
                ok, msg = False, f"the serving checkout moved to {head[:12]} on {branch or 'a detached HEAD'}"
        if not ok:
            abandon(data_dir, f"restart_refused: {msg}")
            return False, f"adoption of {str(handoff['cand'])[:12]} was not armed: {msg}"
        handoff["restart_bound"] = True
        body_switch.record(str(helper_dir(data_dir)), handoff, "authorized", "bound to this restart")
        return ok, msg
    return run


def arm(data_dir: Any, *, worker_exits: Any, live_children: Any, owned_stop: Any, owner_restart: bool) -> str:
    """Publish the pointer on the existing stop owners' evidence that the old readers are gone.

    ``worker_exits`` is the pool's own exit census of the kill that just ran
    (``supervisor.workers.last_worker_exit_census``): ``None`` when that kill did
    not complete. ``live_children`` are the child process handles still alive
    after the exit tail reaped them. ``owned_stop`` is ``stop_owned_work``'s
    outcome. Nothing is probed, waited for or guessed here.

    It decides and publishes under the helper's lock, without waiting for it. With another holder (a retirement of this
    record, ``_archive``, whatever its Git step takes) nothing is armed: that holder's outcome stands and the next boot
    settles what is left. So this record is either armed and stays, or goes with no pointer behind it.

    Never raises and never waits. Returns the resulting phase ("" when no handoff).
    """
    try:
        handoff = read(data_dir)
        if handoff.get("phase") != "authorized" or not handoff.get("restart_bound"):
            return str(handoff.get("phase") or "")
        directory = str(helper_dir(data_dir))
        with body_switch._locked(directory, wait=False):
            handoff = read(data_dir)  # the read that counts: the one no retirement can overtake
            if handoff.get("phase") != "authorized" or not handoff.get("restart_bound"):
                return str(handoff.get("phase") or "")
            stop = owned_stop if isinstance(owned_stop, dict) else {}
            census = worker_exits if isinstance(worker_exits, dict) else None
            blockers = [name for name, blocked in (
                # The Update's own activation is the owner's restart; it still arms only on the stop evidence below.
                ("owner_restart", owner_restart and handoff.get("kind") != "update"),
                ("worker_exits_unconfirmed", census is None or bool(census.get("unconfirmed"))),
                ("child_processes_alive", bool(live_children)),
                ("owned_work_unconfirmed", stop.get("state") != "completed" or bool(stop.get("unconfirmed"))),
                ("panic_stop", body_switch.installation_panic(data_dir)),
            ) if blocked]
            if blockers:
                handoff["restart_bound"] = False
                body_switch.record(directory, handoff, "authorized", _NOT_ARMED + ", ".join(blockers))
                _log(data_dir, "body_adoption_not_armed", handoff, blockers=blockers)
                return "authorized"
            handoff["readers_evidence"] = {"worker_exits": census, "owned_stop": {
                key: stop.get(key) for key in ("state", "targets", "confirmed")}}
            pointer = pathlib.Path(body_switch.pointer_path(str(handoff["repo_dir"])))
            temp = pointer.with_name(pointer.name + ".tmp")
            temp.write_text(directory + "\n", encoding="utf-8")
            os.replace(temp, pointer)
            body_switch.record(directory, handoff, "armed", "old generation's subprocess readers confirmed stopped")
            _log(data_dir, "body_adoption_armed", handoff)
            return "armed"
    except BlockingIOError:
        _log(data_dir, "body_adoption_not_armed", handoff, blockers=["handoff_busy"])
        return "authorized"
    except Exception:
        log.critical("Body adoption could not be armed; the next generation boots the unchanged tree",
                     exc_info=True)
        return ""


def bound_candidate_file(data_dir: Any, rel_path: str) -> Optional[bytes]:
    """``rel_path`` as the NEXT generation will read it, when this restart carries a bound adoption
    that changes it; ``None`` otherwise (the landed checkout's own file is then the answer).

    For an exit-tail owner whose contract is "what the landed checkout selects" (the engine pin):
    an adoption lands after this generation's exit, so its candidate commit is that checkout.
    """
    handoff = read(data_dir)
    if handoff.get("phase") != "authorized" or not handoff.get("restart_bound"):
        return None
    row = next((row for row in handoff.get("switch") or [] if row.get("path") == rel_path), None)
    if row is None:
        return None
    if "new_file_sha" in row:  # an Update: the next generation reads its FILE side, local edits included
        if not row["new_file_sha"]:
            return None
        shown = subprocess.run(["git", "cat-file", "blob", row["new_file_sha"]], cwd=str(handoff["repo_dir"]),
                               capture_output=True)
    else:
        shown = subprocess.run(["git", "show", f"{handoff['cand']}:{rel_path}"], cwd=str(handoff["repo_dir"]),
                               capture_output=True)
    return shown.stdout if shown.returncode == 0 else None


def holds_checkout(data_dir: Any, repo_dir: Any) -> bool:
    """This boot inherits an adoption transition of THIS checkout: the generic bootstrap reset must not run.

    True for the tree a switch just landed (``switched``), and equally for a
    transition that was returned (``abandoned``) or is still open: its tree may
    hold the owner's later work that the helper deliberately kept, and a reset
    here would be exactly the rescue-then-overwrite the transition refused. An
    ``authorized`` handoff never touched the tree and holds nothing.
    """
    handoff = read(data_dir)
    if handoff.get("phase") not in ("armed", "switching", "stuck", "switched", "abandoned", "returned"):
        return False
    try:
        return pathlib.Path(str(handoff.get("repo_dir"))).resolve() == pathlib.Path(repo_dir).resolve()
    except OSError:
        return False


def _loaded_generation(handoff: Dict[str, Any]) -> Dict[str, Any]:
    """What THIS process verified before its imports (``body_switch._attest_loaded``), for this handoff."""
    import sys

    fact = getattr(sys, "_ouroboros_body_generation", None)
    return fact if isinstance(fact, dict) and fact.get("adoption_id") == handoff.get("id") else {}


def _update_not_ready(data_dir: Any, repo_dir: Any, handoff: Dict[str, Any], why: str,
                      request_restart) -> Dict[str, Any]:
    """A successor of an Update that reached its boot but is not ready: return only on compatible evidence."""
    cand, old = str(handoff.get("cand") or "")[:12], str(handoff.get("old") or "")[:12]
    outcome, reasons = body_switch.qualify_return(str(repo_dir), handoff)
    base = {"kind": "update", "candidate_id": handoff.get("candidate_id"), "adoption_id": handoff.get("id"),
            "sha": handoff.get("cand"), "previous": handoff.get("old"), "reasons": reasons, "why": why,
            "environment": handoff.get("environment")}
    if outcome == "compatible" and request_restart is not None:
        with body_switch._locked(str(helper_dir(data_dir))):
            live = read(data_dir)
            if live.get("id") == handoff.get("id") and live.get("phase") == "switched":
                live["return_requested"] = {"why": why, "at": utc_now_iso()}
                body_switch.write_handoff(str(helper_dir(data_dir)), live)
        request_restart()
        return {**base, "outcome": "returning",
                "note": f"The update to {cand} is not ready ({why}); the previous version {old} is restored at "
                        "this restart because its compatibility with the current data is proven."}
    handoff["return"] = {"outcome": outcome, "trigger": why, "reasons": reasons, "at": utc_now_iso()}
    if not _archive(data_dir, handoff, f"start_failed:{outcome}"):
        return {}  # it moved on since this boot read it: what holds it now settles it
    return {**base, "outcome": "start_failed",
            "note": f"The update to {cand} is in place, but this start is not ready ({why}). The previous version "
                    f"{old} was not restored automatically ({', '.join(reasons) or outcome}). Data and local changes "
                    "were kept; Restore in Dashboard > Updates chooses a version explicitly."}


def _card_row(handoff: Dict[str, Any]) -> Dict[str, Any]:
    """The last wait row this handoff kept for its task's card (``_waiting``); ``{}`` before any."""
    row = handoff.get("card_row")
    return row if isinstance(row, dict) and type(row.get("revision")) is int else {}


def _settled(handoff: Dict[str, Any], outcome: str, note: str, **facts: Any) -> Dict[str, Any]:
    """One settled outcome in words, with the identity of the handoff it settles: the task that
    authorized it (``""`` for a handoff that names none), the candidate commit and the serving one.

    ``card_row_revision`` orders the outcome's row on that task's card. A settled end follows the
    last wait its handoff kept (``_waiting``), so no task-result write, made or lost, decides it."""
    kept = _card_row(handoff).get("revision", 0)
    return {"outcome": outcome, "note": note, "adoption_id": handoff.get("id"),
            "task_id": str(handoff.get("task_id") or ""), "candidate": str(handoff.get("cand") or ""),
            "previous": handoff.get("old"), "branch": str(handoff.get("branch") or ""),
            "card_row_revision": kept if outcome == "unconfirmed" else kept + 1, **facts}


def _waiting(data_dir: Any, handoff: Dict[str, Any], note: str, **facts: Any) -> Dict[str, Any]:
    """An unconfirmed boot; the handoff stays for the next one. Words the card row does not already say
    take the next revision, kept on the handoff before anything is sent, so every later row orders
    after it. A handoff that cannot keep it fails this settlement, which then sends nothing."""
    row = _card_row(handoff)
    if row.get("note") != note:
        handoff["card_row"] = {"revision": row.get("revision", 0) + 1, "note": note}
        body_switch.write_handoff(str(helper_dir(data_dir)), handoff)
    return _settled(handoff, "unconfirmed", note, **facts)


def finalize_on_boot(data_dir: Any, repo_dir: Any, *, supervisor_ready: bool,
                     native_smoke=None, request_restart=None, providerless_ready=None) -> Dict[str, Any]:
    """Settle the handoff this generation inherited; returns ``_settled`` facts, or an Update's outcome for its
    candidate record (``{}`` if none).

    ``adopted`` needs THIS generation ready on exactly the candidate commit and,
    where a native host is selected, its readback. A boot that is not ready
    leaves ``switched`` in place for the existing recovery owners; an Update's
    unready successor asks ``body_switch.qualify_return`` instead.
    ``providerless_ready`` (a generation without a provider runs no supervisor and
    no bootstrap) observes its readiness, its recovery's: it stands in for an Update,
    whose own switch probe was the tree's import test, never for a body adoption.
    An outcome is said only for a handoff its retirement closed (``_archive``): one that
    moved on since this boot read it (armed by a restart meanwhile, say) settles nothing
    here and stays for what holds it now and the boot after it.
    """
    handoff = read(data_dir)
    phase = handoff.get("phase")
    if not handoff or phase in ("armed", "switching", "stuck"):
        return {}
    cand = str(handoff.get("cand") or "")[:12]
    # An Update's every outcome reaches its candidate record, never the owner's chat (``settle_on_boot``).
    update = ({"kind": "update", "candidate_id": handoff.get("candidate_id"), "adoption_id": handoff.get("id")}
              if handoff.get("kind") == "update" else {})
    if phase == "returned":
        decision = handoff.get("return") or {}
        detail = str(decision.get("detail") or decision.get("trigger") or "it did not start")
        if not _archive(data_dir, handoff, f"returned:{decision.get('trigger') or 'unknown'}"):
            return {}
        # Why it did not start: a requested return carries the unready start's own reason.
        why = str((handoff.get("return_requested") or {}).get("why") or decision.get("trigger") or "")
        return {"outcome": "returned", "kind": "update", "candidate_id": handoff.get("candidate_id"),
                "adoption_id": handoff.get("id"), "sha": handoff.get("old"), "previous": handoff.get("cand"),
                "decision": decision, "why": why,
                "note": f"The update to {cand} did not start ({detail}). The previous version "
                        f"{str(handoff.get('old') or '')[:12]} was written back with its staged and unstaged local "
                        "changes; data was not rolled back."}
    if phase == "authorized":  # found at boot: the generation that authorized it is gone
        # Only the exit tail's blockers explain it; the other events quote the caller's reason or mark the binding.
        last = str((handoff.get("events") or [{}])[-1].get("detail") or "")
        why = last if last.startswith(_NOT_ARMED) else "the adoption was not armed before this boot"
        if not _archive(data_dir, handoff, "abandoned:not_armed"):
            return {}  # armed since this boot read it: that restart switches it and the next boot says so
        note = f"Adoption of {cand} was not applied ({why}). The serving body is unchanged; the candidate is retained."
        return {**update, "outcome": "not_applied", "note": note} if update else _settled(handoff, "not_applied", note)
    if phase == "abandoned":
        last = (handoff.get("events") or [{}])[-1].get("detail") or "unknown"
        interpreter = (" Installed dependencies may already have changed; the boot re-synced them for the "
                       "serving tree." if handoff.get("interpreter_state") else "")
        if not _archive(data_dir, handoff, f"abandoned:{last}"):
            return {}
        note = (f"Adoption of {cand} was abandoned before completion ({last}). The serving body is at its previous "
                f"commit.{interpreter}")
        return {**update, "outcome": "not_applied", "note": note} if update else _settled(handoff, "not_applied", note)
    if phase != "switched":
        return {}
    head = _git(repo_dir, "rev-parse", "HEAD", check=False).stdout.strip()
    if head != handoff.get("cand"):
        if not _archive(data_dir, handoff, f"superseded:head={head[:12]}"):
            return {}
        note = f"Adoption of {cand} landed, then the checkout moved to {head[:12]}."
        return {**update, "outcome": "superseded", "note": note} if update else _settled(handoff, "superseded", note)
    loaded = _loaded_generation(handoff)
    if loaded.get("sha") != handoff.get("cand"):
        # HEAD on disk says the candidate; nothing proves this process imported it. The handoff stays switched for a
        # start that does; an Update's observation names its candidate, whose record shows it (never terminal).
        _log(data_dir, "body_adoption_unconfirmed", handoff, reason="loaded_tree_unattested", loaded=loaded)
        note = (f"The checkout is at {cand}, but this process did not verify that tree before its imports; the "
                "adoption stays unconfirmed until a start that does.")
        if update:
            return {**update, "outcome": "unconfirmed", "why": "loaded_tree_unattested", "note": note}
        return _waiting(data_dir, handoff, note, why="loaded_tree_unattested")
    update = handoff.get("kind") == "update"
    if not supervisor_ready and update and handoff.get("probe_command") and providerless_ready is not None:
        # Switched, so its probe imported the boot's modules and the server surface on this interpreter first.
        supervisor_ready = bool(providerless_ready())
    if not supervisor_ready:
        _log(data_dir, "body_adoption_unconfirmed", handoff, reason="supervisor_not_ready")
        if update:
            return _update_not_ready(data_dir, repo_dir, handoff, "supervisor_not_ready", request_restart)
        return _settled(handoff, "unconfirmed", "")
    if handoff.get("native"):  # an app transition: the actual app, entry, runtime, environment and roots
        smoke = _native_readback(handoff, repo_dir)
    else:
        smoke = (native_smoke or (lambda: {"ok": True, "skipped": "no_native_check"}))()
    if not smoke.get("ok"):
        _log(data_dir, "body_adoption_unconfirmed", handoff, reason="native_host", smoke=smoke)
        if update:
            return _update_not_ready(data_dir, repo_dir, handoff, "native_host_unconfirmed", request_restart)
        return _waiting(data_dir, handoff, f"Body {cand} is running, but the native host did not confirm its "
                                           "artifact; the adoption stays unconfirmed.")
    body_switch.record(str(helper_dir(data_dir)), handoff, "adopted", f"ready on {head[:12]} (pid {os.getpid()})")
    _archive(data_dir, handoff, "adopted")
    kept = sorted({*(loaded.get("later_edits") or []), *(handoff.get("later_edit_kept") or [])})
    if update:
        native = handoff.get("native") or {}
        app = (f" The app is Ouroboros {native.get('version')} now; the previous one stays at {native.get('backup')}."
               if native else "")
        return {"outcome": "adopted", "kind": "update", "sha": head, "previous": handoff.get("old"),
                "candidate_id": handoff.get("candidate_id"), "adoption_id": handoff.get("id"),
                "tests": dict(handoff.get("tests") or {}),
                "environment": handoff.get("environment"), **({"native": native} if native else {}),
                "note": f"Update {cand} is running on {handoff.get('branch')} with the local changes it carried; "
                        f"previous commit {str(handoff.get('old') or '')[:12]}." + app
                        + (f" Later local edits were kept at: {', '.join(kept)}." if kept else "")}
    return _settled(handoff, "adopted", f"Adopted candidate commit {cand} on {handoff.get('branch')}; previous commit "
                                        f"{str(handoff.get('old') or '')[:12]}."
                                        + (f" Later local edits were kept at: {', '.join(kept)}." if kept else ""),
                    sha=head, tests=dict(handoff.get("tests") or {}), created_tag=str(handoff.get("created_tag") or ""))


def _adopted_commits(repo_dir: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    """What an adoption brought in, newest first: ``previous..candidate``, bounded; complete says no commit was cut,
    and a cut range counts the rest (``commits_omitted``) that Git's range ``previous_sha..commit_sha`` still holds."""
    if result.get("outcome") != "adopted" or not result.get("previous"):
        return {}
    span = f"{result['previous']}..{result['candidate']}"
    listed = _git(repo_dir, "rev-list", f"--max-count={_INTERVAL_KEPT + 1}", span, check=False)
    shas = listed.stdout.split() if listed.returncode == 0 else []
    fact = {"commits": shas[:_INTERVAL_KEPT], "commits_complete": listed.returncode == 0 and len(shas) <= _INTERVAL_KEPT}
    counted = _git(repo_dir, "rev-list", "--count", span, check=False) if len(shas) > _INTERVAL_KEPT else None
    if counted is not None and counted.returncode == 0 and counted.stdout.strip().isdigit():
        fact["commits_omitted"] = int(counted.stdout) - _INTERVAL_KEPT
    return fact


def _publish_on_task(data_dir: Any, repo_dir: Any, result: Dict[str, Any], text: str) -> None:
    """The settled outcome as its authorizing task's fact (DESIGN "Chat authorship and System rows").

    It is kept on the task result (``body_adoptions``, one row per adoption at the revision of its card
    row, ``_settled``) for the task's Reviews to read, and sent through the terminal outbox as a row of the
    task's card in the task's own room; a page without that card shows the row alone, and no owner-wide
    notice repeats it. A result that could not keep the fact leaves the Reviews without it: the row says
    the outcome, and nothing later restores what was not written.
    """
    from ouroboros.task_results import load_task_result, require_writable_task_result_schema, stamp_task_result_schema
    from ouroboros.task_store import patch_document
    from supervisor.terminal_delivery import enqueue_terminal_delivery_outcome, lineage_chat_id

    task_id, adoption_id = result["task_id"], str(result.get("adoption_id") or "")
    revision = int(result.get("card_row_revision") or 1)
    fact = {"adoption_id": adoption_id, "outcome": result["outcome"], "commit_sha": result["candidate"],
            "previous_sha": str(result.get("previous") or ""), "branch": result["branch"], "note": text,
            "settled_at": utc_now_iso(), "revision": revision, **_adopted_commits(repo_dir, result)}

    def _mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        require_writable_task_result_schema(current)
        if not current.get("status"):  # never mint a partial row the result store would refuse
            raise ValueError(f"task {task_id} has no result record to hold its adoption")
        rows = [dict(row) for row in current.get(ADOPTIONS_KEY) or [] if isinstance(row, dict)]
        if any(row.get("adoption_id") == adoption_id and int(row.get("revision") or 0) >= revision for row in rows):
            return None  # this row, or a later one of the same adoption, is already the task's fact
        rows = [row for row in rows if row.get("adoption_id") != adoption_id] + [fact]
        return stamp_task_result_schema({**current, ADOPTIONS_KEY: rows})

    try:
        patch_document(data_dir, task_id, _mutate)
    except Exception:  # the row still reaches the task's card; its Reviews then cannot name the adoption
        log.warning("Body adoption %s could not be recorded on task %s", adoption_id, task_id, exc_info=True)
    row_id = f"body-adoption:{adoption_id}"
    try:
        task = load_task_result(data_dir, task_id) or {}
    except Exception:
        task = {}
    # Its own room: its Project's chat, else the chat its record names, the hidden partition (0) included.
    # A record that names none (absent, unreadable) has no room here, and Main is never one by default.
    named = task.get("chat_id")
    chat_id = lineage_chat_id(data_dir, task, task_id) or (named if type(named) is int else None)
    if chat_id is None:
        log.warning("Body adoption %s has no chat for task %s; its row was not sent", adoption_id, task_id)
        return
    sent = enqueue_terminal_delivery_outcome(data_dir, {
        "type": "send_message", "chat_id": chat_id, "task_id": task_id, "text": text, "is_progress": True,
        "role": "system", "system_type": "body_adoption", "ts": utc_now_iso(),
        "delivery_id": f"{row_id}:{revision}",
        "progress_meta": {"card_row": "reviews", "card_row_id": row_id, "card_row_revision": revision,
                          "narration": False}})
    if sent == "unavailable":
        log.warning("Body adoption %s row for task %s was not queued; an owed copy replays if registered",
                    adoption_id, task_id)


def settle_on_boot(data_dir: Any, repo_dir: Any, *, supervisor_ready: bool, request_restart=None,
                   providerless_ready=None) -> Dict[str, Any]:
    """Boot's one call: settle the inherited handoff, then the follow-ups an adopted commit is owed.

    The follow-ups are each best-effort: the handoff is already settled when they run. The
    serving push follows only an adoption that full evidence qualified; an Update pushes
    nothing and reports its outcome to its candidate record instead. First, with or without
    a handoff, an Update preparation the previous generation left mid-step fails as interrupted.
    """
    from supervisor.update_prepare import reconcile_interrupted

    reconcile_interrupted(data_dir)
    try:
        from supervisor.update_merge import external_host_restart_smoke

        result = finalize_on_boot(data_dir, repo_dir, supervisor_ready=supervisor_ready,
                                  native_smoke=external_host_restart_smoke, request_restart=request_restart,
                                  providerless_ready=providerless_ready)
    except Exception:
        log.warning("Body adoption could not be settled at boot; the handoff is left as found", exc_info=True)
        return {}
    lines = [str(result.get("note") or "")]  # the card row: the outcome, then the publication on its own line

    def published(sentence: str) -> None:
        lines.append(sentence)
        result["note"] += f" {sentence}"

    def record_update() -> None:
        from supervisor.update_prepare import record_activation

        record_activation(data_dir, result)

    def record_healthy() -> None:
        from supervisor.git_ops_reset import record_healthy_composition

        record_healthy_composition()

    def record_and_push() -> None:
        from supervisor.git_ops import push_to_remote
        from supervisor.git_ops_reset import _record_checkout_facts

        branch = _git(repo_dir, "symbolic-ref", "-q", "--short", "HEAD", check=False).stdout.strip()
        _record_checkout_facts({"current_branch": branch, "current_sha": result["sha"]})
        tier = str(result["tests"].get("tier") or "")
        if tier != "full":
            # This boot's automatic push needs full evidence; ordinary later pushes are separate.
            # Neither the branch nor a tag is pushed and nothing schedules it later: the intent stays
            # on record, withheld rather than failed or done.
            result["publication"] = {"status": "withheld", "branch": branch, "sha": result["sha"],
                                     "tier": tier or "unrecorded",
                                     **({"created_tag": result["created_tag"]} if result.get("created_tag") else {})}
            published(f"Publication withheld: this adoption rests on {tier or 'unrecorded'} test evidence "
                      "and the automatic post-adoption push needs a full run, so nothing was pushed.")
            _log(data_dir, "body_adoption_publication",
                 {"id": result["adoption_id"], "cand": result["sha"], "old": result["previous"]}, **result["publication"])
            return
        # The consumed handoff carries only this commit's operation-created tag.
        pushed, push_msg = push_to_remote(branch or None, created_tag=str(result.get("created_tag") or ""),
                                         expected_commit_sha=result["sha"])
        published(f"Publication: {push_msg}.")
        log.info("Serving push after body adoption: %s (%s)", "ok" if pushed else "skipped", push_msg)

    def notify() -> None:
        if result.get("task_id"):  # the authorizing task's own fact: a row of its card, no owner-wide bubble
            _publish_on_task(data_dir, repo_dir, result, "\n".join(lines))
            return
        from supervisor.message_bus import send_with_budget
        from supervisor.state import load_state

        owner_chat = int((load_state() or {}).get("owner_chat_id") or 0)
        if owner_chat:  # a handoff that names no task stays a runtime fact
            send_with_budget(owner_chat, f"🧬 Body adoption: {result['note']}", role="system",
                             system_type="restart_notice")

    # An Update's facts reach the panel and the mind through its candidate record; the host never messages the owner.
    update = result.get("kind") == "update"
    # A ready start records the composition on disk as healthy (an Update adopted on providerless readiness too: its
    # probe imported the tree here), except one that did not verify the switched tree before its imports.
    healthy = ((supervisor_ready or (update and result.get("outcome") == "adopted"))
               and result.get("outcome") not in ("returning", "start_failed")
               and result.get("why") != "loaded_tree_unattested")
    for applies, step in ((update, record_update), (result.get("outcome") == "adopted" and not update, record_and_push),
                          (bool(result.get("note")) and not update, notify), (healthy, record_healthy)):
        if applies:
            try:
                step()
            except Exception:
                log.warning("Body adoption follow-up %s failed", step.__name__, exc_info=True)
    return result
