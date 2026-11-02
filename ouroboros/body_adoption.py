"""Restart-bound adoption of ONE exact reviewed body-candidate commit.

Adoption is deliberate and never automatic: a plain restart, a crash or a boot
adopts nothing, and a found candidate authorizes nothing. The mind names the
exact commit on ``request_restart``; this module carries that one transition.

* ``authorize`` (old generation, tree untouched) checks the candidate against
  the serving checkout, pins the commit, captures the stdlib switch helper
  OUTSIDE the tree and writes ``state/body_adoption/handoff.json``.
* ``bind_restart`` re-verifies under the existing update/restart serialization
  and marks the one restart that may arm it; a refused restart abandons it.
* ``arm`` (exit tail, after the existing stop owners) publishes the Git-dir
  pointer only on those owners' own evidence that the old generation's
  subprocess readers are gone: the worker pool's exit census (each worker
  process handle confirmed dead), the owned-work stop's completed outcome and
  no live child process handle. It probes nothing itself. Exit or exec then
  ends the in-process readers. Panic and the owner's Restart never arm and
  never wait.
* The switch itself is ``body_switch``, run by the package-init hook of the next
  cold entry before any other body module is imported.
* ``finalize_on_boot`` records ``adopted`` only for a generation that is
  actually ready and that verified, before its own imports, that it was loading
  exactly that commit (``body_switch._attest_loaded``), with native-host proof
  where one applies. A later read of HEAD is not that proof.
* ``settle_on_boot`` gives every outcome to the task that authorized it: its task
  result keeps ``body_adoptions`` and its card gets the row in the task's own room
  (a room it cannot name gets none); only a handoff that names no task stays an
  owner-wide runtime notice.
"""

from __future__ import annotations

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
_PENDING = ("authorized", "armed", "switching", "switched", "stuck")
_NOT_ARMED = "not armed: "  # the exit tail's own record of the stop owners' blockers
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
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True,
                          encoding="utf-8", check=check)


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
    directory = helper_dir(data_dir)
    if directory.exists():
        _archive(data_dir, current, f"replaced:{current.get('phase')}" if current else "stale_helper")
    directory.mkdir(parents=True, exist_ok=True)
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
    # The helper is captured from the body that is running, never from the candidate.
    shutil.copyfile(pathlib.Path(body_switch.__file__), directory / "switch.py")
    from ouroboros import installation_stop
    shutil.copyfile(pathlib.Path(installation_stop.__file__), directory / "installation_stop.py")
    _git(serving, "update-ref", f"{PIN_PREFIX}{handoff['id']}", cand)
    body_switch.record(str(directory), handoff, "authorized", f"restart reason: {reason}")
    _log(data_dir, "body_adoption_authorized", handoff, branch=branch, paths=len(rows))
    return handoff


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


def _archive(data_dir: Any, handoff: Dict[str, Any], outcome: str) -> None:
    """Retire the helper directory: pointer and pin removed, the record kept under ``logs/``."""
    directory = helper_dir(data_dir)
    repo = pathlib.Path(str(handoff.get("repo_dir") or "")) if handoff else None
    pointer_gone = True
    if repo is not None and repo.is_dir():
        try:
            pathlib.Path(body_switch.pointer_path(str(repo))).unlink(missing_ok=True)
        except Exception:
            pointer_gone = False
            log.warning("body adoption pointer could not be removed", exc_info=True)
        if handoff.get("id"):
            _git(repo, "update-ref", "-d", f"{PIN_PREFIX}{handoff['id']}", check=False)
    if handoff:
        _log(data_dir, "body_adoption_closed", handoff, outcome=outcome, phase=handoff.get("phase"),
             events=handoff.get("events"), interpreter_state=handoff.get("interpreter_state"),
             tests=handoff.get("tests"))
    if handoff and not pointer_gone:
        # The hook still finds the pointer: it must read a CLOSED record, never a missing helper
        # (which stops every boot).
        body_switch.record(str(directory), handoff, "closed", outcome)
        return
    shutil.rmtree(directory, ignore_errors=True)


def abandon(data_dir: Any, reason: str, *, task_id: str = "", candidate_id: str = "") -> None:
    """Close an unarmed handoff, scoped to the caller's identity when supplied."""
    handoff = read(data_dir)
    if (handoff.get("phase") == "authorized"
            and (not task_id or handoff.get("task_id") == task_id)
            and (not candidate_id or handoff.get("candidate_id") == candidate_id)):
        _archive(data_dir, handoff, f"abandoned:{reason}")


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

    Never raises and never waits. Returns the resulting phase ("" when no handoff).
    """
    try:
        handoff = read(data_dir)
        if handoff.get("phase") != "authorized" or not handoff.get("restart_bound"):
            return str(handoff.get("phase") or "")
        stop = owned_stop if isinstance(owned_stop, dict) else {}
        census = worker_exits if isinstance(worker_exits, dict) else None
        blockers = [name for name, blocked in (
            ("owner_restart", owner_restart),
            ("worker_exits_unconfirmed", census is None or bool(census.get("unconfirmed"))),
            ("child_processes_alive", bool(live_children)),
            ("owned_work_unconfirmed", stop.get("state") != "completed" or bool(stop.get("unconfirmed"))),
            ("panic_stop", body_switch.installation_panic(data_dir)),
        ) if blocked]
        directory = str(helper_dir(data_dir))
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
    if not any(row.get("path") == rel_path for row in handoff.get("switch") or []):
        return None
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
    if handoff.get("phase") not in ("armed", "switching", "stuck", "switched", "abandoned"):
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


def _waiting(data_dir: Any, handoff: Dict[str, Any], note: str) -> Dict[str, Any]:
    """An unconfirmed boot; the handoff stays for the next one. Words the card row does not already say
    take the next revision, kept on the handoff before anything is sent, so every later row orders
    after it. A handoff that cannot keep it fails this settlement, which then sends nothing."""
    row = _card_row(handoff)
    if row.get("note") != note:
        handoff["card_row"] = {"revision": row.get("revision", 0) + 1, "note": note}
        body_switch.write_handoff(str(helper_dir(data_dir)), handoff)
    return _settled(handoff, "unconfirmed", note)


def finalize_on_boot(data_dir: Any, repo_dir: Any, *, supervisor_ready: bool,
                     native_smoke=None) -> Dict[str, Any]:
    """Settle the handoff this generation inherited; returns ``_settled`` facts (``{}`` if none).

    ``adopted`` needs THIS generation ready on exactly the candidate commit and,
    where a native host is selected, its readback. A boot that is not ready
    leaves ``switched`` in place for the existing recovery owners.
    """
    handoff = read(data_dir)
    phase = handoff.get("phase")
    if not handoff or phase in ("armed", "switching", "stuck"):
        return {}
    cand = str(handoff.get("cand") or "")[:12]
    if phase == "authorized":  # found at boot: the generation that authorized it is gone
        # Only the exit tail's blockers explain it; the other events quote the caller's reason or mark the binding.
        last = str((handoff.get("events") or [{}])[-1].get("detail") or "")
        why = last if last.startswith(_NOT_ARMED) else "the adoption was not armed before this boot"
        _archive(data_dir, handoff, "abandoned:not_armed")
        return _settled(handoff, "not_applied", f"Adoption of {cand} was not applied ({why}). "
                                                "The serving body is unchanged; the candidate is retained.")
    if phase == "abandoned":
        last = (handoff.get("events") or [{}])[-1].get("detail") or "unknown"
        interpreter = (" Installed dependencies may already have changed; the boot re-synced them for the "
                       "serving tree." if handoff.get("interpreter_state") else "")
        _archive(data_dir, handoff, f"abandoned:{last}")
        return _settled(handoff, "not_applied", f"Adoption of {cand} was abandoned before completion ({last}). "
                                                f"The serving body is at its previous commit.{interpreter}")
    if phase != "switched":
        return {}
    head = _git(repo_dir, "rev-parse", "HEAD", check=False).stdout.strip()
    if head != handoff.get("cand"):
        _archive(data_dir, handoff, f"superseded:head={head[:12]}")
        return _settled(handoff, "superseded", f"Adoption of {cand} landed, then the checkout moved to {head[:12]}.")
    loaded = _loaded_generation(handoff)
    if loaded.get("sha") != handoff.get("cand"):
        # HEAD on disk says the candidate; nothing proves this process imported it.
        _log(data_dir, "body_adoption_unconfirmed", handoff, reason="loaded_tree_unattested", loaded=loaded)
        return _waiting(data_dir, handoff, f"The checkout is at {cand}, but this process did not verify that "
                                           "tree before its imports; the adoption stays unconfirmed until a "
                                           "start that does.")
    if not supervisor_ready:
        _log(data_dir, "body_adoption_unconfirmed", handoff, reason="supervisor_not_ready")
        return _settled(handoff, "unconfirmed", "")
    smoke = (native_smoke or (lambda: {"ok": True, "skipped": "no_native_check"}))()
    if not smoke.get("ok"):
        _log(data_dir, "body_adoption_unconfirmed", handoff, reason="native_host", smoke=smoke)
        return _waiting(data_dir, handoff, f"Body {cand} is running, but the native host did not confirm its "
                                           "artifact; the adoption stays unconfirmed.")
    body_switch.record(str(helper_dir(data_dir)), handoff, "adopted", f"ready on {head[:12]} (pid {os.getpid()})")
    _archive(data_dir, handoff, "adopted")
    kept = sorted({*(loaded.get("later_edits") or []), *(handoff.get("later_edit_kept") or [])})
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


def settle_on_boot(data_dir: Any, repo_dir: Any, *, supervisor_ready: bool) -> Dict[str, Any]:
    """Boot's one call: settle the inherited handoff, then the follow-ups an adopted commit is owed.

    The follow-ups are each best-effort: the handoff is already settled when they run. The
    serving push follows only an adoption that full evidence qualified.
    """
    try:
        from supervisor.update_merge import external_host_restart_smoke

        result = finalize_on_boot(data_dir, repo_dir, supervisor_ready=supervisor_ready,
                                  native_smoke=external_host_restart_smoke)
    except Exception:
        log.warning("Body adoption could not be settled at boot; the handoff is left as found", exc_info=True)
        return {}
    lines = [str(result.get("note") or "")]  # the card row: the outcome, then the publication on its own line

    def published(sentence: str) -> None:
        lines.append(sentence)
        result["note"] += f" {sentence}"

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

    for applies, step in ((result.get("outcome") == "adopted", record_and_push), (bool(result.get("note")), notify)):
        if applies:
            try:
                step()
            except Exception:
                log.warning("Body adoption follow-up %s failed", step.__name__, exc_info=True)
    return result
