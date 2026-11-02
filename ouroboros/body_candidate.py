"""Own-body authoring candidates: unfinished self-change never enters serving imports.

An ordinary author of Ouroboros's body — a queued root, a direct turn, an
evolution cycle — writes a task-owned linked ``git worktree`` of the serving
repository instead of the checkout the running server imports. The candidate
starts from the serving HEAD COMMIT (never the serving tree's dirt), lives under
the existing isolated worktree root, and is one more ``kind`` of row in the
existing worktree registry, which is what restores the same binding through a
retry, a Resume or a new worker.

Binding is one fact used by every consumer. ``bind`` points the tool context's
body root (``repo_dir`` / ``system_repo_dir`` / ``branch_dev``) at the
candidate, so target resolution, the mode and protected-path guards, the
handlers, process cwd, the review subject and commit all see ONE identity and
keep treating it as Ouroboros's own body — a candidate is never an external
workspace. The checkout that is actually running stays reachable as
``serving_repo_dir_for``: governance for review, restart and rollback read it.

Nothing here chooses an adoption. ``body_adoption`` owns the restart-bound
transition of an exact reviewed candidate commit into the serving checkout.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

from ouroboros import subagent_worktrees as _wt
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

KIND = "body_candidate"
BRANCH_PREFIX = "candidate/"
# Private preservation pins: reachable objects for retained work, never a published line.
PIN_PREFIX = "refs/ouroboros/candidates/"
# Parallel tool calls of one task reach the seam together: only one may provision, and none
# may bind a checkout that is still being populated.
_prepare_lock = threading.RLock()
_BODY_ROOTS = ("active_workspace", "system_repo")


class CandidateRefused(Exception):
    """A typed refusal to prepare or resume a candidate; nothing was changed."""

    def __init__(self, code: str, text: str):
        super().__init__(text)
        self.code, self.text = code, text


# --------------------------------------------------------------------------- #
# Identity and accessors
# --------------------------------------------------------------------------- #
def _metadata(ctx: Any) -> Dict[str, Any]:
    metadata = getattr(ctx, "task_metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def owner_id(ctx: Any) -> str:
    """The task lineage that owns a candidate: the root task, else this task."""
    return str(_metadata(ctx).get("root_task_id") or getattr(ctx, "task_id", "") or "").strip()


def descriptor(ctx: Any) -> Dict[str, Any]:
    """The candidate this context is bound to, or ``{}``."""
    value = _metadata(ctx).get("body_candidate")
    return value if isinstance(value, dict) and value.get("path") else {}


def is_bound(ctx: Any) -> bool:
    return bool(descriptor(ctx)) and getattr(ctx, "serving_repo_dir", None) is not None


def serving_repo_dir_for(ctx: Any) -> pathlib.Path:
    """The checkout the running server imports, whatever this context authors."""
    serving = getattr(ctx, "serving_repo_dir", None)
    return pathlib.Path(serving or getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir"))


def serving_alias_root(ctx: Any, root: Any) -> Optional[pathlib.Path]:
    """The serving checkout whose spelling of a body path still names ``root``'s file.

    Only when ``root`` IS this context's bound candidate and holds no entry named like
    the serving checkout (which would make ``repo/x`` a genuine nested path).
    """
    bound = descriptor(ctx)
    if not bound or not is_bound(ctx):
        return None
    candidate = pathlib.Path(str(bound["path"])).resolve(strict=False)
    serving = serving_repo_dir_for(ctx).resolve(strict=False)
    if pathlib.Path(root).resolve(strict=False) != candidate or (candidate / serving.name).exists():
        return None
    return serving


def _git(cwd: Any, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return _wt._git(pathlib.Path(cwd), *args, check=check)


def _rows(data_dir: Optional[Any] = None, *, op: str = "", strict: bool = False) -> List[Dict[str, Any]]:
    return [row for row in _wt._records(data_dir, op=op, strict=strict) if row.get("kind") == KIND]


def list_candidates(data_dir: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Registered candidates (soft read for context facts and inspection)."""
    return _rows(data_dir)


def find(owner: str, data_dir: Optional[Any] = None, *, strict: bool = False) -> Optional[Dict[str, Any]]:
    owner = str(owner or "").strip()  # ``strict``: an unreadable copy record raises
    return next((row for row in (_rows(data_dir, strict=strict) if owner else [])
                 if row.get("task_id") == owner and not row.get("retiring")), None)


def context_fact(owner: str = "", data_dir: Optional[Any] = None, *, limit: int = 8) -> List[Dict[str, Any]]:
    """Retained candidates as one compact Runtime fact: a registry read, no Git process.

    The mind sees what exists and who owns it, so continuing one is a deliberate
    ``prepare_self_change(resume=...)`` and never an inference from a room or a date.
    A candidate the GC is retiring is left out: no resume would accept it.
    """
    rows = sorted((row for row in list_candidates(data_dir) if not row.get("retiring")),
                  key=lambda row: float(row.get("created_at") or 0), reverse=True)
    shown: List[Dict[str, Any]] = [{
        "id": row.get("candidate_id"), "owner_task": row.get("task_id"), "branch": row.get("branch"),
        "path": row.get("path"), "base": str(row.get("base_sha") or "")[:12],
        "reviewed_commits": len(row.get("reviewed_commits") or []),
        "yours": bool(owner) and row.get("task_id") == owner,
    } for row in rows[:limit]]
    if len(rows) > limit:
        shown.append({"omitted": len(rows) - limit, "source": "state/private_copies/*.json (kind=body_candidate records)"})
    return shown


def _select(selector: str, data_dir: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    selector = str(selector or "").strip()
    for row in _rows(data_dir) if selector else []:
        if not row.get("retiring") and selector in (
                row.get("candidate_id"), row.get("task_id"), row.get("path"), row.get("branch")):
            return row
    return None


def _event(data_dir: Optional[Any], kind: str, **fields: Any) -> None:
    try:
        append_jsonl(_wt._data_dir(data_dir) / "logs" / "events.jsonl",
                     {"ts": utc_now_iso(), "type": kind, **fields})
    except Exception:
        log.debug("body candidate event append failed", exc_info=True)


# --------------------------------------------------------------------------- #
# Provisioning and ownership
# --------------------------------------------------------------------------- #
def _provision(serving: pathlib.Path, owner: str, data_dir: Optional[Any]) -> Dict[str, Any]:
    """A new candidate at the serving HEAD commit: row first, then the admin dir."""
    serving = serving.resolve()
    if _git(serving, "rev-parse", "--is-inside-work-tree", check=False).returncode != 0:
        raise CandidateRefused(
            "CANDIDATE_UNAVAILABLE",
            f"the serving body at {serving} is not a Git working tree, so no candidate can be prepared.")
    root, data_root = _wt._resolve_root(), _wt._data_dir(data_dir)
    safe = _wt._safe_name(owner)
    candidate_id = f"body_{safe}"
    wt_path = _wt.copy_path(root, candidate_id).resolve()
    _wt._assert_root_isolated(wt_path, serving, data_root)
    base_sha = _git(serving, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
    serving_branch = _git(serving, "symbolic-ref", "-q", "--short", "HEAD", check=False).stdout.strip()
    common = pathlib.Path(_git(serving, "rev-parse", "--git-common-dir").stdout.strip())
    git_dir = str((common if common.is_absolute() else serving / common).resolve())
    branch = f"{BRANCH_PREFIX}{safe}"
    row: Dict[str, Any] = {
        "kind": KIND, "candidate_id": candidate_id, "task_id": owner, "owners": [owner],
        "path": str(wt_path), "branch": branch, "base_sha": base_sha, "repo_dir": str(serving),
        "git_dir": git_dir, "serving_branch": serving_branch, "created_at": time.time(),
        "reviewed_commits": [],
    }
    root.mkdir(parents=True, exist_ok=True)
    with _wt._record_lock(wt_path, data_dir, op="provision_body_candidate", task_id=owner):
        if _wt._record_at(wt_path, data_dir, op="provision_body_candidate") is not None or wt_path.exists():
            raise CandidateRefused(
                "CANDIDATE_PATH_OCCUPIED",
                f"{wt_path} already holds work that is not recorded for this task; it was left untouched.")
        # A branch left by an earlier candidate of this owner may carry unique
        # commits: never move it, take the next free name instead.
        suffix = 1
        while _git(serving, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}", check=False).returncode == 0:
            suffix += 1
            branch = f"{BRANCH_PREFIX}{safe}-{suffix}"
        row["branch"] = branch
        _wt._put_record(row, data_dir)  # the record before the admin dir
        _wt._forget_checkout(serving, wt_path)  # a crashed earlier attempt's registration of this path
        _git(serving, "worktree", "add", "--no-checkout", "-b", branch, str(wt_path), base_sha)
    row = _populate(row, data_dir)
    _event(data_dir, "body_candidate_prepared", candidate_id=candidate_id, owner=owner,
           path=str(wt_path), branch=branch, base_sha=base_sha)
    return row


def _populate(row: Dict[str, Any], data_dir: Optional[Any] = None) -> Dict[str, Any]:
    """Check the files out (outside the lock: tree-proportional; no post-checkout hook), then mark ``ready``.

    A row that is not ``ready`` was never handed to an author — a crash between
    the admin dir and the checkout leaves an EMPTY index, where a commit would
    delete the whole body — so populating it again loses nothing.
    """
    if row.get("ready"):
        return row
    _git(row["path"], "reset", "--hard", "--quiet", "--no-recurse-submodules")
    return _update_row(row, lambda entry: entry.__setitem__("ready", True), data_dir)


def _update_row(row: Dict[str, Any], change, data_dir: Optional[Any] = None) -> Dict[str, Any]:
    """Read-modify-write of one candidate record under that record's lock."""
    def apply(entry: Dict[str, Any]) -> Dict[str, Any]:
        if entry.get("kind") != KIND:
            raise CandidateRefused("CANDIDATE_MISSING", "the candidate's record is gone; nothing was changed.")
        if entry.get("retiring"):
            raise CandidateRefused("CANDIDATE_MISSING", "the candidate is being retired; nothing was changed.")
        change(entry)
        return entry

    updated = _wt._update_record(str(row.get("path") or ""), apply, data_dir, op="update_body_candidate",
                                 task_id=str(row.get("task_id") or ""))
    if updated is None:
        raise CandidateRefused("CANDIDATE_MISSING", "the candidate's record is gone; nothing was changed.")
    return dict(updated)


def _recorded_terminal(task_id: str, results_root: Any) -> bool:
    """Positive lifecycle evidence only: an unreadable or missing record is not terminal."""
    try:
        from ouroboros.task_status import FINAL_STATUSES, load_effective_task_result

        record = load_effective_task_result(_wt._data_dir(results_root), task_id, materialize_artifacts=False) or {}
        return record.get("status") in FINAL_STATUSES
    except Exception:
        log.debug("candidate owner status unavailable for %s", task_id, exc_info=True)
        return False


def _results_root(ctx: Any) -> pathlib.Path:
    return pathlib.Path(str(_metadata(ctx).get("budget_drive_root")
                            or getattr(ctx, "budget_drive_root", "") or getattr(ctx, "drive_root")))


def _owner_is_terminal(ctx: Any, task_id: str) -> bool:
    return _recorded_terminal(task_id, _results_root(ctx))


def live_lineage_processes(results_root: Any, task_id: str) -> List[int]:
    """Known processes of this candidate's root AND its children that have not settled.

    The installation's indexed ownership set is the bounded address, not a disk walk.
    A terminal task can still own a kept service. Unknown ownership/exit evidence
    on a matching record retains its checkout; unrelated records do not block it.
    """
    try:
        from ouroboros.owned_shutdown import owned_records
        from ouroboros.platform_layer import pid_provably_gone
        from ouroboros.process_custody import _fingerprint_matches, _service_group_survives_leader
        from ouroboros.task_status import load_effective_task_result
        from ouroboros import workspace_executor as executor

        root = _wt._data_dir(results_root)
        effective = load_effective_task_result(root, task_id, materialize_artifacts=False)
        owners = {task_id, str(effective.get("task_id") or effective.get("id") or task_id)}
        owners.update(str(row.get("task_id") or "") for row in effective.get("retry_lineage", []))
        owners.discard("")
        lineage: Dict[str, str] = {}
        active: List[int] = []
        for entry in owned_records(root, strict=True):
            member = str(entry.get("owner_task") or entry.get("task_id") or "")
            if not member:
                continue  # no known task attribution to this candidate
            if member not in owners and str(entry.get("root_task_id") or "") not in owners:
                if member not in lineage:
                    lineage[member] = str(load_effective_task_result(
                        root, member, materialize_artifacts=False).get("root_task_id") or "")
                if lineage[member] not in owners:
                    continue
            pid = int(entry.get("host_pid") or 0)
            if entry.get("unconfirmed_since"):
                active.append(pid or -1)
                continue
            ledger = entry.get("ledger_entry")
            if isinstance(ledger, dict):
                if _fingerprint_matches(ledger) or _service_group_survives_leader(ledger):
                    active.append(pid or -1)
                continue
            if entry.get("kind") not in {"foreground", "service"}:
                continue
            path = pathlib.Path(str(entry.get("record_path") or ""))
            record = executor._load_process_record(path)
            if record is None or not executor._valid_process_record(path, record, check_identity=False):
                active.append(-1)  # a named record vanished or became unreadable/invalid
                continue
            if record.get("executor_type") == "local":
                if pid <= 0 or not pid_provably_gone(pid):
                    active.append(pid or -1)  # identity mismatch is not proof of exit
            elif (record.get("record_type") == "foreground"
                  and record.get("backend_completed") is True and (pid <= 0 or pid_provably_gone(pid))):
                continue
            elif (record.get("record_type") == "service" and (pid <= 0 or pid_provably_gone(pid))
                  and executor._docker_pid_state(str(record.get("container_name") or ""),
                                                 str(record.get("backend_pid") or "")) == "exited"):
                continue
            else:
                active.append(pid or -1)  # Docker backend may live despite host_pid=0
        return active
    except Exception:
        log.debug("ownership set unavailable for %s", task_id, exc_info=True)
        return [-1]


def continuation_predecessor(carrier: Any) -> str:
    """The host-validated predecessor whose candidate ``prepare`` inherits: the contract's
    source, or a queued row's admitted pointer before its worker projects it there."""
    row = carrier if isinstance(carrier, dict) else {}
    contract = row.get("task_contract") if row else getattr(carrier, "task_contract", None)
    authority = contract.get("predecessor_authority") if isinstance(contract, dict) else None
    source = authority.get("source") if isinstance(authority, dict) else row.get("predecessor_authority_source")
    return str(source.get("task_id") or "").strip() if isinstance(source, dict) else ""


def _transfer(ctx: Any, row: Dict[str, Any], owner: str, *, via: str) -> Dict[str, Any]:
    """Hand a retained candidate to this lineage once its previous owner has ended."""
    previous = str(row.get("task_id") or "")
    if previous == owner:
        return _update_row(row, lambda entry: None)  # recheck retirement even for an explicit self-resume
    if not _owner_is_terminal(ctx, previous):
        raise CandidateRefused(
            "CANDIDATE_OWNER_LIVE",
            f"candidate {row.get('candidate_id')} belongs to task {previous}, which has no terminal "
            "record. Its work stays with that owner; continue or stop that task first.")
    live = live_lineage_processes(_results_root(ctx), previous)
    if live:
        raise CandidateRefused(
            "CANDIDATE_OWNER_PROCESSES_LIVE",
            f"candidate {row.get('candidate_id')} belongs to task {previous}, whose processes "
            f"{live} are still alive (or the ownership set is unreadable). Its work stays with that owner "
            "until they end.")

    def change(entry: Dict[str, Any]) -> None:
        if str(entry.get("task_id") or "") != previous:
            raise CandidateRefused(
                "CANDIDATE_OWNER_CHANGED", "another task took this candidate first; nothing was changed.")
        entry["task_id"] = owner
        entry["owners"] = [*list(entry.get("owners") or [previous]), owner]

    moved = _update_row(row, change, None)
    _event(None, "body_candidate_transferred", candidate_id=row.get("candidate_id"),
           previous_owner=previous, owner=owner, via=via)
    return moved


def bind(ctx: Any, row: Dict[str, Any]) -> Dict[str, Any]:
    """Make the candidate this context's body root; the serving root stays addressable."""
    if getattr(ctx, "serving_repo_dir", None) is None:
        ctx.serving_repo_dir = pathlib.Path(row.get("repo_dir") or serving_repo_dir_for(ctx))
    candidate = pathlib.Path(str(row["path"]))
    ctx.repo_dir = ctx.system_repo_dir = candidate
    ctx.branch_dev = str(row.get("branch") or getattr(ctx, "branch_dev", ""))
    bound = {key: row.get(key) for key in ("candidate_id", "path", "branch", "base_sha", "repo_dir")}
    _metadata(ctx)["body_candidate"] = bound
    _attribute_surface(ctx, candidate, row)
    if time.time() - float(row.get("used_at") or 0) > 3600:
        # Retention counts from last use, so a long-lived owner keeps a clean candidate.
        try:
            _update_row(row, lambda entry: entry.__setitem__("used_at", time.time()))
        except Exception:
            log.debug("candidate last-use stamp failed", exc_info=True)
    return bound


def _attribute_surface(ctx: Any, candidate: pathlib.Path, row: Dict[str, Any]) -> None:
    """Append the candidate to this lineage's mutation baseline as a late surface (evidence only).

    The commit gate stages what the task's own baseline attributes to it, on the
    candidate exactly as on the serving checkout; a candidate taken over from an
    ended owner names that owner as the predecessor whose quiescent changes this
    lineage adopts. Without a host baseline (a direct turn, a manual context) the
    legacy explicit/whole-tree staging contract applies, as it does unbound.
    """
    task_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not task_id or str(_metadata(ctx).get("delegation_role") or "").lower() == "subagent":
        return
    try:
        from ouroboros.mutation_attribution import attribution_task_id, capture_mutation_baseline

        results_root = _results_root(ctx)
        evidence_task_id = attribution_task_id(results_root, (owner_id(ctx), task_id))
        if not evidence_task_id:
            return
        owners = [str(entry) for entry in (row.get("owners") or []) if str(entry)]
        predecessor = owners[-2] if len(owners) >= 2 and owners[-1] == owner_id(ctx) else ""
        capture_mutation_baseline(
            results_root, evidence_task_id,
            [{"surface_type": "system_repo", "host_root": str(candidate)}],
            owner_kind="task_root", owner_id=owner_id(ctx),
            predecessor_source={"task_id": predecessor} if predecessor else None)
    except Exception:
        log.warning("candidate %s was not added to the mutation baseline", row.get("candidate_id"), exc_info=True)


def restore(ctx: Any) -> Dict[str, Any]:
    """Rebind the candidate this lineage already owns (retry, Resume, new worker)."""
    try:
        if str(_metadata(ctx).get("delegation_role") or "").lower() == "subagent":
            return {}
        row = find(owner_id(ctx))
        if row and pathlib.Path(str(row.get("path") or "")).is_dir():
            return bind(ctx, _populate(row))
    except Exception:
        log.warning("body candidate binding could not be restored", exc_info=True)
    return {}


def _refusal_before_prepare(ctx: Any) -> Optional[CandidateRefused]:
    from ouroboros.config import get_runtime_mode
    from ouroboros.consciousness_authority import effective_runtime_mode
    from ouroboros.contracts.task_constraint import normalize_task_constraint

    if not owner_id(ctx):
        return CandidateRefused("CANDIDATE_OWNER_UNKNOWN", "this context has no task identity to own a candidate.")
    try:
        mode = effective_runtime_mode(get_runtime_mode(), _metadata(ctx))
    except Exception:
        mode = "advanced"
    if mode == "light":
        return CandidateRefused(
            "LIGHT_MODE_BLOCKED", "runtime_mode=light does not author Ouroboros's own body, in a candidate or in place.")
    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    if constraint is not None and getattr(constraint, "mode", "") in {"acting_subagent", "local_readonly_subagent"}:
        return CandidateRefused(
            "CANDIDATE_NOT_APPLICABLE",
            "a subagent works in the copy or the read-only view its parent selected; the parent owns the candidate.")
    try:
        from ouroboros.tools.registry_guards import _authorized_managed_update_resolver

        if _authorized_managed_update_resolver(ctx):
            return CandidateRefused(
                "CANDIDATE_NOT_APPLICABLE", "the managed-update resolver completes its merge in the serving checkout.")
    except Exception:
        log.debug("managed-update resolver check unavailable", exc_info=True)
    return None


def prepare(ctx: Any, *, resume: str = "") -> Dict[str, Any]:
    """Prepare this lineage's candidate, or deliberately resume an exact retained one.

    Idempotent for a bound context. With no selector the lineage's own row wins,
    then the host-validated continuation predecessor's candidate; otherwise a
    new candidate starts at the serving HEAD commit. ``resume`` names one exact
    retained candidate (id, owner task, path or branch). Ownership moves only
    from an owner whose task record is terminal.
    """
    with _prepare_lock:
        return _prepare_locked(ctx, resume)


def _forget_missing(row: Dict[str, Any]) -> None:
    """Drop the row of a checkout that is gone; its branch keeps whatever it committed."""
    path = str(row.get("path") or "")
    with _wt._record_lock(path, None, op="forget_body_candidate", task_id=str(row.get("task_id") or "")):
        current = _wt._record_at(path, None, op="forget_body_candidate")
        if current is not None and current.get("kind") == KIND:
            _wt._drop_record(path, None)
        _wt._forget_checkout(pathlib.Path(str(row.get("git_dir") or row.get("repo_dir") or ".")), pathlib.Path(path))
    _event(None, "body_candidate_checkout_missing", candidate_id=row.get("candidate_id"), branch=row.get("branch"))


def _prepare_locked(ctx: Any, resume: str) -> Dict[str, Any]:
    if is_bound(ctx) and not resume:
        return {**descriptor(ctx), "state": "bound"}
    refusal = _refusal_before_prepare(ctx)
    if refusal is not None:
        raise refusal
    owner = owner_id(ctx)
    serving = serving_repo_dir_for(ctx)
    state = "resumed"
    if resume:
        row = _select(resume)
        if row is None:
            raise CandidateRefused("CANDIDATE_MISSING", f"no retained candidate matches {resume!r}.")
        if not pathlib.Path(str(row.get("path") or "")).is_dir():
            raise CandidateRefused("CANDIDATE_MISSING", f"candidate {row.get('candidate_id')} has no checkout left.")
        mine = find(owner)
        if mine is not None and mine.get("path") != row.get("path"):
            raise CandidateRefused(
                "CANDIDATE_ALREADY_OWNED",
                f"this task already owns candidate {mine.get('candidate_id')}; one task authors one candidate.")
        row = _transfer(ctx, row, owner, via="explicit_resume")
    else:
        row = find(owner)
        predecessor = continuation_predecessor(ctx)
        if row is None and predecessor and (inherited := find(predecessor)) is not None \
                and pathlib.Path(str(inherited.get("path") or "")).is_dir():
            try:
                row = _transfer(ctx, inherited, owner, via="continuation")
            except CandidateRefused as exc:
                if exc.code != "CANDIDATE_MISSING":
                    raise
                # GC won the decision lock after find: continue in a fresh checkout.
        lost_branch = ""
        if row is not None and not pathlib.Path(str(row.get("path") or "")).is_dir():
            # Nothing on disk to lose: start again; the old branch keeps its commits.
            lost_branch = str(row.get("branch") or "")
            _forget_missing(row)
            row = None
        if row is None:
            row, state = _provision(serving, owner, None), "new"
        if lost_branch:
            return {**bind(ctx, row), "state": state, "previous_branch": lost_branch}
    return {**bind(ctx, _populate(row)), "state": state}


def choose_in_place(ctx: Any) -> None:
    """Record Cyber Pro's deliberate decision to author the serving checkout directly."""
    _metadata(ctx)["body_candidate"] = {"in_place": True}


# --------------------------------------------------------------------------- #
# Dispatch seam, process environment, commit facts
# --------------------------------------------------------------------------- #
def _targets_body(ctx: Any, root: str) -> bool:
    from ouroboros.tools.tool_resolution import active_repo_dir_for, system_repo_dir_for

    if root == "system_repo":
        return True
    try:
        return root == "active_workspace" and (
            active_repo_dir_for(ctx).resolve(strict=False) == system_repo_dir_for(ctx).resolve(strict=False))
    except Exception:
        return False


def authoring_seam(ctx: Any, name: str, args: Dict[str, Any]) -> Optional[CandidateRefused]:
    """Bind the candidate BEFORE an ordinary body write, a PR-integration verb or an acting
    body child resolves a target.

    Process tools are deliberately absent: no command text is classified. The
    mind prepares explicitly before process-first work. A refusal that the
    existing mode gates own (light, subagent, resolver) is left to them.
    """
    from ouroboros.tools.git_pr import BODY_WORKTREE_TOOLS
    from ouroboros.tools.tool_resolution import _ROOT_ARG_REPO_WRITE_TOOLS

    marker = _metadata(ctx).get("body_candidate")
    if is_bound(ctx) or (isinstance(marker, dict) and marker.get("in_place")):
        return None
    if name in _ROOT_ARG_REPO_WRITE_TOOLS:  # the one set every repo-write fence keys on
        applies = _targets_body(ctx, str(args.get("root") or "active_workspace"))
    elif name in BODY_WORKTREE_TOOLS:  # PR integration checks out and commits in the body itself
        applies = True
    elif name == "schedule_subagent":  # only an own-body copy asks the scheduler's own folder selection
        from ouroboros.tools.control_scheduling import child_copies_serving_body

        applies = (str(args.get("write_surface") or "").strip().lower() == "self_worktree"
                   and child_copies_serving_body(ctx, args))
    else:
        applies = False
    if not applies or _refusal_before_prepare(ctx) is not None:
        return None
    try:
        if _git(serving_repo_dir_for(ctx), "rev-parse", "--is-inside-work-tree", check=False).returncode != 0:
            return None  # no Git body (wheel/image install): the legacy in-place contract, disclosed in the map
        prepare(ctx)
    except CandidateRefused as exc:
        return exc
    except Exception as exc:
        return CandidateRefused("CANDIDATE_UNAVAILABLE", f"the candidate could not be prepared: {type(exc).__name__}: {exc}")
    return None


def process_environment(ctx: Any, work_dir: Any, *, source=None) -> Optional[Dict[str, str]]:
    """The isolated environment for a process whose cwd is inside the bound candidate.

    Reuses the test-environment owner: its own HOME, data root, settings, caches
    and temp beside the candidate, so candidate code never resolves the serving
    ``DATA_DIR`` or repository. ``None`` for every other cwd. When the cwd IS
    inside the candidate and the isolated environment cannot be prepared, the
    refusal is typed (``CANDIDATE_ENVIRONMENT_UNAVAILABLE``): the process is not
    started with the serving environment instead.
    """
    bound = descriptor(ctx)
    if bound and is_bound(ctx):
        candidate = pathlib.Path(str(bound["path"])).resolve(strict=False)
    else:
        from ouroboros.workspace_copies import is_system_copy
        from ouroboros.tools.tool_resolution import active_repo_dir_for

        if not is_system_copy(ctx):
            return None
        candidate = active_repo_dir_for(ctx).resolve(strict=False)
    cwd = pathlib.Path(work_dir).resolve(strict=False)
    if cwd != candidate and not cwd.is_relative_to(candidate):
        return None
    try:
        from ouroboros.settings_integrity import runtime_environ
        from ouroboros.test_environment import SCRATCH_MARKER, declare_scratch, isolated_environment

        env_root = candidate.with_name(candidate.name + ".env")
        try:
            env_root.mkdir(parents=True)
        except FileExistsError:
            # A new process invalidates a prior quiescent observation. An older,
            # undeclared environment keeps its unknown provenance.
            if (env_root / SCRATCH_MARKER).is_file() and not (env_root / SCRATCH_MARKER).is_symlink():
                (env_root / SCRATCH_MARKER).touch()
        else:
            declare_scratch(env_root, "body-process", owner=json.dumps({
                "task_id": str((bound or {}).get("task_id") or getattr(ctx, "task_id", "") or ""), "copy": str(candidate)}))
        return isolated_environment(env_root, candidate,
                                    source=runtime_environ() if source is None else source)
    except Exception as exc:
        log.warning("candidate process environment unavailable", exc_info=True)
        raise CandidateRefused(
            "CANDIDATE_ENVIRONMENT_UNAVAILABLE",
            f"the isolated environment for body candidate {bound.get('candidate_id')} could not be prepared "
            f"({type(exc).__name__}: {exc}); a process inside the candidate is not started with the serving "
            "environment") from exc


def executor_environment(ctx: Any, cwd: Any, *, executor, map_path) -> Optional[Dict[str, str]]:
    """Project the body's isolated defaults into an executor without inherited credentials."""
    env = process_environment(ctx, cwd, source={} if executor.kind != "local" else None)
    if env is None or executor.kind == "local":
        return env
    try:
        return {key: "/dev/null" if key == "GIT_CONFIG_GLOBAL" else
                map_path(executor, pathlib.Path(value)) if pathlib.Path(value).is_absolute() else value
                for key, value in env.items()}
    except ValueError as exc:
        raise CandidateRefused(
            "CANDIDATE_ENVIRONMENT_UNAVAILABLE",
            f"the executor must map both the body copy and its sibling .env directory: {exc}") from exc


def record_reviewed_commit(ctx: Any, commit_sha: str, *, row: Optional[Dict[str, Any]] = None, created_tag: str = "") -> None:
    """The commit gate created ``commit_sha`` on this candidate under the configured enforcement
    (``row``: the candidate whose interrupted gate an evolution intent proves). The author's
    test widening stays with that commit, so a later run of the same unadopted delta keeps it."""
    bound = row or descriptor(ctx)
    if not bound or not commit_sha:
        return
    request = getattr(ctx, "_commit_test_request", None)

    def record(entry: Dict[str, Any]) -> None:
        entry["reviewed_commits"] = [
            *[sha for sha in entry.get("reviewed_commits") or [] if sha != commit_sha], str(commit_sha)]
        if isinstance(request, dict):  # an explicit default is known; recovery without a context is not
            entry["test_requests"] = {**(entry.get("test_requests") or {}), str(commit_sha):
                                      {**request, "extra_tests": list(request.get("extra_tests") or ())}}
        if created_tag:
            entry["reviewed_release_tags"] = {
                **(entry.get("reviewed_release_tags") or {}), str(commit_sha): str(created_tag)}

    try:
        _update_row(bound, record)
    except Exception:
        log.warning("reviewed candidate commit %s was not recorded on its row", commit_sha[:12], exc_info=True)


def interval_widening(row: Dict[str, Any], base: str, head: str = "HEAD") -> Dict[str, Any]:
    """Combine first-parent ``base..head`` widening still owed before adoption."""
    recorded, reviewed = row.get("test_requests") or {}, set(row.get("reviewed_commits") or ())
    listed = _git(row["path"], "rev-list", "--first-parent", f"{base}..{head}").stdout.split()
    requests = [recorded.get(sha) for sha in reversed(listed) if sha in reviewed]
    if any(not isinstance(request, dict) for request in requests):
        return {"extra_tests": [], "full_tests": True, "test_reason": "prior reviewed test scope is unknown"}
    reasons = (str(request.get("test_reason") or "").strip() for request in requests)
    return {"extra_tests": sorted({str(p) for request in requests for p in request.get("extra_tests") or ()}),
            "full_tests": any(request.get("full_tests") is True for request in requests),
            "test_reason": "; ".join(dict.fromkeys(reason for reason in reasons if reason))}


def qualification_request(ctx: Any, request: Dict[str, Any]) -> Dict[str, Any]:
    """Select against the adoption base; adoption independently re-checks any unreadable earlier widening."""
    from ouroboros.test_selection import inherit_widening

    base, earlier = adoption_base(ctx), {}
    try:
        row = find(owner_id(ctx))
        earlier = interval_widening(row, base) if row and base else {}
    except Exception:
        log.warning("earlier authored test widening unreadable for this candidate", exc_info=True)
    return {**inherit_widening(pathlib.Path(ctx.repo_dir), request, earlier), **({"base": base} if base else {})}


def intent_capture(task_id: str, serving_capture):
    """``(capture, recover)`` for matching an interrupted evolution gate's commit intent.

    A task that authored in its body candidate committed THERE: ``capture`` runs Git
    in that checkout and ``recover`` restores the reviewed provenance the gate writes
    after the receipt. Otherwise (no task, no candidate) the serving capture, unchanged.
    """
    row = find(task_id) if task_id else None
    if row is None or not pathlib.Path(str(row.get("path") or "")).is_dir():
        return serving_capture, lambda sha: sha

    def capture(cmd):
        done = subprocess.run(cmd, cwd=str(row["path"]), capture_output=True, text=True, timeout=30)
        return done.returncode, done.stdout, done.stderr

    def recover(sha: str) -> str:
        record_reviewed_commit(None, sha, row=row)
        return sha

    return capture, recover


def holds_commit(task_id: str, commit_sha: str, data_dir: Optional[Any] = None) -> Optional[bool]:
    """Whether ``task_id``'s own candidate still holds ``commit_sha`` on its branch (what adoption needs).

    ``False`` only on evidence: no candidate of that task, no checkout, or Git
    answering that the commit is gone or off the branch. An unreadable record or
    an unanswered Git is ``None``: unknown never proves the reviewed commit lost.
    """
    try:
        row = next((entry for entry in _rows(data_dir, op="read_body_candidate_commit")
                    if task_id and entry.get("task_id") == task_id), None)
        if row is None and task_id:  # a listing skips an unreadable record; this task's own one raises
            name = f"body_{_wt._safe_name(task_id)}"  # under either name the copy may have been born with
            for path in (_wt._resolve_root() / name, _wt.copy_path(_wt._resolve_root(), name)):
                _wt._record_at(path, data_dir, op="read_body_candidate_commit")
    except Exception:
        return None
    if row is None or not commit_sha or not row.get("path") or not pathlib.Path(str(row["path"])).is_dir():
        return False
    try:
        for args in (("rev-parse", "--verify", "--quiet", f"{commit_sha}^{{commit}}"),
                     ("merge-base", "--is-ancestor", commit_sha, "HEAD")):
            code = subprocess.run(["git", *args], cwd=str(row["path"]), capture_output=True, timeout=30).returncode
            if code:
                return False if code == 1 else None
        return True
    except Exception:
        return None


def adoption_base(ctx: Any) -> str:
    """Where the bound candidate forks from the serving HEAD (``""`` if unresolved): the selection
    base of its post-commit tests, so a proof covers the whole delta an adoption switches."""
    try:
        serving = _git(serving_repo_dir_for(ctx), "rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
        return _git(ctx.repo_dir, "merge-base", serving, "HEAD").stdout.strip()
    except Exception:
        log.warning("candidate adoption base unresolved; its run selects against the last commit only", exc_info=True)
        return ""


def publication_note(ctx: Any) -> str:
    """Why a candidate commit is not auto-pushed: the serving push publishes the serving line only."""
    if not is_bound(ctx):
        return ""
    return (f" [not pushed: commit is on candidate branch {descriptor(ctx).get('branch')}; "
            "publication stays an explicit Git/PR step]")


def unreviewed_commits(row: Dict[str, Any], candidate_sha: str) -> List[str]:
    """First-parent commits ``base..candidate`` that did not come through the commit gate."""
    listed = _git(row["path"], "rev-list", "--first-parent", f"{row['base_sha']}..{candidate_sha}").stdout.split()
    reviewed = set(row.get("reviewed_commits") or [])
    return [sha for sha in listed if sha not in reviewed]


# --------------------------------------------------------------------------- #
# Retention: unique work outlives age; an incomplete capture keeps the original
# --------------------------------------------------------------------------- #
def unique_work(row: Dict[str, Any]) -> Dict[str, Any]:
    """What only this candidate holds: dirty/untracked/ignored files and unadopted commits.

    Env/cache/junk paths (the patch-capture static rules) are not part of a capture; they are
    listed as ``excluded``, since those rules decide transport and no producer declared them
    disposable. A failed read is reported as unique: unknown never means empty.
    """
    from ouroboros.workspace_patch_rules import _patch_exclude_reason

    path = pathlib.Path(str(row.get("path") or ""))
    try:
        raw = checkout_work(path)
        # The static rules name env/cache DIRECTORIES and runtime junk tails; a plain file that
        # merely carries such a directory's name ("env", "venv") is the author's own and unique.
        files = [*raw["untracked"], *raw["ignored"]]
        loose = [rel for rel in files
                 if not _patch_exclude_reason(rel) or ("/" not in rel and not (path / rel).is_dir()
                                                       and _patch_exclude_reason(rel).startswith("top-level"))]
        serving = pathlib.Path(str(row["repo_dir"]))
        # Only the serving checkout establishes adoption. Another candidate or a
        # temporary child's branch can retain this tip without ever having adopted it.
        ahead = _git(serving, "rev-list", "--count", raw["tip"], "--not", "HEAD").stdout.strip()
        return {"readable": True, "tracked": raw["tracked"], "loose": loose, "tip": raw["tip"],
                "unadopted_commits": int(ahead or 0), "excluded": sorted(set(files) - set(loose))}
    except Exception as exc:
        return {"readable": False, "error": f"{type(exc).__name__}: {exc}"}


def checkout_work(path: pathlib.Path) -> Dict[str, Any]:
    """A checkout's raw state beyond its HEAD, no path judged by its name: changed tracked paths,
    untracked and ignored files, and its tip. A failed read raises."""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    status = _wt._git_env(path, "status", "--porcelain", "-z", "--untracked-files=all", env=env).stdout
    ignored = _wt._git_env(path, "ls-files", "-z", "--others", "--ignored", "--exclude-standard", env=env).stdout
    tracked = [chunk[3:].decode("utf-8", "surrogateescape") for chunk in status.split(b"\0")
               if chunk and not chunk.startswith(b"??")]
    untracked = [chunk[3:].decode("utf-8", "surrogateescape") for chunk in status.split(b"\0")
                 if chunk.startswith(b"??")]
    return {"tracked": tracked, "untracked": untracked,
            "ignored": [chunk.decode("utf-8", "surrogateescape") for chunk in ignored.split(b"\0") if chunk],
            "tip": _git(path, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip()}


def has_unique_work(work: Dict[str, Any]) -> bool:
    return (not work.get("readable")) or bool(
        work.get("tracked") or work.get("loose") or work.get("unadopted_commits"))


def _work_signature(row: Dict[str, Any], work: Dict[str, Any]) -> str:
    """Content of the paths a removing verdict captured, including its staged index."""
    path = pathlib.Path(str(row["path"]))
    try:
        digest = hashlib.sha256()
        digest.update(_wt._git_env(path, "ls-files", "-s", "-z",
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).stdout)
        for rel in sorted(set(work.get("tracked") or []) | set(work.get("loose") or [])):
            target = path / rel
            digest.update(rel.encode("utf-8", "surrogateescape") + b"\0")
            if target.is_symlink():
                digest.update(b"link\0" + os.fsencode(os.readlink(target)))
            elif target.is_file():
                digest.update(b"file\0")
                with target.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
            elif not target.exists():
                digest.update(b"deleted\0")
            else:
                return ""  # a directory/special file was not captured as a blob
            digest.update(b"\0")
        return digest.hexdigest()
    except (OSError, ValueError, subprocess.CalledProcessError):
        return ""


def _preserve(row: Dict[str, Any], work: Dict[str, Any]) -> str:
    """Capture the candidate's whole unique state on a private pin; ``""`` unless COMPLETE.

    A temporary index refreshes the tracked and staged paths, then adds every unique loose file by
    its exact name, ignored ones included. The ``excluded`` paths are never staged, nor written as
    objects: they reach a capture only as the owner's accepted loss (``retention_verdict``'s
    ``discard_excluded``), and would carry whole caches into Git. The capture counts only when each
    loose path became a blob entry and the whole staged tree still equals the files on disk;
    anything Git cannot hold (a nested repository, an unreadable file) leaves it
    incomplete, and the caller then keeps the original directory.
    """
    path = pathlib.Path(str(row["path"]))
    index = path.with_name(path.name + f".preserve_index_{os.getpid()}")
    env = _wt._git_env_index(index)
    env.update({"GIT_AUTHOR_NAME": "Ouroboros", "GIT_AUTHOR_EMAIL": "ouroboros@localhost",
                "GIT_COMMITTER_NAME": "Ouroboros", "GIT_COMMITTER_EMAIL": "ouroboros@localhost"})
    try:
        # Start from the candidate's REAL index, not HEAD: a version staged but not yet on
        # disk (or staged while the file was edited again) is unique work too.
        real_index = _wt._git_env(path, "ls-files", "-s", "-z", env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).stdout
        _wt._git_env(path, "read-tree", "--empty", env=env)
        if real_index.strip(b"\0"):
            _wt._git_env(path, "update-index", "-z", "--index-info", env=env, input_bytes=real_index)
        staged_tree = _wt._git_env(path, "write-tree", env=env).stdout.decode().strip()
        _wt._git_env(path, "add", "-u", "--", ".", env=env)  # modified, deleted and staged paths; no untracked one
        loose = [str(rel) for rel in work.get("loose") or []]
        if loose:  # each by its exact name: a pattern ("*") would sweep in the excluded paths too
            spec = b"\0".join(rel.encode("utf-8", "surrogateescape") for rel in loose) + b"\0"
            _wt._git_env(path, "add", "-f", "--pathspec-from-file=-", "--pathspec-file-nul",
                         env={**env, "GIT_LITERAL_PATHSPECS": "1"}, input_bytes=spec)
            staged = _wt._git_env(path, "ls-files", "-s", "-z", env=env).stdout
            blobs = {chunk.partition(b"\t")[2].decode("utf-8", "surrogateescape")
                     for chunk in staged.split(b"\0") if chunk and not chunk.startswith(b"160000")}
            if not set(loose) <= blobs:
                return ""
        if _wt._git_env(path, "diff-files", "--quiet", env=env, check=False).returncode != 0:
            return ""
        tree = _wt._git_env(path, "write-tree", env=env).stdout.decode().strip()
        parent = "HEAD"
        head_tree = _git(path, "rev-parse", "HEAD^{tree}").stdout.strip()
        if staged_tree not in (tree, head_tree):
            # The staged version differs from both the files and HEAD: keep it as its own commit.
            parent = _wt._git_env(path, "commit-tree", staged_tree, "-p", "HEAD", "-m",
                                  f"ouroboros: retained body candidate {row.get('candidate_id')} (staged index)",
                                  env=env).stdout.decode().strip()
        sha = _wt._git_env(path, "commit-tree", tree, "-p", parent, "-m",
                           f"ouroboros: retained body candidate {row.get('candidate_id')}",
                           env=env).stdout.decode().strip()
        pin = f"{PIN_PREFIX}{row.get('candidate_id')}"
        _git(path, "update-ref", pin, sha)
        return pin
    except Exception:
        log.warning("candidate %s could not be captured; the checkout is retained", row.get("candidate_id"),
                    exc_info=True)
        return ""
    finally:
        try:
            index.unlink()
        except OSError:
            pass


def retention_verdict(row: Dict[str, Any], *, expired: bool, data_dir: Optional[Any] = None,
                      discard_excluded: bool = False) -> Dict[str, Any]:
    """Whether the worktree GC may delete this candidate's checkout, and with what kept.

    ``{"remove": bool, "keep_branch": bool, "reason": str, "pin": str}``. Age alone
    never removes unique work: it is first captured COMPLETELY on a private pin,
    and an incomplete capture retains the original directory. ``discard_excluded`` is the
    owner's explicit discard alone (``copy_capture_proof.discard_unsaved_copy``): the paths
    the capture leaves out are the accepted loss, everything else is still captured first.
    """
    path = pathlib.Path(str(row.get("path") or ""))
    if not path.is_dir():
        # Nothing on disk to lose; the branch keeps any commits it carries.
        return {"remove": True, "keep_branch": True, "reason": "checkout_missing", "pin": ""}
    if not expired:
        return {"remove": False, "keep_branch": True, "reason": "within_retention", "pin": ""}
    if not _recorded_terminal(str(row.get("task_id") or ""), data_dir):
        # Its owner may still be working, or its record is unreadable: unknown is not ended.
        return {"remove": False, "keep_branch": True, "reason": "owner_not_terminal", "pin": ""}
    if live_lineage_processes(data_dir, str(row.get("task_id") or "")):
        return {"remove": False, "keep_branch": True, "reason": "owner_processes_live", "pin": ""}
    if _retained_scratch(row, classify=True, data_dir=data_dir):
        return {"remove": False, "keep_branch": True, "reason": "retained_process_environment", "pin": ""}
    work = unique_work(row)
    if work.get("excluded") and not discard_excluded:
        # No evidence these are temporary or saved, and a pin would copy whole caches into Git.
        return {"remove": False, "keep_branch": True, "reason": "unproven_excluded_inputs", "pin": ""}
    if not has_unique_work(work):
        return {"remove": True, "keep_branch": False, "reason": "no_unique_work", "pin": "", "work": work,
                "owner": _owner_stamp(row)}
    before = _work_signature(row, work) if work.get("readable") else ""
    pin = _preserve(row, work) if before else ""
    after = _work_signature(row, work) if pin else ""
    if not pin or before != after:
        _event(data_dir, "body_candidate_retained", candidate_id=row.get("candidate_id"), path=str(path),
               reason="capture_incomplete", detail=str(work.get("error") or ""))
        return {"remove": False, "keep_branch": True, "reason": "capture_incomplete", "pin": ""}
    _event(data_dir, "body_candidate_preserved", candidate_id=row.get("candidate_id"), pin=pin,
           branch=row.get("branch"), tip=work.get("tip"))
    return {"remove": True, "keep_branch": True, "reason": "preserved_on_pin", "pin": pin, "work": work,
            "signature": after,
            "owner": _owner_stamp(row)}


def _owner_stamp(row: Dict[str, Any]) -> Dict[str, Any]:
    return {"task_id": str(row.get("task_id") or ""), "used_at": float(row.get("used_at") or 0)}


def verdict_holds(row: Dict[str, Any], verdict: Dict[str, Any], *, data_dir: Optional[Any] = None) -> bool:
    """Recheck a removing verdict under the registry lock before deleting its checkout.

    Owner lifecycle/process custody and the captured bytes must still agree, not
    merely the row's name and a set of paths. A later GC pass can judge changes.
    """
    if not verdict.get("remove"):
        return True
    if _retained_scratch(row):
        return False
    if verdict.get("reason") == "checkout_missing":
        return not pathlib.Path(str(row.get("path") or "")).is_dir()
    if verdict.get("owner") != _owner_stamp(row):
        return False
    if not _recorded_terminal(str(row.get("task_id") or ""), data_dir) or live_lineage_processes(
            data_dir, str(row.get("task_id") or "")):
        return False
    seen = verdict.get("work") or {}
    now = unique_work(row)
    if not (bool(now.get("readable")) and all(now.get(key) == seen.get(key)
                                                for key in ("tip", "tracked", "loose", "excluded"))):
        return False
    if verdict.get("reason") == "preserved_on_pin":
        if not verdict.get("signature") or _work_signature(row, now) != verdict["signature"]:
            return False
    return True


def _retained_scratch(row: Dict[str, Any], *, classify: bool = False, data_dir=None, record_witness: bool = True) -> bool:
    """Keep a candidate's one quiescent scratch observation on its existing row
    (``record_witness=False``: the same answer for a reader that must write nothing)."""
    from ouroboros.test_environment import scratch_retention

    path = pathlib.Path(str(row.get("path") or ""))
    env_root = path.with_name(path.name + ".env")
    previous = row.get("scratch_retention")
    previous = previous if isinstance(previous, dict) else {}
    owner = _owner_stamp(row)
    observed = scratch_retention(env_root, previous=previous.get("observation")
                                 if previous.get("owner") == owner else None, classify=classify)
    record = {"owner": owner, "observation": observed}
    if record_witness and classify and row.get("kind") == KIND and record != previous:
        # The walk is outside the lock. Publish only against its exact preview,
        # including retirement/ownership; update that preview for GC's same-row check.
        with _wt._record_lock(path, data_dir, op="record_scratch_retention"):
            if _wt._record_at(path, data_dir, op="record_scratch_retention") != row:
                return True
            _wt._put_record({**row, "scratch_retention": record}, data_dir)
            row["scratch_retention"] = record
    return observed["retained"]


def discard_scratch(row: Dict[str, Any]) -> None:
    """Delete declared disposable scratch only when no nested teardown retained it."""
    path = pathlib.Path(str(row.get("path") or ""))
    env_root = path.with_name(path.name + ".env")
    if path.name and _wt._deletable(env_root, _wt._resolve_root()) and env_root.exists() and not _retained_scratch(
            row, classify=row.get("kind") != KIND):
        _wt._force_rmtree(env_root)
