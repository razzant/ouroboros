"""Install, update and uninstall skills from connected sources: one lifecycle for every caller.

The Skills page, Ouroboros's own tools and the starter installer all use these functions inside the
one skill lifecycle lane. An install resolves a catalog item, copies the release from the source's
local cache (a pointer entry fetches its author's repository at the exact commit), verifies the copy
against the release digest (a mismatch is refused), lands it atomically with the existing rollback
snapshot and writes the install record, whose basis is the source row's trust or an explicit check. A
built-in lands through the built-in source's own installer (``skill_builtin``), which writes the same
record. ``trusted``, recorded ``self_check`` and owner update ``owner_skip`` keep preflight, then grant under
the owner's auto-grant setting, install dependencies and turn it on once ready; ``review``
lands the skill, starts its ordinary review job and returns: the review grants, installs dependencies
and turns it on only when it leaves no blockers. Turning on is ``reconcile_turn_on_when_ready``: an
explicit choice of the owner is never overridden. One installed skill per name: another source's skill
of the same name is replaced only on an explicit confirmation, and the replaced payload is kept as the
previous copy.

An update names its exact candidate (digest) and refuses when the source moved meanwhile. ``replace``
keeps the previous payload; review-basis replacement also persists lifecycle state until review ends
or the owner accepts it. Rejected/interrupted updates restore that snapshot in the lane; startup settles
those without a live review. ``keep`` records the skipped release, ``merge`` lays out base, new and local
trees (three-way only when the base release can be fetched again): the owner's request goes through the
owner's door like Repair, so the task's edits count as the owner's; Ouroboros asking for itself gets
the trees for its own task, whose edits stay unreviewed until a review passes. Nothing here updates or
repairs a skill on its own.
"""

from __future__ import annotations

import logging
import pathlib
import shutil
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional

from ouroboros.skill_install_record import (
    clear_install_record,
    discard_update_snapshot,
    install_basis_fields,
    read_install_record,
    keep_previous_payload,
    settle_pending_update,
    restore_update_snapshot,
    retain_update_snapshot,
    update_backup_path,
    update_install_record,
    write_install_record,
)
from ouroboros.skill_source_catalog import (
    entry_payload_dir,
    find_item,
    installed_index,
    installed_release_digest,
    record_matches_row,
    tree_release_digest,
)
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

BUCKETS = {"builtin": "native", "hub": "ouroboroshub", "git": "ouroboroshub"}


class SkillInstallError(RuntimeError):
    """A typed refusal; ``payload()`` is the HTTP/tool answer."""

    def __init__(self, code: str, message: str, status: int = 409, **facts: Any) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.facts = facts

    def payload(self) -> Dict[str, Any]:
        return {"ok": False, "code": self.code, "error": str(self), **self.facts}


def _note(progress: Optional[Callable[[str], None]], message: str) -> None:
    if progress is not None:
        try:
            progress(message)
        except Exception:
            log.debug("install progress callback failed", exc_info=True)


def _source_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: row[key] for key in ("id", "kind", "name", "url", "part") if row.get(key)}


def _select_check(plan: Dict[str, Any], **options: Any) -> None:
    try:
        plan.update(install_basis_fields(plan["basis"], **options))
    except ValueError as exc:
        raise SkillInstallError("bad_check", str(exc), 400) from exc


# --- planning ----------------------------------------------------------------------------------------

def plan_install(drive_root: pathlib.Path, item_id: str, *, replace: bool = False,
                 rows: Optional[List[Dict[str, Any]]] = None,
                 repo_dir: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Resolve an install from local data and refuse what cannot happen; nothing is written."""
    try:
        row, item, snapshot = find_item(drive_root, item_id, rows=rows, repo_dir=repo_dir)
    except LookupError as exc:
        raise SkillInstallError("not_found", f"{exc}.", 404) from exc
    if row["kind"] == "local":
        raise SkillInstallError("local_source", f"{item['name']} is read in place from {row['name']}; it is "
                                                 "already installed there.")
    if item.get("withdrawn"):
        reason = str((item["withdrawn"] or {}).get("reason") or "no reason given")
        raise SkillInstallError("withdrawn", f"{item['name']} was withdrawn by {row['name']}: {reason}.")
    if row["kind"] != "registry" and not item.get("digest"):
        raise SkillInstallError("no_release", f"{item['name']} cannot be installed: its files could not be read.")
    plan = {"row": row, "item": item, "snapshot": snapshot, "name": item.get("name", ""), "basis": row["trust"],
            "occupant": None, "already_installed": False}
    occupant = installed_index(drive_root).get(item["name"]) if item.get("name") else None
    if not occupant:
        return plan
    record = occupant.get("record")
    same = bool(record and record_matches_row(record, row) and record.get("entry") == item["name"])
    holder = ((record or {}).get("source") or {}).get("name") or ", ".join(occupant["locations"])
    facts = {"occupant": {"name": item["name"], "source": holder, "locations": list(occupant["locations"])}}
    if same and (record.get("release") or {}).get("digest") == item["digest"]:
        plan["already_installed"] = True
        return plan
    if same:
        raise SkillInstallError("update_instead", f"{item['name']} is installed from {row['name']}; use Update.",
                                **facts)
    if len(occupant["locations"]) > 1:
        raise SkillInstallError("name_collision", f"Several folders already use the name {item['name']}; "
                                                  "rename one first.", **facts)
    if "user_repo" in occupant["locations"]:
        raise SkillInstallError("name_taken", f"{item['name']} lives in your local skills folder; rename or "
                                              "remove it there first.", **facts)
    if not replace:
        raise SkillInstallError("name_taken", f"A skill named {item['name']} is already installed ({holder}). "
                                              "Keep it, or replace it.", **facts)
    plan["occupant"] = occupant
    return plan


def _install_record(plan: Dict[str, Any], *, actor: str, reason: str, task_id: str,
                    turn_on_when_ready: bool) -> Dict[str, Any]:
    row, item, snapshot = plan["row"], plan["item"], plan["snapshot"]
    entry = next((e for e in snapshot.get("entries") or [] if e.get("name") == item["name"]), {})
    source = entry.get("source") or {}
    release: Dict[str, Any] = {"version": item.get("version", ""), "digest": item["digest"]}
    if source.get("kind") == "git":
        release.update(commit=source.get("commit", ""), repo=source.get("repo", ""), path=source.get("path", ""))
    elif row["kind"] in {"hub", "git"}:
        prefix = str(snapshot.get("path") or "")
        release.update(commit=str(snapshot.get("commit") or ""),
                       path="/".join(part for part in (prefix, str(source.get("path") or "")) if part))
    record = {"source": _source_fields(row), "entry": item["name"], "release": release,
              "base_digest": item["digest"], "basis": plan["basis"], "official": bool(item.get("official")),
              "installed_by": "ouroboros" if actor == "ouroboros" else "owner",
              "turn_on_when_ready": bool(turn_on_when_ready)}
    if reason:
        record["reason"] = str(reason)[:500]
    if task_id:
        record["task_id"] = str(task_id)
    if plan.get("check"):
        record["check"] = plan["check"]
    return record


# --- landing -----------------------------------------------------------------------------------------

def _move_aside(skill_dir: pathlib.Path, name: str) -> pathlib.Path:
    rollback = skill_dir.parent / ".rollback"
    rollback.mkdir(parents=True, exist_ok=True)
    aside = rollback / f"{name}.replaced-{uuid.uuid4().hex[:12]}"
    skill_dir.rename(aside)
    return aside


def _unload(name: str) -> None:
    from ouroboros.extension_loader import unload_extension
    from ouroboros.extension_registry_state import _lifecycle_lock_for, realtime_load_refusal

    with _lifecycle_lock_for(name):
        if refusal := realtime_load_refusal(name):
            raise SkillInstallError(refusal, "A live call is using this skill; retry after it ends.")
        unload_extension(name)


def _land_builtin(drive_root: pathlib.Path, plan: Dict[str, Any], record: Dict[str, Any], *,
                  repo_dir: Optional[pathlib.Path], progress: Optional[Callable[[str], None]]) -> Dict[str, Any]:
    """A built-in lands through its own source's installer, inside the same rollback as every release."""
    from ouroboros.config import REPO_DIR, ensure_data_skills_dir
    from ouroboros.marketplace.install import restore_payload_state, snapshot_payload_state
    from ouroboros.skill_builtin import install_builtin_skill

    name = plan["name"]
    target = ensure_data_skills_dir(pathlib.Path(drive_root)) / BUCKETS["builtin"] / name
    _note(progress, f"Installing {name} from the built-in skills…")
    _unload(name)
    rollback = snapshot_payload_state(drive_root, name, target)
    occupant_dir = pathlib.Path((plan.get("occupant") or {}).get("skill_dir") or target)
    aside = None
    try:
        if occupant_dir.exists() and occupant_dir.resolve() != target.resolve():
            aside = _move_aside(occupant_dir, name)
        if plan.get("update"):
            retain_update_snapshot(rollback, record)
        result = install_builtin_skill(
            name, drive_root=pathlib.Path(drive_root), seed_dir=pathlib.Path(repo_dir or REPO_DIR) / "skills",
            turn_on_when_ready=bool(record.get("turn_on_when_ready")), installed_by=record["installed_by"],
            reason=str(record.get("reason") or ""), task_id=str(record.get("task_id") or ""))
        if not result.get("ok"):
            raise SkillInstallError(str(result.get("code") or "builtin_install_failed"),
                                    f"{name} was not installed: {result.get('error') or 'the built-in did not land'}")
        builtin = read_install_record(pathlib.Path(drive_root), name) or {}
        write_install_record(pathlib.Path(drive_root), name, {**builtin, **record,
                             "release": {**builtin.get("release", {}), **record["release"]}})
    except Exception:
        if record.get("pending_update"):
            restore_update_snapshot(drive_root, name, "landing_failed")
        else:
            restore_payload_state(rollback)
        if aside is not None and not occupant_dir.exists():
            aside.rename(occupant_dir)
        raise
    if not record.get("pending_update"):
        keep_previous_payload(rollback.backup_dir, name)
    keep_previous_payload(aside, name)
    return read_install_record(pathlib.Path(drive_root), name) or {}


def _land_release(drive_root: pathlib.Path, plan: Dict[str, Any], record: Dict[str, Any], *,
                  repo_dir: Optional[pathlib.Path], progress: Optional[Callable[[str], None]]) -> Dict[str, Any]:
    from ouroboros.config import ensure_data_skills_dir
    from ouroboros.marketplace.fetcher import FetchError, copy_tree_checked, land_staged_tree
    from ouroboros.marketplace.install import restore_payload_state, snapshot_payload_state

    row, item, name = plan["row"], plan["item"], plan["name"]
    if row["kind"] == "builtin":
        return _land_builtin(drive_root, plan, record, repo_dir=repo_dir, progress=progress)
    bucket = ensure_data_skills_dir(pathlib.Path(drive_root)) / BUCKETS[row["kind"]]
    target = bucket / name
    _note(progress, f"Reading {name} from {row['name']}…")
    try:
        folder = entry_payload_dir(drive_root, row, plan["snapshot"], name, repo_dir=repo_dir)
    except Exception as exc:
        raise SkillInstallError("fetch_failed", f"{name} could not be read from {row['name']}: {exc}", 502) from exc
    staging = bucket / ".staging" / f"{name}-{uuid.uuid4().hex[:12]}"
    try:
        copy_tree_checked(folder, staging)
    except (FetchError, OSError) as exc:
        raise SkillInstallError("payload_refused", f"{name} was not installed: {exc}") from exc
    try:
        digest = tree_release_digest(staging, bucket=BUCKETS[row["kind"]])
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise SkillInstallError("payload_unreadable", f"{name} was not installed: {exc}") from exc
    if digest != item["digest"]:
        shutil.rmtree(staging, ignore_errors=True)
        raise SkillInstallError("digest_mismatch", f"The copy of {name} does not match its release digest; "
                                                   "nothing was installed.", expected=item["digest"], actual=digest)
    _note(progress, f"Installing {name}…")
    _unload(name)
    rollback = snapshot_payload_state(drive_root, name, target)
    aside = None
    try:
        occupant_dir = pathlib.Path((plan.get("occupant") or {}).get("skill_dir") or target)
        if occupant_dir.exists() and occupant_dir.resolve() != target.resolve():
            aside = _move_aside(occupant_dir, name)
        if plan.get("update"):
            retain_update_snapshot(rollback, record)
        land_staged_tree(staging, target, replacement_suffix="replaced-source")
        written = write_install_record(pathlib.Path(drive_root), name, record)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if record.get("pending_update"):
            restore_update_snapshot(drive_root, name, "landing_failed")
        else:
            restore_payload_state(rollback)
        if aside is not None and not occupant_dir.exists():
            aside.rename(occupant_dir)
        raise
    if not record.get("pending_update"):
        keep_previous_payload(rollback.backup_dir, name)
    keep_previous_payload(aside, name)
    return written


def _install_registry(drive_root: pathlib.Path, plan: Dict[str, Any], *, actor: str, reason: str, task_id: str,
                      turn_on_when_ready: bool, repo_dir: Optional[pathlib.Path],
                      progress: Optional[Callable[[str], None]]) -> Dict[str, Any]:
    from ouroboros.config import REPO_DIR
    from ouroboros.marketplace.install import install_skill

    row, item = plan["row"], plan["item"]
    record = {"source": _source_fields(row), "basis": plan["basis"], "official": False,
              "installed_by": "ouroboros" if actor == "ouroboros" else "owner",
              "turn_on_when_ready": bool(turn_on_when_ready),
              **({"reason": str(reason)[:500]} if reason else {}), **({"task_id": task_id} if task_id else {})}
    if plan.get("check"):
        record["check"] = plan["check"]
    result = install_skill(pathlib.Path(drive_root), pathlib.Path(repo_dir or REPO_DIR),
                           slug=item["registry_slug"], auto_review=False, progress_callback=progress,
                           install_record=record)
    if not result.ok:
        raise SkillInstallError("registry_install_failed", result.error or "install failed",
                                int(result.error_status or 400))
    plan["name"] = result.sanitized_name  # the adapter decides the local name (it may avoid a clash)
    return read_install_record(pathlib.Path(drive_root), result.sanitized_name) or {}


# --- acceptance and turning on -------------------------------------------------------------------------

def accept_and_turn_on(drive_root: pathlib.Path, name: str, *, repo_path: Optional[str] = None,
                       repo_dir: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """After an accepted release landed (inside its lane job): the deterministic preflight floor, then the
    grants the owner's auto-grant setting gives, the dependencies, and turning on when ready.

    A failed preflight keeps the release off (a payload failure persists the pending record the gate
    honours before the install basis; an infrastructure failure persists nothing).
    """
    from ouroboros.config import REPO_DIR
    from ouroboros.marketplace.install import _MarketplaceReviewCtx
    from ouroboros.skill_lifecycle_actions import reconcile_turn_on_when_ready
    from ouroboros.skill_dependencies import ensure_skill_dependencies
    from ouroboros.skill_loader import auto_grant_if_enabled, find_skill
    from ouroboros.skill_review import run_acceptance_preflight

    drive = pathlib.Path(drive_root)
    if repo_path is None:
        from ouroboros.config import get_skills_repo_path

        repo_path = get_skills_repo_path()
    out: Dict[str, Any] = {"accepted": False, "granted_keys": [], "granted_permissions": [], "deps_status": ""}
    preflight = run_acceptance_preflight(_MarketplaceReviewCtx(drive, pathlib.Path(repo_dir or REPO_DIR)), drive, name)
    if not preflight.get("ok"):
        return {**out, "turn_on": "not_ready", "preflight_error": str(preflight.get("error") or "")[:500],
                "setup_error": "Its files did not pass the check; it remains off."}
    loaded = find_skill(drive, name, repo_path=repo_path)
    if loaded is not None and not loaded.load_error and loaded.review.gate_for(loaded.content_hash)["executable_review"]:
        grant = auto_grant_if_enabled(drive, loaded)
        out.update(accepted=True, granted_keys=list(grant.granted_keys),
                   granted_permissions=list(grant.granted_permissions))
        out["deps_status"], deps_error = ensure_skill_dependencies(drive, loaded)
        if deps_error:
            out["deps_error"] = deps_error
            out["setup_error"] = "Its packages did not install; it remains unavailable."
    if out["deps_status"] in {"installed", "not_required"}:  # only proven readiness permits turn-on
        out["turn_on"] = reconcile_turn_on_when_ready(drive, name, repo_path=repo_path,
                                                      reason="installed release accepted")["action"]
    else:
        out["turn_on"] = "not_ready"
    return out


def run_in_lane(drive_root: pathlib.Path, kind: str, target: str, runner: Callable[[], Any], *,
                message: str = "", chat_id: int = 0) -> Any:
    """Run one blocking operation through the skill lifecycle lane (tools, starter)."""
    from ouroboros.skill_lifecycle_queue import LifecycleJobOptions, run_lifecycle_job_blocking

    def _error(value: Any) -> str:
        return str(value.get("error") or "") if isinstance(value, dict) and value.get("ok") is False else ""

    return run_lifecycle_job_blocking(
        kind=kind, target=target, source="skills", message=message or f"{kind} {target}", chat_id=chat_id,
        runner=runner, options=LifecycleJobOptions(drive_root=pathlib.Path(drive_root), result_error=_error),
    )


def start_review(drive_root: pathlib.Path, name: str, *, repo_dir: Optional[pathlib.Path] = None,
                 reviewers: Optional[List[str]] = None, reason: str = "", chosen_by: str = "owner", panel: Any = None) -> None:
    """Start the ordinary review job of an install in the background.

    The review lends the lifecycle lane out while its panel reads the frozen payload; when it leaves no
    blockers it grants, installs dependencies and turns the skill on as its record asks
    (``skill_review_runner``). An interrupted review stays an honest interrupted job, never an acceptance.
    """
    from ouroboros.reviewer_slot_config import compose_review_panel
    panel = panel or compose_review_panel(reviewers or (), reason, surface="skill", chosen_by=chosen_by)
    digest = str(((read_install_record(drive_root, name) or {}).get("release") or {}).get("digest") or "")
    def _run() -> None:
        try:
            from ouroboros.config import REPO_DIR, get_skills_repo_path
            from ouroboros.marketplace.install import _MarketplaceReviewCtx
            from ouroboros.skill_review_runner import run_skill_review_lifecycle_blocking

            ctx = _MarketplaceReviewCtx(pathlib.Path(drive_root), pathlib.Path(repo_dir or REPO_DIR))
            run_skill_review_lifecycle_blocking(ctx, name, source="skills", repo_path=get_skills_repo_path(), _panel=panel)
        except Exception:
            log.warning("Review after install failed for %s", name, exc_info=True)
        finally:
            if update_backup_path(drive_root, name).exists():
                run_in_lane(drive_root, "update_settle", name, lambda: settle_pending_update(
                    drive_root, name, expected_digest=digest))

    threading.Thread(target=_run, daemon=True, name=f"skill-install-review-{name}").start()


def settle_pending_updates_at_startup(drive_root: pathlib.Path) -> None:
    root = pathlib.Path(drive_root) / "state" / "skills"
    for path in sorted(root.glob("*/update_backup.json")):
        name = path.parent.name
        try:
            run_in_lane(drive_root, "update_settle", name, lambda: settle_pending_update(drive_root, name))
        except Exception:
            log.warning("Pending update settlement failed for %s; kept evidence at %s", name, path, exc_info=True)


def settle_before_replacement(drive_root: pathlib.Path, name: str) -> None:
    """A later explicit replacement first returns an unfinished candidate to its kept base."""
    from ouroboros.extension_registry_state import _lifecycle_lock_for, realtime_load_refusal

    with _lifecycle_lock_for(name):
        if refusal := realtime_load_refusal(name):
            raise SkillInstallError(refusal, "A live call is using this skill; retry after it ends.")
        result = settle_pending_update(drive_root, name, force=True)
    if result["status"] == "restore_failed":
        raise SkillInstallError("restore_failed", f"The previous update of {name} could not be restored; "
                                                 "its payload and state evidence were kept.")


# --- the operations ----------------------------------------------------------------------------------

def install_item(drive_root: pathlib.Path, item_id: str, *, replace: bool = False, actor: str = "owner",
                 reason: str = "", task_id: str = "", turn_on_when_ready: bool = True,
                 rows: Optional[List[Dict[str, Any]]] = None, repo_dir: Optional[pathlib.Path] = None,
                 progress: Optional[Callable[[str], None]] = None,
                 check: str = "review", check_notes: str = "") -> Dict[str, Any]:
    """Install one catalog item (inside the lane). A ``review`` result carries ``needs_review``."""
    plan = plan_install(drive_root, item_id, replace=replace, rows=rows, repo_dir=repo_dir)
    row, item = plan["row"], plan["item"]
    base = {"ok": True, "id": item["id"], "source": _source_fields(row), "basis": plan["basis"],
            "official": bool(item.get("official"))}
    if plan["already_installed"]:
        return {**base, "name": plan["name"], "already_installed": True, "version": item.get("version", "")}
    _select_check(plan, check=check, check_notes=check_notes, task_id=task_id)
    if plan.get("occupant"):
        settle_before_replacement(drive_root, plan["name"])
    if row["kind"] == "registry":
        record = _install_registry(drive_root, plan, actor=actor, reason=reason, task_id=task_id,
                                   turn_on_when_ready=turn_on_when_ready, repo_dir=repo_dir, progress=progress)
    else:
        record = _land_release(drive_root, plan, _install_record(plan, actor=actor, reason=reason, task_id=task_id,
                                                                 turn_on_when_ready=turn_on_when_ready),
                               repo_dir=repo_dir, progress=progress)
    release = record.get("release") or {}
    basis = str(record.get("basis") or plan["basis"])  # the basis the record holds is what the gate reads
    result = {**base, "name": plan["name"], "basis": basis, "version": release.get("version", item.get("version", "")),
              "digest": release.get("digest", ""), "replaced": bool(plan.get("occupant"))}
    if basis == "trusted" and (check != "review" or check_notes):
        result["check_ignored"] = "This source is trusted; no additional check is needed."
    if basis in {"trusted", "self_check", "owner_skip"}:
        result["acceptance"] = accept_and_turn_on(drive_root, result["name"], repo_dir=repo_dir)
    else:
        result["needs_review"] = True
    return result


def _row_for_record(record: Dict[str, Any], rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    exact = next((row for row in rows if row["id"] == (record.get("source") or {}).get("id")), None)
    return exact or next((row for row in rows if record_matches_row(record, row)), None)


def _candidate_plan(drive_root: pathlib.Path, name: str, record: Dict[str, Any], rows: List[Dict[str, Any]],
                    repo_dir: Optional[pathlib.Path]) -> Dict[str, Any]:
    """The update candidate of an install: its source's current entry (any part of the same hub)."""
    row = _row_for_record(record, rows)
    if row is None:
        raise SkillInstallError("source_gone", f"The source {name} came from is no longer connected.")
    for candidate_row in [row, *[r for r in rows if r is not row and record_matches_row(record, r)]]:
        try:
            found_row, item, snapshot = find_item(drive_root, f"{candidate_row['id']}/{record.get('entry')}",
                                                  rows=rows, repo_dir=repo_dir)
        except LookupError:
            continue
        if item.get("withdrawn"):
            raise SkillInstallError("withdrawn", f"{name} was withdrawn by {found_row['name']}; it cannot be "
                                                 "updated.")
        return {"row": found_row, "item": item, "snapshot": snapshot, "name": name, "basis": found_row["trust"],
                "occupant": None, "already_installed": False}
    raise SkillInstallError("left_source", f"{name} is no longer offered by {row['name']}; it keeps working.")


def _merge_trees(drive_root: pathlib.Path, name: str, record: Dict[str, Any], plan: Dict[str, Any],
                 repo_dir: Optional[pathlib.Path]) -> Dict[str, str]:
    """Lay out the new release and, when it can be fetched again, the release the local copy started
    from (verified against its base digest), under the data root."""
    from ouroboros.marketplace.fetcher import copy_tree_checked
    from ouroboros.skill_source_git import checkout_at_commit, scratch_dir

    drive = pathlib.Path(drive_root)
    scratch = scratch_dir(drive, f"merge-{name}")
    copy_tree_checked(entry_payload_dir(drive, plan["row"], plan["snapshot"], name, repo_dir=repo_dir), scratch / "new")
    release = record.get("release") or {}
    url = release.get("repo") or (record.get("source") or {}).get("url")
    if release.get("commit") and url and record.get("base_digest"):
        try:
            base_root = checkout_at_commit(drive, str(url), str(release["commit"]))
            copy_tree_checked((base_root / str(release.get("path") or "")).resolve(), scratch / "base")
            if tree_release_digest(scratch / "base") != record["base_digest"]:
                shutil.rmtree(scratch / "base", ignore_errors=True)
        except Exception as exc:
            log.info("Base release of %s is not retrievable: %s", name, exc)
    return {"new_tree": (scratch / "new").relative_to(drive).as_posix(),
            "base_tree": (scratch / "base").relative_to(drive).as_posix() if (scratch / "base").is_dir() else ""}


def _merge_instructions(name: str, item: Dict[str, Any], payload_root: str, trees: Dict[str, str], *,
                        by_owner: bool) -> str:
    base = trees["base_tree"]
    return (
        (f"The owner asked you to merge an update into the locally modified skill {name}.\n\n" if by_owner else
         f"Merge the update of the locally modified skill {name} in this task.\n\n")
        + f"Local copy (keep its intent): {payload_root}\n"
        f"New release {item.get('version', '')} ({item['digest']}): {trees['new_tree']} (data root)\n"
        "Original release it was installed from: "
        + (f"{base} (data root)" if base else "unknown: that release cannot be fetched again") + "\n\n"
        "Merge the new release into the local copy in place: keep the local changes, take the release's "
        "changes, and resolve conflicts by judgment. "
        + ("Use the original release as the common base for a three-way merge. " if base else
           "Without the original release this is a two-way merge: say so in your summary. ")
        + "Then run skill_review for it. Do not enable it or grant keys yourself."
    )


def _start_merge(drive_root: pathlib.Path, name: str, record: Dict[str, Any], plan: Dict[str, Any],
                 loaded: Any, repo_dir: Optional[pathlib.Path], *, actor: str) -> Dict[str, Any]:
    """The owner's request becomes a selected-skill task through the owner's door (the same door as
    Repair), so its edits are owner edits; Ouroboros asking for itself merges in its own task."""
    item = plan["item"]
    trees = _merge_trees(drive_root, name, record, plan, repo_dir)
    try:
        payload_root = pathlib.Path(loaded.skill_dir).resolve().relative_to(pathlib.Path(drive_root).resolve()).as_posix()
    except ValueError:
        payload_root = ""
    by_owner = actor != "ouroboros"
    instructions = _merge_instructions(name, item, payload_root, trees, by_owner=by_owner)
    if by_owner:
        from supervisor.message_bus import get_bridge

        try:
            bridge = get_bridge()
        except AssertionError as exc:
            raise SkillInstallError("not_ready", "Complete provider setup before asking Ouroboros.", 409) from exc
        bridge.ui_send(instructions, broadcast=False, task_constraint={
            "mode": "normal", "skill_name": name, "payload_root": payload_root, "allow_enable": False,
            "allow_review": True}, task_metadata={"client_surface": {"channel": "skill_update"}})
    update_install_record(pathlib.Path(drive_root), name, update_rejected=None, skipped_release_digest=item["digest"], merge={
        "digest": item["digest"], "version": item.get("version", ""), "requested_at": utc_now_iso(),
        "requested_by": "owner" if by_owner else "ouroboros", **trees})
    out = {"ok": True, "name": name, "mode": "merge", "base_available": bool(trees["base_tree"]),
           "candidate": {"version": item.get("version", ""), "digest": item["digest"]}}
    if by_owner:
        return {**out, "task": "queued"}
    return {**out, "task": "current", "local_tree": payload_root, **trees, "instructions": instructions}


def update_installed(drive_root: pathlib.Path, name: str, *, mode: str, digest: str = "", local_digest: str = "",
                     actor: str = "owner", reason: str = "", task_id: str = "",
                     rows: Optional[List[Dict[str, Any]]] = None, repo_dir: Optional[pathlib.Path] = None,
                     progress: Optional[Callable[[str], None]] = None,
                     check: str = "review", check_notes: str = "", skip_review: bool = False) -> Dict[str, Any]:
    """Apply one update choice (inside the lane): ``replace``, ``keep`` or ``merge``."""
    from ouroboros.config import get_skills_repo_path
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_sources import load_sources

    if mode not in {"replace", "keep", "merge"}:
        raise SkillInstallError("bad_mode", "mode is replace, keep or merge.", 400)
    if skip_review and mode != "replace":
        raise SkillInstallError("bad_mode", "skip_review applies only to mode replace.", 400)
    record = read_install_record(pathlib.Path(drive_root), name)
    loaded = find_skill(pathlib.Path(drive_root), name, repo_path=get_skills_repo_path())
    if record is None or loaded is None:
        raise SkillInstallError("not_from_source", f"{name} was not installed from a source.", 404)
    rows = rows if rows is not None else load_sources()
    current = installed_release_digest(loaded)
    if local_digest and local_digest != current:
        raise SkillInstallError("local_changed", f"The local copy of {name} changed; look at it again.",
                                local_digest=current)
    if (record.get("source") or {}).get("kind") == "registry":
        return _update_registry(drive_root, name, mode=mode, digest=digest, repo_dir=repo_dir, progress=progress,
                                acceptance_options={"check": check, "check_notes": check_notes,
                                                    "skip_review": skip_review, "task_id": task_id})
    plan = _candidate_plan(drive_root, name, record, rows, repo_dir)
    item = plan["item"]
    if digest and digest != item["digest"]:
        raise SkillInstallError("candidate_changed", f"{plan['row']['name']} now offers a different release of "
                                                     f"{name}; look at it again.",
                                candidate={"version": item.get("version", ""), "digest": item["digest"]})
    if mode == "keep":
        update_install_record(pathlib.Path(drive_root), name, skipped_release_digest=item["digest"], update_rejected=None)
        return {"ok": True, "name": name, "mode": "keep", "skipped": {"version": item.get("version", ""),
                                                                       "digest": item["digest"]}}
    if mode == "merge":
        return _start_merge(drive_root, name, record, plan, loaded, repo_dir, actor=actor)
    _select_check(plan, check=check, check_notes=check_notes, task_id=task_id, skip_review=skip_review)
    settle_before_replacement(drive_root, name)
    record = read_install_record(pathlib.Path(drive_root), name) or record
    plan["update"] = True
    plan["occupant"] = {"skill_dir": str(loaded.skill_dir), "locations": [loaded.location]}
    new_record = _install_record(plan, actor=actor, reason=reason, task_id=task_id,
                                 turn_on_when_ready=bool(record.get("turn_on_when_ready", True)))
    landed = _land_release(drive_root, plan, new_record, repo_dir=repo_dir, progress=progress)
    basis = str(landed.get("basis") or plan["basis"])
    result = {"ok": True, "name": name, "mode": "replace", "basis": basis,
              "version": item.get("version", ""), "digest": item["digest"], "previous_kept": True}
    if basis == "trusted" and (check != "review" or check_notes):
        result["check_ignored"] = "This source is trusted; no additional check is needed."
    if basis in {"trusted", "self_check", "owner_skip"}:
        result["acceptance"] = accept_and_turn_on(drive_root, name, repo_dir=repo_dir)
    else:
        result["needs_review"] = True
    return result


def _update_registry(drive_root: pathlib.Path, name: str, *, mode: str, digest: str,
                     repo_dir: Optional[pathlib.Path], progress: Optional[Callable[[str], None]],
                     acceptance_options: Dict[str, Any]) -> Dict[str, Any]:
    if mode != "replace":
        raise SkillInstallError("bad_mode", "A registry skill updates by replacing it with the newest version.", 400)
    from ouroboros.config import REPO_DIR
    from ouroboros.marketplace.install import update_skill

    rejected = (read_install_record(drive_root, name) or {}).get("update_rejected") or {}
    version = rejected.get("version") if digest and rejected.get("digest") == digest else None
    result = update_skill(pathlib.Path(drive_root), pathlib.Path(repo_dir or REPO_DIR), sanitized_name=name,
                          version=version, expected_digest=digest, progress_callback=progress,
                          auto_review=False, acceptance_options=acceptance_options)
    if not result.ok:
        raise SkillInstallError(result.error_code or "registry_update_failed", result.error or "update failed",
                                int(result.error_status or 400),
                                **({"candidate": result.candidate} if result.candidate else {}))
    record = read_install_record(pathlib.Path(drive_root), name) or {}
    trusted = record.get("basis") in {"trusted", "self_check", "owner_skip"}
    out = {"ok": True, "name": name, "mode": "replace", "basis": record.get("basis"),
            **({"acceptance": accept_and_turn_on(drive_root, name, repo_dir=repo_dir)} if trusted
               else {"needs_review": True})}
    if record.get("basis") == "trusted" and (acceptance_options.get("check") != "review"
                                             or acceptance_options.get("check_notes")):
        out["check_ignored"] = "This source is trusted; no additional check is needed."
    return out


def uninstall_installed(drive_root: pathlib.Path, name: str) -> Dict[str, Any]:
    """One uninstall for every source (inside the lane). Grants and the publication receipt survive."""
    from ouroboros.config import get_skills_repo_path
    from ouroboros.marketplace.isolated_deps import DEPS_STATE_FILENAME
    from ouroboros.skill_loader import find_skill, skill_state_dir_path
    from ouroboros.skill_uninstall_state import removal_action, write_uninstall_tombstone

    loaded = find_skill(pathlib.Path(drive_root), name, repo_path=get_skills_repo_path())
    if loaded is None:
        raise SkillInstallError("not_installed", f"{name} is not installed.", 404)
    record = read_install_record(pathlib.Path(drive_root), name)
    if removal_action(loaded, record) != "uninstall":  # the one admission rule; the words are this remover's
        if loaded.identity_collision:
            raise SkillInstallError("name_collision", f"Several folders use the name {name}; rename one first.")
        if loaded.location == "user_repo":
            raise SkillInstallError("local_folder", f"{name} lives in your local skills folder; remove it there.")
        raise SkillInstallError("not_from_source", f"{name} was made on this computer; use Delete to remove it.")
    skill_dir = pathlib.Path(loaded.skill_dir)
    if loaded.location == "clawhub" and (skill_dir / ".clawhub.json").is_file():
        from ouroboros.marketplace.install import uninstall_skill

        result = uninstall_skill(pathlib.Path(drive_root), sanitized_name=name)
        if not result.ok:
            raise SkillInstallError("uninstall_failed", result.error)
        return {"ok": True, "name": name, "source": "clawhub"}
    from ouroboros.extension_loader import unload_extension

    unload_extension(name)  # an explicit owner removal keeps ordinary cancellation
    shutil.rmtree(skill_dir)  # nothing seeds built-ins at startup, so a removed one stays removed
    discard_update_snapshot(pathlib.Path(drive_root), name)
    (skill_state_dir_path(pathlib.Path(drive_root), name) / DEPS_STATE_FILENAME).unlink(missing_ok=True)
    source = str(((record or {}).get("source") or {}).get("id") or loaded.location)
    clear_install_record(pathlib.Path(drive_root), name)
    write_uninstall_tombstone(pathlib.Path(drive_root), name, source=source)
    return {"ok": True, "name": name, "source": source}


__all__ = [
    "BUCKETS", "SkillInstallError", "accept_and_turn_on", "install_item", "plan_install", "run_in_lane",
    "start_review", "uninstall_installed", "update_installed",
]
