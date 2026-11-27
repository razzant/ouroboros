"""Uninstall tombstones for per-skill owner state (CPL4-C11, owner batch 3A).

Uninstalling a skill removes its payload but its ``state/skills/<name>/``
directory used to outlive it forever (only ``deps.json`` was cleared). The
hub uninstall paths now write an ``uninstalled.json`` tombstone, and the
startup sweep clears the dead state BY that mark — keeping ``grants.json``
(granted keys are OWNER authority, preserved across reinstall), the
``ouroboroshub.json`` publication receipt (the owner's local submission
history, not payload state) and the tombstone itself. A reinstall self-heals:
the sweep sees a live payload and retires the tombstone instead of sweeping.

Explicit local deletion also lives here. Unlike marketplace uninstall, it
removes the entire state directory under the owner's specific delete action.
``removal_action`` is the one rule both removers admit by; the listing serializes it
as ``removal`` and the Skills card renders exactly that, never its own derivation.
"""

from __future__ import annotations

import contextlib
import logging
import pathlib
import shutil
from typing import Any, Dict

from ouroboros.contracts.schema_versions import with_schema_version
from ouroboros.marketplace.provenance import PUBLICATION_FILENAME
from ouroboros.utils import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)

UNINSTALL_TOMBSTONE_FILENAME = "uninstalled.json"
# Owner facts survive the sweep: grants are the owner's durable key/permission
# decisions (batch №8 3A), and the publication receipt is local submission
# history that a later reinstall still shows (#1314) — neither is payload state.
_SWEEP_KEEP = frozenset({UNINSTALL_TOMBSTONE_FILENAME, "grants.json", PUBLICATION_FILENAME})


def removal_action(loaded: Any, record: Any) -> str:
    """Which removal the host offers one installed skill: ``uninstall`` (``uninstall_installed``:
    installed from a source, or bundled, with or without an install record), ``delete``
    (``delete_local_skill``: made on this computer under ``skills/external/``) or ``""`` (a name
    collision, the owner's own skills folder, or a sourced payload that lost its record)."""
    if loaded is None or getattr(loaded, "identity_collision", False):
        return ""
    location = str(getattr(loaded, "location", "") or "")
    if location == "user_repo":
        return ""
    if location == "external" and record is None:
        return "delete" if str(getattr(loaded, "source", "") or "") in {"self_authored", "external"} else ""
    return "uninstall"


@contextlib.contextmanager
def _own_sign_ins(name: str, services: Any, drive: pathlib.Path):
    """Hold every sign-in lock of one skill across its explicit deletion, in the writers' order (``service_sign_in``: each service's
    pending lock, then its saved accounts', the clients' last), each taken once and never waited for: a lock a sign-in step holds is a
    ``SignInRequired`` (``sign_in_busy``) before anything changes. Its begin, complete, token and handover then wait the deletion out
    and find no skill. Yields ``(forget, left)``: ``forget()`` removes the skill's account and pending files and its own handover
    client rows and raises on any failure; once the block ends ``left`` names the folders that kept other content. Emptied folders
    go by ``rmdir``; nothing else is deleted."""
    import errno

    from ouroboros import oauth_engine, service_sign_in as sign_in
    from ouroboros.service_sign_in_declaration import _ID

    files: list = []
    left: list = []
    with sign_in.at_data_dir(drive), contextlib.ExitStack() as held:
        if sign_in.canonical_skill_name(name) != name:  # no sign-in exists under another name
            yield (lambda: None), left
            return
        folder = sign_in._grant(name, "x", "pending").path.parent.parent  # this skill's grants, no other skill's
        if folder.is_symlink() or any((folder / str(service)).is_symlink() for service in services):  # never follow a link out of the grants
            raise oauth_engine.SignInRequired("sign_in_storage_unavailable", "a sign-in folder is a link; nothing was deleted")
        found = [path.name for path in folder.iterdir() if path.is_dir() and not path.is_symlink() and _ID.fullmatch(path.name)] if folder.is_dir() else []
        for service in sorted({*services, *found}):
            pending = sign_in._grant(name, service, "pending")
            held.enter_context(oauth_engine.lock_all([pending]))
            accounts = [sign_in._grant(name, service, path.stem) for path in sign_in._account_files(name, service)]  # none can appear now
            held.enter_context(oauth_engine.lock_all(accounts))
            files += [*(grant.path for grant in accounts), pending.path]
        registry = sign_in._clients_file()
        if registry.path.is_file() or files:  # a first registration may be writing before the file exists; no empty registry is written
            held.enter_context(oauth_engine.lock_all([registry]))

        def forget() -> None:
            for path in files:
                path.unlink(missing_ok=True)
            rows = sign_in.clients() if registry.path.is_file() else []
            kept = [row for row in rows if row.get("source") != f"handover:{name}"]
            if len(kept) != len(rows):
                oauth_engine._save_sign_in(registry, {"clients": kept})

        yield forget, left
    for path in [*sorted({file.parent for file in files}), folder]:
        try:
            path.rmdir()
        except FileNotFoundError:
            pass
        except OSError as exc:
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
            left.append(f"{path.parent.name}/{path.name}" if path != folder else path.name)


def delete_local_skill(
    drive_root: pathlib.Path, loaded: Any, *, payload_root: str = "", repo_path: str = "",
) -> Dict[str, Any]:
    """Delete the exact local external payload after host action admission; the path, symlink
    and same-name checks are rmtree safety, not eligibility (``removal_action`` is)."""
    from ouroboros import extension_loader
    from ouroboros.extension_registry_state import _lifecycle_lock_for
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_loader import discover_skills

    def refuse(error: str, status: int = 403, **facts: Any) -> Dict[str, Any]:
        return {"ok": False, "error": error, "status_code": status, **facts}

    from ouroboros.deployment_extensions import change_refusal

    if managed := change_refusal(loaded.name, "delete"):  # before any unload or file change
        return refuse(managed, code="managed_by_deployment")
    requested_root = payload_root or f"skills/external/{loaded.name}"
    parts = pathlib.PurePosixPath(requested_root).parts
    if len(parts) != 3 or parts[:2] != ("skills", "external"):
        return refuse("local skill delete requires payload_root=skills/external/<name>")
    drive = pathlib.Path(drive_root).absolute()
    skills_root = drive / "skills"
    external_root = skills_root / "external"
    payload_dir = external_root / parts[2]
    if any(path.is_symlink() for path in (skills_root, external_root, payload_dir)):
        return refuse("local skill delete refuses symlinked data/skills/external payloads")
    if pathlib.Path(loaded.skill_dir).absolute() != payload_dir:
        return refuse("selected skill payload does not match the requested delete path", 409)
    skills = discover_skills(drive, repo_path=repo_path)
    if getattr(loaded, "identity_collision", False) or any(
            item.name == loaded.name and pathlib.Path(item.skill_dir).absolute() != payload_dir for item in skills):
        return refuse("refusing to delete while another skill uses the same sanitized name; rename one first", 409)
    if removal_action(loaded, read_install_record(drive, loaded.name)) != "delete":
        return refuse("local skill delete is limited to self-authored/external skills")
    state_root = (drive / "state" / "skills").absolute()
    state_dir = state_root / loaded.name
    if state_root.is_symlink() or state_dir.is_symlink() or not state_dir.is_relative_to(state_root):
        return refuse(f"refusing to delete unsafe state path for {loaded.name!r}", 500)
    from ouroboros import oauth_engine
    from ouroboros.service_sign_in_declaration import declared
    from ouroboros.skill_grant_follow import end_receipt

    with _lifecycle_lock_for(loaded.name), contextlib.ExitStack() as owned:
        try:  # every sign-in lock of this skill, taken before anything changes and held to the end
            forget, left = owned.enter_context(_own_sign_ins(loaded.name, declared(loaded), drive))
        except oauth_engine.SignInRequired as exc:
            return refuse(f"a sign-in step of {loaded.name!r} holds its lock, or its sign-ins cannot be read ({exc.reason}); "
                          "nothing was deleted; try again", 409)
        try:  # before any file of the skill: a failure leaves it installed, to delete again
            forget()  # explicit deletion forgets its sign-ins; uninstall keeps them, as it keeps grants
        except Exception as exc:
            return refuse(f"the sign-ins of {loaded.name!r} could not all be forgotten ({type(exc).__name__}); some may already be gone, "
                          "the skill itself was not deleted; delete it again to finish", 500)
        extension_loader.unload_extension(loaded.name)
        shutil.rmtree(payload_dir)
        end_receipt(drive, loaded.name)  # the installation ended: a later folder of this name starts unconfirmed
        had_state = state_dir.exists()
        if had_state:
            shutil.rmtree(state_dir)
        deleted_state = had_state and not state_dir.exists()
    from ouroboros.extension_reconcile_queue import announce_extension_state_change
    announce_extension_state_change(drive, loaded.name, reason="deleted")
    try:
        from supervisor.queue import sync_skill_schedules

        sync_skill_schedules(discover_skills(drive, repo_path=repo_path), drive_root=drive)
    except Exception:
        log.debug("local skill delete schedule sync failed", exc_info=True)
    if payload_dir.exists() or state_dir.exists():
        return refuse(f"failed to fully delete local skill {loaded.name!r}", 500)
    return {
        "ok": True, "skill": loaded.name, "source": loaded.source,
        "content_hash": loaded.content_hash, "deleted_payload_root": requested_root,
        "deleted_state": deleted_state, "extension_action": "extension_unloaded",
        "extension_reason": "deleted", **({"sign_ins_left": left} if left else {}),
    }


def write_uninstall_tombstone(drive_root: pathlib.Path, name: str, *, source: str) -> None:
    """Durably mark a skill's payload as uninstalled. Never raises."""
    from ouroboros.skill_grant_follow import end_receipt
    from ouroboros.skill_loader import SKILL_OWNER_STATE_SCHEMA_VERSION, skill_state_dir

    try:
        end_receipt(drive_root, name)  # the payload is gone: a reinstall starts unconfirmed
        atomic_write_json(
            skill_state_dir(pathlib.Path(drive_root), name) / UNINSTALL_TOMBSTONE_FILENAME,
            with_schema_version(
                {"uninstalled_at": utc_now_iso(), "source": str(source or "")},
                SKILL_OWNER_STATE_SCHEMA_VERSION,
            ),
        )
        from ouroboros.maintenance_retirement import artifact_changed
        artifact_changed(drive_root, immediate=True)
    except Exception:
        log.debug("uninstall tombstone write failed for %s", name, exc_info=True)


def sweep_uninstalled_skill_state(drive_root: pathlib.Path) -> Dict[str, Any]:
    """Clear owner state of tombstoned skills; self-heal reinstalled ones.

    Fail-closed per entry: anything that cannot be removed is kept and
    reported, never half-guessed. A state dir WITHOUT a tombstone is never
    touched — the mark is the only authority to sweep by.
    """
    from ouroboros.skill_loader import find_skill

    report: Dict[str, Any] = {"swept": [], "restored": [], "errors": []}
    state_root = pathlib.Path(drive_root) / "state" / "skills"
    try:
        state_dirs = sorted(p for p in state_root.iterdir() if p.is_dir())
    except OSError:
        return report
    for state_dir in state_dirs:
        tombstone = state_dir / UNINSTALL_TOMBSTONE_FILENAME
        if not tombstone.exists():
            continue
        name = state_dir.name
        try:
            live = find_skill(pathlib.Path(drive_root), name) is not None
        except Exception:
            report["errors"].append({"skill": name, "error": "payload_probe_failed"})
            continue  # fail-closed: cannot prove the payload is gone
        if live:
            # Reinstalled since the tombstone landed: the mark is stale.
            try:
                tombstone.unlink()
                report["restored"].append(name)
            except OSError:
                report["errors"].append({"skill": name, "error": "tombstone_unlink_failed"})
            continue
        removed_any = False
        for entry in sorted(state_dir.iterdir()):
            if entry.name in _SWEEP_KEEP:
                continue
            try:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                removed_any = True
            except OSError:
                report["errors"].append({"skill": name, "entry": entry.name,
                                         "error": "remove_failed"})
        if removed_any:
            report["swept"].append(name)
    return report


__all__ = [
    "UNINSTALL_TOMBSTONE_FILENAME",
    "delete_local_skill",
    "removal_action",
    "sweep_uninstalled_skill_state",
    "write_uninstall_tombstone",
]
