"""The built-in skill source: the skills bundled with the app.

The repository's ``skills/`` folder is an ordinary source with id ``builtin``; its index
``skills/builtin.json`` names each bundled skill with its categories and whether the
maintainer recommends it for a new install. Installing a built-in copies the bundled
folder into ``data/skills/native/<name>`` (the per-install note ``.seed-origin`` keeps
older readers working) and writes the same install record every installer writes, with
the basis the built-in row of the source list gives (trusted by default; the derived list
reads ``OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS``). The acceptance gate admits exactly those bytes;
no synthetic review verdict is written, and grants follow the ordinary auto-grant
setting.

A new install gets only the built-ins its starter selection ticks (the starter installer
calls ``install_builtin_skill``; the maintainer's default-on built-ins for this host come
ticked). An existing install keeps what it has and gets each default-on built-in once at an
update: installed and asked to turn on when ready, or, when already installed with no
enablement choice, asked to turn on; a removal or an Off stands. Built-ins ride
app releases: at startup an unmodified built-in whose bundled version changed is re-seeded
in place, keeping its enablement, its Telegram state and the owner's grants while the
requested keys and permissions are unchanged; a copy with local changes is kept and the
owner chooses on update. Nothing re-installs a built-in the owner removed.
A built-in no longer bundled is uninstalled when its copy is unchanged (grants survive), after a
clean re-seed of the same pass; an edited or unknown copy loses its launcher marker and keeps running
as the owner's skill. A built-in whose manifest ``os`` excludes this host is not offered or newly
installed; an installed copy is still re-seeded (discovery hides it, except a Unix
computer-use copy kept on Windows for remote machines). ``state/native_seed.json`` records
the committed bundled tree the last complete pass reconciled, so an unchanged tree costs a
receipt read and two Git queries; bytes accepted without a review get their declared
dependencies at boot while enabled.
"""

from __future__ import annotations

import logging
import pathlib
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

BUILTIN_SOURCE = {"id": "builtin", "kind": "builtin", "name": "Built in"}
SEED_MARKER = ".seed-origin"
RECEIPT_SCHEMA_VERSION = 4  # bundle reconciliation; package/turn-on work has its own pass
_MANIFEST_NAMES = ("SKILL.md", "skill.json")


def bundled_skills_dir() -> pathlib.Path:
    from ouroboros.config import REPO_DIR

    return pathlib.Path(REPO_DIR) / "skills"


def builtin_trust() -> str:
    """The built-in source's trust: its row in the source list (while no list is saved, the derived row
    follows ``OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS``, as does a saved list without a built-in row)."""
    from ouroboros.config import get_trust_native_seeded_skills
    from ouroboros.skill_sources import BUILTIN_ID, load_sources, source_by_id

    row = source_by_id(load_sources(), BUILTIN_ID)
    if row is not None:
        return str(row["trust"])
    return "trusted" if get_trust_native_seeded_skills() else "review"


def _is_seed(entry: pathlib.Path) -> bool:
    return entry.is_dir() and not entry.name.startswith(".") and any(
        (entry / name).is_file() for name in _MANIFEST_NAMES
    )


def _manifest(skill_dir: pathlib.Path) -> Any:
    from ouroboros.skill_identity import _parse_manifest

    return _parse_manifest(skill_dir)


def _version(skill_dir: pathlib.Path) -> str:
    manifest = _manifest(skill_dir)
    return str(getattr(manifest, "version", "") or "").strip() if manifest is not None else ""


def applies_here(skill_dir: pathlib.Path) -> bool:
    """Whether a bundled folder's manifest ``os`` covers this host; an unreadable manifest
    counts (the loader shows it as broken)."""
    manifest = _manifest(pathlib.Path(skill_dir))
    return manifest is None or manifest.applies_to_platform()


def _legacy_remote_line(drive: pathlib.Path, seed: pathlib.Path, target: pathlib.Path) -> str:
    """Keep a Unix computer-use copy someone on Windows enabled for remote machines visible
    after a release narrows the bundled manifest to macOS/Linux (``skill_loader``)."""
    from ouroboros.contracts.skill_manifest import host_platform
    from ouroboros.skill_loader import _LEGACY_UNIX_REMOTE_PLATFORM, skill_state_dir_path
    from ouroboros.utils import read_json_dict

    if target.name != "unix_computer_use" or not (target / SEED_MARKER).is_file():
        return ""
    try:
        if _LEGACY_UNIX_REMOTE_PLATFORM in (target / SEED_MARKER).read_text(encoding="utf-8").splitlines():
            return _LEGACY_UNIX_REMOTE_PLATFORM + "\n"
    except (OSError, UnicodeError):
        return ""
    old = _manifest(target)
    if host_platform() != "win32" or applies_here(seed) or old is None:
        return ""
    choice = read_json_dict(skill_state_dir_path(drive, target.name) / "enabled.json") or {}
    # An actor-less ON may be the owner's or an early automatic one, so it is kept as it was;
    # only a recorded launcher auto-ON is known not to be an owner choice.
    if (old.os == "any" and "net" in (old.permissions or []) and choice.get("enabled") is True
            and choice.get("actor") != "native_seed"):
        return _LEGACY_UNIX_REMOTE_PLATFORM + "\n"
    return ""


def _write_record(drive_root: pathlib.Path, name: str, *, version: str, digest: str, base_digest: Optional[str],
                  turn_on_when_ready: bool, installed_by: str, reason: str = "", task_id: str = "",
                  converted: bool = False) -> Dict[str, Any]:
    from ouroboros.skill_install_record import write_install_record

    record: Dict[str, Any] = {
        "source": dict(BUILTIN_SOURCE),
        "entry": name,
        "release": {"version": version, "digest": digest, "path": f"skills/{name}"},
        "basis": builtin_trust(),
        "official": True,  # the app's own bundled release, exactly as shipped
        "installed_by": installed_by,
        "turn_on_when_ready": bool(turn_on_when_ready),
    }
    if base_digest is not None:
        record["base_digest"] = base_digest  # "" = the original bytes are unknown
    if reason:
        record["reason"] = reason
    if task_id:
        record["task_id"] = task_id
    if converted:
        record["converted_from"] = SEED_MARKER
    return write_install_record(drive_root, name, record)


def install_builtin_skill(
    name: str,
    *,
    drive_root: Optional[pathlib.Path] = None,
    seed_dir: Optional[pathlib.Path] = None,
    turn_on_when_ready: bool = False,
    installed_by: str = "owner",
    reason: str = "",
    task_id: str = "",
) -> Dict[str, Any]:
    """Install, or re-seed in place, one bundled built-in; returns a typed result.

    Lands the bundled folder atomically, writes the install record, carries the owner's
    grants across a re-seed and applies the auto-grant setting. Enablement is left to the
    caller's ``reconcile_turn_on_when_ready`` (inside its lifecycle job) or the owner.
    Refuses a name another installed payload already uses.
    """
    from ouroboros.config import DATA_DIR
    from ouroboros.skill_acceptance import carry_grants_to_current
    from ouroboros.skill_loader import auto_grant_if_enabled, discover_skill_identity, load_skill

    drive = pathlib.Path(drive_root or DATA_DIR)
    seed_root = pathlib.Path(seed_dir or bundled_skills_dir())
    seed = seed_root / name
    if not _is_seed(seed):
        return {"ok": False, "name": name, "error": f"no bundled built-in skill named {name!r}"}
    target = drive / "skills" / "native" / name
    if target.exists() and not (target / SEED_MARKER).is_file():
        return {"ok": False, "name": name, "error": f"{name!r} in the native folder is user-managed; not replaced"}
    if not target.exists() and not applies_here(seed):
        return {"ok": False, "name": name, "code": "other_platform",
                "error": f"built-in {name!r} is for another operating system"}
    resolved_target = target.resolve()
    occupants = [skill for skill in discover_skill_identity(drive, name)
                 if pathlib.Path(skill.skill_dir).resolve() != resolved_target]
    if occupants:
        return {"ok": False, "name": name, "code": "name_taken",
                "error": f"another installed skill already uses the name {name!r}: {occupants[0].skill_dir}"}
    native_root = drive / "skills" / "native"
    staging_root = native_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    legacy_remote = _legacy_remote_line(drive, seed, target)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=f"{name}-", dir=str(staging_root)))
    try:
        shutil.copytree(seed, staging / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (staging / name / SEED_MARKER).write_text(f"seeded_from={seed_root.name}\nsource=builtin\n{legacy_remote}",
                                                  encoding="utf-8")
        from ouroboros.marketplace.fetcher import land_staged_tree

        land_staged_tree(staging / name, target, replacement_suffix="replaced-builtin")
    except OSError as exc:
        return {"ok": False, "name": name, "error": f"could not land built-in {name!r}: {exc}"}
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    try:  # the next startup pass must see this install's dependency and turn-on work
        (drive / "state" / "native_seed.json").unlink(missing_ok=True)
    except OSError:
        log.warning("Built-in receipt could not be invalidated after installing %s", name, exc_info=True)
    skill = load_skill(target, drive)
    if skill is None or skill.load_error or not skill.content_hash:
        return {"ok": False, "name": name, "error": f"built-in {name!r} did not load: "
                f"{getattr(skill, 'load_error', '') or 'no manifest'}"}
    record = _write_record(drive, skill.name, version=str(skill.manifest.version or ""), digest=skill.content_hash,
                           base_digest=None, turn_on_when_ready=turn_on_when_ready,
                           installed_by=installed_by, reason=reason, task_id=task_id)
    skill = load_skill(target, drive)  # the gate now reads the record just written
    carried = carry_grants_to_current(drive, skill)
    granted = auto_grant_if_enabled(drive, load_skill(target, drive))
    return {"ok": True, "name": skill.name, "digest": skill.content_hash, "basis": record["basis"],
            "turn_on_when_ready": bool(turn_on_when_ready), "grants_carried": carried,
            "auto_granted_keys": list(granted.granted_keys), "auto_granted_permissions": list(granted.granted_permissions)}


def _changed_since_install(drive_root: pathlib.Path, name: str, target: pathlib.Path) -> Optional[bool]:
    """Whether the installed copy differs from its recorded base digest; ``None`` when no
    base or evidence of edits is recorded (the launcher's version re-seed still applies).
    A stale native-seed review proves edits even when the original release is unknown. Unreadable bytes
    count as changed: nothing proves them unmodified."""
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_loader import compute_content_hash

    record = read_install_record(drive_root, name) or {}
    base = str(record.get("base_digest") or "")
    if not base:
        return True if record.get("modified") else None
    try:
        return compute_content_hash(target) != base
    except Exception:
        return True


def reseed_changed_builtins(drive_root: pathlib.Path, *, seed_dir: Optional[pathlib.Path] = None,
                            log_obj: Any = None, failures: Optional[List[str]] = None) -> int:
    """Re-seed in place the unmodified built-ins whose bundled version changed (marker-owned only).

    A copy with local changes against its recorded base is kept: the owner chooses on
    update (replace, keep or merge). A payload that differs at an equal version is kept
    and logged; a removed built-in is never resurrected; enablement and Telegram state
    live outside the payload and stay.
    """
    from ouroboros.skill_install_record import read_install_record

    logger = log_obj or log
    seed_root = pathlib.Path(seed_dir or bundled_skills_dir())
    native_root = pathlib.Path(drive_root) / "skills" / "native"
    if not seed_root.is_dir() or not native_root.is_dir():
        return 0
    upgraded = 0
    for seed in sorted(seed_root.iterdir()):
        target = native_root / seed.name
        if not _is_seed(seed) or not target.is_dir() or not (target / SEED_MARKER).is_file():
            continue
        seed_version, target_version = _version(seed), _version(target)
        if not seed_version or not target_version:
            if failures is not None:
                failures.append(seed.name)  # an unreadable version is unfinished work, never a completed pass
            continue
        if seed_version == target_version:
            _warn_equal_version_drift(seed, target, seed_version, logger)
            continue
        if _changed_since_install(pathlib.Path(drive_root), seed.name, target):
            logger.warning("Built-in skill %s has local changes (installed=%s, bundled=%s); kept — "
                           "the owner chooses on update", seed.name, target_version, seed_version)
            continue
        logger.info("Built-in skill %s version drift (bundled=%s, installed=%s) — re-seeding",
                    seed.name, seed_version, target_version)
        previous = read_install_record(pathlib.Path(drive_root), seed.name) or {}
        result = install_builtin_skill(
            seed.name, drive_root=pathlib.Path(drive_root), seed_dir=seed_root,
            turn_on_when_ready=previous.get("turn_on_when_ready") is True,
            installed_by=str(previous.get("installed_by") or "ouroboros"),
            reason=f"app update {target_version} -> {seed_version}",
        )
        if result.get("ok"):
            upgraded += 1
        else:
            if failures is not None:
                failures.append(seed.name)
            logger.warning("Built-in skill %s re-seed failed: %s", seed.name, result.get("error"))
    return upgraded


def _warn_equal_version_drift(seed: pathlib.Path, target: pathlib.Path, version: str, logger: Any) -> None:
    from ouroboros.skill_loader import compute_content_hash

    try:
        seed_hash, target_hash = compute_content_hash(seed), compute_content_hash(target)
    except Exception as exc:
        logger.warning("Built-in skill %s version %s payload comparison unavailable (%s)",
                       seed.name, version, type(exc).__name__)
        return
    if seed_hash != target_hash:
        logger.warning("Built-in skill %s version %s payload differs (bundled=%s, installed=%s); "
                       "installed files retained because the manifest version is unchanged",
                       seed.name, version, seed_hash, target_hash)


def release_removed_builtins(drive_root: pathlib.Path, *, seed_dir: Optional[pathlib.Path] = None,
                             log_obj: Any = None, retire: bool = True) -> bool:
    """A built-in no longer bundled: a copy nobody changed is uninstalled (the bundle carries its
    successor; grants survive the uninstall), an edited or unknown copy keeps its files and records and
    only the launcher marker goes, so it is the owner's from now on. ``retire=False`` (a re-seed of
    this pass failed) leaves unchanged copies marked for the next boot. Returns whether every one went."""
    from ouroboros.skill_source_install import SkillInstallError, uninstall_installed

    logger = log_obj or log
    seed_root = pathlib.Path(seed_dir or bundled_skills_dir())
    native_root = pathlib.Path(drive_root) / "skills" / "native"
    if not native_root.is_dir():
        return True
    complete = True
    for entry in sorted(native_root.iterdir()):
        marker = entry / SEED_MARKER
        if not entry.is_dir() or entry.name.startswith(".") or not marker.is_file() or (seed_root / entry.name).is_dir():
            continue
        try:
            if _changed_since_install(pathlib.Path(drive_root), entry.name, entry) is False:
                if not retire:
                    complete = False
                    continue
                try:
                    uninstall_installed(pathlib.Path(drive_root), entry.name)
                    logger.info("Built-in skill %r is no longer bundled and was unchanged; uninstalled.", entry.name)
                    continue
                except SkillInstallError:  # another folder uses the name: release this copy as the owner's
                    logger.warning("Built-in skill %r could not be uninstalled; kept as the owner's copy.",
                                   entry.name, exc_info=True)
            marker.unlink()
            logger.info("Built-in skill %r is no longer bundled; its installed copy is now user-managed.", entry.name)
        except OSError:
            complete = False
            logger.warning("Failed to release removed built-in %s", entry, exc_info=True)
    return complete


def record_converted_builtin(drive_root: pathlib.Path, skill: Any, legacy: Any, *,
                             seed_dir: Optional[pathlib.Path] = None) -> bool:
    """Identity-conversion hook: give an existing launcher-seeded built-in its install record.

    The bytes the launcher trusted (a native-seed verdict on a version-1 hash of the
    current bytes) become the installed release. A stale native-seed verdict proves local
    edits, so the unknown original is recorded and reseeding preserves the copy. Without
    that evidence the bundled release supplies the base at the same version; otherwise
    the legacy version reseed still applies. Existing records are never rewritten.
    """
    from ouroboros.skill_install_record import read_install_record, write_install_record
    from ouroboros.skill_loader import compute_content_hash
    from ouroboros.utils import read_json_dict
    from ouroboros.skill_loader import skill_state_dir_path

    if getattr(skill, "location", "") != "native" or not (pathlib.Path(skill.skill_dir) / SEED_MARKER).is_file():
        return False
    if read_install_record(pathlib.Path(drive_root), skill.name) is not None:
        return False
    review = read_json_dict(skill_state_dir_path(pathlib.Path(drive_root), skill.name) / "review.json") or {}
    if review.get("review_profile") == "native_seed" and review.get("content_hash") not in set(legacy or ()):
        write_install_record(pathlib.Path(drive_root), skill.name, {
            "source": dict(BUILTIN_SOURCE), "entry": skill.name, "basis": "review", "legacy": True,
            "release": {"version": str(skill.manifest.version or ""), "digest": ""}, "base_digest": "",
            "modified": True, "official": False, "installed_by": "ouroboros", "turn_on_when_ready": False,
            "converted_from": SEED_MARKER,
        })
        log.warning("Built-in skill %s changed after native seeding; original unknown, installed files retained",
                    skill.name)
        return True
    covered = (str(review.get("review_profile") or "") == "native_seed"
               and str(review.get("content_hash") or "") in set(legacy or ()))
    version = str(skill.manifest.version or "")
    if covered:
        digest, base = skill.content_hash, skill.content_hash
    else:
        seed = pathlib.Path(seed_dir or bundled_skills_dir()) / skill.name
        if not _is_seed(seed):
            return False
        digest = compute_content_hash(seed)
        base = digest if _version(seed) == version else ""
        version = _version(seed) or version
    _write_record(pathlib.Path(drive_root), skill.name, version=version, digest=digest, base_digest=base,
                  turn_on_when_ready=False, installed_by="ouroboros", converted=True)
    return True


def _seed_identity(repo_dir: pathlib.Path) -> Optional[Dict[str, Any]]:
    """The committed bundled tree, read without its payloads; a dirty or unknown tree stays uncached."""
    from ouroboros.git_observation import host_git_env
    if not (repo_dir / ".git").exists():
        return None
    try:
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "skills"],
                                cwd=str(repo_dir), capture_output=True, text=True, timeout=10, env=host_git_env())
        if status.returncode or status.stdout.strip():
            return None
        tree = subprocess.run(["git", "rev-parse", "--verify", "HEAD:skills"], cwd=str(repo_dir),
                              capture_output=True, text=True, timeout=10)
        if tree.returncode or not tree.stdout.strip():
            return None
        return {"schema_version": RECEIPT_SCHEMA_VERSION, "repo": str(repo_dir.resolve()),
                "seed_tree": tree.stdout.strip()}
    except (OSError, subprocess.SubprocessError):
        return None


def _bundled_names(seed_root: pathlib.Path) -> Optional[List[str]]:
    try:
        return sorted(entry.name for entry in seed_root.iterdir() if _is_seed(entry))
    except OSError:
        return None


def default_on_names(seed_root: pathlib.Path) -> List[str]:
    """Bundled built-ins the maintainer turns on by default on this host (``"default_on"`` in
    ``skills/builtin.json``)."""
    from ouroboros.utils import read_json_dict

    index = read_json_dict(pathlib.Path(seed_root) / "builtin.json") or {}
    names = [str(item.get("name") or "") for item in index.get("skills") or []
             if isinstance(item, dict) and item.get("default_on") is True]
    return [name for name in names if name and _is_seed(pathlib.Path(seed_root) / name)
            and applies_here(pathlib.Path(seed_root) / name)]


def _offer_default_builtins(drive: pathlib.Path, seed_root: pathlib.Path, logger: Any, failures: List[str]) -> None:
    """An existing install gets each default-on built-in once (a new install's starter presents them)."""
    from ouroboros.skill_install_record import read_install_record, write_install_record
    from ouroboros.skill_loader import skill_state_dir_path
    from ouroboros.skill_starter import read_marker, record_defaults_offered

    marker = read_marker(drive)
    if str(marker.get("state") or "") in {"", "fresh", "running", "error"}:
        return  # the install's own starter has yet to present them
    from ouroboros.skill_uninstall_state import UNINSTALL_TOMBSTONE_FILENAME

    done = set(marker.get("defaults_offered") or [])
    native = drive / "skills" / "native"
    offered: List[str] = []
    for name in default_on_names(seed_root):
        if name in done:
            continue
        target = native / name
        if (skill_state_dir_path(drive, name) / UNINSTALL_TOMBSTONE_FILENAME).is_file() and not target.exists():
            offered.append(name)  # the owner uninstalled it (before it was on by default, or since)
            continue
        if target.exists():
            if (target / SEED_MARKER).is_file() and not (skill_state_dir_path(drive, name) / "enabled.json").exists():
                record = read_install_record(drive, name)
                if record is not None and record.get("turn_on_when_ready") is not True:
                    write_install_record(drive, name, {**record, "turn_on_when_ready": True})
            offered.append(name)  # a markerless payload is the owner's own
            continue
        if (native / f".post-bootstrap-seed-{name}").exists():
            offered.append(name)  # an earlier release offered it; the owner removed it
            continue
        result = install_builtin_skill(name, drive_root=drive, seed_dir=seed_root, turn_on_when_ready=True,
                                       installed_by="ouroboros", reason="on by default for this computer")
        if result.get("ok"):
            offered.append(name)
        else:
            failures.append(name)  # e.g. another installed skill holds the name: offered once it is gone
            logger.warning("Default-on built-in %s was not installed: %s", name, result.get("error"))
    record_defaults_offered(drive, offered)


def _requested_on(drive: pathlib.Path, skill: Any) -> bool:
    """An install that asked to turn on when ready and that no owner choice keeps off (the one
    intent predicate, shared with ``reconcile_turn_on_when_ready``)."""
    from ouroboros.skill_lifecycle_actions import turn_on_blocker

    return not skill.enabled and turn_on_blocker(drive, skill.name) == ""


def _wants_lifecycle_work(drive: pathlib.Path, skill: Any) -> bool:
    """An installed skill that should run (on, or asked to turn on) whose exact bytes the review
    gate admits: its packages are readied at startup and a requested turn-on finished.
    The installed source does not change that intent."""
    if skill is None or skill.load_error or not skill.review.gate_for(skill.content_hash)["executable_review"]:
        return False
    return bool(skill.enabled) or _requested_on(drive, skill)


def _skills_needing_work(drive: pathlib.Path, names: Optional[List[str]], logger: Any) -> Optional[List[str]]:
    """Missing packages or unfulfilled turn-ons among ``names`` (``None``: all installed skills).
    Inspect each skill independently; unknowns are logged, not read as ready. With no known work,
    any inspection failure returns ``None``. This never governs the seed receipt."""
    try:
        from ouroboros.config import get_skills_repo_path
        from ouroboros.skill_dependencies import skill_deps_not_ready
        from ouroboros.skill_loader import discover_skills, find_skill

        repo_path = get_skills_repo_path() or None
        found = ([find_skill(drive, name, repo_path=repo_path) for name in names] if names is not None
                 else list(discover_skills(drive, repo_path=repo_path)))
    except Exception:
        logger.warning("Installed skills could not be inspected", exc_info=True)
        return None
    pending = []
    unknown = False
    for skill in found:
        try:
            if _wants_lifecycle_work(drive, skill) and (
                    skill_deps_not_ready(drive, skill, strict=True)[1] or _requested_on(drive, skill)):
                pending.append(skill.name)
        except Exception:
            unknown = True
            logger.warning("Skill %s dependencies could not be inspected", getattr(skill, "name", "unknown"), exc_info=True)
    return pending if pending or not unknown else None


def _dependency_pass(drive: pathlib.Path, names: Optional[List[str]], logger: Any) -> List[str]:
    """Ready the packages of skills that want them (``_wants_lifecycle_work``) and finish a requested
    turn-on, one lifecycle job each, then reconcile the extension. Off the boot path and independent of
    the seed receipt; a failure stays in ``deps.json`` for the Skills page and the next boot retries it
    (an owner's OFF waits for the enable, which runs the same step)."""
    pending = _skills_needing_work(drive, names, logger)
    if not pending:
        return []
    try:
        from ouroboros import extension_loader
        from ouroboros.config import get_skills_repo_path, load_settings
        from ouroboros.skill_dependencies import ensure_skill_dependencies
        from ouroboros.skill_lifecycle_actions import reconcile_turn_on_when_ready
        from ouroboros.skill_lifecycle_queue import LifecycleJobOptions, run_lifecycle_job_blocking
        from ouroboros.skill_loader import find_skill

        repo_path = get_skills_repo_path() or None
    except Exception:
        logger.warning("Built-in dependency pass could not start", exc_info=True)
        return []

    def step(name: str) -> Dict[str, Any]:
        skill = find_skill(drive, name, repo_path=repo_path)
        if not _wants_lifecycle_work(drive, skill):
            return {"deps_status": "skipped"}
        status, error = ensure_skill_dependencies(drive, skill)
        # A failure just recorded is not re-run by the turn-on in the same job (Enable retries it).
        turn_on = (reconcile_turn_on_when_ready(drive, name, repo_path=repo_path, reason="dependencies ready at startup")["action"]
                   if status in {"installed", "not_required"} else "not_ready")
        state = (extension_loader.reconcile_extension(name, drive, load_settings, repo_path=repo_path)
                 if skill.manifest.is_extension() else {})
        return {"deps_status": status, "error": error, "turn_on": turn_on, "extension_action": state.get("action")}

    for name in pending:
        try:
            run_lifecycle_job_blocking(
                kind="deps", target=name, source="startup", runner=lambda name=name: step(name),
                dedupe_key=f"dependencies:{name}",
                options=LifecycleJobOptions(drive_root=drive, result_error=lambda result: result.get("error", "")),
            )
        except Exception:
            logger.warning("Built-in dependency step failed for %s", name, exc_info=True)
    return pending


def ensure_builtin_skills(drive_root: Optional[pathlib.Path] = None, *, log_obj: Any = None,
                          repo_dir: Optional[pathlib.Path] = None, dependency_pass: bool = True) -> int:
    """Startup work for an existing data root: the one-time identity conversion (with install
    records for launcher-seeded built-ins), then, unless ``state/native_seed.json`` already names
    this committed bundled tree, re-seeding changed built-ins and releasing built-ins the app no
    longer bundles. The receipt covers those bundle phases alone. The package/turn-on pass over
    every installed skill runs off the boot path whether or not the receipt matches.
    Returns the number re-seeded."""
    from ouroboros.config import DATA_DIR
    from ouroboros.skill_identity import convert_skill_identities

    logger = log_obj or log
    drive = pathlib.Path(drive_root or DATA_DIR)
    seed_root = pathlib.Path(repo_dir) / "skills" if repo_dir is not None else bundled_skills_dir()
    repo = seed_root.parent
    try:
        convert_skill_identities(
            drive, log_obj=logger,
            on_skill=lambda skill, legacy: record_converted_builtin(drive, skill, legacy, seed_dir=seed_root),
        )
    except Exception:
        logger.warning("Skill identity conversion raised", exc_info=True)
    receipt = drive / "state" / "native_seed.json"
    identity = _seed_identity(repo)
    try:
        return _reconcile_bundle(drive, seed_root, repo, receipt, identity, logger)
    finally:
        if dependency_pass:  # after the bundle phases, whether or not the receipt matched
            threading.Thread(target=_dependency_pass, args=(drive, None, logger),
                             name="skill-dependencies", daemon=True).start()


def _reconcile_bundle(drive: pathlib.Path, seed_root: pathlib.Path, repo: pathlib.Path, receipt: pathlib.Path,
                      identity: Optional[Dict[str, Any]], logger: Any) -> int:
    from ouroboros.utils import atomic_write_json, read_json_dict

    if identity and read_json_dict(receipt) == identity:
        return 0
    try:
        receipt.unlink(missing_ok=True)  # a partly applied tree must not keep an older receipt
    except OSError:
        logger.warning("Built-in receipt could not be invalidated; reconciliation deferred", exc_info=True)
        return 0
    failures: List[str] = []
    upgraded = 0
    try:
        upgraded = reseed_changed_builtins(drive, seed_dir=seed_root, log_obj=logger, failures=failures)
    except Exception:
        failures.append("reseed")
        logger.warning("Built-in skill re-seed raised", exc_info=True)
    try:  # a successor that failed to re-seed keeps its retired partner until the next boot
        if not release_removed_builtins(drive, seed_dir=seed_root, log_obj=logger, retire=not failures):
            failures.append("release")
    except Exception:
        failures.append("release")
        logger.warning("Removed built-in release raised", exc_info=True)
    try:
        _offer_default_builtins(drive, seed_root, logger, failures)
    except Exception:
        failures.append("defaults")
        logger.warning("Default-on built-in offer raised", exc_info=True)
    if identity and not failures and identity == _seed_identity(repo):
        try:
            atomic_write_json(receipt, identity)
        except OSError:
            logger.warning("Built-in receipt not recorded; the next boot reconciles again", exc_info=True)
    return upgraded


__all__ = [
    "BUILTIN_SOURCE",
    "applies_here",
    "builtin_trust",
    "bundled_skills_dir",
    "ensure_builtin_skills",
    "install_builtin_skill",
    "record_converted_builtin",
    "release_removed_builtins",
    "reseed_changed_builtins",
]
