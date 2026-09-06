"""Managed git bootstrap helpers for the desktop launcher."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable


BUNDLE_REPO_NAME = "repo.bundle"
BUNDLE_MANIFEST_NAME = "repo_bundle_manifest.json"
MANAGED_REPO_META_NAME = "ouroboros-managed.json"
BOOTSTRAP_PIN_MARKER_NAME = "ouroboros-bootstrap-pending"
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MANAGED_REMOTE_NAME = "managed"
DEFAULT_MANAGED_LOCAL_BRANCH = "ouroboros"
DEFAULT_MANAGED_LOCAL_STABLE_BRANCH = "ouroboros-stable"
DEFAULT_MANAGED_REMOTE_STABLE_BRANCH = "ouroboros-stable"


@dataclass(frozen=True)
class BootstrapContext:
    bundle_dir: pathlib.Path
    repo_dir: pathlib.Path
    data_dir: pathlib.Path
    settings_path: pathlib.Path
    embedded_python: str
    app_version: str
    hidden_run: Callable[..., Any]
    save_settings: Callable[[dict], None]
    log: Any


def embedded_python_env(data_dir: pathlib.Path, base: dict[str, str] | None = None) -> dict[str, str]:
    """Env for running the EMBEDDED interpreter without writing into the bundle.

    The packaged interpreter lives INSIDE the signed application bundle, which is
    read-only in principle (and whose signature any write invalidates). Python
    has exactly two default write targets there and BOTH are redirected under the
    data dir, in one place, so a new caller cannot pick up one and miss the other:

    - bytecode caches -> ``PYTHONPYCACHEPREFIX``;
    - ``pip install`` -> the user site under ``PYTHONUSERBASE`` (paired with the
      ``--user`` flag at the pip call sites; ``site`` puts that directory on
      ``sys.path`` automatically, so every child launched through this env
      IMPORTS what was installed through it).

    ``PYTHONNOUSERSITE`` is cleared for the same reason: an inherited value would
    install to the user site and then refuse to import from it.

    DISCLOSED, not fixed: ``site`` puts the user site AHEAD of the bundle's own
    site-packages, and nothing prunes it. So a package installed here by an older
    version keeps OUTRANKING the signed bundle's copy for as long as the data dir
    survives — an upgrade can ship a newer dependency and still import the old one.
    Making that impossible means either pruning logic or versioning this directory,
    and versioning it silently abandons every dependency an existing install already
    paid to download. Both are product decisions with real cost, so the behaviour is
    stated here rather than changed unilaterally. Recovery is manual and total:
    delete ``<data>/state/python-userbase`` and relaunch.
    """
    env = dict(os.environ if base is None else base)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    state_dir = pathlib.Path(data_dir) / "state"
    pycache_dir = state_dir / "pycache"
    user_base = state_dir / "python-userbase"
    for directory in (pycache_dir, user_base):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    env["PYTHONPYCACHEPREFIX"] = str(pycache_dir)
    env["PYTHONUSERBASE"] = str(user_base)
    env.pop("PYTHONNOUSERSITE", None)
    return env


def check_git(is_windows: bool) -> bool:
    if shutil.which("git") is not None:
        return True
    if is_windows:
        for candidate in (
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Git", "cmd", "git.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Git", "cmd", "git.exe"),
        ):
            if os.path.isfile(candidate):
                git_dir = os.path.dirname(candidate)
                os.environ["PATH"] = git_dir + ";" + os.environ.get("PATH", "")
                return True
    return False


def _bundle_manifest_path(context: BootstrapContext) -> pathlib.Path:
    return context.bundle_dir / BUNDLE_MANIFEST_NAME


def _managed_meta_path(repo_dir: pathlib.Path) -> pathlib.Path:
    return repo_dir / ".git" / MANAGED_REPO_META_NAME


def _bootstrap_pin_marker_path(repo_dir: pathlib.Path) -> pathlib.Path:
    return repo_dir / ".git" / BOOTSTRAP_PIN_MARKER_NAME


def _read_json_file(path: pathlib.Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _normalize_bundle_manifest(raw: dict[str, Any], *, app_version: str) -> dict[str, Any]:
    manifest = dict(raw)
    return {
        "schema_version": int(manifest.get("schema_version") or MANIFEST_SCHEMA_VERSION),
        "bundle_file": str(manifest.get("bundle_file") or BUNDLE_REPO_NAME),
        "app_version": str(manifest.get("app_version") or app_version),
        "source_sha": str(manifest.get("source_sha") or ""),
        "release_tag": str(manifest.get("release_tag") or ""),
        "bundle_sha256": str(manifest.get("bundle_sha256") or ""),
        "source_branch": str(manifest.get("source_branch") or ""),
        "managed_remote_name": str(manifest.get("managed_remote_name") or DEFAULT_MANAGED_REMOTE_NAME),
        "managed_remote_url": str(manifest.get("managed_remote_url") or ""),
        "managed_remote_branch": str(manifest.get("managed_remote_branch") or manifest.get("source_branch") or ""),
        "managed_local_branch": str(manifest.get("managed_local_branch") or DEFAULT_MANAGED_LOCAL_BRANCH),
        "managed_local_stable_branch": str(
            manifest.get("managed_local_stable_branch") or DEFAULT_MANAGED_LOCAL_STABLE_BRANCH
        ),
        "managed_remote_stable_branch": str(
            manifest.get("managed_remote_stable_branch") or DEFAULT_MANAGED_REMOTE_STABLE_BRANCH
        ),
    }


def load_bundle_manifest(context: BootstrapContext) -> dict[str, Any]:
    manifest_path = _bundle_manifest_path(context)
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Embedded managed repo manifest is missing: {manifest_path}. "
            "Rebuild the app bundle with scripts/build_repo_bundle.py."
        )
    manifest = _normalize_bundle_manifest(_read_json_file(manifest_path), app_version=context.app_version)
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported managed repo manifest schema {manifest['schema_version']} "
            f"(expected {MANIFEST_SCHEMA_VERSION})."
        )
    if not manifest["source_sha"]:
        raise RuntimeError("Managed repo manifest is missing source_sha.")
    if not manifest["bundle_sha256"]:
        raise RuntimeError("Managed repo manifest is missing bundle_sha256.")
    if not manifest["managed_remote_branch"]:
        raise RuntimeError("Managed repo manifest is missing managed_remote_branch.")
    if manifest["app_version"] != context.app_version:
        raise RuntimeError(
            f"Managed repo manifest app_version {manifest['app_version']!r} does not "
            f"match launcher app version {context.app_version!r}."
        )
    expected_tag = f"v{manifest['app_version']}"
    if manifest["release_tag"] and manifest["release_tag"] != expected_tag:
        raise RuntimeError(
            f"Managed repo manifest release_tag {manifest['release_tag']!r} does not "
            f"match app_version {manifest['app_version']!r}."
        )
    _assert_bundle_integrity(context, manifest)
    return manifest


def load_repo_manifest(repo_dir: pathlib.Path) -> dict[str, Any]:
    meta_path = _managed_meta_path(repo_dir)
    if not meta_path.is_file():
        return {}
    return _read_json_file(meta_path)


def _write_repo_manifest(repo_dir: pathlib.Path, manifest: dict[str, Any]) -> None:
    # Atomic: a torn manifest makes the next launch misdetect the installed
    # repo generation and re-bootstrap over a healthy checkout.
    from ouroboros.utils import atomic_write_json

    atomic_write_json(_managed_meta_path(repo_dir), manifest, trailing_newline=True)


def _mark_bootstrap_pin_pending(repo_dir: pathlib.Path) -> None:
    marker = _bootstrap_pin_marker_path(repo_dir)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("pending\n", encoding="utf-8")


def _repo_manifest_matches(repo_dir: pathlib.Path, bundle_manifest: dict[str, Any]) -> bool:
    installed = load_repo_manifest(repo_dir)
    if not installed:
        return False
    tracked_keys = (
        "schema_version",
        "app_version",
        "source_sha",
        "release_tag",
        "bundle_sha256",
        "managed_remote_name",
        "managed_remote_url",
        "managed_remote_branch",
        "managed_local_branch",
        "managed_local_stable_branch",
        "managed_remote_stable_branch",
    )
    return all(str(installed.get(key) or "") == str(bundle_manifest.get(key) or "") for key in tracked_keys)


def _run_git(context: BootstrapContext, args: list[str], *, cwd: pathlib.Path, check: bool = True) -> Any:
    return context.hidden_run(
        args,
        cwd=str(cwd),
        check=check,
        capture_output=True,
        text=True,
    )


def _remote_url(context: BootstrapContext, repo_dir: pathlib.Path, remote_name: str) -> str:
    result = _run_git(context, ["git", "remote", "get-url", remote_name], cwd=repo_dir, check=False)
    return str(getattr(result, "stdout", "") or "").strip()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_bundle_integrity(context: BootstrapContext, manifest: dict[str, Any]) -> None:
    bundle_path = context.bundle_dir / manifest["bundle_file"]
    if not bundle_path.is_file():
        raise RuntimeError(
            f"Embedded managed repo bundle is missing: {bundle_path}. "
            "Rebuild the app bundle with scripts/build_repo_bundle.py."
        )
    actual_sha = _sha256_file(bundle_path)
    expected_sha = str(manifest.get("bundle_sha256") or "").strip()
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(
            f"Embedded managed repo bundle hash mismatch for {bundle_path}: "
            f"expected {expected_sha}, got {actual_sha}."
        )


def _archive_existing_repo(context: BootstrapContext, reason: str) -> pathlib.Path | None:
    if not context.repo_dir.exists():
        return None
    archive_root = context.data_dir / "archive" / "managed_repo"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_dir = archive_root / f"{int(time.time())}-{uuid.uuid4().hex[:8]}-{reason}"
    shutil.move(str(context.repo_dir), str(archive_dir))
    context.log.info("Archived existing repo to %s (%s)", archive_dir, reason)
    return archive_dir


def _remove_if_exists(path: pathlib.Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _configure_managed_clone(context: BootstrapContext, repo_dir: pathlib.Path, manifest: dict[str, Any]) -> None:
    source_sha = str(manifest.get("source_sha") or "").strip()
    local_branch = manifest["managed_local_branch"]
    local_stable_branch = manifest["managed_local_stable_branch"]
    remote_name = manifest["managed_remote_name"]
    remote_url = manifest["managed_remote_url"]

    source_sha_check = _run_git(
        context,
        ["git", "rev-parse", "--verify", source_sha],
        cwd=repo_dir,
        check=False,
    )
    if getattr(source_sha_check, "returncode", 1) != 0:
        raise RuntimeError(
            f"Embedded managed repo bundle does not contain manifest source_sha {source_sha}."
        )
    _run_git(context, ["git", "checkout", "-B", local_branch, source_sha], cwd=repo_dir)
    head = _run_git(context, ["git", "rev-parse", "HEAD"], cwd=repo_dir)
    head_sha = str(getattr(head, "stdout", "") or "").strip()
    if head_sha != source_sha:
        raise RuntimeError(
            f"Managed repo bootstrap checked out {head_sha or '(unknown)'} but manifest "
            f"requires {source_sha}."
        )
    if local_stable_branch and local_stable_branch != local_branch:
        if head_sha:
            _run_git(context, ["git", "branch", "-f", local_stable_branch, head_sha], cwd=repo_dir)

    origin = _run_git(context, ["git", "remote"], cwd=repo_dir, check=False)
    existing_remotes = {
        line.strip() for line in str(getattr(origin, "stdout", "") or "").splitlines() if line.strip()
    }
    if "origin" in existing_remotes:
        _run_git(context, ["git", "remote", "remove", "origin"], cwd=repo_dir, check=False)
    if remote_name in existing_remotes:
        _run_git(context, ["git", "remote", "remove", remote_name], cwd=repo_dir, check=False)
    if remote_url:
        _run_git(context, ["git", "remote", "add", remote_name, remote_url], cwd=repo_dir)

    _run_git(context, ["git", "config", "user.name", "Ouroboros"], cwd=repo_dir, check=False)
    _run_git(context, ["git", "config", "user.email", "ouroboros@local.mac"], cwd=repo_dir, check=False)
    _write_repo_manifest(repo_dir, manifest)
    _mark_bootstrap_pin_pending(repo_dir)


def _ensure_managed_remote(context: BootstrapContext, repo_dir: pathlib.Path, manifest: dict[str, Any]) -> None:
    remote_name = manifest["managed_remote_name"]
    remote_url = manifest["managed_remote_url"]

    remotes = _run_git(context, ["git", "remote"], cwd=repo_dir, check=False)
    existing_remotes = {
        line.strip() for line in str(getattr(remotes, "stdout", "") or "").splitlines() if line.strip()
    }
    if remote_url:
        if remote_name in existing_remotes:
            _run_git(context, ["git", "remote", "set-url", remote_name, remote_url], cwd=repo_dir)
        else:
            _run_git(context, ["git", "remote", "add", remote_name, remote_url], cwd=repo_dir)

    _run_git(context, ["git", "config", "user.name", "Ouroboros"], cwd=repo_dir, check=False)
    _run_git(context, ["git", "config", "user.email", "ouroboros@local.mac"], cwd=repo_dir, check=False)
    _write_repo_manifest(repo_dir, manifest)


def _clone_repo_from_bundle(context: BootstrapContext, manifest: dict[str, Any]) -> pathlib.Path:
    bundle_path = context.bundle_dir / manifest["bundle_file"]
    if not bundle_path.is_file():
        raise RuntimeError(
            f"Embedded managed repo bundle is missing: {bundle_path}. "
            "Rebuild the app bundle with scripts/build_repo_bundle.py."
        )

    temp_repo = context.repo_dir.parent / f".repo-bootstrap-{uuid.uuid4().hex[:8]}"
    _remove_if_exists(temp_repo)
    try:
        _run_git(context, ["git", "clone", str(bundle_path), str(temp_repo)], cwd=context.bundle_dir)
        _configure_managed_clone(context, temp_repo, manifest)
        return temp_repo
    except Exception:
        _remove_if_exists(temp_repo)
        raise


def _install_managed_repo(context: BootstrapContext, manifest: dict[str, Any], *, reason: str) -> str:
    preserved_origin_url = _remote_url(context, context.repo_dir, "origin") if (context.repo_dir / ".git").exists() else ""
    archived_repo = _archive_existing_repo(context, reason)
    temp_repo = _clone_repo_from_bundle(context, manifest)
    try:
        shutil.move(str(temp_repo), str(context.repo_dir))
        if preserved_origin_url:
            _run_git(context, ["git", "remote", "add", "origin", preserved_origin_url], cwd=context.repo_dir, check=False)
    except Exception:
        _remove_if_exists(temp_repo)
        if archived_repo is not None and not context.repo_dir.exists():
            shutil.move(str(archived_repo), str(context.repo_dir))
        raise
    return "replaced" if archived_repo is not None else "created"


def ensure_managed_repo(context: BootstrapContext) -> str:
    """Ensure REPO_DIR is a managed git clone backed by the embedded bundle."""
    manifest = load_bundle_manifest(context)
    if not context.repo_dir.exists():
        return _install_managed_repo(context, manifest, reason="missing")
    if not (context.repo_dir / ".git").exists():
        return _install_managed_repo(context, manifest, reason="legacy-no-git")
    if not _repo_manifest_matches(context.repo_dir, manifest):
        _ensure_managed_remote(context, context.repo_dir, manifest)
        context.log.info(
            "Updated managed repo metadata for embedded bundle without replacing local checkout."
        )
        return "metadata-updated"

    _ensure_managed_remote(context, context.repo_dir, manifest)
    return "unchanged"


def sync_existing_repo_from_bundle(context: BootstrapContext) -> None:
    """Reconcile the managed repo against the embedded bundle metadata."""
    outcome = ensure_managed_repo(context)
    context.log.info("Managed repo sync outcome: %s", outcome)


def _pip_output_tail(result: Any, limit: int = 800) -> str:
    """Last chars of a captured pip run, whether it captured bytes or text."""
    parts = []
    for stream in (getattr(result, "stdout", None), getattr(result, "stderr", None)):
        if isinstance(stream, bytes):
            stream = stream.decode("utf-8", errors="replace")
        if isinstance(stream, str) and stream.strip():
            parts.append(stream.strip())
    return ("\n".join(parts))[-limit:]


def install_deps(context: BootstrapContext) -> bool:
    """Install/update Python deps for the embedded interpreter.

    ``pip_install_target_args`` keeps a packaged install OUT of the signed bundle
    the embedded interpreter lives in; the return value reports whether pip
    actually succeeded.
    """
    try:
        requirements = context.repo_dir / "requirements-runtime.lock"
        if not requirements.exists():
            # Managed repositories from before the uv migration may still use
            # the legacy filename during an in-place upgrade.
            requirements = context.repo_dir / "requirements.txt"
        if not requirements.exists():
            return True
        from ouroboros.platform_layer import pip_install_target_args

        result = context.hidden_run(
            [context.embedded_python, "-m", "pip", "install",
             *pip_install_target_args(context.embedded_python), "-r", str(requirements)],
            env=embedded_python_env(context.data_dir),
            timeout=240,
            capture_output=True,
        )
    except Exception as exc:
        context.log.warning("Dependency install/update failed: %s", exc)
        return False
    # A non-zero pip exit is a real, actionable failure (unreachable index, an
    # unbuildable wheel, a read-only target): silently discarding it leaves the
    # app to fail much later with an unexplained ImportError.
    returncode = int(getattr(result, "returncode", 0) or 0)
    if returncode != 0:
        context.log.warning(
            "Dependency install/update pip exited %d; the runtime may be missing packages:\n%s",
            returncode, _pip_output_tail(result),
        )
        return False
    return True


_SEED_COMPLETE_MARKER = ".bootstrap-seed-complete"
_POST_BOOTSTRAP_NEW_NATIVE_SEEDS = frozenset({"telegram", "unix_computer_use"})


def _read_skill_manifest_version(skill_dir: pathlib.Path) -> str:
    """Return a seed skill manifest version via the shared parser, or ``""``."""
    for candidate in ("SKILL.md", "skill.json"):
        path = skill_dir / candidate
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
            manifest = parse_skill_manifest_text(text)
        except Exception:
            return ""
        return str(manifest.version or "").strip()
    return ""


def _reseed_native_skill_in_place(
    seed_skill: pathlib.Path,
    target_skill: pathlib.Path,
    log_obj: Any,
    *,
    drive_root: pathlib.Path | None = None,
    skill_name: str | None = None,
    old_version: str = "",
    new_version: str = "",
) -> bool:
    """Replace launcher-owned native skill files in place.

    Refreshes the hash-pinned native-trust verdict (review.json) for the new
    payload when ``drive_root`` is provided; an explicit owner enable/disable
    choice is preserved (auto-enable fires only when no choice exists yet).
    """
    try:
        if target_skill.exists():
            shutil.rmtree(target_skill)
        shutil.copytree(seed_skill, target_skill)
        # Preserve launcher-owned provenance on the replacement copy.
        (target_skill / ".seed-origin").write_text(
            f"seeded_from={seed_skill.parent.name}\nupgrade=true\n",
            encoding="utf-8",
        )
        if drive_root is not None:
            # The launcher just wrote repo-reviewed bytes: refresh the
            # hash-pinned native-trust verdict for the new payload.
            _stamp_native_seed_trust(pathlib.Path(drive_root), target_skill, log_obj)
            _carry_grants_across_reseed(pathlib.Path(drive_root), target_skill, log_obj)
        return True
    except OSError as exc:
        log_obj.warning(
            "Failed to upgrade native skill in place %s -> %s: %s",
            seed_skill, target_skill, exc,
        )
        return False


def _per_skill_version_resync(
    seed_dir: pathlib.Path,
    native_root: pathlib.Path,
    log_obj: Any,
    *,
    drive_root: pathlib.Path | None = None,
) -> int:
    """Re-seed only marker-owned native skills whose seed version changed."""
    if not seed_dir.is_dir() or not native_root.is_dir():
        return 0
    upgraded = 0
    for entry in sorted(seed_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if not any((entry / candidate).is_file() for candidate in ("SKILL.md", "skill.json")):
            continue
        target = native_root / entry.name
        if not target.exists():
            # Respect deletion/absence; resync upgrades only existing skills.
            continue
        if not (target / ".seed-origin").is_file():
            # User-managed skill in native/: never touch.
            continue
        seed_version = _read_skill_manifest_version(entry)
        target_version = _read_skill_manifest_version(target)
        if not seed_version or not target_version:
            continue
        if seed_version == target_version:
            continue
        log_obj.info(
            "Native skill %s version drift (seed=%s, installed=%s) — re-seeding",
            entry.name, seed_version, target_version,
        )
        if _reseed_native_skill_in_place(
            entry, target, log_obj,
            drive_root=drive_root,
            skill_name=entry.name,
            old_version=target_version,
            new_version=seed_version,
        ):
            upgraded += 1
    return upgraded


# Auto-enable exemption: tool registration + local subprocess are the basic
# extension substrate. Externally-facing capabilities (net, route, widget,
# inject_chat, subscribe_event, ...) keep the skill DISABLED until the owner
# enables it, even though the native-trust review verdict is stamped.
_NATIVE_AUTO_ENABLE_EXEMPT_PERMISSIONS = frozenset({"tool", "subprocess"})


def _stamp_native_seed_trust(
    drive_root: pathlib.Path,
    skill_dir: pathlib.Path,
    log_obj: Any,
) -> None:
    """Hash-pin a native-trust review verdict for a launcher-seeded skill.

    The payload bytes shipped through the repo commit gate (triad+scope), so a
    launcher-written native skill gets ``status=clean`` bound to the payload
    hash computed AFTER seeding (control files excluded from the hash). Any
    later edit flips the verdict stale exactly like an ordinary review.
    Zero-grant skills (no secret keys, no privileged permissions, only
    tool/subprocess surface) also auto-enable. Owner opt-out:
    ``OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS=false``. Never raises.
    """
    try:
        from ouroboros.config import get_trust_native_seeded_skills

        if not get_trust_native_seeded_skills():
            return
        from ouroboros.skill_loader import (
            SkillReviewState,
            load_skill,
            requested_core_setting_keys,
            requested_skill_permissions,
            save_enabled,
            save_review_state,
        )
        from ouroboros.utils import utc_now_iso

        skill = load_skill(skill_dir, drive_root)
        if skill is None or skill.load_error or not skill.content_hash:
            log_obj.warning(
                "Native-trust stamp skipped for %s: %s",
                skill_dir.name,
                (skill.load_error if skill else "no manifest"),
            )
            return
        # ABI-1: native_seed is a PASS-minting path and shares the NEW-PASS
        # admission predicate — a seeded extension without the plugin_api
        # field gets NO trust verdict (bundled skills ship the field).
        from ouroboros.contracts.plugin_api import extension_new_pass_admission_error

        admission_error = extension_new_pass_admission_error(skill.manifest)
        if admission_error:
            log_obj.warning(
                "Native-trust stamp refused for %s (PluginAPI 2.0 admission): %s",
                skill.name, admission_error,
            )
            return
        save_review_state(
            drive_root,
            skill.name,
            SkillReviewState(
                status="clean",
                content_hash=skill.content_hash,
                findings=[],
                reviewer_models=["repo_commit_gate"],
                timestamp=utc_now_iso(),
                review_profile="native_seed",
            ),
        )
        keys = requested_core_setting_keys(list(skill.manifest.env_from_settings or []))
        perms = requested_skill_permissions(
            list(skill.manifest.permissions or []),
            list(skill.manifest.subscribe_events or []),
        )
        zero_grant = (
            not keys
            and not perms
            and set(skill.manifest.permissions or []) <= _NATIVE_AUTO_ENABLE_EXEMPT_PERMISSIONS
        )
        # Owner sovereignty: auto-enable only when NO explicit choice exists.
        # A version resync must never override an owner's disable (same
        # precedent as OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS: existing explicit
        # choices are preserved).
        from ouroboros.skill_loader import skill_state_dir

        no_explicit_choice = not (skill_state_dir(drive_root, skill.name) / "enabled.json").exists()
        auto_enabled = bool(zero_grant and no_explicit_choice)
        if auto_enabled:
            save_enabled(drive_root, skill.name, True)
        log_obj.info(
            "Native-trust review stamped for seeded skill %s (auto_enabled=%s)",
            skill.name, auto_enabled,
        )
    except Exception:
        log_obj.warning("Native-trust stamp failed for %s", skill_dir, exc_info=True)


def _carry_grants_across_reseed(
    drive_root: pathlib.Path,
    skill_dir: pathlib.Path,
    log_obj: Any,
) -> None:
    """Carry owner grants across a native re-seed when the requested sets are
    unchanged (owner «A», v7next §6.1-Δ). Grants are content-hash-bound, so a
    version resync used to orphan them (the telegram precedent) even though
    the skill requests EXACTLY the same keys/permissions; a changed requested
    set keeps the ordinary re-grant flow. Never raises."""
    try:
        from ouroboros.skill_loader import (
            load_skill,
            load_skill_grants,
            requested_core_setting_keys,
            requested_skill_permissions,
            save_skill_grants,
        )

        skill = load_skill(skill_dir, drive_root)
        if skill is None or skill.load_error or not skill.content_hash:
            return
        grants = load_skill_grants(drive_root, skill.name)
        old_hash = str(grants.get("content_hash") or "")
        if not old_hash or old_hash == skill.content_hash:
            return
        granted_keys = list(grants.get("granted_keys") or [])
        granted_permissions = list(grants.get("granted_permissions") or [])
        if not granted_keys and not granted_permissions:
            return
        new_keys = requested_core_setting_keys(list(skill.manifest.env_from_settings or []))
        new_permissions = requested_skill_permissions(
            list(skill.manifest.permissions or []),
            list(skill.manifest.subscribe_events or []),
        )
        if sorted(grants.get("requested_keys") or []) != sorted(new_keys):
            return
        if sorted(grants.get("requested_permissions") or []) != sorted(new_permissions):
            return
        save_skill_grants(
            drive_root,
            skill.name,
            granted_keys,
            content_hash=skill.content_hash,
            requested_keys=new_keys,
            granted_permissions=granted_permissions,
            requested_permissions=new_permissions,
        )
        log_obj.info(
            "Carried owner grants across native re-seed for %s (requested sets unchanged)",
            skill.name,
        )
    except Exception:
        log_obj.warning("Grant carry across re-seed failed for %s", skill_dir, exc_info=True)


def _migrate_control_file_hashes(target_root: pathlib.Path, log_obj: Any) -> None:
    """One-shot re-pin for reviews recorded with the legacy control-file hash.

    v6.31 excluded lifecycle control files (``.seed-origin``) from the payload
    hash; a review pinned to the LEGACY hash would go stale on update without
    any payload change. When the stored hash equals the legacy-computed hash
    (proving no edits happened), re-pin it to the new-semantics hash with the
    same verdict. Never raises.
    """
    native_root = target_root / "native"
    if not native_root.is_dir():
        return
    try:
        from ouroboros.skill_loader import (
            compute_content_hash,
            load_skill,
            save_review_state,
        )
    except Exception:
        return
    # Only native-bucket payloads are affected by the exemption (the hash for
    # other buckets is byte-identical before/after the semantics change), so
    # scanning native/ alone is complete by construction.
    for entry in sorted(native_root.iterdir()):
        try:
            if not entry.is_dir() or not (entry / ".seed-origin").is_file():
                continue
            drive_root = target_root.parent
            skill = load_skill(entry, drive_root)
            if skill is None or skill.load_error or not skill.content_hash:
                continue
            review = skill.review
            if not review.content_hash or review.content_hash == skill.content_hash:
                continue
            legacy_hash = compute_content_hash(
                entry,
                manifest_entry=skill.manifest.entry,
                manifest_scripts=skill.manifest.scripts,
                include_control_files=True,
            )
            if review.content_hash != legacy_hash:
                continue  # genuinely edited payload — stays stale
            review.content_hash = skill.content_hash
            save_review_state(drive_root, skill.name, review)
            log_obj.info(
                "Re-pinned legacy control-file review hash for native skill %s",
                skill.name,
            )
        except Exception:
            log_obj.debug("Control-file hash migration failed for %s", entry, exc_info=True)


def _seed_skills_into(seed_dir: pathlib.Path, target_root: pathlib.Path, log_obj: Any) -> int:
    """Copy seed skills into ``target_root/native/`` once, guarded by a marker.

    The marker preserves deletion intent: after first bootstrap, an empty
    native bucket is not auto-populated again.
    """
    if not seed_dir.is_dir():
        return 0
    native_root = target_root / "native"
    try:
        # Prefer the canonical layout helper; fall back to manual mkdir.
        try:
            from ouroboros.config import ensure_data_skills_dir
            ensure_data_skills_dir(target_root.parent)
        except Exception:
            target_root.mkdir(parents=True, exist_ok=True)
            for bucket in ("native", "clawhub", "external"):
                (target_root / bucket).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log_obj.warning("Skills data root setup failed: %s", exc)
        return 0

    marker_path = native_root / _SEED_COMPLETE_MARKER
    if marker_path.is_file():
        # Bootstrap already ran; do not resurrect deleted seed skills.
        try:
            from ouroboros.skill_loader import discover_skills

            installed_names = {
                skill.name for skill in discover_skills(target_root.parent)
            }
        except Exception:
            log_obj.warning(
                "New native skill seeding skipped: installed skill names could not be inspected",
                exc_info=True,
            )
            return 0
        copied = 0
        for name in sorted(_POST_BOOTSTRAP_NEW_NATIVE_SEEDS):
            entry = seed_dir / name
            dest = native_root / name
            offered_marker = native_root / f".post-bootstrap-seed-{name}"
            if offered_marker.exists():
                continue
            if dest.exists() or not any((entry / candidate).is_file() for candidate in ("SKILL.md", "skill.json")):
                continue
            if name in installed_names:
                log_obj.warning(
                    "New bundled native skill %s was not seeded because an installed payload already uses that name",
                    name,
                )
                continue
            try:
                shutil.copytree(entry, dest)
                (dest / ".seed-origin").write_text(
                    f"seeded_from={seed_dir.name}\npost_bootstrap_new_seed=true\n",
                    encoding="utf-8",
                )
                offered_marker.write_text("offered\n", encoding="utf-8")
                copied += 1
                installed_names.add(name)
                _stamp_native_seed_trust(target_root.parent, dest, log_obj)
            except OSError as exc:
                log_obj.warning("Failed to copy new bundled native skill %s -> %s: %s", entry, dest, exc)
        return copied

    # Existing unmarked native content is treated as user-managed; mark complete
    # without copying to avoid clobbering it.
    try:
        existing = [p for p in native_root.iterdir() if not p.name.startswith(".")]
    except OSError:
        existing = []
    if existing:
        try:
            marker_path.write_text(
                "Bootstrap inferred from pre-existing native/ contents.\n",
                encoding="utf-8",
            )
        except OSError as exc:
            log_obj.warning("Failed to write %s: %s", marker_path, exc)
        return 0

    copied = 0
    for entry in sorted(seed_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if not any((entry / candidate).is_file() for candidate in ("SKILL.md", "skill.json")):
            continue
        dest = native_root / entry.name
        if dest.exists():
            continue
        try:
            shutil.copytree(entry, dest)
            # Per-skill marker lets source classification prove launcher ownership.
            (dest / ".seed-origin").write_text(
                f"seeded_from={seed_dir.name}\n", encoding="utf-8",
            )
            copied += 1
            _stamp_native_seed_trust(target_root.parent, dest, log_obj)
        except OSError as exc:
            log_obj.warning("Failed to copy seed skill %s -> %s: %s", entry, dest, exc)

    # Always mark completion so subsequent launches do not retry seeding.
    try:
        marker_path.write_text(
            f"Bootstrap-seed completed; copied {copied} skill(s) from {seed_dir}.\n",
            encoding="utf-8",
        )
    except OSError as exc:
        log_obj.warning("Failed to write %s: %s", marker_path, exc)

    if copied:
        log_obj.info(
            "Bootstrapped %d native skill(s) from seed %s into %s",
            copied, seed_dir, native_root,
        )
    return copied


def ensure_data_skills_seeded() -> int:
    """Seed native skills once, then version-resync marker-owned native skills."""
    import logging as _logging
    from ouroboros.config import DATA_DIR, REPO_DIR

    log_obj = _logging.getLogger(__name__)
    seed_dir = pathlib.Path(REPO_DIR) / "skills"
    target_root = pathlib.Path(DATA_DIR) / "skills"
    copied = _seed_skills_into(seed_dir, target_root, log_obj)
    drive_root = pathlib.Path(DATA_DIR)
    try:
        # One-shot v6.31 hash-semantics migration (control files left the hash).
        _migrate_control_file_hashes(target_root, log_obj)
    except Exception:  # pragma: no cover - defensive
        log_obj.warning("Control-file hash migration raised", exc_info=True)
    try:
        upgraded = _per_skill_version_resync(
            seed_dir, target_root / "native", log_obj,
            drive_root=drive_root,
        )
    except Exception:  # pragma: no cover - defensive
        log_obj.warning("Native skill version-resync raised", exc_info=True)
        upgraded = 0
    try:
        cleanup_orphaned_seed_markers(seed_dir, target_root / "native", log_obj)
    except Exception:  # pragma: no cover - defensive
        log_obj.warning("Orphaned seed-marker cleanup raised", exc_info=True)
    return copied + upgraded


def cleanup_orphaned_seed_markers(
    seed_dir: pathlib.Path,
    native_root: pathlib.Path,
    log_obj,
) -> None:
    """Strip seed markers for native skills no longer shipped in ``repo/skills``.

    Payloads stay in place; only ownership is reclassified to user-managed.
    """
    if not native_root.is_dir():
        return
    for entry in native_root.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        marker = entry / ".seed-origin"
        if not marker.is_file():
            continue
        if (seed_dir / entry.name).is_dir():
            continue
        try:
            marker.unlink()
            log_obj.info(
                "Native skill %r seed has been removed from repo/skills/; "
                "re-classifying installed copy as external (user-managed).",
                entry.name,
            )
        except OSError:  # pragma: no cover - defensive
            log_obj.warning(
                "Failed to strip orphaned .seed-origin from %s",
                entry, exc_info=True,
            )


def bootstrap_native_skills(context: BootstrapContext) -> None:
    """Best-effort one-time copy of ``repo/skills/*`` into the data plane."""
    _seed_skills_into(
        context.repo_dir / "skills",
        context.data_dir / "skills",
        context.log,
    )


def bootstrap_repo(context: BootstrapContext) -> bool:
    """Ensure the launcher-managed git repo exists and matches the embedded bundle.

    Returns whether the dependency install (when one was needed) succeeded —
    a freshly created/synced checkout that could not land its requirements is
    a fact the caller must be able to disclose, not a silently swallowed pip
    exit code that resurfaces later as an unexplained ImportError.
    """
    context.data_dir.mkdir(parents=True, exist_ok=True)
    outcome = ensure_managed_repo(context)
    context.log.info("Bootstrapping managed repository to %s (outcome=%s)", context.repo_dir, outcome)

    try:
        memory_dir = context.data_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        world_path = memory_dir / "WORLD.md"
        if not world_path.exists():
            env = os.environ.copy()
            env["PYTHONPATH"] = str(context.repo_dir)
            context.hidden_run(
                [
                    context.embedded_python,
                    "-c",
                    f"import sys; sys.path.insert(0, '{context.repo_dir}'); "
                    f"from ouroboros.world_profiler import generate_world_profile; "
                    f"generate_world_profile('{world_path}')",
                ],
                env=embedded_python_env(context.data_dir, env),
                timeout=30,
                capture_output=True,
            )
    except Exception as exc:
        context.log.warning("World profile generation failed: %s", exc)

    bootstrap_native_skills(context)
    deps_ok = True
    if outcome != "unchanged":
        deps_ok = install_deps(context)
        if not deps_ok:
            context.log.warning(
                "Bootstrap finished with a FAILED dependency install (repo outcome=%s): "
                "the checkout changed but its requirements did not land, so the server "
                "may be missing packages. See the pip output above for the cause.",
                outcome,
            )
    context.log.info("Bootstrap complete.")
    return deps_ok
