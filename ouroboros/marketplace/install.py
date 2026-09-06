"""Install, update, and uninstall ClawHub skills in the data plane.

The synchronous pipeline keeps registry lookup, download, staging, adaptation,
atomic landing, review, dependency install, and provenance as separate helpers
so routes can move work across ``asyncio.to_thread`` boundaries.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import shutil
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ouroboros.marketplace.adapter import AdapterResult, adapt_openclaw_skill
from ouroboros.marketplace.clawhub import (
    ClawHubArchive,
    ClawHubClientError,
    ClawHubSkillSummary,
    download as _registry_download,
    info as _registry_info,
)
from ouroboros.marketplace.fetcher import (
    FetchError,
    StagedSkill,
    land_staged_tree,
    stage as _stage_archive,
)
from ouroboros.marketplace.install_specs import install_specs_hash
from ouroboros.marketplace.isolated_deps import DEPS_STATE_FILENAME, install_isolated_dependencies, read_deps_state
from ouroboros.marketplace.provenance import (
    delete_provenance,
    read_provenance,
    write_provenance,
)
from ouroboros.skill_review_status import skill_review_gate
from ouroboros.utils import atomic_write_json, read_json_dict

log = logging.getLogger(__name__)


def _registry_error_status(exc: Exception) -> int:
    """Map a registry-client error to the HTTP status the gateway should surface
    (404 for a missing slug, 429 for rate limiting); 0 means use the default so
    the install/update routes preserve upstream semantics instead of a blanket 400."""
    from ouroboros.marketplace.clawhub import ClawHubNotFoundError, ClawHubRateLimitError

    if isinstance(exc, ClawHubNotFoundError):
        return 404
    if isinstance(exc, ClawHubRateLimitError):
        return 429
    return 0


@dataclass
class InstallResult:
    """Outcome of ``install_skill``."""

    ok: bool
    sanitized_name: str
    target_dir: Optional[pathlib.Path] = None
    summary: Optional[ClawHubSkillSummary] = None
    archive: Optional[ClawHubArchive] = None
    staged: Optional[StagedSkill] = None
    adapter: Optional[AdapterResult] = None
    review_status: str = ""
    review_findings: List[Dict[str, Any]] = field(default_factory=list)
    review_error: str = ""
    deps_status: str = ""
    deps_error: str = ""
    deps_fingerprint: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    # Upstream HTTP status to surface (404/429); 0 means the gateway uses its
    # default for a failed install (400). Preserves registry semantics end-to-end.
    error_status: int = 0
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UninstallResult:
    ok: bool
    sanitized_name: str
    error: str = ""


@dataclass
class PayloadRollbackSnapshot:
    """Rollback handle for marketplace payload/provenance/deps state."""

    drive_root: pathlib.Path
    skill_name: str
    target_dir: pathlib.Path
    backup_dir: Optional[pathlib.Path] = None
    state_provenance: Optional[Dict[str, Any]] = None
    deps_state: Optional[Dict[str, Any]] = None


def snapshot_payload_state(
    drive_root: pathlib.Path,
    skill_name: str,
    target_dir: pathlib.Path,
    *,
    include_state_provenance: bool = False,
) -> PayloadRollbackSnapshot:
    """Copy current live payload/state so a failed install/update can restore it."""

    drive_root = pathlib.Path(drive_root)
    target_dir = pathlib.Path(target_dir)
    backup_dir: Optional[pathlib.Path] = None
    if target_dir.exists():
        rollback_root = target_dir.parent / ".rollback"
        rollback_root.mkdir(parents=True, exist_ok=True)
        backup_dir = rollback_root / f"{target_dir.name}.{uuid.uuid4().hex}"
        shutil.copytree(target_dir, backup_dir)
    deps_path = _deps_state_path(drive_root, skill_name)
    deps_state = read_json_dict(deps_path) if deps_path.is_file() else None
    return PayloadRollbackSnapshot(
        drive_root=drive_root,
        skill_name=skill_name,
        target_dir=target_dir,
        backup_dir=backup_dir,
        state_provenance=read_provenance(drive_root, skill_name) if include_state_provenance else None,
        deps_state=deps_state if isinstance(deps_state, dict) else None,
    )


def _deps_state_path(drive_root: pathlib.Path, skill_name: str) -> pathlib.Path:
    from ouroboros.skill_loader import skill_state_dir

    return skill_state_dir(drive_root, skill_name) / DEPS_STATE_FILENAME


def _write_deps_state(drive_root: pathlib.Path, skill_name: str, state: Dict[str, Any]) -> None:
    if not state:
        return
    atomic_write_json(_deps_state_path(drive_root, skill_name), state, trailing_newline=True)


def _valid_existing_clawhub_provenance(
    drive_root: pathlib.Path,
    skill_name: str,
    target_dir: pathlib.Path,
    *,
    slug: str,
) -> Optional[Dict[str, Any]]:
    sidecar = read_json_dict(pathlib.Path(target_dir) / ".clawhub.json") or {}
    durable = read_provenance(drive_root, skill_name) or {}
    if (
        str(sidecar.get("source") or "") != "clawhub"
        or str(sidecar.get("slug") or "") != str(slug or "")
        or str(sidecar.get("sanitized_name") or "") != skill_name
    ):
        return None
    if (
        str(durable.get("source") or "") != "clawhub"
        or str(durable.get("slug") or "") != str(slug or "")
        or str(durable.get("sanitized_name") or "") != skill_name
    ):
        return None
    return durable


def _has_repairable_clawhub_partial(drive_root: pathlib.Path, skill_name: str, target_dir: pathlib.Path) -> bool:
    target = pathlib.Path(target_dir)
    return (
        (_deps_state_path(drive_root, skill_name)).is_file()
        or (target / ".ouroboros_env").exists()
        or (target / ".clawhub.json").is_file()
    )


def restore_payload_state(snapshot: PayloadRollbackSnapshot, *, restore_state_provenance: bool = False) -> None:
    """Restore or remove the live payload plus deps/provenance after a failed transaction."""

    target_dir = pathlib.Path(snapshot.target_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    if snapshot.backup_dir is not None and snapshot.backup_dir.exists():
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot.backup_dir.rename(target_dir)
    deps_path = _deps_state_path(snapshot.drive_root, snapshot.skill_name)
    if snapshot.deps_state is None:
        deps_path.unlink(missing_ok=True)
    else:
        atomic_write_json(deps_path, snapshot.deps_state, trailing_newline=True)
    if restore_state_provenance:
        if snapshot.state_provenance is None:
            delete_provenance(snapshot.drive_root, snapshot.skill_name)
        else:
            write_provenance(snapshot.drive_root, snapshot.skill_name, snapshot.state_provenance)


def discard_payload_snapshot(snapshot: PayloadRollbackSnapshot) -> None:
    """Drop rollback copies after a successful marketplace transaction."""

    if snapshot.backup_dir is not None:
        shutil.rmtree(snapshot.backup_dir, ignore_errors=True)


def _clawhub_skills_root(drive_root: pathlib.Path) -> pathlib.Path:
    """Return the ClawHub skills bucket, creating the canonical layout."""
    try:
        from ouroboros.config import ensure_data_skills_dir
        ensure_data_skills_dir(pathlib.Path(drive_root))
    except ImportError:
        pass
    target = pathlib.Path(drive_root) / "skills" / "clawhub"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _land_staged_into_data_plane(
    staged: StagedSkill,
    target_dir: pathlib.Path,
    *,
    overwrite: bool,
) -> None:
    """Atomically land staged content, preserving the old tree until success."""
    target_dir = pathlib.Path(target_dir)
    if target_dir.exists():
        if not overwrite:
            raise RuntimeError(
                f"Target {target_dir} already exists — use overwrite=True to replace"
            )
    land_staged_tree(staged.staging_dir, target_dir, replacement_suffix=f"replaced-{staged.sha256[:8]}")


class _MarketplaceReviewCtx:
    """Minimal ToolContext-compatible carrier for headless auto-review."""

    def __init__(self, drive_root: pathlib.Path, repo_dir: pathlib.Path) -> None:
        self.drive_root: pathlib.Path = pathlib.Path(drive_root)
        self.repo_dir: pathlib.Path = pathlib.Path(repo_dir)
        self.task_id: Any = "marketplace_install"
        self.current_chat_id: Any = 0
        self.pending_events: List[Any] = []
        self.emit_progress_fn = lambda _msg: None
        self.event_queue = None  # _emit_usage_event tolerates None
        self.messages: List[Any] = []

    def repo_path(self, rel: str) -> pathlib.Path:
        return (self.repo_dir / rel).resolve()

    def drive_path(self, rel: str) -> pathlib.Path:
        return (self.drive_root / rel).resolve()

    def drive_logs(self) -> pathlib.Path:
        target = self.drive_root / "logs"
        target.mkdir(parents=True, exist_ok=True)
        return target


def _run_skill_review(
    drive_root: pathlib.Path,
    repo_dir: pathlib.Path,
    skill_name: str,
) -> tuple[str, List[Dict[str, Any]], str]:
    """Run ``review_skill``; missing review code leaves install pending."""
    try:
        from ouroboros.skill_review import review_skill as _review_skill_impl
    except ImportError as exc:
        return "pending", [], f"review pipeline unavailable: {exc}"

    try:
        outcome = _review_skill_impl(
            _MarketplaceReviewCtx(drive_root, repo_dir), skill_name
        )
    except Exception as exc:
        log.exception("review_skill raised during marketplace install")
        return "pending", [], f"review_skill raised: {type(exc).__name__}: {exc}"
    return (
        str(outcome.status or "pending"),
        list(outcome.findings or []),
        str(outcome.error or ""),
    )


def dedupe_marketplace_skill_name(
    drive_root: pathlib.Path,
    target_root: pathlib.Path,
    base_name: str,
    *,
    suffix: str,
) -> str:
    """Return a skill identity that does not collide with a skill in a different
    bucket.

    Skill identity is the directory basename and every bucket shares
    ``data/state/skills/<name>/``; a cross-bucket basename clash (e.g. clawhub
    ``weather`` vs native ``weather``) makes BOTH skills fail to load. We keep
    our own bucket directory on reinstall/overwrite (in-bucket names are never
    treated as a collision) and only rename to dodge a foreign-bucket skill,
    appending ``-<suffix>`` (then a numeric tail) until the identity is free.
    """
    from ouroboros.skill_loader import _sanitize_skill_name, _skill_location_inventory

    try:
        bucket_root = target_root.resolve()
    except OSError:
        bucket_root = target_root
    foreign_names: set[str] = set()
    for skill in _skill_location_inventory(drive_root):
        try:
            in_bucket = skill.skill_dir.resolve().parent == bucket_root
        except OSError:
            in_bucket = False
        if not in_bucket:
            foreign_names.add(skill.name)
    if base_name not in foreign_names:
        return base_name
    candidates = [f"{base_name}-{suffix}"] + [f"{base_name}-{suffix}-{i}" for i in range(2, 50)]
    for candidate in candidates:
        sanitized = _sanitize_skill_name(candidate)
        if sanitized != base_name and sanitized not in foreign_names:
            return sanitized
    return base_name


def _rewrite_staged_identity(staging_dir: pathlib.Path, new_name: str) -> Optional[str]:
    """After a collision/override rename, make the staged payload self-consistent:
    set the ``SKILL.md`` frontmatter ``name`` and the ``.clawhub.json`` sidecar
    ``sanitized_name`` to the final landed name so the directory, sidecar, and
    manifest all agree. Runtime identity is still the directory basename; this
    only keeps the informational sidecar/manifest from describing a stale name.

    Returns the recomputed ``translated_manifest_sha256`` of the rewritten
    ``SKILL.md`` (or ``None`` if it was not rewritten) so the caller can keep the
    durable provenance digest in sync with the landed file."""
    new_hash: Optional[str] = None
    skill_md = staging_dir / "SKILL.md"
    try:
        if skill_md.is_file():
            text = skill_md.read_text(encoding="utf-8")
            m = re.match(r"^(---\n)(.*?\n)(---\n)(.*)$", text, re.DOTALL)
            if m:
                front = re.sub(r"(?m)^name:.*$", f"name: {new_name}", m.group(2), count=1)
                new_text = m.group(1) + front + m.group(3) + m.group(4)
                skill_md.write_text(new_text, encoding="utf-8")
                import hashlib
                new_hash = hashlib.sha256(new_text.encode("utf-8", errors="replace")).hexdigest()
    except Exception:
        log.debug("Failed to rewrite staged SKILL.md name for %s", new_name, exc_info=True)
    side = staging_dir / ".clawhub.json"
    try:
        if side.is_file():
            data = json.loads(side.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data["sanitized_name"] = new_name
                if new_hash:
                    # Keep the audit digest matching the rewritten SKILL.md.
                    data["translated_manifest_sha256"] = new_hash
                side.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except Exception:
        log.debug("Failed to rewrite staged .clawhub.json for %s", new_name, exc_info=True)
    return new_hash


def install_skill(
    drive_root: pathlib.Path,
    repo_dir: pathlib.Path,
    *,
    slug: str,
    version: Optional[str] = None,
    auto_review: bool = True,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    target_name_override: str = "",
) -> InstallResult:
    """Install one skill; ``overwrite=True`` is required for replacement.

    ``target_name_override`` pins the landed directory name (used by updates so a
    re-install stays in its existing directory instead of re-deriving via dedupe).

    ``progress_callback`` receives worker-thread stage labels for the UI and
    must stay cheap/non-throwing.
    """

    def _progress(stage: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(stage)
            except Exception:
                log.debug("install_skill progress callback raised", exc_info=True)
    fail = lambda error, sanitized_name="", **kwargs: InstallResult(ok=False, sanitized_name=sanitized_name, error=error, **kwargs)

    _progress("Resolving registry…")

    cleaned_slug = (slug or "").strip()
    if not cleaned_slug:
        return fail("slug must be non-empty")

    requested_version = (version or "").strip()

    try:
        summary = _registry_info(cleaned_slug)
    except ClawHubClientError as exc:
        return fail(f"Registry lookup failed: {exc}", error_status=_registry_error_status(exc))

    if summary.is_plugin:
        return fail(
            "Package is an OpenClaw Node/TypeScript plugin and cannot be installed via the Ouroboros marketplace. Skills only.",
            summary=summary,
        )

    target_version = requested_version or summary.latest_version
    if not target_version:
        return fail("Registry returned no version metadata; cannot resolve install target.", summary=summary)

    _progress(f"Downloading v{target_version}…")
    try:
        archive = _registry_download(cleaned_slug, version=target_version)
    except ClawHubClientError as exc:
        return fail(f"Download failed: {exc}", summary=summary, error_status=_registry_error_status(exc))

    try:
        # archive.sha256 is a local recomputation check, not a MITM anchor.
        # Stage under the target bucket so the final move is same-FS atomic.
        staging_root = _clawhub_skills_root(drive_root) / ".staging"
        staged = _stage_archive(
            archive.content,
            slug=cleaned_slug,
            version=target_version,
            expected_sha256=archive.sha256,
            staging_root=staging_root,
        )
    except FetchError as exc:
        return fail(f"Archive validation failed: {exc}", summary=summary, archive=archive)

    _progress("Adapting manifest…")
    try:
        adapter_result = adapt_openclaw_skill(
            staged.staging_dir,
            slug=cleaned_slug,
            version=target_version,
            sha256=archive.sha256,
            is_plugin=staged.has_plugin_manifest,
        )
    except Exception as exc:
        staged.cleanup()
        log.exception("adapter raised during install")
        return fail(f"Adapter raised: {type(exc).__name__}: {exc}", summary=summary, archive=archive, staged=staged)

    if not adapter_result.ok:
        staged.cleanup()
        return fail(
            "Adapter rejected the package: " + "; ".join(adapter_result.blockers),
            sanitized_name=adapter_result.sanitized_name,
            summary=summary,
            archive=archive,
            staged=staged,
            adapter=adapter_result,
        )

    _progress("Landing into data plane…")
    target_root = _clawhub_skills_root(drive_root)
    from ouroboros.skill_loader import _sanitize_skill_name
    # Decide the FINAL landed identity before landing. An update pins the existing
    # directory via target_name_override so a re-install never drifts to a new dir
    # when an earlier cross-bucket collision is gone. A fresh install dodges a
    # collision with another bucket (identity is the directory basename; a clash
    # would break both skills).
    original_name = adapter_result.sanitized_name
    if target_name_override:
        final_name = _sanitize_skill_name(target_name_override) or original_name
    else:
        final_name = dedupe_marketplace_skill_name(
            drive_root, target_root, original_name, suffix="clawhub"
        )
    if final_name != original_name:
        adapter_result.sanitized_name = final_name
        adapter_result.target_dirname = final_name
        # Keep the landed payload self-consistent with the renamed directory.
        new_manifest_hash = _rewrite_staged_identity(staged.staging_dir, final_name)
        if isinstance(adapter_result.provenance, dict):
            adapter_result.provenance["sanitized_name"] = final_name
            if new_manifest_hash:
                adapter_result.provenance["translated_manifest_sha256"] = new_manifest_hash
    target_dir = target_root / adapter_result.target_dirname
    from ouroboros.skill_loader import _skill_location_conflict_error

    identity_error = _skill_location_conflict_error(
        drive_root,
        name=adapter_result.sanitized_name,
        location="clawhub",
        target_dir=target_dir,
    )
    if identity_error:
        staged.cleanup()
        return fail(
            identity_error,
            sanitized_name=adapter_result.sanitized_name,
            summary=summary,
            archive=archive,
            staged=staged,
            adapter=adapter_result,
        )
    auto_specs = list((adapter_result.provenance.get("install_specs") or {}).get("auto") or [])
    repair_partial_existing = False
    if target_dir.exists() and not overwrite and auto_specs:
        deps_state = read_deps_state(drive_root, adapter_result.sanitized_name, target_dir)
        existing_provenance = _valid_existing_clawhub_provenance(
            drive_root,
            adapter_result.sanitized_name,
            target_dir,
            slug=cleaned_slug,
        )
        if (
            str(deps_state.get("status") or "") == "installed"
            and str(deps_state.get("specs_hash") or "") == install_specs_hash(auto_specs)
            and existing_provenance is not None
        ):
            _progress("Already installed with current dependencies…")
            _write_deps_state(drive_root, adapter_result.sanitized_name, deps_state)
            staged.cleanup()
            return InstallResult(
                ok=True,
                sanitized_name=adapter_result.sanitized_name,
                target_dir=target_dir,
                summary=summary,
                archive=archive,
                staged=staged,
                adapter=adapter_result,
                review_status="",
                deps_status="installed",
                deps_fingerprint=deps_state,
                provenance=existing_provenance,
            )
        repair_partial_existing = _has_repairable_clawhub_partial(
            drive_root,
            adapter_result.sanitized_name,
            target_dir,
        )
    rollback_snapshot = snapshot_payload_state(
        drive_root,
        adapter_result.sanitized_name,
        target_dir,
        include_state_provenance=True,
    )
    try:
        _land_staged_into_data_plane(staged, target_dir, overwrite=overwrite or repair_partial_existing)
    except Exception as exc:
        restore_payload_state(rollback_snapshot, restore_state_provenance=True)
        staged.cleanup()
        log.exception("Failed to land staged skill into data plane")
        return fail(
            f"Could not land skill into data plane: {exc}",
            sanitized_name=adapter_result.sanitized_name,
            summary=summary,
            archive=archive,
            staged=staged,
            adapter=adapter_result,
        )
    # Do not repoint staged.staging_dir to target_dir: cleanup() rmtrees it.
    # Persist provenance before review so reviewers can cross-check origin.
    from ouroboros.config import get_clawhub_registry_url
    provenance = dict(adapter_result.provenance)
    provenance.update({
        "registry_url": get_clawhub_registry_url(),
        "version": target_version,
        "homepage": summary.homepage,
        "license": summary.license,
        "primary_env": summary.primary_env,
    })
    try:
        write_provenance(drive_root, adapter_result.sanitized_name, provenance)
    except Exception:
        log.warning("Failed to persist provenance for %s", adapter_result.sanitized_name, exc_info=True)

    # Seed grants.json for core settings so the owner-grant bridge has one file.
    try:
        from ouroboros.skill_loader import (
            load_skill,
            requested_core_setting_keys,
            save_skill_grants,
        )
        # Load the payload that just landed. Global discovery would also load
        # unrelated same-name checkouts and create their lifecycle state.
        installed_skill = load_skill(target_dir, drive_root)
        if installed_skill is not None:
            requested = requested_core_setting_keys(
                list(installed_skill.manifest.env_from_settings or [])
            )
            if requested:
                save_skill_grants(
                    drive_root,
                    installed_skill.name,
                    granted_keys=[],
                    content_hash=installed_skill.content_hash,
                    requested_keys=requested,
                )
    except Exception:
        log.debug("requires.config -> grants.json bootstrap failed", exc_info=True)

    review_status = "pending"
    review_findings: List[Dict[str, Any]] = []
    review_error = ""
    deps_status = "not_required"
    deps_error = ""
    deps_fingerprint: Dict[str, Any] = {}
    if auto_review:
        _progress("Running security review…")
        review_status, review_findings, review_error = _run_skill_review(
            drive_root, repo_dir, adapter_result.sanitized_name
        )
    if auto_specs:
        deps_status = "pending_review"
        if skill_review_gate(review_status)["executable_review"] and not review_error:
            _progress("Installing dependencies…")
            try:
                deps_fingerprint = install_isolated_dependencies(
                    drive_root,
                    adapter_result.sanitized_name,
                    target_dir,
                    auto_specs,
                )
                deps_status = "installed"
                provenance["dependency_fingerprint"] = deps_fingerprint
                write_provenance(drive_root, adapter_result.sanitized_name, provenance)
            except Exception as exc:
                log.exception("isolated dependency install failed for %s", adapter_result.sanitized_name)
                deps_status = "failed"
                deps_error = f"{type(exc).__name__}: {exc}"
                restore_payload_state(rollback_snapshot, restore_state_provenance=True)
        if deps_status != "failed":
            discard_payload_snapshot(rollback_snapshot)
    else:
        discard_payload_snapshot(rollback_snapshot)
    _progress("Done")

    return InstallResult(
        ok=deps_status != "failed",
        sanitized_name=adapter_result.sanitized_name,
        target_dir=target_dir,
        summary=summary,
        archive=archive,
        staged=staged,
        adapter=adapter_result,
        review_status=review_status,
        review_findings=review_findings,
        review_error=review_error,
        deps_status=deps_status,
        deps_error=deps_error,
        deps_fingerprint=deps_fingerprint,
        error=deps_error if deps_status == "failed" else "",
        provenance=provenance,
    )


def uninstall_skill(
    drive_root: pathlib.Path,
    *,
    sanitized_name: str,
) -> UninstallResult:
    """Remove a ClawHub skill payload/provenance while keeping durable state.

    Path traversal is blocked by sanitize round-trip, root containment, and a
    required ``.clawhub.json`` sidecar proving marketplace ownership.
    """
    from ouroboros.skill_loader import _sanitize_skill_name

    cleaned = (sanitized_name or "").strip()
    if (
        not cleaned
        or cleaned in {".", ".."}
        or "/" in cleaned
        or "\\" in cleaned
        or "\x00" in cleaned
        or _sanitize_skill_name(cleaned) != cleaned
    ):
        return UninstallResult(
            False,
            sanitized_name,
            "invalid sanitized_name — must round-trip through _sanitize_skill_name and contain no path separators",
        )

    root = _clawhub_skills_root(drive_root).resolve()
    target = (root / cleaned).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return UninstallResult(False, sanitized_name, f"target escapes clawhub root: {target}")
    if target == root:
        return UninstallResult(False, sanitized_name, "refusing to delete the clawhub bucket root")

    if not target.is_dir():
        return UninstallResult(False, sanitized_name, f"Not found: {target}")

    # Do not remove folders the marketplace pipeline did not install.
    if not (target / ".clawhub.json").is_file():
        return UninstallResult(
            False,
            sanitized_name,
            f"refusing to remove {cleaned!r}: no .clawhub.json sidecar (not a marketplace-installed skill)",
        )

    # Unload in-process extensions before deleting their source tree.
    try:
        from ouroboros.extension_loader import unload_extension
        unload_extension(cleaned)
    except Exception:  # pragma: no cover — defensive
        log.debug("extension unload pre-uninstall failed for %s", cleaned, exc_info=True)
    try:
        shutil.rmtree(target)
    except OSError as exc:
        return UninstallResult(False, sanitized_name, f"Failed to remove {target}: {exc}")
    try:
        from ouroboros.skill_loader import skill_state_dir
        (skill_state_dir(drive_root, cleaned) / DEPS_STATE_FILENAME).unlink(missing_ok=True)
    except Exception:
        log.debug("failed to clear deps state for %s", cleaned, exc_info=True)
    delete_provenance(drive_root, cleaned)
    # CPL4-C11: mark payload-gone so the startup sweep clears the rest of the
    # owner state (grants preserved as owner authority).
    from ouroboros.skill_uninstall_state import write_uninstall_tombstone

    write_uninstall_tombstone(drive_root, cleaned, source="clawhub")
    return UninstallResult(True, cleaned)


def update_skill(
    drive_root: pathlib.Path,
    repo_dir: pathlib.Path,
    *,
    sanitized_name: str,
    version: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> InstallResult:
    """Reinstall by resolving the original slug from persisted provenance."""
    from ouroboros.skill_loader import _skill_location_conflict_error

    expected_target = _clawhub_skills_root(drive_root) / sanitized_name
    identity_error = _skill_location_conflict_error(
        drive_root,
        name=sanitized_name,
        location="clawhub",
        target_dir=expected_target,
    )
    if identity_error:
        return InstallResult(
            False,
            sanitized_name,
            error=identity_error,
        )
    record = read_provenance(drive_root, sanitized_name)

    def _progress(stage: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(stage)
            except Exception:
                log.debug("update_skill progress callback raised", exc_info=True)

    if not record:
        return InstallResult(
            False,
            sanitized_name,
            error=f"No clawhub.json provenance for {sanitized_name!r} — this skill was not installed via the marketplace.",
        )
    slug = str(record.get("slug") or "").strip()
    if not slug:
        return InstallResult(False, sanitized_name, error="provenance is missing slug")
    # Preserve live extension state across unload/swap when possible.
    was_live = False
    try:
        from ouroboros.extension_loader import is_extension_live, unload_extension
        was_live = bool(is_extension_live(sanitized_name, drive_root))
        _progress("Unloading existing extension…")
        unload_extension(sanitized_name)
    except Exception:  # pragma: no cover — defensive
        log.debug("pre-update unload failed for %s", sanitized_name, exc_info=True)
    result = install_skill(
        drive_root,
        repo_dir,
        slug=slug,
        version=version,
        auto_review=True,
        overwrite=True,
        progress_callback=progress_callback,
        target_name_override=sanitized_name,
    )
    if was_live and (
        not getattr(result, "ok", False)
        or skill_review_gate(getattr(result, "review_status", ""))["executable_review"]
    ):
        try:
            from ouroboros.extension_loader import reconcile_extension
            from ouroboros.config import load_settings
            _progress("Reloading extension…")
            reconcile_extension(sanitized_name, drive_root, load_settings)
        except Exception:  # pragma: no cover — defensive
            log.debug("post-update reconcile failed for %s", sanitized_name, exc_info=True)
    return result


__all__ = [
    "InstallResult",
    "PayloadRollbackSnapshot",
    "UninstallResult",
    "discard_payload_snapshot",
    "install_skill",
    "restore_payload_state",
    "snapshot_payload_state",
    "uninstall_skill",
    "update_skill",
]
