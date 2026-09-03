"""Static GitHub catalog client for official OuroborosHub skills."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import pathlib
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from ouroboros.marketplace import AllowlistRedirectHandler
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.config import get_ouroboroshub_catalog_url, get_ouroboroshub_skills_dir
from ouroboros.marketplace.fetcher import FetchError, land_staged_tree
from ouroboros.marketplace.install import (
    discard_payload_snapshot,
    restore_payload_state,
    snapshot_payload_state,
)
from ouroboros.marketplace.install_specs import install_specs_hash
from ouroboros.marketplace.isolated_deps import DEPS_STATE_FILENAME, read_deps_state
from ouroboros.skill_dependencies import normalize_declared_dependency_specs
from ouroboros.skill_loader import (
    _sanitize_skill_name,
    _skill_location_conflict_error,
    skill_state_dir,
)
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)


_MAX_CATALOG_BYTES = 2 * 1024 * 1024
_MAX_FILE_BYTES = 5 * 1024 * 1024
_ALLOWED_HOSTS = frozenset({"raw.githubusercontent.com", "github.com", "localhost", "127.0.0.1"})


class OuroborosHubError(RuntimeError):
    pass


def _raise_if(condition: bool, message: str) -> None:
    if condition:
        raise OuroborosHubError(message)


# Lazy, proxy-free opener: an import-time build_opener snapshots the process
# proxy environment (and triggers macOS proxy lookup in forked workers); the
# clawhub module's lazy no-proxy pattern is the SSOT behavior to match.
_OPENER: urllib.request.OpenerDirector | None = None


def _hub_opener() -> urllib.request.OpenerDirector:
    global _OPENER
    if _OPENER is None:
        _OPENER = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            AllowlistRedirectHandler(
                _ALLOWED_HOSTS,
                lambda target: urllib.error.URLError(
                    f"OuroborosHub redirect host {target!r} is not allowed"
                ),
            ),
        )
    return _OPENER


@dataclass
class HubSkillSummary:
    slug: str
    name: str = ""
    description: str = ""
    version: str = ""
    homepage: str = ""
    files: List[Dict[str, Any]] = field(default_factory=list)
    install_specs: Any = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)
    # Server-computed canonical identity facts (§7.2): the Python sanitizer is
    # the only slug→name authority; JS never re-implements it. identity_conflict
    # marks entries whose canonical name is shared by >1 catalog slugs.
    sanitized_name: str = ""
    identity_conflict: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slug": self.slug,
            "display_name": self.name or self.slug,
            "summary": self.description,
            "description": self.description,
            "latest_version": self.version,
            "versions": [self.version] if self.version else [],
            "homepage": self.homepage,
            "install_specs": self.install_specs,
            "source": "ouroboroshub",
            "stats": {},
            "badges": {"official": True},
            "is_plugin": False,
            "sanitized_name": self.sanitized_name or _sanitize_skill_name(self.slug),
            "identity_conflict": bool(self.identity_conflict),
        }


@dataclass
class HubInstallResult:
    ok: bool
    sanitized_name: str
    error: str = ""
    target_dir: Optional[pathlib.Path] = None
    summary: Optional[HubSkillSummary] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    # Typed error code (e.g. "catalog_identity_conflict"); empty for plain
    # errors so existing payload shapes stay byte-identical.
    code: str = ""


def _fetch_bytes(url: str, *, max_bytes: int, timeout_sec: int = 15) -> bytes:
    parsed = urllib.parse.urlparse(url)
    _raise_if(parsed.scheme not in {"https", "http"}, f"URL must use https:// (or localhost http): {url}")
    _raise_if(parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1"}, f"URL must use https:// for non-localhost hosts: {url}")
    _raise_if(parsed.hostname not in _ALLOWED_HOSTS, f"Host {parsed.hostname!r} is not allowed for OuroborosHub")
    with _hub_opener().open(url, timeout=timeout_sec) as resp:  # noqa: S310 - host allowlist above
        data = resp.read(max_bytes + 1)
    _raise_if(len(data) > max_bytes, f"Response exceeded {max_bytes} bytes: {url}")
    return data


def _raw_base(catalog: Dict[str, Any], catalog_url: str) -> str:
    raw_base = str(catalog.get("raw_base_url") or "").rstrip("/")
    if raw_base:
        return raw_base
    parsed = urllib.parse.urlparse(catalog_url)
    if parsed.hostname == "raw.githubusercontent.com":
        path = parsed.path.strip("/").split("/")
        if len(path) >= 3:
            owner, repo, ref = path[:3]
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}"
    raise OuroborosHubError("catalog must include raw_base_url")


# Display-plane catalog memo (§7.1a): ONLY gateway display reads pass
# ``fresh=False``. install/adopt/update and the official-hub verifier always
# call ``load_catalog()`` with the fresh default and never consume the memo;
# every successful fetch refreshes it so display lags at most the TTL.
_CATALOG_CACHE_TTL_SEC = 120.0
_CATALOG_CACHE_LOCK = threading.Lock()
_CATALOG_CACHE: Optional[tuple[float, Dict[str, Any]]] = None


def _catalog_cache_clear() -> None:
    """Test hook: drop the display-plane catalog memo."""
    global _CATALOG_CACHE
    with _CATALOG_CACHE_LOCK:
        _CATALOG_CACHE = None


def _catalog_cache_inject(catalog: Dict[str, Any], *, age_sec: float = 0.0) -> None:
    """Test hook: seed the display-plane memo with a catalog aged ``age_sec``."""
    global _CATALOG_CACHE
    with _CATALOG_CACHE_LOCK:
        _CATALOG_CACHE = (time.monotonic() - float(age_sec), copy.deepcopy(catalog))


def _catalog_cache_get() -> Optional[Dict[str, Any]]:
    with _CATALOG_CACHE_LOCK:
        if _CATALOG_CACHE is None:
            return None
        fetched_at, catalog = _CATALOG_CACHE
        if (time.monotonic() - fetched_at) > _CATALOG_CACHE_TTL_SEC:
            return None
        # Defensive copy: callers may mutate the returned dict.
        return copy.deepcopy(catalog)


def _catalog_cache_store(catalog: Dict[str, Any]) -> None:
    global _CATALOG_CACHE
    with _CATALOG_CACHE_LOCK:
        _CATALOG_CACHE = (time.monotonic(), copy.deepcopy(catalog))


def load_catalog(fresh: bool = True) -> Dict[str, Any]:
    """Fetch the hub catalog; ``fresh=False`` is reserved for display reads."""
    if not fresh:
        cached = _catalog_cache_get()
        if cached is not None:
            return cached
    url = get_ouroboroshub_catalog_url()
    data = _fetch_bytes(url, max_bytes=_MAX_CATALOG_BYTES)
    try:
        catalog = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OuroborosHubError(f"catalog is not valid JSON: {exc}") from exc
    if not isinstance(catalog, dict):
        raise OuroborosHubError("catalog root must be an object")
    catalog.setdefault("raw_base_url", _raw_base(catalog, url))
    _catalog_cache_store(catalog)
    return catalog


def _summaries(catalog: Dict[str, Any]) -> List[HubSkillSummary]:
    raw_skills = catalog.get("skills") or []
    if not isinstance(raw_skills, list):
        raise OuroborosHubError("catalog.skills must be a list")
    out: List[HubSkillSummary] = []
    for item in raw_skills:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug") or "").strip()
        if not slug:
            continue
        out.append(
            HubSkillSummary(
                slug=slug,
                name=str(item.get("name") or slug),
                description=str(item.get("description") or ""),
                version=str(item.get("version") or ""),
                homepage=str(item.get("homepage") or ""),
                files=list(item.get("files") or []),
                install_specs=item.get("install_specs") or item.get("install") or [],
                raw=item,
            )
        )
    # Canonical-name facts are catalog-wide: a slug whose canonical name is
    # shared with another catalog entry stays flagged even after search filters.
    counts: Dict[str, int] = {}
    for entry in out:
        entry.sanitized_name = _sanitize_skill_name(entry.slug)
        counts[entry.sanitized_name] = counts.get(entry.sanitized_name, 0) + 1
    for entry in out:
        entry.identity_conflict = counts[entry.sanitized_name] > 1
    return out


def search(query: str = "", *, fresh: bool = True) -> List[HubSkillSummary]:
    q = str(query or "").strip().lower()
    entries = _summaries(load_catalog(fresh=fresh))
    if not q:
        return entries
    return [
        item for item in entries
        if q in item.slug.lower() or q in item.name.lower() or q in item.description.lower()
    ]


def info(slug: str) -> HubSkillSummary:
    for item in _summaries(load_catalog()):
        if item.slug == slug:
            return item
    raise OuroborosHubError(f"OuroborosHub skill not found: {slug}")


def _safe_rel(path: str) -> pathlib.PurePosixPath:
    text = str(path or "").strip()
    if "\\" in text or ":" in text:
        raise FetchError(f"unsafe catalog file path: {path!r}")
    rel = pathlib.PurePosixPath(text)
    if not rel.parts or rel.is_absolute() or ".." in rel.parts:
        raise FetchError(f"unsafe catalog file path: {path!r}")
    if any(part in {"node_modules", ".ouroboros_env"} for part in rel.parts):
        raise FetchError(f"catalog file path uses review-opaque dependency directory: {path!r}")
    if "__pycache__" in rel.parts or rel.suffix.lower() in {".pyc", ".pyo", ".so", ".dylib", ".dll"}:
        raise FetchError(f"catalog file path uses generated or binary artifact: {path!r}")
    return rel


def _download_skill_files(summary: HubSkillSummary, raw_base: str, staging_dir: pathlib.Path) -> None:
    files = summary.files
    if not files:
        raise OuroborosHubError(f"catalog entry {summary.slug!r} has no files")
    for item in files:
        if not isinstance(item, dict):
            raise OuroborosHubError(f"catalog file entry for {summary.slug!r} is not an object")
        rel = _safe_rel(str(item.get("path") or ""))
        expected = str(item.get("sha256") or "").strip().lower()
        if not expected:
            raise OuroborosHubError(f"catalog file {rel} is missing sha256")
        url = f"{raw_base.rstrip('/')}/skills/{urllib.parse.quote(summary.slug)}/{urllib.parse.quote(rel.as_posix(), safe='/')}"
        data = _fetch_bytes(url, max_bytes=_MAX_FILE_BYTES)
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise OuroborosHubError(f"sha256 mismatch for {rel}: expected {expected}, got {actual}")
        target = staging_dir / pathlib.Path(*rel.parts)
        try:
            target.resolve(strict=False).relative_to(staging_dir.resolve(strict=False))
        except ValueError as exc:
            raise FetchError(f"catalog file path escapes staging dir: {rel}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    if not (staging_dir / "SKILL.md").is_file():
        raise OuroborosHubError(f"catalog entry {summary.slug!r} did not include SKILL.md")


def _read_hub_marker(target_dir: pathlib.Path) -> Dict[str, Any]:
    marker = pathlib.Path(target_dir) / ".ouroboroshub.json"
    if not marker.is_file():
        return {}
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _valid_existing_hub_marker(target_dir: pathlib.Path, sanitized: str) -> Dict[str, Any]:
    marker = _read_hub_marker(target_dir)
    marker_slug = str(marker.get("slug") or "").strip()
    try:
        schema_version = int(marker.get("schema_version") or 0)
    except (TypeError, ValueError):
        schema_version = 0
    if (
        schema_version == 1
        and str(marker.get("source") or "") == "ouroboroshub"
        and str(marker.get("sanitized_name") or "") == sanitized
        and marker_slug
        and _sanitize_skill_name(marker_slug) == sanitized
    ):
        return marker
    return {}


def _has_repairable_hub_partial(drive_root: pathlib.Path, sanitized: str, target_dir: pathlib.Path) -> bool:
    target = pathlib.Path(target_dir)
    return (
        (skill_state_dir(drive_root, sanitized) / DEPS_STATE_FILENAME).is_file()
        or (target / ".ouroboros_env").exists()
        or (target / ".ouroboroshub.json").is_file()
    )


def install_identity_error(
    sanitized_name: str,
    *,
    drive_root: pathlib.Path | None = None,
) -> str:
    """Return a pure pre-landing identity conflict, or an empty string."""

    sanitized = _sanitize_skill_name(sanitized_name)
    target_root = (
        pathlib.Path(drive_root) / "skills" / "ouroboroshub"
        if drive_root is not None
        else get_ouroboroshub_skills_dir()
    )
    target_dir = target_root / sanitized
    identity_root = pathlib.Path(drive_root) if drive_root is not None else target_root.parent.parent
    return _skill_location_conflict_error(
        identity_root,
        name=sanitized,
        location="ouroboroshub",
        target_dir=target_dir,
    )


def install(slug: str, *, overwrite: bool = False, catalog: Optional[Dict[str, Any]] = None) -> HubInstallResult:
    # ``catalog`` lets the adopt transaction reuse its prelude fetch: no second
    # network read (whose failure would be an untyped 400 instead of the typed
    # 502), and no version drift between the confirm dialog and the download.
    if catalog is None:
        catalog = load_catalog()
    raw_base = str(catalog.get("raw_base_url") or "").rstrip("/")
    summary = next((item for item in _summaries(catalog) if item.slug == slug), None)
    if summary is None:
        return HubInstallResult(False, "", error=f"skill not found: {slug}")
    sanitized = _sanitize_skill_name(summary.slug)
    if summary.identity_conflict:
        # Server-side guard (§7.2/§7.3): >1 catalog slugs sanitize to the same
        # canonical name — installing either would be an ambiguous identity.
        return HubInstallResult(
            False,
            sanitized,
            error=(
                f"catalog identity conflict: multiple catalog entries sanitize to "
                f"{sanitized!r}; refusing to install an ambiguous identity"
            ),
            summary=summary,
            code="catalog_identity_conflict",
        )
    target_root = get_ouroboroshub_skills_dir()
    target_dir = target_root / sanitized
    drive_root = target_root.parent.parent
    identity_error = install_identity_error(sanitized, drive_root=drive_root)
    if identity_error:
        return HubInstallResult(False, sanitized, error=identity_error, summary=summary)
    raw_install = summary.install_specs or summary.raw.get("dependencies") or []
    auto_specs, manual_specs, _warnings = normalize_declared_dependency_specs(raw_install)
    if target_dir.exists() and not overwrite:
        deps_state = read_deps_state(drive_root, sanitized, target_dir)
        marker = _valid_existing_hub_marker(target_dir, sanitized)
        if (
            auto_specs
            and str(deps_state.get("status") or "") == "installed"
            and str(deps_state.get("specs_hash") or "") == install_specs_hash(auto_specs)
            and marker
        ):
            atomic_write_json(
                skill_state_dir(drive_root, sanitized) / DEPS_STATE_FILENAME,
                deps_state,
                trailing_newline=True,
            )
            return HubInstallResult(True, sanitized, target_dir=target_dir, summary=summary, provenance=marker)
        if not _has_repairable_hub_partial(drive_root, sanitized, target_dir):
            return HubInstallResult(False, sanitized, error=f"{sanitized} already installed", summary=summary)
    staging_root = target_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix="ouroboroshub_skill_", dir=str(staging_root)))
    try:
        _download_skill_files(summary, raw_base, staging)
        provenance = {
            "schema_version": 1,
            "source": "ouroboroshub",
            "slug": summary.slug,
            "sanitized_name": sanitized,
            "version": summary.version,
            "catalog_url": get_ouroboroshub_catalog_url(),
            "raw_base_url": raw_base,
            "installed_at": utc_now_iso(),
            "files": summary.files,
        }
        if auto_specs or manual_specs:
            provenance["install_specs"] = {
                "schema_version": 1,
                "auto": auto_specs,
                "manual": manual_specs,
                "raw": raw_install,
                "specs_hash": install_specs_hash(auto_specs),
            }
        atomic_write_json(staging / ".ouroboroshub.json", provenance, trailing_newline=True)
        land_staged_tree(staging, target_dir, replacement_suffix="replaced-ouroboroshub")
        return HubInstallResult(True, sanitized, target_dir=target_dir, summary=summary, provenance=provenance)
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        return HubInstallResult(False, sanitized, error=str(exc), summary=summary)


def uninstall(sanitized_name: str) -> HubInstallResult:
    name = _sanitize_skill_name(sanitized_name)
    if not name or name != sanitized_name:
        return HubInstallResult(False, name, error="invalid skill name")
    target_root = get_ouroboroshub_skills_dir()
    target = target_root / name
    marker = target / ".ouroboroshub.json"
    if not target.exists():
        return HubInstallResult(False, name, error=f"{name} is not installed")
    if not marker.is_file():
        return HubInstallResult(False, name, error="missing OuroborosHub provenance marker")
    # Unload live extension before removing payload so registries do not point at deleted modules.
    try:
        from ouroboros.extension_loader import unload_extension
        unload_extension(name)
    except Exception:  # pragma: no cover — defensive
        pass
    shutil.rmtree(target)
    try:
        (skill_state_dir(target_root.parent.parent, name) / DEPS_STATE_FILENAME).unlink(missing_ok=True)
    except Exception:
        pass
    return HubInstallResult(True, name, target_dir=target)


def serialize_hub_install_result(result: HubInstallResult) -> Dict[str, Any]:
    """Project a :class:`HubInstallResult` into the gateway's JSON payload."""
    payload: Dict[str, Any] = {
        "ok": result.ok,
        "sanitized_name": result.sanitized_name,
        "error": result.error,
        "provenance": result.provenance,
        "summary": result.summary.to_dict() if result.summary else None,
    }
    if result.target_dir is not None:
        payload["target_dir"] = str(result.target_dir)
    if result.code:
        payload["code"] = result.code
    return payload


async def run_hub_update(
    name: str,
    *,
    drive_root: pathlib.Path,
    progress: Any,
    run_blocking: Callable[..., Any],
    apply_review_and_deps: Callable[[Dict[str, Any], str], Any],
) -> Dict[str, Any]:
    """Update-transaction body for the gateway's OuroborosHub update endpoint.

    ``run_blocking`` (the lifecycle thread bridge) and ``apply_review_and_deps``
    (the shared review/deps orchestration) are injected by the gateway so their
    HTTP-layer seams — and the test monkeypatch points on the gateway module —
    stay authoritative.
    """
    from ouroboros.skill_loader import review_status_allows_execution

    drive_root = pathlib.Path(drive_root)
    target_dir = drive_root / "skills" / "ouroboroshub" / name
    marker = target_dir / ".ouroboroshub.json"
    if not target_dir.exists():
        return serialize_hub_install_result(
            HubInstallResult(
                False,
                name,
                error=f"{name} is not installed",
            )
        )
    if not marker.is_file():
        return serialize_hub_install_result(
            HubInstallResult(
                False,
                name,
                error="missing OuroborosHub provenance marker",
                target_dir=target_dir,
            )
        )
    marker_data = read_json_dict(marker) or {}
    marker_name = str(marker_data.get("sanitized_name") or "").strip()
    marker_slug = str(marker_data.get("slug") or "").strip()
    if (
        marker_data.get("schema_version") != 1
        or str(marker_data.get("source") or "") != "ouroboroshub"
        or marker_name != name
        or not marker_slug
        or _sanitize_skill_name(marker_slug) != name
    ):
        return serialize_hub_install_result(
            HubInstallResult(
                False,
                name,
                error="invalid OuroborosHub provenance marker",
                target_dir=target_dir,
            )
        )
    rollback_snapshot = snapshot_payload_state(drive_root, name, target_dir)
    was_live = False

    async def _restore_previous_live(log_label: str) -> None:
        if not was_live:
            return
        try:
            from ouroboros.config import load_settings
            from ouroboros.extension_loader import reconcile_extension

            await run_blocking(
                reconcile_extension,
                name,
                drive_root,
                load_settings,
                log_label=log_label,
            )
        except Exception:
            log.debug("OuroborosHub failed-update re-reconcile failed for %s", name, exc_info=True)

    try:
        try:
            from ouroboros.extension_loader import is_extension_live, unload_extension

            was_live = bool(is_extension_live(name, drive_root))
            progress.set("Unloading existing extension…")
            await run_blocking(
                unload_extension,
                name,
                log_label="OuroborosHub update extension unload lifecycle operation",
            )
        except Exception:
            log.debug("OuroborosHub pre-update extension unload failed for %s", name, exc_info=True)
        progress.set("Downloading from OuroborosHub…")
        result = await run_blocking(
            install,
            marker_slug,
            overwrite=True,
            log_label="OuroborosHub update lifecycle operation",
        )
        payload = serialize_hub_install_result(result)
        if result.ok:
            status, error, deps_status = await apply_review_and_deps(payload, result.sanitized_name)
            if was_live and review_status_allows_execution(status) and not error and deps_status != "failed":
                try:
                    from ouroboros.config import load_settings
                    from ouroboros.extension_loader import reconcile_extension

                    progress.set("Reloading extension…")
                    live_state = await run_blocking(
                        reconcile_extension,
                        result.sanitized_name,
                        drive_root,
                        load_settings,
                        log_label="OuroborosHub update extension reload lifecycle operation",
                    )
                    payload.update({
                        "extension_action": live_state.get("action"),
                        "extension_reason": live_state.get("reason"),
                    })
                except Exception:
                    log.debug("OuroborosHub post-update reconcile failed for %s", name, exc_info=True)
            if deps_status == "failed" or error or not review_status_allows_execution(status):
                restore_payload_state(rollback_snapshot)
                payload["rolled_back"] = True
                await _restore_previous_live("OuroborosHub non-executable update restore lifecycle operation")
            else:
                discard_payload_snapshot(rollback_snapshot)
        elif was_live:
            restore_payload_state(rollback_snapshot)
            payload["rolled_back"] = True
            await _restore_previous_live("OuroborosHub failed-update extension restore lifecycle operation")
        else:
            discard_payload_snapshot(rollback_snapshot)
        return payload
    except Exception as exc:
        restore_payload_state(rollback_snapshot)
        await _restore_previous_live("OuroborosHub exception-update extension restore lifecycle operation")
        log.warning("OuroborosHub update failed after snapshot for %s", name, exc_info=True)
        payload = serialize_hub_install_result(
            HubInstallResult(
                False,
                name,
                error=f"Update failed: {type(exc).__name__}: {exc}",
                target_dir=target_dir,
            )
        )
        payload["rolled_back"] = True
        return payload


# --- Adopt transaction (§7.3): replace an external occupant with the hub payload ---

# The state files the adopt transaction may (re)write through install + review
# + deps + auto-grant, snapshotted before the transaction and byte-restored on
# rollback. review_history.jsonl / other append-only ledgers are intentionally
# NOT restored (disclosed residual: abortive-review entries remain).
_ADOPT_STATE_SNAPSHOT_FILENAMES = (
    "review.json",
    "review_job.json",
    "deps.json",
    "grants.json",
    "accepted_rebuttals.json",
)

# Physical occupant location -> typed adopt_not_eligible reason (§7.3).
_ADOPT_NOT_ELIGIBLE_REASONS = (
    ("ouroboroshub", "already_hub"),
    ("native", "native_seed"),
    ("user_repo", "user_repo"),
    ("clawhub", "clawhub_unsupported_v1"),
)


@dataclass
class _AdoptContext:
    """Rollback handle for one started adopt transaction."""

    name: str
    drive_root: pathlib.Path
    source_dir: pathlib.Path
    aside_dir: pathlib.Path
    dest_dir: pathlib.Path
    state_snapshot: Dict[str, Optional[bytes]]
    was_live: bool = False
    desired_live: bool = False
    catalog: Optional[Dict[str, Any]] = None


def _adopt_refusal(sanitized: str, error: str, code: str = "", **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"ok": False, "sanitized_name": sanitized, "error": error}
    if code:
        payload["code"] = code
    payload.update(extra)
    return payload


def _reconcile_extension_quiet(name: str, drive_root: pathlib.Path) -> str:
    """Reconcile the live extension; returns '' or the failure description.

    The rollback caller counts a failed reconcile as a restore error — a
    rollback that leaves the previously live extension offline (or a zombie
    hub extension loaded over deleted files) must not report rolled_back:true.
    """
    try:
        from ouroboros.config import load_settings
        from ouroboros.extension_loader import reconcile_extension

        state = reconcile_extension(name, pathlib.Path(drive_root), load_settings)
        if isinstance(state, dict):
            load_error = state.get("load_error") or (
                str(state.get("reason") or "extension reload failed")
                if str(state.get("action") or "") == "extension_load_error" else ""
            )
            if load_error:
                return f"live_reconcile: {load_error}"
        return ""
    except Exception as exc:
        log.warning("adopt reconcile failed for %s", name, exc_info=True)
        return f"live_reconcile: {type(exc).__name__}: {exc}"


def _restore_adopt_state(drive_root: pathlib.Path, name: str, snapshot: Dict[str, Optional[bytes]]) -> List[str]:
    """Byte-restore the state quintet; files absent pre-adopt are removed."""
    errors: List[str] = []
    state_dir = skill_state_dir(pathlib.Path(drive_root), name)
    for filename, blob in snapshot.items():
        path = state_dir / filename
        try:
            if blob is None:
                path.unlink(missing_ok=True)
            else:
                tmp = path.with_name(path.name + ".adopt-restore.tmp")
                tmp.write_bytes(blob)
                tmp.replace(path)
        except OSError as exc:
            log.error("adopt rollback could not restore state file %s for %s", filename, name, exc_info=True)
            errors.append(f"state:{filename}: {type(exc).__name__}: {exc}")
    return errors


def _adopt_rollback(ctx: _AdoptContext) -> List[str]:
    """§7.3 rollback: remove dest, restore source payload + state quintet, reconcile.

    Returns the list of restore failures. An empty list is the only state the
    caller may report as ``rolled_back: true`` — a swallowed filesystem error
    here previously let the API claim a completed rollback over a missing or
    duplicated occupant (final-gate fault injection).
    """
    errors: List[str] = []
    try:
        if ctx.dest_dir.exists():
            shutil.rmtree(ctx.dest_dir)
    except OSError as exc:
        log.error("adopt rollback could not remove the dest payload for %s", ctx.name, exc_info=True)
        errors.append(f"dest_remove: {type(exc).__name__}: {exc}")
    try:
        if ctx.aside_dir.exists() and not ctx.source_dir.exists():
            ctx.source_dir.parent.mkdir(parents=True, exist_ok=True)
            ctx.aside_dir.rename(ctx.source_dir)
        elif ctx.aside_dir.exists() and ctx.source_dir.exists():
            # An out-of-band writer recreated the source while it was moved
            # aside. Never clobber the newer occupant, but a rollback that did
            # NOT byte-restore the pre-adopt payload may not claim success.
            errors.append(
                "source_restore: source path was recreated concurrently; "
                f"pre-adopt payload preserved at {ctx.aside_dir}, not restored"
            )
        elif not ctx.aside_dir.exists() and not ctx.source_dir.exists():
            errors.append("source_restore: aside tree missing and source absent")
    except OSError as exc:
        log.error("adopt rollback could not restore source payload for %s", ctx.name, exc_info=True)
        errors.append(
            f"source_restore: {type(exc).__name__}: {exc} (source preserved at {ctx.aside_dir})"
        )
    try:
        errors.extend(_restore_adopt_state(ctx.drive_root, ctx.name, ctx.state_snapshot))
    except OSError as exc:
        log.error("adopt rollback state restore crashed for %s", ctx.name, exc_info=True)
        errors.append(f"state: {type(exc).__name__}: {exc}")
    if ctx.desired_live or ctx.was_live:
        reconcile_error = _reconcile_extension_quiet(ctx.name, ctx.drive_root)
        if reconcile_error:
            errors.append(reconcile_error)
    return errors


def _adopt_finalize(ctx: _AdoptContext) -> Tuple[bool, str]:
    """Retain the moved-aside source as the depth-1 pre-adopt snapshot (§7.3).

    Returns ``(retained, error)``; the adopt itself already succeeded, so a
    retention failure is DISCLOSED on the success payload
    (``pre_adopt_retained: false`` + ``retention_error``), never silently
    swallowed — the aside tree stays in the dot-prefixed rollback area.
    """
    keep = ctx.aside_dir.parent / f"{ctx.name}.pre-adopt"
    try:
        if keep.exists():
            shutil.rmtree(keep)
        ctx.aside_dir.rename(keep)
        return True, ""
    except OSError as exc:
        log.warning("adopt could not retain the pre-adopt snapshot for %s", ctx.name, exc_info=True)
        note = f"{type(exc).__name__}: {exc} (source preserved at {ctx.aside_dir})"
        return False, note


def _adopt_begin(slug: str, drive_root: pathlib.Path, expected_content_hash: str):
    """§7.3 prelude: eligibility -> CAS -> unload -> state snapshot -> move-aside.

    Returns a typed refusal payload (source payload untouched) or an
    :class:`_AdoptContext` whose source dir is already moved aside into the
    bucket's dot-prefixed (discovery-safe) ``.rollback`` area.
    """
    from ouroboros.skill_loader import _skill_location_inventory, load_skill

    drive_root = pathlib.Path(drive_root)
    sanitized = _sanitize_skill_name(slug)
    try:
        catalog = load_catalog()
    except Exception as exc:
        return _adopt_refusal(sanitized, f"OuroborosHub catalog unavailable: {exc}", "catalog_unavailable")
    summary = next((item for item in _summaries(catalog) if item.slug == slug), None)
    if summary is None:
        return _adopt_refusal(sanitized, f"skill not found: {slug}")
    if summary.identity_conflict:
        return _adopt_refusal(
            sanitized,
            (
                f"catalog identity conflict: multiple catalog entries sanitize to "
                f"{sanitized!r}; refusing to adopt an ambiguous identity"
            ),
            "catalog_identity_conflict",
        )

    occupants = tuple(
        item for item in _skill_location_inventory(drive_root) if item.name == sanitized
    )
    if not occupants:
        return _adopt_refusal(
            sanitized,
            f"no local skill named {sanitized!r} occupies the identity - use install instead",
            "adopt_not_eligible",
            reason="no_local_occupant",
        )
    for location, reason in _ADOPT_NOT_ELIGIBLE_REASONS:
        if any(item.location == location for item in occupants):
            return _adopt_refusal(
                sanitized,
                f"local skill {sanitized!r} lives in the {location!r} bucket and cannot be adopted",
                "adopt_not_eligible",
                reason=reason,
            )

    # Only external occupants remain. CAS selects the exact payload the caller
    # confirmed; with pathological same-name external duplicates the
    # non-matching sibling keeps the identity occupied and the ordinary install
    # precheck fails the transaction into rollback.
    live_hashes: List[str] = []
    selected = None
    for candidate in occupants:
        loaded = load_skill(candidate.skill_dir, drive_root)
        live = str(getattr(loaded, "content_hash", "") or "") if loaded is not None else ""
        live_hashes.append(live)
        if live and live == expected_content_hash:
            selected = candidate
            break
    if selected is None:
        return _adopt_refusal(
            sanitized,
            "local payload does not match expected_content_hash - refresh the skill card and confirm again",
            "adopt_cas_mismatch",
            live_content_hash=live_hashes[0] if live_hashes else "",
        )

    was_live = False
    try:
        from ouroboros.extension_loader import is_extension_live, unload_extension

        was_live = bool(is_extension_live(sanitized, drive_root))
        unload_extension(sanitized)
    except Exception:
        log.debug("pre-adopt extension unload failed for %s", sanitized, exc_info=True)
    # Post-install reconcile must run for every ENABLED occupant, not only a
    # LIVE one: an enabled extension whose previous load failed is exactly the
    # case the strict load_error check exists for (final-gate finding).
    desired_live = was_live
    try:
        from ouroboros.skill_loader import load_enabled

        desired_live = was_live or bool(load_enabled(drive_root, sanitized))
    except Exception:
        log.debug("adopt enabled-state read failed for %s", sanitized, exc_info=True)

    try:
        state_dir = skill_state_dir(drive_root, sanitized)
        state_snapshot: Dict[str, Optional[bytes]] = {}
        for filename in _ADOPT_STATE_SNAPSHOT_FILENAMES:
            path = state_dir / filename
            state_snapshot[filename] = path.read_bytes() if path.is_file() else None
        rollback_root = selected.skill_dir.parent / ".rollback"
        rollback_root.mkdir(parents=True, exist_ok=True)
        aside_dir = rollback_root / f"{sanitized}.adopt.{uuid.uuid4().hex}"
        selected.skill_dir.rename(aside_dir)
        # Re-verify the CAS on the moved-aside tree: the window between the
        # eligibility CAS read and the rename is otherwise unguarded, and a
        # concurrent writer's edits would be silently adopted over. A mismatch
        # restores the source in place and refuses typed.
        from ouroboros.skill_loader import compute_content_hash

        aside_live = ""
        try:
            aside_live = str(compute_content_hash(aside_dir) or "")
        except Exception:
            aside_live = ""
        if aside_live != expected_content_hash:
            try:
                aside_dir.rename(selected.skill_dir)
                restore_note = ""
            except OSError as exc:
                # The rename-back can lose a race with exactly the concurrent
                # writer a CAS mismatch implies; never strand the payload
                # silently — name where it survives.
                log.error("adopt CAS-mismatch restore failed for %s", sanitized, exc_info=True)
                restore_note = f" SOURCE NOT RESTORED ({type(exc).__name__}); preserved at {aside_dir}"
            if was_live:
                reconcile_note = _reconcile_extension_quiet(sanitized, drive_root)
                if reconcile_note:
                    restore_note += f" ({reconcile_note})"
            return _adopt_refusal(
                sanitized,
                "local payload changed between confirmation and replacement - refresh the skill card and confirm again"
                + restore_note,
                "adopt_cas_mismatch",
                live_content_hash=aside_live,
            )
    except Exception as exc:
        log.warning("adopt move-aside failed for %s", sanitized, exc_info=True)
        if was_live:
            _reconcile_extension_quiet(sanitized, drive_root)
        return _adopt_refusal(
            sanitized,
            f"Adopt failed before replacing the local copy: {type(exc).__name__}: {exc}",
        )
    return _AdoptContext(
        name=sanitized,
        drive_root=drive_root,
        source_dir=selected.skill_dir,
        aside_dir=aside_dir,
        dest_dir=get_ouroboroshub_skills_dir() / sanitized,
        state_snapshot=state_snapshot,
        was_live=was_live,
        desired_live=desired_live,
        catalog=catalog,
    )


async def run_hub_adopt(
    slug: str,
    *,
    drive_root: pathlib.Path,
    expected_content_hash: str,
    progress: Any,
    run_blocking: Callable[..., Any],
    apply_review_and_deps: Callable[[Dict[str, Any], str], Any],
) -> Dict[str, Any]:
    """§7.3 adopt transaction: move the external occupant aside, run the
    ordinary install (its built-in identity precheck sees a clean tree), share
    the update path's review/deps orchestration, then reload.

    Any failure after the source payload was moved aside rolls back the dest
    payload plus the state quintet and reports ``rolled_back: true``. Unlike
    update, a reload/load_error failure is TERMINAL for adopt (disclosed
    asymmetry; aligning update is a separate issue).
    """
    from ouroboros.skill_loader import review_status_allows_execution

    drive_root = pathlib.Path(drive_root)
    sanitized = _sanitize_skill_name(slug)
    progress.set("Checking adopt eligibility…")
    try:
        prelude = await run_blocking(
            _adopt_begin,
            slug,
            drive_root,
            expected_content_hash,
            log_label="OuroborosHub adopt eligibility lifecycle operation",
        )
    except Exception as exc:
        log.warning("adopt prelude failed for %s", sanitized, exc_info=True)
        return _adopt_refusal(sanitized, f"Adopt failed: {type(exc).__name__}: {exc}")
    if isinstance(prelude, dict):
        return prelude
    ctx: _AdoptContext = prelude

    async def _rollback() -> List[str]:
        return await run_blocking(
            _adopt_rollback,
            ctx,
            log_label="OuroborosHub adopt rollback lifecycle operation",
        )

    def _stamp_rollback(payload: Dict[str, Any], errors: List[str]) -> None:
        # rolled_back is a VERIFIED claim: true only when every restore step
        # landed. A partial rollback is disclosed with its exact failures.
        payload["rolled_back"] = not errors
        if errors:
            payload["rollback_errors"] = errors
            payload["error"] = (
                str(payload.get("error") or "adopt failed")
                + " ROLLBACK INCOMPLETE: " + "; ".join(errors)
            )

    try:
        progress.set("Downloading from OuroborosHub…")
        result = await run_blocking(
            install,
            slug,
            catalog=ctx.catalog,
            log_label="OuroborosHub adopt install lifecycle operation",
        )
        payload = serialize_hub_install_result(result)
        if not result.ok:
            _stamp_rollback(payload, await _rollback())
            return payload
        status, error, deps_status = await apply_review_and_deps(payload, result.sanitized_name)
        if deps_status == "failed" or error or not review_status_allows_execution(status):
            payload["ok"] = False
            if not payload.get("error"):
                payload["error"] = error or f"review status {status!r} does not allow execution"
            _stamp_rollback(payload, await _rollback())
            return payload
        if ctx.desired_live:
            reload_error = ""
            try:
                from ouroboros.config import load_settings
                from ouroboros.extension_loader import reconcile_extension

                progress.set("Reloading extension…")
                live_state = await run_blocking(
                    reconcile_extension,
                    result.sanitized_name,
                    drive_root,
                    load_settings,
                    log_label="OuroborosHub adopt extension reload lifecycle operation",
                )
                payload.update({
                    "extension_action": live_state.get("action"),
                    "extension_reason": live_state.get("reason"),
                })
                if str(live_state.get("action") or "") == "extension_load_error" or live_state.get("load_error"):
                    reload_error = str(live_state.get("load_error") or "extension reload failed")
            except Exception as exc:
                reload_error = f"{type(exc).__name__}: {exc}"
            if reload_error:
                # Terminal for adopt (§7.3): never leave a hub payload that
                # cannot load where a loadable local copy used to live.
                payload["ok"] = False
                payload["error"] = f"extension reload failed after adopt: {reload_error}"
                _stamp_rollback(payload, await _rollback())
                return payload
        retained, retention_error = await run_blocking(
            _adopt_finalize,
            ctx,
            log_label="OuroborosHub adopt finalize lifecycle operation",
        )
        payload["adopted"] = True
        payload["pre_adopt_retained"] = bool(retained)
        if retention_error:
            payload["retention_error"] = retention_error
        return payload
    except Exception as exc:
        log.warning("OuroborosHub adopt failed after move-aside for %s", sanitized, exc_info=True)
        payload = _adopt_refusal(sanitized, f"Adopt failed: {type(exc).__name__}: {exc}")
        try:
            errors = await _rollback()
        except Exception as rb_exc:
            log.error("OuroborosHub adopt rollback itself failed for %s", sanitized, exc_info=True)
            errors = [f"rollback_crashed: {type(rb_exc).__name__}: {rb_exc}"]
        _stamp_rollback(payload, errors)
        return payload
