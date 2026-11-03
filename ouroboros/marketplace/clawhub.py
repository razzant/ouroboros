"""Read-only ClawHub registry client for search/info/version/download."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

from ouroboros.marketplace import AllowlistRedirectHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


_DEFAULT_TIMEOUT_SEC = 15
_MAX_JSON_RESPONSE_BYTES = 4 * 1024 * 1024  # 4 MB JSON cap
_MAX_ARCHIVE_BYTES = 50 * 1024 * 1024       # mirrors fetcher cap
_USER_AGENT = "Ouroboros-Marketplace/4.50 (+https://github.com/razzant/ouroboros)"
_BROWSE_PATH = "packages"
_LEXICAL_SEARCH_PATH = "search"
_SEARCH_ENRICH_WORKERS = 8
_SEARCH_ENRICH_LIMIT = 16
_SEARCH_ENRICH_TIMEOUT_SEC = 2
_MAX_RATE_LIMIT_RETRIES = 2
_MAX_RATE_LIMIT_SLEEP_SEC = 3.0

# Allow only audited ClawHub hosts plus localhost dev mirrors; reject legacy aliases.
_ALLOWED_REGISTRY_HOSTS = frozenset({"clawhub.ai", "www.clawhub.ai", "registry.clawhub.ai", "localhost", "127.0.0.1"})


class ClawHubClientError(RuntimeError):
    pass


class ClawHubRateLimitError(ClawHubClientError):
    """Raised when ClawHub keeps rate-limiting after bounded retries."""

    def __init__(self, url: str, retry_after: Optional[float] = None) -> None:
        self.url = url
        self.retry_after = retry_after
        wait = _format_retry_after(retry_after)
        suffix = f" Try again in {wait}." if wait else " Try again in a few minutes."
        super().__init__(f"ClawHub rate limit reached.{suffix}")


class ClawHubClientHostBlocked(ClawHubClientError):
    pass


class ClawHubNotFoundError(ClawHubClientError):
    """Raised when the registry returns HTTP 404 for a slug/endpoint."""


@dataclass
class ClawHubSkillSummary:
    """Permissive per-skill record returned by ``search`` / ``info``."""

    slug: str
    display_name: str = ""
    summary: str = ""
    description: str = ""
    latest_version: str = ""
    versions: List[str] = field(default_factory=list)
    license: str = ""
    homepage: str = ""
    os_list: List[str] = field(default_factory=list)
    requires_env: List[str] = field(default_factory=list)
    requires_bins: List[str] = field(default_factory=list)
    primary_env: str = ""
    install_specs: List[Dict[str, Any]] = field(default_factory=list)
    badges: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    is_plugin: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    def catalog_facts(self) -> Dict[str, Any]:
        """Normalize ClawHub's author and millisecond update time at the adapter boundary."""
        publisher = self.raw.get("publisher") if isinstance(self.raw.get("publisher"), dict) else {}
        facts: Dict[str, Any] = {"owner": str(self.raw.get("ownerHandle") or publisher.get("handle") or "").strip()}
        stamp = self.raw.get("updatedAt")
        if isinstance(stamp, (int, float)) and not isinstance(stamp, bool) and stamp > 0:
            try:
                facts["updated_at"] = datetime.fromtimestamp(stamp / 1000, timezone.utc).isoformat()
            except (OverflowError, OSError, ValueError):
                pass
        return facts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slug": self.slug,
            "display_name": self.display_name,
            "summary": self.summary,
            "description": self.description,
            "latest_version": self.latest_version,
            "versions": list(self.versions),
            "license": self.license,
            "homepage": self.homepage,
            "os": list(self.os_list),
            "requires_env": list(self.requires_env),
            "requires_bins": list(self.requires_bins),
            "primary_env": self.primary_env,
            "install_specs": list(self.install_specs),
            "badges": dict(self.badges),
            "stats": dict(self.stats),
            "is_plugin": self.is_plugin,
        }


def _registry_base_url(override: Optional[str] = None) -> str:
    if override is None or not str(override).strip():
        from ouroboros.config import get_clawhub_registry_url
        url = get_clawhub_registry_url()
    else:
        url = str(override).strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("https", "http"):
        raise ClawHubClientHostBlocked(
            f"Registry URL {url!r} must use https:// (or http:// for localhost dev)."
        )
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1"}:
        raise ClawHubClientHostBlocked(
            f"Registry URL {url!r} must use https:// for non-localhost hosts."
        )
    if not parsed.hostname:
        raise ClawHubClientHostBlocked(f"Registry URL {url!r} has no hostname.")
    if parsed.hostname not in _ALLOWED_REGISTRY_HOSTS:
        raise ClawHubClientHostBlocked(
            f"Registry host {parsed.hostname!r} is not in the marketplace allowlist {sorted(_ALLOWED_REGISTRY_HOSTS)}."
        )
    return url.rstrip("/")


def _build_url(base: str, path: str, query: Optional[Dict[str, Any]] = None) -> str:
    rel = path.lstrip("/")
    composed = f"{base.rstrip('/')}/{rel}"
    if query:
        composed = f"{composed}?{urllib.parse.urlencode(query, doseq=True)}"
    return composed


def _redirect_handler() -> AllowlistRedirectHandler:
    return AllowlistRedirectHandler(
        _ALLOWED_REGISTRY_HOSTS,
        lambda target: ClawHubClientHostBlocked(
            f"Refused to follow redirect to {target!r} outside marketplace allowlist"
        ),
    )


def _build_opener(no_proxy: bool) -> urllib.request.OpenerDirector:
    handlers: List[Any] = [_redirect_handler()]
    # In worker processes, disable system proxy resolution for fork-safety
    # (macOS _scproxy/SCDynamicStoreCopyProxies crashes on the child side of a
    # multi-threaded fork). The supervisor keeps proxy support for corporate
    # networks. Marketplace hosts are allowlisted, so no-proxy is also correct.
    if no_proxy:
        handlers.append(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers)


# Openers are built lazily and cached per process role. A worker NEVER builds
# the proxy-discovery opener: build_opener()'s default ProxyHandler() calls
# getproxies() at construction, and we must not run that system-proxy lookup in
# a worker (the macOS _scproxy crash class on the fork escape-hatch, and an
# unnecessary lookup on spawn). The supervisor (main process, never forked)
# builds the proxy-honoring opener on first use.
_OPENER: Optional[urllib.request.OpenerDirector] = None
_WORKER_OPENER: Optional[urllib.request.OpenerDirector] = None


def _active_opener() -> urllib.request.OpenerDirector:
    global _OPENER, _WORKER_OPENER
    worker = False
    try:
        from ouroboros.utils import in_worker_process
        worker = in_worker_process()
    except Exception:
        worker = False
    if worker:
        if _WORKER_OPENER is None:
            _WORKER_OPENER = _build_opener(no_proxy=True)
        return _WORKER_OPENER
    if _OPENER is None:
        _OPENER = _build_opener(no_proxy=False)
    return _OPENER


def _parse_retry_after(value: Any) -> Optional[float]:
    try:
        seconds = float(str(value or "").strip())
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return seconds


def _format_retry_after(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    rounded = max(1, int(seconds))
    if rounded >= 60:
        minutes = max(1, round(rounded / 60))
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    return f"{rounded} second{'s' if rounded != 1 else ''}"


def _sleep_for_rate_limit(retry_after: Optional[float], attempt: int) -> None:
    fallback = min(_MAX_RATE_LIMIT_SLEEP_SEC, 0.5 * (2 ** attempt))
    delay = retry_after if retry_after is not None else fallback
    time.sleep(min(_MAX_RATE_LIMIT_SLEEP_SEC, max(0.1, float(delay))))


def _http_get(
    url: str,
    *,
    timeout: int = _DEFAULT_TIMEOUT_SEC,
    accept: str = "application/json",
    max_bytes: int = _MAX_JSON_RESPONSE_BYTES,
) -> Tuple[bytes, Dict[str, str]]:
    last_retry_after: Optional[float] = None
    for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": accept}, method="GET")
        try:
            with _active_opener().open(request, timeout=timeout) as response:
                status = int(getattr(response, "status", 0) or response.getcode() or 0)
                if status == 429:
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                    last_retry_after = retry_after
                    if attempt < _MAX_RATE_LIMIT_RETRIES:
                        _sleep_for_rate_limit(retry_after, attempt)
                        continue
                    raise ClawHubRateLimitError(url, retry_after)
                if status >= 400:
                    if status == 404:
                        raise ClawHubNotFoundError(f"GET {url} returned HTTP 404")
                    raise ClawHubClientError(f"GET {url} returned HTTP {status}")
                buf = bytearray()
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    if len(buf) + len(chunk) > max_bytes:
                        raise ClawHubClientError(
                            f"GET {url} response exceeds {max_bytes} byte cap (possible registry abuse)."
                        )
                    buf.extend(chunk)
                headers = {k.lower(): v for k, v in response.headers.items()}
                return bytes(buf), headers
        except urllib.error.HTTPError as exc:
            if int(exc.code or 0) == 429:
                retry_after = _parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
                last_retry_after = retry_after
                if attempt < _MAX_RATE_LIMIT_RETRIES:
                    _sleep_for_rate_limit(retry_after, attempt)
                    continue
                raise ClawHubRateLimitError(url, retry_after) from exc
            if int(exc.code or 0) == 404:
                raise ClawHubNotFoundError(f"GET {url}: HTTP 404: {exc.reason}") from exc
            raise ClawHubClientError(f"GET {url}: HTTP {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise ClawHubClientError(f"GET {url}: transport error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ClawHubClientError(f"GET {url}: timed out after {timeout}s") from exc
    raise ClawHubRateLimitError(url, last_retry_after)


def _decode_json(body: bytes, *, url: str) -> Any:
    try:
        text = body.decode("utf-8", errors="replace")
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClawHubClientError(f"GET {url} returned invalid JSON: {exc}") from exc


def _coerce_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            text = _coerce_version(item) if isinstance(item, dict) else item
            cleaned = str(text or "").strip()
            if cleaned:
                result.append(cleaned)
        return result
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _coerce_version(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("version") or value.get("name") or value.get("tag") or value.get("value") or "").strip()
    return str(value or "").strip()


def _detect_plugin(raw: Dict[str, Any]) -> bool:
    kind = str(raw.get("kind") or raw.get("package_kind") or raw.get("family") or "").lower()
    return kind in {"plugin", "code-plugin"} or bool(raw.get("plugin_manifest")) or raw.get("has_plugin") is True


def _extract_metadata_openclaw(raw: Dict[str, Any]) -> Dict[str, Any]:
    parsed = raw.get("parsed")
    for metadata in (raw.get("metadata"), parsed.get("metadata") if isinstance(parsed, dict) else None):
        if not isinstance(metadata, dict):
            continue
        for key in ("openclaw", "clawdis", "clawdbot"):
            block = metadata.get(key)
            if isinstance(block, dict) and block:
                return block
    return {}


def _normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(raw.get("package"), dict):
        normalized = dict(raw["package"])
    elif isinstance(raw.get("skill"), dict):
        normalized = dict(raw["skill"])
    else:
        normalized = dict(raw)
    for key in ("latestVersion", "latest_version", "versions", "availableVersions"):
        if key in raw and key not in normalized:
            normalized[key] = raw[key]
    owner = raw.get("owner")
    if isinstance(owner, dict) and "ownerHandle" not in normalized:
        normalized["ownerHandle"] = owner.get("handle") or owner.get("name")
    return normalized


def _extract_items_and_cursor(parsed: Any, *, path: str) -> Tuple[List[Any], str]:
    if isinstance(parsed, list):
        return parsed, ""
    if not isinstance(parsed, dict):
        raise ClawHubClientError(f"Unexpected response shape from {path}: {type(parsed).__name__}")
    container = parsed.get("data") if isinstance(parsed.get("data"), dict) else parsed
    if not isinstance(container, dict):
        raise ClawHubClientError(f"Unexpected data shape from {path}: {type(container).__name__}")
    items = container.get("results") or container.get("items") or container.get("skills") or container.get("packages") or []
    if not isinstance(items, list):
        raise ClawHubClientError(f"Unexpected items shape from {path}: {type(items).__name__}")
    next_cursor = str(container.get("nextCursor") or container.get("next_cursor") or container.get("cursor") or "")
    return items, next_cursor


def _summary_from_record(raw: Any) -> ClawHubSkillSummary:
    if not isinstance(raw, dict):
        raise ClawHubClientError(f"Registry record must be an object, got {type(raw).__name__}")
    raw = _normalize_record(raw)
    slug = str(raw.get("slug") or raw.get("name") or "").strip()
    if not slug:
        raise ClawHubClientError("Registry record missing required 'slug'/'name'")

    display_name = str(raw.get("displayName") or raw.get("display_name") or slug).strip()
    summary_text = str(raw.get("summary") or raw.get("description") or "").strip()
    description = str(raw.get("description") or summary_text).strip()

    tags = raw.get("tags") or {}
    if not isinstance(tags, dict):
        tags = {}
    latest_version = _coerce_version(
        raw.get("latestVersion")
        or raw.get("latest_version")
        or raw.get("version")
        or tags.get("latest")
    )
    versions = _coerce_str_list(raw.get("versions") or raw.get("availableVersions"))
    if latest_version and latest_version not in versions:
        versions.insert(0, latest_version)

    license_text = str(raw.get("license") or "").strip()
    homepage = str(raw.get("homepage") or raw.get("website") or raw.get("url") or "").strip()

    metadata_block = _extract_metadata_openclaw(raw)
    requires = metadata_block.get("requires") or {}
    if not isinstance(requires, dict):
        requires = {}
    compatibility = raw.get("compatibility") or {}
    if not isinstance(compatibility, dict):
        compatibility = {}
    os_list = _coerce_str_list(
        metadata_block.get("os")
        or raw.get("os")
        or compatibility.get("os")
        or compatibility.get("platforms")
    )
    requires_env = _coerce_str_list(requires.get("env"))
    requires_bins = _coerce_str_list(requires.get("bins") or requires.get("anyBins"))
    primary_env = str(metadata_block.get("primaryEnv") or "").strip()
    install_specs_raw = metadata_block.get("install") or []
    install_specs: List[Dict[str, Any]] = []
    if isinstance(install_specs_raw, list):
        for spec in install_specs_raw:
            if isinstance(spec, dict):
                install_specs.append(dict(spec))

    badges = raw.get("badges") or {}
    if not isinstance(badges, dict):
        badges = {}
    if raw.get("isOfficial") is True or str(raw.get("channel") or "").lower() == "official":
        badges = {**badges, "official": True}
    stats = raw.get("stats") or {}
    if not isinstance(stats, dict):
        stats = {}

    return ClawHubSkillSummary(
        slug=slug,
        display_name=display_name,
        summary=summary_text,
        description=description,
        latest_version=latest_version,
        versions=versions,
        license=license_text,
        homepage=homepage,
        os_list=os_list,
        requires_env=requires_env,
        requires_bins=requires_bins,
        primary_env=primary_env,
        install_specs=install_specs,
        badges=badges,
        stats=stats,
        is_plugin=_detect_plugin(raw),
        raw=raw,
    )


def _merge_enriched_summary(
    bare: ClawHubSkillSummary,
    rich: ClawHubSkillSummary,
) -> ClawHubSkillSummary:
    rich_display = rich.display_name
    if rich_display == rich.slug and bare.display_name:
        rich_display = ""
    raw = dict(bare.raw)
    raw.update(rich.raw)
    if "score" in bare.raw and "search_score" not in raw:
        raw["search_score"] = bare.raw.get("score")
    values = {
        item.name: getattr(rich, item.name) or getattr(bare, item.name)
        for item in fields(ClawHubSkillSummary)
        if item.name not in {"slug", "display_name", "badges", "is_plugin", "raw"}
    }
    return ClawHubSkillSummary(
        slug=bare.slug,
        display_name=rich_display or bare.display_name,
        badges={**bare.badges, **rich.badges},
        is_plugin=rich.is_plugin or bare.is_plugin,
        raw=raw,
        **values,
    )


def _enrich_search_summaries(
    summaries: List[ClawHubSkillSummary],
    *,
    registry_url: Optional[str],
    timeout_sec: int,
) -> Tuple[List[ClawHubSkillSummary], List[str]]:
    if not summaries:
        return summaries, []
    enriched = list(summaries)
    warnings: List[str] = []
    enrich_count = min(_SEARCH_ENRICH_LIMIT, len(summaries))
    workers = max(1, min(_SEARCH_ENRICH_WORKERS, enrich_count))
    detail_timeout = max(
        1,
        min(int(timeout_sec or _DEFAULT_TIMEOUT_SEC), _SEARCH_ENRICH_TIMEOUT_SEC),
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_index = {
            pool.submit(
                _detail_summary,
                summary.slug,
                registry_url=registry_url,
                timeout_sec=detail_timeout,
                merge_skill_detail=True,
            ): idx
            for idx, summary in enumerate(summaries[:enrich_count])
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                enriched[idx] = _merge_enriched_summary(
                    summaries[idx],
                    future.result(),
                )
            except ClawHubRateLimitError as exc:
                warnings.append(str(exc))
                log.warning(
                    "Keeping bare ClawHub search result for %s after rate-limit during enrich.",
                    summaries[idx].slug,
                    exc_info=True,
                )
            except Exception:
                log.warning(
                    "Keeping bare ClawHub search result for %s after enrich failure.",
                    summaries[idx].slug,
                    exc_info=True,
                )
    return enriched, warnings


def search(
    query: str = "",
    *,
    limit: int = 25,
    cursor: Optional[str] = None,
    sort: str = "registry",
    official_only: bool = False,
    registry_url: Optional[str] = None,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
    include_metadata: bool = False,
) -> Any:
    """Browse or lexical-search the modern ClawHub package catalogue."""
    base = _registry_base_url(registry_url)
    sort_key = (sort or "downloads").strip().lower()
    if sort_key not in {"registry", "updated"}:
        sort_key = "registry"
    cleaned_query = (query or "").strip()
    max_limit = _SEARCH_ENRICH_LIMIT if cleaned_query else 100
    safe_limit = max(1, min(int(limit or 25), max_limit))
    path = _LEXICAL_SEARCH_PATH if cleaned_query else _BROWSE_PATH
    query_params: Dict[str, Any] = {"q": cleaned_query, "limit": safe_limit} if cleaned_query else {
        "family": "skill",
        "limit": safe_limit,
    }
    if not cleaned_query:
        if official_only:
            query_params["isOfficial"] = "true"
        cleaned_cursor = str(cursor or "").strip()
        if cleaned_cursor:
            query_params["cursor"] = cleaned_cursor

    url = _build_url(base, path, query_params)
    body, _headers = _http_get(url, timeout=timeout_sec)
    parsed = _decode_json(body, url=url)
    items, next_cursor = _extract_items_and_cursor(parsed, path=path)
    if cleaned_query:
        items = items[:safe_limit]
    summaries: List[ClawHubSkillSummary] = []
    for record in items:
        try:
            summaries.append(_summary_from_record(record))
        except ClawHubClientError:
            log.warning("Skipping malformed registry record: %r", record, exc_info=True)
            continue
    enrich_warnings: List[str] = []
    if cleaned_query:
        enriched_result = _enrich_search_summaries(
            summaries,
            registry_url=registry_url,
            timeout_sec=timeout_sec,
        )
        if isinstance(enriched_result, tuple):
            summaries, enrich_warnings = enriched_result
        else:  # Backward-compatible for tests/patches that stub the old shape.
            summaries = enriched_result
    if include_metadata:
        return {
            "results": summaries,
            "next_cursor": next_cursor,
            "path": path,
            "attempts": [{"path": path, "count": len(summaries), "ok": True}],
            "warnings": enrich_warnings,
            "sort": sort_key,
            "filters": {
                "family": "skill" if not cleaned_query else "",
                "official_only": bool(official_only) if not cleaned_query else False,
            },
        }
    return summaries


def _validate_slug(slug: str) -> str:
    cleaned = (slug or "").strip()
    if not cleaned:
        raise ClawHubClientError("'slug' must be non-empty")
    if cleaned.startswith("/") or cleaned.startswith("\\"):
        raise ClawHubClientError("'slug' must not be absolute")
    parts = cleaned.replace("\\", "/").split("/")
    if any(part == ".." or part == "." for part in parts):
        raise ClawHubClientError("'slug' must not contain '..' or '.' segments")
    return cleaned


def _fetch_summary_path(base: str, path: str, *, timeout_sec: int) -> ClawHubSkillSummary:
    url = _build_url(base, path)
    body, _headers = _http_get(url, timeout=timeout_sec)
    parsed = _decode_json(body, url=url)
    return _summary_from_record(parsed)


def _detail_summary(
    slug: str,
    *,
    registry_url: Optional[str] = None,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
    merge_skill_detail: bool = False,
) -> ClawHubSkillSummary:
    cleaned = _validate_slug(slug)
    base = _registry_base_url(registry_url)
    quoted = urllib.parse.quote(cleaned, safe='/-')
    package_error: Optional[ClawHubClientError] = None
    package_summary: Optional[ClawHubSkillSummary] = None
    try:
        package_summary = _fetch_summary_path(
            base,
            f"packages/{quoted}",
            timeout_sec=timeout_sec,
        )
    except ClawHubClientError as exc:
        package_error = exc
    if not merge_skill_detail:
        if package_summary is not None:
            return package_summary
        return _fetch_summary_path(base, f"skills/{quoted}", timeout_sec=timeout_sec)
    try:
        skill_summary = _fetch_summary_path(base, f"skills/{quoted}", timeout_sec=timeout_sec)
    except ClawHubClientError:
        if package_summary is not None:
            return package_summary
        if package_error is not None:
            raise package_error
        raise
    if package_summary is None:
        return skill_summary
    return _merge_enriched_summary(package_summary, skill_summary)


def info(
    slug: str,
    *,
    registry_url: Optional[str] = None,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
) -> ClawHubSkillSummary:
    """Resolve the latest version metadata for ``slug``."""
    return _detail_summary(
        slug,
        registry_url=registry_url,
        timeout_sec=timeout_sec,
    )


def list_versions(
    slug: str,
    *,
    registry_url: Optional[str] = None,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
) -> List[str]:
    """Return every published version for ``slug`` (latest first when known)."""
    summary = info(slug, registry_url=registry_url, timeout_sec=timeout_sec)
    versions = list(summary.versions)
    if summary.latest_version and summary.latest_version not in versions:
        versions.insert(0, summary.latest_version)
    return versions


@dataclass
class ClawHubArchive:
    """Result of a successful :func:`download` call."""
    slug: str
    version: str
    content: bytes
    sha256: str
    content_type: str = ""

    def __post_init__(self) -> None:
        if not self.content:
            raise ClawHubClientError("Downloaded archive is empty")


def download(
    slug: str,
    *,
    version: Optional[str] = None,
    registry_url: Optional[str] = None,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC * 2,
) -> ClawHubArchive:
    """Download an archive and return bytes plus a local sha256 fingerprint.

    The sha256 describes what was received, not a registry-advertised digest;
    TLS is the current integrity anchor.
    """
    import hashlib
    cleaned = _validate_slug(slug)
    base = _registry_base_url(registry_url)
    query: Dict[str, Any] = {}
    cleaned_version = (version or "").strip()
    if cleaned_version:
        query["version"] = cleaned_version
    query["slug"] = cleaned
    url = _build_url(base, "download", query)
    body, headers = _http_get(
        url,
        timeout=timeout_sec,
        accept="application/octet-stream, application/zip",
        max_bytes=_MAX_ARCHIVE_BYTES,
    )
    if len(body) == 0:
        raise ClawHubClientError(f"Empty archive returned by {url}")
    return ClawHubArchive(
        slug=cleaned,
        version=cleaned_version,
        content=body,
        sha256=hashlib.sha256(body).hexdigest(),
        content_type=headers.get("content-type", ""),
    )


__all__ = [
    "ClawHubArchive",
    "ClawHubClientError",
    "ClawHubClientHostBlocked",
    "ClawHubNotFoundError",
    "ClawHubRateLimitError",
    "ClawHubSkillSummary",
    "download",
    "info",
    "list_versions",
    "search",
]
