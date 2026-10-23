"""Client-only MCP integration for external HTTP/SSE/stdio tool servers.

Configured servers are hot-reloaded from settings, listed tools are exposed
through ToolRegistry as provider-safe ``mcp_<server>__<tool>`` names, and each
call opens a fresh session. Secrets, server descriptions/results, and obvious
metadata SSRF targets are handled defensively because MCP servers are external.

For ``stdio``, ``command`` is an executable and ``args`` is passed as an exact
list without a shell. Optional cwd, literal and settings-backed environment selections
apply to both discovery and calls; omitted fields retain SDK defaults.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import re
import tempfile
import threading
import urllib.parse
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ouroboros.secret_masking import (
    looks_masked_secret as looks_masked_secret,
)
from ouroboros.secret_masking import (
    MCP_RESPONSE_ONLY_FIELDS,
    mask_prefixed_secret,
)
from ouroboros.mcp_headers import mask_headers, validate_headers
from ouroboros.mcp_schema import filter_schema_literals, redact_metadata, tool_metadata
from ouroboros.tools.tool_result import ToolResult
from ouroboros.platform_layer import IS_WINDOWS
from ouroboros.config import get_runtime_mode
from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency
from ouroboros.workspace_executor import resolve_process_env, validate_process_env

from ouroboros import mcp_catalog
from ouroboros.mcp_schema import (
    _redact_error_text as _redact_error_text,
    _model_facing_description, _model_facing_result, _normalize_input_schema,
    _wrap_schema_text_fields as _wrap_schema_text_fields,
)

log = logging.getLogger(__name__)


try:  # pragma: no cover - import guard exercised by tests via monkeypatch
    from mcp import ClientSession  # type: ignore
    from mcp.client.sse import sse_client  # type: ignore
    from mcp.client.stdio import StdioServerParameters, stdio_client  # type: ignore
    from mcp.client.streamable_http import streamablehttp_client  # type: ignore
    from mcp.shared._httpx_utils import create_mcp_http_client  # type: ignore  # the SDK's default factory

    _MCP_SDK_AVAILABLE = True
    _MCP_SDK_IMPORT_ERROR: Optional[str] = None
except Exception as _import_exc:  # pragma: no cover - defensive
    ClientSession = None  # type: ignore[assignment]
    streamablehttp_client = None  # type: ignore[assignment]
    sse_client = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    create_mcp_http_client = None  # type: ignore[assignment]
    _MCP_SDK_AVAILABLE = False
    _MCP_SDK_IMPORT_ERROR = f"{type(_import_exc).__name__}: {_import_exc}"


SUPPORTED_TRANSPORTS = ("streamable_http", "sse", "stdio")
TOOL_NAME_PREFIX = "mcp_"
_TOOL_NAME_PATTERN = re.compile(r"^mcp_[A-Za-z0-9_]+__[A-Za-z0-9_]+$")
_MAX_TOOL_NAME_LEN = 64
_MAX_SERVER_SLUG = 24
_MAX_TOOL_SLUG = 32
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
_SDK_DIAGNOSTIC_CONFIG = ContextVar("mcp_sdk_diagnostic_config", default=None)

# Block obvious metadata SSRF targets, but allow localhost/private LAN MCP servers.
_DENIED_HOSTS = frozenset(
    {
        "169.254.169.254",  # AWS / Azure / OCI metadata
        "100.100.100.200",  # Alibaba metadata
        "metadata.google.internal",
        "metadata",
    }
)


@dataclass(frozen=True)
class MCPServerConfig:
    """Validated MCP server config."""

    id: str
    name: str
    enabled: bool
    transport: str
    url: str
    command: str
    args: List[str]
    auth_header: str
    auth_token: str = field(repr=False)
    allowed_tools: List[str]
    cwd: str = ""
    env_from_settings: Dict[str, str] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict, repr=False)
    secret_values: tuple[str, ...] = field(default=(), repr=False)
    configuration_warnings: tuple[str, ...] = ()
    headers: Dict[str, str] = field(default_factory=dict, repr=False)

    def has_auth(self) -> bool:
        return bool(self.auth_token.strip())

    def wire_headers(self) -> Dict[str, str]:
        """Literal nonempty values, with the non-colliding legacy pair."""
        result = {name: value for name, value in self.headers.items() if value}
        if self.has_auth():
            result[self.auth_header] = self.auth_token
        return result

    def carries_credentials(self) -> bool:
        """Whether requests carry saved values: a header, the legacy token or URL userinfo."""
        parsed = urllib.parse.urlparse(self.url)
        return bool(self.wire_headers() or parsed.username or parsed.password)


@dataclass
class MCPTool:
    """Discovered MCP tool normalized for ToolRegistry."""

    server_id: str
    raw_name: str
    prefixed_name: str
    description: str
    schema: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPServerRuntime:
    """Mutable per-server tools and status."""

    config: MCPServerConfig
    tools: List[MCPTool] = field(default_factory=list)
    tool_name_collisions: List[Dict[str, str]] = field(default_factory=list)
    schema_omissions: List[str] = field(default_factory=list)
    last_error: str = ""
    last_refreshed: str = ""
    last_attempted: str = ""
    catalog: Optional[dict] = None
    publication_error: str = ""
    published_before_failure: Optional[dict] = None


def _normalized_stem(value: str) -> str:
    """The character-class normalization every slug starts from (before any cap or digest)."""
    safe = re.sub(r"[^A-Za-z0-9_]", "_", str(value or "").strip())
    return re.sub(r"_+", "_", safe).strip("_").lower()


def _name_digest(value: str) -> str:
    return hashlib.sha1(str(value or "").strip().encode("utf-8", errors="replace")).hexdigest()[:12]


def _slugify(value: str, *, max_len: int, injective: bool = False) -> str:
    """Return a provider-safe slug, hashing truncated tails to avoid collisions.

    ``injective=True`` (E5, capinv-447) also appends the hash tail whenever the
    character-class normalization was LOSSY (case folding, ``-``/``.`` → ``_``),
    so distinct legal MCP tool names like ``get-user`` / ``get_user`` /
    ``get.User`` can no longer collide into one slug and silently drop tools.
    Server ids stay non-injective: their slugs are persisted settings/UI keys.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    safe = _normalized_stem(text)
    if not safe:
        return ""
    lossy = injective and safe != text
    if len(safe) <= max_len and not lossy:
        return safe
    digest = _name_digest(text)
    keep = max_len - len(digest) - 1
    if keep <= 0:
        return digest
    return f"{safe[:keep]}_{digest}"


def canonical_server_id(value: str) -> str:
    """Canonicalize the id shared by settings, UI routes, Presence and MCPManager.

    A fixed point (#1328): one ``_slugify`` pass can join a cut stem ending in
    ``_`` to its digest as ``__``, which ``parse_tool_name`` then splits early.
    The second pass collapses that join; it is exactly the slug every advertised
    ``mcp_<server>__<tool>`` name already carried, so wire names are unchanged
    and ``parse_tool_name(make_tool_name(cfg.id, t))["server_slug"] == cfg.id``.
    """
    return _slugify(_slugify(value, max_len=_MAX_SERVER_SLUG), max_len=_MAX_SERVER_SLUG)


def raw_server_id(entry: Any) -> str:
    """The canonical identity one raw ``MCP_SERVERS`` entry claims (loader expression)."""
    if not isinstance(entry, dict):
        return ""
    return canonical_server_id(entry.get("id") or entry.get("slug") or entry.get("name"))


def make_tool_name(server_id: str, tool_name: str) -> str:
    """Return provider-safe ``mcp_<server>__<tool>``."""
    server_slug = canonical_server_id(server_id)
    tool_slug = _slugify(tool_name, max_len=_MAX_TOOL_SLUG, injective=True)
    if not server_slug or not tool_slug:
        return ""
    candidate = f"{TOOL_NAME_PREFIX}{server_slug}__{tool_slug}"
    if len(candidate) > _MAX_TOOL_NAME_LEN:
        # If capped parts still overflow, hash the combined tail.
        digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:6]
        candidate = f"{TOOL_NAME_PREFIX}{server_slug}__{digest}"
    return candidate


def parse_tool_name(name: str) -> Optional[Dict[str, str]]:
    """Split an ``mcp_<server>__<tool>``-shaped name, or return ``None``.

    Syntactic only: it proves neither catalog membership nor a raw name (a
    digest is not reversible); :meth:`MCPManager.resolve_tool_name` answers those.
    """
    text = str(name or "")
    if not text.startswith(TOOL_NAME_PREFIX):
        return None
    if not _TOOL_NAME_PATTERN.match(text):
        return None
    body = text[len(TOOL_NAME_PREFIX):]
    if "__" not in body:
        return None
    server, tool = body.split("__", 1)
    return {"server_slug": server, "tool_slug": tool}


def is_mcp_tool_name(name: str) -> bool:
    """Return whether ``name`` has the MCP wire-name shape (not whether it is listed)."""
    return parse_tool_name(name) is not None


_DIGEST_TAIL_RE = re.compile(r"_([0-9a-f]{12})$")


def naming_rule_matches(requested_tool: str, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows of ONE server that the naming rule relates exactly to a missed name.

    ``requested_tool`` is the part after ``mcp_<server>__``; ``tools`` are that
    server's callable rows (``name`` + ``raw_name``). A row matches when the
    requested part normalizes like the row's raw name, equals the row's
    registered stem without its digest, or ends in the row's digest (sha1 of
    the raw name). These are identities of ``make_tool_name``, not similarity:
    nothing is ranked, and several matches are an ambiguity the caller resolves.
    """
    text = str(requested_tool or "").strip()
    wanted = _normalized_stem(text)
    tail = _DIGEST_TAIL_RE.search(text)
    matches: List[Dict[str, Any]] = []
    for tool in tools:
        raw = str(tool.get("raw_name") or "")
        digest = _name_digest(raw)
        slug = str(tool.get("name") or "").partition("__")[2]
        stem = slug[:-len(digest) - 1] if slug.endswith("_" + digest) else ""
        related = bool(wanted) and wanted in {_normalized_stem(raw), stem}
        if raw and (related or (tail is not None and tail.group(1) == digest)):
            matches.append(tool)
    return matches


@dataclass(frozen=True)
class MCPNameResolution:
    """What this process's current MCP catalog says about one requested name.

    A pure lookup under the manager lock: no transport, refresh or settings
    write. Only ``callable`` may proceed to the paid safety check and the call;
    every other status is its own typed fact, kept distinct from a catalog miss.
    """

    status: str  # callable | not_found | disallowed | server_disabled | catalog_unavailable | mcp_disabled
    server_id: str = ""
    raw_name: str = ""
    detail: str = ""

    def refusal(self, requested: str) -> ToolResult:
        """The reason alone; the registry composes the caller's callable view beside it."""
        name, server = str(requested or ""), self.server_id
        if self.status == "disallowed":
            text = (
                f"⚠️ MCP_TOOL_DISALLOWED: {self.raw_name!r} is not on the "
                f"allowed_tools list for server {server!r}."
            )
            return ToolResult(status="blocked", code="ACCESS_BLOCKED", text=text)
        if self.status == "not_found":
            where = (f"the current tool catalog of MCP server {server!r}" if server
                     else "the tool catalog of any configured MCP server")
            text = f"⚠️ MCP_TOOL_NOT_FOUND: {name!r} is {self.detail or 'not in ' + where}. Nothing was executed."
            return ToolResult(status="error", code="UNKNOWN_TOOL", text=text)
        if self.status == "mcp_disabled":
            text = "⚠️ MCP_DISABLED: enable MCP in Settings → Advanced to use this tool."
        elif self.status == "server_disabled":
            text = (f"⚠️ MCP_DISABLED: MCP server {server!r} is disabled in Settings → "
                    f"Advanced; {name!r} was not executed.")
        else:
            text = (f"⚠️ MCP_CATALOG_UNAVAILABLE: MCP server {server!r} has incomplete discovery evidence "
                    f"({self.detail}); whether {name!r} exists is "
                    "unknown. Nothing was executed.")
        return ToolResult(status="unavailable", code="MCP_CATALOG_UNAVAILABLE" if self.status == "catalog_unavailable" else "MCP_UNAVAILABLE", text=text)


def _validate_url(url: str) -> str:
    """Normalize HTTP(S) URLs, applying metadata policy outside Cyber."""
    text = str(url or "").strip()
    if not text:
        raise ValueError("url is required")
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            "MCP server url must use http:// or https:// (got "
            f"{parsed.scheme or 'no scheme'!r})"
        )
    host = (parsed.hostname or "").strip().lower()
    host = host.rstrip(".")
    if not host:
        raise ValueError("MCP server url is missing a hostname")
    # Python 3.10.11 accepts malformed bracketed hosts while newer urllib
    # releases reject them during parsing. Keep URL admission consistent on
    # the older CI runtimes by applying the RFC 3986 bracket-host check here.
    if "[" in parsed.netloc:
        if host.startswith("v"):
            if not re.fullmatch(r"v[0-9a-f]+\..+", host):
                raise ValueError("MCP server url contains an invalid IPvFuture host")
        else:
            try:
                address = ipaddress.ip_address(host)
            except ValueError as exc:
                raise ValueError("MCP server url contains an invalid bracketed host") from exc
            if isinstance(address, ipaddress.IPv4Address):
                raise ValueError("MCP server url cannot bracket an IPv4 host")
    parsed.port  # Raises for a malformed or out-of-range port.
    if mode_has_unrestricted_agency(get_runtime_mode()):
        return text
    if parsed.username or parsed.password:
        raise ValueError("MCP server url must not include username/password credentials")
    if host in _DENIED_HOSTS:
        raise ValueError(f"MCP server hostname {host!r} is on the deny list")
    # Also block link-local IPs when supplied with a port or IPv6 mapping.
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        addr = None
    if addr is not None and addr.is_link_local:
        raise ValueError(
            f"MCP server hostname {host!r} is a link-local address"
        )
    mapped = getattr(addr, "ipv4_mapped", None) if addr is not None else None
    if mapped is not None and (str(mapped) in _DENIED_HOSTS or mapped.is_link_local):
        raise ValueError(
            f"MCP server hostname {host!r} maps to a denied IPv4 address"
        )
    return text


def _validate_auth_header(value: str) -> str:
    text = str(value or "Authorization").strip() or "Authorization"
    if not _HEADER_NAME_RE.match(text):
        raise ValueError("MCP auth_header must be a single HTTP header token")
    return text


def _validate_auth_token(value: str) -> str:
    text = str(value or "").strip()
    if _CONTROL_CHARS_RE.search(text):
        raise ValueError("MCP auth_token must not contain control characters")
    return text


def _validate_stdio_command(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("MCP stdio command must be a string")
    command = value.strip()
    if not command:
        raise ValueError("MCP stdio command is required")
    if _CONTROL_CHARS_RE.search(command):
        raise ValueError("MCP stdio command must not contain control characters")
    return command


def _validate_stdio_args(value: Any) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("MCP stdio args must be a list of strings")
    if any("\x00" in item for item in value):
        raise ValueError("MCP stdio args must not contain NUL characters")
    return list(value)


def _coerce_str_list(value: Any) -> List[str]:
    if value in (None, "", [], ()):
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def normalize_server_config(
    raw: Dict[str, Any], *, settings: Optional[Dict[str, Any]] = None,
    errors: Optional[List[str]] = None,
) -> Optional[MCPServerConfig]:
    """Validate one ``MCP_SERVERS`` entry; return ``None`` if unsalvageable."""
    def invalid(message: str) -> None:
        if errors is not None:
            errors.append(f"MCP_CONFIG_ERROR: {message}")
        log.warning("Invalid MCP server config: %s", message)

    if not isinstance(raw, dict):
        return invalid("server must be an object")
    raw = {key: value for key, value in raw.items() if key not in MCP_RESPONSE_ONLY_FIELDS}

    server_slug = raw_server_id(raw)
    if not server_slug:
        return invalid("server id or name is required")

    transport = str(raw.get("transport") or "streamable_http").strip().lower()
    if transport not in SUPPORTED_TRANSPORTS:
        return invalid("unsupported transport; use streamable_http, sse, or stdio")

    try:
        known = {"id", "slug", "name", "label", "enabled", "transport", "url", "command",
                 "args", "auth_header", "auth_token", "allowed_tools", "cwd", "env", "env_from_settings", "headers"}
        unknown = set(raw) - known
        warnings = ("Fields retained but not applied: " + ", ".join(sorted(map(str, unknown))),) if unknown else ()
        cwd = raw.get("cwd", "")
        if not isinstance(cwd, str) or "\x00" in cwd:
            raise ValueError("cwd must be a string without NUL")
        refs = validate_process_env(raw.get("env_from_settings"))
        env, secret_values = resolve_process_env(raw.get("env"), refs, settings=settings)
        unused = ("url", "auth_token", "headers") if transport == "stdio" else ("command", "args", "cwd", "env", "env_from_settings")
        if any(raw.get(key) for key in unused):
            raise ValueError("fields unsupported by this transport: " + ", ".join(key for key in unused if raw.get(key)))
        if transport == "stdio" and raw.get("auth_header", "Authorization") not in (None, "", "Authorization"):
            raise ValueError("auth_header is unsupported for stdio")
        if transport == "stdio":
            url = ""
            command = _validate_stdio_command(raw.get("command"))
            args = _validate_stdio_args(raw.get("args"))
            auth_header = "Authorization"
            auth_token = ""
        else:
            url = _validate_url(raw.get("url") or "")
            parsed_url = urllib.parse.urlparse(url)
            url_secrets = tuple(value for value in (parsed_url.username, parsed_url.password) if value)
            secret_values = (*secret_values, *url_secrets, *(urllib.parse.unquote(value) for value in url_secrets))
            command = ""
            args = []
            auth_header = _validate_auth_header(raw.get("auth_header") or "Authorization")
            auth_token = _validate_auth_token(raw.get("auth_token") or "")
        # Header values are forwarded configuration, not redaction keys: Settings masks them
        # and the origin guard bounds their destination, but text from or about the server
        # is never searched for them (a value may be "7" or "en"; an echo stays visible).
        headers = validate_headers(raw.get("headers"), legacy_header=auth_header, legacy_token=auth_token)
    except ValueError as exc:
        return invalid(str(exc))

    name = str(raw.get("name") or raw.get("label") or server_slug).strip() or server_slug
    enabled_raw = raw.get("enabled", False)
    if isinstance(enabled_raw, bool):
        enabled = enabled_raw
    else:
        enabled = str(enabled_raw or "").strip().lower() in {"1", "true", "yes", "on"}

    allowed_tools = _coerce_str_list(raw.get("allowed_tools"))

    return MCPServerConfig(
        id=server_slug,
        name=name,
        enabled=enabled,
        transport=transport,
        url=url,
        command=command,
        args=args,
        auth_header=auth_header,
        auth_token=auth_token,
        allowed_tools=allowed_tools,
        cwd=cwd,
        env_from_settings=refs,
        env=env,
        secret_values=secret_values,
        configuration_warnings=warnings,
        headers=headers,
    )


def parse_servers(
    raw_list: Any, *, settings: Optional[Dict[str, Any]] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
) -> List[MCPServerConfig]:
    """Normalize a raw ``MCP_SERVERS`` list. Invalid entries are warned and skipped.

    Identity is judged on the FULL raw list before anything is dropped: when
    several entries claim one canonical id, none of them is used — keeping the
    first would silently retarget its tools, Presence grants and saved secrets
    to whichever entry happens to come first. That id gets one typed
    ``MCP_ID_AMBIGUOUS`` row; every other server keeps working.
    """
    if not isinstance(raw_list, list):
        return []
    claimants: Dict[str, List[Dict[str, Any]]] = {}
    for entry in raw_list:
        server_id = raw_server_id(entry)
        if server_id:
            claimants.setdefault(server_id, []).append(entry)
    ambiguous = {server_id: entries for server_id, entries in claimants.items() if len(entries) > 1}
    out: List[MCPServerConfig] = []
    for entry in raw_list:
        if raw_server_id(entry) in ambiguous:
            continue
        entry_errors: List[str] = []
        cfg = normalize_server_config(entry, settings=settings, errors=entry_errors)
        if cfg is None:
            if errors is not None:
                source = entry if isinstance(entry, dict) else {}
                errors.append({"id": raw_server_id(source),
                               "enabled": source.get("enabled", False), "tool_count": 0,
                               "last_error": "; ".join(entry_errors), "code": "MCP_CONFIG_ERROR"})
            continue
        out.append(cfg)
    for server_id, entries in ambiguous.items():
        message = (f"MCP_ID_AMBIGUOUS: {len(entries)} configured servers resolve to server id "
                   f"{server_id!r}; none of them is used until their ids are made distinct.")
        log.warning("Invalid MCP server config: %s", message)
        if errors is not None:
            errors.append({"id": server_id, "enabled": any(e.get("enabled", False) for e in entries),
                           "tool_count": 0, "last_error": message, "code": "MCP_ID_AMBIGUOUS"})
    return out


def redact_servers_for_status(configs: List[MCPServerConfig]) -> List[Dict[str, Any]]:
    """Return UI-safe server configs with auth tokens masked."""
    out: List[Dict[str, Any]] = []
    for cfg in configs:
        out.append(
            {
                "id": cfg.id,
                "name": cfg.name,
                "enabled": cfg.enabled,
                "transport": cfg.transport,
                "url": _redact_error_text(cfg.url, cfg),
                "auth_header": cfg.auth_header,
                "auth_token": mask_prefixed_secret(cfg.auth_token, visible_chars=4),
                "auth_configured": cfg.has_auth(),
                "headers": mask_headers(cfg.headers),
                "allowed_tools": list(cfg.allowed_tools),
                "cwd": cfg.cwd,
                "env_from_settings": dict(cfg.env_from_settings),
            }
        )
    return out

# Async transport.


@contextmanager
def _redacted_sdk_logs(cfg: MCPServerConfig, logger_name: str):
    """Project this connection's SDK log records through ``_redact_error_text``.

    Invalid server JSON makes the transport logger record a validation traceback,
    and an invalid message makes the shared session warn on the root logger, both
    with the raw input (an echoed saved token or environment secret) before Ouroboros
    sees an error. The record's message and formatted exception become one
    redacted message. The selection is a context variable the SDK's tasks
    inherit, so concurrent servers and other root records are never projected;
    the filters leave with the connection.
    """
    sdk_logs = (logging.getLogger(logger_name), logging.getLogger())

    def project_log(record: logging.LogRecord) -> bool:
        if _SDK_DIAGNOSTIC_CONFIG.get() is cfg:
            message = record.getMessage()
            if record.exc_info:
                message += "\n" + logging.Formatter().formatException(record.exc_info)
                record.exc_info = record.exc_text = None
            record.msg, record.args = _redact_error_text(message, cfg), ()
        return True

    token = _SDK_DIAGNOSTIC_CONFIG.set(cfg)
    for sdk_log in sdk_logs:
        sdk_log.addFilter(project_log)
    try:
        yield
    finally:
        for sdk_log in sdk_logs:
            sdk_log.removeFilter(project_log)
        _SDK_DIAGNOSTIC_CONFIG.reset(token)


@asynccontextmanager
async def _stdio_with_diagnostics(params: Any, cfg: MCPServerConfig):
    # SDK stderr otherwise goes straight to the host console, outside masking.
    with _redacted_sdk_logs(cfg, "mcp.client.stdio"):
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stderr:
            try:
                async with stdio_client(params, errlog=stderr) as streams:
                    yield streams
            finally:
                stderr.seek(0)
                diagnostic = _redact_error_text(stderr.read(), cfg)
                if diagnostic:
                    log.warning("MCP stdio stderr (%s): %s", cfg.id, diagnostic)


class MCPRedirectRefused(RuntimeError):
    """A request that would carry saved credentials to another origin; it was never sent."""

    code = "MCP_REDIRECT_REFUSED"


def _origin(url: Any) -> tuple:
    """``(scheme, host, port)`` with the scheme's default port made explicit."""
    import httpx

    parsed = url if isinstance(url, httpx.URL) else httpx.URL(str(url))
    return parsed.scheme, parsed.host, parsed.port or {"http": 80, "https": 443}.get(parsed.scheme, 0)


def _origin_text(origin: tuple) -> str:
    """An origin for messages: never userinfo, path or query."""
    scheme, host, port = origin
    shown = f"[{host}]" if ":" in host else host
    return f"{scheme}://{shown}" + ("" if port == {"http": 80, "https": 443}.get(scheme) else f":{port}")


class _CredentialOriginGuard:
    """One connection's HTTP client factory: saved credentials stay on the configured origin.

    The SDK client follows redirects, and the HTTP client strips only
    ``Authorization`` when a redirect leaves the origin — every other configured
    header would follow. While the server carries credentials, a request to a
    different scheme, host or port (``http`` → ``https`` included) is stopped by
    a request hook before it is sent; requests on the configured origin,
    same-origin redirects included, are unchanged. A server without credentials
    keeps the SDK's redirect behavior. The SDK factory keeps its timeouts and
    the transport keeps the client's lifetime.
    """

    def __init__(self, cfg: MCPServerConfig) -> None:
        self.cfg = cfg
        self.refusal: Optional[MCPRedirectRefused] = None

    def client(self, headers: Any = None, timeout: Any = None, auth: Any = None) -> Any:
        client = create_mcp_http_client(headers=headers, timeout=timeout, auth=auth)
        if self.cfg.carries_credentials():
            async def stay_on_origin(request: Any) -> None:
                expected, actual = _origin(self.cfg.url), _origin(request.url)
                if actual == expected:
                    return
                self.refusal = MCPRedirectRefused(
                    f"MCP_REDIRECT_REFUSED: server {self.cfg.id!r} at {_origin_text(expected)} redirected a "
                    f"request to {_origin_text(actual)}. Saved headers go only to the configured origin (scheme, "
                    "host and port), so that request was stopped before it was sent. If the service has moved, "
                    "enter its new address as the Server URL.")
                # Ending the connection fails the pending session call now; an
                # SSE message POST would otherwise wait for its tool timeout.
                await client.aclose()
                raise self.refusal

            client.event_hooks["request"].append(stay_on_origin)
        return client


@asynccontextmanager
async def _http_with_guards(cfg: MCPServerConfig):
    """One HTTP/SSE connection: the SDK transport behind the origin guard, with redacted SDK logs.

    Configured headers are the client's defaults and go as written, except names the
    transport sets per request: Streamable HTTP sets Accept and Content-Type on every
    request and its session and protocol-version headers once known; SSE sets Accept and
    Cache-Control on its stream, while its message POSTs keep a configured Accept or
    Content-Type. URL userinfo (Cyber only) makes the HTTP client send Basic credentials
    in Authorization instead of a configured value.
    """
    guard = _CredentialOriginGuard(cfg)
    sse = cfg.transport == "sse"
    with _redacted_sdk_logs(cfg, "mcp.client.sse" if sse else "mcp.client.streamable_http"):
        opened = (sse_client if sse else streamablehttp_client)(
            cfg.url, headers=cfg.wire_headers(), httpx_client_factory=guard.client)
        try:
            async with opened as streams:
                yield streams
        except Exception:
            # The SDK's task groups wrap the hook's refusal or report a closed
            # connection instead; name the stopped request. Cancellation and
            # other base exceptions pass through unchanged.
            if guard.refusal is None:
                raise
            raise MCPRedirectRefused(str(guard.refusal)) from None


def _transport_factory(cfg: MCPServerConfig):
    if cfg.transport in ("streamable_http", "sse"):
        return _http_with_guards(cfg)
    if cfg.transport == "stdio":
        # Leaving env/cwd unset uses the SDK's small cross-platform default
        # environment and its context-managed process shutdown sequence.
        # The SDK merges a plain dict onto its defaults. Windows env names are
        # case-insensitive: canonical names prevent a second Path/PATH entry.
        from ouroboros.process_scope import installation_scope_entries

        # The installation scope marker rides every stdio server (a long-lived shared
        # helper belongs to the installation, never to the task that first touched it).
        env = {**installation_scope_entries(),
               **{key.upper() if IS_WINDOWS else key: value for key, value in cfg.env.items()}}
        selected = {"env": env} if env else {}
        if cfg.cwd:
            selected["cwd"] = cfg.cwd
        params = StdioServerParameters(command=cfg.command, args=list(cfg.args), **selected)
        return _stdio_with_diagnostics(params, cfg) if cfg.env else stdio_client(params)
    raise RuntimeError(f"Unsupported transport: {cfg.transport!r}")


async def _list_tools_async(cfg: MCPServerConfig, *, timeout_sec: int) -> List[Dict[str, Any]]:
    """Connect to ``cfg`` and return raw tools; errors surface to status."""
    if not _MCP_SDK_AVAILABLE:
        raise RuntimeError(
            "MCP client SDK not installed. Add `mcp>=1.10` to the runtime."
        )
    from ouroboros.owner_pause import submit_async_preparation

    async def _do_with_session() -> List[Dict[str, Any]]:
        async with _transport_factory(cfg) as transport_ctx:
            # Both transports yield read/write streams.
            streams = transport_ctx
            if isinstance(streams, tuple):
                read, write = streams[0], streams[1]
            else:
                read, write = streams.read, streams.write  # pragma: no cover
            async with ClientSession(read, write) as session:
                await session.initialize()
                # Follow nextCursor (E5, capinv-447): the MCP listing is paginated
                # and a single call silently loses every tool after page one. The
                # page bound only guards against a cursor loop from a broken server.
                tools_raw: List[Dict[str, Any]] = []
                cursor: Any = None
                for _page in range(50):
                    result = await (
                        session.list_tools(cursor=cursor) if cursor else session.list_tools()
                    )
                    for tool in result.tools or []:
                        tools_raw.append(
                            {
                                "name": getattr(tool, "name", ""),
                                "description": getattr(tool, "description", "") or "",
                                "input_schema": getattr(tool, "inputSchema", {}) or {},
                            }
                        )
                        for key in ("annotations", "metadata"):
                            value = getattr(tool, key, None)
                            if value is not None:
                                tools_raw[-1][key] = value.model_dump(mode="json", by_alias=True) if hasattr(value, "model_dump") else value
                        if metadata := tool_metadata(tool):
                            tools_raw[-1]["metadata"] = {**(tools_raw[-1].get("metadata") or {}), **metadata}
                    cursor = getattr(result, "nextCursor", None)
                    if not cursor:
                        break
                else:
                    if cursor:
                        # Cap exhausted with pages remaining: a partial catalog
                        # must not read as the complete one (#447 P1).
                        tools_raw.append({
                            "name": "",
                            "_pagination_truncated": True,
                        })
                        log.warning(
                            "MCP tool listing stopped at the 50-page bound with "
                            "nextCursor still present; catalog is PARTIAL",
                        )
                return tools_raw

    # Server listings and draft probes name no task; no task's Pause gates them.
    return await asyncio.wait_for(
        submit_async_preparation(_do_with_session), timeout=timeout_sec
    )


async def _call_tool_async(
    cfg: MCPServerConfig, tool_name: str, arguments: Dict[str, Any], *, timeout_sec: int
) -> ToolResult:
    """Open a fresh session and preserve the SDK-owned error bit."""
    if not _MCP_SDK_AVAILABLE:
        raise RuntimeError(
            "MCP client SDK not installed. Add `mcp>=1.10` to the runtime."
        )
    from ouroboros.owner_pause import submit_async_operation, submit_async_preparation, OwnerPauseRefused, _TOOL_OPERATION
    from ouroboros.model_wait import ModelWaitInterrupted
    from ouroboros.usage_accounting import UsageAccountingError
    from ouroboros.store import StoreError
    unsent_joined = False

    async def _do() -> ToolResult:
        nonlocal unsent_joined
        refusal = None
        async with _transport_factory(cfg) as transport_ctx:
            streams = transport_ctx
            if isinstance(streams, tuple):
                read, write = streams[0], streams[1]
            else:
                read, write = streams.read, streams.write  # pragma: no cover
            async with ClientSession(read, write) as session:
                try:
                    await session.initialize()
                except BaseException as exc:
                    # Includes timeout cancellation before tools/call. Keep the
                    # original exception, but join both contexts before proof.
                    refusal = exc
                else:
                    try:
                        pending = submit_async_operation(None, session.call_tool, tool_name, arguments)
                    except (OwnerPauseRefused, ModelWaitInterrupted, UsageAccountingError, StoreError) as exc:
                        refusal = exc
                    else:
                        result = await pending  # any error here may follow a send
        if refusal is not None:
            # Failed/unfinished cleanup never reaches this positive phase fact.
            unsent_joined = True
            active = _TOOL_OPERATION.get()
            if active:
                active[2]["settled"] = True
            refusal.mcp_unsent_joined = True
            raise refusal
        return _tool_result_from_call_result(result)

    # Launch admission precedes transport entry: a stdio server starts there.
    try:
        pending = submit_async_preparation(_do)
    except (OwnerPauseRefused, ModelWaitInterrupted, UsageAccountingError, StoreError) as exc:
        exc.mcp_unsent_joined = True  # no transport task was handed off
        raise
    try:
        return await asyncio.wait_for(pending, timeout=timeout_sec)
    except BaseException as exc:
        if unsent_joined:
            # wait_for replaces CancelledError with TimeoutError; carry proof,
            # not exception-class inference, over that conversion.
            exc.mcp_unsent_joined = True
        raise


def _stringify_call_result(result: Any) -> str:
    """Stringify MCP content parts without inventing fields."""
    parts: List[str] = []
    is_error = bool(getattr(result, "isError", False) or getattr(result, "is_error", False))
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
            continue
        # Best-effort JSON dump for non-text parts.
        try:
            parts.append(json.dumps(_serialize_content_part(item), ensure_ascii=False))
        except Exception:
            parts.append(repr(item))
    # structuredContent is protocol data in its own right (E5, capinv-447):
    # carry it even when text/content parts exist, unless it duplicates one.
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        try:
            dumped = json.dumps(structured, ensure_ascii=False)
        except Exception:
            dumped = repr(structured)
        if dumped not in parts:
            parts.append(f"structuredContent: {dumped}" if parts else dumped)
    body = "\n\n".join(parts).strip() or "(empty result)"
    if is_error:
        return f"⚠️ MCP_TOOL_ERROR: {body}"
    return body


def _tool_result_from_call_result(result: Any) -> ToolResult:
    """Preserve the SDK error bit without trusting result-body markers."""
    is_error = bool(
        getattr(result, "isError", False)
        or getattr(result, "is_error", False)
    )
    return ToolResult(
        status="error" if is_error else "ok",
        code="MCP_ERROR" if is_error else "OK",
        text=_stringify_call_result(result),
        meta={"mcp_is_error": is_error},
    )


def _serialize_content_part(item: Any) -> Dict[str, Any]:
    """Best-effort conversion of an MCP content part into a JSON-safe dict."""
    out: Dict[str, Any] = {}
    for attr in ("type", "uri", "mimeType", "data", "annotations"):
        value = getattr(item, attr, None)
        if value is not None and not callable(value):
            out[attr] = value
    # EmbeddedResource carries its payload in a NESTED resource object (E5,
    # capinv-447): omitting it dropped the text/blob of every embedded resource.
    resource = getattr(item, "resource", None)
    if resource is not None and not callable(resource):
        nested: Dict[str, Any] = {}
        for attr in ("uri", "mimeType", "text", "blob"):
            value = getattr(resource, attr, None)
            if value is not None and not callable(value):
                nested[attr] = str(value) if attr == "uri" else value
        if nested:
            out["resource"] = nested
    if "uri" in out:
        out["uri"] = str(out["uri"])  # AnyUrl is not JSON-serializable
    return out


# Sync wrapper.


def _run_async(coro_factory: Callable[[], Awaitable[Any]], *, join_timeout: Optional[int] = None) -> Any:
    """Run async work from sync code; the factory avoids reusing closed coroutines."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    # Existing loop: run the coroutine in a sub-thread.
    holder: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            holder["value"] = asyncio.run(coro_factory())
        except BaseException as exc:
            holder["error"] = exc

    import contextvars

    context = contextvars.copy_context()
    thread = threading.Thread(target=lambda: context.run(_runner), name="mcp-sync-runner", daemon=True)
    thread.start()
    thread.join(timeout=join_timeout)
    if thread.is_alive():
        raise TimeoutError("MCP async runner did not finish before timeout")
    if "error" in holder:
        raise holder["error"]
    return holder.get("value")


# Manager singleton.


def normalize_listing(cfg: MCPServerConfig, tools_raw: list, complete: bool) -> tuple:
    """Pure provider presentation shared by publication, seeding and adoption."""
    tools_raw, literal_omissions = filter_schema_literals(tools_raw, lambda text: _redact_error_text(text, cfg))
    normalized = [
        MCPTool(
            server_id=cfg.id,
            raw_name=str(item.get("name") or "").strip(),
            prefixed_name=make_tool_name(cfg.id, item.get("name") or ""),
            description=_redact_error_text(str(item.get("description") or ""), cfg),
            schema=_normalize_input_schema(item.get("input_schema"), cfg),
            metadata=redact_metadata(item.get("metadata") or {}, lambda text: _redact_error_text(text, cfg)),
        )
        for item in tools_raw
        if str(item.get("name") or "").strip()
    ]
    normalized = [tool for tool in normalized if tool.prefixed_name]
    # Drop duplicates caused by slug collisions. A residual collision (the
    # 12-hex tail makes it astronomically rare) DROPS a tool — a capability
    # omission that must be disclosed, not silently swallowed (#447 P1).
    seen: Dict[str, MCPTool] = {}
    deduped: List[MCPTool] = []
    collisions: List[Dict[str, str]] = []
    for tool in normalized:
        if tool.prefixed_name in seen:
            kept = seen[tool.prefixed_name]
            collisions.append({
                "prefixed_name": tool.prefixed_name,
                "kept_raw_name": kept.raw_name,
                "dropped_raw_name": tool.raw_name,
            })
            continue
        seen[tool.prefixed_name] = tool
        deduped.append(tool)
    omission_notes: List[str] = literal_omissions
    if collisions:
        omission_notes.append(
            f"{len(collisions)} tool(s) unavailable due to slug collision: "
            + ", ".join(item["dropped_raw_name"] for item in collisions[:5])
        )
    if not complete:
        omission_notes.append(
            "tool catalog is PARTIAL: listing stopped at the 50-page bound "
            "with more pages remaining"
        )

    return deduped, collisions, omission_notes


class MCPManager:
    """Process-wide MCP server/tool registry."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._enabled = False
        self._tool_timeout_sec = 60
        self._servers: Dict[str, MCPServerRuntime] = {}
        self._configuration_errors: List[Dict[str, Any]] = []
        self._configured = False
        self._settings_fingerprint = ""
        self._settings_mtime_ns: Optional[int] = None
        self._refresh_running = False
        self._refresh_pending = False
        self._listing_locks: dict[str, threading.Lock] = {}
        self._catalog_reader = mcp_catalog.CatalogReader()
        self._lister_pid: Optional[int] = None
        # The last server-owned event, separate from passive GET reconfiguration.
        self._listing_fingerprint = ""
        # Test hook; production uses the real async transports.
        self._async_list_tools: Callable[[MCPServerConfig, int], Awaitable[List[Dict[str, Any]]]] = (
            lambda cfg, timeout: _list_tools_async(cfg, timeout_sec=timeout)
        )
        self._async_call_tool: Callable[
            [MCPServerConfig, str, Dict[str, Any], int], Awaitable[ToolResult]
        ] = (
            lambda cfg, name, args, timeout: _call_tool_async(
                cfg, name, args, timeout_sec=timeout
            )
        )

    @property
    def is_lister(self) -> bool:
        return self._lister_pid == os.getpid()

    def become_lister(self) -> None:
        # A forked worker must not inherit its parent's discovery role.
        self._lister_pid = os.getpid()

    # -- configuration ------------------------------------------------------

    @staticmethod
    def _fingerprint(settings: Dict[str, Any]) -> str:
        payload = {
            "MCP_ENABLED": settings.get("MCP_ENABLED"),
            "MCP_TOOL_TIMEOUT_SEC": settings.get("MCP_TOOL_TIMEOUT_SEC"),
            "MCP_SERVERS": settings.get("MCP_SERVERS"),
            "effective_runtime_mode": get_runtime_mode(),
            "saved_runtime_mode": settings.get("OUROBOROS_RUNTIME_MODE"),
        }
        references = set()
        servers = settings.get("MCP_SERVERS")
        for server in servers if isinstance(servers, list) else []:
            refs = server.get("env_from_settings") if isinstance(server, dict) else None
            if isinstance(refs, dict):
                references.update(ref for ref in refs.values() if isinstance(ref, str))
        payload["selected_settings"] = {ref: settings.get(ref) for ref in references}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()

    def reconfigure(self, settings: Dict[str, Any], *, settings_mtime_ns: Optional[int] = None) -> bool:
        """Rebuild servers, preserving tools for unchanged configs."""
        with self._lock:
            fingerprint = self._fingerprint(settings)
            if self._configured and fingerprint == self._settings_fingerprint:
                if settings_mtime_ns is not None:
                    self._settings_mtime_ns = settings_mtime_ns
                self.reload_catalogs()
                return False
            self._configured = True
            self._settings_fingerprint = fingerprint
            self._settings_mtime_ns = settings_mtime_ns
            self._enabled = bool(settings.get("MCP_ENABLED"))
            try:
                self._tool_timeout_sec = max(1, int(settings.get("MCP_TOOL_TIMEOUT_SEC") or 60))
            except (TypeError, ValueError):
                self._tool_timeout_sec = 60
            self._configuration_errors = []
            new_configs = parse_servers(settings.get("MCP_SERVERS"), settings=settings, errors=self._configuration_errors)
            new_servers: Dict[str, MCPServerRuntime] = {}
            for cfg in new_configs:
                old = self._servers.get(cfg.id)
                if old is not None and mcp_catalog.identity(old.config) == mcp_catalog.identity(cfg):
                    old.config = cfg
                    new_servers[cfg.id] = old
                else:
                    new_servers[cfg.id] = MCPServerRuntime(config=cfg)
            self._servers = new_servers
            self._catalog_reader.retain(new_configs)
            self.reload_catalogs()
            return True

    def _adopt_catalog(self, runtime: MCPServerRuntime, envelope: Optional[dict]) -> None:
        """ONE post-listing item path for live publication, boot seed and adoption."""
        runtime.catalog = envelope
        items = envelope["items"] if envelope else []
        runtime.tools, runtime.tool_name_collisions, notes = normalize_listing(
            runtime.config, items, bool(envelope and envelope["complete"]))
        notes.extend(envelope.get("schema_omissions", []) if envelope else [])
        runtime.schema_omissions = list(dict.fromkeys(note for note in notes if note.startswith("MCP_SCHEMA_")))
        from ouroboros.tool_effects import record_mcp_annotations  # every process learns the published hints
        record_mcp_annotations(runtime.config.id, items)
        attempt = envelope["latest_attempt"] if envelope else {}
        runtime.last_refreshed = (envelope.get("listed_at") or "") if envelope else ""
        runtime.last_attempted = attempt.get("at", "")
        runtime.last_error = attempt.get("error", "") or ("; ".join(notes) if envelope else "")

    def reload_catalogs(self) -> None:
        """Adopt changed published files, without modifying any resident request array."""
        with self._lock:
            for runtime in self._servers.values():
                envelope = self._catalog_reader.load(runtime.config)
                # A failed publication retains the local listing until disk changes.
                if runtime.publication_error:
                    if envelope is runtime.published_before_failure:
                        continue
                    runtime.publication_error = ""
                if envelope is not runtime.catalog:
                    self._adopt_catalog(runtime, envelope)

    def configuration_changed(self, settings: dict, *, reason: str) -> None:
        """Server-owned boot/save/mtime event: prune, then schedule the latest pass."""
        with self._lock:
            self.reconfigure(settings)
            mcp_catalog.prune_catalogs([rt.config for rt in self._servers.values()])
            self._listing_fingerprint = self._settings_fingerprint
        self.refresh_all_background(reason=reason)

    def catalog_notices(self) -> list[dict]:
        with self._lock:
            if not self._enabled:
                return []
            return [{"id": item["id"], "detail": f"not listed under the current configuration: {item['last_error']}"}
                    for item in self._configuration_errors if item["enabled"]] + [
                    {"id": rt.config.id, "detail": mcp_catalog.catalog_notice(rt)}
                    for rt in self._servers.values() if rt.config.enabled and (
                        not rt.catalog or not rt.catalog.get("complete") or rt.catalog["latest_attempt"].get("partial")
                        or rt.last_error or rt.publication_error)]

    # -- introspection ------------------------------------------------------

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def server_ids(self) -> List[str]:
        with self._lock:
            return list(self._servers.keys())

    def server_count(self) -> int:
        with self._lock:
            return len(self._servers)

    def settings_mtime_ns(self) -> Optional[int]:
        with self._lock:
            return self._settings_mtime_ns

    def is_configured(self) -> bool:
        with self._lock:
            return self._configured

    def tool_timeout_sec(self) -> int:
        with self._lock:
            return self._tool_timeout_sec

    def enabled_servers_without_tools(self) -> List[Dict[str, str]]:
        """Only complete empty listings whose latest attempt succeeded in full."""
        with self._lock:
            if not self._enabled:
                return []
            return [{"id": rt.config.id, "last_error": ""} for rt in self._servers.values()
                    if rt.config.enabled and rt.catalog and rt.catalog["complete"] and not rt.catalog["items"]
                    and rt.catalog["latest_attempt"].get("ok") is True
                    and not rt.catalog["latest_attempt"].get("partial")]

    def list_tools_for_registry(self) -> List[Dict[str, Any]]:
        """Return enabled MCP tools in ToolRegistry shape."""
        with self._lock:
            if not self._enabled:
                return []
            results: List[Dict[str, Any]] = []
            for runtime in self._servers.values():
                cfg = runtime.config
                if not cfg.enabled:
                    continue
                allowed = set() if mode_has_unrestricted_agency(get_runtime_mode()) else set(cfg.allowed_tools)
                for tool in runtime.tools:
                    if allowed and tool.raw_name not in allowed:
                        continue
                    results.append(
                        {
                            "name": tool.prefixed_name,
                            "description": _model_facing_description(tool),
                            "schema": tool.schema,
                            "server_id": tool.server_id,
                            "raw_name": tool.raw_name,
                            "raw_description": _redact_error_text(tool.description, cfg),
                            "metadata": tool.metadata,
                        }
                    )
            return results

    def tool_name_collisions(self) -> List[Dict[str, str]]:
        """Return provider-name collisions omitted by first-wins normalization."""

        with self._lock:
            if not self._enabled:
                return []
            return [
                dict(item, server_id=runtime.config.id)
                for runtime in self._servers.values()
                if runtime.config.enabled
                for item in runtime.tool_name_collisions
                if (
                    mode_has_unrestricted_agency(get_runtime_mode())
                    or not runtime.config.allowed_tools
                    or item.get("kept_raw_name") in runtime.config.allowed_tools
                    or item.get("dropped_raw_name") in runtime.config.allowed_tools
                )
            ]

    def get_tool(self, prefixed_name: str) -> Optional[Dict[str, Any]]:
        for tool in self.list_tools_for_registry():
            if tool["name"] == prefixed_name:
                return tool
        return None

    def status_payload(self) -> Dict[str, Any]:
        """Return a redacted status snapshot for ``/api/mcp/status``."""
        with self._lock:
            servers: List[Dict[str, Any]] = [dict(item) for item in self._configuration_errors]
            for runtime in self._servers.values():
                cfg = runtime.config
                servers.append(
                    {
                        "id": cfg.id,
                        "name": cfg.name,
                        "enabled": cfg.enabled,
                        "transport": cfg.transport,
                        "url": _redact_error_text(cfg.url, cfg),
                        "auth_header": cfg.auth_header,
                        "auth_configured": cfg.has_auth(),
                        "header_names": list(cfg.headers),
                        "allowed_tools": list(cfg.allowed_tools),
                        "cwd": cfg.cwd,
                        "env_from_settings": dict(cfg.env_from_settings),
                        "configuration_warnings": list(cfg.configuration_warnings),
                        "tool_count": len(runtime.tools),
                        "tools": [
                            {
                                "name": tool.raw_name,
                                "prefixed_name": tool.prefixed_name,
                                "description": _redact_error_text(tool.description, cfg),
                                **({"metadata": tool.metadata} if tool.metadata else {}),
                            }
                            for tool in runtime.tools
                        ],
                        "tool_name_collisions": [
                            dict(item) for item in runtime.tool_name_collisions
                        ],
                        "last_error": runtime.last_error,
                        "last_refreshed": runtime.last_refreshed,
                        "last_attempted": runtime.last_attempted,
                        "listed_at": runtime.last_refreshed or None,
                        "latest_attempt": (runtime.catalog or {}).get("latest_attempt"),
                        "complete": bool(runtime.catalog and runtime.catalog["complete"]),
                        "publication_error": runtime.publication_error,
                    }
                )
            return {
                "enabled": self._enabled,
                "sdk_available": _MCP_SDK_AVAILABLE,
                "sdk_error": _MCP_SDK_IMPORT_ERROR or "",
                "tool_timeout_sec": self._tool_timeout_sec,
                "servers": servers,
            }

    def refresh_server(self, server_id: str) -> Dict[str, Any]:
        """Server-owned listing; serialize per id without a lock over network I/O."""
        with self._lock:
            serial = self._listing_locks.setdefault(server_id, threading.Lock())
        with serial:
            return self._refresh_server(server_id)

    def _refresh_server(self, server_id: str) -> Dict[str, Any]:
        with self._lock:
            if not self._enabled:
                return {"ok": False, "error": "MCP client is disabled."}
            runtime = self._servers.get(server_id)
            if runtime is None:
                for error in self._configuration_errors:
                    if error["id"] == server_id:
                        return {"ok": False, "code": error["code"], "error": error["last_error"]}
                return {"ok": False, "error": f"unknown server id: {server_id!r}"}
            cfg, timeout = runtime.config, self._tool_timeout_sec
            if not cfg.enabled:
                return {"ok": False, "error": f"MCP server {server_id!r} is disabled."}
        from ouroboros.owner_pause import OwnerPauseRefused

        items, complete, error = None, False, ""
        schema_omissions = []
        try:
            raw = _run_async(lambda: self._async_list_tools(cfg, timeout), join_timeout=timeout + 3)
            raw, schema_omissions = filter_schema_literals(raw, lambda text: _redact_error_text(text, cfg))
            items, complete = mcp_catalog.sanitize_items(raw, lambda text: _redact_error_text(text, cfg))
        except OwnerPauseRefused as exc:
            return {"ok": False, "not_started": True, "error": f"MCP listing NOT STARTED: {exc}"}
        except BaseException as exc:  # noqa: BLE001 - retain last-known tools on every failure
            error = f"{type(exc).__name__}: {_redact_error_text(exc, cfg)}"
        at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            target = self._servers.get(server_id)
            if target is not runtime or mcp_catalog.identity(target.config) != mcp_catalog.identity(cfg):
                return {"ok": False, "published": False, "error": f"stale MCP refresh discarded for server {server_id!r}"}
            envelope = mcp_catalog.attempt_envelope(cfg, target.catalog, at=at, error=error,
                                                   items=items, complete=complete, schema_omissions=schema_omissions)
            target.published_before_failure = self._catalog_reader.load(cfg)
            try:
                mcp_catalog.publish(cfg, envelope)
                loaded = self._catalog_reader.load(cfg)
                if loaded is None:
                    raise OSError("published catalog could not be loaded")
                envelope = loaded
                target.publication_error = ""
            except Exception as exc:
                target.publication_error = f"{type(exc).__name__}: {_redact_error_text(exc, cfg)}"
                log.warning("MCP catalog publication failed for %s: %s", cfg.id, target.publication_error)
            self._adopt_catalog(target, envelope)
            return {"ok": items is not None, "published": not bool(target.publication_error),
                    "error": error, "publication_error": target.publication_error,
                    "server_id": cfg.id, "tool_count": len(target.tools),
                    "configuration_warnings": list(cfg.configuration_warnings),
                    "tool_name_collisions": list(target.tool_name_collisions),
                    "tools": [{"name": tool.raw_name, "prefixed_name": tool.prefixed_name,
                               "description": tool.description} for tool in target.tools]}

    def refresh_all(self) -> Dict[str, Any]:
        """Server-owned refresh of every enabled server, publishing each on completion."""
        with self._lock:
            if not self._enabled:
                return {"refreshed": {}, "error": "MCP client is disabled."}
            ids = [key for key, rt in self._servers.items() if rt.config.enabled]
        return {"refreshed": {key: self.refresh_server(key) for key in ids}}

    def refresh_all_background(self, *, reason: str = "settings") -> None:
        """``refresh_all`` on a daemon thread, one at a time; with nothing to list, none starts."""
        with self._lock:
            if not self._enabled or not any(rt.config.enabled for rt in self._servers.values()):
                return
            if self._refresh_running:
                self._refresh_pending = True
                return
            self._refresh_running = True

        def _runner() -> None:
            while True:
                try:
                    log.info("Refreshing MCP tools in background (%s)", reason)
                    self.refresh_all()
                except Exception:
                    log.warning("Background MCP refresh failed", exc_info=True)
                with self._lock:
                    if not self._refresh_pending:
                        self._refresh_running = False
                        return
                    self._refresh_pending = False

        threading.Thread(target=_runner, name=f"mcp-refresh-{reason}", daemon=True).start()

    def test_server(self, raw_config: Dict[str, Any], *, settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Probe a candidate config without persisting it."""
        errors: List[str] = []
        cfg = normalize_server_config(raw_config, settings=settings, errors=errors)
        if cfg is None:
            return {
                "ok": False,
                "code": "MCP_CONFIG_ERROR",
                "error": "Invalid MCP server config: " + "; ".join(errors),
            }
        timeout = self._tool_timeout_sec
        try:
            tools_raw = _run_async(lambda: self._async_list_tools(cfg, timeout), join_timeout=timeout + 3)
        except BaseException as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {_redact_error_text(exc, cfg)}"}
        truncated = any(
            isinstance(t, dict) and t.get("_pagination_truncated") for t in tools_raw
        )
        tools_raw = [
            t for t in tools_raw
            if not (isinstance(t, dict) and t.get("_pagination_truncated"))
        ]
        tools_raw, literal_omissions = filter_schema_literals(tools_raw, lambda text: _redact_error_text(text, cfg))
        return {
            "ok": True,
            "server_id": cfg.id,
            "tool_count": len(tools_raw),
            "configuration_warnings": list(cfg.configuration_warnings),
            **({"schema_omissions": literal_omissions} if literal_omissions else {}),
            **({"pagination_truncated": True} if truncated else {}),
            "tools": [
                {
                    "name": str(t.get("name") or ""),
                    "description": _redact_error_text(str(t.get("description") or ""), cfg),
                    "metadata": redact_metadata(t.get("metadata") or {}, lambda text: _redact_error_text(text, cfg)),
                }
                for t in tools_raw
            ],
        }

    def _resolve_locked(self, prefixed_name: str) -> tuple[MCPNameResolution, Any, Any]:
        """Exact catalog lookup (caller holds ``self._lock``): (fact, config, tool)."""
        if not self._enabled:
            return MCPNameResolution("mcp_disabled"), None, None
        parsed = parse_tool_name(prefixed_name)
        server = parsed["server_slug"] if parsed else ""
        runtime = self._servers.get(server)
        if runtime is None:
            broken = next((item for item in self._configuration_errors
                           if server and item["id"] == server and item["enabled"]), None)
            if broken is not None:
                detail = f"its configuration is invalid: {broken['last_error']}"
                return MCPNameResolution("catalog_unavailable", server, detail=detail), None, None
            return MCPNameResolution("not_found"), None, None
        cfg = runtime.config
        if not cfg.enabled:
            return MCPNameResolution("server_disabled", cfg.id), cfg, None
        allowed = set() if mode_has_unrestricted_agency(get_runtime_mode()) else set(cfg.allowed_tools)
        for tool in runtime.tools:
            if tool.prefixed_name == prefixed_name:
                status = "disallowed" if allowed and tool.raw_name not in allowed else "callable"
                return MCPNameResolution(status, cfg.id, tool.raw_name), cfg, tool
        envelope = runtime.catalog or {}
        attempt = envelope.get("latest_attempt") or {}
        if not envelope.get("complete") or attempt.get("ok") is False or attempt.get("partial"):
            return MCPNameResolution("catalog_unavailable", cfg.id,
                                     detail=mcp_catalog.catalog_notice(runtime)), cfg, None
        return MCPNameResolution("not_found", cfg.id,
                                 detail=f"absent from the listing of {runtime.last_refreshed}"), cfg, None

    def resolve_tool_name(self, prefixed_name: str) -> MCPNameResolution:
        """The current catalog's fact for one name; performs no transport or refresh."""
        with self._lock:
            return self._resolve_locked(prefixed_name)[0]

    def _call_tool_result(
        self, prefixed_name: str, arguments: Dict[str, Any]
    ) -> ToolResult:
        """Invoke one MCP tool while retaining host-attested provider facts."""
        with self._lock:
            resolution, cfg, tool = self._resolve_locked(prefixed_name)
            timeout = self._tool_timeout_sec
        if resolution.status != "callable":
            return resolution.refusal(prefixed_name)
        from ouroboros.tool_custody import external_handoff
        confirmed = external_handoff("mcp_transport")
        try:
            result = _run_async(
                lambda: self._async_call_tool(cfg, tool.raw_name, arguments or {}, timeout),
                join_timeout=timeout + 3,
            )
            if not isinstance(result, ToolResult):
                raise TypeError("MCP transport returned a non-ToolResult outcome")
        except asyncio.TimeoutError as exc:
            unsent = getattr(exc, "mcp_unsent_joined", False) is True
            if unsent:
                confirmed()
            text = (
                f"⚠️ MCP_TOOL_TIMEOUT: server {cfg.id!r} did not respond in {timeout}s. "
                + ("The tool call was not submitted; its session and transport have closed."
                   if unsent else "The remote outcome is unknown: side effects may already have happened, "
                   "and remote cancellation is not confirmed. Use the server-specific status/read "
                   "tool, if available, to reconcile the operation before retrying.")
            )
            return ToolResult(status="timeout", code="MCP_TIMEOUT", text=text,
                              meta={"mcp_call_not_sent": unsent})
        except BaseException as exc:  # noqa: BLE001 - any failure is reported
            from ouroboros.owner_pause import OwnerPauseRefused, _TOOL_OPERATION
            from ouroboros.store import StoreError
            active = _TOOL_OPERATION.get()
            unsent = getattr(exc, "mcp_unsent_joined", False) is True or (
                isinstance(exc, OwnerPauseRefused) and active and active[2].get("not_started") is True)
            if unsent:
                confirmed()  # no call, with either no transport entry or joined cleanup
            if unsent and isinstance(exc, (OwnerPauseRefused, StoreError)):
                from ouroboros.tools.tool_result import launch_refusal_result, _replace_tool_result
                return _replace_tool_result(launch_refusal_result(exc),
                                            meta_updates={"mcp_call_not_sent": True})
            body = f"⚠️ MCP_TOOL_ERROR: {type(exc).__name__}: {_redact_error_text(exc, cfg)}"
            text = _model_facing_result(cfg, tool.raw_name, body)
            return ToolResult(
                status="error",
                code="MCP_ERROR",
                text=text,
                meta={"dynamic_provider": True, "mcp_call_not_sent": bool(unsent)},
            )
        # Joined normal return (either SDK error bit): this invocation's call is over.
        from ouroboros.owner_pause import record_mcp_call_returned

        record_mcp_call_returned(prefixed_name)
        confirmed()
        text = _model_facing_result(
            cfg,
            tool.raw_name,
            _redact_error_text(result.text, cfg),
        )
        return ToolResult(
            status=result.status,
            code=result.code,
            text=text,
            meta={**dict(result.meta), "dynamic_provider": True},
        )

    def call_tool(self, prefixed_name: str, arguments: Dict[str, Any]) -> str:
        """Synchronously invoke an MCP tool and return its text projection."""
        return self._call_tool_result(prefixed_name, arguments).text


_manager_lock = threading.Lock()
_manager: Optional[MCPManager] = None


def get_manager() -> MCPManager:
    """Return the process-global manager."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = MCPManager()
        return _manager


def reset_manager_for_tests() -> None:
    """Drop the module-level singleton for tests."""
    global _manager
    with _manager_lock:
        _manager = None


def reconfigure_from_settings(settings: Dict[str, Any]) -> None:
    """Reconfigure the global manager from settings."""
    get_manager().reconfigure(settings)


def ensure_configured_from_settings() -> None:
    """Adopt catalogs on reads; server settings-mtime changes also run configuration_changed."""
    from ouroboros.config import SETTINGS_PATH, load_settings

    manager = get_manager()
    manager.reload_catalogs()
    try:
        mtime_ns = SETTINGS_PATH.stat().st_mtime_ns
    except OSError:
        mtime_ns = None
    if not manager.is_configured() or manager.settings_mtime_ns() != mtime_ns or mtime_ns is None:
        settings = load_settings()
        manager.reconfigure(settings, settings_mtime_ns=mtime_ns)
        if manager.is_lister and manager._listing_fingerprint != manager._settings_fingerprint:
            manager.configuration_changed(settings, reason="settings-file")


def configuration_changed(settings: dict, *, reason: str = "settings") -> None:
    """Called only by server-owned boot and Settings-save handlers."""
    get_manager().configuration_changed(settings, reason=reason)


def refresh_all_background(*, reason: str = "settings") -> None:
    get_manager().refresh_all_background(reason=reason)


def call_mcp_tool(name: str, arguments: Dict[str, Any]) -> str:
    """ToolRegistry sync call helper."""
    return get_manager().call_tool(name, arguments or {})


def _call_mcp_tool_result(name: str, arguments: Dict[str, Any]) -> ToolResult:
    """Internal typed ToolRegistry call helper."""
    return get_manager()._call_tool_result(name, arguments or {})
