"""Adapter-owned CyberGym sidecar contracts.

The launcher owns execution and waiting.  This module is intentionally pure:
it validates the daemon/network boundary, builds Docker argv, and checks
sanitised observations supplied by an injected runner.  There is no Docker
SDK or network client import, so these rules remain testable on CI hosts that
do not have Docker installed.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

from devtools.benchmarks.cybergym.cybergym_observations import (
    _CONNECTIVITY_EXPECTATIONS,  # noqa: F401
    _DIGEST,  # noqa: F401
    _PROTECTED_DENIAL_STATUS_RANGE,  # noqa: F401
    _PROTECTED_ROUTE_KEY,  # noqa: F401
    SCHEMA_VERSION,  # noqa: F401
    _bindings,  # noqa: F401
    _bool_mapping_value,  # noqa: F401
    _digests,  # noqa: F401
    _http_status,  # noqa: F401
    _id,  # noqa: F401
    _labels_from,  # noqa: F401
    _mapping_value,  # noqa: F401
    _mounts,  # noqa: F401
    _name,  # noqa: F401
    _nested,  # noqa: F401
    _network,  # noqa: F401
    _network_mode,  # noqa: F401
    _pid,  # noqa: F401
    _probe_value,  # noqa: F401
    _protected_route_record,  # noqa: F401
    _protected_route_summary,  # noqa: F401
    _running,  # noqa: F401
    evaluate_connectivity_checks,  # noqa: F401
)

NETWORK_NAME = "cybergym-internal"
DOCKER_HOST_ENV = "DOCKER_HOST"
LOOPBACK_HOST = "127.0.0.1"
DEFAULT_SOCKET_TARGET = "/var/run/docker.sock"
API_KEY_ENV = "CYBERGYM_API_KEY"
EXECUTOR_NETWORK_DECLARATION = "host"

_FORBIDDEN_NETWORKS = frozenset({"", "none", "host", "bridge", "default", "docker0"})
_WILDCARD_HOSTS = frozenset({"", "*", "0.0.0.0", "::", "::0", "[::]"})
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_DNS = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")

# The upstream CyberGym distribution documents a public example key.  Keep
# only its non-reversible digest prefix here; the key itself must never enter
# source, logs, or an attestation artifact.
_PUBLIC_DEFAULT_API_KEY_FINGERPRINT = "9605ed570966a4e0"
_PROTECTED_ENV_NAMES = frozenset(
    {
        DOCKER_HOST_ENV,
        API_KEY_ENV,
        "CYBERGYM_SERVER_URL",
        "CYBERGYM_TASK_ID",
        "CYBERGYM_AGENT_ID",
        "NO_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    }
)


class SidecarConfigurationError(ValueError):
    """Raised when an input would weaken the adapter-owned boundary."""


class SidecarAttestationError(SidecarConfigurationError):
    """Raised by the strict attestation entry point."""

    def __init__(self, message: str, report: Mapping[str, Any]):
        super().__init__(message)
        self.report = dict(report)


def _text(value: Any, name: str, *, max_len: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > max_len:
        raise SidecarConfigurationError(f"{name} must be a non-empty string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value) or any(char.isspace() for char in value):
        raise SidecarConfigurationError(f"unsafe {name}")
    return value


def _safe_id(value: Any, name: str) -> str:
    value = _text(value, name)
    if value in {".", ".."} or not _SAFE_ID.fullmatch(value):
        raise SidecarConfigurationError(f"unsafe {name}")
    return value


def _safe_name(value: Any, name: str) -> str:
    value = _text(value, name)
    if not _SAFE_NAME.fullmatch(value):
        raise SidecarConfigurationError(f"unsafe {name}")
    return value


def _safe_path(value: Any, name: str, *, absolute: bool = True) -> str:
    value = _text(value, name, max_len=4096)
    if "\x00" in value or "*" in value or "?" in value or (absolute and not value.startswith("/")):
        raise SidecarConfigurationError(f"unsafe {name}")
    if value == "/" or "/../" in f"/{value.strip('/')}/" or value.endswith("/.."):
        raise SidecarConfigurationError(f"unsafe {name}")
    return value


def _port(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise SidecarConfigurationError(f"{name} must be a TCP port")
    return value


def _dns_label(value: Any, name: str) -> str:
    value = _text(value, name, max_len=63).lower()
    if not _SAFE_DNS.fullmatch(value):
        raise SidecarConfigurationError(f"unsafe DNS label for {name}")
    return value


def _loopback(value: Any, name: str = "bind_host") -> str:
    value = _text(value, name)
    if value in _WILDCARD_HOSTS:
        raise SidecarConfigurationError(f"{name} must not be wildcard")
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise SidecarConfigurationError(f"{name} must be numeric loopback") from exc
    if not address.is_loopback:
        raise SidecarConfigurationError(f"{name} must be loopback")
    return value


def _recognised_rootless_socket(path: str) -> bool:
    if path in {"/var/run/docker.sock", "/run/docker.sock", "/docker.sock"}:
        return False
    if re.fullmatch(r"/run/user/[0-9]+/(?:docker|docker-rootless)(?:-[A-Za-z0-9_.-]+)?\.sock", path):
        return True
    return "/rootless/" in path or (path.startswith("/mnt/data/") and "/docker" in path)


@dataclass(frozen=True)
class DockerHostRef:
    """Explicit rootless daemon endpoint, distinct from Docker network mode."""

    value: str
    socket_path: str
    allow_custom: bool = False

    def __post_init__(self) -> None:
        value = _text(self.value, "docker_host", max_len=4096)
        path = _safe_path(self.socket_path, "docker_socket")
        if path in {"/var/run/docker.sock", "/run/docker.sock", "/docker.sock"}:
            raise SidecarConfigurationError("the shared/rootful Docker socket is forbidden")
        if value != f"unix://{path}" or (not self.allow_custom and not _recognised_rootless_socket(path)):
            raise SidecarConfigurationError("docker_host must be a recognised rootless unix socket")

    @property
    def env(self) -> Mapping[str, str]:
        return {DOCKER_HOST_ENV: self.value}

    def __str__(self) -> str:
        return self.value


def resolve_rootless_docker_host(
    value: str | DockerHostRef | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    allow_custom: bool = False,
) -> DockerHostRef:
    """Resolve explicit ``DOCKER_HOST``; reject TCP, rootful and default endpoints."""

    if isinstance(value, DockerHostRef):
        return value
    source = os.environ if environ is None else environ
    raw = value if value is not None else source.get(DOCKER_HOST_ENV)
    if not isinstance(raw, str) or not raw:
        raise SidecarConfigurationError("DOCKER_HOST must be explicitly set")
    if any(char.isspace() for char in raw) or any(char in raw for char in "*?\x00"):
        raise SidecarConfigurationError("unsafe DOCKER_HOST")
    if raw.startswith("/"):
        path, canonical = raw, f"unix://{raw}"
    else:
        parsed = urlsplit(raw)
        if parsed.scheme != "unix" or parsed.netloc or parsed.query or parsed.fragment:
            raise SidecarConfigurationError("DOCKER_HOST must be a unix socket")
        path, canonical = parsed.path, f"unix://{parsed.path}"
    path = _safe_path(path, "docker_socket")
    if path in {"/var/run/docker.sock", "/run/docker.sock", "/docker.sock"}:
        raise SidecarConfigurationError("the shared/rootful Docker socket is forbidden")
    if not allow_custom and not _recognised_rootless_socket(path):
        raise SidecarConfigurationError("DOCKER_HOST is not recognisably rootless")
    return DockerHostRef(canonical, path, allow_custom=allow_custom)


def require_explicit_rootless_docker_host(environ: Mapping[str, str] | None = None) -> DockerHostRef:
    return resolve_rootless_docker_host(environ=environ)


def docker_host_environment(host: str | DockerHostRef) -> dict[str, str]:
    """Environment shared by launcher/sidecar callbacks for daemon selection."""

    return dict(resolve_rootless_docker_host(host).env)


def _dns_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "item"


def make_opaque_agent_id(campaign_id: str, task_id: str, attempt_id: str = "") -> str:
    """Derive an attempt-scoped agent id without exposing the task identity.

    Reattaching the same attempt keeps the same id; a retry supplies a new
    persisted ``attempt_id`` and therefore cannot collide with old PoC records.
    The optional argument preserves the stable two-argument helper contract for
    callers that only need a deterministic display id.
    """

    campaign = _safe_id(campaign_id, "campaign_id")
    task = _safe_id(task_id, "task_id")
    attempt = ""
    if attempt_id:
        attempt = _safe_id(attempt_id, "attempt_id")
    digest = hashlib.sha256(f"cybergym-agent\0{campaign}\0{task}\0{attempt}".encode()).hexdigest()[:24]
    return f"agent-{digest}"


def make_dns_alias(campaign_id: str, role: str, task_id: str = "") -> str:
    campaign = _safe_id(campaign_id, "campaign_id")
    role = _safe_id(role, "role").lower()
    if role == "workspace" and task_id:
        task_component = make_opaque_agent_id(campaign, task_id)
    else:
        task_component = "campaign"
    digest = hashlib.sha256(f"{campaign}\0{role}\0{task_component}".encode()).hexdigest()[:10]
    prefix = f"cybergym-{_dns_slug(role)}-{_dns_slug(campaign)}-{_dns_slug(task_component)}"
    return _dns_label((prefix[:52].rstrip("-") + "-" + digest)[:63], "dns_alias")


@dataclass(frozen=True)
class NetworkPlan:
    campaign_id: str
    task_id: str
    server_port: int
    verifier_host_port: int
    network_name: str = NETWORK_NAME
    server_alias: str = ""
    workspace_alias: str = ""
    server_container_port: int | None = None
    verifier_bind_host: str = LOOPBACK_HOST
    opaque_agent_id: str = ""

    def __post_init__(self) -> None:
        campaign = _safe_id(self.campaign_id, "campaign_id")
        task = _safe_id(self.task_id, "task_id")
        if self.network_name != NETWORK_NAME:
            raise SidecarConfigurationError(f"network must be {NETWORK_NAME}")
        _port(self.server_port, "server_port")
        _port(self.verifier_host_port, "verifier_host_port")
        container_port = self.server_port if self.server_container_port is None else self.server_container_port
        _port(container_port, "server_container_port")
        object.__setattr__(self, "campaign_id", campaign)
        object.__setattr__(self, "task_id", task)
        object.__setattr__(self, "server_container_port", container_port)
        object.__setattr__(self, "verifier_bind_host", _loopback(self.verifier_bind_host, "verifier_bind_host"))
        _reject_task_exposure((campaign,), self)
        agent_id = self.opaque_agent_id or make_opaque_agent_id(campaign, task)
        if not isinstance(agent_id, str) or not re.fullmatch(r"agent-[0-9a-f]{24}", agent_id):
            raise SidecarConfigurationError("opaque_agent_id must be an agent-<24 hex> value")
        server_alias = self.server_alias or make_dns_alias(campaign, "server")
        workspace_alias = self.workspace_alias or f"cybergym-workspace-{agent_id}"
        if self.workspace_alias:
            _reject_task_exposure((workspace_alias,), self)
        # The server alias is injected into the workspace URL/NO_PROXY too.
        # Reject a campaign spelling that would smuggle the raw task id into
        # either DNS name, including when a caller supplied a custom alias.
        _reject_task_exposure((server_alias, workspace_alias), self)
        object.__setattr__(self, "server_alias", _dns_label(server_alias, "server_alias"))
        object.__setattr__(self, "workspace_alias", _dns_label(workspace_alias, "workspace_alias"))
        object.__setattr__(self, "opaque_agent_id", agent_id)
        if self.server_alias == self.workspace_alias:
            raise SidecarConfigurationError("server and workspace aliases must differ")

    @property
    def server_url(self) -> str:
        return f"http://{self.server_alias}:{self.server_port}"

    @property
    def verifier_url(self) -> str:
        return f"http://{self.verifier_bind_host}:{self.verifier_host_port}"

    @property
    def no_proxy(self) -> str:
        return build_no_proxy(self.server_alias, self.server_port)


def build_network_plan(
    campaign_id: str,
    task_id: str,
    server_port: int,
    verifier_host_port: int,
    *,
    server_alias: str | None = None,
    workspace_alias: str | None = None,
    verifier_bind_host: str = LOOPBACK_HOST,
    server_container_port: int | None = None,
    opaque_agent_id: str | None = None,
) -> NetworkPlan:
    return NetworkPlan(
        campaign_id,
        task_id,
        server_port,
        verifier_host_port,
        server_alias=server_alias or "",
        workspace_alias=workspace_alias or "",
        verifier_bind_host=verifier_bind_host,
        server_container_port=server_container_port,
        opaque_agent_id=opaque_agent_id or "",
    )


def _no_proxy_token(value: str) -> str:
    value = _text(value, "NO_PROXY entry", max_len=256)
    if value in _WILDCARD_HOSTS or value.startswith("*") or "://" in value or "=" in value or "," in value:
        raise SidecarConfigurationError("wildcard/invalid NO_PROXY entry")
    return value


def build_no_proxy(alias: str, port: int | None = None, *, existing: Iterable[str] | str = ()) -> str:
    alias = _dns_label(alias, "server_alias")
    if port is not None:
        _port(port, "server_port")
    old = existing.split(",") if isinstance(existing, str) else list(existing)
    values: list[str] = []
    for item in [*old, alias, f"{alias}:{port}" if port is not None else None]:
        if item is not None and item != "":
            item = _no_proxy_token(item)
            if item not in values:
                values.append(item)
    return ",".join(values)


def build_private_route(plan: NetworkPlan, endpoint: str, *, audience: str) -> str:
    endpoint = _text(endpoint, "endpoint", max_len=512)
    if not endpoint.startswith("/") or "//" in endpoint or ".." in endpoint or "?" in endpoint or "#" in endpoint:
        raise SidecarConfigurationError("endpoint must be a relative absolute path")
    audience = _safe_id(audience, "audience").lower()
    protected = {"/query-poc", "/submit-fix", "/fix", "/protected/query", "/protected/fix"}
    if audience == "agent":
        endpoint_value = endpoint.rstrip("/")
        if any(endpoint_value == item or endpoint_value.startswith(item + "/") for item in protected):
            raise SidecarConfigurationError("agent cannot receive protected verifier route")
        return f"{plan.server_url}{endpoint}"
    if audience in {"verifier", "host-verifier", "sidecar-verifier"}:
        return f"{plan.verifier_url}{endpoint}"
    raise SidecarConfigurationError("unknown route audience")


def build_connectivity_probe_plan(plan: NetworkPlan) -> tuple[dict[str, Any], ...]:
    return (
        # FastAPI's documented route is stable across the upstream server
        # versions used by this adapter.
        {"name": "agent_to_server", "target": f"{plan.server_url}/docs", "expected_reachable": True},
        {
            "name": "verifier_to_private",
            "targets": (f"{plan.verifier_url}/query-poc", f"{plan.verifier_url}/submit-fix"),
            "expected_reachable": True,
            "requires_all": True,
        },
        {"name": "agent_to_public", "target": "https://example.com/", "expected_reachable": True},
        {
            "name": "agent_to_verifier",
            "targets": (f"{plan.verifier_url}/query-poc", f"{plan.verifier_url}/submit-fix"),
            "expected_reachable": False,
            "requires_all": True,
        },
        {
            # The agent shares the server alias, but must not be able to use
            # the verifier routes without authentication.  GET is intentional:
            # it exercises transport and the denial response without creating
            # a PoC or changing verifier state.
            "name": "agent_to_server_protected",
            "targets": (f"{plan.server_url}/query-poc", f"{plan.server_url}/submit-fix"),
            "method": "POST",
            "authentication": "none",
            "expected_reachable": True,
            "expected_authorized": False,
            "expected_denied": True,
            "expected_statuses": [401, 403, 404],
            "expected_mutating": False,
            "side_effect_free": True,
            "requires_all": True,
        },
        {"name": "agent_socket_visible", "target": "unix://docker.sock", "expected_visible": False},
    )


def _is_rootless_security_option(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    option = value.strip().lower()
    return option in {"rootless", "name=rootless"} or option.endswith("=rootless")


def _validate_image(value: str) -> str:
    value = _text(value, "image", max_len=1024)
    if value.lower() in {"latest", "placeholder", "changeme", "example", "example/image"} or "${" in value:
        raise SidecarConfigurationError("placeholder image reference")
    if any(char in value for char in ";,\n\r"):
        raise SidecarConfigurationError("unsafe image reference")
    return value


def _digest(value: str | None) -> str | None:
    if value is not None and (not isinstance(value, str) or not _DIGEST.fullmatch(value)):
        raise SidecarConfigurationError("image digest must be sha256:<64 hex>")
    return value


def _required_digest(value: str | None, name: str = "image_digest") -> str:
    value = _digest(value)
    if value is None:
        raise SidecarConfigurationError(f"{name} is required for production custody")
    return value


def _env_name(value: str) -> str:
    value = _text(value, "environment name")
    if not _ENV_NAME.fullmatch(value):
        raise SidecarConfigurationError("unsafe environment name")
    return value


def _safe_env_items(values: Mapping[str, str]) -> None:
    if not isinstance(values, Mapping):
        raise SidecarConfigurationError("extra_env must be a mapping")
    for key, value in values.items():
        key = _env_name(key)
        _text(value, f"environment value for {key}", max_len=4096)
        upper = key.upper()
        if upper in _PROTECTED_ENV_NAMES:
            raise SidecarConfigurationError(f"protected control environment cannot be overridden: {key}")
        if any(marker in upper for marker in ("SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")) or upper.endswith("_API_KEY"):
            raise SidecarConfigurationError(f"secret-bearing environment must be injected by name only: {key}")


def _container_docker_host(value: str, socket_target: str) -> str:
    """Validate the in-container Docker endpoint used by the server only.

    The host-side daemon remains the explicitly selected rootless socket.  A
    server may opt into ``DOCKER_HOST`` only when it points at the socket that
    this command mounts into the container; arbitrary TCP/rootful endpoints
    would let a caller bypass that custody boundary.
    """

    value = _text(value, "container_docker_host", max_len=4096)
    expected = f"unix://{socket_target}"
    if value != expected:
        raise SidecarConfigurationError("container DOCKER_HOST must reference the mounted socket")
    return value


def _reject_task_exposure(values: Iterable[str], plan: NetworkPlan) -> None:
    task = plan.task_id.lower()
    slug = _dns_slug(task)

    def _contains_token(value: str, token: str) -> bool:
        if not token:
            return False
        pattern = rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])"
        return re.search(pattern, value) is not None

    for value in values:
        if not isinstance(value, str):
            raise SidecarConfigurationError("workspace command/environment must contain strings")
        lowered = value.lower()
        if _contains_token(lowered, task) or _contains_token(lowered, slug):
            raise SidecarConfigurationError("workspace command/environment must use opaque_agent_id, not task_id")


def _mount_path(value: str, name: str) -> str:
    value = _safe_path(value, name)
    if "," in value:
        raise SidecarConfigurationError(f"{name} cannot contain comma")
    return value


def _labels(plan: NetworkPlan, role: str, custom: Mapping[str, str] | None = None) -> dict[str, str]:
    role = _safe_id(role, "role").lower()
    if role not in {"server", "workspace"}:
        raise SidecarConfigurationError("unsupported sidecar role")
    task_label = "campaign" if role == "server" else plan.opaque_agent_id
    agent_label = "campaign" if role == "server" else plan.opaque_agent_id
    result = {
        "com.ouroboros.benchmark": "cybergym",
        "com.ouroboros.run": plan.campaign_id,
        "com.ouroboros.campaign": plan.campaign_id,
        "com.ouroboros.task": task_label,
        "com.ouroboros.agent_id": agent_label,
        "com.ouroboros.role": role,
        "com.ouroboros.network": plan.network_name,
        "com.ouroboros.owner": "ouroboros",
    }
    for key, value in (custom or {}).items():
        key, value = _text(key, "label key", max_len=256), _text(value, "label value", max_len=512)
        if key in result and result[key] != value:
            raise SidecarConfigurationError(f"custom label attempts to override {key}")
        result[key] = value
    return result


def required_resource_labels(plan: NetworkPlan, role: str) -> dict[str, str]:
    return _labels(plan, role)


def _append_labels(argv: list[str], labels: Mapping[str, str]) -> None:
    for key in sorted(labels):
        argv.extend(("--label", f"{key}={labels[key]}"))


def _host(value: str | DockerHostRef) -> DockerHostRef:
    return resolve_rootless_docker_host(value)


@dataclass(frozen=True)
class SidecarCommandSpec:
    docker_host: str | DockerHostRef
    plan: NetworkPlan
    image: str
    container_name: str
    command: tuple[str, ...] = ()
    api_key_env: str = API_KEY_ENV
    socket_target: str = DEFAULT_SOCKET_TARGET
    data_host_path: str | None = None
    data_container_path: str = "/cybergym-data"
    image_digest: str | None = None
    labels: Mapping[str, str] = field(default_factory=dict)
    extra_env: Mapping[str, str] = field(default_factory=dict)
    read_only_mounts: Mapping[str, str] = field(default_factory=dict)
    container_docker_host: str | None = None
    platform: str = "linux/amd64"
    # Production callers keep the sidecar private and use the server
    # container's immutable-id exec channel; the default remains True for the
    # legacy pure argv contract.
    publish_host_port: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "docker_host", _host(self.docker_host))
        _validate_image(self.image)
        _safe_name(self.container_name, "container_name")
        api_key_env = _env_name(self.api_key_env)
        if api_key_env.upper() in _PROTECTED_ENV_NAMES and api_key_env != API_KEY_ENV:
            raise SidecarConfigurationError("api_key_env collides with a protected control environment")
        object.__setattr__(self, "api_key_env", api_key_env)
        socket_target = _mount_path(self.socket_target, "socket_target")
        object.__setattr__(self, "socket_target", socket_target)
        data_container_path = _mount_path(self.data_container_path, "data_container_path")
        object.__setattr__(self, "data_container_path", data_container_path)
        if self.data_host_path is not None:
            data_host_path = _mount_path(self.data_host_path, "data_host_path")
            object.__setattr__(self, "data_host_path", data_host_path)
            if data_host_path != data_container_path:
                raise SidecarConfigurationError(
                    "server data mount must preserve the same absolute server-root path"
                )
        if self.container_docker_host is not None:
            _container_docker_host(self.container_docker_host, socket_target)
        _digest(self.image_digest)
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", self.platform):
            raise SidecarConfigurationError("unsafe platform")
        if not isinstance(self.publish_host_port, bool):
            raise SidecarConfigurationError("publish_host_port must be a boolean")
        for item in self.command:
            _text(item, "command argument", max_len=4096)
        _safe_env_items(self.extra_env)
        mounts: dict[str, str] = {}
        for source, target in self.read_only_mounts.items():
            source_path = _mount_path(source, "read_only_mount source")
            target_path = _mount_path(target, "read_only_mount target")
            if target_path in {socket_target, data_container_path}:
                raise SidecarConfigurationError("read-only mount collides with a protected sidecar mount")
            mounts[source_path] = target_path
        object.__setattr__(self, "read_only_mounts", mounts)
        _labels(self.plan, "server", self.labels)


@dataclass(frozen=True)
class WorkspaceCommandSpec:
    docker_host: str | DockerHostRef
    plan: NetworkPlan
    image: str
    container_name: str
    workspace_host_path: str
    command: tuple[str, ...] = ()
    workspace_container_path: str = "/workspace"
    vulnerable_runtime_host_path: str | None = None
    vulnerable_runtime_container_path: str = "/workspace/.cybergym-runtime"
    labels: Mapping[str, str] = field(default_factory=dict)
    extra_env: Mapping[str, str] = field(default_factory=dict)
    container_docker_host: str | None = None
    platform: str = "linux/amd64"

    def __post_init__(self) -> None:
        object.__setattr__(self, "docker_host", _host(self.docker_host))
        _validate_image(self.image)
        _safe_id(self.container_name, "container_name")
        _mount_path(self.workspace_host_path, "workspace_host_path")
        _mount_path(self.workspace_container_path, "workspace_container_path")
        if self.workspace_container_path == "/":
            raise SidecarConfigurationError("workspace mount cannot target root")
        runtime_target = _mount_path(
            self.vulnerable_runtime_container_path,
            "vulnerable_runtime_container_path",
        )
        object.__setattr__(self, "vulnerable_runtime_container_path", runtime_target)
        if self.vulnerable_runtime_host_path is not None:
            runtime_source = _mount_path(
                self.vulnerable_runtime_host_path,
                "vulnerable_runtime_host_path",
            )
            object.__setattr__(self, "vulnerable_runtime_host_path", runtime_source)
            if runtime_target == self.workspace_container_path:
                raise SidecarConfigurationError("vulnerable runtime cannot replace the workspace mount")
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", self.platform):
            raise SidecarConfigurationError("unsafe platform")
        for item in self.command:
            _text(item, "command argument", max_len=4096)
        _safe_env_items(self.extra_env)
        _reject_task_exposure(self.extra_env.values(), self.plan)
        _reject_task_exposure((*self.labels.keys(), *self.labels.values()), self.plan)
        if self.container_docker_host is not None:
            raise SidecarConfigurationError("workspace must not receive DOCKER_HOST")
        _labels(self.plan, "workspace", self.labels)


def _mount_arg(source: str, destination: str) -> str:
    source = _mount_path(source, "mount source")
    destination = _mount_path(destination, "mount destination")
    return f"type=bind,src={source},dst={destination}"


def build_network_create_argv(
    docker_host: str | DockerHostRef,
    plan: NetworkPlan,
    *,
    labels: Mapping[str, str] | None = None,
) -> list[str]:
    host = _host(docker_host)
    base = {
        "com.ouroboros.benchmark": "cybergym",
        "com.ouroboros.run": plan.campaign_id,
        "com.ouroboros.campaign": plan.campaign_id,
        "com.ouroboros.network": plan.network_name,
    }
    for key, value in (labels or {}).items():
        key, value = _text(key, "label key", max_len=256), _text(value, "label value", max_len=512)
        if key in base and base[key] != value:
            raise SidecarConfigurationError(f"custom label attempts to override {key}")
        base[key] = value
    argv = ["docker", "--host", host.value, "network", "create", "--driver", "bridge"]
    _append_labels(argv, base)
    argv.append(plan.network_name)
    return argv


def build_sidecar_argv(spec: SidecarCommandSpec) -> list[str]:
    """Build server argv; API key is inherited by env name and never serialized."""

    plan = spec.plan
    argv = [
        "docker", "--host", spec.docker_host.value, "run", "--detach", "--init", "--name", spec.container_name,
        "--platform", spec.platform, "--network", plan.network_name, "--network-alias", plan.server_alias,
    ]
    _append_labels(argv, _labels(plan, "server", spec.labels))
    if spec.publish_host_port:
        argv.extend((
            "--publish", f"{plan.verifier_bind_host}:{plan.verifier_host_port}:{plan.server_container_port}/tcp",
        ))
    argv.extend((
        "--mount", _mount_arg(spec.docker_host.socket_path, spec.socket_target),
        "--env", spec.api_key_env,
    ))
    if spec.container_docker_host is not None:
        argv.extend(("--env", f"{DOCKER_HOST_ENV}={_text(spec.container_docker_host, 'container_docker_host')}"))
    for key in sorted(spec.extra_env):
        argv.extend(("--env", f"{key}={spec.extra_env[key]}"))
    if spec.data_host_path is not None:
        argv.extend(("--mount", _mount_arg(spec.data_host_path, spec.data_container_path)))
    for source, target in sorted(spec.read_only_mounts.items()):
        argv.extend(("--mount", _mount_arg(source, target) + ",readonly"))
    argv.append(spec.image)
    argv.extend(spec.command)
    return argv


def build_workspace_argv(spec: WorkspaceCommandSpec) -> list[str]:
    """Build workspace argv; no Docker socket is mounted into the agent."""

    plan = spec.plan
    no_proxy = plan.no_proxy
    argv = [
        "docker", "--host", spec.docker_host.value, "run", "--detach", "--init", "--name", spec.container_name,
        "--platform", spec.platform, "--network", plan.network_name, "--network-alias", plan.workspace_alias,
    ]
    _append_labels(argv, _labels(plan, "workspace", spec.labels))
    argv.extend((
        "--mount", _mount_arg(spec.workspace_host_path, spec.workspace_container_path),
        "--env", f"CYBERGYM_SERVER_URL={plan.server_url}",
        "--env", f"NO_PROXY={no_proxy}", "--env", f"no_proxy={no_proxy}",
        "--env", f"CYBERGYM_AGENT_ID={plan.opaque_agent_id}",
    ))
    if spec.vulnerable_runtime_host_path is not None:
        argv.extend((
            "--mount",
            _mount_arg(
                spec.vulnerable_runtime_host_path,
                spec.vulnerable_runtime_container_path,
            ) + ",readonly",
            "--env",
            f"CYBERGYM_VULNERABLE_RUNTIME={spec.vulnerable_runtime_container_path}",
        ))
    if spec.container_docker_host is not None:
        # WorkspaceCommandSpec rejects this field; retain the branch only as
        # a defensive guard for objects produced by untrusted deserializers.
        raise SidecarConfigurationError("workspace must not receive DOCKER_HOST")
    for key in sorted(spec.extra_env):
        argv.extend(("--env", f"{key}={spec.extra_env[key]}"))
    argv.append(spec.image)
    argv.extend(spec.command)
    return argv


def api_key_attestation(value: Any) -> dict[str, Any]:
    """Record only presence, placeholder status and a short non-reversible fingerprint."""

    if isinstance(value, Mapping):
        present = value.get("present") is True
        placeholder = value.get("placeholder") is True
        fingerprint = value.get("fingerprint") if isinstance(value.get("fingerprint"), str) else None
        if fingerprint:
            fingerprint = fingerprint.lower()
            if not re.fullmatch(r"[0-9a-f]{8,64}", fingerprint):
                fingerprint = None
        # Sanitised runner observations may carry only a digest.  Do not trust
        # a caller-provided ``placeholder=false`` when that digest identifies
        # the public key shipped in upstream CyberGym documentation.
        placeholder = placeholder or _is_public_default_key_fingerprint(fingerprint)
        return {"present": present, "placeholder": placeholder, "fingerprint": fingerprint}
    if not isinstance(value, str) or not value:
        return {"present": False, "placeholder": False, "fingerprint": None}
    lowered = value.strip().lower()
    placeholders = {
        "placeholder", "changeme", "change-me", "your_api_key", "your-api-key", "api_key", "test-key", "test_key",
        "none", "null", "<api-key>", "${cybergym_api_key}", "cybergym_api_key", "your-cybergym-api-key",
    }
    digest = hashlib.sha256(value.encode()).hexdigest()
    fingerprint = digest[:16]
    placeholder = (
        lowered in placeholders
        or lowered.startswith(("replace_me", "replace-me"))
        or _is_public_default_key_fingerprint(fingerprint)
    )
    return {"present": True, "placeholder": placeholder, "fingerprint": fingerprint}


def _is_public_default_key_fingerprint(fingerprint: str | None) -> bool:
    """Return whether a redacted key fingerprint matches the upstream public example."""

    return bool(
        isinstance(fingerprint, str)
        and len(fingerprint) >= len(_PUBLIC_DEFAULT_API_KEY_FINGERPRINT)
        and fingerprint.startswith(_PUBLIC_DEFAULT_API_KEY_FINGERPRINT)
    )


def is_placeholder_api_key(value: Any) -> bool:
    return api_key_attestation(value)["placeholder"] is True


def require_api_key(value: Any) -> dict[str, Any]:
    status = api_key_attestation(value)
    if not status["present"] or status["placeholder"]:
        raise SidecarConfigurationError("CyberGym API key is missing or a placeholder")
    return status


@dataclass(frozen=True)
class SidecarExpectation:
    plan: NetworkPlan
    docker_host: str | DockerHostRef
    server_container_name: str
    workspace_container_name: str
    server_container_id: str | None = None
    workspace_container_id: str | None = None
    network_id: str | None = None
    socket_path: str | None = None
    image_digest: str | None = None
    server_pid: int | None = None
    workspace_pid: int | None = None
    executor_network_declaration: str = EXECUTOR_NETWORK_DECLARATION
    # New callers may attest distinct immutable server/workspace images.  The
    # legacy ``image_digest`` remains a compatibility shorthand and applies to
    # both roles when the role-specific values are omitted.
    server_image_digest: str | None = None
    workspace_image_digest: str | None = None
    # When false, no host port is published and host-side private calls must
    # use ``docker exec`` against server_id; the custom bridge may still have
    # outbound NAT.
    publish_host_port: bool = True

    def __post_init__(self) -> None:
        host = _host(self.docker_host)
        object.__setattr__(self, "docker_host", host)
        _safe_name(self.server_container_name, "server_container_name")
        _safe_name(self.workspace_container_name, "workspace_container_name")
        for value, name in ((self.server_container_id, "server_container_id"), (self.workspace_container_id, "workspace_container_id"), (self.network_id, "network_id")):
            if value is not None:
                _safe_id(value, name)
        if self.socket_path is not None:
            path = _safe_path(self.socket_path, "socket_path")
            if path != host.socket_path:
                raise SidecarConfigurationError("socket_path must equal selected rootless socket")
            object.__setattr__(self, "socket_path", path)
        legacy_digest = _digest(self.image_digest)
        server_digest = _digest(self.server_image_digest)
        workspace_digest = _digest(self.workspace_image_digest)
        if server_digest is None:
            server_digest = legacy_digest
        if workspace_digest is None:
            workspace_digest = legacy_digest
        if server_digest is None or workspace_digest is None:
            raise SidecarConfigurationError("image_digest is required for production custody")
        object.__setattr__(self, "image_digest", legacy_digest or server_digest)
        object.__setattr__(self, "server_image_digest", server_digest)
        object.__setattr__(self, "workspace_image_digest", workspace_digest)
        for value, name in ((self.server_pid, "server_pid"), (self.workspace_pid, "workspace_pid")):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise SidecarConfigurationError(f"{name} must be positive")
        if self.executor_network_declaration != EXECUTOR_NETWORK_DECLARATION:
            raise SidecarConfigurationError("executor declaration must be non-none host value")
        if not isinstance(self.publish_host_port, bool):
            raise SidecarConfigurationError("publish_host_port must be a boolean")


def _container_report(
    observation: Mapping[str, Any],
    expected: SidecarExpectation,
    role: str,
    expected_name: str,
    expected_id: str | None,
    expected_pid: int | None,
) -> tuple[dict[str, Any], list[str]]:
    plan = expected.plan
    failures: list[str] = []
    labels = _labels_from(observation)
    label_report: dict[str, Any] = {}
    for key, wanted in _labels(plan, role).items():
        actual = labels.get(key)
        passed = actual == wanted
        label_report[key] = {"observed": actual, "expected": wanted, "pass": passed}
        if not passed:
            failures.append(f"{role}.label.{key}")
    actual_name, actual_id, actual_pid = _name(observation), _id(observation), _pid(observation)
    if actual_name != expected_name:
        failures.append(f"{role}.name")
    if expected_id is not None and actual_id != expected_id:
        failures.append(f"{role}.id")
    if actual_pid is None or not _running(observation):
        failures.append(f"{role}.process")
    if expected_pid is not None and actual_pid != expected_pid:
        failures.append(f"{role}.pid")
    mode = _network_mode(observation)
    if mode != plan.network_name or mode in _FORBIDDEN_NETWORKS:
        failures.append(f"{role}.network_mode")
    network = _network(observation, plan.network_name)
    all_networks = _nested(observation, "NetworkSettings", "Networks")
    aliases = network.get("Aliases", ()) if network else ()
    expected_alias = plan.server_alias if role == "server" else plan.workspace_alias
    if not network or not isinstance(aliases, Sequence) or expected_alias not in aliases:
        failures.append(f"{role}.network_membership")
    if isinstance(all_networks, Mapping) and any(str(name).lower() in _FORBIDDEN_NETWORKS for name in all_networks):
        failures.append(f"{role}.forbidden_network_attachment")
    observed_network_id = network.get("NetworkID") if network else None
    if expected.network_id is not None and observed_network_id != expected.network_id:
        failures.append(f"{role}.network_id")
    expected_digest = (
        expected.server_image_digest if role == "server" else expected.workspace_image_digest
    ) or expected.image_digest
    digest_ok = expected_digest in _digests(observation)
    if not digest_ok:
        failures.append(f"{role}.image_digest")
    return {
        "name": actual_name, "id": actual_id, "pid": actual_pid, "running": _running(observation),
        "network_mode": mode, "network_name": plan.network_name if network else None,
        "network_id": observed_network_id,
        "aliases": list(aliases) if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)) else [],
        "labels": label_report, "image_digests": sorted(_digests(observation)),
        "expected_image_digest": expected_digest, "image_digest_ok": digest_ok,
    }, failures


def _socket_report(server: Mapping[str, Any], workspace: Mapping[str, Any], expected: SidecarExpectation) -> tuple[dict[str, Any], list[str]]:
    source = expected.socket_path or expected.docker_host.socket_path
    server_mounts, workspace_mounts = _mounts(server), _mounts(workspace)
    server_ok = any(src == source and dst == DEFAULT_SOCKET_TARGET for src, dst in server_mounts)
    workspace_visible = any(src == source or dst == DEFAULT_SOCKET_TARGET or "docker.sock" in dst for src, dst in workspace_mounts)
    failures = ([] if server_ok else ["server.socket_mount"]) + (["workspace.socket_visibility"] if workspace_visible else [])
    return {
        "socket_path": source, "server_mount": server_ok, "workspace_visible": workspace_visible,
        "server_mounts": [{"source": src, "destination": dst} for src, dst in server_mounts],
        "workspace_mounts": [{"source": src, "destination": dst} for src, dst in workspace_mounts],
    }, failures


def _publish_report(
    server: Mapping[str, Any], plan: NetworkPlan, *, required: bool = True
) -> tuple[dict[str, Any], list[str]]:
    bindings = _bindings(server, int(plan.server_container_port))
    if not required:
        # Outbound NAT does not require a host-published sidecar port.  An
        # unexpected mapping would widen the verifier boundary, so absence is
        # the only passing observation for exec transport.
        values = _nested(server, "NetworkSettings", "Ports")
        if not isinstance(values, Mapping):
            values = server.get("Ports")
        published_ports = sorted(
            str(port)
            for port, rows in (values.items() if isinstance(values, Mapping) else ())
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) and rows
        )
        ok = not published_ports
        return {
            "mode": "container_exec",
            "host_ip": None,
            "host_port": None,
            "container_port": plan.server_container_port,
            "loopback_only": False,
            "container_exec": True,
            "bindings": len(published_ports),
            "published_ports": published_ports,
        }, ([] if ok else ["server.unexpected_publish"])
    host_ip = bindings[0].get("HostIp") if len(bindings) == 1 else None
    host_port = bindings[0].get("HostPort") if len(bindings) == 1 else None
    try:
        host_port = int(host_port)
    except (TypeError, ValueError):
        host_port = None
    ok = len(bindings) == 1 and host_ip == plan.verifier_bind_host and host_port == plan.verifier_host_port
    return {"host_ip": host_ip, "host_port": host_port, "container_port": plan.server_container_port, "loopback_only": ok}, ([] if ok else ["server.loopback_publish"])


def _daemon_identity_report(
    observation: Mapping[str, Any],
    expected_host: DockerHostRef,
    *,
    required: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Validate structured evidence returned by the selected Docker daemon.

    A socket path is only a routing intent; it is not proof that the command
    reached the selected rootless daemon.  Production callers therefore pass
    a redacted ``docker info``/daemon-identity mapping.  Pure unit fixtures can
    omit it by leaving ``required`` false.
    """

    raw = observation.get("docker_info")
    source = "docker_info"
    if raw is None:
        raw = observation.get("daemon_identity")
        source = "daemon_identity"
    if raw is None:
        report = {
            "required": required,
            "source": None,
            "status": "missing" if required else "not_required",
            "ok": not required,
            "daemon_id": None,
            "server_version": None,
            "rootless": None,
            "endpoint": None,
            "docker_root_dir": None,
        }
        return report, (["docker_daemon_evidence"] if required else [])
    if not isinstance(raw, Mapping):
        if not required:
            return {
                "required": False,
                "source": source,
                "status": "not_required",
                "ok": True,
                "daemon_id": None,
                "server_version": None,
                "rootless": None,
                "endpoint": None,
                "docker_root_dir": None,
            }, []
        return {
            "required": required,
            "source": source,
            "status": "invalid",
            "ok": False,
            "daemon_id": None,
            "server_version": None,
            "rootless": None,
            "endpoint": None,
            "docker_root_dir": None,
        }, ["docker_daemon_evidence", "docker_daemon.type"]

    # Some runners wrap the ``docker info`` payload in a daemon key.  Merge
    # only this known structure; arbitrary objects are never trusted as proof.
    nested = raw.get("daemon")
    if isinstance(nested, Mapping):
        info: Mapping[str, Any] = {**dict(nested), **{key: value for key, value in raw.items() if key != "daemon"}}
    else:
        info = raw

    daemon_id: str | None = None
    raw_id = _mapping_value(info, ("ID", "Id", "id", "daemon_id", "daemonId"))
    if isinstance(raw_id, str):
        try:
            daemon_id = _safe_id(raw_id, "daemon_id")
        except SidecarConfigurationError:
            daemon_id = None
    server_version: str | None = None
    raw_version = _mapping_value(info, ("ServerVersion", "server_version", "version"))
    if isinstance(raw_version, str):
        try:
            server_version = _text(raw_version, "server_version", max_len=128)
        except SidecarConfigurationError:
            server_version = None
    rootless = _bool_mapping_value(info, ("Rootless", "rootless"))
    security_options = _mapping_value(info, ("SecurityOptions", "security_options"))
    if rootless is None and isinstance(security_options, str):
        security_options = (security_options,)
    if rootless is None and isinstance(security_options, Sequence) and not isinstance(security_options, (str, bytes)):
        rootless = any(_is_rootless_security_option(item) for item in security_options)
    raw_endpoint = _mapping_value(
        info,
        ("DockerHost", "docker_host", "Endpoint", "endpoint", "Socket", "socket", "socket_path"),
    )
    endpoint: str | None = None
    endpoint_ok = True
    endpoint_present = raw_endpoint is not None
    if endpoint_present:
        if isinstance(raw_endpoint, str):
            try:
                endpoint_ref = resolve_rootless_docker_host(raw_endpoint, allow_custom=True)
                endpoint = endpoint_ref.value
                endpoint_ok = endpoint == expected_host.value
            except SidecarConfigurationError:
                endpoint_ok = False
        else:
            endpoint_ok = False
    elif required:
        # In production, the selected socket must be part of the daemon
        # response itself; the separate observation path is only intent.
        endpoint_ok = False
    raw_root_dir = _mapping_value(info, ("DockerRootDir", "docker_root_dir", "root_dir"))
    docker_root_dir: str | None = None
    root_dir_ok = True
    root_dir_present = raw_root_dir is not None
    if root_dir_present:
        if isinstance(raw_root_dir, str):
            try:
                docker_root_dir = _safe_path(raw_root_dir, "docker_root_dir")
            except SidecarConfigurationError:
                root_dir_ok = False
        else:
            root_dir_ok = False

    checks = {
        "id": daemon_id is not None,
        "server_version": server_version is not None,
        "rootless": rootless is True,
        "endpoint": endpoint_ok,
        "docker_root_dir": root_dir_ok,
    }
    evidence_ok = all(checks[name] for name in ("id", "server_version", "rootless")) and endpoint_ok and root_dir_ok
    # Optional daemon evidence is diagnostic only.  A pure/library caller may
    # pass a partial observation (for example ``{"status": "not_checked"}`)
    # while exercising the container contract; that must not be confused with
    # a production attestation.  Production callers set ``required=True`` and
    # therefore retain the fail-closed behavior for every missing/malformed
    # field.
    ok = evidence_ok or not required
    failures = [] if ok else ["docker_daemon_evidence"]
    if required:
        failures.extend(f"docker_daemon.{name}" for name, passed in checks.items() if not passed)
    return {
        "required": required,
        "source": source,
        "status": "verified" if evidence_ok else "not_required" if not required else "invalid",
        "ok": ok,
        "daemon_id": daemon_id,
        "server_version": server_version,
        "rootless": rootless,
        "endpoint": endpoint,
        "docker_root_dir": docker_root_dir,
        "checks": checks,
        "endpoint_present": endpoint_present,
        "docker_root_dir_present": root_dir_present,
    }, failures


def check_sidecar_attestation(
    observation: Mapping[str, Any],
    expected: SidecarExpectation,
    *,
    api_key: Any = None,
    connectivity: Mapping[str, Any] | None = None,
    require_connectivity: bool = True,
    require_daemon_evidence: bool = False,
    require_protected_route_evidence: bool = False,
) -> dict[str, Any]:
    """Return a complete secret-free report; unknown custody facts fail closed."""

    server, workspace = observation.get("server"), observation.get("workspace")
    if not isinstance(server, Mapping) or not isinstance(workspace, Mapping):
        return {"schema": SCHEMA_VERSION, "ok": False, "failed_checks": ["server_or_workspace_observation"]}
    failures: list[str] = []
    observed_host = observation.get("docker_host")
    if observed_host is None:
        failures.append("docker_host_unknown")
    else:
        try:
            observed_host_ref = _host(observed_host)
        except SidecarConfigurationError:
            observed_host_ref = None
        if observed_host_ref is None or observed_host_ref.value != expected.docker_host.value:
            failures.append("docker_host_mismatch")
    daemon_report, more = _daemon_identity_report(
        observation,
        expected.docker_host,
        required=require_daemon_evidence,
    )
    failures.extend(more)
    network_observation = observation.get("network")
    network_report: dict[str, Any] = {
        "name": expected.plan.network_name,
        "id": expected.network_id,
        "internal": None,
        "driver": None,
    }
    if isinstance(network_observation, Mapping):
        observed_network_name = network_observation.get("Name") or network_observation.get("name")
        observed_network_id = network_observation.get("Id") or network_observation.get("ID") or network_observation.get("id")
        internal = network_observation.get("Internal")
        driver = network_observation.get("Driver") or network_observation.get("driver")
        network_report = {"name": observed_network_name, "id": observed_network_id, "internal": internal, "driver": driver}
        if observed_network_name != expected.plan.network_name:
            failures.append("network.name")
        if expected.network_id is not None and observed_network_id != expected.network_id:
            failures.append("network.id")
        if internal is not False:
            failures.append("network.internal")
        if driver != "bridge":
            failures.append("network.driver")
    else:
        failures.append("network_observation_unknown")
    server_report, more = _container_report(
        server, expected, "server", expected.server_container_name, expected.server_container_id, expected.server_pid
    )
    failures.extend(more)
    workspace_report, more = _container_report(
        workspace, expected, "workspace", expected.workspace_container_name, expected.workspace_container_id, expected.workspace_pid
    )
    failures.extend(more)
    socket_report, more = _socket_report(server, workspace, expected)
    failures.extend(more)
    publish_report, more = _publish_report(
        server, expected.plan, required=expected.publish_host_port
    )
    failures.extend(more)
    key_report = api_key_attestation(api_key)
    if api_key is None:
        key_report = {"present": None, "placeholder": None, "fingerprint": None}
        failures.append("api_key_unknown")
    elif not key_report["present"] or key_report["placeholder"]:
        failures.append("api_key")
    declaration = observation.get("executor_network", expected.executor_network_declaration)
    if declaration != EXECUTOR_NETWORK_DECLARATION:
        failures.append("executor_network_declaration")
    if connectivity is None:
        connectivity_report: Mapping[str, Any] = {"schema": f"{SCHEMA_VERSION}.connectivity", "ok": False, "status": "not_provided"}
        if require_connectivity:
            failures.append("connectivity_unknown")
        elif require_protected_route_evidence:
            failures.append("protected_route_connectivity_unknown")
    else:
        connectivity_report = evaluate_connectivity_checks(
            connectivity,
            require_protected_route_evidence=require_protected_route_evidence,
        )
        if not connectivity_report.get("ok"):
            failures.append("connectivity")
    network_id = _nested(server, "NetworkSettings", "Networks", expected.plan.network_name, "NetworkID")
    if expected.network_id is None and not network_id:
        failures.append("network.id_unknown")
    if network_report["id"] is None:
        network_report["id"] = network_id or expected.network_id
    return {
        "schema": SCHEMA_VERSION, "ok": not failures, "failed_checks": sorted(set(failures)),
        "docker_host": {"value": expected.docker_host.value, "socket_path": expected.docker_host.socket_path, "rootless": True},
        "docker_info": dict(daemon_report),
        "docker_daemon": dict(daemon_report),
        "network": network_report,
        "server": server_report, "workspace": workspace_report, "socket": socket_report,
        "published_verifier": publish_report, "api_key": key_report,
        "executor_network_declaration": declaration, "executor_network_is_docker_host": False,
        "connectivity": dict(connectivity_report), "cleanup": {"status": "pending", "owned_only": True},
    }


def attest_sidecar_runtime(
    observation: Mapping[str, Any],
    expected: SidecarExpectation,
    *,
    require_daemon_evidence: bool = False,
    require_protected_route_evidence: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Attest a sidecar; production callers pass ``require_daemon_evidence=True``."""

    report = check_sidecar_attestation(
        observation,
        expected,
        require_daemon_evidence=require_daemon_evidence,
        require_protected_route_evidence=require_protected_route_evidence,
        **kwargs,
    )
    if not report.get("ok"):
        raise SidecarAttestationError("CyberGym sidecar attestation failed", report)
    return report


validate_sidecar_attestation = check_sidecar_attestation


@dataclass(frozen=True)
class ProcessCustody:
    role: str
    pid: int
    container_id: str
    command_digest: str | None = None
    cleanup_status: str = "pending"
    cwd: str | None = None
    port: int | None = None

    def __post_init__(self) -> None:
        _safe_id(self.role, "process_role")
        if isinstance(self.pid, bool) or not isinstance(self.pid, int) or self.pid <= 0:
            raise SidecarConfigurationError("process pid must be positive")
        _safe_id(self.container_id, "container_id")
        _digest(self.command_digest)
        if self.cwd is not None:
            _safe_path(self.cwd, "process_cwd")
        if self.port is not None:
            _port(self.port, "process_port")
        if self.cleanup_status not in {"pending", "removed", "verified", "failed"}:
            raise SidecarConfigurationError("invalid cleanup status")


def build_process_custody(
    role: str,
    pid: int,
    container_id: str,
    *,
    command: Sequence[str] | None = None,
    cwd: str | None = None,
    port: int | None = None,
) -> ProcessCustody:
    digest = None
    if command is not None:
        values = tuple(_text(item, "command argument", max_len=4096) for item in command)
        digest = "sha256:" + hashlib.sha256("\0".join(values).encode()).hexdigest()
    return ProcessCustody(role, pid, container_id, digest, cwd=cwd, port=port)


def attest_process_custody(observed: Mapping[str, Any], expected: ProcessCustody) -> dict[str, Any]:
    try:
        pid = int(observed.get("pid"))
    except (TypeError, ValueError):
        pid = None
    container_id = observed.get("container_id") or observed.get("Id")
    cwd = observed.get("cwd")
    port = observed.get("port")
    try:
        port = int(port) if port is not None else None
    except (TypeError, ValueError):
        port = None
    checks = {
        "pid": pid == expected.pid,
        "container_id": container_id == expected.container_id,
        "cwd": expected.cwd is None or cwd == expected.cwd,
        "port": expected.port is None or port == expected.port,
    }
    return {
        "role": expected.role,
        "pid": pid,
        "container_id": container_id,
        "cwd": cwd,
        "port": port,
        "expected_pid": expected.pid,
        "expected_container_id": expected.container_id,
        "expected_cwd": expected.cwd,
        "expected_port": expected.port,
        "checks": checks,
        "ok": all(checks.values()),
        "cleanup_status": expected.cleanup_status,
    }


def _owned_target(value: str, name: str) -> str:
    value = _safe_id(value, name)
    if value.lower() in {"all", "none", "default", "*"}:
        raise SidecarConfigurationError(f"unsafe cleanup target {name}")
    return value


@dataclass(frozen=True)
class CleanupPlan:
    """Exact, campaign-owned Docker cleanup targets.

    ``network_id`` is deliberately mandatory even though the human-readable
    network name is retained for attestation/reporting.  A name fallback can
    resolve to an unrelated network after a retry or concurrent campaign.
    Cleanup observations must carry the campaign ownership evidence described
    by :func:`validate_cleanup_observation` before a caller treats removal as
    complete.
    """

    docker_host: str | DockerHostRef
    campaign_id: str
    network_name: str = NETWORK_NAME
    server_container_id: str | None = None
    workspace_container_ids: tuple[str, ...] = ()
    network_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "docker_host", _host(self.docker_host))
        campaign = _safe_id(self.campaign_id, "campaign_id")
        object.__setattr__(self, "campaign_id", campaign)
        if self.network_name != NETWORK_NAME:
            raise SidecarConfigurationError("cleanup network must be cybergym-internal")
        if self.server_container_id is None and not self.workspace_container_ids:
            raise SidecarConfigurationError("cleanup requires owned container ids")
        if isinstance(self.workspace_container_ids, (str, bytes, Mapping)):
            raise SidecarConfigurationError("workspace_container_ids must be an iterable of ids")
        if self.server_container_id is not None:
            object.__setattr__(self, "server_container_id", _owned_target(self.server_container_id, "server_container_id"))
        workspace_ids = tuple(_owned_target(value, "workspace_container_id") for value in self.workspace_container_ids)
        object.__setattr__(self, "workspace_container_ids", workspace_ids)
        if self.network_id is None:
            raise SidecarConfigurationError("cleanup requires an explicit resolved network_id")
        object.__setattr__(self, "network_id", _owned_target(self.network_id, "network_id"))

    @property
    def owner_label(self) -> str:
        """Canonical campaign label that cleanup evidence must attest."""

        return f"com.ouroboros.campaign={self.campaign_id}"


def build_cleanup_plan(
    expected: SidecarExpectation,
    *,
    workspace_container_ids: Iterable[str] = (),
    network_id: str | None = None,
) -> CleanupPlan:
    if isinstance(workspace_container_ids, (str, bytes, Mapping)):
        raise SidecarConfigurationError("workspace_container_ids must be an iterable of ids")
    workspace_ids = tuple(workspace_container_ids)
    if not workspace_ids and expected.workspace_container_id is not None:
        workspace_ids = (expected.workspace_container_id,)
    resolved_network_id = expected.network_id if network_id is None else network_id
    if network_id is not None and expected.network_id is not None and network_id != expected.network_id:
        raise SidecarConfigurationError("cleanup network_id does not match attested expectation")
    return CleanupPlan(
        expected.docker_host,
        expected.plan.campaign_id,
        server_container_id=expected.server_container_id,
        workspace_container_ids=workspace_ids,
        network_id=resolved_network_id,
    )


def cleanup_argv(plan: CleanupPlan) -> tuple[tuple[str, ...], ...]:
    """Emit only exact owned ids; never emits prune, broad names or wildcards."""

    targets = [*plan.workspace_container_ids]
    if plan.server_container_id is not None:
        targets.append(plan.server_container_id)
    commands: list[tuple[str, ...]] = []
    if targets:
        commands.append(("docker", "--host", plan.docker_host.value, "rm", "--force", *targets))
    commands.append(("docker", "--host", plan.docker_host.value, "network", "rm", plan.network_id))
    return tuple(commands)


build_cleanup_commands = cleanup_argv


def validate_cleanup_observation(observation: Mapping[str, Any], plan: CleanupPlan) -> dict[str, Any]:
    """Validate exact removals plus explicit campaign ownership evidence.

    The runner may put ownership facts in an ``ownership`` mapping (the
    canonical wire shape) or at the top level for compatibility.  In either
    shape all three facts are required: the campaign label, the complete set
    of owned container ids, and the owned network id.  Missing or malformed
    facts fail closed; no network-name or ``all`` fallback is accepted.
    """

    def _id_tuple(value: Any) -> tuple[str, ...] | None:
        if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
            return None
        values = tuple(value)
        if any(not isinstance(item, str) for item in values):
            return None
        if len(set(values)) != len(values):
            return None
        return values

    removed = _id_tuple(observation.get("removed_container_ids", ()))
    removed_ids = set(removed or ())
    expected = set(plan.workspace_container_ids)
    if plan.server_container_id is not None:
        expected.add(plan.server_container_id)
    network_removed = observation.get("network_removed") is True
    unexpected = removed_ids - expected
    observed_network = observation.get("removed_network_id")
    network_matches = observed_network == plan.network_id

    ownership = observation.get("ownership")
    ownership_map = ownership if isinstance(ownership, Mapping) else {}
    owner_campaign = observation.get("owner_campaign_id")
    if owner_campaign is None:
        owner_campaign = ownership_map.get("campaign_id")
    owner_label = observation.get("owner_label")
    if owner_label is None:
        owner_label = ownership_map.get("owner_label")
    owned_raw = observation.get("owned_container_ids")
    if owned_raw is None:
        owned_raw = ownership_map.get("container_ids")
    owned = _id_tuple(owned_raw)
    owned_ids = set(owned or ())
    owned_network = observation.get("owned_network_id")
    if owned_network is None:
        owned_network = ownership_map.get("network_id")
    campaign_matches = (
        (owner_campaign == plan.campaign_id and (owner_label is None or owner_label == plan.owner_label))
        or (owner_campaign is None and owner_label == plan.owner_label)
    )
    containers_match = owned is not None and owned_ids == expected
    owned_network_matches = owned_network == plan.network_id
    ownership_ok = campaign_matches and containers_match and owned_network_matches
    ok = expected == removed_ids and removed is not None and network_removed and network_matches and ownership_ok
    return {
        "schema": f"{SCHEMA_VERSION}.cleanup", "ok": ok, "expected_container_ids": sorted(expected),
        "removed_container_ids": sorted(removed_ids), "network": plan.network_id,
        "unexpected_container_ids": sorted(unexpected), "removed_network_id": observed_network,
        "network_removed": network_removed, "network_matches": network_matches,
        "ownership": {
            "campaign_id": owner_campaign, "owner_label": owner_label,
            "container_ids": sorted(owned_ids), "network_id": owned_network,
            "campaign_matches": campaign_matches, "containers_match": containers_match,
            "network_matches": owned_network_matches, "ok": ownership_ok,
        },
        "status": "verified" if ok else "failed",
    }


def build_lifecycle_commands(
    server: SidecarCommandSpec,
    workspace: WorkspaceCommandSpec,
    *,
    create_network: bool = True,
) -> tuple[tuple[str, ...], ...]:
    """Return start commands; execution and process waiting stay launcher-owned."""

    commands: list[tuple[str, ...]] = []
    if create_network:
        commands.append(tuple(build_network_create_argv(server.docker_host, server.plan)))
    commands.extend((tuple(build_sidecar_argv(server)), tuple(build_workspace_argv(workspace))))
    return tuple(commands)


class CommandRunner(Protocol):
    """Injected command seam; implementations own subprocess/process custody."""

    def __call__(self, argv: Sequence[str], *, env: Mapping[str, str]) -> Any: ...


__all__ = [
    "API_KEY_ENV", "CleanupPlan", "CommandRunner", "DEFAULT_SOCKET_TARGET", "DOCKER_HOST_ENV", "DockerHostRef",
    "EXECUTOR_NETWORK_DECLARATION", "LOOPBACK_HOST", "NETWORK_NAME", "NetworkPlan", "ProcessCustody",
    "SCHEMA_VERSION", "SidecarAttestationError", "SidecarCommandSpec", "SidecarConfigurationError",
    "SidecarExpectation", "WorkspaceCommandSpec", "api_key_attestation", "attest_process_custody",
    "attest_sidecar_runtime", "build_cleanup_commands", "build_cleanup_plan", "build_connectivity_probe_plan",
    "build_network_create_argv", "build_network_plan", "build_no_proxy", "build_private_route", "build_process_custody",
    "build_lifecycle_commands",
    "build_server_sidecar_argv", "build_sidecar_argv", "build_task_workspace_argv", "build_workspace_argv", "check_sidecar_attestation",
    "cleanup_argv", "docker_host_environment", "evaluate_connectivity_checks", "is_placeholder_api_key", "make_dns_alias",
    "make_opaque_agent_id",
    "required_resource_labels", "require_api_key", "require_explicit_rootless_docker_host", "resolve_rootless_docker_host",
    "validate_cleanup_observation", "validate_sidecar_attestation",
]


# Keep one spelling for callers that prefer an assertion-shaped API.
build_task_workspace_argv = build_workspace_argv
build_server_sidecar_argv = build_sidecar_argv
