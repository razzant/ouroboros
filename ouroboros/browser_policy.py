"""Task browser targets and Ouroboros control-request policy.

Playwright lifecycle and operations remain in tools.browser. This owner composes
existing resource authority, concrete origins and live service identity.
"""
from __future__ import annotations

import ipaddress
import pathlib
import re
import socket
from typing import Any
from urllib.parse import urlparse

from ouroboros.contracts.task_contract import normalize_allowed_origins, normalize_browser_origin
from ouroboros.server_auth import is_loopback_host
from ouroboros.server_process import SERVICE_IDENTITY_UNKNOWN, runtime_service_identity

_NONSTANDARD_NUMERIC_IPV4_RE = re.compile(r"^(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+)){0,3}$", re.I)
# IPv4 metadata is link-local; the IPv6 metadata address also needs its exact check.
_METADATA_IPV6_ADDRESSES = frozenset({ipaddress.ip_address("fd00:ec2::254")})


def _resolved_addresses(host: str) -> set:
    try:
        return {ipaddress.ip_address(host)}
    except ValueError:
        addresses = {ipaddress.ip_address(info[4][0])
                     for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)}
        if not addresses:
            raise OSError(f"DNS lookup returned no addresses for {host}")
        return addresses


def _is_local_address(address: Any) -> bool:
    """Test local interface ownership without connecting to the destination."""
    if address.is_loopback:
        return True
    family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind((str(address), 0))
        return True
    except OSError:
        return False


def runtime_service_kind(url: str, ctx: Any = None) -> str:
    """The Ouroboros service behind ``url``: a proven kind, unknown, or nothing.

    ``main``/``host_service``/``local_model`` when a live binding or this
    installation's own process facts prove the endpoint ours;
    ``SERVICE_IDENTITY_UNKNOWN`` when a recorded or configured Ouroboros endpoint
    matches but its process cannot be verified; ``""`` for any other target,
    including an unrelated application on a default port. Truthy means "treat as
    Ouroboros" — an unverifiable expected endpoint is refused, never opened.
    Identity is read from the actual binding and live process, never inferred
    from a port number or an ``/api/owner/...`` pathname alone.
    """
    from ouroboros.tools.core_secret_paths import restricted_data_roots

    origin = normalize_browser_origin(url)
    if not origin:
        return ""
    parsed = urlparse(origin)
    resolved: dict = {}

    def host_matches(bind_host: str) -> bool:
        # Resolve once for matched endpoints and the legacy loopback check.
        try:
            bound = ipaddress.ip_address(bind_host)
        except ValueError:
            return False
        if "addresses" not in resolved:
            resolved["addresses"] = _resolved_addresses(parsed.hostname or "")
        return any(address == bound or (bound.is_unspecified and _is_local_address(address))
                   for address in resolved["addresses"])

    try:
        for root in restricted_data_roots(ctx):
            if identity := runtime_service_identity(root, parsed.port, host_matches):
                return identity
        return ""
    except OSError:
        # A recorded endpoint's port matched and the target could not be resolved.
        return SERVICE_IDENTITY_UNKNOWN


def _task_allowed_origins(ctx: Any) -> list[str]:
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    contract = getattr(ctx, "task_contract", None) or metadata.get("task_contract")
    contract = contract if isinstance(contract, dict) else {}
    policy = contract.get("resource_policy") or metadata.get("resource_policy")
    return normalize_allowed_origins(policy.get("allowed_origins")) if isinstance(policy, dict) else []


def browser_url_block_reason(url: str, ctx: Any = None, *, restricted: bool) -> str:
    """One target decision for navigation, actions and requests exposed by Playwright."""
    parsed = urlparse(str(url or ""))
    if not restricted:
        return "BROWSER_METADATA_BLOCKED: link-local/cloud metadata target" if _is_metadata_blocked_browser_url(url) else ""
    if parsed.scheme == "file":
        return "" if _file_url_under_workspace(parsed, ctx) else "BROWSER_LOCAL_READONLY_BLOCKED: file URL is outside the task workspace"
    origin = normalize_browser_origin(url)
    if not origin:
        return "BROWSER_LOCAL_READONLY_BLOCKED: expected a concrete HTTP(S) target"
    host = (parsed.hostname or "").rstrip(".").lower()
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if _NONSTANDARD_NUMERIC_IPV4_RE.match(host):
            return "BROWSER_LOCAL_READONLY_BLOCKED: nonstandard numeric IP spelling"
    try:
        addresses = _resolved_addresses(host)
    except OSError as exc:
        # An unclassifiable target is refused before any navigation; this is a
        # policy-availability fact, not an invented denial of authority.
        return f"BROWSER_POLICY_UNAVAILABLE: DNS lookup failed for {host} ({exc}); the target could not be classified"
    if any(_is_metadata_ip(address) or address.is_unspecified for address in addresses):
        return "BROWSER_LOCAL_READONLY_BLOCKED: metadata/link-local/reserved target"
    identity = runtime_service_kind(url, ctx)
    if identity == SERVICE_IDENTITY_UNKNOWN:
        return "BROWSER_LOCAL_READONLY_BLOCKED: expected Ouroboros control-service endpoint whose identity could not be verified"
    if identity:
        return "BROWSER_LOCAL_READONLY_BLOCKED: actual Ouroboros control-service endpoint"
    # The pre-existing literal-loopback capability is unchanged. DNS aliases
    # resolving privately need the same concrete origin grant as LAN services.
    if is_loopback_host(host) or host == "localhost":
        return ""
    if any(address.is_reserved for address in addresses):
        return "BROWSER_LOCAL_READONLY_BLOCKED: reserved target"
    if origin in _task_allowed_origins(ctx):
        return ""
    if any(_is_blocked_subagent_ip(address) for address in addresses):
        return f"RESOURCE_POLICY_BLOCKED: browser origin_not_granted; task resource_policy.allowed_origins does not include {origin}"
    return ""


def browser_request_block_reason(request: Any, ctx: Any, *, restricted: bool) -> str:
    """One request decision: the target decision plus owner-operation shapes at Ouroboros.

    The owner POST shapes apply at a proven Ouroboros endpoint and at an expected
    one whose identity is unknown; an unrelated application reusing the pathname
    on any other port keeps working."""
    reason = browser_url_block_reason(request.url, ctx, restricted=restricted)
    if reason or restricted:
        return reason  # Restricted target checks already refused every runtime identity.
    if any(predicate(request) for predicate in (
        _is_context_mode_owner_post, _is_safety_mode_owner_post,
        _is_owner_skill_attest_post, _is_owner_settings_self_elevation_post,
    )) and runtime_service_kind(request.url, ctx):
        return "BROWSER_OWNER_CONTROL_BLOCKED: this operation belongs to the owner"
    return ""


def _file_url_under_workspace(parsed: Any, ctx: Any) -> bool:
    """True only when a file:// path resolves under the task's EXPLICIT workspace
    root, so a subagent can view its own built app but not the data root/secrets."""
    if ctx is None:
        return False
    ws = str(getattr(ctx, "workspace_root", "") or "").strip()
    if not ws:
        return False
    try:
        from urllib.request import url2pathname

        path = pathlib.Path(url2pathname(parsed.path)).resolve(strict=False)
        base = pathlib.Path(ws).resolve(strict=False)
        path.relative_to(base)
        return True
    except (ValueError, OSError):
        return False



def _is_blocked_subagent_ip(ip: ipaddress._BaseAddress) -> bool:
    return bool(
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_unspecified
        or ip.is_reserved
    )



def _is_metadata_ip(ip: ipaddress._BaseAddress) -> bool:
    # Unwrap IPv4-mapped IPv6 (http://[::ffff:169.254.169.254]/) so the
    # link-local check sees the real IPv4 — mirrors mcp_client's guard.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return bool(ip.is_link_local) or ip in _METADATA_IPV6_ADDRESSES



def _is_metadata_blocked_browser_url(url: str) -> bool:
    """Main-agent guard: True only for link-local/cloud-metadata destinations."""
    parsed = urlparse(str(url or ""))
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").strip().rstrip(".").lower()
    if not host:
        return False
    try:
        return _is_metadata_ip(ipaddress.ip_address(host))
    except ValueError:
        pass
    if _NONSTANDARD_NUMERIC_IPV4_RE.match(host):
        # Decimal/hex IPv4 spellings (e.g. http://2852039166/) bypass naive
        # string checks; resolve via inet_aton normalization below.
        try:
            packed = socket.inet_aton(host)
            return _is_metadata_ip(ipaddress.ip_address(socket.inet_ntoa(packed)))
        except OSError:
            return True
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return False  # unresolvable hosts fail naturally at fetch time
    for info in infos:
        try:
            if _is_metadata_ip(ipaddress.ip_address(str(info[4][0]))):
                return True
        except ValueError:
            continue
    return False



def _blocks_context_mode_self_lowering_js(value: str) -> bool:
    low = str(value or "").lower()
    return "low" in low and (
        "/api/owner/context-mode" in low
        or ("ouroboros_context_mode" in low and ("settings.json" in low or "save_settings" in low))
    )



def _blocks_safety_mode_self_lowering_js(value: str) -> bool:
    """Block browser JS that tries to change the owner-only LLM-safety coverage mode
    (v6.54.3) — the click+fetch bypass of the dedicated owner endpoint. URL-decode
    first so a percent-encoded path (``safety%2Dmode``) cannot slip the literal
    match (review round 6; mirrors the owner-attestation guard)."""
    import urllib.parse

    low = str(value or "").lower()
    decoded = urllib.parse.unquote(urllib.parse.unquote(low)).lower()
    text = f"{low} {decoded}"
    return (
        "/api/owner/safety-mode" in text
        or ("ouroboros_safety_mode" in text and (
            "settings.json" in text or "save_settings" in text or "/api/settings" in text
        ))
    )



def _blocks_mutative_toggle_js(value: str) -> bool:
    """Block browser JS that tries to enable the owner-only mutative-subagents toggle."""
    low = str(value or "").lower()
    return "ouroboros_allow_mutative_subagents" in low and (
        "settings.json" in low or "save_settings" in low or "/api/settings" in low
    )



def _blocks_post_task_evolution_js(value: str) -> bool:
    """Block browser JS that tries to set an owner-only self-evolution control (the
    post-task evolution toggle or the persistent evolution-objective steer)."""
    low = str(value or "").lower()
    return (
        "ouroboros_post_task_evolution" in low
        or "ouroboros_evolution_persistent_objective" in low
    ) and (
        "settings.json" in low or "save_settings" in low or "/api/settings" in low
    )



def _blocks_owner_skill_attest_js(value: str) -> bool:
    """Block browser JS that tries to loopback-POST the OWNER-ONLY skill-attestation
    endpoint (C1, v6.39) — owner-attestation skips the LLM skill review and must be
    owner-issued, never agent self-callable from a browser fetch. URL-decode first so a
    percent-encoded path (``%61ttest-review``) cannot slip past the literal match."""
    import urllib.parse
    low = str(value or "").lower()
    decoded = urllib.parse.unquote(urllib.parse.unquote(low)).lower()
    text = f"{low} {decoded}"
    return "/api/owner/skills/" in text and "attest-review" in text



def _is_context_mode_owner_post(request: Any) -> bool:
    try:
        parsed = urlparse(str(request.url or ""))
        method = str(request.method or "").upper()
    except Exception:
        return False
    return method == "POST" and parsed.path.rstrip("/") == "/api/owner/context-mode"



def _is_safety_mode_owner_post(request: Any) -> bool:
    """POST to the owner safety-mode endpoint — decoded, so a percent-encoded
    path cannot slip past (the broad ``**/api/owner/**`` route registration
    feeds RAW URLs here; Starlette decodes server-side, so we must too)."""
    import urllib.parse

    try:
        parsed = urlparse(str(request.url or ""))
        method = str(request.method or "").upper()
    except Exception:
        return False
    path = urllib.parse.unquote(urllib.parse.unquote(parsed.path)).rstrip("/")
    return method == "POST" and path == "/api/owner/safety-mode"



def _is_owner_skill_attest_post(request: Any) -> bool:
    """A browser POST to the owner-only skill owner-attestation endpoint — the click/form
    bypass of the evaluate-only JS guard (C1, v6.39)."""
    try:
        import urllib.parse
        parsed = urlparse(str(request.url or ""))
        method = str(request.method or "").upper()
        # Decode so a percent-encoded path (which the server decodes before routing) is
        # matched the same way the route is registered.
        path = urllib.parse.unquote(urllib.parse.unquote(parsed.path)).rstrip("/").lower()
    except Exception:
        return False
    return method == "POST" and path.startswith("/api/owner/skills/") and path.endswith("/attest-review")



def _is_owner_settings_self_elevation_post(request: Any) -> bool:
    """A browser POST /api/settings carrying an owner-only self-modification toggle —
    the click+Save bypass of the evaluate-only JS guards."""
    try:
        if str(request.method or "").upper() != "POST":
            return False
        parsed = urlparse(str(request.url or ""))
        if parsed.path.rstrip("/") != "/api/settings":
            return False
        body = str(request.post_data or "").lower()
    except Exception:
        return False
    return (
        "ouroboros_post_task_evolution" in body
        or "ouroboros_allow_mutative_subagents" in body
        or "ouroboros_evolution_persistent_objective" in body
        # v6.88: the delegated-executor POLICY. D1 makes the executor axis the OWNER's,
        # and this key is the whole of it — which route answers, on whose subscription.
        # It rides the generic settings path deliberately (the Settings UI sets a route
        # string often, and a dedicated endpoint would be ceremony for nothing), so it
        # joins the keys already guarded here rather than getting a mechanism of its own.
        or "ouroboros_subagent_harness" in body
    )


def browser_evaluate_block_reason(url: str, value: str, ctx: Any = None) -> str:
    """Keep owner-operation JavaScript policy at the same owner as URL policy."""
    if not runtime_service_kind(url, ctx):
        return ""
    if _blocks_context_mode_self_lowering_js(value):
        return (
            "⚠️ CONTEXT_MODE_SELF_LOWERING_BLOCKED: browser JavaScript "
            "looks like an attempt to lower OUROBOROS_CONTEXT_MODE. "
            "Context mode is owner-controlled — ask the owner to use "
            "the Low/Max toggle."
        )
    if _blocks_safety_mode_self_lowering_js(value):
        return (
            "⚠️ SAFETY_MODE_SELF_LOWERING_BLOCKED: browser JavaScript "
            "looks like an attempt to change OUROBOROS_SAFETY_MODE. "
            "LLM-safety coverage is owner-controlled (BIBLE P3) — the agent "
            "must not reduce its own supervision."
        )
    if _blocks_mutative_toggle_js(value):
        return (
            "⚠️ ELEVATION_BLOCKED: browser JavaScript looks like an attempt to enable "
            "OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS. This master toggle is owner-controlled — "
            "the agent must not self-enable mutative subagents."
        )
    if _blocks_post_task_evolution_js(value):
        return (
            "⚠️ ELEVATION_BLOCKED: browser JavaScript looks like an attempt to enable "
            "OUROBOROS_POST_TASK_EVOLUTION. Post-task self-evolution is owner-controlled — "
            "the agent must not self-enable it."
        )
    if _blocks_owner_skill_attest_js(value):
        return (
            "⚠️ OWNER_SKILL_ATTESTATION_SELF_CALL_BLOCKED: browser JavaScript looks like an "
            "attempt to POST /api/owner/skills/<skill>/attest-review. Owner-attestation skips "
            "the LLM skill review and is owner-only — the agent must not self-attest its own skill."
        )
    return ""
