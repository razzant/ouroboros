"""Frozen PluginAPI contract for extension skills.

``plugin.py`` exposes ``register(api: PluginAPI)`` and may use only this ABI.
Signature or permission tightening requires a schema/version bump; additive
optional methods are allowed when documented here and pinned by contract tests.
Registrations are declarative and are torn down when a skill unloads.
"""

from __future__ import annotations

import enum
import hashlib
import inspect
import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Sequence, TypedDict, runtime_checkable

from ouroboros.contracts.skill_manifest import VALID_SKILL_PERMISSIONS
from ouroboros.skill_token import SkillToken

# 2.0 (ABI 7.0): manifest-negotiated generations. A payload declares the
# ``plugin_api`` manifest field (major strict, minor = minimum); a payload
# WITHOUT the field binds against the LEGACY generation by construction and
# keeps loading on its existing hash-bound review PASS (grandfather), but is
# refused a NEW PASS at issuance (``extension_new_pass_admission_error``).
# 2.0 absorbs 1.4's presence capability (reviewed transport skills may submit
# authenticated non-owner presence events through the Host Service ceiling).
PLUGIN_API_VERSION = "2.0"
# The generation an extension WITHOUT the manifest field binds against —
# "absent ≡ 1.3 by construction" (owner-ratified §6.1-Δ; deliberately NOT 1.4).
LEGACY_PLUGIN_API_GENERATION = "1.3"


# Core settings keys require explicit content-hash-bound owner grants.
FORBIDDEN_SKILL_SETTINGS: frozenset[str] = frozenset({
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_COMPATIBLE_API_KEY",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY", "GIGACHAT_CREDENTIALS", "GIGACHAT_PASSWORD",
    "ANTHROPIC_API_KEY", "MINIMAX_API_KEY", "DEEPSEEK_API_KEY", "GITHUB_TOKEN",
    "OUROBOROS_NETWORK_PASSWORD",
})


# Single carrier (ABI-1): ``skill_manifest.VALID_SKILL_PERMISSIONS`` is the
# one permission vocabulary; the extension contract re-derives it so the two
# can never drift again (the historical iframe_raw desync).
VALID_EXTENSION_PERMISSIONS: frozenset[str] = VALID_SKILL_PERMISSIONS

VALID_EXTENSION_ROUTE_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "POST", "PUT", "DELETE", "PATCH"})


class ExecutionMode(enum.Enum):
    """Where an extension's ``register()`` and handlers execute.

    ``OUT_OF_PROCESS`` is the short-lived, per-call child used for isolated-dep /
    native-marker extensions (``OUROBOROS_EXTENSION_PROCESS_CHILD == "1"``). Such a
    child cannot host a persistent in-process subscription or asyncio task, so two
    capabilities are unavailable there; a manifest-declared ``companion_process``
    (host-spawned and supervised) is the supported alternative for long-running
    work and host-event subscription.
    """

    IN_PROCESS = "in_process"
    OUT_OF_PROCESS = "out_of_process"


# PluginAPI side-effect/registration methods governed by the execution-mode matrix.
# Kept in lockstep with the guarded/cataloged surface in extension_loader; the union
# of this set and ALWAYS_AVAILABLE_CAPABILITIES must cover the whole PluginAPI
# surface (pinned by tests/test_oop_extension_parity.py) so a new method cannot be
# added without classifying its out-of-process availability.
MATRIX_CAPABILITIES: frozenset[str] = frozenset({
    "register_tool", "register_route", "register_ws_handler", "register_ui_tab",
    "register_settings_section", "send_ws_message", "register_companion_process",
    "register_supervised_task", "subscribe_event", "on_unload",
})

# Runtime-access/introspection methods always available in every execution mode
# (the complement of MATRIX_CAPABILITIES over the PluginAPI surface).
ALWAYS_AVAILABLE_CAPABILITIES: frozenset[str] = frozenset({
    "log", "get_settings", "get_state_dir", "skill_job_dir",
    "get_skill_token", "get_runtime_info",
})

# Capabilities a short-lived OUT_OF_PROCESS child cannot use directly: a persistent
# host-event subscription and an in-process supervised asyncio task have no meaning
# in a per-call child. Long-running work, host events, and supervised loops belong
# in a manifest-declared companion_process.
OUT_OF_PROCESS_UNAVAILABLE_CAPABILITIES: frozenset[str] = frozenset({
    "subscribe_event",
    "register_supervised_task",
})


def capability_available(capability: str, mode: ExecutionMode) -> bool:
    """Return whether a PluginAPI capability may be used in ``mode``."""
    if mode is ExecutionMode.OUT_OF_PROCESS:
        return capability not in OUT_OF_PROCESS_UNAVAILABLE_CAPABILITIES
    return True


def available_capabilities(mode: ExecutionMode) -> frozenset[str]:
    """Return the matrix capabilities available in ``mode`` (for negotiation)."""
    return frozenset(c for c in MATRIX_CAPABILITIES if capability_available(c, mode))


class RuntimeInfo(TypedDict):
    """Frozen read-only runtime snapshot returned by ``get_runtime_info``.

    Additive only within a PluginAPI generation; the key set is part of the
    versioned surface fingerprint below, so a silent shape change without a
    version bump fails closed.
    """

    runtime_mode: str
    app_version: str
    data_dir: str
    skill_dir: str
    state_dir: str
    server_port: int
    execution_mode: str
    capabilities: List[str]
    # Negotiated generation served to THIS extension (LEGACY for grandfathered
    # field-less payloads), not merely the host's own PLUGIN_API_VERSION.
    plugin_api_version: str


# --- ABI-1: manifest negotiation, surface fingerprints, admission ---

_PLUGIN_API_VERSION_RE = re.compile(r"^([0-9]{1,4})\.([0-9]{1,4})$")


@dataclass(frozen=True)
class PluginAPINegotiation:
    """Typed outcome of negotiating one manifest against this host."""

    ok: bool
    generation: str
    declared: bool
    capabilities: tuple[str, ...] = ()
    error: str = ""


def manifest_plugin_api_field(manifest: Any) -> Optional[Dict[str, Any]]:
    """Return the normalized manifest ``plugin_api`` mapping, or None (absent)."""
    raw = getattr(manifest, "plugin_api", None)
    return raw if isinstance(raw, dict) else None


def api_generation(manifest: Any) -> str:
    """The PluginAPI generation a payload binds against: the declared manifest
    field, or — absent ≡ by construction — ``LEGACY_PLUGIN_API_GENERATION``."""
    field = manifest_plugin_api_field(manifest)
    if field is None:
        return LEGACY_PLUGIN_API_GENERATION
    return str(field.get("version") or "").strip() or LEGACY_PLUGIN_API_GENERATION


def plugin_api_surface_fingerprint() -> str:
    """Canonical digest of the live versioned PluginAPI surface.

    The digest covers what an extension author can actually collide with: each
    public method's NAME **and full signature text**, and each ``RuntimeInfo``
    key **and its annotation**. Names and keys alone were not the surface this
    module's header promises — reordering a parameter, dropping a default,
    tightening ``Sequence[str]`` to ``list[str]`` or turning ``server_port``
    from ``int`` into ``str`` all break a reviewed ``register()`` while leaving
    the recorded digest byte-identical, which is the one thing the
    fail-closed-both-directions contract below is supposed to make impossible.

    ``from __future__ import annotations`` is active here, so both halves are
    the SOURCE text of the annotation (``ForwardRef.__forward_arg__`` for the
    TypedDict, the stringified annotations inside ``str(signature)``): stable
    across interpreter versions, and exactly what a reviewer reads.
    """
    methods = sorted(
        m for m in dir(PluginAPI)
        if not m.startswith("_") and callable(getattr(PluginAPI, m, None))
    )
    payload = json.dumps({
        "version": PLUGIN_API_VERSION,
        "methods": {m: str(inspect.signature(getattr(PluginAPI, m))) for m in methods},
        "matrix": sorted(MATRIX_CAPABILITIES),
        "always": sorted(ALWAYS_AVAILABLE_CAPABILITIES),
        "out_of_process_unavailable": sorted(OUT_OF_PROCESS_UNAVAILABLE_CAPABILITIES),
        "runtime_info": {
            key: getattr(annotation, "__forward_arg__", None) or str(annotation)
            for key, annotation in RuntimeInfo.__annotations__.items()
        },
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Fingerprint of the contract surface, keyed BY VERSION — fail-closed in both
# directions: a declared version missing from this table is refused, and a
# live surface that no longer matches its own version's fingerprint (a shape
# change without a version bump) refuses negotiation until the version is
# bumped and the new fingerprint recorded here.
PLUGIN_API_SURFACE_FINGERPRINTS: Dict[str, str] = {
    "2.0": "03fabdf4334e6b2bde217b4cb83a80faaebc773ddce870546dd2108b75de17ca",
}


def negotiate_plugin_api(
    manifest: Any,
    *,
    mode: ExecutionMode = ExecutionMode.IN_PROCESS,
) -> PluginAPINegotiation:
    """Negotiate one manifest against this host's PluginAPI (full contract).

    Absent field -> the LEGACY generation loads (the grandfather's teeth live
    at NEW-PASS issuance, not here). A declared field is held to the full
    contract: exact ``major.minor`` shape, major strictly equal, declared
    minor as the required MINIMUM, the version present in the fingerprint
    table, and every requested capability inside the closed set and available
    in the actual execution mode. Refusals are typed and educational.
    """
    live = plugin_api_surface_fingerprint()
    recorded = PLUGIN_API_SURFACE_FINGERPRINTS.get(PLUGIN_API_VERSION)
    if recorded != live:
        return PluginAPINegotiation(
            ok=False, generation="", declared=True,
            error=(
                f"host PluginAPI surface drifted from its recorded {PLUGIN_API_VERSION} "
                "fingerprint (a surface change without a version bump); refusing to "
                "negotiate until the version is bumped and the fingerprint re-recorded"
            ),
        )
    field = manifest_plugin_api_field(manifest)
    if field is None:
        return PluginAPINegotiation(ok=True, generation=LEGACY_PLUGIN_API_GENERATION, declared=False)
    declared_version = str(field.get("version") or "").strip()
    match = _PLUGIN_API_VERSION_RE.match(declared_version)
    if match is None:
        return PluginAPINegotiation(
            ok=False, generation="", declared=True,
            error=(
                f"plugin_api version {declared_version!r} is not 'major.minor' "
                f"(declare e.g. plugin_api: \"{PLUGIN_API_VERSION}\")"
            ),
        )
    host_major, host_minor = (int(part) for part in PLUGIN_API_VERSION.split("."))
    major, minor = int(match.group(1)), int(match.group(2))
    if major != host_major:
        return PluginAPINegotiation(
            ok=False, generation="", declared=True,
            error=(
                f"plugin_api major {major} does not match this host's PluginAPI "
                f"{PLUGIN_API_VERSION} (major is strict). Pre-2.0 payloads must OMIT "
                "the field (they load grandfathered on an existing review PASS); new "
                f"payloads declare plugin_api: \"{PLUGIN_API_VERSION}\""
            ),
        )
    if minor > host_minor:
        return PluginAPINegotiation(
            ok=False, generation="", declared=True,
            error=(
                f"plugin_api {declared_version} requires minimum minor {minor}, but "
                f"this host serves PluginAPI {PLUGIN_API_VERSION}; upgrade Ouroboros "
                "or lower the declared minimum"
            ),
        )
    if declared_version not in PLUGIN_API_SURFACE_FINGERPRINTS:
        return PluginAPINegotiation(
            ok=False, generation="", declared=True,
            error=(
                f"plugin_api {declared_version} has no recorded surface fingerprint "
                f"on this host (known: {sorted(PLUGIN_API_SURFACE_FINGERPRINTS)}); "
                "refusing an unverifiable generation"
            ),
        )
    closed_set = MATRIX_CAPABILITIES | ALWAYS_AVAILABLE_CAPABILITIES
    requested = tuple(
        str(item).strip() for item in (field.get("capabilities") or []) if str(item).strip()
    )
    unknown = sorted(set(requested) - closed_set)
    if unknown:
        return PluginAPINegotiation(
            ok=False, generation="", declared=True,
            error=(
                f"plugin_api requests unknown capabilities {unknown}; the closed "
                f"capability set is {sorted(closed_set)}"
            ),
        )
    unavailable = sorted(
        cap for cap in requested
        if cap in MATRIX_CAPABILITIES and not capability_available(cap, mode)
    )
    if unavailable:
        return PluginAPINegotiation(
            ok=False, generation="", declared=True,
            error=(
                f"plugin_api requires capabilities {unavailable} that are not "
                f"available {mode.value}; declare a manifest companion_process for "
                "long-running work and host-event subscription"
            ),
        )
    return PluginAPINegotiation(
        ok=True, generation=PLUGIN_API_VERSION, declared=True, capabilities=requested,
    )


def extension_new_pass_admission_error(manifest: Any) -> str:
    """Admission predicate for ISSUING a NEW executable review PASS (ABI-1).

    Common to every PASS-minting path (LLM review, owner attestation, native
    seed trust) and deliberately OUTSIDE the deterministic preflight: a
    ``type: extension`` payload without the ``plugin_api`` manifest field is
    refused a NEW PASS (its existing hash-bound PASS keeps loading —
    grandfather), and a declared field must negotiate cleanly. Returns the
    typed refusal, or "" when admissible.
    """
    if manifest is None or not str(getattr(manifest, "type", "") or "") == "extension":
        return ""
    if manifest_plugin_api_field(manifest) is None:
        return (
            "extension manifest declares no plugin_api field; new review PASSes "
            f"require plugin_api: \"{PLUGIN_API_VERSION}\" (an already-reviewed "
            "payload keeps loading on its existing hash-bound PASS)"
        )
    negotiation = negotiate_plugin_api(manifest)
    if not negotiation.ok:
        return negotiation.error
    return ""


@runtime_checkable
class PluginAPI(Protocol):
    """Runtime-checkable ABI exposed to each extension's ``register(api)``."""

    # registration

    def register_tool(
        self,
        name: str,
        handler: Callable[..., str] | Callable[..., Awaitable[str]],
        *,
        description: str,
        schema: Dict[str, Any],
        timeout_sec: int = 60,
    ) -> None:
        """Register a namespaced tool.

        ``name`` is alphanumeric/underscore and <=24 chars. Handlers may be sync
        or async; async handlers run on a helper-thread event loop with timeout.
        """
        ...

    def register_route(
        self,
        path: str,
        handler: Callable[..., Any],
        *,
        methods: Sequence[str] = ("GET",),
    ) -> None:
        """Register ``/api/extensions/<skill>/<path>`` for allowed methods.

        The host owns GET/HEAD for the exact paths ``manifest`` and
        ``settings_section`` and for everything under ``module/`` in that
        namespace, so a skill route registered there is shadowed for GET/HEAD
        (POST and the other methods still reach the skill); use another path.
        """
        ...

    def register_ws_handler(
        self,
        message_type: str,
        handler: Callable[..., Awaitable[Any]] | Callable[..., Any],
    ) -> None:
        """Register a namespaced WS handler; message_type follows tool-name limits."""
        ...

    def register_ui_tab(
        self,
        tab_id: str,
        title: str,
        *,
        icon: str = "extension",
        render: Dict[str, Any] | None = None,
    ) -> None:
        """Register a Widgets-page UI declaration.

        ``render`` is host-owned declarative UI, iframe, or a reviewed sandboxed
        module served only for a live tab and bridged to this skill's route prefix.
        Same-origin SPA modules are outside this contract.

        ``render.start`` declares the card's launch policy: ``"auto"`` starts when
        the Widgets page is shown and stops when the owner leaves; ``"manual"``
        shows a Start button and leaving the page is an ordered Stop; ``"retain"``
        starts on the first Widgets visit and keeps running while the owner is on
        other pages until Stop, skill disable/unload/delete, app reload, or closing
        Ouroboros — it never outlives the window; a same-SHA server reconnect keeps
        a retained frame whose skill is live again with the same revision, while a
        changed revision is re-mounted at once when Widgets is visible and at the
        next Widgets entry while it is hidden. Defaults:
        ``module``/``iframe`` → ``"manual"``; ``declarative`` → ``"auto"`` and
        accepts nothing else. The validator
        (``ouroboros/extension_ui_validation.py::WIDGET_START_MODES``) is the SSOT
        and fills the default into the stored declaration; the owner's per-card
        override (``ui_preferences.widget_start_mode``) always wins.
        """
        ...

    def send_ws_message(self, message_type: str, data: Dict[str, Any]) -> None:
        """Best-effort broadcast of a namespaced extension WS event."""
        ...

    def register_settings_section(
        self,
        section_id: str,
        title: str,
        *,
        schema: Dict[str, Any],
    ) -> None:
        """Register a host-rendered Settings panel with no extension JS."""
        ...

    def register_supervised_task(
        self,
        name: str,
        factory: Callable[[], Awaitable[None]],
        *,
        restart_policy: str = "on_failure",
        max_restarts: int = 5,
        backoff_seconds: float = 2.0,
    ) -> None:
        """Register an enabled-state-bound in-process asyncio task."""
        ...

    def register_companion_process(
        self,
        name: str,
    ) -> None:
        """Register a companion subprocess declared in the reviewed manifest."""
        ...

    def subscribe_event(
        self,
        topic: str,
        handler: Callable[[Dict[str, Any]], Awaitable[None] | None],
    ) -> str:
        """Subscribe to a manifest-declared event; unload removes subscriptions."""
        ...

    def get_skill_token(self) -> SkillToken:
        """Return the opaque Host Service API token for this skill."""
        ...

    def on_unload(self, callback: Callable[[], Any]) -> None:
        """Register fast, idempotent best-effort cleanup on skill unload."""
        ...

    # runtime access

    def log(self, level: str, message: str, **fields: Any) -> None:
        """Structured log. ``level`` one of ``debug``/``info``/``warning``/``error``."""
        ...

    def get_settings(self, keys: Sequence[str]) -> Dict[str, Any]:
        """Return allowlisted settings; core keys require owner grants."""
        ...

    def get_state_dir(self) -> str:
        """Return the canonical private state dir for reviewed extension state."""
        ...

    def skill_job_dir(self, job_id: str) -> pathlib.Path:
        """Return a sanitized per-job state directory with assets/output/tmp."""
        ...

    def get_runtime_info(self) -> Dict[str, Any]:
        """Return a read-only runtime snapshot; additive within schema v1."""
        ...


class ExtensionRegistrationError(Exception):
    """Raised when a registration violates namespace, permission, or schema."""


def normalize_extension_route_methods(methods: Any, *, subject: str) -> tuple[str, ...]:
    """One route declaration's HTTP methods, normalized against the vocabulary.

    Used by the in-process ``register_route`` AND by the host-side re-check of
    a child catalog's route descriptors, so the vocabulary cannot admit
    out-of-process what it refuses in-process. A bare string is a single
    method; case and surrounding space are normalized, order is preserved and
    duplicates collapse. ``subject`` names the route in the refusal.
    """
    declared = (methods,) if isinstance(methods, str) else (methods or ())
    normalized = tuple(dict.fromkeys(
        str(method).strip().upper() for method in declared if str(method).strip()
    ))
    if not normalized:
        raise ExtensionRegistrationError(f"{subject} methods must be non-empty")
    unsupported = [m for m in normalized if m not in VALID_EXTENSION_ROUTE_METHODS]
    if unsupported:
        raise ExtensionRegistrationError(
            f"{subject} methods {unsupported!r} are unsupported; "
            f"expected subset of {sorted(VALID_EXTENSION_ROUTE_METHODS)}"
        )
    return normalized


__all__ = [
    "PluginAPI", "ExtensionRegistrationError", "FORBIDDEN_SKILL_SETTINGS",
    "PLUGIN_API_VERSION", "LEGACY_PLUGIN_API_GENERATION",
    "PLUGIN_API_SURFACE_FINGERPRINTS", "PluginAPINegotiation", "RuntimeInfo",
    "VALID_EXTENSION_PERMISSIONS", "VALID_EXTENSION_ROUTE_METHODS",
    "normalize_extension_route_methods",
    "ExecutionMode", "MATRIX_CAPABILITIES", "ALWAYS_AVAILABLE_CAPABILITIES",
    "OUT_OF_PROCESS_UNAVAILABLE_CAPABILITIES",
    "api_generation", "capability_available", "available_capabilities",
    "extension_new_pass_admission_error", "manifest_plugin_api_field",
    "negotiate_plugin_api", "plugin_api_surface_fingerprint",
]
