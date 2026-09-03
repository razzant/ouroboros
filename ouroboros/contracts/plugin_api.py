"""Frozen PluginAPI contract for extension skills.

``plugin.py`` exposes ``register(api: PluginAPI)`` and may use only this ABI.
Signature or permission tightening requires a schema/version bump; additive
optional methods are allowed when documented here and pinned by contract tests.
Registrations are declarative and are torn down when a skill unloads.
"""

from __future__ import annotations

import enum
import pathlib
from typing import Any, Awaitable, Callable, Dict, Protocol, Sequence, runtime_checkable

from ouroboros.skill_token import SkillToken

# 1.4: additive/widening — reviewed transport skills may submit authenticated
# non-owner presence events through the Host Service's positive capability ceiling.
PLUGIN_API_VERSION = "1.4"


# Core settings keys require explicit content-hash-bound owner grants.
FORBIDDEN_SKILL_SETTINGS: frozenset[str] = frozenset({
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_COMPATIBLE_API_KEY",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY", "GIGACHAT_CREDENTIALS", "GIGACHAT_PASSWORD",
    "ANTHROPIC_API_KEY", "MINIMAX_API_KEY", "GITHUB_TOKEN",
    "OUROBOROS_NETWORK_PASSWORD",
})
# Backwards-compatible alias for the extension name.
FORBIDDEN_EXTENSION_SETTINGS: frozenset[str] = FORBIDDEN_SKILL_SETTINGS


# Keep in sync with skill_manifest.VALID_SKILL_PERMISSIONS.
VALID_EXTENSION_PERMISSIONS: frozenset[str] = frozenset({
    "net", "fs", "subprocess", "widget", "ws_handler", "route", "tool",
    "read_settings", "companion_process", "supervised_task", "subscribe_event", "inject_chat",
    "presence",
})

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


__all__ = [
    "PluginAPI", "ExtensionRegistrationError", "FORBIDDEN_SKILL_SETTINGS",
    "FORBIDDEN_EXTENSION_SETTINGS", "PLUGIN_API_VERSION", "VALID_EXTENSION_PERMISSIONS",
    "VALID_EXTENSION_ROUTE_METHODS",
    "ExecutionMode", "MATRIX_CAPABILITIES", "ALWAYS_AVAILABLE_CAPABILITIES",
    "OUT_OF_PROCESS_UNAVAILABLE_CAPABILITIES",
    "capability_available", "available_capabilities",
]
