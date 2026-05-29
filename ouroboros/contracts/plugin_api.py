"""Frozen PluginAPI contract for extension skills.

``plugin.py`` exposes ``register(api: PluginAPI)`` and may use only this ABI.
Signature or permission tightening requires a schema/version bump; additive
optional methods are allowed when documented here and pinned by contract tests.
Registrations are declarative and are torn down when a skill unloads.
"""

from __future__ import annotations

import pathlib
from typing import Any, Awaitable, Callable, Dict, List, Protocol, Sequence, runtime_checkable

from ouroboros.skill_token import SkillToken

PLUGIN_API_VERSION = "1.2"


# Core settings keys require explicit content-hash-bound owner grants.
FORBIDDEN_SKILL_SETTINGS: frozenset[str] = frozenset({
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_COMPATIBLE_API_KEY",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY", "GIGACHAT_CREDENTIALS", "GIGACHAT_PASSWORD",
    "ANTHROPIC_API_KEY", "GITHUB_TOKEN",
    "OUROBOROS_NETWORK_PASSWORD",
})
# Backwards-compatible alias for the extension name.
FORBIDDEN_EXTENSION_SETTINGS: frozenset[str] = FORBIDDEN_SKILL_SETTINGS


# Keep in sync with skill_manifest.VALID_SKILL_PERMISSIONS.
VALID_EXTENSION_PERMISSIONS: frozenset[str] = frozenset({
    "net", "fs", "subprocess", "widget", "ws_handler", "route", "tool",
    "read_settings", "companion_process", "supervised_task", "subscribe_event", "inject_chat",
})

VALID_EXTENSION_ROUTE_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "POST", "PUT", "DELETE", "PATCH"})


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
        """Register ``/api/extensions/<skill>/<path>`` for allowed methods."""
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
]
