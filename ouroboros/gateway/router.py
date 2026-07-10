"""Gateway route collection.

This module is the Starlette route single source of truth for the browser
boundary. Domain handlers live in sibling gateway modules; a few legacy
server-owned handlers are passed in while the migration pays down ``server.py``.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Mapping
from typing import Any

from starlette.routing import BaseRoute, Route, WebSocketRoute


def collect_routes(
    *,
    data_dir: pathlib.Path,
    settings_handlers: Mapping[str, Callable[..., Any]] | None = None,
) -> list[BaseRoute]:
    """Return web-facing Starlette routes.

    New endpoint families should live directly under ``ouroboros.gateway`` and
    be imported here.
    """
    from ouroboros.gateway.extensions import (
        api_extension_dispatch,
        api_extension_manifest,
        api_extension_module,
        api_extension_settings_section,
        api_extensions_index,
        api_skill_daemons,
        api_skill_delete,
        api_skill_grants,
        api_owner_skill_attest_review,
        api_skill_lifecycle_queue,
        api_skill_reconcile,
        api_skill_review,
        api_skill_toggle,
    )
    from ouroboros.gateway.files import (
        api_chat_upload,
        api_chat_upload_delete,
        file_browser_routes,
    )
    from ouroboros.gateway.history import make_chat_history_endpoint, make_cost_breakdown_endpoint
    from ouroboros.gateway.logs import api_logs_tail
    from ouroboros.gateway.marketplace import (
        api_marketplace_info,
        api_marketplace_install,
        api_marketplace_installed,
        api_marketplace_preview,
        api_marketplace_search,
        api_marketplace_uninstall,
        api_marketplace_update,
        api_ouroboroshub_catalog,
        api_ouroboroshub_install,
        api_ouroboroshub_installed,
        api_ouroboroshub_preview,
        api_ouroboroshub_uninstall,
        api_ouroboroshub_update,
    )
    from ouroboros.gateway.mcp import api_mcp_refresh, api_mcp_status, api_mcp_test
    from ouroboros.gateway.models import (
        api_local_model_install_runtime,
        api_local_model_start,
        api_local_model_status,
        api_local_model_stop,
        api_local_model_test,
        api_model_catalog,
        api_openai_compatible_models,
    )
    from ouroboros.gateway.schedules import (
        api_schedules_delete,
        api_schedules_list,
        api_schedules_upsert,
    )
    from ouroboros.gateway.control import (
        api_command,
        api_evolution_data,
        api_git_log,
        api_git_promote,
        api_git_rollback,
        api_reset,
        api_update_apply,
        api_update_check,
        api_update_preflight,
        api_update_status,
    )
    from ouroboros.gateway.projects import (
        api_fs_dirs,
        api_project_delete,
        api_project_from_task,
        api_project_update,
        api_projects_create,
        api_projects_list,
    )
    from ouroboros.gateway.state import api_health, api_state
    from ouroboros.gateway.tasks import (
        api_task_artifact,
        api_task_cancel,
        api_task_events,
        api_task_get,
        api_tasks_create,
        api_tasks_list,
    )
    from ouroboros.gateway.ui_preferences import (
        api_ui_preferences_get,
        api_ui_preferences_post,
    )
    from ouroboros.gateway.settings import (
        api_claude_code_install,
        api_claude_code_status,
        api_acknowledge_capability,
        api_onboarding,
        api_owner_auto_grant,
        api_owner_context_mode,
        api_owner_safety_mode,
        api_owner_scope_review_floor,
        api_owner_runtime_mode,
        api_settings_get,
        api_settings_post,
    )
    from ouroboros.gateway.ws import ws_endpoint

    settings_handlers = settings_handlers or {}
    settings_get = settings_handlers.get("api_settings_get", api_settings_get)
    settings_post = settings_handlers.get("api_settings_post", api_settings_post)
    onboarding = settings_handlers.get("api_onboarding", api_onboarding)
    claude_status = settings_handlers.get("api_claude_code_status", api_claude_code_status)
    claude_install = settings_handlers.get("api_claude_code_install", api_claude_code_install)

    routes: list[BaseRoute] = [
        Route("/api/health", endpoint=api_health),
        Route("/api/state", endpoint=api_state),
        Route("/api/extensions", endpoint=api_extensions_index, methods=["GET"]),
        Route("/api/extensions/{skill}/manifest", endpoint=api_extension_manifest, methods=["GET"]),
        Route("/api/extensions/{skill}/module/{entry}", endpoint=api_extension_module, methods=["GET"]),
        Route(
            "/api/extensions/{skill}/settings_section",
            endpoint=api_extension_settings_section,
            methods=["GET"],
        ),
        Route(
            "/api/extensions/{skill}/{rest:path}",
            endpoint=api_extension_dispatch,
            methods=["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH"],
        ),
        Route("/api/skills/{skill}/toggle", endpoint=api_skill_toggle, methods=["POST"]),
        Route("/api/skills/{skill}/delete", endpoint=api_skill_delete, methods=["POST"]),
        Route("/api/skills/daemons", endpoint=api_skill_daemons, methods=["GET"]),
        Route("/api/skills/lifecycle-queue", endpoint=api_skill_lifecycle_queue, methods=["GET"]),
        Route("/api/skills/{skill}/review", endpoint=api_skill_review, methods=["POST"]),
        Route("/api/owner/skills/{skill}/attest-review", endpoint=api_owner_skill_attest_review, methods=["POST"]),
        Route("/api/skills/{skill}/grants", endpoint=api_skill_grants, methods=["POST"]),
        Route("/api/skills/{skill}/reconcile", endpoint=api_skill_reconcile, methods=["POST"]),
        Route("/api/marketplace/clawhub/search", endpoint=api_marketplace_search, methods=["GET"]),
        Route("/api/marketplace/clawhub/installed", endpoint=api_marketplace_installed, methods=["GET"]),
        Route("/api/marketplace/clawhub/info/{slug:path}", endpoint=api_marketplace_info, methods=["GET"]),
        Route("/api/marketplace/clawhub/preview/{slug:path}", endpoint=api_marketplace_preview, methods=["GET"]),
        Route("/api/marketplace/clawhub/install", endpoint=api_marketplace_install, methods=["POST"]),
        Route("/api/marketplace/clawhub/update/{name}", endpoint=api_marketplace_update, methods=["POST"]),
        Route("/api/marketplace/clawhub/uninstall/{name}", endpoint=api_marketplace_uninstall, methods=["POST"]),
        Route("/api/marketplace/ouroboroshub/catalog", endpoint=api_ouroboroshub_catalog, methods=["GET"]),
        Route("/api/marketplace/ouroboroshub/installed", endpoint=api_ouroboroshub_installed, methods=["GET"]),
        Route(
            "/api/marketplace/ouroboroshub/preview/{slug:path}",
            endpoint=api_ouroboroshub_preview,
            methods=["GET"],
        ),
        Route("/api/marketplace/ouroboroshub/install", endpoint=api_ouroboroshub_install, methods=["POST"]),
        Route("/api/marketplace/ouroboroshub/update/{name}", endpoint=api_ouroboroshub_update, methods=["POST"]),
        Route(
            "/api/marketplace/ouroboroshub/uninstall/{name}",
            endpoint=api_ouroboroshub_uninstall,
            methods=["POST"],
        ),
        *file_browser_routes(),
        Route("/api/onboarding", endpoint=onboarding),
        Route("/api/claude-code/status", endpoint=claude_status),
        Route(
            "/api/claude-code/install",
            endpoint=claude_install,
            methods=["POST"],
        ),
        Route("/api/settings", endpoint=settings_get, methods=["GET"]),
        Route("/api/settings", endpoint=settings_post, methods=["POST"]),
        Route("/api/ui/preferences", endpoint=api_ui_preferences_get, methods=["GET"]),
        Route("/api/ui/preferences", endpoint=api_ui_preferences_post, methods=["POST"]),
        Route("/api/owner/runtime-mode", endpoint=api_owner_runtime_mode, methods=["POST"]),
        Route("/api/owner/auto-grant", endpoint=api_owner_auto_grant, methods=["POST"]),
        Route("/api/owner/context-mode", endpoint=api_owner_context_mode, methods=["POST"]),
        Route("/api/owner/scope-review-floor", endpoint=api_owner_scope_review_floor, methods=["POST"]),
        Route("/api/owner/safety-mode", endpoint=api_owner_safety_mode, methods=["POST"]),
        Route("/api/owner/capability-ack", endpoint=api_acknowledge_capability, methods=["POST"]),
        Route("/api/model-catalog", endpoint=api_model_catalog),
        Route("/api/projects", endpoint=api_projects_list, methods=["GET"]),
        Route("/api/projects", endpoint=api_projects_create, methods=["POST"]),
        Route("/api/projects/from-task", endpoint=api_project_from_task, methods=["POST"]),
        Route("/api/projects/{project_id}/update", endpoint=api_project_update, methods=["POST"]),
        Route("/api/projects/{project_id}/delete", endpoint=api_project_delete, methods=["POST"]),
        Route("/api/fs/dirs", endpoint=api_fs_dirs, methods=["GET"]),
        Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"]),
        Route("/api/tasks", endpoint=api_tasks_list, methods=["GET"]),
        Route("/api/tasks/{task_id}/artifacts/{name}", endpoint=api_task_artifact, methods=["GET"]),
        Route("/api/tasks/{task_id}", endpoint=api_task_get, methods=["GET"]),
        Route("/api/tasks/{task_id}/events", endpoint=api_task_events, methods=["GET"]),
        Route("/api/tasks/{task_id}/cancel", endpoint=api_task_cancel, methods=["POST"]),
        Route("/api/schedules", endpoint=api_schedules_list, methods=["GET"]),
        Route("/api/schedules", endpoint=api_schedules_upsert, methods=["POST"]),
        Route("/api/schedules/{schedule_id}", endpoint=api_schedules_delete, methods=["DELETE"]),
        Route("/api/command", endpoint=api_command, methods=["POST"]),
        Route("/api/reset", endpoint=api_reset, methods=["POST"]),
        Route("/api/git/log", endpoint=api_git_log),
        Route("/api/git/rollback", endpoint=api_git_rollback, methods=["POST"]),
        Route("/api/git/promote", endpoint=api_git_promote, methods=["POST"]),
        Route("/api/update/status", endpoint=api_update_status),
        Route("/api/update/check", endpoint=api_update_check, methods=["POST"]),
        Route("/api/update/preflight", endpoint=api_update_preflight, methods=["POST"]),
        Route("/api/update/apply", endpoint=api_update_apply, methods=["POST"]),
        Route("/api/cost-breakdown", endpoint=make_cost_breakdown_endpoint(data_dir)),
        Route("/api/evolution-data", endpoint=api_evolution_data),
        Route("/api/chat/history", endpoint=make_chat_history_endpoint(data_dir)),
        Route("/api/logs/{name}", endpoint=api_logs_tail, methods=["GET"]),
        Route("/api/chat/upload", endpoint=api_chat_upload, methods=["POST"]),
        Route("/api/chat/upload", endpoint=api_chat_upload_delete, methods=["DELETE"]),
        Route("/api/openai-compatible/models", endpoint=api_openai_compatible_models, methods=["POST"]),
        Route("/api/local-model/start", endpoint=api_local_model_start, methods=["POST"]),
        Route("/api/local-model/stop", endpoint=api_local_model_stop, methods=["POST"]),
        Route("/api/local-model/status", endpoint=api_local_model_status),
        Route("/api/local-model/test", endpoint=api_local_model_test, methods=["POST"]),
        Route(
            "/api/local-model/install-runtime",
            endpoint=api_local_model_install_runtime,
            methods=["POST"],
        ),
        Route("/api/mcp/status", endpoint=api_mcp_status, methods=["GET"]),
        Route("/api/mcp/refresh", endpoint=api_mcp_refresh, methods=["POST"]),
        Route("/api/mcp/test", endpoint=api_mcp_test, methods=["POST"]),
        WebSocketRoute("/ws", endpoint=ws_endpoint),
    ]
    return routes


__all__ = ["collect_routes"]
