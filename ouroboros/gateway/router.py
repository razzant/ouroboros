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


def _ui_client_routes() -> list[BaseRoute]:
    """Installation-wide owner UI preferences and the install's
    interface language with its translation memory (``/api/ui/*``)."""
    from ouroboros.gateway.ui_i18n import (
        api_ui_i18n_export_get,
        api_ui_i18n_get,
        api_ui_i18n_import_post,
        api_ui_i18n_language_post,
        api_ui_i18n_missing_post,
        api_ui_i18n_regenerate_post,
    )
    from ouroboros.gateway.ui_preferences import api_ui_preferences_get, api_ui_preferences_post
    from ouroboros.gateway.contributions import api_ui_contributions
    from ouroboros.gateway.ui_holds import api_ui_holds_post

    return [
        Route("/api/ui/contributions", endpoint=api_ui_contributions, methods=["GET"]),
        Route("/api/ui/preferences", endpoint=api_ui_preferences_get, methods=["GET"]),
        Route("/api/ui/preferences", endpoint=api_ui_preferences_post, methods=["POST"]),
        Route("/api/ui/holds", endpoint=api_ui_holds_post, methods=["POST"]),
        Route("/api/ui/i18n", endpoint=api_ui_i18n_get, methods=["GET"]),
        Route("/api/ui/i18n/language", endpoint=api_ui_i18n_language_post, methods=["POST"]),
        Route("/api/ui/i18n/missing", endpoint=api_ui_i18n_missing_post, methods=["POST"]),
        Route("/api/ui/i18n/import", endpoint=api_ui_i18n_import_post, methods=["POST"]),
        Route("/api/ui/i18n/export", endpoint=api_ui_i18n_export_get, methods=["GET"]),
        Route("/api/ui/i18n/regenerate", endpoint=api_ui_i18n_regenerate_post, methods=["POST"]),
    ]


def _owner_answer_routes() -> list[BaseRoute]:
    """The owner's answers: decision cards (``/api/decisions``, every family) and the
    living cards Ouroboros shows in its messages (``/api/cards/*``: reads, pins)."""
    from ouroboros.gateway.cards import card_routes
    from ouroboros.gateway.task_decision import api_decision_answer

    return [Route("/api/decisions", endpoint=api_decision_answer, methods=["POST"]), *card_routes()]


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
        api_skill_review_history_detail,
        api_skill_toggle,
    )
    from ouroboros.gateway.widgets import api_widget_fault_post, api_widgets
    from ouroboros.gateway.widget_assets import api_widget_asset_lease, api_widget_bridge_script
    from ouroboros.gateway.contributions import api_extension_host_asset
    from ouroboros.gateway.files import (
        api_chat_upload,
        api_chat_upload_delete,
        file_browser_routes,
    )
    from ouroboros.gateway.history import make_chat_history_endpoint, make_cost_breakdown_endpoint
    from ouroboros.gateway.logs import api_logs_tail
    from ouroboros.gateway.storage import storage_routes
    from ouroboros.gateway.marketplace import (
        api_hub_installed,
        api_marketplace_info,
        api_marketplace_install,
        api_marketplace_installed,
        api_marketplace_preview,
        api_marketplace_search,
        api_marketplace_uninstall,
        api_marketplace_update,
        api_skill_publication_clear,
    )
    from ouroboros.gateway.skill_my_skills import skill_source_routes
    from ouroboros.gateway.claudexor_accounts import (
        api_claudexor_credential_profile,
        api_claudexor_login,
        api_claudexor_login_job,
        api_claudexor_login_job_reconcile,
        api_claudexor_status,
        api_claudexor_wake,
    )
    from ouroboros.gateway.claudexor_quota import api_claudexor_account_reset, api_claudexor_quota_refresh
    from ouroboros.gateway.harness_maintenance import harness_maintenance_routes
    from ouroboros.gateway.onboarding import (
        api_onboarding_complete,
        api_onboarding_subagents_preview,
    )
    from ouroboros.gateway.settings import api_review_pool
    from ouroboros.gateway.mcp import mcp_routes
    from ouroboros.gateway.models import (
        api_local_model_install_runtime,
        api_local_model_start,
        api_local_model_status,
        api_local_model_stop,
        api_local_model_test,
        api_model_catalog,
        api_openai_compatible_models,
        api_provider_test,
    )
    from ouroboros.gateway.schedules import (
        api_schedules_action, api_schedules_delete, api_schedules_list, api_schedules_upsert,
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
        api_update_description,
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
    from ouroboros.gateway.state import api_consciousness, api_contract, api_health, api_state
    from ouroboros.gateway.skill_publish import skill_publish_routes
    from ouroboros.gateway.task_reviews import task_review_routes
    from ouroboros.gateway.tasks import (
        api_task_artifact,
        api_task_cancel,
    api_task_hurry, owner_tree_control_routes,
        api_task_resume,
        api_task_events,
        api_task_get,
        api_tasks_create,
        api_tasks_list,
    )
    from ouroboros.gateway import desktop_autostart as desktop_host
    from ouroboros.gateway.onboarding_host import api_onboarding_attempt, api_onboarding_bootstrap, onboarding_page
    from ouroboros.gateway.owner_effort import api_owner_chat_model, api_owner_effort_range, api_owner_subagents_mode
    from ouroboros.gateway.safety_settings import api_owner_safety_rules, api_owner_safety_model, api_owner_safety_mode
    from ouroboros.gateway.settings import (
        api_acknowledge_capability,
        api_onboarding,
        api_owner_auto_grant,
        api_owner_context_mode,
        api_owner_runtime_mode,
        api_settings_get,
        api_settings_post,
    )
    from ouroboros.gateway.settings_secrets import api_settings_secret
    from ouroboros.gateway.places_api import places_routes
    from ouroboros.gateway.ws import ws_endpoint
    from ouroboros.gateway.call_ws import call_routes
    settings_handlers = settings_handlers or {}
    settings_get = settings_handlers.get("api_settings_get", api_settings_get)
    settings_post = settings_handlers.get("api_settings_post", api_settings_post)
    onboarding = settings_handlers.get("api_onboarding", api_onboarding)

    routes: list[BaseRoute] = [
        *call_routes(),
        Route("/api/health", endpoint=api_health),
        Route("/api/contract", endpoint=api_contract, methods=["GET"]),
        Route("/api/state", endpoint=api_state),
        Route("/api/consciousness", endpoint=api_consciousness, methods=["GET"]),
        Route("/api/widgets", endpoint=api_widgets, methods=["GET"]),
        Route("/api/widgets/fault", endpoint=api_widget_fault_post, methods=["POST"]),
        Route("/api/widget-bridge.js", endpoint=api_widget_bridge_script, methods=["GET"]),
        Route("/api/widgets/asset-leases", endpoint=api_widget_asset_lease, methods=["POST"]),
        Route("/api/widgets/asset-leases/{op}", endpoint=api_widget_asset_lease, methods=["POST"]),
        Route("/api/extensions", endpoint=api_extensions_index, methods=["GET"]),
        Route("/api/extensions/{skill}/manifest", endpoint=api_extension_manifest, methods=["GET"]),
        Route("/api/extensions/{skill}/module/{entry:path}", endpoint=api_extension_module, methods=["GET"]),
        Route("/api/extensions/{skill}/ui/{revision}/{path:path}", endpoint=api_extension_host_asset, methods=["GET"]),
        Route("/api/extensions/{skill}/settings_section", endpoint=api_extension_settings_section, methods=["GET"]),
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
        *skill_publish_routes(),
        Route(
            "/api/skills/{skill}/review-history/{job_id}",
            endpoint=api_skill_review_history_detail,
            methods=["GET"],
        ),
        Route("/api/owner/skills/{skill}/attest-review", endpoint=api_owner_skill_attest_review, methods=["POST"]),
        *places_routes(),
        Route("/api/skills/{skill}/grants", endpoint=api_skill_grants, methods=["POST"]),
        Route("/api/skills/{skill}/reconcile", endpoint=api_skill_reconcile, methods=["POST"]),
        Route("/api/marketplace/clawhub/search", endpoint=api_marketplace_search, methods=["GET"]),
        Route("/api/marketplace/clawhub/installed", endpoint=api_marketplace_installed, methods=["GET"]),
        Route("/api/marketplace/ouroboroshub/installed", endpoint=api_hub_installed, methods=["GET"]),
        Route("/api/marketplace/clawhub/info/{slug:path}", endpoint=api_marketplace_info, methods=["GET"]),
        Route("/api/marketplace/clawhub/preview/{slug:path}", endpoint=api_marketplace_preview, methods=["GET"]),
        Route("/api/marketplace/clawhub/install", endpoint=api_marketplace_install, methods=["POST"]),
        Route("/api/marketplace/clawhub/update/{name}", endpoint=api_marketplace_update, methods=["POST"]),
        Route("/api/marketplace/clawhub/uninstall/{name}", endpoint=api_marketplace_uninstall, methods=["POST"]),
        Route("/api/skills/{skill}/publication/clear", endpoint=api_skill_publication_clear, methods=["POST"]),
        *skill_source_routes(),
        *file_browser_routes(),
        # The onboarding wizard PAGE (one host for desktop/browser/overlay); the
        # /api/onboarding sibling below stays the readiness probe + legacy body.
        Route("/onboarding", endpoint=onboarding_page, methods=["GET"]),
        Route("/api/onboarding", endpoint=onboarding),
        # The same wizard bootstrap as JSON, for a client that renders its own setup.
        Route("/api/onboarding/bootstrap", endpoint=api_onboarding_bootstrap, methods=["GET"]),
        # ONE atomic owner-scoped completion (D-8): replaces the wizard's old
        # POST /api/settings + POST /api/owner/runtime-mode pair.
        Route("/api/onboarding/subagents/preview", endpoint=api_onboarding_subagents_preview, methods=["POST"]),
        Route("/api/onboarding/complete", endpoint=api_onboarding_complete, methods=["POST"]),
        # Reads back one completion attempt by the id its client sent; writes nothing.
        Route("/api/onboarding/complete", endpoint=api_onboarding_attempt, methods=["GET"]),
        Route("/api/settings", endpoint=settings_get, methods=["GET"]),
        Route("/api/settings", endpoint=settings_post, methods=["POST"]),
        Route("/api/settings/secret", endpoint=api_settings_secret, methods=["POST"]),
        *_ui_client_routes(),
        Route("/api/desktop/autostart", endpoint=desktop_host.api_desktop_autostart_get, methods=["GET"]),
        Route("/api/desktop/autostart", endpoint=desktop_host.api_desktop_autostart_post, methods=["POST"]),
        Route("/api/desktop/background", endpoint=desktop_host.api_desktop_background_get, methods=["GET"]),
        Route("/api/desktop/background", endpoint=desktop_host.api_desktop_background_post, methods=["POST"]),
        Route("/api/owner/runtime-mode", endpoint=api_owner_runtime_mode, methods=["POST"]),
        Route("/api/owner/auto-grant", endpoint=api_owner_auto_grant, methods=["POST"]),
        Route("/api/owner/context-mode", endpoint=api_owner_context_mode, methods=["POST"]),
        Route("/api/owner/effort-range", endpoint=api_owner_effort_range, methods=["POST"]),
        Route("/api/owner/subagents-mode", endpoint=api_owner_subagents_mode, methods=["POST"]),
        Route("/api/owner/chat-model", endpoint=api_owner_chat_model, methods=["POST"]),
        Route("/api/owner/safety-mode", endpoint=api_owner_safety_mode, methods=["POST"]),
        Route("/api/owner/safety-rules", endpoint=api_owner_safety_rules, methods=["POST"]),
        Route("/api/owner/safety-model", endpoint=api_owner_safety_model, methods=["POST"]),
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
        Route("/api/tasks/{task_id}/events", endpoint=api_task_events, methods=["GET", "POST"]),
        Route("/api/tasks/{task_id}/cancel", endpoint=api_task_cancel, methods=["POST"]),
        Route("/api/tasks/{task_id}/hurry", endpoint=api_task_hurry, methods=["POST"]),
        *owner_tree_control_routes(),
        *task_review_routes(),
        Route("/api/tasks/{task_id}/resume", endpoint=api_task_resume, methods=["POST"]),
        *_owner_answer_routes(),
        Route("/api/schedules", endpoint=api_schedules_list, methods=["GET"]),
        Route("/api/schedules", endpoint=api_schedules_upsert, methods=["POST"]),
        Route("/api/schedules/{schedule_id}/action", endpoint=api_schedules_action, methods=["POST"]),
        Route("/api/schedules/{schedule_id}", endpoint=api_schedules_delete, methods=["DELETE"]),
        Route("/api/command", endpoint=api_command, methods=["POST"]),
        Route("/api/reset", endpoint=api_reset, methods=["POST"]),
        Route("/api/git/log", endpoint=api_git_log),
        Route("/api/git/rollback", endpoint=api_git_rollback, methods=["POST"]),
        Route("/api/git/promote", endpoint=api_git_promote, methods=["POST"]),
        Route("/api/update/status", endpoint=api_update_status),
        Route("/api/update/check", endpoint=api_update_check, methods=["POST"]),
        Route("/api/update/preflight", endpoint=api_update_preflight, methods=["POST"]),
        Route("/api/update/description", endpoint=api_update_description, methods=["POST"]),
        Route("/api/update/apply", endpoint=api_update_apply, methods=["POST"]),
        Route("/api/cost-breakdown", endpoint=make_cost_breakdown_endpoint(data_dir)),
        *storage_routes(data_dir),
        Route("/api/evolution-data", endpoint=api_evolution_data),
        Route("/api/chat/history", endpoint=make_chat_history_endpoint(data_dir)),
        Route("/api/logs/{name}", endpoint=api_logs_tail, methods=["GET"]),
        Route("/api/chat/upload", endpoint=api_chat_upload, methods=["POST"]),
        Route("/api/chat/upload", endpoint=api_chat_upload_delete, methods=["DELETE"]),
        Route("/api/openai-compatible/models", endpoint=api_openai_compatible_models, methods=["POST"]),
        Route("/api/providers/test", endpoint=api_provider_test, methods=["POST"]),
        Route("/api/local-model/start", endpoint=api_local_model_start, methods=["POST"]),
        Route("/api/local-model/stop", endpoint=api_local_model_stop, methods=["POST"]),
        Route("/api/local-model/status", endpoint=api_local_model_status),
        Route("/api/local-model/test", endpoint=api_local_model_test, methods=["POST"]),
        Route(
            "/api/local-model/install-runtime",
            endpoint=api_local_model_install_runtime,
            methods=["POST"],
        ),
        *mcp_routes(),
        # Agent accounts (D30): thin proxies of the owned Claudexor daemon's
        # own account surface; zero auth logic on this side.
        Route("/api/review-pool", endpoint=api_review_pool, methods=["GET"]),
        Route("/api/claudexor/status", endpoint=api_claudexor_status, methods=["GET"]),
        Route("/api/claudexor/account-resets", endpoint=api_claudexor_account_reset, methods=["POST"]),
        Route("/api/claudexor/account-resets/{operation_id}", endpoint=api_claudexor_account_reset, methods=["GET"]),
        *harness_maintenance_routes(),
        Route("/api/claudexor/quota/refresh", endpoint=api_claudexor_quota_refresh, methods=["POST"]),
        Route("/api/claudexor/wake", endpoint=api_claudexor_wake, methods=["POST"]),
        Route("/api/claudexor/login", endpoint=api_claudexor_login, methods=["POST"]),
        Route(
            "/api/claudexor/login/{job_id}",
            endpoint=api_claudexor_login_job,
            methods=["GET", "DELETE"],
        ),
        Route(
            "/api/claudexor/login/{job_id}/input",
            endpoint=api_claudexor_login_job,
            methods=["POST"],
        ),
        Route(
            "/api/claudexor/login/{job_id}/reconcile",
            endpoint=api_claudexor_login_job_reconcile,
            methods=["POST"],
        ),
        Route(
            "/api/claudexor/credential-profiles/{harness}/{profile_id}",
            endpoint=api_claudexor_credential_profile,
            # DELETE = forget the named account; PATCH = the Enabled toggle
            # (both thin proxies of the engine's own credential-profile
            # contract; the handler dispatches on the method).
            methods=["DELETE", "PATCH"],
        ),
        WebSocketRoute("/ws", endpoint=ws_endpoint),
    ]
    return routes


__all__ = ["collect_routes"]
