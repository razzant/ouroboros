"""Descriptive HTTP + WebSocket Gateway Boundary contracts (v1).

TypedDicts document payloads, not runtime validation. Keep discriminating
``type`` keys required; mark genuinely optional fields with ``NotRequired``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # Python 3.11+
    from typing import Literal, NotRequired, Required, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import Literal, NotRequired, Required, TypedDict  # type: ignore[assignment]


class ChatAttachmentInbound(TypedDict, total=False):
    """One uploaded chat attachment reference (file already stored by
    /api/chat/upload under data/uploads/; ``filename`` is the stored
    basename). Image attachments are delivered to vision models as native
    image blocks (v6.26.0)."""

    filename: str
    display_name: str
    mime: str


class ChatInbound(TypedDict):
    """Inbound WS chat message. ``type`` and ``content`` are required."""

    type: Literal["chat"]
    content: str
    sender_session_id: NotRequired[str]
    client_message_id: NotRequired[str]
    force_plan: NotRequired[bool]
    attachments: NotRequired[list]  # list[ChatAttachmentInbound] (additive, v6.26.0)
    # Multi-project (additive, v6.32.0): per-project chat routing. The owner
    # stays user_id 1; chat_id selects the thread, project_id scopes memory.
    chat_id: NotRequired[int]
    project_id: NotRequired[str]


class TaskConstraintInbound(TypedDict, total=False):
    mode: str
    skill_name: str
    payload_root: str
    allow_enable: bool
    allow_review: bool
    extra_allowlist: list[str]


class CommandInbound(TypedDict):
    """Inbound WS command message."""

    type: Literal["command"]
    cmd: str


class ExtensionInbound(TypedDict, total=False):
    """Inbound extension-owned WS message.

    The concrete ``type`` value is provider-safe and namespaced as
    ``ext_<len>_<token>_<message>`` by ``extension_loader``.
    """

    type: str
    data: Any


class TransportMetadata(TypedDict, total=False):
    """Generic external transport provenance for bridge skills."""

    kind: str
    conversation_id: str
    sender_label: str


class ChatOutbound(TypedDict):
    """Outbound WS chat frame."""

    type: Literal["chat"]
    role: Literal["user", "assistant", "system"]
    content: str
    ts: str
    markdown: NotRequired[bool]
    is_progress: NotRequired[bool]
    task_id: NotRequired[str]
    lifecycle: NotRequired[Dict[str, Any]]
    subagent_event: NotRequired[str]
    subagent_task_id: NotRequired[str]
    root_task_id: NotRequired[str]
    parent_task_id: NotRequired[str]
    delegation_role: NotRequired[str]
    subagent_role: NotRequired[str]
    accepted: NotRequired[bool]
    active_subagent_count: NotRequired[int]
    max_active_subagents: NotRequired[int]
    queued_behind_active_cap: NotRequired[bool]
    required_capabilities: NotRequired[list[str]]
    write_surface: NotRequired[str]
    model_lane: NotRequired[str]
    requested_model_lane: NotRequired[str]
    effective_model_lane: NotRequired[str]
    model: NotRequired[str]
    task_group_id: NotRequired[str]
    task_event: NotRequired[str]
    status: NotRequired[str]
    cost_usd: NotRequired[float]
    result: NotRequired[str]
    result_truncated: NotRequired[bool]  # P3: WS preview was capped; fetch full via task id
    trace_summary: NotRequired[str]
    trace_summary_truncated: NotRequired[bool]  # P3: WS preview capped
    error: NotRequired[str]
    artifact_status: NotRequired[str]
    artifact_bundle: NotRequired[Dict[str, Any]]
    outcome_axes: NotRequired[Dict[str, Any]]
    task_contract: NotRequired[Dict[str, Any]]
    reason_code: NotRequired[str]
    review_status: NotRequired[Dict[str, Any]]
    worker_saturation_warning: NotRequired[bool]
    source: NotRequired[str]
    sender_label: NotRequired[str]
    sender_session_id: NotRequired[str]
    client_message_id: NotRequired[str]
    transport: NotRequired[TransportMetadata]
    # Deprecated compatibility field: runtime emits ``transport`` instead.
    telegram_chat_id: NotRequired[int]
    # UI-only system annotation emitted by skill-repair visible commands.
    system_type: NotRequired[str]
    # Present on some transport re-broadcast paths.
    chat_id: NotRequired[int]


class PhotoOutbound(TypedDict):
    """Outbound WS photo frame."""

    type: Literal["photo"]
    role: Literal["user", "assistant"]
    image_base64: str
    mime: str
    ts: str
    caption: NotRequired[str]
    content: NotRequired[str]
    source: NotRequired[str]
    sender_label: NotRequired[str]
    sender_session_id: NotRequired[str]
    client_message_id: NotRequired[str]
    transport: NotRequired[TransportMetadata]
    chat_id: NotRequired[int]
    # Deprecated compatibility field: runtime emits ``transport`` instead.
    telegram_chat_id: NotRequired[int]


class VideoOutbound(TypedDict):
    """Outbound WS video frame."""

    type: Literal["video"]
    role: Literal["user", "assistant"]
    video_base64: str
    mime: str
    ts: str
    caption: NotRequired[str]
    content: NotRequired[str]
    source: NotRequired[str]
    sender_label: NotRequired[str]
    sender_session_id: NotRequired[str]
    client_message_id: NotRequired[str]
    transport: NotRequired[TransportMetadata]
    chat_id: NotRequired[int]
    # Deprecated compatibility field: runtime emits ``transport`` instead.
    telegram_chat_id: NotRequired[int]


class DocumentOutbound(TypedDict):
    """Outbound WS document/file frame."""

    type: Literal["document"]
    role: Literal["user", "assistant"]
    file_base64: str
    mime: str
    filename: str
    ts: str
    caption: NotRequired[str]
    # Loopback /api/files/download?path=<root-relative> URL for the durable
    # artifact copy, used by the desktop host-bridge download (WKWebView-safe)
    # and to rebuild the bubble on reload without persisting base64.
    download_url: NotRequired[str]
    content: NotRequired[str]
    source: NotRequired[str]
    sender_label: NotRequired[str]
    sender_session_id: NotRequired[str]
    client_message_id: NotRequired[str]
    transport: NotRequired[TransportMetadata]
    chat_id: NotRequired[int]
    # Deprecated compatibility field: runtime emits ``transport`` instead.
    telegram_chat_id: NotRequired[int]


class TypingOutbound(TypedDict):
    """Outbound WS typing indicator."""

    type: Literal["typing"]
    action: str
    # Multi-project: stamps the thread so the client fan-out routes a project
    # task's typing indicator to its panel instead of defaulting to main.
    chat_id: NotRequired[int]


class LogOutbound(TypedDict):
    """Outbound WS log event."""

    type: Literal["log"]
    data: Dict[str, Any]
    # Multi-project: surfaced at top level so live task progress routes to the
    # owning project panel (and mirrors into main) by thread.
    chat_id: NotRequired[int]


class HeartbeatOutbound(TypedDict):
    """Outbound heartbeat emitted by ``server_runtime.ws_heartbeat_loop``."""

    type: Literal["heartbeat"]
    ts: NotRequired[str]


class ExtensionLifecycleOutbound(TypedDict):
    """Outbound extension lifecycle notification."""

    type: Literal["extension_lifecycle"]
    skill: NotRequired[str]
    action: NotRequired[str]
    status: NotRequired[str]
    reason: NotRequired[str]
    data: NotRequired[Dict[str, Any]]


class ProjectsChangedOutbound(TypedDict):
    """Outbound notice that the project registry changed server-side (e.g. the
    agent's ``promote_chat_to_task`` created/bound a project). The client refreshes
    its project nav + WS-fan-out ``projectChatIds`` on receipt; ``chat_id`` lets it
    learn the new project thread immediately, before the /api/state round-trip."""

    type: Literal["projects_changed"]
    project_id: NotRequired[str]
    chat_id: NotRequired[int]


class ProjectCreateRequest(TypedDict, total=False):
    """POST /api/projects body (v6.59.0). ONE source: ``path`` (attach an existing
    owner folder; optional ``init_git`` attach-snapshot commit — never auto-init),
    ``git_url`` (server-side clone, typed ``auth_required`` on credential failure),
    ``with_workspace`` (genesis provision), or none (file-less project).
    ``name``-only creation derives a filesystem id."""

    id: str
    name: str
    path: str
    init_git: bool
    git_url: str
    with_workspace: bool


class ProjectEntry(TypedDict, total=False):
    """A registry project row as returned by the projects endpoints. ``provenance``
    (attached|cloned|genesis|none) and ``clone_url`` are historical facts;
    operational git data is always read live from ``.git``."""

    id: str
    name: str
    chat_id: int
    working_dir: str
    provenance: str
    clone_url: str
    trusted_at: str
    origin: str
    created_at: str
    last_active_at: str


class ProjectDeleteResponse(TypedDict):
    """POST /api/projects/{project_id}/delete — unregister + unbind; the working
    folder and per-project memory store are never touched."""

    ok: bool
    project_id: str
    folder_untouched: bool


class FsDirsEntry(TypedDict):
    name: str
    path: str
    is_git: bool


class FsDirsResponse(TypedDict):
    """GET /api/fs/dirs — owner-facing directory browser for the New Project attach
    picker (server-side; works in web/Docker). Directories only, confined to the
    home tree, never file contents. ``truncated`` is True when the directory holds
    more children than the 500-entry cap (no silent truncation)."""

    path: str
    parent: str
    home: str
    dirs: List[FsDirsEntry]
    truncated: bool


class TaskNamedOutbound(TypedDict):
    """Outbound notice that the proactive card namer coined a project name for a fresh
    main-chat task (v6.40). The client sets the live card's title to ``suggested_name``;
    turn-into-project later reuses the same name. Not chat-scoped — carries only
    ``task_id`` and is a no-op unless a thread already holds that card."""

    type: Literal["task_named"]
    task_id: str
    suggested_name: str


class ErrorResponse(TypedDict):
    error: str


class StatusResponse(TypedDict):
    status: str


class HealthResponse(TypedDict):
    """Shape of ``GET /api/health``."""

    status: Literal["ok"]
    version: str
    runtime_version: str
    app_version: str


class EvolutionStateSnapshot(TypedDict):
    """Nested ``evolution_state`` block inside ``StateResponse``."""

    enabled: bool
    status: str
    detail: str
    cycle: int
    owner_chat_bound: bool
    last_task_at: str
    consecutive_failures: int
    # None when the remaining budget is infinite (unbudgeted): inf is not JSON
    # compliant, so the snapshot serializes it as null.
    budget_remaining_usd: Optional[float]
    budget_reserve_usd: float
    pending_count: int
    running_count: int
    queued_task_id: str
    running_task_id: str
    campaign: NotRequired[Dict[str, Any]]


class StateResponse(TypedDict):
    """Shape of ``GET /api/state`` (happy path)."""

    uptime: int
    workers_alive: int
    workers_total: int
    pending_count: int
    running_count: int
    spent_usd: float
    budget_limit: float
    budget_pct: float
    branch: str
    sha: str
    evolution_enabled: bool
    bg_consciousness_enabled: bool
    evolution_cycle: int
    evolution_state: EvolutionStateSnapshot
    bg_consciousness_state: Dict[str, Any]
    spent_calls: int
    supervisor_ready: bool
    supervisor_error: Optional[str]
    runtime_mode: str
    context_mode: str
    safety_mode: str
    skills_repo_configured: bool
    github_token_configured: bool
    # Multi-project sidebar feed (additive, v6.32.0): compact registered
    # projects [{id, name, chat_id, working_dir, last_active_at, has_thread_activity}].
    projects: list
    # COMPLETE (uncapped) registered project chat_ids — the live WS fan-out
    # isolation SSOT, distinct from the capped/filtered `projects` list.
    project_chat_ids: list
    # Task->project bindings ({task_id: {project_id, chat_id}}) so the frontend
    # can recognise a project-scoped task card: suppress the stray "turn into
    # project" button (v6.33.0 P2) and render a pointer that opens the bound
    # project's panel (v6.33.0 F4).
    task_bindings: dict


class SettingsNetworkMeta(TypedDict):
    """Network fields inside the ``GET /api/settings`` ``_meta`` block."""

    bind_host: str
    bind_port: int
    lan_ip: str
    reachability: Literal["loopback_only", "lan_reachable", "host_ip_unknown"]
    recommended_url: str
    warning: str


class SettingsMeta(SettingsNetworkMeta, total=False):
    """Complete ``GET /api/settings`` ``_meta`` block."""

    custom_secret_keys: list[str]
    setup_contract: Dict[str, Any]


class SettingsSaveResponse(TypedDict, total=False):
    status: str
    no_changes: bool
    restart_required: bool
    restart_keys: list[str]
    immediate_changed: bool
    next_task_changed: bool
    warnings: list[str]


class OwnerRuntimeModeResponse(TypedDict):
    ok: bool
    runtime_mode: str
    restart_required: bool


class OwnerAutoGrantResponse(TypedDict):
    ok: bool
    enabled: bool


class OwnerContextModeResponse(TypedDict):
    ok: bool
    context_mode: str


class OwnerScopeReviewFloorResponse(TypedDict):
    ok: bool
    scope_review_floor: str  # blocking_1m | advisory (v6.34.0, CW1)


class OwnerSafetyModeResponse(TypedDict):
    ok: bool
    safety_mode: str  # full | light | off (v6.54.3)


class SkillGrantResponse(TypedDict, total=False):
    ok: bool
    skill: str
    granted_keys: list[str]
    granted_permissions: list[str]
    extension_action: str
    extension_reason: str
    load_error: str
    grants: Dict[str, Any]
    error: str


class SkillDeleteResponse(TypedDict, total=False):
    ok: bool
    skill: str
    source: str
    deleted_payload_root: str
    deleted_state: bool
    extension_action: str
    extension_reason: str
    error: str


class UiPreferencesResponse(TypedDict):
    ok: NotRequired[bool]
    widget_order: list[str]
    nested_subagents_expanded: bool
    sidebar_width: int  # px; 0 = CSS default (resizable side sections, v6.33.0)
    project_panel_width: int  # px; 0 = CSS default
    project_last_viewed: dict[str, str]  # {project_id: ISO ts}; drives the unread dot (v6.33.0)
    project_hidden: dict[str, bool]  # {project_id: hidden}; sidebar presentation only (v6.59.0)


class GitLogResponse(TypedDict):
    commits: list[Dict[str, Any]]
    tags: list[str]
    branch: str
    sha: str


class EvolutionDataResponse(TypedDict):
    points: list[Dict[str, Any]]
    checkpoints: NotRequired[list[Dict[str, Any]]]
    generated_at: str
    cached: bool


class ScheduledTasksResponse(TypedDict):
    schema_version: int
    tasks: list[Dict[str, Any]]


class ScheduleUpsertResponse(TypedDict):
    ok: bool
    schedule: Dict[str, Any]


class ScheduleDeleteResponse(TypedDict):
    ok: bool


class UploadResponse(TypedDict):
    ok: bool
    filename: str
    display_name: str
    path: str
    size: int
    mime: str


class ExtensionsIndexResponse(TypedDict, total=False):
    extensions: list[Dict[str, Any]]
    skills: list[Dict[str, Any]]
    lifecycle: Dict[str, Any]
    error: str


class SkillLifecycleQueueResponse(TypedDict, total=False):
    active: Dict[str, Any]
    events: list[Dict[str, Any]]


class MarketplaceSearchResponse(TypedDict, total=False):
    items: list[Dict[str, Any]]
    results: list[Dict[str, Any]]
    installed: list[Dict[str, Any]]
    error: str


class MarketplaceInstalledResponse(TypedDict, total=False):
    installed: list[Dict[str, Any]]
    skills: list[Dict[str, Any]]
    error: str


class LocalModelStatusResponse(TypedDict, total=False):
    status: str
    running: bool
    ready: bool
    port: int
    message: str
    error: str


class McpStatusResponse(TypedDict, total=False):
    enabled: bool
    servers: list[Dict[str, Any]]
    tools: list[Dict[str, Any]]
    error: str


class ModelCatalogResponse(TypedDict, total=False):
    providers: list[Dict[str, Any]]
    models: list[Dict[str, Any]]
    error: str


class FileBrowserListResponse(TypedDict, total=False):
    root: str
    path: str
    entries: list[Dict[str, Any]]
    error: str


class ChatHistoryResponse(TypedDict, total=False):
    messages: list[Dict[str, Any]]
    has_more: bool
    next_before_ts: str
    error: str


class ExecutorRef(TypedDict, total=False):
    type: Required[Literal["local", "docker_exec"]]
    id: NotRequired[str]
    network: NotRequired[Literal["host", "none"]]
    workspace_host_path: NotRequired[str]
    workspace_backend_path: NotRequired[str]
    # Required at runtime when type == "docker_exec".
    container_name: NotRequired[str]
    path_mappings: NotRequired[list[Dict[str, str]]]


class _TaskCreateRequestRequired(TypedDict):
    description: str


class TaskCreateRequest(_TaskCreateRequestRequired, total=False):
    task_id: str
    type: str
    chat_id: int
    depth: int
    session_id: str
    workspace_root: str
    workspace_mode: str
    memory_mode: str
    project_id: str
    attachments: list[Dict[str, Any]]
    acceptance_claims: list[Dict[str, Any]]
    # v6.60.0: "" | "final_answer_line" — adapter-declared machine-extractable answer
    # protocol; flows into task_contract.answer_protocol and inherits to subagents.
    answer_protocol: str
    allowed_resources: Dict[str, Any]
    resource_policy: Dict[str, Any]
    disabled_tools: list[str]
    executor_ref: ExecutorRef
    service_teardown: Literal["stop", "keep"]
    deadline_at: str
    timeout_sec: float
    timeout: float
    context: str
    expected_output: str
    constraints: str
    context_requires_self_body_docs: bool
    actor_id: str
    source: str
    metadata: Dict[str, Any]


class TaskCreateResponse(TypedDict, total=False):
    ok: bool
    task_id: str
    status: str
    error: str


class TaskListResponse(TypedDict, total=False):
    tasks: list[Dict[str, Any]]
    queue: Dict[str, Any]
    error: str


class TaskEvent(TypedDict, total=False):
    seq: int
    source: str
    line: int
    ts: str
    type: str
    task_id: str
    root: str
    data: Dict[str, Any]


class TaskCancelResponse(TypedDict, total=False):
    ok: bool
    task_id: str
    error: str


class LogTailResponse(TypedDict, total=False):
    name: str
    entries: list[Dict[str, Any]]
    error: str


# Human/test-visible contract index; routers own executable Route objects.
HTTP_ENDPOINTS: tuple[str, ...] = (
    "GET /api/health",
    "GET /api/state",
    "GET /api/settings",
    "POST /api/settings",
    "GET /api/ui/preferences",
    "POST /api/ui/preferences",
    "POST /api/owner/runtime-mode",
    "POST /api/owner/auto-grant",
    "POST /api/owner/context-mode",
    "POST /api/owner/scope-review-floor",
    "POST /api/owner/safety-mode",
    "POST /api/owner/capability-ack",
    "POST /api/owner/skills/{skill}/attest-review",
    "GET /api/model-catalog",
    "POST /api/tasks",
    "GET /api/tasks",
    "GET /api/tasks/{task_id}",
    "GET /api/tasks/{task_id}/artifacts/{name}",
    "GET /api/tasks/{task_id}/events",
    "POST /api/tasks/{task_id}/cancel",
    "GET /api/schedules",
    "POST /api/schedules",
    "DELETE /api/schedules/{schedule_id}",
    "POST /api/command",
    "POST /api/reset",
    "GET /api/git/log",
    "POST /api/git/rollback",
    "POST /api/git/promote",
    "GET /api/update/status",
    "POST /api/update/check",
    "POST /api/update/preflight",
    "POST /api/update/apply",
    "GET /api/cost-breakdown",
    "GET /api/evolution-data",
    "GET /api/projects",
    "POST /api/projects",
    "POST /api/projects/from-task",
    "POST /api/projects/{project_id}/update",
    "POST /api/projects/{project_id}/delete",
    "GET /api/fs/dirs",
    "GET /api/chat/history",
    "GET /api/logs/{name}",
    "POST /api/chat/upload",
    "DELETE /api/chat/upload",
    "POST /api/openai-compatible/models",
    "GET /api/local-model/status",
    "POST /api/local-model/start",
    "POST /api/local-model/stop",
    "POST /api/local-model/test",
    "POST /api/local-model/install-runtime",
    "GET /api/mcp/status",
    "POST /api/mcp/refresh",
    "POST /api/mcp/test",
    "GET /api/extensions",
    "GET /api/extensions/{skill}/manifest",
    "GET /api/extensions/{skill}/module/{entry}",
    "GET /api/extensions/{skill}/settings_section",
    "ANY /api/extensions/{skill}/{rest:path}",
    "GET /api/skills/daemons",
    "POST /api/skills/{skill}/toggle",
    "POST /api/skills/{skill}/delete",
    "GET /api/skills/lifecycle-queue",
    "POST /api/skills/{skill}/review",
    "POST /api/skills/{skill}/grants",
    "POST /api/skills/{skill}/reconcile",
    "GET /api/marketplace/clawhub/search",
    "GET /api/marketplace/clawhub/installed",
    "GET /api/marketplace/clawhub/info/{slug:path}",
    "GET /api/marketplace/clawhub/preview/{slug:path}",
    "POST /api/marketplace/clawhub/install",
    "POST /api/marketplace/clawhub/update/{name}",
    "POST /api/marketplace/clawhub/uninstall/{name}",
    "GET /api/marketplace/ouroboroshub/catalog",
    "GET /api/marketplace/ouroboroshub/installed",
    "GET /api/marketplace/ouroboroshub/preview/{slug:path}",
    "POST /api/marketplace/ouroboroshub/install",
    "POST /api/marketplace/ouroboroshub/update/{name}",
    "POST /api/marketplace/ouroboroshub/uninstall/{name}",
    "GET /api/onboarding",
    "GET /api/claude-code/status",
    "POST /api/claude-code/install",
    "GET /api/files/list",
    "GET /api/files/read",
    "GET /api/files/content",
    "GET /api/files/download",
    "POST /api/files/upload",
    "POST /api/files/mkdir",
    "POST /api/files/write",
    "POST /api/files/delete",
    "POST /api/files/transfer",
    "WS /ws",
)

WS_MESSAGE_TYPES: tuple[str, ...] = (
    "chat",
    "command",
    "photo",
    "video",
    "document",
    "typing",
    "log",
    "heartbeat",
    "extension_lifecycle",
    "projects_changed",
    "task_named",
)


__all__ = [
    "ChatInbound",
    "TaskConstraintInbound",
    "CommandInbound",
    "ExtensionInbound",
    "TransportMetadata",
    "ChatOutbound",
    "PhotoOutbound",
    "VideoOutbound",
    "DocumentOutbound",
    "TypingOutbound",
    "LogOutbound",
    "HeartbeatOutbound",
    "ExtensionLifecycleOutbound",
    "ProjectsChangedOutbound",
    "ProjectCreateRequest",
    "ProjectEntry",
    "ProjectDeleteResponse",
    "FsDirsEntry",
    "FsDirsResponse",
    "TaskNamedOutbound",
    "ErrorResponse",
    "StatusResponse",
    "HealthResponse",
    "StateResponse",
    "EvolutionStateSnapshot",
    "SettingsNetworkMeta",
    "SettingsMeta",
    "SettingsSaveResponse",
    "OwnerRuntimeModeResponse",
    "OwnerAutoGrantResponse",
    "OwnerContextModeResponse",
    "OwnerScopeReviewFloorResponse",
    "OwnerSafetyModeResponse",
    "SkillGrantResponse",
    "SkillDeleteResponse",
    "UiPreferencesResponse",
    "GitLogResponse",
    "EvolutionDataResponse",
    "ScheduledTasksResponse",
    "ScheduleUpsertResponse",
    "ScheduleDeleteResponse",
    "UploadResponse",
    "ExtensionsIndexResponse",
    "SkillLifecycleQueueResponse",
    "MarketplaceSearchResponse",
    "MarketplaceInstalledResponse",
    "LocalModelStatusResponse",
    "McpStatusResponse",
    "ModelCatalogResponse",
    "FileBrowserListResponse",
    "ChatHistoryResponse",
    "ExecutorRef",
    "TaskCreateRequest",
    "TaskCreateResponse",
    "TaskListResponse",
    "TaskEvent",
    "TaskCancelResponse",
    "LogTailResponse",
    "HTTP_ENDPOINTS",
    "WS_MESSAGE_TYPES",
]
