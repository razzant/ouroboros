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
    # X3: a repair receipt whose managed task id does not exist yet (the router
    # mints it at promotion). Typed truth instead of an invented id.
    task_id_pending: NotRequired[bool]
    ephemeral_decision: NotRequired[bool]
    task_incident: NotRequired[str]
    toast_once: NotRequired[str]
    lifecycle: NotRequired[Dict[str, Any]]
    # C4 multi-chat dedupe: a duplicate lifecycle initiator's typed pointer to
    # the job that already owns the routing ({job_id, kind, target, status,
    # chat_id}); the first initiator's chat keeps the progress stream.
    lifecycle_pointer: NotRequired[Dict[str, Any]]
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
    # Phase 6: the OPAQUE harness route RESOLVED AT DISPATCH for this
    # bubble/subagent (`resolve_subagent_dispatch`, stamped once) — a delegated
    # route only; absent/empty means the ordinary native path and the UI draws
    # no chip. It is the route the run was sent to, not a receipt from the
    # engine saying where it landed: a landing below the ask is disclosed on
    # `capability_delta`, not by rewriting this field.
    executor_route: NotRequired[str]
    # The completion-seam EVIDENCE the route decision is reconciled against
    # (subagents.envelope_from_task): delegated runs started/settled/succeeded,
    # terminal failure states, disclosed subscription spend (+estimated flag),
    # engine-reported models. Terminal frames only; its absence means "no
    # evidence yet", never "ran natively".
    execution_evidence: NotRequired[Dict[str, Any]]
    # The FACT beside the executor_route plan, from the same custody evidence:
    # "harness_used" | "harness_attempted" | "native_only". Terminal frames only; absent =
    # no substrate claim.
    actual_substrate: NotRequired[str]
    model: NotRequired[str]
    task_group_id: NotRequired[str]
    task_event: NotRequired[str]
    status: NotRequired[str]
    # v6.82 (P5): host-attested marker — this frame's task is a supervisor-queue
    # task that POST /api/tasks/{id}/cancel can force-cancel (never set for
    # in-process direct-chat turns). Gates the UI "Cancel run" card action.
    cancelable: NotRequired[bool]
    # Monetary projections are nullable when the physical-attempt ledger cannot
    # be read.  ``None`` is deliberately distinct from a confirmed $0 result.
    cost_usd: NotRequired[Optional[float]]
    # C2 (owner 10=B): the additive HONEST names — accounted upper bounds, not
    # settled receipts. Same values as the deprecated cost_usd[_with_children]
    # aliases (ouroboros/cost_projection.py is the one author); the aliases stay
    # outbound because their removal is a separate approved ABI break.
    accounted_upper_bound_usd: NotRequired[Optional[float]]
    accounted_upper_bound_usd_with_children: NotRequired[Optional[float]]
    cost_accounting_status: NotRequired[Literal["available", "unavailable"]]
    cost_accounting_error: NotRequired[str]
    cost_final: NotRequired[bool]
    cost_usd_with_children: NotRequired[Optional[float]]
    cost_with_children_partial: NotRequired[bool]
    reserved_usd: NotRequired[Optional[float]]
    unresolved_upper_bound_usd: NotRequired[Optional[float]]
    unknown_unmetered: NotRequired[Optional[int]]
    # v6.87.48: count of OPEN ledger rows — the disclosed cause of
    # ``cost_final: false``, which can hold with every dollar bucket at zero.
    non_final_rows: NotRequired[Optional[int]]
    # C12: the ledger's own INTEGRITY marker. The cost authority has always
    # produced it (`reconstruct_task_cost`), but no carry list named it, so an
    # amount computed over a degraded ledger reached every surface looking exactly
    # like one computed over a sound ledger.
    ledger_integrity_degraded: NotRequired[Optional[bool]]
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
    review_projection: NotRequired[Dict[str, Any]]
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


#: The CLOSED status vocabulary of `connection_state`, as a runtime value beside the
#: `Literal` that declares it — a `Literal` is a type-checker fact and every producer
#: of this frame is a runtime one. `gateway/connections._public_live_fields` clamps to
#: it at the single boundary they all cross, because a sixth word is not additive but
#: INVISIBLE: `_record_runtime_health` drops a status it does not know, and
#: `web/modules/remote_task_state.js` reads one as *no typed status at all* and falls
#: back to derivation — which withdraws the Reconnect button at the one moment a
#: reconnect is the fix.
CONNECTION_STATUSES: frozenset[str] = frozenset(
    {"connecting", "ready", "degraded", "disconnected", "unknown"}
)


class ConnectionStateOutbound(TypedDict, total=False):
    """Bounded live connection/admission projection; durable secrets are absent.

    Every field here comes out of ONE projection
    (``gateway/connections.py::_public_live_fields``), which is also what fills
    ``ConnectionEntry``'s live half and ``ConnectionActionResponse`` — so the frame
    carries the target-identity and evidence fields too, and declaring less than the
    projection emits would leave the browser mirror pinned to a shape the server does
    not send. ``task_id``/``project_id`` are present on the task-SCOPED frames
    ``_broadcast_connection_state`` fans out per live task on the connection.
    """

    type: Required[Literal["connection_state"]]
    connection_id: Required[str]
    task_id: NotRequired[str]
    project_id: NotRequired[str]
    status: NotRequired[
        Literal["connecting", "ready", "degraded", "disconnected", "unknown"]
    ]
    phase: NotRequired[str]
    completion: NotRequired[str]
    error_code: NotRequired[str]
    action: NotRequired[str]
    platform: NotRequired[str]
    architecture: NotRequired[str]
    build: NotRequired[str]
    bootstrap_compatible: NotRequired[bool]
    health_fresh: NotRequired[bool]
    # The bootstrap claim CHECKED against this build's contract set: True means the
    # host carries an execd that predates a shared-contract change, so every remote
    # tool call on it would fail and Bootstrap is the action. It rides beside
    # `bootstrap_compatible` rather than inside it because "never bootstrapped" and
    # "bootstrapped too long ago" are different sentences with the same next step.
    execd_outdated: NotRequired[bool]
    required_contract_set: NotRequired[int]
    bootstrap_contract_set: NotRequired[int]
    # THE one blocker in front of this connection, derived server-side from the
    # evidence above (``remote_refusal_actions.connection_blocker``) and absent
    # exactly when the connection is selectable. `blocker_action` is the single
    # action that removes it and `blocker_hint` the owner sentence naming it; no
    # surface composes its own, which is how the New Project picker came to offer
    # "run Bootstrap (or Test to refresh health)" for a block Test cannot move.
    # `blocker_rank` is the position in the removal ladder: a HIGHER rank means
    # fewer remaining steps, so a picker advising about one of several blocked
    # connections picks the highest.
    blocked_by: NotRequired[str]
    blocker_action: NotRequired[str]
    blocker_hint: NotRequired[str]
    blocker_rank: NotRequired[int]
    diagnostic: NotRequired[Dict[str, Any]]
    log_refs: NotRequired[List[Dict[str, Any]]]
    warnings: NotRequired[List[Dict[str, Any]]]
    log_refs_count: NotRequired[int]
    warnings_count: NotRequired[int]


class MessageAnnotationOutbound(TypedDict):
    """Bubble-free presentation update for one canonical owner message."""

    type: Literal["message_annotation"]
    annotation_type: Literal["routing_ack"]
    client_message_id: str
    action: str
    status: str
    suppress_bubble: bool
    chat_id: NotRequired[int]
    target: NotRequired[str]
    options: NotRequired[List[Dict[str, Any]]]
    ts: NotRequired[str]


class UpdateMergePlan(TypedDict, total=False):
    """Exact, channel-bound merge plan shared by preflight and apply responses."""

    available: bool
    auto_mergeable: bool
    kind: Literal["clean", "conflicting", "current", "unavailable", "unknown"]
    error: str
    remote: str
    remote_branch: str
    target_ref: str
    update_channel: str
    current_branch: str
    base_sha: str
    target_sha: str
    local_dirty_count: int
    local_snapshot: str
    merge_commit: str
    code_conflict_paths: List[str]
    doc_conflict_paths: List[str]
    hot_code_paths: List[str]
    recommended_strategy: Literal["auto_merge", "assisted"]


class UpdatePreflightRequest(TypedDict, total=False):
    """POST /api/update/preflight has an intentionally empty JSON body."""


class UpdatePreflightResponse(TypedDict):
    merge_plan: UpdateMergePlan


class UpdateApplyRequest(TypedDict):
    strategy: Literal["auto_merge", "assisted", "manual", "replace"]
    expected_base_sha: NotRequired[str]
    expected_target_sha: NotRequired[str]
    confirm_recovery: NotRequired[bool]


class UpdateApplySuccessResponse(TypedDict):
    status: Literal["ok", "restart_required", "assisted_started", "manual"]
    restarting: NotRequired[bool]
    strategy: NotRequired[Literal["auto_merge", "assisted", "manual", "replace"]]
    task_id: NotRequired[str]
    merge_plan: NotRequired[UpdateMergePlan]
    error: NotRequired[str]


class UpdateApplyErrorResponse(TypedDict):
    error: str
    reason: NotRequired[str]
    blockers: NotRequired[List[str]]
    rolled_back: NotRequired[bool]
    rollback: NotRequired[str]
    restart_required: NotRequired[bool]
    merge_plan: NotRequired[UpdateMergePlan]
    smoke: NotRequired[Dict[str, Any]]


class UpdateStatusReadyOutbound(TypedDict):
    type: Literal["update_status_ready"]
    available: bool
    check_ok: Optional[bool]


class ProjectCreateRequest(TypedDict, total=False):
    """POST /api/projects body (v6.59.0). ONE source: ``path`` (attach an existing
    owner folder; optional ``init_git`` attach-snapshot commit — never auto-init),
    ``git_url`` (server-side clone, typed ``auth_required`` on credential failure),
    ``with_workspace`` (genesis provision), ``connection_id`` + ``remote_root`` (a
    folder on a remote host — RWS v2), or none (file-less project).
    ``name``-only creation derives a filesystem id.

    The remote source is the TWO HALVES the owner can know, deliberately not a
    serialized ``workspace_ref``: the placement's third field (the workspace
    identity) is allocated by the TARGET at admission, so a client that could name it
    could claim a workspace it never opened, and a client that picked the
    discriminated variant itself would be choosing its own placement authority."""

    id: str
    name: str
    path: str
    init_git: bool
    git_url: str
    with_workspace: bool
    connection_id: str
    remote_root: str


class ProjectUpdateRequest(TypedDict, total=False):
    """POST /api/projects/{project_id}/update body.

    Two mutations, both optional and combinable: ``name`` renames the project, and
    ``connection_id`` + ``remote_root`` REBIND its remote placement (the same two
    halves ``ProjectCreateRequest`` takes, admitted the same way). A rebind advances
    ``routing_generation``, so work already resolved against the previous target is
    refused at queue insertion rather than run there."""

    name: str
    connection_id: str
    remote_root: str


class ProjectEntry(TypedDict, total=False):
    """A registry project row as returned by the projects endpoints. ``provenance``
    (attached|cloned|genesis|remote|none) and ``clone_url`` are historical facts;
    operational git data is always read live from ``.git``.

    ``placement`` is the sealed remote placement (RWS v2) and is present ONLY for a
    remote project — ``working_dir`` is then empty, because a remote project has no
    Home folder and offering one would be a local path standing in for the target."""

    id: str
    name: str
    chat_id: int
    working_dir: str
    placement: Optional[ProjectWorkspaceRef]
    provenance: str
    clone_url: str
    trusted_at: str
    origin: str
    created_at: str
    last_active_at: str
    lifecycle: Literal["active", "deleting", "tombstoned"]
    routing_generation: int
    visible_revision: int
    delete_error: str


class ProjectDeleteResponse(TypedDict):
    """POST /api/projects/{project_id}/delete — fence, quiesce, and tombstone;
    the immutable binding, working folder, history, and memory are preserved."""

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
    spent_usd: Optional[float]
    budget_limit: float
    budget_pct: Optional[float]
    branch: str
    sha: str
    evolution_enabled: bool
    bg_consciousness_enabled: bool
    evolution_cycle: int
    evolution_state: EvolutionStateSnapshot
    bg_consciousness_state: Dict[str, Any]
    spent_calls: Optional[int]
    supervisor_ready: bool
    supervisor_error: Optional[str]
    runtime_mode: str
    context_mode: str
    # True when the EFFECTIVE `low` is a system auto-downgrade rather than an owner
    # selection: the owner control needs it to offer "confirm Low" on a no-op click.
    context_mode_auto_low: bool
    safety_mode: str
    skills_repo_configured: bool
    github_token_configured: bool
    accounting: Dict[str, Any]
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
    # True when the save landed while an agent task was already STARTED: that
    # task keeps its start-time config (snapshot boundary); changes apply from
    # the next task. Queued-but-unstarted tasks re-read settings at start.
    agent_task_running: bool


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
    # v6.80.0: the value is STORED but enforcement-inert — scope-review applicability
    # follows the owner-only context mode. The notice says so on every write.
    deprecation_notice: str


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
    project_seen_revision: dict[str, int]  # monotonic paint ACK per active Project
    project_last_viewed: dict[str, str]  # deprecated one-minor accepted no-op
    project_hidden: dict[str, bool]  # deprecated one-minor accepted no-op


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
    # "ssh" is a DERIVED projection of the persisted WorkspaceRef (RWS v2);
    # nothing stores an ssh executor_ref independently — see workspace_ref.py.
    type: Required[Literal["local", "docker_exec", "ssh"]]
    id: NotRequired[str]
    network: NotRequired[Literal["host", "none"]]
    workspace_host_path: NotRequired[str]
    workspace_backend_path: NotRequired[str]
    # Required at runtime when type == "docker_exec".
    container_name: NotRequired[str]
    path_mappings: NotRequired[list[Dict[str, str]]]
    # Required at runtime when type == "ssh"; Home path mappings are forbidden
    # for that arm (workspace_executor.normalize_executor_ref).
    connection_id: NotRequired[str]
    remote_root: NotRequired[str]
    workspace_id: NotRequired[str]


class ProjectWorkspaceRef(TypedDict, total=False):
    """Wire mirror of the persisted placement descriptor (workspace_ref.py)."""

    kind: Required[Literal["local", "ssh"]]
    local_root: NotRequired[str]
    connection_id: NotRequired[str]
    remote_root: NotRequired[str]
    workspace_id: NotRequired[str]


class ConnectionEntry(TypedDict, total=False):
    """One owner connection: durable store row + bounded live projection.

    Durable fields come from ``connection_store.py`` (never secrets). The live
    status/diagnostic fields are process-local projections and are never persisted
    back into the store. ``bootstrap_compatible`` is DERIVED from the durable
    ``bootstrapped_at``: a compatible executor stays installed on that host across a
    Home restart, so the claim is owner state. ``health_fresh`` is inherently
    process-local — "the target answered within the last few minutes" is a statement
    about this run, which no durable record can make.
    """

    id: Required[str]
    name: Required[str]
    ssh_alias: Required[str]
    expected_host_id: NotRequired[str]
    host_id_history: NotRequired[List[Dict[str, Any]]]
    lifecycle: NotRequired[Literal["active", "retired"]]
    retired_at: NotRequired[Optional[str]]
    bootstrapped_at: NotRequired[Optional[str]]
    bootstrap_build: NotRequired[str]
    # The Home↔execd contract set the last bootstrap installed. `bootstrap_build`
    # says WHICH artifact is on the host; this says whether it can still talk to this
    # Home, which is the only one of the two a status surface can compare against
    # anything (`connection_store.record_bootstrap` states why the release id cannot).
    bootstrap_contract_set: NotRequired[int]
    created_at: NotRequired[str]
    updated_at: NotRequired[str]
    status: NotRequired[
        Literal["connecting", "ready", "degraded", "disconnected", "unknown"]
    ]
    phase: NotRequired[str]
    # The project whose live session the broker reported for this connection
    # (``remote_workspace.status`` rows carry it, and ``api_connections_list``
    # merges the bounded live projection of those rows).
    project_id: NotRequired[str]
    platform: NotRequired[str]
    architecture: NotRequired[str]
    build: NotRequired[str]
    bootstrap_compatible: NotRequired[bool]
    health_fresh: NotRequired[bool]
    # The bootstrap claim CHECKED against this build's contract set: True means the host
    # carries an execd that predates a shared-contract change, so every remote tool call
    # on it would fail and Bootstrap is the action. It rides BESIDE
    # `bootstrap_compatible` rather than inside it, because "never bootstrapped" and
    # "bootstrapped against an older contract set" are different sentences with the same
    # next step, and a surface that says only "incompatible" cannot say which it means.
    execd_outdated: NotRequired[bool]
    required_contract_set: NotRequired[int]
    # The one blocker in front of this connection and the single action that
    # removes it, derived server-side (``remote_refusal_actions.connection_blocker``)
    # and ABSENT exactly when the row is selectable — so "has no blocker" and
    # "may be offered in the New Project picker" are one fact, not two opinions.
    blocked_by: NotRequired[str]
    blocker_action: NotRequired[str]
    blocker_hint: NotRequired[str]
    blocker_rank: NotRequired[int]
    completion: NotRequired[str]
    error_code: NotRequired[str]
    action: NotRequired[str]
    diagnostic: NotRequired[Dict[str, Any]]
    log_refs: NotRequired[List[Dict[str, Any]]]
    # Bounded non-fatal transport observations (e.g. an ssh alias whose
    # forwarding directives were neutralized). ``gateway/connections.py``
    # ``_public_live_fields`` already emits these on the wire.
    warnings: NotRequired[List[Dict[str, Any]]]
    # The TOTAL each bounded list above was capped from. A silent cap makes "four
    # warnings" indistinguishable from "four of nine warnings"; the count is one
    # number rather than a number plus a `*_truncated` flag that could disagree
    # with it, since truncation IS ``count > len(list)``.
    log_refs_count: NotRequired[int]
    warnings_count: NotRequired[int]


class ConnectionAddRequest(TypedDict):
    name: str
    ssh_alias: str


class ConnectionListResponse(TypedDict, total=False):
    connections: List[ConnectionEntry]
    error: str
    error_code: str
    action: str


class ConnectionActionResponse(TypedDict, total=False):
    """The answer of every transport-dependent owner connection action.

    ``gateway/connections.py::_connection_action`` returns the broker's envelope
    with Home's own fields merged in, so this declares the keys a CONSUMER reads
    (the Settings card, the CLI, the retrust flow) rather than every key a broker
    may happen to include.

    ``host_id`` and ``handshake`` are load-bearing and were both undeclared:
    retrust exists to accept a REPLACEMENT host identity, and the only way any
    surface learns the currently observed one is by reading them off this response
    (``connections_ui.js::observedHostId``,
    ``cli_connections._observed_cli_host_id``, and the gateway's own
    ``_observed_host_id`` — the three must agree, or a retrust confirmation pair is
    assembled from a field one of them cannot see).
    """

    ok: bool
    connection: ConnectionEntry
    connection_id: str
    status: str
    phase: str
    completion: str
    error: str
    error_code: str
    action: str
    host_id: str
    handshake: Dict[str, Any]
    platform: str
    architecture: str
    build: str
    bootstrap_compatible: bool
    health_fresh: bool
    # `_connection_action` merges Home's own evidence into every answer, so a Test
    # or Bootstrap response carries the contract-set verdict AND what is still in
    # the way after it succeeded. That last part is the whole point: a Test on a
    # host with an outdated executor answers `ok` with `health_fresh: true`, and
    # without these fields the surface had nothing with which to correct the advice
    # it had already given.
    execd_outdated: bool
    required_contract_set: int
    bootstrap_contract_set: int
    blocked_by: str
    blocker_action: str
    blocker_hint: str
    blocker_rank: int
    diagnostic: Dict[str, Any]
    log_refs: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    log_refs_count: int
    warnings_count: int


class ConnectionDirsResponse(TypedDict, total=False):
    connection_id: str
    path: str
    parent: str
    dirs: List[FsDirsEntry]
    truncated: bool
    error: str
    error_code: str
    action: str


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


class TaskCostBreakdown(TypedDict):
    """Read-time "where did the money go" projection on ``GET /api/tasks/{task_id}``
    (ROOT tasks only). Computed from the physical-attempt ledger at read time and
    never persisted; ``own + children + unattributed == subtree``. When the object
    is present every key is present; the WHOLE object is absent — never a confident
    $0 — when accounting is unavailable or holds no attributable row for the
    subtree, and on non-root task details."""

    own_usd: float
    children_usd: float
    unattributed_usd: float
    delegated_disclosed_usd: float
    # C2: the explicit subtree total under its honest name —
    # own + children + unattributed, an accounted UPPER BOUND, not a receipt.
    accounted_upper_bound_usd: float
    subscription_sessions: int
    unknown_unmetered: int
    non_final_rows: int
    cost_final: bool
    authority: Literal["physical_attempt_ledger"]


class TaskDetailResponse(TypedDict, total=False):
    """``GET /api/tasks/{task_id}`` — the public task-result envelope (open shape;
    stored task-result keys pass through) plus additive typed projections."""

    cost_breakdown: TaskCostBreakdown
    # Poltergeist phase A cancel projection (additive-optional): ``"pending"``
    # while a durable cancel intent is open and the supervisor teardown has not
    # settled — the status itself honestly stays running/scheduled. Absent on
    # settled results and on tasks nobody asked to cancel. The UI renders the
    # interim "Cancelling…" from this field, never from a status value.
    cancel_state: str
    # Rides beside ``cancel_state`` when the intent carries a reason (GR2-11):
    # the WHY of the pending cancellation (owner text, "subtree cancellation of
    # <root>", "evolution stopped", …). Absent when no reason was recorded.
    cancel_reason: str
    error: str


ClaudexorReadState = Literal["ok", "not_read", "failed"]


class ClaudexorStatusReads(TypedDict):
    """PROVENANCE for each independent facet of ``GET /api/claudexor/status``.

    An empty collection cannot say whether the daemon was ASKED: the owned
    Claudexor daemon starts lazily, so an idle machine used to serve empty
    lists that every consumer read as "no account connected" while real
    accounts sat in the agent home. Each facet answers only for itself, since
    one fanned-out read can fail while its siblings land:

    - ``ok`` — read; the matching collection is AUTHORITATIVE (empty means empty)
    - ``not_read`` — this facet was never asked: the daemon was not running, or
      discovery/handshake died BEFORE the fan-out (which leaves every facet
      untouched while the aggregate state reports ``unreachable``)
    - ``failed`` — asked, and no usable answer came back: the read refused, or
      the body arrived in a shape the facet does not promise

    Facets map to ``harnesses`` (catalog), ``profiles`` (accounts) and ``quota``.
    The manifest read behind the login-capability filter is deliberately NOT a
    facet: its failure is absorbed (fail-open), never reported."""

    catalog: ClaudexorReadState
    accounts: ClaudexorReadState
    quota: ClaudexorReadState


class ClaudexorStatusResponse(TypedDict, total=False):
    """``GET /api/claudexor/status`` — owned-daemon lifecycle plus the daemon's
    own catalog/account/quota truth, each stamped with its read state. Read-only;
    never spawns the daemon (waking it is the owner-initiated POST)."""

    daemon: Dict[str, Any]
    config_dir: str
    harnesses: List[Dict[str, Any]]
    profiles: Dict[str, Any]
    quota: List[Dict[str, Any]]
    reads: ClaudexorStatusReads
    subagent_last_delegation: Dict[str, Any]
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
    # v6.82 (P5): echoed when the optional request body {"cascade": true} asked
    # for the subtree cancel, which is COMPLETE by the time this answer is sent;
    # the plain envelope is unchanged.
    cascade: bool
    error: str


class LogTailResponse(TypedDict, total=False):
    name: str
    entries: list[Dict[str, Any]]
    error: str


class OnboardingCompleteRequest(TypedDict, total=False):
    """``POST /api/onboarding/complete`` — the wizard payload plus two
    DECLARATIONS about the onboarding run itself.

    The settings keys of the shared setup contract ride through unchanged (open
    shape, same payload the wizard already builds); only the two subscription
    flags are typed here, because they are not settings. Neither is authority:
    ``subscriptionsConnected`` only tells the server to read the live
    agent account state, and the server re-proves fresh-install status
    on its own before applying anything."""

    subscriptionsConnected: bool
    skipSubscriptionPresets: bool


class OnboardingPresetProjection(TypedDict):
    """What the install-time agent preset did, on the success envelope.

    ``applied=False`` with a ``reason`` is the normal shape for an install that
    connected no subscription, opted out, or is no longer in first-run
    onboarding — absence is reported as absence, never as an empty success."""

    applied: bool
    reason: str
    harnesses: list[str]
    receipt: Dict[str, Any]


class OnboardingCompleteResponse(TypedDict):
    """The ONE success envelope. Settings, the next-boot runtime mode, the
    fresh-install safety default and the durable completion fact land ATOMICALLY
    — every success carries all four. The preset keys and their one-shot marker
    ride the same write only when ``preset.applied`` is true; an ordinary success
    with ``not_requested``, ``skipped_by_owner`` or ``not_install_time`` persists
    no preset and no marker, which is the D-4 design, not a partial save."""

    ok: bool
    status: str
    runtime_mode: str
    restart_required: bool
    preset: OnboardingPresetProjection


class SettingsPostCommitFailureResponse(TypedDict):
    """500 from an owner settings write whose BYTES ALREADY LANDED.

    The distinction the broad handlers used to erase: a failure BEFORE the write
    is "nothing was saved", a failure AFTER it is "saved, and then this step
    failed". ``post_commit_failed`` names the step (environment projection,
    supervisor start, hot-reload…) so the owner knows what to retry — never that
    the settings themselves need saving again. Shared by ``POST /api/settings``
    and ``POST /api/onboarding/complete``."""

    error: str
    status: str
    saved: bool
    post_commit_failed: str


class OnboardingPresetFailureResponse(TypedDict):
    """503: the connected agent accounts could not be verified, so
    NOTHING was persisted. ``can_skip`` tells the wizard the secondary
    "finish without agent defaults" action will succeed."""

    error: str
    code: str
    detail: str
    can_skip: bool
    saved: bool


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
    "GET /api/owner/connections",
    "POST /api/owner/connections",
    "POST /api/owner/connections/{connection_id}/test",
    "POST /api/owner/connections/{connection_id}/bootstrap",
    "POST /api/owner/connections/{connection_id}/reconnect",
    "POST /api/owner/connections/{connection_id}/retrust",
    "GET /api/owner/connections/{connection_id}/dirs",
    "DELETE /api/owner/connections/{connection_id}",
    "GET /api/model-catalog",
    "POST /api/tasks",
    "GET /api/tasks",
    "GET /api/tasks/{task_id}",
    "GET /api/tasks/{task_id}/artifacts/{name}",
    "GET /api/tasks/{task_id}/events",
    "POST /api/tasks/{task_id}/cancel",
    "POST /api/tasks/{task_id}/resume",
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
    "GET /api/reviewer-slots",
    "GET /api/claudexor/status",
    "POST /api/claudexor/wake",
    "POST /api/claudexor/login",
    "GET /api/claudexor/login/{job_id}",
    "DELETE /api/claudexor/login/{job_id}",
    "POST /api/claudexor/login/{job_id}/input",
    "DELETE /api/claudexor/credential-profiles/{harness}/{profile_id}",
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
    # The wizard PAGE (one onboarding host: desktop setup window, blocking
    # overlay frame, plain browser). /api/onboarding stays the readiness probe.
    "GET /onboarding",
    "GET /api/onboarding",
    "POST /api/onboarding/complete",
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
    "message_annotation",
    "projects_changed",
    "task_named",
    "update_status_ready",
    "connection_state",
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
    "CONNECTION_STATUSES",
    "ConnectionStateOutbound",
    "MessageAnnotationOutbound",
    "UpdateMergePlan",
    "UpdatePreflightRequest",
    "UpdatePreflightResponse",
    "UpdateApplyRequest",
    "UpdateApplySuccessResponse",
    "UpdateApplyErrorResponse",
    "UpdateStatusReadyOutbound",
    "ProjectCreateRequest",
    "ProjectUpdateRequest",
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
    "OnboardingCompleteRequest",
    "OnboardingCompleteResponse",
    "OnboardingPresetFailureResponse",
    "OnboardingPresetProjection",
    "SettingsPostCommitFailureResponse",
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
    "ProjectWorkspaceRef",
    "ConnectionEntry",
    "ConnectionAddRequest",
    "ConnectionListResponse",
    "ConnectionActionResponse",
    "ConnectionDirsResponse",
    "TaskCreateRequest",
    "TaskCreateResponse",
    "TaskListResponse",
    "TaskCostBreakdown",
    "TaskDetailResponse",
    "ClaudexorReadState",
    "ClaudexorStatusReads",
    "ClaudexorStatusResponse",
    "TaskEvent",
    "TaskCancelResponse",
    "LogTailResponse",
    "HTTP_ENDPOINTS",
    "WS_MESSAGE_TYPES",
]
