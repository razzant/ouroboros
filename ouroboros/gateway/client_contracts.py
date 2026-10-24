"""Envelopes a replaceable client needs beside ``contracts.py``: their own module, one owner each.

Descriptive TypedDicts, the twin of ``web/modules/client_contract_types.js``: the main
interface a server generation serves (``StateResponse.served_ui``), the read-only setup
bootstrap a client renders onboarding from (``GET /api/onboarding/bootstrap``, whose rows
mirror ``settings_setup_contract``), the settings writers' durable-outcome refusal, the
auth gate's JSON login, the derived ABI document ``GET /api/contract`` publishes, and the
bodies of the everyday journey routes (projects, files, places, owner controls, schedules) whose
handlers check their fields by hand.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # Python 3.11+
    from typing import Literal, NotRequired, Required, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import Literal, NotRequired, Required, TypedDict  # type: ignore[assignment]


class ServedUiState(TypedDict):
    """The main interface THIS server generation serves at ``/`` and ``/static``: the applied
    selection (fixed until the core restarts), not the requested value and not an asset digest.
    A directory that could not be applied serves the builtin interface and says why."""

    source: Literal["builtin", "directory", "none"]
    error: NotRequired[str]


class SetupStep(TypedDict):
    id: str
    title: str
    railCopy: str
    copy: str
    footer: str


class SetupProviderField(TypedDict):
    """One credential or endpoint input; ``settingKey`` is the settings key it saves into."""

    id: str
    stateKey: str
    settingKey: str
    settingsInputId: str
    label: str
    placeholder: str
    note: str
    inputType: str
    group: Literal["primary", "more"]


class SetupProviderProfile(TypedDict):
    label: str
    providerCopy: str
    modelCopy: str


class SetupModelSlot(TypedDict):
    slot: str
    stateKey: str
    settingKey: str
    inputId: str
    label: str
    note: str
    settingsInputId: str
    settingsToggleId: str


class SetupModeChoice(TypedDict):
    """A ``reviewModes`` or ``runtimeModes`` row; ``value`` is what the setting stores."""

    value: str
    label: str
    tone: str
    className: str
    copy: str


class SetupLocalRoutingMode(TypedDict):
    value: str
    buttonLabel: str
    label: str
    flags: List[bool]  # USE_LOCAL_MAIN, USE_LOCAL_LIGHT, USE_LOCAL_CONSCIOUSNESS, USE_LOCAL_FALLBACK


class SetupBudgetField(TypedDict):
    stateKey: str
    settingKey: str
    inputId: str
    settingsInputId: str
    title: str
    label: str
    note: str
    default: float
    min: str
    step: str


class SetupSubscriptionField(TypedDict):
    """A wizard declaration (not a setting) sent with ``POST /api/onboarding/complete``."""

    id: str
    payloadKey: str
    label: str
    note: str


class SetupContract(TypedDict):
    version: int
    hostMode: Literal["web", "desktop"]
    steps: List[SetupStep]
    providerFields: List[SetupProviderField]
    providerProfiles: Dict[str, SetupProviderProfile]
    modelSlots: List[SetupModelSlot]
    reviewModes: List[SetupModeChoice]
    runtimeModes: List[SetupModeChoice]
    localRoutingModes: List[SetupLocalRoutingMode]
    budgetFields: List[SetupBudgetField]
    subscriptionFields: List[SetupSubscriptionField]


class SetupLocalPreset(TypedDict):
    label: str
    source: str
    filename: str
    contextLength: int
    chatFormat: str


class SetupInitialState(TypedDict):
    """The wizard's starting values. Besides these keys it carries one key per
    ``contract.providerFields[].stateKey`` (a configured secret reads as ``secretPlaceholder``,
    never the secret), per ``contract.budgetFields[].stateKey`` and per ``contract.modelSlots[].stateKey``."""

    safetyModel: str
    providerProfile: str
    reviewEnforcement: str
    runtimeMode: str
    skillsRepoPath: str
    localPreset: str
    localSource: str
    localFilename: str
    localContextLength: int
    localGpuLayers: int
    localChatFormat: str
    localRoutingMode: str
    modelAccounts: Dict[str, Any]
    modelContextWindows: Dict[str, Any]
    processingPreference: str
    modelProcessingPreferences: Dict[str, Any]


class SetupBootstrap(TypedDict):
    """``settings_setup_contract.build_setup_bootstrap``: what the builtin wizard page embeds."""

    hostMode: Literal["web", "desktop"]
    freshInstall: bool  # display provenance only; completion re-proves install eligibility
    supportsLocalRuntimeControls: bool
    stepOrder: List[str]
    modelDefaults: Dict[str, Dict[str, str]]  # provider profile -> slot -> model id
    localPresets: Dict[str, SetupLocalPreset]
    modelSuggestions: List[str]
    secretPlaceholder: str  # a prefilled credential that is configured holds this marker
    contract: SetupContract
    initialState: SetupInitialState


class OnboardingBootstrapResponse(TypedDict):
    """``GET /api/onboarding/bootstrap``: a read that writes nothing. ``ready`` is the readiness
    ``GET /api/onboarding`` answers with 204; setup completes through ``POST /api/onboarding/complete``."""

    ready: bool
    bootstrap: SetupBootstrap


class SettingsReadResponse(TypedDict, total=False):
    """``GET /api/settings``: every effective setting key with its value (secrets masked: a
    password-class key reads as the configured placeholder, others as a short prefix plus
    ``...``) and ``_meta``. Sending a masked value back means "unchanged"; ``""`` clears."""

    _meta: SettingsMeta


class SettingsWriteRefusal(TypedDict):
    """A settings-document writer's failure (Settings, onboarding, owner controls, language).
    ``saved`` is the durable fact: false = nothing was written; true = the bytes landed and a
    later step failed (``post_commit_failed``: retry that step, never re-save); null = unknown,
    the writer outlived the wait: re-read before deciding. An unknown outcome is no retry authority."""

    error: str
    saved: Optional[bool]
    code: NotRequired[str]
    post_commit_failed: NotRequired[str]


class AuthLoginRequest(TypedDict):
    """``POST /auth/login`` as JSON (the gate also takes the form fields ``password``/``next``)."""

    password: str
    next: NotRequired[str]  # a same-origin path to return to; anything else becomes "/"


class AuthLoginResponse(TypedDict):
    """JSON login success; the session is the HttpOnly ``ouroboros_auth`` cookie set beside it."""

    ok: bool
    next: str


class ContractBody(TypedDict):
    """One request or response body: its media type and the ``schemas`` names it may take
    (empty = the shape is not published)."""

    content_type: str
    schemas: List[str]


class ContractHttpEntry(TypedDict):
    # ANY is the extension catch-all: GET/HEAD/POST/PUT/DELETE/PATCH, not OPTIONS.
    method: str
    path: str  # Starlette template: {name} is one segment, {name:path} the rest of the path
    path_params: List[str]
    auth: Literal["gate", "public"]  # gate: needs a session when the core enforces its network password
    typed: bool  # False: the route is published without shapes (an explicit gap, not an empty schema)
    query: NotRequired[List[str]]
    headers: NotRequired[List[str]]  # request headers the handler reads; a trailing "!" marks a required one
    request: NotRequired[ContractBody]
    responses: NotRequired[Dict[str, ContractBody]]  # by HTTP status
    note: NotRequired[str]


class ContractWsEntry(TypedDict):
    type: str
    inbound: NotRequired[str]  # schema of the client -> server frame
    outbound: NotRequired[str]  # schema of the server -> client frame


class ContractWs(TypedDict):
    path: str
    messages: List[ContractWsEntry]
    extension_namespace: str


class GatewayContractResponse(TypedDict):
    """``GET /api/contract``: the gateway ABI derived from the router-verified endpoint index,
    the WS envelope index and the contract TypedDicts (``gateway/schema.py``). Unknown fields are
    additive; an older core without this route predates the published contract."""

    abi_version: str
    runtime_version: str
    core_paths: List[str]  # prefixes a same-origin proxy routes to the core
    interface_paths: List[str]  # the replaceable main interface ("/" and "/static")
    http: List[ContractHttpEntry]
    ws: ContractWs
    schemas: Dict[str, Dict[str, Any]]


# --- Journey bodies of routes whose handlers check fields by hand ---------------------------

class OwnerRuntimeModeRequest(TypedDict):
    """``POST /api/owner/runtime-mode``; the mode applies when the core next starts."""

    mode: Literal["light", "advanced", "pro", "cyber_pro"]


class OwnerContextModeRequest(TypedDict):
    """Save the room's context for future actors; existing actors retain theirs."""

    mode: Literal["nano", "low", "max"]
    project_id: NotRequired[str]


class OwnerEffortRangeRequest(TypedDict):
    min: str
    recommended: str
    max: str
    project_id: NotRequired[str]


class OwnerSafetyModeRequest(TypedDict):
    kind: NotRequired[Literal["your_work", "consciousness"]]
    mode: Literal["full", "light", "off"]


class OwnerSafetyRulesRequest(TypedDict):
    rules: str


class OwnerSafetyRulesResponse(TypedDict):
    ok: bool
    rules: str


class OwnerSafetyModelRequest(TypedDict):
    settings: Dict[str, Any]


class OwnerSafetyModelResponse(TypedDict):
    ok: bool


class OwnerEnabledRequest(TypedDict):
    """``{enabled}`` as a JSON boolean: ``POST /api/owner/auto-grant``, ``/api/desktop/autostart``,
    ``/api/desktop/background`` and ``PATCH /api/claudexor/credential-profiles/...``. Send only this
    key: the two desktop routes refuse a body with any other (400)."""

    enabled: bool


class PlaceConnectRequest(TypedDict):
    """Connect an installed connector; omitted account and conversation ids default to empty."""

    connector: str
    account_id: NotRequired[str]
    conversation_id: NotRequired[str]


class PlaceRoomSummary(TypedDict):
    """One captured local conversation; an unknown open-focus count stays null."""

    room_chat_id: int
    unanswered_count: Optional[int]
    address: str


class PlaceRoomsResponse(TypedDict):
    place_id: str
    rooms: List[PlaceRoomSummary]


class PlaceOwnerAccountRequest(TypedDict):
    """Confirm a person; omitted account_id uses the place's account."""

    platform_actor_id: str
    label: NotRequired[str]
    account_id: NotRequired[str]


class PlaceUpdateRequest(TypedDict, total=False):
    """Atomic owner choices. Send a nonempty body; expected_revision compares before writing."""

    state: Literal["active", "silent", "paused"]
    wakes: List[str]
    add_note: str
    remove_note: str
    working_folder: str
    model: str
    confirm_owner_account: PlaceOwnerAccountRequest
    revoke_owner_account: str
    expected_revision: int


class SettingsSecretRequest(TypedDict, total=False):
    """``POST /api/settings/secret``: exactly one selector, ``{key}``, ``{mcp_server_id}`` or
    ``{mcp_server_id, header_name}``. Reads one stored secret on purpose; writes nothing."""

    key: str
    mcp_server_id: str
    header_name: str


class SettingsSecretResponse(TypedDict):
    value: str  # "" when the secret is not set; answered with Cache-Control: no-store


class ChatUploadDeleteRequest(TypedDict):
    """``DELETE /api/chat/upload``: only a pending upload (one no message accepted yet) is deleted."""

    filename: str  # the stored id ``POST /api/chat/upload`` returned


class ChatUploadDeleteResponse(TypedDict):
    ok: bool
    filename: str


class ProjectListResponse(TypedDict):
    """``GET /api/projects``: active and deleting projects, at most 200, most recently active first.
    Rows are the compact projection ``GET /api/state`` also carries: id, name, pinned, chat_id, working_dir,
    provenance, last_active_at, lifecycle, routing_generation, visible_revision, delete_error and
    has_thread_activity."""

    projects: List[ProjectEntry]


class ProjectCreateResponse(TypedDict):
    """``POST /api/projects``: ``project`` is the full registry row. ``project.created`` is false when
    the id already existed and no source was named: an idempotent repeat that changes nothing and
    announces nothing. ``init_git_warnings``/``init_git_skipped`` disclose what an ``init_git``
    attach snapshot left out."""

    project: ProjectEntry
    init_git_warnings: NotRequired[List[Any]]
    init_git_skipped: NotRequired[List[str]]


class ProjectUpdateRequest(TypedDict, total=False):
    """``POST /api/projects/{project_id}/update``: name and/or pin; at least one."""

    name: str  # 1 to 80 characters
    pinned: bool


class ProjectUpdateResponse(TypedDict):
    project: Optional[ProjectEntry]


class ProjectFromTaskRequest(TypedDict):
    """``POST /api/projects/from-task``. The default ``id`` is ``task-<task_id>``: a request naming no
    other room joins the project the task's work already has. ``project_id`` is an alias of ``id``.
    Without ``name`` the core names the project, which may wait for one model call."""

    task_id: str
    id: NotRequired[str]
    project_id: NotRequired[str]
    name: NotRequired[str]
    objective_hint: NotRequired[str]  # the owner's original request, a naming hint


class ProjectTaskBinding(TypedDict, total=False):
    """The task's immutable binding to its project."""

    task_id: str
    project_id: str
    project_chat_id: int
    bound_at: str
    source_ref: Dict[str, Any]  # the owner message the task came from
    origin_absent: Any  # present instead of source_ref when no origin message is known
    source_text: str


class ProjectFromTaskResponse(TypedDict):
    """``handoff_id`` is derived by the core from the project, the task and its origin message, so a
    repeat for the same task answers the same id; it is not a caller key. ``adopted`` is true when
    the task joined a project its work already had. ``handoff_receipt``: durable, already_delivered,
    unregistered, unavailable, unbound, origin_unproven or not_routable."""

    project: ProjectEntry
    binding: ProjectTaskBinding
    adopted: bool
    handoff_receipt: str
    handoff_id: str


class FileBrowserEntry(TypedDict):
    """One ``GET /api/files/list`` row; directories come first."""

    name: str
    path: str  # relative to the file-browser root, as every other files route takes it
    type: Literal["dir", "file"]
    is_symlink: bool
    size: NotRequired[int]  # files only


class FileBrowserListResponse(TypedDict):
    """``GET /api/files/list``: one directory of the file-browser root, at most 500 entries
    (``truncated``). The root is the configured file-browser folder; a loopback request without
    one browses the home folder, a remote one is refused (400)."""

    root_path: str
    path: str
    display_path: str
    parent_path: Optional[str]
    breadcrumb: List[Dict[str, str]]  # {name, path}
    entries: List[FileBrowserEntry]
    truncated: bool
    default_path: str
    default_display_path: str


class FileBrowserReadResponse(TypedDict):
    """``GET /api/files/read``: text is capped (256 KiB or 120,000 characters, ``truncated``);
    an image or PDF carries ``media_type`` and a ``content_url`` instead of text."""

    root_path: str
    path: str
    display_path: str
    name: str
    size: int
    is_text: bool
    is_image: bool
    is_pdf: bool
    content: str
    truncated: bool
    media_type: NotRequired[str]
    content_url: NotRequired[str]  # /api/files/content?path=...


class FileMkdirRequest(TypedDict):
    name: str
    path: NotRequired[str]  # the parent directory, default "."


class FileWriteRequest(TypedDict):
    """Writes unconditionally, so repeating it is safe. ``create: true`` allows a new file."""

    path: str
    content: str
    create: NotRequired[bool]


class FilePathRequest(TypedDict):
    """``POST /api/files/delete``: a directory is deleted recursively; a repeat answers 404."""

    path: str


class FileTransferRequest(TypedDict):
    """``POST /api/files/transfer``: a repeated copy answers 409 (the target exists), a repeated
    move 404 (the source is gone)."""

    source_path: str
    destination_dir: NotRequired[str]  # default "."
    mode: NotRequired[Literal["copy", "move"]]  # default copy


class FileChangeResponse(TypedDict, total=False):
    """Answer of the file-browser writers: upload and write ``{ok, path, display_path, name,
    size}`` (write adds ``created`` for a new file), mkdir ``{..., name, type: "dir"}``, transfer
    ``{ok, mode, path, display_path, name, type}``, delete ``{ok, path, type}``."""

    ok: Required[bool]
    path: Required[str]
    display_path: str
    name: str
    size: int
    type: str  # dir | file | symlink
    mode: Literal["copy", "move"]
    created: bool


class ScheduleTriggerRequest(TypedDict):
    type: Literal["cron", "once"]
    expr: NotRequired[str]  # cron: five fields
    run_at: NotRequired[str]  # once: an ISO 8601 instant


class ScheduleTaskTemplate(TypedDict, total=False):
    """What a schedule starts: an ordinary task, never a workspace or drive field."""

    type: Literal["task"]
    text: str
    priority: int
    metadata: Dict[str, Any]
    project_id: str
    reasoning_effort: str


class ScheduleUpsertRequest(TypedDict, total=False):
    """``POST /api/schedules``. A new ``id`` creates, an existing one edits; without an ``id`` a
    repeat creates another schedule. ``cron`` is accepted as an alias of ``trigger.expr``."""

    id: str
    name: str
    description: str
    enabled: bool  # default true
    timezone: str  # an IANA name
    trigger: Required[ScheduleTriggerRequest]
    task: ScheduleTaskTemplate
    reason: str


class ScheduleActionRequest(TypedDict):
    """``POST /api/schedules/{schedule_id}/action``: a lifecycle action governs future runs and
    never stops a run already admitted."""

    action: Literal["disable", "delete", "restore"]
    reason: str
    expected_hold_id: NotRequired[str]
    relation: NotRequired[str]


class TaskResumeResponse(TypedDict, total=False):
    """``POST /api/tasks/{task_id}/resume`` after budget, owner, sleep or technical authority pause. The other keys say which
    resume path ran (``same_generation``, ``owner_pause_released``, ``never_started``, ``warm``,
    ``grant_id``, ``held_siblings``, ``snapshot_persisted``, ``exact_continuation``,
    ``grant_generation``, ``paused_duration_sec``, ``eligible_descendants``, ``late_phase``).
    Not idempotent: a repeat is refused (409 or 404); read the task to learn what happened."""

    ok: Required[bool]
    task_id: Required[str]
    root_task_id: str
    late_phase: Literal["resumed", "settled"]
    tree_only: bool
    owner_pause_released: bool
    snapshot_persisted: bool
    exact_continuation: bool
    grant_id: str
    grant_generation: int
    paused_duration_sec: float
    eligible_descendants: List[str]
    held_siblings: List[str]
    rebound_held_siblings: List[str]
    released_hold: str


# Last, not first: ``contracts.py`` imports ServedUiState from this module, so the shared
# SettingsMeta is bound after this module's own envelopes exist, whichever module loads first.
from ouroboros.gateway.contracts import ProjectEntry, SettingsMeta  # noqa: E402

__all__ = [
    "ServedUiState",
    "SetupStep", "SetupProviderField", "SetupProviderProfile", "SetupModelSlot", "SetupModeChoice",
    "SetupLocalRoutingMode", "SetupBudgetField", "SetupSubscriptionField", "SetupContract",
    "SetupLocalPreset", "SetupInitialState", "SetupBootstrap", "OnboardingBootstrapResponse",
    "SettingsReadResponse", "SettingsWriteRefusal",
    "AuthLoginRequest", "AuthLoginResponse",
    "ContractBody", "ContractHttpEntry", "ContractWsEntry", "ContractWs", "GatewayContractResponse",
    "OwnerRuntimeModeRequest", "OwnerContextModeRequest", "OwnerEffortRangeRequest", "OwnerSafetyModeRequest", "OwnerSafetyRulesRequest", "OwnerSafetyRulesResponse", "OwnerSafetyModelRequest", "OwnerSafetyModelResponse", "OwnerEnabledRequest",
    "PlaceConnectRequest", "PlaceOwnerAccountRequest", "PlaceUpdateRequest",
    "SettingsSecretRequest", "SettingsSecretResponse", "ChatUploadDeleteRequest", "ChatUploadDeleteResponse",
    "ProjectListResponse", "ProjectCreateResponse", "ProjectUpdateRequest", "ProjectUpdateResponse",
    "ProjectFromTaskRequest", "ProjectTaskBinding", "ProjectFromTaskResponse",
    "FileBrowserEntry", "FileBrowserListResponse", "FileBrowserReadResponse", "FileMkdirRequest", "FileWriteRequest", "FilePathRequest",
    "FileTransferRequest", "FileChangeResponse",
    "ScheduleTriggerRequest", "ScheduleTaskTemplate", "ScheduleUpsertRequest", "ScheduleActionRequest",
    "TaskResumeResponse",
]
