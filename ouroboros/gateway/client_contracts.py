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

from ouroboros.delegate_custody_usage import CostCustody
from ouroboros.cost_projection import CostPresentation
from ouroboros.gateway.work_contracts import ChatModelRoute

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
    managed: bool  # a deployment policy fixes it: shown, never edited or written


class SetupModeChoice(TypedDict):
    """A ``runtimeModes`` row; ``value`` is what the setting stores."""

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

    connections: List[Dict[str, Any]]  # masked OUROBOROS_CONNECTIONS rows, keyed by kind/name
    legacyCompatibleBaseUrl: str  # saved OPENAI_BASE_URL, display only: completion never writes it
    safetyModel: str
    providerProfile: str
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
    automaticModels: List[str]  # model-slot stateKeys neither the document nor an operator's variable sets


class SetupConsciousnessOffer(TypedDict):
    """``settings_setup_contract.consciousness_offer``: the Summary's optional, unticked Consciousness row."""

    offered: bool  # setup has never completed on this install; a completed setup shows no row
    enabled: Optional[bool]  # the control as the state shows it; null = unknown
    autonomy: Literal["observe", "act", "full"]
    dailyUsd: float  # rolling-24h recorded-spend allowance; subscription quota is separate
    maxTasks: int  # how many of its own tasks may run at once; 0 = it starts none


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
    consciousnessOffer: SetupConsciousnessOffer  # display; a ticked row sends the ordinary Start after the save
    deployment: Optional["SettingsDeploymentPolicy"]  # Settings' ``_meta.policy_state.deployment``; null unmanaged


class OnboardingBootstrapResponse(TypedDict):
    """``GET /api/onboarding/bootstrap``: a read that writes nothing. ``ready`` is the readiness
    ``GET /api/onboarding`` answers with 204; setup completes through ``POST /api/onboarding/complete``."""

    ready: bool
    bootstrap: SetupBootstrap


class SettingsSaveRequest(TypedDict, total=False):
    """Open partial settings document; the two control fields are never stored."""
    if_unchanged: Dict[str, str]
    unset: List[str]


class SettingsChange(TypedDict, total=False):
    """Sanitized commit facts. A secret has no token; a withheld value has no inverse."""
    old: Any
    new: Any
    old_absent: bool
    token: str
    withheld: Literal["secret", "structure"]


class SettingsReceipt(TypedDict, total=False):
    changes: Dict[str, SettingsChange]
    receipt: Literal["unavailable"]


class SettingsTiming(TypedDict):
    effects: Dict[str, Literal["now", "next_call", "next_wake", "next_task", "restart", "local_model"]]
    tokens: Dict[str, str]


class SettingsRestartState(TypedDict, total=False):
    restart_required: bool
    restart_keys: List[str]
    restart_source_unknown_keys: List[str]
    unknown_keys: List[str]
    local_model: Dict[str, Any]
    summary: str
    running: Dict[str, Any]


class SettingsReadResponse(TypedDict, total=False):
    """``GET /api/settings``: every effective setting key with its value (secrets masked: a
    password-class key reads as the configured placeholder, others as a short prefix plus
    ``...``) and ``_meta``. Sending a masked value back means "unchanged"; ``""`` clears."""

    _meta: _contracts.SettingsMeta


class SettingsPolicyState(TypedDict):
    """``_meta.policy_state`` of ``GET /api/settings``: configured versus effective Access and Supervisor
    (``gateway.settings._build_policy_state``), plus what a managed deployment's policy fixes."""

    access: _contracts.SettingsPolicyAxis
    supervisor: _contracts.SettingsPolicyAxis
    running_task_snapshot: bool
    deployment: NotRequired[SettingsDeploymentPolicy]  # absent outside a managed deployment


class SettingsDeploymentChecker(TypedDict):
    """The organization's Safety Supervisor route; no connection value or secret."""

    model: str
    account: str
    fallbacks: List[str]
    connections: List[str]  # the connections its routes use, by picker id (``kind`` or ``kind@name``)
    fields: List[str]  # the flat settings its built-in routes read now (a compatible lane's pair included)
    local: bool  # a route answers from the local model server


class SettingsDeploymentRequirement(TypedDict):
    """One extension the policy requires: its pin and the custom settings it fixes, by name only; Settings and
    Skills add how the observing process sees it (``deployment_extensions.requirement_state``)."""

    name: str
    path: str
    content_hash: str
    settings: List[str]
    state: NotRequired[Literal["met", "unmet"]]
    reason: NotRequired[str]  # running, missing, collision, elsewhere, changed, review, grants, load_error, ...
    detail: NotRequired[str]  # the owner's sentence, with no fixed value in it
    observed_hash: NotRequired[str]  # the digest of other bytes found at the path
    process: NotRequired[Dict[str, Any]]  # role, pid, started (birth token) and observed_at of the observer


class SettingsDeploymentExtensions(TypedDict):
    """The policy's required extensions and what ordinary work does while one is not running ("" when none is);
    with an execution domain, ``trusted`` lists the only packages ({name, path, content_hash}) whose code the
    service runs."""

    when_unmet: Literal["hold", "warn", ""]
    required: List[SettingsDeploymentRequirement]
    trusted: NotRequired[List[Dict[str, str]]]


class SettingsDeploymentPolicy(TypedDict):
    """What a managed deployment's policy file fixes (``deployment_policy.policy_state``)."""

    source: Literal["deployment_policy"]
    version: int
    digest: str  # the first 16 hex digits of the file's SHA-256, as this server generation read it
    access_max: Optional[str]
    min_coverage: Optional[str]
    checker: Optional[SettingsDeploymentChecker]
    allowed_exits: List[str]  # agent repeat, human release after DANGEROUS, human unchecked release
    organization_rules: bool
    extensions: Optional[SettingsDeploymentExtensions]


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
    auth: Literal["gate", "public", "asset_lease"]  # asset_lease: scoped read capability, never a session
    typed: bool  # False: the route is published without shapes (an explicit gap, not an empty schema)
    query: NotRequired[List[str]]
    headers: NotRequired[List[str]]  # request headers the handler reads; a trailing "!" marks a required one
    request: NotRequired[ContractBody]
    responses: NotRequired[Dict[str, ContractBody]]  # HTTP status, then 4xx/5xx for unlisted errors
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
    call_socket: NotRequired[Dict[str, Any]]  # separate PCM/media protocol, see CallSocketContract
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


class OwnerSubagentsModeRequest(TypedDict):
    mode: Literal["off", "auto", "swarm"]
    project_id: NotRequired[str]


class OwnerChatModelRequest(TypedDict):
    """A Project follows Main or chooses a model; without project_id this writes Main itself."""

    kind: Literal["main", "model"]
    model: NotRequired[str]  # kind ``model`` only: the raw route, its source included
    use_local: NotRequired[bool]  # kind ``model`` only
    project_id: NotRequired[str]


class ChatModelChoice(TypedDict):
    """A room follows Main or chooses a route; a reference is never the same model pinned."""

    kind: Literal["main", "model"]
    model: NotRequired[str]  # kind ``model`` only: the raw route, its source included
    use_local: NotRequired[bool]  # kind ``model`` only


class OwnerChatModelResponse(TypedDict):
    ok: bool
    project_id: str
    model_choice: NotRequired[ChatModelChoice]  # Project response
    previous_model_choice: NotRequired[ChatModelChoice]
    main_route: NotRequired[ChatModelRoute]  # Main response
    previous_main_route: NotRequired[ChatModelRoute]
    account: NotRequired[Literal["kept", "auto", "none"]]
    window_reset: NotRequired[bool]


class OwnerSafetyModeRequest(TypedDict):
    kind: NotRequired[Literal["your_work", "consciousness"]]
    mode: Literal["full", "light", "off"]


class OwnerSafetyRulesRequest(TypedDict):
    rules: str


class OwnerSafetyRulesResponse(SettingsReceipt):
    ok: bool
    rules: str


class OwnerSafetyModelRequest(TypedDict):
    settings: Dict[str, Any]
    if_unchanged: NotRequired[Dict[str, str]]


class OwnerSafetyModelResponse(SettingsReceipt):
    ok: bool


class OwnerEnabledRequest(TypedDict):
    """``{enabled}`` as a JSON boolean: ``POST /api/owner/auto-grant``, ``/api/desktop/autostart``,
    ``/api/desktop/background`` and ``PATCH /api/claudexor/credential-profiles/...``. Send only this
    key: the two desktop routes refuse a body with any other (400)."""

    enabled: bool


class RecentWakeCheck(TypedDict):
    """One stored safety check, retaining both its chip and the tool it checked."""

    chip: str
    tool: str


_RecentWakeIdentity = TypedDict("_RecentWakeIdentity", {"as": NotRequired[Dict[str, Any]]})  # {label: str, owner: bool}


class RecentWake(_RecentWakeIdentity):
    """GET /api/consciousness recent_wakes: ledger-selected facts, never authored by the view."""

    task_id: str
    chat_id: Optional[int]
    cause: Literal["own", "focus"]
    place_label: str
    place_id: NotRequired[str]  # a focus wake's place, for its sheet
    conversation: str
    started_at: Optional[str]
    first_reply_at: NotRequired[Optional[str]]
    ended_at: Optional[str]
    live: bool
    phase: Literal["running", "waiting", "queued", "paused", "unknown", "working", "done", "warn", "error", "cancelled"]
    quiet: bool
    summary: NotRequired[str]
    note: Optional[str]
    tool_calls: NotRequired[Optional[int]]
    rounds: NotRequired[Optional[int]]
    checks: List[RecentWakeCheck]
    said_owner: NotRequired[bool]
    accounted_upper_bound_usd: NotRequired[Optional[float]]
    accounted_upper_bound_usd_with_children: NotRequired[Optional[float]]
    cost_presentation: NotRequired[Optional[CostPresentation]]
    cost_accounting_status: NotRequired[Literal["available", "unavailable"]]
    cost_accounting_error: NotRequired[str]
    cost_final: NotRequired[bool]
    cost_with_children_partial: NotRequired[bool]
    cost_custody: NotRequired[Optional[CostCustody]]
    unknown_unmetered: NotRequired[Optional[int]]
    non_final_rows: NotRequired[Optional[int]]
    reserved_usd: NotRequired[Optional[float]]
    unresolved_upper_bound_usd: NotRequired[Optional[float]]
    ledger_integrity_degraded: NotRequired[Optional[bool]]


class RecentWakesView(TypedDict):
    """Recent-wakes fields of /api/consciousness; status fields remain an open JSON shape."""

    recent_wakes: Optional[List[RecentWake]]
    recent_wakes_coverage: NotRequired[str]  # names the cap or the typed read failure; absent without a clock


class PlaceConnectRequest(TypedDict):
    """Connect Silent; ids default to empty. An instance label is prefixed with the service. An Add's note,
    description and circle_id land with a new place only; a connected one answers 409 place_exists."""

    connector: str
    account_id: NotRequired[str]
    conversation_id: NotRequired[str]
    label: NotRequired[str]
    note: NotRequired[str]
    description: NotRequired[str]
    circle_id: NotRequired[str]


class PlaceCircleChoice(TypedDict):
    """The circle a place is in and who placed it there; the words and their author are the circle's."""

    circle_id: str
    placed_by: str
    placed_at: str


class PlaceDescription(TypedDict):
    """A short purpose line under the place's name: never a rule or permission."""

    text: str
    author: str
    at: str


class CircleSummary(TypedDict):
    """One circle's words, authorship and model; an empty model inherits consciousness."""

    circle_id: str
    name: str
    text: str
    text_author: str
    text_at: str
    revision: int
    model: str


class CircleCreateRequest(TypedDict):
    """A new circle last in the order; its words are the owner's."""

    name: str
    text: NotRequired[str]
    model: NotRequired[str]


class CircleUpdateRequest(TypedDict, total=False):
    """Change name, words or model; expected_revision compares the circle's revision before writing."""

    name: str
    text: str
    model: str
    expected_revision: int


class CircleDeleteRequest(TypedDict, total=False):
    expected_revision: int


class PlaceIcon(TypedDict, total=False):
    """Sanitized host mark or SVG geometry; an unavailable mark is {}."""

    kind: Literal["host", "svg"]
    name: str
    svg: str


class PlaceReadiness(TypedDict):
    text: str
    tone: str


class PlaceModelRoute(TypedDict):
    """The model a place thinks with now, and the scope that chose it."""

    value: str
    layer: Literal["place", "circle", "consciousness"]


class PlaceSummary(TypedDict):
    """One projection for list, write/conflict responses and the model's place map."""

    place_id: str
    connector: str
    provider: str
    account_id: str
    conversation_id: str
    label: str
    state: Literal["active", "silent", "paused"]
    only_hears: NotRequired[bool]  # derived per read: true while consciousness is off (or Panic) and the place not Paused
    wakes: List[Dict[str, str]]
    notes: List[Dict[str, str]]
    owner_accounts: List[Dict[str, str]]
    working_folder: str
    model: str
    model_route: PlaceModelRoute
    context_topics: List[str]
    legacy_binding_ids: List[str]
    revision: int
    thread_id: str
    destination: str
    scope: str
    contribution_id: str
    reply_destination: Optional[Dict[str, str]]
    icon: PlaceIcon
    readiness: PlaceReadiness
    wake_suggestions: List[str]
    facts: NotRequired[List[Any]]
    source: NotRequired[Dict[str, Any]]
    settings: NotRequired[str]
    safety_mode: Literal["off", "light", "full"]
    labels: Dict[str, str]
    rooms: Optional[List[Dict[str, Any]]]
    unanswered: Optional[int]  # open focus turns, not owed replies
    last_activity_at: str
    activity_coverage: str
    spent_24h_usd: Optional[float]
    account_candidates: List[Dict[str, Any]]
    shared_owner_accounts: List[Dict[str, str]]  # confirmed in another place of this bot/app; count here
    circle: Optional[PlaceCircleChoice]
    description: Optional[PlaceDescription]


class PlaceDisconnectRequest(TypedDict, total=False):
    expected_revision: int


class PlaceRoomSummary(TypedDict):
    """One captured local conversation; an unknown open-focus count stays null."""

    room_chat_id: int
    unanswered_count: Optional[int]
    address: str
    label: Optional[str]  # latest captured arrival's conversation title/name, absent stays null
    last_at: str  # latest captured arrival, not latest remote message or outgoing receipt
    conversation_id: str
    thread_id: str


class PlaceRoomsCoverage(TypedDict):
    """Discovery scope and store-wide chat import/gaps, not provider completeness."""
    discovery: Literal['arrivals']
    importing: bool
    gaps: bool


class PlaceRoomsResponse(TypedDict):
    place_id: str
    rooms: List[PlaceRoomSummary]
    coverage: PlaceRoomsCoverage


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
    circle_id: str  # "" takes it out of its circle
    description: str  # "" clears it
    expected_revision: int


class SettingsSecretRequest(TypedDict, total=False):
    """``POST /api/settings/secret``: exactly one selector, ``{key}``, ``{mcp_server_id}``,
    ``{mcp_server_id, header_name}`` or ``{connection, field}`` (an added API key ``kind@name`` and
    one of its secret fields). Reads one stored secret on purpose; writes nothing."""

    key: str
    mcp_server_id: str
    header_name: str
    connection: str
    field: str


class SettingsSecretResponse(TypedDict):
    value: str  # "" when the secret is not set; answered with Cache-Control: no-store


class ChatUploadDeleteRequest(TypedDict):
    """``DELETE /api/chat/upload``: only a pending upload (one no message accepted yet) is deleted."""

    filename: str  # the stored id ``POST /api/chat/upload`` returned


class ChatUploadDeleteResponse(TypedDict):
    ok: bool
    filename: str


class ProjectListResponse(TypedDict):
    """``GET /api/projects``: active and deleting projects, at most 200, newest message first (``last_message_at``);
    ``?all=1`` lists every one of them. ``?id=<project_id>`` answers that project while active or deleting,
    ``?chat_id=<n>`` the active project of that room; either answers none otherwise.
    Rows are the compact projection ``GET /api/state`` also carries: id, name, pinned, chat_id, working_dir,
    provenance, last_active_at, last_message_at, lifecycle, optional closed, routing_generation, visible_revision,
    delete_error and has_thread_activity."""

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
    """``POST /api/projects/{project_id}/update``: name, pin or closure; at least one."""

    name: str  # 1 to 80 characters
    pinned: bool
    closed: bool  # true closes as owner; false removes the closure record


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


class TaskResumeRequest(TypedDict, total=False):
    """The optional body of ``POST /api/tasks/{task_id}/resume``: the held calls whose outcome is
    unknown (a refusal's ``acceptable_unknown``) the owner goes on past, and why. Recorded, never re-run."""

    accept_unknown: List[str]
    reason: str


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
    # A Continue whose earlier attempt's writers all still run returned to its own writer hold: it starts by itself
    # once ``blockers`` (that census) settle, unless a newer Pause, Stop or Panic replaces the wait.
    held: bool
    blockers: List[Dict[str, Any]]
    # The still-closed Pause's census settled without these unreadable rows nothing ties to the task:
    # {"rows": [{"task_id", "reason"}] (at most 20), "count"}; ``message`` words them.
    membership_gaps: Dict[str, Any]
    message: str


# Resolve shared shapes after either module has finished loading.
from ouroboros.gateway import contracts as _contracts  # noqa: E402
from ouroboros.gateway.contracts import ProjectEntry  # noqa: E402

__all__ = [
    "ServedUiState",
    "SetupStep", "SetupProviderField", "SetupProviderProfile", "SetupModelSlot", "SetupModeChoice",
    "SetupLocalRoutingMode", "SetupBudgetField", "SetupSubscriptionField", "SetupContract",
    "SetupLocalPreset", "SetupInitialState", "SetupConsciousnessOffer", "SetupBootstrap", "OnboardingBootstrapResponse",
    "SettingsSaveRequest", "SettingsChange", "SettingsReceipt", "SettingsTiming", "SettingsRestartState",
    "SettingsReadResponse", "SettingsPolicyState", "SettingsDeploymentChecker", "SettingsDeploymentRequirement", "SettingsDeploymentExtensions",
    "SettingsDeploymentPolicy", "SettingsWriteRefusal",
    "AuthLoginRequest", "AuthLoginResponse",
    "ContractBody", "ContractHttpEntry", "ContractWsEntry", "ContractWs", "GatewayContractResponse",
    "OwnerRuntimeModeRequest", "OwnerContextModeRequest", "OwnerEffortRangeRequest", "OwnerSubagentsModeRequest", "OwnerChatModelRequest", "ChatModelChoice", "ChatModelRoute", "OwnerChatModelResponse", "OwnerSafetyModeRequest", "OwnerSafetyRulesRequest", "OwnerSafetyRulesResponse", "OwnerSafetyModelRequest", "OwnerSafetyModelResponse", "OwnerEnabledRequest",
    "RecentWake", "RecentWakeCheck", "RecentWakesView",
    "PlaceConnectRequest", "PlaceIcon", "PlaceReadiness", "PlaceModelRoute", "PlaceSummary", "PlaceDisconnectRequest",
    "PlaceOwnerAccountRequest", "PlaceUpdateRequest", "PlaceCircleChoice", "PlaceDescription",
    "CircleSummary", "CircleCreateRequest", "CircleUpdateRequest", "CircleDeleteRequest",
    "SettingsSecretRequest", "SettingsSecretResponse", "ChatUploadDeleteRequest", "ChatUploadDeleteResponse",
    "ProjectListResponse", "ProjectCreateResponse", "ProjectUpdateRequest", "ProjectUpdateResponse",
    "ProjectFromTaskRequest", "ProjectTaskBinding", "ProjectFromTaskResponse",
    "FileBrowserEntry", "FileBrowserListResponse", "FileBrowserReadResponse", "FileMkdirRequest", "FileWriteRequest", "FilePathRequest",
    "FileTransferRequest", "FileChangeResponse",
    "ScheduleTriggerRequest", "ScheduleTaskTemplate", "ScheduleUpsertRequest", "ScheduleActionRequest",
    "TaskResumeRequest", "TaskResumeResponse",
]
