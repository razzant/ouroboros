"""Product-management envelopes beside ``contracts.py`` (skills, marketplaces, agent accounts, local model,
MCP, model catalog, updates, core control): descriptive TypedDicts, the twin of ``web/modules/management_contract_types.js``
(twins older tests read stay in ``api_types.js``). Account routes relay the bundled Claudexor engine's own API v2 objects
(protocol major 3, release pinned in ``ouroboros/claudexor_runtime_pin.json``, source https://github.com/razzant/claudexor), named by their engine schema type."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from ouroboros.gateway.widgets import ExtensionLiveSnapshot
from ouroboros.gateway.contributions import UiContributionsResponse

try:  # Python 3.11+
    from typing import Literal, NotRequired, Required, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import Literal, NotRequired, Required, TypedDict  # type: ignore[assignment]


# --- Skills and extensions ---------------------------------------------------------------------

class InstalledSkill(TypedDict, total=False):
    """One ``GET /api/extensions`` ``skills`` row, sorted by name; a name two skills claim gives one
    not-live row each (``identity_collision``). ``live_reason`` says why a skill is or is not live."""

    name: Required[str]
    type: Required[str]  # instruction | script | extension
    version: Required[str]
    description: Required[str]
    enabled: Required[bool]
    permissions: Required[List[str]]
    conflicts: Required[List[str]]
    load_error: Required[Optional[str]]
    is_self_authored: Required[bool]
    source: Required[str]  # native | clawhub | ouroboroshub | external | self_authored | user_repo
    content_hash: Required[str]  # the revision every expected_content_hash names
    payload_root: Required[str]  # skills/<bucket>/<dir>, or ""
    installed_at: Required[str]
    removal: str  # uninstall | delete | "": the server's available removal action
    review_status: str  # clean | warnings | blockers | pending
    review_stale: bool
    review_gate: Dict[str, Any]
    author_disposition: Dict[str, Any]
    reviewed_content_hash: str
    executable_review: bool
    review_profile: str
    owner_attestable: Optional[bool]
    submit_hub: Dict[str, Any]
    conflict: Optional[Dict[str, Any]]  # {code: "skill_conflict", skills, omitted}
    desired_live: bool
    live_loaded: bool
    live_reason: str
    process: str  # "", server or worker
    server_reconcile: str
    health_regressed: bool
    last_known_good: Optional[Dict[str, Any]]
    health_observations: Dict[str, Any]
    dispatch_live: bool
    review_findings: List[Dict[str, Any]]
    skill_review: Dict[str, Any]  # {} or {current, history, history_omitted}
    grants: Dict[str, Any]  # requested/granted/missing keys and permissions, all_granted, usable
    published: Optional[Dict[str, Any]]
    published_malformed: bool
    presence_runtime: Dict[str, Any]  # a reviewed presence profile's runtime choice
    provenance: Dict[str, Any]  # a ClawHub install record
    identity_collision: bool
    # The install record joined with the cached catalog of its source (skill_source_catalog.install_listing): source {id, name, kind, part}, official (exact endorsed bytes),
    # release, basis, base_known, modified (null: base unknown), update_available {version, digest} | null, withdrawn ({reason, version} for a withdrawn release), left_source,
    # turn_on_when_ready, installed_by, installed_at, and reason and task_id when the record has them (an install Ouroboros made says why and in which task); check {by, notes,
    # at}, pending_update {version, digest}, and update_rejected {version, digest, reason: review_blockers|review_unfinished|preflight_failed, at}. null: not installed from a
    # source (made here, a local folder, a built-in seed).
    install: Optional[Dict[str, Any]]
    category: List[str]  # the source's category ids for this skill, [] when it has none


class ExtensionsIndexResponse(UiContributionsResponse):
    """``GET /api/extensions``: every installed skill and what the extension loader has live.
    A passive read: no network, no repair."""

    skills: List[InstalledSkill]
    live: ExtensionLiveSnapshot


class ExtensionManifestResponse(TypedDict, total=False):
    """``GET /api/extensions/{skill}/manifest``: the loaded manifest and the same review facts as
    an ``InstalledSkill`` row."""

    name: Required[str]
    manifest: Required[Dict[str, Any]]  # name, description, version, type, entry, permissions, ...
    enabled: Required[bool]
    content_hash: Required[str]
    load_error: Required[Optional[str]]
    review_status: str
    review_stale: bool
    review_gate: Dict[str, Any]
    author_disposition: Dict[str, Any]
    reviewed_content_hash: str
    executable_review: bool
    review_profile: str
    owner_attestable: Optional[bool]
    submit_hub: Dict[str, Any]


class ExtensionSettingsSectionResponse(TypedDict):
    """``GET /api/extensions/{skill}/settings_section``; an unknown skill has no sections."""

    skill: str
    sections: List[Dict[str, Any]]  # {skill, section_id, title, render, key}


class SkillToggleRequest(TypedDict):
    """``POST /api/skills/{skill}/toggle``. ``enabled`` is a boolean, or 0/1, or one of the strings
    1, true, yes, on, 0, false, no, off. Repeating it is harmless."""

    enabled: Union[bool, int, str]
    expected_content_hash: NotRequired[str]  # the revision shown; a different live one answers 409


class SkillRevisionRequest(TypedDict, total=False):
    """A body naming the skill revision the owner saw (``POST /api/owner/skills/{skill}/attest-review``)."""

    expected_content_hash: str


class SkillGrantRequest(TypedDict, total=False):
    """``POST /api/skills/{skill}/grants``: the first present of ``items``, ``keys`` or ``granted_keys``;
    each item a key or permission name, or ``{value|key|permission|name}``. Only what the current
    manifest requests is grantable; a repeat merges into the saved grants."""

    items: List[Any]
    keys: List[Any]
    granted_keys: List[Any]
    expected_content_hash: str
    enable_after_grant: bool  # explicit Enable intent stays in this same lifecycle job


class SkillDeleteRequest(TypedDict, total=False):
    """``POST /api/skills/{skill}/delete``: only an external or self-authored skill, at exactly
    ``skills/external/<name>``. It removes the payload and the skill's state; a repeat answers 404."""

    payload_root: str  # default skills/external/<name>
    expected_content_hash: str


class SkillToggleResponse(TypedDict, total=False):
    """``POST /api/skills/{skill}/toggle`` success. A refusal answers ``SkillActionRefusal``; a
    toggle with an ``extension_action`` also broadcasts ``extension_lifecycle`` when its job ends.
    Enabling installs missing packages; when that outlasts the answer bound the body is
    ``{skill, ok, pending: true, job_id, message}`` with no ``enabled`` yet: follow the job on
    ``GET /api/skills/lifecycle-queue``."""

    skill: Required[str]
    ok: bool
    pending: bool
    job_id: str
    message: str
    enabled: bool
    review_status: str
    review_stale: bool
    review_gate: Dict[str, Any]
    executable_review: bool
    grants: Dict[str, Any]
    extension_action: Optional[str]  # extension_loaded | extension_unloaded | ... ; null for non-extensions
    extension_reason: Optional[str]
    process: str
    server_reconcile: str


class SkillActionRefusal(TypedDict, total=False):
    """A refused owner action on one skill; the HTTP status repeats ``status_code`` (404 unknown
    skill; 409 a changed revision, a conflict, a review that cannot run or missing grants; else
    400). Current facts are optional; a revision-guard refusal includes no revision, so
    re-read the installed-skill index before another action."""

    ok: bool
    error: Required[str]
    status_code: int
    skill: str
    code: str
    detail: str
    source: str
    content_hash: str
    review_status: str
    review_stale: bool
    review_gate: Dict[str, Any]
    executable_review: bool
    grants: Dict[str, Any]
    conflict: Dict[str, Any]
    deps_status: str
    enabled: bool
    extension_action: Optional[str]
    extension_reason: Optional[str]
    process: str
    server_reconcile: str
    load_error: Optional[str]
    allowed_keys: List[str]
    allowed_permissions: List[str]
    rejected_items: List[str]


class SkillDaemonsResponse(TypedDict):
    """``GET /api/skills/daemons``: companion processes, keyed ``<skill>:<name>``."""

    companions: Dict[str, Dict[str, Any]]  # {skill_name, name, pid, returncode, ports, retiring, ...}


class SkillLifecycleJob(TypedDict, total=False):
    """One job of the skill lifecycle lane; ``status``: queued, running, succeeded, failed or cancelled."""

    id: Required[str]  # skill-job-<32 hex>
    kind: Required[str]
    target: Required[str]
    source: str
    status: Required[str]
    message: str
    error: str
    dedupe_key: str
    chat_id: int
    group_id: str
    task_id: str
    root_task_id: str
    origin_task_id: str
    origin_root_task_id: str
    presentation_owner_task_id: str
    chat_task_id: str
    queued_at: str
    started_at: str
    finished_at: str
    age_seconds: float
    stale: bool
    stale_reason: str
    recovery_hint: str


class SkillLifecycleQueueResponse(TypedDict):
    """``GET /api/skills/lifecycle-queue``: the running job and the recent ones (at most 80, oldest
    first, kept in memory only). Poll it, or follow the ``skill_lifecycle`` progress rows."""

    active: Optional[SkillLifecycleJob]
    events: List[SkillLifecycleJob]


class SkillReviewResponse(TypedDict, total=False):
    """``POST /api/skills/{skill}/review`` (200 once the review ran, which takes minutes) and
    ``POST /api/owner/skills/{skill}/attest-review`` (200 when ``ok``, else 409). A missing skill or
    a review already running for the same revision answers ``status: "pending"`` with ``error``; for
    a running review ``job_id`` names it. ``GET /api/skills/{skill}/review-history/{job_id}`` reads a
    finished one."""

    skill: Required[str]
    status: Required[str]  # clean | warnings | blockers | pending
    content_hash: str
    error: str
    ok: bool  # attest only
    status_code: int  # attest refusal only
    pending: bool  # attest: the job outlasted the answer bound; review: the wave is still open (author left)
    message: str  # attest only, with pending
    reviewer_models: List[str]
    review_profile: str
    findings: List[Dict[str, Any]]
    raw_actor_records: List[Dict[str, Any]]
    advisory_result: Any
    review_gate: Dict[str, Any]
    executable_review: bool
    auto_flow: Any
    auto_granted_keys: List[str]
    requested_keys: List[str]
    auto_granted_permissions: List[str]
    requested_permissions: List[str]
    deps_status: str
    deps_error: str
    extension_action: Optional[str]
    extension_reason: Optional[str]
    extension_process: str
    extension_server_reconcile: str
    extension_load_error: Optional[str]
    extension_live_loaded: Optional[bool]
    record_id: str  # the review's ledger record
    reused: bool  # a settled record of the same subject answered at $0
    paid: bool
    seats: List[Dict[str, Any]]
    governance: Dict[str, Any]
    job_id: str
    job_status: str
    group_id: str
    review_round: int
    snapshot_attempt: int
    snapshot_revised: bool
    task_id: str
    root_task_id: str
    origin_task_id: str
    origin_root_task_id: str
    presentation_owner_task_id: str
    chat_id: int
    source: str
    terminal_reason: str
    executions: List[Dict[str, Any]]


class SkillReviewHistoryDetailResponse(TypedDict):
    """``GET /api/skills/{skill}/review-history/{job_id}``: one terminal review rendered as markdown
    (a 404 is final; a 503 is a retryable read failure)."""

    markdown: str
    status: str
    content_hash: str
    job_status: str


class SkillReconcileResponse(TypedDict):
    """``POST /api/skills/{skill}/reconcile``: bring one extension's live state in line with its
    saved state (the desktop launcher calls it after a restart). Broadcasts ``extension_lifecycle``
    when an action ran."""

    skill: str
    extension_action: Optional[str]
    extension_reason: Optional[str]
    live_loaded: bool
    load_error: Optional[str]
    process: str
    server_reconcile: str


# --- Marketplaces ------------------------------------------------------------------------------

class MarketplaceSkillSummary(TypedDict, total=False):
    """A catalog row. ClawHub rows carry the registry's fields (``badges`` and ``stats`` pass through
    as the registry sends them); OuroborosHub rows add ``source``, ``sanitized_name`` (the local
    name it installs as) and ``identity_conflict``."""

    slug: Required[str]
    display_name: str
    summary: str
    description: str
    latest_version: str
    versions: List[Any]
    license: str
    homepage: str
    os: List[str]
    requires_env: List[str]
    requires_bins: List[str]
    primary_env: str
    install_specs: List[Any]
    badges: Dict[str, Any]
    stats: Dict[str, Any]
    is_plugin: bool
    source: str
    sanitized_name: str
    identity_conflict: bool


class MarketplaceInstalledSkill(TypedDict, total=False):
    """One row of a marketplace's ``installed`` list; ClawHub rows add their install ``provenance``."""

    name: Required[str]
    type: str
    version: str
    review_status: str
    review_stale: bool
    review_gate: Dict[str, Any]
    executable_review: bool
    review_findings: List[Dict[str, Any]]
    enabled: bool
    load_error: Optional[str]
    grants: Dict[str, Any]
    payload_root: str
    provenance: Dict[str, Any]


class ClawHubPreviewResponse(TypedDict):
    """``GET /api/marketplace/clawhub/preview/{slug}``: registry metadata only; nothing is downloaded."""

    slug: str
    version: str
    summary: MarketplaceSkillSummary
    metadata_only: bool
    adapter: Dict[str, Any]  # {ok, metadata_only, warnings, blockers}
    staging: Dict[str, Any]  # {is_plugin}


class MarketplaceInstallRequest(TypedDict):
    """``POST /api/marketplace/clawhub/install``. Installing reviews the skill unless ``auto_review``
    is false; an existing skill is replaced only with ``overwrite``."""

    slug: str
    version: NotRequired[str]
    auto_review: NotRequired[bool]  # default true
    overwrite: NotRequired[bool]  # default false


class MarketplaceUpdateRequest(TypedDict, total=False):
    """``POST /api/marketplace/clawhub/update/{name}``."""

    version: str


class MarketplaceInstallResponse(TypedDict, total=False):
    """ClawHub install or update result. An install keeps ``ok`` true when review finds blockers (the
    skill is installed but cannot run); a dependency failure gives ``ok: false`` and rolls back."""

    ok: Required[bool]
    sanitized_name: str
    error: Optional[str]
    target_dir: str
    summary: Optional[MarketplaceSkillSummary]
    archive: Dict[str, Any]  # {sha256, size_bytes, version}
    adapter: Dict[str, Any]
    review_status: str
    review_findings: List[Dict[str, Any]]
    review_error: str
    deps_status: str
    deps_error: str
    deps_fingerprint: str
    provenance: Dict[str, Any]


class MarketplaceUninstallResponse(TypedDict):
    """Uninstall result; a repeat answers 400 (not installed)."""

    ok: bool
    sanitized_name: str
    error: Optional[str]


class MarketplaceSearchResponse(TypedDict, total=False):
    """``GET /api/marketplace/clawhub/search``: a text search (``q``) or, without ``q``, a registry
    browse paged by ``cursor``/``next_cursor``. ``limit`` echoes the request."""

    query: str
    sort: str
    limit: Any
    offset: int
    cursor: Optional[str]
    next_cursor: Optional[str]
    official: bool
    registry_path: str
    registry_attempts: List[Dict[str, Any]]
    registry_warnings: List[str]
    registry_empty: bool
    count: Required[int]
    results: Required[List[MarketplaceSkillSummary]]


class MarketplaceInstalledResponse(TypedDict):
    """``GET /api/marketplace/{clawhub|ouroboroshub}/installed`` (the hub route remains frozen)."""

    count: int
    skills: List[MarketplaceInstalledSkill]


class HubPublicationClearRequest(TypedDict):
    """Forget a publication receipt; ``expected_published`` is the receipt as shown (409 if it changed)."""

    expected_published: Dict[str, Any]


class HubPublicationClearResponse(TypedDict):
    ok: bool
    sanitized_name: str
    publication_cleared: bool  # false on a repeat: nothing was left to clear


# --- Agent accounts (the bundled Claudexor engine) ---------------------------------------------

class ClaudexorProblem(TypedDict):
    """An engine refusal relayed with its status: the engine's sentence, its code and the actions it
    asks for. A refusal without an HTTP answer from the engine is 503 ``{error}`` without ``code``."""

    error: str
    code: str
    required_actions: NotRequired[List[str]]


class ClaudexorMaintenanceProblem(TypedDict):
    """Engine or host preparation problem: ``{code, message, retryable?, requiredActions?, context?}``.
    POST may prepare Node/npm: 409 ``maintenance_restart_required`` asks for chat-header Restart;
    409/503 ``maintenance_preparation_failed`` reports failure. Both use
    ``context.acceptance=not_submitted, scope=this_attempt``. Optional inspection failure still
    permits POST. A 503 ``acceptance=unknown`` may have been accepted: retain its body and
    Idempotency-Key even after a later preparation-only reply, then resend both."""

    error: Dict[str, Any]


class ClaudexorAccountTarget(TypedDict):
    harness: str
    profile_id: str


class ClaudexorMaintenanceRequest(TypedDict):
    """``POST /api/claudexor/maintenance/operations`` (engine ControlHarnessMaintenanceCreateRequest),
    sent with a required ``Idempotency-Key`` header (1-256 printable ASCII characters): the same key
    and body join the same operation."""

    harness: str
    target: Dict[str, Any]  # {kind: latest} | {kind: version, version} | {kind: previous} | {kind: baseline}


class ClaudexorMaintenanceOperation(TypedDict, total=False):
    """Engine ControlHarnessMaintenanceOperation, passed through. ``state``: queued, running,
    succeeded, failed, cancelled or interrupted; poll while queued or running, and while
    ``termination`` is ``unconfirmed``."""

    id: Required[str]
    state: Required[str]
    harness: str
    phase: str
    mechanism: str
    target: Dict[str, Any]
    before: Any
    after: Any
    mutation: str  # none | applied | unknown
    termination: str  # not_applicable | confirmed | unconfirmed
    limitations: Any
    progress: Any
    problem: Any
    createdAt: str
    startedAt: Optional[str]
    finishedAt: Optional[str]


class ClaudexorMaintenanceInventory(TypedDict):
    """Engine ControlHarnessMaintenanceInventory, passed through: per harness what is installed,
    tested and available, and its current ``operation`` (rejoin it after a reload)."""

    observedAt: str
    harnesses: List[Dict[str, Any]]


class ClaudexorQuotaRefreshRequest(TypedDict, total=False):
    """``POST /api/claudexor/quota/refresh``: no body refreshes every account; ``target`` one."""

    target: ClaudexorAccountTarget


class ClaudexorAccountResources(TypedDict, total=False):
    """Engine ControlAccountResourcesResponse (or the older ControlQuotaResponse), passed through."""

    snapshots: Required[List[Dict[str, Any]]]
    absences: List[Dict[str, Any]]
    refreshed_at: str
    refresh_skipped: Any
    resources: List[Dict[str, Any]]


class ClaudexorAccountResetRequest(TypedDict):
    """``POST /api/claudexor/account-resets``, sent with a required ``Idempotency-Key`` header: the
    same key and body replay the same receipt. Keep both until the reset settles."""

    target: ClaudexorAccountTarget
    offer_id: str
    grant_id: NotRequired[str]


class ClaudexorAccountResetReceipt(TypedDict, total=False):
    """Engine ControlAccountResetResponse, passed through. Poll ``GET .../account-resets/{id}`` while
    ``state`` is running. ``outcome``: pending, reset, already_redeemed, already_used,
    nothing_to_reset, no_credit, not_eligible, cooldown, unavailable or unknown."""

    id: Required[str]
    request: Dict[str, Any]
    state: Required[str]  # running | completed
    created_at: str
    completed_at: Optional[str]
    outcome: str
    detail: str
    readback: Dict[str, Any]  # {state: pending|fresh|failed, attempted_at, detail}
    resources: Any


class ClaudexorLoginRequest(TypedDict):
    """``POST /api/claudexor/login``: start a subscription sign-in job (this can start the engine).
    Not idempotent: every call may create a new job, so never resend after a lost answer; the
    sign-in can be started again later."""

    harness: str
    profile_id: NotRequired[str]  # a named account; created if needed
    transport: NotRequired[Literal["client_pty"]]
    login_flow: NotRequired[str]  # codex only, for example device_auth


class ClaudexorLoginInputRequest(TypedDict):
    """``POST /api/claudexor/login/{job_id}/input``: one answer the job asked for."""

    value: str  # 1 to 1024 characters, sent as typed


class ClaudexorProfileEnabledResponse(TypedDict):
    """``PATCH /api/claudexor/credential-profiles/{harness}/{profile_id}``; re-read status for the effect."""

    ok: bool
    harness: str
    profile_id: str
    enabled: bool


class ReviewPoolResponse(TypedDict):
    """``GET /api/review-pool``: the review-eligible agent rows and what one review costs on each.
    A broken configuration answers a non-empty ``config_error`` with an empty pool."""

    limits: Dict[str, Any]
    catalog: Dict[str, Any]  # {present, enabled, rows, eligible}
    pool: List[Dict[str, Any]]  # {subagent_id, handle, route, effort, delivery, access, enabled, cost, ...}
    excluded: List[Dict[str, Any]]  # {subagent_id, reason}
    last_executions: Dict[str, Any]
    row_costs: Dict[str, Any]
    config_error: str
    migration: Optional[Dict[str, Any]]
    pool_without_credentials: List[Any]


# --- Local model, MCP and the model catalog ----------------------------------------------------

class ModelCatalogItem(TypedDict, total=False):
    """One selectable model. ``value`` is what a model setting stores: the raw id for OpenRouter,
    ``<provider_id>::<id>`` for another API provider, ``<kind>@<name>::<id>`` for a named API
    connection (its ``provider``/``provider_id`` = ``<kind>@<name>``, ``provider_label`` its
    picker label), ``claudexor::<source>=<model>`` for an agent account's model, which adds
    that account's facts."""

    provider_id: Required[str]
    provider: Required[str]
    provider_label: str
    source: str
    id: Required[str]
    name: str
    value: Required[str]
    label: str
    source_id: str
    credential_profile_id: str
    account_fingerprint: str
    context_window: Any
    max_context_window: Any
    max_output_tokens: Any
    is_default: bool
    supported_options: Any
    reasoning_efforts: List[str]
    default_reasoning_effort: Optional[str]
    input_modalities: List[str]
    provenance: Any
    observed_at: Optional[str]
    availability: Any
    problem: Any
    processing: Any


class ModelCatalogResponse(TypedDict, total=False):
    """``GET /api/model-catalog``: with ``source_id`` (and optionally ``credential_profile_id``) one
    agent account's models only; otherwise every configured provider's, each fetched with its own
    20-second bound. A provider that failed is an ``errors`` row and the read is still 200: absent
    models are unknown, not proof of none. ``model_sources`` and ``account_catalogs`` are the
    engine's model-source and per-account catalog objects, passed through; ``partial`` marks an
    incomplete account view."""

    items: Required[List[ModelCatalogItem]]
    errors: Required[List[Dict[str, Any]]]  # {provider_id, error, stage?, duration_ms?, source_id?, code?, ...}
    model_sources: Required[List[Dict[str, Any]]]
    account_catalogs: List[Dict[str, Any]]
    partial: bool


class OpenAICompatibleModelsRequest(TypedDict):
    """``POST /api/openai-compatible/models``: list ``{baseUrl}/models`` once; nothing is saved."""

    baseUrl: str
    apiKey: NotRequired[str]
    connection: NotRequired[str]  # named compatible row; omitted = the built-in pair


class LocalModelStartRequest(TypedDict):
    """``POST /api/local-model/start``: the request stays open while the model downloads; then poll
    ``GET /api/local-model/status`` (downloading, loading, ready or error). Not idempotent."""

    source: str  # a local path or a Hugging Face repo
    filename: NotRequired[str]  # required for a repo
    port: NotRequired[int]  # default 8766
    n_gpu_layers: NotRequired[int]  # default -1
    n_ctx: NotRequired[int]  # 0 means 16384
    chat_format: NotRequired[str]


class LocalModelStartResponse(TypedDict):
    status: Literal["starting"]
    model_path: str


class LocalModelRuntimeMissingResponse(TypedDict):
    """412: the local runtime is not installed; ``POST /api/local-model/install-runtime`` installs it."""

    error: Literal["runtime_missing"]
    message: str
    hint: str


class LocalModelStatusResponse(TypedDict, total=False):
    """``GET /api/local-model/status``. ``status``: offline, downloading, loading, ready or error
    (an internal failure answers 200 with only ``status`` and ``error``). ``runtime_status``:
    unknown, ok, missing, installing, install_ok or install_error. ``settings_application`` says
    whether the saved local-model settings are the ones running."""

    status: Required[str]
    error: Optional[str]
    model_path: Optional[str]
    model_name: Any
    context_length: Any
    port: Any
    download_progress: Any
    runtime_status: str
    runtime_install_log: str  # the last 500 characters
    settings_application: Dict[str, Any]  # {status, pending_keys, unknown_keys, summary}


class LocalModelTestResponse(TypedDict):
    """``POST /api/local-model/test``: two real completions; a failed probe is still 200."""

    success: bool
    chat_ok: bool
    tool_call_ok: bool
    details: Any
    tokens_per_sec: Any


class McpListingAttempt(TypedDict):
    """The latest listing attempt, independent of any retained successful listing."""

    at: str
    ok: bool
    error: str
    partial: bool


class McpServerStatus(TypedDict, total=False):
    """One configured MCP server. A configuration error row has only ``id``, ``enabled``,
    ``tool_count``, ``last_error`` and ``code`` (MCP_CONFIG_ERROR or MCP_ID_AMBIGUOUS)."""

    id: Required[str]
    name: str
    enabled: Required[bool]
    transport: str
    url: str  # credentials redacted
    auth_header: Any
    auth_configured: bool
    header_names: List[str]
    allowed_tools: List[str]
    cwd: str
    env_from_settings: Any
    configuration_warnings: List[str]
    schemas_at_start: bool  # the owner's pin: call schemas in every request from task start
    instructions: str  # the server's own usage text from initialize, untrusted
    map_tokens: int  # this server's part of the connections map (Main, Low/Max)
    schema_tokens: int  # all its call schemas, loaded or not
    tool_count: Required[int]
    tools: List[Dict[str, Any]]  # {name, prefixed_name, description}
    tool_name_collisions: List[Dict[str, Any]]
    last_error: Required[Optional[str]]
    last_refreshed: Any
    last_attempted: Any
    listed_at: Optional[str]
    latest_attempt: Optional[McpListingAttempt]
    complete: bool
    publication_error: str
    code: str


class McpStatusResponse(TypedDict):
    """``GET /api/mcp/status``: configuration and the last listed tools; makes no network call."""

    enabled: bool
    sdk_available: bool
    sdk_error: Any
    tool_timeout_sec: Any
    servers: List[McpServerStatus]
    weights: Dict[str, int]  # {tools, map_tokens, schema_tokens_from_task_start}: catalog potential


class McpRefreshRequest(TypedDict, total=False):
    """``POST /api/mcp/refresh``: one server, or every server without ``server_id``. Safe to repeat."""

    server_id: str


class McpRefreshResponse(TypedDict, total=False):
    """One server: ``{server_id, ok, tool_count, ...}``, failure ``{server_id, ok: false, error,
    code?, not_started?}``. Every server: ``{refreshed: {server_id: one-server answer}}``, plus
    ``error`` when MCP is disabled."""

    server_id: str
    ok: bool
    published: bool
    publication_error: str
    tool_count: int
    configuration_warnings: List[str]
    tool_name_collisions: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]
    error: str
    code: str
    not_started: bool
    refreshed: Dict[str, Any]


class McpTestRequest(TypedDict, total=False):
    """``POST /api/mcp/test``: connect once and list tools; saves nothing. ``server`` is a draft
    server entry; with ``server_id`` its masked secrets are filled from that saved server."""

    server_id: str
    server: Dict[str, Any]


class McpTestResponse(TypedDict, total=False):
    """A failed test is 200 with ``ok: false``."""

    ok: Required[bool]
    server_id: str
    tool_count: int
    configuration_warnings: List[str]
    pagination_truncated: bool
    tools: List[Dict[str, Any]]  # {name, description}
    code: str
    error: str


# --- Updates and core control ------------------------------------------------------------------

class UpdateProgress(TypedDict, total=False):
    """The last admitted apply attempt, kept across restarts; ``{}`` when there is none, a named ``unknown``
    when it cannot be read. An observation, never authority. ``active`` is true only while THIS server
    executes it. ``outcome`` stays empty until concluded: ``applied`` only after a later server's startup
    checks passed while serving the landed commit; ``not_applied``, ``rolled_back``, ``failed`` or
    ``unknown`` (``reason`` names the missing evidence). ``work_stop`` appears only on the executor's own
    conclusion; ``none`` means it reached no worker or service teardown. A version, SHA or reconnect
    alone proves nothing."""

    operation_id: str
    generation: str
    current_generation: bool
    stage: str
    started_at: str
    stage_started_at: str
    active: bool
    result: str
    error: str
    restart_required: bool
    outcome: Literal["", "applied", "not_applied", "rolled_back", "failed", "unknown"]
    outcome_at: str
    concluded_by: Literal["executor", "boot"]
    reason: str
    blockers: List[str]
    work_stop: Literal["", "none", "invoked"]
    base_sha: str
    target_sha: str
    tx_attempt_id: str
    keep_branch: str
    stash_note: str

    retention_unconfirmed: bool  # live only: saving this observation failed, not the update itself
    generation_facts: Optional[Dict[str, Any]]


class UpdateLetter(TypedDict):
    """The update description, written by Ouroboros as ordinary work and never by a check.
    ``relation``: pending, applied, superseded, other, or '' without text. ``state`` none: nothing
    saved yet (the projection still carries the offered range and the owner's request)."""

    state: str  # ready | failed (old records only) | none
    relation: str
    text: str
    author_version: str
    target_version: str
    written_at: str
    error_kind: str
    error_text: str
    key: Dict[str, Any]  # {base_sha, target_sha, update_channel, target_ref}
    has_last_good: bool
    description_current: bool
    failed_at: str
    latest_failed_key: Optional[Dict[str, Any]]
    qualification: NotRequired[Optional[Dict[str, Any]]]
    revision: int
    author: Dict[str, Any]  # {task_id, model, requested} — host-derived
    checked_key: Optional[Dict[str, Any]]
    material: Optional[Dict[str, Any]]  # {key, collected_at, text_bytes (query_data's count), commits, bodies_read, bodies_omitted}
    request: Optional[Dict[str, Any]]  # {task_id, current_task_id, mode, key, requested_at, status (retry/Continue), saved}


class UpdateTransactionState(TypedDict, total=False):
    """An update in progress: ``{active: false}``, or active with its phase and assisted task."""

    active: Required[bool]
    phase: str
    task_id: str
    restart_required: bool
    local_work_recovery: bool


class UpdateStatusResponse(TypedDict, total=False):
    """``GET /api/update/status`` (no network) and ``POST /api/update/check`` (fetches first, records range
    facts but never prose, and broadcasts ``update_progress_changed``). ``running_sha`` is the commit the
    running core started from; ``current_sha`` is the checkout's. ``check_ok`` is null on status.
    ``ahead``/``behind`` count commits (history); ``local_changes`` says whether the checkout holds
    content of its own, omitted when no comparison was attempted: ``present`` is a tree delta since
    the shared base, not authorship or upstream equivalence, and never replaces the apply merge plan."""

    running_sha: Required[str]
    current_version: str
    latest_version: str
    official_tags: List[Dict[str, Any]]  # check only: {tag, sha, source}
    update_tx: Required[UpdateTransactionState]
    update_progress: Required[UpdateProgress]
    letter: Optional[UpdateLetter]
    after_update: Optional[Dict[str, Any]]
    managed: bool
    remote: str
    remote_branch: str
    target_ref: str
    official_repo_url: str
    update_channel: str
    current_branch: str
    current_sha: str
    current_short_sha: str
    latest_sha: str
    latest_short_sha: str
    latest_message: str
    ahead: int
    behind: int
    local_changes: Literal["absent", "present", "unknown"]
    local_changes_base: str
    local_changes_reason: Literal["unresolved_sha", "shallow_history", "git_error", "several_bases", "no_shared_base"]
    dirty: bool
    dirty_count: int
    dirty_preview: List[str]
    warnings: List[str]
    check_ok: Optional[bool]
    available: bool
    safe_to_apply: bool
    checked_at: str
    checked_target_sha: str
    from_cache: bool


class CommandRequest(TypedDict):
    """``POST /api/command``: a slash command (``/restart``, ``/panic``, ``/review``, ``/evolve``,
    ``/bg``, ``/status``) or any other text, which becomes an owner message in Main. ``visible_text``
    adds a system row; the same ``visible_task_id`` within 5 seconds is answered as deduped."""

    cmd: str
    client_surface: NotRequired[Dict[str, Any]]
    visible_text: NotRequired[str]
    visible_task_id: NotRequired[str]
    task_constraint: NotRequired[Dict[str, Any]]


class CommandResponse(TypedDict, total=False):
    """``{status: "ok"}``: queued, except Panic acknowledges its latch. ``{ok, deduped, task_id}`` for a deduped repeat."""

    status: str
    ok: bool
    deduped: bool
    task_id: str


class GitRollbackRequest(TypedDict):
    target: str  # a commit or tag


class GitRollbackResponse(TypedDict):
    """Running work is interrupted and the core restarts; confirm ``running_sha`` afterwards."""

    status: str  # ok | restart_required
    message: str
    restarting: bool
    switch_deferred: bool


class GitPromoteResponse(TypedDict):
    """``POST /api/git/promote``: move the local stable branch to the current commit (no push)."""

    status: str
    sha: str
    message: str


class ResetResponse(TypedDict):
    """``POST /api/reset``: irreversible. The listed data folders and the settings were deleted and
    a restart was requested; a dropped connection is an unknown outcome, never a reason to resend."""

    status: str
    deleted: List[str]
    restarting: bool


class CostBreakdownResponse(TypedDict, total=False):
    """``GET /api/cost-breakdown``: spend from the physical-attempt ledger. Table rows (``by_model``,
    ``by_api_key``, ``by_model_category``, ``by_task_category``) are ``{cost, calls, unknown_unmetered,
    non_final_rows, cost_final, ...tokens}``; ``cost_final`` false means rows are still open. A 503
    carries ``accounting.available: false`` (ledger_unavailable): unknown spend, never zero."""

    total_cost: float
    total_calls: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cached_tokens: int
    total_cache_write_tokens: int
    prompt_cache_ttls: Dict[str, Any]
    by_model: Dict[str, Any]
    by_api_key: Dict[str, Any]
    by_model_category: Dict[str, Any]
    by_task_category: Dict[str, Any]
    accounting: Required[Dict[str, Any]]
    unattributed: Dict[str, Any]


__all__ = [
    "InstalledSkill", "ExtensionsIndexResponse", "ExtensionManifestResponse", "ExtensionSettingsSectionResponse",
    "SkillToggleRequest", "SkillRevisionRequest", "SkillGrantRequest", "SkillDeleteRequest",
    "SkillToggleResponse", "SkillActionRefusal", "SkillDaemonsResponse", "SkillLifecycleJob",
    "SkillLifecycleQueueResponse",
    "SkillReviewResponse", "SkillReviewHistoryDetailResponse", "SkillReconcileResponse",
    "MarketplaceSkillSummary", "MarketplaceInstalledSkill", "ClawHubPreviewResponse",
    "MarketplaceInstallRequest", "MarketplaceUpdateRequest", "MarketplaceInstallResponse",
    "MarketplaceUninstallResponse", "MarketplaceSearchResponse", "MarketplaceInstalledResponse",
    "HubPublicationClearRequest", "HubPublicationClearResponse",
    "ClaudexorProblem", "ClaudexorMaintenanceProblem", "ClaudexorAccountTarget",
    "ClaudexorMaintenanceRequest", "ClaudexorMaintenanceOperation", "ClaudexorMaintenanceInventory",
    "ClaudexorQuotaRefreshRequest", "ClaudexorAccountResources", "ClaudexorAccountResetRequest",
    "ClaudexorAccountResetReceipt", "ClaudexorLoginRequest", "ClaudexorLoginInputRequest",
    "ClaudexorProfileEnabledResponse", "ReviewPoolResponse",
    "ModelCatalogItem", "ModelCatalogResponse", "OpenAICompatibleModelsRequest", "LocalModelStartRequest",
    "LocalModelStartResponse", "LocalModelRuntimeMissingResponse", "LocalModelStatusResponse",
    "LocalModelTestResponse", "McpListingAttempt", "McpServerStatus", "McpStatusResponse", "McpRefreshRequest",
    "McpRefreshResponse", "McpTestRequest", "McpTestResponse",
    "UpdateProgress", "UpdateLetter", "UpdateTransactionState", "UpdateStatusResponse",
    "CommandRequest", "CommandResponse", "GitRollbackRequest", "GitRollbackResponse", "GitPromoteResponse",
    "ResetResponse", "CostBreakdownResponse",
]
