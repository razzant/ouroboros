"""Human/test-visible HTTP endpoint index for the browser gateway ABI.

Routers own the executable Route objects; this tuple is the reviewable
contract index consumed by the parity/reliability suites. It lives beside
``gateway/contracts.py`` (which re-exports it) so the mirror file keeps its
TypedDict envelopes within the module size gate.

``HTTP_SHAPES`` and ``WS_SHAPES`` bind contract TypedDicts (by name) to entries
of the two frozen index tuples; ``GET /api/contract`` derives the published
ABI from them (``gateway/schema.py::contract_document``). A token without a
shape is published as untyped: an explicit gap, never an empty schema.
"""

from __future__ import annotations

from typing import NamedTuple

HTTP_ENDPOINTS: tuple[str, ...] = (
    "GET /api/health",
    "GET /api/contract",
    "GET /api/state",
    "GET /api/consciousness",
    "GET /api/settings",
    "POST /api/settings",
    "POST /api/settings/secret",
    "GET /api/ui/preferences",
    "POST /api/ui/preferences",
    "POST /api/ui/holds",
    "GET /api/ui/i18n",
    "POST /api/ui/i18n/language",
    "POST /api/ui/i18n/missing",
    "POST /api/ui/i18n/import",
    "GET /api/ui/i18n/export",
    "POST /api/ui/i18n/regenerate",
    "GET /api/desktop/autostart",
    "POST /api/desktop/autostart",
    "GET /api/desktop/background",
    "POST /api/desktop/background",
    "POST /api/owner/runtime-mode",
    "POST /api/owner/auto-grant",
    "POST /api/owner/context-mode",
    "POST /api/owner/effort-range",
    "POST /api/owner/safety-mode",
    "POST /api/owner/safety-rules",
    "POST /api/owner/safety-model",
    "POST /api/owner/capability-ack",
    "POST /api/owner/skills/{skill}/attest-review",
    "GET /api/places",
    "GET /api/places/{place_id}/rooms",
    "POST /api/places",
    "POST /api/places/{place_id}",
    "DELETE /api/places/{place_id}",
    "POST /api/skills/{skill}/publish-preflight",
    "GET /api/model-catalog",
    "POST /api/tasks",
    "GET /api/tasks",
    "GET /api/tasks/{task_id}",
    "GET /api/tasks/{task_id}/artifacts/{name}",
    "GET /api/tasks/{task_id}/events",
    "POST /api/tasks/{task_id}/events",
    "POST /api/tasks/{task_id}/cancel",
    "POST /api/tasks/{task_id}/hurry",
    "POST /api/tasks/{task_id}/pause",
    "POST /api/tasks/{task_id}/continue",
    "POST /api/tasks/{task_id}/resume",
    "POST /api/decisions",
    "GET /api/cards/{card_id}",
    "GET /api/cards/{card_id}/app",
    "POST /api/cards/{card_id}/pin",
    "DELETE /api/cards/{card_id}/pin",
    "GET /api/schedules",
    "POST /api/schedules",
    "POST /api/schedules/{schedule_id}/action",
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
    "GET /api/storage",
    "POST /api/storage/measure",
    "POST /api/storage/measure/cancel",
    "POST /api/storage/free/preview",
    "POST /api/storage/free",
    "POST /api/storage/free/cancel",
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
    "POST /api/providers/test",
    "GET /api/local-model/status",
    "POST /api/local-model/start",
    "POST /api/local-model/stop",
    "POST /api/local-model/test",
    "POST /api/local-model/install-runtime",
    "GET /api/mcp/status",
    "POST /api/mcp/refresh",
    "POST /api/mcp/test",
    "POST /api/mcp/import/preview",
    "GET /api/review-pool",
    "GET /api/claudexor/status",
    "GET /api/claudexor/maintenance/harnesses",
    "POST /api/claudexor/maintenance/operations",
    "GET /api/claudexor/maintenance/operations/{operation_id}",
    "POST /api/claudexor/maintenance/operations/{operation_id}/cancel",
    "POST /api/claudexor/quota/refresh",
    "POST /api/claudexor/account-resets",
    "GET /api/claudexor/account-resets/{operation_id}",
    "POST /api/claudexor/wake",
    "POST /api/claudexor/login",
    "GET /api/claudexor/login/{job_id}",
    "DELETE /api/claudexor/login/{job_id}",
    "POST /api/claudexor/login/{job_id}/input",
    "POST /api/claudexor/login/{job_id}/reconcile",
    "DELETE /api/claudexor/credential-profiles/{harness}/{profile_id}",
    "PATCH /api/claudexor/credential-profiles/{harness}/{profile_id}",
    "GET /api/widgets",
    "POST /api/widgets/fault",
    "GET /api/extensions",
    "GET /api/extensions/{skill}/manifest",
    "GET /api/extensions/{skill}/module/{entry:path}",
    "GET /api/extensions/{skill}/ui/{revision}/{path:path}",
    "GET /api/ui/contributions",
    "GET /api/extensions/{skill}/settings_section",
    "ANY /api/extensions/{skill}/{rest:path}",
    "GET /api/skills/daemons",
    "POST /api/skills/{skill}/toggle",
    "POST /api/skills/{skill}/delete",
    "GET /api/skills/lifecycle-queue",
    "POST /api/skills/{skill}/review",
    "GET /api/skills/{skill}/review-history/{job_id}",
    "POST /api/skills/{skill}/grants",
    "POST /api/skills/{skill}/reconcile",
    "GET /api/marketplace/clawhub/search",
    "GET /api/marketplace/clawhub/installed",
    "GET /api/marketplace/ouroboroshub/installed",
    "GET /api/marketplace/clawhub/info/{slug:path}",
    "GET /api/marketplace/clawhub/preview/{slug:path}",
    "POST /api/marketplace/clawhub/install",
    "POST /api/marketplace/clawhub/update/{name}",
    "POST /api/marketplace/clawhub/uninstall/{name}",
    "POST /api/skills/{skill}/publication/clear",
    "GET /api/skill-sources",
    "POST /api/skill-sources",
    "POST /api/skill-sources/probe",
    "POST /api/skill-sources/my-skills",
    "POST /api/skill-sources/{source_id}",
    "POST /api/skill-sources/{source_id}/remove",
    "POST /api/skill-sources/{source_id}/refresh",
    "GET /api/skill-catalog",
    "GET /api/skill-catalog/search",
    "GET /api/skill-catalog/item/{source_id}/{name}",
    "POST /api/skill-catalog/install",
    "POST /api/skills/{skill}/update",
    "POST /api/skills/{skill}/uninstall",
    "GET /api/onboarding/starter-skills",
    # The wizard PAGE (one onboarding host: desktop setup window, blocking
    # overlay frame, plain browser). /api/onboarding stays the readiness probe.
    "GET /onboarding",
    "GET /api/onboarding",
    "GET /api/onboarding/bootstrap",
    "POST /api/onboarding/subagents/preview",
    "POST /api/onboarding/complete",
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


class Shape(NamedTuple):
    """Published shapes of one ``HTTP_ENDPOINTS`` token.

    ``responses`` reads ``"200:TaskCreateResponse 400,409,503:ErrorResponse"``: statuses, then
    a body that is a contract TypedDict name (JSON), a media kind (``json`` = JSON whose shape is
    not published, ``html``, ``text``, ``sse``, ``file``, ``zip``, ``js``, ``empty``, ``any``), a kind with
    the TypedDict its items follow (``sse/TaskEvent``), or alternatives joined by ``|``.
    ``request`` uses the same body words (plus ``multipart``); ``query`` names the parameters
    the handler reads and ``headers`` the request headers it reads (a ``!`` suffix: required).
    Gateway-wide rules cover what a row does not list: 401 from the network auth gate, and 500
    as ``ErrorResponse`` or plain text for an unexpected failure.
    """

    responses: str
    request: str = ""
    query: str = ""
    note: str = ""
    headers: str = ""


_TASK_ERRORS = "400,404,409,503:ErrorResponse"
_FILE_ERRORS = "400,403,404,409,500:ErrorResponse"
_SETTINGS_REFUSAL = "400,409,503:SettingsWriteRefusal"
_SKILL_REFUSAL = "400,404,409:SkillActionRefusal"
_LOGIN_JOB = "200:ClaudexorLoginJobResponse 400:ErrorResponse 404,410:ClaudexorLoginJobProblem 503:ErrorResponse"
_MAINTENANCE = "400,409,503:ClaudexorMaintenanceProblem"
_CLAWHUB_ERRORS = "400,404,429,502:ErrorResponse"
_INSTALL = "200:MarketplaceInstallResponse 400:MarketplaceInstallResponse|ErrorResponse 404,429:MarketplaceInstallResponse"
_IN_LANE = "runs in the one skill lifecycle lane until it settles (watch GET /api/skills/lifecycle-queue)"
_SOURCES_WRITE = "503:SettingsWriteRefusal"
_INSTALL_REFUSAL = "404,409,429,502:SkillCatalogInstallResponse"
_ENGINE = "relays the bundled Claudexor engine"

HTTP_SHAPES: dict[str, Shape] = {
    "GET /api/health": Shape("200:HealthResponse"),
    "GET /api/contract": Shape("200:GatewayContractResponse"),
    "GET /api/state": Shape("200:StateResponse 500:ErrorResponse"),
    "GET /api/consciousness": Shape("200:json 503:json", note="consciousness panel: status, spend by cause, places, recent wakes"),
    "GET /onboarding": Shape("200:html", query="builtin", note="selected setup or builtin wizard; builtin=1 forces builtin"),
    "GET /api/onboarding": Shape("200:html 204:empty", note="204: a ready provider exists, no setup needed"),
    "GET /api/onboarding/bootstrap": Shape("200:OnboardingBootstrapResponse", note="read-only; writes nothing"),
    "POST /api/onboarding/subagents/preview": Shape(
        "200:OnboardingSubagentsPreviewResponse 400,503:SettingsWriteRefusal", "OnboardingCompleteRequest",
        note="read-only actor draft for the same payload completion would save"),
    "POST /api/onboarding/complete": Shape(
        "200:OnboardingCompleteResponse 400,403,409,500,503:SettingsWriteRefusal|OnboardingPresetFailureResponse",
        "OnboardingCompleteRequest", note="the one atomic setup writer; a 503 with can_skip offers completion without agent presets"),
    "GET /api/settings": Shape(
        "200:SettingsReadResponse", query="websearch_preview response_limit_preview model account local backend",
        note="masked secrets read back as placeholders; sending a placeholder keeps the stored value"),
    "POST /api/settings": Shape(
        f"200:SettingsSaveResponse {_SETTINGS_REFUSAL} 500:SettingsPostCommitFailureResponse", "json",
        note="partial document of setting keys; restart_state names keys that apply only after a restart"),
    "POST /api/settings/secret": Shape("200:SettingsSecretResponse 400,404,409:ErrorResponse", "SettingsSecretRequest",
                                       note="reveals one stored secret, no-store; writes nothing"),
    "GET /api/ui/preferences": Shape("200:UiPreferencesResponse"),
    "GET /api/ui/contributions": Shape("200:UiContributionsResponse", note="live contributions, offers and owner preferences; no-store"),
    "POST /api/ui/preferences": Shape("200:UiPreferencesResponse 400,503:ErrorResponse", "json"),
    "POST /api/ui/holds": Shape("200:UiHoldsResponse 400,503:ErrorResponse", "UiHoldsRequest",
                                note="names new holds in one System row per owning chat; a repeat names nothing"),
    "GET /api/ui/i18n": Shape("200:UiI18nResponse"),
    "POST /api/ui/i18n/language": Shape(
        "200:UiI18nResponse 400:ErrorResponse|SettingsWriteRefusal 409,502,503:SettingsWriteRefusal",
        "UiI18nLanguageRequest", note="also broadcasts ui_language_changed"),
    "POST /api/ui/i18n/missing": Shape("200:UiI18nMissingResponse 400,409:ErrorResponse", "UiI18nMissingRequest"),
    "POST /api/ui/i18n/import": Shape("200:UiI18nImportResponse 400,503:ErrorResponse", "UiI18nMemoryDocument",
                                      note="replaces the imported layer; broadcasts ui_i18n_updated for the current language"),
    "GET /api/ui/i18n/export": Shape("200:UiI18nMemoryDocument 404,409:ErrorResponse", query="language",
                                     note="the memory file as an attachment; English has none (404)"),
    "POST /api/ui/i18n/regenerate": Shape(
        "200:UiI18nRegenerateResponse 400,409,503:ErrorResponse", "UiI18nRegenerateRequest",
        note="broadcasts ui_i18n_updated; generation continues in the background (GET /api/ui/i18n)"),
    "GET /api/desktop/autostart": Shape("200:DesktopAutostartResponse 500:ErrorResponse"),
    "POST /api/desktop/autostart": Shape("200:DesktopAutostartResponse 400,409,500:ErrorResponse", "OwnerEnabledRequest",
                                         note="the OS sign-in entry; 409 when unavailable on this host"),
    "GET /api/desktop/background": Shape("200:DesktopAutostartResponse"),
    "POST /api/desktop/background": Shape(f"200:DesktopAutostartResponse {_SETTINGS_REFUSAL}", "OwnerEnabledRequest"),
    "POST /api/owner/runtime-mode": Shape(f"200:OwnerRuntimeModeResponse {_SETTINGS_REFUSAL}", "OwnerRuntimeModeRequest"),
    "POST /api/owner/auto-grant": Shape(f"200:OwnerAutoGrantResponse {_SETTINGS_REFUSAL}", "OwnerEnabledRequest"),
    "POST /api/owner/context-mode": Shape(f"200:OwnerContextModeResponse {_SETTINGS_REFUSAL} 404:SettingsWriteRefusal", "OwnerContextModeRequest"),
    "POST /api/owner/effort-range": Shape(f"200:OwnerEffortRangeResponse {_SETTINGS_REFUSAL} 404:SettingsWriteRefusal", "OwnerEffortRangeRequest",
                                          note="helper min <= max; primary independent; optional project_id, otherwise Main"),
    "POST /api/owner/safety-rules": Shape(f"200:OwnerSafetyRulesResponse {_SETTINGS_REFUSAL}", "OwnerSafetyRulesRequest"),
    "POST /api/owner/safety-model": Shape(f"200:OwnerSafetyModelResponse {_SETTINGS_REFUSAL}", "OwnerSafetyModelRequest"),
    "POST /api/owner/safety-mode": Shape(f"200:OwnerSafetyModeResponse {_SETTINGS_REFUSAL}", "OwnerSafetyModeRequest"),
    "POST /api/owner/capability-ack": Shape(
        "200:ResponseLimitAckResponse|ContextWindowAckResponse 400:ErrorResponse", "CapabilityAckRequest",
        note="writes acknowledgement evidence for one exact route, never settings; a repeat overwrites it"),
    "POST /api/owner/skills/{skill}/attest-review": Shape(
        "200,409:SkillReviewResponse 400:ErrorResponse", "SkillRevisionRequest",
        note=f"the owner's review verdict; 409 when refused (ok false); {_IN_LANE}"),
    "GET /api/places": Shape("200:json", note="Places and installed connector declarations; §12 Places"),
    "GET /api/places/{place_id}/rooms": Shape("200:PlaceRoomsResponse 404,503:ErrorResponse", note="lazy indexed captured-room discovery; history uses /api/chat/history"),
    "POST /api/places": Shape("200:json 400:json", "PlaceConnectRequest", note="connect from a declared connector"),
    "POST /api/places/{place_id}": Shape("200:json 400,409:json", "PlaceUpdateRequest", note="atomic owner-authored place choices"),
    "DELETE /api/places/{place_id}": Shape("200:json 400:json", note="disconnect; room history stays"),
    "POST /api/skills/{skill}/publish-preflight": Shape(
        "200,400,404,409,503:SkillPublishPreflightResponse", "SkillPublishPreflightRequest",
        note="read-only; every status carries the same shape; state hard_block is a refusal; destination "
             "{source_id, mode} picks where it would go (default: the skill's origin, else the hub)"),
    "GET /api/model-catalog": Shape("200:ModelCatalogResponse 400:ErrorResponse", query="source_id credential_profile_id",
                                    note="read-only; a provider that failed is an errors row, not a failed read"),
    "POST /api/providers/test": Shape("200:ProviderTestResponse 400:ErrorResponse", "ProviderTestRequest",
                                      note="a failed probe is 200 with ok false; nothing is saved"),
    "POST /api/tasks": Shape(
        f"200:TaskCreateResponse {_TASK_ERRORS} 422:TaskCreateResponse", "TaskCreateRequest",
        note="task_id is the operation identity; 503 status unconfirmed is an unknown outcome: never resend under a new id"),
    "GET /api/tasks": Shape("200:TaskListResponse 503:ErrorResponse", query="status limit queue_only"),
    "GET /api/tasks/{task_id}": Shape("200:TaskDetailResponse 400,404,503:ErrorResponse"),
    "GET /api/tasks/{task_id}/artifacts/{name}": Shape(
        "200,206:file 400,404,409,500,503:ErrorResponse", query="relpath archive source", headers="Range",
        note="Range supported; x-ouroboros-artifact-sha256 when the bytes are verified"),
    "GET /api/tasks/{task_id}/events": Shape(
        "200:sse/TaskEvent 400,404:sse", query="cursor wait",
        note="resume with ?cursor=<last event id>; a 400/404 stream carries one error event"),
    "POST /api/tasks/{task_id}/events": Shape("200:sse/TaskEvent 400,404:ErrorResponse", "TaskEventsRequest",
                                             note="v2 cursor: send back the cursor of the last event processed"),
    "POST /api/tasks/{task_id}/cancel": Shape(
        f"200,202:TaskCancelResponse {_TASK_ERRORS}", "TaskCancelRequest",
        note="stop_action_id identifies one Stop; an exact retry joins a pending Stop; after teardown it may return 404; 202 = the Stop is recorded and still pending (graceful, or an immediate teardown not yet confirmed)"),
    "POST /api/tasks/{task_id}/hurry": Shape(f"200:TaskHurryResponse {_TASK_ERRORS}", "TaskHurryRequest"),
    "POST /api/tasks/{task_id}/pause": Shape(f"200,202:TaskPauseResponse {_TASK_ERRORS}", "TaskPauseRequest"),
    "POST /api/tasks/{task_id}/continue": Shape(f"200:TaskContinueResponse {_TASK_ERRORS}", "TaskContinueRequest",
                                                note="a timeout is not a refusal: retry with the same action_nonce"),
    "POST /api/tasks/{task_id}/resume": Shape("200:TaskResumeResponse 400,404,409,503:ErrorResponse",
                                              note="a budget-paused task; not idempotent: a repeat is refused"),
    "POST /api/decisions": Shape(
        "200,202:DecisionResponse 400,404:ErrorResponse 409:DecisionResponse|ErrorResponse 501,503:ErrorResponse",
        "DecisionRequest", note="request_id is the idempotency key; the first answer wins (a card press: unless once)"),
    "GET /api/cards/{card_id}": Shape("200:CardResponse 404:ErrorResponse", note="read-only; the current body and presses"),
    "GET /api/cards/{card_id}/app": Shape(
        "200:text 404:ErrorResponse", note="the app's HTML as text/plain for a sandboxed srcdoc; never served as a page"),
    "POST /api/cards/{card_id}/pin": Shape(
        "200:CardPinResponse 404,409,503:ErrorResponse", note="component cards only; broadcasts extension_lifecycle"),
    "DELETE /api/cards/{card_id}/pin": Shape("200:CardPinResponse 503:ErrorResponse", note="a repeat answers pinned false"),
    "GET /api/chat/history": Shape(
        "200,400,409,503:ChatHistoryResponse", query="chat_id cursor around n_human n_progress limit",
        note="an error page echoes the cursor sent, so the same request can be repeated"),
    "POST /api/chat/upload": Shape("200:UploadResponse 400:ErrorResponse", "multipart", note="form field file"),
    "DELETE /api/chat/upload": Shape("200:ChatUploadDeleteResponse 400,404,409:ErrorResponse", "ChatUploadDeleteRequest",
                                     note="pending uploads only; a repeat answers 404"),
    "GET /api/projects": Shape("200:ProjectListResponse 500:ErrorResponse"),
    "POST /api/projects": Shape("200:ProjectCreateResponse 400,401,409,500:ErrorResponse", "ProjectCreateRequest",
                                note="401 error_code auth_required: the git clone needs credentials (not the network gate)"),
    "POST /api/projects/from-task": Shape("200:ProjectFromTaskResponse 400,409,500:ErrorResponse", "ProjectFromTaskRequest",
                                          note="409: the task already belongs to another project"),
    "POST /api/projects/{project_id}/update": Shape("200:ProjectUpdateResponse 400,404,409,500:ErrorResponse",
                                                    "ProjectUpdateRequest"),
    "POST /api/projects/{project_id}/delete": Shape("200:ProjectDeleteResponse 404,500:ErrorResponse"),
    "GET /api/fs/dirs": Shape("200:FsDirsResponse 400,403,404,500:ErrorResponse", query="path"),
    "GET /api/files/list": Shape(f"200:FileBrowserListResponse {_FILE_ERRORS}", query="path"),
    "GET /api/files/read": Shape(f"200:FileBrowserReadResponse {_FILE_ERRORS}", query="path"),
    "GET /api/files/content": Shape(f"200:file {_FILE_ERRORS}", query="path"),
    "GET /api/files/download": Shape(f"200,206:file {_FILE_ERRORS} 503:ErrorResponse", query="path upload",
                                     headers="Range", note="?upload=<stored id> reads a chat attachment, never a path"),
    "POST /api/files/upload": Shape(f"200:FileChangeResponse {_FILE_ERRORS} 413:ErrorResponse", "multipart",
                                    note="form fields file and path (target directory); an existing name answers 409"),
    "POST /api/files/mkdir": Shape(f"200:FileChangeResponse {_FILE_ERRORS}", "FileMkdirRequest"),
    "POST /api/files/write": Shape(f"200:FileChangeResponse {_FILE_ERRORS}", "FileWriteRequest"),
    "POST /api/files/delete": Shape(f"200:FileChangeResponse {_FILE_ERRORS}", "FilePathRequest"),
    "POST /api/files/transfer": Shape(f"200:FileChangeResponse {_FILE_ERRORS}", "FileTransferRequest"),
    "GET /api/logs/{name}": Shape("200:LogTailResponse 404:ErrorResponse", query="limit task_id"),
    "GET /api/extensions": Shape("200:ExtensionsIndexResponse 500:ErrorResponse"),
    "GET /api/widgets": Shape("200:WidgetsResponse"),
    "POST /api/widgets/fault": Shape("200:WidgetFaultResponse 400,413,503:ErrorResponse", "WidgetFaultRequest",
                                     note="one task-less widget_fault row in logs/events.jsonl, bounded per widget per minute"),
    "GET /api/extensions/{skill}/manifest": Shape("200:ExtensionManifestResponse 400,404:ErrorResponse"),
    "GET /api/extensions/{skill}/module/{entry:path}": Shape("200:js 400,404,409:ErrorResponse",
                                                            note="reviewed module text; host protocol: docs/WIDGET_HOST_PROTOCOL.md"),
    "GET /api/extensions/{skill}/settings_section": Shape("200:ExtensionSettingsSectionResponse 400:ErrorResponse"),
    "GET /api/extensions/{skill}/ui/{revision}/{path:path}": Shape("200:any 404:empty", note="captured live ui assets; stale revisions are unavailable"),
    "ANY /api/extensions/{skill}/{rest:path}": Shape(
        "200:any 404,405,409,413,502:ErrorResponse", "any", note="the skill's own route, inside its own prefix"),
    "POST /api/skills/{skill}/toggle": Shape(f"200:SkillToggleResponse {_SKILL_REFUSAL}", "SkillToggleRequest",
                                             note=f"broadcasts extension_lifecycle when an extension changed; {_IN_LANE}"),
    "POST /api/skills/{skill}/delete": Shape("200:SkillDeleteResponse 400,403,404,409,500:SkillActionRefusal",
                                             "SkillDeleteRequest", note=_IN_LANE),
    "GET /api/skills/daemons": Shape("200:SkillDaemonsResponse 500:ErrorResponse"),
    "GET /api/skills/lifecycle-queue": Shape("200:SkillLifecycleQueueResponse"),
    "POST /api/skills/{skill}/review": Shape(
        "200:SkillReviewResponse", note=f"the review runs inside the request (minutes); a duplicate is status pending; {_IN_LANE}"),
    "GET /api/skills/{skill}/review-history/{job_id}": Shape("200:SkillReviewHistoryDetailResponse 400,404,503:ErrorResponse"),
    "POST /api/skills/{skill}/grants": Shape(f"200:SkillGrantResponse {_SKILL_REFUSAL}", "SkillGrantRequest", note=_IN_LANE),
    "POST /api/skills/{skill}/reconcile": Shape("200:SkillReconcileResponse 400:ErrorResponse"),
    "GET /api/marketplace/clawhub/search": Shape(
        f"200:MarketplaceSearchResponse {_CLAWHUB_ERRORS}",
        query="q query sort limit include_plugins official only_official cursor"),
    "GET /api/marketplace/clawhub/installed": Shape("200:MarketplaceInstalledResponse"),
    "GET /api/marketplace/ouroboroshub/installed": Shape("200:MarketplaceInstalledResponse"),
    "GET /api/marketplace/clawhub/info/{slug:path}": Shape(f"200:MarketplaceSkillSummary {_CLAWHUB_ERRORS}"),
    "GET /api/marketplace/clawhub/preview/{slug:path}": Shape(f"200:ClawHubPreviewResponse {_CLAWHUB_ERRORS}",
                                                             query="version"),
    "POST /api/marketplace/clawhub/install": Shape(_INSTALL, "MarketplaceInstallRequest", note=_IN_LANE),
    "POST /api/marketplace/clawhub/update/{name}": Shape(_INSTALL, "MarketplaceUpdateRequest", note=_IN_LANE),
    "POST /api/marketplace/clawhub/uninstall/{name}": Shape("200,400:MarketplaceUninstallResponse", note=_IN_LANE),
    "POST /api/skills/{skill}/publication/clear": Shape(
        "200:HubPublicationClearResponse 400,409:ErrorResponse", "HubPublicationClearRequest",
        note="forgets the local receipt only; the pull request and the payload stay"),
    "GET /api/skill-sources": Shape("200:SkillSourcesResponse 500:ErrorResponse"),
    "POST /api/skill-sources": Shape(
        f"200:SkillSourceChangeResponse 400:ErrorResponse|SkillSourceChangeResponse {_SOURCES_WRITE}",
        "SkillSourceAddRequest", note="reads the address once before saving; a saved repository starts a background fetch"),
    "POST /api/skill-sources/probe": Shape("200:SkillSourceProbeResponse 400:ErrorResponse", "SkillSourceProbeRequest",
                                           note="read-only; the address is fetched into a scratch folder that is removed"),
    "POST /api/skill-sources/my-skills": Shape("200,400,403,502:MySkillsResponse", "MySkillsRequest"),
    "POST /api/skill-sources/{source_id}": Shape(
        f"200:SkillSourceChangeResponse 400:ErrorResponse|SkillSourceChangeResponse {_SOURCES_WRITE}",
        "SkillSourceChangeRequest", note="a trust change never rewrites an existing install"),
    "POST /api/skill-sources/{source_id}/remove": Shape(
        f"200,400:SkillSourceChangeResponse {_SOURCES_WRITE}", note="installed skills keep working"),
    "POST /api/skill-sources/{source_id}/refresh": Shape("200:SkillSourceChangeResponse 404:ErrorResponse",
                                                         note="starts a background fetch and answers at once"),
    "GET /api/skill-catalog": Shape("200:SkillCatalogResponse 500:ErrorResponse", query="refresh",
                                    note="local cache only; a stale cache or refresh=1 starts a background fetch"),
    "GET /api/skill-catalog/search": Shape("200:SkillCatalogSearchResponse 500:ErrorResponse", query="q source",
                                           note="registry (ClawHub) hits need the network; a failure is registry_error"),
    "GET /api/skill-catalog/item/{source_id}/{name}": Shape(
        "200:SkillCatalogItemDetailResponse 404,500:ErrorResponse",
        note="a pointer entry's files are fetched at its exact commit"),
    "POST /api/skill-catalog/install": Shape(
        f"200:SkillCatalogInstallResponse 400:ErrorResponse|SkillCatalogInstallResponse {_INSTALL_REFUSAL}",
        "SkillCatalogInstallRequest", note=f"answers the started job at once; {_IN_LANE}"),
    "POST /api/skills/{skill}/update": Shape(
        "200:SkillUpdateResponse 400:ErrorResponse|SkillUpdateResponse 404,409,429,502:SkillUpdateResponse",
        "SkillUpdateRequest",
        note=f"replace answers the started job; keep and merge answer when done; {_IN_LANE}"),
    "POST /api/skills/{skill}/uninstall": Shape("200:SkillUninstallResponse 404,409:SkillUninstallResponse",
                                                note=_IN_LANE),
    "GET /api/onboarding/starter-skills": Shape("200:StarterSkillsResponse 500:ErrorResponse",
                                                note="local data; may start a background fetch of the hub"),
    "GET /api/schedules": Shape("200:ScheduledTasksResponse 503:ErrorResponse"),
    "POST /api/schedules": Shape("200:ScheduleUpsertResponse 400,409:ErrorResponse", "ScheduleUpsertRequest"),
    "POST /api/schedules/{schedule_id}/action": Shape("200:ScheduleActionResponse 400:ErrorResponse",
                                                      "ScheduleActionRequest"),
    "DELETE /api/schedules/{schedule_id}": Shape("200:ScheduleActionResponse 400:ErrorResponse"),
    "POST /api/command": Shape(
        "200:CommandResponse 400,409,503:ErrorResponse", "CommandRequest",
        note="queued, not done, except /panic: accepted immediately, never queued; /panic and /restart may end the connection before or after the answer"),
    "POST /api/reset": Shape("200:ResetResponse 409:ErrorResponse",
                             note="irreversible: deletes data and settings, then restarts; never resend"),
    "GET /api/git/log": Shape("200:GitLogResponse 500:ErrorResponse"),
    "POST /api/git/rollback": Shape("200:GitRollbackResponse 400:ErrorResponse 409,500:UpdateApplyErrorResponse",
                                    "GitRollbackRequest", note="interrupts running work and restarts the core; no check, not safe to repeat"),
    "POST /api/git/promote": Shape("200:GitPromoteResponse 400,409:ErrorResponse"),
    "GET /api/update/status": Shape("200:UpdateStatusResponse 500:ErrorResponse",
                                    note="no network; running_sha is the executing core"),
    "POST /api/update/check": Shape("200:UpdateStatusResponse 500:ErrorResponse",
                                    note="reads no body; fetches, may write the update letter; broadcasts update_progress_changed"),
    "POST /api/update/preflight": Shape("200:UpdatePreflightResponse 500:ErrorResponse", "UpdatePreflightRequest"),
    "POST /api/update/apply": Shape(
        "200:UpdateApplySuccessResponse 400:ErrorResponse 409,500:UpdateApplyErrorResponse", "UpdateApplyRequest",
        note="200: inspect status (manual, assisted_started, restart_required, or ok with restarting); adoption requires the landed transaction/merge identity and running_sha after startup"),
    "GET /api/cost-breakdown": Shape("200:CostBreakdownResponse 503:ErrorResponse",
                                     note="503: the ledger is unavailable (accounting.available false); spend is unknown, not zero"),
    "GET /api/storage": Shape("200:StorageResponse", query="count",
                              note="passive; ?count=<finished_at> omits a count the client already holds"),
    "POST /api/storage/measure": Shape("202:StorageResponse", note="starts one count in a child, or joins the running one"),
    "POST /api/storage/measure/cancel": Shape("202:StorageResponse", note="kills the running count; the saved count stays"),
    "POST /api/storage/free/preview": Shape("202:StorageResponse",
                                            note="a pure preview of what the retirers would remove now; a second preview joins"),
    "POST /api/storage/free": Shape("202:StorageResponse", "StorageFreeRequest",
                                    note="frees exactly the previewed items of the named groups; started/reason say whether it ran"),
    "POST /api/storage/free/cancel": Shape("202:StorageResponse", note="stops the listing or the run; what was freed stays freed"),
    "GET /api/evolution-data": Shape("200:EvolutionDataResponse", query="force",
                                     note="answers from the saved copy at once; new tags or force start one background re-measurement"),
    "POST /api/openai-compatible/models": Shape("200:OpenAICompatibleModelsResponse 400,502:ErrorResponse",
                                                "OpenAICompatibleModelsRequest", note="lists {baseUrl}/models once; saves nothing"),
    "GET /api/local-model/status": Shape("200:LocalModelStatusResponse"),
    "POST /api/local-model/start": Shape(
        "200:LocalModelStartResponse 400,409:ErrorResponse 412:LocalModelRuntimeMissingResponse", "LocalModelStartRequest",
        note="not idempotent; the request stays open while the model downloads; then poll status"),
    "POST /api/local-model/stop": Shape("200:StatusResponse", note="status stopped; also aborts a runtime install"),
    "POST /api/local-model/test": Shape("200:LocalModelTestResponse 400:ErrorResponse",
                                        note="two real completions; a failed probe is 200"),
    "POST /api/local-model/install-runtime": Shape("200:StatusResponse",
                                                   note="installing or already_installing; poll status runtime_status"),
    "GET /api/mcp/status": Shape("200:McpStatusResponse"),
    "POST /api/mcp/refresh": Shape("200:McpRefreshResponse", "McpRefreshRequest", note="safe to repeat"),
    "POST /api/mcp/test": Shape("200:McpTestResponse 400,404,409:ErrorResponse", "McpTestRequest",
                                note="connects once and saves nothing; a failed test is 200 with ok false"),
    "POST /api/mcp/import/preview": Shape("200,400:MCPImportPreviewResponse", "MCPImportPreviewRequest",
                                         note="translates pasted client JSON into draft rows; saves nothing"),
    "GET /api/review-pool": Shape("200:ReviewPoolResponse"),
    "GET /api/claudexor/status": Shape(
        "200:ClaudexorStatusResponse|ClaudexorQuotaResponse", query="view include",
        note=f"passive read, never starts the engine; view=quota answers ClaudexorQuotaResponse; {_ENGINE}"),
    "GET /api/claudexor/maintenance/harnesses": Shape(
        "200:ClaudexorMaintenanceInventory 400,503:ClaudexorMaintenanceProblem", query="harness fresh checkLatest",
        note=f"passive; an engine refusal keeps the engine's status; {_ENGINE}"),
    "POST /api/claudexor/maintenance/operations": Shape(
        f"202:ClaudexorMaintenanceOperation {_MAINTENANCE}", "ClaudexorMaintenanceRequest", headers="Idempotency-Key!",
        note="can start the engine and prepare Node/npm; 409 maintenance_restart_required or 409/503 "
             "maintenance_preparation_failed: acceptance=not_submitted, scope=this_attempt; "
             "use chat-header Restart when required; external/attach-only Node/npm needs external repair; "
             "optional inspection failure still submits; the same key and body join one operation; "
             "keep both after 503 acceptance unknown, even after a later preparation-only reply"),
    "GET /api/claudexor/maintenance/operations/{operation_id}": Shape(
        f"200:ClaudexorMaintenanceOperation {_MAINTENANCE}", note="passive; poll while queued or running"),
    "POST /api/claudexor/maintenance/operations/{operation_id}/cancel": Shape(
        f"200:ClaudexorMaintenanceOperation {_MAINTENANCE}",
        note="passive; an acknowledgement is not termination: poll until termination is confirmed"),
    "POST /api/claudexor/quota/refresh": Shape(
        "200:ClaudexorAccountResources 400:ErrorResponse 503:ClaudexorProblem|ErrorResponse", "ClaudexorQuotaRefreshRequest",
        note=f"passive; an engine refusal keeps its status with code; Retry-After passes through; {_ENGINE}"),
    "POST /api/claudexor/account-resets": Shape(
        "200:ClaudexorAccountResetReceipt 400:ErrorResponse 409:ClaudexorProblem 503:ClaudexorProblem|ErrorResponse",
        "ClaudexorAccountResetRequest", headers="Idempotency-Key!",
        note="the same key and body replay one receipt; after a lost answer resend both or read the receipt"),
    "GET /api/claudexor/account-resets/{operation_id}": Shape(
        "200:ClaudexorAccountResetReceipt 404:ClaudexorProblem 503:ClaudexorProblem|ErrorResponse",
        note="poll while state is running"),
    "POST /api/claudexor/wake": Shape("200:ClaudexorStatusResponse 503:ErrorResponse",
                                      note="starts the engine when needed, then reads status"),
    "POST /api/claudexor/login": Shape(
        "200:ClaudexorLoginJobResponse 400,409:ErrorResponse|ClaudexorLoginJobProblem 503:ClaudexorLoginJobProblem|ErrorResponse",
        "ClaudexorLoginRequest", note="not idempotent: every call may start a job; can start the engine"),
    "GET /api/claudexor/login/{job_id}": Shape(_LOGIN_JOB, note="a snapshot {job, cursor, sequence, deviceCode?}; poll it"),
    "DELETE /api/claudexor/login/{job_id}": Shape(_LOGIN_JOB, note="cancel; a repeat is harmless"),
    "POST /api/claudexor/login/{job_id}/input": Shape(
        "200:ClaudexorLoginJobResponse 400:ErrorResponse 404,409:ClaudexorLoginJobProblem 503:ErrorResponse",
        "ClaudexorLoginInputRequest"),
    "POST /api/claudexor/login/{job_id}/reconcile": Shape(
        "200:ClaudexorLoginJobResponse 404,409,410:ClaudexorLoginJobProblem 503:ErrorResponse"),
    "DELETE /api/claudexor/credential-profiles/{harness}/{profile_id}": Shape(
        "200:ClaudexorCredentialProfileDeleteResponse 400,503:ErrorResponse", note="every engine refusal answers 503"),
    "PATCH /api/claudexor/credential-profiles/{harness}/{profile_id}": Shape(
        "200:ClaudexorProfileEnabledResponse 400,503:ErrorResponse", "OwnerEnabledRequest",
        note="safe to repeat; every engine refusal answers 503"),
}

# Middleware-owned routes: ``server_auth.NetworkAuthGate`` answers them itself, and only while it
# enforces a configured network password for a non-loopback peer (otherwise they are not mounted).
AUTH_ENDPOINTS: dict[str, Shape] = {
    "GET /auth/login": Shape("200:html", query="next builtin", note="selected login or builtin form; builtin=1 forces builtin; next is a same-origin path"),
    "POST /auth/login": Shape(
        "200:AuthLoginResponse 303:empty 401:ErrorResponse|html", "AuthLoginRequest|form",
        note="JSON answers 200 with the session cookie; a form answers 303 to next"),
    "GET /auth/logout": Shape("303:empty", query="next"),
    "POST /auth/logout": Shape("303:empty", "json|form"),
}

# What a same-origin proxy routes to the core (segment prefixes), and the replaceable main
# interface the core serves only for a builtin or directory OUROBOROS_UI_DIR selection.
CORE_PATHS: tuple[str, ...] = ("/api", "/auth", "/onboarding", "/ws")
INTERFACE_PATHS: tuple[str, ...] = ("/", "/static")

# (inbound, outbound) contract TypedDict per ``contracts.WS_MESSAGE_TYPES`` member; "" = that direction
# does not exist. Extension types are namespaced per skill instead (``EXTENSION_WS_NAMESPACE``).
WS_SHAPES: dict[str, tuple[str, str]] = {
    "chat": ("ChatInbound", "ChatOutbound"),
    "command": ("CommandInbound", ""),
    "panic_accepted": ("", "PanicAcceptedOutbound"),
    "photo": ("", "PhotoOutbound"),
    "video": ("", "VideoOutbound"),
    "document": ("", "DocumentOutbound"),
    "links": ("", "LinksOutbound"),
    "quiz": ("", "QuizOutbound"),
    "quiz_state": ("", "QuizStateOutbound"),
    "card": ("", "CardOutbound"),
    "card_state": ("", "CardStateOutbound"),
    "typing": ("", "TypingOutbound"),
    "log": ("", "LogOutbound"),
    "heartbeat": ("", "HeartbeatOutbound"),
    "extension_lifecycle": ("", "ExtensionLifecycleOutbound"),
    "message_annotation": ("", "MessageAnnotationOutbound"),
    "projects_changed": ("", "ProjectsChangedOutbound"),
    "task_named": ("", "TaskNamedOutbound"),
    "update_status_ready": ("", "UpdateStatusReadyOutbound"),
    "update_progress_changed": ("", "UpdateProgressChangedOutbound"),
    "ui_language_changed": ("", "UiLanguageChangedOutbound"),
    "ui_i18n_updated": ("", "UiI18nUpdatedOutbound"),
}
EXTENSION_WS_NAMESPACE = (
    "ext_<n>_<token>_<message>: one prefix per skill (WidgetTab.ws_prefix); an inbound frame's handler "
    "result returns as <type>.reply {data}, a refused or failed one as a log frame; skills broadcast "
    "only inside their own prefix")
