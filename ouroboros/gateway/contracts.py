"""Gateway HTTP/WebSocket contracts (v1): descriptive, not runtime validation.

Keep discriminating ``type`` keys required; optional fields use ``NotRequired``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from ouroboros.cost_projection import CostPresentation
from ouroboros.physical_attempt_context import ContextAttemptFact  # noqa: F401 -- public re-export
from ouroboros.delegate_custody_usage import CostCustody
from ouroboros.gateway.reference_contracts import (  # noqa: F401 -- public contract facade
    ChatInbound, ChatReferenceFields, ConnectionProfileOutbound, HostNoticeField, ReplyExcerpt, SharedRowFrame,
)
from ouroboros.storage import (  # noqa: F401 -- public re-exports
    StorageCount, StorageCounting, StorageError, StoragePart, StoragePlace, StorageResponse, StorageRest,
    StorageVolume, StorageRemover,
)
from ouroboros.owner_keep import OwnerKeepRequest, OwnerKeepResponse  # noqa: F401 -- public re-exports
from ouroboros.launch_custody import (  # noqa: F401 -- public re-exports
    KeptService, KeptServiceStopRequest, KeptServiceStopResponse, KeptServicesResponse,
)
from ouroboros.storage_deleted import (  # noqa: F401 -- public re-exports
    StorageActiveProject, StorageDeletedFolder, StorageDeletedProject, StorageDeletedProjects, StorageFolderPart,
)
from ouroboros.storage_free import (  # noqa: F401 -- public re-exports
    StorageFree, StorageFreeEntry, StorageFreeGroup, StorageFreeItem, StorageFreeItemsResponse, StorageFreeKept, StorageFreePool,
    StorageFreeRequest, StorageFreeRun,
)
from ouroboros.gateway.claudexor_contracts import (  # noqa: F401 -- public re-exports
    ClaudexorReadState, ClaudexorStatusReads, ClaudexorPassiveReadError, ClaudexorQuotaResponse,
    ClaudexorStatusResponse,
)
from ouroboros.gateway.work_contracts import ProcessingTiers, TaskCancelRequest, WorkLeadFields, WorkStateFields  # noqa: F401
from ouroboros.gateway.project_folder_contracts import FsDirsEntry, FsDirsResponse  # noqa: F401 -- public re-exports
from ouroboros.gateway.history_contracts import AgentNoteFields, ChatHistoryResponse, PeerLetterCardFact, PresenceContact, RequestedExecutorRoute  # noqa: F401 -- public re-exports
from ouroboros.gateway.health_contracts import BootDocument, BootJob, BootStep, HealthResponse  # noqa: F401 -- public re-exports
from ouroboros.gateway.attachment_contracts import (  # noqa: F401 -- public re-exports
    AttachmentManifestEntry, ChatAttachmentInbound, ChatAttachmentView, QuizAnswerFile,
)
from ouroboros.gateway.contributions import UiContribution, UiContributionsResponse, UiPreferencesResponse
from ouroboros.gateway.widgets import (
    ExtensionLiveSnapshot, WidgetFaultRequest, WidgetFaultResponse, WidgetTab, WidgetsResponse,
    WidgetAssetLeaseRequest, WidgetAssetTokenRequest, WidgetAssetLeaseResponse, WidgetAssetRevokeResponse,
)
from ouroboros.gateway.decision_contracts import (  # noqa: F401 -- public re-exports
    DecisionRequest, DecisionResponse, SecretDecisionRequest, SecretDecisionResponse, SecretEvent, SecretOutcome,
    SecretAllow, SecretAllowRequest, SecretConsumer, SecretKeyState, SecretRequestField, SecretRequestMeta,
    SecretSkillFacts,
)
from ouroboros.gateway.card_contracts import CardOutbound, CardStateOutbound  # noqa: F401 -- public re-exports
from ouroboros.gateway.cost_breakdown import TaskCostBreakdown, TaskCostSlice  # noqa: F401 -- public re-exports
from ouroboros.gateway.update_contracts import (  # noqa: F401 -- public re-exports
    UpdateApplyErrorResponse, UpdateApplyRequest, UpdateApplySuccessResponse, UpdateDescriptionErrorResponse,
    UpdateDescriptionLatestResponse, UpdateDescriptionRequest, UpdateDescriptionResponse, UpdateMergePlan,
    UpdatePreflightRequest, UpdatePreflightResponse, UpdateProgressChangedOutbound, UpdateStatusReadyOutbound,
)
from ouroboros.gateway.management_contracts import (  # noqa: F401 -- public re-exports
    ClaudexorLoginJobProblem,
    ExtensionsIndexResponse, LocalModelStatusResponse, MarketplaceInstalledResponse, MarketplaceSearchResponse,
    McpStatusResponse, ModelCatalogResponse, OpenAICompatibleModelsResponse, SkillLifecycleQueueResponse,
)
from ouroboros.gateway.schedule_contracts import (  # noqa: F401 -- public re-exports
    ScheduleActionResponse, ScheduledTasksResponse, ScheduleDeleteResponse, ScheduleUpsertResponse,
)

try:  # Python 3.11+
    from typing import Literal, NotRequired, Required, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import Literal, NotRequired, Required, TypedDict  # type: ignore[assignment]


class TaskReviewHistoryResponse(TypedDict):
    """One exact task's paged ledger projection; local gaps do not hide healthy records."""

    task_id: str
    review_projection: Dict[str, Any]  # compact panels with local gaps, never a complete-history claim
    next_offset: Optional[int]  # Null marks the last ledger page.


class TaskConstraintInbound(TypedDict, total=False):
    mode: str
    skill_name: str
    payload_root: str
    allow_enable: bool
    allow_review: bool
    extra_allowlist: list[str]


class CommandInbound(TypedDict):
    """Inbound WS command message. ``ui_control`` (``{name, owner_text?}``) marks a command a host-drawn
    control composed: the press is the owner's, its words are not, so no owner row is written for it."""

    type: Literal["command"]
    cmd: str
    client_surface: NotRequired[dict]
    ui_control: NotRequired[dict]


class PanicAcceptedOutbound(TypedDict):
    """The emergency owner accepted this connection's request; exit is unproved."""

    type: Literal["panic_accepted"]


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


class ChatOutbound(ChatReferenceFields, SharedRowFrame, HostNoticeField, AgentNoteFields):
    """Outbound WS chat frame."""

    presence_contact: NotRequired[PresenceContact]
    type: Literal["chat"]
    role: Literal["user", "assistant", "system"]
    content: str
    ts: str
    ingress_accepted: NotRequired[bool]  # Canonical inbound row saved; not processing/start proof.
    ingress_dispatched: NotRequired[bool]  # This live host process accepted the row and entered its dispatch. Absence after a host restart is unknown.
    ingress_pending: NotRequired[bool]  # This live host process accepted the row and has entered or refused neither yet. A later echo or history read determines whether dispatch occurred or was refused.
    attachments: NotRequired[List[Union[ChatAttachmentView, QuizAnswerFile]]]  # owner message's attachments (same views as history); a project_question_pointer's: its answer's files
    text_placeholder: NotRequired[bool]  # owner row whose text the host wrote (no words were sent); shown as no caption
    reply_excerpt_disabled: NotRequired[bool]  # a wordless or mixed card display; whole-row Reply still works
    markdown: NotRequired[bool]
    is_progress: NotRequired[bool]
    task_id: NotRequired[str]
    task_attempt: NotRequired[int]
    origin_message_ref: NotRequired[Dict[str, Any]]  # Host-captured inbound identity for a correlated operation's terminal reply.
    # X3: a repair receipt whose managed task id the router mints only at promotion (typed truth, no invented id).
    task_id_pending: NotRequired[bool]
    # The confirmed work this final answer replies about (finish_task about_work): a direct turn's
    # terminal frame addresses exactly the task this owner turn started or told, never a fresher id.
    about_work_task_id: NotRequired[str]
    # "finalizing": early answer during synthesis, NOT terminal; task_done settles the card/turn.
    task_phase: NotRequired[str]
    # Direct/ephemeral completed/failed/cancelled/rejected_duplicate settles activity, not early answers.
    task_terminal_status: NotRequired[str]  # Stamped on a direct turn’s terminal error path as well as its final.
    ephemeral_decision: NotRequired[bool]
    tool_calls: NotRequired[int]
    rounds: NotRequired[int]
    suggested_name: NotRequired[str]
    model_execution: NotRequired[Dict[str, Any]]
    quiz_id: NotRequired[str]
    quiz_state: NotRequired[str]
    project_chat_id: NotRequired[int]
    source_status: NotRequired[str]
    owner_wait_state: NotRequired[str]
    owner_wait_resume_reason: NotRequired[str]
    wait_for_answer: NotRequired[bool]
    wait_ended_at: NotRequired[str]
    question: NotRequired[str]
    options: NotRequired[List[str]]
    option_details: NotRequired[List[str]]  # aligned with options; absent for a legacy label-only ask
    stake: NotRequired[str]
    assumption: NotRequired[str]
    recommended_index: NotRequired[int]
    answered_index: NotRequired[int]
    comment: NotRequired[str]
    host_facts: NotRequired[str]  # the mirrored card's host sentence (QuizOutbound.host_facts)
    closed_reason: NotRequired[str]
    closed_at: NotRequired[str]
    closed_by_task_id: NotRequired[str]
    closed_actor: NotRequired[str]
    closed_request_id: NotRequired[str]
    materials: NotRequired[List[Dict[str, Any]]]  # validated existing task artifact references: {task_id, name, label, size, sha256, download_url}
    question_delivery: NotRequired[Dict[str, Any]]  # canonical kind/state; pending stays current; kind: publication | answer | closure; state: owed | delivered; reason_code and dismissed are optional settlement facts
    # A mirrored secret request: its protected form, outcome and unconfirmed-save facts (no value).
    secret_request: NotRequired[SecretRequestMeta]
    secret_outcome: NotRequired[SecretOutcome]
    secret_save: NotRequired[Dict[str, Any]]  # An attempt's claim; state "writing" is an unconfirmed save.
    task_incident: NotRequired[str]
    # Historical rows only: the retired cancellation-fault notice named the PHYSICAL task it could not
    # settle when that differed from the logical id. Declared so stored rows keep their shape.
    cancel_physical_task_id: NotRequired[str]
    toast_once: NotRequired[str]
    # The one-shot toast's valence (warn/ok/error; the reaper's rail ``warning`` is normalizeTone's ``warn``
    # alias), stamped by the producer that knows wait/recovery/exhaustion; absent = the browser's alarm tone.
    toast_tone: NotRequired[str]
    lifecycle: NotRequired[Dict[str, Any]]
    # C4 multi-chat dedupe: a duplicate lifecycle initiator's typed pointer to the job that already owns the
    # routing ({job_id, kind, target, status, chat_id}); the first initiator's chat keeps the progress stream.
    lifecycle_pointer: NotRequired[Dict[str, Any]]
    subagent_event: NotRequired[str]
    subagent_task_id: NotRequired[str]
    root_task_id: NotRequired[str]
    parent_task_id: NotRequired[str]
    delegation_role: NotRequired[str]
    subagent_role: NotRequired[str]
    accepted: NotRequired[bool]
    active_subagent_count: NotRequired[int]
    max_active_subagents: NotRequired[int]  # the owner's number per root; absent: no number
    queued_behind_active_cap: NotRequired[bool]
    required_capabilities: NotRequired[list[str]]
    write_surface: NotRequired[str]
    model_lane: NotRequired[str]
    requested_model_lane: NotRequired[str]
    effective_model_lane: NotRequired[str]
    # The opaque route resolve_subagent_dispatch stamped (a dispatch decision, no landing receipt: a reduction is
    # capability_delta's), and the {kind, target_id} the delegation selected, frozen at scheduling, shown from Queued.
    executor_route: NotRequired[str]
    requested_executor_route: NotRequired[RequestedExecutorRoute]  # absent = not recorded; the pre-dispatch identity an agent_session shows as its harness chip; api_model and unrouted native work show none
    # Recorded coordinator facts on live/history frames; absence carries no billing or provenance claim.
    parent_cognitive_route: NotRequired[Dict[str, Any]]  # never infer it from the child's lane
    coordinator_route: NotRequired[str]
    use_local_model: NotRequired[bool]
    coordinator_billing: NotRequired[str]  # Recorded subscription, measured or unknown; replay never infers it.
    coordinator_provenance: NotRequired[str]
    # The effort decision at dispatch (a session row: its leaf's): level, the parent's request, source
    # auto|pin|model_name|cyber. Empty/absent = unknown, no chip.
    effort_level: NotRequired[str]
    effort_requested: NotRequired[str]  # The parent's request when it made one.
    effort_source: NotRequired[str]  # auto | pin | model_name | cyber.
    # Latest observed progress actor, NOT terminal evidence or current liveness. Own task_id/task_attempt/run_id/
    # attempt_id, harness_id, phase, revision; optional model has explicit model_source (requested or observed).
    executor_observation: NotRequired[Dict[str, Any]]
    # Host observation: v/task_id/run_id/after_seq/through_seq, source (run_events|timeline_window, read_through/ref/
    # provisional), typed parts/technical/gaps/omitted/latest_message; (run_id, seq) is identity, cuts count Unicode
    # code points and cuts_truncated/seqs_truncated disclose preview loss; source.ref = redacted JSONL via confined
    # task-file ?source= (503 if unsupported). Never narration or terminal evidence; shape defined by ouroboros/delegate_activity.py.
    delegated_activity: NotRequired[Dict[str, Any]]
    # Terminal evidence from subagents.envelope_from_task: run started/settled/succeeded counts, failure states,
    # disclosed spend/estimated flag, engine models, nanny_nudge_recorded, delegate_start_attempted. Absence = no
    # evidence yet, never native execution.
    execution_evidence: NotRequired[Dict[str, Any]]  # evidence_read_failed makes its zero counts unknown (ouroboros/delegate_evidence.py)
    # Custody fact beside executor_route: "harness_used" | "harness_attempted" | "native_only" (terminal; absent = no claim).
    actual_substrate: NotRequired[str]  # From custody, never usage or rounds: succeeded = harness_used; started without success = harness_attempted; no start = native_only, asserted only with readable evidence.
    model: NotRequired[str]
    task_group_id: NotRequired[str]
    task_event: NotRequired[str]
    status: NotRequired[str]
    # Host-attested marker (P5), stamped by the supervisor's delivery seam ONLY for a task POST /api/tasks/{id}/cancel
    # will actually stop — a lineage-resolved pooled ROOT or the live in-process direct-chat turn (the endpoint's own
    # ownership reader, supervisor.workers.direct_chat_turn); never a subagent or ephemeral decision frame. Gates "Cancel run".
    cancelable: NotRequired[bool]
    _is_direct_chat: NotRequired[bool]  # lane fact stamped on a direct turn's own frames (progress, tool and each task_done), so the UI needs no census for it
    # Host checkpoint/fallback/plan/acceptance/nudge/transport/density rows stay visible without
    # claiming the title or activity line. Older rows keep their original promotion behavior.
    narration: NotRequired[bool]  # progress VOICE: the model's own round narration (true) vs a host note (false); absent = legacy
    reasoning: NotRequired[bool]  # true only on narration that is the model's display reasoning: details, never title/line; absent on visible speech or legacy rows
    spoken: NotRequired[Dict[str, Any]]  # call_id, item_id, played_ms (milliseconds), status: played/generated/interrupted; kept on history replay
    initiator: NotRequired[str]  # origin label: "consciousness" on a wake-up's frames/rows (and its roots); absent on an owner's turn
    continue_requester: NotRequired[str]  # who asked for this Continue, read from its admission: owner|ouroboros|unknown (undeclared); absent elsewhere
    # A place's work (and what it starts): {place_id, place_label?, provider?, actor?: {id?, label?, owner?, bot?}, wakes,
    # labels?, routed?}, by value from its admission's focus (``subagent_messages.place_origin``); absent on other work.
    place_origin: NotRequired[Dict[str, Any]]
    # Monetary projections are nullable when the physical-attempt ledger cannot be read: ``None`` is distinct from a
    # confirmed $0. The honest names (accounted upper bounds, not settled receipts), the only outbound spellings since
    # ABI 7.0 dropped the ``cost_usd[_with_children]`` aliases; cost_projection.py is the one author.
    accounted_upper_bound_usd: NotRequired[Optional[float]]
    accounted_upper_bound_usd_with_children: NotRequired[Optional[float]]  # C2: subtree upper bound (formerly aliased `cost_usd_with_children`); null when unknown.
    cost_accounting_status: NotRequired[Literal["available", "unavailable"]]
    cost_accounting_error: NotRequired[str]
    cost_final: NotRequired[bool]
    cost_with_children_partial: NotRequired[bool]
    reserved_usd: NotRequired[Optional[float]]
    unresolved_upper_bound_usd: NotRequired[Optional[float]]
    unknown_unmetered: NotRequired[Optional[int]]
    # Count of OPEN ledger rows — the disclosed cause of ``cost_final: false``, which can hold at zero dollars.
    non_final_rows: NotRequired[Optional[int]]
    # C12: the ledger's own INTEGRITY marker from the cost authority (`reconstruct_task_cost`), carried so an
    # amount computed over a degraded ledger never reaches a surface looking like one over a sound ledger.
    ledger_integrity_degraded: NotRequired[Optional[bool]]
    # Closed shape owned by the cost producer; null means ledger unavailable.
    cost_presentation: NotRequired[Optional[CostPresentation]]
    cost_custody: NotRequired[Optional[CostCustody]]  # null if unreadable; beside it, cost_presentation.tracked_amount needs a priced, bounded row: empty or all-unpriced is not a measured zero
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
    cancel_origin: NotRequired[Dict[str, Any]]  # the typed stop an owner-stopped turn's final carries: {source, stop_policy}
    finalization_recovery: NotRequired[Dict[str, Any]]  # an adopted answer's custody (cause, answer, delivery_id), read by the status fold
    terminal_origin: NotRequired[str]  # who authored the terminal text: model_final or a host origin
    delegated_runs_unreconciled: NotRequired[List[str]]  # the row's current custody debt beside its cause (#1583)
    peer_letter_card: NotRequired[PeerLetterCardFact]  # a letter's card on the progress line of the work that wrote it
    review_status: NotRequired[Dict[str, Any]]
    # panels retain subject/goal/revision/wave; actors separate requested/observed models,
    # dict failure_details and current source selectors. gaps name failed records;
    # review_records_omitted is 0|"1+"|"unknown", never proof of a review by itself.
    # See ouroboros/review_projection.py. Host-stopped waves name the masked failure and its own source;
    # findings counts are present only for parsed responses. Late settlements bind the reviewed subject
    # to emitted bytes (or identify owed/unknown); send_handler_returned is not human receipt.
    # Acceptance incidents retain real attempts even without a panel, resolved rows remain history.
    # Latest 16 commit landings stay visible; omitted landings remain in the retained source; absence
    # means no recorded landing.
    review_projection: NotRequired[Dict[str, Any]]
    body_adoptions: NotRequired[List[Dict[str, Any]]]  # the task's settled adoptions (TaskDetailResponse), on its summary
    review_overrides: NotRequired[Dict[str, Any]]  # the work's review-gate field (TaskDetailResponse), on its summary
    worker_saturation_warning: NotRequired[bool]
    source: NotRequired[str]
    sender_label: NotRequired[str]  # a program note's rendered identity, as in history
    sender_session_id: NotRequired[str]
    client_message_id: NotRequired[str]
    subagents_mode: NotRequired[Literal["swarm"]]  # Accepted owner message, never a client claim.
    transport: NotRequired[TransportMetadata]
    # Typed message kind; role alone selects authorship (DESIGN: Chat authorship and System rows).
    system_type: NotRequired[str]
    # Live frame only: a routing refusal's System row names the carrier its receipt selected (routing_ack = the ACK).
    routing_notice_carrier: NotRequired[Literal["routing_ack", "system"]]
    # Host placement of a task's System row: timeline or Reviews; absent = ordinary. card_row_id is stable.
    card_row: NotRequired[Literal["timeline", "reviews"]]
    card_row_id: NotRequired[str]
    card_row_revision: NotRequired[int]  # canonical source order, independent of delivery timestamp
    terminal_receipt: NotRequired[Literal["retained_work"]]  # a placed row typed as a terminal receipt, not progress
    # Uses its retained report source_ref and does not embed the transcript.
    late_evidence: NotRequired[Dict[str, Any]]  # applied source or merge_review identity/revision + retained report; merge_review refreshes on history reads only
    target_label: NotRequired[str]  # event-time human presentation; raw task/project ids remain machine keys
    project_id: NotRequired[str]
    project_name: NotRequired[str]
    task_name: NotRequired[str]  # structured work title on Project creation/transfer entries.
    handoff_id: NotRequired[str]  # immutable origin/destination receipt identity
    terminal_time: NotRequired[Dict[str, Any]]  # host-owned occurrence, separate from publication ts
    completion_answer: NotRequired[str]  # a Project root's model-authored final answer, mirrored into Main (DESIGN)
    completion_reason: NotRequired[str]  # one recorded nonclean-outcome fact beside the answer
    wake_summary: NotRequired[str]  # a Project root's own closing sentence, for its Main portal
    completion_notice: NotRequired[Dict[str, Any]]  # transport view: neutral/detail/phase/salvage; Main adds cause and undescribed codes
    delivery_id: NotRequired[str]  # stable logical publication identity across transport retries
    set_at: NotRequired[str]  # a `reminder` row's provenance, live frame only (its text carries the signature): when written
    scheduled_for: NotRequired[str]  # a `reminder` row: the due point it was written for
    delivered_at: NotRequired[str]  # when the host showed it (later than due after downtime)
    chat_id: NotRequired[int]  # present on some transport re-broadcast paths


class _MediaOutbound(ChatReferenceFields):
    """The fields the photo, video and document frames share; each frame adds its type, payload and own keys."""

    role: Literal["user", "assistant"]
    mime: str
    ts: str
    caption: NotRequired[str]
    # Canonical durable task-artifact URL of the stored media (a document's captured file), replayed by chat history:
    # the live frame carries the bytes inline, so replay never needs to persist them.
    download_url: NotRequired[str]
    # Second address for the SAME bytes on the long-shipped /api/files/download route, non-empty only when the stored
    # file resolves inside the current file-browser root. Packaged desktop launchers gate their file bridge to a URL
    # allowlist that predates the artifact route, so the browser uses download_url and the host bridge prefers this one.
    download_url_compat: NotRequired[str]
    content: NotRequired[str]
    source: NotRequired[str]
    sender_label: NotRequired[str]
    sender_session_id: NotRequired[str]
    client_message_id: NotRequired[str]
    transport: NotRequired[TransportMetadata]
    chat_id: NotRequired[int]
    task_id: NotRequired[str]


class PhotoOutbound(_MediaOutbound):
    """Outbound WS photo frame."""

    type: Literal["photo"]
    image_base64: str
    subagents_mode: NotRequired[Literal["swarm"]]  # Accepted owner message, never a client claim.


class VideoOutbound(_MediaOutbound):
    """Outbound WS video frame."""

    type: Literal["video"]
    video_base64: str


class DocumentOutbound(_MediaOutbound):
    """Outbound WS document/file frame."""

    type: Literal["document"]
    file_base64: str
    filename: str
    file_ref: NotRequired[Dict[str, Any]]
    size_bytes: NotRequired[int]


class LinkAction(TypedDict):
    """One validated HTTP(S) action in a structured links frame."""

    label: str
    url: str


class LinksOutbound(ChatReferenceFields):
    """Outbound group of first-class external link buttons."""

    type: Literal["links"]
    role: Literal["assistant"]
    actions: list[LinkAction]
    ts: str
    title: NotRequired[str]
    chat_id: NotRequired[int]
    task_id: NotRequired[str]
    transport: NotRequired[TransportMetadata]


class QuizOption(TypedDict):
    """One selectable option on an owner quiz card."""

    label: str
    detail: NotRequired[str]
    recommended: NotRequired[bool]


class QuizOutbound(ChatReferenceFields, SharedRowFrame):
    """Outbound owner quiz card: a typed question with option buttons.

    Optional questions continue under ``assumption``; ``wait_for_answer`` marks
    required waiting without an implied answer. ``state`` carries lifecycle.
    Replay merges the stored ``answered_index``, verbatim ``comment`` (without an index,
    the owner's whole answer) and ``answer_row_address``, the answer row its card shows.
    A private Presence question's one frame names its work's Project in ``also_rooms``."""

    type: Literal["quiz"]
    role: Literal["assistant"]
    quiz_id: str
    question: str
    options: list[QuizOption]
    stake: str
    assumption: str
    wait_for_answer: NotRequired[bool]
    state: str
    wait_ended_at: NotRequired[str]
    presence_contact: NotRequired[PresenceContact]
    ts: str
    answered_index: NotRequired[int]
    comment: NotRequired[str]
    attachments: NotRequired[List[QuizAnswerFile]]  # replay: the files a Telegram Reply answered with, never a link
    answer_row_address: NotRequired[str]  # replay only, never the live quiz_state frame
    host_facts: NotRequired[str]  # host-written, never the model's: asking task, run start, owner's last message
    materials: NotRequired[List[Dict[str, Any]]]  # as ChatOutbound.materials
    question_delivery: NotRequired[Dict[str, Any]]  # as ChatOutbound.question_delivery
    closed_reason: NotRequired[str]
    closed_at: NotRequired[str]
    closed_by_task_id: NotRequired[str]
    closed_actor: NotRequired[str]
    closed_request_id: NotRequired[str]
    chat_id: NotRequired[int]
    task_id: NotRequired[str]
    transport: NotRequired[TransportMetadata]
    secret_request: NotRequired[SecretRequestMeta]  # an integration-secret request: answered only by its form


class QuizStateOutbound(TypedDict):
    """Outbound WS lifecycle update for an already-rendered quiz card.

    A separate discriminator (not a second ``quiz`` frame): the display path dedupes
    quiz frames by ``quiz:{quiz_id}:{ts}``, so a state change must never look like a
    new card. ``answered_index`` rides only with the ``answered`` state.
    """

    type: Literal["quiz_state"]
    question_delivery: NotRequired[Dict[str, Any]]  # as ChatOutbound.question_delivery
    quiz_id: str
    task_id: str
    state: str
    ts: str
    answered_index: NotRequired[int]
    wait_for_answer: NotRequired[bool]  # additive: False once a bounded wait closed (the card stays open)
    comment: NotRequired[str]  # #471: the recorded free-text answer, absent when empty, shown as replay shows it
    attachments: NotRequired[List[QuizAnswerFile]]  # its Telegram Reply's files (chat_uploads.answer_files), likewise
    chat_id: NotRequired[int]
    closed_reason: NotRequired[str]
    closed_at: NotRequired[str]
    closed_by_task_id: NotRequired[str]
    closed_actor: NotRequired[str]
    closed_request_id: NotRequired[str]
    secret_outcome: NotRequired[SecretOutcome]  # a secret request's typed completion, metadata only


class TypingOutbound(TypedDict):
    """Outbound WS typing indicator."""

    type: Literal["typing"]
    action: str
    # Multi-project: stamps the thread so the client fan-out routes a project
    # task's typing indicator to its panel instead of defaulting to main.
    chat_id: NotRequired[int]
    # Server-stamped when chat_id is a reserved Project thread: Main never adopts it, even before the browser learns it.
    project_thread: NotRequired[bool]
    activity_id: NotRequired[str]
    client_message_id: NotRequired[str]
    phase: NotRequired[str]
    # Stamped for a registry-tracked turn ("direct_chat") or a RUNNING queue root ("managed_task"); empty for children.
    # No in-repo client reads it (wire compatibility): only the census inserts into the header live-set.
    kind: NotRequired[str]


class LogOutbound(TypedDict):
    """Outbound WS log event."""

    type: Literal["log"]
    data: Dict[str, Any]
    # Multi-project: surfaced at top level so live task progress routes to
    # its own thread (Main admits only unstamped non-project frames).
    chat_id: NotRequired[int]
    # Server-stamped when chat_id is a reserved Project thread: Main never adopts it, even before the browser learns it.
    project_thread: NotRequired[bool]


class HeartbeatOutbound(TypedDict):
    """Outbound heartbeat emitted by ``server_runtime.ws_heartbeat_loop``."""

    type: Literal["heartbeat"]
    ts: NotRequired[str]


class ExtensionLifecycleOutbound(TypedDict):  # On this notice, re-read GET /api/widgets and GET /api/extensions.
    """Outbound extension lifecycle notification."""

    type: Literal["extension_lifecycle"]
    skill: NotRequired[str]
    action: NotRequired[str]
    status: NotRequired[str]
    reason: NotRequired[str]
    data: NotRequired[Dict[str, Any]]


class ProjectsChangedOutbound(TypedDict):
    """Outbound notice that the project registry changed server-side (e.g. the agent's ``promote_chat_to_task``
    created/bound a project). The client refreshes its project nav + WS-fan-out ``projectChatIds`` on receipt;
    ``chat_id`` lets it learn the new project thread immediately, before the /api/state round-trip."""

    type: Literal["projects_changed"]
    project_id: NotRequired[str]
    chat_id: NotRequired[int]


class UiPreferencesChangedOutbound(TypedDict):
    """Outbound notice that ``POST /api/ui/preferences`` committed a change of the named groups
    (``contributions``, ``model_pins``); every window reads the named state again."""

    type: Literal["ui_preferences_changed"]
    keys: List[str]


class MessageAnnotationOutbound(TypedDict):
    """Bubble-free presentation update for one owner message; ``cause``: the host's sentence for a refused act."""

    type: Literal["message_annotation"]
    annotation_type: Literal["routing_ack"]
    client_message_id: str
    action: str
    status: str
    suppress_bubble: bool
    chat_id: NotRequired[int]
    source_row_address: NotRequired[str]
    target: NotRequired[str]
    project_name: NotRequired[str]
    target_task_ids: NotRequired[list[str]]
    target_titles: NotRequired[list[str]]
    target_label: NotRequired[str]
    project_id: NotRequired[str]
    project_chat_id: NotRequired[int]
    options: NotRequired[List[Dict[str, Any]]]
    attachment_manifest: NotRequired[List[AttachmentManifestEntry]]
    # #198: the exact refusal-attempt identity — the picker card composes its decision_id
    # (routing:{client_message_id}:{routing_token}) from it; a frame without it renders text, never a card.
    routing_token: NotRequired[str]
    cause: NotRequired[str]  # absent on scheduled/delivered/pending and on the picker frame
    reasoning_effort: NotRequired[str]  # #1539: the explicit start a New task picked from this card requests
    # #966: the notice's SELECTED carrier (never delivery proof): this ACK, answering reply_to, or the System row.
    routing_notice_carrier: NotRequired[Literal["routing_ack", "system"]]  # A System row is selected when it answers no chat row.
    reply_to: NotRequired[Dict[str, Any]]  # the owner message that notice answers (host Reply reference).
    ts: NotRequired[str]


class ProjectCreateRequest(TypedDict, total=False):
    """POST /api/projects body (v6.59.0). ONE source: ``path`` (attach an existing
    owner folder; optional ``init_git`` attach-snapshot commit — never auto-init),
    ``git_url`` (server-side clone, typed ``auth_required`` on credential failure),
    ``with_workspace`` (genesis provision), or none (file-less project).
    ``name``-only creation derives a filesystem id."""

    id: str  # ``project_id`` is accepted as an alias
    name: str
    path: str
    init_git: bool
    git_url: str
    with_workspace: bool
    no_folder: bool  # records the choice to keep none; absent, a file-less room's first promoted task provisions one


class ProjectClosure(TypedDict):
    """A reversible decision; absent ``ProjectEntry.closed`` means open."""

    at: str
    by: str  # owner or ouroboros:<focus task id>
    note: str


class ProjectEntry(TypedDict, total=False):
    """A registry project row. ``provenance`` (attached|cloned|genesis|none) and ``clone_url``
    are historical facts; operational git data is read live from ``.git``. List rows are a
    compact projection with ``has_thread_activity``; a create/update answer is the full row
    (``created: false``: an idempotent repeat create)."""

    id: str
    name: str
    chat_id: int
    working_dir: str
    no_folder: Dict[str, str]  # {at, by}: the recorded choice to keep no folder, until one is chosen
    pinned: bool  # absent on legacy rows means unpinned; no routing/activity effect
    cognition: ProjectCognition
    model_choice: ChatModelChoice  # the room's own model for new owner work; never copied from Main
    processing_choice: Literal["", "standard", "fast", "economy", "ultrafast"]  # the room's own tier for new owner work; '' = Default
    provenance: str
    clone_url: str
    trusted_at: str
    origin: str
    created_at: str
    last_active_at: str  # technical activity: any row, journal, binding (the agent's list order)
    last_message_at: str  # the order a client lists by: newest message, else created_at; absent while undated
    lifecycle: Literal["active", "deleting", "tombstoned"]
    closed: ProjectClosure  # independent of lifecycle; room, history and routing stay available; absent means open
    routing_generation: int
    routing_incarnation: str
    visible_revision: int
    delete_error: str
    has_thread_activity: bool
    thread_activity_seen: bool
    last_task_result_id: str
    created: bool


class ProjectDeleteResponse(TypedDict):
    """POST /api/projects/{project_id}/delete — fence, quiesce, and tombstone;
    the immutable binding, working folder, history, and memory are preserved."""

    ok: bool
    project_id: str
    folder_untouched: bool


class TaskNamedOutbound(TypedDict):
    """Outbound notice that a project name was coined for a task's card (inline naming of
    a turn-into-project conversion; direct turns are not named in the background). The
    client sets the live card's title to ``suggested_name``. Not chat-scoped — carries
    only ``task_id`` and is a no-op unless a thread already holds that card."""

    type: Literal["task_named"]
    task_id: str
    suggested_name: str


class ErrorResponse(TypedDict):
    """A refusal: ``error`` is a sentence; ``code`` or ``reason_code`` types it when present."""

    error: str
    code: NotRequired[str]
    reason_code: NotRequired[str]


class StatusResponse(TypedDict):
    status: str


class EvolutionStateSnapshot(TypedDict):
    """Nested ``evolution_state`` block inside ``StateResponse``."""

    enabled: Optional[bool]  # None: unknown control (state unavailable/recovering, #1307)
    status: str
    detail: str
    cycle: int
    owner_chat_bound: bool
    last_task_at: str
    consecutive_failures: int
    # None when the remaining budget is infinite (unbudgeted): inf is not JSON
    # compliant, so the snapshot serializes it as null.
    budget_remaining_usd: Optional[float]
    budget_reserve_usd: float  # always 0.0: Evolution keeps no reserve of its own (owner, 2026-11-02)
    pending_count: int
    running_count: int
    queued_task_id: str
    running_task_id: str
    campaign: NotRequired[Dict[str, Any]]


class ActiveDirectTurn(WorkLeadFields):
    """An active owner turn: a conversation-lane row or an inline executor's actor.

    Rows of ``StateResponse.active_direct_turns`` (``conversation_lane.lane_turn_rows`` and
    ``DirectActivityRegistry.snapshot()``) always emit the seven identity/activity fields
    (empty string for absent values), so the mirror marks them required; ``model_waits``
    comes only from its live owner.
    """

    activity_id: str
    chat_id: int
    project_id: str
    client_message_id: str  # empty for managed queue rows.
    kind: str  # direct_chat | managed_task | presence — display label, not liveness. Moved Presence on the ordinary queue is managed_task.
    # Direct lane/registry: preparing | working | unknown; thinking is a compatible direct value used by typing frames.
    # The root census can also show budget_paused for a warm Pause. Queue/late work: queued | budget_pausing
    # | budget_paused | preparing (exact Resume not taken up)
    # | working | finalizing | unknown, even on a row labelled direct_chat.
    # Inline Presence: preparing | working; carried Presence follows its carrier. A pause keeps id/kind.
    phase: str
    started_at: float
    model_waits: NotRequired[Dict[str, Any]]  # Current owner facts; missing action evidence is unavailable. Per-model wait rows retain revision and task_attempt, and requires_owner_action when known.
    model_activity: NotRequired[Dict[str, Any]]  # content-free observation bound to the active Main call; not a quota wait or task liveness
    context_attempt: NotRequired[ContextAttemptFact]  # latest observation; released retracts its send, previous survives; kept past the call
    task_attempt: NotRequired[int]
    turn_phase: NotRequired[str]  # the last start fact its execution passed (TURN_PHASES); display only
    executor: NotRequired[str]  # where the turn runs: worker, inline_call or inline_degraded (task_carrier.EXECUTORS)
    carrier: NotRequired[str]  # a lane row: awaiting (no carrier holds it yet) or worker; a worker-held running root: worker


class ActiveChatActivity(ActiveDirectTurn):  # global root census (kind also presence); phase and control facts keep their own authority
    last_line: NotRequired[str]  # visible model narration, never host notes or display reasoning; empty when not recorded
    last_line_at: NotRequired[str]
    tool_calls: NotRequired[int]
    detached: NotRequired[bool]  # durable detached_at, independent of carrier/phase
    place_origin: NotRequired[Dict[str, Any]]  # ChatOutbound.place_origin's shape: a Main portal's only carrier of it
    project_name: NotRequired[str]
    status: NotRequired[str]  # recorded lifecycle outcome; phase still owns current activity.
    outcome_axes: NotRequired[Dict[str, Any]]
    reason_code: NotRequired[str]
    cancel_origin: NotRequired[Dict[str, Any]]  # the stored result's stop origin, read by the same status fold
    finalization_recovery: NotRequired[Dict[str, Any]]  # the stored result's adopted answer, read by that fold
    terminal_origin: NotRequired[str]
    delegated_runs_unreconciled: NotRequired[List[str]]  # the stored result's current custody debt, read by that fold
    root_phase_checkpoint: NotRequired[Dict[str, Any]]  # compact post_task_synthesis fact only.
    timeout_retry_from: NotRequired[str]  # explicit automatic-retry predecessor, never owner Continue.
    original_task_id: NotRequired[str]
    owner_wait: NotRequired[Dict[str, Any]]  # the waited quiz_id and its wait facts, independent of a Project pointer; carries its quiz state and optional wait_ended_at
    required_question: NotRequired[Dict[str, Any]]  # read-only pointer to the current required Project quiz.
    required_question_unavailable: NotRequired[bool]  # a recorded owner-question wait whose detail could not be read: possibly blocked, never "no question".
    project_admission_hold: NotRequired[Dict[str, Any]]  # accepted unstarted work waiting for original Project authority.
    pause_cause: NotRequired[str]  # budget | owner | restart | sleep | authority | unknown; display only
    finishing_reviews: NotRequired[bool]  # review work an owner Pause lets finish runs on; display only, no count
    late_work: NotRequired[Dict[str, Any]]  # answered root whose late work remains: title, answered_at; display only
    can_resume: NotRequired[bool]  # Positive current queue/grant authority, never derived from a label.
    close_expired: NotRequired[Dict[str, Any]]  # exact WorkControl, not Resume authority
    cancelable: NotRequired[bool]
    presence_pause: NotRequired[bool]  # a live Presence turn moved to a Project as text (never a live call): Pause saves it, Restart returns it

class EffortRange(TypedDict):  # helper min ≤ max; primary recommended is independent; each an EFFORT_SCALE tier
    min: str
    recommended: str
    max: str


class ProjectCognition(TypedDict):
    subagents_mode: Literal["off", "auto", "swarm"]
    context_mode: str
    effort_range: EffortRange


class OwnerEffortRangeResponse(TypedDict):
    ok: bool
    effort_range: EffortRange
    project_id: str
    cognition: ProjectCognition


class OwnerSubagentsModeResponse(TypedDict):
    ok: bool
    project_id: str
    subagents_mode: Literal["off", "auto", "swarm"]
    cognition: ProjectCognition


class StateResponse(WorkStateFields):
    """Shape of ``GET /api/state`` (happy path); work and recovery fields come from ``WorkStateFields``."""

    uptime: int
    workers_alive: int  # live worker slots of both lanes; inline carriers may be thread-backed.
    workers_total: int  # every slot of both lanes; never the project ceiling or assignable capacity.
    pending_count: int
    running_count: int
    spent_usd: Optional[float]
    budget_limit: float
    budget_pct: Optional[float]
    branch: Optional[str]  # null for detached HEAD or unavailable checkout identity
    sha: str
    evolution_enabled: Optional[bool]  # None: an unknown control (state unavailable/recovering), never "off"
    bg_consciousness_enabled: Optional[bool]  # Null means the control is unknown, never off.
    state_quality: Dict[str, Any]  # state.json read quality: {quality, source, unconfirmed}
    evolution_cycle: int
    evolution_state: EvolutionStateSnapshot
    bg_consciousness_state: Dict[str, Any]  # the alarm clock's snapshot + server projection (status/detail).
    spent_calls: Optional[int]
    supervisor_ready: bool
    boot: NotRequired[BootDocument]  # the same boot phases as /api/health; the header paints them
    supervisor_error: Optional[str]
    runtime_mode: str
    context_mode: str
    subagents_mode: Literal["off", "auto", "swarm"]  # Main stored choice
    subagents_availability: Literal["available", "disabled", "unsaved"]
    context_mode_auto_low: bool  # frozen compatibility field, always False (persistent auto-Low is retired)
    effort_range: EffortRange  # the owner's effort range (POST /api/owner/effort-range), the tolerant read
    main_route: ChatModelRoute  # Main itself, written by POST /api/owner/chat-model without project_id
    processing_tiers: ProcessingTiers  # Main's own tier and the default one; POST /api/owner/chat-processing
    safety_mode: str
    skills_repo_configured: bool
    github_token_configured: bool
    accounting: Dict[str, Any]  # physical-attempt ledger projection.
    # Multi-project sidebar feed: the pins and the 50 newest others by last_message_at (ProjectEntry rows).
    projects: list
    project_messages_revision: Optional[str]  # moves with any Project's last_message_at; null until history is dated
    project_listable_count: Optional[int]  # Projects the sidebar can list; null when unavailable or rooms unknown
    project_chat_ids: Optional[list]  # COMPLETE (uncapped) registered chat_ids, the live WS fan-out isolation SSOT; null: unknown, keep the previously observed set
    # Main's room revision (its standalone messages, the kind a Project's visible_revision counts); null while unreadable.
    main_visible_revision: NotRequired[Optional[int]]
    # Task->project bindings ({task_id: {project_id, chat_id}}; a Project's name is in the census rows and
    # ``projects``): a bound card hides the stray "turn into project" button and opens its project. Null only when
    # ``?bindings_since`` named the current opaque ``task_bindings_revision`` (the map's digest; "" after a failed read).
    task_bindings: Optional[dict]
    task_bindings_revision: str  # The map digest, empty after a read failure; an empty value is not a cache hit.
    task_bindings_unchanged: bool  # The request named the current bindings_since revision; keep the held map.
    active_direct_turns: NotRequired[List[ActiveDirectTurn]]
    # Combined activity snapshot (direct, root managed and live inline Presence turns). Additive beside
    # active_direct_turns, which stays unchanged for compatibility; new clients hydrate from this field.
    active_chat_activities: NotRequired[List[ActiveChatActivity]]
    active_chat_activities_complete: NotRequired[bool]
    current_questions: NotRequired[List[Dict[str, Any]]]  # canonical forms independent of history window; Project and Main unanswered questions alike, by identity
    current_questions_complete: NotRequired[bool]  # False retains previously observed questions its gaps name.
    current_questions_gaps: NotRequired[List[Dict[str, Any]]]  # typed source gaps, no read-side repair; each gives reason and, where known, task_id or quiz_id
    main_attention: NotRequired[List[Dict[str, Any]]]  # live human recovery owners, including children; census stays root-only
    main_attention_complete: NotRequired[bool]  # False retains previously observed actions.
    main_attention_gaps: NotRequired[List[Dict[str, Any]]]  # Typed gaps in current action evidence.
    served_ui: NotRequired[ServedUiState]  # this generation's main interface; absent on older cores


from ouroboros.gateway.client_contracts import (  # noqa: F401 -- public re-exports
    AvailableSubagentsSettingsMeta, NanoPublicationSettingsMeta, SettingsMeta, SettingsNetworkMeta,
    SettingsPolicyAxis, SettingsReceipt, SettingsRestartState, SettingsTiming,
)


class SettingsSaveResponse(SettingsReceipt, total=False):  # `POST /api/settings` success; `restart_keys` apply only after a restart.
    status: str
    no_changes: bool
    restart_required: bool
    restart_keys: list[str]
    restart_state: SettingsRestartState
    immediate_changed: bool
    next_task_changed: bool
    warnings: list[str]
    ignored_keys: Dict[str, str]
    # True when the save landed while an agent task was already STARTED: that task keeps its start-time reviewers,
    # subagents, context and effort; models, accounts and keys reach work that follows them at its next call.
    agent_task_running: bool
    connection_references: list[Dict[str, str]]  # rows naming a connection that cannot serve them (they refuse): {setting, entry?, route, connection, reason}


class OwnerRuntimeModeResponse(SettingsReceipt):
    ok: bool
    runtime_mode: str
    restart_required: bool


class OwnerAutoGrantResponse(SettingsReceipt):
    ok: bool
    enabled: bool


class OwnerContextModeResponse(SettingsReceipt):
    ok: bool
    context_mode: str
    project_id: str
    cognition: ProjectCognition


class OwnerSafetyModeResponse(SettingsReceipt):
    ok: bool
    safety_mode: str  # full | light | off (v6.54.3)


class OwnerSkillPresenceRuntimeRequest(TypedDict):
    expected_state_fingerprint: str
    runtime_overrides: Dict[str, Any]  # The overrides are model_slot (main, light, or null) and inline_max_rounds (count or null).
    workspace_root: NotRequired[str]


class OwnerSkillPresenceRuntimeResponse(TypedDict):
    ok: bool
    skill: str
    presence_runtime: Dict[str, Any]


class SkillGrantResponse(TypedDict, total=False):
    """``POST /api/skills/{skill}/grants``; a refusal answers ``SkillActionRefusal``. A grant that
    turns the skill on may install packages: past the answer bound the body is
    ``{ok, pending: true, job_id, message}`` (follow the lifecycle queue)."""

    ok: bool
    pending: bool
    job_id: str
    message: str
    skill: str
    content_hash: str
    granted_keys: list[str]
    granted_permissions: list[str]
    turn_on_when_ready: str
    setup_error: str  # grants succeeded; the requested setup did not
    setup_code: str  # that setup's typed refusal code (as SkillActionRefusal.code), else "error"
    enabled: bool  # present with the owner's Enable, except a grants refusal over an unreadable enabled.json
    extension_action: Optional[str]  # null, as live_loaded, when no reconcile answered (an Enable refused first)
    extension_reason: Optional[str]
    load_error: Optional[str]
    grants: Dict[str, Any]
    process: str
    server_reconcile: str
    live_loaded: Optional[bool]
    retained_companions: Dict[str, str]  # as SkillToggleResponse: registered names awaiting confirmed exit
    error: str


class SkillDeleteResponse(TypedDict, total=False):  # `POST /api/skills/{skill}/delete`; refusals use SkillActionRefusal (e.g. code=revision_changed).
    ok: bool
    pending: bool  # queued behind a long job past the answer bound: {ok, pending, job_id, message}
    job_id: str
    message: str
    skill: str
    source: str
    content_hash: str
    deleted_payload_root: str
    deleted_state: bool
    extension_action: str
    extension_reason: str
    error: str



class DesktopAutostartResponse(TypedDict):  # GET/POST /api/desktop/autostart (OS registration) and /api/desktop/background (unavailable|off|on)
    state: Literal["unavailable", "off", "on", "other_copy", "disabled_by_os"]
    reason: NotRequired[str]  # present only when unavailable


class GitLogResponse(TypedDict):
    commits: list[Dict[str, Any]]
    # ``list_versions`` emits tag rows: {tag, date, sha (peeled commit), message}.
    tags: list[Dict[str, Any]]
    branch: str
    sha: str


class EvolutionRefresh(TypedDict):  # the background re-measurement behind GET /api/evolution-data
    state: Literal["idle", "running", "failed"]
    started_at: str
    finished_at: str
    error: str  # "" unless failed; the saved copy stays
    unmeasured: int  # versions of the saved copy whose commit could not be read (retried by force)
    unmeasured_error: str  # the first such reason, "" when every version was measured


class EvolutionDataResponse(TypedDict):  # answered at once from the saved copy (ouroboros/evolution_history.py)
    # {tag, aliases, version, date, commit, level, milestone, code_lines, bible_bytes, system_bytes (None = not measured),
    #  release: {source: readme|tag|"", title, kind, heading, body}}, oldest first
    versions: list[Dict[str, Any]]
    saved_at: str  # "" = no saved copy yet
    refresh: EvolutionRefresh
    # identity / scratchpad: {at_versions: [characters | None per version], first_at, latest: {at, chars} | None}
    install: Dict[str, Any]
    # Published before the version history above and kept for older clients: one point per tag name {tag, date, code_lines, bible_kb,
    # system_kb, identity_kb, scratchpad_kb, memory_kb}; unmeasured values and the three memory fields are null.
    points: list[Dict[str, Any]]
    cached: bool  # true when the answer comes from a saved copy (always, once one exists)
    checkpoints: NotRequired[list[Dict[str, Any]]]
    generated_at: str


class UploadResponse(TypedDict):
    ok: bool
    filename: str
    display_name: str
    path: str
    size: int
    sha256: NotRequired[str]
    mime: str  # the extension's type, as the model-input rail reads it
    view: NotRequired[ChatAttachmentView]  # the sender's own bubble renders exactly this (kind proven from bytes)


class SkillPublishPreflightResponse(TypedDict):
    """Safe selected-skill publication facts; never raw scanner output."""

    ok: bool
    skill: str  # Canonical selected-skill name.
    repository: str  # Canonical case-preserving owner/repo from the configured catalog.
    state: Literal["ready", "warnings", "needs_attention", "repairable", "hard_block"]
    publication_ready: bool
    task_start_allowed: bool
    snapshot_hash: str
    review: Dict[str, Any]  # {status, stale, profile}; a loaded review adds reviewed_content_hash, author_disposition and author_accepted
    scanner: Dict[str, Any]  # {status, engine, version, ruleset_sha256}; a scanner_error adds reason_code and repair_hint
    findings: list[Dict[str, Any]]
    omitted_count: int
    blocker_count: int
    warning_count: int
    audited_false_positive_count: int
    reason_code: str
    summary: str
    repair_hint: str


class ProviderTestRequest(TypedDict):
    provider_id: str
    overrides: NotRequired[Dict[str, str]]  # flat keys; with ``connection``, that row's field names
    connection: NotRequired[str]  # a named connection of ``provider_id``; absent = the built-in one
    # The built-in compatible pair under test; absent = the historical resolution (its Clear blanks both addresses).
    api_key_setting: NotRequired[Literal["OPENAI_COMPATIBLE_API_KEY", "OPENAI_API_KEY"]]


class ProviderTestResponse(TypedDict):
    ok: bool
    error: NotRequired[str]
    caution: NotRequired[str]  # the request went to a server that named no program, whatever the outcome
    code: NotRequired[Literal["ollama_guard_consent", "ollama_guard_held", "ollama_guard_unavailable"]]  # a failed Test's withheld send: consent needed, an address hold, an Ollama to upgrade


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
    # v6.115.0: the run's owner-facing name. Supplied, it fills both name slots like a promoted chat turn; omitted,
    # admission derives a display-only name from the request's first line. `metadata.title` is refused (400).
    title: str
    chat_id: int
    depth: int
    session_id: str
    workspace_root: str
    workspace_mode: str
    # Trimmed and lowercased: forked | empty | shared. External workspaces allow only forked or empty.
    memory_mode: str  # Omitted or empty string defaults to forked for an external workspace, shared otherwise.
    project_id: str  # Per-project facts scope id (else derived from the workspace path).
    attachments: list[Dict[str, Any]]
    # Partial staging is the default (В25c, capinv-447): omitted/true stages the good
    # attachments and discloses rejected rows; explicit false keeps the old atomic admission.
    allow_partial_attachments: bool  # absent or true: admit with the attachments that staged; false: any rejection answers 422 attachment_admission_rejected
    acceptance_claims: list[Dict[str, Any]]  # Advisory Observable Acceptance Claims (`claim`/`surface`/`support`/`priority`).
    # v6.60.0: "" | "final_answer_line" — adapter-declared machine-extractable answer
    # protocol; flows into task_contract.answer_protocol and inherits to subagents.
    answer_protocol: str
    # true: task review Required reviews the whole final answer even without files (task_contract.review_answer).
    review_answer: bool
    allowed_resources: Dict[str, Any]
    resource_policy: Dict[str, Any]
    disabled_tools: list[str]  # Declarative tool-policy denylist: tool names withheld from the agent (independent of allowed_resources).
    executor_ref: ExecutorRef
    service_teardown: Literal["stop", "keep"]  # keep leaves services available to external verifiers; on Windows, a cancel or hard timeout still kills their task process tree.
    deadline_at: str
    timeout_sec: float
    timeout: float
    execution_timeout_sec: float  # excludes queue and confirmed pauses; 0 adds no limit; execution seconds, independent of the calendar timeout and deadline
    context: str
    expected_output: str
    constraints: str
    context_requires_self_body_docs: bool
    reasoning_effort: str  # optional explicit starting effort: an EFFORT_SCALE tier, checked by the handler; omitted uses the Task default; metadata.reasoning_effort is refused
    actor_id: str  # Top-level task actor/provenance id; metadata.actor_id is reserved.
    source: str  # Top-level task source/provenance label.
    metadata: Dict[str, Any]  # Arbitrary task metadata; executor_ref/workspace_executor keys are reserved.


class TaskCreateResponse(TypedDict, total=False):
    ok: bool
    task_id: str
    status: str
    reason_code: str
    error: str
    attachment_manifest: list[AttachmentManifestEntry]
    attachment_manifest_ref: Dict[str, Any]


class TaskListResponse(TypedDict, total=False):
    tasks: list[Dict[str, Any]]
    queue: Dict[str, Any]
    error: str


class HistoryRetentionProjection(TypedDict):
    """Storage placement facts, independent of task outcome or review authority."""

    status: Literal["pending", "problem", "complete"]
    pending_count: int
    problem_count: int
    promoted_ref_count: int
    promoted_source_handle_count: int
    problem_reasons: NotRequired[List[Dict[str, Any]]]  # Optional grouped {reason: str, count: int} rows.


class QuestionRoom(TypedDict, total=False):
    """Current display proof for one exact owner_quiz key; not publication or task context."""

    state: Required[Literal["known", "unknown"]]
    chat_id: int
    project_id: str
    also_project_ids: List[str]  # a private Main question: the Project that shows the same row


class TaskDetailResponse(TypedDict, total=False):
    """Open ``GET /api/tasks/{task_id}`` result: stored keys pass through beside typed projections."""

    artifacts: List[Dict[str, Any]]  # Open result rows; a nested file carries additive ``relpath``. measured: false identifies a stat-only listing, not a verified capture; size is bytes.
    # Per result directory: {name, files, size, excluded, available} for its ``?archive=`` ZIP.
    artifact_archives: Dict[str, Dict[str, Any]]  # Rows use one confined stat per member; excluded counts omissions. Offer the ZIP only if available.
    # Wrap up's saved-work document {name, request_id}: ``?wrap_up=<request_id>`` writes it on read; no size/digest.
    wrap_up_document: Dict[str, str]
    cost_breakdown: TaskCostBreakdown
    history_retention: HistoryRetentionProjection  # routine progress is detail-only; only a problem appears on the collapsed card
    model_waits: Dict[str, Any]
    question_rooms: Dict[str, QuestionRoom]  # keyed by the exact owner_quiz id
    # "pending" means "Cancelling…"; running/scheduled stays truthful. Absent when settled or unrequested.
    cancel_state: str
    # Pending intent's recorded reason (owner text or subtree cause); absent if none (GR2-11).
    cancel_reason: str
    # Soft finalize_then_cancel: "Finalizing…" plus hard escalation; absent on immediate intents (S3/Q1).
    stop_policy: str
    # Current hurry plus prior same-id attempts; detail-only, absent if never hurried (S3/HQ1).
    continuation_offer: ContinuationOffer
    owner_hurry: OwnerHurryProjection  # Current typed hurry projection, absent unless the task was hurried; task detail only, never chat.
    owner_hurry_history: list[OwnerHurryProjection]  # Prior attempts’ typed hurry records, archived separately from the current block.
    project_admission_hold: Dict[str, Any]  # while the queue snapshot lists a non-final row: {reason, detail, label} or {}; its wait for original Project/scope evidence
    # The same row's current stop ({} when none). Saved work that does not fit the model's window adds ``window``: {pause_id,
    # pause_generation, input_tokens, window_tokens, reply_floor_tokens, over_tokens, overview?} (TaskResumeRequest.context_choice).
    current_stop: Dict[str, Any]
    # Every adoption this task authorized, as boot settled them: {adoption_id, outcome, commit_sha, previous_sha, branch,
    # note, settled_at, revision, commits?, commits_complete?, commits_omitted?}; commits (newest first) only when adopted.
    body_adoptions: List[Dict[str, Any]]  # The bounded commits range is previous..candidate; commits_omitted counts the cut remainder.
    review_overrides: Dict[str, Any]  # a root's review trace: {count?, latest?, continued?, landed?, skills?} (review_overrides.py): FAIL/no-verdict reviews and the decisions to continue or land over them
    # Host switches after tool use: {from_model, to_model, status, selection?, work_observed?, reply_observed?}.
    # reply_observed=true confirms a reply; legacy rows may not say. Absent list means history unavailable.
    authoring_handovers: List[Dict[str, Any]]
    place_origin: Dict[str, Any]  # ChatOutbound.place_origin's shape, from the task's own admitted focus; absent elsewhere
    requested_executor_route: RequestedExecutorRoute  # a child's selection frozen at scheduling (ChatOutbound's); absent if none
    error: str


class ClaudexorLoginJobResponse(TypedDict, total=False):
    """Every login operation returns one required bare ControlSetupJob as job.
    Create metadata, input ok and snapshot cursor/sequence/deviceCode stay beside
    it. Snapshots pass through verbatim; attach_command/attach_shell require the
    proven packaged role. Optional fields are operation-scoped; never job.job."""

    job: Required[Dict[str, Any]]
    # snapshot-only (daemon envelope verbatim)
    cursor: str
    sequence: int
    deviceCode: Dict[str, Any]
    # create-only metadata
    job_id: str
    disclosure_native: bool
    setup_login_source: Literal[
        "per_harness", "setup_job_admission", "legacy_global_operation"
    ]
    # Only for a client_pty job with an id and confirmed setup_attach on the serving package.
    attach_command: str
    attach_shell: Literal["posix", "powershell"]
    # input-only compatibility bit
    ok: bool


class ClaudexorVendorCredentialDisposition(TypedDict):
    """What profile deletion did to a vendor-owned host-user credential."""

    owner: Literal["vendor"]
    state: Literal["left_unchanged"]
    scope: Literal["os_user"]


class ClaudexorCredentialProfileDeleteResponse(TypedDict):
    """Exact daemon receipt returned by credential-profile deletion."""

    profile: Dict[str, Any]
    removed: bool
    credentialCleanup: Literal["config_dir_removed", "secret_deleted", "none"]
    cleanupWarning: NotRequired[str]
    vendorCredentialDisposition: NotRequired[ClaudexorVendorCredentialDisposition]


class TaskEventCursor(TypedDict):
    v: Literal[2]
    seq: int
    view: str
    positions: Dict[str, Dict[str, int]]


class TaskEventsRequest(TypedDict):
    v: Literal[2]
    wait: NotRequired[int]
    cursor: NotRequired[Optional[TaskEventCursor]]


class TaskEvent(TypedDict, total=False):
    seq: int
    source: str
    line: int
    ts: str
    type: str
    task_id: str
    root: str
    data: Dict[str, Any]
    event_id: str
    cursor: TaskEventCursor
    reason: str
    error: str


class TaskPauseRequest(TypedDict):
    """Tree Pause; reuse request_id on retry. ``actor`` (owner|ouroboros) and ``reason`` declare who and why."""
    request_id: str
    actor: NotRequired[str]
    reason: NotRequired[str]


class TaskPauseResponse(TypedDict, total=False):
    ok: bool
    task_id: str
    root_task_id: str
    fence_id: str
    state: Literal["requested", "paused", "released"]
    duplicate: bool
    members: List[str]
    snapshot_persisted: bool
    latch_pending: bool
    error: str
    reason_code: str


class TaskContinueRequest(TypedDict):
    """Retry one action, including after an unconfirmed admission/reload."""
    action_nonce: str


class TaskContinueResponse(TypedDict, total=False):
    ok: bool
    task_id: str
    predecessor_task_id: str
    successor_task_id: str
    replay: bool
    recovered: bool
    held: bool
    status: str
    blockers: List[Dict[str, Any]]
    error: str
    reason_code: str
    cause: str
    gaps: List[str]
    unconfirmed: bool
    state: Literal["bound", "admitted"]
    action_nonce: str
    membership_gaps: Dict[str, Any]  # unreadable rows nothing ties to the predecessor, not waited for: {rows: [{task_id, reason}], count}


class ContinuationOffer(TypedDict, total=False):
    """Bound is retryable using action_nonce; only admitted is a successor pointer."""
    eligible: bool
    refusal: str
    cause: str
    successor_task_id: str
    state: Literal["bound", "admitted"]
    action_nonce: str


class TaskCancelResponse(TypedDict, total=False):
    ok: bool
    task_id: str
    # Cascade echoes subtree scope; immediate success follows teardown.
    cascade: bool  # Can also echo already-recorded cascade scope; a single-task envelope is unchanged.
    # A Stop's 202 keeps its recorded intent pending and echoes the EFFECTIVE policy, never softening an immediate
    # intent: a soft stop, or a teardown not yet confirmed (request_id, reason_code "teardown_unconfirmed"); its 200
    # omits them. A Close (gateway/task_close.py) has no stop_policy: 202 only while owed; 200 closed or "result_kept".
    cancel_state: str
    stop_policy: str
    request_id: str  # Custody id on immediate Stop 202, Close 200/202, or recorded unconfirmed/corrupt Close 503 (distinct from stop_action_id).
    # Stop 202: "teardown_unconfirmed". Close (gateway/task_close.py): 200 "result_kept"; 409
    # "expired_park_changed" (this request made no change; a retry after Stop replacement establishes
    # only that the retry did nothing), "expired_park_partly_closed", or "not_found"; 503
    # "cancel_intent_projection_corrupt" (an unconfirmed Close 503, including an in-flight call replaced
    # by Stop, omits reason_code).
    reason_code: str
    error: str


class TaskHurryRequest(TypedDict):
    """Text-free owner hurry (HQ1), POST /api/tasks/{task_id}/hurry.
    Only stable request_id is accepted, reused on retry. No chat side effect;
    durable facts are the mailbox control, owner_hurry projection and one event.
    """
    request_id: str


class OwnerHurryProjection(TypedDict, total=False):
    """Task-detail owner_hurry, never chat: requested|applied|not_applied_before_terminal.
    Effects map host rails; same-ID requeues retain archived_at/archived_reason."""

    attempt_key: int
    request_id: str
    requested_by: str
    requested_at: str
    reason: str
    state: str
    effects: Dict[str, str]
    applied_at: str
    reconciled_at: str
    archived_at: str
    archived_reason: str


class TaskHurryResponse(TypedDict, total=False):
    """Acknowledgement of the typed task-local acceleration control.

    ``duplicate=True`` is the idempotent shape: the same ``request_id`` on the
    live attempt (or a different id collapsing onto the one armed latch)
    returns the existing acknowledgement without a second control."""

    ok: bool
    task_id: str
    request_id: str
    state: str
    attempt_key: int
    duplicate: bool
    error: str


class LogTailResponse(TypedDict, total=False):
    name: str
    entries: list[Dict[str, Any]]
    error: str


class OnboardingCompleteRequest(TypedDict, total=False):  # POST /api/onboarding/subagents/preview accepts this same draft and declarations and returns an editable actor list without saving it.
    """Open setup-settings payload plus subscription declarations, actor draft and starter choices. None is
    authority: subscriptionsConnected requests a live account read; the server re-proves fresh install."""

    subscriptionsConnected: bool
    skipSubscriptionPresets: bool
    OUROBOROS_SUBAGENTS: Dict[str, Any]
    allow_empty_review_pool: bool  # confirms saving agents that leave no review row
    starter_skills: List[str]  # catalog ids installed once after completion; absent: the default starter list
    skill_source_trust: Dict[str, str]  # {hub row id: trusted|review}; marks those rows confirmed
    attempt_id: str  # client-made per save (16-128 of A-Z a-z 0-9 _ -); read back by GET with ?attempt=
    consciousness: bool  # setup's Consciousness switch: true turns it on (owner's start); false and absent change nothing


class OnboardingSubagentsPreviewResponse(TypedDict):
    """Read-only canonical actor draft returned to Settings/onboarding."""

    ok: bool
    model_settings: Dict[str, Any]  # the model setting keys the draft would save
    available_subagents: Dict[str, Any]
    known_spend_gaps: List[str]  # what the draft's limits cannot count, as GET /api/settings names it after a save
    partly_counted_spend: List[Dict[str, str]]  # what they count only under a price condition, the same way: {name, condition}, as SettingsMeta
    source: str
    diagnostics: list[Dict[str, Any]]


class OnboardingPresetProjection(TypedDict):
    """What the install-time agent preset did, on the success envelope.

    ``applied=False`` with a ``reason`` is the normal shape for an install that
    connected no subscription, opted out, or is no longer in first-run
    onboarding — absence is reported as absence, never as an empty success."""

    applied: bool
    # Open string ABI. Emitted values include not_requested, not_install_time,
    # skipped_by_owner, configured_by_owner, and applied.
    reason: str
    harnesses: list[str]
    receipt: Dict[str, Any]  # per-seat resolution record; {} when nothing was applied.


class OnboardingCompleteResponse(TypedDict):
    """Settings, next-boot mode, safety default and completion persist atomically; preset and marker if applied."""

    ok: bool
    status: str
    runtime_mode: str
    restart_required: bool
    preset: OnboardingPresetProjection


class OnboardingAttemptResponse(TypedDict):
    """GET /api/onboarding/complete?attempt=: one completion attempt read back, never a readiness probe.
    ``saved``: true while the settings document holds this attempt's write, false when this server
    refused it before writing, null when unconfirmed (still running, never arrived or replaced since).
    ``receipt`` is the attempt's own success envelope once completion finished; the other fields repeat
    its recorded failure. Only a receipt completes; anything else keeps the draft and never retries."""

    saved: Optional[bool]
    receipt: NotRequired[OnboardingCompleteResponse]
    error: NotRequired[str]
    code: NotRequired[str]
    post_commit_failed: NotRequired[str]


class SettingsPostCommitFailureResponse(SettingsReceipt):
    """500 after settings bytes landed; post_commit_failed names the later step.
    Retry that step, never re-save. Shared by settings and onboarding writes."""

    error: str
    status: str  # saved_with_post_commit_error.
    saved: bool  # always true.
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
from ouroboros.gateway.endpoint_index import HTTP_ENDPOINTS
from ouroboros.gateway.client_contracts import (  # noqa: E402,F401
    ChatModelChoice, ChatModelRoute, FileBrowserListResponse, ServedUiState, SettingsPolicyState,
)

WS_MESSAGE_TYPES: tuple[str, ...] = (
    "conversation_finished", "connection_profile", "chat", "command",
    "panic_accepted",  # immediate acceptance only, not proof of process exit
    "photo", "video", "document", "links",
    "quiz", "quiz_state",
    "card", "card_state",  # gateway/card_contracts.py frames
    "typing",
    "log",
    "heartbeat",
    "extension_lifecycle",
    "extension_health_changed",
    "message_annotation",
    "projects_changed",
    "ui_preferences_changed",
    "task_named",
    "update_status_ready",
    "update_progress_changed",
    "ui_language_changed", "ui_i18n_updated",  # gateway/ui_i18n_contracts.py frames
)


__all__ = [
    "CostPresentation",
    "ChatInbound", "TaskConstraintInbound", "CommandInbound", "PanicAcceptedOutbound", "ConnectionProfileOutbound", "ExtensionInbound",
    "TransportMetadata",
    "ChatOutbound",
    "PhotoOutbound", "VideoOutbound", "DocumentOutbound",
    "LinkAction", "LinksOutbound",
    "QuizOption", "QuizOutbound", "QuizStateOutbound",
    "CardOutbound", "CardStateOutbound",
    "DecisionRequest", "DecisionResponse",
    "SecretAllow", "SecretAllowRequest", "SecretConsumer", "SecretDecisionRequest", "SecretDecisionResponse",
    "SecretEvent", "SecretKeyState", "SecretOutcome", "SecretRequestField", "SecretRequestMeta", "SecretSkillFacts",
    "TypingOutbound", "LogOutbound", "HeartbeatOutbound", "ExtensionLifecycleOutbound",
    "ProjectsChangedOutbound", "UiPreferencesChangedOutbound", "MessageAnnotationOutbound",
    "UpdateMergePlan", "UpdatePreflightRequest", "UpdatePreflightResponse", "UpdateApplyRequest",
    "UpdateApplySuccessResponse", "UpdateApplyErrorResponse",
    "UpdateDescriptionRequest", "UpdateDescriptionResponse", "UpdateDescriptionLatestResponse",
    "UpdateDescriptionErrorResponse",
    "UpdateStatusReadyOutbound", "UpdateProgressChangedOutbound",
    "ProjectCreateRequest", "ProjectEntry", "ProjectClosure", "ProjectDeleteResponse", "ProjectCognition",
    "FsDirsEntry", "FsDirsResponse",
    "TaskNamedOutbound",
    "ErrorResponse", "StatusResponse", "HealthResponse", "StateResponse",
    "BootStep", "BootJob", "BootDocument",
    "ActiveDirectTurn", "ActiveChatActivity",
    "EvolutionStateSnapshot",
    "SettingsNetworkMeta",
    "AvailableSubagentsSettingsMeta",
    "SettingsPolicyAxis", "SettingsPolicyState", "SettingsMeta",
    "SettingsSaveResponse",
    "OwnerRuntimeModeResponse", "OwnerAutoGrantResponse", "OwnerContextModeResponse", "OwnerSafetyModeResponse",
    "OwnerSkillPresenceRuntimeRequest", "OwnerSkillPresenceRuntimeResponse",
    "OnboardingCompleteRequest", "OnboardingCompleteResponse", "OnboardingAttemptResponse",
    "OnboardingSubagentsPreviewResponse",
    "OnboardingPresetFailureResponse",
    "OnboardingPresetProjection",
    "SettingsPostCommitFailureResponse",
    "SkillGrantResponse", "SkillDeleteResponse",
    "UiContribution", "UiContributionsResponse", "UiPreferencesResponse", "DesktopAutostartResponse",
    "GitLogResponse",
    "EvolutionDataResponse",
    "EvolutionRefresh",
    "ScheduledTasksResponse", "ScheduleUpsertResponse", "ScheduleDeleteResponse", "ScheduleActionResponse",
    "UploadResponse",
    "ExtensionsIndexResponse", "ExtensionLiveSnapshot", "WidgetTab",
    "WidgetsResponse", "WidgetFaultRequest", "WidgetFaultResponse",
    "WidgetAssetLeaseRequest", "WidgetAssetTokenRequest", "WidgetAssetLeaseResponse", "WidgetAssetRevokeResponse",
    "SkillPublishPreflightResponse",
    "SkillLifecycleQueueResponse",
    "MarketplaceSearchResponse",
    "MarketplaceInstalledResponse",
    "LocalModelStatusResponse",
    "McpStatusResponse",
    "ModelCatalogResponse", "OpenAICompatibleModelsResponse",
    "ProviderTestRequest", "ProviderTestResponse",
    "FileBrowserListResponse",
    "ChatHistoryResponse",
    "AttachmentManifestEntry",
    "ChatAttachmentInbound",
    "ChatAttachmentView", "QuizAnswerFile",
    "ExecutorRef",
    "TaskCreateRequest", "TaskCreateResponse",
    "TaskListResponse", "TaskCostBreakdown", "TaskDetailResponse", "QuestionRoom",
    "ClaudexorReadState",
    "ClaudexorStatusReads", "ClaudexorPassiveReadError", "ClaudexorQuotaResponse",
    "ClaudexorStatusResponse",
    "ClaudexorLoginJobResponse",
    "ClaudexorLoginJobProblem",
    "ClaudexorVendorCredentialDisposition",
    "ClaudexorCredentialProfileDeleteResponse",
    "TaskEvent", "TaskEventCursor",
    "TaskEventsRequest", "TaskReviewHistoryResponse",
    "TaskCancelResponse", "TaskCancelRequest",
    "TaskPauseRequest", "TaskPauseResponse",
    "TaskContinueRequest", "TaskContinueResponse", "ContinuationOffer",
    "TaskHurryRequest", "TaskHurryResponse",
    "OwnerHurryProjection",
    "EffortRange", "OwnerEffortRangeResponse", "OwnerSubagentsModeResponse", "LogTailResponse",
    "StorageCount", "StorageCounting", "StorageError", "StoragePart", "StoragePlace", "StorageResponse",
    "StorageRest", "StorageVolume", "StorageRemover",
    "StorageActiveProject", "StorageDeletedFolder", "StorageDeletedProject", "StorageDeletedProjects", "StorageFolderPart",
    "StorageFree", "StorageFreeEntry", "StorageFreeGroup", "StorageFreeItem", "StorageFreeItemsResponse", "StorageFreeKept",
    "StorageFreePool", "StorageFreeRequest", "StorageFreeRun", "OwnerKeepRequest", "OwnerKeepResponse",
    "KeptService", "KeptServicesResponse", "KeptServiceStopRequest", "KeptServiceStopResponse",
    "HTTP_ENDPOINTS",
    "WS_MESSAGE_TYPES",
]
