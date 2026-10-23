"""Gateway HTTP/WebSocket contracts (v1): descriptive, not runtime validation.

Keep discriminating ``type`` keys required; optional fields use ``NotRequired``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from ouroboros.cost_projection import CostPresentation
from ouroboros.gateway.reference_contracts import ChatReferenceFields
from ouroboros.storage import (  # noqa: F401 -- public re-exports
    StorageCount, StorageCounting, StorageError, StoragePart, StoragePlace, StorageResponse, StorageRest,
    StorageVolume,
)
from ouroboros.storage_free import (  # noqa: F401 -- public re-exports
    StorageFree, StorageFreeGroup, StorageFreeItem, StorageFreeKept, StorageFreeRequest, StorageFreeRun,
)
from ouroboros.gateway.claudexor_contracts import (  # noqa: F401 -- public re-exports
    ClaudexorReadState, ClaudexorStatusReads, ClaudexorPassiveReadError, ClaudexorQuotaResponse,
    ClaudexorStatusResponse,
)
from ouroboros.gateway.work_contracts import WorkStateFields
from ouroboros.gateway.history_contracts import ChatHistoryResponse, PresenceContact  # noqa: F401 -- public re-exports
from ouroboros.gateway.health_contracts import BootDocument, BootJob, BootStep, HealthResponse  # noqa: F401 -- public re-exports
from ouroboros.gateway.attachment_contracts import (  # noqa: F401 -- public re-exports
    AttachmentManifestEntry, ChatAttachmentInbound, ChatAttachmentView,
)
from ouroboros.gateway.contributions import UiContribution, UiContributionsResponse, UiPreferencesResponse
from ouroboros.gateway.widgets import ExtensionLiveSnapshot, WidgetFaultRequest, WidgetFaultResponse, WidgetTab, WidgetsResponse
from ouroboros.gateway.decision_contracts import DecisionRequest, DecisionResponse  # noqa: F401 -- public re-exports
from ouroboros.gateway.card_contracts import CardOutbound, CardStateOutbound  # noqa: F401 -- public re-exports
from ouroboros.gateway.update_contracts import (  # noqa: F401 -- public re-exports
    UpdateApplyErrorResponse, UpdateApplyRequest, UpdateApplySuccessResponse, UpdateMergePlan,
    UpdatePreflightRequest, UpdatePreflightResponse, UpdateProgressChangedOutbound, UpdateStatusReadyOutbound,
)
from ouroboros.gateway.management_contracts import (  # noqa: F401 -- public re-exports
    ExtensionsIndexResponse, LocalModelStatusResponse, MarketplaceInstalledResponse, MarketplaceSearchResponse,
    McpStatusResponse, ModelCatalogResponse, SkillLifecycleQueueResponse,
)
from ouroboros.gateway.schedule_contracts import (  # noqa: F401 -- public re-exports
    ScheduleActionResponse,
    ScheduledTasksResponse,
    ScheduleDeleteResponse,
    ScheduleUpsertResponse,
)

try:  # Python 3.11+
    from typing import Literal, NotRequired, Required, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import Literal, NotRequired, Required, TypedDict  # type: ignore[assignment]


class ChatInbound(TypedDict):
    """Inbound WS chat message. ``type`` and ``content`` are required."""

    type: Literal["chat"]
    content: str
    sender_session_id: NotRequired[str]
    client_message_id: NotRequired[str]
    force_plan: NotRequired[bool]
    attachments: NotRequired[list]  # list[ChatAttachmentInbound] (additive, v6.26.0)
    # Multi-project chat routing: the owner stays user_id 1; chat_id selects the thread, project_id scopes memory.
    chat_id: NotRequired[int]
    project_id: NotRequired[str]
    reply_to: NotRequired[str]  # Host resolves the selected row/card at acceptance.
    client_surface: NotRequired[dict]


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
    client_surface: NotRequired[dict]


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


class ChatOutbound(ChatReferenceFields):
    """Outbound WS chat frame."""

    presence_contact: NotRequired[PresenceContact]
    type: Literal["chat"]
    role: Literal["user", "assistant", "system"]
    content: str
    ts: str
    ingress_accepted: NotRequired[bool]  # Canonical inbound row saved; not processing/start proof.
    ingress_dispatched: NotRequired[bool]  # This live host process accepted the row and entered its dispatch.
    ingress_pending: NotRequired[bool]  # This live host process accepted the row and has entered or refused neither yet.
    attachments: NotRequired[List[ChatAttachmentView]]  # owner message's attachments (same views as history)
    text_placeholder: NotRequired[bool]  # owner row whose text the host wrote (no words were sent); shown as no caption
    markdown: NotRequired[bool]
    is_progress: NotRequired[bool]
    task_id: NotRequired[str]
    origin_message_ref: NotRequired[Dict[str, Any]]
    # X3: a repair receipt whose managed task id the router mints only at promotion (typed truth, no invented id).
    task_id_pending: NotRequired[bool]
    # "finalizing": early answer during synthesis, NOT terminal; task_done settles the card/turn.
    task_phase: NotRequired[str]
    # Direct/ephemeral completed/failed/cancelled/rejected_duplicate settles activity, not early answers.
    task_terminal_status: NotRequired[str]
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
    materials: NotRequired[List[Dict[str, Any]]]  # validated existing task artifact references
    question_delivery: NotRequired[Dict[str, Any]]  # canonical kind/state; pending stays current
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
    max_active_subagents: NotRequired[int]
    queued_behind_active_cap: NotRequired[bool]
    required_capabilities: NotRequired[list[str]]
    write_surface: NotRequired[str]
    model_lane: NotRequired[str]
    requested_model_lane: NotRequired[str]
    effective_model_lane: NotRequired[str]
    # Opaque delegated route stamped once by resolve_subagent_dispatch; absent/empty = native/no chip. Dispatch
    # intent is not a landing receipt: a reduction belongs in capability_delta, never rewrites this route.
    executor_route: NotRequired[str]
    # Recorded coordinator facts on live/history frames; absence carries no billing or provenance claim.
    parent_cognitive_route: NotRequired[Dict[str, Any]]
    coordinator_route: NotRequired[str]
    use_local_model: NotRequired[bool]
    coordinator_billing: NotRequired[str]
    coordinator_provenance: NotRequired[str]
    # The effort decision at dispatch (a session row: its leaf's): level, the parent's request, source
    # auto|pin|model_name|cyber. Empty/absent = unknown, no chip.
    effort_level: NotRequired[str]
    effort_requested: NotRequired[str]
    effort_source: NotRequired[str]
    # Latest observed progress actor, NOT terminal evidence or current liveness. Own task_id/task_attempt/run_id/
    # attempt_id, harness_id, phase, revision; optional model has explicit model_source (requested or observed).
    executor_observation: NotRequired[Dict[str, Any]]
    # Host observation: v/task_id/run_id/after_seq/through_seq, source (run_events|timeline_window, read_through/ref/
    # provisional), typed parts/technical/gaps/omitted/latest_message; (run_id, seq) is identity, cuts count Unicode
    # code points and cuts_truncated/seqs_truncated disclose preview loss; source.ref = redacted JSONL via confined
    # task-file ?source= (503 if unsupported). Never narration or terminal evidence; full shape in api_types.js.
    delegated_activity: NotRequired[Dict[str, Any]]
    # Terminal evidence from subagents.envelope_from_task: run started/settled/succeeded counts, failure states,
    # disclosed spend/estimated flag, engine models, nanny_nudge_recorded, delegate_start_attempted. Absence = no
    # evidence yet, never native execution.
    execution_evidence: NotRequired[Dict[str, Any]]
    # Custody fact beside executor_route: "harness_used" | "harness_attempted" | "native_only". Terminal only;
    # absent = no substrate claim.
    actual_substrate: NotRequired[str]
    model: NotRequired[str]
    task_group_id: NotRequired[str]
    task_event: NotRequired[str]
    status: NotRequired[str]
    # Host-attested marker (P5), stamped by the supervisor's delivery seam ONLY for a task POST /api/tasks/{id}/cancel
    # will actually stop — a lineage-resolved pooled ROOT or the live in-process direct-chat turn (the endpoint's own
    # ownership reader, supervisor.workers.direct_chat_turn); never a subagent or ephemeral decision frame. Gates "Cancel run".
    cancelable: NotRequired[bool]
    _is_direct_chat: NotRequired[bool]  # lane fact stamped on a direct turn's own frames
    narration: NotRequired[bool]  # progress VOICE: the model's own round narration (true) vs a host note (false); absent = legacy
    spoken: NotRequired[Dict[str, Any]]  # call_id, item_id, played_ms, status: played/generated/interrupted
    initiator: NotRequired[str]  # origin label: "consciousness" on a wake-up's frames/rows (and its roots); absent on an owner's turn
    # Monetary projections are nullable when the physical-attempt ledger cannot be read: ``None`` is distinct from a
    # confirmed $0. The honest names (accounted upper bounds, not settled receipts), the only outbound spellings since
    # ABI 7.0 dropped the ``cost_usd[_with_children]`` aliases; cost_projection.py is the one author.
    accounted_upper_bound_usd: NotRequired[Optional[float]]
    accounted_upper_bound_usd_with_children: NotRequired[Optional[float]]
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
    # Typed message kind; role alone selects authorship (DESIGN: Chat authorship and System rows).
    system_type: NotRequired[str]
    # Host placement of a task's System row: timeline or Reviews; absent = ordinary. card_row_id is stable.
    card_row: NotRequired[Literal["timeline", "reviews"]]
    card_row_id: NotRequired[str]
    card_row_revision: NotRequired[int]  # canonical source order, independent of delivery timestamp
    terminal_receipt: NotRequired[Literal["retained_work"]]  # a placed row typed as a terminal receipt, not progress
    late_evidence: NotRequired[Dict[str, Any]]  # late review identity/revision + exact applied source; no transcript
    # Event-time human presentation; raw task/project ids remain machine keys.
    target_label: NotRequired[str]
    project_id: NotRequired[str]
    project_name: NotRequired[str]
    task_name: NotRequired[str]
    handoff_id: NotRequired[str]  # immutable origin/destination receipt identity
    terminal_time: NotRequired[Dict[str, Any]]  # host-owned occurrence, separate from publication ts
    completion_answer: NotRequired[str]  # a Project root's model-authored final answer, mirrored into Main (DESIGN)
    completion_reason: NotRequired[str]  # one recorded nonclean-outcome fact beside the answer
    completion_notice: NotRequired[Dict[str, Any]]  # transport view: neutral/detail/phase/salvage; Main adds cause and undescribed codes
    delivery_id: NotRequired[str]  # stable logical publication identity across transport retries
    set_at: NotRequired[str]  # a `reminder` row's provenance, live frame only (its text carries the signature): when written
    scheduled_for: NotRequired[str]  # the due point it was written for
    delivered_at: NotRequired[str]  # when the host showed it (later than due after downtime)
    chat_id: NotRequired[int]  # present on some transport re-broadcast paths
    # Server-stamped when chat_id is a reserved Project thread: Main never adopts it, even before the browser learns it.
    project_thread: NotRequired[bool]


class PhotoOutbound(ChatReferenceFields):
    """Outbound WS photo frame."""

    type: Literal["photo"]
    role: Literal["user", "assistant"]
    image_base64: str
    mime: str
    ts: str
    caption: NotRequired[str]
    # Durable task-artifact URL for the stored media, replayed by chat history
    # (the live frame carries the bytes inline instead).
    download_url: NotRequired[str]
    # Second address for the SAME bytes on the long-shipped
    # /api/files/download route, present only when the stored file resolves
    # inside the current file-browser root. Packaged desktop launchers gate
    # their file bridge to a URL allowlist that predates the artifact route,
    # so the browser uses download_url and the host bridge prefers this one.
    download_url_compat: NotRequired[str]
    content: NotRequired[str]
    source: NotRequired[str]
    sender_label: NotRequired[str]
    sender_session_id: NotRequired[str]
    client_message_id: NotRequired[str]
    transport: NotRequired[TransportMetadata]
    chat_id: NotRequired[int]
    task_id: NotRequired[str]
    # Server-stamped when chat_id is a reserved Project thread: Main never adopts it, even before the browser learns it.
    project_thread: NotRequired[bool]


class VideoOutbound(ChatReferenceFields):
    """Outbound WS video frame."""

    type: Literal["video"]
    role: Literal["user", "assistant"]
    video_base64: str
    mime: str
    ts: str
    caption: NotRequired[str]
    # Durable task-artifact URL for the stored media, replayed by chat history
    # (the live frame carries the bytes inline instead).
    download_url: NotRequired[str]
    # Same dual-address contract as PhotoOutbound.download_url_compat above.
    download_url_compat: NotRequired[str]
    content: NotRequired[str]
    source: NotRequired[str]
    sender_label: NotRequired[str]
    sender_session_id: NotRequired[str]
    client_message_id: NotRequired[str]
    transport: NotRequired[TransportMetadata]
    chat_id: NotRequired[int]
    task_id: NotRequired[str]
    # Same Project-thread stamp contract as PhotoOutbound.project_thread.
    project_thread: NotRequired[bool]


class DocumentOutbound(ChatReferenceFields):
    """Outbound WS document/file frame."""

    type: Literal["document"]
    role: Literal["user", "assistant"]
    file_base64: str
    mime: str
    filename: str
    ts: str
    caption: NotRequired[str]
    # Canonical captured-file URL; replay never needs to persist inline bytes.
    download_url: NotRequired[str]
    download_url_compat: NotRequired[str]
    file_ref: NotRequired[Dict[str, Any]]
    content: NotRequired[str]
    source: NotRequired[str]
    sender_label: NotRequired[str]
    sender_session_id: NotRequired[str]
    client_message_id: NotRequired[str]
    transport: NotRequired[TransportMetadata]
    chat_id: NotRequired[int]
    task_id: NotRequired[str]
    size_bytes: NotRequired[int]
    # Server-stamped when chat_id is a reserved Project thread: Main never adopts it, even before the browser learns it.
    project_thread: NotRequired[bool]


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
    project_thread: NotRequired[bool]
    transport: NotRequired[TransportMetadata]


class QuizOption(TypedDict):
    """One selectable option on an owner quiz card."""

    label: str
    detail: NotRequired[str]
    recommended: NotRequired[bool]


class QuizOutbound(ChatReferenceFields):
    """Outbound owner quiz card: a typed question with option buttons.

    Optional questions continue under ``assumption``; ``wait_for_answer`` marks
    required waiting without an implied answer. ``state`` carries lifecycle.
    Replay merges the stored ``answered_index`` and verbatim ``comment``; a
    comment without an index is the owner's whole answer, not an option choice."""

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
    host_facts: NotRequired[str]  # host-written, never the model's: asking task, run start, owner's last message
    materials: NotRequired[List[Dict[str, Any]]]
    question_delivery: NotRequired[Dict[str, Any]]
    closed_reason: NotRequired[str]
    closed_at: NotRequired[str]
    closed_by_task_id: NotRequired[str]
    closed_actor: NotRequired[str]
    closed_request_id: NotRequired[str]
    chat_id: NotRequired[int]
    task_id: NotRequired[str]
    project_thread: NotRequired[bool]
    transport: NotRequired[TransportMetadata]


class QuizStateOutbound(TypedDict):
    """Outbound WS lifecycle update for an already-rendered quiz card.

    A separate discriminator (not a second ``quiz`` frame): the display path dedupes
    quiz frames by ``quiz:{quiz_id}:{ts}``, so a state change must never look like a
    new card. ``answered_index`` rides only with the ``answered`` state.
    """

    type: Literal["quiz_state"]
    question_delivery: NotRequired[Dict[str, Any]]
    quiz_id: str
    task_id: str
    state: str
    ts: str
    answered_index: NotRequired[int]
    wait_for_answer: NotRequired[bool]  # additive: False once a bounded wait closed (the card stays open)
    # #471: the owner's recorded free-text answer rides the live frame (absent
    # when empty) so the open card renders `Owner's answer:` as replay does.
    comment: NotRequired[str]
    chat_id: NotRequired[int]
    closed_reason: NotRequired[str]
    closed_at: NotRequired[str]
    closed_by_task_id: NotRequired[str]
    closed_actor: NotRequired[str]
    closed_request_id: NotRequired[str]


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
    # Stamped for a registry-tracked turn ("direct_chat") or a RUNNING queue
    # root ("managed_task"); empty for children. No in-repo client reads it
    # (wire compatibility): only the census inserts into the header live-set.
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


class ExtensionLifecycleOutbound(TypedDict):
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


class MessageAnnotationOutbound(TypedDict):
    """Bubble-free presentation update for one owner message; ``cause``: the host's sentence for a refused act."""

    type: Literal["message_annotation"]
    annotation_type: Literal["routing_ack"]
    client_message_id: str
    action: str
    status: str
    suppress_bubble: bool
    chat_id: NotRequired[int]
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
    cause: NotRequired[str]
    reasoning_effort: NotRequired[str]  # #1539: the explicit start a New task picked from this card requests
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


class ProjectEntry(TypedDict, total=False):
    """A registry project row. ``provenance`` (attached|cloned|genesis|none) and ``clone_url``
    are historical facts; operational git data is read live from ``.git``. List rows are a
    compact projection with ``has_thread_activity``; a create/update answer is the full row
    (``created: false``: an idempotent repeat create)."""

    id: str
    name: str
    chat_id: int
    working_dir: str
    pinned: bool  # absent on legacy rows means unpinned; no routing/activity effect
    cognition: ProjectCognition
    provenance: str
    clone_url: str
    trusted_at: str
    origin: str
    created_at: str
    last_active_at: str
    lifecycle: Literal["active", "deleting", "tombstoned"]
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
    budget_reserve_usd: float
    pending_count: int
    running_count: int
    queued_task_id: str
    running_task_id: str
    campaign: NotRequired[Dict[str, Any]]


class ActiveDirectTurn(TypedDict):
    """An active in-process direct chat or ephemeral decision turn.

    Rows of ``StateResponse.active_direct_turns``: ``DirectActivityRegistry.snapshot()``
    always emits the seven identity/activity fields (empty string for absent values),
    so the mirror marks them required; ``model_waits`` comes only from its live owner.
    """

    activity_id: str
    chat_id: int
    project_id: str
    client_message_id: str
    kind: str
    phase: str
    started_at: float
    model_waits: NotRequired[Dict[str, Any]]
    model_activity: NotRequired[Dict[str, Any]]  # content-free observation bound to the active Main call
    task_attempt: NotRequired[int]


class ActiveChatActivity(ActiveDirectTurn):  # global root census; phase and control facts keep their own authority
    last_line: NotRequired[str]  # visible model narration, never host notes
    last_line_at: NotRequired[str]
    tool_calls: NotRequired[int]
    detached: NotRequired[bool]  # durable detached_at, independent of carrier/phase
    project_name: NotRequired[str]
    status: NotRequired[str]
    outcome_axes: NotRequired[Dict[str, Any]]
    reason_code: NotRequired[str]
    root_phase_checkpoint: NotRequired[Dict[str, Any]]
    timeout_retry_from: NotRequired[str]
    original_task_id: NotRequired[str]
    owner_wait: NotRequired[Dict[str, Any]]  # the waited quiz_id and its wait facts, independent of a Project pointer
    required_question: NotRequired[Dict[str, Any]]
    required_question_unavailable: NotRequired[bool]
    project_admission_hold: NotRequired[Dict[str, Any]]
    pause_cause: NotRequired[str]  # budget | owner | restart | sleep | authority | unknown; display only
    finishing_reviews: NotRequired[bool]  # review work an owner Pause lets finish runs on; display only, no count
    late_work: NotRequired[Dict[str, Any]]  # answered root whose late work remains: title, answered_at; display only
    can_resume: NotRequired[bool]
    cancelable: NotRequired[bool]

class EffortRange(TypedDict):  # helper min ≤ max; primary recommended is independent
    min: str
    recommended: str
    max: str


class ProjectCognition(TypedDict):
    context_mode: str
    effort_range: EffortRange


class OwnerEffortRangeResponse(TypedDict):
    ok: bool
    effort_range: EffortRange
    project_id: str
    cognition: ProjectCognition


class StateResponse(WorkStateFields):
    """Shape of ``GET /api/state`` (happy path); the work projection fields come from ``WorkStateFields``."""

    uptime: int
    workers_alive: int
    workers_total: int
    pending_count: int
    running_count: int
    spent_usd: Optional[float]
    budget_limit: float
    budget_pct: Optional[float]
    branch: Optional[str]  # null for detached HEAD or unavailable checkout identity
    sha: str
    evolution_enabled: Optional[bool]  # None: an unknown control (state unavailable/recovering), never "off"
    bg_consciousness_enabled: Optional[bool]
    state_quality: Dict[str, Any]  # state.json read quality: {quality, source, unconfirmed}
    evolution_cycle: int
    evolution_state: EvolutionStateSnapshot
    bg_consciousness_state: Dict[str, Any]
    spent_calls: Optional[int]
    supervisor_ready: bool
    boot: NotRequired[BootDocument]  # the same boot phases as /api/health; the header paints them
    supervisor_error: Optional[str]
    runtime_mode: str
    context_mode: str
    context_mode_auto_low: bool  # frozen compatibility field, always False (persistent auto-Low is retired)
    effort_range: EffortRange  # the owner's effort range (POST /api/owner/effort-range), the tolerant read
    safety_mode: str
    skills_repo_configured: bool
    github_token_configured: bool
    accounting: Dict[str, Any]
    # Multi-project sidebar feed: compact registered projects [{id, name, chat_id, working_dir, last_active_at, ...}].
    projects: list
    # COMPLETE (uncapped) registered project chat_ids — the live WS fan-out isolation SSOT, unlike `projects`.
    project_chat_ids: list
    # Main's room revision (its standalone messages, the kind a Project's visible_revision counts); null while unreadable.
    main_visible_revision: NotRequired[Optional[int]]
    # Task->project bindings ({task_id: {project_id, chat_id}}) so the frontend
    # can recognise a project-scoped task card: suppress the stray "turn into
    # project" button (v6.33.0 P2) and render a pointer that opens the bound
    # project's panel (v6.33.0 F4).
    task_bindings: dict
    active_direct_turns: NotRequired[List[ActiveDirectTurn]]
    # Combined activity snapshot (direct/ephemeral turns + root managed queue
    # tasks). Additive beside active_direct_turns, which stays unchanged for
    # compatibility; new clients hydrate from this field.
    active_chat_activities: NotRequired[List[ActiveChatActivity]]
    active_chat_activities_complete: NotRequired[bool]
    current_questions: NotRequired[List[Dict[str, Any]]]  # canonical forms independent of history window
    current_questions_complete: NotRequired[bool]
    current_questions_gaps: NotRequired[List[Dict[str, Any]]]  # typed source gaps, no read-side repair
    main_attention: NotRequired[List[Dict[str, Any]]]  # live human recovery owners, including children; census stays root-only
    main_attention_complete: NotRequired[bool]
    main_attention_gaps: NotRequired[List[Dict[str, Any]]]
    served_ui: NotRequired[ServedUiState]  # this generation's main interface; absent on older cores


class SettingsNetworkMeta(TypedDict):
    """Network fields inside the ``GET /api/settings`` ``_meta`` block."""

    bind_host: str
    bind_port: int
    lan_ip: str
    reachability: Literal["loopback_only", "lan_reachable", "host_ip_unknown"]
    recommended_url: str
    warning: str


class AvailableSubagentsSettingsMeta(TypedDict, total=False):
    """Saved/migrated intent returned beside ``GET /api/settings``.

    ``candidate`` is an unsaved canonical object.  Absence stays ``None``; a
    read never materializes the value on disk.  ``automatic``: the catalog is
    the read's own, following Main and access until an edit saves one.
    """

    source: str
    diagnostic: str
    diagnostics: list[Dict[str, Any]]
    candidate: Optional[Dict[str, Any]]
    automatic: bool


class SettingsPolicyAxis(TypedDict, total=False):
    """Configured/effective owner policy values shown by Settings."""

    configured: str
    effective: str
    restart_required: bool
    pending: bool
    applies: Literal["restart", "next_task"]


class SettingsPolicyState(TypedDict):
    access: SettingsPolicyAxis
    supervisor: SettingsPolicyAxis
    review: SettingsPolicyAxis
    running_task_snapshot: bool


class SettingsMeta(SettingsNetworkMeta, total=False):
    """Complete ``GET /api/settings`` ``_meta`` block."""

    custom_secret_keys: list[str]
    setup_contract: Dict[str, Any]
    available_subagents: AvailableSubagentsSettingsMeta
    policy_state: SettingsPolicyState
    restart_state: Dict[str, Any]


class SettingsSaveResponse(TypedDict, total=False):
    status: str
    no_changes: bool
    restart_required: bool
    restart_keys: list[str]
    restart_state: Dict[str, Any]
    immediate_changed: bool
    next_task_changed: bool
    warnings: list[str]
    ignored_keys: Dict[str, str]
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
    project_id: str
    cognition: ProjectCognition


class OwnerSafetyModeResponse(TypedDict):
    ok: bool
    safety_mode: str  # full | light | off (v6.54.3)


class OwnerSkillPresenceRuntimeRequest(TypedDict):
    expected_state_fingerprint: str
    runtime_overrides: Dict[str, Any]
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
    enabled: bool  # present when this grant also carried the owner's Enable
    extension_action: str
    extension_reason: str
    load_error: str
    grants: Dict[str, Any]
    process: str
    server_reconcile: str
    live_loaded: bool
    error: str


class SkillDeleteResponse(TypedDict, total=False):
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
    # Tag rows: {tag, date, sha (peeled commit), message} — the mirror said ``list[str]`` while ``list_versions``
    # has always emitted dicts; corrected (behavioural documentation) in the 2026-08-31 updates redesign.
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
    skill: str
    repository: str
    state: Literal["ready", "warnings", "needs_attention", "repairable", "hard_block"]
    publication_ready: bool
    task_start_allowed: bool
    snapshot_hash: str
    review: Dict[str, Any]
    scanner: Dict[str, Any]
    findings: list[Dict[str, Any]]
    omitted_count: int
    blocker_count: int
    warning_count: int
    audited_false_positive_count: int
    reason_code: str
    summary: str
    repair_hint: str


class OpenAICompatibleModelsResponse(TypedDict):
    """``POST /api/openai-compatible/models``: upstream ``data[].id`` in upstream order."""

    models: List[str]
    error: NotRequired[str]


class ProviderTestRequest(TypedDict):
    provider_id: str
    overrides: NotRequired[Dict[str, str]]


class ProviderTestResponse(TypedDict):
    ok: bool
    error: NotRequired[str]


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
    # v6.115.0: the run's owner-facing name. Supplied, it fills both name slots
    # like a promoted chat turn; omitted, admission derives a display-only name
    # from the request's first line. `metadata.title` is refused (400).
    title: str
    chat_id: int
    depth: int
    session_id: str
    workspace_root: str
    workspace_mode: str
    memory_mode: str
    project_id: str
    attachments: list[Dict[str, Any]]
    # Partial staging is the default (В25c, capinv-447): omitted/true stages the good
    # attachments and discloses rejected rows; explicit false keeps the old atomic admission.
    allow_partial_attachments: bool
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
    execution_timeout_sec: float  # excludes queue and confirmed pauses; 0 adds no limit
    context: str
    expected_output: str
    constraints: str
    context_requires_self_body_docs: bool
    reasoning_effort: str  # optional explicit starting effort: an EFFORT_SCALE tier, checked by the handler
    actor_id: str
    source: str
    metadata: Dict[str, Any]


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


class TaskCostBreakdown(TypedDict):
    """Root-only task-detail ledger projection, computed on read, never persisted.
    All keys appear together; ``own + children + unattributed == subtree``. Omit
    the object for non-roots, unavailable accounting or no attributable subtree rows; never invent $0."""

    own_usd: float
    children_usd: float
    unattributed_usd: float
    delegated_disclosed_usd: float
    # C2: own + children + unattributed is an accounted UPPER BOUND, not a receipt.
    accounted_upper_bound_usd: float
    subscription_sessions: int
    unknown_unmetered: int
    non_final_rows: int
    cost_final: bool
    authority: Literal["physical_attempt_ledger"]


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


class TaskDetailResponse(TypedDict, total=False):
    """Open ``GET /api/tasks/{task_id}`` result: stored keys pass through beside typed projections."""

    artifacts: List[Dict[str, Any]]  # Open result rows; a nested file carries additive ``relpath``.
    # Per result directory: {name, files, size, excluded, available} for its ``?archive=`` ZIP.
    artifact_archives: Dict[str, Dict[str, Any]]
    # Wrap up's saved-work document {name, request_id}: ``?wrap_up=<request_id>`` writes it on read; no size/digest.
    wrap_up_document: Dict[str, str]
    cost_breakdown: TaskCostBreakdown
    history_retention: HistoryRetentionProjection
    model_waits: Dict[str, Any]
    question_rooms: Dict[str, QuestionRoom]
    # Open cancel intent/teardown: "pending" drives "Cancelling…", not lifecycle status.
    # Absent for settled or unrequested cancellation; running/scheduled status stays truthful.
    cancel_state: str
    # Pending intent's recorded reason (owner text or subtree cause); absent if none (GR2-11).
    cancel_reason: str
    # Soft finalize_then_cancel: "Finalizing…" plus hard escalation; absent on immediate intents (S3/Q1).
    stop_policy: str
    # Current hurry plus prior same-id attempts; detail-only, absent if never hurried (S3/HQ1).
    continuation_offer: ContinuationOffer
    owner_hurry: OwnerHurryProjection
    owner_hurry_history: list[OwnerHurryProjection]
    project_admission_hold: Dict[str, Any]  # While the queue snapshot lists the row: its hold {reason, detail, label} or {}.
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


class ClaudexorLoginJobProblem(TypedDict, total=False):
    """Typed job errors preserve engine prose/status/code and bounded actions
    (16 strings, 512 chars each). Covers absent jobs, input/reconcile conflicts,
    setup 400/409 and marked retryable terminal-probe 503. Other daemon 5xx and
    unmarked transport/discovery failures use the generic proxy 503."""

    error: Required[str]
    code: str
    required_actions: List[str]


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


class ContinuationOffer(TypedDict, total=False):
    """Bound is retryable using action_nonce; only admitted is a successor pointer."""
    eligible: bool
    refusal: str
    cause: str
    successor_task_id: str
    state: Literal["bound", "admitted"]
    action_nonce: str


class TaskCancelRequest(TypedDict, total=False):
    """Optional body; stop_action_id retries one action, distinct from custody ID.
    Omission preserves legacy ingress without exact-replay assurance."""
    cascade: bool
    stop_policy: str
    stop_action_id: str


class TaskCancelResponse(TypedDict, total=False):
    ok: bool
    task_id: str
    # Cascade echoes subtree scope; immediate success follows teardown.
    cascade: bool
    # Every 202 keeps the recorded intent pending and echoes the EFFECTIVE policy,
    # never softening an immediate intent: a soft stop, or an immediate teardown not
    # yet confirmed (request_id, reason_code "teardown_unconfirmed"). 200 omits them.
    cancel_state: str
    stop_policy: str
    request_id: str
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


class OnboardingCompleteRequest(TypedDict, total=False):
    """Open setup-settings payload plus subscription declarations, actor draft and starter choices. None is
    authority: subscriptionsConnected requests a live account read; the server re-proves fresh install."""

    subscriptionsConnected: bool
    skipSubscriptionPresets: bool
    OUROBOROS_SUBAGENTS: Dict[str, Any]
    allow_empty_review_pool: bool  # confirms saving agents that leave no review row
    starter_skills: List[str]  # catalog ids installed once after completion; absent: the default starter list
    skill_source_trust: Dict[str, str]  # {hub row id: trusted|review}; marks those rows confirmed


class OnboardingSubagentsPreviewResponse(TypedDict):
    """Read-only canonical actor draft returned to Settings/onboarding."""

    ok: bool
    model_settings: Dict[str, Any]  # the model setting keys the draft would save
    available_subagents: Dict[str, Any]
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
    receipt: Dict[str, Any]


class OnboardingCompleteResponse(TypedDict):
    """Settings, next-boot mode, safety default and completion persist atomically; preset and marker if applied."""

    ok: bool
    status: str
    runtime_mode: str
    restart_required: bool
    preset: OnboardingPresetProjection


class SettingsPostCommitFailureResponse(TypedDict):
    """500 after settings bytes landed; post_commit_failed names the later step.
    Retry that step, never re-save. Shared by settings and onboarding writes."""

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
from ouroboros.gateway.endpoint_index import HTTP_ENDPOINTS
from ouroboros.gateway.client_contracts import FileBrowserListResponse, ServedUiState  # noqa: E402,F401

WS_MESSAGE_TYPES: tuple[str, ...] = (
    "conversation_finished",
    "chat",
    "command",
    "panic_accepted",  # immediate acceptance only, not proof of process exit
    "photo",
    "video",
    "document",
    "links",
    "quiz",
    "quiz_state",
    "card", "card_state",  # gateway/card_contracts.py frames
    "typing",
    "log",
    "heartbeat",
    "extension_lifecycle",
    "message_annotation",
    "projects_changed",
    "task_named",
    "update_status_ready",
    "update_progress_changed",
    "ui_language_changed", "ui_i18n_updated",  # gateway/ui_i18n_contracts.py frames
)


__all__ = [
    "CostPresentation",
    "ChatInbound", "TaskConstraintInbound", "CommandInbound", "PanicAcceptedOutbound", "ExtensionInbound",
    "TransportMetadata",
    "ChatOutbound",
    "PhotoOutbound", "VideoOutbound", "DocumentOutbound",
    "LinkAction", "LinksOutbound",
    "QuizOption", "QuizOutbound", "QuizStateOutbound",
    "CardOutbound", "CardStateOutbound",
    "DecisionRequest", "DecisionResponse",
    "TypingOutbound",
    "LogOutbound",
    "HeartbeatOutbound",
    "ExtensionLifecycleOutbound",
    "ProjectsChangedOutbound",
    "MessageAnnotationOutbound",
    "UpdateMergePlan",
    "UpdatePreflightRequest", "UpdatePreflightResponse",
    "UpdateApplyRequest",
    "UpdateApplySuccessResponse", "UpdateApplyErrorResponse",
    "UpdateStatusReadyOutbound",
    "UpdateProgressChangedOutbound",
    "ProjectCreateRequest", "ProjectEntry", "ProjectDeleteResponse", "ProjectCognition",
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
    "OnboardingCompleteRequest", "OnboardingCompleteResponse",
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
    "ChatAttachmentView",
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
    "TaskEventsRequest",
    "TaskCancelResponse", "TaskCancelRequest",
    "TaskPauseRequest", "TaskPauseResponse",
    "TaskContinueRequest", "TaskContinueResponse", "ContinuationOffer",
    "TaskHurryRequest", "TaskHurryResponse",
    "OwnerHurryProjection",
    "EffortRange", "OwnerEffortRangeResponse",
    "LogTailResponse",
    "StorageCount", "StorageCounting", "StorageError", "StoragePart", "StoragePlace", "StorageResponse",
    "StorageRest", "StorageVolume",
    "StorageFree", "StorageFreeGroup", "StorageFreeItem", "StorageFreeKept", "StorageFreeRequest", "StorageFreeRun",
    "HTTP_ENDPOINTS",
    "WS_MESSAGE_TYPES",
]
