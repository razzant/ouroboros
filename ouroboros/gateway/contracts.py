"""Descriptive HTTP + WebSocket Gateway Boundary contracts (v1).

TypedDicts document payloads, not runtime validation. Keep discriminating
``type`` keys required; mark genuinely optional fields with ``NotRequired``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ouroboros.gateway.history_contracts import ChatHistoryResponse  # noqa: F401 -- public re-export
from ouroboros.gateway.widgets import ExtensionLiveSnapshot, WidgetTab, WidgetsResponse
from ouroboros.gateway.decision_contracts import DecisionRequest, DecisionResponse  # noqa: F401 -- public re-exports

try:  # Python 3.11+
    from typing import Literal, NotRequired, Required, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import Literal, NotRequired, Required, TypedDict  # type: ignore[assignment]


class ChatAttachmentInbound(TypedDict, total=False):
    """Reference to a file stored by /api/chat/upload under data/uploads/.
    ``filename`` is its stored basename. Images reach vision models as
    native image blocks."""

    filename: str
    display_name: str
    mime: str


class AttachmentManifestEntry(TypedDict, total=False):
    """One declared task attachment after staging admission."""

    ordinal: int
    status: Literal["staged", "rejected"]
    reason: str
    label: str
    root: str
    relpath: str
    abs_path: str
    mime: str
    is_image: bool
    size: int
    sha256: str
    rule: str


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
    # Per-message sending-surface observables (additive-optional): raw facts the SPA measures at send
    # time (pywebview bridge presence, ua, viewport, matchMedia booleans, captured_at), normalized by
    # the closed-key `client_surface.normalize_client_surface`; absence is an honest gap, never defaulted.
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
    origin_message_ref: NotRequired[Dict[str, Any]]
    # X3: a repair receipt whose managed task id the router mints only at promotion (typed truth, no invented id).
    task_id_pending: NotRequired[bool]
    # "finalizing" on a root's early final answer: the answer is delivered
    # while post-task synthesis still runs, so the frame is NOT the task's
    # terminal conclusion — task_done settles the card/turn.
    task_phase: NotRequired[str]
    # Direct/ephemeral finals and errors settle client activity without a snapshot:
    # completed/failed/cancelled/rejected_duplicate, not an early answer.
    task_terminal_status: NotRequired[str]
    ephemeral_decision: NotRequired[bool]
    tool_calls: NotRequired[int]
    rounds: NotRequired[int]
    suggested_name: NotRequired[str]
    model_execution: NotRequired[Dict[str, Any]]
    # Project question mirrored into Main as the Project's own form; the durable question stays in Project.
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
    task_incident: NotRequired[str]
    # A cancellation fault names the PHYSICAL task it could not settle when it differs from the logical task id.
    cancel_physical_task_id: NotRequired[str]
    toast_once: NotRequired[str]
    # #628: the incident's valence for the one-shot toast (warn/ok/error),
    # stamped by the producer that knows whether the boundary is a wait, a
    # recovery or an exhaustion; absent = the browser keeps its alarm tone.
    toast_tone: NotRequired[str]
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
    # Phase 6: the OPAQUE harness route RESOLVED AT DISPATCH for this bubble/subagent (`resolve_subagent_dispatch`,
    # stamped once) — a delegated route only; absent/empty means the ordinary native path and the UI draws no chip.
    # It is the route the run was sent to, not a receipt from the engine saying where it landed: a landing below the
    # ask is disclosed on `capability_delta`, not by rewriting this field.
    executor_route: NotRequired[str]
    # Latest observed progress actor, NOT terminal evidence or current liveness. Own
    # task_id/task_attempt/run_id/attempt_id, harness_id, phase, revision; optional model has explicit model_source
    # (requested or observed).
    executor_observation: NotRequired[Dict[str, Any]]
    # The completion-seam EVIDENCE the route decision is reconciled against (subagents.envelope_from_task):
    # delegated runs started/settled/succeeded, terminal failure states, disclosed subscription spend (+estimated
    # flag), engine-reported models, the additive `nanny_nudge_recorded` flag (a non-empty finalization nudge was
    # durably stamped), and the additive `delegate_start_attempted` flag (any durable delegate_start attempt,
    # refused or started). Terminal frames only; its absence means "no evidence yet", never "ran natively".
    execution_evidence: NotRequired[Dict[str, Any]]
    # The FACT beside the executor_route plan, from the same custody evidence:
    # "harness_used" | "harness_attempted" | "native_only". Terminal frames only; absent =
    # no substrate claim.
    actual_substrate: NotRequired[str]
    model: NotRequired[str]
    task_group_id: NotRequired[str]
    task_event: NotRequired[str]
    status: NotRequired[str]
    # v6.82 (P5): host-attested marker, stamped by the supervisor's delivery seam ONLY for a task POST /api/tasks/{id}/cancel
    # will actually stop — a lineage-resolved pooled ROOT (its RUNNING row) or the live in-process direct-chat turn (resolved
    # through the same ownership reader the endpoint uses, supervisor.workers.direct_chat_turn); never a subagent frame, never
    # an ephemeral decision turn. Gates the UI "Cancel run" action.
    cancelable: NotRequired[bool]
    _is_direct_chat: NotRequired[bool]  # lane fact stamped on a direct turn's own frames
    narration: NotRequired[bool]  # progress VOICE: the model's own round narration (true) vs a host note (false); absent = legacy
    initiator: NotRequired[str]  # origin label: "consciousness" on a wake-up's frames/rows (and its roots); absent on an owner's turn
    # Monetary projections are nullable when the physical-attempt ledger cannot be read: ``None`` is
    # distinct from a confirmed $0. These are the honest names (accounted upper bounds, not settled
    # receipts) and the only outbound spellings since ABI 7.0 dropped the ``cost_usd[_with_children]``
    # aliases; ouroboros/cost_projection.py is the one author, ``resolve_cost_pair`` reads legacy records.
    accounted_upper_bound_usd: NotRequired[Optional[float]]
    accounted_upper_bound_usd_with_children: NotRequired[Optional[float]]
    cost_accounting_status: NotRequired[Literal["available", "unavailable"]]
    cost_accounting_error: NotRequired[str]
    cost_final: NotRequired[bool]
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
    # UI-only system annotation emitted by skill-repair visible commands.
    system_type: NotRequired[str]
    # A host-stamped placement fact for a task-keyed System row: "timeline" = a timeline item of the task's card,
    # "reviews" = the card's Reviews group carries the fact (the row is still attached to the card); absent = an
    # ordinary row. ``card_row_id`` is the row's stable identity across live delivery, outbox replay and history.
    card_row: NotRequired[Literal["timeline", "reviews"]]
    card_row_id: NotRequired[str]
    # Event-time human presentation; raw task/project ids remain machine keys.
    target_label: NotRequired[str]
    project_id: NotRequired[str]
    project_name: NotRequired[str]
    # Present on some transport re-broadcast paths.
    chat_id: NotRequired[int]
    # Server-stamped when chat_id is a reserved Project thread: Main never
    # adopts it, even before the browser has learned the project.
    project_thread: NotRequired[bool]


class PhotoOutbound(TypedDict):
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
    # Server-stamped when chat_id is a reserved Project thread: Main never
    # adopts it, even before the browser has learned the project.
    project_thread: NotRequired[bool]


class VideoOutbound(TypedDict):
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


class DocumentOutbound(TypedDict):
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
    # Server-stamped when chat_id is a reserved Project thread: Main never
    # adopts it, even before the browser has learned the project.
    project_thread: NotRequired[bool]


class LinkAction(TypedDict):
    """One validated HTTP(S) action in a structured links frame."""

    label: str
    url: str


class LinksOutbound(TypedDict):
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


class QuizOutbound(TypedDict):
    """Outbound owner quiz card: a typed question with option buttons.

    Optional questions continue under ``assumption``; ``wait_for_answer`` marks
    required waiting without an implied answer. ``state`` carries lifecycle.
    Replay merges the stored ``answered_index`` and verbatim ``comment``; a
    comment without an index is the owner's whole answer, not an option choice.
    """

    type: Literal["quiz"]
    role: Literal["assistant"]
    quiz_id: str
    question: str
    options: list[QuizOption]
    stake: str
    assumption: str
    wait_for_answer: NotRequired[bool]
    state: str
    ts: str
    answered_index: NotRequired[int]
    comment: NotRequired[str]
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


class TypingOutbound(TypedDict):
    """Outbound WS typing indicator."""

    type: Literal["typing"]
    action: str
    # Multi-project: stamps the thread so the client fan-out routes a project
    # task's typing indicator to its panel instead of defaulting to main.
    chat_id: NotRequired[int]
    # Server-stamped when chat_id is a reserved Project thread: Main never
    # adopts it, even before the browser has learned the project.
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
    # Server-stamped when chat_id is a reserved Project thread: Main never
    # adopts it, even before the browser has learned the project.
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
    """Outbound notice that the project registry changed server-side (e.g. the
    agent's ``promote_chat_to_task`` created/bound a project). The client refreshes
    its project nav + WS-fan-out ``projectChatIds`` on receipt; ``chat_id`` lets it
    learn the new project thread immediately, before the /api/state round-trip."""

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
    target_label: NotRequired[str]
    project_id: NotRequired[str]
    project_chat_id: NotRequired[int]
    options: NotRequired[List[Dict[str, Any]]]
    attachment_manifest: NotRequired[List[AttachmentManifestEntry]]
    # #198: the exact refusal-attempt identity — the picker card composes its decision_id
    # (routing:{client_message_id}:{routing_token}) from it; a frame without it renders text, never a card.
    routing_token: NotRequired[str]
    cause: NotRequired[str]
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
    # Stash-first prologue disclosures (additive): how the owner's stashed work
    # was unwound on an aborted update, and the wave-floor admission numbers.
    stash_note: NotRequired[str]
    estimated_wave_usd: NotRequired[Optional[float]]
    remaining_usd: NotRequired[Optional[float]]


class UpdateProgressChangedOutbound(TypedDict):
    """Invalidation only; never proof that boot finalization completed."""
    type: Literal["update_progress_changed"]


class UpdateStatusReadyOutbound(TypedDict):
    type: Literal["update_status_ready"]
    available: bool
    check_ok: Optional[bool]


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
    """Outbound notice that a project name was coined for a task's card (inline naming of
    a turn-into-project conversion; direct turns are not named in the background). The
    client sets the live card's title to ``suggested_name``. Not chat-scoped — carries
    only ``task_id`` and is a no-op unless a thread already holds that card."""

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


class ActiveDirectTurn(TypedDict):
    """An active in-process direct chat or ephemeral decision turn.

    Snapshot rows in ``StateResponse.active_direct_turns``; the seven identity
    and activity fields are always emitted by ``DirectActivityRegistry.snapshot()``
    (empty-string for absent values), so the mirror marks them required.
    A model_waits projection is optional and must be supplied by its live owner.
    """

    activity_id: str
    chat_id: int
    project_id: str
    client_message_id: str
    kind: str
    phase: str
    started_at: float
    model_waits: NotRequired[Dict[str, Any]]
    task_attempt: NotRequired[int]


class ActiveChatActivity(ActiveDirectTurn):
    """One in-flight chat activity in ``StateResponse.active_chat_activities``.

    The combined snapshot: direct/ephemeral registry turns (same rows as
    ``active_direct_turns``) plus ROOT managed queue tasks projected as
    ``kind="managed_task"`` with ``phase`` ``queued`` | ``budget_paused``
    (zero-dispatch member awaiting an explicit resume — never plain
    "queued") | ``working`` | ``finalizing`` (final answer stored, post-task
    synthesis still open).
    Field shape mirrors ``ActiveDirectTurn`` so one client reducer hydrates
    both; managed rows carry an empty ``client_message_id``.
    """

    required_question: NotRequired[Dict[str, Any]]


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
    active_direct_turns: NotRequired[List[ActiveDirectTurn]]
    # Combined activity snapshot (direct/ephemeral turns + root managed queue
    # tasks). Additive beside active_direct_turns, which stays unchanged for
    # compatibility; new clients hydrate from this field.
    active_chat_activities: NotRequired[List[ActiveChatActivity]]
    active_chat_activities_complete: NotRequired[bool]


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
    read never materializes the value on disk.
    """

    source: str
    diagnostic: str
    diagnostics: list[Dict[str, Any]]
    candidate: Optional[Dict[str, Any]]


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
    widget_start_mode: dict[str, Literal["auto", "manual", "retain"]]  # owner per-card launch-policy override
    nested_subagents_expanded: bool
    sidebar_width: int  # px; 0 = CSS default (resizable side sections, v6.33.0)
    project_panel_width: int  # px; 0 = CSS default
    project_seen_revision: dict[str, int]  # monotonic paint ACK per active Project


class GitLogResponse(TypedDict):
    commits: list[Dict[str, Any]]
    # Tag rows: {tag, date, sha (peeled commit), message} — the mirror said
    # ``list[str]`` while ``list_versions`` has always emitted dicts; corrected
    # (behavioural documentation) in the 2026-08-31 updates redesign.
    tags: list[Dict[str, Any]]
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
    sha256: NotRequired[str]
    mime: str


class ExtensionsIndexResponse(TypedDict, total=False):
    extensions: list[Dict[str, Any]]
    skills: list[Dict[str, Any]]
    live: ExtensionLiveSnapshot
    lifecycle: Dict[str, Any]
    error: str


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
    settings_application: Dict[str, Any]


class McpStatusResponse(TypedDict, total=False):
    enabled: bool
    servers: list[Dict[str, Any]]
    tools: list[Dict[str, Any]]
    error: str


class ModelCatalogResponse(TypedDict, total=False):
    providers: list[Dict[str, Any]]
    models: list[Dict[str, Any]]
    error: str


class ProviderTestRequest(TypedDict):
    provider_id: str
    overrides: NotRequired[Dict[str, str]]


class ProviderTestResponse(TypedDict):
    ok: bool
    error: NotRequired[str]


class FileBrowserListResponse(TypedDict, total=False):
    root: str
    path: str
    entries: list[Dict[str, Any]]
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
    # Partial staging is the default (В25c, capinv-447): omitted/true stages
    # the good attachments and discloses rejected rows; explicit false keeps
    # the old atomic all-or-nothing admission.
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
    reason_code: str
    error: str
    attachment_manifest: list[AttachmentManifestEntry]
    attachment_manifest_ref: Dict[str, Any]


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
    model_waits: Dict[str, Any]
    # Cancel projection (additive-optional): ``"pending"`` while a durable cancel intent is open and the
    # supervisor teardown has not settled — the status itself honestly stays running/scheduled; absent on
    # settled results and on tasks nobody asked to cancel. The UI's interim "Cancelling…" reads this, never a status.
    cancel_state: str
    # Rides beside ``cancel_state`` when the intent carries a reason (GR2-11):
    # the WHY of the pending cancellation (owner text, "subtree cancellation of
    # <root>", "evolution stopped", …). Absent when no reason was recorded.
    cancel_reason: str
    # S3 (Q1, additive-optional): rides beside a pending ``cancel_state`` when
    # the open intent is the SOFT stop ("finalize_then_cancel") — the UI shows
    # "Finalizing…" and offers the hard escalation. Absent on immediate intents.
    stop_policy: str
    # S3 (HQ1, additive-optional): the typed owner-hurry observability — the
    # current block plus the archived history of prior same-id attempts.
    # Absent on tasks nobody hurried. Task-detail data only, never chat.
    owner_hurry: OwnerHurryProjection
    owner_hurry_history: list[OwnerHurryProjection]
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
    quota_absences: List[Dict[str, Any]]
    reads: ClaudexorStatusReads
    # UNIFIED ACCOUNT MODEL feature fact (additive-optional): True only when the engine's own
    # /v2/operations catalog was read and advertises `GET /v2/account-pools` (every default CLI login
    # becomes a named registry row, `harnessAccounts` empties, pool routing rides `profiles.accountPools`).
    # False, or absent on an older backend, means the legacy native-pseudo-row rendering; an unreadable
    # catalog fails closed to False.
    unified_accounts: bool
    subagent_last_delegation: Dict[str, Any]
    error: str


class ClaudexorLoginJobResponse(TypedDict, total=False):
    """The ONE login-job success envelope (frozen browser gateway ABI,
    issues #124/#151): every ``/api/claudexor/login`` operation — create,
    snapshot poll, cancel, input, reconcile — answers exactly one top-level
    bare ``job`` (the daemon's ``ControlSetupJob``), never another envelope
    nested under it (the double ``job.job`` was issue #124).

    Operation metadata rides BESIDE the job: create adds ``job_id``,
    ``disclosure_native``, ``setup_login_source`` and (external-terminal
    flows whose exact packaged attach role was proven) the labelled
    ``attach_command`` / ``attach_shell`` pair;
    input keeps its ``ok`` bit; the snapshot poll is the daemon's own
    ``{job, cursor, sequence, deviceCode?}`` envelope passed through
    verbatim, so the transient sign-in disclosure lives at the ENVELOPE
    level, not inside ``job``. ``job`` is required on every operation; all
    other keys are operation-scoped."""

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
    """The narrow login-job error envelope (frozen beside the success DTO —
    the recovery UI consumes both sides of one operation contract): required
    ``error`` prose, optional stable machine ``code``, optional bounded
    ``required_actions`` naming the engine's continuation (e.g. reconcile's
    409 ``setup_termination_unconfirmed`` carries
    ``["retry_setup_reconciliation"]``). Daemon 404/410 job-absence verdicts,
    the operation-scoped input/reconcile 409s, and setup-create 400/409 or the
    frozen retryable 503 terminal-transport probe verdict ride this shape with
    their original status, stable code, actions and the engine's own sentence.
    Unmarked transport/discovery 503s and other daemon 5xx stay the proxy's
    generic 503.
    Not an action framework: the list mirrors the daemon's own top-level
    ``ControlProblem.requiredActions`` (at most 16 strings of at most 512
    chars) and nothing else."""

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


class TaskCancelResponse(TypedDict, total=False):
    ok: bool
    task_id: str
    # v6.82 (P5): echoed when the optional request body {"cascade": true} asked
    # for the subtree cancel, which is COMPLETE by the time this answer is sent;
    # the plain envelope is unchanged.
    cascade: bool
    # Additive on the 202 acknowledgement of a ``{"stop_policy": "finalize_then_cancel"}`` request: the
    # durable intent is open ("pending") while the bounded finalization attempt runs, and ``stop_policy``
    # echoes the EFFECTIVE policy of the durable intent ("immediate" | "finalize_then_cancel") — a graceful
    # request over an already-hard intent never softens it, and the answer says so. Absent on the legacy immediate path.
    cancel_state: str
    stop_policy: str
    error: str


class TaskHurryRequest(TypedDict):
    """``POST /api/tasks/{task_id}/hurry`` — the text-free owner hurry control
    (HQ1 owner decision, paraphrased: no visible chat message ever).

    The body carries ONLY a client-generated stable ``request_id`` (reused on
    retry so the acknowledgement is idempotent); any other field is refused.
    There is deliberately no text and no chat side effect anywhere on this
    path — the durable facts are the typed owner-mailbox control, the
    ``owner_hurry`` task-result projection, and one non-chat event."""

    request_id: str


class OwnerHurryProjection(TypedDict, total=False):
    """The ``owner_hurry`` block on the task result — task-detail
    observability, never a chat message. ``state`` is the closed vocabulary
    requested | applied | not_applied_before_terminal; ``effects`` maps each
    host-rail effect to its recorded status. ``owner_hurry_history`` rows
    carry the same shape plus ``archived_at``/``archived_reason`` (rolled over
    on every same-id requeue by the shared retry-reset)."""

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
    """``POST /api/onboarding/complete`` — the wizard payload plus two
    DECLARATIONS about the onboarding run itself.

    The settings keys of the shared setup contract ride through unchanged (open
    shape, same payload the wizard already builds); the two subscription flags
    and canonical actor draft are typed here. None is authority:
    ``subscriptionsConnected`` only tells the server to read the live
    agent account state, and the server re-proves fresh-install status
    on its own before applying anything."""

    subscriptionsConnected: bool
    skipSubscriptionPresets: bool
    OUROBOROS_SUBAGENTS: Dict[str, Any]


class OnboardingSubagentsPreviewResponse(TypedDict):
    """Read-only canonical actor draft returned to Settings/onboarding."""

    ok: bool
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
from ouroboros.gateway.endpoint_index import HTTP_ENDPOINTS

WS_MESSAGE_TYPES: tuple[str, ...] = (
    "chat",
    "command",
    "photo",
    "video",
    "document",
    "links",
    "quiz",
    "quiz_state",
    "typing",
    "log",
    "heartbeat",
    "extension_lifecycle",
    "message_annotation",
    "projects_changed",
    "task_named",
    "update_status_ready",
    "update_progress_changed",
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
    "LinkAction",
    "LinksOutbound",
    "QuizOption",
    "QuizOutbound",
    "QuizStateOutbound",
    "DecisionRequest",
    "DecisionResponse",
    "TypingOutbound",
    "LogOutbound",
    "HeartbeatOutbound",
    "ExtensionLifecycleOutbound",
    "ProjectsChangedOutbound",
    "MessageAnnotationOutbound",
    "UpdateMergePlan",
    "UpdatePreflightRequest",
    "UpdatePreflightResponse",
    "UpdateApplyRequest",
    "UpdateApplySuccessResponse",
    "UpdateApplyErrorResponse",
    "UpdateStatusReadyOutbound",
    "UpdateProgressChangedOutbound",
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
    "ActiveDirectTurn",
    "ActiveChatActivity",
    "EvolutionStateSnapshot",
    "SettingsNetworkMeta",
    "AvailableSubagentsSettingsMeta",
    "SettingsPolicyAxis",
    "SettingsPolicyState",
    "SettingsMeta",
    "SettingsSaveResponse",
    "OwnerRuntimeModeResponse",
    "OwnerAutoGrantResponse",
    "OwnerContextModeResponse",
    "OwnerSafetyModeResponse",
    "OwnerSkillPresenceRuntimeRequest",
    "OwnerSkillPresenceRuntimeResponse",
    "OnboardingCompleteRequest",
    "OnboardingCompleteResponse",
    "OnboardingSubagentsPreviewResponse",
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
    "ExtensionLiveSnapshot",
    "WidgetTab",
    "WidgetsResponse",
    "SkillPublishPreflightResponse",
    "SkillLifecycleQueueResponse",
    "MarketplaceSearchResponse",
    "MarketplaceInstalledResponse",
    "LocalModelStatusResponse",
    "McpStatusResponse",
    "ModelCatalogResponse",
    "ProviderTestRequest",
    "ProviderTestResponse",
    "FileBrowserListResponse",
    "ChatHistoryResponse",
    "AttachmentManifestEntry",
    "ExecutorRef",
    "TaskCreateRequest",
    "TaskCreateResponse",
    "TaskListResponse",
    "TaskCostBreakdown",
    "TaskDetailResponse",
    "ClaudexorReadState",
    "ClaudexorStatusReads",
    "ClaudexorStatusResponse",
    "ClaudexorLoginJobResponse",
    "ClaudexorLoginJobProblem",
    "ClaudexorVendorCredentialDisposition",
    "ClaudexorCredentialProfileDeleteResponse",
    "TaskEvent",
    "TaskEventCursor",
    "TaskEventsRequest",
    "TaskCancelResponse",
    "TaskHurryRequest",
    "TaskHurryResponse",
    "OwnerHurryProjection",
    "LogTailResponse",
    "HTTP_ENDPOINTS",
    "WS_MESSAGE_TYPES",
]
