"""Descriptive Chat history response and its rows, re-exported by the gateway facade.

``GET /api/chat/history`` projects the dialogue store's stored chat and progress rows
(``state/dialog.sqlite``) plus a few synthesized ones (``gateway/history.py``). The JSDoc twin is
``web/modules/client_contract_types.js``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ouroboros.gateway.attachment_contracts import ChatAttachmentView
from ouroboros.gateway.reference_contracts import ChatReplyReference

try:
    from typing import Literal, Required, TypedDict
except ImportError:  # pragma: no cover - Python 3.10 compatibility.
    from typing_extensions import Literal, Required, TypedDict


class ChatHistoryPosition(TypedDict):
    """Where a stored row lies: its stream and the row's chain coordinate in the dialogue store (the
    byte offset the row has in the frozen archive plus the rows appended since). It names a stored
    record, never current task or review authority."""

    source: Literal["chat", "progress"]
    offset: int


class PresenceContact(TypedDict, total=False):
    """Host-retained source of a private Main contact; external context, never owner authority."""

    task_id: Required[str]
    binding_id: Required[str]
    source_event_id: Required[str]
    label: Required[str]
    conversation_key: Required[str]
    transport_skill: str
    behavior_skill: str
    profile_fingerprint: str
    state_fingerprint: str
    selection_fingerprint: str
    provider: str
    account_id: str
    conversation_id: str
    thread_id: str
    actor_id: str
    place_id: str
    url: str


class ChatHistoryQuiz(TypedDict, total=False):
    """The question card of a ``msg_type: "quiz"`` row (the question itself is the row's ``text``).
    ``state`` is the durable lifecycle merged at read time: open, answered, closed,
    expired_terminal or superseded. ``answered_index`` and ``comment`` are the recorded answer."""

    quiz_id: Required[str]
    options: List[Dict[str, Any]]  # QuizOption rows: {label, detail?, recommended?}
    stake: str
    assumption: str
    state: str
    wait_for_answer: bool
    wait_ended_at: str
    presence_contact: PresenceContact
    host_facts: str
    answered_index: int
    comment: str
    owner_wait_state: str
    owner_wait_resume_reason: str


class ChatHistoryCard(TypedDict):
    """The living card of a ``msg_type: "card"`` row (its plain ``fallback_text`` is the row's ``text``):
    its identity as first published. The current body, revision and presses are read from
    ``GET /api/cards/{card_id}``; a ``card_state`` frame says when they changed."""

    card_id: str
    kind: Literal["components", "widget", "app"]
    title: str
    revision: int


class PlaceHistoryAttention(TypedDict, total=False):
    decision: Required[Literal['heard', 'focus_started']]
    reason: str  # retained heard reason; no inferred matched wake


class PlaceHistoryImport(TypedDict):
    """An earlier message its connector imported: who sent it on the platform and when the Host
    received it; the row's ``ts`` is the provider's send time."""

    authorship: Literal['participant', 'platform_account']
    received_at: str


class PlaceHistoryFacts(TypedDict, total=False):
    """Retained room speech/notices only. Absence is unknown, never delivered or woken."""

    kind: Required[Literal['message', 'notice', 'delivery']]
    delivery_state: Literal['authored', 'accepted', 'delivered', 'failed', 'uncertain']
    attention: PlaceHistoryAttention
    owner_confirmed: bool  # current stored account confirmation, never inferred from role=user
    facts: List[str]  # a heard row's declared event facts ([] when none were declared)
    identity: Dict[str, Any]  # captured incoming rows: the connector's declared identity at that moment
    sends_as: Dict[str, Any]  # receipt rows: {id, label, kind} frozen from the delivering focus
    sends_as_owner: bool
    held_text: str  # hold notices: the whole reply the hold kept back
    held: bool
    checked: bool  # False on holds: the hold precedes the Safety check
    historical_import: PlaceHistoryImport  # never owner confirmation or attention
    sent_at: str  # the provider send time its arrival recorded (``source_order``); absent: unknown, never inferred


class ChatHistoryMessage(TypedDict, total=False):
    """One ``GET /api/chat/history`` row. Every row has the six required keys; a stored row also
    has ``history_id`` and ``history_position``. Live frames call ``text`` ``content``.

    A row may also carry the task-progress and task-outcome fields a live ``chat`` frame carries
    (``ChatOutbound``: lineage, cost, outcome, model and card placement). A replay-evidence row
    (``system_type`` ``quiz_answer``, ``quiz_closed`` or ``task_evidence``, or ``summary_kind`` with
    ``historical_terminal``) carries facts for cards whose rows lie on other pages and is never
    a message to show. A derived row (``task_model_wait``, ``project_question_pointer``) carries its
    source row's identity; a synthesized one (``origin_projected``, ``lifecycle_virtual``) has none."""

    text: Required[str]
    role: Required[Literal["user", "assistant", "system"]]
    ts: Required[str]  # the stored row's time, the same string its live frame carried
    is_progress: Required[bool]
    system_type: Required[str]  # the row's kind; "" for an ordinary message
    task_id: Required[str]
    presence_contact: PresenceContact
    place_history: PlaceHistoryFacts
    wait_ended_at: str
    row_address: str
    row_client_message_id: str  # a row's own saved name (outgoing rows; host-fact ingress such as a call's opening)
    reply_to: ChatReplyReference
    relay: Dict[str, Any]
    reply_to_relay: str
    room_revisions: Dict[str, int]  # a row shown in several rooms: the revision it took in each room it counted in
    history_id: str  # "<source>:<offset>": the stored row's identity in this core's retained history
    history_position: ChatHistoryPosition
    markdown: bool
    source: str
    sender_label: str
    sender_session_id: str
    subagents_mode: Literal["swarm"]  # Effective room mode at owner-message acceptance.
    client_message_id: str  # correlation id; owner rows have role=user, quiz publications also use it
    ingress_accepted: bool
    ingress_dispatched: bool
    ingress_pending: bool
    attachments: List[ChatAttachmentView]
    text_placeholder: bool
    chat_annotation: Dict[str, Any]  # the routing outcome shown on an owner message
    origin_id: str
    origin_projected: bool  # a Project's start message rebuilt from its binding's own copy
    msg_type: Literal["document", "photo", "video", "links", "quiz", "card"]
    filename: str
    mime: str
    download_url: str
    download_url_compat: str
    caption: str
    size_bytes: int
    actions: List[Dict[str, str]]  # LinkAction rows: {label, url}
    title: str
    quiz: ChatHistoryQuiz
    card: ChatHistoryCard
    summary_kind: str
    historical_terminal: Dict[str, Any]
    model_waits: Dict[str, Any]  # system_type task_model_wait: wait_id -> the wait's latest revision
    lifecycle_virtual: bool  # the running skill operation, rebuilt on every read, never stored
    review_group: Dict[str, Any]  # folded skill review attempts of one task-bound group
    executions: List[Dict[str, Any]]
    surface: str
    review_fingerprint: str
    state_revision: str
    presentation_owner_task_id: str
    group_id: str
    skill: str
    content_hash: str
    job_id: str
    job_status: str
    terminal_reason: str
    origin_task_id: str
    origin_root_task_id: str
    replayed_from_ts: str
    review_round: int
    snapshot_attempt: int
    snapshot_revised: bool
    history_retention: Dict[str, Any]
    tool_errors: Optional[int]
    routing_tool_calls: Optional[int]
    completion_tool_calls: Optional[int]
    tool_call_counts: Optional[Dict[str, Any]]
    addressing_only: str
    outcome_phase: str
    outcome_final: bool
    checkpoint_kind: str
    episode_id: str
    phase: str
    elapsed_sec: Any
    tool_evidence: Dict[str, Any]  # system_type task_evidence: the task's recorded tool calls
    cancel_origin: Dict[str, Any]
    continuation_offer: Dict[str, Any]  # ContinuationOffer of an interrupted root


class ChatHistoryCoverage(TypedDict):
    """``coverage`` v1: which coordinates this response delivered, per stream. A span is
    ``{from, to, chain, gaps}``: the coordinate range ``[from, to)`` of the stream, the chain
    witness and the gaps inside it (a line the import could not store, a segment it could not
    import); null when the stream was not read. Stored order counts bytes under the one chain
    ``dialog``. A room with imported history counts its chat rows in time order under
    ``dialog-chrono-<end>`` (``-at-<position>`` around a row), and its span adds ``keys``
    ``{from, to, frozen}``: the reading key ``[time, order, position]`` of the first frozen row above
    each cut (null: the beginning or the present) and the frozen end (CLIENT_CONTRACT §5). Coverage
    never proves the conversation's start."""

    v: int
    view: str
    upper: Dict[str, int]  # per stream: the end coordinate of this read's frozen rows
    spans: Dict[str, Optional[Dict[str, Any]]]  # per stream: {from, to, chain, gaps, keys?} or null


class ChatHistoryWindow(TypedDict, total=False):
    """Bounds of this response. ``complete`` is true only when no quota, page, cap, gap or the
    archive import bounded it; ``truncated_by`` names each cause (quota, page, lineage_cap,
    ``<stream>_<gap>`` such as chat_malformed_jsonl, or ``<stream>_importing``: older rows exist
    below the archive import's frontier and are not readable yet; the older page that would cross
    it answers 503 migrating). ``latest_message`` (Project rooms) names the latest eligible
    non-owner standalone message by stored arrival, excluding card content; null means unknown.
    Its ``out_of_order`` flag means it sorts
    above earlier arrivals or before the recent read. With a null message, ``latest_before`` says
    the bounded search found none at or after that chat offset; older pages continue the search.
    ``latest_absent`` means the search reached the chat's start without a gap and found no eligible
    message below the frozen upper bound, not that the chat has no stored rows."""

    complete: Required[bool]
    truncated_by: Required[List[str]]
    latest_message: Optional[Dict[str, Any]]  # {history_id, out_of_order}
    latest_before: int
    latest_absent: bool


class ChatHistoryAnchor(TypedDict, total=False):
    """``around``: a task's card or an exact stored row. ``found`` binds ``history_id``; ``moved``
    names a row's current ``chat_id``. Row misses distinguish absent (both sources complete),
    ambiguous and unavailable (no visible active home). Source failures/imports answer 503.
    A row has ``row_address``; a task has ``task_id``. Neither may substitute for the other."""

    state: Required[str]
    task_id: str
    row_address: str
    chat_id: int
    history_id: str


class ChatHistoryResponse(TypedDict, total=False):
    """``GET /api/chat/history``: conversation rows oldest first; replay evidence may follow them
    regardless of timestamp and recur across pages. An error answer (400 history_cursor_invalid or
    history_around_invalid, 409 history_view_changed or history_delta_reset, 503 history_source_unavailable
    or migrating) keeps this shape with no rows, ``has_more`` true and the cursor that was sent as ``next_cursor``."""

    messages: Required[List[ChatHistoryMessage]]
    has_more: Required[bool]
    next_before_ts: str  # legacy, no longer sent
    next_cursor: Required[Optional[str]]  # pass back as ?cursor= for the next older page
    page_cursor: Optional[str]  # re-reads this same page
    after_cursor: str  # the newest page and every ``after`` read: pass back as ?cursor= for what arrived since
    window: Required[ChatHistoryWindow]
    coverage: ChatHistoryCoverage
    error: str
    anchor: ChatHistoryAnchor  # an ``around`` read only
    reason_code: str


__all__ = [
    "ChatHistoryPosition", "PresenceContact", "PlaceHistoryAttention", "PlaceHistoryImport", "PlaceHistoryFacts",
    "ChatHistoryQuiz", "ChatHistoryCard", "ChatHistoryMessage",
    "ChatHistoryCoverage", "ChatHistoryWindow", "ChatHistoryAnchor", "ChatHistoryResponse",
]
