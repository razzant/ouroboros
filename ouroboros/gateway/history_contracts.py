"""Descriptive Chat history response and its rows, re-exported by the gateway facade.

``GET /api/chat/history`` projects stored ``chat.jsonl``/``progress.jsonl`` rows plus a few
synthesized ones (``gateway/history.py``). The JSDoc twin is ``web/modules/client_contract_types.js``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ouroboros.gateway.attachment_contracts import ChatAttachmentView

try:
    from typing import Literal, Required, TypedDict
except ImportError:  # pragma: no cover - Python 3.10 compatibility.
    from typing_extensions import Literal, Required, TypedDict


class ChatHistoryPosition(TypedDict):
    """Where a stored row lies: its stream and the row's byte offset in that stream's retained
    chain. It names a stored record, never current task or review authority."""

    source: Literal["chat", "progress"]
    offset: int


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
    host_facts: str
    answered_index: int
    comment: str
    owner_wait_state: str
    owner_wait_resume_reason: str


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
    row_address: str
    reply_to: Dict[str, Any]
    relay: Dict[str, Any]
    reply_to_relay: str
    history_id: str  # "<source>:<offset>": the stored row's identity in this core's retained history
    history_position: ChatHistoryPosition
    markdown: bool
    source: str
    spoken: Dict[str, Any]  # live-call item identity, playback position and generated/played/interrupted status
    sender_label: str
    sender_session_id: str
    client_message_id: str  # correlation id; owner rows have role=user, quiz publications also use it
    ingress_accepted: bool
    ingress_dispatched: bool
    ingress_pending: bool
    ingress_undispatched: bool
    attachments: List[ChatAttachmentView]
    text_placeholder: bool
    chat_annotation: Dict[str, Any]  # the routing outcome shown on an owner message
    origin_id: str
    origin_projected: bool  # a Project's start message rebuilt from its binding's own copy
    msg_type: Literal["document", "photo", "video", "links", "quiz"]
    filename: str
    mime: str
    download_url: str
    download_url_compat: str
    caption: str
    size_bytes: int
    actions: List[Dict[str, str]]  # LinkAction rows: {label, url}
    title: str
    quiz: ChatHistoryQuiz
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
    """``coverage`` v1: which stored bytes this response delivered, per stream. A span is
    ``{from, to, chain, gaps}``: the byte range ``[from, to)`` of the stream's retained chain,
    that chain's witness and the read gaps inside it; null when the stream was not read or is
    unreadable. Coverage is byte delivery, never chronology or proof of the conversation's start."""

    v: int
    view: str
    upper: Dict[str, int]  # per stream: the frozen end of this read
    spans: Dict[str, Optional[Dict[str, Any]]]  # per stream: {from, to, chain, gaps} or null


class ChatHistoryWindow(TypedDict, total=False):
    """Bounds of this response. ``complete`` is true only when no quota, page, cap or read gap
    bounded it; ``truncated_by`` names each cause (quota, page, lineage_cap, or ``<stream>_<gap>``
    such as chat_source_unavailable). ``latest_message`` (Project rooms) names the latest eligible
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
    """What an ``around=task:<id>`` read found: ``found`` (its page lands on ``history_id``), ``absent``
    (the thread's saved history holds no row of the task a client could show) or ``moved`` (another
    thread's)."""

    state: Required[str]
    task_id: Required[str]
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
    "ChatHistoryPosition", "ChatHistoryQuiz", "ChatHistoryMessage",
    "ChatHistoryCoverage", "ChatHistoryWindow", "ChatHistoryAnchor", "ChatHistoryResponse",
]
