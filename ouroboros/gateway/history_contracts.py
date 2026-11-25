"""Descriptive Chat history response and its rows, re-exported by the gateway facade.

``GET /api/chat/history`` projects the dialogue store's stored chat and progress rows
(``state/dialog.sqlite``) plus a few synthesized ones (``gateway/history.py``); ``GET /api/chat/search``
reads a room's saved messages for a word search (``gateway/chat_search.py``); ``GET /api/recall`` is the
neighbour door over the words reader (``gateway/recall.py``; its JSDoc twins are
``web/modules/recall_contract_types.js``). The JSDoc twin of the rest is ``web/modules/client_contract_types.js``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from ouroboros.gateway.attachment_contracts import ChatAttachmentView
from ouroboros.gateway.reference_contracts import ChatReplyReference, HostNoticeField

try:
    from typing import Literal, Required, TypedDict
except ImportError:  # pragma: no cover - Python 3.10 compatibility.
    from typing_extensions import Literal, Required, TypedDict


class RequestedExecutorRoute(TypedDict):
    """The executor a subagent selected as a card shows it: no local connection name, not the launch address."""

    kind: Literal["api_model", "agent_session"]
    target_id: str


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
    expired_terminal or superseded. ``answered_index`` and ``comment`` are the recorded answer,
    ``answer_row_address`` its own row, which the card shows (DESIGN "Replies")."""

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
    answer_row_address: str
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
    recipient: str  # the work or author holding a heard message (``place_intake``); whether its model read it is unknown


class PlaceHistoryImport(TypedDict):
    """An earlier message its connector imported: who sent it on the platform and when the Host
    received it; the row's ``ts`` is the provider's send time."""

    authorship: Literal['participant', 'platform_account']
    received_at: str


class PlaceHistoryReaction(TypedDict):
    """The host's English words for a row's typed reaction, never speech; words the row sent stay its ``text``."""

    summary: str


class PlaceHistoryFacts(TypedDict, total=False):
    """Retained room speech/notices only. Absence is unknown, never delivered or woken."""

    kind: Required[Literal['message', 'notice', 'delivery']]
    delivery_state: Literal['authored', 'accepted', 'delivered', 'failed', 'uncertain']
    attention: PlaceHistoryAttention
    actor_origin: Literal['machine']  # an incoming program, as recorded by the connector; absent carries no origin claim
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
    reaction: PlaceHistoryReaction  # a typed reaction arrival or receipt; absent: not one
    non_text: List[Dict[str, str]]  # the connector's {kind, description} beside the words (kind state_change: a change), never speech


class HistoricalTerminal(TypedDict, total=False):
    """Facts frozen on one published ending. Its publication time and observed end time are separate."""

    status: Required[str]
    phase: Required[Literal["done", "warn", "error", "cancelled"]]
    ts: Required[str]
    provenance: Required[Literal["canonical_task_result_after_finalization"]]
    model_execution: Dict[str, Any]
    task_attempt: int
    terminal_time: Dict[str, Any]
    terminal_delivery: Dict[str, Any]
    wake_summary: str
    suggested_name: str
    schedule_name: str
    place_origin: Dict[str, Any]
    wake_silent: bool
    quiet_effective: bool
    schedule_report_only: bool


class DeclaredAuthor(TypedDict, total=False):
    """An agent's note's author as the host recorded it (``dialogue_provenance.recorded_author``): the label it declared,
    never verified (``basis: "self_declared"``); ``link_key`` with ``key_id`` is reserved for a host-attested key."""

    kind: Required[Literal["agent"]]
    label: Required[str]
    basis: Required[Literal["self_declared", "link_key"]]
    key_id: str


class AgentNoteFields(TypedDict, total=False):
    """The program's author and optional letter id, on both the live echo and its history row."""

    declared_author: DeclaredAuthor  # an agent's note in Main: never the owner's words
    late_report_id: str  # the letter's display id, not the host return reference


class ChatHistoryMessage(HostNoticeField, AgentNoteFields, total=False):
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
    kind: str  # room_echo for a program's message saved in its recipient's room
    row_address: str
    reply_excerpt_disabled: bool
    reply_author_label: str  # whom a Reply to this row names: You, Ouroboros, System or the recorded speaker
    row_client_message_id: str  # a row's own saved name (outgoing rows; host-fact ingress such as a call's opening)
    reply_to: ChatReplyReference
    relay: Dict[str, Any]
    reply_to_relay: str
    room_revisions: Dict[str, int]  # a row shown in several rooms: the revision it took in each room it counted in
    history_id: str  # "<source>:<offset>": the stored row's identity in this core's retained history
    history_position: ChatHistoryPosition
    markdown: bool
    source: str
    sender_label: str  # an agent's note: its rendered identity (``agent "<label>" (declared, not verified)``)
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
    source_row_address: str  # the physical original of a retained copy, even outside this store
    origin_projected: bool  # a Project's start message rebuilt from its binding's own copy
    opening_input_kind: Literal["owner_press", "owner_card_input", "owner_quiz_answer", "owner_mailbox"]
    chat_id: int  # on that copy: the room its original was written in (its receipt's room)
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
    schedule_report_only: bool  # terminal card links to a saved report, not a transcript message
    status: str
    task_terminal_status: str
    task_phase: str
    task_attempt: int
    task_outcome: Union[str, Dict[str, Any]]  # a legacy status or the recorded ending
    historical_terminal: HistoricalTerminal
    current_outcome: Dict[str, Any]  # separately scoped by its own task_attempt and terminal_time when known
    terminal_time: Dict[str, Any]
    terminal_delivery: Dict[str, Any]  # scope="task" for unbound debt, preserved_path=None for an unverified copy
    reason_code: str
    outcome_axes: Dict[str, Any]
    _result_text_present: bool
    delegated_runs_unreconciled: List[str]
    review_projection: Dict[str, Any]
    wake_summary: str
    suggested_name: str
    schedule_name: str
    place_origin: Dict[str, Any]
    wake_silent: bool
    quiet_effective: bool
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
    finalization_recovery: Dict[str, Any]  # an adopted answer's custody: answer, cause, delivery_id, attempt
    terminal_origin: str
    continuation_offer: Dict[str, Any]  # ContinuationOffer of an interrupted or author-stopped root


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
    it answers 503 migrating). ``latest_message`` (Main and Project rooms) names the latest eligible
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
    or migrating) keeps this shape with no rows, ``has_more`` true and the cursor that was sent as ``next_cursor``;
    a 503 history_source_unavailable may name the store's cause in ``detail``. ``project_registry`` is Main's
    alone: ``"unavailable"`` on its refusal while the registry cannot tell its rows from a Project's (its read-only
    archive is ``GET /api/chat/archive``), ``"restored"`` on a successful newest page after the registry came back
    from its saved copy; an absent field says neither."""

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
    detail: str  # a 503 history_source_unavailable: the store's cause, when it has one
    project_registry: Literal["unavailable", "restored"]  # Main only: its refusal, or the list back from its copy
    profile_scope: str  # the serving generation's frozen data-profile scope, on a successful page; the
                        # browser binds a delayed body to it before consuming it (#3294)


class ChatSearchAttachment(TypedDict):
    """An owner attachment as its bubble draws it (history's ``ChatAttachmentView``, no address): its file card or
    player shows ``name``, an available image only its picture; the composer's file-name tail the bubble hides is
    built from the names."""

    name: str
    kind: Literal["image", "video", "audio", "file"]
    available: bool


class ChatSearchLink(TypedDict):
    """A link button of a delivered links row: its label is shown, its address only decides whether it is."""

    label: str
    url: str


class ChatSearchQuestion(TypedDict, total=False):
    """A question row's card as its words are drawn (the question is the row's ``text``): the offered answers'
    labels and details (``QuizOption`` rows, or a legacy row's labels with ``option_details``), the stake and the
    owner's own words (``comment``) once the card is settled as answered."""

    quiz_id: Required[str]
    options: List[Any]
    option_details: List[str]
    stake: str
    comment: str


class ChatSearchCandidate(HostNoticeField, AgentNoteFields, total=False):
    """One saved message of the room as the fields the room's renderer draws its words from: the text history
    gives it (``system_type``, ``markdown``, ``narration``, ``source`` and the owner's ``attachments`` names choose
    how) or a host notice's line (``notice``; its details, template and values are not searched), a photo, video or file's ``caption`` and a file's ``filename``, a links row's ``title`` and ``actions``, a
    question's ``quiz``, a Project lifecycle row's answer and ending (``completion_answer``, ``project_id``,
    ``project_name``, ``task_name``, ``completion_notice``). An agent's note carries its literal body as ``text`` under
    its ``declared_author`` (``role`` ``user``), as history shows it, never the neutral receipt fact. Whether its words
    contain the query is the client's reading of those words. ``history_id`` and ``row_address`` are the row's identity in
    ``GET /api/chat/history``, whose ``around=<row_address>`` reads the page around it; ``role`` is its author as history names it."""

    history_id: Required[str]
    row_address: Required[str]
    role: Required[Literal["user", "assistant", "system"]]
    ts: Required[str]
    text: Required[str]
    system_type: str
    markdown: bool
    narration: bool
    source: str
    sender_label: str
    task_id: str
    attachments: List[ChatSearchAttachment]
    text_placeholder: bool  # the host's words for an owner who sent only attachments: not shown
    msg_type: Literal["photo", "video", "document", "links", "quiz"]
    caption: str
    filename: str
    title: str
    actions: List[ChatSearchLink]
    quiz: ChatSearchQuestion
    completion_answer: str
    project_id: str
    project_name: str
    task_name: str
    completion_reason: str
    completion_notice: Dict[str, Any]


class ChatSearchSpan(TypedDict):
    """What one search read covered: ``rows`` stored chat rows of the room below the cut, back to the
    time of the oldest (``reached_ts``, "" when none was read). ``beginning``: it reached the room's
    first message. ``importing``: older history exists that the archive import has not made readable
    yet, so it was not searched. ``gaps``: the read range holds lines that could not be read
    (``malformed_jsonl``, ``invalid_jsonl_row``, ``segment_unimportable``) or a Project's rows written
    elsewhere not yet known (``appearances_deriving``, ``reply_target_unreadable``)."""

    rows: int
    reached_ts: str
    beginning: bool
    importing: bool
    gaps: List[str]


class ChatSearchResponse(TypedDict, total=False):
    """``GET /api/chat/search?chat_id=&q=``: one bounded read of the room's saved messages, newest first, as
    candidates whose shown words the client reads for the query. ``query`` is the query as searched (NFC, white
    space collapsed). ``has_more`` and ``next_cursor`` continue older; a page may hold no candidate and still
    continue. The search covered the whole room only when a chain of pages reached ``beginning`` with no
    ``gaps``. An error answer (400 search_query_invalid or search_cursor_invalid, 409 search_view_changed, 503
    history_source_unavailable or migrating) keeps this shape with no candidates and the cursor that was sent as
    ``next_cursor``."""

    query: Required[str]
    candidates: Required[List[ChatSearchCandidate]]
    has_more: Required[bool]
    next_cursor: Required[Optional[str]]
    searched: Required[ChatSearchSpan]
    error: str
    reason_code: str


class RecallReader(TypedDict):
    """A read the caller can run with this door alone: ``path`` on this server (``GET``), or ``cli`` as the
    arguments after the ``ouroboros`` command."""

    path: str
    cli: List[str]


class RecallField(TypedDict):
    """One typed field of a dialogue row with its own speaker: ``text`` (the message), ``question``,
    ``option <n>``, ``press``, ``comment``, ``closure`` or ``receipt`` (a secret request's non-value record,
    the only field such a card has). ``text`` is an excerpt on a search answer, whole on a reopened row.
    ``speaker`` is one of ``owner`` (a press or comment on a card), ``host``, ``task <id>``, "the asking task",
    "the closing task", or the row's author label (``RecallRow.speaker`` without its provenance) for its message."""

    name: str
    speaker: str
    text: str


class RecallRow(TypedDict, total=False):
    """One original dialogue row. ``address`` is its full content address (``row:<room>@<time>#<sha256>``);
    ``rooms`` every room it shows in, its own first; ``speaker`` the label Ouroboros's own store gives the author,
    in its own voice (a place row may read "my human: <name>"), with its provenance, and "API door, sender not
    recorded" on a row the API door stamped; ``matched`` the fields a search matched ([] on reopen);
    ``reply_to`` the recorded reply target's address, ``reply_unresolved`` when it was not resolved;
    ``reopen`` this row whole."""

    address: Required[str]
    room_id: Required[str]
    room: Required[str]
    rooms: Required[List[str]]
    ts: Required[str]
    speaker: Required[str]
    matched: Required[List[str]]
    fields: Required[List[RecallField]]
    reply_to: str
    reply_unresolved: bool
    reopen: Required[RecallReader]


class RecallSearchResponse(TypedDict):
    """``GET /api/recall?words=…``: one page of the dialogue rows that the rooms a person reads show (or one
    ``room_id``) whose words match, in the caller's ``order`` of time, never ranked. ``total`` is exact when the read was complete, else
    null with ``observed`` and the named ``gaps``; ``next`` continues the same capture (``snapshot``) and is null
    when the capture did not stabilize (``notes`` says so). ``notes`` qualify labels without touching the count."""

    mode: Literal["search"]
    words: List[str]
    room_id: str
    voice: str
    order: Literal["newest", "oldest"]
    total: Optional[int]
    observed: int
    offset: int
    more: int
    snapshot: Optional[str]
    next: Optional[RecallReader]
    gaps: List[str]
    notes: List[str]
    rows: List[RecallRow]


class RecallRowResponse(TypedDict):
    """``GET /api/recall?row=<address>``: one admitted dialogue row reopened whole; ``notes`` as on a search."""

    mode: Literal["row"]
    row: RecallRow
    notes: List[str]


class ChatArchiveRoom(TypedDict, total=False):
    """A room the archive reads: ``main``, or a historical Project room the saved records name (``project_id``,
    ``name``; ``evidence`` lists the kinds that name it: ``registry``, ``binding``, ``marker``, ``main_record``)."""

    room: Required[str]
    name: Required[str]
    chat_id: int
    project_id: str
    evidence: List[str]


class ChatArchiveMessage(TypedDict, total=False):
    """Speaker, saved words, time, pinned notice kind and attachment names. A handoff names its destination;
    a completion notice includes its stored label, status and any detail. ``id`` is a key within this read-only
    archive, not an address for replies, continuations, moves or links."""

    id: Required[str]
    role: Required[Literal["user", "assistant", "system"]]
    text: Required[str]
    ts: Required[str]
    kind: Literal["project_started", "project_handoff", "project_completion_summary"]
    attachments: List[str]


class ChatArchiveResponse(TypedDict, total=False):
    """``GET /api/chat/archive?room=&before=``: while Main's history refuses for the Project registry, one page of
    a room's rows that positive evidence places there, oldest first; ``before=<next_cursor>`` continues older,
    and a page may hold none and still continue. ``complete`` is always false: rows whose room is not proven are
    not shown, and ``limits`` names what the read could not include (``placement_unproven``,
    ``bindings_unavailable``, ``rooms_withheld_unverified``, ``legacy_origins_limited``, ``importing``),
    never a count. ``state: "not_needed"`` alone: the registry reads,
    so Main's history answers."""

    state: Required[Literal["available", "not_needed"]]
    room: str
    rooms: List[ChatArchiveRoom]
    messages: List[ChatArchiveMessage]
    has_more: bool
    next_cursor: Optional[str]
    complete: bool
    limits: List[str]


__all__ = [
    "ChatHistoryPosition", "PresenceContact", "PlaceHistoryAttention", "PlaceHistoryImport", "PlaceHistoryReaction", "PlaceHistoryFacts",
    "ChatHistoryQuiz", "ChatHistoryCard", "HistoricalTerminal", "ChatHistoryMessage",
    "ChatHistoryCoverage", "ChatHistoryWindow", "ChatHistoryAnchor", "ChatHistoryResponse",
    "ChatSearchAttachment", "ChatSearchLink", "ChatSearchQuestion", "ChatSearchCandidate", "ChatSearchSpan",
    "ChatSearchResponse", "RecallReader", "RecallField", "RecallRow", "RecallSearchResponse", "RecallRowResponse",
    "ChatArchiveRoom", "ChatArchiveMessage", "ChatArchiveResponse",
]
