"""``/api/state`` work projection contracts (v1): one row per work, one rollup per room, and the owner's model and
tier writes those rows and rooms carry (re-exported by ``client_contracts``, twinned in ``client_contract_types.js``).

Descriptive, not runtime validation. The browser twin is ``web/modules/work_types.js``; the frozen census
(``active_chat_activities``) stays beside these fields unchanged (docs/architecture/11-frozen-contracts-v1.md).
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

try:  # Python 3.11+
    from typing import NotRequired, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import NotRequired, TypedDict  # type: ignore[assignment]


class ChatModelRoute(TypedDict):
    """A model route shared by room choices, Main and owner recents."""

    model: str
    use_local: bool


class CitedModelWords(TypedDict):
    """Up to 1024 Unicode characters of the verified span; clipped says whether more was retained."""

    text: str
    clipped: bool


class WorkPrimaryRoute(ChatModelRoute):
    basis: str
    cited: NotRequired[CitedModelWords]


class WorkPendingRoute(ChatModelRoute):
    """The next route's scope, when known separately from the acting route's."""

    basis: NotRequired[str]


class WorkLead(WorkPrimaryRoute):
    """Acting route and chooser; a departure names the primary it left."""

    role: NotRequired[str]
    primary: NotRequired[WorkPrimaryRoute]
    since: NotRequired[str]
    cause: NotRequired[str]
    pending: NotRequired[WorkPendingRoute]


ProcessingTier = Literal["", "standard", "fast", "economy", "ultrafast"]  # '' = no tier of its own: the default resolution


class ProcessingTiers(TypedDict):
    """Main's own tier ('' = none: Main takes the default) and the default every lead pays (Settings values)."""

    main: ProcessingTier
    default: Literal["", "standard", "fast", "economy"]


class ProcessingReceipt(TypedDict):
    """One call's processing as its receipt states it: what was asked, sent and seen (``_usage_response``), the
    provider's own words beside the folded ones (an older Codex reports ``ultrafast`` it folds into ``fast``)."""

    requested: Optional[str]
    submitted: Optional[str]
    submittedNative: NotRequired[Optional[str]]
    observed: str  # standard | fast | economy | ultrafast | mixed | unknown
    observedNative: NotRequired[List[str]]
    reason: Optional[str]  # processing_not_submitted, submitted_mode_differs, provider_mode_changed


class WorkProcessing(TypedDict):
    """A root's own tier (its room's at admission, or the owner's card act; '' = the default resolution) and the
    newest call's receipt this host saw, so a choice is never shown as what ran."""

    choice: ProcessingTier
    resolved: NotRequired[ProcessingTier]  # server resolver, including the acting fallback row
    source: NotRequired[Literal["room", "owner"]]
    last: NotRequired[ProcessingReceipt]


class WorkLeadFields(TypedDict):
    lead: NotRequired[WorkLead]
    processing: NotRequired[WorkProcessing]


class TaskModelRequest(TypedDict):
    request_id: str
    action: Literal["set", "follow", "return", "wait", "keep", "processing"]
    model: NotRequired[str]
    use_local: NotRequired[bool]
    processing: NotRequired[ProcessingTier]  # action ``processing`` only: the work's own tier from its next call


class TaskModelResponse(TypedDict):
    ok: Literal[True]
    task_id: str
    applied: Literal["next_step", "at_start"]
    lead: NotRequired[WorkLead]  # every model act
    processing: NotRequired[WorkProcessing]  # a tier act


class ChatModelChoice(TypedDict):
    """A room follows Main or chooses a route; a reference is never the same model pinned."""

    kind: Literal["main", "model"]
    model: NotRequired[str]  # kind ``model`` only: the raw route, its source included
    use_local: NotRequired[bool]  # kind ``model`` only


class MainBindingConditions(TypedDict):
    """``GET /api/settings`` read tokens of Main's binding as the owner was shown it, compared under the writer's lock
    (``409:settings_changed_elsewhere`` when one moved); ``connection`` is the shown ``_meta.connections[].revision``
    of the API connection the written route sends to."""

    OUROBOROS_MODEL: str
    USE_LOCAL_MAIN: str
    OUROBOROS_MODEL_ACCOUNTS: str
    OUROBOROS_MODEL_PROCESSING_PREFERENCES: str
    ultrafast: NotRequired[str]  # confirmed JSON [model, local, pin, store, fingerprint]; Auto binds no membership
    connection: NotRequired[str]  # required where Ultrafast is written or carried on an API connection


class OwnerChatModelRequest(ChatModelChoice):
    """``POST /api/owner/chat-model``: a Project follows Main or chooses a model; without project_id this writes
    Main itself, on the binding its ``if_unchanged`` names when sent (Main only)."""

    project_id: NotRequired[str]
    if_unchanged: NotRequired[MainBindingConditions]


class OwnerChatModelResponse(TypedDict):
    ok: bool
    project_id: str
    model_choice: NotRequired[ChatModelChoice]  # Project response
    previous_model_choice: NotRequired[ChatModelChoice]
    main_route: NotRequired[ChatModelRoute]  # Main response
    previous_main_route: NotRequired[ChatModelRoute]
    account: NotRequired[Literal["kept", "auto", "none"]]
    window_reset: NotRequired[bool]


class OwnerChatProcessingRequest(TypedDict):
    """A room's processing tier for new owner work, or Main's own without project_id. An empty tier is Default;
    ``if_unchanged`` belongs to Main's Ultrafast alone."""

    processing: ProcessingTier
    project_id: NotRequired[str]
    if_unchanged: NotRequired[MainBindingConditions]


class OwnerChatProcessingResponse(TypedDict):
    ok: bool
    project_id: str
    processing_choice: NotRequired[ProcessingTier]  # Project response
    previous_processing_choice: NotRequired[ProcessingTier]
    main_processing: NotRequired[ProcessingTier]
    previous_main_processing: NotRequired[ProcessingTier]
    processing_default: NotRequired[Literal["", "standard", "fast", "economy"]]  # what Default resolves to


class WorkReason(TypedDict):
    """One typed reason: ``code`` is closed per release, ``text`` its English sentence (unknown codes stay raw)."""

    code: str
    text: str
    count: NotRequired[int]
    ids: NotRequired[List[str]]
    since: NotRequired[Any]
    by: NotRequired["PauseAuthor"]
    wake: NotRequired[List["SleepWake"]]
    behind: NotRequired["QueuedBehind"]


class QueuedBehind(TypedDict):
    """The work a ``queued`` reason stands behind (``work_status.queued_reason``): its tree's id, the ``lane``
    (``project`` or ``folder``, the latter with its ``folder`` path) and that work's ``name`` as the list shows it."""

    task_id: str
    lane: str
    name: str
    folder: NotRequired[str]


class PauseAuthor(TypedDict):
    """The declared author of a decided Pause (``control_actor.pause_author``): ``owner`` or ``ouroboros``, with the
    words the press recorded (``""`` for a press without words), said verbatim after its head."""

    actor: str
    text: str
    at: Any


class SleepWake(TypedDict):
    """One clause of what a ``sleeping`` reason waits for (``work_status.sleep_wake``): a ``kind`` of its own selection,
    with a task's census ``name``, a ``count`` or the chosen instant ``at`` (said in the viewer's clock). A reason's
    clauses are joined by "or", its time last."""

    kind: str  # task | task_or_writes | tasks | all_tasks | runs | service | reviewer | sender | mail | time
    name: NotRequired[str]
    count: NotRequired[int]
    at: NotRequired[str]


class LastResumeBy(TypedDict):
    """Who asked, as the door bound it (``work_stop.resume_asked``): ``owner``, ``ouroboros``, or ``host`` for the
    host's own start of a Continue whose earlier work settled."""

    actor: str


class LastResume(TypedDict):
    """A held episode's latest refused Resume about the control holding it now (``work_stop.last_resume``): dated
    history beside the reason, never a door or proof that a blocker still exists; the host's refused start is the
    reason of a waiting Continue (``start_refused``)."""

    at: Any  # when it was asked
    by: LastResumeBy
    error: str  # the door's typed refusal
    text: str  # the door's own words after "Resume refused: "


class WorkTail(TypedDict):
    """What still runs beside or after the current episode, said as a second fact, never as the work's word."""

    kind: str  # helper | run | review | model
    state: str  # live | unknown | open (an observed inactive run) | held (a paused helper)
    code: str
    text: str
    count: NotRequired[int]
    ids: NotRequired[List[str]]


class WorkTurn(TypedDict):
    """What waits for the owner: ``question`` (a task waits for the answer), ``open_question`` (a current question
    Main's pill counts while its work goes on, after it ended, or one held by no work at all) or ``action`` (only the
    owner takes it). ``code`` ``delivery_unknown`` marks an ``open_question`` that was answered but whose delivery a
    restart left unknown. In a room's ``turn`` an item a work holds names its ``work_id``; one no work holds has none."""

    kind: str  # question | open_question | action
    work_id: NotRequired[str]  # in a room's turn: the work that holds it
    task_id: str
    quiz_id: NotRequired[str]
    wait_id: NotRequired[str]
    code: NotRequired[str]


class WorkControl(TypedDict):
    """An action the named door accepts now, on its exact target."""

    action: str
    target: str
    scope: NotRequired[str]


class WorkEpisode(TypedDict):
    task_id: str
    live: bool


class WorkMemberState(TypedDict):
    """One task's own activity. Concurrent execution is reported separately in its addressed tails."""

    activity: Literal["working", "queued", "waiting", "paused", "unknown"]
    transition: Literal["none", "finalizing", "pausing"]
    call: Literal["", "preparing", "sending", "thinking", "working"]
    live: bool  # this task executes, independent of its external runs
    release: Literal["self", "owner", "mind", "none"]
    confirmed: bool
    turn: List[WorkTurn]
    because: List[WorkReason]
    tails: List[WorkTail]  # run/review IDs are external run IDs; model IDs are attempt IDs
    mark: str  # the member's own mark, independent of its tails
    queue_attempt: NotRequired[int]  # observed queue ordinal of this member's state, not an executor ordinal
    last_resume: NotRequired[LastResume]


class WorkMembers(TypedDict):
    ids: List[str]
    states: NotRequired[Dict[str, WorkMemberState]]


class WorkFocus(TypedDict):
    text: str
    at: Any


class Work(WorkLeadFields):
    """One request with its Continue episodes and live members."""

    work_id: str
    room: Any
    name: str
    kind: str
    project_id: str
    episode: WorkEpisode
    activity: str
    transition: str
    call: str
    because: List[WorkReason]
    tails: List[WorkTail]
    live: bool
    turn: List[WorkTurn]
    release: str
    members: WorkMembers
    controls: List[WorkControl]
    since: Any
    confirmed: bool
    mark: str
    stop: NotRequired[Dict[str, Any]]
    refusal: NotRequired[Dict[str, Any]]
    last_resume: NotRequired[LastResume]  # also on each held member's state
    outcome: NotRequired[str]
    focus: NotRequired[WorkFocus]
    gaps: NotRequired[List[Dict[str, Any]]]
    client_message_id: NotRequired[str]
    place_turn: NotRequired[bool]  # a correspondent's Presence turn: Telegram's owner list leaves it out
    unbound_start: NotRequired[bool]  # an accepted delegate start no engine run answers yet: its Stop is a request
    # taken against the recorded request and the tree, never against a remote run handle; the remote start's true
    # fate stays unknown until a stop is confirmed (#3428; ``task_control_menu.js``). Said only when true.


class RoomStatus(TypedDict):
    """One room's status from its works: the most active work speaks; ``amber`` while something waits for the owner.
    ``turn`` holds its works' items (with ``work_id``) and the room's own current questions no work holds (without):
    a room at rest can be the owner's turn."""

    activity: str
    transition: str
    call: str
    turn: List[Dict[str, Any]]
    amber: bool
    mark: str
    confirmed: bool
    works: List[str]


class WorkSummary(TypedDict):
    counts: Dict[str, int]
    turn: int  # what waits for the owner: each work with a turn, and each standalone question no counted work carries
    confirmed: bool


class WorkStateFields(TypedDict):
    """The work projection's ``/api/state`` fields, inherited by ``contracts.StateResponse``."""

    works: List[Work]
    rooms: Dict[str, RoomStatus]
    work_summary: WorkSummary
    works_complete: bool
    works_gaps: List[Dict[str, Any]]
