"""``/api/state`` work projection contracts (v1): one row per work, one rollup per room, and the owner's model and
tier writes those rows and rooms carry (re-exported by ``client_contracts``).

Descriptive, not runtime validation. The frozen census
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


# The owner's cited words or pressed label, kept as authored (CLIENT_CONTRACT §5).
class CitedModelWords(TypedDict):
    """Up to 1024 Unicode characters of the verified span; clipped says whether more was retained."""

    text: str
    clipped: bool


# A work's primary route and whose choice it is.
class WorkPrimaryRoute(ChatModelRoute):
    basis: str  # main | a role it follows | room | owner | owner_words | owner_press | mind | fallback | place | circle.
    cited: NotRequired[CitedModelWords]


class WorkPendingRoute(ChatModelRoute):
    """The next route's scope, when known separately from the acting route's."""

    basis: NotRequired[str]


# Departures by fallback, mind switch or owner wait-card switch carry `since` and `cause`, with
# `basis` respectively `fallback`, `mind` or `owner`. A wait-card switch has `cause=wait_card`.
class WorkLead(WorkPrimaryRoute):
    """Acting route and chooser; a departure names the primary it left."""

    role: NotRequired[str]  # a fallback's chain row (`fallback:<n>`, n from 0): tells apart two rows of one model.
    primary: NotRequired[WorkPrimaryRoute]
    since: NotRequired[str]  # an ISO instant in UTC.
    cause: NotRequired[str]  # a short code or phrase: `switch_model`, a failure's code, `configured fallback`.
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


# One act on a work's model (POST /api/tasks/{task_id}/model): `set` gives it a model, `follow` returns
# it to Main; `return`, `wait` and `keep` leave a fallback; `processing` sets its own tier from its next
# call ('' = Default).
class TaskModelRequest(TypedDict):
    request_id: str  # a retry of the same act carries the same id.
    action: Literal["set", "follow", "return", "wait", "keep", "processing"]
    model: NotRequired[str]  # `set` only.
    use_local: NotRequired[bool]  # `set` only.
    processing: NotRequired[ProcessingTier]  # action ``processing`` only: the work's own tier from its next call


class TaskModelResponse(TypedDict):
    ok: Literal[True]
    task_id: str
    applied: Literal["next_step", "at_start"]  # a running work switches at its next step, a queued one starts on it.
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


# Without `project_id` only kind `model` is a choice.
class OwnerChatModelRequest(ChatModelChoice):
    """``POST /api/owner/chat-model``: a Project follows Main or chooses a model; without project_id this writes
    Main itself, on the binding its ``if_unchanged`` names when sent (Main only)."""

    project_id: NotRequired[str]
    if_unchanged: NotRequired[MainBindingConditions]


# A Project's answer carries its choice; Main's own carries what Main is and was, what became of its
# pinned account (`kept`, dropped to `auto`, or `none` on a route without accounts) and whether a window
# set by hand was cleared.
class OwnerChatModelResponse(TypedDict):
    ok: bool
    project_id: str  # '' for Main.
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
    project_id: str  # '' for Main.
    processing_choice: NotRequired[ProcessingTier]  # Project response
    previous_processing_choice: NotRequired[ProcessingTier]
    main_processing: NotRequired[ProcessingTier]  # Main response: Main's own tier (a Settings value).
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
    at: Any  # the press time.


class SleepWake(TypedDict):
    """One clause of what a ``sleeping`` reason waits for (``work_status.sleep_wake``): a ``kind`` of its own selection,
    with a task's census ``name``, a ``count`` or the chosen instant ``at`` (said in the viewer's clock). A reason's
    clauses are joined by "or", its time last."""

    kind: str  # task | task_or_writes | tasks | all_tasks | runs | service | reviewer | sender | mail | time
    name: NotRequired[str]  # a chosen task's or sender's work name, as the census holds it.
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


# `stop_action_id` is at most 200 characters; a later Stop needs a new one.
class TaskCancelRequest(TypedDict, total=False):
    """Optional body; stop_action_id retries one action, distinct from custody ID.
    Omission preserves legacy ingress without exact-replay assurance."""
    cascade: bool
    stop_policy: str  # "immediate" | "finalize_then_cancel" | "close_expired".
    stop_action_id: str
    close_expired: str  # only with stop_policy="close_expired", no cascade or Stop action id


class WorkControl(TypedDict):
    """An action the named door accepts now, on its exact target."""

    action: str  # stop | stop_earlier | pause | resume | hurry | wrap_up | model | close.
    target: str  # the exact task id the door acts on.
    scope: NotRequired[str]  # single | cascade | work.
    close_expired: NotRequired[str]  # exact park offer, only for close.


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
    close_expired: NotRequired[WorkControl]


class WorkMembers(TypedDict):
    ids: List[str]  # every member task id (episode roots and helpers): a card of another id re-reads the state.
    states: NotRequired[Dict[str, WorkMemberState]]  # keyed by the exact member task id.


class WorkFocus(TypedDict):
    text: str
    at: Any


class Work(WorkLeadFields):
    """One request with its Continue episodes and live members."""

    work_id: str
    room: Any
    name: str
    kind: str  # direct | task.
    project_id: str
    episode: WorkEpisode
    activity: str  # working | queued | waiting | paused | unknown.
    transition: str  # none | finalizing | pausing.
    call: str  # preparing | sending | thinking | working, when every working member agrees.
    because: List[WorkReason]  # the current episode's own reason (or, once it ended, nothing).
    tails: List[WorkTail]
    live: bool  # something of the work actually executes now.
    turn: List[WorkTurn]
    release: str  # self | owner | mind | none.
    members: WorkMembers
    controls: List[WorkControl]
    since: Any
    confirmed: bool
    mark: str  # moving | still | pause | ring | none (docs/DESIGN.md "Status marks").
    stop: NotRequired[Dict[str, Any]]  # current stop and any independent member stop.
    refusal: NotRequired[Dict[str, Any]]  # the continuation's last refusal, rechecked by Resume.
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

    activity: str  # working | queued | waiting | paused | idle | unknown.
    transition: str
    call: str
    turn: List[Dict[str, Any]]
    amber: bool
    mark: str  # moving | still | pause | ring | none.
    confirmed: bool
    works: List[str]
    deadline_expired: NotRequired[bool]


class WorkSummary(TypedDict):
    counts: Dict[str, int]
    turn: int  # what waits for the owner: each work with a turn, and each standalone question no counted work carries
    confirmed: bool


class PanicRecoveryMember(TypedDict):
    """One saved queue member with an opaque hold identity for explicit owner selection."""

    task_id: str
    task_name: str
    panic_hold_id: str
    close_expired: NotRequired[WorkControl]
    project_id: NotRequired[str]
    chat_id: NotRequired[int]
    root_task_id: NotRequired[str]


class WorkStateFields(TypedDict):
    """Work and recovery fields of ``/api/state``, inherited by ``contracts.StateResponse``."""

    works: List[Work]
    rooms: Dict[str, RoomStatus]  # keyed by chat_id
    work_summary: WorkSummary  # the tray's one line.
    works_complete: bool  # false: a source was unreadable; keep what was seen, never read it as empty.
    works_gaps: List[Dict[str, Any]]  # typed gaps of the work projection.
    panic_recovery: NotRequired[List[PanicRecoveryMember]]  # global pending members, not Main's attention filter
    panic_recovery_complete: NotRequired[bool]  # complete post-readiness census with custody of the saved queue; false or absent: do not offer to return all work
