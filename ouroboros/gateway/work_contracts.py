"""``/api/state`` work projection contracts (v1): one row per work, one rollup per room.

Descriptive, not runtime validation. The browser twin is ``web/modules/work_types.js``; the frozen census
(``active_chat_activities``) stays beside these fields unchanged (docs/architecture/11-frozen-contracts-v1.md).
"""
from __future__ import annotations

from typing import Any, Dict, List

try:  # Python 3.11+
    from typing import NotRequired, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import NotRequired, TypedDict  # type: ignore[assignment]


class WorkReason(TypedDict):
    """One typed reason: ``code`` is closed per release, ``text`` its English sentence (unknown codes stay raw)."""

    code: str
    text: str
    count: NotRequired[int]
    ids: NotRequired[List[str]]
    since: NotRequired[Any]
    by: NotRequired["PauseAuthor"]
    wake: NotRequired[List["SleepWake"]]


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
    state: str  # live | unknown | held (a helper a sleeping episode chose, paused until its Resume)
    code: str
    text: str
    count: NotRequired[int]
    ids: NotRequired[List[str]]


class WorkTurn(TypedDict):
    """What waits for the owner: an unanswered question or an action only the owner takes."""

    kind: str
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


class WorkMembers(TypedDict):
    ids: List[str]
    states: NotRequired[Dict[str, Dict[str, Any]]]


class WorkFocus(TypedDict):
    text: str
    at: Any


class Work(TypedDict):
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


class RoomStatus(TypedDict):
    """One room's status from its works: the most active work speaks; ``amber`` while something waits for the owner."""

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
    turn: int
    confirmed: bool


class WorkStateFields(TypedDict):
    """The work projection's ``/api/state`` fields, inherited by ``contracts.StateResponse``."""

    works: List[Work]
    rooms: Dict[str, RoomStatus]
    work_summary: WorkSummary
    works_complete: bool
    works_gaps: List[Dict[str, Any]]
