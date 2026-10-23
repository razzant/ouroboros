"""One work, one status: the pure classification behind ``/api/state`` ``works``.

A work is one request — the owner's or Ouroboros's own — together with every
Continue episode and every live member (helper tasks, delegated runs still working,
answer upkeep still in flight). The census (``gateway/state.py``) keeps its
frozen per-root rows; this module groups the same facts into works and gives
each work one state: an activity word from a closed set, typed reasons with one
English sentence each, its tails, whose turn it is, and what can release it.
Every surface and Ouroboros's own context read the result; none derives a word
of its own.

The word and its reason describe one subject. While the current episode lives,
the word is the episode's own state and its reason the episode's; what still runs
beside it (helpers, external runs, review work beside an episode that does not
execute, runs nobody can confirm) is a typed ``tails`` list, said as a second fact and never promoted into
the word: a paused episode whose review still finishes reads Paused with
"Review work finishing", not Working. ``live`` says whether anything actually
executes, which the room's mark shows. Only an ended episode, or a Continue held
for its earlier attempt, takes its word from what still runs.

The vocabulary splits by what releases a state. ``paused`` stays until someone
presses Resume, and its reason says who or what stopped it: the author a Pause
declared (the owner's press, Ouroboros's own sentence), the app stopping, the
spending limit, a technical problem; a Pause that declared nobody names nobody.
``waiting`` continues when what it waits for happens: helpers of an earlier
attempt finishing, a model becoming available, the owner's sign-in or answer.
``turn`` names what waits for the owner; ``release`` says who can move it.

No I/O, no clocks, no thresholds: the server adapter (``work_projection``) and
the worker-side roster supply the observations; equal inputs give equal works.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

ACTIVITY_ORDER = ("working", "queued", "waiting", "paused", "unknown")
TRANSITIONS = ("none", "finalizing", "pausing")

from ouroboros.work_stop import STOP_TEXT, activity_stop_view

# One owner sentence per reason code (the TASK_CAUSE_PHRASES discipline): keyed on the code
# only, a code with no sentence stays raw, and the raw code stays typed beside it.
REASON_TEXT = {
    **STOP_TEXT,
    "model_handoff": "a model request has no recorded result",
    "earlier_attempt_finishing": ("Continue starts when the helper of the earlier attempt finishes",
                                  "Continue starts when {count} helpers of the earlier attempt finish"),
    "earlier_runs_finishing": ("Continue starts when the external run of the earlier attempt finishes",
                               "Continue starts when {count} external runs of the earlier attempt finish"),
    "helpers_running": ("1 helper still working", "{count} helpers still working"),
    "delegated_running": ("1 external run still working", "{count} external runs still working"),
    "owner_answer": "Waiting for your answer",
    "model_access": "The model needs you: sign in or restore access",
    "model_wait": "Waiting for the model to become available",
    "project_admission": "Waiting for Project verification",
    "queued": "Waiting for a free worker",
    "review_work_finishing": "Review work finishing",
    "external_runs_unconfirmed": ("1 external run may still be running", "{count} external runs may still be running"),
}


def reason(code: str, *, count: Optional[int] = None, ids: Iterable[str] = (), since: Any = None,
           raw: str = "") -> Dict[str, Any]:
    """One typed reason with its English sentence; an unknown code keeps its raw text."""
    phrase = REASON_TEXT.get(code)
    if isinstance(phrase, tuple):
        text = phrase[0] if count == 1 else phrase[1].format(count=count)
    else:
        text = phrase if phrase is not None else code
    if code == "held" and raw:
        text = f"Held: {raw}"
    row: Dict[str, Any] = {"code": code, "text": text}
    if count is not None:
        row["count"] = count
    ids = [str(item) for item in ids if item]
    if ids:
        row["ids"] = ids
    if since:
        row["since"] = since
    return row


def _owner_wait(row: Mapping[str, Any]) -> str:
    """``owner_wait`` while a question positively waits for the owner, ``unknown`` on a gap, else ``''``.

    The census rule the cards apply (task_phase_chip.activityWaitPhase), on the server."""
    question = row.get("owner_wait") if isinstance(row.get("owner_wait"), dict) else row.get("required_question")
    question = question if isinstance(question, dict) else None
    if question:
        state = question.get("quiz_state") or question.get("state")
        if (question.get("owner_wait_state") == "resumed" or question.get("wait_ended_at")
                or state in {"answered", "expired_terminal", "superseded", "closed"}):
            return ""
        if question.get("owner_wait_state") == "waiting" or question.get("wait_for_answer") is True \
                or question.get("state") == "waiting":
            return "owner_wait"
        return "unknown"
    return "unknown" if row.get("required_question_unavailable") else ""


def _current_waits(row: Mapping[str, Any]) -> List[tuple]:
    attempt = int(row.get("task_attempt") or 1)
    waits = row.get("model_waits") if isinstance(row.get("model_waits"), dict) else {}
    return [(key, wait) for key, wait in waits.items()
            if isinstance(wait, dict) and wait.get("state") == "waiting" and wait.get("task_attempt") == attempt]


def _call(row: Mapping[str, Any]) -> str:
    """The exact-call word a working member is in (the cards' modelActivityPresentation)."""
    phase = row.get("phase")
    if phase == "preparing":
        return "preparing"
    activity = row.get("model_activity") if isinstance(row.get("model_activity"), dict) else {}
    operation = activity.get("phase")
    if operation in {"dispatching", "awaiting_response"}:
        return "sending"
    if operation == "preparing":
        return "preparing"
    if phase == "thinking":
        return "thinking"
    return "working"


def classify_member(row: Mapping[str, Any], raw: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Classify one census row (``decision_owners`` shape) with the raw facts the census drops.

    ``raw`` carries the queue row's hold and pause records, the declared author of the tree's Pause
    (``pause_by``, ``control_actor.pause_author``), the members that Pause already parked and the task's
    own identity facts (see ``work_projection``), ``running`` when the queue runs it, its current worker ``owner_wait``,
    and ``sent_review`` while a
    review it sent from a warm park still runs. Returns the member's
    own state only; ``live`` says whether the member itself executes now: under a settling Pause only a running
    member the Pause has not parked does, never a queued one or one that already saved its place."""
    raw = raw or {}
    task_id = str(row.get("activity_id") or "")
    phase = str(row.get("phase") or "unknown")
    state: Dict[str, Any] = {"task_id": task_id, "activity": "unknown", "transition": "none", "call": "",
                             "reasons": [], "turn": [], "release": "self", "confirmed": True, "live": False}
    if phase == "unknown":
        state["confirmed"] = False
        return state
    if phase in {"budget_paused", "budget_pausing"}:
        view = activity_stop_view(dict(row), dict(raw))
        state["stop"], state["refusal"] = view["stop"], view["refusal"]
        state["reasons"].append(view["reason"])
        state["resumable"] = view["resumable"]
        state["activity"] = "waiting" if view["waiting"] else "paused"
        state["release"] = ("owner" if view["release"] == "owner" else "mind" if view["resumable"]
                            else "self" if view["waiting"] else "none")
        if phase == "budget_pausing" and not view["waiting"] and raw.get("running", row.get("live")) \
                and task_id not in set(raw.get("parked") or ()):
            state.update(activity="working", transition="pausing", live=True)
        state["reviews_finishing"] = bool(row.get("finishing_reviews") or raw.get("sent_review"))
        state["model_handoffs"] = list(raw.get("model_handoffs") or ())
        if _owner_actions({**row, "stop": view["stop"], "refusal": view["refusal"]})[1]:
            state["turn"].append({"kind": "action", "task_id": task_id, "code": view["reason"]["code"]})
        if raw.get("execution_transport") is not None:  # a stopped call has ended; it never resumes
            state.update(release="none", turn=[], resumable=False)
        return state
    wait = _owner_wait(row)
    if wait == "owner_wait":
        question = row.get("owner_wait") if isinstance(row.get("owner_wait"), dict) else row.get("required_question")
        quiz = str((question or {}).get("quiz_id") or "")
        state.update(activity="waiting", release="owner")
        state["reasons"].append(reason("owner_answer"))
        state["turn"].append({"kind": "question", "task_id": task_id, **({"quiz_id": quiz} if quiz else {})})
        return state
    if wait == "unknown":
        state["confirmed"] = False
        return state
    # A warm sleeper keeps its worker and frozen census phase, but executes nothing.
    # A current Pause above wins; an ended or older attempt's wait proves no sleep.
    waiting = raw.get("owner_wait")
    if isinstance(waiting, dict) and waiting.get("state") == "waiting" and waiting.get("reason") == "sleep" \
            and waiting.get("task_attempt") == int(row.get("task_attempt") or 1) \
            and isinstance(waiting.get("sleep"), dict) and waiting["sleep"].get("mode") == "warm":
        state["activity"] = "waiting"
        state["reasons"].append(reason("sleeping"))
        return state
    waits = _current_waits(row)
    if waits:
        human = sorted(_owner_actions(row)[0])
        state["activity"] = "waiting"
        if human:
            state["release"] = "owner"
            state["reasons"].append(reason("model_access", ids=human))
            state["turn"].extend({"kind": "action", "task_id": task_id, "wait_id": key, "code": "model_access"}
                                 for key in human)
        else:
            state["reasons"].append(reason("model_wait", ids=[key for key, _ in waits]))
        if phase == "finalizing":
            state["transition"] = "finalizing"
        return state
    hold_fact = row.get("project_admission_hold") if isinstance(row.get("project_admission_hold"), dict) else {}
    if hold_fact:
        label = str(hold_fact.get("label") or "")
        state["activity"] = "waiting"
        entry = reason("project_admission")
        if label:
            entry["text"] = label
        state["reasons"].append(entry)
        return state
    if phase == "queued":
        state["activity"] = "queued"
        state["reasons"].append(reason("queued"))
        return state
    if phase == "finalizing":
        state.update(activity="working", transition="finalizing", live=True)
        return state
    if phase in {"working", "thinking", "preparing"}:
        state.update(activity="working", call=_call(row), live=True)
        return state
    state["confirmed"] = False
    return state


_RUNNING_PHASES = {"working", "thinking", "preparing", "finalizing"}


def controls_for(current: Optional[Mapping[str, Any]], current_row: Optional[Mapping[str, Any]], *,
                 earlier_live: Iterable[str] = (), live_roots: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """The actions the work's doors accept for its present state, each on its exact target.

    ``current``/``current_row`` are the live current episode's classified state and census row (None when the
    episode ended); ``earlier_live`` the earlier episode roots that still own live members; ``live_roots`` every
    episode root to stop, current first. The doors stay the authority: a refusal they still return is shown."""
    earlier = [str(task_id) for task_id in earlier_live if task_id]
    roots = [str(task_id) for task_id in live_roots if task_id]
    controls: List[Dict[str, Any]] = []
    if current is not None and current_row is not None:
        target = str(current.get("task_id") or "")
        phase = str(current_row.get("phase") or "")
        codes = {entry.get("code") for entry in current.get("reasons") or ()}
        # A live root, managed or a direct turn, and an answered root's upkeep: the hurry, stop and Pause doors
        # take them all; the Pause ends that upkeep at once. A warm sleep keeps both: either control wakes it.
        if ((current["activity"] == "working" and current.get("transition") in {"none", "finalizing"})
                or (current["activity"] == "waiting" and "sleeping" in codes)) and phase in _RUNNING_PHASES:
            controls += [{"action": "wrap_up", "target": target}, {"action": "hurry", "target": target}]
        # A Pause still settling keeps its Pause: the door coalesces a repeat, which retries an uncertain reply.
        if current.get("transition") in {"none", "finalizing", "pausing"} \
                and current["activity"] in {"working", "queued", "waiting"} \
                and "earlier_work_unconfirmed" not in codes and phase != "budget_paused":
            controls.append({"action": "pause", "target": target})
        if current["activity"] == "paused" and current.get("resumable"):
            controls.append({"action": "resume", "target": target})
    for task_id in [*([str(current.get("task_id"))] if current is not None else []), *roots]:
        if task_id and task_id not in {entry["target"] for entry in controls if entry["action"] == "stop"}:
            controls.append({"action": "stop", "target": task_id, "scope": "cascade"})
    for task_id in earlier:
        controls.append({"action": "stop_earlier", "target": task_id, "scope": "cascade"})
    return controls


def _owner_actions(row: Mapping[str, Any]) -> tuple:
    from ouroboros.gateway.main_decisions import owner_actions

    return owner_actions(dict(row))


def _rank(activity: str) -> int:
    return ACTIVITY_ORDER.index(activity) if activity in ACTIVITY_ORDER else len(ACTIVITY_ORDER)


def _outcome(row: Mapping[str, Any]) -> str:
    axes = row.get("outcome_axes") if isinstance(row.get("outcome_axes"), dict) else {}
    lifecycle = axes.get("lifecycle") if isinstance(axes.get("lifecycle"), dict) else {}
    return str(lifecycle.get("status") or row.get("status") or "")


def mark_of(activity: str, confirmed: bool, *, live: bool = False, unknown: bool = False) -> str:
    """The shape a work or a room shows on every surface (docs/DESIGN.md "Status marks"): what the work does, never
    whose move it is (that is ``turn``). ``live`` (something actually executes) moves the mark whatever the word; an
    execution nobody can confirm draws a ring while nothing else moves. Work that is observed running keeps moving
    even in an incomplete read."""
    if live:
        return "moving"
    if unknown:
        return "ring"
    if activity == "working":
        return "moving"
    if not confirmed:
        return "ring"
    return {"queued": "still", "waiting": "still", "paused": "pause", "idle": "none"}.get(activity, "ring")


# Tail kinds: what still runs beside (or after) the current episode, each said by one REASON_TEXT sentence.
RUN_KINDS = ("run", "review")
TAIL_STATES = ("live", "unknown")
# A run whose own task ended this way may outlive it unobserved: its liveness is unconfirmed, not running.
_FAILED_OWNER = frozenset({"failed", "cancelled"})


def run_tail(source: Any, owner: Optional[Mapping[str, Any]], *, stop_unknown: bool = False) -> tuple:
    """How one open delegated run is said beside its work, ``(kind, state)``: the one typing the server's work
    projection and Ouroboros's own roster share. A review-owned ``source`` is review work; a run nobody can confirm
    is one whose own task (``owner``, its stored result) ended failed or cancelled, or whose stop that task's live
    Pause could not confirm (``stop_unknown``, ``external_runs.EXTERNAL_STOP_UNKNOWN``); every other open run is
    live. Facts the readers already hold, never a new probe."""
    from ouroboros.delegate_registration_policy import review_owned_source

    if review_owned_source(source):
        return "review", "live"
    return "run", "unknown" if stop_unknown or _outcome(owner or {}) in _FAILED_OWNER else "live"


def _tail(kind: str, code: str, ids: List[str], *, state: str = "live", count: Optional[int] = None) -> Dict[str, Any]:
    entry = reason(code, count=len(ids) if count is None else count, ids=ids)
    entry.update(kind=kind, state=state)
    if code == "review_work_finishing":
        entry.pop("count", None)  # the fence names reviewed tasks, never a reviewer count
    return entry


def work_tails(helper_ids: List[str], runs: Iterable[Mapping[str, Any]], *, reviewing: bool = False,
               named: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """What still runs beside an episode, in the order and words every reader says it: working helpers, live
    external runs, review work (never beside an episode that executes), runs nobody can confirm. ``runs`` are
    typed by ``run_tail``; ``named`` holds the kinds the episode's own sentence already says (a Continue held for
    its earlier attempt)."""
    runs = [run for run in runs if run.get("kind") in RUN_KINDS and run.get("state") in TAIL_STATES]
    live = [str(run["run_id"]) for run in runs if run["kind"] == "run" and run["state"] == "live"]
    reviews = [str(run["run_id"]) for run in runs if run["kind"] == "review" and run["state"] == "live"]
    unknown = [str(run["run_id"]) for run in runs if run["state"] == "unknown"]
    tails: List[Dict[str, Any]] = []
    if helper_ids and "helper" not in named:
        tails.append(_tail("helper", "helpers_running", list(helper_ids)))
    if live and "run" not in named:
        tails.append(_tail("run", "delegated_running", live))
    if reviewing or reviews:
        tails.append(_tail("review", "review_work_finishing", reviews))
    if unknown:
        tails.append(_tail("run", "external_runs_unconfirmed", unknown, state="unknown"))
    return tails


def model_tails(blockers: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    attempts = sorted({str(b["attempt_id"]) for b in blockers if b.get("kind") == "model_handoff"
                       and not b.get("review_owned") and b.get("attempt_id")})
    return [_tail("model", "model_handoff", attempts, state="unknown")] if attempts else []


def roll_up_work(work_id: str, members: List[Dict[str, Any]], *, episode: Mapping[str, Any],
                 name: str = "", room: Any = None, focus: Any = None, runs: Iterable[Mapping[str, Any]] = (),
                 outcome: str = "", gaps: Iterable[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """One work from its classified members and its open delegated runs.

    ``episode`` is the current episode: ``{task_id, live, state?}`` — ``state`` is its own classified
    member when the episode root is live; an ended episode carries only its outcome. Helpers of any
    episode are the other members. ``runs`` are the work's unsettled delegated runs as
    ``{run_id, kind: run|review, state: live|unknown}`` (``work_projection`` types them)."""
    gaps = list(gaps)
    current = episode.get("state") if isinstance(episode.get("state"), dict) else None
    # Review work is a tail beside an episode that is not executing; a running episode's own review is its own work.
    executing = bool(current is not None and current.get("live") and current["transition"] != "pausing")
    runs = [run for run in runs if run.get("kind") in RUN_KINDS and run.get("state") in TAIL_STATES
            and not (executing and run["kind"] == "review")]
    helpers = [member for member in members if member is not current]
    working_helpers = [member for member in helpers if member["activity"] == "working" and member["live"]]
    helper_ids = [member["task_id"] for member in working_helpers]
    live_runs = [str(run["run_id"]) for run in runs if run["kind"] == "run" and run["state"] == "live"]
    review_runs = [str(run["run_id"]) for run in runs if run["kind"] == "review" and run["state"] == "live"]
    unknown_runs = [str(run["run_id"]) for run in runs if run["state"] == "unknown"]
    reviewing = bool(review_runs) or any(member.get("reviews_finishing") for member in members)
    turn = [item for member in members for item in member["turn"]]
    held = current is not None and any(entry.get("code") == "earlier_work_unconfirmed" for entry in current["reasons"])
    named = set()  # tails a held Continue already names in its own sentence
    if current is not None and held and (helper_ids or live_runs):
        # A Continue held for its earlier attempt says what it waits for: those helpers or runs (owner's 17.10 words).
        sentence = (reason("earlier_attempt_finishing", count=len(helper_ids), ids=helper_ids) if helper_ids
                    else reason("earlier_runs_finishing", count=len(live_runs)))
        reasons = [sentence if entry.get("code") == "earlier_work_unconfirmed" else entry for entry in current["reasons"]]
        activity, transition, call, release = "working", "none", "", "self"
        named = {"helper", "run"}
    elif current is not None:
        # The live episode speaks: its word, its reason; what runs beside it is a tail, never its word.
        reasons = list(current["reasons"])
        activity, transition, call, release = current["activity"], current["transition"], current["call"], current["release"]
    elif working_helpers or live_runs or reviewing:
        # The episode ended: what still runs is all there is.
        reasons, activity, release = [], "working", "self"
        transitions = {member["transition"] for member in working_helpers}
        transition = "finalizing" if transitions == {"finalizing"} else (
            "pausing" if "pausing" in transitions and "none" not in transitions and not live_runs else "none")
        calls = {member["call"] for member in working_helpers if member["call"]}
        call = calls.pop() if len(calls) == 1 and not live_runs else ""
    elif members:
        lead = min(members, key=lambda member: _rank(member["activity"]))
        reasons = list(lead["reasons"])
        activity, transition, call, release = lead["activity"], lead["transition"], lead["call"], lead["release"]
    else:
        # Only runs nobody can confirm remain: the work stays listed (its Stop reaches them), its word unconfirmed.
        reasons, activity, transition, call, release = [], "unknown", "none", "", "none"
    tails = work_tails(helper_ids, runs, reviewing=reviewing, named=named)
    tails += model_tails(b for member in members for b in member.get("model_handoffs") or ())
    episode_live = bool(current is not None and current.get("live"))
    live = episode_live or bool(working_helpers or live_runs or reviewing)
    if turn:
        release = "owner"
    work: Dict[str, Any] = {
        "work_id": work_id, "room": room, "name": name,
        "episode": {"task_id": str(episode.get("task_id") or work_id), "live": bool(current is not None)},
        "activity": activity, "transition": transition, "call": call,
        "because": reasons, "tails": tails, "live": live, "turn": turn, "release": release,
        **({"stop": current.get("stop", {}), "refusal": current.get("refusal", {})} if current else {}),
        "members": {"ids": [member["task_id"] for member in members], "states": {
            member["task_id"]: {**{key: member[key] for key in (
                "activity", "transition", "call", "live", "release", "confirmed", "turn")},
                "because": member["reasons"], "tails": model_tails(member.get("model_handoffs") or ())}
            for member in members}},
        "confirmed": all(member["confirmed"] for member in members) and not gaps,
    }
    # The work's own mark follows its episode (the Paused mark lights when the tasks are parked; what they let
    # finish, or a run nobody can confirm, is the line beside it); an ended episode's mark follows what still runs.
    # The room's mark moves while anything in it runs (roll_up_room).
    work["mark"] = mark_of(activity, work["confirmed"], live=episode_live if current is not None else live,
                           unknown=current is None and bool(unknown_runs) and not live)
    if outcome:
        work["outcome"] = outcome
    if isinstance(focus, dict) and str(focus.get("text") or "").strip():
        work["focus"] = {"text": str(focus["text"]), "at": focus.get("at") or focus.get("authored_at")}
    if gaps:
        work["gaps"] = gaps
    return work


def roll_up_room(works: List[Mapping[str, Any]], *, complete: bool = True) -> Dict[str, Any]:
    """One room from its works: the most active work speaks; whose turn it is never replaces it. The mark moves
    while anything in the room actually runs, a paused work's finishing review included: in a room, motion wins and
    each work's row says the rest."""
    if not works:
        activity = "idle" if complete else "unknown"
        return {"activity": activity, "transition": "none", "call": "", "turn": [], "amber": False,
                "mark": "none" if complete else "ring", "confirmed": complete, "works": []}
    known = [work["activity"] for work in works if work["activity"] != "unknown"]
    activity = min(known, key=_rank) if known else "unknown"
    working = [work for work in works if work["activity"] == "working"]
    transition = "none"
    if working and all(work["transition"] == "finalizing" for work in working):
        transition = "finalizing"
    elif working and all(work["transition"] in {"pausing", "finalizing"} for work in working) \
            and any(work["transition"] == "pausing" for work in working):
        transition = "pausing"
    calls = {work["call"] for work in working if work.get("call")}
    turn = [{"work_id": work["work_id"], **item} for work in works for item in work["turn"]]
    confirmed = complete and all(work["confirmed"] for work in works)
    live = any(work.get("live") for work in works)
    unknown = any(tail.get("state") == "unknown" for work in works for tail in work.get("tails") or ())
    return {"activity": activity, "transition": transition, "call": calls.pop() if len(calls) == 1 else "",
            "turn": turn, "amber": bool(turn), "mark": mark_of(activity, confirmed, live=live, unknown=unknown),
            "confirmed": confirmed, "works": [work["work_id"] for work in works]}


def roll_up_summary(rooms: Mapping[Any, Mapping[str, Any]], works: List[Mapping[str, Any]],
                    *, complete: bool = True) -> Dict[str, Any]:
    """The tray's one line: counts by activity, how many wait for the owner, and confirmation."""
    counts = {activity: 0 for activity in ACTIVITY_ORDER}
    for work in works:
        counts[work["activity"] if work["activity"] in counts else "unknown"] += 1
    return {"counts": counts, "turn": sum(bool(work["turn"]) for work in works),
            "confirmed": complete and all(room.get("confirmed", True) for room in rooms.values())}
