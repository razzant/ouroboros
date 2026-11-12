"""One work, one status: the pure classification behind /api/state ``works`` (gateway/work_status.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ouroboros.control_actor import pause_author
from ouroboros.gateway import work_status as ws
from ouroboros.gateway import work_projection as wp


def row(task_id, phase="working", **extra):
    return {"activity_id": task_id, "phase": phase, "kind": extra.pop("kind", "managed_task"),
            "chat_id": extra.pop("chat_id", 42), "task_attempt": 1, "started_at": extra.pop("started_at", 100.0), **extra}


# ---- one member ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("is_call", [False, True])
def test_stopped_call_has_no_resume_release_but_text_still_does(is_call, tmp_path):
    parked = row("call", "budget_paused", pause_cause="restart", can_resume=True)
    metadata = {"execution_transport": {"type": "live_conversation"}} if is_call else {}
    raw = {"metadata": metadata, "_budget_pause_hold": {"reason": "owner_restart_hold"}}
    facts = wp._member_raw(parked, raw, wp._Observations(tmp_path), "call")
    state = ws.classify_member(parked, facts)
    # A stopped call has ended: nobody resumes it. A text task keeps the release the stop view gives it.
    assert (state["release"] == "none") is is_call
    if is_call:
        assert state["turn"] == [] and state.get("resumable") is False
    actions = {control["action"] for control in ws.controls_for(state, parked)}
    assert ("resume" in actions) is (not is_call)
    assert "stop" in actions


def test_working_member_names_its_exact_call():
    state = ws.classify_member(row("t", model_activity={"phase": "awaiting_response"}))
    assert (state["activity"], state["call"], state["release"]) == ("working", "sending", "self")
    assert ws.classify_member(row("t", "thinking"))["call"] == "thinking"
    assert ws.classify_member(row("t", "preparing", kind="direct_chat"))["call"] == "preparing"


@pytest.mark.parametrize("turn_phase, text", [
    ("admitted", "Setting up the turn"), ("started", "Starting the turn"),
    ("context_building", "Building context"),
])
def test_a_preparing_direct_turn_says_the_start_step_it_stands_at(turn_phase, text):
    state = ws.classify_member(row("d", "preparing", kind="direct_chat", turn_phase=turn_phase))
    assert (state["activity"], state["call"], state["live"]) == ("working", "preparing", True)
    assert state["reasons"] == [{"code": f"turn_{turn_phase}", "text": text}]


@pytest.mark.parametrize("kind", ["direct_chat", "presence", "managed_task"])
def test_a_carried_turn_says_its_start_step_until_its_own_call(kind):
    # A conversation-lane row reads `working` once a carrier holds it (conversation_lane.lane_turn_rows); its
    # frame-fed start step is still a start step (lost on that path with #2654, 02.11). A pooled root a worker runs
    # says the same (`carrier` from gateway.state, #2862); one no worker is said to hold stays Working below.
    def carried(**extra):
        return row("d", "working", kind=kind, carrier="worker", **extra)

    for step, text in (("started", "Starting the turn"), ("context_building", "Building context"),
                       ("context_built", "Context ready")):
        state = ws.classify_member(carried(turn_phase=step))
        assert (state["activity"], state["call"], state["live"]) == ("working", "preparing", True), step
        assert state["reasons"] == [{"code": f"turn_{step}", "text": text}], step
    # Its own call started or observed, no start fact yet, or a carrier still awaited: no start step from this rule.
    for done in (carried(turn_phase="requested"), carried(turn_phase="dispatched"),
                 carried(turn_phase="context_built", model_activity={"phase": "awaiting_response"}), carried()):
        state = ws.classify_member(done)
        assert state["reasons"] == [] and state["call"] in {"working", "sending"}, done
    awaiting = ws.classify_member(row("d", "preparing", kind=kind, carrier="awaiting"))
    assert (awaiting["call"], awaiting["reasons"]) == ("preparing", [])
    unheld = ws.classify_member(row("d", "working", kind=kind, turn_phase="context_building"))
    assert (unheld["call"], unheld["reasons"]) == ("working", [])


def test_a_start_step_speaks_only_while_preparing_and_a_started_call_reads_the_census_move():
    # Its own Main call started before its census phase moved: the word that move gives, never Sending ahead of it.
    for step in ("requested", "dispatched"):
        lagging = ws.classify_member(row("d", "preparing", kind="direct_chat", turn_phase=step))
        assert (lagging["call"], lagging["reasons"]) == ("working", []), step
    # Without a start fact, or once its census phase moved on, a turn's word and reasons stay what they were.
    assert ws.classify_member(row("d", "preparing", kind="direct_chat"))["reasons"] == []
    moved = ws.classify_member(row("d", "working", kind="direct_chat", turn_phase="context_built"))
    assert (moved["call"], moved["reasons"]) == ("working", [])
    assert ws.classify_member(row("d", "working", kind="direct_chat", turn_phase="dispatched"))["call"] == "working"
    # Sending is the census's own observation of the request going out.
    sending = row("d", "working", kind="direct_chat", turn_phase="requested", model_activity={"phase": "dispatching"})
    assert ws.classify_member(sending)["call"] == "sending"


def test_writer_held_continue_waits_and_is_never_the_owners_turn():
    held = row("c", "budget_paused", pause_cause="unknown", can_resume=False)
    state = ws.classify_member(held, {"hold": {"reason": "continuation_writer_unsettled"}})
    assert state["activity"] == "waiting"
    assert [entry["code"] for entry in state["reasons"]] == ["earlier_work_unconfirmed"]
    assert state["turn"] == [] and state["release"] == "self"
    # The same parked row without the writer hold is a pause, not a wait.
    assert ws.classify_member(held, {"hold": {}})["activity"] == "paused"
    # A hold the owner's selection already released explains nothing.
    assert ws.classify_member(held, {"hold": {}, "_resume_selection": {"selected_by": "owner"}})[
        "activity"] == "paused"


@pytest.mark.parametrize("author, code, release, text", [
    ({"actor": "owner", "text": "", "at": "T0"}, "paused_by_owner", "mind", "You paused it"),
    ({"actor": "ouroboros", "text": "pause before the update you asked for", "at": "T0"}, "paused_by_mind", "mind",
     "Ouroboros paused it"),
    ({"actor": "", "text": "", "at": "T0"}, "paused_unattributed", "mind", ""),
])
def test_a_decided_pause_says_its_declared_author_never_the_door(author, code, release, text):
    state = ws.classify_member(row("t", "budget_paused", pause_cause="owner"), {"pause_by": author, "paused_at": "T9"})
    assert (state["activity"], state["release"], state["turn"]) == ("paused", release, [])
    entry = state["reasons"][0]
    assert (entry["code"], entry["text"], entry["since"]) == (code, text, "T0")  # dated from the press
    assert ("by" in entry) is bool(author["actor"])
    assert "paused_by" not in state


def test_pause_author_reads_the_declared_record_and_never_defaults_to_the_owner():
    assert pause_author({"by": {"actor": "owner", "text": "", "at": "A"}, "requested_at": "R"}) == {
        "actor": "owner", "text": "", "at": "A"}
    # Legacy fences: the mind's own record, at the top or as its first press on the same fence.
    legacy = {"fence_id": "f1", "actor": "mind", "reason": " for the update ", "requested_at": "R", "paused_at": "P"}
    assert pause_author(legacy) == {"actor": "ouroboros", "text": "for the update", "at": "R"}
    assert pause_author({**legacy, "text": "new wording"}) == {
        "actor": "ouroboros", "text": "new wording", "at": "R"}
    pressed_later = {"fence_id": "f1", "requested_by": "owner", "requested_at": "R",
                     "requests": {"a": {"fence_id": "f1"}, "b": {"fence_id": "f1", "actor": "mind", "reason": "x"},
                                  "c": {"fence_id": "f0", "actor": "mind", "reason": "older fence"}}}
    assert pause_author(pressed_later) == {"actor": "ouroboros", "text": "x", "at": "R"}
    # A press that declared nobody, a missing or an unreadable fence name nobody: never "owner" by default.
    for fence in ({"requested_by": "owner", "requested_at": "R"}, {}, None, {"by": {"actor": "admin"}}):
        assert pause_author(fence)["actor"] == ""


@pytest.mark.parametrize("cause, hold, code", [
    ("restart", {"reason": "saved_work_hold"}, "saved_on_stop"),
    ("restart", {"reason": "owner_restart_hold"}, "after_restart"),
    ("budget", {}, "spending_limit"),
])
def test_owner_required_resume_is_the_owners_turn_only_when_the_census_says_so(cause, hold, code):
    parked = row("t", "budget_paused", pause_cause=cause, can_resume=True)
    state = ws.classify_member(parked, {"hold": hold})
    assert state["activity"] == "paused" and state["reasons"][0]["code"] == code
    assert state["turn"] == [{"kind": "action", "task_id": "t", "code": code}] and state["release"] == ("owner" if code == "after_panic" else "mind")
    quiet = ws.classify_member(dict(parked, can_resume=False), {"hold": hold})
    assert quiet["turn"] == []


def test_cold_sleep_waits_and_pausing_is_still_work():
    sleeping = ws.classify_member(row("t", "budget_paused", pause_cause="sleep"))
    assert (sleeping["activity"], sleeping["release"]) == ("waiting", "self")  # it wakes by itself
    owner = {"pause_by": {"actor": "owner"}}
    pausing = ws.classify_member(row("t", "budget_pausing", pause_cause="owner"), {**owner, "running": True})
    assert (pausing["activity"], pausing["transition"], pausing["live"]) == ("working", "pausing", True)
    # Under a Pause still settling, a member the queue holds (never started, or already saved and parked) is
    # Paused with Resume because it executes nothing; nor does a running one the settled census already parked.
    queued = ws.classify_member(row("t", "budget_pausing", pause_cause="owner"), owner)
    assert (queued["transition"], queued["live"]) == ("none", False)
    parked = ws.classify_member(row("t", "budget_pausing", pause_cause="owner"), {"parked": ["t"], "running": True})
    assert (parked["transition"], parked["live"]) == ("none", False)


def test_a_warm_sleep_names_only_the_helpers_it_chose_that_a_stop_holds_and_who_resumes_them():
    def sleeper(*chosen):
        return ws.classify_member(row("root"), {"running": True, "owner_wait": {
            "state": "waiting", "reason": "sleep", "task_attempt": 1,
            "sleep": {"mode": "warm", "tasks": list(chosen), "senders": list(chosen)}}})

    def helper(task_id, raw):
        return ws.classify_member(row(task_id, "budget_paused", pause_cause="unknown", can_resume=False,
                                      parent_task_id="root", root_task_id="root"), raw)

    selected = helper("chosen", {"_budget_pause": {"reason": "owner", "exact_continuation": True}})
    other = helper("other", {"_budget_pause": {"reason": "owner", "exact_continuation": True}})
    panic = helper("panic", {"hold": {"reason": "panic_hold", "release": "owner"}})
    root = sleeper("chosen", "panic")
    work = ws.roll_up_work("root", [root, selected, other, panic], episode={"task_id": "root", "state": root})
    # The sleep keeps its word, reason, controls and turn; its line names what its saved sleep chose, by release.
    assert (work["activity"], work["because"][0]["code"], work["turn"], work["mark"]) == (
        "waiting", "sleeping", [], "still")
    assert [(tail["code"], tail["ids"], tail["state"], tail["text"]) for tail in work["tails"]] == [
        ("awaited_helpers_mind", ["chosen"], "held", "1 helper it waits for is paused until Ouroboros resumes it"),
        ("awaited_helpers_owner", ["panic"], "held", "1 helper it waits for is paused until you resume it")]
    assert "other" not in str(work["tails"]), "a paused helper the sleep did not choose is not why it waits"
    controls = ws.controls_for(root, row("root"))
    assert "resume" not in [entry["action"] for entry in controls], "no Resume door is invented for a helper"
    # A Continue episode is neither the helper's root nor its parent: it holds no request door, so the line
    # names no releaser for a helper only a mind could ask to resume (the owner's own release keeps its words).
    later = ws.classify_member(row("later"), {"running": True, "owner_wait": {
        "state": "waiting", "reason": "sleep", "task_attempt": 1,
        "sleep": {"mode": "warm", "tasks": ["chosen", "panic"]}}})
    work = ws.roll_up_work("root", [later, selected, panic], episode={"task_id": "later", "state": later})
    assert [(tail["code"], tail["ids"], tail["text"]) for tail in work["tails"]] == [
        ("awaited_helpers_owner", ["panic"], "1 helper it waits for is paused until you resume it"),
        ("awaited_helpers_paused", ["chosen"], "1 helper it waits for is paused")]
    # Without the saved selection (or once the episode ended) the line says no such thing.
    awake = ws.roll_up_work("root", [sleeper(), selected], episode={"task_id": "root", "state": sleeper()})
    assert awake["tails"] == []
    assert ws.roll_up_work("root", [selected], episode={"task_id": "root"})["tails"] == []


_AT = "2026-11-03T18:05:00+00:00"
_NAMED, _TIME = {"kind": "task", "name": "Draft the notes"}, {"kind": "time", "at": _AT}
@pytest.mark.parametrize(("sleep", "text", "wake"), [
    ({"tasks": ["kid"]}, "Sleeping until Draft the notes finishes", [_NAMED]),
    ({"tasks": ["gone"]}, "Sleeping until the task it chose finishes", [{"kind": "tasks", "count": 1}]),
    # Every source it chose is said, joined by "or", its time last: a time alone also wakes on any message.
    ({"wake_at": _AT, "any_mail": True}, f"Sleeping until a message arrives or {_AT}", [{"kind": "mail"}, _TIME]),
    ({"mode": "warm"}, "Sleeping until it is woken", None),
])
def test_a_sleep_says_what_it_chose_to_wait_for(sleep, text, wake):
    names = {"kid": "Draft the notes", "x": ""}
    warm = ws.classify_member(row("root"), {"running": True, "owner_wait": {
        "state": "waiting", "reason": "sleep", "task_attempt": 1, "sleep": {"mode": "warm", **sleep}}}, names)
    cold = ws.classify_member(row("root", "budget_paused", pause_cause="sleep"),
                              {"_budget_pause": {"reason": "sleep", "sleep": {"mode": "cold", **sleep}}}, names)
    for state in (warm, cold):
        (said,) = state["reasons"]
        assert (said["code"], said["text"], said.get("wake")) == ("sleeping", text, wake)
        assert "gone" not in said["text"] and "kid" not in said["text"], "a task is never named by its id"
    assert warm["activity"] == cold["activity"] == "waiting"


def test_the_projection_names_a_sleeps_helper_from_its_own_census_row():
    """No stored result is read here (``work_fields`` answers none): the name is the census row's own."""
    from tests._work_projection_shared import work_fields

    sleep = {"mode": "warm", "tasks": ["kid"], "senders": ["kid"]}
    owners = [row("root"), row("kid", parent_task_id="root", root_task_id="root")]
    raw = {"root": {"running": True, "owner_wait": {"state": "waiting", "reason": "sleep", "task_attempt": 1,
                                                    "sleep": sleep}},
           "kid": {"running": True, "suggested_name": "Continue: Draft the notes"}}
    (work,) = work_fields(owners, raw)["works"]
    assert work["because"] == [{"code": "sleeping", "text": "Sleeping until Draft the notes finishes or writes",
                                "wake": [{"kind": "task_or_writes", "name": "Draft the notes"}]}]
    # A helper the census does not hold is counted, never named by its id.
    (work,) = work_fields(owners[:1], {"root": raw["root"]})["works"]
    assert work["because"][0]["text"] == "Sleeping until the task it chose finishes or a message arrives"


def test_under_a_settling_pause_only_running_helpers_are_said_still_working():
    owner = {"pause_by": {"actor": "owner"}}
    member = lambda task_id, **raw: ws.classify_member(  # noqa: E731
        row(task_id, "budget_pausing", pause_cause="owner", **({"parent_task_id": "r"} if task_id != "r" else {})),
        {**owner, **raw})
    root = member("r", running=True)
    work = ws.roll_up_work("r", [root, member("h1", running=True), member("h2")], episode={"task_id": "r", "state": root})
    assert (work["activity"], work["transition"]) == ("working", "pausing")
    assert [(tail["code"], tail["ids"]) for tail in work["tails"]] == [("helpers_running", ["h1"])]
    # The running helper parks too: no helper is said working, and the work still reads Pausing….
    work = ws.roll_up_work("r", [root, member("h1", running=True, parked=["h1"]), member("h2")],
                           episode={"task_id": "r", "state": root})
    assert (work["activity"], work["transition"], work["tails"]) == ("working", "pausing", [])


def test_refused_source_keeps_resume_for_rechecking_and_unknown_without_a_stop_stays_held():
    state = ws.classify_member(row("t", "budget_paused", pause_cause="unknown"), {"hold": {"reason": "restore_refused:x"}})
    assert state["reasons"][0]["code"] == "restore_failed" and state["release"] == "mind"
    raw = ws.classify_member(row("t", "budget_paused", pause_cause="mystery"), {"hold": {}})
    assert raw["reasons"][0]["code"] == "held" and raw["reasons"][0]["text"] == "Held"


def test_owner_question_is_a_wait_on_the_owner_and_a_gap_is_unconfirmed():
    asked = row("t", owner_wait={"quiz_id": "q1", "owner_wait_state": "waiting", "quiz_state": "open"})
    state = ws.classify_member(asked)
    assert state["activity"] == "waiting" and state["turn"] == [{"kind": "question", "task_id": "t", "quiz_id": "q1"}]
    answered = row("t", owner_wait={"quiz_id": "q1", "owner_wait_state": "resumed", "quiz_state": "answered"})
    assert ws.classify_member(answered)["activity"] == "working"
    gap = ws.classify_member(row("t", required_question_unavailable=True))
    assert gap["confirmed"] is False and gap["activity"] == "unknown"


def test_model_wait_is_the_owners_turn_only_when_it_requires_him():
    waits = {"w1": {"state": "waiting", "task_attempt": 1, "requires_owner_action": True, "revision": 2}}
    state = ws.classify_member(row("t", model_waits=waits))
    assert state["activity"] == "waiting" and state["turn"][0]["wait_id"] == "w1"
    calm = ws.classify_member(row("t", model_waits={"w1": {**waits["w1"], "requires_owner_action": False}}))
    assert calm["activity"] == "waiting" and calm["turn"] == [] and calm["release"] == "self"
    other_attempt = ws.classify_member(row("t", model_waits={"w1": {**waits["w1"], "task_attempt": 2}}))
    assert other_attempt["activity"] == "working"


def test_project_hold_queue_finalizing_and_unknown():
    held = ws.classify_member(row("t", "queued", project_admission_hold={"label": "Waiting for Project verification"}))
    assert held["activity"] == "waiting" and held["reasons"][0]["text"] == "Waiting for Project verification"
    queued = ws.classify_member(row("t", "queued"))
    # A queued row records no cause and therefore has no secondary reason line.
    assert queued["activity"] == "queued" and queued["reasons"] == [
        {"code": "queued", "text": ""}]
    final = ws.classify_member(row("t", "finalizing"))
    assert (final["activity"], final["transition"]) == ("working", "finalizing")
    unknown = ws.classify_member(row("t", "unknown"))
    assert unknown["activity"] == "unknown" and unknown["confirmed"] is False


# ---- one work -----------------------------------------------------------------------------------------

def _helpers(count, root="p"):
    return [ws.classify_member(row(f"h{index}", parent_task_id=root, root_task_id=root)) for index in range(count)]


def test_held_continue_with_running_helpers_is_working_and_says_why():
    current = ws.classify_member(row("c", "budget_paused", pause_cause="unknown"),
                                 {"hold": {"reason": "continuation_writer_unsettled"}})
    members = [current, *_helpers(3)]
    work = ws.roll_up_work("p", members, episode={"task_id": "c", "state": current}, name="Release 7.7", room=42)
    assert work["activity"] == "working" and work["release"] == "self" and work["turn"] == []
    assert work["because"][0]["code"] == "earlier_attempt_finishing" and work["because"][0]["count"] == 3
    assert work["because"][0]["text"] == "Continue starts when 3 helpers of the earlier attempt finish"
    assert work["episode"] == {"task_id": "c", "live": True} and work["live"] is True
    assert work["tails"] == []  # the sentence already names the helpers it waits for
    assert sorted(work["members"]["ids"]) == ["c", "h0", "h1", "h2"]
    one = ws.roll_up_work("p", [current, *_helpers(1)], episode={"task_id": "c", "state": current})
    assert one["because"][0]["text"] == "Continue starts when the helper of the earlier attempt finishes"


def test_held_continue_waiting_on_external_runs_or_on_nothing_visible():
    current = ws.classify_member(row("c", "budget_paused", pause_cause="unknown"),
                                 {"hold": {"reason": "continuation_writer_unsettled"}})
    live = [{"run_id": "r1", "kind": "run", "state": "live"}, {"run_id": "r2", "kind": "run", "state": "live"}]
    runs = ws.roll_up_work("p", [current], episode={"task_id": "c", "state": current}, runs=live)
    assert runs["activity"] == "working" and runs["because"][0]["code"] == "earlier_runs_finishing"
    assert runs["because"][0]["count"] == 2 and runs["tails"] == []
    alone = ws.roll_up_work("p", [current], episode={"task_id": "c", "state": current})
    assert alone["activity"] == "waiting" and alone["because"][0]["code"] == "earlier_work_unconfirmed"
    assert alone["release"] == "self"


def test_helpers_of_an_ended_attempt_keep_the_work_working_with_its_outcome():
    work = ws.roll_up_work("p", _helpers(2), episode={"task_id": "p"}, outcome="failed")
    assert work["activity"] == "working" and work["outcome"] == "failed" and work["live"] is True
    assert work["because"] == []
    assert work["tails"] == [{**ws.reason("helpers_running", count=2, ids=["h0", "h1"]), "kind": "helper",
                              "state": "live"}]
    assert work["tails"][0]["text"] == "2 helpers still working" and work["mark"] == "moving"
    assert work["episode"]["live"] is False


def test_finalizing_answer_and_paused_work():
    final = ws.classify_member(row("t", "finalizing", status="done"))
    work = ws.roll_up_work("t", [final], episode={"task_id": "t", "state": final}, outcome="done")
    assert (work["activity"], work["transition"], work["outcome"]) == ("working", "finalizing", "done")
    paused = ws.classify_member(row("t", "budget_paused", pause_cause="owner"), {"pause_by": {"actor": "owner"}})
    work = ws.roll_up_work("t", [paused], episode={"task_id": "t", "state": paused})
    assert (work["activity"], work["release"], work["because"][0]["code"]) == ("paused", "mind", "paused_by_owner")
    assert (work["tails"], work["live"], work["mark"]) == ([], False, "pause") and "paused_by" not in work


def test_any_members_turn_is_the_works_turn_and_unknown_never_erases_known_work():
    asked = ws.classify_member(row("h", owner_wait={"quiz_id": "q", "owner_wait_state": "waiting"},
                                   parent_task_id="t", root_task_id="t"))
    lead = ws.classify_member(row("t"))
    work = ws.roll_up_work("t", [lead, asked], episode={"task_id": "t", "state": lead})
    assert work["activity"] == "working" and work["turn"][0]["kind"] == "question" and work["release"] == "owner"
    unknown = ws.classify_member(row("u", "unknown", parent_task_id="t", root_task_id="t"))
    work = ws.roll_up_work("t", [lead, unknown], episode={"task_id": "t", "state": lead})
    assert work["activity"] == "working" and work["confirmed"] is False


def test_structure_is_the_same_for_one_and_eighteen_helpers():
    current = ws.classify_member(row("c", "budget_paused", pause_cause="unknown"),
                                 {"hold": {"reason": "continuation_writer_unsettled"}})
    small = ws.roll_up_work("p", [current, *_helpers(1)], episode={"task_id": "c", "state": current})
    large = ws.roll_up_work("p", [current, *_helpers(18)], episode={"task_id": "c", "state": current})
    assert set(small) == set(large) and small["activity"] == large["activity"] == "working"
    assert [entry["code"] for entry in small["because"]] == [entry["code"] for entry in large["because"]]


# ---- one room -----------------------------------------------------------------------------------------

def _work(activity, *, transition="none", turn=(), confirmed=True, work_id="w"):
    return {"work_id": work_id, "activity": activity, "transition": transition, "call": "", "turn": list(turn),
            "confirmed": confirmed}


def test_room_takes_the_most_active_work_and_amber_follows_the_turn():
    room = ws.roll_up_room([_work("waiting", turn=[{"kind": "question", "task_id": "a"}], work_id="a"),
                            _work("working", work_id="b")])
    assert (room["activity"], room["mark"], room["amber"]) == ("working", "moving", True)
    assert ws.roll_up_room([_work("queued")])["mark"] == "still"
    # A wait that ends by itself reads like the queue; a pause stands until Resume (docs/DESIGN.md "Status marks").
    assert ws.roll_up_room([_work("waiting")])["mark"] == "still"
    assert ws.roll_up_room([_work("paused")])["mark"] == "pause" and not ws.roll_up_room([_work("paused")])["amber"]


def test_mark_of_names_what_the_work_does_and_never_whose_move_it_is():
    assert [ws.mark_of(activity, True) for activity in ("working", "queued", "waiting", "paused", "idle", "unknown")] \
        == ["moving", "still", "still", "pause", "none", "ring"]
    # Observed work keeps moving in an incomplete read; anything else unconfirmed is the ring.
    assert ws.mark_of("working", False) == "moving"
    assert {ws.mark_of(activity, False) for activity in ("queued", "waiting", "paused", "unknown")} == {"ring"}
    # Something that actually runs moves any shape; a run nobody can confirm is a ring while nothing else runs.
    assert ws.mark_of("paused", True, live=True) == "moving" and ws.mark_of("paused", True, unknown=True) == "ring"
    assert ws.mark_of("paused", True, live=True, unknown=True) == "moving"


def test_room_finalizing_only_says_finalizing_and_empty_rooms_are_idle_or_unknown():
    assert ws.roll_up_room([_work("working", transition="finalizing")])["transition"] == "finalizing"
    mixed = ws.roll_up_room([_work("working", transition="finalizing", work_id="a"), _work("working", work_id="b")])
    assert mixed["transition"] == "none"
    assert ws.roll_up_room([], complete=True)["activity"] == "idle"
    assert ws.roll_up_room([], complete=False)["mark"] == "ring"
    unconfirmed = ws.roll_up_room([_work("paused", confirmed=False)])
    assert unconfirmed["confirmed"] is False and unconfirmed["mark"] == "ring"


def _question(task_id, quiz_id="q1", *, state="open", project=None, **extra):
    asked = {"task_id": task_id, "quiz_id": quiz_id, "quiz_state": state, "chat_id": 1, **extra}
    return {**asked, "project_id": project[0], "project_chat_id": project[1]} if project else {**asked, "native": True}


def test_a_current_question_is_its_rooms_turn_as_mains_pill_counts_it():
    """The owner's turn is one set: what Main's pill counts in a room is that room's amber, whether the asking task
    still waits, goes on with an assumption or has ended; a question that waits for nobody lights nothing."""
    project = ("p1", 42)
    live = {**_work("working", work_id="w"), "room": 42, "release": "self", "members": {"ids": ["w", "helper"]}}
    asleep = {**_work("waiting", work_id="s"), "room": 42, "release": "self", "members": {"ids": ["s"]}}
    waits = {**_work("waiting", work_id="b", turn=[{"kind": "question", "task_id": "b"}]), "room": 42,
             "release": "owner", "members": {"ids": ["b"]}}
    works = [live, asleep, waits]
    loose = ws.place_questions(works, [
        _question("helper", project=project),                                  # a helper asked and went on
        _question("s", "q2", project=project),                                 # its task sleeps on something else
        _question("b", "q3", project=project),                                 # the member's own wait already says it
        _question("ended", "q4", state="expired_terminal", project=project),   # its task ended: no work holds it
        _question("main-ended", "q5"),                                         # Main's own
        _question("answered", "q6", state="answered", project=project),        # settled: waits for nobody
        _question("elsewhere", "q7", project=("p2", 77)),                      # a Project with no work at all
    ])
    assert live["turn"] == [{"kind": "open_question", "task_id": "helper", "quiz_id": "q1"}]
    assert asleep["turn"] == [{"kind": "open_question", "task_id": "s", "quiz_id": "q2"}]
    # Said once, as the wait its task holds: the wait that named no quiz takes the one open question's address.
    assert waits["turn"] == [{"kind": "question", "task_id": "b", "quiz_id": "q3"}]
    # The works keep their word and who moves them: an open question is the owner's turn, never the work's state.
    assert [(work["activity"], work["release"]) for work in works] == [
        ("working", "self"), ("waiting", "self"), ("waiting", "owner")]
    assert loose == {"42": [{"kind": "open_question", "task_id": "ended", "quiz_id": "q4"}],
                     "1": [{"kind": "open_question", "task_id": "main-ended", "quiz_id": "q5"}],
                     "77": [{"kind": "open_question", "task_id": "elsewhere", "quiz_id": "q7"}]}
    room = ws.roll_up_room(works, questions=loose["42"])
    assert room["amber"] is True and room["activity"] == "working"
    assert [item["task_id"] for item in room["turn"]] == ["helper", "s", "b", "ended"]
    assert room["turn"][0]["work_id"] == "w" and "work_id" not in room["turn"][-1]
    # A room at rest with an open question is the owner's turn too; without one it stays quiet.
    rest = ws.roll_up_room([], questions=loose["77"])
    assert (rest["activity"], rest["mark"], rest["amber"]) == ("idle", "none", True)
    assert ws.roll_up_room([])["amber"] is False and ws.place_questions([{**live, "turn": []}], []) == {}
    # The tray's count: each work with a turn once, and each question no work holds.
    assert ws.roll_up_summary({"42": room, "77": rest}, works)["turn"] == 3 + 2


def test_a_wait_that_names_no_quiz_never_absorbs_an_exact_question():
    """An exact current question keeps its address: a member's wait that names no quiz takes the address of its
    task's one open question (it stays the question the task waits for); with several, each stands beside it as its own
    item, and the tray still counts each question once."""
    def waiting():
        return {**_work("waiting", work_id="b", turn=[{"kind": "question", "task_id": "b"}]), "room": 42,
                "release": "owner", "members": {"ids": ["b"]}}
    project = ("p1", 42)
    one = waiting()
    assert ws.place_questions([one], [_question("b", "q1", project=project)]) == {}
    assert one["turn"] == [{"kind": "question", "task_id": "b", "quiz_id": "q1"}]
    two = waiting()
    ws.place_questions([two], [_question("b", "q1", project=project), _question("b", "q2", project=project)])
    assert two["turn"] == [{"kind": "question", "task_id": "b"},
                           {"kind": "open_question", "task_id": "b", "quiz_id": "q1"},
                           {"kind": "open_question", "task_id": "b", "quiz_id": "q2"}]
    assert ws.roll_up_summary({"42": ws.roll_up_room([two])}, [two])["turn"] == 1


def test_an_answer_whose_delivery_is_unknown_is_named_by_its_code():
    """A Project's answered question whose delivery a restart left unknown is the owner's turn (Main's pill counts it)
    and says so: its item carries ``delivery_unknown``; an unanswered one carries no code."""
    owed = {"state": "owed", "reason_code": "late_answer_delivery_unknown"}
    loose = ws.place_questions([], [_question("ended", "q1", state="answered", project=("p1", 42), question_delivery=owed),
                                    _question("open", "q2", project=("p1", 42))])
    assert loose == {"42": [{"kind": "open_question", "task_id": "ended", "quiz_id": "q1", "code": "delivery_unknown"},
                            {"kind": "open_question", "task_id": "open", "quiz_id": "q2"}]}


def test_one_question_is_counted_once_while_both_of_its_rooms_show_it():
    """A private Presence question is asked in Main while the work that asked it waits in its Project: both rooms are
    the owner's turn (Main, where the card is answered; the Project, where the work stands), and the tray counts the
    question once. A loose question no counted work carries still counts, and another quiz of the same task too."""
    waits = {**_work("waiting", work_id="b", turn=[{"kind": "question", "task_id": "b", "quiz_id": "q1"}]),
             "room": 42, "release": "owner", "members": {"ids": ["b"]}}
    loose = ws.place_questions([waits], [_question("b", "q1"), _question("b", "q2"), _question("main", "q3")])
    rooms = {"42": ws.roll_up_room([waits]), "1": ws.roll_up_room([], questions=loose["1"])}
    assert rooms["42"]["amber"] is True and rooms["1"]["amber"] is True
    assert [item["quiz_id"] for item in rooms["1"]["turn"]] == ["q1", "q2", "q3"]
    assert ws.roll_up_summary(rooms, [waits])["turn"] == 1 + 2  # the work, then q2 and q3; q1 is the work's own
    # A work's turn that names no quiz carries its task's question as well; an action never stands for a question.
    bare = {**waits, "turn": [{"kind": "question", "task_id": "b"}]}
    assert ws.roll_up_summary(rooms, [bare])["turn"] == 1 + 1
    acting = {**waits, "turn": [{"kind": "action", "task_id": "b"}]}
    assert ws.roll_up_summary(rooms, [acting])["turn"] == 1 + 3


# ---- the server adapter ------------------------------------------------------------------------------

_REAL_OPEN_RUNS = wp._open_runs  # the custody reader itself, before a fixture fakes it

@pytest.fixture
def stored(monkeypatch):
    results = {}
    monkeypatch.setattr(wp._Observations, "result", lambda self, task_id: results.get(task_id))
    monkeypatch.setattr(wp, "_open_runs", lambda drive_root: {})
    monkeypatch.setattr(wp, "_inflight_upkeep", lambda drive_root, known: [])
    return results


def test_an_answered_root_whose_late_work_runs_keeps_its_works_identity(stored, tmp_path):
    """The census marks an answered root's late work under a settling Pause as running and nothing else: that entry
    carries no identity, so the work still takes its id and name from the episode's stored result."""
    stored["late"] = {"task_id": "late", "status": "completed", "chat_id": 42,
                      "metadata": {"continuation": {"work_id": "prev-1", "work_name": "Original work"}}}
    owners = [row("late", "budget_pausing", pause_cause="owner")]
    [work] = wp.works_snapshot(tmp_path, owners, {"late": {"running": True}})["works"]
    assert (work["work_id"], work["name"], work["live"]) == ("prev-1", "Original work", True)


def test_a_held_continue_with_helpers_of_its_failed_attempt_is_one_working_work(stored, tmp_path):
    stored["p"] = {"task_id": "p", "status": "failed", "suggested_name": "Ouroboros 7.7.0 — finish the saved release",
                   "title": "НОВОЕ ПРЯМОЕ решение"}
    owners = [row("c", "budget_paused", pause_cause="unknown", can_resume=False),
              *[row(f"h{index}", parent_task_id="p", root_task_id="p") for index in range(3)]]
    raw = {"c": {"_budget_pause_hold": {"reason": "continuation_writer_unsettled"},
                 "title": "Continue: НОВОЕ ПРЯМОЕ решение", "continuation": {"predecessor_task_id": "p"}}}
    snap = wp.works_snapshot(tmp_path, owners, raw)
    assert snap["works_complete"] is True and len(snap["works"]) == 1
    work = snap["works"][0]
    assert work["work_id"] == "p" and work["name"] == "Ouroboros 7.7.0 — finish the saved release"
    assert work["activity"] == "working" and work["because"][0]["count"] == 3
    assert work["room"] == 42
    assert [(entry["action"], entry["target"]) for entry in work["controls"]] == [
        ("stop", "c"), ("stop", "p"), ("stop_earlier", "p")]
    room = snap["rooms"]["42"]
    assert room["activity"] == "working" and room["amber"] is False


def test_every_live_owner_row_is_a_member_of_exactly_one_work(stored, tmp_path):
    stored["p"] = {"task_id": "p"}
    owners = [row("p", "working"), row("x", "queued", chat_id=1), row("d", "preparing", kind="direct_chat", chat_id=1),
              row("h", parent_task_id="p", root_task_id="p"), row("c2", "queued"),
              row("c3", "budget_paused", pause_cause="owner")]
    raw = {"c2": {"continuation": {"predecessor_task_id": "p"}}, "c3": {"continuation": {"work_id": "p"}}}
    snap = wp.works_snapshot(tmp_path, owners, raw)
    seen = [member for work in snap["works"] for member in work["members"]["ids"]]
    assert sorted(seen) == sorted(item["activity_id"] for item in owners)
    by_id = {work["work_id"]: work for work in snap["works"]}
    assert sorted(by_id["p"]["members"]["ids"]) == ["c2", "c3", "h", "p"]


def test_legacy_chain_of_three_episodes_groups_under_its_first_root(stored, tmp_path):
    stored.update({"r": {"task_id": "r", "suggested_name": "First"},
                   "e2": {"task_id": "e2", "metadata": {"continuation": {"predecessor_task_id": "r"}}}})
    owners = [row("e3", "queued"), row("h", parent_task_id="r", root_task_id="r")]
    raw = {"e3": {"continuation": {"predecessor_task_id": "e2"}}}
    snap = wp.works_snapshot(tmp_path, owners, raw)
    assert [work["work_id"] for work in snap["works"]] == ["r"] and snap["works"][0]["name"] == "First"


def test_an_unreadable_episode_record_is_a_disclosed_gap(stored, tmp_path):
    owners = [row("e3", "queued")]
    snap = wp.works_snapshot(tmp_path, owners, {"e3": {"continuation": {"predecessor_task_id": "gone"}}})
    assert snap["works"][0]["work_id"] == "e3"
    assert snap["works_gaps"] == [{"reason": "episode_record_unavailable", "task_id": "gone"}]


def test_a_failing_source_never_reads_as_an_empty_room(monkeypatch, tmp_path):
    monkeypatch.setattr(wp, "_snapshot", lambda *args, **kwargs: 1 / 0)
    snap = wp.works_snapshot(tmp_path, [row("t")], {})
    assert snap["works_complete"] is False and snap["works_gaps"][0]["reason"] == "work_projection_unavailable"


def test_started_delegated_runs_of_an_ended_root_are_its_live_work(stored, monkeypatch, tmp_path):
    """Through the real custody store: started runs under an ended root are live members that its Stop reaches,
    a settled run custody still holds is not, and the stored ISO start becomes epoch seconds."""
    from datetime import datetime, timezone

    from ouroboros import delegate_custody as custody

    monkeypatch.setattr(wp, "_open_runs", _REAL_OPEN_RUNS)
    stored["p"] = {"task_id": "p", "status": "done", "chat_id": 42, "started_at": "2026-10-17T11:00:00Z"}
    stored["q"] = {"task_id": "q", "status": "done", "chat_id": 43, "started_at": "2026-10-17T10:00:00Z"}
    for run_id in ("run-1", "run-2"):
        assert custody.emit(tmp_path, custody.STARTED, {"run_id": run_id, "task_id": "p-h", "root_task_id": "p"})
    assert custody.emit(tmp_path, custody.STARTED, {"run_id": "kept", "task_id": "q", "root_task_id": "q",
                                                     "project_id": "x", "project_owned": True})
    assert custody.emit(tmp_path, custody.SETTLED, {"run_id": "kept", "task_id": "q", "state": "succeeded"})
    snap = wp.works_snapshot(tmp_path, [], {})
    assert snap["works_complete"] is True, snap["works_gaps"]
    [work] = snap["works"]
    assert work["work_id"] == "p" and work["activity"] == "working" and work["because"] == []
    assert [(tail["kind"], tail["state"], tail["count"]) for tail in work["tails"]] == [("run", "live", 2)]
    assert work["tails"][0]["text"] == "2 external runs still working"
    assert work["since"] == datetime(2026, 10, 17, 11, tzinfo=timezone.utc).timestamp()
    # The cascade Stop cancels a settled tree's open runs (tests/test_settled_tree_run_stop.py drives the door).
    assert work["controls"] == [{"action": "stop", "target": "p", "scope": "cascade"}]


def _run(run_id, task_id, source="delegated_subagent", terminal_state=""):
    return SimpleNamespace(run_id=run_id, task_id=task_id, root_task_id="", source=source, terminal_state=terminal_state)


def test_working_helpers_and_runs_of_an_ended_root_are_its_tails(stored, monkeypatch, tmp_path):
    stored["p"] = {"task_id": "p", "status": "failed", "chat_id": 42}
    monkeypatch.setattr(wp, "_open_runs", lambda drive_root: {"p": [_run("run-1", "p-a"), _run("run-2", "p-b")]})
    snap = wp.works_snapshot(tmp_path, [row("h0", parent_task_id="p", root_task_id="p")], {})
    [work] = snap["works"]
    assert work["activity"] == "working" and work["because"] == []
    assert [(tail["kind"], tail["count"], tail["state"]) for tail in work["tails"]] == [
        ("helper", 1, "live"), ("run", 2, "live")]


# ---- what still runs beside a paused episode (the 20.10 card) -----------------------------------------

def _paused(**raw):
    return ws.classify_member(row("t", "budget_paused", pause_cause="owner", finishing_reviews=raw.pop("reviews", False)),
                              {"pause_by": {"actor": "owner", "at": "T0"}, **raw})


@pytest.mark.parametrize("tail", [
    {"run_id": "rv", "kind": "review", "state": "live"},
    {"run_id": "x", "kind": "run", "state": "live"},
])
def test_a_paused_episode_keeps_its_word_whatever_still_runs_beside_it(tail):
    paused = _paused(reviews=tail["kind"] == "review")
    work = ws.roll_up_work("t", [paused], episode={"task_id": "t", "state": paused}, runs=[tail])
    # Never "Working · by you": the word and its reason are the episode's; what runs is the second fact.
    assert (work["activity"], work["because"][0]["code"], work["live"]) == ("paused", "paused_by_owner", True)
    assert [entry["kind"] for entry in work["tails"]] == [tail["kind"]]
    assert work["mark"] == "pause"  # the work's own mark follows its parked episode (owner 06–08.10)
    room = ws.roll_up_room([work])
    assert (room["activity"], room["mark"]) == ("paused", "moving")  # the room moves while anything runs
    controls = ws.controls_for(paused, row("t", "budget_paused"))
    assert [entry["action"] for entry in controls] == ["resume", "stop"]
    # Once it ends, the bars stand alone.
    quiet = ws.roll_up_work("t", [_paused()], episode={"task_id": "t", "state": _paused()})
    assert (quiet["tails"], quiet["live"], ws.roll_up_room([quiet])["mark"]) == ([], False, "pause")


def test_a_run_nobody_can_confirm_beside_a_live_episode_is_its_line_never_its_mark():
    # A Pause held for the owner beside a failed helper's open run: bars and the owner's move, the run on the line.
    held = ws.classify_member(row("t", "budget_paused", pause_cause="restart", can_resume=True),
                              {"hold": {"reason": "owner_restart_hold"}})
    lost = {"run_id": "u", "kind": "run", "state": "unknown"}
    work = ws.roll_up_work("t", [held], episode={"task_id": "t", "state": held}, runs=[lost])
    assert (work["activity"], work["mark"], work["live"]) == ("paused", "pause", False)
    assert [tail["code"] for tail in work["tails"]] == ["external_runs_unconfirmed"]
    assert ws.roll_up_room([work])["mark"] == "ring", "the room: nothing runs, one run nobody can confirm"
    # With its episode ended, that run is all the work is: unconfirmed, its ring.
    ended = ws.roll_up_work("t", [], episode={"task_id": "t"}, runs=[lost])
    assert (ended["activity"], ended["mark"]) == ("unknown", "ring")


def test_a_running_episodes_own_review_is_its_work_and_review_beside_a_pause_is_its_tail():
    running = ws.classify_member(row("t", "thinking"))
    review = {"run_id": "rv", "kind": "review", "state": "live"}
    work = ws.roll_up_work("t", [running], episode={"task_id": "t", "state": running}, runs=[review])
    assert (work["activity"], work["tails"], work["live"]) == ("working", [], True)
    paused = _paused()
    work = ws.roll_up_work("t", [paused], episode={"task_id": "t", "state": paused}, runs=[review])
    assert [tail["code"] for tail in work["tails"]] == ["review_work_finishing"]


@pytest.mark.parametrize("source, owner, typed", [
    ("review_substrate", {"status": "failed"}, ("review", "live")),  # a review finishes whatever its task did
    ("review_substrate_plan", None, ("review", "live")),
    ("delegated_subagent", None, ("run", "live")),
])
def test_one_rule_types_an_open_run_for_the_card_and_for_ouroboros(source, owner, typed):
    assert ws.run_tail(source, owner) == typed
    # A stop its own task's live Pause could not confirm leaves an ordinary run unconfirmed; review work stays review.
    assert ws.run_tail(source, owner, stop_unknown=True) == ((typed[0], "live") if typed[0] == "review"
                                                             else ("run", "unknown"))
    entry = SimpleNamespace(run_id="r", task_id="h" if owner is not None else "", source=source)
    stops = {}
    obs = SimpleNamespace(result=lambda task_id: owner if task_id == "h" else None,
                          stop_unknown=lambda task_id, drive_root=None, owner=None: frozenset(stops.get(task_id, ())))
    assert wp._typed_run(entry, obs, {}) == {"run_id": "r", "kind": typed[0], "state": typed[1]}
    stops["h"] = ["r"]
    stopped = "unknown" if typed[0] == "run" and entry.task_id else typed[1]
    assert wp._typed_run(entry, obs, {}) == {"run_id": "r", "kind": typed[0], "state": stopped}


def test_a_stop_is_unconfirmed_only_while_its_tasks_pause_lives(tmp_path):
    from ouroboros import budget_pause
    from ouroboros.store_hot_facts import controls
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(tmp_path, "t", STATUS_RUNNING)
    assert wp.unconfirmed_stops(tmp_path, "t") == frozenset()  # no pause row proves no trouble
    observed = {"runs": [{"run_id": "a", "state": "stop_unknown"}, {"run_id": "b", "state": "stop_requested"},
                         {"run_id": "", "state": "stop_unknown", "invocation_id": "i"}]}
    budget_pause.set_budget_pause(tmp_path, "t", {"pause_id": "p", "state": budget_pause.STATE_PAUSED,
                                                  "external_runs": observed})
    assert wp.unconfirmed_stops(tmp_path, "t") == frozenset({"a"})
    budget_pause.set_budget_pause(tmp_path, "t", {**controls(tmp_path, "t")["budget_pause"],
                                                  "state": budget_pause.STATE_RESUMED}, expected_pause_id="p")
    assert wp.unconfirmed_stops(tmp_path, "t") == frozenset(), "a resumed Pause's old observation is history"
    # A warm park keeps its own observation on its stored result while it waits under the owner's Pause.
    warm = {"owner_wait": {"state": "waiting", "reason": "owner_pause",
                           "owner_pause": {"external_runs": {"runs": [{"run_id": "w", "state": "stop_unknown"},
                                                                      {"run_id": "x", "state": "stop_confirmed"}]}}}}
    assert wp.unconfirmed_stops(tmp_path, "t", warm) == frozenset({"w"})
    for ended in ({**warm["owner_wait"], "state": "resumed"}, {**warm["owner_wait"], "reason": "question"}):
        assert wp.unconfirmed_stops(tmp_path, "t", {"owner_wait": ended}) == frozenset()


def test_a_review_the_pause_lets_finish_is_one_review_tail_from_either_source(stored, monkeypatch, tmp_path):
    monkeypatch.setattr(wp, "_open_runs", lambda drive_root: {"t": [_run("run-rv", "t", source="review_substrate")]})
    owners = [row("t", "budget_paused", pause_cause="owner", finishing_reviews=True)]
    snap = wp.works_snapshot(tmp_path, owners, {})
    [work] = snap["works"]
    assert work["activity"] == "paused"
    assert [(tail["kind"], tail["text"], tail["ids"]) for tail in work["tails"]] == [
        ("review", "Review work finishing", ["run-rv"])]
    assert "count" not in work["tails"][0]
    # A native review (no custody row) is the same tail; with no review at all there is none.
    monkeypatch.setattr(wp, "_open_runs", lambda drive_root: {})
    [work] = wp.works_snapshot(tmp_path, owners, {})["works"]
    assert [(tail["kind"], tail.get("ids")) for tail in work["tails"]] == [("review", None)]
    [work] = wp.works_snapshot(tmp_path, [row("t", "budget_paused", pause_cause="owner")], {})["works"]
    assert work["tails"] == [] and work["live"] is False


def test_a_run_whose_own_task_failed_is_unconfirmed_and_its_work_stays_stoppable(stored, monkeypatch, tmp_path):
    stored["p"] = {"task_id": "p", "status": "failed", "chat_id": 42}
    stored["h"] = {"task_id": "h", "status": "failed"}
    monkeypatch.setattr(wp, "_open_runs", lambda drive_root: {"p": [_run("run-u", "h")]})
    [work] = wp.works_snapshot(tmp_path, [], {})["works"]
    assert (work["activity"], work["live"], work["mark"]) == ("unknown", False, "ring")
    assert [(tail["state"], tail["text"]) for tail in work["tails"]] == [("unknown", "1 external run may still be running")]
    # Pause reaches it too once its root ended (#2669: ``owner_pause_control._ended_tail``, the door's own read).
    assert work["confirmed"] is True and work["controls"] == [{"action": "pause", "target": "p"},
                                                              {"action": "stop", "target": "p", "scope": "cascade"}]
    # The same run of a task that answered is live work (owner 19.10: Stop reaches it).
    stored["h"] = {"task_id": "h", "status": "completed"}
    [work] = wp.works_snapshot(tmp_path, [], {})["works"]
    assert (work["activity"], work["mark"], work["tails"][0]["state"]) == ("working", "moving", "live")


def test_a_run_the_engine_already_finished_is_no_member(stored, monkeypatch, tmp_path):
    """Through the real custody store: an engine-finished run whose settlement is still owed runs nothing."""
    from ouroboros import delegate_custody as custody

    monkeypatch.setattr(wp, "_open_runs", _REAL_OPEN_RUNS)
    stored["p"] = {"task_id": "p", "status": "completed", "chat_id": 42}
    assert custody.emit(tmp_path, custody.STARTED, {"run_id": "r-done", "task_id": "p", "root_task_id": "p"})
    assert custody.emit(tmp_path, custody.STARTED, {"run_id": "r-live", "task_id": "p", "root_task_id": "p"})
    [work] = wp.works_snapshot(tmp_path, [], {})["works"]
    assert work["tails"][0]["ids"] == ["r-done", "r-live"]
    assert custody.emit(tmp_path, custody.TERMINAL_OBSERVED, {
        "run_id": "r-done", "task_id": "p", "state": "succeeded", "observation_seq": 1,
        "source_ref": {"sha256": "0" * 64}})
    [work] = wp.works_snapshot(tmp_path, [], {})["works"]
    assert work["tails"][0]["ids"] == ["r-live"]



# ---- controls -----------------------------------------------------------------------------------------

def _actions(controls):
    return [(entry["action"], entry["target"]) for entry in controls]


def test_held_continue_offers_stop_and_stop_earlier_never_resume_or_pause():
    held_row = row("c", "budget_paused", pause_cause="unknown", can_resume=False)
    current = ws.classify_member(held_row, {"hold": {"reason": "continuation_writer_unsettled"}})
    controls = ws.controls_for(current, held_row, earlier_live=["p"], live_roots=["p"])
    assert _actions(controls) == [("stop", "c"), ("stop", "p"), ("stop_earlier", "p")]
    assert not {"resume", "pause"} & {entry["action"] for entry in controls}


@pytest.mark.parametrize("kind", ["managed_task", "direct_chat"])
def test_a_live_root_offers_wrap_up_hurry_pause_and_stop_managed_or_direct(kind):
    working = row("t", kind=kind)
    controls = ws.controls_for(ws.classify_member(working), working)
    assert [entry["action"] for entry in controls] == ["wrap_up", "hurry", "pause", "stop"]
    held = row("c", "budget_paused", kind=kind, pause_cause="unknown", can_resume=False)
    state = ws.classify_member(held, {"hold": {"reason": "continuation_writer_unsettled"}})
    assert [entry["action"] for entry in ws.controls_for(state, held)] == ["stop"], "held: only Stop"


@pytest.mark.parametrize("kind", ["managed_task", "direct_chat"])
def test_post_answer_upkeep_keeps_its_controls_and_a_settling_pause_keeps_its_pause(kind):
    finishing = row("f", "finalizing", kind=kind)
    assert [entry["action"] for entry in ws.controls_for(ws.classify_member(finishing), finishing)] == [
        "wrap_up", "hurry", "pause", "stop"]
    pausing = row("f", "budget_pausing", kind=kind, pause_cause="owner")
    assert [entry["action"] for entry in ws.controls_for(ws.classify_member(pausing, {"running": True}), pausing)] == ["pause", "stop"]
    paused = row("f", "budget_paused", kind=kind, pause_cause="owner", can_resume=True)
    assert "pause" not in [entry["action"] for entry in ws.controls_for(ws.classify_member(paused), paused)]


def test_a_live_retry_is_the_current_episode_of_its_original(stored, tmp_path):
    """A timeout retry carries its original's identity: it is that work's live episode, at its own id."""
    stored["held"] = {"task_id": "held", "status": "interrupted", "chat_id": 1}
    retry = row("t-retry", "budget_paused", pause_cause="owner", can_resume=True, original_task_id="held")
    snap = wp.works_snapshot(tmp_path, [retry], {"t-retry": {"_budget_pause": {"reason": "owner"}}})
    [work] = snap["works"]
    assert work["work_id"] == "held" and work["episode"] == {"task_id": "t-retry", "live": True}
    assert work["activity"] == "paused" and work["room"] == 42
    actions = [(entry["action"], entry["target"]) for entry in work["controls"]]
    assert ("resume", "t-retry") in actions
    # Stop ends the whole work: the retry, then its original's tree (a helper may still name that root).
    assert [target for action, target in actions if action == "stop"] == ["t-retry", "held"]
    assert not any(action == "stop_earlier" for action, _ in actions)


def test_a_retry_and_the_helpers_it_started_are_one_work(stored, tmp_path):
    """Helpers that name the retry as their root still belong to the original's work: one row, the retry live."""
    stored["held"] = {"task_id": "held", "status": "interrupted", "chat_id": 42}
    stored["t-retry"] = {"task_id": "t-retry", "status": "running", "chat_id": 42, "original_task_id": "held"}
    retry = row("t-retry", original_task_id="held")
    helper = row("h", parent_task_id="t-retry", root_task_id="t-retry")
    snap = wp.works_snapshot(tmp_path, [retry, helper], {"t-retry": {"original_task_id": "held"}})
    [work] = snap["works"]
    assert work["work_id"] == "held" and work["episode"] == {"task_id": "t-retry", "live": True}
    assert work["members"]["ids"] == ["t-retry", "h"] and work["activity"] == "working"
    assert not any(entry["action"] == "stop_earlier" for entry in work["controls"])


def test_a_held_continue_offers_stop_earlier_only_where_the_stop_door_reaches(stored, tmp_path):
    """A writer the queue does not list (an unbound start, a handoff, a process) is a wait the Stop door cannot end
    (it answers 404 for a settled tree with no live row): no dead control. A live helper of that attempt can."""
    stored["p"] = {"task_id": "p", "status": "failed", "chat_id": 42}
    held = row("c", "budget_paused", pause_cause="unknown", can_resume=False)
    raw = {"c": {"_budget_pause_hold": {"reason": "continuation_writer_unsettled"},
                 "continuation": {"predecessor_task_id": "p"}}}
    [alone] = wp.works_snapshot(tmp_path, [held], raw)["works"]
    assert [(entry["action"], entry["target"]) for entry in alone["controls"]] == [("stop", "c")]
    helper = row("h0", parent_task_id="p", root_task_id="p")
    [helped] = wp.works_snapshot(tmp_path, [held, helper], raw)["works"]
    assert [(entry["action"], entry["target"]) for entry in helped["controls"]] == [
        ("stop", "c"), ("stop", "p"), ("stop_earlier", "p")]


@pytest.mark.parametrize("cause, raw, resumable", [
    ("owner", {"fence": {"requested_by": "owner"}}, True),
    ("restart", {"hold": {"reason": "saved_work_hold"}}, True),
    ("mystery", {}, True),  # an existing census Resume survives an absent stop / older cause vocabulary
])
def test_resume_is_offered_only_where_the_resume_door_releases_the_pause(cause, raw, resumable):
    parked = row("t", "budget_paused", pause_cause=cause, can_resume=True)
    actions = {entry["action"] for entry in ws.controls_for(ws.classify_member(parked, raw), parked)}
    assert ("resume" in actions) is resumable and "stop" in actions and "pause" not in actions


def test_an_ended_episode_with_running_helpers_offers_only_stop_on_its_tree():
    controls = ws.controls_for(None, None, earlier_live=[], live_roots=["p"])
    assert _actions(controls) == [("stop", "p")]


def test_the_works_turn_and_mains_attention_share_one_rule():
    from ouroboros.gateway.main_decisions import owner_actions

    rows = [
        row("a", "budget_paused", pause_cause="budget", can_resume=True),
        row("b", "budget_paused", pause_cause="owner", can_resume=False),
        row("c", model_waits={"w": {"state": "waiting", "task_attempt": 1, "requires_owner_action": True, "revision": 1}}),
        row("d", model_waits={"w": {"state": "waiting", "task_attempt": 1, "requires_owner_action": True}}),  # no revision
        row("e", "budget_paused", pause_cause="restart", can_resume=False),
    ]
    for item in rows:
        waits, resume = owner_actions(item)
        owner_turns = [turn for turn in ws.classify_member(item)["turn"] if turn["kind"] == "action"]
        assert bool(owner_turns) is bool(waits or resume), item["activity_id"]


def test_a_silent_working_member_says_since_when_until_its_own_progress():
    """The dated silence fact (owner 09.11, Q2 A) leads a working member's reasons; it is gone once the work moved,
    and a member that is not working never borrows it."""
    silent = {"silence_attention": {"state": "pending_attention", "last_progress_at": 1794190200.0, "episode_id": "e"}}
    state = ws.classify_member(row("t", "working"), {"running": True, **silent})
    assert state["reasons"][0] == {"code": "no_progress", "text": "No recorded progress", "since": 1794190200.0}
    assert state["activity"] == "working" and state["live"] is True
    # The owner's recorded Stop leads the line; the silence fact stays right behind it.
    stopped = ws.classify_member(row("t", "working"), {"running": True, "cancel_pending": True, **silent})["reasons"]
    assert [r["code"] for r in stopped[:2]] == ["stop_requested", "no_progress"]
    # ...also over the step its turn stands at, which follows both.
    building = row("t", "preparing", turn_phase="context_building")
    assert [r["code"] for r in ws.classify_member(building, {"running": True, "cancel_pending": True, **silent})["reasons"]] == [
        "stop_requested", "no_progress", "turn_context_building"]
    assert ws.classify_member(building, {"running": True})["reasons"][0]["code"] == "turn_context_building"
    moved = {"silence_attention": {**silent["silence_attention"], "state": "progress_observed"}}
    assert all(r["code"] != "no_progress" for r in ws.classify_member(row("t", "working"), {"running": True, **moved})["reasons"])
    assert all(r["code"] != "no_progress" for r in ws.classify_member(row("t", "working"), {"running": True})["reasons"])
    assert all(r["code"] != "no_progress" for r in ws.classify_member(row("q", "queued"), {**silent})["reasons"])


@pytest.mark.parametrize("carrier", ["managed", "lane_turn"])
def test_the_live_silence_fact_reaches_the_work_line_from_the_running_row(tmp_path, monkeypatch, carrier):
    """The fact lives on the RUNNING row (``silence_attention.observe``), never on its task: the census carries it to
    the work projection, so the line says since when while the episode is open and drops it at the source's progress.
    An owner turn the conversation lane carries is listed as its own direct row first; its silence joins that row."""
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.gateway.work_projection import works_snapshot
    from ouroboros.task_results import write_task_result
    from supervisor import silence_attention
    from tests._budget_pause_exact_helpers import _install_queue

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "quiet", "running", chat_id=7)
    meta = {"task": {"id": "quiet", "type": "task", "chat_id": 7, "root_task_id": "quiet"}, "attempt": 1,
            "started_at": 1794190000.0, "last_progress_at": 1794190200.0}
    if carrier == "lane_turn":
        from supervisor.conversation_lane import LANE_CONVERSATION
        meta["task"].update(chat_id=1, _is_direct_chat=True)
        meta.update(lane=LANE_CONVERSATION, worker_id=3)
    workers.RUNNING["quiet"] = meta

    def reasons():
        raw, owners = {}, []
        _chat_activities_snapshot_safe(tmp_path, work_facts=raw, decision_owners=owners)
        [work] = works_snapshot(tmp_path, owners, raw)["works"]
        return [(reason["code"], reason.get("since")) for reason in work["because"]]

    assert all(code != "no_progress" for code, _since in reasons())
    silence_attention.observe(queue, "quiet", meta, 1794193800.0, 600.0)
    assert reasons()[0] == ("no_progress", 1794190200.0)
    meta["last_progress_at"] = 1794193900.0
    silence_attention.recovered(queue, "quiet", meta, 1794193900.0)
    assert all(code != "no_progress" for code, _since in reasons())


# #2801: the word beside Finalizing… is a settled lifecycle's. A split-drive root's census row stays its canonical
# hot row (`running`) until copy-back; its settled word is the effective result's, the one its card reads.
_AUTHOR_STOP_AXES = {"schema_version": 1, "lifecycle": {"status": "completed"},
                     "execution": {"status": "ok", "task_completion": {"action": "stop", "source": "finish_task"}},
                     "objective": {"status": "fail", "source": "task_completion", "review_status": "skipped",
                                   "reason": "author_stop"}}


@pytest.mark.parametrize("child_status, expected", [("completed", "cancelled"), ("running", None)])
def test_a_finalizing_split_root_says_its_effective_outcome_not_its_hot_row(tmp_path, child_status, expected):
    from ouroboros.task_results import write_task_result

    data, child = tmp_path / "data", tmp_path / "child"
    data.mkdir()
    child.mkdir()
    write_task_result(child, "fr", child_status, **({"outcome_axes": _AUTHOR_STOP_AXES, "result": "partial"}
                                                      if child_status == "completed" else {}))
    write_task_result(data, "fr", "running", child_drive_root=str(child),
                      root_phase_checkpoint={"post_task_synthesis": "running"})
    row = {"activity_id": "fr", "phase": "finalizing", "status": "running", "kind": "managed_task", "chat_id": 1,
           "task_attempt": 1, "started_at": 100.0}
    work = next(w for w in wp.works_snapshot(data, [row], {"fr": {"running": True}})["works"] if w["work_id"] == "fr")
    assert work["transition"] == "finalizing"
    assert work.get("outcome") == expected
