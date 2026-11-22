"""One /api/state, one answer for where the owner's action waits: Main's pill (``main_attention``) and every room's amber.

``main_decisions_snapshot`` proves each owner action's address (the nearest durable binding, never the census's
provisional room) and availability; the work projection places its owner turn by that same measurement (``verdicts``),
so a nested helper's Resume lights the room it is offered in and not its root's, a read that could not prove an
address advertises none, and no work or question is counted or lost twice. Fictional tasks and rooms throughout.
"""
from __future__ import annotations

import pytest

from ouroboros import projects_registry
from ouroboros.gateway import work_projection as wp
from ouroboros.gateway.main_decisions import main_decisions_snapshot
from ouroboros.owner_quiz import record_asked, record_publication
from ouroboros.projects_registry import bind_task_to_project, create_project
from ouroboros.task_results import STATUS_RUNNING, write_task_result
from tests._queue_state import gateway, read_state  # noqa: F401

# ---- the real /api/state: a root in one room, a helper below a task another room owns ---------------------------

def _chain(root, bound_to):
    """``atlas-root → atlas-middle (bound to ``bound_to``) → atlas-parent → atlas-helper``: Main work whose middle
    task the owner turned into a Project. The census re-homes ``atlas-helper`` from its root's room, skipping that."""
    names = ["atlas-root", "atlas-middle", "atlas-parent", "atlas-helper"]
    for index, task_id in enumerate(names):
        write_task_result(root, task_id, "running", chat_id=1, root_task_id="atlas-root",
                          parent_task_id=names[index - 1] if index else "",
                          delegation_role="subagent" if index else "root")
    if bound_to is not None:
        bind_task_to_project(root, "atlas-middle", bound_to["id"], origin={"absent": "system"})


def _held(gateway, task_id="atlas-helper", parent="atlas-parent"):  # noqa: F811
    """A helper held until its owner's Resume, as its queue row carries it."""
    helper = {"id": task_id, "chat_id": 1, "budget_drive_root": str(gateway.root), "parent_task_id": parent,
              "root_task_id": "atlas-root", "delegation_role": "subagent",
              "_budget_pause_hold": {"reason": "owner_restart_hold", "selected": False, "dispatchable": False}}
    gateway.pending.append(helper)
    return helper


def _nested(gateway, *, bound=True):  # noqa: F811
    """The live root leases the OUTER room; its held helper sits below a middle task bound to the NEARER room."""
    outer = create_project(gateway.root, "outer-room", name="Outer room")
    nearer = create_project(gateway.root, "nearer-room", name="Nearer room")
    _chain(gateway.root, nearer if bound else None)
    bind_task_to_project(gateway.root, "atlas-root", outer["id"], origin={"absent": "system"})
    root = {"id": "atlas-root", "root_task_id": "atlas-root", "chat_id": outer["chat_id"], "project_id": outer["id"],
            "budget_drive_root": str(gateway.root), "_attempt": 3}
    gateway.running["atlas-root"] = {"task": root, "attempt": 3, "started_at": 1.0}
    _held(gateway)
    return outer, nearer


def _turn(room):
    return [(item["kind"], item["task_id"]) for item in room["turn"]]


def _amber(body, project):
    room = body["rooms"].get(str(project["chat_id"]))
    return None if room is None else (room["amber"], _turn(room))


def test_a_nested_helpers_action_lights_the_room_it_is_offered_in_not_its_roots(gateway):  # noqa: F811
    outer, nearer = _nested(gateway)
    body = read_state(gateway)
    [offer] = body["main_attention"]
    assert (offer["task_id"], offer["project_id"], offer["chat_id"]) == ("atlas-helper", nearer["id"], nearer["chat_id"])
    assert _amber(body, nearer) == (True, [("action", "atlas-helper")]), "the action stands where Main's pill offers it"
    assert _amber(body, outer) == (False, []), "and not at the root's provisional room"
    [work] = body["works"]
    assert (work["work_id"], work["room"], work["turn"], work["release"]) == (
        "atlas-root", outer["chat_id"], [], "self"), "the work keeps its home and its own word: the helper waits elsewhere"
    assert body["work_summary"]["turn"] == 1, "the one action is counted once"
    assert body["works_gaps"] == []


def test_a_read_that_cannot_prove_the_address_advertises_none_and_names_it(gateway, monkeypatch):  # noqa: F811
    """While the census cannot read the bindings the helper's room is the census's alone (``main_attention`` offers
    none). Its root's room is no proof either: no amber stands there, and the gap names the task."""
    outer, nearer = _nested(gateway)
    census_read = projects_registry.all_task_project_bindings

    def unreadable(*args, strict=False, **kwargs):
        if strict:
            raise ValueError("Project bindings are unavailable")
        return census_read(*args, **kwargs)
    monkeypatch.setattr(projects_registry, "all_task_project_bindings", unreadable)
    body = read_state(gateway)
    assert body["main_attention"] == [] and body["main_attention_complete"] is False
    assert _amber(body, outer) == (False, []) and _amber(body, nearer) is None
    assert body["work_summary"]["turn"] == 0
    assert body["works_gaps"] == [{"reason": "owner_action_unconfirmed", "task_id": "atlas-helper"}]
    [work] = body["works"]
    assert work["room"] == outer["chat_id"], "the work stays where the census put it"


def test_an_old_body_gives_way_to_the_fresh_one_when_a_binding_lands(gateway):  # noqa: F811
    """Reads of one real state across its changes: the room an action stands in is the room the NEWEST read proves
    (the publication's older body never holds an address its binding has left), and a resumed helper leaves both."""
    outer, nearer = _nested(gateway, bound=False)  # no task between the helper and its root is bound yet
    first = read_state(gateway)
    [offer] = first["main_attention"]
    assert (offer["project_id"], offer["chat_id"]) == (outer["id"], outer["chat_id"])
    assert _amber(first, outer) == (True, [("action", "atlas-helper")]) and _amber(first, nearer) is None
    bind_task_to_project(gateway.root, "atlas-middle", nearer["id"], origin={"absent": "system"})
    second = read_state(gateway)
    [offer] = second["main_attention"]
    assert (offer["project_id"], offer["chat_id"]) == (nearer["id"], nearer["chat_id"])
    assert _amber(second, nearer) == (True, [("action", "atlas-helper")]) and _amber(second, outer) == (False, [])
    assert second["work_summary"]["turn"] == 1
    gateway.pending.clear()  # the helper is resumed: its row leaves the queue
    third = read_state(gateway)
    assert third["main_attention"] == [] and _amber(third, nearer) is None and _amber(third, outer) == (False, [])
    assert third["work_summary"]["turn"] == 0


def test_the_projection_reads_the_registry_no_more_for_more_helpers(gateway, monkeypatch):  # noqa: F811
    """The strict room rule is warmed once per measurement and the work projection adds no read of its own: nine held
    helpers cost the registry and bindings exactly what one does."""
    outer, nearer = _nested(gateway)
    counts = {"registry": 0, "bindings": 0, "resolver": 0}
    for module, name, key in ((projects_registry, "list_reserved_projects", "registry"),
                              (projects_registry, "_load_bindings", "bindings")):
        real = getattr(module, name)

        def counted(*args, _real=real, _key=key, **kwargs):
            counts[_key] += 1
            return _real(*args, **kwargs)
        monkeypatch.setattr(module, name, counted)
    real_resolver = projects_registry.RoomResolver

    class CountedResolver(real_resolver):
        def __init__(self, *args, **kwargs):
            counts["resolver"] += 1
            super().__init__(*args, **kwargs)
    monkeypatch.setattr(projects_registry, "RoomResolver", CountedResolver)

    def read():
        for key in counts:
            counts[key] = 0
        body = read_state(gateway)
        return dict(counts), body
    one, body = read()
    assert [row["task_id"] for row in body["main_attention"]] == ["atlas-helper"]
    assert all(one.values()), f"each reader is exercised, or the comparison below proves nothing: {one}"
    for index in range(8):
        write_task_result(gateway.root, f"atlas-helper-{index}", "running", chat_id=1, root_task_id="atlas-root",
                          parent_task_id="atlas-parent", delegation_role="subagent")
        _held(gateway, f"atlas-helper-{index}")
    nine, body = read()
    assert len(body["main_attention"]) == 9 and all(row["chat_id"] == nearer["chat_id"] for row in body["main_attention"])
    assert sorted(item["task_id"] for item in body["rooms"][str(nearer["chat_id"])]["turn"]) == sorted(
        row["task_id"] for row in body["main_attention"]), "every one of them stands in the proven room"
    assert body["work_summary"]["turn"] == 1, "nine actions of one work: the work is counted once, as when they stood in its turn"
    assert nine == one, "nothing multiplies by helpers"


# ---- census-shaped rows answered by both halves of one measurement ------------------------------------------------

def row(task_id, phase="working", **extra):
    return {"activity_id": task_id, "phase": phase, "kind": "managed_task", "chat_id": extra.pop("chat_id", 1),
            "task_attempt": 1, "started_at": extra.pop("started_at", 100.0), **extra}


def parked(task_id, **extra):
    """A parked row only its owner's Resume continues, addressed to Main unless ``extra`` says otherwise."""
    return row(task_id, "budget_paused", pause_cause=extra.pop("pause_cause", "budget"), can_resume=True,
               cancelable=extra.pop("cancelable", True), **extra)


WAIT = {"wait_id": "w1", "state": "waiting", "task_attempt": 1, "revision": 2, "reason": "auth",
        "requires_owner_action": True}


@pytest.fixture
def adapter(monkeypatch):
    """The projection's custody and upkeep reads answered empty, as the adapter's own tests do."""
    monkeypatch.setattr(wp, "_open_runs", lambda drive_root, state=None: {})
    monkeypatch.setattr(wp, "_inflight_upkeep", lambda drive_root, known: [])


def answer(root, rows, raw=None, *, bindings_complete=True, measured=True):
    """What ``/api/state`` builds from these census rows: the one measurement (``main_decisions_snapshot``) and the work
    projection that places its owner turn by that measurement (``measured``), or by the census rows alone (as before)."""
    verdicts = {}
    decisions = main_decisions_snapshot(root, rows, {}, bindings_complete=bindings_complete, verdicts=verdicts)
    works = wp.works_snapshot(root, rows, raw or {}, complete=bindings_complete,
                              questions=decisions["current_questions"], verdicts=verdicts if measured else None)
    return decisions, works, verdicts


def test_a_healthy_main_without_a_project_says_what_it_said(tmp_path, adapter):
    """No Project, readable bindings: Main's work, its questions and its actions are placed at Main by both halves of
    the measurement, so the measured projection is the very body the census rows alone gave."""
    write_task_result(tmp_path, "survey", STATUS_RUNNING, chat_id=1, root_task_id="survey")
    record_asked(tmp_path, "survey", quiz_id="scope", question="The whole folder?", options=["Yes", "No"], chat_id=1)
    record_publication(tmp_path, "survey", "scope", 1)
    rows = [parked("survey"), row("digest", model_waits={"w1": WAIT}), row("quiet")]
    decisions, measured, verdicts = answer(tmp_path, rows)
    _decisions, census_only, _ = answer(tmp_path, rows, measured=False)
    assert decisions["main_attention_complete"] is True and decisions["current_questions_complete"] is True
    assert {task: (verdict["state"], verdict["room"]) for task, verdict in verdicts.items()} == {
        "survey": ("offered", 1), "digest": ("offered", 1)}
    assert measured == census_only
    main = measured["rooms"]["1"]
    assert main["amber"] is True
    assert sorted((item["kind"], item["task_id"], item.get("wait_id") or item.get("quiz_id") or "") for item in main["turn"]) == [
        ("action", "digest", "w1"), ("action", "survey", ""), ("open_question", "survey", "scope")]
    assert measured["work_summary"]["turn"] == 2 and measured["works_gaps"] == []


def test_a_rooms_independent_questions_and_actions_survive_a_moved_action(tmp_path, adapter):
    """The root's own question and its Project's other work's action stay where they were while a nested helper's action
    leaves for the room the strict rule proved: nothing vanishes, and one work is one wait in the tray."""
    outer = create_project(tmp_path, "outer-room", name="Outer room")
    nearer = create_project(tmp_path, "nearer-room", name="Nearer room")
    _chain(tmp_path, nearer)
    bind_task_to_project(tmp_path, "atlas-root", outer["id"], origin={"absent": "system"})
    record_asked(tmp_path, "atlas-root", quiz_id="pick", question="Which one?", options=["A", "B"], chat_id=outer["chat_id"])
    record_publication(tmp_path, "atlas-root", "pick", outer["chat_id"])
    write_task_result(tmp_path, "side-work", STATUS_RUNNING, chat_id=nearer["chat_id"], project_id=nearer["id"])
    rows = [row("atlas-root", chat_id=outer["chat_id"], project_id=outer["id"]),
            parked("atlas-helper", chat_id=outer["chat_id"], project_id=outer["id"], pause_cause="restart", cancelable=False,
                   parent_task_id="atlas-parent", root_task_id="atlas-root", delegation_role="subagent"),
            row("side-work", chat_id=nearer["chat_id"], project_id=nearer["id"], model_waits={"w1": WAIT})]
    decisions, body, _ = answer(tmp_path, rows)
    assert decisions["main_attention_complete"] is True and decisions["current_questions_complete"] is True
    assert [(item["task_id"], item["chat_id"]) for item in decisions["main_attention"]] == [
        ("atlas-helper", nearer["chat_id"]), ("side-work", nearer["chat_id"])]
    outer_room, nearer_room = body["rooms"][str(outer["chat_id"])], body["rooms"][str(nearer["chat_id"])]
    assert _turn(outer_room) == [("open_question", "atlas-root")], "its own question stays"
    assert sorted(_turn(nearer_room)) == [("action", "atlas-helper"), ("action", "side-work")]
    assert [item.get("work_id") for item in sorted(nearer_room["turn"], key=lambda item: item["task_id"])] == [None, "side-work"], (
        "the helper's action stands with no work of this room; the room's own work still carries its action")
    assert body["work_summary"]["turn"] == 2, "the root (its question and its helper's action) and the side work: one each"


def test_a_partial_read_advertises_no_address_it_could_not_prove_and_keeps_the_ones_it_could(tmp_path, adapter):
    outer = create_project(tmp_path, "outer-room", name="Outer room")
    nearer = create_project(tmp_path, "nearer-room", name="Nearer room")
    _chain(tmp_path, nearer)
    bind_task_to_project(tmp_path, "atlas-root", outer["id"], origin={"absent": "system"})
    write_task_result(tmp_path, "side-work", STATUS_RUNNING, chat_id=nearer["chat_id"], project_id=nearer["id"])
    rows = [row("atlas-root", chat_id=outer["chat_id"], project_id=outer["id"]),
            parked("atlas-helper", chat_id=outer["chat_id"], project_id=outer["id"], pause_cause="restart", cancelable=False,
                   parent_task_id="atlas-parent", root_task_id="atlas-root", delegation_role="subagent"),
            parked("side-work", chat_id=nearer["chat_id"], project_id=nearer["id"])]
    decisions, body, verdicts = answer(tmp_path, rows, bindings_complete=False)
    assert decisions["main_attention_complete"] is False
    assert [item["task_id"] for item in decisions["main_attention"]] == ["side-work"]
    assert verdicts["atlas-helper"] == {"state": "unconfirmed"} and verdicts["side-work"]["state"] == "offered"
    assert body["rooms"][str(outer["chat_id"])]["amber"] is False, "the root's room is no proof of the helper's"
    assert _turn(body["rooms"][str(nearer["chat_id"])]) == [("action", "side-work")], (
        "a row at its own registered room needs no ancestry: it stays advertised")
    assert body["works_gaps"] == [{"reason": "owner_action_unconfirmed", "task_id": "atlas-helper"}]
    # Without the measurement the census rows decide, as before: the provisional room lights.
    _decisions, census_only, _ = answer(tmp_path, rows, bindings_complete=False, measured=False)
    assert census_only["rooms"][str(outer["chat_id"])]["amber"] is True and census_only["works_gaps"] == []


def test_rooms_the_measurement_does_not_cover_keep_what_the_census_says(tmp_path, adapter):
    """Work in a chat that is neither Main nor a registered Project is another room's: the measurement never offers
    its action (and says nothing of it), so the room's own turn stands as the census gives it."""
    other = create_project(tmp_path, "other-room", name="Other room")
    rows = [parked("hidden-work", chat_id=0), parked("foreign-work", chat_id=9),
            parked("contextual", chat_id=9, project_id=other["id"])]
    decisions, measured, verdicts = answer(tmp_path, rows)
    _decisions, census_only, _ = answer(tmp_path, rows, measured=False)
    assert decisions["main_attention"] == [] and decisions["main_attention_complete"] is True
    assert {task: verdict["state"] for task, verdict in verdicts.items()} == {
        "hidden-work": "elsewhere", "foreign-work": "elsewhere", "contextual": "elsewhere"}
    assert measured == census_only
    assert [measured["rooms"][chat]["amber"] for chat in ("0", "9")] == [True, True]
    assert measured["work_summary"]["turn"] == 3 and measured["works_gaps"] == []


def test_real_stop_and_review_states_are_nobodys_turn_with_or_without_the_measurement(tmp_path, monkeypatch):
    """The owner's own Pause settling, paused with a review it let finish, and paused: no amber in Main, no action in
    its pill, and the projection the measurement placed is the one the census rows alone gave. Real queue rows, the
    real Pause and its fence; only the review operation's liveness is answered."""
    from ouroboros import owner_pause, review_operation
    from ouroboros.gateway import state as gateway_state
    from supervisor.owner_pause_control import request_owner_pause
    from tests._budget_pause_exact_helpers import _install_queue

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "orbit-root", STATUS_RUNNING, chat_id=1, root_task_id="orbit-root")
    write_task_result(tmp_path, "orbit-child", STATUS_RUNNING, chat_id=1, root_task_id="orbit-root",
                      parent_task_id="orbit-root", delegation_role="subagent")
    workers.RUNNING["orbit-root"] = {"task": {"id": "orbit-root", "type": "task", "chat_id": 1,
                                              "root_task_id": "orbit-root"}, "worker_id": 0, "attempt": 1}
    workers.RUNNING["orbit-child"] = {"task": {"id": "orbit-child", "type": "task", "chat_id": 1,
                                               "root_task_id": "orbit-root", "parent_task_id": "orbit-root",
                                               "delegation_role": "subagent"}, "worker_id": 1, "attempt": 1}

    def body(measured):
        owners, facts, verdicts = [], {}, {}
        gateway_state._chat_activities_snapshot_safe(tmp_path, None, availability={"complete": True},
                                                     decision_owners=owners, work_facts=facts)
        decisions = main_decisions_snapshot(tmp_path, owners, {}, verdicts=verdicts)
        works = wp.works_snapshot(tmp_path, owners, facts, questions=decisions["current_questions"],
                                  verdicts=verdicts if measured else None)
        return decisions, works

    def same_and_quiet(state):
        decisions, works = body(True)
        assert works == body(False)[1], state
        assert decisions["main_attention"] == [] and works["rooms"]["1"]["amber"] is False, state
        assert works["work_summary"]["turn"] == 0 and works["works_gaps"] == [], state
        return works["works"][0]

    assert same_and_quiet("working")["activity"] == "working"
    owner = {"actor": "owner", "text": "", "at": "2026-11-20T00:43:00Z"}
    assert request_owner_pause("orbit-root", request_id="pause", by=owner)["ok"]
    assert same_and_quiet("pause requested")["transition"] == "pausing"
    fence = owner_pause.read_fence(tmp_path, "orbit-root")
    workers.RUNNING.clear()
    with monkeypatch.context() as patch:
        patch.setattr(review_operation, "task_has_live_review_operation",
                      lambda root, task_id, sent_only=False, **_: sent_only)
        owner_pause.set_fence_state(tmp_path, "orbit-root", fence_id=fence["fence_id"], state="paused",
                                    finishing_reviews=["orbit-root", "send-1"])
        reviewing = same_and_quiet("paused, its review finishing")
        assert [tail["code"] for tail in reviewing["tails"]] == ["review_work_finishing"]
    owner_pause.set_fence_state(tmp_path, "orbit-root", fence_id=fence["fence_id"], state="paused", finishing_reviews=[])
    assert same_and_quiet("paused")["activity"] == "paused"
