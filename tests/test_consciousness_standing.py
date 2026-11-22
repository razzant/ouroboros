"""What stands in every own-clock wake: the works that ended without success in an open project
(``consciousness_standing.fallen_works``), beside the outstanding owner cards, in every form of the wake."""

from __future__ import annotations

import json

import pytest

from ouroboros import consciousness_standing as standing
from ouroboros import consciousness_wake as wake
from ouroboros.owner_continue import OWNER_STOP_SOURCES
from ouroboros.terminal_time import task_attempt_witness
from tests._store_shared import read_result, seed_result

T0 = 1_800_000_000.0
PROJECTS = [{"id": "p1", "name": "Audit", "lifecycle": "active"}]


def _iso(ts):
    return wake._iso(ts)


def _root(root, task_id, *, ts, status="failed", project="p1", ended=None, **fields):
    """A work root as stored: ``ts`` is its start; ``ended`` its recorded terminal instant."""
    row = {"task_id": task_id, "status": status, "ts": _iso(ts), "updated_at": _iso(ended or ts), "_schema_version": 1,
           "description": f"do {task_id}", "metadata": {}, "reason_code": "provider_unavailable", **fields}
    if project:
        row["project_id"] = project
    if ended is not None:
        row["terminal_time"] = {"v": 1, "occurred_at": _iso(ended), "source": "executor_terminal",
                                "attempt": task_attempt_witness(row)}
    seed_result(root, task_id, row)
    return row


def _registry(root, *rows):
    (root / "state").mkdir(exist_ok=True)
    (root / "state" / "projects.json").write_text(json.dumps({"projects": list(rows)}), encoding="utf-8")


def _listed(root, projects=PROJECTS, now=T0):
    return [work.task_id for work in standing.fallen_works(root, now=now, projects=projects)]


def _observe(root, boundary=None, now=T0):
    return wake.observe_wake(root, boundary=boundary, since=T0 - 3600, now=now, reason="heartbeat")


def test_a_failed_root_stands_in_two_consecutive_wakes_while_its_transition_reaches_one(tmp_path):
    """The transition is the delta it was (once); the fact stands until an act retires it."""
    _registry(tmp_path, *PROJECTS)
    _root(tmp_path, "fell", ts=T0 - 1800, ended=T0 - 600, wake_summary="Mail audit: provider down at step 3.")
    first = _observe(tmp_path)
    assert any("fell" in line for _kind, _offset, line in first.events)
    second = _observe(tmp_path, first.boundary, now=T0 + 600)
    assert second.events == ()
    for observation, ago in ((first, "10 min ago"), (second, "20 min ago")):
        assert observation.standing[0].lines == (
            f"- task fell Failed in project Audit, ended {ago}: do fell; cause provider_unavailable; "
            "Continue open (provider_unavailable); get_task_result(task_id='fell')",)
        assert observation.counts()["fallen_work"] == 1
        text = observation.full_text()
        assert ("\nWorks that ended without success; nothing has started in their project since (1):\n"
                f"{observation.standing[0].lines[0]}") in text
    assert "provider down at step 3" not in second.full_text(), "the summary came once, with the transition"
    third = _observe(tmp_path, second.boundary, now=T0 + 1200)
    assert third.events == () and third.counts() == {"fallen_work": 1}


def test_the_section_follows_the_outstanding_cards_and_the_composition_counts_it(tmp_path):
    _registry(tmp_path, *PROJECTS)
    _root(tmp_path, "fell", ts=T0 - 1800, ended=T0 - 600)
    _root(tmp_path, "asker", ts=T0 - 7200, status="running", project="",
          owner_quiz={"q1": {"state": "open", "asked_at": _iso(T0 - 60), "question": "Go on?"}})
    observation = _observe(tmp_path)
    text = observation.full_text()
    assert text.index("Outstanding owner cards (1):") < text.index("Works that ended without success")
    assert observation.composition().endswith("1 outstanding card, 1 work that ended without success")


def test_a_root_started_later_in_its_project_takes_it_off_whatever_it_is_about(tmp_path):
    """Residual named in the map: an unrelated later work retires an earlier failure too."""
    _root(tmp_path, "fell", ts=T0 - 1800, ended=T0 - 600)
    assert _listed(tmp_path) == ["fell"]
    _root(tmp_path, "later", ts=T0 - 300, status="running")
    assert _listed(tmp_path) == []


def test_a_root_that_fails_while_a_later_started_root_exists_is_never_listed(tmp_path):
    """Residual named in the map: the index orders by start, so the newest-started root decides."""
    _root(tmp_path, "later", ts=T0 - 600, status="completed", ended=T0 - 500)
    _root(tmp_path, "overlap", ts=T0 - 1800, ended=T0 - 60)
    assert _listed(tmp_path) == []


def test_a_continue_successor_takes_its_predecessor_off_and_a_failed_one_names_it(tmp_path):
    _root(tmp_path, "fell", ts=T0 - 3600, ended=T0 - 3000)
    _root(tmp_path, "fell-c1", ts=T0 - 1200, status="running",
          metadata={"continuation": {"predecessor_task_id": "fell", "work_id": "work-7"}})
    seed_result(tmp_path, "fell", {**read_result(tmp_path, "fell"), "continued_by": {
        "successor_task_id": "fell-c1", "state": "admitted", "action_nonce": "n0000001"}})
    assert _listed(tmp_path) == []
    _root(tmp_path, "fell-c1", ts=T0 - 1200, ended=T0 - 60,
          metadata={"continuation": {"predecessor_task_id": "fell", "work_id": "work-7"}})
    [work] = standing.fallen_works(tmp_path, now=T0, projects=PROJECTS)
    assert work.task_id == "fell-c1"
    assert "; continuation of fell, work work-7; get_task_result(task_id='fell-c1')" in work.line
    _root(tmp_path, "fell-c1", ts=T0 - 1200, ended=T0 - 60, metadata={"continuation": {"predecessor_task_id": "fell"}})
    assert "; continuation of fell; get_task_result" in standing.fallen_works(tmp_path, now=T0, projects=PROJECTS)[0].line


def test_a_bound_continue_stays_listed_and_says_its_admission_is_unconfirmed(tmp_path):
    _root(tmp_path, "fell", ts=T0 - 1800, ended=T0 - 600,
          continued_by={"successor_task_id": "fell-c2", "state": "bound", "action_nonce": "n0000002"})
    [work] = standing.fallen_works(tmp_path, now=T0, projects=PROJECTS)
    assert "; Continue admission unconfirmed (successor fell-c2);" in work.line
    _root(tmp_path, "fell", ts=T0 - 1800, ended=T0 - 600,
          continued_by={"successor_task_id": "fell-c2", "state": "admitted", "action_nonce": "n0000002"})
    assert _listed(tmp_path) == [], "a confirmed Continue is a successor"


@pytest.mark.parametrize("pointer", ["superseded_by", "retry_task_id"])
def test_a_supersession_or_a_retry_under_another_id_is_a_successor(tmp_path, pointer):
    _root(tmp_path, "fell", ts=T0 - 1800, ended=T0 - 600, **{pointer: "fell"})  # its own id: attempt metadata
    assert _listed(tmp_path) == ["fell"]
    _root(tmp_path, "fell", ts=T0 - 1800, ended=T0 - 600, **{pointer: "fell-r1"})
    assert _listed(tmp_path) == []


def test_closing_the_project_takes_it_off_and_reopening_brings_it_back(tmp_path):
    _root(tmp_path, "fell", ts=T0 - 1800, ended=T0 - 600)
    closed = {**PROJECTS[0], "closed": {"at": _iso(T0 - 30), "by": "owner", "note": "done"}}
    assert _listed(tmp_path, [closed]) == []
    assert _listed(tmp_path, [{**PROJECTS[0], "lifecycle": "deleting"}]) == []
    assert _listed(tmp_path, PROJECTS) == ["fell"]


@pytest.mark.parametrize("source", sorted(OWNER_STOP_SOURCES))
def test_an_owner_stop_is_excluded_at_entry_even_when_the_offer_would_name_a_budget_first(tmp_path, source):
    """A budget-paused root my human stopped: ``offer_view`` answers ``hard_limit_reached`` before it reads the
    stop, so the stop is read on the row itself."""
    from ouroboros.owner_continue import offer_view

    row = _root(tmp_path, "fell", ts=T0 - 1800, status="cancelled", ended=T0 - 600, reason_code="budget_exhausted",
                cancel_origin={"source": source})
    assert offer_view(row, "fell")["refusal"] == "hard_limit_reached"
    assert _listed(tmp_path) == []
    _root(tmp_path, "fell", ts=T0 - 1800, status="cancelled", ended=T0 - 600, reason_code="budget_exhausted",
          cancel_origin={"source": "owner_restart"})
    [work] = standing.fallen_works(tmp_path, now=T0, projects=PROJECTS)
    assert "; cause budget_exhausted; cancel source owner_restart; Continue refused (hard_limit_reached);" in work.line


@pytest.mark.parametrize("status", ["scheduled", "running", "interrupted", "completed"])
def test_a_held_live_or_finished_root_is_no_candidate(tmp_path, status):
    """A Panic hold keeps its work live, never failed or cancelled; a finished one is done."""
    row = _root(tmp_path, "held", ts=T0 - 1800, status=status, cancel_origin={"source": "panic"})
    assert _listed(tmp_path) == []
    assert not standing.ended_without_success({**row, "cancel_origin": {}})
    _root(tmp_path, "held", ts=T0 - 1800, status="failed")
    assert _listed(tmp_path) == ["held"]


@pytest.mark.parametrize("excluded, fields", [
    ("wrap_up", {"reason_code": "owner_requested_finalization"}),
    ("wrap_up_beneath_a_custody_stamp", {"reason_code": "delegated_custody_unreconciled", "outcome_axes": {
        "execution": {"status": "failed", "reason_code": "owner_requested_finalization"}}}),
    ("another_focus_tool", {"status": "cancelled", "cancel_origin": {"source": "agent_tool"}}),
    ("own_clock_wake", {"metadata": {"initiator": "consciousness", "wake_cause": "own_clock"}}),
    ("focus_wake", {"metadata": {"initiator": "consciousness", "wake_cause": "focus"}}),
    ("a_child", {"parent_task_id": "parent", "delegation_role": "subagent"}),
])
def test_exclusions_at_entry_each_against_the_same_row_without_it(tmp_path, excluded, fields):
    row = _root(tmp_path, "fell", ts=T0 - 1800, ended=T0 - 600, **fields)
    assert not standing.ended_without_success(row), excluded
    if excluded != "a_child":  # a child row is never a probe's root
        assert _listed(tmp_path) == []
    plain = {"status": "failed", "reason_code": "provider_unavailable", "metadata": {}}
    assert standing.ended_without_success({**row, **{key: plain.get(key) for key in fields}})


def test_work_a_wake_started_is_listed_it_is_not_the_wake(tmp_path):
    """``is_consciousness_origin`` holds for everything a wake starts; only the wake itself carries a wake cause."""
    _root(tmp_path, "mine", ts=T0 - 1800, ended=T0 - 600,
          metadata={"initiator": "consciousness", "consciousness_cause": "started", "work_kind": "consciousness"})
    assert _listed(tmp_path) == ["mine"]


def test_a_wake_is_admitted_without_a_project_by_construction(tmp_path):
    """The clock admits ``{Main routing facts, wake_task_metadata}`` (``consciousness._launch``); neither carries a
    project, and the lane's seed takes a project only from its metadata (``worker_chat_lane._seed_task``)."""
    from ouroboros.projects_registry import create_project
    from ouroboros.server_routing_context import main_lane_routing_metadata
    from supervisor.worker_chat_lane import _seed_task
    from tests.test_routing_decision import _decision_ctx

    create_project(tmp_path, "racer", name="Racer")
    facts = main_lane_routing_metadata(_decision_ctx(tmp_path), 1)
    assert [row["project_id"] for row in facts["main_routing_manifest"]["projects"]] == ["racer"]
    metadata = {**facts, **wake.wake_task_metadata("act", "heartbeat")}
    assert "project_id" not in metadata
    assert "project_id" not in _seed_task(1, "[Wake-up · heartbeat]", None, None, metadata)
    assert _seed_task(1, "work", None, None, {"project_id": "racer"})["project_id"] == "racer"


@pytest.mark.parametrize("carrier", ["presence", "presence_binding_authority"])
def test_presence_own_rows_neither_stand_nor_retire(tmp_path, carrier):
    """A speaker turn and work acting for a binding are skipped by their typed columns, in both roles."""
    binding = {"binding_id": "place-1"}
    _root(tmp_path, "turn", ts=T0 - 1800, ended=T0 - 600, metadata={carrier: binding})
    assert _listed(tmp_path) == []
    _root(tmp_path, "fell", ts=T0 - 3600, ended=T0 - 3000)
    _root(tmp_path, "turn", ts=T0 - 1800, status="completed", ended=T0 - 600, metadata={carrier: binding})
    assert _listed(tmp_path) == ["fell"]


def test_the_owners_related_work_stands_and_retires(tmp_path):
    """Related work keeps the binding id with ``presence_authority_recorded = 0``: it is my human's work."""
    relation = {"presence_binding_relation": {"binding_id": "place-1"}}
    _root(tmp_path, "related", ts=T0 - 1800, ended=T0 - 600, metadata=relation)
    assert _listed(tmp_path) == ["related"]
    _root(tmp_path, "fell", ts=T0 - 3600, ended=T0 - 3000)
    _root(tmp_path, "related", ts=T0 - 1800, status="completed", ended=T0 - 600, metadata=relation)
    assert _listed(tmp_path) == []


@pytest.mark.parametrize("fields", [
    {"metadata": {"presence": {"binding_id": ""}}},  # a malformed carrier is still a Presence one
    {"task_contract": {"capability_ceiling": {"tools": []}}},  # a ceiling whose carrier was lost
])
def test_a_presence_row_whose_carrier_lost_its_id_neither_stands_nor_retires(tmp_path, fields):
    """``presence_authority_recorded`` marks such a row beside an empty ``presence_binding_id``
    (``store.typed_columns``): it is Presence's, like a speaker turn."""
    _root(tmp_path, "turn", ts=T0 - 1800, ended=T0 - 600, **fields)
    assert _listed(tmp_path) == []
    _root(tmp_path, "fell", ts=T0 - 3600, ended=T0 - 3000)
    _root(tmp_path, "turn", ts=T0 - 1800, status="completed", ended=T0 - 600, **fields)
    assert _listed(tmp_path) == ["fell"]


def test_an_answer_that_reached_the_chat_reads_done_and_does_not_stand(tmp_path):
    """A final that reached the chat before its finalization was cut is stored ``failed`` and reads Done on every
    owner surface (owner 15.11); an owed one keeps its Failed word and stands."""
    from ouroboros.outcomes import EXECUTION_INFRA_FAILED, terminal_outcome_axes
    from ouroboros.working_checkpoint import REASON_FINALIZATION_INCOMPLETE

    def cut(answer):
        return _root(tmp_path, "cut", ts=T0 - 1800, ended=T0 - 600, reason_code=REASON_FINALIZATION_INCOMPLETE,
                     terminal_origin="model_final", result="Here is the finished audit.",
                     finalization_recovery={"cause": "worker_crash", "answer": answer, "delivery_id": "d1",
                                            "attempt": 0},
                     outcome_axes=terminal_outcome_axes(lifecycle="failed", execution=EXECUTION_INFRA_FAILED,
                                                        reason_code=REASON_FINALIZATION_INCOMPLETE,
                                                        review_trigger="supervisor_terminal"))

    assert not standing.ended_without_success(cut("delivered")) and _listed(tmp_path) == []
    assert standing.ended_without_success(cut("owed"))
    [work] = standing.fallen_works(tmp_path, now=T0, projects=PROJECTS)
    assert work.task_id == "cut" and "- task cut Failed in project Audit," in work.line


def _bind(root, **projects_by_task):
    """``state/project_task_bindings.json`` as "Turn into project" leaves it: the binding names a work's project."""
    (root / "state").mkdir(exist_ok=True)
    rows = {task_id: {"task_id": task_id, "project_id": project_id, "project_chat_id": 0, "bound_at": _iso(T0 - 900)}
            for task_id, project_id in projects_by_task.items()}
    (root / "state" / "project_task_bindings.json").write_text(json.dumps({"bindings": rows}), encoding="utf-8")


@pytest.mark.parametrize("copy", ["", "other"])
def test_a_root_counts_in_the_project_its_binding_names_never_in_the_copy_it_started_with(tmp_path, copy):
    """A work turned into a project while it ran keeps, in its own terminal row, the project copy it started with
    (Main's empty one, or another room's); the binding is its project (owner B4=A): there it stands or retires."""
    projects = [*PROJECTS, {"id": "other", "name": "Other", "lifecycle": "active"}]
    _root(tmp_path, "earlier", ts=T0 - 9000, project="other", ended=T0 - 8000)
    _root(tmp_path, "converted", ts=T0 - 1800, project=copy, ended=T0 - 600)
    assert _listed(tmp_path, projects) == (["converted"] if copy else ["earlier"]), "the copy alone decides"
    _bind(tmp_path, converted="p1")
    works = standing.fallen_works(tmp_path, now=T0, projects=projects)
    assert {work.task_id: work.project_id for work in works} == {"converted": "p1", "earlier": "other"}
    assert "- task converted Failed in project Audit," in works[0].line


def test_a_root_bound_to_the_project_later_retires_its_earlier_failure(tmp_path):
    _root(tmp_path, "fell", ts=T0 - 3600, ended=T0 - 3000)
    _root(tmp_path, "moved", ts=T0 - 600, project="", status="running")
    assert _listed(tmp_path) == ["fell"]
    _bind(tmp_path, moved="p1")
    assert _listed(tmp_path) == []
    _bind(tmp_path, moved="elsewhere", unrelated="p1")  # bound to no open project, and a binding with no row
    assert _listed(tmp_path) == ["fell"]


@pytest.mark.parametrize("damage", ["malformed", "unreadable", "null_row", "bad_project_id"])
def test_an_unreadable_binding_store_is_the_sections_gap_never_the_copy_a_root_started_with(tmp_path, damage):
    """The binding names a converted root's project, so the store is read strictly: one that cannot be read, or a row
    the strict per-task reader refuses (not an object, or no valid ``project_id``), is the section's named gap, as an
    unreadable registry is, never "no bindings" with the root back in its old project. A readable store places it by
    its binding, and an absent one (nothing converted yet) by its copy, with no gap."""
    projects = [*PROJECTS, {"id": "other", "name": "Other", "lifecycle": "active"}]
    _registry(tmp_path, *projects)
    _root(tmp_path, "earlier", ts=T0 - 9000, project="other", ended=T0 - 8000)
    _root(tmp_path, "converted", ts=T0 - 1800, project="other", ended=T0 - 600)
    _bind(tmp_path, converted="p1")
    placed = _observe(tmp_path)
    assert [line.split(",")[0] for line in placed.standing[0].lines] == [
        "- task converted Failed in project Audit", "- task earlier Failed in project Other"]
    assert not [gap for gap in placed.gaps if gap.startswith("fallen_work")]
    store = tmp_path / "state" / "project_task_bindings.json"
    rows = {"null_row": None, "bad_project_id": {"task_id": "converted", "project_id": "Not a project"}}
    if damage == "unreadable":  # a directory where the store is: opening it fails with the platform's OSError
        store.unlink()
        store.mkdir()
    else:
        body = "{ not json" if damage == "malformed" else json.dumps({"bindings": {"converted": rows[damage]}})
        store.write_text(body, encoding="utf-8")
    unread = _observe(tmp_path)
    [gap] = [gap for gap in unread.gaps if gap.startswith("fallen_work unreadable: ")]
    assert unread.standing == () and (damage == "unreadable" or gap == "fallen_work unreadable: ValueError")
    store.rmdir() if damage == "unreadable" else store.unlink()
    absent = _observe(tmp_path)
    assert [line.split(",")[0] for line in absent.standing[0].lines] == ["- task converted Failed in project Other"]
    assert not [gap for gap in absent.gaps if gap.startswith("fallen_work")]


def test_the_roots_bound_to_an_open_project_are_read_by_primary_key(tmp_path, monkeypatch):
    """A wake reads the bound roots by their ids, so its cost does not grow with the roots history holds: the plan
    searches the primary key, never the parent index every root shares (a null parent) for the root predicate."""
    import contextlib

    from ouroboros import task_store

    for index in range(3):
        _root(tmp_path, f"old{index}", ts=T0 - 9000 + index, project="", status="completed", ended=T0 - 8000)
    _root(tmp_path, "moved", ts=T0 - 600, project="", ended=T0 - 300)
    _bind(tmp_path, moved="p1")
    real, statements = task_store.open_store, []

    @contextlib.contextmanager
    def traced(*args, **kwargs):  # the statements as SQLite runs them, bound values in place
        with real(*args, **kwargs) as txn:
            txn.conn.set_trace_callback(statements.append)
            yield txn
            txn.conn.set_trace_callback(None)

    monkeypatch.setattr(task_store, "open_store", traced)
    assert _listed(tmp_path) == ["moved"]
    [bound] = [sql for sql in statements if "json_each" in sql and "presence_authority_recorded" in sql]
    with real(tmp_path, write=False) as txn:
        plan = " | ".join(str(row[-1]) for row in txn.conn.execute("EXPLAIN QUERY PLAN " + bound))
    assert "tasks_parent" not in plan and "(task_id=?)" in plan, plan


def test_bodies_are_read_only_for_candidates_and_the_line_stays_compact(tmp_path, monkeypatch):
    from ouroboros import task_store

    projects = [{"id": f"p{index}", "name": f"Project {index}", "lifecycle": "active"} for index in range(6)]
    _root(tmp_path, "done0", ts=T0 - 900, status="completed", project="p0", ended=T0 - 800)
    _root(tmp_path, "older0", ts=T0 - 9000, project="p0", ended=T0 - 8000)
    _root(tmp_path, "fell1", ts=T0 - 1800, project="p1", ended=T0 - 600, description="x" * 400,
          wake_summary="w" * 900)
    _root(tmp_path, "stop2", ts=T0 - 1700, status="cancelled", project="p2", cancel_origin={"source": "http_single"})
    _root(tmp_path, "run3", ts=T0 - 1600, status="running", project="p3")
    _root(tmp_path, "fell5", ts=T0 - 1500, project="p5", ended=T0 - 300)
    fetched, real = [], task_store._fetch

    def spy(txn, where="", params=()):
        fetched.append(json.loads(params[0]))
        return real(txn, where, params)

    monkeypatch.setattr(task_store, "_fetch", spy)
    works = standing.fallen_works(tmp_path, now=T0, projects=projects)
    assert fetched == [["fell1", "fell5", "stop2"]]
    assert [work.task_id for work in works] == ["fell5", "fell1"]  # newest-started first
    line = works[1].line
    assert "w" * 20 not in line and "chars omitted]" in line and len(line) < 300


def test_a_root_is_named_by_its_title_then_its_description_then_its_request(tmp_path):
    """A root titled "Friday report" is named so, never by the instructions it was given."""
    request = "Collect the week's numbers and mail them to the board"
    for fields, named in (({"title": "Friday report"}, "Friday report"), ({}, "do fell"), ({"description": ""}, request)):
        _root(tmp_path, "fell", ts=T0 - 1800, ended=T0 - 600, text=request, **fields)
        [work] = standing.fallen_works(tmp_path, now=T0, projects=PROJECTS)
        assert f", ended 10 min ago: {named}; cause provider_unavailable;" in work.line, work.line


def test_no_open_project_reads_nothing_and_no_section_is_printed(tmp_path, monkeypatch):
    from ouroboros import task_store

    _root(tmp_path, "fell", ts=T0 - 1800, ended=T0 - 600)
    monkeypatch.setattr(task_store, "open_store", lambda *_a, **_k: pytest.fail("no open project, no probe"))
    assert standing.fallen_works(tmp_path, now=T0, projects=[]) == []
    assert standing.fallen_works(tmp_path, now=T0, projects=[{**PROJECTS[0], "lifecycle": "deleting"}]) == []
    monkeypatch.undo()
    observation = _observe(tmp_path)  # no registry on this install: an empty one
    assert observation.standing == () and "fallen_work" not in observation.counts()
    assert "Works that ended without success" not in observation.full_text()


def test_an_unreadable_source_is_a_named_gap_never_a_missing_wake(tmp_path, monkeypatch):
    from ouroboros import projects_registry, task_store
    from ouroboros.store import StoreBusy

    _registry(tmp_path, *PROJECTS)
    _root(tmp_path, "fell", ts=T0 - 1800, ended=T0 - 600)
    monkeypatch.setattr(projects_registry, "list_reserved_projects", lambda _root: (_ for _ in ()).throw(OSError("x")))
    observation = _observe(tmp_path)
    assert observation.standing == ()
    assert {"fallen_work unreadable: LookupError", "project names unreadable: OSError"} <= set(observation.gaps)
    monkeypatch.undo()

    def busy(*_args, **_kwargs):
        raise StoreBusy("busy")

    monkeypatch.setattr(task_store, "_fetch", busy)
    observation = _observe(tmp_path)
    assert "fallen_work unreadable: StoreBusy" in observation.gaps and observation.standing == ()


def test_a_registry_lost_beside_its_commit_witness_is_the_sections_gap_never_no_projects(tmp_path):
    """An empty registry means no projects only where the registry proves it (``projects_registry.readable_rooms``'
    rule): one lost beside its commit witness, or one whose rows none is readable, is the section's named gap; one
    committed empty, or never committed (a fresh install), stays quiet."""
    _registry(tmp_path, *PROJECTS)
    witness = tmp_path / "state" / "projects.json.committed"
    witness.touch()  # what every registry save leaves beside it
    _root(tmp_path, "fell", ts=T0 - 1800, ended=T0 - 600)
    assert _observe(tmp_path).counts()["fallen_work"] == 1
    (tmp_path / "state" / "projects.json").unlink()
    lost = _observe(tmp_path)
    assert lost.standing == () and "fallen_work unreadable: FileNotFoundError" in lost.gaps
    _registry(tmp_path, {"id": "Not a project", "name": "Unreadable"})
    unreadable = _observe(tmp_path)
    assert unreadable.standing == () and "fallen_work unreadable: ValueError" in unreadable.gaps
    _registry(tmp_path)
    empty = _observe(tmp_path)
    witness.unlink()
    (tmp_path / "state" / "projects.json").unlink()
    fresh = _observe(tmp_path)
    for observation in (empty, fresh):
        assert observation.standing == () and not [gap for gap in observation.gaps if gap.startswith("fallen_work")]


def test_an_empty_lenient_read_lists_the_projects_its_validating_read_returns(tmp_path):
    """The rows the validating read (``list_reserved_projects(require_readable=True)``) returns are the section's: a
    registry repaired between the wake's lenient read and that read lists its project's work, never a quiet wake;
    a lenient read that names the project keeps working from its own rows."""
    _registry(tmp_path, *PROJECTS)
    _root(tmp_path, "fell", ts=T0 - 1800, ended=T0 - 600)
    for lenient in ([], PROJECTS):
        gaps = set()
        sections = standing.standing_sections(standing.StandingFacts(root=tmp_path, now=T0, registry=lenient), gaps)
        assert [section.kind for section in sections] == [standing.FALLEN_KIND] and not gaps, lenient
        assert len(sections[0].lines) == 1 and sections[0].lines[0].startswith("- task fell Failed in project Audit")


def test_every_form_of_the_wake_keeps_the_section_and_the_bare_pointer_folds_it(tmp_path):
    """The projection (the form a wake takes when its observation does not fit) keeps the lines; only the bare
    pointer, the physical window's last form, folds them to a count with the kind of their rows at the source."""
    from ouroboros.artifacts import read_actor_source_bytes

    _registry(tmp_path, *PROJECTS)
    _root(tmp_path, "fell", ts=T0 - 1800, ended=T0 - 600)
    observation = _observe(tmp_path)
    task = {"id": "wake-standing", "metadata": {}}
    wake.bind_wake_observation(tmp_path, task, observation, lambda events: events)
    facts = task["metadata"][wake.WAKE_OBSERVATION_KEY]
    [line] = observation.standing[0].lines
    heading = "Works that ended without success; nothing has started in their project since (1):"
    assert f"\n{heading}\n{line}" in facts["projection_text"]
    assert line not in facts["pointer_text"]
    assert facts["pointer_text"].endswith(
        "\nWorks that ended without success; nothing has started in their project since (1): rows of kind "
        "fallen_work at the source")
    assert "then the standing sections, one row per line under its kind" in facts["pointer_text"]
    assert facts["composition"]["fallen_work"] == 1
    source = facts["source"]
    rows = [json.loads(row) for row in read_actor_source_bytes(tmp_path, source["task_id"], source).splitlines()]
    assert rows[-1] == {"kind": "fallen_work", "line": line}


# --- the own clock (consciousness_clock.own_clock_lines as a standing section) ----------------------------------------


NAMED = {"at": "2027-01-15T11:30:00+03:00", "stored_at": "2027-01-15T07:00:00Z", "standing_appointment": None}
ZONE = {"zone": "Europe/Moscow", "at": "2027-01-15T07:20:00Z", "message": {"chat_id": 1, "client_message_id": "c1"}}


def _clock(**state):
    """The own clock's facts as ``consciousness._launch`` carries them: the state keys it read, the owner's bounds."""
    from ouroboros.consciousness import APPOINTMENT_STATE_KEY

    return {APPOINTMENT_STATE_KEY: T0 - 30, "floor": 900, "ceiling": 14400, **state}


def _state(**state):
    from ouroboros.consciousness_clock import OWNER_ZONE_STATE_KEY, WAKE_TARGET_STATE_KEY

    return {{"named": WAKE_TARGET_STATE_KEY, "zone": OWNER_ZONE_STATE_KEY}[key]: value for key, value in state.items()}


@pytest.mark.parametrize("stored", [{"named": NAMED}, {"zone": ZONE}, {"named": NAMED, "zone": ZONE}])
def test_the_own_clock_stands_while_a_named_time_or_a_reported_zone_does(tmp_path, stored):
    from ouroboros.consciousness_clock import own_clock_lines

    clock = _clock(**_state(**stored))
    observation = wake.observe_wake(tmp_path, boundary=None, since=T0 - 3600, now=T0, reason="heartbeat", clock=clock)
    [section] = observation.standing
    assert section.kind == "own_clock" and section.title == "Own clock"
    assert section.lines == tuple(own_clock_lines(clock, now=T0, floor=900, ceiling=14400))  # every line, as said
    assert observation.counts()["own_clock"] == 4 and observation.composition().endswith("4 own clock facts")
    assert "\nOwn clock (4):\n" + "\n".join(section.lines) in observation.full_text()
    assert ("named next wake-up time: none" in section.lines[0]) is ("named" not in stored)
    assert section.lines[-1].endswith(": unknown") is ("zone" not in stored)


@pytest.mark.parametrize("clock", [
    None,  # a preparation that carried no clock (an older admission)
    _clock(),  # only this wake's appointment and the owner's bounds
    _clock(**_state(named={"at": "tomorrow at nine"}, zone={"zone": "Not/A_Zone", "at": "2027-01-15T07:20:00Z"})),
])
def test_the_appointment_and_the_bounds_alone_make_no_own_clock_section(tmp_path, clock):
    observation = wake.observe_wake(tmp_path, boundary=None, since=T0 - 3600, now=T0, reason="heartbeat", clock=clock)
    assert observation.standing == () and "own_clock" not in observation.counts()
    assert not [gap for gap in observation.gaps if gap.startswith("own_clock")]
    assert "Own clock" not in observation.full_text()


def test_the_own_clock_follows_the_fallen_works_in_every_form_and_the_bare_pointer_folds_it(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes

    _registry(tmp_path, *PROJECTS)
    _root(tmp_path, "fell", ts=T0 - 1800, ended=T0 - 600)
    observation = wake.observe_wake(tmp_path, boundary=None, since=T0 - 3600, now=T0, reason="heartbeat",
                                    clock=_clock(**_state(named=NAMED, zone=ZONE)))
    assert [section.kind for section in observation.standing] == ["fallen_work", "own_clock"]
    task = {"id": "wake-clock", "metadata": {}}
    wake.bind_wake_observation(tmp_path, task, observation, lambda events: events)
    facts = task["metadata"][wake.WAKE_OBSERVATION_KEY]
    lines = observation.standing[1].lines
    block = "\nOwn clock (4):\n" + "\n".join(lines)
    assert observation.full_text().endswith(block) and facts["projection_text"].endswith(block)
    assert facts["pointer_text"].endswith("\nOwn clock (4): rows of kind own_clock at the source")
    assert not any(line in facts["pointer_text"] for line in lines)
    assert facts["composition"]["own_clock"] == 4
    source = facts["source"]
    rows = [json.loads(row) for row in read_actor_source_bytes(tmp_path, source["task_id"], source).splitlines()]
    assert rows[-4:] == [{"kind": "own_clock", "line": line} for line in lines]


def test_a_prepared_wake_message_shows_its_own_clock_and_leaves_punctuality_to_the_reason_line(tmp_path):
    """``prepare_wake`` over the clock's facts renders the real prompt: the section stands in the wake's text and in
    its projection; whether the wake came on time is said once, in the reason line."""
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    facts = {"boundary": None, "since": T0 - 3600, "now": T0, "reason": "heartbeat", "level": "act",
             "repo_dir": str(repo), "last_wake_at": T0 - 3600, "last_wake_task_id": "", "disabled_tools": [],
             "window": {"settled_usd": 0.0, "unknown_unmetered": 0, "limit_usd": 5.0, "spend_by_cause": None},
             "counted_roots": [], "max_tasks": 2, "interval": 3300, "appointment_at": T0 - 30}
    for stored, shown in (({"named": NAMED}, True), ({"zone": ZONE}, True), ({}, False)):
        task = {"id": f"wake-{len(stored)}{'z' if 'zone' in stored else ''}", "chat_id": 1, "metadata": {},
                "text": "[Wake-up · heartbeat]", "_wake_prepare": {**facts, "clock": _clock(**_state(**stored))}}
        wake.prepare_wake(task, tmp_path)
        projection = task["metadata"][wake.WAKE_OBSERVATION_KEY]["projection_text"]
        for text in (task["text"], projection):
            assert ("\nOwn clock (4):\n- named next wake-up time: " in text) is shown
            assert text.count("- reason: heartbeat (on time)") == 1 and "due " not in text
        if shown:
            assert "- this wake's appointment: " in task["text"] and "the reason line)" in task["text"]
