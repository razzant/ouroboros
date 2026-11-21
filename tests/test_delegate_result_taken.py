"""Ouroboros tells the engine once that it has taken a finished delegated result (owner Q2=A).

A delegated Git run whose patch Ouroboros captured into its task record, and whose line can no longer be
continued, gets exactly one typed ``taken`` decision; a continuable, uncaptured, ``files``, in-place, plan or
review run gets none; an engine whose decision vocabulary names no ``taken`` keeps the task waiting without a
decision record; an unanswered decision is repeated under the same key. Results whose task settled before the pass
could ask get the same decision from the one-time walk, which leaves what it cannot decide to the pass's waiting set
and whose counts reach the owner's chat once. The engine is an in-memory MOCK of Claudexor's C7 route shape
(``POST /v2/runs/:id/decision {"action": "taken", "reason": "captured_result"}``), its thread-retirement catalog
flag and its managed release route, never the real one.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import delegate_custody, delegate_custody_current, maintenance, maintenance_jobs, obligations, owner_keep
from ouroboros import delegate_result_taken as taken
from ouroboros import maintenance_retirement as retirement
from ouroboros import review_thread_retention as rt
from ouroboros.delegate_custody import RunCustody
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.maintenance_retirement import JOB
from ouroboros.task_store import patch_document, read_document
from tests import _dialogue as dialogue_test
from tests._store_shared import seed_result
from tests.test_store import document

TASK = "task-taken-1"
RELEASE = {"id": "post:resources.release", "method": "POST", "path": "/v2/resources/:id/release",
           "requestSchema": "ControlResourceReleaseRequest"}


class Engine:
    """The decision route and the decision vocabulary, recorded; the thread-retirement flag and the managed
    release route the step's other duties ask on the pass after the walk."""

    def __init__(self, *, supported=True, version="5.1.0"):
        self.supported, self.engine_version, self.store_identity = supported, version, "store-1"
        self.decisions, self.fail, self.released = [], [], []
        self.capability_reads = 0
        self.answer = {"accepted": True, "status": "taken", "message": "acknowledged"}

    def run_result_taken_supported(self):
        self.capability_reads += 1
        return self.supported

    def take_run_result(self, run_id, *, idempotency_key):
        self.decisions.append((run_id, idempotency_key))
        if self.fail:
            raise self.fail.pop(0)
        return self.answer

    def thread_retirement_supported(self):
        return True

    def operations(self, **_kw):
        return [dict(RELEASE)]

    def release_resource(self, resource_id, request):
        self.released.append(resource_id)
        return {"storeId": self.store_identity, "resourceId": resource_id, "sha256": request["sha256"],
                "state": "released", "releasedAt": "2026-11-08T00:00:00Z"}

    def close(self):
        pass


def _git_run(run_id="run-1", **facts):
    """A finished Git copy's run: settled, its patch captured into the task record."""
    base = dict(run_id=run_id, task_id=TASK, root_task_id=TASK, settled=True, terminal_state="succeeded",
                snapshot_id=f"snap-{run_id}", execution_root=f"/copies/{run_id}", target_root="/owner/repo",
                access="workspace_write", mode="agent", patch_captured=True)
    base.update(facts)
    return RunCustody(**base)


def _pass(root, engine, *, stops=()):
    stops = list(stops)
    report = {"holds": {}, "observed": []}
    run = SimpleNamespace(root=root, live=lambda _task: False, census=lambda: set(), report=report, stopped=False,
                          engine=lambda: engine, stop=lambda: stops.pop(0) if stops else False)
    run.hold = lambda reason: report["holds"].__setitem__(reason, report["holds"].get(reason, 0) + 1)
    return run


@pytest.fixture
def world(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    seed_result(root, TASK, document(TASK, "completed"))
    custody = {"runs": {"run-1": _git_run()}}
    proof = {"why": ""}
    monkeypatch.setattr(delegate_custody_current, "runs", lambda _root, _where="", params=(): {
        rid: run for rid, run in custody["runs"].items() if run.task_id == params[0]})
    asked = []

    def tree_hold(_root, task_id, _document, *, live, census):
        asked.append(task_id)
        return (proof["why"], {}) if proof["why"] else ("", {
            rid: run for rid, run in custody["runs"].items() if run.task_id == task_id})

    monkeypatch.setattr(rt, "tree_hold", tree_hold)
    engine = Engine()
    return SimpleNamespace(root=root, custody=custody, proof=proof, engine=engine, asked=asked,
                           records=lambda: (read_document(root, TASK) or {}).get(taken.RECORDS, {}),
                           tell=lambda engine=engine, **kw: taken.tell_engine(_pass(root, engine, **kw), TASK))


def test_a_captured_result_of_an_ended_tree_is_taken_exactly_once(world):
    assert world.tell() == rt.DONE
    key = hashlib.sha256(b"taken:run-1").hexdigest()
    assert world.engine.decisions == [("run-1", key)]
    assert world.records()["run-1"]["state"] == "taken"
    assert world.tell() == rt.DONE and world.engine.decisions == [("run-1", key)], "a repeat sends nothing"
    assert world.asked == [TASK], "a decided task is not even asked for its tree again"


@pytest.mark.parametrize("why", ["continuable", "task_not_proven_over", "custody_open", "census_unknown"])
def test_a_line_that_may_still_continue_waits_and_sends_nothing(world, why):
    world.proof["why"] = why
    assert world.tell() == rt.WAIT
    assert world.engine.decisions == [] and world.records() == {}
    world.proof["why"] = ""  # the line ended since: the next pass decides
    assert world.tell() == rt.DONE and [rid for rid, _key in world.engine.decisions] == ["run-1"]


@pytest.mark.parametrize("facts", [
    {"patch_captured": False},  # uncaptured, nothing left to try: no copy Ouroboros holds whole
    {"resource_ref": {"workspace_kind": "directory", "strategy": "copy"}},  # a files result: the engine applies it
    {"resource_ref": {"workspace_kind": "git", "strategy": "in_place"}},  # edits already in place
    {"mode": "plan"},
    {"source": "review_substrate_plan"},  # a review seat: its panel's and its thread's
    {"snapshot_id": "", "execution_root": ""},  # a read-only run: no patch
    {"superseded_by": "run-2"},  # its successor's capture is the one
    {"patch_apply_pending": True},  # an apply of unknown outcome
    {"settled": True, "terminal_state": ""},  # closed absent: the engine holds nothing
])
def test_runs_that_are_not_a_held_git_result_get_no_decision(world, facts):
    world.custody["runs"]["run-1"] = _git_run(**facts)
    assert world.tell() == rt.DONE
    assert world.engine.decisions == [] and world.records() == {} and world.asked == []


def test_a_collection_still_owed_waits_until_it_lands(world, monkeypatch):
    from ouroboros import delegate_capture_attempt

    pending = {"value": True}
    monkeypatch.setattr(delegate_custody_current, "collection_pending", lambda _entry, root=None: pending["value"])
    monkeypatch.setattr(delegate_capture_attempt, "collection_due", lambda _entry, **_kw: True)
    assert world.tell() == rt.WAIT and world.engine.decisions == []
    pending["value"] = False
    assert world.tell() == rt.DONE and len(world.engine.decisions) == 1
    # A collection that can heal only with new input is no reason to wait: nothing is ever sent for it.
    world.custody["runs"]["run-2"] = _git_run("run-2")
    pending["value"] = True
    monkeypatch.setattr(delegate_capture_attempt, "collection_due", lambda _entry, **_kw: False)
    assert world.tell() == rt.DONE and len(world.engine.decisions) == 1


def test_an_engine_without_the_taken_action_waits_until_a_later_pass_supports_it(world):
    old = Engine(supported=False, version="5.0.1")
    run = _pass(world.root, old)
    assert taken.tell_engine(run, TASK) == rt.WAIT
    assert old.decisions == [] and world.records() == {}
    assert run.report["holds"] == {"result_taken_unsupported": 1}
    assert taken.tell_engine(run, TASK) == rt.WAIT and old.capability_reads == 1
    old.supported = True  # an upgrade is read by the next pass, which sends the one keyed decision
    assert world.tell(engine=old) == rt.DONE and old.capability_reads == 2
    key = hashlib.sha256(b"taken:run-1").hexdigest()
    assert old.decisions == [("run-1", key)] and world.records()["run-1"]["state"] == "taken"
    assert world.tell(engine=old) == rt.DONE and old.decisions == [("run-1", key)]


@pytest.mark.parametrize("status,code", [
    (0, "daemon_unreachable"), (409, "delivery_in_progress"), (409, "idempotency_conflict"),
    (403, "forbidden"), (408, "timeout"), (429, "rate_limited"), (500, "delivery_failed"), (200, "unaccepted")])
def test_an_unanswered_decision_waits_and_the_next_pass_repeats_the_same_key(world, status, code):
    if status == 200:
        world.engine.answer = {"accepted": False, "status": code}
    else:
        world.engine.fail.append(ClaudexorUnavailable(code, "unanswered", status_code=status))
    assert world.tell() == rt.WAIT and world.records() == {}
    world.engine.answer = {"accepted": True, "status": "taken"}
    assert world.tell() == rt.DONE and world.records()["run-1"]["state"] == "taken"
    first, second = world.engine.decisions
    assert first == second, "the same keyed decision, which the engine answers once"


def test_an_interrupted_delivery_resends_once_under_the_same_derived_key(world):
    interrupted = lambda: ClaudexorUnavailable("delivery_interrupted", "restart", status_code=409)
    world.engine.fail.extend([interrupted(), interrupted()])
    assert world.tell() == rt.WAIT and world.records() == {}
    key = hashlib.sha256(b"taken:run-1").hexdigest()
    retry_key = hashlib.sha256(f"{key}:interrupted".encode()).hexdigest()
    assert world.engine.decisions == [("run-1", key), ("run-1", retry_key)], "one resend, no new-key loop"
    world.engine.fail.append(interrupted())  # original receipt remains interrupted; derived delivery now succeeds
    assert world.tell() == rt.DONE and world.records()["run-1"]["state"] == "taken"
    assert world.engine.decisions == [("run-1", key), ("run-1", retry_key)] * 2
    assert world.tell() == rt.DONE and len(world.engine.decisions) == 4


@pytest.mark.parametrize("status,code,state", [
    (409, "run_taken_continuable", "refused"), (409, "run_taken_not_patch", "refused"),
    (400, "invalid_request", "refused"), (404, "not_found", "absent"), (404, "project_not_registered", "absent")])
def test_a_typed_answer_is_recorded_once_and_never_repeated(world, status, code, state):
    world.engine.fail.append(ClaudexorUnavailable(code, code, status_code=status))
    assert world.tell() == rt.DONE
    record = world.records()["run-1"]
    assert record["state"] == state and (state == "absent" or record["code"] == code)
    assert world.tell() == rt.DONE and len(world.engine.decisions) == 1


def test_the_owner_keep_holds_the_engine_copy_too(world, monkeypatch):
    kept = {"value": True}
    monkeypatch.setattr(owner_keep, "is_kept", lambda _root, _task: kept["value"])
    run = _pass(world.root, world.engine)
    assert taken.tell_engine(run, TASK) == rt.WAIT and run.report["holds"] == {"result_owner_keep": 1}
    assert world.engine.decisions == [] and world.asked == [] and world.records() == {}
    kept["value"] = False
    released = _pass(world.root, world.engine)
    assert taken.tell_engine(released, TASK) == rt.DONE and released.report["holds"] == {}
    assert world.records()["run-1"]["state"] == "taken" and len(world.engine.decisions) == 1
    assert world.tell() == rt.DONE and len(world.engine.decisions) == 1


def test_a_stop_between_decisions_leaves_the_rest_waiting(world):
    world.custody["runs"]["run-2"] = _git_run("run-2")
    assert world.tell(stops=[True]) == rt.WAIT
    assert [rid for rid, _key in world.engine.decisions] == ["run-1"] and list(world.records()) == ["run-1"]
    assert world.tell() == rt.DONE and [rid for rid, _key in world.engine.decisions] == ["run-1", "run-2"]


# -- the wire: Claudexor's C7 shape on the existing decision route ------------------------------------------

def test_the_gateway_sends_the_taken_action_and_reads_the_decision_vocabulary():
    from ouroboros.gateways.claudexor import ClaudexorGateway, DaemonEndpoint

    seen, actions = [], {"value": ["discard", "taken"]}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        seen.append((request.method, request.url.path, body, request.headers.get("Idempotency-Key", "")))
        if request.url.path == "/v2/agent-capabilities":
            return httpx.Response(200, json={"harnesses": [], "runDecisionActions": actions["value"]})
        return httpx.Response(200, json={"accepted": True, "status": "taken", "message": "ok"})

    gateway = ClaudexorGateway(DaemonEndpoint(host="127.0.0.1", port=1, token="token"))
    gateway._client.close()
    gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(handler),
                                   headers={"Authorization": "Bearer token"})
    try:
        assert gateway.run_result_taken_supported() is True
        assert gateway.take_run_result("run-1", idempotency_key="key-1")["status"] == "taken"
        actions["value"] = ["discard"]  # an engine before Claudexor C7
        assert gateway.run_result_taken_supported() is False
    finally:
        gateway.close()
    assert ("POST", "/v2/runs/run-1/decision", {"action": "taken", "reason": "captured_result"}, "key-1") in seen


# -- the real pass: the runtime-retirement step over real task rows and custody --------------------------------

R = "taken-root"


@pytest.fixture
def real(tmp_path):
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import utc_now_iso
    from tests._shared import write_queue_document

    root = tmp_path / "data"
    root.mkdir()
    write_queue_document(root / "state" / "queue_snapshot.json", {"ts": utc_now_iso(), "pending": [], "running": []})
    write_task_result(root, R, "running", root_task_id=R, parent_task_id="", delegation_role="root")
    engine = Engine()
    assert rt.retire_finished_review_threads(root, engine=engine) == {"activated": True}
    invocation = "inv-git-1"
    assert delegate_custody.record_start_requested(
        root, run_id="", task_id=R, invocation_id=invocation, idempotency_key=invocation, operation_id="",
        max_seconds=0, request={"prompt": "change it"}, project_id="", project_owned=False, route="codex",
        root_task_id=R, parent_task_id="")
    entry = RunCustody(run_id="run-git-1", task_id=R, invocation_id=invocation, idempotency_key=invocation,
                       route_id="codex", root_task_id=R, ledger_root=str(root), snapshot_id="snap-1",
                       execution_root=str(tmp_path / "copy"), target_root=str(tmp_path / "repo"))
    assert delegate_custody.record_started(root, entry, shape={"access": "workspace_write", "mode": "agent"})
    assert delegate_custody.emit(root, delegate_custody.SETTLED, {"run_id": entry.run_id, "task_id": R,
                                                                  "state": "succeeded"})
    assert delegate_custody.record_patch_captured(root, entry, status="ready", sha256="0" * 64,
                                                  capture_dir=str(tmp_path / "capture"))
    delegate_custody._CUSTODY.clear()

    def end(status, **fields):  # the task's terminal write, with the facts its Continue offer reads
        patch_document(root, R, lambda current: {**{k: v for k, v in current.items() if k != "cancel_origin"},
                                                 "status": status, **fields})

    def run():
        write_queue_document(root / "state" / "queue_snapshot.json", {"ts": utc_now_iso(), "pending": [], "running": []})
        return rt.retire_finished_review_threads(root, engine=engine)  # the real live probe and queue census

    return SimpleNamespace(root=root, engine=engine, end=end, run=run,
                           record=lambda: ((read_document(root, R) or {}).get(taken.RECORDS) or {}).get("run-git-1"))


def test_real_a_finished_root_result_is_taken_once_by_the_runtime_retirement_pass(real):
    real.end("completed")
    real.engine.supported = False
    report = real.run()
    assert report["waiting"] == 1 and report["holds"] == {"result_taken_unsupported": 1}
    assert real.engine.decisions == [] and real.record() is None
    real.engine.supported = True  # no new terminal write: the waiting list must retain the duty
    report = real.run()
    assert [rid for rid, _key in real.engine.decisions] == ["run-git-1"] and real.record()["state"] == "taken"
    assert {"task_id": R, "run_id": "run-git-1", "taken": "taken"} in report["observed"]
    assert report["waiting"] == 0
    real.run()  # its own record was a terminal write: the next pass looks again and sends nothing
    assert len(real.engine.decisions) == 1


def test_real_a_root_the_owner_may_still_continue_holds_its_result_until_it_cannot(real):
    real.end("cancelled", cancel_origin={"source": "", "reason": "server_shutdown"})  # a technical break: Continue
    report = real.run()
    assert real.engine.decisions == [] and real.record() is None and report["waiting"] == 1
    assert report["holds"].get("result_continuable") == 1, report
    real.end("completed")  # the line can no longer be continued
    real.run()
    assert [rid for rid, _key in real.engine.decisions] == ["run-git-1"]


# -- the one-time walk: results whose task settled before the pass could ask -------------------------------------

def _settle(root, scratch, task, status="completed", runs=1, parent="", thread="", **fields):
    """A task with finished Git runs whose patches are captured (``run-<task>``, then ``run-<task>-2``...): copies
    the engine still holds as actionable; a subagent of ``parent``'s tree when one is named; ``thread`` adds a plan
    seat's reviewer thread and the settled read-only run that started in it. Settled before the step's first pass it
    is never one of the pass's own tasks; settled after it, its terminal write makes it one; left ``running``, it is
    neither yet."""
    from ouroboros import plan_record
    from ouroboros.task_results import write_task_result

    tree = parent or task
    write_task_result(root, task, "running", root_task_id=tree, parent_task_id=parent,
                      delegation_role="subagent" if parent else "root")
    shapes = [(f"run-{task}" + (f"-{number}" if number > 1 else ""), {"prompt": "change it"}, True)
              for number in range(1, runs + 1)]
    shapes += [(f"seat-{task}", {"prompt": "review", "_thread_id": thread}, False)] if thread else []
    for run_id, request, copy in shapes:
        invocation = f"inv-{run_id}"
        assert delegate_custody.record_start_requested(
            root, run_id="", task_id=task, invocation_id=invocation, idempotency_key=invocation, operation_id="",
            max_seconds=0, request=request, project_id="", project_owned=False, route="codex", root_task_id=tree,
            parent_task_id=parent)
        place = {"snapshot_id": f"snap-{run_id}", "execution_root": str(scratch / run_id / "copy"),
                 "target_root": str(scratch / "repo")} if copy else {}
        entry = RunCustody(run_id=run_id, task_id=task, invocation_id=invocation, idempotency_key=invocation,
                           route_id="codex", root_task_id=tree, ledger_root=str(root), **place)
        shape = {"access": "workspace_write", "mode": "agent"} if copy else {"access": "readonly", "mode": "ask"}
        assert delegate_custody.record_started(root, entry, shape=shape)
        assert delegate_custody.emit(root, delegate_custody.SETTLED, {"run_id": run_id, "task_id": task,
                                                                      "state": "succeeded"})
        if copy:
            assert delegate_custody.record_patch_captured(root, entry, status="ready", sha256="0" * 64,
                                                          capture_dir=str(scratch / run_id / "capture"))
    delegate_custody._CUSTODY.clear()
    if thread:
        record = plan_record.empty_record()
        record["asks"].append({"ask_id": "a1", "record_id": f"rec-{task}", "ts": "2026-11-01T00:00:00+00:00",
                               "revision": 1, "message": "", "participants": [
                                   {"seat_id": "seat-1", "status": "answered", "review_thread_id": thread}]})
        fields["plan_record"] = record
    if status != "running" or fields:
        patch_document(root, task, lambda current: {**current, "status": status, **fields})


def _owe_import(root, task):
    """One managed import TASK owes the engine a release of, recorded as ``delegate_documents._import`` records it."""
    from ouroboros.delegate_documents import (
        DOCUMENT_IMPORT_OBSERVED,
        DOCUMENT_IMPORT_REQUESTED,
        DOCUMENT_IMPORT_SENDING,
    )

    sha, fact = "sha256:" + "a" * 64, {"task_id": task, "material": "brief", "import_key": "key-1"}
    request = {"expectedStoreId": "store-1", "kind": "file", "mime": "text/markdown", "name": "brief.md",
               "source": f"ouroboros-task-input:{task}/attachments/brief.md", "lifetime": "managed",
               "content": {"sizeBytes": 3, "sha256": sha}}
    assert delegate_custody.emit(root, DOCUMENT_IMPORT_REQUESTED, {**fact, "store_id": "store-1", "request": request,
                                                                     "owner_task_id": task, "origin": "imported",
                                                                     "input": {"relpath": "attachments/brief.md"}})
    assert delegate_custody.emit(root, DOCUMENT_IMPORT_SENDING, fact)
    assert delegate_custody.emit(root, DOCUMENT_IMPORT_OBSERVED, {**fact, "state": "ready", "resource": {
        "resourceId": "res-1", "sha256": sha, "sizeBytes": 3, "mime": "text/markdown", "name": "brief.md"}})


def _unreadable(monkeypatch, *tasks):
    """The documents of TASKS cannot be read (every reader of the task store raises for them) until undone."""
    from ouroboros import task_store

    read = task_store.read_document

    def reader(root, task_id, *args, **kwargs):
        if task_id in tasks:
            raise OSError(f"document of {task_id} unreadable")
        return read(root, task_id, *args, **kwargs)

    monkeypatch.setattr(task_store, "read_document", reader)
    return lambda: monkeypatch.setattr(task_store, "read_document", read)


def _answers(engine, failing):
    """The engine's decision route with FAILING: run id -> the exception it raises, or the unsettled body it answers."""
    take = engine.take_run_result

    def answer(run_id, *, idempotency_key):
        if run_id not in failing:
            return take(run_id, idempotency_key=idempotency_key)
        engine.decisions.append((run_id, idempotency_key))
        if isinstance(failing[run_id], Exception):
            raise failing[run_id]
        return failing[run_id]

    engine.take_run_result = answer
    return lambda: setattr(engine, "take_run_result", take)


@pytest.fixture
def backlog(tmp_path):
    """A real store, custody and queue. ``first()`` is the step's first pass, which pins its cursor after the tasks a
    test settled and walks them; ``run()`` is any later pass, over a fresh queue snapshot."""
    from ouroboros.utils import utc_now_iso
    from tests._shared import write_queue_document

    root = tmp_path / "data"
    root.mkdir()

    def queue(ts=None):  # the published queue the tree proof and the census read: nothing running
        write_queue_document(root / "state" / "queue_snapshot.json",
                             {"ts": ts or utc_now_iso(), "pending": [], "running": []})

    queue()
    engine = Engine()
    # These tests read the ``taken`` walk alone: the step's historical walk (its own tests,
    # tests/test_historical_retirement.py) is recorded over, so it neither lists custody nor writes the row here.
    maintenance_jobs.record(root, JOB, historical_walk={"done_at": utc_now_iso(), "waiting": []})

    def run(**kwargs):
        queue()
        return rt.retire_finished_review_threads(root, engine=engine, **kwargs)  # the real live probe and census

    def row():
        return maintenance_jobs.job(root, JOB) or {}

    return SimpleNamespace(
        root=root, engine=engine, run=run, queue=queue, row=row,
        settle=lambda task, **kw: _settle(root, tmp_path, task, **kw),
        first=lambda **kwargs: rt.retire_finished_review_threads(root, engine=engine, **kwargs),
        sent=lambda: [rid for rid, _key in engine.decisions],
        record=lambda run_id, task="": ((read_document(root, task or run_id.removeprefix("run-")) or {})
                                        .get(taken.RECORDS) or {}).get(run_id),
        walked=lambda: row().get(taken.WALK), waiting=lambda: row()[rt.JOB_FIELD]["waiting"])


def test_the_first_pass_walks_what_settled_before_it_and_the_walk_is_written_once(backlog, monkeypatch):
    backlog.settle("old-1")
    backlog.settle("old-2")
    backlog.settle("live-1", status="running")  # a captured run of work still running: never walked
    listed, full = [], delegate_custody_current.runs
    monkeypatch.setattr(delegate_custody_current, "runs",
                        lambda root, where="", params=(): listed.append(where) or full(root, where, params))
    report = backlog.first()  # the step's activation walks the moment custody holds a run
    key = hashlib.sha256(b"taken:run-old-1").hexdigest()
    assert backlog.engine.decisions[0] == ("run-old-1", key) and backlog.sent() == ["run-old-1", "run-old-2"]
    assert backlog.record("run-old-1") == {"state": "taken", "at": backlog.record("run-old-1")["at"], "walk": True}
    walked = backlog.walked()
    assert walked == {"taken": 2, "absent": 0, "refused": {}, "waiting": 0, "unknown": 0,
                      "done_at": walked["done_at"]}
    assert report["observed"] == [{taken.WALK: {**walked, "tasks": 2}}], "one summary, never a line per run"
    assert report["activated"] and report["tasks"] == 0, "the first pass asked no task of its own"
    assert "" in listed and backlog.waiting() == [] and not taken.walk_open(backlog.row())
    listed.clear()
    backlog.run()  # its own records were terminal writes: the pass asks those tasks again and sends nothing
    backlog.run()
    assert backlog.sent() == ["run-old-1", "run-old-2"] and backlog.walked() == walked, "written once"
    assert "" not in listed, "a walk that is over never lists custody again"


def test_a_first_pass_with_nothing_delegated_closes_the_walk_before_it_begins(backlog):
    assert backlog.first() == {"activated": True}
    walked = backlog.walked()
    assert walked == {"taken": 0, "absent": 0, "refused": {}, "waiting": 0, "unknown": 0,
                      "done_at": walked["done_at"]}
    assert not taken.walk_open(backlog.row()) and backlog.engine.capability_reads == 0
    backlog.settle("new-1")  # delegated after the first pass: the pass's own task, never the walk's
    backlog.run()
    assert backlog.sent() == ["run-new-1"] and backlog.walked() == walked


def test_an_open_walk_whose_custody_holds_nothing_is_over(backlog):
    from ouroboros.utils import utc_now_iso

    maintenance_jobs.record(backlog.root, JOB, **{rt.JOB_FIELD: {"activated_at": utc_now_iso(), "counter": 0,
                                                                 "waiting": []}})  # activated by an earlier body
    assert taken.walk_open(backlog.row()), "an activated step without a finished walk record has it open"
    maintenance_jobs.record(backlog.root, JOB, **{taken.WALK: {"waiting": 2, "unknown": 1}})  # begun with custody
    backlog.run()
    walked = backlog.walked()
    assert walked == {"taken": 0, "absent": 0, "refused": {}, "waiting": 2, "unknown": 1,
                      "done_at": walked["done_at"]}
    assert backlog.engine.capability_reads == 0 and not taken.walk_open(backlog.row())


def test_an_open_walk_is_taken_up_by_a_child_started_for_another_job(backlog, monkeypatch):
    from ouroboros import claudexor_daemon

    backlog.settle("old-1")
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda **_kw: backlog.engine)  # the step's attach
    backlog.queue(ts="2000-01-01T00:00:00+00:00")  # a stale census: no tree can be proven over yet
    retirement.run(backlog.root, lambda: False)  # the job's first run: history-sized steps, activation, the walk
    row = backlog.row()
    assert backlog.sent() == [] and row[rt.JOB_FIELD]["waiting"] == [] and taken.walk_open(row)
    assert retirement.due(backlog.root, row, {}, child=True), "an open walk is work whenever the child runs"
    assert not retirement.due(backlog.root, row, {}, child=False), "never by itself a reason to start one"
    backlog.queue()
    assert maintenance.run_job(backlog.root, "runtime_retirement") is not None  # the child's own look
    assert backlog.sent() == ["run-old-1"] and backlog.walked()["done_at"]
    assert not retirement.due(backlog.root, backlog.row(), {}, child=True), "an ended walk rests"


def test_a_lost_first_walk_write_still_leaves_the_walk_to_the_next_child(backlog, monkeypatch):
    from ouroboros import claudexor_daemon

    backlog.settle("old-1")
    backlog.settle("old-2")
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda **_kw: backlog.engine)
    record = maintenance_jobs.record

    def lost(root, name, **fields):  # the first walk pass's own job-row write is lost after its decisions
        if taken.WALK in fields:
            raise OSError("database is locked")
        return record(root, name, **fields)

    monkeypatch.setattr(maintenance_jobs, "record", lost)
    backlog.queue()  # a fresh census for the job's run
    assert retirement.run(backlog.root, lambda: False)["state"] == "failed"  # the step's error, caught by the job
    row = backlog.row()
    assert backlog.sent() == ["run-old-1", "run-old-2"] and backlog.record("run-old-1")["walk"] is True
    assert taken.WALK not in row and row[rt.JOB_FIELD]["waiting"] == [], "decided, with no walk record written"
    assert taken.walk_open(row) and retirement.due(backlog.root, row, {}, child=True)
    assert not retirement.due(backlog.root, row, {}, child=False), "never by itself a reason to start a child"
    monkeypatch.setattr(maintenance_jobs, "record", record)
    backlog.queue()
    assert maintenance.run_job(backlog.root, "runtime_retirement") is not None  # a child started for another job
    walked = backlog.walked()
    assert walked == {"taken": 2, "absent": 0, "refused": {}, "waiting": 0, "unknown": 0,
                      "done_at": walked["done_at"]}, "counted from the walk's records"
    assert backlog.sent() == ["run-old-1", "run-old-2"], "nothing sent twice"


def test_walk_tasks_get_only_the_taken_duty_and_the_passs_own_tasks_are_never_walked(backlog, monkeypatch):
    backlog.settle("old-1")
    backlog.settle("new-1", status="running")
    backlog.engine.supported = False
    backlog.first()  # the engine lacks ``taken``: the walk stays open, nothing is sent, written or left waiting
    assert backlog.walked() is None and taken.walk_open(backlog.row()) and backlog.waiting() == []
    patch_document(backlog.root, "new-1", lambda current: {**current, "status": "completed"})  # the pass's own task
    monkeypatch.setattr(owner_keep, "is_kept", lambda _root, task: task == "new-1")
    steps, threads, documents = [], rt._Pass.task, rt._Pass.documents
    monkeypatch.setattr(rt._Pass, "task", lambda self, task: steps.append(("threads", task)) or threads(self, task))
    monkeypatch.setattr(rt._Pass, "documents",
                        lambda self, task: steps.append(("documents", task)) or documents(self, task))
    backlog.engine.supported = True
    report = backlog.run()
    assert steps == [("threads", "new-1"), ("documents", "new-1")], "the pass's own task gets every step"
    assert backlog.sent() == ["run-old-1"], "the walk task only its decision"
    assert report["holds"] == {"result_owner_keep": 1} and backlog.waiting() == ["new-1"]
    assert backlog.walked()["waiting"] == 0, "the kept task waits as the pass's own, never counted by the walk"


def test_an_open_walk_that_changed_nothing_writes_nothing(backlog):
    backlog.settle("old-1")
    backlog.engine.supported = False
    backlog.first()  # activation; the walk waits for ``taken``
    row = backlog.row()
    backlog.run()  # an engine without ``taken``
    backlog.queue(ts="2000-01-01T00:00:00+00:00")
    rt.retire_finished_review_threads(backlog.root, engine=backlog.engine)  # a census that proves nothing
    backlog.engine.supported = True
    stops = [False, True]  # the walk's start; before old-1
    assert backlog.run(should_stop=lambda: stops.pop(0) if stops else False)["stopped"]
    assert backlog.sent() == [] and taken.walk_open(row)
    assert backlog.row() == row, "no job-row write (``updated_at`` included) while the open walk stands still"
    backlog.run()
    assert backlog.sent() == ["run-old-1"] and backlog.walked()["done_at"] and backlog.row() != row


def test_the_pass_after_the_walk_treats_a_walked_task_as_any_terminal_write(backlog, monkeypatch):
    backlog.settle("old-1", thread="thr-old")  # a plan seat's reviewer thread whose run started before activation
    backlog.engine.supported = False
    backlog.first()  # activation; the walk waits for ``taken``
    _owe_import(backlog.root, "old-1")  # a managed import release the task owes; no terminal write of its own
    steps, threads, documents = [], rt._Pass.task, rt._Pass.documents
    monkeypatch.setattr(rt._Pass, "task", lambda self, task: steps.append(("threads", task)) or threads(self, task))
    monkeypatch.setattr(rt._Pass, "documents",
                        lambda self, task: steps.append(("documents", task)) or documents(self, task))
    backlog.engine.supported = True
    backlog.run()  # the walk: the decision alone
    assert backlog.sent() == ["run-old-1"] and steps == [] and backlog.engine.released == []
    report = backlog.run()  # its record was a terminal write: every duty of the step asks it now
    assert steps == [("threads", "old-1"), ("documents", "old-1")]
    assert report["holds"].get("predates_activation") == 1, "a thread whose host run predates activation stays"
    assert backlog.engine.released == ["res-1"], "the owed import is released under its own rule"
    assert backlog.sent() == ["run-old-1"] and not report["errors"]


def test_a_raising_walk_task_is_left_waiting_and_the_rest_are_still_walked(backlog, monkeypatch):
    backlog.settle("old-a")  # sorts first
    backlog.settle("old-b")
    kept = owner_keep.is_kept

    def raising(root, task):
        if task == "old-a":
            raise OSError("keep state unreadable")
        return kept(root, task)

    monkeypatch.setattr(owner_keep, "is_kept", raising)
    report = backlog.first()
    assert backlog.sent() == ["run-old-b"] and backlog.waiting() == ["old-a"]
    assert report["errors"] == ["old-a: OSError: keep state unreadable"]
    walked = backlog.walked()
    assert walked["done_at"] and walked["waiting"] == 1 and walked["unknown"] == 0, "its document reads: waiting"
    monkeypatch.setattr(owner_keep, "is_kept", kept)
    backlog.run()  # the ordinary duty asks the waiting task again
    assert backlog.sent() == ["run-old-b", "run-old-a"] and backlog.waiting() == []


def test_an_unreadable_task_counts_as_unknown_never_as_waiting_or_zero(backlog, monkeypatch):
    backlog.settle("old-a", runs=2)
    backlog.settle("old-b")
    readable = _unreadable(monkeypatch, "old-a")
    report = backlog.first()
    assert backlog.sent() == ["run-old-b"] and backlog.waiting() == ["old-a"], "the others are still decided"
    assert report["errors"] == ["old-a: OSError: document of old-a unreadable"]
    walked = backlog.walked()
    assert walked == {"taken": 1, "absent": 0, "refused": {}, "waiting": 0, "unknown": 2,
                      "done_at": walked["done_at"]}
    assert "2 runs could not be read and were left to the ordinary pass as well" in taken.receipt(walked)
    readable()
    backlog.run()  # readable again: the ordinary duty decides it, and the sealed counts stand
    assert backlog.sent() == ["run-old-b", "run-old-a", "run-old-a-2"] and backlog.walked() == walked


def test_an_unreadable_recount_owner_counts_as_unknown_never_by_its_earlier_records(backlog, monkeypatch):
    for task in ("old-a", "old-b", "old-c"):
        backlog.settle(task)
    stops = [False, False, True]  # the walk's start; old-a; before old-b
    backlog.first(should_stop=lambda: stops.pop(0) if stops else False)
    assert backlog.sent() == ["run-old-a"] and backlog.walked() is None, "decided by the walk, then stopped"
    settled, readable = taken._settled, []

    def enumerated(root, owing):  # old-a, the pass's own task now, was asked; its document turns unreadable
        readable.append(_unreadable(monkeypatch, "old-a"))
        return settled(root, owing)

    monkeypatch.setattr(taken, "_settled", enumerated)
    report = backlog.run()
    walked = backlog.walked()
    assert backlog.sent() == ["run-old-a", "run-old-b", "run-old-c"] and not report["errors"]
    assert backlog.waiting() == ["old-a"], "left to the ordinary duty, which re-reads it"
    assert walked == {"taken": 2, "absent": 0, "refused": {}, "waiting": 0, "unknown": 1,
                      "done_at": walked["done_at"]}, "old-a's run is unknown: never waiting, never dropped"
    readable[0]()
    backlog.run()  # readable again: its record already answers, and the sealed counts stand
    assert backlog.sent() == ["run-old-a", "run-old-b", "run-old-c"] and backlog.walked() == walked
    assert backlog.waiting() == []


def test_a_held_task_is_decided_by_a_later_ordinary_pass_once_its_hold_ends(backlog):
    from ouroboros.task_results import write_task_result

    write_task_result(backlog.root, "proj", "running", root_task_id="proj", parent_task_id="", delegation_role="root")
    backlog.settle("kid-1", parent="proj", runs=3)  # settled while its root still runs
    backlog.first()
    walked = backlog.walked()
    assert backlog.sent() == [] and backlog.waiting() == ["kid-1"]
    assert walked["done_at"] and walked["waiting"] == 3, "runs, never tasks"
    patch_document(backlog.root, "proj", lambda current: {**current, "status": "completed"})  # the tree ends
    backlog.run()
    assert backlog.sent() == ["run-kid-1", "run-kid-1-2", "run-kid-1-3"] and backlog.waiting() == []


def test_an_unknown_proof_is_asked_again_never_taken_from_what_could_not_be_read(backlog, monkeypatch):
    backlog.settle("old-1")
    verified = rt._continued_roots
    monkeypatch.setattr(rt, "_continued_roots", lambda _root, _tree: None)  # what it answers for any unread claim
    report = backlog.first()
    assert backlog.sent() == [] and backlog.waiting() == ["old-1"]
    assert report["holds"] == {"result_continuation_unverified": 1} and backlog.walked()["waiting"] == 1
    monkeypatch.setattr(rt, "_continued_roots", verified)  # the read works again
    backlog.run()
    assert backlog.sent() == ["run-old-1"] and backlog.waiting() == []
    assert "walk" not in backlog.record("run-old-1"), "the ordinary duty's decision, not the walk's"


def test_a_stop_leaves_the_walk_open_and_a_later_pass_finishes_it(backlog, monkeypatch):
    backlog.settle("old-1")
    backlog.settle("old-2", runs=2)
    backlog.engine.supported = False
    backlog.first()
    backlog.settle("new-1")  # written terminal after the first pass: the pass's own task
    backlog.engine.supported = True
    listed, full = [], delegate_custody_current.runs
    monkeypatch.setattr(delegate_custody_current, "runs",
                        lambda root, where="", params=(): listed.append(where) or full(root, where, params))
    stops = [True]  # the pass stops before its own task
    report = backlog.run(should_stop=lambda: stops.pop(0) if stops else False)
    assert report["stopped"] and backlog.sent() == [] and "" not in listed, "a stopped pass lists nothing to walk"
    stops = [False, False, False, False, True]  # new-1; the walk's start; old-1; old-2; between old-2's decisions
    report = backlog.run(should_stop=lambda: stops.pop(0) if stops else False)
    assert report["stopped"] and backlog.sent() == ["run-new-1", "run-old-1", "run-old-2"]
    assert backlog.walked() == {"waiting": 1, "unknown": 0}, "open; old-2's last run is left to the ordinary duty"
    assert backlog.waiting() == ["old-2"]
    backlog.run()  # old-2's last run is that duty's to send; the walk counts its own records and is over
    assert backlog.sent() == ["run-new-1", "run-old-1", "run-old-2", "run-old-2-2"]
    walked = backlog.walked()
    assert walked == {"taken": 2, "absent": 0, "refused": {}, "waiting": 1, "unknown": 0,
                      "done_at": walked["done_at"]}


def test_typed_refusals_and_absent_runs_are_counted_by_their_engine_code(backlog):
    for task in ("old-1", "old-2", "old-3"):
        backlog.settle(task)
    _answers(backlog.engine, {"run-old-2": ClaudexorUnavailable("run_taken_not_patch", "no patch", status_code=409),
                              "run-old-3": ClaudexorUnavailable("not_found", "no such run", status_code=404)})
    backlog.first()
    walked = backlog.walked()
    assert walked == {"taken": 1, "absent": 1, "refused": {"run_taken_not_patch": 1}, "waiting": 0, "unknown": 0,
                      "done_at": walked["done_at"]}
    assert backlog.record("run-old-2")["code"] == "run_taken_not_patch"
    assert backlog.record("run-old-3")["state"] == "absent"


@pytest.mark.parametrize("answer", [
    ClaudexorUnavailable("delivery_failed", "engine error", status_code=500),
    ClaudexorUnavailable("daemon_stopping", "shutting down", status_code=503),
    ClaudexorUnavailable("daemon_unreachable", "read timed out", status_code=0),
    {"accepted": False, "status": "pending"},
], ids=["http_500", "daemon_stopping", "timeout", "unaccepted"])
def test_an_unsettled_answer_leaves_its_task_waiting_and_ends_the_walks_pass(backlog, answer):
    backlog.settle("old-1", runs=2)
    backlog.settle("old-2")
    backlog.settle("old-3")
    _answers(backlog.engine, {"run-old-1": answer})  # this run never settles
    backlog.first()
    assert backlog.sent() == ["run-old-1"], "neither old-1's other run nor old-2 and old-3 are asked in this pass"
    assert backlog.waiting() == ["old-1"] and backlog.walked() == {"waiting": 2, "unknown": 0}, "open: no done_at"
    report = backlog.run()  # old-1 is the ordinary duty's now: it holds that one run and goes on; the walk moves past
    assert backlog.sent() == ["run-old-1", "run-old-1", "run-old-1-2", "run-old-2", "run-old-3"]
    assert report["holds"] == {"result_taken_unanswered": 1} and backlog.waiting() == ["old-1"]
    walked = backlog.walked()
    assert walked == {"taken": 2, "absent": 0, "refused": {}, "waiting": 2, "unknown": 0,
                      "done_at": walked["done_at"]}


def test_the_last_owner_unanswered_leaves_the_walk_open(backlog):
    backlog.settle("old-9")
    restore = _answers(backlog.engine, {"run-old-9": ClaudexorUnavailable("daemon_unreachable", "down")})
    backlog.first()
    assert backlog.waiting() == ["old-9"] and backlog.walked() == {"waiting": 1, "unknown": 0}
    assert taken.walk_open(backlog.row()), "an unsettled answer never seals the walk, last owner included"
    restore()
    backlog.run()  # the ordinary duty decides it; the walk, with nothing left, is over
    assert backlog.sent() == ["run-old-9", "run-old-9"] and backlog.walked()["done_at"]


def test_one_unanswered_decision_never_holds_the_ordinary_passs_others(backlog):
    assert backlog.first() == {"activated": True}
    backlog.settle("new-a")
    backlog.settle("new-b")
    _answers(backlog.engine, {"run-new-a": ClaudexorUnavailable("daemon_unreachable", "read timed out")})
    report = backlog.run()
    assert backlog.sent() == ["run-new-a", "run-new-b"] and backlog.record("run-new-b")["state"] == "taken"
    assert backlog.waiting() == ["new-a"] and report["holds"] == {"result_taken_unanswered": 1}


def test_a_timeout_retried_predecessor_is_walked_once_its_retry_settled(backlog):
    from ouroboros.task_results import write_task_result

    for tree in ("proj", "proj2"):  # an earlier body's timeout retry of a subagent, inside an ended tree
        write_task_result(backlog.root, tree, "completed", root_task_id=tree, parent_task_id="", delegation_role="root")
    backlog.settle("pred", status="interrupted", parent="proj", superseded_by="retry", retry_task_id="retry")
    backlog.settle("retry", runs=0, parent="proj", timeout_retry_from="pred", original_task_id="pred")
    backlog.settle("pred2", status="interrupted", parent="proj2", superseded_by="retry2",
                   retry_task_id="retry2")
    backlog.settle("retry2", status="running", runs=0, parent="proj2", timeout_retry_from="pred2",
                   original_task_id="pred2")
    backlog.first()
    assert backlog.sent() == ["run-pred"], "a predecessor whose retry is still running waits for that retry's end"
    assert backlog.walked()["waiting"] == 0 and backlog.waiting() == []
    patch_document(backlog.root, "retry2", lambda current: {**current, "status": "completed"})
    backlog.run()  # its terminal write names the predecessor (``retry_line``): the ordinary duty takes it
    assert backlog.sent() == ["run-pred", "run-pred2"]


def test_a_queue_census_that_goes_stale_leaves_the_walk_open(backlog, monkeypatch):
    for task in ("old-1", "old-2", "old-3"):
        backlog.settle(task)
    listed, full = [], delegate_custody_current.runs
    monkeypatch.setattr(delegate_custody_current, "runs",
                        lambda root, where="", params=(): listed.append(where) or full(root, where, params))
    backlog.queue(ts="2000-01-01T00:00:00+00:00")  # stale before the walk starts: nothing is listed or asked
    backlog.first()
    assert backlog.sent() == [] and backlog.walked() is None and taken.walk_open(backlog.row())
    assert "" not in listed, "a census that proves nothing never lists custody"
    take = backlog.engine.take_run_result

    def answer(run_id, *, idempotency_key):  # the queue's heartbeat stops right after the first decision
        backlog.queue(ts="2000-01-01T00:00:00+00:00")
        return take(run_id, idempotency_key=idempotency_key)

    backlog.engine.take_run_result = answer
    backlog.run()
    assert backlog.sent() == ["run-old-1"] and backlog.waiting() == [], "the rest is not handed over"
    assert taken.walk_open(backlog.row())
    backlog.engine.take_run_result = take
    backlog.run()  # a fresh census
    walked = backlog.walked()
    assert backlog.sent() == ["run-old-1", "run-old-2", "run-old-3"]
    assert walked == {"taken": 3, "absent": 0, "refused": {}, "waiting": 0, "unknown": 0,
                      "done_at": walked["done_at"]}


def test_a_walk_that_fails_is_its_passs_error_never_the_passs_end(backlog, monkeypatch):
    backlog.settle("old-1")
    probe = delegate_custody_current.has_runs
    monkeypatch.setattr(delegate_custody_current, "has_runs", lambda _root: (_ for _ in ()).throw(OSError("unread")))
    report = backlog.first()  # an unreadable probe walks; the walk fails and stays open
    assert report["errors"] == ["taken walk: OSError: unread"] and backlog.walked() is None
    assert taken.walk_open(backlog.row())
    backlog.settle("new-1")  # the pass's own task
    pinned = backlog.row()[rt.JOB_FIELD]["counter"]
    report = backlog.run()
    assert report["errors"] == ["taken walk: OSError: unread"] and taken.walk_open(backlog.row())
    assert backlog.sent() == ["run-new-1"], "the pass's own task was asked"
    assert backlog.row()[rt.JOB_FIELD]["counter"] > pinned, "and the pass kept its cursor"
    monkeypatch.setattr(delegate_custody_current, "has_runs", probe)
    backlog.run()
    assert backlog.sent() == ["run-new-1", "run-old-1"] and backlog.walked()["done_at"]


@pytest.fixture
def owner_chat(monkeypatch, tmp_path):
    """One install on the real state, owner-chat, chat-row and notice-receipt writers; the maintenance child's
    server half with its spawn and report reading stubbed (``tests/test_legacy_memory_notice.py``'s boot)."""
    from ouroboros.startup_migrations import prepare_startup_state
    from supervisor import message_bus as bus
    from supervisor import state as ss

    for name, path in {"DRIVE_ROOT": tmp_path, "STATE_PATH": tmp_path / "state/state.json",
                       "STATE_LAST_GOOD_PATH": tmp_path / "state/state.last_good.json",
                       "STATE_LOCK_PATH": tmp_path / "locks/state.lock"}.items():
        monkeypatch.setattr(ss, name, path)
    monkeypatch.setattr(bus, "DATA_DIR", tmp_path)
    prepare_startup_state(tmp_path)
    monkeypatch.setattr(bus, "get_bridge", lambda: SimpleNamespace(send_message=lambda *a, **kw: None))
    ss.save_state({"owner_chat_id": 7, "owner_id": 1})
    child, stop = maintenance.MaintenanceChild(), {"during": False}

    def finish(unit, *_a, **_kw):  # the child's run ends; Stop or Panic may have arrived meanwhile
        if stop["during"]:
            unit._stopped.set()
        return {}

    monkeypatch.setattr(maintenance, "due", lambda _root: ["runtime_retirement"])
    monkeypatch.setattr(maintenance, "spawn_maintenance_job", lambda *_a, **_kw: SimpleNamespace(pid=1))
    monkeypatch.setattr(maintenance, "finish_maintenance_job", finish)
    return SimpleNamespace(
        root=tmp_path, state=ss, child=child, stop=stop, child_ran=lambda: child._run(tmp_path, tmp_path),
        walk=lambda **record: maintenance_jobs.record(tmp_path, JOB, **{taken.WALK: record}),
        rows=lambda: [row for row in dialogue_test.chat_rows(tmp_path) if row.get("type") == taken.NOTICE])


def test_the_walks_receipt_reaches_the_owners_chat_once_after_a_child_run(owner_chat):
    counts = {"taken": 4131, "absent": 12, "refused": {"run_taken_continuable": 2, "run_taken_not_patch": 1},
              "waiting": 40, "unknown": 2}
    owner_chat.walk(**counts)  # counted, not over yet
    owner_chat.child_ran()
    assert owner_chat.rows() == []
    owner_chat.walk(**counts, done_at="2026-11-08T00:00:00+00:00")
    owner_chat.state.update_state(lambda st: st.__setitem__("owner_chat_id", 0))  # nowhere to deliver it
    owner_chat.child_ran()
    assert owner_chat.rows() == []
    owner_chat.state.update_state(lambda st: st.__setitem__("owner_chat_id", 7))
    owner_chat.child_ran()
    [row] = owner_chat.rows()
    assert row["direction"] == "system" and row["chat_id"] == 7
    assert obligations.members(owner_chat.root, "upgrade_notices")[f"7:{taken.NOTICE}"]["recorded"] is True
    text = row["text"]
    for fact in ("One-time catch-up with the engine", "taken 4,131 runs", "ordinary retention window",
                 "keeps its own copy", "refused 3 runs (2 still continuable in the engine, 1 not a patch result)",
                 "no longer had 12 runs", "40 runs were left to the ordinary pass",
                 "2 runs could not be read and were left to the ordinary pass as well"):
        assert fact in text, (fact, text)
    assert "run_taken" not in text, "known engine codes reach the owner in words"
    owner_chat.child_ran()
    owner_chat.child_ran()
    assert len(owner_chat.rows()) == 1, "its landed row is its receipt: never repeated"


def test_no_receipt_while_the_dialogue_cannot_answer_after_stop_or_for_a_walk_that_met_nothing(owner_chat,
                                                                                               monkeypatch):
    from ouroboros import dialogue_store

    assert taken.receipt({"taken": 0, "absent": 0, "refused": {}, "waiting": 0, "unknown": 0, "done_at": "x"}) == ""
    one = taken.receipt({"taken": 1, "absent": 0, "refused": {"400": 1}, "waiting": 1, "unknown": 1, "done_at": "x"})
    assert "taken 1 run," in one and "1 as an invalid request" in one and "1 run was left" in one
    assert "1 run could not be read and was left" in one
    owner_chat.walk(taken=0, absent=0, refused={}, waiting=0, unknown=0, done_at="2026-11-08T00:00:00+00:00")
    owner_chat.child_ran()
    assert owner_chat.rows() == [], "a walk that met nothing tells the owner nothing"
    owner_chat.walk(taken=2, absent=0, refused={}, waiting=0, unknown=0, done_at="2026-11-08T00:00:00+00:00")
    importing = dialogue_store.require_imported

    def pending(*_a, **_kw):  # the archive import has not answered for the owner's chat yet
        raise dialogue_store.StoreMigrating("chat history still importing")

    monkeypatch.setattr(dialogue_store, "require_imported", pending)
    owner_chat.child_ran()
    assert owner_chat.rows() == [], "nothing is sent while the dialogue cannot say whether it was"
    monkeypatch.setattr(dialogue_store, "require_imported", importing)
    owner_chat.stop["during"] = True  # Stop or Panic arrived while the child ran
    owner_chat.child_ran()
    assert owner_chat.rows() == [], "nothing is written after Stop"
    owner_chat.child._stopped.clear()
    owner_chat.stop["during"] = False
    owner_chat.child_ran()
    assert len(owner_chat.rows()) == 1
