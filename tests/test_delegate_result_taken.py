"""Ouroboros tells the engine once that it has taken a finished delegated result (owner Q2=A).

A delegated Git run whose patch Ouroboros captured into its task record, and whose line can no longer be
continued, gets exactly one typed ``taken`` decision; a continuable, uncaptured, ``files``, in-place, plan or
review run gets none; an engine whose decision vocabulary names no ``taken`` keeps the task waiting without a
decision record; an unanswered decision is repeated under the same key. The engine is an in-memory MOCK of Claudexor's C7
route shape (``POST /v2/runs/:id/decision {"action": "taken", "reason": "captured_result"}``), never the real one.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import delegate_custody, delegate_custody_current, owner_keep
from ouroboros import delegate_result_taken as taken
from ouroboros import review_thread_retention as rt
from ouroboros.delegate_custody import RunCustody
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.task_store import patch_document, read_document
from tests._store_shared import seed_result
from tests.test_store import document

TASK = "task-taken-1"


class Engine:
    """The decision route and the decision vocabulary, recorded."""

    def __init__(self, *, supported=True, version="5.1.0"):
        self.supported, self.engine_version, self.store_identity = supported, version, "store-1"
        self.decisions, self.fail = [], []
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
