"""Card rounds are accepted loop replies, independent of monetary physical attempts."""
import asyncio
import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros import agent_task_pipeline as pipeline
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.loop_llm_call import call_llm_with_retry
from ouroboros.utils import append_jsonl
from supervisor.events_worker_reports import _handle_log_event, _handle_task_metrics
from supervisor.log_addressing import TurnEventQueue
from tests.test_task_correlation_facts import _Clock, _TimedLLM


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("attempt", [None, 0, 2])
def test_round_producer_transport_storage_and_cold_history(tmp_path, monkeypatch, direct, attempt):
    from ouroboros import loop_llm_call

    logs = tmp_path / "logs"
    logs.mkdir()
    clock, usage, queue_sink, pushed = _Clock(), {"rounds": 2}, queue.Queue(), []
    monkeypatch.setattr(loop_llm_call, "time", clock)
    events = TurnEventQueue(queue_sink, "counter-task", 7, direct_chat=direct, telemetry_only=not direct)
    # Two physical replies, one usable response. The loop's index is deliberately different.
    llm = _TimedLLM(clock, [({"content": "", "tool_calls": []}, {}),
                           ({"content": "Checked the source.", "tool_calls": []}, {})])
    message, _ = call_llm_with_retry(llm, [{"role": "user", "content": "check"}], "fixture/model",
                                     None, "medium", 2, logs, "counter-task", 18, events, usage)
    assert message["content"] == "Checked the source." and usage["rounds"] == 3
    raw = next(evt for evt in queue_sink.queue if evt.get("data", {}).get("type") == "llm_round_finished")
    # JSON serialization and the real supervisor handler preserve the addressed fact even after departure.
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, bridge=SimpleNamespace(push_log=pushed.append),
                          append_jsonl=lambda *_: None)
    _handle_log_event(json.loads(json.dumps(raw)), ctx)
    assert pushed[-1]["rounds"] == 3 and pushed[-1]["chat_id"] == 7
    assert pushed[-1]["round"] == 18
    assert bool(pushed[-1].get("_is_direct_chat")) is direct
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_task_cost_fields", lambda *a: {"total_rounds": 99})
    task = {"id": "counter-task", "type": "task", "chat_id": 7, "text": "check",
            "_is_direct_chat": direct, "_skip_post_task_synthesis": True}
    if attempt is not None:
        task["_attempt"] = attempt
        pipeline.write_task_result(tmp_path, task["id"], "running", task_attempt=attempt)
    pending = []
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    pipeline.emit_task_results(env, None, None, pending, task, message["content"], usage,
                               {"tool_calls": [], "reasoning_notes": []}, 0.0, logs)
    metrics = next(evt for evt in pending if evt["type"] == "task_metrics")
    evaluation = next(row for line in (logs / "events.jsonl").read_text().splitlines()
                      if (row := json.loads(line)).get("type") == "task_eval")
    _handle_task_metrics(json.loads(json.dumps(metrics)), ctx)
    assert pushed[-1]["rounds"] == 3 and pushed[-1]["tool_calls"] == 0
    for row in (metrics, evaluation, pushed[-1]):
        assert row.get("task_attempt") == attempt and ("task_attempt" in row) == (attempt is not None)
    stored = pipeline.load_task_result(tmp_path, "counter-task")
    assert stored["total_rounds"] == 99
    assert stored["loop_outcome"]["usage"]["total_rounds"] == 3
    # History uses real stored results and the producer's saved task facts; no hand-built usage path.
    request = SimpleNamespace(query_params={"limit": "20", "chat_id": "7"})
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(request))
    rows = json.loads(response.body)["messages"]
    summaries = [row for row in rows if row.get("system_type") == "task_summary"]
    assert summaries and all(row["rounds"] == 3 for row in summaries)


def _card_attempt(frame):
    """The attempt a card folds a frame into (``settled_card.attemptOf``): its own number, else its task's owner."""
    owner = frame.get("execution_owner") if isinstance(frame.get("execution_owner"), dict) else {}
    ordinals = (frame.get("task_attempt"), frame.get("_attempt"),
                owner.get("task_attempt") if owner.get("task_id") == frame.get("task_id") else None)
    return next((value for value in ordinals if type(value) is int and value >= 0), None)


@pytest.mark.parametrize("running", ["same", "next", "gone"])
@pytest.mark.parametrize("lane", ["first-direct", "pooled"])
def test_a_round_drained_after_the_next_assignment_keeps_its_own_attempt(tmp_path, monkeypatch, lane, running):
    """The round ended, its worker died and the same task was assigned again before the bus was drained. The
    endpoint made for that attempt names it, so the drain's RUNNING owner (the next attempt's, or none) never
    counts the round toward another attempt's card; an explicit producer number still wins."""
    from ouroboros import loop_llm_call
    from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION
    from ouroboros.task_status import execution_owner_record
    from supervisor.log_addressing import task_event_queue

    logs = tmp_path / "logs"
    logs.mkdir()
    clock, queue_sink, pushed = _Clock(), queue.Queue(), []
    monkeypatch.setattr(loop_llm_call, "time", clock)
    task = {"id": "drained-round", "chat_id": 7}
    if lane == "first-direct":  # admitted without a queue ordinal: physical attempt 0, the loop's number None
        task["_is_direct_chat"] = True
        task["_execution_owner"] = {**execution_owner_record(tmp_path, task, "direct"),
                                    "lane": LANE_CONVERSATION, "executor": EXECUTOR_WORKER}
        own, ordinal, newer = 0, None, 2
    else:
        task["_attempt"] = 2
        task["_execution_owner"] = execution_owner_record(tmp_path, task, "pooled")
        own, ordinal, newer = 2, 2, 3
    assert task["_execution_owner"]["task_attempt"] == own
    llm = _TimedLLM(clock, [({"content": "Checked.", "tool_calls": []}, {})])
    call_llm_with_retry(llm, [{"role": "user", "content": "check"}], "fixture/model", None, "medium", 2, logs,
                        task["id"], 1, task_event_queue(queue_sink, task), {}, task_attempt=ordinal)
    raw = next(evt for evt in queue_sink.queue if evt.get("data", {}).get("type") == "llm_round_finished")
    assigned = {**task, "_attempt": newer, "_execution_owner": {**task["_execution_owner"], "task_attempt": newer}}
    rows = {"same": {task["id"]: {"task": task}}, "next": {task["id"]: {"task": assigned}}, "gone": {}}[running]
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=rows, bridge=SimpleNamespace(push_log=pushed.append),
                          append_jsonl=lambda *_: None)
    _handle_log_event(json.loads(json.dumps(raw)), ctx)
    assert pushed[-1]["rounds"] == 1 and _card_attempt(pushed[-1]) == own


@pytest.mark.parametrize('owner', [{}, {'task_id': 'other', 'task_attempt': 0},
                                   {'task_id': 'root', 'task_attempt': False}, {'task_id': 'root', 'task_attempt': '0'}])
def test_evaluation_does_not_invent_attempt_from_unbound_execution(tmp_path, monkeypatch, owner):
    logs = tmp_path / 'logs'
    logs.mkdir()
    task = {'id': 'root', 'type': 'task', 'chat_id': 1, 'text': 'Check', '_is_direct_chat': True,
            '_execution_owner': owner, '_attempt': 1, '_skip_post_task_synthesis': True}
    pending = []
    monkeypatch.setattr(pipeline, '_task_cost_fields', lambda *_: {})
    pipeline.emit_task_results(SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), None, None,
                               pending, task, 'Checked.', {'rounds': 3},
                               {'tool_calls': [], 'reasoning_notes': []}, 0.0, logs)
    [metrics] = [row for row in pending if row['type'] == 'task_metrics']
    [evaluation] = [row for line in (logs / 'events.jsonl').read_text().splitlines()
                    if (row := json.loads(line)).get('type') == 'task_eval']
    assert 'task_attempt' not in metrics and 'task_attempt' not in evaluation


@pytest.mark.parametrize(("retained", "saved_usage"), [
    pytest.param(False, {}, id='False-missing'),
    pytest.param(False, {"total_rounds": 0}, id='False-zero'),
    pytest.param(True, {"total_rounds": None}, id='True-unknown'),
    pytest.param(True, {"total_rounds": 3}, id='True-accepted'),
])
def test_history_rounds_override_stale_physical_summary_only_when_known(tmp_path, saved_usage, retained):
    from ouroboros.project_dialogue import append_canonical_task_summary
    from tests._store_shared import read_result, seed_result

    seed_result(tmp_path, "old", {"status": "completed", "total_rounds": 99,
        "loop_outcome": {"usage": {**saved_usage, "extra": "U" * 100_000}, "extra": "L" * 100_000}})
    full = read_result(tmp_path, "old")
    append_canonical_task_summary(tmp_path, retained=True, row={
        "task_id": "old", "chat_id": 7, "direction": "system", "type": "task_summary",
        "summary_id": "task-facts:old", "text": "", "tool_calls": 1, "ts": "2026-10-01T00:00:00Z",
        **({"rounds": 99} if retained else {}),
    })
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"chat_id": "7"})))
    assert response.status_code == 200
    rows = [row for row in json.loads(response.body)["messages"] if row.get("system_type") == "task_summary"]
    assert rows and all(row["outcome_final"] for row in rows)
    expected = saved_usage.get("total_rounds")
    for row in rows:
        if expected is not None:
            assert row["rounds"] == expected
        elif retained:
            assert row["rounds"] == 99, "unknown canonical counts do not erase retained facts"
        else:
            assert "rounds" not in row, "physical task.total_rounds is never a fallback"
    assert read_result(tmp_path, "old") == full


@pytest.mark.parametrize(("own", "replica_attempt", "expected"), [
    pytest.param({"loop_outcome": {"usage": {"total_rounds": 0}}, "terminal_origin": "host_notice"}, 1,
                 (0, "host_notice"), id="own-zero-foreign-replica"),
    pytest.param({}, 1, (None, None), id="own-unknown-foreign-replica"),
    pytest.param({"loop_outcome": {"usage": {"total_rounds": 0}}, "terminal_origin": "host_notice"}, None,
                 (0, "host_notice"), id="own-zero-unnumbered-replica"),
    pytest.param({}, None, (None, None), id="own-unknown-unnumbered-replica"),
    # The same attempt's replica: the host's own failure still shows its own count and origin, not the child's success.
    pytest.param({'loop_outcome': {'usage': {'total_rounds': 0}}, 'terminal_origin': 'host_notice'}, 2,
                 (0, 'host_notice'), id='own-zero-same-number-replica'),
    pytest.param({}, 2, (None, None), id='own-unknown-same-number-replica'),
    pytest.param(None, 2, (3, "model_final"), id="replica-supplies-the-ending"),
])
def test_a_canonical_failure_shows_only_its_own_rounds_and_origin(tmp_path, own, replica_attempt, expected):
    """Attempt 2 failed on the canonical drive while the actor drive still holds an older completed result. The
    effective read keeps the failure's status and number beside that replica's loop facts; on reload its card shows
    the failure's own count and origin, an unknown one staying unknown. A replica that supplies the ending itself
    (its canonical row still running) keeps its own."""
    child = tmp_path / "actor"
    pipeline.write_task_result(tmp_path, "split-root", "running" if own is None else "failed", chat_id=7,
                               task_attempt=2, child_drive_root=str(child), _terminal_observed=own is not None,
                               **(own or {}))
    pipeline.write_task_result(child, "split-root", "completed", chat_id=7, _terminal_observed=True,
                               loop_outcome={"usage": {"total_rounds": 3}}, terminal_origin="model_final",
                               **({} if replica_attempt is None else {"task_attempt": replica_attempt}))
    pipeline._record_task_facts(SimpleNamespace(drive_root=tmp_path),
                                {"id": "split-root", "chat_id": 7, "_attempt": 2}, {}, {"tool_calls": []},
                                tmp_path / "logs")
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"chat_id": "7"})))
    assert response.status_code == 200
    [row] = [r for r in json.loads(response.body)["messages"] if r.get("system_type") == "task_summary"]
    assert row["task_attempt"] == 2 and row["task_terminal_status"] == ("completed" if own is None else "failed")
    assert (row["rounds"], row.get("terminal_origin")) == expected


@pytest.mark.parametrize("source, expected", [("no_response", 0), ("unavailable", None)])
def test_failed_loop_rounds_match_live_saved_facts_and_history(tmp_path, monkeypatch, source, expected):
    from ouroboros import loop_llm_call
    from ouroboros.agent import _task_exception_terminal
    from ouroboros.loop import run_llm_loop
    from ouroboros.tools.registry import ToolRegistry
    from tests import _dialogue

    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    task = {"id": "failed-counter", "type": "task", "chat_id": 7, "text": "check",
            "_is_direct_chat": True, "_skip_post_task_synthesis": True}
    logs = tmp_path / "logs"
    logs.mkdir()
    if source == "no_response":
        monkeypatch.setattr(loop_llm_call, "time", _Clock())

        class EmptyLLM:
            calls = 0

            def default_model(self):
                return "fixture/model"

            def chat(self, **_kwargs):
                self.calls += 1
                return {"content": "", "tool_calls": []}, {"response_finish_reason": "stop"}

        llm = EmptyLLM()
        text, usage, trace = run_llm_loop(
            messages=[{"role": "user", "content": "check"}],
            tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), llm=llm,
            drive_logs=logs, emit_progress=lambda *_a, **_k: None, incoming_messages=queue.Queue(),
            task_id=task["id"], drive_root=tmp_path,
        )
        assert llm.calls > 0 and "rounds" not in usage
        assert usage["execution_status"] == "infra_failed"
    else:
        text, usage, trace = _task_exception_terminal(env, task, RuntimeError("no retained loop snapshot"), logs)
        assert usage["loop_evidence_unavailable"] and trace["loop_evidence_unavailable"]

    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    pending, pushed = [], []
    pipeline.emit_task_results(env, None, None, pending, task, text, usage, trace, 0.0, logs)
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, bridge=SimpleNamespace(push_log=pushed.append),
                          append_jsonl=append_jsonl)
    _handle_task_metrics(json.loads(json.dumps(next(evt for evt in pending if evt["type"] == "task_metrics"))), ctx)
    stored = pipeline.load_task_result(tmp_path, task["id"])
    facts = next(row for row in _dialogue.chat_rows(tmp_path) if row.get("summary_kind") == "host_task_facts")
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"chat_id": "7"})))
    assert response.status_code == 200
    summaries = [row for row in json.loads(response.body)["messages"] if row.get("system_type") == "task_summary"]
    assert summaries and stored["status"] == "failed"
    assert {"live": pushed[-1]["rounds"], "saved": stored["loop_outcome"]["usage"]["total_rounds"],
            "facts": facts["rounds"], "history": summaries[0].get("rounds")} == dict.fromkeys(
                ("live", "saved", "facts", "history"), expected)


@pytest.mark.parametrize("lane", ["ordinary", "split", "split-project"])
@pytest.mark.parametrize("skip_post", [False, True], ids=["post", "skip-post"])
@pytest.mark.parametrize("usage, expected", [
    ({"rounds": 3}, 3), ({"rounds": 0}, 0), ({}, 0),
    ({"rounds": 17, "loop_evidence_unavailable": True}, None),
], ids=["accepted", "zero", "no-response", "unknown"])
def test_emitted_rounds_survive_split_dispatch_and_result_pruning(
    tmp_path, monkeypatch, lane, skip_post, usage, expected,
):
    from ouroboros import llm, post_task_evolution
    from tests import _dialogue
    from tests._store_shared import delete_result

    canonical = tmp_path / "canonical"
    actor = canonical if lane == "ordinary" else tmp_path / "actor"
    for root in {canonical, actor}:
        (root / "logs").mkdir(parents=True)
    task = {"id": "root-counter", "root_task_id": "root-counter", "type": "task",
            "chat_id": 7, "text": "Check the result", "drive_root": str(actor),
            "budget_drive_root": str(canonical), "_is_direct_chat": lane == "ordinary",
            "_skip_post_task_synthesis": skip_post}
    if lane == "split-project":
        task["project_id"] = "counter-project"
    trace = {"tool_calls": [], "reasoning_notes": [],
             "loop_evidence_unavailable": bool(usage.get("loop_evidence_unavailable"))}
    post_roots = []
    # Keep real dispatch, snapshots, checkpoint and facts writes. Only paid work is replaced.
    monkeypatch.setattr(llm, "LLMClient", SimpleNamespace)
    monkeypatch.setattr(pipeline, "_run_memory_fallback_draft",
                        lambda env, *_args: post_roots.append(env.drive_root))
    monkeypatch.setattr(post_task_evolution, "maybe_promote", lambda *_args: None)
    monkeypatch.setattr(pipeline, "_task_cost_fields", lambda *_args: {"total_rounds": 99})
    events = []
    pipeline.emit_task_results(
        SimpleNamespace(drive_root=actor, repo_dir=tmp_path), None, None, events, task,
        "Checked the result.", dict(usage), trace, 0.0, actor / "logs",
    )
    assert post_roots == ([] if skip_post else [canonical if lane == "split" else actor])
    metrics = next(event for event in events if event["type"] == "task_metrics")
    assert metrics["total_rounds"] == 99, "ledger attempts differ from accepted loop replies"
    stored = pipeline.load_task_result(actor, task["id"])
    if actor != canonical:
        assert "loop_outcome" not in (pipeline.load_task_result(canonical, task["id"]) or {}), (
            "the facts must survive before the actor result is copied back")
    [facts] = [row for row in _dialogue.chat_rows(canonical)
               if row.get("summary_kind") == "host_task_facts" and row.get("task_id") == task["id"]]

    # Remove both result copies, then use the real cold-history consumer of the retained facts.
    for root in {canonical, actor}:
        delete_result(root, task["id"])
        assert pipeline.load_task_result(root, task["id"]) is None
    response = asyncio.run(make_chat_history_endpoint(canonical)(
        SimpleNamespace(query_params={"chat_id": "7"})))
    assert response.status_code == 200
    summaries = [row for row in json.loads(response.body)["messages"]
                 if row.get("system_type") == "task_summary" and row.get("task_id") == task["id"]
                 and "rounds" in row]
    assert summaries, "pruned results still have retained counter facts in history"
    assert {"live": metrics["rounds"], "saved": stored["loop_outcome"]["usage"]["total_rounds"],
            "facts": facts["rounds"], "history": summaries[0]["rounds"]} == dict.fromkeys(
                ("live", "saved", "facts", "history"), expected)
    assert all(row["rounds"] == expected for row in summaries)


@pytest.mark.parametrize("snapshot, expected", [
    ({}, None), ({"loop_outcome": {"usage": {"total_rounds": None}}}, None),
    ({"loop_outcome": {"usage": {"total_rounds": 0}}}, 0),
    ({"loop_outcome": {"usage": {"total_rounds": 3}}}, 3),
], ids=["missing", "unknown", "zero", "accepted"])
def test_recovery_facts_use_retained_loop_rounds(tmp_path, monkeypatch, snapshot, expected):
    from tests import _dialogue

    pipeline.write_task_result(tmp_path, "recovered", pipeline.STATUS_COMPLETED, chat_id=7,
        root_task_id="recovered", objective="finish recovery", total_rounds=99, **snapshot,
        root_phase_checkpoint={"phase": "task_acceptance", "status": "pass", "post_task_synthesis": "pending_once"})

    def free_facts(env, task, usage, trace, logs, **_kwargs):
        pipeline._record_task_facts(env, task, usage, trace, logs)
        pipeline._set_root_post_task_checkpoint(env, task, "completed")

    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", free_facts)
    assert pipeline.recover_pending_root_post_task_synthesis(tmp_path, tmp_path) == 1
    facts = next(row for row in _dialogue.chat_rows(tmp_path) if row.get("summary_kind") == "host_task_facts")
    assert facts["rounds"] == expected
    assert pipeline.recover_pending_root_post_task_synthesis(tmp_path, tmp_path) == 0


def test_recovery_facts_use_only_their_own_saved_execution_owner(tmp_path, monkeypatch):
    from ouroboros.task_status import execution_owner_record
    from tests._dialogue import chat_rows

    for tid in ('bound', 'unbound'):
        owner_id = tid if tid == 'bound' else 'other'
        pipeline.write_task_result(tmp_path, tid, pipeline.STATUS_COMPLETED, chat_id=1, root_task_id=tid,
            objective='finish recovery', _is_direct_chat=True,
            execution_owner=execution_owner_record(tmp_path, {'id': owner_id}, 'direct'),
            loop_outcome={'usage': {'total_rounds': 3}},
            root_phase_checkpoint={'phase': 'task_acceptance', 'status': 'pass', 'post_task_synthesis': 'pending_once'})

    def free_facts(env, task, usage, trace, logs, **_kwargs):
        pipeline._record_task_facts(env, task, usage, trace, logs)
        pipeline._set_root_post_task_checkpoint(env, task, 'completed')

    monkeypatch.setattr(pipeline, '_run_post_task_processing_async', free_facts)
    assert pipeline.recover_pending_root_post_task_synthesis(tmp_path, tmp_path) == 2
    facts = {row['task_id']: row for row in chat_rows(tmp_path) if row.get('summary_kind') == 'host_task_facts'}
    assert facts['bound']['task_attempt'] == 0 and facts['bound']['rounds'] == 3
    assert 'task_attempt' not in facts['unbound'] and facts['unbound']['rounds'] == 3


@pytest.mark.parametrize('stale', [True, False], ids=['older-beside-failed', 'same-attempt-finishes'])
def test_a_late_copy_keeps_each_attempts_stored_ending_and_card_its_own(tmp_path, stale):
    # Attempt 2 failed with zero rounds, a host origin and its own reason. Attempt 1's old completed result is copied
    # back late: the stored row and the history card read the same ending before and after. Attempt 2's OWN completed
    # result, copied the same way, does land on both.
    from ouroboros import headless
    from tests._store_shared import read_result

    parent = tmp_path / 'data'
    parent.mkdir()
    tid, child = 'late-copy', headless.prepare_task_drive(parent, 'late-copy', 'empty')
    own = {'attempt': 2, 'status': 'failed', 'reason': 'provider_unavailable', 'origin': 'host_notice', 'rounds': 0}
    if stale:
        pipeline.write_task_result(parent, tid, 'failed', chat_id=7, task_attempt=2, child_drive_root=str(child),
                                   _terminal_observed=True, result='Attempt 2 stopped by the host.',
                                   reason_code='provider_unavailable', terminal_origin='host_notice',
                                   loop_outcome={'usage': {'total_rounds': 0}})
    else:
        pipeline.write_task_result(parent, tid, 'running', chat_id=7, task_attempt=2, child_drive_root=str(child))
    pipeline.write_task_result(child, tid, 'completed', chat_id=7, task_attempt=1 if stale else 2,
                               _terminal_observed=True, result='Attempt answer.', terminal_origin='model_final',
                               loop_outcome={'usage': {'total_rounds': 3}})
    pipeline._record_task_facts(SimpleNamespace(drive_root=parent), {'id': tid, 'chat_id': 7, '_attempt': 2}, {},
                                {'tool_calls': []}, parent / 'logs')

    def seen():
        row = read_result(parent, tid)
        response = asyncio.run(make_chat_history_endpoint(parent)(SimpleNamespace(query_params={'chat_id': '7'})))
        assert response.status_code == 200
        [card] = [r for r in json.loads(response.body)['messages'] if r.get('system_type') == 'task_summary']
        usage = (row.get('loop_outcome') or {}).get('usage') or {}
        return {'stored': {'attempt': row.get('task_attempt'), 'status': row['status'], 'reason': row.get('reason_code'),
                           'origin': row.get('terminal_origin'), 'rounds': usage.get('total_rounds')},
                'card': {'attempt': card.get('task_attempt'), 'status': card.get('task_terminal_status'),
                         'reason': card.get('reason_code') or None, 'origin': card.get('terminal_origin'),
                         'rounds': card.get('rounds')}}

    if stale:
        assert seen() == {'stored': own, 'card': own}  # a real setup: attempt 2 holds its own non-empty ending
    else:
        assert seen()['stored']['status'] == 'running'
    headless.copy_child_task_result(parent, {'id': tid, 'drive_root': str(child), '_attempt': 1 if stale else 2})
    ending = own if stale else {'attempt': 2, 'status': 'completed', 'reason': None, 'origin': 'model_final', 'rounds': 3}
    assert seen() == {'stored': ending, 'card': ending}


def test_a_same_attempt_conflicting_failure_card_shows_one_ending_not_a_mixture(tmp_path):
    # Same physical attempt 2 on both drives, both "failed": the canonical row holds the host's provider
    # failure with zero rounds and its own reason and origin, the worker's drive holds a different failed
    # end — another reason, counter and closing sentence. The reloaded card must name one ending only:
    # the selected host counters and provenance together, never the worker's rounds or origin beside the
    # host's words (the "three rounds instead of zero" class the reviewers found in a retried attempt's
    # history, now within one status).
    child = tmp_path / "actor"
    pipeline.write_task_result(tmp_path, "split-root", "failed", chat_id=7, task_attempt=2,
                               child_drive_root=str(child), _terminal_observed=True,
                               result="Attempt 2 stopped by the host.", reason_code="provider_unavailable",
                               terminal_origin="host_notice", loop_outcome={"usage": {"total_rounds": 0}})
    pipeline.write_task_result(child, "split-root", "failed", chat_id=7, task_attempt=2,
                               _terminal_observed=True, result="Attempt 2 stopped by the model.",
                               reason_code="model_refusal", terminal_origin="model_final",
                               wake_summary="the model stopped itself",
                               loop_outcome={"usage": {"total_rounds": 2}})
    pipeline._record_task_facts(SimpleNamespace(drive_root=tmp_path),
                                {"id": "split-root", "chat_id": 7, "_attempt": 2}, {}, {"tool_calls": []},
                                tmp_path / "logs")
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"chat_id": "7"})))
    assert response.status_code == 200
    [row] = [r for r in json.loads(response.body)["messages"] if r.get("system_type") == "task_summary"]
    assert row["task_attempt"] == 2 and row["task_terminal_status"] == "failed"
    assert (row["rounds"], row.get("terminal_origin"), row.get("reason_code")) == (
        0, "host_notice", "provider_unavailable")
    assert "the model stopped itself" not in json.dumps(row)  # the loser's closing sentence never reaches the card


def test_a_word_only_same_attempt_conflict_shows_one_ending_cold_and_after_copy_back(tmp_path):
    # The same class at its narrowest: ONE physical attempt, both drives "failed", the SAME reason code
    # and the SAME terminal origin, and the endings differ only in the answer's WORDS, beside the losing
    # copy's own counters and closing sentence. A display view carries no words and only the bit that one
    # exists, so the reloaded card lent the loser's rounds and sentence where the whole-document read and
    # the copy-back both refuse them (R15's test above does not catch it: there the visible fields clash).
    from ouroboros import headless
    from ouroboros.task_reads import TaskReads
    from tests._store_shared import read_result

    parent = tmp_path / "data"
    parent.mkdir()
    tid, child = "word-only", headless.prepare_task_drive(parent, "word-only", "empty")
    host_words, model_words = "Attempt 2 stopped by the host.", "Attempt 2 stopped by the model."
    pipeline.write_task_result(parent, tid, "failed", chat_id=7, task_attempt=2, child_drive_root=str(child),
                               _terminal_observed=True, result=host_words, reason_code="provider_unavailable",
                               terminal_origin="host_notice", loop_outcome={"usage": {"total_rounds": 0}})
    pipeline.write_task_result(child, tid, "failed", chat_id=7, task_attempt=2, _terminal_observed=True,
                               result=model_words, reason_code="provider_unavailable",
                               terminal_origin="host_notice", wake_summary="the model stopped itself",
                               loop_outcome={"usage": {"total_rounds": 2}})
    pipeline._record_task_facts(SimpleNamespace(drive_root=parent), {"id": tid, "chat_id": 7, "_attempt": 2},
                                {}, {"tool_calls": []}, parent / "logs")

    def card():
        response = asyncio.run(make_chat_history_endpoint(parent)(SimpleNamespace(query_params={"chat_id": "7"})))
        assert response.status_code == 200
        [row] = [r for r in json.loads(response.body)["messages"] if r.get("system_type") == "task_summary"]
        return row

    # The page's own batch reader agrees with the whole document it stands for, and stays wordless.
    overlaid = TaskReads(parent).effective(tid)
    assert "result" not in overlaid and overlaid["_result_text_present"] is True  # the winner's answer exists
    assert ((overlaid.get("loop_outcome") or {}).get("usage") or {}).get("total_rounds") == 0
    assert not overlaid.get("wake_summary") and model_words not in json.dumps(overlaid)

    before = card()
    assert (before["rounds"], before.get("terminal_origin"), before.get("reason_code")) == (
        0, "host_notice", "provider_unavailable")
    assert not before.get("wake_summary") and model_words not in json.dumps(before)

    # One ending before the copy-back and the same one after it, in the row the card stands for.
    headless.copy_child_task_result(parent, {"id": tid, "drive_root": str(child), "_attempt": 2})
    stored = read_result(parent, tid)
    assert stored["status"] == "failed" and stored.get("result") == host_words
    assert stored.get("wake_summary") in (None, "")
    assert ((stored.get("loop_outcome") or {}).get("usage") or {}).get("total_rounds") == 0
    after = card()
    assert (after["rounds"], after.get("terminal_origin")) == (0, "host_notice")
    assert not after.get("wake_summary")


@pytest.mark.parametrize("case,words,paid,rounds", [
    ("contrary", "differing", 2, 0),  # the host's own ending, exactly as the whole documents settle it
    ("agreeing", "same", 2, 2),  # no contradiction: the copy still lends its counters and its sentence
    ("completed", "differing", 0, 2),  # completed/completed is enrichment, and no page reads a body for it
    ("foreign-attempt", "differing", 0, 0),  # another attempt's copy lends no ending at all
    ("canonical-wordless", "replica-only", 0, 2),  # a row holding no answer is contradicted by none
])
def test_the_compared_words_are_read_back_only_where_a_view_could_still_contradict(tmp_path, monkeypatch,
                                                                                    case, words, paid, rounds):
    # The other side of the same boundary, with the count said exactly: the narrow read of one field is
    # paid only for the pair whose views could still disagree, and every reading of it carries the
    # revision the compared view was taken at.
    from ouroboros import headless, task_store
    from ouroboros.task_reads import TaskReads

    calls = []
    real = task_store.result_text_at

    def counted(root, task_id, write_seq):
        calls.append((str(root), str(task_id), write_seq))
        return real(root, task_id, write_seq)

    monkeypatch.setattr(task_store, "result_text_at", counted)
    parent = tmp_path / "data"
    parent.mkdir()
    tid, child = "views", headless.prepare_task_drive(parent, "views", "empty")
    host_status = "completed" if case == "completed" else "failed"
    host_words = "" if case == "canonical-wordless" else "The host's own answer."
    replica_words = "The copy's own answer." if words == "differing" else host_words
    pipeline.write_task_result(parent, tid, host_status, chat_id=7, task_attempt=2, child_drive_root=str(child),
                               _terminal_observed=True, result=host_words, reason_code="provider_unavailable",
                               terminal_origin="host_notice",
                               loop_outcome={"usage": {"total_rounds": 3 if host_status == "completed" else 0}})
    pipeline.write_task_result(child, tid, host_status, chat_id=7,
                               task_attempt=1 if case == "foreign-attempt" else 2, _terminal_observed=True,
                               result=replica_words, reason_code="provider_unavailable",
                               terminal_origin="host_notice", wake_summary="the copy's own closing sentence",
                               loop_outcome={"usage": {"total_rounds": 2}})

    row = TaskReads(parent).effective(tid)
    assert "result" not in row  # the compared words never enter the lightweight page
    assert ((row.get("loop_outcome") or {}).get("usage") or {}).get("total_rounds") == rounds
    assert bool(row.get("wake_summary")) is (rounds == 2)
    assert {tid for _root, tid, _seq in calls} <= {tid}
    assert all(type(seq) is int for _root, _tid, seq in calls)  # pinned to the view's own revision
    assert len(calls) == paid
    if paid:
        assert len({root for root, _tid, _seq in calls}) == paid  # once per row, not once per page step


def test_answer_words_a_page_cannot_read_back_keep_its_own_ending_rather_than_lend_one(tmp_path, monkeypatch):
    # The gate said these words could still settle the pair and the store cannot supply them (a row that
    # moved since its view, a body it cannot decode). Unknown is never agreement, and it is no licence to
    # lend either: the card keeps its own row's counters, sentence and answer existence.
    from ouroboros import headless, task_store
    from ouroboros.task_reads import TaskReads

    parent = tmp_path / "data"
    parent.mkdir()
    tid, child = "unreadable", headless.prepare_task_drive(parent, "unreadable", "empty")
    pipeline.write_task_result(parent, tid, "failed", chat_id=7, task_attempt=2, child_drive_root=str(child),
                               _terminal_observed=True, result="Attempt 2 stopped by the host.",
                               reason_code="provider_unavailable", terminal_origin="host_notice",
                               loop_outcome={"usage": {"total_rounds": 0}})
    pipeline.write_task_result(child, tid, "failed", chat_id=7, task_attempt=2, _terminal_observed=True,
                               result="Attempt 2 stopped by the model.", reason_code="provider_unavailable",
                               terminal_origin="host_notice", wake_summary="the model stopped itself",
                               loop_outcome={"usage": {"total_rounds": 2}})
    tried = []
    monkeypatch.setattr(task_store, "result_text_at",
                        lambda root, task_id, write_seq: tried.append((str(task_id), write_seq)) or None)

    row = TaskReads(parent).effective(tid)
    assert len(tried) == 2 and all(seq is not None for _tid, seq in tried)
    assert ((row.get("loop_outcome") or {}).get("usage") or {}).get("total_rounds") == 0
    assert not row.get("wake_summary") and row["_result_text_present"] is True
