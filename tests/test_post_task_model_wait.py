"""Post-task live custody without rerunning answered work or completed stages."""

from copy import deepcopy
import queue
import threading
import time
from types import SimpleNamespace

import pytest
from tests._dialogue import chat_rows

from ouroboros import agent_task_pipeline as pipeline, config, model_wait
from ouroboros import llm_claudexor as transport
from ouroboros.post_task_checkpoint import post_task_model_wait, post_task_model_waits
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_llm_claudexor import Gateway, MODEL, result, ledger


def until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("controlled post-task state did not arrive")


@pytest.fixture
def phase(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    # Foreground pytest may inherit the calling agent worker's environment;
    # these tests model detached post-work except where a test opts into a pool worker.
    monkeypatch.delenv("OUROBOROS_IN_WORKER", raising=False)
    monkeypatch.setattr(config, "CLAUDEXOR_MODEL_POLL_INTERVAL_SEC", 0.005)
    monkeypatch.setattr(config, "NETWORK_WAIT_BACKOFF_START_SEC", 0.005)
    monkeypatch.setattr(config, "NETWORK_WAIT_BACKOFF_MAX_SEC", 0.01)
    task = {"id": "post-owner", "root_task_id": "post-owner", "chat_id": 1, "type": "task", "text": "Already answered", "drive_root": str(root)}
    write_task_result(root, task["id"], "completed", result="Already answered",
                      root_phase_checkpoint={"post_task_synthesis": "pending_once"})
    env = SimpleNamespace(drive_root=root, repo_dir=tmp_path, drive_path=lambda rel: root / rel)
    events, done, ready = queue.Queue(), threading.Event(), threading.Event()
    engine = Gateway([result(outcome="failed", problem={"code": "subscription_window_exhausted", "message": "quota"}), result()],
                     ["not_started", "response_received"])
    monkeypatch.setattr(transport, "ensure_owned_gateway", lambda **_kw: engine)
    monkeypatch.setattr(transport, "model_sources", lambda **_kwargs: {"sources": [{"id": "codex", "credentialHarness": "fixture"}]})
    monkeypatch.setattr(transport, "model_catalog", lambda _source, account=None, *, requested_model=None: {
        "source": "codex", "credentialProfileId": account or "account-a",
        "models": [{"id": "exact-model"}] if ready.is_set() else []})
    stages = []
    monkeypatch.setattr(pipeline, "_run_memory_fallback_draft", lambda *a: stages.append("draft"))
    monkeypatch.setattr(pipeline, "_record_task_facts", lambda *a, **k: stages.append("facts"))
    from ouroboros import post_task_evolution
    real_promote = post_task_evolution.maybe_promote

    def promote(_env, _task, _sealed, llm):
        # The late phase's last paid stage, buying one model call after the completed prior stages.
        stages.append("promotion")
        llm.chat([{"role": "user", "content": "choose once after completed prior stages"}], MODEL, model_role="light")
        return None

    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote", promote)
    original = pipeline._set_root_post_task_checkpoint

    def checkpoint(*args, **kwargs):
        saved = original(*args, **kwargs)
        if args[2] in {"completed", "degraded"}:
            done.set()
        return saved

    monkeypatch.setattr(pipeline, "_set_root_post_task_checkpoint", checkpoint)
    yield SimpleNamespace(root=root, env=env, task=task, events=events, engine=engine,
                          ready=ready, done=done, stages=stages, real_promote=real_promote)
    task["_skip_post_task_synthesis"] = True
    ready.set()
    done.wait(5)
    until(lambda: not post_task_model_waits(root))


def launch(f):
    return pipeline._run_post_task_processing_async(f.env, f.task, {"rounds": 3}, {}, f.root / "logs", event_queue=f.events)


def active(f):
    owner = post_task_model_wait(f.root, f.task["id"])
    return owner if owner and any(row["state"] == "waiting" and row.get("credential_harness") for row in owner.waits.values()) else None


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_history_keeps_answered_post_phase_open_but_never_revives_cancelled(tmp_path, status):
    from ouroboros.gateway.history import _annotate_terminal_task_truth

    result = {"status": status, "root_phase_checkpoint": {"post_task_synthesis": "running"}}
    messages = [{"task_id": "root", "role": "system", "system_type": "task_model_wait"}]
    _annotate_terminal_task_truth(messages, tmp_path, {"root": result})
    assert (messages[0].get("task_phase") == "finalizing") == (status != "cancelled")
    assert messages[0].get("task_terminal_status") == ("cancelled" if status == "cancelled" else None)
    assert result["status"] == status


@pytest.mark.parametrize("main_status", ["completed", "failed"])
def test_detached_parent_returns_and_post_wait_keeps_override_and_prior_stages(phase, main_status):
    f = phase
    if main_status == "failed":
        f.task.update(id="post-failed", root_task_id="post-failed")
        pipeline._store_task_result(f.env, f.task, "Main execution failed", {}, {},
            loop_outcome={"outcome_axes": {"execution": {"status": "failed"}}})
    assert pipeline._is_root_post_task(f.task)
    initial = load_task_result(f.root, f.task["id"])
    assert initial["status"] == main_status
    assert initial["root_phase_checkpoint"]["post_task_synthesis"] == "pending_once"
    with model_wait.task_model_wait_scope(task=f.task, drive_root=f.root, event_queue=f.events,
                                          worker_slot_held=False) as parent:
        parent.overrides["light"] = {"model": MODEL, "use_local": False, "model_account_override": "original-choice"}
        assert launch(f) is None
    assert parent.closed
    until(lambda: active(f))
    owner = active(f)
    assert owner is not parent and not owner.closed and owner.worker_slot_held is False
    assert f.stages == ["facts", "draft", "promotion"] and not f.done.is_set()
    assert load_task_result(f.root, f.task["id"])["root_phase_checkpoint"]["post_task_synthesis"] == "running"
    assert f.engine.uploads[0][0]["account"] == {"mode": "pin", "profileId": "original-choice"}
    f.ready.set()
    assert f.done.wait(5)
    until(lambda: post_task_model_wait(f.root, f.task["id"]) is None)
    assert f.stages == ["facts", "draft", "promotion"]
    assert len(f.engine.creates) == 2 and f.engine.uploads[0][0]["messages"] == f.engine.uploads[1][0]["messages"]
    until(lambda: owner.closed)
    assert load_task_result(f.root, f.task["id"])["status"] == main_status


def test_detached_decision_mailbox_and_activity_remain_live_after_task_done(phase):
    from ouroboros.gateway import task_model_wait as gateway
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from supervisor.terminal_delivery import cleanup_settled_owner_mailbox
    from ouroboros.owner_mailbox import write_owner_message, _mailbox_path
    from supervisor.task_model_wait import handle_task_model_wait
    from ouroboros.utils import append_jsonl

    f = phase
    launch(f)
    until(lambda: active(f))
    owner = active(f)
    row = deepcopy(next(row for row in owner.waits.values() if row["state"] == "waiting"))
    action = {"request_id": "post-switch", "decision_id": f"model_wait:{f.task['id']}:{row['wait_id']}",
              "revision": row["revision"], "action": "switch", "model": MODEL,
              "credential_profile_id": "replacement", "use_local": False, "persist_role": False}
    write_owner_message(f.root, "retained owner message", f.task["id"], msg_id="keep")
    cleanup_settled_owner_mailbox(f.root, f.task["id"], f.task)
    assert _mailbox_path(f.root, f.task["id"]).exists()
    activities = [row for row in _chat_activities_snapshot_safe(f.root) if row["activity_id"] == f.task["id"]]
    assert len(activities) == 1 and activities[0]["phase"] == "finalizing" and activities[0]["model_waits"]
    forwarded = []
    ctx = SimpleNamespace(RUNNING={}, DRIVE_ROOT=f.root, append_jsonl=append_jsonl,
                          bridge=SimpleNamespace(push_log=forwarded.append))
    handle_task_model_wait({"type": "task_model_wait", "task_id": f.task["id"], **row}, ctx)
    assert len(forwarded) == 1 and forwarded[0]["chat_id"] == 1
    response = gateway._decide(f.root, action)
    assert response.status_code == 202
    assert f.done.wait(5)
    until(lambda: not _mailbox_path(f.root, f.task["id"]).exists())
    assert f.engine.uploads[-1][0]["account"] == {"mode": "pin", "profileId": "replacement"}
    assert gateway._decide(f.root, action).status_code == 409
    handle_task_model_wait({"type": "task_model_wait", "task_id": f.task["id"], **row}, ctx)
    assert len(forwarded) == 1  # An ended post owner cannot be resurrected.


@pytest.mark.parametrize("cause", ["budget", "deadline", "unknown", "ordinary"])
def test_paid_stage_interruption_closes_checkpoint_without_buying_following_stages(phase, monkeypatch, cause):
    from ouroboros.usage_accounting import BudgetExceeded

    f = phase
    failures = {
        "budget": BudgetExceeded("root wallet spent"),
        "deadline": model_wait.ModelWaitInterrupted("deadline"),
        "unknown": transport.ClaudexorModelError({"code": "model_outcome_unknown", "message": "unknown"}, unknown=True),
        "ordinary": RuntimeError("one stage failed"),
    }
    def draft(*_args):
        f.stages.append("draft")
        raise failures[cause]
    monkeypatch.setattr(pipeline, "_run_memory_fallback_draft", draft)
    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote", lambda *a: f.stages.append("promotion"))
    launch(f)
    assert f.done.wait(5)
    stored = load_task_result(f.root, f.task["id"])
    checkpoint = stored["root_phase_checkpoint"]
    assert checkpoint["post_task_synthesis"] == "degraded"
    if cause == "ordinary":
        assert f.stages == ["facts", "draft", "promotion"]
    else:
        assert f.stages == ["facts", "draft"]
        assert checkpoint["post_task_stop_reason"].startswith({
            "budget": "budget_exhausted", "deadline": "deadline", "unknown": "provider_outcome_unknown"}[cause])
        assert checkpoint["post_task_stop_reason"].endswith("skipped=promotion")
    assert not f.engine.creates  # no provider send after the first interrupted stage


@pytest.mark.parametrize("cause", ["budget", "api_unknown", "ordinary"])
def test_real_light_error_controls_remaining_post_task_stages(
    phase, monkeypatch, cause,
):
    """Exercise the real shared Light catch and the real fallback stage adapter, not a throwing stage
    stub (the retired scratchpad pass used to carry this; #918). The draft's unit selection, which
    needs an activated chronicle, stands in as one Light call."""
    from ouroboros import consolidator, context_fit, memory_fallback, post_task_synthesis
    from ouroboros.capability_evidence import CapabilityEvidence
    from ouroboros.usage_accounting import BudgetExceeded

    f = phase
    monkeypatch.setattr(consolidator, "_consolidation_route", lambda: ("test/model", False))
    monkeypatch.setattr(context_fit, "resolve_context_fit_route", lambda task, *, allow_fetch: (
        {"model": task["model"], "provider": "openrouter"},
        CapabilityEvidence(100_000, "confirmed", "test", "route-test",
                           model=task["model"], provider="openrouter")))
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_, **_kw: 1.0)
    error = BudgetExceeded("root wallet spent") if cause == "budget" else RuntimeError("provider failed")
    if cause == "api_unknown":
        error.physical_attempt_capture = SimpleNamespace(state="unresolved")

    class FailingLight:
        def chat(self, **_kwargs):
            f.stages.append("draft-model")
            raise error

    def run_fallback_draft(_env, _task, llm, _drive_logs, _trace):
        _text, usage = consolidator._call_consolidation_llm(llm, "captured draft", memory_fallback.LABEL)
        return memory_fallback.FallbackRun("failed", usage=usage, errors=usage["_consolidation_errors"])

    monkeypatch.setattr("ouroboros.llm.LLMClient", FailingLight)
    monkeypatch.setattr(memory_fallback, "run_fallback_draft", run_fallback_draft)
    monkeypatch.setattr(pipeline, "_run_memory_fallback_draft", post_task_synthesis._run_memory_fallback_draft)
    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote", lambda *_args: f.stages.append("promotion"))

    launch(f)
    assert f.done.wait(5)
    checkpoint = load_task_result(f.root, f.task["id"])["root_phase_checkpoint"]
    if cause == "ordinary":
        # TZ-2 C3: an ordinary failure is isolated to its stage — later stages still
        # run — but a stage that lost work is unfinished, so the checkpoint is
        # degraded, never completed, and nothing was skipped.
        assert checkpoint["post_task_synthesis"] == "degraded"
        assert not checkpoint.get("post_task_stop_reason")
        assert f.stages == ["facts", "draft-model", "promotion"]
    else:
        assert checkpoint["post_task_synthesis"] == "degraded"
        assert checkpoint["post_task_stop_reason"].startswith(
            "budget_exhausted:" if cause == "budget" else "provider_outcome_unknown:")
        assert f.stages == ["facts", "draft-model"]
        assert checkpoint["post_task_stop_reason"].endswith("skipped=promotion")


def test_budget_refusal_inside_promotion_is_never_swallowed_into_completed(phase, monkeypatch):
    """TZ-2 C3: `propagate_model_error` re-raises only control/unknown facts, so a
    `BudgetExceeded` raised inside a stage adapter's own catch (promotion,
    consolidation setup) used to be logged and the checkpoint written
    `completed`. The shared `propagate_paid_interruption` lets the wallet stop the
    remaining paid post-work like the other two interruptions."""
    from ouroboros.usage_accounting import BudgetExceeded

    f = phase

    def refuse(*_args):
        f.stages.append("promotion-model")
        raise BudgetExceeded("root wallet spent")

    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote", refuse)
    f.ready.set()
    launch(f)
    assert f.done.wait(5)
    checkpoint = load_task_result(f.root, f.task["id"])["root_phase_checkpoint"]
    assert checkpoint["post_task_synthesis"] == "degraded"
    assert checkpoint["post_task_stop_reason"] == "budget_exhausted:skipped="
    assert f.stages[-2:] == ["draft", "promotion-model"]


def test_returned_paid_error_stops_post_task_after_its_own_stage(phase, monkeypatch):
    """A returned typed failure from the fallback draft (formerly scratchpad compression, retired by
    #918) also stops the later paid stage."""
    from ouroboros import memory_fallback, post_task_synthesis

    f = phase
    error = {"kind": "provider_outcome_unknown"}
    monkeypatch.setattr(memory_fallback, "run_fallback_draft", lambda *_: memory_fallback.FallbackRun(
        "failed", usage={}, errors=[error]))
    monkeypatch.setattr(pipeline, "_run_memory_fallback_draft",
                        post_task_synthesis._run_memory_fallback_draft)
    launch(f)
    assert f.done.wait(5)
    checkpoint = load_task_result(f.root, f.task["id"])["root_phase_checkpoint"]
    assert checkpoint["post_task_synthesis"] == "degraded"
    assert checkpoint["post_task_stop_reason"] == "provider_outcome_unknown:skipped=promotion"
    assert "promotion" not in f.stages


@pytest.mark.parametrize("unknown", [False, True])
def test_stop_or_unknown_never_marks_post_work_completed(phase, unknown):
    f = phase
    if unknown:
        f.engine.results[0]["outcome"] = "unknown"
        f.engine.dispatch[0] = "unknown"
    launch(f)
    if not unknown:
        until(lambda: active(f))
        f.task["_skip_post_task_synthesis"] = True
    assert f.done.wait(5)
    assert load_task_result(f.root, f.task["id"])["root_phase_checkpoint"]["post_task_synthesis"] == "degraded"
    assert len(f.engine.creates) == 1 and f.stages == ["facts", "draft", "promotion"]
    assert ledger(f.root)[-1]["state"] == ("unresolved" if unknown else "released")


def test_blocking_post_work_exempts_solve_ceiling_only_within_its_scope(phase, monkeypatch):
    f = phase
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 1)
    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote", lambda *a: None)
    observed = []
    def check(*_args):
        observed.append(model_wait.current_model_wait().control_reason())
    monkeypatch.setattr(pipeline, "_run_memory_fallback_draft", check)
    with model_wait.task_model_wait_scope(task=f.task, drive_root=f.root, event_queue=f.events,
                                          worker_slot_held=True) as owner:
        monkeypatch.setattr(owner, "executed_seconds", lambda **_kwargs: 100)
        assert owner.control_reason() == "absolute_ceiling"
        pipeline._run_post_task_processing_async(f.env, f.task, {}, {}, f.root / "logs", blocking=True)
        assert owner.control_reason() == "absolute_ceiling"  # restored for the solve owner
    assert observed == [None]
    assert load_task_result(f.root, f.task["id"])["root_phase_checkpoint"]["post_task_synthesis"] == "completed"


def test_pooled_post_work_holds_return_but_delivers_answer_early_once(phase, monkeypatch):
    from ouroboros.utils import in_worker_process

    f = phase
    monkeypatch.setenv("OUROBOROS_IN_WORKER", "1")
    assert in_worker_process()
    pending = [{"type": "send_message", "task_id": f.task["id"], "chat_id": 1, "text": "Already answered"},
               {"type": "task_done", "task_id": f.task["id"], "status": "completed"}]
    returned = threading.Event()

    def run():
        with model_wait.task_model_wait_scope(task=f.task, drive_root=f.root, event_queue=f.events,
                                              worker_slot_held=True):
            pipeline._dispatch_root_post_task(f.env, f.task, "Already answered", f.events, pending,
                {"rounds": 3}, {}, f.root / "logs", budget_drive_root="", split_drive=False,
                project_scoped=False, project_task=False, parent_env=None, parent_task=None)
        returned.set()

    thread = threading.Thread(target=run)
    thread.start()
    until(lambda: len(f.engine.creates) == 1)
    live = list(f.events.queue)
    answers = [row for row in live if row["type"] == "send_message"]
    assert len(answers) == 1 and answers[0]["text"] == "Already answered"
    assert not returned.is_set() and not any(row["type"] == "task_done" for row in live)
    assert answers[0]["delivery_id"] == pending[0]["delivery_id"]
    assert post_task_model_wait(f.root, f.task["id"]) is None  # no cross-process owner registry
    f.ready.set()
    thread.join(5)
    assert returned.is_set() and not thread.is_alive() and len(f.engine.creates) == 2


def test_outer_paid_stage_cannot_swallow_typed_unknown(tmp_path, monkeypatch):
    from ouroboros import memory_fallback, post_task_synthesis as synthesis

    error = transport.ClaudexorModelError({"code": "unknown", "message": "paid outcome unknown"}, unknown=True)
    def fail(*args, **kwargs):
        raise error
    env = SimpleNamespace(drive_root=tmp_path, drive_path=lambda rel: tmp_path / rel)
    monkeypatch.setattr(memory_fallback, "run_fallback_draft", fail)
    with pytest.raises(transport.ClaudexorModelError) as raised:
        synthesis._run_memory_fallback_draft(env, {"id": "task"}, None, tmp_path / "logs", {})
    assert raised.value is error


def test_promotion_outer_wrapper_propagates_typed_control_and_ordinary_failure_alike(tmp_path, monkeypatch):
    """A typed control keeps propagating; an ordinary chooser failure now reaches the
    stage too instead of returning the same None a genuine "no promotion" returns."""
    from ouroboros import post_task_evolution as promotion

    monkeypatch.setattr(config, "get_post_task_evolution_enabled", lambda: True)
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "advanced")
    monkeypatch.setattr(config, "get_post_task_evolution_cadence", lambda: "llm")
    monkeypatch.setattr(promotion, "_eligible", lambda *_: True)
    monkeypatch.setattr(promotion, "_is_canonical_run", lambda *_: True)
    def fail(*args, **kwargs):
        raise model_wait.ModelWaitInterrupted("owner_stopped", role="main")
    monkeypatch.setattr(promotion, "_decide_promotion", fail)
    env = SimpleNamespace(drive_root=tmp_path)
    with pytest.raises(model_wait.ModelWaitInterrupted):
        promotion.maybe_promote(env, {"id": "task"}, None)
    monkeypatch.setattr(promotion, "_decide_promotion", lambda *a, **k: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        promotion.maybe_promote(env, {"id": "task"}, None)


def _controlled_worker(input_queue, output_queue, data_root, repo_root, resume):
    """Run the actual worker dequeue loop with controlled post-task cognition."""
    from contextlib import ExitStack
    from pathlib import Path
    from unittest.mock import patch
    from ouroboros.usage_accounting import UsageScope, usage_scope
    from supervisor.worker_process import worker_main

    root = Path(data_root)
    env = SimpleNamespace(drive_root=root, repo_dir=Path(repo_root), drive_path=lambda rel: root / rel)
    engine = Gateway([result(outcome="failed", problem={"code": "subscription_window_exhausted", "message": "fixture"}), result()],
                     ["not_started", "response_received"])
    def promote(_env, _task, _sealed, client):
        output_queue.put({"type": "fixture_promotion_started"})
        client.chat([{"role": "user", "content": "one chooser call"}], MODEL, model_role="light")
        return None
    class Agent:
        def handle_task(self, task):
            output_queue.put({"type": "fixture_task_started", "task_id": task["id"]})
            if task["id"] == "second":
                return []
            write_task_result(root, task["id"], "completed", result="answer",
                              root_phase_checkpoint={"post_task_synthesis": "pending_once"})
            pending = [{"type": "send_message", "task_id": task["id"], "chat_id": 1, "text": "answer"},
                       {"type": "task_done", "task_id": task["id"], "status": "completed"}]
            with usage_scope(UsageScope(drive_root=root, task_id=task["id"], root_task_id=task["id"])), model_wait.task_model_wait_scope(
                    task=task, drive_root=root, event_queue=output_queue, worker_slot_held=True):
                pipeline._dispatch_root_post_task(env, task, "answer", output_queue, pending, {"rounds": 3}, {}, root / "logs",
                    budget_drive_root="", split_drive=False, project_scoped=False, project_task=False, parent_env=None, parent_task=None)
            output_queue.put({"type": "fixture_generation_count", "count": len(engine.creates)})
            return pending
    with ExitStack() as stack:
        replacements = {
            "ouroboros.agent.make_agent": lambda **kwargs: Agent(),
            "ouroboros.extension_loader.reload_all": lambda *args, **kwargs: None,
            "supervisor.worker_process._prepare_worker_task_runtime": lambda: None,
            "supervisor.worker_process._adopt_published_extensions": lambda *_: None,
            "ouroboros.llm_claudexor.ensure_owned_gateway": lambda **_kw: engine,
            "ouroboros.llm_claudexor.model_sources": lambda **_kwargs: {"sources": [{"id": "codex", "credentialHarness": "fixture"}]},
            "ouroboros.llm_claudexor.model_catalog": lambda source, account=None, **kwargs: {
                "source": source, "credentialProfileId": account or "account-a", "models": [{"id": "exact-model"}] if resume.is_set() else []},
            "ouroboros.agent_task_pipeline._run_memory_fallback_draft": lambda *_: None,
            "ouroboros.agent_task_pipeline._record_task_facts": lambda *args, **kwargs: None,
            "ouroboros.post_task_evolution.maybe_promote": promote,
        }
        for name, value in replacements.items():
            stack.enter_context(patch(name, value))
        config.CLAUDEXOR_MODEL_POLL_INTERVAL_SEC = 0.01
        config.NETWORK_WAIT_BACKOFF_START_SEC = 0.01
        config.NETWORK_WAIT_BACKOFF_MAX_SEC = 0.01
        worker_main(0, input_queue, output_queue, repo_root, data_root)


@pytest.mark.serial
def test_real_pooled_process_does_not_dequeue_next_task_during_post_wait(tmp_path, monkeypatch):
    import multiprocessing
    from pathlib import Path

    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setenv("OUROBOROS_APP_ROOT", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    context = multiprocessing.get_context("spawn")
    incoming, outgoing, resume = context.Queue(), context.Queue(), context.Event()
    proc = context.Process(target=_controlled_worker, args=(incoming, outgoing, str(root), str(Path(__file__).resolve().parents[1]), resume))
    events = []
    def receive_until(predicate):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            events.append(outgoing.get(timeout=max(0.01, deadline - time.monotonic())))
            if predicate(events[-1]):
                return
        raise AssertionError("worker fixture did not reach requested state")
    proc.start()
    try:
        incoming.put({"id": "first", "type": "task", "chat_id": 1, "drive_root": str(root)})
        incoming.put({"id": "second", "type": "task", "chat_id": 1, "drive_root": str(root)})
        receive_until(lambda event: event.get("type") == "task_model_wait" and event.get("state") == "waiting")
        assert events[-1]["worker_slot_held"] is True
        assert any(event.get("type") == "send_message" for event in events)
        assert not any(event.get("type") == "task_done" or event.get("task_id") == "second" for event in events)
        assert load_task_result(root, "first")["root_phase_checkpoint"]["post_task_synthesis"] == "running"
        resume.set()
        receive_until(lambda event: event.get("type") == "fixture_task_started" and event.get("task_id") == "second")
        assert sum(event.get("type") == "fixture_promotion_started" for event in events) == 1
        assert [event["count"] for event in events if event.get("type") == "fixture_generation_count"] == [2]
        assert any(event.get("type") == "task_done" and event.get("task_id") == "first" for event in events)
        answers = [event for event in events if event.get("type") == "send_message"]
        assert len(answers) == 2 and len({event["delivery_id"] for event in answers}) == 1
        incoming.put(None)
        proc.join(10)
        assert proc.exitcode == 0
    finally:
        resume.set()
        if proc.is_alive():
            incoming.put(None)
            proc.join(5)
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
        incoming.close()
        outgoing.close()
        incoming.join_thread()
        outgoing.join_thread()


@pytest.mark.serial
@pytest.mark.parametrize("first_settled", ["attachments", "post_work"])
def test_mailbox_survives_until_both_attachment_and_post_work_custody_settle(tmp_path, first_settled):
    from ouroboros.owner_mailbox import _mailbox_path, write_owner_message
    from supervisor.terminal_delivery import cleanup_settled_owner_mailbox

    task = {"id": "retained-input", "drive_root": str(tmp_path)}
    pending = [{"kind": "task_attachment", "source_task_id": task["id"]}]
    post = "running"
    write_owner_message(tmp_path, "Accepted owner input", task["id"], msg_id="accepted-owner")
    path = _mailbox_path(tmp_path, task["id"])
    original = path.read_bytes()
    for step in range(3):
        write_task_result(tmp_path, task["id"], "completed",
                          child_ref_promotion={"pending_refs": pending},
                          root_phase_checkpoint={"post_task_synthesis": post})
        cleanup_settled_owner_mailbox(tmp_path, task["id"], task)
        if step < 2:
            assert path.read_bytes() == original
        else:
            assert not path.exists()
        if step == 0 and first_settled == "attachments":
            pending = []
        elif step == 0:
            post = "completed"
        else:
            pending, post = [], "completed"


def test_budget_refusal_inside_the_promotion_chooser_degrades_the_checkpoint(phase, monkeypatch):
    """Finding 2: `_decide_promotion` caught the wallet's BudgetExceeded with
    `propagate_model_error` and returned None, so `maybe_promote` read as "no
    promotion" and the coordinator wrote `completed`. Raised by the chooser's own
    LLM call through the REAL adapters (`maybe_promote`, the promotion stage), the
    refusal reaches the coordinator: degraded, the typed stop reason, and no
    global callback afterwards."""
    from ouroboros import llm_observability, post_task_evolution as promotion
    from ouroboros.usage_accounting import BudgetExceeded

    f = phase
    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote", f.real_promote)
    monkeypatch.setattr(config, "get_post_task_evolution_enabled", lambda: True)
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "advanced")
    monkeypatch.setattr(config, "get_post_task_evolution_cadence", lambda: "llm")
    monkeypatch.setattr(promotion, "_eligible", lambda *_: True)
    monkeypatch.setattr(promotion, "_is_canonical_run", lambda *_: True)
    monkeypatch.setattr(promotion, "_closed_objectives_digest", lambda *_: "")

    def refuse(*_args, **_kwargs):
        f.stages.append("chooser")
        raise BudgetExceeded("root wallet spent")

    monkeypatch.setattr(llm_observability, "chat_observed", refuse)
    callbacks = []
    f.ready.set()
    pipeline._run_post_task_processing_async(
        f.env, f.task, {"rounds": 3}, {}, f.root / "logs", event_queue=f.events,
        on_reflection=lambda *args: callbacks.append(args))
    assert f.done.wait(5)
    checkpoint = load_task_result(f.root, f.task["id"])["root_phase_checkpoint"]
    assert checkpoint["post_task_synthesis"] == "degraded"
    assert checkpoint["post_task_stop_reason"] == "budget_exhausted:skipped="
    assert f.stages[-2:] == ["draft", "chooser"] and callbacks == []


def test_budget_refusal_below_the_global_promotion_adapter_is_re_raised(tmp_path, monkeypatch):
    """Finding 2: the split-root global promotion-only path caught BudgetExceeded
    with `propagate_model_error` too."""
    from ouroboros.usage_accounting import BudgetExceeded

    def refuse(*_args, **_kwargs):
        raise BudgetExceeded("root wallet spent")

    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote", refuse)
    with pytest.raises(BudgetExceeded):
        pipeline._run_global_backlog_promotion_only(
            SimpleNamespace(drive_root=tmp_path), {"id": "t"},
            {"backlog_candidates": [{"summary": "s", "category": "process"}]}, None)


def test_ordinary_promotion_failure_degrades_the_checkpoint_but_keeps_the_global_callback(phase, monkeypatch):
    """Finding 6: the promotion stage logged an ordinary chooser failure at debug and
    completed. It now returns a typed stage failure (degraded, nothing skipped); the
    split-root global callback is still run — only a paid interruption stops it."""
    f = phase

    def fail(*_args, **_kwargs):
        f.stages.append("promotion-model")
        raise RuntimeError("chooser provider failed")

    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote", fail)
    callbacks = []
    f.ready.set()
    pipeline._run_post_task_processing_async(
        f.env, f.task, {"rounds": 3}, {}, f.root / "logs", event_queue=f.events,
        on_reflection=lambda *args: callbacks.append(args))
    assert f.done.wait(5)
    checkpoint = load_task_result(f.root, f.task["id"])["root_phase_checkpoint"]
    assert checkpoint["post_task_synthesis"] == "degraded"
    assert not checkpoint.get("post_task_stop_reason")
    assert f.stages[-2:] == ["draft", "promotion-model"] and len(callbacks) == 1


def _generic_unknown(carrier):
    """A generic API exception (not the typed Claudexor error) whose physical attempt
    was dispatched with no terminal provider fact — directly or on ``__cause__``."""
    inner = RuntimeError("connection reset mid-request")
    inner.physical_attempt_capture = SimpleNamespace(state="unresolved")
    if carrier == "direct":
        return inner
    try:
        raise RuntimeError("request failed") from inner
    except RuntimeError as wrapped:
        return wrapped


def _real_promotion_chooser(monkeypatch, f):
    from ouroboros import post_task_evolution as promotion

    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote", f.real_promote)
    monkeypatch.setattr(config, "get_post_task_evolution_enabled", lambda: True)
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "advanced")
    monkeypatch.setattr(config, "get_post_task_evolution_cadence", lambda: "llm")
    monkeypatch.setattr(promotion, "_eligible", lambda *_: True)
    monkeypatch.setattr(promotion, "_is_canonical_run", lambda *_: True)
    monkeypatch.setattr(promotion, "_closed_objectives_digest", lambda *_: "")


@pytest.mark.parametrize("carrier", ["direct", "cause"])
def test_generic_unknown_outcome_in_a_real_adapter_stops_every_later_paid_call(phase, monkeypatch, carrier):
    """F1: only the typed Claudexor error and ``BudgetExceeded`` were interruptions,
    so a generic API exception carrying an unresolved attempt was swallowed and the
    promotion stage then bought the global callback. The consolidator's chain
    classifier now reads it provider-independently in the REAL chooser adapter:
    nothing paid runs after it."""
    from ouroboros import llm_observability

    f = phase
    error = _generic_unknown(carrier)

    def dispatch(*_args, call_type="", **_kwargs):
        f.stages.append(call_type)
        raise error

    monkeypatch.setattr(llm_observability, "chat_observed", dispatch)
    _real_promotion_chooser(monkeypatch, f)
    callbacks = []
    pipeline._run_post_task_processing_async(
        f.env, f.task, {"rounds": 3}, {}, f.root / "logs", event_queue=f.events,
        on_reflection=lambda *args: callbacks.append(args))
    assert f.done.wait(5)
    checkpoint = load_task_result(f.root, f.task["id"])["root_phase_checkpoint"]
    assert checkpoint["post_task_synthesis"] == "degraded"
    assert checkpoint["post_task_stop_reason"] == "provider_outcome_unknown:skipped="
    assert f.stages == ["facts", "draft", "post_task_evolution_decision"], "no paid call after an unknown outcome"
    assert callbacks == [] and not f.engine.creates


def test_split_root_facts_row_counts_the_actor_store_when_synthesis_runs_canonically(phase, monkeypatch):
    """F2 (TZ-2 C2): a split NON-Project root synthesizes with the parent env and task,
    so ``env.drive_root`` IS the canonical drive; passing it as the child store folded
    two identical canonical stores into one, and a file present only in the actor's
    child store (before copy-back) was never walked — a confirmed-looking zero. Through
    the real dispatch the actor store is the row's recorded ``child_drive_root``."""
    from ouroboros import post_task_synthesis
    from ouroboros.headless import task_artifacts_dir

    f = phase
    child = f.root.parent / "child"
    child.mkdir()
    (task_artifacts_dir(child, f.task["id"]) / "report.md").write_text("r", encoding="utf-8")
    monkeypatch.setattr(pipeline, "_record_task_facts", post_task_synthesis._record_task_facts)
    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote", lambda *a: None)
    child_env = SimpleNamespace(drive_root=child, repo_dir=f.root.parent, drive_path=lambda rel: child / rel)
    child_task = {**f.task, "budget_drive_root": str(f.root), "drive_root": str(child)}
    parent_task = {**child_task, "drive_root": str(f.root), "child_drive_root": str(child)}
    pipeline._dispatch_root_post_task(
        child_env, child_task, "Already answered", None, [], {"rounds": 3}, {}, child / "logs",
        budget_drive_root=str(f.root), split_drive=True, project_scoped=False, project_task=False,
        parent_env=f.env, parent_task=parent_task)
    assert f.done.wait(5)
    rows = chat_rows(f.root)
    [row] = [r for r in rows if r.get("summary_id") == f"task-facts:{f.task['id']}"]
    fact = row["files_rescued"]
    assert (fact["count"], fact["state"], fact["hash_computed"]) == (1, "positive", False)
    assert [s["store"] for s in fact["stores"]] == [
        str(task_artifacts_dir(f.root, f.task["id"], create=False)),
        str(task_artifacts_dir(child, f.task["id"], create=False))]
