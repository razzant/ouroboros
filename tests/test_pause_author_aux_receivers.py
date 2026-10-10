"""Forced-final and compaction sends rejoin the author's real cold Pause rail."""
from __future__ import annotations

from copy import deepcopy
import json
import threading
from types import SimpleNamespace

import pytest

from ouroboros import budget_pause, context_compaction, loop, model_wait, owner_pause
from ouroboros import usage_accounting as ua, working_checkpoint
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message
from ouroboros.task_results import write_task_result
from tests._usage_store_testing import attempt_rows_in_start_order
from tests.test_context_reclaim_materializer import _SPEC, _unit
from tests.test_loop_transport_wait import _loop_kwargs
from tests.test_pause_model_receiver import _extract, _request, _until
from tests.test_working_checkpoint import _registry

pytestmark = pytest.mark.serial


def _start_loop(tmp_path, kwargs):
    outcome, returned = {}, threading.Event()
    owner = model_wait.TaskModelWait(task={"id": "t-wait", "_attempt": 1}, drive_root=tmp_path,
                                     event_queue=None, worker_slot_held=True)
    kwargs["tools"]._ctx.model_wait_context = owner

    def run():
        try:
            with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="t-wait", root_task_id="t-wait")), \
                    model_wait.operation_wait_scope(owner):
                outcome["result"] = loop.run_llm_loop(**kwargs)
        except BaseException as error:
            outcome["error"] = error
        finally:
            returned.set()

    thread = threading.Thread(target=run)
    thread.start()
    return thread, returned, outcome


@pytest.fixture
def author(tmp_path, monkeypatch):
    budget_pause.end_dispatch_fence("t-wait")
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("ouroboros._usage_wait.ABANDON_POLL_SEC", 0.02)
    write_task_result(tmp_path, "t-wait", "running", root_task_id="t-wait")
    registry = _registry(tmp_path, monkeypatch, "t-wait", 1)
    monkeypatch.setattr(loop, "_forced_fallback_result", lambda *_a, **_k: pytest.fail("Pause published a fallback final"))
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *_a, **_k: pytest.fail("no provider"))
    monkeypatch.setattr("ouroboros.loop_llm_call.estimate_cost_optional",
                        lambda *_a, **_k: pytest.fail("fake chat must return known usage without pricing fallback"))
    yield registry
    budget_pause.end_dispatch_fence("t-wait")


@pytest.mark.parametrize("reason", ["round_limit", "deadline"])
def test_forced_final_model_wait_pauses_without_publishing_its_late_answer(tmp_path, monkeypatch, author, reason):
    sent, release = threading.Event(), threading.Event()
    calls = []

    def sender():
        sent.set()
        assert release.wait(10)
        return {"role": "assistant", "content": "late forced final", "tool_calls": []}

    def chat(**kwargs):
        calls.append(kwargs)
        return ua.execute_physical_attempt(_request(), sender, extractor=_extract), {}

    if reason == "round_limit":
        monkeypatch.setattr(loop, "_resolve_loop_max_rounds", lambda *_a: 0)
    else:
        from datetime import datetime, timedelta, timezone

        author._ctx.task_metadata = {"deadline_at": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()}
    llm = SimpleNamespace(default_model=lambda: "test-model", chat=chat)
    thread, returned, outcome = _start_loop(tmp_path, _loop_kwargs(tmp_path, author, [], llm=llm))
    try:
        assert sent.wait(5), repr(outcome)
        owner_pause.install_fence(tmp_path, "t-wait", request_id="pause-forced")
        assert returned.wait(3), "the forced answer's local receiver still waited for its sender"
        assert isinstance(outcome.get("error"), budget_pause.BudgetPauseRequested), repr(outcome)
        assert len(calls) == 1 and "result" not in outcome
        assert author._ctx._delivery_candidate is None
        row = outcome["error"].pause
        saved = json.loads(read_actor_source_bytes(tmp_path, "t-wait", row["source_ref"]))
        assert saved["delivery_candidate"] is None
        assert "late forced final" not in str(saved)
        assert saved["resume_point"]["abandoned_model_attempts"]
        assert attempt_rows_in_start_order(tmp_path)[0]["state"] == "dispatched"
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert _until(lambda: attempt_rows_in_start_order(tmp_path)[0]["state"] == "settled")
    assert len(calls) == 1 and author._ctx._delivery_candidate is None


@pytest.mark.parametrize("local", [False, True])
def test_legacy_compaction_pause_keeps_raw_source_pending_request_and_acks_held_mail(tmp_path, monkeypatch, author, local):
    from tests.test_context_fit_v664 import _plan

    sent, release = threading.Event(), threading.Event()
    calls = []
    kwargs = _loop_kwargs(tmp_path, author, [])
    kwargs["messages"] = _plan().messages_for("max") + _unit("old", "x") + _unit("recent", "y")
    original = deepcopy(kwargs["messages"])
    author._ctx._pending_compaction = 1
    author._ctx._last_context_observation = {
        "exposed_units": context_compaction.exposed_context_units(original, original)}
    monkeypatch.setattr(context_compaction, "_summarizer_spec", lambda: {**_SPEC, "use_local": local})
    monkeypatch.setattr(loop, "_dispatch_round_model", lambda *_a, **_k: pytest.fail("Pause bought a main call"))
    # The previous ready point predates new owner mail and must stay recoverable.
    working_checkpoint.save_round(SimpleNamespace(tools=author, accumulated_usage={}, messages=original,
        llm_trace={}, round_idx=0, tool_schemas=[], owner_msg_seen=set()), "ready")
    previous = working_checkpoint.checkpoint_path(tmp_path, "t-wait", 1).read_bytes()
    assert write_owner_message(tmp_path, "retain my unsaved instruction", "t-wait", msg_id="new-mail")

    def sender():
        sent.set()
        assert release.wait(10)
        return {"role": "assistant", "content": "late compacted text", "tool_calls": []}

    def chat(_client, **kwargs):
        calls.append(kwargs)
        return ua.execute_physical_attempt(_request(), sender, extractor=_extract), {}

    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", chat)
    thread, returned, outcome = _start_loop(tmp_path, kwargs)
    try:
        assert sent.wait(5), repr(outcome)
        owner_pause.install_fence(tmp_path, "t-wait", request_id="pause-compaction")
        assert returned.wait(3), "the compaction receiver still waited for its sender"
        assert isinstance(outcome.get("error"), budget_pause.BudgetPauseRequested), repr(outcome)
        assert "result" not in outcome and len(calls) == 1
        assert working_checkpoint.checkpoint_path(tmp_path, "t-wait", 1).read_bytes() == previous
        saved = json.loads(read_actor_source_bytes(tmp_path, "t-wait", outcome["error"].pause["source_ref"]))
        assert saved["context_observations"]["_pending_compaction"] == 1
        for call_id in ("old", "recent"):
            assert next(row for row in saved["messages"] if row.get("tool_call_id") == call_id) == \
                next(row for row in original if row.get("tool_call_id") == call_id)
        assert "retain my unsaved instruction" in str(saved["messages"])
        # The durable exact source holds it and Resume restores its ``seen``: only now is it ACKed.
        assert "new-mail" in saved["seen"] and drain_owner_entries(tmp_path, "t-wait", set(), 1) == []
        assert "late compacted text" not in str(saved)
        assert saved["resume_point"]["abandoned_model_attempts"]
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert _until(lambda: attempt_rows_in_start_order(tmp_path)[0]["state"] == "settled")
    assert len(calls) == 1, "the interrupted structured summarizer must not buy a JSON retry"


@pytest.mark.parametrize("surface", ["forced", "compaction"])
def test_warm_resume_restarts_author_wait_while_the_original_review_still_runs(tmp_path, monkeypatch, author, surface):
    from dataclasses import replace

    from ouroboros import owner_wait, review_pause
    from ouroboros.review_substrate import run_review_request
    from supervisor.budget_resume import resume_warm_owner_pause_root
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_context_fit_v664 import _plan
    from tests.test_pause_review_completion import EpisodeModel
    from tests.test_review_operation_lifetime import _request as review_request, _slot

    _queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    monkeypatch.setattr(review_pause, "DETACH_POLL_SEC", 0.01)
    workers.RUNNING["t-wait"] = {"task": {"id": "t-wait", "type": "task", "chat_id": 7,
                                          "root_task_id": "t-wait"}, "worker_id": 0, "attempt": 1}
    reviewer = EpisodeModel()
    review_returned, sent, parked, release = (threading.Event() for _ in range(4))
    review_outcome, calls = {}, []

    def review():
        try:
            with model_wait.task_model_wait_scope(task={"id": "t-wait", "_attempt": 1},
                    drive_root=tmp_path, event_queue=None, worker_slot_held=False):
                review_outcome["result"] = run_review_request(replace(review_request(drain=False), task_id="t-wait"),
                    slots=[_slot()], drive_root=tmp_path, usage_ctx=author._ctx, llm=reviewer)
        except BaseException as error:
            review_outcome["error"] = error
        finally:
            review_returned.set()

    review_thread = threading.Thread(target=review)
    review_thread.start()
    # Serialize only the two consumers noticing the same real fence. The
    # reviewer must have detached before the author chooses its warm carrier.
    control = loop._handle_model_wait_control

    def after_review_detaches(*args, **kwargs):
        assert review_returned.wait(5), repr(review_outcome)
        assert review_pause.live_detached_operations("t-wait"), repr(review_outcome)
        return control(*args, **kwargs)

    monkeypatch.setattr(loop, "_handle_model_wait_control", after_review_detaches)

    def park(ctx, checkpoint):
        workers.RUNNING["t-wait"]["owner_wait"] = {**checkpoint, "state": "waiting"}
        parked.set()
        return owner_wait.direct_owner_wait(ctx, checkpoint)

    author._ctx.owner_wait_callback = park
    kwargs = _loop_kwargs(tmp_path, author, [])

    def chat(**options):
        calls.append(options)
        first = len(calls) == 1

        def send():
            if first:
                sent.set()
                assert release.wait(15)
                return {"role": "assistant", "content": "abandoned original reply", "tool_calls": []}
            assert not reviewer.release_first.is_set(), "Resume waited for the review"
            if surface == "compaction":
                payload = json.loads(options["messages"][-1]["content"].splitlines()[-1])
                return {"role": "assistant", "content": json.dumps({"summaries": [
                    {"source_id": row["source_id"], "summary": "retained short source"} for row in payload]})}
            return {"role": "assistant", "content": "resumed forced answer", "tool_calls": []}

        response = ua.execute_physical_attempt(_request(), send, extractor=_extract)
        usage, cost, cost_final = _extract(response)
        return response, {**usage, "cost": cost, "cost_final": cost_final}

    if surface == "forced":
        monkeypatch.setattr(loop, "_resolve_loop_max_rounds", lambda *_a: 0)
        kwargs["llm"] = SimpleNamespace(default_model=lambda: "test-model", chat=chat)
    else:
        kwargs["messages"] = _plan().messages_for("max") + _unit("old", "x") + _unit("recent", "y")
        author._ctx._pending_compaction = 1
        author._ctx._last_context_observation = {"exposed_units":
            context_compaction.exposed_context_units(kwargs["messages"], kwargs["messages"])}
        monkeypatch.setattr(context_compaction, "_summarizer_spec", lambda: {**_SPEC, "use_local": True})
        monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda _client, **options: chat(**options))
        monkeypatch.setattr(loop, "_dispatch_round_model", lambda *_a, **_k: (
            {"role": "assistant", "content": "resumed main answer", "tool_calls": []}, 0.0))
    thread = None
    try:
        assert reviewer.first_sent.wait(5), repr(review_outcome)
        thread, returned, outcome = _start_loop(tmp_path, kwargs)
        assert sent.wait(5), repr(outcome)
        owner_pause.install_fence(tmp_path, "t-wait", request_id="pause-with-review")
        assert parked.wait(3), repr(outcome)
        assert not returned.is_set() and len(calls) == 1
        assert review_pause.live_detached_operations("t-wait")
        resumed = resume_warm_owner_pause_root("t-wait")
        assert resumed and resumed["ok"] and resumed["warm"], resumed
        assert returned.wait(5), repr(outcome)
        assert "error" not in outcome, repr(outcome)
        assert outcome["result"][0] == ("resumed forced answer" if surface == "forced" else "resumed main answer")
        assert len(calls) == 2 and review_pause.live_detached_operations("t-wait")
        grant = owner_pause.read_fence(tmp_path, "t-wait")["resume_grant"]
        assert grant["consumed_by"] == "t-wait" and grant["consumed_at"]
        assert any(row["state"] == "dispatched" and row["model"] == "m"
                   for row in attempt_rows_in_start_order(tmp_path)), "old money was not erased by Resume"
    finally:
        release.set()
        reviewer.release_first.set()
        if thread is not None:
            thread.join(10)
        review_thread.join(10)
        assert _until(lambda: not review_pause.live_detached_operations("t-wait"))
    assert not thread.is_alive() and not review_thread.is_alive()
