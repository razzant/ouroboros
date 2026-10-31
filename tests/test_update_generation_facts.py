"""Update facts survive generations and reach both consumers without host speech."""
import asyncio
import builtins
import concurrent.futures
import io
import itertools
import json
import os
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import config, context, server_process, update_generation_facts as facts
from ouroboros.gateway import control, update_progress as progress
from ouroboros.utils import atomic_write_json
from tests.test_update_progress import (isolated_progress, _boot, _next_server,  # noqa: F401
                                        _restart_requested_attempt, _seen)


@pytest.fixture(autouse=True)
def samples(monkeypatch):
    monkeypatch.setattr(facts, "_duties", {})
    monkeypatch.setattr(facts, "_direct", None)
    monkeypatch.setattr(facts, "_dispatch_pending", False)
    monkeypatch.setattr(facts, "_cpu_sample", None)
    monkeypatch.setattr(facts, "_rss", lambda: (512 * 1024 * 1024, "current"))
    monkeypatch.setattr(server_process, "server_source_baseline", lambda: "a" * 40)


def test_direct_dispatch_does_no_io_until_supervisor_handles_its_frame(isolated_progress, monkeypatch, tmp_path):  # noqa: F811
    from supervisor import events
    from supervisor.active_activity import TurnExecution
    from supervisor.log_addressing import TurnEventQueue

    _restart_requested_attempt()
    _next_server(isolated_progress, monkeypatch)
    _boot(monkeypatch, _seen("applied"))
    execution = TurnExecution(chat_id=1, task_id="owner", owner_turn=True)
    frames = queue.Queue()
    turn = TurnEventQueue(frames, "owner", 1, on_frame=execution.observe)
    frame = {"type": "main_llm_call_state", "task_id": "owner", "phase": "started", "task_attempt": 1,
             "llm_call_id": "call-1", "execution_id": "execution-1", "round_id": "round-1", "call_attempt": 1}
    # Dispatch is the first observed send of the call (lane O), not its start.
    sent = {**frame, "phase": "observed", "model_activity": {"phase": "dispatching"}}
    calls = []

    def watch(owner, name):
        original = getattr(owner, name)

        def observed(*args, **kwargs):
            calls.append((name, threading.get_ident()))
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, observed)

    for owner, name in ((builtins, "open"), (io, "open"), (os, "open"),
                        (progress, "_read_saved"), (facts, "sample"), (progress, "_store")):
        watch(owner, name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as producer:
        producer.submit(turn.put, frame).result()
        assert execution.phase == "requested"
        # The supervisor handles the start frame before any send: nothing is pending, nothing is read.
        events._handle_main_llm_call_state(frames.get_nowait(), SimpleNamespace(RUNNING={}))
        producer.submit(turn.put, sent).result()
    assert execution.phase == "dispatched"
    assert calls == [], "the turn observer must not read state, sample or write before model dispatch"
    assert isolated_progress.saved[progress.STATE_KEY]["generation_facts"]["after"]["direct_message_first_dispatch"] is None

    # The send frame publishes the fact its observer stamped. The turn can have ended before the
    # supervisor drains it: no live registry entry is needed. The same frame carries the turn's
    # receipt, which the consumer writes to the journal after the update facts (its own opens).
    sent_frame = frames.get_nowait()
    assert sent_frame["direct_turn"]["execution_id"] == execution.execution_id
    monkeypatch.setattr("supervisor.active_activity._DIRECT_TURNS", __import__("collections").deque(maxlen=100))
    events._handle_main_llm_call_state(sent_frame, SimpleNamespace(RUNNING={}, DRIVE_ROOT=tmp_path))
    after = isolated_progress.saved[progress.STATE_KEY]["generation_facts"]["after"]["direct_message_first_dispatch"]
    assert after == {"task_id": "owner", "attempt_id": execution.execution_id,
                     "accepted_at": execution.marks[0][1], "dispatched_at": execution.marks[-1][1],
                     "seconds": execution.marks[-1][1] - execution.marks[0][1]}
    assert [call for call in calls if call[0] != "open"] == [(name, threading.get_ident()) for name in ("_read_saved", "sample", "_store")]
    assert calls.index(("_store", threading.get_ident())) < len(calls) - 1  # the receipt's journal open comes after
    calls.clear()
    # A repeat and another round carry no new first-dispatch fact and do no persistence work.
    events._handle_main_llm_call_state(frame, SimpleNamespace(RUNNING={}))
    turn.put({**frame, "round_id": "round-2", "llm_call_id": "call-2"})
    events._handle_main_llm_call_state(frames.get_nowait(), SimpleNamespace(RUNNING={}))
    assert calls == []


def test_update_records_before_and_after_and_events_update_only_its_generation(isolated_progress, monkeypatch):  # noqa: F811
    # Each sample's monotonic read advances by one second, so the CPU interval is never the zero
    # that two fast reads of a coarse clock (Windows) give; the CPU time itself stays real.
    ticks = itertools.count(1.0)
    monkeypatch.setattr(facts, "time", SimpleNamespace(monotonic=lambda: next(ticks), process_time=time.process_time))
    facts.observe_duty_overrun("reconcile")
    facts.observe_direct_dispatch(task_id="before", accepted_at=100, dispatched_at=102, attempt_id="attempt-before")
    _restart_requested_attempt()
    before = isolated_progress.saved[progress.STATE_KEY]["generation_facts"]["before"]
    assert before["rss_bytes"] == 512 * 1024 * 1024 and before["cpu_seconds"] >= 0
    assert before["duty_overruns"] == {"reconcile": 1}
    assert before["direct_message_first_dispatch"]["seconds"] == 2
    assert isolated_progress.saved[progress.STATE_KEY]["generation_facts"]["after"] is None
    _next_server(isolated_progress, monkeypatch)
    monkeypatch.setattr(facts, "_duties", {})
    monkeypatch.setattr(facts, "_direct", None)
    view = _boot(monkeypatch, _seen("applied"))
    assert view["generation_facts"]["before"] == before
    assert view["generation_facts"]["after"]["source_sha"] == "c" * 40
    assert view["generation_facts"]["after"]["direct_message_first_dispatch"] is None
    facts.observe_direct_dispatch(task_id="after", accepted_at=200, dispatched_at=233, attempt_id="attempt-after")
    facts.observe_duty_overrun("custody")
    after = isolated_progress.saved[progress.STATE_KEY]["generation_facts"]["after"]
    assert after["direct_message_first_dispatch"]["seconds"] == 33
    assert after["direct_message_first_dispatch"]["task_id"] == "after"
    assert after["duty_overruns"] == {"custody": 1}
    assert isinstance(after["cpu_percent"], float)
    # A later ordinary restart cannot relabel the prior generation's evidence.
    _next_server(isolated_progress, monkeypatch, generation="server-c")
    facts.observe_direct_dispatch(task_id="other-generation", accepted_at=0, dispatched_at=1)
    facts.publish_pending_dispatch()
    assert isolated_progress.saved[progress.STATE_KEY]["generation_facts"]["after"] == after
    # One attempt, no growing sample history: admission replaces it.
    progress.begin(base_sha="c" * 40, target_sha="d" * 40)
    assert isolated_progress.saved[progress.STATE_KEY]["generation_facts"]["after"] is None
    assert isolated_progress.saved[progress.STATE_KEY]["generation_facts"]["before"]["generation"] == "server-c"


@pytest.mark.parametrize("outcome,bootstrap", [("not_applied", True), ("rolled_back", True), ("applied", False)])
def test_refusal_or_unconfirmed_boot_does_not_claim_an_after_sample(isolated_progress, monkeypatch, outcome, bootstrap):  # noqa: F811
    _restart_requested_attempt()
    _next_server(isolated_progress, monkeypatch)
    _boot(monkeypatch, _seen(outcome), bootstrap_ok=bootstrap)
    facts.observe_direct_dispatch(task_id="unconfirmed", accepted_at=1, dispatched_at=2)
    facts.publish_pending_dispatch()
    assert isolated_progress.saved[progress.STATE_KEY]["generation_facts"]["after"] is None


def test_record_reaches_panel_api_and_real_context_builder_without_chat(isolated_progress, tmp_path, monkeypatch):  # noqa: F811
    from supervisor import git_ops, message_bus
    monkeypatch.setattr(message_bus, "send_with_budget", lambda *a, **k: pytest.fail("host must not speak"))
    _restart_requested_attempt()
    _next_server(isolated_progress, monkeypatch)
    _boot(monkeypatch, _seen("applied"))
    facts.observe_direct_dispatch(task_id="visible-task", accepted_at=1, dispatched_at=4)
    facts.publish_pending_dispatch()
    state_file = tmp_path / "state" / "state.json"
    atomic_write_json(state_file, isolated_progress.saved)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "compute_managed_update_status", lambda **k: {"current_sha": "c" * 40, "latest_sha": ""})
    monkeypatch.setattr(context, "get_git_info", lambda _: ("local", "c" * 40))
    expected = facts.projection(drive_root=tmp_path)
    assert expected["after"]["direct_message_first_dispatch"]["task_id"] == "visible-task"
    panel = json.loads(asyncio.run(control.api_update_status(None)).body)
    assert panel["after_update"] == expected
    env = SimpleNamespace(repo_dir=tmp_path / "repo", drive_root=tmp_path,
                          repo_path=lambda path: tmp_path / "repo" / path,
                          drive_path=lambda path: tmp_path / path)
    before = state_file.read_bytes()
    text = context.build_runtime_section(env, {"id": "reader", "type": "task"})
    for rendered in (text, text.operative_view):
        assert isinstance(rendered, str)
        runtime, _ = json.JSONDecoder().raw_decode(rendered.split("\n\n", 1)[1])
        # The mind reads the samples once, as the body fact's last update; the Runtime copy of the
        # update projection no longer repeats them (the panel keeps its own reader above).
        assert "after_update" not in runtime["official_update"]
        assert runtime["vitals"]["last_update"].startswith("applied " + "b" * 12 + ": first request from handler entry before ")
        assert runtime["vitals"]["last_update"].endswith("; record state/state.json managed_update_attempt")
        assert " → after 3.0 s; rss 512 → 512 MiB" in runtime["vitals"]["last_update"]
        assert "cpu" not in runtime["vitals"]["last_update"]  # a share over a sub-second interval is noise, not a fact
    assert state_file.read_bytes() == before  # passive readers never resample/write


def test_unknown_rss_and_unobserved_dispatch_are_not_zero(monkeypatch):
    monkeypatch.setattr(facts, "_rss", lambda: (None, "unobserved"))
    monkeypatch.setattr(facts, "_duties", None)
    row = facts.sample()
    assert row["rss_bytes"] is None and row["direct_message_first_dispatch"] is None
    assert row["duty_overruns"] is None
    assert row["cpu_percent"] is None
    facts.observe_direct_dispatch(task_id="bad-clock", accepted_at=2, dispatched_at=1)
    assert facts.sample()["direct_message_first_dispatch"] is None
