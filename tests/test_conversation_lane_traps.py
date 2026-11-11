"""The four traps a lane carrier would fall into if a process predicate stood for a fact (A5), and the
wake's preparation as the carrier's first step (A6).

Transport: the spawner states the proxy policy to every child; a spawned or forkserver carrier keeps
the server's. Presence post-task: the Host's waiting request is an admission fact on the task.
Detach: the verb's "already in background" is the carrier's lane, never the process; a detach
transfers its exact bound runs like any cold sleep. Wakes: the clock admits with the facts only its
process holds; the carrier observes, renders and binds, and the accepted boundary returns with the terminal.
"""

from types import SimpleNamespace

import pytest

from ouroboros import utils
from ouroboros.task_carrier import EXECUTOR_INLINE_DEGRADED, EXECUTOR_WORKER, LANE_CONVERSATION, LANE_PROJECT


def test_transport_policy_is_the_spawners_fact_not_the_process_predicate(monkeypatch):
    monkeypatch.delenv(utils.TRANSPORT_NO_PROXY_ENV, raising=False)
    monkeypatch.delenv(utils.WORKER_PROCESS_ENV, raising=False)
    assert utils.transport_no_proxy() is False
    monkeypatch.setenv(utils.WORKER_PROCESS_ENV, "1")
    assert utils.transport_no_proxy() is True  # a child that received no statement keeps the legacy worker default
    monkeypatch.setenv(utils.TRANSPORT_NO_PROXY_ENV, "0")
    assert utils.transport_no_proxy() is False  # a spawned carrier keeps the server's policy
    monkeypatch.setenv(utils.TRANSPORT_NO_PROXY_ENV, "1")
    assert utils.transport_no_proxy() is True  # a forked child avoids OS proxy lookup


class _FakeProcess:
    def __init__(self):
        self.pid, self.daemon = 1, True

    def start(self):
        pass


def test_every_child_is_told_the_spawners_transport_policy(monkeypatch):
    """Review M3: a spawned or forkserver child is TOLD the server's own policy (proxies honoured), never left to
    the unstated worker default (no proxy); a forked child is told to avoid OS proxy lookup (fork safety)."""
    from supervisor import worker_process

    captured = []

    class Ctx:
        def __init__(self, method):
            self.method = method

        def get_start_method(self):
            return self.method

        def Process(self, *, target, args):
            captured.append(args)
            return _FakeProcess()

    monkeypatch.delenv(utils.TRANSPORT_NO_PROXY_ENV, raising=False)
    monkeypatch.delenv(utils.WORKER_PROCESS_ENV, raising=False)
    for method, expected in (("spawn", False), ("forkserver", False), ("fork", True)):
        captured.clear()
        worker_process.spawn_worker_process(Ctx(method), 0, None, None, "/repo", "/data")
        assert len(captured[0]) == 8 and captured[0][-1] is expected, method
    monkeypatch.setenv(utils.TRANSPORT_NO_PROXY_ENV, "1")  # a server whose own policy avoids proxies
    captured.clear()
    worker_process.spawn_worker_process(Ctx("spawn"), 0, None, None, "/repo", "/data")
    assert captured[0][-1] is True


def _policy_entry(*args):
    """The real ``worker_main`` in a real child with only the agent replaced: its task reports the child's policy."""
    from ouroboros import agent, extension_loader
    from supervisor.worker_process import worker_main

    class PolicyAgent:
        def handle_task(self, task):
            args[2].put(("transport_policy", utils.transport_no_proxy(), utils.in_worker_process()))
            return []

    agent.make_agent = lambda **_: PolicyAgent()
    extension_loader.reload_all = lambda *_a, **_k: None
    worker_main(*args)


@pytest.mark.parametrize("method", ["spawn", "forkserver"])
def test_a_spawned_or_forkserver_carrier_keeps_the_servers_proxy_policy(tmp_path, monkeypatch, method):
    """Trap 1 in the child itself: a carrier started by the default start methods (spawn on macOS/Windows,
    forkserver on Linux) is a worker AND reaches providers through the proxies the server honours."""
    import multiprocessing
    import queue as stdqueue
    import time

    from supervisor import worker_process
    from tests._shared import stop_socket_sharer

    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{method} is not available on this platform")
    monkeypatch.delenv(utils.TRANSPORT_NO_PROXY_ENV, raising=False)
    monkeypatch.delenv(utils.WORKER_PROCESS_ENV, raising=False)
    assert utils.transport_no_proxy() is False  # the server's own policy
    monkeypatch.setattr(worker_process, "worker_main", _policy_entry)
    ctx = multiprocessing.get_context(method)
    incoming, outgoing = ctx.Queue(), ctx.Queue()
    proc = worker_process.spawn_worker_process(ctx, 0, incoming, outgoing, tmp_path, tmp_path)
    report = None
    try:
        incoming.put({"id": "policy-probe", "type": "task"})
        deadline = time.monotonic() + 180  # a cold child imports the worker stack on a loaded machine
        while report is None and time.monotonic() < deadline:
            try:
                event = outgoing.get(timeout=1)
            except stdqueue.Empty:
                assert proc.is_alive(), ("the child ended before its report", proc.exitcode)
                continue
            if isinstance(event, tuple) and event[:1] == ("transport_policy",):
                report = event
        assert report == ("transport_policy", False, True), report
    finally:
        try:
            incoming.put(None)  # the worker loop's own end
            proc.join(timeout=15)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
        finally:
            worker_process.close_worker_stop_channel(proc)
            for channel in (incoming, outgoing):
                channel.close()
                channel.cancel_join_thread()
            stop_socket_sharer()


def test_presence_post_task_reads_the_admission_fact_not_the_process():
    from ouroboros.agent_task_pipeline import presence_request_waits

    waiting = {"_is_direct_chat": True, "metadata": {"source": "presence", "presence": {"binding_id": "b"},
                                                     "presence_request_waits": True}}
    assert presence_request_waits(waiting) is True
    not_waiting = {**waiting, "metadata": {**waiting["metadata"], "presence_request_waits": False}}
    assert presence_request_waits(not_waiting) is False
    assert presence_request_waits({"_is_direct_chat": True, "metadata": {}}) is False


def test_detach_is_already_background_only_on_the_project_lane():
    from ouroboros.model_detach import arm_detach

    def ctx(owner):
        return SimpleNamespace(task_metadata={"_execution_owner": owner}, task_id="t", _model_sleep=None,
                               _owner_wait_requested="", owner_wait_callback=None)

    pooled = arm_detach(ctx({"kind": "pooled", "lane": LANE_PROJECT, "executor": EXECUTOR_WORKER}), "r")
    assert pooled["status"] == "already_in_background"
    lane = arm_detach(ctx({"kind": "direct", "lane": LANE_CONVERSATION, "executor": EXECUTOR_WORKER}), "r")
    assert lane.get("status") != "already_in_background"  # a carried owner turn still holds the foreground
    inline = arm_detach(ctx({"kind": "direct", "lane": LANE_CONVERSATION, "executor": EXECUTOR_INLINE_DEGRADED}), "r")
    assert inline.get("status") != "already_in_background"


def test_a_detach_transfers_its_bound_run_like_any_cold_sleep(monkeypatch, tmp_path):
    """The one delegated run the body can take is no blocker for a detach, exactly as for an await."""
    import threading

    from ouroboros import model_sleep

    run = {"run_id": "run-1", "invocation_id": "inv-1"}
    monkeypatch.setattr("ouroboros.budget_pause.observe_task_runs",
                        lambda root, task_id, **k: {"custody_read": "ok", "runs": [run] if task_id == "t" else []})
    monkeypatch.setattr("ouroboros.delegate_custody.custody_root", lambda ctx: tmp_path)
    monkeypatch.setattr("ouroboros.sleep_custody.transferable_pairs", lambda root, tid: [("run-1", "inv-1")])
    monkeypatch.setattr("ouroboros.tools.services._SERVICES", {})
    monkeypatch.setattr("ouroboros.tools.services._LOCK", threading.Lock())
    monkeypatch.setattr("ouroboros.owner_pause._member_coordinates", lambda ctx: (tmp_path, "t", "t"))
    monkeypatch.setattr("ouroboros.owner_pause.tree_model_requests", lambda root, tree: [])
    monkeypatch.setattr("ouroboros.owner_pause.tree_member_results",
                        lambda root, tree, linked=(): SimpleNamespace(unresolved={}, members={}))
    monkeypatch.setattr("ouroboros.delegate_custody_current.tree_task_ids", lambda root, tree: set())
    monkeypatch.setattr("ouroboros.tool_custody.task_process_blockers", lambda root, members: [])
    monkeypatch.setattr("ouroboros.process_custody._read_ledger_strict", lambda root: (True, []))
    monkeypatch.setattr(model_sleep, "_canonical_root", lambda ctx: tmp_path)
    ctx = SimpleNamespace(task_id="t", _model_sleep={"intent": "continue_in_background"}, drive_root=str(tmp_path),
                          budget_drive_root=str(tmp_path))
    detach = {"intent": "continue_in_background", "tasks": [], "senders": []}
    assert model_sleep.cold_blockers(ctx, chosen=detach, allow_transfer=True) == []
    assert [b["kind"] for b in model_sleep.cold_blockers(ctx, chosen=detach, allow_transfer=False)] == ["delegated_run"]


def test_the_clock_admits_with_its_facts_and_the_carrier_prepares_the_wake(monkeypatch, tmp_path):
    from ouroboros import consciousness as clock_module, consciousness_wake

    assert not hasattr(clock_module, "observe_wake") and not hasattr(clock_module, "render_wake_message")
    task = {"id": "w1", "chat_id": 1, "text": "[Wake-up · heartbeat]", "metadata": {"wake_cause": "own_clock"},
            "_wake_prepare": {"boundary": {"seq": 3}, "since": 10.0, "now": 20.0, "reason": "heartbeat", "level": "act",
                              "repo_dir": str(tmp_path), "last_wake_at": 5.0, "last_wake_task_id": "w0",
                              "disabled_tools": [], "window": {"settled_usd": 1.0, "unknown_unmetered": 0,
                                                               "limit_usd": 5.0, "spend_by_cause": None},
                              "counted_roots": [], "max_tasks": 2, "interval": 600, "appointment_at": 20.0}}
    observed, rendered = [], []
    observation = SimpleNamespace(full_text=lambda: "EVENTS", overview_text=lambda s: "OVERVIEW",
                                  source_bytes=lambda: b"{}\n", captured_at="t", window={}, counts=lambda: {},
                                  gaps=[], boundary=None)
    monkeypatch.setattr(consciousness_wake, "observe_wake", lambda root, **kw: observed.append(kw) or observation)
    monkeypatch.setattr(consciousness_wake, "render_wake_message", lambda *a, **kw: rendered.append(kw) or f"RENDERED:{kw['events']}")
    monkeypatch.setattr(consciousness_wake, "previous_wake_text", lambda *a, **kw: "prev")
    monkeypatch.setattr(consciousness_wake, "bind_wake_observation",
                        lambda root, task, obs, render: {"task_id": task["id"], "seq": 4})
    monkeypatch.setattr("ouroboros.consciousness_view.place_views", lambda root, window: None)
    consciousness_wake.prepare_wake(task, tmp_path)
    assert observed == [{"boundary": {"seq": 3}, "since": 10.0, "now": 20.0, "reason": "heartbeat"}]
    assert task["text"] == "RENDERED:EVENTS" and "_wake_prepare" not in task
    assert task["metadata"]["wake_observation"]["accepted_boundary"] == {"task_id": "w1", "seq": 4}
    assert rendered[0]["previous_wake"] == "prev" and rendered[0]["running"] == 0


def test_a_carried_turn_detaches_against_the_servers_published_pool(tmp_path, monkeypatch):
    """Astra C1: a conversation carrier owns no pool table (its own ``supervisor.workers`` is empty). The verb
    reads the server's published pool facts, so a started healthy pool (at idle only the lane's spare) admits
    the handoff, and a disabled one keeps the dialogue in its process with the server's reason."""
    import json

    from ouroboros import model_detach, model_sleep
    from supervisor import workers

    monkeypatch.setenv(utils.WORKER_PROCESS_ENV, "1")  # a carrier process ...
    monkeypatch.setattr(workers, "WORKERS", {})  # ... whose own pool table is empty
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    monkeypatch.setattr("supervisor.update_merge.active_update_tx", lambda: None)
    monkeypatch.setattr("ouroboros.native_task_wait.armed_conflict", lambda ctx: "")
    armed = []
    monkeypatch.setattr(model_sleep, "request_sleep",
                        lambda ctx, chosen, mode: armed.append(chosen) or {"reason": "sleep_armed"})

    def publish(**pool):
        (tmp_path / "state").mkdir(exist_ok=True)
        (tmp_path / "state" / "queue_snapshot.json").write_text(json.dumps({"pending": [], "running": [], **pool}))

    ctx = SimpleNamespace(task_metadata={"_execution_owner": {"kind": "direct", "lane": LANE_CONVERSATION,
                                                              "executor": EXECUTOR_WORKER}},
                          task_id="t", _model_sleep=None, _owner_wait_requested="", owner_wait_callback=lambda *a: None)
    publish(worker_total=0, lane_carrier_total=1, worker_pool_disabled_reason="")
    accepted = model_detach.arm_detach(ctx, "long work")
    assert accepted.get("ok") is True and accepted.get("status") == "detach_armed" and len(armed) == 1
    publish(worker_total=2, lane_carrier_total=1, worker_pool_disabled_reason="worker_readiness_exhausted")
    refused = model_detach.arm_detach(ctx, "long work")
    assert refused["error"] == "worker_pool_unavailable" and refused["pool"]["disabled_reason"] == "worker_readiness_exhausted"
    assert len(armed) == 1
