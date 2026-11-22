"""A measured refusal before the first model send is a retained, non-speaking Presence outcome."""
import contextlib
import json
import threading
from types import SimpleNamespace

import pytest

from ouroboros import agent as agent_module, call_peers, call_protocol as cp, live_conversation, presence_call, places
from ouroboros.context_fit import MinimumViewUnfit
from ouroboros.context_window import WindowLimit
from ouroboros.contracts.realtime import RouteFacts
from ouroboros.presence_admission import PresenceAdmission
from ouroboros.presence_bindings import PresenceEndpoint
from ouroboros.presence_executions import PresenceTurnExecutions
from ouroboros.presence_recovery import retained_work_view
from ouroboros.presence_runner import PresenceTurnEvent, PresenceTurnGate, run_presence_turn
from ouroboros.task_carrier import EXECUTOR_INLINE_DEGRADED, EXECUTOR_WORKER, carrier_executor
from ouroboros.task_results import load_task_result
from ouroboros.tools.tool_context import ToolContext
from tests._dialogue import chat_rows
from tests.conversation_carrier_support import lone_host_world as lone_host_world  # noqa: F401 -- fixture

SKILL = "meet_bridge"
PRIVATE = "synthetic fit assembly diagnostic"


def _wire():
    return {"source_event_id": "synthetic-event", "provider": "meet", "account_id": "account",
            "conversation_id": "room", "thread_id": "", "conversation_key": "meet:account:room:",
            "actor": {"platform_actor_id": "visitor"}, "conversation": {}, "message": {}, "text": "Hello."}


def _unfit(kind):
    return MinimumViewUnfit(PRIVATE, fixed_tokens=96_000, memory_tokens=3_000, reserve_tokens=10_000,
                            window_tokens=102_000 if kind == "input" else 92_000, ratio=1.0,
                            limiting=WindowLimit(92_000, kind, "synthetic fixture"))


def _agent_factory(monkeypatch, failure):
    """Use the real bound body and finalizer; inject at the ordinary pre-model preparation seam."""
    seen, make_agent = [], agent_module.make_agent
    monkeypatch.setattr(agent_module.OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr(agent_module, "run_llm_loop", lambda **_kw: pytest.fail("the preparation did not fit"))
    monkeypatch.setattr(live_conversation.LiveConversationTurn, "run",
                        lambda *a, **kw: pytest.fail("a call cannot reach its live host after this refusal"))
    monkeypatch.setattr("ouroboros.agent_task_pipeline._run_post_task_processing_async",
                        lambda *a, **kw: pytest.fail("pre-dispatch Presence failure cannot start synthesis"))

    def factory(*, repo_dir, drive_root, event_queue=None):
        actor = make_agent(repo_dir, drive_root, event_queue=event_queue)
        actor.handle_task = actor._handle_task_scoped  # don't test the installation's settings/billing preflight here

        def prepare(task, _authority_refusal):
            root = actor.env.drive_root
            row = load_task_result(root, task["id"], strict=True)
            assert row["status"] == "running"
            assert row["metadata"]["presence_event_identity"] == task["metadata"]["presence_event_identity"]
            seen.append(carrier_executor(task))
            actor.tools.set_context(ToolContext(repo_dir=actor.env.repo_dir, drive_root=root,
                task_id=task["id"], current_chat_id=task["chat_id"], current_task_type="presence",
                is_direct_chat=True, task_metadata=task["metadata"]))
            raise failure

        monkeypatch.setattr(actor, "_prepare_task_context", prepare)
        return actor

    return factory, seen


def _check_terminal(root, task_id, kind):
    row = load_task_result(root, task_id, strict=True)
    assert (row["status"], row["reason_code"], row["terminal_origin"]) == ("failed", "context_capacity", "host_notice")
    assert row["outcome_axes"]["execution"]["status"] == "infra_failed"
    label = "input limit" if kind == "input" else "total window"
    assert f"{label} here is 92,000 tokens" in row["result"] and "102,000" not in row["result"]
    assert PRIVATE not in row["result"] and "MinimumViewUnfit" not in row["result"]
    assert not row.get("presence_recovery_recorded") and not row.get("status_reconciled_from")
    assert not {"presence_retry_proof", "presence_unknown_outcome"} & row["metadata"].keys()
    assert not row.get("presence_retry_next")
    status, work = retained_work_view(row, task_id, root)
    assert (status, work["status"], work["outcome"], work["text"]) == (200, "failed", "silent", "")
    assert not work.get("output_ref") and not work.get("child_work_ref")
    rows = [r for r in chat_rows(root) if r.get("task_id") == task_id]
    assert sum(r.get("direction") == "in" for r in rows) == 1
    assert not any(r.get("direction") == "out" for r in rows)
    return row


@pytest.mark.parametrize("executor,kind", [("", "total"), (EXECUTOR_WORKER, "input")], ids=["legacy", "worker"])
def test_text_presence_start_keeps_measured_failed_silence(tmp_path, monkeypatch, executor, kind):
    wire = _wire()
    endpoint = PresenceEndpoint(wire["provider"], wire["account_id"], wire["conversation_id"], wire["thread_id"])
    admission = PresenceAdmission(binding_id="legacy-binding", transport_skill=SKILL, origin=endpoint, destination=endpoint)
    factory, seen = _agent_factory(monkeypatch, _unfit(kind))
    args = dict(admission=admission, event=PresenceTurnEvent(**wire), repo_dir=tmp_path, drive_root=tmp_path,
                gate=PresenceTurnGate(1), agent_factory=factory)
    if executor:
        args["carrier_owner"] = {"kind": "presence", "executor": executor, "lane": "conversation"}
    first = run_presence_turn(**args)
    assert (first.outcome, first.text, first.output_ref) == ("silent", "", "")
    assert run_presence_turn(**args) == first and seen == [executor]
    row = _check_terminal(tmp_path, first.task_id, kind)
    assert "before the next event" in row["result"] and "send the request again" not in row["result"]


@pytest.fixture
def waking_place(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance",
                        lambda _root: {"status": "available", "remaining_usd": 2})
    monkeypatch.setattr("ouroboros.config.get_consciousness_autonomy", lambda: "act")
    return places.connect_place(tmp_path, connector=SKILL, provider="meet", account_id="account",
                                conversation_id="room", wakes=["mention"])


@pytest.mark.parametrize("version", [0, 2])
def test_degraded_host_replays_the_same_silent_capacity_terminal(tmp_path, monkeypatch, lone_host_world, waking_place,
                                                               version):
    from starlette.testclient import TestClient
    from ouroboros.gateway import host_service

    async def authenticated(*_args):
        return SKILL, {}

    (tmp_path / "repo").mkdir(exist_ok=True)
    monkeypatch.setattr(host_service, "_authenticated", authenticated)
    factory, seen = _agent_factory(monkeypatch, _unfit("input"))
    monkeypatch.setattr(agent_module, "make_agent", factory)  # the native carrier resolves it at call time
    payload = {"binding_id": waking_place.place_id, "event": {**_wire(), "facts": ["mention"]},
               "continuation_version": version}
    with TestClient(host_service.create_host_service_app(tmp_path)) as client:
        first = client.post("/presence/turn", json=payload)
        assert first.status_code == 200, first.json()
        replay = client.post("/presence/turn", json=payload)
        assert replay.status_code == 200 and replay.json() == first.json()
    body = first.json()
    assert (body["outcome"], body["text"]) == ("silent", "")
    assert not body.get("work_ref") and not body.get("output_ref") and seen == [EXECUTOR_INLINE_DEGRADED]
    _check_terminal(tmp_path, body["turn_ref"], "input")


@pytest.mark.parametrize("case", ["input", "total", "fault"])
def test_place_call_closes_once_and_keeps_its_typed_preparation_result(tmp_path, monkeypatch, waking_place, case):
    # The place opens its actual process-local hub and proactive thread; attach a test peer as the hub opens.
    # No provider/configuration backend, audio stream, skill handler or outbound bridge participates.
    closed, hubs, notices, executions = [], [], [], []
    fault = RuntimeError(PRIVATE) if case == "fault" else _unfit(case)
    factory, seen = _agent_factory(monkeypatch, fault)
    registry = PresenceTurnExecutions()
    native_start = registry.start_thread
    native_open, native_hub_close = call_peers.open_hub, call_peers.LiveCallHub.close

    @contextlib.contextmanager
    def configuration(_name):
        yield {}, SimpleNamespace(enabled=True, provider="synthetic", model="voice", voice="")

    def bind(self, _feature, _config, spec, publish):
        self.spec, self.model, self.publish = spec, spec.model, publish
        self.request = {"kind": "live_conversation", "call_id": spec.call_id}
        self.catalog_model = SimpleNamespace(facts=RouteFacts(conversation_window_tokens=92_000, source="synthetic"))
        self._lock, self._closed = threading.Lock(), False

    def opened(spec):
        hub = native_open(spec)
        hub.join(cp.PeerInfo("peer", cp.PEER_MEETING, True, True, skill=SKILL), 24000)
        hubs.append(hub)
        return hub

    def closed_once(hub, reason, **words):
        closed.append((reason, words.get("detail", "")))
        native_hub_close(hub, reason, **words)

    def started(*args, **kwargs):
        execution, ok = native_start(*args, **kwargs)
        executions.append(execution)
        return execution, ok

    monkeypatch.setattr(live_conversation, "feature_configuration", configuration)
    monkeypatch.setattr(live_conversation.LiveConversationTurn, "__init__", bind)
    monkeypatch.setattr(call_peers, "open_hub", opened)
    monkeypatch.setattr(call_peers.LiveCallHub, "close", closed_once)
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: SimpleNamespace(broadcast=notices.append))
    monkeypatch.setattr(presence_call, "PROACTIVE_TURNS", registry)
    monkeypatch.setattr(registry, "start_thread", started)
    args = dict(drive_root=tmp_path, repo_dir=tmp_path, skill_name=SKILL, binding_id=waking_place.place_id,
                event={**_wire(), "facts": ["mention"]}, agent_factory=factory)
    opening = presence_call.open_presence_call(**args)
    if case == "fault":
        with pytest.raises(RuntimeError, match=PRIVATE):
            executions[0].result.result(timeout=10)
    else:
        result = executions[0].result.result(timeout=10)
        assert (result.task_id, result.outcome, result.text) == (opening.task_id, "silent", "")
        with pytest.raises(presence_call.PresenceCallRefused) as repeat:
            presence_call.open_presence_call(**args)
        assert repeat.value.code == "presence_call_settled" and len(hubs) == 1
    assert seen == [""] and len(executions) == 1
    hub = hubs[0]
    assert hub.closed and call_peers.get_hub(opening.call_id) is None
    frames = []
    while (frame := hub.next_outgoing("peer")) is not None:
        assert isinstance(frame, str), "no speech audio"
        frames.append(json.loads(frame))
    [frame] = [f for f in frames if f.get("state") == "closed"]
    assert not any(f["type"] == "caption" for f in frames)
    [(reason, detail)] = closed  # the place runner's generic cleanup cannot close this hub again
    assert frame["reason"] == reason and PRIVATE not in detail
    journal = tmp_path / "logs" / "events.jsonl"
    logs = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()] if journal.exists() else []
    fits = [row for row in logs if row["type"] == "call.fit_refused"]
    if case == "fault":
        assert reason == "presence_ended" and not fits and not notices
        row = load_task_result(tmp_path, opening.task_id, strict=True)
        assert row["status"] == "running" and row["presence_recovery_recorded"]
        assert retained_work_view(row, opening.task_id, tmp_path)[1]["code"] == "presence_attempt_outcome_unknown"
        return
    row = _check_terminal(tmp_path, opening.task_id, case)
    assert reason == "context_capacity" and detail == frame["detail"] == row["result"]
    assert "Calls (beta) settings" in detail and "continue by text" in detail
    assert [n for n in notices if n.get("type") == "conversation_finished"] == [{
        "type": "conversation_finished", "chat_id": hub.spec.chat_id, "call_id": opening.call_id,
        "task_id": opening.task_id, "reason": "context_capacity"}]
    [fit] = fits
    assert fit["stage"] == "task_preparation" and (fit["task_id"], fit["call_id"]) == (opening.task_id, opening.call_id)
    m = fit["fit"]["measurement"]
    assert (m["limit_kind"], m["limit_tokens"], m["capacity_total_tokens"]) == (case, 92_000, _unfit(case).window_tokens)
