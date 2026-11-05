"""#572: the window the VLM tool selects from its task deadline also bounds the send its tracked child makes.

The real tool body and parent arithmetic, the real tracked child process and its IPC, and in the child the real
``LLMClient.vision_query``, recovery driver, accounting and OpenAI SDK. Only the child's network transport (in
memory, sockets refused) or engine gateway, its wall clock and its price lookup are fixtures. The child is a separate
process: no parent context reaches it except its serialized payload.
"""
import contextlib
import itertools
import json
import queue
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import deadline_utils, model_wait
from ouroboros import usage_accounting as ua
from ouroboros.llm_claudexor import ClaudexorModelError, ClaudexorModelNotDispatched
from ouroboros.model_wait import ModelWaitInterrupted
from ouroboros.tools import vision
from ouroboros.transport_custody import outcome_unknown_on_chain
from tests._usage_store_testing import ledger_rows

# Not serial: each child is this test's own tracked process over tmp-owned data, and every module global it
# replaces is restored by monkeypatch; no port, shared file or process outlives the test.

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgAAIAAAUAAXpeqz8AAAAASUVORK5CYII="
MODEL = "openai-compatible::vision-fixture"
SUBSCRIPTION = "claudexor::fixture=image-model"
NOW = 1000.0  # the parent's clock while the tool selects the call's window
# The tool keeps the 120 s finalization grace and two 30 s settlement margins: this window ends at 1120.
DEADLINE = NOW + 120 + 2 * 30 + 120
GATEWAY_FIXTURE = Path(__file__).with_name("test_vision_model_wait.py")


def _install_child(spec_path):
    """Runs in the tracked child before ``child_main``: its clock starts where the parent's stood plus the
    child's own start-up, recording its candidate spends more of it, and the SDK's transport is in memory."""
    import socket

    import httpx

    from ouroboros import llm_claudexor as transport
    from ouroboros import net_transport, observability, pricing
    from ouroboros.llm import LLMClient
    from ouroboros.tools import vision_process

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if spec["gateway"]:
        import runpy

        runpy.run_path(str(GATEWAY_FIXTURE))["_install_child_fixture"](spec["mode"], spec["events"])
    now = [spec["child_start"]]

    def event(kind, **fields):
        with open(spec["events"], "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, "now": now[0], **fields}) + "\n")

    deadline_utils.utc_now = lambda: datetime.fromtimestamp(now[0], timezone.utc)
    deadline_utils.time = SimpleNamespace(time=lambda: now[0])
    model_wait.monotonic_now = lambda slot=None: now[0]
    persist = observability.persist_physical_candidate

    def recorded(*args, **kwargs):
        result = persist(*args, **kwargs)
        now[0] += spec["recording_sec"]
        event("recorded")
        return result

    observability.persist_physical_candidate = recorded
    if spec["control"]:  # a readable control refuses the send, as an owner Pause's fence does
        model_wait.dispatch_control_reason = lambda *a, **k: spec["control"]
    if spec["gateway"]:
        publish = vision_process._private_json

        def published(path, value):  # the custody receipt written after the handoff, right before the create
            publish(path, value)
            if path.name == "receipt.json" and not value["custody"]["operation_id"]:
                now[0] += spec["claim_sec"]

        vision_process._private_json = published
        gateway = transport.ensure_owned_gateway()
        create = gateway.create_model_operation

        def created(*args, **kwargs):
            try:
                return create(*args, **kwargs)
            finally:
                now[0] += spec["create_sec"]

        gateway.create_model_operation = created
    if spec["failure"] == "before_reservation":
        def unrelated(_request):
            raise RuntimeError("unrelated failure with no receipt")

        ua.reserve_attempt = unrelated
    if spec["failure"] == "release":  # the ledger cannot record that a claimed attempt went unsent
        ua.release_pre_dispatch_attempt = lambda *_a, **_k: False
    if spec["failure"] == "after_create":
        observe = transport._ModelInvocation.observe_operation

        def failing(self, *, accepted=False, phase=""):
            if accepted:
                raise RuntimeError("unrelated failure after the create")
            return observe(self, accepted=accepted, phase=phase)

        transport._ModelInvocation.observe_operation = failing

    def respond(request):
        event("http", path=request.url.path, timeout=dict(request.extensions["timeout"]))
        return httpx.Response(200, json={
            "id": "generation", "object": "chat.completion", "created": 1, "model": "vision-fixture",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "one pixel"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2}})

    net_transport.remote_httpx_transport = lambda *a, **k: httpx.MockTransport(respond)
    pricing.estimate_cost_optional = lambda *a, **k: None
    ua._reservation_cost = lambda request: 1.0
    LLMClient._get_supported_parameters = lambda *a, **k: None

    def offline(*_args, **_kwargs):
        raise OSError("the fixture child has no network")

    socket.socket.connect = offline


def at(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


@pytest.fixture
def vlm(tmp_path, monkeypatch):
    from ouroboros.tools import shell

    data, events, spec = tmp_path / "data", tmp_path / "events.jsonl", tmp_path / "child.json"
    data.mkdir()
    state = {"child_timeouts": [], "payloads": [], "terminals": []}
    original = shell._tracked_subprocess_run

    def run(argv, **kwargs):
        state["child_timeouts"].append(kwargs["timeout"])
        state["payloads"].append(json.loads(Path(argv[-1]).read_text(encoding="utf-8")))
        child = list(argv)
        child[2] = f"import runpy; runpy.run_path({__file__!r})['_install_child']({str(spec)!r}); " + argv[2]
        result = original(child, **kwargs)
        lines = [line for line in str(result.stdout or "").splitlines() if line.strip()]
        state["terminals"].append(json.loads(lines[-1]) if lines else None)
        return result

    monkeypatch.setattr(shell, "_tracked_subprocess_run", run)
    monkeypatch.setattr(deadline_utils, "utc_now", lambda: datetime.fromtimestamp(NOW, timezone.utc))
    monkeypatch.setattr(deadline_utils, "time", SimpleNamespace(time=lambda: NOW))
    for key, value in {"OUROBOROS_DATA_DIR": str(data), "OUROBOROS_SETTINGS_PATH": str(data / "settings.json"),
                       "TOTAL_BUDGET": "100", "OUROBOROS_PER_TASK_COST_USD": "100",
                       "OUROBOROS_FINALIZATION_GRACE_SEC": "120", "OUROBOROS_VISION_CAPTION_TIMEOUT_SEC": "300",
                       "OPENAI_COMPATIBLE_BASE_URL": "https://vision.invalid/v1",
                       "OPENAI_COMPATIBLE_API_KEY": "fixture"}.items():
        monkeypatch.setenv(key, value)

    def prepare_child(*, child_start=NOW, recording_sec=0.0, model=MODEL, **child):
        spec.write_text(json.dumps({"child_start": child_start, "recording_sec": recording_sec, "events": str(events),
                                    "gateway": model == SUBSCRIPTION, "mode": "success", "claim_sec": 0, "create_sec": 0,
                                    "control": "", "failure": "", **child}), encoding="utf-8")

    def call(*, deadline=None, model=MODEL, tool=vision._vlm_query, **child):
        prepare_child(model=model, **child)
        ctx = SimpleNamespace(task_id="image-task", task_metadata={"deadline_at": at(deadline)} if deadline else {},
                              event_queue=None, task_attempt=None, _accumulated_usage=None, current_task_type="task",
                              browser_state=SimpleNamespace(last_screenshot_b64=PNG))
        arguments = {} if tool is vision._analyze_screenshot else {"image_base64": PNG}
        with ua.usage_scope(ua.UsageScope(drive_root=data, task_id="image-task", root_task_id="image-task")):
            return tool(ctx, "Describe this pixel", model=model, **arguments)

    return SimpleNamespace(call=call, prepare_child=prepare_child, state=state, events=events, data=data)


def sent(env):
    return [row["timeout"] for row in _events(env.events) if row["kind"] == "http"]


def test_the_child_sends_with_what_is_left_of_the_selected_window(vlm):
    """The tool selects 120 s; starting the child takes 5 s and recording its candidate 10 s more. The child
    used to keep the whole 120 s socket timeout: only that relative number crossed the process boundary."""
    assert "one pixel" in vlm.call(deadline=DEADLINE, child_start=NOW + 5, recording_sec=10)
    assert [timeout["read"] for timeout in sent(vlm)] == [105.0]
    assert vlm.state["child_timeouts"] == [120.0 + 30]  # the parent's own wait is unchanged


@pytest.mark.parametrize("start,recording,ledger", [
    (5, 120, ["released"]),  # recording its candidate spends the rest: the reserved attempt is released
    (130, 0, []),            # its start-up alone outlasts the window: nothing is even reserved
])
def test_a_window_spent_before_the_child_sends_sends_nothing(vlm, start, recording, ledger):
    """Before the fix the child sent anyway, with the whole 120 s socket timeout, after the window had ended."""
    result = vlm.call(deadline=DEADLINE, child_start=NOW + start, recording_sec=recording)
    assert sent(vlm) == []
    assert "VLM_QUERY_FAILED" in result and "deadline" in result
    assert [row["state"] for row in ledger_rows(vlm.data)] == ledger
    assert vlm.state["terminals"][0]["kind"] == "error"  # this route's IPC is unchanged


def test_an_ordinary_child_keeps_its_plain_failure_for_any_refusal(vlm):
    result = vlm.call(control="owner_pause")
    assert "VLM_QUERY_FAILED" in result and "owner_pause" in result and sent(vlm) == []
    assert vlm.state["terminals"][0]["kind"] == "error"


def test_without_a_deadline_the_socket_timeout_never_becomes_a_lifetime(vlm):
    assert "one pixel" in vlm.call(child_start=NOW + 5, recording_sec=10_000)
    assert [timeout["read"] for timeout in sent(vlm)] == [300.0]
    assert "caller_dispatch_deadline_ts" not in vlm.state["payloads"][0]


def gateway_calls(env):
    return [row["kind"] for row in _events(env.events) if row["kind"] in {"upload", "create", "generation", "ack", "cancel"}]


def ledger_states(env):
    return [row["state"] for row in ledger_rows(env.data)]


def test_an_unbounded_subscription_call_still_gets_no_bound(vlm):
    """No task deadline and no execution window: no child timeout and no deadline crosses to the child."""
    assert "pixels read exactly once" in vlm.call(model=SUBSCRIPTION, child_start=NOW + 10_000)
    assert vlm.state["child_timeouts"] == [None]
    assert "caller_dispatch_deadline_ts" not in vlm.state["payloads"][0]
    assert gateway_calls(vlm) == ["upload", "create", "generation", "ack"]


@pytest.mark.parametrize("deadline,bound", [(None, None), (DEADLINE, 1120.0)])
def test_the_pause_adjusted_execution_window_stays_with_the_parent(vlm, monkeypatch, deadline, bound):
    """It narrows the parent's own wait; only the wall-clock deadline window ever crosses as an absolute bound."""
    monkeypatch.setattr(vision, "_vision_execution_window", lambda: 50.0)
    assert "pixels read exactly once" in vlm.call(model=SUBSCRIPTION, deadline=deadline, child_start=NOW + 5)
    assert vlm.state["child_timeouts"] == [50.0 + 30]
    assert vlm.state["payloads"][0].get("caller_dispatch_deadline_ts") == bound


def test_a_subscription_call_with_time_left_runs_and_its_window_never_cancels_it(vlm):
    """15 s of the 120 s are spent before a timely create; 200 s more pass while the operation runs. The bound
    only admits the send: the running operation keeps its custody, polls to its end and settles."""
    text = vlm.call(model=SUBSCRIPTION, deadline=DEADLINE, child_start=NOW + 5, recording_sec=10,
                    mode="slow", create_sec=200)
    assert "pixels read exactly once" in text
    assert vlm.state["child_timeouts"] == [120.0 + 30]
    assert vlm.state["payloads"][0]["caller_dispatch_deadline_ts"] == 1120.0
    assert gateway_calls(vlm) == ["upload", "create", "generation", "ack"]
    assert ledger_states(vlm) == ["settled"]


@pytest.fixture
def slow_parent(monkeypatch):
    """The parent's clock moves 3 s at every read, and its execution-window lookup takes 3 s more."""
    clock = [NOW]

    def read():
        clock[0] += 3
        return clock[0] - 3

    monkeypatch.setattr(deadline_utils, "utc_now", lambda: datetime.fromtimestamp(read(), timezone.utc))
    monkeypatch.setattr(deadline_utils, "time", SimpleNamespace(time=read))
    monkeypatch.setattr(vision, "_vision_execution_window", lambda: clock.__setitem__(0, clock[0] + 3) or 50.0)
    return clock


@pytest.mark.parametrize("model", [MODEL, SUBSCRIPTION], ids=["api", "subscription"])
def test_the_child_window_is_the_deadline_less_the_reserve_however_long_the_parent_takes(vlm, slow_parent, model):
    """The bound used to be re-projected at a later parent read as now + remaining, carrying the time between past
    the window. The instant itself, 1300 − 180 = 1120, now crosses, and the child takes no reserve again: started
    at 1020, the API child sends with exactly 100 s."""
    text = vlm.call(model=model, deadline=DEADLINE, child_start=NOW + 20)
    assert [timeout["read"] for timeout in sent(vlm)] == ([100.0] if model == MODEL else [])
    assert ("one pixel" if model == MODEL else "pixels read exactly once") in text and slow_parent[0] >= NOW + 6
    assert vlm.state["payloads"][0]["caller_dispatch_deadline_ts"] == 1120.0


@pytest.mark.parametrize("model", [MODEL, SUBSCRIPTION], ids=["api", "subscription"])
def test_a_child_starting_just_past_that_window_sends_nothing(vlm, slow_parent, model):
    """At 1121 the window has ended, though a bound re-projected after the parent's own reads had not."""
    result = vlm.call(model=model, deadline=DEADLINE, child_start=1121)
    assert "VLM_QUERY_FAILED" in result and "deadline" in result
    assert sent(vlm) == [] and "create" not in gateway_calls(vlm) and ledger_states(vlm) == []


@pytest.mark.parametrize("owned,spent,capture,ledger,gateway", [
    # Start-up alone outlasts the window: nothing is reserved or uploaded.
    (False, {"child_start": NOW + 130}, None, [], []),
    # Recording its candidate spends the rest: the claim refuses and the reservation is released.
    (False, {"child_start": NOW + 5, "recording_sec": 120}, "released", ["released"], ["upload"]),
    # The owner boundary is a separate witness; after the claim the custody receipt spends the window.
    (True, {"child_start": NOW + 5, "claim_sec": 120}, "released", ["released"], ["upload"]),
], ids=["startup-bare", "recording-bare", "before-create-owner"])
def test_a_subscription_window_spent_before_the_create_is_an_unsent_tool_failure(
        vlm, owned, spent, capture, ledger, gateway):
    """The call's selected window is not the task's: admission past it is already a tool failure, and so is expiry
    while the child prepares. A real wait owner neither rewraps nor retries it and still reads no control afterwards.
    The capture is what happened: none before a reservation, a real release after one."""
    owner = (model_wait.task_model_wait_scope(task={"id": "image-task"}, drive_root=vlm.data, event_queue=None,
                                              worker_slot_held=False) if owned else contextlib.nullcontext())
    with owner as live:
        result = vlm.call(model=SUBSCRIPTION, deadline=DEADLINE, **spent)
        assert live is None or live.control_reason() is None
    assert "VLM_QUERY_FAILED" in result and "_PhysicalSendNotStarted" in result and "deadline" in result
    terminal = vlm.state["terminals"][0]
    assert (terminal["kind"], terminal["unknown"], terminal["control_reason"]) == ("send_refused", False, "deadline")
    assert (terminal["capture"] or {}).get("state") == capture
    assert getattr(ua.last_physical_attempt_capture(), "state", None) == capture
    assert gateway_calls(vlm) == gateway and ledger_states(vlm) == ledger
    assert len(vlm.state["payloads"]) == 1  # no retry


def test_the_screenshot_tool_reports_the_same_unsent_failure(vlm):
    result = vlm.call(model=SUBSCRIPTION, deadline=DEADLINE, child_start=NOW + 130, tool=vision._analyze_screenshot)
    assert "VLM_ANALYSIS_FAILED" in result and "_PhysicalSendNotStarted" in result
    assert vlm.state["terminals"][0]["kind"] == "send_refused" and gateway_calls(vlm) == []


@pytest.mark.parametrize("offset,claim_sec,capture", [(10, 0, None), (-5, 10, "released")],
                         ids=["startup", "before_create"])
def test_the_task_finishes_its_remaining_work_after_a_refused_send(vlm, monkeypatch, offset, claim_sec, capture):
    """A real loop round under a real wait owner, on the real clock with the deadline far off: only the child's own
    clock spends its window. The next round reads the failure and the model answers, with no forced finish."""
    from ouroboros.task_results import write_task_result
    from tests.test_billing_authority_unavailable import _loop_driver

    monkeypatch.setattr(deadline_utils, "utc_now", lambda: datetime.now(timezone.utc))
    monkeypatch.setattr(deadline_utils, "time", time)
    deadline = time.time() + 1800
    vlm.prepare_child(model=SUBSCRIPTION, child_start=deadline - 120 - 2 * 30 + offset, claim_sec=claim_sec)
    tid = "image-task"
    write_task_result(vlm.data, tid, "running", root_task_id=tid)
    loop, registry, _ = _loop_driver(vlm.data, monkeypatch, tid)
    registry._ctx.is_direct_chat = True
    registry._ctx.task_metadata = {"root_task_id": tid, "deadline_at": at(deadline)}
    sends, transcripts = [], []

    def model(call):
        first = not sends
        transcripts.append(list(call.messages))
        response = {"role": "assistant", "content": None if first else "Answered without the image."}
        if first:
            response["tool_calls"] = [{"id": "vlm-1", "type": "function", "function": {
                "name": "vlm_query", "arguments": json.dumps({
                    "prompt": "Describe this pixel", "image_base64": PNG, "model": SUBSCRIPTION})}}]
        ua.execute_physical_attempt(
            ua.AttemptRequest(model="fixture", provider="fixture", reservation_usd=0.1),
            lambda: sends.append(response) or response,
            extractor=lambda _response: ({"prompt_tokens": 1, "completion_tokens": 1}, 0.01, True))
        return response, 0.01, call.active_context_mode

    monkeypatch.setattr(loop, "_call_round_model", model)
    task = {"id": tid, "root_task_id": tid, "_attempt": 1, "_is_direct_chat": True}
    with ua.usage_scope(ua.UsageScope(drive_root=vlm.data, task_id=tid, root_task_id=tid,
                                      root_limit_usd=20, global_limit_usd=1000)), \
            model_wait.task_model_wait_scope(task=task, drive_root=vlm.data, event_queue=None, worker_slot_held=False):
        text, usage, trace = loop.run_llm_loop(
            messages=[{"role": "user", "content": "Describe this pixel"}], tools=registry,
            llm=SimpleNamespace(default_model=lambda: "fixture"), drive_logs=vlm.data / "logs",
            emit_progress=lambda *_a, **_k: None, incoming_messages=queue.Queue(),
            task_id=tid, drive_root=vlm.data, task_type="presence")
    assert text == "Answered without the image." and len(sends) == 2
    assert "forced_finalization" not in trace and usage.get("reason_code") != "deadline_local"
    tool_results = [str(row.get("content")) for row in transcripts[1] if row.get("role") == "tool"]
    assert len(tool_results) == 1 and "VLM_QUERY_FAILED" in tool_results[0] and "_PhysicalSendNotStarted" in tool_results[0]
    terminal = vlm.state["terminals"][0]
    assert terminal["kind"] == "send_refused" and (terminal["capture"] or {}).get("state") == capture
    assert "create" not in gateway_calls(vlm) and len(vlm.state["payloads"]) == 1


def test_a_lost_create_rejoins_its_operation_past_the_window(vlm):
    """Its first create may have been accepted, so repeating it is no new send and claims nothing unsent."""
    text = vlm.call(model=SUBSCRIPTION, deadline=DEADLINE, child_start=NOW + 5, mode="lose_create", create_sec=200)
    assert "pixels read exactly once" in text
    assert gateway_calls(vlm) == ["upload", "create", "generation", "create", "ack"]
    assert ledger_states(vlm) == ["settled"]


def on_the_task_rail(error, root, monkeypatch):
    """The task loop's real reading of a decoded control; only its forced reply's composition is stubbed."""
    from ouroboros import loop_round_limits as limits

    monkeypatch.setattr(limits, "_drain_incoming_messages", lambda *_a, **_k: {})
    monkeypatch.setattr(limits._loop(), "_finalize_forced_services", lambda *_a: None)
    monkeypatch.setattr(limits._loop(), "_forced_fallback_result", lambda *_a, **_k: ("", {}, {}))
    ctx = SimpleNamespace(status_drive_root=root, drive_root=root, task_id="image-task", accumulated_usage={},
                          llm_trace={}, messages=[], incoming_messages=None, event_queue=None, owner_msg_seen=set(),
                          tools=SimpleNamespace(_ctx=None))
    return ctx.accumulated_usage, limits._handle_model_wait_control(ctx, error)[2]["forced_finalization"]


def test_the_task_owners_own_deadline_control_keeps_its_rail(vlm, monkeypatch):
    """Same word, other owner: the task's deadline reaches the child as its parent's control and ends the task there."""
    with model_wait.task_model_wait_scope(task={"id": "image-task"}, drive_root=vlm.data, event_queue=None,
                                          worker_slot_held=False) as owner:
        reads = itertools.count()  # the waiter's first read admits the call; its watcher's next one sees the deadline
        monkeypatch.setattr(owner, "control_reason", lambda **_k: None if next(reads) == 0 else "deadline")
        with pytest.raises(ModelWaitInterrupted) as interrupted:
            vlm.call(model=SUBSCRIPTION, deadline=DEADLINE, child_start=NOW + 5)
    terminal = vlm.state["terminals"][0]
    assert (terminal["kind"], terminal["control_reason"], terminal["capture"]["state"]) == (
        "not_dispatched", "deadline", "released")
    assert interrupted.value.control_reason == "deadline" and "create" not in gateway_calls(vlm)
    usage, forced = on_the_task_rail(interrupted.value, vlm.data, monkeypatch)
    assert usage["reason_code"] == "deadline_local" and forced["control_reason"] == "deadline"


def test_another_typed_refusal_crosses_with_its_own_reason(vlm):
    """With no bound or wait owner its reservation takes no control check: the claim refuses, then releases it."""
    with pytest.raises(ModelWaitInterrupted) as refused:
        vlm.call(model=SUBSCRIPTION, control="owner_pause")
    assert refused.value.control_reason == "owner_pause" and vlm.state["terminals"][0]["kind"] == "interrupted"
    assert refused.value.physical_attempt_capture.state == "released" and not outcome_unknown_on_chain(refused.value)
    assert vlm.state["child_timeouts"] == [None]
    assert gateway_calls(vlm) == ["upload"] and ledger_states(vlm) == ["released"]


@pytest.mark.parametrize("failure,capture,gateway,ledger", [
    ("before_reservation", None, [], []),  # missing custody alone is no evidence that nothing was sent
    ("after_create", "unresolved", ["upload", "create", "generation"], ["unresolved"]),
])
def test_an_unrelated_child_failure_stays_an_unknown_outcome(vlm, failure, capture, gateway, ledger):
    with pytest.raises(ClaudexorModelError) as caught:
        vlm.call(model=SUBSCRIPTION, deadline=DEADLINE, child_start=NOW + 5, failure=failure)
    assert caught.value.code == "model_outcome_unknown" and not isinstance(caught.value, ClaudexorModelNotDispatched)
    assert vlm.state["terminals"][0]["kind"] == "error"
    assert getattr(getattr(caught.value, "physical_attempt_capture", None), "state", None) == capture
    assert gateway_calls(vlm) == gateway and ledger_states(vlm) == ledger
    assert len(vlm.state["payloads"]) == 1  # no retry


def test_a_refusal_with_an_attempt_it_could_not_release_stays_unknown(vlm):
    """Nothing was created, but the ledger holds an open attempt, so no unsent outcome is claimed over it."""
    with pytest.raises(ClaudexorModelError) as caught:
        vlm.call(model=SUBSCRIPTION, deadline=DEADLINE, child_start=NOW + 5, claim_sec=120, failure="release")
    assert caught.value.code == "model_outcome_unknown" and vlm.state["terminals"][0]["kind"] == "error"
    assert caught.value.physical_attempt_capture.state == "unresolved" and outcome_unknown_on_chain(caught.value)
    assert "create" not in gateway_calls(vlm) and ledger_states(vlm) == ["unresolved"]


@pytest.mark.parametrize("forged", [
    {"state": "dispatched"}, {"unknown": True},
], ids=["dispatched", "unknown_flag"])
def test_a_malformed_refused_send_stays_an_unknown_outcome(monkeypatch, forged):
    from ouroboros.tools import shell, vision_process

    def forge(argv, **_kwargs):
        capture = asdict(ua.PhysicalAttemptCapture("a", SUBSCRIPTION, "claudexor", forged.get("state", "released"), "opaque"))
        return SimpleNamespace(stdout=json.dumps({
            "receipt_id": json.loads(Path(argv[-1]).read_text(encoding="utf-8"))["_receipt_id"], "custody": None,
            "capture": capture, "kind": "send_refused", "text": "", "usage": {}, "ledger_attempt_ids": [], "error": "forged",
            "problem": {}, "operation_id": "", "model_role": "vision", "route": {}, "unknown": forged.get("unknown", False),
            "control_reason": forged.get("control_reason", "deadline"), "model_result": None}), stderr="", returncode=0)

    monkeypatch.setattr(shell, "_tracked_subprocess_run", forge)
    with pytest.raises(ClaudexorModelError) as caught:
        vision_process.run_vision_child(child_timeout=1.0, subscription=True, model_role="vision")
    assert caught.value.code == "model_outcome_unknown"
