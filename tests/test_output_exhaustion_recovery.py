"""Empty length replies: actual loop/SDK recovery, exact sources and checkpoint boundaries.

The endpoint is scripted, not evidence that shrinking fixes any real provider.
Only dispatch counts, preservation and transitions are asserted.
"""
import json
from dataclasses import replace

import httpx
import openai
import pytest

from ouroboros import loop, loop_output_exhaustion as recovery
from ouroboros.loop_messages import CONTEXT_FACTS_NAME, _record_owner_directive
from ouroboros.context_budget import bound_host_row
from ouroboros.llm import LLMClient
from tests.test_context_fit_v664 import _plan
from tests.test_context_source_view import BODY, _tool, _observation
from tests.test_output_exhaustion import _loop_kwargs, _events, _incident_reply, _ScriptedLLM, no_sleep as no_sleep
from tests.test_transport_b_stream_deadlines import isolated as isolated, chunk, sse

PRIMARY = "openai-compatible::primary-fixture"
FALLBACK = "openai-compatible::fallback-fixture"


@pytest.fixture
def endpoint(monkeypatch):
    opened = []
    def make(script, *, stream=True, fallbacks=FALLBACK):
        for key, value in {"OPENAI_COMPATIBLE_API_KEY": "fixture-not-live",
                           "OPENAI_COMPATIBLE_BASE_URL": "https://synthetic.invalid/v1",
                           "OUROBOROS_MODEL": PRIMARY, "OUROBOROS_MODEL_FALLBACKS": fallbacks,
                           "OUROBOROS_TASK_REVIEW_MODE": "off", "OUROBOROS_MAX_ROUNDS": "40"}.items():
            monkeypatch.setenv(key, value)
        script, sent = list(script), []
        def respond(request):
            body = json.loads(request.content)
            assert request.url.host == "synthetic.invalid"
            sent.append(body)
            assert script, "unexpected additional generation"
            step = script.pop(0)
            if callable(step):
                return step(request)
            if step == "400":
                return httpx.Response(400, json={"error": {"message": "Upstream request failed"}})
            finish = "length" if step in ("empty", "visible-length") else "stop"
            message = {"role": "assistant", "content": "" if step == "empty" else "done"}
            if step == "empty":
                message["reasoning_content"] = "FAILED-REASONING-MUST-NOT-REPLAY"
            usage = {"prompt_tokens": 100, "completion_tokens": 50,
                     "completion_tokens_details": {"reasoning_tokens": 50 if step == "empty" else 0}, "cost": 0}
            if body.get("stream"):
                return httpx.Response(200, headers={"content-type": "text/event-stream"},
                    content=sse(chunk(message, finish, usage=usage, model=body["model"])))
            return httpx.Response(200, json={"id": f"fake-{len(sent)}", "object": "chat.completion",
                "model": body["model"], "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage})
        http = httpx.Client(transport=httpx.MockTransport(respond))
        sdk = openai.OpenAI(api_key="fixture-not-live", base_url="https://synthetic.invalid/v1", max_retries=0, http_client=http)
        opened.append(sdk)
        client = LLMClient()
        monkeypatch.setattr(client, "_get_remote_client", lambda *a, **k: sdk)
        def no_proxy_client(*_args, **_kwargs):
            # The production caller closes this dedicated client after a caption.
            caption_http = httpx.Client(transport=httpx.MockTransport(respond))
            caption_sdk = openai.OpenAI(api_key="fixture-not-live", base_url="https://synthetic.invalid/v1",
                                       max_retries=0, http_client=caption_http)
            opened.append(caption_sdk)
            return caption_sdk, caption_http
        monkeypatch.setattr(client, "_make_no_proxy_client", no_proxy_client)
        chat = client.chat
        monkeypatch.setattr(client, "chat", lambda **kw: chat(**dict(kw, stream=stream)))
        return client, sent
    yield make
    for sdk in opened:
        sdk.close()


def configured_loop(tmp_path, monkeypatch, client, *, sources=False, exact=False):
    notes = []
    kw = _loop_kwargs(tmp_path, client, notes)
    ctx = kw["tools"]._ctx
    ctx.task_model_override, ctx.exact_model_route = PRIMARY, exact
    ctx.context_fit_plan = replace(_plan(preferred="max", window=1_000_000, known=True), model=PRIMARY)
    def rebind(plan, tools, _messages, *, model, use_local, **kwargs):
        rebound = replace(plan, model=model, route_fp=f"route-{model}", model_role=kwargs.get("model_role", "main"))
        tools._ctx.context_fit_plan = rebound
        return rebound, "max"
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", rebind)
    if sources:
        kw["messages"] = [{"role": "system", "content": "Required memory and books: KEEP."},
                          {"role": "user", "content": "OWNER WORDS: keep every source."},
                          *_tool("seen", invocation="seen-invocation")]
        from ouroboros.tool_policy import initial_tool_schemas
        # A recorded send carries its tool list too: a missing one would read as a changed one.
        ctx._last_context_observation = {**_observation(kw["messages"]),
                                         "tool_schemas": initial_tool_schemas(kw["tools"], context_mode="max")}
        kw["messages"] += [*_tool("unseen", invocation="unseen-invocation", text="UNSEEN-KEEP " * 1500),
            bound_host_row(CONTEXT_FACTS_NAME, "OLD-HOST " * 5000),
            bound_host_row(CONTEXT_FACTS_NAME, "CURRENT-HOST-KEEP")]
    return kw, notes


@pytest.mark.parametrize("stream", [False, True])
def test_informed_chance_then_one_smaller_real_request_preserves_sources(tmp_path, monkeypatch, isolated, endpoint, stream):
    client, sent = endpoint(["empty", "empty", "answer"], stream=stream)
    kw, notes = configured_loop(tmp_path, monkeypatch, client, sources=True)
    text, usage, trace = loop.run_llm_loop(**kw)
    assert text == "done" and len(sent) == 3
    assert {s["model"] for s in sent} == {"primary-fixture"}
    assert len(json.dumps(sent[2]["messages"])) < len(json.dumps(sent[1]["messages"]))
    last = json.dumps(sent[-1]["messages"], ensure_ascii=False)
    for keep in ("Required memory and books: KEEP.", "OWNER WORDS: keep every source.", "UNSEEN-KEEP", "[CONTEXT_FACTS]"):
        assert keep in last
    assert BODY not in last and "OLD-HOST OLD-HOST" not in last
    assert "empty-output recovery" in last and "exposure proves no understanding" in last
    assert all("FAILED-REASONING-MUST-NOT-REPLAY" not in json.dumps(s) for s in sent)
    assert sent[2]["max_tokens"] <= sent[1]["max_tokens"] and {s["reasoning_effort"] for s in sent} == {"medium"}
    events = [r for r in _events(tmp_path, "context_reclaim") if r.get("cause") == "output_exhaustion"]
    assert [(r["rung"], r["status"]) for r in events] == [("host_copies", "applied"), ("bodies", "applied")]
    from ouroboros.artifacts import read_actor_source_bytes
    snapshots = [json.loads(read_actor_source_bytes(tmp_path, "t-exhaust-loop", e["checkpoint_ref"])) for e in events]
    assert any(BODY == r.get("content") for snap in snapshots for r in snap["messages"])
    assert recovery.KEY not in usage and "forced_finalization" not in trace
    assert any("smaller request" in note for note in notes)


@pytest.mark.parametrize("tail", [["answer"], ["empty", "answer"], ["400", "answer"]])
def test_configured_fallbacks_are_disclosed_and_tried_once(tmp_path, monkeypatch, isolated, endpoint, tail):
    client, sent = endpoint(["empty", "empty", *tail], fallbacks=f"{FALLBACK},openai-compatible::third-fixture")
    kw, notes = configured_loop(tmp_path, monkeypatch, client)
    text, usage, _trace = loop.run_llm_loop(**kw)
    assert text == "done" and len(sent) == 2 + len(tail)
    assert [s["model"] for s in sent] == ["primary-fixture", "primary-fixture", "fallback-fixture", "third-fixture"][:len(sent)]
    assert len([n for n in notes if "Fallback:" in n]) == len(tail)
    assert recovery.KEY not in usage


@pytest.mark.parametrize("exact", [True, False])
def test_all_routes_empty_save_incomplete_without_wrapup(tmp_path, monkeypatch, isolated, endpoint, exact):
    client, sent = endpoint(["empty"] * (2 if exact else 4), fallbacks=f"{FALLBACK},openai-compatible::third-fixture")
    kw, _notes = configured_loop(tmp_path, monkeypatch, client, exact=exact)
    kw["messages"] += [{"role": "assistant", "content": "Saved finding, not a completed answer."}]
    text, usage, trace = loop.run_llm_loop(**kw)
    assert len(sent) == (2 if exact else 4)
    assert text == "Saved finding, not a completed answer."
    assert usage["reason_code"] == recovery.REASON and usage["execution_status"] == "infra_failed"
    assert "Work is incomplete" in usage["terminal_host_notice"] and trace["output_recovery"]["stage"] == "exhausted"
    from ouroboros.outcomes import derive_loop_outcome
    outcome = derive_loop_outcome(text, usage, trace)
    assert outcome["outcome_axes"]["execution"]["status"] == "infra_failed"
    from ouroboros.owner_continue import continuation_eligibility
    assert continuation_eligibility({"status": "failed", "reason_code": recovery.REASON, "root_task_id": "r"}, "r")["eligible"]


def test_opaque_400_is_not_window_evidence_or_output_recovery(tmp_path, monkeypatch, isolated, endpoint):
    client, sent = endpoint(["400", "answer"])
    kw, _notes = configured_loop(tmp_path, monkeypatch, client)
    text, usage, _ = loop.run_llm_loop(**kw)
    assert text == "done" and len(sent) == 2
    assert recovery.KEY not in usage and not _events(tmp_path, "context_reclaim")


@pytest.mark.parametrize("smaller", [False, True], ids=["ordinary_refusal", "smaller_request_refusal"])
def test_a_numbered_refusal_caps_the_route_on_either_send(tmp_path, monkeypatch, isolated, endpoint, smaller):
    from ouroboros import capability_evidence as ce, context_fit as cf
    from ouroboros.context_window import resolve_window

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)

    def refuse(_request):
        return httpx.Response(400, json={"error": {"code": "context_length_exceeded", "message": (
            "This model's maximum context length is 200000 tokens. However, you requested 210000 tokens "
            "(200000 in the messages, 10000 in the completion).")}})

    client, sent = endpoint(["empty", "empty", refuse, "answer"] if smaller else [refuse, "answer"])
    kw, _notes = configured_loop(tmp_path, monkeypatch, client, sources=True)
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr(ce, "probe", lambda _root, **k: ce.CapabilityEvidence(1_000_000, "confirmed", "metadata",
        ce.route_fingerprint(**{name: k[name] for name in ("provider", "model", "base_url", "options")})))
    core = cf.ContextCore("Required memory and books: KEEP.", "", "", "", "", "", '"Assignment"', False)
    plan = cf.build_context_fit_plan(None, core, {}, preferred_mode="max", route_resolver=cf.resolve_context_fit_route)
    kw["tools"]._ctx.context_fit_plan = plan
    assert resolve_window(plan, 65_536).usable == 934_464 and plan.window_witness_bound
    text, _usage, _ = loop.run_llm_loop(**kw)
    assert text == "done" and len(sent) == (4 if smaller else 2)
    final = resolve_window(plan, 65_536)
    assert final.usable == 134_464 and final.source.startswith("refusal")  # the provider's stated 200,000


@pytest.mark.parametrize("predicted", [False, True], ids=["empty_replies", "predicted_smaller_send"])
def test_empty_replies_and_a_predicted_smaller_send_record_no_refusal(tmp_path, monkeypatch, isolated, endpoint,
                                                                       predicted):
    from ouroboros import capability_evidence as ce, context_fit as cf
    from ouroboros.context_anchor import REFUSED_INPUTS_KEY
    from ouroboros.context_window import record_observation
    from ouroboros.loop_llm_call import REFUSED_CANDIDATE_KEY

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    client, sent = endpoint(["empty", "empty", "empty"])
    kw, _notes = configured_loop(tmp_path, monkeypatch, client, sources=True)
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr(ce, "resolve_main_token_density", lambda *_a, **_kw: (1.0, "cold_estimate"))
    monkeypatch.setattr(ce, "probe", lambda _root, **k: ce.CapabilityEvidence(1_000_000, "confirmed", "metadata",
        ce.route_fingerprint(**{name: k[name] for name in ("provider", "model", "base_url", "options")})))
    core = cf.ContextCore("Required memory and books: KEEP.", "", "", "", "", "", '"Assignment"', False)
    plan = cf.build_context_fit_plan(None, core, {}, preferred_mode="max", route_resolver=cf.resolve_context_fit_route)
    kw["tools"]._ctx.context_fit_plan = plan
    send_smaller = recovery._send_smaller

    class Inspected(BaseException):
        pass

    def inspect(ctx, cost):
        if predicted:  # another task on this route and account learned a tighter ceiling before this send
            record_observation(tmp_path, plan.window_record.route_fp, refused=70_000)
        send_smaller(ctx, cost)
        rows = ce._load(tmp_path).get("main_windows", {}).get(plan.window_record.route_fp, {}).get("observations", [])
        assert len(sent) == (2 if predicted else 3)
        assert [r["refused"] for r in rows if r.get("refused")] == ([70_000] if predicted else [])
        if predicted:
            assert ctx.accumulated_usage[REFUSED_CANDIDATE_KEY]["predicted_free_tokens"] < 0
        assert REFUSED_INPUTS_KEY not in ctx.accumulated_usage  # neither is a provider-refused input
        raise Inspected

    monkeypatch.setattr(recovery, "_send_smaller", inspect)
    with pytest.raises(Inspected):
        loop.run_llm_loop(**kw)


@pytest.mark.parametrize("stage", ["informed", "fallback", "exhausted"])
def test_only_new_owner_input_resets_episode(tmp_path, monkeypatch, stage):
    from tests.test_loop_compaction import _ctx
    ctx = _ctx(tmp_path)
    ctx.context_fit_plan = None  # no sizing measurement in this input-authority test
    _record_owner_directive(ctx.tools._ctx, source="owner_mailbox", content="continue", msg_id="owner-1")
    record = {"stage": stage, "owner_revision": 1, "tried": []}
    ctx.accumulated_usage[recovery.KEY] = record
    _record_owner_directive(ctx.tools._ctx, source="principal_task_message", content="a peer", msg_id="peer-1")
    _record_owner_directive(ctx.tools._ctx, source="owner_mailbox", content="continue", msg_id="owner-1")
    def fail(_ctx):
        _ctx.accumulated_usage["_last_llm_error_kind"] = recovery.KIND
        return None, 0, "max"
    recovery.call(ctx, fail)
    assert ctx.accumulated_usage[recovery.KEY] is record
    _record_owner_directive(ctx.tools._ctx, source="owner_mailbox", content="use new evidence", msg_id="owner-2")
    recovery.call(ctx, fail)
    assert ctx.accumulated_usage[recovery.KEY] is not record
    assert ctx.accumulated_usage[recovery.KEY]["owner_revision"] == 2


def persistent_loop(tmp_path, monkeypatch, client, *, target="t-output-recovery", handoff=None, sources=False):
    from tests.test_working_checkpoint import _registry
    from ouroboros.task_results import write_task_result
    kw, notes = configured_loop(tmp_path, monkeypatch, client, sources=sources)
    old = kw["tools"]._ctx
    kw["tools"] = _registry(tmp_path, monkeypatch, target, 2 if handoff else 1)
    ctx = kw["tools"]._ctx
    ctx.context_fit_plan, ctx.task_model_override, ctx.working_recovery = old.context_fit_plan, PRIMARY, handoff
    ctx._last_context_observation = getattr(old, "_last_context_observation", None)
    kw["task_id"] = target
    write_task_result(tmp_path, target, "running", root_task_id=target)
    return kw, notes


@pytest.mark.parametrize("new_id", [False, True])
@pytest.mark.parametrize("lost_at", ["informed", "fallback", "informed_send", "fallback_send", "smaller", "smaller_send", "fallback_informed"])
def test_real_checkpoint_recovery_never_reopens_spent_step(tmp_path, monkeypatch, isolated, endpoint, new_id, lost_at):
    from ouroboros import working_checkpoint as wc
    from tests.test_stream_timeout_recovery import WorkerLost
    def die(_request):
        raise WorkerLost()
    script = {"informed_send": ["empty", die, "answer"], "fallback_send": ["empty", "empty", die, "answer"],
              "smaller_send": ["empty", "empty", die, "answer"], "fallback_informed": ["400", "empty", "answer"]}
    client, sent = endpoint(script.get(lost_at, ["empty", "empty", "answer"]))
    sources = lost_at.startswith("smaller")
    kw, _ = persistent_loop(tmp_path, monkeypatch, client, sources=sources)
    original = wc.save_model_call
    def save_then_lose(**kwargs):
        original(**kwargs)
        _raw, state = wc.read_working(tmp_path, kw["task_id"], 1)
        record = state["usage"].get(recovery.KEY, {})
        stage = "informed" if lost_at == "fallback_informed" else lost_at
        if record.get("stage") == stage and not record.get("pending"):
            raise WorkerLost()
    with monkeypatch.context() as crash:
        crash.setattr(wc, "save_model_call", save_then_lose)
        with pytest.raises(WorkerLost):
            loop.run_llm_loop(**kw)
    before = len(sent)
    _, state = wc.read_working(tmp_path, kw["task_id"], 1)
    if "informed" in lost_at:
        expected = loop._output_exhausted_notice({"sent_allowance_tokens": 65536, "reasoning_tokens": 50})
        assert any(row.get("content") == expected for row in state["messages"])
    if lost_at == "fallback_informed":
        assert state["route"]["active_model"] == FALLBACK
    target = kw["task_id"] + "-successor" if new_id else kw["task_id"]
    task = {"id": target}
    assert wc.attach_recovery(tmp_path, task, source_task_id=kw["task_id"], from_attempt=1, cause="worker_crash")
    resumed, _ = persistent_loop(tmp_path, monkeypatch, client, target=target, handoff=task["_working_recovery"])
    text, usage, trace = loop.run_llm_loop(**resumed)
    if lost_at.endswith("_send"):
        assert len(sent) == before and trace["output_recovery"]["interrupted"]
        assert usage["reason_code"] == recovery.REASON and "unconfirmed" in text
    else:
        assert text == "done" and len(sent) == 3
        if lost_at == "fallback_informed":
            assert [s["model"] for s in sent] == ["primary-fixture", "fallback-fixture", "fallback-fixture"]


@pytest.mark.parametrize("tail", ["answer", "empty"])
def test_failed_smaller_request_reaches_reserve_without_second_shrink(tmp_path, monkeypatch, isolated, endpoint, tail):
    client, sent = endpoint(["empty", "empty", "empty", tail])
    kw, notes = configured_loop(tmp_path, monkeypatch, client, sources=True)
    text, usage, _trace = loop.run_llm_loop(**kw)
    assert [s["model"] for s in sent] == ["primary-fixture"] * 3 + ["fallback-fixture"]
    assert len(_events(tmp_path, "context_reclaim")) == 2
    assert len([n for n in notes if "Fallback:" in n]) == 1
    assert text == "done" if tail == "answer" else usage["reason_code"] == recovery.REASON


def test_round_limit_before_recovery_exhausted_keeps_ordinary_wrapup(tmp_path, monkeypatch, no_sleep):
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    llm = _ScriptedLLM(_incident_reply(0))
    text, usage, trace = loop.run_llm_loop(**_loop_kwargs(tmp_path, llm, []))
    assert llm.calls == 2 and usage["reason_code"] == "round_limit"
    assert trace["forced_finalization"]["source"] != "output_exhaustion_no_recap"
    assert text == "done"


@pytest.mark.parametrize("timing", ["before_send", "during_send"])
def test_warm_pause_distinguishes_unsent_from_unknown(tmp_path, monkeypatch, isolated, endpoint, timing):
    from ouroboros import owner_pause, owner_wait, review_pause
    from ouroboros.model_wait import ModelWaitInterrupted
    from ouroboros.task_results import load_task_result
    from ouroboros.model_wait import task_model_wait_scope
    CHILD = "t-output-recovery"
    from types import SimpleNamespace
    import time
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    def pause(*_args):
        owner_pause.install_fence(tmp_path, CHILD, request_id="output-pause")
        raise ModelWaitInterrupted("owner_pause")
    llm, sent = endpoint(["empty", "answer"])
    kw, _ = persistent_loop(tmp_path, monkeypatch, llm)
    ctx = kw["tools"]._ctx
    ctx.owner_wait_callback, ctx.pending_events = owner_wait.direct_owner_wait, []
    original, interrupted = llm.chat, []
    def pre_send_pause(**kwargs):
        if timing == "before_send" and sent and not interrupted:
            interrupted.append(True)
            pause()
        result = original(**kwargs)
        if timing == "during_send" and len(sent) == 2:
            pause()  # sender finished, but its receiver left before incorporating the answer
        return result
    monkeypatch.setattr(llm, "chat", pre_send_pause)
    monkeypatch.setattr(review_pause, "live_detached_operations", lambda _: [{"owner_id": "fixture-review", "surface": "test"}])
    parked = []
    def resume(_seconds):
        row = load_task_result(tmp_path, CHILD)["owner_wait"]
        parked.append(row)
        owner_pause.release_fence(tmp_path, CHILD, reason="owner_resume")
    monkeypatch.setattr(owner_wait, "time", SimpleNamespace(time=time.time, monotonic=time.monotonic, sleep=resume))
    with task_model_wait_scope(task={"id": CHILD, "_attempt": 1, "budget_drive_root": str(tmp_path)},
                               drive_root=tmp_path, event_queue=None, worker_slot_held=False) as waiter:
        ctx.model_wait_context, waiter.tool_context = waiter, ctx
        text, usage, trace = loop.run_llm_loop(**kw)
    assert parked and len(sent) == 2
    if timing == "during_send":
        assert trace["output_recovery"]["interrupted"] and "outcome is unconfirmed" in usage["terminal_host_notice"]
    else:
        assert text == "done" and recovery.KEY not in usage
    assert load_task_result(tmp_path, CHILD)["owner_wait"]["state"] == "resumed"


def test_unknown_after_an_empty_reply_uses_existing_no_resend_rail(tmp_path, monkeypatch, no_sleep):
    from tests.test_output_exhaustion import _death
    from tests.test_stream_timeout_recovery import run_child
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    llm = _ScriptedLLM(_incident_reply(0), _death)
    # End only the passive network wait; classification and the no-resend terminal remain real.
    monkeypatch.setattr(loop, "_transport_wait_step", lambda *_a, **_k: False)
    text, usage, trace = run_child(tmp_path, monkeypatch, llm)
    assert llm.calls == 2
    assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"


def test_usable_tool_round_resets_recovery_without_replaying_tool(tmp_path, monkeypatch, no_sleep):
    from tests.test_stream_timeout_recovery import run_child
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    tool = ({"role": "assistant", "content": "", "tool_calls": [{"id": "one-tool", "type": "function",
        "function": {"name": "list_available_tools", "arguments": "{}"}}]}, {"cost": 0})
    llm = _ScriptedLLM(_incident_reply(0), tool, _incident_reply(1),
                       ({"role": "assistant", "content": "Recovered after tool", "tool_calls": []}, {"cost": 0}))
    text, usage, trace = run_child(tmp_path, monkeypatch, llm)
    assert text == "Recovered after tool" and llm.calls == 4
    assert recovery.KEY not in usage and len(trace["tool_calls"]) == 1


@pytest.mark.parametrize("control", ["stop", "deadline"])
def test_outer_control_keeps_its_reason_without_recovery_or_paid_recap(tmp_path, monkeypatch, no_sleep, control):
    import datetime as dt
    from ouroboros.owner_mailbox import write_owner_message, KIND_FINALIZE_NOW
    from ouroboros.loop_round_limits import REASON_OWNER_STOPPED_DIRECT_TURN
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    kw = _loop_kwargs(tmp_path, None, [])
    class ControlledLLM(_ScriptedLLM):
        def chat(self, **kwargs):
            result = super().chat(**kwargs)
            if control == "stop":
                assert write_owner_message(tmp_path, REASON_OWNER_STOPPED_DIRECT_TURN,
                    kw["task_id"], msg_id="stop-output", kind=KIND_FINALIZE_NOW)
            else:
                kw["tools"]._ctx.task_metadata = {"deadline_at":
                    (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)).isoformat()}
            return result
    llm = ControlledLLM(_incident_reply(0))
    kw["llm"] = llm
    text, usage, trace = loop.run_llm_loop(**kw)
    assert llm.calls == 1
    # Existing control owners keep their own vocabulary, rather than the output-recovery cause.
    assert usage["reason_code"] == ("owner_requested_finalization" if control == "stop" else "deadline_local")
    if control == "deadline":
        assert trace["forced_finalization"]["source"] == "deadline_local_window_elapsed"
    else:
        assert kw["tools"]._ctx._skip_post_task_synthesis is True


@pytest.mark.parametrize("kind", ["auth_error", "quota_exhausted", "transport_unavailable", "deadline_exhausted", "provider_outcome_unknown"])
def test_other_failure_rails_are_not_relabelled_as_output_exhaustion(kind):
    usage = {recovery.KEY: {"stage": "fallback"}, "_last_llm_error_kind": kind}
    assert recovery.next_step(usage) == ""


def test_ordinary_failure_does_not_add_a_checkpoint_write(tmp_path, monkeypatch):
    from tests.test_loop_compaction import _ctx
    ctx = _ctx(tmp_path)
    ctx.accumulated_usage["_last_llm_error_kind"] = "transport_unavailable"
    monkeypatch.setattr(recovery, "_save", lambda *_: pytest.fail("unrelated failure checkpointed"))
    assert recovery.call(ctx, lambda _: (None, None, "max")) == (None, None, "max")


def test_transient_error_of_informed_chance_keeps_its_rail_and_one_send(tmp_path, monkeypatch, isolated, endpoint):
    client, sent = endpoint(["empty", lambda _: httpx.Response(503, json={"error": {"message": "unavailable"}})], fallbacks="")
    kw, _ = configured_loop(tmp_path, monkeypatch, client)
    monkeypatch.setattr(loop, "_transport_wait_step", lambda *_a, **_k: False)
    text, usage, trace = loop.run_llm_loop(**kw)
    assert len(sent) == 2 and usage["reason_code"] == "provider_unavailable"
    assert trace["forced_finalization"]["source"] != "output_exhaustion_no_recap"
    assert usage[recovery.KEY]["last_failure_kind"] == "provider_transient"


def test_output_terminal_reaches_benchmark_disclosure_without_becoming_success():
    from devtools.benchmarks.common.result_index import runtime_terminal_disclosure
    assert runtime_terminal_disclosure({"status": "failed", "reason_code": recovery.REASON})["truncated"] is True


@pytest.mark.parametrize("recovering", [False, True])
def test_vision_caption_does_not_consume_main_recovery_claim(tmp_path, monkeypatch, isolated, endpoint, recovering):
    """The actual caption/SDK send is a helper, not the pending Main alternative."""
    import base64
    import io
    from PIL import Image
    from ouroboros import vision_routing as vr
    vision = "openai-compatible::vision-fixture"
    client, sent = endpoint([*(["empty", "empty"] if recovering else []), "answer", "answer"])
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", vision)
    monkeypatch.setattr(loop, "_transport_wait_step", lambda *_a, **_k: False)
    monkeypatch.setattr(vr, "_image_input_verdict", lambda model, **_kw: model != FALLBACK)
    monkeypatch.setattr(vr, "_metadata_says_no", lambda _model: "fixture catalog lists no image input")
    image = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(image, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
    kw, _ = persistent_loop(tmp_path, monkeypatch, client)
    if not recovering:
        ctx = kw["tools"]._ctx
        ctx.task_model_override = FALLBACK
        ctx.context_fit_plan = replace(ctx.context_fit_plan, model=FALLBACK)
    kw["messages"] = [{"role": "user", "content": [
        {"type": "text", "text": "Read this image."}, {"type": "image_url", "image_url": {"url": url}}]}]
    text, usage, _trace = loop.run_llm_loop(**kw)
    expected = (["primary-fixture"] * 2 if recovering else []) + ["vision-fixture", "fallback-fixture"]
    assert [row["model"] for row in sent] == expected
    assert text == "done" and recovery.KEY not in usage
    assert "image_url" not in json.dumps(sent[-1]["messages"])
    assert "done" in json.dumps(sent[-1]["messages"])
