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
_UNSEEN_KEEP = "UNSEEN-KEEP " * 1500
_OLD_HOST = "OLD-HOST " * 5000


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
            # A tuple step is (visible text, provider finish[, typed tool calls]).
            text, finish, *calls = step if isinstance(step, tuple) else (
                "" if step == "empty" else "done", "length" if step in ("empty", "visible-length") else "stop")
            message = {"role": "assistant", "content": text, **({"tool_calls": calls[0]} if calls else {})}
            if step == "empty":
                message["reasoning_content"] = "FAILED-REASONING-MUST-NOT-REPLAY"
            usage = {"prompt_tokens": 100, "completion_tokens": 50,
                     "completion_tokens_details": {"reasoning_tokens": 50 if step == "empty" else 0}, "cost": 0}
            if body.get("stream"):
                delta = {**message, **({"tool_calls": [dict(call, index=i) for i, call in enumerate(calls[0])]} if calls else {})}
                return httpx.Response(200, headers={"content-type": "text/event-stream"},
                    content=sse(chunk(delta, finish, usage=usage, model=body["model"])))
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
        kw["messages"] += [*_tool("unseen", invocation="unseen-invocation", text=_UNSEEN_KEEP),
            bound_host_row(CONTEXT_FACTS_NAME, _OLD_HOST),
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
    assert len([n for n in notes if "Fallback: trying" in n]) == len(tail)
    assert [n for n in notes if "active route" in n] == [f"⚡ Fallback: openai-compatible::{sent[-1]['model']} is now the active route; it answered."]
    assert recovery.KEY not in usage


def test_a_reply_that_spent_its_allowance_is_adopted_as_no_answer_until_the_primary_returns(
        tmp_path, monkeypatch, isolated, endpoint):
    """Real loop and SDK. The primary refuses; the fallback's empty length reply is adopted as the active route, but
    no row or host fact calls it an answer and no accepted route records it. The primary then answers on its own return."""
    client, sent = endpoint(["400", "empty", "empty", "answer"])
    kw, notes = configured_loop(tmp_path, monkeypatch, client)
    text, usage, _trace = loop.run_llm_loop(**kw)
    assert text == "done" and [s["model"] for s in sent] == ["primary-fixture", "fallback-fixture", "fallback-fixture", "primary-fixture"]
    assert [n for n in notes if "Fallback:" in n] == [
        f'⚡ Fallback: trying {FALLBACK} after a rejected request on {PRIMARY}. The provider said: "Upstream request failed"',
        f"⚡ Fallback: {FALLBACK} is now the active route, with no usable answer after a reply that reached its length limit "
        "before any visible output.",
        f"⚡ Fallback: trying {PRIMARY} (primary) after a reply that reached its length limit before any visible output on {FALLBACK}.",
        f"⚡ Fallback: back on the primary route {PRIMARY}; it answered.",
    ]
    facts = [json.dumps(m["content"], ensure_ascii=False) for m in sent[2]["messages"] if "[ROUTE FACTS]" in str(m.get("content"))]
    assert len(facts) == 1, "the adopted route's next own round carries the facts once"
    assert "is now acting without a usable answer this round after" in facts[0]
    assert "That route has no accepted call this round." in facts[0]
    assert "answered this round" not in facts[0] and "That answering call" not in facts[0]
    assert [row["to_binding"][0] for row in usage["accepted_routes"]] == [PRIMARY], "a spent reply is no accepted route"


@pytest.mark.parametrize("exact", [True, False])
def test_all_routes_empty_save_incomplete_without_wrapup(tmp_path, monkeypatch, isolated, endpoint, exact):
    client, sent = endpoint(["empty"] * (2 if exact else 4), fallbacks=f"{FALLBACK},openai-compatible::third-fixture")
    kw, notes = configured_loop(tmp_path, monkeypatch, client, exact=exact)
    kw["messages"] += [{"role": "assistant", "content": "Saved finding, not a completed answer."}]
    text, usage, trace = loop.run_llm_loop(**kw)
    assert len(sent) == (2 if exact else 4)
    assert text == "Saved finding, not a completed answer."
    assert [n for n in notes if "the last attempt" in n] == ([] if exact else [
        "⚡ Fallback: the last attempt, on openai-compatible::third-fixture, ended without a usable reply after a reply "
        "that reached its length limit before any visible output."])
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


@pytest.mark.parametrize(("lost_at", "new_id"), [
    ("informed", True),
    ("fallback", True),
    ("informed_send", False),
    ("fallback_send", True),
    ("smaller", False),
    ("smaller_send", False),
    ("fallback_informed", True),
])
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
    assert len([n for n in notes if "Fallback: trying" in n]) == 1
    assert text == "done" if tail == "answer" else usage["reason_code"] == recovery.REASON


def test_round_limit_before_recovery_exhausted_keeps_ordinary_wrapup(tmp_path, monkeypatch, no_sleep):
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    llm = _ScriptedLLM(_incident_reply(0))
    text, usage, trace = loop.run_llm_loop(**_loop_kwargs(tmp_path, llm, []))
    assert llm.calls == 2 and usage["reason_code"] == "round_limit"
    assert trace["forced_finalization"]["source"] != "output_exhaustion_no_recap"
    assert text == "done"


@pytest.mark.parametrize("first", ["empty", "cut"])  # either episode's informed send takes the same claim
@pytest.mark.parametrize("timing", ["before_send", "during_send"])
def test_warm_pause_distinguishes_unsent_from_unknown(tmp_path, monkeypatch, isolated, endpoint, timing, first):
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
    llm, sent = endpoint([CUTS["length"] if first == "cut" else "empty", "answer"])
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
        assert text == "done" and not {recovery.KEY, recovery.UNFINISHED_KEY} & set(usage)
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


@pytest.mark.parametrize("kind", ["auth_error", "deadline_exhausted", "provider_outcome_unknown"])
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


# --- text with no typed call that its provider typed as cut: one informed turn (#1472) -------------------
# The SDK lane writes the choice's finish; the loop reads only that typed fact, never the reply's words.

PARTIAL = "The probe finished: both files were read and the first count is 12, while the second"
LATER = "count is 12 as well, so both"
CUTS = {"length": (PARTIAL, "length"), "tool_calls": ("I'll read the recent chat history first.", "tool_calls")}


def _cut_notices(messages):
    return [index for index, row in enumerate(messages) if row.get("role") == "user"
            and "The provider ended your previous reply" in json.dumps(row.get("content"), ensure_ascii=False)]


def _kept_at(messages, words):
    return [index for index, row in enumerate(messages) if row.get("role") == "assistant"
            and words in json.dumps(row.get("content"), ensure_ascii=False)]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("cut", list(CUTS))
def test_typed_cut_keeps_its_text_and_buys_one_ordinary_turn(tmp_path, monkeypatch, isolated, endpoint, cut, stream):
    """Real SDK and loop: callless text a `length` or tool-call finish ended is no final. The next ordinary
    request on the same route keeps those exact words and adds one host fact naming the finish; its answer is
    the final. No configured route, smaller view or second allowance is involved."""
    words, finish = CUTS[cut]
    client, sent = endpoint([CUTS[cut], "answer"], stream=stream)
    kw, notes = configured_loop(tmp_path, monkeypatch, client)
    text, usage, trace = loop.run_llm_loop(**kw)
    assert text == "done" and len(sent) == 2 and {s["model"] for s in sent} == {"primary-fixture"}
    assert _cut_notices(sent[0]["messages"]) == [] and _kept_at(sent[0]["messages"], words) == []
    [kept], [notice] = _kept_at(sent[1]["messages"], words), _cut_notices(sent[1]["messages"])
    assert kept + 1 == notice and f"(finish_reason={finish})" in json.dumps(sent[1]["messages"][notice])
    assert sent[1]["max_tokens"] == sent[0]["max_tokens"] and not _events(tmp_path, "context_reclaim")
    assert not [note for note in notes if "Fallback" in note]
    assert recovery.UNFINISHED_KEY not in usage and "forced_finalization" not in trace
    from ouroboros.outcomes import derive_loop_outcome
    assert derive_loop_outcome(text, usage, trace)["outcome_axes"]["execution"]["status"] == "ok"


@pytest.mark.parametrize("second", list(CUTS))
def test_a_repeated_typed_cut_ends_incomplete_with_its_text_kept(tmp_path, monkeypatch, isolated, endpoint, second):
    """The bound: one informed turn per episode. Another typed cut ends the work with no third send, no recap and no
    configured route; its words are the result and the host notice says the work is incomplete."""
    client, sent = endpoint([CUTS["length"], (LATER, second)])
    kw, notes = configured_loop(tmp_path, monkeypatch, client)
    text, usage, trace = loop.run_llm_loop(**kw)
    assert len(sent) == 2 and text == LATER and not [note for note in notes if "Fallback" in note]
    assert (usage["reason_code"], usage["execution_status"], usage["terminal_origin"]) == (
        recovery.REASON, "infra_failed", "host_salvage")
    assert "Work is incomplete" in usage["terminal_host_notice"] and f"(finish_reason={second})" in usage["terminal_host_notice"]
    assert (trace["output_recovery"]["stage"], trace["output_recovery"]["finishes"]) == ("exhausted", ["length", second])
    assert trace["forced_finalization"]["source"] == "output_exhaustion_no_recap"
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.project_dialogue import TASK_CAUSE_PHRASES, _completion_verdict
    axes = derive_loop_outcome(text, usage, trace)["outcome_axes"]
    assert axes["execution"]["status"] == "infra_failed"
    # The card's cause line is the shared incomplete-reply sentence; the exact finish stays in the host notice.
    stored = {"status": "failed", "reason_code": usage["reason_code"], "outcome_axes": axes}
    assert _completion_verdict(stored, {}) == f"{TASK_CAUSE_PHRASES[recovery.REASON]}."


@pytest.mark.parametrize("words", ["done", '<tool_call>\n{"name": "chat_history", "arguments": {"count": 5}}\n</tool_call>',
                                   '```json\n{"tool": "chat_history", "arguments": {"count": 5}}\n```'])
def test_stop_finished_text_stays_the_plain_final_whatever_it_quotes(tmp_path, monkeypatch, isolated, endpoint, words):
    """Negative control: under `stop` the reply is the final, a call quoted in its text included; nothing reads it."""
    client, sent = endpoint([(words, "stop")])
    kw, _notes = configured_loop(tmp_path, monkeypatch, client)
    text, usage, trace = loop.run_llm_loop(**kw)
    assert (text, len(sent)) == (words, 1)
    assert recovery.UNFINISHED_KEY not in usage and "forced_finalization" not in trace and not trace["tool_calls"]


def test_a_whole_finish_task_under_an_outer_length_cut_keeps_its_waiver(tmp_path, monkeypatch, isolated, endpoint):
    """A typed completion call is a typed call, not callless text: no informed turn is bought and the request
    selects its answer as before (the forced rail's disclosed-cut waiver: tests/test_forced_completion_request.py)."""
    answer = "Wrap-up: both counts are 12."
    call = {"id": "final", "type": "function", "function": {
        "name": "finish_task", "arguments": json.dumps({"action": "finish", "answer": answer})}}
    client, sent = endpoint([("", "length", [call])], stream=False)
    kw, _notes = configured_loop(tmp_path, monkeypatch, client)
    text, usage, trace = loop.run_llm_loop(**kw)
    assert (text, len(sent)) == (answer, 1)
    assert recovery.UNFINISHED_KEY not in usage and "forced_finalization" not in trace
    assert [(row["tool"], bool(row.get("is_error"))) for row in trace["tool_calls"]] == [("finish_task", False)]


@pytest.mark.parametrize("new_id", [False, True])
def test_a_spent_typed_cut_survives_checkpoint_recovery(tmp_path, monkeypatch, isolated, endpoint, new_id):
    """The kept words, the host fact and the episode reach the next ready checkpoint. Work lost right after it
    resumes there (same execution or a new one) with the turn already spent: a second typed cut ends it."""
    from ouroboros import working_checkpoint as wc
    from tests.test_stream_timeout_recovery import WorkerLost
    client, sent = endpoint([CUTS["length"], (LATER, "length")])
    kw, _ = persistent_loop(tmp_path, monkeypatch, client)
    original = wc.save_ready
    def save_then_lose(limit_ctx):
        original(limit_ctx)
        if (limit_ctx.accumulated_usage.get(recovery.UNFINISHED_KEY) or {}).get("stage") == "informed":
            raise WorkerLost()
    with monkeypatch.context() as crash:
        crash.setattr(wc, "save_ready", save_then_lose)
        with pytest.raises(WorkerLost):
            loop.run_llm_loop(**kw)
    _, state = wc.read_working(tmp_path, kw["task_id"], 1)
    assert len(sent) == 1 and state["usage"][recovery.UNFINISHED_KEY]["stage"] == "informed"
    assert _kept_at(state["messages"], PARTIAL) and _cut_notices(state["messages"])
    target = kw["task_id"] + "-successor" if new_id else kw["task_id"]
    task = {"id": target}
    assert wc.attach_recovery(tmp_path, task, source_task_id=kw["task_id"], from_attempt=1, cause="worker_crash")
    resumed, _ = persistent_loop(tmp_path, monkeypatch, client, target=target, handoff=task["_working_recovery"])
    text, usage, trace = loop.run_llm_loop(**resumed)
    assert len(sent) == 2 and text == LATER and usage["reason_code"] == recovery.REASON
    assert trace["output_recovery"]["finishes"] == ["length", "length"]


def _resume(tmp_path, monkeypatch, client, kw, new_id=False):
    from ouroboros import working_checkpoint as wc
    target = kw["task_id"] + "-successor" if new_id else kw["task_id"]
    task = {"id": target}
    assert wc.attach_recovery(tmp_path, task, source_task_id=kw["task_id"], from_attempt=1, cause="worker_crash")
    resumed, _ = persistent_loop(tmp_path, monkeypatch, client, target=target, handoff=task["_working_recovery"])
    return loop.run_llm_loop(**resumed)


def _lose_worker_after_candidate(crash):
    """Lose the worker as the no-call consumer returns, before any later boundary."""
    from tests.test_stream_timeout_recovery import WorkerLost
    consumer, calls = loop._finalize_loop_candidate, []
    def lose(*args, **kwargs):
        calls.append(consumer(*args, **kwargs))
        raise WorkerLost()
    crash.setattr(loop, "_finalize_loop_candidate", lose)
    return calls


@pytest.mark.parametrize("tail", ["answer", "repeated_cut"])
@pytest.mark.parametrize("new_id", [False, True])
@pytest.mark.parametrize("cut", list(CUTS))
def test_work_lost_right_after_a_typed_cut_keeps_it_and_its_owed_turn(
        tmp_path, monkeypatch, isolated, endpoint, cut, new_id, tail):
    """Real loop and SDK. Work lost as the no-call consumer returns, before any later boundary, resumes (same
    execution or a new one) from the exact reply, its replay fields, its one host fact and the one turn it is owed:
    not a fresh episode. The next send carries each once, and its own typed cut ends the work with no third send."""
    import copy
    from ouroboros import working_checkpoint as wc
    from tests.test_stream_timeout_recovery import WorkerLost
    words, finish = CUTS[cut]
    replay = {"reasoning_content": "KEPT-REASONING", "reasoning_details": [{"type": "reasoning.text", "text": "KEPT"}],
              "nativeContinuation": {"route": {"source": "fixture"}, "state": "opaque"}}
    exact = {"role": "assistant", "content": words, **replay}  # lanes that retain them carry these on the normalized reply
    client, sent = endpoint([CUTS[cut], "answer" if tail == "answer" else (LATER, "length")], stream=False, fallbacks="")
    requested, chat = [], client.chat
    def with_replay_fields(**call):
        requested.append(copy.deepcopy(call["messages"]))
        reply, response_usage = chat(**call)
        return ({**reply, **replay} if len(sent) == 1 else reply), response_usage
    monkeypatch.setattr(client, "chat", with_replay_fields)
    kw, _ = persistent_loop(tmp_path, monkeypatch, client)
    with monkeypatch.context() as crash:
        calls = _lose_worker_after_candidate(crash)
        with pytest.raises(WorkerLost):
            loop.run_llm_loop(**kw)
    assert calls == [None] and len(sent) == 1
    _, state = wc.read_working(tmp_path, kw["task_id"], 1)
    record = state["usage"].get(recovery.UNFINISHED_KEY) or {}
    assert (record.get("stage"), record.get("finishes"), "pending" in record) == ("informed", [finish], False)
    [kept], [notice] = _kept_at(state["messages"], words), _cut_notices(state["messages"])
    assert kept + 1 == notice and state["working"]["boundary"] == "ready", "a ready state repeats its logical round"
    assert state["messages"][kept] == exact
    text, usage, trace = _resume(tmp_path, monkeypatch, client, kw, new_id)
    assert len(sent) == 2, "the owed turn is sent once and the cut reply is not generated again"
    [kept] = _kept_at(requested[1], words)
    assert requested[1][kept] == exact, "the restored transcript hands the model call the reply as it was written"
    [kept], [notice] = _kept_at(sent[1]["messages"], words), _cut_notices(sent[1]["messages"])
    assert kept + 1 == notice and sent[1]["messages"][kept]["content"] == words
    assert f"(finish_reason={finish})" in json.dumps(sent[1]["messages"][notice])
    if tail == "answer":
        assert text == "done" and recovery.UNFINISHED_KEY not in usage and "output_recovery" not in trace
    else:
        assert text == LATER and usage["reason_code"] == recovery.REASON
        assert trace["output_recovery"]["finishes"] == [finish, "length"]


def _flaky_checkpoint_file(monkeypatch):
    """Only the physical file write fails, once per entry of the returned list; every holder above it stays real."""
    from ouroboros import utils, working_checkpoint as wc
    real_write, failing = utils.write_bytes_atomic, []
    def write(path, data, **kwargs):
        if path.name.startswith(wc.FILE_PREFIX) and failing:
            failing.pop()
            raise OSError(28, "checkpoint disk full")
        return real_write(path, data, **kwargs)
    monkeypatch.setattr(utils, "write_bytes_atomic", write)
    return failing


def _second_cut_returned(tmp_path, monkeypatch, endpoint, *, peer_mail=False, end_unsaved=False):
    """Real loop and SDK: a repeated cut ends the work and ``run_llm_loop`` returns it with no caller to store it, as when
    the worker is lost before its terminal write. ``peer_mail``: content the informed round drained stays unacknowledged
    past its failed ready save, so the loop's exit saves it (``save_terminal``). ``end_unsaved``: the cut's own save fails."""
    from ouroboros.owner_mailbox import write_task_message
    client, sent = endpoint([CUTS["length"], (LATER, "tool_calls"), "answer"], stream=False, fallbacks="")
    kw, _ = persistent_loop(tmp_path, monkeypatch, client)
    monkeypatch.setattr(loop, "_transport_wait_step", lambda *_a, **_k: False)  # a send past the script ends, never waits
    failing, consumer, chat = _flaky_checkpoint_file(monkeypatch), loop._finalize_loop_candidate, client.chat
    def after_the_first_cut(*args, **kwargs):
        result = consumer(*args, **kwargs)
        if result is None and peer_mail:
            assert write_task_message(tmp_path, "A peer's context.", kw["task_id"], source_task_id="t-peer",
                                      provenance="peer_task", msg_id="peer-mail")
            failing.append("the informed round's ready save")
        return result
    def after_the_second_reply(**call):
        reply = chat(**call)
        if end_unsaved and len(sent) == 2:
            failing.append("the ended cut's own save")
        return reply
    monkeypatch.setattr(loop, "_finalize_loop_candidate", after_the_first_cut)
    monkeypatch.setattr(client, "chat", after_the_second_reply)
    returned = loop.run_llm_loop(**kw)
    monkeypatch.setattr(loop, "_finalize_loop_candidate", consumer)
    return client, sent, kw, returned


@pytest.mark.parametrize("peer_mail", [False, True], ids=["ordinary_return", "saved_by_the_loop_exit"])
@pytest.mark.parametrize("new_id", [False, True])
def test_work_lost_after_the_loop_returned_a_repeated_cut_ends_as_it_did(tmp_path, monkeypatch, isolated, endpoint, new_id, peer_mail):
    """Real producer, loop exit and recovery. Work lost after the loop returned a repeated cut and before the caller's
    terminal write resumes (same execution or a new one) to the same end: the latest words and finishes, no third send."""
    from ouroboros import working_checkpoint as wc
    client, sent, kw, (returned, _, returned_trace) = _second_cut_returned(tmp_path, monkeypatch, endpoint, peer_mail=peer_mail)
    assert returned == LATER and len(sent) == 2 and returned_trace["output_recovery"]["finishes"] == ["length", "tool_calls"]
    _, state = wc.read_working(tmp_path, kw["task_id"], 1)
    record = state["usage"][recovery.UNFINISHED_KEY]
    assert (record["stage"], record["finishes"]) == ("exhausted", ["length", "tool_calls"]) and "pending" not in record
    assert _kept_at(state["messages"], LATER) and _kept_at(state["messages"], PARTIAL)
    text, usage, trace = _resume(tmp_path, monkeypatch, client, kw, new_id)
    assert len(sent) == 2, "a third generation was bought"
    assert (text, usage["reason_code"], usage["execution_status"], usage["terminal_origin"]) == (
        LATER, recovery.REASON, "infra_failed", "host_salvage")
    assert trace["output_recovery"]["finishes"] == ["length", "tool_calls"] and "interrupted" not in trace["output_recovery"]
    assert "(finish_reason=tool_calls)" in usage["terminal_host_notice"] and "unconfirmed" not in usage["terminal_host_notice"]


@pytest.mark.parametrize("new_id", [False, True])
def test_a_repeated_cut_that_could_not_be_saved_stays_its_unconfirmed_claim(tmp_path, monkeypatch, isolated, endpoint, new_id):
    """Control: when the file refuses the ended cut, what the checkpoint holds is still the informed send's claim. Work
    lost then is that unconfirmed send, kept words and the honest notice, and never a third one."""
    client, sent, kw, (returned, _, _) = _second_cut_returned(tmp_path, monkeypatch, endpoint, end_unsaved=True)
    assert returned == LATER and len(sent) == 2
    text, usage, trace = _resume(tmp_path, monkeypatch, client, kw, new_id)
    assert len(sent) == 2 and text == PARTIAL and usage["reason_code"] == recovery.REASON
    assert trace["output_recovery"]["interrupted"] and "outcome is unconfirmed" in usage["terminal_host_notice"]


def test_a_recovered_ended_cut_buys_no_wrap_up(tmp_path, monkeypatch, isolated, endpoint):
    """A paid forced rail reached first on recovery keeps its reason and, like the empty episode's end, buys no recap."""
    client, sent, kw, _ = _second_cut_returned(tmp_path, monkeypatch, endpoint)
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")  # the recovered round is past the limit: the round-limit wrap-up rail
    text, usage, trace = _resume(tmp_path, monkeypatch, client, kw)
    assert len(sent) == 2 and text == LATER and usage["reason_code"] == "round_limit"
    assert trace["output_recovery"]["finishes"] == ["length", "tool_calls"]


def test_a_cut_the_file_refused_is_held_until_saved_before_its_informed_send(tmp_path, monkeypatch, isolated, endpoint):
    """Real loop, SDK and holder; only the file write fails. The cut's own save and the next ready boundary do not gate,
    so the informed send's claim holds on the existing typed hold until the file lands: what leaves was saved first."""
    from ouroboros import budget_pause, working_checkpoint as wc
    from tests.test_output_exhaustion import _events
    monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", 0.001)
    on_disk = []
    def informed_send(request):
        on_disk.append(wc.read_working(tmp_path, kw["task_id"], 1)[1])
        return httpx.Response(200, json={"id": "fake-informed", "object": "chat.completion", "model": json.loads(request.content)["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0}})
    client, sent = endpoint([CUTS["length"], informed_send], stream=False, fallbacks="")
    kw, _ = persistent_loop(tmp_path, monkeypatch, client)
    failing, chat = _flaky_checkpoint_file(monkeypatch), client.chat
    def after_the_cut_arrives(**call):
        reply = chat(**call)
        if len(sent) == 1:  # the cut's own save, the next ready boundary, then two attempts under the claim's hold
            failing.extend(["own save", "ready save", "held claim", "held claim"])
        return reply
    monkeypatch.setattr(client, "chat", after_the_cut_arrives)
    text, usage, _trace = loop.run_llm_loop(**kw)
    assert text == "done" and len(sent) == 2 and not failing, "the loop went on and every refused write was retried"
    [state] = on_disk
    record = state["usage"][recovery.UNFINISHED_KEY]
    assert (record["stage"], record["pending"]) == ("informed", "informed") and record["attempt_id"]
    assert _kept_at(state["messages"], PARTIAL) and _cut_notices(state["messages"])
    assert [row["hold_reason"] for row in _events(tmp_path / "logs", "task_checkpoint") if row.get("checkpoint_kind") == "budget_pause_hold"
            and row.get("hold_reason")] == [wc.HOLD_UNWRITABLE], "the existing hold is what waited, visibly and once"
    assert "budget_pause_hold" not in usage


@pytest.mark.parametrize("new_id", [False, True])
@pytest.mark.parametrize("cut", list(CUTS))
def test_a_lost_informed_turn_is_never_sent_again(tmp_path, monkeypatch, isolated, endpoint, cut, new_id):
    """The informed turn's send takes the same physical claim as every recovery send. Work lost while it is in
    flight resumes (same execution or a new one) with that dispatch unconfirmed: nothing is sent again, the kept
    words are the result and the host notice says the attempt was interrupted and not repeated."""
    from ouroboros import working_checkpoint as wc
    from tests.test_stream_timeout_recovery import WorkerLost
    def lose_worker(_request):
        raise WorkerLost()
    client, sent = endpoint([CUTS[cut], lose_worker, "answer"])
    kw, _ = persistent_loop(tmp_path, monkeypatch, client)
    with pytest.raises(WorkerLost):
        loop.run_llm_loop(**kw)
    _, state = wc.read_working(tmp_path, kw["task_id"], 1)
    text, usage, trace = _resume(tmp_path, monkeypatch, client, kw, new_id)
    assert len(sent) == 2, "the unconfirmed informed send was dispatched again"
    assert state["usage"][recovery.UNFINISHED_KEY]["pending"] == "informed"
    assert text == CUTS[cut][0] and usage["reason_code"] == recovery.REASON
    assert trace["output_recovery"]["interrupted"] and "outcome is unconfirmed" in usage["terminal_host_notice"]


@pytest.mark.parametrize("new_id", [False, True])
@pytest.mark.parametrize("cut", list(CUTS))
def test_an_informed_turn_that_returned_with_no_known_end_is_never_sent_again(tmp_path, monkeypatch, isolated, endpoint, cut, new_id):
    """Real loop and SDK: the informed send's connection dies after dispatch and the call returns normally, with no
    known end of that attempt. The return settles nothing: the claim is what the checkpoint holds, so work lost right
    after it resumes (same execution or a new one) with nothing sent again and the existing unconfirmed notice."""
    from ouroboros import working_checkpoint as wc
    from tests.test_stream_timeout_recovery import WorkerLost
    def connection_dies(_request):
        raise httpx.ReadError("synthetic connection death")
    client, sent = endpoint([CUTS[cut], connection_dies, "answer"], stream=False, fallbacks="")
    kw, _ = persistent_loop(tmp_path, monkeypatch, client)
    original = wc.save_model_call
    def save_then_lose(**kwargs):
        original(**kwargs)
        if len(sent) == 2:  # the save right after that normal return
            raise WorkerLost()
    with monkeypatch.context() as crash:
        crash.setattr(wc, "save_model_call", save_then_lose)
        with pytest.raises(WorkerLost):
            loop.run_llm_loop(**kw)
    _, state = wc.read_working(tmp_path, kw["task_id"], 1)
    text, usage, trace = _resume(tmp_path, monkeypatch, client, kw, new_id)
    assert len(sent) == 2, "the unconfirmed informed send was dispatched again"
    assert state["usage"][recovery.UNFINISHED_KEY]["pending"] == "informed"
    assert text == CUTS[cut][0] and usage["reason_code"] == recovery.REASON
    assert trace["output_recovery"]["interrupted"] and "outcome is unconfirmed" in usage["terminal_host_notice"]


def test_the_execution_that_heard_no_known_end_keeps_its_configured_route(tmp_path, monkeypatch, isolated, endpoint):
    """Control: only a restart meets that claim as interrupted. The execution that heard the call return keeps the
    round's ordinary unknown-outcome rail, here its configured route after one missed upstream observation, whose
    send is claimed as its own."""
    def connection_dies(_request):
        raise httpx.ReadError("synthetic connection death")
    client, sent = endpoint([CUTS["length"], connection_dies, "answer"], stream=False)
    kw, _notes = configured_loop(tmp_path, monkeypatch, client)
    from ouroboros import loop_transport
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda *_a: False)  # the wait's entry step, at once
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable", lambda *_a, **_k: {})
    saved = _spy_pending(monkeypatch, sent, recovery.UNFINISHED_KEY)
    text, usage, trace = loop.run_llm_loop(**kw)
    assert text == "done" and [s["model"] for s in sent] == ["primary-fixture", "primary-fixture", "fallback-fixture"]
    assert [(step, before) for step, _attempt, before in saved] == [("informed", 1), ("informed", 2), ("informed", 2)]
    assert saved[1][1] == saved[0][1] != saved[2][1], "the claim stood after that return; the route's send took its own"
    assert recovery.UNFINISHED_KEY not in usage and "output_recovery" not in trace


@pytest.mark.parametrize("ladder", [["answer"], ["empty", "answer"]], ids=["informed-answers", "reaches-the-route"])
def test_a_lost_answers_new_attempt_that_is_cut_takes_the_ordinary_ladder(tmp_path, monkeypatch, isolated, endpoint, ladder):
    """The new attempt a primary gets after its lost answer is an ordinary attempt. An empty length reply to it
    takes the steps it takes on any round (the informed chance on that route, then the configured route): the
    lost attempt adds no route change of its own and removes none."""
    def connection_dies(_request):
        raise httpx.ReadError("synthetic connection death")
    client, sent = endpoint([connection_dies, "empty", *ladder], stream=False)
    kw, _notes = configured_loop(tmp_path, monkeypatch, client)
    from ouroboros import loop_transport
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda *_a: False)
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable",
                        lambda *_a, **_k: {"kind": "upstream_http", "status_code": 200})
    text, _usage, _trace = loop.run_llm_loop(**kw)
    assert text == "done"
    assert [s["model"] for s in sent] == ["primary-fixture"] * 3 + ["fallback-fixture"] * (len(ladder) - 1)


def _refuse_size(_request):
    return httpx.Response(400, json={"error": {"code": "context_length_exceeded", "message": "Context length exceeded"}})


@pytest.mark.parametrize("first", ["cut", "empty"])
def test_a_size_refused_recovery_send_keeps_the_context_owners_smaller_retry(tmp_path, monkeypatch, isolated, endpoint,
                                                                            first):
    """Real loop and SDK, exact route: the informed send's provider refuses it for size, a known outcome with no
    generation. The context owner's strictly smaller request replaces that attempt in the same step, each send saved
    pending under its own attempt before dispatch, and its answer is the final."""
    key = recovery.UNFINISHED_KEY if first == "cut" else recovery.KEY
    client, sent = endpoint([CUTS["length"] if first == "cut" else "empty", _refuse_size, "answer"],
                            stream=False, fallbacks="")
    kw, _notes = configured_loop(tmp_path, monkeypatch, client, sources=True, exact=True)
    saved, original = [], recovery._save
    def spy(ctx):
        record = ctx.accumulated_usage.get(key) or {}
        if record.get("pending"):
            saved.append((record["pending"], record["attempt_id"], len(sent)))
        original(ctx)
    monkeypatch.setattr(recovery, "_save", spy)
    text, usage, trace = loop.run_llm_loop(**kw)
    assert text == "done" and len(sent) == 3 and {s["model"] for s in sent} == {"primary-fixture"}
    assert len(json.dumps(sent[2]["messages"])) < len(json.dumps(sent[1]["messages"]))
    assert [(step, before) for step, _attempt, before in saved] == [("informed", 1), ("informed", 2)]
    assert len({attempt for _step, attempt, _before in saved}) == 2
    assert not {recovery.KEY, recovery.UNFINISHED_KEY} & set(usage) and "forced_finalization" not in trace


@pytest.mark.parametrize("new_id", [False, True])
def test_a_lost_replacement_of_a_size_refused_send_is_never_sent_again(tmp_path, monkeypatch, isolated, endpoint, new_id):
    """The replacement's own pending attempt is what the checkpoint holds: work lost while it is in flight resumes
    (same execution or a new one) with its outcome unconfirmed, and nothing is sent again."""
    from ouroboros import working_checkpoint as wc
    from tests.test_stream_timeout_recovery import WorkerLost
    def lose_worker(_request):
        raise WorkerLost()
    client, sent = endpoint([CUTS["length"], _refuse_size, lose_worker, "answer"], stream=False)
    kw, _ = persistent_loop(tmp_path, monkeypatch, client, sources=True)
    with pytest.raises(WorkerLost):
        loop.run_llm_loop(**kw)
    _, state = wc.read_working(tmp_path, kw["task_id"], 1)
    record = state["usage"][recovery.UNFINISHED_KEY]
    refused = [row.get("physical_attempt_id") for row in state["usage"].get("llm_call_refs", [])
               if row.get("failure_code") == "context_overflow"]
    assert len(sent) == 3 and record["pending"] == "informed" and len(refused) == 1
    assert record["attempt_id"] and record["attempt_id"] != refused[0]
    text, usage, trace = _resume(tmp_path, monkeypatch, client, kw, new_id)
    assert len(sent) == 3, "the unconfirmed replacement was dispatched again"
    assert text == PARTIAL and usage["reason_code"] == recovery.REASON
    assert trace["output_recovery"]["interrupted"] and "outcome is unconfirmed" in usage["terminal_host_notice"]


def _failure(**facts):
    error = RuntimeError("synthetic provider answer")
    for name, value in facts.items():
        setattr(error, name, value)
    return error


_ENDS = {  # how the claimed attempt "first" ended, as the physical boundary reports it: (attempt, state, status, error)
    "provider_refusal": ("first", "unresolved", 400, _failure()),
    "refusal_frame_before_output": ("first", "unresolved", 400, _failure(
        stream_incomplete=True, stream_receipt={"model_output_received": False})),
    "settled_engine_refusal": ("first", "settled", None, _failure()),
    "never_sent": ("first", "released", None, _failure()),
    "response_read": ("first", "settled", None, None),
    "unpriced_response": ("first", "unresolved", None, None),
    "unknown_outcome": ("first", "unresolved", None, _failure()),
    "output_already_streamed": ("first", "unresolved", 400, _failure(
        stream_incomplete=True, stream_receipt={"model_output_received": True})),
    "declared_timeout": ("first", "unresolved", 504, _failure(explicit_timeout=True)),
    "receiver_abandoned": ("first", "dispatched", 400, _failure(receiver_abandoned=True)),
    "typed_unknown": ("first", "settled", None, _failure(code="model_outcome_unknown")),
    "another_attempt": ("elsewhere", "unresolved", 400, _failure()),
    "not_heard": None,
}


@pytest.mark.parametrize("end", list(_ENDS))
def test_only_a_known_end_of_the_current_attempt_admits_a_replacement(tmp_path, monkeypatch, end):
    """The step's tracker hears each attempt's end at the physical boundary. Only the current attempt's own known end
    (never sent, its response read priced or not, a typed provider or engine answer) admits a replacement, saved
    pending under the new attempt, and only a sent one counts as sent; an unknown end, another attempt's end or none
    keeps the step single-use. Whether a replacement is earned stays with the owners of the refusal."""
    from types import SimpleNamespace
    from tests.test_loop_compaction import _ctx
    from ouroboros import usage_accounting as ua
    from ouroboros.llm_attempt import single_physical_send_active
    from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed
    ctx = _ctx(tmp_path)
    ctx.context_fit_plan = None
    record = {"stage": "informed", "finishes": ["length"], "owner_revision": 0}
    ctx.accumulated_usage[recovery.UNFINISHED_KEY] = record
    saved, trackers = [], []
    monkeypatch.setattr(recovery, "_save", lambda _c: saved.append((record.get("pending"), record.get("attempt_id"))))
    def ordinary(call_ctx):
        tracker = call_ctx.physical_send_claim
        trackers.append(tracker)
        assert not single_physical_send_active(), "the step's tracker leaves the lanes' own repairs in place"
        tracker(SimpleNamespace(attempt_id="first"))
        if _ENDS[end] is not None:
            attempt, state, status, error = _ENDS[end]
            tracker.attempt_ended(ua.PhysicalAttemptCapture(
                attempt_id=attempt, model="m", provider="p", state=state,
                candidate_measurement_kind="canonical_json_v1", provider_status_code=status), error)
        tracker(SimpleNamespace(attempt_id="second"))
        return {"role": "assistant", "content": "done"}, 0, "max"
    if end in {"provider_refusal", "refusal_frame_before_output", "settled_engine_refusal", "never_sent",
               "response_read", "unpriced_response"}:
        assert recovery.call(ctx, ordinary)[0] and saved == [("informed", "first"), ("informed", "second")]
        assert trackers[0].sent is (end != "never_sent"), "a proven-unsent attempt never spends the step"
    else:
        with pytest.raises(PhysicalAttemptPreconditionFailed):
            recovery.call(ctx, ordinary)
        assert saved == [("informed", "first")] and (record["pending"], record["attempt_id"]) == ("informed", "first")


def _image_turn(kw):
    from tests.test_image_refusal_retry import IMAGE_A
    kw["messages"].append({"role": "user", "content": [{"type": "text", "text": "Inspect this image."},
                                                        {"type": "image_url", "image_url": {"url": IMAGE_A}}]})
    return IMAGE_A


def _spy_pending(monkeypatch, sent, key):
    saved, original = [], recovery._save
    def spy(ctx):
        record = ctx.accumulated_usage.get(key) or {}
        if record.get("pending"):
            saved.append((record["pending"], record["attempt_id"], len(sent)))
        original(ctx)
    monkeypatch.setattr(recovery, "_save", spy)
    return saved


def _refuse_image(_request):
    return httpx.Response(400, json={"error": {"code": "image_not_supported",
                                               "message": "Image input is not supported for this model"}})


@pytest.mark.parametrize("first", ["answered", "cut", "empty"])
def test_an_image_refusal_keeps_its_same_round_retry_inside_the_informed_turn(tmp_path, monkeypatch, isolated, endpoint, first):
    """Real loop and SDK, exact route: a completed refusal of a request carrying an image earns its one same-round
    retry without that image, also when that request is the informed turn after a cut or empty reply. The refusal is
    a known end, so the retry replaces the refused attempt inside the step, saved pending under its own attempt first."""
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "inline")
    key = {"cut": recovery.UNFINISHED_KEY, "empty": recovery.KEY}.get(first)
    script = {"cut": [CUTS["length"]], "empty": ["empty"], "answered": []}[first] + [_refuse_image, "answer"]
    client, sent = endpoint(script, stream=False, fallbacks="")
    kw, notes = configured_loop(tmp_path, monkeypatch, client, exact=True)
    image = _image_turn(kw)
    saved = _spy_pending(monkeypatch, sent, key or recovery.UNFINISHED_KEY)
    text, usage, trace = loop.run_llm_loop(**kw)
    assert text == "done" and len(sent) == len(script)
    assert [image in json.dumps(body) for body in sent] == [True] * (len(sent) - 1) + [False]
    assert any("retried with a note" in note for note in notes)
    assert [(step, before) for step, _attempt, before in saved] == ([("informed", 1), ("informed", 2)] if key else [])
    assert len({attempt for _step, attempt, _before in saved}) == len(saved)
    assert not {recovery.KEY, recovery.UNFINISHED_KEY} & set(usage) and "forced_finalization" not in trace


@pytest.mark.parametrize("first", ["cut", "empty"])
def test_a_request_wire_repair_keeps_its_retry_inside_the_informed_turn(tmp_path, monkeypatch, isolated, endpoint, first):
    """Real loop and SDK, exact route: the informed send's provider rejects one request parameter, a typed HTTP 400
    before any output. The lane's own request-wire repair resends without it inside the same step, each attempt saved
    pending under its own id before dispatch, and its answer is the final."""
    key = recovery.UNFINISHED_KEY if first == "cut" else recovery.KEY
    def reject_effort(_request):
        return httpx.Response(400, json={"error": {"message": "Unsupported parameter", "param": "reasoning_effort"}})
    client, sent = endpoint([CUTS["length"] if first == "cut" else "empty", reject_effort, "answer"],
                            stream=False, fallbacks="")
    kw, _notes = configured_loop(tmp_path, monkeypatch, client, exact=True)
    saved = _spy_pending(monkeypatch, sent, key)
    text, usage, trace = loop.run_llm_loop(**kw)
    assert text == "done" and len(sent) == 3 and {s["model"] for s in sent} == {"primary-fixture"}
    assert "reasoning_effort" in sent[1] and "reasoning_effort" not in sent[2]
    assert [(step, before) for step, _attempt, before in saved] == [("informed", 1), ("informed", 2)]
    assert len({attempt for _step, attempt, _before in saved}) == 2
    assert not {recovery.KEY, recovery.UNFINISHED_KEY} & set(usage) and "forced_finalization" not in trace


@pytest.mark.parametrize("new_id", [False, True])
def test_a_lost_image_retry_replacement_is_never_sent_again(tmp_path, monkeypatch, isolated, endpoint, new_id):
    """The image retry's own pending attempt is what the checkpoint holds when work is lost in flight: it resumes,
    in the same execution or a new one, with that dispatch unconfirmed, and nothing is sent again."""
    from ouroboros import working_checkpoint as wc
    from tests.test_stream_timeout_recovery import WorkerLost
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "inline")
    def lose_worker(_request):
        raise WorkerLost()
    client, sent = endpoint([CUTS["length"], _refuse_image, lose_worker, "answer"], stream=False)
    kw, _ = persistent_loop(tmp_path, monkeypatch, client)
    _image_turn(kw)
    with pytest.raises(WorkerLost):
        loop.run_llm_loop(**kw)
    _, state = wc.read_working(tmp_path, kw["task_id"], 1)
    record = state["usage"][recovery.UNFINISHED_KEY]
    refused = [row.get("physical_attempt_id") for row in state["usage"].get("llm_call_refs", [])
               if row.get("status_code") == 400]
    assert len(sent) == 3 and record["pending"] == "informed" and len(refused) == 1
    assert record["attempt_id"] and record["attempt_id"] != refused[0], "the checkpoint holds the replacement's own attempt"
    text, usage, trace = _resume(tmp_path, monkeypatch, client, kw, new_id)
    assert len(sent) == 3, "the unconfirmed replacement was dispatched again"
    assert text == PARTIAL and usage["reason_code"] == recovery.REASON
    assert trace["output_recovery"]["interrupted"] and "outcome is unconfirmed" in usage["terminal_host_notice"]


def test_a_proven_unsent_replacement_cannot_erase_the_send_that_reached_the_provider(tmp_path, monkeypatch, isolated, endpoint):
    """Real loop and SDK: the informed send's refusal reached the provider, then its image retry never left the host
    (a refused connect, released). The step's sent attempt stands: the ladder moves on and the turn is not re-owed."""
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "inline")
    monkeypatch.setattr(loop, "_transport_wait_step", lambda *_a, **_k: False)
    def refuse_connect(_request):
        raise httpx.ConnectError("synthetic connect refusal")
    client, sent = endpoint(["empty", _refuse_image, refuse_connect], stream=False, fallbacks="")
    kw, _notes = configured_loop(tmp_path, monkeypatch, client, exact=True)
    _image_turn(kw)
    text, usage, trace = loop.run_llm_loop(**kw)
    assert len(sent) == 3 and not usage[recovery.KEY].get("pending")
    assert usage[recovery.KEY]["stage"] == "fallback" and usage[recovery.KEY]["last_failure_kind"]


@pytest.mark.parametrize("repeat", [False, True], ids=["tracked_only", "tracked_timeout_repeat"])
def test_a_timeout_repeat_inside_a_tracked_step_keeps_both_obligations(tmp_path, monkeypatch, isolated, endpoint, repeat):
    """Real SDK through the round call: a step's tracker records and hears the attempt either way; a granted timeout
    repeat is additionally the strict single send, so the lane's wire repair does not run for it."""
    from ouroboros import working_checkpoint
    from ouroboros.loop_llm_call import TIMEOUT_REPEAT_KEY, call_llm_with_retry
    def reject_effort(_request):
        return httpx.Response(400, json={"error": {"message": "Unsupported parameter", "param": "reasoning_effort"}})
    client, sent = endpoint([reject_effort, *([] if repeat else ["answer"])], stream=False, fallbacks="")
    (tmp_path / "logs").mkdir(exist_ok=True)
    class Tracker:
        def __init__(self):
            self.claimed, self.heard = [], []
        def __call__(self, reservation):
            self.claimed.append(reservation.attempt_id)
        def attempt_ended(self, capture, error):
            self.heard.append((capture.attempt_id, capture.provider_status_code, error is None))
    tracker = Tracker()
    usage = {"execution_id": "exec", **({TIMEOUT_REPEAT_KEY: {"round_id": "exec:round:1", "first": {}, "state": "granted"}} if repeat else {})}
    monkeypatch.setattr(working_checkpoint, "save_model_call", lambda **_kw: None)
    msg, _cost = call_llm_with_retry(client, [{"role": "user", "content": "hello"}], PRIMARY, None, "low", 1,
                                     tmp_path / "logs", "t-both", 1, None, usage, "task", physical_send_claim=tracker,
                                     attempt_cap=1)
    assert len(sent) == (1 if repeat else 2) and (msg is None) is repeat
    assert len(tracker.claimed) == len(sent) and [attempt for attempt, *_rest in tracker.heard] == tracker.claimed
    assert tracker.heard[0][1:] == (400, False)
    if repeat:
        assert usage[TIMEOUT_REPEAT_KEY]["state"] == "spent"
        assert usage[TIMEOUT_REPEAT_KEY]["repeat"]["physical_attempt_id"] == tracker.claimed[0]


def test_a_refunded_informed_claim_is_still_owed_after_recovery(tmp_path, monkeypatch, isolated, endpoint):
    """Positive control: a claimed send the transport proves unsent (a connect refusal) is refunded and saved, so work
    lost right after that refund resumes with the informed turn still owed, sends it once and takes its answer."""
    from ouroboros import usage_accounting as ua
    from ouroboros import working_checkpoint as wc
    from tests.test_stream_timeout_recovery import WorkerLost
    def refuse_connect(_request):
        raise httpx.ConnectError("synthetic connect refusal")
    client, sent = endpoint([CUTS["length"], refuse_connect, "answer"])
    kw, _ = persistent_loop(tmp_path, monkeypatch, client)
    original, claims = wc.save_model_call, []
    def save_then_lose(**kwargs):
        original(**kwargs)
        claims.append((wc.read_working(tmp_path, kw["task_id"], 1)[1]["usage"].get(recovery.UNFINISHED_KEY) or {}).get("pending"))
        if claims[-2:] == ["informed", None]:
            assert ua.last_physical_attempt_capture().state == "released"
            raise WorkerLost()
    with monkeypatch.context() as crash:
        crash.setattr(wc, "save_model_call", save_then_lose)
        with pytest.raises(WorkerLost):
            loop.run_llm_loop(**kw)
    text, usage, trace = _resume(tmp_path, monkeypatch, client, kw)
    assert claims == ["informed", None] and text == "done" and len(sent) == 3
    assert recovery.UNFINISHED_KEY not in usage and "output_recovery" not in trace


def test_a_usable_informed_reply_clears_its_claim_for_recovery(tmp_path, monkeypatch, isolated, endpoint):
    """Positive control: a usable informed reply (a typed tool call) ends the episode and its claim at the next
    checkpoint; work lost during its tool resumes without an interruption and answers on the next ordinary send."""
    from ouroboros import working_checkpoint as wc
    from tests.test_stream_timeout_recovery import WorkerLost
    tool = {"id": "one-tool", "type": "function", "function": {"name": "list_available_tools", "arguments": "{}"}}
    client, sent = endpoint([CUTS["length"], ("", "tool_calls", [tool]), "answer"], stream=False)
    kw, _ = persistent_loop(tmp_path, monkeypatch, client)
    original = wc.save_before_effects
    def save_then_lose(limit_ctx):
        original(limit_ctx)
        raise WorkerLost()
    with monkeypatch.context() as crash:
        crash.setattr(wc, "save_before_effects", save_then_lose)
        with pytest.raises(WorkerLost):
            loop.run_llm_loop(**kw)
    _, state = wc.read_working(tmp_path, kw["task_id"], 1)
    assert len(sent) == 2 and recovery.UNFINISHED_KEY not in state["usage"]
    text, usage, trace = _resume(tmp_path, monkeypatch, client, kw)
    assert text == "done" and len(sent) == 3 and "output_recovery" not in trace


@pytest.mark.parametrize("state", ["owed", "owner_wrote", "claimed"])
def test_new_owner_words_end_an_owed_turn_never_a_claimed_one(tmp_path, state):
    """A genuinely new owner row ends an owed informed turn before its send, which is then ordinary and unclaimed;
    a claimed send whose outcome is unconfirmed is never reset by owner words, only ended as interrupted."""
    from tests.test_loop_compaction import _ctx
    ctx = _ctx(tmp_path)
    ctx.context_fit_plan = None
    record = {"stage": "informed", "finishes": ["length"], "owner_revision": 0,
              **({"pending": "informed"} if state == "claimed" else {})}
    ctx.accumulated_usage[recovery.UNFINISHED_KEY] = record
    if state != "owed":
        _record_owner_directive(ctx.tools._ctx, source="owner_mailbox", content="use new evidence", msg_id="owner-1")
    claims = []
    def ordinary(call_ctx):
        claims.append(call_ctx.physical_send_claim)
        return {"role": "assistant", "content": "done"}, 0, "max"
    msg, _cost, _mode = recovery.call(ctx, ordinary)
    if state == "claimed":
        assert msg is None and claims == [] and record["interrupted"] and recovery.no_recap(ctx.accumulated_usage)
        assert ctx.accumulated_usage["_last_llm_error_kind"] == "output_recovery_interrupted"
    else:
        assert msg and [claim is not None for claim in claims] == [state == "owed"]
        assert recovery.UNFINISHED_KEY not in ctx.accumulated_usage


@pytest.mark.parametrize("retention", ["unfinished", "held"])
@pytest.mark.parametrize("custody", ["anthropic_consumed_receipts", "claudexor_native_continuation", None])
def test_a_kept_reply_carries_its_custody_to_compaction(tmp_path, custody, retention):
    """The kept reply carries every custody key the reply itself carries (``custody_private_key``), so the real
    compaction consumer sees what an ordinary retained reply would show: a native Anthropic tool unit its replay
    answered stays reclaimable. Control: a reply with no custody keeps its plain prose row unchanged."""
    from types import SimpleNamespace

    from ouroboros.anthropic_native_custody import (
        ANTHROPIC_CONSUMED_RECEIPTS_KEY,
        anthropic_replay_scope,
        retain_native_assistant_content,
    )
    from ouroboros.context_compaction import _atomic_units, _is_dialogue_prose_row
    target = {"provider": "anthropic", "resolved_model": "claude-fixture", "usage_model": "anthropic/claude-fixture",
              "base_url": "https://anthropic.invalid/v1"}
    native = [{"type": "text", "text": "I will read the file."},
              {"type": "tool_use", "id": "read-one", "name": "read_file", "input": {"path": "fixture.txt"}}]
    history = [retain_native_assistant_content({"role": "assistant", "content": "I will read the file.", "tool_calls": [
        {"id": "read-one", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"fixture.txt"}'}}]},
        native, target), {"role": "tool", "tool_call_id": "read-one", "content": "Fixture contents."}]
    reply = {"role": "assistant", "content": PARTIAL, "reasoning_content": "kept reasoning", "response_id": "resp-1"}
    expected = {"role": "assistant", "content": PARTIAL, "reasoning_content": "kept reasoning"}
    if custody == "anthropic_consumed_receipts":
        with anthropic_replay_scope():  # the native lane's own producer marks the replay this reply consumed
            LLMClient(api_key="fixture-only")._build_anthropic_messages(history, target)
            reply, _usage = LLMClient(api_key="fixture-only")._normalize_anthropic_response({
                "content": [{"type": "text", "text": PARTIAL}], "stop_reason": "max_tokens", "usage": {}}, target)
        expected = {"role": "assistant", "content": PARTIAL,
                    ANTHROPIC_CONSUMED_RECEIPTS_KEY: reply[ANTHROPIC_CONSUMED_RECEIPTS_KEY]}
        assert reply[ANTHROPIC_CONSUMED_RECEIPTS_KEY] and reply["stop_reason"] == "max_tokens"
    elif custody == "claudexor_native_continuation":
        reply["nativeContinuation"] = expected["nativeContinuation"] = {"route": {"source": "codex"}, "state": "opaque"}
    if retention == "unfinished":
        ctx = SimpleNamespace(tools=SimpleNamespace(_ctx=SimpleNamespace(_owner_directives=[], _round_response_facts={
            "finish_reason_present": True, "finish_reason": "max_tokens"})), accumulated_usage={}, messages=list(history))
        assert recovery.unfinished_reply(ctx, reply) is None
        retained = ctx.messages
    else:
        # A usable informed answer can itself be held by the existing completion control.
        # It must keep the same replay custody as the unfinished answer preceding it.
        from tests.test_delivery_forced_finalization import _arm_latch_with_candidate, _forced_test_context
        current_loop, registry, ctx, trace = _forced_test_context(tmp_path)
        _arm_latch_with_candidate(current_loop, registry, ctx, trace)
        start = len(ctx.messages)
        ctx.messages.extend(history)
        assert current_loop._finalize_loop_candidate(PARTIAL, ctx, registry, lambda *a, **k: None,
                                                     assistant_message=reply) is None
        retained = ctx.messages[start:]
    kept = retained[len(history)]
    assert kept == expected  # transport facts (stop_reason, response_id) stay with the reply, as before
    assert [(u.start, u.end) for u in _atomic_units(retained)] == [
        (u.start, u.end) for u in _atomic_units([*history, reply])]  # the ordinary whole-reply retention
    assert _is_dialogue_prose_row(kept) is (custody is None)


def test_new_owner_words_and_a_usable_tool_round_each_reopen_the_turn(tmp_path, monkeypatch, isolated, endpoint):
    """Only a usable reply or new owner words ends an episode: after the owner writes during the informed turn, and
    again after a tool round, another typed cut gets its own informed turn; rounds alone never reset it."""
    from ouroboros.owner_mailbox import write_owner_message
    tool = {"id": "one-tool", "type": "function", "function": {"name": "list_available_tools", "arguments": "{}"}}
    client, sent = endpoint([CUTS["length"], (LATER, "length"), ("", "tool_calls", [tool]), CUTS["tool_calls"], "answer"],
                            stream=False)
    kw, _notes = configured_loop(tmp_path, monkeypatch, client)
    chat = client.chat
    def owner_writes_once(**kwargs):
        result = chat(**kwargs)
        if len(sent) == 1:
            assert write_owner_message(tmp_path, "Use the newer counts.", kw["task_id"], msg_id="owner-newer")
        return result
    monkeypatch.setattr(client, "chat", owner_writes_once)
    text, usage, trace = loop.run_llm_loop(**kw)
    assert text == "done" and len(sent) == 5 and recovery.UNFINISHED_KEY not in usage
    assert "Use the newer counts." in json.dumps(sent[1]["messages"])
    assert [len(_cut_notices(s["messages"])) for s in sent] == [0, 1, 2, 2, 3]
    assert [(row["tool"], bool(row.get("is_error"))) for row in trace["tool_calls"]] == [("list_available_tools", False)]
