"""The start hint's shared seams (#3896): which refusal is this send's own (a current provider object's, through the
real stamp), which send is signed, what a chapter-map send proves, how out-of-order answers retire a refusal and what
the bounded store keeps, what the first-send choice may fold and how it names it; which connection a hinted first view
answers to when any route rebind moves it before anything is sent, when a call has spent it (its own observed
dispatch), and what a wrap-up sends and retires. Synthetic bodies and mocked outcomes only: no endpoint is reached."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

MODEL, CAP, ROUND = "openai-compatible::mock-model", 65_536, "turn:round:1"
FINISH_TASK = {"type": "function", "function": {"name": "finish_task", "parameters": {"type": "object"}}}


class Refused(Exception):
    """An innocuous synthetic provider failure carrying a received body, as an OpenAI-compatible SDK raises one;
    ``printed`` is its display alone."""

    def __init__(self, body, status=400, printed=""):
        super().__init__(printed or f"Error code: {status}")
        self.status_code, self.body, self.response = status, body, None


ZAI = {"error": {"code": "1261", "message": "Prompt too long", "type": "invalid_request_error"}}  # #3858's own pair
TYPED = {"error": {"code": "context_length_exceeded", "type": "invalid_request_error",  # a direct, numberless message
                   "message": "Your input exceeds the context window of this model."}}
HISTORY = {"error": {"code": 400, "message": "Provider returned error", "metadata": {  # only an earlier attempt sized it
    "provider_name": "Gateway", "previous_errors": [{"type": "context_length_exceeded", "provider_name": "Earlier",
                                                      "message": "This model's maximum context length was exceeded."}]}}}
PRINTED = {"error": {"message": "Bad request", "type": "invalid_request_error"}}  # with a display that names a size
FIELD = {"error": {"code": "string_above_max_length", "type": "invalid_request_error", "message": (
    "Invalid 'messages[0].content': string too long. Expected a string with maximum length 10485760.")}}
NUMBERED = {"error": {"code": "context_length_exceeded", "type": "invalid_request_error", "message": (
    "This model's maximum context length is 500000 tokens. However, your messages resulted in 548000 tokens.")}}


def stamped(body=TYPED, status=400, printed=""):
    """A failure row's real stamps (``size_refusal.stamp_failure``: the owner quote and the size fact)."""
    from ouroboros.size_refusal import stamp_failure

    usage = {}
    stamp_failure(usage, Refused(body, status, printed))
    return usage


@pytest.fixture
def seam(tmp_path, monkeypatch):
    from ouroboros import capability_evidence as ce, context_window as cw, usage_accounting as ua
    from ouroboros.response_limits import dispatch_target

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    target = dispatch_target(MODEL, {"OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:1/v1",
                                     "OPENAI_COMPATIBLE_API_KEY": "fixture-not-a-credential"})
    scope = cw.first_contact_scope(target, CAP)
    physical = ua.PhysicalAttemptContext("owner_max", "max", "fresh_model_usage", "r", ROUND, None, None, False, False,
                                         first_contact_scope=scope, view_kind="owner_turn", preferred_mode="max",
                                         window_witness_bound=True, window_observation_fp="r",
                                         first_contact_plan=(scope, False))
    plan = SimpleNamespace(route_fp="r", window_witness_bound=True, provider="openai-compatible", model=MODEL)

    def capture(order, raw, status=None, **context):
        return ua.PhysicalAttemptCapture(
            str(order), MODEL, plan.provider, "settled", "canonical_json_v1", max_completion_tokens=CAP,
            physical_context=replace(physical, first_contact_dispatch=(order, ce.utc_now_iso()), **context),
            candidate_manifest_ref={"fixture": True}, prompt_tokens_bounded_estimate=raw, provider_status_code=status)

    def failed(candidate, **usage):
        return {"_last_llm_failure": {"physical_attempt_id": candidate.attempt_id, "failure_code": "context_overflow",
                                      **usage.pop("call", {})}, **usage}

    def answer(candidate):
        cw.record_start_answer({"execution_id": "turn", "_last_llm_call_meta": {
            "physical_attempt_id": candidate.attempt_id, "round_id": ROUND}}, candidate, 1)

    return SimpleNamespace(cw=cw, ce=ce, root=tmp_path, target=target, scope=scope, physical=physical, plan=plan,
                           capture=capture, failed=failed, answer=answer,
                           limits=lambda: cw.first_contact_limits(target, CAP),
                           windows=lambda: ce._read(tmp_path).get("main_windows", {}))


def test_only_this_sends_own_numberless_refusal_teaches_a_start(seam):
    """Through the real stamp: a current object's numberless refusal (#3858's pair, a direct size message), an empty
    reply's size finish or a 200 body's own error is this send's own. A refusal only an earlier upstream attempt or the
    exception's display sized still types the round (its fact stays, numberless) but teaches no start; one field's
    length, 413, a numbered limit and an excluded quotation teach none either."""
    from ouroboros.llm_attempt import body_error_facts
    from ouroboros import size_refusal as sr

    refusal, fail = seam.cw.record_start_refusal, seam.failed
    assert refusal(fail(c := seam.capture(1, 548_000, 400), **stamped(ZAI)), c, ROUND)
    assert refusal(fail(c := seam.capture(2, 547_000, 400), **stamped(TYPED)), c, ROUND)
    for borrowed in (stamped(HISTORY), stamped(PRINTED, printed="Error code: 400 - maximum context length exceeded")):
        fact = sr.stamped(borrowed)
        assert fact is not None and not fact.has_numbers and not fact.own, borrowed  # the round still reads it
        assert not refusal(fail(c := seam.capture(3, 548_000, 400), **borrowed), c, ROUND)
    assert not refusal(fail(c := seam.capture(4, 548_000, 400), **stamped(FIELD)), c, ROUND)
    assert not refusal(fail(c := seam.capture(5, 548_000, 413), **stamped(TYPED, 413)), c, ROUND)
    assert not refusal(fail(c := seam.capture(6, 548_000, 400), **stamped(NUMBERED)), c, ROUND)
    assert not refusal(fail(c := seam.capture(7, 548_000, 400), **{sr.USAGE_KEY: sr.recorded(True, None).to_json()}),
                       c, ROUND)
    finish = fail(c := seam.capture(8, 530_000), call={"size_finish": "model_context_window_exceeded"})
    assert refusal(finish, c, ROUND)  # HTTP 200, no content: the finish is the refusal
    assert not refusal(fail(c := seam.capture(9, 520_000)), c, ROUND)  # no finish, no body: nothing
    own_body = fail(c := seam.capture(10, 529_000), call={"provider_body_error": True},
                    _last_llm_provider_answer=body_error_facts(ZAI["error"], False)["answer"])
    assert refusal(own_body, c, ROUND)  # a 200 body's own definite refusal
    quoted_body = fail(c := seam.capture(11, 400_000), call={"provider_body_error": True},
                       _last_llm_provider_answer=body_error_facts(HISTORY["error"], False)["answer"])
    assert not refusal(quoted_body, c, ROUND)
    generic_quote = {"error": "Unsupported option. Example error text: 'prompt is too long'."}
    assert not refusal(fail(c := seam.capture(12, 415_000, 400), **stamped(generic_quote)), c, ROUND)
    good = {"error": {**TYPED["error"], "message": "Prompt exceeds max length"}}
    assert refusal(fail(c := seam.capture(13, 550_000, 400), **stamped(good)), c, ROUND)
    assert refusal(fail(c := seam.capture(14, 550_001, 400), **stamped({"error": "Context length exceeded"})), c, ROUND)
    finish_with_body = fail(c := seam.capture(15, 410_000), call={"size_finish": "model_context_window_exceeded"},
                            _last_llm_provider_answer=body_error_facts(NUMBERED["error"], False)["answer"])
    assert not refusal(finish_with_body, c, ROUND)
    assert seam.limits() == (529_000, 0, 529_000)


def test_the_empty_size_finish_is_stamped_on_its_own_failure_row(tmp_path):
    from ouroboros.loop_llm_call import _record_and_emit_empty_response

    def stamp(finish, **usage):
        accumulated = {"_last_llm_call_meta": {"physical_attempt_id": "a", "size_finish": "stale"}}
        _record_and_emit_empty_response(
            usage={"response_finish_reason": finish, **usage}, msg={"content": "", "finish_reason": finish},
            accumulated_usage=accumulated, event_queue=None, drive_logs=tmp_path, task_id="t", execution_id="e",
            round_id="e:round:1", llm_call_id="c", round_idx=1, attempt=0, model=MODEL, task_type="task",
            content="", tool_calls=[], request_ref=None, response_ref=None, transient_budget=1,
            context_fit_event_fields={})
        return accumulated["_last_llm_failure"]

    assert stamp("model_context_window_exceeded")["size_finish"] == "model_context_window_exceeded"
    assert "size_finish" not in stamp("length")


def test_only_a_fresh_direct_chat_plan_is_signed_and_others_keep_window_learning(seam):
    """A root, wake or Presence send is unsigned, so its measured numberless refusal still caps its window (baseline);
    a custom tool grammar is not the reader's measure; a hint read for another scope is flagged, never sent."""
    from ouroboros.context_anchor import record_refused_input, refused_input

    payload = {"model": seam.target["resolved_model"], "messages": [{"role": "user", "content": "sample"}],
               "tools": [FINISH_TASK]}
    root = replace(seam.physical, first_contact_scope="", first_contact_plan=None)
    assert not seam.cw.signed_first_contact(root, seam.target, payload, CAP).first_contact_scope
    signed = seam.cw.signed_first_contact(seam.physical, seam.target, payload, CAP)
    assert signed.first_contact_scope == seam.scope and not seam.cw.first_contact_rebound(signed)
    grammar = {**payload, "tools": [{"type": "custom", "custom": {"name": "shell"}}]}
    assert not seam.cw.signed_first_contact(seam.physical, seam.target, grammar, CAP).first_contact_scope
    hinted = replace(seam.physical, first_contact_plan=("chat_raw_v1:another-key:65536", True))
    assert seam.cw.first_contact_rebound(seam.cw.signed_first_contact(hinted, seam.target, payload, CAP))
    assert seam.cw.first_contact_rebound(seam.cw.signed_first_contact(
        replace(seam.physical, first_contact_plan=(seam.scope, True)), seam.target, payload, CAP - 1))

    unsigned = seam.capture(1, 548_000, 400, first_contact_scope="", first_contact_plan=None)
    record_refused_input(seam.failed(unsigned, **stamped()), seam.plan, 560_000, measured=True, candidate=unsigned,
                         round_id=ROUND)
    rows = seam.windows()["r"]["observations"]
    assert rows[-1]["refused"] == 560_000 and rows[-1]["basis"] == "anchored_estimate"
    assert seam.limits() == (0, 0, 0)
    direct = seam.capture(2, 548_000, 400, window_observation_fp="d")
    record_refused_input(seam.failed(direct, **stamped()), replace_plan(seam.plan, "d"), 560_000, measured=True,
                         candidate=direct, round_id=ROUND)
    assert "d" not in seam.windows() and seam.limits() == (548_000, 0, 548_000)
    borrowed = seam.capture(3, 400_000, 400, window_observation_fp="h")  # same-round sizing alone
    usage, plan = seam.failed(borrowed, **stamped(HISTORY)), replace_plan(seam.plan, "h")
    record_refused_input(usage, plan, 420_000, measured=True, candidate=borrowed, round_id=ROUND)
    assert refused_input(usage, plan) == 420_000
    assert "h" not in seam.windows() and seam.limits() == (548_000, 0, 548_000)


@pytest.mark.parametrize("source", ["history", "detail", "nested", "nested-counts", "field", "body-history", "413", "unknown",
                                    "different-attempt", "local", "numbered", "plain", "finish", "zai", "forwarded",
                                    "numeric-quotation"])
def test_only_current_whole_size_can_make_durable_observations(seam, source):
    from ouroboros.context_anchor import record_refused_input, refused_input
    from ouroboros.context_window import WindowLimit, WindowRecord
    from ouroboros.llm_attempt import body_error_facts

    body = {
        "history": HISTORY, "body-history": HISTORY,
        "detail": {"error": {"code": 400, "message": "Provider returned error", "diagnostic": NUMBERED["error"]["message"]}},
        "nested": {"error": {**PRINTED["error"], "metadata": {"diagnostic": NUMBERED["error"]["message"]}}},
        "nested-counts": {"error": {**TYPED["error"], "diagnostic": {
            "type": "exceed_context_size_error", "n_ctx": 500_000, "n_prompt_tokens": 548_000}}},
        "field": {"error": {**TYPED["error"], "param": "messages[0].content"}},
        "numbered": NUMBERED, "zai": ZAI,
        "forwarded": {"error": {**PRINTED["error"], "metadata": {"provider_name": "Upstream",
            "raw": json.dumps({"error": {"code": "1261", "message": "Prompt too long; maximum context length is 80000 tokens"}})}}},
        "plain": {"error": {"code": 400, "message": TYPED["error"]["message"]}},
        "numeric-quotation": {"error": "Unsupported option. Example error text: 'maximum context length is 80000 tokens'."},
    }.get(source, TYPED)
    status = None if source in {"body-history", "unknown", "finish", "local"} else 413 if source == "413" else 400
    candidate = seam.capture(3, 548_000, status)
    if source == "local":
        candidate = SimpleNamespace(attempt_id=candidate.attempt_id, input_tokens=548_000,
                                    context_window_tokens=560_000)
    call = {"provider_body_error": True} if source == "body-history" else (
        {"size_finish": "model_context_window_exceeded"} if source == "finish" else {})
    stamps = ({"_last_llm_provider_answer": body_error_facts(body["error"], False)["answer"]}
              if source == "body-history" else {} if source == "finish" else stamped(body, status))
    usage = seam.failed(candidate, call=call, **stamps)
    if source == "different-attempt":
        usage["_last_llm_failure"]["physical_attempt_id"] = "another"
    plan = replace_plan(seam.plan, "r")
    plan.window_record = WindowRecord("r", None, WindowLimit(560_000, "total", "metadata"), None, reserve=CAP)
    record_refused_input(usage, plan, 548_000, measured=True, candidate=candidate, round_id=ROUND)
    assert refused_input(usage, plan) == 548_000
    observations = seam.windows().get("r", {}).get("observations", [])
    expected = {"local": (560_000, "total", "exact_local_count"), "numbered": (500_000, "total", "provider_stated"),
                "forwarded": (80_000, "total", "provider_stated"),
                "plain": (560_000, "total", "anchored_estimate"), "finish": (560_000, "total", "anchored_estimate")}
    assert [(row["refused"], row["kind"], row["basis"]) for row in observations] == (
        [expected[source]] if source in expected else [])
    assert seam.limits() == ((548_000, 0, 548_000) if source in {"plain", "finish", "zai"} else (0, 0, 0))


@pytest.mark.parametrize("current", [False, True], ids=["diagnostic", "direct"])
def test_early_rebind_requires_the_same_whole_request_evidence(seam, current):
    from ouroboros.window_learning import learn_window_from_refusal

    candidate = seam.capture(1, 548_000, 400)
    error = NUMBERED if current else {"error": {**PRINTED["error"], "detail": NUMBERED["error"]["message"]}}
    plan = replace_plan(seam.plan, "r")
    ctx = SimpleNamespace(accumulated_usage=seam.failed(candidate, **stamped(error)), context_fit_plan=plan,
                          active_use_local=False, messages=["ordinary"], active_context_mode="max", active_model=MODEL,
                          tools=object(), tool_schemas=[], model_role="main")

    def rebind(_plan, _tools, messages, **_kwargs):
        messages[0] = "rebound"
        return plan, "max"

    assert learn_window_from_refusal(ctx, SimpleNamespace(_rebind_context_fit_plan=rebind), candidate, ROUND) is current
    assert ctx.messages == (["rebound"] if current else ["ordinary"])
    assert bool(seam.windows()) is current


def replace_plan(plan, route_fp):
    return SimpleNamespace(**{**vars(plan), "route_fp": route_fp})


@pytest.mark.parametrize("fixture,status,kind", [
    ("openai-code", 400, "total"), ("openai-text-only", 400, "total"), ("openai-code", 413, "total"),
    ("openrouter-upstream-prompt-too-long", 400, "input"), ("detail-only", 400, "total"),
    ("openai-text-only", 413, None), ("bare-413", 413, None), ("body-size-413", 413, None),
    ("indexed-field", 400, None), ("guillemets", 400, None), ("markdown", 400, None),
    ("curly-apostrophe", 413, None), ("ascii-apostrophe", 413, None), ("unstructured-field", 400, None),
    ("generic-request", 400, "total"),
])
def test_real_numeric_refusal_records_and_rebinds_without_an_advisory_receipt(seam, fixture, status, kind):
    from ouroboros.context_anchor import record_refused_input
    from ouroboros.loop_llm_call import classify_llm_exception
    from ouroboros.size_refusal import stamp_failure
    from ouroboros.window_learning import learn_window_from_refusal
    from tests._size_refusal import _sdk_error

    bodies = Path(__file__).parent / "fixtures/size_refusals/bodies"
    sample = "maximum context length is 80000 tokens"
    controls = {
        "detail-only": {"detail": "This model supports a maximum context length of 80000 tokens."},
        "bare-413": {},
        "body-size-413": {"error": {"code": "payload_too_large", "message": f"Payload too large. {sample}."}},
        "indexed-field": {"error": {**json.loads((bodies / "openai-code.body.json").read_bytes())["error"],
                                    "param": "messages[0].content"}},
        **{label: {"error": {**TYPED["error"], "message": f"Context length exceeded; diagnostic {open_}{sample}{close}."}}
           for label, open_, close in (("guillemets", "«", "»"), ("markdown", "`", "`"))},
        **{label: {"error": {**TYPED["error"], "message": (
            f"Context length exceeded. Earlier attempt said {open_}This model{close}s "
            f"maximum context length is 8192 tokens.{close}")}}
           for label, open_, close in (("curly-apostrophe", "‘", "’"), ("ascii-apostrophe", "'", "'"))},
        "unstructured-field": {"error": {"type": "invalid_request_error", "message": (
            "Invalid 'messages[0].content': maximum length is 8192 tokens.")}},
        "generic-request": {"error": {"type": "invalid_request_error", "message": f"Invalid request: {sample}."}},
    }
    raw = json.dumps(controls[fixture]).encode() if fixture in controls else (bodies / f"{fixture}.body.json").read_bytes()
    failure = _sdk_error(status, raw)
    classification = classify_llm_exception(failure)
    assert classification.kind == "context_overflow" or fixture == "body-size-413"
    capture = seam.capture(1, 98_192, status, first_contact_scope="")
    usage = seam.failed(capture)
    fact = stamp_failure(usage, failure, classification)
    if kind:
        assert fact.own and fact.token_window == 80_000
    if fixture in {"curly-apostrophe", "ascii-apostrophe", "unstructured-field"}:
        assert fact.token_window == 8192  # the retry still sees the number; the durable writer below must not
    plan = seam.plan
    assert not seam.cw.record_start_refusal(usage, capture, ROUND)
    record_refused_input(usage, plan, 98_192, candidate=capture, round_id=ROUND)
    expected = [(80_000, kind, "provider_stated")] if kind else []
    assert [(row["refused"], row["kind"], row["basis"])
            for row in seam.windows().get("r", {}).get("observations", [])] == expected
    ctx = SimpleNamespace(accumulated_usage=usage, context_fit_plan=plan, active_use_local=False, active_model=MODEL,
                          active_context_mode="max", messages=["ordinary"], tools=object(), tool_schemas=[], model_role="main")

    def rebind(_plan, _tools, messages, **_kwargs):
        messages[0] = "rebound"
        return plan, "max"

    assert learn_window_from_refusal(ctx, SimpleNamespace(_rebind_context_fit_plan=rebind), capture, ROUND) is bool(kind)
    assert ctx.messages == (["rebound"] if kind else ["ordinary"])
    if fixture in {"curly-apostrophe", "ascii-apostrophe", "unstructured-field"}:
        assert not fact.own
    assert seam.limits() == (0, 0, 0)  # independent of the advisory history


def test_openrouter_advice_and_accepts_use_the_final_provider_filters(seam, monkeypatch):
    from ouroboros import llm_routing

    monkeypatch.setattr(llm_routing, "runtime_setting", lambda *_: '{"require_parameters": true}')
    target = llm_routing.resolve_target("openrouter::anthropic/test", {"OPENROUTER_API_KEY": "synthetic-or-key"})
    a = {**target, "provider_routing": {"only": ["Provider-A"], "allow_fallbacks": False}}
    equivalent = {**target, "provider_routing": '{"allow_fallbacks":false,"only":["Provider-A"]}'}
    b = {**a, "provider_routing": {**a["provider_routing"], "only": ["Provider-B"]}}
    scope = seam.cw.first_contact_scope(a, CAP)
    assert scope and seam.cw.first_contact_scope(equivalent, CAP) == scope
    assert seam.cw.first_contact_scope(b, CAP) != scope
    payload = {"model": a["resolved_model"], "messages": [{"role": "user", "content": "sample"}], "tools": [FINISH_TASK]}

    def signed(order, actual, status=None):
        # Target still describes A; a retry/late configuration may replace the SDK carrier.
        routing = llm_routing.openrouter_chat_routing(actual, payload["messages"])
        physical = seam.cw.signed_first_contact(replace(seam.physical, first_contact_plan=(scope, True)), a,
                       {**payload, "extra_body": {"provider": routing}}, CAP)
        return seam.capture(order, 548_000, status, first_contact_scope=physical.first_contact_scope,
                            first_contact_plan=physical.first_contact_plan), physical

    refusal, same = signed(1, equivalent, 400)
    assert not seam.cw.first_contact_rebound(same) and _refuse(seam, refusal, body=ZAI)
    malformed = seam.cw.signed_first_contact(same, a, {**payload, "extra_body": {"provider": []}}, CAP)
    assert not malformed.first_contact_scope and seam.cw.first_contact_rebound(malformed)
    acceptance, moved = signed(2, b)
    assert seam.cw.first_contact_rebound(moved)
    seam.answer(acceptance)
    assert seam.cw.first_contact_limits(a, CAP) == (548_000, 0, 548_000)
    assert seam.cw.first_contact_limits(b, CAP) == (0, 0, 0)
    accepted, _ = signed(3, a)
    seam.answer(accepted)
    assert seam.cw.first_contact_limits(a, CAP) == (0, 0, 0)


def test_plain_start_key_rotation_keeps_soft_evidence_but_not_old_window(seam):
    from ouroboros.capability_evidence import observe_token_density
    from ouroboros.context_anchor import record_input_anchor, record_refused_input
    from ouroboros.llm_attempt import _attempt_request
    from ouroboros.usage_accounting import bind_physical_attempt_context
    from ouroboros.window_learning import learn_window_from_refusal

    configured = {"provider": seam.target["provider"], "model": MODEL, "use_local": False}
    previous = seam.cw.route_observation_scope(configured, None, target=seam.target)
    rotated = {**seam.target, "api_key": "other-synthetic-key"}
    plan = SimpleNamespace(**{**vars(seam.plan), "route_fp": previous, "first_contact": seam.scope,
                              "initial_mode": "max", "output_reserve_tokens": CAP})
    payload = {"model": rotated["resolved_model"], "max_tokens": CAP, "tools": [FINISH_TASK],
               "messages": [{"role": "user", "content": "sample"}]}
    with bind_physical_attempt_context(replace(seam.physical, window_witness_bound=True,
                                               window_observation_fp=previous, first_contact_scope="")):
        request = _attempt_request(rotated, payload)
    physical = request.physical_context
    assert not seam.cw.first_contact_rebound(physical)  # no hint was used; the ordinary send must proceed
    keyless = {**seam.target, "api_key": "", "keyless": True}
    assert not seam.cw.signed_first_contact(replace(seam.physical, window_witness_bound=True,
        window_observation_fp=previous), keyless, payload, CAP).window_witness_bound

    def sent(order, status=None):
        candidate = seam.capture(order, 548_000, status)
        return replace(candidate, physical_context=replace(physical, first_contact_dispatch=(order, seam.ce.utc_now_iso())))

    numbered = sent(1, 400)
    usage = seam.failed(numbered, **stamped(NUMBERED))
    record_refused_input(usage, plan, 548_000, candidate=numbered, round_id=ROUND)
    ctx = SimpleNamespace(accumulated_usage=usage, context_fit_plan=plan, active_use_local=False, active_model=MODEL,
                          active_context_mode="max", messages=["ordinary"], tools=object(), tool_schemas=[], model_role="main")

    def rebind(*_args, **_kwargs):
        pytest.fail("the old connection must not be rebound from the new connection's refusal")

    assert not learn_window_from_refusal(ctx, SimpleNamespace(_rebind_context_fit_plan=rebind), numbered, ROUND)
    assert not learn_window_from_refusal(ctx, SimpleNamespace(_rebind_context_fit_plan=rebind), None, ROUND)
    numberless = sent(2, 400)
    assert _refuse(seam, numberless, body=ZAI)
    assert seam.limits() == (0, 0, 0)  # the observed send was on B
    assert seam.cw.first_contact_limits(rotated, CAP) == (548_000, 0, 548_000)

    accepted = sent(3)
    use = {"execution_id": "turn", "_last_llm_call_meta": {"physical_attempt_id": accepted.attempt_id, "round_id": ROUND},
           "_last_round_cache_usage": {"prompt_tokens": 549_000, "provider": plan.provider}}
    record_input_anchor(use, plan, round_idx=1, estimate_tokens=548_000, candidate=accepted)
    observe_token_density(request, {"prompt_tokens": 549_000}, drive_root_resolver=lambda _: seam.root)
    assert "_context_input_anchor" not in use
    assert previous not in seam.windows()  # neither a numeric refusal nor an acceptance belongs to A
    assert seam.cw.first_contact_limits(rotated, CAP) == (0, 0, 0)  # comparable acceptance belongs to B


def _refuse(seam, candidate, round_id=ROUND, body=TYPED):
    return seam.cw.record_start_refusal(seam.failed(candidate, **stamped(body)), candidate, round_id)


@pytest.mark.parametrize("change", ["none", "key", "surface", "surface-and-key"])
def test_named_chat_window_and_density_require_the_actual_wire_route(seam, change):
    from ouroboros.context_anchor import record_input_anchor, record_refused_input, refused_input
    from ouroboros.llm_attempt import _attempt_request
    from ouroboros.response_limits import dispatch_target
    from ouroboros.usage_accounting import bind_physical_attempt_context
    from ouroboros.window_learning import learn_window_from_refusal

    model = "openai@fixture::gpt-fixture"
    def target(surface, key):
        return dispatch_target(model, {"OUROBOROS_CONNECTIONS": [
            {"kind": "openai", "name": "fixture", "api_key": key, "api_surface": surface}]})

    chat = target("chat_completions", "synthetic-A")
    responses = "surface" in change
    actual = target("responses" if responses else "chat_completions", "synthetic-B" if "key" in change else "synthetic-A")
    fp = seam.cw.route_observation_scope({"provider": "openai", "model": model, "use_local": False}, None, target=chat)
    planned = seam.cw.first_contact_scope(chat, CAP)
    plan = SimpleNamespace(model=model, provider="openai", route_fp=fp, window_witness_bound=True,
                           first_contact=planned, initial_mode="max", output_reserve_tokens=CAP, view_kind="owner_turn")
    messages = [{"role": "user", "content": "sample " * 4000}]  # qualify the density reader's minimum input
    payload = {"model": actual["resolved_model"], **({"input": messages, "max_output_tokens": CAP} if responses else
               {"messages": messages, "max_tokens": CAP, "tools": [FINISH_TASK]})}
    with bind_physical_attempt_context(replace(seam.physical, route_fp=fp, window_observation_fp=fp,
            first_contact_plan=(planned, False), first_contact_scope="")):
        request = _attempt_request(actual, payload)
    same = change == "none"
    assert request.physical_context.window_witness_bound is same
    assert not seam.cw.first_contact_rebound(request.physical_context)  # each ordinary send is legal
    refused = replace(seam.capture(1, 548_000, 400), physical_context=request.physical_context)
    usage = seam.failed(refused, **stamped(NUMBERED))
    record_refused_input(usage, plan, 548_000, candidate=refused, round_id=ROUND)
    assert refused_input(usage, plan) == (548_000 if same else None)
    ctx = SimpleNamespace(accumulated_usage=usage, context_fit_plan=plan, active_use_local=False, active_model=model,
                          active_context_mode="max", messages=["ordinary"], tools=object(), tool_schemas=[], model_role="main")
    def rebind(_plan, _tools, rows, **_kwargs):
        rows[0] = "rebound"
        return plan, "max"

    assert learn_window_from_refusal(ctx, SimpleNamespace(_rebind_context_fit_plan=rebind), refused, ROUND) is same
    accepted = replace(refused, attempt_id="2", provider_status_code=None)
    tokens = request.prompt_tokens_bounded_estimate
    use = {"execution_id": "turn", "_last_llm_call_meta": {"physical_attempt_id": "2", "round_id": ROUND},
           "_last_round_cache_usage": {"prompt_tokens": tokens, "provider": "openai"}}
    record_input_anchor(use, plan, round_idx=1, estimate_tokens=tokens, candidate=accepted)
    seam.ce.observe_token_density(request, {"prompt_tokens": tokens}, drive_root_resolver=lambda _: seam.root)
    window = seam.windows().get(fp, {}).get("observations", [])
    assert any(row.get("refused") == 500_000 for row in window) is same
    assert any(row.get("accepted") == tokens for row in window) is same
    assert bool(use.get("_context_input_anchor")) is same
    densities = seam.ce._read(seam.root).get("token_density", {})
    assert bool(densities) is same and all(pair["route_fp"] == fp for entry in densities.values() for pair in entry["pairs"])


@pytest.mark.parametrize("tools", [[], [FINISH_TASK, {"type": "openrouter:web_search"}], [FINISH_TASK]])
def test_only_a_function_tools_reply_can_retire_the_function_tools_refusal(seam, tools):
    from ouroboros.llm_attempt import _attempt_request
    from ouroboros.usage_accounting import bind_physical_attempt_context

    target = seam.target
    fp = seam.cw.route_observation_scope({"provider": "openai-compatible", "model": MODEL, "use_local": False}, None, target=target)
    assert _refuse(seam, seam.capture(1, 548_000, 400), body=ZAI)
    payload = {"model": target["resolved_model"], "max_tokens": CAP, "tools": tools,
               "messages": [{"role": "user", "content": "sample"}]}
    with bind_physical_attempt_context(replace(seam.physical, window_observation_fp=fp)):
        signed = _attempt_request(target, payload).physical_context
    comparable = tools == [FINISH_TASK]
    assert bool(signed.first_contact_scope) is comparable and signed.window_witness_bound
    accepted = seam.capture(2, 548_000)
    seam.answer(replace(accepted, physical_context=replace(signed, first_contact_dispatch=(2, seam.ce.utc_now_iso()))))
    assert seam.limits() == ((0, 0, 0) if comparable else (548_000, 0, 548_000))


def test_a_chapter_map_send_bounds_no_max_start_and_only_a_retry_proves_maps(seam):
    low = seam.capture(4, 111_000, 400, rendered_mode="low")  # a start already in maps: not an in-round rung
    assert _refuse(seam, low)
    assert seam.limits() == (0, 0, 111_000)
    whole = seam.capture(2, 400_000, 400, round_id="turn:round:2")
    rung = seam.capture(3, 300_000, rendered_mode="low", automatic_pass_used=True, round_id="turn:round:2",
                        prefix={"origins": ["low_retry"]})
    assert _refuse(seam, whole, "turn:round:2")
    seam.cw.record_start_answer({"execution_id": "turn", "_last_llm_call_meta": {
        "physical_attempt_id": rung.attempt_id, "round_id": "turn:round:2"}}, rung, 2)
    assert seam.limits() == (400_000, 400_000, 111_000)
    assert {row["form"] for row in seam.ce._read(seam.root)["main_first_contact"][seam.scope]} == {"low", "ordinary", "maps"}


def test_a_refused_maps_rung_bounds_a_chapter_map_start_but_never_a_max_one(seam):
    """The rung's refusal still proves the round's maps rung (``before maps``) and now bounds what a chapter-map start
    compares with (``any form``); the Max wire keeps the Max refusal alone."""
    whole = seam.capture(1, 400_000, 400, round_id="turn:round:2")
    rung = seam.capture(2, 300_000, 400, rendered_mode="low", automatic_pass_used=True, round_id="turn:round:2",
                        prefix={"origins": ["low_retry"]})
    assert _refuse(seam, whole, "turn:round:2") and _refuse(seam, rung, "turn:round:2")
    assert seam.limits() == (400_000, 400_000, 300_000)


def test_a_later_acceptance_published_first_still_answers_an_older_refusal(seam):
    """Dispatch order decides cover: 560k@8 stored, 548k@9 still settling, 550k@10 settles and publishes first.
    No stored reply is both later and at least as large as 550k@10, so it is kept; then 548k@9 meets its answer."""
    seam.answer(seam.capture(8, 560_000))
    seam.answer(seam.capture(10, 550_000))  # smaller than the earlier reply, yet the only later one
    assert _refuse(seam, seam.capture(9, 548_000, 400))
    assert seam.limits() == (0, 0, 0)
    assert _refuse(seam, seam.capture(11, 600_000, 400))  # dispatched after every stored reply: it stands
    assert seam.limits() == (600_000, 0, 600_000)


def test_an_older_larger_reply_still_answers_an_earlier_larger_refusal(seam):
    seam.answer(seam.capture(8, 560_000))
    seam.answer(seam.capture(10, 500_000))  # later but smaller: 560k@8 stays the answer for earlier sizes above it
    refusals = {7: 555_000, 9: 520_000}  # 555k@7 is answered by 560k@8; 520k@9 by nothing (500k@10 is smaller)
    for order, raw in refusals.items():
        assert _refuse(seam, seam.capture(order, raw, 400))
    assert seam.limits() == (520_000, 0, 520_000)


def test_more_than_sixteen_undominated_replies_fold_into_one_cover_that_still_answers(seam):
    """Twenty-four usable replies, each dispatched later and smaller than the one before, are all undominated, so the
    bound folds the eight oldest into one cover row (their largest order and size). A refusal dispatched before a
    folded reply and no larger, settling only now, is answered by it instead of reviving; a refusal larger than every
    folded reply keeps its own witness, and an equal later reply answers that one."""
    for index in range(24):
        seam.answer(seam.capture(100 + index, 500_000 - 1_000 * index))
    rows = seam.ce._read(seam.root)["main_first_contact"][seam.scope]
    cover = [row for row in rows if row["form"] == "cover"]
    assert [(row["order"], row["raw"], row["accepted"]) for row in cover] == [(107, 500_000, True)]
    assert sum(row["accepted"] and row["form"] != "cover" for row in rows) == seam.cw._START_ROWS
    assert _refuse(seam, seam.capture(99, 499_500, 400))  # before 500k@100, settling after it was folded
    assert seam.limits() == (0, 0, 0)
    assert _refuse(seam, seam.capture(98, 501_000, 400))  # larger than every folded reply: still live
    assert seam.limits() == (501_000, 0, 501_000)
    seam.answer(seam.capture(130, 501_000))  # the same size, dispatched later, accepted
    assert seam.limits() == (0, 0, 0)


def test_only_a_reply_a_stored_later_larger_one_answers_for_is_not_written(seam):
    """The exact rule's cost: each non-dominated usable reply of a signed scope is one locked write, and it
    prunes the replies it dominates (earlier and no larger) so the set stays a dispatch-ordered skyline."""
    store = seam.ce._store_path(seam.root)
    for order, raw in enumerate(range(10_000, 410_000, 2_000), start=1):
        seam.answer(seam.capture(order, raw))
    assert [row["order"] for row in seam.ce._read(seam.root)["main_first_contact"][seam.scope]] == [200]
    written = store.read_bytes()
    seam.answer(seam.capture(150, 300_000))  # earlier on the wire and smaller than 408k@200: changes no cover
    seam.answer(seam.capture(200, 408_000))  # the same receipt again
    assert store.read_bytes() == written


def _contents(story, *, runs=1, room=None, maps_note=""):
    facts = {"floor": {"runs": runs}, **({"max_room": room} if room else {})}
    return {mode: ([{"type": "text", "text": ("B" * 400_000 if mode == "max" else "M" * 40_000) + maps_note},
                    {"type": "text", "text": "S" * story}], dict(facts)) for mode in ("max", "low")}


def _prefer(normal, limits, *, required=0, story_after=200_000, mode_after=None, compact=False, target=None):
    from ouroboros.context_fit import _prefer_first_contact
    from ouroboros.response_limits import dispatch_target

    asked = []
    core = SimpleNamespace(user_content_json='"hi"', supplementary_messages_json="[]", compact_reference_docs=compact,
                           required_book_tokens=lambda mode: required)

    def forms(start_saving):
        asked.append(start_saving)
        return _contents(story_after, runs=2, room=normal[0]["max"][1].get("max_room")), mode_after or normal[1]

    target = target or dispatch_target(MODEL, {"OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:1/v1",
                                               "OPENAI_COMPATIBLE_API_KEY": "fixture-not-a-credential"})
    return _prefer_first_contact(core, "max", normal, forms, target, [FINISH_TASK], limits), asked


def _story(chosen):
    contents, mode = chosen
    return len(contents[mode][0][-1]["text"].split("\n\n### First-send composition")[0])


def test_a_known_window_low_start_folds_its_own_tellings_and_keeps_its_books():
    normal = (_contents(800_000), "low")
    chosen, asked = _prefer(normal, (0, 0, 150_000))
    assert chosen is not normal and chosen[1] == "low" and _story(chosen) == 200_000 and asked
    assert _prefer(normal, (150_000, 0, 0))[0] is normal  # only a Max-wire refusal: a Low start owes it nothing


def test_the_required_handbook_is_counted_physically_but_never_paid_by_memory():
    normal = (_contents(800_000), "max")
    _chosen, plain = _prefer(normal, (250_000, 0, 250_000))
    _chosen, with_book = _prefer(normal, (250_000, 0, 250_000), required=30_000)
    assert with_book == [plain[0] - 30_000]


def test_the_required_handbook_never_takes_the_books_to_maps():
    """The ordinary size decides on maps: a folded view under the proven rung's goal stays Max with whole books even
    when this task's directly required handbook puts the physical request over it."""
    normal = (_contents(800_000, room={"cause": "held", "cuts": [], "window": 1}), "max")
    assert _prefer(normal, (140_000, 140_000, 140_000))[0][1] == "low"
    kept, asked = _prefer(normal, (140_000, 140_000, 140_000), required=30_000)
    assert kept[1] == "max" and asked and _story(kept) == 200_000  # its tellings folded, its books whole


def test_a_maps_rung_keeps_the_folded_tellings_and_says_why():
    held = {"cause": "held", "cuts": ["F2"], "window": 1}
    normal = (_contents(800_000, room=held), "max")
    chosen, _ = _prefer(normal, (90_000, 120_000, 90_000), story_after=200_000)
    assert chosen[1] == "low" and _story(chosen) == 200_000  # the folded tellings, not every story body again
    facts = chosen[0]["low"][1]
    assert facts["first_contact_preference"] and facts["max_room"]["cause"] == "first_contact"
    folded, _ = _prefer(normal, (90_000, 0, 90_000), story_after=200_000)
    assert folded[1] == "max" and _story(folded) == 200_000  # still above F1, yet smaller than the whole: kept
    note = folded[0]["max"][0][-1]["text"].split("### First-send composition")[1]
    assert "smaller than its ordinary form" in note and "larger request" not in note


def test_a_fold_below_the_material_margin_does_not_replace_the_ordinary_view():
    normal = (_contents(800_000), "max")
    assert _prefer(normal, (150_000, 0, 150_000), story_after=790_000)[0] is normal
    assert _prefer(normal, (150_000, 0, 150_000), story_after=200_000)[0] is not normal


def test_openrouter_server_web_skips_the_speculative_fit(monkeypatch):
    from ouroboros import llm_openai_compatible
    from ouroboros.response_limits import dispatch_target

    normal = (_contents(800_000), "max")
    target = dispatch_target("openrouter::google/gemini-fixture", {"OPENROUTER_API_KEY": "synthetic-or-key"})
    limits = (150_000, 0, 150_000)
    monkeypatch.setattr(llm_openai_compatible, "runtime_setting", lambda *_a: "server")
    assert _prefer(normal, limits, target=target) == (normal, [])  # never invokes forms(start_saving=...)
    monkeypatch.setattr(llm_openai_compatible, "runtime_setting", lambda *_a: "off")
    assert _prefer(normal, limits, target=target)[0] is not normal


def test_a_forced_call_keeps_its_own_receipt_and_publishes_nothing(monkeypatch, tmp_path):
    """The receipt is taken at the call, from that call's own capture; publishing is the rail's (below)."""
    import contextlib
    from ouroboros import context_window, loop, loop_forced_finalization as forced, task_pacing, working_checkpoint

    published, reply = [], {"content": "done"}
    sentinel = object()
    monkeypatch.setattr(context_window, "start_answer", lambda usage, capture, idx: ["scope", {"capture": capture, "idx": idx}])
    monkeypatch.setattr(context_window, "publish_start_outcome", published.append)
    monkeypatch.setattr(forced, "_forced_physical_context", lambda *_a: None)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_kw: (dict(reply) if reply else None, 0.0))
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: sentinel)
    monkeypatch.setattr(loop, "_retire_main_reasoning", lambda _ctx: False)
    monkeypatch.setattr(loop, "_server_web_allowed_by_task", lambda _ctx: False)
    monkeypatch.setattr(working_checkpoint, "model_call_scope", lambda _ctx: contextlib.nullcontext())
    monkeypatch.setattr(task_pacing, "wrapup_model_options", lambda _ctx: {})
    ctx = SimpleNamespace(llm=None, tools=None, messages=[], active_model=MODEL, active_effort="", max_retries=1,
                          drive_logs=tmp_path, task_id="t", round_idx=3, event_queue=None, accumulated_usage={},
                          task_type="task", active_use_local=False, deadline_ts=None)
    assert forced._call_forced_model_once(ctx) == "done"
    assert ctx.accumulated_usage["_forced_response_meta"]["start_answer"] == ["scope", {"capture": sentinel, "idx": 3}]
    reply = None  # no reply: no receipt
    forced._call_forced_model_once(ctx)
    assert "start_answer" not in ctx.accumulated_usage["_forced_response_meta"] and published == []


RECEIPT = ["chat_function_raw_v2:fixture:65536", {"attempt": "forced", "order": 7, "accepted": True}]
COMPLETE = {"finish_reason_present": True, "finish_reason": "stop", "tool_call_count": 0, "reply_calls": []}


HELD = "The retained complete answer."


@pytest.mark.parametrize(("meta", "text", "held", "kept"), [
    pytest.param(COMPLETE, "The final answer.", None, True, id="kept-words"),
    pytest.param({**COMPLETE, "finish_reason": "length"}, "Partial prose the provider cut", None, False, id="cut-prose"),
    pytest.param({**COMPLETE, "finish_reason": "tool_calls", "tool_call_count": 1, "reply_calls": [
        {"name": "finish_task", "arguments": json.dumps({"action": "finish", "answer": "Native final."})}]},
                 "", None, True, id="kept-finish-task"),
    pytest.param(COMPLETE, json.dumps({"action": "finish", "answer_sha256": "0" * 64}), HELD, False,
                 id="held-answer-kept-for-a-refused-control"),
])
def test_only_a_reply_the_forced_rail_keeps_retires_the_refusal(tmp_path, monkeypatch, meta, text, held, kept):
    """A wrap-up's usable reply answers its own send's size only once the rail accepts that reply (words, or an
    adopted ``finish_task``); discarded partial prose publishes nothing, and neither does a refused control beside
    an answer the rail already held: that older answer stays the result, this reply is not accepted."""
    from ouroboros import context_window, loop_forced_finalization as forced
    from tests.test_delivery_forced_finalization import _forced_test_context

    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    ctx.tool_schemas = [FINISH_TASK]
    if held:
        loop._replace_delivery_candidate(registry, ctx, trace, held, control="awaiting_control")
    published = []
    monkeypatch.setattr(context_window, "publish_start_outcome", published.append)

    def forced_call(_ctx, **_kwargs):  # the real call stamps its own receipt and author on any reply
        ctx.accumulated_usage["_forced_response_meta"] = {**meta, "start_answer": RECEIPT, "author_model": MODEL}
        return text

    monkeypatch.setattr(loop, "_call_forced_model_once", forced_call)
    returned, _usage, _trace = forced._forced_final_answer(ctx, prompt="Finish now", fallback_text="host words",
                                                           reason_code="round_limit")
    assert published == ([RECEIPT] if kept else [])
    if held:  # the rail really kept the older answer
        assert returned == HELD and registry._ctx._delivery_candidate.full_text == HELD


NOTE = "First-send composition"  # the hint's own sentence in the view it shaped (context_fit._prefer_first_contact)
OTHER = "openai-compatible::other-model"  # the same endpoint and key, another model: another connection


def _hinted_start(seam, tmp_path, monkeypatch, *, maps_rung=False):
    """A fresh direct chat's real start fit after its connection refused a comparable larger request, before any send:
    the plan the planner chose (the smaller view with its note, the ordinary one kept beside it) and the plan it would
    have sent without the hint; with ``maps_rung`` the refusal's chapter-map retry was answered too, so the hint takes
    the books' maps. Fixture input only: each room's page carries the telling a hint may fold it to."""
    from ouroboros import context, dialogue_store
    from ouroboros import context_fit as cf
    from ouroboros.chronicle_lead import LEAD_RULE_VERSION
    from ouroboros.owner_relay import owner_row_source
    from ouroboros.response_limits import bind_context_route
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools.registry import ToolRegistry
    from tests import _turn_shapes
    from tests._stand import lay_rooms
    from tests.test_cache_optimization import _make_env_and_memory

    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    env, memory = _make_env_and_memory(tmp_path / "world")
    page = _turn_shapes._page
    with monkeypatch.context() as shapes:
        shapes.setattr(_turn_shapes, "_page", lambda room, ids, text: {
            **page(room, ids, f"Room {room} settled its mock work.\n\n{text}"),
            "metadata": {"lead_rule": LEAD_RULE_VERSION}})
        lay_rooms(memory.drive_root, 12, 120_000)
    def admit(client_id, room=700_001):
        row = dialogue_store.append(memory.drive_root, "chat", {"direction": "in", "source": "web", "user_id": 1,
            "chat_id": room, "client_message_id": client_id, "ts": seam.ce.utc_now_iso(), "text": "hi"}).row
        origin, _ = owner_row_source(row)
        return {"type": "task", "text": "hi", "id": client_id, "chat_id": room, "_is_direct_chat": True,
                "origin_message_ref": origin}

    task = admit("tmain")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.event_queue, ctx._accumulated_usage = "tmain", None, {}
    ctx.active_model, ctx.active_use_local, ctx.active_effort = MODEL, False, ""
    schemas = initial_tool_schemas(registry, context_mode="max")

    def resolve(asked, **_kw):  # the route as the fit asks it: this model's own connection, its window known
        route = {"model": asked.get("model") or MODEL, "provider": "openai-compatible", "use_local": False,
                 "base_url": "", "options": {}}
        bind_context_route(route, {"OPENAI_COMPATIBLE_BASE_URL": seam.target["base_url"],
                                   "OPENAI_COMPATIBLE_API_KEY": seam.target["api_key"]})
        return route, SimpleNamespace(route_fp=route["model"], status="asserted", stale=False, window_tokens=1_000_000)

    monkeypatch.setattr(context, "_context_fit_route", resolve)
    def capture(turn):
        core = context._capture_context_core(env, memory, turn, None, None)
        return replace(core, architecture_md=core.architecture_md + "\n\n" + "word " * 80_000)  # whole books beside their maps

    def fit(turn=task, core=None):
        return context._build_context_fit_plan(env, core or capture(turn), turn, preferred_mode="max", tool_schemas=schemas,
                                               route_resolver=resolve)

    core = capture(task)
    ordinary = fit(core=core)
    whole = ordinary.projection(ordinary.initial_mode).estimated_tokens
    refused = seam.capture(1, int(whole * (0.45 if maps_rung else 0.7)), 400)
    assert _refuse(seam, refused, body=ZAI)
    if maps_rung:
        seam.answer(seam.capture(2, int(whole * 0.2), rendered_mode="low", automatic_pass_used=True,
                                 prefix={"origins": ["low_retry"]}))
    hinted = fit(core=core)
    assert hinted.first_contact == seam.scope and hinted.first_contact_normal is not None
    assert NOTE in hinted.projection(hinted.initial_mode).system_content_json
    assert (hinted.initial_mode, ordinary.initial_mode) == (("low" if maps_rung else "max"), "max")
    return SimpleNamespace(plan=hinted, ordinary=ordinary, registry=registry, tools=schemas,
                           admit=admit, fit=fit, task=task, drive=memory.drive_root, capture=capture, env=env)


@pytest.mark.parametrize("early_key,late_key", [("a", None), ("b", "a"), ("a", "b")],
                         ids=["known-window-one-read", "unknown-window-adopts-hint", "unknown-window-drops-old-hint"])
def test_start_hint_uses_only_the_final_horizon_binding(seam, tmp_path, monkeypatch, early_key, late_key):
    """An advisory Max→maps is not a physical narrowing. M2 still reads an unknown window again; a second usable
    route answer at the same size/cap nevertheless changes where the start hint must be read."""
    from ouroboros import context_fit as cf, context_view_facts as facts
    from ouroboros.response_limits import bind_context_route

    seed = _hinted_start(seam, tmp_path, monkeypatch, maps_rung=True)
    monkeypatch.setattr(cf, "_route_key", lambda _task: "fixture-early-route")
    reads, scopes = [], []
    limits = seam.cw.first_contact_limits

    def resolve(task, *, allow_fetch):
        assert allow_fetch
        reads.append(bool(task.get("route_read_again")))
        key = early_key if len(reads) == 1 or late_key is None else late_key
        route = {"model": MODEL, "provider": "openai-compatible", "use_local": False, "base_url": "", "options": {}}
        bind_context_route(route, {"OPENAI_COMPATIBLE_BASE_URL": seam.target["base_url"],
                                   "OPENAI_COMPATIBLE_API_KEY": seam.target["api_key"] if key == "a" else "fixture-other-key"})
        status = "asserted" if late_key is None else "failed" if len(reads) == 1 else "unprobeable"
        window = 1_000_000 if late_key is None else 0
        return route, seam.ce.CapabilityEvidence(window, status, "fixture", "same-measured-frame")

    def observed_limits(target, cap, *, scope):
        scopes.append(scope)
        return limits(target, cap, scope=scope)

    monkeypatch.setattr(seam.cw, "first_contact_limits", observed_limits)
    plan = cf.build_context_fit_plan(seed.env, seed.ordinary.core, seed.task, preferred_mode="max",
                                     route_resolver=cf.RouteQuestion(seed.task, resolve), tool_schemas=seed.tools)
    assert reads == ([False] if late_key is None else [False, True])
    assert scopes == [plan.first_contact]  # no obsolete connection evidence read along the way
    if (late_key or early_key) == "a":
        assert plan.first_contact == seam.scope and plan.start == ("low", "first_contact_maps")
        normal = plan.ordinary_first_contact()
        assert normal.start == ("max", "" if late_key is None else "unmeasured")
        assert "first-send chapter maps" in facts.mode_words(plan, "low")
        line = facts.memory_facts(plan, "low")
        assert "first send chose chapter maps" in line and "that refusal stated no window" in line
        assert "Max held" not in line and "temporarily, for the window" not in line
        assert not facts.low_start_line(plan, "low", seed.task["id"])
    else:
        assert plan.first_contact != seam.scope and plan.first_contact_normal is None
        assert plan.start == ("max", "unmeasured") and plan.window_record.usable is None


def test_a_window_lowered_start_keeps_its_cause_beside_an_advisory_memory_fold(seam, tmp_path, monkeypatch):
    """The known window takes the whole books. The remembered refusal can only choose extra memory folds in Low."""
    from ouroboros import context_fit, context_view_facts as facts, memory_floor
    from ouroboros.context_measurement import mode_cause
    from ouroboros.memory_view import snapshot_from_json

    seed = _hinted_start(seam, tmp_path, monkeypatch)
    core = replace(seed.ordinary.core, architecture_md=seed.ordinary.core.architecture_md + "word " * 800_000)
    plan = seed.fit(core=core)
    assert plan.start == ("low", "route") and plan.window_record.usable is not None
    # The stored refusal must actually shape the selected Low horizon; the physical fit alone leaves more memory.
    assert plan.first_contact_normal is not None and plan.low_projection.memory_facts["first_contact_preference"] is True
    assert plan.ordinary_first_contact().start == plan.start
    assert mode_cause(plan, "low") == "route" and "cause route" in facts.mode_words(plan, "low")
    line = facts.memory_facts(plan, "low")
    assert "Low, temporarily, for the window, cause route" in line and "first send chose a smaller memory view" in line
    assert "folding my memory cannot change that" in facts.low_start_line(plan, "low", seed.task["id"])

    snapshot = snapshot_from_json(core.memory_view_json)
    arms = {"now": snapshot, "proposed": replace(snapshot, story=snapshot.story[:-1])}
    ordinary = plan.ordinary_first_contact()
    before = memory_floor.counterfactual_starts(ordinary, arms)
    assert before["frame"]["maps"] and all(arm["mode"] == "low" for arm in before["arms"].values())

    # The hint was spent and the current selection moved: neither can stand in for the captured Low start.
    shown = replace(plan.first_contact_spent(), initial_mode="nano")
    comparison = memory_floor.counterfactual_starts(shown, arms)
    assert comparison["frame"]["started"] == "low" and comparison["frame"]["maps"]
    lines = "\n".join(memory_floor.counterfactual_lines(comparison))
    assert "This task started in Low, temporarily, for the window" in lines
    for name in arms:
        assert f"- {name}: not measured (its start took a first send's preference this fit does not repeat)" in lines

    # A newly captured Low projection cannot describe a lost original start.
    unknown = memory_floor.counterfactual_starts(context_fit.restored_start(shown, None), arms)
    assert unknown["frame"]["started"] == "unknown"
    assert all(arm == {"unmeasured": "the original start did not survive its continuation"}
               for arm in unknown["arms"].values())


def test_only_a_new_rooms_first_owner_turn_uses_the_hint(seam, tmp_path, monkeypatch):
    start = _hinted_start(seam, tmp_path, monkeypatch)
    continued = start.fit(start.admit("followup", start.task["chat_id"]))
    assert continued.first_contact == seam.scope  # outcomes (including a later usable reply) can still teach
    assert continued.first_contact_normal is None and continued.initial_mode == "max"
    assert NOTE not in continued.projection("max").system_content_json
    unbound = start.fit({key: value for key, value in start.admit("no-ref", 700_003).items()
                         if key != "origin_message_ref"})
    assert unbound.first_contact == seam.scope and unbound.first_contact_normal is None
    fresh = start.fit(start.admit("other-first", 700_002))
    assert fresh.first_contact_normal is not None


def test_project_note_exists_before_the_first_physical_owner_row(seam, tmp_path, monkeypatch):
    from ouroboros import dialogue_store, projects_registry, room_appearances
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.memory_view import snapshot_from_json
    from tests._turn_shapes import _author

    start = _hinted_start(seam, tmp_path, monkeypatch)
    room = projects_registry.create_project(start.drive, "existing-logical-room")["chat_id"]
    assert ChronicleStore(start.drive).publish([{"kind": "note", "room_id": str(room), "task_id": "earlier",
        "author": _author("earlier"), "text": "This room has work before its first physical chat."}]).ok
    task = start.admit("first-physical", room)
    assert room_appearances.run(start.drive, lambda: False)["state"] == "done"
    core = start.capture(task)
    view = snapshot_from_json(core.memory_view_json).room
    assert dialogue_store.first_room_row(start.drive, room, task["id"])  # physical first, logically not
    assert view["room_id"] == str(room) and view["facts"]["notes"] == 1 and not view["facts"]["pages"]
    assert len(view["lane1"]) == 1 and not view["lane2"] and not view["project_words"]
    assert start.fit(task, core).first_contact_normal is None
    assert start.fit(start.admit("plain-ws-room", 700_012)).first_contact_normal is not None


@pytest.mark.parametrize("maps_rung", [False, True], ids=["folded-memory", "chapter-maps"])
def test_a_route_change_before_the_first_send_does_not_carry_the_refusing_connections_view(
        seam, tmp_path, monkeypatch, maps_rung):
    """#3896 R2: a fresh chat's smaller view answers to the connection that refused. Nothing has been sent when Settings
    or the owner moves Main to another model at the round's binding, so that route's first request starts from the
    ordinary view (and its mode) fit to its own window: no hint note, no hint fact, the same core, owner words, tools."""
    from ouroboros import loop_routes

    start = _hinted_start(seam, tmp_path, monkeypatch, maps_rung=maps_rung)
    plan, ordinary, registry = start.plan, start.ordinary, start.registry
    # Every fallback form is ordinary, including Nano if that new route's physical window takes the view there.
    assert plan.projection("nano").system_content_json != ordinary.projection("nano").system_content_json
    assert plan.ordinary_first_contact().projection("nano") == ordinary.projection("nano")
    messages = plan.messages_for(plan.initial_mode)
    owner_rows, tools = json.dumps(messages[1:]), json.dumps(start.tools)
    registry._ctx.active_model_override = OTHER
    model, _local, _effort, rebound, mode = loop_routes.bind_round_route(
        registry, messages, (MODEL, False, ""), plan, plan.initial_mode, "max", start.tools)
    view = rebound.projection(mode)
    assert (model, rebound.route_fp) == (OTHER, OTHER)  # the route really moved
    assert NOTE not in json.dumps(messages[0]) and NOTE not in view.system_content_json
    assert not view.memory_facts.get("first_contact_preference")
    assert (mode, view.system_content_json, rebound.start) == (
        ordinary.initial_mode, ordinary.projection(ordinary.initial_mode).system_content_json, ordinary.start)
    assert rebound.first_contact_normal is None and registry._ctx.context_fit_plan is rebound
    assert rebound.core_sha256 == plan.core_sha256 and rebound.user_content_json == plan.user_content_json
    assert json.dumps(messages[1:]) == owner_rows and json.dumps(start.tools) == tools


def _bind(start, plan, mode, messages, **chosen):
    """``loop_routes.bind_round_route`` as a round's boundary calls it, with the one-shot choices (a role act, a
    switch) set on the context first: ``(model, local, effort, plan, mode)``."""
    from ouroboros import loop_routes

    for name, value in chosen.items():
        setattr(start.registry._ctx, name, value)
    return loop_routes.bind_round_route(start.registry, messages, (MODEL, False, ""), plan, mode, "max", start.tools)


def test_a_handover_on_the_same_connection_or_after_a_send_keeps_the_shown_view(seam, tmp_path, monkeypatch):
    """The correction reaches only an unsent hint on another route. A role act that names no new model rebinds the round
    on the same connection: the shown view stays, and so does the ordinary form no send has spent, so a later move to
    another model still starts plain. Once a request has gone the shown view may stay while the next route admits it."""
    start = _hinted_start(seam, tmp_path, monkeypatch)
    plan, ordinary = start.plan, start.ordinary
    messages = plan.messages_for(plan.initial_mode)
    shown = json.dumps(messages[0])
    *_route, kept, mode = _bind(start, plan, plan.initial_mode, messages, active_role_override="main")
    assert NOTE in kept.projection(mode).system_content_json and json.dumps(messages[0]) == shown
    assert kept.first_contact_normal is plan.first_contact_normal  # nothing was sent: the ordinary form stays beside it
    *_route, moved, mode = _bind(start, kept, mode, messages, active_model_override=OTHER)
    assert NOTE not in json.dumps(messages[0]) and mode == ordinary.initial_mode
    assert moved.projection(mode).system_content_json == ordinary.projection(ordinary.initial_mode).system_content_json
    spent = plan.first_contact_spent()  # a request has gone: the plan keeps no ordinary form
    messages = spent.messages_for(spent.initial_mode)
    shown = json.dumps(messages[0])
    *_route, rebound, mode = _bind(start, spent, spent.initial_mode, messages, active_model_override=OTHER)
    assert NOTE in rebound.projection(mode).system_content_json and json.dumps(messages[0]) == shown


def test_a_role_handover_keeps_the_hint_armed_for_a_key_that_moves_before_the_send(seam, tmp_path, monkeypatch):
    """#3896 R2: a role act rebinds the round without a new model, nothing is sent, and the transport key may still move
    before the request goes out. The physical gate stays the authority on that: it refuses the hinted view for another
    key (the ordinary form goes instead) and admits it for the key it was fit for."""
    from ouroboros.context_fit import _physical_context_for_fit, measure_main_fit
    from ouroboros.response_limits import dispatch_target

    start = _hinted_start(seam, tmp_path, monkeypatch)
    messages = start.plan.messages_for(start.plan.initial_mode)
    *_route, kept, mode = _bind(start, start.plan, start.plan.initial_mode, messages, active_role_override="main")
    fit = measure_main_fit(kept, messages, start.tools, profile="owner_max", rendered_mode=mode, round_id=ROUND)
    physical = _physical_context_for_fit(fit, kept)

    def signed(key):  # the send seam's own stamp for a payload that goes out on this key
        target = dispatch_target(MODEL, {"OPENAI_COMPATIBLE_BASE_URL": seam.target["base_url"],
                                         "OPENAI_COMPATIBLE_API_KEY": key})
        payload = {"model": target["resolved_model"], "messages": [{"role": "user", "content": "sample"}],
                   "tools": [FINISH_TASK]}
        return seam.cw.signed_first_contact(physical, target, payload, CAP)

    assert not seam.cw.first_contact_rebound(signed(seam.target["api_key"]))
    assert seam.cw.first_contact_rebound(signed("another-fixture-key"))


def _round_ctx(start, plan):
    """The round call's own context as ``loop_routes.first_contact_guard`` reads and updates it."""
    start.registry._ctx.context_fit_plan = plan
    return SimpleNamespace(context_fit_plan=plan, tools=start.registry)


def test_only_this_calls_own_hinted_dispatch_spends_the_ordinary_form(seam, tmp_path, monkeypatch):
    """Spent means sent: a candidate refused before its bytes (published as released), a helper's or an ordinary
    view's dispatch and any older capture still left in the thread spend nothing. The hinted view's dispatch spends it
    at once, so a later attempt of the same call that resolves another key keeps the view it showed; a release proving
    that send never left arms the ordinary form again, and the physical gate refuses the hinted view for the moved
    key once more. Every outer listener still hears what an inner attempt's own listener hears; once the call has
    returned, a sender's late outcome changes no plan."""
    import contextvars
    from ouroboros import loop
    from ouroboros.loop_routes import first_contact_guard
    from ouroboros.physical_attempt_context import dispatched, observe_dispatch

    start = _hinted_start(seam, tmp_path, monkeypatch)
    ctx = _round_ctx(start, start.plan)
    hinted = replace(seam.physical, first_contact_plan=(seam.scope, True))
    moved = SimpleNamespace(physical_context=replace(hinted, first_contact_scope="chat_raw_v1:another-key:65536"))

    def capture(state, attempt, context=hinted):
        return SimpleNamespace(state=state, attempt_id=attempt, physical_context=context)

    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: capture("settled", "older-call"))
    with first_contact_guard(ctx):
        pass  # nothing dispatched: an older capture in the thread is no send of this call
    assert ctx.context_fit_plan is start.plan
    heard = []
    with first_contact_guard(ctx) as hint, observe_dispatch(heard.append):
        dispatched(capture("released", "refused-before-bytes"))
        dispatched(capture("dispatched", "helper", replace(seam.physical, first_contact_plan=None)))
        assert ctx.context_fit_plan is start.plan and len(heard) == 2
        dispatched(capture("dispatched", "hinted"))
        assert ctx.context_fit_plan.first_contact_normal is None
        assert ctx.tools._ctx.context_fit_plan is ctx.context_fit_plan
        assert hint(lambda _request: True)(moved) is True and not hint.rebound  # shown: it stays
        dispatched(capture("released", "hinted"))  # that send was proven never to have left
        assert ctx.context_fit_plan.first_contact_normal is start.plan.first_contact_normal
        assert hint(lambda _request: True)(moved) is False and hint.rebound
        sender = contextvars.copy_context()  # a sender thread's context, copied while the call was live
    armed = ctx.context_fit_plan
    sender.run(dispatched, capture("settled", "late"))  # heard after the call returned: no later plan changes
    assert ctx.context_fit_plan is armed and armed.first_contact_normal is not None


def test_every_route_rebind_keeps_an_unsent_hint_only_on_its_own_scope(seam, tmp_path, monkeypatch):
    """The shared rebind (a walk candidate, a wait's reprepare, the post-wait resend, a round's handover): another
    model's or an unresolved route's first request starts from the ordinary view, with no hint note, fact or facts
    row of the hinted preparation; the same scope keeps the hint armed for the send seam; a spent hint keeps the view
    it showed on the next route, its note naming the connection it was composed for."""
    import hashlib
    from ouroboros import context, context_fit, loop
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, HOST_TEXT_SHA_KEY
    from ouroboros.context_view_facts import CONTEXT_FACTS_NAME
    from ouroboros.response_limits import dispatch_target

    start = _hinted_start(seam, tmp_path, monkeypatch)
    plan, ordinary, registry = start.plan, start.ordinary, start.registry

    def rebind(plan, model=MODEL):
        messages = plan.messages_for(plan.initial_mode)
        line = "[CONTEXT_FACTS] the hinted preparation's own row"
        messages.append({"role": "user", "content": line, HOST_CONTEXT_KIND_KEY: CONTEXT_FACTS_NAME,
                         HOST_TEXT_SHA_KEY: hashlib.sha256(line.encode("utf-8")).hexdigest()})  # as its producer marks it
        registry._ctx._accumulated_usage = {"_context_facts_line": line, "_context_facts_key": ["k"]}
        rebound, mode = loop._rebind_context_fit_plan(plan, registry, messages, model=model, use_local=False,
                                                      tool_schemas=start.tools, start_mode=plan.initial_mode,
                                                      model_role="main", model_route={})
        return rebound, mode, messages, line in json.dumps(messages)

    moved, mode, messages, facts_row = rebind(plan, OTHER)
    assert (moved.route_fp, mode, moved.first_contact_normal) == (OTHER, ordinary.initial_mode, None)
    assert NOTE not in json.dumps(messages[0]) and not facts_row  # the unsent row described a view never sent
    assert not moved.projection(mode).memory_facts.get("first_contact_preference")
    same, mode, messages, facts_row = rebind(plan)
    assert same.first_contact_normal is plan.first_contact_normal and NOTE in json.dumps(messages[0]) and facts_row
    original = context._context_fit_route
    monkeypatch.setattr(context, "_context_fit_route", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("offline")))
    unresolved, mode, messages, _row = rebind(plan)
    assert unresolved.first_contact_normal is None and NOTE not in json.dumps(messages[0])
    monkeypatch.setattr(context, "_context_fit_route", original)
    shown, mode, messages, _row = rebind(plan.first_contact_spent(), OTHER)
    assert NOTE in json.dumps(messages[0]) and "connection this first send was composed for" in json.dumps(messages[0])
    target = dispatch_target(OTHER, {"OPENAI_COMPATIBLE_BASE_URL": seam.target["base_url"],
                                     "OPENAI_COMPATIBLE_API_KEY": seam.target["api_key"]})
    assert shown.first_contact == moved.first_contact == seam.cw.first_contact_scope(target, CAP) != plan.first_contact
    fit = context_fit.measure_main_fit(shown, messages, start.tools, profile="owner_max", rendered_mode=mode, round_id=ROUND)
    signed = seam.cw.signed_first_contact(context_fit._physical_context_for_fit(fit, shown), target,
              {"model": target["resolved_model"], "messages": [{"role": "user", "content": "sample"}], "tools": [FINISH_TASK]}, CAP)
    assert signed.window_witness_bound and signed.first_contact_scope == shown.first_contact


def test_a_wrap_up_before_the_first_hinted_send_prices_and_sends_the_ordinary_view(seam, tmp_path, monkeypatch):
    """No send gate guards a forced call, so a wrap-up before the hint's first send takes the ordinary view before it
    is priced; once the hinted view was sent the wrap-up keeps it."""
    from ouroboros import loop_forced_finalization as forced

    start = _hinted_start(seam, tmp_path, monkeypatch)
    for plan, plain in ((start.plan, True), (start.plan.first_contact_spent(), False)):
        start.registry._ctx.context_fit_plan, start.registry._ctx.active_context_mode = plan, plan.initial_mode
        ctx = SimpleNamespace(tools=start.registry, messages=plan.messages_for(plan.initial_mode), active_model=MODEL,
                              active_use_local=False, active_effort="", model_role="main", tool_schemas=start.tools)
        assert forced._bind_forced_route(ctx) is plain
        held = start.registry._ctx.context_fit_plan
        assert (NOTE in json.dumps(ctx.messages[0])) is not plain and held.first_contact_normal is None
        assert held.initial_mode == (start.ordinary.initial_mode if plain else plan.initial_mode)


@pytest.mark.parametrize("maps_rung", [False, True], ids=["folded-memory", "chapter-maps"])
def test_a_hinted_start_names_its_own_folds_and_no_comparison_repeats_it(seam, tmp_path, monkeypatch, maps_rung):
    """Its receipt names the cuts the first send's preference took (never the window), keeps naming them after a
    selection redraws the view, and the pairwise start fit of the redrawn plan names the start as it was (Max with
    chapter maps when the hint chose maps) while leaving each arm unmeasured: the hint's sizes are not this plan's to
    repeat, and an authored redraw carries the start's own fact, never measuring it again."""
    from ouroboros import memory_floor
    from ouroboros.memory_view import snapshot_from_json

    start = _hinted_start(seam, tmp_path, monkeypatch, maps_rung=maps_rung).plan
    view = start.projection(start.initial_mode)
    assert view.memory_facts["floor"].get("preference") is not None  # the steps taken before its soft ask
    assert "first send's smaller view" in view.system_content_json and "cuts for the window" not in view.system_content_json
    plan = memory_floor.select_view(start, start.initial_mode, [])  # the mind's own redraw
    redrawn = plan.projection(plan.initial_mode)
    assert "first send's smaller view" in redrawn.system_content_json and "cuts for the window" not in redrawn.system_content_json
    assert redrawn.memory_facts.get("first_contact_preference") == view.memory_facts["first_contact_preference"]
    fit = memory_floor.counterfactual_starts(plan, {"now": snapshot_from_json(plan.core.memory_view_json)})
    assert fit["frame"]["maps"] is maps_rung and "unmeasured" in fit["arms"]["now"]
    lines = "\n".join(memory_floor.counterfactual_lines(fit))
    assert ("started in Max with chapter maps" in lines) is maps_rung and "not measured" in lines


@pytest.mark.parametrize("maps_rung", [False, True], ids=["folded-memory", "chapter-maps"])
@pytest.mark.parametrize("phase,boundary", [("ready", "resumed"), ("candidate", "candidate")])
def test_a_cold_restore_reads_only_the_start_the_continuation_actually_carried(
        seam, tmp_path, monkeypatch, maps_rung, phase, boundary):
    from ouroboros import context_fit, memory_floor, owner_wait, working_checkpoint
    from ouroboros.context_measurement import mode_cause
    from ouroboros.memory_view import snapshot_from_json

    state = _hinted_start(seam, tmp_path, monkeypatch, maps_rung=maps_rung)
    hinted, ordinary, registry = state.plan, state.ordinary, state.registry
    ctx = registry._ctx
    snapshot = snapshot_from_json(ordinary.core.memory_view_json)

    def save(plan):
        ctx.context_fit_plan = plan
        return json.loads(json.dumps(owner_wait.continuation_state(ctx, [], {}, {}, 1, [], set())))

    lost = save(hinted)  # no physical dispatch
    assert lost["context_start"] == list(context_fit.UNKNOWN_START)
    unknown = context_fit.restored_start(ordinary, lost["context_start"])
    assert "unmeasured" in memory_floor.counterfactual_starts(unknown, {"now": snapshot})["arms"]["now"]
    assert save(hinted.ordinary_first_contact())["context_start"] == list(ordinary.start)

    def resume_write(source):
        ctx.context_fit_plan = ordinary  # the new worker's capture, before exact Resume writes its checkpoint
        source.update(resume_point={"phase": phase}, context_model_role="fallback:source-role")
        working_checkpoint.save_resumed(registry, source, [], {}, {}, set())
        _, read = working_checkpoint.read_working(working_checkpoint._root(ctx), ctx.task_id, 1)
        assert read["working"]["boundary"] == boundary
        return read

    read = resume_write(lost)
    assert read["context_start"] == list(context_fit.UNKNOWN_START)
    assert read["context_model_role"] == lost["context_model_role"]
    saved = save(hinted.first_contact_spent())
    read = resume_write(saved)
    assert read["context_start"] == saved["context_start"] != list(ordinary.start)
    assert read["context_model_role"] == saved["context_model_role"]
    restored = context_fit.restored_start(ordinary, read["context_start"])
    assert not restored.projection("max").memory_facts.get("first_contact_preference")  # new capture, old start
    after = memory_floor.counterfactual_starts(restored, {"now": snapshot})
    assert after["frame"]["maps"] is maps_rung and "first send's preference" in after["arms"]["now"]["unmeasured"]
    assert mode_cause(restored, hinted.initial_mode) == ""  # not a physical Low cause
    assert saved["context_start"] == save(restored)["context_start"]
