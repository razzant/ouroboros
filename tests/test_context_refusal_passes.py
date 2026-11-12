"""The refusal path sizes its model-free passes on this refusal and this conversation (17.10.2026).

86aa died on a 933k-token request the host had estimated at 678k with another task's density,
so no pass knew how much to shed. These pins keep the target measured on the conversation's
own anchor (unknown when nothing comparable measures it) and each pass shedding only what its
earlier steps left of that target: the gradual order the owner asked for, never a fixed share.
"""

from __future__ import annotations

from types import SimpleNamespace
import pytest

from ouroboros import loop_model_call as lmc
from ouroboros.context_anchor import REFUSED_INPUTS_KEY
from tests.test_context_economy_window import live_wait as wait_fixture
from tests.test_context_economy_window import main_call as main_fixture
from tests.test_context_economy_window import setup as gateway_fixture

setup, live_wait, main_call = gateway_fixture, wait_fixture, main_fixture  # the real Main call fixtures


def _fit(estimate, *, anchor=874_829, accepted=921_840, reply=65_536):
    return SimpleNamespace(measurement=SimpleNamespace(
        estimated_input_tokens=estimate, anchor_input_tokens=anchor, accepted_input_tokens=accepted,
        capacity_total_tokens=None, window_reading_tokens=None, reply_allowance_tokens=reply,
        capacity_deficit_tokens=None))


def _ctx(message=""):
    return SimpleNamespace(accumulated_usage={"_last_llm_provider_message": message}, active_context_mode="max",
                           context_fit_plan=SimpleNamespace(provider="claudexor", model="codex=gpt-6-astra"))


def _target(ctx, fit, refused=None):
    return lmc._refusal_target(ctx, fit, refused)[0]


def _refused_size(ctx):
    (size,) = ctx.accumulated_usage[REFUSED_INPUTS_KEY].values()
    return size


def test_the_target_is_measured_on_this_conversation_or_stays_unknown():
    ctx = _ctx()
    # 86aa r1482: E = its raw 848,751 at its own density 874,829/795,458; A = the largest input Codex accepted.
    assert lmc._refusal_target(ctx, _fit(933_440)) == (933_440 - 921_840 + 115_230, 115_230)  # + ceil(A/8), L33 2=A
    assert ctx.accumulated_usage[REFUSED_INPUTS_KEY]  # recorded: the facts stop stating free room
    assert _target(_ctx(), _fit(900_000)) is None  # E at or below A contradicts the reading
    assert _target(_ctx(), _fit(933_440, anchor=None)) is None  # a cold count sizes nothing
    stated = _ctx("This model's maximum context length is 400,000 tokens.")
    assert _target(stated, _fit(390_000, accepted=None, reply=20_000)) == 10_000 + 50_000
    # Anthropic and Gemini wordings state both numbers; the provider's own count beats the estimate,
    # and a reply the refused request reserved, where stated, is the one the retry reserves again.
    for message, size, reply, limit in (
            ("prompt is too long: 215000 tokens > 200000 maximum", 215_000, 8_192, 200_000),
            ("input length and `max_tokens` exceed context limit: 190000 + 65536 > 200000", 190_000, 65_536, 200_000),
            ("The input token count (1050000) exceeds the maximum number of tokens allowed (1048576).",
             1_050_000, 8_192, 1_048_576),
            ("This model's maximum context length is 128000 tokens. However, your messages resulted in "
             "131000 tokens.", 131_000, 8_192, 128_000)):
        refusal = _ctx(message)
        assert _target(refusal, _fit(1, anchor=None, accepted=None, reply=8_192)) == (
            max(0, size + reply - limit) + -(-limit // 8)), message
        assert _refused_size(refusal) == size, message
    # A local serving instance that refused before sending gives its own exact counts (#1018 lane).
    local = SimpleNamespace(input_tokens=34_000, context_window_tokens=32_768)
    assert _target(_ctx(), _fit(30_000, anchor=None, accepted=None, reply=8_192), local) == (
        34_000 + 8_192 - 32_768 + 4_096)
    assert _target(_ctx(), _fit(30_000, anchor=None, accepted=None, reply=8_192)) is None  # cold, numberless
    # A stated limit beside a COLD estimate measures nothing (chars/4 under-counts code and Cyrillic): unknown;
    # with the provider's own count of the refused request it measures even the first refusal (review r2, 18.10).
    assert _target(stated, _fit(390_000, anchor=None, accepted=None, reply=20_000)) is None


def test_a_providers_requested_total_holds_the_reply_once():
    """Review r3 (18.10): "you requested (about) N tokens" is the input plus the reply that request reserved.

    Read as input, it added the reply twice: 96,072 instead of 30,536 on OpenAI's own wording."""
    openai = _ctx("This model's maximum context length is 200000 tokens. However, you requested 205536 tokens "
                  "(140000 in the messages, 65536 in the completion). Please reduce the length of the messages "
                  "or completion.")
    assert lmc._refusal_target(openai, _fit(1, anchor=None, accepted=None, reply=20_000)) == (5_536 + 25_000, 25_000)
    assert _refused_size(openai) == 140_000
    openrouter = _ctx("This endpoint's maximum context length is 128000 tokens. However, you requested about 129782 "
                      "tokens (928 of text input, 854 of tool input, 128000 in the output). Please reduce the length "
                      "of either one, or use the \"middle-out\" transform to compress your prompt automatically.")
    assert _target(openrouter, _fit(1, anchor=None, accepted=None, reply=20_000)) == 1_782 + 16_000
    assert _refused_size(openrouter) == 1_782
    parts = _ctx("This endpoint's maximum context length is 200000 tokens. However, you requested about 237000 tokens "
                 "(205000 of text input, 32000 in the output).")
    assert _target(parts, _fit(205_000, accepted=None, reply=32_000)) == 37_000 + 25_000  # anchored E agrees
    # OpenAI counts tool definitions as their own share ("in the functions"); the input is the total less the
    # reply whatever parts it lists (review r4, 18.10: the messages share alone shed 16,000 of a 38,000 need).
    tools = _ctx("This model's maximum context length is 128000 tokens. However, you requested 150000 tokens "
                 "(110000 in the messages, 25000 in the functions, and 15000 in the completion). Please reduce the "
                 "length of the messages, functions, or completion.")
    assert _target(tools, _fit(1, anchor=None, accepted=None, reply=20_000)) == 22_000 + 16_000
    assert _refused_size(tools) == 135_000
    whole = _ctx("This model's maximum context length is 128000 tokens. However, your messages resulted in 140000 "
                 "tokens (115000 in the messages, 25000 in the functions). Please reduce the length of the messages "
                 "or functions.")
    assert _target(whole, _fit(1, anchor=None, accepted=None, reply=8_192)) == 140_000 + 8_192 - 128_000 + 16_000
    assert _refused_size(whole) == 140_000
    # A bare total already holds its reply: the excess is the total over the limit, the input the rest.
    bare = _ctx("This endpoint's maximum context length is 400000 tokens. However, you requested about 410000 tokens")
    assert _target(bare, _fit(300_000, anchor=None, accepted=None, reply=20_000)) == 10_000 + 50_000
    assert _refused_size(bare) == 390_000


def test_a_refusal_is_measured_on_the_bytes_it_refused_and_a_rebound_view_is_not_asked_twice():
    """Astra 11(ii) and R4 A4: the refused size is the last physical attempt's own (its capture's estimate of the bytes
    it sent), not the candidate a rebind narrowed since: 950k refused stays recorded as 950k beside a 600k view. Beside
    an unchanged view the measured excess holds; beside a rebound one, what that view already dropped is not asked."""
    fit = SimpleNamespace(measurement=SimpleNamespace(**vars(_fit(600_000, accepted=900_000).measurement),
                                                      measurement_density=1.0))
    sent = SimpleNamespace(prompt_tokens_bounded_estimate=950_000, physical_context=SimpleNamespace(route_fp="r"))
    for rebound, target in ((False, 50_000 + 112_500), (True, 112_500)):  # E - A = 950k - 900k; slack ceil(A/8)
        ctx = _ctx()
        assert lmc._refusal_target(ctx, fit, sent, rebound=rebound) == (target, 112_500)
        assert _refused_size(ctx) == 950_000


def test_a_numberless_first_refusal_drops_the_slack_and_a_further_one_the_rest(tmp_path, monkeypatch):
    """R4 A2: a first-step refusal with no number and no anchor beside a known window, the view unchanged and the
    host's own estimate fitting (no deficit), measures no excess. The first pass asks memory for the owner's
    return-after-fold slack, ceil(W/8): a strictly smaller retry that keeps most of memory's detail, never the
    shortest view at once; after a further refusal memory's rest takes everything it can give."""
    import ouroboros.loop as loop
    from ouroboros import context_fit as cf, memory_view as mv
    from tests import _memory_view_synthetic as syn

    words = [syn.conversation(i, words=40, word_chars=2_000) for i in range(30)]
    core = cf.ContextCore("Core", "", "", "", "", "", '"Continue."', False,
                          memory_view_json=mv.snapshot_json(syn.snapshot(room_facts=syn.room("1"), conversations=words)))
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    evidence = SimpleNamespace(route_fp="probe", status="confirmed", stale=False, window_tokens=1_000_000, source="fixture")
    plan = cf.build_context_fit_plan(SimpleNamespace(drive_root=tmp_path), core, {"id": "a2", "_is_direct_chat": True},
        preferred_mode="max", route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "synthetic"}, evidence))
    ctx = SimpleNamespace(context_fit_plan=plan, active_context_mode=plan.initial_mode, tool_schemas=[], task_id="a2",
        tools=SimpleNamespace(_ctx=SimpleNamespace()), messages=plan.messages_for(plan.initial_mode),
        event_queue=None, drive_logs=tmp_path, round_idx=1)

    def facts():
        shown = ctx.context_fit_plan.projection(ctx.active_context_mode)
        return shown.memory_facts["floor"]["steps"].get("F6w", 0), shown.memory_facts["blocks"]["room_tokens"]

    fit = SimpleNamespace(measurement=SimpleNamespace(capacity_total_tokens=1_000_000, measurement_density=1.0,
        reprojection_input_tokens=None, estimated_input_tokens=0, capacity_deficit_tokens=0, route_fp="probe"))
    monkeypatch.setattr(loop, "_measure_round_main_fit", lambda *_a, **_kw: fit)
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **_kw: None)
    folded, room = facts()
    assert folded == 0 and room > 500_000  # the start held every word
    slack = lmc._refusal_target(_ctx(), SimpleNamespace(measurement=SimpleNamespace(**{
        **vars(_fit(0, anchor=None).measurement), "capacity_total_tokens": 1_000_000, "measurement_density": 1.0})))[1]
    assert slack == 125_000
    lmc._apply_refusal_step(ctx, "floor", fit, None, slack)
    first, after_first = facts()
    assert 0 < first < 1_200 and slack <= room - after_first < 2 * slack  # about the slack, most words still whole
    lmc._apply_refusal_step(ctx, "memory_rest", fit, None, slack)
    assert facts()[0] == 1_200 and facts()[1] < after_first - slack  # the rest: every word by address


def test_a_refused_first_send_retries_the_view_its_route_rebind_narrowed(main_call, monkeypatch, tmp_path):
    """The path of 07.11 at once (window C, A and 1.8): an Auto start with no known window sends its whole memory and
    Codex refuses one input string's length, its route naming the refusing account but no model. The receipt names the
    requested model, so the rebind binds that account and its known window narrows the view; that view is retried
    first on the same route. The configured chain never starts, no refusal pass folds memory, and memory keeps every
    word the narrowed view kept instead of its shortest forms."""
    import json

    from ouroboros import capability_evidence as ce, context, memory_view as mv
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
    from tests import _memory_view_synthetic as syn
    from tests.test_context_economy_window import MODEL, ROUTE, _answer, _drained, _native, _run
    from tests.test_llm_claudexor import result

    words = syn.live_scale(groups=60, word_chars=1_000)
    ctx, gateway, tools, events, plan, windows = _native(main_call, monkeypatch, tmp_path, owner_mode="max",
                                                          view=mv.snapshot_json(words))
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "", "fallback": ["account-b"]}))  # Main runs Auto
    windows.update({"account-a": 0, "account-b": 400_000})

    def route(task, **_kw):  # the account an operation's receipt binds (``model_account_options``), else Auto
        options = ce.model_account_options(MODEL, role=task["model_role"], model_route=task.get("model_route"))
        account = options["credential_profile_id"] if options.get("account_fingerprint") else "account-a"
        bound = {"source_id": "codex", "credential_profile_id": account, "account_fingerprint": f"fingerprint-{account[-1]}"}
        return {"model": MODEL, "provider": "claudexor", "options": bound}, SimpleNamespace(
            route_fp=ce.route_fingerprint(provider="claudexor", model=MODEL, base_url="", options=bound), stale=False,
            status="confirmed" if windows[account] else "unprobeable", window_tokens=windows[account], source="advertised",
            source_id="codex", credential_profile_id=account, account_fingerprint=bound["account_fingerprint"])
    monkeypatch.setattr(context, "_context_fit_route", route)
    built = plan(0)
    ctx.messages[:] = built.messages_for(built.initial_mode)
    refusing = {"source": "codex", "credentialProfileId": "account-b", "accountFingerprint": "fingerprint-b", "model": None}
    gateway.results = [
        result(outcome="failed", route=refusing, problem={"code": "invalid_request", "message": "Codex model request was "
               "refused (HTTP 400).", "context": {"httpStatus": 400, "vendorCode": "string_above_max_length",
                                                  "parameter": "input[3].content[0].text"}}),
        _answer({**ROUTE, "credentialProfileId": "account-b", "accountFingerprint": "fingerprint-b"},
                {"role": "assistant", "content": "Done."})]
    gateway.dispatch = ["response_received"] * 2
    text, _usage, _trace = _run(ctx, tools, events, built)
    assert text == "Done." and len(gateway.creates) == 2  # the narrowed view answered
    first, retry = (payload for payload, _key in gateway.uploads)
    # Auto again, no chain: the unknown start named no account; the retry names the account its rebound plan rests on
    # (account-b's known window sized it) as a preference, never the fallback's pin (W2 H5).
    assert (first["account"], retry["account"]) == ({"mode": "auto"}, {"mode": "auto", "preferredProfileId": "account-b"})
    assert len(json.dumps(retry["messages"])) < len(json.dumps(first["messages"]))  # strictly smaller, same route
    rebound = tools._ctx.context_fit_plan
    assert rebound.model_route["accountFingerprint"] == "fingerprint-b" and rebound.window_record.usable is not None
    kinds = [row.get("checkpoint_kind") for row in _drained(events)]
    assert "context_fit_route_rebound" in kinds and "context_memory_coarsened" not in kinds  # no pass folded memory
    folded = rebound.projection(rebound.initial_mode).memory_facts["floor"]["steps"].get("F6w", 0)
    assert 0 < folded < 60 * 33  # narrowed for the account's window, never every word


def test_the_size_parser_reads_input_reply_and_total_apart():
    from ouroboros.capability_evidence import _parse_ctx_requested

    assert _parse_ctx_requested("However, you requested 205536 tokens (140000 in the messages, 65536 in the "
                                "completion).") == (140_000, 65_536, 205_536)
    assert _parse_ctx_requested("you requested about 129782 tokens (928 of text input, 854 of tool input, "
                                "128000 in the output)") == (1_782, 128_000, 129_782)
    assert _parse_ctx_requested("you requested about 237000 tokens (32000 in the output)") == (205_000, 32_000, 237_000)
    assert _parse_ctx_requested("you requested about 410000 tokens") == (0, 0, 410_000)  # a total, never an input
    assert _parse_ctx_requested("you requested 150000 tokens (110000 in the messages, 25000 in the functions, "
                                "and 15000 in the completion)") == (135_000, 15_000, 150_000)
    assert _parse_ctx_requested("your messages resulted in 140000 tokens (115000 in the messages, 25000 in the "
                                "functions)") == (140_000, 0, 0)  # the whole input, not its messages share
    assert _parse_ctx_requested("you requested 140000 tokens (115000 in the messages, 25000 in the functions)") == (
        140_000, 0, 140_000)  # no reply share: the parts are the input
    assert _parse_ctx_requested("context limit: 190000 + 65536 > 200000") == (190_000, 65_536, 0)
    assert _parse_ctx_requested("The input token count (1050000) exceeds the limit") == (1_050_000, 0, 0)
    assert _parse_ctx_requested("Your input exceeds the context window of this model") == (0, 0, 0)


def test_bodies_shed_only_what_the_host_copies_left_of_the_measured_excess(monkeypatch):
    """Audit of early stops (06.11): shown bodies pay the measured excess only; the return-after-fold
    slack is memory's to pay, so bodies the host copies already covered stay in place."""
    calls, shed = [], {}

    def address(_ctx, _fit, *, rung, goal_tokens=None):
        calls.append((rung, goal_tokens))
        return SimpleNamespace(status="applied" if rung in shed else "no_eligible", reclaimed_tokens=shed.get(rung, 0))

    monkeypatch.setattr(lmc, "_run_emergency_address_pass", address)
    first = ("host_copies", "bodies")
    shed.update(host_copies=6_000, bodies=70_000)  # target 126,830 = excess 11,600 + slack 115,230
    assert lmc._apply_refusal_pass(_ctx(), first, _fit(1), 126_830, 115_230) is True
    assert calls == [("host_copies", None), ("bodies", 5_600)]  # the rest of the excess, never the slack
    calls.clear()
    shed.update(host_copies=60_000)
    assert lmc._apply_refusal_pass(_ctx(), first, _fit(1), 126_830, 115_230) is True
    assert calls == [("host_copies", None)]  # superseded host state paid the excess: every body stays
    calls.clear()
    assert lmc._apply_refusal_pass(_ctx(), first, _fit(1), 126_830) is True  # the slack read from the window
    assert calls == [("host_copies", None)]
    calls.clear()
    assert lmc._apply_refusal_pass(_ctx(), first, _fit(1), None) is True
    assert calls == [("host_copies", None), ("bodies", None)]  # unknown: each step takes its category
    calls.clear()
    shed.clear()
    assert lmc._apply_refusal_pass(_ctx(), first, _fit(1), 126_830, 115_230) is False  # nothing changed, no retry


def test_bodies_pay_the_excess_memory_the_slack_and_books_wait_for_a_further_refusal(monkeypatch):
    """Owner order 7A with the 06.11 sizing: superseded host rows, shown bodies for the measured excess,
    memory's detail for the rest and the return-after-fold slack; books and unseen bodies only in the
    second pass, after a further refusal."""
    import ouroboros.loop as loop

    calls, sizes = [], []
    monkeypatch.setattr(lmc, "_coarsen_memory_view", lambda _ctx, _fit, *, saving=None, slack=0: (
        calls.append(("floor", saving, slack)) or True))
    monkeypatch.setattr(lmc, "_reproject_actual_overflow_low",
                        lambda ctx: calls.append(("books",)) or setattr(ctx, "active_context_mode", "low"))
    monkeypatch.setattr(loop, "_measure_round_main_fit", lambda _ctx, **_kw: _fit(sizes.pop(0)))
    monkeypatch.setattr(lmc, "_run_emergency_address_pass", lambda _ctx, _fit, *, rung, goal_tokens=None: (
        calls.append((rung, goal_tokens)) or SimpleNamespace(status="applied", reclaimed_tokens=goal_tokens or 0)))
    first, second = lmc.REFUSAL_PASSES
    assert first == ("host_copies", "bodies", "floor") and second == ("books", "unseen_bodies")
    stated = _ctx("This model's maximum context length is 400,000 tokens.")  # slack 50,000
    sizes[:] = [830_000]  # after memory narrowed
    assert lmc._apply_refusal_pass(stated, first, _fit(900_000, accepted=None), 70_000, 50_000) is True
    assert calls == [("host_copies", None), ("bodies", 20_000), ("floor", 0, 50_000)]  # slack alone: no excess
    assert stated.active_context_mode == "max"  # books stay through the first pass
    calls.clear()
    sizes[:] = [880_000, 830_000]  # before and after the books step
    assert lmc._apply_refusal_pass(stated, second, _fit(900_000, accepted=None), 70_000, 50_000) is True
    assert calls == [("books",), ("unseen_bodies", 20_000)]  # books first, then only what they left
    assert stated.active_context_mode == "low"
    unsized = _fit(900_000, accepted=None)  # no window reading anywhere: no slack, the whole target is excess
    calls.clear()
    sizes[:] = [800_000]
    assert lmc._apply_refusal_pass(_ctx(), first, unsized, None) is True
    # Unknown: memory's detail pays all it can, a whole-or-outline section nothing without a measured excess.
    assert calls == [("host_copies", None), ("bodies", None), ("floor", 0, lmc.WHOLE_CATEGORY)]
    calls.clear()
    sizes[:] = [800_000, 790_000, 780_000]  # the books step's before and after, then memory's rest
    assert lmc._apply_refusal_pass(_ctx(), second, unsized, None) is True
    # After a further refusal memory's rest may take everything, whole-or-outline sections included.
    assert calls == [("books",), ("unseen_bodies", None), ("floor", lmc.WHOLE_CATEGORY, 0)]
    calls.clear()
    sizes[:] = [800_000]
    known = SimpleNamespace(measurement=SimpleNamespace(**{**vars(unsized.measurement), "capacity_deficit_tokens": 18_887}))
    assert lmc._apply_refusal_pass(_ctx(), first, known, None) is True
    assert calls[-1] == ("floor", 18_887, 0)  # no target, but a known window's deficit sizes memory (no slack here)


def test_a_failed_candidates_walk_keeps_the_rounds_refusal_and_the_conversation_facts():
    """Review r2 (18.10): restoring the primary's route-fit state after a failed fallback erased the round's
    size-refusal stamp (so the renewal never ran) and the per-model anchors. Route-fit keys still restore."""
    from ouroboros.context_anchor import ACCEPTED_INPUTS_KEY, INPUT_ANCHOR_KEY, REFUSED_INPUTS_KEY

    usage = {"_context_capacity_total_tokens": 872_000, INPUT_ANCHOR_KEY: {"a": {}}}
    snapshot = lmc._snapshot_context_fit_usage(usage)
    usage.update({"_context_capacity_total_tokens": 128_000, lmc.CONTEXT_REFUSAL_ROUND_KEY: 7,
                  INPUT_ANCHOR_KEY: {"a": {}, "b": {}}, ACCEPTED_INPUTS_KEY: {"b": 90_000}, REFUSED_INPUTS_KEY: {"b": 130_000}})
    lmc._restore_context_fit_usage(usage, snapshot)
    assert usage["_context_capacity_total_tokens"] == 872_000  # route-fit state is the primary's again
    assert usage[lmc.CONTEXT_REFUSAL_ROUND_KEY] == 7 and set(usage[INPUT_ANCHOR_KEY]) == {"a", "b"}
    assert usage[ACCEPTED_INPUTS_KEY] == {"b": 90_000} and usage[REFUSED_INPUTS_KEY] == {"b": 130_000}


def test_a_local_pre_send_refusal_uses_the_reply_its_exact_request_reserved():
    """Review r2 (18.10): a 131,072 window, 125,000 exact input and the 8,192 reply floor of the refused
    request exceed by 2,120, not by the estimate's 14,688-token reply."""
    local = SimpleNamespace(input_tokens=125_000, context_window_tokens=131_072, max_completion_tokens=8_192)
    assert _target(_ctx(), _fit(100_000, anchor=None, accepted=None, reply=14_688), local) == 2_120 + -(-131_072 // 8)


def test_books_pay_remaining_room_before_any_unseen_body(monkeypatch):
    """The owner's 22 October priority: memory and host books yield before chosen bodies; since 06.11 books
    wait for the second pass, after a further refusal."""
    import ouroboros.loop as loop

    calls, sizes, unseen = [], [], {}
    monkeypatch.setattr(lmc, "_reproject_actual_overflow_low",
                        lambda ctx: calls.append(("books",)) or setattr(ctx, "active_context_mode", "low"))
    monkeypatch.setattr(loop, "_measure_round_main_fit", lambda _ctx, **_kw: _fit(sizes.pop(0)))
    monkeypatch.setattr(lmc, "_run_emergency_address_pass", lambda _ctx, _fit, *, rung, goal_tokens=None: (
        calls.append((rung, goal_tokens)) or SimpleNamespace(status=unseen.get("status", "no_eligible"),
                                                             reclaimed_tokens=unseen.get("shed", 0))))
    stated = _ctx("This model's maximum context length is 400,000 tokens.")  # slack 50,000
    second = lmc.REFUSAL_PASSES[1]
    sizes[:] = [880_000, 810_000]  # before and after the books step
    assert lmc._apply_refusal_pass(stated, second, _fit(900_000, accepted=None), 70_000) is True
    assert calls == [("books",)]  # the books paid the whole goal: unseen bodies stay
    assert stated.active_context_mode == "low"
    calls.clear()
    paid = _ctx("This model's maximum context length is 400,000 tokens.")
    unseen.update(status="applied", shed=50_000)
    sizes[:] = [880_000, 860_000]
    assert lmc._apply_refusal_pass(paid, second, _fit(900_000, accepted=None), 70_000) is True
    assert calls == [("books",), ("unseen_bodies", 50_000)]  # then only what the books left


@pytest.mark.parametrize("refused", [False, True])
def test_a_just_chosen_body_pays_last_and_only_what_a_providers_refusal_left(monkeypatch, refused):
    """Owner 2A: an estimate never cuts a chosen read. After a provider's refusal memory, books and unasked
    bodies go first; a chosen body that fits alone pays only what they left (everything when nothing measures it)."""
    shed, chosen = {}, []
    monkeypatch.setattr(lmc, "_apply_refusal_step", lambda _ctx, step, _fit, _target, _slack=None: shed.get(step))
    monkeypatch.setattr(lmc, "_run_emergency_address_pass", lambda _ctx, _fit, *, rung, goal_tokens=None: (
        chosen.append((rung, goal_tokens)) or SimpleNamespace(status="applied", reclaimed_tokens=1)))
    first, second = lmc.REFUSAL_PASSES
    for steps, target, paid, expected in (
            (second, None, {}, [("chosen_bodies", None)]),  # nothing else changed an unmeasured refusal
            (second, None, {"books": 0}, []),  # the books changed the candidate: its retry answers first
            (second, 100_000, {"books": 30_000, "unseen_bodies": 10_000}, [("chosen_bodies", 60_000)]),
            (second, 100_000, {"books": 100_000}, []),  # the books paid the whole measured target
            (first, None, {}, [])):  # the first pass never reaches a chosen body
        shed.clear(), shed.update(paid), chosen.clear()
        changed = lmc._apply_refusal_pass(_ctx(), steps, _fit(1), target, None, refused)
        assert chosen == (expected if refused else [])
        assert changed is bool(paid or chosen)


@pytest.mark.parametrize("ceiling,mode", [(0, "low"), (200_000, "low"), (50_000, "nano")])
def test_books_recovery_keeps_the_current_refusal_frame_and_selected_mode(tmp_path, monkeypatch, ceiling, mode):
    from ouroboros import capability_evidence as ce, context_fit as cf, memory_view as mv
    from ouroboros.context_window import record_observation
    from tests import _memory_view_synthetic as syn
    import ouroboros.loop as loop

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    snapshot = syn.snapshot(story=syn.pages(2, 200), room_facts=syn.room())
    core = cf.ContextCore("Core", "", "Whole books. " * 8_000, "", "", "", '"Assignment"', False,
                          memory_view_json=mv.snapshot_json(snapshot))
    plan = cf.build_context_fit_plan(None, core, {"type": "task"}, preferred_mode="max",
        route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "test"},
            ce.CapabilityEvidence(1_000_000, "confirmed", "metadata", "books-route")))
    assert plan.initial_mode == "max"
    ctx = SimpleNamespace(context_fit_plan=plan, active_context_mode="max", messages=plan.messages_for("max"),
        tools=SimpleNamespace(_ctx=SimpleNamespace()), tool_schemas=[], accumulated_usage={}, task_id="books01",
        event_queue=None, drive_logs=tmp_path / "logs", round_idx=2)
    def measure(ctx, **_kwargs):
        return cf.measure_main_fit(ctx.context_fit_plan, ctx.messages, [],
            profile=lmc._main_context_profile(ctx.context_fit_plan, ctx.active_context_mode),
            rendered_mode=ctx.active_context_mode, round_id="books")
    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **_kw: None)
    monkeypatch.setattr(lmc, "invalidate_task_cache_splits", lambda *_a: None)
    if ceiling:
        record_observation(tmp_path, plan.route_fp, refused=ceiling)
    lmc._reproject_actual_overflow_low(ctx)
    assert ctx.active_context_mode == ctx.tools._ctx.active_context_mode == mode
    from ouroboros.transcript_prefix import SCOPED_SANCTION_ATTR  # the next send names this rebuild (C2 S4)
    assert getattr(ctx.tools._ctx, SCOPED_SANCTION_ATTR) == {"system": "low_retry"}
    assert ctx.messages[0] == ctx.context_fit_plan.projection(mode).system_message()
    assert ctx.context_fit_plan.window_record is not None
    assert measure(ctx).measurement.capacity_total_tokens == (ceiling or 1_000_000)
    assert ctx.context_fit_plan.window_record.source.startswith("refusal" if ceiling else "metadata")
