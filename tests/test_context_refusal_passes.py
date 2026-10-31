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


def test_bodies_shed_only_what_the_host_copies_left_of_the_target(monkeypatch):
    calls, shed = [], {}

    def address(_ctx, _fit, *, rung, goal_tokens=None):
        calls.append((rung, goal_tokens))
        return SimpleNamespace(status="applied" if rung in shed else "no_eligible", reclaimed_tokens=shed.get(rung, 0))

    monkeypatch.setattr(lmc, "_run_emergency_address_pass", address)
    first = ("host_copies", "bodies")
    shed.update(host_copies=60_000, bodies=70_000)
    assert lmc._apply_refusal_pass(_ctx(), first, _fit(1), 126_830) is True
    assert calls == [("host_copies", None), ("bodies", 66_830)]  # the rest, not the whole target again
    calls.clear()
    assert lmc._apply_refusal_pass(_ctx(), first, _fit(1), 50_000) is True
    assert calls == [("host_copies", None)]  # superseded host state freed the room: every body stays
    calls.clear()
    assert lmc._apply_refusal_pass(_ctx(), first, _fit(1), None) is True
    assert calls == [("host_copies", None), ("bodies", None)]  # unknown: each step takes its category
    calls.clear()
    shed.clear()
    assert lmc._apply_refusal_pass(_ctx(), first, _fit(1), 126_830) is False  # nothing changed, no retry


def test_memory_narrowed_enough_addresses_no_unseen_body(monkeypatch):
    import ouroboros.loop as loop

    calls, sizes = [], []
    monkeypatch.setattr(lmc, "_coarsen_memory_view", lambda _ctx, _fit, *, saving=None: calls.append(("floor", saving)) or True)
    monkeypatch.setattr(lmc, "_reproject_actual_overflow_low",
                        lambda ctx: calls.append(("books",)) or setattr(ctx, "active_context_mode", "low"))
    monkeypatch.setattr(loop, "_measure_round_main_fit", lambda _ctx, **_kw: _fit(sizes.pop(0)))
    monkeypatch.setattr(lmc, "_run_emergency_address_pass", lambda _ctx, _fit, *, rung, goal_tokens=None: (
        calls.append((rung, goal_tokens)) or SimpleNamespace(status="applied", reclaimed_tokens=goal_tokens)))
    second = lmc.REFUSAL_PASSES[1]
    unsized = _fit(900_000, accepted=None)  # no window reading anywhere: no slack, the whole target is excess
    sizes[:] = [800_000]
    assert lmc._apply_refusal_pass(_ctx(), second, unsized, 80_000) is True
    assert calls == [("floor", 80_000)]  # memory alone saved it: books stay, no unseen body leaves
    calls.clear()
    sizes[:] = [800_000, 760_000]
    assert lmc._apply_refusal_pass(_ctx(), second, unsized, 150_000) is True
    assert calls == [("floor", 150_000), ("books",), ("unseen_bodies", 10_000)]  # books, then only the rest
    calls.clear()
    sizes[:] = [800_000, 760_000]
    assert lmc._apply_refusal_pass(_ctx(), second, unsized, None) is True
    assert calls == [("floor", lmc.WHOLE_CATEGORY), ("books",), ("unseen_bodies", None)]  # unknown: each whole
    calls.clear()
    sizes[:] = [800_000, 760_000]
    known = SimpleNamespace(measurement=SimpleNamespace(**{**vars(unsized.measurement), "capacity_deficit_tokens": 18_887}))
    assert lmc._apply_refusal_pass(_ctx(), second, known, None) is True
    assert calls[0] == ("floor", 18_887)  # no target, but a known window's deficit sizes memory, not "everything"
    # Memory narrows by the measured excess only; the return-after-fold slack falls to books and sources
    # (review r1, 17.10: narrowing memory for slack showed this room's own rows by address).
    calls.clear()
    sizes[:] = [880_000, 880_000, 700_000]
    stated = _ctx("This model's maximum context length is 400,000 tokens.")  # slack 50,000
    assert lmc._apply_refusal_pass(stated, second, _fit(900_000, accepted=None), 70_000) is True
    # Memory paid the excess; books pay the remaining work room before an unseen body leaves.
    assert calls == [("floor", 20_000), ("books",)]


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
    """The owner's 22 October priority: memory and host books yield before chosen bodies."""
    import ouroboros.loop as loop

    calls, sizes, unseen = [], [], {}
    monkeypatch.setattr(lmc, "_coarsen_memory_view", lambda _ctx, _fit, *, saving=None: calls.append(("floor", saving)) or True)
    monkeypatch.setattr(lmc, "_reproject_actual_overflow_low",
                        lambda ctx: calls.append(("books",)) or setattr(ctx, "active_context_mode", "low"))
    monkeypatch.setattr(loop, "_measure_round_main_fit", lambda _ctx, **_kw: _fit(sizes.pop(0)))
    monkeypatch.setattr(lmc, "_run_emergency_address_pass", lambda _ctx, _fit, *, rung, goal_tokens=None: (
        calls.append((rung, goal_tokens)) or SimpleNamespace(status=unseen.get("status", "no_eligible"),
                                                             reclaimed_tokens=unseen.get("shed", 0))))
    stated = _ctx("This model's maximum context length is 400,000 tokens.")  # slack 50,000
    second = lmc.REFUSAL_PASSES[1]
    sizes[:] = [880_000, 880_000, 830_000]  # after the floor; before and after the books step
    assert lmc._apply_refusal_pass(stated, second, _fit(900_000, accepted=None), 70_000) is True
    assert calls == [("floor", 20_000), ("books",)]
    assert stated.active_context_mode == "low"
    calls.clear()
    paid = _ctx("This model's maximum context length is 400,000 tokens.")
    unseen.update(status="applied", shed=50_000)
    sizes[:] = [830_000]
    assert lmc._apply_refusal_pass(paid, second, _fit(900_000, accepted=None), 70_000) is True
    assert calls == [("floor", 20_000)]  # the whole goal was paid: books and bodies stay
    assert paid.active_context_mode == "max"


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
            (second, None, {"floor": 0}, []),  # memory changed the candidate: its retry answers first
            (second, 100_000, {"floor": 30_000, "unseen_bodies": 10_000}, [("chosen_bodies", 60_000)]),
            (second, 100_000, {"floor": 100_000}, []),  # memory paid the whole measured target
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
    assert ctx.messages[0] == ctx.context_fit_plan.projection(mode).system_message()
    assert ctx.context_fit_plan.window_record is not None
    assert measure(ctx).measurement.capacity_total_tokens == (ceiling or 1_000_000)
    assert ctx.context_fit_plan.window_record.source.startswith("refusal" if ceiling else "metadata")
