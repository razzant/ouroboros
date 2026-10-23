"""Owner acceptance: incident shape, both density classes, chosen reads and authored boundaries."""
import dataclasses
import json
from types import SimpleNamespace

import pytest
from ouroboros import context_fit as cf, memory_floor as mf, memory_view as mv, capability_evidence as ce
from ouroboros.context_budget import ContextReclaimRequest, WHOLE_CATEGORY
from ouroboros import context_compaction as cc
from ouroboros.context_compaction import emergency_address_view
from ouroboros.context_renewal import renewal_seed
from tests import _memory_view_synthetic as syn
from tests.test_memory_view_lever import plan as lever_plan
from ouroboros.tools.compact_context import record_context_view, _compact_context
from ouroboros.loop_round_limits import _run_round_compaction
from tests.test_context_view_tool import _context
from tests.test_main_authored_context import main_loop as main_loop, call


@pytest.mark.parametrize('role', ['integrator', 'consciousness', 'child'])
def test_million_token_text_root_wake_and_helper_keep_notes_whole(tmp_path, monkeypatch, role):
    from tests import _memory_scale_fixture as scale
    snap = scale.snapshot()
    snap = dataclasses.replace(snap, spec=dataclasses.replace(mv.ROLE_DEFAULTS[role], room_id='1'))
    plan = scale.build(tmp_path, monkeypatch, snap=snap, window=1000000, reserve=None)
    shown = plan.projection(plan.initial_mode)
    assert not mf.WINDOW_ONLY.intersection(shown.memory_facts['floor']['steps'])
    for note in snap.room['notes']:
        assert mv._note_lines([note])[0] in ''.join(part['text'] for part in json.loads(shown.system_content_json))


def test_notes_remain_whole_even_when_an_owner_nano_target_is_smaller(tmp_path, monkeypatch):
    from tests import _memory_scale_fixture as scale
    snap = scale.snapshot()
    plan = scale.build(tmp_path, monkeypatch, snap=snap, window=1000000, reserve=None, mode='nano')
    shown = plan.projection(plan.initial_mode)
    assert shown.memory_facts['floor']['budget_allowance_tokens'] is not None
    assert not mf.WINDOW_ONLY.intersection(shown.memory_facts['floor']['steps'])
    assert all(note['text'] in shown.system_content_json for note in snap.room['notes'])


def _core(mature):
    spoken = [syn.spoken(i, "ouroboros", 4_000) for i in range(24)]
    snap = syn.snapshot(role="consciousness", story=syn.pages(103 if mature else 2, 200),
        room_facts=syn.room(lane1=spoken, lane2=359 if mature else 2),
        live=[{**syn.live_room(i), "directory": True, "words": [], "notes": []} for i in range(164 if mature else 2)])
    snap = dataclasses.replace(snap, spec=dataclasses.replace(snap.spec, work_room_levels=2))
    core = cf.ContextCore("Required core", "Constitution", "b" * 600_000, "",
        "Identity", "f" * 1_000, '"Assignment"', False, memory_view_json=mv.snapshot_json(snap))
    return core, snap


def _start(core, window, monkeypatch, density, kind="total"):
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: density)
    return cf.build_context_fit_plan(None, core, {"id": "accept01", "usage_category": "consciousness"},
        preferred_mode="max", route_resolver=lambda *_a, **_kw: (
            {"provider": "test", "model": "test"}, ce.CapabilityEvidence(
                window, "confirmed", "metadata", "accept-route", window_kind=kind)))


@pytest.mark.parametrize("window", [128_000, 200_000, 272_000, 400_000, 922_000, 1_000_000])
@pytest.mark.parametrize("density", [1.0, 4 / 2.7])
@pytest.mark.parametrize("mature", [False, True])
@pytest.mark.parametrize("kind", ["input", "total"])
def test_fresh_and_mature_envelopes_leave_one_work_room(window, density, mature, kind, monkeypatch):
    core, snap = _core(mature)
    plan = _start(core, window, monkeypatch, density, kind)
    shown = plan.projection(plan.initial_mode)
    reply = plan.output_reserve_tokens if kind == "total" else 0
    assert shown.calibrated_tokens + reply + 2 * -(-window // 8) <= window
    assert plan.core.memory_view_json == core.memory_view_json
    facts = shown.memory_facts["floor"]
    assert facts["target_tokens"] is None
    if plan.initial_mode == "max":
        assert core.architecture_md in shown.system_content_json
        assert "Low: the books did not fit" not in shown.system_content_json
    else:
        assert core.architecture_md not in shown.system_content_json
        assert "Low: the books did not fit beside the story and the work room" in shown.system_content_json
    other = _start(core, 1_000_000 if plan.initial_mode == "low" else 128_000, monkeypatch, density, kind)
    assert other.initial_mode != plan.initial_mode


@pytest.mark.parametrize("usable", [50_000, None, 1_000])
def test_only_the_late_chosen_rung_addresses_a_chosen_body_that_fits_alone(tmp_path, usable):
    """Owner 2A: host copies, bodies and unasked unseen bodies never take a chosen body that fits alone; only the
    ``chosen_bodies`` rung does (``loop_model_call._apply_refusal_pass`` runs it last, for a provider's refusal).
    A body that alone exceeds usable input leaves with the unasked ones, under its named receipt."""
    body, other = "Chosen exact source. " * 2_000, "Unasked shell output. " * 2_000
    fits = usable != 1_000
    messages = [{"role": "system", "content": "Core"}, {"role": "user", "content": "Assignment"},
        {"role": "assistant", "tool_calls": [{"id": "read", "type": "function", "function": {
            "name": "memory_read", "arguments": '{"from":"row-a","to":"row-b"}'}}]},
        {"role": "tool", "tool_call_id": "read", "content": body},
        {"role": "assistant", "tool_calls": [{"id": "shell", "type": "function", "function": {
            "name": "run_shell", "arguments": '{"cmd":"ls"}'}}]},
        {"role": "tool", "tool_call_id": "shell", "content": other}]
    request = ContextReclaimRequest(route_fp="refused-route", round_id="2",
        transcript_sha256=cc.context_reclaim_transcript_sha256(messages), measurement_basis="fresh_route_usage",
        measurement_density=1.0, reclaim_goal_tokens=WHOLE_CATEGORY, allow_partial_shrink=True)

    def view(rung):
        return emergency_address_view(messages, request, rung=rung, observation={}, drive_root=tmp_path,
                                      task_id="refusal01", usable_input=usable)
    for rung in ("host_copies", "bodies", "unseen_bodies"):
        rebuilt, _receipt = view(rung)
        shown = json.dumps(rebuilt)
        assert (body in shown) is (fits or rung != "unseen_bodies")
        assert (other in shown) is (rung != "unseen_bodies")
        assert ("chosen_body_exceeds_usable_input" in shown) is (not fits and rung == "unseen_bodies")
    if fits:
        seed, _left = renewal_seed(view("unseen_bodies")[0])
        assert messages[2:4] == [row for row in seed if row in messages[2:4]]
    rebuilt, receipt = view("chosen_bodies")
    shown = json.dumps(rebuilt)
    assert receipt.status == ("applied" if fits else "no_eligible")
    assert (body in shown) is (not fits) and other in shown
    assert ("just-chosen body after a real refusal" in shown) is fits


@pytest.mark.parametrize("size,refused", [(100_000, False), (140_000, True)])
@pytest.mark.parametrize("transmitted", [0, 1_234, 65_536])
def test_final_send_prices_the_reserved_reply_even_when_the_wire_omits_it(size, refused, transmitted):
    from ouroboros.llm_attempt import main_context_fit_refusal
    from ouroboros.usage_accounting import AttemptRequest, PhysicalAttemptContext

    physical = PhysicalAttemptContext(profile="owner_low", rendered_mode="low", measurement_basis="cold_estimate",
        route_fp="bound", round_id="r1", target_total_tokens=250_000, capacity_total_tokens=200_000,
        context_target_miss=False, automatic_pass_used=False, response_reserve_tokens=65_536)
    request = AttemptRequest(model="model", provider="claudexor", prompt_tokens_bounded_estimate=size,
        max_completion_tokens=transmitted, physical_context=physical)
    result = main_context_fit_refusal(request)
    assert (result is not None) is refused
    if result:
        assert result["response_reserve_tokens"] == 65_536


@pytest.mark.parametrize("note", ["", "The owner's earlier constraint still governs the planned implementation."])
def test_owner_memory_selection_requires_and_retains_the_minds_account(tmp_path, monkeypatch, note):
    plan, snap = lever_plan()
    ctx = SimpleNamespace(context_fit_plan=plan, active_context_mode="max")
    messages, schemas = plan.messages_for("max"), []
    record_context_view(ctx, messages, schemas)
    address = "F7:" + snap.room["lane1"][0]["address"]
    answer = _compact_context(ctx, view_address=[address], working_note=note)
    if not note:
        assert "require an authored working_note" in answer or "require" in answer
        assert not getattr(ctx, "_pending_compaction", None)
    else:
        assert "requested" in answer
        result = _run_round_compaction(messages, _context(tmp_path, ctx, schemas, monkeypatch,
            lambda m, t: {"accepted": True, "input_tokens": mf.view_tokens(json.dumps(m))}))
        assert ctx._context_view_receipt["status"] == "applied"
        assert note in json.dumps(result) and snap.room["lane1"][0]["line"] not in json.dumps(result[0])
        seed, _left = renewal_seed(result)
        assert note in json.dumps(seed)


def test_core_without_memory_still_reserves_two_eighths_for_a_root(monkeypatch):
    core = cf.ContextCore("c" * 160_000, "", "", "", "", "", '"Assignment"', False)
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    def start(task):
        return cf.build_context_fit_plan(None, core, {"id": "core001", **task}, preferred_mode="low",
            route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "test"},
                ce.CapabilityEvidence(128_000, "confirmed", "metadata", "core-route")))
    assert start({"type": "task", "_is_direct_chat": True}).initial_mode == "low"  # a direct chat turn
    for root in ({"type": "task"}, {"type": "evolution"}):
        with pytest.raises(cf.MinimumViewUnfit, match="Core, assignment, resident control tools and work room"):
            start(root)


@pytest.mark.parametrize("replies,mode", [(2_400, "max"), (400, "max"), (24, "max")])
def test_owner_max_is_judged_on_the_completed_start_view_with_the_real_catalog(main_loop, monkeypatch, replies, mode):
    f = main_loop
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    snap = syn.snapshot(role="integrator", story=syn.pages(2, 200),
        room_facts=syn.room(lane1=[syn.spoken(i, "ouroboros", 400) for i in range(replies)], lane2=2))
    snap = dataclasses.replace(snap, spec=dataclasses.replace(snap.spec, work_room_levels=2))
    core = cf.ContextCore("Required core", "Constitution", "b" * 600_000, "", "Identity", "f" * 1_000,
        '"Assignment"', False, memory_view_json=mv.snapshot_json(snap))
    plan = cf.build_context_fit_plan(None, core, {"id": "authored-main", "type": "task"}, preferred_mode="max",
        tool_schemas=f.registry.schemas(), route_resolver=lambda *_a, **_kw: (
            {"provider": "openai", "model": "openai/test-model"},
            ce.CapabilityEvidence(400_000, "confirmed", "metadata", "start-hosts")))
    f.ctx.context_fit_plan, f.messages = plan, plan.messages_for(plan.initial_mode)
    assert f.run([{"content": "done"}])[0] == "done"
    sent = cf.estimate_context_prompt_tokens(f.inputs[0]["messages"], f.inputs[0]["tools"], provider="openai")
    assert plan.initial_mode == mode
    if replies == 2400:
        assert plan.projection(mode).memory_facts['floor']['steps']['F7a'] == 1
    assert (core.architecture_md in json.dumps(f.inputs[0]["messages"])) is (mode == "max")
    assert sent + plan.output_reserve_tokens + 2 * 400_000 // 8 <= 400_000


@pytest.mark.parametrize("refused", [True, False])
def test_chosen_read_then_provider_refusal_preserves_the_body_on_real_main(main_loop, refused):
    import httpx
    from ouroboros import usage_accounting as ua
    from ouroboros.context_budget import bound_host_row
    from ouroboros.peer_roster import ROSTER_SNAPSHOT_NAME
    from ouroboros.llm_attempt import _attempt_request, _physical_candidate

    f = main_loop
    old = bound_host_row(ROSTER_SNAPSHOT_NAME, "Superseded room roster. " * 3000)
    f.messages.extend([old, bound_host_row(ROSTER_SNAPSHOT_NAME, "Current roster")])
    first = True
    def answer(kwargs):
        nonlocal first
        if refused and first:
            first = False
            candidate = _physical_candidate({key: kwargs[key] for key in ("messages", "tools", "model")})
            request = _attempt_request({"provider": "openai", "usage_model": kwargs["model"]}, candidate)
            ua.adopt_physical_attempt_capture(ua.PhysicalAttemptCapture(
                "chosen-refused", kwargs["model"], "openai", "unresolved", "canonical_json_v1",
                physical_context=request.physical_context, candidate_raw_sha256=request.candidate_raw_sha256,
                candidate_raw_size_bytes=request.candidate_raw_size_bytes,
                candidate_context_sha256=request.candidate_context_sha256,
                candidate_context_size_bytes=request.candidate_context_size_bytes))
            raise httpx.HTTPStatusError("context_length_exceeded",
                request=httpx.Request("POST", "https://fixture.invalid"), response=httpx.Response(400,
                json={"error": {"code": "context_length_exceeded", "message": "Input exceeds context window"}}))
        return {"content": "I received the chosen source whole."}
    text, _usage, _trace = f.run([call("read_file", {"path": "evidence.txt", "start_char": 0,
        "max_chars": len(f.source)}, "chosen"), answer, answer])
    assert text == "I received the chosen source whole."
    received = [next(row["content"] for row in send["messages"] if row.get("tool_call_id") == "chosen")
                for send in f.inputs[1:]]
    assert len(received) == (2 if refused else 1)
    assert all(body == received[0] and f.source.splitlines()[-1] in body for body in received)
    assert (old in f.inputs[-1]["messages"]) is (not refused)
