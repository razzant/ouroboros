"""Owner acceptance: incident shape, both density classes, chosen reads and authored boundaries."""
import dataclasses
import json
from types import SimpleNamespace

import pytest
from ouroboros import context_fit as cf, memory_floor as mf, memory_view as mv, capability_evidence as ce
from ouroboros import usage_accounting as ua
from ouroboros.context_budget import ContextReclaimRequest, WHOLE_CATEGORY
from ouroboros import context_compaction as cc
from ouroboros.context_source_view import emergency_address_view
from ouroboros.context_renewal import renewal_seed
from tests import _memory_view_synthetic as syn
from tests.test_memory_view_lever import plan as lever_plan
from ouroboros.tools.compact_context import record_context_view, _compact_context
from ouroboros.loop_round_limits import _run_round_compaction
from tests.test_context_view_tool import _context
from tests.test_main_authored_context import main_loop as main_loop, call
from tests.test_memory_low_start import sent as sent
from tests.test_memory_max_room import _crowded_window
from tests.system_e2e.harness import MOCK_SLUG, ScriptedStubModel


@pytest.mark.parametrize('role', ['integrator', 'consciousness', 'child'])
def test_million_token_text_root_wake_and_helper_keep_notes_whole(tmp_path, monkeypatch, role):
    from tests import _memory_scale_fixture as scale
    snap = scale.snapshot()
    snap = dataclasses.replace(snap, spec=dataclasses.replace(mv.ROLE_DEFAULTS[role], room_id='1'))
    plan = scale.build(tmp_path, monkeypatch, snap=snap, window=1000000, reserve=None)
    shown = plan.projection(plan.initial_mode)
    assert not {'F8', 'F8a'}.intersection(shown.memory_facts['floor']['steps'])
    for note in snap.room['notes']:
        assert mv._note_lines([note])[0] in ''.join(part['text'] for part in json.loads(shown.system_content_json))


def test_notes_fold_when_an_owner_nano_target_is_smaller(tmp_path, monkeypatch):
    from tests import _memory_scale_fixture as scale
    snap = scale.snapshot()
    plan = scale.build(tmp_path, monkeypatch, snap=snap, window=1000000, reserve=None, mode='nano')
    shown = plan.projection(plan.initial_mode)
    assert shown.memory_facts['floor']['budget_allowance_tokens'] is not None
    assert shown.memory_facts['floor']['steps']['F8'] == len(snap.room['notes'])
    assert shown.memory_facts['floor']['by_budget'] > 0
    assert all(note['id'] in shown.system_content_json for note in snap.room['notes'])


def test_this_rooms_notes_stay_whole_for_an_owner_low_target(tmp_path, monkeypatch):
    """This room does not yield to Low's target (owner 15.11.2026, 3A): the owner's Low target leaves this room's
    notes whole, led or not, and records that it is missed; only the window may shorten them. Nano's number still
    folds them (the test above)."""
    from tests import _memory_scale_fixture as scale
    for led in (False, True):
        snap = scale.snapshot()
        for note in snap.room['notes'] if led else ():
            note['lead_rule'] = 1
        plan = scale.build(tmp_path, monkeypatch, snap=snap, window=1000000, reserve=None, mode='low')
        shown = plan.projection(plan.initial_mode)
        floor = shown.memory_facts['floor']
        assert floor['budget_allowance_tokens'] is not None and floor['target_missed']
        assert 'F8' not in floor['steps'] and 'F8a' not in floor['steps']
        text = "\n".join(block["text"] for block in json.loads(shown.system_content_json))
        assert all(note['text'] in text for note in snap.room['notes'])


def _core(mature, root):
    """A Body Map of 600,000 characters with a short parts form and a reader of its full text (``root``): Max carries
    the Map whole, the lowered frame its parts (before 26.11.2026: the whole books beside their chapter maps)."""
    spoken = [syn.spoken(i, "ouroboros", 4_000) for i in range(24)]
    snap = syn.snapshot(role="consciousness", story=syn.pages(103 if mature else 2, 200),
        room_facts=syn.room(lane1=spoken, lane2=359 if mature else 2),
        conversations=[syn.conversation(i, engaged=False) for i in range(164 if mature else 2)])
    snap = dataclasses.replace(snap, spec=dataclasses.replace(snap.spec, work_room_levels=1))
    core = cf.ContextCore("Required core", "Constitution", "b" * 600_000, "",
        "Identity", "f" * 1_000, '"Assignment"', body_map_parts="## Body map (parts and addresses)\n- Overview: `x.md`",
        memory_view_json=mv.snapshot_json(snap), section_root=str(root), section_task_id="accept01")
    return core, snap


def _start(core, window, monkeypatch, density, kind="total"):
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: density)
    return cf.build_context_fit_plan(None, core, {"id": "accept01", "usage_category": "consciousness"},
        preferred_mode="max", route_resolver=lambda *_a, **_kw: (
            {"provider": "test", "model": "test"}, ce.CapabilityEvidence(
                window, "confirmed", "metadata", "accept-route", window_kind=kind)))


@pytest.mark.parametrize(("kind", "mature", "density", "window"), [
    ("input", False, 1.0, 128_000),
    ("input", False, 1.0, 272_000),
    ("input", False, 1.0, 922_000),
    pytest.param("total", True, 4 / 2.7, 200_000, id='total-True-1.4814814814814814-200000'),
    pytest.param("total", True, 4 / 2.7, 400_000, id='total-True-1.4814814814814814-400000'),
    pytest.param("total", True, 4 / 2.7, 1_000_000, id='total-True-1.4814814814814814-1000000'),
])
def test_fresh_and_mature_envelopes_leave_one_work_room(window, density, mature, kind, tmp_path, monkeypatch):
    core, snap = _core(mature, tmp_path)
    plan = _start(core, window, monkeypatch, density, kind)
    shown = plan.projection(plan.initial_mode)
    reply = plan.output_reserve_tokens if kind == "total" else 0
    assert shown.calibrated_tokens + reply + -(-window // 8) <= window
    assert plan.core.memory_view_json == core.memory_view_json
    facts = shown.memory_facts["floor"]
    assert facts["target_tokens"] is None
    if plan.initial_mode == "max":
        assert core.body_map in shown.system_content_json
        assert "Low with chapter maps" not in shown.system_content_json and shown.memory_facts["max_room"]["cause"] == "held"
    else:
        assert core.body_map not in shown.system_content_json and "- Overview: `x.md`" in shown.system_content_json
        cause = shown.memory_facts["max_room"]["cause"]  # the route's disclosed fact, or memory/schemas took the room
        assert cause in ("route", "crowded") and ("a disclosed fact of this route" if cause == "route" else
                                                  "The host's measure of this start") in shown.system_content_json
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


@pytest.mark.parametrize(("transmitted", "size", "refused"), [
    (0, 100_000, False),
    (1_234, 140_000, True),
    (65_536, 100_000, False),
])
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


def test_core_without_memory_reserves_one_eighth_for_roots_and_direct_turns(monkeypatch):
    core = cf.ContextCore("c" * 160_000, "", "", "", "", "", '"Assignment"')
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    def start(task):
        return cf.build_context_fit_plan(None, core, {"id": "core001", **task}, preferred_mode="low",
            route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "test"},
                ce.CapabilityEvidence(128_000, "confirmed", "metadata", "core-route")))
    assert start({"type": "task", "_is_direct_chat": True}).initial_mode == "low"  # a direct chat turn
    for root in ({"type": "task"}, {"type": "evolution"}):
        plan = start(root)
        assert plan.initial_mode == "low"
        assert plan.projection("low").calibrated_tokens + plan.output_reserve_tokens + 128_000 // 8 <= 128_000


def _installation(tmp_path, monkeypatch):
    """A small installation with the repository's own composed books (``tests/_live_world.py``), as its data root."""
    from ouroboros import config
    from tests._live_world import _own_installation

    env, memory, _rooms = _own_installation(tmp_path, books=True)
    monkeypatch.setattr(config, "DATA_DIR", env.drive_root)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(env.drive_root))
    return env, memory


def _replies(core, count):
    """``core`` with the start's only synthetic part: one room of ``count`` short replies without a gist, in the real
    start's own work room (a root task keeps two eighths: ``memory_view.start_room``)."""
    levels = mv.snapshot_from_json(core.memory_view_json).spec.work_room_levels
    snap = syn.snapshot(role="integrator", story=syn.pages(2, 200),
        room_facts=syn.room(lane1=[syn.spoken(i, "ouroboros", 400) for i in range(count)], lane2=2))
    snap = dataclasses.replace(snap, spec=dataclasses.replace(snap.spec, work_room_levels=levels))
    return dataclasses.replace(core, memory_view_json=mv.snapshot_json(snap))


def _owner_start(env, core, window, schemas):
    return cf.build_context_fit_plan(env, core, {"id": "authored-main", "type": "task"}, preferred_mode="max",
        tool_schemas=schemas, route_resolver=lambda *_a, **_kw: ({"provider": "openai", "model": "openai/test-model"},
            ce.CapabilityEvidence(window, "confirmed", "metadata", "start-hosts")))


def _tipping_window(held):
    """The largest window whose Max frame cannot hold even with no memory beside the tool schemas, while it holds
    the Body Map, core, reply reserve, work room and this memory without them: the schemas alone take the Max room.
    The lowered frame (the Map at its parts) then has what the parts form saves for the request and memory the window
    keeps. Before 26.11.2026 half the schemas tipped it, when whole books stood beside chapter maps; the Map's parts
    save far less than a book did, so the tipping window moved next to the request's own boundary."""
    window = held["zero_need"] - held["work"]  # the request with no memory, without its work room
    while held["zero_need"] - held["work"] + held["levels"] * -(-window // 8) <= window:
        window -= 1
    while held["zero_need"] - held["work"] + held["levels"] * -(-(window + 1) // 8) > window + 1:
        window += 1
    return window


def _map_band(probe, window, cause, *, target=1_500):
    """``(window, gap, plan)`` of a start in the band the resident Body Map opens (BIBLE P1: the Map steps down to its
    parts and addresses before any of my memory goes below a telling): Max misses by ``gap`` (the gate's need at my
    tellings less the window, on the probe's own plan), more than the loop's own host rows (about 630 tokens) and less
    than the Map's step saves (about 2,800), so the lowered start holds my tellings and its first real send keeps the
    reply reserve and the work room. ``None`` when no window gives ``cause`` inside that band."""
    import math as _math

    for _ in range(8):
        plan = probe(window)
        fact = plan.projection(plan.initial_mode).memory_facts["max_room"]
        need = fact.get("telling_need", fact["need"])
        gap = need - fact["window"]
        if plan.start == ("low", cause) and 700 < gap < 2_700:
            return window, gap, plan
        if cause == "request" and need - fact["zero_need"] >= 2_700:
            return None  # no memory beside a request-tipped start fits the band: memory alone exceeds the Map's step
        window = _math.ceil((need - fact["work"] - target) / (1 - fact["levels"] / 8))
    return None


class _FirstSendGap(AssertionError):
    """The first real send of a start the window lowered outside the Map's band keeps less than its reply reserve and
    work room: the start fit does not price the loop's own first-send host rows (roster snapshot, context facts)."""


FIRST_SEND_GAP = ("razzant/ouroboros#4351: a request-tipped Low start's first real send overruns the reply "
                  "reserve and the work room by about 630 tokens, because the start fit does not price the loop's own "
                  "first-send host rows (roster snapshot, context facts); strict, so the fix turns this case red")


@pytest.mark.parametrize("replies,window_of,start", [
    (24, "crowded", ("max", "")), (400, "crowded", ("max", "")), (2_400, "crowded", ("low", "memory")),
    pytest.param(24, "tipping", ("low", "request"),
                 marks=pytest.mark.xfail(strict=True, raises=_FirstSendGap, reason=FIRST_SEND_GAP))])
def test_owner_max_or_its_recorded_reason_reaches_what_the_person_sees(main_loop, sent, tmp_path, monkeypatch,
                                                                       replies, window_of, start):
    """Audit item 23(a): a start succeeds when it holds the selected mode or records the reason the person then
    reads. The repository's own books, core and tool catalog, the Main loop to its first send; memory is the only
    synthetic part. Each window comes from a held start's own measure, so book growth moves the window, never the
    verdict: no fixed window the Body Map must fit. A reason is proven by its twin, never by the code that names it:
    memory crowds a start that a small memory holds at the same window; tool schemas tip a start that holds without
    them. The person reads it on the Context capsule (the request's recorded context), on the task card's start
    row and, for memory, in one Main line. A lowered start's window sits in the band the resident Body Map opens
    (``_map_band``: Max misses by more than the loop's own host rows and less than the Map's step saves), the band the
    constitution names (the Map steps down to its parts before any of my memory goes below a telling); a request-tipped
    start cannot reach it with this memory, so it pins the truthful facts and what the start itself priced."""
    from ouroboros.context import _capture_context_core

    f = main_loop
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    env, memory = _installation(tmp_path / "world", monkeypatch)
    real = _capture_context_core(env, memory, {"id": "authored-main", "type": "task", "text": "Assignment"}, None, None)
    schemas, core = f.registry.schemas(), _replies(real, replies)
    if window_of == "crowded":  # half of a 2,400-reply memory's ordinary forms fit beside the Body Map
        held = _owner_start(env, _replies(real, 2_400), 2_000_000, schemas)
        band = _map_band(lambda w: _owner_start(env, _replies(real, 2_400), w, schemas),
                         _crowded_window(held.projection("max").memory_facts["max_room"]), "memory")
        assert band, "no window puts the 2,400-reply start in the Body Map's band"
        window = band[0]
        twin = _owner_start(env, _replies(real, 24), window, schemas)
    else:
        held = _owner_start(env, core, 2_000_000, schemas)
        window = _tipping_window(held.projection("max").memory_facts["max_room"])
        for _ in range(6):  # host sections take their shorter forms in a smaller window: walk to its own boundary
            probe = _owner_start(env, core, window, schemas)
            if probe.start == ("low", "request"):
                break
            window = min(window - 1, _tipping_window(probe.projection(probe.initial_mode).memory_facts["max_room"]))
        # No request-tipped window lies in the Map's band: with no memory the request already misses Max, so the gate
        # misses by at least my 24 replies' tellings (about 3,360 tokens), more than the Map's step saves (about 2,700).
        assert _map_band(lambda w: _owner_start(env, core, w, schemas), window, "request") is None
        band = None
        twin = _owner_start(env, core, window, [])
    plan = _owner_start(env, core, window, schemas)
    assert held.start == ("max", "") and plan.preferred_mode == "max" and plan.start == start
    assert twin.start == ("max", "")  # what the reason names is all that lowers this start
    f.ctx.context_fit_plan, f.messages = plan, plan.messages_for(plan.initial_mode)
    ua.adopt_physical_attempt_capture(None)  # a fresh task: the fixture's attempt ids repeat from test to test
    assert f.run([{"content": "done"}])[0] == "done"
    first = f.inputs[0]
    text = "".join(part["text"] for row in first["messages"] if row["role"] == "system" for part in row["content"])
    assert real.body_map_parts and (core.body_map in text) is (start[0] == "max")  # the whole Map iff Max
    tokens = cf.estimate_context_prompt_tokens(first["messages"], first["tools"], provider="openai")
    room = held.projection("max").memory_facts["max_room"]["levels"] * -(-window // 8)
    fits = tokens + plan.output_reserve_tokens + room <= window  # the reply reserve and the start's work room stay
    if start[0] == "max" or band:  # held, or lowered inside the Map's band: memory at my tellings, the Map at its parts
        assert fits
    else:  # the known gap (FIRST_SEND_GAP) is asserted last; what the start itself priced and the rows it misses hold
        planned = cf.estimate_context_prompt_tokens(plan.messages_for(plan.initial_mode), first["tools"], provider="openai")
        assert planned + plan.output_reserve_tokens + room <= window  # what the start itself priced keeps them
        assert {row.get("_host_context_kind") for row in first["messages"][len(plan.messages_for(plan.initial_mode)):]} == {
            "ouroboros_roster_snapshot", "ouroboros_context_facts"}  # all the send adds are the loop's own rows
    if start[0] == "low":
        shown = plan.projection("low").memory_facts
        assert "body_map" in {row["section_id"] for row in shown.get("host_sections_addressed", ())}  # Map at its parts
        if band:  # and none of my memory below a telling
            from ouroboros.memory_forms import stage_of
            snapshot = mv.snapshot_from_json(core.memory_view_json)
            level = mf.kept_level(snapshot, shown)
            assert all(stage_of(snapshot, step, ident) in mf.TELLING_STAGES for step, ids in level.addressed for ident in ids)
    if start[0] == "max":  # the Max measure prices what is sent: an over-count would move every derived window unseen
        fact = plan.projection("max").memory_facts["max_room"]
        assert fact["need"] - fact["reserve"] - fact["work"] <= tokens + -(-window // 100)
    events = [f.events.get_nowait() for _ in range(f.events.qsize())]
    sends = [event["context_attempt"] for event in events if "context_attempt" in event]  # the census the capsule reads
    # A Low the window took from Max is named Low beside the owner's Max (21.11.2026)
    assert sends and {(send["preferred_at_send"], send["applied"], send["books"], send["cause"]) for send in sends} == {
        ("max", start[0], "body_map", start[1])}
    if not fits:  # every other fact of this case held above; only the first send's room is short
        raise _FirstSendGap(f"{tokens + plan.output_reserve_tokens + room - window} tokens past the window")
    rows = [event["data"] for event in events if event["type"] == "log_event"
            and event["data"].get("checkpoint_kind") == "context_fit_physical_mode"]
    assert [row["cause"] for row in rows] == ([start[1]] if start[1] else [])  # the task card's start row
    assert [(row["effective_mode"], row["books"]) for row in rows] == ([("low", "body_map")] if start[1] else [])
    assert [event["system_type"] for _root, event in sent] == (["low_start_notice"] if start[1] else [])  # every cause


@pytest.mark.parametrize("window", [40_000, 200_000, 1_000_000])
def test_an_environment_only_install_starts_in_its_factory_max_or_records_why(tmp_path, monkeypatch, window):
    """Audit item 23(g): no settings document, the provider, its key and the model from the environment alone (a
    Docker unit's shape), the production start with its own route probe against a free model stub. The selected
    mode is the factory Max; the outcome is that mode or a recorded reason. The probe must have read the stub's
    window: a failed route keeps Max, so an unread window would pass as a held start. At 1M the Body Map leaves
    Max ample room; at 200k the Map fits beside the core and what the request itself carries (the factory tool
    schemas) tips it (``request``; whole books could not fit there before the resident Map: ``route``); 40k is not
    pinned beyond a recorded reason or the named refusal, so book growth never turns it red."""
    from ouroboros import config, context
    from ouroboros.agent_dispatch import _context_capacity_terminal
    from ouroboros.provider_models import ALL_PROVIDER_CREDENTIAL_KEYS
    from ouroboros.tools.tool_context import ToolContext
    from ouroboros.tools.registry import ToolRegistry

    env, memory = _installation(tmp_path / "world", monkeypatch)
    monkeypatch.setattr(config, "SETTINGS_PATH", env.drive_root / "absent" / "settings.json")
    for key in (*ALL_PROVIDER_CREDENTIAL_KEYS, "OUROBOROS_CONTEXT_MODE", "OUROBOROS_MODEL_CONTEXT_WINDOWS"):
        monkeypatch.delenv(key, raising=False)
    task, ctx = {"id": "env-only", "type": "task", "text": "Say hello."}, ToolContext(env.repo_dir, env.drive_root)
    schemas = ToolRegistry(env.repo_dir, env.drive_root).schemas()
    with ScriptedStubModel() as stub:
        stub.window_tokens = window
        for key, value in (("OPENAI_COMPATIBLE_BASE_URL", stub.base_url), ("OUROBOROS_MODEL", MOCK_SLUG),
                           ("OPENAI_COMPATIBLE_API_KEY", "stub-key-not-a-credential"),
                           ("OUROBOROS_MODEL_LIGHT", MOCK_SLUG)):
            monkeypatch.setenv(key, value)
        try:
            context.build_llm_messages(env, memory, task, ctx=ctx, tool_schemas=schemas)
        except cf.MinimumViewUnfit as refusal:  # the named refusal the work's result carries
            assert window == 40_000 and _context_capacity_terminal(refusal)[1]["reason_code"] == "context_capacity"
            return
    plan = ctx.context_fit_plan
    assert plan.preferred_mode == config.get_context_mode() == "max" and plan.provider == "openai-compatible"
    assert plan.status == "confirmed" and plan.window_record.boundary == window  # the stub's own window was read
    mode, cause = plan.start
    assert (mode == "max") is (not cause)  # never a lower start without its recorded reason
    if window == 1_000_000:
        assert plan.start == ("max", "")  # ample room beside the Body Map: the selected mode holds
    elif window == 200_000:
        room = plan.projection(mode).memory_facts["max_room"]
        assert mode == "low" and cause == "request"  # the factory tool schemas tip it; the Map and core fit
        assert room["floor_need"] <= room["window"] < room["zero_need"]


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
