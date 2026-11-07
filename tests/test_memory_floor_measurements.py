"""Count complete measurements, not just the rendered size of a small fixture."""
import dataclasses
import hashlib
import json
import time
from types import SimpleNamespace

import pytest

from ouroboros import context_fit as cf, memory_floor as mf, memory_view as mv
from ouroboros.section_floor import HostSection
from tests import _memory_scale_fixture as scale, _memory_view_synthetic as syn


def test_owner_floor_measures_each_selected_form_once(monkeypatch):
    snapshot = scale.snapshot()
    calls = []

    def measure(level):
        calls.append(level.addressed)
        return mf.view_tokens(mv.render_story(snapshot, level)) + mf.view_tokens(mv.render_room(snapshot, level))

    start = time.perf_counter()
    level = mf.fit_memory_view(snapshot, {'margin': 0, 'physical': 0}, measure=measure)
    elapsed = time.perf_counter() - start
    rendered = mv.render_story(snapshot, level) + mv.render_room(snapshot, level)
    print('FLOOR_MEASUREMENTS ' + json.dumps({'rows': 158, 'calls': len(calls),
        'unique': len(set(calls)), 'seconds': elapsed, 'sha256': hashlib.sha256(rendered.encode()).hexdigest()}))
    assert len(calls) <= 60, f'owner shape: at most 60 complete measurements, got {len(calls)}'
    assert len(calls) == len(set(calls)), 'a selected form must never be re-tokenized in the same walk'
    with monkeypatch.context() as uncached:
        uncached.setattr(mf, '_memoized_measure', lambda measure: measure)
        reference = mf.fit_memory_view(snapshot, {'margin': 0, 'physical': 0}, measure=measure)
    assert level == reference
    assert rendered == mv.render_story(snapshot, reference) + mv.render_room(snapshot, reference)


def test_no_note_measurement_without_a_physical_bound():
    calls = []
    mf.fit_memory_view(scale.snapshot(), {'margin': None, 'physical': None},
                       measure=lambda level: calls.append(level.addressed) or 100)
    assert len(calls) == 1, f'unpressured view: one initial measurement, got {len(calls)}'


@pytest.mark.parametrize('rows', [158, 300, 600])
def test_ordinary_scale_fit_measurements(tmp_path, monkeypatch, rows):
    calls = 0
    original = cf._request_tokens

    def count(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cf, '_request_tokens', count)
    start = time.perf_counter()
    plan = scale.build(tmp_path, monkeypatch, snap=scale.snapshot(rows=rows))
    elapsed = time.perf_counter() - start
    projection = plan.projection(plan.initial_mode)
    print('ORDINARY_FIT_MEASUREMENTS ' + json.dumps({'rows': rows, 'calls': calls, 'seconds': elapsed,
        'mode': plan.initial_mode, 'input': projection.calibrated_tokens,
        'sha256': hashlib.sha256(projection.system_content_json.encode()).hexdigest()}))
    assert plan.initial_mode == 'nano'
    assert calls <= rows // 3 + 15, f'{rows} rows: at most rows/3 + 15 complete envelope measurements, got {calls}'


def test_one_start_lists_a_snapshot_s_floor_elements_once(tmp_path, monkeypatch):
    """Live scale (520 other conversations, ~17,000 words of people, no owner text): every walk, level, receipt and
    Max-room measure of one start reads one listing of the snapshot's elements; a changed snapshot lists its own. The
    frame walks its shortest view once and re-prices that level beside each later host form: 33 full room renders
    here, where a walk beside every host form took 60."""
    listed, probes, renders = [], [], []
    original, render, room = mf._list_elements, mf.render_view_for_mode, mv.render_room
    monkeypatch.setattr(mf, '_list_elements', lambda snapshot: listed.append(snapshot) or original(snapshot))
    monkeypatch.setattr(mv, 'render_room', lambda *args, **kwargs: renders.append(1) or room(*args, **kwargs))

    def recorded(*args, **kwargs):
        if kwargs.get('shortest') or kwargs.get('level') is not None:
            probes.append('walk' if kwargs.get('level') is None else 're-price')
        return render(*args, **kwargs)
    monkeypatch.setattr(mf, 'render_view_for_mode', recorded)
    snap = syn.live_scale()
    plan = scale.build(tmp_path, monkeypatch, snap=snap, window=600_000)
    facts = plan.projection(plan.initial_mode).memory_facts
    print('LIVE_SCALE_START ' + json.dumps({'mode': plan.initial_mode, 'steps': facts['floor']['steps'],
                                            'probes': probes, 'renders': len(renders)}))
    assert plan.initial_mode == 'max' and facts['floor']['steps'].get('F6w')  # the walk ran at this scale
    assert len(listed) == 1, f'one listing per snapshot object, got {len(listed)}'
    assert probes[0] == 'walk' and probes.count('walk') == 1, probes  # one Max frame, one walk
    assert probes.count('re-price') <= len(facts['host_sections_addressed']) and len(renders) <= 33
    fewer = dataclasses.replace(snap, conversations=snap.conversations[:1])
    assert len(mf.floor_elements(fewer)) < len(mf.floor_elements(listed[0])) and len(listed) == 2


def test_a_snapshot_s_element_table_is_never_read_under_a_different_ladder(monkeypatch):
    """The table lives on the snapshot object, which the parse memo shares between starts: one listed under a narrowed
    ``LADDER`` (a measurement's monkeypatch) is not read once the ladder differs; the next reader lists anew."""
    forms = {'F7a', 'F5a', 'F8', 'F8a'}
    snap = syn.live_scale()
    with monkeypatch.context() as narrowed:
        narrowed.setattr(mf, 'LADDER', tuple(step for step in mf.LADDER if step not in forms))
        assert not forms & {step for step, *_ in mf.floor_elements(snap)}
    assert mf.floor_elements(snap) == mf.floor_elements(dataclasses.replace(snap))  # a fresh object's own listing
    assert forms & {step for step, *_ in mf.floor_elements(snap)}


def _near_even_plan(root, monkeypatch, window, tail=61, owner=5199, pad=2, shift=1, identity=0, skills=None):
    """A small real start whose note's label sits within a token of its receipt clause, so each host step's own bytes
    can change whether the label saves: the whole request rounds differently beside every new address. ``identity``
    pads the core and ``skills`` sizes that host section (by default its form saves about 300 tokens)."""
    lane = [syn.spoken(i, 'ouroboros', 300) for i in range(2)]
    lane.append({**syn.spoken(90, 'human', owner), 'owner_side': True, 'carried': False})
    room = syn.room('1', lane1=lane)
    room['notes'] = [{'id': 'note-here-0', 'date': '2026-09-02', 'role': 'root', 'text': 'Where this stands. ' + 'x' * tail}]
    snap = syn.snapshot(room_facts=room, story=syn.pages(2, 300), other_conversations=False)
    runtime = HostSection('runtime', '## runtime\n\n' + 'o' * (1600 + pad), '## runtime\n\nOperative.' + 'q' * shift)
    skills = HostSection('installed_skills', '## skills\n\n' + 's' * (1200 + pad if skills is None else skills),
                         '## skills\n\nPurpose.')
    index = HostSection('knowledge_index', '## knowledge\n\n' + 'k' * 2400, '## knowledge\n\nOutline.')
    captured = cf.ContextCore(base_prompt='SYSTEM', bible_md='BIBLE', architecture_md='# Architecture', development_md='',
        docs_need_development=False, semi_stable_text='IDENTITY' + 'i' * identity, dynamic_text='\n\n'.join((runtime, skills)),
        dynamic_head_text=index, dynamic_sections=(runtime, skills), dynamic_head_sections=(index,),
        user_content_json=json.dumps('Continue.'), section_root=str(root), section_task_id='task-1',
        memory_view_json=mv.snapshot_json(snap))
    monkeypatch.setattr(cf, '_route_calibration_ratio', lambda *a, **kw: 1.0)
    monkeypatch.setattr(cf, 'main_output_reserve_tokens', lambda **kw: 2000)
    evidence = SimpleNamespace(route_fp='synthetic-route', status='confirmed', stale=False, window_tokens=window, source='synthetic')
    return cf.build_context_fit_plan(SimpleNamespace(drive_root=root), captured, {'id': 'task-1', '_is_direct_chat': True},
        preferred_mode='low', route_resolver=lambda *a, **kw: ({'provider': 'test', 'model': 'synthetic'}, evidence))


def _hosts(plan):
    return [(row['section_id'], row.get('start_room', False))
            for row in plan.projection(plan.initial_mode).memory_facts.get('host_sections_addressed', ())]


@pytest.mark.parametrize('window,shape,hosts,steps', [
    # Beside the second host form the label starts to save and the request fits: the index stays whole. Forms chosen
    # beside the first host form would price it a token over and outline the index for no physical need.
    (4745, {}, [('runtime', False), ('installed_skills', False)], {'F2': 2, 'F5': 2, 'F8': 1}),
    # The window holds the request after the first step, and the next one pays only the start's room.
    (4982, {}, [('runtime', False), ('installed_skills', True)], {'F2': 2, 'F5': 2}),
    # Three characters fewer, and beside the middle host form the whole request still rounds to the same price.
    (4082, {'tail': 63, 'owner': 1600, 'pad': 0, 'shift': 0},
     [('runtime', False), ('installed_skills', True)], {'F2': 2, 'F5': 2, 'F8': 1}),
])
def test_each_host_address_walks_its_own_shortest_forms(tmp_path, monkeypatch, window, shape, hosts, steps):
    """A real start, near the label's break-even: which forms are shortest changes between one frame's host steps. The
    frame walks once, then re-prices that level beside each new host form: a re-price that misses the window walks
    again, beside its own host forms, and chooses other forms; one that fits (the index stays whole) walks no more. A tie
    within a token takes the whole request's two measures, and a walk that measures every optional form whole chooses
    exactly the same at every probe."""
    render, probes, ties = mf.render_view_for_mode, [], []

    def recorded(*args, **kwargs):
        result = render(*args, **kwargs)
        if kwargs.get('shortest') or kwargs.get('level') is not None:
            probes.append(('walk' if kwargs.get('level') is None else 're-price', kwargs['mode'],
                           mf.kept_level(args[0], result[2]).addressed, result[1]))
        return result

    prices = mf._shortest_prices
    monkeypatch.setattr(mf, 'render_view_for_mode', recorded)
    monkeypatch.setattr(mf, '_shortest_prices', lambda sent, measure: prices(
        sent, lambda level: ties.append(level.addressed) or measure(level)))
    settled = _near_even_plan(tmp_path / 'settled', monkeypatch, window, **shape)
    assert _hosts(settled) == hosts and settled.projection(settled.initial_mode).memory_facts['floor']['steps'] == steps
    walks = {str(forms) for kind, _mode, forms, _room in probes if kind == 'walk'}
    assert [kind for kind, *_ in probes] == ['walk', 're-price', 'walk', 're-price']  # the last re-price fit: no walk
    assert ties and len(walks) == 2  # a tie was measured; the walk after the missed re-price chose other forms
    chosen, probes[:] = list(probes), []
    monkeypatch.setattr(mf, '_shortest_prices', lambda sent, measure: (
        measure, lambda taken, trial: (measure(taken), measure(trial))))  # the control: every optional form measured whole
    measured = _near_even_plan(tmp_path / 'measured', monkeypatch, window, **shape)
    assert probes == chosen and measured.initial_mode == settled.initial_mode
    for mode in mf.MODES:
        assert measured.projection(mode).system_content_json == settled.projection(mode).system_content_json
        assert measured.projection(mode).memory_facts == settled.projection(mode).memory_facts


@pytest.mark.parametrize('window,probes,hosts', [
    # The skills step saves 3 tokens, and the margin it frees reaches the size at which the uncovered owner row alone
    # would fit: the receipt's clause turns to the owner words, 12 tokens longer, and the level that fitted costs more.
    (11_416, ['walk', 're-price', 're-price', 'walk'],
     [('runtime', False), ('installed_skills', True), ('knowledge_index', False)]),
    # Three tokens less window and the clause stays short: the second re-price fits and the index stays whole.
    (11_413, ['walk', 're-price', 're-price'], [('runtime', False), ('installed_skills', True)]),
])
def test_a_host_step_that_grows_the_receipt_walks_the_shortest_view_again(tmp_path, monkeypatch, window, probes, hosts):
    """Astra's counterexample (R2 11(i)) on a real start: a fit proven beside one host form is proven again beside the
    next. After the runtime step the walk's level fits under its receipt; when the next step makes that receipt longer
    the re-price misses, the frame walks again and the index outlines for the physical window. Kept as proven, the fit
    would leave the index whole beside a request the window cannot hold."""
    render, seen = mf.render_view_for_mode, []

    def recorded(*args, **kwargs):
        if kwargs.get('shortest') or kwargs.get('level') is not None:
            seen.append('walk' if kwargs.get('level') is None else 're-price')
        return render(*args, **kwargs)
    monkeypatch.setattr(mf, 'render_view_for_mode', recorded)
    plan = _near_even_plan(tmp_path, monkeypatch, window, identity=26_680, skills=290)
    assert seen == probes and _hosts(plan) == hosts


def test_a_shortest_walk_measures_no_form_its_text_settles(tmp_path, monkeypatch):
    """Owner shape at 92K: Max and Low cannot hold even their least request, so neither walks; Nano walks once, then at
    each of its next three host forms re-prices that level, misses the window (even its shortest view does not fit) and
    walks again. Those walks' forms all differ by far more than a token, so none takes a whole measurement."""
    render, probes, ties = mf.render_view_for_mode, [], []

    def recorded(*args, **kwargs):
        if kwargs.get('shortest') or kwargs.get('level') is not None:
            probes.append(('walk' if kwargs.get('level') is None else 're-price', kwargs['mode']))
        return render(*args, **kwargs)

    prices = mf._shortest_prices
    monkeypatch.setattr(mf, 'render_view_for_mode', recorded)
    monkeypatch.setattr(mf, '_shortest_prices', lambda sent, measure: prices(
        sent, lambda level: ties.append(level.addressed) or measure(level)))
    assert scale.build(tmp_path, monkeypatch).initial_mode == 'nano'
    assert probes == [('walk', 'nano')] + [('re-price', 'nano'), ('walk', 'nano')] * 3 and ties == []


@pytest.mark.parametrize("step, target, outlined", [
    ("floor", 12_500, []), ("floor", 25_000, ["knowledge_index"]),  # slack alone / a measured excess
    ("floor", None, []), ("memory_rest", None, ["knowledge_index"]),  # unknown: first pass / after a further refusal
])
def test_the_slack_alone_never_outlines_the_knowledge_index(tmp_path, monkeypatch, step, target, outlined):
    """BIBLE P1, both directions, on the real floor (review 06.11): when the measured excess is paid and only the
    return-after-fold slack remains, memory and ordinary host forms pay it and the index stays whole; a measured
    excess the host forms cannot pay outlines it. Against an unknown target the first pass outlines nothing; memory's
    rest after a further refusal may."""
    from ouroboros import loop, loop_model_call as lmc

    index = HostSection("knowledge_index", "## knowledge\n\n" + "k" * 10000, "## knowledge\n\nOutline.")
    core = cf.ContextCore(base_prompt="SYSTEM", bible_md="BIBLE",
        architecture_md="# Architecture", development_md="", docs_need_development=False,
        semi_stable_text="IDENTITY", dynamic_text="", dynamic_head_text=index,
        dynamic_sections=(), dynamic_head_sections=(index,),
        user_content_json=json.dumps("Continue."), section_root=str(tmp_path), section_task_id="slack-probe")
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *a, **kw: 1.0)
    monkeypatch.setattr(cf, "main_output_reserve_tokens", lambda **kw: 2000)
    evidence = SimpleNamespace(route_fp="probe", status="confirmed", stale=False, window_tokens=100000, source="synthetic")
    plan = cf.build_context_fit_plan(SimpleNamespace(drive_root=tmp_path), core,
        {"id": "slack-probe", "_is_direct_chat": True}, preferred_mode="max",
        route_resolver=lambda *a, **kw: ({"provider": "test", "model": "synthetic"}, evidence))
    initial = plan.projection(plan.initial_mode)
    assert not initial.memory_facts.get("host_sections_addressed")
    fit = SimpleNamespace(measurement=SimpleNamespace(capacity_total_tokens=100000, measurement_density=1.0,
        reprojection_input_tokens=None, estimated_input_tokens=initial.calibrated_tokens,
        capacity_deficit_tokens=0, route_fp="probe"))
    ctx = SimpleNamespace(context_fit_plan=plan, active_context_mode=plan.initial_mode,
        tools=SimpleNamespace(_ctx=SimpleNamespace()), tool_schemas=[],
        messages=[initial.system_message(), {"role": "user", "content": "Continue."}],
        task_id="slack-probe", event_queue=None, drive_logs=tmp_path, round_idx=1)
    monkeypatch.setattr(loop, "_measure_round_main_fit", lambda *a, **kw: fit)
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *a, **kw: None)
    lmc._apply_refusal_step(ctx, step, fit, target, 12_500)  # a known target holds the 12,500 slack
    hosts = ctx.context_fit_plan.projection(ctx.active_context_mode).memory_facts.get("host_sections_addressed", [])
    assert [row["section_id"] for row in hosts] == outlined
