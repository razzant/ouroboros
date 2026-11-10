"""Count complete measurements, not just the rendered size of a small fixture."""
import dataclasses
import json
import random
from types import SimpleNamespace

import pytest

from ouroboros import context_fit as cf, memory_floor as mf, memory_view as mv
from ouroboros import memory_view_conversations as conversations
from ouroboros.section_floor import HostSection
from ouroboros.context_timing import measured_sections, section_spans
from tests import _memory_scale_fixture as scale, _memory_view_synthetic as syn


def test_a_floor_walk_prices_from_pieces_and_chooses_as_a_drawn_view_would():
    """Owner shape under a zero bound: the walk prices every level from the snapshot's pieces, so no form of the view is
    drawn whole (a piece's text is built to be measured), no piece is made twice, and the level is exactly the one a
    walk that draws each form chooses."""
    snapshot = scale.snapshot()

    def drawn(level):
        return mf.view_tokens(mv.render_story(snapshot, level)) + mf.view_tokens(mv.render_room(snapshot, level))

    with measured_sections():
        level = mf.fit_memory_view(snapshot, {'margin': 0, 'physical': 0})
        work = section_spans()['work']
    assert not work.get('view_renders') and not work.get('view_pieces_again')
    assert level == mf.fit_memory_view(snapshot, {'margin': 0, 'physical': 0}, measure=drawn)


def test_no_note_measurement_without_a_physical_bound():
    calls = []
    mf.fit_memory_view(scale.snapshot(), {'margin': None, 'physical': None},
                       measure=lambda level: calls.append(level.addressed) or 100)
    assert len(set(calls)) == 1, f'unpressured view: one form measured, got {set(calls)}'


def test_a_start_s_request_is_serialized_once_per_form_at_every_row_count(tmp_path, monkeypatch):
    """Serializations equal the request's forms and do not follow the rows: a form is a frame's shape, never a probe."""
    seen = {}
    for rows in (158, 300, 600):
        root = tmp_path / str(rows)
        root.mkdir()
        with measured_sections():
            plan = scale.build(root, monkeypatch, snap=scale.snapshot(rows=rows))
            work = section_spans()['work']
        assert plan.initial_mode == 'nano'
        assert work['request_serializations'] == work['request_forms']
        seen[rows] = work['request_serializations']
    assert len(set(seen.values())) == 1, seen


def test_a_start_s_request_forms_do_not_follow_the_size_of_its_assignment(tmp_path, monkeypatch):
    """Two assignments of different sizes walk the same frames: the same mode, the same number of request forms, each
    serialized once. A form is a frame's shape, not a probe of the walk."""
    seen = {}
    for size in (5_000, 50_000):
        root = tmp_path / str(size)
        root.mkdir()
        captured = dataclasses.replace(scale.core(root), user_content_json=json.dumps('Assignment line. ' * (size // 17)))
        with measured_sections():
            plan = scale.build(root, monkeypatch, captured=captured)
            work = section_spans()['work']
        assert work['request_serializations'] == work['request_forms']
        seen[size] = (plan.initial_mode, work['request_serializations'])
    assert seen[5_000] == seen[50_000], seen


def _drawn_prices(monkeypatch):
    """The control of every price: story and room characters read from the drawn text, what the pieces must sum to."""
    monkeypatch.setattr(mv, 'story_chars', lambda snapshot, level=mv.FULL_VIEW: mv.json_chars(mv.render_story(snapshot, level)))
    monkeypatch.setattr(mv, 'room_chars', lambda snapshot, level=mv.FULL_VIEW, *, floor_note='': mv.json_chars(
        mv.render_room(snapshot, level, floor_note=floor_note)))


def test_one_start_lists_a_snapshot_s_floor_elements_once(tmp_path, monkeypatch):
    """Live scale (520 other conversations, ~17,000 words of people, no owner text): every walk, level, receipt and
    Max-room measure of one start reads one listing of the snapshot's elements; a changed snapshot lists its own. The
    frame walks its shortest view once and re-prices that level beside each later host form; the walk draws nothing, so
    the one full room render is the view this start delivers."""
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
    assert probes.count('re-price') <= len(facts['host_sections_addressed']) and len(renders) == 1
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
    render, probes = mf.render_view_for_mode, []

    def recorded(*args, **kwargs):
        result = render(*args, **kwargs)
        if kwargs.get('shortest') or kwargs.get('level') is not None:
            probes.append(('walk' if kwargs.get('level') is None else 're-price', kwargs['mode'],
                           mf.kept_level(args[0], result.facts).addressed, result.room))
        return result

    monkeypatch.setattr(mf, 'render_view_for_mode', recorded)
    settled = _near_even_plan(tmp_path / 'settled', monkeypatch, window, **shape)
    assert _hosts(settled) == hosts and settled.projection(settled.initial_mode).memory_facts['floor']['steps'] == steps
    walks = {str(forms) for kind, _mode, forms, _room in probes if kind == 'walk'}
    assert [kind for kind, *_ in probes] == ['walk', 're-price', 'walk', 're-price']  # the last re-price fit: no walk
    assert len(walks) == 2  # the walk after the missed re-price chose other forms
    chosen, probes[:] = list(probes), []
    _drawn_prices(monkeypatch)  # the control: every level priced by drawing it
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


def test_a_shortest_walk_walks_again_only_when_its_level_no_longer_fits(tmp_path, monkeypatch):
    """Owner shape at 92K: Max and Low cannot hold even their least request, so neither walks; Nano walks once, then at
    each of its next three host forms re-prices that level, misses the window (even its shortest view does not fit) and
    walks again."""
    render, probes = mf.render_view_for_mode, []

    def recorded(*args, **kwargs):
        if kwargs.get('shortest') or kwargs.get('level') is not None:
            probes.append(('walk' if kwargs.get('level') is None else 're-price', kwargs['mode']))
        return render(*args, **kwargs)

    monkeypatch.setattr(mf, 'render_view_for_mode', recorded)
    assert scale.build(tmp_path, monkeypatch).initial_mode == 'nano'
    assert probes == [('walk', 'nano')] + [('re-price', 'nano'), ('walk', 'nano')] * 3


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


def _files(root):
    return sorted((str(path.relative_to(root)), path.stat().st_size) for path in root.rglob('*') if path.is_file())


def test_a_pairwise_start_fit_is_the_production_start_fit_and_writes_nothing(tmp_path, monkeypatch):
    """``memory_floor.counterfactual_starts`` refits a memory in this plan's one captured frame by the start's own
    fitter: fed the very view this task captured, it chooses the start's mode, Max room and host steps; each host
    step is priced at the digest address its capture would publish, and nothing reaches the artifact store."""
    from ouroboros import artifacts
    from ouroboros.section_floor import predicted_section, retain_section

    for window in (92000, 84000, 76000, 68000):  # a frame tight enough that the start addresses a host section
        plan = scale.build(tmp_path, monkeypatch, window=window)
        shown = plan.projection(plan.initial_mode).memory_facts
        if shown.get('host_sections_addressed'):
            break
    assert shown.get('host_sections_addressed'), 'no fixture window made the start address a host section'
    captured = mv.snapshot_from_json(plan.core.memory_view_json)
    files = _files(tmp_path)
    monkeypatch.setattr(artifacts, 'store_actor_source_bytes', lambda *a, **kw: pytest.fail('a pairwise fit wrote'))
    fit = mf.counterfactual_starts(plan, {'as captured': captured})
    assert _files(tmp_path) == files
    arm = fit['arms']['as captured']
    assert fit['measured'] and arm['mode'] == plan.initial_mode and arm['max_room'] == shown['max_room']
    assert arm['host'] == [(row['section_id'], row['tokens_saved']) for row in shown['host_sections_addressed']]
    sections = {part.section_id: part for part in (*plan.core.semi_stable_sections, *plan.core.dynamic_head_sections,
                                                    *plan.core.dynamic_sections)}
    for row in shown['host_sections_addressed']:  # the same address, hence the same pointer cost
        assert predicted_section(sections[row['section_id']])['reader'] == row['reader']
    # Both arms: one captured frame, only the memory view differs; nothing is fetched or captured again.
    from ouroboros import context as context_module, tool_policy

    calls, fit_projections = [], cf._view_projections
    monkeypatch.setattr(tool_policy, 'initial_tool_schemas', lambda *a, **kw: pytest.fail('schemas refetched'))
    monkeypatch.setattr(context_module, '_capture_context_core', lambda *a, **kw: pytest.fail('core recaptured'))
    monkeypatch.setattr(cf, '_view_projections', lambda *a, **kw: (calls.append((a, kw)), fit_projections(*a, **kw))[1])
    other = dataclasses.replace(captured, story=captured.story[:-1])  # a different story, the same everything else
    mf.counterfactual_starts(plan, {'now': captured, 'proposed': other})
    (core_a, governance_a, user_a), kw_a = calls[0][0], calls[0][1]
    (core_b, governance_b, user_b), kw_b = calls[1][0], calls[1][1]
    assert core_a.memory_view_json != core_b.memory_view_json and governance_a is governance_b and user_a == user_b
    assert dataclasses.replace(core_a, memory_view_json='') == dataclasses.replace(core_b, memory_view_json='') == (
        dataclasses.replace(plan.core, memory_view_json=''))  # runtime, knowledge, marks text, assignment, books
    assert kw_a == kw_b and kw_a['tool_schemas'] == json.loads(plan.start_tool_schemas_json) == scale.tools()
    assert mv.snapshot_from_json(core_b.memory_view_json).marks == captured.marks
    monkeypatch.undo()
    retained = retain_section(sections[shown['host_sections_addressed'][0]['section_id']], root=str(tmp_path),
                              task_id='scale-fit')
    assert {**retained, 'predicted': True} == predicted_section(sections[retained['section_id']])
    text = '\n'.join(mf.counterfactual_lines(fit))
    assert f"- as captured: starts {plan.initial_mode.capitalize()}; Max room {shown['max_room']['cause']}: " in text
    assert f"This task started in {plan.initial_mode} (its start projection, a separate fact" in text
    # Without the inputs the frame needs, the fit is named as not measured, never guessed.
    for missing, bare in (('the tool schemas its start priced', dataclasses.replace(plan, start_tool_schemas_json=None)),
                          ('its window record', dataclasses.replace(plan, window_record=None)),
                          ('the captured context core and memory view', dataclasses.replace(plan, core=None))):
        assert mf.counterfactual_starts(bare, {'as captured': captured}) == {'measured': False, 'missing': [missing]}
    monkeypatch.setattr(cf, '_view_projections', lambda *a, **kw: {}['broken'])
    unknown = mf.counterfactual_starts(plan, {'as captured': captured})
    assert unknown['arms'] == {'as captured': {'unmeasured': 'KeyError'}}
    assert mf.counterfactual_lines(unknown)[-1] == '- as captured: not measured (KeyError); its mode is unknown'
    assert mf.counterfactual_lines({'measured': False, 'missing': ['x']})[-1] == (
        'not measured: this task\'s plan does not hold x.')


def test_a_lower_mode_names_its_cause_from_the_plans_own_start_facts(tmp_path, monkeypatch):
    """``context_measurement.mode_cause`` reads why a candidate renders below the owner's mode from the plan alone:
    a start the window lowered because memory crowded its Max room is ``memory``; a Max start renders no cause; a
    lower mode the task took after a Max start is ``after_start``. The physical context carries both facts."""
    from ouroboros.context_measurement import measure_main_fit, mode_cause

    def plan_at(window):
        return scale.build(tmp_path, monkeypatch, window=window)

    held, crowded = plan_at(1_000_000), None
    for window in (300_000, 200_000, 150_000, 120_000, 100_000, 92_000):
        candidate = plan_at(window)
        if candidate.initial_mode == 'low' and mf.crowded_by_memory(
                candidate.projection('low').memory_facts.get('max_room')):
            crowded = candidate
            break
    assert held.initial_mode == 'max' and crowded is not None
    assert mode_cause(crowded, 'low') == 'memory' and mode_cause(held, 'max') == ''
    assert mode_cause(held, 'low') == 'after_start' and mode_cause(crowded, 'nano') == 'after_start'
    disposition = measure_main_fit(crowded, crowded.messages_for('low'), scale.tools(), profile='task_local_low',
                                   rendered_mode='low', round_id='e:round:1')
    physical = cf._physical_context_for_fit(disposition)
    assert (physical.preferred_mode, physical.rendered_mode, physical.mode_cause) == ('max', 'low', 'memory')
    # The start is its own fact (``ContextFitPlan.start``): a route rebind that lowers a Max start for a smaller
    # window (``initial_mode`` follows that selection) is a lower mode taken after the start, and the next-start
    # read says the task started in Max; a Low start narrowed again keeps its start's cause.
    assert held.start == ('max', '') and crowded.start == ('low', 'memory')
    for window in (300_000, 200_000, 150_000, 120_000, 100_000):  # as ``loop_model_call`` rebinds: the new record
        record = plan_at(window).window_record
        rebound = dataclasses.replace(held, window_record=record, window_tokens=record.total_equivalent).reproject_for_route(
            window_tokens=record.total_equivalent, known_window=True, ratio=1.0, output_reserve=held.output_reserve_tokens,
            tool_schemas=scale.tools(), start_mode='max', current_messages=held.messages_for('max'))
        if rebound.initial_mode == 'low':
            break
    assert rebound.initial_mode == 'low' and rebound.start == held.start
    assert (rebound.projection('low').memory_facts.get('floor') or {}).get('mode_switch')  # the rebind's own switch
    assert mode_cause(rebound, 'low') == 'after_start'
    fit = mf.counterfactual_starts(rebound, {'as captured': mv.snapshot_from_json(rebound.core.memory_view_json)})
    assert fit['frame']['started'] == 'max'
    narrowed = crowded.reproject_for_route(
        window_tokens=crowded.window_tokens, known_window=True, ratio=1.0, output_reserve=crowded.output_reserve_tokens,
        tool_schemas=scale.tools(), start_mode='low', refused_tokens=2_000, current_messages=crowded.messages_for('low'))
    assert narrowed.initial_mode == 'low' and mode_cause(narrowed, 'low') == 'memory'


def test_a_pairwise_start_fit_reaches_the_starts_own_rescued_input(tmp_path, monkeypatch):
    """A large owner-Nano assignment no view of the window holds whole starts on its retained exact input pointer
    (``context_input_selection.rescued_start``): fed the very view it captured, the pairwise next-start fit applies
    that captured rescue as production does and chooses the same start; it neither reads nor writes for it. Without a
    captured rescue the next start's input is unknown: not measured, never a refusal."""
    from ouroboros import artifacts

    big = dataclasses.replace(scale.core(tmp_path), user_content_json=json.dumps("Large assignment line. " * 90_000))
    plan = scale.build(tmp_path, monkeypatch, window=140_000, mode='nano', captured=big)
    shown = plan.projection('nano')
    assert plan.initial_mode == 'nano' and shown.user_content_json and plan.projection('max').user_content_json is None
    captured = mv.snapshot_from_json(plan.core.memory_view_json)
    files = _files(tmp_path)
    monkeypatch.setattr(artifacts, 'store_actor_source_bytes', lambda *a, **kw: pytest.fail('a pairwise fit wrote'))
    arm = mf.counterfactual_starts(plan, {'as captured': captured})['arms']['as captured']
    assert _files(tmp_path) == files
    assert (arm['mode'], arm['input']) == ('nano', 'rescued')
    assert arm['steps'] == shown.memory_facts['floor']['steps']
    assert arm['host'] == [(row['section_id'], row['tokens_saved'])
                           for row in shown.memory_facts.get('host_sections_addressed', ())]
    assert '- as captured: starts Nano on the input this start rescued' in '\n'.join(
        mf.counterfactual_lines(mf.counterfactual_starts(plan, {'as captured': captured})))
    bare = dataclasses.replace(plan, nano_projection=dataclasses.replace(shown, user_content_json=None))
    assert mf.counterfactual_starts(bare, {'as captured': captured})['arms']['as captured'] == {
        'unmeasured': 'the whole input does not fit and this start captured no rescue of it'}


# ---- pieces of the view: each local form priced once, only the price stays, and a level costs what its text costs ------


def _snapshot() -> mv.MemoryViewSnapshot:
    lane = []
    for i in range(40):
        kind = "human" if i % 5 == 0 else "ouroboros"
        lane.append({**syn.spoken(i, kind, 200 + i), "owner_side": kind == "human", "carried": bool(i % 10)})
    room = syn.room("1", lane1=lane, lane2=12, notes=3, under=2)
    groups = [syn.conversation(i, notes=i % 3 == 0, words=4 if i % 4 else 2, word_chars=80, owner=i % 7 == 0)
              for i in range(12)]
    return syn.snapshot(room_facts=room, conversations=groups, story=syn.pages(12, 300) + syn.pointers(4, 3), mark_count=2)


def _levels(snapshot: mv.MemoryViewSnapshot, count: int = 25) -> list:
    table, rnd = mf._elements(snapshot), random.Random(11)
    pairs = [(step, ident) for step, ids in table.ids.items() for ident in ids]
    levels = [mv.FULL_VIEW]
    for _ in range(count):
        taken = set(rnd.sample(pairs, rnd.randint(1, len(pairs))))
        taken -= {("F5l", i) for i in table.ids.get("F5l", ()) if ("F5", i) in taken}
        for form, priors in mf._LIST_FORMS.items():
            if not any(step in priors for step, _ in taken):
                taken -= {(form, i) for i in table.ids.get(form, ())}
        levels.append(table.level(taken))
    return levels


def _priced(snapshot: mv.MemoryViewSnapshot, gone, piece=None) -> list:
    """Each part of ``## Other conversations`` at ``gone`` through the snapshot's pieces, or through another ``piece``."""
    return conversations.pieces(list(snapshot.conversations), gone, piece or snapshot.piece)


def _costs_as_drawn(snapshot: mv.MemoryViewSnapshot, gone) -> bool:
    prices = _priced(snapshot, gone)
    text = syn.conversations_text(dataclasses.replace(snapshot).conversations, gone)
    return sum(prices) + 4 * (len(prices) - 1) == mv.json_chars(text)  # JSON spells the blank line in four characters


def test_conversation_pieces_price_the_drawn_text_at_every_level():
    """The price of ``## Other conversations`` by its pieces is what its text costs, at any level: a group's key that
    forgets something its form reads would answer another level's price here."""
    snapshot = _snapshot()
    for level in _levels(snapshot):
        assert _costs_as_drawn(snapshot, dict(level.addressed))


def test_a_repeated_level_makes_no_new_piece_and_a_new_form_makes_only_its_own():
    """Inside a measured window a group's form is rendered once per key: pricing the same level again adds no piece,
    a level that changes one group's form makes only that group's new piece, and nothing repeats."""
    snapshot, level_gone = _snapshot(), {"F6w": []}
    with measured_sections():
        _priced(snapshot, level_gone)
        first = section_spans()["work"]["view_pieces"]
        _priced(snapshot, level_gone)
        assert section_spans()["work"]["view_pieces"] == first
        word = snapshot.conversations[0]["rooms"][0]["words"][0]["id"]
        _priced(snapshot, {"F6w": [word]})
        work = section_spans()["work"]
    assert work["view_pieces"] == first + 1 and not work.get("view_pieces_again")


def test_without_the_memo_the_counter_names_the_repeat():
    """The guard's other direction: pricing the same forms through a piece that keeps nothing counts them again."""
    from ouroboros.context_timing import counted

    snapshot = _snapshot()

    def unkept(key, make):
        counted("view_pieces", key=(id(snapshot), key))
        return mv.json_chars(make())

    with measured_sections():
        _priced(snapshot, {}, unkept)
        _priced(snapshot, {}, unkept)
        work = section_spans()["work"]
    assert work["view_pieces_again"] == work["view_pieces"] / 2


def test_pieces_never_reach_the_snapshot_s_canonical_form():
    """The memo is a non-field attribute: the snapshot's JSON and equality do not see it, a replaced snapshot has its own."""
    snapshot = _snapshot()
    before = mv.snapshot_json(snapshot)
    _priced(snapshot, {})
    assert mv.snapshot_json(snapshot) == before and "_pieces" in snapshot.__dict__
    assert "_pieces" not in dataclasses.replace(snapshot).__dict__


def test_a_snapshot_keeps_the_price_of_a_form_and_no_text():
    """After a level is walked and delivered the snapshot holds a number per form it priced: no text of a form the
    floor priced and left (the walk prices far more forms than it delivers) outlives its price."""
    snapshot = _snapshot()
    level = mf.fit_memory_view(snapshot, {"margin": 0, "physical": 0, "budget": None})
    fit = mf.ViewFit(snapshot, level, "", {})
    assert fit.story and fit.room
    held = snapshot.__dict__["_pieces"]
    assert len(held) > 2 and all(isinstance(price, int) for price in held.values()), held


@pytest.mark.parametrize("lined, counted", [(0, False), (1, False), (2, True)])
def test_the_counted_rung_and_the_group_lines_cost_their_drawn_text(lined, counted):
    """The last rung's block and the group lines are pieces too: their prices are the drawn text's."""
    snapshot = _snapshot()
    keys = [group["key"] for group in snapshot.conversations][:lined]
    assert _costs_as_drawn(snapshot, {**({"F6g": keys} if keys else {}), **({"F6a": ["conversations"]} if counted else {})})


@pytest.mark.parametrize("note", ["", "### Physical floor\nThis view keeps \"quotes\" and a \\ backslash."])
def test_a_level_is_priced_by_its_pieces_exactly_as_its_drawn_text(note):
    """The price of a level is the sum of its pieces and the separators that join them: at any level, with or without the
    floor note, ``story_chars`` and ``room_chars`` are the JSON characters of the texts a snapshot without pieces draws.
    A piece whose key forgets a step its text reads answers another level's price here."""
    snapshot = _snapshot()
    fresh = dataclasses.replace(snapshot)
    for level in _levels(snapshot):
        assert mv.story_chars(snapshot, level) == mv.json_chars(mv.render_story(fresh, level))
        assert mv.room_chars(snapshot, level, floor_note=note) == mv.json_chars(mv.render_room(fresh, level, floor_note=note))


def test_a_level_holds_the_whole_story_only_when_its_text_is_the_whole_story():
    """Equal prices alone prove nothing for a level naming story steps: those texts decide, and the whole view needs no text."""
    snapshot = _snapshot()
    table = mf._elements(snapshot)
    story_step = next(step for step in mv.STORY_STEPS if table.ids.get(step))
    assert mv.whole_story_at(snapshot, mv.FULL_VIEW)
    assert not mv.whole_story_at(snapshot, table.level({(story_step, table.ids[story_step][0])}))
    with measured_sections():
        assert mv.whole_story_at(snapshot, mv.FloorLevel(addressed=(("F5", ("no-such-page",)),)))
        assert section_spans()["work"]["view_renders"] == 2  # the two texts an equal price left to compare


def test_an_unactivated_snapshot_is_priced_without_the_floor_note():
    """Before activation the room is one line and the note is not added: the price says the same as the text."""
    snapshot = _snapshot()
    inactive = dataclasses.replace(snapshot, store_status={"state": "importing", "reason": "not yet"})
    assert mv.room_chars(inactive, mv.FULL_VIEW, floor_note="### Physical floor") == mv.json_chars(
        mv.render_room(inactive, mv.FULL_VIEW, floor_note="### Physical floor"))
    assert "### Physical floor" not in mv.render_room(inactive, mv.FULL_VIEW, floor_note="### Physical floor")


def test_a_view_fit_is_drawn_only_when_read():
    """A frame the fit walks away from is priced by its pieces, never drawn whole: ``ViewFit`` draws its texts on the
    first read."""
    snapshot = _snapshot()
    fit = mf.ViewFit(snapshot, mv.FULL_VIEW, "", {})
    with measured_sections():
        assert fit.story_chars and fit.room_chars
        assert not section_spans()["work"].get("view_renders")
        story, room = fit.story, fit.room
        assert section_spans()["work"]["view_renders"] == 2
    assert (story, room) == (mv.render_story(snapshot), mv.render_room(snapshot)) and fit.facts == {}


def test_a_start_builds_a_story_or_room_text_only_as_a_counted_piece_or_render(tmp_path, monkeypatch):
    """Counted at the builders themselves: a start builds a story or room text once per piece key to price it, keeping
    its length (``view_pieces``), and whole for what it delivers (``view_renders``), never elsewhere. One text built
    outside those two paths breaks the equality."""
    from ouroboros import context_timing

    built = []
    for name in ("_story_text", "_room_text"):
        def building(*args, _build=getattr(mv, name), **kwargs):
            built.append(name)
            return _build(*args, **kwargs)

        monkeypatch.setattr(mv, name, building)

    def counted():  # the story and room pieces the window counted, and its whole renders
        work, seen = context_timing._WORK.get()
        return sum(verb == "view_pieces" and key[1][0] in ("story", "room") for verb, key in seen) + work["view_renders"]

    snapshot = scale.snapshot()
    mv._parsed_view.cache_clear()  # a fresh process's parse: no piece priced by an earlier test of this worker
    with measured_sections():
        scale.build(tmp_path, monkeypatch, snap=snapshot)
        assert section_spans()["work"]["view_renders"] == 2 and len(built) == counted() > 2
        mv._story_text(snapshot, {})  # a text built outside a piece and a render
        assert len(built) == counted() + 1
