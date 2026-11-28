"""First-send room and a running call's measured window recovery."""
from dataclasses import replace
import json
import math

import pytest

from ouroboros.contracts.live_conversation import LiveOptions
from ouroboros.realtime_setup import measure_session_wire, session_candidate
from tests._live_world import host_world as host_world


@pytest.mark.parametrize('first', [False, True])
def test_late_frame_gets_one_start_room_only_on_the_first_full_candidate(host_world, first):
    from tests.test_memory_view_lever import plan
    host = host_world.host
    source, _ = plan()
    host.plan = host.ctx.context_fit_plan = replace(host.plan, core=source.core, core_sha256=source.core_sha256,
        max_projection=source.max_projection, low_projection=source.low_projection,
        nano_projection=replace(source.max_projection, mode='nano'), output_reserve_tokens=10000)
    host.messages[:] = host.plan.messages_for('nano')
    options = LiveOptions(instructions='Late feature instructions and provider envelope. ' * 150)
    factory = host_world.provider['session']
    _, _, before, _ = measure_session_wire(host, host.plan, host.messages, host.schemas, factory, options)
    initial = before['fit']['measurement']['estimated_input_tokens']
    window = math.ceil((initial + 10000) / .94)
    host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=window))
    host._session_opened = not first
    previous_system = host.messages[0]
    _, _, after = session_candidate(host, factory, options)
    measured, room = after['fit']['measurement'], after['work_room']
    assert room['first_send'] is first
    assert room['requested_tokens'] == (math.ceil(window / 8) if first else 0)
    assert room['remaining_tokens'] == window - 10000 - measured['estimated_input_tokens']
    assert room['deficit_tokens'] == 0, after
    if first:
        assert measured['estimated_input_tokens'] < initial
        assert room['remaining_tokens'] >= math.ceil(window / 8)
    else:
        assert host.messages[0] == previous_system, 'renewal may not buy another start room by narrowing memory'
        _, actual, _, _ = measure_session_wire(host, host.plan, host.messages, host.schemas, factory, options, _room_fact=False)
        assert measured['estimated_input_tokens'] == actual.prompt_tokens_estimate
        assert room['remaining_tokens'] < math.ceil(window / 8)


def test_start_with_less_room_is_not_a_physical_refusal(host_world):
    from ouroboros.contracts.realtime import RealtimeSetupError
    host = host_world.host
    options = LiveOptions(instructions='Held feature instructions. ' * 1200)
    factory = host_world.provider['session']
    # This fixture is already at its small memory floor; a large immutable
    # feature frame fits physically but cannot leave another eighth empty.
    _, _, baseline, _ = measure_session_wire(host, host.plan, host.messages, host.schemas, factory, options)
    window = baseline['fit']['measurement']['estimated_input_tokens'] + 10000 + 500
    host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=window))
    _, _, result = session_candidate(host, factory, options)
    room = result['work_room']
    assert 0 <= room['remaining_tokens'] < room['requested_tokens'] and room['deficit_tokens'] > 0, result
    assert any(row.get('_host_context_kind') == 'context_start_room' and
               'Start room is smaller than requested' in row['content'] for row in host.messages)
    assert result['fit']['measurement']['capacity_deficit_tokens'] == 0
    with pytest.raises(RealtimeSetupError, match='context_capacity'):
        session_candidate(host, factory, replace(options, instructions=options.instructions * 2))


def test_running_voice_names_a_book_only_refusal_without_cutting(host_world):
    from ouroboros import context_fit as cf
    from ouroboros.contracts.realtime import RealtimeSetupError

    host = host_world.host
    core = replace(host.plan.core, development_md='Required handbook rules. ' * 3_000, development_required=True)
    books = {m: cf._governance_blocks(None, core, mode=m) for m in ('max', 'low')}
    mode = host.plan.initial_mode
    view, _ = cf._view_projections(core, books, core.user_content_json, preferred=mode, tool_schemas=host.schemas,
        window_tokens=0, known_window=False, output_reserve=host.plan.output_reserve_tokens, ratio=1)
    held = replace(host.plan.projection(mode), system_content_json=json.dumps(view[mode][0]), memory_facts=view[mode][1])
    host.plan = replace(host.plan, core=core, **{f'{mode}_projection': held})
    host.messages[:] = host.plan.messages_for(mode)
    host._session_opened = True
    options, factory = LiveOptions(resume_handle='fixture'), host_world.provider['session']
    _, request, _, _ = measure_session_wire(host, host.plan, host.messages, host.schemas, factory, options)
    window = request.prompt_tokens_estimate + request.physical_context.response_reserve_tokens - 2_000
    host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=window))
    original = list(host.messages)
    with pytest.raises(RealtimeSetupError, match='context_capacity:.*whole development handbook'):
        session_candidate(host, factory, options)
    assert host.messages == original
    assert host.trace['fit_refusal']['recovery'] == []
    assert all(not s.opened for s in host_world.sessions)


def test_running_voice_exposes_only_one_body_for_the_physical_miss(host_world):
    from ouroboros.context_budget import RECLAIM_LOW_WATER_DIVISOR
    from ouroboros.llm_attempt import main_context_fit_refusal
    from tests._context_source_fixtures import _observation, _tool

    host = host_world.host
    host._session_opened = True
    first = _tool('first', invocation='inv-first', text='A first observed line.\n' * 900)
    second = _tool('second', invocation='inv-second', text='A second observed line.\n' * 900)
    own = {'role': 'user', 'content': 'Keep the exact owner instructions. ' * 2_000}
    host.messages.extend([*first, *second, own])
    host.ctx._last_context_observation = _observation(host.messages)
    options, factory = LiveOptions(), host_world.provider['session']
    _, before, _, _ = measure_session_wire(host, host.plan, host.messages, host.schemas, factory, options, _room_fact=True)
    window = before.prompt_tokens_estimate + before.physical_context.response_reserve_tokens - 2_000
    host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=window))
    _, request, facts = session_candidate(host, factory, options)
    assert main_context_fit_refusal(request) is None
    assert sum(row in host.messages for row in (first[-1], second[-1])) == 1
    assert own in host.messages and [step['rung'] for step in facts['recovery']].count('bodies') == 1
    assert 0 <= facts['work_room']['remaining_tokens'] < math.ceil(window / RECLAIM_LOW_WATER_DIVISOR)


@pytest.mark.parametrize('host_world', [('low', 500_000, 'estimator')], indirect=True)
def test_native_resume_prices_nano_schemas_and_keeps_loaded_and_feature_tools(host_world, monkeypatch):
    from ouroboros.llm_attempt import main_context_fit_refusal

    host = host_world.host
    host._session_opened = True
    host._reselect_tools('low')
    schema = lambda name, text: {'type': 'function', 'function': {'name': name,
        'description': text, 'parameters': {'type': 'object', 'properties': {}}}}
    loaded = schema('fixture_already_enabled', 'Already loaded explicitly.')
    revoked = schema('fixture_permission_revoked', 'No longer permitted.')
    discarded = schema('fixture_low_only', 'Large Low-only schema documentation. ' * 20_000)
    list_current, get_current = host.tools.schemas, host.tools.get_schema_by_name
    monkeypatch.setattr(host.tools, 'schemas', lambda: [*list_current(), loaded])
    monkeypatch.setattr(host.tools, 'get_schema_by_name', lambda name: loaded if name == 'fixture_already_enabled' else get_current(name))
    host.schemas.extend([loaded, revoked, discarded])
    host.ctx._actor_loaded_tool_names.update({'fixture_already_enabled', 'fixture_permission_revoked'})
    host.messages.append({'role': 'user', 'content': 'Keep the end of the resumed transcript.'})
    options, factory = LiveOptions(resume_handle='fixture-native'), host_world.provider['session']
    _, before, _, _ = measure_session_wire(host, host.plan, host.messages, host.schemas, factory, options, _room_fact=False)
    window = 145_000
    assert before.prompt_tokens_estimate + before.physical_context.response_reserve_tokens > window
    host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=window))
    session, request, facts = session_candidate(host, factory, options)
    assert not session.opened and main_context_fit_refusal(request) is None
    assert host._schemas_mode == host.plan.initial_mode == 'nano'
    names = {entry['function']['name'] for entry in host.schemas}
    assert {'fixture_already_enabled', 'set_test_language'} <= names and 'fixture_low_only' not in names
    assert 'fixture_permission_revoked' not in names | host.ctx._actor_loaded_tool_names
    assert facts['work_room']['remaining_tokens'] == window - request.physical_context.response_reserve_tokens - request.prompt_tokens_estimate


@pytest.mark.parametrize('host_world', [('max', 1_000_000, 'estimator')], indirect=True)
@pytest.mark.parametrize('route,resume,excess,mode,deeper', [
    ('fixture', '', 4_000, 'max', False), ('fixture', '', 87_000, 'low', False),
    ('fixture', '', 285_000, 'low', True), ('fixture', 'fixture-resume', 4_000, 'max', False),
    ('grok', '', 87_000, 'low', False),
])
def test_running_voice_wall_lowers_the_body_map_before_going_below_tellings(host_world, route, resume, excess, mode, deeper):
    """A running call over its window takes the lowered frame (the Body Map at its parts) before any of my memory goes
    below a telling. The stimulus is a Body Map the size the repository's whole Architecture book had (about 464K
    characters) with its real parts form: before 26.11.2026 Max carried that whole book and Low its chapter maps."""
    from time import perf_counter

    from ouroboros import context_fit as cf, memory_floor as mf, memory_view as mv
    from ouroboros.context_budget import RECLAIM_LOW_WATER_DIVISOR
    from ouroboros.llm_attempt import main_context_fit_refusal
    from ouroboros.memory_forms import stage_of
    from tests._memory_view_synthetic import pages, room, said, snapshot

    host = host_world.host
    owner = said(1, 'human', 'Keep the route and all my words exactly.', owner_side=True, immune=True)
    story = [{**page, **({'lead_rule': True} if i < 32 else {})} for i, page in enumerate(pages(64, 12_000))]
    memory = snapshot(story=story, room_facts=room(lane1=[owner]))
    real = host.plan.core
    assert real.body_map_parts and real.section_root and real.section_task_id  # the Map can step to its parts here
    core = replace(real, memory_view_json=mv.snapshot_json(memory),
                   body_map=real.body_map + '\n\n' + 'chapter opening words ' * 21_000)
    whole = host.plan.projection('max')
    books = {m: cf._governance_blocks(None, core, mode=m) for m in ('max', 'low')}
    view, _ = cf._view_projections(core, books, host.plan.user_content_json, preferred='max', tool_schemas=host.schemas,
        window_tokens=0, known_window=False, output_reserve=host.plan.output_reserve_tokens, ratio=1)
    host.plan = host.ctx.context_fit_plan = replace(host.plan, core=core, initial_mode='max',
        max_projection=replace(whole, system_content_json=json.dumps(view['max'][0]), memory_facts=view['max'][1]))
    host.messages[:] = host.plan.messages_for('max')
    host.messages.append({'role': 'user', 'content': 'Exact transcript tail for the resumed conversation.'})
    host._session_opened = True
    options = LiveOptions(resume_handle=resume, instructions='Additional feature instructions. ' * 200)
    expected_reserve = 10_000
    if route == 'grok':
        from skills.grok_realtime.plugin import GrokSession

        def no_connection(*_a, **_kw):
            pytest.fail('sizing must not open a real connection')

        def factory(*args, **kw):
            return GrokSession(*args, connect=no_connection, **kw)

        expected_reserve = 65_536  # no native output bound; Main admission charges the selected allowance
        host.plan = replace(host.plan, output_reserve_tokens=expected_reserve)
    else:
        factory = host_world.provider['session']
    _, request, _, _ = measure_session_wire(host, host.plan, host.messages, host.schemas, factory, options)
    initial = request.prompt_tokens_estimate
    window = initial + expected_reserve - excess
    host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=window))
    started = perf_counter()
    session, request, evidence = session_candidate(host, factory, options)
    elapsed_ms = (perf_counter() - started) * 1_000
    print(f'voice reconnect fit: {resume or "fresh"} {excess=} {initial=} {window=} '
          f'mode={host.plan.initial_mode} free={evidence["work_room"]["remaining_tokens"]} {elapsed_ms=:.2f}')

    assert evidence['work_room']['first_send'] is False
    assert main_context_fit_refusal(request) is None
    measurement = evidence['fit']['measurement']
    free = window - expected_reserve - measurement['estimated_input_tokens']
    assert request.physical_context.response_reserve_tokens == measurement['response_reserve_tokens'] == expected_reserve
    assert evidence['work_room']['remaining_tokens'] == free >= math.ceil(window / RECLAIM_LOW_WATER_DIVISOR)
    assert host.plan.preferred_mode == 'max' and host.plan.initial_mode == mode
    level = mf.kept_level(memory, host.plan.projection(mode).memory_facts)
    assert (any(stage_of(memory, step, ident) not in mf.TELLING_STAGES
                for step, ids in level.addressed for ident in ids)) is deeper
    assert owner['line'] in mv.render_room(memory, level)
    if route == 'grok':
        assert not session.ready and 'Exact transcript tail for the resumed conversation.' in json.dumps(session.wire)
    else:
        assert not session.opened
        assert any(row.get('content') == 'Exact transcript tail for the resumed conversation.' for row in session.wire['messages'])
    if mode == 'low' and not deeper:
        assert sum(dict(level.steps).values()) < 32  # the full Max trial is not the starting point of Low


@pytest.mark.parametrize('host_world', [('low', 1_000_000, 'estimator')], indirect=True)
@pytest.mark.parametrize('route', ['grok', 'gemini'])
def test_a_running_nano_trial_prices_its_own_reserve_not_lows_reply(host_world, route):
    """Low's selected reply is not Nano's working floor: the engine admits every page Nano holds beside its own reply.

    On these wires Low's admission charges the selected 65,536-token reply (Grok declares no native bound; Gemini's is
    that allowance), the route's working floor is 10,000 and Nano's reply takes what its input leaves. A running Low
    call that misses its window and steps down must not price the Nano trial with Low's reply as its floor.
    """
    from ouroboros import context_fit as cf
    from ouroboros import memory_floor as mf
    from ouroboros import memory_view as mv
    from ouroboros.context_budget import RECLAIM_LOW_WATER_DIVISOR
    from ouroboros.llm_attempt import main_context_fit_refusal
    from tests._memory_view_synthetic import pages, room, said, snapshot

    def no_connection(*_a, **_kw):
        pytest.fail('sizing must not open a real connection')

    if route == 'grok':
        from skills.grok_realtime.plugin import CATALOG
        from skills.grok_realtime.plugin import GrokSession as Session
        facts = CATALOG.models[0].facts
    else:
        from skills.gemini_live.plugin import FACTS as facts
        from skills.gemini_live.plugin import GeminiSession as Session

    def factory(*args, **kw):
        return Session(*args, connect=no_connection, **kw)

    host = host_world.host
    assert host._schemas_mode == 'low'
    owner = said(1, 'human', 'Keep my words exactly.', owner_side=True, immune=True)
    memory = snapshot(story=pages(20, 2_000), room_facts=room(lane1=[owner]))
    core = replace(host.plan.core, memory_view_json=mv.snapshot_json(memory))
    books = {m: cf._governance_blocks(None, core, mode=m) for m in ('max', 'low')}
    view, _ = cf._view_projections(core, books, host.plan.user_content_json, preferred='low', tool_schemas=host.schemas,
        window_tokens=0, known_window=False, output_reserve=65_536, ratio=1)
    nano = host.plan.nano_projection or replace(host.plan.low_projection, mode='nano')
    host.plan = host.ctx.context_fit_plan = replace(host.plan, core=core, initial_mode='low', output_reserve_tokens=65_536,
        low_projection=replace(host.plan.low_projection, system_content_json=json.dumps(view['low'][0]),
                               memory_facts=view['low'][1]),
        nano_projection=replace(nano, system_content_json=json.dumps(view['nano'][0]), memory_facts=view['nano'][1]))
    host.messages[:] = host.plan.messages_for('low')
    host.messages.append({'role': 'user', 'content': 'Exact transcript tail of the running call.'})
    host._session_opened = True
    host.model = replace(host.model, facts=facts)
    window = facts.conversation_window_tokens

    _, request, evidence = session_candidate(host, factory, LiveOptions())
    shown = host.plan.projection(host.plan.initial_mode).memory_facts
    level = mf.kept_level(memory, shown)
    folded = sorted({ident for _, ids in level.addressed for ident in ids if ident.startswith('page-')})
    outlined = [section['section_id'] for section in shown.get('host_sections_addressed') or ()]
    measurement = evidence['fit']['measurement']
    print(f'running nano trial: {route} {window=} mode={host.plan.initial_mode} folded_pages={len(folded)} '
          f'outlined={outlined} input={measurement["estimated_input_tokens"]} '
          f'reserve={measurement["response_reserve_tokens"]} free={evidence["work_room"]["remaining_tokens"]} '
          f'recovery={evidence["recovery"]}')
    assert main_context_fit_refusal(request) is None
    assert host.plan.initial_mode == host._schemas_mode == 'nano'
    assert not folded and not outlined, f'{len(folded)} pages and {outlined} given up for a reserve Nano does not charge'
    # The return eighth where memory could buy it: Nano's reply takes what its input leaves above the floor, so a fold
    # would lengthen that reply, not the free room (the envelope's own few tokens stay outside the reply's estimate).
    assert [step['rung'] for step in evidence['recovery']] == ['floor']
    assert (evidence['work_room']['remaining_tokens'] >= math.ceil(window / RECLAIM_LOW_WATER_DIVISOR)
            or measurement['response_reserve_tokens'] > facts.working_reserve_tokens)


@pytest.mark.parametrize('engine', ['gemini', 'grok'])
def test_running_nano_prices_reply_after_serializing_the_wire(host_world, engine):
    from ouroboros.context_budget import reply_allowance_tokens
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from ouroboros.contracts.realtime import canonical_json, sha256
    from ouroboros.llm_attempt import main_context_fit_refusal
    from skills.gemini_live.plugin import GeminiSession
    from skills.grok_realtime.plugin import GrokSession
    from tests._context_source_fixtures import _tool

    def no_connection(*_a, **_kw):
        pytest.fail('only an unopened wire is measured here')

    def factory(*args, **kwargs):
        cls = GeminiSession if engine == 'gemini' else GrokSession
        return cls(*args, connect=no_connection, **kwargs)

    host = host_world.host
    host._session_opened = True
    host.plan = replace(host.plan, output_reserve_tokens=65_536)
    host.model = replace(host.model, facts=replace(host.model.facts, basis='estimator'))
    # Tool history takes the archive form on both routes: its escaping is charged twice in the setup wire.
    host.messages.extend(_tool('archive', text='"\\' * 40_000))
    options = LiveOptions(resume_handle='synthetic-account-resume' if engine == 'gemini' else '')
    _, baseline, _, _ = measure_session_wire(host, host.plan, host.messages, host.schemas, factory, options)
    window = baseline.prompt_tokens_bounded_estimate + 28_000
    host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=window))

    session, request, evidence = session_candidate(host, factory, options)
    raw, reply = request.prompt_tokens_bounded_estimate, session.selected.output_allowance_tokens
    canonical = estimate_context_prompt_tokens(json.loads(session.selected.messages_json),
                                                json.loads(session.selected.tools_json))
    old_reply = reply_allowance_tokens(caller_max_tokens=65_536, nano=True, owner_nano=True,
                                      input_tokens=canonical, window_tokens=window)
    assert raw + old_reply > window  # the canonical allowance would trigger a false refusal
    assert reply == reply_allowance_tokens(caller_max_tokens=65_536, nano=True, owner_nano=True,
                                         input_tokens=raw, window_tokens=window)
    assert request.physical_context.response_reserve_tokens == max(10_000, reply)
    assert evidence['fit']['measurement']['reply_allowance_tokens'] == reply
    assert request.candidate_raw_sha256 == evidence['wire_sha256'] == sha256(canonical_json(session.wire))
    assert main_context_fit_refusal(request) is None and not evidence['recovery']
    assert 'archive' in session.selected.messages_json
    if engine == 'gemini':
        assert session.events[0]['setup']['generationConfig']['maxOutputTokens'] == reply


def test_a_new_nano_setup_can_raise_the_reply_after_pricing_its_wire(host_world, monkeypatch):
    from ouroboros import capability_evidence as ce
    from ouroboros.context_budget import reply_allowance_tokens
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from ouroboros.context_window import WindowLimit, WindowRecord
    from ouroboros.llm_attempt import main_context_fit_refusal
    from skills.gemini_live.plugin import GeminiSession

    def no_connection(*_a, **_kw):
        pytest.fail('only the Gemini serializer is used')

    def factory(*args, **kwargs):
        return GeminiSession(*args, connect=no_connection, **kwargs)

    host = host_world.host
    host.messages.append({'role': 'user', 'content': '"\\' * 24_000})
    options = LiveOptions()
    canonical = estimate_context_prompt_tokens(host.messages, host.schemas)
    _, initially, _, _ = measure_session_wire(host, host.plan, host.messages, host.schemas, factory, options, _room_fact=False)
    wire = initially.prompt_tokens_bounded_estimate
    assert wire > canonical
    width = wire + 28_000
    limit = WindowLimit(width, 'total', 'provider_metadata')
    host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=width))
    host.plan = replace(host.plan, output_reserve_tokens=65_536, window_tokens=width,
        window_record=WindowRecord('voice', total=limit, declared=(limit,), reserve=10_000))

    def density(*_args, estimate_tokens, **_kwargs):
        return (2.0, 'fresh_route_usage') if estimate_tokens <= (wire + canonical) // 2 else (1.0, 'cold_estimate')

    monkeypatch.setattr(ce, 'resolve_main_token_density', density)
    session, request, evidence, _ = measure_session_wire(host, host.plan, host.messages, host.schemas, factory, options,
                                                       _room_fact=False)
    old = reply_allowance_tokens(caller_max_tokens=65_536, nano=True, owner_nano=True,
        input_tokens=2 * canonical, window_tokens=width)
    reply = session.events[0]['setup']['generationConfig']['maxOutputTokens']
    assert reply > old and reply == evidence['fit']['measurement']['reply_allowance_tokens'] == request.max_completion_tokens
    assert reply == reply_allowance_tokens(caller_max_tokens=65_536, nano=True, owner_nano=True,
        input_tokens=request.prompt_tokens_bounded_estimate, window_tokens=width)
    assert main_context_fit_refusal(request) is None
    print(f'opening {canonical=}, actual {wire=}, window={width}; reply {old} → {reply}')


def test_nano_wire_reply_pricing_converges_across_digit_and_density_boundaries(host_world, monkeypatch):
    from ouroboros import capability_evidence as ce
    from ouroboros.context_budget import reply_allowance_tokens
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from ouroboros.context_window import WindowLimit, WindowRecord
    from ouroboros.contracts.realtime import canonical_json, sha256
    from ouroboros.llm_attempt import main_context_fit_refusal
    from skills.gemini_live.plugin import GeminiSession

    builds = []
    factory = lambda *a, **kw: (builds.append(a[0].output_allowance_tokens) or
        GeminiSession(*a, connect=lambda *_a, **_kw: pytest.fail('only the Gemini serializer is used'), **kw))
    host, options = host_world.host, LiveOptions()
    base = list(host.messages)
    for pad in range(4):
        host.messages[:] = [*base, {'role': 'user', 'content': '"\\' * 24_000 + 'a' * pad}]
        probe, _, _, _ = measure_session_wire(host, host.plan, host.messages, host.schemas, factory, options,
                                              _room_fact=False, opening_allowance_tokens=8_192)
        chars4 = estimate_context_prompt_tokens([probe.wire], chars=True)
        if chars4 % 4 == 0:
            break
    canonical, raw4, raw5 = estimate_context_prompt_tokens(host.messages, host.schemas), (chars4 + 3) // 4, (chars4 + 4) // 4
    assert raw5 == raw4 + 1

    def set_window(width):
        limit = WindowLimit(width, 'total', 'provider_metadata') if width else None
        host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=width))
        host.plan = replace(host.plan, output_reserve_tokens=65_536, window_tokens=width or 0,
            window_record=WindowRecord('voice', total=limit, declared=(limit,) if limit else (), reserve=10_000))

    def width_for(measured_input, target):
        w = math.ceil((measured_input + target) * 8 / 7)
        while w - math.ceil(w / 8) != measured_input + target:
            w += 1 if w - math.ceil(w / 8) < measured_input + target else -1
        return w

    def check(width, density_fn=None):
        set_window(width)
        if density_fn is not None:
            monkeypatch.setattr(ce, 'resolve_main_token_density', density_fn)
        builds.clear()
        sess, req, ev, _ = measure_session_wire(host, host.plan, host.messages, host.schemas, factory, options, _room_fact=False)
        reply = sess.events[0]['setup']['generationConfig']['maxOutputTokens']
        allowed = reply_allowance_tokens(caller_max_tokens=65_536, nano=True, owner_nano=True,
            input_tokens=req.prompt_tokens_estimate, raw_input_tokens=req.prompt_tokens_bounded_estimate, window_tokens=width)
        assert reply == ev['fit']['measurement']['reply_allowance_tokens'] == req.max_completion_tokens
        assert ev['fit']['measurement']['response_reserve_tokens'] == max(10_000, reply)
        assert req.candidate_raw_sha256 == ev['wire_sha256'] == sha256(canonical_json(sess.wire))
        assert reply <= allowed and main_context_fit_refusal(req) is None
        return reply, allowed, req.prompt_tokens_bounded_estimate

    # Real local density witness valid for canonical and raw4 (raw4/2), dropping out at raw5: two fixed points
    # (8,192 refused at raw4 vs 5-digit reply fitting at raw5); top-down pricing picks the larger safe one.
    ce.record_token_density(host.ctx.drive_root, host.plan.model, prompt_chars=2 * raw4, prompt_tokens=raw4,
        route_fp=host.plan.route_fp, basis=ce.MAIN_DENSITY_BASIS, view_kind=host.plan.view_kind)
    reply, allowed, raw = check(raw5 + 28_000)
    assert raw == raw5 and reply == allowed > 10_000

    # Fixed-point raise/shrink, no-fixed-point 4<->5 edges (density 1 and 2), and upward density jump at raw5.
    cold = lambda *_a, **_kw: (1.0, 'cold_estimate')
    raise_d = lambda *_a, estimate_tokens, **_kw: (
        (2.0, 'fresh_route_usage') if estimate_tokens <= (raw5 + canonical) // 2 else (1.0, 'cold_estimate'))
    cases = [
        (raw5 + 28_000, raise_d, raw5, None, True),
        (width_for(raw4, 9_500), cold, raw4, 9_500, True),
        (width_for(raw4, 10_000), raise_d, raw4, 9_999, False),
        (width_for(raw4, 10_000), cold, raw4, 9_999, False),
        (width_for(2 * raw4, 10_000), lambda *_a, estimate_tokens, **_kw: (
            (4.0, 'fresh_route_usage') if estimate_tokens <= (raw5 + canonical) // 2 else (2.0, 'fresh_route_usage')), raw4, 9_999, False),
        (width_for(2 * raw4, 10_000), lambda *_a, **_kw: (2.0, 'fresh_route_usage'), raw4, 9_999, False),
        (raw5 + 28_000, lambda *_a, estimate_tokens, **_kw: (
            (2.0, 'fresh_route_usage') if estimate_tokens <= (raw5 + canonical) // 2
            else (1.0, 'cold_estimate') if estimate_tokens <= raw4 else (1.5, 'fresh_route_usage')), raw4, 9_999, False),
    ]
    for width, fn, exp_raw, exp_reply, exact in cases:
        reply, allowed, raw = check(width, fn)
        assert raw == exp_raw and (reply == allowed if exact else reply < allowed)
        if exp_reply is not None:
            assert reply == exp_reply

    # Unknown window keeps the initial Nano allowance in one build; an open socket keeps its opening allowance.
    set_window(None)
    monkeypatch.setattr(ce, 'resolve_main_token_density', cold)
    builds.clear()
    sess, req, _, _ = measure_session_wire(host, host.plan, host.messages, host.schemas, factory, options, _room_fact=False)
    expected_unknown = reply_allowance_tokens(caller_max_tokens=65_536, nano=True, owner_nano=True, input_tokens=canonical)
    assert builds == [expected_unknown] and req.max_completion_tokens == expected_unknown
    set_window(raw5 + 28_000)
    builds.clear()
    sess, req, ev, _ = measure_session_wire(host, host.plan, host.messages, host.schemas, factory, options,
                                            _room_fact=False, opening_allowance_tokens=12_345)
    assert builds == [12_345] and sess.events[0]['setup']['generationConfig']['maxOutputTokens'] == 12_345
    assert req.max_completion_tokens == ev['fit']['measurement']['response_reserve_tokens'] == 12_345


@pytest.mark.parametrize('host_world', [('max', 1_000_000, 'estimator')], indirect=True)
def test_running_low_trial_prices_the_catalog_it_will_send(host_world, monkeypatch):
    from ouroboros.llm_attempt import main_context_fit_refusal

    host = host_world.host
    assert host._schemas_mode == host.plan.initial_mode == 'max'
    host._session_opened = True
    host.plan = replace(host.plan, output_reserve_tokens=10_000)
    host.schemas.append({'type': 'function', 'function': {'name': 'fixture_no_longer_in_registry',
        'description': 'An obsolete declaration from the serving catalog. ' * 12_000}})
    host.messages.append({'role': 'user', 'content': 'Keep the newest owner words.'})
    factory, options = host_world.provider['session'], LiveOptions(resume_handle='synthetic-resume')
    choose = host._tool_schemas_for_mode
    low_schemas = choose('low')
    low = replace(host.plan, initial_mode='low')
    rows = low.reproject_transcript(host.messages, 'low')
    _, admitted, _, _ = measure_session_wire(host, low, rows, low_schemas, factory, options)
    _, stale, _, _ = measure_session_wire(host, low, rows, host.schemas, factory, options)
    window = admitted.prompt_tokens_estimate + 45_000
    assert admitted.prompt_tokens_estimate + 10_000 < window < stale.prompt_tokens_estimate + 10_000
    host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=window))
    trials = []

    def select(mode):
        trials.append(mode)
        return choose(mode)

    monkeypatch.setattr(host, '_tool_schemas_for_mode', select)
    session, request, facts = session_candidate(host, factory, options)
    assert main_context_fit_refusal(request) is None
    assert session.selected.mode == host.plan.initial_mode == host._schemas_mode == 'low'
    assert trials.count('low') == 1
    assert json.loads(session.selected.tools_json) == low_schemas == host.schemas
    assert facts['tool_count'] == len(low_schemas)
    assert any(row['content'] == 'Keep the newest owner words.' for row in json.loads(session.selected.messages_json))
