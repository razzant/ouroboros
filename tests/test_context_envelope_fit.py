"""Measured framing and ordinary note shortage share the unchanged window arithmetic."""
import json
import math
from types import SimpleNamespace

import pytest

from ouroboros import context_fit as cf, memory_view as mv
from tests import _memory_view_synthetic as syn


@pytest.mark.parametrize(("ratio", "held", "step"), [(1.0, 'note', 'F8'), (1.5, 'owner_row', 'F7')])
def test_measured_envelope_prices_physical_pressure_without_changing_window(monkeypatch, held, step, ratio):
    row = {**syn.spoken(0, 'human', 8000), 'owner_side': True, 'carried': False}
    room = syn.room(lane1=[row] if held == 'owner_row' else [], notes=1 if held == 'note' else 0)
    if room['notes']:
        room['notes'][0]['text'] = 'Held note label. ' + 'n' * 8000
    snap = syn.snapshot(room_facts=room, story=[], conversations=[])
    core = cf.ContextCore('Core', 'BIBLE', '', '', 'Identity', 'Facts', '"Assignment"', False,
                          memory_view_json=mv.snapshot_json(snap))
    monkeypatch.setattr(cf, '_route_calibration_ratio', lambda *a, **kw: ratio)
    plan = cf.build_context_fit_plan(None, core, {'id': 'envelope', '_is_direct_chat': True},
        preferred_mode='max', tool_schemas=[], route_resolver=lambda *a, **kw: (
            {'provider': 'fixture', 'model': 'fixture'},
            SimpleNamespace(route_fp='', status='confirmed', stale=False, window_tokens=1000000)))
    messages = plan.messages_for(plan.initial_mode)
    raw = cf.estimate_context_prompt_tokens(messages, [])
    reserve = 1000
    window = math.ceil(raw * ratio) + reserve + 3000
    measured = math.ceil((window - reserve + 500) / ratio)
    args = dict(window_tokens=window, known_window=True, ratio=ratio, output_reserve=reserve,
                tool_schemas=[], start_mode=plan.initial_mode, current_messages=messages, refused_tokens=501)
    body = room['notes'][0]['text'] if held == 'note' else row['line']
    ample = plan.reproject_for_route(**{**args, 'refused_tokens': 0})
    assert body in ample.projection(ample.initial_mode).system_content_json
    for witness in (None, raw):
        requested = plan.reproject_for_route(**args, measured_input_tokens=witness)
        shown = requested.projection(requested.initial_mode)
        assert (step in shown.memory_facts['floor']['steps']) is (held == 'note')
        assert (body in shown.system_content_json) is (held == 'owner_row')

    fitted = plan.reproject_for_route(**args, measured_input_tokens=measured)
    shown = fitted.projection(fitted.initial_mode)
    assert shown.memory_facts['floor']['steps'][step] == 1
    assert body not in shown.system_content_json
    address = room['notes'][0]['id'] if held == 'note' else row['address']
    assert address in shown.system_content_json
    estimate = cf.estimate_context_prompt_tokens(fitted.messages_for(fitted.initial_mode), [])
    margin = window - reserve - math.ceil((estimate + measured - raw) * ratio)
    assert margin >= 0, dict(margin=margin, input=estimate, framing=measured - raw, ratio=ratio)
    assert fitted.window_tokens == window and fitted.output_reserve_tokens == reserve
    assert fitted.core.memory_view_json == core.memory_view_json
    assert body in json.dumps(json.loads(core.memory_view_json), ensure_ascii=False)


@pytest.mark.parametrize("effort", ["", "high"])
def test_a_route_reprojection_counts_the_shown_request_as_its_round_does(monkeypatch, effort):
    """A rebind measures the shown request as the round does (``measure_main_fit``): its messages and schemas in one
    count, in the shape this provider and effort send (direct OpenAI reasoning: a custom catalog and calls). A carried
    witness of that count leaves no envelope difference; a larger one leaves exactly its own. 15.11.2026: schemas
    priced apart, with separator spaces, and the leading system row's host-only block count read as envelope."""
    tools = [{'type': 'function', 'function': {'name': f'tool_{i}', 'description': 'Do one small fixture thing.',
        'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': 'A path.'}},
                       'required': ['path']}}} for i in range(60)]  # small enough for a custom catalog to carry
    snap = syn.snapshot(room_facts=syn.room('1'), story=[], conversations=[])
    core = cf.ContextCore('Core', 'BIBLE', '', '', 'Identity', 'Facts', '"Assignment"', False,
                          memory_view_json=mv.snapshot_json(snap))
    monkeypatch.setattr(cf, '_route_calibration_ratio', lambda *a, **kw: 1.0)
    plan = cf.build_context_fit_plan(None, core, {'id': 'envelope', '_is_direct_chat': True}, preferred_mode='max',
        tool_schemas=tools, reasoning_effort=effort, route_resolver=lambda *a, **kw: (
            {'provider': 'openai', 'model': 'fixture'},
            SimpleNamespace(route_fp='', status='confirmed', stale=False, window_tokens=1000000)))
    mode, shape = plan.initial_mode, {'provider': 'openai', 'reasoning_effort': effort}
    call = {'id': 'call_1', 'type': 'function', 'function': {'name': 'tool_1', 'arguments': '{"path": "x"}'}}
    messages = [*plan.messages_for(mode), {'role': 'assistant', 'content': '', 'tool_calls': [call]},
                {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'fixture text'}]
    carried = cf.estimate_context_prompt_tokens(plan.reproject_transcript(messages, mode), tools, **shape)
    seen, walk = [], cf._view_projections
    monkeypatch.setattr(cf, '_view_projections', lambda *a, **kw: seen.append(kw) or walk(*a, **kw))
    for witness in (carried, carried + 37):
        plan.reproject_for_route(window_tokens=1000000, known_window=True, ratio=1.0, output_reserve=1000,
                                 tool_schemas=tools, start_mode=mode, current_messages=messages, select_mode=mode,
                                 measured_input_tokens=witness, reasoning_effort=effort)
    assert [kw['envelope_overhead_tokens'] for kw in seen] == [0, 37]
    assert {(kw['provider'], kw['reasoning_effort']) for kw in seen} == {('openai', effort)}
    canonical = cf.estimate_context_prompt_tokens(messages, tools)
    assert (carried > canonical) is bool(effort)  # the reasoning shape is longer, and it is the one measured
    assert plan.projected_tokens_with_tools(mode, tools, **shape) == cf.estimate_context_prompt_tokens(
        plan.messages_for(mode), tools, **shape)
