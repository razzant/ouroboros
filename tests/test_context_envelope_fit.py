"""Measured framing and ordinary note shortage share the unchanged window arithmetic."""
import json
import math
from types import SimpleNamespace

import pytest

from ouroboros import context_fit as cf, memory_view as mv
from tests import _memory_view_synthetic as syn


@pytest.mark.parametrize('held, step', [('note', 'F8'), ('owner_row', 'F7')])
@pytest.mark.parametrize('ratio', [1.0, 1.5])
def test_measured_envelope_prices_physical_pressure_without_changing_window(monkeypatch, held, step, ratio):
    row = {**syn.spoken(0, 'human', 8000), 'owner_side': True, 'carried': False}
    room = syn.room(lane1=[row] if held == 'owner_row' else [], notes=1 if held == 'note' else 0)
    if room['notes']:
        room['notes'][0]['text'] = 'Held note label. ' + 'n' * 8000
    snap = syn.snapshot(room_facts=room, story=[], live=[])
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
