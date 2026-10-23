"""R13 boundary checks on synthetic captures and stores only."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import memory_floor as mf, memory_view as mv
from tests import _memory_view_synthetic as syn


@pytest.mark.parametrize('shape', ['row', 'page'])
def test_optional_aggregate_cannot_increase_minimum(shape, monkeypatch):
    snap = syn.snapshot(room_facts=syn.room(lane1=[syn.spoken(0, 'ouroboros', 4000)] if shape == 'row' else []),
                        story=syn.pages(1, 8000) if shape == 'page' else ())
    with monkeypatch.context() as prior:
        prior.setattr(mf, 'LADDER', tuple(s for s in mf.LADDER if s not in {'F7a', 'F5a'}))
        without = mf.minimal_view_tokens(snap, window_tokens=92000)
    with_optional = mf.minimal_view_tokens(snap, window_tokens=92000)
    assert with_optional <= without, (shape, without, with_optional)


def test_uneconomic_aggregates_do_not_force_a_fitting_low_view_to_nano(monkeypatch):
    from ouroboros import context_fit as cf
    from tests.test_memory_view_lever import plan
    original, _ = plan()
    protected = {**syn.spoken(1, 'human', 90000), 'owner_side': True, 'immune': True}
    snap = syn.snapshot(room_facts=syn.room(lane1=[syn.spoken(0, 'ouroboros', 4000), protected]),
                        story=syn.pages(1, 8000))
    core = replace(original.core, memory_view_json=mv.snapshot_json(snap))
    monkeypatch.setattr(cf, '_route_calibration_ratio', lambda *a, **kw: 1.0)

    def start(window):
        evidence = SimpleNamespace(window_tokens=window, status='confirmed', stale=False,
            route_fp='', source='synthetic', response_limit={},
            route_facts={'working_reserve_tokens': 10000, 'basis': 'provider'})
        return cf.build_context_fit_plan(None, core, {'id': 'minimum-bound', 'chat_id': 1, '_is_direct_chat': True},
            preferred_mode='low', tool_schemas=[],
            route_resolver=lambda *a, **kw: ({'provider': 'test', 'model': 'test'}, evidence))

    # Locate the physical edge on the same source with only the optional list
    # forms disabled. The last protected owner row cannot be reduced to meet it.
    with monkeypatch.context() as prior:
        prior.setattr(mf, 'LADDER', tuple(s for s in mf.LADDER if s not in {'F7a', 'F5a'}))
        lower, upper = 80000, 100000
        assert start(upper).initial_mode == 'low'
        while upper - lower > 1:
            middle = (lower + upper) // 2
            if start(middle).initial_mode == 'low':
                upper = middle
            else:
                lower = middle
        before = start(upper)
        p = before.projection('low')
        assert p.calibrated_tokens + before.output_reserve_tokens <= upper
    after = start(upper)
    assert after.initial_mode == 'low', (upper, before.initial_mode, after.initial_mode,
        p.calibrated_tokens, before.output_reserve_tokens)


def test_captured_row_range_excludes_later_rows_and_other_rooms(tmp_path):
    from ouroboros.tools import chronicle
    from tests.test_chronicle_tools import addr, chat, ctx_for, ts
    rows = chat(tmp_path, [{'chat_id': 1, 'direction': 'out' if i % 2 == 0 else 'in',
                           'ts': ts(i), 'text': f'Original exact row {i}.'} for i in range(5)])
    items = [dict(pos=i, address=addr(row), ts=row['ts'], kind='ouroboros')
             for i, row in enumerate(rows) if i in (0, 4)]
    text = mv._rows_line({'room_id': '1'}, items)
    chat(tmp_path, [{'chat_id': 1, 'direction': 'in', 'ts': ts(6), 'text': 'NEW AFTER THE CAPTURE'},
                    {'chat_id': 2, 'direction': 'in', 'ts': ts(2), 'text': 'OTHER ROOM'}])
    result = chronicle._memory_read(ctx_for(tmp_path), room_id='1', rows=True,
                                   **{'from': addr(rows[0]), 'to': addr(rows[-1])})
    assert all(row['text'] in result for row in rows)
    assert 'NEW AFTER THE CAPTURE' not in result and 'OTHER ROOM' not in result
    assert mv._rows_line({'room_id': '1'}, items) == text


def test_restoration_uses_the_existing_main_advisory_fit_contract(tmp_path, monkeypatch):
    from ouroboros.loop_model_call import _measure_main_context_view
    from ouroboros.loop_round_limits import _run_round_compaction
    from ouroboros.tools.compact_context import record_context_view, _compact_context
    from tests.test_memory_view_lever import note_plan
    from tests.test_context_view_tool import _context

    original, snap = note_plan()
    address = 'F8:' + snap.room['notes'][0]['id']
    held = mf.select_view(original, 'max', [address])
    base = _measure_main_context_view(held, held.messages_for('max'), [], 'max', 'medium', 'test')
    bound = base['estimated_input_tokens'] + base['response_reserve_tokens'] + 500
    held = replace(held, window_tokens=bound)
    messages = held.messages_for('max')
    ctx = SimpleNamespace(context_fit_plan=held, active_context_mode='max')
    record_context_view(ctx, messages, [])
    assert 'requested' in _compact_context(ctx, view_address=[address + ':whole'])
    # None selects the real default Main measurement, not a stub accepted flag.
    result = _run_round_compaction(messages, _context(tmp_path, ctx, [], monkeypatch, fit=None))
    receipt = ctx._context_view_receipt
    assert receipt['status'] == 'applied'
    assert receipt['fit_after']['capacity_deficit_tokens'] > 0
    assert 'F8' not in ctx.context_fit_plan.max_projection.memory_facts['floor']['steps']
    assert result != messages
