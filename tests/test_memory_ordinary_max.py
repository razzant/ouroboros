"""Both Max gates admit ordinary tellings, while chosen Nano keeps its own 85K target."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import context_budget as cb, memory_floor as mf
from tests import _memory_view_synthetic as syn, _memory_scale_fixture as scale


def core(root, marked):
    pages = syn.pages(5, 100000)
    if marked:
        for page in pages:
            page['lead_rule'] = 1
    snap = syn.snapshot(story=pages, room_facts=syn.room())
    return replace(scale.core(root, snap), base_prompt='SYSTEM', bible_md='BIBLE', architecture_md='# Architecture\nWhole books.',
                   semi_stable_text='Identity', semi_stable_sections=(), dynamic_text='', dynamic_sections=(),
                   dynamic_head_text='', dynamic_head_sections=())


@pytest.mark.parametrize('marked,mode', [(True, 'max'), (False, 'low')])
@pytest.mark.parametrize('late_gate', [False, True])
def test_both_max_gates_use_ordinary_forms_not_whole_story(tmp_path, monkeypatch, marked, mode, late_gate):
    if late_gate:
        monkeypatch.setattr(mf, 'max_room_lost', lambda _fact: False)
    plan = scale.build(tmp_path, monkeypatch, window=100000, captured=core(tmp_path, marked))
    assert plan.initial_mode == mode
    shown = plan.projection(mode)
    assert shown.calibrated_tokens + 10000 <= 100000
    if marked:
        assert shown.memory_facts['floor']['steps'].get('F5l')
        assert not shown.memory_facts['floor']['forms']['emergency']
    else:
        assert shown.memory_facts['max_room']['need'] > 100000
    if marked and not late_gate:
        for ratio in (0.83, 1.003, 1.17):
            window = 100000
            monkeypatch.setattr(scale.cf, '_route_calibration_ratio', lambda *a, **kw: ratio)
            for _ in range(12):
                evidence = SimpleNamespace(window_tokens=window, status='confirmed', stale=False,
                    route_fp='', source='boundary', response_limit={},
                    route_facts={'working_reserve_tokens': 10000, 'basis': 'provider'})
                tight = scale.cf.build_context_fit_plan(SimpleNamespace(drive_root=tmp_path), plan.core,
                    {'id': 'boundary', 'chat_id': 1, '_is_direct_chat': True}, preferred_mode='max',
                    tool_schemas=scale.tools(), route_resolver=lambda *a, **kw: ({'provider': 'fixture', 'model': 'fixture'}, evidence))
                facts = tight.projection(tight.initial_mode).memory_facts
                need = facts['max_room']['need']
                if need == window:
                    break
                window = need
            assert need == facts['max_room']['window'] == window and tight.initial_mode == 'max'
            assert not facts['floor']['ordinary_unfit'] and not facts['floor']['forms']['emergency']


def test_owner_selected_nano_keeps_85000_target_and_requested_work_room(tmp_path, monkeypatch):
    plan = scale.build(tmp_path, monkeypatch, window=200000, mode='nano', captured=core(tmp_path, True))
    shown = plan.projection('nano')
    floor = shown.memory_facts['floor']
    assert plan.initial_mode == 'nano' and floor['target_tokens'] == 85000
    reserve = cb.context_mode_limits('nano', 'nano', plan.output_reserve_tokens,
                                     working_reserve_tokens=plan.working_reserve_tokens)[1]
    assert shown.calibrated_tokens + reserve + 10625 <= 85000
    assert floor['budget_allowance_tokens'] is not None


def test_a_form_version_change_replaces_kept_bytes_even_when_counts_match(tmp_path, monkeypatch):
    from copy import deepcopy
    plan = scale.build(tmp_path, monkeypatch, window=1000000, captured=core(tmp_path, True))
    old = plan.max_projection
    facts = deepcopy(old.memory_facts)
    facts['floor']['forms_version'] = 1
    old = replace(old, memory_facts=facts,
                  system_content_json=old.system_content_json.replace('## My story', '## Old form wording'))
    restored = replace(plan, max_projection=old)
    fresh = restored.reproject_for_route(window_tokens=1000000, known_window=True, ratio=1,
        output_reserve=10000, tool_schemas=scale.tools(), start_mode='max', select_mode='max')
    assert fresh.max_projection.memory_facts['floor']['steps'] == facts['floor']['steps']
    assert fresh.max_projection.memory_facts['floor']['forms_version'] == mf.FORMS_VERSION
    assert 'Old form wording' not in fresh.max_projection.system_content_json
