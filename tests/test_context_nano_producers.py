"""Producer forms preserve meaning and source custody on both sides of shortage."""
import copy
import json

import pytest

from ouroboros import context_sections, memory_floor
from ouroboros.section_floor import HostSection
from tests import _nano_context_fixture as shape
from tests.test_section_floor import _core, _plan, _text, _host_facts


@pytest.mark.parametrize('mode', ['max', 'low', 'nano'])
def test_producer_forms_only_enter_under_shortage_and_keep_exact_sources(tmp_path, monkeypatch, mode):
    from ouroboros.artifacts import read_actor_source_bytes
    sections = shape.host_sections(tmp_path, monkeypatch)
    assert all(isinstance(part, HostSection) for part in sections)
    assert {part.section_id for part in sections} <= set(memory_floor.HOST_STEPS)
    core = _core(tmp_path, sections)
    roomy = _plan(core, 2_000_000, mode)
    assert not _host_facts(roomy, mode)
    assert all(part.full_text in _text(roomy, mode) for part in sections)
    tight = roomy.reproject_for_route(window_tokens=8193, known_window=True, ratio=1,
                                     output_reserve=8192, tool_schemas=[], start_mode=mode)
    receipts = _host_facts(tight)
    assert {r['section_id'] for r in receipts} == {p.section_id for p in sections}
    for part in sections:
        receipt = next(r for r in receipts if r['section_id'] == part.section_id)
        assert part.operative_view in _text(tight)
        assert receipt['reader'] in _text(tight)
        assert read_actor_source_bytes(tmp_path, 'task-1', receipt['source_ref']).decode() == part.full_text


def test_capture_failure_keeps_producer_full(tmp_path, monkeypatch):
    parts = shape.host_sections(tmp_path, monkeypatch)
    def unavailable(*_a, **_kw):
        raise OSError('fixture retention unavailable')
    roomy = _plan(_core(tmp_path, parts), 2_000_000)
    monkeypatch.setattr(memory_floor, 'retain_section', unavailable)
    plan = roomy.reproject_for_route(window_tokens=8193, known_window=True, ratio=1,
                                    output_reserve=8192, tool_schemas=[])
    assert not _host_facts(plan)
    assert all(part.full_text in _text(plan) for part in parts)


def test_pool_and_continuity_keep_current_facts_once_with_history_by_address(tmp_path, monkeypatch):
    parts = {p.section_id: p for p in shape.host_sections(tmp_path, monkeypatch)}
    pool = parts['review_pool']
    assert 'enforcement=advisory; mode=advanced' in pool.operative_view
    assert all(row['subagent_id'] in pool.operative_view for row in shape.pool()['pool'])
    continuity = parts['review_continuity']
    for debt in shape.review_state(tmp_path).commit_readiness_debts:
        assert continuity.operative_view.count(debt.summary) == 1
    assert continuity.operative_view.count('2026-10-23T12:00:00Z') == 1
    assert 'v7.4.0' in continuity.full_text and 'v7.4.0' not in continuity.operative_view
    assert 'review_status()' in continuity.operative_view
    from ouroboros.subagent_runtime import review_pool_host_section
    error = review_pool_host_section({**shape.pool(), 'source': 'error', 'error': 'Malformed catalog',
                                     'pool_without_credentials': ['opus-0']})
    assert 'Malformed catalog' in error.operative_view and 'pool_without_credentials' in error.operative_view


def test_context_capture_preserves_continuity_producer_metadata(tmp_path, monkeypatch):
    from ouroboros import context
    from ouroboros.memory import Memory
    from tests.test_book_context_capture import _capture
    env, _captured, task = _capture(tmp_path)
    continuity = next(p for p in shape.host_sections(env.drive_root, monkeypatch) if p.section_id == 'review_continuity')
    captured = context._capture_context_core(env, Memory(drive_root=env.drive_root, repo_dir=env.repo_dir),
                                             task, lambda: continuity, None)
    assert continuity in captured.dynamic_sections
    actual = next(p for p in captured.dynamic_sections if p.section_id == 'review_continuity')
    assert actual.operative_view == continuity.operative_view and actual.full_text == continuity.full_text


def test_runtime_shortens_declared_history_but_preserves_authority_unknowns_and_vitals():
    source = shape.runtime_data()
    before = copy.deepcopy(source)
    part = context_sections.runtime_host_section(source)
    projected = json.loads(part.operative_view.split('\n\n', 1)[1])
    assert source == before and json.loads(part.full_text.split('\n\n', 1)[1]) == source
    for key in ('task_authority_origin', 'roots', 'budget', 'unknown_observation', 'vitals'):
        assert projected[key] == source[key]
    assert projected['live_works_present']['works'] == source['live_works_present']['works']
    foreign = projected['live_works_present']['other_live_work']
    assert foreign['count'] == 12 and foreign['counted_root_statuses'] == {'failed': 10, 'running': 2}
    for n in range(2):
        assert foreign['unabridged'][n]['tails'] == source['live_works_present']['other_live_work'][n]['tails']
    qualification = projected['official_update']['letter']['qualification']
    assert qualification['commits']['count'] == 2
    assert qualification['commits']['checks'][0]['source'] == 'author_declared'
    assert qualification['commits']['unabridged'][0]['receipt']['review']['observed'] == 'unknown'
    assert qualification['github']['runs']['completed'] == {'failure': 2}
    assert qualification['github']['missing_shas'] == ['commit-absent']
    # A producer has not declared a meaning for new fields: no generic JSON clipping.
    source['live_works_present']['other_live_work'][0]['future_authority'] = ['whole']
    source['official_update']['letter']['qualification']['commits'][0]['future_evidence'] = 'whole'
    new = json.loads(context_sections.runtime_host_section(source).operative_view.split('\n\n', 1)[1])
    assert new['live_works_present']['other_live_work']['unabridged'][0] == source['live_works_present']['other_live_work'][0]
    assert new['official_update']['letter']['qualification']['commits']['unabridged'][0] == source['official_update']['letter']['qualification']['commits'][0]


def test_roster_advice_stays_whole_and_skill_legend_factors_exact_repeats(tmp_path, monkeypatch):
    from ouroboros.subagent_runtime import subagent_catalog_section
    catalog = shape.catalog()
    before = copy.deepcopy(catalog)
    part = subagent_catalog_section(catalog)
    assert catalog == before and json.loads(part.full_text.split('\n\n', 1)[1]) == catalog
    # Roster advice is written for routing decisions: repeated paragraphs stay whole in the operative form too.
    for row in catalog['rows']:
        same = sum(1 for other in catalog['rows'] if other['recommended_use'] == row['recommended_use'])
        assert part.operative_view.count('recommended_use: ' + row['recommended_use']) == same
    assert 'same advice as' not in part.operative_view
    skills = shape.skills_section(tmp_path, monkeypatch)
    for text in (skills.full_text, skills.operative_view):
        assert text.count('Current payload accepted by author under Advisory; reviewer evidence remains at its original hash.') == 1
        assert text.count('  author_accepted=true') == 8


def test_runtime_unrecognized_shapes_remain_whole():
    from ouroboros.live_works import compact_other_live_work
    from ouroboros.update_qualification import compact_qualification
    foreign = {'work_id': 'future', 'root_status': 'unknown', 'tails': {'new': 'format'}}
    assert compact_other_live_work([foreign])['unabridged'] == [foreign]
    unidentified = {'root_status': 'failed', 'tails': [{'state': 'unknown'}]}
    assert compact_other_live_work([unidentified])['unabridged'] == [unidentified]
    commit = {'sha': 'future', 'receipt': {'schema': 1, 'review': [{'unknown': 'shape'}]}}
    assert compact_qualification({'commits': [commit]})['commits']['unabridged'] == [commit]


@pytest.mark.parametrize('look, expected', [(None, False), ('2026-10-23T11:00:00Z', True),
                                           ('2026-10-23T13:00:00Z', False)])
def test_compact_continuity_uses_current_preflight_freshness(tmp_path, monkeypatch, look, expected):
    from ouroboros.agent_task_pipeline import build_review_context
    from types import SimpleNamespace
    state = shape.review_state(tmp_path)
    monkeypatch.setattr('ouroboros.review_state.load_state', lambda *_: state)
    monkeypatch.setattr('ouroboros.review_ledger.latest_preflight_record',
                        lambda *_a, **_kw: {'ts': look} if look else None)
    part = build_review_context(SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path))
    assert ('stale_marker=' in part.full_text) == expected
    assert ('stale_marker=' in part.operative_view) == expected
    fallback = context_sections.review_status_host_section(state, tmp_path, tmp_path)
    assert ('stale_marker=' in fallback.operative_view) == expected
    assert 'v7.4.0' in fallback.full_text  # historical attempts remain readable
