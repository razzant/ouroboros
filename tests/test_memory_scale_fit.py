"""A large room must fit through the ordinary ladder, with measured margins."""
import pytest
from ouroboros import context_fit as cf, memory_floor as mf, memory_view as mv
from ouroboros.tool_policy import select_tool_schemas
from tests import _memory_scale_fixture as scale


def test_owner_max_remaining_work_room_deficit_is_not_hidden(tmp_path, monkeypatch):
    # The new lists fit physically. The existing Low start-work-room rule still
    # refuses; changing that policy is an owner decision, not another memory cut.
    with pytest.raises(cf.MinimumViewUnfit) as raised:
        scale.build(tmp_path, monkeypatch, reply_limit=16384)
    error = raised.value
    assert not error.physical and error.deficit_tokens == 0
    assert error.work_room_deficit_tokens == 1355
    assert error.memory_view_tokens < 10000


@pytest.mark.parametrize('mode', ['max', 'nano'])
def test_owner_shape_admits_92000_without_shortening_protected_text(tmp_path, monkeypatch, mode):
    plan = scale.build(tmp_path, monkeypatch, mode=mode)
    shown = plan.projection(plan.initial_mode)
    input_tokens = shown.calibrated_tokens + cf.tool_schema_tokens(list(select_tool_schemas(
        scale.tools(), context_mode=plan.initial_mode).schemas))
    margin = 92000 - 10000 - input_tokens
    assert margin >= 0, f'input={input_tokens}, reserve=10000, window=92000, margin={margin}'
    assert plan.preferred_mode == mode and plan.initial_mode == 'nano'
    assert 'F7a' in shown.memory_facts['floor']['steps']
    assert not mf.WINDOW_ONLY.intersection(shown.memory_facts['floor']['steps']), f'whole notes, margin={margin}'
    content = shown.system_content_json
    for block, size in scale.PROTECTED.items():
        label = {'system': 'SYSTEM', 'bible': 'BIBLE', 'identity': 'Identity',
                 'shared_understanding': 'Overview'}.get(block, block)
        assert scale.text(label, size) in content, block
    assert all(note['text'] in content for note in scale.snapshot().room['notes'])


def test_same_ordinary_plan_holds_five_resident_tools(tmp_path, monkeypatch):
    plan = scale.build(tmp_path, monkeypatch)
    plan = plan.reproject_for_route(window_tokens=92000, known_window=True, ratio=1.0,
        output_reserve=plan.output_reserve_tokens, tool_schemas=scale.tools(), start_mode=plan.initial_mode)
    tokens = plan.projection('nano').calibrated_tokens + cf.tool_schema_tokens(scale.tools())
    margin = 92000 - 10000 - tokens
    assert margin >= 0, f'5 tools: input={tokens}, reserve=10000, window=92000, margin={margin}'
    print(f'FIVE_TOOLS input={tokens}, reserve=10000, window=92000, margin={margin}')


def test_same_room_admits_128000_text_route(tmp_path, monkeypatch):
    plan = scale.build(tmp_path, monkeypatch, window=128000, reserve=None)
    shown = plan.projection(plan.initial_mode)
    from ouroboros.context_budget import context_mode_limits
    tokens = shown.calibrated_tokens + cf.tool_schema_tokens(list(select_tool_schemas(
        scale.tools(), context_mode=plan.initial_mode).schemas))
    reserve = context_mode_limits(plan.initial_mode, 'max', plan.output_reserve_tokens)[1]
    assert tokens + reserve <= 128000, f'input={tokens}, reserve={reserve}, window=128000'
    print(f'TEXT_ROUTE mode={plan.initial_mode}, input={tokens}, reserve={reserve}, margin={128000 - tokens - reserve}')


def test_floor_size_depends_on_protected_items_not_history_count():
    sizes = []
    for rows, pages in ((150, 100), (1500, 1000)):
        snap = scale.snapshot(rows, pages)
        level = mf.fit_memory_view(snap, {'margin': 0, 'physical': 0, 'budget': None})
        rendered = mv.render_story(snap, level) + mv.render_room(snap, level)
        sizes.append(len(rendered.encode()))
        assert dict(level.steps)['F8'] == 6 and dict(level.steps)['F8a'] == 1
        assert mv._notes_line(snap.room['notes']) in rendered
        assert all(row['line'] in rendered for row in snap.room['lane1'] if row.get('immune'))
    assert abs(sizes[1] / sizes[0] - 1) <= .05, f'floor bytes at 150/100 and 1500/1000: {sizes}'
    print(f'FLOOR_BYTES rows/pages=150/100:{sizes[0]}, 1500/1000:{sizes[1]}')
