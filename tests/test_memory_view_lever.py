"""Mind-selected memory forms share the authored boundary and preserve concurrent arrivals."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from ouroboros import memory_view as mv, memory_floor as floor
from ouroboros.context_fit import ContextCore, ContextFitPlan, ContextFitProjection, _system_blocks, _governance_blocks
from ouroboros.tools.compact_context import record_context_view, _compact_context
from ouroboros.loop_round_limits import _run_round_compaction
from tests._memory_view_synthetic import snapshot, room, spoken, pages
from tests.test_context_view_tool import _context


def plan():
    words = [spoken(0, "human", 4000), spoken(1, "ouroboros", 4000), spoken(2, "human", 4000)]
    words[2]["immune"] = True
    snap = snapshot(room_facts=room(lane1=words), story=pages(2, 8000))
    core = ContextCore("Core", "BIBLE", "", "", "Identity", "Facts", '"Assignment"', False,
                       memory_view_json=mv.snapshot_json(snap))
    facts = floor.view_facts(snap, mv.FULL_VIEW, window_tokens=200000, mode="max", allowances={})
    content = _system_blocks(core, _governance_blocks(None, core, mode="max"), mv.render_story(snap), mv.render_room(snap),
                             mode="max")
    projection = ContextFitProjection("max", json.dumps(content), 10000, 10000, 1.0, memory_facts=facts)
    return ContextFitPlan("capture", "max", "max", "test", "test", "route", "confirmed", False, 200000,
                          65536, '"Assignment"', projection, replace(projection, mode="low"), core=core), snap


def note_plan():
    from tests.test_memory_view_floor import note_snapshot
    p, snap = plan()
    snap = replace(snap, room={**snap.room, 'notes': note_snapshot().room['notes']})
    p = replace(p, core=replace(p.core, memory_view_json=mv.snapshot_json(snap)))
    return floor.select_view(p, 'max', []), snap


def test_note_selection_and_restoration_survive_the_next_projection():
    p, snap = note_plan()
    note = snap.room['notes'][0]
    address = 'F8:' + note['id']
    inventory = next(row for row in floor.view_inventory(p, 'max') if row['view_address'] == address)
    assert inventory['saving_tokens'] > 800 and inventory['form'] == 'whole'
    selected = floor.select_view(p, 'max', [address])
    assert mv._note_pointer(note) in selected.max_projection.system_content_json
    for restore in (False, True):
        if restore:
            selected = floor.select_view(selected, 'max', [inventory['restore_view_address']])
        rebound = selected.reproject_for_route(window_tokens=1000000, known_window=True, ratio=1.0,
            output_reserve=65536, tool_schemas=[], start_mode='max')
        level = floor.kept_level(snap, rebound.max_projection.memory_facts)
        assert (note['id'] in dict(level.addressed).get('F8', ())) is (not restore)
        assert (mv._note_lines([note])[0] in json.loads(rebound.max_projection.system_content_json)[-1]['text']) is restore
    aggregate = floor.select_view(p, 'max', [*('F8:' + n['id'] for n in snap.room['notes']), 'F8a:1'])
    assert next(row for row in floor.view_inventory(aggregate, 'max') if row['view_address'] == address)['form'] == 'aggregate'
    label = floor.select_view(aggregate, 'max', [address])
    assert 'F8a' not in label.max_projection.memory_facts['floor']['steps']
    assert mv._note_pointer(note) in label.max_projection.system_content_json
    whole = floor.select_view(aggregate, 'max', [inventory['restore_view_address']])
    assert whole.max_projection.memory_facts['floor']['steps']['F8'] == 5
    assert '5 notes not yet sealed' in whole.max_projection.system_content_json


@pytest.mark.parametrize('accept_restore', [False, True])
def test_note_restoration_honours_explicit_fit_callback_verdict(tmp_path, monkeypatch, accept_restore):
    """An explicit refusal blocks publication; Main's advisory estimate need not refuse."""
    p, snap = note_plan()
    note = snap.room['notes'][0]
    ctx = SimpleNamespace(context_fit_plan=p, active_context_mode='max')
    messages, schemas = p.messages_for('max'), []
    record_context_view(ctx, messages, schemas)
    assert 'requested' in _compact_context(ctx, view_address=['F8:' + note['id']])
    messages = _run_round_compaction(messages, _context(tmp_path, ctx, schemas, monkeypatch))
    assert ctx._context_view_receipt['status'] == 'applied'
    assert ctx.context_fit_plan.max_projection.memory_facts['floor']['steps']['F8'] == 1
    record_context_view(ctx, messages, schemas)
    shown = json.loads(_compact_context(ctx, inspect=True))
    row = next(row for row in shown['units'] if row.get('view_address') == 'F8:' + note['id'])
    assert row['form'] == 'line' and row['selected']
    assert 'requested' in _compact_context(ctx, view_address=[row['restore_view_address']])
    result = _run_round_compaction(messages, _context(tmp_path, ctx, schemas, monkeypatch,
        lambda m, t: {'accepted': accept_restore, 'input_tokens': floor.view_tokens(json.dumps(m))}))
    assert ctx._context_view_receipt['status'] == ('applied' if accept_restore else 'fit_rejected')
    assert ('F8' in ctx.context_fit_plan.max_projection.memory_facts['floor']['steps']) is (not accept_restore)
    if not accept_restore:
        assert result == messages


def test_aggregate_lever_preserves_element_addresses_and_requires_uncovered_note():
    p, snap = plan()
    owner = 'F7:' + snap.room['lane1'][0]['address']
    held = floor.level_with(snap, mv.FULL_VIEW, [owner], owner_note=True)
    with pytest.raises(ValueError, match='working_note'):
        floor.level_with(snap, held, ['F7a:1'])
    grouped = floor.level_with(snap, held, ['F7a:1'], owner_note=True)
    assert dict(grouped.addressed)['F7'] == dict(held.addressed)['F7']
    first = floor.select_view(p, 'max', ['F5:page-000', 'F5a:story'])
    second = floor.select_view(first, 'max', ['F5:page-001'])
    assert dict(floor.kept_level(snap, second.max_projection.memory_facts).addressed)['F5'] == ('page-000', 'page-001')
    assert '2 older pages and parts' in second.max_projection.system_content_json
    with pytest.raises(ValueError, match='already addressed'):
        floor.level_with(snap, mv.FULL_VIEW, ['F7a:1'])


def test_compact_range_form_keeps_an_authored_capsule_for_uncovered_owner_words(tmp_path, monkeypatch):
    p, snap = plan()
    p = floor.select_view(p, 'max', ['F7:' + snap.room['lane1'][0]['address']], owner_note=True)
    ctx = SimpleNamespace(context_fit_plan=p, active_context_mode='max')
    messages, schemas = p.messages_for('max'), []
    record_context_view(ctx, messages, schemas)
    assert 'working_note' in _compact_context(ctx, view_address=['F7a:1'])
    assert 'requested' in _compact_context(ctx, view_address=['F7a:1'], working_note='Keep the owner commitment.')
    result = _run_round_compaction(messages, _context(tmp_path, ctx, schemas, monkeypatch))
    assert ctx._context_view_receipt['status'] == 'applied'
    assert 'Keep the owner commitment.' in json.dumps(result)
    assert dict(floor.kept_level(snap, ctx.context_fit_plan.max_projection.memory_facts).addressed)['F7a'] == ('1',)
    sources = list(tmp_path.rglob('*.json'))
    assert any(snap.room['lane1'][0]['line'] in path.read_text() for path in sources)


def test_a_view_selection_keeps_its_mode_s_connections_form():
    p, _snap = plan()
    p = replace(p, core=replace(p.core, connections=("FULL MAP", "NAMES MAP")),
                nano_projection=replace(p.max_projection, mode="nano"))
    for mode, form, other in (("nano", "NAMES MAP", "FULL MAP"), ("max", "FULL MAP", "NAMES MAP")):
        text = floor.select_view(p, mode, ["F5:page-000"]).projection(mode).system_content_json
        assert form in text and other not in text, mode


def test_addresses_refuse_core_and_governing_words_and_keep_prior_choices():
    p, snap = plan()
    first = floor.select_view(p, "max", ["F5:page-000"])
    second = floor.select_view(first, "max", ["F5:page-001"])
    assert "Page 0." in p.projection("max").system_content_json
    assert "Page 0." not in second.projection("max").system_content_json
    assert dict(floor.kept_level(snap, second.projection("max").memory_facts).addressed)["F5"] == ("page-000", "page-001")
    for address in ("core:BIBLE", "F7:" + snap.room["lane1"][-1]["address"]):
        with pytest.raises(ValueError, match="protected"):
            floor.select_view(p, "max", [address])
    assert p.core.memory_view_json == second.core.memory_view_json


@pytest.mark.parametrize("accept", [True, False])
def test_boundary_measures_before_publish_and_keeps_arrivals(tmp_path, monkeypatch, accept):
    p, snap = plan()
    ctx = SimpleNamespace(context_fit_plan=p, active_context_mode="max")
    messages, schemas = p.messages_for("max"), []
    record_context_view(ctx, messages, schemas)
    seen = json.loads(_compact_context(ctx, inspect=True))
    assert any(row.get("view_address") == "F5:page-000" for row in seen["units"])
    assert "re-read" in seen["memory_re_read_price"] or "re-reading" in seen["memory_re_read_price"]
    assert "protected" in _compact_context(ctx, view_address=["core:BIBLE"])
    assert "requested" in _compact_context(ctx, expected_view_revision=seen["view_revision"], view_address=["F5:page-000"])
    tail = [{"role": "user", "content": "A new exact owner instruction"},
            {"role": "assistant", "tool_calls": [{"id": "late", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "late", "content": "Newly chosen exact evidence"}]
    messages += tail
    result = _run_round_compaction(messages, _context(tmp_path, ctx, schemas, monkeypatch,
        lambda m, t: {"accepted": accept, "input_tokens": floor.view_tokens(json.dumps(m))}))
    assert all(row in result for row in tail)
    assert ctx._context_view_receipt["status"] == ("applied" if accept else "fit_rejected")
    if accept:
        assert "Page 0." not in json.dumps(result[0]) and "Page 1." in json.dumps(result[0])
        assert ctx._context_view_receipt["fit_after"]["input_tokens"] > 0
        assert ctx._context_view_receipt["memory_view"]["re_read_input_tokens_estimate"] > 0
    else:
        assert result == messages and ctx.context_fit_plan is p


@pytest.mark.parametrize("with_memory", [False, True])
def test_observed_memory_and_queued_selection_survive_a_json_continuation(tmp_path, monkeypatch, with_memory):
    from ouroboros.owner_wait import continuation_state, restore_continuation_state

    p, _snap = plan()
    ctx = SimpleNamespace(task_id="memory-wait", drive_root=tmp_path, budget_drive_root=tmp_path,
                          context_fit_plan=p if with_memory else None, active_context_mode="max")
    messages, schemas = p.messages_for("max"), []
    record_context_view(ctx, messages, schemas)
    seen = json.loads(_compact_context(ctx, inspect=True))
    assert (seen["memory_view_status"] == "observed_snapshot") is with_memory
    if with_memory:
        assert "requested" in _compact_context(ctx, view_address=["F5:page-000"])
    state = json.loads(json.dumps(continuation_state(ctx, messages, {}, {}, 1, schemas, set())))
    resumed = SimpleNamespace(task_id=ctx.task_id, drive_root=tmp_path, budget_drive_root=tmp_path,
                              context_fit_plan=ctx.context_fit_plan)
    restored = []
    restore_continuation_state(SimpleNamespace(_ctx=resumed), state, restored, {}, {}, set())
    assert restored == messages and resumed._last_context_observation == state["context_observations"]["_last_context_observation"]
    if with_memory:
        result = _run_round_compaction(restored, _context(tmp_path, resumed, schemas, monkeypatch))
        assert resumed._context_view_receipt["status"] == "applied"
        assert "Page 0." not in json.dumps(result[0]) and "Page 1." in json.dumps(result[0])
    else:
        assert not getattr(resumed, "_pending_compaction", None)
        assert json.loads(_compact_context(resumed, inspect=True))["inventory"]["by_kind"]["memory"]["units"] == 0


@pytest.mark.parametrize("changed_snapshot", [False, True])
def test_inspection_keeps_observed_forms_and_boundary_requires_the_same_source(tmp_path, monkeypatch, changed_snapshot):
    p, _snap = plan()
    ctx = SimpleNamespace(context_fit_plan=p, active_context_mode="max")
    messages, schemas = p.messages_for("max"), []
    record_context_view(ctx, messages, schemas)
    ctx.context_fit_plan = floor.select_view(p, "max", ["F5:page-001"])
    shown = json.loads(_compact_context(ctx, inspect=True))
    assert not next(row for row in shown["units"] if row.get("view_address") == "F5:page-001")["selected"]
    assert "requested" in _compact_context(ctx, view_address=["F5:page-000"])
    if changed_snapshot:
        ctx.context_fit_plan = replace(ctx.context_fit_plan, core_sha256="different-source")
    result = _run_round_compaction(messages, _context(tmp_path, ctx, schemas, monkeypatch))
    assert ctx._context_view_receipt["status"] == ("selection_failed" if changed_snapshot else "applied")
    if changed_snapshot:
        assert result == messages and "snapshot changed" in ctx._context_view_receipt["reason"]
    else:
        assert "Page 0." not in json.dumps(result[0]) and "Page 1." not in json.dumps(result[0])
