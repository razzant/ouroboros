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


def test_a_selection_declares_whether_the_story_it_shows_is_whole():
    """``select_view`` writes B's declaration from the level it selected (``llm_messages.SHARED_HEAD_KEY``): a folded
    page narrows my story for this work alone, a folded room note leaves the story every work shares. The declared
    head ends exactly where the selected story begins."""
    from ouroboros.llm_messages import SHARED_HEAD_KEY

    p, snap = note_plan()
    for address, whole in (('F5:page-000', False), ('F8:' + snap.room['notes'][0]['id'], True)):
        selected = floor.select_view(p, 'max', [address])
        block = json.loads(selected.max_projection.system_content_json)[-2]
        declared = block[SHARED_HEAD_KEY]
        assert declared['story_whole'] is whole, address
        story = mv.render_story(snap, floor.kept_level(snap, selected.max_projection.memory_facts))
        assert block['text'][declared['chars']:] == story and (story == mv.render_story(snap)) is whole, address


def test_a_selection_keeps_what_the_starts_request_took_with_no_memory():
    """3A: Low's missed target at the start names what the request took with no memory (``floor.no_memory_tokens``),
    the start's fact. A selection the mind makes keeps it, so the mode chooser can warn while later sends still miss
    Low, until a new start measures again (``context_measurement._no_memory_tokens``); a start that met its target adds
    none."""
    p, _snap = plan()
    shown = p.projection("max")
    missed = replace(p, max_projection=replace(shown, memory_facts={
        **shown.memory_facts, "floor": {**shown.memory_facts["floor"], "no_memory_tokens": 315_536}}))
    assert floor.select_view(missed, "max", []).projection("max").memory_facts["floor"]["no_memory_tokens"] == 315_536
    assert "no_memory_tokens" not in floor.select_view(p, "max", []).projection("max").memory_facts["floor"]


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


@pytest.mark.parametrize("saved", ["with_its_write_time", "before_requests_carried_one"])
def test_compact_range_form_keeps_an_authored_capsule_for_uncovered_owner_words(tmp_path, monkeypatch, saved):
    """The commitments' note is dated like every note of that call, by the compact_context call that wrote it; a
    request queued before requests carried that time is dated when it is applied."""
    import datetime
    from ouroboros import utils
    from ouroboros.context_compaction import _capsule_metadata

    written = '2026-11-13T19:40:00+00:00'
    p, snap = plan()
    p = floor.select_view(p, 'max', ['F7:' + snap.room['lane1'][0]['address']], owner_note=True)
    ctx = SimpleNamespace(context_fit_plan=p, active_context_mode='max')
    messages, schemas = p.messages_for('max'), []
    record_context_view(ctx, messages, schemas)
    assert 'working_note' in _compact_context(ctx, view_address=['F7a:1'])
    with monkeypatch.context() as at_call:  # when the mind wrote its note: the compact_context call
        at_call.setattr(utils, 'utc_now_iso', lambda *_args, **_kwargs: written)
        assert 'requested' in _compact_context(ctx, view_address=['F7a:1'], working_note='Keep the owner commitment.')
    if saved == 'before_requests_carried_one':
        del ctx._pending_compaction['written_at']
    result = _run_round_compaction(messages, _context(tmp_path, ctx, schemas, monkeypatch))
    assert ctx._context_view_receipt['status'] == 'applied'
    assert 'Keep the owner commitment.' in json.dumps(result)
    stamps = {meta.get('written_at') for row in result if (meta := _capsule_metadata(row)[1])}
    if saved == 'with_its_write_time':
        assert stamps == {written}  # one call, one write time, the owner commitments' note included
    else:
        assert stamps and all(abs(datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(
            stamp)) < datetime.timedelta(minutes=5) for stamp in stamps)
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


def test_a_selection_over_a_fact_of_other_forms_starts_from_the_whole_view_never_from_its_runs_or_counts():
    """A projection whose floor fact names other forms (a later or an earlier body wrote it) keeps no choice it named:
    the next selection is read anew, so the page it showed by address is whole again and no other page takes its
    place by position or count."""
    p, snap = plan()
    newer = floor.select_view(p, "max", ["F5:page-001"])
    facts = newer.max_projection.memory_facts
    for other in ({"forms_version": floor.FORMS_VERSION + 1}, {"element_order": "another order"}):
        held = replace(newer, max_projection=replace(newer.max_projection, memory_facts={
            **facts, "floor": {**facts["floor"], **other}}))
        again = floor.select_view(held, "max", ["F7:" + snap.room["lane1"][0]["address"]], owner_note=True)
        assert dict(floor.kept_level(snap, again.max_projection.memory_facts).addressed) == {
            "F7": (snap.room["lane1"][0]["address"],)}
        assert "Page 1." in again.max_projection.system_content_json


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


def _ordinary_context(tmp_path, *, low=False, recorded_delta=0):
    from ouroboros.tools.registry import ToolContext

    p, snap = plan()
    ordinary = floor.ordinary_level(snap)
    max_room = {"story": floor.tokens_of(mv.story_chars(snap, ordinary)) + recorded_delta,
                "room": floor.tokens_of(mv.room_chars(snap, ordinary)), "room_id": snap.spec.room_id,
                "levels": snap.spec.work_room_levels, "allowance": 12345, "need": 190000}
    p = replace(p, max_projection=replace(p.max_projection, memory_facts={**p.max_projection.memory_facts, "max_room": max_room}),
                low_projection=replace(p.low_projection, memory_facts={**p.low_projection.memory_facts, "max_room": max_room}))
    if low:
        p = floor.select_view(p, "low", ["F5:page-000"])
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="ordinary-map")
    ctx.context_fit_plan, ctx.active_context_mode = p, "low" if low else "max"
    messages = p.messages_for(ctx.active_context_mode)
    record_context_view(ctx, messages, [], fit_measurement={"accepted": True})
    return ctx, messages, snap


def test_inspect_retains_ordinary_instead_of_selected_low_and_preserves_prompt(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes

    ctx, messages, snapshot = _ordinary_context(tmp_path, low=True)
    p = ctx.context_fit_plan
    before = json.dumps(messages), mv.snapshot_json(snapshot), p.low_projection.system_content_json
    shown = json.loads(_compact_context(ctx, inspect=True))
    summary = shown["ordinary_memory"]
    source = json.loads(read_actor_source_bytes(tmp_path, ctx.task_id, summary["source"]))
    assert summary["status"] == "available" and shown["selected_memory_mode"] == "low"
    assert summary["max_room_comparison"]["status"] == "same_raw_measure"
    assert source["identity"] == summary["identity"]
    assert source["identity"]["room_id"] == snapshot.spec.room_id and source["identity"]["role"] == snapshot.spec.role
    assert source["ordinary_addressed"] == json.loads(json.dumps(floor.ordinary_level(snapshot).addressed))
    selected = floor.kept_level(snapshot, p.low_projection.memory_facts)
    assert source["blocks"]["story"]["json_chars"] > mv.story_chars(snapshot, selected)
    assert summary["estimated_tokens"] == sum(b["estimated_tokens"] for b in source["blocks"].values())
    assert summary["source"]["read"]["tool"] == "read_file"
    assert ctx.context_fit_plan is p
    assert before == (json.dumps(messages), mv.snapshot_json(snapshot), p.low_projection.system_content_json)


def test_retained_map_reads_completely_across_observations_without_transcript_offsets(tmp_path, monkeypatch):
    import hashlib
    from ouroboros import memory_form_facts
    from ouroboros.tools.core_file_tools import _read_file, display_source_segments

    ctx, messages, _snapshot = _ordinary_context(tmp_path)
    first = json.loads(_compact_context(ctx, inspect=True))
    ref = first["ordinary_memory"]["source"]
    args = {**ref["read"]["arguments"], "max_lines": 7}
    def read_page():
        shown = _read_file(ctx, **args)
        view = ctx.last_read_view
        assert view and view["opened_root"] == "artifact_store"
        assert view["complete_sha256"] == ref["sha256"]
        segments = display_source_segments(view, shown)
        assert segments is not None
        args["start_line"] = view["end_line"] + 1
        return "".join(shown[start:end] for start, end, _ in segments)
    chunks = [read_page()]
    pair = [{"role": "assistant", "tool_calls": [{"id": "inspect-one", "function": {
        "name": "compact_context", "arguments": '{"inspect":true}'}}]},
        {"role": "tool", "tool_call_id": "inspect-one", "content": json.dumps(first)}]
    # The second send contains the first tool pair. Reusing the retained map must not render it again.
    monkeypatch.setattr(memory_form_facts, "ordinary_composition", lambda *_: pytest.fail("same snapshot re-rendered"))
    record_context_view(ctx, messages + pair, [], fit_measurement={"accepted": True})
    with monkeypatch.context() as page:
        page.setattr("ouroboros.artifacts.read_actor_source_bytes", lambda *_: pytest.fail("offset read the entire map"))
        second = json.loads(_compact_context(ctx, inspect=True, offset=999999))
    assert second["ordinary_memory"]["status"] == "retained_unverified"
    assert second["view_revision"] != first["view_revision"]
    assert second["units"] == [] and second["ordinary_memory"]["source"] == ref
    while args["start_line"] <= ctx.last_read_view["total_lines"]:
        chunks.append(read_page())
    raw = "".join(chunks).encode("utf-8")
    assert len(raw) == ref["size"] and hashlib.sha256(raw).hexdigest() == ref["sha256"]
    assert json.loads(raw)["estimated_tokens"] == second["ordinary_memory"]["estimated_tokens"]
    # A JSON-restored observation keeps the source through its next send too.
    restored = SimpleNamespace(task_id=ctx.task_id, drive_root=tmp_path, context_fit_plan=ctx.context_fit_plan,
                              active_context_mode="max", _last_context_observation=json.loads(json.dumps(ctx._last_context_observation)))
    record_context_view(restored, messages + pair + [{"role": "user", "content": "Next"}], [], fit_measurement={})
    assert json.loads(_compact_context(restored, inspect=True))["ordinary_memory"]["source"] == ref


@pytest.mark.parametrize("change,reason", [("missing", "retained_snapshot_missing"), ("core", "different_core"),
    ("legacy", "retained_snapshot_missing"), ("forms", "different_forms")])
def test_ordinary_inspection_never_substitutes_a_new_source_for_the_observed_one(tmp_path, monkeypatch, change, reason):
    from ouroboros import memory_form_facts

    ctx, _messages, _snapshot = _ordinary_context(tmp_path)
    first = json.loads(_compact_context(ctx, inspect=True))["ordinary_memory"]
    monkeypatch.setattr(memory_form_facts, "ordinary_composition", lambda *_: pytest.fail("stale source rendered"))
    if change == "missing":
        ctx.context_fit_plan = None
    elif change == "core":
        ctx.context_fit_plan = replace(ctx.context_fit_plan, core_sha256="another-core")
    elif change == "legacy":
        ctx._last_context_observation.pop("memory_core_sha256")
    else:
        ctx._last_context_observation["memory_facts"]["floor"]["forms_version"] += 1
    shown = json.loads(_compact_context(ctx, inspect=True))["ordinary_memory"]
    assert shown["status"] == "unavailable" and shown["reason"] == reason
    which = "previous" if change in ("legacy", "forms") else "retained"
    assert shown[which + "_source"] == first["source"] and shown[which + "_identity"] == first["identity"]
    assert ("retained_source" in shown) == (which == "retained")


@pytest.mark.parametrize("damage", ["absent", "corrupt"])
def test_unreadable_retained_metadata_is_not_silently_regenerated(tmp_path, monkeypatch, damage):
    from ouroboros import artifacts, memory_form_facts

    ctx, _messages, _snapshot = _ordinary_context(tmp_path)
    first = json.loads(_compact_context(ctx, inspect=True))["ordinary_memory"]
    ref = first["source"]
    path = artifacts.task_artifact_dir_path(tmp_path, ctx.task_id) / ref["path"]
    if damage == "absent":
        path.unlink()
    else:
        path.write_bytes(b"!" * ref["size"])  # Same size can still fail the digest.
    monkeypatch.setattr(memory_form_facts, "ordinary_composition", lambda *_: pytest.fail("missing map regenerated"))
    shown = json.loads(_compact_context(ctx, inspect=True))["ordinary_memory"]
    assert shown["status"] == "unavailable" and shown["reason"] == "retained_metadata_source_unreadable"
    assert shown["retained_source"] == first["source"]
    with monkeypatch.context() as page:
        page.setattr(artifacts, "read_actor_source_bytes", lambda *_: pytest.fail("offset read the entire map"))
        unverified = json.loads(_compact_context(ctx, inspect=True, offset=1))["ordinary_memory"]
    assert unverified["status"] == "retained_unverified" and unverified["source"] == ref
    assert unverified["identity"] == first["identity"]


def test_a_newly_observed_snapshot_is_disclosed_and_keeps_the_old_reader(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.tool_call_log import invocation_scope

    ctx, messages, snap = _ordinary_context(tmp_path)
    first = json.loads(_compact_context(ctx, inspect=True))["ordinary_memory"]
    issuer = ctx._last_context_observation
    snap = replace(snap, owner_words="New governing words.")
    ctx.context_fit_plan = replace(ctx.context_fit_plan, core_sha256="new-core",
        core=replace(ctx.context_fit_plan.core, memory_view_json=mv.snapshot_json(snap)))
    record_context_view(ctx, messages + [{"role": "user", "content": "Another observation"}], [], fit_measurement={})
    second = json.loads(_compact_context(ctx, inspect=True))["ordinary_memory"]
    assert second["status"] == "different_source" and second["previous_source_sha256"] == first["source"]["sha256"]
    assert second["source"] != first["source"]
    assert second["max_room_comparison"]["status"] == "different_raw_measure"
    ctx._issued_views = {"queued-in-first": issuer}
    with invocation_scope({"tool_call_id": "queued-in-first"}):
        detached = json.loads(_compact_context(ctx, inspect=True))["ordinary_memory"]
    assert detached["status"] == "unavailable" and detached["reason"] == "different_core"
    assert detached["retained_source"] == first["source"] and detached["retained_identity"] == first["identity"]
    assert ctx._last_context_observation["ordinary_memory"]["source"] == second["source"]
    assert json.loads(read_actor_source_bytes(tmp_path, ctx.task_id, first["source"]))["identity"] == first["identity"]


def test_old_max_room_measure_is_reported_without_adjusting_the_balance(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes

    ctx, _messages, snap = _ordinary_context(tmp_path, recorded_delta=17)
    summary = json.loads(_compact_context(ctx, inspect=True))["ordinary_memory"]
    source = json.loads(read_actor_source_bytes(tmp_path, ctx.task_id, summary["source"]))
    comparison = summary["max_room_comparison"]
    assert comparison["status"] == "different_raw_measure"
    assert comparison["recorded"]["story"] == comparison["measured"]["story"] + 17
    assert source["captured_max_room"]["need"] == 190000
    assert source["blocks"]["story"]["estimated_tokens"] == floor.tokens_of(mv.story_chars(snap, floor.ordinary_level(snap)))


def test_another_max_frame_is_a_different_source_even_when_memory_is_identical(tmp_path):
    ctx, messages, _snapshot = _ordinary_context(tmp_path)
    first = json.loads(_compact_context(ctx, inspect=True))["ordinary_memory"]
    projection = ctx.context_fit_plan.max_projection
    facts = projection.memory_facts
    ctx.context_fit_plan = replace(ctx.context_fit_plan, max_projection=replace(projection,
        memory_facts={**facts, "max_room": {**facts["max_room"], "allowance": 6789}}))
    record_context_view(ctx, messages, [], fit_measurement={})
    second = json.loads(_compact_context(ctx, inspect=True))["ordinary_memory"]
    assert second["status"] == "different_source" and second["source"] != first["source"]
    assert second["identity"]["snapshot_sha256"] == first["identity"]["snapshot_sha256"]
    assert second["previous_identity"] == first["identity"] and second["previous_source"] == first["source"]
    assert second["max_room_comparison"]["allowance"] == 6789
    assert second["blocks"] == first["blocks"]


def test_queued_b_with_only_a_map_cannot_report_it_as_bs_source_after_c(tmp_path):
    from ouroboros.tool_call_log import invocation_scope
    from ouroboros.owner_wait import continuation_state, restore_continuation_state

    ctx, messages, snap = _ordinary_context(tmp_path)
    a = json.loads(_compact_context(ctx, inspect=True))["ordinary_memory"]
    for name in ("B", "C"):
        ctx.context_fit_plan = replace(ctx.context_fit_plan, core_sha256=name,
            core=replace(ctx.context_fit_plan.core, memory_view_json=mv.snapshot_json(replace(snap, owner_words=name))))
        record_context_view(ctx, messages, [], fit_measurement={})
        if name == "B":
            issuer = ctx._last_context_observation
    c = json.loads(_compact_context(ctx, inspect=True))["ordinary_memory"]
    ctx._issued_views = {"queued-B": issuer}
    with invocation_scope({"tool_call_id": "queued-B"}):
        b = json.loads(_compact_context(ctx, inspect=True))["ordinary_memory"]
    assert b["observed_frame"]["core_sha256"] == "B" and b["reason"] == "different_core"
    assert b["previous_source"] == a["source"] and b["previous_identity"] == a["identity"]
    assert "retained_source" not in b and c["previous_identity"] == a["identity"]
    state = json.loads(json.dumps(continuation_state(ctx, messages, {}, {}, 2, [], set())))
    resumed = SimpleNamespace(task_id=ctx.task_id, drive_root=tmp_path, budget_drive_root=tmp_path,
                              context_fit_plan=ctx.context_fit_plan, active_context_mode="max")
    restore_continuation_state(SimpleNamespace(_ctx=resumed), state, [], {}, {}, set())
    after = json.loads(_compact_context(resumed, inspect=True))["ordinary_memory"]
    assert after["status"] == "available" and after["source"] == c["source"] and after["identity"] == c["identity"]


def test_live_exposures_merge_inspects_that_finish_after_the_worker_calculates(tmp_path, monkeypatch):
    import asyncio
    import copy
    from ouroboros.live_context_view import drain_exposure, record_live_exposure

    ctx, messages, snap = _ordinary_context(tmp_path)
    host = SimpleNamespace(ctx=ctx, messages=messages.copy(), schemas=[])
    view = SimpleNamespace(ctx=copy.copy(ctx), _connection_owner=host,
                           _context_session=SimpleNamespace(response_context_rows={"B": [], "C": []}))
    host._connection_view = view
    async def run():
        finished, release = asyncio.Queue(), asyncio.Queue()
        worker = asyncio.to_thread
        async def paused(calculate):
            value = await worker(calculate)
            await finished.put(True)
            await release.get()
            return value
        monkeypatch.setattr(asyncio, "to_thread", paused)
        host.messages.append({"role": "user", "content": "B"})
        record_live_exposure(view, "B")
        await asyncio.wait_for(finished.get(), 5)
        a = json.loads(_compact_context(ctx, inspect=True))["ordinary_memory"]
        ctx.context_fit_plan = replace(ctx.context_fit_plan, core_sha256="C",
            core=replace(ctx.context_fit_plan.core, memory_view_json=mv.snapshot_json(replace(snap, owner_words="C"))))
        host.messages.append({"role": "user", "content": "C"})
        record_live_exposure(view, "C")  # queued before B publishes
        await release.put(True)
        await asyncio.wait_for(finished.get(), 5)
        assert ctx._last_context_observation["messages"][-1]["content"] == "B"
        assert view.ctx._last_context_observation["ordinary_memory"]["source"] == a["source"]
        await release.put(True)
        await drain_exposure(host)
        assert ctx._last_context_observation["messages"][-1]["content"] == "C"
        assert "ordinary_memory" not in ctx._last_context_observation
        assert ctx._last_context_observation["previous_ordinary_memory"]["identity"] == a["identity"]
        c = json.loads(_compact_context(ctx, inspect=True))["ordinary_memory"]
        assert c["status"] == "different_source" and c["previous_source"] == a["source"]
        assert c["identity"]["core_sha256"] == "C" and c["previous_identity"] == a["identity"]
    asyncio.run(run())


@pytest.mark.parametrize("changed", [False, True])
def test_ordinary_source_survives_cold_continuation_after_uninspected_round(tmp_path, monkeypatch, changed):
    from ouroboros import memory_form_facts
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.owner_wait import continuation_state, restore_continuation_state

    ctx, messages, snap = _ordinary_context(tmp_path)
    first = json.loads(_compact_context(ctx, inspect=True))["ordinary_memory"]
    # A subsequent send does not inspect. Its observation, not a private side cache,
    # is the one the actual cold-continuation serializer must preserve.
    record_context_view(ctx, messages + [{"role": "user", "content": "No inspect this round"}], [], fit_measurement={})
    state = json.loads(json.dumps(continuation_state(ctx, messages, {}, {}, 2, [], set())))
    resumed = SimpleNamespace(task_id=ctx.task_id, drive_root=tmp_path, budget_drive_root=tmp_path,
                              context_fit_plan=ctx.context_fit_plan, active_context_mode="max")
    restore_continuation_state(SimpleNamespace(_ctx=resumed), state, [], {}, {}, set())
    if changed:
        resumed.context_fit_plan = replace(resumed.context_fit_plan, core_sha256="new-capture",
            core=replace(resumed.context_fit_plan.core, memory_view_json=mv.snapshot_json(replace(snap, owner_words="New words"))))
    else:
        monkeypatch.setattr(memory_form_facts, "ordinary_composition", lambda *_: pytest.fail("same source regenerated"))
    record_context_view(resumed, messages, [], fit_measurement={})
    second = json.loads(_compact_context(resumed, inspect=True))["ordinary_memory"]
    if changed:
        assert second["status"] == "different_source"
        assert second["previous_source_sha256"] == first["source"]["sha256"]
        assert second["source"] != first["source"]
    else:
        assert second["status"] == "available" and second["source"] == first["source"]
    assert json.loads(read_actor_source_bytes(tmp_path, ctx.task_id, first["source"]))["identity"] == first["identity"]


def test_ordinary_rounds_never_encode_snapshot_and_repeated_inspect_reuses_its_digest(tmp_path, monkeypatch):
    from ouroboros import memory_form_facts

    ctx, messages, _snap = _ordinary_context(tmp_path)
    first = json.loads(_compact_context(ctx, inspect=True))["ordinary_memory"]
    class Captured(str):
        def encode(self, *_args, **_kwargs):
            pytest.fail("unchanged snapshot encoded again")
    ctx.context_fit_plan = replace(ctx.context_fit_plan, core=replace(ctx.context_fit_plan.core,
        memory_view_json=Captured(ctx.context_fit_plan.core.memory_view_json)))
    monkeypatch.setattr(memory_form_facts, "ordinary_composition", lambda *_: pytest.fail("ordinary composition rerun"))
    uninspected = SimpleNamespace(context_fit_plan=ctx.context_fit_plan, active_context_mode="max")
    for n in range(3):
        record_context_view(uninspected, messages, [], fit_measurement={})
        record_context_view(ctx, messages + [{"role": "user", "content": str(n)}], [], fit_measurement={})
    assert json.loads(_compact_context(ctx, inspect=True))["ordinary_memory"]["source"] == first["source"]


@pytest.mark.parametrize("changed", [False, True])
def test_detached_live_issuer_publishes_reader_to_saved_observation_without_rebinding(tmp_path, monkeypatch, changed):
    from ouroboros import memory_form_facts, context_fit as cf, capability_evidence as ce
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.live_tool_lane import record_request
    from ouroboros.owner_wait import continuation_state, store_continuation_source, restore_continuation_state
    from ouroboros.tool_call_log import invocation_scope

    ctx, messages, snap = _ordinary_context(tmp_path)
    issuer = ctx._last_context_observation
    host = SimpleNamespace(ctx=ctx, executed={}, requested={}, messages=list(messages), _context_changed=lambda: None)
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    def rebuild(core):
        return cf.build_context_fit_plan(None, core, {"id": ctx.task_id}, preferred_mode="max",
            route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "test"},
                ce.CapabilityEvidence(1_000_000, "confirmed", "metadata", "issuer-route")))
    ctx.context_fit_plan = rebuild(ctx.context_fit_plan.core)
    messages = ctx.context_fit_plan.messages_for("max")
    record_context_view(ctx, messages, [], fit_measurement={})
    issuer = ctx._last_context_observation
    host.messages = list(messages)
    call = {"id": "queued-inspect", "function": {"name": "compact_context", "arguments": '{"inspect":true}'}}
    record_request(host, call)
    record_context_view(ctx, messages + [{"role": "user", "content": "Exposure B"}], [], fit_measurement={})
    current = ctx._last_context_observation
    assert current is not issuer and host.requested[call["id"]]["view"] is issuer
    # The real effect lane takes exactly this reference out of the queued request.
    ctx._issued_views = {call["id"]: host.requested[call["id"]].pop("view")}
    with invocation_scope({"tool_call_id": call["id"]}):
        shown = json.loads(_compact_context(ctx, inspect=True))
    ctx._issued_views = {}
    first = shown["ordinary_memory"]
    assert shown["view_revision"] == issuer["revision"] != current["revision"]
    assert ctx._last_context_observation is current and current["ordinary_memory"]["source"] == first["source"]
    if changed:
        core = replace(ctx.context_fit_plan.core, memory_view_json=mv.snapshot_json(replace(snap, owner_words="New frame")))
        ctx.context_fit_plan = rebuild(core)
        assert ctx.context_fit_plan.core_sha256 != issuer["memory_core_sha256"]
    record_context_view(ctx, messages + [{"role": "user", "content": "No inspect in C"}], [], fit_measurement={})
    state = continuation_state(ctx, messages, {}, {}, 2, [], set())
    saved = store_continuation_source(ctx, state, "detached-issuer-continuation")
    state = json.loads(read_actor_source_bytes(tmp_path, ctx.task_id, saved))
    resumed = SimpleNamespace(task_id=ctx.task_id, drive_root=tmp_path, budget_drive_root=tmp_path,
                              context_fit_plan=ctx.context_fit_plan, active_context_mode="max")
    restore_continuation_state(SimpleNamespace(_ctx=resumed), state, [], {}, {}, set())
    if not changed:
        monkeypatch.setattr(memory_form_facts, "ordinary_composition", lambda *_: pytest.fail("same source regenerated"))
    second = json.loads(_compact_context(resumed, inspect=True))["ordinary_memory"]
    assert second["status"] == ("different_source" if changed else "available")
    if changed:
        assert second["previous_source_sha256"] == first["source"]["sha256"] and second["source"] != first["source"]
    else:
        assert second["source"] == first["source"]
    assert json.loads(read_actor_source_bytes(tmp_path, ctx.task_id, first["source"]))["identity"] == first["identity"]
