"""Real fitting/selection regressions found by the independent contribution review."""
import json

import pytest

from ouroboros import memory_fallback as mf
from ouroboros import memory_inventory as mi
from ouroboros.chronicle_store import ChronicleStore
from tests import _memory_inventory_shared as shared
from tests.test_memory_fallback import _consciousness, _Light, _mind_page, _run, _shortage
from tests.test_memory_fallback import light as _light

light = _light


def test_rejected_fitted_page_reaches_untouched_row_suffix(tmp_path, monkeypatch, light):
    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    _mind_page(tmp_path, "1", 10, 19)
    for n in range(4):
        shared.append(tmp_path / "logs/chat.jsonl", shared.msg(
            f"2026-09-04T00:0{n}:00+00:00", f"UNTOUCHED-{n}: " + "x" * 1800,
            client_message_id=f"row-{n}"))
    store = ChronicleStore(tmp_path)
    trace = _shortage("1", 23, tmp_path)
    unit = next(mf.candidate_units(tmp_path, store, mi.shortage_from_trace(trace), allow_legacy=False))
    segment = mf.replace(unit.segment, rows=unit.segment.rows[:1], to_addr=unit.segment.rows[0][0])
    single = mf.build_writer_input(tmp_path, store, mf.replace(unit, segment=segment), budget_tokens=None)
    monkeypatch.setattr(mf, "_light_fit", lambda _: (len(single.prompt) + 10, len))
    first = _run(tmp_path, _Light(), direct=True, trace=trace)
    assert first.outcome == "published"
    assert store.get(first.record_id)["covers"]["count"] == 1
    assert store.decide(first.record_id, False, shared.MIND, "The helper misattributed that first row.").ok
    shared.append(tmp_path / "logs/chat.jsonl", shared.msg(
        "2026-09-04T00:04:00+00:00", "A later owner turn must not make untouched history unreachable.",
        client_message_id="later-owner"))
    llm = _Light()
    second = _run(tmp_path, llm, direct=True, trace=_shortage("1", 24, tmp_path))
    assert second.outcome == "published", "unattempted row suffix must stay reachable"
    assert len(llm.prompts) == 1
    assert "UNTOUCHED-1" in llm.prompts[0] and "UNTOUCHED-0" not in llm.prompts[0]
    assert store.get(first.record_id)["metadata"]["input_sha256"] != store.get(second.record_id)["metadata"]["input_sha256"]


def test_part_prefix_search_handles_later_terminal_stamp_reduction(tmp_path, monkeypatch, light):
    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    store, ids = ChronicleStore(tmp_path), []
    earlier = [{"task_id": f"task-{n:03d}", "status": "completed", "outcome_phase": "done",
                "source": "task_results"} for n in range(24)]
    later = [{**row, "source": "terminal_root_projection", "reason_code": ""} for row in earlier]
    for n in range(5):
        result = store.publish_page(room_id="1", text=f"Report {n}." + ("x" * 9000 if n == 4 else ""),
            author=shared.MIND, covers={"rows": [f"row-{n}"], "stream_span": [100+n, 100+n]},
            host_stamp={"tasks": later if n >= 3 else earlier})
        assert result.ok
        ids.append(result.record["id"])
    unit = mf.FallbackUnit("part", "1", "part:1:" + ids[0], member_ids=tuple(ids),
                           head_sequence=store.room_head("1"))
    _, measure = mf._light_fit(mf.light_binding())
    sizes = [measure(mf.build_writer_input(tmp_path, store, mf.replace(unit, member_ids=tuple(ids[:n])),
                                          budget_tokens=None).prompt) for n in range(1, 6)]
    budget = sizes[3] + 1
    assert sizes[3] < min(sizes[:3]) and sizes[4] > budget
    selected = mf.build_writer_input(tmp_path, store, unit, budget_tokens=budget, measure=measure)
    assert selected.input_tokens <= budget
    assert selected.member_ids == tuple(ids[:4])
    light.window = mf.ANSWER_RESERVE_TOKENS + budget
    trace = {mi.VIEW_TRACE_KEY: {"role": "integrator", "room_id": "1", "floor": {
        "steps": {"F5": 5}, "pointer_records": ids, "by_budget": 0, "mode": "max", "window_tokens": 272000}}}
    llm = _Light()
    assert _run(tmp_path, llm, direct=True, trace=trace).outcome == "published"
    assert len(llm.prompts) == 1


@pytest.mark.parametrize("count", [12, 24])
def test_nested_part_suffixes_reuse_period_sources_and_rendered_members(tmp_path, monkeypatch, light, count):
    shared.world(tmp_path)
    store, ids = ChronicleStore(tmp_path), []
    for n in range(count):
        members = []
        for j in range(2):
            page = store.publish_page(room_id="1", text=f"Source {n}-{j}.", author=shared.MIND,
                covers={"rows": [f"nested-{n}-{j}"], "stream_span": [100+n*2+j, 100+n*2+j]})
            assert page.ok
            members.append(page.record["id"])
        part = store.publish_part(room_id="1", text=f"Part {n}: " + "history " * 200,
            member_ids=members, author=shared.MIND, expected_sequence=store.room_head("1"))
        assert part.ok
        ids.append(part.record["id"])
    unit = mf.FallbackUnit("part", "1", "part:1:"+ids[0], member_ids=tuple(ids), head_sequence=store.room_head("1"))
    for ident in ids:
        single = mf.replace(unit, key="part:1:"+ident, member_ids=(ident,))
        draft = mf.build_writer_input(tmp_path, store, single, budget_tokens=None)
        mf._set_refusal(store, single.key, {"input_sha256": draft.input_sha256, "light_binding": mf.light_binding()})
    budget = len(draft.prompt) + 20
    # Isolate the selection/fitting work from the pointer discovery cost.
    monkeypatch.setattr(mf, "candidate_units", lambda *a, **k: iter([unit]))
    reads, rendered = [], []
    original_get, original_indent = store.get, mf._indented
    def get(ident):
        reads.append(ident)
        return original_get(ident)
    def indent(text):
        rendered.append(text)
        return original_indent(text)
    monkeypatch.setattr(store, "get", get)
    monkeypatch.setattr(mf, "_indented", indent)
    assert mf.select_unit(tmp_path, store, None, allow_legacy=False,
        binding=mf.light_binding(), fit=lambda: (budget, len)) is None
    assert len(reads) <= count, (count, len(reads))
    assert sum(str(text).startswith("Part ") for text in rendered) == count
    # A new selection pass observes new source content, rather than a persistent cache.
    assert store.correct(ids[0], "The earlier account needs this new qualification.", shared.MIND).ok
    unit = mf.replace(unit, head_sequence=store.room_head("1"))
    chosen = mf.select_unit(tmp_path, store, None, allow_legacy=False,
        binding=mf.light_binding(), fit=lambda: (None, len))
    assert chosen is not None and "new qualification" in chosen[1].prompt


def _legacy_two_row_input(tmp_path, monkeypatch, *, middle=False):
    from tests import _dialogue as dialogue_test

    # The legacy chain files are edited before the dialogue import reads them (a frozen segment is
    # imported once; a later edit of the file is a mismatch, never a new row).
    with monkeypatch.context() as deferred:
        deferred.setattr(shared, 'import_all', lambda root: None)
        rooms = shared.world(tmp_path, activate=False)
    _consciousness(monkeypatch, False)
    # A real imported legacy segment; both source rows remain in the original archive.
    archive = tmp_path / 'archive' / shared.ARCHIVE_TWO
    rows = []
    for line in archive.read_text().splitlines(keepends=True):
        if middle and 'beta answer' in line:
            rows.append(json.dumps(shared.msg('2026-09-02T00:02:30+00:00',
                'beta middle' + ' x' * 900, chat_id=rooms['beta'], client_message_id='b-middle')) + '\n')
        if 'beta question' in line or 'beta answer' in line:
            row = json.loads(line)
            row['text'] += ' x' * 900
            row['direction'] = 'in'  # Both are spoken rows that may not become address-only input.
            row.pop('task_id', None)
            line = json.dumps(row) + '\n'
        rows.append(line)
    archive.write_text(''.join(rows))
    if middle:
        meta_path = tmp_path / 'memory/dialogue_meta.json'
        meta = json.loads(meta_path.read_text())
        meta['last_consolidated_offset'] += 1  # The imported range includes the inserted source row.
        meta_path.write_text(json.dumps(meta))
        blocks_path = tmp_path / 'memory/dialogue_blocks.json'
        blocks = json.loads(blocks_path.read_text())
        blocks[1]['message_count'] += 1
        next(room for room in blocks[1]['rooms'] if room['room_id'] == str(rooms['beta']))['message_count'] += 1
        blocks_path.write_text(json.dumps(blocks))
    dialogue_test.import_all(tmp_path)
    store = ChronicleStore(tmp_path)
    store.ensure_activated()
    unit = next(u for u in mf.candidate_units(tmp_path, store, None, allow_legacy=True)
                if u.room_id == str(rooms['beta']))
    assert unit.kind == 'legacy' and len(unit.segment.rows) == (3 if middle else 2)
    segment = mf.replace(unit.segment, rows=unit.segment.rows[:1], to_addr=unit.segment.rows[0][0])
    single = mf.build_writer_input(tmp_path, store, mf.replace(unit, segment=segment), budget_tokens=None)
    budget = len(single.prompt) + 10
    prefix = mf.build_writer_input(tmp_path, store, unit, budget_tokens=budget, measure=len)
    assert prefix.covers['count'] == 1
    return store, unit, prefix, budget


def test_rejected_legacy_prefix_keeps_its_untouched_rows_reachable(tmp_path, monkeypatch, light):
    store, unit, _prefix, budget = _legacy_two_row_input(tmp_path, monkeypatch)
    monkeypatch.setattr(mf, '_light_fit', lambda _: (budget, len))
    monkeypatch.setattr(mf, 'candidate_units', lambda *a, **k: iter([unit]))
    first = _run(tmp_path, _Light())
    assert first.outcome == 'published' and store.get(first.record_id)['covers']['count'] == 1
    assert store.decide(first.record_id, False, shared.MIND, 'Keep the source.').ok
    second_llm = _Light()
    second = _run(tmp_path, second_llm)
    assert second.outcome == 'published' and len(second_llm.prompts) == 1
    assert 'beta answer' in second_llm.prompts[0] and 'beta question' not in second_llm.prompts[0]
    assert second.unit.record_id == unit.record_id and second.unit.key != first.unit.key
    assert store.decide(second.record_id, False, shared.MIND, 'Keep this source too.').ok
    third_llm = _Light()
    assert _run(tmp_path, third_llm).outcome == 'nothing' and not third_llm.prompts


def test_fitted_endpoint_includes_addressed_rows_not_only_sealed_rows(tmp_path, monkeypatch, light):
    shared.world(tmp_path)
    store = ChronicleStore(tmp_path)
    unit = next(mf.candidate_units(tmp_path, store, mi.shortage_from_trace(_shortage('1', 19, tmp_path)),
                                   allow_legacy=False))
    assert len(unit.segment.rows) >= 3
    # The first row was read, the second addressed, the remaining ones unattempted.
    draft = mf.WriterInput('bounded input', 'hash', None, covers={'last': unit.segment.rows[0][0]},
                           fitted_to=unit.segment.rows[1][0])
    suffix = mf._untouched_suffix(unit, draft)
    assert suffix.segment.rows == unit.segment.rows[2:]
    assert suffix.segment.from_addr == unit.segment.rows[2][0]


def test_snapshot_periods_equal_live_periods_for_nested_and_missing_sources(tmp_path):
    from tests import _memory_production_shape as shape

    shape.production(tmp_path)
    store = ChronicleStore(tmp_path)
    records = {r['id']: r for r in store.records()}
    units = {u.record_id: u for u in mi.legacy_units(store, tmp_path)}
    for record in records.values():
        assert mi.record_period(store, record, units, records=records) == mi.record_period(store, record, units)
    missing = {'kind': 'part', 'covers': {'member_ids': ['missing']}}
    assert mi.record_period(store, missing, units, records=records) == mi.record_period(store, missing, units)


@pytest.mark.parametrize('rejected', [False, True], ids=['refusal', 'rejected-draft'])
def test_legacy_suffix_identity_survives_the_mind_sealing_only_its_prefix(tmp_path, monkeypatch, light, rejected):
    store, unit, prefix, budget = _legacy_two_row_input(tmp_path, monkeypatch)
    candidates = mf.candidate_units
    monkeypatch.setattr(mf, '_light_fit', lambda _: (budget, len))
    monkeypatch.setattr(mf, 'candidate_units', lambda *a, **k: iter([unit]))
    for ordinal in range(2):
        llm = _Light() if rejected else _Light('not a memory answer')
        run = _run(tmp_path, llm)
        assert len(llm.prompts) == 1
        if rejected:
            assert run.outcome == 'published'
            assert store.decide(run.record_id, False, shared.MIND, 'Preserve the source.').ok
        else:
            assert run.outcome == 'refused' and run.kind == 'invalid'
        if ordinal == 0:
            assert run.unit.key == unit.record_id
        else:
            assert run.unit.key != unit.record_id
    suffix = run.unit
    suffix_draft = mf.build_writer_input(tmp_path, store, suffix, budget_tokens=budget, measure=len)
    quiet = _Light()
    assert _run(tmp_path, quiet).outcome == 'nothing' and not quiet.prompts
    assert store.publish_page(room_id=unit.room_id, text='The mind records the first row only.',
                              author=shared.MIND, covers=prefix.covers, host_stamp=prefix.stamp).ok
    # Actual enumeration rekeys the same remaining rows as the legacy record's head.
    head = next(u for u in candidates(tmp_path, store, None, allow_legacy=True)
                if u.kind == 'legacy' and u.room_id == unit.room_id)
    assert head.key == unit.record_id and head.segment.rows == suffix.segment.rows
    again = mf.build_writer_input(tmp_path, store, head, budget_tokens=budget, measure=len)
    assert again.input_sha256 == suffix_draft.input_sha256
    monkeypatch.setattr(mf, 'candidate_units', lambda *a, **k: iter([head]))
    quiet = _Light()
    assert _run(tmp_path, quiet).outcome == 'nothing' and not quiet.prompts
    light.model = 'test/another-light'
    changed = _Light()
    assert _run(tmp_path, changed).outcome == 'published' and len(changed.prompts) == 1


def test_legacy_suffix_also_honors_an_existing_bare_key_receipt(tmp_path, monkeypatch, light):
    store, unit, prefix, budget = _legacy_two_row_input(tmp_path, monkeypatch)
    suffix = mf._untouched_suffix(unit, prefix)
    draft = mf.build_writer_input(tmp_path, store, suffix, budget_tokens=budget, measure=len)
    mf._set_refusal(store, unit.record_id, {'input_sha256': draft.input_sha256,
                                         'light_binding': mf.light_binding()})
    monkeypatch.setattr(mf, 'candidate_units', lambda *a, **k: iter([suffix]))
    assert mf.select_unit(tmp_path, store, None, allow_legacy=True,
                          binding=mf.light_binding(), fit=lambda: (budget, len)) is None


@pytest.mark.parametrize('rejected', [False, True], ids=['refusal', 'rejected-draft'])
def test_legacy_later_open_segment_survives_a_published_middle_row(tmp_path, monkeypatch, light, rejected):
    store, unit, _prefix, budget = _legacy_two_row_input(tmp_path, monkeypatch, middle=True)
    candidates = mf.candidate_units
    monkeypatch.setattr(mf, '_light_fit', lambda _: (budget, len))
    # Keep real discovery and its sealed-row segmentation, isolating only the room.
    monkeypatch.setattr(mf, 'candidate_units', lambda *a, **k: (
        u for u in candidates(*a, **k) if u.room_id == unit.room_id))
    first = _run(tmp_path, _Light() if rejected else _Light('invalid'))
    if rejected:
        assert first.outcome == 'published'
        assert store.decide(first.record_id, False, shared.MIND, 'Keep this prefix.').ok
    else:
        assert first.outcome == 'refused'
    middle_llm = _Light()
    middle = _run(tmp_path, middle_llm)
    assert middle.outcome == 'published' and len(middle_llm.prompts) == 1
    assert 'beta middle' in middle_llm.prompts[0] and 'beta question' not in middle_llm.prompts[0]
    assert store.get(middle.record_id)['covers']['count'] == 1
    last_llm = _Light()
    last = _run(tmp_path, last_llm)
    assert last.outcome == 'published' and len(last_llm.prompts) == 1
    assert 'beta answer' in last_llm.prompts[0] and 'beta middle' not in last_llm.prompts[0]
    assert 'beta question' not in last_llm.prompts[0]
