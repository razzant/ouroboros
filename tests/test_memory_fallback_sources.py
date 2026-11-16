"""Part-source provenance and fitting work at the actual writer-input boundary."""
import pytest

from ouroboros import memory_fallback as mf
from ouroboros.chronicle_store import ChronicleStore
from tests import _memory_production_shape as shape
from tests.test_memory_fallback import _part_fixture
from tests.test_memory_fallback import light as _light

light = _light


def test_part_input_carries_each_members_period_status_and_signer(tmp_path):
    made = shape.production(tmp_path)
    store = ChronicleStore(tmp_path)
    ident = made.corrected
    unit = mf.FallbackUnit('part', made.alpha, 'k', member_ids=(ident,))
    draft = mf.build_writer_input(tmp_path, store, unit, budget_tokens=None)
    assert 'nanny, task nan00001' in draft.prompt
    assert 'status accepted' in draft.prompt
    assert 'period 2026-09-01 00:02 → 2026-09-01 00:05' in draft.prompt
    assert shape.ADDITION in draft.prompt
    # Creation time is not the period covered; a missing period stays unknown.
    pending = store.publish_page(room_id=made.alpha, text='Unaccepted account.',
        author=shape.NANNY, covers={'rows': ['separate-fixture-row'], 'stream_span': [100, 100]})
    assert pending.ok
    unit = mf.replace(unit, member_ids=(pending.record['id'],))
    prompt = mf.build_writer_input(tmp_path, store, unit, budget_tokens=None).prompt
    assert 'status draft' in prompt and 'period unknown' in prompt
    assert 'Unaccepted account.' in prompt


def test_part_answer_format_precedes_authored_guidance(tmp_path, monkeypatch, light):
    store, ids, _ = _part_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(mf, '_remembering', lambda _: 'MY AUTHORED GUIDANCE')
    prompt = mf.build_writer_input(tmp_path, store,
        mf.FallbackUnit('part', '1', 'k', member_ids=tuple(ids)), budget_tokens=None).prompt
    assert prompt.index(mf._KEEP_ORIGINAL.strip()) < prompt.index('## Authored guidance')
    assert prompt.count('MY AUTHORED GUIDANCE') == 1


@pytest.mark.parametrize('count', [32, 64])
def test_small_part_window_does_not_rebuild_every_decrement(tmp_path, monkeypatch, light, count):
    store, ids, _ = _part_fixture(tmp_path, monkeypatch, count=count)
    unit = mf.FallbackUnit('part', '1', 'k', member_ids=tuple(ids))
    single = mf.build_writer_input(tmp_path, store, mf.replace(unit, member_ids=tuple(ids[:1])), budget_tokens=None)
    calls, labels, rendered = [], [], []
    label, indent = mf._room_label, mf._indented
    def room(*args):
        labels.append(args)
        return label(*args)
    def measure(text):
        calls.append(len(text))
        return len(text)
    def indented(text):
        rendered.append(len(text))
        return indent(text)
    monkeypatch.setattr(mf, '_room_label', room)
    monkeypatch.setattr(mf, '_indented', indented)
    draft = mf.build_writer_input(tmp_path, store, unit, budget_tokens=len(single.prompt)+10, measure=measure)
    assert draft.member_ids == tuple(ids[:1])
    assert len(labels) == 1
    assert len(rendered) <= count + 1
    # Stamps can shrink at a later prefix; exact descending measurement is
    # linear, while source blocks and periods are prepared only once.
    assert len(calls) <= count + 1
