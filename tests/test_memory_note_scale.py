"""Ordinary note shortage on the owner's room geometry, through the same fitter."""
import json

import pytest

from devtools.context_fit_report import measure_core
from ouroboros import memory_floor as mf, memory_view as mv
from tests import _memory_scale_fixture as scale


@pytest.mark.parametrize('extra_chars, note_lines, aggregate, fits', [
    (0, 6, True, True), (32000, 6, True, True), (48000, 6, True, True),
    (52000, 6, True, False), (53000, 6, True, False), (54000, 6, True, False),
])
def test_note_pressure_measurement(tmp_path, monkeypatch, extra_chars, note_lines, aggregate, fits):
    snap = scale.snapshot()
    # More retained words in a recent, immune row; no protected source or reserve is reduced.
    snap.room['lane1'][-1]['line'] += 'w' * extra_chars
    result = measure_core(scale.core(tmp_path, snap), scale.tools(), root=tmp_path, window=92000, reserve=10000)
    diagnostic = (f"input={result['input_tokens']}, reserve=10000, window=92000, "
                  f"margin={result['margin_tokens']}, forms={result['ladder']}")
    assert (result['verdict'] == 'fits') is fits, diagnostic
    assert (result['margin_tokens'] >= 0) is fits, diagnostic
    assert result['ladder'].get('F8', 0) == note_lines, diagnostic
    assert bool(result['ladder'].get('F8a')) is aggregate, diagnostic
    assert result['ladder']['F7a'] == 1 and result['ladder']['F5a'] == 1
    assert result['ladder']['F7'] == {0: 18, 32000: 34, 48000: 42, 52000: 43, 53000: 43, 54000: 43}[extra_chars]
    with monkeypatch.context() as earlier:
        earlier.setattr(mf, 'LADDER', tuple(s for s in mf.LADDER if s not in ('F8', 'F8a')))
        before = measure_core(scale.core(tmp_path, snap), scale.tools(), root=tmp_path, window=92000, reserve=10000)
    if extra_chars:
        assert before['verdict'] == 'MinimumViewUnfit' and before['deficit_tokens'] > 0, before
    else:
        assert before['verdict'] == 'fits' and before['ladder']['F7'] > result['ladder']['F7']
    for key, size in scale.PROTECTED.items():
        label = {'system': 'SYSTEM', 'bible': 'BIBLE'}.get(key, key)
        assert next(row for row in result['blocks'] if row['block'] == label)['chars'] == size
    assert sum(row['tokens'] for row in result['blocks']) == result['input_tokens']
    print('NOTE_PRESSURE ' + json.dumps({'extra_chars': extra_chars, **result}, sort_keys=True))


@pytest.mark.parametrize('last_step', ['nothing', 'F5a', 'F8', 'F8a'])
def test_main_nano_floor_composition_by_available_forms(tmp_path, monkeypatch, last_step):
    # A counterfactual minimum, not the automatic choice: expose each form's price
    # on identical source bytes. The automatic-choice test above keeps notes whole when they fit.
    allowed = set(mf.LADDER)
    if last_step == 'nothing':
        allowed -= {'F7a', 'F5a', 'F8', 'F8a'}
    elif last_step == 'F5a':
        allowed -= {'F8', 'F8a'}
    elif last_step == 'F8':
        allowed -= {'F8a'}
    def minimum(snapshot, *_args, **_kwargs):
        elements = mf.floor_elements(snapshot)
        return mv.FloorLevel(tuple((step, tuple(ident for s, ident, *_ in elements if s == step))
                                  for step in mf.LADDER if step in allowed))
    monkeypatch.setattr(mf, 'fit_memory_view', minimum)
    result = measure_core(scale.core(tmp_path), scale.tools(), root=tmp_path, window=92000, reserve=10000)
    assert result['reserve'] == 10000 and result['mode'] == 'nano'
    assert (result['margin_tokens'] < 0) is (last_step == 'nothing'), result
    print('NOTE_FLOOR ' + json.dumps({'last_step': last_step, **result}, sort_keys=True))
