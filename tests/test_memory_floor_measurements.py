"""Count complete measurements, not just the rendered size of a small fixture."""
import hashlib
import json
import time

import pytest

from ouroboros import context_fit as cf, memory_floor as mf, memory_view as mv
from tests import _memory_scale_fixture as scale


def test_owner_floor_measures_each_selected_form_once(monkeypatch):
    snapshot = scale.snapshot()
    calls = []

    def measure(level):
        calls.append(level.addressed)
        return mf.view_tokens(mv.render_story(snapshot, level)) + mf.view_tokens(mv.render_room(snapshot, level))

    start = time.perf_counter()
    level = mf.fit_memory_view(snapshot, {'margin': 0, 'physical': 0}, measure=measure)
    elapsed = time.perf_counter() - start
    rendered = mv.render_story(snapshot, level) + mv.render_room(snapshot, level)
    print('FLOOR_MEASUREMENTS ' + json.dumps({'rows': 158, 'calls': len(calls),
        'unique': len(set(calls)), 'seconds': elapsed, 'sha256': hashlib.sha256(rendered.encode()).hexdigest()}))
    assert len(calls) <= 60, f'owner shape: at most 60 complete measurements, got {len(calls)}'
    assert len(calls) == len(set(calls)), 'a selected form must never be re-tokenized in the same walk'
    with monkeypatch.context() as uncached:
        uncached.setattr(mf, '_memoized_measure', lambda measure: measure)
        reference = mf.fit_memory_view(snapshot, {'margin': 0, 'physical': 0}, measure=measure)
    assert level == reference
    assert rendered == mv.render_story(snapshot, reference) + mv.render_room(snapshot, reference)


def test_no_note_measurement_without_a_physical_bound():
    calls = []
    mf.fit_memory_view(scale.snapshot(), {'margin': None, 'physical': None},
                       measure=lambda level: calls.append(level.addressed) or 100)
    assert len(calls) == 1, f'unpressured view: one initial measurement, got {len(calls)}'


@pytest.mark.parametrize('rows', [158, 300, 600])
def test_ordinary_scale_fit_measurements(tmp_path, monkeypatch, rows):
    calls = 0
    original = cf._request_tokens

    def count(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cf, '_request_tokens', count)
    start = time.perf_counter()
    plan = scale.build(tmp_path, monkeypatch, snap=scale.snapshot(rows=rows))
    elapsed = time.perf_counter() - start
    projection = plan.projection(plan.initial_mode)
    print('ORDINARY_FIT_MEASUREMENTS ' + json.dumps({'rows': rows, 'calls': calls, 'seconds': elapsed,
        'mode': plan.initial_mode, 'input': projection.calibrated_tokens,
        'sha256': hashlib.sha256(projection.system_content_json.encode()).hexdigest()}))
    assert plan.initial_mode == 'nano'
    assert calls <= rows // 3 + 15, f'{rows} rows: at most rows/3 + 15 complete envelope measurements, got {calls}'
