"""#1329: a seat refused for an unreachable exact source keeps its stronger physical custody.

The acceptance packet and its source-first delivery are gone with the
task-acceptance machine. What outlived them is the review engine's custody
projection: a typed source refusal never downgrades a seat whose send was
already dispatched, started or settled to "not dispatched".
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize('fact,expected', [
    ('none', 'not_dispatched'), ('reserved', 'not_dispatched'), ('released', 'not_dispatched'),
    ('settled', 'settled'), ('dispatched', 'custody_lost'), ('unresolved', 'custody_lost'),
    ('malformed', 'custody_lost'), ('run', 'settled'), ('started', 'settled'),
    ('pending', 'custody_lost'), ('native_round', 'settled'),
])
def test_source_refusal_preserves_stronger_physical_custody(monkeypatch, fact, expected):
    from types import SimpleNamespace
    from ouroboros.review_custody import _ReviewAttemptHistory, _review_exception_projection
    from ouroboros.review_execution import ReviewRouteUnavailable

    # Each real slot starts a fresh capture scope; no prior test's send belongs here.
    monkeypatch.setattr('ouroboros.usage_accounting.last_physical_attempt_capture', lambda: None)
    error = ReviewRouteUnavailable('exact source missing', code='degraded_source_unreachable')
    if fact in {'reserved', 'released', 'settled', 'dispatched', 'unresolved', 'malformed'}:
        error.physical_attempt_capture = SimpleNamespace(state=fact)
    elif fact == 'run':
        error.delegated_run_id = 'synthetic-run'
    elif fact == 'started':
        error.delegated_run_started = True
    history = _ReviewAttemptHistory()
    history.observe(error)
    custody = {'native_rounds': 1} if fact == 'native_round' else {}
    retry = {'pending_invocation_id': 'synthetic-pending'} if fact == 'pending' else {}
    assert _review_exception_projection(error, custody, history, retry)[3] == expected
