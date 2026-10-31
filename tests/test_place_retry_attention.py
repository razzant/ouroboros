"""Certified retries keep one captured arrival and the current attempt's custody."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store, place_arrivals, places, presence_runner as runner
from ouroboros.place_intake import settle_heard
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_place_focus import focus as focus


@pytest.fixture
def refused(focus, monkeypatch):
    root, _, admission, event = focus
    now = [datetime(2026, 10, 21, 12, tzinfo=timezone.utc)]
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now[0].astimezone(tz)
    monkeypatch.setattr(runner, 'datetime', Clock)
    task = runner.capture_presence_event(admission, event, drive_root=root)
    proof = runner.presence_retry_proof(task, {'_presence_pre_dispatch_only': True,
        'resource_refusal': {'temporary': True, 'reset_at': (now[0] + timedelta(minutes=5)).isoformat()}},
        {'tool_calls': []}, SimpleNamespace(_swarm_handoff_attempt=None))
    assert proof
    write_task_result(root, task['id'], 'failed', metadata={**task['metadata'], 'presence_retry_proof': proof},
                      chat_id=task['chat_id'], reason_code='resource_refusal_no_resend', result='private diagnostic')
    return task, now


@pytest.mark.parametrize('attention', ['paused', 'off', 'no_wake_match', 'focus'])
def test_certified_retry_can_settle_original_arrival_or_start_focus(focus, refused, monkeypatch, attention):
    root, place, admission, event = focus
    original, now = refused
    if attention == 'paused':
        places.edit_place(root, place.place_id, 'set_state', author='owner', state='paused')
    elif attention == 'off':
        monkeypatch.setattr('ouroboros.presence_admission.focus_enabled', lambda _: False)
    elif attention == 'no_wake_match':
        places.edit_place(root, place.place_id, 'remove_wake', author='owner', fact='mention')
    calls = []
    def handle(task):
        calls.append(task['id'])
        assert attention == 'focus', 'heard-only retry must not generate'
        write_task_result(root, task['id'], 'completed', result='Answer', terminal_origin='model_final',
            metadata={**task['metadata'], 'presence_outcome': 'message', 'presence_result_text': 'Answer'})
        return [{'type': 'presence_result', 'task_id': task['id'], 'outcome': 'message', 'text': 'Answer'}]
    def run():
        return runner.run_presence_turn(admission=admission, event=event, repo_dir=root, drive_root=root,
                                       agent_factory=lambda **_: SimpleNamespace(handle_task=handle))
    # A valid certificate does not authorize an early retry, even when now paused.
    with pytest.raises(runner.PresenceTurnError, match='presence_resources_unavailable'):
        run()
    assert not calls
    now[0] += timedelta(minutes=10)
    result = run()
    successor = load_task_result(root, original['id'])['presence_retry_next']
    assert successor != original['id']
    assert result.outcome == ('message' if attention == 'focus' else 'silent')
    assert calls == ([successor] if attention == 'focus' else [])
    assert (load_task_result(root, successor) is not None) is (attention == 'focus')
    with dialogue_store.transaction(root, write=False) as conn:
        rows = conn.execute("SELECT body,dispatch_input FROM dialogue WHERE kind='place_arrival'").fetchall()
    assert len(rows) == 1 and json.loads(rows[0]['body'])['task_id'] == original['id']
    decision = json.loads(rows[0]['dispatch_input'] or '{}').get('place_attention')
    if attention != 'focus':
        assert decision['reason'] == attention
        assert not list((root/'task_results').rglob('presence-event-*.json'))
    else:
        assert decision is None
    places.edit_place(root, place.place_id, 'set_state', author='owner', state='active')
    places.edit_place(root, place.place_id, 'add_wake', author='owner', fact='mention')
    monkeypatch.setattr('ouroboros.presence_admission.focus_enabled', lambda _: True)
    assert run() == result  # later On cannot reopen the final attention decision
    assert calls == ([successor] if attention == 'focus' else [])


@pytest.mark.parametrize('admitted', [False, True])
def test_retry_heard_decision_checks_current_attempt_under_intake_lock(focus, refused, monkeypatch, admitted):
    root, _, _, event = focus
    original, now = refused
    now[0] += timedelta(minutes=10)
    identity = original['metadata']['presence_event_identity']
    successor = runner._retry_target(root, original['id'], identity, claim=True)
    current = {**original, 'id': successor}
    if admitted:
        write_task_result(root, successor, 'running', metadata=current['metadata'])
    locked = []
    real_lock, real_read = place_arrivals.intake_lock, runner._stored_turn
    @contextmanager
    def intake(*args):
        with real_lock(*args):
            locked.append(True)
            try:
                yield
            finally:
                locked.pop()
    def read(*args):
        assert locked and args[1] == successor
        return real_read(*args)
    monkeypatch.setattr(place_arrivals, 'intake_lock', intake)
    monkeypatch.setattr(runner, '_stored_turn', read)
    result = settle_heard(root, current, event, 'paused', captured_task_id=original['id'])
    assert (result is None) is admitted
    assert (place_arrivals.heard_replay(root, original['id'], identity) is None) is admitted
    if admitted:
        assert load_task_result(root, successor)['status'] == 'running'


@pytest.mark.parametrize('proof_change', ['missing', 'foreign_event'])
def test_paused_place_does_not_turn_uncertified_failure_into_heard(focus, refused, monkeypatch, proof_change):
    root, place, admission, event = focus
    original, now = refused
    stored = load_task_result(root, original['id'])
    metadata = dict(stored['metadata'])
    proof = metadata['presence_retry_proof']
    metadata['presence_retry_proof'] = {} if proof_change == 'missing' else {**proof, 'event_identity': 'foreign'}
    write_task_result(root, original['id'], 'failed', metadata=metadata)
    now[0] += timedelta(minutes=10)
    places.edit_place(root, place.place_id, 'set_state', author='owner', state='paused')
    def run():
        return runner.run_presence_turn(admission=admission, event=event, repo_dir=root, drive_root=root,
                                       agent_factory=lambda **_: pytest.fail('paused event cannot generate'))
    with pytest.raises(runner.PresenceTurnError, match='presence_resources_unavailable'):
        run()
    assert place_arrivals.heard_replay(root, original['id']) is None
    assert not load_task_result(root, original['id']).get('presence_retry_next')
    write_task_result(root, original['id'], 'failed', metadata=stored['metadata'])
    assert run().outcome == 'silent'
