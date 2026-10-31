"""Current question detail/census and task-scoped delivery use the same owners."""
from types import SimpleNamespace

import pytest

from ouroboros.gateway.main_decisions import main_decisions_snapshot
from ouroboros.owner_quiz import record_asked, record_publication
from ouroboros.projects_registry import create_project
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_question_room_detail import detail_client
from tests.test_main_terminal_delivery import owed, _due  # noqa: F401


@pytest.mark.parametrize('kind', ['context-main', 'missing-parent'])
def test_native_question_census_agrees_with_current_detail(tmp_path, kind):
    project = create_project(tmp_path, 'context-a', name='Context A')
    fields = ({'project_id': project['id']} if kind == 'context-main'
              else {'parent_task_id': 'absent'})
    write_task_result(tmp_path, kind, 'completed', chat_id=1, **fields)
    asked = record_asked(tmp_path, kind, quiz_id='q', question='Which?', options=['A'])
    record_publication(tmp_path, kind, 'q', 1)
    detail = detail_client(tmp_path).get('/api/tasks/' + kind).json()
    snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert detail['question_rooms']['q'] == {'state': 'known', 'chat_id': 1, 'project_id': ''}
    assert snapshot['current_questions_complete']
    assert snapshot['current_questions_gaps'] == []
    # Main's own card is its native history: the census names it in the detail's room, never a Project copy.
    assert snapshot['current_questions'] == [{'task_id': kind, 'quiz_id': 'q', 'chat_id': 1, 'native': True,
                                              'quiz_state': 'open', 'ts': asked['asked_at']}]


def test_unreadable_binding_is_unknown_in_both_readers(tmp_path, monkeypatch):
    from ouroboros import projects_registry as registry
    create_project(tmp_path, 'context-a', name='Context A')
    write_task_result(tmp_path, 'native', 'completed', chat_id=1)
    record_asked(tmp_path, 'native', quiz_id='q', question='Which?', options=['A'])
    record_publication(tmp_path, 'native', 'q', 1)
    monkeypatch.setattr(registry, '_load_bindings', lambda *a, **kw: (_ for _ in ()).throw(ValueError('unreadable')))
    detail = detail_client(tmp_path).get('/api/tasks/native').json()
    snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert detail['question_rooms']['q'] == {'state': 'unknown'}
    assert not snapshot['current_questions_complete']


@pytest.mark.parametrize('inflight', [False, True])
def test_canonical_main_reoffer_through_actual_task_endpoint(owed, inflight):  # noqa: F811 - imported pytest fixture
    from ouroboros.terminal_projection import settle_terminal_projection
    from supervisor.event_observation import ObservedEventQueue
    from supervisor.log_addressing import task_event_queue
    bus = ObservedEventQueue()
    endpoint = task_event_queue(bus, {'id': 'root', 'chat_id': owed.project['chat_id']})
    endpoint.put(owed.event)
    if inflight:
        bus.get_nowait()
    _due(owed.root)
    before = load_task_result(owed.root, 'root')['canonical_terminal_projection_ready']['main_delivery']
    settle_terminal_projection(owed.root, 'root', event_queue=endpoint)
    assert bus.qsize() == (0 if inflight else 1)
    assert load_task_result(owed.root, 'root')['canonical_terminal_projection_ready']['main_delivery'] == before
    if not inflight:
        bus.get_nowait()
    bus.release_delivery(owed.event['delivery_id'])
    settle_terminal_projection(owed.root, 'root', event_queue=endpoint)
    assert bus.get_nowait() == owed.event
    assert bus.empty()


def test_atomic_endpoint_preserves_address_stamp_and_unknown():
    from supervisor.event_observation import ObservedEventQueue, delivery_outstanding
    from supervisor.log_addressing import TurnEventQueue
    bus = ObservedEventQueue()
    endpoint = TurnEventQueue(bus, 'native', 1)
    event = {'type': 'send_message', 'task_id': 'native', 'delivery_id': 'exact', 'text': 'Result'}
    assert endpoint.put_unless_outstanding(event)
    assert delivery_outstanding(endpoint, 'exact') is True
    assert not endpoint.put_unless_outstanding(dict(event))
    assert bus.get_nowait()['chat_id'] == 1
    assert delivery_outstanding(TurnEventQueue(SimpleNamespace(), 'native', 1), 'exact') is None
