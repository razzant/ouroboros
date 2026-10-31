"""Synthetic producer -> authenticated owner API scenario; no connector or model runs."""
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import dialogue_store as ds, places
from ouroboros.gateway import places_api
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.place_intake import settle_heard
from ouroboros.place_reply import reply_hold
from ouroboros.presence_admission import admit_presence_turn
from ouroboros.presence_bindings import conversation_key
from ouroboros.presence_delivery import PresenceDeliveryRecorder
from ouroboros.presence_runner import PresenceTurnEvent, capture_presence_event, _log_dialogue
from ouroboros.task_results import write_task_result
from ouroboros.task_status import execution_owner_record


def scenario(root, monkeypatch):
    monkeypatch.setattr(places_api, 'request_drive_root', lambda _request: root)
    monkeypatch.setattr('ouroboros.place_projection.place_declarations', lambda _root: [])
    monkeypatch.setattr('ouroboros.presence_admission.focus_enabled', lambda _root: True)
    monkeypatch.setattr('ouroboros.presence_admission.focus_allowance', lambda _root: {'status': 'available', 'remaining_usd': 2})
    place = places.connect_place(root, connector='bridge', provider='chat', account_id='qa',
                                 label='Slack · Acme private workspace', wakes=['mention'])
    places.connect_place(root, connector='bridge', provider='chat', account_id='other', label='Northwind private team')
    places.edit_place(root, place.place_id, 'confirm_owner_account', author='owner', platform_actor_id='owner', account_id='qa')
    admission = admit_presence_turn(drive_root=root, authenticated_transport_skill='bridge', provider='chat', account_id='qa', conversation_id='design')
    event = PresenceTurnEvent('qa-event', 'chat', 'qa', 'design', '', conversation_key('chat', 'qa', 'design', ''),
                              {'platform_actor_id': 'maya', 'display_name': 'Maya Private'},
                              {'title': '#design-review private'}, {}, 'Please check **frame 6**.', facts=('mention',))
    task = capture_presence_event(admission, event, drive_root=root)
    write_task_result(root, task['id'], 'running', metadata=task['metadata'], chat_id=task['chat_id'],
                      execution_owner=execution_owner_record(root, task, 'presence'))
    _log_dialogue(root, task=task, event=event, direction='out', text='The **Continue** button needs more contrast.\n\n![remote](https://images.invalid/private.png)',
                  chat_id=task['chat_id'], user_id=0, task_id=task['id'])
    # A retained legacy v0 answer with the original metadata bug.
    ds.append(root, 'chat', {'ts': '2026-10-22T10:04:00Z', 'direction': 'out', 'chat_id': task['chat_id'],
                           'text': 'Legacy **answer**', 'source': 'presence:chat', 'sender_label': 'Maya Private',
                           'sender_session_id': 'maya', 'transport': {'delivery': {'state': 'authored'}}})
    owner = replace(event, source_event_id='qa-owner', actor={'platform_actor_id': 'owner', 'display_name': 'Anton Private'}, text='Keep the unsaved note.')
    captured = capture_presence_event(admission, owner, drive_root=root)
    settle_heard(root, captured, owner, 'no_wake_match')
    paused = replace(event, source_event_id='qa-paused', text='I can still report this while Paused.')
    captured = capture_presence_event(admission, paused, drive_root=root)
    settle_heard(root, captured, paused, 'paused')
    recorder = PresenceDeliveryRecorder(root)
    for state in ('accepted', 'delivered', 'failed', 'uncertain'):
        recorder.record('bridge', {'schema_version': 1, 'delivery_id': 'qa-' + state, 'part_id': '0', 'state': state,
                                  'provider': 'chat', 'account_id': 'qa', 'conversation_id': 'design', 'thread_id': '',
                                  'text': 'Receipt-backed reply: ' + state, 'format': 'markdown', 'message': {},
                                  'origin': {'kind': 'tool'}})
    places.edit_place(root, place.place_id, 'set_state', author='owner', state='silent')
    reply_hold(root, task['metadata'], task['id'], 'Held reply body stays in its task')
    ds.append(root, 'progress', {'ts': '2026-10-25T10:00:00Z', 'chat_id': task['chat_id'], 'task_id': task['id'], 'text': 'PRIVATE THOUGHT', 'content': 'PRIVATE THOUGHT'})
    ds.append(root, 'chat', {'ts': '2026-10-25T10:00:01Z', 'direction': 'system', 'chat_id': task['chat_id'],
                           'type': 'task_evidence', 'task_id': task['id'], 'text': 'TECHNICAL EVIDENCE'})
    app = Starlette(routes=[*places_api.places_routes(), Route('/api/chat/history', make_chat_history_endpoint(root))])
    return TestClient(app), place, task, event


def watch_transactions(monkeypatch):
    """Probe: how many dialogue transactions are open right now, and every statement they ran."""
    original, watch = ds.transaction, SimpleNamespace(open=0, statements=[])

    @contextmanager
    def watched(*args, **kwargs):
        with original(*args, **kwargs) as conn:
            watch.open += 1
            conn.set_trace_callback(watch.statements.append)
            try:
                yield conn
            finally:
                conn.set_trace_callback(None)
                watch.open -= 1
    monkeypatch.setattr(ds, 'transaction', watched)
    return watch
