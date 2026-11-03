"""Owner room discovery uses captured Presence rows and never creates a focus."""
from dataclasses import replace
from starlette.applications import Starlette
from starlette.testclient import TestClient
import pytest
from ouroboros.gateway import places_api
from ouroboros.presence_runner import capture_presence_event
from ouroboros.place_intake import settle_heard
from ouroboros.presence_admission import admit_presence_turn
from ouroboros import places
from ouroboros.store import StoreError
from tests.place_history_scenario import watch_transactions
from tests.test_place_focus import focus as focus


@pytest.fixture
def owner_client(focus, monkeypatch):
    monkeypatch.setattr(places_api, 'request_drive_root', lambda _request: focus[0])
    with TestClient(Starlette(routes=places_api.places_routes())) as client:
        yield client


def test_rooms_include_heard_only_and_do_not_mix_places(focus, owner_client):
    root, place, admission, event = focus
    task = capture_presence_event(admission, replace(event, facts=()), drive_root=root)
    settle_heard(root, task, replace(event, facts=()), 'no_wake_match')
    other = places.connect_place(root, connector='other', provider='chat', account_id='b', conversation_id='second', wakes=['mention'])
    other_event = replace(event, source_event_id='other', account_id='b', conversation_id='second', conversation_key='chat:b:second')
    other_admission = admit_presence_turn(drive_root=root, authenticated_transport_skill='other', provider='chat', account_id='b', conversation_id='second')
    other_task = capture_presence_event(other_admission, other_event, drive_root=root)
    response = owner_client.get('/api/places/'+place.place_id+'/rooms')
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    body = response.json()
    assert body['place_id'] == place.place_id
    assert [r['room_chat_id'] for r in body['rooms']] == [task['chat_id']]
    assert task['chat_id'] != other_task['chat_id'] and task['chat_id'] > 2**32
    assert body['rooms'][0]['address'] == 'chat:'+str(task['chat_id'])
    assert body['rooms'][0]['unanswered_count'] is None
    assert places.place_room_index(root, other.place_id)[0]['room_chat_id'] == other_task['chat_id']


def test_polls_read_no_arrival_body_and_the_owner_read_decodes_after_its_transaction(focus, owner_client, monkeypatch):
    from ouroboros import consciousness_view, dialogue_store
    root, place, admission, event = focus
    task = capture_presence_event(admission, replace(event, conversation={'title': '#design'}), drive_root=root)
    watch, decode, decoded = watch_transactions(monkeypatch), dialogue_store.decode, []
    def watched_decode(body):
        assert not watch.open, 'an arrival body was decoded inside the read transaction'
        decoded.append(body)
        return decode(body)
    monkeypatch.setattr(dialogue_store, 'decode', watched_decode)
    plain = {'room_chat_id': task['chat_id'], 'unanswered_count': None, 'address': f"chat:{task['chat_id']}"}
    # /api/state and /api/consciousness reach rooms through place_views; they need counts and addresses only.
    assert places.place_room_index(root, place.place_id) == [plain]
    assert next(view['rooms'] for view in consciousness_view.place_views(root) if view['place_id'] == place.place_id) == [plain]
    assert not decoded and not any('SELECT body' in sql for sql in watch.statements), 'polls stay index-only'
    room, = owner_client.get(f'/api/places/{place.place_id}/rooms').json()['rooms']
    assert room == {**plain, 'label': '#design', 'last_at': room['last_at'], 'conversation_id': 'r', 'thread_id': ''}
    assert room['last_at'] and len(decoded) == 1


def test_absent_place_is_not_an_empty_success(owner_client):
    response = owner_client.get('/api/places/absent/rooms')
    assert response.status_code == 404
    assert response.json()['code'] == 'place_not_found'
    assert 'rooms' not in response.json()


def test_failed_store_read_remains_unavailable(owner_client, monkeypatch, focus):
    def fail(*_args, **_kwargs):
        raise StoreError('test store unavailable')
    monkeypatch.setattr(places_api, 'place_room_index', fail)
    response = owner_client.get('/api/places/'+focus[1].place_id+'/rooms')
    assert response.status_code == 503
    assert response.json()['code'] == 'place_rooms_unavailable'
    assert 'rooms' not in response.json()


def test_corrupt_place_store_is_unavailable_not_absent(owner_client, focus):
    root, place, *_ = focus
    (root / 'state' / 'places.json').write_text('{broken', encoding='utf-8')
    response = owner_client.get('/api/places/' + place.place_id + '/rooms')
    assert response.status_code == 503
    assert response.json()['code'] != 'place_not_found'
    assert 'rooms' not in response.json()
