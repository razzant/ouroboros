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



def _arrive(root, place_id, chat_id, n):
    from ouroboros import dialogue_store
    dialogue_store.append(root, 'chat', {'ts': f'2026-11-14T10:00:{n:02d}+00:00', 'direction': 'in', 'type': 'place_arrival',
                                         'chat_id': chat_id, 'text': 'hi', 'client_message_id': f'{place_id}-{n}',
                                         'sender_session_id': f'person-{n}', 'sender_label': f'Person {n}', 'place_id': place_id})


def test_a_place_list_reads_its_stores_in_fixed_transactions_and_a_place_s_rooms_in_one_statement(tmp_path, monkeypatch):
    """#3399 (14.11): a place view opened ten store transactions per place and stepped one statement per captured
    room, and every step hands the interpreter lock over and waits to take it back: on the owner's busy server 31
    places with 2,169 rooms took 4 to 20 seconds. A list opens as many store transactions for any number of places,
    reads a place's rooms in one statement, and answers the same rooms, counts, activity and candidates."""
    from contextlib import contextmanager
    from ouroboros import dialogue_store, place_projection, store
    from tests._store_shared import seed_result
    monkeypatch.setattr(place_projection, 'place_declarations', lambda _root: [])
    opened, statements = [], []
    real_transaction, real_open_store = dialogue_store.transaction, store.open_store

    @contextmanager
    def transaction(*args, **kwargs):
        opened.append('dialogue')
        with real_transaction(*args, **kwargs) as conn:
            conn.set_trace_callback(statements.append)
            try:
                yield conn
            finally:
                conn.set_trace_callback(None)

    def open_store(*args, **kwargs):
        opened.append('tasks')
        return real_open_store(*args, **kwargs)
    monkeypatch.setattr(dialogue_store, 'transaction', transaction)
    monkeypatch.setattr(store, 'open_store', open_store)
    assert place_projection.places_view(tmp_path)['places'] == [] and not opened, 'no place, no store read'
    first = places.connect_place(tmp_path, connector='bridge', provider='chat', account_id='a')
    second = places.connect_place(tmp_path, connector='bridge', provider='chat', account_id='b')
    for n, chat in enumerate((901, 902, 903)):
        _arrive(tmp_path, first.place_id, chat, n)
    _arrive(tmp_path, second.place_id, 904, 3)
    seed_result(tmp_path, 'focus1', {'status': 'running', 'chat_id': 902, 'metadata': {
        'presence': {'binding_id': first.place_id}, 'wake_cause': 'focus'}})

    def read():
        place_projection.places_view(tmp_path)  # a root's first read builds its indexes; count a later one
        opened.clear(), statements.clear()
        rows = {row['place_id']: row for row in place_projection.places_view(tmp_path)['places']}
        return rows, len(opened), sum('INDEXED BY dialogue_place_rooms' in sql for sql in statements)

    rows, transactions, room_statements = read()
    assert [room['room_chat_id'] for room in rows[first.place_id]['rooms']] == [901, 902, 903]
    assert [room['unanswered_count'] for room in rows[first.place_id]['rooms']] == [0, 1, 0]
    assert (rows[first.place_id]['unanswered'], rows[second.place_id]['unanswered']) == (1, 0)
    assert rows[second.place_id]['last_activity_at'] == '2026-11-14T10:00:03+00:00'
    assert [p['platform_actor_id'] for p in rows[first.place_id]['account_candidates']] == ['person-2', 'person-1', 'person-0']
    assert room_statements == 2, 'one rooms statement per place, whatever its rooms'
    for n, chat in enumerate((905, 906, 907, 908), start=4):
        _arrive(tmp_path, first.place_id, chat, n)
    third = places.connect_place(tmp_path, connector='bridge', provider='chat', account_id='c')
    _arrive(tmp_path, third.place_id, 909, 9)
    grown, more_transactions, more_room_statements = read()
    assert [room['room_chat_id'] for room in grown[first.place_id]['rooms']] == [901, 902, 903, 905, 906, 907, 908]
    assert grown[third.place_id]['rooms'] == [{'room_chat_id': 909, 'unanswered_count': 0, 'address': 'chat:909'}]
    assert more_transactions == transactions, 'as many store transactions for the list, whatever its places'
    assert more_room_statements == 3
    # The list's bindings ride one parameter: no SQL variable ceiling however many places an install keeps.
    from types import SimpleNamespace
    from ouroboros.place_arrivals import _open_counts
    many = [SimpleNamespace(place_id=f'many-{n}', legacy_binding_ids=()) for n in range(40_000)]
    assert _open_counts(tmp_path, many) == {place.place_id: {} for place in many}
