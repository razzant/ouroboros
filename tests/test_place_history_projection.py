"""Producer -> canonical store -> room/history APIs, with exact facts and coverage."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import dialogue_import, dialogue_store as ds, places
from ouroboros.chat_references import reference_from_row
from ouroboros.gateway import place_history, places_api
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.presence_delivery import PresenceDeliveryRecorder
from ouroboros.presence_runner import capture_presence_event
from tests._store_shared import seed_result
from tests.place_history_scenario import scenario, watch_transactions


def test_producers_project_only_recorded_facts_without_rewriting_sources(tmp_path, monkeypatch):
    client, place, task, event = scenario(tmp_path, monkeypatch)
    before = [row.row for row in ds.iter_rows(tmp_path, 'chat')]
    index = client.get(f'/api/places/{place.place_id}/rooms').json()
    assert index['coverage'] == {'discovery': 'arrivals', 'importing': False, 'gaps': False}
    room, = index['rooms']
    assert room['room_chat_id'] == task['chat_id'] > 2**32
    assert room['label'] == event.conversation['title'] and room['conversation_id'] == 'design'
    assert room['last_at'] and room['unanswered_count'] == 1
    page = client.get('/api/chat/history', params={'chat_id': task['chat_id']}).json()
    rows = [row for row in page['messages'] if 'place_history' in row]
    by_text = {row['text']: row for row in rows}
    assert by_text['Please check **frame 6**.']['place_history']['attention'] == {'decision': 'focus_started'}
    assert by_text['I can still report this while Paused.']['place_history']['attention'] == {'decision': 'heard', 'reason': 'paused'}
    assert by_text['I can still report this while Paused.']['place_history']['facts'] == ['mention']
    assert 'facts' not in by_text['Please check **frame 6**.']['place_history']
    assert by_text['Keep the unsaved note.']['place_history']['owner_confirmed'] is True
    for state in ('accepted', 'delivered', 'failed', 'uncertain'):
        receipt = by_text['Receipt-backed reply: ' + state]
        assert receipt['place_history']['delivery_state'] == state
        assert receipt['role'] == ('assistant' if state in {'accepted', 'delivered'} else 'system')
    assert by_text['Legacy **answer**']['role'] == 'assistant'
    assert by_text['Legacy **answer**']['sender_label'] == 'Maya Private'
    assert by_text['Legacy **answer**']['place_history'] == {'kind': 'message', 'delivery_state': 'authored'}
    for row in before:
        if row.get('direction') == 'out':
            assert reference_from_row(row)['author_label'] == 'Ouroboros'
            if row.get('task_id') == task['id']:
                assert not row.get('sender_label') and not row.get('sender_session_id'), 'new v0 replies do not copy the participant'
    assert reference_from_row({'direction':'system', 'sender_label':'Maya Private'})['author_label'] == 'System'
    held, = [row for row in rows if row['text'].startswith('Reply held:')]
    assert held['role'] == 'system'
    assert held['place_history'] == {'kind': 'notice', 'held_text': 'Held reply body stays in its task',
                                     'held': True, 'checked': False}
    assert all(row['text'] not in {'PRIVATE THOUGHT', 'TECHNICAL EVIDENCE'} for row in rows)
    assert not any('matched_wakes' in row['place_history'] for row in rows)
    assert before == [row.row for row in ds.iter_rows(tmp_path, 'chat')]
    if target := os.environ.get('OUROBOROS_TEST_EVIDENCE_ROOT'):
        out = Path(target); out.mkdir(parents=True, exist_ok=True)
        (out / 'place-producer-api.json').write_text(json.dumps({'synthetic':True, 'index':index, 'history':page}, indent=2))
    client.close()


@pytest.mark.parametrize("authenticated,owner", [(False, False), (None, True), (True, True)])
def test_reader_owner_label_respects_captured_actor_authentication(tmp_path, monkeypatch, authenticated, owner):
    """Opus N5: a confirmed address alone cannot label explicitly unverified mail '(you)'."""
    from ouroboros.presence_admission import admit_presence_turn
    client, place, task, event = scenario(tmp_path, monkeypatch)
    actor = {"platform_actor_id": "owner", "display_name": "Owner address"}
    if authenticated is not None:
        actor["authenticated"] = authenticated
    event = replace(event, source_event_id="owner-attestation", actor=actor, text="Captured owner-address mail")
    admission = admit_presence_turn(drive_root=tmp_path, authenticated_transport_skill="bridge",
                                    provider="chat", account_id="qa", conversation_id="design")
    capture_presence_event(admission, event, drive_root=tmp_path)
    before = [row.row for row in ds.iter_rows(tmp_path, "chat")]
    page = client.get("/api/chat/history", params={"chat_id": task["chat_id"]}).json()
    message, = [row for row in page["messages"] if row["text"] == event.text]
    assert message["place_history"]["owner_confirmed"] is owner
    assert before == [row.row for row in ds.iter_rows(tmp_path, "chat")]
    client.close()


def test_heard_facts_are_decoded_after_the_read_transaction(tmp_path, monkeypatch):
    client, _place, task, _event = scenario(tmp_path, monkeypatch)
    watch = watch_transactions(monkeypatch)
    def loads(text):
        assert not watch.open, 'heard facts were decoded inside the read transaction'
        return json.loads(text)
    monkeypatch.setattr(place_history, 'json', SimpleNamespace(loads=loads))
    page = client.get('/api/chat/history', params={'chat_id': task['chat_id']}).json()
    facts = [row['place_history'] for row in page['messages'] if 'place_history' in row]
    assert any(row.get('attention') == {'decision': 'heard', 'reason': 'paused'} and row.get('facts') == ['mention'] for row in facts)
    assert any(row.get('attention') == {'decision': 'focus_started'} and 'facts' not in row for row in facts)
    assert any('dispatch_input' in sql for sql in watch.statements), 'the retained heard facts were read'
    client.close()


@pytest.mark.parametrize('owner', [False, True])
def test_captured_identity_and_receipt_snapshot_reach_the_room_reader(tmp_path, monkeypatch, owner):
    client, place, task, event = scenario(tmp_path, monkeypatch)
    from ouroboros.presence_admission import admit_presence_turn
    identity = {'sends_as': {'id': 'writer', 'label': 'Writer café', 'kind': 'user'},
                'reads_as': [{'id': 'reader', 'label': 'Reader', 'kind': 'user'}]}
    event = replace(event, source_event_id='identity-event', text='New arrival with identity')
    object.__setattr__(event, 'identity', identity)
    admission = admit_presence_turn(drive_root=tmp_path, authenticated_transport_skill='bridge',
                                    provider='chat', account_id='qa', conversation_id='design')
    captured = capture_presence_event(admission, event, drive_root=tmp_path)
    captured['metadata']['focus'].update(identity=identity, identity_owner=owner)
    seed_result(tmp_path, captured['id'], {**captured, 'status': 'running'})
    places.edit_place(tmp_path, place.place_id, 'set_state', author='owner', state='active')
    PresenceDeliveryRecorder(tmp_path).record('bridge', {
        'schema_version': 1, 'delivery_id': 'identity-receipt', 'part_id': '0', 'state': 'delivered',
        'provider': 'chat', 'account_id': 'qa', 'conversation_id': 'design', 'thread_id': '',
        'text': 'Reply with snapshot', 'format': 'markdown', 'message': {},
        'origin': {'kind': 'automatic', 'task_id': captured['id'], 'source_event_id': event.source_event_id},
    })
    before = [entry.row for entry in ds.iter_rows(tmp_path, 'chat')]
    page = client.get('/api/chat/history', params={'chat_id': task['chat_id']}).json()
    by_text = {row['text']: row for row in page['messages'] if 'place_history' in row}
    assert by_text[event.text]['place_history']['identity'] == identity
    assert by_text['Reply with snapshot']['place_history'] == {
        'kind': 'delivery', 'delivery_state': 'delivered',
        'sends_as': identity['sends_as'], 'sends_as_owner': owner,
    }
    assert 'identity' not in by_text['Please check **frame 6**.']['place_history']
    assert by_text['Receipt-backed reply: delivered']['place_history'] == {
        'kind': 'delivery', 'delivery_state': 'delivered'}
    assert before == [entry.row for entry in ds.iter_rows(tmp_path, 'chat')]
    client.close()


def test_old_heard_and_hold_rows_keep_their_absent_fields(tmp_path, monkeypatch):
    client, place, task, _event = scenario(tmp_path, monkeypatch)
    # A pre-F1 heard row has a decision, but no retained event-facts list.
    with ds.transaction(tmp_path, write=True) as conn:
        conn.execute("UPDATE dialogue SET dispatch_input=json_remove(dispatch_input, '$.place_attention.facts') "
                     "WHERE json_extract(dispatch_input, '$.place_attention.reason')='paused'")
        conn.execute("UPDATE dialogue SET dispatch_input=json_set(dispatch_input, '$.place_attention.facts', json('[]')) "
                     "WHERE json_extract(dispatch_input, '$.place_attention.reason')='no_wake_match'")
    ds.append(tmp_path, 'chat', {'ts': '2026-10-22T10:05:00Z', 'direction': 'system',
                               'type': 'place_notice', 'chat_id': task['chat_id'], 'place_id': place.place_id,
                               'text': 'Reply held: historical place is Silent', 'transport': {}})
    page = client.get('/api/chat/history', params={'chat_id': task['chat_id']}).json()
    by_text = {row['text']: row['place_history'] for row in page['messages'] if 'place_history' in row}
    assert by_text['I can still report this while Paused.']['attention'] == {'decision': 'heard', 'reason': 'paused'}
    assert 'facts' not in by_text['I can still report this while Paused.']
    assert by_text['Keep the unsaved note.']['attention'] == {'decision': 'heard', 'reason': 'no_wake_match'}
    assert by_text['Keep the unsaved note.']['facts'] == []
    assert by_text['Reply held: historical place is Silent'] == {'kind': 'notice'}
    assert by_text['Legacy **answer**'] == {'kind': 'message', 'delivery_state': 'authored'}
    client.close()


def _rooms_client(root, monkeypatch):
    monkeypatch.setattr(places_api, 'request_drive_root', lambda _: root)
    place = places.connect_place(root, connector='qa', provider='chat', account_id='qa')
    return TestClient(Starlette(routes=[*places_api.places_routes(),
                                      Route('/api/chat/history', make_chat_history_endpoint(root))])), place


def test_legacy_and_outgoing_only_are_not_claimed_as_discovered(tmp_path, monkeypatch):
    client, place = _rooms_client(tmp_path, monkeypatch)
    for direction in ('in','out'):
        ds.append(tmp_path, 'chat', {'direction':direction, 'chat_id':9000000001, 'source':'presence:chat',
                                   'text':'retained old speech', 'place_id':place.place_id})
    data = client.get(f'/api/places/{place.place_id}/rooms').json()
    assert data['rooms'] == [] and data['coverage']['discovery'] == 'arrivals'
    ds.append(tmp_path, 'chat', {'direction':'in', 'type':'place_arrival', 'chat_id':9000000001,
                               'place_id':place.place_id, 'text':'new arrival', 'ts':'2026-10-25T12:00:00Z'})
    data = client.get(f'/api/places/{place.place_id}/rooms').json()
    assert data['rooms'][0]['label'] is None and data['rooms'][0]['unanswered_count'] is None
    client.close()


def test_absent_sqlite_with_pending_archive_and_imported_gap_are_distinct(tmp_path, monkeypatch):
    client, place = _rooms_client(tmp_path, monkeypatch)
    archive = tmp_path / 'archive'; archive.mkdir()
    archived = {'ts':'2026-10-20T12:00:00Z','direction':'in','chat_id':9000000002,
                'type':'place_arrival','place_id':place.place_id,'text':'archived Place message'}
    (archive / 'chat_20261020T000000.jsonl').write_text('{bad json\n' + json.dumps(archived) + '\n')
    assert not ds.present(tmp_path)
    pending = client.get(f'/api/places/{place.place_id}/rooms')
    assert pending.status_code == 200
    assert pending.json()['coverage']['importing'] is True
    pending_history = client.get('/api/chat/history', params={'chat_id': 9000000002})
    assert pending_history.status_code == 503
    assert pending_history.json()['error'] == 'migrating'
    dialogue_import.run_dialogue_import(tmp_path)
    after = client.get(f'/api/places/{place.place_id}/rooms').json()
    assert after['coverage'] == {'discovery':'arrivals','importing':False,'gaps':True}
    assert [row['room_chat_id'] for row in after['rooms']] == [9000000002]
    history = client.get('/api/chat/history', params={'chat_id': 9000000002}).json()
    assert history['window']['complete'] is False
    assert 'chat_malformed_jsonl' in history['window']['truncated_by']
    assert any(row['text'] == archived['text'] for row in history['messages'])
    assert history['page_cursor'], 'a partial range has its own retry address'
    retried = client.get('/api/chat/history', params={'chat_id': 9000000002, 'cursor': history['page_cursor']}).json()
    assert retried['messages'] == history['messages']
    assert 'malformed_jsonl' in retried['coverage']['spans']['chat']['gaps']
    _preserve('import-and-gap', {'pending_index':pending.json(), 'pending_history':pending_history.json(),
                               'imported_index':after, 'history':history, 'retry':retried})
    client.close()


def test_ordinary_older_pages_are_bounded_without_claiming_gaps(tmp_path, monkeypatch):
    client, place = _rooms_client(tmp_path, monkeypatch)
    for number in range(6):
        ds.append(tmp_path, 'chat', {'ts':f'2026-10-25T12:00:0{number}Z', 'direction':'in', 'chat_id':9000000003,
                                   'type':'place_arrival', 'place_id':place.place_id, 'text':f'Message {number}'})
    cursor = None
    pages = []
    while True:
        params = {'chat_id':9000000003, 'n_human':2, 'n_progress':0}
        if cursor:
            params['cursor'] = cursor
        response = client.get('/api/chat/history', params=params)
        assert response.status_code == 200
        page = response.json(); pages.append(page)
        assert set(page['window']['truncated_by']) <= {'page', 'quota', 'human_quota'}
        assert all(not span or not span['gaps'] for span in page['coverage']['spans'].values())
        assert all(row['place_history'] == {'kind':'message', 'owner_confirmed':False} for row in page['messages'])
        if not page['has_more']:
            break
        assert page['next_cursor'] and page['next_cursor'] != cursor and len(pages) < 6
        cursor = page['next_cursor']
    assert len(pages) > 1
    assert sorted(row['text'] for page in pages for row in page['messages']) == [f'Message {number}' for number in range(6)]
    assert pages[-1]['window']['truncated_by'] == ['page']
    _preserve('ordinary-pages', {'pages':pages})
    client.close()


def _preserve(name, value):
    if target := os.environ.get('OUROBOROS_TEST_EVIDENCE_ROOT'):
        out = Path(target); out.mkdir(parents=True, exist_ok=True)
        (out / (name + '.json')).write_text(json.dumps({'synthetic':True, **value}, indent=2))
