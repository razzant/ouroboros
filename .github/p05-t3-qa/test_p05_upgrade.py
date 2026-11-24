"""Real old-writer/Restart controls plus Host intake during delayed connector publication."""
import asyncio
from contextlib import ExitStack
import socket
from types import SimpleNamespace
import time

import httpx
import pytest
from starlette.testclient import TestClient
from ouroboros import config, dialogue_store, extension_registry_state, places
from ouroboros.agent import OuroborosAgent
from ouroboros.extension_contributions import live_place_declarations, stage_contributions
from ouroboros.gateway.host_service import create_host_service_app
from ouroboros.presence_admission import focus_state
from ouroboros.presence_runner import presence_turn_task_id
from ouroboros.skill_loader import find_skill
from ouroboros.task_results import load_task_result
from supervisor import state, workers
from tests.conversation_carrier_support import join_host_lane_carriers, lone_host_world as lone_host_world
from tests.fixtures_mock_llm import MockLLMServer

from p05_live import ROLE, ROOT, live, publish_fact, publish_step, seed_mock_settings
from p05_place_fixture import (BRIDGE, DIRECT, INBOX, LEGACY, LEGACY_ID, TOKEN, await_on, get_row, old_same_list,
                               pending, seed_binding, sha, turn, turn_payload, typed, wakes)
from p05_proof import PAIRED_FIRST_DM, READINESS, TRANSITIONS

pytestmark = pytest.mark.ui_browser


def arrivals(server, source):
    return sum(row.row.get('client_message_id') == source for row in dialogue_store.iter_rows(server.data, 'chat'))


def task_read(server, ref):
    result = httpx.get(server.url + '/api/tasks/' + ref, timeout=8, trust_env=False)
    if result.status_code == 404:
        return None
    assert result.status_code == 200
    stored = result.json()
    assert {stored[key] for key in ('id', 'task_id') if key in stored} == {ref}
    return stored


def first_dm_envelope(response, ref):
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, dict) and body['ok'] is True and body['turn_ref'] == ref
    assert type(body['continuation_version']) is int and body['continuation_version'] == 1
    assert type(body['delivery_reporting_version']) is int and body['delivery_reporting_version'] == 1
    status, outcome = body['status'], body['outcome']
    assert status in ('completed', 'continuing')
    assert outcome in ('message', 'silent', 'tool_delivered', 'deferred')
    assert body['continuation_ref'] == (ref if status == 'continuing' else '')
    assert all(isinstance(body[key], str) for key in ('text', 'output_ref', 'work_ref'))
    if outcome in ('silent', 'tool_delivered'):
        assert body['text'] == ''
    return status, outcome


@pytest.mark.parametrize('scenario', TRANSITIONS)
def test_real_old_writer_or_ordinary_restart(tmp_path, scenario):
    save_time = TRANSITIONS[scenario]
    old_branch = save_time != 'none'
    with live(tmp_path / 'new', managed=not old_branch) as server:
        if save_time == 'after-connector':
            # The old core must materialize the place before the connector upgrade, without a new-core proof.
            seed_binding(server.data)
        else:
            pending(server.data)
        if old_branch:
            old_same_list(server, tmp_path / 'old', connector_first=save_time == 'after-connector')
        typed(server.data)
        boot = server.start()
        publish_step(scenario, 'server_ready')
        store = places.places_path(server.data)
        before = sha(store)
        first, second = get_row(server), get_row(server)
        review = 'ambiguous' if old_branch else 'pending'
        for field in ('revision', 'wakes', 'wake_review', 'wake_suggestions'):
            assert first[field] == second[field]
        assert sha(store) == before
        assert first['wake_review'] == review and first['wake_suggestions'] == [DIRECT] and wakes(first) == LEGACY
        await_on(server)
        if not old_branch:
            pid = server.proc.pid
            assert server.policy_governs(), 'the real Restart case must start under the exact managed policy'
            publish_step(scenario, 'restart_requested')
            assert server.restart(boot) != boot
            assert server.proc.pid == pid and server.proc.poll() is None and server.policy_governs()
            publish_step(scenario, 'restart_ready')
            resumed = get_row(server)
            assert wakes(resumed) == LEGACY and resumed['wake_review'] == 'pending'
            assert focus_state(server.data) == 'on'
        source = 'p05-' + scenario + '-first-dm'
        ref = presence_turn_task_id(LEGACY_ID, source)
        assert arrivals(server, source) == 0 and task_read(server, ref) is None
        response = turn(server, source)
        publish_step(scenario, 'dm_sent')
        status, outcome = first_dm_envelope(response, ref)
        if old_branch:
            assert (status, outcome) == ('completed', 'silent')
            assert wakes(get_row(server)) == LEGACY and task_read(server, ref) is None
            task, matched = 'absent', []
            publish_step(scenario, 'dm_settled')
        else:
            until = time.monotonic() + 25
            while time.monotonic() < until:
                stored = task_read(server, ref)
                if stored and stored.get('status') in ('completed', 'failed', 'cancelled'):
                    break
                time.sleep(.25)
            else:
                raise AssertionError('admitted DM did not settle')
            assert stored['status'] == 'completed'
            publish_step(scenario, 'dm_settled')
            matched = stored['metadata']['focus']['matched_wakes']
            assert matched == [DIRECT]
            assert wakes(get_row(server)) == [*LEGACY, DIRECT]
            task = 'completed'
        assert arrivals(server, source) == 1
    publish_fact(scenario, {'http': 200, 'ok': True, 'status': status, 'outcome': outcome,
                            'turn_ref_exact': True, 'arrivals': 1, 'task': task, 'matched_wakes': matched,
                            'old_owner_save': old_branch, 'old_save_time': save_time, 'ordinary_restart': not old_branch,
                            'passive_place_reads': True, **({} if old_branch else {'restart_mode': 'managed_same_program'})})
    publish_step(scenario, 'witness_published')


def installed_projection(root):
    """Use the loader's actual staging; delay only the live publication of these rows."""
    loaded = find_skill(root, BRIDGE)
    assert loaded is not None and loaded.enabled
    stage = SimpleNamespace(_staged=SimpleNamespace())
    stage_contributions(stage, loaded, loaded.content_hash)
    assert not stage._staged.contribution_problems
    entry, = stage._staged.contributions
    assert entry['skill'] == BRIDGE and entry['provides'] == 'place' and entry['wakes'] == [DIRECT, INBOX]
    return [entry]


def source_rows(root, source):
    return [item.row for item in dialogue_store.iter_rows(root, 'chat')
            if item.row.get('client_message_id') == source and item.row.get('direction') == 'in']


def bind_inline_world(root, monkeypatch):
    """One disposable world for the real inline agent, including its process-global state readers."""
    monkeypatch.setattr(workers, 'REPO_DIR', ROOT / ROLE)
    monkeypatch.setattr(config, 'DATA_DIR', root)
    monkeypatch.setattr(config, 'SETTINGS_PATH', root / 'settings.json')
    monkeypatch.setenv('OUROBOROS_DATA_DIR', str(root))
    monkeypatch.setenv('OUROBOROS_SETTINGS_PATH', str(root / 'settings.json'))
    monkeypatch.setattr(state, 'DRIVE_ROOT', root)
    monkeypatch.setattr(state, 'STATE_PATH', root / 'state/state.json')
    monkeypatch.setattr(state, 'STATE_LAST_GOOD_PATH', root / 'state/state.last_good.json')
    monkeypatch.setattr(state, 'STATE_LOCK_PATH', root / 'locks/state.lock')
    monkeypatch.setattr(state, 'TOTAL_BUDGET_LIMIT', 10.0)
    state.save_state(state.ensure_state_defaults({'bg_consciousness_enabled': True}))
    assert focus_state(root) == 'on'


def base_silence(client, first, headers, payload, root, source, ref, calls, store, before, publish):
    """The older Host fully settles the ordinary DM as heard before publication, rather than asking for retry."""
    assert first_dm_envelope(first, ref) == ('completed', 'silent')
    row, = source_rows(root, source)
    assert row['task_id'] == ref and row['presence_event_identity']
    assert not calls and load_task_result(root, ref) is None
    saved = [wake.fact for wake in places.place_records(root).places[LEGACY_ID].wakes]
    assert saved == LEGACY and sha(store) == before
    publish()
    replay = client.post('/presence/turn', headers=headers, json=payload)
    assert first_dm_envelope(replay, ref) == ('completed', 'silent')
    assert replay.json() == first.json()
    assert source_rows(root, source) == [row] and not calls and load_task_result(root, ref) is None
    assert sha(store) == before
    publish_fact('base-' + PAIRED_FIRST_DM, {
        'order': 'core-first', 'event_kind': 'plain', 'surface': 'host-asgi-inline',
        'unpublished': {'http': first.status_code, 'status': 'completed', 'outcome': 'silent', 'arrivals': 1,
                        'focus_starts': 0, 'task_absent': True, 'saved_wakes': saved},
        'replay': {'http': replay.status_code, 'same_turn': True, 'outcome': 'silent', 'arrivals': 1,
                   'focus_starts': 0, 'task_absent': True, 'places_unchanged': True},
    })
    publish_step('base-' + PAIRED_FIRST_DM, 'witness_published')


@pytest.mark.usefixtures('lone_host_world')
@pytest.mark.parametrize('witness', READINESS)
def test_first_dm_across_publication(tmp_path, monkeypatch, witness):
    """The real standalone Host and inline carrier, only publication timing controlled; model is loopback."""
    order, kind = READINESS[witness]
    bind_inline_world(tmp_path, monkeypatch)
    store = places.places_path(tmp_path)
    visible = []
    monkeypatch.setattr(extension_registry_state, 'live_contribution_projection', lambda: list(visible))
    if order == 'core-first':
        pending(tmp_path)
        assert store.is_file()
        typed(tmp_path, inbox=True)
    else:
        seed_binding(tmp_path)
        typed(tmp_path, inbox=True)
        assert not store.exists(), 'the new core has not imported the legacy binding'
    records = places.place_records(tmp_path)
    place = records.places[LEGACY_ID]
    assert place.state == 'active' and [wake.fact for wake in place.wakes] == LEGACY
    if ROLE == 'candidate':
        from ouroboros.places_migration import wake_review
        assert wake_review(records, place, [DIRECT, INBOX]) == 'pending'
    else:
        assert witness == PAIRED_FIRST_DM  # the base predates the pending-import proof and custom-signal retry
    before = sha(store)
    staged = installed_projection(tmp_path)
    assert not live_place_declarations()

    def publish():
        visible[:] = staged
        assert live_place_declarations()[0]['wakes'] == [DIRECT, INBOX]

    calls = []
    original_handle = OuroborosAgent.handle_task

    def counted_handle(agent, task):
        calls.append(task['id'])
        return original_handle(agent, task)

    monkeypatch.setattr(OuroborosAgent, 'handle_task', counted_handle)
    source = f'p05-{witness}'
    ref = presence_turn_task_id(LEGACY_ID, source)
    facts = [DIRECT, INBOX] if kind == 'inbox-signal' else [DIRECT]
    payload = turn_payload(source, facts)
    headers = {'X-Skill-Token': TOKEN}
    with MockLLMServer() as llm, ExitStack() as cleanup:
        seed_mock_settings(tmp_path, llm.base_url, inline=True)
        cleanup.callback(join_host_lane_carriers)
        with TestClient(create_host_service_app(tmp_path)) as client:
            assert not source_rows(tmp_path, source) and load_task_result(tmp_path, ref) is None
            deferred = client.post('/presence/turn', headers=headers, json=payload)
            if ROLE == 'base':
                return base_silence(client, deferred, headers, payload, tmp_path, source, ref, calls, store, before,
                                    publish)
            assert deferred.status_code == 429
            retry = deferred.json()
            assert (retry['ok'], retry['code'], retry['disposition'], retry['retry_after']) == (
                False, 'wake_import_declarations_pending', 'retry', 60)
            assert retry['ok'] is False and type(retry['retry_after']) is int and 'field' not in retry
            assert not calls and load_task_result(tmp_path, ref) is None
            early_rows = source_rows(tmp_path, source)
            assert len(early_rows) == (0 if kind == 'inbox-signal' else 1)
            assert sha(store) == before

            publish()
            accepted = client.post('/presence/turn', headers=headers, json=payload)
            publish_step(witness, 'dm_sent')
            status, outcome = first_dm_envelope(accepted, ref)
            assert status == 'completed' and outcome in ('message', 'silent')
            # This fixture model always returns OK. A focus can legitimately choose silence;
            # a connector envelope that does carry speech must carry that exact answer.
            assert accepted.json()['text'] == ('OK' if outcome == 'message' else '')
            join_host_lane_carriers()
            finished = load_task_result(tmp_path, ref)
            assert finished and finished['status'] == 'completed' and calls == [ref]
            meta = finished['metadata']
            matched = meta['focus']['matched_wakes']
            assert matched == facts
            if kind == 'inbox-signal':
                assert meta['focus']['matched_labels'][INBOX] == 'Priority inbox'
            row, = source_rows(tmp_path, source)
            assert row['task_id'] == ref and row['presence_event_identity'] == meta['presence_event_identity']
            if early_rows:
                assert early_rows[0]['presence_event_identity'] == row['presence_event_identity']
            records = places.place_records(tmp_path)
            place = records.places[LEGACY_ID]
            assert place.wakes == (places.WakeItem(LEGACY[0], 'migration'),
                                   places.WakeItem(DIRECT, 'connector'), places.WakeItem(INBOX, 'connector'))
            assert records.migration['presence_bindings_v1'] is True and 'wake_import_open' not in records.migration
            assert wake_review(records, place, [DIRECT, INBOX]) is None
            closed = sha(store)

            repeated = client.post('/presence/turn', headers=headers, json=payload)
            assert first_dm_envelope(repeated, ref) == (status, outcome)
            for key in ('text', 'output_ref', 'work_ref'):
                assert repeated.json()[key] == accepted.json()[key]
            join_host_lane_carriers()
            assert load_task_result(tmp_path, ref)['metadata']['presence_event_identity'] == meta['presence_event_identity']
            assert source_rows(tmp_path, source) == [row] and calls == [ref] and sha(store) == closed

    publish_fact(witness, {
        'order': order, 'event_kind': kind, 'surface': 'host-asgi-inline',
        'unpublished': {'http': deferred.status_code, 'ok': retry['ok'], 'code': retry['code'],
                        'disposition': retry['disposition'], 'retry_after': retry['retry_after'], 'focus_starts': 0,
                        'import': 'pending',
                        'arrivals': len(early_rows), 'task_absent': True, 'wakes_unchanged': True},
        'accepted': {'http': accepted.status_code, 'status': status, 'outcome': outcome, 'turn_ref_exact': True,
                     'task': finished['status'], 'arrivals': 1, 'focus_starts': len(calls), 'matched_wakes': matched,
                     'import': 'completed', 'saved_wakes': [wake.fact for wake in place.wakes]},
        'replay': {'http': repeated.status_code, 'same_turn': True, 'arrivals': 1, 'focus_starts': 1,
                   'places_unchanged': True},
    })
    publish_step(witness, 'witness_published')


def test_public_gateway_dns_verdict(monkeypatch):
    """Ordinary observer responses, served by a local synthetic HTTP stub, reach each subject's real public probe.

    Only NXDOMAIN for exactly the probed name, with no answer records, may say the host is absent; a temporary
    resolver failure, another name's answer or a contradictory NXDOMAIN must stay an unavailable observer.
    """
    from tests.test_telegram_miniapp_companion import gateway

    assert (ROOT / ROLE / 'skills' / 'telegram' / 'scripts' / 'public_gateway.py').samefile(gateway.__file__)
    host = 'p05-name.trycloudflare.com'
    client = httpx.AsyncClient
    outcomes = {}
    address = {'name': host + '.', 'type': 1, 'data': '104.16.230.132'}
    for name, status, asked, answers in [('nxdomain', 3, host, []), ('servfail', 2, host, []),
                                         ('foreign_question', 3, 'p05-other.trycloudflare.com', []),
                                         ('nxdomain_with_answers', 3, host, [address])]:
        seen = []

        def respond(request):
            seen.append(request.url.host)
            if request.url.host == host:
                raise httpx.ConnectError('synthetic lookup failure', request=request) from socket.gaierror(socket.EAI_NONAME, 'fixture')
            assert request.url.host == '1.1.1.1' and request.url.path == '/dns-query'
            assert dict(request.url.params) == {'name': host, 'type': 'A'}
            packet = {'Status': status, 'TC': False, 'Question': [{'name': asked.upper() + '.', 'type': 1}]}
            if answers:
                packet['Answer'] = answers
            return httpx.Response(200, headers={'content-type': 'application/dns-json'}, json=packet)

        monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: client(transport=httpx.MockTransport(respond), **kw))
        result = asyncio.run(gateway.probe_public_gateway_once(f'https://{host}/', SimpleNamespace(returncode=None)))
        assert seen == [host, '1.1.1.1']
        outcomes[name] = result.value
    assert outcomes == {'nxdomain': 'host_absent' if ROLE == 'candidate' else 'observer_unavailable',
                        'servfail': 'observer_unavailable', 'foreign_question': 'observer_unavailable',
                        'nxdomain_with_answers': 'observer_unavailable'}
    publish_fact(f'{ROLE}-public-dns', outcomes)
    publish_step(f'{ROLE}-public-dns', 'witness_published')
