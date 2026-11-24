"""Synthetic legacy Places fixture. e758 does its own real same-list save in a separate home."""
import hashlib
from pathlib import Path
import shutil
import time

import httpx
from ouroboros import places
from ouroboros.gateway.host_service import AUTH_TOKEN_FILENAME
from ouroboros.presence_admission import focus_state
from ouroboros.presence_bindings import PresenceBinding, PresenceEndpoint, conversation_key
from ouroboros.skill_loader import compute_content_hash, save_enabled, save_review_state, save_skill_grants, SkillReviewState
from ouroboros.utils import atomic_write_json
from tests.frozen_presence_bindings import save_presence_binding
from tests.test_host_service_api import _seed_token

from p05_live import live

BRIDGE = 'qa-typed-bridge'
ROOM = 'p05-synthetic-room'
ACCOUNT = 'p05-synthetic-account'
TOKEN = 'p05-synthetic-local-test-only'
LEGACY_ID = 'd' * 32
MIND = 'p05-fixture-mind'
LEGACY = ['legacy_addressed']
DIRECT = 'direct_message'
INBOX = 'signal.priority_inbox'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_binding(root):
    _seed_token(root, skill=BRIDGE, token=TOKEN, permissions=['presence'], manifest_permissions=['presence'])
    mind = root / 'skills' / 'external' / MIND
    mind.mkdir()
    (mind / 'SKILL.md').write_text(f'''---
name: {MIND}
description: Synthetic migration behavior
version: 0.1.0
type: instruction
presence:
  instructions: Answer briefly or finish silently in this isolated test.
---
# QA
''', encoding='utf-8')
    origin = PresenceEndpoint('slack', ACCOUNT, ROOM)
    save_presence_binding(root, PresenceBinding(LEGACY_ID, BRIDGE, MIND, origin, origin))


def pending(root):
    seed_binding(root)
    place, = places.load_places(root)
    assert place.place_id == LEGACY_ID and place.connector == BRIDGE
    assert [w.fact for w in place.wakes] == LEGACY
    if place.state != 'active':
        places.edit_place(root, place.place_id, 'set_state', author='owner', state='active')


def typed(root, *, inbox=False, enabled=True):
    """Install the inert connector update on disk; live publication is separate."""
    skill = root / 'skills' / 'external' / BRIDGE
    manifest = skill / 'SKILL.md'
    old = manifest.read_text(encoding='utf-8')
    assert 'contributes:' not in old and old.count('---\n# Test\n') == 1
    wake_list = ', '.join([DIRECT, INBOX] if inbox else [DIRECT])
    facts = (f'\n      - {DIRECT}\n      - id: {INBOX}\n        label: Priority inbox' if inbox else f' [{DIRECT}]')
    release = f'''contributes:
  - id: place
    provides: place
    label: QA inbox
    destination: vendor:Slack
    wakes: [{wake_list}]
    facts:{facts}
'''
    manifest.write_text(old.replace('---\n# Test\n', release + '---\n# Test\n'), encoding='utf-8')
    content = compute_content_hash(skill, manifest_entry='plugin.py')
    save_review_state(root, BRIDGE, SkillReviewState(status='pass', content_hash=content))
    save_skill_grants(root, BRIDGE, [], content_hash=content, requested_keys=[], requested_permissions=['presence'],
                      granted_permissions=['presence'])
    atomic_write_json(root / 'state' / 'skills' / BRIDGE / AUTH_TOKEN_FILENAME,
                      {'token': TOKEN, 'content_hash': content, 'issued_at': 'now'})
    if not enabled:
        save_enabled(root, BRIDGE, False)


def get_row(server):
    response = httpx.get(server.url + '/api/places', timeout=10, trust_env=False)
    assert response.status_code == 200
    row, = [item for item in response.json()['places'] if item['place_id'] == LEGACY_ID]
    return row


def wakes(place):
    return [item['fact'] for item in place['wakes']]


def old_same_list(candidate, old_root, *, connector_first=False):
    """Return the stopped old core's Places document after a real same-list save, on either connector release."""
    store = places.places_path(candidate.data)
    assert store.exists() == (not connector_first)
    initial_digest = None if connector_first else sha(store)
    with live(old_root, 'old') as old:
        names = ('state/presence_bindings.json', f'skills/external/{BRIDGE}',
                 f'skills/external/{MIND}', f'state/skills/{BRIDGE}')
        for name in (names if connector_first else ('state/places.json', *names)):
            src, dst = candidate.data / name, old.data / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            assert not dst.exists()
            (shutil.copytree if src.is_dir() else shutil.copyfile)(src, dst)
        old_store = places.places_path(old.data)
        if connector_first:
            assert not old_store.exists()
            old.start()
            initial = get_row(old)
            assert wakes(initial) == LEGACY and DIRECT not in initial['wake_suggestions']
            old.stop()
            assert old_store.is_file()
            typed(old.data)
        else:
            assert sha(old_store) == initial_digest
        old.start()
        current = get_row(old)
        assert wakes(current) == LEGACY
        if connector_first:
            assert current['wake_suggestions'] == [DIRECT]  # the old core offers the new fact before the explicit save
        response = httpx.post(old.url + '/api/places/' + LEGACY_ID,
                              json={'wakes': LEGACY, 'expected_revision': current['revision']},
                              timeout=15, trust_env=False)
        assert response.status_code == 200 and response.json()['ok'] is True
        saved = response.json()['place']
        assert saved['place_id'] == LEGACY_ID and wakes(saved) == LEGACY
        assert wakes(get_row(old)) == LEGACY
        # An older body may retire the pending proof in its read path before applying the same wakes.
        # What matters here is its successful POST and the new body's real ambiguous readback.
    assert (not store.exists()) if connector_first else (sha(store) == initial_digest)
    shutil.copyfile(old_store, store)


def await_on(server):
    response = httpx.post(server.url + '/api/command', json={'cmd': '/bg start'}, timeout=12, trust_env=False)
    assert response.status_code == 200 and response.json()['status'] == 'ok'
    for _ in range(70):
        if focus_state(server.data) == 'on':
            return
        time.sleep(.15)
    raise AssertionError('synthetic switch did not read back On')


def turn_payload(source, facts=(DIRECT,)):
    return {'binding_id': LEGACY_ID, 'staged_files': [], 'continuation_version': 1, 'delivery_reporting_version': 1,
            'event': {'source_event_id': source, 'provider': 'slack', 'account_id': ACCOUNT, 'conversation_id': ROOM,
                      'thread_id': '', 'conversation_key': conversation_key('slack', ACCOUNT, ROOM),
                      'actor': {'platform_actor_id': 'qa-fixture-person', 'display_name': 'QA guest'},
                      'conversation': {}, 'message': {}, 'text': 'Synthetic first direct message.', 'facts': list(facts)}}


def turn(server, source):
    return httpx.post(server.host + '/presence/turn', headers={'X-Skill-Token': TOKEN},
                      json=turn_payload(source), timeout=75, trust_env=False)
