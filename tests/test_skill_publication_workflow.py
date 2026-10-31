"""Version disclosure and owner clearing share existing publication owners."""
from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
import json

import pytest

from ouroboros.gateway import marketplace as api
from ouroboros.marketplace import provenance
from ouroboros.skill_publish_snapshot import CapturedSkillFile
from ouroboros.tools import skill_publish
from tests.test_marketplace_api import _BodyRequest, _stub_marketplace_roots
from tests.test_marketplace_publication_record import _published
from tests.test_skill_publish_transaction import BASE_SHA, _install_transaction_fakes, _snapshot, _submit


def test_clear_only_displayed_publication_preserves_sibling_state_and_republish(tmp_path):
    shown = _published()
    path = provenance.write_publication_record(tmp_path, 'demo', shown)
    provenance.merge_state_record(tmp_path, 'demo', provenance.PUBLICATION_FILENAME, {'future': {'kept': True}})
    grants = path.parent / 'grants.json'
    grants.write_text('{"untouched":true}')
    assert provenance.clear_publication_record(tmp_path, 'demo', shown)
    assert provenance.read_publication_record(tmp_path, 'demo') == (None, None)
    assert json.loads(path.read_text()) == {'schema_version': 1, 'published': None, 'future': {'kept': True}}
    assert grants.read_text() == '{"untouched":true}'
    assert not provenance.clear_publication_record(tmp_path, 'demo', shown)
    newer = _published(pr_number=8, pr_url='https://github.com/hub/project/pull/8')
    provenance.write_publication_record(tmp_path, 'demo', newer)
    before = path.read_bytes()
    with pytest.raises(provenance.PublicationRecordChanged, match='changed'):
        provenance.clear_publication_record(tmp_path, 'demo', shown)
    assert path.read_bytes() == before
    assert provenance.read_publication_record(tmp_path, 'demo') == (newer, None)


def test_clear_absent_record_is_idempotent_without_creating_receipt(tmp_path):
    assert not provenance.clear_publication_record(tmp_path, 'demo', _published())
    assert not (tmp_path / 'state/skills/demo/ouroboroshub.json').exists()


@pytest.mark.parametrize('contents', ['{}', '[]', 'not json', '{"schema_version":1}'])
def test_clear_does_not_repair_or_erase_malformed_existing_record(tmp_path, contents):
    path = tmp_path / 'state/skills/demo/ouroboroshub.json'
    path.parent.mkdir(parents=True)
    path.write_text(contents)
    with pytest.raises(ValueError):
        provenance.clear_publication_record(tmp_path, 'demo', _published())
    assert path.read_text() == contents


def test_clear_api_uses_actual_receipt_and_never_contacts_hub(tmp_path, monkeypatch):
    _stub_marketplace_roots(monkeypatch, tmp_path)
    shown = _published()
    path = provenance.write_publication_record(tmp_path, 'demo', shown)
    before = path.read_bytes()
    stale = _published(version='older')
    response = asyncio.run(api.api_skill_publication_clear(
        _BodyRequest({'expected_published': stale}, {'skill': 'demo'})))
    assert response.status_code == 409 and json.loads(response.body)['code'] == 'publication_changed'
    assert path.read_bytes() == before
    response = asyncio.run(api.api_skill_publication_clear(
        _BodyRequest({'expected_published': shown}, {'skill': 'demo'})))
    assert response.status_code == 200 and json.loads(response.body)['publication_cleared']
    assert provenance.read_publication_record(tmp_path, 'demo') == (None, None)


@pytest.mark.parametrize('name,body', [('..', {'expected_published': {}}), ('demo', {}), ('demo', {'expected_published': []})])
def test_clear_api_rejects_invalid_target_and_missing_displayed_receipt(tmp_path, monkeypatch, name, body):
    _stub_marketplace_roots(monkeypatch, tmp_path)
    response = asyncio.run(api.api_skill_publication_clear(_BodyRequest(body, {'skill': name})))
    assert response.status_code == 400


def test_clear_write_failure_is_visible_and_keeps_receipt(tmp_path, monkeypatch):
    _stub_marketplace_roots(monkeypatch, tmp_path)
    shown = _published()
    path = provenance.write_publication_record(tmp_path, 'demo', shown)
    before = path.read_bytes()
    def broken_write(*args, **kwargs):
        raise OSError('controlled write failure')
    monkeypatch.setattr(provenance, 'update_json_locked', broken_write)
    response = asyncio.run(api.api_skill_publication_clear(
        _BodyRequest({'expected_published': shown}, {'skill': 'demo'})))
    assert response.status_code == 500
    assert 'controlled write failure' in json.loads(response.body)['error']
    assert path.read_bytes() == before


@pytest.mark.parametrize('before,after', [('2.0.0', '1.0.0'), ('0.9.0', '0.10.0'), ('May release', 'nightly blue')])
@pytest.mark.parametrize('model_body', ['', '## Summary\nModel summary\n## Version change\nInvented versions'])
def test_publish_discloses_both_opaque_versions_in_result_and_host_pr_body(tmp_path, monkeypatch, before, after, model_body):
    snapshot = _snapshot()
    manifest_file = CapturedSkillFile.from_bytes('skill.json', json.dumps({'name': 'demo', 'version': after}).encode())
    files = (manifest_file, snapshot.public_files[1])
    snapshot = replace(snapshot, manifest_file=manifest_file, manifest=replace(snapshot.manifest, version=after),
                       full_files=files, public_files=files)
    ctx, _, captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=snapshot, model_body=model_body)
    catalog = {'skills': [{'slug': 'demo', 'version': before, 'files': []}]}
    monkeypatch.setattr(skill_publish, 'fetch_upstream_catalog', lambda *a: (copy.deepcopy(catalog), BASE_SHA))
    result = _submit(ctx)
    assert result['ok'] and result['catalog_version'] == before and result['proposed_version'] == after
    assert f'Catalog version: {json.dumps(before)}' in captured['pr_body']
    assert f'Proposed version: {json.dumps(after)}' in captured['pr_body']
    assert 'Invented versions' not in captured['pr_body']
    assert json.dumps(before) in captured['prompts'][0] and json.dumps(after) in captured['prompts'][0]


def test_same_version_remains_existing_refusal_before_public_mutation(tmp_path, monkeypatch):
    ctx, events, _ = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())
    monkeypatch.setattr(skill_publish, 'fetch_upstream_catalog',
                        lambda *a: ({'skills': [{'slug': 'demo', 'version': '1.0.0', 'files': []}]}, BASE_SHA))
    result = _submit(ctx)
    assert result['reason_code'] == 'catalog_version_exists'
    assert not any(event[0] == 'mutation' for event in events)
