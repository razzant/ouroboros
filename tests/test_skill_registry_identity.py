"""A registry (ClawHub) release keeps one identity from staging to landing, with or without declared
dependencies, and every reader that hashes a payload agrees on the bucket's per-install notes."""
import hashlib
import io
import json
import zipfile

import pytest

from ouroboros.skill_identity import PER_INSTALL_NOTE_BUCKETS
from ouroboros.skill_install_record import read_install_record
from ouroboros.skill_loader import find_skill, load_skill
from ouroboros.skill_publish_snapshot import capture_skill_publish_candidate
from ouroboros.skill_review_packs import frozen_payload_files, payload_pack


def _registry(tmp_path, monkeypatch, dependencies):
    from ouroboros.marketplace import install
    from ouroboros.marketplace.clawhub import ClawHubArchive, ClawHubSkillSummary
    drive = tmp_path / 'data'
    monkeypatch.setattr('ouroboros.config.SETTINGS_PATH', drive / 'settings.json')
    monkeypatch.delenv('OUROBOROS_SKILLS_REPO_PATH', raising=False)
    monkeypatch.setenv('OUROBOROS_REVIEW_ENFORCEMENT', 'blocking')
    row = {'id': 'clawhub', 'kind': 'registry', 'name': 'ClawHub', 'trust': 'review', 'confirmed': True}
    monkeypatch.setenv('OUROBOROS_SKILL_SOURCES', json.dumps([row]))
    latest = {'version': '1.0.0'}
    monkeypatch.setattr(install, '_registry_info', lambda slug: ClawHubSkillSummary(
        slug=slug, latest_version=latest['version']))

    def download(slug, *, version=None):
        version = version or latest['version']
        front = {'name': 'demo', 'description': 'Real adapted registry payload', 'version': version}
        if dependencies:
            front['metadata'] = {'openclaw': {'install': [{'kind': 'pip', 'package': 'requests'}]}}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('SKILL.md', '---\n' + json.dumps(front) + '\n---\nDemo.\n')
        content = buffer.getvalue()
        return ClawHubArchive(slug=slug, version=version, content=content,
                              sha256=hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(install, '_registry_download', download)
    return drive, install, latest


@pytest.mark.parametrize('dependencies', [False, True], ids=['no-dependencies', 'declared-dependencies'])
def test_real_registry_install_record_matches_landed_payload(tmp_path, monkeypatch, dependencies):
    drive, storage, _ = _registry(tmp_path, monkeypatch, dependencies)
    result = storage.install_skill(drive, tmp_path, slug='demo', auto_review=False,
        install_record={'source': {'id': 'clawhub', 'kind': 'registry'}, 'basis': 'trusted'})
    assert result.ok, result.error
    loaded = find_skill(drive, 'demo')
    assert not loaded.load_error
    record = read_install_record(drive, 'demo')
    assert record['release']['digest'] == loaded.content_hash
    assert loaded.review.gate_for(loaded.content_hash)['executable_review']


@pytest.mark.parametrize('dependencies', [False, True], ids=['no-dependencies', 'declared-dependencies'])
def test_registry_rejection_records_the_digest_actually_reviewed(tmp_path, monkeypatch, dependencies):
    from ouroboros import skill_source_install as installer
    from ouroboros.skill_loader import SkillReviewState, save_review_state
    drive, storage, latest = _registry(tmp_path, monkeypatch, dependencies)
    assert storage.install_skill(drive, tmp_path, slug='demo', auto_review=False).ok
    latest['version'] = '2.0.0'
    assert installer.update_installed(drive, 'demo', mode='replace')['needs_review']
    digest = find_skill(drive, 'demo').content_hash
    save_review_state(drive, 'demo', SkillReviewState(status='blockers', content_hash=digest))
    assert installer.settle_pending_update(drive, 'demo')['status'] == 'restored'
    assert read_install_record(drive, 'demo')['update_rejected']['digest'] == digest


@pytest.mark.parametrize('dependencies', [False, True], ids=['no-dependencies', 'declared-dependencies'])
def test_registry_owner_skip_accepts_the_confirmed_real_payload(tmp_path, monkeypatch, dependencies):
    from ouroboros import skill_source_install as installer
    from ouroboros.skill_loader import SkillReviewState, save_review_state
    drive, storage, latest = _registry(tmp_path, monkeypatch, dependencies)
    # Dependency execution is outside this identity test; it must not install packages.
    monkeypatch.setattr('ouroboros.skill_dependencies.ensure_skill_dependencies',
                        lambda *a, **kw: ('installed', ''))
    assert storage.install_skill(drive, tmp_path, slug='demo', auto_review=False).ok
    latest['version'] = '2.0.0'
    assert installer.update_installed(drive, 'demo', mode='replace')['needs_review']
    digest = find_skill(drive, 'demo').content_hash
    save_review_state(drive, 'demo', SkillReviewState(status='blockers', content_hash=digest))
    assert installer.settle_pending_update(drive, 'demo')['status'] == 'restored'
    rejected = read_install_record(drive, 'demo')['update_rejected']
    result = installer.update_installed(drive, 'demo', mode='replace',
                                       digest=rejected['digest'], skip_review=True)
    assert result['acceptance']['accepted'], result


@pytest.mark.parametrize('note,bucket', sorted(PER_INSTALL_NOTE_BUCKETS.items()))
@pytest.mark.parametrize('own_bucket', [False, True])
def test_loader_review_and_publication_agree_for_bucket_notes(tmp_path, note, bucket, own_bucket):
    directory = tmp_path / 'data' / 'skills' / (bucket if own_bucket else 'user') / 'demo'
    directory.mkdir(parents=True)
    (directory / 'SKILL.md').write_text('---\nname: demo\ndescription: Probe\ntype: instruction\n---\nDemo.\n', encoding='utf-8')
    (directory / note).write_text(json.dumps({'installed_at': 'synthetic',
        'install_specs': {'raw': [{'kind': 'pip', 'package': 'requests'}]}}), encoding='utf-8')
    loaded = load_skill(directory, tmp_path / 'data')
    assert loaded and not loaded.load_error
    assert capture_skill_publish_candidate(loaded).content_hash == loaded.content_hash
    assert payload_pack(frozen_payload_files(directory, manifest_entry=loaded.manifest.entry,
                                             manifest_scripts=loaded.manifest.scripts))
