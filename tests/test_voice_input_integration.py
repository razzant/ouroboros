"""Live extension catalog/dispatch seam; no microphone, model or review bypass."""
from __future__ import annotations

from ouroboros.contracts.skill_manifest import SkillManifestError, parse_skill_manifest_text
from tests._shared import clean_extension_runtime_state
from tests.test_extensions_api import _make_client, _stop_patches, _write_ext


def test_voice_manifest_optional_and_strict():
    old = parse_skill_manifest_text('---\nname: old\ntype: instruction\n---\nhello')
    assert old.voice_input is None
    valid = '''---
name: voice
type: extension
entry: plugin.py
permissions: [route]
voice_input:
  version: 1
  route: transcribe
  mime_types: [audio/webm, audio/mpeg]
  max_bytes: 393216
---
voice'''
    assert 'audio/mpeg' in parse_skill_manifest_text(valid).voice_input['mime_types']
    transcriber = parse_skill_manifest_text(valid.replace('voice_input:', 'media_transcription:'))
    assert 'audio/mpeg' in transcriber.media_transcription['mime_types']
    for broken in (valid.replace('max_bytes: 393216', 'max_bytes: 524289'),
                   valid.replace('version: 1', 'version: 2'),
                   valid.replace('permissions: [route]', 'permissions: []')):
        try:
            parse_skill_manifest_text(broken)
        except SkillManifestError:
            pass
        else:
            raise AssertionError('invalid voice declaration was accepted')


def test_live_reviewed_voice_route_advertises_and_revokes(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_enabled, save_review_state

    clean_extension_runtime_state()
    skills = tmp_path / 'skills'
    skill_dir = _write_ext(skills, 'voice_fixture', permissions=['route'], plugin=(
        "def register(api):\n"
        "    api.register_route('transcribe', lambda request: {'text': 'test words'}, methods=('POST',))\n"
    ))
    manifest = skill_dir / 'SKILL.md'
    text = manifest.read_text(encoding='utf-8').replace(
        'permissions: ["route"]\n',
        'permissions: ["route"]\nvoice_input:\n  version: 1\n  route: transcribe\n  mime_types: [audio/webm]\n  max_bytes: 393216\n',
    )
    manifest.write_text(text, encoding='utf-8')
    monkeypatch.setenv('OUROBOROS_SKILLS_REPO_PATH', str(skills))
    client, drive, patches = _make_client(tmp_path, monkeypatch)
    try:
        sha = compute_content_hash(skill_dir, manifest_entry='plugin.py')
        save_review_state(drive, 'voice_fixture', SkillReviewState(status='pass', content_hash=sha))
        save_enabled(drive, 'voice_fixture', True)
        state = extension_loader.reconcile_extension('voice_fixture', drive, lambda: {}, repo_path=str(skills))
        assert state.get('live_loaded'), state
        row = next(s for s in client.get('/api/extensions').json()['skills'] if s['name'] == 'voice_fixture')
        assert row['voice_input']['route_url'] == '/api/extensions/voice_fixture/transcribe'
        assert client.post(row['voice_input']['route_url'], content=b'fixture').status_code == 200
        save_enabled(drive, 'voice_fixture', False)
        row = next(s for s in client.get('/api/extensions').json()['skills'] if s['name'] == 'voice_fixture')
        assert 'voice_input' not in row
        assert client.post('/api/extensions/voice_fixture/transcribe', content=b'fixture').status_code in (404, 409)
    finally:
        _stop_patches(patches)
        clean_extension_runtime_state()
