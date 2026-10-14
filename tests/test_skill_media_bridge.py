"""Host-authenticated skill audio dispatch; no real Telegram or model calls."""
from __future__ import annotations

from pathlib import Path
import asyncio

import pytest
from starlette.requests import Request
from starlette.responses import Response
from starlette.testclient import TestClient

from ouroboros import extension_loader
from ouroboros.gateway.host_service import create_host_service_app
from ouroboros.skill_loader import (
    SkillReviewState, compute_content_hash, save_enabled, save_review_state, save_skill_grants,
)
from tests._shared import clean_extension_runtime_state
from tests.test_host_service_api import _seed_token


def _target(root: Path, *, name: str = "voice_target", exposed: bool = True, isolated: bool = False) -> None:
    folder = root / "skills" / "external" / name
    folder.mkdir(parents=True)
    declaration = (
        "media_transcription:\n  version: 1\n  route: transcribe\n"
        "  mime_types: [audio/ogg]\n  max_bytes: 393216\n"
    ) if exposed else (
        "voice_input:\n  version: 1\n  route: transcribe\n"
        "  mime_types: [audio/ogg]\n  max_bytes: 393216\n"
    )
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: local audio\nversion: 0.1.0\n"
        "type: extension\nentry: plugin.py\npermissions: [route]\n"
        + ("dependencies: [dummy_pkg]\n" if isolated else "") + declaration + "---\n# Test\n",
        encoding="utf-8",
    )
    (folder / "plugin.py").write_text(
        "async def recognize(request):\n"
        "    audio = await request.body()\n"
        "    if audio == b'OggSsilent':\n"
        "        return {'text': '', 'status': 'no_speech'}\n"
        "    return {'text': 'TOKEN_LEAK' if request.headers.get('x-skill-token') else "
        "'recognized' if audio == b'OggSvoice' else 'bad audio'}\n"
        "def register(api):\n"
        "    api.register_route('transcribe', recognize, methods=('POST',))\n",
        encoding="utf-8",
    )
    sha = compute_content_hash(folder, manifest_entry="plugin.py")
    save_review_state(root, name, SkillReviewState(status="pass", content_hash=sha))
    save_enabled(root, name, True)
    save_skill_grants(root, name, [], content_hash=sha, requested_keys=[],
                      requested_permissions=["route"], granted_permissions=["route"])
    if isolated:
        from ouroboros.skill_loader import find_skill
        from tests.test_extension_loader import _add_fake_native_dep, _mark_isolated_deps_installed

        loaded = find_skill(root, name)
        assert loaded is not None
        _add_fake_native_dep(loaded)
        _mark_isolated_deps_installed(root, loaded)
    state = extension_loader.reconcile_extension(name, root, lambda: {})
    assert state.get("live_loaded"), state


def test_bridge_requires_caller_grant_and_distinct_target_declaration(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    clean_extension_runtime_state()
    try:
        _seed_token(tmp_path, skill="transport", token="private", permissions=[],
                    manifest_permissions=["inject_chat", "transcribe_audio"])
        _target(tmp_path)
        client = TestClient(create_host_service_app(tmp_path))
        path = "/media/transcribe/voice_target"
        headers = {"X-Skill-Token": "private", "Content-Type": "audio/ogg"}
        assert client.post(path, headers=headers, content=b"OggSvoice").status_code == 403
    finally:
        clean_extension_runtime_state()


def test_bridge_sanitizes_token_and_regates_target(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    clean_extension_runtime_state()
    try:
        _seed_token(tmp_path, skill="transport", token="private", permissions=["inject_chat", "transcribe_audio"],
                    manifest_permissions=["inject_chat", "transcribe_audio"])
        _target(tmp_path)
        client = TestClient(create_host_service_app(tmp_path))
        path = "/media/transcribe/voice_target"
        headers = {"X-Skill-Token": "private", "Content-Type": "audio/ogg"}
        result = client.post(path, headers=headers, content=b"OggSvoice")
        assert result.status_code == 200, result.text
        assert result.json() == {"ok": True, "text": "recognized"}
        wrong_type = client.post(path, headers={**headers, "Content-Type": "text/plain"}, content=b"OggSvoice")
        assert wrong_type.status_code == 415
        assert client.post(path, headers=headers, content=b"OggS" + b"x" * 393216).status_code == 413
        save_enabled(tmp_path, "voice_target", False)
        assert client.post(path, headers=headers, content=b"OggSvoice").status_code == 409
    finally:
        clean_extension_runtime_state()


def test_composer_hint_alone_does_not_grant_cross_skill_call(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    clean_extension_runtime_state()
    try:
        _seed_token(tmp_path, skill="transport", token="private", permissions=["inject_chat", "transcribe_audio"],
                    manifest_permissions=["inject_chat", "transcribe_audio"])
        _target(tmp_path, exposed=False)
        client = TestClient(create_host_service_app(tmp_path))
        result = client.post("/media/transcribe/voice_target",
                             headers={"X-Skill-Token": "private", "Content-Type": "audio/ogg"},
                             content=b"OggSvoice")
        assert result.status_code == 409
    finally:
        clean_extension_runtime_state()


def test_presence_caller_requires_exact_owner_binding_before_audio(tmp_path, monkeypatch):
    from ouroboros.presence_bindings import (
        PresenceBinding, PresenceEndpoint, new_presence_binding_id, save_presence_binding,
    )

    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    clean_extension_runtime_state()
    try:
        _seed_token(tmp_path, skill="transport", token="private",
                    permissions=["presence", "transcribe_audio"],
                    manifest_permissions=["presence", "transcribe_audio"])
        _target(tmp_path)
        binding = save_presence_binding(tmp_path, PresenceBinding(
            new_presence_binding_id(), "transport", "voice_target",
            PresenceEndpoint("telegram", "account-2", "-100123", "topic-47"),
            PresenceEndpoint("telegram", "account-2", "-100123", "topic-47"),
        ))
        client = TestClient(create_host_service_app(tmp_path))
        path = "/media/transcribe/voice_target"
        headers = {"X-Skill-Token": "private", "Content-Type": "audio/ogg",
                   "X-Presence-Binding": binding.binding_id, "X-Media-Provider": "telegram",
                   "X-Media-Account": "account-2", "X-Media-Conversation": "-100123",
                   "X-Media-Thread": "topic-47"}
        wrong_topic = client.post(path, headers={**headers, "X-Media-Thread": "topic-48"}, content=b"OggSvoice")
        assert wrong_topic.status_code == 403
        assert client.post(path, headers={key: val for key, val in headers.items()
                                          if key != "X-Presence-Binding"}, content=b"OggSvoice").status_code == 403
        assert client.post(path, headers=headers, content=b"OggSvoice").json() == {"ok": True, "text": "recognized"}
    finally:
        clean_extension_runtime_state()


def test_disable_during_inference_withholds_late_text(tmp_path, monkeypatch):
    from ouroboros.gateway import skill_media

    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    clean_extension_runtime_state()
    try:
        _seed_token(tmp_path, skill="transport", token="private", permissions=["inject_chat", "transcribe_audio"],
                    manifest_permissions=["inject_chat", "transcribe_audio"])
        _target(tmp_path)
        original = skill_media._dispatch

        async def disabling(*args, **kwargs):
            result = await original(*args, **kwargs)
            save_enabled(tmp_path, "voice_target", False)
            return result

        monkeypatch.setattr(skill_media, "_dispatch", disabling)
        client = TestClient(create_host_service_app(tmp_path))
        response = client.post("/media/transcribe/voice_target",
                               headers={"X-Skill-Token": "private", "Content-Type": "audio/ogg"},
                               content=b"OggSvoice")
        assert response.status_code == 409
        assert "recognized" not in response.text
    finally:
        clean_extension_runtime_state()


def test_bridge_consumes_lazy_asgi_response_instead_of_unopened_body(tmp_path, monkeypatch):
    from ouroboros.gateway import extensions, skill_media

    class Deferred(Response):
        def __init__(self):
            super().__init__(status_code=200)  # body is empty until __call__

        async def __call__(self, scope, receive, send):
            assert scope["path"] == "/api/extensions/voice_target/transcribe"
            assert set(dict(scope["headers"])) == {b"content-type", b"content-length"}
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b'{"text":"from child"}', "more_body": False})

    async def fake_dispatch(_request):
        return Deferred()

    monkeypatch.setattr(extensions, "api_extension_dispatch", fake_dispatch)
    scope = {"type": "http", "method": "POST", "path": "/media/transcribe/voice_target",
             "headers": [], "query_string": b""}
    status, body = asyncio.run(skill_media._dispatch(
        Request(scope), tmp_path, "voice_target", {"route": "transcribe"}, b"OggSvoice", "audio/ogg",
    ))
    assert (status, body) == (200, b'{"text":"from child"}')


@pytest.mark.serial
def test_host_bridge_runs_isolated_extension_response(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    clean_extension_runtime_state()
    try:
        _seed_token(tmp_path, skill="transport", token="private",
                    permissions=["inject_chat", "transcribe_audio"],
                    manifest_permissions=["inject_chat", "transcribe_audio"])
        _target(tmp_path, isolated=True)
        route = extension_loader.list_routes()["/api/extensions/voice_target/transcribe"]
        assert route["out_of_process"] is True
        client = TestClient(create_host_service_app(tmp_path))
        response = client.post("/media/transcribe/voice_target",
                               headers={"X-Skill-Token": "private", "Content-Type": "audio/ogg"},
                               content=b"OggSvoice")
        assert response.status_code == 200, response.text
        assert response.json() == {"ok": True, "text": "recognized"}
    finally:
        clean_extension_runtime_state()


def test_silent_audio_is_a_real_no_speech_result_not_failed_infrastructure(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    clean_extension_runtime_state()
    try:
        _seed_token(tmp_path, skill="transport", token="private",
                    permissions=["inject_chat", "transcribe_audio"],
                    manifest_permissions=["inject_chat", "transcribe_audio"])
        _target(tmp_path)
        client = TestClient(create_host_service_app(tmp_path))
        result = client.post("/media/transcribe/voice_target", content=b"OggSsilent",
                             headers={"X-Skill-Token": "private", "Content-Type": "audio/ogg"})
        assert result.status_code == 200, result.text
        assert result.json() == {"ok": True, "text": "", "status": "no_speech"}
    finally:
        clean_extension_runtime_state()


def test_account_wide_presence_binding_can_transcribe_a_topic(tmp_path, monkeypatch):
    from ouroboros.presence_bindings import (
        PresenceBinding, PresenceEndpoint, new_presence_binding_id, save_presence_binding,
    )

    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    clean_extension_runtime_state()
    try:
        _seed_token(tmp_path, skill="transport", token="private",
                    permissions=["presence", "transcribe_audio"],
                    manifest_permissions=["presence", "transcribe_audio"])
        _target(tmp_path)
        binding = save_presence_binding(tmp_path, PresenceBinding(
            new_presence_binding_id(), "transport", "voice_target",
            PresenceEndpoint("telegram", "account-2", "*", ""),
            PresenceEndpoint("telegram", "account-2", "-100123", "42"),
        ))
        headers = {"X-Skill-Token": "private", "Content-Type": "audio/ogg",
                   "X-Presence-Binding": binding.binding_id, "X-Media-Provider": "telegram",
                   "X-Media-Account": "account-2", "X-Media-Conversation": "-100123",
                   "X-Media-Thread": "42"}
        client = TestClient(create_host_service_app(tmp_path))
        assert client.post("/media/transcribe/voice_target", headers=headers,
                           content=b"OggSvoice").json() == {"ok": True, "text": "recognized"}
        assert client.post("/media/transcribe/voice_target",
                           headers={**headers, "X-Media-Account": "other"},
                           content=b"OggSvoice").status_code == 403
    finally:
        clean_extension_runtime_state()


def test_unicode_reviewed_target_is_quoted_only_on_raw_asgi_path(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    clean_extension_runtime_state()
    try:
        _seed_token(tmp_path, skill="transport", token="private",
                    permissions=["inject_chat", "transcribe_audio"],
                    manifest_permissions=["inject_chat", "transcribe_audio"])
        _target(tmp_path, name="голос")
        client = TestClient(create_host_service_app(tmp_path))
        result = client.post("/media/transcribe/голос", content=b"OggSvoice",
                             headers={"X-Skill-Token": "private", "Content-Type": "audio/ogg"})
        assert result.status_code == 200, result.text
        assert result.json() == {"ok": True, "text": "recognized"}
    finally:
        clean_extension_runtime_state()
