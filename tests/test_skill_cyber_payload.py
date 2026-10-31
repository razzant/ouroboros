"""Review projection does not deny Cyber the bytes of an inert skill resource."""

import hashlib
import importlib.util
import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import config
from ouroboros.skill_loader import SkillPayloadUnreadable, compute_content_hash, load_skill
from ouroboros.skill_review_packs import (
    _SkillBinaryPayload, _SkillFileUnreadable, _read_skill_file, frozen_payload_files, frozen_payload_source, payload_pack,
)


@pytest.fixture(params=["pro", "cyber_pro"])
def access(request, monkeypatch):
    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline(request.param)
    # Caller-controlled env cannot elevate the effective boot mode.
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    yield request.param
    config.reset_runtime_mode_baseline_for_tests()


@pytest.mark.parametrize("name", [".env", "prod.env", "credentials.json", "id_rsa"])
def test_sensitive_named_inert_resource_is_hashed_and_read_in_cyber(tmp_path, access, name):
    skill = tmp_path / "demo"
    skill.mkdir()
    manifest = b'{"name":"demo","description":"Resource fixture","type":"instruction"}'
    payload = b"INERT_TEST_RESOURCE=first\n"
    (skill / "skill.json").write_bytes(manifest)
    resource = skill / name
    resource.write_bytes(payload)
    if access == "pro":
        with pytest.raises(SkillPayloadUnreadable, match="credential filename"):
            compute_content_hash(skill)
        return
    from ouroboros.skill_identity import release_digest_from_file_digests

    expected = release_digest_from_file_digests(
        (path.name, hashlib.sha256(path.read_bytes()).digest()) for path in skill.iterdir())
    content_hash = compute_content_hash(skill)
    assert content_hash == expected
    pack = payload_pack(frozen_payload_files(skill))
    assert f"### {name}\n" in pack and payload.decode() in pack
    assert not load_skill(skill, tmp_path / "data").load_error
    resource.write_bytes(b"INERT_TEST_RESOURCE=second\n")
    assert compute_content_hash(skill) != content_hash
    # The frozen copy a seat reads is bound to the reviewed hash: bytes that moved past it are unavailable.
    moved = frozen_payload_source(tmp_path / "data", SimpleNamespace(name="demo", skill_dir=skill,
                                                                  manifest=SimpleNamespace(entry="", scripts=[])), content_hash)
    assert "unavailable" in moved and not moved.get("view")


@pytest.mark.parametrize("payload", [
    b"\x7fELF" + b"\0" * 16,  # Valid UTF-8 bytes still represent a binary-format fixture.
    b"MZ\x90\0" + b"\xff" * 16,
    b"\xcf\xfa\xed\xfe" + b"\0" * 16,
    importlib.util.MAGIC_NUMBER + b"\0" * 16,
])
def test_native_magic_resource_has_exact_descriptor_not_decoded_text(tmp_path, access, payload):
    resource = tmp_path / "notes.txt"  # The suffix does not determine binary content.
    resource.write_bytes(payload)
    content_hash = compute_content_hash(tmp_path)  # Native bytes were already hashable.
    if access == "pro":
        with pytest.raises(_SkillBinaryPayload):
            _read_skill_file(resource)
        return
    text, digest, descriptor = _read_skill_file(resource)
    assert text is None and digest == hashlib.sha256(payload).digest()
    assert descriptor["path"] == "notes.txt" and descriptor["size"] == len(payload)
    assert descriptor["sha256"] == hashlib.sha256(payload).hexdigest()
    assert descriptor["format_from_magic"]
    pack = payload_pack(frozen_payload_files(tmp_path))
    assert "descriptor only, content not inlined" in pack
    assert descriptor["sha256"] in pack and "\0" not in pack
    resource.write_bytes(payload + b"\0")
    assert compute_content_hash(tmp_path) != content_hash


def test_binary_content_and_sensitive_name_share_one_cyber_hash_surface(tmp_path, access):
    resource = tmp_path / "credentials.json"
    payload = b"\x7fELF" + b"\0" * 16
    resource.write_bytes(payload)
    if access == "pro":
        with pytest.raises(SkillPayloadUnreadable):
            compute_content_hash(tmp_path)
        return
    compute_content_hash(tmp_path)  # native bytes were already hashable
    pack = payload_pack(frozen_payload_files(tmp_path))
    assert hashlib.sha256(payload).hexdigest() in pack
    assert "credentials.json (binary file" in pack


def test_real_unreadable_source_stays_an_error(tmp_path, access, monkeypatch):
    resource = tmp_path / "resource.dat"
    resource.write_bytes(b"real bytes")
    original = pathlib.Path.read_bytes

    def unavailable(path):
        if path == resource:
            raise OSError("fixture read failed")
        return original(path)

    monkeypatch.setattr(pathlib.Path, "read_bytes", unavailable)
    with pytest.raises(_SkillFileUnreadable, match="fixture read failed"):
        _read_skill_file(resource)


def test_binary_manifest_is_not_fabricated_as_supported_text(tmp_path, access):
    (tmp_path / "SKILL.md").write_bytes(b"\xff\xfe\x80opaque manifest")
    loaded = load_skill(tmp_path, tmp_path / "state-root")
    assert loaded is not None and loaded.load_error
    assert "UnicodeDecodeError" in loaded.load_error
    assert not loaded.content_hash
    assert not loaded.available_for_execution
    # Unsupported manifest parsing does not prevent raw resource inspection.
    text, digest, descriptor = _read_skill_file(tmp_path / "SKILL.md")
    assert text is None and descriptor["sha256"] == digest.hex()
    assert json.loads(json.dumps(descriptor))["path"] == "SKILL.md"
