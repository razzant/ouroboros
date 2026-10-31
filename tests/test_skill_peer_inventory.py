from __future__ import annotations

import dataclasses
import json
import os

import pytest

from ouroboros import skill_loader
from ouroboros import skill_peer_inventory as peer_inventory
from ouroboros.skill_loader import save_enabled
from ouroboros.skill_peer_inventory import discover_skill_peers


def _manifest(name: str, *, conflicts=()) -> str:
    conflict_text = "[" + ", ".join(json.dumps(item) for item in conflicts) + "]"
    return (
        "---\n"
        f"name: {name}\n"
        "description: peer fixture\n"
        "version: 1.0.0\n"
        "type: instruction\n"
        f"conflicts: {conflict_text}\n"
        "---\n"
        "body\n"
    )


def test_peer_projection_preserves_conflicts_without_payload_hashing(tmp_path, monkeypatch):
    drive = tmp_path / "drive"
    skills = drive / "skills"
    first = skills / "alpha"
    second = skills / "beta"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "SKILL.md").write_text(_manifest("display-alpha", conflicts=("beta",)), encoding="utf-8")
    (first / "payload.bin").write_bytes(b"large payload")
    (second / "SKILL.md").write_text(_manifest("display-beta"), encoding="utf-8")
    save_enabled(drive, "alpha", True)
    save_enabled(drive, "beta", True)

    def fail_hash(*_args, **_kwargs):
        raise AssertionError("peer inventory must not hash payloads")

    monkeypatch.setattr(skill_loader, "compute_content_hash", fail_hash)
    peers = discover_skill_peers(drive, repo_path="")
    by_name = {peer.name: peer for peer in peers}
    assert by_name["alpha"].enabled is True
    assert by_name["alpha"].conflicts == ("beta",)
    assert by_name["beta"].enabled is True
    assert all(not peer.identity_collision for peer in peers)


def test_peer_projection_keeps_collision_and_malformed_as_disabled_placeholders(tmp_path):
    drive = tmp_path / "drive"
    skills = drive / "skills"
    (skills / "same").mkdir(parents=True)
    (skills / "same").joinpath("SKILL.md").write_text(_manifest("one"), encoding="utf-8")
    # A second canonical location is supplied by the user checkout.
    checkout = tmp_path / "checkout"
    (checkout / "same").mkdir(parents=True)
    (checkout / "same").joinpath("SKILL.md").write_text(_manifest("two"), encoding="utf-8")
    (skills / "broken").mkdir(parents=True)
    (skills / "broken" / "SKILL.md").write_text("---\nname: [\n---\n", encoding="utf-8")

    peers = discover_skill_peers(drive, repo_path=str(checkout))
    same = [peer for peer in peers if peer.name == "same"]
    assert len(same) == 2
    assert all(peer.identity_collision and not peer.enabled and not peer.conflicts for peer in same)
    broken = next(peer for peer in peers if peer.name == "broken")
    assert broken.manifest_error
    assert broken.enabled is False
    assert broken.conflicts == ()


def test_explicit_empty_locations_do_not_repeat_discovery(tmp_path, monkeypatch):
    import ouroboros.skill_peer_inventory as peer_inventory

    def unexpected(*args, **kwargs):
        raise AssertionError("known-empty locations must not fall back to discovery")

    monkeypatch.setattr(skill_loader, "_skill_location_inventory", unexpected)
    monkeypatch.setattr(peer_inventory, "_skill_location_inventory", unexpected)
    assert skill_loader.discover_skill_identity(tmp_path, "missing", locations=()) == []
    assert discover_skill_peers(tmp_path, locations=()) == ()


def _counted(monkeypatch):
    peer_inventory._peer_manifest_facts.cache_clear()
    reads, parses = [], []
    real_read, real_parse = peer_inventory._manifest_text_for_dir, peer_inventory.parse_skill_manifest_text
    monkeypatch.setattr(peer_inventory, "_manifest_text_for_dir", lambda d: reads.append(d) or real_read(d))
    monkeypatch.setattr(peer_inventory, "parse_skill_manifest_text", lambda t: parses.append(t) or real_parse(t))
    return reads, parses


def test_identical_peer_text_is_parsed_once_and_an_equal_size_edit_is_seen_at_once(tmp_path, monkeypatch):
    drive = tmp_path / "drive"
    (drive / "skills" / "alpha").mkdir(parents=True)
    manifest = drive / "skills" / "alpha" / "SKILL.md"
    manifest.write_text(_manifest("alpha", conflicts=("beta_1",)), encoding="utf-8")
    save_enabled(drive, "alpha", True)
    reads, parses = _counted(monkeypatch)

    def conflicts():
        (peer,) = discover_skill_peers(drive, repo_path="")
        return peer.conflicts

    assert conflicts() == conflicts() == ("beta_1",)
    assert (len(reads), len(parses)) == (2, 1)  # the bytes are read every time
    stat, raw = manifest.stat(), manifest.read_bytes()
    manifest.write_bytes(raw.replace(b"beta_1", b"gamma1"))
    os.utime(manifest, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert manifest.stat().st_size == stat.st_size and manifest.stat().st_mtime_ns == stat.st_mtime_ns
    assert conflicts() == ("gamma1",)
    assert (len(reads), len(parses)) == (3, 2)


def test_a_failed_peer_parse_is_attempted_again_and_never_stored(tmp_path, monkeypatch):
    drive = tmp_path / "drive"
    (drive / "skills" / "broken").mkdir(parents=True)
    manifest = drive / "skills" / "broken" / "SKILL.md"
    manifest.write_text("---\nname: [\n---\n", encoding="utf-8")
    _reads, parses = _counted(monkeypatch)
    for _ in range(2):
        (peer,) = discover_skill_peers(drive, repo_path="")
        assert peer.manifest_error == "SkillManifestError" and not peer.enabled and not peer.conflicts
    assert len(parses) == 2 and peer_inventory._peer_manifest_facts.cache_info().currsize == 0
    manifest.write_text(_manifest("broken"), encoding="utf-8")
    (peer,) = discover_skill_peers(drive, repo_path="")
    assert not peer.manifest_error


def test_reused_facts_are_frozen_bounded_and_read_the_host_platform_at_each_call(monkeypatch):
    peer_inventory._peer_manifest_facts.cache_clear()
    text = _manifest("seed").replace("type: instruction\n", "type: instruction\nos: darwin\n")
    first = peer_inventory._peer_manifest_facts(text)
    monkeypatch.setattr("ouroboros.contracts.skill_manifest.host_platform", lambda: "darwin")
    assert first.applies_to_platform()
    monkeypatch.setattr("ouroboros.contracts.skill_manifest.host_platform", lambda: "win32")
    again = peer_inventory._peer_manifest_facts(text)
    assert again is first and not again.applies_to_platform()
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.conflicts = ("other",)
    for index in range(peer_inventory.PEER_PARSE_REUSE_ENTRIES + 1):
        peer_inventory._peer_manifest_facts(_manifest(f"peer_{index}"))
    info = peer_inventory._peer_manifest_facts.cache_info()
    assert info.currsize == info.maxsize == peer_inventory.PEER_PARSE_REUSE_ENTRIES
