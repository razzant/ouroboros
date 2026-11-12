from __future__ import annotations

import dataclasses
import json
import os
import shutil

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
    peer_inventory._HELD_FACTS.clear()  # a new process starts empty
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
    assert len(parses) == 2 and peer_inventory._HELD_FACTS[(str(drive), "")] == {}
    manifest.write_text(_manifest("broken"), encoding="utf-8")
    (peer,) = discover_skill_peers(drive, repo_path="")
    assert not peer.manifest_error


def _held(drive, repo_path=""):
    return peer_inventory._HELD_FACTS[(str(drive), repo_path)]


def test_held_facts_are_frozen_and_read_the_host_platform_at_each_call(tmp_path, monkeypatch):
    drive = tmp_path / "drive"
    (drive / "skills" / "seed").mkdir(parents=True)
    (drive / "skills" / "seed" / "SKILL.md").write_text(
        _manifest("seed").replace("type: instruction\n", "type: instruction\nos: darwin\n"), encoding="utf-8")
    _counted(monkeypatch)
    discover_skill_peers(drive, repo_path="")
    ((_text, first),) = _held(drive).values()
    monkeypatch.setattr("ouroboros.contracts.skill_manifest.host_platform", lambda: "darwin")
    assert first.applies_to_platform()
    monkeypatch.setattr("ouroboros.contracts.skill_manifest.host_platform", lambda: "win32")
    discover_skill_peers(drive, repo_path="")
    ((_text, again),) = _held(drive).values()
    assert again is first and not again.applies_to_platform()
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.conflicts = ("other",)


def test_any_number_of_peers_is_parsed_once_and_each_location_keeps_only_its_current_text(tmp_path, monkeypatch):
    """#1217: 140 distinct texts outran the former 128 reused ones, so every observation parsed them all again.
    Now each is parsed once; an edit, a removal and a rename each leave only the current install held."""
    drive = tmp_path / "drive"
    names = [f"peer_{index:03d}" for index in range(140)]
    for name in names:
        (drive / "skills" / name).mkdir(parents=True)
        (drive / "skills" / name / "SKILL.md").write_text(_manifest(name), encoding="utf-8")
    reads, parses = _counted(monkeypatch)
    assert len(discover_skill_peers(drive, repo_path="")) == len(discover_skill_peers(drive, repo_path="")) == 140
    assert (len(reads), len(parses)) == (280, 140)  # every manifest read twice, each text parsed once
    edited, removed, renamed = (drive / "skills" / name for name in names[:3])
    (edited / "SKILL.md").write_text(_manifest(names[0], conflicts=("beta",)), encoding="utf-8")
    shutil.rmtree(removed)
    renamed.rename(renamed.with_name("peer_new"))
    parses.clear()
    peers = {peer.name: peer for peer in discover_skill_peers(drive, repo_path="")}
    assert peers[names[0]].conflicts == ("beta",) and names[1] not in peers and names[2] not in peers
    assert len(parses) == 2  # the edited text and the renamed location's text, nothing else
    assert set(_held(drive)) == {peer.skill_dir for peer in peers.values()}
    assert _held(drive)[peers[names[0]].skill_dir][0] == (edited / "SKILL.md").read_text(encoding="utf-8")


def test_each_install_and_repository_spelling_keeps_its_own_facts_within_the_install_bound(tmp_path, monkeypatch):
    drive, checkout = tmp_path / "drive", tmp_path / "checkout"
    for root, name in ((drive / "skills", "alpha"), (checkout, "beta")):
        (root / name).mkdir(parents=True)
        (root / name / "SKILL.md").write_text(_manifest(name), encoding="utf-8")
    _reads, parses = _counted(monkeypatch)
    for _ in range(3):  # alternating spellings list different locations; neither releases the other's facts
        assert {peer.name for peer in discover_skill_peers(drive, repo_path=str(checkout))} == {"alpha", "beta"}
        assert {peer.name for peer in discover_skill_peers(drive, repo_path="")} == {"alpha"}
    assert len(parses) == 3  # alpha under each spelling, beta once
    others = []
    for index in range(peer_inventory.HELD_INSTALLS):
        other = tmp_path / f"other_{index}"
        (other / "skills" / "gamma").mkdir(parents=True)
        (other / "skills" / "gamma" / "SKILL.md").write_text(_manifest("gamma"), encoding="utf-8")
        discover_skill_peers(other, repo_path="")
        others.append(other)
    assert len(peer_inventory._HELD_FACTS) == peer_inventory.HELD_INSTALLS  # a memory bound over installs
    assert (str(drive), "") not in peer_inventory._HELD_FACTS and len(_held(others[-1])) == 1
    parses.clear()
    discover_skill_peers(drive, repo_path="")
    assert len(parses) == 1  # a released install is parsed again: meaning is unchanged
