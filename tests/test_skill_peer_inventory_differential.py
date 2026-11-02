"""F4: the cheap peer projection must decide EXACTLY what full discovery decided.

`discover_skill_peers` replaces `discover_skills` in the conflict/enablement
position only.  This module is the differential proof: over every bucket pair,
identity collision, missing / unreadable / malformed manifest, valid manifest
with an unreadable payload, and one-sided conflict declaration, the projection
and the full loader must agree on (name, enabled, conflicts, collision) and on
the resulting `skill_conflict_status` for every subject.

It also pins the two properties the cheap scan exists for: it hashes no
payloads, and it writes nothing into the root it reads.
"""
from __future__ import annotations

import json
import os
import pathlib
import stat

import pytest

from ouroboros import skill_loader, skill_peer_inventory
from ouroboros.skill_loader import (
    discover_skills,
    save_enabled,
    skill_conflict_status,
)
from ouroboros.skill_peer_inventory import SkillPeer, discover_conflict_peers, discover_skill_peers

BUCKETS = ("native", "clawhub", "external", "ouroboroshub")


def _manifest(name: str, *, conflicts=(), extra: str = "") -> str:
    conflict_text = "[" + ", ".join(json.dumps(item) for item in conflicts) + "]"
    return (
        "---\n"
        f"name: {name}\n"
        "description: differential fixture\n"
        "version: 1.0.0\n"
        "type: instruction\n"
        f"conflicts: {conflict_text}\n"
        f"{extra}"
        "---\n"
        "body\n"
    )


def _write(skill_dir: pathlib.Path, text: str | None, *, payload: bytes | None = b"payload") -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    if text is not None:
        (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")
    if payload is not None:
        (skill_dir / "data.bin").write_bytes(payload)


def _tree(root: pathlib.Path) -> dict:
    out = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        out[rel] = "dir" if path.is_dir() else path.stat().st_size
    return out


def _projection(items) -> dict:
    return {
        item.name: (
            bool(item.enabled),
            tuple(sorted(item.conflicts)),
            bool(getattr(item, "identity_collision", False)),
        )
        for item in items
    }


@pytest.fixture()
def differential_root(tmp_path):
    """One root holding every peer shape the projection has to reproduce."""
    drive = tmp_path / "drive"
    skills = drive / "skills"
    checkout = tmp_path / "checkout"

    # One ordinary skill per bucket, with a one-sided conflict declaration
    # pointing at the NEXT bucket's skill.
    for index, bucket in enumerate(BUCKETS):
        target = BUCKETS[(index + 1) % len(BUCKETS)]
        _write(skills / bucket / f"in_{bucket}",
               _manifest(f"display-{bucket}", conflicts=(f"in_{target}",)))
        save_enabled(drive, f"in_{bucket}", index % 2 == 0)

    # A loose data-plane package (classified external) and a user-checkout one.
    _write(skills / "loose_pkg", _manifest("display-loose"))
    save_enabled(drive, "loose_pkg", True)
    _write(checkout / "repo_pkg", _manifest("display-repo", conflicts=("loose_pkg",)))
    save_enabled(drive, "repo_pkg", True)

    # Malformed manifest, unreadable manifest, missing manifest.
    _write(skills / "external" / "malformed_pkg", "---\nname: [\n---\n")
    save_enabled(drive, "malformed_pkg", True)
    _write(skills / "external" / "unreadable_manifest", _manifest("x"))
    _write(skills / "external" / "missing_manifest", None, payload=b"orphan")
    save_enabled(drive, "missing_manifest", True)

    # Valid manifest whose PAYLOAD cannot be read: the loader keeps such a peer
    # enabled and keeps its conflicts, with a load_error.  The projection must
    # not silently disable it.
    _write(skills / "external" / "unreadable_payload",
           _manifest("display-unreadable", conflicts=("loose_pkg",)))
    save_enabled(drive, "unreadable_payload", True)

    # A launcher seed for another platform: discovery hides it, so it is no peer.
    _write(skills / "native" / "elsewhere_seed",
           _manifest("elsewhere", conflicts=("in_clawhub",), extra="os: [plan9]\n"))
    (skills / "native" / "elsewhere_seed" / ".seed-origin").write_text("seeded_from=skills\n", encoding="utf-8")
    save_enabled(drive, "elsewhere_seed", True)

    # Identity collision: the same sanitised basename in two canonical roots.
    _write(skills / "clawhub" / "twin_pkg", _manifest("twin-one", conflicts=("loose_pkg",)))
    _write(checkout / "twin_pkg", _manifest("twin-two"))
    save_enabled(drive, "twin_pkg", True)

    return {"drive": drive, "checkout": checkout, "skills": skills}


def _make_unreadable(paths: list[pathlib.Path]) -> bool:
    if os.name == "nt" or os.getuid() == 0:
        return False
    for path in paths:
        path.chmod(0)
    return True


def test_peer_projection_matches_full_discovery_over_every_shape(differential_root):
    drive = differential_root["drive"]
    checkout = str(differential_root["checkout"])
    skills = differential_root["skills"]
    restricted = [
        skills / "external" / "unreadable_manifest" / "SKILL.md",
        skills / "external" / "unreadable_payload" / "data.bin",
    ]
    readable = not _make_unreadable(restricted)
    try:
        full = discover_skills(drive, repo_path=checkout)
        peers = list(discover_skill_peers(drive, repo_path=checkout))
        assert _projection(peers) == _projection(full)
        if not readable:
            # The two deliberately unreadable shapes have to be present and
            # classified the way the loader classifies them.
            by_name = {peer.name: peer for peer in peers}
            assert by_name["unreadable_manifest"].enabled is False
            assert by_name["unreadable_manifest"].manifest_error
            assert by_name["unreadable_payload"].enabled is True
            assert by_name["unreadable_payload"].conflicts == ("loose_pkg",)
        assert {peer.name for peer in peers if peer.identity_collision} == {"twin_pkg"}
        assert "missing_manifest" not in {peer.name for peer in peers}
        assert "elsewhere_seed" not in {peer.name for peer in peers}

        # Every subject's conflict verdict must be identical under either peer list.
        for subject in full:
            assert skill_conflict_status(subject, peers) == skill_conflict_status(subject, full), subject.name
    finally:
        for path in restricted:
            try:
                path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass


def test_conflict_projection_reaches_the_full_verdict_without_off_manifests(differential_root, monkeypatch):
    """The execution projection keeps every verdict and opens only ON, unique peers' manifests."""
    drive = differential_root["drive"]
    checkout = str(differential_root["checkout"])
    skills = differential_root["skills"]
    restricted = [skills / "external" / "unreadable_manifest" / "SKILL.md",
                  skills / "external" / "unreadable_payload" / "data.bin"]
    _make_unreadable(restricted)
    opened, real = [], skill_peer_inventory._manifest_text_for_dir
    monkeypatch.setattr(skill_peer_inventory, "_manifest_text_for_dir",
                        lambda path: opened.append(path.name) or real(path))
    try:
        full = discover_skills(drive, repo_path=checkout)
        everything = list(discover_skill_peers(drive, repo_path=checkout))
        assert {"in_clawhub", "in_ouroboroshub"} <= set(opened)  # the identity scan reads OFF peers
        for subject in full:
            opened.clear()
            projected = discover_conflict_peers(drive, subject.name, repo_path=checkout)
            assert skill_conflict_status(subject, list(projected)) == skill_conflict_status(subject, full), subject.name
            assert skill_conflict_status(subject, list(projected)) == skill_conflict_status(subject, everything)
            # Enabled, unique, readable-or-not manifests of others: never the subject, an OFF
            # peer or a collision placeholder; the other-platform seed is read, then hidden.
            assert sorted(opened) == sorted({"in_native", "in_external", "loose_pkg", "repo_pkg", "malformed_pkg",
                                             "unreadable_payload", "elsewhere_seed"} - {subject.name}), subject.name
            assert all(peer.enabled and not peer.manifest_error and peer.name != subject.name for peer in projected)
            assert {peer.name for peer in projected} == {"in_native", "in_external", "loose_pkg", "repo_pkg",
                                                         "unreadable_payload"} - {subject.name}
    finally:
        for path in restricted:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_peer_projection_writes_nothing_into_the_root_it_reads(differential_root):
    """A passive peer scan is a pure read: no state dirs, no enabled.json stubs."""
    drive = differential_root["drive"]
    before = _tree(drive)
    discover_skill_peers(drive, repo_path=str(differential_root["checkout"]))
    assert _tree(drive) == before


def test_peer_projection_hashes_no_payloads(differential_root, monkeypatch):
    calls = {"n": 0}
    real = skill_loader.compute_content_hash

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(skill_loader, "compute_content_hash", counted)
    peers = discover_skill_peers(drive_root := differential_root["drive"],
                                 repo_path=str(differential_root["checkout"]))
    assert peers
    assert calls["n"] == 0, "the cheap peer scan hashed a payload"
    # The full loader on the same root does hash: the comparison is meaningful.
    discover_skills(drive_root, repo_path=str(differential_root["checkout"]))
    assert calls["n"] > 0


def test_one_sided_conflicts_are_binding_only_while_the_other_peer_is_enabled(tmp_path):
    drive = tmp_path / "drive"
    skills = drive / "skills" / "external"
    _write(skills / "declares", _manifest("declares", conflicts=("target",)))
    _write(skills / "target", _manifest("target"))
    save_enabled(drive, "declares", True)
    save_enabled(drive, "target", True)

    def verdicts():
        full = discover_skills(drive, repo_path="")
        peers = list(discover_skill_peers(drive, repo_path=""))
        return [
            (subject.name, skill_conflict_status(subject, peers),
             skill_conflict_status(subject, full))
            for subject in sorted(full, key=lambda item: item.name)
        ]

    for name, projected, loaded in verdicts():
        assert projected == loaded
        assert projected is not None, name  # symmetric enforcement while both are enabled

    save_enabled(drive, "target", False)
    result = dict((name, projected) for name, projected, _loaded in verdicts())
    for name, projected, loaded in verdicts():
        assert projected == loaded, name
    # A DISABLED peer is inert: the enabled declarer no longer conflicts.  The
    # disabled subject still sees the enabled declarer, which is the existing
    # asymmetry the projection must reproduce rather than "improve".
    assert result["declares"] is None
    assert result["target"] == {"code": "skill_conflict", "skills": ["declares"], "omitted": 0}


def test_collision_placeholders_never_carry_enablement_or_conflicts(tmp_path):
    drive = tmp_path / "drive"
    checkout = tmp_path / "checkout"
    _write(drive / "skills" / "external" / "dup", _manifest("dup-a", conflicts=("other",)))
    _write(checkout / "dup", _manifest("dup-b", conflicts=("other",)))
    _write(drive / "skills" / "external" / "other", _manifest("other"))
    save_enabled(drive, "dup", True)
    save_enabled(drive, "other", True)

    peers = list(discover_skill_peers(drive, repo_path=str(checkout)))
    full = discover_skills(drive, repo_path=str(checkout))
    assert _projection(peers) == _projection(full)
    collided = [peer for peer in peers if peer.name == "dup"]
    assert len(collided) == 2
    assert all(not peer.enabled and not peer.conflicts and peer.identity_collision
               for peer in collided)
    other = next(item for item in full if item.name == "other")
    # A collided placeholder is disabled, so it cannot conflict anything out.
    assert skill_conflict_status(other, peers) is None
    assert skill_conflict_status(other, full) is None


def test_skill_peer_is_immutable_and_non_executable():
    peer = SkillPeer(name="x", location="external", skill_dir=pathlib.Path("/tmp/x"))
    with pytest.raises(Exception):
        peer.enabled = True  # type: ignore[misc]
    assert not hasattr(peer, "content_hash")
    assert not hasattr(peer, "review")
    assert not hasattr(peer, "manifest")


def test_a_peer_is_a_valid_subject_of_the_conflict_projection(tmp_path):
    """Duck typing holds on BOTH sides: a SkillPeer subject reaches the same verdict.

    `enabled_skill_conflicts` reads `skill.conflicts`, never `skill.manifest`,
    because `SkillPeer` deliberately has no manifest.
    """
    drive = tmp_path / "drive"
    skills = drive / "skills" / "external"
    _write(skills / "declares", _manifest("declares", conflicts=("target",)))
    _write(skills / "target", _manifest("target"))
    save_enabled(drive, "declares", True)
    save_enabled(drive, "target", True)
    full = {s.name: s for s in discover_skills(drive, repo_path="")}
    peers = {p.name: p for p in discover_skill_peers(drive, repo_path="")}
    for name in ("declares", "target"):
        as_full = skill_conflict_status(full[name], list(peers.values()))
        as_peer = skill_conflict_status(peers[name], list(peers.values()))
        assert as_peer is not None, name
        assert as_full == as_peer, name
