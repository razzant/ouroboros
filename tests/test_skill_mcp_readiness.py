"""Skill binding reads reviewed entry paths and lightweight conflict facts."""
from types import SimpleNamespace

import pytest

from ouroboros import mcp_catalog, skill_mcp


def test_entry_alias_cannot_select_unreviewed_dependency(tmp_path, monkeypatch):
    root = tmp_path / "skill"
    dependency = root / "node_modules" / "pkg"
    dependency.mkdir(parents=True)
    (dependency / "server.mjs").write_text("// dependency")
    try:
        (root / "bin").symlink_to(dependency, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlink unavailable: {exc}")
    monkeypatch.setattr("ouroboros.tools.skill_exec._resolve_runtime_binary", lambda _: ("/node", ""))
    launch, reason = skill_mcp._derive_launch(SimpleNamespace(skill_dir=root),
                                             {"entry": "bin/server.mjs", "runtime": "node"})
    assert launch is None and "cache or dependency folder" in reason
    (root / "server.mjs").write_text("// reviewed entry")
    launch, reason = skill_mcp._derive_launch(SimpleNamespace(skill_dir=root),
                                             {"entry": "server.mjs", "runtime": "node"})
    assert not reason and launch["args"] == [str(root / "server.mjs")]


@pytest.mark.parametrize("bucket,note", [("native", ".seed-origin"), ("ouroboroshub", ".ouroboroshub.json"),
                                        ("clawhub", ".clawhub.json"), ("external", ".self_authored.json")])
def test_entry_cannot_select_unhashed_install_notes(tmp_path, monkeypatch, bucket, note):
    from ouroboros.skill_identity import payload_digest

    root = tmp_path / bucket / "skill"
    root.mkdir(parents=True)
    (root / note).write_text('{"note": "v1"}')
    before = payload_digest(root)
    (root / note).write_text('{"note": "v2"}')
    assert payload_digest(root) == before, "these bytes do not belong to the accepted release"
    monkeypatch.setattr("ouroboros.tools.skill_exec._resolve_runtime_binary", lambda _: ("/node", ""))
    for entry in (note, "server.mjs"):
        if entry != note:
            (root / entry).symlink_to(note)
        launch, reason = skill_mcp._derive_launch(SimpleNamespace(skill_dir=root), {"entry": entry, "runtime": "node"})
        assert launch is None and "outside the reviewed files" in reason


@pytest.mark.parametrize("bucket,entry", [("native", "scripts/server.mjs"), ("external", ".seed-origin"),
                                         ("native", "scripts/.seed-origin")])
def test_entry_accepts_hashed_payload_including_non_note_names(tmp_path, monkeypatch, bucket, entry):
    from ouroboros.skill_identity import payload_digest

    root = tmp_path / bucket / "skill"
    path = root / entry
    path.parent.mkdir(parents=True)
    path.write_text("// reviewed v1")
    before = payload_digest(root)
    path.write_text("// reviewed v2")
    assert payload_digest(root) != before
    monkeypatch.setattr("ouroboros.tools.skill_exec._resolve_runtime_binary", lambda _: ("/node", ""))
    launch, reason = skill_mcp._derive_launch(SimpleNamespace(skill_dir=root), {"entry": entry, "runtime": "node"})
    assert not reason and launch["args"] == [str(path)]


def test_readiness_uses_conflict_inventory_not_peer_payloads(tmp_path, monkeypatch):
    from ouroboros import skill_loader, skill_peer_inventory, skill_readiness
    peer = SimpleNamespace(name="other")
    subject = SimpleNamespace(name="subject", load_error="", manifest=SimpleNamespace(is_script=lambda: True))
    calls = []
    monkeypatch.setattr(skill_loader, "discover_skills", lambda *_a, **_k: pytest.fail("must not hash peer payloads"))
    monkeypatch.setattr(skill_peer_inventory, "discover_conflict_peers",
                        lambda root, name, **kw: calls.append((root, name)) or (peer,))
    def readiness(root, skill, *, skills):
        assert root == tmp_path and skill is subject and skills == [peer]
        return SimpleNamespace(blockers=["skill_conflict:other"])
    monkeypatch.setattr(skill_readiness, "skill_readiness_for_execution", readiness)
    assert skill_mcp._not_ready(tmp_path, subject) == ["skill_conflict:other"]
    assert calls == [(tmp_path, "subject")]


def test_unavailable_skill_notice_matches_dispatch_reason():
    reason = "MCP_SKILL_UNAVAILABLE: This skill is off. Nothing was started."
    runtime = SimpleNamespace(config=SimpleNamespace(skill_error=reason), catalog=None)
    assert mcp_catalog.catalog_notice(runtime) == reason
