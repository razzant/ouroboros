"""CPL4-C11 pins (owner batch №8, 3A): uninstall tombstones the skill state.

Hub uninstalls write ``state/skills/<name>/uninstalled.json``; the startup
sweep clears the dead owner state BY that mark, preserving ``grants.json``
(owner authority) and the ``ouroboroshub.json`` publication receipt (local
submission history, issue #1314), and self-healing a reinstall. Unmarked dirs
are never touched; the tombstone filename is forgery-guarded like every other
owner state file. Explicit local Delete still removes the whole state dir.
"""

from __future__ import annotations

import json

import ouroboros.skill_loader as skill_loader
from ouroboros.marketplace import ouroboroshub
from ouroboros.marketplace.provenance import (
    PUBLICATION_FILENAME,
    merge_state_record,
    read_publication_record,
    write_publication_record,
)
from ouroboros.skill_uninstall_state import (
    UNINSTALL_TOMBSTONE_FILENAME,
    delete_local_skill,
    sweep_uninstalled_skill_state,
    write_uninstall_tombstone,
)

_MANIFEST = (
    "---\nname: demo\ndescription: Retention fixture\nversion: \"1.0.0\"\n"
    "type: script\nruntime: python3\nscripts:\n  - name: check.py\n    description: Fixture\n---\n# Fixture\n"
)
_RECEIPT = {
    "slug": "demo", "version": "1.0.0", "content_hash": "a" * 64,
    "repository": "razzant/OuroborosHub", "pr_number": 60,
    "pr_url": "https://github.com/razzant/OuroborosHub/pull/60",
    "published_at": "2026-09-14T00:00:00Z",
}


def _seed_state(tmp_path, name):
    state = skill_loader.skill_state_dir(tmp_path, name)
    (state / "review.json").write_text('{"status": "pending"}', encoding="utf-8")
    (state / "enabled.json").write_text('{"enabled": true}', encoding="utf-8")
    (state / "grants.json").write_text('{"granted_keys": ["K"]}', encoding="utf-8")
    (state / "review_history.jsonl").write_text('{"status": "clean"}\n', encoding="utf-8")
    (state / "review_dispatch").mkdir()
    (state / "review_dispatch" / "w1.json").write_text("{}", encoding="utf-8")
    return state


def test_tombstone_written_and_stamped(tmp_path):
    write_uninstall_tombstone(tmp_path, "s", source="clawhub")
    marker = skill_loader.skill_state_dir(tmp_path, "s") / UNINSTALL_TOMBSTONE_FILENAME
    data = json.loads(marker.read_text(encoding="utf-8"))
    assert data["source"] == "clawhub" and data["uninstalled_at"]
    assert data["_schema_version"] == skill_loader.SKILL_OWNER_STATE_SCHEMA_VERSION


def test_sweep_clears_dead_state_but_keeps_grants(tmp_path, monkeypatch):
    state = _seed_state(tmp_path, "dead")
    write_uninstall_tombstone(tmp_path, "dead", source="ouroboroshub")
    untouched = _seed_state(tmp_path, "alive-unmarked")
    monkeypatch.setattr(skill_loader, "find_skill", lambda root, name, **kw: None)

    report = sweep_uninstalled_skill_state(tmp_path)

    assert report["swept"] == ["dead"] and not report["errors"]
    assert sorted(p.name for p in state.iterdir()) == ["grants.json", UNINSTALL_TOMBSTONE_FILENAME]
    # An unmarked dir is never touched — the tombstone is the only authority.
    assert (untouched / "review.json").exists() and (untouched / "review_dispatch").is_dir()


def _hub_install(monkeypatch, data_root):
    """The real Hub installer with only the catalog read and download faked."""
    hub_root = data_root / "skills" / "ouroboroshub"
    monkeypatch.setattr(ouroboroshub, "get_ouroboroshub_skills_dir", lambda: hub_root)
    summary = ouroboroshub.HubSkillSummary(
        slug="demo", name="demo", version="1.0.0", files=[{"path": "SKILL.md", "sha256": "x", "size": 1}],
    )
    monkeypatch.setattr(ouroboroshub, "load_catalog", lambda *a, **kw: {
        "raw_base_url": "https://raw.githubusercontent.com/razzant/OuroborosHub/main"})
    monkeypatch.setattr(ouroboroshub, "_summaries", lambda _catalog: [summary])

    def fake_download(_summary, _raw_base, staging_dir):
        (staging_dir / "SKILL.md").write_text(_MANIFEST, encoding="utf-8")
        (staging_dir / "scripts").mkdir()
        (staging_dir / "scripts" / "check.py").write_text("print('fixture')\n", encoding="utf-8")

    monkeypatch.setattr(ouroboroshub, "_download_skill_files", fake_download)
    result = ouroboroshub.install("demo", overwrite=True)
    assert result.ok, result.error
    assert skill_loader.find_skill(data_root, "demo") is not None


def _seed_receipt(data_root):
    write_publication_record(data_root, "demo", _RECEIPT)
    merge_state_record(data_root, "demo", PUBLICATION_FILENAME, {"future": {"keep": True}})
    return (skill_loader.skill_state_dir(data_root, "demo") / PUBLICATION_FILENAME).read_bytes()


def test_publication_receipt_survives_hub_uninstall_sweep_and_reinstall(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _hub_install(monkeypatch, data)
    state = _seed_state(data, "demo")
    receipt_bytes = _seed_receipt(data)

    assert ouroboroshub.uninstall("demo").ok
    report = sweep_uninstalled_skill_state(data)

    assert report["swept"] == ["demo"] and not report["errors"]
    assert sorted(p.name for p in state.iterdir()) == sorted(
        ["grants.json", PUBLICATION_FILENAME, UNINSTALL_TOMBSTONE_FILENAME])
    assert (state / PUBLICATION_FILENAME).read_bytes() == receipt_bytes
    # A second startup keeps it too: the sweep is idempotent over kept files.
    assert not sweep_uninstalled_skill_state(data)["swept"]

    _hub_install(monkeypatch, data)
    report = sweep_uninstalled_skill_state(data)

    assert report["restored"] == ["demo"] and not report["swept"]
    assert (state / PUBLICATION_FILENAME).read_bytes() == receipt_bytes
    assert read_publication_record(data, "demo") == (_RECEIPT, None)


def test_publication_receipt_survives_reinstall_before_the_sweep(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _hub_install(monkeypatch, data)
    state = _seed_state(data, "demo")
    receipt_bytes = _seed_receipt(data)

    assert ouroboroshub.uninstall("demo").ok
    _hub_install(monkeypatch, data)
    report = sweep_uninstalled_skill_state(data)

    assert report["restored"] == ["demo"] and not report["swept"]
    assert (state / "review.json").exists()
    assert (state / PUBLICATION_FILENAME).read_bytes() == receipt_bytes


def test_explicit_local_delete_still_removes_the_receipt(tmp_path):
    data = tmp_path / "data"
    payload = data / "skills" / "external" / "demo"
    (payload / "scripts").mkdir(parents=True)
    (payload / "SKILL.md").write_text(_MANIFEST, encoding="utf-8")
    (payload / "scripts" / "check.py").write_text("print('fixture')\n", encoding="utf-8")
    state = _seed_state(data, "demo")
    _seed_receipt(data)
    loaded = skill_loader.find_skill(data, "demo")

    result = delete_local_skill(data, loaded, payload_root="skills/external/demo")

    assert result["ok"] and result["deleted_state"], result
    assert not state.exists() and not payload.exists()
    assert read_publication_record(data, "demo") == (None, None)


def test_sweep_self_heals_a_reinstalled_skill(tmp_path, monkeypatch):
    state = _seed_state(tmp_path, "back")
    write_uninstall_tombstone(tmp_path, "back", source="clawhub")
    monkeypatch.setattr(skill_loader, "find_skill", lambda root, name, **kw: object())

    report = sweep_uninstalled_skill_state(tmp_path)

    assert report["restored"] == ["back"] and not report["swept"]
    assert not (state / UNINSTALL_TOMBSTONE_FILENAME).exists()
    assert (state / "review.json").exists()  # nothing swept


def test_sweep_fails_closed_when_payload_probe_fails(tmp_path, monkeypatch):
    state = _seed_state(tmp_path, "murky")
    write_uninstall_tombstone(tmp_path, "murky", source="clawhub")

    def _boom(root, name, **kw):
        raise RuntimeError("discovery unavailable")

    monkeypatch.setattr(skill_loader, "find_skill", _boom)
    report = sweep_uninstalled_skill_state(tmp_path)

    assert report["errors"] and not report["swept"]
    assert (state / "review.json").exists()  # kept: cannot prove payload-gone


def test_hub_uninstall_paths_write_the_tombstone():
    import inspect

    import ouroboros.marketplace.install as install
    import ouroboros.marketplace.ouroboroshub as hub

    assert "write_uninstall_tombstone" in inspect.getsource(install.uninstall_skill)
    assert "write_uninstall_tombstone" in inspect.getsource(hub.uninstall)


def test_tombstone_filename_is_forgery_guarded():
    from ouroboros.contracts.skill_payload_policy import (
        SKILL_OWNER_STATE_FILENAMES,
        SKILL_OWNER_STATE_STEMS,
    )

    assert UNINSTALL_TOMBSTONE_FILENAME in SKILL_OWNER_STATE_FILENAMES
    assert "uninstalled" in SKILL_OWNER_STATE_STEMS


def test_startup_prune_sweeps_run_the_tombstone_sweep():
    import inspect

    import ouroboros.server_maintenance as sm

    assert "sweep_uninstalled_skill_state" in inspect.getsource(sm._run_deferred_startup_prunes)
