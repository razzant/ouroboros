"""CPL4-C11 pins (owner batch №8, 3A): uninstall tombstones the skill state.

Hub uninstalls write ``state/skills/<name>/uninstalled.json``; the startup
sweep clears the dead owner state BY that mark, preserving ``grants.json``
(owner authority) and self-healing a reinstall. Unmarked dirs are never
touched; the tombstone filename is forgery-guarded like every other owner
state file.
"""

from __future__ import annotations

import json

import ouroboros.skill_loader as skill_loader
from ouroboros.skill_uninstall_state import (
    UNINSTALL_TOMBSTONE_FILENAME,
    sweep_uninstalled_skill_state,
    write_uninstall_tombstone,
)


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

    assert "sweep_uninstalled_skill_state" in inspect.getsource(sm._startup_prune_sweeps)
