"""``reconcile_extension``: which desired state wins, and what the runtime state reports.

Split out of ``tests/test_extension_loader.py`` when that module was divided by
theme; every moved block is verbatim. Covers unload callbacks running outside
the loader lock, concurrent reconciles converging, warnings reviews under both
enforcement modes, the single discovery snapshot, light mode keeping extensions
live, live extensions staying loaded, code changes reloading, load errors
preserved and reported, and the enable-revert flag on failed enables.
"""

from __future__ import annotations

import pathlib

from ouroboros import extension_loader
from ouroboros.skill_loader import SkillReviewState, save_enabled, save_review_state

from tests._extension_loader_shared import (
    _prepare_extension,
    _write_ext_skill,
)
from tests._extension_loader_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clear_loader_state,
)


def test_reconcile_unload_callbacks_do_not_hold_loader_lock(tmp_path):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "lock_probe",
        "import pathlib, threading\n"
        "def register(api):\n"
        "    state_dir = pathlib.Path(api.get_state_dir())\n"
        "    def cleanup():\n"
        "        done = state_dir / 'snapshot_done.txt'\n"
        "        def worker():\n"
        "            from ouroboros import extension_loader\n"
        "            extension_loader.snapshot()\n"
        "            done.write_text('done', encoding='utf-8')\n"
        "        thread = threading.Thread(target=worker)\n"
        "        thread.start()\n"
        "        thread.join(timeout=1.0)\n"
        "        if not done.exists():\n"
        "            raise RuntimeError('snapshot blocked by loader lock')\n"
        "    api.on_unload(cleanup)\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n",
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err

    # Make the extension undesired so reconcile unloads it through the normal path.
    save_enabled(drive_root, "lock_probe", False)
    state = extension_loader.reconcile_extension("lock_probe", drive_root, lambda: {})

    done_file = drive_root / "state" / "skills" / "lock_probe" / "snapshot_done.txt"
    assert done_file.read_text(encoding="utf-8") == "done"
    assert state["action"] == "extension_unloaded"


def test_concurrent_reconcile_converges_to_one_live_extension(tmp_path):
    import threading

    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "race_ext",
        "import time\n"
        "def register(api):\n"
        "    time.sleep(0.05)\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n",
        permissions=["tool"],
    )

    results = []
    repo_path = str(tmp_path / "skills")
    threads = [
        threading.Thread(
            target=lambda: results.append(
                extension_loader.reconcile_extension("race_ext", drive_root, lambda: {}, repo_path=repo_path)
            )
        )
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert len(results) == 2
    assert {r["action"] for r in results} <= {"extension_loaded", "extension_already_live"}
    snap = extension_loader.snapshot()
    assert snap["extensions"] == ["race_ext"]
    assert snap["tools"] == [extension_loader.extension_surface_name("race_ext", "ping")]
    assert extension_loader.runtime_state_for_skill_name("race_ext", drive_root, repo_path=repo_path)["reason"] == "ready"


def test_reconcile_extension_allows_warnings_review(tmp_path, monkeypatch):
    from ouroboros.skill_loader import find_skill
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")

    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "advisory_live",
        "def register(api):\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n",
        permissions=["tool"],
    )
    save_review_state(
        drive_root,
        "advisory_live",
        SkillReviewState(status="warnings", content_hash=loaded.content_hash),
    )
    loaded = find_skill(drive_root, "advisory_live", repo_path=str(repo_root))
    assert loaded is not None

    state = extension_loader.reconcile_extension(
        "advisory_live",
        drive_root,
        lambda: {},
        repo_path=str(repo_root),
    )

    assert state["action"] == "extension_loaded"
    assert extension_loader.runtime_state_for_skill_name(
        "advisory_live",
        drive_root,
        repo_path=str(repo_root),
    )["reason"] == "ready"


def test_reconcile_extension_allows_warnings_under_blocking(tmp_path, monkeypatch):
    from ouroboros.skill_loader import find_skill
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")

    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "advisory_warnings",
        "def register(api):\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n",
        permissions=["tool"],
    )
    save_review_state(
        drive_root,
        "advisory_warnings",
        SkillReviewState(status="warnings", content_hash=loaded.content_hash),
    )
    loaded = find_skill(drive_root, "advisory_warnings", repo_path=str(repo_root))
    assert loaded is not None

    state = extension_loader.reconcile_extension(
        "advisory_warnings",
        drive_root,
        lambda: {},
        repo_path=str(repo_root),
    )

    assert state["action"] == "extension_loaded"
    assert state["reason"] == "ready"


def test_reconcile_reuses_one_discovered_peer_snapshot(tmp_path, monkeypatch):
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "single_scan",
        "def register(api):\n    pass\n",
        permissions=[],
    )
    calls = 0
    real_discover = extension_loader.discover_skills

    def counted_discover(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_discover(*args, **kwargs)

    monkeypatch.setattr(extension_loader, "discover_skills", counted_discover)

    state = extension_loader.reconcile_extension(
        loaded.name,
        drive_root,
        lambda: {},
        repo_path=str(repo_root),
    )

    assert state["action"] == "extension_loaded"
    assert calls == 1


def test_reconcile_extension_stays_loaded_in_light_mode(tmp_path, monkeypatch):
    """v5.1.2 Frame A: ``light`` no longer unloads extensions. The
    ``runtime_mode_light`` reason is gone from
    ``_extension_runtime_state``. Extensions follow the same
    enabled / review / content-hash gates regardless of mode.
    """
    plugin = (
        "def _echo(ctx):\n"
        "    return 'ok'\n"
        "def register(api):\n"
        "    api.register_tool('echo', _echo, description='echo', schema={})\n"
    )
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "lightstop",
        plugin,
        permissions=["tool"],
    )
    grant_roots = []
    real_grant_status = extension_loader.grant_status_for_skill

    def record_grant_root(root, skill):
        grant_roots.append(pathlib.Path(root))
        return real_grant_status(root, skill)

    monkeypatch.setattr(extension_loader, "grant_status_for_skill", record_grant_root)
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    assert grant_roots and grant_roots[0] == drive_root
    assert "lightstop" in extension_loader.snapshot()["extensions"]

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    state = extension_loader.reconcile_extension(
        "lightstop",
        drive_root,
        lambda: {},
        repo_path=repo_root,
    )
    # The ``runtime_mode_light`` reason was removed in v5.1.2; the
    # extension stays live.
    assert state["reason"] != "runtime_mode_light"
    assert state["action"] != "extension_unloaded"
    assert "lightstop" in extension_loader.snapshot()["extensions"]


def test_reconcile_extension_keeps_live_extension_loaded(tmp_path, monkeypatch):
    plugin = (
        "def _echo(ctx):\n"
        "    return 'ok'\n"
        "def register(api):\n"
        "    api.register_tool('echo', _echo, description='echo', schema={})\n"
    )
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "steady",
        plugin,
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    unload_calls: list[str] = []
    monkeypatch.setattr(extension_loader, "unload_extension", unload_calls.append)

    state = extension_loader.reconcile_extension(
        "steady",
        drive_root,
        lambda: {},
        repo_path=repo_root,
    )
    assert state["reason"] == "ready"
    assert state["action"] == "extension_already_live"
    assert unload_calls == []
    assert "steady" in extension_loader.snapshot()["extensions"]


def test_reconcile_extension_reloads_when_live_code_changes(tmp_path):
    from ouroboros.skill_loader import find_skill

    skill_dir = tmp_path / "skills" / "reloadme"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            "name: reloadme\n"
            "description: Live reload.\n"
            "version: 0.1.0\n"
            "type: extension\n"
            "entry: plugin.py\n"
            "permissions: [\"tool\"]\n"
            "env_from_settings: []\n"
            "---\n"
            "body\n"
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(
        (
            "def _echo(ctx):\n"
            "    return 'v1'\n"
            "def register(api):\n"
            "    api.register_tool('echo', _echo, description='echo', schema={})\n"
        ),
        encoding="utf-8",
    )
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    save_enabled(drive_root, "reloadme", True)
    loaded = find_skill(drive_root, "reloadme", repo_path=str(skill_dir.parent))
    assert loaded is not None
    save_review_state(
        drive_root,
        "reloadme",
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )
    loaded = find_skill(drive_root, "reloadme", repo_path=str(skill_dir.parent))
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    tool = extension_loader.get_tool(extension_loader.extension_surface_name("reloadme", "echo"))
    assert tool is not None
    assert tool["handler"](None) == "v1"

    (skill_dir / "plugin.py").write_text(
        (
            "def _echo(ctx):\n"
            "    return 'v2'\n"
            "def register(api):\n"
            "    api.register_tool('echo', _echo, description='echo', schema={})\n"
        ),
        encoding="utf-8",
    )
    loaded = find_skill(drive_root, "reloadme", repo_path=str(skill_dir.parent))
    assert loaded is not None
    save_review_state(
        drive_root,
        "reloadme",
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )

    state = extension_loader.reconcile_extension(
        "reloadme",
        drive_root,
        lambda: {},
        repo_path=skill_dir.parent,
        retry_load_error=True,
    )
    assert state["action"] == "extension_loaded"
    assert state["live_loaded"] is True
    tool = extension_loader.get_tool(extension_loader.extension_surface_name("reloadme", "echo"))
    assert tool is not None
    assert tool["handler"](None) == "v2"


def test_runtime_state_preserves_matching_load_error(tmp_path):
    plugin = (
        "def _hello(request):\n"
        "    return {'hello': 'world'}\n"
        "def register(api):\n"
        "    api.register_route('/absolute', _hello, methods=('GET',))\n"
    )
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "brokenlive",
        plugin,
        permissions=["route"],
    )
    state = extension_loader.reconcile_extension(
        "brokenlive",
        drive_root,
        lambda: {},
        repo_path=repo_root,
        retry_load_error=True,
    )
    assert state["action"] == "extension_load_error"
    refreshed = extension_loader.runtime_state_for_skill_name(
        "brokenlive",
        drive_root,
        repo_path=repo_root,
    )
    assert refreshed["reason"] == "load_error"
    assert "absolute" in str(refreshed["load_error"])
    assert refreshed["live_loaded"] is False


def test_runtime_state_for_skill_name_reports_missing_skill(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    state = extension_loader.runtime_state_for_skill_name(
        "ghost",
        drive_root,
        repo_path=tmp_path / "skills",
    )
    assert state["desired_live"] is False
    assert state["live_loaded"] is False
    assert state["reason"] == "missing"
    assert state["process"] in {"server", "worker"}


def test_standalone_reconcile_reads_a_fresh_health_stamp_each_time(tmp_path, monkeypatch):
    """The batch optimization must not reuse a stamp across separate reconciles."""
    from ouroboros import extension_health, utils

    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    _write_ext_skill(
        repo_root,
        "standalone_stamp",
        plugin_body="def register(api):\n    pass\n",
        permissions=[],
    )
    calls = {"n": 0}

    def get_git_info(_repo):
        calls["n"] += 1
        return "main", f"stamp-{calls['n']}"

    extension_health.code_stamp.cache_clear()
    monkeypatch.setattr(utils, "get_git_info", get_git_info)

    for _ in range(2):
        extension_loader.reconcile_extension(
            "standalone_stamp", drive_root, lambda: {}, repo_path=str(repo_root)
        )

    assert calls["n"] == 2


def test_reconcile_reverts_enabled_on_load_error(tmp_path):
    """Atomic enable: a failed enable-time load reverts enabled.json to False."""
    from ouroboros.skill_loader import load_enabled

    plugin = "def register(api):\n    raise RuntimeError('boom in register')\n"
    loaded, repo_root, drive_root = _prepare_extension(tmp_path, "boomext", plugin, permissions=[])
    assert load_enabled(drive_root, "boomext") is True

    state = extension_loader.reconcile_extension(
        "boomext", drive_root, lambda: {}, repo_path=str(repo_root),
        retry_load_error=True, revert_enabled_on_error=True,
    )
    assert state.get("action") == "extension_load_error"
    assert state.get("reverted_enabled") is True
    assert load_enabled(drive_root, "boomext") is False


def test_reconcile_does_not_revert_when_flag_off(tmp_path):
    """Non-enable reconcile (default flag) must not disable a skill on load error."""
    from ouroboros.skill_loader import load_enabled

    plugin = "def register(api):\n    raise RuntimeError('boom')\n"
    loaded, repo_root, drive_root = _prepare_extension(tmp_path, "boomext2", plugin, permissions=[])
    state = extension_loader.reconcile_extension(
        "boomext2", drive_root, lambda: {}, repo_path=str(repo_root), retry_load_error=True,
    )
    assert state.get("action") == "extension_load_error"
    assert load_enabled(drive_root, "boomext2") is True
