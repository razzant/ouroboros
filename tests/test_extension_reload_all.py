"""``reload_all`` and the staged-import sweep.

Split out of ``tests/test_extension_loader.py`` when that module was divided by
theme; every moved block is verbatim. Covers the conflict fail-closed, the
settings-save and server-startup wiring pins, stale extensions torn down, one
extension's exception not blocking the rest, per-extension load-error logging,
the staged import root cleanup, and the per-PID sweep predicate that keeps live
peers and grace-fresh trees while reaping dead orphans.
"""

from __future__ import annotations

import pathlib

from ouroboros import extension_loader
from ouroboros.skill_loader import SkillReviewState, find_skill, save_enabled, save_review_state

from tests._shared import clean_extension_runtime_state
from tests._extension_loader_shared import (
    _prepare_extension,
    _write_ext_skill,
)
from tests._extension_loader_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clear_loader_state,
)


def test_reload_all_fails_closed_when_conflicting_extensions_are_both_enabled(tmp_path):
    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    plugin = (
        "def register(api):\n"
        "    api.register_tool('ping', lambda ctx: 'ok', description='ping', schema={})\n"
    )
    telegram_dir = _write_ext_skill(
        repo_root,
        "telegram",
        plugin_body=plugin,
        permissions=["tool"],
        extra_frontmatter="conflicts: [telegram-bridge]\n",
    )
    bridge_dir = _write_ext_skill(
        repo_root,
        "telegram-bridge",
        plugin_body=plugin,
        permissions=["tool"],
    )
    for name, skill_dir in (("telegram", telegram_dir), ("telegram-bridge", bridge_dir)):
        loaded = find_skill(drive_root, name, repo_path=str(repo_root))
        assert loaded is not None
        save_enabled(drive_root, name, True)
        save_review_state(
            drive_root,
            name,
            SkillReviewState(status="pass", content_hash=loaded.content_hash),
        )

    results = extension_loader.reload_all(
        drive_root,
        lambda: {},
        repo_path=str(repo_root),
    )

    assert results == {
        "telegram": "skill_conflict",
        "telegram-bridge": "skill_conflict",
    }
    assert extension_loader.snapshot()["extensions"] == []

    save_enabled(drive_root, "telegram-bridge", False)
    state = extension_loader.reconcile_extension(
        "telegram",
        drive_root,
        lambda: {},
        repo_path=str(repo_root),
    )
    assert state["action"] == "extension_loaded"
    assert "telegram" in extension_loader.snapshot()["extensions"]


def test_reload_all_called_on_settings_save():
    """Phase 4 regression: ``server.py::api_settings_post`` must
    reconcile the live extension registry when OUROBOROS_SKILLS_REPO_PATH
    changes; otherwise switching repo path leaves stale extensions
    registered from the old path."""
    import ast
    src = (
        pathlib.Path(__file__).resolve().parent.parent
        / "ouroboros"
        / "gateway"
        / "settings.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    # The reload lives in the extracted post-save side-effects helper; the pin
    # follows the seam so the regression teeth survive the extraction: the
    # endpoint must reach the helper, and the helper must reach the reload.
    endpoint_text = sync_body_text = locked_body_text = helper_text = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "api_settings_post":
            endpoint_text = ast.unparse(node)
        if isinstance(node, ast.FunctionDef) and node.name == "_api_settings_post_sync":
            sync_body_text = ast.unparse(node)
        if isinstance(node, ast.FunctionDef) and node.name == "_api_settings_post_locked":
            locked_body_text = ast.unparse(node)
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_settings_save_side_effects":
            helper_text = ast.unparse(node)
    # The endpoint hands the whole save body to a worker thread (the loop must
    # not freeze for the save) and the thread serializes under the save lock;
    # the chain the pin protects is endpoint -> sync (lock) -> locked body ->
    # side-effects helper -> reload.
    assert "_api_settings_post_sync" in endpoint_text, (
        "api_settings_post must delegate the save body off the event loop."
    )
    assert "_api_settings_post_locked" in sync_body_text, (
        "the threaded save body must run under the save lock wrapper."
    )
    assert "_apply_settings_save_side_effects" in locked_body_text, (
        "the settings save body must invoke the post-save side-effects helper."
    )
    assert "reload_all" in helper_text or "_reload_extensions" in helper_text, (
        "the post-save side-effects helper must call extension_loader.reload_all "
        "on OUROBOROS_SKILLS_REPO_PATH change."
    )
    assert "OUROBOROS_SKILLS_REPO_PATH" in helper_text
    assert "OUROBOROS_RUNTIME_MODE" in helper_text, (
        "the post-save side-effects helper must also reconcile extensions when "
        "runtime mode changes."
    )


def test_reload_all_called_from_server_startup():
    """Phase 4 regression: server.py main() must call
    ``extension_loader.reload_all`` during startup so enabled extensions
    survive a restart. Without this, only ``toggle_skill`` could ever
    load a plugin. v6.17 also requires the same reload in spawned workers,
    because extension schemas and dispatch registries are process-local."""
    import ast
    src = (pathlib.Path(__file__).resolve().parent.parent / "server.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan":
            body_text = ast.unparse(node)
            assert "_reload_extensions(" in body_text or "reload_all(" in body_text, (
                "server.py does not wire extension_loader.reload_all into startup — "
                "enabled extensions would not survive a process restart."
            )
            assert "if repo_path" not in body_text, (
                "startup extension reload must run even when only bundled "
                "skills are present."
            )
            assert "pytest_default_real_data_dir" in body_text
            assert "Skipping extension reload_all against real DATA_DIR during pytest" in body_text
            break
    else:
        assert False, "lifespan function not found in server.py"

    # v7next D08: worker_main lives in its extraction owner supervisor/worker_process.py
    worker_src = (pathlib.Path(__file__).resolve().parent.parent / "supervisor" / "worker_process.py").read_text(encoding="utf-8")
    worker_tree = ast.parse(worker_src)
    for node in ast.walk(worker_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "worker_main":
            body_text = ast.unparse(node)
            assert "_reload_extensions(" in body_text or "reload_all(" in body_text, (
                "supervisor worker_main must reload enabled extension tools before make_agent; "
                "otherwise worker processes expose a smaller tool surface than server schemas."
            )
            assert body_text.index("_reload_extensions") < body_text.index("make_agent"), (
                "worker extension reload must happen before make_agent builds ToolRegistry schemas."
            )
            assert "pytest_default_real_data_dir" in body_text
            return
    assert False, "worker_main function not found in supervisor/worker_process.py"


def test_reload_all_tears_down_stale_extensions(tmp_path):
    """reload_all must unload extensions that no longer exist on disk."""
    plugin = (
        "def register(api):\n"
        "    api.register_tool('t', lambda ctx: 'ok', description='', schema={})\n"
    )
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "staleish",
        plugin,
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None
    assert "staleish" in extension_loader.snapshot()["extensions"]
    # Nuke the skill directory; reload_all should tear it down.
    import shutil
    shutil.rmtree(repo_root / "staleish")
    extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root))
    assert "staleish" not in extension_loader.snapshot()["extensions"]


def test_reload_all_reads_git_info_once_for_the_health_batch(tmp_path, monkeypatch):
    """One startup reconciliation batch must share one fresh code stamp."""
    from ouroboros import extension_health, utils

    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    for name in ("stamp_a", "stamp_b"):
        _write_ext_skill(
            repo_root,
            name,
            plugin_body="def register(api):\n    pass\n",
            permissions=[],
        )

    calls = {"n": 0}

    def get_git_info(_repo):
        calls["n"] += 1
        return "main", "feedface"

    extension_health.code_stamp.cache_clear()
    monkeypatch.setattr(utils, "get_git_info", get_git_info)

    extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root))

    assert calls["n"] == 1


def test_reload_all_continues_after_one_extension_exception(tmp_path, monkeypatch, caplog):
    """A reconcile bug in one extension must not block later extensions."""
    import logging

    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    plugin = (
        "def register(api):\n"
        "    api.register_tool('t', lambda ctx: 'ok', description='', schema={})\n"
    )
    for name in ("a_bad", "z_good"):
        _write_ext_skill(repo_root, name, plugin_body=plugin, permissions=["tool"])
        loaded = find_skill(drive_root, name, repo_path=str(repo_root))
        assert loaded is not None
        save_enabled(drive_root, name, True)
        save_review_state(drive_root, name, SkillReviewState(status="pass", content_hash=loaded.content_hash))

    original_reconcile = extension_loader.reconcile_extension

    def flaky_reconcile(skill_name, *args, **kwargs):
        if skill_name == "a_bad":
            raise RuntimeError("boom")
        return original_reconcile(skill_name, *args, **kwargs)

    monkeypatch.setattr(extension_loader, "reconcile_extension", flaky_reconcile)

    with caplog.at_level(logging.ERROR):
        results = extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root))

    assert "RuntimeError: boom" in results["a_bad"]
    assert results["z_good"] is None
    assert "z_good" in extension_loader.snapshot()["extensions"]
    assert any("Extension reload failed for a_bad; continuing" in rec.message for rec in caplog.records)


def test_reload_all_logs_per_extension_load_error(tmp_path, caplog):
    import logging

    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    _write_ext_skill(
        repo_root,
        "bad_register",
        plugin_body="def register(api):\n    raise RuntimeError('register failed')\n",
        permissions=[],
    )
    loaded = find_skill(drive_root, "bad_register", repo_path=str(repo_root))
    assert loaded is not None
    save_enabled(drive_root, "bad_register", True)
    save_review_state(drive_root, "bad_register", SkillReviewState(status="pass", content_hash=loaded.content_hash))

    with caplog.at_level(logging.ERROR):
        results = extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root))

    assert "register failed" in str(results["bad_register"])
    assert any("Extension reload failed for bad_register" in rec.message for rec in caplog.records)


def test_clean_extension_runtime_state_unloads_staged_import_root(tmp_path):
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path,
        "cleanup_ext",
        "def register(api):\n    pass\n",
        [],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    with extension_loader._lock:
        import_root = pathlib.Path(extension_loader._extensions["cleanup_ext"].import_root)
    assert import_root.exists()

    clean_extension_runtime_state()

    assert not import_root.exists()
    assert "cleanup_ext" not in extension_loader.snapshot()["extensions"]


def test_reload_all_sweeps_stale_extension_imports(tmp_path, monkeypatch):
    import os as _os, time as _time, uuid as _uuid
    _DEAD = 999999  # owner PID treated as dead by the stub below
    monkeypatch.setattr("ouroboros.platform_layer.pid_is_alive", lambda pid: pid != _DEAD)
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "sweep_ext",
        "def register(api):\n    pass\n",
        [],
    )
    imports_dir = drive_root / "state" / "skills" / "sweep_ext" / "__extension_imports"
    # A genuine orphan: owner PID dead AND mtime past the spawn grace (per-PID leaf name).
    stale_root = imports_dir / f"{_DEAD}-{_uuid.uuid4().hex}"
    (stale_root / "skill").mkdir(parents=True)
    _old = _time.time() - 2 * extension_loader._IMPORT_SWEEP_GRACE_SEC
    _os.utime(stale_root, (_old, _old))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))

    results = extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root))

    assert results["sweep_ext"] is None
    assert not stale_root.exists()  # dead-owner + past-grace orphan reaped
    live_roots = list(imports_dir.iterdir())
    assert len(live_roots) == 1
    # The freshly-staged tree is tagged with THIS process's PID.
    assert live_roots[0].name.startswith(f"{_os.getpid()}-")
    assert (live_roots[0] / "skill").exists()


def test_reload_all_preserves_live_import_root_while_sweeping_stale_roots(tmp_path, monkeypatch):
    import os as _os, time as _time, uuid as _uuid
    _DEAD = 999999  # dead owner
    _PEER = 888888  # a DIFFERENT, still-alive worker PID
    monkeypatch.setattr("ouroboros.platform_layer.pid_is_alive", lambda pid: pid != _DEAD)
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "live_sweep_ext",
        "def register(api):\n    pass\n",
        [],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    with extension_loader._lock:
        live_root = pathlib.Path(extension_loader._extensions["live_sweep_ext"].import_root)
    imports_dir = drive_root / "state" / "skills" / "live_sweep_ext" / "__extension_imports"
    # A LIVE PEER worker's freshly-staged tree (DIFFERENT, alive PID) must NOT be reaped —
    # the cross-worker race the fix targets; the old single-survivor test could not express
    # it, and the 'skip if pid==mine' miscoding would wrongly delete this.
    peer_root = imports_dir / f"{_PEER}-{_uuid.uuid4().hex}"
    (peer_root / "skill").mkdir(parents=True)
    # A genuine orphan: dead owner + mtime past the spawn grace.
    stale_root = imports_dir / f"{_DEAD}-{_uuid.uuid4().hex}"
    (stale_root / "skill").mkdir(parents=True)
    _old = _time.time() - 2 * extension_loader._IMPORT_SWEEP_GRACE_SEC
    _os.utime(stale_root, (_old, _old))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))

    results = extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root))

    assert results["live_sweep_ext"] is None
    assert live_root.exists()       # this process's live bundle tree kept (keep-set + alive)
    assert peer_root.exists()       # a LIVE peer worker's tree NOT reaped (regression direction)
    assert not stale_root.exists()  # dead + aged orphan reaped
    survivors = set(imports_dir.iterdir())
    assert {live_root, peer_root} <= survivors
    assert stale_root not in survivors


def test_sweep_predicate_keeps_live_peer_and_grace_fresh_reaps_dead_orphan(tmp_path, monkeypatch):
    """Per-PID sweep predicate in isolation (empty keep-set): a live-owner tree and a
    dead-owner-but-within-grace tree survive; a dead-owner past-grace tree and a legacy
    bare-uuid tree are reaped. Pins the MAX_WORKERS>1 cross-worker race fix at the
    predicate level — the 'skip if pid==mine' and dropped-grace miscodings both fail here.
    """
    import os as _os, time as _time, uuid as _uuid
    from ouroboros.skill_loader import skill_state_dir
    _DEAD = 999999
    _PEER = 888888  # different, alive
    monkeypatch.setattr("ouroboros.platform_layer.pid_is_alive", lambda pid: pid != _DEAD)
    drive_root = tmp_path / "drive"
    imports_dir = skill_state_dir(drive_root, "predx") / "__extension_imports"
    imports_dir.mkdir(parents=True)

    def _mk(name, age=0.0):
        d = imports_dir / name
        (d / "skill").mkdir(parents=True)
        if age:
            t = _time.time() - age
            _os.utime(d, (t, t))
        return d

    peer_live = _mk(f"{_PEER}-{_uuid.uuid4().hex}")  # owner alive (different PID)
    dead_fresh = _mk(f"{_DEAD}-{_uuid.uuid4().hex}")  # owner dead, fresh
    dead_old = _mk(f"{_DEAD}-{_uuid.uuid4().hex}", age=2 * extension_loader._IMPORT_SWEEP_GRACE_SEC)
    legacy = _mk(_uuid.uuid4().hex)  # bare-uuid legacy (no parseable owner)
    # An all-digit legacy uuid would int-parse to a huge number; the PID-range guard
    # must treat it as legacy (reaped) and NOT feed it to pid_is_alive (which would
    # OverflowError os.kill and escape the sweep).
    all_digit_legacy = _mk("9" * 32)

    # No bundle registered for "predx" -> keep-set empty -> isolates the new predicate.
    extension_loader._sweep_stale_extension_imports(drive_root, "predx")

    assert peer_live.exists(), "a LIVE peer (different PID) tree must NOT be reaped"
    assert dead_fresh.exists(), "a dead-owner tree within the spawn grace must survive"
    assert not dead_old.exists(), "a dead-owner past-grace orphan must be reaped"
    assert not legacy.exists(), "a legacy bare-uuid tree (no live bundle) is reaped as before"
    assert not all_digit_legacy.exists(), "an all-digit legacy uuid is reaped, not crashed on"
