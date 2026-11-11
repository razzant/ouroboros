import contextlib
import hmac
import json
import logging
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import types

import pytest

import ouroboros.gateway.host_service as host_service

from ouroboros.contracts.plugin_api import ExtensionRegistrationError
from ouroboros.extension_companion import (
    CompanionDescriptor,
    companion_spawn_env,
    CompanionSupervisor,
    init_server_process_pid,
)
from ouroboros.extension_loader import PluginAPIImpl, _PluginAPIConfig
from ouroboros.extension_plugin_api import ExtensionStaleRecoveryError
import ouroboros.extension_loader as extension_loader
import ouroboros.extension_plugin_api as extension_plugin_api
from ouroboros import extension_health
from ouroboros.skill_loader import SkillReviewState, find_skill, save_enabled, save_review_state


def test_companion_supervisor_starts_and_stops_process(tmp_path: pathlib.Path) -> None:
    init_server_process_pid()
    supervisor = CompanionSupervisor(tmp_path)
    descriptor = CompanionDescriptor(
        skill_name="demo",
        name="sleepy",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env={},
    )

    assert supervisor.start(descriptor)
    snapshot = supervisor.snapshot()
    assert "demo:sleepy" in snapshot
    pid = int(snapshot["demo:sleepy"]["pid"])
    assert pid > 0

    supervisor.stop("demo", "sleepy", timeout_sec=1)
    assert supervisor.snapshot() == {}


def test_panic_kill_all_clears_runtime_table(tmp_path: pathlib.Path) -> None:
    init_server_process_pid()
    supervisor = CompanionSupervisor(tmp_path)
    descriptor = CompanionDescriptor(
        skill_name="demo",
        name="panic",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env={},
    )

    assert supervisor.start(descriptor)
    assert supervisor.snapshot()
    supervisor.panic_kill_all()
    assert supervisor.snapshot() == {}


def test_exhausted_companion_remains_in_durable_runtime_health(tmp_path: pathlib.Path, monkeypatch) -> None:
    init_server_process_pid()
    drive_root = tmp_path / "drive"
    repo_root = tmp_path / "skills"
    skill_dir = repo_root / "connector"
    drive_root.mkdir()
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: connector\ndescription: connector\nversion: 0.1.0\n"
        "type: extension\nentry: plugin.py\npermissions: []\n---\nbody\n",
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text("def register(api):\n    pass\n", encoding="utf-8")
    loaded = find_skill(drive_root, "connector", repo_path=str(repo_root))
    assert loaded is not None
    save_enabled(drive_root, loaded.name, True)
    save_review_state(
        drive_root,
        loaded.name,
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )
    loaded = find_skill(drive_root, loaded.name, repo_path=str(repo_root))
    assert loaded is not None
    # Hold the monitor's terminal health writer at entry: until it writes, the exited
    # companion must stay visible. Then join that monitor, whose real writer may wait
    # seconds for its lock (#985); 60 s is only a hang bound.
    real_writer = extension_health.record_companion_restart_exhausted
    writer_entered, release_writer, monitors = threading.Event(), threading.Event(), []

    def held_writer(*args, **kwargs):
        monitors.append(threading.current_thread())
        writer_entered.set()
        release_writer.wait()
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(extension_health, "record_companion_restart_exhausted", held_writer)
    try:
        assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root) is None
        supervisor = CompanionSupervisor(drive_root)
        descriptor = CompanionDescriptor(
            skill_name=loaded.name,
            name="bridge",
            command=[sys.executable, "-c", "raise SystemExit(7)"],
            cwd=tmp_path,
            env={},
            max_restarts=0,
        )

        assert supervisor.start(descriptor)
        assert writer_entered.wait(60), "the monitor never reached the terminal health writer"
        assert supervisor.snapshot()["connector:bridge"]["returncode"] == 7
        release_writer.set()
        monitors[0].join(60)
        assert not monitors[0].is_alive(), "the companion monitor did not finish its terminal lifecycle"

        assert supervisor.snapshot() == {}
        health = extension_health.read_extension_health(drive_root, loaded.name) or {}
        observed = health.get("last_observed") or {}
        assert observed["status"] == extension_health.BROKEN
        assert observed["reason"] == "companion_restart_exhausted:bridge"
        state = extension_loader.runtime_state_for_loaded_skill(
            loaded, drive_root, skills=[loaded],
        )
        assert state["live_loaded"] is True
        assert state["companion_failed"] is True
        assert state["reason"] == "companion_restart_exhausted:bridge"
        assert "exhausting its restart budget" in state["load_error"]
        assert extension_health.status_for_runtime_state(state) == extension_health.BROKEN
    finally:
        release_writer.set()
        extension_loader.unload_extension(loaded.name)


def test_plugin_api_companion_registration_uses_reviewed_manifest_descriptor(tmp_path: pathlib.Path) -> None:
    init_server_process_pid(999999)
    state_dir = tmp_path / "state"
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    api = PluginAPIImpl(_PluginAPIConfig(
        skill_name="demo",
        permissions=["companion_process"],
        env_allowlist=[],
        state_dir=state_dir,
        settings_reader=lambda: {},
        companion_processes=[{
            "name": "daemon",
            "command": ["python3", "scripts/daemon.py"],
            "runtime": "python3",
        }],
        skill_dir=skill_dir,
    ))

    api.register_companion_process("daemon")

    api._close_registration()
    init_server_process_pid()


def test_plugin_api_companion_uses_staged_skill_root_as_cwd(tmp_path: pathlib.Path, monkeypatch) -> None:
    init_server_process_pid()
    staged_skill = tmp_path / "import" / "skill"
    (staged_skill / "scripts").mkdir(parents=True)
    (staged_skill / "scripts" / "daemon.py").write_text("print('ok')\n", encoding="utf-8")
    captured = {}

    class FakeSupervisor:
        def start(self, descriptor):
            captured["descriptor"] = descriptor
            return True

        def stop(self, skill_name, name):
            return True

    # The companion supervisor is read by PluginAPIImpl (its owner) at register
    # time and by the loader's spawn/ensure paths; patch both readers.
    monkeypatch.setattr(extension_plugin_api, "get_global_supervisor", lambda: FakeSupervisor())
    monkeypatch.setattr(extension_loader, "get_global_supervisor", lambda: FakeSupervisor())
    api = PluginAPIImpl(_PluginAPIConfig(
        skill_name="demo",
        permissions=["companion_process"],
        env_allowlist=[],
        state_dir=tmp_path / "state",
        settings_reader=lambda: {},
        companion_processes=[{
            "name": "daemon",
            "command": ["python3", "scripts/daemon.py"],
            "runtime": "python3",
            "env": {"HOST_SERVICE_URL": "https://evil.example", "HOST_SERVICE_TOKEN": "evil"},
        }],
        skill_dir=tmp_path / "mutable",
        runtime_skill_dir=staged_skill,
    ))

    api.register_companion_process("daemon")
    # ABI-9: the spawn is a deferred side effect; publication starts it.
    api._publish_registrations()

    descriptor = captured["descriptor"]
    assert descriptor.cwd == staged_skill
    assert (descriptor.cwd / "scripts" / "daemon.py").is_file()
    assert descriptor.env["HOST_SERVICE_URL"].startswith("http://127.0.0.1:")
    assert descriptor.env["HOST_SERVICE_TOKEN"] != "evil"


_COMPANION_FRONTMATTER = (
    "companion_processes:\n"
    "  - name: daemon\n"
    "    runtime: python3\n"
    "    command: [\"python3\", \"scripts/daemon.py\"]\n"
)


class _RecordingSupervisor:
    """Fake supervisor: records spawns/stops; reports no running companions."""

    def __init__(self) -> None:
        self.started: list = []
        self.stopped: list = []

    def start(self, descriptor):
        self.started.append(descriptor)
        return True

    def snapshot(self):
        return {}

    def stop(self, *args, **kwargs):
        self.stopped.append(args)

    def stop_skill(self, skill_name):
        self.stopped.append((skill_name,))


def _reviewed_companion_skill(
    tmp_path: pathlib.Path, name: str = "compskill", extra_frontmatter: str = "",
):
    """A reviewed+enabled companion extension the recovery tests can load."""
    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir(exist_ok=True)
    skill_dir = repo_root / name
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "scripts" / "daemon.py").write_text("print('ok')\n", encoding="utf-8")
    (skill_dir / "plugin.py").write_text(
        "def register(api):\n    api.register_companion_process('daemon')\n", encoding="utf-8"
    )
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: companion skill\n"
        "version: 0.1.0\n"
        "type: extension\n"
        "entry: plugin.py\n"
        "permissions: [companion_process]\n"
        f"{_COMPANION_FRONTMATTER}"
        f"{extra_frontmatter}"
        "---\n"
        "body\n",
        encoding="utf-8",
    )
    loaded = find_skill(drive_root, name, repo_path=str(repo_root))
    assert loaded is not None
    save_enabled(drive_root, loaded.name, True)
    save_review_state(
        drive_root, loaded.name,
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )
    loaded = find_skill(drive_root, loaded.name, repo_path=str(repo_root))
    assert loaded is not None
    return loaded, repo_root, drive_root


def _patch_supervisor(monkeypatch) -> _RecordingSupervisor:
    # The companion supervisor is read by PluginAPIImpl (its owner) at register
    # time and by the loader's spawn/ensure/unload paths; patch both readers.
    fake = _RecordingSupervisor()
    monkeypatch.setattr(extension_plugin_api, "get_global_supervisor", lambda: fake)
    monkeypatch.setattr(extension_loader, "get_global_supervisor", lambda: fake)
    return fake


def test_publish_out_of_process_registration_host_spawns_declared_name(tmp_path: pathlib.Path, monkeypatch) -> None:
    """Out-of-process catalog -> host spawns the manifest-declared companion; an
    undeclared cataloged name is rejected at the host trust boundary, and an
    ambiguous publication form (neither initial load nor generation-bound
    recovery) is a typed refusal (Ф3.1 fix-round-4)."""
    init_server_process_pid()
    loaded, _repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    fake = _patch_supervisor(monkeypatch)
    # Production-shape per-skill state dir (fix-round-5): the same directory
    # extension_loader resolves via skill_state_dir, where auth_token.json lives.
    state_dir = extension_loader.skill_state_dir(drive_root, loaded.name)
    try:
        with pytest.raises(ExtensionRegistrationError, match="exactly one of"):
            extension_loader._publish_out_of_process_registration(
                loaded,
                catalog={"companions": ["daemon"]},
                state_dir=state_dir,
                settings_reader=lambda: {},
                granted_keys=[],
                dependency_site_dirs_enabled=False,
            )
        with pytest.raises(ExtensionRegistrationError, match="escaped manifest"):
            extension_loader._publish_out_of_process_registration(
                loaded,
                catalog={"companions": ["evil"]},
                state_dir=state_dir,
                settings_reader=lambda: {},
                granted_keys=[],
                dependency_site_dirs_enabled=False,
                current_hash=loaded.content_hash,
            )
        assert fake.started == []
        assert not (state_dir / host_service.AUTH_TOKEN_FILENAME).exists(), (
            "a refused publication must not have minted a companion token"
        )

        extension_loader._publish_out_of_process_registration(
            loaded,
            catalog={"companions": ["daemon"]},
            state_dir=state_dir,
            settings_reader=lambda: {},
            granted_keys=[],
            dependency_site_dirs_enabled=False,
            current_hash=loaded.content_hash,
        )
        assert len(fake.started) == 1
        assert fake.started[0].name == "daemon"
        # Initial load legitimately mints the token — at publication, into the
        # spawned descriptor's env.
        token_payload = json.loads(
            (state_dir / host_service.AUTH_TOKEN_FILENAME).read_text(encoding="utf-8")
        )
        assert fake.started[0].env["HOST_SERVICE_TOKEN"] == token_payload["token"]
    finally:
        extension_loader.unload_extension("compskill")
        init_server_process_pid()


def test_recovery_publication_requires_a_pre_existing_live_bundle(tmp_path: pathlib.Path, monkeypatch) -> None:
    """Ф3.1 fix-round-4 REPLACEMENT (disclosed): the clause this test replaces
    pinned the OPPOSITE contract — the recovery-form publication helper
    accepted no pre-existing live bundle and spawned the companion anyway,
    which pinned the resurrection bug itself (a stale recovery re-created an
    empty companion-only bundle after disable/unload). The recovery form now
    REQUIRES a still-live bundle at the observed generation: with no live
    bundle it refuses with zero effects — filesystem included (fix-round-5:
    no auth_token.json is minted) — and creates nothing."""
    init_server_process_pid()
    loaded, _repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    fake = _patch_supervisor(monkeypatch)
    state_dir = extension_loader.skill_state_dir(drive_root, loaded.name)
    try:
        with pytest.raises(ExtensionStaleRecoveryError):
            extension_loader._publish_out_of_process_registration(
                loaded,
                catalog={"companions": ["daemon"]},
                state_dir=state_dir,
                settings_reader=lambda: {},
                granted_keys=[],
                dependency_site_dirs_enabled=False,
                expected_generation="0" * 32,
            )
        assert fake.started == []
        with extension_loader._lock:
            assert "compskill" not in extension_loader._extensions
        assert "compskill" not in extension_loader.snapshot()["extensions"]
        assert not (state_dir / host_service.AUTH_TOKEN_FILENAME).exists(), (
            "stale recovery minted auth_token.json before its refusal"
        )
    finally:
        extension_loader.unload_extension("compskill")
        init_server_process_pid()


def test_unload_completing_between_snapshot_and_publication_refuses_recovery(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    """Ф3.1 fix-round-4 pin (ABI-9а, TOCTOU): an unload that COMPLETES between
    recovery's liveness/bundle snapshot and its publication turns the
    publication into a typed zero-effect refusal — registries and supervisor
    stay empty and NO new bundle is created. (Before the fix the stale
    recovery re-created an empty companion-only bundle and started its
    companion, resurrecting the extension after disable/unload.)"""
    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    fake = _patch_supervisor(monkeypatch)
    try:
        assert extension_loader.load_extension(
            loaded, lambda: {}, drive_root=drive_root, skills=[loaded],
        ) is None
        assert len(fake.started) == 1  # the initial publication's spawn
        # Fix-round-5: the token file the initial publication minted must
        # survive the stale refusal byte-for-byte (unload keeps it; only a
        # post-fence publication may touch it).
        token_path = (
            extension_loader.skill_state_dir(drive_root, loaded.name)
            / host_service.AUTH_TOKEN_FILENAME
        )
        token_bytes = token_path.read_bytes()

        real_state_path = extension_loader.skill_state_dir_path

        def _unload_wins_the_race(drive_root_arg, skill_name_arg):
            # Deterministic interleave: the concurrent unload COMPLETES after
            # recovery snapshotted liveness/bundle and before it publishes.
            extension_loader.unload_extension(skill_name_arg)
            return real_state_path(drive_root_arg, skill_name_arg)

        monkeypatch.setattr(extension_loader, "skill_state_dir_path", _unload_wins_the_race)
        result = extension_loader.ensure_companions_running(
            loaded.name, drive_root, lambda: {}, repo_path=str(repo_root),
        )

        assert result["action"] == "stale_recovery_refused"
        assert result["started"] == []
        assert "recovery publication refused" in result["reason"]
        assert len(fake.started) == 1, "stale recovery must not start a companion"
        with extension_loader._lock:
            assert loaded.name not in extension_loader._extensions
            assert not any(
                entry.get("skill") == loaded.name
                for entry in extension_loader._tools.values()
            )
        assert loaded.name not in extension_loader.snapshot()["extensions"]
        assert token_path.read_bytes() == token_bytes, (
            "stale recovery touched auth_token.json"
        )
    finally:
        extension_loader.unload_extension(loaded.name)
        init_server_process_pid()


def test_recovery_publication_refuses_on_generation_mismatch_without_effects(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    """Ф3.1 fix-round-4 pin (ABI-9б), rebuilt through the REAL recovery entry
    in fix-round-5: an unload/reload that COMPLETES between recovery's
    generation snapshot and its publication leaves a live bundle under a NEW
    generation — the recovery publication naming the old generation is a
    typed zero-effect refusal: the reloaded bundle keeps its generation, no
    companion is started, and auth_token.json is byte-untouched."""
    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    fake = _patch_supervisor(monkeypatch)
    try:
        assert extension_loader.load_extension(
            loaded, lambda: {}, drive_root=drive_root, skills=[loaded],
        ) is None
        first_generation = extension_loader.extension_generation_digest(loaded.name)
        assert first_generation
        token_path = (
            extension_loader.skill_state_dir(drive_root, loaded.name)
            / host_service.AUTH_TOKEN_FILENAME
        )
        token_bytes = token_path.read_bytes()

        real_state_path = extension_loader.skill_state_dir_path

        def _reload_wins_the_race(drive_root_arg, skill_name_arg):
            # Deterministic interleave: the unload/reload COMPLETES after
            # recovery snapshotted the generation and before it publishes;
            # the bundle EXISTS again, under a fresh generation. Restore the
            # real resolver first — the reload itself resolves state dirs.
            monkeypatch.setattr(extension_loader, "skill_state_dir_path", real_state_path)
            extension_loader.unload_extension(skill_name_arg)
            assert extension_loader.load_extension(
                loaded, lambda: {}, drive_root=drive_root_arg, skills=[loaded],
            ) is None
            return real_state_path(drive_root_arg, skill_name_arg)

        monkeypatch.setattr(extension_loader, "skill_state_dir_path", _reload_wins_the_race)
        result = extension_loader.ensure_companions_running(
            loaded.name, drive_root, lambda: {}, repo_path=str(repo_root),
        )

        assert result["action"] == "stale_recovery_refused"
        assert result["started"] == []
        second_generation = extension_loader.extension_generation_digest(loaded.name)
        assert second_generation and second_generation != first_generation
        with extension_loader._lock:
            assert loaded.name in extension_loader._extensions
        assert len(fake.started) == 2, (
            "only the initial load and the in-window reload may spawn"
        )
        assert token_path.read_bytes() == token_bytes
    finally:
        extension_loader.unload_extension(loaded.name)
        init_server_process_pid()


def test_stale_recovery_does_not_break_live_publication_authorization(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    """Ф3.1 fix-round-5 pin (HIGH): a recovery holding a STALE payload snapshot
    (an old skill root that still exists with the pre-update content) loses
    the race to an unload/reload with CHANGED content; its typed refusal must
    leave auth_token.json byte-untouched. Pre-fix, the recovery minted the
    token DURING descriptor build — the stale root's content hash mismatched
    the G2-bound token file, so the mint ROTATED it before the generation
    fence refused the publication — and the live G2 companion, spawned with
    the current token in its env while the Host Service rereads the file on
    every request, was left permanently unauthorized."""
    init_server_process_pid()
    loaded_v1, repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    fake = _patch_supervisor(monkeypatch)
    try:
        assert extension_loader.load_extension(
            loaded_v1, lambda: {}, drive_root=drive_root, skills=[loaded_v1],
        ) is None
        token_path = (
            extension_loader.skill_state_dir(drive_root, loaded_v1.name)
            / host_service.AUTH_TOKEN_FILENAME
        )
        # The update installs v2 under a NEW payload root; the G1 root
        # (repo_root) survives on disk with the v1 content — the recovery's
        # stale snapshot keeps pointing there.
        repo_root_v2 = tmp_path / "skills_v2"
        shutil.copytree(repo_root, repo_root_v2)
        (repo_root_v2 / loaded_v1.name / "scripts" / "daemon.py").write_text(
            "print('v2')\n", encoding="utf-8"
        )
        real_state_path = extension_loader.skill_state_dir_path
        g2_token_bytes: dict = {}

        def _reload_v2_wins_the_race(drive_root_arg, skill_name_arg):
            # Deterministic interleave: the unload + v2 reload COMPLETES after
            # recovery snapshotted generation/payload and before it publishes;
            # the G2 token is bound to the NEW root's content hash while the
            # stale recovery still holds the v1 snapshot at the old root.
            monkeypatch.setattr(extension_loader, "skill_state_dir_path", real_state_path)
            extension_loader.unload_extension(skill_name_arg)
            loaded_v2 = find_skill(
                drive_root, skill_name_arg, repo_path=str(repo_root_v2),
            )
            assert loaded_v2 is not None
            assert loaded_v2.content_hash != loaded_v1.content_hash
            save_review_state(
                drive_root, skill_name_arg,
                SkillReviewState(status="pass", content_hash=loaded_v2.content_hash),
            )
            loaded_v2 = find_skill(
                drive_root, skill_name_arg, repo_path=str(repo_root_v2),
            )
            assert extension_loader.load_extension(
                loaded_v2, lambda: {}, drive_root=drive_root_arg, skills=[loaded_v2],
            ) is None
            g2_token_bytes["value"] = token_path.read_bytes()
            return real_state_path(drive_root_arg, skill_name_arg)

        monkeypatch.setattr(extension_loader, "skill_state_dir_path", _reload_v2_wins_the_race)
        result = extension_loader.ensure_companions_running(
            loaded_v1.name, drive_root, lambda: {}, repo_path=str(repo_root),
            selected_skill=loaded_v1,  # the stale v1 snapshot the recovery holds
        )

        assert result["action"] == "stale_recovery_refused"
        assert token_path.read_bytes() == g2_token_bytes["value"], (
            "stale recovery rotated auth_token.json; the live publication's "
            "companion is now permanently unauthorized"
        )
        # The live G2 companion's spawn-env token still matches the file the
        # Host Service rereads per request...
        g2_env_token = fake.started[-1].env["HOST_SERVICE_TOKEN"]
        file_token = json.loads(token_path.read_text(encoding="utf-8"))["token"]
        assert hmac.compare_digest(file_token, g2_env_token)
        # ...and end-to-end host authorization for G2 still succeeds.
        # (find_skill is patched only because HostServiceContext resolves the
        # default skills repo path, not this test's temporary v2 root.)
        monkeypatch.setattr(
            host_service, "find_skill",
            lambda data_dir, name: find_skill(
                data_dir, name, repo_path=str(repo_root_v2),
            ),
        )
        ctx = host_service.HostServiceContext(data_dir=drive_root)
        authed_name, payload = ctx.authenticate_token_payload(g2_env_token)
        assert authed_name == loaded_v1.name
        assert payload["token"] == g2_env_token
    finally:
        extension_loader.unload_extension(loaded_v1.name)
        init_server_process_pid()


def _data_root_tree(root: pathlib.Path) -> list:
    # Names AND content identity (size + mtime): a rewrite-in-place of an
    # existing file is a mutation this snapshot must catch (fix-round-7).
    out = []
    for path in sorted(root.rglob("*")):
        stat = path.stat()
        out.append((str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
    return out


def test_stale_recovery_with_env_from_settings_has_zero_filesystem_effects(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    """Ф3.1 fix-round-6 pin (MEDIUM), hardened in fix-round-7: after the
    recovery's state-dir resolution and before the generation fence there is
    no ``load_settings`` read at EITHER seam (``skill_exec.load_settings``
    used by ``_scrub_env`` and the direct ``config.load_settings`` used by
    the grant projection — the legitimate pre-fence grant/liveness projection
    calls land before the tripwire arms), no settings-lock file appears, and
    the data-root tree is identical by name, size and mtime. The
    settings-derived value, the manifest companion env overlay, the isolated
    PYTHONPATH, the state dir and the auth token materialize only in the
    post-fence attach — proved on the initial publication's spawn env."""
    init_server_process_pid()
    import ouroboros.config as config_mod
    import ouroboros.tools.skill_exec as skill_exec

    # Repoint settings beside this test so the lock-file assertion cannot race
    # a concurrent test process sharing the run-wide settings path.
    monkeypatch.setattr(config_mod, "SETTINGS_PATH", tmp_path / "settings.json")
    settings_calls: list = []
    real_load_settings = config_mod.load_settings

    def _counted_load_settings():
        settings_calls.append(1)
        settings = dict(real_load_settings())
        settings["EXT_DEMO_VALUE"] = "demo-from-settings"
        return settings

    # Both seams: skill_exec's bound reference (_scrub_env) and the direct
    # config attribute (requested_core_setting_keys and any other caller).
    monkeypatch.setattr(config_mod, "load_settings", _counted_load_settings)
    monkeypatch.setattr(skill_exec, "load_settings", _counted_load_settings)

    loaded, repo_root, drive_root = _reviewed_companion_skill(
        tmp_path, extra_frontmatter=(
            "    env:\n"
            "      EXT_OVERLAY: from-manifest\n"
            "env_from_settings: [EXT_DEMO_VALUE]\n"
        ),
    )
    # An isolated-dep site dir so the spawn env must carry PYTHONPATH
    # (.ouroboros_env is outside the content hash by construction).
    site_dir = loaded.skill_dir / ".ouroboros_env" / "python" / "lib" / "python{}.{}".format(*sys.version_info[:2]) / "site-packages"
    site_dir.mkdir(parents=True)
    # EXT_DEMO_VALUE present in settings = custom secret; grant it.
    from ouroboros.skill_loader import save_skill_grants

    save_skill_grants(
        drive_root, loaded.name, ["EXT_DEMO_VALUE"],
        content_hash=loaded.content_hash, requested_keys=["EXT_DEMO_VALUE"],
    )
    fake = _patch_supervisor(monkeypatch)
    lock_path = pathlib.Path(str(config_mod.SETTINGS_PATH) + ".lock")
    try:
        assert extension_loader.load_extension(
            loaded, lambda: {}, drive_root=drive_root, skills=[loaded],
        ) is None
        # The initial publication DID materialize the env — post-fence — and
        # actually DELIVERED every plane: the settings-derived value, the
        # manifest overlay, the isolated PYTHONPATH and the bridge keys.
        assert settings_calls, "publication must read settings for env_from_settings"
        env = fake.started[0].env
        assert env["OUROBOROS_SKILL_NAME"] == loaded.name
        assert env["HOST_SERVICE_TOKEN"]
        assert env["HOST_SERVICE_URL"].startswith("http://")
        assert env["EXT_DEMO_VALUE"] == "demo-from-settings"
        assert env["EXT_OVERLAY"] == "from-manifest"
        assert str(site_dir.resolve()) in (env.get("PYTHONPATH") or "")

        real_state_path = extension_loader.skill_state_dir_path
        snapshot: dict = {}

        def _unload_wins_and_snapshots(drive_root_arg, skill_name_arg):
            # Deterministic interleave (as in the round-4 pin), plus the
            # tripwires armed exactly at the recovery's state-dir resolution:
            # the measured window is "after this snapshot, up to the fence".
            extension_loader.unload_extension(skill_name_arg)
            settings_calls.clear()
            snapshot["tree"] = _data_root_tree(drive_root_arg)
            assert not lock_path.exists()
            return real_state_path(drive_root_arg, skill_name_arg)

        monkeypatch.setattr(extension_loader, "skill_state_dir_path", _unload_wins_and_snapshots)
        result = extension_loader.ensure_companions_running(
            loaded.name, drive_root, lambda: {}, repo_path=str(repo_root),
        )

        assert result["action"] == "stale_recovery_refused"
        assert settings_calls == [], (
            "stale recovery read settings pre-fence (lock file / migration hazard)"
        )
        assert not lock_path.exists(), "settings lock file left behind in the window"
        assert _data_root_tree(drive_root) == snapshot["tree"], (
            "stale recovery mutated the data root at/before its refusal"
        )
    finally:
        extension_loader.unload_extension(loaded.name)
        init_server_process_pid()


def test_transient_hash_error_never_rotates_a_valid_token(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    """Ф3.1 fix-round-7 pin (LOW): a transient hash/read failure inside
    ``mint_skill_token`` reuses the stored valid token BYTE-FOR-BYTE instead
    of rotating it — the running companion whose spawn env holds that token
    stays authorized end-to-end. Pre-fix, any hash error collapsed to
    ``content_hash=\"\"``, mismatched the stored hash and rotated the live
    token, de-authorizing the running companion instantly."""
    init_server_process_pid()
    loaded, repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    fake = _patch_supervisor(monkeypatch)
    try:
        assert extension_loader.load_extension(
            loaded, lambda: {}, drive_root=drive_root, skills=[loaded],
        ) is None
        env_token = fake.started[0].env["HOST_SERVICE_TOKEN"]
        state_dir = extension_loader.skill_state_dir_path(drive_root, loaded.name)
        token_path = state_dir / host_service.AUTH_TOKEN_FILENAME
        before = token_path.read_bytes()

        def _transient_failure(*_args, **_kwargs):
            raise OSError("transient read failure")

        monkeypatch.setattr(
            extension_plugin_api, "compute_content_hash", _transient_failure,
        )
        token = extension_plugin_api.mint_skill_token(
            state_dir, loaded.name, loaded.skill_dir,
        )
        assert token == env_token
        assert token_path.read_bytes() == before, (
            "transient hash error rotated a live valid token"
        )
        # End-to-end: the running companion's spawn-env token still authorizes.
        monkeypatch.setattr(
            host_service, "find_skill",
            lambda data_dir, name: find_skill(
                data_dir, name, repo_path=str(repo_root),
            ),
        )
        ctx = host_service.HostServiceContext(data_dir=drive_root)
        authed_name, payload = ctx.authenticate_token_payload(env_token)
        assert authed_name == loaded.name
        assert payload["token"] == env_token
    finally:
        extension_loader.unload_extension(loaded.name)
        init_server_process_pid()


def test_transient_hash_error_without_reusable_token_fails_closed(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    """Ф3.1 fix-round-7 pin (LOW): with no reusable stored token, a transient
    hash failure is a typed fail-closed refusal — never a token minted
    against an empty content hash."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    def _transient_failure(*_args, **_kwargs):
        raise OSError("transient read failure")

    monkeypatch.setattr(
        extension_plugin_api, "compute_content_hash", _transient_failure,
    )
    with pytest.raises(extension_plugin_api.SkillTokenHashUnavailableError):
        extension_plugin_api.mint_skill_token(state_dir, "demo", tmp_path / "skill")
    assert not (state_dir / host_service.AUTH_TOKEN_FILENAME).exists()


def test_generation_bound_disposal_skips_a_newer_publication(
    tmp_path: pathlib.Path, monkeypatch, caplog,
) -> None:
    """Ф3.1 fix-round-4 pin (ABI-9в): disposal bound to an OLD generation is a
    disclosed no-op when the live publication is newer — recovery cleanup can
    never unload a publication it did not make. The same call naming the live
    generation unloads it."""
    init_server_process_pid()
    loaded, _repo_root, drive_root = _reviewed_companion_skill(tmp_path)
    _patch_supervisor(monkeypatch)
    try:
        assert extension_loader.load_extension(
            loaded, lambda: {}, drive_root=drive_root, skills=[loaded],
        ) is None
        live_generation = extension_loader.extension_generation_digest(loaded.name)
        assert live_generation

        with caplog.at_level(logging.WARNING, logger="ouroboros.extension_loader"):
            assert extension_loader.unload_extension(
                loaded.name, expected_generation="f" * 32,
            ) is False
        assert "generation-bound disposal skipped" in caplog.text
        with extension_loader._lock:
            assert loaded.name in extension_loader._extensions

        assert extension_loader.unload_extension(
            loaded.name, expected_generation=live_generation,
        ) is True
        with extension_loader._lock:
            assert loaded.name not in extension_loader._extensions
    finally:
        extension_loader.unload_extension(loaded.name)
        init_server_process_pid()

def _node_companion_api(tmp_path: pathlib.Path, monkeypatch, *, runtime: str, command: list[str]) -> PluginAPIImpl:
    init_server_process_pid()
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir(exist_ok=True)

    class FakeSupervisor:
        def start(self, descriptor):
            return True

    monkeypatch.setattr(extension_plugin_api, "get_global_supervisor", lambda: FakeSupervisor())
    return PluginAPIImpl(_PluginAPIConfig(
        skill_name="demo",
        permissions=["companion_process"],
        env_allowlist=[],
        state_dir=tmp_path / "state",
        settings_reader=lambda: {},
        companion_processes=[{
            "name": "daemon",
            "command": command,
            "runtime": runtime,
        }],
        skill_dir=skill_dir,
    ))


def _staged_companion_env(api: PluginAPIImpl) -> tuple:
    """The staged descriptor plus its env AFTER the post-fence materialization
    (the campaign structure defers env to `extension_companion.companion_spawn_env`)."""

    spawn = api._staged.companion_spawns[0]
    spawn.descriptor.env.update(companion_spawn_env(spawn.spec, "test-token", env_allow=api._env_allow, granted_upper=api._granted_upper, skill=api._skill, skill_dir=api._skill_dir, state_dir=api._state_dir))
    return spawn.descriptor, spawn.descriptor.env


def test_companion_node_command_rewritten_via_policy_helper(tmp_path: pathlib.Path, monkeypatch) -> None:
    """Node symmetry with the python->sys.executable rewrite: the policy helper
    (bundled-first + health rollback) owns the choice; a healthy PATH means no
    emergency, so the child env stays byte-identical."""
    import os

    from ouroboros import node_runtime

    selected = str(tmp_path / "bundle" / "bin" / "node")
    monkeypatch.setattr(
        node_runtime, "select_skill_node_runtime", lambda timeout_sec=10: (selected, "bundled")
    )
    monkeypatch.setattr(node_runtime, "skill_node_emergency_path_dir", lambda timeout_sec=10: "")
    api = _node_companion_api(tmp_path, monkeypatch, runtime="node", command=["node", "server.js"])

    api.register_companion_process("daemon")

    descriptor, env = _staged_companion_env(api)
    assert descriptor.command == [selected, "server.js"]
    assert env.get("PATH", os.environ["PATH"]) == os.environ["PATH"]


def test_companion_npm_not_rewritten_but_emergency_path_prepended(tmp_path: pathlib.Path, monkeypatch) -> None:
    """npm is NOT bundled: argv stays npm; in the emergency state (PATH node
    dead, bundled selected) the child PATH gains the bundled-node dir so npm's
    `#!/usr/bin/env node` shebang finds the working runtime."""
    import os

    from ouroboros import node_runtime

    bundle_bin = str(tmp_path / "bundle" / "bin")
    monkeypatch.setattr(
        node_runtime,
        "select_skill_node_runtime",
        lambda timeout_sec=10: (str(pathlib.Path(bundle_bin) / "node"), "bundled"),
    )
    monkeypatch.setattr(
        node_runtime, "skill_node_emergency_path_dir", lambda timeout_sec=10: bundle_bin
    )
    api = _node_companion_api(tmp_path, monkeypatch, runtime="npm", command=["npm", "run", "start"])

    api.register_companion_process("daemon")

    descriptor, env = _staged_companion_env(api)
    assert descriptor.command == ["npm", "run", "start"]
    assert env["PATH"].split(os.pathsep)[0] == bundle_bin
    assert env["PATH"].endswith(os.environ["PATH"])


def test_companion_node_unusable_leaves_command_and_env_untouched(tmp_path: pathlib.Path, monkeypatch) -> None:
    """Nothing usable -> honest failure at spawn time, no rewrite, no env edit."""
    import os

    from ouroboros import node_runtime

    monkeypatch.setattr(
        node_runtime,
        "select_skill_node_runtime",
        lambda timeout_sec=10: ("", "bundled:absent; path:broken:signal:SIGKILL"),
    )
    monkeypatch.setattr(node_runtime, "skill_node_emergency_path_dir", lambda timeout_sec=10: "")
    api = _node_companion_api(tmp_path, monkeypatch, runtime="node", command=["node", "server.js"])

    api.register_companion_process("daemon")

    descriptor, env = _staged_companion_env(api)
    assert descriptor.command == ["node", "server.js"]
    assert env.get("PATH", os.environ["PATH"]) == os.environ["PATH"]




def test_windows_companion_start_does_not_request_console_process_group(tmp_path: pathlib.Path, monkeypatch) -> None:
    init_server_process_pid()
    captured = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-should-not-leak")
    monkeypatch.setenv("windir", "C:\\Windows")
    monkeypatch.setenv("ComSpec", "C:\\Windows\\System32\\cmd.exe")

    class FakeProcess:
        pid = 12345
        stdout = None
        stderr = None

        def poll(self):
            return None

        def wait(self):
            return 0

    monkeypatch.setattr("ouroboros.extension_companion.IS_WINDOWS", True)
    monkeypatch.setattr("ouroboros.extension_companion.create_kill_on_close_job", lambda: object())
    monkeypatch.setattr("ouroboros.extension_companion.assign_pid_to_job", lambda _job, _pid: True)
    monkeypatch.setattr("ouroboros.extension_companion.close_job", lambda _job: None)

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr("ouroboros.extension_companion.subprocess.Popen", fake_popen)
    supervisor = CompanionSupervisor(tmp_path)
    descriptor = CompanionDescriptor(
        skill_name="skill",
        name="daemon",
        command=["python", "-c", "print('ok')"],
        cwd=tmp_path,
        env={},
    )

    assert supervisor.start(descriptor) is True
    assert captured.get("creationflags", 0) & getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) == 0
    captured_upper = {key.upper(): value for key, value in captured["env"].items()}
    assert "PATH" in captured_upper
    assert captured_upper["WINDIR"] == "C:\\Windows"
    assert captured_upper["COMSPEC"].endswith("cmd.exe")
    assert "OPENROUTER_API_KEY" not in captured["env"]


def test_companion_processes_start_outside_the_registry_lock_after_publication(tmp_path: pathlib.Path, monkeypatch) -> None:
    """A companion spawn forks a child and probes its ports — seconds, under
    load. Every other extension call (dispatch, event delivery, a sibling
    skill's load) serializes on the registry lock, so a spawn held UNDER it
    stalled the whole extension surface (the Telegram companion wedge). The
    process starts after the lock is released, against the already-published
    bundle; a start failure still routes to the standard dispose+unload path."""
    init_server_process_pid()
    staged_skill = tmp_path / "import" / "skill"
    (staged_skill / "scripts").mkdir(parents=True)
    (staged_skill / "scripts" / "daemon.py").write_text("print('ok')\n", encoding="utf-8")
    observed = {}

    class LockProbeSupervisor:
        def __init__(self, fail=False):
            self.fail = fail

        def start(self, descriptor):
            free = extension_plugin_api._lock.acquire(blocking=False)
            if free:
                extension_plugin_api._lock.release()
            observed["lock_free_at_start"] = free
            observed["published_at_start"] = extension_plugin_api._extensions.get("demo") is not None
            if self.fail:
                raise RuntimeError("spawn failed")
            return True

        def stop(self, skill_name, name):
            return True

    def _api(supervisor):
        monkeypatch.setattr(extension_plugin_api, "get_global_supervisor", lambda: supervisor)
        monkeypatch.setattr(extension_loader, "get_global_supervisor", lambda: supervisor)
        api = PluginAPIImpl(_PluginAPIConfig(
            skill_name="demo",
            permissions=["companion_process"],
            env_allowlist=[],
            state_dir=tmp_path / "state",
            settings_reader=lambda: {},
            companion_processes=[{
                "name": "daemon",
                "command": ["python3", "scripts/daemon.py"],
                "runtime": "python3",
            }],
            skill_dir=tmp_path / "mutable",
            runtime_skill_dir=staged_skill,
        ))
        api.register_companion_process("daemon")
        return api

    api = _api(LockProbeSupervisor())
    api._publish_registrations()
    assert observed["lock_free_at_start"] is True
    assert observed["published_at_start"] is True

    with extension_plugin_api._lock:
        extension_plugin_api._extensions.pop("demo", None)
    failing = _api(LockProbeSupervisor(fail=True))
    with pytest.raises(RuntimeError, match="spawn failed"):
        failing._publish_registrations()


def test_companion_base_env_forwards_login_identity_without_secrets(monkeypatch) -> None:
    """The login name reaches a companion (its CLIs key keychain/credential lookups on it); a
    secret-shaped sibling that merely starts the same way does not."""
    from ouroboros.extension_companion import _companion_base_env

    for key in ("USER", "LOGNAME", "USERNAME"):
        monkeypatch.setenv(key, "synthetic-login")
    monkeypatch.setenv("USER_API_TOKEN", "synthetic-host-only")
    env = _companion_base_env()
    assert env["USER"] == env["LOGNAME"] == env["USERNAME"] == "synthetic-login"
    assert "USER_API_TOKEN" not in env


# --- settings refresh of a declared companion (PluginAPI 2.4) -----------------


def _sleeper(tmp_path: pathlib.Path, env: dict) -> CompanionDescriptor:
    return CompanionDescriptor(skill_name="demo", name="daemon", cwd=tmp_path, env=dict(env),
                               command=[sys.executable, "-c", "import time; time.sleep(60)"])


@pytest.mark.parametrize("intervening", ["refresh", "stop", "stop_skill", "stop_all", "panic", "unchanged"])
def test_delayed_crash_restart_obeys_the_current_descriptor(tmp_path, monkeypatch, intervening):
    """An old monitor resumes only its still-current registration, including when both generations crash.

    Hold the real monitor at its backoff (before its final admission), not inside start's lock. Real
    children record their token and exit on a private marker; neither timing nor a fake PID decides it.
    The unchanged case proves ordinary on-failure restart still works.
    """
    import ouroboros.extension_companion as companion

    init_server_process_pid()
    entered = {key: threading.Event() for key in ("A", "B")}
    release = {key: threading.Event() for key in ("A", "B")}
    monitors, errors = [], []
    prefix = "descriptor-restart-"
    real_time = companion.time

    def backoff(seconds):
        thread = threading.current_thread()
        if seconds == 0.5 and thread.name.startswith(prefix):
            token = thread.name.removeprefix(prefix)
            entered[token].set()
            assert release[token].wait(30), "the test did not release the monitor"
        else:
            real_time.sleep(seconds)

    monkeypatch.setattr(companion, "time", types.SimpleNamespace(monotonic=real_time.monotonic, sleep=backoff))
    supervisor = CompanionSupervisor(tmp_path)

    def observe_monitor(key, runtime):
        def run():
            try:
                supervisor._monitor_runtime(key, runtime)
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=run, name=prefix + runtime.descriptor.env["TOKEN"], daemon=True)
        monitors.append(thread)
        thread.start()

    monkeypatch.setattr(supervisor, "_start_monitor", observe_monitor)
    child = tmp_path / "companion.py"
    child.write_text(
        "import os, sys, time\nfrom pathlib import Path\n"
        "root = Path(sys.argv[1]); token = os.environ['TOKEN']\n"
        "(root / ('started-' + str(os.getpid()))).write_text(token)\n"
        "while not (root / ('crash-' + token)).exists(): time.sleep(.01)\n"
        "raise SystemExit(7)\n", encoding="utf-8")
    descriptor = CompanionDescriptor(skill_name="demo", name="daemon", cwd=tmp_path, env={"TOKEN": "A"},
                                     command=[sys.executable, str(child), str(tmp_path)])
    try:
        assert supervisor.start(descriptor)
        (tmp_path / "crash-A").touch()
        assert entered["A"].wait(30), "A never reached the delayed restart"
        (tmp_path / "crash-A").unlink()
        assert not supervisor.snapshot()
        if intervening == "refresh":
            assert supervisor.refresh("demo", "daemon", lambda: {"TOKEN": "B"}, compare_keys=["TOKEN"],
                                      still_valid=lambda: True) == "started"
            (tmp_path / "crash-B").touch()
            assert entered["B"].wait(30), "B never reached the delayed restart"
            (tmp_path / "crash-B").unlink()
        elif intervening == "stop":
            supervisor.stop("demo", "daemon")
        elif intervening == "stop_skill":
            supervisor.stop_skill("demo")
        elif intervening == "stop_all":
            supervisor.stop_all()
        elif intervening == "panic":
            supervisor.panic_kill_all()
        release["A"].set()
        monitors[0].join(30)
        assert not monitors[0].is_alive() and not errors
        if intervening != "unchanged":
            assert not supervisor.snapshot(), "an obsolete monitor restarted A"
        if intervening == "refresh":
            release["B"].set()
            monitors[1].join(30)
            assert not monitors[1].is_alive() and not errors
        if intervening in ("refresh", "unchanged"):
            current = supervisor._runtimes["demo:daemon"]
            expected = "B" if intervening == "refresh" else "A"
            assert current.descriptor.env["TOKEN"] == expected and current.process.poll() is None
            marker = tmp_path / f"started-{current.process.pid}"
            deadline = real_time.monotonic() + 30
            while not marker.exists():
                assert real_time.monotonic() < deadline, "the restarted child did not announce itself"
                real_time.sleep(.01)
            assert marker.read_text() == expected
    finally:
        for event in release.values():
            event.set()
        supervisor.panic_kill_all()
        for thread in monitors:
            thread.join(30)
        assert all(not thread.is_alive() for thread in monitors), "a test monitor was left running"


def test_supervisor_refresh_compares_settings_keys_and_respawns_only_after_observed_death(tmp_path, monkeypatch):
    """Only the settings-derived keys decide: a changed bridge token, PATH or PYTHONPATH is no change. A
    change stops the old process, observes its death, builds the environment again (the save made while it
    stopped wins: A→B→C) and starts once. Neither the restart history nor a second refresh resets anything."""
    init_server_process_pid()
    supervisor = CompanionSupervisor(tmp_path)
    try:
        assert supervisor.start(_sleeper(tmp_path, {"TOKEN": "A", "HOST_SERVICE_TOKEN": "h1", "PATH": "/a"}))
        unrelated = CompanionDescriptor(skill_name="other", name="daemon", cwd=tmp_path, env={"TOKEN": "A"},
                                        command=[sys.executable, "-c", "import time; time.sleep(60)"])
        assert supervisor.start(unrelated)
        other = supervisor._runtimes["other:daemon"].process
        first = supervisor._runtimes["demo:daemon"].process
        same = {"TOKEN": "A", "HOST_SERVICE_TOKEN": "h2", "PATH": "/b", "PYTHONPATH": "/x"}
        assert supervisor.refresh("demo", "daemon", lambda: same, compare_keys=["TOKEN"],
                                  still_valid=lambda: True) == "unchanged"
        assert supervisor._runtimes["demo:daemon"].process is first and first.poll() is None

        reads = iter([{"TOKEN": "B"}, {"TOKEN": "C"}])
        dead_at_start = []
        real_start = supervisor.start
        monkeypatch.setattr(supervisor, "start", lambda d: dead_at_start.append(first.poll() is not None) or real_start(d))
        supervisor._restart_history["demo:daemon"] = [1.0]
        assert supervisor.refresh("demo", "daemon", lambda: next(reads), compare_keys=["TOKEN"],
                                  still_valid=lambda: True) == "restarted"
        current = supervisor._runtimes["demo:daemon"]
        assert dead_at_start == [True] and current.process is not first
        assert current.descriptor.env == {"TOKEN": "C"}  # the latest save, read after the stop
        assert supervisor._restart_history["demo:daemon"] == [1.0]
        assert supervisor.refresh("demo", "daemon", lambda: {"TOKEN": "C"}, compare_keys=["TOKEN"],
                                  still_valid=lambda: True) == "unchanged"
        # A cleared value is a change too; nothing is started when the caller's admission ended meanwhile.
        assert supervisor.refresh("demo", "daemon", lambda: {}, compare_keys=["TOKEN"],
                                  still_valid=lambda: False) == "refused"
        assert "demo:daemon" not in supervisor._runtimes
        assert supervisor._runtimes["other:daemon"].process is other and other.poll() is None  # untouched
    finally:
        supervisor.stop_all(timeout_sec=1)


def test_supervisor_refresh_settles_only_the_values_its_caller_read(tmp_path):
    """``expected`` binds the four settled answers to the caller's read. A process kept on A while the caller read
    B, and one started on C (saved during the stop) while it read B, both answer ``settings_changed``; the second
    still applied C. Blanks around a value aside, empty, blank and absent are one missing value; without
    ``expected`` the answer is as before, and a refused admission stays a refusal."""
    init_server_process_pid()
    supervisor = CompanionSupervisor(tmp_path)

    def refresh(*reads, expected, still_valid=lambda: True):
        values = iter(reads)
        return supervisor.refresh("demo", "daemon", lambda: dict(next(values)), compare_keys=["TOKEN"],
                                  still_valid=still_valid, expected=expected)

    try:
        assert supervisor.start(_sleeper(tmp_path, {"TOKEN": "A", "PATH": "/a"}))
        first = supervisor._runtimes["demo:daemon"].process
        assert refresh({"TOKEN": "A"}, expected={"TOKEN": "B"}) == "settings_changed"
        assert supervisor._runtimes["demo:daemon"].process is first and first.poll() is None
        assert refresh({"TOKEN": "A"}, expected={"TOKEN": " A "}) == "unchanged"
        assert refresh({"TOKEN": "B"}, {"TOKEN": "C"}, expected={"TOKEN": "B"}) == "settings_changed"
        assert supervisor._runtimes["demo:daemon"].descriptor.env == {"TOKEN": "C"} and first.poll() is not None
        assert refresh({"TOKEN": "C"}, expected={"TOKEN": "C"}) == "unchanged"
        assert refresh({}, {}, expected={"TOKEN": ""}) == "restarted"
        assert "TOKEN" not in supervisor._runtimes["demo:daemon"].descriptor.env
        assert refresh({}, expected={"TOKEN": None}) == "unchanged"
        assert refresh({"TOKEN": "   "}, {"TOKEN": "   "}, expected={"TOKEN": ""}) == "restarted"
        assert refresh({"TOKEN": "   "}, expected={"TOKEN": "C"}) == "settings_changed"
        assert refresh({"TOKEN": "   "}, expected=None) == "unchanged"
        assert refresh({"TOKEN": "D"}, expected={"TOKEN": "Z"}, still_valid=lambda: False) == "refused"
    finally:
        supervisor.stop_all(timeout_sec=1)


def test_supervisor_answers_from_memory_only_while_the_companion_carries_the_read(tmp_path):
    """``refresh_blocker`` with ``expected``: ``unchanged`` while the running companion carries it, nothing to say
    while it carries another value. A crash with the budget spent leaves the monitor's ``not_running``. A refresh
    that stopped the companion and could not read Settings again owes a start: no answer from memory, and its
    next refresh starts one on the value the stopped one carried. An explicit stop forgets it all."""
    from ouroboros.settings_integrity import SettingsDocumentUnknown

    init_server_process_pid()
    supervisor = CompanionSupervisor(tmp_path)

    def gone():
        deadline = time.monotonic() + 15
        while "demo:daemon" in supervisor._runtimes:
            assert time.monotonic() < deadline, "the companion was never retired"
            time.sleep(0.05)

    try:
        assert supervisor.start(_sleeper(tmp_path, {"TOKEN": "A"}))
        assert supervisor.refresh_blocker("demo", "daemon", {"TOKEN": " A "}) == "unchanged"
        assert supervisor.refresh_blocker("demo", "daemon", {"TOKEN": "B"}) == ""
        assert supervisor.refresh_blocker("demo", "daemon") == ""  # without expected the read decides
        crashed = supervisor._runtimes["demo:daemon"]
        supervisor._restart_history["demo:daemon"] = [time.monotonic()] * crashed.descriptor.max_restarts
        crashed.process.kill()
        gone()
        assert supervisor.refresh_blocker("demo", "daemon", {"TOKEN": "A"}) == "not_running"
        assert supervisor.refresh_blocker("demo", "daemon", {"TOKEN": "B"}) == ""
        assert supervisor.refresh("demo", "daemon", lambda: {"TOKEN": "A"}, compare_keys=["TOKEN"],
                                  still_valid=lambda: True, expected={"TOKEN": "A"}) == "not_running"
        assert "demo:daemon" not in supervisor._runtimes, "a refresh restarted what the crash budget stopped"

        assert supervisor.start(_sleeper(tmp_path, {"TOKEN": "A"}))
        reads = iter([{"TOKEN": "B"}])

        def read_then_unknown():
            for values in reads:
                return values
            raise SettingsDocumentUnknown("unreadable after the stop")

        with pytest.raises(SettingsDocumentUnknown):
            supervisor.refresh("demo", "daemon", read_then_unknown, compare_keys=["TOKEN"], still_valid=lambda: True)
        assert "demo:daemon" not in supervisor._runtimes
        assert supervisor.refresh_blocker("demo", "daemon", {"TOKEN": "A"}) == ""
        assert supervisor.refresh("demo", "daemon", lambda: {"TOKEN": "A"}, compare_keys=["TOKEN"],
                                  still_valid=lambda: True, expected={"TOKEN": "A"}) == "started"
        assert supervisor._runtimes["demo:daemon"].process.poll() is None
        assert supervisor.refresh_blocker("demo", "daemon", {"TOKEN": "A"}) == "unchanged"
        supervisor.stop("demo", "daemon")
        assert supervisor.refresh_blocker("demo", "daemon", {"TOKEN": "A"}) == "not_started"
    finally:
        supervisor.stop_all(timeout_sec=1)


def test_supervisor_refresh_never_replaces_an_unconfirmed_stop_and_lifecycle_controls_win(tmp_path, monkeypatch):
    init_server_process_pid()
    supervisor = CompanionSupervisor(tmp_path)
    try:
        assert supervisor.start(_sleeper(tmp_path, {"TOKEN": "A"}))
        held = supervisor._runtimes["demo:daemon"].process
        monkeypatch.setattr(supervisor, "_terminate_runtime", lambda runtime, timeout_sec: None)  # it outlives the stop
        assert supervisor.refresh("demo", "daemon", lambda: {"TOKEN": "B"}, compare_keys=["TOKEN"],
                                  still_valid=lambda: True) == "stop_unconfirmed"
        assert supervisor._runtimes["demo:daemon"].process is held and supervisor.snapshot()["demo:daemon"]["retiring"]
        assert supervisor.refresh_blocker("demo", "daemon") == "retained"
        assert supervisor.refresh("demo", "daemon", lambda: {"TOKEN": "B"}, compare_keys=["TOKEN"],
                                  still_valid=lambda: True) == "retained"
        monkeypatch.undo()
        supervisor.stop("demo", "daemon", timeout_sec=1)  # an explicit stop forgets the descriptor: no revival
        assert supervisor.refresh_blocker("demo", "daemon") == "not_started"
        assert supervisor.refresh("demo", "daemon", lambda: {"TOKEN": "C"}, compare_keys=["TOKEN"],
                                  still_valid=lambda: True) == "not_started"
        assert supervisor.start(_sleeper(tmp_path, {"TOKEN": "A"}))
        supervisor.panic_kill_all()
        assert supervisor.refresh("demo", "daemon", lambda: {"TOKEN": "C"}, compare_keys=["TOKEN"],
                                  still_valid=lambda: True) == "panic"
        assert supervisor.snapshot() == {}
    finally:
        supervisor.stop_all(timeout_sec=1)


def test_supervisor_refresh_admits_and_spawns_inside_its_admission_and_stops_outside_it(tmp_path, monkeypatch):
    """The caller's admission (its lifecycle lock) covers the final check, the environment and the start as one
    step, never the comparison or the stop: a refused admission starts nothing."""
    init_server_process_pid()
    supervisor = CompanionSupervisor(tmp_path)
    events: list[str] = []

    class Admission:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_exc):
            events.append("exit")

    try:
        assert supervisor.start(_sleeper(tmp_path, {"TOKEN": "A"}))
        terminate, start = supervisor._terminate_runtime, supervisor.start
        monkeypatch.setattr(supervisor, "_terminate_runtime",
                            lambda runtime, timeout_sec: events.append("stop") or terminate(runtime, timeout_sec=timeout_sec))
        monkeypatch.setattr(supervisor, "start", lambda descriptor: events.append("start") or start(descriptor))

        def env(token):
            return lambda: events.append("env") or {"TOKEN": token}

        assert supervisor.refresh("demo", "daemon", env("B"), compare_keys=["TOKEN"], admission=Admission(),
                                  still_valid=lambda: events.append("admit") or True) == "restarted"
        assert events == ["env", "stop", "enter", "admit", "env", "start", "exit"]
        events.clear()
        assert supervisor.refresh("demo", "daemon", env("C"), compare_keys=["TOKEN"], admission=Admission(),
                                  still_valid=lambda: events.append("admit") or False) == "refused"
        assert events == ["env", "stop", "enter", "admit", "exit"] and "demo:daemon" not in supervisor._runtimes
    finally:
        supervisor.stop_all(timeout_sec=1)


_PLUGIN_API_24 = "plugin_api: \"2.4\"\n"
_REFRESH_FRONTMATTER = (
    "env_from_settings: [TELEGRAM_BOT_TOKEN]\n"
    "companion_processes:\n"
    "  - name: daemon\n"
    "    runtime: python3\n"
    "    command: [\"python3\", \"scripts/daemon.py\"]\n"
)


def _refreshable_skill(tmp_path, monkeypatch, token: str, plugin_api: str = _PLUGIN_API_24, payload=None):
    """A granted extension (declaring ``plugin_api``, 2.4 by default) whose companion runs in a real supervisor,
    settings in a real document; ``payload`` adds or replaces skill files (relative path -> text)."""
    import ouroboros.config as config_mod
    from ouroboros.skill_loader import save_skill_grants

    init_server_process_pid()
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(config_mod, "SETTINGS_PATH", settings_path)
    settings_path.write_text(json.dumps({"TELEGRAM_BOT_TOKEN": token}), encoding="utf-8")
    repo_root, drive_root = tmp_path / "skills", tmp_path / "drive"
    skill_dir = repo_root / "tgdemo"
    (skill_dir / "scripts").mkdir(parents=True)
    drive_root.mkdir()
    (skill_dir / "scripts" / "daemon.py").write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    for relative, text in (payload or {}).items():
        (skill_dir / relative).write_text(text, encoding="utf-8")
    (skill_dir / "plugin.py").write_text("def register(api):\n    api.register_companion_process('daemon')\n",
                                         encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(
        "---\nname: tgdemo\ndescription: d\nversion: 0.1.0\ntype: extension\nentry: plugin.py\n"
        "permissions: [companion_process, read_settings]\n" + plugin_api + _REFRESH_FRONTMATTER + "---\nbody\n",
        encoding="utf-8")
    loaded = find_skill(drive_root, "tgdemo", repo_path=str(repo_root))
    save_enabled(drive_root, loaded.name, True)
    save_review_state(drive_root, loaded.name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    save_skill_grants(drive_root, loaded.name, ["TELEGRAM_BOT_TOKEN"], content_hash=loaded.content_hash,
                      requested_keys=["TELEGRAM_BOT_TOKEN"])
    loaded = find_skill(drive_root, "tgdemo", repo_path=str(repo_root))
    supervisor = CompanionSupervisor(tmp_path)
    monkeypatch.setattr(extension_plugin_api, "get_global_supervisor", lambda: supervisor)
    monkeypatch.setattr(extension_loader, "get_global_supervisor", lambda: supervisor)
    monkeypatch.setattr(extension_loader, "get_skills_repo_path", lambda: str(repo_root), raising=False)
    import ouroboros.extension_liveness as liveness
    real_state = liveness.runtime_state_for_skill_name
    monkeypatch.setattr(liveness, "runtime_state_for_skill_name",
                        lambda name, root, **kw: real_state(name, root, **{**kw, "repo_path": str(repo_root)}))
    assert extension_loader.load_extension(loaded, config_mod.load_settings, drive_root=drive_root,
                                           skills=[loaded], repo_path=str(repo_root)) is None
    api = extension_plugin_api._extensions["tgdemo"].api_instances[0]
    return api, supervisor, settings_path, drive_root, loaded


_MAX_PATH = 260  # Win32: UTF-16 code units, the terminating NUL included
_DEEP_DAEMON = """\
import json, os, sys, time
sys.dont_write_bytecode = True  # no __pycache__ path to fit, nor one under an inherited PYTHONPYCACHEPREFIX
import near
with open("ready.txt", encoding="utf-8") as resource:
    print("READY " + json.dumps({"cwd": os.getcwd(), "near": [near.VALUE, near.__file__],
                                 "resource": resource.read()}), flush=True)
print("\\n" * 4096, end="", flush=True)  # the supervisor's drainer hands over whole 4096-byte reads (or EOF)
time.sleep(60)
"""


def _utf16(path) -> int:
    return len(str(path).encode("utf-16-le")) // 2


def test_companion_starts_from_a_deep_staged_import_directory(tmp_path_factory, monkeypatch):
    """Where the two-digest leaf would put the working directory past MAX_PATH while every path the launch reads
    fits, the real companion imports its neighbour and reads a cwd-relative file, then says READY through that
    runtime's own pipe without a restart (the printer's pid is not compared: a Windows venv launcher runs the script
    in a child); unload stops it and releases the tree."""
    import os
    import uuid

    from ouroboros.skill_loader import skill_state_dir_path

    leaf = "sha256-" + "0" * 64  # one full SHA-256, whatever its digits
    earlier = len(f"{leaf}-{'0' * 64}") - len(leaf)  # what the two-digest leaf added

    def staged_cwd(base, name=leaf):
        return skill_state_dir_path(base / "drive", "tgdemo") / "__extension_imports" / name / "skill"

    reads = (("scripts", "daemon.py"), ("scripts", "near.py"), ("ready.txt",))
    longest = max(_utf16(pathlib.Path("x", *parts)) - 1 for parts in reads)  # with its separator
    root = tmp_path_factory.mktemp("deep").resolve()
    below = _utf16(staged_cwd(root / "x")) - _utf16(root / "x")
    low, high = _MAX_PATH + 1 - earlier, _MAX_PATH - 1 - longest  # the earlier leaf overflows / every read fits
    depth = max((low + high) // 2, _utf16(root) + 2 + below)  # the middle, or as shallow as this root allows
    assert depth <= high, f"no depth below {root} where the earlier leaf overflows and every read fits"
    base = root / ("d" * (depth - below - _utf16(root) - 1))
    base.mkdir()
    nonce = uuid.uuid4().hex
    supervisor = None
    try:
        _, supervisor, _, _, loaded = _refreshable_skill(base, monkeypatch, "synthetic", payload={
            "scripts/daemon.py": _DEEP_DAEMON, "scripts/near.py": f"VALUE = 'near:{nonce}'\n",
            "ready.txt": f"resource:{nonce}"})
        runtime = supervisor._runtimes["tgdemo:daemon"]
        pid, cwd = runtime.process.pid, runtime.descriptor.cwd
        assert cwd == staged_cwd(base, extension_loader.extension_import_identity(loaded)) and _utf16(cwd) == depth
        assert max(_utf16(cwd.joinpath(*parts)) for parts in reads) < _MAX_PATH < _utf16(cwd) + earlier
        deadline = time.monotonic() + 60
        while b"\n" not in runtime.stdout and runtime.process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        out = bytes(runtime.stdout)  # the supervisor's drainer owns the pipe
        line = out.split(b"\n", 1)[0]
        assert b"\n" in out and line.startswith(b"READY "), (out[:512], bytes(runtime.stderr))
        ready = json.loads(line[len(b"READY "):])
        assert (ready["near"][0], ready["resource"]) == (f"near:{nonce}", f"resource:{nonce}")
        assert os.path.samefile(ready["cwd"], cwd) and _utf16(ready["cwd"]) == depth
        assert os.path.samefile(ready["near"][1], cwd / "scripts" / "near.py")
        assert supervisor._runtimes["tgdemo:daemon"] is runtime and supervisor.snapshot()["tgdemo:daemon"]["pid"] == pid
        assert runtime.process.poll() is None and not supervisor._restart_history
        extension_loader.unload_extension("tgdemo")
        assert runtime.process.poll() is not None and supervisor.snapshot() == {}
        assert not (cwd.parent / "owners" / str(os.getpid())).exists()
    finally:
        extension_loader.unload_extension("tgdemo")
        if supervisor is not None:
            supervisor.stop_all(timeout_sec=1)


def test_plugin_api_refresh_applies_a_saved_token_through_the_same_api_and_controls_win(tmp_path, monkeypatch):
    """Save, replace and clear reach the companion through ``refresh_companion_process``; a second refresh
    through the same API still works; an unreadable document keeps the process; disable and unload win."""
    api, supervisor, settings_path, drive_root, loaded = _refreshable_skill(tmp_path, monkeypatch, "1:A")
    try:
        runtime = supervisor._runtimes["tgdemo:daemon"]
        assert runtime.descriptor.env["TELEGRAM_BOT_TOKEN"] == "1:A"
        assert api.get_settings_state(["TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY"]) == {
            "status": "ok", "values": {"TELEGRAM_BOT_TOKEN": "1:A"}, "missing": []}
        assert api.refresh_companion_process("daemon") == "unchanged"
        settings_path.write_text(json.dumps({"TELEGRAM_BOT_TOKEN": "1:B"}), encoding="utf-8")
        assert api.refresh_companion_process("daemon") == "restarted"
        replaced = supervisor._runtimes["tgdemo:daemon"]
        assert runtime.process.poll() is not None and replaced.descriptor.env["TELEGRAM_BOT_TOKEN"] == "1:B"
        assert api.refresh_companion_process("daemon") == "unchanged"  # the same API stays valid
        settings_path.write_text(json.dumps({}), encoding="utf-8")
        assert api.get_settings_state(["TELEGRAM_BOT_TOKEN"]) == {
            "status": "ok", "values": {}, "missing": ["TELEGRAM_BOT_TOKEN"]}
        assert api.refresh_companion_process("daemon") == "restarted"
        assert "TELEGRAM_BOT_TOKEN" not in supervisor._runtimes["tgdemo:daemon"].descriptor.env
        cleared = supervisor._runtimes["tgdemo:daemon"].process
        settings_path.write_text("{\"TELEGRAM_BOT_TOKEN\": ", encoding="utf-8")
        assert api.get_settings_state(["TELEGRAM_BOT_TOKEN"]) == {"status": "unknown", "values": {}, "missing": []}
        assert api.refresh_companion_process("daemon") == "settings_unknown"
        assert supervisor._runtimes["tgdemo:daemon"].process is cleared and cleared.poll() is None
        settings_path.write_text(json.dumps({"TELEGRAM_BOT_TOKEN": "1:C"}), encoding="utf-8")
        save_enabled(drive_root, loaded.name, False)  # disabled before its reconcile ran: still no restart
        assert api.refresh_companion_process("daemon") == "disabled"
        assert supervisor._runtimes["tgdemo:daemon"].process is cleared
        save_enabled(drive_root, loaded.name, True)
        from ouroboros.skill_loader import GRANTS_FILENAME, skill_state_dir

        grants = skill_state_dir(drive_root, loaded.name) / GRANTS_FILENAME
        granted = grants.read_bytes()
        grants.unlink()  # the grant is gone before its reconcile ran
        assert api.refresh_companion_process("daemon") == "missing_grants"
        assert supervisor._runtimes["tgdemo:daemon"].process is cleared
        grants.write_bytes(granted)
        extension_loader.unload_extension("tgdemo")
        assert api.refresh_companion_process("daemon") == "access_closed"
        assert api.get_settings_state(["TELEGRAM_BOT_TOKEN"])["status"] == "access_closed"
        assert supervisor.snapshot() == {}
    finally:
        extension_loader.unload_extension("tgdemo")
        supervisor.stop_all(timeout_sec=1)
        init_server_process_pid()


def test_plugin_api_refresh_settles_only_the_token_its_caller_read(tmp_path, monkeypatch):
    """Through the real API: a caller that read A while B is saved gets ``settings_changed`` (B applied), then
    ``unchanged`` for B; a clear read as missing settles. ``only_if_differs`` answers from memory, reading no
    Settings and minting no token, while the companion carries the read, and refreshes when it does not.
    ``expected`` names only ``env_from_settings`` keys, and a refusal names the key, never a value. Controls answer
    as before: a companion that would need a start meets the admission first."""
    api, supervisor, settings_path, drive_root, loaded = _refreshable_skill(tmp_path, monkeypatch, "1:A")
    key = "TELEGRAM_BOT_TOKEN"
    try:
        assert api.refresh_companion_process("daemon", expected={key: "1:A"}) == "unchanged"
        settings_path.write_text(json.dumps({key: "1:B"}), encoding="utf-8")
        assert api.refresh_companion_process("daemon", expected={key: "1:A"}) == "settings_changed"
        assert supervisor._runtimes["tgdemo:daemon"].descriptor.env[key] == "1:B"
        assert api.refresh_companion_process("daemon", expected={key: "1:B"}) == "unchanged"
        settings_path.write_text(json.dumps({}), encoding="utf-8")
        assert api.refresh_companion_process("daemon", expected={key: ""}) == "restarted"
        reads = []
        for name in ("skill_settings", "mint_skill_token"):
            real = getattr(extension_plugin_api, name)
            monkeypatch.setattr(extension_plugin_api, name,
                                lambda *a, _real=real, _name=name, **k: reads.append(_name) or _real(*a, **k))
        assert api.refresh_companion_process("daemon", expected={key: ""}, only_if_differs=True) == "unchanged"
        assert reads == [], "a confirmation read Settings or minted a token"
        settings_path.write_text(json.dumps({key: "1:C"}), encoding="utf-8")
        assert api.refresh_companion_process("daemon", expected={key: "1:C"}, only_if_differs=True) == "restarted"
        assert "skill_settings" in reads and supervisor._runtimes["tgdemo:daemon"].descriptor.env[key] == "1:C"
        with pytest.raises(ExtensionRegistrationError, match="naming every env_from_settings key"):
            api.refresh_companion_process("daemon", only_if_differs=True)
        for stray in ({"OPENAI_API_KEY": "sk-not-for-you"}, {key.lower(): "1:C"}):
            with pytest.raises(ExtensionRegistrationError, match="not env_from_settings") as refused:
                api.refresh_companion_process("daemon", expected=stray)
            assert not any(value in str(refused.value) for value in stray.values())
        with pytest.raises(ExtensionRegistrationError, match="maps settings keys"):
            api.refresh_companion_process("daemon", expected=[(key, "1:C")])
        save_enabled(drive_root, loaded.name, False)
        assert api.refresh_companion_process("daemon", expected={key: "1:C"}) == "disabled"
        assert api.refresh_companion_process("daemon", expected={key: "1:C"}, only_if_differs=True) == "unchanged"
        settings_path.write_text(json.dumps({key: "1:D"}), encoding="utf-8")
        assert api.refresh_companion_process("daemon", expected={key: "1:D"}, only_if_differs=True) == "disabled"
        assert supervisor._runtimes["tgdemo:daemon"].descriptor.env[key] == "1:C"
    finally:
        extension_loader.unload_extension("tgdemo")
        supervisor.stop_all(timeout_sec=1)
        init_server_process_pid()


@pytest.mark.parametrize("failure", ["unreadable", "refused_spawn"])
def test_a_companion_a_refresh_stopped_and_could_not_replace_is_started_by_the_next_one(tmp_path, monkeypatch,
                                                                                          failure):
    """A refresh stops the companion on A to apply B; after that stop Settings cannot be read, or the system refuses
    the spawn, so nothing runs. A is saved back: the next refresh starts a companion on A, although A is what the
    stopped one carried. An absence a crash left (here exhausted restarts) stays the monitor's: ``not_running``."""
    api, supervisor, settings_path, _drive_root, _loaded = _refreshable_skill(tmp_path, monkeypatch, "1:A")
    key, terminate, start = "TELEGRAM_BOT_TOKEN", supervisor._terminate_runtime, supervisor.start
    first = supervisor._runtimes["tgdemo:daemon"].process
    try:
        settings_path.write_text(json.dumps({key: "1:B"}), encoding="utf-8")
        if failure == "unreadable":
            def unreadable_after_stop(*args, **kwargs):
                terminate(*args, **kwargs)
                settings_path.write_text("{", encoding="utf-8")

            monkeypatch.setattr(supervisor, "_terminate_runtime", unreadable_after_stop)
            assert api.refresh_companion_process("daemon", expected={key: "1:B"}) == "settings_unknown"
        else:
            def refused(_descriptor):
                raise OSError(12, "Cannot allocate memory")

            monkeypatch.setattr(supervisor, "start", refused)
            with pytest.raises(OSError):
                api.refresh_companion_process("daemon", expected={key: "1:B"})
        assert first.poll() is not None and "tgdemo:daemon" not in supervisor._runtimes
        monkeypatch.setattr(supervisor, "_terminate_runtime", terminate)
        monkeypatch.setattr(supervisor, "start", start)
        settings_path.write_text(json.dumps({key: "1:A"}), encoding="utf-8")
        assert api.refresh_companion_process("daemon", expected={key: "1:A"}) == "started"
        current = supervisor._runtimes["tgdemo:daemon"]
        assert current.process.poll() is None and current.descriptor.env[key] == "1:A"
        assert api.refresh_companion_process("daemon", expected={key: "1:A"}) == "unchanged"
        with supervisor._lock:
            supervisor._restart_history["tgdemo:daemon"] = [time.monotonic()] * current.descriptor.max_restarts
        current.process.kill()
        deadline = time.monotonic() + 15
        while "tgdemo:daemon" in supervisor._runtimes:
            assert time.monotonic() < deadline, "the exhausted companion was never retired"
            time.sleep(0.05)
        assert api.refresh_companion_process("daemon", expected={key: "1:A"}) == "not_running"
        assert "tgdemo:daemon" not in supervisor._runtimes, "a refresh restarted what the crash budget stopped"
    finally:
        extension_loader.unload_extension("tgdemo")
        supervisor.stop_all(timeout_sec=1)
        init_server_process_pid()


def test_plugin_api_refresh_is_off_loop_and_refused_to_a_reloaded_generation(tmp_path, monkeypatch):
    import asyncio

    api, supervisor, settings_path, drive_root, loaded = _refreshable_skill(tmp_path, monkeypatch, "1:A")
    try:
        async def on_loop():
            return api.refresh_companion_process("daemon")

        with pytest.raises(ExtensionRegistrationError, match="off the event loop"):
            asyncio.run(on_loop())
        with pytest.raises(ExtensionRegistrationError, match="not declared"):
            api.refresh_companion_process("other")
        extension_loader.unload_extension("tgdemo")
        import ouroboros.config as config_mod

        assert extension_loader.load_extension(loaded, config_mod.load_settings, drive_root=drive_root,
                                               skills=[loaded]) is None
        settings_path.write_text(json.dumps({"TELEGRAM_BOT_TOKEN": "1:B"}), encoding="utf-8")
        assert api.refresh_companion_process("daemon") == "access_closed"  # the old generation's API
        fresh = extension_plugin_api._extensions["tgdemo"].api_instances[0]
        assert fresh.refresh_companion_process("daemon") == "restarted"
    finally:
        extension_loader.unload_extension("tgdemo")
        supervisor.stop_all(timeout_sec=1)
        init_server_process_pid()


def _owner_off(drive_root, loaded):
    """The owner's real Off: the lifecycle action Skills' toggle runs."""
    from ouroboros.skill_lifecycle_actions import run_skill_action

    return run_skill_action(types.SimpleNamespace(drive_root=drive_root), loaded.name, "disable",
                            repo_path=str(loaded.skill_dir.parent), _owner_actor="owner_ui")


def _held(monkeypatch, supervisor, name: str, hold):
    """Replace one supervisor step with ``hold`` running before it; returns (entered, release)."""
    entered, release, real = threading.Event(), threading.Event(), getattr(supervisor, name)

    def step(*args, **kwargs):
        if hold(*args):
            entered.set()
            assert release.wait(15)
        return real(*args, **kwargs)

    monkeypatch.setattr(supervisor, name, step)
    return entered, release


@pytest.mark.parametrize("change", ["none", "off", "revoke"])
def test_an_off_or_revocation_saved_during_the_refresh_stop_wins_at_its_spawn(tmp_path, monkeypatch, change):
    """While the old companion stops, no lock is held: the owner's real Off persists at once. The final admission
    holds the lifecycle lock through the spawn and reads that Off, or the grant removed meanwhile, so nothing
    carrying the new token starts. With nothing saved, the same held stop restarts on the new token."""
    from ouroboros.skill_loader import GRANTS_FILENAME, load_enabled, skill_state_dir

    api, supervisor, settings_path, drive_root, loaded = _refreshable_skill(tmp_path, monkeypatch, "1:A")
    first, outcome, starts = supervisor._runtimes["tgdemo:daemon"], {}, []
    stopping, release = _held(monkeypatch, supervisor, "_terminate_runtime", lambda runtime: runtime is first)
    start = supervisor.start
    monkeypatch.setattr(supervisor, "start", lambda descriptor: starts.append(
        (load_enabled(drive_root, loaded.name), descriptor.env.get("TELEGRAM_BOT_TOKEN"))) or start(descriptor))
    grants = skill_state_dir(drive_root, loaded.name) / GRANTS_FILENAME
    granted = grants.read_bytes()
    settings_path.write_text(json.dumps({"TELEGRAM_BOT_TOKEN": "1:B"}), encoding="utf-8")
    refresh = threading.Thread(target=lambda: outcome.update(refresh=api.refresh_companion_process("daemon")))
    owner = threading.Thread(target=lambda: outcome.update(off=_owner_off(drive_root, loaded)))
    try:
        refresh.start()
        assert stopping.wait(15)
        if change == "off":
            owner.start()
            deadline = time.monotonic() + 15
            while load_enabled(drive_root, loaded.name) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not load_enabled(drive_root, loaded.name), "the owner's Off waited for the refresh's stop"
        elif change == "revoke":
            grants.unlink()
        release.set()
        refresh.join(15)
        if owner.ident:
            owner.join(15)
        assert not refresh.is_alive() and not owner.is_alive()
        if change == "none":
            assert outcome["refresh"] == "restarted" and starts == [(True, "1:B")]
            return
        assert starts == [], "a process with the new token started after the owner's choice was saved"
        assert first.process.poll() is not None and "tgdemo:daemon" not in supervisor._runtimes
        if change == "off":
            # Read at admission (disabled), or the unload that followed the Off closed this API first.
            assert outcome["refresh"] in {"disabled", "access_closed"}
            assert outcome["off"]["ok"] and outcome["off"]["extension_action"] == "extension_unloaded"
        else:
            assert outcome["refresh"] == "missing_grants"
            grants.write_bytes(granted)  # the refused admission wedged nothing: the next refresh starts it
            assert api.refresh_companion_process("daemon") == "started" and starts == [(True, "1:B")]
    finally:
        release.set()
        refresh.join(15)
        if owner.ident:
            owner.join(15)
        extension_loader.unload_extension("tgdemo")
        supervisor.stop_all(timeout_sec=1)
        init_server_process_pid()


def test_an_off_saved_while_a_refresh_spawns_waits_for_that_spawn_and_then_stops_it(tmp_path, monkeypatch):
    """The other order: the Off is persisted under the same lifecycle lock, so it never lands between the final
    admission and the spawn; it waits for the spawn, and its own reconcile then stops that new process."""
    from ouroboros.extension_registry_state import _lifecycle_lock_for
    from ouroboros.skill_loader import load_enabled

    api, supervisor, settings_path, drive_root, loaded = _refreshable_skill(tmp_path, monkeypatch, "1:A")
    outcome, starts = {}, []
    spawning, release = _held(monkeypatch, supervisor, "start",
                              lambda _descriptor: starts.append(load_enabled(drive_root, loaded.name)) or True)
    settings_path.write_text(json.dumps({"TELEGRAM_BOT_TOKEN": "1:B"}), encoding="utf-8")
    refresh = threading.Thread(target=lambda: outcome.update(refresh=api.refresh_companion_process("daemon")))
    owner = threading.Thread(target=lambda: outcome.update(off=_owner_off(drive_root, loaded)))
    try:
        refresh.start()
        assert spawning.wait(15)
        lock = _lifecycle_lock_for("tgdemo")
        held_elsewhere = not lock.acquire(blocking=False)
        if not held_elsewhere:
            lock.release()
        assert held_elsewhere, "the final admission released the lifecycle lock before its spawn"
        owner.start()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            assert load_enabled(drive_root, loaded.name), "the Off landed between the final admission and the spawn"
            time.sleep(0.01)
        release.set()
        refresh.join(15)
        owner.join(15)
        assert not refresh.is_alive() and not owner.is_alive()
        assert outcome["refresh"] == "restarted" and starts == [True]
        assert not load_enabled(drive_root, loaded.name) and outcome["off"]["extension_action"] == "extension_unloaded"
        assert supervisor.snapshot() == {}  # the process admitted before the Off is stopped by the Off's reconcile
    finally:
        release.set()
        refresh.join(15)
        if owner.ident:
            owner.join(15)
        extension_loader.unload_extension("tgdemo")
        supervisor.stop_all(timeout_sec=1)
        init_server_process_pid()


def test_panic_during_a_refresh_admission_is_not_delayed_and_wins_its_spawn(tmp_path, monkeypatch):
    api, supervisor, settings_path, _drive_root, _loaded = _refreshable_skill(tmp_path, monkeypatch, "1:A")
    outcome = {}
    spawning, release = _held(monkeypatch, supervisor, "start", lambda _descriptor: True)
    settings_path.write_text(json.dumps({"TELEGRAM_BOT_TOKEN": "1:B"}), encoding="utf-8")
    refresh = threading.Thread(target=lambda: outcome.update(refresh=api.refresh_companion_process("daemon")))
    try:
        refresh.start()
        assert spawning.wait(15)
        panic = threading.Thread(target=supervisor.panic_kill_all)
        panic.start()
        panic.join(5)
        assert not panic.is_alive(), "Panic waited for a held refresh admission"
        release.set()
        refresh.join(15)
        assert outcome["refresh"] == "panic" and supervisor.snapshot() == {}
    finally:
        release.set()
        refresh.join(15)
        extension_loader.unload_extension("tgdemo")
        supervisor.stop_all(timeout_sec=1)
        init_server_process_pid()


def test_every_other_reason_not_to_run_reads_not_desired(tmp_path, monkeypatch):
    """The closed outcome set: a reason beside ``disabled`` and ``missing_grants`` (here an edited payload whose
    review is stale before any reload) answers ``not_desired`` and starts nothing."""
    api, supervisor, settings_path, _drive_root, loaded = _refreshable_skill(tmp_path, monkeypatch, "1:A")
    try:
        running = supervisor._runtimes["tgdemo:daemon"].process
        (loaded.skill_dir / "scripts" / "edited.py").write_text("# an owner edit\n", encoding="utf-8")
        settings_path.write_text(json.dumps({"TELEGRAM_BOT_TOKEN": "1:B"}), encoding="utf-8")
        assert api.refresh_companion_process("daemon") == "not_desired"
        assert supervisor._runtimes["tgdemo:daemon"].process is running and running.poll() is None
    finally:
        extension_loader.unload_extension("tgdemo")
        supervisor.stop_all(timeout_sec=1)
        init_server_process_pid()


@pytest.mark.parametrize("declared", ['plugin_api: "2.0"\n', 'plugin_api: "2.3"\n', ""])
def test_plugin_api_2_4_methods_follow_the_loaders_real_negotiation(tmp_path, monkeypatch, declared):
    """Through the real loader: every declared 2.x minimum negotiates the host's 2.4 and is served its settings
    state and companion refresh; a payload without the field binds legacy 1.3 and is refused both."""
    api, supervisor, settings_path, _drive_root, _loaded = _refreshable_skill(tmp_path, monkeypatch, "1:A", declared)
    try:
        if declared:
            assert api.get_settings_state(["TELEGRAM_BOT_TOKEN"]) == {
                "status": "ok", "values": {"TELEGRAM_BOT_TOKEN": "1:A"}, "missing": []}
            assert api.get_settings_state([]) == {"status": "ok", "values": {}, "missing": []}
            settings_path.write_text(json.dumps({"TELEGRAM_BOT_TOKEN": "1:B"}), encoding="utf-8")
            assert api.refresh_companion_process("daemon") == "restarted"
        else:
            for call in (lambda: api.get_settings_state(["TELEGRAM_BOT_TOKEN"]),
                         lambda: api.refresh_companion_process("daemon")):
                with pytest.raises(ExtensionRegistrationError, match='needs plugin_api "2.4".*negotiated 1.3'):
                    call()
    finally:
        extension_loader.unload_extension("tgdemo")
        supervisor.stop_all(timeout_sec=1)
        init_server_process_pid()


@contextlib.contextmanager
def _native_telegram(tmp_path, monkeypatch, *, token: str = "", miniapp: str = "off", hold=None, payload_edit=None,
                     core_port: int = 0, refuse=None):
    """The shipped Telegram payload through the real loader, supervisor and extension loop, beside a loopback Bot
    API stand-in that records every request; ``hold(token, method)`` may block an answer and ``refuse(token,
    method)`` answers it 401. No tunnel, Telegram or cloud request is made. Yields the world the native witnesses
    drive."""
    import asyncio
    import http.server
    import time
    import uuid

    import supervisor.message_bus as mb
    import supervisor.queue as queue_mod
    import ouroboros.config as config_mod
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros import owner_quiz
    from ouroboros import secret_requests as sr
    from ouroboros.event_bus import EventBus
    from ouroboros.gateway import secret_decision as sd
    from ouroboros.gateway import settings as settings_mod
    from ouroboros.gateway import task_decision as td
    from ouroboros.skill_loader import (requested_core_setting_keys, requested_skill_permissions,
                                        save_skill_grants)
    from ouroboros.task_results import write_task_result

    asked: list[tuple[str, str]] = []
    menus: dict[str, dict] = {}  # each bot's own menu button for the owner chat
    menu_sets: list[tuple[str, dict]] = []

    class BotApi(http.server.BaseHTTPRequestHandler):
        timeout = 5  # an idle connection (a call cancelled after connecting) ends, so closing can join its thread

        def do_POST(self):  # noqa: N802 — the stdlib handler's name
            token_, method = self.path[len("/bot"):].split("/", 1)
            asked.append((token_, method))
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            form = json.loads(raw or b"{}") if "json" in (self.headers.get("Content-Type") or "") else {}
            if hold is not None:
                hold(token_, method)
            refused = refuse is not None and refuse(token_, method)
            bot = token_.split(":", 1)[0]
            if method == "setChatMenuButton" and not refused:
                menus[bot] = form.get("menu_button") or {"type": "default"}
                menu_sets.append((token_, menus[bot]))
            result = ([] if method == "getUpdates"
                      else {"id": int(bot) if bot.isdigit() else 1, "is_bot": True, "username": "stand_in_bot"}
                      if method == "getMe"
                      else {"id": 42, "type": "private"} if method == "getChat"
                      else menus.get(bot, {"type": "default"}) if method == "getChatMenuButton"
                      else {"message_id": len(asked)} if method == "sendMessage" else True)
            body = json.dumps({"ok": False, "error_code": 401, "description": "Unauthorized"} if refused
                              else {"ok": True, "result": result}).encode()
            self.send_response(401 if refused else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    class JoinedServer(http.server.ThreadingHTTPServer):
        daemon_threads = False  # server_close() joins every request thread it started (block_on_close)

    bot_server = JoinedServer(("127.0.0.1", 0), BotApi)
    serving = threading.Thread(target=bot_server.serve_forever, daemon=True)
    serving.start()
    init_server_process_pid()
    drive = tmp_path / "drive"
    drive.mkdir()
    monkeypatch.setattr(config_mod, "DATA_DIR", drive)
    monkeypatch.setattr(config_mod, "SETTINGS_PATH", drive / "settings.json")
    monkeypatch.setattr(config_mod, "get_skills_repo_path", lambda: "")
    if token:
        (drive / "settings.json").write_text(json.dumps({"TELEGRAM_BOT_TOKEN": token}), encoding="utf-8")
    payload = drive / "skills" / "external" / "telegram"
    shutil.copytree(pathlib.Path(__file__).parents[1] / "skills" / "telegram", payload)
    if payload_edit is not None:
        payload_edit(payload)
    state = drive / "state" / "skills" / "telegram"
    state.mkdir(parents=True)
    if core_port:  # the server port the Mini App's runtime configuration names (get_runtime_info)
        (drive / "state" / "server_port").write_text(str(core_port), encoding="utf-8")
    (state / "settings.json").write_text(json.dumps({
        "TELEGRAM_CHAT_ID": "42", "TELEGRAM_MINIAPP_ENABLED": miniapp,
        "TELEGRAM_API_BASE": f"http://127.0.0.1:{bot_server.server_address[1]}"}), encoding="utf-8")

    def admit():
        """Review, enable and grant the payload as it is now; returns the loaded skill."""
        skill = find_skill(drive, "telegram", repo_path="")
        save_enabled(drive, "telegram", True)
        save_review_state(drive, "telegram", SkillReviewState(status="pass", content_hash=skill.content_hash))
        permissions = requested_skill_permissions(list(skill.manifest.permissions or []),
                                                  list(skill.manifest.subscribe_events or []))
        keys = requested_core_setting_keys(list(skill.manifest.env_from_settings or []))
        save_skill_grants(drive, "telegram", keys, content_hash=skill.content_hash, requested_keys=keys,
                          granted_permissions=permissions, requested_permissions=permissions)
        return find_skill(drive, "telegram", repo_path="")

    skill = admit()
    supervisor = CompanionSupervisor(drive)
    monkeypatch.setattr(extension_plugin_api, "get_global_supervisor", lambda: supervisor)
    monkeypatch.setattr(extension_loader, "get_global_supervisor", lambda: supervisor)
    loop = asyncio.new_event_loop()
    looping = threading.Thread(target=loop.run_forever, daemon=True)
    looping.start()
    bus = EventBus()
    bus.set_loop(loop)
    monkeypatch.setattr(extension_plugin_api, "get_global_event_bus", lambda: bus)
    monkeypatch.setattr(extension_loader, "get_global_event_bus", lambda: bus)
    import ouroboros.extension_liveness as liveness
    real_state = liveness.runtime_state_for_skill_name
    monkeypatch.setattr(liveness, "runtime_state_for_skill_name",
                        lambda name, root, **kw: real_state(name, root, **{**kw, "repo_path": ""}))
    # The protected form's owner: a running setup task whose secret question the form answers.
    monkeypatch.setattr(mb, "get_bridge", lambda: SimpleNamespaceBridge())
    monkeypatch.setattr(mb, "DATA_DIR", drive)
    monkeypatch.setattr(mb, "publish_event", lambda *args: None)
    monkeypatch.setattr(td, "_live_root_task", lambda task_id, *, holds_mail=False: ({"id": task_id}, ""))
    monkeypatch.setattr(queue_mod, "_task_drive_for_task", lambda task, task_id: drive, raising=False)
    write_task_result(drive, "setup", "running", chat_id=1)
    form = Starlette(routes=[Route("/api/decisions/secret", endpoint=sd.api_secret_decision, methods=["POST"])])
    form.state.drive_root = drive
    monkeypatch.setattr(settings_mod, "_start_supervisor_if_needed_for_request", lambda *_a, **_k: False)
    monkeypatch.setattr(settings_mod, "_apply_settings_save_side_effects", lambda *_a, **_k: None)
    owner_settings = Starlette(routes=[Route("/api/settings", endpoint=settings_mod.api_settings_post,
                                             methods=["POST"])])
    owner_settings.state.drive_root = owner_settings.state.repo_dir = drive

    def form_save(value):
        request = sr.resolve_secret_request(drive, "telegram", ["botToken"])  # the skill's own secret_fields
        quiz_id = uuid.uuid4().hex
        owner_quiz.record_asked(drive, "setup", quiz_id=quiz_id, question=sr.question_text(request), options=[],
                                wait_for_answer=True, chat_id=1, secret_request=request)
        owner_quiz.record_publication(drive, "setup", quiz_id, 1)
        decision = f"quiz:setup:{quiz_id}"
        client = TestClient(form)
        status = client.post("/api/decisions/secret", json={"decision_id": decision, "action": "status"}).json()
        response = client.post("/api/decisions/secret", json={
            "request_id": uuid.uuid4().hex, "decision_id": decision, "values": {"botToken": value},
            "observed": {row["id"]: row["configured"] for row in status["keys"]}})
        assert response.status_code == 200, response.text

    def settings_save(value):
        response = TestClient(owner_settings).post("/api/settings", json={"TELEGRAM_BOT_TOKEN": value})
        assert response.status_code == 200, response.text

    def until(condition, what, timeout=30.0):
        deadline = time.monotonic() + timeout
        while not condition():
            assert time.monotonic() < deadline, what
            time.sleep(0.05)

    def published():
        try:
            return json.loads((state / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    try:
        yield types.SimpleNamespace(
            drive=drive, state=state, payload=payload, skill=skill, admit=admit, supervisor=supervisor, bus=bus,
            asked=asked, menu_sets=menu_sets, form_save=form_save, settings_save=settings_save, until=until, published=published,
            load_settings=config_mod.load_settings,
            runtime=lambda: supervisor._runtimes.get("telegram:miniapp_gateway"),
            api=lambda: extension_plugin_api._extensions["telegram"].api_instances[0])
    finally:
        extension_loader.unload_extension("telegram")
        supervisor.stop_all(timeout_sec=3)

        async def drain_own_tasks():
            # Unload only requests cancellation of the supervised runners: let them finish on this loop first.
            pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        asyncio.run_coroutine_threadsafe(drain_own_tasks(), loop).result(30)
        loop.call_soon_threadsafe(loop.stop)
        looping.join(10)
        loop.run_until_complete(loop.shutdown_default_executor())  # the observer's to_thread workers
        loop.close()
        bot_server.shutdown()
        bot_server.server_close()
        serving.join(10)
        init_server_process_pid()


def test_the_native_telegram_skill_applies_form_and_settings_saves_to_its_real_companion(tmp_path, monkeypatch):
    """The shipped Telegram payload through the real loader, supervisor and extension loop: its own observer
    applies a token the protected secret form saved, a replacement the Settings route saved, and a clear, to
    the real `companion.py` process (old death observed before each new one) and to its poller. The Mini App
    is Off and the Bot API is a loopback stand-in: no tunnel, Telegram or cloud request is made."""
    import time

    token_a, token_b = "111:SYNTHETIC-A", "222:SYNTHETIC-B"
    with _native_telegram(tmp_path, monkeypatch) as world:
        asked, runtime, published, until = world.asked, world.runtime, world.published, world.until
        assert extension_loader.load_extension(world.skill, world.load_settings, drive_root=world.drive,
                                               skills=[world.skill], repo_path="") is None
        until(lambda: runtime() is not None and published().get("pid") == runtime().process.pid
              and published().get("state") == "disabled", "the companion never published its Off wait")
        first = runtime().process
        assert "TELEGRAM_BOT_TOKEN" not in runtime().descriptor.env
        time.sleep(0.5)
        assert first.poll() is None and not asked, "no token: the companion waits and the poller asks nothing"

        world.form_save(token_a)
        until(lambda: runtime() is not None and runtime().process is not first
              and runtime().descriptor.env.get("TELEGRAM_BOT_TOKEN") == token_a, "the form's token never applied")
        assert first.poll() is not None, "the old companion outlived its replacement"
        second = runtime().process
        until(lambda: published().get("pid") == second.pid, "the new companion never published")
        until(lambda: (token_a, "getUpdates") in asked, "the poller never polled with the form's token")

        world.settings_save(token_b)
        until(lambda: runtime() is not None and runtime().process is not second
              and runtime().descriptor.env.get("TELEGRAM_BOT_TOKEN") == token_b, "the replacement never applied")
        assert second.poll() is not None
        until(lambda: (token_b, "getUpdates") in asked, "the poller never followed the replacement")
        switched = len(asked)
        time.sleep(1.0)
        assert all(token == token_b for token, _method in asked[switched:]), "the old token was asked again"

        third = runtime().process
        world.settings_save("")
        until(lambda: runtime() is not None and runtime().process is not third
              and "TELEGRAM_BOT_TOKEN" not in runtime().descriptor.env, "the clear never applied")
        assert third.poll() is not None
        cleared = len(asked)
        time.sleep(1.5)
        assert len(asked) <= cleared + 1, "the poller kept asking without a token"  # one in-flight poll may finish
        restarts = world.supervisor._restart_history.get("telegram:miniapp_gateway", [])
        assert restarts == [], "a settings refresh spent the crash budget"


@pytest.mark.parametrize("before,saved", [("", "111:SYNTHETIC-A"), ("111:SYNTHETIC-A", "")])
def test_a_token_saved_back_during_a_mini_app_refresh_leaves_no_companion_on_the_other_one(
    tmp_path, monkeypatch, before, saved,
):
    """A → B → A, and clear → B → clear, around one refresh of the shipped payload's real companion. The observer
    reads ``saved`` and asks for a refresh; the owner saves B before that refresh reads Settings, and ``saved`` again
    after its final read, while B's companion spawns. The observer never reads B. What the refresh applied is not
    what the observer asked for, so it does not settle: it asks once more and the real companion converges to the
    saved value, then nothing more is asked. The Mini App is Off and the Bot API a loopback stand-in."""
    other, key = "222:SYNTHETIC-B", "TELEGRAM_BOT_TOKEN"
    with _native_telegram(tmp_path, monkeypatch, token=before) as world:
        supervisor, runtime, until = world.supervisor, world.runtime, world.until
        armed, outcomes, spawned, real_refresh = {}, [], [], supervisor.refresh
        entered, release = threading.Event(), threading.Event()

        def refresh(*args, **kwargs):
            if armed.pop("refresh", False):
                entered.set()
                assert release.wait(15)
            outcomes.append(real_refresh(*args, **kwargs))
            return outcomes[-1]

        monkeypatch.setattr(supervisor, "refresh", refresh)
        spawning, spawn = _held(monkeypatch, supervisor, "start",
                                lambda descriptor: descriptor.env.get(key) == other and armed.pop("start", False))
        held_start = supervisor.start

        def start(descriptor):
            started = held_start(descriptor)
            spawned.append((descriptor.env.get(key, ""), runtime().process if started else None))
            return started

        monkeypatch.setattr(supervisor, "start", start)
        assert extension_loader.load_extension(world.skill, world.load_settings, drive_root=world.drive,
                                               skills=[world.skill], repo_path="") is None
        until(lambda: outcomes == ["unchanged"], "the observer's first refresh never concluded")
        armed["refresh"] = True
        world.settings_save(saved)
        assert entered.wait(15), "the observer never asked to apply the saved value"
        world.settings_save(other)  # before the refresh reads Settings
        armed["start"] = True
        release.set()
        assert spawning.wait(15), "the refresh never spawned the value saved meanwhile"
        world.settings_save(saved)  # after its final read, before the observer's next one
        spawn.set()
        seen = {}

        def converged():
            current = runtime()
            seen["token"] = None if current is None else current.descriptor.env.get(key, "")
            return seen["token"] == saved and current.process.poll() is None and outcomes[-1:] == ["restarted"]

        deadline = time.monotonic() + 20
        while not converged() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert converged(), f"the Mini App stays on {seen['token']!r}, Settings hold {saved!r}; refreshes {outcomes}"
        time.sleep(4.5)  # two more reads of the unchanged value
        assert outcomes == ["unchanged", "settings_changed", "restarted"], "a refresh was asked again when settled"
        assert [token for token, _process in spawned] == [before, other, saved]
        assert all(process.poll() is not None for _token, process in spawned[:-1]) and spawned[-1][1].poll() is None
        assert runtime().process is spawned[-1][1] and world.load_settings().get(key, "") == saved
        assert supervisor._restart_history.get("telegram:miniapp_gateway", []) == []


@pytest.mark.parametrize("saved", ["111:SYNTHETIC-A", ""])
def test_a_companion_recovered_on_other_settings_returns_to_the_token_the_bridge_holds(tmp_path, monkeypatch, saved):
    """The real companion's restarts are exhausted while its settled observer waits on its next read. Generic
    recovery starts it again from Settings that hold B at that moment; ``saved`` is back before the observer reads.
    The observer reads no change, yet the companion returns to ``saved`` within a few reads, and is then left
    alone. The Mini App is Off and the Bot API a loopback stand-in."""
    other, key = "222:SYNTHETIC-B", "TELEGRAM_BOT_TOKEN"
    with _native_telegram(tmp_path, monkeypatch, token=saved) as world:
        supervisor, runtime, until = world.supervisor, world.runtime, world.until
        outcomes, real_refresh = [], supervisor.refresh
        held, release = threading.Event(), threading.Event()

        def refresh(*args, **kwargs):
            outcomes.append(real_refresh(*args, **kwargs))
            return outcomes[-1]

        monkeypatch.setattr(supervisor, "refresh", refresh)
        assert extension_loader.load_extension(world.skill, world.load_settings, drive_root=world.drive,
                                               skills=[world.skill], repo_path="") is None
        until(lambda: outcomes == ["unchanged"], "the observer's first refresh never concluded")
        api = world.api()
        read = api.get_settings_state

        def next_read_waits(keys):
            if keys and not release.is_set():
                held.set()
                assert release.wait(15), "the observer was never released"
            return read(keys)

        monkeypatch.setattr(api, "get_settings_state", next_read_waits)
        try:
            assert held.wait(10), "the observer never reached its next read"
            exhausted = runtime()
            with supervisor._lock:
                supervisor._restart_history["telegram:miniapp_gateway"] = (
                    [time.monotonic()] * exhausted.descriptor.max_restarts)
            exhausted.process.kill()
            until(lambda: runtime() is None, "the exhausted companion was never retired")
            world.settings_save(other)
            recovery = extension_loader.ensure_companions_running("telegram", world.drive, world.load_settings,
                                                                  repo_path="", selected_skill=world.skill)
            assert recovery["action"] == "started_missing", recovery
            recovered = runtime()
            assert recovered.descriptor.env.get(key) == other
            world.settings_save(saved)
        finally:
            release.set()
        seen = {}

        def returned():
            current = runtime()
            seen["token"] = None if current is None else current.descriptor.env.get(key, "")
            return seen["token"] == saved and current.process.poll() is None and outcomes[-1:] == ["restarted"]

        deadline = time.monotonic() + 20
        while not returned() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert returned(), f"the Mini App stays on {seen['token']!r}, Settings hold {saved!r}; refreshes {outcomes}"
        settled = runtime().process
        time.sleep(4.5)  # two more reads of the unchanged value
        assert outcomes == ["unchanged", "restarted"] and runtime().process is settled
        assert recovered.process.poll() is not None and world.load_settings().get(key, "") == saved


def test_a_saved_replacement_brings_back_native_typing_the_refused_token_had_muted(tmp_path, monkeypatch, caplog):
    """The shipped payload through the real loader: the host prepares a reply for the owner's chat and Telegram
    refuses the old token (401), so the bridge's typing cue mutes that chat, once and loudly. The owner saves a
    working token on the Settings route: the skill's own observer applies it and wakes the cue, which types through
    the new token with no message delivered first, and never again through the old one. Only the host's reply
    projection is scripted; the Bot API is the loopback stand-in."""
    from ouroboros.gateway import reply_preparation

    token_a, token_b = "111:SYNTHETIC-A", "222:SYNTHETIC-B"
    monkeypatch.setattr(reply_preparation, "reply_preparation",
                        lambda _data_dir: {"complete": True, "chats": [{"chat_id": 1, "transport": {}}]})
    with _native_telegram(tmp_path, monkeypatch, token=token_a,
                          refuse=lambda token, _method: token == token_a) as world:
        asked, until = world.asked, world.until

        def asked_by(token, method):
            return [1 for asked_token, asked_method in list(asked) if (asked_token, asked_method) == (token, method)]

        def muted():
            return [record for record in caplog.records if "typing indicator stopped" in record.getMessage()]

        assert extension_loader.load_extension(world.skill, world.load_settings, drive_root=world.drive,
                                               skills=[world.skill], repo_path="") is None
        until(lambda: asked_by(token_a, "sendChatAction") and muted(), "the old token's refusal never muted typing")
        world.settings_save(token_b)
        until(lambda: asked_by(token_b, "sendChatAction"), "typing never came back through the replacement")
        assert not asked_by(token_b, "sendMessage"), "it came back before any message reached the chat"
        until(lambda: len(asked_by(token_b, "sendChatAction")) >= 2, "the replacement's typing was not renewed")
        assert len(asked_by(token_a, "sendChatAction")) == 1 and len(muted()) == 1


@pytest.mark.parametrize("control", ["off", "revoke", "reload"])
def test_closing_runtime_access_stops_a_send_in_flight_before_the_unload_callbacks(tmp_path, monkeypatch, control):
    """Off, a revoked grant and a reload all unload through the real lifecycle, which closes the API's runtime
    access before it stops the companion. With that stop held for seconds and the first part of a long reply
    still unanswered, the reply's next request is refused at once: after access closed nothing reaches the Bot
    API, although the skill's unload callbacks (which close the token binding too) have not run yet."""
    import time

    from ouroboros.skill_loader import GRANTS_FILENAME, skill_state_dir

    first_part, stop_held = threading.Event(), threading.Event()
    answered_first = threading.Event()

    def hold(_token, method):
        if method == "sendMessage" and not answered_first.is_set():
            answered_first.set()
            first_part.wait(30)  # the first part stays unanswered until access has closed

    with _native_telegram(tmp_path, monkeypatch, token="111:SYNTHETIC-A", hold=hold) as world:
        assert extension_loader.load_extension(world.skill, world.load_settings, drive_root=world.drive,
                                               skills=[world.skill], repo_path="") is None
        world.until(lambda: ("111:SYNTHETIC-A", "getUpdates") in world.asked, "the poller never started")
        api = world.api()
        real_stop, callbacks_ran = world.supervisor.stop, []

        def slow_stop(skill_name, name, timeout_sec=5.0):
            stop_held.wait(30)  # a companion whose stop takes long
            return real_stop(skill_name, name, timeout_sec=timeout_sec)

        monkeypatch.setattr(world.supervisor, "stop", slow_stop)
        real_callback = extension_loader._run_unload_callback
        monkeypatch.setattr(extension_loader, "_run_unload_callback",
                            lambda *args, **kwargs: (callbacks_ran.append(time.monotonic()),
                                                     real_callback(*args, **kwargs))[-1])
        world.bus.publish("chat.outbound", {"chat_id": 1, "role": "assistant", "markdown": False,
                                            "text": "Every word of this answer matters. " * 700})
        world.until(answered_first.is_set, "the reply's first part never reached the Bot API")

        def lifecycle():
            if control == "off":
                save_enabled(world.drive, "telegram", False)
            elif control == "revoke":
                (skill_state_dir(world.drive, "telegram") / GRANTS_FILENAME).unlink()
            else:  # a new revision of the payload, reviewed: the live load is replaced
                (world.payload / "SKILL.md").write_text(
                    (world.payload / "SKILL.md").read_text(encoding="utf-8") + "\n", encoding="utf-8")
                world.admit()
            return extension_loader.reconcile_extension("telegram", world.drive, world.load_settings, repo_path="")

        outcome: dict = {}
        cycling = threading.Thread(target=lambda: outcome.setdefault("state", lifecycle()), daemon=True)
        cycling.start()
        world.until(lambda: api._runtime_closing, "runtime access never closed")
        closed_at = len(world.asked)
        first_part.set()  # the in-flight part completes now, with its observed outcome
        time.sleep(1.5)
        assert not callbacks_ran, "the unload callbacks ran before the companion stop was released"
        assert world.asked[closed_at:] == [], f"requests after access closed: {world.asked[closed_at:]}"
        stop_held.set()
        cycling.join(30)
        assert outcome["state"]["action"] == ("extension_loaded" if control == "reload" else "extension_unloaded")
        late = [entry for entry in world.asked[closed_at:] if entry[1] == "sendMessage"]
        assert late == [], "the old load's reply went on after it was replaced"


class SimpleNamespaceBridge:
    def send_quiz_state(self, *_args, **_kwargs):
        pass


# The companion entry the Mini App witness runs in its private copy of the payload: the shipped companion.py
# unchanged, with only the two endpoints that would leave the machine replaced: the pinned cloudflared (here a
# stand-in that prints a fresh Quick Tunnel URL per generation and records where it would forward) and the
# public HTTPS probe of that URL (here: the stand-in tunnel is alive).
_SYNTHETIC_COMPANION = """
import json, os, secrets, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cloudflare_tunnel, companion
from public_gateway import PublicProbeOutcome


async def ensure_cloudflared(state_dir):
    root = cloudflare_tunnel._safe_state_root(Path(state_dir))
    binary = cloudflare_tunnel._installed_path(root, cloudflare_tunnel._current_asset())
    binary.parent.mkdir(parents=True, exist_ok=True)
    slug = "gen-" + secrets.token_hex(6)
    record = Path(state_dir) / f"synthetic-tunnel-{slug}.json"
    binary.write_text(
        "#!" + sys.executable + "\\n"
        "import json, sys, time\\n"
        f"open({str(record)!r}, 'w').write(json.dumps({{'host': {slug + '.trycloudflare.com'!r}, 'target': sys.argv[-1]}}))\\n"
        f"print('INF |  https://{slug}.trycloudflare.com', file=sys.stderr, flush=True)\\n"
        "while True: time.sleep(1)\\n", encoding="utf-8")
    binary.chmod(0o700)
    return binary


async def probe_public_gateway_once(public_url, tunnel, **_kwargs):
    return PublicProbeOutcome.READY if tunnel.returncode is None else PublicProbeOutcome.TUNNEL_EXITED


companion.ensure_cloudflared = ensure_cloudflared
companion.probe_public_gateway_once = probe_public_gateway_once
cloudflare_tunnel._verify_binary = lambda *_args, **_kwargs: None
companion.main()
"""


def _use_synthetic_tunnel(payload: pathlib.Path) -> None:
    (payload / "scripts" / "synthetic_companion.py").write_text(_SYNTHETIC_COMPANION, encoding="utf-8")
    manifest = payload / "SKILL.md"
    text = manifest.read_text(encoding="utf-8")
    assert "command: [python3, scripts/companion.py]" in text
    manifest.write_text(text.replace("command: [python3, scripts/companion.py]",
                                     "command: [python3, scripts/synthetic_companion.py]"), encoding="utf-8")


def _init_data(token: str, owner: int) -> str:
    """Telegram's signed WebApp launch data for ``owner``, as a client of the bot ``token`` would send it."""
    import hashlib
    import time
    import urllib.parse

    fields = {"auth_date": str(int(time.time())), "user": json.dumps({"id": owner, "first_name": "Owner"})}
    check = "\n".join(f"{key}={value}" for key, value in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    return urllib.parse.urlencode({**fields, "hash": hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()})


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable stand-in and SIGTERM shutdown")
def test_a_rotated_token_ends_the_old_mini_app_generation_and_the_new_one_serves(tmp_path, monkeypatch):
    """The real companion, sidecar, sessions and menu manager, respawned by the host's settings refresh: after
    the token rotates, the old generation's cookie, WebSocket, port and public URL are gone, the old bot's
    launch data no longer opens a session, and the new generation serves under a new URL its own bot's menu
    points to. Only cloudflared and the public probe are stand-ins; nothing leaves the machine."""
    import asyncio
    import socket
    import time

    import httpx
    import uvicorn
    import websockets
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, PlainTextResponse
    from starlette.routing import Route, WebSocketRoute

    async def echo(websocket):
        await websocket.accept()
        try:
            while True:
                await websocket.send_text("core:" + await websocket.receive_text())
        except Exception:
            return

    core = Starlette(routes=[Route("/api/health", lambda _r: JSONResponse({"status": "ok"})),
                             Route("/", lambda _r: PlainTextResponse("the core")),
                             WebSocketRoute("/ws", echo)])
    core_server = uvicorn.Server(uvicorn.Config(core, host="127.0.0.1", port=0, log_level="warning", lifespan="off"))
    core_thread = threading.Thread(target=core_server.run, daemon=True)
    core_thread.start()
    deadline = time.monotonic() + 30
    while not core_server.started:
        assert time.monotonic() < deadline, "the stand-in core never started"
        time.sleep(0.02)
    core_port = core_server.servers[0].sockets[0].getsockname()[1]
    token_a, token_b = "111:SYNTHETIC-A", "222:SYNTHETIC-B"

    def generation(world, after=()):
        """The public URL the companion published once ready, and the local port its tunnel forwards to."""
        world.until(lambda: world.published().get("state") == "ready"
                    and world.published().get("public_url") not in after, "the Mini App never became ready",
                    timeout=60.0)
        url = world.published()["public_url"]
        host = url[len("https://"):].rstrip("/")
        record = json.loads((world.state / f"synthetic-tunnel-{host.split('.')[0]}.json").read_text())
        return url, host, int(record["target"].rsplit(":", 1)[1])

    def request(port, host, method, path, *, cookie="", json_body=None):
        headers = {"Host": host, "Origin": f"https://{host}"}
        if cookie:
            headers["Cookie"] = cookie
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False) as client:
            return client.request(method, path, headers=headers, json=json_body)

    def session(port, host, token):
        response = request(port, host, "POST", "/__telegram/auth", json_body={"init_data": _init_data(token, 42)})
        assert response.status_code == 204, (response.status_code, response.text)
        value = response.headers["set-cookie"].split(";", 1)[0]
        assert value.startswith("__Host-ouroboros-telegram=")
        return value

    async def open_socket(port, host, cookie):
        sock = socket.create_connection(("127.0.0.1", port))
        connection = await websockets.connect(f"ws://{host}/ws", sock=sock, origin=f"https://{host}",
                                              additional_headers={"Cookie": cookie}, open_timeout=10)
        await connection.send("hello")
        assert await asyncio.wait_for(connection.recv(), 10) == "core:hello"
        return connection

    try:
        with _native_telegram(tmp_path, monkeypatch, token=token_a, miniapp="on",
                              payload_edit=_use_synthetic_tunnel, core_port=core_port) as world:
            assert extension_loader.load_extension(world.skill, world.load_settings, drive_root=world.drive,
                                                   skills=[world.skill], repo_path="") is None
            old_url, old_host, old_port = generation(world)
            first = world.runtime().process
            old_cookie = session(old_port, old_host, token_a)
            assert request(old_port, old_host, "GET", "/__telegram/session", cookie=old_cookie).status_code == 204
            assert request(old_port, old_host, "GET", "/", cookie=old_cookie).text == "the core"
            assert world.menu_sets[-1][0] == token_a and world.menu_sets[-1][1]["web_app"]["url"] == old_url
            loop = asyncio.new_event_loop()
            try:
                old_socket = loop.run_until_complete(open_socket(old_port, old_host, old_cookie))

                world.settings_save(token_b)  # the owner's Save: applied by the host refresh, not by Disable/Enable
                new_url, new_host, new_port = generation(world, after=(old_url,))
                assert first.poll() is not None, "the old generation's process outlived the rotation"
                with pytest.raises(websockets.ConnectionClosed):  # the old socket ended with its process
                    loop.run_until_complete(asyncio.wait_for(old_socket.recv(), 10))
                new_cookie = session(new_port, new_host, token_b)
                new_socket = loop.run_until_complete(open_socket(new_port, new_host, new_cookie))
                loop.run_until_complete(new_socket.close())
            finally:
                loop.close()
            assert new_url != old_url and new_port != old_port
            with pytest.raises(httpx.ConnectError):  # nothing listens where the old generation served
                request(old_port, old_host, "GET", "/__telegram/session", cookie=old_cookie)
            assert request(new_port, old_host, "GET", "/", cookie=new_cookie).status_code == 421, "the old URL"
            assert request(new_port, new_host, "GET", "/__telegram/session", cookie=old_cookie).status_code == 401
            refused = request(new_port, new_host, "POST", "/__telegram/auth",
                              json_body={"init_data": _init_data(token_a, 42)})
            assert refused.status_code == 401, "the old bot's launch data opened a session"
            assert request(new_port, new_host, "GET", "/", cookie=new_cookie).text == "the core"
            menu = [(token, button.get("web_app", {}).get("url")) for token, button in world.menu_sets]
            assert menu[-1] == (token_b, new_url), menu
            assert (token_a, None) in menu, "the old bot's menu was not given back on its way out"
    finally:
        core_server.should_exit = True
        core_thread.join(10)
