import importlib
import json
import os
import pathlib
import subprocess
import sys
import types

import ouroboros.launcher_bootstrap as bootstrap_module


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_REPO_BUNDLE = REPO_ROOT / "scripts" / "build_repo_bundle.py"


# These keys must be scrubbed from every ``build_repo_bundle.py``
# subprocess: GitHub Actions tag-push runs set GITHUB_REF_* globally,
# which otherwise bleeds into temp-repo subprocesses and confuses
# ``_resolve_release_tag``.
_BUILD_BUNDLE_ENV_SCRUB_KEYS = (
    "OUROBOROS_RELEASE_TAG",
    "GITHUB_REF",
    "GITHUB_REF_TYPE",
    "GITHUB_REF_NAME",
)


def _scrubbed_env() -> "dict[str, str]":
    env = dict(os.environ)
    for key in _BUILD_BUNDLE_ENV_SCRUB_KEYS:
        env.pop(key, None)
    return env


def _reload_bootstrap():
    return importlib.reload(bootstrap_module)


def _log_stub():
    return types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )


def _make_context(bundle_dir, repo_dir):
    return bootstrap_module.BootstrapContext(
        bundle_dir=bundle_dir,
        repo_dir=repo_dir,
        data_dir=repo_dir.parent / "data",
        settings_path=repo_dir.parent / "settings.json",
        embedded_python=sys.executable,
        app_version="4.50.0-rc.2",
        hidden_run=subprocess.run,
        save_settings=lambda settings: None,
        log=_log_stub(),
    )


def _run(cmd, *, cwd):
    subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True, text=True)


def _git_output(cwd, *args):
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _make_bundle_source(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _run(["git", "init"], cwd=src)
    _run(["git", "config", "user.name", "Test"], cwd=src)
    _run(["git", "config", "user.email", "test@example.com"], cwd=src)
    _run(["git", "checkout", "-b", "ouroboros"], cwd=src)
    _run(["git", "remote", "add", "origin", "https://github.com/razzant/ouroboros.git"], cwd=src)
    (src / "VERSION").write_text("4.50.0-rc.2\n", encoding="utf-8")
    (src / "server.py").write_text("print('bundle-v1')\n", encoding="utf-8")
    _run(["git", "add", "VERSION", "server.py"], cwd=src)
    _run(["git", "commit", "-m", "bundle v1"], cwd=src)
    _run(["git", "tag", "-a", "v4.50.0-rc.2", "-m", "Release v4.50.0-rc.2"], cwd=src)
    return src


def _write_bundle(repo_src, bundle_dir):
    bundle_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(BUILD_REPO_BUNDLE),
            "--repo-root",
            str(repo_src),
            "--output-bundle",
            str(bundle_dir / "repo.bundle"),
            "--output-manifest",
            str(bundle_dir / "repo_bundle_manifest.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_scrubbed_env(),
    )


def test_ensure_managed_repo_clones_from_embedded_bundle(tmp_path):
    bootstrap = _reload_bootstrap()
    src = _make_bundle_source(tmp_path)
    bundle_dir = tmp_path / "bundle"
    repo_dir = tmp_path / "repo"
    _write_bundle(src, bundle_dir)

    ctx = _make_context(bundle_dir, repo_dir)
    outcome = bootstrap.ensure_managed_repo(ctx)

    assert outcome == "created"
    assert (repo_dir / ".git").is_dir()
    assert (repo_dir / "server.py").read_text(encoding="utf-8") == "print('bundle-v1')\n"
    assert _git_output(repo_dir, "branch", "--show-current") == "ouroboros"
    meta = bootstrap.load_repo_manifest(repo_dir)
    assert meta["managed_remote_branch"] == "ouroboros"
    assert meta["managed_local_branch"] == "ouroboros"
    assert meta["release_tag"] == "v4.50.0-rc.2"
    assert meta["bundle_sha256"]
    assert (repo_dir / ".git" / bootstrap.BOOTSTRAP_PIN_MARKER_NAME).exists()


def test_ensure_managed_repo_preserves_origin_on_unchanged_boot(tmp_path):
    bootstrap = _reload_bootstrap()
    src = _make_bundle_source(tmp_path)
    bundle_dir = tmp_path / "bundle"
    repo_dir = tmp_path / "repo"
    _write_bundle(src, bundle_dir)

    ctx = _make_context(bundle_dir, repo_dir)
    assert bootstrap.ensure_managed_repo(ctx) == "created"

    _run(["git", "remote", "add", "origin", "https://github.com/example/fork.git"], cwd=repo_dir)

    outcome = bootstrap.ensure_managed_repo(ctx)

    assert outcome == "unchanged"
    remotes = set(_git_output(repo_dir, "remote").splitlines())
    assert remotes == {"managed", "origin"}


def test_sync_existing_repo_from_bundle_replaces_legacy_snapshot(tmp_path):
    bootstrap = _reload_bootstrap()
    src = _make_bundle_source(tmp_path)
    bundle_dir = tmp_path / "bundle"
    repo_dir = tmp_path / "repo"
    _write_bundle(src, bundle_dir)

    repo_dir.mkdir()
    (repo_dir / "server.py").write_text("print('legacy-snapshot')\n", encoding="utf-8")

    ctx = _make_context(bundle_dir, repo_dir)
    bootstrap.sync_existing_repo_from_bundle(ctx)

    assert (repo_dir / ".git").is_dir()
    assert (repo_dir / "server.py").read_text(encoding="utf-8") == "print('bundle-v1')\n"
    archived = list((ctx.data_dir / "archive" / "managed_repo").iterdir())
    assert archived
    assert (archived[0] / "server.py").read_text(encoding="utf-8") == "print('legacy-snapshot')\n"


def test_ensure_managed_repo_preserves_checkout_when_embedded_bundle_changes(tmp_path):
    bootstrap = _reload_bootstrap()
    src = _make_bundle_source(tmp_path)
    bundle_dir = tmp_path / "bundle"
    repo_dir = tmp_path / "repo"
    _write_bundle(src, bundle_dir)

    ctx = _make_context(bundle_dir, repo_dir)
    assert bootstrap.ensure_managed_repo(ctx) == "created"
    _run(["git", "remote", "add", "origin", "https://github.com/example/fork.git"], cwd=repo_dir)
    (repo_dir / "server.py").write_text("print('local-self-modification')\n", encoding="utf-8")
    _run(["git", "add", "server.py"], cwd=repo_dir)
    _run(["git", "commit", "-m", "local self modification"], cwd=repo_dir)
    local_head = _git_output(repo_dir, "rev-parse", "HEAD")

    (src / "server.py").write_text("print('bundle-v2')\n", encoding="utf-8")
    _run(["git", "add", "server.py"], cwd=src)
    _run(["git", "commit", "-m", "bundle v2"], cwd=src)
    # Re-point the annotated release tag onto the new HEAD so the bundle
    # builder's HEAD-tag check still passes (the test is exercising the
    # bundle-replacement path, not a VERSION bump).
    _run(["git", "tag", "-d", "v4.50.0-rc.2"], cwd=src)
    _run(["git", "tag", "-a", "v4.50.0-rc.2", "-m", "Release v4.50.0-rc.2 (v2)"], cwd=src)
    _write_bundle(src, bundle_dir)

    outcome = bootstrap.ensure_managed_repo(ctx)

    assert outcome == "metadata-updated"
    assert _git_output(repo_dir, "rev-parse", "HEAD") == local_head
    assert (repo_dir / "server.py").read_text(encoding="utf-8") == "print('local-self-modification')\n"
    assert set(_git_output(repo_dir, "remote").splitlines()) == {"managed", "origin"}
    assert bootstrap.load_repo_manifest(repo_dir)["source_sha"] == _git_output(src, "rev-parse", "HEAD")
    assert not (ctx.data_dir / "archive" / "managed_repo").exists()


def test_load_bundle_manifest_rejects_app_version_mismatch(tmp_path):
    bootstrap = _reload_bootstrap()
    src = _make_bundle_source(tmp_path)
    bundle_dir = tmp_path / "bundle"
    repo_dir = tmp_path / "repo"
    _write_bundle(src, bundle_dir)

    ctx = bootstrap.BootstrapContext(
        bundle_dir=bundle_dir,
        repo_dir=repo_dir,
        data_dir=repo_dir.parent / "data",
        settings_path=repo_dir.parent / "settings.json",
        embedded_python=sys.executable,
        app_version="4.50.0-rc.3",
        hidden_run=subprocess.run,
        save_settings=lambda settings: None,
        log=_log_stub(),
    )

    try:
        bootstrap.load_bundle_manifest(ctx)
        assert False, "Expected app_version mismatch to raise"
    except RuntimeError as exc:
        assert "app_version" in str(exc)


def test_load_bundle_manifest_rejects_release_tag_mismatch(tmp_path):
    bootstrap = _reload_bootstrap()
    src = _make_bundle_source(tmp_path)
    bundle_dir = tmp_path / "bundle"
    repo_dir = tmp_path / "repo"
    _write_bundle(src, bundle_dir)

    manifest_path = bundle_dir / "repo_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_tag"] = "v4.50.0-rc.3"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    ctx = _make_context(bundle_dir, repo_dir)

    try:
        bootstrap.load_bundle_manifest(ctx)
        assert False, "Expected release_tag mismatch to raise"
    except RuntimeError as exc:
        assert "release_tag" in str(exc)


def test_ensure_managed_repo_rejects_tampered_bundle(tmp_path):
    bootstrap = _reload_bootstrap()
    src = _make_bundle_source(tmp_path)
    bundle_dir = tmp_path / "bundle"
    repo_dir = tmp_path / "repo"
    _write_bundle(src, bundle_dir)
    (bundle_dir / "repo.bundle").write_bytes(b"tampered-bundle")

    ctx = _make_context(bundle_dir, repo_dir)

    try:
        bootstrap.ensure_managed_repo(ctx)
        assert False, "Expected bundle hash mismatch to raise"
    except RuntimeError as exc:
        assert "bundle hash mismatch" in str(exc)


# ---------------------------------------------------------------------------
# Host service port cleanup (formerly test_launcher_host_service_cleanup.py)
# ---------------------------------------------------------------------------


def test_host_service_cleanup_uses_configured_port(monkeypatch):
    import launcher

    killed: list[int] = []

    monkeypatch.setenv("OUROBOROS_HOST_SERVICE_PORT", "9876")
    monkeypatch.setattr(launcher, "_kill_stale_on_port", lambda port: killed.append(port))

    launcher._kill_stale_runtime_ports(8765)

    assert killed == [8765, 9876]


def test_agent_lifecycle_preflight_cleans_host_service_port(monkeypatch):
    import launcher

    killed: list[int] = []

    class FakeProcess:
        pid = 12345
        returncode = 0

        def wait(self):
            launcher._shutdown_event.set()

    launcher._shutdown_event.clear()
    monkeypatch.setattr(launcher, "_host_service_port", lambda: 9876)
    monkeypatch.setattr(launcher, "_kill_stale_on_port", lambda port: killed.append(port))
    monkeypatch.setattr(launcher, "start_agent", lambda port: FakeProcess())
    monkeypatch.setattr(launcher, "_poll_port_file", lambda timeout=30: 8765)
    monkeypatch.setattr(launcher, "_wait_for_server", lambda port, timeout=30.0: True)
    monkeypatch.setattr(launcher, "_agent_job", None)
    monkeypatch.setattr(launcher, "log", types.SimpleNamespace(info=lambda *args, **kwargs: None))

    try:
        launcher.agent_lifecycle_loop(port=8765)
    finally:
        launcher._shutdown_event.clear()

    assert killed[:2] == [8765, 9876]


def test_start_agent_unix_uses_process_group_and_writes_server_record(monkeypatch, tmp_path):
    import launcher

    data_dir = tmp_path / "data"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "server.py").write_text("print('server')\n", encoding="utf-8")
    captured = {}

    class FakeStdout:
        def readline(self):
            return b""

    class FakeProcess:
        pid = 12345
        stdout = FakeStdout()

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(launcher, "IS_WINDOWS", False)
    monkeypatch.setattr(launcher, "DATA_DIR", data_dir)
    monkeypatch.setattr(launcher, "REPO_DIR", repo_dir)
    monkeypatch.setattr(launcher, "EMBEDDED_PYTHON", sys.executable)
    bundle_dir = tmp_path / "immutable-bundle"
    monkeypatch.setattr(launcher, "_bundle_dir", lambda: bundle_dir)
    # An inherited value must NOT win: the asset path is sealed by the launcher,
    # not read from whatever environment the process happened to start in.
    monkeypatch.setenv(
        "OUROBOROS_EXECD_BUNDLE_DIR",
        str(tmp_path / "untrusted-inherited-path"),
    )
    monkeypatch.setattr(launcher, "_load_settings", lambda: {})
    monkeypatch.setattr(launcher, "_apply_settings_to_env", lambda _settings: None)
    monkeypatch.setattr(launcher, "subprocess_new_group_kwargs", lambda: {"start_new_session": True})
    monkeypatch.setattr(launcher, "_hidden_popen", fake_popen)
    monkeypatch.setattr(launcher, "process_group_id", lambda _pid: 12345)

    proc = launcher.start_agent(port=9876)

    assert proc.pid == 12345
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["env"]["OUROBOROS_EXECD_BUNDLE_DIR"] == str(
        (bundle_dir / "assets" / "execd").resolve()
    )
    record = json.loads((data_dir / "state" / "server_process.json").read_text(encoding="utf-8"))
    assert record["pid"] == 12345
    assert record["pgid"] == 12345
    assert record["requested_port"] == 9876
    assert record["port"] == 9876
    assert record["repo_dir"] == str(repo_dir.resolve())
    assert record["server_path"] == str((repo_dir / "server.py").resolve())
    assert record["argv"] == [sys.executable, str((repo_dir / "server.py").resolve())]
    assert record["created_at"]


def test_recorded_server_cleanup_ignores_unrelated_pid(monkeypatch, tmp_path):
    import launcher

    data_dir = tmp_path / "data"
    repo_dir = tmp_path / "repo"
    (data_dir / "state").mkdir(parents=True)
    repo_dir.mkdir()
    server_py = repo_dir / "server.py"
    server_py.write_text("print('server')\n", encoding="utf-8")
    (data_dir / "state" / "server_process.json").write_text(
        json.dumps({
            "pid": 22222,
            "pgid": 22222,
            "server_path": str(server_py.resolve()),
            "repo_dir": str(repo_dir.resolve()),
            "port": 8765,
        }),
        encoding="utf-8",
    )
    killed = []
    monkeypatch.setattr(launcher, "IS_WINDOWS", False)
    monkeypatch.setattr(launcher, "DATA_DIR", data_dir)
    monkeypatch.setattr(launcher, "REPO_DIR", repo_dir)
    monkeypatch.setattr(launcher, "pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(launcher, "process_group_id", lambda _pid: 22222)
    monkeypatch.setattr(launcher, "process_command", lambda _pid: "/usr/bin/python unrelated.py")
    monkeypatch.setattr(launcher, "kill_pid_tree", lambda pid: killed.append(("pid", pid)))
    monkeypatch.setattr(launcher, "kill_process_group_id", lambda pgid: killed.append(("pgid", pgid)))
    monkeypatch.setattr(launcher, "terminate_process_group_id", lambda pgid: killed.append(("term", pgid)))

    launcher._cleanup_recorded_server_process("test")

    assert killed == []
    assert not (data_dir / "state" / "server_process.json").exists()


def test_recorded_server_cleanup_kills_verified_process_group(monkeypatch, tmp_path):
    import launcher

    data_dir = tmp_path / "data"
    repo_dir = tmp_path / "repo"
    (data_dir / "state").mkdir(parents=True)
    repo_dir.mkdir()
    server_py = repo_dir / "server.py"
    server_py.write_text("print('server')\n", encoding="utf-8")
    (data_dir / "state" / "server_process.json").write_text(
        json.dumps({
            "pid": 33333,
            "pgid": 33333,
            "server_path": str(server_py.resolve()),
            "repo_dir": str(repo_dir.resolve()),
            "port": 8765,
        }),
        encoding="utf-8",
    )
    killed = []
    monkeypatch.setattr(launcher, "IS_WINDOWS", False)
    monkeypatch.setattr(launcher, "DATA_DIR", data_dir)
    monkeypatch.setattr(launcher, "REPO_DIR", repo_dir)
    monkeypatch.setattr(launcher, "pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(launcher, "process_group_id", lambda _pid: 33333)
    monkeypatch.setattr(launcher, "process_command", lambda _pid: f"{sys.executable} {server_py}")
    monkeypatch.setattr(launcher, "current_process_group_id", lambda: 99999)
    monkeypatch.setattr(launcher, "time", types.SimpleNamespace(sleep=lambda *_a, **_k: None))
    monkeypatch.setattr(launcher, "terminate_process_group_id", lambda pgid: killed.append(("term", pgid)))
    monkeypatch.setattr(launcher, "kill_process_group_id", lambda pgid: killed.append(("killpg", pgid)))
    monkeypatch.setattr(launcher, "kill_pid_tree", lambda pid: killed.append(("pid", pid)))

    launcher._cleanup_recorded_server_process("test")

    assert ("term", 33333) in killed
    assert ("killpg", 33333) in killed
    assert ("pid", 33333) in killed
    assert not (data_dir / "state" / "server_process.json").exists()


def test_recorded_server_cleanup_ignores_mismatched_process_group(monkeypatch, tmp_path):
    import launcher

    data_dir = tmp_path / "data"
    repo_dir = tmp_path / "repo"
    (data_dir / "state").mkdir(parents=True)
    repo_dir.mkdir()
    server_py = repo_dir / "server.py"
    server_py.write_text("print('server')\n", encoding="utf-8")
    record_path = data_dir / "state" / "server_process.json"
    record_path.write_text(
        json.dumps({
            "pid": 44444,
            "pgid": 55555,
            "server_path": str(server_py.resolve()),
            "repo_dir": str(repo_dir.resolve()),
            "port": 8765,
        }),
        encoding="utf-8",
    )
    killed = []
    monkeypatch.setattr(launcher, "IS_WINDOWS", False)
    monkeypatch.setattr(launcher, "DATA_DIR", data_dir)
    monkeypatch.setattr(launcher, "REPO_DIR", repo_dir)
    monkeypatch.setattr(launcher, "pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(launcher, "process_group_id", lambda _pid: 44444)
    monkeypatch.setattr(launcher, "process_command", lambda _pid: f"{sys.executable} {server_py}")
    monkeypatch.setattr(launcher, "terminate_process_group_id", lambda pgid: killed.append(("term", pgid)))
    monkeypatch.setattr(launcher, "kill_process_group_id", lambda pgid: killed.append(("killpg", pgid)))
    monkeypatch.setattr(launcher, "kill_pid_tree", lambda pid: killed.append(("pid", pid)))

    launcher._cleanup_recorded_server_process("test")

    assert killed == []
    assert not record_path.exists()


def test_update_server_process_record_port_records_actual_port(monkeypatch, tmp_path):
    import launcher

    data_dir = tmp_path / "data"
    record_dir = data_dir / "state"
    record_dir.mkdir(parents=True)
    record_path = record_dir / "server_process.json"
    record_path.write_text(
        json.dumps({
            "pid": 123,
            "pgid": 123,
            "server_path": "/tmp/server.py",
            "repo_dir": "/tmp/repo",
            "requested_port": 8765,
            "port": 8765,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "DATA_DIR", data_dir)

    launcher._update_server_process_record_port(123, 8769)

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["requested_port"] == 8765
    assert record["port"] == 8769
    assert record["port_updated_at"]


def test_start_agent_windows_assigns_job_before_resume_and_records(monkeypatch, tmp_path):
    import launcher

    data_dir = tmp_path / "data"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "server.py").write_text("print('server')\n", encoding="utf-8")
    calls: list[tuple[str, object]] = []

    class FakeStdout:
        def readline(self):
            return b""

    class FakeProcess:
        pid = 54321
        stdout = FakeStdout()

        def kill(self):
            calls.append(("kill", self.pid))

    def fake_popen(_cmd, **kwargs):
        calls.append(("popen_flags", kwargs.get("creationflags")))
        calls.append(("popen_env", kwargs.get("env", {})))
        return FakeProcess()

    monkeypatch.setattr(launcher, "IS_WINDOWS", True)
    monkeypatch.setattr(launcher, "DATA_DIR", data_dir)
    monkeypatch.setattr(launcher, "REPO_DIR", repo_dir)
    monkeypatch.setattr(launcher, "EMBEDDED_PYTHON", sys.executable)
    monkeypatch.setattr(launcher, "_CREATE_NEW_PROCESS_GROUP", 0x200)
    monkeypatch.setattr(launcher, "_CREATE_SUSPENDED", 0x4)
    monkeypatch.setattr(launcher, "_load_settings", lambda: {})
    monkeypatch.setattr(launcher, "_apply_settings_to_env", lambda _settings: None)
    monkeypatch.setattr(launcher, "_hidden_popen", fake_popen)
    monkeypatch.setattr(launcher, "create_kill_on_close_job", lambda: calls.append(("create_job", None)) or "job")
    monkeypatch.setattr(launcher, "assign_pid_to_job", lambda job, pid: calls.append(("assign", (job, pid))) or True)
    monkeypatch.setattr(launcher, "resume_process", lambda pid: calls.append(("resume", pid)) or True)

    proc = launcher.start_agent(port=8765)

    assert proc.pid == 54321
    env = next(value for key, value in calls if key == "popen_env")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONPYCACHEPREFIX"] == str(data_dir / "state" / "pycache")
    lifecycle_calls = [call for call in calls if call[0] != "popen_env"]
    assert lifecycle_calls[:4] == [
        ("popen_flags", 0x204),
        ("create_job", None),
        ("assign", ("job", 54321)),
        ("resume", 54321),
    ]
    record = json.loads((data_dir / "state" / "server_process.json").read_text(encoding="utf-8"))
    assert record["pid"] == 54321
    assert record["port"] == 8765
