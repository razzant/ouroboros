from __future__ import annotations

import json
import os
import subprocess

from ouroboros.marketplace.install_specs import normalize_install_specs
from ouroboros.marketplace.isolated_deps import (
    DEPS_STATE_FILENAME,
    FINGERPRINT_FILENAME,
    _installer_env,
    _run,
    augment_env_for_skill_deps,
    isolated_env_dir,
    read_deps_state,
)
from ouroboros.skill_loader import skill_state_dir


def test_installer_env_scrubs_secret_keys_and_uses_isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setenv("USERPROFILE", "/host/profile")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = _installer_env(tmp_path / ".ouroboros_env")
    assert "OPENAI_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert env["HOME"].startswith(str(tmp_path))
    assert env["USERPROFILE"].startswith(str(tmp_path))
    assert env["APPDATA"].startswith(str(tmp_path))
    assert env["LOCALAPPDATA"].startswith(str(tmp_path))
    assert env["PIP_CACHE_DIR"].startswith(str(tmp_path))
    assert env["npm_config_cache"].startswith(str(tmp_path))


def test_installer_env_node_emergency_prepends_bundled_node_dir(monkeypatch, tmp_path):
    """Emergency only (PATH node dead + healthy bundled selected): the node
    installer env PATH gains the bundled-node dir so npm's `#!/usr/bin/env node`
    shebang resolves the working runtime. Python installs never get it."""
    from ouroboros import node_runtime

    monkeypatch.setenv("PATH", "/usr/bin")
    bundle_bin = str(tmp_path / "bundle" / "bin")
    monkeypatch.setattr(
        node_runtime, "skill_node_emergency_path_dir", lambda timeout_sec=10: bundle_bin
    )

    node_env = _installer_env(tmp_path / ".ouroboros_env", ecosystem="node")
    assert node_env["PATH"] == os.pathsep.join([bundle_bin, "/usr/bin"])

    python_env = _installer_env(tmp_path / ".ouroboros_env", ecosystem="python")
    assert python_env["PATH"] == "/usr/bin"


def test_installer_env_node_healthy_system_is_byte_identical(monkeypatch, tmp_path):
    from ouroboros import node_runtime

    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(
        node_runtime, "skill_node_emergency_path_dir", lambda timeout_sec=10: ""
    )

    node_env = _installer_env(tmp_path / ".ouroboros_env", ecosystem="node")
    baseline = _installer_env(tmp_path / ".ouroboros_env")
    assert node_env == baseline
    assert node_env["PATH"] == "/usr/bin"


def test_install_node_package_missing_npm_fails_honestly(monkeypatch, tmp_path):
    """npm itself is not bundled: its absence stays a typed RuntimeError."""
    import pytest

    from ouroboros.marketplace import isolated_deps

    monkeypatch.setattr(isolated_deps.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="npm is not available on PATH"):
        isolated_deps._install_node_package("left-pad", tmp_path / ".ouroboros_env", 1)


def test_normalize_install_specs_rejects_vcs_urls_and_expands_packages():
    auto, manual, warnings = normalize_install_specs([
        {"kind": "pip", "package": "git+https://example.com/pkg.git"},
        {"kind": "npm", "packages": ["left-pad", "@scope/pkg"]},
    ])
    assert [item["package"] for item in auto] == ["left-pad", "@scope/pkg"]
    assert manual and "git+https" in manual[0]["package"]
    assert warnings


def test_augment_env_exposes_python_venv_and_node_path(tmp_path):
    skill_dir = tmp_path / "skill"
    py_bin = skill_dir / ".ouroboros_env" / "python" / ("Scripts" if os.name == "nt" else "bin")
    py_bin.mkdir(parents=True)
    (py_bin / ("python.exe" if os.name == "nt" else "python")).write_text("", encoding="utf-8")
    node_modules = skill_dir / ".ouroboros_env" / "node" / "node_modules"
    node_modules.mkdir(parents=True)
    env = augment_env_for_skill_deps({"PATH": "/usr/bin"}, skill_dir)
    assert str(py_bin) in env["PATH"]
    assert env["VIRTUAL_ENV"].startswith(str(skill_dir / ".ouroboros_env" / "python"))
    assert env["NODE_PATH"] == str(node_modules)


def test_read_deps_state_verifies_live_env_fingerprint(tmp_path):
    drive_root = tmp_path / "drive"
    skill_dir = tmp_path / "skill"
    state_dir = skill_state_dir(drive_root, "skill")
    state_dir.mkdir(parents=True, exist_ok=True)
    state = {"status": "installed", "specs_hash": "abc"}
    (state_dir / DEPS_STATE_FILENAME).write_text(json.dumps(state), encoding="utf-8")

    assert read_deps_state(drive_root, "skill")["status"] == "installed"
    assert read_deps_state(drive_root, "skill", skill_dir)["status"] == "missing"

    env_dir = isolated_env_dir(skill_dir)
    env_dir.mkdir(parents=True)
    (env_dir / FINGERPRINT_FILENAME).write_text(json.dumps(state), encoding="utf-8")

    assert read_deps_state(drive_root, "skill", skill_dir)["status"] == "installed"


def test_run_discards_unbounded_installer_output(monkeypatch, tmp_path):
    captured = {}

    class Proc:
        returncode = 0
        def wait(self, timeout=None):
            captured["timeout"] = timeout
            return 0

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        return Proc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    result = _run(["tool"], cwd=tmp_path, env={}, timeout_sec=1)
    assert captured["stdout"] is subprocess.DEVNULL
    assert captured["stderr"] is subprocess.PIPE
    assert captured["stdin"] is subprocess.DEVNULL
    assert captured["timeout"] == 1
    assert "stdout_tail" not in result
    assert "stderr_tail" not in result


def test_failed_python_install_persists_stderr_tail(monkeypatch, tmp_path):
    from ouroboros.marketplace import isolated_deps

    def fake_run(cmd, **kwargs):
        if "venv" in cmd:
            bin_dir = tmp_path / "skill" / ".ouroboros_env" / "python" / ("Scripts" if os.name == "nt" else "bin")
            bin_dir.mkdir(parents=True, exist_ok=True)
            (bin_dir / ("python.exe" if os.name == "nt" else "python")).write_text("", encoding="utf-8")
            return {"returncode": 0}
        return {"returncode": 1, "stderr_tail": "ERROR: no matching distribution found for a2a-sdk"}

    monkeypatch.setattr(isolated_deps, "_run", fake_run)
    try:
        isolated_deps.install_isolated_dependencies(
            tmp_path,
            "skill",
            tmp_path / "skill",
            [{"kind": "pip", "package": "a2a-sdk"}],
        )
    except RuntimeError as exc:
        assert "no matching distribution" in str(exc)
    else:
        raise AssertionError("install should fail")

    state = json.loads((skill_state_dir(tmp_path, "skill") / DEPS_STATE_FILENAME).read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert "pip install failed" in state["error"]
    assert "no matching distribution" in state["error"]


def test_python_and_npm_install_commands_disable_build_scripts(monkeypatch, tmp_path):
    from ouroboros.marketplace import isolated_deps

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "venv" in cmd:
            bin_dir = tmp_path / ".ouroboros_env" / "python" / ("Scripts" if os.name == "nt" else "bin")
            bin_dir.mkdir(parents=True, exist_ok=True)
            (bin_dir / ("python.exe" if os.name == "nt" else "python")).write_text("", encoding="utf-8")
        return {"returncode": 0}

    monkeypatch.setattr(isolated_deps, "_run", fake_run)
    monkeypatch.setattr(isolated_deps.shutil, "which", lambda name: "/usr/bin/npm" if name == "npm" else None)
    isolated_deps._install_python_packages(["wheelpkg"], tmp_path / ".ouroboros_env", 1)
    isolated_deps._install_node_package("left-pad", tmp_path / ".ouroboros_env", 1)
    assert any("--only-binary=:all:" in cmd for cmd in calls)
    assert any("--ignore-scripts" in cmd for cmd in calls)
    assert not any("freeze" in cmd for cmd in calls)


def test_python_packages_are_batched_into_one_pip_invocation(monkeypatch, tmp_path):
    from ouroboros.marketplace import isolated_deps

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "venv" in cmd:
            bin_dir = tmp_path / "skill" / ".ouroboros_env" / "python" / ("Scripts" if os.name == "nt" else "bin")
            bin_dir.mkdir(parents=True, exist_ok=True)
            (bin_dir / ("python.exe" if os.name == "nt" else "python")).write_text("", encoding="utf-8")
        return {"returncode": 0}

    monkeypatch.setattr(isolated_deps, "_run", fake_run)
    isolated_deps.install_isolated_dependencies(
        tmp_path,
        "skill",
        tmp_path / "skill",
        [
            {"kind": "pip", "package": "firstpkg"},
            {"kind": "pip", "package": "secondpkg"},
        ],
    )
    pip_installs = [cmd for cmd in calls if "pip" in cmd and "install" in cmd]
    assert len(pip_installs) == 1
    assert "firstpkg" in pip_installs[0]
    assert "secondpkg" in pip_installs[0]
