"""The deterministic guards that block a self-elevating command or write.

Split verbatim out of ``tests/test_runtime_mode_elevation.py`` by theme. This module
owns the shell elevation and context-mode lowering indicators together with the
read-only diagnostics they must not flag, the files-API owner-only helper, and the
``run_shell`` scans that catch an obfuscated, delayed or detached owner-state writer.

Hermetic — no network, no supervisor boot. Uses temp dirs for ``DATA_DIR`` /
``SETTINGS_PATH`` overrides via monkeypatching ``ouroboros.config`` module-level
constants.
"""

from __future__ import annotations

import pathlib

import pytest


from tests._runtime_mode_elevation_shared import isolated_settings as _isolated_settings

# The fixture is requested by name as a test parameter, so it is re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
isolated_settings = _isolated_settings


def _clear_safety_provider_env(monkeypatch) -> None:
    """Keep post-check tests from depending on live safety LLM credentials."""
    for key in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# 8. Runtime mode elevation chokepoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "blocked_cmd",
    [
        # Combination: save_settings + OUROBOROS_RUNTIME_MODE → blocked.
        "python -c \"from ouroboros.config import save_settings; save_settings({'OUROBOROS_RUNTIME_MODE': 'pro'}, allow_elevation=True)\"",
        "python3 -c \"import ouroboros.config; ouroboros.config.save_settings({'OUROBOROS_RUNTIME_MODE': 'pro'})\"",
        # Dotted-path short-circuit: ouroboros.config.save_settings.
        "python -c \"import ouroboros.config; ouroboros.config.save_settings({})\"",
    ],
)
def test_elevation_indicators_block_attack_patterns_in_all_modes(blocked_cmd, tmp_path, monkeypatch):
    """Iteration-2 fix (real triad finding T1, iter-2 multi-critic F2-6):
    the elevation indicators block actual attack patterns — runs
    ``ToolRegistry.execute("run_command", ...)`` end-to-end in each
    runtime mode and asserts ``ELEVATION_BLOCKED`` is returned. The
    earlier string-level test only verified substring presence; this
    covers the dispatch wiring."""
    from ouroboros.tools.registry import ToolRegistry

    for mode in ("light", "advanced", "pro"):
        monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
        reg = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
        result = reg.execute("run_command", {"cmd": blocked_cmd})
        assert "ELEVATION_BLOCKED" in result, (
            f"mode={mode!r} cmd={blocked_cmd!r}: "
            f"got {result[:200]!r}"
        )


def test_workspace_mode_still_blocks_runtime_mode_elevation(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    workspace.mkdir()
    repo.mkdir()
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg = ToolRegistry(repo_dir=repo, drive_root=data)
    reg.set_context(ToolContext(
        repo_dir=repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
    ))
    result = reg.execute(
        "run_command",
        {"cmd": "python -c \"from ouroboros.config import save_settings; save_settings({'OUROBOROS_RUNTIME_MODE': 'pro'}, allow_elevation=True)\""},
    )

    assert "ELEVATION_BLOCKED" in result


@pytest.mark.parametrize(
    "diagnostic_cmd",
    [
        # Diagnostic queries about the chokepoint must NOT be blocked.
        "echo \"$OUROBOROS_RUNTIME_MODE\"",
        "printenv OUROBOROS_RUNTIME_MODE",
        "grep save_settings ouroboros/config.py",
        "rg save_settings ouroboros/",
        "git log -S save_settings",
        # save_settings without OUROBOROS_RUNTIME_MODE: legitimate dev work.
        "grep -n 'def save_settings' ouroboros/config.py",
    ],
)
def test_elevation_indicators_do_not_false_positive(diagnostic_cmd, tmp_path, monkeypatch):
    """Iteration-2 fix (multi-critic F2-2): diagnostic shell commands
    that mention ``save_settings`` OR ``OUROBOROS_RUNTIME_MODE`` (but
    not both, and not the dotted-path attack form) must NOT trip
    ELEVATION_BLOCKED. The conjunctive check is the discriminator."""
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    result = reg.execute("run_command", {"cmd": diagnostic_cmd})
    assert "ELEVATION_BLOCKED" not in result, (
        f"Diagnostic cmd {diagnostic_cmd!r} was wrongly blocked as "
        "elevation attempt. The conjunctive check should let this pass."
    )


@pytest.mark.parametrize(
    "blocked_cmd",
    [
        "curl -X POST http://127.0.0.1:8765/api/owner/context-mode -d '{\"mode\":\"low\"}'",
        "python -c \"from ouroboros.config import save_settings; save_settings({'OUROBOROS_CONTEXT_MODE': 'low'})\"",
        "python -c \"import json; p='data/settings.json'; json.dump({'OUROBOROS_CONTEXT_MODE':'low'}, open(p,'w'))\"",
        "ouroboros settings context-mode low",
        "python -m ouroboros.cli settings context-mode low",
    ],
)
def test_context_mode_self_lowering_indicators_block_attack_patterns(blocked_cmd, tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    for mode in ("light", "advanced", "pro"):
        monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
        reg = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
        result = reg.execute("run_command", {"cmd": blocked_cmd})
        assert "CONTEXT_MODE_SELF_LOWERING_BLOCKED" in result, (
            f"mode={mode!r} cmd={blocked_cmd!r}: got {result[:200]!r}"
        )


@pytest.mark.parametrize(
    "diagnostic_cmd",
    [
        "echo \"$OUROBOROS_CONTEXT_MODE\"",
        "rg OUROBOROS_CONTEXT_MODE ouroboros/",
        "curl http://127.0.0.1:8765/api/state",
    ],
)
def test_context_mode_guard_does_not_block_readonly_diagnostics(diagnostic_cmd, tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    result = reg.execute("run_command", {"cmd": diagnostic_cmd})
    assert "CONTEXT_MODE_SELF_LOWERING_BLOCKED" not in result


def test_browser_evaluate_context_mode_self_lowering_guard():
    from types import SimpleNamespace

    from ouroboros.tools.browser import _blocks_context_mode_self_lowering_js, _is_context_mode_owner_post

    assert _blocks_context_mode_self_lowering_js(
        "fetch('/api/owner/context-mode', {method:'POST', body: JSON.stringify({mode:'low'})})"
    )
    assert not _blocks_context_mode_self_lowering_js("fetch('/api/state').then(r => r.json())")
    assert _is_context_mode_owner_post(SimpleNamespace(url="http://127.0.0.1:8765/api/owner/context-mode", method="POST"))
    assert not _is_context_mode_owner_post(SimpleNamespace(url="http://127.0.0.1:8765/api/state", method="POST"))


def test_files_api_write_blocks_settings_json(isolated_settings, monkeypatch):
    """Iteration-2 real triad+scope finding SR2: the Files API
    (``/api/files/write``) is a parallel write path that previously
    bypassed both ``_data_write`` and the ``save_settings`` chokepoint.
    Verify the owner-only guard rejects
    writes to the owner-only file. String-level test against the source
    so the assertion is hermetic (full HTTP round-trip belongs in a
    Starlette TestClient suite, but the guard helper is the SSOT)."""
    from ouroboros.gateway import files as fba_mod

    source = pathlib.Path(fba_mod.__file__).read_text(encoding="utf-8")
    # The shared helpers must exist...
    assert "_is_owner_only_settings_file" in source
    assert "_is_owner_only_file" in source
    # ...and must be invoked from each mutating endpoint.
    for endpoint in (
        "api_files_write",
        "api_files_delete",
        "api_files_transfer",
        "api_files_upload",
    ):
        endpoint_idx = source.find(f"async def {endpoint}(")
        assert endpoint_idx != -1, f"Endpoint {endpoint} not found"
        # Find the next async def boundary so we scope the guard search.
        next_idx = source.find("\nasync def ", endpoint_idx + 1)
        body = source[endpoint_idx:next_idx if next_idx != -1 else len(source)]
        assert "_is_owner_only_file" in body or "_contains_owner_only_file" in body, (
            f"Endpoint {endpoint} must call ``_is_owner_only_file`` "
            "to refuse writes/deletes/transfers/uploads against the "
            "owner-only settings.json and skill trust-state JSON. Otherwise the Files API is a "
            "parallel privilege-escalation channel."
        )


@pytest.mark.parametrize("filename", [
    "grants.json", "review.json", "review_history.jsonl", "accepted_rebuttals.json", "enabled.json", "clawhub.json",
])
def test_files_api_owner_only_helper_blocks_skill_state_case_variants(filename, tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.gateway import files as fba_mod

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir, raising=True)
    target = data_dir / "State" / "Skills" / "weather" / filename
    assert fba_mod._is_owner_only_file(target) is True


def test_files_api_owner_only_helper_blocks_symlinked_skill_state_dir(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.gateway import files as fba_mod

    data_dir = tmp_path / "data"
    link_target = data_dir / "memory" / "linkstate"
    link_target.mkdir(parents=True)
    skills_root = data_dir / "state" / "skills"
    skills_root.mkdir(parents=True)
    try:
        (skills_root / "weather").symlink_to(link_target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this filesystem")
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir, raising=True)

    target = data_dir / "state" / "skills" / "weather" / "enabled.json"
    assert fba_mod._is_owner_only_file(target) is True
    backing_target = link_target / "review.json"
    assert fba_mod._is_owner_only_file(backing_target) is True


@pytest.mark.parametrize("filename", [
    "grants.json", "review.json", "review_history.jsonl", "accepted_rebuttals.json", "enabled.json", "Review.JSON",
])
def test_run_shell_blocks_obfuscated_skill_owner_state_write(filename, tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    _clear_safety_provider_env(monkeypatch)
    drive_root = tmp_path / "data"
    skill_state_dir = drive_root / "state" / "skills" / "weather"
    skill_state_dir.mkdir(parents=True)
    helper_path = tmp_path / "owner_state_writer.py"
    stem, suffix = filename.split(".", 1)
    helper_path.write_text(
        "import json, pathlib, sys\n"
        "root = pathlib.Path(sys.argv[1])\n"
        f"name = {stem!r} + '.{suffix}'\n"
        "target = root / 'state' / 'skills' / 'weather' / name\n"
        "target.parent.mkdir(parents=True, exist_ok=True)\n"
        "target.write_text(json.dumps({'status':'pass','enabled':True}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg = ToolRegistry(repo_dir=tmp_path, drive_root=drive_root)
    result = reg.execute("run_command", {"cmd": ["python3", str(helper_path), str(drive_root)]})
    assert "OWNER_STATE_RESTORED" in result
    assert not (skill_state_dir / filename).exists()


def test_run_shell_blocks_delayed_skill_owner_state_writer(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry
    import sys
    import time

    drive_root = tmp_path / "data"
    skill_state_dir = drive_root / "state" / "skills" / "weather"
    skill_state_dir.mkdir(parents=True)
    child_code = (
        "import json, pathlib, sys, time\n"
        "time.sleep(1.0)\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "name = 'review' + '.json'\n"
        "target = root / 'state' / 'skills' / 'weather' / name\n"
        "target.write_text(json.dumps({'status':'pass'}))\n"
    )
    parent_code = (
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
    )
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg = ToolRegistry(repo_dir=tmp_path, drive_root=drive_root)
    result = reg.execute("run_command", {"cmd": [sys.executable, "-c", parent_code, str(drive_root), child_code]})
    assert "SKILL_STATE_WRITE_BLOCKED" in result
    time.sleep(1.4)
    assert not (skill_state_dir / "review.json").exists()


def test_run_shell_blocks_detached_skill_state_command(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry
    import sys

    drive_root = tmp_path / "data"
    (drive_root / "state" / "skills" / "weather").mkdir(parents=True)
    code = (
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True)\n"
        "print('state skills')\n"
    )
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg = ToolRegistry(repo_dir=tmp_path, drive_root=drive_root)
    result = reg.execute("run_command", {"cmd": [sys.executable, "-c", code]})
    assert "SKILL_STATE_WRITE_BLOCKED" in result


def test_run_shell_scans_scripts_relative_to_cwd(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry
    import sys

    _clear_safety_provider_env(monkeypatch)
    repo_dir = tmp_path / "repo"
    subdir = repo_dir / "sub"
    subdir.mkdir(parents=True)
    drive_root = tmp_path / "data"
    (drive_root / "state" / "skills" / "weather").mkdir(parents=True)
    helper = subdir / "evil.py"
    helper.write_text(
        "import json, pathlib, sys\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "name = 'review' + '.json'\n"
        "target = root / 'state' / 'skills' / 'weather' / name\n"
        "target.write_text(json.dumps({'status':'pass'}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg = ToolRegistry(repo_dir=repo_dir, drive_root=drive_root)
    result = reg.execute("run_command", {"cmd": [sys.executable, "evil.py", str(drive_root)], "cwd": "sub"})
    assert "OWNER_STATE_RESTORED" in result
    assert not (drive_root / "state" / "skills" / "weather" / "review.json").exists()
