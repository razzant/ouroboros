"""Ordinary named project files survive real access, staging and capture paths."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import pytest

from ouroboros import artifacts, config
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.headless import SCRATCH_MANIFEST_NAME, write_workspace_patch_artifacts
from ouroboros.tool_access import resource_root_path
from ouroboros.tools.registry import ToolContext, ToolRegistry


pytestmark = pytest.mark.serial


@pytest.fixture
def files(tmp_path, monkeypatch):
    home, system, data = (tmp_path / name for name in ("home", "system", "data"))
    work = home / "project"
    for path in (home, system, data, work):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    ctx = ToolContext(repo_dir=system, drive_root=data, workspace_root=work,
                      workspace_mode="external", task_id="agency-files")
    registry = ToolRegistry(repo_dir=system, drive_root=data)
    registry.set_context(ctx)
    return registry, ctx, home, work, data


def test_cyber_nonexternal_root_reads_exact_outside_home_and_skill_state(files, tmp_path, monkeypatch):
    registry, ctx, _home, _work, data = files
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "cyber_pro")
    ctx.workspace_mode = ""
    ctx.workspace_root = None
    outside = tmp_path / "outside.txt"
    outside.write_text("outside exact bytes")
    state = data / "state/skills/example/grants.json"
    state.parent.mkdir(parents=True)
    state.write_text('{"ordinary":"observed grant state"}')
    for args, expected in [
        ({"root": "user_files", "path": str(outside)}, "outside exact bytes"),
        ({"root": "runtime_data", "path": "state/skills/example/grants.json"}, "observed grant state"),
    ]:
        result = registry.execute_result("read_file", args)
        assert result.status == "ok" and expected in result.text, result.text


@pytest.mark.parametrize("mode,profile", [
    ("advanced", "local_readonly_subagent"), ("advanced", "acting_subagent"),
    ("cyber_pro", "local_readonly_subagent"), ("cyber_pro", "acting_subagent"),
])
def test_child_reads_lists_and_searches_ordinary_credential_named_files(files, monkeypatch, mode, profile):
    registry, ctx, _home, work, _data = files
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", mode)
    ctx.task_constraint = TaskConstraint(mode=profile, surface="external_workspace" if profile == "acting_subagent" else "")
    names = ("tokens.json", "credentials.json", "secret.json", "report.key", "public.pem")
    for index, name in enumerate(names):
        content = f"AGENCY_FILE_{index}: ordinary example data\n"
        if name == "public.pem":
            content = "-----BEGIN CERTIFICATE-----\n" + content + "-----END CERTIFICATE-----\n"
        (work / name).write_text(content, encoding="utf-8")
        result = registry.execute_result("read_file", {"root": "active_workspace", "path": name})
        assert result.status == "ok" and f"AGENCY_FILE_{index}" in result.text, result.text
    listing = registry.execute_result("list_files", {"root": "active_workspace", "path": "."})
    assert listing.status == "ok" and all(name in listing.text for name in names), listing.text
    search = registry.execute_result("search_code", {"root": "active_workspace", "query": "AGENCY_FILE_", "path": "."})
    assert search.status == "ok" and all(name in search.text for name in names), search.text


def test_ordinary_child_reads_parent_visible_owner_files_and_dotenv(files, monkeypatch):
    registry, ctx, _home, work, data = files
    # The selected workspace contains the fixture HOME. These are readable
    # by the parent; credential-write and staging policy stay independent.
    monkeypatch.setattr(Path, "home", lambda: work)
    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent")
    owner_key = work / ".ssh" / "id_rsa"
    owner_key.parent.mkdir()
    owner_key.write_text("OWNER_LOCATION_CANARY", encoding="utf-8")
    alias = work / "innocent.txt"
    try:
        alias.symlink_to(owner_key)
    except OSError:
        pytest.skip("symlinks unavailable")
    (data / "settings.json").write_text('{"value":"OWNER_LOCATION_CANARY"}', encoding="utf-8")
    (work / ".env").write_text("VALUE=OWNER_LOCATION_CANARY", encoding="utf-8")
    for root, path in (("active_workspace", ".ssh/id_rsa"), ("active_workspace", "innocent.txt"),
                       ("active_workspace", ".env"),
                       ("runtime_data", "settings.json")):
        result = registry.execute_result("read_file", {"root": root, "path": path})
        assert result.status == "ok" and "OWNER_LOCATION_CANARY" in result.text, result.text
    assert owner_key.read_text() == "OWNER_LOCATION_CANARY"


def test_root_writes_ordinary_dot_artifacts_but_not_authority(files):
    registry, ctx, _home, _work, _data = files
    store = resource_root_path(ctx, "artifact_store")
    for relative in (".well-known/security.txt", ".nojekyll"):
        result = registry.execute_result("write_file", {"root": "artifact_store", "path": relative, "content": "PUBLIC_ARTIFACT"})
        assert result.status == "ok", result.text
        assert (store / relative).read_bytes() == b"PUBLIC_ARTIFACT"
        assert "PUBLIC_ARTIFACT" in registry.execute("read_file", {"root": "artifact_store", "path": relative})
    for relative in (".artifact_manifest.json", SCRATCH_MANIFEST_NAME, "verification_receipts.jsonl"):
        target = store / relative
        before = target.read_bytes() if target.exists() else None
        result = registry.execute_result("write_file", {"root": "artifact_store", "path": relative, "content": "forged"})
        assert result.status == "blocked", result.text
        assert (target.read_bytes() if target.exists() else None) == before


def test_declared_outputs_are_created_once_and_captured_with_exact_bytes(files):
    registry, ctx, _home, _work, data = files
    payloads = {"report.log": "Log deliverable\n", "build.manifest": "Manifest deliverable\n", "tokens.json": '{"count":7}\n'}
    body = "from pathlib import Path\n"
    body += "for name, text in " + repr(payloads) + ".items():\n Path(name).write_bytes(text.encode('utf-8'))\n"
    body += "with Path('executions.txt').open('a') as out: out.write('once\\n')\n"
    result = registry.execute_result("run_command", {"cmd": [sys.executable, "-c", body],
        "cwd": "task_drive", "outputs": list(payloads)})
    assert result.status == "ok", result.text
    source = resource_root_path(ctx, "task_drive")
    store = resource_root_path(ctx, "artifact_store").resolve()
    assert source.resolve().is_relative_to(data.resolve()) and store.is_relative_to(data.resolve())
    assert (source / "executions.txt").read_text() == "once\n"
    captured = {row["name"]: row for row in artifacts.collect_task_artifact_records(data, ctx.task_id)}
    for name, content in payloads.items():
        row, expected = captured[name], content.encode()
        assert Path(row["path"]).resolve().is_relative_to(store)
        assert (source / name).read_bytes() == Path(row["path"]).read_bytes() == expected
        assert row["sha256"] == sha256(expected).hexdigest() and row["size"] == len(expected)
        assert row["status"] == "ready" and row["kind"] == "process_output"


def test_workspace_patch_keeps_log_manifest_and_token_report(files, tmp_path):
    _registry, _ctx, _home, work, _data = files
    for args in (("init",), ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                             "commit", "--allow-empty", "-m", "fixture baseline")):
        subprocess.run(["git", *args], cwd=work, check=True, capture_output=True)
    payloads = {"report.log": "LOG_PAYLOAD\n", "build.manifest": "MANIFEST_PAYLOAD\n", "tokens.json": '{"report":7}\n'}
    for name, content in payloads.items():
        (work / name).write_bytes(content.encode("utf-8"))
    output = tmp_path / "captured"
    _records, manifest = write_workspace_patch_artifacts(work, output, task={})
    assert set(payloads) <= set(manifest["untracked_included"]), manifest
    patch = (output / "workspace.patch").read_text()
    restored = tmp_path / "restored"
    restored.mkdir()
    # Assert captured bytes independently of the receiver's newline preference.
    subprocess.run(["git", "-c", "core.autocrlf=false", "apply", str(output / "workspace.patch")],
                   cwd=restored, check=True, capture_output=True)
    for name, content in payloads.items():
        assert f"b/{name}" in patch and content.strip() in patch
        assert (restored / name).read_bytes() == (work / name).read_bytes() == content.encode()


def test_attachment_staging_uses_source_location_and_retains_missing_bytes(files, monkeypatch):
    registry, ctx, home, work, data = files
    ordinary = work / "credentials.json"
    ordinary.write_bytes(b'{"example":true}')
    owner_key = home / ".ssh" / "id_rsa"
    owner_key.parent.mkdir()
    owner_key.write_bytes(b"OWNER_LOCATION_CANARY")
    requests = [{"path": str(path)} for path in (ordinary, owner_key, work / "missing.pem")]
    rows = artifacts.stage_task_attachments(data, ctx.task_id, requests)
    assert [row["status"] for row in rows] == ["staged", "rejected", "rejected"], rows
    assert rows[2]["reason"] == "source_missing"
    staged = Path(rows[0]["abs_path"])
    assert staged.is_relative_to(resource_root_path(ctx, "artifact_store"))
    assert staged.read_bytes() == ordinary.read_bytes() and rows[0]["sha256"] == sha256(ordinary.read_bytes()).hexdigest()
    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent")
    assert '"example":true' in registry.execute("read_file", {"root": "artifact_store", "path": rows[0]["relpath"]})
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "cyber_pro")
    cyber = artifacts.stage_task_attachments(data, "cyber-input", requests[1:])
    assert [row["status"] for row in cyber] == ["staged", "rejected"], cyber
    assert Path(cyber[0]["abs_path"]).read_bytes() == owner_key.read_bytes()


def test_cyber_root_can_write_owner_config_and_explicit_readonly_cannot(files, monkeypatch):
    registry, ctx, home, _work, _data = files
    target = home / ".aws" / "config"
    target.parent.mkdir()
    target.write_text("original", encoding="utf-8")
    args = {"root": "user_files", "path": str(target), "content": "changed"}
    ordinary = registry.execute_result("write_file", args)
    assert ordinary.status == "blocked" and target.read_text() == "original", ordinary.text
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "cyber_pro")
    changed = registry.execute_result("write_file", args)
    assert changed.status == "ok" and target.read_text() == "changed", changed.text
    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent")
    denied = registry.execute_result("write_file", {**args, "content": "readonly mutation"})
    assert denied.status == "blocked" and target.read_text() == "changed", denied.text
