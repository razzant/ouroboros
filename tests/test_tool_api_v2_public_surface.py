import pathlib
import re
import shlex
import sys
from types import ModuleType, SimpleNamespace

from ouroboros.tools.registry import ToolRegistry


LEGACY_PUBLIC_TOOL_NAMES = {
    "repo_read",
    "repo_write",
    "repo_list",
    "str_replace_editor",
    "data_read",
    "data_write",
    "data_list",
    "code_search",
    "run_shell",
    "git_status",
    "git_diff",
    "repo_commit",
    "restore_to_head",
    "revert_commit",
    "rollback_to_target",
    "schedule_task",
    "wait_for_task",
    "wait_for_tasks",
    "advisory_pre_review",
    "review_skill",
    "multi_model_review",
}


def test_legacy_tool_names_are_not_public_schemas(tmp_path):
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    names = {schema["function"]["name"] for schema in registry.schemas()}

    assert names.isdisjoint(LEGACY_PUBLIC_TOOL_NAMES)
    for name in LEGACY_PUBLIC_TOOL_NAMES:
        assert registry.get_schema_by_name(name) is None
        assert registry.execute(name, {}).startswith("⚠️ Unknown tool")

    assert {
        "read_file",
        "write_file",
        "search_code",
        "run_command",
        "claude_code_edit",
        "commit_reviewed",
        "schedule_subagent",
        "skill_review",
        "task_acceptance_review",
    } <= names
    for tool_name in ("read_file", "list_files", "write_file", "edit_text", "search_code"):
        schema = registry.get_schema_by_name(tool_name) or {}
        root = (((schema.get("function") or {}).get("parameters") or {}).get("properties") or {}).get("root") or {}
        assert "user_files" in set(root.get("enum") or []), tool_name


def test_list_files_schema_uses_path_not_dir(tmp_path):
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    schema = registry.get_schema_by_name("list_files") or {}
    props = (((schema.get("function") or {}).get("parameters") or {}).get("properties") or {})

    assert "path" in props
    assert "dir" not in props
    assert "TOOL_ARG_ERROR" in registry.execute("list_files", {"dir": "."})


def test_runtime_prompts_do_not_advertise_legacy_public_tool_names():
    root = pathlib.Path(__file__).resolve().parent.parent
    prompt_text = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in ("prompts/SYSTEM.md", "prompts/SAFETY.md", "prompts/CONSCIOUSNESS.md")
    )
    for name in LEGACY_PUBLIC_TOOL_NAMES:
        assert re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", prompt_text) is None, name
    for name in (
        "read_file",
        "run_command",
        "commit_reviewed",
        "advisory_review",
        "schedule_subagent",
        "task_acceptance_review",
    ):
        assert name in prompt_text
    assert "`runtime_data` for explicit runtime state/memory work" in prompt_text
    assert "`runtime_data` for read-only runtime state" not in prompt_text


def test_frozen_registry_includes_service_tools(monkeypatch, tmp_path):
    monkeypatch.setattr(__import__("sys"), "frozen", True, raising=False)
    registry = ToolRegistry(repo_dir=pathlib.Path(tmp_path), drive_root=pathlib.Path(tmp_path))
    schemas = registry.schemas()
    names = {schema["function"]["name"] for schema in schemas}
    assert {"start_service", "service_status", "service_logs", "stop_service"} <= names
    start_schema = next(schema for schema in schemas if schema["function"]["name"] == "start_service")
    assert "outputs" in start_schema["function"]["parameters"]["properties"]


def test_skill_payload_root_rejects_bucket_skill_traversal(tmp_path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    (data / "settings.json").parent.mkdir(parents=True)
    (data / "settings.json").write_text("secret", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=data)

    result = registry.execute(
        "read_file",
        {"root": "skill_payload", "bucket": "external", "skill_name": "../../settings.json", "path": "."},
    )

    assert "READ_FILE_ERROR" in result or "TOOL_ARG_ERROR" in result
    assert "secret" not in result


def test_skill_payload_write_named_bible_is_not_system_protected(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    (data / "skills" / "external" / "alpha").mkdir(parents=True)
    registry = ToolRegistry(repo_dir=repo, drive_root=data)

    result = registry.execute(
        "write_file",
        {
            "root": "skill_payload",
            "bucket": "external",
            "skill_name": "alpha",
            "path": "BIBLE.md",
            "content": "skill docs",
        },
    )

    assert result.startswith("OK:"), result
    assert (data / "skills" / "external" / "alpha" / "BIBLE.md").read_text(encoding="utf-8") == "skill docs"


def _registry_under_fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    repo = home / "Ouroboros" / "repo"
    data = home / "Ouroboros" / "data"
    desktop = home / "Desktop"
    repo.mkdir(parents=True)
    data.mkdir(parents=True)
    desktop.mkdir(parents=True)
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry._ctx.task_id = "task1"
    return registry, repo, data, desktop


def test_user_files_root_is_public_and_task_artifact_audited(tmp_path, monkeypatch):
    from ouroboros.artifacts import collect_task_artifact_records

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)

    result = registry.execute(
        "write_file",
        {"root": "user_files", "path": "Desktop/report.html", "content": "<h1>ok</h1>"},
    )

    assert result.startswith("OK:"), result
    assert "LIGHT_MODE_BLOCKED" not in result
    assert (desktop / "report.html").read_text(encoding="utf-8") == "<h1>ok</h1>"
    assert (data / "task_results" / "artifacts" / "task1" / "report.html").read_text(encoding="utf-8") == "<h1>ok</h1>"
    records = collect_task_artifact_records(data, "task1")
    assert records[0]["kind"] == "user_file"
    assert records[0]["source_path"] == str(desktop / "report.html")


def test_user_files_root_blocks_ouroboros_control_plane(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)

    result = registry.execute(
        "write_file",
        {"root": "user_files", "path": str(repo / "README.md"), "content": "bad"},
    )

    # v6.54.3 root-label hybrid: a user_files WRITE whose absolute path resolves
    # under the active workspace now gets the actionable ROOT_REQUIRED redirect
    # BEFORE the handler (the retry under root=active_workspace passes through the
    # full light-mode/protected-path discipline). Elsewhere the legacy block stays.
    # The hard security invariant is identical either way: nothing is written.
    assert (
        "ROOT_REQUIRED_ACTIVE_WORKSPACE" in result
        or ("WRITE_FILE_ERROR" in result and "user_files path blocked" in result)
    ), result
    assert not (repo / "README.md").exists()

    case_variant = pathlib.Path.home() / "ouroboros" / "repo" / "README.md"
    case_result = registry.execute(
        "write_file",
        {"root": "user_files", "path": str(case_variant), "content": "bad"},
    )

    assert (
        "ROOT_REQUIRED_ACTIVE_WORKSPACE" in case_result
        or ("WRITE_FILE_ERROR" in case_result and "user_files path blocked" in case_result)
    ), case_result
    assert not case_variant.exists()


def test_user_files_root_blocks_workspace_parent_and_home_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    home = pathlib.Path.home()

    parent_result = registry.execute(
        "write_file",
        {"root": "user_files", "path": str(home / "Ouroboros" / "AGENTS.md"), "content": "bad"},
    )
    secret_result = registry.execute(
        "read_file",
        {"root": "user_files", "path": ".ssh/id_rsa"},
    )

    assert "WRITE_FILE_ERROR" in parent_result
    assert "user_files path blocked" in parent_result
    assert "READ_FILE_ERROR" in secret_result
    assert "credential-like" in secret_result
    assert not (home / "Ouroboros" / "AGENTS.md").exists()


def test_user_files_root_blocks_case_insensitive_home_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    registry, _repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)

    library = registry.execute("read_file", {"root": "user_files", "path": "library/Keychains/login.keychain-db"})
    creds = registry.execute("write_file", {"root": "user_files", "path": "Desktop/Credentials.json", "content": "{}"})
    pem = registry.execute("write_file", {"root": "user_files", "path": "Desktop/id_rsa.PEM", "content": "secret"})

    assert "credential-like" in library
    assert "credential-like" in creds
    assert "credential-like" in pem


def test_list_files_user_files_blocks_ouroboros_control_plane(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    (repo / "README.md").write_text("secret repo", encoding="utf-8")

    result = registry.execute("list_files", {"root": "user_files", "path": "Ouroboros/repo"})

    assert "LIST_FILES_ERROR" in result
    assert "user_files path blocked" in result
    assert "README.md" not in result


def test_list_files_user_files_root_prunes_control_plane_children(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, _data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    (desktop / "visible.txt").write_text("ok", encoding="utf-8")

    result = registry.execute("list_files", {"root": "user_files", "path": "."})

    assert "Desktop/" in result
    assert "Ouroboros/" not in result
    assert "hidden/control" in result


def test_light_mode_blocks_runtime_data_as_artifact_workaround(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)

    result = registry.execute(
        "write_file",
        {"root": "runtime_data", "path": "uploads/report.html", "content": "bad"},
    )

    assert "LIGHT_MODE_BLOCKED" in result
    assert not (data / "uploads" / "report.html").exists()


def test_light_mode_blocks_process_runtime_data_upload_writes(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    upload = data / "uploads" / "report.html"

    result = registry.execute(
        "run_command",
        {
            "cmd": [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path\n"
                    f"p = Path({str(upload)!r})\n"
                    "p.parent.mkdir(parents=True, exist_ok=True)\n"
                    "p.write_text('bad')\n"
                ),
            ],
            "cwd": str(registry._ctx.task_drive_root()),
        },
    )

    assert "LIGHT_MODE_BLOCKED" in result
    assert "runtime_data" in result
    assert not upload.exists()


def test_light_mode_blocks_run_script_runtime_data_upload_writes(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    upload = data / "uploads" / "report.html"

    result = registry.execute(
        "run_script",
        {
            "script": (
                "from pathlib import Path\n"
                f"p = Path({str(upload)!r})\n"
                "p.parent.mkdir(parents=True, exist_ok=True)\n"
                "p.write_text('bad')\n"
            ),
        },
    )

    assert "LIGHT_MODE_BLOCKED" in result
    assert "runtime_data" in result
    assert not upload.exists()


def test_light_mode_blocks_relative_process_runtime_data_upload_writes(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    upload = data / "uploads" / "relative-report.html"
    task_drive = registry._ctx.task_drive_root()

    result = registry.execute(
        "run_command",
        {
            "cmd": [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path\n"
                    "p = Path('../../uploads/relative-report.html')\n"
                    "p.parent.mkdir(parents=True, exist_ok=True)\n"
                    "p.write_text('bad')\n"
                ),
            ],
            "cwd": str(task_drive),
        },
    )

    assert "LIGHT_MODE_BLOCKED" in result
    assert "runtime_data" in result
    assert not upload.exists()


def test_light_mode_blocks_env_runtime_data_upload_writes(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    upload = data / "uploads" / "env-report.html"

    result = registry.execute(
        "run_command",
        {
            "cmd": ["sh", "-c", "mkdir -p \"$OUROBOROS_DATA_DIR/uploads\" && echo bad > \"$OUROBOROS_DATA_DIR/uploads/env-report.html\""],
            "cwd": str(registry._ctx.task_drive_root()),
        },
    )

    assert "LIGHT_MODE_BLOCKED" in result
    assert "runtime_data" in result
    assert not upload.exists()


def test_light_mode_blocks_interpreter_runtime_data_touch_without_write_marker(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    upload = data / "uploads" / "touch-report.html"
    python_exe = str(pathlib.Path(sys.executable))
    if not pathlib.PurePath(python_exe).name.lower().endswith(".exe"):
        python_exe = f"{python_exe}.exe"

    result = registry.execute(
        "run_command",
        {
            "cmd": [
                python_exe,
                "-c",
                (
                    "from pathlib import Path\n"
                    f"p = Path({str(upload)!r})\n"
                    "p.parent.mkdir(parents=True, exist_ok=True)\n"
                    "p.touch()\n"
                ),
            ],
            "cwd": str(registry._ctx.task_drive_root()),
        },
    )

    assert "LIGHT_MODE_BLOCKED" in result
    assert "runtime_data" in result
    assert not upload.exists()


def test_light_mode_blocks_home_runtime_data_upload_writes(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    home_upload = data / "uploads" / "home-report.html"
    tilde_upload = data / "uploads" / "tilde-report.html"

    home_result = registry.execute(
        "run_command",
        {
            "cmd": [
                "sh",
                "-c",
                'mkdir -p "$HOME/Ouroboros/data/uploads" && touch "$HOME/Ouroboros/data/uploads/home-report.html"',
            ],
            "cwd": str(registry._ctx.task_drive_root()),
        },
    )
    tilde_result = registry.execute(
        "run_command",
        {
            "cmd": ["sh", "-c", "mkdir -p ~/Ouroboros/data/uploads && touch ~/Ouroboros/data/uploads/tilde-report.html"],
            "cwd": str(registry._ctx.task_drive_root()),
        },
    )

    assert "LIGHT_MODE_BLOCKED" in home_result
    assert "runtime_data" in home_result
    assert "LIGHT_MODE_BLOCKED" in tilde_result
    assert "runtime_data" in tilde_result
    assert not home_upload.exists()
    assert not tilde_upload.exists()


def test_light_mode_blocks_relative_run_script_runtime_data_upload_writes(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    upload = data / "uploads" / "relative-report.html"

    result = registry.execute(
        "run_script",
        {
            "script": (
                "from pathlib import Path\n"
                "p = Path('../../uploads/relative-report.html')\n"
                "p.parent.mkdir(parents=True, exist_ok=True)\n"
                "p.write_text('bad')\n"
            ),
        },
    )

    assert "LIGHT_MODE_BLOCKED" in result
    assert "runtime_data" in result
    assert not upload.exists()


def test_light_run_script_allows_readonly_repo_analysis_with_external_write(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    out = registry._ctx.task_drive_root() / "evidence.json"

    result = registry.execute(
        "run_script",
        {
            "script": (
                "from pathlib import Path\n"
                f"repo = {str(repo)!r}\n"
                f"out = Path({str(out)!r})\n"
                "out.write_text(repo)\n"
                "import sys; sys.stdout.write(repo)\n"
            ),
            "outputs": [str(out)],
        },
    )

    assert "LIGHT_MODE_BLOCKED" not in result, result
    assert out.read_text(encoding="utf-8") == str(repo)


def test_light_run_script_blocks_dynamic_repo_write_even_from_task_drive(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    target = repo / "dynamic.txt"

    result = registry.execute(
        "run_script",
        {
            "script": (
                "from pathlib import Path\n"
                f"repo = Path({str(repo)!r})\n"
                "name = 'dynamic.txt'\n"
                "(repo / name).write_text('bad')\n"
            ),
        },
    )

    assert "LIGHT_MODE_BLOCKED" in result
    assert not target.exists()


def test_light_run_script_blocks_path_open_repo_write(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    target = repo / "path-open.txt"

    result = registry.execute(
        "run_script",
        {
            "script": (
                "from pathlib import Path\n"
                f"Path({str(target)!r}).open('w').write('bad')\n"
            ),
        },
    )

    assert "LIGHT_MODE_BLOCKED" in result
    assert not target.exists()


def test_light_run_script_allows_constant_expression_task_drive_write(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    target = registry._ctx.task_drive_root() / "out.txt"

    result = registry.execute(
        "run_script",
        {
            "script": (
                "from pathlib import Path\n"
                "name = 'out' + '.txt'\n"
                "Path(name).write_text('ok')\n"
            ),
        },
    )

    assert "LIGHT_MODE_BLOCKED" not in result, result
    assert target.read_text(encoding="utf-8") == "ok"


def test_light_run_script_allows_resolved_open_handle_task_drive_write(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    target = registry._ctx.task_drive_root() / "handle.txt"

    result = registry.execute(
        "run_script",
        {
            "script": (
                "f = open('handle.txt', 'w')\n"
                "f.write('ok')\n"
                "f.close()\n"
            ),
        },
    )

    assert "LIGHT_MODE_BLOCKED" not in result, result
    assert target.read_text(encoding="utf-8") == "ok"


def test_light_run_script_allows_with_open_handle_task_drive_write(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    target = registry._ctx.task_drive_root() / "with-open.txt"

    result = registry.execute(
        "run_script",
        {"script": "with open('with-open.txt', 'w') as f:\n    f.write('ok')\n"},
    )

    assert "LIGHT_MODE_BLOCKED" not in result, result
    assert target.read_text(encoding="utf-8") == "ok"


def test_light_run_script_allows_path_cwd_task_drive_write(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    target = registry._ctx.task_drive_root() / "cwd-write.txt"

    result = registry.execute(
        "run_script",
        {
            "script": (
                "from pathlib import Path\n"
                "(Path.cwd() / 'cwd-write.txt').write_text('ok')\n"
            ),
        },
    )

    assert "LIGHT_MODE_BLOCKED" not in result, result
    assert target.read_text(encoding="utf-8") == "ok"


def test_light_run_script_allows_in_memory_write_method(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)

    result = registry.execute(
        "run_script",
        {
            "script": (
                "import io\n"
                "buf = io.StringIO()\n"
                "buf.write('ok')\n"
                "print(buf.getvalue())\n"
            ),
        },
    )

    assert "LIGHT_MODE_BLOCKED" not in result, result
    assert "ok" in result


def test_artifact_store_blocks_control_manifest_edits(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)

    write_result = registry.execute(
        "write_file",
        {"root": "artifact_store", "path": ".artifact_manifest.json", "content": "{}"},
    )
    hidden_result = registry.execute(
        "write_file",
        {"root": "artifact_store", "path": ".meta/report.json", "content": "{}"},
    )
    registry._ctx.is_direct_chat = True
    edit_result = registry.execute(
        "edit_text",
        {"root": "artifact_store", "path": ".artifact_manifest.json", "old_str": "x", "new_str": "y"},
    )

    assert "WRITE_FILE_BLOCKED" in write_result
    assert "WRITE_FILE_BLOCKED" in hidden_result
    assert "EDIT_TEXT_BLOCKED" in edit_result
    assert not (data / "task_results" / "artifacts" / "task1" / ".artifact_manifest.json").exists()


def test_write_file_batch_partial_failure_is_semantic_failure(tmp_path, monkeypatch):
    from ouroboros.loop_tool_execution import _extract_result_metadata, _is_tool_execution_failure

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)

    result = registry.execute(
        "write_file",
        {
            "root": "artifact_store",
            "files": [
                {"path": "ok.html", "content": "<p>ok</p>"},
                {"path": ".artifact_manifest.json", "content": "{}"},
            ],
        },
    )

    assert result.startswith("⚠️ WRITE_FILE_BATCH_PARTIAL_FAILURE"), result
    assert "OK:" in result
    assert "WRITE_FILE_BLOCKED" in result
    assert _is_tool_execution_failure(True, result)
    assert _extract_result_metadata("write_file", result, True)["status"] == "write_file_blocked"
    assert (data / "task_results" / "artifacts" / "task1" / "ok.html").exists()
    assert not (data / "task_results" / "artifacts" / "task1" / ".artifact_manifest.json").exists()


def test_run_script_light_default_cwd_is_task_drive_and_outputs_are_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)

    result = registry.execute(
        "run_script",
        {
            "script": "from pathlib import Path\nif 1 > 0:\n    Path('probe.txt').write_text('ok')\n",
            "outputs": ["probe.txt"],
        },
    )

    assert "LIGHT_MODE_BLOCKED" not in result
    assert "exit_code=0" in result
    assert not (repo / "probe.txt").exists()
    assert (data / "task_drives" / "task1" / "probe.txt").read_text(encoding="utf-8") == "ok"
    assert (data / "task_results" / "artifacts" / "task1" / "probe.txt").read_text(encoding="utf-8") == "ok"


def test_run_script_output_failure_starts_with_failure_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *args, **kwargs: (True, ""))
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry._ctx.task_id = "task1"
    result = registry.execute(
        "run_script",
        {"script": "print('no output created')", "outputs": ["missing.html"]},
    )

    assert result.startswith("⚠️ ARTIFACT_OUTPUT_ERROR")
    assert "# script_path=" in result


def test_run_script_light_cwd_user_files_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)

    result = registry.execute(
        "run_script",
        {
            "script": "from pathlib import Path\nPath('external.html').write_text('<p>ok</p>')\n",
            "cwd": str(desktop),
            "outputs": ["external.html"],
        },
    )

    assert "SHELL_CWD_BLOCKED" not in result
    assert "LIGHT_MODE_BLOCKED" not in result
    assert "exit_code=0" in result
    assert (desktop / "external.html").read_text(encoding="utf-8") == "<p>ok</p>"
    assert (data / "task_results" / "artifacts" / "task1" / "external.html").read_text(encoding="utf-8") == "<p>ok</p>"


def test_run_script_registers_directory_outputs_as_manifest_and_zip(tmp_path, monkeypatch):
    import json
    import zipfile

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)

    result = registry.execute(
        "run_script",
        {
            "script": (
                "from pathlib import Path\n"
                "Path('site/assets').mkdir(parents=True)\n"
                "Path('site/index.html').write_text('<h1>ok</h1>')\n"
                "Path('site/assets/app.js').write_text('console.log(1)')\n"
            ),
            "outputs": ["site"],
        },
    )

    artifact_dir = data / "task_results" / "artifacts" / "task1"
    manifests = list(artifact_dir.glob("site.*.manifest.json"))
    zips = list(artifact_dir.glob("site.*.zip"))
    assert "registered directory output" in result
    assert len(manifests) == 1
    assert len(zips) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["file_count"] == 2
    assert {item["path"] for item in manifest["files"]} == {"index.html", "assets/app.js"}
    with zipfile.ZipFile(zips[0]) as archive:
        assert sorted(archive.namelist()) == ["assets/app.js", "index.html"]


def test_run_command_light_creates_fresh_task_scoped_cwds(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry._ctx.task_id = "task1"

    task_drive = data / "task_drives" / "task1"
    artifact_store = data / "task_results" / "artifacts" / "task1"
    assert not task_drive.exists()
    assert not artifact_store.exists()

    for cwd, filename in ((artifact_store, "canonical.html"), (task_drive, "scratch.html")):
        result = registry.execute(
            "run_command",
            {
                "cmd": [sys.executable, "-c", f"from pathlib import Path; Path({filename!r}).write_text('ok')"],
                "cwd": str(cwd),
                "outputs": [filename],
            },
        )

        assert "SHELL_CWD_BLOCKED" not in result
        assert "exit_code=0" in result
        assert (cwd / filename).read_text(encoding="utf-8") == "ok"
        assert (artifact_store / filename).read_text(encoding="utf-8") == "ok"


def test_run_command_user_files_audit_gap_is_effect_based(tmp_path, monkeypatch):
    # R5: the artifact-audit nudge is now EFFECT-BASED — it fires only when a user_files
    # command actually changed the cwd, not on every command. A read-only command no
    # longer false-triggers it; a command that creates a deliverable (whose name is not a
    # literal in the cmd, so the declaration-regex does not pre-empt it) still does.
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, _data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)

    read_only = registry.execute("run_command", {"cmd": ["python3", "-c", "print('ok')"], "cwd": str(desktop)})
    assert "exit_code=0" in read_only
    assert "ARTIFACT_AUDIT_GAP" not in read_only

    creates_file = registry.execute(
        "run_command",
        {"cmd": ["python3", "-c", "open(chr(100)+'eliverable.dat','w').write('x')"], "cwd": str(desktop)},
    )
    assert "exit_code=0" in creates_file
    assert "ARTIFACT_AUDIT_GAP" in creates_file


def test_run_command_outputs_registers_artifact(tmp_path, monkeypatch):
    from ouroboros.artifacts import collect_task_artifact_records

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)

    result = registry.execute(
        "run_command",
        {
            "cmd": ["python3", "-c", "from pathlib import Path; Path('deliverable.txt').write_text('ok')"],
            "cwd": str(desktop),
            "outputs": ["deliverable.txt"],
        },
    )

    assert "exit_code=0" in result
    assert "ARTIFACT_OUTPUT_ERROR" not in result
    assert "registered output" in result
    assert (data / "task_results" / "artifacts" / "task1" / "deliverable.txt").read_text(encoding="utf-8") == "ok"
    records = collect_task_artifact_records(data, "task1")
    assert records[0]["kind"] == "process_output"
    assert records[0]["source_path"] == str(desktop / "deliverable.txt")


def test_run_command_unchanged_output_is_cosmetic(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    (desktop / "personal_notes.txt").write_text("old", encoding="utf-8")

    result = registry.execute(
        "run_command",
        {
            "cmd": ["python3", "-c", "print('no file generated')"],
            "cwd": str(desktop),
            "outputs": ["personal_notes.txt"],
        },
    )

    # C2 (v6.36.0): present-but-unchanged declared output is a cosmetic note, not a
    # blocking ARTIFACT_OUTPUT_ERROR (a missing output still blocks — sibling test).
    assert not result.startswith("⚠️ ARTIFACT_OUTPUT_ERROR"), result
    assert "unchanged output (cosmetic)" in result
    assert not (data / "task_results" / "artifacts" / "task1" / "personal_notes.txt").exists()
    # round-5: a cosmetic-only note must NOT borrow the canonical "ARTIFACT_OUTPUTS"
    # marker — downstream (outcomes.py / loop_tool_execution.py) reads that exact
    # substring as a real registration / false recovery signal. Use ARTIFACT_OUTPUT_NOTE.
    assert "ARTIFACT_OUTPUT_NOTE" in result
    assert "ARTIFACT_OUTPUTS" not in result


def test_run_command_outputs_missing_is_semantic_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, _data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)

    result = registry.execute(
        "run_command",
        {"cmd": ["python3", "-c", "print('ok')"], "cwd": str(desktop), "outputs": ["missing.txt"]},
    )

    assert result.startswith("⚠️ ARTIFACT_OUTPUT_ERROR"), result
    assert "missing output: missing.txt" in result


def test_run_command_outputs_block_protected_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    (data / "settings.json").write_text('{"secret":"x"}', encoding="utf-8")

    result = registry.execute(
        "run_command",
        {
            "cmd": ["python3", "-c", "print('ok')"],
            "cwd": str(desktop),
            "outputs": [str(data / "settings.json")],
        },
    )

    assert result.startswith("⚠️ ARTIFACT_OUTPUT_ERROR"), result
    assert "protected user_files output" in result
    assert not (data / "task_results" / "artifacts" / "task1" / "settings.json").exists()


def test_run_command_outputs_block_protected_repo_files(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    (repo / "BIBLE.md").write_text("constitutional", encoding="utf-8")

    result = registry.execute(
        "run_command",
        {"cmd": ["python3", "-c", "print('ok')"], "cwd": str(repo), "outputs": ["BIBLE.md"]},
    )

    assert result.startswith("⚠️ ARTIFACT_OUTPUT_ERROR"), result
    assert "protected repo output BIBLE.md" in result
    assert not (data / "task_results" / "artifacts" / "task1" / "BIBLE.md").exists()


def test_run_command_without_outputs_blocks_absolute_user_file_writes(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    target = desktop / "live-regression.html"

    result = registry.execute(
        "run_command",
        {
            "cmd": [
                "python3",
                "-c",
                f"from pathlib import Path; Path({str(target)!r}).write_text('<h1>ok</h1>')",
            ],
            "cwd": str(repo),
        },
    )

    assert result.startswith("⚠️ ARTIFACT_OUTPUT_ERROR"), result
    assert "without declaring outputs" in result
    assert target.read_text(encoding="utf-8") == "<h1>ok</h1>"
    assert not (data / "task_results" / "artifacts" / "task1" / target.name).exists()


def test_run_command_artifact_error_still_invalidates_repo_advisory(tmp_path, monkeypatch):
    import subprocess

    calls = []
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setattr("ouroboros.tools.shell._invalidate_advisory", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    registry, repo, _data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    target = desktop / "undeclared.html"

    result = registry.execute(
        "run_command",
        {
            "cmd": [
                "python3",
                "-c",
                (
                    "from pathlib import Path; "
                    "Path('changed.txt').write_text('repo'); "
                    f"Path({str(target)!r}).write_text('external')"
                ),
            ],
            "cwd": str(repo),
        },
    )

    assert result.startswith("⚠️ ARTIFACT_OUTPUT_ERROR"), result
    assert calls
    assert calls[-1][1]["source_tool"] == "run_command"
    assert "changed.txt" in calls[-1][1]["changed_paths"]


def test_run_command_without_outputs_allows_absolute_user_file_reads(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, _data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    target = desktop / "input.txt"
    target.write_text("hello", encoding="utf-8")

    result = registry.execute(
        "run_command",
        {
            "cmd": [
                "python3",
                "-c",
                f"open({str(target)!r}).read(); print('ok')",
            ],
            "cwd": str(repo),
        },
    )

    assert "ARTIFACT_OUTPUT_ERROR" not in result
    assert "exit_code=0" in result


def test_run_command_without_outputs_blocks_absolute_user_file_open_writes(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    target = desktop / "open-write.html"

    result = registry.execute(
        "run_command",
        {
            "cmd": [
                "python3",
                "-c",
                f"open({str(target)!r}, 'w').write('<h1>ok</h1>')",
            ],
            "cwd": str(repo),
        },
    )

    assert result.startswith("⚠️ ARTIFACT_OUTPUT_ERROR"), result
    assert "without declaring outputs" in result
    assert target.read_text(encoding="utf-8") == "<h1>ok</h1>"
    assert not (data / "task_results" / "artifacts" / "task1" / target.name).exists()


def test_run_script_without_outputs_flags_absolute_user_file_writes(tmp_path, monkeypatch):
    # v6.56.0: run_script body-audit moved to POST-exec stat verification (parity
    # with run_command), so a script that writes an undeclared absolute user_files
    # path RUNS (the write happens — a post-exec side effect can't be un-done) and
    # is then flagged ARTIFACT_OUTPUT_ERROR ("wrote"), not blocked before it runs.
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    target = desktop / "script-write.html"

    result = registry.execute(
        "run_script",
        {
            "script": f"from pathlib import Path\nPath({str(target)!r}).write_text('<h1>ok</h1>')\n",
            "interpreter": "python3",
        },
    )

    assert result.startswith("⚠️ ARTIFACT_OUTPUT_ERROR"), result
    assert "run_script wrote user_files without declaring outputs" in result
    assert str(target) in result
    # The write happened (post-exec) but the file is NOT registered as an artifact.
    assert target.read_text(encoding="utf-8") == "<h1>ok</h1>"
    assert not (data / "task_results" / "artifacts" / "task1" / target.name).exists()


def test_run_command_without_outputs_detects_shell_redirection_to_user_files(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    target = desktop / "redirect.html"

    result = registry.execute(
        "run_command",
        {"cmd": ["sh", "-c", f"echo ok > {shlex.quote(target.as_posix())}"], "cwd": str(desktop)},
    )

    assert result.startswith("⚠️ ARTIFACT_OUTPUT_ERROR"), result
    assert "without declaring outputs" in result
    assert target.read_text(encoding="utf-8").strip() == "ok"
    assert not (data / "task_results" / "artifacts" / "task1" / target.name).exists()


def test_run_command_outputs_block_preexisting_dirty_repo_secret(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    (repo / ".env").write_text("TOKEN=x", encoding="utf-8")

    result = registry.execute(
        "run_command",
        {"cmd": ["python3", "-c", "print('ok')"], "cwd": str(repo), "outputs": [".env"]},
    )

    assert result.startswith("⚠️ ARTIFACT_OUTPUT_ERROR"), result
    assert "credential-like output .env" in result
    assert not (data / "task_results" / "artifacts" / "task1" / ".env").exists()


def test_run_command_outputs_block_new_credential_like_repo_file(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    registry, repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    result = registry.execute(
        "run_command",
        {
            "cmd": ["python3", "-c", "from pathlib import Path; Path('.env').write_text('TOKEN=x')"],
            "cwd": str(repo),
            "outputs": [".env"],
        },
    )

    assert result.startswith("⚠️ ARTIFACT_OUTPUT_ERROR"), result
    assert "credential-like output .env" in result
    assert not (data / "task_results" / "artifacts" / "task1" / ".env").exists()


def test_workspace_run_command_outputs_cannot_import_user_files(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    registry, _repo, data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (desktop / "outside.txt").write_text("outside", encoding="utf-8")
    registry._ctx.workspace_mode = "workspace"
    registry._ctx.workspace_root = str(workspace)
    registry._ctx.active_repo_dir = lambda: workspace

    result = registry.execute(
        "run_command",
        {"cmd": ["python3", "-c", "print('ok')"], "cwd": str(workspace), "outputs": [str(desktop / "outside.txt")]},
    )

    assert result.startswith("⚠️ ARTIFACT_OUTPUT_ERROR"), result
    assert "output escapes allowed artifact roots" in result
    assert not (data / "task_results" / "artifacts" / "task1" / "outside.txt").exists()


def test_light_mode_allows_claude_code_edit_external_user_files_cwd(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    registry, _repo, _data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    gateway = ModuleType("ouroboros.gateways.claude_code")
    gateway.DEFAULT_CLAUDE_CODE_MAX_TURNS = 3
    gateway.resolve_claude_code_model = lambda: "claude-test"
    captured = {}

    def fake_run_edit(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            success=True,
            cost_usd=0,
            usage={},
            changed_files=[],
            diff_stat="",
            validation_summary="",
            error="",
            result_text="ok",
            to_tool_output=lambda: "OK: external edit",
        )

    gateway.run_edit = fake_run_edit
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", gateway)

    result = registry.execute("claude_code_edit", {"prompt": "edit external file", "cwd": str(desktop)})

    assert "LIGHT_MODE_BLOCKED" not in result
    assert result.startswith("OK: external edit")
    assert "ARTIFACT_AUDIT_GAP" in result
    assert pathlib.Path(captured["cwd"]) == desktop


def test_claude_code_edit_user_files_cwd_reuses_user_files_secret_guard(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    registry, _repo, _data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    gateway = ModuleType("ouroboros.gateways.claude_code")
    gateway.DEFAULT_CLAUDE_CODE_MAX_TURNS = 3
    gateway.resolve_claude_code_model = lambda: "claude-test"

    def fake_run_edit(**kwargs):
        blocker = kwargs.get("write_path_blocker")
        control_plane_case_variant = pathlib.Path.home() / "ouroboros" / "repo" / "README.md"
        assert blocker is not None
        assert "credential-like" in blocker(desktop / ".env")
        assert "credential-like" in blocker(desktop / "Credentials.json")
        assert "overlaps" in blocker(control_plane_case_variant)
        assert blocker(desktop / "report.html") == ""
        return SimpleNamespace(
            success=True,
            cost_usd=0,
            usage={},
            changed_files=[],
            diff_stat="",
            validation_summary="",
            error="",
            result_text="ok",
            to_tool_output=lambda: "OK: external edit",
        )

    gateway.run_edit = fake_run_edit
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", gateway)

    result = registry.execute("claude_code_edit", {"prompt": "edit external file", "cwd": str(desktop)})

    assert result.startswith("OK: external edit")


def test_claude_code_edit_artifact_store_cwd_reuses_artifact_control_guard(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    registry, _repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    artifact_store = pathlib.Path(registry._ctx.drive_root) / "task_results" / "artifacts" / "task1"
    gateway = ModuleType("ouroboros.gateways.claude_code")
    gateway.DEFAULT_CLAUDE_CODE_MAX_TURNS = 3
    gateway.resolve_claude_code_model = lambda: "claude-test"

    def fake_run_edit(**kwargs):
        blocker = kwargs.get("write_path_blocker")
        assert blocker is not None
        assert "reserved" in blocker(artifact_store / ".artifact_manifest.json")
        assert blocker(artifact_store / "report.html") == ""
        return SimpleNamespace(
            success=True,
            cost_usd=0,
            usage={},
            changed_files=[],
            diff_stat="",
            validation_summary="",
            error="",
            result_text="ok",
            to_tool_output=lambda: "OK: artifact edit",
        )

    gateway.run_edit = fake_run_edit
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", gateway)

    result = registry.execute("claude_code_edit", {"prompt": "edit artifact", "cwd": str(artifact_store)})

    assert result.startswith("OK: artifact edit")


def test_claude_code_edit_external_outputs_are_registered(tmp_path, monkeypatch):
    from ouroboros.artifacts import collect_task_artifact_records

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    registry, _repo, data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    gateway = ModuleType("ouroboros.gateways.claude_code")
    gateway.DEFAULT_CLAUDE_CODE_MAX_TURNS = 3
    gateway.resolve_claude_code_model = lambda: "claude-test"

    def fake_run_edit(**kwargs):
        pathlib.Path(kwargs["cwd"], "deck.html").write_text("<h1>ok</h1>", encoding="utf-8")
        return SimpleNamespace(
            success=True,
            cost_usd=0,
            usage={},
            changed_files=[],
            diff_stat="",
            validation_summary="",
            error="",
            result_text="ok",
            to_tool_output=lambda: "OK: external edit",
        )

    gateway.run_edit = fake_run_edit
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", gateway)

    result = registry.execute(
        "claude_code_edit",
        {"prompt": "create external file", "cwd": str(desktop), "outputs": ["deck.html"]},
    )

    assert result.startswith("OK: external edit")
    assert "ARTIFACT_OUTPUTS" in result
    assert (data / "task_results" / "artifacts" / "task1" / "deck.html").read_text(encoding="utf-8") == "<h1>ok</h1>"
    records = collect_task_artifact_records(data, "task1")
    assert records[0]["kind"] == "process_output"
    assert records[0]["source_path"] == str(desktop / "deck.html")


def test_claude_code_edit_outputs_require_fresh_file_change(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    registry, _repo, data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    (desktop / "deck.html").write_text("old", encoding="utf-8")
    gateway = ModuleType("ouroboros.gateways.claude_code")
    gateway.DEFAULT_CLAUDE_CODE_MAX_TURNS = 3
    gateway.resolve_claude_code_model = lambda: "claude-test"

    def fake_run_edit(**_kwargs):
        return SimpleNamespace(
            success=True,
            cost_usd=0,
            usage={},
            changed_files=[],
            diff_stat="",
            validation_summary="",
            error="",
            result_text="ok",
            to_tool_output=lambda: "OK: external edit",
        )

    gateway.run_edit = fake_run_edit
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", gateway)

    result = registry.execute("claude_code_edit", {"prompt": "create external file", "cwd": str(desktop), "outputs": ["deck.html"]})

    # C2 (v6.36.0): present-but-unchanged declared output is cosmetic, not blocking.
    assert not result.startswith("⚠️ ARTIFACT_OUTPUT_ERROR"), result
    assert "unchanged output (cosmetic)" in result
    assert not (data / "task_results" / "artifacts" / "task1" / "deck.html").exists()


def test_repeated_user_file_write_updates_same_canonical_artifact(tmp_path, monkeypatch):
    from ouroboros.artifacts import collect_task_artifact_records

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    artifact_dir = data / "task_results" / "artifacts" / "task1"
    version_dir = data / "task_results" / "artifact_versions" / "task1" / "report.html"

    first = registry.execute(
        "write_file",
        {"root": "user_files", "path": "Desktop/report.html", "content": "<p>draft</p>"},
    )
    second = registry.execute(
        "write_file",
        {"root": "user_files", "path": "Desktop/report.html", "content": "<p>final</p>", "mode": "append"},
    )

    assert "artifact_store:report.html" in first
    assert "artifact_store:report.html" in second
    assert (artifact_dir / "report.html").read_text(encoding="utf-8") == "<p>draft</p><p>final</p>"
    assert not list(artifact_dir.glob("report.*.html"))
    versions = sorted(version_dir.glob("*"))
    assert len(versions) == 1
    assert versions[0].read_text(encoding="utf-8") == "<p>draft</p>"
    records = collect_task_artifact_records(data, "task1")
    assert [record["name"] for record in records] == ["report.html"]


def test_user_file_artifact_history_retains_last_five_versions(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    version_dir = data / "task_results" / "artifact_versions" / "task1" / "report.html"

    for i in range(7):
        result = registry.execute(
            "write_file",
            {"root": "user_files", "path": "Desktop/report.html", "content": f"v{i}"},
        )
        assert "artifact_store:report.html" in result

    assert (data / "task_results" / "artifacts" / "task1" / "report.html").read_text(encoding="utf-8") == "v6"
    version_contents = [path.read_text(encoding="utf-8") for path in sorted(version_dir.glob("*"))]
    assert len(version_contents) == 5
    assert version_contents == ["v1", "v2", "v3", "v4", "v5"]


def test_merge_artifact_records_preserves_specific_kind_against_generic_collection():
    from ouroboros.artifacts import merge_artifact_records

    merged = merge_artifact_records(
        [{"kind": "verification_ledger", "name": "verification_ledger.json", "path": "/tmp/a/verification_ledger.json"}],
        [{"kind": "task_artifact", "name": "verification_ledger.json", "path": "/tmp/a/verification_ledger.json", "size": 12}],
    )

    assert merged[0]["kind"] == "verification_ledger"
    assert merged[0]["name"] == "verification_ledger.json"
    assert merged[0]["size"] == 12


def test_collect_task_artifact_records_skips_symlinks(tmp_path):
    from ouroboros.artifacts import collect_task_artifact_records, task_artifact_dir_path

    drive = tmp_path / "data"
    artifact_dir = task_artifact_dir_path(drive, "task1", create=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        (artifact_dir / "leak.txt").symlink_to(outside)
    except OSError:
        return

    records = collect_task_artifact_records(drive, "task1")

    assert records == []


def test_search_code_user_files_default_path_prunes_protected_children(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, _data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    (desktop / "notes.txt").write_text("needle", encoding="utf-8")

    result = registry.execute("search_code", {"root": "user_files", "query": "needle"})

    assert "SEARCH_ERROR" not in result
    assert "Desktop/notes.txt" in result


def test_claude_code_edit_workspace_allows_task_drive_cwd(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    registry, _repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry._ctx.workspace_mode = "workspace"
    registry._ctx.workspace_root = str(workspace)
    registry._ctx.active_repo_dir = lambda: workspace
    task_drive = registry._ctx.task_drive_root()
    task_drive.mkdir(parents=True, exist_ok=True)
    gateway = ModuleType("ouroboros.gateways.claude_code")
    gateway.DEFAULT_CLAUDE_CODE_MAX_TURNS = 3
    gateway.resolve_claude_code_model = lambda: "claude-test"
    captured = {}

    def fake_run_edit(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            success=True,
            cost_usd=0,
            usage={},
            changed_files=[],
            diff_stat="",
            validation_summary="",
            error="",
            result_text="ok",
            to_tool_output=lambda: "OK: task drive edit",
        )

    gateway.run_edit = fake_run_edit
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", gateway)

    result = registry.execute("claude_code_edit", {"prompt": "edit artifact", "cwd": str(task_drive)})

    assert result == "OK: task drive edit"
    assert pathlib.Path(captured["cwd"]) == task_drive
    assert (data / "task_results" / "artifacts" / "task1").exists() is False


def test_task_drive_root_is_scratch_even_for_forked_workspace(tmp_path, monkeypatch):
    registry, _repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    child_drive = data / "headless_tasks" / "task1" / "data"
    registry._ctx.workspace_mode = "workspace"
    registry._ctx.workspace_root = str(tmp_path / "workspace")
    registry._ctx.task_metadata["child_drive_root"] = str(child_drive)

    assert registry._ctx.task_drive_root() == data / "task_drives" / "task1"


def test_invalid_workspace_overlap_blocked_at_tool_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    registry, repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    registry._ctx.workspace_mode = "workspace"
    registry._ctx.workspace_root = str(repo)
    called = {"value": False}
    gateway = ModuleType("ouroboros.gateways.claude_code")
    gateway.DEFAULT_CLAUDE_CODE_MAX_TURNS = 3
    gateway.resolve_claude_code_model = lambda: "claude-test"

    def fake_run_edit(**_kwargs):
        called["value"] = True
        return SimpleNamespace(success=True, cost_usd=0, usage={}, changed_files=[], diff_stat="", validation_summary="", error="", result_text="", to_tool_output=lambda: "OK")

    gateway.run_edit = fake_run_edit
    monkeypatch.setitem(sys.modules, "ouroboros.gateways.claude_code", gateway)

    result = registry.execute("claude_code_edit", {"prompt": "try forged workspace", "cwd": str(repo)})

    assert "WORKSPACE_MODE_BLOCKED" in result
    assert "overlaps" in result
    assert called["value"] is False

    registry._ctx.workspace_root = str(pathlib.Path(str(repo).replace("Ouroboros", "ouroboros")))
    case_result = registry.execute("claude_code_edit", {"prompt": "try forged workspace", "cwd": str(repo)})

    assert "WORKSPACE_MODE_BLOCKED" in case_result
    assert "overlaps" in case_result


def test_system_repo_write_blocks_when_active_workspace_differs(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    active = tmp_path / "workspace"
    data = tmp_path / "data"
    repo.mkdir()
    active.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry._ctx.active_repo_dir = lambda: active
    registry._ctx.system_repo_dir = str(repo)

    result = registry.execute("write_file", {"root": "system_repo", "path": "x.txt", "content": "x"})

    assert "WRITE_FILE_BLOCKED" in result
    assert not (active / "x.txt").exists()
    assert not (repo / "x.txt").exists()


def test_light_mode_blocks_interpreter_inline_repo_writes(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)

    result = registry.execute(
        "run_script",
        {"cwd": str(repo), "script": "open('tmp_probe_no_write', 'w').write('x')"},
    )

    assert "LIGHT_MODE_BLOCKED" in result
    assert not (repo / "tmp_probe_no_write").exists()


def test_light_mode_default_root_does_not_treat_repo_skills_path_as_payload(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    (repo / "skills" / "external" / "alpha").mkdir(parents=True)
    (data / "skills" / "external" / "alpha").mkdir(parents=True)
    registry = ToolRegistry(repo_dir=repo, drive_root=data)

    result = registry.execute(
        "write_file",
        {
            "path": "skills/external/alpha/plugin.py",
            "content": "x",
        },
    )

    assert "LIGHT_MODE_BLOCKED" in result
    assert not (repo / "skills" / "external" / "alpha" / "plugin.py").exists()


def test_get_runtime_mode_prefers_boot_baseline(monkeypatch):
    from ouroboros import config as cfg

    monkeypatch.setattr(cfg, "_BOOT_RUNTIME_MODE", "light", raising=True)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")

    assert cfg.get_runtime_mode() == "light"
