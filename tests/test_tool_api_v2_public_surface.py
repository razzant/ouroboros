import pathlib
import shlex
import sys

import pytest

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

    # D10: the external coding gateway tool was retired; delegate_start is its
    # successor. The dead name must be gone from the public schema surface.
    # (Not folded into LEGACY_PUBLIC_TOOL_NAMES: docs/DEVELOPMENT.md keeps the
    # D10 retirement record that mentions the old name; the runtime prompts no
    # longer do — the prompt audit removed the retirement note from SYSTEM.md.)
    assert "claude_code_edit" not in names
    assert registry.get_schema_by_name("claude_code_edit") is None
    assert registry.execute("claude_code_edit", {}).startswith("⚠️ Unknown tool")

    # `run_ci_tests` (push, dispatch, wait) is retired with no alias; `github_read(method="checks")`
    # succeeds it for reading only (tests/test_disabled_tools_policy.py pins the absent shim and table rows).
    assert "run_ci_tests" not in names and "github_read" in names
    assert registry.get_schema_by_name("run_ci_tests") is None
    assert registry.execute("run_ci_tests", {}).startswith("⚠️ Unknown tool")

    # `task_acceptance_review` is retired with no alias: a task's result is reviewed as
    # review_change(subject="result") (owner 17.10, Q2-A).
    assert "task_acceptance_review" not in names
    assert registry.get_schema_by_name("task_acceptance_review") is None
    assert registry.execute("task_acceptance_review", {}).startswith("⚠️ Unknown tool")

    assert {
        "read_file",
        "write_file",
        "search_code",
        "run_command",
        "delegate_start",
        "commit_reviewed",
        "schedule_subagent",
        "skill_review",
        "review_change",
    } <= names
    for tool_name in ("read_file", "list_files", "write_file", "edit_text", "search_code"):
        schema = registry.get_schema_by_name(tool_name) or {}
        root = (((schema.get("function") or {}).get("parameters") or {}).get("properties") or {}).get("root") or {}
        assert "user_files" in set(root.get("enum") or []), tool_name


def test_schedule_subagent_handler_validation_is_derived_from_the_published_schema(tmp_path):
    """The handler's closed keyword set and the model-visible schema must be ONE object.

    They used to be two: a hand-written frozenset whose own comment admitted it "mirrors" the
    schema. A mirror drifts silently — add a parameter to the schema and the handler refuses it as
    "unsupported"; drop one and the handler still accepts it. Derivation makes that
    unrepresentable; this test asserts the derivation is still in force, and that adding a
    parameter to the schema really does open the handler to it (a copy would fail here)."""
    from ouroboros.tools import control

    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    schema = registry.get_schema_by_name("schedule_subagent") or {}
    parameters = ((schema.get("function") or {}).get("parameters") or {})
    published = set((parameters.get("properties") or {}).keys())

    assert published, "schedule_subagent must publish parameters"
    assert control.schedule_subagent_param_names() == published
    assert parameters.get("additionalProperties") is False
    assert set(parameters.get("required") or []) <= published
    # Internal scheduling options ride the positional-only `internal` mapping and must stay
    # OUT of the published surface (and therefore out of the derived allowed set).
    assert published.isdisjoint(control._INTERNAL_SCHEDULE_OPTIONS)

    # The schema each caller gets is a fresh object: mutating one must not poison the next.
    first = control.schedule_subagent_properties()
    first["objective"]["description"] = "mutated"
    assert control.schedule_subagent_properties()["objective"]["description"] != "mutated"


def test_list_files_schema_uses_path_not_dir(tmp_path):
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    schema = registry.get_schema_by_name("list_files") or {}
    props = (((schema.get("function") or {}).get("parameters") or {}).get("properties") or {})

    assert "path" in props
    assert "dir" not in props
    assert "TOOL_ARG_ERROR" in registry.execute("list_files", {"dir": "."})


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
    payload = data / "skills" / "external" / "alpha"
    payload.mkdir(parents=True)
    (payload / "SKILL.md").write_text("# alpha\n", encoding="utf-8")
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


def _assert_refused_before_launch(result, marker):
    """#1797: a refused output declaration answers before the process starts (owner choice A)."""
    assert result.code == "RESOURCE_POLICY_BLOCKED", result.text
    assert result.meta.get("operation_outcome") == "completed_no_effect"
    assert "refused before the process started; nothing ran" in result.text
    assert "ARTIFACT_OUTPUT_ERROR" not in result.text
    assert not marker.exists(), "the command must not have run"


_MARK = "from pathlib import Path; Path('ran.marker').write_text('ran')"


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
    from ouroboros.tools import shell

    original_env = shell._shell_env_for_cwd

    def fixture_child_env(ctx, cwd):
        return {**original_env(ctx, cwd), "HOME": str(home), "USERPROFILE": str(home),
                "OUROBOROS_DATA_DIR": str(data), "OUROBOROS_SETTINGS_PATH": str(data / "settings.json")}

    # Bind the process environment without moving the parent pytest live-data
    # guard onto this fixture's production-shaped home/Ouroboros/data tree.
    monkeypatch.setattr(shell, "_shell_env_for_cwd", fixture_child_env)
    child_env = shell._shell_env_for_cwd(registry._ctx, registry._ctx.task_drive_root())
    assert child_env["HOME"] == child_env["USERPROFILE"] == str(home)
    assert child_env["OUROBOROS_DATA_DIR"] == str(data)
    assert child_env["OUROBOROS_SETTINGS_PATH"] == str(data / "settings.json")
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


def test_user_files_root_blocks_workspace_parent_reads_home_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    home = pathlib.Path.home()
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_rsa").write_text("plain marker content", encoding="utf-8")

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
    # capinv-447 / В23=A: the ROOT principal READS credential-shaped home
    # paths (secret BYTES are masked at egress, not refused at read time).
    assert "USER_FILES_PATH_BLOCKED" not in secret_result
    assert "plain marker content" in secret_result
    assert not (home / "Ouroboros" / "AGENTS.md").exists()


def test_user_files_root_preserves_ordinary_credential_named_files(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    registry, _repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    home = pathlib.Path.home()
    keychain = home / "library" / "Keychains" / "login.keychain-db"
    keychain.parent.mkdir(parents=True)
    keychain.write_text("keychain marker", encoding="utf-8")

    # READS of credential-shaped paths are allowed for root (capinv-447 / В23=A)...
    library = registry.execute("read_file", {"root": "user_files", "path": "library/Keychains/login.keychain-db"})
    # A project filename alone is not an owner credential store.
    creds = registry.execute("write_file", {"root": "user_files", "path": "Desktop/Credentials.json", "content": "{}"})
    # A key/certificate SUFFIX is not a credential: the write lands (owner
    # answer Q6 of batch 2 - refusal authority is leaves and locations, not names).
    pem = registry.execute("write_file", {"root": "user_files", "path": "Desktop/id_rsa.PEM", "content": "secret"})

    assert "USER_FILES_PATH_BLOCKED" not in library
    assert "keychain marker" in library
    assert "ARTIFACT_OUTPUTS" in creds
    assert (home / "Desktop/Credentials.json").read_text() == "{}"
    assert "credential-like" not in pem
    assert (home / "Desktop" / "id_rsa.PEM").read_text(encoding="utf-8") == "secret"


def test_list_files_user_files_blocks_ouroboros_control_plane(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    (repo / "README.md").write_text("secret repo", encoding="utf-8")

    result = registry.execute("list_files", {"root": "user_files", "path": "Ouroboros/repo"})

    # Typed policy-denial prefix (v6.57.0 partition) — no generic LIST_FILES_ERROR wrap.
    assert "USER_FILES_PATH_BLOCKED" in result
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


def test_light_mode_executes_process_runtime_data_upload_writes(tmp_path, monkeypatch):
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

    assert "exit_code=0" in result, result
    assert upload.exists()


def test_light_mode_executes_run_script_runtime_data_upload_writes(tmp_path, monkeypatch):
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
            "cwd": "task_drive",
        },
    )

    assert "exit_code=0" in result, result
    assert upload.exists()


def test_light_mode_executes_relative_process_runtime_data_upload_writes(tmp_path, monkeypatch):
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

    assert "exit_code=0" in result, result
    assert upload.exists()


def test_light_mode_executes_env_runtime_data_upload_writes(tmp_path, monkeypatch):
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

    assert "exit_code=0" in result, result
    assert upload.read_text() == "bad\n"



def test_light_mode_executes_interpreter_runtime_data_touch_without_write_marker(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    upload = data / "uploads" / "touch-report.html"
    python_exe = str(pathlib.Path(sys.executable))

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

    assert "exit_code=0" in result, result
    assert upload.exists()


def test_light_mode_executes_home_runtime_data_upload_writes(tmp_path, monkeypatch):
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

    assert "exit_code=0" in home_result, home_result
    assert "exit_code=0" in tilde_result, tilde_result
    assert home_upload.read_bytes() == tilde_upload.read_bytes() == b""



def test_light_mode_executes_relative_run_script_runtime_data_upload_writes(tmp_path, monkeypatch):
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
            "cwd": "task_drive",
        },
    )

    assert "exit_code=0" in result, result
    assert upload.exists()


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
            "cwd": "task_drive",
            "outputs": [str(out)],
        },
    )

    assert "LIGHT_MODE_BLOCKED" not in result, result
    assert out.read_text(encoding="utf-8") == str(repo)


def test_light_run_script_executes_dynamic_repo_write_even_from_task_drive(tmp_path, monkeypatch):
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
            "cwd": "task_drive",
        },
    )

    assert "exit_code=0" in result, result
    assert target.read_text(encoding="utf-8") == "bad"


def test_light_run_script_executes_path_open_repo_write(tmp_path, monkeypatch):
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
            "cwd": "task_drive",
        },
    )

    assert "exit_code=0" in result, result
    assert target.read_text(encoding="utf-8") == "bad"


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
            "cwd": "task_drive",
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
            "cwd": "task_drive",
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
        {"script": "with open('with-open.txt', 'w') as f:\n    f.write('ok')\n", "cwd": "task_drive"},
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
            "cwd": "task_drive",
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
            "cwd": "task_drive",
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
    assert "OK: wrote artifact_store:.meta/report.json" in hidden_result
    assert (data / "task_results/artifacts/task1/.meta/report.json").read_text() == "{}"
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


def test_run_script_light_explicit_task_drive_outputs_are_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)

    result = registry.execute(
        "run_script",
        {
            "script": "from pathlib import Path\nif 1 > 0:\n    Path('probe.txt').write_text('ok')\n",
            "cwd": "task_drive",
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
            "cwd": "task_drive",
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


def test_run_command_audit_rejects_invalid_git_marker(tmp_path):
    from ouroboros.tools.shell import _resolve_git_root

    fake_repo = tmp_path / "fake-repo"
    child = fake_repo / "child"
    (fake_repo / ".git").mkdir(parents=True)
    child.mkdir()

    assert _resolve_git_root(child) is None


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
    assert "ARTIFACT_OUTPUT_UNDECLARED" not in result
    assert "registered output" in result
    assert (data / "task_results" / "artifacts" / "task1" / "deliverable.txt").read_text(encoding="utf-8") == "ok"
    records = collect_task_artifact_records(data, "task1")
    assert records[0]["kind"] == "process_output"
    assert records[0]["source_path"] == str(desktop / "deliverable.txt")


def test_run_command_unchanged_output_is_captured_once_then_cosmetic(tmp_path, monkeypatch):
    from ouroboros.artifacts import collect_task_artifact_records

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    (desktop / "personal_notes.txt").write_text("old", encoding="utf-8")
    args = {"cmd": ["python3", "-c", "print('no file generated')"], "cwd": str(desktop),
            "outputs": ["personal_notes.txt"]}

    first = registry.execute_result("run_command", args)
    repeat = registry.execute_result("run_command", args)

    # A declared output the run left unchanged is still a deliverable: its first
    # declaration registers the canonical capture of the bytes as found.
    assert first.code == "OK" and first.meta.get("artifact_registered") is True, first.text
    assert "ARTIFACT_OUTPUTS" in first.text and "(existing output, captured as found)" in first.text
    assert (data / "task_results" / "artifacts" / "task1" / "personal_notes.txt").read_text(encoding="utf-8") == "old"
    # C2 (v6.36.0): a repeat with that capture current is a cosmetic note, not a blocking
    # ARTIFACT_OUTPUT_ERROR (a missing output still blocks — sibling test). round-5: it must
    # NOT borrow the canonical "ARTIFACT_OUTPUTS" marker or the typed registration fact that
    # outcomes.py / _outcome_tool_errors.py read as a real effect or recovery.
    assert repeat.code == "OK" and repeat.meta.get("artifact_registered") is not True, repeat.text
    assert ("unchanged output (cosmetic): personal_notes.txt; current capture artifact_store:personal_notes.txt"
            in repeat.text)
    assert "ARTIFACT_OUTPUT_NOTE" in repeat.text and "ARTIFACT_OUTPUTS" not in repeat.text
    assert [row["name"] for row in collect_task_artifact_records(data, "task1")] == ["personal_notes.txt"]


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

    result = registry.execute_result(
        "run_command",
        {
            "cmd": ["python3", "-c", _MARK],
            "cwd": str(desktop),
            "outputs": [str(data / "settings.json")],
        },
    )

    _assert_refused_before_launch(result, desktop / "ran.marker")
    assert "protected user_files output" in result.text
    assert not (data / "task_results" / "artifacts" / "task1" / "settings.json").exists()


def test_run_command_outputs_block_protected_repo_files(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    (repo / "BIBLE.md").write_text("constitutional", encoding="utf-8")

    result = registry.execute_result(
        "run_command",
        {"cmd": ["python3", "-c", _MARK], "cwd": str(repo), "outputs": ["BIBLE.md"]},
    )

    _assert_refused_before_launch(result, repo / "ran.marker")
    assert "protected repo output BIBLE.md" in result.text
    assert not (data / "task_results" / "artifacts" / "task1" / "BIBLE.md").exists()


def _assert_the_note_closes_the_text(text: str) -> None:
    """The note closes a shell result's own text (a cut view keeps a result's tail); only the
    dispatcher's one-line runtime fact may follow it."""
    after = text.split("otherwise nothing is needed.", 1)[1].strip()
    assert not after or (after.startswith("Runtime selection:") and "\n" not in after), after


def test_run_command_without_outputs_notes_absolute_user_file_writes(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    target = desktop / "live-regression.html"

    typed = registry.execute_result(
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

    result = typed.text
    # A finished command with a note: no error, the note's own code, and the paths as a typed fact.
    assert (typed.status, typed.code) == ("ok", "ARTIFACT_OUTPUT_UNDECLARED")
    assert typed.meta["undeclared_outputs"] == [str(target.resolve())] and typed.meta["undeclared_outputs_total"] == 1
    assert result.startswith("exit_code=0"), result  # the command's own answer leads; the note trails it
    assert "\n\n⚠️ ARTIFACT_OUTPUT_UNDECLARED: this run names these files as write targets" in result
    assert "without being registered as task outputs" in result
    _assert_the_note_closes_the_text(result)
    assert target.read_text(encoding="utf-8") == "<h1>ok</h1>"
    assert not (data / "task_results" / "artifacts" / "task1" / target.name).exists()

    # The sensor reads names and modification times: a file modified just before a run that names it
    # as a write target in a branch it never takes is reported too, so the sentence states that
    # evidence and never says the run changed the file.
    kept = desktop / "kept.html"
    kept.write_text("before", encoding="utf-8")
    untouched = registry.execute_result("run_command", {"cwd": str(repo), "cmd": [
        "python3", "-c", f"if False:\n    open({str(kept)!r}, 'w').write('after')"]})
    assert untouched.code == "ARTIFACT_OUTPUT_UNDECLARED" and kept.read_text(encoding="utf-8") == "before"
    assert "they were modified around the time it ran" in untouched.text and "changed during" not in untouched.text


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

    assert result.startswith("exit_code=0"), result
    assert "\n\n⚠️ ARTIFACT_OUTPUT_UNDECLARED: this run names these files as write targets" in result
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
    assert "ARTIFACT_OUTPUT_UNDECLARED" not in result
    assert "exit_code=0" in result


def test_run_command_without_outputs_notes_absolute_user_file_open_writes(tmp_path, monkeypatch):
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

    assert result.startswith("exit_code=0"), result  # the command's own answer leads; the note trails it
    assert "\n\n⚠️ ARTIFACT_OUTPUT_UNDECLARED: this run names these files as write targets" in result
    assert "without being registered as task outputs" in result
    assert target.read_text(encoding="utf-8") == "<h1>ok</h1>"
    assert not (data / "task_results" / "artifacts" / "task1" / target.name).exists()


def test_undeclared_output_note_names_every_path_and_its_fact_stays_bounded(tmp_path, monkeypatch):
    """The sentence names every path, so the result itself keeps the whole list (a script that names only
    a folder leaves no other record of its files); the typed fact keeps whole paths only (a cut path is a
    false address) inside the result's encoded meta budget, with the total beside them."""
    import json

    from ouroboros.tools.shell_audit import undeclared_outputs_meta, undeclared_outputs_note

    assert undeclared_outputs_note([]) == "" and undeclared_outputs_meta([]) == {}
    paths = [f"/home/проект-{index}/" + "отчёт" * 60 + ".md" for index in range(9)]
    note = undeclared_outputs_note(paths)
    assert note.startswith("⚠️ ARTIFACT_OUTPUT_UNDECLARED: this run names these files as write targets")
    assert all(path in note for path in paths) and "Re-run" not in note
    fact = undeclared_outputs_meta(paths)
    assert fact["undeclared_outputs_total"] == 9 and 0 < len(fact["undeclared_outputs"]) < 9
    assert fact["undeclared_outputs"] == paths[:len(fact["undeclared_outputs"])]
    assert len(json.dumps(fact["undeclared_outputs"], ensure_ascii=True)) <= 2048

    # The same through run_script, whose body names only the folder: the result is the one place
    # that names each file, and its typed fact still fits.
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, _data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    folder = desktop / "reports"
    typed = registry.execute_result("run_script", {"interpreter": "python3", "script": (
        "from pathlib import Path\n"
        f"folder = Path({str(folder)!r})\nfolder.mkdir()\n"
        "for index in range(12):\n    (folder / ('отчёт' * 8 + f'-{index}.md')).write_text('x')\n")})
    written = [str(path.resolve()) for path in folder.iterdir()]
    assert typed.code == "ARTIFACT_OUTPUT_UNDECLARED" and len(written) == 12, typed.text
    assert all(path in typed.text for path in written), typed.text
    assert typed.meta["undeclared_outputs_total"] == 12 and 0 < len(typed.meta["undeclared_outputs"]) < 12
    assert set(typed.meta["undeclared_outputs"]) <= set(written)


@pytest.mark.parametrize("tool", ["run_command", "run_script"])
def test_a_run_in_a_user_files_cwd_carries_one_note_about_its_undeclared_files(tmp_path, monkeypatch, tool):
    """A named write target and a changed user_files cwd are one fact: the run carries the note that names
    the file, once, and not the cwd note beside it. With no named target the cwd note still speaks, and
    both give the same advice."""
    from ouroboros.tools.shell_audit import REGISTER_OUTPUT_ADVICE

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, _data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)

    def run(body):
        call = {"script": body, "interpreter": "python3"} if tool == "run_script" else {"cmd": ["python3", "-c", body]}
        return registry.execute_result(tool, {"cwd": str(desktop), **call})

    target = desktop / "named.html"
    named = run(f"from pathlib import Path\nPath({str(target)!r}).write_text('x')\n")
    assert named.code == "ARTIFACT_OUTPUT_UNDECLARED", named.text
    assert named.text.count("⚠️ ARTIFACT_OUTPUT_UNDECLARED") == 1 and "ARTIFACT_AUDIT_GAP" not in named.text
    assert named.meta["undeclared_outputs"] == [str(target.resolve())]
    assert REGISTER_OUTPUT_ADVICE in named.text
    _assert_the_note_closes_the_text(named.text)

    unnamed = run("open(chr(117)+'nnamed.dat','w').write('x')\n")
    assert unnamed.status == "ok" and unnamed.text.count("⚠️ ARTIFACT_AUDIT_GAP") == 1, unnamed.text
    assert "ARTIFACT_OUTPUT_UNDECLARED" not in unnamed.text and "undeclared_outputs" not in unnamed.meta
    assert REGISTER_OUTPUT_ADVICE in unnamed.text and "rerun" not in unnamed.text


@pytest.mark.parametrize("mode", ["advanced", "cyber_pro"])
def test_undeclared_output_audit_skips_what_ouroboros_owns_in_every_mode(tmp_path, monkeypatch, mode):
    """A fresh file inside Ouroboros's own data drive or program checkout is its own data, never a
    person's deliverable, in every runtime mode: under unrestricted agency the access check admits
    everything, so ownership is asked on its own. A real user file beside them is still named."""
    import time

    from ouroboros.tools.shell_audit import _mentioned_user_file_outputs_without_declaration

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    registry, repo, data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    started = time.time()
    own_data = data / "task_drives" / "task1" / "notes.txt"
    own_source = repo / "module.py"
    theirs = desktop / "report.html"
    for path in (own_data, own_source, theirs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    body = "; ".join(f"open({str(path)!r}, 'w').write('x')" for path in (own_data, own_source, theirs))

    found = _mentioned_user_file_outputs_without_declaration(
        registry._ctx, ["python3", "-c", body], None, command_start_ts=started, cwd=repo)

    assert found == [str(theirs.resolve())]


def test_run_script_without_outputs_flags_absolute_user_file_writes(tmp_path, monkeypatch):
    # v6.56.0: run_script body-audit moved to POST-exec stat verification (parity
    # with run_command), so a script that writes an undeclared absolute user_files
    # path RUNS (the write happens — a post-exec side effect can't be un-done) and
    # is then flagged ARTIFACT_OUTPUT_ERROR ("wrote"), not blocked before it runs.
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    target = desktop / "script-write.html"

    typed = registry.execute_result(
        "run_script",
        {
            "script": f"from pathlib import Path\nPath({str(target)!r}).write_text('<h1>ok</h1>')\n",
            "interpreter": "python3",
        },
    )

    result = typed.text
    assert (typed.status, typed.code) == ("ok", "ARTIFACT_OUTPUT_UNDECLARED")
    assert typed.meta["undeclared_outputs"] == [str(target.resolve())]
    assert result.startswith("# script_path="), result  # the script's own answer leads; the note trails it
    _assert_the_note_closes_the_text(result)
    assert "\n\n⚠️ ARTIFACT_OUTPUT_UNDECLARED: this run names these files as write targets" in result
    assert "without being registered as task outputs" in result
    assert str(target) in result
    # #447/D5: the nudge must not REPLACE the payload. It held line 1 (the typed
    # policy-denial surface the classifier reads) and returned nothing else, so a
    # successful script's own output was discarded. The script path and the run
    # result now ride behind the marker.
    assert "# script_path=" in result
    # The write happened (post-exec) but the file is NOT registered as an artifact.
    assert target.read_text(encoding="utf-8") == "<h1>ok</h1>"
    assert not (data / "task_results" / "artifacts" / "task1" / target.name).exists()

    # A script that wrote a file and then failed is still that failure: its marker owns line 1 and
    # its code and exit are untouched. The note follows it: the file is on disk whatever the exit says.
    half = desktop / "half-done.html"
    failed = registry.execute_result("run_script", {"interpreter": "python3", "script": (
        f"from pathlib import Path\nPath({str(half)!r}).write_text('x')\nraise SystemExit(3)\n")})
    assert (failed.code, failed.meta["exit_code"]) == ("SHELL_EXIT_ERROR", 3) and failed.status != "ok", failed.text
    assert failed.text.startswith("⚠️ SHELL_EXIT_ERROR") and failed.text.count("⚠️ ARTIFACT_OUTPUT_UNDECLARED") == 1
    assert failed.meta["undeclared_outputs"] == [str(half.resolve())]


def test_run_command_without_outputs_detects_shell_redirection_to_user_files(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    target = desktop / "redirect.html"

    result = registry.execute(
        "run_command",
        {"cmd": ["sh", "-c", f"echo ok > {shlex.quote(target.as_posix())}"], "cwd": str(desktop)},
    )

    assert result.startswith("exit_code=0"), result  # the command's own answer leads; the note trails it
    assert "\n\n⚠️ ARTIFACT_OUTPUT_UNDECLARED: this run names these files as write targets" in result
    assert "without being registered as task outputs" in result
    assert target.read_text(encoding="utf-8").strip() == "ok"
    assert not (data / "task_results" / "artifacts" / "task1" / target.name).exists()


def test_run_command_outputs_block_preexisting_dirty_repo_secret(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    (repo / ".env").write_text("TOKEN=x", encoding="utf-8")

    result = registry.execute_result(
        "run_command",
        {"cmd": ["python3", "-c", _MARK], "cwd": str(repo), "outputs": [".env"]},
    )

    _assert_refused_before_launch(result, repo / "ran.marker")
    assert "credential-like output .env" in result.text
    assert not (data / "task_results" / "artifacts" / "task1" / ".env").exists()


def test_run_command_outputs_block_new_credential_like_repo_file(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    registry, repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    result = registry.execute_result(
        "run_command",
        {
            "cmd": ["python3", "-c", "from pathlib import Path; Path('.env').write_text('TOKEN=x')"],
            "cwd": str(repo),
            "outputs": [".env"],
        },
    )

    # The credential-like name is known before the run: the secret is never even written.
    _assert_refused_before_launch(result, repo / ".env")
    assert "credential-like output .env" in result.text
    assert not (data / "task_results" / "artifacts" / "task1" / ".env").exists()


def test_workspace_run_command_outputs_can_reference_policy_visible_user_files(tmp_path, monkeypatch):
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

    assert not result.startswith("⚠️ ARTIFACT_OUTPUT_ERROR"), result
    assert "registered output" in result and "(existing output, captured as found)" in result
    assert (data / "task_results" / "artifacts" / "task1" / "outside.txt").read_text(encoding="utf-8") == "outside"


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


def _task_drive_registry(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    drive = registry._ctx.task_drive_root()
    drive.mkdir(parents=True, exist_ok=True)
    return registry, data, drive


def _write(registry, root, path, text):
    written = registry.execute("write_file", {"root": root, "path": path, "content": text})
    assert written.startswith("OK"), written


def _declare_unchanged(registry, tool, cwd, output="report.md"):
    """A successful process that verifies its declared report without writing it."""
    args = {"cwd": str(cwd), "outputs": [output]}
    if tool == "run_script":
        args.update(script="print('verified')", interpreter=sys.executable)
    else:
        args["cmd"] = [sys.executable, "-c", "print('verified')"]
    result = registry.execute_result(tool, args)
    assert result.code == "OK", result.text
    return result


def _capture_deliveries(registry, monkeypatch):
    delivered = []
    registry._ctx.current_chat_id = 11
    monkeypatch.setattr("ouroboros.tools.owner_delivery.deliver_owner_event",
                        lambda _ctx, event: delivered.append(event) or "live")
    return delivered


@pytest.mark.parametrize("tool", ["run_command", "run_script"])
def test_declared_unchanged_report_is_captured_reused_and_recaptured(tmp_path, monkeypatch, tool):
    registry, data, drive = _task_drive_registry(tmp_path, monkeypatch)
    versions = data / "task_results" / "artifact_versions" / "task1" / "report.md"

    def canonical():  # the consumer's read of the canonical deliverable
        return registry.execute("read_file", {"root": "artifact_store", "path": "report.md"})

    _write(registry, "task_drive", "report.md", "# report v1\n")
    first, repeat = _declare_unchanged(registry, tool, drive), _declare_unchanged(registry, tool, drive)
    assert first.meta.get("artifact_registered") is True and "report v1" in canonical(), first.text
    assert repeat.meta.get("artifact_registered") is not True and "ARTIFACT_OUTPUT_NOTE" in repeat.text, repeat.text
    assert "ARTIFACT_OUTPUTS" not in repeat.text  # the marker downstream reads as a registration

    _write(registry, "task_drive", "report.md", "# report v2 edited\n")  # edited before a no-op declaration
    edited = _declare_unchanged(registry, tool, drive)
    assert edited.meta.get("artifact_registered") is True and "report v2 edited" in canonical(), edited.text
    # A rewrite of the process capture keeps its previous bytes as recovery history.
    assert [path.read_text(encoding="utf-8") for path in versions.iterdir()] == ["# report v1\n"]


def test_unchanged_output_repairs_a_missing_corrupt_or_stale_capture_beside_an_immutable_delivery(tmp_path, monkeypatch):
    from ouroboros.artifacts import registered_task_artifact, stream_artifact_file

    registry, data, drive = _task_drive_registry(tmp_path, monkeypatch)
    store = data / "task_results" / "artifacts" / "task1"
    delivered = _capture_deliveries(registry, monkeypatch)

    def served():  # what an expected-row consumer serves: the row and the retained bytes agree
        row = registered_task_artifact(data, "task1", "report.md")
        assert row and not row.get("immutable") and stream_artifact_file(store / "report.md", expected=row)
        return (store / "report.md").read_text(encoding="utf-8")

    def registered():
        return _declare_unchanged(registry, "run_command", drive).meta.get("artifact_registered") is True

    _write(registry, "task_drive", "report.md", "# report v1\n")
    assert registry.execute_result("send_file", {"file_path": str(drive / "report.md")}).code == "OK"
    delivery = delivered[-1]["file_ref"]
    # The immutable delivery of this source never stands in for its missing mutable capture.
    assert registered() and served() == "# report v1\n"
    (store / "report.md").unlink()  # retained capture missing
    assert registered() and served() == "# report v1\n"
    _write(registry, "artifact_store", "report.md", "# tampered!\n")  # retained bytes changed, row stale
    assert registered() and served() == "# report v1\n"
    _write(registry, "task_drive", "report.md", "# report v2\n")
    _write(registry, "artifact_store", "report.md", "# report v2\n")  # bytes match the source, row still v1
    assert registered() and served() == "# report v2\n"
    assert not registered()  # proven current again: a cosmetic repeat
    row = registered_task_artifact(data, "task1", delivery["path"])
    assert row["immutable"] is True and row["sha256"] == delivery["sha256"]
    assert stream_artifact_file(store / delivery["path"], expected=row)["sha256"] == delivery["sha256"]


def test_unchanged_output_whose_source_changes_during_its_capture_fails_explicitly(tmp_path, monkeypatch):
    from ouroboros import artifacts

    registry, data, drive = _task_drive_registry(tmp_path, monkeypatch)
    _write(registry, "task_drive", "report.md", "# report v1\n")
    assert _declare_unchanged(registry, "run_command", drive).meta.get("artifact_registered") is True
    _write(registry, "task_drive", "report.md", "# report v2\n")  # the capture is stale: it must be refreshed
    source, original = (drive / "report.md").resolve(), artifacts.copy_artifact_file

    def changing(path, destination, *, expected=None):  # the copier's own answer to an unstable source
        if pathlib.Path(path).resolve() == source:
            raise OSError(f"artifact source changed during read: {path}")
        return original(path, destination, expected=expected)

    monkeypatch.setattr(artifacts, "copy_artifact_file", changing)
    result = registry.execute_result("run_command", {"cmd": [sys.executable, "-c", "print('verified')"],
                                                     "cwd": str(drive), "outputs": ["report.md"]})
    assert result.code == "ARTIFACT_OUTPUT_ERROR" and result.meta.get("artifact_registered") is not True, result.text
    assert "failed output copy report.md: OSError: artifact source changed during read" in result.text
    assert "unchanged output (cosmetic)" not in result.text
    stale = data / "task_results" / "artifacts" / "task1" / "report.md"
    assert stale.read_text(encoding="utf-8") == "# report v1\n"  # the earlier capture is kept whole, never torn


def test_process_and_user_file_rewrites_of_one_source_share_its_recovery_history(tmp_path, monkeypatch):
    from ouroboros.artifacts import registered_task_artifact, stream_artifact_file

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    registry, _repo, data, desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    store = data / "task_results" / "artifacts" / "task1"
    versions = data / "task_results" / "artifact_versions" / "task1" / "report.html"
    delivered = _capture_deliveries(registry, monkeypatch)

    def produce(text):
        code = f"from pathlib import Path; Path('report.html').write_text({text!r})"
        result = registry.execute_result("run_command", {"cmd": ["python3", "-c", code], "cwd": str(desktop),
                                                         "outputs": ["report.html"]})
        assert result.code == "OK" and result.meta.get("artifact_registered") is True, result.text

    _write(registry, "user_files", "Desktop/report.html", "<p>v1</p>")  # the old user_file capture
    assert registry.execute_result("send_file", {"file_path": str(desktop / "report.html")}).code == "OK"
    produce("<p>v2</p>")  # a process rewrite of that capture
    _write(registry, "user_files", "Desktop/report.html", "<p>v3</p>")  # a user_file rewrite of a process capture
    produce("<p>v4</p>")

    assert (store / "report.html").read_text(encoding="utf-8") == "<p>v4</p>"
    assert [path.read_text(encoding="utf-8") for path in sorted(versions.iterdir())] == [
        "<p>v1</p>", "<p>v2</p>", "<p>v3</p>"]
    # The earlier immutable delivery is a separate capture: never reused, rewritten or versioned.
    ref = delivered[-1]["file_ref"]
    row = registered_task_artifact(data, "task1", ref["path"])
    assert row["immutable"] is True and stream_artifact_file(store / ref["path"], expected=row)["sha256"] == ref["sha256"]
    assert (store / ref["path"]).read_text(encoding="utf-8") == "<p>v1</p>"


def _run_in_store(registry, code, output):
    return registry.execute_result("run_command", {"cmd": [sys.executable, "-c", code], "cwd": "artifact_store",
                                                   "outputs": [output]})


@pytest.mark.parametrize("origin", ["task_drive", "user_files"])
def test_store_declaration_of_a_captured_report_keeps_its_source_capture_canonical(tmp_path, monkeypatch, origin):
    from ouroboros.artifacts import collect_task_artifact_records, registered_task_artifact

    registry, data, drive = _task_drive_registry(tmp_path, monkeypatch)

    def produce(text):  # the source's own writer: a task_drive process declaration or a user_files write
        _write(registry, origin, "report.md" if origin == "task_drive" else "Desktop/report.md", text)
        if origin == "task_drive":
            assert _declare_unchanged(registry, "run_command", drive).meta.get("artifact_registered") is True

    produce("# report v1\n")
    captured = registered_task_artifact(data, "task1", "report.md")
    for _ in range(2):  # a process in the store re-verifies the deliverable it declares: the capture is its proof
        verified = _declare_unchanged(registry, "run_command", "artifact_store")
        assert verified.meta.get("artifact_registered") is not True, verified.text
        assert "ARTIFACT_OUTPUT_NOTE" in verified.text and "ARTIFACT_OUTPUTS" not in verified.text
        assert registered_task_artifact(data, "task1", "report.md") == captured
    # An in-place conversion is a distinct store result, no longer a postimage of the original source.
    converted = _run_in_store(registry, "from pathlib import Path; Path('report.md').write_text('# v1 converted\\n')",
                              "report.md")
    assert converted.code == "OK" and converted.meta.get("artifact_registered") is True, converted.text
    row = registered_task_artifact(data, "task1", "report.md")
    assert (row["kind"], row["source_path"]) == ("process_output", row["path"])
    produce("# report v2\n")  # preserve the conversion and establish one current capture for the original
    records = collect_task_artifact_records(data, "task1")
    (current,) = [item for item in records if item.get("source_path") == captured["source_path"]]
    assert "# report v2" in registry.execute("read_file", {"root": "artifact_store", "path": current["name"]})
    assert "# v1 converted" in registry.execute("read_file", {"root": "artifact_store", "path": "report.md"})
    assert len(records) == 2


@pytest.mark.parametrize("tool", ["run_command", "run_script"])
def test_declared_store_metadata_is_refused_before_launch_but_normal_outputs_work(tmp_path, monkeypatch, tool):
    from ouroboros.artifacts import task_artifact_dir_path

    registry, data, drive = _task_drive_registry(tmp_path, monkeypatch)
    _write(registry, "task_drive", "report.md", "retained")
    _declare_unchanged(registry, "run_command", drive)
    store = task_artifact_dir_path(data, "task1")
    manifest = store / ".artifact_manifest.json"
    before = manifest.read_bytes()
    code = "from pathlib import Path; Path('ran').write_text('unexpected')"
    args = {"cwd": "artifact_store", "outputs": [manifest.name]}
    args.update({"cmd": [sys.executable, "-c", code]} if tool == "run_command" else
                {"script": code, "interpreter": sys.executable})
    refused = registry.execute_result(tool, args)
    assert refused.code != "OK" and "reserved" in refused.text, refused.text
    assert not (store / "ran").exists() and manifest.read_bytes() == before
    assert _declare_unchanged(registry, tool, "artifact_store").meta.get("artifact_registered") is not True


def test_store_declaration_of_an_immutable_delivery_verifies_it_without_rewriting_it(tmp_path, monkeypatch):
    from ouroboros.artifacts import registered_task_artifact

    registry, data, drive = _task_drive_registry(tmp_path, monkeypatch)
    delivered = _capture_deliveries(registry, monkeypatch)
    _write(registry, "task_drive", "report.md", "# report v1\n")
    assert registry.execute_result("send_file", {"file_path": str(drive / "report.md")}).code == "OK"
    name = delivered[-1]["file_ref"]["path"]
    row = registered_task_artifact(data, "task1", name)
    for _ in range(2):
        verified = _declare_unchanged(registry, "run_command", "artifact_store", name)
        assert verified.meta.get("artifact_registered") is not True, verified.text
        assert "ARTIFACT_OUTPUT_NOTE" in verified.text and "ARTIFACT_OUTPUTS" not in verified.text
        assert registered_task_artifact(data, "task1", name) == row  # identity, kind and source kept
    resent = registry.execute_result("send_file", {"file_path": row["path"]})  # the copier's own self-capture
    assert resent.code == "OK" and delivered[-1]["file_ref"]["path"] == name, resent.text
    assert registered_task_artifact(data, "task1", name) == row
    # Delivered bytes a process rewrote in place are refused, never registered under the delivery.
    rewritten = _run_in_store(registry, f"from pathlib import Path; Path({name!r}).write_text('# forged\\n')", name)
    assert rewritten.code == "ARTIFACT_OUTPUT_ERROR" and rewritten.meta.get("artifact_registered") is not True
    assert f"failed output copy {name}: ArtifactIdentityError" in rewritten.text, rewritten.text
    assert registered_task_artifact(data, "task1", name) == row


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

    result = registry.execute("run_command", {"command": "echo forged-workspace-probe"})

    assert "WORKSPACE_MODE_BLOCKED" in result
    assert "overlaps" in result
    assert "forged-workspace-probe" not in result

    registry._ctx.workspace_root = str(pathlib.Path(str(repo).replace("Ouroboros", "ouroboros")))
    case_result = registry.execute("run_command", {"command": "echo forged-workspace-probe"})

    assert "WORKSPACE_MODE_BLOCKED" in case_result
    assert "overlaps" in case_result
    assert "forged-workspace-probe" not in case_result


def test_system_repo_write_targets_system_when_active_workspace_differs(tmp_path, monkeypatch):
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

    assert result.startswith("✅ Written")
    assert "system_repo:x.txt" in result
    assert not (active / "x.txt").exists()
    assert (repo / "x.txt").read_text(encoding="utf-8") == "x"


def test_light_mode_executes_interpreter_inline_repo_writes(tmp_path, monkeypatch):
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

    assert "exit_code=0" in result, result
    assert (repo / "tmp_probe_no_write").read_text(encoding="utf-8") == "x"


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


# --- submarine unwind (2026-08-08): legible confinement + typed policy denials --

def test_user_files_block_names_subagent_projects_root(tmp_path, monkeypatch):
    """A refused coop-tree path names the ONE root that can actually reach it
    (root=subagent_projects + the exact relative path) instead of four roots
    that cannot (waves 2-3 rediscovery loop). Message-only: the root stays
    read-only and the shell/write boundary is unchanged."""
    registry, _repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    home = pathlib.Path.home()
    projects = home / "Ouroboros" / "projects"
    (projects / "coop_abc123" / "render").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(projects))

    result = registry.execute(
        "list_files",
        {"root": "user_files", "path": str(projects / "coop_abc123" / "render")},
    )
    assert "USER_FILES_PATH_BLOCKED" in result
    assert "root=subagent_projects" in result
    assert "coop_abc123" in result
    # The suggested call actually works (read-only root, same profile).
    listing = registry.execute(
        "list_files", {"root": "subagent_projects", "path": "coop_abc123/render"}
    )
    assert "⚠️" not in listing


def test_user_files_read_block_status_is_policy_denial(tmp_path, monkeypatch):
    """The typed USER_FILES_PATH_BLOCKED status is partitioned as a policy denial
    (v6.57.0): an unrecovered read-side confinement refusal lands in
    execution.policy_denials, never unresolved/tool_failure."""
    from ouroboros.loop_tool_execution import _extract_result_metadata
    from ouroboros.outcomes import _POLICY_DENIAL_STATUSES, _classify_tool_errors

    meta = _extract_result_metadata(
        "list_files",
        "⚠️ USER_FILES_PATH_BLOCKED: user_files path blocked: path overlaps ...",
        True,
    )
    assert meta["status"] == "user_files_path_blocked"
    assert meta["status"] in _POLICY_DENIAL_STATUSES
    buckets = _classify_tool_errors({
        "tool_calls": [
            {"tool": "list_files", "is_error": True, "status": "user_files_path_blocked", "args": {}},
        ]
    })
    assert buckets["unresolved"] == []
    assert len(buckets["policy_denials"]) == 1


def test_artifact_output_note_is_no_failure_and_registration_failure_is():
    """The undeclared-outputs note rides a finished command: no error, its own bucket kept for
    the journal and the trace. The real 'declared output registration failed' case stays a
    genuine blocking error. A retired marker-first text reads the same way."""
    from ouroboros.loop_tool_execution import _extract_result_metadata, _is_tool_execution_failure
    from ouroboros.outcomes import _BLOCKING_TOOL_STATUSES, _POLICY_DENIAL_STATUSES
    from ouroboros.tools.tool_result import TOOL_CODE_SPECS

    spec = TOOL_CODE_SPECS["ARTIFACT_OUTPUT_UNDECLARED"]
    assert (spec.status, spec.outcome_bucket) == ("ok", "artifact_output_undeclared")
    note = (
        "⚠️ ARTIFACT_OUTPUT_UNDECLARED: command appears to write user_files outputs "
        "without declaring outputs=[...]. Paths: /x/y.txt.\n\nexit_code=0\n"
    )
    assert _is_tool_execution_failure(True, note) is False
    note_meta = _extract_result_metadata("run_command", note, False)
    assert note_meta["status"] == "artifact_output_undeclared"

    real = "⚠️ ARTIFACT_OUTPUT_ERROR: command succeeded but declared output registration failed.\nexit_code=0\n"
    assert _is_tool_execution_failure(True, real) is True
    real_meta = _extract_result_metadata("run_command", real, True)
    assert real_meta["status"] == "artifact_output_error"
    assert real_meta["status"] in _BLOCKING_TOOL_STATUSES
    assert real_meta["status"] not in _POLICY_DENIAL_STATUSES


def test_python_cwd_failure_emits_canonical_cwd_message(tmp_path, monkeypatch):
    """A python/python3 launch with a confined cwd gets the SAME self-healing
    SHELL_CWD_BLOCKED root list a non-python command gets — not an opaque
    interpreter-provenance message (submarine waves 1/3, python3 -m http.server)."""
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    registry, _repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    home = pathlib.Path.home()
    blocked_cwd = home / "Ouroboros" / "projects" / "coop_x"
    blocked_cwd.mkdir(parents=True)

    result = registry.execute(
        "run_command",
        {"cmd": ["python3", "-m", "http.server"], "cwd": str(blocked_cwd)},
    )
    assert "SHELL_CWD_BLOCKED" in result
    assert "PYTHON_INTERPRETER_UNAVAILABLE" not in result
    assert "task_drive=" in result  # label=path root list (v6.54.3)


def test_service_cwd_failure_emits_canonical_cwd_message(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    registry, _repo, _data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    home = pathlib.Path.home()
    blocked_cwd = home / "Ouroboros" / "projects" / "coop_y"
    blocked_cwd.mkdir(parents=True)

    result = registry.execute(
        "start_service",
        {"cmd": ["node", "server.js"], "name": "svc-cwd", "cwd": str(blocked_cwd)},
    )
    assert "SHELL_CWD_BLOCKED" in result
    assert "SERVICE_CWD_ERROR" not in result
    assert "task_drive=" in result


def test_preflight_review_rename_keeps_a_callable_unadvertised_alias(tmp_path):
    """Q1 rename: `preflight_review` is the advertised organ; `advisory_review`
    stays CALLABLE as a compat alias (saved prompts/memories/configs keep
    working) but never appears on the public schema surface, and a contract
    that withholds either spelling silences both."""
    from ouroboros.safety import POLICY_SKIP, TOOL_POLICY
    from ouroboros.contracts.task_contract import build_task_contract

    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    names = {schema["function"]["name"] for schema in registry.schemas()}
    assert "preflight_review" in names
    assert "advisory_review" not in names
    assert "advisory_review" not in registry.available_tools()
    # The core/heal envelope (schemas(core_only=True)) must not advertise the
    # alias either — B1 class: the default path filtered it, the core path
    # leaked it into heal-mode prompts.
    core_names = {schema["function"]["name"] for schema in registry.schemas(core_only=True)}
    assert "preflight_review" in core_names
    assert "advisory_review" not in core_names
    # The alias dispatches to the same organ (a refusal about tool identity
    # would start with "Unknown tool").
    assert not registry.execute("advisory_review", {}).startswith("⚠️ Unknown tool")
    # And the unknown-tool refusal's Available listing never advertises it.
    listing = registry.execute("definitely_not_a_tool", {})
    assert "advisory_review" not in listing
    # Policy rows exist for BOTH spellings (the alias call is policy-skipped
    # exactly like the canonical one).
    assert TOOL_POLICY.get("preflight_review") is POLICY_SKIP
    assert TOOL_POLICY.get("advisory_review") is POLICY_SKIP
    # disabled_tools compat: either spelling withholds both.
    for withheld in ("advisory_review", "preflight_review"):
        contract = build_task_contract({"description": "x", "disabled_tools": [withheld]})
        reg = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
        reg._ctx.task_metadata = {"task_contract": contract}
        listed = {schema["function"]["name"] for schema in reg.schemas()}
        assert "preflight_review" not in listed
