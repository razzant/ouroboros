"""The protected black-box policy over executor artifacts and control state.

Split verbatim out of ``tests/test_tool_capabilities.py`` by theme. This
module owns the introspection fence: which paths the black-box policy
blocks, how it maps executor backend paths recursively, and the runtime
data-write block on workspace executor control state.
"""
import os
import base64
import pathlib
import sys


def test_protected_black_box_artifact_policy_blocks_introspection(tmp_path, monkeypatch):
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    if os.name == "nt":
        protected = repo / "reference.cmd"
        generated = repo / "generated.cmd"
        direct_cmd = ["cmd.exe", "/c", str(protected)]
        protected.write_text("@echo reference\r\n", encoding="utf-8")
        generated.write_text("@echo generated\r\n", encoding="utf-8")
    else:
        protected = repo / "reference.sh"
        generated = repo / "generated.sh"
        direct_cmd = [str(protected)]
        protected.write_text("#!/bin/sh\nprintf 'reference\\n'\n", encoding="utf-8")
        generated.write_text("#!/bin/sh\nprintf 'generated\\n'\n", encoding="utf-8")
    protected_dir = repo / "protected_dir"
    protected_dir.mkdir()
    (protected_dir / "secret.txt").write_text("secret\n", encoding="utf-8")
    protected.chmod(0o755)
    generated.chmod(0o755)
    task_contract = build_task_contract({
        "resource_policy": {
            "protected_artifacts": [
                {
                    "id": "reference",
                    "role": "black_box_reference",
                    "paths": [str(protected)],
                    "allow": ["execute"],
                    "deny": ["read_bytes", "copy", "hash", "static_introspection", "dynamic_trace", "debug"],
                },
                {
                    "id": "reference-dir",
                    "role": "black_box_reference",
                    "paths": [str(protected_dir)],
                    "allow": ["execute"],
                }
            ]
        }
    })
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ToolContext(
        repo_dir=repo,
        drive_root=data,
        task_contract=task_contract,
        task_metadata={"task_contract": task_contract},
    ))
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))

    direct = registry.execute("run_command", {"cmd": direct_cmd})
    assert "RESOURCE_POLICY_BLOCKED" not in direct
    assert "reference" in direct
    assert "RESOURCE_POLICY_BLOCKED" in registry.execute("read_file", {"path": protected.name})
    protected_content = protected.read_text(encoding="utf-8")
    write_attempt = registry.execute("write_file", {"path": protected.name, "content": "tamper\n"})
    assert "RESOURCE_POLICY_BLOCKED" in write_attempt
    assert protected.read_text(encoding="utf-8") == protected_content
    edit_attempt = registry.execute("edit_text", {"path": protected.name, "old_str": "reference", "new_str": "tamper"})
    assert "RESOURCE_POLICY_BLOCKED" in edit_attempt
    assert protected.read_text(encoding="utf-8") == protected_content
    shell_write_attempt = registry.execute(
        "run_command",
        {"cmd": ["sh", "-c", f"printf tamper > {protected.name}"], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in shell_write_attempt
    assert protected.read_text(encoding="utf-8") == protected_content
    shell_delete_attempt = registry.execute(
        "run_command",
        {"cmd": ["rm", protected.name], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in shell_delete_attempt
    assert protected.exists()
    recursive_delete_attempt = registry.execute(
        "run_command",
        {"cmd": ["rm", "-rf", "."], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in recursive_delete_attempt
    assert protected.exists()
    glob_delete_attempt = registry.execute(
        "run_command",
        {"cmd": ["sh", "-c", "rm -rf *"], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in glob_delete_attempt
    assert protected.exists()
    glob_read_attempt = registry.execute(
        "run_command",
        {"cmd": ["sh", "-c", "cat *"], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in glob_read_attempt
    find_exec_read = registry.execute(
        "run_command",
        {"cmd": ["find", ".", "-type", "f", "-exec", "cat", "{}", "+"], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in find_exec_read
    find_delete = registry.execute(
        "run_command",
        {"cmd": ["find", ".", "-delete"], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in find_delete
    assert protected.exists()
    pathless_find_exec_read = registry.execute(
        "run_command",
        {"cmd": ["find", "-type", "f", "-exec", "cat", "{}", "+"], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in pathless_find_exec_read
    pathless_find_delete = registry.execute(
        "run_command",
        {"cmd": ["find", "-delete"], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in pathless_find_delete
    assert protected.exists()
    safe_interpreter = registry.execute(
        "run_command",
        {
            "cmd": [
                sys.executable,
                "-c",
                "print(1)",
            ],
            "cwd": str(repo),
        },
    )
    assert "RESOURCE_POLICY_BLOCKED" not in safe_interpreter
    assert "1" in safe_interpreter
    assert "RESOURCE_POLICY_BLOCKED" in registry.execute("list_files", {"path": protected_dir.name})
    interpreter_read = registry.execute(
        "run_command",
        {
            "cmd": [
                sys.executable,
                "-c",
                f"from pathlib import Path; print(Path(r'{protected}').read_bytes())",
            ]
        },
    )
    assert "RESOURCE_POLICY_BLOCKED" in interpreter_read
    relative_interpreter_read = registry.execute(
        "run_command",
        {
            "cmd": [
                sys.executable,
                "-c",
                f"from pathlib import Path; print(Path({protected.name!r}).read_bytes())",
            ],
            "cwd": str(repo),
        },
    )
    assert "RESOURCE_POLICY_BLOCKED" in relative_interpreter_read
    versioned_interpreter_read = registry.execute(
        "run_command",
        {
            "cmd": [
                "python3.12",
                "-c",
                f"from pathlib import Path; print(Path({protected.name!r}).read_bytes())",
            ],
            "cwd": str(repo),
        },
    )
    assert "RESOURCE_POLICY_BLOCKED" in versioned_interpreter_read
    constructed_path_read = registry.execute(
        "run_command",
        {
            "cmd": [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"print((Path(r'{protected.parent}') / ({protected.stem!r} + {protected.suffix!r})).read_bytes())"
                ),
            ]
        },
    )
    assert "RESOURCE_POLICY_BLOCKED" in constructed_path_read
    backslash_parent = str(protected.parent).replace("/", "\\")
    backslash_constructed_path_read = registry.execute(
        "run_command",
        {
            "cmd": [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"print((Path({backslash_parent!r}) / ({protected.stem!r} + {protected.suffix!r})).read_bytes())"
                ),
            ]
        },
    )
    assert "RESOURCE_POLICY_BLOCKED" in backslash_constructed_path_read
    env_assignment_read = registry.execute(
        "run_command",
        {
            "cmd": [
                f"REF={protected}",
                sys.executable,
                "-c",
                "import os; print(open(os.environ['REF'], 'rb').read())",
            ]
        },
    )
    assert "RESOURCE_POLICY_BLOCKED" in env_assignment_read
    shell_script_read = registry.execute("run_command", {"cmd": ["sh", str(protected)]})
    assert "RESOURCE_POLICY_BLOCKED" in shell_script_read
    for cmd in (
        ["cmd.exe", "/c", "type", protected.name],
        ["cmd.exe", "/c", "copy", protected.name, str(repo / "copy.cmd")],
        ["cmd.exe", "/c", "xcopy", protected.name, str(repo / "copy-dir")],
        ["powershell.exe", "-Command", "Get-Content", protected.name],
        ["powershell.exe", "-Command", "Select-String", "reference", protected.name],
        ["powershell.exe", "-Command", "Copy-Item", protected.name, str(repo / "copy.ps1")],
        ["pwsh", "-Command", "Get-FileHash", protected.name],
        ["cmd.exe", "/c", "certutil", "-hashfile", protected.name],
    ):
        result = registry.execute("run_command", {"cmd": cmd, "cwd": str(repo)})
        assert "RESOURCE_POLICY_BLOCKED" in result, cmd
    encoded_read = base64.b64encode(f"Get-Content {protected.name}".encode("utf-16le")).decode("ascii")
    for cmd in (
        ["powershell.exe", "-EncodedCommand", encoded_read],
        ["pwsh", "-enc", encoded_read],
    ):
        result = registry.execute("run_command", {"cmd": cmd, "cwd": str(repo)})
        assert "RESOURCE_POLICY_BLOCKED" in result, cmd
    search_direct = registry.execute("search_code", {"query": "reference", "path": protected.name})
    assert "RESOURCE_POLICY_BLOCKED" in search_direct
    search_protected_dir = registry.execute("search_code", {"query": "secret", "path": protected_dir.name})
    assert "RESOURCE_POLICY_BLOCKED" in search_protected_dir
    query_protected = registry.execute("query_code", {"op": "structural", "query": "reference", "path": protected.name})
    assert "RESOURCE_POLICY_BLOCKED" not in query_protected
    assert protected.name not in query_protected
    for cache in (data / "state" / "code_intel").glob("*/inventory.json"):
        assert protected.name not in cache.read_text(encoding="utf-8")
    grep_read = registry.execute("run_command", {"cmd": ["grep", "reference", str(protected)]})
    assert "RESOURCE_POLICY_BLOCKED" in grep_read
    grep_recursive = registry.execute("run_command", {"cmd": ["grep", "-R", "reference", "."], "cwd": str(repo)})
    assert "RESOURCE_POLICY_BLOCKED" in grep_recursive
    rg_read = registry.execute("run_command", {"cmd": ["rg", "reference", str(protected)]})
    assert "RESOURCE_POLICY_BLOCKED" in rg_read
    rg_recursive = registry.execute("run_command", {"cmd": ["rg", "reference", "."], "cwd": str(repo)})
    assert "RESOURCE_POLICY_BLOCKED" in rg_recursive
    copy_recursive = registry.execute("run_command", {"cmd": ["cp", "-R", ".", str(repo / "copy")], "cwd": str(repo)})
    assert "RESOURCE_POLICY_BLOCKED" in copy_recursive
    for cmd in (
        ["git", "diff", "--", protected.name],
        ["git", "diff"],
        ["git", "show", f"HEAD:{protected.name}"],
        ["git", "show", "HEAD"],
        ["git", "grep", "reference", "--", protected.name],
        ["git", "grep", "reference"],
        ["git", "cat-file", "-p", f"HEAD:{protected.name}"],
        ["git", "log", "-p", "--", protected.name],
        ["git", "log", "-p"],
    ):
        result = registry.execute("run_command", {"cmd": cmd, "cwd": str(repo)})
        assert "RESOURCE_POLICY_BLOCKED" in result, cmd
    assert "RESOURCE_POLICY_BLOCKED" in registry.execute("vcs_diff", {"path": protected.name})
    assert "RESOURCE_POLICY_BLOCKED" in registry.execute("vcs_diff", {})
    import ouroboros.code_intelligence as code_intelligence

    original_file_fact = code_intelligence._file_fact

    def guarded_file_fact(repo_root, path):
        assert pathlib.Path(path).resolve(strict=False) != protected.resolve(strict=False)
        return original_file_fact(repo_root, path)

    monkeypatch.setattr(code_intelligence, "_file_fact", guarded_file_fact)
    digest = registry.execute("query_code", {"op": "digest"})
    assert protected.name not in digest
    assert generated.name in digest
    run_output_export = registry.execute("run_command", {"cmd": direct_cmd, "outputs": [protected.name], "cwd": str(repo)})
    assert "ARTIFACT_OUTPUT_ERROR" in run_output_export
    assert "RESOURCE_POLICY_BLOCKED" in run_output_export
    script_output_export = registry.execute(
        "run_script",
        {"interpreter": "python3", "script": "print('ok')", "outputs": [protected.name], "cwd": str(repo)},
    )
    assert "RESOURCE_POLICY_BLOCKED" in script_output_export
    service_cmd = ["cmd.exe", "/c", "ping", "127.0.0.1", "-n", "30"] if os.name == "nt" else ["sleep", "30"]
    service_start = registry.execute(
        "start_service",
        {
            "name": "protected-output",
            "cmd": service_cmd,
            "cwd": str(repo),
            "outputs": [protected.name],
        },
    )
    assert "protected-output" in service_start
    service_stop = registry.execute("stop_service", {"name": "protected-output"})
    assert "ARTIFACT_OUTPUT_ERROR" in service_stop
    assert "RESOURCE_POLICY_BLOCKED" in service_stop
    for cmd in (
        ["strings", str(protected)],
        ["objdump", "-d", str(protected)],
        ["cat", str(protected)],
        ["sha256sum", str(protected)],
        ["strace", str(protected)],
        ["gdb", str(protected)],
        ["lldb", str(protected)],
        ["cp", str(protected), str(repo / "copy.sh")],
        ["dd", f"if={protected}", f"of={repo / 'copy2.sh'}"],
        ["tar", "-czf", str(repo / "out.tgz"), protected.name],
        ["tar", "-czf", str(repo / "tree.tgz"), "."],
        ["zip", str(repo / "out.zip"), protected.name],
        ["rsync", protected.name, str(repo / "copy.sh")],
    ):
        result = registry.execute("run_command", {"cmd": cmd, "cwd": str(repo)})
        assert "RESOURCE_POLICY_BLOCKED" in result, cmd

    generated_result = registry.execute("run_command", {"cmd": ["strings", str(generated)]})
    assert "RESOURCE_POLICY_BLOCKED" not in generated_result


def test_protected_black_box_recursive_policy_maps_executor_backend_paths(tmp_path, monkeypatch):
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for path in (system_repo, workspace, data):
        path.mkdir(parents=True, exist_ok=True)
    protected = workspace / "executable"
    protected.write_text("reference bytes\n", encoding="utf-8")
    task_contract = build_task_contract({
        "resource_policy": {
            "protected_artifacts": [
                {
                    "id": "reference",
                    "role": "black_box_reference",
                    "paths": ["/workspace/executable"],
                    "allow": ["execute"],
                    "deny": ["read_bytes", "copy", "hash", "static_introspection", "dynamic_trace", "debug"],
                }
            ]
        }
    })
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(
        ToolContext(
            repo_dir=system_repo,
            drive_root=data,
            workspace_root=workspace,
            workspace_mode="external",
            task_contract=task_contract,
            task_metadata={"task_contract": task_contract},
            executor_ref={
                "type": "docker_exec",
                "id": "pb-container",
                "container_name": "pb-container",
                "network": "none",
                "workspace_host_path": str(workspace),
                "workspace_backend_path": "/workspace",
            },
        )
    )
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))

    grep_recursive = registry.execute("run_command", {"cmd": ["grep", "-R", "reference", "."], "cwd": str(workspace)})
    copy_recursive = registry.execute("run_command", {"cmd": ["cp", "-R", ".", str(workspace / "copy")], "cwd": str(workspace)})
    import ouroboros.code_intelligence as code_intelligence

    original_file_fact = code_intelligence._file_fact

    def guarded_file_fact(repo_root, path):
        assert pathlib.Path(path).resolve(strict=False) != protected.resolve(strict=False)
        return original_file_fact(repo_root, path)

    monkeypatch.setattr(code_intelligence, "_file_fact", guarded_file_fact)
    digest = registry.execute("query_code", {"op": "digest"})

    assert "RESOURCE_POLICY_BLOCKED" in grep_recursive
    assert "RESOURCE_POLICY_BLOCKED" in copy_recursive
    assert "executable" not in digest


def test_runtime_data_write_blocks_workspace_executor_control_state(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    state_dir = data / "state" / "workspace_executor_processes"
    state_dir.mkdir(parents=True)
    existing = state_dir / "foreground-forged.json"
    existing.write_text("original", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ToolContext(repo_dir=repo, drive_root=data))
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))

    direct_write = registry.execute(
        "write_file",
        {
            "root": "runtime_data",
            "path": "state/workspace_executor_processes/foreground-forged.json",
            "content": "{}",
        },
    )
    assert "DATA_WRITE_BLOCKED" in direct_write
    assert existing.read_text(encoding="utf-8") == "original"

    nested_write = registry.execute(
        "write_file",
        {
            "root": "runtime_data",
            "path": "state/headless_tasks/child/data/state/workspace_executor_processes/foreground-forged.json",
            "content": "{}",
        },
    )
    assert "DATA_WRITE_BLOCKED" in nested_write

    edit = registry.execute(
        "edit_text",
        {
            "root": "runtime_data",
            "path": "state/workspace_executor_processes/foreground-forged.json",
            "old_str": "original",
            "new_str": "tampered",
        },
    )
    assert "EDIT_TEXT_BLOCKED" in edit
    assert existing.read_text(encoding="utf-8") == "original"

    shell_write = registry.execute(
        "run_command",
        {
            "cmd": [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"Path(r'{existing}').write_text('{{\"owner\":\"ouroboros_workspace_executor\"}}')"
                ),
            ],
        },
    )
    assert "WORKSPACE_EXECUTOR_STATE_WRITE_BLOCKED" in shell_write
    assert existing.read_text(encoding="utf-8") == "original"

    node_eval_write = registry.execute(
        "run_command",
        {
            "cmd": [
                "node",
                "-e",
                f"require('fs').writeFileSync({str(existing)!r}, '{{}}')",
            ],
        },
    )
    assert "WORKSPACE_EXECUTOR_STATE_WRITE_BLOCKED" in node_eval_write
    assert existing.read_text(encoding="utf-8") == "original"
