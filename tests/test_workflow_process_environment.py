"""The real foreground consumers share one selected environment and its egress."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

from ouroboros.settings_integrity import skill_settings
from tests._process_environment import process_context as _process_context

process_context = _process_context



@pytest.mark.parametrize(("tool", "backend"), [
    ("run_command", "host"),
    ("run_script", "local"),
    ("verify_and_record", "host"),
])
def test_registered_foreground_env_reaches_child_and_redacts_egress(process_context, monkeypatch, backend, tool):
    from ouroboros import workspace_executor
    from ouroboros.outcomes import verification_receipts_path
    from ouroboros.loop_tool_execution import _execute_single_tool

    registry, ctx, workspace, data = process_context
    if backend == "local":
        ctx.executor_ref = {"type": "local", "workspace_host_path": str(workspace), "workspace_backend_path": "/workspace"}
    from ouroboros import config
    from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot

    secret = 'private-test-"quoted"\nsecond-line'
    expected_hash = hashlib.sha256(secret.encode()).hexdigest()
    monkeypatch.setattr(config, "SETTINGS_PATH", data / "settings.json")
    config.SETTINGS_PATH.write_text(json.dumps({"CUSTOM_KEY": "old-selected-value"}), encoding="utf-8")
    snapshot = task_settings_snapshot(config.load_settings(), dict(os.environ))
    config.SETTINGS_PATH.write_text(json.dumps({"CUSTOM_KEY": secret, "OPENAI_BASE_URL": "ordinary-8080"}), encoding="utf-8")
    calls = []
    resolve = workspace_executor.resolve_process_env

    def observe(*args, **kwargs):
        calls.append(1)
        return resolve(*args, **kwargs)

    monkeypatch.setattr(workspace_executor, "resolve_process_env", observe)
    program = "import os,hashlib; print(os.environ['TOKEN']); print(os.environ['PORT']); print(hashlib.sha256(os.environ['TOKEN'].encode()).hexdigest())"
    args = {"cwd": str(workspace), "env_from_settings": {"TOKEN": "CUSTOM_KEY", "PORT": "OPENAI_BASE_URL"}}
    if tool == "run_script":
        args.update(script=program, interpreter=sys.executable)
    elif tool == "verify_and_record":
        args.update(contract_kind="explicit_command", check=[sys.executable, "-c", program], expected=expected_hash)
    else:
        args.update(cmd=[sys.executable, "-c", program])
    logs = data / "logs"
    logs.mkdir(exist_ok=True)
    with task_settings_scope(snapshot):
        _execute_single_tool(registry, {"id": "env-proof", "function": {"name": tool, "arguments": json.dumps(args)}}, logs, task_id=ctx.task_id)
        assert config.runtime_settings()["CUSTOM_KEY"] == "old-selected-value"
    records = (logs / "tools.jsonl").read_text(encoding="utf-8")
    assert expected_hash in records and "ordinary-8080" in records
    assert secret not in records and "private-test-" not in records
    assert calls == [1], "one frozen Settings selection per public invocation"
    if tool == "verify_and_record":
        receipt = json.loads(verification_receipts_path(data, ctx.task_id).read_text(encoding="utf-8").splitlines()[-1])
        assert receipt["status"] == "pass" and receipt["returncode"] == 0
        assert expected_hash in receipt["summary"]
        assert "private-test-" not in json.dumps(receipt)
        assert receipt["runtime_provenance"]["selected_path"] == sys.executable


@pytest.mark.parametrize("tool", ["run_command", "run_script", "verify_and_record", "start_service"])
def test_next_process_resolves_rotation_new_empty_and_deleted_keys(process_context, monkeypatch, tool):
    from ouroboros import config
    from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot

    registry, _ctx, workspace, data = process_context
    monkeypatch.setattr(config, "SETTINGS_PATH", data / "settings.json")
    config.SETTINGS_PATH.write_text(json.dumps({"CUSTOM_KEY": "old"}), encoding="utf-8")
    snapshot = task_settings_snapshot(config.load_settings(), dict(os.environ))
    program = "import os,hashlib; print(hashlib.sha256((os.environ['TOKEN']+'|'+os.environ['ADDED']).encode()).hexdigest(),flush=True)"
    args = {"cwd": str(workspace), "env_from_settings": {"TOKEN": "CUSTOM_KEY", "ADDED": "NEW_KEY"}}
    if tool == "run_script":
        args.update(script=program, interpreter=sys.executable)
    elif tool == "verify_and_record":
        args.update(contract_kind="explicit_command", check=[sys.executable, "-c", program])
    else:
        args.update(cmd=[sys.executable, "-c", program])
    with task_settings_scope(snapshot):
        for value in ("rotated", ""):
            config.SETTINGS_PATH.write_text(json.dumps({"CUSTOM_KEY": value, "NEW_KEY": "added"}), encoding="utf-8")
            expected = hashlib.sha256((value + "|added").encode()).hexdigest()
            if tool == "start_service":
                args.update(name="full" if value else "empty", readiness={"stdout_contains": expected, "timeout_sec": 5})
            result = registry.execute_result(tool, args)
            assert result.status == "ok", result.text
            if tool == "start_service":
                try:
                    assert expected in registry.execute("service_logs", {"name": args["name"]})
                finally:
                    registry.execute("stop_service", {"name": args["name"]})
            else:
                assert expected in result.text
            assert config.runtime_settings()["CUSTOM_KEY"] == "old"
            assert "NEW_KEY" not in config.runtime_settings()
        config.SETTINGS_PATH.write_text(json.dumps({"NEW_KEY": "added"}), encoding="utf-8")
        result = registry.execute_result(tool, args)
        assert result.status == "error" and "setting for 'TOKEN' is missing" in result.text
        assert config.runtime_settings()["CUSTOM_KEY"] == "old"


@pytest.mark.parametrize("key", ["GITHUB_TOKEN", "OPENAI_API_KEY"])
@pytest.mark.parametrize("change", ["clear", "delete"])
def test_process_refs_remove_own_mirrors_but_keep_operator_env(process_context, monkeypatch, key, change):
    from ouroboros import config
    from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot
    from ouroboros.tools.process_facts import process_environment_scope, selected_process_environment

    _registry, ctx, _workspace, data = process_context
    monkeypatch.setattr(os, "environ", dict(os.environ))  # restore the entire Settings projection after the test
    monkeypatch.setattr(config, "SETTINGS_PATH", data / "settings.json")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-operator")
    config.SETTINGS_PATH.write_text(json.dumps({key: "synthetic-mirror", "OUROBOROS_CONTEXT_MODE": "max", "OUROBOROS_CONTEXT_MODE_AUTO_LOW": False}), encoding="utf-8")
    config.apply_settings_to_env(config.load_settings())
    snapshot = task_settings_snapshot(config.load_settings(), dict(os.environ))
    values = {key: ""} if change == "clear" else {}
    config.SETTINGS_PATH.write_text(json.dumps({**values, "OUROBOROS_CONTEXT_MODE": "low", "OUROBOROS_CONTEXT_MODE_AUTO_LOW": False}), encoding="utf-8")
    args = {"env_from_settings": {"SELECTED": key, "OPERATOR": "DEEPSEEK_API_KEY"}}
    with task_settings_scope(snapshot):
        with process_environment_scope(ctx, "run_command", args) as error:
            assert error is None
            assert selected_process_environment() == {"SELECTED": "", "OPERATOR": "synthetic-operator"}
        assert skill_settings()["OUROBOROS_CONTEXT_MODE"] == "low"
        assert config.runtime_settings()["OUROBOROS_CONTEXT_MODE"] == "max"
        assert config.runtime_environ()["OUROBOROS_CONTEXT_MODE"] == "max"
        assert os.environ[key] == "synthetic-mirror"


def test_nested_process_selection_is_stable_until_next_invocation(process_context, monkeypatch):
    from ouroboros import config
    from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot
    from ouroboros.tools.process_facts import process_environment_scope, selected_process_environment

    _registry, ctx, _workspace, data = process_context
    monkeypatch.setattr(config, "SETTINGS_PATH", data / "settings.json")
    config.SETTINGS_PATH.write_text(json.dumps({"CUSTOM_KEY": "old"}), encoding="utf-8")
    snapshot = task_settings_snapshot(config.load_settings(), dict(os.environ))
    args = {"env_from_settings": {"TOKEN": "CUSTOM_KEY"}}
    with task_settings_scope(snapshot):
        with process_environment_scope(ctx, "run_command", args) as error:
            assert error is None and selected_process_environment() == {"TOKEN": "old"}
            config.SETTINGS_PATH.write_text(json.dumps({"CUSTOM_KEY": "new"}), encoding="utf-8")
            with process_environment_scope(ctx, "run_command", args) as nested_error:
                assert nested_error is None and selected_process_environment() == {"TOKEN": "old"}
        with process_environment_scope(ctx, "run_command", args) as error:
            assert error is None and selected_process_environment() == {"TOKEN": "new"}
        assert config.runtime_settings()["CUSTOM_KEY"] == "old"


@pytest.mark.parametrize("document", ["readable", "absent", "unreadable", "pin_mismatch"])
def test_explicit_process_refs_preserve_effective_route_read_rules(process_context, monkeypatch, document):
    from ouroboros import config
    from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot
    from ouroboros.tools.process_facts import process_environment_scope, selected_process_environment

    _registry, ctx, _workspace, data = process_context
    monkeypatch.setattr(config, "SETTINGS_PATH", data / "settings.json")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-operator-key")
    config.SETTINGS_PATH.write_text(json.dumps({"OUROBOROS_MODEL": "openai/entry-model"}), encoding="utf-8")
    snapshot = task_settings_snapshot(config.load_settings(), dict(os.environ))
    if document == "absent":
        config.SETTINGS_PATH.unlink()
    else:
        config.SETTINGS_PATH.write_text("[]" if document == "unreadable" else json.dumps({
            "OUROBOROS_MODEL": "openai/current-model", "OPENAI_API_KEY": "synthetic-current-key",
        }), encoding="utf-8")
    with task_settings_scope(snapshot):
        if document == "pin_mismatch":
            monkeypatch.setenv(config.SETTINGS_INTEGRITY_ENV, "0" * 64)
            with pytest.raises(config.SettingsIntegrityError):
                with process_environment_scope(ctx, "run_command", {"env_from_settings": {"TOKEN": "OPENAI_API_KEY"}}):
                    pytest.fail("a refused integrity pin must not admit the process")
            return
        with process_environment_scope(ctx, "run_command", {"env_from_settings": {"TOKEN": "OPENAI_API_KEY"}}) as error:
            if document == "unreadable":
                assert error is not None and "setting for 'TOKEN' is missing" in error.text
            else:
                expected = "synthetic-current-key" if document == "readable" else "synthetic-operator-key"
                assert error is None and selected_process_environment() == {"TOKEN": expected}
        # Explicit selection uses the current effective fallback when the overlay
        # has no document. Ordinary task reads still retain their entry fallback.
        expected_model = config.runtime_settings()["OUROBOROS_MODEL"] if document == "readable" else skill_settings()["OUROBOROS_MODEL"]
        with process_environment_scope(ctx, "run_command", {"env_from_settings": {"MODEL": "OUROBOROS_MODEL"}}) as error:
            assert error is None and selected_process_environment() == {"MODEL": expected_model}
        if document != "readable":
            assert config.runtime_settings()["OUROBOROS_MODEL"] == "openai/entry-model"


def test_verify_matches_raw_selected_value_before_receipt_redaction(process_context, monkeypatch):
    from ouroboros.outcomes import verification_receipts_path

    registry, ctx, workspace, data = process_context
    secret = "!"
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: {"CUSTOM_KEY": secret})
    result = registry.execute_result("verify_and_record", {"contract_kind": "explicit_command",
        "check": [sys.executable, "-c", "import os; print(os.environ['TOKEN'])"],
        "cwd": str(workspace), "expected": secret, "env_from_settings": {"TOKEN": "CUSTOM_KEY"}})
    receipt = json.loads(verification_receipts_path(data, ctx.task_id).read_text(encoding="utf-8").splitlines()[-1])
    assert receipt["status"] == "pass" and receipt["matched"] is True
    assert receipt["expected"] == "***" and secret not in result.text


@pytest.mark.parametrize(("returncode", "stream", "tool", "padding", "suffix"), [
    (0, "stdout", "run_command", 24982, 60000),
    (0, "stdout", "run_script", 24982, 60000),
    (0, "stdout", "verify_and_record", 3990, 60000),
    (7, "stderr", "run_command", 60000, 24982),
    (7, "stderr", "run_script", 60000, 24982),
    (7, "stderr", "verify_and_record", 19990, 60000),
])
def test_selected_secret_is_masked_before_diagnostic_bounds(
    process_context, monkeypatch, tool, padding, suffix, returncode, stream,
):
    from ouroboros.outcomes import verification_receipts_path

    registry, ctx, workspace, data = process_context
    secret = "UNIQUE_CONFIDENTIAL_FRAGMENT_0123456789_END"
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: {"CUSTOM_KEY": secret})
    program = (
        "import os,sys; "
        f"print('x'*{padding}+os.environ['TOKEN']+'y'*{suffix}, file=sys.{stream}); "
        f"sys.exit({returncode})"
    )
    args = {"cwd": str(workspace), "env_from_settings": {"TOKEN": "CUSTOM_KEY"}}
    if tool == "run_script":
        args.update(script=program, interpreter=sys.executable)
    elif tool == "verify_and_record":
        args.update(contract_kind="explicit_command", check=[sys.executable, "-c", program], expected=secret)
    else:
        args.update(cmd=[sys.executable, "-c", program])
    result = registry.execute_result(tool, args)
    rendered = result.text
    if tool == "verify_and_record":
        assert "truncated" in rendered
        receipt = json.loads(verification_receipts_path(data, ctx.task_id).read_text(encoding="utf-8").splitlines()[-1])
        assert receipt["returncode"] == returncode and receipt["matched"] is True
        assert receipt["status"] == ("fail" if returncode else "pass")
        assert len(receipt["summary"]) < 20100
        assert len(rendered) < 6000
        rendered += json.dumps(receipt)
    else:
        # The command tools return the producer's whole text; only the loop's exact-source delivery bounds
        # the view (tests/test_tool_output_source_retention.py), so the complete line is here, masked.
        assert result.meta["exit_code"] == returncode
        assert result.status == ("error" if returncode else "ok")
        assert "truncated" not in rendered
        assert "x" * padding + "***" + "y" * suffix in rendered
    assert "xxxxxxxxxx" in rendered, "ordinary output remains available"
    assert secret[:10] not in rendered and secret[-10:] not in rendered


@pytest.mark.parametrize(("matches", "match_mode"), [(True, "exact"), (True, "json_equals"), (False, "exact_line")])
def test_verify_secret_masking_preserves_strict_expected_match(
    process_context, monkeypatch, match_mode, matches,
):
    from ouroboros.outcomes import verification_receipts_path

    registry, ctx, workspace, data = process_context
    secret = "synthetic-exact-output-secret"
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: {"CUSTOM_KEY": secret})
    output = "json.dumps(os.environ['TOKEN'])" if match_mode == "json_equals" else "os.environ['TOKEN']"
    expected = secret if matches else "wrong-value"
    if match_mode == "json_equals":
        expected = json.dumps(expected)
    result = registry.execute_result("verify_and_record", {
        "contract_kind": "explicit_command", "cwd": str(workspace),
        "check": [sys.executable, "-c", f"import os,json; print({output})"],
        "expected": expected, "expected_match": match_mode,
        "env_from_settings": {"TOKEN": "CUSTOM_KEY"},
    })
    receipt = json.loads(verification_receipts_path(data, ctx.task_id).read_text(encoding="utf-8").splitlines()[-1])
    assert receipt["matched"] is matches and receipt["returncode"] == 0
    assert receipt["status"] == ("pass" if matches else "fail")
    assert secret not in result.text + json.dumps(receipt)
    assert "***" in result.text and "***" in receipt["summary"]


@pytest.mark.parametrize("failure", ["nonzero", "spawn", "timeout"])
@pytest.mark.parametrize("tool", ["run_command", "run_script", "verify_and_record"])
def test_selected_secret_does_not_escape_failed_process(process_context, monkeypatch, failure, tool):
    from ouroboros.outcomes import verification_receipts_path

    registry, ctx, workspace, data = process_context
    secret = "synthetic-failed-process-secret"
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: {"CUSTOM_KEY": secret})
    interpreter = str(workspace / secret) if failure == "spawn" else sys.executable
    program = "import os,sys,time; print(os.environ['TOKEN'],flush=True); " + ("sys.exit(7)" if failure == "nonzero" else "time.sleep(5)")
    args = {"cwd": str(workspace), "timeout_sec": 1, "env_from_settings": {"TOKEN": "CUSTOM_KEY"}}
    if tool == "run_script":
        args.update(script=program, interpreter=interpreter)
    elif tool == "verify_and_record":
        args.update(contract_kind="explicit_command", check=[interpreter, "-c", program])
    else:
        args.update(cmd=[interpreter, "-c", program])
    result = registry.execute_result(tool, args)
    assert secret not in result.text
    if tool == "verify_and_record" and failure != "spawn":
        receipt = json.loads(verification_receipts_path(data, ctx.task_id).read_text(encoding="utf-8").splitlines()[-1])
        assert receipt["status"] == "fail"
        assert receipt["returncode"] == (7 if failure == "nonzero" else None)
        assert secret not in json.dumps(receipt)
    else:
        assert result.status != "ok"
    if failure == "nonzero" and tool != "verify_and_record":
        assert result.meta["exit_code"] == 7 and "***" in result.text


def test_reference_authority_precedes_settings_read_and_spawn(process_context, monkeypatch):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.shell import _run_shell

    _registry, ctx, workspace, _data = process_context
    ctx.task_constraint = TaskConstraint(mode="acting_subagent", surface="external_workspace", write_root=str(workspace))
    monkeypatch.setattr("ouroboros.config.selection_settings", lambda **kwargs: pytest.fail("read forbidden Settings"))
    result = _run_shell(ctx, [sys.executable, "-c", "raise SystemExit('must not run')"],
        cwd=str(workspace), env_from_settings={"TOKEN": "CUSTOM_KEY"})
    assert "PROCESS_ENV_REFERENCE_BLOCKED" in result


def test_non_run_verify_refuses_environment_instead_of_ignoring_it(process_context):
    registry, _ctx, _workspace, _data = process_context
    result = registry.execute_result("verify_and_record", {"contract_kind": "artifact_observation",
        "env_from_settings": {"TOKEN": "CUSTOM_KEY"}, "artifact_paths": []})
    assert result.code == "TOOL_ARG_ERROR" and "only to run-kind" in result.text


@pytest.mark.skipif(os.name == "nt", reason="POSIX shim fixture; Windows environment merging has separate coverage")
@pytest.mark.parametrize("runtime", ["python", "node"])
def test_runtime_uses_explicit_child_path_not_host_path(process_context, monkeypatch, runtime):
    registry, _ctx, workspace, _data = process_context
    binary_dir = workspace / "chosen-bin"
    binary_dir.mkdir()
    binary = binary_dir / runtime
    if runtime == "python":
        binary.symlink_to(sys.executable)
        command = [runtime, "-c", "print('chosen-runtime')"]
    else:
        binary.write_text("#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo v22.1.0; else echo chosen-runtime; fi\n", encoding="utf-8")
        binary.chmod(0o755)
        command = [runtime, "-e", "unused"]
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: {"OPENAI_BASE_URL": str(binary_dir)})
    result = registry.execute_result("run_command", {"cmd": command, "cwd": str(workspace),
        "env_from_settings": {"PATH": "OPENAI_BASE_URL"}})
    assert result.status == "ok" and "chosen-runtime" in result.text, result.text
    if runtime == "python":
        assert str(binary) in result.text and '"source": "PATH"' in result.text


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable shims; native Windows execution is not exercised")
@pytest.mark.parametrize(("relative_dir", "backend"), [("relative-node-bin", "host"), ("", "local")])
def test_relative_selected_path_reaches_workspace_node(process_context, monkeypatch, backend, relative_dir):
    from ouroboros import process_interpreters

    registry, ctx, workspace, data = process_context
    if backend == "local":
        ctx.executor_ref = {"type": "local", "workspace_host_path": str(workspace), "workspace_backend_path": "/workspace"}
    selected_dir = workspace / relative_dir
    selected_dir.mkdir(exist_ok=True)
    bundled_dir = data / "bundle"
    bundled_dir.mkdir()
    for directory, label in ((selected_dir, "chosen-relative-node"), (bundled_dir, "incorrect-bundled-node")):
        binary = directory / "node"
        binary.write_text(
            '#!/bin/sh\nif [ "$1" = "--version" ]; then echo v22.1.0; else echo ' + label + '; fi\n',
            encoding="utf-8",
        )
        binary.chmod(0o755)
    selected_path = os.pathsep.join((relative_dir, "/usr/bin", "/bin"))
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: {"OPENAI_BASE_URL": selected_path})
    monkeypatch.setattr(process_interpreters, "resolve_bundled_node", lambda: str(bundled_dir / "node"))
    result = registry.execute_result("run_command", {"cmd": ["node", "-e", "unused"], "cwd": str(workspace),
        "env_from_settings": {"PATH": "OPENAI_BASE_URL"}})
    assert result.status == "ok" and result.meta["exit_code"] == 0, result.text
    assert "chosen-relative-node" in result.text and "incorrect-bundled-node" not in result.text
    events = [json.loads(line) for line in (data / "logs/events.jsonl").read_text(encoding="utf-8").splitlines()]
    trace = next(event for event in reversed(events) if event.get("type") == "node_runtime_resolution")
    assert trace["requested_interpreter"] == trace["resolved_interpreter"] == "node"
    assert trace["runtime_path"] == str(selected_dir / "node")
    assert trace["path_snapshot"] == selected_path and not trace["env_path_prepend"]


def test_unknown_wrapper_provenance_never_invents_a_python_path(process_context):
    registry, _ctx, workspace, _data = process_context
    if os.name == "nt":
        pytest.skip("POSIX shell fixture")
    result = registry.execute_result("run_command", {"cmd": ["sh", "-c", "printf ready"], "cwd": str(workspace)})
    assert result.status == "ok" and "runtime inside wrapper not inspected" in result.text
    assert '"selected_path": null' in result.text


@pytest.mark.parametrize("receipt", ["completed", ""])
def test_docker_target_env_uses_only_inert_host_aliases(monkeypatch, tmp_path, receipt):
    from types import SimpleNamespace
    from ouroboros import workspace_executor as executor

    seen = {}
    probes = []
    class Process:
        pid, returncode = 123, 0
        def __init__(self, command, **kwargs):
            seen.update(command=command, kwargs=kwargs)
        def communicate(self, **kwargs):
            return "done", ""
    def probe(command, **kwargs):
        assert command[:3] == ["docker", "exec", "isolated-fixture"]
        assert command[-1].endswith('= completed ] && printf completed')
        assert kwargs["timeout"] == 5 and "env" not in kwargs
        probes.append(command)
        return subprocess.CompletedProcess(command, 0, receipt, "")
    monkeypatch.setattr(executor.subprocess, "Popen", Process)
    monkeypatch.setattr(executor.subprocess, "run", probe)
    monkeypatch.setattr(executor, "_register_process", lambda *a: None)
    monkeypatch.setattr(executor, "_forget_process", lambda *a: None)
    ref = SimpleNamespace(network="default", container_name="isolated-fixture", kind="docker_exec", executor_id="fixture")
    target = {"DOCKER_HOST": "target-only", "PATH": "/target/bin", "TOKEN": "fake-$('literal')\nsecret"}
    result = executor._execute_docker(ref, ["python3", "-c", "print('done')"], "/workspace", 5, drive_root=tmp_path, target_env=target)
    assert result.returncode == 0
    assert result.operation_outcome == ("completed" if receipt == "completed" else "unknown")
    assert len(probes) == 1
    host_env = seen["kwargs"]["env"]
    assert host_env.get("DOCKER_HOST") == os.environ.get("DOCKER_HOST")
    assert host_env["PATH"] == os.environ["PATH"]
    assert all(value not in " ".join(command) for command in [seen["command"], *probes] for value in target.values())
    aliases = [seen["command"][i + 1] for i, value in enumerate(seen["command"]) if value == "--env"]
    assert len(aliases) == 3 and {host_env[key] for key in aliases} == set(target.values())
