"""Headless CLI surface: ouroboros run/watch/wait/patch and bench adapters.

The task/workspace halves of this suite were split verbatim into
``tests/test_headless_task_api.py``, ``tests/test_headless_task_events.py``,
``tests/test_headless_workspace_shell.py``,
``tests/test_headless_workspace_patch.py`` and
``tests/test_headless_task_artifacts.py``; what remains is the command-line
entry point itself plus the benchmark adapters that drive it.
"""
from __future__ import annotations

import json
import importlib.util
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros.utils import utc_now_iso


from tests._headless_cli_shared import (  # noqa: F401  (autouse fixture applies on import)
    _managed_worker_pool_available,
)


def test_cli_patch_downloads_http_artifact():
    from ouroboros.cli import _patch_from_result

    class FakeClient:
        def __init__(self):
            self.paths = []

        def get_bytes(self, path):
            self.paths.append(path)
            return b"diff --git a/a b/a\n"

    client = FakeClient()
    result = {"artifact_status": "ready", "artifacts": [{"kind": "workspace_patch", "name": "workspace.patch"}]}

    assert _patch_from_result(client, "task-1", result, strict=True).startswith("diff --git")
    assert client.paths == ["/api/tasks/task-1/artifacts/workspace.patch"]


def test_cli_patch_falls_back_to_workspace_patch_name():
    from ouroboros.cli import _patch_from_result

    class FakeClient:
        def __init__(self):
            self.paths = []

        def get_bytes(self, path):
            self.paths.append(path)
            return b"diff --git a/a b/a\n"

    client = FakeClient()
    result = {"artifact_status": "ready", "artifacts": [{"kind": "task_artifact", "name": "workspace.patch"}]}

    assert _patch_from_result(client, "task-1", result, strict=True).startswith("diff --git")
    assert client.paths == ["/api/tasks/task-1/artifacts/workspace.patch"]


def test_cli_patch_strict_rejects_empty_artifact():
    from ouroboros.cli import PatchCLIError, _patch_from_result

    class FakeClient:
        def get_bytes(self, path):
            return b""

    result = {"artifact_status": "ready", "artifacts": [{"kind": "workspace_patch", "name": "workspace.patch"}]}
    with pytest.raises(PatchCLIError, match="empty"):
        _patch_from_result(FakeClient(), "task-1", result, strict=True)


def test_cli_terminal_success_uses_outcome_axes():
    from ouroboros.cli import _is_terminal_success

    base = {
        "status": "completed",
        "artifact_status": "ready",
        "outcome_axes": {
            "execution": {"status": "ok"},
            "objective": {"status": "not_evaluated"},
        },
    }
    assert _is_terminal_success(base) is True

    failed_objective = {
        **base,
        "outcome_axes": {
            "execution": {"status": "ok"},
            "objective": {"status": "fail", "source": "task_acceptance_review"},
        },
    }
    assert _is_terminal_success(failed_objective) is False

    degraded_execution = {
        **base,
        "outcome_axes": {
            "execution": {"status": "degraded"},
            "objective": {"status": "not_evaluated"},
        },
    }
    assert _is_terminal_success(degraded_execution) is False

    # Pin the documented best_effort contract: a forced-finalization best-effort
    # completion is NOT clean terminal success (CLI strict modes must not treat
    # it as a clean pass)...
    best_effort_execution = {
        **base,
        "outcome_axes": {
            "execution": {"status": "best_effort"},
            "objective": {"status": "not_evaluated"},
        },
    }
    assert _is_terminal_success(best_effort_execution) is False


def test_cli_has_no_file_or_review_commit_groups():
    from ouroboros.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["run", "hello"]).command == "run"
    with pytest.raises(SystemExit):
        parser.parse_args(["files"])
    with pytest.raises(SystemExit):
        parser.parse_args(["commit"])
    with pytest.raises(SystemExit):
        parser.parse_args(["review"])
    with pytest.raises(SystemExit):
        parser.parse_args(["skills", "review", "demo"])


def test_source_server_start_is_blocked_in_packaged_cli_env(monkeypatch):
    from ouroboros import cli

    monkeypatch.setenv("OUROBOROS_PACKAGED_CLI", "1")
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("direct server start"))

    with pytest.raises(cli.CLIError, match="packaged CLI must launch the desktop app"):
        cli._start_local_server("http://127.0.0.1:8765")


def test_packaged_cli_run_start_scan_skips_timeout_value():
    from ouroboros.packaged_cli import _run_start_index

    assert _run_start_index(["run", "--timeout", "5", "--start", "hello"], 0) == 3


def test_cli_run_no_stream_waits_without_jsonl(monkeypatch, capsys):
    from ouroboros import cli

    class FakeClient:
        def request(self, method, path, body=None):
            assert method == "POST"
            assert path == "/api/tasks"
            return {"task_id": "abc123"}

    monkeypatch.setattr(cli, "_client", lambda args, start=False: FakeClient())
    monkeypatch.setattr(cli, "_wait_task", lambda client, task_id, timeout_sec: {"status": "completed", "result": "done"})

    assert cli.main(["run", "--no-stream", "hello"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "done"


def test_cli_run_timeout_waits_through_finalization_grace(monkeypatch):
    from ouroboros import cli
    from supervisor import queue

    captured = {}

    class FakeClient:
        def request(self, method, path, body=None):
            return {"task_id": "abc123"}

    def fake_wait(_client, _task_id, timeout_sec):
        captured["timeout_sec"] = timeout_sec
        return {"status": "completed", "result": "done"}

    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "2")
    queue.init(pathlib.Path("/tmp/ouroboros-test-data"))
    assert queue.FINALIZATION_GRACE_SEC == 2
    monkeypatch.setattr(cli, "_client", lambda args, start=False: FakeClient())
    monkeypatch.setattr(cli, "_wait_task", fake_wait)

    assert cli.main(["run", "--no-stream", "--timeout", "7", "hello"]) == 0
    assert captured["timeout_sec"] == 14.0
    monkeypatch.delenv("OUROBOROS_FINALIZATION_GRACE_SEC", raising=False)
    assert cli._deadline_wait_timeout(7) == 132.0


def test_cli_run_detach_prints_task_id_without_waiting(monkeypatch, capsys):
    from ouroboros import cli

    class FakeClient:
        def request(self, method, path, body=None):
            assert method == "POST"
            assert path == "/api/tasks"
            return {"task_id": "abc123"}

    monkeypatch.setattr(cli, "_client", lambda args, start=False: FakeClient())
    monkeypatch.setattr(cli, "_watch_task", lambda *args, **kwargs: pytest.fail("detach should not watch"))
    monkeypatch.setattr(cli, "_wait_task", lambda *args, **kwargs: pytest.fail("detach should not wait"))

    assert cli.main(["run", "--detach", "hello"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "abc123"


def test_cli_run_actor_id_is_sent_as_gateway_root_field(monkeypatch, capsys):
    from ouroboros import cli

    captured = {}

    class FakeClient:
        def request(self, method, path, body=None):
            captured["method"] = method
            captured["path"] = path
            captured["body"] = body
            return {"task_id": "abc123"}

    monkeypatch.setattr(cli, "_client", lambda args, start=False: FakeClient())
    monkeypatch.setattr(cli, "_watch_task", lambda *args, **kwargs: pytest.fail("detach should not watch"))

    assert cli.main(["run", "--detach", "--timeout", "7", "--actor-id", "operator-1", "hello"]) == 0
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/tasks"
    assert captured["body"]["description"] == "hello"
    assert "text" not in captured["body"]
    assert "prompt" not in captured["body"]
    assert captured["body"]["actor_id"] == "operator-1"
    assert captured["body"]["timeout_sec"] == 7.0
    assert captured["body"]["source"] == "cli"
    assert captured["body"]["metadata"]["source"] == "cli"
    assert "actor_id" not in captured["body"]["metadata"]
    assert capsys.readouterr().out.strip() == "abc123"


def test_cli_run_disable_tools_sent_as_gateway_root_field(monkeypatch, capsys):
    from ouroboros import cli

    captured = {}

    class FakeClient:
        def request(self, method, path, body=None):
            captured["method"] = method
            captured["path"] = path
            captured["body"] = body
            return {"task_id": "abc123"}

    monkeypatch.setattr(cli, "_client", lambda args, start=False: FakeClient())
    monkeypatch.setattr(cli, "_watch_task", lambda *args, **kwargs: pytest.fail("detach should not watch"))

    assert cli.main([
        "run", "--detach",
        "--disable-tools", "web_search,browse_page",
        "--disable-tools", "claude_code_edit",
        "hello",
    ]) == 0
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/tasks"
    assert captured["body"]["disabled_tools"] == ["web_search", "browse_page", "claude_code_edit"]
    assert capsys.readouterr().out.strip() == "abc123"


def test_cli_run_task_metadata_json_merges_but_cannot_forge_service_keys(monkeypatch, capsys):
    """--task-metadata-json (v6.56.0) merges user metadata (e.g. budget_profile)
    into body.metadata, but the host-owned delegation_role/source keys are spread
    last and can never be overridden by the user JSON (subagent forgery)."""
    from ouroboros import cli

    captured = {}

    class FakeClient:
        def request(self, method, path, body=None):
            captured["body"] = body
            return {"task_id": "abc123"}

    monkeypatch.setattr(cli, "_client", lambda args, start=False: FakeClient())
    monkeypatch.setattr(cli, "_watch_task", lambda *args, **kwargs: pytest.fail("detach should not watch"))

    payload = (
        '{"budget_profile": {"improvement_policy": "adaptive", "cost_hard_stop_pct": 0},'
        ' "delegation_role": "subagent", "source": "forged"}'
    )
    assert cli.main(["run", "--detach", "--task-metadata-json", payload, "hello"]) == 0
    metadata = captured["body"]["metadata"]
    assert metadata["budget_profile"] == {
        "improvement_policy": "adaptive",
        "cost_hard_stop_pct": 0,
    }
    assert metadata["delegation_role"] == "root"
    assert metadata["source"] == "cli"
    assert capsys.readouterr().out.strip() == "abc123"


def test_cli_run_task_metadata_json_rejects_invalid_payloads(monkeypatch):
    from ouroboros import cli

    monkeypatch.setattr(cli, "_client", lambda *args, **kwargs: pytest.fail("client should not be created"))

    for bad in ("not json", "[1, 2]"):
        args = SimpleNamespace(
            prompt=["hello"], delegation_role="root", task_metadata_json=bad,
        )
        with pytest.raises(cli.CLIError, match="task-metadata-json"):
            cli._run_command(args)


def test_cli_run_rejects_forged_subagent_role_before_request(monkeypatch):
    from ouroboros import cli

    monkeypatch.setattr(cli, "_client", lambda *args, **kwargs: pytest.fail("client should not be created"))

    args = SimpleNamespace(prompt=["hello"], delegation_role="subagent")
    with pytest.raises(cli.CLIError, match="internal schedule_subagent"):
        cli._run_command(args)


def test_cli_watch_caps_sse_wait_by_timeout(monkeypatch):
    from ouroboros import cli

    calls = []
    times = iter([100.0, 100.1, 100.2, 101.0])

    class FakeClient:
        def stream_sse(self, path, timeout=120.0, *, body=None):
            calls.append((path, timeout, body))
            return iter(())

    monkeypatch.setattr(cli.time, "time", lambda: next(times))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    with pytest.raises(cli.TaskTimeoutCLIError):
        cli._watch_task(FakeClient(), "abc123", jsonl=False, quiet=True, timeout_sec=0.5)
    assert calls[0][2]["wait"] == 0
    assert calls[0][1] <= 1.5


def test_cli_wait_task_caps_poll_request_by_timeout(monkeypatch):
    from ouroboros import cli

    calls = []
    times = iter([100.0, 100.1, 100.6])

    class FakeClient:
        timeout = 30.0

        def request(self, method, path, body=None, *, timeout=None):
            calls.append(timeout)
            raise cli.ConnectionCLIError("poll timed out")

    monkeypatch.setattr(cli.time, "time", lambda: next(times))

    with pytest.raises(cli.TaskTimeoutCLIError):
        cli._wait_task(FakeClient(), "abc123", timeout_sec=0.5)
    assert calls and calls[0] <= 0.5


def test_swebench_helper_records_cli_timeout_with_continue(tmp_path, monkeypatch):
    script_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "devtools"
        / "benchmarks"
        / "swe_bench"
        / "swebench_predictions.py"
    )
    spec = importlib.util.spec_from_file_location("swebench_predictions_test", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rows_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    logs_dir = tmp_path / "logs"
    rows_path.write_text(
        json.dumps({"instance_id": "inst1", "workspace_root": str(workspace), "problem_statement": "fix"}) + "\n",
        encoding="utf-8",
    )

    run_timeouts = []

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="abc\n", stderr="")
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        run_timeouts.append(kwargs.get("timeout"))
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1), output="partial-out", stderr="partial-err")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "get_finalization_grace_sec", lambda: 7)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "swebench_predictions.py",
            # --allow-dirty-seed: this test exercises the CLI-timeout ledger path, not the
            # v6.75.0 seed-provenance gate, so it must not depend on the developer's tree state.
            "--allow-dirty-seed",
            "--input",
            str(rows_path),
            "--output",
            str(output_path),
            "--timeout",
            "1",
            "--continue-on-error",
            "--logs-dir",
            str(logs_dir),
        ],
    )

    assert module.main() == 0
    errors = (tmp_path / "predictions.jsonl.errors.jsonl").read_text(encoding="utf-8")
    assert '"timeout": true' in errors
    assert run_timeouts == [68]
    assert (logs_dir / "inst1" / "ouroboros.stdout").read_text(encoding="utf-8") == "partial-out"
    assert (logs_dir / "inst1" / "ouroboros.stderr").read_text(encoding="utf-8") == "partial-err"


def test_terminal_bench_harbor_adapter_imports_without_harbor():
    script_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "devtools"
        / "benchmarks"
        / "terminal_bench"
        / "harbor_installed_agent.py"
    )
    spec = importlib.util.spec_from_file_location("terminal_bench_harbor_adapter_test", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.OuroborosTerminalBenchAgent.name() == "Ouroboros Installed"
    assert module._repo_root() == pathlib.Path(__file__).resolve().parent.parent


def test_queue_restore_accepts_headless_chat_zero(tmp_path, monkeypatch):
    import supervisor.queue as queue

    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "QUEUE_SEQ_COUNTER_REF", {"value": 0})
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", tmp_path / "queue_snapshot.json")
    monkeypatch.setattr(queue, "append_jsonl", lambda *args, **kwargs: None)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": True)
    (tmp_path / "queue_snapshot.json").write_text(
        json.dumps({
            "ts": utc_now_iso(),
            "pending": [{"task": {"id": "headless1", "type": "task", "chat_id": 0, "text": "x"}}],
        }),
        encoding="utf-8",
    )

    assert queue.restore_pending_from_snapshot(max_age_sec=900) == 1
    assert queue.PENDING[0]["id"] == "headless1"
