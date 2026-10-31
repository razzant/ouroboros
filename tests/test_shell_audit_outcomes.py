"""A completed process stays authoritative when its output audit cannot finish."""

import errno
import json
import pathlib
import sys

import pytest

from ouroboros.tools import shell
from ouroboros.tools.process_facts import consume_last_process_facts
from tests.test_tool_api_v2_public_surface import _registry_under_fake_home


pytestmark = pytest.mark.serial


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    consume_last_process_facts()
    registry, _, _, _ = _registry_under_fake_home(tmp_path, monkeypatch)
    yield registry
    consume_last_process_facts()


def _arguments(tool, body, **extra):
    args = {"cwd": "task_drive", **extra}
    if tool == "run_script":
        return {**args, "script": body, "interpreter": sys.executable}
    return {**args, "cmd": [sys.executable, "-c", body]}


def _body(ending=""):
    return ("from pathlib import Path\nimport sys, time\n"
            "with Path('once.txt').open('a', encoding='utf-8') as f: f.write('once\\n')\n"
            "print('AUDIT_PROBE stdout', flush=True)\n"
            "print('AUDIT_PROBE stderr', file=sys.stderr, flush=True)\n" + ending)


@pytest.mark.parametrize("tool", ["run_command", "run_script"])
@pytest.mark.parametrize("inject_stat_error", [False, True])
def test_long_prose_is_not_a_path_or_an_audit_failure(registry, monkeypatch, tool, inject_stat_error):
    prose = "/primitives, not a path; " + "documentation " * 35
    if inject_stat_error:
        original = pathlib.Path.is_dir

        def is_dir(path):
            if str(path).startswith("/primitives,"):
                raise OSError(errno.ENAMETOOLONG, "prose is not a filename")
            return original(path)

        monkeypatch.setattr(pathlib.Path, "is_dir", is_dir)
    result = registry.execute_result(tool, _arguments(
        tool, _body(f"Path('note.md').write_text({prose!r}, encoding='utf-8')\n")))
    root = registry._ctx.task_drive_root()
    assert (root / "once.txt").read_text(encoding="utf-8") == "once\n"
    assert (root / "note.md").read_text(encoding="utf-8") == prose
    assert result.status == "ok", result.text
    assert "AUDIT_PROBE stdout" in result.text
    assert "ARTIFACT_AUDIT_GAP" not in result.text
    assert "ARTIFACT_OUTPUT_UNDECLARED" not in result.text
    facts = consume_last_process_facts()
    assert facts["exit_code"] == 0 and "pre_exec_failure" not in facts


@pytest.mark.parametrize("tool", ["run_command", "run_script"])
@pytest.mark.parametrize("ending,code", [("", "OK"), ("sys.exit(7)\n", "SHELL_EXIT_ERROR"),
                                         ("time.sleep(30)\n", "TOOL_TIMEOUT")])
def test_secondary_audit_failure_preserves_real_process_outcome(registry, monkeypatch, tool, ending, code):
    scan_calls = []

    def unavailable(_ctx, cmd, *_args, **_kwargs):
        # A run_script's inner argv names its temporary file, while its outer
        # audit receives the actual body. Fail precisely that latter audit.
        if any("AUDIT_PROBE" in str(arg) for arg in cmd):
            scan_calls.append(1)
            raise OSError("secondary observation unavailable")
        return []

    monkeypatch.setattr(shell, "_mentioned_user_file_outputs_without_declaration", unavailable)
    result = registry.execute_result(tool, _arguments(tool, _body(ending), timeout_sec=1))
    facts = consume_last_process_facts()
    assert (registry._ctx.task_drive_root() / "once.txt").read_text(encoding="utf-8") == "once\n"
    assert result.code == code, result.text
    assert "pre_exec_failure" not in facts
    assert facts["duration_ms"] >= 0
    assert facts["runtime_provenance"]["selected_path"]
    if code == "TOOL_TIMEOUT":
        assert result.status == "timeout"
        assert facts["timed_out"] and facts["killed_by_host"]
        assert "exit_code" not in facts
    else:
        assert facts["exit_code"] == (7 if ending else 0)
        assert result.meta["exit_code"] == facts["exit_code"]
        assert "AUDIT_PROBE stdout" in result.text and "AUDIT_PROBE stderr" in result.text
    audited = tool == "run_script" or code == "OK"
    assert len(scan_calls) == int(audited)
    assert ("ARTIFACT_AUDIT_GAP" in result.text) is audited
    assert "ARTIFACT_OUTPUT_UNDECLARED" not in result.text


def test_audit_gap_survives_registry_and_loop_fact_projection(registry, monkeypatch, tmp_path):
    from ouroboros.loop_tool_execution import _execute_single_tool

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("auxiliary scanner unavailable")

    monkeypatch.setattr(shell, "_mentioned_user_file_outputs_without_declaration", unavailable)
    logs = tmp_path / "trace" / "logs"
    logs.mkdir(parents=True)
    outcome = _execute_single_tool(registry, {
        "id": "audit-probe", "function": {"name": "run_command", "arguments": json.dumps(
            _arguments("run_command", _body()))}}, logs, "task1")
    assert outcome["is_error"] is False, outcome
    meta = outcome["result_meta"]
    assert meta["exit_code"] == 0 and "pre_exec_failure" not in meta
    assert meta["tool_result_meta"]["output_audit_unavailable"] == "RuntimeError"
    assert "AUDIT_PROBE stdout" in outcome["result"]


def test_audit_failure_never_hides_declared_output_copy_failure(registry, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise OSError("scanner unavailable")

    monkeypatch.setattr(shell, "_mentioned_user_file_outputs_without_declaration", unavailable)
    result = registry.execute_result("run_script", _arguments(
        "run_script", _body(), outputs=["not-created.txt"]))
    assert result.code == "ARTIFACT_OUTPUT_ERROR", result.text
    assert result.status == "error" and result.meta["exit_code"] == 0
    assert not result.meta.get("artifact_registered")
    assert result.text.count("ARTIFACT_AUDIT_GAP") == 1


def test_post_execution_registration_exception_is_not_a_spawn_failure(registry, monkeypatch):
    def failed_registration(*_args, **_kwargs):
        raise OSError("copy service unavailable after execution")

    monkeypatch.setattr(shell, "_register_process_outputs", failed_registration)
    result = registry.execute_result("run_command", _arguments("run_command", _body()))
    assert result.status == "error", result.text
    facts = consume_last_process_facts()
    assert facts["exit_code"] == 0 and "pre_exec_failure" not in facts
    assert (registry._ctx.task_drive_root() / "once.txt").read_text(encoding="utf-8") == "once\n"
