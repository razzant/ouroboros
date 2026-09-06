"""Full-result redaction precedes preview clipping in every manifest writer."""

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.loop_tool_execution import _execute_single_tool, _make_timeout_result
from ouroboros.tools.tool_result import ToolResult
from ouroboros.utils import sanitize_tool_result_for_log, truncate_for_log


@pytest.mark.parametrize("writer", ["argument_error", "tool_error", "timeout"])
@pytest.mark.parametrize("shape", ["url", "assignment"])
def test_manifest_redacts_full_result_before_url_or_assignment_boundary(tmp_path, writer, shape):
    canary = "CANARY42"  # synthetic; deliberately shorter than literal-secret threshold
    before, after = ((" postgres://operator:", "@database.invalid/example\n") if shape == "url"
                     else (" password=", "LongerOpaqueSecret7890\n"))
    prefix = {"argument_error": "⚠️ TOOL_ARG_ERROR: Could not parse arguments for '",
              "tool_error": "", "timeout": "⚠️ TOOL_TIMEOUT (probe): exceeded 5s limit. "
              "The tool is still running in background but control is returned to you. "}[writer]
    injected = "x" * (300 - len(prefix) - len(before) - len(canary)) + before + canary + after + "z" * 800
    tools = SimpleNamespace(CODE_TOOLS=set(), _ctx=SimpleNamespace(task_metadata={}),
        execute_result=lambda *_a: ToolResult(status="error", code="EXECUTOR_ERROR", text=injected))
    logs = tmp_path / "logs"
    logs.mkdir()
    if writer == "timeout":
        result = _make_timeout_result("probe", "tc", False, {"function": {"arguments": "{}"}},
                                      logs, 5, task_id="task", reset_msg=injected)
    else:
        result = _execute_single_tool(tools, {"id": "tc", "function": {
            "name": injected if writer == "argument_error" else "probe",
            "arguments": "{" if writer == "argument_error" else "{}"}}, logs, "task")
    raw = result["result"]
    assert raw[292:300] == canary
    # The old composition loses either the URL terminator or the rest of the value.
    assert canary in sanitize_tool_result_for_log(truncate_for_log(raw, 600))
    manifest = json.loads(Path(result["trace_ref"]["manifest_ref"]["path"]).read_text())
    assert manifest["error_preview"] == truncate_for_log(sanitize_tool_result_for_log(raw), 600)
    assert canary not in manifest["error_preview"]
    assert manifest["tool_code"] == {"argument_error": "TOOL_ARG_ERROR", "tool_error": "EXECUTOR_ERROR",
                                      "timeout": "TOOL_TIMEOUT"}[writer]
    with gzip.open(manifest["full_payload_ref"]["path"], "rt") as source:
        assert canary not in json.load(source)["result"]
