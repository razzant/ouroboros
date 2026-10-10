"""Clean tool sources survive when only a model-request closure is published."""

from __future__ import annotations

from tests._tool_result_delivery_shared import measured_fit

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.headless import copy_child_task_result, prepare_task_drive, prune_headless_task_drives, retry_child_task_refs
from ouroboros.loop_tool_execution import process_tool_results
from ouroboros.observability import persist_call, read_blob_ref
from ouroboros.task_results import STATUS_COMPLETED, write_task_result
from ouroboros.tools.core import _read_file
from ouroboros.tools.extension_dispatch import _extension_completion
from ouroboros.tools.tool_context import ToolContext


def _marker_ref(text: str, marker: str) -> dict:
    return json.loads(next(line[len(marker):] for line in text.splitlines() if line.startswith(marker)))


@pytest.mark.parametrize("large", [False, True])
def test_clean_source_in_model_request_survives_child_copyback_and_pruning(tmp_path, large):
    parent = tmp_path / "canonical"
    task_id = "producer-copyback"
    child = prepare_task_drive(parent, task_id, "empty")
    assert child is not None
    repo = tmp_path / "repo"
    repo.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=child, task_id=task_id)
    payload = json.dumps({"ok": False, "text": "雪" * (20000 if large else 1)}, ensure_ascii=False)
    warning = "⚠️ SAFETY_WARNING: the returned content still needs inspection."
    typed = _extension_completion(payload, warning)
    messages, trace = [], {"tool_calls": []}
    process_tool_results(
        [{"fn_name": "ext_fixture", "tool_call_id": "fixture-call", "result": typed.text,
          "tool_result": typed, "is_error": True, "tool_args": {}, "args_for_log": {}}],
        messages, trace, lambda *a, **kw: None, SimpleNamespace(_ctx=ctx),
        fit_candidate=measured_fit(window=16_000, reserve=4_000),
    )
    source = _marker_ref(messages[0]["content"], "PRODUCER_RESULT_SOURCE_JSON=")
    request = persist_call(child, task_id=task_id, call_id="producer-request", call_type="llm_request",
                           payload={"messages": messages})
    # The actual collector retains call refs, not the in-memory tool result row.
    # Publish only the model request to prove marker-based dependency custody.
    write_task_result(child, task_id, STATUS_COMPLETED, result="done", artifact_status="ready",
                      trace_refs={"llm_call_refs": [{"request_ref": request["manifest_ref"]}]})
    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})
    copied = retry_child_task_refs(parent, child, task_id)
    assert copied is not None and copied["child_ref_promotion"]["status"] == "complete"
    request_ref = copied["trace_refs"]["llm_call_refs"][0]["request_ref"]
    manifest = json.loads(Path(request_ref["path"]).read_text(encoding="utf-8"))
    promoted = read_blob_ref(parent, manifest["full_payload_ref"])["messages"][0]["content"]
    assert warning in promoted
    assert _marker_ref(promoted, "PRODUCER_RESULT_SOURCE_JSON=") == source
    prune_headless_task_drives(parent, retention_days=0, now=4_000_000_000.0, live=lambda _task: False)
    assert not child.exists()
    assert read_actor_source_bytes(parent, task_id, source) == payload.encode("utf-8")
    canonical_ctx = ToolContext(repo_dir=repo, drive_root=parent, task_id=task_id)
    assert "雪" in _read_file(canonical_ctx, **source["read"]["arguments"])
    assert json.loads(read_actor_source_bytes(parent, task_id, source))["ok"] is False
    if large:
        full = _marker_ref(promoted, "FULL_RESULT_SOURCE_JSON=")
        assert read_actor_source_bytes(parent, task_id, full).decode("utf-8") == typed.text
