"""A saved result read later, or by a retry successor, keeps its run's named-evidence summary (#2070).

Its status and a reader the receiving task can open ride the existing capture projection: one
addressed read of the retained summary, never a recapture, with the handle qualified for another
task exactly as the patch's and the manifest's are.
"""
from __future__ import annotations

import json

import pytest

from ouroboros import delegate_custody as custody
from tests.test_delegate_retry_consumers import (
    TerminalGateway,
    bind_gateway,
    call,
    durable_run,
    retry_context,
)
from tests.test_delegate_retry_consumers import (
    isolated_retry_state as isolated_retry_state,  # noqa: F401 -- re-export the autouse pytest fixture
)

pytestmark = pytest.mark.serial


def _captured(tmp_path, monkeypatch, *, named=True, split=False):
    """A finished candidate run owned by retry-b whose capture and named evidence were retained."""
    ctx = retry_context(tmp_path, monkeypatch, chain=("retry-a", "retry-b", "retry-c"), split=split)
    held = durable_run(ctx, owner="retry-b", snapshot_id="capture-one", execution_root=str(tmp_path / "snapshot"))
    data = custody.custody_root(ctx)
    capture = custody.delegated_capture_dir(data, held.task_id, held.snapshot_id)
    capture.mkdir(parents=True)
    (capture / "workspace.patch").write_text("--- a/r.txt\n+++ b/r.txt\n@@ -1 +1 @@\n-old\n+new\n", encoding="utf-8")
    (capture / "workspace_patch.json").write_text(json.dumps({"status": "ready_with_changes"}), encoding="utf-8")
    summary = capture / "evidence" / "summary.json"
    if named:  # as the retention owner writes it, its own handle relative to the owner's store
        (capture / "evidence" / "files" / ".review-evidence").mkdir(parents=True)
        (capture / "evidence" / "files" / ".review-evidence" / "result.json").write_text("{}\n", encoding="utf-8")
        summary.write_text(json.dumps({
            "status": "retained", "evidence_dir": str(summary.parent), "gaps": [], "witness": "w",
            "summary_read": {"root": "artifact_store", "path": f"delegated_runs/{capture.name}/evidence/summary.json"},
            "files": [{"path": ".review-evidence/result.json", "size": 3, "sha256": "s"}]}), encoding="utf-8")
    assert custody.record_patch_captured(data, held, status="ready_with_changes")
    custody._CUSTODY.clear()
    return ctx, held, data, summary


@pytest.mark.parametrize("split", [False, True])
def test_a_retry_successor_is_shown_the_named_evidence_and_can_open_its_summary(tmp_path, monkeypatch, split):
    ctx, held, _data, summary = _captured(tmp_path, monkeypatch, split=split)
    bind_gateway(monkeypatch, TerminalGateway())
    waited = call(ctx, "delegate_wait", run_id=held.run_id)  # retry-c reads retry-b's run
    assert waited.status == "ok", waited.text
    named = json.loads(waited.text)["workspace_capture"]["named_evidence"]
    assert named["status"] == "retained" and [row["path"] for row in named["files"]] == [".review-evidence/result.json"]
    assert named["summary_read"] == {"root": "artifact_store", "path": str(summary)}
    opened = call(ctx, "read_file", **named["summary_read"])
    assert opened.status == "ok" and '"status": "retained"' in opened.text, opened.text


def test_the_owner_reads_its_saved_summary_through_its_own_handle_and_nothing_is_recaptured(tmp_path, monkeypatch):
    from ouroboros.delegate_collection import capture_projection
    from ouroboros.tools import subagent_integration_delegated as delegated

    ctx, held, data, _summary = _captured(tmp_path, monkeypatch)
    for name in ("_retain_named_evidence", "run_named_evidence"):
        monkeypatch.setattr(delegated, name, lambda *_a, **_k: pytest.fail("a read never recaptures"))
    block = capture_projection(data, custody.replay(data)[held.run_id], reader_task_id=held.task_id)
    handle = block["named_evidence"]["summary_read"]
    assert handle == {"root": "artifact_store", "path": "delegated_runs/capture-one/evidence/summary.json"}
    ctx.task_id = held.task_id
    opened = call(ctx, "read_file", **handle)
    assert opened.status == "ok" and '"status": "retained"' in opened.text, opened.text


def test_a_run_that_retained_no_named_evidence_shows_none(tmp_path, monkeypatch):
    from ouroboros.delegate_collection import capture_projection

    _ctx, held, data, _summary = _captured(tmp_path, monkeypatch, named=False)
    assert "named_evidence" not in capture_projection(data, custody.replay(data)[held.run_id], reader_task_id="retry-c")
