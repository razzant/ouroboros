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



def _captured(tmp_path, monkeypatch, *, named=True, split=False, fixed_clock=False):
    """A finished candidate run owned by retry-b whose capture and named evidence were retained.
    ``fixed_clock`` keeps the synthetic queue as fresh as a live supervisor keeps the real one, across reads."""
    ctx = retry_context(tmp_path, monkeypatch, chain=("retry-a", "retry-b", "retry-c"), split=split,
                        fixed_ownership_clock=fixed_clock)
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
    ctx, held, _data, summary = _captured(tmp_path, monkeypatch, split=split, fixed_clock=True)
    bind_gateway(monkeypatch, TerminalGateway())
    waited = call(ctx, "delegate_wait", run_id=held.run_id)  # retry-c reads retry-b's run
    assert waited.status == "ok", waited.text
    named = json.loads(waited.text)["workspace_capture"]["named_evidence"]
    assert named["status"] == "retained" and [row["path"] for row in named["files"]] == [".review-evidence/result.json"]
    assert named["retained_bytes"] == 3
    assert named["summary_read"] == {"root": "artifact_store", "path": str(summary)}
    opened = call(ctx, "read_file", **named["summary_read"])
    assert opened.status == "ok" and '"status": "retained"' in opened.text, opened.text
    # Every copy of the capture in this delivery addresses the same files the one way, and each opens.
    delivered = json.loads(waited.text)
    shown = [delivered["workspace_capture"], delivered["collection"]["workspace_capture"]]  # this wait collected
    for block in shown:
        assert block["named_evidence"]["summary_read"] == named["summary_read"]
        assert block["patch_read"] == {"root": "artifact_store", "path": block["patch_artifact"]}
        assert block["manifest_read"] == {"root": "artifact_store", "path": block["manifest_artifact"]}
    for handle, text in ((shown[0]["patch_read"], "+new"), (shown[0]["manifest_read"], "ready_with_changes")):
        opened = call(ctx, "read_file", **handle)
        assert opened.status == "ok" and text in opened.text, opened.text
    from ouroboros.tools.delegate_terminal_evidence import retained_terminal_result

    saved = retained_terminal_result(ctx, held.run_id, complete_only=False)  # the saved-result wait
    assert saved is not None and saved["workspace_capture"]["named_evidence"]["summary_read"] == named["summary_read"]
    assert {key: saved["workspace_capture"][key] for key in ("patch_read", "manifest_read")} == {
        key: shown[0][key] for key in ("patch_read", "manifest_read")}


def test_a_task_outside_the_run_s_custody_is_given_no_capture_address(tmp_path, monkeypatch):
    ctx, held, _data, _summary = _captured(tmp_path, monkeypatch)
    bind_gateway(monkeypatch, TerminalGateway())
    ctx.task_id = "unrelated-task"
    waited = call(ctx, "delegate_wait", run_id=held.run_id)
    assert "workspace_capture" not in waited.text and "summary.json" not in waited.text, waited.text


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


@pytest.mark.parametrize("settings", [
    {"MCP_SERVERS": [{"auth_token": "fixture-value"}]},
    {"MCP_SERVERS": [{"headers": {"Authorization": "Bearer fixture-value", "Accept": "application/json"}}]},
    {"MCP_SERVERS": [{"url": "https://fixture-value@example.invalid/"}]},
    {"OUROBOROS_CONNECTIONS": [{"kind": "openai", "name": "test", "api_key": "fixture-value"}]},
    {"OPENAI_COMPATIBLE_API_KEY": "fixture-value"},  # a local server's key is often one short word
])
def test_only_installation_settings_with_keys_are_excluded_from_evidence(tmp_path, monkeypatch, settings):
    """A copy of the settings file (a secret beside its own name) stays out; a report that only mentions the
    same value, as a note about a local `ollama` server would, is ordinary evidence and is kept."""
    from ouroboros import config

    monkeypatch.setattr(config, "load_settings", lambda **kw: settings)
    from ouroboros.delegate_named_evidence import retain_workspace_evidence

    copy = tmp_path / "copy"
    declared = copy / ".review-evidence"
    private = declared / "state/copy-of-settings.txt"
    quoted = declared / "logs/tools.jsonl"  # a tool result that read the file: the copy sits inside a JSON string
    public = declared / "examples/settings.json"
    report = declared / "notes/report.md"
    for path in (private, quoted, public, report):
        path.parent.mkdir(parents=True)
    private.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    quoted.write_text(json.dumps({"result": json.dumps(settings, ensure_ascii=False, indent=2)}) + "\n", encoding="utf-8")
    public.write_text(json.dumps({"OPENAI_API_KEY": "${OPENAI_API_KEY}", "Accept": "application/json"}), encoding="utf-8")
    report.write_text("The server answered fixture-value and application/json.\n", encoding="utf-8")
    retained = tmp_path / "saved"
    summary = retain_workspace_evidence(copy, retained)
    assert summary["status"] == "retained"
    assert sorted(f["path"] for f in summary["files"]) == [".review-evidence/examples/settings.json",
                                                            ".review-evidence/notes/report.md"]
    assert not (retained / "files" / private.relative_to(copy)).exists()
    assert sorted(summary["gaps"], key=lambda gap: gap["path"]) == [
        {"path": path.relative_to(copy).as_posix(), "reason": "installation_settings", "size": path.stat().st_size}
        for path in (quoted, private)
    ]


def test_private_settings_do_not_enter_the_fallback_workspace_patch(tmp_path, monkeypatch):
    from ouroboros import config, delegate_named_evidence

    monkeypatch.setattr(config, "load_settings", lambda **kw: {"OPENROUTER_API_KEY": "fixture-private"})
    from ouroboros.workspace_patch_capture import write_workspace_patch_artifacts
    from tests.test_headless_workspace_patch import _evidence_copy

    copy, task, _ = _evidence_copy(tmp_path)
    for folder, content in (
        ("private", {"OPENROUTER_API_KEY": "fixture-private"}),
        ("example", {"theme": "fixture-public"}),
    ):
        settings = copy / ".review-evidence" / folder / "settings.json"
        settings.parent.mkdir()
        settings.write_text(json.dumps(content))

    def full(*args, **kwargs):
        raise OSError(28, "volume full")

    monkeypatch.setattr(delegate_named_evidence, "retain_named_evidence", full)
    artifacts = tmp_path / "artifacts"
    _, manifest = write_workspace_patch_artifacts(
        copy, artifacts, task=task, retain_evidence_dir=tmp_path / "saved"
    )
    assert manifest["named_evidence"]["status"] == "capture_failed"
    patch = (artifacts / "workspace.patch").read_text(encoding="utf-8")
    assert "fixture-public" in patch
    assert "fixture-private" not in patch
    assert manifest["named_evidence"]["gaps"] == 1
    assert any(
        row["reason"] == "installation_settings" and row["size"] > 0 for row in manifest["untracked_excluded"]
    )


def test_investigation_receipt(tmp_path, monkeypatch):
    from ouroboros import review_checking as rc
    from pathlib import Path

    src = tmp_path / "src"
    src.mkdir()
    data = tmp_path / "data"
    source = rc.freeze_checking_source(src, data_root=data)
    created = rc.prepare_seat_area(data, "task", "review", source)
    area = Path(created)
    named = area / rc.EVIDENCE_DIRNAME
    named.mkdir()
    (named / "outline.txt").write_bytes(b"abcd")
    result = rc.capture_seat_evidence(data, "task", "review")
    assert result.get("retained_bytes") == 4


def test_product_patch_can_contain_owner_configured_values(tmp_path, monkeypatch):
    from ouroboros import config
    from ouroboros.workspace_patch_capture import write_workspace_patch_artifacts
    from tests.test_headless_workspace_patch import _evidence_copy, _git

    copy, task, _ = _evidence_copy(tmp_path)
    key = "fixture-cross-chunk"
    monkeypatch.setattr(config, "load_settings", lambda **kw: {"OPENAI_API_KEY": key})
    (copy / "old.txt").write_text(key)
    _git(copy, "add", "old.txt")
    _git(copy, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "fixture with earlier value")
    task["metadata"]["workspace_preflight"]["git"]["head"] = _git(copy, "rev-parse", "HEAD")
    (copy / "old.txt").unlink()
    (copy / "copy.txt").write_bytes(b"x" * ((1 << 20) - 4) + key.encode())
    (copy / "sample.json").write_text('{"OPENAI_API_KEY": "example"}')
    _, manifest = write_workspace_patch_artifacts(copy, tmp_path / "result", task=task)
    patch = (tmp_path / "result/workspace.patch").read_bytes()
    assert all(name in patch for name in (b"sample.json", b"old.txt", b"copy.txt"))
    assert key.encode() in patch
    assert not any(row["reason"] == "installation_settings"
                   for row in manifest["tracked_excluded"] + manifest["untracked_excluded"])
