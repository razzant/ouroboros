"""DESIGN and the BIBLE path reach actual external inputs without changing work-order source bytes."""

import json
from hashlib import sha256
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.subagent_work_order import (
    compile_external_work_order,
    session_design_navigation,
    work_order_fingerprint,
    work_order_source_projection,
)


def _doc(root, heading):
    path = root / "docs/DESIGN.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Design\n\n## {heading}\nOriginal rule text, not navigation.\n", encoding="utf-8")
    (root / "BIBLE.md").write_text(f"# BIBLE.md\n\n## Principle 1: Continuity\n{heading} constitution text.\n",
                                   encoding="utf-8")
    return path


def _start(tmp_path, monkeypatch, *, compiled=True, kind="body_copy", missing=False, missing_bible=False,
           input_sources="declared", prepare=None):
    from ouroboros import claudexor_daemon, subagent_runtime
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools import delegate
    from ouroboros.tools.registry import ToolContext

    system, target, prepared, drive = [tmp_path / name for name in ("system", "target", "prepared", "drive")]
    for root in (system, target, prepared, drive):
        root.mkdir()
    _doc(system, "Installed heading")
    if kind == "system":
        target = system
    live_path = _doc(target, "Live target heading")
    prepared_path = _doc(prepared, "Prepared copy heading")
    if missing:
        prepared_path.unlink()
    if missing_bible:
        (prepared / "BIBLE.md").unlink()
    metadata = ({"workspace_copy": {
        "source_root": str(system), "execution_root": str(target),
        "source_is_system_repo": kind != "foreign_copy",
    }} if kind.endswith("copy") else {})
    contract = {"objective": "Inspect the selected source", "input_sources": input_sources,
                "context_requires_self_body_docs": True}
    ctx = ToolContext(
        repo_dir=system, system_repo_dir=system, drive_root=drive,
        task_id="design-session", task_metadata=metadata, task_contract=contract,
        workspace_root=target, workspace_mode="self_worktree" if kind.endswith("copy") else "external",
        task_constraint=TaskConstraint(mode="local_readonly_subagent"),
    )
    if prepare:
        prepare(ctx)
    task = {"id": ctx.task_id, "objective": contract["objective"], "task_contract": contract,
            "workspace_root": str(target), "metadata": metadata}
    canonical = compile_external_work_order(task)
    snapshot = {"schema": 1, "selected_subagent_id": "design-fixture", "config_fingerprint": "fixture-v1",
                "route": {"kind": "agent_session", "target_id": "fixture=small", "credential_profile_id": ""},
                "effort": "low", "access": "workspace_write"}
    requests = []

    class Gateway:
        engine_version = "3.8.1"

        def handshake(self, **_kwargs):
            return {}

        def agent_capabilities(self):
            return {"harnesses": [{"id": "fixture", "enabled": True, "status": "ok",
                                    "accessProfilesSupported": ["workspace_write"]}]}

        def quota_snapshots(self):
            return []

        def find_project_id(self, root):
            return "fixture-project"

        def start_run(self, request, *, idempotency_key=""):
            requests.append(request)
            return {"runId": "fixture-run", "runDir": str(drive / "run")}

        def close(self):
            pass

    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", Gateway)
    monkeypatch.setattr(delegate, "snapshot_checkout", lambda *_args: str(target))
    monkeypatch.setattr(delegate, "_provision_snapshot", lambda *_args, **_kwargs: (
        SimpleNamespace(snapshot_id="fixture-snapshot", baseline_sha="a" * 40, path=str(prepared)), None))
    custody._CUSTODY.clear()
    try:
        result = subagent_runtime.exact_start(ctx, canonical if compiled else "Inspect source", {
            "snapshot": snapshot, "compiled_work_order": compiled,
        })
        payload = json.loads(result.text)
        assert payload["status"] == "started", payload
        rows = [row for row in custody._iter_rows(custody.event_log_path(drive))
                if row.get("type") == custody.START_REQUESTED]
        assert len(rows) == 1 and len(requests) == 1
        record = custody.invocation_record(drive, rows[0]["invocation_id"])
        return requests[0], {**rows[0], **record}, task, ctx, live_path, prepared_path
    finally:
        custody._CUSTODY.clear()


@pytest.mark.parametrize("compiled", [True, False])
@pytest.mark.parametrize("kind", ["system", "body_copy"])
def test_prepared_body_navigation_reaches_compiled_and_direct_requests(tmp_path, monkeypatch, compiled, kind):
    request, row, task, _ctx, live, prepared = _start(tmp_path, monkeypatch, compiled=compiled, kind=kind)
    instructions = request["instructions"]
    assert "DESIGN NAVIGATION DELIVERY" in instructions
    assert "Prepared copy heading — lines 3-4" in instructions
    assert str(prepared) in instructions and str(live) not in instructions.split("DESIGN NAVIGATION DELIVERY", 1)[1]
    assert "Live target heading" not in instructions and "Installed heading" not in instructions
    assert "Original rule text" not in instructions
    assert "Navigation only" in instructions and "reading or application" in instructions
    assert sha256(prepared.read_bytes()).hexdigest() in instructions
    assert row["request"] == request  # the actual wire bytes are already in durable custody
    assert request["execution"]["workspaceRoot"] == str(prepared.parent.parent)
    assert request["prompt"] == (compile_external_work_order(task) if compiled else "Inspect source")
    assert ("HOST TASK CONTRACT AUTHORITY" in instructions) is not compiled


@pytest.mark.parametrize("kind", ["foreign_copy", "unknown"])
def test_foreign_and_unidentified_workspaces_get_no_body_design(tmp_path, monkeypatch, kind):
    request, row, _task, _ctx, _live, _prepared = _start(tmp_path, monkeypatch, kind=kind)
    assert "DESIGN NAVIGATION DELIVERY" not in request["instructions"]
    assert "heading" not in request["instructions"]
    assert row["request"] == request


@pytest.mark.parametrize("kind", ["system", "body_copy"])
def test_own_body_session_gets_the_constitution_path_not_its_text(tmp_path, monkeypatch, kind):
    request, _row, task, _ctx, _live, prepared = _start(tmp_path, monkeypatch, kind=kind)
    instructions, bible = request["instructions"], prepared.parent.parent / "BIBLE.md"
    assert "CONSTITUTION DELIVERY" in instructions and f"which {bible} governs" in instructions
    assert "Principle 1" in instructions and sha256(bible.read_bytes()).hexdigest() in instructions
    assert "constitution text" not in instructions  # a path and digest, never the text
    assert "CONSTITUTION DELIVERY" not in compile_external_work_order(task)


@pytest.mark.parametrize("kind", ["foreign_copy", "unknown"])
def test_foreign_and_unidentified_sessions_get_no_constitution(tmp_path, monkeypatch, kind):
    request, *_rest = _start(tmp_path, monkeypatch, kind=kind)
    instructions = request["instructions"]
    assert "CONSTITUTION DELIVERY" not in instructions and "This work concerns the body of Ouroboros" not in instructions


def test_missing_copy_constitution_is_named_without_live_fallback(tmp_path, monkeypatch):
    request, _row, _task, _ctx, live, prepared = _start(tmp_path, monkeypatch, missing_bible=True)
    instructions = request["instructions"]
    assert f"BIBLE.md source unavailable at {prepared.parent.parent / 'BIBLE.md'}" in instructions
    assert str(live.parent.parent / "BIBLE.md") not in instructions
    assert "This work concerns the body of Ouroboros" not in instructions


def test_missing_copy_source_is_explicit_and_never_falls_back_to_live_body(tmp_path, monkeypatch):
    request, _row, _task, _ctx, _live, prepared = _start(tmp_path, monkeypatch, missing=True)
    instructions = request["instructions"]
    assert "DESIGN.md source unavailable" in instructions and str(prepared) in instructions
    assert "No DESIGN content or navigation delivered" in instructions
    assert "Live target heading" not in instructions and "Installed heading" not in instructions


def test_retained_delivery_and_canonical_source_survive_design_changes(tmp_path, monkeypatch):
    request, row, task, ctx, live, prepared = _start(tmp_path, monkeypatch)
    canonical, fingerprint = compile_external_work_order(task), work_order_fingerprint(task)
    delivered = request["instructions"]
    live.write_text("## Later live heading\n", encoding="utf-8")
    prepared.write_text("## Later prepared heading\n", encoding="utf-8")
    restored = custody.invocation_record(ctx.drive_root, row["invocation_id"])
    assert restored["request"]["instructions"] == delivered
    assert "Prepared copy heading" in delivered and "Later" not in delivered
    assert compile_external_work_order(task) == canonical and work_order_fingerprint(task) == fingerprint
    projection, reason = work_order_source_projection(task, 0, len(canonical))
    assert reason == "" and projection["text"] == canonical
    assert "DESIGN NAVIGATION DELIVERY" not in canonical


def test_nested_body_start_uses_corresponding_private_repo_root(tmp_path):
    system, prepared = tmp_path / "system", tmp_path / "prepared"
    target, start = system / "ouroboros", prepared / "ouroboros"
    target.mkdir(parents=True)
    start.mkdir(parents=True)
    _doc(system, "Live")
    path = _doc(prepared, "Private")
    ctx = SimpleNamespace(repo_dir=system, workspace_root=target, workspace_mode="external",
                          drive_root=tmp_path / "drive", task_metadata={}, task_contract={})
    delivered = session_design_navigation(ctx, target_root=str(target), source_root=str(start))
    assert "Private — lines 3-4" in delivered and str(path) in delivered
    assert "Live —" not in delivered


def test_body_task_cannot_lend_its_policy_to_an_unrelated_session_target(tmp_path):
    system, foreign = tmp_path / "system", tmp_path / "foreign"
    _doc(system, "Body")
    _doc(foreign, "Foreign")
    ctx = SimpleNamespace(repo_dir=system, drive_root=tmp_path / "drive", task_metadata={}, task_contract={})
    assert session_design_navigation(ctx, target_root=str(foreign), source_root=str(foreign)) == ""


@pytest.mark.parametrize("own_body", [True, False])
def test_nested_admitted_copy_keeps_its_body_identity_and_document_root(tmp_path, own_body):
    system, copy, prepared = [tmp_path / name for name in ("system", "copy", "prepared")]
    for root in (system, copy, prepared):
        (root / "ouroboros").mkdir(parents=True)
        _doc(root, root.name)
    ctx = SimpleNamespace(repo_dir=system, drive_root=tmp_path / "drive",
                          workspace_root=copy / "ouroboros", workspace_mode="self_worktree",
                          task_metadata={"workspace_copy": {"execution_root": str(copy),
                                         "source_is_system_repo": own_body}}, task_contract={})
    text = session_design_navigation(ctx, target_root=str(copy / "ouroboros"),
                                     source_root=str(prepared / "ouroboros"))
    assert bool(text) is own_body
    if own_body:
        assert str(prepared / "docs/DESIGN.md") in text and "prepared — lines" in text


def test_nested_git_body_source_maps_the_worktree_document(tmp_path):
    import subprocess

    system, copy, prepared = [tmp_path / name for name in ("system", "copy", "prepared")]
    for root in (system, copy, prepared):
        (root / "ouroboros").mkdir(parents=True)
        _doc(root, root.name)
    subprocess.run(["git", "init", "-q"], cwd=copy, check=True, capture_output=True)
    subprocess.run(["git", "remote", "add", "origin", str(system)], cwd=copy, check=True, capture_output=True)
    ctx = SimpleNamespace(repo_dir=system, drive_root=tmp_path / "drive", workspace_root=copy / "ouroboros",
                          workspace_mode="external", task_metadata={}, task_contract={})
    text = session_design_navigation(ctx, target_root=str(copy / "ouroboros"),
                                     source_root=str(prepared / "ouroboros"))
    assert str(prepared / "docs/DESIGN.md") in text and "prepared — lines" in text


def test_declared_folderless_body_work_keeps_system_source_outside_scratch(tmp_path):
    from ouroboros.tools.registry import ToolContext

    system, prepared = [tmp_path / name for name in ("system", "prepared")]
    ctx = ToolContext(repo_dir=system, drive_root=tmp_path / "drive", project_id="folderless",
                      task_contract={"context_requires_self_body_docs": True})
    scratch = ctx.active_repo_dir()
    path = _doc(system, "System body")
    _doc(scratch, "Scratch source")
    _doc(prepared, "Prepared scratch")
    delivered = session_design_navigation(ctx, target_root=str(scratch), source_root=str(prepared))
    assert "System body — lines 3-4" in delivered and str(path) in delivered
    assert "Scratch source" not in delivered and "Prepared scratch" not in delivered
