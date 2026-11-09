"""Explicit ``input_files`` for read-only delegated research (#2593), at the real start seams.

Every start goes through the registered ``delegate_start`` handler (``exact_start`` and
``_delegate_start``); only the engine is a recording double. Scheduling goes through
``_schedule_task``. The real schedule -> admission -> snapshot -> worker chain, and what a
child actually receives, is ``tests/test_delegate_input_files_admission.py``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import shutil
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from tests import _delegate_plan_shared
from tests.test_delegated_skill_payload import _payload_ctx, _StartStub
from tests._cases import case_folders

plan_setup = _delegate_plan_shared.plan_setup


class Recording(_StartStub):
    """Records every request; only this engine double is synthetic."""

    def __init__(self, fail_first=False):
        super().__init__({})
        self.requests, self.fail_first = [], fail_first

    def start_run(self, request, *, idempotency_key=""):
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        self.requests.append(copy.deepcopy(request))
        if self.fail_first and len(self.requests) == 1:
            raise ClaudexorUnavailable("daemon_unreachable", "start response lost")
        return {"runId": f"run-{len(self.requests)}", "runDir": "/tmp/run"}


def _ctx(tmp_path, monkeypatch, *, folderless=False):
    from ouroboros import claudexor_daemon, safety

    ctx = _payload_ctx(tmp_path, monkeypatch)
    if folderless:
        ctx.task_metadata["resource_intent"] = {"kind": "explicit_none"}
    engine = Recording()
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: engine)
    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))
    custody._CUSTODY.clear()
    return ctx, engine


def _start(ctx, **params):
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return json.loads(registry.execute("delegate_start", {
        "subagent_id": "payload-session", "access": "readonly", "prompt": "Compare the selected inputs.", **params}))


def _tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(pathlib.Path(root).rglob("*")) if p.is_file()}


def _drive(ctx):
    return pathlib.Path(ctx.drive_root) / "task_drives" / ctx.task_id


def _retained_home(ctx):
    """The task's canonical retained home, the one that outlives its drive."""
    from ouroboros.artifacts import task_artifact_dir_path

    return task_artifact_dir_path(ctx.drive_root, ctx.task_id) / "source_handles" / "delegated_readonly_inputs"


def _sources(ctx):
    """Two same-named files, binary bytes and a Unicode name, in two permitted roots."""
    drive = _drive(ctx)
    (drive / "a").mkdir(parents=True)
    (drive / "a" / "same.txt").write_bytes(b"drive copy\r\n")
    (drive / "отчёт.bin").write_bytes(b"\xff\x00\x10binary\n")
    (ctx.repo_dir / "src").mkdir()
    (ctx.repo_dir / "src" / "same.txt").write_bytes(b"repository copy\n")
    return [{"root": "task_drive", "path": "a/same.txt"}, {"root": "task_drive", "path": "отчёт.bin"},
            {"root": "active_workspace", "path": "src/same.txt"}]


def _expected(ctx):
    drive = _drive(ctx)
    return [drive / "a" / "same.txt", drive / "отчёт.bin", ctx.repo_dir / "src" / "same.txt"]


def test_repository_start_keeps_its_root_and_names_verified_copies_outside_it(tmp_path, monkeypatch):
    ctx, engine = _ctx(tmp_path, monkeypatch)
    selection = _sources(ctx)
    repo_before = _tree(ctx.repo_dir)
    result = _start(ctx, input_files=selection)
    assert result["status"] == "started", result
    [request] = engine.requests
    # The repository stays the run's scope and cwd: nothing is written into it.
    assert request["scope"]["root"] == str(ctx.repo_dir) == result["root"]
    assert "execution" not in request and _tree(ctx.repo_dir) == repo_before
    receipt = result["input_files"]
    assert [(row["root"], row["path"], row["status"], row["carrier"]) for row in receipt] == [
        (entry["root"], entry["path"], "prepared", "outside_scope_path") for entry in selection]
    locals_ = [pathlib.Path(row["local"]) for row in receipt]
    assert len(set(locals_)) == 3, "same basenames never collide"
    for row, local, source in zip(receipt, locals_, _expected(ctx)):
        assert local.read_bytes() == source.read_bytes() and row["source"] == str(source.resolve())
        assert row["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest() and row["size"] == len(source.read_bytes())
        assert ctx.repo_dir not in local.parents and str(local) in request["instructions"]
    assert locals_[1].name == "отчёт.bin"
    assert "SELECTED INPUT FILES" in request["instructions"] and "Preparation is not a read" in request["instructions"]
    assert "cannot confirm this harness may read outside its folder" in request["instructions"]
    staged = locals_[0].parents[3]
    assert staged == _retained_home(ctx) / result["invocation_id"], "the retained bytes are what the run reads"
    assert all(row["retained"] == row["local"] for row in receipt)
    assert not (_drive(ctx) / "delegated_readonly_inputs").exists(), "no second, disposable copy"
    manifest = json.loads((staged / "inputs.json").read_text(encoding="utf-8"))
    assert [row["selected_path"] for row in manifest] == [entry["path"] for entry in selection]
    # The source map is durable custody; the retained bytes are no deliverable of the task.
    record = custody.invocation_record(ctx.drive_root, result["invocation_id"])
    assert record["input_files"] == receipt
    from ouroboros.artifacts import collect_task_artifact_records
    assert not [row for row in collect_task_artifact_records(ctx.drive_root, ctx.task_id) if str(staged) in row["path"]]
    # An earlier retained copy can itself be selected: it is a permitted file, only never walked automatically.
    again = _start(ctx, input_files=[{
        "root": "artifact_store", "path": f"source_handles/delegated_readonly_inputs/{staged.name}/inputs.json"}])
    assert again["status"] == "started" and again["input_files"][0]["status"] == "prepared", again


def test_folderless_start_copies_the_selection_inside_its_scope_even_from_a_prior_staging(tmp_path, monkeypatch):
    ctx, engine = _ctx(tmp_path, monkeypatch, folderless=True)
    (_drive(ctx)).mkdir(parents=True)
    (_drive(ctx) / "notes.txt").write_bytes(b"own notes\n")
    first = _start(ctx)
    assert first["status"] == "started" and "input_files" not in first, first
    assert not _retained_home(ctx).exists(), "no selection retains nothing"
    prior = pathlib.Path(engine.requests[0]["scope"]["root"])
    assert json.loads((prior / "inputs.json").read_text(encoding="utf-8"))
    selection = [{"root": "task_drive", "path": f"delegated_readonly_inputs/{prior.name}/inputs.json"}]
    second = _start(ctx, input_files=selection)
    assert second["status"] == "started", second
    scope = pathlib.Path(engine.requests[1]["scope"]["root"])
    assert scope != prior and scope.parent == prior.parent
    [row] = second["input_files"]
    assert row["status"] == "prepared" and row["carrier"] == "scope"
    full = hashlib.sha256((prior / "inputs.json").read_bytes()).hexdigest()
    digest = full[:16]  # the short content address; the receipt keeps the whole digest
    assert row["sha256"] == full
    assert pathlib.Path(row["local"]) == scope / "inputs" / "selected" / digest / "inputs.json", "hash-stable name"
    assert pathlib.Path(row["local"]).read_bytes() == (prior / "inputs.json").read_bytes()
    manifest = json.loads((scope / "inputs.json").read_text(encoding="utf-8"))
    assert [m for m in manifest if "selected_path" not in m] == [], "selection replaces automatic lineage"
    assert [m["local"] for m in manifest if "selected_path" in m] == [f"inputs/selected/{digest}/inputs.json"]
    instructions = engine.requests[1]["instructions"]
    assert "READONLY INPUTS" in instructions and "SELECTED INPUT FILES" in instructions
    assert "inside this run's folder" in instructions
    assert f"inputs/selected/{digest}/inputs.json in this folder" in instructions
    # The scope copy is scratch; the retained bytes and the source map outlive the drive that held it.
    retained = pathlib.Path(row["retained"])
    assert retained == _retained_home(ctx) / second["invocation_id"] / "inputs" / "selected" / digest / "inputs.json"
    expected = (prior / "inputs.json").read_bytes()
    shutil.rmtree(_drive(ctx))
    assert retained.read_bytes() == expected
    assert custody.invocation_record(ctx.drive_root, second["invocation_id"])["input_files"] == [row]
    assert json.loads((retained.parents[3] / "inputs.json").read_text(encoding="utf-8"))[0]["source"] == row["source"]


def test_nonempty_selection_skips_unrelated_lineage_walk_but_empty_keeps_default(tmp_path, monkeypatch):
    from ouroboros import delegate_readonly_inputs as inputs

    ctx, engine = _ctx(tmp_path, monkeypatch, folderless=True)
    drive = _drive(ctx)
    drive.mkdir(parents=True)
    (drive / "chosen.md").write_bytes(b"chosen\n")
    (drive / "unrelated.md").write_bytes(b"unrelated\n")
    walked = []

    def unreadable_lineage(_base):
        walked.append(True)
        raise OSError("unrelated archive is unreadable")
        yield  # generator: the failure occurs when enumeration begins

    monkeypatch.setattr(inputs, "_input_files", unreadable_lineage)
    selected = _start(ctx, input_files=[{"root": "task_drive", "path": "chosen.md"}])
    assert selected["status"] == "started" and not walked, selected
    [request] = engine.requests
    scope = pathlib.Path(request["scope"]["root"])
    manifest = json.loads((scope / "inputs.json").read_text(encoding="utf-8"))
    assert len(manifest) == 1 and manifest[0]["selected_path"] == "chosen.md"
    assert not any("unrelated" in str(row) for row in manifest)
    for options in ({}, {"input_files": []}, {"input_files": None}):
        result = _start(ctx, **options)
        assert result["reason"] == "readonly_inputs_unavailable" and result["definitely_unrun"], result
    assert len(walked) == 3 and len(engine.requests) == 1


def test_an_unrelated_nonregular_lineage_file_refuses_only_the_automatic_inputs(tmp_path, monkeypatch):
    """Real files, no patched walk: a FIFO beside the chosen file refuses the default staging before anything is
    sent, while a nonempty selection never enumerates it."""
    import os

    if not hasattr(os, "mkfifo"):
        pytest.skip("real FIFO requires POSIX")
    ctx, engine = _ctx(tmp_path, monkeypatch, folderless=True)
    drive = _drive(ctx)
    drive.mkdir(parents=True)
    (drive / "chosen.md").write_bytes(b"chosen\n")
    os.mkfifo(drive / "stream.fifo")
    for options in ({}, {"input_files": []}):
        refused = _start(ctx, **options)
        assert (refused["reason"], refused["definitely_unrun"]) == ("readonly_inputs_unavailable", True), refused
    assert engine.requests == []
    selected = _start(ctx, input_files=[{"root": "task_drive", "path": "chosen.md"}])
    assert selected["status"] == "started", selected
    [request] = engine.requests
    manifest = json.loads((pathlib.Path(request["scope"]["root"]) / "inputs.json").read_text(encoding="utf-8"))
    assert [row.get("selected_path") for row in manifest] == ["chosen.md"]


def test_unpreparable_selection_refuses_typed_with_every_entry_before_any_send(tmp_path, monkeypatch):
    ctx, engine = _ctx(tmp_path, monkeypatch)
    drive = _drive(ctx)
    (drive / "folder").mkdir(parents=True)
    (drive / "ok.txt").write_bytes(b"ok\n")
    oracle = drive / "oracle.txt"
    oracle.write_bytes(b"black box\n")
    ctx.task_contract = {"resource_policy": {"protected_artifacts": [{
        "id": "oracle", "role": "black_box_reference", "paths": [str(oracle)]}]}}
    result = _start(ctx, input_files=[
        {"root": "task_drive", "path": "ok.txt"}, {"root": "task_drive", "path": "absent.txt"},
        {"root": "task_drive", "path": "folder"}, {"root": "task_drive", "path": "oracle.txt"},
        {"root": "task_drive", "path": "../../../escape.txt"}])
    assert result["status"] == "refused" and result["reason"] == "input_files_unavailable", result
    assert [row["status"] for row in result["input_files"]] == [
        "checked", "missing", "not_a_file", "denied", "denied"]
    assert not any(row["status"] == "prepared" for row in result["input_files"])
    assert engine.requests == [] and not (drive / "delegated_readonly_inputs").exists()
    assert not _retained_home(ctx).exists()
    assert not custody.pending_invocations(ctx.drive_root)
    for bad in ("x", [{"path": ""}], [{"path": "ok.txt", "extra": 1}], [{"root": 3, "path": "ok.txt"}]):
        refused = _start(ctx, input_files=bad)
        assert refused["status"] == "refused" and refused["reason"] in {"input_files_invalid", "TOOL_ARG_ERROR"}, refused
        assert refused["host_code"] == "TOOL_ARG_ERROR", refused
    assert engine.requests == []


def test_explicit_empty_and_omitted_selection_send_the_same_request(tmp_path, monkeypatch):
    ctx, engine = _ctx(tmp_path, monkeypatch)
    omitted, empty = _start(ctx), _start(ctx, input_files=[])
    assert omitted["status"] == empty["status"] == "started"
    assert engine.requests[0] == engine.requests[1]
    assert "input_files" not in omitted and empty["input_files"] == []
    assert "SELECTED INPUT FILES" not in engine.requests[0]["instructions"]
    assert not (_drive(ctx) / "delegated_readonly_inputs").exists() and not _retained_home(ctx).exists()


@pytest.mark.parametrize("failure", [OSError("artifact source changed during read"),
                                      FileNotFoundError("missing file in /changed/input.txt")],
                         ids=["changed-during-read", "missing-under-changed-path"])
def test_a_copy_failure_is_never_prepared(tmp_path, monkeypatch, failure):
    from ouroboros import artifacts

    ctx, engine = _ctx(tmp_path, monkeypatch)
    selection = _sources(ctx)
    real = artifacts.copy_artifact_file
    calls = []

    def flaky(source, destination, *, expected=None):
        calls.append(source)
        if len(calls) == 2:
            raise failure
        return real(source, destination, expected=expected)

    monkeypatch.setattr(artifacts, "copy_artifact_file", flaky)
    result = _start(ctx, input_files=selection)
    assert result["reason"] == "input_files_unavailable" and result["definitely_unrun"] is True, result
    assert [row["status"] for row in result["input_files"]] == ["prepared", "copy_failed", "not_attempted"]
    assert result["input_files"][1]["detail"] == str(failure)
    assert result["host_code"] == "TOOL_REPORTED_FAILURE", result
    assert engine.requests == [] and not custody.pending_invocations(ctx.drive_root)


@pytest.mark.parametrize("reason", ["attachment_path_unknown", "attachment_manifest_unavailable"])
def test_input_files_does_not_change_other_selection_refusals(tmp_path, monkeypatch, reason):
    from ouroboros import delegate_documents

    ctx, engine = _ctx(tmp_path, monkeypatch)

    def refuse(*_args, **_kwargs):
        raise delegate_documents.DocumentSelectionError(reason, "existing document refusal")

    monkeypatch.setattr(delegate_documents, "task_inputs", refuse)
    result = _start(ctx, attachment_paths=["attachments/unknown.txt"])
    assert result["status"] == "refused" and result["reason"] == reason, result
    assert "definitely_unrun" not in result and engine.requests == []


def test_retry_replays_the_recorded_request_and_copies_without_recopying_a_changed_source(tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon

    ctx, _engine = _ctx(tmp_path, monkeypatch)
    engine = Recording(fail_first=True)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: engine)
    selection = _sources(ctx)
    lost = _start(ctx, input_files=selection)
    invocation = lost["pending_invocation_id"]
    staged = sorted(_retained_home(ctx).iterdir())
    copies = {path: path.read_bytes() for path in staged[0].rglob("*") if path.is_file()}
    recorded = custody.invocation_record(ctx.drive_root, invocation)["input_files"]
    (_drive(ctx) / "a" / "same.txt").write_bytes(b"later revision\n")
    conflict = _start(ctx, retry_of=invocation, prompt="", subagent_id=None, access=None, input_files=selection)
    assert conflict["reason"] in {"retry_selector_conflict"}, conflict
    retried = _start(ctx, retry_of=invocation, prompt="", subagent_id=None, access=None)
    assert retried["status"] == "started" and retried["idempotent_recovery"] is True, retried
    assert engine.requests[0] == engine.requests[1]
    assert sorted(_retained_home(ctx).iterdir()) == staged, "no second copy"
    assert {path: path.read_bytes() for path in copies} == copies
    # The caller gets the original source map, read off the record, not a new reading of the changed source.
    assert retried["input_files"] == recorded and len(recorded) == 3
    assert recorded[0]["sha256"] == hashlib.sha256(b"drive copy\r\n").hexdigest()
    assert not (_drive(ctx) / "delegated_readonly_inputs").exists()


@pytest.mark.parametrize("selection", [None, []], ids=["omitted", "explicit-empty"])
def test_a_lost_start_retry_returns_exactly_the_recorded_receipt_or_none(tmp_path, monkeypatch, selection):
    from ouroboros import claudexor_daemon

    ctx, _engine = _ctx(tmp_path, monkeypatch)
    engine = Recording(fail_first=True)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: engine)
    lost = _start(ctx, **({} if selection is None else {"input_files": selection}))
    invocation = lost["pending_invocation_id"]
    record = custody.invocation_record(ctx.drive_root, invocation)
    assert ("input_files" in record) is (selection is not None) and record.get("input_files", selection) == selection
    retried = _start(ctx, retry_of=invocation, prompt="", subagent_id=None, access=None)
    assert retried["status"] == "started" and retried["idempotent_recovery"] is True, retried
    assert engine.requests[0] == engine.requests[1], "the receipt rides custody, never the request body"
    assert ("input_files" in retried) is (selection is not None) and retried.get("input_files", selection) == selection
    assert not _retained_home(ctx).exists() and not (_drive(ctx) / "delegated_readonly_inputs").exists()


def test_continuation_copies_only_its_own_new_selection(tmp_path, monkeypatch):
    from ouroboros import delegate_readonly_inputs as inputs
    from tests.test_investigation_start_roots import _consumer, _folderless

    scratch, metadata = _folderless(tmp_path)
    ctx, engine, start = _consumer(tmp_path, monkeypatch, "readonly", metadata=metadata)
    (scratch / "first.md").write_bytes(b"first\n")
    (scratch / "second.md").write_bytes(b"second\n")
    first = start(input_files=[{"root": "task_drive", "path": "first.md"}])
    assert first["status"] == "started", first
    assert custody.emit(ctx.drive_root, custody.SETTLED, {"run_id": first["run_id"], "state": "succeeded"})
    custody._CUSTODY.clear()
    (scratch / "unrelated-archive.md").write_bytes(b"unrelated\n")
    real_lineage, real_walk = inputs._lineage_copies, inputs._input_files

    def unexpected_lineage(*_args, **_kwargs):
        raise AssertionError("a selected continuation must not enumerate lineage")

    monkeypatch.setattr(inputs, "_lineage_copies", unexpected_lineage)
    monkeypatch.setattr(inputs, "_input_files", unexpected_lineage)
    second = start(continue_from=first["run_id"], input_files=[{"root": "task_drive", "path": "second.md"}])
    assert second["status"] == "started", second
    [row] = second["input_files"]
    assert row["path"] == "second.md" and pathlib.Path(row["local"]).read_bytes() == b"second\n"
    assert pathlib.Path(row["retained"]).read_bytes() == b"second\n" and row["retained"] != row["local"]
    request = engine.starts[1]
    assert row["local"] in request["instructions"] and first["input_files"][0]["local"] not in request["instructions"]
    assert row["local"] in request["prompt"], "a resumed session reads its own selection in the prompt"
    manifest = json.loads((pathlib.Path(request["scope"]["root"]) / "inputs.json").read_text(encoding="utf-8"))
    assert len(manifest) == 1 and manifest[0]["selected_path"] == "second.md"
    assert pathlib.Path(request["scope"]["root"]) != pathlib.Path(engine.starts[0]["scope"]["root"])
    # Without a selection, an ordinary continuation restages the whole lineage, fresh, the unrelated file included.
    monkeypatch.setattr(inputs, "_lineage_copies", real_lineage)
    monkeypatch.setattr(inputs, "_input_files", real_walk)
    assert custody.emit(ctx.drive_root, custody.SETTLED, {"run_id": second["run_id"], "state": "succeeded"})
    custody._CUSTODY.clear()
    third = start(continue_from=second["run_id"])
    assert third["status"] == "started" and "input_files" not in third, third
    scope = pathlib.Path(engine.starts[2]["scope"]["root"])
    lineage = {pathlib.Path(item["source"]).name: (scope / item["local"]).read_bytes()
               for item in json.loads((scope / "inputs.json").read_text(encoding="utf-8")) if "selected_path" not in item}
    texts = {"brief.md": scratch.parent / "parent" / "brief.md", "notes.txt": scratch / "notes.txt"}  # text mode
    assert lineage == {**{name: path.read_bytes() for name, path in texts.items()}, "first.md": b"first\n",
                       "second.md": b"second\n", "unrelated-archive.md": b"unrelated\n"}  # exact: CRLF on Windows
    assert scope not in (pathlib.Path(engine.starts[0]["scope"]["root"]), pathlib.Path(request["scope"]["root"]))
    custody._CUSTODY.clear()


# -- scheduled children ---------------------------------------------------------------------------

def _scheduler(tmp_path, monkeypatch, kind="agent_session"):
    from ouroboros.tools.registry import ToolContext
    from tests._shared import configure_test_subagent

    data = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    selected = configure_test_subagent(monkeypatch, kind=kind,
                                       target="some-route=model" if kind == "agent_session" else "openai/test-model")
    (tmp_path / "repo").mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=data)
    ctx.task_id, ctx.pending_events = "t-parent", []
    ctx.task_metadata = {"root_task_id": "t-parent"}
    drive = data / "task_drives" / "t-parent"
    drive.mkdir(parents=True)
    (drive / "brief.md").write_bytes(b"parent brief\n")
    return ctx, selected, drive


def test_schedule_copies_the_selection_before_queueing_and_the_work_order_names_it(tmp_path, monkeypatch):
    from ouroboros.subagent_work_order import compile_external_work_order
    from ouroboros.tools import control
    from tests._store_shared import read_result

    ctx, selected, drive = _scheduler(tmp_path, monkeypatch)
    result = control._schedule_task(ctx, subagent_id=selected, objective="Read the brief", expected_output="Summary",
                                    input_files=[{"root": "task_drive", "path": "brief.md"}])
    [event] = [row for row in ctx.pending_events if row["type"] == "schedule_subagent"]
    saved = read_result(ctx.drive_root, event["task_id"])
    [row] = saved["prepared_input_files"]
    assert row["status"] == "prepared" and pathlib.Path(row["local"]).read_bytes() == b"parent brief\n"
    assert pathlib.Path(row["local"]).parents[3] == _retained_home(ctx) / f"scheduled-{event['task_id']}"
    assert row["retained"] == row["local"] and not (drive / "delegated_readonly_inputs").exists()
    assert '"status": "prepared"' in result and "prepared is not read" in result
    order = compile_external_work_order(saved)
    assert "SELECTED INPUT FILES" in order and "- task_drive:brief.md" in order
    (drive / "brief.md").write_bytes(b"edited after queueing\n")
    assert pathlib.Path(row["local"]).read_bytes() == b"parent brief\n", "the queued copy is immutable"


@pytest.mark.parametrize("kind", ["agent_session", "api_model"])
def test_schedule_refuses_unpreparable_and_writing_selections_without_queueing(tmp_path, monkeypatch, kind):
    from ouroboros.tools import control

    ctx, selected, drive = _scheduler(tmp_path, monkeypatch, kind=kind)
    missing = control._schedule_task(ctx, subagent_id=selected, objective="Read", expected_output="Summary",
                                     input_files=[{"root": "task_drive", "path": "absent.md"}])
    assert "input_files_unavailable" in missing and '"status": "missing"' in missing
    writer = control._schedule_task(ctx, subagent_id=selected, objective="Edit", expected_output="Patch",
                                    write_surface="self_worktree", input_files=[{"path": "brief.md"}])
    assert "input_files_readonly_only" in writer
    ctx.task_contract = {"resource_policy": {"protected_artifacts": [{  # the child inherits this policy
        "id": "brief", "role": "black_box_reference", "paths": [str(drive / "brief.md")]}]}}
    protected = control._schedule_task(ctx, subagent_id=selected, objective="Read", expected_output="Summary",
                                       input_files=[{"root": "task_drive", "path": "brief.md"}])
    assert "input_files_unavailable" in protected and '"status": "denied"' in protected, protected
    assert not [row for row in ctx.pending_events if row["type"] == "schedule_subagent"]
    assert not _retained_home(ctx).exists()
    assert not list((ctx.drive_root / "state" / "headless_tasks").glob("*")), "no child state was left behind"


@pytest.mark.parametrize("stage", ["copy", "retained-root"])
def test_schedule_reports_failed_storage_without_blaming_the_arguments(tmp_path, monkeypatch, stage):
    from ouroboros import artifacts, delegate_readonly_inputs
    from ouroboros.tools import control
    from ouroboros.tools.tool_result import (
        _install_tool_result_sidecar, _published_tool_result, _restore_tool_result_sidecar,
    )

    ctx, selected, _drive_path = _scheduler(tmp_path, monkeypatch)

    def full_disk(*_args, **_kwargs):
        raise OSError("No space left on device")

    target, name = ((artifacts, "copy_artifact_file") if stage == "copy" else
                    (delegate_readonly_inputs, "_retained_root"))
    monkeypatch.setattr(target, name, full_disk)
    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        text = control._schedule_task(ctx, subagent_id=selected, objective="Read", expected_output="Summary",
                                      input_files=[{"root": "task_drive", "path": "brief.md"}])
        result = _published_tool_result(ctx, sentinel)
    finally:
        _restore_tool_result_sidecar(token)
    assert "No space left on device" in text and result.text == text
    assert result.code == "TOOL_REPORTED_FAILURE" and result.meta["reason"] == "input_files_unavailable"
    assert not [row for row in ctx.pending_events if row["type"] == "schedule_subagent"]


def test_bootstrap_hands_the_schedule_receipt_to_the_first_start_only(plan_setup, monkeypatch):
    from ouroboros import subagent_bootstrap, subagent_runtime
    from ouroboros.delegate_shared import delegate_result

    p, calls = plan_setup, []
    monkeypatch.setattr(subagent_runtime, "exact_start", lambda ctx, prompt, spec:
                        calls.append(spec) or delegate_result({"status": "started", "run_id": "initial",
                                                                "custody_durable": True}))
    rows = [{"root": "task_drive", "path": "brief.md", "source": "/x/brief.md", "status": "prepared",
             "local": "/x/copy/brief.md", "sha256": "0" * 64, "size": 1}]
    task = {"id": p.ctx.task_id, "objective": "Read the brief", "configured_subagent": p.snapshot,
            "task_contract": {"objective": "Read the brief"}, "prepared_input_files": rows}
    result = json.loads(subagent_bootstrap.bootstrap_before_context(
        p.ctx, task, SimpleNamespace(executor="harness", blocked=False)))
    assert result["status"] == "configured_session_started", result
    spec, = calls
    assert spec["_prepared_inputs"] == rows and "input_files" not in spec


@pytest.mark.parametrize("folderless", [False, True])
def test_first_start_uses_the_schedule_copy_verified_at_the_sender(tmp_path, monkeypatch, folderless):
    from ouroboros import subagent_runtime
    from ouroboros.delegate_readonly_inputs import schedule_selected_inputs

    ctx, engine = _ctx(tmp_path, monkeypatch, folderless=folderless)
    _drive(ctx).mkdir(parents=True)
    (_drive(ctx) / "brief.md").write_bytes(b"queued bytes\n")
    rows = schedule_selected_inputs(ctx, [{"root": "task_drive", "path": "brief.md"}], "child1")
    (_drive(ctx) / "brief.md").write_bytes(b"parent edited later\n")  # the start never rereads the original
    started = json.loads(subagent_runtime.exact_start(ctx, "Read.", {
        "subagent_id": "payload-session", "access": "readonly", "_prepared_inputs": rows}).text)
    assert started["status"] == "started", started
    [row] = started["input_files"]
    assert row["source"] == rows[0]["source"] and pathlib.Path(row["local"]).read_bytes() == b"queued bytes\n"
    assert row["local"] in engine.requests[0]["instructions"]
    if folderless:  # restaged once inside the scope; the lineage walker skipped the schedule copy
        scope = pathlib.Path(engine.requests[0]["scope"]["root"])
        assert scope in pathlib.Path(row["local"]).parents and row["local"] != rows[0]["local"]
        manifest = json.loads((scope / "inputs.json").read_text(encoding="utf-8"))
        assert sum(m["sha256"] == rows[0]["sha256"] for m in manifest) == 1, "the same bytes are staged once"
    else:  # the repository keeps its root; the schedule copy is referenced in place, not copied again
        assert row["local"] == rows[0]["local"] and engine.requests[0]["scope"]["root"] == str(ctx.repo_dir)
    pathlib.Path(rows[0]["local"]).write_bytes(b"tampered\n")
    refused = json.loads(subagent_runtime.exact_start(ctx, "Read.", {
        "subagent_id": "payload-session", "access": "readonly", "_prepared_inputs": rows}).text)
    assert refused["reason"] == "input_files_unavailable" and refused["input_files"][0]["status"] == "changed"
    assert len(engine.requests) == 1


def test_the_receiving_childs_policy_limits_what_the_caller_may_select(tmp_path, monkeypatch):
    from ouroboros.delegate_readonly_inputs import InputSelectionError, schedule_selected_inputs

    ctx, _selected, drive = _scheduler(tmp_path, monkeypatch)
    policy = {"protected_artifacts": [{"id": "brief", "role": "black_box_reference", "paths": [str(drive / "brief.md")]}]}
    with pytest.raises(InputSelectionError) as refused:  # the caller itself may read it; the child may not
        schedule_selected_inputs(ctx, [{"root": "task_drive", "path": "brief.md"}], "c1", policy)
    assert [row["status"] for row in refused.value.results] == ["denied"]
    assert "RESOURCE_POLICY_BLOCKED" in refused.value.results[0]["detail"]
    assert not _retained_home(ctx).exists()
    [row] = schedule_selected_inputs(ctx, [{"root": "task_drive", "path": "brief.md"}], "c2")
    assert row["status"] == "prepared"


@pytest.mark.parametrize("call,bootstrap_state,carried", [
    ({"attachment_paths": ["attachments/brief.md"]}, {}, True),  # its own documents keep the parent's input copies
    ({"input_files": [{"path": "own.md"}]}, {}, False),  # its own selection replaces them
    ({}, {"physical_started": True}, False),  # only the work order's first physical start carries them
    ({}, {"initial_invocation_id": "inv-1"}, False),
])
def test_a_model_issued_first_start_carries_the_schedule_receipt_unless_it_selects(monkeypatch, call, bootstrap_state,
                                                                                   carried):
    from ouroboros import subagent_runtime
    from ouroboros.delegate_shared import delegate_result

    rows = [{"root": "task_drive", "path": "brief.md", "source": "/x/brief.md", "status": "prepared",
             "local": "/x/copy/brief.md", "sha256": "0" * 64, "size": 1}]
    ctx = SimpleNamespace(task_id="child", _configured_actor_bootstrap={
        "canonical_work_order": "WORK ORDER", "work_order_fingerprint": "f", "snapshot": {},
        "prepared_input_files": rows, **bootstrap_state})
    seen = []
    monkeypatch.setattr(subagent_runtime, "exact_start", lambda _ctx, _prompt, spec: seen.append(spec) or
                        delegate_result({"status": "started", "run_id": "r", "custody_durable": True}))
    subagent_runtime.delegate_start_entry(ctx, prompt="Continue with the brief.", **call)
    [spec] = seen
    assert (spec.get("_prepared_inputs") == rows) is carried, spec


# -- the root vocabulary ----------------------------------------------------------------------------

_ROOT_BASE = {
    "active_workspace": lambda ctx, tmp: pathlib.Path(ctx.repo_dir),
    "system_repo": lambda ctx, tmp: pathlib.Path(ctx.repo_dir),
    "runtime_data": lambda ctx, tmp: pathlib.Path(ctx.drive_root),
    "task_drive": lambda ctx, tmp: _drive(ctx),
    "artifact_store": lambda ctx, tmp: _retained_home(ctx).parents[1],
    "user_files": lambda ctx, tmp: tmp / "home",
    "subagent_projects": lambda ctx, tmp: tmp / "projects",
    "deliverables": lambda ctx, tmp: tmp / "deliverables",
}


def test_the_root_vocabulary_is_every_ordinary_file_root_and_the_schema_says_so():
    from ouroboros.delegate_readonly_inputs import INPUT_FILE_ROOTS, input_files_parameter
    from ouroboros.tool_access_types import _ALL_ROOTS
    from ouroboros.tools.control_subagent_spec import schedule_subagent_properties
    from ouroboros.tools.delegate import get_tools

    assert set(INPUT_FILE_ROOTS) == set(_ROOT_BASE) == _ALL_ROOTS - {"skill_payload"}
    start = next(tool.schema for tool in get_tools() if tool.name == "delegate_start")["parameters"]["properties"]["input_files"]
    schedule = schedule_subagent_properties()["input_files"]
    assert start["items"]["properties"]["root"]["enum"] == list(INPUT_FILE_ROOTS)
    assert start["items"]["required"] == ["path"] and start["items"]["additionalProperties"] is False
    assert schedule["items"] == start["items"], "one typed item schema; schedule_subagent's lead pays its budget"
    for parameter in (start, schedule):
        assert all(root in parameter["description"] for root in INPUT_FILE_ROOTS)
        assert "mandatory" in parameter["description"] and "skill_payload and others refuse" in parameter["description"]
    assert input_files_parameter("x")["input_files"]["description"].endswith(" x")


def test_every_supported_root_prepares_through_the_real_start(subtests, tmp_path, monkeypatch):
    case_paths = case_folders(tmp_path)
    for root in sorted(_ROOT_BASE):
        tmp_path = next(case_paths)
        with subtests.test(root=root), monkeypatch.context() as monkeypatch:
            monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(tmp_path / "home"))
            monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path / "projects"))
            monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(tmp_path / "deliverables"))
            ctx, engine = _ctx(tmp_path, monkeypatch)
            source = _ROOT_BASE[root](ctx, tmp_path) / "folder" / "input.txt"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"bytes of {root}\n".encode())
            result = _start(ctx, input_files=[{"root": root, "path": "folder/input.txt"}])
            assert result["status"] == "started", result
            [row] = result["input_files"]
            assert (row["root"], row["status"]) == (root, "prepared")
            assert pathlib.Path(row["local"]).read_bytes() == source.read_bytes() and row["local"] in engine.requests[0]["instructions"]


@pytest.mark.parametrize("root", ["skill_payload", "no_such_root"])
def test_an_unsupported_root_refuses_at_both_consumers_before_anything(tmp_path, monkeypatch, root):
    from ouroboros.delegate_readonly_inputs import InputSelectionError, resolve_selection

    ctx, engine = _ctx(tmp_path, monkeypatch)
    _drive(ctx).mkdir(parents=True)
    (_drive(ctx) / "ok.txt").write_bytes(b"ok\n")
    entries = [{"root": "task_drive", "path": "ok.txt"}, {"root": root, "path": "x.txt"}]
    with pytest.raises(InputSelectionError) as refused:
        resolve_selection(ctx, entries)
    assert [row["status"] for row in refused.value.results] == ["checked", "unsupported_root"]
    assert "active_workspace" in refused.value.results[1]["detail"]
    result = _start(ctx, input_files=entries)
    assert result["status"] == "refused" and engine.requests == [], result
    assert result["reason"] in {"input_files_unavailable", "TOOL_ARG_ERROR"}, result
    from ouroboros.tools import control

    sched_ctx, selected, _drive_dir = _scheduler(tmp_path / "schedule", monkeypatch)
    queued = control._schedule_task(sched_ctx, subagent_id=selected, objective="Read", expected_output="Summary",
                                    input_files=[{"root": root, "path": "x.txt"}])
    assert "input_files_unavailable" in queued and '"status": "unsupported_root"' in queued
    assert not [row for row in sched_ctx.pending_events if row["type"] == "schedule_subagent"]


def test_an_omitted_root_is_inferred_from_an_absolute_path_as_read_file_infers_it(tmp_path, monkeypatch):
    ctx, _engine = _ctx(tmp_path, monkeypatch)
    _drive(ctx).mkdir(parents=True)
    source = _drive(ctx) / "absolute.txt"
    source.write_bytes(b"absolute\n")
    (ctx.repo_dir / "relative.txt").write_bytes(b"relative\n")
    result = _start(ctx, input_files=[{"path": str(source)}, {"path": "relative.txt"}])
    assert [row["root"] for row in result["input_files"]] == ["task_drive", "active_workspace"], result


def test_a_packed_handle_reads_back_before_it_is_retained(tmp_path, monkeypatch):
    from ouroboros import evidence_packs

    ctx, _engine = _ctx(tmp_path, monkeypatch)
    packed = _retained_home(ctx).parents[0] / "context_checkpoints" / "handle.txt"
    restored = []

    def materialize(path):
        restored.append(path)
        packed.parent.mkdir(parents=True, exist_ok=True)
        packed.write_bytes(b"restored from the pack\n")
        return packed

    monkeypatch.setattr(evidence_packs, "materialize_path", materialize)
    result = _start(ctx, input_files=[{"root": "artifact_store", "path": "source_handles/context_checkpoints/handle.txt"}])
    assert result["status"] == "started" and restored, result
    assert pathlib.Path(result["input_files"][0]["local"]).read_bytes() == b"restored from the pack\n"


# -- old calls are never gated ------------------------------------------------------------------------

def _writer(tmp_path, monkeypatch):
    """A real acting caller whose write root is a Git tree: its start provisions a private snapshot."""
    from tests._delegated_transport_shared import _delegating_ctx

    _payload_ctx(tmp_path, monkeypatch)  # the configured row and the data root
    ctx, engine = _ctx(tmp_path, monkeypatch)
    acting = _delegating_ctx(tmp_path, acting=True, task_id="t-writer")
    acting.drive_root = ctx.drive_root
    from ouroboros.task_results import write_task_result

    write_task_result(ctx.drive_root, "t-root", "running", root_task_id="t-root")  # its root exists before it delegates
    return acting, engine


def test_a_writable_start_refuses_only_a_nonempty_selection(tmp_path, monkeypatch):
    ctx, engine = _writer(tmp_path, monkeypatch)
    _drive(ctx).mkdir(parents=True, exist_ok=True)
    (_drive(ctx) / "input.txt").write_bytes(b"input\n")
    refused = _start(ctx, access="workspace_write", input_files=[{"root": "task_drive", "path": "input.txt"}])
    assert refused["reason"] == "input_files_readonly_only" and refused["definitely_unrun"] is True, refused
    assert refused["host_code"] == "TOOL_ARG_ERROR", refused
    assert engine.requests == [] and not _retained_home(ctx).exists()
    omitted, empty = _start(ctx, access="workspace_write"), _start(ctx, access="workspace_write", input_files=[])
    assert omitted["status"] == empty["status"] == "started", (omitted, empty)
    assert "input_files" not in omitted and empty["input_files"] == []
    assert [request["access"] for request in engine.requests] == ["workspace_write", "workspace_write"]
    assert "SELECTED INPUT FILES" not in engine.requests[1]["instructions"]


# -- evidence-only investigations: the other read-only shape ----------------------------------------------

def test_a_folderless_investigation_finds_the_selection_in_the_folder_its_copy_is_made_from(tmp_path, monkeypatch):
    from ouroboros import delegate_readonly_inputs as inputs
    from tests.test_investigation_start_roots import _consumer, _folderless

    scratch, metadata = _folderless(tmp_path)
    ctx, engine, start = _consumer(tmp_path, monkeypatch, "workspace_write", metadata=metadata)
    (scratch / "selected.md").write_bytes(b"selected\n")
    (scratch / "unrelated-archive.md").write_bytes(b"unrelated\n")

    def unexpected_lineage(*_args, **_kwargs):
        raise AssertionError("a selected investigation must not enumerate lineage")

    monkeypatch.setattr(inputs, "_lineage_copies", unexpected_lineage)
    monkeypatch.setattr(inputs, "_input_files", unexpected_lineage)
    result = start(input_files=[{"root": "task_drive", "path": "selected.md"}])
    assert result["status"] == "started", result
    [request] = engine.starts
    staged = pathlib.Path(request["scope"]["root"])
    assert request["execution"]["workspaceKind"] == "directory" and request["execution"]["scopePaths"] == ["."]
    [row] = result["input_files"]
    relative = pathlib.Path(row["local"]).relative_to(staged)
    assert relative.parts[:2] == ("inputs", "selected") and (staged / relative).read_bytes() == b"selected\n"
    manifest = json.loads((staged / "inputs.json").read_text(encoding="utf-8"))
    assert len(manifest) == 1 and manifest[0]["selected_path"] == "selected.md"
    assert f"{relative.as_posix()} in this folder" in request["instructions"]
    custody._CUSTODY.clear()


def test_a_git_investigation_keeps_its_copy_and_folder_and_names_the_retained_bytes(tmp_path, monkeypatch):
    from tests.test_investigation_start_roots import _consumer, _project

    _project_root, folder = _project(tmp_path)
    ctx, engine, start = _consumer(tmp_path, monkeypatch, "workspace_write", workspace=folder)
    _drive(ctx).mkdir(parents=True, exist_ok=True)
    (_drive(ctx) / "selected.md").write_bytes(b"selected\n")
    baseline = start()
    selected = start(input_files=[{"root": "task_drive", "path": "selected.md"}])
    assert baseline["status"] == selected["status"] == "started", (baseline, selected)
    plain, chosen = engine.starts
    assert plain["scope"] == chosen["scope"] and plain["mode"] == chosen["mode"] and plain["access"] == chosen["access"]
    assert plain["execution"].keys() == chosen["execution"].keys()
    [row] = selected["input_files"]
    assert row["carrier"] == "outside_scope_path" and row["local"] in chosen["instructions"]
    assert pathlib.Path(chosen["execution"]["workspaceRoot"]).name == folder.name
    assert pathlib.Path(chosen["execution"]["workspaceRoot"]) not in pathlib.Path(row["local"]).parents
    custody._CUSTODY.clear()


def test_a_model_cannot_name_the_hosts_private_prepared_rows(tmp_path, monkeypatch):
    """``_prepared_inputs`` copies without read checks, so it is reachable only from a saved task record."""
    from ouroboros.tools.registry import ToolRegistry

    ctx, engine = _ctx(tmp_path, monkeypatch)
    secret = tmp_path / "outside.txt"
    secret.write_bytes(b"not selected\n")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    forged = [{"root": "task_drive", "path": "x", "source": str(secret), "local": str(secret), "status": "prepared",
               "sha256": hashlib.sha256(b"not selected\n").hexdigest(), "size": 13}]
    text = registry.execute("delegate_start", {"subagent_id": "payload-session", "access": "readonly", "prompt": "Read.",
                                               "_prepared_inputs": forged})
    assert "TOOL_ARG_ERROR" in text and engine.requests == [], text


def test_a_long_unicode_name_keeps_its_whole_name_and_leaves_no_temporary(tmp_path, monkeypatch):
    ctx, engine = _ctx(tmp_path, monkeypatch)
    _drive(ctx).mkdir(parents=True)
    name = "отчёт" * 24 + ".md"  # 243 UTF-8 bytes: too long for a temporary sibling to keep whole
    (_drive(ctx) / name).write_bytes(b"long name\n")
    result = _start(ctx, input_files=[{"root": "task_drive", "path": name}])
    assert result["status"] == "started", result
    [row] = result["input_files"]
    assert pathlib.Path(row["local"]).name == name and pathlib.Path(row["local"]).read_bytes() == b"long name\n"
    assert name in engine.requests[0]["instructions"]
    assert not [path.name for path in pathlib.Path(row["local"]).parents[1].iterdir() if path.name.startswith(".")]


@pytest.mark.parametrize("points_outside", [False, True], ids=["inside-its-root", "outside-its-root"])
def test_a_selected_symlink_prepares_exactly_when_read_file_would_read_it(tmp_path, monkeypatch, points_outside):
    from ouroboros.tools.registry import ToolRegistry

    ctx, _engine = _ctx(tmp_path, monkeypatch)
    _drive(ctx).mkdir(parents=True)
    (_drive(ctx) / "real.txt").write_bytes(b"real bytes\n")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside bytes\n")
    (_drive(ctx) / "link.txt").symlink_to(outside if points_outside else _drive(ctx) / "real.txt")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    read = registry.execute("read_file", {"root": "task_drive", "path": "link.txt"})
    readable = "bytes" in read and "⚠️" not in read
    result = _start(ctx, input_files=[{"root": "task_drive", "path": "link.txt"}])
    [row] = result["input_files"]
    assert (row["status"] == "prepared") is readable, (read, row)
    if readable:
        assert pathlib.Path(row["local"]).read_bytes() == pathlib.Path(row["source"]).read_bytes()


# -- #2593 R1: the Presence ceiling, the receiver's contract carriers, plan threads, the projected schema ------

def _presence(ctx, *grants):
    """A positive Presence ceiling on ``ctx``'s contract, as an admitted Presence turn carries it."""
    from ouroboros.presence_authority import (PresenceCapabilityCeiling, PresenceResourceGrant, PresenceToolGrant,
                                              presence_ceiling_payload)

    ceiling = PresenceCapabilityCeiling(
        skill_name="reader", skill_content_hash="a" * 64, profile_fingerprint="b" * 64, state_fingerprint="c" * 64,
        selection_fingerprint="d" * 64, model_slot="main", inline_max_rounds=5,
        tool_grants=tuple(PresenceToolGrant(name) for name in ("delegate_start", "read_file")),
        resource_grants=tuple(PresenceResourceGrant(root, ("read",), prefix) for root, prefix in grants), digest="0" * 64)
    ctx.task_contract = {**(ctx.task_contract or {}), "capability_ceiling": presence_ceiling_payload(ceiling)}
    metadata = ctx.task_metadata if isinstance(getattr(ctx, "task_metadata", None), dict) else {}
    if metadata.get("task_contract"):  # attach_task_contract keeps the same contract on both carriers
        metadata["task_contract"] = {**metadata["task_contract"], "capability_ceiling": ctx.task_contract["capability_ceiling"]}


def test_a_presence_ceiling_admits_a_selection_exactly_as_it_admits_read_file(tmp_path, monkeypatch):
    from ouroboros.tools.core_file_tools import read_admission
    from ouroboros.tools.registry import ToolRegistry

    ctx, engine = _ctx(tmp_path, monkeypatch)
    selection = _sources(ctx)
    drive_file, repo_file = selection[0], selection[2]
    _presence(ctx, ("active_workspace", "."), ("system_repo", "."))  # repository reads only, no task drive
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    assert "PRESENCE_RESOURCE_BLOCKED" in registry.execute("read_file", drive_file)
    _root, _binding, denial = read_admission(ctx, **drive_file)
    assert denial.status == "blocked" and denial.code == "ACCESS_BLOCKED"
    assert "PRESENCE_RESOURCE_BLOCKED" in denial.text
    refused = _start(ctx, input_files=[repo_file, drive_file])
    assert refused["reason"] == "input_files_unavailable" and refused["definitely_unrun"] is True, refused
    assert [row["status"] for row in refused["input_files"]] == ["checked", "denied"]
    assert "PRESENCE_RESOURCE_BLOCKED" in refused["input_files"][1]["detail"]
    assert refused["input_files"][1]["detail"].startswith("PermissionError:"), refused
    assert engine.requests == [] and not _retained_home(ctx).exists()
    started = _start(ctx, input_files=[repo_file])  # the granted repository source still hands off
    assert started["status"] == "started", started
    [row] = started["input_files"]
    assert row["status"] == "prepared" and pathlib.Path(row["local"]).read_bytes() == b"repository copy\n"


def test_schedule_refuses_a_source_outside_the_presence_ceiling_before_queueing(tmp_path, monkeypatch):
    from ouroboros.tools import control

    ctx, selected, drive = _scheduler(tmp_path, monkeypatch, kind="api_model")
    (ctx.repo_dir / "notes.md").write_bytes(b"repository notes\n")
    _presence(ctx, ("active_workspace", "."), ("system_repo", "."))
    refused = control._schedule_task(ctx, subagent_id=selected, objective="Read", expected_output="Summary",
                                     input_files=[{"root": "task_drive", "path": "brief.md"}])
    assert "input_files_unavailable" in refused and "PRESENCE_RESOURCE_BLOCKED" in refused, refused
    assert not [row for row in ctx.pending_events if row["type"] == "schedule_subagent"]
    assert not _retained_home(ctx).exists()


def test_the_childs_policy_binds_even_beside_a_metadata_contract(tmp_path, monkeypatch):
    """``protected_artifacts`` reads a nonempty metadata contract before ``task_contract``: both carry the policy."""
    from ouroboros.delegate_readonly_inputs import InputSelectionError, schedule_selected_inputs

    ctx, _selected, drive = _scheduler(tmp_path, monkeypatch)
    ctx.task_metadata["task_contract"] = {"source": "parent_delegation", "objective": "parent's own"}
    policy = {"protected_artifacts": [{"id": "brief", "role": "black_box_reference", "paths": [str(drive / "brief.md")]}]}
    with pytest.raises(InputSelectionError) as refused:
        schedule_selected_inputs(ctx, [{"root": "task_drive", "path": "brief.md"}], "c1", policy)
    assert [row["status"] for row in refused.value.results] == ["denied"]
    assert "RESOURCE_POLICY_BLOCKED" in refused.value.results[0]["detail"]
    assert ctx.task_metadata["task_contract"] == {"source": "parent_delegation", "objective": "parent's own"}
    assert not _retained_home(ctx).exists()


def test_an_empty_plan_continuation_restates_its_new_selection_outside_the_threads_folder(plan_setup):
    p = plan_setup
    p.snapshot["access"] = "readonly"
    p.ctx.workspace_root, p.ctx.task_constraint = "", {"mode": "local_readonly_subagent"}
    p.ctx.task_metadata = {**(p.ctx.task_metadata or {}), "resource_intent": {"kind": "explicit_none"}}
    p.engine.questions = {"parse": "found", "questions": []}
    first = p.start(mode="plan")
    assert first["status"] == "started", first
    [(_path, thread, _key)] = [call for call in p.engine.calls if call[0] == "/v2/threads"]
    folder = pathlib.Path(thread["scope"]["root"])  # the thread's folder, fixed at its creation
    scopes_before = set(folder.parent.iterdir())
    prior, _payload = p.finish(first["run_id"])
    drive = pathlib.Path(p.ctx.drive_root) / "task_drives" / p.ctx.task_id
    (drive / "second.md").write_bytes(b"second input\n")
    turn = p.start("", continue_from=prior.run_id, mode="plan", input_files=[{"root": "task_drive", "path": "second.md"}])
    assert turn["status"] == "started", turn
    path, request, _key = p.engine.calls[-1]
    assert path == f"/v2/threads/{prior.thread_id}/turns" and "scope" not in request, "the thread keeps its folder"
    assert set(folder.parent.iterdir()) == scopes_before, "a turn must not create an unused scope"
    [row] = turn["input_files"]
    assert row["carrier"] == "outside_scope_path" and folder not in pathlib.Path(row["local"]).parents
    assert pathlib.Path(row["local"]).read_bytes() == b"second input\n"
    assert "SELECTED INPUT FILES" in request["prompt"] and row["local"] in request["prompt"]
    assert "outside this run's folder" in request["prompt"] and "in this folder" not in request["prompt"]
    assert "ACCESS: The starting folder is a location, not a read boundary." in request["prompt"]


def _readonly_plan_thread(p, monkeypatch):
    """A folderless read-only plan made a thread on its first start's staged folder; recorded: staging, registration."""
    from ouroboros import delegate_readonly_inputs

    p.snapshot["access"] = "readonly"
    p.ctx.workspace_root, p.ctx.task_constraint = "", {"mode": "local_readonly_subagent"}
    p.ctx.task_metadata = {**(p.ctx.task_metadata or {}), "resource_intent": {"kind": "explicit_none"}}
    p.engine.questions = {"parse": "found", "questions": []}
    drive = pathlib.Path(p.ctx.drive_root) / "task_drives" / p.ctx.task_id
    drive.mkdir(parents=True, exist_ok=True)
    (drive / "lineage.bin").write_bytes(b"\x00lineage bytes\n" * 4096)
    staged, asked, real = [], [], delegate_readonly_inputs._lineage_copies
    monkeypatch.setattr(delegate_readonly_inputs, "_lineage_copies",
                        lambda ctx, root: staged.append(pathlib.Path(root)) or real(ctx, root))
    monkeypatch.setattr(type(p.engine), "find_project_id", lambda self, root: asked.append(str(root)) or "existing-project")
    first = p.start(mode="plan")
    assert first["status"] == "started", first
    [(_path, thread, _key)] = [call for call in p.engine.calls if call[0] == "/v2/threads"]
    prior, _payload = p.finish(first["run_id"])
    asked.clear()
    return prior, thread["scope"]["root"], drive / "delegated_readonly_inputs", staged, asked


@pytest.mark.parametrize("version", ["4.5.0", "5.1.0"], ids=["engine-registers-threads", "ephemeral-threads"])
def test_plan_turns_stage_nothing_and_keep_their_threads_folder_and_registration(plan_setup, monkeypatch, version):
    """A thread turn has no reader for a new automatic tree: the engine drops the turn's scope
    (``start_review_thread_turn``) and nothing else names one. So each turn stages nothing, says nothing of a scope
    of its own, and records and registers the folder the engine's thread record names (``repoRoot``): adopted where
    the engine registers threads, nothing registered where it does not; never the live scratch. Two empty turns in
    a row, each with its own selection: every turn's prompt still names its copy by absolute path."""
    p = plan_setup
    p.engine.engine_version = version
    prior, folder, container, staged, asked = _readonly_plan_thread(p, monkeypatch)
    drive = container.parent
    for number, name in enumerate(("second.md", "third.md"), 2):
        (drive / name).write_bytes(f"input for turn {number}\n".encode())
        turn = p.start("", continue_from=prior.run_id, mode="plan", input_files=[{"root": "task_drive", "path": name}])
        assert turn["status"] == "started", turn
        path, request, _key = p.engine.calls[-1]
        assert path == f"/v2/threads/{prior.thread_id}/turns" and "scope" not in request
        [row] = turn["input_files"]
        assert row["carrier"] == "outside_scope_path" and row["local"] in request["prompt"]
        assert pathlib.Path(row["local"]).read_bytes() == f"input for turn {number}\n".encode()
        assert staged == [pathlib.Path(folder)] and [item.name for item in container.iterdir()] == [pathlib.Path(folder).name]
        assert "READONLY INPUTS" not in request["instructions"] and "scratch scope" not in json.dumps(request)
        assert custody.invocation_record(p.root, turn["invocation_id"])["request"]["scope"]["root"] == folder == turn["root"]
        assert asked == ([] if version == "5.1.0" else [folder] * (number - 1))
        assert custody.replay(p.root)[turn["run_id"]].continuation_of == prior.run_id
        prior, _payload = p.finish(turn["run_id"])

    # With no new selection or unseen owner words, the turn is still empty;
    # no access paragraph, unused input tree or alternate registration appears.
    turn = p.start("", continue_from=prior.run_id, mode="plan")
    assert turn["status"] == "started" and "input_files" not in turn
    path, request, _key = p.engine.calls[-1]
    assert path == f"/v2/threads/{prior.thread_id}/turns" and "scope" not in request
    assert request["prompt"] == "" and "READONLY INPUTS" not in request["instructions"]
    assert staged == [pathlib.Path(folder)] and [item.name for item in container.iterdir()] == [pathlib.Path(folder).name]
    assert asked == ([] if version == "5.1.0" else [folder] * 3)
    assert custody.invocation_record(p.root, turn["invocation_id"])["request"]["scope"]["root"] == folder == turn["root"]


def test_a_folderless_turn_whose_thread_record_names_no_folder_is_refused_before_anything(plan_setup, monkeypatch):
    """No root stands in for an unknown one: not an empty folder, not the live scratch. The turn is refused typed and
    unrun before its selection is retained, anything is staged or registered, or a request row exists."""
    from ouroboros.artifacts import task_artifact_dir_path

    p = plan_setup
    prior, folder, container, staged, asked = _readonly_plan_thread(p, monkeypatch)
    (container.parent / "second.md").write_bytes(b"input for the refused turn\n")
    p.engine.threads[prior.thread_id]["thread"]["repoRoot"] = None  # the engine's record names no folder
    retained = task_artifact_dir_path(p.ctx.drive_root, p.ctx.task_id) / "source_handles" / "delegated_readonly_inputs"
    calls, kept = len(p.engine.calls), sorted(path.name for path in retained.iterdir()) if retained.is_dir() else []
    requested = len([row for row in custody._iter_rows(custody.event_log_path(p.root)) if row.get("type") == custody.START_REQUESTED])
    turn = p.start("", continue_from=prior.run_id, mode="plan", input_files=[{"root": "task_drive", "path": "second.md"}])
    assert (turn["status"], turn["reason"], turn["definitely_unrun"]) == ("refused", "readonly_inputs_unavailable", True)
    assert "names no folder (repoRoot)" in turn["detail"] and "input_files" not in turn
    assert len(p.engine.calls) == calls and asked == [] and staged == [pathlib.Path(folder)]
    assert [item.name for item in container.iterdir()] == [pathlib.Path(folder).name]
    assert (sorted(path.name for path in retained.iterdir()) if retained.is_dir() else []) == kept
    assert len([row for row in custody._iter_rows(custody.event_log_path(p.root))
                if row.get("type") == custody.START_REQUESTED]) == requested and not custody.pending_invocations(p.root)


@pytest.mark.parametrize("prompt", ["", "answers"])
def test_a_plan_turn_carries_the_map_and_a_fresh_start_is_unchanged(prompt):
    from ouroboros import delegate_continuation as continuation

    plan = continuation.ContinuationStart(facts={"thread_id": "t", "continuation_of": "r"}, prompt=prompt)
    expected = f"{prompt}\n\nSELECTED" if prompt else "SELECTED"
    assert continuation.with_run_binding(plan, None, "x", "/r", "SELECTED").prompt == expected
    assert continuation.with_run_binding(plan, None, "x", "/r") is plan
    assert continuation.with_run_binding(continuation.NO_CONTINUATION, None, "x", "/r", "SELECTED") \
        is continuation.NO_CONTINUATION


def test_a_plan_turn_restates_its_access_only_beside_what_it_owes():
    """A plan-thread turn owing my human's unseen rows, a selection note or a promotion carries this turn's typed
    access with them, the rows kept whole; a turn owing none is returned untouched even with a real authority."""
    from ouroboros import delegate_continuation as continuation
    from ouroboros.delegate_start_instructions import access_instruction

    readonly = SimpleNamespace(access="readonly", investigation=False)
    access = access_instruction("readonly").strip()
    words = "MY HUMAN'S WORDS FOR THIS WORK\nGo on with the plan."
    plan = continuation.ContinuationStart(facts={"thread_id": "t", "continuation_of": "r"}, prompt="answers")
    assert continuation.with_run_binding(plan, readonly, "x", "/r") is plan
    owed = continuation.with_run_binding(plan._replace(words=words), readonly, "x", "/r").prompt
    assert owed == f"answers\n\n{words}\n\n{access}" and access.startswith("ACCESS: The starting folder is a location")
    assert continuation.with_run_binding(plan, readonly, "x", "/r", "SELECTED").prompt == f"answers\n\n{access}\n\nSELECTED"
    promoted = plan._replace(facts={**plan.facts, "promotion": {"to_access": "workspace_write"}})
    writing = continuation.with_run_binding(promoted, SimpleNamespace(access="workspace_write", investigation=False),
                                            "<run binding sentinel>", "/r").prompt
    assert writing == "answers\n\n" + access_instruction("workspace_write").strip()  # no binding invented


@pytest.mark.parametrize("profile", [{}, {"mode": "local_readonly_subagent"}], ids=["full", "readonly"])
def test_the_projected_schedule_schema_types_each_entry_as_the_handler_does(tmp_path, profile):
    from jsonschema import Draft202012Validator

    from ouroboros.delegate_readonly_inputs import InputSelectionError, selection_entries
    from ouroboros.openai_chat_custom import project_function_tools_to_openai_custom
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    registry._ctx.task_constraint = profile
    schema = next(tool for tool in registry.schemas() if tool["function"]["name"] == "schedule_subagent")
    catalog = project_function_tools_to_openai_custom([schema])
    wire = catalog.wire_tools()[0]["custom"]["description"]
    projected = json.loads(wire.rpartition("Input must be one JSON object matching this schema: ")[2])
    validator = Draft202012Validator(projected)
    base = {"subagent_id": "x", "objective": "o", "expected_output": "e"}
    for bad in ([{}], [{"path": "a.md", "extra": 1}], ["a.md"]):  # refused on the wire and by the handler alike
        assert list(validator.iter_errors({**base, "input_files": bad})), bad
        with pytest.raises(InputSelectionError):
            selection_entries(bad)
    # An unlisted root fails the wire schema; the handler refuses it per entry (unsupported_root, above).
    assert list(validator.iter_errors({**base, "input_files": [{"path": "a.md", "root": "skill_payload"}]}))
    for good in ([{"path": "a.md"}], [{"root": "task_drive", "path": "a.md"}], []):
        assert not list(validator.iter_errors({**base, "input_files": good})), good
        assert selection_entries(good) is not None


def test_exact_start_marks_input_file_refusals_without_changing_other_selectors(tmp_path, monkeypatch):
    ctx, engine = _ctx(tmp_path, monkeypatch)
    _drive(ctx).mkdir(parents=True)
    missing = _start(ctx, input_files=[{"root": "task_drive", "path": "absent.md"}])  # the exact_start pre-check
    assert missing["reason"] == "input_files_unavailable" and missing["definitely_unrun"] is True, missing
    conflict = _start(ctx, retry_of="pending-1", input_files=[])
    assert conflict["reason"] == "retry_selector_conflict" and "definitely_unrun" not in conflict, conflict
    assert engine.requests == []


def test_the_content_address_is_short_and_never_overwrites_other_bytes(tmp_path):
    from ouroboros.delegate_readonly_inputs import ADDRESS_HEX, InputSelectionError, copy_selection

    source = tmp_path / "brief.md"
    source.write_bytes(b"selected bytes\n")
    digest = hashlib.sha256(b"selected bytes\n").hexdigest()
    row = {"root": "task_drive", "path": "brief.md", "source": str(source)}
    [first] = copy_selection([row], tmp_path / "a")
    local = pathlib.Path(first["local"])
    assert local.parent.name == digest[:ADDRESS_HEX] and first["sha256"] == digest
    # 48 characters shorter than the whole digest: Windows' 260-character limit keeps more of a long name.
    assert len(str(local)) == len(str(tmp_path / "a" / "inputs" / "selected" / digest / "brief.md")) - 48
    [again] = copy_selection([row], tmp_path / "a")  # the same bytes at the same address stay
    assert again["local"] == first["local"] and local.read_bytes() == b"selected bytes\n"
    local.write_bytes(b"other bytes\n")  # a colliding or planted occupant is refused, never replaced
    with pytest.raises(InputSelectionError) as refused:
        copy_selection([row], tmp_path / "a")
    assert refused.value.results[0]["status"] == "copy_failed" and "other bytes" in refused.value.results[0]["detail"]
    assert local.read_bytes() == b"other bytes\n"
    assert not [path.name for path in local.parent.iterdir() if path.name.startswith(".")]
