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
import os
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
    # c50-2: the automatic lineage copies are counted in the receipt; a selection that replaces them is not.
    assert "prepared: inputs: automatic lineage, 1 file, 10 bytes" in first["note"], first["note"]
    assert not _retained_home(ctx).exists(), "no selection retains nothing"
    prior = pathlib.Path(engine.requests[0]["scope"]["root"])
    assert json.loads((prior / "inputs.json").read_text(encoding="utf-8"))
    selection = [{"root": "task_drive", "path": f"delegated_readonly_inputs/{prior.name}/inputs.json"}]
    second = _start(ctx, input_files=selection)
    assert second["status"] == "started", second
    assert "automatic lineage" not in second["note"] and "outside the session's folder" not in second["note"]
    assert second["note"].startswith("receipt:\nrequested: "), "a copy inside its scope differs in nothing: no line"
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
    # A refusal is its own answer: the per-entry rows and reasons, never a receipt beside them.
    assert "receipt" not in result and "receipt:" not in json.dumps(result), result
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
    assert "receipt" not in result and "receipt:" not in json.dumps(result), "a refusal is its own answer"
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
    assert "automatic lineage" not in retried["note"], "outside a folderless Project no lineage copies were possible"
    assert not _retained_home(ctx).exists() and not (_drive(ctx) / "delegated_readonly_inputs").exists()


def test_a_folderless_replay_that_selected_nothing_names_its_lineage_gap(tmp_path, monkeypatch):
    """The one start that could have received automatic lineage copies, replayed: their count is not recorded."""
    ctx, engine = _ctx(tmp_path, monkeypatch, folderless=True)
    _drive(ctx).mkdir(parents=True, exist_ok=True)
    (_drive(ctx) / "fact.txt").write_bytes(b"old\n")
    engine.fail_first = True
    lost = _start(ctx, input_files=[])
    retried = _start(ctx, retry_of=lost["pending_invocation_id"], subagent_id=None, access=None, prompt="")
    assert retried["status"] == "started" and retried["idempotent_recovery"] is True, retried
    manifest = json.loads((pathlib.Path(engine.requests[0]["scope"]["root"]) / "inputs.json").read_text(encoding="utf-8"))
    assert manifest and all("selected_path" not in row for row in manifest), "the first send carried lineage copies"
    assert "inputs: automatic lineage, no record" in retried["note"], retried["note"]


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
    # The receipt names count and bytes, not a manifest dump; the stage keeps "prepared is not read" (#2593).
    assert "prepared: inputs: 1 file, 13 bytes, retained before queueing (prepared is not read)" in result
    assert row["local"] not in result
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
    assert "input_files_unavailable" in missing and '"status": "missing"' in missing and "receipt:" not in missing
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
    ordinary, ranged = start["items"]["anyOf"]
    assert ranged["required"] == ["git_range"] and ranged["additionalProperties"] is False
    assert ordinary["properties"]["root"]["enum"] == list(INPUT_FILE_ROOTS)
    assert ordinary["required"] == ["path"] and ordinary["additionalProperties"] is False
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

def _presence(ctx, *grants, vcs=False, read_bindings=(), vcs_bindings=()):
    """A positive Presence ceiling on ``ctx``'s contract, as an admitted Presence turn carries it."""
    from ouroboros.presence_authority import (PresenceCapabilityCeiling, PresenceResourceGrant, PresenceToolGrant,
                                              presence_ceiling_payload)

    tools = ["delegate_start", "read_file"] + (["vcs_read"] if vcs else [])
    operations = ("read", "vcs") if vcs else ("read",)
    ceiling = PresenceCapabilityCeiling(
        skill_name="reader", skill_content_hash="a" * 64, profile_fingerprint="b" * 64, state_fingerprint="c" * 64,
        selection_fingerprint="d" * 64, model_slot="main", inline_max_rounds=5,
        tool_grants=tuple(PresenceToolGrant(name, {"read_file": read_bindings, "vcs_read": vcs_bindings}.get(name, ()))
                          for name in tools),
        resource_grants=tuple(PresenceResourceGrant(root, operations, prefix) for root, prefix in grants), digest="0" * 64)
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


@pytest.mark.parametrize("projected_reader", ["caller", "scheduled-child"])
def test_selected_file_uses_the_actual_read_file_dispatch_root(tmp_path, monkeypatch, projected_reader):
    from ouroboros.tools import control
    from ouroboros.tools.registry import ToolRegistry

    home = tmp_path / "files"
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    workspace = home / "workspace"
    workspace.mkdir(parents=True)
    source = workspace / "sample.md"
    source.write_bytes(b"ordinary sample\n")
    selection = [{"root": "user_files", "path": str(source)}]
    if projected_reader == "caller":
        ctx, engine = _ctx(tmp_path, monkeypatch)
        ctx.workspace_root, ctx.workspace_mode = workspace, "external"
    else:
        ctx, selected, _ = _scheduler(tmp_path, monkeypatch, kind="api_model")
        options = {"subagent_id": selected, "objective": "Read", "expected_output": "Summary",
                   "workspace_root": str(workspace), "input_files": selection, "memory_mode": "empty"}
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    grants = (("user_files", "workspace/sample.md"),)
    _presence(ctx, *grants)
    if projected_reader == "caller":
        observed = registry.execute("read_file", selection[0])
        assert "PRESENCE_RESOURCE_BLOCKED" in observed, observed
        denied = _start(ctx, input_files=selection)
        assert denied["definitely_unrun"] and denied["input_files"][0]["status"] == "denied"
        assert "PRESENCE_RESOURCE_BLOCKED" in denied["input_files"][0]["detail"] and not engine.requests
    else:
        observed = registry.execute("read_file", selection[0])
        assert "ordinary sample" in observed and "PRESENCE_RESOURCE_BLOCKED" not in observed
        denied = control._schedule_task(ctx, **options)
        assert "PRESENCE_RESOURCE_BLOCKED" in denied and '"status": "denied"' in denied and not ctx.pending_events
    assert not _retained_home(ctx).exists()

    # A real direct start also needs read authority for its entire bound scope.
    _presence(ctx, *grants, ("active_workspace", "." if projected_reader == "caller" else "sample.md"))
    if projected_reader == "caller":
        allowed = _start(ctx, input_files=selection)
        assert allowed["status"] == "started", allowed
        row, = allowed["input_files"]
        assert row["root"] == "active_workspace"
    else:
        allowed = control._schedule_task(ctx, **options)
        event, = ctx.pending_events
        row, = event["prepared_input_files"]
        assert row["root"] == "user_files", allowed  # caller's address, not the recipient's policy root
    assert row["source"] == str(source) and pathlib.Path(row["local"]).read_bytes() == b"ordinary sample\n"


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
    ranged = {"git_range": {"base": "v1", "head": "HEAD", "repository": "nested", "path": "src"}}
    for bad in ([{}], [{"path": "a.md", "extra": 1}], ["a.md"], [{**ranged, "path": "other"}],
                [{"git_range": {"head": "HEAD"}}], [{"git_range": {**ranged["git_range"], "flags": "--stat"}}],
                [{"git_range": {"base": "", "head": "HEAD"}}], [{"git_range": {"base": "v1", "head": ""}}]):
        assert list(validator.iter_errors({**base, "input_files": bad})), bad
        with pytest.raises(InputSelectionError):
            selection_entries(bad)
    # An unlisted root fails the wire schema; the handler refuses it per entry (unsupported_root, above).
    assert list(validator.iter_errors({**base, "input_files": [{"path": "a.md", "root": "skill_payload"}]}))
    for good in ([{"path": "a.md"}], [{"root": "task_drive", "path": "a.md"}], [ranged],
                 [{"git_range": {**ranged["git_range"], "repository": "", "path": ""}}], []):
        assert not list(validator.iter_errors({**base, "input_files": good})), good
        assert selection_entries(good) is not None


def test_exact_start_marks_input_file_refusals_without_changing_other_selectors(tmp_path, monkeypatch):
    from ouroboros import delegate_git_range

    ctx, engine = _ctx(tmp_path, monkeypatch)
    _drive(ctx).mkdir(parents=True)
    monkeypatch.setattr(delegate_git_range, "range_binding", lambda *_a: pytest.fail("Git ran during preflight"))
    missing = _start(ctx, input_files=[{"root": "task_drive", "path": "absent.md"},
                                       {"git_range": {"base": "unknown-local-rev", "head": "HEAD"}}])
    assert missing["reason"] == "input_files_unavailable" and missing["definitely_unrun"] is True, missing
    assert [row["status"] for row in missing["input_files"]] == ["missing", "deferred"]
    assert not missing.get("engine_discovery_completed") and not _retained_home(ctx).exists()
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


def _range_fixture(repo, *, large=False, unchanged=(), padded=False):
    """Two unrelated local commits. Plumbing constructs byte-named entries even on a UTF-8-only filesystem."""
    import os
    import subprocess

    repo.mkdir(parents=True, exist_ok=True)

    def git(*args, data=b""):
        return subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.org", *args],
                              cwd=repo, input=data, stdout=subprocess.PIPE, check=True).stdout.strip().decode("ascii")

    def tree(rows):
        data = b"".join(mode + b" " + oid.encode() + b"\t" + name + b"\0" for mode, oid, name in rows)
        return git("mktree", "-z", data=data)

    git("init", "-q")
    old = git("hash-object", "-w", "--stdin", data=b"old\n")
    stable = [(b"100644 blob", old, name) for name in unchanged]
    archive = tree([(b"100644 blob", old, b"notes.md")])
    before = tree([(b"100644 blob", old, b"readme[1]"), (b"100644 blob", old, b"removed"),
                   (b"040000 tree", archive, b"archive"), *stable])
    base = git("commit-tree", before, "-m", "private base message")
    git("update-ref", "refs/heads/before", base)
    new = git("hash-object", "-w", "--stdin", data=b"new \xff\n")
    binary = git("hash-object", "-w", "--stdin", data=b"\x00\xff\x02")
    rows = [(b"100755 blob", new, b"readme[1]"), (b"100644 blob", binary, b"image"),
            (b"160000 commit", base, b"linked"), (b"100644 blob", old, b"readme1"), *stable]
    if os.name == "posix":
        rows.extend((b"100644 blob", old, name) for name in (b"byte-\xff", b"padded-name "))
    if padded and os.name != "posix":  # the POSIX rows above already hold it; plumbing writes it on Windows too
        rows.append((b"100644 blob", old, b"padded-name "))
    if large:
        blob = git("hash-object", "-w", "--stdin", data=b"long line for the complete patch\n" * 40000)
        rows.append((b"100644 blob", blob, b"large"))
    after = tree(rows)
    head = git("commit-tree", after, "-m", "private head message")
    git("update-ref", "refs/heads/after", head)
    git("config", "diff.ignoreSubmodules", "all")
    git("config", "diff.external", "nonexistent-fixture-diff")
    return {"base": base, "head": head}, {"base": before, "head": after}


def _expected_range(repo, commits, path=""):
    import subprocess

    return subprocess.check_output([
        "git", "diff", "--binary", "--full-index", "--no-color", "--no-ext-diff", "--no-textconv", "--no-renames",
        "--src-prefix=a/", "--dst-prefix=b/", "--ignore-submodules=none", "--submodule=short",
        commits["base"], commits["head"], "--", *([f":(literal){path}"] if path else [])], cwd=repo)


def _unescape_range(text):
    import re

    return re.sub(rb"\\\\|\\x([0-9a-f]{2})", lambda m: bytes.fromhex(m[1].decode()) if m[1] else b"\\", text.encode())


def test_a_scheduled_childs_reader_metadata_is_what_dispatch_gives_the_child(tmp_path, monkeypatch):
    """The recipient's reader answers for the metadata the child will actually run with
    (``supervisor.task_dispatch``): tree origin and Presence binding authority, never a speaker's event or
    a parent's own handoff. A benign permitted file still reads through it."""
    from ouroboros.consciousness_authority import consciousness_origin_metadata
    from ouroboros.delegate_readonly_inputs import receiving_reader, schedule_selected_inputs
    from ouroboros.dialogue_provenance import presence_binding_authority_metadata
    from supervisor.task_dispatch import build_scheduled_task_payload

    ctx, _selected, drive = _scheduler(tmp_path, monkeypatch, kind="api_model")
    ctx.task_metadata = {
        "root_task_id": "t-parent", "initiator": "consciousness", "usage_category": "consciousness",
        "consciousness_cause": "own_clock", "focus": {"label": "watch"},
        "presence": {"binding_id": "binding-1", "event": {"text": "the speaker's own words"}},
        "prepared_input_files": [{"local": "parent-handoff"}], "held_owner_messages": ["private"], "tool_profile": "x"}
    reader = receiving_reader(ctx, child_task_id="c1", child_drive=tmp_path / "child-drive")
    canonical = build_scheduled_task_payload({
        "tid": "c1", "parent_id": "t-parent", "root_task_id": "t-parent", "delegation_role": "subagent",
        "origin_metadata": consciousness_origin_metadata({**ctx.task_metadata, "task_id": "t-parent"}),
        **presence_binding_authority_metadata(ctx.task_metadata)})["metadata"]
    seen = reader.task_metadata
    for key in ("initiator", "usage_category", "consciousness_cause", "focus", "work_kind", "root_task_id",
                "parent_task_id", "presence_binding_authority"):
        assert seen[key] == canonical[key], key
    assert seen["presence_binding_authority"] == {"binding_id": "binding-1"}
    for key in ("presence", "prepared_input_files", "held_owner_messages", "tool_profile"):
        assert key not in seen, key
    [row] = schedule_selected_inputs(ctx, [{"root": "task_drive", "path": "brief.md"}], "c1")
    assert pathlib.Path(row["local"]).read_bytes() == b"parent brief\n"
    assert drive.exists()


@pytest.mark.parametrize("static, admitted", [("", True), ("native", False)], ids=["inert-empty", "real-selector"])
def test_a_static_bucket_binding_is_inert_only_while_empty(tmp_path, monkeypatch, static, admitted):
    """An ordinary root's read_file binding may fix ``bucket`` to its empty default without refusing a whole-file
    selection; a nonempty selector (or any other fixed operand) still changes what is read and refuses."""
    from ouroboros.presence_authority import PresenceArgumentBinding

    ctx, engine = _ctx(tmp_path, monkeypatch)
    selection = _sources(ctx)
    _presence(ctx, ("active_workspace", "."), ("system_repo", "."), ("task_drive", "."), read_bindings=(PresenceArgumentBinding(("bucket",), "static", static_value=static),))
    result = _start(ctx, input_files=[selection[0]])
    if admitted:
        assert result["status"] == "started", result
        [row] = result["input_files"]
        assert row["status"] == "prepared" and pathlib.Path(row["local"]).is_file()
    else:
        assert result["definitely_unrun"] is True and "PRESENCE_RESOURCE_BLOCKED" in json.dumps(result), result
        assert engine.requests == []


def test_posix_historic_names_without_a_windows_spelling_are_recognized_on_every_host(monkeypatch):
    """A legitimate POSIX name that ends in a space (or a dot, or is reserved) has no Windows directory entry;
    an ordinary name and bracketed names do."""
    import types

    from ouroboros import delegate_git_range

    assert [delegate_git_range._unsafe_win_part(part) for part in
            ("padded-name ", "dotted.", "CON", "a:b", "readme[1]", "ordinary-name.md")] == [
        True, True, True, True, False, False]
    monkeypatch.setattr(delegate_git_range, "os", types.SimpleNamespace(name="nt"))
    assert delegate_git_range._relative_entry("dir/ordinary-name.md").as_posix() == "dir/ordinary-name.md"
    for name in ("padded-name ", "dir /file"):
        with pytest.raises(ValueError, match="no unambiguous Windows spelling"):
            delegate_git_range._relative_entry(name)
    monkeypatch.setattr(delegate_git_range, "os", types.SimpleNamespace(name="posix"))
    assert delegate_git_range._relative_entry("padded-name ").as_posix() == "padded-name "


@pytest.mark.skipif(os.name != "nt", reason="the range refuses a POSIX-only historic name only on Windows")
def test_windows_refuses_a_range_holding_a_posix_name_with_a_trailing_space_and_keeps_ordinary_ones(tmp_path, monkeypatch):
    ctx, engine = _ctx(tmp_path, monkeypatch)
    repo = ctx.repo_dir / "nested"
    _range_fixture(repo, padded=True)
    spec = {"repository": "nested", "base": "before", "head": "after"}
    refused = _start(ctx, input_files=[{"git_range": spec}])
    assert refused["definitely_unrun"] is True and "no unambiguous Windows spelling" in json.dumps(refused), refused
    assert engine.requests == []
    ordinary = _start(ctx, input_files=[{"git_range": {**spec, "path": "readme1"}}])
    assert ordinary["status"] == "started", ordinary


def test_direct_range_delivers_the_complete_two_tree_patch_and_byte_paths(tmp_path, monkeypatch):
    import base64
    import os
    from ouroboros.tools import vcs_read

    ctx, engine = _ctx(tmp_path, monkeypatch)
    repo = ctx.repo_dir / ("nested" if os.name == "nt" else "nested ")
    commits, trees = _range_fixture(repo, large=True)
    (repo / "readme[1]").write_bytes(b"uncommitted contents\n")
    commands = []
    git_command = vcs_read._read_git_command

    def observe(args):
        command, environment = git_command(args)
        assert environment["GIT_NO_REPLACE_OBJECTS"] == environment["GIT_NO_LAZY_FETCH"] == "1"
        assert environment["GIT_ALLOW_PROTOCOL"] == ""
        commands.append(args)
        return command, environment

    monkeypatch.setattr(vcs_read, "_read_git_command", observe)
    result = _start(ctx, input_files=[{"git_range": {"repository": repo.name, "base": "before", "head": "after"}}])
    assert result["status"] == "started", result
    # Each revision is observed once as written, with core.warnAmbiguousRefs forced on so Git itself
    # reports an ambiguous short ref; unrelated diagnostics never block a commit. Only the ids named are typed.
    looked = [args for args in commands if args[-1] in {"before", "after"}]
    assert [args[-1] for args in looked] == ["before", "after"]
    assert all("core.warnAmbiguousRefs=true" in args for args in looked)
    assert [args[-1] for args in commands if args[-1].endswith("^{commit}")] == [
        f"{commits['base']}^{{commit}}", f"{commits['head']}^{{commit}}"]
    diffs = [args for args in commands if args[0] == "diff-tree"]
    assert len(diffs) == 2 and all({"--no-ext-diff", "--no-textconv", "--no-renames", "--ignore-submodules=none",
                                    "--submodule=short"} <= set(args) for args in diffs)
    rows = result["input_files"]
    files = {row["path"]: pathlib.Path(row["local"]) for row in rows}
    manifest = json.loads(files["range.json"].read_bytes())
    raw = files["range.patch"].read_bytes()
    assert manifest["repository"] == str(repo) and manifest["trees"] == trees and manifest["commits"] == commits
    assert len(raw) > 1024 * 1024 and raw == _expected_range(repo, commits)
    assert b"GIT binary patch" in raw and b"Subproject commit" in raw and b"old mode 100644" in raw
    assert b"private base message" not in raw and b"private head message" not in raw
    view = manifest["patch"]["text_view"]
    assert view["encoding"] == "utf8-byte-escapes-v1" and _unescape_range(files[view["name"]].read_text(encoding="utf-8")) == raw
    assert manifest["patch"]["size"] == len(raw) and manifest["patch"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert {"path": "removed", "status": "D"} in manifest["changes"]
    assert {"path": "archive/notes.md", "status": "D"} in manifest["changes"]
    if os.name == "posix":
        assert {"path_base64": base64.b64encode(b"byte-\xff").decode(), "status": "A"} in manifest["changes"]
        assert {"path": "padded-name ", "status": "A"} in manifest["changes"]
    request, = engine.requests
    assert request["scope"]["root"] == str(ctx.repo_dir) and all(str(p) in request["instructions"] for p in files.values())
    assert str(repo) in request["instructions"] and commits["base"] in request["instructions"]


def test_a_range_names_its_commits_by_an_ordinary_message_search_and_refuses_a_non_commit(tmp_path, monkeypatch):
    from ouroboros.tools import vcs_read

    ctx, _engine = _ctx(tmp_path, monkeypatch)
    commits, _trees = _range_fixture(ctx.repo_dir / "nested")
    looked = []
    capture = vcs_read._capture_read_git
    monkeypatch.setattr(vcs_read, "_capture_read_git", lambda path, args: looked.append(args[-1]) or capture(path, args))
    spec = {"repository": "nested", "base": ":/private base message", "head": ":/private head message"}
    started = _start(ctx, input_files=[{"git_range": spec}])
    assert started["status"] == "started", started
    manifest = json.loads(next(pathlib.Path(row["local"]) for row in started["input_files"]
                               if row["path"] == "range.json").read_bytes())
    assert manifest["commits"] == commits
    # Each search is read once as written, then only the full id it named is typed: no peel joins user text.
    assert [name for name in looked if name.startswith(":/") or name.endswith("^{commit}")] == [
        spec["base"], f"{commits['base']}^{{commit}}", spec["head"], f"{commits['head']}^{{commit}}"]
    looked.clear()
    same = _start(ctx, input_files=[{"git_range": {**spec, "base": spec["head"]}}])  # one spelling: one observation
    assert same["status"] == "started" and looked.count(spec["head"]) == 1, same
    refused = _start(ctx, input_files=[{"git_range": {**spec, "base": "before", "head": "after^{tree}"}}])
    assert refused["reason"] == "input_files_unavailable" and refused["definitely_unrun"] is True, refused
    assert "head='after^{tree}' did not resolve to a commit" in json.dumps(refused)  # the user's selector, not an id


def test_unattempted_range_products_keep_their_immutable_range_identity(tmp_path, monkeypatch):
    from ouroboros import artifacts

    ctx, _engine = _ctx(tmp_path, monkeypatch)
    commits, _trees = _range_fixture(ctx.repo_dir / "nested")
    real = artifacts.copy_artifact_file

    def flaky(source, destination, *, expected=None):
        if pathlib.Path(source).name == "range.json":
            raise OSError("artifact source changed during read")
        return real(source, destination, expected=expected)

    monkeypatch.setattr(artifacts, "copy_artifact_file", flaky)
    result = _start(ctx, input_files=[{"git_range": {"repository": "nested", "base": "before", "head": "after"}}])
    assert result["reason"] == "input_files_unavailable" and result["definitely_unrun"] is True, result
    failed, *later = result["input_files"]
    assert failed["status"] == "copy_failed" and {row["path"] for row in later} >= {"range.patch"}
    assert all(row["status"] == "not_attempted" for row in later)
    assert all(row["git_range"] == {**commits, "path": ""} for row in (failed, *later))


@pytest.mark.parametrize("fault", ["missing-commit", "protected-deletion", "unsupported-domain", "docker", "presence"])
def test_an_unavailable_range_refuses_the_entire_mixed_selection(tmp_path, monkeypatch, fault):
    from ouroboros.tools import control

    ctx, sid, _drive_dir = _scheduler(tmp_path, monkeypatch)
    repo = ctx.repo_dir / "nested"
    _range_fixture(repo)
    spec = {"base": "before", "head": "after", "repository": "nested", "path": "removed"}
    if fault == "missing-commit":
        spec["head"] = "missing-local-ref"
    elif fault == "protected-deletion":
        ctx.task_contract = {"resource_policy": {"protected_artifacts": [
            {"id": "removed", "role": "black_box_reference", "paths": [str(repo / "removed")]}]}}
    elif fault == "unsupported-domain":
        monkeypatch.setattr("ouroboros.work_domain.managed", lambda: True)
    elif fault == "docker":
        ctx.task_metadata["executor_ref"] = {"type": "docker_exec", "container_name": "fixture",
                                             "workspace_host_path": str(repo), "workspace_backend_path": "/app"}
    else:
        _presence(ctx, ("active_workspace", "."), ("task_drive", "."))
    reply = control._schedule_task(ctx, subagent_id=sid, objective="Read", expected_output="Summary",
                                   input_files=[{"root": "task_drive", "path": "brief.md"}, {"git_range": spec}])
    assert "input_files_unavailable" in reply, reply
    if fault in {"docker", "unsupported-domain"}:
        assert "managed or Docker work domain" in reply
    assert not [row for row in ctx.pending_events if row["type"] == "schedule_subagent"]
    assert not _retained_home(ctx).exists()


def test_a_receiver_cannot_read_a_protected_historic_directory(tmp_path, monkeypatch):
    from ouroboros.delegate_readonly_inputs import InputSelectionError, schedule_selected_inputs

    ctx, _sid, _drive_dir = _scheduler(tmp_path, monkeypatch)
    repo = ctx.repo_dir / "source"
    _range_fixture(repo)
    policy = {"protected_artifacts": [{"id": "old", "role": "black_box_reference", "paths": [str(repo / "gone")]}]}
    with pytest.raises(InputSelectionError) as error:
        schedule_selected_inputs(ctx, [{"git_range": {"repository": "source", "base": "before", "head": "after",
                                                      "path": "gone/child.txt"}}], "new-child", policy)
    assert "RESOURCE_POLICY_BLOCKED" in error.value.results[0]["detail"]


def test_retry_of_a_range_reuses_the_frozen_bytes_and_request(tmp_path, monkeypatch):
    import subprocess
    from ouroboros import claudexor_daemon

    ctx, _ = _ctx(tmp_path, monkeypatch)
    repo = ctx.repo_dir / "nested"
    commits, _ = _range_fixture(repo)
    engine = Recording(fail_first=True)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: engine)
    lost = _start(ctx, input_files=[{"git_range": {"repository": "nested", "base": "before", "head": "after"}}])
    invocation = lost["pending_invocation_id"]
    recorded = custody.invocation_record(ctx.drive_root, invocation)["input_files"]
    subprocess.run(["git", "update-ref", "refs/heads/after", commits["base"]], cwd=repo, check=True)
    retry = _start(ctx, retry_of=invocation, prompt="", subagent_id=None, access=None)
    assert retry["status"] == "started" and retry["idempotent_recovery"]
    assert retry["input_files"] == recorded and engine.requests[0] == engine.requests[1]
    raw = next(pathlib.Path(row["local"]) for row in recorded if row["path"] == "range.patch")
    assert raw.read_bytes() == _expected_range(repo, commits)


def test_text_view_recovers_carriage_returns_across_chunk_boundaries(tmp_path, monkeypatch):
    from ouroboros.delegate_git_range import _text_view
    from ouroboros.tools.registry import ToolRegistry

    ctx, _ = _ctx(tmp_path, monkeypatch)
    patch = ctx.repo_dir / "range.patch"
    raw = b"abc" * 21845 + "ß 中\u2028line\u2029again\u0085\r\n".encode() + b"literal \\x0d\r" + bytes(range(256))
    patch.write_bytes(raw)
    view, info = _text_view(patch)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    output = registry.execute("read_file", {"root": "system_repo", "path": view.name})
    body = "".join(line.split("\t", 1)[1] for line in output.splitlines(keepends=True)[1:])
    assert info["encoding"] == "utf8-byte-escapes-v1" and "ß 中" in body and _unescape_range(body) == raw
    patch.write_bytes("plain UTF-8 ß\n".encode())
    assert _text_view(patch) is None


@pytest.mark.parametrize("root", ["active_workspace", "system_repo"])
@pytest.mark.parametrize(("protected", "path"), [("removed", "removed"), ("archive", "")],
                         ids=["exact-file", "whole-range-with-removed-directory"])
def test_receiver_relative_policy_uses_the_selected_child_folder_too(tmp_path, monkeypatch, root, protected, path):
    from ouroboros import delegate_git_range
    from ouroboros.delegate_readonly_inputs import InputSelectionError, schedule_selected_inputs

    ctx, _, _ = _scheduler(tmp_path, monkeypatch)
    repo = ctx.repo_dir / "source"
    _range_fixture(repo)
    assert not (repo / protected).exists()

    write_names = delegate_git_range._write_diff

    def only_names(repo, args, dest):
        assert args[0] == "--name-status", "policy must settle before producing any patch bytes"
        write_names(repo, args, dest)

    monkeypatch.setattr(delegate_git_range, "_write_diff", only_names)
    policy = {"protected_artifacts": [{"paths": [protected], "deny": ["read_bytes"]}]}
    with pytest.raises(InputSelectionError) as error:
        schedule_selected_inputs(ctx, [{"git_range": {"root": root, "repository": "source", "base": "before", "head": "after",
                                                      "path": path}}], "child", policy, workspace_root=str(repo))
    assert "operation 'read_bytes'" in error.value.results[0]["detail"]
    assert not _retained_home(ctx).exists()


@pytest.mark.parametrize("lane", ["range", "file"])
@pytest.mark.parametrize("participant", ["caller", "scheduled-child"])
def test_each_participants_own_working_folder_anchors_relative_policy_symmetrically(tmp_path, monkeypatch, participant, lane):
    """A selection's admission answers for EVERY participant's own working folder, not a positional flag:
    the caller and the scheduled child each anchor a relative protected path at their own folder, so a
    direct start and a scheduled one refuse symmetrically, an ordinary file as a range. The same contract
    held from another folder names another address, so that selection is admitted (the permitted control)."""
    from ouroboros.delegate_readonly_inputs import InputSelectionError, resolve_selection, schedule_selected_inputs

    ctx, _sid, _drive_dir = _scheduler(tmp_path, monkeypatch)
    own, elsewhere = ctx.repo_dir / "range", ctx.repo_dir / "elsewhere"
    elsewhere.mkdir()
    if lane == "range":
        _range_fixture(own)
        name = "removed"
        entry = {"git_range": {"root": "system_repo", "repository": "range", "base": "before", "head": "after",
                               "path": "removed"}}
    else:
        own.mkdir()
        name, entry = "kept.md", {"root": "system_repo", "path": "range/kept.md"}  # one physical file from any folder
        (own / name).write_bytes(b"kept\n")
    policy = {"protected_artifacts": [{"paths": [name], "deny": ["read_bytes"]}]}  # relative: this folder's
    range_dir = tmp_path / "range-dir"

    def select(folder, child):
        if participant == "caller":  # a direct start: the caller answers for its own folder
            ctx.task_contract, ctx.workspace_root, ctx.workspace_mode = {"resource_policy": policy}, folder, "read_only"
            ctx.task_constraint = {"mode": "local_readonly_subagent"}  # the mode in which a folder inside the repo is a focus
            range_dir.mkdir()
            return resolve_selection(ctx, [entry], range_dir=range_dir)
        return schedule_selected_inputs(ctx, [entry], child, policy, workspace_root=str(folder))

    with pytest.raises(InputSelectionError) as error:
        select(own, "child")
    assert "operation 'read_bytes'" in error.value.results[0]["detail"]
    assert not _retained_home(ctx).exists()
    range_dir.rmdir() if range_dir.exists() else None
    rows = select(elsewhere, "other-child")
    assert [row["path"] for row in rows] == (["range.json", "range.patch"] if lane == "range" else ["range/kept.md"])
    if participant == "scheduled-child":
        assert all(row["status"] == "prepared" and pathlib.Path(row["local"]).is_file() for row in rows)


@pytest.mark.parametrize("participant", ["caller", "scheduled-child"])
def test_a_folderless_participant_anchors_relative_policy_at_its_task_drive(tmp_path, monkeypatch, participant):
    """A folderless task has no workspace_root, yet ``active_repo_dir()`` is its task drive: relative policy anchors
    there, so a protected task-drive path selected through ``root=runtime_data`` is refused before any copy
    while an unprotected sibling is admitted. The receiving reader answers by the same rule."""
    from ouroboros.delegate_readonly_inputs import InputSelectionError, resolve_selection, schedule_selected_inputs
    from ouroboros.tool_access import resource_root_path

    ctx, _sid, drive = _scheduler(tmp_path, monkeypatch)
    ctx.task_metadata["resource_intent"] = {"kind": "explicit_none"}
    assert not ctx.workspace_root and ctx.active_repo_dir() == resource_root_path(ctx, "task_drive")
    # The caller's own drive for a direct start; the receiving child's, which it will read, for a scheduled one.
    owner = "t-parent" if participant == "caller" else "child"
    folder = drive.parent / owner
    folder.mkdir(exist_ok=True)
    (folder / "secret.md").write_bytes(b"protected\n")
    (folder / "open.md").write_bytes(b"sibling\n")
    runtime = f"task_drives/{owner}/"
    policy = {"protected_artifacts": [{"paths": ["secret.md"], "deny": ["read_bytes"]}]}  # relative: the drive's
    if participant == "caller":
        ctx.task_contract = {"resource_policy": policy}

    def select(name):
        entry = {"root": "runtime_data", "path": runtime + name}
        if participant == "caller":
            return resolve_selection(ctx, [entry])
        return schedule_selected_inputs(ctx, [entry], "child", policy)

    with pytest.raises(InputSelectionError) as error:
        select("secret.md")
    assert "operation 'read_bytes'" in error.value.results[0]["detail"]
    assert not _retained_home(ctx).exists()
    rows = select("open.md")
    assert [row["path"] for row in rows] == [runtime + "open.md"]
    if participant == "scheduled-child":
        assert all(row["status"] == "prepared" and pathlib.Path(row["local"]).is_file() for row in rows)


def test_the_range_renderer_takes_its_admission_and_never_imports_the_coordinator():
    """One direction (DEVELOPMENT §3): the coordinator imports the renderer and hands it ``RangeAdmission``."""
    import ast

    from ouroboros import delegate_git_range

    tree = ast.parse(pathlib.Path(delegate_git_range.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update({node.module or "", *(f"{node.module}.{alias.name}" for alias in node.names)})
    assert not [name for name in imported if "delegate_readonly_inputs" in name]


def test_range_presence_ceiling_preserves_the_granted_subdirectory(tmp_path, monkeypatch):
    from ouroboros.delegate_readonly_inputs import InputSelectionError, schedule_selected_inputs

    ctx, _, _ = _scheduler(tmp_path, monkeypatch)
    repo = ctx.repo_dir / "source"
    _range_fixture(repo)
    _presence(ctx, ("active_workspace", "source/removed"), vcs=True)
    spec = {"repository": "source", "base": "before", "head": "after", "path": "removed"}
    rows = schedule_selected_inputs(ctx, [{"git_range": spec}], "allowed-child")
    manifest = next(pathlib.Path(row["local"]) for row in rows if row["path"] == "range.json")
    assert json.loads(manifest.read_bytes())["changes"] == [{"status": "D", "path": "removed"}]
    other = ctx.repo_dir / "other"
    other.mkdir()
    for path, workspace in (("image", ""), (".", ""), ("removed", str(other))):
        with pytest.raises(InputSelectionError) as denied:
            schedule_selected_inputs(ctx, [{"git_range": {**spec, "path": path}}], "other-child",
                                     workspace_root=workspace)
        assert "PRESENCE_RESOURCE_BLOCKED" in denied.value.results[0]["detail"]
    assert not (_retained_home(ctx) / "scheduled-other-child").exists()


def test_a_fixed_vcs_repository_names_one_physical_scope_in_both_reader_folders(tmp_path, monkeypatch):
    from ouroboros import delegate_git_range
    from ouroboros.delegate_readonly_inputs import InputSelectionError, schedule_selected_inputs
    from ouroboros.presence_authority import PresenceArgumentBinding

    ctx, _, _ = _scheduler(tmp_path, monkeypatch)
    repo = ctx.repo_dir / "source"
    _range_fixture(repo)
    other = ctx.repo_dir / "child"
    # Only a second worktree address for the policy check; no Git reads here.
    (other / "source" / ".git").mkdir(parents=True)
    scope = (PresenceArgumentBinding(("repository",), "static", static_value="source"),
             PresenceArgumentBinding(("path",), "static", static_value="removed"))

    def fix(*more):
        _presence(ctx, ("active_workspace", "source/removed"), vcs=True, vcs_bindings=(*scope, *more))

    def argument(name, value):
        return PresenceArgumentBinding((name,), "static", static_value=value)

    entry = {"git_range": {"repository": "source", "base": "before", "head": "after", "path": "removed"}}
    # The range is the whole two-tree patch: vcs_read's diff with every presentation argument at its default.
    # A fixed argument may restate that exact typed default, which restricts nothing; a narrower value, another
    # type or an operand the diff does not take would restrict the reader, so the range still refuses it.
    defaults = (("staged", False), ("stat", False), ("name_only", False), ("max_chars", 0))
    for index, restated in enumerate([*([case] for case in defaults), list(defaults)]):
        fix(*(argument(*case) for case in restated))
        allowed = schedule_selected_inputs(ctx, [entry], f"same-repository-{index}", workspace_root=str(ctx.repo_dir))
        assert {row["source"] for row in allowed} == {str(repo)}, restated
        assert all(row["status"] == "prepared" for row in allowed), restated
    for index, (name, value) in enumerate((
            ("staged", True), ("stat", True), ("name_only", True), ("max_chars", 1),  # narrower
            ("staged", 0), ("stat", 0), ("name_only", "false"), ("max_chars", False), ("max_chars", "0"),  # not typed
            ("max_count", 20), ("unknown", False))):  # not the diff's presentation
        fix(argument(name, value))
        with pytest.raises(InputSelectionError) as restricted:
            schedule_selected_inputs(ctx, [entry], f"restricted-{index}", workspace_root=str(ctx.repo_dir))
        assert "PRESENCE_RESOURCE_BLOCKED" in restricted.value.results[0]["detail"], (name, value)
    fix()
    monkeypatch.setattr(delegate_git_range, "materialize_range", lambda *_a: pytest.fail("range not admitted"))
    with pytest.raises(InputSelectionError) as refused:
        schedule_selected_inputs(ctx, [entry], "different-repository", workspace_root=str(other))
    assert "PRESENCE_RESOURCE_BLOCKED" in refused.value.results[0]["detail"]
    assert not (_retained_home(ctx) / "scheduled-different-repository").exists()


def test_the_git_reported_toplevel_must_be_the_admitted_worktree_itself(tmp_path, monkeypatch):
    """Identity, not spelling: another name for the admitted directory passes, any other directory
    (or one whose identity cannot be read) refuses, and only this one comparison changes."""
    import stat
    import subprocess

    from ouroboros import delegate_git_range
    from ouroboros.tools import vcs_read

    worktree = tmp_path / "Worktree"
    (worktree / "inner").mkdir(parents=True)
    source = (SimpleNamespace(repository_path=worktree, root="active_workspace"), [])

    def accepted(*_args, **_kwargs):
        raise LookupError("past the toplevel check")

    monkeypatch.setattr(vcs_read, "_object", accepted)

    def check(reported):
        done = subprocess.CompletedProcess([], 0, os.fsencode(reported) + b"\n", b"")
        monkeypatch.setattr(vcs_read, "_capture_read_git", lambda *_a: done)
        return delegate_git_range.materialize_range((), source, {"base": "one", "head": "two"}, tmp_path / "out", None)

    names = [worktree]
    try:
        (tmp_path / "alias").symlink_to(worktree, target_is_directory=True)
        names.append(tmp_path / "alias")
    except OSError:
        pass  # a host that cannot link keeps the rest
    if (tmp_path / "worktree").exists():  # a case-insensitive volume: a spelling no resolve() can canonicalize
        names.append(tmp_path / "worktree")
    for name in names:
        with pytest.raises(LookupError, match="past the toplevel check"):
            check(name)
    for other in (worktree / "inner", tmp_path / "missing", tmp_path):
        with pytest.raises(ValueError, match="must name the Git worktree root"):
            check(other)
    assert not (tmp_path / "out").exists()

    real_stat = os.stat
    with monkeypatch.context() as patched:  # an identity that cannot be read proves nothing
        def unreadable(path, *args, **kwargs):
            if str(path) == str(worktree / "inner"):
                raise PermissionError("not readable")
            return real_stat(path, *args, **kwargs)

        patched.setattr(os, "stat", unreadable)
        assert delegate_git_range._same_directory(worktree, worktree) is True
        assert delegate_git_range._same_directory(worktree, worktree / "inner") is False
        blank = os.stat_result((stat.S_IFDIR | 0o755, 0, 7, 1, 0, 0, 0, 0, 0, 0))  # no inode: not an identity
        patched.setattr(os, "stat", lambda *_a, **_k: blank)
        assert delegate_git_range._same_directory(worktree, worktree / "inner") is False


def test_ordinary_input_respects_presence_in_the_receiving_workspace(tmp_path, monkeypatch):
    from ouroboros.tools import control

    ctx, selected, _ = _scheduler(tmp_path, monkeypatch, kind="api_model")
    nested = ctx.repo_dir / "nested"
    nested.mkdir()
    (nested / "one.md").write_bytes(b"nested brief\n")
    (ctx.repo_dir / "other.md").write_bytes(b"outside the starting folder\n")
    options = {"subagent_id": selected, "objective": "Read", "expected_output": "Summary",
               "workspace_root": str(nested)}
    nested_file = {"root": "active_workspace", "path": "nested/one.md"}
    inherited_file = {"root": "active_workspace", "path": "other.md"}
    grants = (("active_workspace", "nested/one.md"), ("active_workspace", "other.md"))
    _presence(ctx, *grants)
    refused = control._schedule_task(ctx, **options, input_files=[nested_file])
    assert "PRESENCE_RESOURCE_BLOCKED" in refused and not ctx.pending_events, refused
    assert not _retained_home(ctx).exists()

    _presence(ctx, *grants, ("active_workspace", "one.md"))
    refused = control._schedule_task(ctx, **options, input_files=[nested_file, inherited_file])
    assert "PRESENCE_RESOURCE_BLOCKED" in refused, refused
    rows = json.loads(refused.partition("input_files: ")[2])  # each entry keeps its own status
    assert [(row["path"], row["status"]) for row in rows] == [("nested/one.md", "checked"), ("other.md", "denied")]
    assert not ctx.pending_events and not _retained_home(ctx).exists()

    _presence(ctx, ("active_workspace", "."))  # ordinary inherited reads remain available
    reply = control._schedule_task(ctx, **options, input_files=[nested_file, inherited_file])
    event, = ctx.pending_events
    assert event["workspace_root"] == str(nested)
    assert [pathlib.Path(row["local"]).read_bytes() for row in event["prepared_input_files"]] == [
        b"nested brief\n", b"outside the starting folder\n"], reply


@pytest.mark.parametrize("ranged", [False, True])
@pytest.mark.parametrize("principal", ["caller", "child"])
def test_selected_copy_obeys_copy_only_policy_before_retention(tmp_path, monkeypatch, ranged, principal):
    from ouroboros import delegate_git_range as git_range, delegate_readonly_inputs as inputs

    ctx, _, _ = _scheduler(tmp_path, monkeypatch)
    root = ctx.repo_dir / "fixture"
    policy = {"protected_artifacts": [{"id": "fixture-policy", "paths": [str(root / "removed")], "deny": ["copy"]}]}
    if principal == "caller":
        ctx.task_contract = {"resource_policy": policy}
    if ranged:
        _range_fixture(root)
        # Only give the host a harmless name/status record. No patch extraction on this denied case.
        def names(_repo, args, dest):
            assert args[0] == "--name-status"
            dest.write_bytes(b"D\0removed\0")

        monkeypatch.setattr(git_range, "_write_diff", names)
        entry = {"git_range": {"repository": root.name, "base": "before", "head": "after"}}
    else:
        root.mkdir()
        (root / "removed").write_text("fixture only", encoding="utf-8")
        entry = {"root": "system_repo", "path": "fixture/removed"}
    monkeypatch.setattr(inputs, "copy_selection", lambda *_a: pytest.fail("copy started after refusal"))
    with pytest.raises(inputs.InputSelectionError) as exc:
        inputs.schedule_selected_inputs(ctx, [entry], "child", policy if principal == "child" else None)
    assert "operation 'copy'" in exc.value.results[0]["detail"]
    assert not _retained_home(ctx).exists()


def test_unchanged_protected_sibling_is_not_part_of_range_content_admission(tmp_path, monkeypatch):
    from ouroboros.delegate_readonly_inputs import schedule_selected_inputs

    ctx, _, _ = _scheduler(tmp_path, monkeypatch)
    root = ctx.repo_dir / "fixture"
    commits, _ = _range_fixture(root, unchanged=(b"unchanged.txt",))
    policy = {"protected_artifacts": [{"paths": [str(root / "unchanged.txt")], "deny": ["read_bytes", "copy"]}]}
    rows = schedule_selected_inputs(ctx, [{"git_range": {"repository": root.name, "base": "before", "head": "after"}}],
                                    "child", policy)
    patch = next(pathlib.Path(row["local"]) for row in rows if row["path"] == "range.patch")
    assert patch.read_bytes() == _expected_range(root, commits) and b"unchanged.txt" not in patch.read_bytes()


@pytest.mark.parametrize("refusal", ["redirected", "protected-destination", "receiving-drive"])
def test_unreadable_copy_address_refuses_before_publication_for_native_child(tmp_path, monkeypatch, refusal):
    from ouroboros.presence_authority import PresenceArgumentBinding
    from ouroboros.tools import control

    ctx, selected, drive = _scheduler(tmp_path, monkeypatch, kind="api_model")
    source = str(drive / "brief.md")
    bindings = [PresenceArgumentBinding(("root",), "static", static_value="runtime_data")]
    if refusal in {"redirected", "receiving-drive"}:
        if refusal == "receiving-drive":
            source = (drive / "brief.md").relative_to(ctx.drive_root).as_posix()
        bindings.append(PresenceArgumentBinding(("path",), "static", static_value=source))
        reason = "PRESENCE_RESOURCE_BLOCKED"
    else:
        ctx.task_contract = {"resource_policy": {"protected_artifacts": [
            {"paths": [str(_retained_home(ctx))], "deny": ["read_bytes"]}]}}
        reason = "RESOURCE_POLICY_BLOCKED"
    _presence(ctx, ("runtime_data", "."), read_bindings=tuple(bindings))
    reply = control._schedule_task(ctx, subagent_id=selected, objective="Read input", expected_output="Summary",
                                   input_files=[{"root": "runtime_data", "path": source}])
    assert reason in reply and '"status": "denied"' in reply, reply
    assert not ctx.pending_events and not any(path.is_file() for path in _retained_home(ctx).rglob("*"))


def test_ambiguous_range_ref_is_not_silently_resolved_after_engine_discovery(tmp_path, monkeypatch):
    import subprocess

    from ouroboros.delegate_evidence import START_BLOCKED

    ctx, engine = _ctx(tmp_path, monkeypatch)
    repo = ctx.repo_dir / "fixture"
    commits, _ = _range_fixture(repo)
    subprocess.run(["git", "tag", "before", commits["head"]], cwd=repo, check=True)
    subprocess.run(["git", "config", "core.warnAmbiguousRefs", "false"], cwd=repo, check=True)
    result = _start(ctx, input_files=[{"git_range": {"repository": "fixture", "base": "before", "head": "after"}}])
    assert result["reason"] == "input_files_unavailable" and result["definitely_unrun"] and result["engine_discovery_completed"]
    assert "ambiguous" in result["input_files"][0]["detail"] and not engine.requests
    rows = list(custody._iter_rows(custody.event_log_path(ctx.drive_root)))
    assert [row["reason"] for row in rows if row["type"] == START_BLOCKED] == [result["reason"]]
    assert not any(row["type"] == custody.START_REQUESTED for row in rows)


@pytest.mark.parametrize("record", [b"R100\0one\0two\0", b"M\0one", b"D\0\0"])
def test_inconsistent_name_status_never_reaches_patch_extraction(tmp_path, monkeypatch, record):
    from ouroboros import delegate_git_range as git_range, delegate_readonly_inputs as inputs

    ctx, _, _ = _scheduler(tmp_path, monkeypatch)
    _range_fixture(ctx.repo_dir / "fixture")

    def names(_repo, args, dest):
        assert args[0] == "--name-status"
        dest.write_bytes(record)

    monkeypatch.setattr(git_range, "_write_diff", names)
    with pytest.raises(inputs.InputSelectionError) as exc:
        inputs.schedule_selected_inputs(ctx, [{"git_range": {"repository": "fixture", "base": "before", "head": "after"}}],
                                        "child")
    assert "name-status" in exc.value.results[0]["detail"] and not _retained_home(ctx).exists()


def test_a_scheduled_reader_has_the_serving_body_its_worker_binds_beside_a_parents_candidate(tmp_path, monkeypatch):
    """A parent authoring a body candidate moved its own repo_dir/system_repo_dir (``body_candidate.bind``); its
    scheduled child's worker binds the serving body (``restore`` skips subagents). The receiver is checked in the
    child's roots, with the parent's contract from either carrier under the child's policy and the child's own
    resource intent; a source passes only where both readers name the same file."""
    from ouroboros.delegate_readonly_inputs import InputSelectionError, receiving_reader, schedule_selected_inputs
    from ouroboros.tools.tool_resolution import system_repo_dir_for

    ctx, _selected, _drive = _scheduler(tmp_path, monkeypatch)
    serving, candidate = ctx.repo_dir, tmp_path / "candidate"
    candidate.mkdir()
    for root, text in ((serving, b"serving\n"), (candidate, b"candidate\n")):
        (root / "same.md").write_bytes(text)
    [agreed] = schedule_selected_inputs(ctx, [{"root": "system_repo", "path": "same.md"}], "unbound")
    assert pathlib.Path(agreed["local"]).read_bytes() == b"serving\n"

    ctx.serving_repo_dir, ctx.repo_dir, ctx.system_repo_dir = serving, candidate, candidate
    ctx.task_metadata["task_contract"] = {"objective": "parent", "resource_policy": {"allowed_origins": []}}
    policy, intent = {"allowed_origins": ["https://example.org"]}, {"kind": "explicit_resource", "root": str(candidate)}
    reader = receiving_reader(ctx, policy, workspace_root=str(candidate), child_task_id="c1", resource_intent=intent)
    assert system_repo_dir_for(reader) == pathlib.Path(reader.repo_dir) == serving and reader.serving_repo_dir is None
    assert reader.task_contract == reader.task_metadata["task_contract"] == {"objective": "parent", "resource_policy": policy}
    assert reader.task_metadata["resource_intent"] == intent and ctx.task_contract == {}
    [handed] = schedule_selected_inputs(ctx, [{"root": "active_workspace", "path": "same.md"}], "candidate-folder",
                                        workspace_root=str(candidate), resource_intent=intent)
    assert pathlib.Path(handed["local"]).read_bytes() == b"candidate\n"
    with pytest.raises(InputSelectionError) as refused:
        schedule_selected_inputs(ctx, [{"root": "system_repo", "path": "same.md"}], "other-body",
                                 workspace_root=str(candidate), resource_intent=intent)
    assert refused.value.results[0]["status"] == "denied", refused.value.results
    assert not (_retained_home(ctx) / "scheduled-other-body").exists()


@pytest.mark.skipif(os.name != "posix", reason="a padded historic name is a literal directory entry on POSIX")
@pytest.mark.parametrize("static, admitted", [("", True), ("native", False)], ids=["inert-empty", "real-selector"])
def test_an_inert_empty_selector_keeps_a_padded_historic_name_literal(tmp_path, monkeypatch, static, admitted):
    """The inert static selector is dropped before the historical resolver is chosen, so a POSIX name ending in a
    space stays literal; a nonempty selector still changes what is read and refuses."""
    from ouroboros.presence_authority import PresenceArgumentBinding

    ctx, engine = _ctx(tmp_path, monkeypatch)
    _range_fixture(ctx.repo_dir / "nested")
    _presence(ctx, ("active_workspace", "."), ("system_repo", "."), vcs=True,
              read_bindings=(PresenceArgumentBinding(("bucket",), "static", static_value=static),))
    result = _start(ctx, input_files=[{"git_range": {"repository": "nested", "base": "before", "head": "after",
                                                      "path": "padded-name "}}])
    if not admitted:
        assert result["definitely_unrun"] is True and "PRESENCE_RESOURCE_BLOCKED" in json.dumps(result), result
        assert engine.requests == []
        return
    assert result["status"] == "started", result
    manifest = next(pathlib.Path(row["local"]) for row in result["input_files"] if row["path"] == "range.json")
    assert json.loads(manifest.read_bytes())["changes"] == [{"status": "A", "path": "padded-name "}]


@pytest.mark.skipif(os.name != "posix", reason="a padded historic name is a literal directory entry on POSIX")
def test_a_root_only_fixed_grant_keeps_the_padded_historic_name_literal(tmp_path, monkeypatch):
    """A Presence grant may fix a read root without a path: that addresses the source's
    root, never its spelling. The padded historical name stays literal instead of
    passing the public reader's dispatch grammar (which strips it into another target),
    while an ordinary selected file with a space in the same root keeps that grammar."""
    from ouroboros.presence_authority import PresenceArgumentBinding

    ctx, engine = _ctx(tmp_path, monkeypatch)
    _range_fixture(ctx.repo_dir / "nested")
    (ctx.repo_dir / "file name.md").write_bytes(b"spaced\n")
    _presence(ctx, ("active_workspace", "."), ("system_repo", "."), vcs=True,
              read_bindings=(PresenceArgumentBinding(("root",), "static", static_value="active_workspace"),))
    result = _start(ctx, input_files=[
        {"root": "active_workspace", "path": "file name.md"},
        {"git_range": {"repository": "nested", "base": "before", "head": "after", "path": "padded-name "}},
    ])
    assert result["status"] == "started", result
    manifest = next(pathlib.Path(row["local"]) for row in result["input_files"] if row["path"] == "range.json")
    assert json.loads(manifest.read_bytes())["changes"] == [{"status": "A", "path": "padded-name "}]
    kept = next(pathlib.Path(row["local"]) for row in result["input_files"] if row["path"] == "file name.md")
    assert kept.read_bytes() == b"spaced\n"


def test_a_diagnostic_git_prints_either_way_is_not_ambiguity(tmp_path, monkeypatch):
    """A repository whose Git prints the same unrelated warning on every read (a deprecated setting) still resolves
    an unambiguous local revision; a genuinely ambiguous one still refuses."""
    import subprocess

    from ouroboros.delegate_readonly_inputs import InputSelectionError, schedule_selected_inputs
    from ouroboros.tools import vcs_read

    ctx, _, _ = _scheduler(tmp_path, monkeypatch)
    repo = ctx.repo_dir / "fixture"
    commits, _ = _range_fixture(repo)
    capture = vcs_read._capture_read_git

    looked = []

    def deprecated(path, args):
        done = capture(path, args)
        if args[-1] in {"before", "after"}:
            looked.append(args[-1])
        # The measured real benign form (git 2.53): a deprecated setting warns on every read.
        return subprocess.CompletedProcess(done.args, done.returncode, done.stdout,
                                           b"warning: core.fsyncObjectFiles is deprecated; use core.fsync instead\n"
                                           + done.stderr)

    monkeypatch.setattr(vcs_read, "_capture_read_git", deprecated)
    entry = {"git_range": {"repository": "fixture", "base": "before", "head": "after"}}
    rows = schedule_selected_inputs(ctx, [entry], "warned")
    manifest = next(pathlib.Path(row["local"]) for row in rows if row["path"] == "range.json")
    assert json.loads(manifest.read_bytes())["commits"] == commits
    assert looked == ["before", "after"]  # one observation per ref
    subprocess.run(["git", "tag", "before", commits["head"]], cwd=repo, check=True)
    with pytest.raises(InputSelectionError) as refused:
        schedule_selected_inputs(ctx, [entry], "ambiguous")
    assert "ambiguous" in refused.value.results[0]["detail"], refused.value.results


def test_every_kind_of_helper_answers_with_the_one_receipt(tmp_path, monkeypatch):
    """c50-1 (owner item 50): a native child, a session child under its nanny and a direct start answer
    through the one builder, in one vocabulary: differences first, then one line per standing of fact."""
    from ouroboros import helper_receipt
    from ouroboros.tools import control, delegate

    built, real = [], helper_receipt.build
    assert delegate.build_receipt is real, "the direct start's name is the same builder"
    spy = lambda facts: built.append(dict(facts)) or real(facts)  # noqa: E731
    monkeypatch.setattr(helper_receipt, "build", spy)
    monkeypatch.setattr(delegate, "build_receipt", spy)
    answers = []
    for kind in ("api_model", "agent_session"):
        ctx, selected, _drive_dir = _scheduler(tmp_path / kind, monkeypatch, kind=kind)
        answers.append(control._schedule_task(ctx, subagent_id=selected, objective="Read", expected_output="Summary",
                                              input_files=[{"root": "task_drive", "path": "brief.md"}]))
    (tmp_path / "direct").mkdir()
    ctx, _engine = _ctx(tmp_path / "direct", monkeypatch)
    started = _start(ctx, input_files=_sources(ctx))
    answers.append(started["note"])
    assert started["note"].startswith("receipt:"), "a durable start has no custody warning ahead of its receipt"
    assert len(built) == 3, built
    for answer in answers:
        receipt = answer[answer.index("receipt:"):].split("\n")
        stages = [row.split(":", 1)[0] for row in receipt if row.split(":", 1)[0] in ("requested", "prepared", "observed")]
        assert stages == sorted(set(stages), key=("requested", "prepared", "observed").index), receipt
        assert "prepared is not read" in answer, answer
        assert row_count(receipt) == len(receipt), receipt
    # Only a session's applied access waits for the engine; a native child's profile is the host's own, known now.
    assert ["applied: pending engine evidence" in answer for answer in answers] == [False, True, True]
    assert "room, origin and marked rows read at its start" in answers[0]
    assert "read at its start" not in answers[1], "a session's work order carries its rows by value"
    # The session child and the direct start both lie outside a session's folder; the native child reads its copies.
    assert ["outside the session's folder" in answer for answer in answers] == [False, True, True]
    assert "evidence-only private copy" not in answers[1], "scheduling promises no copy a later start may not make"
    assert started["input_files"] and all(row["carrier"] == "outside_scope_path" for row in started["input_files"])


def row_count(receipt):
    """Every receipt line is the head, a difference ending in its standing, or a standing's own line."""
    stages = ("requested: ", "prepared: ", "observed: ")
    return sum(1 for row in receipt if row.startswith("receipt:") or row.startswith(stages)
               or row.endswith(("[requested]", "[prepared]", "[observed]")))


@pytest.mark.parametrize("view,named", [({"marks": "everything"}, True), ({"knowledge": True}, False)],
                         ids=["unknown-value", "known-value"])
def test_a_memory_view_value_the_view_does_not_know_is_named_at_scheduling(tmp_path, monkeypatch, view, named):
    """c50-2: the view refuses an unknown value at the child's start and uses the role default; the parent learns
    it from the schedule receipt, with that default, while a known value reads as nothing differing."""
    from ouroboros.tools import control

    ctx, selected, _drive_dir = _scheduler(tmp_path, monkeypatch, kind="api_model")
    answer = control._schedule_task(ctx, subagent_id=selected, objective="Read", expected_output="Summary",
                                    memory_view=view)
    assert ctx.pending_events, answer  # the request is queued either way: nothing here refuses it
    unknown = ("memory_view {'marks': 'everything'} not used (marks is one of all, room_and_global, none): "
               "the child default stands [requested]")
    assert (unknown in answer) is named and ("receipt:\nrequested: " in answer) is not named
    assert ("memory view: child default with knowledge=True" in answer) is not named
