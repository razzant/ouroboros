"""A Git writer its starter puts directly in its own write root: ``delegate_start(workspace='in_place')``.

Owner decision 25.10 (Q2-A): a delegated Git writer gets a private copy by default; its starter
may instead let it work in place, in the task's own write root. Its edits land there as it makes
them: no copy, no capture, nothing to apply or reject (undo is through Git), and the start receipt
and the custody record say so. Every other shape refuses the choice typed and unrun ($0): a skill
payload, an investigation, a read-only or ask/plan run, folder options, a non-Git write root and a
continuation of an undisposed copy. ``schedule_subagent(workspace=...)`` carries the same choice
to a configured session's first start. Only the engine is a double.
"""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.delegate_directory import in_place_refusal, is_in_place_run
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from tests._delegated_transport_shared import (  # noqa: F401 -- autouse transport fixture
    _owned_gateway_uses_each_test_transport,
)
from tests.test_delegate_continuation import _settle, _start

IN_PLACE = {"workspace_kind": "git", "strategy": "in_place"}
REFUSED = "workspace_in_place_unavailable"


def _records(tmp_path):
    """This test's private copies (the process data root is shared by the session)."""
    from ouroboros import subagent_worktrees

    mine = (tmp_path / "snap_root").resolve()
    return [row for row in subagent_worktrees._records(op="test")
            if row.get("kind") == subagent_worktrees._KIND_DELEGATED_EXEC
            and pathlib.Path(str(row.get("path") or "")).resolve().is_relative_to(mine)]


def _target(ctx) -> pathlib.Path:
    return pathlib.Path(ctx.workspace_root).resolve()


# --------------------------------------------------------------------------- the shape rule

@pytest.mark.parametrize("facts,refused", [
    ({}, ""),                                                     # an acting Git writer: allowed
    ({"selector_root": "/payloads/alpha"}, "skill payload"),
    ({"investigation": True}, "evidence-only investigation"),
    ({"access": "readonly"}, "read-only run"),
    ({"mode": "ask"}, "read-only run"), ({"mode": "plan"}, "read-only run"),
    ({"strategy": "copy", "scope_paths": ["."]}, "directory options"),
    ({"strategy": "direct"}, ""),                                  # the documented default is no option
])
def test_only_a_git_writer_may_work_in_place(facts, refused):
    authority = SimpleNamespace(investigation=facts.get("investigation", False),
                                access=facts.get("access", "workspace_write"))
    refusal = in_place_refusal(authority, facts.get("selector_root", ""), "in_place", facts.get("mode"),
                               facts.get("strategy"), facts.get("scope_paths"))
    if not refused:
        assert refusal is None
        return
    payload = json.loads(refusal.text)
    assert payload["reason"] == REFUSED and payload["definitely_unrun"] is True
    assert refused in payload["detail"] and "Nothing was started" in payload["detail"]


@pytest.mark.parametrize("workspace", [None, "", "copy"])
def test_the_default_copy_is_never_refused(workspace):
    authority = SimpleNamespace(investigation=True, access="readonly")  # even the narrowest shape
    assert in_place_refusal(authority, "/payloads/alpha", workspace, "ask", "copy", ["."]) is None


def test_an_unknown_place_is_refused_by_name():
    authority = SimpleNamespace(investigation=False, access="workspace_write")
    payload = json.loads(in_place_refusal(authority, "", "elsewhere", None, None, None).text)
    assert payload["reason"] == REFUSED and "must be 'copy' or 'in_place'" in payload["detail"]


def test_one_predicate_reads_custody_requests_and_invocations():
    assert is_in_place_run({"resource_ref": IN_PLACE})
    assert is_in_place_run(custody.RunCustody(run_id="r", task_id="t", route_id="x", resource_ref=dict(IN_PLACE)))
    for other in ({}, {"resource_ref": {"workspace_kind": "directory", "strategy": "direct"}},
                  {"resource_ref": {"workspace_kind": "git", "strategy": "copy"}}, {"resource_ref": "in_place"}):
        assert not is_in_place_run(other)


# --------------------------------------------------------------------------- the start

@pytest.mark.parametrize("workspace", [None, "copy", "in_place"])
def test_an_in_place_start_works_in_the_write_root_and_records_the_choice(tmp_path, monkeypatch, workspace):
    options = {} if workspace is None else {"workspace": workspace}
    request, payload, ctx, calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-place", start_kwargs=options)
    assert payload["status"] == "started" and len(calls) == 1, payload
    target, entry = _target(ctx), custody.replay(tmp_path)["run-place"]
    if workspace != "in_place":  # the other direction: the 11.08 default, a private copy
        assert payload["workspace"] == "copy" and len(_records(tmp_path)) == 1 and entry.snapshot_id
        assert pathlib.Path(request["execution"]["workspaceRoot"]).resolve() != target
        assert not is_in_place_run(entry) and "IN-PLACE" not in request["instructions"]
        return
    execution = request["execution"]
    assert {**execution, "workspaceRoot": ""} == {"isolation": "live", "delegated": True, "workspaceRoot": ""}
    assert pathlib.Path(execution["workspaceRoot"]).resolve() == target  # the engine works in the write root itself
    assert pathlib.Path(request["scope"]["root"]).resolve() == target
    assert "DELEGATED IN-PLACE BINDING" in request["instructions"]
    assert "never reset, clean, stash, check out or commit" in request["instructions"]
    assert payload["workspace"] == "in_place" and "baseline_id" not in payload
    assert pathlib.Path(payload["execution_root"]).resolve() == pathlib.Path(payload["authority_target_root"]).resolve() == target
    assert "DIRECTLY in your write root" in payload["note"] and "undo through git" in payload["note"]
    assert _records(tmp_path) == [] and not (tmp_path / "snap_root").exists()  # no copy was made
    assert entry.snapshot_id == "" and entry.resource_ref == IN_PLACE
    assert pathlib.Path(entry.execution_root).resolve() == pathlib.Path(entry.target_root).resolve() == target


@pytest.mark.parametrize("case,acting,kwargs", [
    ("ask", True, {"start_kwargs": {"workspace": "in_place", "mode": "ask"}}),
    ("plan", True, {"start_kwargs": {"workspace": "in_place", "mode": "plan"}}),
    ("readonly", True, {"access": "readonly", "start_kwargs": {"workspace": "in_place"}}),
    ("folder", True, {"start_kwargs": {"workspace": "in_place", "directory_strategy": "copy", "scope_paths": ["."]}}),
    ("investigation", False, {"start_kwargs": {"workspace": "in_place"}}),  # a task without write authority
])
def test_the_wired_refusals_are_typed_unrun_and_send_nothing(tmp_path, monkeypatch, case, acting, kwargs):
    request, payload, _ctx, calls = _start(tmp_path, monkeypatch, acting=acting, run_id=f"run-{case}", **kwargs)
    assert request is None and calls == [], payload
    assert payload["reason"] == REFUSED and payload["definitely_unrun"] is True, payload
    assert _records(tmp_path) == []  # refused before any copy
    if case == "investigation":
        assert "evidence-only investigation" in payload["detail"]


def test_a_non_git_write_root_is_refused_toward_the_folder_strategy(tmp_path, monkeypatch):
    from ouroboros.force_remove import force_rmtree
    from tests import test_delegate_continuation as continuation_tests

    original = continuation_tests._delegating_ctx

    def plain_folder(tmp, **kwargs):
        ctx = original(tmp, **kwargs)
        git_dir = pathlib.Path(ctx.workspace_root) / ".git"
        force_rmtree(git_dir)  # Git's object files are read-only, which Windows refuses to unlink
        assert not git_dir.exists(), "the write root is an ordinary folder now"
        return ctx

    monkeypatch.setattr(continuation_tests, "_delegating_ctx", plain_folder)
    request, payload, _ctx, calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-plain",
                                           start_kwargs={"workspace": "in_place"})
    assert request is None and calls == [] and payload["reason"] == REFUSED, payload
    assert "not a Git working tree" in payload["detail"] and "directory_strategy='direct'" in payload["detail"]


def test_a_continuation_of_an_undisposed_copy_stays_in_that_copy(tmp_path, monkeypatch):
    first, _payload, ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-copy")
    _settle(tmp_path, "run-copy", ctx.task_id)
    request, payload, _ctx, calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-next",
                                           start_kwargs={"continue_from": "run-copy", "workspace": "in_place"})
    assert request is None and calls == [] and payload["reason"] == REFUSED, payload
    assert "resumes an undisposed private copy" in payload["detail"]
    pred = custody.replay(tmp_path)["run-copy"]
    assert not pred.superseded_by and pred.patch_disposed == ""  # nothing was claimed
    assert pathlib.Path(first["execution"]["workspaceRoot"]).exists()
    # The other direction: the same continuation without the choice resumes in the copy.
    request, payload, _ctx, calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-next",
                                           start_kwargs={"continue_from": "run-copy"})
    assert payload["status"] == "started" and payload["continuation"]["workspace"] == "same_snapshot", payload


def test_a_continuation_of_an_in_place_run_is_told_its_edits_are_already_there(tmp_path, monkeypatch):
    _start(tmp_path, monkeypatch, acting=True, run_id="run-first", start_kwargs={"workspace": "in_place"})
    _settle(tmp_path, "run-first", "t-nanny-write")
    request, payload, ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-next",
                                           start_kwargs={"continue_from": "run-first", "workspace": "in_place"})
    assert payload["status"] == "started" and payload["continuation"]["workspace"] == "fresh", payload
    assert payload["continuation"]["prior_patch_disposition"] == "not_applicable"
    assert "It wrote DIRECTLY into the target" in request["prompt"]
    assert pathlib.Path(request["execution"]["workspaceRoot"]).resolve() == _target(ctx) and _records(tmp_path) == []


# --------------------------------------------------------------------------- its disposable scratch

def _in_place_owners(ctx) -> pathlib.Path:
    from ouroboros.workspace_preflight import SESSION_SCRATCH_DIRNAME

    return pathlib.Path(ctx.drive_root) / "task_drives" / ctx.task_id / SESSION_SCRATCH_DIRNAME


def test_a_full_access_in_place_run_is_named_its_own_disposable_scratch_and_granted_nothing(tmp_path, monkeypatch):
    """The host's own folder must stay free of environments, so the run is told, before it starts, the
    one place its starting task's drive owns for them, and what happens there; the request is unchanged."""
    from ouroboros.delegate_start_instructions import DISPOSABLE_SCRATCH, scratch_fact
    from tests.test_continuation_fresh_snapshot_binding import _GRANT_ROW, _ctx, _leg

    ctx = _ctx(tmp_path, "git")
    request, payload = _leg(tmp_path, monkeypatch, ctx, "run-1", prompt="edit here", access="full", row=_GRANT_ROW,
                            workspace="in_place")
    assert payload["workspace"] == "in_place" and request["access"] == "full"
    assert "auxiliaryWritableRoots" not in request["execution"]  # ownership, never a new grant
    held = custody.replay(tmp_path)["run-1"]
    [record] = _in_place_owners(ctx).glob("*/scratch.json")
    assert record.parent.name == f"in-place-{held.invocation_id}"
    scratch = pathlib.Path(json.loads(record.read_text(encoding="utf-8"))["path"])
    assert scratch.is_dir() and not any(scratch.iterdir()) and _target(ctx) not in scratch.parents
    binding = request["instructions"].split("\n\nDELEGATED IN-PLACE BINDING", 1)[1]
    assert f"outside every Git working tree, in {scratch}, this run's disposable scratch, {DISPOSABLE_SCRATCH}." in binding
    assert scratch_fact() not in request["instructions"]


def test_a_workspace_write_in_place_run_keeps_its_rights_and_hears_the_unowned_default(tmp_path, monkeypatch):
    """Owning a scratch for it would need a grant it never had, so nothing is made or sent: an explicit gap."""
    from ouroboros.delegate_start_instructions import scratch_fact
    from tests.test_continuation_fresh_snapshot_binding import _GRANT_ROW, _ctx, _leg

    ctx = _ctx(tmp_path, "git")
    request, payload = _leg(tmp_path, monkeypatch, ctx, "run-1", prompt="edit here", access="workspace_write",
                            row=_GRANT_ROW, workspace="in_place")
    assert payload["workspace"] == "in_place" and "auxiliaryWritableRoots" not in request["execution"]
    assert f"outside every Git working tree, {scratch_fact()}." in request["instructions"]
    assert not _in_place_owners(ctx).exists()


def test_a_refused_in_place_start_prepares_no_scratch(tmp_path):
    from ouroboros.delegate_directory import in_place_start

    def prepared():
        raise AssertionError("a refused start must not make a scratch")

    (tmp_path / "repo" / ".git").mkdir(parents=True)
    for adopted, root in ((object(), tmp_path / "repo"), (None, tmp_path)):  # an undisposed copy; no Git tree
        snapshot, refusal, _ref, binding = in_place_start(adopted, str(root), prepared)
        assert snapshot is None and json.loads(refusal.text)["reason"] == REFUSED and binding == ""


# --------------------------------------------------------------------------- the terminal side

def _finished_in_place(tmp_path, monkeypatch):
    _request, payload, ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-place",
                                            start_kwargs={"workspace": "in_place"})
    assert payload["status"] == "started", payload
    (_target(ctx) / "README.md").write_text("the child's edit, made in place\n", encoding="utf-8")
    _settle(tmp_path, "run-place", ctx.task_id, state="succeeded", reason="")
    return ctx, custody.replay(tmp_path)["run-place"]


def test_nothing_is_captured_owed_or_held_for_an_in_place_run(tmp_path, monkeypatch):
    from ouroboros.delegate_capture_attempt import capture_required, named_required
    from ouroboros.delegate_collection import capture_projection
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive

    ctx, entry = _finished_in_place(tmp_path, monkeypatch)
    target = _target(ctx)
    assert capture_required(entry) is False and named_required(entry) is False
    block = capture_terminal_patch_for_drive(tmp_path, entry)
    assert (block["status"], block["disposition"], block["patch_artifact"]) == ("in_place", "in_place", None)
    assert pathlib.Path(block["execution_root"]).resolve() == target
    assert "already there" in block["note"] and "nothing to apply or reject" in block["note"]
    assert capture_projection(tmp_path, entry) == block
    assert not custody.delegated_capture_dir(tmp_path, entry.task_id, custody.capture_key(entry)).exists()
    assert custody.undisposed_patches(tmp_path) == [] and entry.snapshot_id not in custody.open_snapshot_ids(tmp_path)
    assert (target / "README.md").read_text(encoding="utf-8") == "the child's edit, made in place\n"
    # The other direction: a copy run of the same task owes its capture.
    _start(tmp_path, monkeypatch, acting=True, run_id="run-copy")
    _settle(tmp_path, "run-copy", ctx.task_id)
    assert capture_required(custody.replay(tmp_path)["run-copy"]) is True


def test_a_background_collection_requests_no_capture_job(tmp_path, monkeypatch):
    from ouroboros import delegate_collection as body
    from ouroboros import delegate_custody_current as current
    from ouroboros import maintenance
    from tests.test_delegate_body_collection import terminal

    requests = []
    monkeypatch.setattr(maintenance.child, "request", lambda: requests.append(True))
    ctx, entry = _finished_in_place(tmp_path, monkeypatch)
    result = body.collect_terminal(tmp_path, entry.run_id, detail=terminal(entry.run_id), background=True)
    assert result["workspace_capture"]["status"] == "in_place", result
    assert current.run(tmp_path, entry.run_id).capture_job == {} and not requests


@pytest.mark.parametrize("decision", ["apply", "reject"])
def test_integrate_refuses_an_in_place_run_and_changes_nothing(tmp_path, monkeypatch, decision):
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, entry = _finished_in_place(tmp_path, monkeypatch)
    out = _integrate_delegated_patch(ctx, entry.run_id, decision, "")
    assert out.startswith("⚠️ INTEGRATE_DELEGATED_NOT_ISOLATED") and "workspace='in_place'" in out, out
    assert "undo through git" in out and "nothing to apply or reject" in out
    assert custody.replay(tmp_path)[entry.run_id].patch_disposed == ""
    assert (_target(ctx) / "README.md").read_text(encoding="utf-8") == "the child's edit, made in place\n"


# --------------------------------------------------------------------------- retry

def test_an_unknown_in_place_start_replays_on_retry(tmp_path, monkeypatch):
    lost = ClaudexorUnavailable("daemon_unreachable", "connection reset", status_code=503)
    original, unknown, ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-place",
                                            start_error=lost, start_kwargs={"workspace": "in_place"})
    token = unknown["pending_invocation_id"]
    pending, = custody.pending_invocations(tmp_path)
    assert pending["invocation_id"] == token and is_in_place_run(pending)
    assert pathlib.Path(pending["execution_root"]).resolve() == _target(ctx) and not pending.get("snapshot_id")
    request, payload, _ctx, calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-place",
                                           start_kwargs={"retry_of": token})
    assert payload["status"] == "started" and "retry_binding_absent" not in json.dumps(payload), payload
    assert request == original and len(calls) == 1
    assert custody.replay(tmp_path)["run-place"].resource_ref == IN_PLACE


def test_a_retry_cannot_choose_a_place(tmp_path, monkeypatch):
    from ouroboros import subagent_runtime

    lost = ClaudexorUnavailable("daemon_unreachable", "connection reset", status_code=503)
    _original, unknown, ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-place",
                                             start_error=lost, start_kwargs={"workspace": "in_place"})
    refused = json.loads(subagent_runtime.exact_start(
        ctx, "finish the remaining work", {"retry_of": unknown["pending_invocation_id"], "workspace": "copy"}).text)
    assert refused["reason"] == "retry_selector_conflict", refused


# --------------------------------------------------------------------------- schedule_subagent

@pytest.mark.parametrize("surface", ["self_worktree", "external_workspace", "genesis"])
def test_schedule_accepts_in_place_only_for_an_acting_git_surface(surface):
    from ouroboros.tools.control_subagent_spec import _validated_schedule_fields, schedule_subagent_properties

    assert schedule_subagent_properties()["workspace"]["enum"] == ["copy", "in_place"]
    base = {"objective": "Edit", "expected_output": "Files"}
    fields, error = _validated_schedule_fields({**base, "write_surface": surface, "workspace": "in_place"})
    assert not error and fields["workspace"] == "in_place"
    fields, error = _validated_schedule_fields({**base, "write_surface": surface, "workspace": "in_place",
                                                "directory_strategy": "direct", "scope_paths": []})
    assert not error and fields["workspace"] == "in_place"  # naming the default geometry is no folder request
    fields, error = _validated_schedule_fields({**base, "write_surface": surface, "workspace": "copy"})
    assert not error and "workspace" not in fields  # the default is not a recorded choice
    for refused in ({"write_surface": "read_only", "workspace": "in_place"}, {"workspace": "in_place"},
                    {"write_surface": surface, "workspace": "in_place", "directory_strategy": "copy",
                     "scope_paths": ["."]},
                    {"write_surface": surface, "workspace": "elsewhere"}):
        fields, error = _validated_schedule_fields({**base, **refused})
        assert not fields and "TOOL_ARG_ERROR (schedule_subagent)" in error and "workspace" in error, refused


def test_a_native_child_cannot_be_put_in_place_and_nothing_is_created(tmp_path, monkeypatch):
    from tests.test_directory_execution_options import _schedule

    ctx, events, response = _schedule(tmp_path, monkeypatch, kind="api_model", options={"workspace": "in_place"})
    assert "TOOL_ARG_ERROR" in response and "workspace=in_place" in response
    assert events.empty() and not list((ctx.drive_root / "task_results").glob("*.json"))


def test_the_choice_survives_schedule_dispatch_and_reaches_the_first_start(tmp_path, monkeypatch):
    from ouroboros import subagent_runtime
    from ouroboros.delegate_shared import delegate_result
    from ouroboros.subagent_bootstrap import bootstrap_before_context
    from supervisor import events
    from tests.test_directory_execution_options import _schedule
    from tests.test_nested_rights_depth import _fake_ctx

    parent, event_queue, response = _schedule(tmp_path, monkeypatch, kind="agent_session",
                                              options={"workspace": "in_place"})
    event = event_queue.get_nowait()
    assert event["workspace"] == "in_place", response
    enqueued = []
    supervisor = _fake_ctx(parent.drive_root, enqueued)
    supervisor.REPO_DIR = parent.repo_dir
    events._handle_schedule_task(event, supervisor)
    task, = enqueued
    assert task.get("workspace") == "in_place", sorted(task)
    starts = []

    def exact_start(ctx, prompt, spec):
        starts.append(spec)
        return delegate_result({"status": "started", "run_id": "in-place-run"})

    monkeypatch.setattr(subagent_runtime, "exact_start", exact_start)
    child = SimpleNamespace(
        task_id=event["task_id"], drive_root=pathlib.Path(task["drive_root"]), budget_drive_root=str(parent.drive_root),
        task_metadata=task["metadata"], workspace_root=pathlib.Path(task["workspace_root"]),
        workspace_mode=task["workspace_mode"])
    wake = bootstrap_before_context(child, task, SimpleNamespace(blocked=False))
    assert json.loads(wake)["status"] == "configured_session_started"
    assert len(starts) == 1 and starts[0]["workspace"] == "in_place"
    assert child._configured_actor_bootstrap["workspace"] == "in_place"
