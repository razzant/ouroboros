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
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import test_environment
from ouroboros.delegate_directory import in_place_refusal, is_in_place_run
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from tests._delegated_transport_shared import (  # noqa: F401 -- autouse transport fixture
    _owned_gateway_uses_each_test_transport,
)
from tests.test_delegate_continuation import _settle, _start
from tests._private_scratch import _synthetic_temp_parent, qualifying_home  # noqa: F401 -- autouse fixture

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


SOURCED = {"observation_seq": 1, "source_ref": {"sha256": "a" * 64}}


@pytest.mark.parametrize("ending, live", [
    ((), True),
    ((custody.TERMINAL_OBSERVED, {**SOURCED, "state": "stopping"}), True),  # no terminal state: the reducer drops it
    ((custody.TERMINAL_OBSERVED, {"observation_seq": 1, "source_ref": {}, "state": "cancelled"}), True),  # unsourced
    ((custody.TERMINAL_OBSERVED, {**SOURCED, "state": "cancelled"}), False),
    ((custody.SETTLED, {"state": "succeeded"}), False),  # written only after the engine reported a terminal state
    ((custody.CLOSED_ABSENT, {"reason": "absent from its owning store"}), False),
], ids=["started", "unknown_state", "unsourced", "terminal", "settled", "closed_absent"])
def test_a_direct_writer_stays_live_until_custody_proves_its_end(tmp_path, ending, live):
    """The real custody rows a cancelled child's shared folder waits on: an unproven end keeps the writer live, a proven
    absence ends it. A pending start, a copy run and a writer of another folder or task answer as such."""
    from ouroboros.delegate_directory import live_direct_writers

    folder, elsewhere = tmp_path / "shared", tmp_path / "elsewhere"

    def started(run_id, where, ref, task_id="child"):
        assert custody.record_started(tmp_path, custody.RunCustody(
            run_id=run_id, task_id=task_id, route_id="fake", invocation_id=f"{run_id}-start", execution_root=str(where),
            target_root=str(where), resource_ref=ref))

    started("leaf", folder, dict(IN_PLACE))
    if ending:
        assert custody.emit(tmp_path, ending[0], {"run_id": "leaf", "task_id": "child", **ending[1]})
    started("copy", folder, {"workspace_kind": "git", "strategy": "copy"})
    started("other-folder", elsewhere, {"workspace_kind": "directory", "strategy": "direct"})
    started("other-task", folder, dict(IN_PLACE), task_id="sibling")
    assert live_direct_writers(tmp_path, "child", folder) == (["run leaf"] if live else [])
    assert custody.emit(tmp_path, custody.START_REQUESTED, {
        "task_id": "child", "invocation_id": "folder-start", "request": {"prompt": "work"}, "target_root": str(folder),
        "execution_root": str(folder), "resource_ref": {"workspace_kind": "directory", "strategy": "direct"}})
    assert live_direct_writers(tmp_path, "child", folder) == (["run leaf"] if live else []) + ["start folder-start"]


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
    from ouroboros.delegate_start_instructions import DISPOSABLE_SCRATCH, SHARED_SCRATCH_ANCESTRY, scratch_fact
    from tests.test_continuation_fresh_snapshot_binding import _GRANT_ROW, _ctx, _leg

    ctx = _ctx(tmp_path, "git")
    request, payload = _leg(tmp_path, monkeypatch, ctx, "run-1", prompt="edit here", access="full", row=_GRANT_ROW,
                            workspace="in_place")
    assert payload["workspace"] == "in_place" and request["access"] == "full"
    assert "auxiliaryWritableRoots" not in request["execution"]  # ownership, never a new grant
    held = custody.replay(tmp_path)["run-1"]
    [record] = _in_place_owners(ctx).glob("*/scratch.json")
    assert record.parent.name == f"in-place-{held.invocation_id}"
    scratch = pathlib.Path(json.loads(record.read_text(encoding="utf-8"))["path"]) / "w"
    assert scratch.is_dir() and not any(scratch.iterdir()) and _target(ctx) not in scratch.parents
    ancestry = SHARED_SCRATCH_ANCESTRY if test_environment._short_temp_platform() else ""
    binding = request["instructions"].split("\n\nDELEGATED IN-PLACE BINDING", 1)[1]
    assert (f"outside every Git working tree, in {scratch}, this run's disposable scratch{ancestry}, "
            f"{DISPOSABLE_SCRATCH}.") in binding  # POSIX fixture uses shared parent; non-POSIX hosts emit no POSIX ancestry notice
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


# #4093, the helper side: a separate benign process (stdlib only, ``-I -S``) reads the exact instructions the engine
# received, extracts the ONE granted scratch and requires it to EQUAL the newly allocated empty directory the test passes.
# Before it looks at or binds in that directory it refuses any path outside the test-owned root (a lexical check). It then
# runs its ordinary permission check (each component from that root down through the scratch is a real directory of its
# user or root that group and others cannot write, and the scratch is writable) and ONLY THEN opens a single ordinary
# AF_UNIX socket there, absolute where the real bound allows, else relative after chdir, and closes it.
_CHILD = r'''
import json, os, re, socket, stat, sys
text, expected, root = sys.argv[1], sys.argv[2], os.path.realpath(sys.argv[3])
binding = open(text, encoding="utf-8").read().split("DELEGATED IN-PLACE BINDING", 1)[1].split("\n\n", 1)[0]
found = re.findall(r"outside every Git working tree, in (.+?), this run's disposable scratch", binding)
assert found == [expected], f"the instructions name {found!r}, not exactly the one issued scratch {expected!r}"
assert os.path.isabs(expected) and os.path.commonpath([root, expected]) == root and expected != root, f"outside the test-owned root: {expected}"
assert os.path.realpath(expected) == expected and os.listdir(expected) == [], "not a new, empty, physical directory"
parts = os.path.relpath(expected, root).split(os.sep)
for count in range(len(parts) + 1):
    walk = os.path.join(root, *parts[:count])
    seen = os.lstat(walk)
    assert stat.S_ISDIR(seen.st_mode) and seen.st_uid in (0, os.geteuid()) and not seen.st_mode & 0o022, f"refused: {walk}"
assert os.access(expected, os.W_OK | os.X_OK), "the scratch is not writable"
sun = 104 if sys.platform == "darwin" else 108  # the real AF_UNIX bound, the NUL included
address = os.path.join(expected, "ctl")
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    if len(os.fsencode(address)) < sun:
        sock.bind(address)
        placed = "absolute"
    else:
        os.chdir(expected)
        sock.bind("ctl")
        placed = "relative"
finally:
    sock.close()
print(json.dumps({"placed": placed, "socket": stat.S_ISSOCK(os.lstat(address).st_mode), "entries": sorted(os.listdir(expected))}))
'''
POSIX = pytest.mark.skipif(os.name != "posix", reason="the private scratch parent and AF_UNIX are macOS and Linux paths")


def _granted_scratch(ctx) -> pathlib.Path:
    [record] = _in_place_owners(ctx).glob("*/scratch.json")
    return pathlib.Path(json.loads(record.read_text(encoding="utf-8"))["path"]) / "w"


@POSIX
def test_the_child_opens_one_ordinary_socket_in_its_granted_private_scratch_and_the_host_releases_exactly_that_scratch(
        tmp_path, monkeypatch):
    """The #4093 normal-use regression: the real decision, allocator, binding text, child process, socket and release.
    DOUBLED, each on purpose (``tests/_private_scratch.py``): the engine transport (``_leg``); the OS-managed family and
    the chain above this test's own HOME (the run's root sits under the real ``/tmp``), while the child judges every
    component from that HOME down for real; and length, ``SUN_PATH`` widened for the allocation decision only (the
    synthetic HOME is longer than the real 21/25-byte budget), the child's own bind obeying the real bound. The native length budget stays in tests/test_test_environment.py and the
    native lifecycle in tests/test_session_scratch_release.py."""
    from ouroboros import task_custody
    from ouroboros.workspace_preflight import release_finished_scratch
    from tests._store_shared import seed_result
    from tests.test_continuation_fresh_snapshot_binding import _GRANT_ROW, _ctx, _leg

    home = qualifying_home(tmp_path, monkeypatch)
    ctx, data = _ctx(tmp_path, "git"), tmp_path.resolve()
    request, _payload = _leg(tmp_path, monkeypatch, ctx, "run-1", prompt="edit here", access="full", row=_GRANT_ROW,
                             workspace="in_place")
    scratch = _granted_scratch(ctx)
    assert scratch.parent.parent == home and "auxiliaryWritableRoots" not in request["execution"]  # no new right
    binding = request["instructions"].split("DELEGATED IN-PLACE BINDING", 1)[1]
    assert f"in {scratch}, this run's disposable scratch (POSIX owner/mode check: " in binding
    sibling = home / ".obs-sibling"  # the test's own, never recorded and not the record's name: nothing the host owns
    sibling.mkdir(mode=0o700)
    (sibling / "mine.txt").write_text("kept\n", encoding="utf-8")
    text, script = tmp_path / "instructions.txt", tmp_path / "child.py"
    text.write_text(request["instructions"], encoding="utf-8")
    script.write_text(_CHILD, encoding="utf-8")
    done = subprocess.run([sys.executable, "-I", "-S", str(script), str(text), str(scratch), str(home)],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    facts = json.loads(done.stdout.splitlines()[-1])
    assert facts["socket"] is True and facts["entries"] == ["ctl"] and facts["placed"] in ("absolute", "relative"), facts
    _settle(tmp_path, "run-1", ctx.task_id, state="succeeded", reason="")
    seed_result(data, ctx.task_id, {"status": "completed"})
    assert release_finished_scratch(data, lambda _task: False)["released"]  # the host's ordinary release
    assert task_custody.remove_custody_trash(data)["removed"] == 1
    assert not scratch.parent.exists() and home.is_dir()  # exactly that container, socket and all
    assert sorted(entry.name for entry in home.iterdir()) == [sibling.name]
    assert (sibling / "mine.txt").read_text(encoding="utf-8") == "kept\n"


@POSIX
def test_the_ancestry_an_in_place_binding_states_follows_what_was_read_and_an_unknown_read_claims_nothing(
        tmp_path, monkeypatch):
    from ouroboros.delegate_start_instructions import (
        PRIVATE_SCRATCH_ANCESTRY,
        SHARED_SCRATCH_ANCESTRY,
        scratch_ancestry,
    )

    root = tmp_path.resolve()
    home, shared = root / "home", root / "shared"
    for scratch in (home / ".obs-abcd1234" / "w", shared / "obs-abcd1234" / "w"):
        scratch.mkdir(parents=True)
    monkeypatch.setattr(test_environment, "private_scratch_home", lambda: home)
    monkeypatch.setattr(test_environment, "session_scratch_parent", lambda: shared)
    monkeypatch.setattr(test_environment, "issued_scratch_problem", lambda _scratch: "")
    private, plain = str(home / ".obs-abcd1234" / "w"), str(shared / "obs-abcd1234" / "w")
    assert scratch_ancestry(private) == PRIVATE_SCRATCH_ANCESTRY  # words say a POSIX owner/mode check, not access
    assert "ACLs and mounts are not examined" in PRIVATE_SCRATCH_ANCESTRY and "effective" not in PRIVATE_SCRATCH_ANCESTRY
    assert scratch_ancestry(plain) == SHARED_SCRATCH_ANCESTRY  # named shared, ancestry not established
    assert scratch_ancestry(str(home / "obs-abcd1234" / "w")) == ""  # the home names only its exact private children
    assert scratch_ancestry(str(root / "elsewhere" / "obs-1" / "w")) == ""  # a parent the host cannot name
    alias = root / "alias"
    alias.symlink_to(shared, target_is_directory=True)
    assert scratch_ancestry(str(alias / "obs-abcd1234" / "w")) == ""  # a name that is a link now is not what was issued
    real_platform = test_environment.sys.platform
    monkeypatch.setattr(test_environment.sys, "platform", "win32")  # a nonlocal geometry states no POSIX fact at all
    assert scratch_ancestry(private) == "" and scratch_ancestry(plain) == ""
    monkeypatch.setattr(test_environment.sys, "platform", real_platform)
    monkeypatch.setattr(test_environment, "issued_scratch_problem", lambda _scratch: "its w is writable by others")
    assert scratch_ancestry(private) == ""  # a failed check claims nothing

    def unreadable(_scratch):
        raise PermissionError("unreadable")

    monkeypatch.setattr(test_environment, "issued_scratch_problem", unreadable)
    assert scratch_ancestry(private) == ""  # an unknown read claims nothing


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
