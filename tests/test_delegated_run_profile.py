"""The access profile a delegated run may hold, and the guards that keep it there.

Split verbatim out of ``tests/test_delegated_subagent_transport.py`` by theme. This
module owns what a mutating and a read-only child may ask the harness for, why an
effective profile is verified rather than assumed, the task that alone may touch a
live run, and the write-root and workspace refusals that precede any daemon call.
"""

from __future__ import annotations

import datetime
import json
import pytest
from ouroboros.config import CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION

from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _delegating_ctx,
    _owned_gateway_uses_each_test_transport,
    _started_request,
)


def test_a_mutating_child_runs_live_in_a_private_snapshot_not_the_shared_tree(tmp_path, monkeypatch):
    # C1: `live` still means the harness edits its scope root in place — but that root
    # is a PRIVATE execution snapshot of the nanny's write root. The shared tree gets
    # nothing until the nanny explicitly integrates the captured diff.
    import pathlib as _pl

    request, payload = _started_request(tmp_path, acting=True, monkeypatch=monkeypatch)
    assert request["access"] == "workspace_write"
    assert request["mode"] == "agent"
    assert request["execution"] == {"isolation": "live", "delegated": True}
    worktree = tmp_path.parent / f"wt-{tmp_path.name}"
    assert request["scope"]["kind"] == "project"
    scope_root = _pl.Path(str(request["scope"]["root"]))
    assert scope_root.resolve() != worktree.resolve(), "the run must NEVER scope the shared tree"
    assert scope_root.resolve().is_relative_to((tmp_path / "snap_root").resolve())
    assert payload["execution_root"] == str(request["scope"]["root"])
    assert _pl.Path(payload["authority_target_root"]).resolve() == worktree.resolve()
    assert payload["baseline_id"], "the baseline commit is the binding's third leg"
    # The snapshot genuinely carries the target's current state.
    assert (scope_root / "README.md").read_text(encoding="utf-8") == "seed\n"
    assert payload["access"] == "workspace_write" and payload["isolation"] == "live"


def test_a_read_only_child_uses_the_same_transport_with_a_narrower_profile(tmp_path, monkeypatch):
    # One nanny, one transport: the ONLY difference is the derived profile and the run
    # shape it implies. `execution.isolation='live'` is agent-only in Claudexor — a
    # non-agent run carrying it is refused at the boundary — and a read-only child has
    # nothing to write back anyway.
    request, payload = _started_request(tmp_path, acting=False, monkeypatch=monkeypatch)
    assert request["access"] == "readonly"
    assert request["mode"] == "ask"
    assert "execution" not in request
    assert payload["access"] == "readonly"


def test_the_host_states_its_prohibitions_on_every_delegated_run(tmp_path, monkeypatch):
    request, _ = _started_request(tmp_path, acting=True, monkeypatch=monkeypatch)
    instructions = request["instructions"].lower()
    assert "git commit" in instructions and "outside this root" in instructions


def test_the_model_has_no_argument_that_could_widen_the_profile():
    from ouroboros.tools import delegate

    entry = next(e for e in delegate.get_tools() if e.name == "delegate_start")
    properties = set(entry.schema["parameters"]["properties"])
    # `retry_of` names an INVOCATION, not authority (ownership-checked replay);
    # root/bucket/skill_name are a SELECTOR resolved through the same
    # ResolvedResourceBinding authorizer as ordinary writes (R1 item 9).
    assert properties == {
        "prompt", "subagent_id", "max_seconds", "retry_of", "root", "bucket", "skill_name",
    }
    assert entry.schema["parameters"]["properties"]["root"]["enum"] == ["skill_payload"]
    assert not properties & {"access", "mode", "isolation", "scope", "write_surface", "cwd"}


def test_a_read_only_task_cannot_obtain_workspace_write(tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.delegate import _derive_authority
    from ouroboros.tools.registry import ToolContext

    for constraint in (
        None,                                                       # no constraint at all
        TaskConstraint(mode="local_readonly_subagent"),             # explicitly read-only
        TaskConstraint(mode="acting_subagent", surface=""),         # acting but unresolved surface
        TaskConstraint(mode="acting_subagent", surface="bogus"),    # acting with an invalid surface
    ):
        ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_constraint=constraint)
        ctx.task_metadata = {"parent_task_id": "p"}
        authority = _derive_authority(ctx)
        assert authority.access == "readonly", constraint
        assert authority.mode == "ask" and authority.isolation == ""


@pytest.mark.parametrize("effective,entitled,widened", [
    ("readonly", "readonly", ""),
    ("readonly", "workspace_write", ""),          # narrower than asked is fine
    ("workspace_write", "workspace_write", ""),
    ("workspace_write", "readonly", "workspace_write"),
    ("full", "workspace_write", "full"),
    ("inherit_native", "workspace_write", "inherit_native"),
    ("a-profile-from-a-future-engine", "workspace_write", "a-profile-from-a-future-engine"),
])
def test_effective_access_is_verified_not_assumed(effective, entitled, widened):
    from ouroboros.tools.delegate import _widened_access

    detail = {"lastSeq": 12, "summary": {"effectiveAccess": effective, "state": "running"}}
    assert _widened_access(detail, entitled) == widened


def test_an_undisclosed_effective_profile_is_unverified_not_compliant():
    """Absence of evidence is not evidence of narrowness.

    An earlier version returned "" (compliant) whenever the field was missing, and a test
    codified that as `# not disclosed yet: nothing to judge` — so any daemon build, harness
    or malformed response that omitted the field turned the only containment gate into a
    silent no-op while the run kept writing. It also fell back to `summary["access"]`,
    which the daemon computes as `effectiveAccess ?? the client's own request`: that
    compares our request against itself and can only ever pass.
    """
    from ouroboros.tools.delegate import _ACCESS_UNVERIFIED, _widened_access

    # Before admission there really is nothing to judge.
    assert _widened_access({"summary": {"state": "queued"}}, "readonly") == ""
    assert _widened_access({"summary": {}}, "readonly") == ""

    # Absence only means "no evidence" while the run can still ACT, and only after it
    # has produced anything. The daemon marks a run `running` at DEQUEUE — before the
    # orchestrator writes the contract the profile is derived from — so judging that
    # moment cancelled healthy runs, and judging a terminal state reported a run that
    # merely failed to start as a containment breach.
    assert _widened_access({"lastSeq": 0, "summary": {"state": "running"}}, "readonly") == ""
    for state in ("succeeded", "failed", "cancelled", "interrupted"):
        detail = {"lastSeq": 40, "summary": {"state": state}}
        assert _widened_access(detail, "readonly") == "", state

    # A live run that HAS produced events and still discloses nothing has no evidence.
    live = {"lastSeq": 12, "summary": {"state": "running"}}
    assert _widened_access(live, "readonly") == _ACCESS_UNVERIFIED

    # The echo must not be accepted as an independent witness.
    detail = {"lastSeq": 12, "summary": {"state": "running", "access": "workspace_write"}}
    assert _widened_access(detail, "workspace_write") == _ACCESS_UNVERIFIED

    # A really widened profile is still caught in every state.
    for state in ("running", "succeeded"):
        detail = {"lastSeq": 12, "summary": {"state": state, "effectiveAccess": "full"}}
        assert _widened_access(detail, "readonly") == "full", state


def test_a_succeeded_run_that_never_proved_its_profile_says_so_in_its_result():
    """P34P1.4: a SUCCEEDED run whose summary carries no `effectiveAccess` was accepted
    as compliant — a result with no evidence that the profile the host asked for is the
    profile the engine enforced, which is the name-without-proof class this module
    exists to refuse.

    Enforcement is NOT the answer for a finished run: it is over, there is nothing left
    to contain, and routing absence through the breach path would CANCEL a succeeded run
    and destroy the very result the lane exists to fetch (the v6.87.37 lesson — the
    containment gate stopped cancelling healthy runs for exactly this reason). So it is
    DISCLOSED, on the same terminal payload the parent reads, like the HOME half's
    missing fact. Both lanes get it: `readonly` staying `readonly` is the profile that
    matters most, and the `containment` block is asked only of marker-carrying runs."""
    from ouroboros.subagents import delegated_run_shape
    from ouroboros.tools.delegate import _terminal_payload

    # A succeeded run with NO disclosed profile: unverified, and it says why.
    silent = {"lastSeq": 40, "summary": {"state": "succeeded"}}
    evidence = _terminal_payload("run-1", silent, delegated_run_shape(False))["access_evidence"]
    assert evidence["verified"] is False and evidence["effective"] == ""
    assert evidence["requested"] == "readonly" and evidence["state"] == "succeeded"
    assert "SUCCEEDED without ever disclosing" in evidence["note"]

    # A succeeded run that DID disclose one is verified, with no note.
    proven = {"lastSeq": 40, "summary": {"state": "succeeded", "effectiveAccess": "readonly"}}
    evidence = _terminal_payload("run-1", proven, delegated_run_shape(False))["access_evidence"]
    assert evidence == {"requested": "readonly", "effective": "readonly",
                        "verified": True, "state": "succeeded"}

    # A run that did NOT succeed keeps the softer wording: it may never have had a
    # profile at all, so this is absence of evidence rather than a missing proof.
    for state in ("failed", "cancelled", "interrupted"):
        detail = {"lastSeq": 40, "summary": {"state": state}}
        evidence = _terminal_payload("run-1", detail, delegated_run_shape(False))["access_evidence"]
        assert evidence["verified"] is False, state
        assert "absence of evidence, not a breach" in evidence["note"], state

    # The ECHO is never a witness: the daemon computes `access` as
    # `effectiveAccess ?? our own request`, so a payload carrying only the echo must
    # still read unverified.
    echo = {"lastSeq": 40, "summary": {"state": "succeeded", "access": "readonly"}}
    assert _terminal_payload("run-1", echo, delegated_run_shape(False))[
        "access_evidence"]["verified"] is False

    # The mutating lane carries BOTH halves, and neither displaces the other.
    mutating = _terminal_payload("run-1", silent, delegated_run_shape(True))
    assert mutating["access_evidence"]["verified"] is False
    assert mutating["containment"]["verified"] is False


def test_a_mutating_run_requires_an_ACTIVE_workspace_not_merely_agreement(tmp_path):
    """Agreement alone reopened the critical it was written to close.

    `active_repo_dir_for` falls back to `repo_dir` when workspace mode is off, so a
    constraint whose `write_root` happens to name that same directory made the equality
    check pass — and handed an external shell the live repository, which is exactly the
    original defect.
    """
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.subagents import delegated_run_shape
    from ouroboros.tools.delegate import _mutation_authority
    from ouroboros.tools.registry import ToolContext

    repo = tmp_path / "repo"
    repo.mkdir()
    ctx = ToolContext(
        repo_dir=repo, drive_root=tmp_path,
        task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree",
                                       write_root=str(repo)),
    )
    ctx.workspace_root = None
    ctx.workspace_mode = ""
    record, refusal = _mutation_authority(
        ctx, delegated_run_shape(True))
    assert refusal and "workspace_not_active" in refusal, refusal
    assert record == {}


def test_a_widened_run_is_cancelled_and_typed_not_reported_as_progress(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    cancelled = {}

    class _Stub:
        def handshake(self, **_kw): return {}
        def get_run(self, rid, **_kw):
            return {"lastSeq": 7, "summary": {
                "state": "cancelled" if cancelled else "running",
                "effectiveAccess": "full",
            }}
        def cancel_run(self, rid, reason=""):
            cancelled["reason"] = reason
            return {"accepted": True}
        def remove_project(self, pid): pass
        def close(self): pass

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    ctx = _delegating_ctx(tmp_path, acting=True)
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(
        task_id="t-nanny", route_id="some-route", model="m",
        project_id="prj", project_owned=False,
    )
    out = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
    delegate._CUSTODY.clear()
    assert out["status"] == "refused"
    assert out["reason"] == "access_profile_widened"
    assert out["effective_access"] == "full" and out["entitled_access"] == "workspace_write"
    assert cancelled["reason"] == "access_profile_widened"


def test_a_delegated_run_can_only_be_touched_by_the_task_that_started_it(tmp_path):
    """The daemon bearer token grants the ENTIRE Claudexor API, so naming a run is
    reaching it. Without custody binding, a child could pass any run id it observed and
    read — or CANCEL — the owner's own unrelated work, or a sibling reviewer's run, and
    cancelling a reviewer destroys the verdict that was the whole point of running it."""
    import json

    import ouroboros.tools.delegate as delegate
    from ouroboros.tools.registry import ToolContext

    def _ctx(task_id):
        ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
        ctx.task_id = task_id
        ctx.task_metadata = {"root_task_id": task_id}
        return ctx

    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-mine"] = delegate._RunCustody(
        task_id="task-a", route_id="codex", model="m", project_id="prj", project_owned=False,
    )

    for tool, call in (
        ("delegate_wait", lambda ctx, rid: delegate._delegate_wait(ctx, rid, wait_sec=1)),
        ("delegate_cancel", lambda ctx, rid: delegate._delegate_cancel(ctx, rid, reason="x")),
    ):
        # A run with NO durable start record anywhere: ownership is UNKNOWN, which is a
        # different fact from "demonstrably someone else's" and is refused on its own name.
        out = json.loads(call(_ctx("task-a"), "run-someone-elses"))
        assert out["status"] == "refused", (tool, out)
        assert out["reason"] == "run_ownership_unknown", (tool, out)

        # A run a SIBLING task started in the same worker process.
        out = json.loads(call(_ctx("task-b"), "run-mine"))
        assert out["status"] == "refused", (tool, out)
        assert out["reason"] == "run_not_owned", (tool, out)

    delegate._CUSTODY.clear()


def test_a_mutating_run_is_refused_when_the_root_and_the_granted_write_root_disagree(tmp_path, monkeypatch):
    """AUTHORITY and ROOT came from two different predicates and were never compared.

    Authority comes from `task_constraint` via `active_tool_profile`. The root came from
    `active_repo_dir_for`, and `ToolContext.active_repo_dir()` falls back to `repo_dir` —
    the LIVE Ouroboros source tree — whenever `is_workspace_mode()` is false, which
    `workspace_mode_block_reason` makes happen for a worktree overlapping the repo or the
    data drive, or for a task record missing its workspace fields. In that state the host
    would have handed an external SHELL `workspace_write` on its own repository, and no
    per-tool guard applies because a shell is not a tool. Two independent reviewers found
    this on the same branch.
    """
    import json

    import ouroboros.tools.delegate as delegate
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools.registry import ToolContext

    class _Stub:
        engine_version = CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION

        def handshake(self, **_kw): return {}
        def agent_capabilities(self):
            return {"harnesses": [{"id": "some-route", "enabled": True, "status": "ok",
                                   "accessProfilesSupported": ["readonly", "workspace_write"]}]}
        def quota_snapshots(self): return []
        def start_run(self, request, *, idempotency_key=""):
            raise AssertionError("must refuse before starting")
        def close(self): pass

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())

    repo = tmp_path / "repo"
    repo.mkdir()
    inside_the_drive = tmp_path / "wt"
    inside_the_drive.mkdir()

    ctx = ToolContext(
        repo_dir=repo, drive_root=tmp_path,
        task_constraint=TaskConstraint(
            mode="acting_subagent", surface="self_worktree",
            write_root=str(inside_the_drive),
        ),
    )
    ctx.task_id = "t-nanny"
    ctx.task_metadata = {"root_task_id": "t-root"}
    ctx.workspace_root = str(inside_the_drive)
    ctx.workspace_mode = "self_worktree"

    out = json.loads(delegate._delegate_start(ctx, "edit the README"))
    assert out["status"] == "refused", out
    # A worktree overlapping the data drive is refused as "not an active workspace" —
    # `workspace_mode_block_reason` fires first and is the stronger statement.
    assert out["reason"] in ("write_root_mismatch", "workspace_not_active"), out

    # And a mutating child whose constraint granted no write_root at all is refused too,
    # rather than the host picking a directory on its behalf.
    ctx.task_constraint = TaskConstraint(mode="acting_subagent", surface="self_worktree")
    out = json.loads(delegate._delegate_start(ctx, "edit the README"))
    assert out["status"] == "refused", out
    assert out["reason"] in ("write_root_missing", "workspace_not_active"), out


def test_the_guards_that_protect_a_delegated_run_fail_closed(tmp_path, monkeypatch):
    """Three guards that each failed OPEN in exactly the case they existed for."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.tools.registry import ToolContext

    # 1. Custody with an unknown identity on either side is refused, not waved through.
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-x"] = delegate._RunCustody(
        task_id="", route_id="r", model="m", project_id="p", project_owned=False)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "t-a"
    ctx.task_metadata = {"root_task_id": "t-a"}
    assert json.loads(delegate._delegate_cancel(ctx, "run-x"))["reason"] == "run_not_owned"
    ctx.task_id = ""
    delegate._CUSTODY["run-x"] = delegate._RunCustody(
        task_id="t-a", route_id="r", model="m", project_id="p", project_owned=False)
    assert json.loads(delegate._delegate_cancel(ctx, "run-x"))["reason"] == "run_not_owned"
    delegate._CUSTODY.clear()

    # 2. A run with no knowable deadline gets a conservative cap, never an omitted one:
    #    an omitted cap is Claudexor's 7-day schema bound on a run nobody can cancel.
    bare = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    bare.task_id = "t-a"
    bare.task_metadata = {"root_task_id": "t-a"}          # no deadline_at at all
    # The cap is the EXISTING task ceiling SSOT, not a second hardcoded one: a 1h guess
    # would have truncated a headless/benchmark run that legitimately has no deadline.
    from ouroboros.config import get_task_abs_ceiling_sec

    assert delegate._bounded_max_seconds(bare, None) == int(get_task_abs_ceiling_sec())

    # ...but never past Claudexor's own schema bound. The task ceiling clamps only from
    # BELOW, so an owner who raises it past a week would make every deadline-less start
    # send an out-of-schema value and get a 400 instead of a run.
    monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", "1000000")
    assert delegate._bounded_max_seconds(bare, None) == delegate._CLAUDEXOR_MAX_SECONDS

    # ...and an EXPLICIT ask is clamped by the same bound. `max_seconds` is a
    # model-supplied tool argument with no maximum in its schema, so clamping only the
    # fallback branch left the ask itself able to sail past it — the same defect, one
    # branch over from the one that was fixed.
    assert delegate._bounded_max_seconds(bare, 1_000_000) == delegate._CLAUDEXOR_MAX_SECONDS
    assert delegate._bounded_max_seconds(bare, 120) == 120
    # An explicit narrower ask still wins — the cap is a floor for the unknown case only.
    assert delegate._bounded_max_seconds(bare, 120) == 120

    # 3. P34P1.8: an EXPIRED deadline is NOT the same fact as having none.
    #    `deadline_remaining_sec` answers 0.0 for both, so the fallback above handed an
    #    already-expired nanny the absolute task ceiling — hours of delegated work, and
    #    real quota, beginning after the instant its own deadline demanded it stop.
    monkeypatch.delenv("OUROBOROS_TASK_ABS_CEILING_SEC", raising=False)
    expired = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    expired.task_id = "t-a"
    expired.task_metadata = {"root_task_id": "t-a", "deadline_at": "2020-01-01T00:00:00Z"}
    assert delegate.deadline_expired(expired) is True
    assert delegate.deadline_expired(bare) is False, "no deadline is not an expired one"

    from ouroboros.deadline_utils import utc_now

    live = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    live.task_id = "t-a"
    live.task_metadata = {"root_task_id": "t-a",
                          "deadline_at": (utc_now() + datetime.timedelta(hours=1)).isoformat()}
    assert delegate.deadline_expired(live) is False
    # ...and the live deadline still NARROWS the bound, as it always did.
    assert 0 < delegate._bounded_max_seconds(live, None) <= 3600

    # The refusal is at the START, before the daemon is touched: nothing spent, nothing
    # registered, and the reason names the honest next move.
    reached = []
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")

    class _NeverReached:
        def handshake(self, **_kw): reached.append("handshake"); return {}
        def close(self): pass

    from ouroboros.gateways import claudexor as _gw

    monkeypatch.setattr(_gw, "ClaudexorGateway", lambda *a, **k: _NeverReached())
    refused = json.loads(delegate._delegate_start(expired, "start something new"))
    assert refused["status"] == "refused" and refused["reason"] == "task_deadline_expired"
    assert refused["definitely_unrun"] is True
    assert reached == [], "expired nanny never reaches daemon"


def test_an_unresolvable_write_root_is_a_typed_refusal_not_a_traceback(tmp_path):
    """"Can this path be resolved at all" is ONE question, not an exception set.

    Embedded nulls and symlink loops have changed their exact `Path.resolve()` failure
    behaviour across supported Python versions. Either escaping `_mutating_run_root`
    aborts `delegate_start` with a traceback instead of the typed refusal the function
    exists to produce — and a guard that raises delivers no decision at all.
    """
    import os

    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.delegate_containment import _resolved as containment_resolved
    from ouroboros.subagents import delegated_run_shape
    from ouroboros.tools.delegate import _mutation_authority, _resolved
    from ouroboros.tools.registry import ToolContext

    os.symlink(tmp_path / "b", tmp_path / "a")
    os.symlink(tmp_path / "a", tmp_path / "b")
    assert _resolved(tmp_path / "a" / "x") is None, "a symlink loop must resolve to None"
    assert containment_resolved(tmp_path / "a" / "x") is None
    assert _resolved("/etc/passwd\x00") is None, "an embedded null must resolve to None"
    assert _resolved(tmp_path) == tmp_path.resolve(), "an ordinary path still resolves"
    missing = tmp_path / "missing" / "leaf"
    assert _resolved(missing) == missing.resolve(strict=False)
    assert containment_resolved(missing) == missing.resolve(strict=False)

    workspace = tmp_path.parent / f"ws-{tmp_path.name}"
    workspace.mkdir()
    ctx = ToolContext(
        repo_dir=tmp_path / "repo", drive_root=tmp_path,
        task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree",
                                       write_root=str(tmp_path / "a" / "x")),
    )
    ctx.workspace_root = str(workspace)
    ctx.workspace_mode = "self_worktree"
    record, refusal = _mutation_authority(
        ctx, delegated_run_shape(True))
    assert refusal and "write_root_mismatch" in refusal, refusal
    assert record == {}


def test_an_inactive_workspace_is_refused_even_when_the_root_is_set(tmp_path):
    """The DISTINGUISHING case for the round-3 predicate fix, which had no test.

    The old check was `workspace_mode_block_reason(ctx) == "" and workspace_root set`,
    and `workspace_mode_block_reason` returns "" precisely WHEN `workspace_mode` is
    empty — so with a root set and the mode empty, the old condition passed and handed a
    shell the fallback root. Every existing test cleared BOTH fields, which the old
    predicate also refused via its `workspace_root` leg, so reverting the fix left the
    suite green. This is the one shape that tells the two predicates apart.
    """
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tool_access import workspace_mode_block_reason
    from ouroboros.subagents import delegated_run_shape
    from ouroboros.tools.delegate import _mutation_authority
    from ouroboros.tools.registry import ToolContext

    repo = tmp_path / "repo"
    repo.mkdir()
    ctx = ToolContext(
        repo_dir=repo, drive_root=tmp_path,
        task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree",
                                       write_root=str(repo)),
    )
    ctx.workspace_root = str(repo)   # SET...
    ctx.workspace_mode = ""          # ...but the mode is not, so the workspace is not active

    assert workspace_mode_block_reason(ctx) == "", "the old predicate's leg is satisfied here"
    assert ctx.is_workspace_mode() is False, "yet the workspace is genuinely inactive"

    record, refusal = _mutation_authority(
        ctx, delegated_run_shape(True))
    assert refusal, "an inactive workspace must be refused"
    assert "workspace_not_active" in refusal, refusal
    assert record == {}
