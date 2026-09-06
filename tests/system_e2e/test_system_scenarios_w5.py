"""S24-S25 — Ф4 wave 5 of the deep-integration suite: MUTATING delegated runs.

The waves before this one delegated only READ-ONLY runs, and every wave disclosed
the same gap (ADOPTION row DEFER-E2E-DELEG-MUT): the ONE delegation branch that
changes the owner's own tree on behalf of an external harness — snapshot
provisioning, the terminal capture, ``integrate_delegated_patch`` — had no
system-level cover. This module closes it against the same keyless stack the other
waves use, with the fake daemon now doing what a WRITING harness does: editing the
private execution snapshot its start body named, and recording the applied
containment facts of its attempt (``attempts/a01/attempt.yaml``).

* S24 — CLEAN PULL-IN. An EXTERNAL-WORKSPACE task delegates a mutating run: the
  host provisions a private Git snapshot of the workspace and sends its path as
  ``execution.workspaceRoot`` while ``scope.root`` stays the live tree; the fake
  harness edits the SNAPSHOT (one new untracked file, one tracked rewrite); the
  terminal wait captures the diff; and only the explicit
  ``integrate_delegated_patch(decision="apply")`` moves it into the live workspace,
  staged and never committed. Pinned: the durable custody chain
  (STARTED(access=workspace_write, snapshot_id, execution_root, baseline_sha) →
  PATCH_CAPTURED → APPLY_STARTED → DISPOSED(applied) → the ``delegate_run_patch_verdict``
  row and its artifact), the capture artifacts on disk (baseline manifest, patch,
  patch manifest, sha256 == the manifest's), the CONTAINMENT read from the attempt
  record (no ``delegate_run_unconfined`` row for this run), the ISOLATION proof —
  the live workspace still lacks the run's file at the moment the model is asked
  for the step AFTER the wait returned, and carries it (staged) afterwards — and
  the released snapshot (registry row gone, worktree gone).
* S25 — CONFLICTING PULL-IN. Same run, but the live tree drifts on a patched path
  between the capture and the decision (the scenario writes the drift from the
  script step itself, so the interleaving is causal, not timed). The apply is
  REFUSED typed (``INTEGRATE_CONFLICT`` … «YOU own this conflict»), nothing is
  disposed, the live file keeps the OWNER's content — not the run's — and the
  snapshot, its registry row and the captured patch all survive as the resolution
  material the nanny owns — and the TASK's own terminal is honest about it: a task
  that ends holding an undisposed captured patch reads ``failed`` with
  ``reason_code=delegated_custody_unreconciled``, not ``completed``.

The default-lane test pins the fake's new mutating half against the REAL readers:
the edits land in the workspace root the body named (never in ``scope.root``), and
``gateways/claudexor.py::attempt_containment`` reads the attempt record back.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

from tests.system_e2e.harness import (
    LANE_MOCK,
    ArtifactOracle,
    ScriptedStubModel,
    body_text,
    keyless_settings,
    require_lane,
    start_server,
    submit_running,
    wait_durable_result,
)
from tests.system_e2e.interfaces import (
    DEFAULT_WORKSPACE_EDITS,
    FAKE_MUTATE_MARKER,
    FakeClaudexorDaemon,
)
# The wave-3b delegated-transport glue, reused rather than re-derived: one author
# for the run-id regex the scripts read and for the custody-row reader, so a change
# to either cannot drift between the two delegation waves.
from tests.system_e2e.test_system_scenarios_w3b import (
    _RUN_ID_RE,
    _custody_rows,
    _roster,
    _wait_step,
)

# ===========================================================================
# Default lane: the fake daemon's MUTATING half, against the real readers.
# ===========================================================================


def test_fake_daemon_mutating_run_edits_only_the_execution_workspace(tmp_path):
    from ouroboros.gateways.claudexor import (
        ClaudexorGateway,
        attempt_containment,
        discover_daemon_at,
    )

    live = tmp_path / "live"
    snapshot = tmp_path / "snapshot"
    for path in (live, snapshot):
        path.mkdir()
        (path / "tracked.txt").write_text("one\n", encoding="utf-8")
    with FakeClaudexorDaemon(runs_dir=tmp_path / "runs") as daemon:
        daemon.install(tmp_path / "cx")
        with ClaudexorGateway(discover_daemon_at(tmp_path / "cx")) as gateway:
            gateway.handshake()
            handle = gateway.start_run({
                "prompt": FAKE_MUTATE_MARKER + " do the work",
                "instructions": "i", "authPreference": "subscription", "mode": "agent",
                "scope": {"kind": "project", "root": str(live)},
                "harnesses": [daemon.harness_id], "primaryHarness": daemon.harness_id,
                "access": "workspace_write", "maxSeconds": 60,
                "execution": {"isolation": "live", "delegated": True,
                              "workspaceRoot": str(snapshot)},
            }, idempotency_key="inv-w5-1")
            run_id = str(handle.get("runId") or "")
            assert run_id

    # The SNAPSHOT carries the harness's work; the live tree is byte-untouched.
    assert (snapshot / "delegated_new.txt").read_text(encoding="utf-8") == \
        DEFAULT_WORKSPACE_EDITS["delegated_new.txt"]
    assert (snapshot / "tracked.txt").read_text(encoding="utf-8") == \
        DEFAULT_WORKSPACE_EDITS["tracked.txt"]
    assert (live / "tracked.txt").read_text(encoding="utf-8") == "one\n"
    assert not (live / "delegated_new.txt").exists()

    # The applied CONTAINMENT facts are readable by the tree's own reader, and a
    # mechanism is only believed WITH its proven denied path.
    attempts = attempt_containment(str(tmp_path / "runs" / run_id))
    assert len(attempts) == 1, attempts
    assert attempts[0].home_isolated is True
    assert attempts[0].boundary_mechanism == "fake-sandbox"
    assert attempts[0].home_dir


# ===========================================================================
# Shared scenario pieces
# ===========================================================================

_BUILDER_ROW = {
    "subagent_id": "cx-builder",
    "recommended_use": "Delegated builder for the system_e2e mutation scenarios.",
    "route": {"kind": "agent_session", "target_id": "fake-harness=mock-model"},
    "effort": "low",
}
S24_MARKER = "S24_PARENT_FINAL_e2e_w5"
S25_MARKER = "S25_PARENT_FINAL_e2e_w5"
_MUTATE_PROMPT = FAKE_MUTATE_MARKER + " add the delegated file and extend tracked.txt"


def _git(args, cwd, check: bool = True) -> str:
    proc = subprocess.run(["git", *args], cwd=str(cwd), check=check,
                          capture_output=True, text=True)
    return (proc.stdout or "").strip()


def _seed_workspace(root: pathlib.Path) -> pathlib.Path:
    """A real external workspace: a Git tree with one committed tracked file."""
    workspace = pathlib.Path(root) / "workspace"
    workspace.mkdir(parents=True)
    _git(["init", "-q", "-b", "main"], workspace)
    _git(["config", "user.name", "SystemHarness"], workspace)
    _git(["config", "user.email", "system-harness@e2e.invalid"], workspace)
    (workspace / "tracked.txt").write_text("one\n", encoding="utf-8")
    _git(["add", "-A"], workspace)
    _git(["commit", "-q", "-m", "seed"], workspace)
    return workspace


def _mutating_settings(stub: ScriptedStubModel, root: pathlib.Path) -> dict:
    """Keyless settings whose snapshot root lives INSIDE the scenario tmp tree.

    The default snapshot root is the install's own ``subagent_worktrees`` directory;
    a scenario that left it there would provision the run's private worktree outside
    everything the test owns.
    """
    return keyless_settings(
        stub,
        OUROBOROS_SUBAGENTS=_roster(_BUILDER_ROW),
        OUROBOROS_SUBAGENT_WORKTREE_ROOT=str(pathlib.Path(root) / "snaps"),
    )


def _run_id_from(body: dict) -> str:
    ids = _RUN_ID_RE.findall(body_text(body))
    return ids[-1] if ids else ""


def _capture_dir(data_root: pathlib.Path, task_id: str, snapshot_id: str) -> pathlib.Path:
    return (pathlib.Path(data_root) / "task_results" / "artifacts" / str(task_id)
            / "delegated_runs" / str(snapshot_id))


def _registry_rows(data_root: pathlib.Path, snapshot_id: str) -> list:
    """The durable snapshot-registry rows for one snapshot id (empty = released)."""
    path = pathlib.Path(data_root) / "state" / "subagent_worktrees.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = raw.get("worktrees") if isinstance(raw, dict) else raw
    return [row for row in (rows or []) if isinstance(row, dict)
            and str(row.get("snapshot_id") or "") == str(snapshot_id)]


# ===========================================================================
# S24 — mutating delegated run, clean pull-in
# ===========================================================================


@pytest.mark.integration
@pytest.mark.serial
def test_s24_mutating_delegated_run_is_isolated_until_an_explicit_clean_apply(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s24")
    data_root = pathlib.Path(root) / "data"
    workspace = _seed_workspace(root)
    # What the LIVE tree looked like at the moment the model was asked for the step
    # AFTER the terminal wait: the isolation proof, taken causally (the script step
    # runs in this process, between the wait's result and the integrate call) rather
    # than by racing the server with a sleep.
    observed: dict = {}

    def _integrate_step(body: dict) -> dict:
        run_id = _run_id_from(body)
        if not run_id:
            return {"final": "E2E_SCRIPT_ERROR: no delegated run id visible"}
        observed["new_file_before_apply"] = (workspace / "delegated_new.txt").exists()
        observed["tracked_before_apply"] = (workspace / "tracked.txt").read_text(encoding="utf-8")
        return {"tool": "integrate_delegated_patch",
                "arguments": {"run_id": run_id, "decision": "apply",
                              "reason": "reviewed the captured diff"}}

    script = [
        {"tool": "delegate_start", "arguments": {
            "subagent_id": "cx-builder", "prompt": _MUTATE_PROMPT}},
        _wait_step,
        _integrate_step,
    ]
    with FakeClaudexorDaemon() as daemon, \
            ScriptedStubModel(script, final_answer=f"{S24_MARKER}: patch integrated.") as stub:
        daemon.install(data_root / "claudexor")
        server = start_server(e2e_clone, root, _mutating_settings(stub, root))
        try:
            task_id = submit_running(
                server,
                "Delegate the workspace change to your builder, wait it out, then "
                "integrate its captured patch.",
                workspace_root=str(workspace))
            result = server.wait_task(task_id, timeout=600)
            assert result.get("status") == "completed", result
            oracle = ArtifactOracle(server.data_root)
            stored = wait_durable_result(oracle, task_id)
            assert S24_MARKER in str(stored.get("result") or ""), stored
            assert stub.script_consumed(), "S24 script was not fully consumed"

            # -- the run was admitted as MUTATING, into its own snapshot ------
            started = _custody_rows(oracle, "delegate_run_started")
            assert len(started) == 1, started
            entry = started[0]
            run_id = str(entry.get("run_id") or "")
            snapshot_id = str(entry.get("snapshot_id") or "")
            execution_root = str(entry.get("execution_root") or "")
            assert entry.get("access") == "workspace_write", entry
            assert entry.get("mode") == "agent", entry
            assert snapshot_id and execution_root and entry.get("baseline_sha"), entry
            assert entry.get("target_root") == str(workspace), entry
            assert execution_root != str(workspace), entry
            assert not pathlib.Path(execution_root).is_relative_to(workspace), entry

            # -- the wire said the same thing ---------------------------------
            posts = daemon.run_start_posts()
            assert len(posts) == 1, posts
            body = posts[0]["body"]
            assert body.get("access") == "workspace_write" and body.get("mode") == "agent", body
            assert body.get("execution") == {"isolation": "live", "delegated": True,
                                             "workspaceRoot": execution_root}, body
            assert (body.get("scope") or {}).get("root") == str(workspace), body

            # -- ISOLATION: nothing reached the live tree before the decision --
            assert observed.get("new_file_before_apply") is False, observed
            assert observed.get("tracked_before_apply") == "one\n", observed

            # -- the capture is a durable artifact, sha-verified ---------------
            cap_dir = _capture_dir(server.data_root, task_id, snapshot_id)
            manifest = json.loads((cap_dir / "workspace_patch.json").read_text(encoding="utf-8"))
            assert manifest.get("status") == "ready_with_changes", manifest
            assert "tracked.txt" in manifest.get("tracked_changed", []), manifest
            assert "delegated_new.txt" in manifest.get("untracked_included", []), manifest
            assert (cap_dir / "workspace.patch").is_file()
            assert (cap_dir / "baseline_manifest.json").is_file()
            baseline = json.loads((cap_dir / "baseline_manifest.json").read_text(encoding="utf-8"))
            assert baseline.get("snapshot_id") == snapshot_id, baseline
            assert baseline.get("target_root") == str(workspace), baseline

            # -- the disposition chain, durable and typed ---------------------
            assert _custody_rows(oracle, "delegate_run_patch_captured", run_id)
            assert _custody_rows(oracle, "delegate_run_patch_apply_started", run_id)
            disposed = _custody_rows(oracle, "delegate_run_patch_disposed", run_id)
            assert disposed and disposed[-1].get("disposition") == "applied", disposed
            verdicts = [row for row in oracle.events("delegate_run_patch_verdict")
                        if str(row.get("child_task_id") or "") == f"run_{run_id}"]
            assert verdicts, oracle.events("delegate_run_patch_verdict")
            assert verdicts[-1].get("pipeline") == "delegated", verdicts[-1]
            assert verdicts[-1].get("applied") is True, verdicts[-1]
            assert verdicts[-1].get("patch_sha256") == manifest.get("sha256"), verdicts[-1]

            # -- CONTAINMENT: the applied facts were READ, and reached the model
            transcript = "\n".join(body_text(call_body) for _kind, call_body in stub.calls)
            assert '"os_boundary": "fake-sandbox"' in transcript, transcript[-2000:]
            assert not _custody_rows(oracle, "delegate_run_unconfined", run_id), (
                "the run served applied containment facts, yet was recorded unconfined")

            # -- the live workspace now carries the run's work, STAGED ---------
            assert (workspace / "delegated_new.txt").read_text(encoding="utf-8") == \
                DEFAULT_WORKSPACE_EDITS["delegated_new.txt"]
            assert (workspace / "tracked.txt").read_text(encoding="utf-8") == \
                DEFAULT_WORKSPACE_EDITS["tracked.txt"]
            staged = _git(["diff", "--cached", "--name-only"], workspace).split()
            assert {"delegated_new.txt", "tracked.txt"} <= set(staged), staged
            assert _git(["rev-list", "--count", "HEAD"], workspace) == "1", (
                "integration committed on the owner's behalf")

            # -- the snapshot was released by the recorded disposition ---------
            assert not _registry_rows(server.data_root, snapshot_id)
            assert not pathlib.Path(execution_root).exists()
        finally:
            server.stop()


# ===========================================================================
# S25 — mutating delegated run, conflicting pull-in
# ===========================================================================


@pytest.mark.integration
@pytest.mark.serial
def test_s25_mutating_delegated_patch_conflict_is_refused_and_keeps_its_material(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s25")
    data_root = pathlib.Path(root) / "data"
    workspace = _seed_workspace(root)
    owner_text = "the owner's own edit\n"

    def _drift_then_integrate(body: dict) -> dict:
        """The interleaving that makes a conflict REAL: the live tree moves on a
        patched path AFTER the capture and BEFORE the decision. Written from the
        script step itself, so the ordering is causal rather than timed."""
        run_id = _run_id_from(body)
        if not run_id:
            return {"final": "E2E_SCRIPT_ERROR: no delegated run id visible"}
        (workspace / "tracked.txt").write_text(owner_text, encoding="utf-8")
        return {"tool": "integrate_delegated_patch",
                "arguments": {"run_id": run_id, "decision": "apply"}}

    script = [
        {"tool": "delegate_start", "arguments": {
            "subagent_id": "cx-builder", "prompt": _MUTATE_PROMPT}},
        _wait_step,
        _drift_then_integrate,
    ]
    with FakeClaudexorDaemon() as daemon, \
            ScriptedStubModel(script, final_answer=f"{S25_MARKER}: conflict is mine to resolve.") as stub:
        daemon.install(data_root / "claudexor")
        server = start_server(e2e_clone, root, _mutating_settings(stub, root))
        try:
            task_id = submit_running(
                server,
                "Delegate the workspace change to your builder, wait it out, then "
                "integrate its captured patch.",
                workspace_root=str(workspace))
            result = server.wait_task(task_id, timeout=600)
            # A task that ENDS holding an undisposed captured patch completes with a
            # DISCLOSED custody debt (upstream 09ac51b2, absorbed by F2): the model's
            # answer is kept verbatim, the execution axis stays the model's own work,
            # the one Reason line names the unreconciled custody, and the debt rides
            # objective.warning(s) plus the row's debt list — never a fabricated failure.
            assert result.get("status") == "completed", result
            oracle = ArtifactOracle(server.data_root)
            stored = wait_durable_result(oracle, task_id)
            assert S25_MARKER in str(stored.get("result") or ""), stored
            assert stub.script_consumed(), "S25 script was not fully consumed"
            assert stored.get("reason_code") == "delegated_custody_unreconciled", stored
            axes = stored.get("outcome_axes") or {}
            assert (axes.get("execution") or {}).get("status") == "ok", axes
            assert "delegated_custody_unreconciled" in ((axes.get("objective") or {}).get("warnings") or []), axes
            envelope = stored.get("delegate_terminal_reconciliation") or {}
            assert envelope.get("audit_status") == "ok", envelope

            started = _custody_rows(oracle, "delegate_run_started")
            assert len(started) == 1, started
            run_id = str(started[0].get("run_id") or "")
            snapshot_id = str(started[0].get("snapshot_id") or "")
            execution_root = str(started[0].get("execution_root") or "")

            assert stored.get("delegated_runs_unreconciled") == [f"patch:{run_id}"], stored
            assert envelope.get("undisposed_patch_run_ids") == [run_id], envelope

            # -- the refusal reached the model, typed and owned ----------------
            transcript = "\n".join(body_text(call_body) for _kind, call_body in stub.calls)
            assert "INTEGRATE_CONFLICT" in transcript, transcript[-2000:]
            assert "YOU own this conflict" in transcript, transcript[-2000:]

            # -- nothing was disposed, and the tree kept the OWNER's content ---
            assert not _custody_rows(oracle, "delegate_run_patch_disposed", run_id)
            assert (workspace / "tracked.txt").read_text(encoding="utf-8") == owner_text
            assert not (workspace / "delegated_new.txt").exists()
            assert not _git(["diff", "--cached", "--name-only"], workspace)
            resolved = _custody_rows(oracle, "delegate_run_patch_apply_resolved", run_id)
            assert resolved and resolved[-1].get("reason") == "baseline_drift", resolved

            # -- the resolution material SURVIVES: snapshot, registry, patch ---
            assert _registry_rows(server.data_root, snapshot_id), (
                "the conflicted run's snapshot row was released")
            assert pathlib.Path(execution_root).is_dir()
            cap_dir = _capture_dir(server.data_root, task_id, snapshot_id)
            assert (cap_dir / "workspace.patch").is_file()
            manifest = json.loads((cap_dir / "workspace_patch.json").read_text(encoding="utf-8"))
            assert manifest.get("status") == "ready_with_changes", manifest
        finally:
            server.stop()
