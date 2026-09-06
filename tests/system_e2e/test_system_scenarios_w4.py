"""S18-S23 — Ф4 wave 4 of the deep-integration suite (v7next plan §8 remainder).

Update variations, chat-lineage cancellation, evolution absorb kill-recovery and
the delegated interactive answer, keyless throughout, every assertion a DURABLE
artifact / a recorded WIRE fact / the byte truth of the worktree:

* S18 — MANAGED UPDATE, CARRIER PATH: a diverged fork (a local commit pinning its
  own ``VERSION``) takes an official update that bumps ``VERSION`` — a merge
  conflict confined to the declared version-carrier span. The carrier engine
  resolves it to the official side inside the isolated planner (plan kind=clean,
  ``carrier_resolved_paths == ["VERSION"]``), the apply lands a real 2-parent
  merge commit, and after the re-exec EVERY carrier (pyproject, web/package.json,
  README badge, ARCHITECTURE header, uv.lock, install pages) names the official
  version — the Q8 projection transferred the carriers, none drifted. The
  configured update source (W2-F2) survives the whole cycle.
* S19 — MANAGED UPDATE, CONFLICTING → TYPED REFUSAL, TREE INTACT: a genuine code
  conflict routes the smart apply to the assisted lane, whose admission refuses
  TYPED when the model budget is exhausted (the ledger is seeded to the exact
  configured limit through the tree's own usage-accounting writers). Nothing is
  changed: the full worktree fingerprint (dirty work included) is byte-identical
  before and after, HEAD never moves, no MERGE_HEAD, no tx marker survives, no
  assisted resolver task is enqueued, and the server stays healthy.
* S20 — MANAGED UPDATE, CRASH MID-APPLY (subprocess driver on a real isolated
  install, the S10 idiom): three honest boot-finalize outcomes — (a) crash
  between the durable ``stashing_local_work`` marker and the ``stash_sha``
  write → boot looks the stash up by attempt id, restores the owner's work
  uncommitted and clears the marker (nothing was applied); (b) a half-written
  ``pending_boot_smoke`` tx whose merge commit never reached HEAD → boot rolls
  back typed (no junk ``failed-update-*`` ref is minted for a non-attempt);
  (c) merge applied + crash before the restart smoke → boot runs the smoke,
  finalizes, restores the stashed dirty work and clears the tx.
* S21 — CANCELLATION WITH CHAT LINEAGE (the wave-2 W2-F1 counterpart of S7): the
  ONLY input delta vs S7 is ``chat_id`` on the task-create body, and it flips the
  owed-answer accounting from the ``no_lineage_chat`` handoff row to the real
  thing: an outbox delivery ``cancel:<task>:<request>`` registered AND delivered,
  the ``cancel_receipt`` chat row durably in ``logs/chat.jsonl`` (no WS client
  ever connected — the outbox is crash insurance, not a presence buffer), the
  ``cancel_receipt`` block on the stored result, and the same
  requested→claimed→settled forensics.
* S22 — EVOLUTION ABSORB KILL-RECOVERY: a REAL evolution cycle (campaign seeded,
  supervisor mints the task, the scripted agent lands a reviewed commit through
  the blocking triad+scope organ) settles with auto-restart disabled, which (W4-F3,
  owner 5 = A) skips ONLY the restart: generation A pins that the exact
  restart-verify marker IS still written and the tree stays up, then SIGKILLs the
  tree and removes the marker — the durable state a crash between the campaign's
  ``waiting_for_restart`` write and the marker write leaves. Generation B
  boots on the same clone+data root and the markerless boot reconciliation
  absorbs the cycle exactly once (``verified_by=boot_reconciliation``, commit
  still on HEAD — no loss); generation C boots again and NOTHING is absorbed
  twice (no second count, no duplicate history row, HEAD unmoved).
* S23 — DELEGATED INTERACTIVE ANSWER: a ``[FAKE:ASK]`` run pauses on a pending
  interaction; ``delegate_wait`` returns IMMEDIATELY as ``waiting_on_user`` with
  the full question set; the nanny answers through ``delegate_answer`` (the
  exact ``questionId``/``selectedLabels`` wire lands on
  ``POST /v2/runs/:id/interactions/:iid/answer``); the run resumes and settles.
  Custody order is pinned (STARTED < INTERACTION_ANSWERED < SETTLED, one run id,
  ONE physical POST, no cancel), and the last-delegation receipt stays honest.

The default-lane tests pin the fake daemon's NEW interactive surface with the
REAL gateway client (pendingInteractions normalization, the answer verb's typed
delivered / already_resolved / rejected statuses, resume-to-terminal).
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import signal
import subprocess
import sys

import pytest

from tests.system_e2e.harness import (
    LANE_MOCK,
    REPO_ROOT,
    ArtifactOracle,
    ScriptedStubModel,
    body_text,
    clone_repo,
    keyless_settings,
    pids_with_env_value,
    process_tree_pids,
    require_lane,
    start_server,
    submit_running,
    wait_durable_result,
    wait_until,
)
from tests.system_e2e.interfaces import (
    FAKE_ANSWER_LABEL,
    FAKE_ASK_MARKER,
    FAKE_QUESTION_TEXT,
    FakeClaudexorDaemon,
)
from tests.system_e2e.test_system_scenarios_w2 import (
    KEEPALIVE_STEP,
    OFFICIAL_UPDATE_URL,
    _git,
)
from tests.system_e2e.test_system_scenarios_w3b import (
    _RUN_ID_RE,
    _SCOUT_ROW,
    _custody_rows,
    _roster,
    _wait_step,
)

from devtools.benchmarks.common.server_runner import (
    _api,
    _api_status,
    seed_owner_state,
)

# ===========================================================================
# Default lane: the fake daemon's interactive surface, pinned with the REAL
# gateway client (no server, loopback only).
# ===========================================================================


def test_fake_daemon_interactive_answer_contract(tmp_path):
    from ouroboros.gateways.claudexor import (
        ClaudexorGateway,
        discover_daemon_at,
        pending_interactions,
    )

    with FakeClaudexorDaemon(runs_dir=tmp_path / "runs") as daemon:
        daemon.install(tmp_path / "cx")
        with ClaudexorGateway(discover_daemon_at(tmp_path / "cx")) as gateway:
            gateway.handshake()
            request = {
                "prompt": FAKE_ASK_MARKER + " probe", "instructions": "i",
                "authPreference": "subscription", "mode": "ask",
                "scope": {"kind": "project", "root": str(tmp_path)},
                "harnesses": [daemon.harness_id], "primaryHarness": daemon.harness_id,
                "access": "readonly", "maxSeconds": 60,
            }
            run_id = str(gateway.start_run(request, idempotency_key="inv-ask-1")["runId"])
            # The run WAITS: state stays running, the detail carries the full
            # pending interaction, and the tree's own normalizer reads it whole.
            detail = gateway.get_run(run_id)
            assert (detail.get("summary") or {}).get("state") == "running"
            assert (detail.get("summary") or {}).get("waitingOnUser") is True
            pending = pending_interactions(detail)
            assert len(pending) == 1, detail
            row = pending[0]
            iid = row["interaction_id"]
            assert iid and row["source_tool"] == "AskUserQuestion"
            question = row["questions"][0]
            assert question["question_id"] == "q1"
            assert question["question"] == FAKE_QUESTION_TEXT
            assert {opt["label"] for opt in question["options"]} == {FAKE_ANSWER_LABEL, "9090"}
            assert question["multi_select"] is False
            # Typed refusals at their real HTTP codes, returned AS ANSWERS.
            wrong = gateway.answer_interaction(run_id, "int-nonexistent", [
                {"questionId": "q1", "selectedLabels": [FAKE_ANSWER_LABEL], "freeText": None}])
            assert wrong.get("status") == "already_resolved"
            empty = gateway.answer_interaction(run_id, iid, [])
            assert empty.get("status") == "rejected"
            # Still pending after both refusals; the real answer delivers.
            assert pending_interactions(gateway.get_run(run_id)), "refusal consumed the question"
            answered = gateway.answer_interaction(run_id, iid, [
                {"questionId": "q1", "selectedLabels": [FAKE_ANSWER_LABEL], "freeText": None}])
            assert answered.get("status") == "delivered"
            assert answered.get("accepted") is True
            # The run resumes: the next poll is terminal with settlement facts,
            # and a late duplicate answer is already_resolved, never a re-run.
            done = gateway.get_run(run_id)
            assert (done.get("summary") or {}).get("state") == "succeeded"
            assert (done.get("summary") or {}).get("waitingOnUser") is False
            assert pending_interactions(done) == []
            late = gateway.answer_interaction(run_id, iid, [
                {"questionId": "q1", "selectedLabels": [FAKE_ANSWER_LABEL], "freeText": None}])
            assert late.get("status") == "already_resolved"
            assert len(daemon.run_start_posts()) == 1
            answer_posts = daemon.calls("POST", f"/v2/runs/{run_id}/interactions/")
            # wrong-iid + empty + delivered + late duplicate = four answer POSTs.
            assert [p["path"].endswith("/answer") for p in answer_posts] == [True] * 4
            assert daemon.runs[run_id]["answers"] == [{
                "interaction_id": iid,
                "answers": [{"questionId": "q1", "selectedLabels": [FAKE_ANSWER_LABEL],
                             "freeText": None}],
            }]


# ===========================================================================
# Shared helpers of the wave-4 update scenarios
# ===========================================================================

S18_OFFICIAL_VERSION = "9.9.9"


def _tree_fingerprint(clone: pathlib.Path) -> str:
    """sha256 over every non-.git file (path + content; symlink targets) — the
    byte truth "tree intact" is asserted against."""
    digest = hashlib.sha256()
    clone = pathlib.Path(clone)
    for path in sorted(clone.rglob("*")):
        rel = path.relative_to(clone)
        if rel.parts and rel.parts[0] == ".git":
            continue
        # __pycache__ is machine-generated bytecode the SERVER SUBPROCESS writes
        # while importing this clone (the isolated server sets no
        # PYTHONPYCACHEPREFIX). It is never repo content, so a .pyc appearing
        # between the baseline and the assertion is not evidence that the
        # refused update touched the worktree — but it DID flip this digest
        # under load, when a late first-import landed after the snapshot.
        if "__pycache__" in rel.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            digest.update(b"L" + bytes(rel) + os.readlink(path).encode())
        elif path.is_file():
            digest.update(b"F" + bytes(rel))
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _configured_managed_install(root: pathlib.Path):
    """A managed install whose CONFIGURED update source is a local upstream clone
    (``managed_remote_url`` in the managed metadata — the W2-F2 fork-install
    shape), with the hardcoded official URL belt-redirected to a non-existent
    path so a regressed repin fails loudly without network egress. Returns
    ``(clone, upstream)`` with NO divergence yet — scenarios author their own
    local/official commits."""
    clone = clone_repo(root)
    upstream = pathlib.Path(root) / "upstream"
    subprocess.run(["git", "clone", "--no-hardlinks", "-q", str(clone), str(upstream)],
                   check=True, capture_output=True)
    _git(["checkout", "-q", "ouroboros"], upstream)
    _git(["config", "user.name", "Managed Upstream"], upstream)
    _git(["config", "user.email", "upstream@e2e.invalid"], upstream)
    _git(["remote", "add", "managed", str(upstream)], clone)
    _git(["config",
          f"url.{pathlib.Path(root) / 'nonexistent-official-mirror'}.insteadOf",
          OFFICIAL_UPDATE_URL], clone)
    (clone / ".git" / "ouroboros-managed.json").write_text(json.dumps({
        "managed_remote_name": "managed",
        "managed_remote_url": str(upstream),
        "managed_remote_branch": "ouroboros",
        "managed_local_branch": "ouroboros",
    }), encoding="utf-8")
    return clone, upstream


# ===========================================================================
# S18 — managed update carrier path: span-confined VERSION conflict,
# carriers transferred to the official version.
# ===========================================================================


@pytest.mark.integration
@pytest.mark.serial
def test_s18_carrier_conflict_resolves_to_official_version_and_transfers_carriers(
        tmp_path_factory, monkeypatch):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s18")
    clone, upstream = _configured_managed_install(root)

    # Official side: a release bump of the VERSION carrier.
    (upstream / "VERSION").write_text(S18_OFFICIAL_VERSION + "\n", encoding="utf-8")
    _git(["add", "VERSION"], upstream)
    _git(["commit", "-q", "-m", f"release: v{S18_OFFICIAL_VERSION} (system_e2e S18)"], upstream)
    target_sha = _git(["rev-parse", "HEAD"], upstream)
    # Fork side: its own committed VERSION pin — the same carrier span, so the
    # merge conflicts EXACTLY inside a declared version-carrier span.
    (clone / "VERSION").write_text("0.0.1\n", encoding="utf-8")
    _git(["add", "VERSION"], clone)
    _git(["commit", "-q", "-m", "fork: local version pin (system_e2e S18)"], clone)
    base_sha = _git(["rev-parse", "HEAD"], clone)

    monkeypatch.setenv("PIP_NO_INDEX", "1")  # same belt as S9: pip must be a no-op
    with ScriptedStubModel([]) as stub:
        settings = keyless_settings(stub, OUROBOROS_UPDATE_CHANNEL="development")
        server = start_server(clone, root, settings)
        try:
            oracle = ArtifactOracle(server.data_root)
            plan = (_api(server.base_url, "POST", "/api/update/preflight", {}, timeout=300)
                    .get("merge_plan") or {})
            # The span-confined conflict left the plan's conflict inventory: the
            # carrier engine resolved it to the official side in the ISOLATED
            # planner worktree, and the plan stayed on the clean auto lane.
            assert plan.get("available") is True and plan.get("kind") == "clean", plan
            assert plan.get("auto_mergeable") is True, plan
            assert plan.get("carrier_resolved_paths") == ["VERSION"], plan
            assert plan.get("code_conflict_paths") == [] and plan.get("doc_conflict_paths") == [], plan
            assert plan.get("base_sha") == base_sha and plan.get("target_sha") == target_sha, plan
            assert plan.get("recommended_strategy") == "auto_merge", plan

            applied = _api(server.base_url, "POST", "/api/update/apply", {
                "strategy": "auto_merge",
                "expected_base_sha": base_sha,
                "expected_target_sha": target_sha,
            }, timeout=600)
            assert applied.get("status") == "ok" and applied.get("restarting") is True, applied
            merge_commit = str((applied.get("merge_plan") or {}).get("merge_commit") or "")
            assert merge_commit and merge_commit not in (base_sha, target_sha), applied

            # Boot-finalize honesty (the S9 contract): tx consumed, receipt names
            # the merged HEAD, the server is back.
            tx_marker = clone / ".git" / "ouroboros-update-tx.json"
            assert wait_until(lambda: not tx_marker.exists(), 300), "update tx marker never cleared"
            assert server.wait_for_health(300)
            finalized = wait_until(
                lambda: oracle.supervisor_rows("managed_update_finalized") or None, 60)
            assert finalized, "no managed_update_finalized receipt"
            assert finalized[-1].get("head") == merge_commit, finalized[-1]

            # The landed commit is a REAL 2-parent merge [fork base, official target].
            assert _git(["rev-parse", "HEAD"], clone) == merge_commit
            assert _git(["rev-parse", "HEAD^1"], clone) == base_sha
            assert _git(["rev-parse", "HEAD^2"], clone) == target_sha

            # THE CARRIER TRANSFER: the update landed under the official version
            # and every mechanical carrier token was projected to it (Q8) — the
            # tree's own carrier-sync authority reports zero desyncs.
            from ouroboros.tools.release_sync import (
                check_worktree_version_sync,
                extract_architecture_header_version,
                extract_readme_badge_version,
            )

            assert (clone / "VERSION").read_text(encoding="utf-8").strip() == S18_OFFICIAL_VERSION
            readme = (clone / "README.md").read_text(encoding="utf-8")
            assert extract_readme_badge_version(readme) == S18_OFFICIAL_VERSION, "README badge not transferred"
            arch = (clone / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
            assert extract_architecture_header_version(arch) == S18_OFFICIAL_VERSION, "ARCHITECTURE header not transferred"
            assert f'version = "{S18_OFFICIAL_VERSION}"' in (clone / "pyproject.toml").read_text(encoding="utf-8")
            assert check_worktree_version_sync(clone) == "", "carrier desync after the update"

            # W2-F2: the configured source survived the whole cycle.
            assert _git(["remote", "get-url", "managed"], clone) == str(upstream), (
                "configured update source was retargeted")
        finally:
            server.stop()


# ===========================================================================
# S19 — conflicting update: typed budget refusal at the assisted gate,
# tree byte-identical.
# ===========================================================================

S19_CONFLICT_PATH = "Makefile"
S19_TRACKED_DIRTY = "CONTRIBUTING.md"
S19_DIRTY_MARK = "e2e-w4: dirty work must survive the refused update\n"
S19_UNTRACKED = "e2e_w4_local_note.txt"


def _seed_exhausted_budget(data_root: pathlib.Path, limit_usd: float) -> None:
    """Consume the whole configured budget through the tree's OWN monetary
    writers (reserve → dispatch → settle at exactly the limit), so
    ``budget_remaining`` honestly reports zero — no forged ledger rows."""
    from ouroboros.usage_accounting import (
        AttemptRequest,
        mark_dispatched,
        reserve_attempt,
        settle_attempt,
    )

    data_root.mkdir(parents=True, exist_ok=True)
    reservation = reserve_attempt(AttemptRequest(
        model="mock-model", provider="openai-compatible",
        drive_root=data_root, task_id="e2e-s19-seed", root_task_id="e2e-s19-seed",
        category="agent", source="system_e2e",
    ))
    mark_dispatched(reservation)
    settle_attempt(reservation, {"prompt_tokens": 1, "completion_tokens": 1},
                   cost_usd=float(limit_usd), cost_final=True)


@pytest.mark.integration
@pytest.mark.serial
def test_s19_conflicting_update_refused_typed_with_tree_intact(
        tmp_path_factory, monkeypatch):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s19")
    clone, upstream = _configured_managed_install(root)

    # A GENUINE code conflict: both sides rewrite the same tracked file.
    (upstream / S19_CONFLICT_PATH).write_text(
        "# official side of the system_e2e S19 conflict\nall:\n\ttrue\n", encoding="utf-8")
    _git(["add", S19_CONFLICT_PATH], upstream)
    _git(["commit", "-q", "-m", "build: official Makefile rework (system_e2e S19)"], upstream)
    target_sha = _git(["rev-parse", "HEAD"], upstream)
    (clone / S19_CONFLICT_PATH).write_text(
        "# fork side of the system_e2e S19 conflict\nall:\n\tfalse\n", encoding="utf-8")
    _git(["add", S19_CONFLICT_PATH], clone)
    _git(["commit", "-q", "-m", "build: fork Makefile rework (system_e2e S19)"], clone)
    base_sha = _git(["rev-parse", "HEAD"], clone)

    # Exhaust the budget BEFORE boot: the keyless settings pin TOTAL_BUDGET=10.0
    # and the seeded settled attempt consumes exactly that.
    _seed_exhausted_budget(pathlib.Path(root) / "data", 10.0)

    monkeypatch.setenv("PIP_NO_INDEX", "1")
    with ScriptedStubModel([]) as stub:
        settings = keyless_settings(stub, OUROBOROS_UPDATE_CHANNEL="development")
        assert float(settings["TOTAL_BUDGET"]) == 10.0, "budget seed and settings drifted apart"
        server = start_server(clone, root, settings)
        try:
            oracle = ArtifactOracle(server.data_root)
            # Dirty local work on top: the refusal must bring it back untouched.
            tracked = clone / S19_TRACKED_DIRTY
            tracked.write_text(tracked.read_text(encoding="utf-8") + S19_DIRTY_MARK,
                               encoding="utf-8")
            (clone / S19_UNTRACKED).write_text("local operator note\n", encoding="utf-8")

            plan = (_api(server.base_url, "POST", "/api/update/preflight", {}, timeout=300)
                    .get("merge_plan") or {})
            assert plan.get("available") is True and plan.get("kind") == "conflicting", plan
            assert plan.get("code_conflict_paths") == [S19_CONFLICT_PATH], plan
            assert plan.get("recommended_strategy") == "assisted", plan
            assert plan.get("base_sha") == base_sha and plan.get("target_sha") == target_sha, plan

            before = _tree_fingerprint(clone)
            envelope = _api_status(server.base_url, "POST", "/api/update/apply", {
                "strategy": "auto_merge",
                "expected_base_sha": base_sha,
                "expected_target_sha": target_sha,
            }, timeout=600)
            # The assisted admission refuses TYPED on the exhausted budget.
            assert envelope.get("status") == 409, envelope
            error = str((envelope.get("body") or {}).get("error") or "")
            assert "model budget" in error and "nothing was changed" in error, envelope

            # TREE INTACT — the byte truth, not a porcelain glance: HEAD unmoved,
            # no live merge, no tx marker, dirty work restored, and the FULL
            # worktree fingerprint identical to the pre-apply snapshot.
            assert _git(["rev-parse", "HEAD"], clone) == base_sha
            assert _git(["rev-parse", "--abbrev-ref", "HEAD"], clone) == "ouroboros"
            merge_head = subprocess.run(
                ["git", "rev-parse", "-q", "--verify", "MERGE_HEAD"], cwd=str(clone),
                capture_output=True, text=True)
            assert merge_head.returncode != 0, "a staged merge survived the refusal"
            assert not (clone / ".git" / "ouroboros-update-tx.json").exists(), (
                "the update tx marker survived the refusal")
            assert tracked.read_text(encoding="utf-8").endswith(S19_DIRTY_MARK)
            assert (clone / S19_UNTRACKED).is_file()
            assert _tree_fingerprint(clone) == before, (
                "worktree is not byte-identical after the typed refusal")

            # No assisted resolver task exists anywhere in the durable queue.
            snapshot = json.dumps(oracle.queue_snapshot())
            assert "update_assisted_merge_" not in snapshot, snapshot
            # The server survived the refusal (writers respawned).
            assert server.wait_for_health(120)
        finally:
            server.stop()


# ===========================================================================
# S20 — crash mid-apply: boot-finalize honesty (subprocess driver, S10 idiom).
# ===========================================================================

S20_DRIVER = r'''
import json, os, pathlib, subprocess, sys

clone = pathlib.Path(sys.argv[1])
data = pathlib.Path(sys.argv[2])

from supervisor import git_ops
git_ops.init(clone, data, "")
from supervisor.update_merge import (
    finalize_managed_update_on_boot,
    stash_local_changes_for_update,
    write_update_tx,
)

marker = clone / ".git" / "ouroboros-update-tx.json"
report = {}

def _git(*args):
    proc = subprocess.run(["git", *args], cwd=str(clone), check=True,
                          capture_output=True, text=True)
    return (proc.stdout or "").strip()

def _supervisor_rows(row_type):
    path = data / "logs" / "supervisor.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("type") == row_type:
                rows.append(row)
    return rows

DIRTY = clone / "CONTRIBUTING.md"
MARK = "e2e-w4 s20: local work in the crash window\n"
head0 = _git("rev-parse", "HEAD")

# --- Phase A: crash between the durable stashing_local_work marker and the
# stash_sha write. Boot must find the stash BY ATTEMPT ID, restore the work
# uncommitted, and clear the marker: nothing was applied.
orig = DIRTY.read_text(encoding="utf-8")
DIRTY.write_text(orig + MARK, encoding="utf-8")
(clone / "s20_untracked.txt").write_text("phase-a untracked\n", encoding="utf-8")
write_update_tx({"phase": "stashing_local_work", "pre_update_sha": head0,
                 "pre_update_branch": "ouroboros", "base_sha": head0,
                 "target_sha": "", "attempt_id": "e2ew4stasha",
                 "stash_sha": "", "local_work_carrier": "none"})
stash_status, stash_sha, stash_error = stash_local_changes_for_update("e2ew4stasha")
assert stash_status == "ok" and stash_sha, (stash_status, stash_error)
assert _git("status", "--porcelain") == ""
# CRASH HERE (the stash_sha never reached the tx marker).
boot_a = finalize_managed_update_on_boot(True)
report["stash_crash"] = {
    "boot": boot_a,
    "restored": DIRTY.read_text(encoding="utf-8").endswith(MARK),
    "untracked": (clone / "s20_untracked.txt").is_file(),
    "marker_gone": not marker.exists(),
    "head_moved": _git("rev-parse", "HEAD") != head0,
    "recovered_rows": len(_supervisor_rows("managed_update_stash_recovered_on_boot")),
}
_git("reset", "--hard", "HEAD")
_git("clean", "-fd")

# --- Phase B: a half-written pending_boot_smoke tx whose merge commit never
# reached HEAD (crash between the tx write and the apply). Boot must roll back
# TYPED — and must NOT mint a junk failed-update ref for a non-attempt.
(clone / "S20_FAKE_TARGET.txt").write_text("never applied\n", encoding="utf-8")
_git("add", "S20_FAKE_TARGET.txt")
_git("commit", "-q", "-m", "system_e2e s20 phase-b side target")
side_sha = _git("rev-parse", "HEAD")
_git("reset", "--hard", head0)
write_update_tx({"phase": "pending_boot_smoke", "pre_update_sha": head0,
                 "pre_update_branch": "ouroboros", "base_sha": head0,
                 "target_sha": side_sha, "merge_commit": side_sha,
                 "pre_restart_smoke": "pending", "attempt_id": "e2ew4halfb",
                 "stash_sha": "", "local_work_carrier": "none",
                 "rollback_attempted": False})
boot_b = finalize_managed_update_on_boot(True)
report["half_written"] = {
    "boot": boot_b,
    "head": _git("rev-parse", "HEAD"), "head0": head0,
    "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    "porcelain": _git("status", "--porcelain"),
    "marker_gone": not marker.exists(),
    "failed_refs": _git("branch", "--list", "failed-update-*"),
    "rollback_rows": len(_supervisor_rows("managed_update_rollback_after_failed_boot")),
    "rolled_back_rows": [row.get("pre_update_sha")
                         for row in _supervisor_rows("managed_update_rolled_back")],
}

# --- Phase C: the merge IS applied, the crash lands before the restart smoke.
# Boot must run the smoke, finalize, restore the stashed dirty work and clear
# the tx — the applied update survives, the owner's work comes back.
DIRTY.write_text(orig + MARK, encoding="utf-8")
(clone / "s20_untracked_c.txt").write_text("phase-c untracked\n", encoding="utf-8")
write_update_tx({"phase": "stashing_local_work", "pre_update_sha": head0,
                 "pre_update_branch": "ouroboros", "base_sha": head0,
                 "target_sha": "", "attempt_id": "e2ew4applyc",
                 "stash_sha": "", "local_work_carrier": "none"})
stash_status_c, stash_sha_c, stash_error_c = stash_local_changes_for_update("e2ew4applyc")
assert stash_status_c == "ok" and stash_sha_c, (stash_status_c, stash_error_c)
payload = clone / "docs" / "notes" / "s20_applied_payload.md"
payload.parent.mkdir(parents=True, exist_ok=True)
payload.write_text("# applied by the system_e2e S20 phase-c update\n", encoding="utf-8")
_git("add", str(payload.relative_to(clone)))
_git("commit", "-q", "-m", "system_e2e s20 phase-c applied update")
merge_c = _git("rev-parse", "HEAD")
write_update_tx({"phase": "pending_boot_smoke", "pre_update_sha": head0,
                 "pre_update_branch": "ouroboros", "base_sha": head0,
                 "target_sha": merge_c, "merge_commit": merge_c,
                 "pre_restart_smoke": "pending", "attempt_id": "e2ew4applyc",
                 "stash_sha": stash_sha_c, "local_work_carrier": "stash",
                 "rollback_attempted": False})
# CRASH HERE (applied, never restarted).
boot_c = finalize_managed_update_on_boot(True)
report["applied_crash"] = {
    "boot": boot_c,
    "head": _git("rev-parse", "HEAD"), "merge_c": merge_c,
    "restored": DIRTY.read_text(encoding="utf-8").endswith(MARK),
    "untracked": (clone / "s20_untracked_c.txt").is_file(),
    "payload_present": payload.is_file(),
    "marker_gone": not marker.exists(),
    "finalized_heads": [row.get("head")
                        for row in _supervisor_rows("managed_update_finalized")],
}
print(json.dumps(report))
'''


@pytest.mark.integration
@pytest.mark.serial
def test_s20_crash_mid_apply_boot_finalize_and_rollback_are_honest(tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s20")
    clone = clone_repo(root)
    data = pathlib.Path(root) / "data"
    (data / "logs").mkdir(parents=True, exist_ok=True)
    driver = pathlib.Path(root) / "s20_driver.py"
    driver.write_text(S20_DRIVER, encoding="utf-8")
    env = {
        **os.environ,
        "OUROBOROS_APP_ROOT": str(root),
        "OUROBOROS_REPO_DIR": str(clone),
        "OUROBOROS_DATA_DIR": str(data),
        "OUROBOROS_SETTINGS_PATH": str(data / "settings.json"),
        "PYTHONPATH": str(REPO_ROOT),
        "PIP_NO_INDEX": "1",  # phase-c smoke's dependency sync must be a no-op
    }
    proc = subprocess.run(
        [sys.executable, str(driver), str(clone), str(data)],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    report = json.loads(proc.stdout.strip().splitlines()[-1])

    # A: pre-apply stash crash — work restored uncommitted, marker cleared,
    # HEAD untouched, the durable recovery receipt written.
    stash_crash = report["stash_crash"]
    assert stash_crash["boot"].get("finalized") is False, stash_crash
    assert "recovered pre-apply stash crash" in str(stash_crash["boot"].get("reason") or ""), stash_crash
    assert stash_crash["restored"] is True and stash_crash["untracked"] is True, stash_crash
    assert stash_crash["marker_gone"] is True and stash_crash["head_moved"] is False, stash_crash
    assert stash_crash["recovered_rows"] == 1, stash_crash

    # B: half-written tx, merge never on HEAD — honest typed rollback, tree
    # clean at the pre-update sha, and NO junk failed-update ref for a
    # non-attempt (the branch would replay-clobber a real preserved attempt).
    half = report["half_written"]
    assert half["boot"].get("rolled_back") is True, half
    assert half["head"] == half["head0"], half
    assert half["branch"] == "ouroboros" and half["porcelain"] == "", half
    assert half["marker_gone"] is True, half
    assert half["failed_refs"].strip() == "", half
    assert half["rollback_rows"] == 1, half
    assert half["rolled_back_rows"] == [half["head0"]], half

    # C: applied-but-unrestarted — the boot smoke passes, the update FINALIZES
    # (never rolled back), the stashed dirty work comes back uncommitted.
    applied = report["applied_crash"]
    assert applied["boot"].get("finalized") is True, applied
    assert applied["head"] == applied["merge_c"], applied
    assert applied["restored"] is True and applied["untracked"] is True, applied
    assert applied["payload_present"] is True, applied
    assert applied["marker_gone"] is True, applied
    assert applied["finalized_heads"] == [applied["merge_c"]], applied


# ===========================================================================
# S21 — cancellation WITH chat lineage: outbox receipt + chat row + forensics.
# ===========================================================================

S21_PROJECT_NAME = "s21 lineage"  # a file-less project: its thread IS the lineage


@pytest.mark.integration
@pytest.mark.serial
def test_s21_chat_lineage_cancel_delivers_receipt_to_outbox_and_chat(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s21")
    with ScriptedStubModel([dict(KEEPALIVE_STEP) for _ in range(300)],
                           latency_sec=0.15) as stub:
        server = start_server(e2e_clone, root, keyless_settings(stub))
        try:
            oracle = ArtifactOracle(server.data_root)
            # The ONLY delta vs S7: the task is admitted WITH chat lineage. Under
            # the ingress capture rule (upstream 68eab3ea/fed16935, absorbed by
            # F2) an externally launched task lives in its REGISTERED project's
            # thread or in the hidden partition — never in a conversation of its
            # own — so the lineage is a project thread: register a file-less
            # project and address the task into its room.
            project = (_api(server.base_url, "POST", "/api/projects",
                            {"name": S21_PROJECT_NAME}, timeout=60).get("project") or {})
            project_id = str(project.get("id") or "")
            lineage_chat = int(project.get("chat_id") or 0)
            assert project_id and lineage_chat, project
            created = _api(server.base_url, "POST", "/api/tasks", {
                "description": "Keep listing the repository root until stopped.",
                "project_id": project_id,
                "chat_id": lineage_chat,
                "memory_mode": "forked",
                "actor_id": "e2e-driver", "source": "e2e-driver",
                "metadata": {"source": "e2e-driver", "delegation_role": "root"},
            }, timeout=60)
            task_id = str(created.get("task_id") or "")
            assert task_id, created
            assert wait_until(lambda: task_id in oracle.running_ids(), 120)

            envelope = server.cancel_task(task_id)
            assert envelope.get("status") == 200, envelope
            assert (envelope.get("body") or {}).get("ok") is True, envelope

            stored = wait_durable_result(oracle, task_id)
            assert stored.get("status") == "cancelled", stored
            assert str(stored.get("result") or "").strip(), "cancelled terminal owes an answer"

            # Forensics: same trail as S7 — and its request id NAMES the delivery.
            rows = [row for row in oracle.supervisor_rows("cancel_intent")
                    if str(row.get("task_id") or "") == task_id]
            events = {str(row.get("event") or "") for row in rows}
            assert {"requested", "claimed", "settled"} <= events, rows
            requested = next(row for row in rows if row.get("event") == "requested")
            assert requested.get("source") == "http_single", requested
            assert requested.get("scope") == "single", requested
            settled = next(row for row in rows if row.get("event") == "settled")
            assert settled.get("outcome") == "cancelled", settled
            delivery_id = f"cancel:{task_id}:{requested.get('request_id')}"

            # (a) The owed answer is a REAL outbox delivery — registered AND
            # DELIVERED (the pending row drains into `delivered` without any WS
            # client ever connecting; the outbox is crash insurance).
            assert wait_until(
                lambda: delivery_id in (oracle.terminal_deliveries().get("delivered") or []),
                90), oracle.terminal_deliveries()
            assert delivery_id not in (oracle.terminal_deliveries().get("pending") or {})

            # (a') ...and the receipt LANDED in the durable chat stream.
            def _receipt_rows():
                return [row for row in oracle._jsonl("logs/chat.jsonl",
                                                     type_filter="cancel_receipt")
                        if str(row.get("task_id") or "") == task_id]

            chat_rows = wait_until(_receipt_rows, 90)
            assert chat_rows, "no cancel_receipt row in chat.jsonl"
            chat_row = chat_rows[-1]
            assert chat_row.get("direction") == "system", chat_row
            assert int(chat_row.get("chat_id") or 0) == lineage_chat, chat_row
            assert f"Task {task_id} was cancelled" in str(chat_row.get("text") or ""), chat_row

            # (b) The details-panel block on the durable result (the W2-F1 fact
            # a chatless task never gets).
            receipt = wait_until(
                lambda: (oracle.task_result(task_id).get("cancel_receipt")
                         if isinstance(oracle.task_result(task_id).get("cancel_receipt"), dict)
                         else None), 90)
            assert receipt, oracle.task_result(task_id)
            assert receipt.get("settled_status") == "cancelled", receipt
            assert receipt.get("outcome") == "cancelled", receipt
            assert receipt.get("delivery_id") == delivery_id, receipt
            assert isinstance(receipt.get("salvage"), dict), receipt

            # The chatless degradation path did NOT fire: lineage flipped the
            # accounting to the outbox, not the typed handoff row.
            handoffs = [row for row in oracle.supervisor_rows("terminal_delivery_handoff")
                        if str(row.get("task_id") or "") == task_id]
            assert handoffs == [], handoffs

            assert wait_until(lambda: task_id not in oracle.cancel_intents(), 60)
            assert wait_until(lambda: task_id not in oracle.running_ids(), 60)
        finally:
            server.stop()


# ===========================================================================
# S22 — evolution absorb kill-recovery: crash after the reviewed commit,
# markerless boot reconcile absorbs exactly once.
# ===========================================================================

S22_DOC_PATH = "docs/notes/system_e2e_s22_evolution.md"
S22_COMMIT_MESSAGE = "docs: system_e2e S22 evolution cycle note"
S22_SCRIPT = [
    {"tool": "write_file", "arguments": {
        "root": "system_repo",
        "path": S22_DOC_PATH,
        "content": "# system_e2e S22\n\nSelf-evolution cycle payload.\n",
    }},
    {"tool": "commit_reviewed", "arguments": {
        "commit_message": S22_COMMIT_MESSAGE,
        "paths": [S22_DOC_PATH],
        "skip_advisory_review": True,
        "skip_tests": True,
        "goal": "Land the S22 evolution note through the blocking review organ.",
        "scope": f"{S22_DOC_PATH} only.",
    }},
]


def _campaign(data_root: pathlib.Path) -> dict:
    path = pathlib.Path(data_root) / "state" / "evolution_campaign.json"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _cycle_outcome_checkpoints(data_root: pathlib.Path) -> list:
    path = pathlib.Path(data_root) / "state" / "evolution_checkpoints.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("kind") == "cycle_outcome":
                rows.append(row)
    return rows


def _sigkill_server_tree(server) -> None:
    pids = process_tree_pids(server.proc.pid)
    os.killpg(os.getpgid(server.proc.pid), signal.SIGKILL)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    server.proc.wait(timeout=30)


@pytest.mark.integration
@pytest.mark.serial
@pytest.mark.skipif(sys.platform != "linux",
                    reason="process-group SIGKILL + /proc scans are Linux-only")
def test_s22_absorb_kill_recovery_absorbs_once_and_never_twice(
        tmp_path_factory, monkeypatch):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s22")
    clone = clone_repo(root)          # PRIVATE clone: this scenario moves HEAD
    data_root = pathlib.Path(root) / "data"
    seed_owner_state(data_root, evolution_enabled=True)
    campaign_seed = _campaign(data_root)
    assert campaign_seed.get("status") == "active", campaign_seed
    # Auto-restart OFF keeps the tree ALIVE once the cycle settles — since W4-F3
    # (owner 5 = A) the knob skips only the restart itself, and the marker IS
    # written (pinned below). The crash window under test — a crash between the
    # campaign's waiting_for_restart write and the marker write — leaves exactly
    # this durable state: commit on HEAD, open transaction, NO marker. The two
    # writes are separate atomic files, so the scenario shapes that state
    # deterministically by removing the marker after the kill.
    monkeypatch.setenv("OUROBOROS_EVOLUTION_AUTO_RESTART", "false")

    settings_kwargs = dict(
        OUROBOROS_RUNTIME_MODE="advanced",       # light hard-blocks evolution
        OUROBOROS_REVIEW_ENFORCEMENT="blocking",  # the landed commit PROVES the organ passed
    )
    marker_glob = "pending_restart_verify*"
    state_dir = data_root / "state"

    # ---- Generation A: the real evolution cycle, killed mid-absorb ----------
    commit_sha = ""
    transaction_id = ""
    with ScriptedStubModel(S22_SCRIPT) as stub:
        server = start_server(clone, root, keyless_settings(stub, **settings_kwargs))
        killed = False
        try:
            # The supervisor mints the evolution task itself once idle; the
            # scripted agent lands the reviewed commit and the campaign records
            # the receipt (restart_required, not yet verified).
            assert wait_until(
                lambda: str(((_campaign(data_root).get("active_transaction") or {})
                             ).get("commit_sha") or "") or None, 600), (
                "the evolution cycle never recorded a reviewed commit")
            tx = _campaign(data_root)["active_transaction"]
            commit_sha = str(tx["commit_sha"])
            transaction_id = str(tx.get("transaction_id") or "")
            assert transaction_id, tx
            assert _git(["rev-parse", "HEAD"], clone) == commit_sha
            # The cycle settles: waiting_for_restart, restart NOT verified. Auto-restart
            # off skips ONLY the restart (W4-F3): the exact restart-verify claim is
            # still written, and the tree stays up (the harness runs the server
            # without a launcher, so a restart exit would have ended the process).
            assert wait_until(
                lambda: str(((_campaign(data_root).get("active_transaction") or {})
                             ).get("cycle_outcome") or "") == "waiting_for_restart", 300), (
                _campaign(data_root))
            marker_path = state_dir / "pending_restart_verify.json"

            def _marker():
                try:
                    return json.loads(marker_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    return None

            marker = wait_until(_marker, 60)
            assert marker, "auto-restart off no longer writes the restart-verify marker (W4-F3)"
            assert marker.get("expected_sha") == commit_sha, marker
            assert marker.get("evolution_claim") == {
                "campaign_id": str(_campaign(data_root).get("id") or ""),
                "transaction_id": transaction_id,
                "task_id": str(tx.get("task_id") or ""),
                "commit_sha": commit_sha,
            }, marker
            assert server.proc.poll() is None, "auto-restart off must not restart the tree"
            assert "triad_review" in stub.kinds() and "scope_review" in stub.kinds(), (
                stub.kinds())

            # HARD CRASH in the window.
            _sigkill_server_tree(server)
            killed = True
            assert wait_until(lambda: not pids_with_env_value(str(server.data_root)), 30)
            # Shape the crash-window durable state: the campaign write and the
            # marker write are two atomic files, so a crash BETWEEN them leaves
            # exactly the campaign (open transaction, commit on HEAD) and no
            # marker — removing the marker after the kill is that state.
            for stale in state_dir.glob(marker_glob):
                stale.unlink()
            assert not list(state_dir.glob(marker_glob))
        finally:
            if not killed:
                server.stop()

    # Deterministic generation-B shape: the owner's evolution toggle goes OFF so
    # the supervisor cannot mint cycle 2 mid-scenario; the boot reconciliation
    # deliberately does not read the flag (campaign stays active).
    state_path = state_dir / "state.json"
    state_blob = json.loads(state_path.read_text(encoding="utf-8"))
    state_blob["evolution_mode_enabled"] = False
    state_path.write_text(json.dumps(state_blob), encoding="utf-8")

    # ---- Generation B: markerless boot reconcile absorbs the cycle ---------
    with ScriptedStubModel([{"tool": "list_files", "arguments": {"path": "."}}]) as stub_b:
        server_b = start_server(clone, root, keyless_settings(stub_b, **settings_kwargs))
        try:
            oracle = ArtifactOracle(server_b.data_root)
            # A worker boot runs the restart-verify path; with no marker at all
            # the dangling-transaction reconciliation must absorb the cycle.
            task_b = submit_running(server_b, "List the repository root and finish.")
            assert server_b.wait_task(task_b, timeout=300).get("status") == "completed"
            absorbed = wait_until(
                lambda: int(_campaign(data_root).get("absorbed_cycles_done") or 0) == 1, 300)
            assert absorbed, _campaign(data_root)
            campaign_b = _campaign(data_root)
            # NO LOSS: the transaction closed as absorbed by boot reconciliation,
            # the commit is still on HEAD, nothing dangles.
            assert "active_transaction" not in campaign_b, campaign_b
            history = [row for row in (campaign_b.get("transaction_history") or [])
                       if str(row.get("transaction_id") or "") == transaction_id]
            assert len(history) == 1, campaign_b.get("transaction_history")
            assert history[0].get("cycle_outcome") == "absorbed", history[0]
            assert history[0].get("verified_by") == "boot_reconciliation", history[0]
            assert history[0].get("commit_sha") == commit_sha, history[0]
            assert _git(["rev-parse", "HEAD"], clone) == commit_sha
            assert not list(state_dir.glob(marker_glob))
            reconciled = oracle.events("evolution_tx_reconciled")
            assert reconciled and reconciled[-1].get("ok") is True, reconciled
            assert reconciled[-1].get("commit_sha") == commit_sha, reconciled[-1]
            outcomes = _cycle_outcome_checkpoints(data_root)
            assert [row.get("cycle_outcome") for row in outcomes].count("absorbed") == 1, outcomes
        finally:
            server_b.stop()

    # ---- Generation C: nothing absorbs twice --------------------------------
    with ScriptedStubModel([{"tool": "list_files", "arguments": {"path": "."}}]) as stub_c:
        server_c = start_server(clone, root, keyless_settings(stub_c, **settings_kwargs))
        try:
            task_c = submit_running(server_c, "List the repository root once more and finish.")
            assert server_c.wait_task(task_c, timeout=300).get("status") == "completed"
            campaign_c = _campaign(data_root)
            assert int(campaign_c.get("absorbed_cycles_done") or 0) == 1, campaign_c
            history = [row for row in (campaign_c.get("transaction_history") or [])
                       if str(row.get("transaction_id") or "") == transaction_id]
            assert len(history) == 1, campaign_c.get("transaction_history")
            assert _git(["rev-parse", "HEAD"], clone) == commit_sha
            assert [row.get("cycle_outcome")
                    for row in _cycle_outcome_checkpoints(data_root)].count("absorbed") == 1
            assert not list(state_dir.glob(marker_glob))
        finally:
            server_c.stop()


# ===========================================================================
# S23 — delegated interactive answer over the FakeClaudexorDaemon.
# ===========================================================================

_IID_RE = re.compile(r'"interaction_id": "([^"]+)"')
_QID_RE = re.compile(r'"question_id": "([^"]+)"')

S23_PARENT_MARKER = "S23_PARENT_FINAL_e2e_w4"


def _answer_step(body: dict) -> dict:
    text = body_text(body)
    run_ids = _RUN_ID_RE.findall(text)
    interaction_ids = _IID_RE.findall(text)
    question_ids = _QID_RE.findall(text)
    if not (run_ids and interaction_ids and question_ids):
        return {"final": "E2E_SCRIPT_ERROR: no pending interaction visible in the transcript"}
    return {"tool": "delegate_answer", "arguments": {
        "run_id": run_ids[-1],
        "interaction_id": interaction_ids[-1],
        "answers": [{"question_id": question_ids[-1],
                     "selected_labels": [FAKE_ANSWER_LABEL]}],
    }}


S23_SCRIPT = [
    {"tool": "delegate_start", "arguments": {
        "subagent_id": "cx-scout",
        "prompt": FAKE_ASK_MARKER + " survey the repository and ask before finishing"}},
    _wait_step,      # returns IMMEDIATELY: status=waiting_on_user + the question
    _answer_step,    # delegate_answer -> delivered
    _wait_step,      # the resumed run settles
]


@pytest.mark.integration
@pytest.mark.serial
def test_s23_interactive_delegated_run_pauses_answers_and_resumes(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s23")
    data_root = pathlib.Path(root) / "data"
    with FakeClaudexorDaemon() as daemon, \
            ScriptedStubModel(S23_SCRIPT,
                              final_answer=f"{S23_PARENT_MARKER}: interactive run absorbed.") as stub:
        daemon.install(data_root / "claudexor")
        settings = keyless_settings(stub, OUROBOROS_SUBAGENTS=_roster(_SCOUT_ROW))
        server = start_server(e2e_clone, root, settings)
        try:
            task_id = submit_running(
                server, "Delegate the survey, answer the scout's question, then finish.")
            result = server.wait_task(task_id, timeout=600)
            assert result.get("status") == "completed", result
            oracle = ArtifactOracle(server.data_root)
            stored = wait_durable_result(oracle, task_id)
            assert S23_PARENT_MARKER in str(stored.get("result") or ""), stored
            assert stub.script_consumed(), (
                "S23 script was not fully consumed — the run never paused on its question")

            # -- durable custody chain, ORDERED: an interactive pause is one run
            # (started < answered < settled), never a second attempt or a cancel.
            started = _custody_rows(oracle, "delegate_run_started")
            assert len(started) == 1, started
            run_id = str(started[0].get("run_id") or "")
            answered = _custody_rows(oracle, "delegate_interaction_answered", run_id)
            assert len(answered) == 1, answered
            assert answered[0].get("status") == "delivered", answered[0]
            assert int(answered[0].get("questions_answered") or 0) == 1, answered[0]
            assert answered[0].get("task_id") == task_id, answered[0]
            interaction_id = str(answered[0].get("interaction_id") or "")
            assert interaction_id, answered[0]
            settled = _custody_rows(oracle, "delegate_run_settled", run_id)
            assert settled and settled[-1].get("state") == "succeeded", settled
            assert settled[-1].get("cost_final") is True, settled[-1]
            all_events = oracle.events()
            def _index(event_type):
                return next(index for index, row in enumerate(all_events)
                            if str(row.get("type") or "") == event_type
                            and str(row.get("run_id") or "") == run_id)
            assert _index("delegate_run_started") < _index("delegate_interaction_answered") \
                < _index("delegate_run_settled"), [
                    (row.get("type"), row.get("run_id")) for row in all_events]
            assert _custody_rows(oracle, "delegate_run_cancel_outcome", run_id) == [], (
                "the nanny cancelled a run that merely asked a question")

            # -- wire truth: ONE physical run, ONE answer POST with the exact
            # camelCase answer shape on the exact interaction path.
            assert len(daemon.run_start_posts()) == 1
            answer_posts = [row for row in daemon.calls("POST", f"/v2/runs/{run_id}/")
                            if row["path"].endswith("/answer")]
            assert len(answer_posts) == 1, answer_posts
            assert answer_posts[0]["path"] == (
                f"/v2/runs/{run_id}/interactions/{interaction_id}/answer"), answer_posts[0]
            assert answer_posts[0]["body"] == {"answers": [{
                "questionId": "q1", "selectedLabels": [FAKE_ANSWER_LABEL], "freeText": None,
            }]}, answer_posts[0]
            assert daemon.runs[run_id]["pending"] == [], daemon.runs[run_id]
            assert daemon.runs[run_id]["state"] == "succeeded", daemon.runs[run_id]

            # -- transcript truth: the model actually SAW the pause, the
            # question, the delivered relay and the terminal result.
            transcript = "\n".join(body_text(call_body) for _kind, call_body in stub.calls)
            assert '"status": "waiting_on_user"' in transcript
            assert FAKE_QUESTION_TEXT in transcript
            assert '"status": "delivered"' in transcript
            assert "Keep watching with delegate_wait." in transcript
            assert f"FAKE_RUN_RESULT {run_id}" in transcript

            # -- the last-delegation receipt settled exactly once, honestly.
            receipt = oracle._json("state/subagent_last_delegation.json")
            assert receipt.get("run_id") == run_id, receipt
            assert receipt.get("requested_model") == "mock-model", receipt
            assert receipt.get("applied_model") == daemon.applied_model, receipt
            assert receipt.get("selected_subagent_id") == "cx-scout", receipt
        finally:
            server.stop()
