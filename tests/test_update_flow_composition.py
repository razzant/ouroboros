"""Cross-lane COMPOSITION tests for the update-flow redesign (phase S.2).

Each test drives seams from two or more lanes (update-core, sizes, cycles,
review-object) through the MERGED machinery. Lane-internal behavior is not
re-tested here; where a composition item is already covered verbatim by a lane
test, that coverage is cited instead of duplicated:

- B5 (advisory half): ``test_advisory_replay_reasons_drive_the_stage_cycle_to_a
  _disclosed_pass`` (tests/test_review_cycles_dispatch.py) drives BOTH
  free-replay reasons — identical verdict and exhausted ceiling — through the
  real stage cycle to a disclosed advisory pass; the blocking half end-to-end
  lives here (``test_blocking_exhaustion_end_to_end_...``).
- B8 (rerere immunity of M0): ``test_materialize_is_immune_to_a_poisoned_
  rerere_cache`` (tests/test_update_merge_assisted.py) poisons the rr-cache
  with a recorded WRONG resolution and proves the pinned M0 baseline (the only
  M0 authority — pinned once at materialization) is not contaminated;
  ``_MERGE_NEUTRAL_FLAGS`` is applied by every mechanical merge in the flow.
- B3 (restore-conflict / marker-guarded refusal legs):
  ``test_conflicting_restore_keeps_stash_and_discloses`` and
  ``test_restore_with_marker_refuses_a_dirty_tree`` pin the conflict and
  dirty-tree refusal semantics; the full-flow legs (real stash prolog -> real
  clean apply -> boot finalize / rollback) live here.
- B2 (per-consumer packet content): capture/triad/scope/advisory packet
  contents are pinned per consumer in tests/test_managed_review_subject.py
  (``test_managed_capture_returns_resolution_delta_only``,
  ``test_triad_session_task_inlines_managed_delta``,
  ``test_scope_session_task_inlines_managed_delta``,
  ``test_fit_error_with_session_quorum_drops_api_rows``); the one-fixture
  fan-out to ALL consumers plus the binding assertion lives here.
"""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import supervisor.git_ops as git_ops
import supervisor.update_candidate as update_candidate
import supervisor.update_merge as update_merge
from tests import test_managed_review_subject as tmrs
from tests import test_repo_health_smoke as trh
from tests import test_update_dirty_stash as tds
from tests import test_update_merge_assisted as tua

KEY = "OUROBOROS_REVIEW_MAX_CYCLES"


def _events(drive_root: pathlib.Path, event_type: str) -> list:
    path = pathlib.Path(drive_root) / "logs" / "events.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("type") == event_type]


# ---------------------------------------------------------------------------
# B1 — update x ratchet: a merge that grows a module past the hard boundary
# warns locally (readiness + health), never blocks the commit, and the CI
# pairwise validator (base = pre-update HEAD, tip = merge result) catches it.
# ---------------------------------------------------------------------------


def test_update_merge_size_growth_warns_locally_and_pairwise_ci_catches(tmp_path, monkeypatch):
    """Neither parent carries the debt — the MERGE creates it (official grows
    the top of a module, the fork grows its bottom; the textual merge crosses
    1600). Locally that is warnings only; the official CI pairwise transition
    (Q7-C/Q18-A/Q19-A) is the enforcing surface."""
    from ouroboros.review import (
        MAX_MODULE_LINES,
        validate_size_ratchet,
        validate_size_ratchet_transition_against_base,
    )
    from ouroboros.tools.health import _codebase_health
    from ouroboros.tools.review_helpers import check_worktree_readiness

    repo = tmp_path / "repo"
    body = ["mid-%d" % i for i in range(800)]
    pre0 = trh._bootstrap_repo(repo, files={"mid.py": "\n".join(body) + "\n"})
    trh._write_manifest(repo, trh._manifest(sha=pre0))
    trh._git(repo, "add", ".")
    trh._git(repo, "commit", "-qm", "bootstrap ratchet")
    head = trh._git(repo, "symbolic-ref", "--short", "HEAD")

    # Official line: +500 lines at the TOP of mid.py (1300 total — no debt).
    trh._git(repo, "checkout", "-qb", "official")
    (repo / "mid.py").write_text(
        "\n".join(["official-%d" % i for i in range(500)] + body) + "\n", encoding="utf-8"
    )
    trh._git(repo, "add", ".")
    trh._git(repo, "commit", "-qm", "official grows the top")
    target = trh._git(repo, "rev-parse", "HEAD")

    # Local fork: +400 lines at the BOTTOM (1200 total — no debt either).
    trh._git(repo, "checkout", "-q", head)
    (repo / "mid.py").write_text(
        "\n".join(body + ["local-%d" % i for i in range(400)]) + "\n", encoding="utf-8"
    )
    trh._git(repo, "add", ".")
    trh._git(repo, "commit", "-qm", "fork grows the bottom")
    pre_update = trh._git(repo, "rev-parse", "HEAD")

    # The managed merge: clean textually, 1700 lines — debt neither line saw.
    trh._git(repo, "merge", "-q", "--no-ff", "--no-commit", target)
    assert MAX_MODULE_LINES < 1701

    # Mid-resolution: the local validator reports the growth as FINDINGS…
    findings = validate_size_ratchet(repo)
    assert findings and any("mid.py" in f for f in findings)
    # …readiness surfaces them as loud CI-attributed WARNINGS (never a block)…
    warnings = check_worktree_readiness(repo)
    ci_warnings = [w for w in warnings if w.startswith("official CI will enforce: ")]
    assert ci_warnings and any("mid.py" in w for w in ci_warnings)
    # …and codebase_health renders the same section.
    health = _codebase_health(SimpleNamespace(repo_dir=repo))
    assert "Size-Ratchet Findings (official CI will enforce)" in health
    assert "mid.py" in health

    # The resolver regenerates the manifest and the merge COMMITS — nothing on
    # the local line gates on size (commit_reviewed has no size coupling; the
    # only local consumers are the two warning surfaces asserted above).
    trh._write_manifest(repo, trh._manifest(giant_paths=frozenset({"mid.py"}), sha=pre0))
    trh._git(repo, "add", ".")
    trh._git(repo, "commit", "-qm", "managed update merge")
    merge_sha = trh._git(repo, "rev-parse", "HEAD")
    parents = trh._git(repo, "log", "-1", "--format=%P", merge_sha).split()
    assert parents == [pre_update, target]  # [local, target] order preserved

    # Locally the landed debt is grandfathered: green.
    assert validate_size_ratchet(repo) == []
    # The CI pairwise transition — exactly what the size_ratchet lane runs with
    # OURO_SIZE_RATCHET_BASE_REF — still catches it against the pre-update base.
    assert validate_size_ratchet_transition_against_base(repo, pre_update) == [
        "new module debt above 1600 lines: mid.py"
    ]


def test_pairwise_validator_accepts_update_merge_topology_as_base(tmp_path, monkeypatch):
    """B9 (manifest half): an [L, T] two-parent update merge is a VALID pairwise
    base — the retired first-parent history replay can no longer condemn the
    fork topology. (Manifest resolution THROUGH a merge parent is pinned by
    ``test_previous_manifest_resolves_through_any_merge_parent``.)"""
    from ouroboros.review import validate_size_ratchet_transition_against_base

    repo = tmp_path / "repo"
    pre0 = trh._bootstrap_repo(repo)
    trh._write_manifest(repo, trh._manifest(sha=pre0))
    trh._git(repo, "add", ".")
    trh._git(repo, "commit", "-qm", "bootstrap ratchet")
    head = trh._git(repo, "symbolic-ref", "--short", "HEAD")
    trh._git(repo, "checkout", "-qb", "official")
    (repo / "official.txt").write_text("released\n", encoding="utf-8")
    trh._git(repo, "add", ".")
    trh._git(repo, "commit", "-qm", "official")
    target = trh._git(repo, "rev-parse", "HEAD")
    trh._git(repo, "checkout", "-q", head)
    (repo / "local.txt").write_text("fork\n", encoding="utf-8")
    trh._git(repo, "add", ".")
    trh._git(repo, "commit", "-qm", "fork")
    trh._git(repo, "merge", "-q", "--no-ff", "-m", "managed update merge", target)
    merge_sha = trh._git(repo, "rev-parse", "HEAD")
    assert len(trh._git(repo, "log", "-1", "--format=%P", merge_sha).split()) == 2

    # A later ordinary commit, validated pairwise WITH THE MERGE AS THE BASE.
    (repo / "after.txt").write_text("later\n", encoding="utf-8")
    trh._git(repo, "add", ".")
    trh._git(repo, "commit", "-qm", "after the update")
    assert validate_size_ratchet_transition_against_base(repo, merge_sha) == []


def test_legacy_pre_redesign_tx_boots_resumes_and_verifies(tmp_path, monkeypatch):
    """B9 (tx half): a pre-redesign assisted tx — no ``local_work_carrier``,
    ``stash_sha``, ``attempt_id``, ``m0_tree`` — boots through the real
    dispatcher without error: the in-progress resolution is resumed, the
    unrecoverable mechanical baseline is REPRESENTED (never fabricated), and
    the pre-commit verifier exempts the legacy shape from the m0 mandate."""
    repo, head, plan, tx = tua._materialized_conflict_tx(tmp_path, monkeypatch)
    tua._stub_worker_gates(monkeypatch)
    enqueued = []
    monkeypatch.setattr(
        update_merge, "enqueue_assisted_resolution_task",
        lambda tx_arg: enqueued.append(dict(tx_arg)) or "resolver",
    )
    assert "local_work_carrier" not in tx and "m0_tree" not in tx  # legacy shape

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result == {"finalized": False, "resumed": True, "resolution_attempts": 1}
    assert enqueued and enqueued[0]["task_id"] == "resolver"
    recovered = update_merge.read_update_tx()
    # The baseline gap is represented, never filled in after resolver edits.
    assert recovered.get("m0_tree") == ""
    assert recovered.get("m0_missing_reason") == "resumed_with_progress_before_m0_pin"
    # No stash restore was owed or attempted (nothing was ever stashed).
    assert not tua._git(repo, "stash", "list").stdout.strip()
    assert (repo / "a.txt").read_text() == "the resolver's precious resolution\n"
    # The legacy tx passes the pre-commit verifier (m0 mandate is new-format only).
    tua._git(repo, "add", "-A")
    ok, error = update_merge.managed_assisted_precommit_verify(recovered)
    assert ok, error


# ---------------------------------------------------------------------------
# B2 — update x review subject: ONE managed fixture feeds every consumer the
# same M0->S delta artifact, and the binding gate passes the honest candidate.
# ---------------------------------------------------------------------------


def test_one_managed_fixture_feeds_all_review_consumers_and_binds(tmp_path, monkeypatch):
    import ouroboros.tools.claude_advisory_review as adv
    import ouroboros.tools.git as git_mod
    from ouroboros.tools.review_helpers import REPO_ROOT
    from ouroboros.tools.review_subject import (
        build_triad_session_task,
        capture_review_diff,
        managed_review_subject,
    )
    from ouroboros.tools.scope_review_session import (
        ScopeBriefInputs,
        ScopeIntentContext,
        build_scope_session_task,
    )

    repo, ctx, tx = tmrs._managed_resolution_repo(tmp_path, monkeypatch)
    ctx.drive_root = str(tmp_path / "data")

    subject = managed_review_subject(ctx, repo)
    assert subject is not None and subject.m0_tree == tx["m0_tree"]
    delta = subject.render_prompt_diff()
    assert "resolved by the agent" in delta and "released official change" not in delta

    # Triad + scope api packets read the SHARED capture — byte-identical artifact.
    assert capture_review_diff(ctx, repo) == delta

    # Advisory (worktree surface, HAPPY path — not the oversize skip): same
    # delta content, scoped to delta ∪ conflict anchors, managed and not early.
    diff_text, context_paths, early, managed = adv._advisory_review_diff(
        pathlib.Path(repo), ctx, None
    )
    assert managed is True and early is None
    assert "resolved by the agent" in diff_text
    assert "released official change" not in diff_text
    assert {"conflict.txt", "resolver_note.txt"} <= set(context_paths or [])

    # Both SESSION deliveries inline the same artifact.
    triad_task = build_triad_session_task(subject=subject, **tmrs._SESSION_SECTIONS)
    scope_task, _manifest = build_scope_session_task(repo, ScopeBriefInputs(
        commit_message="land the update",
        intent=ScopeIntentContext(goal="g", scope="s"),
        governance_repo_dir=pathlib.Path(REPO_ROOT), managed_subject=subject,
    ))
    for task_text in (triad_task, scope_task):
        assert "AUTHORITATIVE review subject" in task_text
        assert "resolved by the agent" in task_text

    # Binding: only GATE subjects registered a tree (advisory never does), it
    # is exactly the tree the commit would write, and the honest candidate
    # passes the mismatch gate while a forged tree is a typed block.
    assert getattr(ctx, "_last_review_subject_trees") == {subject.staged_tree}
    honest = {"binding": {"tree_sha": subject.staged_tree}, "fingerprint": "fp"}
    assert git_mod._subject_binding_mismatch_outcome(ctx, "m", 0.0, honest, {}) is None
    forged = {"binding": {"tree_sha": "f" * 40}, "fingerprint": "fp"}
    outcome = git_mod._subject_binding_mismatch_outcome(ctx, "m", 0.0, forged, {})
    assert outcome and outcome["block_reason"] == "review_subject_binding_mismatch"


# ---------------------------------------------------------------------------
# B3 — update x stash: dirty local work survives the FULL managed flow (real
# stash prolog -> real clean apply -> boot finalize / rollback).
# ---------------------------------------------------------------------------


def _stub_apply_collaborators(monkeypatch, control):
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update", lambda: None)
    monkeypatch.setattr(
        control, "_restart_response",
        lambda request, *, strategy, plan: {"ok": True, "strategy": strategy},
    )
    monkeypatch.setattr(git_ops, "_create_rescue_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(git_ops, "_collect_repo_sync_state", lambda: {})
    monkeypatch.setattr(update_merge, "update_restart_smoke", lambda: {"ok": True})


def _prolog_and_clean_apply(tmp_path, monkeypatch):
    """Dirty tree -> REAL ``_stash_local_work_fenced`` -> REAL replan from the
    clean tree -> REAL ``_apply_clean_merge_fenced``. Returns facts for the
    finalize/rollback legs."""
    from ouroboros.gateway import control

    repo, head = tds._diverged_clean_repo(tmp_path, monkeypatch)
    base = tds._git(repo, "rev-parse", "HEAD").stdout.strip()
    target = tds._git(repo, "rev-parse", "remote-sim").stdout.strip()
    (repo / "a.txt").write_text("owner dirty work\n")
    (repo / "untracked.txt").write_text("scratch\n")
    _stub_apply_collaborators(monkeypatch, control)

    tx, error = control._stash_local_work_fenced(
        branch=head, base_sha=base, target_sha=target, plan={}
    )
    assert error is None and tx is not None
    assert tx["local_work_carrier"] == "stash" and tx["stash_sha"]
    # The tree is clean and the durable attempt_id finds exactly this entry.
    assert not tds._git(repo, "status", "--porcelain").stdout.strip()
    assert update_candidate.find_update_stash_sha(tx["attempt_id"]) == tx["stash_sha"]

    plan = update_merge.plan_managed_update_merge(build=True)  # replan, clean tree
    assert plan["kind"] == "clean", plan
    response = control._apply_clean_merge_fenced(object(), plan, tx)
    assert response == {"ok": True, "strategy": "auto_merge"}
    persisted = update_merge.read_update_tx()
    assert persisted["phase"] == "pending_boot_smoke"
    assert persisted["pre_restart_smoke"] == "passed"
    assert persisted["stash_sha"] == tx["stash_sha"]
    assert tds._git(repo, "rev-parse", "HEAD").stdout.strip() == plan["merge_commit"]
    return repo, head, base, tx


def test_dirty_work_survives_the_full_managed_clean_update(tmp_path, monkeypatch):
    repo, _head, _base, tx = _prolog_and_clean_apply(tmp_path, monkeypatch)

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result.get("finalized") is True, result
    assert (repo / "a.txt").read_text() == "owner dirty work\n"
    assert (repo / "untracked.txt").read_text() == "scratch\n"
    assert (repo / "remote.txt").read_text() == "official\n"  # the update landed
    assert tx["stash_sha"] in tds._git(repo, "stash", "list", "--format=%H").stdout
    assert update_merge.read_update_tx() == {}


def test_dirty_work_survives_rollback_after_the_stash_prolog(tmp_path, monkeypatch):
    repo, _head, base, tx = _prolog_and_clean_apply(tmp_path, monkeypatch)
    tua._stub_worker_gates(monkeypatch)

    ok, message = update_merge.rollback_managed_update("composition-test")

    assert ok, message
    assert tds._git(repo, "rev-parse", "HEAD").stdout.strip() == base
    assert (repo / "a.txt").read_text() == "owner dirty work\n"
    assert (repo / "untracked.txt").read_text() == "scratch\n"
    assert "restored" in message
    assert update_merge.read_update_tx() == {}


def test_conflicting_restore_after_full_flow_is_disclosed_not_dropped(tmp_path, monkeypatch):
    """The stash carries an UNTRACKED file the update itself lands: the boot
    restore conflicts, the update stays finalized, the stash entry is KEPT and
    the note names the manual recovery command — no silent drop."""
    from ouroboros.gateway import control

    repo, head = tds._diverged_clean_repo(tmp_path, monkeypatch)
    base = tds._git(repo, "rev-parse", "HEAD").stdout.strip()
    target = tds._git(repo, "rev-parse", "remote-sim").stdout.strip()
    (repo / "remote.txt").write_text("owner's colliding draft\n")  # untracked
    _stub_apply_collaborators(monkeypatch, control)
    tx, error = control._stash_local_work_fenced(
        branch=head, base_sha=base, target_sha=target, plan={}
    )
    assert error is None and tx["stash_sha"]
    plan = update_merge.plan_managed_update_merge(build=True)
    assert plan["kind"] == "clean", plan
    assert control._apply_clean_merge_fenced(object(), plan, tx)["ok"] is True

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result.get("finalized") is True, result
    note = result.get("stash_note", "")
    assert "git stash apply" in note and tx["stash_sha"][:12] in note
    # The landed update wins the tree; the owner's draft survives in the stash.
    assert (repo / "remote.txt").read_text() == "official\n"
    assert tx["stash_sha"] in tds._git(repo, "stash", "list", "--format=%H").stdout


# ---------------------------------------------------------------------------
# B4 + B11 — update x runner proof: reuse only while HEAD and candidate match.
# A commit requires new proof even when the staged tree stays byte-identical.
# ---------------------------------------------------------------------------


def test_managed_flow_runs_the_hermetic_suite_once_per_head(tmp_path, monkeypatch):
    import ouroboros.tools.git as git_mod
    from tests.test_advisory_preflight import _stub_preflight_lanes

    repo, ctx, tx = tmrs._managed_resolution_repo(tmp_path, monkeypatch)
    drive = tmp_path / "data"
    ctx.drive_root = str(drive)
    ctx.drive_logs = lambda: drive / "logs"
    progress: list = []
    ctx.emit_progress_fn = progress.append

    # Exercise actual assembly, receipt creation and reuse. Physical pytest
    # work is stubbed here; test_preflight_test_proof covers the real processes.
    lanes = _stub_preflight_lanes(repo, monkeypatch)

    # _repo_commit_push ordering: the managed tx snapshot is taken BEFORE the
    # stage cycle — i.e. BEFORE the compensating preflight records its proof.
    managed_tx, block = update_merge.managed_assisted_tx_for(
        ctx.task_id, ctx.task_metadata
    )
    assert managed_tx and not block

    # B11 first half: no evidence covers the candidate tree yet -> the managed
    # mandate forces the pre-commit run even under skip_tests + doc-only.
    assert git_mod._managed_candidate_needs_proof(ctx) is True
    outcome = git_mod._advisory_and_tests_gate(
        ctx, "land the managed resolution", 0.0,
        classification_paths=["docs/x.md"], advisory_paths=None,
        skip_advisory_pre_review=True, skip_tests=True,
    )
    assert outcome is None
    assert len(lanes) == 2 and len({worktree for worktree, _ in lanes}) == 1
    assert any("mandatory hermetic suite" in note for note in progress)
    evidence = update_merge.read_update_tx().get("tests_evidence") or {}
    assert evidence.get("tree")
    proof = ctx._preflight_test_proof
    assert proof.tree == evidence["tree"]
    assert git_mod._managed_candidate_needs_proof(ctx) is False

    # Commit the exact candidate; the new HEAD requires a fresh test pair.
    tmrs._git(repo, "commit", "-q", "-m", "managed resolution")
    committed_tree = tmrs._git(repo, "rev-parse", "HEAD^{tree}").stdout.strip()
    assert committed_tree == evidence["tree"]
    # The REAL commit path's phase transition, driven with the same STALE
    # snapshot _repo_commit_push holds: the merge-write must preserve the
    # in-attempt forensic evidence (a wholesale write dropped tests_evidence).
    committing = update_merge.update_tx_phase(
        managed_tx, {"phase": "committing_assisted"}
    )
    assert committing["phase"] == "committing_assisted"
    assert (update_merge.read_update_tx().get("tests_evidence") or {}).get("tree") == evidence["tree"]
    gate = git_mod._managed_post_commit_tests_gate(
        ctx, "managed resolution", 0.0, True, [""], update_merge.read_update_tx(),
    )
    assert gate is None
    assert len(lanes) == 4 and len({worktree for worktree, _ in lanes}) == 2
    # Synthesis F2: the authority the gate consulted is the PROCESS-HELD ctx
    # record pinned by the recording site — not the durable (forensic) tx copy.
    assert committed_tree == ctx._preflight_test_proof.tree == proof.tree
    assert ctx._preflight_test_proof.head != proof.head


def test_managed_phase_writes_merge_onto_fresh_tx_not_stale_snapshot(
    tmp_path, monkeypatch,
):
    """Synthesis wave W2 pin (both converted sites): every managed phase
    transition is a read-modify-write on a FRESH tx read. The commit flow's
    snapshot predates the compensating preflight's ``tests_evidence`` write —
    a wholesale snapshot write (the old ``write_update_tx(dict(snapshot))``)
    silently dropped the forensic test receipt. The process-held proof, not
    that durable telemetry, remains the reuse authority."""
    from ouroboros.commit_admission import run_tests_preflight_with_proof
    from ouroboros.tools.review_helpers import _run_review_preflight_tests
    from tests.test_advisory_preflight import _stub_preflight_lanes

    repo, head, plan, tx = tua._materialized_conflict_tx(tmp_path, monkeypatch)
    meta = tua._authority_metadata(tx)
    lanes = _stub_preflight_lanes(repo, monkeypatch)
    ctx = SimpleNamespace(task_id="resolver", task_metadata=meta, repo_dir=repo)

    # Snapshot BEFORE the stage cycle (git.py _repo_commit_push ordering).
    managed_tx, block = update_merge.managed_assisted_tx_for("resolver", meta)
    assert managed_tx and not block

    # Mid-attempt, a runner-issued receipt is projected to durable telemetry.
    assert run_tests_preflight_with_proof(ctx, runner=_run_review_preflight_tests) is None
    tree = ctx._preflight_test_proof.tree
    assert len(lanes) == 2
    assert tree and update_merge.managed_tests_evidence_covers(tree)

    # Site 1: the committing_assisted transition with the STALE snapshot.
    committing = update_merge.update_tx_phase(
        managed_tx, {"phase": "committing_assisted"}
    )
    assert committing["phase"] == "committing_assisted"
    assert update_merge.managed_tests_evidence_covers(tree), (
        "the committing_assisted phase write clobbered the in-attempt "
        "tests_evidence"
    )

    # Site 2: managed_assisted_postcommit (also fed the stale snapshot) must
    # preserve the proof across BOTH of its phase writes.
    monkeypatch.setattr(update_merge, "update_restart_smoke", lambda: {"ok": True})
    ok, msg = update_merge.managed_assisted_postcommit(dict(managed_tx), "f" * 40)
    assert ok, msg
    current = update_merge.read_update_tx()
    assert current["phase"] == "pending_boot_smoke"
    assert current["pre_restart_smoke"] == "passed"
    assert current["merge_commit"] == "f" * 40
    assert update_merge.managed_tests_evidence_covers(tree), (
        "the postcommit phase writes clobbered the in-attempt tests_evidence"
    )

    # Helper contract: an absent durable marker falls back to the caller's
    # snapshot (the only substrate left) instead of resurrecting nothing.
    assert update_merge.clear_update_tx()
    merged = update_merge.update_tx_phase(dict(managed_tx), {"phase": "rolling_back"})
    assert merged["phase"] == "rolling_back"
    assert merged["task_id"] == managed_tx["task_id"]


# ---------------------------------------------------------------------------
# B5 (blocking half) + exhaustion chain — the REAL authorized resolver hits the
# paid-cycle ceiling: free typed refusal (money never bought again), the staged
# merge survives the refusal, and the orphan-terminal rollback preserves the
# attempt on the deterministic retry branch.
# ---------------------------------------------------------------------------


def test_blocking_exhaustion_end_to_end_rolls_back_and_preserves_retry_pointer(
    tmp_path, monkeypatch,
):
    import ouroboros.tools.git as git_mod
    from ouroboros.review_state import CommitAttemptRecord, make_repo_key, update_state, _utc_now

    repo, head, plan, tx = tua._materialized_conflict_tx(tmp_path, monkeypatch)
    drive = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv(KEY, "1")
    monkeypatch.setattr(git_mod, "commit_review_contract_fingerprint", lambda: "cf-1")
    ctx = SimpleNamespace(
        repo_dir=str(repo), drive_root=drive, task_id="resolver",
        task_metadata=tua._authority_metadata(tx), event_queue=None,
        drive_logs=lambda: drive / "logs",
        _current_review_tool_name="commit_reviewed",
    )
    repo_key = make_repo_key(pathlib.Path(str(repo)))
    update_state(drive, lambda s: s.attempts.append(CommitAttemptRecord(
        ts=_utc_now(), commit_message="m", status="succeeded", repo_key=repo_key,
        tool_name="commit_reviewed", task_id="resolver", attempt=1, phase="commit",
        paid=True, root_task_id="resolver", pre_review_fingerprint="fp-old",
    )))

    outcome = git_mod._free_cycle_gate(
        ctx, "resolve", 0.0, pre_fingerprint={"fingerprint": "fp-new"}, review_rebuttal="",
    )

    assert outcome is not None and outcome["block_reason"] == "review_cycles_exhausted"
    assert "REVIEW_CYCLES_EXHAUSTED" in outcome["message"]
    exhausted = _events(drive, "review_cycles_exhausted")
    assert exhausted and exhausted[-1]["surface"] == "commit_gate"
    assert exhausted[-1]["enforcement"] == "blocking"
    # The refusal path repaired the REAL merge state for the resolver: the
    # staged managed merge survives the typed refusal.
    assert update_merge._merge_head_sha() == plan["target_sha"]
    assert (repo / "a.txt").read_text() == "the resolver's precious resolution\n"

    # The resolver terminates without a commit -> the orphan watchdog rolls
    # back and PRESERVES the attempt on the deterministic branch.
    tua._stub_worker_gates(monkeypatch)
    result = update_merge.abort_orphaned_assisted_tx("resolver", tua._authority_metadata(tx))
    assert result.get("rolled_back") is True, result
    assert tua._git(repo, "rev-parse", "HEAD").stdout.strip() == plan["base_sha"]
    name = f"failed-update-{plan['target_sha'][:12]}"
    preserved = tua._git(repo, "show", f"{name}:a.txt").stdout
    assert preserved == "the resolver's precious resolution\n"
    # Retry pointer: a fresh apply at the same target is pointed at the branch
    # (assisted_objective wording is pinned by
    # test_rollback_preserves_uncommitted_resolution_on_deterministic_branch).
    assert update_candidate.existing_failed_update_ref(
        plan["target_sha"], not_at=plan["base_sha"]
    ) == name


# ---------------------------------------------------------------------------
# B6 / A1 — binding-mismatch x cycles: the revalidation block is an INFRA fact.
# ---------------------------------------------------------------------------


def test_binding_mismatch_is_infra_never_streak_or_anchor(tmp_path, monkeypatch):
    """A ``review_subject_binding_mismatch`` block (phase=revalidation) — both
    the newly-recorded shape (block_class="infra") and a legacy row without a
    recorded class — never refuses a resubmission for free, never anchors a
    quote, and its dispatched wave still counts money toward the ceiling."""
    import ouroboros.tools.git as git_mod
    from ouroboros.review_state import CommitAttemptRecord, make_repo_key, update_state, _utc_now
    from ouroboros.tools.commit_gate import attempt_block_class, count_paid_review_cycles

    monkeypatch.setenv(KEY, "10")  # keep the money ceiling out of the identity question
    monkeypatch.setattr(git_mod, "commit_review_contract_fingerprint", lambda: "cf-1")
    monkeypatch.setattr(git_mod, "run_cmd", lambda *a, **k: "")
    monkeypatch.setattr(git_mod, "_authorized_managed_update_resolver", lambda ctx: False)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    ctx = SimpleNamespace(
        repo_dir=tmp_path, drive_root=pathlib.Path(tmp_path), task_id="root-1",
        task_metadata={}, event_queue=None,
        drive_logs=lambda: pathlib.Path(tmp_path) / "logs",
        _current_review_tool_name="commit_reviewed",
    )
    repo_key = make_repo_key(pathlib.Path(tmp_path))

    def _mismatch_row(attempt, block_class):
        return CommitAttemptRecord(
            ts=_utc_now(), commit_message="m", status="blocked",
            block_reason="review_subject_binding_mismatch", block_class=block_class,
            repo_key=repo_key, tool_name="commit_reviewed", task_id="root-1",
            attempt=attempt, phase="revalidation", paid=True, root_task_id="root-1",
            pre_review_fingerprint="fp-1", review_contract_fingerprint="cf-1",
        )

    legacy = _mismatch_row(1, "")       # pre-fix row: class inferred at read time
    recorded = _mismatch_row(2, "infra")  # post-fix row: class pinned at record time
    update_state(pathlib.Path(tmp_path), lambda s: s.attempts.extend([legacy, recorded]))

    assert attempt_block_class(legacy) == "infra"
    assert attempt_block_class(recorded) == "infra"
    # An identical resubmit is NOT refused for free off the mismatch rows…
    for enforcement in ("blocking", "advisory"):
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
        assert git_mod._free_cycle_gate(
            ctx, "msg", 0.0, pre_fingerprint={"fingerprint": "fp-1"}, review_rebuttal="",
        ) is None
    # …while the dispatched waves both count toward the money ceiling.
    assert count_paid_review_cycles(ctx, root_task_id="root-1") == 2


def test_dispatched_infra_terminal_counts_money_but_never_identity(tmp_path, monkeypatch):
    """B7: one dispatched infra terminal (quorum failure after a paid dispatch)
    counts a paid cycle — and ONLY the ceiling, never the identical-diff
    refusal, can stop the next attempt."""
    import ouroboros.tools.git as git_mod
    from ouroboros.review_state import CommitAttemptRecord, make_repo_key, update_state, _utc_now
    from ouroboros.tools.commit_gate import count_paid_review_cycles

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setattr(git_mod, "commit_review_contract_fingerprint", lambda: "cf-1")
    monkeypatch.setattr(git_mod, "run_cmd", lambda *a, **k: "")
    monkeypatch.setattr(git_mod, "_authorized_managed_update_resolver", lambda ctx: False)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    ctx = SimpleNamespace(
        repo_dir=tmp_path, drive_root=pathlib.Path(tmp_path), task_id="root-1",
        task_metadata={}, event_queue=None,
        drive_logs=lambda: pathlib.Path(tmp_path) / "logs",
        _current_review_tool_name="commit_reviewed",
    )
    repo_key = make_repo_key(pathlib.Path(tmp_path))
    update_state(pathlib.Path(tmp_path), lambda s: s.attempts.append(CommitAttemptRecord(
        ts=_utc_now(), commit_message="m", status="blocked",
        block_reason="review_quorum", block_class="infra",
        repo_key=repo_key, tool_name="commit_reviewed", task_id="root-1",
        attempt=1, phase="blocking_review", paid=True, root_task_id="root-1",
        pre_review_fingerprint="fp-1", review_contract_fingerprint="cf-1",
    )))

    assert count_paid_review_cycles(ctx, root_task_id="root-1") == 1
    # Below the ceiling: the identical resubmit passes the gate (infra != identity).
    monkeypatch.setenv(KEY, "10")
    assert git_mod._free_cycle_gate(
        ctx, "msg", 0.0, pre_fingerprint={"fingerprint": "fp-1"}, review_rebuttal="",
    ) is None
    # With the ceiling at the money already spent, ONLY exhaustion refuses.
    monkeypatch.setenv(KEY, "1")
    outcome = git_mod._free_cycle_gate(
        ctx, "msg", 0.0, pre_fingerprint={"fingerprint": "fp-1"}, review_rebuttal="",
    )
    assert outcome is not None and outcome["block_reason"] == "review_cycles_exhausted"


# ---------------------------------------------------------------------------
# A2 — ledger re-strip: a terminal merge onto a compacted row must not
# resurrect the heavy reviewer payloads.
# ---------------------------------------------------------------------------


def test_terminal_merge_onto_stripped_row_stays_compacted(tmp_path):
    from ouroboros.review_state import (
        CommitAttemptRecord,
        load_state,
        make_repo_key,
        update_state,
        _utc_now,
    )

    repo_key = make_repo_key(pathlib.Path(tmp_path))
    heavy = "HEAVYRESURRECT" * 500

    # A stale in-flight row that slid past the newest-50 window and was
    # compacted (raw_stripped=True) while its terminal was still pending.
    def _seed(state):
        state.attempts.append(CommitAttemptRecord(
            ts=_utc_now(), commit_message="m", status="reviewing",
            repo_key=repo_key, tool_name="commit_reviewed", task_id="root-1",
            attempt=7, phase="review", paid=True, root_task_id="root-1",
            pre_review_fingerprint="fp-7", raw_stripped=True,
        ))

    update_state(pathlib.Path(tmp_path), _seed)

    # The late terminal arrives WITH full forensic payloads.
    def _terminal(state):
        state.record_attempt(CommitAttemptRecord(
            ts=_utc_now(), commit_message="m" * 500, status="blocked",
            block_reason="critical_findings", block_class="verdict",
            block_details="details " + heavy,
            repo_key=repo_key, tool_name="commit_reviewed", task_id="root-1",
            attempt=7, phase="blocking_review",
            pre_review_fingerprint="fp-7", review_contract_fingerprint="cf-1",
            critical_findings=[{"item": "bug_late", "reason": "r", "severity": "critical"}],
            triad_raw_results=[{"model_id": "m1", "raw_text": heavy}],
            scope_raw_result={"status": "responded", "raw_text": heavy},
        ))

    update_state(pathlib.Path(tmp_path), _terminal)

    [row] = load_state(pathlib.Path(tmp_path)).filter_attempts(repo_key=repo_key)
    # Still compacted: the merge re-stripped the resurrected payloads…
    assert row.raw_stripped is True
    assert row.triad_raw_results == [] and row.scope_raw_result == {}
    assert heavy not in row.block_details and len(row.block_details) <= 700
    assert len(row.commit_message) <= 400
    # …while every accounting fact of the merged terminal is intact.
    assert row.status == "blocked" and row.block_class == "verdict"
    assert row.paid is True and row.root_task_id == "root-1"
    assert row.pre_review_fingerprint == "fp-7"
    assert row.critical_findings and row.critical_findings[0]["item"] == "bug_late"
    # Serialized state carries no resurrected payload either.
    raw = (pathlib.Path(tmp_path) / "state" / "advisory_review.json").read_text(encoding="utf-8")
    assert heavy not in raw


# ---------------------------------------------------------------------------
# A3 — a panel returning ZERO model results keeps the withheld (Q28-A
# not_dispatched) seat records in the durable raw results.
# ---------------------------------------------------------------------------


def test_zero_result_panel_preserves_withheld_seat_records(monkeypatch):
    import ouroboros.tools.review as review_mod

    withheld = [
        {"model_id": "api-1", "status": "not_dispatched", "detail": "Q28-A oversize drop"},
        {"model_id": "api-2", "status": "not_dispatched", "detail": "Q28-A oversize drop"},
    ]
    ctx = SimpleNamespace(
        _triad_withheld_seat_records=list(withheld),
        _last_triad_raw_results=[],
        _review_degraded_reasons=[],
        _review_advisory=[],
    )
    monkeypatch.setattr(
        review_mod, "_handle_multi_model_review", lambda *a, **k: json.dumps({"results": []})
    )
    prepared = {
        "prompt": "p", "stable_prefix_len": 0, "models": ["session-1"],
        "routes": ["agent_session"], "row_plan": {}, "session_task": "t",
        "target_repo": ".", "blocking_review": True,
    }

    blocked = review_mod._dispatch_unified_review(ctx, "msg", prepared)

    assert blocked and "no results from any model" in blocked
    assert ctx._last_review_block_reason == "infra_failure"
    # The withheld seat identities survived into the durable channel the gate
    # records (previously only the degraded-reason string survived).
    assert ctx._last_triad_raw_results == withheld


# ---------------------------------------------------------------------------
# A4 — authority evidence unreadable at the predicate read fails LOUD, never
# a silent non-managed full capture.
# ---------------------------------------------------------------------------


def test_authority_read_failure_fails_loud_not_silent_capture(tmp_path, monkeypatch):
    import pytest

    from ouroboros.tools.review_binary_context import StagedDiffUnavailable
    from ouroboros.tools.review_subject import managed_review_subject

    repo, ctx, _tx = tmrs._managed_resolution_repo(tmp_path, monkeypatch)

    # Healthy authority read: the subject builds (and the marker stays clear).
    assert managed_review_subject(ctx, repo) is not None
    assert getattr(ctx, "_managed_authority_read_error", "") == ""

    # REAL corruption: the durable tx marker file exists but cannot be parsed.
    # authorized_assisted_task maps that to {} without raising — the predicate
    # must still propagate the corrupt distinction (typed ctx marker) so the
    # subject builder fails LOUDLY instead of silently reviewing the managed
    # candidate as an ordinary full staged diff (synthesis wave W3).
    marker_path = update_merge._update_tx_marker_path()
    marker_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(StagedDiffUnavailable, match="authority evidence is unreadable"):
        managed_review_subject(ctx, repo)
    assert "update_tx_corrupt" in getattr(ctx, "_managed_authority_read_error", "")
    # The corrupt marker is loud for ANY task under the active-but-unreadable
    # tx — an ordinary diff must not be reviewed as provably non-managed.
    corrupt_stranger = SimpleNamespace(
        task_id="someone-else", task_metadata=None, repo_dir=str(repo)
    )
    with pytest.raises(StagedDiffUnavailable, match="authority evidence is unreadable"):
        managed_review_subject(corrupt_stranger, repo)

    # The exception channel (tx STORAGE read raising) stays loud too.
    def _boom(*_a, **_k):
        raise OSError("tx storage unreadable")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(update_merge, "authorized_assisted_task_strict", _boom)
        with pytest.raises(StagedDiffUnavailable, match="authority evidence is unreadable"):
            managed_review_subject(ctx, repo)

    # An honest "not the resolver" (valid marker, other task) still degrades
    # to the ordinary path (None) with a clear marker.
    update_merge.write_update_tx(_tx)
    stranger = SimpleNamespace(task_id="someone-else", task_metadata=None, repo_dir=str(repo))
    assert managed_review_subject(stranger, repo) is None
    assert getattr(stranger, "_managed_authority_read_error", "") == ""


# ---------------------------------------------------------------------------
# B10 — carrier projection is token-exact and leaves zero desyncs.
# ---------------------------------------------------------------------------


def test_carrier_projection_is_token_exact_and_desync_free(tmp_path, monkeypatch):
    from ouroboros.tools.release_sync import version_carrier_desyncs

    repo, head = tua._init_repo(tmp_path)
    (repo / "VERSION").write_text("1.0.0\n")
    (repo / "web").mkdir()
    (repo / "web" / "package.json").write_text('{\n  "version": "1.0.0"\n}\n')
    tua._git(repo, "add", "-A")
    tua._git(repo, "commit", "-q", "-m", "carrier base")
    tua._git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "VERSION").write_text("2.0.0\n")
    (repo / "official.txt").write_text("official\n")
    tua._git(repo, "add", "-A")
    tua._git(repo, "commit", "-q", "-m", "official release")
    target = tua._git(repo, "rev-parse", "HEAD").stdout.strip()
    tua._git(repo, "checkout", "-q", head)
    (repo / "VERSION").write_text("1.5.0\n")
    (repo / "web" / "package.json").write_text('{\n  "version": "1.5.0"\n}\n')
    tua._git(repo, "add", "-A")
    tua._git(repo, "commit", "-q", "-m", "fork release")
    tua._point_at(monkeypatch, tmp_path, repo, head)
    # A REAL conflicted merge state (VERSION collides on both lines).
    tua._git(repo, "merge", "--no-commit", "--no-ff", target)

    ok, note, error = update_candidate.project_version_carriers(target)

    assert ok, error
    assert "VERSION projected to the target's version" in note
    # Token-level: the exact target VERSION, in the worktree AND the index.
    assert (repo / "VERSION").read_text() == "2.0.0\n"
    assert tua._git(repo, "show", ":VERSION").stdout == "2.0.0\n"
    # The mechanical carrier is token-synced to the same version…
    package_text = (repo / "web" / "package.json").read_text()
    assert '"version": "2.0.0"' in package_text
    # …and the SSOT desync detector confirms ZERO residual desyncs.
    assert version_carrier_desyncs("2.0.0", web_package_text=package_text) == []
    # Contrast: the pre-projection fork token WOULD have been a desync.
    assert version_carrier_desyncs("2.0.0", web_package_text='{"version": "1.5.0"}') != []


# ---------------------------------------------------------------------------
# Synthesis panel fixes (commit #9): F1 corrupt-tx fail-closed at both
# converted phase-write sites; F2 process-held test-proof authority surviving
# the advisory→commit tool-call boundary.
# ---------------------------------------------------------------------------


def _corrupt_marker():
    path = update_merge._update_tx_marker_path()
    path.write_text("{ this is not json", encoding="utf-8")
    return path


def test_update_tx_phase_refuses_to_overwrite_a_corrupt_marker(tmp_path, monkeypatch):
    """F1 core contract: an ABSENT marker keeps the snapshot fallback (creation
    semantics), a CORRUPT marker raises the typed failure WITHOUT writing —
    the corruption evidence read_update_tx_strict preserves stays byte-intact."""
    import pytest

    repo, head, plan, tx = tua._materialized_conflict_tx(tmp_path, monkeypatch)
    marker = _corrupt_marker()
    corrupt_bytes = marker.read_bytes()
    with pytest.raises(update_merge.UpdateTxCorrupt):
        update_merge.update_tx_phase(dict(tx), {"phase": "committing_assisted"})
    assert marker.read_bytes() == corrupt_bytes, "corrupt marker was overwritten"
    # Absent marker -> snapshot fallback still creates the record.
    marker.unlink()
    merged = update_merge.update_tx_phase(dict(tx), {"phase": "rolling_back"})
    assert merged["phase"] == "rolling_back"
    assert update_merge.read_update_tx().get("phase") == "rolling_back"


def test_committing_site_fails_typed_on_a_corrupt_marker(tmp_path, monkeypatch):
    """F1 site 1 (git.py commit flow): the committing_assisted transition on a
    corrupt marker returns the typed MANAGED_UPDATE_ERROR and leaves the marker
    bytes untouched — consistent with the corrupt-preflight block."""
    import inspect

    import ouroboros.tools.git as git_mod

    repo, head, plan, tx = tua._materialized_conflict_tx(tmp_path, monkeypatch)
    marker = _corrupt_marker()
    corrupt_bytes = marker.read_bytes()
    error = git_mod._managed_committing_phase_error(dict(tx))
    assert error and "MANAGED_UPDATE_ERROR" in error and "corrupt" in error
    assert marker.read_bytes() == corrupt_bytes, "corrupt marker was overwritten"
    # The real commit flow is bound to this helper (fail-closed return path).
    src = inspect.getsource(git_mod._repo_commit_push)
    assert "_managed_committing_phase_error(_managed_tx)" in src
    assert "return _fail(_phase_error)" in src
    # Valid marker -> the transition proceeds and merge-writes onto the fresh tx.
    marker.unlink()
    update_merge.write_update_tx(dict(tx))
    assert git_mod._managed_committing_phase_error(dict(tx)) is None
    assert update_merge.read_update_tx().get("phase") == "committing_assisted"


def test_postcommit_site_skips_phase_write_loudly_on_a_corrupt_marker(tmp_path, monkeypatch):
    """F1 site 2 (managed_assisted_postcommit): a corrupt marker skips BOTH
    phase writes with a loud supervisor row; the commit already landed, so the
    flow keeps going and the marker bytes stay untouched for the owner."""
    repo, head, plan, tx = tua._materialized_conflict_tx(tmp_path, monkeypatch)
    marker = _corrupt_marker()
    corrupt_bytes = marker.read_bytes()
    rows = []
    monkeypatch.setattr(update_merge, "_log_supervisor", rows.append)
    monkeypatch.setattr(update_merge, "update_restart_smoke", lambda: {"ok": True})
    ok, msg = update_merge.managed_assisted_postcommit(dict(tx), "f" * 40)
    assert ok, msg
    assert marker.read_bytes() == corrupt_bytes, "corrupt marker was overwritten"
    skipped = [r for r in rows if r.get("type") == "managed_update_tx_phase_write_skipped_corrupt"]
    assert len(skipped) == 2, rows
    assert "phase" in skipped[0]["patch_keys"]


def test_ctx_proof_survives_the_advisory_to_commit_boundary(tmp_path, monkeypatch):
    """F2(c): the proof pinned by the ADVISORY recording site is consulted by
    the COMMIT-side gates through the same task ctx (one ctx spans every tool
    call of a task); both recording sites are bound to the shared helper."""
    import inspect

    import ouroboros.tools.claude_advisory_review as adv_mod
    import ouroboros.tools.git as git_mod
    from ouroboros.commit_admission import run_tests_preflight_with_proof
    from ouroboros.tools.review_helpers import _run_review_preflight_tests
    from tests.test_advisory_preflight import _stub_preflight_lanes

    repo, head, plan, tx = tua._materialized_conflict_tx(tmp_path, monkeypatch)
    meta = tua._authority_metadata(tx)
    lanes = _stub_preflight_lanes(repo, monkeypatch)
    ctx = SimpleNamespace(task_id="resolver", task_metadata=meta, repo_dir=str(repo),
                          emit_progress_fn=lambda *_a, **_k: None)

    # The advisory's test consumer lets the runner mint the actual proof.
    assert run_tests_preflight_with_proof(ctx, runner=_run_review_preflight_tests) is None
    proof = ctx._preflight_test_proof
    assert proof and len(lanes) == 2
    tree = proof.tree
    # Both gates run their pytest preflight through the commit-admission SSOT,
    # whose helper owns the green-run -> proof binding (Q3=A extraction).
    for site_src in (
        inspect.getsource(adv_mod._advisory_pre_sdk_gate),
        inspect.getsource(git_mod._advisory_and_tests_gate),
    ):
        assert "run_tests_preflight_with_proof" in site_src
    assert "record_managed_tests_proof(ctx)" in inspect.getsource(
        run_tests_preflight_with_proof)

    # ...and the commit-side consumers see it on the SAME ctx.
    assert git_mod._managed_candidate_needs_proof(ctx) is False
    tua._git(repo, "add", "-A")
    tua._git(repo, "commit", "-q", "-m", "managed resolution")
    committed_tree = tua._git(repo, "rev-parse", "HEAD^{tree}").stdout.strip()
    assert committed_tree == tree
    assert git_mod._managed_post_commit_tests_gate(
        ctx, "msg", 0.0, True, [""], update_merge.read_update_tx(),
    ) is None
    assert len(lanes) == 4 and ctx._preflight_test_proof.tree == tree
    assert ctx._preflight_test_proof.head != proof.head
    # A FRESH ctx (restart analogue) holds no proof -> the mandate re-runs once.
    fresh = SimpleNamespace(task_id="resolver", task_metadata=meta, repo_dir=str(repo),
                            emit_progress_fn=lambda *_a, **_k: None)
    assert git_mod._managed_candidate_needs_proof(fresh) is True


def test_forged_durable_evidence_never_satisfies_needs_proof(tmp_path, monkeypatch):
    """F2(a), pre-commit side: durable tests_evidence forged to the candidate
    tree (the tx marker is resolver-writable) without a ctx record leaves the
    compensating preflight MANDATORY."""
    repo, head, plan, tx = tua._materialized_conflict_tx(tmp_path, monkeypatch)
    meta = tua._authority_metadata(tx)
    import ouroboros.tools.git as git_mod

    cand_tree, err = update_merge.worktree_snapshot_tree("HEAD")
    assert cand_tree, err
    forged = dict(update_merge.read_update_tx())
    forged["tests_evidence"] = {"tree": cand_tree, "at": "forged"}
    update_merge.write_update_tx(forged)
    ctx = SimpleNamespace(task_id="resolver", task_metadata=meta, repo_dir=str(repo),
                          emit_progress_fn=lambda *_a, **_k: None)
    assert update_merge.managed_tests_evidence_covers(cand_tree)  # forensic view
    assert git_mod._managed_candidate_needs_proof(ctx) is True, (
        "forged durable tests_evidence suppressed the mandatory pre-commit suite"
    )
