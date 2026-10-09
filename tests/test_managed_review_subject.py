"""Lane L-review: the managed resolution-delta review subject (Δ4), the
assembly-before-dispatch admission (Q25=A), the Q28-A oversized outcomes, and
the enforcement-honest advisory guidance (O1).

The managed fixtures drive the REAL flow: a temp repo with a genuine conflicted
merge materialized by ``materialize_assisted_merge_live`` (which pins M0), a
resolver edit, and the durable tx + authority metadata the registry predicate
actually verifies.
"""

import pathlib
import subprocess
from types import SimpleNamespace

import pytest

import supervisor.git_ops as git_ops
import supervisor.update_merge as update_merge
from ouroboros.tools.review_binary_context import capture_staged_diff
from ouroboros.tools.review_subject import (
    build_triad_session_task,
    capture_review_diff,
    managed_review_subject,
)


@pytest.fixture(autouse=True)
def _packet_default_panel(monkeypatch):
    """This module pins the PACKET assembly of the review pool: three packet seats
    on the factory models (the pool's own default delivery is native)."""
    from tests.review_pool_rosters import set_review_pool

    set_review_pool(monkeypatch)


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def _point_at(monkeypatch, tmp_path, repo, head):
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "BRANCH_DEV", head)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    (tmp_path / "data" / "logs").mkdir(parents=True, exist_ok=True)


def _managed_resolution_repo(tmp_path, monkeypatch, official_binary=False):
    """A real conflicted managed merge, materialized (M0 pinned), then resolved.

    Layout: the official target edits ``conflict.txt`` (collides with a local
    commit) AND adds ``official.txt`` (merges mechanically — it must NEVER
    appear in the resolution delta). The resolver resolves the conflict and
    adds one file of its own. ``official_binary=True`` also lets the official
    target add ``payload`` — an EXTENSIONLESS binary absent from HEAD (the R4
    managed-deletion topology)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "conflict.txt").write_text("base\n")
    (repo / "keep.txt").write_bytes(b"k1\nk2\nk3\nk4\nk5\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    head = _git(repo, "symbolic-ref", "--short", "HEAD").stdout.strip()
    base_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "conflict.txt").write_text("official\n")
    (repo / "official.txt").write_text("released official change\n")
    if official_binary:
        (repo / "payload").write_bytes(b"\x00\x01\x02official binary\x00\xff")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "official release")
    target_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "-q", head)
    (repo / "conflict.txt").write_text("local\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "local line")
    pre_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _point_at(monkeypatch, tmp_path, repo, head)

    ok, msg, m0_tree = update_merge.materialize_assisted_merge_live(
        head, pre_sha, target_sha, base_sha
    )
    assert ok and m0_tree, msg
    # The resolver's work: resolve the conflict, add one file of its own.
    (repo / "conflict.txt").write_text("resolved by the agent\n")
    (repo / "resolver_note.txt").write_text("resolver-added\n")
    _git(repo, "add", "-A")

    tx = {
        "phase": "assisted_resolution",
        "task_id": "resolver-task",
        "pre_update_sha": pre_sha,
        "target_sha": target_sha,
        "m0_tree": m0_tree,
        "conflict_paths": ["conflict.txt"],
    }
    update_merge.write_update_tx(tx)
    ctx = SimpleNamespace(
        task_id="resolver-task",
        task_metadata={
            "managed_update": {
                "authority_fingerprint": update_merge.assisted_authority_fingerprint(tx),
            }
        },
        repo_dir=str(repo),
    )
    return repo, ctx, tx


# ---------------------------------------------------------------------------
# A. The resolution-delta capture
# ---------------------------------------------------------------------------

def test_managed_capture_returns_resolution_delta_only(tmp_path, monkeypatch):
    repo, ctx, tx = _managed_resolution_repo(tmp_path, monkeypatch)

    rendered = capture_review_diff(ctx, repo)

    # Only the resolver's work is rendered...
    assert "resolved by the agent" in rendered
    assert "resolver_note.txt" in rendered
    # ...the mechanically merged official delta is NOT re-rendered...
    assert "official.txt" not in rendered
    # ...while the FULL staged candidate provably contains it (the contrast).
    assert "official.txt" in capture_staged_diff(repo)
    # Disclosure header: identities, both parents, counters, anchors.
    assert f"mechanical merge M0 {tx['m0_tree'][:12]}" in rendered
    assert tx["pre_update_sha"][:12] in rendered and tx["target_sha"][:12] in rendered
    assert "full candidate paths:" in rendered
    assert "reviewed resolution paths:" in rendered
    assert "conflict anchors" in rendered and "conflict.txt" in rendered
    assert "is not re-rendered" in rendered


def test_managed_capture_supports_zero_context_rung(tmp_path, monkeypatch):
    repo, ctx, _tx = _managed_resolution_repo(tmp_path, monkeypatch)
    # A mid-file resolver edit gives the ladder real context lines to drop.
    (repo / "keep.txt").write_bytes(b"k1\nk2\nk3-resolved\nk4\nk5\n")
    _git(repo, "add", "-A")

    full = capture_review_diff(ctx, repo)
    compact = capture_review_diff(ctx, repo, unified=0)

    assert compact != full
    assert " k1\n" in full and " k1\n" not in compact  # -U0 drops unchanged context
    assert "resolved by the agent" in compact
    assert "official.txt" not in compact


def test_dual_counters_reflect_candidate_vs_resolution(tmp_path, monkeypatch):
    repo, ctx, _tx = _managed_resolution_repo(tmp_path, monkeypatch)

    subject = managed_review_subject(ctx, repo)

    # Candidate vs local HEAD: conflict.txt + official.txt (>= 2 paths);
    # resolution delta: conflict.txt + resolver_note.txt (exactly 2).
    assert subject.full_candidate_paths >= 2
    assert subject.resolution_paths == 2
    assert {p for _s, p in subject.name_status} == {"conflict.txt", "resolver_note.txt"}
    assert subject.touched_paths() == ["conflict.txt", "resolver_note.txt"]
    assert subject.counters_line() in subject.header()


def test_gate_subject_carries_index_content_not_worktree(tmp_path, monkeypatch):
    """M1 regression: the reviewed gate subject is bound to the tree that
    commits. Pre-stage EVIL content, then restore an innocent worktree copy:
    the old worktree-snapshot S showed reviewers the innocent copy while the
    EVIL index committed."""
    repo, ctx, _tx = _managed_resolution_repo(tmp_path, monkeypatch)
    (repo / "conflict.txt").write_text("EVIL-PAYLOAD\n")
    _git(repo, "add", "conflict.txt")
    (repo / "conflict.txt").write_text("resolved by the agent\n")  # index keeps EVIL

    subject = managed_review_subject(ctx, repo)  # gate surface (default)

    index_tree = _git(repo, "write-tree").stdout.strip()
    assert subject.staged_tree == index_tree  # same tree the fingerprint pins
    rendered = subject.render_prompt_diff()
    assert "EVIL-PAYLOAD" in rendered           # what commits IS what is reviewed
    assert "resolved by the agent" not in rendered  # worktree-only content is not
    # The gate subject records its tree for the commit gate's binding assert.
    assert index_tree in getattr(ctx, "_last_review_subject_trees", set())

    # The ADVISORY surface stays on the worktree by contract (work-in-progress
    # review; freshness handles staleness) and records NO gate binding tree.
    advisory = managed_review_subject(ctx, repo, surface="advisory")
    assert "resolved by the agent" in advisory.render_prompt_diff()
    assert advisory.staged_tree != subject.staged_tree
    assert advisory.staged_tree not in ctx._last_review_subject_trees


def test_frozen_system_index_of_a_managed_resolution_is_the_managed_artifact(tmp_path, monkeypatch):
    """§6 Subject operation: freezing the system repo's ``index`` goes through
    the gate's own managed path — the frozen subject carries the SAME artifact
    and S tree as ``managed_review_subject(surface="gate")``, records the gate
    binding tree, and the wave's capture seam keeps reading the system index
    through the live managed call (byte-identical to today)."""
    from ouroboros.tools.review import _capture_triad_staged_diff
    from ouroboros.tools.review_subject import ReviewSubjectSpec, freeze_subject

    repo, ctx, _tx = _managed_resolution_repo(tmp_path, monkeypatch)
    frozen = freeze_subject(ctx, ReviewSubjectSpec(
        root_kind="system_repo", root=str(repo), kind="index", surface="commit_gate", layer="body"))
    subject = managed_review_subject(ctx, repo)

    assert frozen.is_system_index and frozen.managed is not None
    assert frozen.diff_text == subject.render_prompt_diff() == capture_review_diff(ctx, repo)
    assert "resolver_note.txt" in frozen.diff_text and "official.txt" not in frozen.diff_text
    assert frozen.tree_sha == subject.staged_tree == _git(repo, "write-tree").stdout.strip()
    assert frozen.parent_sha == _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert frozen.name_status == subject.name_status
    assert frozen.tree_sha in ctx._last_review_subject_trees  # the gate's binding assert input
    assert frozen.render_prompt_diff(unified=0) == capture_review_diff(ctx, repo, unified=0)
    # The triad capture seam: the system index is read through the live managed
    # call (same artifact, same S tree), not replayed from the frozen copy.
    diff_text, seam_subject, block = _capture_triad_staged_diff(ctx, str(repo), True, frozen=frozen)
    assert block is None and diff_text == frozen.diff_text
    assert seam_subject is not None and seam_subject.staged_tree == frozen.tree_sha


def test_gate_subject_binding_mismatch_blocks_commit(tmp_path, monkeypatch):
    """M1 defense-in-depth: the commit gate asserts (typed failure) that the
    reviewed subject tree equals the review-binding fingerprint's tree_sha."""
    from ouroboros.tools import git as git_mod
    from ouroboros.tools.registry import ToolContext

    repo = tmp_path / "gaterepo"
    repo.mkdir()
    drive = tmp_path / "gatedata"
    (drive / "logs").mkdir(parents=True)
    (drive / "locks").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    (repo / "a.txt").write_text("b\n")
    ctx = ToolContext(repo_dir=repo, drive_root=drive)

    recorded_tree = {}

    def _fake_parallel(ctx, *a, **kw):
        # Simulate the divergence: the gate subjects reviewed some OTHER tree.
        ctx._last_review_subject_trees = {recorded_tree["value"]}
        return None, None, "", []

    monkeypatch.setattr(git_mod, "_run_review_preflight_tests", lambda *a, **kw: None)
    monkeypatch.setattr(git_mod, "_run_parallel_review", _fake_parallel)
    monkeypatch.setattr(
        git_mod, "_aggregate_review_verdict", lambda *a, **kw: (False, "", "", [], [])
    )

    recorded_tree["value"] = "f" * 40  # not the staged index tree
    outcome = git_mod._run_reviewed_stage_cycle(
        ctx, commit_message="binding mismatch", commit_start=0.0,
        skip_advisory_pre_review=True,
    )
    assert outcome["status"] == "blocked"
    assert outcome["block_reason"] == "review_subject_binding_mismatch"
    assert "not bound to the staged candidate" in outcome["message"]

    # Positive control: a subject tree equal to the binding passes the assert.
    def _fake_parallel_ok(ctx, *a, **kw):
        tree = subprocess.run(
            ["git", "-C", str(repo), "write-tree"], capture_output=True, text=True
        ).stdout.strip()
        ctx._last_review_subject_trees = {tree}
        return None, None, "", []

    monkeypatch.setattr(git_mod, "_run_parallel_review", _fake_parallel_ok)
    outcome = git_mod._run_reviewed_stage_cycle(
        ctx, commit_message="binding ok", commit_start=0.0,
        skip_advisory_pre_review=True,
    )
    assert outcome["status"] == "passed"


def test_non_managed_capture_is_byte_identical(tmp_path, monkeypatch):
    repo, _ctx, _tx = _managed_resolution_repo(tmp_path, monkeypatch)
    stranger = SimpleNamespace(task_id="someone-else", task_metadata=None, repo_dir=str(repo))

    assert capture_review_diff(stranger, repo) == capture_staged_diff(repo)
    assert capture_review_diff(None, repo, unified=0) == capture_staged_diff(repo, unified=0)


def test_m0_missing_falls_back_to_full_diff_with_loud_disclosure(tmp_path, monkeypatch):
    repo, ctx, tx = _managed_resolution_repo(tmp_path, monkeypatch)
    tx.pop("m0_tree")
    tx["m0_missing_reason"] = "resumed_with_progress_before_m0_pin"
    update_merge.write_update_tx(tx)
    ctx.task_metadata = {
        "managed_update": {
            "authority_fingerprint": update_merge.assisted_authority_fingerprint(tx),
        }
    }

    rendered = capture_review_diff(ctx, repo)

    assert "M0 BASELINE UNAVAILABLE" in rendered
    assert "resumed_with_progress_before_m0_pin" in rendered
    assert "official.txt" in rendered  # the full candidate diff is the fallback body
    # M4: no delta exists — the counters must not claim a resolution count.
    assert "reviewed resolution paths: n/a (M0 missing" in rendered
    subject = managed_review_subject(ctx, repo)
    assert subject.fallback_full_diff is True
    # M2: the reviewed path set covers the FULL candidate (what the full diff
    # and the commit contain), never just the conflict anchors.
    assert set(subject.touched_paths()) >= {
        "conflict.txt", "official.txt", "resolver_note.txt",
    }


def test_m0_missing_session_fallback_texts_are_honest(tmp_path, monkeypatch):
    """M4: the SESSION variant of the M0-missing packet renders NO diff body —
    its header must instruct retrieval, not claim a rendering below, for BOTH
    the packet seat's session task and the retrieving seat's two-part brief
    (whose Part 1 is that same task)."""
    from ouroboros.tools.review_brief_coupling import BriefInputs, BriefIntent, build_retrieving_brief
    from ouroboros.tools.review_helpers import REPO_ROOT

    repo, ctx, tx = _managed_resolution_repo(tmp_path, monkeypatch)
    tx.pop("m0_tree")
    tx["m0_missing_reason"] = "resumed_with_progress_before_m0_pin"
    update_merge.write_update_tx(tx)
    ctx.task_metadata = {
        "managed_update": {
            "authority_fingerprint": update_merge.assisted_authority_fingerprint(tx),
        }
    }
    subject = managed_review_subject(ctx, repo)
    assert subject.fallback_full_diff is True

    triad_task = build_triad_session_task(subject=subject, **_SESSION_SECTIONS)
    brief, _m = build_retrieving_brief(repo, BriefInputs(
        commit_message="land the update",
        intent=BriefIntent(goal="g", scope="s"),
        governance_repo_dir=pathlib.Path(REPO_ROOT),
        managed_subject=subject,
    ))
    for task in (triad_task, brief):
        assert "M0 BASELINE UNAVAILABLE" in task
        assert "retrieve the FULL staged candidate diff yourself" in task
        assert "`git diff --cached`" in task
        assert "rendered below" not in task  # no body follows in session delivery
        assert "reviewed resolution paths: n/a (M0 missing" in task


def test_binary_rows_carry_m0_baseline_identity(tmp_path, monkeypatch):
    from ouroboros.tools.review_binary_context import render_staged_binary_metadata

    repo, ctx, tx = _managed_resolution_repo(tmp_path, monkeypatch)
    (repo / "blob.bin").write_bytes(b"\x00\x01resolver binary\x02")
    _git(repo, "add", "blob.bin")

    plain = render_staged_binary_metadata(repo, "blob.bin")
    managed = render_staged_binary_metadata(repo, "blob.bin", m0_tree=tx["m0_tree"])

    assert plain is not None and "mechanical merge M0 blob" not in plain
    assert managed is not None
    assert "- mechanical merge M0 blob: `absent`" in managed
    # Both real merge parents stay rendered.
    assert "pre-merge HEAD blob" in managed and "official MERGE_HEAD blob" in managed


# ---------------------------------------------------------------------------
# Session task texts inline the managed artifact
# ---------------------------------------------------------------------------

_SESSION_SECTIONS = dict(
    goal_section="## Goal\nland the update",
    scope_section="## Scope\nresolution only",
    checklist_section="## Review Checklist\n- correctness",
    rebuttal_section="",
    review_history_section="",
    dev_guide_text="# Dev\n\n## Rules\n\ntext\n",
    architecture_text="## Parent\nbody\n",
)


def test_triad_session_task_inlines_managed_delta(tmp_path, monkeypatch):
    repo, ctx, _tx = _managed_resolution_repo(tmp_path, monkeypatch)
    subject = managed_review_subject(ctx, repo)

    task = build_triad_session_task(subject=subject, **_SESSION_SECTIONS)

    assert "AUTHORITATIVE review subject" in task
    assert "resolved by the agent" in task
    assert "do NOT substitute your own `git diff --cached`" in task
    assert "Retrieve it yourself" not in task
    # An ordinary commit keeps the retrieval pointer.
    plain = build_triad_session_task(subject=None, **_SESSION_SECTIONS)
    assert "Retrieve it yourself" in plain and "resolved by the agent" not in plain


def test_two_part_brief_inlines_managed_delta(tmp_path, monkeypatch):
    from ouroboros.tools.review_brief_coupling import BriefInputs, BriefIntent, build_retrieving_brief
    from ouroboros.tools.review_helpers import REPO_ROOT

    repo, ctx, _tx = _managed_resolution_repo(tmp_path, monkeypatch)
    subject = managed_review_subject(ctx, repo)

    task, manifest = build_retrieving_brief(repo, BriefInputs(
        commit_message="land the update",
        intent=BriefIntent(goal="g", scope="s"),
        governance_repo_dir=pathlib.Path(REPO_ROOT),
        managed_subject=subject,
    ))

    assert "AUTHORITATIVE review subject" in task
    assert "resolved by the agent" in task
    assert "conflict.txt" in task
    assert manifest["diff_delivery"] == "inline" and manifest["parts"] == ["change", "coupling"]
    # The subject is the resolution delta: the already-released official change
    # is NOT re-rendered, and the brief never points at `git diff --cached`.
    assert "released official change" not in task
    assert "do NOT substitute your own `git diff --cached`" in task

    # An ordinary commit's subject is the staged diff itself, official delta
    # included — and it carries none of the managed subject's authority wording.
    plain_task, plain_manifest = build_retrieving_brief(repo, BriefInputs(
        commit_message="land the update",
        intent=BriefIntent(goal="g", scope="s"),
        governance_repo_dir=pathlib.Path(REPO_ROOT),
    ))
    assert "AUTHORITATIVE review subject" not in plain_task
    assert "released official change" in plain_task
    assert plain_manifest["diff_delivery"] == "inline"


# ---------------------------------------------------------------------------
# B. Assembly-before-dispatch admission (Q25=A) — $0 on deterministic no-fit
# ---------------------------------------------------------------------------

def _admission_ctx(repo):
    return SimpleNamespace(
        repo_dir=str(repo),
        drive_root=str(repo.parent / "data"),
        task_id="t-admission",
        task_metadata=None,
        _review_history=[],
        _review_advisory=[],
    )


def _plain_repo(tmp_path):
    repo = tmp_path / "plainrepo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "x.txt").write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    (repo / "x.txt").write_text("y\n")
    _git(repo, "add", "-A")
    return repo


def test_packet_assembly_block_dispatches_nothing(tmp_path, monkeypatch):
    """A deterministic assembly block exits BEFORE the wave: no seat is
    dispatched and no coupling outcome is invented for a wave that never had
    seats."""
    from ouroboros.tools import parallel_review as pr
    from ouroboros.tools import review as review_mod

    repo = _plain_repo(tmp_path)
    ctx = _admission_ctx(repo)
    block = "⚠️ REVIEW_BLOCKED: deterministic assembly failure"
    monkeypatch.setattr(
        review_mod, "_prepare_unified_review", lambda *a, **k: (None, block, True)
    )
    monkeypatch.setattr(
        review_mod, "_dispatch_unified_review",
        lambda *a, **k: pytest.fail("the wave dispatched despite assembly block"),
    )

    review_err, coupling, _reason, _advisory = pr.run_parallel_review(ctx, "msg")

    assert review_err == block
    assert coupling is None
    assert ctx._last_review_structured["assembly_refusal"] == block
    assert ctx._last_review_structured["rows"] == []


def test_brief_assembly_failure_blocks_the_whole_wave_before_dispatch(tmp_path, monkeypatch):
    """The REAL assembly over the plain repo, with the retrieving seat's brief
    builder failing: the wave exits typed (infra_failure) naming the seat, and
    nothing — neither the packet seats nor the retrieving seat — is dispatched
    ($0). There is no second review to fall back to."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools import parallel_review as pr
    from ouroboros.tools import review as review_mod
    from ouroboros.tools import review_admission as admission
    import ouroboros.reviewer_slot_config as slot_cfg

    repo = _plain_repo(tmp_path)
    ctx = _admission_ctx(repo)
    ctx._review_iteration_count = 0
    plan = _row_plan(["api", "api"])
    plan.update(subagent_ids=["", "critic"], retrieves=[False, True])
    plan["routes"] = [ReviewRouteKind.API_CHAT, ReviewRouteKind.API_CHAT]
    monkeypatch.setattr(slot_cfg, "commit_triad_delivery", lambda: plan)

    def _boom(**_kwargs):
        raise RuntimeError("the coupling checklist could not be loaded")

    monkeypatch.setattr(admission, "retrieving_brief_for_seat", _boom)
    monkeypatch.setattr(
        review_mod, "_dispatch_unified_review",
        lambda *a, **k: pytest.fail("the wave dispatched despite the brief failure"),
    )
    from ouroboros import config as cfg

    monkeypatch.setattr(cfg, "get_review_enforcement", lambda: "blocking")

    review_err, coupling, reason, _advisory = pr.run_parallel_review(ctx, "msg")

    assert review_err and "Failed to build the review brief" in review_err
    assert "Seat slot_1: RuntimeError: the coupling checklist could not be loaded" in review_err \
        or "Seat slot_1: the coupling checklist could not be loaded" in review_err
    assert reason == "infra_failure" and coupling is None
    assert ctx._last_triad_raw_results == []


def test_healthy_assembly_dispatches_the_one_wave(tmp_path, monkeypatch):
    """The REAL assembly runs over the plain-repo fixture; only the LLM
    dispatch seam is patched — the dispatched packet must carry the ACTUAL
    staged hunk, proving assembly assembled the real evidence (m3a), and the
    one seat list carries every seat's ``parts`` — the retrieving seats asked the
    coupling question beside the packet seats."""
    from ouroboros.review_ledger import CouplingOutcome
    from ouroboros.tools import parallel_review as pr
    from ouroboros.tools import review as review_mod
    from tests.review_pool_rosters import mixed_pool_rows, pool_roster, set_review_pool

    # A mixed pool: the retrieving seats carry the coupling question of the one wave.
    set_review_pool(monkeypatch, pool_roster(*mixed_pool_rows()))
    repo = _plain_repo(tmp_path)
    ctx = _admission_ctx(repo)
    ctx._review_iteration_count = 0
    calls = {"wave": 0}

    def fake_dispatch(_ctx, _msg, prepared):
        calls["wave"] += 1
        # The REAL assembled api pack carries the actual staged hunk (x -> y).
        assert "+y" in prepared["prompt"] and "-x" in prepared["prompt"]
        assert prepared["models"], "resolved reviewer rows must ride with the packet"
        plan = prepared["row_plan"]
        assert len(plan["parts"]) == len(prepared["models"])
        asked = [i for i, p in enumerate(plan["parts"]) if "coupling" in p]
        assert asked, "the one wave carries the coupling question"
        for i in asked:
            assert "## Part 2 — Coupling questions" in plan["session_tasks"][i]
            assert "+y" in plan["session_tasks"][i]  # the brief's Part 1 carries the change too
        _ctx._last_coupling_result = CouplingOutcome(verdict="PASS", status="responded")
        return None

    monkeypatch.setattr(review_mod, "_dispatch_unified_review", fake_dispatch)

    review_err, coupling, _reason, _advisory = pr.run_parallel_review(
        ctx, "healthy assembly test commit"
    )

    assert review_err is None
    assert calls == {"wave": 1}
    assert coupling is not None and coupling.status == "responded"
    structured = ctx._last_review_structured
    assert [r["parts"] for r in structured["rows"]] == [["change", "coupling"], ["change", "coupling"], ["change"]]
    assert structured["brief_texts"] and all(
        r["brief_sha"] in structured["brief_texts"] for r in structured["rows"] if "coupling" in r["parts"])


# ---------------------------------------------------------------------------
# Q28-A oversized outcomes
# ---------------------------------------------------------------------------

def _triad_real_fit_env(tmp_path, monkeypatch, row_plan):
    """_prepare_unified_review over the REAL managed repo with the REAL fit
    ladder (m3c): every api slot's calibrated input limit is pinned tiny
    through the documented patch seam, so the genuinely assembled pack — full
    snapshots, then the fit note, then the subject's own -U0 rung — overflows
    at every rung and terminates in the ladder's own block message. No error
    string is injected anywhere."""
    from ouroboros.tools import review as review_mod
    import ouroboros.reviewer_slot_config as slot_cfg

    repo, ctx, _tx = _managed_resolution_repo(tmp_path, monkeypatch)
    ctx.drive_root = str(tmp_path / "data")
    ctx._review_history = []
    ctx._review_advisory = []
    ctx._review_iteration_count = 0
    monkeypatch.setattr(slot_cfg, "commit_triad_delivery", lambda: row_plan)
    # The configured panel alone: the transitional fold of the old scope rows
    # is pinned by test_healthy_assembly_dispatches_the_one_wave.
    monkeypatch.setattr(
        review_mod, "calibrated_input_token_limit", lambda *a, **k: 50
    )
    return review_mod, ctx


def _row_plan(routes):
    from ouroboros.review_execution import ReviewRouteKind

    kinds = {
        "api": ReviewRouteKind.API_CHAT,
        "session": ReviewRouteKind.AGENT_SESSION,
    }
    return {
        "models": [f"m/{i}-{r}" for i, r in enumerate(routes)],
        "routes": [kinds[r] for r in routes],
        "efforts": ["" for _ in routes],
        "session_targets": ["" for _ in routes],
        "session_profiles": ["" for _ in routes],
        "use_local": [False for _ in routes],
        "slot_ids": [f"slot_{i}" for i, _ in enumerate(routes)],
    }


def test_fit_error_with_session_quorum_drops_api_rows(tmp_path, monkeypatch):
    review_mod, ctx = _triad_real_fit_env(
        tmp_path, monkeypatch, _row_plan(["api", "session", "session"])
    )

    prepared, early, exited = review_mod._prepare_unified_review(
        ctx, "resolve managed update conflicts"
    )

    assert not exited and early is None
    assert prepared["models"] == ["m/1-session", "m/2-session"]
    assert prepared["prompt"] == ""
    # Sessions still get the REAL two-part brief, with the managed delta inlined
    # in Part 1 — one brief per seat, the parts vector dropped in step.
    plan = prepared["row_plan"]
    assert plan["parts"] == [("change", "coupling"), ("change", "coupling")]
    for brief in plan["session_tasks"]:
        assert "AUTHORITATIVE review subject" in brief
        assert "resolved by the agent" in brief
        assert "## Part 2 — Coupling questions" in brief
    assert any(
        "triad_api_rows_dropped_oversize_pack" in r for r in ctx._review_degraded_reasons
    )
    assert ctx._last_triad_models == ["m/1-session", "m/2-session"]


def test_fit_error_without_session_quorum_is_typed_zero_spend_with_guidance(
    tmp_path, monkeypatch
):
    from ouroboros.tools.review_admission import (
        MANAGED_OVERSIZE_GUIDANCE,
        MANAGED_SPLIT_IMPOSSIBLE,
    )

    review_mod, ctx = _triad_real_fit_env(
        tmp_path, monkeypatch, _row_plan(["api", "api", "session"])
    )

    prepared, early, exited = review_mod._prepare_unified_review(
        ctx, "resolve managed update conflicts"
    )

    assert exited and prepared is None
    # The REAL ladder's terminal, with the managed wording REPLACING the
    # structurally impossible split clause (M3) — never appended below it.
    assert "irreducible one-pass triad prompt" in early
    assert MANAGED_SPLIT_IMPOSSIBLE in early
    assert MANAGED_OVERSIZE_GUIDANCE in early  # managed resolver carries Q28-A guidance
    assert "Settings → Agents, Reviewer rows" in early
    assert "Split or shrink the staged change" not in early
    assert ctx._last_review_block_reason == "fixed_overflow"


def test_a_retrieving_seat_is_never_refused_for_the_packet_it_does_not_receive(tmp_path, monkeypatch):
    """One wave, two deliveries: the packet fit ladder sizes the PACKET seats;
    a retrieving seat is given its own two-part brief sized to its own first
    send, so an oversized packet drops the packet api rows and leaves every
    retrieving seat — both parts — in the wave (the former scope row's "yield
    quorum" has no object any more: no seat is refused for a packet it never
    receives)."""
    from ouroboros.review_execution import ReviewRouteKind

    plan = _row_plan(["api", "api", "session"])
    plan.update(subagent_ids=["", "critic", ""], retrieves=[False, True, True])
    plan["routes"] = [ReviewRouteKind.API_CHAT, ReviewRouteKind.API_CHAT, ReviewRouteKind.AGENT_SESSION]
    review_mod, ctx = _triad_real_fit_env(tmp_path, monkeypatch, plan)

    prepared, early, exited = review_mod._prepare_unified_review(
        ctx, "resolve managed update conflicts"
    )

    assert not exited and early is None
    assert prepared["models"] == ["m/1-api", "m/2-session"]
    assert prepared["row_plan"]["parts"] == [("change", "coupling"), ("change", "coupling")]
    assert all("## Part 2 — Coupling questions" in t for t in prepared["row_plan"]["session_tasks"])
    withheld = getattr(ctx, "_triad_withheld_seat_records", [])
    assert [r["model_id"] for r in withheld] == ["m/0-api"]


@pytest.mark.parametrize("additional", [False, True], ids=["counted_control", "outside_quorum"])
def test_an_oversize_drop_never_promotes_an_added_critic_into_the_quorum(tmp_path, monkeypatch, additional):
    """D2-NEW: a pool of two marked seats (packet + reading) plus a critic the author
    added beside the pool; the patch does not fit the packet seat. The Q28-A yield is
    decided over the COUNTED seats and their own quorum: with the added critic outside
    the pool only ONE counted retrieving seat remains, so the wave is the typed $0
    ``fixed_overflow`` terminal — the critic is heard, never the second vote. The
    control marks the same third row: it IS a counted seat, the packet row is dropped
    and the two retrieving seats carry the wave to PASS. Composed through the REAL
    ``compose_panel`` and the REAL fit ladder on the managed repo; only the model
    answers are supplied."""
    import json

    from ouroboros import review_ledger as ledger
    from ouroboros import reviewer_slot_config as slots
    from ouroboros.tools.review_change import ReviewChangeRequest, compose_panel
    from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
    from tests.test_workflow_review_outcomes import _seat, _two_part

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    set_review_pool(monkeypatch, pool_roster(
        pool_seat("packet", "openai/gpt-5"),
        pool_seat("reading", "anthropic/claude-opus-4.1", delivery="native"),
        pool_seat("extra", "google/gemini-2.5-pro", delivery="native", marked=not additional),
    ))
    panel = compose_panel(ReviewChangeRequest(root="system_repo", subject="index", reviewers=("extra",)), adds_only=True)
    with slots.composed_review_pool(panel.seats):
        plan = slots.commit_triad_delivery()
    assert plan["additional"] == [False, False, additional]
    review_mod, ctx = _triad_real_fit_env(tmp_path, monkeypatch, plan)

    prepared, early, exited = review_mod._prepare_unified_review(ctx, "resolve the managed update")

    if additional:
        assert exited and prepared is None, "one counted retrieving seat cannot make the quorum of two"
        assert ctx._last_review_block_reason == "fixed_overflow" and "irreducible one-pass triad prompt" in early
        assert ctx._triad_withheld_seat_records == [] and ctx._last_triad_raw_results == []
        return
    assert not exited and early is None
    post = prepared["row_plan"]
    assert (post["slot_ids"], post["additional"]) == (["reading", "extra"], [False, False])
    results = [_seat(row_id, model, _two_part()) for row_id, model in zip(post["slot_ids"], post["models"])]
    monkeypatch.setattr(review_mod, "_handle_multi_model_review", lambda *_a, **_kw: json.dumps({"results": results}))
    error = review_mod._dispatch_unified_review(ctx, "resolve the managed update", prepared)
    rows = ledger.rows_from_plan(post, post["routes"], ctx._last_triad_raw_results)
    assert error is None and [(r["seat_id"], r["additional"]) for r in rows] == [("reading", False), ("extra", False)]
    verdict = ctx._last_review_verdict
    assert verdict["aggregate"] == "PASS" and (verdict["quorum"]["assigned"], verdict["quorum"]["required"]) == (2, 2)
    assert [r["model_id"] for r in ctx._triad_withheld_seat_records] == ["openai/gpt-5"]


def test_drop_api_rows_keeps_the_added_seat_bit_aligned_with_the_surviving_rows():
    """The aligned ``additional`` vector is filtered by the same indices as ``slot_ids``
    and ``parts``: after the packet rows leave, the added critic is still the added one."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools.review_admission import counted_retrieving_seats, drop_api_rows

    plan = _row_plan(["api", "session", "session"])
    plan.update(retrieves=[False, True, True], additional=[False, False, True],
                parts=[("change",), ("change", "coupling"), ("change", "coupling")])
    kept = drop_api_rows(plan)
    assert kept["slot_ids"] == ["slot_1", "slot_2"] and kept["additional"] == [False, True]
    assert kept["routes"] == [ReviewRouteKind.AGENT_SESSION] * 2 and kept["parts"] == [("change", "coupling")] * 2
    # The yield arithmetic: two counted seats owe a quorum of two; one of them is the packet row.
    assert counted_retrieving_seats(plan, [0]) == (1, 2)
    # Without the added bit the same rows are three counted seats: two retrieving ones meet the 2-of-3 quorum.
    assert counted_retrieving_seats({**plan, "additional": [False, False, False]}, [0]) == (2, 2)
    assert counted_retrieving_seats({k: v for k, v in plan.items() if k != "additional"}, [0]) == (2, 2)


# ---------------------------------------------------------------------------
# Panel fix round (R2-R9): S-consistent fallback, loud tx failure, M0-aware
# binary deletion, typed withheld-seat records, n/a counters, friendly reason
# ---------------------------------------------------------------------------

def _drop_m0(ctx, tx, reason="resumed_with_progress_before_m0_pin"):
    tx.pop("m0_tree", None)
    tx["m0_missing_reason"] = reason
    update_merge.write_update_tx(tx)
    ctx.task_metadata = {
        "managed_update": {
            "authority_fingerprint": update_merge.assisted_authority_fingerprint(tx),
        }
    }


def test_m0_missing_gate_fallback_body_is_pinned_index_tree(tmp_path, monkeypatch):
    """R2 (gate surface): the fallback body renders from the PINNED S — the
    index write-tree — via ``git diff HEAD..S``, never a second ``--cached``
    capture (which would weaken the binding to the pinned S)."""
    import ouroboros.tools.review_subject as rs

    repo, ctx, tx = _managed_resolution_repo(tmp_path, monkeypatch)
    _drop_m0(ctx, tx)
    # Index/worktree divergence: EVIL is staged, the worktree copy is innocent.
    (repo / "conflict.txt").write_text("EVIL-PAYLOAD\n")
    _git(repo, "add", "conflict.txt")
    (repo / "conflict.txt").write_text("innocent worktree copy\n")

    subject = managed_review_subject(ctx, repo)  # gate surface
    assert subject.fallback_full_diff is True
    monkeypatch.setattr(
        rs._rbc, "capture_staged_diff",
        lambda *a, **k: pytest.fail("fallback re-captured --cached instead of the pinned S"),
    )
    rendered = subject.render_prompt_diff()
    compact = subject.render_prompt_diff(unified=0)

    for body in (rendered, compact):
        assert "EVIL-PAYLOAD" in body          # what commits IS what is rendered
        assert "innocent worktree copy" not in body  # worktree-only content is not
    # R10a: the fallback lead must not claim the official delta is withheld.
    assert "is not re-rendered" not in rendered
    assert "INCLUDES the already-released official" in rendered


def test_m0_missing_advisory_fallback_body_matches_worktree_subject(tmp_path, monkeypatch):
    """R2 (advisory surface): S is the worktree snapshot — unstaged/untracked
    work must appear in the BODY, not only in the counters and name-status
    (the old ``--cached`` fallback body omitted it)."""
    repo, ctx, tx = _managed_resolution_repo(tmp_path, monkeypatch)
    _drop_m0(ctx, tx)
    (repo / "wip_unstaged.txt").write_text("worktree-only wip line\n")  # untracked

    subject = managed_review_subject(ctx, repo, surface="advisory")

    assert subject.fallback_full_diff is True
    assert "wip_unstaged.txt" in subject.touched_paths()
    rendered = subject.render_prompt_diff()
    assert "worktree-only wip line" in rendered  # body describes the same S
    assert "wip_unstaged.txt" in rendered


def test_authorized_resolver_with_broken_tx_gets_loud_fallback(tmp_path, monkeypatch):
    """R3: once the authority predicate says MANAGED, an unreadable/missing tx
    must never silently degrade to the non-managed full capture — it becomes
    the LOUD M0-missing fallback subject."""
    import ouroboros.tools.registry as registry

    repo, ctx, _tx = _managed_resolution_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(registry, "_authorized_managed_update_resolver", lambda _ctx: True)

    def _boom(*_a, **_k):
        raise RuntimeError("tx storage exploded")

    monkeypatch.setattr(update_merge, "authorized_assisted_task", _boom)
    subject = managed_review_subject(ctx, repo)
    assert subject is not None, "authorized resolver must never get a silent None"
    assert subject.fallback_full_diff is True
    assert subject.m0_missing_reason.startswith("tx_unreadable:")
    assert "tx storage exploded" in subject.m0_missing_reason
    rendered = subject.render_prompt_diff()
    assert "M0 BASELINE UNAVAILABLE" in rendered
    assert "tx_unreadable" in rendered

    # An EMPTY tx for an authorized resolver is the same loud fallback.
    monkeypatch.setattr(update_merge, "authorized_assisted_task", lambda *_a, **_k: {})
    empty_tx_subject = managed_review_subject(ctx, repo)
    assert empty_tx_subject is not None
    assert empty_tx_subject.fallback_full_diff is True
    assert empty_tx_subject.m0_missing_reason.startswith("tx_missing:")

    # Genuinely non-managed contexts still resolve to None (byte-identical path).
    monkeypatch.setattr(registry, "_authorized_managed_update_resolver", lambda _ctx: False)
    assert managed_review_subject(ctx, repo) is None


def test_managed_binary_deletion_is_rendered_with_m0_evidence(tmp_path, monkeypatch):
    """R4: the official target added an extensionless binary absent from HEAD;
    the resolver DELETES it. The deletion row must render against M0/parent
    evidence, and the binary probe must call the path binary."""
    from ouroboros.tools.review_binary_context import (
        render_staged_binary_metadata,
        staged_path_is_binary,
    )
    from ouroboros.tools.review_helpers import build_touched_file_pack

    repo, ctx, tx = _managed_resolution_repo(tmp_path, monkeypatch, official_binary=True)
    result = _git(repo, "rm", "-qf", "payload")  # the resolver deletes the official binary
    assert result.returncode == 0, result.stderr

    subject = managed_review_subject(ctx, repo)
    m0_tree, staged_tree = subject.m0_tree, subject.staged_tree
    assert ("D", "payload") in subject.name_status  # M0→S sees the deletion

    # Detection: binary in the reviewed M0→S delta.
    assert staged_path_is_binary(repo, "payload", m0_tree=m0_tree, staged_tree=staged_tree)
    # Rendering: a real deletion row with M0 evidence, not a silent None.
    metadata = render_staged_binary_metadata(repo, "payload", m0_tree=m0_tree)
    assert metadata is not None, "managed binary deletion must be represented"
    assert "binary deletion is represented" in metadata
    assert "staged blob: `absent (deletion)`" in metadata
    assert "mechanical merge M0 blob" in metadata and "absent" not in metadata.split(
        "mechanical merge M0 blob"
    )[1].splitlines()[0]
    # Triad touched pack renders the metadata row instead of an omission.
    pack, omitted = build_touched_file_pack(
        pathlib.Path(repo), ["payload"], represent_binary=True,
        m0_tree=m0_tree, staged_tree=staged_tree,
    )
    assert "payload" not in omitted
    assert "mechanical merge M0 blob" in pack
    # The non-managed probe stays byte-identical (HEAD-only, blind to this
    # topology — documented).
    assert not staged_path_is_binary(repo, "payload")


def test_withheld_seats_get_typed_records_on_wave_refusal(tmp_path, monkeypatch):
    """R5a: a PREPARED wave that the money admission refuses leaves every seat
    a typed $0 not_dispatched actor record (seat identity survives), not just a
    degraded-reason string — and the coupling outcome is typed the same way."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools import parallel_review as pr
    from ouroboros.tools import review as review_mod
    from ouroboros.tools import review_admission as admission

    repo = _plain_repo(tmp_path)
    ctx = _admission_ctx(repo)
    plan = _row_plan(["api", "session"])
    plan.update(retrieves=[False, True], parts=[("change",), ("change", "coupling")],
                routes=[ReviewRouteKind.API_CHAT, ReviewRouteKind.AGENT_SESSION])
    prepared = {"prompt": "p", "blocking_review": False, "row_plan": plan,
                "models": list(plan["models"]), "routes": list(plan["routes"])}
    monkeypatch.setattr(
        review_mod, "_prepare_unified_review", lambda *a, **k: (prepared, None, False)
    )
    monkeypatch.setattr(
        review_mod, "_dispatch_unified_review",
        lambda *a, **k: pytest.fail("the wave dispatched despite the refusal"),
    )
    monkeypatch.setattr(admission, "commit_gate_paid_seats", lambda *a, **k: [{"surface": "multi_model_review"}])
    monkeypatch.setattr(admission, "admit_commit_gate_wave",
                        lambda *a, **k: "⚠️ REVIEW_BLOCKED: commit-gate review wave declined before dispatch ($0 spent)")

    _err, coupling, reason, _adv = pr.run_parallel_review(ctx, "msg")

    assert reason == "review_wave_budget_insufficient"
    assert coupling is not None and coupling.status == "not_dispatched"
    records = ctx._last_triad_raw_results
    assert [r["model_id"] for r in records] == ["m/0-api", "m/1-session"]
    assert all(r["status"] == "not_dispatched" for r in records)
    assert [r["slot_id"] for r in records] == ["slot_0", "slot_1"]
    assert all(r["cost_usd"] == 0.0 and r["tokens_in"] == 0 for r in records)
    assert all("$0 spent" in r["raw_text"] for r in records)


def test_q28_dropped_api_seats_survive_into_raw_results(tmp_path, monkeypatch):
    """R5b: a Q28-A-dropped api seat keeps its identity as a typed $0
    not_dispatched record MERGED beside the dispatched panel's records."""
    review_mod, ctx = _triad_real_fit_env(
        tmp_path, monkeypatch, _row_plan(["api", "session", "session"])
    )

    prepared, early, exited = review_mod._prepare_unified_review(
        ctx, "resolve managed update conflicts"
    )
    assert not exited and early is None

    withheld = getattr(ctx, "_triad_withheld_seat_records", [])
    assert [r["model_id"] for r in withheld] == ["m/0-api"]
    assert withheld[0]["status"] == "not_dispatched"
    assert withheld[0]["slot"] == 1 and withheld[0]["slot_id"] == "slot_0"

    # The dispatched panel reports; the dropped seat's record must survive.
    model_results = [
        {"model": "m/1-session", "slot_id": "slot_1", "text": "[]"},
        {"model": "m/2-session", "slot_id": "slot_2", "text": "[]"},
    ]
    review_mod._collect_review_findings(ctx, model_results, prepared["row_plan"])
    statuses = {r["model_id"]: r["status"] for r in ctx._last_triad_raw_results}
    assert statuses["m/0-api"] == "not_dispatched"
    assert set(statuses) == {"m/0-api", "m/1-session", "m/2-session"}


def test_full_candidate_count_failure_renders_na(tmp_path, monkeypatch):
    """R8: a failed full-candidate count renders "n/a", never a fake 0."""
    import dataclasses

    import ouroboros.tools.review_subject as rs

    repo, ctx, _tx = _managed_resolution_repo(tmp_path, monkeypatch)
    assert rs._full_candidate_path_count(repo, "0" * 40) is None  # git failure

    subject = managed_review_subject(ctx, repo)
    broken = dataclasses.replace(subject, full_candidate_paths=None)
    assert "full candidate paths: n/a (count unavailable)" in broken.counters_line()
    assert "full candidate paths: n/a" in broken.header()
    assert "full candidate paths: 0" not in broken.header()


def test_review_status_message_names_subject_binding_mismatch():
    """R9: the review-status projection renders a friendly reason for
    review_subject_binding_mismatch instead of echoing the raw token."""
    from ouroboros.review_evidence import _review_status_message

    attempt = SimpleNamespace(status="blocked", block_reason="review_subject_binding_mismatch")
    message = _review_status_message({
        "selected_attempt": attempt,
        "effective_status": "stale",
        "open_debts": [],
    })
    assert "review_subject_binding_mismatch" in message  # the typed token stays
    assert "not the tree this commit would write" in message
    assert "Re-stage the intended candidate" in message


# ---------------------------------------------------------------------------
# Hardening round: C5 per-attempt subject memoization, C6 crashed-predicate
# marker probe.
# ---------------------------------------------------------------------------


def test_subject_is_built_once_per_key_and_reset_invalidates(tmp_path, monkeypatch):
    """C5: N consumers of one attempt (triad + scope rows + fit rungs) share
    ONE built subject per (repo, M0, S, surface) key — the memo hit returns
    the SAME object, the advisory surface is a separate key, and the
    per-attempt reset (clearing ``_managed_review_subject_memo``) forces a
    fresh build."""
    import ouroboros.tools.review_subject as review_subject_mod

    repo, ctx, tx = _managed_resolution_repo(tmp_path, monkeypatch)
    builds = {"count": 0}
    real_delta = review_subject_mod._tree_delta_diff

    def _counting_delta(*args, **kwargs):
        builds["count"] += 1
        return real_delta(*args, **kwargs)

    monkeypatch.setattr(review_subject_mod, "_tree_delta_diff", _counting_delta)

    first = managed_review_subject(ctx, repo)
    second = managed_review_subject(ctx, repo)
    third = managed_review_subject(ctx, repo)
    assert first is not None and first is second is third
    assert builds["count"] == 1

    advisory_first = managed_review_subject(ctx, repo, surface="advisory")
    advisory_second = managed_review_subject(ctx, repo, surface="advisory")
    assert advisory_first is advisory_second and advisory_first is not first
    assert builds["count"] == 2  # the advisory surface is its own key

    # The per-attempt reset boundary invalidates the memo.
    ctx._managed_review_subject_memo = {}
    fresh = managed_review_subject(ctx, repo)
    assert fresh is not first
    assert builds["count"] == 3
    # Content is identical across the rebuild: memoization never changed
    # anything a consumer sees.
    assert fresh.render_prompt_diff() == first.render_prompt_diff()


def test_crashed_predicate_with_present_tx_marker_fails_loud(tmp_path, monkeypatch):
    """C6: an exception ESCAPING the authority predicate (programming/import
    error) while the managed update tx MARKER exists must raise the typed
    StagedDiffUnavailable — never silently review an apparently-managed
    candidate as an ordinary staged diff."""
    import ouroboros.tools.registry as registry_mod
    from ouroboros.tools.review_binary_context import StagedDiffUnavailable

    repo, ctx, tx = _managed_resolution_repo(tmp_path, monkeypatch)
    assert update_merge._update_tx_marker_path().is_file()

    def _boom(_ctx):
        raise RuntimeError("predicate programming error")

    monkeypatch.setattr(registry_mod, "_authorized_managed_update_resolver", _boom)
    with pytest.raises(StagedDiffUnavailable):
        managed_review_subject(ctx, repo)


def test_crashed_predicate_without_marker_stays_non_managed(tmp_path, monkeypatch):
    """C6, the other branch: no tx marker → the crash is logged loudly and the
    caller stays on the ordinary staged-diff path (a non-managed commit must
    never be blocked by a managed-code bug)."""
    import ouroboros.tools.registry as registry_mod

    repo, ctx, tx = _managed_resolution_repo(tmp_path, monkeypatch)
    assert update_merge.clear_update_tx()
    assert not update_merge._update_tx_marker_path().is_file()

    def _boom(_ctx):
        raise RuntimeError("predicate programming error")

    monkeypatch.setattr(registry_mod, "_authorized_managed_update_resolver", _boom)
    assert managed_review_subject(ctx, repo) is None
