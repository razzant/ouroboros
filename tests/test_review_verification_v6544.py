"""Focused tests for the v6.54.4 review+verification commit (Commit 2 of the sprint).

Covers: budget_profile in the task contract (2.1), the task_pacing SSOT
(snapshot, reserve, emit window and the time/cost notices), criterion provenance,
the FINAL ANSWER latch and web answer_type (2.3). The acceptance launch floor and
improvement-pass gates (2.1), the review dissent layer and acceptance obligations
(2.2) and the CANDIDATES sub-latch (2.3) left with task acceptance (owner 17.10,
Q2-A).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from ouroboros import task_pacing
from ouroboros.contracts.task_contract import build_task_contract, normalize_budget_profile


def _deadline_ctx(remaining_sec: float, *, total_sec: float = 1000.0, profile=None):
    now = datetime.now(timezone.utc)
    created = now - timedelta(seconds=total_sec - remaining_sec)
    deadline = now + timedelta(seconds=remaining_sec)
    return SimpleNamespace(
        task_metadata={
            "created_at": created.isoformat(),
            "deadline_at": deadline.isoformat(),
        },
        task_contract={"budget_profile": normalize_budget_profile(profile or {})},
    )


# ---------------------------------------------------------------------------
# 2.1 — budget_profile contract


def test_normalize_budget_profile_defaults_are_conservative():
    out = normalize_budget_profile(None)
    assert out == {
        "improvement_policy": "fixed",
        "max_improvement_passes": None,
        "reserve_finalization_pct": None,
        "cost_hard_stop_pct": None,
    }
    out2 = normalize_budget_profile({
        "improvement_policy": "ADAPTIVE",
        "max_improvement_passes": "3",
        "reserve_finalization_pct": 150,
        "cost_hard_stop_pct": 150,
    })
    assert out2["improvement_policy"] == "adaptive"
    assert out2["max_improvement_passes"] == 3
    assert out2["reserve_finalization_pct"] == 100  # clamped
    assert out2["cost_hard_stop_pct"] == 100  # clamped
    # ABI 7.0 (Q10=A): retired spellings normalize AWAY - the alias policy
    # falls back to "fixed" and the stall knob is not part of the shape.
    gone = normalize_budget_profile(
        {"improvement_policy": "until_deadline", "stall_rounds_threshold": 12})
    assert gone["improvement_policy"] == "fixed"
    assert "stall_rounds_threshold" not in gone
    # 0 is a MEANINGFUL value (no in-task cost stop), preserved verbatim.
    assert normalize_budget_profile({"cost_hard_stop_pct": 0})["cost_hard_stop_pct"] == 0


def test_budget_profile_rides_the_contract_and_inherits():
    task = {
        "id": "t1", "type": "task", "description": "x",
        "budget_profile": {
            "improvement_policy": "adaptive",
            "max_improvement_passes": 2,
            "cost_hard_stop_pct": 0,
        },
    }
    contract = build_task_contract(task)
    assert contract["budget_profile"]["improvement_policy"] == "adaptive"
    # Child inheritance via the parent-contract spread (metadata.task_contract).
    child = build_task_contract({
        "id": "c1", "type": "task", "description": "y",
        "metadata": {"task_contract": {**contract, "objective": "y"}},
    })
    assert child["budget_profile"]["improvement_policy"] == "adaptive"
    assert child["budget_profile"]["max_improvement_passes"] == 2
    # The meaningful 0 (no in-task cost stop) survives inheritance verbatim.
    assert child["budget_profile"]["cost_hard_stop_pct"] == 0


# ---------------------------------------------------------------------------
# 2.1 — task_pacing snapshot and reserve (plan §193-200 time schema)


def test_snapshot_without_deadline_disables_time_axis():
    ctx = SimpleNamespace(task_metadata={}, task_contract={})
    snap = task_pacing.build_budget_snapshot(ctx)
    assert snap.has_deadline is False
    assert snap.spendable_sec == float("inf")
    assert snap.inside_reserve is False


def test_snapshot_marks_the_finalization_reserve(monkeypatch):
    """The snapshot's reserve facts (``delegate_supervision`` reports them): inside the
    120 s grace with 100 s left, above it with 400 s left. (The acceptance launch floor
    that once read them left with task acceptance.)"""
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    snap = task_pacing.build_budget_snapshot(_deadline_ctx(remaining_sec=100.0))
    assert snap.inside_reserve
    above = task_pacing.build_budget_snapshot(_deadline_ctx(remaining_sec=400.0))
    assert not above.inside_reserve


def test_reserve_uses_profile_pct_when_larger_than_grace(monkeypatch):
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "60")
    profile = normalize_budget_profile({"reserve_finalization_pct": 20})
    ctx = _deadline_ctx(remaining_sec=500.0, total_sec=1000.0, profile={"reserve_finalization_pct": 20})
    snap = task_pacing.build_budget_snapshot(ctx, profile=profile)
    assert snap.reserve_sec == pytest.approx(200.0, rel=0.05)  # 20% of 1000 > 60s grace


def test_finalize_emit_window_is_grace_not_pct_reserve(monkeypatch):
    """Adversarial r1 #6/#15: the deadline-finalize/tool-clamp emit window is the
    plain GRACE, NOT the pct reserve — else a long task's tail is amputated (a 6h
    task would self-finalize ~54 min early on a 15% profile). The pct reserve lives
    only on the snapshot."""
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    # A big-budget task on a 15% profile: snapshot reserve is large (pct), but the
    # emit window the finalize path uses must be the small grace (120s).
    ctx = _deadline_ctx(remaining_sec=3600.0, total_sec=21600.0,
                        profile={"reserve_finalization_pct": 15})
    snap = task_pacing.build_budget_snapshot(ctx)
    assert snap.reserve_sec == pytest.approx(3240.0, rel=0.05)  # 15% of 6h (snapshot reserve)
    assert task_pacing.effective_finalization_reserve_sec(ctx) == pytest.approx(120.0)  # emit window
    # No deadline -> still the grace window.
    assert task_pacing.effective_finalization_reserve_sec(
        SimpleNamespace(task_metadata={})) == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# 2.1 — milestone content moved to task_pacing (behavior parity)


def test_time_budget_note_fires_tightest_crossed_milestone():
    ctx = _deadline_ctx(remaining_sec=50.0, total_sec=1000.0)  # 5% remaining
    note = task_pacing.build_time_budget_note(ctx)
    assert note is not None
    assert note.checkpoint["milestone"] == "10%"
    assert "[TIME BUDGET — 10% remaining crossed]" in note.text
    assert "Deadline:" in note.text and "Remaining: ~0.8 min" in note.text
    # The host's own finalization instant is stated as a fact (owner 13 Oct: the
    # one-time crossing stays, the advice goes) — no save/flush instruction.
    assert "one tool-less final answer is requested when ~2.0 min remain" in note.text
    for advice in ("WRITE", "verify_and_record", "planning context", "shortest path", "prefer"):
        assert advice not in note.text, advice
    # All coarser milestones marked seen — nothing refires.
    assert task_pacing.build_time_budget_note(ctx) is None

    # The answer protocol no longer adds a FINAL ANSWER instruction here; the facts
    # line states whether a FINAL ANSWER line is latched.
    proto_ctx = _deadline_ctx(remaining_sec=50.0, total_sec=1000.0)
    proto_ctx.task_contract["answer_protocol"] = "final_answer_line"
    proto_note = task_pacing.build_time_budget_note(proto_ctx)
    assert proto_note is not None and "FINAL ANSWER" not in proto_note.text


def test_no_time_bound_means_no_time_notice(monkeypatch):
    """The 10-minute intrinsic [PACING] note is gone (#1748): without an explicit
    deadline or execution budget there is no time axis to cross."""
    now = datetime.now(timezone.utc)
    ctx = SimpleNamespace(
        task_metadata={"created_at": (now - timedelta(seconds=1300)).isoformat()},
        task_contract={},
    )
    assert task_pacing.build_time_budget_note(ctx) is None


def test_budget_snapshot_latches_fallback_created():
    """Fable-5 cumulative review F4: a metadata-poor task (deadline but no
    created_at/started_at) must LATCH its fallback anchor like the note path
    does, so consecutive snapshots keep a stable total instead of re-anchoring
    to "now" and silently degrading the pct reserve toward the grace floor."""
    now = datetime.now(timezone.utc)
    ctx = SimpleNamespace(
        task_metadata={"deadline_at": (now + timedelta(seconds=1000)).isoformat()},
        task_contract={},
    )
    snap1 = task_pacing.build_budget_snapshot(ctx)
    assert snap1.has_deadline
    assert getattr(ctx, "_time_budget_started_at", None) is not None
    snap2 = task_pacing.build_budget_snapshot(ctx)
    assert abs(snap2.total_sec - snap1.total_sec) < 1.0


# ---------------------------------------------------------------------------
# 2.3 — criterion provenance


def test_receipt_defaults_to_agent_defined(tmp_path, monkeypatch):
    from ouroboros.tools.verify import _verify_and_record

    ctx = SimpleNamespace(
        task_id="t-crit", drive_root=tmp_path, task_metadata={}, task_contract={},
        repo_dir=tmp_path / "repo",
    )
    monkeypatch.setattr("ouroboros.tools.verify._confine_cwd", lambda *a, **k: (str(tmp_path), ""), raising=False)
    _verify_and_record(
        ctx, contract_kind="artifact_observation",
        artifact_paths=[str(tmp_path / "missing.txt")],
    )
    receipts_path = tmp_path / "task_results" / "artifacts" / "t-crit" / "verification_receipts.jsonl"
    rows = [json.loads(line) for line in receipts_path.read_text().splitlines()]
    assert rows and rows[-1]["criterion_source"] == "agent_defined"


def test_latest_agent_defined_verification_helper(tmp_path):
    from ouroboros.outcomes import latest_agent_defined_verification
    from ouroboros.utils import append_jsonl

    receipts = tmp_path / "task_results" / "artifacts" / "t-x" / "verification_receipts.jsonl"
    receipts.parent.mkdir(parents=True)
    append_jsonl(receipts, {"status": "pass", "criterion_source": "task_stated"})
    assert latest_agent_defined_verification(tmp_path, "t-x") is None
    append_jsonl(receipts, {"status": "pass", "criterion_source": "agent_defined"})
    hit = latest_agent_defined_verification(tmp_path, "t-x")
    assert hit is not None and hit["criterion_source"] == "agent_defined"
    # A stated basis reconciles the nudge condition.
    append_jsonl(receipts, {"status": "pass", "criterion_source": "agent_defined", "criterion_basis": "task asks exactly this"})
    assert latest_agent_defined_verification(tmp_path, "t-x") is None


# ---------------------------------------------------------------------------
# 2.3 — FINAL ANSWER latch (its opt-in CANDIDATES sub-latch left with task acceptance)


def test_final_answer_latches_beside_a_candidates_block():
    from ouroboros.loop import _latch_final_answer_marker

    llm_trace: dict = {"tool_calls": []}
    content = (
        "Research complete.\n\nCANDIDATES:\n- 42 — counting only published papers\n"
        "- 45 — including preprints\n\nFINAL ANSWER: 42"
    )
    _latch_final_answer_marker(llm_trace, content)
    assert llm_trace["best_valid_final_answer"] == "42"
    assert "candidate_answers" not in llm_trace


def test_no_candidates_block_leaves_trace_unchanged():
    from ouroboros.loop import _latch_final_answer_marker

    llm_trace: dict = {"tool_calls": []}
    _latch_final_answer_marker(llm_trace, "FINAL ANSWER: 7")
    assert "candidate_answers" not in llm_trace
    assert llm_trace["best_valid_final_answer"] == "7"


def test_candidates_marker_only_ignores_inline_prose_mention():
    """Adversarial r2 #4: a mid-sentence 'CANDIDATES:' followed later by an
    ordinary markdown bullet list must NOT latch those bullets — the marker is
    line-anchored and the items must be adjacent to it."""
    from ouroboros.loop import _latch_final_answer_marker

    llm_trace: dict = {"tool_calls": []}
    content = (
        "I evaluated several CANDIDATES: see the comparison below.\n\n"
        "- source A is stale\n- source B contradicts it\n\nFINAL ANSWER: B"
    )
    _latch_final_answer_marker(llm_trace, content)
    assert llm_trace["best_valid_final_answer"] == "B"
    assert "candidate_answers" not in llm_trace


# ---------------------------------------------------------------------------
# 2.3 — web_search answer_type


def test_web_search_results_carry_answer_type(monkeypatch, tmp_path):
    import sys, types

    from ouroboros.tools import search as search_mod

    class _FakeStream:
        def __iter__(self):
            return iter([
                types.SimpleNamespace(type="response.output_text.delta", delta="fresh answer"),
                types.SimpleNamespace(
                    type="response.completed",
                    response=types.SimpleNamespace(output_text="fresh answer", output=[], usage=None),
                ),
            ])

    class _Responses:
        def create(self, **kwargs):
            if kwargs.get("stream"):
                return _FakeStream()
            return types.SimpleNamespace(output_text="fresh answer", output=[])

    class _Client:
        def __init__(self, api_key=None, base_url=None, **kwargs):
            self.responses = _Responses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    out = search_mod._web_search(SimpleNamespace(task_id="t", drive_root=tmp_path), query="q")
    data = json.loads(out)
    assert data.get("answer_type") == "summary"


def test_time_budget_flush_tree_sentence_workspace_gated():
    """v6.74.4 time-axis freeze, now a fact: the 10% crossing states that the working
    tree is delivered as it stands ONLY for workspace deliveries; non-workspace text
    stays byte-identical (cache/latch semantics untouched)."""
    fact = "the working tree is delivered as it stands"
    base_ctx = _deadline_ctx(remaining_sec=50.0, total_sec=1000.0)
    base_note = task_pacing.build_time_budget_note(base_ctx)
    assert base_note is not None and fact not in base_note.text

    ws_ctx = _deadline_ctx(remaining_sec=50.0, total_sec=1000.0)
    ws_ctx.workspace_root = "/tmp/ws"  # fallback attribute path
    ws_note = task_pacing.build_time_budget_note(ws_ctx)
    assert ws_note is not None and fact in ws_note.text
    # Commit-neutral on purpose: acting self_worktree subagents cannot commit; and
    # no instruction rides the fact any more.
    assert "commit" not in ws_note.text.lower() and "revert" not in ws_note.text
    # The canonical authority wins over the raw attribute when present.
    canon_ctx = _deadline_ctx(remaining_sec=50.0, total_sec=1000.0)
    canon_ctx.workspace_root = "/tmp/ws"
    canon_ctx.is_workspace_mode = lambda: False
    canon_note = task_pacing.build_time_budget_note(canon_ctx)
    assert canon_note is not None and fact not in canon_note.text


def test_time_budget_flush_non_workspace_text_fully_identical(monkeypatch):
    """codex finding 4 / commit triad r4: TRUE byte-equality of the WHOLE
    non-workspace note — clock frozen, both contexts built from identical
    deadline metadata, full-text comparison with no stripping."""
    frozen = datetime.now(timezone.utc)
    monkeypatch.setattr(task_pacing, "utc_now", lambda: frozen)
    created = (frozen - timedelta(seconds=950)).isoformat()
    deadline = (frozen + timedelta(seconds=50)).isoformat()

    def _ctx():
        return SimpleNamespace(
            task_metadata={"created_at": created, "deadline_at": deadline},
            task_contract={"budget_profile": normalize_budget_profile({})},
        )

    plain = task_pacing.build_time_budget_note(_ctx())
    denied_ctx = _ctx()
    denied_ctx.workspace_root = "/tmp/ws"
    denied_ctx.is_workspace_mode = lambda: False
    denied = task_pacing.build_time_budget_note(denied_ctx)
    assert plain is not None and denied is not None
    assert plain.text == denied.text


def test_cost_wrapup_tree_sentence_workspace_gated(monkeypatch):
    """v6.74.4 (commit triad r1 advisory, sol): the ~80% spent crossing — a
    sibling forced-end channel — carries the same commit-neutral delivery fact
    for workspace deliveries only; non-workspace text stays byte-identical."""

    def _note(with_ws):
        ctx = SimpleNamespace(
            task_metadata={}, task_contract={},
            _cost_budget_milestones_seen={"50%", "25%", "10%"},
            _cost_wrapup_seen=False,
        )
        if with_ws:
            ctx.workspace_root = "/tmp/ws"
        return task_pacing.build_cost_budget_note(ctx, cost_ceiling_usd=10.0, task_cost=8.5)

    plain = _note(False)
    ws = _note(True)
    assert plain is not None and ws is not None
    assert "working tree is delivered" not in plain.text
    assert "working tree is delivered" in ws.text
    assert ws.text.replace(task_pacing._WORKSPACE_DELIVERY_FACT, "") == plain.text


def test_cost_late_first_milestone_carries_tree_sentence_for_workspace():
    """v6.74.4 (commit triad r2, sol advisory): a workspace task whose FIRST
    cost milestone fires already past the wrap-up fraction must still see the
    tree sentence (the suppressed wrap-up was its only carrier)."""
    ctx = SimpleNamespace(task_metadata={}, task_contract={},
                          workspace_root="/tmp/ws")
    note = task_pacing.build_cost_budget_note(ctx, cost_ceiling_usd=10.0, task_cost=9.5)
    assert note is not None
    assert note.checkpoint["checkpoint_kind"] == "cost_budget_milestone"
    assert "working tree is delivered" in note.text
    # Non-workspace late milestone stays byte-identical (no delivery fact).
    plain_ctx = SimpleNamespace(task_metadata={}, task_contract={})
    plain = task_pacing.build_cost_budget_note(plain_ctx, cost_ceiling_usd=10.0, task_cost=9.5)
    assert plain is not None and "working tree is delivered" not in plain.text
    assert note.text.replace(task_pacing._WORKSPACE_DELIVERY_FACT, "") == plain.text
    # An EARLY milestone (before the ~80% fraction) is unchanged even for
    # workspace tasks — the ~80% crossing will carry the fact later.
    early_ctx = SimpleNamespace(task_metadata={}, task_contract={},
                                workspace_root="/tmp/ws")
    early = task_pacing.build_cost_budget_note(early_ctx, cost_ceiling_usd=10.0, task_cost=6.0)
    assert early is not None and "working tree is delivered" not in early.text
