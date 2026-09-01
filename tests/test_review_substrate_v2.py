import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

from ouroboros.review_substrate import ReviewRequest, ReviewSlot, _render_prompt, run_review_request
from ouroboros.triad_review import parse_model_review_results


def test_render_prompt_requires_outcome_tier_and_independence():
    """T1 (v6.35.0): for task acceptance, outcome_tier/completion_coach are part of
    the REQUIRED JSON keys (not trailing prose models drop), and the reviewer is
    told to judge evidence independence + environment-vs-deliverable."""
    req = ReviewRequest(
        surface="task_acceptance",
        goal="verify",
        subject="done",
        policy={"classify_outcome_tier": True},
        task_id="t",
    )
    prompt = _render_prompt(req, ReviewSlot(slot_id="a", model="m"))
    keys_line = next(line for line in prompt.splitlines() if line.startswith("Return JSON with keys:"))
    assert "outcome_tier" in keys_line and "completion_coach" in keys_line
    assert "EVIDENCE INDEPENDENCE" in prompt
    assert "ENVIRONMENT vs DELIVERABLE" in prompt
    assert "ABSENT-PREMISE / INFEASIBLE DISPOSITION" in prompt
    assert "PREMISE ARGUMENT, not the named artifact" in prompt
    assert "FULL goal/spec narrative" in prompt
    assert "affected components/surfaces" in prompt
    assert "per-criterion evidence" in prompt
    assert "VISIBLE UI EVIDENCE" in prompt
    assert "real consumer flow" in prompt
    assert "screenshot file or attachment" in prompt
    assert "mobile and WebKit are not universal requirements" in prompt
    assert "unavailable optional engine alone is not degradation" in prompt

    # A non-tier surface keeps the lean key list (no tier keys).
    plain = _render_prompt(
        ReviewRequest(surface="scope", goal="g", task_id="t"),
        ReviewSlot(slot_id="a", model="m"),
    )
    plain_keys = next(line for line in plain.splitlines() if line.startswith("Return JSON with keys:"))
    assert "outcome_tier" not in plain_keys
    assert "VISIBLE UI EVIDENCE" not in plain


class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        body = {
            "verdict": "PASS",
            "findings": [],
            "summary": f"reviewed by {kwargs['model']}",
        }
        return {"content": json.dumps(body)}, {"prompt_tokens": 10, "completion_tokens": 5}


def test_review_slot_passes_explicit_local_transport_to_llm(tmp_path):
    llm = FakeLLM()
    result = run_review_request(
        ReviewRequest(surface="scope", goal="review locally", task_id="local-review"),
        slots=[ReviewSlot(slot_id="local", model="owner/local-main", use_local=True)],
        drive_root=tmp_path,
        llm=llm,
    )

    assert result.actors
    assert llm.calls and llm.calls[0]["use_local"] is True


class FencedArrayLLM:
    def chat(self, **kwargs):
        body = (
            "Here is the review:\n"
            "```json\n"
            "[{\"verdict\":\"FAIL\",\"severity\":\"critical\",\"item\":\"x\",\"evidence\":\"e\",\"recommendation\":\"r\"}]\n"
            "```"
        )
        return {"content": body}, {"prompt_tokens": 10, "completion_tokens": 5}


class FencedObjectLLM:
    def chat(self, **kwargs):
        body = (
            "Verdict below:\n"
            "```json\n"
            "{\"verdict\":\"PASS\",\"findings\":[]}\n"
            "```"
        )
        return {"content": body}, {"prompt_tokens": 10, "completion_tokens": 5}


class ConcernsLLM:
    def chat(self, **kwargs):
        # Valid JSON, transport ok, but a non-PASS/FAIL/DEGRADED verdict.
        return {"content": "{\"verdict\":\"CONCERNS\",\"findings\":[]}"}, {"prompt_tokens": 10, "completion_tokens": 5}


class ParseDegradedSlotLLM:
    """Two slots PASS; the '-2' slot returns a successful but DEGRADED-verdict
    response (a reviewer doubt, NOT a transport/participation fault)."""

    def chat(self, **kwargs):
        if str(kwargs.get("model", "")).endswith("-2"):
            return {"content": json.dumps({"verdict": "DEGRADED", "findings": [], "summary": "unsure"})}, {}
        return {"content": json.dumps({"verdict": "PASS", "findings": [], "summary": "ok"})}, {}


class ActorErrorSlotLLM:
    """Two slots PASS; the '-2' slot raises (a participation fault)."""

    def chat(self, **kwargs):
        if str(kwargs.get("model", "")).endswith("-2"):
            raise RuntimeError("provider exploded")
        return {"content": json.dumps({"verdict": "PASS", "findings": [], "summary": "ok"})}, {}


def test_degraded_or_errored_acceptance_slot_abstains_when_quorum_remains(tmp_path):
    """T1 (v6.35.0): a single unparseable/DEGRADED-verdict slot must NOT poison a
    clean 2-of-3 PASS quorum. A participation fault also abstains on task
    acceptance when the configured PASS quorum remains; no-quorum is DEGRADED."""
    slots = [ReviewSlot(slot_id=f"s{i}", model=f"m-{i}") for i in range(3)]

    def _req():
        return ReviewRequest(
            surface="task_acceptance", goal="g", subject="done",
            policy={"fail_closed_on_errors": True, "min_successful_slots": 2}, task_id="t",
        )

    ok = run_review_request(_req(), slots=slots, drive_root=tmp_path, llm=ParseDegradedSlotLLM())
    assert ok.aggregate_signal == "PASS"
    assert ok.degraded is False

    bad = run_review_request(_req(), slots=slots, drive_root=tmp_path, llm=ActorErrorSlotLLM())
    assert bad.aggregate_signal == "PASS"
    assert bad.degraded is False


class PassNoTierLLM:
    """PASS verdict but NO outcome_tier — the non-compliant reviewer the required-
    tier contract must catch (a tier-less PASS must not aggregate to a clean PASS)."""

    def chat(self, **kwargs):
        return {"content": json.dumps({"verdict": "PASS", "findings": [], "summary": "ok"})}, {}


class PassWithTierLLM:
    def chat(self, **kwargs):
        body = {"verdict": "PASS", "outcome_tier": "solved", "completion_coach": "ship",
                "criteria_used": [{"criterion": "verified", "status": "supported", "evidence_refs": ["verification_summary"]}], "findings": [], "summary": "ok"}
        return {"content": json.dumps(body)}, {}


class PassTierNoCoachLLM:
    """PASS with a valid outcome_tier but EMPTY completion_coach — still
    non-responsive to the required-tier contract (both keys required)."""

    def chat(self, **kwargs):
        body = {"verdict": "PASS", "outcome_tier": "solved", "completion_coach": "",
                "criteria_used": [{"criterion": "verified", "status": "supported", "evidence_refs": ["verification_summary"]}], "findings": [], "summary": "ok"}
        return {"content": json.dumps(body)}, {}


class PoisonDegradedSlotLLM:
    """Two slots PASS+solved; the '-2' slot returns a DEGRADED verdict carrying a
    BLOCKED outcome_tier + a critical finding — a parse-degraded actor that must
    NOT poison the clean quorum PASS capsule (v6.36.0 review finding)."""

    def chat(self, **kwargs):
        if str(kwargs.get("model", "")).endswith("-2"):
            return {"content": json.dumps({
                "verdict": "DEGRADED",
                "outcome_tier": "blocked_with_evidence",
                "completion_coach": "STOP everything",
                "findings": [{"verdict": "FAIL", "severity": "critical",
                              "item": "poison", "recommendation": "do not ship this"}],
                "summary": "unsure",
            })}, {}
        return {"content": json.dumps({
            "verdict": "PASS", "outcome_tier": "solved", "completion_coach": "ship it",
            "criteria_used": [{"criterion": "verified", "status": "supported",
                               "evidence_refs": ["verification_summary"]}],
            "findings": [], "summary": "ok",
        })}, {}


def test_degraded_actor_does_not_poison_acceptance_capsule(tmp_path):
    """v6.36.0 (scope review finding): aggregate_outcome_tier / build_improvement_
    capsule must draw tier/coach/findings ONLY from actors that contributed to the
    aggregate verdict — a single parse-degraded slot carrying a BLOCKED tier must
    not inject a blocking improvement note into an otherwise-clean quorum PASS.
    v6.55.0 (codex/fable-5 cumulative review): a DELIBERATE minority DEGRADED
    verdict carrying a concrete recommendation now surfaces as ONE labeled
    non-veto [DISSENT] line (the GAIA 3cef3a44 class) — while the mainline
    capsule (tier / coach / bullets) stays unpoisoned exactly as before."""
    from ouroboros.review_substrate import aggregate_outcome_tier, build_improvement_capsule
    slots = [ReviewSlot(slot_id=f"s{i}", model=f"m-{i}") for i in range(3)]
    req = ReviewRequest(
        surface="task_acceptance", goal="g", subject="done",
        policy={"classify_outcome_tier": True, "min_successful_slots": 2}, task_id="t",
    )
    res = run_review_request(req, slots=slots, drive_root=tmp_path, llm=PoisonDegradedSlotLLM())
    assert res.aggregate_signal == "PASS"
    # The degraded '-2' slot's BLOCKED tier / coach must NOT surface.
    assert aggregate_outcome_tier(res) == "solved"
    capsule = build_improvement_capsule(res)
    assert "STOP everything" not in capsule
    assert "blocked" not in capsule.lower()
    # ...but its deliberate DEGRADED verdict + concrete recommendation IS the
    # dissent class: one labeled line, never a mainline bullet.
    assert "[DISSENT — s2 said DEGRADED]" in capsule
    assert "do not ship this" in capsule
    assert "- do not ship this" not in capsule


class ContractDegradedPassLLM:
    """Two slots PASS+solved+coach (contract-valid); the '-2' slot returns
    verdict=PASS but a BLOCKED outcome_tier with an EMPTY completion_coach — a
    CONTRACT-DEGRADED PASS (non-responsive to the required-tier contract). It must
    not contribute its blocked tier / finding to the clean quorum capsule
    (v6.36.0 round-2 scope finding: the live PASS-but-contract-degraded path)."""

    def chat(self, **kwargs):
        if str(kwargs.get("model", "")).endswith("-2"):
            return {"content": json.dumps({
                "verdict": "PASS", "outcome_tier": "blocked_with_evidence", "completion_coach": "",
                "findings": [{"verdict": "FAIL", "severity": "critical",
                              "item": "poison2", "recommendation": "block this hard"}],
                "summary": "x",
            })}, {}
        return {"content": json.dumps({
            "verdict": "PASS", "outcome_tier": "solved", "completion_coach": "ship",
            "criteria_used": [{"criterion": "verified", "status": "supported",
                               "evidence_refs": ["verification_summary"]}],
            "findings": [], "summary": "ok",
        })}, {}


def test_contract_degraded_pass_does_not_poison_capsule(tmp_path):
    """v6.36.0 round-2 scope finding: a verdict=PASS actor that VIOLATES the
    required tier/coach contract is demoted to non-contributing (signal->DEGRADED),
    so it can't feed its blocked tier / finding into the clean quorum PASS capsule —
    the live path the DEGRADED-verdict-only test did not cover."""
    from ouroboros.review_substrate import (
        aggregate_outcome_tier,
        build_improvement_capsule,
        compact_review_projection,
    )
    slots = [ReviewSlot(slot_id=f"s{i}", model=f"m-{i}") for i in range(3)]
    req = ReviewRequest(
        surface="task_acceptance", goal="g", subject="done",
        policy={"classify_outcome_tier": True, "min_successful_slots": 2}, task_id="t",
    )
    res = run_review_request(req, slots=slots, drive_root=tmp_path, llm=ContractDegradedPassLLM())
    assert res.aggregate_signal == "PASS"          # the two contract-valid solved PASS reach quorum
    assert aggregate_outcome_tier(res) == "solved"  # the blocked contract-degraded PASS is excluded
    malformed = next(actor for actor in res.actors if actor["slot_id"] == "s2")
    assert malformed["transport_status"] == "success"
    assert malformed["parse_status"] == "malformed"
    assert malformed["semantic_verdict"] == ""
    assert malformed["quorum_contribution"] is False
    assert "violated the required" in malformed["reason"]
    projected = compact_review_projection([dict(res.__dict__)])["panels"][0]
    projected_malformed = next(
        actor for actor in projected["actors"] if actor["slot_id"] == "s2"
    )
    assert projected_malformed["parse_status"] == "malformed"
    assert projected_malformed["semantic_verdict"] == ""
    capsule = build_improvement_capsule(res)
    assert "block this hard" not in capsule
    assert "blocked" not in capsule.lower()


def test_solved_pass_with_required_coach_does_not_force_reloop(tmp_path):
    """v6.36.0 round-2 cross-module finding: a contract-valid SOLVED review carries
    a required completion_coach, but a coach ALONE must not force a revise round —
    build_improvement_capsule returns '' for a solved/no-findings result."""
    from ouroboros.review_substrate import build_improvement_capsule
    slots = [ReviewSlot(slot_id=f"s{i}", model=f"m-{i}") for i in range(3)]
    req = ReviewRequest(
        surface="task_acceptance", goal="g", subject="done",
        policy={"classify_outcome_tier": True, "min_successful_slots": 2}, task_id="t",
    )
    res = run_review_request(req, slots=slots, drive_root=tmp_path, llm=PassWithTierLLM())
    assert res.aggregate_signal == "PASS"
    assert build_improvement_capsule(res) == ""  # solved + coach, no findings -> finalize, no re-loop


def test_single_configured_reviewer_marks_no_diversity(tmp_path):
    """v6.36.0 (Bible P3, centralized): a one-slot review through the coordinator
    is honored but records single_reviewer_no_diversity durably on EVERY surface —
    so a one-slot acceptance review can never quietly look like an ordinary
    multi-reviewer PASS. v6.74.0 (A6): the note is the TYPED FIELD (an orthogonal
    label projected on the panel), no longer a degraded_reason — the panel reason
    must name the real blocker, not lead with a diversity footnote."""
    one = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="g", subject="d",
                      policy={"min_successful_slots": 1}, task_id="t"),
        slots=[ReviewSlot(slot_id="s0", model="m-0")],
        drive_root=tmp_path, llm=PassWithTierLLM(),
    )
    assert one.single_reviewer_no_diversity is True
    assert "single_reviewer_no_diversity" not in one.degraded_reasons
    from ouroboros.review_substrate import compact_review_projection

    projection = compact_review_projection([
        {**one.__dict__, "authority": "host_root"},
    ])
    assert projection["panels"][0]["single_reviewer_no_diversity"] is True

    three = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="g", subject="d",
                      policy={"min_successful_slots": 2}, task_id="t"),
        slots=[ReviewSlot(slot_id=f"s{i}", model=f"m-{i}") for i in range(3)],
        drive_root=tmp_path, llm=PassWithTierLLM(),
    )
    assert three.single_reviewer_no_diversity is False
    assert "single_reviewer_no_diversity" not in three.degraded_reasons


def test_required_outcome_tier_is_enforced_at_quorum(tmp_path):
    """T1 (v6.35.0): with classify_outcome_tier policy, a PASS WITHOUT a valid
    outcome_tier cannot count toward a clean quorum — the required-tier contract
    is enforced at the parser/quorum level, not just asked for in the prompt.

    v6.46.0 (Q7): on the ADVISORY task-acceptance surface, a SOLVED deliverable has
    no tier-up step, so an empty completion_coach must NOT demote a solved PASS to
    DEGRADED. A tier-LESS PASS is still non-responsive."""
    slots = [ReviewSlot(slot_id=f"s{i}", model=f"m-{i}") for i in range(3)]

    def _req():
        return ReviewRequest(
            surface="task_acceptance", goal="g", subject="done",
            policy={"classify_outcome_tier": True, "min_successful_slots": 2}, task_id="t",
        )

    no_tier = run_review_request(_req(), slots=slots, drive_root=tmp_path, llm=PassNoTierLLM())
    assert no_tier.aggregate_signal == "DEGRADED"  # tier-less PASS is still non-responsive

    # Advisory carve-out: a SOLVED PASS without a coach is RESPONSIVE (nothing to improve).
    no_coach = run_review_request(_req(), slots=slots, drive_root=tmp_path, llm=PassTierNoCoachLLM())
    assert no_coach.aggregate_signal == "PASS"

    with_tier = run_review_request(_req(), slots=slots, drive_root=tmp_path, llm=PassWithTierLLM())
    assert with_tier.aggregate_signal == "PASS"


def test_p3_surfaces_ignore_task_acceptance_tier_policy(tmp_path):
    """Defense in depth: even if a caller accidentally carries the task-only
    classify_outcome_tier flag, commit/scope FAILs remain authoritative vetoes."""
    for surface in ("multi_model_review", "scope_review"):
        result = run_review_request(
            ReviewRequest(
                surface=surface,
                goal="review",
                subject="diff",
                policy={"classify_outcome_tier": True, "min_successful_slots": 1},
                task_id=f"t-{surface}",
            ),
            slots=[ReviewSlot(slot_id="s0", model="m-0")],
            drive_root=tmp_path,
            llm=FencedArrayLLM(),
        )
        assert result.aggregate_signal == "FAIL"
        assert result.actors[0]["signal"] == "FAIL"


def test_collect_turn_diff_surfaces_tracked_and_untracked(tmp_path):
    """T1 (v6.35.0): collect_turn_diff must surface BOTH tracked modifications and
    untracked NEW files (a self-authored test the agent just wrote) so the
    reviewer can judge evidence independence."""
    import subprocess as sp
    from types import SimpleNamespace as NS

    from ouroboros.review_evidence import collect_turn_diff

    repo = tmp_path / "r"
    repo.mkdir()
    sp.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "src.py").write_text("x = 1\n", encoding="utf-8")
    sp.run(["git", "add", "src.py"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "i"],
           cwd=repo, check=True, capture_output=True)
    (repo / "src.py").write_text("x = 2\n", encoding="utf-8")            # tracked mod
    (repo / "test_new.py").write_text("def test_x(): pass\n", encoding="utf-8")  # untracked new

    diff = collect_turn_diff(NS(repo_dir=repo))
    assert "src.py" in diff
    assert "test_new.py" in diff  # the untracked self-authored test is visible


def test_collect_turn_diff_untracked_survives_large_tracked_diff(tmp_path):
    """T1 round-2 fix: a large tracked diff must NOT clip away the untracked
    new-file names (independent truncation)."""
    import subprocess as sp
    from types import SimpleNamespace as NS

    from ouroboros.review_evidence import collect_turn_diff

    repo = tmp_path / "r"
    repo.mkdir()
    sp.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "big.py").write_text("x = 0\n", encoding="utf-8")
    sp.run(["git", "add", "big.py"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "i"],
           cwd=repo, check=True, capture_output=True)
    # >20000-char tracked modification, plus an untracked self-authored test.
    (repo / "big.py").write_text("\n".join(f"v{i} = {i}" for i in range(5000)), encoding="utf-8")
    (repo / "test_self.py").write_text("def test_self(): assert True\n", encoding="utf-8")

    diff = collect_turn_diff(NS(repo_dir=repo))
    assert "test_self.py" in diff  # untracked name survives despite the huge tracked diff
    assert "Untracked working-tree files" in diff


def test_acceptance_review_evidence_diff_is_host_owned(monkeypatch, tmp_path):
    """T1 (v6.35.0): the host-collected repo_diff must override any agent-supplied
    repo_diff so the EVIDENCE-INDEPENDENCE judgment can't be steered by a stale
    value passed through the public task_acceptance_review tool."""
    from types import SimpleNamespace as NS

    import ouroboros.review_evidence as re_mod
    import ouroboros.review_substrate as rs
    from ouroboros.tools.review import _handle_task_acceptance_review

    captured = {}

    monkeypatch.setattr(re_mod, "collect_turn_diff", lambda ctx, **kw: "HOST_DIFF_REAL")

    def _fake_run(request, **kwargs):
        captured["evidence"] = dict(request.evidence)
        return NS(aggregate_signal="PASS")

    monkeypatch.setattr(rs, "run_review_request", _fake_run)
    monkeypatch.setattr(rs, "reviewer_slots", lambda **k: [ReviewSlot(slot_id="a", model="m")])

    ctx = NS(
        drive_root=str(tmp_path), task_id="t",
        task_metadata={"root_task_id": "root", "parent_task_id": "root"},
    )
    _handle_task_acceptance_review(ctx, claim="done", evidence={"repo_diff": "STALE_AGENT_DIFF"})

    # v6.51.0: host repo_diff stays host-owned; the agent value is demoted (not promoted) under
    # the clearly-tagged `agent_supplied` block (was a top-level key pre-v6.51.0).
    assert captured["evidence"]["repo_diff"] == "HOST_DIFF_REAL"
    assert captured["evidence"]["agent_supplied"]["agent_supplied_repo_diff"] == "STALE_AGENT_DIFF"


def test_acceptance_review_empty_host_diff_does_not_fall_back_to_agent(monkeypatch, tmp_path):
    """T1 (v6.35.0): an EMPTY host diff is a valid fact (clean repo), not a reason
    to promote the agent-supplied diff to host-fact status — else the agent could
    steer EVIDENCE-INDEPENDENCE simply by acting when the host diff is empty."""
    from types import SimpleNamespace as NS

    import ouroboros.review_evidence as re_mod
    import ouroboros.review_substrate as rs
    from ouroboros.tools.review import _handle_task_acceptance_review

    captured = {}
    monkeypatch.setattr(re_mod, "collect_turn_diff", lambda ctx, **kw: "")

    def _fake_run(request, **kwargs):
        captured["evidence"] = dict(request.evidence)
        return NS(aggregate_signal="PASS")

    monkeypatch.setattr(rs, "run_review_request", _fake_run)
    monkeypatch.setattr(rs, "reviewer_slots", lambda **k: [ReviewSlot(slot_id="a", model="m")])

    ctx = NS(
        drive_root=str(tmp_path), task_id="t",
        task_metadata={"root_task_id": "root", "parent_task_id": "root"},
    )
    _handle_task_acceptance_review(ctx, claim="done", evidence={"repo_diff": "FABRICATED_AGENT_DIFF"})

    # repo_diff stays the (empty) host fact; the agent value is only the demoted, tagged key
    # under `agent_supplied` (v6.51.0 relocation — was top-level).
    assert captured["evidence"]["repo_diff"] == ""
    assert captured["evidence"]["agent_supplied"]["agent_supplied_repo_diff"] == "FABRICATED_AGENT_DIFF"


def test_acceptance_review_records_agent_disposition(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS

    import ouroboros.review_evidence as re_mod
    import ouroboros.review_substrate as rs
    from ouroboros.tools.review import _handle_task_acceptance_review

    captured = {}
    monkeypatch.setattr(re_mod, "collect_turn_diff", lambda ctx, **kw: "")

    def _fake_run(request, **kwargs):
        captured["evidence"] = dict(request.evidence)
        return NS(aggregate_signal="PASS", actors=[], parsed_findings=[])

    monkeypatch.setattr(rs, "run_review_request", _fake_run)
    monkeypatch.setattr(rs, "reviewer_slots", lambda **k: [ReviewSlot(slot_id="a", model="m")])
    monkeypatch.setattr(rs, "build_improvement_capsule", lambda _result: "")

    ctx = NS(
        drive_root=str(tmp_path), drive_logs=lambda: tmp_path / "logs", task_id="t",
        task_metadata={"root_task_id": "root", "parent_task_id": "root"},
    )
    raw = _handle_task_acceptance_review(
        ctx,
        claim="done",
        agent_disposition="rejected",
        rationale="Reviewer asked for a benchmark-specific workaround; I reject it as scope drift.",
    )
    payload = json.loads(raw)

    assert captured["evidence"]["agent_supplied"]["agent_decision"]["disposition"] == "rejected"
    assert payload["agent_decision"]["disposition"] == "rejected"
    assert "scope drift" in payload["agent_decision"]["rationale"]
    event = json.loads((tmp_path / "logs" / "events.jsonl").read_text().strip())
    assert event["type"] == "deprecated_task_acceptance_alias"
    assert event["aliases"] == ["agent_disposition"]
    assert event["removal"] == "next_major"


def test_root_acceptance_tool_defers_to_host_without_model_calls(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS

    import ouroboros.review_substrate as rs
    from ouroboros.tools.review import _handle_task_acceptance_review

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    monkeypatch.setattr(
        rs,
        "run_review_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model review must not run")),
    )
    monkeypatch.setattr(
        rs,
        "reviewer_slots",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("review slots must not resolve")),
    )
    ctx = NS(
        drive_root=str(tmp_path),
        task_id="root",
        root_task_id="root",
        task_metadata={"root_task_id": "root"},
        task_contract={},
    )

    first = json.loads(_handle_task_acceptance_review(
        ctx,
        claim="complete",
        goal="ship the result",
        checklist="tests pass",
        evidence={"verification_receipt": "receipt-1"},
    ))
    second = json.loads(_handle_task_acceptance_review(
        ctx,
        claim="complete",
        goal="ship the result",
        checklist="tests pass",
        evidence={"verification_receipt": "receipt-1"},
    ))
    changed_claim = json.loads(_handle_task_acceptance_review(
        ctx,
        claim="complete with a documented limitation",
        goal="ship the result",
        checklist="tests pass and limitation is disclosed",
        evidence={"verification_receipt": "receipt-1"},
    ))

    assert first["status"] == "deferred_to_host_acceptance"
    assert first["authoritative"] is False
    assert first["request"]["checklist"] == "tests pass"
    assert len(first["evidence_revision"]) == 64
    assert second["evidence_revision"] == first["evidence_revision"]
    assert changed_claim["evidence_revision"] != first["evidence_revision"]


def test_typed_retry_root_defers_self_review_and_is_host_eligible(
    monkeypatch, tmp_path,
):
    from types import SimpleNamespace as NS

    import ouroboros.loop as loop_mod
    import ouroboros.review_substrate as rs
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.review import _handle_task_acceptance_review

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    monkeypatch.setattr(
        rs,
        "run_review_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("normalized root self-call must not run a model review")
        ),
    )
    monkeypatch.setattr(
        rs,
        "reviewer_slots",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("normalized root self-call must not resolve review slots")
        ),
    )
    retry_id = "retry-root"
    prior_attempt_id = "logical-root"
    metadata = {
        "task_id": retry_id,
        "root_task_id": prior_attempt_id,
        "parent_task_id": "",
        "delegation_role": "root",
        "original_task_id": prior_attempt_id,
        "timeout_retry_from": prior_attempt_id,
    }
    tool_ctx = NS(
        drive_root=str(tmp_path),
        task_id=retry_id,
        task_metadata=metadata,
        task_contract={},
    )

    payload = json.loads(
        _handle_task_acceptance_review(tool_ctx, claim="retry complete")
    )
    assert payload["status"] == "deferred_to_host_acceptance"

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = retry_id
    registry._ctx.task_metadata = metadata
    registry._ctx.task_contract = {}
    seen = {}
    real_eligible = loop_mod._task_acceptance_eligible

    def capture_eligible(mode, trace, direct, **kwargs):
        result = real_eligible(mode, trace, direct, **kwargs)
        seen.update(is_root_task=kwargs["is_root_task"], result=result)
        return result

    monkeypatch.setattr(loop_mod, "_task_acceptance_eligible", capture_eligible)
    monkeypatch.setattr(
        loop_mod,
        "_begin_task_acceptance_fence",
        lambda *_args, **_kwargs: (False, None),
    )
    assert loop_mod._run_task_acceptance_review_once(
        tools=registry,
        content="retry complete",
        task_id=retry_id,
        task_type="task",
        llm_trace={"tool_calls": []},
        drive_root=tmp_path,
        messages=[],
        emit_progress=lambda _message: None,
    ) is True
    assert seen == {
        "is_root_task": True,
        "result": (True, "auto_nondirect"),
    }


def test_retry_root_markers_must_agree_before_acceptance_authority(
    monkeypatch, tmp_path,
):
    from types import SimpleNamespace as NS

    import ouroboros.review_substrate as rs
    from ouroboros.tools.review import _handle_task_acceptance_review

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    calls = []
    monkeypatch.setattr(
        rs,
        "reviewer_slots",
        lambda **kwargs: [ReviewSlot(slot_id="legacy", model="m")],
    )
    monkeypatch.setattr(rs, "build_improvement_capsule", lambda _result: "")
    monkeypatch.setattr(rs, "dissent_findings", lambda _result: [])

    def fake_run(request, **kwargs):
        calls.append(request.task_id)
        return NS(aggregate_signal="PASS", actors=[], parsed_findings=[])

    monkeypatch.setattr(rs, "run_review_request", fake_run)
    metadata = {
        "root_task_id": "logical-root",
        "parent_task_id": "",
        "delegation_role": "root",
        "original_task_id": "prior-a",
        "timeout_retry_from": "prior-b",
    }
    payload = json.loads(_handle_task_acceptance_review(
        NS(
            drive_root=str(tmp_path),
            task_id="malformed-retry",
            task_metadata=metadata,
            task_contract={},
        ),
        claim="done",
    ))

    assert payload["aggregate_signal"] == "PASS"
    assert calls == ["malformed-retry"]


def test_typed_retry_root_receives_root_acceptance_checkpoint():
    import ouroboros.loop as loop_mod

    trace = {}
    ctx = SimpleNamespace(
        task_id="retry-2",
        task_metadata={
            "root_task_id": "logical-root",
            "parent_task_id": "",
            "delegation_role": "root",
            "original_task_id": "retry-1",
            "timeout_retry_from": "retry-1",
        },
    )

    loop_mod._mark_root_acceptance_checkpoint(
        ctx, trace, status="pass", pass_index=1,
    )

    assert trace["root_phase_checkpoint"] == {
        "phase": "task_acceptance",
        "status": "pass",
        "pass_index": 1,
        "post_task_synthesis": "pending_once",
    }


def test_root_acceptance_agent_refs_reach_host_packet_beyond_trajectory_cap(
    monkeypatch, tmp_path,
):
    from types import SimpleNamespace as NS

    import ouroboros.loop as loop_mod
    from ouroboros.loop_tool_execution import process_tool_results
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.review import _handle_task_acceptance_review

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    # v6.71.1 evidence-parity: the trajectory per-result cap rose from a hidden 700 to
    # the actor's default window (_ACCEPT_RESULT_CAP == DEFAULT_TOOL_RESULT_LIMIT). Push
    # the ref past the NEW cap so the test still exercises "beyond the trajectory cap →
    # still reaches the host packet via the agent_supplied path".
    agent_evidence = {
        "long_note": "x" * 16000,
        "receipt_ref": "artifact://receipt-123",
        "trailing_note": "y" * 5000,
    }
    tool_ctx = NS(
        drive_root=str(tmp_path),
        repo_dir=tmp_path,
        task_id="root",
        root_task_id="root",
        task_metadata={"root_task_id": "root"},
        task_contract={},
    )
    raw = _handle_task_acceptance_review(
        tool_ctx,
        claim="complete",
        goal="ship the verified result",
        checklist="receipt is present",
        evidence=agent_evidence,
    )
    payload = json.loads(raw)
    assert payload["agent_supplied"]["receipt_ref"] == "artifact://receipt-123"

    trace = {"tool_calls": []}
    process_tool_results(
        [{
            "fn_name": "task_acceptance_review",
            "tool_call_id": "acceptance-call",
            "result": raw,
            "is_error": False,
            "args_for_log": {
                "claim": "complete",
                "evidence": agent_evidence,
            },
            "tool_args": {},
            "result_meta": {"status": "ok"},
        }],
        [],
        trace,
        emit_progress=lambda _message: None,
    )

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "root"
    registry._ctx.root_task_id = "root"
    registry._ctx.task_metadata = {"root_task_id": "root"}
    registry._ctx.task_contract = {}
    host_ctx = loop_mod._TaskAcceptanceContext(
        tools=registry,
        content="complete",
        task_id="root",
        task_type="task",
        llm_trace=trace,
        drive_root=tmp_path,
        messages=[],
        emit_progress=lambda _message: None,
        mode="auto",
        subtree_statuses=[],
        budget_profile=None,
        passes_done=0,
    )
    host_evidence = loop_mod._build_host_acceptance_evidence(host_ctx)

    assert host_evidence["agent_supplied"]["receipt_ref"] == (
        "artifact://receipt-123"
    )
    assert "artifact://receipt-123" not in json.dumps(
        host_evidence.get("tool_trajectory") or [], ensure_ascii=False,
    )


def test_off_mode_root_and_auto_mode_child_keep_existing_model_review(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS

    import ouroboros.review_evidence as re_mod
    import ouroboros.review_substrate as rs
    from ouroboros.tools.review import _handle_task_acceptance_review

    calls = []
    monkeypatch.setattr(re_mod, "collect_turn_diff", lambda ctx, **kwargs: "")
    monkeypatch.setattr(rs, "reviewer_slots", lambda **kwargs: [ReviewSlot(slot_id="a", model="m")])
    monkeypatch.setattr(rs, "build_improvement_capsule", lambda _result: "")
    monkeypatch.setattr(rs, "dissent_findings", lambda _result: [])

    def fake_run(request, **kwargs):
        calls.append((request.task_id, request.surface))
        return NS(aggregate_signal="PASS", actors=[], parsed_findings=[])

    monkeypatch.setattr(rs, "run_review_request", fake_run)

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    root_ctx = NS(
        drive_root=str(tmp_path),
        task_id="root",
        root_task_id="root",
        task_metadata={"root_task_id": "root"},
        task_contract={},
    )
    root_payload = json.loads(_handle_task_acceptance_review(root_ctx, claim="root done"))

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    child_ctx = NS(
        drive_root=str(tmp_path),
        task_id="child",
        root_task_id="root",
        parent_task_id="root",
        delegation_role="subagent",
        task_metadata={
            "root_task_id": "root",
            "parent_task_id": "root",
            "delegation_role": "subagent",
        },
        task_contract={},
    )
    child_payload = json.loads(_handle_task_acceptance_review(child_ctx, claim="child done"))

    assert calls == [("root", "task_acceptance"), ("child", "task_acceptance")]
    assert root_payload["aggregate_signal"] == "PASS"
    assert child_payload["aggregate_signal"] == "PASS"


def test_stale_parent_lineage_cannot_trigger_a_second_host_panel(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS

    import ouroboros.loop as loop_mod
    import ouroboros.review_evidence as re_mod
    import ouroboros.review_substrate as rs
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.review import _handle_task_acceptance_review

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    monkeypatch.setattr(re_mod, "collect_turn_diff", lambda ctx, **kwargs: "")
    monkeypatch.setattr(
        rs,
        "reviewer_slots",
        lambda **kwargs: [ReviewSlot(slot_id="a", model="m")],
    )
    monkeypatch.setattr(rs, "build_improvement_capsule", lambda _result: "")
    monkeypatch.setattr(rs, "dissent_findings", lambda _result: [])
    calls = []

    def fake_run(request, **kwargs):
        calls.append((request.task_id, request.surface))
        return NS(aggregate_signal="PASS", actors=[], parsed_findings=[])

    monkeypatch.setattr(rs, "run_review_request", fake_run)
    metadata = {
        # Legacy/malformed snapshot: root id is absent but an old parent remains.
        "parent_task_id": "missing-parent",
        "delegation_role": "root",
    }
    tool_ctx = NS(
        drive_root=str(tmp_path),
        task_id="restored-task",
        task_metadata=metadata,
        task_contract={},
    )
    payload = json.loads(
        _handle_task_acceptance_review(tool_ctx, claim="restored result")
    )
    assert payload["aggregate_signal"] == "PASS"
    assert calls == [("restored-task", "task_acceptance")]

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "restored-task"
    registry._ctx.task_metadata = metadata
    registry._ctx.task_contract = {}
    monkeypatch.setattr(
        loop_mod,
        "_begin_task_acceptance_fence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale-parent lineage must not reach the host panel")
        ),
    )
    trace = {"tool_calls": [], "review_runs": []}
    assert loop_mod._run_task_acceptance_review_once(
        tools=registry,
        content="restored result",
        task_id="restored-task",
        task_type="task",
        llm_trace=trace,
        drive_root=tmp_path,
        messages=[],
        emit_progress=lambda _message: None,
    ) is False
    assert trace["review_decision"] == {
        "eligibility": "not_eligible",
        "trigger": "skipped_child_advisory",
    }
    assert calls == [("restored-task", "task_acceptance")]


class _MixedReviewTruthLLM:
    def chat(self, **kwargs):
        model = str(kwargs.get("model") or "")
        if "timeout" in model:
            # Some timeout types stringify to an empty message.  The transport
            # truth must come from the exception type, not incidental wording.
            raise TimeoutError()
        if "malformed" in model:
            return {"content": "not json"}, {}
        return {
            "content": json.dumps({
                "verdict": "DEGRADED",
                "outcome_tier": "best_effort",
                "summary": "Evidence coverage is incomplete.",
                "findings": [],
            }),
        }, {
            "provider": "openrouter",
            "resolved_model": "google/gemini-3.5-flash",
        }


def test_review_actor_truth_separates_transport_parse_and_semantics(tmp_path):
    from ouroboros.review_substrate import compact_review_projection

    result = run_review_request(
        ReviewRequest(
            surface="task_acceptance", goal="g", subject="candidate",
            policy={"min_successful_slots": 2}, task_id="truth",
        ),
        slots=[
            ReviewSlot("timeout", "anthropic/timeout-model", role_hint="acceptance reviewer"),
            ReviewSlot("malformed", "openai/malformed-model", role_hint="acceptance reviewer"),
            ReviewSlot("degraded", "google/degraded-model", role_hint="acceptance reviewer"),
        ],
        drive_root=tmp_path,
        llm=_MixedReviewTruthLLM(),
    )
    actors = {actor["slot_id"]: actor for actor in result.actors}
    assert result.panel_id.startswith("panel_")
    assert len(result.panel_id) == len("panel_") + 16
    assert actors["timeout"]["transport_status"] == "timeout"
    assert actors["timeout"]["parse_status"] == "malformed"
    assert actors["timeout"]["semantic_verdict"] == ""
    assert actors["malformed"]["transport_status"] == "success"
    assert actors["malformed"]["parse_status"] == "malformed"
    assert actors["degraded"]["parse_status"] == "valid"
    assert actors["degraded"]["semantic_verdict"] == "DEGRADED"
    assert actors["degraded"]["parsed"]["outcome_tier"] == "best_effort"
    assert actors["degraded"]["provider"] == "openrouter"
    assert actors["degraded"]["model"] == "google/gemini-3.5-flash"
    assert actors["degraded"]["actor_role"] == "acceptance reviewer"

    run = dict(result.__dict__)
    run.update({
        "authority": "host_root",
        "candidate_hash": "c" * 64,
        "evidence_revision": "e" * 64,
        "fence_hash": "f" * 64,
        "enforcement_impact": "degrades_completion",
    })
    panel = compact_review_projection([run])["panels"][0]
    assert panel["panel_id"] == result.panel_id
    assert panel["transport_status"] == "partial"
    assert panel["parse_status"] == "malformed"
    assert panel["quorum"] == {"required": 2, "contributed": 0, "configured": 3}
    assert len(panel["actors"]) == 3
    assert next(
        actor for actor in panel["actors"] if actor["slot_id"] == "degraded"
    )["outcome_tier"] == "best_effort"
    assert all("raw_text" not in actor for actor in panel["actors"])


def test_compact_review_projection_redacts_public_reasons_before_truncation():
    from ouroboros.review_substrate import compact_review_projection

    secret = "sk-or-" + ("ReviewSecret123" * 4)
    benign_reason = "Evidence coverage is incomplete, but the consumer flow is clear."
    actor_prefix = benign_reason + ("x" * 400) + " credential="
    panel_prefix = "Panel retained its benign diagnostic context. " + ("y" * 735)
    run = {
        "request": {"surface": "task_acceptance", "policy": {"min_successful_slots": 1}},
        "aggregate_signal": "DEGRADED",
        "reason": panel_prefix + " " + secret,
        "actors": [
            {
                "slot_id": "benign",
                "model": "model-safe",
                "status": "ok",
                "signal": "PASS",
                "parsed": {"verdict": "PASS", "summary": benign_reason, "findings": []},
                "quorum_contribution": True,
            },
            {
                "slot_id": "secret-bearing",
                "model": "model-secret",
                "status": "ok",
                "signal": "DEGRADED",
                "parsed": {
                    "verdict": "DEGRADED",
                    "summary": actor_prefix + secret,
                    "findings": [],
                },
            },
        ],
    }

    panel = compact_review_projection([run])["panels"][0]
    actors = {actor["slot_id"]: actor for actor in panel["actors"]}
    rendered = json.dumps(panel, ensure_ascii=False)

    assert actors["benign"]["reason"] == benign_reason
    assert benign_reason in actors["secret-bearing"]["reason"]
    assert "Panel retained its benign diagnostic context." in panel["reason"]
    assert secret not in rendered
    assert secret[:20] not in rendered
    assert "***REDACTED***" in actors["secret-bearing"]["reason"]
    assert "***REDACTED***" in panel["reason"]


def test_actor_projection_carries_bounded_disclosed_finding_rows():
    from ouroboros.review_substrate import (
        MAX_PROJECTED_ACTOR_FINDINGS, compact_review_projection,
    )

    secret = "sk-or-" + ("FindingSecret456" * 4)
    long_recommendation = "Re-run the verifier with the fixed seed. " * 120
    findings = [
        {
            "severity": "critical",
            "item": f"finding {index}",
            "evidence": f"evidence {index}",
            "recommendation": f"fix {index}",
        }
        for index in range(MAX_PROJECTED_ACTOR_FINDINGS + 2)
    ]
    findings[0]["evidence"] = "credential=" + secret
    findings[1]["recommendation"] = long_recommendation
    run = {
        "request": {"surface": "task_acceptance", "policy": {"min_successful_slots": 1}},
        "aggregate_signal": "FAIL",
        "actors": [
            {
                "slot_id": "with-findings",
                "model": "model-a",
                "status": "ok",
                "signal": "FAIL",
                "parsed": {"verdict": "FAIL", "summary": "s", "findings": findings},
                "quorum_contribution": True,
            },
            {
                "slot_id": "clean",
                "model": "model-b",
                "status": "ok",
                "signal": "PASS",
                "parsed": {"verdict": "PASS", "summary": "ok", "findings": []},
                "quorum_contribution": True,
            },
            {
                "slot_id": "transport-hole",
                "model": "model-c",
                "status": "error",
                "error": "timed out",
                "parsed": None,
            },
            {
                "slot_id": "odd-shape",
                "model": "model-d",
                "status": "ok",
                "signal": "FAIL",
                "parsed": {
                    "verdict": "FAIL",
                    "summary": "s",
                    "findings": [{
                        "weird_key": "the only copy of this evidence",
                        "password": "hunter2-odd-shape",
                    }],
                },
            },
            {
                # A non-string value under a KNOWN key keeps structural
                # key-based masking: str() first would flatten the nested
                # secret past the key-name redactor.
                "slot_id": "nested-evidence",
                "model": "model-f",
                "status": "ok",
                "signal": "FAIL",
                "parsed": {
                    "verdict": "FAIL",
                    "summary": "s",
                    "findings": [{
                        "severity": "high",
                        "item": "nested shape",
                        "evidence": {"password": "hunter2-nested-shape"},
                    }],
                },
            },
            {
                # The array-ladder reviewer contract shapes findings as
                # {item, verdict, severity, reason}: the substantive `reason`
                # text must survive projection.
                "slot_id": "triad-shape",
                "model": "model-e",
                "status": "ok",
                "signal": "FAIL",
                "parsed": [{
                    "item": "missing rollback test",
                    "verdict": "FAIL",
                    "severity": "high",
                    "reason": "the new path has no failure-injection coverage",
                }],
            },
        ],
    }

    panel = compact_review_projection([run])["panels"][0]
    actors = {actor["slot_id"]: actor for actor in panel["actors"]}
    rendered = json.dumps(panel, ensure_ascii=False)

    rows = actors["with-findings"]["findings"]
    assert len(rows) == MAX_PROJECTED_ACTOR_FINDINGS
    assert actors["with-findings"]["findings_omitted"] == 2
    assert rows[2] == {
        "severity": "critical", "item": "finding 2",
        "evidence": "evidence 2", "recommendation": "fix 2",
    }
    # The count stays beside the rows: coverage keeps the full total.
    assert actors["with-findings"]["coverage"]["findings"] == len(findings)
    # Redaction covers finding bodies exactly like reasons.
    assert secret not in rendered
    assert "***REDACTED***" in rows[0]["evidence"]
    # A clipped string discloses its own cut instead of clipping silently.
    assert "OMISSION NOTE" in rows[1]["recommendation"]
    assert len(rows[1]["recommendation"]) < len(long_recommendation)

    # A reviewer that reported no findings states that as an empty disclosed
    # list; a reviewer with no parsed response leaves a hole, not a zero.
    assert actors["clean"]["findings"] == []
    assert actors["clean"]["findings_omitted"] == 0
    assert "findings" not in actors["transport-hole"]
    assert "findings_omitted" not in actors["transport-hole"]

    # An unknown finding shape still ships its evidence as a bounded row, and
    # structural key-based secret masking applies BEFORE serialization.
    odd_rows = actors["odd-shape"]["findings"]
    assert odd_rows and "the only copy of this evidence" in odd_rows[0]["item"]
    assert "hunter2-odd-shape" not in rendered
    # #447 G11: key-name redaction leaves a typed fingerprint, not bare deletion.
    assert "***REDACTED[" in odd_rows[0]["item"]

    nested_rows = actors["nested-evidence"]["findings"]
    assert "hunter2-nested-shape" not in rendered
    assert "***REDACTED[" in nested_rows[0]["evidence"]
    assert nested_rows[0]["item"] == "nested shape"

    # A list-shaped parsed response (array reviewer contract) projects its
    # rows too, and the substantive `reason`/`verdict` fields survive.
    triad_rows = actors["triad-shape"]["findings"]
    assert triad_rows == [{
        "severity": "high", "verdict": "FAIL", "item": "missing rollback test",
        "reason": "the new path has no failure-injection coverage",
    }]
    assert actors["triad-shape"]["findings_omitted"] == 0


class _MixedPassPassFailLLM:
    def chat(self, **kwargs):
        if str(kwargs.get("model") or "").endswith("-2"):
            body = {
                "verdict": "FAIL",
                "outcome_tier": "blocked_with_evidence",
                "completion_coach": "Resolve the verified acceptance gap.",
                "findings": [{
                    "severity": "high",
                    "item": "acceptance_gap",
                    "evidence": "The required behavior is not demonstrated.",
                    "recommendation": "Add independent evidence for the missing behavior.",
                }],
                "summary": "The candidate is not ready.",
            }
        else:
            body = {
                "verdict": "PASS",
                "outcome_tier": "solved",
                "completion_coach": "Ship the candidate.",
                # Production panels always required criteria evidence (the knob was
                # constant-true and is deleted): a contributing solved PASS carries
                # supported criteria with refs.
                "criteria_used": [{
                    "criterion": "candidate is verified", "status": "supported",
                    "evidence_refs": ["verification_summary"],
                }],
                "findings": [],
                "summary": "The candidate is ready.",
            }
        return {"content": json.dumps(body)}, {}


def test_mixed_panel_counts_valid_participation_independently_of_veto(tmp_path):
    from ouroboros.review_substrate import (
        aggregate_outcome_tier,
        compact_review_projection,
        task_acceptance_is_clean,
    )

    result = run_review_request(
        ReviewRequest(
            surface="task_acceptance",
            goal="g",
            subject="candidate",
            policy={"classify_outcome_tier": True, "min_successful_slots": 2},
            task_id="mixed-panel",
        ),
        slots=[ReviewSlot(f"s{i}", f"m-{i}") for i in range(3)],
        drive_root=tmp_path,
        llm=_MixedPassPassFailLLM(),
    )

    assert result.aggregate_signal == "FAIL"
    assert aggregate_outcome_tier(result) == "blocked_with_evidence"
    assert task_acceptance_is_clean(result) is False
    actors = {actor["slot_id"]: actor for actor in result.actors}
    assert all(actor["quorum_contribution"] is True for actor in actors.values())
    assert actors["s0"]["enforcement_impact"] == "supports_pass"
    assert actors["s1"]["enforcement_impact"] == "supports_pass"
    assert actors["s2"]["enforcement_impact"] == "veto"

    run = dict(result.__dict__)
    run["authority"] = "host_root"
    panel = compact_review_projection([run])["panels"][0]
    assert panel["aggregate_signal"] == "FAIL"
    assert panel["quorum"] == {"required": 2, "contributed": 3, "configured": 3}
    assert panel["coverage"]["quorum_contributing"] == 3
    assert [actor["enforcement_impact"] for actor in panel["actors"]] == [
        "supports_pass",
        "supports_pass",
        "veto",
    ]


class _ArrayReviewTruthLLM:
    def chat(self, **_kwargs):
        return {
            "content": json.dumps([{
                "verdict": "FAIL",
                "item": "missing_visual_evidence",
                "evidence": "No inspected screenshot is attached.",
                "recommendation": "Inspect the captured consumer flow.",
            }]),
        }, {
            "provider": "openrouter",
            "resolved_model": "anthropic/claude-fable-5",
        }


def test_review_actor_truth_preserves_array_coverage_and_physical_route(tmp_path):
    result = run_review_request(
        ReviewRequest(
            surface="multi_model_review",
            goal="g",
            subject="candidate",
            policy={"min_successful_slots": 1},
            task_id="array-truth",
        ),
        slots=[ReviewSlot("array", "anthropic/array-model")],
        drive_root=tmp_path,
        llm=_ArrayReviewTruthLLM(),
    )

    actor = result.actors[0]
    assert actor["parse_status"] == "valid"
    assert actor["semantic_verdict"] == "FAIL"
    assert actor["coverage"]["findings"] == 1
    assert actor["reason"] == "No inspected screenshot is attached."
    assert actor["provider"] == "openrouter"
    assert actor["model"] == "anthropic/claude-fable-5"


def test_host_acceptance_enforcement_impact_records_applied_action(tmp_path):
    from types import SimpleNamespace as NS

    import ouroboros.loop as loop_mod

    tool_ctx = NS(_task_acceptance_seen_bindings={})
    ctx = loop_mod._TaskAcceptanceContext(
        tools=NS(_ctx=tool_ctx),
        content="candidate",
        task_id="impact",
        task_type="task",
        llm_trace={"review_runs": []},
        drive_root=tmp_path,
        messages=[],
        emit_progress=lambda _message: None,
        mode="required",
        subtree_statuses=[],
        budget_profile={},
        passes_done=0,
        review_binding={"binding_hash": "b" * 64},
    )
    degraded = NS(
        aggregate_signal="DEGRADED",
        degraded=True,
        actors=[],
        parsed_findings=[],
        degraded_reasons=["no quorum"],
        request={},
    )

    record = loop_mod._record_host_acceptance_run(ctx, degraded)
    assert record["enforcement_impact"] == "degrades_completion"
    loop_mod._set_applied_host_acceptance_impact(
        record,
        degraded,
        requires_revision=True,
    )
    assert record["enforcement_impact"] == "requires_revision"
    loop_mod._set_applied_host_acceptance_impact(
        record,
        degraded,
        requires_revision=False,
    )
    assert record["enforcement_impact"] == "degrades_completion"


def test_review_binding_is_stable_and_tracks_each_exact_input():
    from ouroboros.review_substrate import build_review_binding

    base = build_review_binding(
        candidate="answer", evidence={"claims": ["verified"]}, fence_token_or_state="fence-1",
    )
    assert base == build_review_binding(
        candidate="answer", evidence={"claims": ["verified"]}, fence_token_or_state="fence-1",
    )
    assert base["candidate_hash"] != build_review_binding(
        candidate="changed", evidence={"claims": ["verified"]}, fence_token_or_state="fence-1",
    )["candidate_hash"]
    assert base["evidence_revision"] != build_review_binding(
        candidate="answer", evidence={"claims": ["changed"]}, fence_token_or_state="fence-1",
    )["evidence_revision"]
    assert base["binding_hash"] != build_review_binding(
        candidate="answer", evidence={"claims": ["verified"]}, fence_token_or_state="fence-2",
    )["binding_hash"]
    assert len(base["fence_hash"]) == 64
    assert "fence_token_or_state" not in base
    assert "fence-1" not in json.dumps(base)


def test_task_acceptance_review_schema_exposes_agent_disposition():
    from ouroboros.tools.review import get_tools

    tool = next(entry for entry in get_tools() if entry.name == "task_acceptance_review")
    props = tool.schema["parameters"]["properties"]

    assert props["agent_disposition"]["enum"] == ["accepted", "rejected", "partial", "deferred"]
    assert "rationale" in props


def test_collect_turn_diff_redacts_secrets(tmp_path):
    """T1 (v6.35.0): a tracked credential edit must be REDACTED before the diff
    reaches reviewer LLM slots (no raw secret exfiltration)."""
    import subprocess as sp
    from types import SimpleNamespace as NS

    from ouroboros.review_evidence import collect_turn_diff

    repo = tmp_path / "r"
    repo.mkdir()
    sp.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "conf.py").write_text('API_KEY = "placeholder"\n', encoding="utf-8")
    sp.run(["git", "add", "conf.py"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "i"],
           cwd=repo, check=True, capture_output=True)
    # Assemble the fake provider key from chunks so this test FILE contains no
    # contiguous provider-key literal (secret scanners match source, not runtime).
    # The concatenated runtime value is what the redactor must catch.
    secret = "sk-" + "or-" + "v1-" + "abcdef1234567890" * 2 + "deadbeef"
    (repo / "conf.py").write_text(f'API_KEY = "{secret}"\n', encoding="utf-8")

    diff = collect_turn_diff(NS(repo_dir=repo))
    assert secret not in diff           # the literal secret value is gone
    assert "REDACTED" in diff           # replaced with a redaction marker
    assert "conf.py" in diff            # the file/path (evidence-independence fact) survives


def test_collect_turn_diff_surfaces_committed_change(tmp_path):
    """T1 (v6.35.0): when the turn's work was already committed, `git diff HEAD`
    is empty — collect_turn_diff must still surface the committed files via the
    most recent commit so evidence independence can be judged."""
    import subprocess as sp
    from types import SimpleNamespace as NS

    from ouroboros.review_evidence import collect_turn_diff

    repo = tmp_path / "r"
    repo.mkdir()
    sp.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    sp.run(["git", "add", "a.py"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "base"],
           cwd=repo, check=True, capture_output=True)
    # Commit the turn's work, so `git diff HEAD` is empty.
    (repo / "feature.py").write_text("def feat():\n    return 1\n", encoding="utf-8")
    sp.run(["git", "add", "feature.py"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "feature"],
           cwd=repo, check=True, capture_output=True)

    # Without a current-turn commit signal, the unrelated HEAD commit is NOT shown.
    assert "feature.py" not in collect_turn_diff(NS(repo_dir=repo))
    # With the commit signal (this turn committed), the committed work IS surfaced.
    diff = collect_turn_diff(NS(repo_dir=repo), include_recent_commit=True)
    assert "feature.py" in diff
    assert "committed this turn" in diff


def test_collect_turn_diff_disables_git_exec_drivers(tmp_path):
    """v6.35.0 security: the active workspace may be an UNTRUSTED repo, so
    collect_turn_diff must run git with --no-ext-diff AND --no-textconv — a
    repo-configured textconv/external-diff driver must never execute on the host
    while collecting review evidence (Bible P3)."""
    import subprocess as sp
    from types import SimpleNamespace as NS

    from ouroboros.review_evidence import collect_turn_diff

    repo = tmp_path / "r"
    repo.mkdir()
    sp.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    marker = tmp_path / "pwned"
    # A malicious textconv driver that would create a marker file if git ran it.
    sp.run(["git", "config", "diff.evil.textconv", f"sh -c 'touch {marker}'; cat"],
           cwd=repo, check=True, capture_output=True)
    (repo / ".gitattributes").write_text("*.secret diff=evil\n", encoding="utf-8")
    (repo / "f.secret").write_text("one\n", encoding="utf-8")
    sp.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "x"],
           cwd=repo, check=True, capture_output=True)
    # Modify the attributed file so the tracked diff would render it via textconv.
    (repo / "f.secret").write_text("two\n", encoding="utf-8")

    # Exercises both the `git diff HEAD` and the `git show HEAD` code paths.
    collect_turn_diff(NS(repo_dir=repo), include_recent_commit=True)
    assert not marker.exists()   # the textconv driver must NOT have executed


def test_collect_turn_diff_does_not_assert_untracked_authorship(tmp_path):
    """T1 (v6.35.0): untracked files are labeled honestly as working-tree state,
    NOT asserted as authored 'this turn' — the host has no baseline, so it must
    not steer the reviewer's EVIDENCE-INDEPENDENCE judgment with a false claim."""
    import subprocess as sp
    from types import SimpleNamespace as NS

    from ouroboros.review_evidence import collect_turn_diff

    repo = tmp_path / "r"
    repo.mkdir()
    sp.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    sp.run(["git", "add", "a.py"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "base"],
           cwd=repo, check=True, capture_output=True)
    # A pre-existing untracked file (the host cannot prove it was authored now).
    (repo / "preexisting_test.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")

    diff = collect_turn_diff(NS(repo_dir=repo))
    assert "preexisting_test.py" in diff          # surfaced as evidence
    assert "this turn" not in diff.lower()         # but NOT asserted as authored now
    assert "working-tree" in diff.lower()          # honestly labeled


def test_collect_turn_diff_includes_commit_even_with_leftover_dirty(tmp_path):
    """T1 (v6.35.0): a turn that commits AND leaves further dirty tracked changes
    must surface BOTH — the committed patch is no longer dropped just because the
    working tree is also dirty."""
    import subprocess as sp
    from types import SimpleNamespace as NS

    from ouroboros.review_evidence import collect_turn_diff

    repo = tmp_path / "r"
    repo.mkdir()
    sp.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    sp.run(["git", "add", "a.py"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "base"],
           cwd=repo, check=True, capture_output=True)
    # This turn: commit feature.py ...
    (repo / "feature.py").write_text("def feat():\n    return 1\n", encoding="utf-8")
    sp.run(["git", "add", "feature.py"], cwd=repo, check=True, capture_output=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "feature"],
           cwd=repo, check=True, capture_output=True)
    # ... then leave a further dirty tracked edit (so `git diff HEAD` is NON-empty).
    (repo / "a.py").write_text("x = 2  # tweaked\n", encoding="utf-8")

    diff = collect_turn_diff(NS(repo_dir=repo), include_recent_commit=True)
    assert "tweaked" in diff                       # the leftover dirty tracked change
    assert "feature.py" in diff                    # AND the committed patch
    assert "committed this turn" in diff


def test_review_substrate_treats_duplicate_models_as_independent_slots(tmp_path):
    llm = FakeLLM()
    slots = [
        ReviewSlot(slot_id="triad_a", model="same/model", effort="high"),
        ReviewSlot(slot_id="triad_b", model="same/model", effort="high"),
    ]
    result = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="verify final claim", subject="done", task_id="task-1"),
        slots=slots,
        drive_root=tmp_path,
        llm=llm,
    )

    assert result.aggregate_signal == "PASS"
    assert [actor["slot_id"] for actor in result.actors] == ["triad_a", "triad_b"]
    assert [call["model"] for call in llm.calls] == ["same/model", "same/model"]
    for actor in result.actors:
        assert actor["prompt_ref"]["manifest_ref"]["path"]
        assert actor["response_ref"]["manifest_ref"]["path"]


def test_review_substrate_queues_all_slots_above_concurrency_cap(tmp_path):
    llm = FakeLLM()
    slots = [
        ReviewSlot(slot_id=f"slot_{idx}", model=f"model-{idx}", effort="high")
        for idx in range(10)
    ]
    result = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="verify final claim", subject="done", task_id="task-10"),
        slots=slots,
        drive_root=tmp_path,
        llm=llm,
    )

    assert result.aggregate_signal == "PASS"
    assert [actor["slot_id"] for actor in result.actors] == [slot.slot_id for slot in slots]
    assert {call["model"] for call in llm.calls} == {slot.model for slot in slots}
    assert len(llm.calls) == 10
    assert all(actor["status"] == "ok" for actor in result.actors)

    slow_calls = []
    slow_llm = SimpleNamespace(chat=lambda **kwargs: (
        slow_calls.append(kwargs),
        time.sleep(0.2),
        ({"content": "{\"verdict\":\"PASS\",\"findings\":[],\"summary\":\"late\"}"}, {}),
    )[-1])
    slow_slots = [
        ReviewSlot(slot_id=f"slow_{idx}", model=f"slow-model-{idx}", effort="high", timeout_sec=0.05)
        for idx in range(10)
    ]
    slow_result = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="verify final claim", subject="done", task_id="task-slow"),
        slots=slow_slots,
        drive_root=tmp_path,
        llm=slow_llm,
    )
    assert len(slow_calls) == 10
    assert "Not started before reviewer timeout budget expired" not in "\n".join(slow_result.degraded_reasons)


def test_review_substrate_reports_no_slots_as_degraded(tmp_path):
    result = run_review_request(
        ReviewRequest(surface="plan", goal="review plan", task_id="task-1"),
        slots=[],
        drive_root=tmp_path,
        llm=FakeLLM(),
    )

    assert result.aggregate_signal == "DEGRADED"
    assert result.degraded is True
    assert "no_review_slots" in result.degraded_reasons


def test_review_substrate_emits_usage_when_context_supplied(tmp_path):
    class Ctx:
        task_id = "task-usage"
        pending_events = []

    ctx = Ctx()
    result = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="review claim", task_id="task-usage"),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path,
        llm=FakeLLM(),
        usage_ctx=ctx,
    )

    assert result.aggregate_signal == "PASS"
    usage_events = [event for event in ctx.pending_events if event.get("type") == "llm_usage"]
    assert len(usage_events) == 1
    assert usage_events[0]["task_id"] == "task-usage"
    assert usage_events[0]["source"] == "review_substrate:task_acceptance"
    assert usage_events[0]["slot_id"] == "slot_a"


def test_review_usage_preserves_unknown_cost_as_null():
    from ouroboros.tools.review_helpers import emit_review_usage

    ctx = SimpleNamespace(task_id="unknown-review", pending_events=[])
    emit_review_usage(ctx, model="unknown/model", usage={}, source="test")
    event = ctx.pending_events[0]
    assert event["usage"]["cost"] is None
    assert event["usage"]["cost_known"] is False


def test_review_substrate_parses_fenced_json_array_findings(tmp_path):
    result = run_review_request(
        ReviewRequest(surface="scope", goal="review diff", task_id="task-json-array"),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path,
        llm=FencedArrayLLM(),
    )

    assert result.aggregate_signal == "FAIL"
    assert result.parsed_findings[0]["item"] == "x"
    assert result.actors[0]["parsed"][0]["verdict"] == "FAIL"


def test_review_substrate_parses_fenced_json_object_verdict(tmp_path):
    # A fenced JSON OBJECT (not array) must parse as PASS, not a false DEGRADED.
    result = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="verify claim", subject="done", task_id="task-obj"),
        slots=[
            ReviewSlot(slot_id="slot_a", model="m"),
            ReviewSlot(slot_id="slot_b", model="m"),
        ],
        drive_root=tmp_path,
        llm=FencedObjectLLM(),
    )
    assert result.aggregate_signal == "PASS"
    assert result.degraded is False


def test_review_substrate_degraded_quorum_carries_reason(tmp_path):
    # No FAIL, no PASS quorum, no transport errors -> DEGRADED must still be honest:
    # degraded=True with a non-empty reason (no DEGRADED/degraded=False/empty mismatch).
    result = run_review_request(
        ReviewRequest(
            surface="task_acceptance", goal="verify claim", subject="done", task_id="task-quorum",
            policy={"min_successful_slots": 2},
        ),
        slots=[
            ReviewSlot(slot_id="slot_a", model="m"),
            ReviewSlot(slot_id="slot_b", model="m"),
        ],
        drive_root=tmp_path,
        llm=ConcernsLLM(),
    )
    assert result.aggregate_signal == "DEGRADED"
    assert result.degraded is True
    assert result.degraded_reasons
    assert any("quorum_not_met" in reason for reason in result.degraded_reasons)


def test_p3_commit_actor_retries_same_slot_model_once_then_blocks(tmp_path):
    recovered_llm = Mock()
    recovered_llm.chat.side_effect = [
        TimeoutError("transient timeout"),
        (
            {"content": "{\"verdict\":\"PASS\",\"findings\":[],\"summary\":\"ok\"}"},
            {"prompt_tokens": 1, "completion_tokens": 1},
        ),
    ]
    recovered = run_review_request(
        ReviewRequest(
            surface="multi_model_review", goal="review diff",
            task_id="task-recovered", call_type="multi_model_review",
        ),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path,
        llm=recovered_llm,
    )
    assert recovered.aggregate_signal == "PASS"
    assert recovered.actors[0]["status"] == "ok"
    assert recovered_llm.chat.call_count == 2
    assert recovered_llm.chat.call_args_list[0].kwargs == recovered_llm.chat.call_args_list[1].kwargs

    failed_llm = Mock()
    failed_llm.chat.side_effect = [
        TimeoutError("transient timeout"),
        RuntimeError("provider exploded"),
    ]
    result = run_review_request(
        ReviewRequest(
            surface="multi_model_review", goal="review diff",
            task_id="task-error", call_type="multi_model_review",
        ),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path,
        llm=failed_llm,
    )

    actor = result.actors[0]
    assert result.aggregate_signal == "DEGRADED"
    assert failed_llm.chat.call_count == 2
    assert actor["status"] == "error"
    assert "provider exploded" in actor["error"]
    assert actor["prompt_ref"]["manifest_ref"]["path"]
    assert actor["response_ref"]["manifest_ref"]["path"]
    manifest = json.loads(open(actor["response_ref"]["manifest_ref"]["path"], encoding="utf-8").read())
    assert manifest["call_type"] == "multi_model_review_error"
    assert manifest["status"] == "error"

    from ouroboros.usage_accounting import _claim_physical_dispatch

    over_limit_llm = SimpleNamespace(chat=lambda **_kwargs: (
        _claim_physical_dispatch(),
        _claim_physical_dispatch(),
        _claim_physical_dispatch(),
    ))
    over_limit = run_review_request(
        ReviewRequest(
            surface="multi_model_review", goal="review diff",
            task_id="task-over-limit", call_type="multi_model_review",
        ),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path,
        llm=over_limit_llm,
    )
    assert over_limit.actors[0]["status"] == "error"
    assert "physical attempt limit exhausted (2/2)" in over_limit.actors[0]["error"]


def test_p3_scope_actor_retries_empty_same_slot_model_once_then_blocks(tmp_path, monkeypatch):
    from ouroboros.tools import scope_review

    rows = [
        {
            "item": item,
            "verdict": "PASS",
            "severity": "advisory",
            "reason": "Concrete scope artifact was checked and passes.",
        }
        for item in sorted(scope_review._SCOPE_REQUIRED_ITEMS)
    ]
    recovered_llm = Mock()
    recovered_llm.chat.side_effect = [
        ({"content": ""}, {"prompt_tokens": 0, "completion_tokens": 0}),
        (
            {"content": json.dumps(rows)},
            {"prompt_tokens": 1, "completion_tokens": 1},
        ),
    ]
    monkeypatch.setattr(scope_review, "LLMClient", lambda: recovered_llm)
    monkeypatch.setattr(scope_review, "_build_scope_prompt", lambda *a, **k: ("scope prompt", None))
    monkeypatch.setattr(scope_review, "_scope_window",
                        lambda _model, **_k: scope_review.ReviewerWindow(1_000_000, "confirmed"))
    ctx = SimpleNamespace(
        repo_dir=tmp_path, drive_root=tmp_path,
        task_id="scope-recovered", pending_events=[],
    )
    recovered = scope_review.run_scope_review(ctx, "review scope", scope_model="scope/model")
    assert recovered.status == "responded"
    assert recovered.blocked is False
    assert recovered_llm.chat.call_count == 2
    assert recovered_llm.chat.call_args_list[0].kwargs == recovered_llm.chat.call_args_list[1].kwargs

    empty_llm = Mock()
    empty_llm.chat.side_effect = [
        ({"content": ""}, {"prompt_tokens": 0, "completion_tokens": 0}),
        ({"content": ""}, {"prompt_tokens": 0, "completion_tokens": 0}),
    ]
    monkeypatch.setattr(scope_review, "LLMClient", lambda: empty_llm)
    ctx.task_id = "scope-empty"
    failed = scope_review.run_scope_review(ctx, "review scope", scope_model="scope/model")
    assert failed.blocked is True
    assert failed.status == "empty_response"
    assert failed.operation_id
    assert empty_llm.chat.call_count == 2


def test_review_substrate_persists_timeout_actor_refs(tmp_path):
    import threading
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    release = threading.Event()

    def gated_chat(**_kwargs):
        # Holds the call open until the test releases it, so the 0.01s window
        # is GUARANTEED to expire and the timeout actor is the one persisted.
        # The previous 0.2s-sleep-vs-0.01s-window margin was NOT discriminating
        # on a loaded CI host (the heal wave measured a 0.207s poll oversleep):
        # a pre-window settle replaced the asserted timeout actor with PASS
        # (same event gate as
        # test_replayed_late_review_does_not_charge_same_context_twice).
        assert release.wait(10), "test never released the gated review call"
        return {"content": "{\"verdict\":\"PASS\",\"findings\":[],\"summary\":\"late\"}"}, {}

    hanging_llm = SimpleNamespace(chat=gated_chat)
    request = ReviewRequest(surface="scope", goal="review diff", task_id="task-timeout")
    slot = ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=0.01)
    result = run_review_request(
        request,
        slots=[slot],
        drive_root=tmp_path,
        llm=hanging_llm,
    )

    actor = result.actors[0]
    assert actor["status"] == "error"
    assert "Timeout after" in actor["error"]
    assert actor["prompt_ref"]["manifest_ref"]["path"]
    assert actor["response_ref"]["manifest_ref"]["path"]
    release.set()
    # Drain the released worker before teardown (same drain as
    # test_explicit_retry_key_joins_worker_after_prompt_history_changes).
    key = _attempt_key(request, slot)
    deadline = time.monotonic() + 5.0
    active = True
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            active = key in _ACTIVE
        if not active:
            break
        time.sleep(0.01)
    assert not active


def test_spent_owner_deadline_does_not_dispatch_a_review_worker(tmp_path):
    calls = []
    paid = []

    class NeverCalledLLM:
        def chat(self, **_kwargs):
            calls.append(1)
            raise AssertionError("review transport dispatched after owner deadline")

    ctx = SimpleNamespace(
        task_id="spent-review", task_attempt=1, task_metadata={},
        event_queue=None, pending_events=[],
        _review_paid_stamp=lambda: paid.append(1),
    )
    result = run_review_request(
        ReviewRequest(
            surface="scope", goal="review", task_id="spent-review",
            deadline_at="2000-01-01T00:00:00Z",
        ),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=300)],
        drive_root=tmp_path, llm=NeverCalledLLM(), usage_ctx=ctx,
    )

    actor = result.actors[0]
    assert calls == [] and paid == []
    assert actor["status"] == "not_dispatched"
    assert actor["operation_state"] == "not_dispatched"
    assert actor["operation_id"] == ""
    assert actor["late_result_pending"] is False


def test_review_worker_does_not_retry_after_its_logical_deadline(tmp_path):
    import threading
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    calls = []
    first_finished = threading.Event()

    class LateTransportFailure:
        def chat(self, **_kwargs):
            calls.append(1)
            time.sleep(0.05)
            first_finished.set()
            raise TimeoutError("late transport failure")

    ctx = SimpleNamespace(
        task_id="late-failure", task_attempt=1, task_metadata={},
        event_queue=None, pending_events=[],
    )
    request = ReviewRequest(
        surface="multi_model_review", goal="review", task_id="late-failure",
        task_attempt=1,
    )
    slot = ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=0.01)
    result = run_review_request(
        request, slots=[slot], drive_root=tmp_path,
        llm=LateTransportFailure(), usage_ctx=ctx,
    )
    assert result.actors[0]["operation_state"] == "in_flight"
    assert first_finished.wait(1.0)

    key = _attempt_key(request, slot)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            active = key in _ACTIVE
        if not active:
            break
        time.sleep(0.01)
    assert not active
    assert calls == [1]


def test_late_review_result_is_replayed_without_a_second_paid_dispatch(tmp_path):
    import threading
    from types import SimpleNamespace
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    calls = []
    release = threading.Event()

    class GatedLLM:
        """Holds the paid call open until the test releases it, so the first
        poll window is GUARANTEED to expire while the call is still in flight.
        The previous 0.08s-sleep-vs-0.05s-window race flaked on slow CI hosts:
        an oversleeping poll wait could observe the settled result and return
        'settled' instead of 'in_flight'."""

        def chat(self, **_kwargs):
            calls.append(dict(_kwargs))
            assert release.wait(10), "test never released the gated review call"
            return {"content": '{"verdict":"PASS","findings":[],"summary":"late"}'}, {}

    ctx = SimpleNamespace(
        task_id="late-review",
        task_attempt=1,
        event_queue=None,
        pending_events=[],
    )
    request = ReviewRequest(
        surface="scope",
        goal="review diff",
        task_id="late-review",
        task_attempt=1,
    )
    slot = ReviewSlot(
        slot_id="slot_a", model="same/model", timeout_sec=0.05,
        transport_timeout_sec=10,
    )
    first = run_review_request(request, slots=[slot], drive_root=tmp_path, llm=GatedLLM(), usage_ctx=ctx)
    assert first.actors[0]["operation_state"] == "in_flight"
    assert first.actors[0]["late_result_pending"] is True

    release.set()
    # The settled-attempt cache is written in the same critical section that
    # retires the active attempt, so waiting for the key to leave _ACTIVE is
    # the event that makes the replay below deterministic (same drain wait as
    # test_review_worker_does_not_retry_after_its_logical_deadline).
    key = _attempt_key(request, slot)
    deadline = time.monotonic() + 5.0
    active = True
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            active = key in _ACTIVE
        if not active:
            break
        time.sleep(0.01)
    assert not active
    second_slot = ReviewSlot(
        slot_id="slot_a", model="same/model", timeout_sec=0.02,
        transport_timeout_sec=10,
    )
    second = run_review_request(request, slots=[second_slot], drive_root=tmp_path, llm=GatedLLM(), usage_ctx=ctx)
    assert len(calls) == 1
    assert second.actors[0]["status"] == "ok"
    assert second.actors[0]["operation_state"] == "late_settled"
    assert second.actors[0]["late_result_pending"] is False


def test_explicit_retry_key_joins_worker_after_prompt_history_changes(tmp_path):
    import threading
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    calls = []
    release = threading.Event()

    class GatedLLM:
        """Blocks the single paid call until the test releases it, so BOTH poll
        windows below expire while the worker is provably still in flight (the
        previous 0.08s-sleep-vs-0.02s-window race flaked on slow CI hosts)."""

        def chat(self, **kwargs):
            calls.append(kwargs)
            assert release.wait(10), "test never released the gated review call"
            return {"content": '{"verdict":"PASS","findings":[],"summary":"late"}'}, {}

    ctx = SimpleNamespace(task_id="history-retry", event_queue=None, pending_events=[])
    slot = ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=0.02)
    first_request = ReviewRequest(
        surface="scope_review", goal="review", task_id="history-retry",
        retry_key="snapshot-1/cycle-1", messages=[{"role": "user", "content": "first"}],
    )
    first = run_review_request(
        first_request,
        slots=[slot], drive_root=tmp_path, llm=GatedLLM(), usage_ctx=ctx,
    )
    second = run_review_request(
        ReviewRequest(
            surface="scope_review", goal="review", task_id="history-retry",
            retry_key="snapshot-1/cycle-1",
            messages=[{"role": "user", "content": "first\nprior round: pending"}],
        ),
        slots=[slot], drive_root=tmp_path, llm=GatedLLM(), usage_ctx=ctx,
    )

    assert len(calls) == 1
    assert first.actors[0]["operation_id"] == second.actors[0]["operation_id"]
    assert first.actors[0]["operation_state"] == second.actors[0]["operation_state"] == "in_flight"
    release.set()
    # Drain the released worker before teardown (same wait as
    # test_review_worker_does_not_retry_after_its_logical_deadline).
    key = _attempt_key(first_request, slot)
    deadline = time.monotonic() + 5.0
    active = True
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            active = key in _ACTIVE
        if not active:
            break
        time.sleep(0.01)
    assert not active


def test_explicit_retry_key_replays_normally_settled_actor_without_second_dispatch(tmp_path):
    calls = []

    class FastLLM:
        def chat(self, **kwargs):
            calls.append(kwargs)
            return {"content": '{"verdict":"PASS","findings":[],"summary":"done"}'}, {}

    ctx = SimpleNamespace(task_id="settled-retry", event_queue=None, pending_events=[])
    slot = ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=1)
    first = run_review_request(
        ReviewRequest(
            surface="plan_review", goal="review", task_id="settled-retry",
            retry_key="plan-1/cycle-1", messages=[{"role": "user", "content": "first"}],
        ),
        slots=[slot], drive_root=tmp_path, llm=FastLLM(), usage_ctx=ctx,
    )
    second = run_review_request(
        ReviewRequest(
            surface="plan_review", goal="review", task_id="settled-retry",
            retry_key="plan-1/cycle-1",
            messages=[{"role": "user", "content": "first plus rendered history"}],
        ),
        slots=[slot], drive_root=tmp_path, llm=FastLLM(), usage_ctx=ctx,
    )

    assert len(calls) == 1
    assert first.actors[0]["operation_id"] == second.actors[0]["operation_id"]
    assert second.actors[0]["operation_state"] == "settled"


def test_late_plan_api_error_replays_same_terminal_attempt(tmp_path):
    import threading
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    calls = []
    release = threading.Event()

    class GatedAPIError:
        """Holds the paid call open until the test releases it, so the poll
        window is GUARANTEED to expire while the call is still in flight.  The
        previous 0.05s-sleep-vs-0.01s-window race flaked the in_flight
        assertion on slow CI hosts (same event gate as
        test_replayed_late_review_does_not_charge_same_context_twice)."""

        def chat(self, **kwargs):
            calls.append(kwargs)
            assert release.wait(10), "test never released the gated review call"
            raise RuntimeError("provider ended the paid request")

    ctx = SimpleNamespace(task_id="late-plan-error", event_queue=None, pending_events=[])
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="late-plan-error",
        retry_key="plan-envelope/cycle-1",
    )
    slot = ReviewSlot(
        slot_id="slot_a", model="same/model", timeout_sec=0.01,
        transport_timeout_sec=10,
    )
    first = run_review_request(
        request, slots=[slot], drive_root=tmp_path, llm=GatedAPIError(), usage_ctx=ctx,
    )
    assert first.actors[0]["operation_state"] == "in_flight"
    operation_id = first.actors[0]["operation_id"]
    release.set()
    # The settled-attempt cache is written in the same critical section that
    # retires the active attempt, so draining BOTH facts under _ACTIVE_LOCK is
    # the event that makes the replay below deterministic (same wait as
    # test_replayed_late_review_does_not_charge_same_context_twice).
    key = _attempt_key(request, slot)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            settled = key in getattr(ctx, "_review_settled_attempts", {})
            active = key in _ACTIVE
        if settled and not active:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("late plan API error was not retained for reconciliation")

    second = run_review_request(
        request, slots=[slot], drive_root=tmp_path, llm=GatedAPIError(), usage_ctx=ctx,
    )
    assert len(calls) == 1
    assert second.actors[0]["status"] == "error"
    assert second.actors[0]["operation_id"] == operation_id
    assert second.actors[0]["operation_state"] == "late_settled"
    assert "provider ended the paid request" in second.actors[0]["error"]


def test_review_paid_stamp_is_write_ahead_of_a_slow_worker(tmp_path):
    import threading
    from types import SimpleNamespace
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    order = []
    release = threading.Event()

    class GatedLLM:
        """Holds the transport call open until the test releases it, so the
        poll window is GUARANTEED to expire while the worker is in flight.
        The previous 0.08s-sleep-vs-0.02s-window race flaked the in_flight
        assertion on slow CI hosts (same event gate as
        test_replayed_late_review_does_not_charge_same_context_twice)."""

        def chat(self, **_kwargs):
            order.append("transport")
            assert release.wait(10), "test never released the gated review call"
            return {"content": '{"verdict":"PASS","findings":[],"summary":"ok"}'}, {}

    ctx = SimpleNamespace(
        task_id="paid-before-worker", event_queue=None, pending_events=[],
        _review_paid_stamp=lambda: order.append("paid"),
    )
    request = ReviewRequest(surface="scope", goal="review", task_id="paid-before-worker")
    slot = ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=0.02)
    result = run_review_request(
        request, slots=[slot], drive_root=tmp_path, llm=GatedLLM(), usage_ctx=ctx,
    )
    assert result.actors[0]["operation_state"] == "in_flight"
    release.set()
    # Drain the released worker before judging the order: the caller can
    # return in_flight before the worker thread has entered chat at all, so
    # order[1] raced an IndexError; the drain proves the transport entry
    # exists.  ``order`` is append-only, so the paid-before-transport contract
    # is judged unchanged (same drain as
    # test_explicit_retry_key_joins_worker_after_prompt_history_changes).
    key = _attempt_key(request, slot)
    deadline = time.monotonic() + 5.0
    active = True
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            active = key in _ACTIVE
        if not active:
            break
        time.sleep(0.01)
    assert not active
    assert order[0] == "paid"
    assert order[1] == "transport"


def test_replayed_late_review_does_not_charge_same_context_twice(tmp_path):
    import threading
    from types import SimpleNamespace
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    calls = []
    release = threading.Event()

    class GatedLLM:
        """Holds the paid call open until the test releases it, so the first
        poll window is GUARANTEED to expire while the call is still in flight.
        The previous 0.06s-sleep-vs-0.02s-window race flaked on slow CI hosts:
        an oversleeping poll wait could observe the settled result and return
        'settled' instead of 'in_flight' (same event gate as
        test_late_review_result_is_replayed_without_a_second_paid_dispatch)."""

        def chat(self, **_kwargs):
            calls.append(1)
            assert release.wait(10), "test never released the gated review call"
            return {"content": '{"verdict":"PASS","findings":[],"summary":"late"}'}, {}

    ctx = SimpleNamespace(
        task_id="paid-replay", event_queue=None, pending_events=[],
        _review_paid_stamp=lambda: calls.append("paid"),
    )
    request = ReviewRequest(surface="scope", goal="review", task_id="paid-replay")
    first = run_review_request(
        request,
        slots=[ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=0.02)],
        drive_root=tmp_path, llm=GatedLLM(), usage_ctx=ctx,
    )
    assert first.actors[0]["operation_state"] == "in_flight"
    release.set()
    # The late result is cached only after the worker has actually settled.  The
    # settled-attempt cache is written in the same critical section that retires
    # the active attempt, so this drain is the event that makes the replay below
    # deterministic (same wait as
    # test_review_worker_does_not_retry_after_its_logical_deadline).
    key = _attempt_key(request, ReviewSlot(slot_id="slot_a", model="same/model"))
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            settled = key in getattr(ctx, "_review_settled_attempts", {})
            active = key in _ACTIVE
        if settled and not active:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("late review worker did not settle into same-context custody")
    second = run_review_request(
        request,
        slots=[ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=0.01)],
        drive_root=tmp_path, llm=GatedLLM(), usage_ctx=ctx,
    )
    assert calls.count("paid") == 1
    assert sum(1 for item in calls if item == 1) == 1
    assert second.actors[0]["operation_state"] == "late_settled"


def test_review_slot_timeout_is_not_used_as_transport_timeout(tmp_path):
    captured = []

    class CapturingLLM:
        def chat(self, **kwargs):
            captured.append(kwargs)
            return {"content": '{"verdict":"PASS","findings":[],"summary":"ok"}'}, {}

    result = run_review_request(
        ReviewRequest(surface="scope", goal="review", task_id="transport-separation"),
        slots=[ReviewSlot(
            slot_id="slot_a", model="same/model", timeout_sec=0.5,
            transport_timeout_sec=17,
        )],
        drive_root=tmp_path,
        llm=CapturingLLM(),
    )
    assert result.aggregate_signal == "PASS"
    assert captured and captured[0]["timeout"] == 17


def test_review_transport_timeout_is_narrowed_by_request_deadline(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone

    captured = []

    class CapturingLLM:
        def chat(self, **kwargs):
            captured.append(kwargs)
            return {"content": '{"verdict":"PASS","findings":[],"summary":"ok"}'}, {}

    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "0")
    deadline = (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()
    run_review_request(
        ReviewRequest(
            surface="scope", goal="review", task_id="deadline-transport",
            deadline_at=deadline,
        ),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=30)],
        drive_root=tmp_path,
        llm=CapturingLLM(),
    )
    assert captured and 0 < captured[0]["timeout"] <= 5


def test_api_chat_retry_recomputes_transport_window(tmp_path, monkeypatch):
    from ouroboros import review_execution

    captured = []
    transport_windows = iter((479.96, 1.0))

    class RetryingLLM:
        def chat(self, **kwargs):
            captured.append(kwargs["timeout"])
            if len(captured) == 1:
                return {"content": ""}, {}
            return {"content": '{"verdict":"PASS","findings":[],"summary":"ok"}'}, {}

    monkeypatch.setattr(
        review_execution,
        "review_transport_timeout",
        lambda *_args: next(transport_windows),
    )
    result = run_review_request(
        ReviewRequest(surface="scope_review", goal="retry", task_id="retry-timeout"),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path,
        llm=RetryingLLM(),
    )

    assert result.aggregate_signal == "PASS"
    assert captured == [479.96, 1.0]


def test_direct_anthropic_route_keeps_provider_default_transport(tmp_path):
    captured = []

    class CapturingLLM:
        def chat(self, **kwargs):
            captured.append(kwargs)
            return {"content": '{"verdict":"PASS","findings":[],"summary":"ok"}'}, {}

    run_review_request(
        ReviewRequest(surface="scope", goal="review", task_id="anthropic-timeout"),
        slots=[ReviewSlot(slot_id="slot_a", model="anthropic::claude-test", timeout_sec=0.5)],
        drive_root=tmp_path,
        llm=CapturingLLM(),
    )
    assert captured and captured[0]["timeout"] is None


def test_late_error_is_not_cached_as_a_permanent_review_verdict(tmp_path):
    import threading
    from types import SimpleNamespace
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    calls = []
    release = threading.Event()

    class ErrorThenSuccess:
        def chat(self, **_kwargs):
            calls.append(1)
            if len(calls) == 1:
                # Gated: the first poll window (0.005s) is GUARANTEED to
                # expire while the call is still in flight.  The previous
                # 0.03s-sleep-vs-0.005s-window race flaked the in_flight
                # assertion on slow CI hosts (same event gate as
                # test_late_agent_session_preflight_failure_can_retry).
                assert release.wait(10), "test never released the gated review call"
                raise RuntimeError("transient provider failure")
            return {"content": '{"verdict":"PASS","findings":[],"summary":"ok"}'}, {}

    ctx = SimpleNamespace(task_id="late-error", event_queue=None, pending_events=[])
    request = ReviewRequest(surface="plan_review", goal="review", task_id="late-error")
    slot = ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=0.005)
    first = run_review_request(
        request,
        slots=[slot],
        drive_root=tmp_path,
        llm=ErrorThenSuccess(),
        usage_ctx=ctx,
    )
    assert first.actors[0]["operation_state"] == "in_flight"
    release.set()
    # The first physical operation must settle before the retry is admitted.
    # A transient error is NOT retained for replay, so the deterministic
    # signal that the retry may dispatch is the attempt leaving _ACTIVE.  The
    # previous fixed 0.05s sleep raced the settle on slow CI hosts: the second
    # run then JOINED the still-active attempt instead of retrying (same drain
    # as test_late_agent_session_preflight_failure_can_retry).
    key = _attempt_key(request, slot)
    deadline = time.monotonic() + 5.0
    active = True
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            active = key in _ACTIVE
        if not active:
            break
        time.sleep(0.01)
    assert not active
    second = run_review_request(
        request,
        slots=[ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=0.5)],
        drive_root=tmp_path,
        llm=ErrorThenSuccess(),
        usage_ctx=ctx,
    )
    assert len(calls) >= 2
    assert second.actors[0]["status"] == "ok"


def test_late_agent_session_failure_is_replayed_without_a_second_run(tmp_path, monkeypatch):
    import threading
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewCoordinator

    calls = []
    release = threading.Event()

    def gated_terminal_failure(
        self, request, slot, *, operation_id="", retry_state=None,
        logical_deadline_monotonic=None,
    ):
        # Blocks until the test releases it, so the first poll window is
        # GUARANTEED to expire while the delegated run is still in flight.
        # The previous fixed 0.03s sleep raced the 0.005s poll window on slow
        # CI hosts: an oversleeping poll wait could observe the settled result
        # and surface 'settled' at the in_flight assertion below (same event
        # gate as test_replayed_late_review_does_not_charge_same_context_twice).
        calls.append(operation_id)
        assert release.wait(10), "test never released the gated session run"
        actor = self._error_actor(
            request, slot, "delegated run settled failed",
            operation_id=operation_id,
        )
        actor.usage = {"delegated_run_started": True, "delegated_run_id": "run-1"}
        return actor

    monkeypatch.setattr(ReviewCoordinator, "_run_slot", gated_terminal_failure)
    ctx = SimpleNamespace(task_id="late-session-error", event_queue=None, pending_events=[])
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="late-session-error",
        session_root=str(tmp_path), session_task="review this tree",
    )
    slot = ReviewSlot(
        slot_id="slot_a", model="same/model", timeout_sec=0.005,
        route=ReviewRouteKind.AGENT_SESSION,
    )
    first = run_review_request(
        request, slots=[slot], drive_root=tmp_path, llm=FakeLLM(), usage_ctx=ctx,
    )
    assert first.actors[0]["operation_state"] == "in_flight"
    release.set()
    # The settled-attempt cache is written in the same critical section that
    # retires the active attempt, so draining BOTH facts under _ACTIVE_LOCK is
    # the event that makes the replay below deterministic (same wait as
    # test_replayed_late_review_does_not_charge_same_context_twice).
    key = _attempt_key(request, slot)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            settled = key in getattr(ctx, "_review_settled_attempts", {})
            active = key in _ACTIVE
        if settled and not active:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("late session failure did not settle into custody")
    second = run_review_request(
        request, slots=[slot], drive_root=tmp_path, llm=FakeLLM(), usage_ctx=ctx,
    )
    assert len(calls) == 1
    assert second.actors[0]["status"] == "error"
    assert second.actors[0]["operation_state"] == "late_settled"


def test_late_agent_session_preflight_failure_can_retry(tmp_path, monkeypatch):
    import threading
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewCoordinator

    calls = []
    release = threading.Event()

    def preflight_then_success(
        self, request, slot, *, operation_id="", retry_state=None,
        logical_deadline_monotonic=None,
    ):
        calls.append(operation_id)
        if len(calls) == 1:
            # Gated: the first poll window (0.005s) is GUARANTEED to expire
            # while the preflight attempt is still in flight.  The previous
            # fixed 0.02s sleep raced that window on slow CI hosts and could
            # surface 'settled' at the in_flight assertion below (same event
            # gate as
            # test_explicit_retry_key_joins_worker_after_prompt_history_changes).
            assert release.wait(10), "test never released the gated preflight"
            return self._error_actor(
                request, slot, "route unavailable before dispatch",
                operation_id=operation_id,
            )
        actor = self._error_actor(request, slot, "unused", operation_id=operation_id)
        actor.status, actor.error, actor.raw_text = "ok", "", '{"verdict":"PASS","findings":[]}'
        return actor

    monkeypatch.setattr(ReviewCoordinator, "_run_slot", preflight_then_success)
    ctx = SimpleNamespace(task_id="late-session-preflight", event_queue=None, pending_events=[])
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="late-session-preflight",
        session_root=str(tmp_path), session_task="review this tree",
    )
    slot = ReviewSlot(
        slot_id="slot_a", model="same/model", timeout_sec=0.005,
        route=ReviewRouteKind.AGENT_SESSION,
    )
    first = run_review_request(
        request, slots=[slot], drive_root=tmp_path, llm=FakeLLM(), usage_ctx=ctx,
    )
    assert first.actors[0]["operation_state"] == "in_flight"
    release.set()
    # A plain preflight error is retryable, so it leaves NOTHING in the settled
    # cache; the deterministic signal that the retry may dispatch is the
    # attempt leaving _ACTIVE (retired under _ACTIVE_LOCK in
    # _settle_review_attempt).  The previous fixed 0.04s sleep raced that
    # settle on slow CI hosts: the second run then JOINED the still-active
    # attempt instead of dispatching the retry (same drain as
    # test_late_unknown_session_start_restores_exact_pending_invocation).
    key = _attempt_key(request, slot)
    deadline = time.monotonic() + 5.0
    active = True
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            active = key in _ACTIVE
        if not active:
            break
        time.sleep(0.01)
    assert not active
    second = run_review_request(
        request, slots=[ReviewSlot(
            slot_id="slot_a", model="same/model", timeout_sec=0.5,
            route=ReviewRouteKind.AGENT_SESSION,
        )], drive_root=tmp_path, llm=FakeLLM(), usage_ctx=ctx,
    )
    assert len(calls) == 2
    assert second.actors[0]["status"] == "ok"


def test_late_unknown_session_start_restores_exact_pending_invocation(tmp_path, monkeypatch):
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewCoordinator

    calls, paid = [], []

    def pending_then_success(
        self, request, slot, *, operation_id="", retry_state=None,
        logical_deadline_monotonic=None,
    ):
        calls.append(dict(retry_state or {}))
        if len(calls) == 1:
            time.sleep(0.02)
            actor = self._error_actor(request, slot, "start outcome unknown", operation_id=operation_id)
            actor.usage = {"pending_invocation_id": "invocation-1"}
            return actor
        assert retry_state == {"pending_invocation_id": "invocation-1"}
        actor = self._error_actor(request, slot, "unused", operation_id=operation_id)
        actor.status, actor.error, actor.raw_text = "ok", "", '{"verdict":"PASS","findings":[]}'
        return actor

    monkeypatch.setattr(ReviewCoordinator, "_run_slot", pending_then_success)
    ctx = SimpleNamespace(
        task_id="late-session-pending", event_queue=None, pending_events=[],
        _review_paid_stamp=lambda: paid.append("paid"),
    )
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="late-session-pending",
        session_root=str(tmp_path), session_task="review this tree",
    )
    first_slot = ReviewSlot(
        slot_id="slot_a", model="same/model", timeout_sec=0.005,
        route=ReviewRouteKind.AGENT_SESSION,
    )
    first = run_review_request(
        request, slots=[first_slot], drive_root=tmp_path, llm=FakeLLM(), usage_ctx=ctx,
    )
    assert first.actors[0]["operation_state"] == "in_flight"
    # The pending-invocation row is persisted inside the same _ACTIVE_LOCK
    # critical section that retires the active attempt, so draining the key out
    # of _ACTIVE is the deterministic signal that the restored invocation is
    # durably recorded.  The previous fixed 0.04s sleep raced the worker's
    # settle on slow CI hosts: the second run then JOINED the still-active
    # attempt instead of dispatching the restored pending retry (same drain as
    # test_review_worker_does_not_retry_after_its_logical_deadline).
    key = _attempt_key(request, first_slot)
    deadline = time.monotonic() + 5.0
    active = True
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            active = key in _ACTIVE
        if not active:
            break
        time.sleep(0.01)
    assert not active
    second = run_review_request(
        request, slots=[ReviewSlot(
            slot_id="slot_a", model="same/model", timeout_sec=0.5,
            route=ReviewRouteKind.AGENT_SESSION,
        )], drive_root=tmp_path, llm=FakeLLM(), usage_ctx=ctx,
    )
    assert calls == [{}, {"pending_invocation_id": "invocation-1"}]
    assert paid == ["paid"]
    assert second.actors[0]["status"] == "ok"


def test_review_slots_keep_independent_logical_windows(tmp_path):
    import threading
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    short_release = threading.Event()

    class GatedShortLLM:
        """The short slot's call stays open until the test releases it, so its
        0.05s window is GUARANTEED to expire (in_flight) while the long slot
        answers inside its own independent window.  The previous 0.25s-sleep-
        vs-0.05s-window margin was NOT discriminating on a loaded CI host: the
        heal wave measured a 0.207s poll oversleep there, which eats the whole
        0.2s absolute margin (same event gate as
        test_replayed_late_review_does_not_charge_same_context_twice)."""

        def chat(self, **kwargs):
            if kwargs.get("model") == "short/model":
                assert short_release.wait(10), "test never released the short slot"
            return {"content": '{"verdict":"PASS","findings":[],"summary":"ok"}'}, {}

    request = ReviewRequest(surface="scope", goal="review", task_id="independent-windows")
    short_slot = ReviewSlot(slot_id="short", model="short/model", timeout_sec=0.05)
    result = run_review_request(
        request,
        slots=[
            short_slot,
            ReviewSlot(slot_id="long", model="long/model", timeout_sec=0.5),
        ],
        drive_root=tmp_path,
        llm=GatedShortLLM(),
    )
    rows = {actor["slot_id"]: actor for actor in result.actors}
    assert rows["short"]["operation_state"] == "in_flight"
    assert rows["long"]["status"] == "ok"
    short_release.set()
    # Drain the released short worker before teardown (same drain as
    # test_explicit_retry_key_joins_worker_after_prompt_history_changes).
    key = _attempt_key(request, short_slot)
    deadline = time.monotonic() + 5.0
    active = True
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            active = key in _ACTIVE
        if not active:
            break
        time.sleep(0.01)
    assert not active


def test_review_substrate_preserves_explicit_zero_budget_rails(tmp_path):
    from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

    captured = []

    class ScopeCapturingLLM:
        def chat(self, **_kwargs):
            captured.append(current_usage_scope())
            return {
                "content": json.dumps({"verdict": "PASS", "findings": [], "summary": "ok"}),
            }, {}

    with usage_scope(UsageScope(
        drive_root=tmp_path,
        task_id="zero-rail",
        root_task_id="zero-rail",
        global_limit_usd=0.0,
        root_limit_usd=0.0,
    )):
        result = run_review_request(
            ReviewRequest(surface="task_acceptance", goal="review", task_id="zero-rail"),
            slots=[ReviewSlot(slot_id="slot", model="test/model")],
            drive_root=tmp_path,
            llm=ScopeCapturingLLM(),
        )

    assert result.aggregate_signal == "PASS"
    assert len(captured) == 1
    assert captured[0].global_limit_usd == 0.0
    assert captured[0].root_limit_usd == 0.0


def test_triad_actor_records_preserve_review_refs():
    parsed = parse_model_review_results({
        "results": [{
            "model": "m1",
            "text": "[{\"item\":\"x\",\"verdict\":\"PASS\",\"severity\":\"advisory\",\"reason\":\"ok\"}]",
            "prompt_ref": {"manifest_ref": {"path": "prompt.json"}},
            "response_ref": {"manifest_ref": {"path": "response.json"}},
        }]
    })

    actor = parsed.actor_records[0].to_dict()
    assert actor["prompt_ref"]["manifest_ref"]["path"] == "prompt.json"
    assert actor["response_ref"]["manifest_ref"]["path"] == "response.json"


def test_scope_review_result_preserves_substrate_refs(tmp_path, monkeypatch):
    from ouroboros.tools import scope_review
    from ouroboros.tools.review_helpers import build_scope_actor_record

    class FakeScopeLLM:
        def chat(self, **kwargs):
            rows = [
                {
                    "item": item,
                    "verdict": "PASS",
                    "severity": "advisory",
                    "reason": "Fixture confirms scope substrate refs.",
                }
                for item in sorted(scope_review._SCOPE_REQUIRED_ITEMS)
            ]
            return {"content": json.dumps(rows)}, {"prompt_tokens": 10, "completion_tokens": 5}

    ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="scope-task", pending_events=[])
    monkeypatch.setattr(scope_review, "LLMClient", lambda: FakeScopeLLM())
    monkeypatch.setattr(scope_review, "_build_scope_prompt", lambda *a, **k: ("scope prompt", None))
    monkeypatch.setattr(scope_review, "_get_scope_model", lambda: "test-scope-model")
    # This test isolates durable substrate refs, not the separate P3 authority
    # floor; give its synthetic reviewer explicit >=1M capability evidence.
    monkeypatch.setattr(scope_review, "_scope_window",
                        lambda _model, **_k: scope_review.ReviewerWindow(1_000_000, "confirmed"))

    result = scope_review.run_scope_review(ctx, "commit message")
    record = build_scope_actor_record(result, fallback_model_id="test-scope-model", slot_id="scope_slot_1")

    assert result.status == "responded"
    assert record["prompt_ref"]["manifest_ref"]["path"]
    assert record["response_ref"]["manifest_ref"]["path"]


# --- v6.87.11: the single review execution seam (Phase 5.1 / 5.2) -------------

# Byte-level golden for the api_chat prompt rendering, captured by running the
# generator below against the PRISTINE pre-seam substrate (v6.87.5, ca76d76).
# The seam refactor is a pure move: every digest must still match. Regenerate
# ONLY together with a deliberate, reviewed prompt change:
#     for request, slot in _seam_prompt_cases():
#         sha256(json.dumps(_request_messages(request, slot),
#                           ensure_ascii=False, sort_keys=True).encode())
# The two task_acceptance digests (indexes 2-3) were re-pinned DELIBERATELY when
# D-Q5 added the evidence-ref vocabulary line to the acceptance criteria_key —
# a one-time cache invalidation of the stable governance segment — and re-pinned
# once more when that same line was corrected to state the real claim-id binding
# (a claim counts only while `acceptance_support_refs` shows it supported), and a
# THIRD time when section refs were narrowed to host-attested exhibits (the
# agent's own reasoning_notes/candidate_answers and task_contract stopped
# resolving, so the prompt must stop advertising them), and a FOURTH time when
# receipt refs started enumerating the packet's verification_receipts exhibit
# rows (only a green pass/observed receipt resolves, so the prompt says so).
# Only the acceptance surface moves: the four non-acceptance digests are unchanged.
_PRE_SEAM_PROMPT_DIGESTS = [
    "0261c7c7fe477ad7f8901a28bee1ad23905d40c3c62825d2bc406ecd9ca37f82",
    "9cf4de6f66001c3b4cec7fdd3d8552ecf83fc886004a7020e98a4c28c022c4e3",
    "bc49f3bf1d7273c6cfa3d882dc5738e379f3dcc7af37a15a3686a30f89b8b355",
    "674971a10ccd95822cf790f5038eaf77824d38996f52c61a30a93f8666a324d3",
    "fca0f9401e544e371338f20effa6206db783e7098ff4d11ee2a980ebbe81ecb0",
    "fca0f9401e544e371338f20effa6206db783e7098ff4d11ee2a980ebbe81ecb0",
]


def _seam_prompt_cases():
    generic = ReviewRequest(
        surface="commit_review",
        goal="Judge the staged change.\nSecond line.",
        scope="ouroboros/review_substrate.py",
        subject="diff --git a/x b/x\n+1\n",
        evidence={"files": ["a.py", "b.py"], "nested": {"k": [1, 2, {"deep": "ünicode"}]}},
        evidence_refs=[{"kind": "blob", "sha256": "deadbeef"}],
        checklist="- one\n- two",
        policy={"hardness": "hard_gate", "min_successful_slots": 2},
        task_id="task-1",
    )
    acceptance = ReviewRequest(
        surface="task_acceptance",
        goal="Did the agent finish?",
        scope="",
        subject="the answer",
        evidence={"receipts": [{"tool": "bash", "ok": True}]},
        evidence_refs=[],
        checklist="- criteria",
        policy={
            "classify_outcome_tier": True,
            "require_criterion_evidence": True,
            "hardness": "advisory_visible",
            "min_successful_slots": 1,
        },
        task_id="task-2",
    )
    prebuilt = ReviewRequest(
        surface="scope_review",
        goal="Review the staged change and context above. Output ONLY a JSON array.",
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "STABLE",
                     "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                    {"type": "text", "text": "DYNAMIC"},
                ],
            },
            {"role": "user", "content": "Review the staged change and context above."},
        ],
        task_id="task-3",
        call_type="scope_review",
        max_tokens=64000,
        temperature=0.2,
        no_proxy=True,
    )
    slots = [
        ReviewSlot(slot_id="slot_1", model="anthropic/claude-x", effort="high", role_hint="commit reviewer"),
        ReviewSlot(slot_id="slot_2", model="openai/gpt-x", effort="medium", role_hint=""),
    ]
    for request in (generic, acceptance, prebuilt):
        for slot in slots:
            yield request, slot


def test_api_chat_executor_renders_pre_seam_bytes_exactly():
    """5.2: moving prompt assembly behind the seam is a PURE move — the executor
    reproduces the pre-seam bytes and cache markers exactly."""
    import hashlib

    from ouroboros.review_execution import (
        ApiChatReviewExecutor,
        ReviewAssignment,
        _request_messages,
    )

    digests = []
    for request, slot in _seam_prompt_cases():
        messages = ApiChatReviewExecutor(ReviewAssignment(request=request, slot=slot)).messages
        # Same SSOT renderer, same bytes.
        assert messages == _request_messages(request, slot)
        blob = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
        digests.append(hashlib.sha256(blob).hexdigest())
    assert digests == _PRE_SEAM_PROMPT_DIGESTS

    # Cache segmentation survives verbatim: exactly one marked governance block
    # and one marked task-stable block, mutable tail unmarked, slot label last.
    request, slot = next(iter(_seam_prompt_cases()))
    system_blocks = ApiChatReviewExecutor(
        ReviewAssignment(request=request, slot=slot)
    ).messages[0]["content"]
    assert [bool(block.get("cache_control")) for block in system_blocks] == [True, True]


def test_prompt_record_keeps_request_slot_messages_shape(tmp_path):
    """The durable prompt record still carries request/slot/messages, in order,
    with the route's own projection supplying the last key."""
    llm = FakeLLM()
    run_review_request(
        ReviewRequest(surface="scope", goal="g", task_id="prompt-shape"),
        slots=[ReviewSlot(slot_id="s1", model="m")],
        drive_root=tmp_path,
        llm=llm,
    )
    import gzip

    blobs = sorted((tmp_path / "observability" / "blobs").glob("*.json.gz"))
    payloads = [json.loads(gzip.open(path, "rb").read().decode("utf-8")) for path in blobs]
    prompt_payloads = [p for p in payloads if isinstance(p, dict) and "messages" in p]
    assert prompt_payloads
    assert list(prompt_payloads[0]) == ["messages", "request", "slot"]  # sorted on disk
    assert prompt_payloads[0]["slot"]["route"] == "api_chat"


def test_slot_prompt_is_rendered_once_per_slot(tmp_path, monkeypatch):
    """5.2: the prompt record and both permitted physical sends share ONE lazy
    rendering — the substrate never re-assembles the pack per attempt."""
    import ouroboros.review_execution as rx

    calls = {"n": 0}
    real = rx._request_messages

    def _counted(request, slot):
        calls["n"] += 1
        return real(request, slot)

    # Patch the OWNER module: the api_chat executor renders through it.
    monkeypatch.setattr(rx, "_request_messages", _counted)

    class RepairLLM:
        def __init__(self):
            self.sends = 0

        def chat(self, **kwargs):
            self.sends += 1
            if self.sends == 1:
                return {"content": "not json at all"}, {}
            return {"content": json.dumps({
                "verdict": "PASS", "findings": [], "summary": "ok",
                "outcome_tier": "solved", "completion_coach": "",
            })}, {}

    llm = RepairLLM()
    run_review_request(
        ReviewRequest(
            surface="task_acceptance", goal="g", subject="done",
            policy={"classify_outcome_tier": True, "min_successful_slots": 1},
            task_id="lazy-render",
        ),
        slots=[ReviewSlot(slot_id="s1", model="m")],
        drive_root=tmp_path,
        llm=llm,
    )
    assert llm.sends == 2        # the repair resend still happens
    assert calls["n"] == 1       # rendered once for the record AND both sends


def test_undeliverable_route_is_a_typed_refusal_not_a_fallback(tmp_path):
    """5.1: a route that cannot deliver THIS slot (here: an agent_session slot
    whose surface supplied no session root/task) refuses on its own slot. It
    never silently falls back to another transport, and it never reaches a
    chat client."""
    from ouroboros.review_execution import (
        ReviewAssignment,
        ReviewRouteKind,
        ReviewRouteUnavailable,
        _execute_slot_attempt,
    )

    request = ReviewRequest(surface="scope", goal="g", task_id="route")
    slot = ReviewSlot(slot_id="s1", model="m", timeout_sec=5, route=ReviewRouteKind.AGENT_SESSION)
    assignment = ReviewAssignment(request=request, slot=slot)
    llm = FakeLLM()
    try:
        _execute_slot_attempt(assignment, llm=llm)
    except ReviewRouteUnavailable as exc:
        assert "agent_session" in str(exc)
    else:  # pragma: no cover - the seam must refuse
        raise AssertionError("unimplemented route must raise ReviewRouteUnavailable")
    assert llm.calls == []

    # The refusal is contained before dispatch: the panel stays honest and free.
    result = run_review_request(request, slots=[slot], drive_root=tmp_path, llm=llm)
    assert result.aggregate_signal == "DEGRADED"
    assert result.actors[0]["status"] == "not_dispatched"
    assert llm.calls == []


def test_route_kinds_carry_no_harness_names():
    """Part IV: only api_chat and agent_session ever exist in the core."""
    from ouroboros.review_execution import ReviewRouteKind

    assert {kind.value for kind in ReviewRouteKind} == {"api_chat", "agent_session"}
    assert ReviewSlot(slot_id="s", model="m").route is ReviewRouteKind.API_CHAT


def test_default_drive_root_is_the_absolute_config_root_never_cwd_relative(tmp_path, monkeypatch):
    """ISO-DRIP regression: the coordinator's shipped default was the RELATIVE
    ``../data`` — with any cwd under a repo/ that names the live data root's
    sibling, so default-constructed coordinators dripped synthetic review
    records into live observability (or, on trees with the absolute-root
    guard, silently LOST them into empty refs). The default must resolve to
    the absolute config SSOT: records really land there, and nothing is ever
    created relative to the cwd."""
    import ouroboros.config as config

    apphome = tmp_path / "apphome"
    repo = apphome / "repo"
    repo.mkdir(parents=True)
    configured = tmp_path / "configured_data"
    monkeypatch.setattr(config, "DATA_DIR", configured)
    monkeypatch.chdir(repo)

    class OkLLM:
        def chat(self, **kwargs):
            return {"content": "[]"}, {"prompt_tokens": 2, "completion_tokens": 1}

    result = run_review_request(
        ReviewRequest(surface="multi_model_review", goal="iso-drip probe", task_id="iso-drip"),
        slots=[ReviewSlot(slot_id="slot_1", model="api/m", timeout_sec=10)],
        drive_root=None,  # the shipped default under test
        llm=OkLLM(),
    )
    actor = result.actors[0]
    # The records were REALLY written (not swallowed into empty refs) ...
    assert actor["prompt_ref"].get("manifest_ref", {}).get("path")
    assert actor["response_ref"].get("manifest_ref", {}).get("path")
    # ... into the configured absolute root ...
    assert (configured / "observability").is_dir()
    # ... and never cwd-relative: no ../data sibling, nothing under the cwd.
    assert not (apphome / "data").exists()
    assert list(repo.iterdir()) == []
