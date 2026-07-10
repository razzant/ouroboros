import json
import time
from types import SimpleNamespace

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
    assert "FULL goal/spec narrative" in prompt
    assert "affected components/surfaces" in prompt
    assert "per-criterion evidence" in prompt

    # A non-tier surface keeps the lean key list (no tier keys).
    plain = _render_prompt(
        ReviewRequest(surface="scope", goal="g", task_id="t"),
        ReviewSlot(slot_id="a", model="m"),
    )
    plain_keys = next(line for line in plain.splitlines() if line.startswith("Return JSON with keys:"))
    assert "outcome_tier" not in plain_keys


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


class ErrorLLM:
    def chat(self, **kwargs):
        raise RuntimeError("provider exploded")


class HangingLLM:
    def chat(self, **kwargs):
        time.sleep(0.2)
        return {"content": "{\"verdict\":\"PASS\",\"findings\":[],\"summary\":\"late\"}"}, {}


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


def test_parse_degraded_slot_does_not_poison_quorum_but_actor_error_does(tmp_path):
    """T1 (v6.35.0): a single unparseable/DEGRADED-verdict slot must NOT poison a
    clean 2-of-3 PASS quorum (the old over-degrading bug); a participation fault
    (errored/empty slot) still fail-closes to DEGRADED."""
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
    assert bad.aggregate_signal == "DEGRADED"
    assert bad.degraded is True


class PassNoTierLLM:
    """PASS verdict but NO outcome_tier — the non-compliant reviewer the required-
    tier contract must catch (a tier-less PASS must not aggregate to a clean PASS)."""

    def chat(self, **kwargs):
        return {"content": json.dumps({"verdict": "PASS", "findings": [], "summary": "ok"})}, {}


class PassWithTierLLM:
    def chat(self, **kwargs):
        body = {"verdict": "PASS", "outcome_tier": "solved", "completion_coach": "ship", "findings": [], "summary": "ok"}
        return {"content": json.dumps(body)}, {}


class PassTierNoCoachLLM:
    """PASS with a valid outcome_tier but EMPTY completion_coach — still
    non-responsive to the required-tier contract (both keys required)."""

    def chat(self, **kwargs):
        body = {"verdict": "PASS", "outcome_tier": "solved", "completion_coach": "", "findings": [], "summary": "ok"}
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
            "findings": [], "summary": "ok",
        })}, {}


def test_contract_degraded_pass_does_not_poison_capsule(tmp_path):
    """v6.36.0 round-2 scope finding: a verdict=PASS actor that VIOLATES the
    required tier/coach contract is demoted to non-contributing (signal->DEGRADED),
    so it can't feed its blocked tier / finding into the clean quorum PASS capsule —
    the live path the DEGRADED-verdict-only test did not cover."""
    from ouroboros.review_substrate import aggregate_outcome_tier, build_improvement_capsule
    slots = [ReviewSlot(slot_id=f"s{i}", model=f"m-{i}") for i in range(3)]
    req = ReviewRequest(
        surface="task_acceptance", goal="g", subject="done",
        policy={"classify_outcome_tier": True, "min_successful_slots": 2}, task_id="t",
    )
    res = run_review_request(req, slots=slots, drive_root=tmp_path, llm=ContractDegradedPassLLM())
    assert res.aggregate_signal == "PASS"          # the two contract-valid solved PASS reach quorum
    assert aggregate_outcome_tier(res) == "solved"  # the blocked contract-degraded PASS is excluded
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
    is honored but records single_reviewer_no_diversity durably (field + degraded
    reason) on EVERY surface — so a one-slot acceptance review can never quietly
    look like an ordinary multi-reviewer PASS. A 3-slot run does not."""
    one = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="g", subject="d",
                      policy={"min_successful_slots": 1}, task_id="t"),
        slots=[ReviewSlot(slot_id="s0", model="m-0")],
        drive_root=tmp_path, llm=PassWithTierLLM(),
    )
    assert one.single_reviewer_no_diversity is True
    assert "single_reviewer_no_diversity" in one.degraded_reasons

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

    ctx = NS(drive_root=str(tmp_path), task_id="t")
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

    ctx = NS(drive_root=str(tmp_path), task_id="t")
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

    ctx = NS(drive_root=str(tmp_path), task_id="t")
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


def test_review_substrate_persists_error_actor_response_ref(tmp_path):
    result = run_review_request(
        ReviewRequest(surface="scope", goal="review diff", task_id="task-error"),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path,
        llm=ErrorLLM(),
    )

    actor = result.actors[0]
    assert actor["status"] == "error"
    assert actor["prompt_ref"]["manifest_ref"]["path"]
    assert actor["response_ref"]["manifest_ref"]["path"]
    manifest = json.loads(open(actor["response_ref"]["manifest_ref"]["path"], encoding="utf-8").read())
    assert manifest["call_type"] == "scope_review_error"
    assert manifest["status"] == "error"


def test_review_substrate_persists_timeout_actor_refs(tmp_path):
    result = run_review_request(
        ReviewRequest(surface="scope", goal="review diff", task_id="task-timeout"),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=0.01)],
        drive_root=tmp_path,
        llm=HangingLLM(),
    )

    actor = result.actors[0]
    assert actor["status"] == "error"
    assert "Timeout after" in actor["error"]
    assert actor["prompt_ref"]["manifest_ref"]["path"]
    assert actor["response_ref"]["manifest_ref"]["path"]


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

    result = scope_review.run_scope_review(ctx, "commit message")
    record = build_scope_actor_record(result, fallback_model_id="test-scope-model", slot_id="scope_slot_1")

    assert result.status == "responded"
    assert record["prompt_ref"]["manifest_ref"]["path"]
    assert record["response_ref"]["manifest_ref"]["path"]
