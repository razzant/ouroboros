"""The commit gate's cold-start density rung (owner decision 2026-09-05, answer 1 = A).

The commit gate gets the SAME cold-start tokenizer-density rung the packed deep
self-review has (``capability_evidence.cold_start_density_probe``), at the pre-dispatch
admission seam both packets share (``review_admission``):

- scope ladder: a cold store, a >=1M scope route and the owed-in-full required
  set refused at the floor cap -> exactly ONE bounded probe send on the exact
  model, the witness recorded, the pack rebuilt ONCE and assembled;
- a fresh exact-model witness -> no probe, the existing refusal path unchanged;
- a probe the paid ledger refuses -> typed disclosure (ladder step + review
  event), the existing refusal path, no crash;
- a pack that fits -> no probe (never on every commit);
- triad fit: the same rung before the degradation ladder, bounded, and never
  without a ctx (no drive root to record a witness on).
"""
from __future__ import annotations

import pathlib
import subprocess
from types import SimpleNamespace
from unittest import mock

import pytest

from ouroboros import capability_evidence as ce
from ouroboros.capability_evidence import DENSITY_PROBE_EFFORT, DENSITY_PROBE_MAX_TOKENS
from ouroboros.reviewer_window import ReviewerWindow
from ouroboros.tools import review_admission as admission
from ouroboros.tools import scope_review as sr
from ouroboros.tools.review_helpers import DENSITY_PROBE_SAMPLE_CHARS
from ouroboros.usage_accounting import BudgetExceeded

SCOPE_MODEL = "openai/gpt-5.6-terra"
WINDOW = 1_050_000
# Cold floor cap at a 1,050,000-token window: (1,050,000 - 100,000) / 1.65.
COLD_CAP = int((WINDOW - 100_000) / ce.COLD_START_TOKEN_DENSITY)
# Exact-model witness at 0.9 (x1.05 safety) admits the margin-bounded 795,000.
MEASURED_CAP = WINDOW - 100_000 - 155_000
# An owed-in-full artifact (prompts/) between the two caps, in chars/4 tokens.
OWED_IN_FULL_TOKENS = 650_000


def _git(repo: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=T", *args],
                   cwd=str(repo), capture_output=True, check=True)


def _repo(tmp_path: pathlib.Path, required_tokens: int) -> pathlib.Path:
    """A repo whose UNCHANGED ``prompts/`` artifacts are owed in full: together
    they do not fit the cold floor cap but fit the measured cap of a 1M reviewer
    (each below the atlas per-file cap, so only the INPUT cap refuses them)."""
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "CHECKLISTS.md").write_text(
        "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8")
    (repo / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
    (repo / "BIBLE.md").write_text("constitution\n", encoding="utf-8")
    (repo / "prompts").mkdir()
    for index in range(3):
        (repo / "prompts" / f"owed_{index}.md").write_text(
            "x" * (required_tokens * 4 // 3), encoding="utf-8")
    (repo / "ok.py").write_text("print(1)\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    (repo / "ok.py").write_text("print(2)\n", encoding="utf-8")
    _git(repo, "add", ".")
    return repo


@pytest.fixture
def drive(tmp_path, monkeypatch):
    root = tmp_path / "drive"
    (root / "state").mkdir(parents=True)
    # The cap is computed through review_drive_root(None) -> config.DATA_DIR;
    # the probe records on ctx.drive_root: one root for both, as in production.
    monkeypatch.setattr("ouroboros.config.DATA_DIR", root)
    ce._DENSITY_MEMO.clear()
    return root


@pytest.fixture
def scope_window(monkeypatch):
    monkeypatch.setattr(sr, "_scope_window",
                        lambda _m, **_k: ReviewerWindow(window_tokens=WINDOW, status="confirmed"))
    assert sr._effective_scope_input_limit(scope_model=SCOPE_MODEL) == COLD_CAP


def _ctx(repo: pathlib.Path, drive: pathlib.Path, progress: list) -> SimpleNamespace:
    return SimpleNamespace(
        repo_dir=repo, drive_root=drive, task_id="commit-gate-test",
        emit_progress_fn=progress.append, pending_events=[],
    )


def _probe_chat(calls: list, density: float = 0.9):
    def chat(llm, **kwargs):
        calls.append(kwargs)
        chars = sum(len(m["content"]) for m in kwargs["messages"])
        return {"content": "OK"}, {"prompt_tokens": int(chars / 4 * density), "cost": 0.0}
    return chat


def test_cold_store_owed_in_full_set_probes_once_records_witness_and_assembles(tmp_path, drive, scope_window):
    repo = _repo(tmp_path, OWED_IN_FULL_TOKENS)
    progress: list = []
    calls: list = []
    with mock.patch("ouroboros.llm_observability.chat_observed", side_effect=_probe_chat(calls)):
        prepared, final = admission.prepare_scope_review(
            _ctx(repo, drive, progress), "test commit", scope_model=SCOPE_MODEL)

    assert final is None and prepared is not None, getattr(final, "block_message", final)
    assert [c["call_type"] for c in calls] == [admission.DENSITY_PROBE_CALL_TYPE]
    probe = calls[0]
    assert probe["model"] == SCOPE_MODEL and probe["tools"] is None
    assert probe["max_tokens"] == DENSITY_PROBE_MAX_TOKENS == 256
    assert probe["reasoning_effort"] == DENSITY_PROBE_EFFORT == "low"
    sample = probe["messages"][1]["content"]
    assert sample.startswith("### prompts/owed_"), "the sample is the refused required rows first"
    assert len(sample) <= DENSITY_PROBE_SAMPLE_CHARS + len("### prompts/owed_0.md\n\n")
    density, source = ce.resolve_review_token_density(drive, SCOPE_MODEL)
    assert source == "measured" and density < ce.COLD_START_TOKEN_DENSITY
    assert sr._effective_scope_input_limit(scope_model=SCOPE_MODEL) == MEASURED_CAP
    assert all(f"prompts/owed_{i}.md" in prepared["prompt"] for i in range(3))
    steps = prepared["context_manifest"]["ladder_steps"]
    assert steps[-1] == {"step": "density_probe", "model": SCOPE_MODEL, "outcome": "measured", "rebuilt": True}
    assert not any(s.get("unassembled_required") for s in steps), "the rebuilt trace is the assembled one"
    assert any("bounded probe" in p for p in progress) and any("Token density for" in p for p in progress)


def test_cold_store_probe_discloses_one_review_event(tmp_path, drive, scope_window):
    repo = _repo(tmp_path, OWED_IN_FULL_TOKENS)
    ctx = _ctx(repo, drive, [])
    with mock.patch("ouroboros.llm_observability.chat_observed", side_effect=_probe_chat([])):
        admission.prepare_scope_review(ctx, "test commit", scope_model=SCOPE_MODEL)
    events = [e for e in ctx.pending_events if e.get("type") == admission.DENSITY_PROBE_EVENT]
    assert len(events) == 1
    assert events[0]["surface"] == "scope_review" and events[0]["model"] == SCOPE_MODEL
    assert events[0]["outcome"] == "measured" and events[0]["task_id"] == "commit-gate-test"


def test_fresh_exact_model_witness_never_probes_and_keeps_the_refusal(tmp_path, drive, scope_window):
    repo = _repo(tmp_path, OWED_IN_FULL_TOKENS)
    # A fresh witness that still refuses the set: measured 1.6 x 1.05 = 1.68.
    ce.record_token_density(drive, SCOPE_MODEL, prompt_chars=400_000, prompt_tokens=160_000)
    assert ce.resolve_review_token_density(drive, SCOPE_MODEL)[1] == "measured"
    ctx = _ctx(repo, drive, [])
    calls: list = []
    with mock.patch("ouroboros.llm_observability.chat_observed", side_effect=_probe_chat(calls)):
        prepared, final = admission.prepare_scope_review(ctx, "test commit", scope_model=SCOPE_MODEL)

    assert calls == [], "a warm store must not spend a probe"
    assert prepared is None and final.blocked and final.status == "fixed_overflow"
    assert "prompts/owed_" in final.block_message
    assert not [s for s in final.context_manifest["ladder_steps"] if s["step"] == "density_probe"]
    assert not [e for e in ctx.pending_events if e.get("type") == admission.DENSITY_PROBE_EVENT]


def test_budget_refused_probe_is_a_typed_disclosure_on_the_existing_refusal(tmp_path, drive, scope_window):
    repo = _repo(tmp_path, OWED_IN_FULL_TOKENS)
    progress: list = []
    ctx = _ctx(repo, drive, progress)
    calls: list = []

    def refused(llm, **kwargs):
        calls.append(kwargs)
        raise BudgetExceeded("global budget exhausted", limit_scope="global")

    with mock.patch("ouroboros.llm_observability.chat_observed", side_effect=refused):
        prepared, final = admission.prepare_scope_review(ctx, "test commit", scope_model=SCOPE_MODEL)

    assert len(calls) == 1, "one admission attempt, never a retry"
    assert prepared is None and final.blocked and final.status == "fixed_overflow"
    assert "prompts/owed_" in final.block_message
    steps = final.context_manifest["ladder_steps"]
    assert steps[-1] == {"step": "density_probe", "model": SCOPE_MODEL, "outcome": "budget_refused", "rebuilt": False}
    events = [e for e in ctx.pending_events if e.get("type") == admission.DENSITY_PROBE_EVENT]
    assert len(events) == 1 and events[0]["outcome"] == "budget_refused"
    assert "global budget exhausted" in events[0]["reason"]
    assert any("refused by the budget" in p for p in progress)
    assert ce.resolve_review_token_density(drive, SCOPE_MODEL)[1] == "cold_conservative"


def test_failed_probe_keeps_the_cold_cap_and_the_refusal(tmp_path, drive, scope_window):
    repo = _repo(tmp_path, OWED_IN_FULL_TOKENS)
    ctx = _ctx(repo, drive, [])
    with mock.patch("ouroboros.llm_observability.chat_observed", side_effect=RuntimeError("provider down")):
        prepared, final = admission.prepare_scope_review(ctx, "test commit", scope_model=SCOPE_MODEL)
    assert prepared is None and final.status == "fixed_overflow"
    assert final.context_manifest["ladder_steps"][-1]["outcome"] == "failed"
    assert ce.resolve_review_token_density(drive, SCOPE_MODEL)[1] == "cold_conservative"


def test_a_fitting_pack_never_probes(tmp_path, drive, scope_window):
    repo = _repo(tmp_path, 2_000)  # fits the cold cap with room to spare
    ctx = _ctx(repo, drive, [])
    calls: list = []
    with mock.patch("ouroboros.llm_observability.chat_observed", side_effect=_probe_chat(calls)):
        prepared, final = admission.prepare_scope_review(ctx, "test commit", scope_model=SCOPE_MODEL)
    assert final is None and prepared is not None
    assert calls == [], "the rung never runs on a commit whose pack fits"
    assert not [s for s in prepared["context_manifest"]["ladder_steps"] if s["step"] == "density_probe"]
    assert ce.resolve_review_token_density(drive, SCOPE_MODEL)[1] == "cold_conservative"


def test_scope_pack_starved_is_size_only():
    assert admission._scope_pack_starved(sr._TouchedContextStatus(status="fixed_overflow"), {})
    assert admission._scope_pack_starved(sr._TouchedContextStatus(status="budget_exceeded"), {})
    assert not admission._scope_pack_starved(sr._TouchedContextStatus(status="omitted"), {})
    assert not admission._scope_pack_starved(sr._TouchedContextStatus(status="empty"), {})
    assert not admission._scope_pack_starved(None, {"ladder_steps": [{"step": "compact_atlas", "diff_only_files": 0}]})
    assert admission._scope_pack_starved(None, {"ladder_steps": [{"step": "compact_atlas", "diff_only_files": 2}]})
    assert admission._scope_pack_starved(None, {"ladder_steps": [{"step": "compact_atlas", "zero_context_diff": True}]})


# --- triad fit -----------------------------------------------------------------

def _triad_env(monkeypatch, prefix_tokens: int):
    import ouroboros.tools.review as review

    monkeypatch.setattr(review, "reviewer_context_window", lambda model: 1_000_000)
    monkeypatch.setattr(review, "run_cmd", lambda *a, **kw: "")
    monkeypatch.setattr(review, "_REVIEW_PROMPT_TEMPLATE_STABLE", "{preamble}")
    monkeypatch.setattr(review, "_REVIEW_PROMPT_TEMPLATE_DYNAMIC",
                        "{current_files_section}\n{diff_text}\n{changed_files}")

    def assemble(files_section, staged_diff):
        stable = review._REVIEW_PROMPT_TEMPLATE_STABLE.format(preamble="g" * (prefix_tokens * 4))
        dynamic = review._REVIEW_PROMPT_TEMPLATE_DYNAMIC.format(
            current_files_section=files_section, diff_text=staged_diff, changed_files="a.py")
        return stable + "\n" + dynamic, len(stable) + 1

    return assemble


def test_triad_cold_store_probes_the_overflowed_slot_once_and_fits(tmp_path, drive, monkeypatch):
    # 600K estimated tokens of irreducible prefix: above the cold cap of a 1M
    # slot (900,000 / 1.65 = 545,454), below its measured cap (745,000).
    assemble = _triad_env(monkeypatch, 600_000)
    model = "openai/gpt-5.6-terra"
    progress: list = []
    ctx = SimpleNamespace(drive_root=drive, task_id="triad-test", emit_progress_fn=progress.append,
                          pending_events=[])
    calls: list = []
    with mock.patch("ouroboros.llm_observability.chat_observed", side_effect=_probe_chat(calls)):
        prompt, _stable, overflow = admission.fit_triad_prompt(
            [model], assemble, "full snapshot of a.py", "+x", "a.py", tmp_path, ctx=ctx)

    assert overflow == "", overflow
    assert "full snapshot of a.py" in prompt, "no degradation rung was needed after the witness"
    assert [c["call_type"] for c in calls] == [admission.DENSITY_PROBE_CALL_TYPE]
    assert calls[0]["model"] == model and calls[0]["max_tokens"] == DENSITY_PROBE_MAX_TOKENS
    assert len(calls[0]["messages"][1]["content"]) == DENSITY_PROBE_SAMPLE_CHARS
    assert ce.resolve_review_token_density(drive, model)[1] == "measured"
    events = [e for e in ctx.pending_events if e.get("type") == admission.DENSITY_PROBE_EVENT]
    assert len(events) == 1 and events[0]["surface"] == "triad_review"


def test_triad_warm_store_never_probes(tmp_path, drive, monkeypatch):
    assemble = _triad_env(monkeypatch, 600_000)
    model = "openai/gpt-5.6-terra"
    ce.record_token_density(drive, model, prompt_chars=400_000, prompt_tokens=90_000)
    calls: list = []
    ctx = SimpleNamespace(drive_root=drive, task_id="t", emit_progress_fn=lambda _t: None, pending_events=[])
    with mock.patch("ouroboros.llm_observability.chat_observed", side_effect=_probe_chat(calls)):
        _prompt, _stable, overflow = admission.fit_triad_prompt(
            [model], assemble, "full snapshot of a.py", "+x", "a.py", tmp_path, ctx=ctx)
    assert overflow == "" and calls == []


def test_triad_budget_refused_probe_keeps_the_typed_fit_terminal(tmp_path, drive, monkeypatch):
    assemble = _triad_env(monkeypatch, 800_000)  # above even the measured cap
    model = "openai/gpt-5.6-terra"
    ctx = SimpleNamespace(drive_root=drive, task_id="t", emit_progress_fn=lambda _t: None, pending_events=[])

    def refused(llm, **kwargs):
        raise BudgetExceeded("global budget exhausted")

    with mock.patch("ouroboros.llm_observability.chat_observed", side_effect=refused):
        _prompt, _stable, overflow = admission.fit_triad_prompt(
            [model], assemble, "full snapshot of a.py", "+x", "a.py", tmp_path, ctx=ctx)
    assert "REVIEW_BLOCKED" in overflow
    events = [e for e in ctx.pending_events if e.get("type") == admission.DENSITY_PROBE_EVENT]
    assert len(events) == 1 and events[0]["outcome"] == "budget_refused"


def test_triad_without_a_ctx_never_sends(tmp_path, drive, monkeypatch):
    """A bare fit-check (no ctx, no drive root to record a witness on) is the
    pre-rung behaviour byte for byte: no send, the cold cap, the typed block."""
    assemble = _triad_env(monkeypatch, 600_000)
    calls: list = []
    with mock.patch("ouroboros.llm_observability.chat_observed", side_effect=_probe_chat(calls)):
        _prompt, _stable, overflow = admission.fit_triad_prompt(
            ["openai/gpt-5.6-terra"], assemble, "full snapshot of a.py", "+x", "a.py", tmp_path)
    assert "REVIEW_BLOCKED" in overflow and calls == []


def test_triad_probes_every_overflowing_cold_slot_and_recomputes_the_quorum_cap(tmp_path, drive, monkeypatch):
    """Three DISTINCT cold models (the shipped panel shape): the rung probes
    every overflowing slot — not the first witness only — so the quorum cap
    (the quorum-th largest slot cap) moves and the prompt fits undegraded."""
    import ouroboros.tools.review as review
    from ouroboros.tools.review_synthesis import quorum_input_token_limit

    assemble = _triad_env(monkeypatch, 600_000)
    models = ["openai/gpt-5.6-terra", "anthropic/claude-fable-5.1", "x-ai/grok-4.6"]

    def caps() -> dict:
        return {m: review.calibrated_input_token_limit(
            m, context_window=1_000_000, output_reserve=50_000, tokenizer_margin=50_000,
            drive_root=drive) for m in models}

    cold_caps = caps()
    assert quorum_input_token_limit(models, cold_caps) < 600_000
    ctx = SimpleNamespace(drive_root=drive, task_id="triad-3", emit_progress_fn=lambda _t: None,
                          pending_events=[])
    calls: list = []
    with mock.patch("ouroboros.llm_observability.chat_observed", side_effect=_probe_chat(calls)):
        prompt, _stable, overflow = admission.fit_triad_prompt(
            models, assemble, "full snapshot of a.py", "+x", "a.py", tmp_path, ctx=ctx)

    assert overflow == "", overflow
    assert "full snapshot of a.py" in prompt, "no degradation rung after the witnesses"
    assert [c["model"] for c in calls] == models, "one probe per overflowing cold slot"
    assert all(ce.resolve_review_token_density(drive, m)[1] == "measured" for m in models)
    measured_caps = caps()
    assert all(measured_caps[m] > cold_caps[m] for m in models)
    assert quorum_input_token_limit(models, measured_caps) >= 600_000
    # Why a first-witness short-circuit was wrong: with only the first slot
    # measured the quorum cap is still the cold one and the pack still overflows.
    assert quorum_input_token_limit(models, {**cold_caps, models[0]: measured_caps[models[0]]}) < 600_000
    events = [e for e in ctx.pending_events if e.get("type") == admission.DENSITY_PROBE_EVENT]
    assert sorted(e["model"] for e in events) == sorted(models)


def test_triad_probes_only_the_cold_overflowing_slots(tmp_path, drive, monkeypatch):
    assemble = _triad_env(monkeypatch, 600_000)
    warm, cold_a, cold_b = "openai/gpt-5.6-terra", "anthropic/claude-fable-5.1", "x-ai/grok-4.6"
    ce.record_token_density(drive, warm, prompt_chars=400_000, prompt_tokens=90_000)
    ctx = SimpleNamespace(drive_root=drive, task_id="t", emit_progress_fn=lambda _t: None, pending_events=[])
    calls: list = []
    with mock.patch("ouroboros.llm_observability.chat_observed", side_effect=_probe_chat(calls)):
        _prompt, _stable, overflow = admission.fit_triad_prompt(
            [warm, cold_a, cold_b], assemble, "full snapshot of a.py", "+x", "a.py", tmp_path, ctx=ctx)
    assert overflow == ""
    assert [c["model"] for c in calls] == [cold_a, cold_b], "a warm slot never spends a probe"


# --- one physical attempt --------------------------------------------------------

class _CountingTransport:
    """A fake transport on the REAL physical-attempt rail: ``chat`` dispatches
    through ``execute_physical_attempt`` and redials once on a transient
    failure, exactly like the fallback ladder's body-error reroute."""

    def __init__(self, drive, *, transient_first: bool) -> None:
        self.drive, self.transient_first, self.sends = drive, transient_first, []

    def chat(self, **kwargs):
        from ouroboros.usage_accounting import AttemptRequest, execute_physical_attempt

        request = AttemptRequest(model=kwargs["model"], provider="openrouter", reservation_usd=0.0,
                                 drive_root=self.drive, task_id="probe-rail")

        def physical():
            self.sends.append(kwargs["model"])
            if self.transient_first and len(self.sends) == 1:
                raise RuntimeError("provider body error 429")
            chars = sum(len(m["content"]) for m in kwargs["messages"])
            return {"choices": [{"message": {"content": "OK"}}],
                    "usage": {"prompt_tokens": int(chars / 4 * 0.9), "completion_tokens": 1}}

        try:
            response = execute_physical_attempt(request, physical)
        except RuntimeError:
            response = execute_physical_attempt(request, physical)  # the ladder's one redial
        return response["choices"][0]["message"], dict(response["usage"])


def _probe(drive, transport, model="openai/gpt-5.6-terra", progress=None) -> str:
    return ce.cold_start_density_probe(
        drive, transport, (progress if progress is not None else []).append, model, "z" * 80_000,
        task_id="t", call_type=admission.DENSITY_PROBE_CALL_TYPE, source="commit_gate_cold_start_probe")


def test_probe_is_exactly_one_physical_attempt_even_when_the_transport_would_redial(drive):
    from ouroboros.usage_accounting import PhysicalAttemptLimitExceeded, _claim_physical_dispatch

    transport = _CountingTransport(drive, transient_first=True)
    progress: list = []
    assert _probe(drive, transport, progress=progress) == "failed"
    assert transport.sends == ["openai/gpt-5.6-terra"], "the redial never reached the provider"
    assert ce.resolve_review_token_density(drive, "openai/gpt-5.6-terra")[1] == "cold_conservative"
    assert any("Density probe failed (PhysicalAttemptLimitExceeded)" in p for p in progress)
    _claim_physical_dispatch()  # the rail is released with the probe: no limit leaks out
    with pytest.raises(PhysicalAttemptLimitExceeded):
        transport.transient_first = False
        # Outside the probe the same transport redials freely; inside it cannot.
        from ouroboros.usage_accounting import physical_attempt_limit
        with physical_attempt_limit(1):
            _claim_physical_dispatch()
            _claim_physical_dispatch()


def test_probe_one_successful_physical_attempt_records_the_witness(drive):
    transport = _CountingTransport(drive, transient_first=False)
    assert _probe(drive, transport) == "measured"
    assert transport.sends == ["openai/gpt-5.6-terra"]
    assert ce.resolve_review_token_density(drive, "openai/gpt-5.6-terra")[1] == "measured"


# --- the gate's client seam and the shared rung -----------------------------------

def test_gate_uses_the_review_surface_client_seam_and_types_a_constructor_failure(tmp_path, drive, monkeypatch):
    import ouroboros.tools.review as review

    seam = object()
    monkeypatch.setattr(review, "LLMClient", lambda: seam)
    ctx = SimpleNamespace(drive_root=drive, task_id="t", emit_progress_fn=lambda _t: None, pending_events=[])
    seen: list = []

    def chat(llm, **kwargs):
        seen.append(llm)
        return {"content": "OK"}, {"prompt_tokens": 18_000}

    with mock.patch("ouroboros.llm_observability.chat_observed", side_effect=chat):
        assert admission.density_probe_before_size_refusal(ctx, "m/one", "z" * 80_000, surface="triad_review") == "measured"
    assert seen == [seam], "the probe's client comes from review.LLMClient"

    def broken():
        raise RuntimeError("no provider credentials")

    monkeypatch.setattr(review, "LLMClient", broken)
    ctx = SimpleNamespace(drive_root=drive, task_id="t", emit_progress_fn=lambda _t: None, pending_events=[])
    assert admission.density_probe_before_size_refusal(ctx, "m/two", "z" * 80_000, surface="scope_review") == "failed"
    events = [e for e in ctx.pending_events if e.get("type") == admission.DENSITY_PROBE_EVENT]
    assert len(events) == 1 and events[0]["outcome"] == "failed"
    assert "RuntimeError: no provider credentials" in events[0]["reason"]


def test_density_probe_sample_confines_rows_to_the_repo(tmp_path):
    from ouroboros.tools.review_helpers import density_probe_sample

    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    (repo / "sub" / "inside.md").write_text("inside", encoding="utf-8")
    (tmp_path / "outside.md").write_text("outside", encoding="utf-8")
    rows = [
        {"path": "sub/inside.md"},
        {"path": "sub/../sub/inside.md"},  # normalizes inside: kept
        {"path": "../outside.md"},
        {"path": str(tmp_path / "outside.md")},  # absolute, outside
        {"path": str(repo / "sub" / "inside.md")},  # absolute, inside: never a manifest shape, but contained
        {"path": "C:\\outside.md"},
    ]
    sample = density_probe_sample(repo, {"selected": rows})
    assert "outside" not in sample
    assert sample.count("\ninside\n") == 3  # the two relative rows and the contained absolute one


def test_deep_review_and_gate_share_one_rung(tmp_path, drive, monkeypatch):
    """Behavioural pin: BOTH surfaces reach the one shared rung with the same
    keyword contract, each naming itself as the witness source."""
    from ouroboros import deep_self_review
    from ouroboros.deep_self_review import run_deep_self_review
    from ouroboros.reviewer_slot_config import DEEP_REVIEW_SLOT_ID, ConfiguredReviewerSlot

    assert not hasattr(deep_self_review, "_cold_start_density_probe")
    seen: list = []

    def rung(drive_root, llm, emit_progress, model, sample, *, task_id, call_type, source):
        seen.append({"model": model, "task_id": task_id, "call_type": call_type, "source": source,
                     "sample": len(sample)})
        return "failed"

    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "ARCHITECTURE.md").write_text("# Architecture\n" + ("prose. " * 3000), encoding="utf-8")
    (drive / "memory").mkdir()
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    unfit = ("", {"file_count": 0, "total_chars": 0, "skipped": ["FATAL: required artifact could not be assembled"],
                  "context_manifest": {"status": "budget_omitted", "selected": [],
                                       "unassembled_required": [{"path": "docs/ARCHITECTURE.md"}]}})
    with (
        mock.patch("ouroboros.capability_evidence.cold_start_density_probe", side_effect=rung),
        mock.patch("ouroboros.deep_self_review.build_review_pack", return_value=unfit),
        mock.patch("ouroboros.deep_self_review._run_retrieving_review"),
    ):
        run_deep_self_review(
            repo_dir=repo, drive_root=drive, llm=mock.Mock(), emit_progress=lambda *_a, **_k: None,
            slot=ConfiguredReviewerSlot(slot_id=DEEP_REVIEW_SLOT_ID, kind="api_chat", target_id="m/deep"))
        ctx = SimpleNamespace(drive_root=drive, task_id="gate", emit_progress_fn=lambda _t: None, pending_events=[])
        assert admission.density_probe_before_size_refusal(ctx, "m/gate", "z" * 100, surface="triad_review") == "failed"

    assert [s["source"] for s in seen] == ["deep_review_cold_start_probe", "commit_gate_cold_start_probe"]
    assert [s["model"] for s in seen] == ["m/deep", "m/gate"]
    assert seen[0]["call_type"] == "deep_self_review_density_probe" and seen[1]["call_type"] == admission.DENSITY_PROBE_CALL_TYPE
    assert seen[0]["sample"] > 0, "the deep review measures on the real refused rows"
