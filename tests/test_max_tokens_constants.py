"""Regression tests: verify raised max_tokens / max_turns constants."""


def test_review_query_model_max_tokens():
    """review.py _query_model defaults to a 65536-token response reservation.
    v6.61.1: the literal moved into _review_output_budget (the
    OUROBOROS_REVIEW_MAX_TOKENS operator knob may LOWER it for mega-diff packs,
    floor 8192, never raise) — the un-overridden default must stay 65536."""
    import os
    from unittest import mock

    from ouroboros.tools.review import _review_output_budget

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OUROBOROS_REVIEW_MAX_TOKENS", None)
        assert _review_output_budget() == 65536


def test_project_naming_max_tokens_pinned():
    """v6.40 drift-guard (DEVELOPMENT #13): the LIGHT project-naming one-shot stays a TINY
    budget (256) — pinned so it can't silently drift up and matches the ARCHITECTURE
    'LLM output token budgets' table row."""
    import ast
    from pathlib import Path

    src = Path("ouroboros/project_naming.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    found = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword)
        and node.arg == "max_tokens"
        and isinstance(node.value, ast.Constant)
    ]
    assert 256 in found, f"Expected max_tokens=256 in project_naming.py, got {found}"


def test_provider_test_max_tokens_pinned():
    """The paid Provider Test remains one deliberately tiny 16-token probe."""
    from ouroboros.llm_probe import PROVIDER_TEST_MAX_TOKENS

    assert PROVIDER_TEST_MAX_TOKENS == 16


def test_scope_review_max_tokens():
    """scope_review.py _SCOPE_MAX_TOKENS must be ≥100000."""
    from ouroboros.tools.scope_review import _SCOPE_MAX_TOKENS
    assert _SCOPE_MAX_TOKENS >= 100_000


def test_llm_client_default_max_tokens():
    """Main remote chat defaults must leave enough output room for long tool plans."""
    import inspect
    from ouroboros.llm import LLMClient

    assert inspect.signature(LLMClient.chat).parameters["max_tokens"].default >= 65_536
    assert inspect.signature(LLMClient.chat_async).parameters["max_tokens"].default >= 65_536


def test_main_loop_explicit_max_tokens():
    """The task loop must pin the same 64K output budget even if client defaults move."""
    from ouroboros.loop_llm_call import MAIN_LOOP_MAX_TOKENS

    assert MAIN_LOOP_MAX_TOKENS >= 65_536


def test_vision_query_default_max_tokens():
    """VLM tools inherit the shared vision_query output budget."""
    import inspect
    from ouroboros.llm import LLMClient

    assert inspect.signature(LLMClient.vision_query).parameters["max_tokens"].default >= 32_768


def test_summary_and_background_token_budgets():
    """Summary/reflection/background paths must stay above the raised floors."""
    from pathlib import Path
    from ouroboros import context_compaction

    expectations = {
        "ouroboros/tools/review_synthesis.py": "max_tokens=16384",
        "ouroboros/consolidator.py": "max_tokens=16384",
        "ouroboros/reflection.py": "max_tokens=16384",
        "ouroboros/post_task_synthesis.py": "max_tokens=16384",
        "ouroboros/tools/skill_publish.py": "max_tokens=8192",
        "ouroboros/consciousness.py": "max_tokens=65536",
    }
    for path, needle in expectations.items():
        src = Path(path).read_text(encoding="utf-8").replace(" ", "")
        assert needle in src, f"{path} must contain {needle}"
    assert context_compaction._SUMMARY_OUTPUT_TOKENS == 32_768
    assert context_compaction._summarizer_spec()["output_budget"] == 32_768


def test_native_review_episode_ceiling_is_ssot_and_round_cap_is_retired():
    """The native inspection episode (the advisory/actor-row successor of the
    retired Claude-SDK max-turns budget) reads its transcript CEILING from
    config SSOT with a shipped default, never a hardcoded literal — and has
    no round cap at all (P13: the floor is hardcoded, never the ceiling)."""
    from ouroboros.config import RETIRED_SETTING_KEYS, SETTINGS_DEFAULTS
    from ouroboros import review_native_episode
    from ouroboros.review_native_episode import review_native_max_transcript_chars

    assert SETTINGS_DEFAULTS["OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS"] == "900000"
    assert "OUROBOROS_REVIEW_NATIVE_MAX_ROUNDS" not in SETTINGS_DEFAULTS
    assert "OUROBOROS_REVIEW_NATIVE_MAX_ROUNDS" in RETIRED_SETTING_KEYS
    assert not hasattr(review_native_episode, "review_native_max_rounds")
    assert review_native_max_transcript_chars() >= 10_000


def test_claude_code_sdk_only_no_cli_fallback():
    """shell.py must not contain legacy CLI subprocess fallback."""
    src = open("ouroboros/tools/shell.py", encoding="utf-8").read()
    assert "_run_claude_cli" not in src, "CLI fallback function should be gone"
    assert "ensure_claude_cli" not in src, "CLI install function should be gone"


def test_review_prompt_token_budget_is_ssot():
    """``review_helpers.REVIEW_PROMPT_TOKEN_BUDGET`` is the single source of
    truth for the unified scope/plan/deep-review input gate (920K). Bumping
    the constant must move all three call sites in lockstep so the skip
    threshold cannot silently desync between modules.

    Note: Claude Opus 4.6 has a 1M context window SHARED between input and
    output. ``estimate_tokens`` (chars/4) is approximate, so the 920K gate
    intentionally leaves limited output headroom and remains best-effort.
    """
    from ouroboros.tools.review_helpers import REVIEW_PROMPT_TOKEN_BUDGET
    from ouroboros.tools.scope_review import _SCOPE_BUDGET_TOKEN_LIMIT

    assert REVIEW_PROMPT_TOKEN_BUDGET == 920_000, (
        f"REVIEW_PROMPT_TOKEN_BUDGET drifted to {REVIEW_PROMPT_TOKEN_BUDGET}; "
        "see review_helpers.py docstring before changing — call sites do "
        "not silently re-pin to an old budget."
    )
    assert _SCOPE_BUDGET_TOKEN_LIMIT == REVIEW_PROMPT_TOKEN_BUDGET, (
        f"_SCOPE_BUDGET_TOKEN_LIMIT ({_SCOPE_BUDGET_TOKEN_LIMIT}) must equal "
        f"the SSOT REVIEW_PROMPT_TOKEN_BUDGET ({REVIEW_PROMPT_TOKEN_BUDGET})."
    )
    # Plan review reserves output headroom inside the reviewer window (same class of
    # fix as scope review), but v6.80.0 replaced its module constants with PER-SLOT
    # calibrated caps: every slot cap must still stay inside the shared SSOT.
    from ouroboros.tools.plan_review import _PLAN_REVIEW_MAX_TOKENS
    from ouroboros.tools.review_synthesis import per_slot_input_token_limits

    limits = per_slot_input_token_limits(
        ["anthropic/claude-fable-5", "openai/gpt-5.6-sol"],
        context_window=1_000_000,
        output_reserve=_PLAN_REVIEW_MAX_TOKENS,
        tokenizer_margin=155_000,
    )
    assert limits and all(v <= REVIEW_PROMPT_TOKEN_BUDGET for v in limits.values())
    assert all(v > 0 for v in limits.values())
    assert all(v + _PLAN_REVIEW_MAX_TOKENS <= 1_000_000 for v in limits.values()), (
        "a per-slot plan input cap + reserved output exceeds the reviewer window; "
        "the provider would hard-400."
    )


def test_scope_input_budget_reserves_output_within_window():
    """Scope input cap + reserved output must fit the reviewer context window.

    Regression guard for the deterministic provider 400 where the 920K input gate
    plus the 100K output reservation exceeded the 1M window and fail-closed-blocked
    every commit. The assembled INPUT prompt is gated on ``_SCOPE_INPUT_TOKEN_LIMIT``,
    which must leave room for ``_SCOPE_MAX_TOKENS`` output inside
    ``_SCOPE_MODEL_CONTEXT_WINDOW`` while never exceeding the shared 920K SSOT.
    """
    from ouroboros.tools.scope_review import (
        _SCOPE_BUDGET_TOKEN_LIMIT,
        _SCOPE_INPUT_TOKEN_LIMIT,
        _SCOPE_MAX_TOKENS,
        _SCOPE_MODEL_CONTEXT_WINDOW,
        _SCOPE_OUTPUT_MARGIN_TOKENS,
    )

    assert _SCOPE_INPUT_TOKEN_LIMIT == 745_000
    assert _SCOPE_INPUT_TOKEN_LIMIT + _SCOPE_MAX_TOKENS <= _SCOPE_MODEL_CONTEXT_WINDOW, (
        f"scope input cap ({_SCOPE_INPUT_TOKEN_LIMIT}) + reserved output "
        f"({_SCOPE_MAX_TOKENS}) exceeds the {_SCOPE_MODEL_CONTEXT_WINDOW}-token "
        "reviewer window; the provider would hard-400 and fail closed."
    )
    assert _SCOPE_INPUT_TOKEN_LIMIT + _SCOPE_MAX_TOKENS + _SCOPE_OUTPUT_MARGIN_TOKENS <= _SCOPE_MODEL_CONTEXT_WINDOW, (
        "scope input cap must leave both output reservation and tokenizer-underestimate headroom."
    )
    assert _SCOPE_OUTPUT_MARGIN_TOKENS >= 150_000, (
        "scope review needs a large tokenizer headroom margin for atlas-heavy prompts."
    )
    assert _SCOPE_INPUT_TOKEN_LIMIT <= _SCOPE_BUDGET_TOKEN_LIMIT, (
        "scope input cap must not exceed the shared prompt-size SSOT."
    )


def test_scope_input_limit_is_density_calibrated(monkeypatch, tmp_path):
    """Scope reviewers get a MEASURED-density-calibrated input cap (v6.80.0).

    Regression guard for the deterministic 400 where a 739,508-estimated-token scope
    pack measured 1,166,914 REAL tokens on the Claude tokenizer (~1.58x the chars/4
    estimate on code) and was rejected by every fable-5 upstream as "prompt is too
    long: > 1,000,000 maximum". The hand-set family constant is gone; the cap must
    still keep estimated*density + output inside the 1M window, must never exceed the
    shared SSOT, and must never be LOOSER than the historical absolute-margin cap.
    """
    from ouroboros.capability_evidence import COLD_START_TOKEN_DENSITY
    from ouroboros.tools.scope_review import (
        _SCOPE_BUDGET_TOKEN_LIMIT,
        _SCOPE_INPUT_TOKEN_LIMIT,
        _SCOPE_MAX_TOKENS,
        _SCOPE_MODEL_CONTEXT_WINDOW,
        _effective_scope_input_limit,
    )

    # This test verifies the CALIBRATION at a 1M window, not the window-resolution
    # policy: since the v6.46.0 false-1M fix an off-default model with no Capability
    # Evidence fail-closes to the sub-floor.
    from ouroboros.reviewer_window import ReviewerWindow
    monkeypatch.setattr("ouroboros.tools.scope_review._scope_window",
                        lambda m: ReviewerWindow(1_000_000, "confirmed"))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))

    assert COLD_START_TOKEN_DENSITY >= 1.58, (
        "the conservative cold-start density must cover the measured 1.58x Claude code density"
    )
    cap = _effective_scope_input_limit(scope_model="anthropic/claude-fable-5")
    assert int(cap * COLD_START_TOKEN_DENSITY) + _SCOPE_MAX_TOKENS <= _SCOPE_MODEL_CONTEXT_WINDOW, (
        "calibrated cap * real-token density + output reserve must fit the 1M window"
    )
    assert cap <= _SCOPE_BUDGET_TOKEN_LIMIT
    # Every spelling of the shipped reviewer resolves to the same calibrated cap, and
    # the historical absolute-margin cap remains an upper bound for ANY model.
    for spelling in ("anthropic::claude-fable-5", "fable-5", "~mythos-5", "openai/gpt-5.5"):
        assert _effective_scope_input_limit(scope_model=spelling) <= _SCOPE_INPUT_TOKEN_LIMIT


def test_calibrated_input_limit_shared_helper(tmp_path, monkeypatch):
    """The shared helper is the calibration SSOT for scope, triad, plan AND deep review.

    A MEASURED observation must actually reach it (the old import-time constant froze
    the pre-measurement value for the whole process), and seeding an observation that
    reproduces the legacy effective density must reproduce the legacy limit BYTE-FOR-BYTE.
    """
    import inspect

    from ouroboros import deep_self_review
    from ouroboros.capability_evidence import (
        COLD_START_TOKEN_DENSITY,
        MEASURED_DENSITY_SAFETY_FACTOR,
        record_token_density,
    )
    from ouroboros.tools.review_helpers import calibrated_input_token_limit

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))

    def limit(model):
        return calibrated_input_token_limit(
            model, context_window=1_000_000, output_reserve=100_000,
            tokenizer_margin=155_000, drive_root=tmp_path,
        )

    # Cold start: the conservative density applies to EVERY model, so the cap is at or
    # below the historical GPT-family 745K and never above it.
    assert limit("openai/gpt-5.5") == int(900_000 / COLD_START_TOKEN_DENSITY)
    assert limit("anthropic/claude-fable-5") == int(900_000 / COLD_START_TOKEN_DENSITY)
    assert limit("openai/gpt-5.5") <= 745_000

    # Seed a measurement reproducing the LEGACY effective density (1.65) and the
    # legacy Claude limit comes back byte-identical.
    legacy_density = 1.65 / MEASURED_DENSITY_SAFETY_FACTOR
    chars = 4_000_000
    record_token_density(
        tmp_path, "anthropic/claude-fable-5",
        prompt_chars=chars, prompt_tokens=int(legacy_density * chars / 4),
    )
    assert limit("anthropic/claude-fable-5") == int(900_000 / 1.65)

    # #284 successor: a FRESH EXACT-MODEL witness is authoritative and may
    # undercut the cold floor — the limit loosens to the density form, still
    # bounded above by the historical absolute-margin form (745K here).
    record_token_density(
        tmp_path, "openai/gpt-5.5", prompt_chars=chars, prompt_tokens=int(1.0 * chars / 4),
    )
    assert limit("openai/gpt-5.5") == 1_000_000 - 100_000 - 155_000  # margin-bounded
    assert limit("openai/gpt-5.5") > int(900_000 / COLD_START_TOKEN_DENSITY)

    # Deep self-review's PACKED delivery consumes the same helper for its
    # model-aware gate (the retrieving deliveries have no pack to size).
    assert "calibrated_input_token_limit" in inspect.getsource(deep_self_review._run_packed_review)
    # ...as does the triad, whose pack size moves with the same formula.
    from ouroboros.tools import review as triad
    assert "calibrated_input_token_limit" in inspect.getsource(triad)


def test_scope_cap_of_an_unwitnessed_reviewer_ignores_another_models_witness(tmp_path, monkeypatch):
    """Paid run 2026-09-04, lane SM1_a1, commit-gate attempt 2: gpt-5.6-terra's
    window confirmed at 1,050,000, no terra witness yet, and a gemini-3.8-flash
    witness at 1.81 in the store. The scope cap came out at 499,627 (1.81 x 1.05
    applied to terra) instead of the floor-based 575,757, starving 48 required
    protected-path artifacts before any reviewer was dispatched. The cap of an
    unwitnessed reviewer is the floor's, whatever other models measured."""
    from ouroboros.capability_evidence import _DENSITY_MEMO, COLD_START_TOKEN_DENSITY, record_token_density
    from ouroboros.reviewer_window import ReviewerWindow
    from ouroboros.tools.scope_review import _effective_scope_input_limit

    monkeypatch.setattr("ouroboros.tools.scope_review._scope_window",
                        lambda m: ReviewerWindow(1_050_000, "confirmed"))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    _DENSITY_MEMO.clear()
    record_token_density(
        tmp_path, "google/gemini-3.8-flash", prompt_chars=900_708, prompt_tokens=407_767,
    )

    cap = _effective_scope_input_limit(scope_model="openai/gpt-5.6-terra")
    assert cap == int((1_050_000 - 100_000) / COLD_START_TOKEN_DENSITY) == 575_757
    assert cap != 499_627


def test_measured_density_never_loosens_a_models_own_review_pack_cap(tmp_path, monkeypatch):
    """A still-fresh dense witness may only tighten a model's review cap.

    One normalized identity accumulates observations from every surface (the shipped
    `claude-fable-5` is both the scope reviewer and a triad slot), so a run of doc-only
    commits whose prose-dominated packs measure ~1.1 must not pull a code-heavy model's
    dense witness's TTL and hand the NEXT code-heavy scope pack a bigger cap. The
    guarantee lives in retained timestamped raw witnesses, plus the 1.65 cold floor;
    there is no independently refreshed aggregate scalar.
    """
    from ouroboros.capability_evidence import (
        _DENSITY_MEMO,
        record_token_density,
        resolve_token_density,
    )
    from ouroboros.tools.review_helpers import calibrated_input_token_limit

    _DENSITY_MEMO.clear()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))

    def limit(model):
        return calibrated_input_token_limit(
            model, context_window=1_000_000, output_reserve=100_000,
            tokenizer_margin=155_000, drive_root=tmp_path,
        )

    cold = limit("anthropic/claude-fable-5")
    chars = 4_000_000

    # A real code-heavy pack measures 1.58 on this identity: the cap TIGHTENS.
    record_token_density(
        tmp_path, "anthropic/claude-fable-5",
        prompt_chars=chars, prompt_tokens=int(1.58 * chars / 4),
    )
    density, source = resolve_token_density(tmp_path, "anthropic/claude-fable-5")
    assert source == "measured"
    tightened = limit("anthropic/claude-fable-5")
    assert tightened <= cold

    # A later run of prose-dominated packs measures ~1.1 on the SAME identity: the
    # the fresh dense witness remains retained, so the cap does NOT loosen back.
    _DENSITY_MEMO.clear()
    record_token_density(
        tmp_path, "anthropic/claude-fable-5",
        prompt_chars=chars, prompt_tokens=int(1.1 * chars / 4),
    )
    assert resolve_token_density(tmp_path, "anthropic/claude-fable-5")[0] == density
    assert limit("anthropic/claude-fable-5") == tightened

    # A measured DENSER value still tightens further (the direction that prevents 400s).
    _DENSITY_MEMO.clear()
    record_token_density(
        tmp_path, "anthropic/claude-fable-5",
        prompt_chars=chars, prompt_tokens=int(2.0 * chars / 4),
    )
    assert limit("anthropic/claude-fable-5") < tightened


def test_scope_normalize_defaults_pass_severity_only():
    """PASS rows may omit severity (semantically void there); FAIL rows must
    carry an explicit valid severity because it decides blocking (fail-closed)."""
    from ouroboros.tools.scope_review import _SCOPE_REQUIRED_ITEMS, _normalize_scope_items

    items = []
    required = sorted(_SCOPE_REQUIRED_ITEMS)
    for idx, item_id in enumerate(required):
        if idx == 0:
            items.append({"item": item_id, "verdict": "FAIL", "severity": "advisory",
                          "reason": "a concrete finding with enough words"})
        else:
            # PASS rows WITHOUT severity — the fable-5 output shape.
            items.append({"item": item_id, "verdict": "PASS",
                          "reason": "verified against the staged diff and context"})
    normalized, err = _normalize_scope_items(items)
    assert err == "", f"PASS rows without severity must normalize cleanly: {err}"
    assert len(normalized) == len(required)

    # A FAIL row without severity stays invalid (fail-closed).
    bad = list(items)
    bad[0] = {"item": required[0], "verdict": "FAIL", "reason": "a concrete finding with enough words"}
    _normalized, err = _normalize_scope_items(bad)
    assert "missing or invalid severity" in err


def test_scope_actor_record_surfaces_error_text():
    """A non-responded scope actor record must carry the failure text so a
    provider 400 is visible in the verdict without observability digging."""
    from ouroboros.tools.review_helpers import build_scope_actor_record
    from ouroboros.tools.scope_review import ScopeReviewResult

    failed = ScopeReviewResult(
        blocked=True,
        block_message="SCOPE_REVIEW_BLOCKED: Error code: 400 - prompt is too long",
        model_id="anthropic/claude-fable-5",
        status="error",
    )
    record = build_scope_actor_record(failed, slot_id="scope_slot_1")
    assert "400" in record["error"]
    ok = ScopeReviewResult(model_id="m", status="responded", raw_text="[]")
    assert build_scope_actor_record(ok, slot_id="s")["error"] == ""


def test_deep_self_review_budget_uses_ssot(tmp_path, monkeypatch):
    """The packed deep review gates the FULL assembled prompt (system + user)
    on the model-calibrated, output-reserving input limit the shared helper
    returns (min(SSOT, window − output − margin), as scope/plan review do),
    measured with the shared ``estimate_tokens`` (chars/4) — pinned by
    BEHAVIOR: the limit is what the helper says, the measure includes the
    system prompt, and a pack over it is refused before any send.
    """
    from unittest import mock

    from ouroboros import deep_self_review
    from ouroboros.deep_self_review import (
        _DEEP_INPUT_TOKEN_LIMIT,
        _DEEP_MAX_OUTPUT_TOKENS,
        _DEEP_MODEL_CONTEXT_WINDOW,
        _DEEP_OUTPUT_MARGIN_TOKENS,
        _SYSTEM_PROMPT,
        run_deep_self_review,
    )
    from ouroboros.reviewer_slot_config import DEEP_REVIEW_SLOT_ID, ConfiguredReviewerSlot
    from ouroboros.tools.review_helpers import REVIEW_PROMPT_TOKEN_BUDGET
    from ouroboros.utils import estimate_tokens

    # The uncalibrated window arithmetic: SSOT budget with the output reserve.
    assert _DEEP_INPUT_TOKEN_LIMIT == min(
        REVIEW_PROMPT_TOKEN_BUDGET,
        _DEEP_MODEL_CONTEXT_WINDOW - _DEEP_MAX_OUTPUT_TOKENS - _DEEP_OUTPUT_MARGIN_TOKENS,
    )
    assert _DEEP_INPUT_TOKEN_LIMIT + _DEEP_MAX_OUTPUT_TOKENS <= _DEEP_MODEL_CONTEXT_WINDOW, (
        "deep review input cap + reserved output exceeds the reviewer window; "
        "the provider would hard-400."
    )

    # The ENFORCED limit is the shared helper's answer (here: a sentinel), and
    # the gated measure is system prompt + pack: a pack that fits alone but
    # not with the system prompt is refused, quoting the enforced number.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    limit = estimate_tokens(_SYSTEM_PROMPT) + 100
    monkeypatch.setattr(deep_self_review, "calibrated_input_token_limit", lambda *a, **k: limit)
    row = ConfiguredReviewerSlot(slot_id=DEEP_REVIEW_SLOT_ID, kind="api_chat", target_id="openai/x")
    llm = mock.Mock()
    llm.chat.return_value = ({"content": "ok"}, {"cost": 0.0})
    (tmp_path / "state").mkdir()
    too_big = "x" * (4 * 150)  # ~150 tokens: fits the limit alone, not with the system prompt
    with mock.patch.object(deep_self_review, "build_review_pack",
                           return_value=(too_big, {"file_count": 1, "total_chars": len(too_big), "skipped": []})):
        text, usage = run_deep_self_review(tmp_path, tmp_path, llm, lambda _m: None, slot=row)
    assert "too large" in text and f"~{limit:,} tokens" in text
    assert usage["execution_status"] == "infra_failed" and not llm.chat.called
    fits = "x" * (4 * 50)
    with mock.patch.object(deep_self_review, "build_review_pack",
                           return_value=(fits, {"file_count": 1, "total_chars": len(fits), "skipped": []})):
        text, _usage = run_deep_self_review(tmp_path, tmp_path, llm, lambda _m: None, slot=row)
    assert text.endswith("ok") and llm.chat.call_count == 1


def test_tool_timeout_uses_max_of_settings_and_per_tool():
    """_get_tool_timeout must return max(settings, per_tool) not just settings."""
    from unittest.mock import patch
    import ouroboros.loop_tool_execution as mod

    class FakeTools:
        def get_timeout(self, name):
            return 1200  # per-tool declares 1200s

    # settings says 600, per-tool says 1200 → should return 1200
    with patch.object(mod, "load_settings", return_value={"OUROBOROS_TOOL_TIMEOUT_SEC": 600}):
        result = mod._get_tool_timeout(FakeTools(), "run_command")
    assert result == 1200, f"Expected 1200 (per-tool), got {result}"


def test_tool_timeout_settings_wins_when_higher():
    """_get_tool_timeout: if settings > per_tool, settings wins."""
    from unittest.mock import patch
    import ouroboros.loop_tool_execution as mod

    class FakeTools:
        def get_timeout(self, name):
            return 360  # default per-tool

    with patch.object(mod, "load_settings", return_value={"OUROBOROS_TOOL_TIMEOUT_SEC": 900}):
        result = mod._get_tool_timeout(FakeTools(), "run_command")
    assert result == 900, f"Expected 900 (settings), got {result}"


def test_review_evidence_no_truncation_by_default():
    """format_review_evidence_for_prompt must NOT truncate by default (max_chars=0)."""
    from ouroboros.review_evidence import format_review_evidence_for_prompt
    big = {"has_evidence": True, "task_id": "task-review", "data": "x" * 10000}
    result = format_review_evidence_for_prompt(big)
    assert "truncated" not in result.lower()
    assert len(result) > 10000


def test_review_evidence_bounded_with_omission_note():
    """format_review_evidence_for_prompt truncates with explicit omission note when max_chars>0."""
    from ouroboros.review_evidence import format_review_evidence_for_prompt
    big = {"has_evidence": True, "task_id": "task-review", "data": "x" * 10000}
    result = format_review_evidence_for_prompt(big, max_chars=500)
    assert "OMISSION NOTE" in result
    assert "truncated at 500 chars" in result


def test_review_evidence_no_obligation_cap():
    """collect_review_evidence default max_obligations must be None (no cap)."""
    import inspect
    from ouroboros.review_evidence import collect_review_evidence
    sig = inspect.signature(collect_review_evidence)
    default = sig.parameters["max_obligations"].default
    assert default is None, f"Expected None, got {default}"


def test_run_script_timeout_360():
    """run_script ToolEntry must stay foreground-bounded like run_command."""
    from ouroboros.tools.shell import get_tools
    entries = get_tools()
    rs = [e for e in entries if e.name == "run_script"]
    assert rs, "run_script not found in shell.get_tools()"
    assert rs[0].timeout_sec == 360


def test_advisory_pre_review_timeout_1200():
    """advisory_pre_review ToolEntry must declare timeout_sec=1200."""
    from ouroboros.tools.claude_advisory_review import get_tools
    entries = get_tools()
    apr = [e for e in entries if e.name == "advisory_review"]
    assert apr, "advisory_pre_review not found"
    assert apr[0].timeout_sec == 1200


def test_full_repo_pack_excludes_junk_dirs():
    """build_full_repo_pack must skip broad non-core directories."""
    from ouroboros.tools.review_helpers import _FULL_REPO_SKIP_DIR_PREFIXES
    for prefix in ("assets/", "tests/", "devtools/"):
        assert prefix in _FULL_REPO_SKIP_DIR_PREFIXES, f"{prefix} not in skip list"


def test_summary_and_reflection_callers_use_bounded_evidence():
    """Summary and reflection prompt builders must call format_review_evidence_for_prompt with max_chars."""
    from pathlib import Path

    for filename in ("ouroboros/post_task_synthesis.py", "ouroboros/reflection.py"):
        src = Path(filename).read_text(encoding="utf-8")
        assert "format_review_evidence_for_prompt(" in src
        # Must pass max_chars argument (not rely on default 0)
        assert "max_chars=" in src, f"{filename} must call format_review_evidence_for_prompt with max_chars"


def test_obligation_context_shows_all():
    """build_review_context must not slice open_obligations."""
    src = open("ouroboros/agent_task_pipeline.py", encoding="utf-8").read()
    assert "open_obs[:4]" not in src, "open_obs[:4] cap should be removed"
    assert "obligation_ids[:4]" not in src, "obligation_ids[:4] cap should be removed"


def test_unevidenced_reviewer_keeps_the_full_window_assumption(tmp_path, monkeypatch):
    """An UNKNOWN reviewer route must not be silently assumed to be small.

    v6.87.27: the window is resolved per reviewer from Capability Evidence, but
    the ABSENT-evidence default stays the full window — the same policy
    `context_fit` applies to the main lane (unknown routes try Max, never a
    silent 200K; BIBLE P1). Guessing small is not the conservative direction on
    these surfaces: the governance packs (BIBLE + DEVELOPMENT + ARCHITECTURE +
    CHECKLISTS) run ~169K tokens, so a sub-floor guess made `plan_slot_fit`
    decline EVERY plan review before dispatch on a cold-evidence install. Only
    scope review fails closed, because its blocking authority is what a wrong
    assumption would forge, and it applies that sub-floor itself.
    """
    from types import SimpleNamespace

    from ouroboros import capability_evidence
    from ouroboros.reviewer_window import REVIEWER_FULL_WINDOW, reviewer_context_window
    from ouroboros.tools.plan_review import _PLAN_REVIEW_MAX_TOKENS
    from ouroboros.tools.review_synthesis import per_slot_input_token_limits

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        capability_evidence, "probe", lambda *a, **k: SimpleNamespace(window_tokens=0, status=""),
    )

    assert reviewer_context_window("unevidenced/reviewer") == REVIEWER_FULL_WINDOW

    # ...and the resulting slot cap is byte-identical to the historical explicit
    # 1M call, so cold-evidence installs keep the pack they always had.
    limits = per_slot_input_token_limits(
        ["unevidenced/reviewer"],
        output_reserve=_PLAN_REVIEW_MAX_TOKENS,
        tokenizer_margin=155_000,
    )
    legacy = per_slot_input_token_limits(
        ["unevidenced/reviewer"],
        context_window=1_000_000,
        output_reserve=_PLAN_REVIEW_MAX_TOKENS,
        tokenizer_margin=155_000,
    )
    assert limits == legacy


def test_known_small_window_reviewer_gets_its_real_window(tmp_path, monkeypatch):
    """The D1 fix proper: CONFIRMED evidence replaces the 1M assumption."""
    from types import SimpleNamespace

    from ouroboros import capability_evidence
    from ouroboros.reviewer_window import reviewer_context_window
    from ouroboros.tools.plan_review import _PLAN_REVIEW_MAX_TOKENS
    from ouroboros.tools.review_synthesis import per_slot_input_token_limits

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        capability_evidence,
        "probe",
        lambda *a, **k: SimpleNamespace(window_tokens=200_000, status="confirmed"),
    )

    assert reviewer_context_window("small/reviewer") == 200_000
    limits = per_slot_input_token_limits(
        ["small/reviewer"], output_reserve=_PLAN_REVIEW_MAX_TOKENS, tokenizer_margin=155_000,
    )
    # Reserves scale to the real window instead of zeroing the slot, and the cap
    # is far below the 1M-sized one that would have drawn a provider 400.
    assert 0 < limits["small/reviewer"] < 200_000


def test_the_triad_sizes_its_shared_prompt_by_quorum_not_by_the_strictest_slot(monkeypatch):
    """The commit that made this change touched no test file, so reverting it to
    `min(...)` left the whole review suite green.

    It matters because the shrink ladder can only cut the diff and the file snapshots —
    the governance prefix is rebuilt unchanged on every rung and is ~138K tokens by
    itself. Sizing the SHARED prompt to a 200K reviewer (calibrated limit ~91K) made
    `_fit_shared_review_prompt` overflow whatever the change was, so every commit came
    back `fixed_overflow` telling the owner to split a diff that was never the cause.
    A small-window reviewer must drop out of the quorum, not shrink the gate.
    """
    import ouroboros.tools.review as review

    panel = ["big/one", "big/two", "small/one"]
    caps = {"big/one": 500_000, "big/two": 500_000, "small/one": 90_000}

    limit = review._quorum_input_token_limit(panel, caps)
    assert limit == 500_000, "the quorum-th largest cap, not the smallest"
    assert limit > min(caps.values()), "a sub-quorum window must not size the shared prompt"

    # With no quorum above it, the small window does bind — the panel cannot outvote it.
    two_slot = {"big/one": 500_000, "small/one": 90_000}
    assert review._quorum_input_token_limit(list(two_slot), two_slot) == 90_000



def _triad_assemble(review, stable_fields, dynamic_fields):
    """The production caller's assemble closure, rebuilt from the (monkeypatched)
    templates — the fused `_fit_triad_prompt` takes assembly from its caller
    (the 5.2/5.3 api/session split), so the test drives the same seam."""
    def _assemble(files_section, staged_diff):
        stable = review._REVIEW_PROMPT_TEMPLATE_STABLE.format(**stable_fields)
        dynamic = review._REVIEW_PROMPT_TEMPLATE_DYNAMIC.format(**{
            **dynamic_fields,
            "current_files_section": files_section,
            "diff_text": staged_diff,
        })
        return stable + "\n" + dynamic, len(stable) + 1
    return _assemble



def test_a_sub_quorum_window_degrades_its_own_slot_instead_of_blocking_the_commit_gate(
    monkeypatch, tmp_path,
):
    """The CALL SITE, pinned by what it DOES rather than by what it says.

    Asserting only the helper leaves `min(...)` free to come back in
    `_fit_shared_review_prompt` — which is exactly how this shipped uncovered. Pinning
    the call site with `inspect.getsource` kills that mutant but fails on correct code
    too: extracting the sizing into a helper, or renaming it, breaks a test about
    sizing behaviour for reasons that have nothing to do with sizing.

    So drive the assembler instead. The governance prefix is IRREDUCIBLE — the shrink
    ladder can only cut the diff and the file snapshots — so a prompt that clears the
    quorum but not the smallest slot is precisely the case that decides which limit the
    call site used, and it is the case the owner actually saw: every commit returning
    `fixed_overflow` and telling them to split a diff that was never the cause.
    """
    import ouroboros.tools.review as review

    caps = {"big/one": 500_000, "big/two": 500_000, "small/one": 90_000}
    monkeypatch.setattr(review, "reviewer_context_window", lambda model: caps[str(model)])
    monkeypatch.setattr(
        review, "calibrated_input_token_limit", lambda model, **kw: caps[str(model)],
    )
    monkeypatch.setattr(review, "run_cmd", lambda *a, **kw: "")  # no git in this unit
    monkeypatch.setattr(review, "_REVIEW_PROMPT_TEMPLATE_STABLE", "{preamble}")
    monkeypatch.setattr(
        review, "_REVIEW_PROMPT_TEMPLATE_DYNAMIC",
        "{current_files_section}\n{diff_text}\n{changed_files}",
    )

    # ~150K tokens of prefix: above the small slot's cap, below the quorum's, and beyond
    # the reach of every rung of the shrink ladder.
    stable_fields = {"preamble": "g" * (150_000 * 4)}
    dynamic_fields = {
        "changed_files": "a.py",
        "diff_text": "+x",
        "current_files_section": "full snapshot of a.py",
    }

    _assemble = _triad_assemble(review, stable_fields, dynamic_fields)

    def fit(models):
        return review._fit_triad_prompt(
            models, _assemble, dynamic_fields["current_files_section"],
            dynamic_fields["diff_text"], dynamic_fields["changed_files"], tmp_path,
        )

    _, _, overflow = fit(["big/one", "big/two", "small/one"])
    assert overflow == "", (
        "a quorum of large reviewers must carry the gate; the small slot degrades "
        f"its own review instead of blocking the commit: {overflow}"
    )

    # The other direction: with no quorum above it the small window really does bind, so
    # this is the QUORUM limit and not simply the largest slot's.
    _, _, overflow_two = fit(["big/one", "small/one"])
    assert "REVIEW_BLOCKED" in overflow_two


def _local_only_install(monkeypatch, tmp_path, n_ctx=16_384):
    """Simulate a supported local-only install and return the probe's call log.

    ``USE_LOCAL_MAIN=1`` with no cloud credential makes
    ``review_model_uses_local`` True for every slot. The probe fake mirrors
    ``capability_evidence.probe``'s routing contract: the LOCAL route answers
    from local health (the running model's n_ctx, which is what
    ``LOCAL_MODEL_CONTEXT_LENGTH`` configures), while the provider route
    inferred from the model TEXT is unreachable offline and yields no evidence
    — so a resolver that fingerprints the wrong route falls back to the
    unknown-route 1M sizing assumption."""
    from types import SimpleNamespace

    from ouroboros import capability_evidence

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("USE_LOCAL_MAIN", "1")
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY", "GIGACHAT_CREDENTIALS",
        "GIGACHAT_USER", "GIGACHAT_PASSWORD", "OPENAI_COMPATIBLE_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    calls = []

    def fake_probe(drive_root, **kwargs):
        calls.append(dict(kwargs))
        if kwargs.get("use_local"):
            return SimpleNamespace(
                window_tokens=n_ctx, status="confirmed", stale=False, ts="t",
            )
        return SimpleNamespace(window_tokens=0, status="unprobeable", stale=False, ts="")

    monkeypatch.setattr(capability_evidence, "probe", fake_probe)
    return calls


def test_reviewer_window_default_resolves_the_effective_local_route(monkeypatch, tmp_path):
    """The resolver's DEFAULT route is the dispatch route, not the model text.

    ``resolve_reviewer_window`` defaulted ``use_local=False``, so every caller
    that did not restate ``review_model_uses_local`` (triad ``_slot_input_limit``,
    plan ``per_slot_input_token_limits``, deep self-review) fingerprinted the
    provider inferred from the model id. On a local-only install that route has
    no evidence, so sizing fell back to the 1M assumption while the request
    itself was dispatched to a 16K local model. The predicate now lives in the
    resolver — one authority for every present and future caller."""
    calls = _local_only_install(monkeypatch, tmp_path)

    from ouroboros.reviewer_window import reviewer_context_window

    assert reviewer_context_window("openai/gpt-5.6-terra") == 16_384
    assert calls[-1]["use_local"] is True
    assert calls[-1]["provider"] == "local"


def test_plan_review_sizing_honors_the_local_route(monkeypatch, tmp_path):
    """XG-7A.1 call site 2: ``per_slot_input_token_limits`` on a local-only install.

    Before the fix this produced the 1M-derived cap — a plan prompt sized for a
    window the local model does not have, i.e. a guaranteed prompt-too-long
    failure on a supported install."""
    _local_only_install(monkeypatch, tmp_path)

    from ouroboros.tools.plan_review import _PLAN_REVIEW_MAX_TOKENS
    from ouroboros.tools.review_synthesis import per_slot_input_token_limits

    model = "openai/gpt-5.6-terra"
    limits = per_slot_input_token_limits(
        [model], output_reserve=_PLAN_REVIEW_MAX_TOKENS, tokenizer_margin=155_000,
    )
    # Byte-identical to sizing pinned at the local window...
    pinned_local = per_slot_input_token_limits(
        [model], context_window=16_384,
        output_reserve=_PLAN_REVIEW_MAX_TOKENS, tokenizer_margin=155_000,
    )
    assert limits == pinned_local
    assert 0 < limits[model] < 16_384
    # ...and far below the 1M-derived cap the old behavior produced.
    oversized = per_slot_input_token_limits(
        [model], context_window=1_000_000,
        output_reserve=_PLAN_REVIEW_MAX_TOKENS, tokenizer_margin=155_000,
    )
    assert limits[model] < oversized[model]


def test_triad_fit_sizes_against_the_local_route(monkeypatch, tmp_path):
    """XG-7A.1 call site 1: ``_fit_shared_review_prompt`` on a local-only install.

    Driven end to end through the REAL resolver (only the evidence probe is
    faked), so the test also fails if the call site regains an explicit
    ``use_local=False``. A ~40K-token irreducible governance prefix fits the
    old 1M assumption comfortably — the old behavior shipped it to the 16K
    local model — while honest local sizing must refuse it loudly."""
    _local_only_install(monkeypatch, tmp_path)

    import ouroboros.tools.review as review

    monkeypatch.setattr(review, "run_cmd", lambda *a, **kw: "")  # no git in this unit
    monkeypatch.setattr(review, "_REVIEW_PROMPT_TEMPLATE_STABLE", "{preamble}")
    monkeypatch.setattr(
        review, "_REVIEW_PROMPT_TEMPLATE_DYNAMIC",
        "{current_files_section}\n{diff_text}\n{changed_files}",
    )

    stable_fields = {"preamble": "g" * (40_000 * 4)}
    dynamic_fields = {
        "changed_files": "a.py",
        "diff_text": "+x",
        "current_files_section": "full snapshot of a.py",
    }

    _assemble = _triad_assemble(review, stable_fields, dynamic_fields)
    _, _, overflow = review._fit_triad_prompt(
        ["openai/gpt-5.6-terra"], _assemble,
        dynamic_fields["current_files_section"], dynamic_fields["diff_text"],
        dynamic_fields["changed_files"], tmp_path,
    )
    assert "REVIEW_BLOCKED" in overflow, (
        "a local-only install must size the triad prompt against the local "
        "route's real window, not the provider inferred from the model text"
    )


def test_update_letter_max_tokens_pinned():
    """The update letter is ONE short paragraph (owner decision 2026-09-03): its LIGHT
    one-shot budget stays 1024 and matches the ARCHITECTURE 'LLM output token budgets' row."""
    from ouroboros.update_letter import UPDATE_LETTER_MAX_TOKENS

    assert UPDATE_LETTER_MAX_TOKENS == 1024


def test_acceptance_panels_reach_the_synthesis_prompts():
    """AP7/F27: the post-task summariser and the reflection were told there was
    no review evidence even when the task bought an acceptance panel — the
    commit/advisory lens simply does not know about it. The panels ride along,
    and the absence statement names the lens it describes."""
    from ouroboros.review_evidence import format_review_evidence_for_prompt

    panels = [{
        "panel_id": "panel_1", "surface": "task_acceptance", "authority": "host",
        "aggregate_signal": "DEGRADED", "transport_status": "not_dispatched",
        "parse_status": "malformed", "superseded": False,
        "quorum": {"required": 2, "contributed": 0, "configured": 3},
        "reason": "slot_1:degraded_partial_source: the exact source is unavailable",
    }]
    out = format_review_evidence_for_prompt({}, max_chars=8000, acceptance_panels=panels)
    assert "TASK ACCEPTANCE PANELS" in out
    assert "no structured review evidence" not in out
    assert "DEGRADED" in out and "not_dispatched" in out
    assert '"required": 2' in out and '"contributed": 0' in out
    assert "degraded_partial_source" in out

    # No lens and no panels: the sentinel names WHICH evidence is absent.
    bare = format_review_evidence_for_prompt({}, max_chars=8000)
    assert bare == "(no commit/advisory review evidence recorded for this task)"

    # Both together, and the existing bound still applies.
    both = format_review_evidence_for_prompt(
        {"has_evidence": True, "task_id": "task-review", "data": "x" * 10_000}, max_chars=500,
        acceptance_panels=panels,
    )
    assert "OMISSION NOTE" in both
    assert "truncated at 500 chars" in both


def test_summary_and_reflection_callers_pass_the_acceptance_panels():
    from pathlib import Path

    # v7 relocated the summary/reflection synthesis callers out of
    # agent_task_pipeline.py into ouroboros/post_task_synthesis.py.
    for filename in ("ouroboros/post_task_synthesis.py", "ouroboros/reflection.py"):
        src = Path(filename).read_text(encoding="utf-8")
        assert "acceptance_panels=" in src, filename
