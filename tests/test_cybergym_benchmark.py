"""Pure structural checks for the CyberGym benchmark profile and documentation.

These tests intentionally do not import CyberGym, Docker, inspect, browser, or
any other optional evaluator dependency.  They validate the repository-owned
contract that a launcher must apply and attest at run time.
"""

from __future__ import annotations

import json
from pathlib import Path

from devtools.benchmarks.common import launcher_audit
from ouroboros.configured_subagents import parse_configured_subagents
from ouroboros.reviewer_slot_config import parse_reviewer_slots

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "devtools" / "benchmarks" / "cybergym" / "settings_base.json"
MODEL = "deepseek/deepseek-v4-flash-0731"


def _settings() -> dict[str, object]:
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def test_profile_pins_one_canonical_model_and_review_panel():
    settings = _settings()
    assert settings["OUROBOROS_MODEL"] == MODEL
    configured = parse_configured_subagents(settings["OUROBOROS_SUBAGENTS"])
    # The template keeps the canonical actor available for review/copying; the
    # launcher turns it off in the applied cohort snapshot.
    assert configured.enabled is True
    assert len(configured.items) == 1
    actor = configured.items[0]
    assert actor.subagent_id == "benchmark-model"
    assert actor.route.kind == "api_model"
    assert actor.route.credential_profile_id == ""
    assert actor.route.target_id == MODEL

    active_slots = (
        "OUROBOROS_MODEL",
        "OUROBOROS_MODEL_LIGHT",
        "OUROBOROS_MODEL_VISION",
        "OUROBOROS_MODEL_CONSCIOUSNESS",
        "OUROBOROS_MODEL_FALLBACKS",
        "OUROBOROS_MODEL_DEEP_SELF_REVIEW",
        "OUROBOROS_WEBSEARCH_MODEL",
    )
    for key in active_slots:
        assert settings[key] == MODEL, key
    # ABI 7.0 (ABI-10): the comma keys are retired — the structured slots
    # below are the ONE reviewer configuration surface of the profile.
    assert "OUROBOROS_REVIEW_MODELS" not in settings
    assert "OUROBOROS_SCOPE_REVIEW_MODELS" not in settings
    assert "CLAUDE_CODE_MODEL" not in settings  # retired transport setting
    assert "OUROBOROS_MODEL_HEAVY" not in settings
    assert "USE_LOCAL_HEAVY" not in settings

    reviewers = parse_reviewer_slots(settings["OUROBOROS_REVIEWER_SLOTS"])
    assert [row.target_id for row in reviewers.triad] == [MODEL]
    assert [row.target_id for row in reviewers.scope] == [MODEL]
    assert all(row.effort == "max" for row in (*reviewers.triad, *reviewers.scope))
    assert all(not row.is_session for row in (*reviewers.triad, *reviewers.scope))
    assert reviewers.advisory.enabled is False


def test_profile_records_safe_runtime_and_budget_defaults():
    settings = _settings()
    assert settings["OUROBOROS_MAX_SUBAGENT_DEPTH"] == 0
    assert settings["OUROBOROS_MAX_WORKERS"] > 1
    assert settings["OUROBOROS_TASK_ABS_CEILING_SEC"] == 14_400
    assert settings["TOTAL_BUDGET"] == 3_500.0
    assert settings["OUROBOROS_RUNTIME_MODE"] == "pro"
    assert settings["OUROBOROS_SAFETY_MODE"] == "off"
    assert settings["OUROBOROS_CONTEXT_MODE"] == "max"
    assert settings["OUROBOROS_TASK_REVIEW_MODE"] == "required"
    assert settings["OUROBOROS_REVIEW_ENFORCEMENT"] == "advisory"
    assert settings["OUROBOROS_REVIEW_MAX_CYCLES"] == "2"
    for key in (
        "OUROBOROS_EFFORT_TASK",
        "OUROBOROS_EFFORT_EVOLUTION",
    ):
        assert settings[key] == "high", key
    for key in (
        "OUROBOROS_EFFORT_REVIEW",
        "OUROBOROS_EFFORT_SCOPE_REVIEW",
        "OUROBOROS_EFFORT_DEEP_SELF_REVIEW",
    ):
        assert settings[key] == "max", key
    for key in (
        "OUROBOROS_EFFORT_CONSCIOUSNESS",
    ):
        assert settings[key] == "high", key
    assert settings["OUROBOROS_POST_TASK_EVOLUTION"] == "false"
    assert settings["OUROBOROS_MAIN_WEB_SEARCH"] == "off"
    assert settings["OUROBOROS_MAIN_WEB_SEARCH_ENGINE"] == "auto"
    assert settings["OUROBOROS_MAIN_WEB_SEARCH_MAX_TOTAL_RESULTS"] == 0
    assert settings["OUROBOROS_WEBSEARCH_BACKEND"] == "ddgs"
    assert settings["MCP_ENABLED"] is False
    assert settings["MCP_SERVERS"] == []
    # The template remains neutral; the launcher applies a run-specific
    # runtime cap (covered by the protocol tests) instead of inheriting a
    # hidden runtime default.
    assert "OUROBOROS_PER_TASK_COST_USD" not in settings


def test_provider_template_is_override_ready_not_a_stale_allowlist():
    provider = json.loads(_settings()["OUROBOROS_OR_PROVIDER"])
    assert provider == {"allow_fallbacks": True, "require_parameters": True}
    assert "only" not in provider
    assert "order" not in provider


def test_template_does_not_carry_credentials_or_local_routes():
    settings = _settings()
    secret_fragments = ("KEY", "TOKEN", "PASSWORD", "CREDENTIAL", "SECRET")
    for key, value in settings.items():
        if any(fragment in key.upper() for fragment in secret_fragments):
            assert value in ("", None), key
    for key in (
        "USE_LOCAL_MAIN",
        "USE_LOCAL_LIGHT",
        "USE_LOCAL_FALLBACK",
        "USE_LOCAL_CONSCIOUSNESS",
    ):
        assert settings[key] is False
    assert "OUROBOROS_NETWORK_PASSWORD" not in settings
    assert "GITHUB_TOKEN" not in settings
    assert settings["LOCAL_MODEL_SOURCE"] == ""
    assert settings["LOCAL_MODEL_FILENAME"] == ""


def test_cybergym_is_registered_and_structural_test_lists_are_synchronized():
    relative = "cybergym/run_cybergym.py"
    assert relative in launcher_audit.MIGRATED_LAUNCHERS
    structural = (REPO / "tests" / "test_devtools_benchmarks.py").read_text(encoding="utf-8")
    assert '"devtools/benchmarks/cybergym/run_cybergym.py"' in structural
    assert 'bench / "cybergym" / "run_cybergym.py"' in structural


def test_benchmark_inventory_points_to_cybergym_docs():
    common_readme = (REPO / "devtools" / "benchmarks" / "README.md").read_text(
        encoding="utf-8"
    )
    architecture = (REPO / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "cybergym/" in common_readme
    assert "devtools/benchmarks/cybergym/" in architecture


def test_cybergym_docs_pin_the_owner_approved_contract():
    readme = (PROFILE.parent / "README.md").read_text(encoding="utf-8")
    methodology = (PROFILE.parent / "METHODOLOGY.md").read_text(encoding="utf-8")
    combined = f"{readme}\n{methodology}"
    required = (
        MODEL,
        "7656b71d07da6694e262f9c34ea994cd4849c0eb",
        "bde190ded494e52bc684b66073b436c9d992c7c6",
        "9cea452cc1e1a3703e0f60c2dfc8642430aab9f50433f976581509de58c7048f",
        "1,507",
        "Level 1",
        "final.poc",
        "any-of",
        "raw_vul_exit_code not in {0, 71, 300}",
        "raw_fix_exit_code == 0",
        "build_isolated_settings",
        "OUROBOROS_OR_PROVIDER",
        "only",
        "order",
        "cybergym-internal",
        "Internal=false",
        "unrestricted outbound",
        "mandatory trajectory audit",
        "issue tracker or bug reports",
        "ready-made PoC",
        "rootless",
        "Docker `--network host`",
        "OUROBOROS_MAX_SUBAGENT_DEPTH=0",
        "schedule_subagent",
        "delegate_start",
        "claude_code_edit",
        "OUROBOROS_TASK_ABS_CEILING_SEC=14400",
        "USD 3,500",
        "eight hours",
        "admit_benchmark_run",
        "finalize_run_manifest",
        "append-only",
        "cleanup",
    )
    for phrase in required:
        assert phrase in combined, phrase
    assert "template" in readme.lower()
    assert "applied" in readme.lower()
    assert "leaderboard" in readme.lower()
