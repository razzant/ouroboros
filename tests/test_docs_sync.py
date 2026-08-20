"""Guardrails for architecture docs after UI/routing overhaul.

README prose pins were retired in v5.8.3-rc.5 — the README is intentionally
allowed to evolve its marketing copy without dragging tests along; the
ARCHITECTURE.md pins below are the load-bearing rationale-layer guards
(P6) that must survive every doc-touch commit.
"""

import os
import pathlib

REPO = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_architecture_mentions_shared_log_grouping_and_direct_provider_review_fallback():
    arch = _read("docs/ARCHITECTURE.md")

    assert "log_events.js" in arch
    assert "live task card" in arch
    assert "grouped task cards" in arch
    # Direct-provider fallback covers official OpenAI, Anthropic, MiniMax, Cloud.ru,
    # and GigaChat, while still excluding OpenRouter/OpenAI-compatible/mixed-provider configs.
    # Keep the generalized name ("Direct-provider review fallback") and a
    # reference to the legacy "OpenAI-only review fallback" phrase for
    # discoverability, and pin the honest scope language so the doc cannot
    # silently re-expand to claim symmetric coverage it does not have yet.
    assert "Direct-provider review fallback" in arch
    assert "OpenAI-only review fallback" in arch  # legacy name still referenced for discoverability
    # v6.103.0 rewrote the fallback to compile each provider's declarative
    # reviewer-role sequence; the provider scope is pinned via the roles the
    # paragraph names rather than the retired list sentence.
    assert "declarative reviewer-role sequence" in arch
    assert "OpenAI and Anthropic run three independent Main-model slots" in arch
    assert "Cloud.ru and GigaChat use their one available role model" in arch
    assert "_exclusive_direct_remote_provider_env" in arch
    # v4.34.0: direct-provider fallback now documents the
    # `main_model.startswith(provider_prefix)` guard in get_review_models —
    # previously absent, allowing OpenAI/Anthropic-only setups with a
    # cross-provider free-text main model to silently miss the fallback.
    assert "migrate_model_value" in arch
    assert "already start with the exclusive provider prefix" in arch
    # v4.34.0: Claude Runtime Status doc widened to cover both backend and
    # browser-side `catch` block paths that set `claudeRuntimeHasError`.
    assert "refreshClaudeCodeStatus" in arch
    assert "transport failure" in arch


def test_architecture_documents_skill_schedule_lifecycle_and_evolution_light_block():
    arch = _read("docs/ARCHITECTURE.md")

    # v6.9 RC2: skill schedule readiness SSOT, lifecycle resync, tombstone
    # retention, DST contract, and the evolution light-mode hard block.
    assert "resync_skill_schedules()" in arch
    assert "skill_readiness_for_execution()" in arch
    assert "DST-aware system" in arch
    assert "hard-blocked in `light` runtime mode" in arch
    # Experience Review memory write-back data flow is documented.
    assert "MEMORY_ACTIONS_JSON" in arch
    assert "apply_memory_actions" in arch
    assert "never auto-written to `identity.md`" in arch


def test_consciousness_prompt_matches_scope_limited_contracts():
    consciousness = _read("prompts/CONSCIOUSNESS.md")

    assert "schedule subagents" in consciousness
    assert "wait on subagents" in consciousness
    assert "Update your scratchpad or identity" in consciousness
    assert "Message the user proactively" in consciousness
    assert "recent_tasks" in consciousness


def test_phase3_governance_language_is_pinned_without_new_qa_surface():
    bible = _read("BIBLE.md")
    development = _read("docs/DEVELOPMENT.md")
    system = _read("prompts/SYSTEM.md")
    authoring = _read("docs/CREATING_SKILLS.md")
    architecture = _read("docs/ARCHITECTURE.md")
    checklists = _read("docs/CHECKLISTS.md")
    development_flat = " ".join(development.split())

    assert (
        "Uncertainty calls for judgment, not permission: within its legitimate "
        "authority, Ouroboros decides autonomously."
    ) in bible
    assert (
        "Structural depth is not scope breadth: choose the smallest change that "
        "eliminates the proven failure class."
    ) in bible

    for principle in (
        "Single Responsibility Principle",
        "Open/Closed Principle",
        "Liskov Substitution Principle",
        "Interface Segregation Principle",
        "Dependency Inversion Principle",
    ):
        assert principle in development
    assert "DI container" in development
    assert "AST analyzer" in development
    assert "Diff size, line count, and file count alone are not findings" in development

    assert "Mutable external-fact inventory" in development
    for column in (
        "Location",
        "Fact",
        "Mutability",
        "Current authority",
        "Live/probe option",
        "Risk",
        "Recommendation",
    ):
        assert f"| {column} " in development
    assert "does not migrate their runtime representations" in development_flat

    for text in (development, system, authoring, architecture, checklists):
        flat = " ".join(text.split())
        assert "real consumer flow" in flat
        assert "screenshot" in flat.lower()
        assert "vision" in flat.lower()
        assert "not a universal" in flat or "not universal" in flat or "no universal" in flat
    assert "No visual-QA runner, endpoint, ledger" in " ".join(architecture.split())


def test_phase3_widget_authoring_docs_match_recursive_schema_v1():
    development = _read("docs/DEVELOPMENT.md")
    authoring = _read("docs/CREATING_SKILLS.md")
    architecture = _read("docs/ARCHITECTURE.md")
    checklists = _read("docs/CHECKLISTS.md")
    authoring_flat = " ".join(authoring.split())

    for text in (development, authoring, architecture, checklists):
        for component in ("group", "metric", "callout"):
            assert component in text
    assert "maximum depth of 8" in authoring
    assert "256 nodes" in authoring
    assert "stable tree path" in authoring
    assert "transitively passive" in authoring_flat
    assert "dynamic_ui_schema" in authoring


def test_architecture_mirror_matches_the_split_axes_contracts():
    """XG-2.2/XG-2.3 (v6.87.28 review gate): the P6 mirror tracks schedule-vs-dispatch.

    Every pin here failed on the pre-fix docs: the module map claimed
    `swarm_efficiency.lanes_used` while `_build_swarm_efficiency` emits
    `lanes_requested`; `subagents.py` was said to own task-group compaction after
    `compact_task_group` was deleted; the control map said `schedule_subagent`
    surfaces effective lane(s) after `_finalize_schedule_emission` went
    request-only; the `swarm_fanout` enumeration promised requested/effective
    lanes after `_emit_swarm_fanout` dropped `effective_model_lanes`; and both
    `wait_tasks` field enumerations omitted the emitted `capability_delta`.
    """
    arch = _read("docs/ARCHITECTURE.md")
    development = _read("docs/DEVELOPMENT.md")
    arch_flat = " ".join(arch.split())
    dev_flat = " ".join(development.split())

    # swarm_efficiency reports the REQUEST: lanes_requested, never lanes_used.
    assert "lanes_requested" in arch
    assert "lanes_used" not in arch
    # Task-group compaction left with the degenerate lane fan-out (v6.87.28).
    assert "task-group compaction" not in arch
    assert "compact_task_group" not in arch
    # schedule_subagent reports the request only; the axes resolve at dispatch.
    assert "schedule_subagent surfaces effective_lane(s)" not in arch_flat
    assert "schedule_subagent reports the requested lane only" in arch_flat
    # swarm_fanout carries the requested lane; a wave event written before any
    # child starts cannot know what the children ran on.
    assert "requested/effective lanes" not in arch
    # Both wait_tasks projection enumerations disclose capability_delta.
    assert "trace_summary, capability_delta when the child has something to disclose" in arch_flat
    assert "trace_summary, capability_delta when disclosable, duplicate_of" in dev_flat
