"""Guardrails for architecture docs after UI/routing overhaul.

README prose pins were retired in v5.8.3-rc.5 — the README is intentionally
allowed to evolve its marketing copy without dragging tests along; the
ARCHITECTURE.md pins below are the load-bearing rationale-layer guards
(P6) that must survive every doc-touch commit.
"""

import os
import pathlib
import re

from ouroboros.tools.registry import ToolRegistry

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
    assert "official OpenAI, Anthropic, MiniMax, Cloud.ru, and GigaChat" in arch
    assert "_exclusive_direct_remote_provider_env" in arch
    # v4.34.0: direct-provider fallback now documents the
    # `main_model.startswith(provider_prefix)` guard in get_review_models —
    # previously absent, allowing OpenAI/Anthropic-only setups with a
    # cross-provider free-text main model to silently miss the fallback.
    assert "migrate_model_value" in arch
    assert "already start with the exclusive provider prefix" in arch
    # The Claude Runtime Status surface is RETIRED with the Claude-SDK
    # advisory transport (owner-consented, 2026-08-29): the doc must not
    # resurrect its UI plumbing.
    assert "refreshClaudeCodeStatus" not in arch
    assert "claudeRuntimeHasError" not in arch


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


def test_chat_id_addressing_docs_match_the_code_that_routes_it():
    """The chat-0 doctrine is documented where implementers read it (P6).

    ARCHITECTURE previously claimed history replay renders chat-0 frames in the
    Main stream; the Main filter drops them, so a reader following the doc would
    look for a CLI run's dialogue in a surface that never shows it. That claim
    must never come back.
    """
    arch = _read("docs/ARCHITECTURE.md")
    development = _read("docs/DEVELOPMENT.md")

    assert "renders them in the Main stream" not in arch
    assert "a `chat_id=0` history query coerces to Main" in arch
    assert "HIDDEN_CHAT_ID" in arch
    # The headless address is decided at admission, and both outcomes are stated.
    assert "log_addressing.ingress_chat_id" in arch
    assert "is refused with a typed 400 rather than honoured" in arch
    assert "has exactly ONE destination" in arch
    assert "the only address is `HIDDEN_CHAT_ID` (0)" in arch
    assert "Registration alone does not qualify" in arch
    assert "admitted into that project's thread" in arch
    assert "stays in the hidden partition, silent in every chat" in arch
    # Scoped is not bound, so the absent conversion button is documented intent.
    assert "Project-SCOPED is not project-BOUND" in arch
    # Naming is part of the same admission contract, and its two slots differ.
    assert "The run is also NAMED at admission, without a model call" in arch
    assert "`metadata.title` is refused with a" in arch
    assert "never outranks a real name coined later" in arch
    # A degraded delivery names its own cause. The doc must keep saying which
    # code each rail actually produces — the forced rail keeps its own — rather
    # than renaming one after the other.
    assert "which is this" in arch and "forced rail's own code" in arch
    assert "the ordinary repair path records `invalid_delivery_control_after_repair`" in arch
    assert "falls back to `delivery_control_degraded` only for a" in arch
    design = _read("docs/DESIGN.md")
    assert "Where a card does show a cause, it says it in the owner's" in design
    assert "the record keeps the machine code" in design
    # The rule itself lives with the other anti-patterns, not only in a changelog.
    assert "Anti-pattern: a chat id tested for truth" in development
    assert "notification_chat_route" in development and "coerce_chat_identity" in development
    assert "tests/test_chat_id_truthiness_guard.py" in development


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


def test_continuity_projection_contract_is_mirrored_across_governance_docs():
    """Keep the partial-input rule and its concrete data-flow map from drifting."""
    bible = " ".join(_read("BIBLE.md").split()).replace("**", "")
    architecture = " ".join(_read("docs/ARCHITECTURE.md").split())
    development = " ".join(_read("docs/DEVELOPMENT.md").split())
    checklists = _read("docs/CHECKLISTS.md")

    assert (
        "Disclosure is not sufficiency. An omission marker keeps a record honest; "
        "it does not make the record complete. Where material is omitted, the "
        "disclosure must name a source this actor can actually resolve. A view known "
        "to be partial may not authorize PASS, a destructive rewrite, or replacement "
        "of the full contract it was cut from."
    ) in bible
    assert "Continuity data-flow map" in architecture
    assert "state/consciousness_observations.jsonl" in architecture
    assert "Source-complete decision pipeline" in development
    assert "Context and growth matrix" in development
    for item in (
        "source_completeness",
        "actor_readable_projection",
        "canonical_memory_fork",
        "review_artifact_continuity",
        "display_identity_replay",
    ):
        assert item in checklists


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
    assert "`schedule_subagent` reports the requested lane only" in arch_flat
    # swarm_fanout carries the requested lane; a wave event written before any
    # child starts cannot know what the children ran on.
    assert "requested/effective lanes" not in arch
    # Both wait_tasks projection enumerations disclose capability_delta.
    assert "trace_summary, capability_delta when the child has something to disclose" in arch_flat
    assert "trace_summary, capability_delta when disclosable, duplicate_of" in dev_flat


# Identifiers the prompts legitimately name in backticks that are NOT tools:
# parameter names, resource roots, write surfaces, typed outcome/status tokens
# and runtime-context keys. A NEW snake_case identifier in a prompt must either
# be a real tool (or background-whitelisted tool) or be classified here on
# purpose — that classification step is the governance the prompt audit wants:
# a phantom or renamed tool name can no longer hide in the runtime prompts
# (`advisory_review` and the CONSCIOUSNESS "You can" catalog rotted that way).
# Scope: backticked names in all three prompts plus the bare snake_case names
# CONSCIOUSNESS.md writes without backticks; BIBLE.md is deliberately out of scope.
PROMPT_NON_TOOL_IDENTIFIERS = frozenset({
    # resource roots / write surfaces / write roots
    "active_workspace", "artifact_store", "external_workspace", "runtime_data",
    "skill_payload", "subagent_projects", "system_repo", "task_drive", "user_files",
    "write_root", "write_surface",
    # tool parameters named as cross-tool policy
    "project_id", "project_name", "recommended_use", "review_rebuttal",
    # typed outcomes / statuses / runtime-context keys
    "needs_manual_target", "started_uncustodied", "owner_client",
    # safety policy class names (ouroboros/safety.py TOOL_POLICY values) and
    # owner-setting values named as policy
    "check_conditional", "check", "off", "low",
    # package managers / interpreters named as acquisition or process choices
    "pip", "pip3", "uv", "brew", "apt", "python", "python3", "sudo", "grep", "env",
    # git branches / remotes / skill buckets / write surfaces named as policy
    "ouroboros", "main", "managed", "origin", "external", "genesis", "deliverables",
    # fenced-block languages the owner chat renders natively
    "mermaid", "chart",
    # backlog item status value in CONSCIOUSNESS.md
    "done",
})


def _prompt_backticked_identifiers(text: str) -> set:
    """Backticked lowercase identifiers, single-word ones included (a renamed
    single-word tool such as `escalate` must be caught too)."""
    found = set()
    for token in re.findall(r"`([^`]+)`", text):
        head = token.split("(", 1)[0]
        if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", head):
            found.add(head)
    return found


def _prompt_bare_identifiers(text: str) -> set:
    """snake_case tokens written WITHOUT backticks (CONSCIOUSNESS.md's style);
    tokens that are part of a path or filename (`a/b_c`, `x_y.json`) are skipped."""
    return {
        m.group(1)
        for m in re.finditer(r"(?<![\w/.`-])([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?![\w/.`-])", text)
    }


def test_prompt_tool_names_resolve_to_registered_tools(tmp_path):
    """Every backticked snake_case identifier in the three runtime prompts is
    either a registered tool (public schema), a background-consciousness tool,
    or a documented non-tool identifier. Completeness is deliberately NOT
    required (the schemas are the catalog); this only forbids phantoms and
    stale spellings, the drift class the prompt audit found in every prompt."""
    from ouroboros.consciousness import BackgroundConsciousness

    root = pathlib.Path(__file__).resolve().parent.parent
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    registered = {schema["function"]["name"] for schema in registry.schemas()}
    # The background whitelist is not taken on faith: every name in it must be a
    # registered public tool or a ToolEntry the consciousness module registers
    # itself (set_next_wakeup and friends), otherwise the whitelist has rotted.
    consciousness_src = (root / "ouroboros" / "consciousness.py").read_text(encoding="utf-8")
    bg_private = set(re.findall(r'ToolEntry\("([a-z0-9_]+)"', consciousness_src))
    stale_whitelist = set(BackgroundConsciousness._BG_TOOL_WHITELIST) - registered - bg_private
    assert not stale_whitelist, f"_BG_TOOL_WHITELIST names unregistered tools: {sorted(stale_whitelist)}"
    universe = (
        registered
        | set(BackgroundConsciousness._BG_TOOL_WHITELIST)
        | PROMPT_NON_TOOL_IDENTIFIERS
    )
    # CONSCIOUSNESS.md runs on the background registry, which admits ONLY the
    # whitelist (consciousness.py _tool_schemas/_execute_tool), so a public tool
    # that is not whitelisted is a phantom there.
    bg_universe = set(BackgroundConsciousness._BG_TOOL_WHITELIST) | PROMPT_NON_TOOL_IDENTIFIERS
    for rel, allowed in (
        ("prompts/SYSTEM.md", universe),
        ("prompts/SAFETY.md", universe),
        ("prompts/CONSCIOUSNESS.md", bg_universe),
    ):
        text = (root / rel).read_text(encoding="utf-8")
        unresolved = _prompt_backticked_identifiers(text) - allowed
        assert not unresolved, (
            f"{rel} names identifiers that are neither registered tools nor "
            f"classified non-tool identifiers: {sorted(unresolved)}"
        )
    # CONSCIOUSNESS.md writes tool names without backticks; its bare snake_case
    # tokens must resolve the same way (the runtime drift check in
    # context_health only catches names with known prefixes).
    bare = _prompt_bare_identifiers((root / "prompts" / "CONSCIOUSNESS.md").read_text(encoding="utf-8"))
    unresolved_bare = bare - bg_universe
    assert not unresolved_bare, (
        f"prompts/CONSCIOUSNESS.md names bare identifiers that are neither registered tools "
        f"nor classified non-tool identifiers: {sorted(unresolved_bare)}"
    )
