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


def _names_basename(text: str, basename: str) -> bool:
    """Whether ``text`` names this module file as its own token.

    The boundary is stated as "not a file-name character" rather than a list
    of allowed delimiters: the component map introduces modules after a space,
    a backtick, a path separator AND an opening parenthesis (``(clawhub.py
    registry client``), so an allow-list of delimiters would report a module
    the document does name. What must NOT precede the basename is a character
    that could be part of a longer file name — a word character, a dot or a
    hyphen — which is exactly how ``test_s3_task_control_browser.py`` used to
    answer for ``browser.py``. A trailing word character is refused too, so
    ``x.py`` never answers for ``x.pyi``.
    """
    return re.search(
        r"(?<![\w.\-])" + re.escape(basename) + r"(?!\w)", text
    ) is not None


def test_the_domain_quotient_report_ends_without_a_blank_line():
    """The report generator wrote a blank line at EOF, so the whitespace gate
    (`git diff --check`) was red on the one file nobody edits by hand.

    Its sections append a trailing "" separator, and `"\\n".join(L) + "\\n"` then
    turned the last separator into a blank final line. The generator now drops
    the trailing separators; this pins both the artifact and that fix, without
    pinning the report's CONTENT — the header carries a HEAD sha and a tree
    fingerprint, so byte-identity to a regeneration is deliberately not a gate
    (that gate belongs to `docs/DOMAIN_MAP.md`, whose input is the manifest).
    """
    report = _read("docs/v7next/DOMAIN_QUOTIENT_REPORT.md")
    generator = _read("scripts/v7next_domain_report.py")

    assert report.endswith("\n") and not report.endswith("\n\n")
    assert "while L and not L[-1]:" in generator


def test_the_domain_manifest_is_reachable_from_the_handbook():
    """The domain SSOT and its generated map were reachable from neither doc.

    A contributor is told to read DEVELOPMENT.md before moving code, and moving
    code across domains is exactly what `ouroboros/domains.toml` gates. With no
    pointer, the first they learned of it was a red `check_domains` run.
    """
    development = _read("docs/DEVELOPMENT.md")
    for pointer in ("ouroboros/domains.toml", "docs/DOMAIN_MAP.md",
                    "scripts/check_domains.py --write"):
        assert pointer in development, f"DEVELOPMENT.md never mentions {pointer}"


def test_recent_abi_retirements_section_carries_the_abi_70_window():
    """Section 11.4 documented a 5.25.0-rc.4 banner API and nothing since.

    ABI 7.0 is the largest retirement window in the project's history — five
    gateway aliases, the reviewer comma-list configuration keys, two wall-clock
    timeout keys, a plugin-API major, and a durable task-row schema stamp with
    no legacy converter — and an upgrading operator read "recent retirements"
    as though none of it happened. Every key of
    `RETIRED_COMMA_LIST_SETTING_KEYS` must be named there, because those are
    the ones whose migration must happen BEFORE the upgrade.
    """
    arch = _read("docs/ARCHITECTURE.md")
    section = arch.split("### 11.4 Recent ABI Retirements", 1)[1].split("\n## ", 1)[0]

    from ouroboros.settings_defaults import RETIRED_COMMA_LIST_SETTING_KEYS

    assert "**ABI 7.0**" in section
    assert "5.25.0-rc.4" in section, "older entries stay: this is a history"
    missing = [key for key in RETIRED_COMMA_LIST_SETTING_KEYS if key not in section]
    assert not missing, f"11.4 does not name the retired comma-list keys: {missing}"
    assert "OUROBOROS_REVIEWER_SLOTS" in section, "the migration target must be named"


def test_model_send_design_note_matches_the_landed_observability_contract():
    """The CPL-5 note still said DESIGN ONLY and demanded fail-closed dispatch.

    `ouroboros/model_send_seal.py` landed with the opposite rule, pinned by
    `tests/test_model_send_seal.py`: a reconstruction mismatch is a typed
    durable fact and the call is NOT blocked — dispatch authority stays with
    the pre-existing in-memory identity re-check. A design note that outranks
    the code it describes is how the next author reintroduces the gate.
    """
    note = _read("docs/v7next/DESIGN_MODEL_VISIBLE_LOGGED.md")
    note_flat = " ".join(note.split())

    assert (REPO / "ouroboros" / "model_send_seal.py").exists()
    assert "Status: DESIGN ONLY" not in note
    assert "Status: LANDED" in note
    assert "refuse dispatch with the existing `PhysicalAttemptPreparationFailed`" \
        not in note_flat
    assert "The call is NOT blocked" in note_flat


def test_settings_docs_name_every_key_owner_and_what_startup_persists():
    """Three settings-layer claims the code contradicts (all red pre-fix).

    1. The Default settings table skipped two live `SETTINGS_DEFAULTS` keys
       (`OUROBOROS_CONTEXT_MODE_AUTO_LOW`, `OUROBOROS_CLAWHUB_REGISTRY_URL`).
    2. Startup was described as persisting NOTHING, while the server lifespan's
       `load_settings()` runs `normalize_and_persist_context_mode_compat`,
       which rewrites the compat pair when it changed under a held lock.
    3. Invariant 3 and the README pointed at the `config.py` facade as the
       owner of defaults, after the v7next split moved the vocabularies into
       sibling leaves.
    """
    arch = _read("docs/ARCHITECTURE.md")
    arch_flat = " ".join(arch.split())
    readme_flat = " ".join(_read("README.md").split())
    development = _read("docs/DEVELOPMENT.md")

    from ouroboros.settings_defaults import SETTINGS_DEFAULTS

    for key in ("OUROBOROS_CONTEXT_MODE_AUTO_LOW", "OUROBOROS_CLAWHUB_REGISTRY_URL"):
        assert key in SETTINGS_DEFAULTS
        assert f"| {key} |" in arch, f"{key} is missing from the settings table"

    # Startup persistence: the compat migration is a real write, not "nothing".
    # (Other "persists nothing" statements in this document are about the
    # onboarding failure path and a no-change owner transform, both true.)
    assert "boot provider normalization in-process and persists nothing" not in arch_flat
    assert "Startup is a read, with one exception" in arch_flat
    assert "normalize_and_persist_context_mode_compat" in arch_flat
    # 4. What retired is the persistent auto-Low MECHANISM, not the key: the
    #    startup sentence called `OUROBOROS_CONTEXT_MODE` itself retired while
    #    the settings table right below it documents the same key as the live
    #    owner-selected horizon.
    assert "retired `OUROBOROS_CONTEXT_MODE`" not in arch_flat
    assert "left by the RETIRED persistent auto-Low mechanism" in arch_flat

    # Ownership: the leaves own the vocabularies; config.py stays the facade.
    owners = ("settings_defaults", "settings_scales", "model_slots",
              "review_model_routes", "runtime_limits", "settings_integrity")
    invariant = next(
        line for line in arch.splitlines()
        if line.startswith("3. **Configuration and messaging have single owners.**")
    )
    assert all(owner in invariant for owner in owners), invariant
    assert "exact settings and defaults live in" not in readme_flat
    assert "settings_defaults.py" in readme_flat
    assert "an SSOT in `config.py` `SETTINGS_DEFAULTS`" not in development
    assert "`settings_defaults.py`" in development


def test_architecture_does_not_claim_usage_response_is_the_only_usage_reader():
    """`_usage_response` normalizes for accounting; it does not own the block.

    The row claimed "The only reader of a provider's usage block", but every
    provider adapter reads the raw `usage` dict for its own response envelope
    (`llm_openai_compatible.py:285`, `llm_anthropic.py`, `llm_local.py`,
    `local_model.py`). A false absolute in the component map is how the next
    author "consolidates" a read that was never centralized here; the honest
    claim is the narrower one the two importers support.
    """
    arch_flat = " ".join(_read("docs/ARCHITECTURE.md").split())

    assert "The only reader of a provider's usage block" not in arch_flat
    assert "the one NORMALIZER of a provider's usage block" in arch_flat
    assert "Not the only READER of that block" in arch_flat
    for module in ("ouroboros/usage_accounting.py", "ouroboros/loop_llm_call.py"):
        assert "from ouroboros._usage_response import" in _read(module), module
    assert 'resp_dict.get("usage")' in _read("ouroboros/llm_openai_compatible.py")


def test_architecture_deep_review_has_no_compact_manifest_retry_rung():
    """The compact-manifest retry rung was removed; the doc still promised it.

    ``deep_self_review._compile`` is now called once with ``compact=True``
    because compact coverage IS the atlas default (the durable manifest keeps
    full per-file coverage either way), so there is no fuller form to fall back
    from. A failed assembly returns no pack at all (BIBLE P3). The
    final-shrink rebuild — one retry at a hard budget tightened by the measured
    overage — is a different rung and still exists.
    """
    arch_flat = " ".join(_read("docs/ARCHITECTURE.md").split())
    source = _read("ouroboros/deep_self_review.py")

    assert "no compact retry rung anymore" in source
    assert "retries once with the compact manifest" not in arch_flat
    assert "the compact manifest is the atlas default" in arch_flat
    assert "final-shrink rebuild" in arch_flat and "hard_budget_reduction" in source


def test_architecture_component_map_covers_every_live_runtime_module():
    """README calls ARCHITECTURE.md the full component map — so prove it.

    Every non-``__init__`` module of the tracked runtime population must be
    named in the component map (by path or by basename). A module nobody wrote
    a row for is invisible to the one document a contributor is told to read
    before editing, and the omission is silent: no other gate reads this
    document against the tree. Population comes from the domain manifest's own
    SSOT helper (``scripts/domain_graph.tracked_population``) so this pin and
    the domain gates can never disagree about what "live module" means.

    The basename must appear as its OWN token, not as a substring. A plain
    ``name in arch`` accepted a basename buried inside a longer name — so
    ``browser.py`` was "documented" by ``test_s3_task_control_browser.py``,
    ``health.py`` by ``extension_health.py`` and ``vision.py`` by
    ``delegate_supervision.py``, while those three modules had no row of any
    kind.
    """
    from scripts.domain_graph import tracked_population

    arch = _read("docs/ARCHITECTURE.md")
    missing = sorted(
        path for path in tracked_population(REPO)
        if not path.endswith("__init__.py")
        and path not in arch
        and not _names_basename(arch, pathlib.PurePosixPath(path).name)
    )
    assert not missing, (
        "docs/ARCHITECTURE.md names no owner for these live modules: "
        f"{missing}"
    )


def test_architecture_mentions_shared_log_grouping_and_direct_provider_review_fallback():
    arch = _read("docs/ARCHITECTURE.md")

    assert "log_events.js" in arch
    assert "live task card" in arch
    assert "grouped task cards" in arch
    # Direct-provider fallback covers official OpenAI, Anthropic, MiniMax, DeepSeek,
    # Cloud.ru, and GigaChat, while still excluding OpenRouter/OpenAI-compatible/mixed-provider configs.
    # Keep the generalized name ("Direct-provider review fallback") and a
    # reference to the legacy "OpenAI-only review fallback" phrase for
    # discoverability, and pin the honest scope language so the doc cannot
    # silently re-expand to claim symmetric coverage it does not have yet.
    assert "Direct-provider review fallback" in arch
    assert "OpenAI-only review fallback" in arch  # legacy name still referenced for discoverability
    assert "official OpenAI, Anthropic, MiniMax, DeepSeek, Cloud.ru, and GigaChat" in arch
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


def test_architecture_limits_finality_and_verdict_claims_to_actual_rows():
    arch = _read("docs/ARCHITECTURE.md")

    assert "The start row carries neither outcome finality nor a verdict" in arch
    assert "The pre-finalization authored row carries the phase with `outcome_final=false`" in arch
    assert "only terminal `task_summary` rows append the host verdict clause" in arch
    assert "Both Main rows, the Project thread rows" not in arch


def test_architecture_maps_cache_split_and_total_budget_authorities():
    arch = _read("docs/ARCHITECTURE.md")

    assert "_usage_cache_splits.py" in arch
    assert "process-local" in next(
        line for line in arch.splitlines() if "_usage_cache_splits.py" in line
    )
    settings_row = next(
        line for line in arch.splitlines() if "settings_setup_contract.py" in line
    )
    assert "resolve_total_budget_usd" in settings_row


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
    assert "state/skill_review_root_tasks.jsonl" in development
    assert "state/skill_review_root_tasks.jsonl" in architecture
    assert "SKILL_REVIEW_ROOT_TASKS_WARN_BYTES" in architecture
    assert "nine hot stores" in architecture
    assert "nine os.stat calls" in _read("ouroboros/agent_startup_checks.py")
    for item in (
        "source_completeness",
        "actor_readable_projection",
        "canonical_memory_fork",
        "review_artifact_continuity",
        "display_identity_replay",
    ):
        assert item in checklists


def test_architecture_names_all_window_surfaces_and_settlement_order():
    architecture = _read("docs/ARCHITECTURE.md")
    assert (
        "on those four surfaces (triad, plan review, task acceptance, and deep self-review)"
        in architecture
    )
    assert "SETTLED is published before registration retirement" in architecture


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
    # A fanned-out root also reports the depth REQUEST beside those lanes.
    assert "`requested_depth`" in arch
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
    # runtime-context fact (build_runtime_section -> update_letter.official_update_projection)
    "official_update",
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


# --- Documentation contract enforcement (DEVELOPMENT.md "Documentation contract") ---
#
# Three deterministic checks keep the two resident docs a present-tense map:
#   1. residue: version stamps, decision codenames and "used to / previously"
#      narrative may only SHRINK per `## ` section (baseline below, like the code
#      size ratchet — a section that reaches zero may never grow again);
#   2. the endpoint table mirrors the executable route registries;
#   3. the settings table mirrors `config.SETTINGS_DEFAULTS` (env-only and retired
#      rows are declared, not guessed).
# Language-tagged code fences (```yaml, ```python …) are examples and are not
# scanned; the plain ``` fence holding the §1 module tree IS scanned. The first
# ARCHITECTURE line carries the release version by contract and is skipped, as
# are DEVELOPMENT's "Mutable external-fact inventory" (dated provenance is the
# rule there) and the "Documentation contract" section that quotes the markers.

DOC_RESIDUE_PATTERNS = {
    "version_stamp": r"\((?:v\d+\.\d+(?:\.\d+)?(?:-rc\.\d+)?)\)",
    "version_narrative": r"\b(?:since|before|pre-)v\d+",
    "narrative": r"\b(?:used to|previously|formerly|was deleted|replaces the earlier|gate[- ]round|round \d+)\b",
    "codename_paren": r"\((?:GR|AR|BR|CR|D|Q|S|HQ|C|B)\d+[^)]{0,24}\)",
    "codename_word": r"\bPoltergeist\b|\bphase [A-C]\d?\b|owner(?:-| )(?:decision|ratif)",
}
DOC_RESIDUE_SKIPPED_SUBSECTIONS = {
    "docs/DEVELOPMENT.md": ("Mutable external-fact inventory", "Documentation contract"),
}


def doc_residue_counts(rel: str, text: str) -> dict:
    """Per-`## ` section counts of residue markers (see DOC_RESIDUE_PATTERNS)."""
    counts: dict = {}
    section = "(preamble)"
    fence_lang = None
    skipping = False
    skipped = DOC_RESIDUE_SKIPPED_SUBSECTIONS.get(rel, ())
    for lineno, line in enumerate(text.split("\n"), 1):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            fence_lang = (stripped[3:].strip() or "") if fence_lang is None else None
            continue
        if fence_lang:  # language-tagged example block
            continue
        if fence_lang is None and line.startswith("## "):
            section, skipping = line.strip(), False
        if fence_lang is None and line.startswith("### "):
            skipping = any(name in line for name in skipped)
        if skipping or (rel == "docs/ARCHITECTURE.md" and lineno == 1):
            continue
        for kind, pattern in DOC_RESIDUE_PATTERNS.items():
            hits = len(re.findall(pattern, line))
            if hits:
                counts.setdefault(section, {k: 0 for k in DOC_RESIDUE_PATTERNS})[kind] += hits
    return counts


# Zero-residue ceilings recorded after the 07b53365 cleanup: an absent section or kind is a
# zero ceiling, so both documents are held at zero everywhere. A section row exists only
# while a re-baseline records a temporary non-zero ceiling, and such a row may only shrink;
# an absent section stays at zero.
DOC_RESIDUE_BASELINE = {
    "docs/ARCHITECTURE.md": {},
    "docs/DEVELOPMENT.md": {},
}


def test_resident_docs_residue_only_shrinks():
    for rel, baseline in DOC_RESIDUE_BASELINE.items():
        current = doc_residue_counts(rel, _read(rel))
        for section, counts in current.items():
            allowed = baseline.get(section, {})
            for kind, hits in counts.items():
                assert hits <= allowed.get(kind, 0), (
                    f"{rel} {section!r}: {kind} residue grew to {hits} (baseline "
                    f"{allowed.get(kind, 0)}); replace the node's description instead of "
                    "appending history (DEVELOPMENT.md 'Documentation contract')"
                )


def _architecture_section(text: str, heading_prefix: str) -> str:
    lines = text.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith(heading_prefix))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start:end])


def test_architecture_endpoint_table_mirrors_route_registries(tmp_path):
    """Every mounted browser/CLI route and Host Service route has exactly one table
    row in ARCHITECTURE §4, and no row names a route that is not mounted."""
    from ouroboros.gateway.endpoint_index import HTTP_ENDPOINTS
    from ouroboros.gateway import files as gateway_files

    section = _architecture_section(_read("docs/ARCHITECTURE.md"), "## 4.")
    rows = re.findall(r"^\| (GET|POST|PUT|PATCH|DELETE|ANY|WS|STATIC) \| `([^`]+)` \|", section, re.M)
    host_prefix = "127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}"
    documented_public = {f"{m} {p}" for m, p in rows if not p.startswith(host_prefix)}
    documented_host = {f"{m} {p[len(host_prefix):]}" for m, p in rows if p.startswith(host_prefix)}

    expected_public = set(HTTP_ENDPOINTS)
    for route in gateway_files.file_browser_routes():
        for method in sorted(getattr(route, "methods", None) or []):
            if method in ("HEAD", "OPTIONS"):
                continue
            expected_public.add(f"{method} {route.path}")
    # server-level surfaces that are not gateway routes but belong in the map
    expected_public |= {"GET /", "WS /ws", "STATIC /static/*"}
    # /api/extensions/{skill}/{rest:path} is mounted for every verb and documented once as ANY
    expected_public = {e for e in expected_public if not e.endswith("/api/extensions/{skill}/{rest:path}")}
    expected_public.add("ANY /api/extensions/{skill}/{rest:path}")

    from starlette.routing import Route, WebSocketRoute

    from ouroboros.gateway.host_service import create_host_service_app

    expected_host = set()
    for route in create_host_service_app(tmp_path).routes:
        if isinstance(route, WebSocketRoute):
            expected_host.add(f"WS {route.path}")
        elif isinstance(route, Route):
            if route.methods is None:
                expected_host.add(f"ANY {route.path}")
            else:
                expected_host.update(
                    f"{method} {route.path}"
                    for method in route.methods
                    if method not in {"HEAD", "OPTIONS"}
                )
        else:
            raise AssertionError(f"Unhandled Host Service route type: {type(route).__name__}")

    assert documented_public == expected_public, (
        f"missing rows: {sorted(expected_public - documented_public)}; "
        f"stale rows: {sorted(documented_public - expected_public)}"
    )
    assert documented_host == expected_host, (
        f"missing host rows: {sorted(expected_host - documented_host)}; "
        f"stale host rows: {sorted(documented_host - expected_host)}"
    )
    assert len(rows) == len(documented_public) + len(documented_host), "duplicate endpoint rows"


# Rows the settings table documents on purpose although `config.SETTINGS_DEFAULTS`
# has no such key: operator env-only levers (never a settings.json carrier) and the
# retired alias whose migration the table still explains (pinned by test_review_cycles).
SETTINGS_TABLE_ENV_ONLY_ROWS = frozenset({
    "OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD", "OUROBOROS_DISABLE_MANAGED_UPDATES",
    "OUROBOROS_PRESENTATION", "OUROBOROS_USER_FILES_ROOT", "OUROBOROS_OBSERVABILITY_KEEP_RAW",
    "OUROBOROS_OBSERVABILITY_RETENTION_DAYS", "OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC",
    "OUROBOROS_REVIEW_MAX_TOKENS", "OUROBOROS_PREFLIGHT_TIMEOUT_SEC", "OUROBOROS_PREFLIGHT_SERIAL",
    "OUROBOROS_BUNDLE_DIR",
})
SETTINGS_TABLE_RETIRED_ROWS = frozenset({"OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES"})


def _normalize_default_cell(cell: str) -> str:
    value = cell.strip()
    value = re.sub(r"^`(.*)`$", r"\1", value)
    if value in ('""', "(empty)", "(unset)", "unset", ""):
        return ""
    value = re.sub(r'^"(.*)"$', r"\1", value)
    return value.split()[0].lower() if value else ""


def test_architecture_settings_table_mirrors_config_defaults():
    """Every `config.SETTINGS_DEFAULTS` key has one row in ARCHITECTURE's Default
    settings table with the shipped default; every other row is a declared env-only
    lever or retired alias."""
    from ouroboros import config

    section = _architecture_section(_read("docs/ARCHITECTURE.md"), "## 7.")
    table = section[section.index("### Default settings"):]
    rows = re.findall(r"^\| ([A-Z][A-Z0-9_]+) \| ([^|]*?) \|", table, re.M)
    keys = [k for k, _ in rows]
    assert len(keys) == len(set(keys)), f"duplicate settings rows: {sorted(k for k in set(keys) if keys.count(k) > 1)}"
    documented = set(keys)
    expected = set(config.SETTINGS_DEFAULTS)
    assert expected <= documented, f"settings missing from the table: {sorted(expected - documented)}"
    undeclared = documented - expected - SETTINGS_TABLE_ENV_ONLY_ROWS - SETTINGS_TABLE_RETIRED_ROWS
    assert not undeclared, f"table rows that are neither shipped defaults nor declared env-only/retired: {sorted(undeclared)}"
    mismatched = [
        (key, cell, str(config.SETTINGS_DEFAULTS[key]))
        for key, cell in rows
        if key in expected and _normalize_default_cell(cell) != str(config.SETTINGS_DEFAULTS[key]).lower()
    ]
    assert not mismatched, f"documented default differs from config.SETTINGS_DEFAULTS: {mismatched}"


def test_the_handbook_names_both_layers_of_the_browser_no_undef_gate():
    """Two surfaces answer "does every browser identifier resolve?": the
    dependency-free acorn walker the hermetic commit gate runs, and ESLint's
    `no-undef` that CI runs as an independent second opinion (D-13). A
    contributor who sees only one of them either removes the "redundant"
    other or, seeing a CI-only red, looks for a gate that never ran it.
    """
    development = _read("docs/DEVELOPMENT.md")
    for pointer in ("web/tests/no_undef.test.js", "web/eslint.config.js", "npm ci"):
        assert pointer in development, f"DEVELOPMENT.md never mentions {pointer}"
