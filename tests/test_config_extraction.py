"""Structural contracts for the semantic-no-op settings-configuration extraction."""

from __future__ import annotations

import ast
import pathlib

from ouroboros import (
    config,
    model_slots,
    provider_models,
    review_model_routes,
    runtime_limits,
    settings_defaults,
    settings_scales,
)

REPO = pathlib.Path(__file__).parents[1]
PACKAGE = REPO / "ouroboros"

_LEAVES = (settings_defaults, settings_scales, model_slots, review_model_routes, runtime_limits)

_MOVED_OWNERS = {
    "ENDPOINT_AUTHORED_SETTINGS": settings_defaults,
    # v6.104.0 upstream: the OpenRouter shipped-model defaults arrive in the
    # vocabulary leaf the v7 split created for exactly this class of fact.
    "OPENROUTER_DEFAULTS": settings_defaults,
    "OPENROUTER_REVIEW_DEFAULTS": settings_defaults,
    "FINALIZATION_GRACE_DEFAULT_SEC": settings_defaults,
    "OWNER_STOP_OUTER_CAP_SEC": settings_defaults,
    "PACING_INTERVAL_DEFAULT_SEC": settings_defaults,
    "RETIRED_SETTING_KEYS": settings_defaults,
    # ABI 7.0 (ABI-7b): the comma-list classification INSIDE the retirement
    # SSOT, born in this leaf (not an extraction) for the RC auditor to snap.
    "RETIRED_COMMA_LIST_SETTING_KEYS": settings_defaults,
    # The stage-2 close-out's second classification inside that same SSOT: the
    # retired keys whose successor SETTING the table states, so the first-boot
    # notice can name it instead of claiming there is none.
    "RETIRED_SETTING_SUCCESSORS": settings_defaults,
    # D-07: the ONE sentence both the read seam's log line and the boot-time
    # owner chat notice say about retired keys, next to the tables it reads.
    "retired_setting_keys_notice": settings_defaults,
    "SETTINGS_DEFAULTS": settings_defaults,
    "SETTINGS_KEYS_NOT_EXPORTED_TO_ENV": settings_defaults,
    "SUPERVISOR_LIVENESS_DEADLINE_DEFAULT_SEC": settings_defaults,
    "_DISK_AUTHORED_SETTINGS": settings_defaults,
    "settings_env_keys": settings_defaults,
    "EFFORT_SCALE": settings_scales,
    "PROMPT_CACHE_TTL_SCALE": settings_scales,
    "VALID_RUNTIME_MODES": settings_scales,
    "VALID_SAFETY_MODES": settings_scales,
    "_RUNTIME_MODE_RANK": settings_scales,
    "_SAFETY_MODE_RANK": settings_scales,
    "clamp_effort_to": settings_scales,
    "effort_one_step_down": settings_scales,
    "effort_rank": settings_scales,
    "normalize_runtime_mode": settings_scales,
    "normalize_safety_mode": settings_scales,
    "resolve_effort": settings_scales,
    "resolve_prompt_cache_ttl": settings_scales,
    "_LEGACY_SLOT_RENAMES": model_slots,
    # ABI-4 (F3.2): the typed resolved-model destination lives with the slot
    # vocabulary leaf (D02-owner seam) and is re-exported by the facade.
    "ResolvedModelTarget": model_slots,
    "_main_model": model_slots,
    "_parse_model_list": model_slots,
    "get_consciousness_model": model_slots,
    "get_deep_self_review_model": model_slots,
    "get_fallback_models": model_slots,
    "get_heavy_model": model_slots,
    "get_image_input_mode": model_slots,
    "get_light_model": model_slots,
    "get_vision_model": model_slots,
    "migrate_legacy_slot_keys": model_slots,
    "parse_fallback_chain": model_slots,
    "_DIRECT_PROVIDER_REVIEW_RUNS": review_model_routes,
    "_exclusive_direct_remote_provider_env": review_model_routes,
    "adaptive_quorum": review_model_routes,
    "direct_provider_review_models_fallback": review_model_routes,
    "get_review_enforcement": review_model_routes,
    "get_review_models": review_model_routes,
    "get_scope_review_models": review_model_routes,
    # ABI-4 (F3.2): typed views over the effective reviewer model lists.
    "get_review_targets": review_model_routes,
    "get_scope_review_targets": review_model_routes,
    "resolved_review_model_target": review_model_routes,
    "DELEGATE_WAIT_CEILING_SEC": runtime_limits,
    "DELEGATE_WAIT_WINDOW_MAX_SEC": runtime_limits,
    "MAX_ACTIVE_SUBAGENTS_HARD_CAP": runtime_limits,
    # Upstream reshaped the hard-cap assignment into a tuple that also binds the
    # nesting ceiling (`MAX_ACTIVE_..., MAX_SUBAGENT_DEPTH_... = 500, 10`), so the
    # unrowed twin rides the rowed statement into the same owner leaf.
    "MAX_SUBAGENT_DEPTH_HARD_CAP": runtime_limits,
    "_bounded_positive_int_setting": runtime_limits,
    "_clamped_number_setting": runtime_limits,
    "get_acceptance_reserve_pct": runtime_limits,
    "get_acceptance_review_est_sec": runtime_limits,
    "get_delegate_wait_max_sec": runtime_limits,
    "get_delegate_wait_sec": runtime_limits,
    "get_llm_transport_read_timeout_sec": runtime_limits,
    "get_claudexor_quota_refresh_timeout_sec": runtime_limits,
    "get_claudexor_harness_install_timeout_sec": runtime_limits,
    "get_onboarding_snapshot_timeout_sec": runtime_limits,
    "get_settings_document_lock_timeout_sec": runtime_limits,
    "get_direct_turn_stop_wait_sec": runtime_limits,
    "get_max_active_subagents_per_root": runtime_limits,
    "get_max_subagent_depth": runtime_limits,
    "get_max_workers": runtime_limits,
    "get_pacing_interval_sec": runtime_limits,
    "get_per_call_timeout_ceiling_sec": runtime_limits,
    "get_plan_task_deadline_min_sec": runtime_limits,
    "get_post_task_evolution_budget_usd": runtime_limits,
    "get_restart_drain_max_sec": runtime_limits,
    "get_safety_call_timeout_sec": runtime_limits,
    "get_safety_max_tokens": runtime_limits,
    "get_search_code_wall_sec": runtime_limits,
    "get_supervisor_liveness_deadline_sec": runtime_limits,
    "get_task_abs_ceiling_sec": runtime_limits,
    "get_task_idle_timeout_sec": runtime_limits,
    "get_vision_caption_timeout_sec": runtime_limits,
    "get_update_letter_timeout_sec": runtime_limits,
    "get_websearch_timeout_sec": runtime_limits,
}

# The settings-file lifecycle, the path roots and the owner-only ratchets stay with the
# parent: every one of them reads or writes ``config.SETTINGS_PATH``/``config.DATA_DIR``,
# or the in-process boot runtime-mode pin, which a leaf could only see through a
# back-edge into its own parent.
_PARENT_RETAINED = (
    "SETTINGS_PATH", "DATA_DIR", "APP_ROOT", "REPO_DIR", "PID_FILE", "PORT_FILE", "HOME",
    "_BOOT_RUNTIME_MODE", "_guard_live_settings_write", "_settings_file_value",
    "_settings_flag_enabled", "_settings_lock_path", "_acquire_settings_lock",
    "_release_settings_lock", "_coerce_setting_value", "load_settings",
    "load_settings_lock_held", "save_settings", "prepare_settings_for_persist",
    "apply_settings_to_env", "get_runtime_mode", "get_safety_mode", "get_context_mode",
    "get_owner_context_mode", "initialize_runtime_mode_baseline",
    "_guard_context_mode_lowering", "_guard_safety_mode_lowering",
)


def _top_level_names(path: pathlib.Path) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            # Upstream binds the two subagent hard caps in one tuple statement.
            for target in node.targets:
                if isinstance(target, ast.Tuple):
                    names.update(e.id for e in target.elts if isinstance(e, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_settings_leaves_never_import_their_parent():
    for module in _LEAVES:
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.ImportFrom) and node.module == "ouroboros.config"
            for node in ast.walk(tree)
        ), module.__name__
        assert not any(
            isinstance(node, ast.Import)
            and any(alias.name == "ouroboros.config" for alias in node.names)
            for node in ast.walk(tree)
        ), module.__name__


def test_provider_models_reads_the_shared_leaves_instead_of_importing_config():
    """The former config <-> provider_models tangle: ``provider_models`` needed the
    fallback chain and the shipped defaults, and could only reach them by importing
    its own importer at call time. Both now live in leaves both sides import."""
    tree = ast.parse(pathlib.Path(provider_models.__file__).read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "ouroboros.config"
        for node in ast.walk(tree)
    )
    top_level = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert {"ouroboros.model_slots", "ouroboros.settings_defaults"} <= top_level
    assert provider_models.parse_fallback_chain is model_slots.parse_fallback_chain
    assert provider_models.SETTINGS_DEFAULTS is settings_defaults.SETTINGS_DEFAULTS
    assert provider_models.OPENROUTER_DEFAULTS is settings_defaults.OPENROUTER_DEFAULTS
    assert (provider_models.OPENROUTER_REVIEW_DEFAULTS
            is settings_defaults.OPENROUTER_REVIEW_DEFAULTS)


def test_config_facade_reexports_every_moved_identity():
    """``config`` keeps the exact objects, so every existing importer and every
    ``monkeypatch.setattr(config, ...)`` consumer sees no identity change."""
    for name, owner in _MOVED_OWNERS.items():
        assert hasattr(config, name), name
        assert getattr(config, name) is getattr(owner, name), name
    owned = {name for module in _LEAVES for name in vars(module)}
    assert set(_MOVED_OWNERS) <= owned


def test_settings_file_lifecycle_and_path_roots_stay_with_the_parent():
    parent_names = _top_level_names(pathlib.Path(config.__file__))
    assert set(_PARENT_RETAINED) <= parent_names
    for module in _LEAVES:
        leaf_names = _top_level_names(pathlib.Path(module.__file__))
        assert not (leaf_names & set(_PARENT_RETAINED)), module.__name__


def test_settings_extraction_owner_inventory_is_exact():
    """Every moved name is owned by exactly one leaf, and no leaf grew a name the
    parent never had (a new symbol would be a redesign, not an extraction)."""
    seen: dict[str, str] = {}
    for module in _LEAVES:
        for name in _top_level_names(pathlib.Path(module.__file__)):
            assert name not in seen, f"{name} owned by {seen.get(name)} and {module.__name__}"
            seen[name] = module.__name__
            assert name in _MOVED_OWNERS, f"{module.__name__} owns an unmapped name: {name}"
    assert set(seen) == set(_MOVED_OWNERS)


def test_settings_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        module.__name__: len(
            pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines()
        )
        for module in (config, *_LEAVES)
    }
    assert counts["ouroboros.config"] <= 1000
    assert all(count <= 1000 for count in counts.values())
    assert 250 <= counts["ouroboros.settings_defaults"] <= 500
    assert (PACKAGE / "config.py").is_file()
