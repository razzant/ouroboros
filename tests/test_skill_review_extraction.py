"""Structural contracts for the semantic-no-op skill_review extraction."""

from __future__ import annotations

import ast
import pathlib

from ouroboros import (
    skill_review,
    skill_review_output,
    skill_review_packs,
    skill_review_prompt,
    skill_review_rebuttals,
)
from ouroboros.tool_module_inventory import (
    build_frozen_tool_manifest,
    discover_tool_module_inventory,
    load_frozen_tool_modules,
)


REPO = pathlib.Path(__file__).parents[1]
TOOLS = REPO / "ouroboros" / "tools"

_LEAVES = (
    skill_review_packs,
    skill_review_rebuttals,
    skill_review_prompt,
    skill_review_output,
)

_MOVED_OWNERS = {
    "_LOADABLE_BINARY_EXTENSIONS": skill_review_packs,
    "_SKILL_PACK_TOKEN_HEADROOM": skill_review_packs,
    "_SkillBinaryPayload": skill_review_packs,
    "_SkillFileOverBudget": skill_review_packs,
    "_SkillFileUnreadable": skill_review_packs,
    "_build_skill_file_packs": skill_review_packs,
    "_read_skill_text": skill_review_packs,
    "_skill_pack_token_budget": skill_review_packs,
    "_accepted_rebuttals_path": skill_review_rebuttals,
    "_build_skill_review_history_section": skill_review_rebuttals,
    "_convergence_hint": skill_review_rebuttals,
    "_fail_items_from_history_entry": skill_review_rebuttals,
    "_load_accepted_rebuttals": skill_review_rebuttals,
    "_persist_rebuttal_flips": skill_review_rebuttals,
    "_record_accepted_rebuttal": skill_review_rebuttals,
    "_render_accepted_rebuttals_section": skill_review_rebuttals,
    "_review_history_path": skill_review_rebuttals,
    "_CRITICAL_ITEMS": skill_review_prompt,
    "_REPO_ROOT": skill_review_prompt,
    "_SKILL_CHECKLIST_SECTION": skill_review_prompt,
    "_SKILL_REVIEW_ITEMS": skill_review_prompt,
    "_build_review_prompt": skill_review_prompt,
    "_build_review_prompt_for_attempt": skill_review_prompt,
    "_emit_skill_advisory_warning": skill_review_prompt,
    "_load_governance_artifact": skill_review_prompt,
    "_review_wave_budget_block": skill_review_prompt,
    "_run_skill_advisory_pre_review": skill_review_prompt,
    "_aggregate_status": skill_review_output,
    "_extract_actor_findings": skill_review_output,
    "_parse_json_array": skill_review_output,
    "render_skill_review_block": skill_review_output,
}


def test_skill_review_leaves_are_non_catalog_owners_without_backedges(tmp_path):
    for module in (skill_review, *_LEAVES):
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "get_tools"
            for node in tree.body
        )
    for module in _LEAVES:
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.ImportFrom)
            and node.module == "ouroboros.skill_review"
            for node in ast.walk(tree)
        )
        assert not any(
            isinstance(node, ast.Import)
            and any(alias.name == "ouroboros.skill_review" for alias in node.names)
            for node in ast.walk(tree)
        )

    source_inventory = discover_tool_module_inventory(TOOLS)
    for module in (skill_review, *_LEAVES):
        assert module.__name__.rsplit(".", 1)[-1] not in source_inventory.tool_modules
    manifest = tmp_path / "_frozen_tool_modules.v1.json"
    build_frozen_tool_manifest(TOOLS, manifest)
    assert load_frozen_tool_modules(manifest) == source_inventory.tool_modules


def test_skill_review_keeps_the_lifecycle_driver_and_its_patchable_seams():
    """The trust gate's own decisions stay with ``skill_review``: the outcome
    record, the deterministic preflight floor, the official-hub payload profile
    the owner-attestation path consults through this module, the quorum-failure
    outcome, and ``review_skill`` itself. The preflight and the hub predicate are
    patched at ``skill_review.<name>`` by the attestation tests, and their only
    readers live here, so those seams do not move."""
    for name in (
        "SkillReviewOutcome",
        "_truncate_raw_result",
        "_apply_auto_grant_outcome",
        "_is_module_widget_skill",
        "_run_deterministic_preflight",
        "_official_hub_review_profile",
        "is_official_hub_payload_verified",
        "_skill_quorum_failure_outcome",
        "review_skill",
    ):
        assert getattr(skill_review, name).__module__ == "ouroboros.skill_review", name
    assert skill_review.__all__ == [
        "SkillReviewOutcome",
        "render_skill_review_block",
        "review_skill",
    ]


def test_skill_review_facade_reexports_every_moved_identity():
    """``skill_review`` keeps the exact objects, so skill_exec, the marketplace
    fetcher and installer, the gateway extension endpoints, the lifecycle runner
    and the skill tests see no identity change at their historical import site."""
    for name, owner in _MOVED_OWNERS.items():
        assert hasattr(skill_review, name), name
        assert getattr(skill_review, name) is getattr(owner, name), name
    owned = {name for module in _LEAVES for name in vars(module)}
    assert set(_MOVED_OWNERS) <= owned


def test_skill_review_prompt_and_parser_share_one_item_contract():
    """The parser validates against the SAME closed item list the prompt demanded,
    so an item can never be asked for and then silently not required."""
    assert skill_review_output._SKILL_REVIEW_ITEMS is skill_review_prompt._SKILL_REVIEW_ITEMS
    assert skill_review._SKILL_REVIEW_ITEMS is skill_review_prompt._SKILL_REVIEW_ITEMS
    assert "bug_hunting" in skill_review_prompt._SKILL_REVIEW_ITEMS


def test_skill_review_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        module.__name__: len(
            pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines()
        )
        for module in (skill_review, *_LEAVES)
    }
    assert counts["ouroboros.skill_review"] <= 800
    assert all(count <= 1000 for count in counts.values())
    assert 300 <= counts["ouroboros.skill_review_prompt"] <= 1000
