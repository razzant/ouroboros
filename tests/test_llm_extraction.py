"""Structural contracts for the semantic-no-op llm.py extraction."""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import sys

from ouroboros import (
    llm,
    llm_anthropic,
    llm_attempt,
    llm_capability_policy,
    llm_fallback,
    llm_gigachat,
    llm_local,
    llm_messages,
    llm_openai_compatible,
    llm_pricing,
    llm_probe,
    llm_routing,
)
from ouroboros.llm import LLMClient

REPO = pathlib.Path(__file__).parents[1]
PKG = REPO / "ouroboros"

_LEAVES = (
    llm_attempt,
    llm_capability_policy,
    llm_routing,
    llm_messages,
    llm_fallback,
    llm_anthropic,
    llm_gigachat,
    llm_local,
    llm_openai_compatible,
    llm_pricing,
    # Not a mixin and not an extraction: the probe transport arrived whole from
    # upstream. It is an llm_* leaf all the same, so the leaf rules bind it —
    # never import the parent, no cycles, real weight.
    llm_probe,
)

# Module-level names that moved. llm.py re-exports every one of them, so its
# import surface (and every existing importer) is unchanged.
_MODULE_OWNERS = {
    llm_attempt: (
        "_CACHE_TTL_SECONDS _VALID_CACHE_TTLS _applied_payload_cache_ttl _attempt_request "
        "_candidate_before_dispatch _canonical_candidate_bytes _execute_candidate "
        "_execute_candidate_async _is_structured_context_overflow_body "
        "_is_structured_context_overflow_exception _physical_candidate "
        "_route_normalizes_cache_breakpoints _structured_error_values cache_ttl_seconds "
        "supports_message_cache_control"
    ),
    llm_capability_policy: (
        "_MANDATORY_VALUE_MARKERS _OPTIONAL_DROPPABLE_PARAMS _OPTIONAL_SAMPLING_PARAMS "
        "normalize_reasoning_effort"
    ),
    llm_routing: "_OR_PROVIDER_PRESETS _resolve_or_provider",
    llm_messages: "_reasoning_signature_portable_across_or_providers",
    llm_local: (
        "LocalContextTooLargeError _LOCAL_COMPACTION_MODES _compact_local_text "
        "_compact_markdown_sections _estimate_message_chars _split_markdown_sections"
    ),
    llm_openai_compatible: "_FALSE_LIKE_ENV_VALUES",
    llm_pricing: "add_usage fetch_cloudru_pricing fetch_openrouter_pricing",
}

# LLMClient members that moved into an owner mixin. LLMClient inherits the exact
# same function objects, so name, signature and body are unchanged.
_MIXIN_OWNERS = {
    (llm_attempt, "_PayloadCachePolicyMixin"): (
        "_MAX_CACHE_BREAKPOINTS _normalize_payload_cache_ttl _payload_cache_breakpoints "
        "_pop_cache_breakpoint_disclosure"
    ),
    (llm_capability_policy, "_CapabilityPolicyMixin"): (
        "_CAPABILITIES_FETCH_OK _CONTEXT_LENGTH_CACHE _EFFORT_CEILING_CACHE _EFFORT_CEILING_LOADED "
        "_EFFORT_FLOOR_CACHE _EFFORT_FLOOR_LOADED _EFFORT_FLOOR_RELOAD_SEC _NESTED_REASONING_PARAM "
        "_REJECTED_PARAMS_CACHE _REJECTED_PARAMS_LOADED _REJECTED_PARAMS_RELOAD_SEC "
        "_SUPPORTED_PARAMS_CACHE _SUPPORTED_PARAMS_FETCHED _apply_rejected_param_cache "
        "_clamp_effort_for_model _effort_ceiling_for _effort_floor_for "
        "_fetch_openrouter_capabilities _get_supported_parameters _known_rejected_params "
        "_mandatory_value_rejection _parameter_rejection_error _payload_effort "
        "_pop_effort_clamp_disclosure _record_effort_ceiling _record_effort_floor "
        "_remember_rejected_params _retry_without_optional_sampling _set_payload_effort "
        "clamp_effort_for_route metadata_fetch_attempted_and_failed openrouter_context_length"
    ),
    (llm_routing, "_ProviderRoutingMixin"): (
        "_explicit_cache_affinity_identity _get_async_remote_client _get_client _get_local_client "
        "_get_remote_client _make_no_proxy_async_client _make_no_proxy_client _new_remote_client "
        "_no_proxy_timeout _openrouter_session_identity _parse_provider_model "
        "_prompt_cache_identity _qualified_model_name _resolve_remote_target "
        "probe_oversized_context probe_provider_readiness"
    ),
    (llm_messages, "_MessageShapingMixin"): (
        "_REASONING_CONTENT_BLOCK_TYPES _content_with_system_notice_marker "
        "_copy_messages_with_cache_policy _has_openrouter_reasoning_details "
        "_has_replayed_reasoning_metadata _is_deferrable_image_user_turn _model_family "
        "_normalize_system_message_placement _replace_image_blocks_with_placeholder "
        "_strip_openrouter_roundtrip_metadata sanitize_reasoning_on_model_switch"
    ),
    (llm_fallback, "_RecoveryLadderMixin"): (
        "_create_chat_completion_with_retries _create_chat_completion_with_retries_async "
        "_is_http_status _is_transient_body_error _openrouter_signature_retry_kwargs "
        "_param_retry_kwargs_for_body_error _provider_body_error _reroute_kwargs_for_body_error "
        "_reroute_same_model_kwargs _retry_without_prompt_cache_parameter "
        "_rotate_openrouter_session_affinity _strip_kwargs_for_encrypted_body_error"
    ),
    (llm_anthropic, "_AnthropicLaneMixin"): (
        "_anthropic_blocks_from_content _anthropic_image_block _build_anthropic_messages "
        "_build_anthropic_tool_choice _cache_write_split _chat_anthropic "
        "_coalesce_anthropic_message _normalize_anthropic_response "
        "_sanitize_anthropic_tool_result_content _stringify_anthropic_content"
    ),
    (llm_gigachat, "_GigaChatLaneMixin"): (
        "_chat_gigachat _get_gigachat_client _gigachat_function_result _gigachat_messages "
        "_gigachat_text _new_gigachat_client _normalize_gigachat_response"
    ),
    (llm_local, "_LocalLaneMixin"): "_chat_local _prepare_messages_for_local_context",
    (llm_openai_compatible, "_OpenAICompatibleLaneMixin"): (
        "_build_remote_kwargs _normalize_remote_response _openrouter_main_web_search_tool "
        "extract_display_reasoning"
    ),
    (llm_pricing, "_GenerationCostMixin"): "_fetch_generation_cost",
}

# Members llm.py keeps: the composition itself, the caller-facing chat surface,
# and the tool-schema/tool-call translators every lane reaches by class name.
_PARENT_MEMBERS = frozenset({
    "__init__", "chat", "chat_async", "_chat_remote", "vision_query", "default_model",
    "available_models", "_strip_reasoning_wrappers", "_parse_tool_calls_from_content",
    "_stringify_tool_description", "_sanitize_chat_completion_tools", "_build_anthropic_tools",
    "_gigachat_sanitize_schema", "_gigachat_functions",
})


def test_llm_leaves_never_import_their_parent():
    for module in _LEAVES:
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "ouroboros.llm", module.__name__
            if isinstance(node, ast.Import):
                assert not any(a.name == "ouroboros.llm" for a in node.names), module.__name__


def test_llm_facade_reexports_every_moved_module_identity():
    """``ouroboros.llm`` keeps the exact objects, so existing importers and
    monkeypatch targets of the module surface see no identity change."""
    for owner, names in _MODULE_OWNERS.items():
        for name in names.split():
            assert hasattr(llm, name), name
            assert getattr(llm, name) is getattr(owner, name), name
    # The shared context-budget seam stays reachable through llm.py as before.
    from ouroboros import context_budget

    assert llm.context_overflow_message is context_budget.context_overflow_message
    assert llm.CONTEXT_OVERFLOW_CODES is context_budget.CONTEXT_OVERFLOW_CODES


def _defined_members(path: pathlib.Path, class_name: str) -> set[str]:
    """Members a class DEFINES in source — immune to monkeypatch residue that an
    earlier test in the same process may have left on the class object."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    out: set[str] = set()
    for sub in node.body:
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(sub.name)
        elif isinstance(sub, ast.Assign):
            out.update(t.id for t in sub.targets if isinstance(t, ast.Name))
        elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
            out.add(sub.target.id)
    return out


def test_llm_client_members_resolve_to_their_mixin_owners():
    parent_defined = _defined_members(pathlib.Path(llm.__file__), "LLMClient")
    for (owner, mixin_name), names in _MIXIN_OWNERS.items():
        mixin = getattr(owner, mixin_name)
        defined = _defined_members(pathlib.Path(owner.__file__), mixin_name)
        for name in names.split():
            assert name in defined, f"{mixin_name}.{name}"
            assert name not in parent_defined, f"{name} is defined twice"
            assert name in mixin.__dict__, name
            assert hasattr(LLMClient, name), name


def test_llm_client_member_inventory_is_unchanged():
    """The composed class exposes exactly the member set of the tree it was split from.

    The digest moved once, deliberately: the final upstream cutoff (PR #257) added three
    methods to ``LLMClient`` in the base — ``_new_remote_client``, ``probe_provider_readiness``
    and ``_new_gigachat_client`` — and the adopting merge re-homed them into the leaves that
    own their siblings. Three genuinely new base members is the only sanctioned reason this
    pin may move; anything else is a member appearing or vanishing without provenance.
    """
    assert _defined_members(pathlib.Path(llm.__file__), "LLMClient") == _PARENT_MEMBERS
    moved = {name for names in _MIXIN_OWNERS.values() for name in names.split()}
    composed = sorted(moved | _PARENT_MEMBERS)
    assert hashlib.sha256(
        json.dumps(composed, separators=(",", ":")).encode()
    ).hexdigest() == "84006da5cb3e40f04b19b559630e4767ef016f01af99c5269b6a81aeea35199f"
    for name in composed:
        assert hasattr(LLMClient, name), name


def test_llm_mixin_composition_order_is_pinned():
    assert [base.__name__ for base in LLMClient.__mro__] == [
        "LLMClient",
        "_PayloadCachePolicyMixin",
        "_CapabilityPolicyMixin",
        "_ProviderRoutingMixin",
        "_MessageShapingMixin",
        "_RecoveryLadderMixin",
        "_AnthropicLaneMixin",
        "_GigaChatLaneMixin",
        "_LocalLaneMixin",
        "_OpenAICompatibleLaneMixin",
        "_GenerationCostMixin",
        "object",
    ]
    # No mixin shadows another: every member has exactly one owner.
    owners: dict[str, str] = {}
    for base in LLMClient.__mro__[1:-1]:
        for name in _defined_members(pathlib.Path(sys.modules[base.__module__].__file__), base.__name__):
            assert name not in owners, f"{name} owned by both {owners.get(name)} and {base.__name__}"
            owners[name] = base.__name__


def test_llm_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        module.__name__: len(
            pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines()
        )
        for module in (llm, *_LEAVES)
    }
    assert counts["ouroboros.llm"] <= 750
    assert all(count <= 1000 for count in counts.values()), counts
    # Every leaf carries real weight; a 40-line leaf would be a seam, not an owner.
    assert all(count >= 200 for count in counts.values()), counts


def test_llm_leaf_import_graph_is_acyclic_and_shallow():
    """Leaves may depend on siblings, never in a cycle."""
    edges: dict[str, set[str]] = {}
    for module in _LEAVES:
        name = module.__name__.rsplit(".", 1)[-1]
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        edges[name] = {
            node.module.rsplit(".", 1)[-1]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and str(node.module or "").startswith("ouroboros.llm_")
        }
    seen: set[str] = set()

    def walk(node: str, stack: tuple[str, ...]) -> None:
        assert node not in stack, f"import cycle: {stack + (node,)}"
        for child in sorted(edges.get(node, ())):
            walk(child, stack + (node,))
        seen.add(node)

    for name in sorted(edges):
        walk(name, ())
    assert seen == set(edges)
