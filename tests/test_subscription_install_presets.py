"""The declarative install-time subscription preset compiler (D-3/D-9)."""

from __future__ import annotations

import json
from itertools import combinations

import pytest

from ouroboros.reviewer_slot_config import parse_reviewer_slots


def _parse_preset_slots(preset):
    """Parse emitted slots WITH the roster the same preset ships: reviewer
    rows are subagent_id references now (4=A), and references resolve against
    the Available-subagents value — exactly as the applied install would."""
    import os

    prior = os.environ.get("OUROBOROS_SUBAGENTS")
    os.environ["OUROBOROS_SUBAGENTS"] = preset.available_subagents
    try:
        return parse_reviewer_slots(preset.reviewer_slots)
    finally:
        if prior is None:
            os.environ.pop("OUROBOROS_SUBAGENTS", None)
        else:
            os.environ["OUROBOROS_SUBAGENTS"] = prior
from ouroboros.subscription_install_presets import (
    PRESET_MARKER_KEY,
    REVIEWER_SLOTS_KEY,
    SUBSCRIPTION_PRESET_VERSION,
    HarnessDiscovery,
    compile_install_preset,
)

# Verbatim from the live Claudexor daemon (GET /v2/harnesses/<id>/models,
# 2026-08-09). Trimmed only of ids no seat can name.
LIVE_MODELS = {
    "claude": (
        "sonnet", "opus", "haiku", "fable", "best",
        "claude-fable-5", "claude-sonnet-5", "claude-opus-5", "claude-opus-4-8",
        "claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5",
        "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-haiku-4-5",
    ),
    "codex": (
        "gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5",
        "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex-spark",
    ),
    "cursor": (
        "auto", "composer-2.5",
        "cursor-grok-4.6-low", "cursor-grok-4.6-medium", "cursor-grok-4.6-high",
        "cursor-grok-4.6-high-fast",
        "gpt-5.6-sol-low", "gpt-5.6-sol-medium", "gpt-5.6-sol-high",
        "gpt-5.6-sol-xhigh", "gpt-5.6-sol-max",
        "gpt-5.6-terra-medium", "gpt-5.6-terra-high",
        "gpt-5.6-luna-medium",
        "claude-opus-5-medium", "claude-opus-5-high",
        "claude-fable-5-thinking-xhigh", "claude-sonnet-5-medium",
    ),
    "agy": (
        "gemini-3.8-flash-low",
        "gemini-3.8-flash-medium",
        "gemini-3.8-flash-high",
    ),
}

HARNESSES = ("claude", "codex", "cursor")
CORE_HARNESSES = HARNESSES
SCOPE_ORDER = ("codex", "claude", "cursor")
COMBINATIONS = tuple(
    combination
    for size in range(1, len(HARNESSES) + 1)
    for combination in combinations(HARNESSES, size)
)
AGY_COMBINATIONS = (("agy",),) + tuple(
    (*combination, "agy")
    for combination in COMBINATIONS
)
EXPECTED_SURFACES = {
    "claude": {
        "subagent": ("claude-opus-5", "medium"),
        "advisory": ("claude-sonnet-5", "low"),
        "triad": ("claude-opus-5", "medium"),
        "scope": ("claude-opus-5", "medium"),
    },
    "codex": {
        "subagent": ("gpt-5.6-sol", "medium"),
        "advisory": ("gpt-5.6-terra", "medium"),
        "triad": ("gpt-5.6-sol", "medium"),
        "scope": ("gpt-5.6-sol", "medium"),
    },
    "cursor": {
        "subagent": ("cursor-grok-4.6-high", "high"),
        "advisory": ("cursor-grok-4.6-medium", "medium"),
        "triad": ("cursor-grok-4.6-medium", "medium"),
        "scope": ("cursor-grok-4.6-high", "high"),
    },
}


def _discoveries(*harnesses, models=None):
    catalog = models or LIVE_MODELS
    return [HarnessDiscovery(harness_id=h, model_ids=tuple(catalog[h])) for h in harnesses]


def _target(harness, surface):
    model, effort = EXPECTED_SURFACES[harness][surface]
    return f"{harness}={model}", effort


def _triad_harnesses(connected):
    core = [harness for harness in CORE_HARNESSES if harness in connected]
    return core * 3 if len(core) == 1 else core


@pytest.mark.parametrize("connected", COMBINATIONS)
def test_every_combination_follows_the_declarative_policy(connected):
    preset = compile_install_preset(_discoveries(*connected))

    assert preset.ok, preset.refusal
    primary = next(harness for harness in HARNESSES if harness in connected)
    subagent_target, subagent_effort = _target(primary, "subagent")
    first_actor = json.loads(preset.available_subagents)["items"][0]
    assert first_actor["route"]["target_id"] == subagent_target
    assert first_actor["effort"] == subagent_effort

    config = _parse_preset_slots(preset)
    assert [(row.target_id, row.effort) for row in config.triad] == [
        _target(harness, "triad") for harness in _triad_harnesses(connected)
    ]
    scope_harness = next(harness for harness in SCOPE_ORDER if harness in connected)
    assert [(row.target_id, row.effort) for row in config.scope] == [
        _target(scope_harness, "scope")
    ]
    assert (config.advisory.target_id, config.advisory.effort) == _target(
        primary, "advisory",
    )
    assert config.advisory.enabled is True
    assert config.advisory.kind == "agent_session"
    assert all(row.is_session for row in config.triad + config.scope)


@pytest.mark.parametrize("connected", COMBINATIONS)
def test_only_exact_discovery_ids_are_ever_written(connected):
    """No owner shorthand (``opus-5``, ``terra``, ``grok-4.6``) may survive
    into the saved value — every model must exist in that harness's live list."""
    preset = compile_install_preset(_discoveries(*connected))
    config = _parse_preset_slots(preset)

    targets = [row.target_id for row in config.triad + config.scope]
    targets.append(config.advisory.target_id)
    targets.extend(
        row["route"]["target_id"]
        for row in json.loads(preset.available_subagents)["items"]
        if row["route"]["kind"] == "agent_session"
    )
    for target in targets:
        harness, _, model = target.partition("=")
        assert model in LIVE_MODELS[harness], f"{model!r} is not in {harness} discovery"


@pytest.mark.parametrize("connected", COMBINATIONS)
def test_credential_profile_is_never_pinned(connected):
    """D28: the daemon rotates accounts; an install-time pin would outlive one."""
    preset = compile_install_preset(_discoveries(*connected))

    assert "profile_id" not in preset.reviewer_slots
    config = _parse_preset_slots(preset)
    assert all(row.profile_id == "" for row in config.triad + config.scope)
    assert config.advisory.profile_id == ""


@pytest.mark.parametrize("harness", CORE_HARNESSES)
def test_single_core_harness_runs_three_independent_same_model_slots(harness):
    config = _parse_preset_slots(compile_install_preset(_discoveries(harness)))

    expected_target, _ = _target(harness, "triad")
    assert [row.target_id for row in config.triad] == [expected_target] * 3
    assert len({row.slot_id for row in config.triad}) == 3


def test_settings_keys_write_new_actor_ssot_and_receipt_not_legacy_singleton():
    from ouroboros.configured_subagents import SUBAGENTS_RECEIPT_KEY, SUBAGENTS_SETTING

    preset = compile_install_preset(_discoveries("claude"))

    assert set(preset.settings_keys()) == {
        REVIEWER_SLOTS_KEY, SUBAGENTS_SETTING, SUBAGENTS_RECEIPT_KEY, PRESET_MARKER_KEY}
    assert preset.settings_keys()[PRESET_MARKER_KEY] == SUBSCRIPTION_PRESET_VERSION
    assert "OUROBOROS_SUBAGENT_HARNESS" not in preset.settings_keys()
    # The API model slots are NOT among them (owner decision D-2).
    assert "OUROBOROS_MODEL" not in preset.settings_keys()


def test_unresolvable_model_refuses_typed_and_emits_nothing():
    models = dict(LIVE_MODELS)
    models["claude"] = tuple(m for m in LIVE_MODELS["claude"] if m != "claude-opus-5")

    preset = compile_install_preset(_discoveries("claude", models=models))

    assert not preset.ok
    assert preset.refusal is not None
    assert preset.refusal.code == "model_not_in_discovery"
    assert preset.refusal.seat is not None
    assert (preset.refusal.seat.surface, preset.refusal.seat.position) == ("subagent", 1)
    assert preset.refusal.seat.preference == "opus-5"
    assert "claude-opus-5" in preset.refusal.candidates
    # Nothing partial: no slots, no subagent value, no settings keys at all.
    assert preset.reviewer_slots == ""
    assert preset.settings_keys() == {}


def test_cursor_without_grok_refuses_instead_of_using_a_costlier_model():
    models = dict(LIVE_MODELS)
    models["cursor"] = tuple(
        model for model in LIVE_MODELS["cursor"]
        if not model.startswith(("cursor-grok-4.6", "grok-4.6"))
    )

    refused = compile_install_preset(_discoveries("cursor", models=models))

    assert not refused.ok
    assert refused.refusal.code == "model_not_in_discovery"
    assert refused.refusal.seat.preference == "grok-4.6"
    assert any(model.startswith("gpt-5.6") for model in models["cursor"])


def test_cursor_effort_rides_the_slug_and_the_row_field_together():
    preset = compile_install_preset(_discoveries("cursor"))
    config = _parse_preset_slots(preset)

    # scope is the high-effort cursor seat: the slug tail and the field agree,
    # so nothing downstream can materialize a DIFFERENT effort by default.
    assert config.scope[0].target_id == "cursor=cursor-grok-4.6-high"
    assert config.scope[0].effort == "high"
    assert preset.receipt["surfaces"]["scope"][0]["effort_in_model_id"] is True
    assert preset.receipt["surfaces"]["subagent"][0]["effort_in_model_id"] is True


@pytest.mark.parametrize("connected", AGY_COMBINATIONS)
def test_antigravity_compiles_task_actor_without_changing_core_reviewer_bytes(connected):
    preset = compile_install_preset(_discoveries(*connected))

    assert preset.ok, preset.refusal
    actor_routes = [
        row["route"]["target_id"]
        for row in json.loads(preset.available_subagents)["items"]
    ]
    assert "agy=gemini-3.8-flash-high" in actor_routes
    core = tuple(harness for harness in connected if harness in CORE_HARNESSES)
    if not core:
        assert preset.reviewer_slots == ""
    else:
        assert preset.reviewer_slots == compile_install_preset(_discoveries(*core)).reviewer_slots


def test_claude_and_codex_rows_carry_effort_only_in_the_field():
    preset = compile_install_preset(_discoveries("claude", "codex"))
    config = _parse_preset_slots(preset)

    for row in config.triad + config.scope:
        model = row.target_id.partition("=")[2]
        assert not model.endswith((":medium", "-medium")), model
        assert row.effort == "medium"
    assert preset.receipt["surfaces"]["triad"][0]["effort_in_model_id"] is False


def test_no_connected_preset_harness_refuses():
    preset = compile_install_preset([HarnessDiscovery("opencode", ("whatever",))])

    assert not preset.ok
    assert preset.refusal.code == "no_available_subagents"
    assert preset.settings_keys() == {}


def test_empty_discovery_refuses_instead_of_guessing():
    preset = compile_install_preset([HarnessDiscovery("claude", ())])

    assert not preset.ok
    assert preset.refusal.code == "discovery_empty"
    assert "claude" in preset.refusal.message


def test_receipt_records_what_was_resolved_and_from_where():
    preset = compile_install_preset(
        _discoveries("claude", "codex"),
        capability={"claude": {"status": "ok"}},
    )

    receipt = preset.receipt
    assert receipt["version"] == SUBSCRIPTION_PRESET_VERSION
    assert receipt["connected"] == ["claude", "codex"]
    assert receipt["profile_pinned"] is False
    assert receipt["discovery_counts"]["codex"] == len(LIVE_MODELS["codex"])
    assert receipt["capability"]["claude"]["status"] == "ok"
    assert receipt["surfaces"]["advisory"]["model"] == "claude-sonnet-5"
    assert len(receipt["surfaces"]["triad"]) == 2
    # The receipt must be JSON-serializable — it rides an API response.
    json.dumps(receipt)


def test_owner_pinned_session_is_reported_truthfully_in_the_receipt():
    from ouroboros.configured_subagents import parse_configured_subagents

    owner = parse_configured_subagents({
        "enabled": True,
        "items": [{
            "subagent_id": "owner-session",
            "name": "Owner session",
            "recommended_use": "Use the explicitly pinned owner account.",
            "route": {
                "kind": "agent_session",
                "target_id": "claude=claude-opus-5",
                "credential_profile_id": "owner-account",
            },
            "effort": "high",
        }],
    })
    preset = compile_install_preset(
        _discoveries("claude"),
        configured_subagents=owner,
        source="configured",
    )

    assert preset.ok, preset.refusal
    assert preset.receipt["profile_pinned"] is True
    saved_route = preset.receipt["available_subagents"]["items"][0]["route"]
    assert saved_route["credential_profile_id"] == "owner-account"


def test_compiler_reads_no_settings_and_carries_no_transport(monkeypatch):
    """The compiler is pure: everything it knows arrives as an argument.

    Structural, not behavioural: a transport import in this module would be the
    second discovery path the endpoint is built to avoid, and a settings read
    would make the compiler's answer depend on state its caller already owns."""
    import pathlib

    import ouroboros.config as config

    monkeypatch.setattr(
        config, "load_settings",
        lambda: (_ for _ in ()).throw(AssertionError("the compiler must not read settings")))
    assert compile_install_preset(_discoveries("codex")).ok

    source = (pathlib.Path(__file__).resolve().parents[1]
              / "ouroboros" / "subscription_install_presets.py").read_text(encoding="utf-8")
    for forbidden in ("import httpx", "import requests", "import socket",
                      "import urllib", "ClaudexorGateway", "load_settings"):
        assert forbidden not in source, f"{forbidden} has no place in a pure compiler"


# Verbatim from the Antigravity CLI the Claudexor 3.5.0 agy adapter pins
# (AGY_KNOWN_MODELS, verified against agy 1.1.13, plus the gemini-3.8-flash
# triple the shipped preset now targets — assumed to be published by the vendor
# CLI on the owner's decision, not read from an installed agy on this host).
# Seventeen ids; effort rides inside the slug, and gemini-3.1-pro exists ONLY
# at high/low.
AGY_LIVE_MODELS = (
    "gemini-3.8-flash-high", "gemini-3.8-flash-medium", "gemini-3.8-flash-low",
    "gemini-3.7-flash-high", "gemini-3.7-flash-medium", "gemini-3.7-flash-low",
    "gemini-3.6-flash-high", "gemini-3.6-flash-medium", "gemini-3.6-flash-low",
    "gemini-3.5-flash-high", "gemini-3.5-flash-medium", "gemini-3.5-flash-low",
    "gemini-3.1-pro-high", "gemini-3.1-pro-low",
    "claude-sonnet-4-6", "claude-opus-4-6-thinking", "gpt-oss-120b-medium",
)


def _catalog_with_agy():
    return {**LIVE_MODELS, "agy": AGY_LIVE_MODELS}


def test_agy_missing_required_flash_is_typed_discovery_failure():
    preset = compile_install_preset([HarnessDiscovery(harness_id="agy", model_ids=())])
    assert preset.refusal is not None
    assert preset.refusal.code == "discovery_empty"


def test_no_recognized_combination_can_raise():
    # The KeyError class: every non-empty subset of PRESET_HARNESSES must come
    # back as a compiled preset or a typed refusal, never an exception.
    import itertools

    from ouroboros.subscription_install_presets import PRESET_HARNESSES

    catalog = _catalog_with_agy()
    for size in range(1, len(PRESET_HARNESSES) + 1):
        for combo in itertools.combinations(PRESET_HARNESSES, size):
            preset = compile_install_preset(_discoveries(*combo, models=catalog))
            assert preset.ok or preset.refusal is not None


def test_agy_alias_table_spells_effort_inside_the_id():
    from ouroboros.subscription_install_presets import (
        _EFFORT_IN_MODEL_ID,
        _MODEL_ALIASES,
        HARNESS_AGY,
    )

    assert HARNESS_AGY in _EFFORT_IN_MODEL_ID
    aliases = _MODEL_ALIASES[HARNESS_AGY]
    # Every alias candidate formats to an id the pinned vendor CLI really
    # publishes, so automatic and manually selected rows resolve exact ids.
    assert aliases["gemini-3.8-flash"][0].format(effort="high") in AGY_LIVE_MODELS
    assert aliases["gemini-3.1-pro"][0].format(effort="high") in AGY_LIVE_MODELS
    assert aliases["gemini-3.1-pro"][0].format(effort="low") in AGY_LIVE_MODELS
    # Documented trap for the future dictation: pro has no -medium slug.
    assert aliases["gemini-3.1-pro"][0].format(effort="medium") not in AGY_LIVE_MODELS


def test_api_only_compiles_main_and_distinct_light_without_daemon_inputs():
    preset = compile_install_preset((), settings={
        "OPENAI_API_KEY": "configured",
        "OUROBOROS_MODEL": "openai::gpt-5.6-sol",
        "OUROBOROS_MODEL_LIGHT": "openai::gpt-5.6-luna",
    })

    assert preset.ok, preset.refusal
    items = json.loads(preset.available_subagents)["items"]
    # `name` is retired (1=A): identity is the neutral id + derived facts.
    assert [(row["subagent_id"], row["route"]["target_id"]) for row in items] == [
        ("primary-builder", "openai::gpt-5.6-sol"),
        ("fast-scout", "openai::gpt-5.6-luna"),
    ]
    assert all("name" not in row for row in items)
    assert preset.reviewer_slots == ""


def test_api_only_identical_main_and_light_deduplicate_without_fake_diversity():
    preset = compile_install_preset((), settings={
        "OPENROUTER_API_KEY": "configured",
        "OUROBOROS_MODEL": "openai/gpt-5.6-luna",
        "OUROBOROS_MODEL_LIGHT": "openai/gpt-5.6-luna",
    })

    assert len(json.loads(preset.available_subagents)["items"]) == 1
    assert [row["code"] for row in preset.diagnostics] == [
        "duplicate_api_routes_omitted",
    ]


def test_legacy_heavy_is_not_an_active_default_actor_source():
    preset = compile_install_preset((), settings={
        "OPENROUTER_API_KEY": "configured",
        "OUROBOROS_MODEL_HEAVY": "anthropic/claude-opus-5",
    })

    assert not preset.ok
    assert preset.refusal is not None
    assert preset.refusal.code == "no_available_subagents"


def test_local_only_materializes_existing_local_suffix_and_no_router_actor():
    preset = compile_install_preset((), settings={
        "LOCAL_MODEL_SOURCE": "owner/model.gguf",
        "USE_LOCAL_MAIN": True,
        "USE_LOCAL_LIGHT": True,
        "OUROBOROS_MODEL": "owner-main",
        "OUROBOROS_MODEL_LIGHT": "owner-light",
    })

    assert preset.ok, preset.refusal
    targets = [
        row["route"]["target_id"]
        for row in json.loads(preset.available_subagents)["items"]
    ]
    assert targets == ["owner-main (local)", "owner-light (local)"]


def test_one_harness_plus_distinct_main_and_light_normally_yields_three_real_actors():
    preset = compile_install_preset(_discoveries("claude"), settings={
        "OPENROUTER_API_KEY": "configured",
        "OUROBOROS_MODEL": "openai/gpt-5.6-sol",
        "OUROBOROS_MODEL_LIGHT": "openai/gpt-5.6-luna",
    })

    items = json.loads(preset.available_subagents)["items"]
    # 4=A: the reviewer seat whose session route matches no task actor is
    # MINTED into the roster and referenced — one SSOT from the first boot.
    assert [row["subagent_id"] for row in items] == [
        "primary-builder", "fast-scout", "independent-perspective", "review-claude",
    ]
    assert all("name" not in row for row in items)  # retired field (1=A)
    slots = json.loads(preset.reviewer_slots)
    referenced = {row.get("subagent_id") for row in slots["triad"] + slots["scope"]}
    referenced.add(slots["advisory"].get("subagent_id"))
    assert referenced <= {row["subagent_id"] for row in items} | {None}
    assert "review-claude" in referenced
    assert [row["route"]["target_id"] for row in items] == [
        "claude=claude-opus-5", "openai/gpt-5.6-luna", "openai/gpt-5.6-sol",
        "claude=claude-sonnet-5",  # the minted review seat's own session route
    ]


def test_roster_cap_overflow_falls_back_to_inline_seats_with_a_diagnostic():
    """The 10-row cap leaves no room to mint: the seat stays an INLINE route
    and says so in diagnostics — honest fallback, never a silent drop."""
    from ouroboros.configured_subagents import (
        ROUTE_KIND_API_MODEL,
        ConfiguredSubagent,
        RouteSpec,
        make_configured_subagents,
    )
    from ouroboros.subscription_install_presets import _reference_reviewer_rows

    full = make_configured_subagents([
        ConfiguredSubagent(
            subagent_id=f"row-{i}",
            recommended_use="Use for owner-selected work.",
            route=RouteSpec(ROUTE_KIND_API_MODEL, f"openai/model-{i}"),
        )
        for i in range(10)
    ])
    extended, slots_json, diagnostics = _reference_reviewer_rows(
        full,
        [{"position": 1, "target_id": "claude=claude-opus-5", "effort": "medium"}],
        [],
        {"target_id": "claude=claude-sonnet-5", "effort": "low"},
    )

    assert len(extended.items) == 10  # nothing minted past the cap
    slots = json.loads(slots_json)
    triad_row = slots["triad"][0]
    assert "subagent_id" not in triad_row
    assert triad_row["route"] == {"kind": "agent_session", "target_id": "claude=claude-opus-5"}
    assert triad_row["effort"] == "medium"
    advisory_row = slots["advisory"]
    assert advisory_row["enabled"] is True
    assert advisory_row["route"] == {"kind": "agent_session", "target_id": "claude=claude-sonnet-5"}
    codes = [row["code"] for row in diagnostics]
    assert codes.count("reviewer_seat_inline_roster_full") == 2


def test_missing_exact_agy_flash_refuses_without_partial_actor_or_reviewer_output():
    preset = compile_install_preset([
        HarnessDiscovery("agy", ("gemini-3.7-flash-medium", "gemini-3.1-pro-high")),
    ], settings={"OPENROUTER_API_KEY": "configured",
                 "OUROBOROS_MODEL": "openai/gpt-5.6-luna"})

    assert not preset.ok
    assert preset.refusal is not None
    assert preset.refusal.code == "model_not_in_discovery"
    assert preset.available_subagents == ""
    assert preset.reviewer_slots == ""


def test_valid_owner_draft_is_validated_not_recompiled_from_missing_agy_default():
    from ouroboros.configured_subagents import parse_configured_subagents

    owner = parse_configured_subagents({
        "enabled": True,
        "items": [{
            "subagent_id": "owner",
            "name": "Owner",
            "recommended_use": "Use when the owner selects it.",
            "route": {"kind": "api_model", "target_id": "openai::gpt-5.6-sol"},
        }],
    })
    preset = compile_install_preset(
        [HarnessDiscovery("agy", ())],
        configured_subagents=owner,
        source="configured",
    )

    assert preset.ok, preset.refusal
    assert json.loads(preset.available_subagents)["items"][0]["subagent_id"] == "owner"
    assert preset.source == "configured"
    assert preset.receipt["source"] == "configured"
    assert preset.reviewer_slots == ""
