"""The declarative install-time subscription preset compiler (D-3/D-9).

PR-3: the wizard's review seats are MARKED catalog rows (the review pool),
not an inline lane — `_pool` reads the preset's roster the way the runtime
does (``review_pool_rows``), so these tests see exactly what review runs.
"""

from __future__ import annotations

import json
from itertools import combinations

import pytest

from ouroboros.reviewer_slot_config import review_pool_rows
from ouroboros.subscription_install_presets import (
    PRESET_MARKER_KEY,
    SUBSCRIPTION_PRESET_VERSION,
    HarnessDiscovery,
    compile_install_preset,
    factory_review_rows,
)


def _pool(preset):
    """The preset's review pool as the runtime reads it (catalog order)."""
    return review_pool_rows({"OUROBOROS_SUBAGENTS": preset.available_subagents})


def _pool_rows(preset):
    return [(row.target_id, row.effort) for row in _pool(preset)]

# Verbatim from the live Claudexor daemon (GET /v2/harnesses/<id>/models,
# 2026-08-09). Trimmed only of ids no seat can name; the Claude generation the
# refreshed policy names (claude-opus-5-5) is added beside the old one. This historical
# Codex fixture lacks gpt-6.1-sol; the current/preferred branch is exercised separately.
LIVE_MODELS = {
    "claude": (
        "sonnet", "opus", "haiku", "fable", "best", "claude-opus-5-5",
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
        "subagent": ("claude-opus-5-5", "medium"),
        "triad": ("claude-opus-5-5", "medium"),
        "scope": ("claude-opus-5-5", "medium"),
    },
    "codex": {
        "subagent": ("gpt-5.6-sol", "medium"),
        "triad": ("gpt-5.6-sol", "medium"),
        "scope": ("gpt-5.6-sol", "medium"),
    },
    "cursor": {
        "subagent": ("cursor-grok-4.6-high", "high"),
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
    """One triad seat per connected reviewer harness, a sole one included (owner Q5=A)."""
    return [harness for harness in CORE_HARNESSES if harness in connected]


def _expected_pool(connected):
    """Every triad seat is one pool row; the scope seat MERGES into a row of
    its exact identity (every pool row judges both parts of the brief) and
    otherwise takes a row of its own."""
    rows = [_target(harness, "triad") for harness in _triad_harnesses(connected)]
    scope = _target(next(harness for harness in SCOPE_ORDER if harness in connected), "scope")
    return rows if scope in rows else [*rows, scope]


@pytest.mark.parametrize("connected", COMBINATIONS)
def test_every_combination_follows_the_declarative_policy(connected):
    preset = compile_install_preset(_discoveries(*connected))

    assert preset.ok, preset.refusal
    primary = next(harness for harness in HARNESSES if harness in connected)
    subagent_target, subagent_effort = _target(primary, "subagent")
    first_actor = json.loads(preset.available_subagents)["items"][0]
    assert first_actor["route"]["target_id"] == subagent_target
    # An actor row is Auto (the owner's range decides) unless its model id spells the level.
    assert first_actor.get("effort", "") == (subagent_effort if primary == "cursor" else "")

    assert sorted(_pool_rows(preset)) == sorted(_expected_pool(connected))
    assert all(row.is_session for row in _pool(preset))
    # No advisory seat is resolved: the preflight reviewer is a row the author
    # names, so a model no row runs can neither refuse the preset nor be receipted.
    assert "advisory" not in preset.receipt["surfaces"]
    assert preset.receipt["review_pool"] == [row.slot_id for row in _pool(preset)]


@pytest.mark.parametrize("connected", COMBINATIONS)
def test_only_exact_discovery_ids_are_ever_written(connected):
    """No owner shorthand (``opus-5.5``, ``grok-4.6``) may survive
    into the saved value — every model must exist in that harness's live list."""
    preset = compile_install_preset(_discoveries(*connected))

    targets = [row.target_id for row in _pool(preset)]
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

    assert all(
        not row["route"].get("credential_profile_id")
        for row in json.loads(preset.available_subagents)["items"]
    )
    assert all(row.profile_id == "" for row in _pool(preset))


@pytest.mark.parametrize("harness", CORE_HARNESSES)
def test_a_single_core_harness_reviews_once_never_as_copies_of_one_model(harness):
    """Owner Q5=A: three identical seats paid three runs of one model for no second
    opinion, so a sole harness takes ONE triad seat. Its scope seat merges into that row
    when the identity matches (Claude, Codex) and keeps its own model otherwise (Cursor's
    scope seat is the high-effort slug), so no two pool rows run one model."""
    preset = compile_install_preset(_discoveries(harness))
    pool = _pool(preset)

    expected_target, _ = _target(harness, "triad")
    assert [row.target_id for row in pool if row.target_id == expected_target] == [expected_target]
    assert [seat["harness"] for seat in preset.receipt["surfaces"]["triad"]] == [harness]
    assert len({row.target_id for row in pool}) == len(pool)


@pytest.mark.parametrize("connected", [c for c in COMBINATIONS if len(c) > 1])
def test_two_or_three_harnesses_keep_one_triad_seat_each(connected):
    preset = compile_install_preset(_discoveries(*connected))

    assert [seat["harness"] for seat in preset.receipt["surfaces"]["triad"]] == [
        harness for harness in HARNESSES if harness in connected]
    assert [seat["position"] for seat in preset.receipt["surfaces"]["triad"]] == list(range(1, len(connected) + 1))


def test_settings_keys_write_new_actor_ssot_and_receipt_not_legacy_singleton():
    from ouroboros.configured_subagents import SUBAGENTS_RECEIPT_KEY, SUBAGENTS_SETTING

    preset = compile_install_preset(_discoveries("claude"))

    assert set(preset.settings_keys()) == {SUBAGENTS_SETTING, SUBAGENTS_RECEIPT_KEY, PRESET_MARKER_KEY}
    assert "OUROBOROS_REVIEWER_SLOTS" not in preset.settings_keys()  # the pool rides the roster
    assert preset.settings_keys()[PRESET_MARKER_KEY] == SUBSCRIPTION_PRESET_VERSION
    assert "OUROBOROS_SUBAGENT_HARNESS" not in preset.settings_keys()
    # The API model slots are NOT among them (owner decision D-2).
    assert "OUROBOROS_MODEL" not in preset.settings_keys()


def test_unresolvable_model_refuses_typed_and_emits_nothing():
    models = dict(LIVE_MODELS)
    models["claude"] = tuple(m for m in LIVE_MODELS["claude"] if m != "claude-opus-5-5")

    preset = compile_install_preset(_discoveries("claude", models=models))

    assert not preset.ok
    assert preset.refusal is not None
    assert preset.refusal.code == "model_not_in_discovery"
    assert preset.refusal.seat is not None
    assert (preset.refusal.seat.surface, preset.refusal.seat.position) == ("subagent", 1)
    assert preset.refusal.seat.preference == "opus-5.5"
    # The previous generation is still discovered and still never substituted.
    assert preset.refusal.candidates == ("claude-opus-5-5",) and "claude-opus-5" in models["claude"]
    # Nothing partial: no subagent value, no settings keys at all.
    assert preset.available_subagents == ""
    assert preset.settings_keys() == {}


def test_codex_takes_6_1_sol_when_discovered_else_5_6_sol_with_the_reason_said():
    """Codex prefers gpt-6.1-sol; only live discovery without it takes gpt-5.6-sol, and the receipt
    and the preview's diagnostics say which and why. Neither discovered is a typed refusal."""
    def codex(*extra):
        return compile_install_preset([HarnessDiscovery("codex", (*LIVE_MODELS["codex"], *extra))])

    newest = codex("gpt-6.1-sol")
    assert {row["model"] for rows in newest.receipt["surfaces"].values() for row in rows} == {"gpt-6.1-sol"}
    assert not any(note["code"] == "model_preference_successor" for note in newest.diagnostics)
    older = codex()
    [note] = [note for note in older.diagnostics if note["code"] == "model_preference_successor"]
    assert (note["preference"], note["chosen_preference"], note["model"]) == ("gpt-6.1-sol", "gpt-5.6-sol", "gpt-5.6-sol")
    assert note in older.receipt["diagnostics"]
    assert note["message"] == "codex discovery has no 'gpt-6.1-sol'; this seat uses the next preference, 'gpt-5.6-sol'."
    assert all(row["reason"] == note["message"] for rows in older.receipt["surfaces"].values() for row in rows)
    refused = compile_install_preset([HarnessDiscovery("codex", ("gpt-5.6", "gpt-5.5"))])
    assert refused.refusal.code == "model_not_in_discovery"
    assert refused.refusal.candidates == ("gpt-6.1-sol", "gpt-5.6-sol")


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

    # scope is the high-effort cursor seat: the slug tail and the field agree,
    # so nothing downstream can materialize a DIFFERENT effort by default. It
    # marks the task actor row (same session route and effort) instead of
    # minting a twin.
    scope_rows = [row for row in _pool(preset) if row.target_id == "cursor=cursor-grok-4.6-high"]
    assert [(row.slot_id, row.effort) for row in scope_rows] == [("primary-builder", "high")]
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
        assert _pool_rows(preset) == []
    else:
        assert _pool_rows(preset) == _pool_rows(compile_install_preset(_discoveries(*core)))


def test_claude_and_codex_rows_carry_effort_only_in_the_field():
    preset = compile_install_preset(_discoveries("claude", "codex"))

    for row in _pool(preset):
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
    assert "advisory" not in receipt["surfaces"]
    assert len(receipt["surfaces"]["triad"]) == 2
    # Reviewer seats keep their explicit levels on rows of their own; the Auto actors stay actors.
    assert receipt["review_pool"] == ["review-claude", "review-codex"]
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
    # The pinned owner row is not the wizard's seat (unpinned, medium): one
    # unpinned row is minted beside it and the owner's row stays unmarked.
    assert [row.slot_id for row in _pool(preset)] == ["review-claude"]
    assert all(row.profile_id == "" for row in _pool(preset))


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
    # One list: the actors are Auto rows (no effort), so the one triad seat mints its
    # own pinned reviewer row and the scope seat merges into it.
    assert [row["subagent_id"] for row in items] == [
        "primary-builder", "fast-scout", "independent-perspective", "review-claude",
    ]
    assert all("name" not in row and "effort" not in row for row in items[:3])  # retired field; Auto actors
    assert [row.slot_id for row in _pool(preset)] == ["review-claude"]
    assert [row.effort for row in _pool(preset)] == ["medium"]
    assert [row["route"]["target_id"] for row in items] == [
        "claude=claude-opus-5-5", "openai/gpt-5.6-luna", "openai/gpt-5.6-sol", "claude=claude-opus-5-5",
    ]
    assert [row.get("minted_from") for row in items] == [None, None, None, "factory_default"]


def test_roster_cap_overflow_omits_the_seat_with_a_diagnostic():
    """The 26-row cap leaves no room to mint: there is no inline lane to fall
    back to any more, so the seat is OMITTED and says so in diagnostics —
    honest disclosure, never a silent drop (and never a 27th row)."""
    from ouroboros.configured_subagents import (
        MAX_CONFIGURED_SUBAGENTS,
        ROUTE_KIND_API_MODEL,
        ConfiguredSubagent,
        RouteSpec,
        make_configured_subagents,
    )
    from ouroboros.subscription_install_presets import _mark_review_rows

    full = make_configured_subagents([
        ConfiguredSubagent(
            subagent_id=f"row-{i}",
            recommended_use="Use for owner-selected work.",
            route=RouteSpec(ROUTE_KIND_API_MODEL, f"openai/model-{i}"),
        )
        for i in range(MAX_CONFIGURED_SUBAGENTS)
    ])
    extended, diagnostics = _mark_review_rows(
        full,
        [{"position": 1, "target_id": "claude=claude-opus-5", "effort": "medium", "surface": "triad"}],
        [{"position": 1, "target_id": "claude=claude-sonnet-5", "effort": "low", "surface": "scope"}],
    )

    assert len(extended.items) == MAX_CONFIGURED_SUBAGENTS  # nothing minted past the cap
    assert not any(row.review_eligible for row in extended.items)
    assert [(row["code"], row["target_id"]) for row in diagnostics] == [
        ("reviewer_seat_omitted_roster_full", "claude=claude-opus-5"),
        ("reviewer_seat_omitted_roster_full", "claude=claude-sonnet-5"),
    ]
    # Through the compiler the result is a typed refusal: an owner draft that
    # leaves the pool empty is not applied (save judge, allow_empty=False).
    preset = compile_install_preset(_discoveries("claude"), configured_subagents=full, source="configured")
    assert not preset.ok and preset.refusal.code == "preset_failed_pool_validation"


def test_marking_the_review_rows_is_idempotent_so_a_shown_catalog_resaved_is_unchanged():
    """The wizard's output fed back as its input (the catalog a preview showed,
    saved as shown) is a fixed point: every marked row already stands for one
    seat, so nothing is re-marked and no further twin is minted. Pinned on
    identical seats, the strongest case (the compiler itself names each harness
    once): each still takes exactly one row."""
    from ouroboros.configured_subagents import (
        ROUTE_KIND_AGENT_SESSION,
        ConfiguredSubagent,
        RouteSpec,
        make_configured_subagents,
    )
    from ouroboros.subscription_install_presets import _mark_review_rows

    seat = {"target_id": "claude=claude-opus-5", "effort": "medium"}
    triad = [{"position": n, "surface": "triad", **seat} for n in (1, 2, 3)]
    scope = [{"position": 1, "surface": "scope", **seat}]
    start = make_configured_subagents([
        ConfiguredSubagent(subagent_id="builder", recommended_use="Builds.",
                           route=RouteSpec(ROUTE_KIND_AGENT_SESSION, seat["target_id"]), effort="medium"),
    ])

    once, diagnostics = _mark_review_rows(start, triad, scope)
    assert diagnostics == []
    assert [(row.subagent_id, row.review_eligible, row.minted_from) for row in once.items] == [
        ("builder", True, ""), ("review-claude", True, "factory_default"), ("review-claude-2", True, "factory_default")]

    twice, diagnostics = _mark_review_rows(once, triad, scope)
    assert diagnostics == [] and twice == once
    # Still one seat per marked row: a fourth identical seat mints exactly one more twin.
    more, _ = _mark_review_rows(once, [*triad, {"position": 4, "surface": "triad", **seat}], scope)
    assert [row.subagent_id for row in more.items] == ["builder", "review-claude", "review-claude-2", "review-claude-3"]


# --- factory review rows (contract §3.2 п.6): the shipped panel as catalog rows --

def _factory(doc):
    return [(row["route"]["target_id"], row["effort"]) for row in factory_review_rows(doc)]


def test_factory_review_rows_mint_the_shipped_panel_for_each_install_class():
    from ouroboros.settings_defaults import OPENROUTER_REVIEW_DEFAULTS

    # An OpenRouter key: the three shipped reviewers, three different models.
    rows = factory_review_rows({"OPENROUTER_API_KEY": "configured"})
    assert [row["route"]["target_id"] for row in rows] == list(OPENROUTER_REVIEW_DEFAULTS["triad"])
    assert len({row["route"]["target_id"] for row in rows}) == 3
    assert [row["subagent_id"] for row in rows] == ["review-1", "review-2", "review-3"]
    assert all(row["review_eligible"] is True and row["minted_from"] == "factory_default" for row in rows)
    assert all("delivery" not in row for row in rows)  # the catalog default (native) applies
    # One exclusive direct provider: its role panel around Main — Main once (owner Q5=A), while
    # MiniMax keeps its mixed panel of two models (Main, Light, Light).
    assert _factory({"OPENAI_API_KEY": "configured", "OUROBOROS_MODEL": "openai::gpt-5.6-sol"}) == [
        ("openai::gpt-5.6-sol", "high")]
    minimax = _factory({"MINIMAX_API_KEY": "configured", "OUROBOROS_MODEL": "minimax::MiniMax-M3",
                        "OUROBOROS_MODEL_LIGHT": "minimax::MiniMax-M3-light"})
    assert [effort for _target, effort in minimax] == ["high"] * 3 and len({t for t, _e in minimax}) == 2
    # A Main the provider cannot run is named, never replaced by the provider's default (owner 2A/4A).
    assert _factory({"DEEPSEEK_API_KEY": "configured", "OUROBOROS_MODEL": "openai/gpt-5.5"}) == []
    assert all(t.startswith("deepseek::") for t, _e in _factory({"DEEPSEEK_API_KEY": "configured"}))
    # No router and no provider panel (owner Q5=A): ONE row on the reachable Main in its full
    # route identity, never copies of it (each copy pays a run and adds no second opinion; the
    # one answer decides, quorum 1 of 1) — a compatible route, a local Main (labelled local, so
    # delegation runs it locally too), a managed Main with its account pin.
    from ouroboros.review_model_routes import adaptive_quorum

    compatible = _factory({"OPENAI_COMPATIBLE_API_KEY": "configured", "OPENAI_COMPATIBLE_BASE_URL": "http://x",
                           "OUROBOROS_MODEL": "openai-compatible::glm-5.3"})
    assert compatible == [("openai-compatible::glm-5.3", "high")] and adaptive_quorum(len(compatible)) == 1
    local = _factory({"USE_LOCAL_MAIN": True, "LOCAL_MODEL_SOURCE": "owner/model.gguf",
                      "OUROBOROS_MODEL": "owner/local-main"})
    assert local == [("owner/local-main (local)", "high")] and adaptive_quorum(len(local)) == 1
    assert [row["subagent_id"] for row in factory_review_rows({"USE_LOCAL_MAIN": True, "LOCAL_MODEL_SOURCE": "o/m.gguf",
                                                               "OUROBOROS_MODEL": "owner/local-main"})] == ["review-1"]
    managed = factory_review_rows({"OUROBOROS_MODEL": "claudexor::codex=gpt-6-astra",
                                   "OUROBOROS_MODEL_ACCOUNTS": json.dumps({"main": "koshak"})})
    assert [row["route"] for row in managed] == [
        {"kind": "api_model", "target_id": "claudexor::codex=gpt-6-astra", "credential_profile_id": "koshak"}]
    # Without a local source the flag routes nowhere: nothing to run. With no configured
    # access at all the shipped (router) panel stands, as the shipped defaults do.
    assert factory_review_rows({"USE_LOCAL_MAIN": True}) == []
    assert _factory({"TOTAL_BUDGET": 5.0}) == [(model, "high") for model in OPENROUTER_REVIEW_DEFAULTS["triad"]]


@pytest.mark.parametrize("provider", ["openai", "anthropic", "cloudru", "gigachat", "deepseek", "zai", "minimax"])
def test_a_direct_provider_panel_seats_main_once_unless_it_mixes_two_models(provider):
    """Owner Q5=A for direct providers: a panel that would repeat Main is Main once; MiniMax's
    mixed panel (Main, Light, Light: two models) keeps its three seats. Saved catalogs are not
    re-derived (``test_defaults_provenance``: a saved three-row pool reads back unchanged)."""
    from ouroboros.provider_models import DIRECT_PROVIDER_DEFAULTS, compute_direct_review_seats

    defaults = DIRECT_PROVIDER_DEFAULTS[provider]
    seats = compute_direct_review_seats(provider, defaults["main"], defaults.get("light", ""))
    if provider == "minimax":
        assert [role for role, _model in seats] == ["main", "light", "light"]
        assert len({model for _role, model in seats}) == 2
    else:
        assert [role for role, _model in seats] == ["main"]


def test_factory_review_rows_take_the_document_effort_and_free_ids():
    doc = {"OPENROUTER_API_KEY": "configured", "OUROBOROS_EFFORT_REVIEW": "medium",
           "OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": [
               {"subagent_id": "review-1", "recommended_use": "x",
                "route": {"kind": "api_model", "target_id": "openai/gpt-5.5"}},
           ]})}
    rows = factory_review_rows(doc)
    assert [row["effort"] for row in rows] == ["medium"] * 3
    assert [row["subagent_id"] for row in rows] == ["review-2", "review-3", "review-4"]
    assert [row["effort"] for row in factory_review_rows(
        {"OPENROUTER_API_KEY": "configured", "OUROBOROS_EFFORT_REVIEW": "bogus"})] == ["high"] * 3


def test_missing_exact_agy_flash_refuses_without_partial_actor_or_reviewer_output():
    preset = compile_install_preset([
        HarnessDiscovery("agy", ("gemini-3.7-flash-medium", "gemini-3.1-pro-high")),
    ], settings={"OPENROUTER_API_KEY": "configured",
                 "OUROBOROS_MODEL": "openai/gpt-5.6-luna"})

    assert not preset.ok
    assert preset.refusal is not None
    assert preset.refusal.code == "model_not_in_discovery"
    assert preset.available_subagents == ""


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
