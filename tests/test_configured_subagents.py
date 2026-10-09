"""Focused contract tests for the Available-subagents settings foundation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ouroboros.config import migrate_legacy_slot_keys
from ouroboros.configured_subagents import (
    MAX_CONFIGURED_SUBAGENTS,
    LEGACY_SUBAGENT_COMPATIBILITY,
    SOURCE_CONFIGURED,
    SOURCE_INVALID,
    SOURCE_LEGACY_MIGRATED,
    SOURCE_ONBOARDING_DEFAULT,
    SOURCE_UNDECIDED,
    SUBAGENTS_RECEIPT_KEY,
    SUBAGENTS_SETTING,
    configured_subagents_fingerprint,
    normalize_configured_subagents,
    parse_configured_subagents,
    serialize_configured_subagents,
    resolve_configured_subagents,
    resolve_settings_subagent_candidate,
)
from ouroboros.server_runtime import apply_runtime_provider_defaults


def _row(row_id: str = "builder", **overrides):
    row = {
        "subagent_id": row_id,
        "name": "Builder",
        "recommended_use": "Use for substantial implementation.",
        "route": {
            "kind": "agent_session",
            "target_id": "codex=gpt-5.6-sol",
            "credential_profile_id": "",
        },
        "effort": "medium",
    }
    row.update(overrides)
    return row


def _config(*rows, enabled=True):
    return {"enabled": enabled, "items": list(rows or (_row(),))}


@pytest.fixture
def _clean_subagent_env(monkeypatch):
    for key in (
        SUBAGENTS_SETTING,
        "OUROBOROS_SUBAGENT_HARNESS",
        "OUROBOROS_SUBAGENT_PROFILE",
    ):
        monkeypatch.delenv(key, raising=False)


def test_strict_config_round_trips_object_and_json_to_one_canonical_string():
    config_from_object, canonical = normalize_configured_subagents(_config())
    config_from_json = parse_configured_subagents(canonical)

    assert config_from_object == config_from_json
    # `name` is retired (1=A): the canonical string is the fixture minus the
    # legacy key — dropping it on serialize IS the migration.
    expected = _config()
    for item in expected["items"]:
        item.pop("name", None)
        if item["route"]["kind"] == "agent_session":
            item["access"] = "full"
    assert json.loads(canonical) == expected
    assert canonical == normalize_configured_subagents(canonical)[1]
    assert configured_subagents_fingerprint(config_from_object) == (configured_subagents_fingerprint(config_from_json))
    assert LEGACY_SUBAGENT_COMPATIBILITY == "remove_after_next_minor_release"


def test_session_access_defaults_full_and_preserves_an_explicit_lower_choice():
    original, original_bytes = normalize_configured_subagents(_config())
    explicit_default, default_bytes = normalize_configured_subagents(_config(_row(access="workspace_write")))
    assert explicit_default != original and default_bytes != original_bytes
    assert explicit_default.items[0].access == "workspace_write"

    configured, serialized = normalize_configured_subagents(_config(
        _row("broad"), _row("ordinary", access="workspace_write"),
    ))
    assert [row.access for row in configured.items] == ["full", "workspace_write"]
    assert json.loads(serialized)["items"][0]["access"] == "full"
    assert json.loads(serialized)["items"][1]["access"] == "workspace_write"
    assert parse_configured_subagents(serialized) == configured
    full_only = parse_configured_subagents(_config(_row(access="full")))
    assert configured_subagents_fingerprint(full_only) == configured_subagents_fingerprint(original)


@pytest.mark.parametrize("access", [None, "", "readonly", "inherit_native", True, {}, "FULL"])
def test_invalid_actor_access_is_not_silently_coerced(access):
    with pytest.raises(ValueError, match="access must be workspace_write or full"):
        parse_configured_subagents(_config(_row(access=access)))


def test_api_actor_cannot_request_session_access():
    row = _row(route={"kind": "api_model", "target_id": "provider/model"})
    config, serialized = normalize_configured_subagents(_config(row))
    assert config.items[0].access == ""
    assert "access" not in json.loads(serialized)["items"][0]
    row["access"] = "full"
    with pytest.raises(ValueError, match="meaningful only for agent_session"):
        parse_configured_subagents(_config(row))


def test_unsaved_candidate_composition_retains_selected_access(_clean_subagent_env):
    candidate = parse_configured_subagents(_config(_row(access="full")))
    resolution = resolve_configured_subagents({}, default_candidate=candidate)
    assert resolution.config == candidate
    assert resolution.config.items[0].access == "full"


def test_access_edit_changes_existing_preset_fingerprint():
    raw = _config()
    receipt = {"source": SOURCE_ONBOARDING_DEFAULT,
               "available_subagents_fingerprint": configured_subagents_fingerprint(parse_configured_subagents(raw)),
               "available_subagents": json.loads(json.dumps(raw))}
    settings = {SUBAGENTS_SETTING: raw, SUBAGENTS_RECEIPT_KEY: receipt}
    assert resolve_configured_subagents(settings).source == SOURCE_ONBOARDING_DEFAULT
    raw["items"][0]["access"] = "workspace_write"
    assert resolve_configured_subagents(settings).source == SOURCE_CONFIGURED


def test_shared_browser_backend_contract_fixture_has_identical_acceptance():
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "web"
        / "tests"
        / "fixtures"
        / "available_subagents_contract.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    for case in fixture["valid"]:
        parsed = parse_configured_subagents(case["value"])
        assert parsed.items, case["name"]

    for case in fixture["invalid"]:
        with pytest.raises(ValueError, match=".+"):
            parse_configured_subagents(case["value"])


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"enabled": "true", "items": []}, "enabled must be a boolean"),
        (_config(_row("bad id")), "subagent_id must match"),
        (_config(_row("same"), _row("same")), "appears twice"),
        (_config(_row(extra=True)), "unknown keys"),
        (_config(_row(route={"kind": "api_model", "target_id": "x", "extra": 1})), "route has unknown keys"),
        (
            _config(_row(route={"kind": "api_model", "target_id": "x", "credential_profile_id": "account"})),
            "meaningful only for agent_session",
        ),
        (_config(_row(route={"kind": "agent_session", "target_id": "codex::model"})), "session target"),
        (
            _config(_row(route={"kind": "agent_session", "target_id": "cursor=cursor-grok-4.6-high"}, effort="medium")),
            "conflicts with compound route effort",
        ),
        (
            _config(_row(route={"kind": "agent_session", "target_id": "cursor=cursor-grok-4.6-high-fast"}, effort="medium")),
            "conflicts with compound route effort",
        ),
        (_config(_row(route={"kind": "agent_session", "target_id": "=gpt-5.6-sol"})), "session harness"),
        (_config(_row(route={"kind": "agent_session", "target_id": "codex="})), "session model is empty"),
        (_config(_row(route={"kind": "agent_session", "target_id": "codex=a=b"})), "at most one"),
        (_config(_row(route={"kind": "agent_session", "target_id": "codex gpt-5.6-sol"})), "without whitespace"),
        (_config(_row(route={"kind": "agent_session", "target_id": "codex=gpt-5.6-sol:high"})), "legacy ':effort'"),
    ],
)
def test_strict_parser_rejects_ambiguous_or_lossy_shapes(payload, match):
    with pytest.raises(ValueError, match=match):
        parse_configured_subagents(payload)


def test_maximum_twenty_six_is_real_and_row_precise():
    """PR-3 (contract §1.1): the roster ceiling is 26 — the pool's seats live
    on the same list as the delegation rows, so the former 10 would not hold a
    full install (factory review rows + harness rows + the owner's own)."""
    assert MAX_CONFIGURED_SUBAGENTS == 26
    parse_configured_subagents(_config(*(_row(f"row-{i}") for i in range(26))))
    with pytest.raises(ValueError, match=f"maximum is {MAX_CONFIGURED_SUBAGENTS}"):
        parse_configured_subagents(_config(*(_row(f"row-{i}") for i in range(27))))


# --- PR-3 package A: review-pool fields on the catalog row -----------------------

def _api_row(row_id: str = "critic", **overrides):
    return _row(row_id, route={"kind": "api_model", "target_id": "openai/gpt-5.5"}, **overrides)


def test_review_eligible_defaults_false_and_is_written_only_when_true():
    config = parse_configured_subagents(_config(_row(), _api_row(review_eligible=True)))
    assert [row.review_eligible for row in config.items] == [False, True]
    serialized = json.loads(serialize_configured_subagents(config))
    assert "review_eligible" not in serialized["items"][0]
    assert serialized["items"][1]["review_eligible"] is True
    with pytest.raises(ValueError, match="review_eligible must be a boolean"):
        parse_configured_subagents(_config(_api_row(review_eligible="yes")))


def test_delivery_is_an_api_row_field_defaulting_to_native_and_packet_round_trips():
    native = parse_configured_subagents(_config(_api_row(review_eligible=True))).items[0]
    assert native.delivery == "native"
    packet_config = parse_configured_subagents(_config(_api_row(review_eligible=True, delivery="packet")))
    assert packet_config.items[0].delivery == "packet"
    serialized = json.loads(serialize_configured_subagents(packet_config))
    assert serialized["items"][0]["delivery"] == "packet"
    # The native default is not written back (defaults stay implicit).
    assert "delivery" not in json.loads(serialize_configured_subagents(
        parse_configured_subagents(_config(_api_row(review_eligible=True)))
    ))["items"][0]
    with pytest.raises(ValueError, match="delivery must be native or packet"):
        parse_configured_subagents(_config(_api_row(delivery="courier")))


def test_delivery_on_a_session_row_is_refused_a_session_always_retrieves():
    with pytest.raises(ValueError, match="delivery is meaningful only for api_model"):
        parse_configured_subagents(_config(_row(delivery="packet", review_eligible=True)))
    with pytest.raises(ValueError, match="delivery is meaningful only for api_model"):
        parse_configured_subagents(_config(_row(delivery="native")))


def test_minted_from_round_trips_and_rejects_unknown_provenance():
    for value in ("review_lane", "factory_default"):
        config = parse_configured_subagents(_config(_api_row(review_eligible=True, minted_from=value)))
        assert config.items[0].minted_from == value
        assert json.loads(serialize_configured_subagents(config))["items"][0]["minted_from"] == value
    assert parse_configured_subagents(_config(_api_row())).items[0].minted_from == ""
    with pytest.raises(ValueError, match="minted_from"):
        parse_configured_subagents(_config(_api_row(minted_from="wizard")))


def test_pre_pool_documents_serialize_byte_identically():
    """Defaults are not written: a document saved before the pool fields
    existed round-trips to the same bytes, so no install is rewritten by the
    mere presence of the new fields."""
    old_document = json.dumps({"enabled": True, "items": [
        {"subagent_id": "builder", "recommended_use": "Use for substantial implementation.",
         "route": {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol", "credential_profile_id": ""},
         "effort": "medium", "access": "full"},
        {"subagent_id": "critic", "recommended_use": "Use for review.",
         "route": {"kind": "api_model", "target_id": "openai/gpt-5.5"}},
    ]}, ensure_ascii=False, separators=(",", ":"))
    assert serialize_configured_subagents(parse_configured_subagents(old_document)) == old_document
    assert "review_eligible" not in old_document and "delivery" not in old_document


def test_unique_engines_allows_marked_twins_and_a_minted_row_beside_the_owners():
    from ouroboros.configured_subagents import validate_unique_engines

    def twins(first: dict, second: dict):
        return parse_configured_subagents(_config(_api_row("one", **first), _api_row("two", **second)))

    # The same engine twice is refused on save...
    with pytest.raises(ValueError, match="runs the same engine"):
        validate_unique_engines(twins({}, {}), {})
    # ...unless BOTH seats are review-eligible (one engine judging twice)...
    validate_unique_engines(twins({"review_eligible": True}, {"review_eligible": True, "delivery": "packet"}), {})
    # ...or at least one of them is a row the runtime minted: beside the owner's own row...
    validate_unique_engines(twins({}, {"review_eligible": True, "minted_from": "factory_default"}), {})
    # ...or beside its own reviewer row (M4: a former advisory/deep seat coinciding with a
    # triad seat gives one marked reviewer and one unmarked helper, both minted; two helpers too).
    validate_unique_engines(twins({"review_eligible": True, "minted_from": "review_lane"}, {"minted_from": "review_lane"}), {})
    validate_unique_engines(twins({"minted_from": "review_lane"}, {"minted_from": "review_lane"}), {})
    # Two rows the owner authored stay refused, marked or not.
    with pytest.raises(ValueError, match="runs the same engine"):
        validate_unique_engines(twins({}, {"review_eligible": True}), {})


def test_legacy_name_is_accepted_and_dropped_never_fabricated():
    """1=A: identity is the neutral id + derived facts; a legacy `name`
    parses and is dropped, nothing invents a title-cased label, and a
    non-string value still refuses typed."""
    row = _row("owner_builder")
    row.pop("name")
    parsed = parse_configured_subagents(_config(row))
    assert parsed.items[0].name == ""

    row["name"] = "Legacy Label"
    parsed = parse_configured_subagents(_config(row))
    assert parsed.items[0].name == ""
    assert '"name"' not in serialize_configured_subagents(parsed)

    row["name"] = 7
    with pytest.raises(ValueError, match="name must be a string"):
        parse_configured_subagents(_config(row))


def test_valid_new_setting_wins_over_every_legacy_selector():
    raw = _config(_row("new"), enabled=False)
    resolution = resolve_configured_subagents(
        {
            SUBAGENTS_SETTING: json.dumps(raw),
            "OUROBOROS_SUBAGENT_HARNESS": "claude=claude-opus-5:high",
            "OUROBOROS_MODEL_HEAVY": "openai/gpt-5.6-sol",
        }
    )

    assert resolution.source == SOURCE_CONFIGURED
    assert resolution.config is not None
    assert resolution.config.enabled is False
    assert [row.subagent_id for row in resolution.config.items] == ["new"]


@pytest.mark.parametrize("source", [SOURCE_ONBOARDING_DEFAULT, SOURCE_LEGACY_MIGRATED])
def test_exact_preset_receipt_preserves_source_until_owner_edits_the_rows(source):
    raw = _config(_row("generated"))
    parsed = parse_configured_subagents(raw)
    receipt = {
        "source": source,
        "available_subagents_fingerprint": configured_subagents_fingerprint(parsed),
    }
    settings = {
        SUBAGENTS_SETTING: json.dumps(raw),
        SUBAGENTS_RECEIPT_KEY: json.dumps(receipt),
    }

    generated = resolve_configured_subagents(settings)
    assert generated.source == source

    edited = json.loads(settings[SUBAGENTS_SETTING])
    # `name` is retired and dropped on parse, so editing it can no longer
    # count as an owner edit; a semantic-field edit still flips the source.
    edited["items"][0]["recommended_use"] = "Owner edited"
    settings[SUBAGENTS_SETTING] = json.dumps(edited)
    configured = resolve_configured_subagents(settings)
    assert configured.source == SOURCE_CONFIGURED


def test_preset_receipt_survives_the_name_retirement_via_its_embedded_rows():
    """A receipt written before `name` retired hashed a serialization that
    carried that key; those bytes are unrecoverable after parse. Provenance
    must survive on an UNTOUCHED install through the receipt's own embedded
    rows — while any owner edit still downgrades, and a receipt with neither
    a matching fingerprint nor embedded rows never relabels anything."""
    raw = _config(_row("generated"))
    receipt = {
        "source": SOURCE_ONBOARDING_DEFAULT,
        "available_subagents_fingerprint": "sha-of-a-serialization-that-carried-name",
        "available_subagents": _config(_row("generated")),
    }
    settings = {
        SUBAGENTS_SETTING: json.dumps(raw),
        SUBAGENTS_RECEIPT_KEY: json.dumps(receipt),
    }
    assert resolve_configured_subagents(settings).source == SOURCE_ONBOARDING_DEFAULT

    edited = _config(_row("generated", recommended_use="Owner edited."))
    settings[SUBAGENTS_SETTING] = json.dumps(edited)
    assert resolve_configured_subagents(settings).source == SOURCE_CONFIGURED

    bare = {
        "source": SOURCE_ONBOARDING_DEFAULT,
        "available_subagents_fingerprint": "stale",
    }
    settings = {
        SUBAGENTS_SETTING: json.dumps(raw),
        SUBAGENTS_RECEIPT_KEY: json.dumps(bare),
    }
    assert resolve_configured_subagents(settings).source == SOURCE_CONFIGURED


@pytest.mark.parametrize("receipt", [
    "not-json",
    {"source": SOURCE_ONBOARDING_DEFAULT, "available_subagents_fingerprint": "wrong"},
    {"source": "invented", "available_subagents_fingerprint": "unused"},
])
def test_untrusted_preset_receipt_never_relabels_owner_configuration(receipt):
    raw = _config(_row("owner"))
    resolution = resolve_configured_subagents({
        SUBAGENTS_SETTING: raw,
        SUBAGENTS_RECEIPT_KEY: (
            receipt if isinstance(receipt, str) else json.dumps(receipt)
        ),
    })
    assert resolution.source == SOURCE_CONFIGURED


def test_legacy_singleton_and_account_pin_migrate_without_persisting(
    _clean_subagent_env,
):
    settings = {
        "OUROBOROS_SUBAGENT_HARNESS": "claude=claude-opus-5:high",
        "OUROBOROS_SUBAGENT_PROFILE": "owner-account",
        "OUROBOROS_MODEL_HEAVY": "openai/gpt-5.6-sol",
        "OUROBOROS_MODEL_LIGHT": "openai/gpt-5.6-luna",
        "OPENROUTER_API_KEY": "configured",
    }
    resolution = resolve_configured_subagents(settings)

    assert resolution.source == SOURCE_LEGACY_MIGRATED
    assert SUBAGENTS_SETTING not in settings
    assert resolution.config is not None and resolution.config.enabled is True
    primary = resolution.config.items[0]
    assert primary.route.target_id == "claude=claude-opus-5"
    assert primary.route.credential_profile_id == "owner-account"
    assert primary.effort == "high"
    assert [row.route.target_id for row in resolution.config.items[1:]] == [
        "openai/gpt-5.6-sol",
        "openai/gpt-5.6-luna",
    ]


def test_legacy_off_is_explicit_false_and_never_becomes_default_candidate(
    _clean_subagent_env,
):
    candidate = parse_configured_subagents(_config(_row("candidate")))
    resolution = resolve_configured_subagents(
        {"OUROBOROS_SUBAGENT_HARNESS": " OFF "},
        default_candidate=candidate,
    )

    assert resolution.source == SOURCE_LEGACY_MIGRATED
    assert resolution.config is not None
    assert resolution.config.enabled is False
    assert resolution.config.items == ()


def test_legacy_off_preserves_custom_heavy_rows_without_reenabling_defaults():
    candidate = parse_configured_subagents(_config(_row("must-not-default")))
    resolution = resolve_configured_subagents(
        {
            "OUROBOROS_SUBAGENT_HARNESS": "off",
            "OUROBOROS_MODEL_HEAVY": "owner/custom-heavy",
        },
        default_candidate=candidate,
    )

    assert resolution.config is not None
    assert resolution.config.enabled is False
    assert [row.subagent_id for row in resolution.config.items] == ["legacy-heavy"]
    assert resolution.config.items[0].route.target_id == "owner/custom-heavy"


def test_empty_is_undecided_and_candidate_remains_unsaved(_clean_subagent_env):
    settings = {"OUROBOROS_SUBAGENT_HARNESS": ""}
    candidate = parse_configured_subagents(_config(_row("candidate")))
    resolution = resolve_configured_subagents(settings, default_candidate=candidate)

    assert resolution.source == SOURCE_UNDECIDED
    assert resolution.config == candidate
    assert settings["OUROBOROS_SUBAGENT_HARNESS"] == ""
    assert SUBAGENTS_SETTING not in settings


def test_malformed_nonempty_new_or_legacy_bytes_fail_closed_and_are_preserved():
    bad_new = "{not-json"
    new_resolution = resolve_configured_subagents({SUBAGENTS_SETTING: bad_new})
    legacy_resolution = resolve_configured_subagents(
        {
            "OUROBOROS_SUBAGENT_HARNESS": "=no-harness",
        }
    )

    assert new_resolution.source == SOURCE_INVALID
    assert new_resolution.raw == bad_new
    assert new_resolution.config is None
    assert legacy_resolution.source == SOURCE_INVALID
    assert legacy_resolution.raw == "=no-harness"
    assert legacy_resolution.config is None


def test_legacy_light_requires_a_truthful_provider_but_custom_heavy_is_preserved():
    resolution = resolve_configured_subagents(
        {
            "OUROBOROS_SUBAGENT_HARNESS": "codex=gpt-5.6-sol",
            "OUROBOROS_MODEL_HEAVY": "owner/custom-heavy",
            "OUROBOROS_MODEL_LIGHT": "openai::gpt-5.6-luna",
        }
    )

    assert resolution.config is not None
    assert [row.route.target_id for row in resolution.config.items] == [
        "codex=gpt-5.6-sol",
        "owner/custom-heavy",
    ]


def test_custom_heavy_without_singleton_is_an_unsaved_undecided_migration_candidate(
    _clean_subagent_env,
):
    settings = {"OUROBOROS_MODEL_HEAVY": "owner/custom-heavy"}
    resolution = resolve_configured_subagents(settings)

    assert resolution.source == SOURCE_UNDECIDED
    assert resolution.config is not None
    assert [row.route.target_id for row in resolution.config.items] == [
        "owner/custom-heavy",
    ]
    assert SUBAGENTS_SETTING not in settings


def test_saved_local_heavy_intent_survives_a_temporarily_missing_source():
    resolution = resolve_configured_subagents({
        "OUROBOROS_MODEL_HEAVY": "owner-model",
        "USE_LOCAL_HEAVY": True,
        "LOCAL_MODEL_SOURCE": "",
    })

    assert resolution.config is not None
    assert resolution.config.items[0].route.target_id == "owner-model (local)"


def test_legacy_code_plus_local_flag_migrates_to_an_explicit_local_actor():
    settings = {
        "OUROBOROS_MODEL_CODE": "openai/gpt-5.4-pro",
        "USE_LOCAL_CODE": True,
    }
    migrate_legacy_slot_keys(settings)
    normalized, changed, changed_keys = apply_runtime_provider_defaults(settings)
    resolution = resolve_configured_subagents(normalized)

    assert not changed
    assert changed_keys == []
    assert "OUROBOROS_MODEL_CODE" not in normalized
    assert "USE_LOCAL_CODE" not in normalized
    assert resolution.config is not None
    assert [(row.subagent_id, row.route.target_id) for row in resolution.config.items] == [
        ("legacy-heavy", "openai/gpt-5.4-pro (local)"),
    ]


def test_canonical_candidate_precedes_and_deduplicates_legacy_model_rows():
    candidate = parse_configured_subagents(_config(
        _row("primary-builder", route={
            "kind": "agent_session",
            "target_id": "claude=claude-opus-5",
            "credential_profile_id": "",
        }),
        _row("fast-scout", route={
            "kind": "api_model",
            "target_id": "openai/gpt-5.6-luna",
        }),
    ))
    resolution = resolve_configured_subagents({
        "OUROBOROS_SUBAGENT_HARNESS": "claude=claude-opus-5",
        "OUROBOROS_MODEL_HEAVY": "owner/custom-heavy",
        "OUROBOROS_MODEL_LIGHT": "openai/gpt-5.6-luna",
        "OPENROUTER_API_KEY": "configured",
    }, default_candidate=candidate)

    assert resolution.config is not None
    assert [row.route.target_id for row in resolution.config.items] == [
        "claude=claude-opus-5",
        "openai/gpt-5.6-luna",
        "owner/custom-heavy",
    ]


def test_settings_legacy_singleton_uses_the_canonical_linear_compiler(
    _clean_subagent_env,
):
    settings = {
        "OUROBOROS_SUBAGENT_HARNESS": "claude=claude-opus-5:high",
        "OUROBOROS_SUBAGENT_PROFILE": "owner-account",
        "OUROBOROS_MODEL": "openai/gpt-5.6-sol",
        "OUROBOROS_MODEL_LIGHT": "openai/gpt-5.6-luna",
        "OUROBOROS_MODEL_HEAVY": "owner/custom-heavy",
        "OPENROUTER_API_KEY": "configured",
    }

    resolution, diagnostics = resolve_settings_subagent_candidate(settings)

    assert diagnostics == ()
    assert resolution.source == SOURCE_LEGACY_MIGRATED
    assert resolution.config is not None
    assert [
        (
            row.subagent_id,
            row.route.target_id,
            row.route.credential_profile_id,
        )
        for row in resolution.config.items
    ] == [
        ("primary-builder", "claude=claude-opus-5", "owner-account"),
        ("fast-scout", "openai/gpt-5.6-luna", ""),
        ("independent-perspective", "openai/gpt-5.6-sol", ""),
        ("legacy-heavy", "owner/custom-heavy", ""),
    ]
    assert all(row.name == "" for row in resolution.config.items)

    duplicate_heavy = dict(settings)
    duplicate_heavy["OUROBOROS_MODEL_HEAVY"] = "openai/gpt-5.6-sol"
    deduplicated, _diagnostics = resolve_settings_subagent_candidate(duplicate_heavy)
    assert deduplicated.config is not None
    assert [row.subagent_id for row in deduplicated.config.items] == [
        "primary-builder",
        "fast-scout",
        "independent-perspective",
    ]


def test_row_enabled_defaults_true_and_only_a_false_row_changes_the_saved_bytes():
    """The per-row owner switch is additive: an existing roster's canonical
    serialization — and every fingerprint, receipt and snapshot bound to it —
    is byte-identical after the field exists, because `true` is omitted."""
    legacy_bytes = _config(_row("builder"))
    legacy = parse_configured_subagents(legacy_bytes)
    assert legacy.items[0].enabled is True
    assert "enabled" not in serialize_configured_subagents(legacy).split('"items"')[1]

    explicit_true = parse_configured_subagents(_config(_row("builder", enabled=True)))
    assert serialize_configured_subagents(explicit_true) == serialize_configured_subagents(legacy)
    assert configured_subagents_fingerprint(explicit_true) == configured_subagents_fingerprint(legacy)

    off = parse_configured_subagents(_config(_row("builder", enabled=False)))
    assert off.items[0].enabled is False
    assert json.loads(serialize_configured_subagents(off))["items"][0]["enabled"] is False
    assert configured_subagents_fingerprint(off) != configured_subagents_fingerprint(legacy)

    # A switched-off row keeps its complete configuration through the round trip.
    round_tripped = parse_configured_subagents(serialize_configured_subagents(off))
    assert round_tripped == off
    assert round_tripped.items[0].route.target_id == "codex=gpt-5.6-sol"
    assert round_tripped.items[0].effort == "medium"


@pytest.mark.parametrize("value", ["false", 0, 1, None, [], {}])
def test_a_nonboolean_row_enabled_is_refused_rather_than_coerced(value):
    with pytest.raises(ValueError, match=r"enabled must be a boolean"):
        parse_configured_subagents(_config(_row("builder", enabled=value)))


def test_the_row_switch_and_the_list_level_switch_are_independent():
    both_axes = parse_configured_subagents(
        _config(_row("a"), _row("b", enabled=False), enabled=False)
    )
    assert both_axes.enabled is False
    assert [row.enabled for row in both_axes.items] == [True, False]

    list_on = parse_configured_subagents(
        _config(_row("a"), _row("b", enabled=False), enabled=True)
    )
    assert list_on.enabled is True
    assert [row.enabled for row in list_on.items] == [True, False]
