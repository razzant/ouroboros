"""Saved model ids are not rewritten — not as "retired", not as one of our former defaults.

A deleted table used to rewrite saved ids (gpt-5.4, gemini previews) in place, claiming
a retirement no provider catalog showed; a later table rewrote a saved value equal to one
of Ouroboros's own former defaults, which equality cannot tell from the owner's choice
(owner decision 2A). These tests pin that an active slot keeps its saved id on every
install class — only its spelling moves onto the one exclusive provider's prefix — and
name the kept exceptions: the retired Heavy and scope-review keys, which no route reads,
still classify our former defaults.
"""
import json

import pytest

from ouroboros.server_runtime import apply_runtime_provider_defaults
from ouroboros.configured_subagents import resolve_configured_subagents
from tests.test_server_runtime import _FORMERLY_REMAPPED_IDS
from tests._cases import chosen_rows


def test_formerly_remapped_review_ids_are_no_longer_rewritten():
    """Raw comma-key review values (load_settings purges these keys; a raw dict still
    reaches the normalizer) stay exactly as written on an aggregator install."""
    old_main = "openai/gpt-" + "5.4"
    old_pro = "openai/gpt-" + "5.4-pro"
    old_mini = "openai/gpt-" + "5.4-mini"
    document = {
        "OPENROUTER_API_KEY": "sk-or",
        "OUROBOROS_REVIEW_MODELS": f"{old_main},{old_mini}",
        "OUROBOROS_SCOPE_REVIEW_MODEL": old_pro,
        "OUROBOROS_SCOPE_REVIEW_MODELS": f"{old_pro},{old_mini}",
    }
    normalized, changed, changed_keys = apply_runtime_provider_defaults(dict(document))

    assert (changed, changed_keys) == (False, [])
    assert normalized == document


def test_a_saved_model_is_not_rewritten_as_retired_on_an_aggregator_install(subtests):
    for model in _FORMERLY_REMAPPED_IDS:
        with subtests.test(model=model):
            from ouroboros.reviewer_slot_config import review_pool_rows

            catalog = json.dumps({"enabled": True, "items": [
                {"subagent_id": "t1", "recommended_use": "Reviews.", "review_eligible": True,
                 "route": {"kind": "api_model", "target_id": model}},
                {"subagent_id": "s1", "recommended_use": "Reviews natively.", "review_eligible": True,
                 "route": {"kind": "api_model", "target_id": model}, "delivery": "native"},
            ]})
            document = {
                "OPENROUTER_API_KEY": "sk-or",
                "OUROBOROS_MODEL": model,
                "OUROBOROS_MODEL_LIGHT": model,
                "OUROBOROS_MODEL_FALLBACKS": model,
                "OUROBOROS_SUBAGENTS": catalog,
            }
            normalized, changed, changed_keys = apply_runtime_provider_defaults(dict(document))

            assert (changed, changed_keys) == (False, [])
            assert normalized == document
            assert [row.target_id for row in review_pool_rows(normalized)] == [model, model]


def test_direct_openai_keeps_a_saved_id_that_was_never_our_default_for_its_slot():
    document = {
        "OPENAI_API_KEY": "sk-openai",
        "OUROBOROS_MODEL": "openai::gpt-5.4-pro",
        "OUROBOROS_MODEL_LIGHT": "openai::gpt-5.4",
        "OUROBOROS_MODEL_FALLBACKS": "openai::gpt-5.4",
    }
    normalized, changed, changed_keys = apply_runtime_provider_defaults(dict(document))

    assert (changed, changed_keys) == (False, [])
    assert normalized == document


@pytest.mark.parametrize("model", (
    "google/gemini-3.1-pro-preview",
    "openai::gpt-5.4-pro",
    # Our direct default was only ever stored as `openai::gpt-5.4`; the slash spelling
    # shipped solely inside the old review list and as a UI suggestion.
    "openai/gpt-5.4",
))
def test_a_saved_heavy_id_that_was_never_our_default_stays_the_owner_actor(model):
    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "OPENROUTER_API_KEY": "sk-or",
        "OUROBOROS_MODEL_HEAVY": model,
    })

    resolution = resolve_configured_subagents(normalized)
    assert (changed, changed_keys) == (False, [])
    assert normalized["OUROBOROS_MODEL_HEAVY"] == model
    assert resolution.config is not None
    assert [(row.subagent_id, row.route.target_id) for row in resolution.config.items] == [
        ("legacy-heavy", model),
    ]


@pytest.mark.parametrize("document,key,migrated", [
    pytest.param({"OPENAI_API_KEY": "sk-openai", "OUROBOROS_SCOPE_REVIEW_MODEL": "openai::gpt-5.4"},
                 "OUROBOROS_SCOPE_REVIEW_MODEL", "openai::gpt-5.6-terra", id="direct-scope-review-legacy-default"),
    pytest.param({"OPENROUTER_API_KEY": "sk-or", "OUROBOROS_SCOPE_REVIEW_MODEL": "openai/gpt-5.5"},
                 "OUROBOROS_SCOPE_REVIEW_MODEL", "openai/gpt-5.6-terra", id="scope-review-prior-default"),
    pytest.param({"OUROBOROS_MODEL_HEAVY": "google/gemini-3.1-flash-lite"},
                 "OUROBOROS_MODEL_HEAVY", "", id="former-code-default-in-heavy"),
])
def test_only_the_retired_keys_still_classify_our_former_defaults(document, key, migrated):
    """The named exceptions: retired keys no route reads (a stale scope-review value, the
    Heavy actor seed) still classify a value equal to one of OUR former shipped defaults."""
    normalized, _changed, changed_keys = apply_runtime_provider_defaults(dict(document))

    assert normalized[key] == migrated
    assert key in changed_keys


@pytest.mark.parametrize("document,key", [
    pytest.param({"OPENAI_API_KEY": "sk-openai", "OUROBOROS_MODEL": "google/gemini-3.1-flash-lite"},
                 "OUROBOROS_MODEL", id="former-main-default-on-direct-openai"),
    pytest.param({"ANTHROPIC_API_KEY": "sk-ant", "OUROBOROS_MODEL_LIGHT": "google/gemini-3.1-flash-lite"},
                 "OUROBOROS_MODEL_LIGHT", id="former-light-default-on-direct-anthropic"),
])
def test_a_saved_former_default_in_an_active_slot_survives(document, key):
    """Owner decision 2A: a saved active slot equal to a former default stays as written —
    on a direct-only install an unreachable router id is honestly unavailable, never
    silently another provider."""
    normalized, _changed, changed_keys = apply_runtime_provider_defaults(dict(document))

    assert normalized[key] == document[key]
    assert key not in changed_keys


_DIRECT_PROVIDER_CREDENTIALS = {
    "openai": {"OPENAI_API_KEY": "sk-openai"},
    "anthropic": {"ANTHROPIC_API_KEY": "sk-ant"},
    "cloudru": {"CLOUDRU_FOUNDATION_MODELS_API_KEY": "cloudru-key"},
    "gigachat": {"GIGACHAT_CREDENTIALS": "giga-creds"},
    "minimax": {"MINIMAX_API_KEY": "minimax-key"},
    "deepseek": {"DEEPSEEK_API_KEY": "deepseek-key"},
    "zai": {"ZAI_API_KEY": "zai-key"},
}
_LOCAL_ONLY = {"LOCAL_MODEL_SOURCE": "repo/model.gguf", "USE_LOCAL_MAIN": True}
# Our shipped Main/Code/Light default from v5.31.0-rc.1 to v5.32.0-rc.1 (Code is Heavy now).
_OUR_FORMER_GEMINI_DEFAULT = "google/gemini-3.1-flash-lite"


def test_our_former_gemini_default_stays_in_active_slots_on_every_direct_provider(subtests):
    """A stored copy of our former Main/Code/Light default on a direct-only install is a saved
    value: it stays (unreachable, so honestly unavailable) in every active slot, while the
    retired Heavy key still drops it rather than seeding an actor."""
    for provider in sorted(_DIRECT_PROVIDER_CREDENTIALS):
        with subtests.test(provider=provider):
            former = _OUR_FORMER_GEMINI_DEFAULT
            normalized, _changed, changed_keys = apply_runtime_provider_defaults({
                **_DIRECT_PROVIDER_CREDENTIALS[provider],
                "OUROBOROS_MODEL": former,
                "OUROBOROS_MODEL_HEAVY": former,
                "OUROBOROS_MODEL_LIGHT": former,
                "OUROBOROS_MODEL_FALLBACKS": former,
            })

            assert [normalized[key] for key in ("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS")] == [
                former] * 3
            assert normalized["OUROBOROS_MODEL_HEAVY"] == ""
            assert changed_keys == ["OUROBOROS_MODEL_HEAVY"]


def test_a_local_only_install_keeps_a_saved_former_gemini_light():
    """Only an UNAUTHORED Light on a local-only install is proposed blank (the loader); a
    saved one stays as written. The retired Heavy key still drops our former default."""
    former = _OUR_FORMER_GEMINI_DEFAULT
    normalized, _changed, changed_keys = apply_runtime_provider_defaults({
        **_LOCAL_ONLY,
        "OUROBOROS_MODEL_HEAVY": former,
        "OUROBOROS_MODEL_LIGHT": former,
        "OUROBOROS_MODEL_FALLBACKS": former,
    })

    assert (normalized["OUROBOROS_MODEL_LIGHT"], normalized["OUROBOROS_MODEL_FALLBACKS"]) == (former, former)
    assert normalized["OUROBOROS_MODEL_HEAVY"] == ""
    assert changed_keys == ["OUROBOROS_MODEL_HEAVY"]


@pytest.mark.parametrize("stored", ["openai::gpt-5.4", "openai/gpt-5.4"])
def test_our_former_openai_direct_main_default_stays_and_only_its_spelling_moves(stored):
    """`openai::gpt-5.4` was our first direct OpenAI Main/Code default (4.44.0 to 4.50.0-rc.9):
    a saved copy stays; the slash spelling reaches the same direct identity by syntax alone."""
    normalized, _changed, changed_keys = apply_runtime_provider_defaults({
        "OPENAI_API_KEY": "sk-openai",
        "OUROBOROS_MODEL": stored,
    })

    assert normalized["OUROBOROS_MODEL"] == "openai::gpt-5.4"
    assert ("OUROBOROS_MODEL" in changed_keys) is (stored != "openai::gpt-5.4")


def test_our_former_openai_code_default_leaves_heavy_without_becoming_an_actor():
    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "OUROBOROS_MODEL_HEAVY": "openai::gpt-5.4",
    })

    resolution = resolve_configured_subagents(normalized)
    rows = () if resolution.config is None else resolution.config.items
    assert (changed, changed_keys) == (True, ["OUROBOROS_MODEL_HEAVY"])
    assert normalized["OUROBOROS_MODEL_HEAVY"] == ""
    assert all(row.subagent_id != "legacy-heavy" for row in rows)


@pytest.mark.parametrize("install", chosen_rows([*sorted(_DIRECT_PROVIDER_CREDENTIALS), "local-only"], 0, 2, 7))
def test_an_explicit_gemini_pro_preview_choice_stays_as_written(install):
    """It was never one of our Main/Code/Light/Fallback defaults (it shipped only inside the
    old review list), so no install class rewrites it."""
    choice = "google/gemini-3.1-pro-preview"
    slots = ("OUROBOROS_MODEL", "OUROBOROS_MODEL_HEAVY", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS")
    credentials = _LOCAL_ONLY if install == "local-only" else _DIRECT_PROVIDER_CREDENTIALS[install]
    normalized, _changed, changed_keys = apply_runtime_provider_defaults({
        **credentials, **{key: choice for key in slots},
    })

    assert {key: normalized[key] for key in slots} == {key: choice for key in slots}
    assert not set(slots) & set(changed_keys)
