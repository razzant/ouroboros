"""MiniMax proposals and saved intent at the settings and roster consumers."""
import json

import pytest

from tests._safety_fixtures import settings_install as install_fixture

install = install_fixture


MAIN = "minimax::MiniMax-M3"
FLASH = "minimax::MiniMax-M3.1-Flash"
OLD = "minimax::MiniMax-M2.7"


@pytest.mark.parametrize("existing", [False, True], ids=["fresh", "unauthored-upgrade"])
def test_minimax_proposals_reach_runtime_scout_review_and_inherited_safety(install, monkeypatch, existing):
    from ouroboros import config
    from ouroboros.model_slots import get_fallback_models, get_light_model, get_safety_model
    from ouroboros.reviewer_slot_config import review_pool_rows
    from ouroboros.settings_integrity import task_settings_scope
    from ouroboros.subagent_runtime import apply_task_start_settings

    if existing:
        install.write_text(json.dumps({"TOTAL_BUDGET": 71.0}))
    monkeypatch.setenv("MINIMAX_API_KEY", "fixture-minimax-key")
    snapshot = apply_task_start_settings()
    settings = snapshot.settings
    assert settings["OUROBOROS_MODEL"] == MAIN
    assert settings["OUROBOROS_MODEL_LIGHT"] == settings["OUROBOROS_MODEL_FALLBACKS"] == FLASH
    assert settings["OUROBOROS_MODEL_SAFETY"] == ""
    with task_settings_scope(snapshot):
        assert get_light_model() == get_safety_model() == FLASH
        assert get_fallback_models() == [FLASH]
    rows = json.loads(settings["OUROBOROS_SUBAGENTS"])["items"]
    helpers = {row["subagent_id"]: row for row in rows if not row.get("review_eligible")}
    assert helpers["primary-builder"]["route"]["target_id"] == MAIN
    assert helpers["fast-scout"]["route"]["target_id"] == FLASH
    pool = review_pool_rows(settings)
    assert [row.target_id for row in pool] == [MAIN, FLASH, FLASH]
    assert [row.effort for row in pool] == ["high"] * 3
    assert all(row.get("delivery", "native") == "native" for row in rows if row.get("review_eligible"))
    # Neither a read nor its environment echo authors the proposed models/catalog.
    assert config.load_settings()["OUROBOROS_MODEL_LIGHT"] == FLASH
    saved = json.loads(install.read_text()) if install.exists() else {}
    assert not {"OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_SUBAGENTS"} & saved.keys()


@pytest.mark.parametrize("light", [OLD, "", "minimax::owner-light"], ids=["old-pin", "blank", "custom"])
def test_minimax_saved_slots_options_and_factory_minted_catalog_stay_authored(install, light):
    from ouroboros import config
    from ouroboros.gateway.owner_settings import _owner_update_settings, fields_on_saved_document
    from ouroboros.reviewer_slot_config import review_pool_rows

    catalog = json.dumps({"enabled": True, "items": [{
        "subagent_id": "saved-review", "recommended_use": "Saved reviewer.",
        "route": {"kind": "api_model", "target_id": OLD}, "effort": "medium",
        "review_eligible": True, "minted_from": "factory_default", "delivery": "packet",
    }]})
    document = {
        "MINIMAX_API_KEY": "fixture-minimax-key", "OUROBOROS_MODEL": MAIN,
        "OUROBOROS_MODEL_LIGHT": light, "OUROBOROS_MODEL_FALLBACKS": light,
        "OUROBOROS_MODEL_SAFETY": OLD, "OUROBOROS_SUBAGENTS": catalog,
        "OUROBOROS_MODEL_ACCOUNTS": '{"safety":"saved-account"}',
        "OUROBOROS_MODEL_CONTEXT_WINDOWS": '{"light":64000}',
        "OUROBOROS_MODEL_PROCESSING_PREFERENCES": '{"light":"economy"}',
    }
    install.write_text(json.dumps(document))
    loaded = config.load_settings()
    assert all(loaded[key] == value for key, value in document.items())
    assert [row.target_id for row in review_pool_rows(loaded)] == [OLD]
    _owner_update_settings(fields_on_saved_document({"TOTAL_BUDGET": 71.0}))
    saved = json.loads(install.read_text())
    assert all(saved[key] == value for key, value in document.items())


def test_minimax_wizard_and_colab_propose_flash_without_rewriting_explicit_choices(install):
    from ouroboros import config
    from ouroboros.colab_bootstrap import build_colab_settings, write_colab_settings
    from ouroboros.settings_setup_contract import build_setup_bootstrap

    setup = build_setup_bootstrap({"MINIMAX_API_KEY": "fixture-minimax-key"}, fresh_install=True)
    assert setup["initialState"]["mainModel"] == MAIN
    assert setup["initialState"]["lightModel"] == FLASH
    assert FLASH in setup["modelSuggestions"] and OLD not in setup["modelSuggestions"]
    for choices in ({}, {"OUROBOROS_MODEL_LIGHT": OLD, "OUROBOROS_MODEL_FALLBACKS": ""}):
        built = build_colab_settings({"MINIMAX_API_KEY": "fixture-minimax-key"}, existing=choices)
        write_colab_settings(install.parent, built)
        loaded = config.load_settings()
        assert loaded["OUROBOROS_MODEL"] == MAIN
        assert loaded["OUROBOROS_MODEL_LIGHT"] == choices.get("OUROBOROS_MODEL_LIGHT", FLASH)
        assert loaded["OUROBOROS_MODEL_FALLBACKS"] == choices.get("OUROBOROS_MODEL_FALLBACKS", FLASH)
