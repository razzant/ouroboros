"""A direct provider's factory review seats run their ROLE's whole binding, at the consumers.

The provider panel is declared as roles (``provider_models.DIRECT_PROVIDER_REVIEW_ROLES``;
MiniMax: Main, Light, Light; the others: Main once). A Light seat runs Light's binding — its ``(local)`` label and
processing — never Main's because the two share a model name, for review and for delegation
alike. Without a Light binding to run (a blank or another provider's Light: the provider's
shipped Light stands in; a local Light without a source) the seat stays the panel's plain
row, and a panel without Light seats keeps Main's binding. Reviewer count and effort stay.
Each test drives a real settings file through the task start, the review pool and a
delegated row's dispatch (the fixtures are the defaults-provenance suite's).
"""

from __future__ import annotations

import pytest

from ouroboros.provider_models import ANTHROPIC_DIRECT_DEFAULTS, MINIMAX_DIRECT_DEFAULTS
from tests.test_defaults_provenance import _task_start, _write
from tests.test_defaults_provenance import install as provenance_install

install = provenance_install

M3, M27 = MINIMAX_DIRECT_DEFAULTS["main"], MINIMAX_DIRECT_DEFAULTS["light"]
CHOSEN_MINIMAX = {"OPENROUTER_API_KEY": "sk-or-present-key", "MINIMAX_API_KEY": "mm-present-key",
                  "OUROBOROS_MODEL": M3, "OUROBOROS_MODEL_PROCESSING_PREFERENCES": {"light": "economy"}}
LOCAL_SOURCE = {"LOCAL_MODEL_SOURCE": "/models/m.gguf"}


def _seats_and_delegate(snapshot, subagent_id):
    from ouroboros.reviewer_slot_config import review_pool_rows, review_pool_slots
    from ouroboros.settings_integrity import task_settings_scope
    from ouroboros.subagent_runtime import resolve_configured_actor_dispatch, select_subagent_snapshot

    with task_settings_scope(snapshot):
        settings = dict(snapshot.settings)
        seats = [(row.target_id, row.processing_preference, slot.use_local, row.effort)
                 for row, slot in zip(review_pool_rows(settings), review_pool_slots(settings), strict=True)]
        row = select_subagent_snapshot(settings, subagent_id=subagent_id)[0]
        dispatch = resolve_configured_actor_dispatch(
            {"id": "child", "configured_subagent": row, "task_constraint": {}}, task_type="task")
    assert not dispatch.blocked
    return seats, (dispatch.lane.model, dispatch.lane.use_local_model, row["processing_preference"])


@pytest.mark.parametrize("light, seat", [
    ({"OUROBOROS_MODEL_LIGHT": M27, "USE_LOCAL_LIGHT": True, **LOCAL_SOURCE}, (f"{M27} (local)", "economy", True)),
    ({"OUROBOROS_MODEL_LIGHT": M3}, (M3, "economy", False)),
    ({"OUROBOROS_MODEL_LIGHT": M3, "USE_LOCAL_LIGHT": True, **LOCAL_SOURCE}, (f"{M3} (local)", "economy", True)),
], ids=["partial-local-light", "light-on-mains-model", "light-on-mains-model-locally"])
def test_a_providers_light_seats_run_lights_binding_for_review_and_delegation(install, light, seat):
    _write(install, {**CHOSEN_MINIMAX, **light})
    seats, delegate = _seats_and_delegate(_task_start(), "review-2")
    assert seats == [(M3, "", False, "high"), (*seat, "high"), (*seat, "high")]
    assert delegate == (seat[0].removesuffix(" (local)"), seat[2], "economy")


@pytest.mark.parametrize("light, model", [
    ({"OUROBOROS_MODEL_LIGHT": ""}, M27), ({"OUROBOROS_MODEL_LIGHT": "openai/gpt-6-luna"}, M27),
    ({"OUROBOROS_MODEL_LIGHT": "minimax::MiniMax-M2.5", "USE_LOCAL_LIGHT": True}, "minimax::MiniMax-M2.5"),
], ids=["saved-blank-shipped-light-stands-in", "other-providers-shipped-light-stands-in", "local-without-source"])
def test_light_seats_without_a_light_binding_keep_the_panels_plain_seat(install, light, model):
    _write(install, {**CHOSEN_MINIMAX, **light})
    seats, delegate = _seats_and_delegate(_task_start(), "review-3")
    assert seats == [(M3, "", False, "high"), (model, "", False, "high"), (model, "", False, "high")]
    assert delegate == (model, False, "")


@pytest.mark.parametrize("main_tier,row_tier", [("fast", "fast"), ("ultrafast", "economy")])
def test_a_panel_without_light_seats_copies_only_a_tier_its_rows_can_store(install, main_tier, row_tier):
    main = ANTHROPIC_DIRECT_DEFAULTS["main"]
    _write(install, {"OPENROUTER_API_KEY": "sk-or-present-key", "ANTHROPIC_API_KEY": "sk-ant-present-key",
                     "OUROBOROS_MODEL": main, "OUROBOROS_MODEL_LIGHT": main, "USE_LOCAL_LIGHT": True,
                     **LOCAL_SOURCE, "OUROBOROS_PROCESSING_PREFERENCE": "economy",
                     "OUROBOROS_MODEL_PROCESSING_PREFERENCES": {"main": main_tier, "light": "economy"}})
    # The generated row uses the global tier when Main has its Main-only preference.
    seats, delegate = _seats_and_delegate(_task_start(), "review-1")
    assert seats == [(main, row_tier, False, "high")]
    assert delegate == (main, False, row_tier)
