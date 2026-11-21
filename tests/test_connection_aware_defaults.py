"""Automatic model and reviewer defaults follow the connection a key belongs to (#2969).

One rule (owner, 8 Nov 2026): automation proposes for the selected connection — the
built-in key first when it serves, else the only serving named connection of that kind;
several serving named connections and no built-in one leave the choice to the owner (Main
is named, nothing is invented). Saved models, catalogs, blanks and Off are never rewritten.
Keys are fixtures; nothing here sends a request.
"""

from __future__ import annotations

import json

import pytest

from ouroboros import review_pool_migration as m
from ouroboros.llm_routing import resolve_target
from ouroboros.provider_models import CONNECTIONS_KEY, compute_direct_review_seats
from ouroboros.server_runtime import apply_runtime_provider_defaults, provider_slot_proposals
from ouroboros.subscription_install_presets import factory_actor_rows, factory_review_rows

MODEL_SLOTS = ("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS")
WORK = {"kind": "openai", "name": "work", "api_key": "row-work"}
HOME = {"kind": "openai", "name": "home", "api_key": "row-home"}
BYOK = {"kind": "openrouter", "name": "byok", "api_key": "row-or", "provider_routing": "repro"}


def _doc(*rows, **flat):
    return {CONNECTIONS_KEY: [dict(row) for row in rows], **flat}


def _targets(rows):
    return [row["route"]["target_id"] for row in rows]


@pytest.fixture(autouse=True)
def _clean_migration_cache():
    m._MIGRATIONS_SEEN.clear()
    yield
    m._MIGRATIONS_SEEN.clear()


# --- the rule ----------------------------------------------------------------

def test_the_proposal_connection_is_the_built_in_first_then_the_only_named_one():
    from ouroboros.provider_models import proposal_connection

    assert proposal_connection("openai", _doc(WORK)) == "work"
    assert proposal_connection("openai", _doc(WORK, OPENAI_API_KEY="k")) == ""  # the built-in key stays first
    assert proposal_connection("openai", _doc(WORK, HOME)) is None  # the owner chooses
    disabled_builtin = {"kind": "openai", "name": "default", "enabled": False}
    assert proposal_connection("openai", _doc(WORK, disabled_builtin, OPENAI_API_KEY="k")) == "work"
    assert proposal_connection("openai", _doc({**HOME, "enabled": False}, WORK)) == "work"
    assert proposal_connection("openai", {}) == ""  # nothing serves: the shipped spelling stands


# --- F1: a named-only direct key ---------------------------------------------

def test_a_named_only_direct_key_proposes_its_models_on_that_connection():
    proposals = provider_slot_proposals(_doc(WORK))
    assert all(proposals[key].startswith("openai@work::") for key in MODEL_SLOTS if proposals[key])
    assert proposals["OUROBOROS_MODEL"].startswith("openai@work::")
    # The built-in twin is unchanged.
    assert provider_slot_proposals({"OPENAI_API_KEY": "k"})["OUROBOROS_MODEL"].startswith("openai::")
    assert provider_slot_proposals(_doc(WORK, OPENAI_API_KEY="k"))["OUROBOROS_MODEL"].startswith("openai::")


def test_a_named_only_direct_key_gets_reviewers_and_a_generalist_paid_by_that_key():
    doc = _doc(WORK)
    reviewers, actors = factory_review_rows(doc), factory_actor_rows(doc)
    assert reviewers and all(target.startswith("openai@work::") for target in _targets(reviewers))
    assert actors and actors[0]["subagent_id"] == "primary-builder"
    assert all(target.startswith("openai@work::") for target in _targets(actors))
    assert resolve_target(_targets(reviewers)[0], doc)["api_key"] == "row-work"


def test_two_named_connections_of_one_kind_leave_main_to_the_owner():
    doc = _doc(WORK, HOME)
    assert factory_review_rows(doc) == [] and factory_actor_rows(doc) == []
    assert "@" not in provider_slot_proposals(doc)["OUROBOROS_MODEL"]  # no connection is guessed
    # Once the owner chooses, everything follows that choice.
    chosen = {**doc, "OUROBOROS_MODEL": "openai@home::gpt-5.6-sol"}
    assert _targets(factory_review_rows(chosen)) == ["openai@home::gpt-5.6-sol"]


def test_headless_names_the_keys_main_waits_for(caplog):
    loaded = _doc(WORK, HOME)
    with caplog.at_level("WARNING", logger="ouroboros.review_pool_migration"):
        m.apply_at_read_seam(loaded, merged=True)
    assert "OUROBOROS_MODEL" in caplog.text and "Serving connections: openai@work, openai@home." in caplog.text


# --- F2: a named OpenRouter Main ---------------------------------------------

def test_a_named_openrouter_main_reviews_on_its_own_connection():
    from ouroboros.settings_defaults import OPENROUTER_REVIEW_DEFAULTS

    main = "openrouter@byok::google/gemini-3.8-flash"
    for doc in (_doc(BYOK, OUROBOROS_MODEL=main), _doc(BYOK, OUROBOROS_MODEL=main, OPENROUTER_API_KEY="k")):
        rows = factory_review_rows(doc)
        assert _targets(rows) == [f"openrouter@byok::{model}" for model in OPENROUTER_REVIEW_DEFAULTS["triad"]]
        target = resolve_target(_targets(rows)[0], doc)
        assert (target["api_key"], target["provider_routing"]) == ("row-or", "repro")
    # The built-in OpenRouter Main keeps the bare shipped panel.
    assert _targets(factory_review_rows({"OPENROUTER_API_KEY": "k"})) == list(OPENROUTER_REVIEW_DEFAULTS["triad"])


def test_a_named_only_openrouter_key_proposes_and_reviews_on_that_connection():
    doc = _doc(BYOK)
    assert provider_slot_proposals(doc)["OUROBOROS_MODEL"].startswith("openrouter@byok::")
    rows = factory_review_rows(doc)
    assert len(rows) == 3 and all(target.startswith("openrouter@byok::") for target in _targets(rows))


def test_the_openrouter_helpers_join_any_main_on_the_connection_the_router_resolves():
    """Owner 16.11: GLM, DeepSeek and Kimi join the factory actors beside any reachable OpenRouter as plain Auto rows,
    never marked. A named OpenRouter Main pays for them with its own key even beside the built-in one, another Main
    takes the router's proposal connection, and two named router keys with no Main on either add none."""
    from ouroboros.settings_defaults import OPENROUTER_HELPER_DEFAULTS

    ids = [row_id for row_id, _model, _use in OPENROUTER_HELPER_DEFAULTS]
    models = [model for _row_id, model, _use in OPENROUTER_HELPER_DEFAULTS]

    def helpers(doc):
        return [row for row in factory_actor_rows(doc) if row["subagent_id"] in ids]

    built_in = helpers({"OPENROUTER_API_KEY": "k"})
    assert _targets(built_in) == models
    assert all(not row.get("review_eligible") and not row.get("effort") for row in built_in)
    named_main = _doc(BYOK, OPENROUTER_API_KEY="k", OUROBOROS_MODEL="openrouter@byok::google/gemini-4-hydrogen")
    assert _targets(helpers(named_main)) == [f"openrouter@byok::{model}" for model in models]
    direct_main = _doc(BYOK, OPENAI_API_KEY="k", OUROBOROS_MODEL="openai::gpt-5.6-terra")
    assert _targets(helpers(direct_main)) == [f"openrouter@byok::{model}" for model in models]
    home = {**BYOK, "name": "home", "api_key": "row-home"}
    ambiguous = _doc(BYOK, home, OPENAI_API_KEY="k", OUROBOROS_MODEL="openai::gpt-5.6-terra")
    assert helpers(ambiguous) == [] and factory_actor_rows(ambiguous), "the direct Main keeps its own helper"
    chosen = _doc(BYOK, home, OUROBOROS_MODEL="openrouter@home::google/gemini-4-hydrogen")
    assert _targets(helpers(chosen)) == [f"openrouter@home::{model}" for model in models]


def _actor_targets(doc):
    return {row["subagent_id"]: row["route"]["target_id"] for row in factory_actor_rows(doc)}


def test_a_local_main_lends_the_helpers_no_router_key_its_id_still_names():
    """A local Main keeps the router id it was chosen under but runs on the local lane (``role_route_binding``), so it
    chose no OpenRouter key: the helpers take the router's proposal connection, the built-in key first, and two named
    router keys with no built-in one add none rather than the stale name's. Unticked, the same Main pays for them."""
    from ouroboros.settings_defaults import OPENROUTER_HELPER_DEFAULTS

    ids = [row_id for row_id, _model, _use in OPENROUTER_HELPER_DEFAULTS]
    models = [model for _row_id, model, _use in OPENROUTER_HELPER_DEFAULTS]
    main = "openrouter@byok::google/gemini-4-hydrogen"
    local = {"OUROBOROS_MODEL": main, "USE_LOCAL_MAIN": True, "LOCAL_MODEL_SOURCE": "/models/m.gguf"}

    beside_built_in = _actor_targets(_doc(BYOK, OPENROUTER_API_KEY="k", **local))
    assert beside_built_in["primary-builder"] == f"{main} (local)"
    assert [beside_built_in[row_id] for row_id in ids] == models
    home = {**BYOK, "name": "home", "api_key": "row-home"}
    ambiguous = _actor_targets(_doc(BYOK, home, **local))
    assert ambiguous["primary-builder"] == f"{main} (local)"
    assert not set(ids) & set(ambiguous), "several named router keys and Main on none: nothing is guessed"
    remote = _actor_targets(_doc(BYOK, OPENROUTER_API_KEY="k", **{**local, "USE_LOCAL_MAIN": False}))
    assert [remote[row_id] for row_id in ids] == [f"openrouter@byok::{model}" for model in models]


def test_a_helper_on_mains_route_is_one_row_however_the_route_is_spelled():
    """Bare, ``openrouter::`` and ``openrouter@default::`` spell one route on the built-in key: Main keeps its saved
    spelling and the GLM helper adds no twin. A local lane, a processing preference or a named connection is another
    route, so the helper stays beside it."""
    glm = "z-ai/glm-5.3"
    for spelling in (glm, f"openrouter::{glm}", f"openrouter@default::{glm}"):
        same = _actor_targets({"OPENROUTER_API_KEY": "k", "OUROBOROS_MODEL": spelling})
        assert same["primary-builder"] == spelling and "analysis-critic" not in same, spelling
    local = _actor_targets({"OPENROUTER_API_KEY": "k", "OUROBOROS_MODEL": glm,
                            "USE_LOCAL_MAIN": True, "LOCAL_MODEL_SOURCE": "/models/m.gguf"})
    assert (local["primary-builder"], local["analysis-critic"]) == (f"{glm} (local)", glm)
    processed = {"OPENROUTER_API_KEY": "k", "OUROBOROS_MODEL": glm,
                 "OUROBOROS_MODEL_PROCESSING_PREFERENCES": {"main": "fast"}}
    rows = {row["subagent_id"]: row for row in factory_actor_rows(processed)}
    assert rows["primary-builder"]["processing_preference"] == "fast"
    critic = rows["analysis-critic"]
    assert critic["route"]["target_id"] == glm and not critic.get("processing_preference")
    named = _actor_targets(_doc(BYOK, OPENROUTER_API_KEY="k", OUROBOROS_MODEL="google/gemini-4-hydrogen",
                                OUROBOROS_MODEL_LIGHT=f"openrouter@byok::{glm}"))
    assert (named["fast-scout"], named["analysis-critic"]) == (f"openrouter@byok::{glm}", glm)


def test_a_rows_identity_reads_the_route_grammar_and_keeps_every_other_axis():
    """The dedup identity (``_row_identity``): one route in three spellings is one; connection, local lane, model,
    account pin, processing and route kind each still separate rows."""
    from ouroboros.route_spec import ROUTE_KIND_AGENT_SESSION, ROUTE_KIND_API_MODEL, RouteSpec
    from ouroboros.subscription_install_presets import _row_identity

    def api(target, pin="", processing=""):
        return _row_identity(RouteSpec(ROUTE_KIND_API_MODEL, target, pin), processing)

    assert api("z-ai/glm-5.3") == api("openrouter::z-ai/glm-5.3") == api("openrouter@default::z-ai/glm-5.3")
    assert api("openai::gpt-6-luna") == api("openai@default::gpt-6-luna") != api("openai/gpt-6-luna")
    distinct = [api("z-ai/glm-5.3"), api("openrouter@byok::z-ai/glm-5.3"), api("z-ai/glm-5.3 (local)"),
                api("z-ai/glm-5.3-flash"), api("z-ai/glm-5.3", processing="fast"),
                api("claudexor::codex=gpt-6-astra"), api("claudexor::codex=gpt-6-astra", pin="koshak"),
                _row_identity(RouteSpec(ROUTE_KIND_AGENT_SESSION, "z-ai/glm-5.3"))]
    assert len(set(distinct)) == len(distinct)


# --- F3: MiniMax's mixed panel on a named connection --------------------------

def test_a_named_minimax_main_keeps_its_mixed_panel_on_its_connection():
    seats = compute_direct_review_seats("minimax", "minimax@cn::MiniMax-M3")
    assert [role for role, _model in seats] == ["main", "", ""]
    assert [model for _role, model in seats][1:] == ["minimax@cn::MiniMax-M3.1-Flash"] * 2
    named_light = compute_direct_review_seats("minimax", "minimax@cn::MiniMax-M3", "minimax@cn::MiniMax-M3.1-Flash")
    assert named_light == [("main", "minimax@cn::MiniMax-M3"), ("light", "minimax@cn::MiniMax-M3.1-Flash"),
                           ("light", "minimax@cn::MiniMax-M3.1-Flash")]
    # The built-in twin is byte for byte the shipped panel.
    assert compute_direct_review_seats("minimax", "minimax::MiniMax-M3") == [
        ("main", "minimax::MiniMax-M3"), ("", "minimax::MiniMax-M3.1-Flash"), ("", "minimax::MiniMax-M3.1-Flash")]


# --- F4: a saved id is not respelled onto a key the install lacks -------------

def test_a_saved_router_id_is_not_respelled_onto_a_missing_built_in_key():
    saved = _doc(WORK, OUROBOROS_MODEL="openai/gpt-5.5")
    normalized, changed, keys = apply_runtime_provider_defaults(saved)
    assert normalized["OUROBOROS_MODEL"] == "openai/gpt-5.5" and "OUROBOROS_MODEL" not in keys
    # The built-in key keeps its respelling.
    builtin = apply_runtime_provider_defaults({"OPENAI_API_KEY": "k", "OUROBOROS_MODEL": "openai/gpt-5.5"})
    assert builtin[0]["OUROBOROS_MODEL"] == "openai::gpt-5.5"


# --- F5: the migration digest sees a connection's readiness -------------------

def test_enabling_a_named_connection_is_a_new_migration_input():
    main = {"OUROBOROS_MODEL": "openai@work::gpt-5.6-sol"}
    off, on = _doc({**WORK, "enabled": False}, **main), _doc(WORK, **main)
    assert m.input_sha256(off) != m.input_sha256(on)
    assert m.input_sha256(_doc(WORK, **main)) == m.input_sha256(_doc(WORK, **main))
    # The digest carries no secret: a rotated key is the same input.
    assert m.input_sha256(on) == m.input_sha256(_doc({**WORK, "api_key": "rotated"}, **main))
    first = dict(off)
    m.apply_at_read_seam(first, merged=True)
    assert not [row for row in json.loads(first.get("OUROBOROS_SUBAGENTS") or '{"items": []}')["items"]
                if row.get("review_eligible")]
    second = dict(on)
    m.apply_at_read_seam(second, merged=True)
    assert _targets(r for r in json.loads(second["OUROBOROS_SUBAGENTS"])["items"]
                    if r.get("review_eligible")) == ["openai@work::gpt-5.6-sol"]


LAB = {"kind": "openai-compatible", "name": "lab", "base_url": "http://127.0.0.1:9/v1", "api_key": "row-lab"}
LAB_MAIN = {"OUROBOROS_MODEL": "openai-compatible@lab::m"}


@pytest.mark.parametrize("base,variant", [
    (_doc(BYOK), _doc({**BYOK, "provider_routing": "other"})),  # read at dispatch: the same rows
    (_doc(LAB, **LAB_MAIN), _doc({**LAB, "base_url": "http://127.0.0.1:10/v1"}, **LAB_MAIN)),
    (_doc(LAB, **LAB_MAIN), _doc({**LAB, "base_url": ""}, **LAB_MAIN)),  # no longer serves
    (_doc(WORK), _doc({**WORK, "enabled": False})),
    (_doc(WORK), _doc(WORK, HOME)),  # a second key: Main waits for the owner
], ids=["provider_routing", "base_url", "base_url-removed", "disabled", "second-key"])
def test_a_warm_migration_cache_never_replays_another_connections_catalog(base, variant):
    def catalog(doc):
        loaded = json.loads(json.dumps(doc))
        m.apply_at_read_seam(loaded, merged=True)
        return loaded.get("OUROBOROS_SUBAGENTS")

    cold = catalog(variant)
    m._MIGRATIONS_SEEN.clear()
    catalog(base)
    assert catalog(variant) == cold


# --- P3: Safety's lane is the one its resolved model names --------------------

def test_an_empty_light_sends_the_supervisor_where_its_own_local_box_says(monkeypatch):
    """Owner, 15.11 (#3292): the box of an empty Light is the owner's choice for every Light call, the Supervisor
    that inherits it included. Ticked under a cloud Main it is an explicit local route (a failing local runtime
    holds); unticked, the Supervisor follows Main's lane, and a local Main it merely inherits tolerates an outage."""
    from ouroboros.model_slots import get_safety_model
    from ouroboros.safety import _resolve_safety_routing
    from ouroboros.safety_config import safety_configuration

    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    monkeypatch.setenv("OUROBOROS_MODEL", "google/gemini-3.8-flash")
    for key in ("OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_SAFETY"):
        monkeypatch.setenv(key, "")
    document = {"OUROBOROS_MODEL": "google/gemini-3.8-flash", "OUROBOROS_MODEL_LIGHT": "", "OUROBOROS_MODEL_SAFETY": "",
                "USE_LOCAL_MAIN": False, "USE_LOCAL_LIGHT": True}
    monkeypatch.setenv("USE_LOCAL_MAIN", "false")
    monkeypatch.setenv("USE_LOCAL_LIGHT", "true")
    assert get_safety_model() == "google/gemini-3.8-flash (local)"
    assert _resolve_safety_routing() == (True, False, None)
    assert safety_configuration(document)[:2] == ("google/gemini-3.8-flash", True)  # the owner guard's snapshot agrees
    # The still-working direction: unticked, the Supervisor stays on the cloud Main.
    monkeypatch.setenv("USE_LOCAL_LIGHT", "false")
    assert get_safety_model() == "google/gemini-3.8-flash"
    assert _resolve_safety_routing() == (False, False, None)
    # A box is read as its reader's values come: a settings document by the loader's spellings (a hand-written
    # "yes" is what the loader types as ticked), the live settings as the Safety routing reads them, so the rule and
    # safety.py never name two lanes for one value.
    for spelling in ("yes", "on"):
        assert safety_configuration({**document, "USE_LOCAL_LIGHT": spelling})[:2] == ("google/gemini-3.8-flash", True)
        monkeypatch.setenv("USE_LOCAL_LIGHT", spelling)  # untyped in the environment: not a tick for either reader
        assert get_safety_model() == "google/gemini-3.8-flash"
        assert _resolve_safety_routing() == (False, False, None)
    assert safety_configuration({**document, "USE_LOCAL_LIGHT": False})[:2] == ("google/gemini-3.8-flash", False)
    # A Light of its own on the local lane checks locally; so does a local Main an unticked Light inherits.
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "owner/light")
    monkeypatch.setenv("USE_LOCAL_LIGHT", "true")
    assert _resolve_safety_routing() == (True, False, None)
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "")
    monkeypatch.setenv("USE_LOCAL_LIGHT", "false")
    monkeypatch.setenv("USE_LOCAL_MAIN", "true")
    assert _resolve_safety_routing() == (True, True, None)


# --- the loader every reader shares (startup, Settings GET, headless) ---------

def test_the_settings_loader_runs_a_named_only_install_on_its_own_key(tmp_path, monkeypatch):
    from ouroboros import config as cfg

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir, raising=True)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", data_dir / "settings.json", raising=True)
    for key in (*cfg.SETTINGS_DEFAULTS, CONNECTIONS_KEY, "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    cfg.reset_runtime_mode_baseline_for_tests()
    (data_dir / "settings.json").write_text(json.dumps({CONNECTIONS_KEY: [WORK]}), encoding="utf-8")
    try:
        loaded = cfg.load_settings()
    finally:
        cfg.reset_runtime_mode_baseline_for_tests()
    assert loaded["OUROBOROS_MODEL"].startswith("openai@work::")
    items = json.loads(loaded["OUROBOROS_SUBAGENTS"])["items"]
    assert [row for row in items if row.get("review_eligible")]
    assert all(row["route"]["target_id"].startswith("openai@work::") for row in items)
    assert "OUROBOROS_MODEL" not in json.loads((data_dir / "settings.json").read_text(encoding="utf-8"))


# --- the owner is asked: several named keys, an unauthored Main ----------------

def settings_read(tmp_path, monkeypatch, document, **environ):
    """The real ``GET /api/settings`` body for ``document`` on disk (temp data dir, no network)."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    import server as srv
    from ouroboros import config as cfg

    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir, raising=True)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", data_dir / "settings.json", raising=True)
    for key in (*cfg.SETTINGS_DEFAULTS, CONNECTIONS_KEY, "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for key, value in environ.items():
        monkeypatch.setenv(key, value)
    (data_dir / "settings.json").write_text(json.dumps(document), encoding="utf-8")
    app = Starlette(routes=[Route("/api/settings", endpoint=srv.api_settings_get, methods=["GET"])])
    app.state.drive_root, app.state.repo_dir = data_dir, tmp_path
    cfg.reset_runtime_mode_baseline_for_tests()
    try:
        with TestClient(app) as client:
            return client.get("/api/settings").json()
    finally:
        cfg.reset_runtime_mode_baseline_for_tests()


def test_settings_asks_the_owner_to_choose_mains_key_among_several_named_ones(tmp_path, monkeypatch):
    body = settings_read(tmp_path, monkeypatch, {CONNECTIONS_KEY: [WORK, HOME]})
    assert body["_meta"]["main_connection_choice"] == ["openai@work", "openai@home"]
    assert body["OUROBOROS_MODEL"].startswith("openai::")  # the shipped id, on no key: nothing guessed
    assert not [row for row in json.loads(body["OUROBOROS_SUBAGENTS"] or '{"items": []}')["items"]
                if row.get("review_eligible")]


@pytest.mark.parametrize("document,environ", [
    ({CONNECTIONS_KEY: [WORK, HOME], "OUROBOROS_MODEL": "openai@home::gpt-5.6-sol"}, {}),  # the owner's choice
    ({CONNECTIONS_KEY: [WORK, HOME], "OUROBOROS_MODEL": ""}, {}),  # a saved blank is a choice too
    ({CONNECTIONS_KEY: [WORK, HOME]}, {"OUROBOROS_MODEL": "openai@work::gpt-5.6-sol"}),  # an operator's variable
    ({CONNECTIONS_KEY: [WORK]}, {}),  # one key: proposals use it
    ({CONNECTIONS_KEY: [WORK, HOME], "OPENAI_API_KEY": "k"}, {}),  # the built-in key comes first
    ({CONNECTIONS_KEY: [WORK, BYOK]}, {}),  # OpenRouter beside it: the shipped router id runs
])
def test_no_choice_is_asked_when_main_is_chosen_or_unambiguous(tmp_path, monkeypatch, document, environ):
    assert settings_read(tmp_path, monkeypatch, document, **environ)["_meta"]["main_connection_choice"] == []


def test_settings_names_what_the_limits_cannot_count(tmp_path, monkeypatch):
    """`_meta.known_spend_gaps`, said under the limits on Dashboard › Costs (pricing.spend_coverage)."""
    direct = settings_read(tmp_path, monkeypatch, {"OPENAI_API_KEY": "k", "OUROBOROS_MODEL": "openai::gpt-5.6-sol"})
    assert direct["_meta"]["known_spend_gaps"][0] == "all model roles", "a direct key has no catalog price"
    routed = settings_read(tmp_path, monkeypatch, {"OPENROUTER_API_KEY": "k"})
    assert routed["_meta"]["known_spend_gaps"] == [], "OpenRouter states a price for every route here"


def test_after_the_owner_picks_a_key_the_question_ends_and_the_factory_rows_use_it(tmp_path, monkeypatch):
    body = settings_read(tmp_path, monkeypatch, {CONNECTIONS_KEY: [WORK, HOME],
                                                 "OUROBOROS_MODEL": "openai@work::gpt-5.6-sol"})
    assert body["_meta"]["main_connection_choice"] == []
    rows = json.loads(body["OUROBOROS_SUBAGENTS"])["items"]
    assert [row for row in rows if row.get("review_eligible")]
    assert all(row["route"]["target_id"].startswith("openai@work::") for row in rows)
    # Unauthored Light and Fallbacks follow the key the owner chose for Main; nothing else is written.
    assert body["OUROBOROS_MODEL_LIGHT"].startswith("openai@work::")
    assert all(model.strip().startswith("openai@work::") for model in body["OUROBOROS_MODEL_FALLBACKS"].split(",")
               if model.strip())
    assert set(json.loads((tmp_path / "data" / "settings.json").read_text(encoding="utf-8"))) == {
        CONNECTIONS_KEY, "OUROBOROS_MODEL"}


# --- the wizard: what it may propose is carried per slot, and every direct provider has its defaults ------

def test_the_bootstrap_names_the_slots_no_document_or_operator_variable_sets(monkeypatch):
    """``automaticModels``: the wizard proposes and retargets these alone. A saved value, blank included, and an
    operator's own variable are choices; this process tree's own projection of a value is not an operator's."""
    from ouroboros import settings_integrity as integrity
    from ouroboros.settings_setup_contract import build_setup_bootstrap

    def automatic(document, **environ):
        for key in ("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_VISION", "OUROBOROS_MODEL_FALLBACKS",
                    "OUROBOROS_MODEL_SAFETY", "OUROBOROS_MODEL_CONSCIOUSNESS",
                    integrity.PROJECTED_PROPOSALS_ENV, integrity.PROJECTED_SETTINGS_ENV):
            monkeypatch.delenv(key, raising=False)
        for key, value in environ.items():
            monkeypatch.setenv(key, value)
        return build_setup_bootstrap({**document, "OUROBOROS_MODEL": "x"}, "web", document=document)[
            "initialState"]["automaticModels"]

    every = ["mainModel", "lightModel", "safetyModel", "visionModel", "consciousnessModel", "fallbackModel"]
    assert automatic({}) == every
    assert automatic({CONNECTIONS_KEY: [WORK, HOME]}) == every  # keys are no model choice
    blanks = {"OUROBOROS_MODEL": "", "OUROBOROS_MODEL_LIGHT": "", "OUROBOROS_MODEL_FALLBACKS": ""}
    assert automatic(blanks) == ["safetyModel", "visionModel", "consciousnessModel"]
    assert automatic({**blanks, "OUROBOROS_MODEL_CONSCIOUSNESS": ""}) == ["safetyModel", "visionModel"]
    assert automatic({}, OUROBOROS_MODEL_LIGHT="openai@work::owner-light") == [key for key in every if key != "lightModel"]
    assert automatic({}, OUROBOROS_MODEL_LIGHT="") == every  # a blank variable sets nothing
    echo = json.dumps({"OUROBOROS_MODEL_LIGHT": "openai::gpt-5.6-luna"})
    assert automatic({}, OUROBOROS_MODEL_LIGHT="openai::gpt-5.6-luna", **{integrity.PROJECTED_PROPOSALS_ENV: echo}) == every
    mirrored = json.dumps({"OUROBOROS_MODEL_VISION": integrity._value_digest("shipped-vision")})
    assert automatic({}, OUROBOROS_MODEL_VISION="shipped-vision", **{integrity.PROJECTED_SETTINGS_ENV: mirrored}) == every
    # A caller with no stored document reads the settings it passed: a loaded mapping carries every slot.
    assert build_setup_bootstrap({"OUROBOROS_MODEL": "x", "OUROBOROS_MODEL_LIGHT": "", "OUROBOROS_MODEL_VISION": "",
                                  "OUROBOROS_MODEL_FALLBACKS": "", "OUROBOROS_MODEL_SAFETY": "",
                                  "OUROBOROS_MODEL_CONSCIOUSNESS": ""}, "web")["initialState"]["automaticModels"] == []


def test_the_wizard_has_every_direct_providers_own_defaults():
    from ouroboros.provider_models import DIRECT_PROVIDER_DEFAULTS
    from ouroboros.settings_setup_contract import build_setup_bootstrap

    shipped = build_setup_bootstrap({}, "web")["modelDefaults"]
    for kind, defaults in DIRECT_PROVIDER_DEFAULTS.items():
        assert kind in shipped, kind
        assert {key: value for key, value in shipped[kind].items() if value} == {
            key: value for key, value in defaults.items() if key != "heavy" and value}, kind


def test_the_wizard_reads_every_way_a_built_in_key_is_ready():
    """Each field a canonical kind needs to be ready (``PROVIDER_KINDS[kind].ready``) is a wizard provider field, so
    its built-in access reaches the bootstrap, the draft and both onboarding endpoints; and each direct kind has
    its own profile copy."""
    from ouroboros.provider_models import DIRECT_PROVIDER_DEFAULTS, PROVIDER_KINDS
    from ouroboros.settings_setup_contract import build_setup_contract

    contract = build_setup_contract("web")
    fields = {field["settingKey"] for field in contract["providerFields"]}
    for kind, spec in PROVIDER_KINDS.items():
        needed = {spec.flat_key(field) for group in spec.ready for field in group}
        assert needed <= fields, (kind, needed - fields)
    assert set(DIRECT_PROVIDER_DEFAULTS) <= set(contract["providerProfiles"])


# --- the key's own API decides which defaults it proposes (OpenAI Responses) ----

RESPONSES_LAB = {"kind": "openai", "name": "lab", "api_key": "row-lab", "api_surface": "responses"}


def test_a_proposal_takes_the_defaults_of_the_api_its_own_key_chose():
    """One table per key, never one for the kind: the only named key on Responses proposes its profile, a key that
    never chose keeps the earlier one, and among several named keys the one Main runs on decides. A saved Main stays."""
    assert provider_slot_proposals(_doc(RESPONSES_LAB))["OUROBOROS_MODEL"] == "openai@lab::gpt-6.1-terra"
    assert provider_slot_proposals(_doc(WORK))["OUROBOROS_MODEL"] == "openai@work::gpt-5.6-terra"
    mixed = _doc(RESPONSES_LAB, WORK)
    assert "@" not in provider_slot_proposals(mixed)["OUROBOROS_MODEL"]  # no key, no API is guessed
    assert provider_slot_proposals({**mixed, "OUROBOROS_MODEL": "openai@lab::gpt-6.1-sol"})["OUROBOROS_MODEL_LIGHT"] == \
        "openai@lab::gpt-6-luna"
    assert provider_slot_proposals({**mixed, "OUROBOROS_MODEL": "openai@work::gpt-5.6-sol"})["OUROBOROS_MODEL_LIGHT"] == \
        "openai@work::gpt-5.6-luna"


@pytest.mark.parametrize(("rows", "main"), [((RESPONSES_LAB,), "openai@lab::gpt-6.1-terra"), ((WORK,), "openai@work::gpt-5.6-terra")])
def test_the_wizard_bootstrap_proposes_on_the_api_of_the_only_named_key(tmp_path, monkeypatch, rows, main):
    """The real ``GET /api/onboarding/bootstrap`` of a document holding one named OpenAI key: the wizard starts from
    the proposal of that key's own API, and its masked row keeps the stored word the browser resolves it by."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import config as cfg
    from ouroboros.gateway import onboarding_host

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir, raising=True)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", data_dir / "settings.json", raising=True)
    for key in (*cfg.SETTINGS_DEFAULTS, CONNECTIONS_KEY, "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    (data_dir / "settings.json").write_text(json.dumps(_doc(*rows)), encoding="utf-8")
    app = Starlette(routes=[Route("/api/onboarding/bootstrap", onboarding_host.api_onboarding_bootstrap)])
    app.state.drive_root = data_dir
    cfg.reset_runtime_mode_baseline_for_tests()
    try:
        with TestClient(app) as client:
            state = client.get("/api/onboarding/bootstrap").json()["bootstrap"]["initialState"]
    finally:
        cfg.reset_runtime_mode_baseline_for_tests()
    assert state["mainModel"] == main
    assert [row.get("api_surface") for row in state["connections"]] == [row.get("api_surface") for row in rows]
