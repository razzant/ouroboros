"""Shipped defaults reach only what nobody chose: the owner's defaults decisions at the consumers.

Each test drives a product seam over a real settings file and environment — the task-start
snapshot every route consumer reads, the Settings save, the owner-endpoint write seam, the
read seam's factory catalog and the review pool built from it, a delegated row's dispatch,
the LLM route resolver, the onboarding preset compiler and setup contract, the Colab
builder — never a private helper alone.

* A saved model value survives as written, blank and default-equal included; only the
  spelling migration onto the exclusive provider applies. A route without access stays
  honestly unavailable.
* An absent value follows the configured access on every read; a save that does not edit
  a slot or a never-configured catalog does not author it (an Agents edit or the wizard's
  Finish does), and an incidental writer lands only its own field under the lock.
* The factory catalog follows the reachable Main (no imaginary OpenRouter panel), carries
  the wizard's Auto helpers on a never-configured install, and keeps its reviewer count and
  effort; a local row runs locally for review and delegation alike. The process's own
  projection of it is no operator catalog, and no Light helper stands in for a Main without access.
* A dedicated compatible endpoint is one pair with its own key (blank = no auth).
* An unused advisory model never refuses a preset; a missing preferred generation is a
  typed refusal, never a fallback to the previous one.
"""

from __future__ import annotations

import json
import os

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from tests._safety_fixtures import pin_safety_model, settings_install as install_fixture
install = install_fixture

from ouroboros import config as cfg
from ouroboros import review_pool_migration as m
from ouroboros.provider_models import ANTHROPIC_DIRECT_DEFAULTS, OPENAI_DIRECT_DEFAULTS
from ouroboros.settings_defaults import OPENROUTER_REVIEW_DEFAULTS, SETTINGS_DEFAULTS

SUBAGENTS = "OUROBOROS_SUBAGENTS"
SLOTS = ("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS")
ANTHROPIC = (ANTHROPIC_DIRECT_DEFAULTS["main"], ANTHROPIC_DIRECT_DEFAULTS["light"], ANTHROPIC_DIRECT_DEFAULTS["fallback"])
OPENAI = (OPENAI_DIRECT_DEFAULTS["main"], OPENAI_DIRECT_DEFAULTS["light"], OPENAI_DIRECT_DEFAULTS["fallback"])
SHIPPED = tuple(SETTINGS_DEFAULTS[key] for key in SLOTS)


def _write(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")


def _task_start():
    """What a task starts with: the normalized projection every route consumer reads."""
    from ouroboros.subagent_runtime import apply_task_start_settings

    return apply_task_start_settings()


def _slots(snapshot):
    return tuple(snapshot.settings.get(key) for key in SLOTS)


def _items(settings):
    return json.loads(settings[SUBAGENTS])["items"] if str(settings.get(SUBAGENTS) or "").strip() else []


def _pool_targets(settings):
    from ouroboros.reviewer_slot_config import review_pool_rows

    return [row.target_id for row in review_pool_rows(settings)]


# --- 2A: saved values survive, absent values follow the configured access ----------------


@pytest.mark.parametrize("document, key, expected", [
    ({"ANTHROPIC_API_KEY": "k", "OUROBOROS_MODEL": "anthropic::claude-opus-4-8"},
     "OUROBOROS_MODEL", "anthropic::claude-opus-4-8"),
    ({"OPENAI_API_KEY": "k", "OUROBOROS_MODEL_LIGHT": "openai::gpt-5.4-mini"},
     "OUROBOROS_MODEL_LIGHT", "openai::gpt-5.4-mini"),
    ({"ANTHROPIC_API_KEY": "k", "OUROBOROS_MODEL": "anthropic::claude-opus-5"},
     "OUROBOROS_MODEL", "anthropic::claude-opus-5"),
    ({"OPENROUTER_API_KEY": "k", "OUROBOROS_MODEL_LIGHT": "openai/gpt-5.6-luna"},
     "OUROBOROS_MODEL_LIGHT", "openai/gpt-5.6-luna"),
    ({"ANTHROPIC_API_KEY": "k", "OUROBOROS_MODEL_LIGHT": SETTINGS_DEFAULTS["OUROBOROS_MODEL_LIGHT"]},
     "OUROBOROS_MODEL_LIGHT", SETTINGS_DEFAULTS["OUROBOROS_MODEL_LIGHT"]),
], ids=["former-anthropic-default", "former-openai-light", "pre-refresh-default", "router-saved-light",
        "saved-shipped-light-on-direct"])
def test_a_saved_model_name_survives_even_when_it_equals_a_default(install, document, key, expected):
    _write(install, document)
    assert _task_start().settings[key] == expected


def test_an_explicit_route_without_access_stays_unavailable_not_rerouted(install):
    from ouroboros.provider_models import model_has_credentials
    from ouroboros.settings_integrity import task_settings_scope

    _write(install, {"ANTHROPIC_API_KEY": "k", "OUROBOROS_MODEL": "google/gemini-3.8-flash"})
    snapshot = _task_start()
    assert snapshot.settings["OUROBOROS_MODEL"] == "google/gemini-3.8-flash"
    with task_settings_scope(snapshot):
        assert model_has_credentials("google/gemini-3.8-flash") is False


@pytest.mark.parametrize("dotted, direct", [("4.6", "4-6"), ("5.5", "5-5")])
def test_spelling_migration_onto_the_exclusive_provider_is_not_a_generation_replacement(install, dotted, direct):
    from ouroboros.llm import LLMClient

    _write(install, {"ANTHROPIC_API_KEY": "k", "OUROBOROS_MODEL": f"anthropic/claude-opus-{dotted}"})
    assert _task_start().settings["OUROBOROS_MODEL"] == f"anthropic::claude-opus-{direct}"
    assert LLMClient()._resolve_remote_target(f"anthropic::claude-opus-{dotted}")["resolved_model"] == (
        f"claude-opus-{direct}")
    assert LLMClient()._resolve_remote_target(f"anthropic/claude-opus-{dotted}")["resolved_model"] == (
        f"anthropic/claude-opus-{dotted}"), "the router keeps its own slug"


def test_saved_blank_light_and_fallbacks_keep_their_meaning_end_to_end(install, monkeypatch):
    """Blank Light inherits Main; blank Fallbacks is "no cross-model fallback" — through the
    loader, the environment projection and the chain parser, even beside the legacy alias."""
    from ouroboros.model_slots import get_fallback_models, get_light_model, parse_fallback_chain
    from ouroboros.settings_integrity import task_settings_scope

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACK", "anthropic::claude-legacy-alias")
    _write(install, {"ANTHROPIC_API_KEY": "k", "OUROBOROS_MODEL_LIGHT": "", "OUROBOROS_MODEL_FALLBACKS": ""})
    snapshot = _task_start()
    assert _slots(snapshot) == (ANTHROPIC[0], "", "")
    with task_settings_scope(snapshot):
        assert get_light_model() == ANTHROPIC[0]
        assert parse_fallback_chain() == [] and get_fallback_models() == []


def test_the_legacy_fallback_alias_still_serves_a_process_with_no_modern_chain(monkeypatch):
    from ouroboros.model_slots import parse_fallback_chain

    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACKS", raising=False)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACK", "a/one, b/two")
    assert parse_fallback_chain() == ["a/one", "b/two"]


@pytest.mark.parametrize("document", [None, {}, {"TOTAL_BUDGET": 50.0}], ids=["no-file", "raw-empty", "unrelated-key"])
@pytest.mark.parametrize("env, expected", [
    ({"ANTHROPIC_API_KEY": "k"}, ANTHROPIC),
    ({"OPENAI_API_KEY": "k"}, OPENAI),
    ({"OPENROUTER_API_KEY": "k"}, SHIPPED),
], ids=["anthropic", "openai", "openrouter"])
def test_absent_slots_follow_the_configured_access_on_every_read(install, monkeypatch, document, env, expected):
    if document is not None:
        _write(install, document)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert _slots(_task_start()) == expected


def test_a_local_only_install_leaves_unauthored_light_and_fallbacks_on_main(install):
    _write(install, {"USE_LOCAL_MAIN": True, "LOCAL_MODEL_SOURCE": "/models/m.gguf"})
    assert _slots(_task_start())[1:] == ("", "")
    _write(install, {"USE_LOCAL_MAIN": True, "LOCAL_MODEL_SOURCE": "/models/m.gguf",
                     "OUROBOROS_MODEL_LIGHT": "openai/gpt-5.6-luna"})
    assert _task_start().settings["OUROBOROS_MODEL_LIGHT"] == "openai/gpt-5.6-luna", "a saved choice is kept"


def _settings_app(monkeypatch):
    from ouroboros.gateway import settings as settings_mod

    monkeypatch.setattr(settings_mod, "_start_supervisor_if_needed_for_request", lambda *_a, **_k: False)
    monkeypatch.setattr(settings_mod, "_apply_settings_save_side_effects", lambda *_a, **_k: None)
    app = Starlette(routes=[Route("/api/settings", endpoint=settings_mod.api_settings_post, methods=["POST"])])
    app.state.drive_root = cfg.DATA_DIR
    app.state.repo_dir = cfg.DATA_DIR
    return app


def test_a_settings_save_that_does_not_edit_a_slot_does_not_author_it(install, monkeypatch):
    """The Settings form posts every model role; a slot absent on disk and posted back exactly
    as the save read it stays unauthored, so switching provider keys moves its proposal along —
    also after this process projected the proposals (our echo, no operator's variable)."""
    from ouroboros.server_runtime import apply_runtime_provider_defaults

    _write(install, {"OPENROUTER_API_KEY": "sk-or-present-key"})
    assert _slots(_task_start()) == SHIPPED
    shown = apply_runtime_provider_defaults(cfg.load_settings())[0]
    assert tuple(shown[key] for key in SLOTS) == SHIPPED
    # Keep this defaults test independent of the owner-only inherited-checker ratchet.
    pin_safety_model()
    response = TestClient(_settings_app(monkeypatch)).post("/api/settings", json={
        "OPENROUTER_API_KEY": "", "ANTHROPIC_API_KEY": "sk-ant-present-key",
        **{key: shown[key] for key in SLOTS}})
    assert response.status_code == 200, response.text
    stored = json.loads(install.read_text(encoding="utf-8"))
    assert not set(SLOTS) & set(stored), stored
    assert _slots(_task_start()) == ANTHROPIC
    # An edited slot is authored; the untouched ones still are not.
    response = TestClient(_settings_app(monkeypatch)).post("/api/settings", json={
        **{key: _task_start().settings[key] for key in SLOTS}, "OUROBOROS_MODEL_LIGHT": ""})
    assert response.status_code == 200, response.text
    stored = json.loads(install.read_text(encoding="utf-8"))
    assert stored["OUROBOROS_MODEL_LIGHT"] == "" and "OUROBOROS_MODEL" not in stored
    assert _slots(_task_start()) == (ANTHROPIC[0], "", ANTHROPIC[2])


@pytest.mark.parametrize("key, value, runs", [
    ("OUROBOROS_MODEL", "anthropic::claude-operator-main", "anthropic::claude-operator-main"),
    ("OUROBOROS_MODEL_LIGHT", "anthropic::claude-operator-light", "anthropic::claude-operator-light"),
    ("OUROBOROS_MODEL_FALLBACKS", "anthropic::claude-operator-fallback", "anthropic::claude-operator-fallback"),
    ("OUROBOROS_MODEL", SHIPPED[0], SHIPPED[0]),  # equal to a shipped default, still the operator's
    ("OUROBOROS_MODEL", "anthropic/claude-sonnet-5", "anthropic::claude-sonnet-5"),  # read respelled onto the provider
], ids=["main", "light", "fallbacks", "default-equal", "respelled"])
def test_an_operator_variable_slot_survives_an_unrelated_save_in_this_process_and_every_next_task(
        install, monkeypatch, key, value, runs):
    """An operator's variable sets a slot the document does not (2A). An unrelated Settings save posts it back as GET
    showed it: it is neither authored nor replaced by a proposal, so this process and each next task run it as the
    operator wrote it — the projection never records the operator's value, respelled or not, as our echo."""
    monkeypatch.setenv(key, value)
    _write(install, {"ANTHROPIC_API_KEY": "sk-ant-present-key"})
    client = _settings_client(monkeypatch)
    shown = client.get("/api/settings").json()
    response = client.post("/api/settings", json={"TOTAL_BUDGET": "25", **{slot: shown[slot] for slot in SLOTS}})
    assert response.status_code == 200, response.text
    assert key not in json.loads(install.read_text(encoding="utf-8"))
    assert [_task_start().settings[key] for _task in range(2)] == [runs, runs]


def test_an_owner_endpoint_write_does_not_materialize_unauthored_slots_or_the_factory_catalog(install, monkeypatch):
    from ouroboros.gateway.owner_settings import _owner_update_settings

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-only-credential")
    _write(install, {"OUROBOROS_UI_LANGUAGE": ""})
    _owner_update_settings(lambda current: {**current, "OUROBOROS_UI_LANGUAGE": "ru"})
    stored = json.loads(install.read_text(encoding="utf-8"))
    assert stored["OUROBOROS_UI_LANGUAGE"] == "ru"
    assert not set(SLOTS) & set(stored) and not str(stored.get(SUBAGENTS) or "").strip(), stored
    assert not stored.get("OPENAI_API_KEY"), "the environment's credential is not written into the document"
    settings = cfg.load_settings()
    assert _slots(_task_start()) == OPENAI
    assert _pool_targets(settings) == [OPENAI[0]]


# D: an unrelated Settings save leaves a never-configured catalog to the read; an Agents edit or the wizard's
# Finish authors one, and a saved one stays the owner's (owner decision, quiz 8cda9017).


def _settings_client(monkeypatch):
    from ouroboros.gateway import onboarding as gw_onboarding
    from ouroboros.gateway import settings as settings_mod

    app = _settings_app(monkeypatch)
    app.router.routes += [
        Route("/api/settings", endpoint=settings_mod.api_settings_get, methods=["GET"]),
        Route("/api/onboarding/subagents/preview", endpoint=gw_onboarding.api_onboarding_subagents_preview,
              methods=["POST"]),
        Route("/api/onboarding/complete", endpoint=gw_onboarding.api_onboarding_complete, methods=["POST"])]
    return TestClient(app)


@pytest.mark.parametrize("posted", ["omitted", "as-read", "as-object"])
def test_an_unrelated_settings_save_leaves_a_never_configured_catalog_to_the_read(install, monkeypatch, posted):
    """Settings posts the catalog it loaded with every save. Posted back unchanged (as read, or as the page's
    object) or not at all, a never-configured install's factory catalog stays off the document, so a later Main
    and access change moves it: at the next task start, and in this process's own projection, which
    ``GET /api/review-pool`` reads. No catalog fetch checks its ids for an owner's typo: nobody typed them."""
    from ouroboros.gateway import settings as settings_mod

    checked = []
    monkeypatch.setattr(settings_mod, "_unrecognised_review_models", lambda models: checked.append(models) or [])
    _write(install, {"ANTHROPIC_API_KEY": "sk-ant-present-key"})
    pin_safety_model()
    client = _settings_client(monkeypatch)
    for changes in ({"TOTAL_BUDGET": "25"},
                    {"OPENAI_API_KEY": "sk-openai-present-key", "ANTHROPIC_API_KEY": "", "OUROBOROS_MODEL": OPENAI[0]}):
        shown = client.get("/api/settings").json()[SUBAGENTS]
        catalog = {"omitted": {}, "as-read": {SUBAGENTS: shown}, "as-object": {SUBAGENTS: json.loads(shown)}}[posted]
        response = client.post("/api/settings", json={**changes, **catalog})
        assert response.status_code == 200, response.text
        assert SUBAGENTS not in json.loads(install.read_text(encoding="utf-8"))
        assert client.get("/api/settings").json()["_meta"]["available_subagents"]["automatic"] is True
    assert _pool_targets(dict(os.environ)) == [OPENAI[0]] and checked == []
    assert _pool_targets(_task_start().settings) == [OPENAI[0]]


WIZARD = {"ANTHROPIC_API_KEY": "sk-ant-present-key", "OUROBOROS_MODEL": ANTHROPIC[0], "TOTAL_BUDGET": 25.0,
          "OUROBOROS_PER_TASK_COST_USD": 5.0, "OUROBOROS_REVIEW_ENFORCEMENT": "advisory",
          "OUROBOROS_RUNTIME_MODE": "advanced", "subscriptionsConnected": False}
AGENTS_EDITS = {  # the owner's edit of the shown factory catalog in Settings → Agents
    "reviewer-effort": lambda catalog: next(row for row in catalog["items"] if row.get("review_eligible")).update(
        effort="xhigh"),
    "helper-removed": lambda catalog: catalog["items"].pop(0),
    "delegation-off": lambda catalog: catalog.update(enabled=False),
    "emptied": lambda catalog: catalog.update(items=[]),
}


@pytest.mark.parametrize("origin", [*AGENTS_EDITS, "saved-equal-to-factory", "wizard-finish"])
def test_an_authored_catalog_stays_the_owners_through_a_main_and_access_change(install, monkeypatch, origin):
    from ouroboros.configured_subagents import canonical_subagents
    from ouroboros.gateway import onboarding
    from ouroboros import skill_starter

    client = _settings_client(monkeypatch)
    if origin == "wizard-finish":  # Finish saves the catalog the wizard displayed
        starters, scheduled = [], []
        # This settings test never launches background work (including a real hub fetch).
        monkeypatch.setattr(skill_starter, "schedule_starter", lambda *a, **kw: scheduled.append((a, kw)))
        monkeypatch.setattr(onboarding, "after_onboarding", lambda root, ids: starters.append((root, ids)))
        catalog = client.post("/api/onboarding/subagents/preview", json=WIZARD).json()["available_subagents"]
        response = client.post("/api/onboarding/complete", json={**WIZARD, SUBAGENTS: catalog})
        assert len(starters) == 1 and starters[0][1] is None
        assert scheduled == []
    else:
        _write(install, {"ANTHROPIC_API_KEY": "sk-ant-present-key"})
        catalog = json.loads(client.get("/api/settings").json()[SUBAGENTS])
        if origin == "saved-equal-to-factory":  # saved before: the owner's even where it equals the factory
            _write(install, {"ANTHROPIC_API_KEY": "sk-ant-present-key", SUBAGENTS: json.dumps(catalog)})
            response = client.post("/api/settings", json={"TOTAL_BUDGET": "25", SUBAGENTS: catalog})
        else:
            AGENTS_EDITS[origin](catalog)
            response = client.post("/api/settings", json={SUBAGENTS: catalog,
                                                          "allow_empty_review_pool": origin == "emptied"})
    assert response.status_code == 200, response.text
    read = client.get("/api/settings").json()
    assert read["_meta"]["available_subagents"]["automatic"] is False, "saved, so Settings says Saved"
    pin_safety_model()
    response = client.post("/api/settings", json={
        "OPENAI_API_KEY": "sk-openai-present-key", "ANTHROPIC_API_KEY": "", "OUROBOROS_MODEL": OPENAI[0],
        SUBAGENTS: read[SUBAGENTS]})
    assert response.status_code == 200, response.text
    saved = canonical_subagents(catalog)
    assert canonical_subagents(json.loads(install.read_text(encoding="utf-8"))[SUBAGENTS]) == saved
    assert canonical_subagents(_task_start().settings[SUBAGENTS]) == saved


def test_an_operator_catalog_stays_the_environments_through_a_settings_save(install, monkeypatch):
    operator = json.dumps({"enabled": True, "items": [{
        "subagent_id": "operator", "recommended_use": "Operator row.", "review_eligible": True, "effort": "high",
        "route": {"kind": "api_model", "target_id": "anthropic::claude-operator"}}]}, indent=1)
    monkeypatch.setenv(SUBAGENTS, operator)
    _write(install, {"ANTHROPIC_API_KEY": "sk-ant-present-key"})
    pin_safety_model()
    client = _settings_client(monkeypatch)
    response = client.post("/api/settings", json={
        "OPENAI_API_KEY": "sk-openai-present-key", SUBAGENTS: client.get("/api/settings").json()[SUBAGENTS]})
    assert response.status_code == 200, response.text
    assert SUBAGENTS not in json.loads(install.read_text(encoding="utf-8"))
    assert os.environ[SUBAGENTS] == operator, "byte for byte: no echo of ours the next read would skip"
    assert client.get("/api/settings").json()["_meta"]["available_subagents"]["automatic"] is False
    assert _pool_targets(_task_start().settings) == ["anthropic::claude-operator"]


SAVED_CHOICES = {"ANTHROPIC_API_KEY": "sk-ant-present-key", "OUROBOROS_MODEL_FALLBACKS": "",
                 "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE": "full",
                 "OUROBOROS_MODEL_LIGHT": SETTINGS_DEFAULTS["OUROBOROS_MODEL_LIGHT"],
                 "OUROBOROS_RUNTIME_MODE": "pro"}  # the owner's pending next-boot mode


def _incidental_write(writer, value):
    import launcher
    from ouroboros.tools.control_runtime import _set_tool_timeout

    confirm = lambda _title, _message: True  # noqa: E731 — the owner confirmed the native dialog
    return {"launcher-runtime-mode": lambda: launcher._request_runtime_mode_change(value, confirm),
            "launcher-auto-grant": lambda: launcher._request_auto_grant_reviewed_skills_change(False, confirm),
            "tool-timeout": lambda: _set_tool_timeout(None, value)}[writer]()


def _running_in(monkeypatch, mode):
    import launcher

    monkeypatch.setattr(launcher, "get_runtime_mode", lambda: mode)
    monkeypatch.setattr(cfg, "_BOOT_RUNTIME_MODE", mode)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)


@pytest.mark.parametrize("writer, key, value", [  # each value differs from its shipped default
    ("launcher-runtime-mode", "OUROBOROS_RUNTIME_MODE", "light"),
    ("launcher-auto-grant", "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "false"),
    ("tool-timeout", "OUROBOROS_TOOL_TIMEOUT_SEC", 900),
])
def test_an_incidental_writer_lands_its_key_not_the_reads_proposals(install, monkeypatch, writer, key, value):
    """The launcher's two natively confirmed owner decisions and the agent's ``set_tool_timeout`` each
    change one key, read and written inside the settings lock. Only that key is authored: the proposals
    (unauthored slots, a never-configured install's factory catalog) follow the next access change, every
    saved value — blank, default-equal, another provider's id, the pending next-boot mode — stays as
    written, and an owner write landing while the writer reads is not reverted."""
    import threading

    from ouroboros.gateway import owner_settings

    _running_in(monkeypatch, "advanced")
    _write(install, SAVED_CHOICES)
    assert _pool_targets(cfg.load_settings()) == [ANTHROPIC[0]], "the read mints the factory panel"
    read, landed = owner_settings._owner_read_settings_raw, threading.Event()

    def read_beside_a_concurrent_owner_write(*args, **kwargs):
        document = read(*args, **kwargs)
        if not getattr(read_beside_a_concurrent_owner_write, "started", False):
            read_beside_a_concurrent_owner_write.started = True
            threading.Thread(target=lambda: (owner_settings._owner_update_settings(
                owner_settings.fields_on_saved_document({"OUROBOROS_UI_LANGUAGE": "ru"})), landed.set()),
                daemon=True).start()
            landed.wait(0.3)  # after a read outside the lock it lands now; inside the lock it waits its turn
        return document

    monkeypatch.setattr(owner_settings, "_owner_read_settings_raw", read_beside_a_concurrent_owner_write)
    _incidental_write(writer, value)
    assert landed.wait(10)
    stored = json.loads(install.read_text(encoding="utf-8"))
    assert stored == {**SAVED_CHOICES, key: value, "OUROBOROS_UI_LANGUAGE": "ru"}, stored
    if writer == "tool-timeout":  # the process takes the field alone: no saved mode reaches the environment
        assert (os.environ["OUROBOROS_TOOL_TIMEOUT_SEC"], os.environ["OUROBOROS_RUNTIME_MODE"]) == ("900", "advanced")
    owner_settings._owner_update_settings(lambda current: {**current, "OPENROUTER_API_KEY": "sk-or-present-key"},
                           allow_safety_lowering=True)
    assert _task_start().settings["OUROBOROS_MODEL"] == SHIPPED[0]
    assert _pool_targets(cfg.load_settings()) == list(OPENROUTER_REVIEW_DEFAULTS["triad"])


@pytest.mark.parametrize("writer, key, value", [
    ("launcher-runtime-mode", "OUROBOROS_RUNTIME_MODE", "pro"),
    ("launcher-auto-grant", "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "false"),
    ("tool-timeout", "OUROBOROS_TOOL_TIMEOUT_SEC", 900),
])
def test_an_incidental_writer_creating_the_document_authors_no_other_choice(install, monkeypatch, writer, key, value):
    """An environment-configured install has no document: the writer creates one with its key and the
    optional bounds as they resolve now (a document's presence turns their absence into the legacy
    finite values), never a default over an environment choice — the running Access is not raised."""
    _running_in(monkeypatch, "light")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-key")
    _incidental_write(writer, value)
    bounds = {bound: SETTINGS_DEFAULTS[bound] for bound in ("OUROBOROS_MAX_ROUNDS", "OUROBOROS_TASK_ABS_CEILING_SEC")}
    assert json.loads(install.read_text(encoding="utf-8")) == {**bounds, key: value}
    settings = cfg.load_settings()
    assert {bound: settings[bound] for bound in bounds} == bounds
    assert settings["OUROBOROS_RUNTIME_MODE"] == (value if key == "OUROBOROS_RUNTIME_MODE" else "light")
    assert _slots(_task_start()) == ANTHROPIC


# --- C / 3A / 4A: the factory catalog over the effective configuration -------------------


@pytest.mark.parametrize("document", [None, {}], ids=["no-file", "raw-empty"])
def test_environment_credentials_reach_the_factory_catalog_with_or_without_a_document(install, monkeypatch, document):
    if document is not None:
        _write(install, document)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    settings = cfg.load_settings()
    assert _pool_targets(settings) == [OPENAI[0]]
    assert not install.exists() or json.loads(install.read_text()) == document, "a settings read writes nothing"


def test_a_never_configured_install_gets_the_wizards_auto_helpers_beside_unchanged_reviewers(install, monkeypatch):
    from ouroboros.configured_subagents import PRIMARY_RECOMMENDATION, SCOUT_RECOMMENDATION

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    items = _items(cfg.load_settings())
    actors = [row for row in items if not row.get("review_eligible")]
    reviewers = [row for row in items if row.get("review_eligible")]
    assert [(row["subagent_id"], row["route"]["target_id"], row.get("effort", ""), row["recommended_use"])
            for row in actors] == [
        ("primary-builder", ANTHROPIC[0], "", PRIMARY_RECOMMENDATION),
        ("fast-scout", ANTHROPIC[1], "", SCOUT_RECOMMENDATION)]
    # A direct provider without a mixed panel reviews on its Main once (owner Q5=A).
    assert [(row["route"]["target_id"], row["effort"]) for row in reviewers] == [(ANTHROPIC[0], "high")]
    # A Light that is Main adds no second helper; the reviewer seat stays one.
    _write(install, {"OUROBOROS_MODEL_LIGHT": ""})
    items = _items(cfg.load_settings())
    assert [row["subagent_id"] for row in items if not row.get("review_eligible")] == ["primary-builder"]
    assert sum(1 for row in items if row.get("review_eligible")) == 1


@pytest.mark.parametrize("stored", [
    {"enabled": True, "items": []},
    {"enabled": False, "items": [{"subagent_id": "mine", "recommended_use": "Owner row.",
                                  "route": {"kind": "api_model", "target_id": "anthropic::claude-opus-5"}}]},
    # The three Main rows the former factory saved stay three (owner Q5=A: saved rows untouched).
    {"enabled": True, "items": [{"subagent_id": f"review-{n}", "recommended_use": "Reviews.", "effort": "high",
                                 "route": {"kind": "api_model", "target_id": "anthropic::claude-opus-5"},
                                 "review_eligible": True, "minted_from": "factory_default"} for n in (1, 2, 3)]},
], ids=["explicit-empty", "delegation-off", "saved-former-three-main-rows"])
def test_an_owner_catalog_is_never_repopulated(install, monkeypatch, stored):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    _write(install, {SUBAGENTS: json.dumps(stored)})
    assert json.loads(cfg.load_settings()[SUBAGENTS]) == stored


def test_several_direct_providers_without_a_main_get_no_imaginary_router_panel(install, monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with caplog.at_level("WARNING"):
        settings = cfg.load_settings()
    assert _pool_targets(settings) == [] and not set(_pool_targets(settings)) & set(OPENROUTER_REVIEW_DEFAULTS["triad"])
    assert _items(settings) == [], "nothing is minted, so the install stays never-configured"
    assert "OUROBOROS_MODEL" in caplog.text


@pytest.mark.parametrize("env, main, pool", [
    ({"OPENAI_API_KEY": "k", "ANTHROPIC_API_KEY": "k", "OUROBOROS_MODEL": "anthropic::claude-opus-5"},
     "anthropic::claude-opus-5", ["anthropic::claude-opus-5"]),
    ({"OUROBOROS_MODEL": "claudexor::codex=gpt-5.6-sol"}, "claudexor::codex=gpt-5.6-sol",
     ["claudexor::codex=gpt-5.6-sol"]),  # no provider panel: Main once, never copies of it (owner Q5=A)
    ({"OPENROUTER_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}, SETTINGS_DEFAULTS["OUROBOROS_MODEL"],
     list(OPENROUTER_REVIEW_DEFAULTS["triad"])),
], ids=["mixed-direct-chosen-main", "managed-main", "router-plus-direct"])
def test_the_factory_panel_follows_the_chosen_main_without_replacing_it(install, monkeypatch, env, main, pool):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    settings = cfg.load_settings()
    assert _task_start().settings["OUROBOROS_MODEL"] == main
    assert _pool_targets(settings) == pool


def test_a_local_factory_row_runs_locally_for_review_and_delegation(install):
    from ouroboros.reviewer_slot_config import review_pool_slots
    from ouroboros.settings_integrity import task_settings_scope
    from ouroboros.subagent_runtime import resolve_configured_actor_dispatch, select_subagent_snapshot

    _write(install, {"USE_LOCAL_MAIN": True, "LOCAL_MODEL_SOURCE": "/models/m.gguf", "OUROBOROS_MODEL": "qwen-local"})
    snapshot = _task_start()
    with task_settings_scope(snapshot):
        settings = dict(snapshot.settings)
        assert _pool_targets(settings) == ["qwen-local (local)"]
        assert all(slot.use_local for slot in review_pool_slots(settings))
        row = select_subagent_snapshot(settings, subagent_id="review-1")[0]
        dispatch = resolve_configured_actor_dispatch(
            {"id": "child", "configured_subagent": row, "task_constraint": {}}, task_type="task")
    assert not dispatch.blocked and dispatch.lane.use_local_model is True and dispatch.lane.model == "qwen-local"


# --- D: the compatible endpoint is one pair --------------------------------------------


def test_a_dedicated_compatible_endpoint_never_takes_another_pairs_key(install, monkeypatch):
    from ouroboros.llm import LLMClient
    from ouroboros.provider_models import provider_has_credentials, provider_has_credentials_in_settings
    from ouroboros.subagent_runtime import resolve_configured_actor_dispatch

    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-official-openai-key")
    target = LLMClient()._resolve_remote_target("openai-compatible::served-model")
    assert (target["api_key"], target["base_url"]) == ("", "http://127.0.0.1:9/v1")
    assert provider_has_credentials("openai-compatible") is True
    assert provider_has_credentials_in_settings("openai-compatible", {"OPENAI_COMPATIBLE_BASE_URL": "http://x/v1"})
    dispatch = resolve_configured_actor_dispatch({"id": "child", "task_constraint": {}, "configured_subagent": {
        "schema": 1, "selected_subagent_id": "compat", "config_fingerprint": "f",
        "route": {"kind": "api_model", "target_id": "openai-compatible::served-model"}}}, task_type="task")
    assert not dispatch.blocked
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-compatible-key")
    assert LLMClient()._resolve_remote_target("openai-compatible::m")["api_key"] == "sk-compatible-key"


def test_the_legacy_openai_compatible_pair_still_routes(monkeypatch):
    from ouroboros.llm import LLMClient
    from ouroboros.provider_models import provider_has_credentials

    for key in ("OPENAI_COMPATIBLE_BASE_URL", "OPENAI_COMPATIBLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-legacy-pair-key")
    target = LLMClient()._resolve_remote_target("openai-compatible::m")
    assert (target["api_key"], target["base_url"]) == ("sk-legacy-pair-key", "https://legacy.example/v1")
    assert provider_has_credentials("openai-compatible") is True
    monkeypatch.delenv("OPENAI_BASE_URL")
    assert provider_has_credentials("openai-compatible") is False, "a key without an endpoint is no route"


# --- 1A / A: subscription presets ------------------------------------------------------
#
# A partial session refresh: Claude sessions take Opus 5.5, Codex sessions gpt-6.1-sol when discovered,
# else gpt-5.6-sol (the installed engine's catalog); a preset names exact discovery ids only.

INSTALLED_CODEX = ("gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5", "gpt-5.4", "gpt-5.4-mini")


def test_an_unused_advisory_model_never_refuses_the_preset():
    from ouroboros.subscription_install_presets import HarnessDiscovery, compile_install_preset

    for harness, model in (("codex", "gpt-5.6-sol"), ("claude", "claude-opus-5-5")):
        preset = compile_install_preset([HarnessDiscovery(harness, (model,))])
        assert preset.ok, preset.refusal
        assert preset.receipt["version"] == "4" and "advisory" not in preset.receipt["surfaces"]
        # A sole harness takes one triad seat (owner Q5=A), never three copies of one model.
        assert [(row["target_id"], row["effort"]) for row in preset.receipt["surfaces"]["triad"]] == [
            (f"{harness}={model}", "medium")]


def test_a_codex_catalog_without_the_newer_sol_still_gets_its_fresh_preset():
    from ouroboros.subscription_install_presets import HarnessDiscovery, compile_install_preset

    preset = compile_install_preset([HarnessDiscovery("codex", INSTALLED_CODEX)])
    assert preset.ok, preset.refusal
    surfaces = preset.receipt["surfaces"]
    assert {row["target_id"] for name in ("subagent", "triad", "scope") for row in surfaces[name]} == {
        "codex=gpt-5.6-sol"}


def test_a_missing_preferred_generation_is_a_typed_refusal_never_the_previous_one():
    from ouroboros.subscription_install_presets import HarnessDiscovery, compile_install_preset

    preset = compile_install_preset([HarnessDiscovery("claude", ("claude-opus-5", "claude-sonnet-5"))])
    assert not preset.ok and preset.settings_keys() == {}
    assert preset.refusal.code == "model_not_in_discovery" and preset.refusal.candidates == ("claude-opus-5-5",)


# --- 4A: the wizard and the Colab bootstrap ----------------------------------------------


def test_the_wizard_proposes_no_router_ids_for_several_direct_providers():
    from ouroboros.settings_setup_contract import build_setup_bootstrap

    defaults = build_setup_bootstrap({}, fresh_install=True)["modelDefaults"]
    assert defaults["direct-multi"]["main"] == "" and not any(defaults["direct-multi"].values())


def test_the_colab_document_proposes_for_unauthored_slots_and_keeps_saved_blanks():
    from ouroboros.colab_bootstrap import build_colab_settings

    settings = build_colab_settings({"OPENAI_API_KEY": "k"}, drive_document_present=False)
    assert tuple(settings[key] for key in SLOTS) == OPENAI
    assert [row["subagent_id"] for row in _items(settings) if not row.get("review_eligible")] == [
        "primary-builder", "fast-scout"]
    assert _pool_targets(settings) == [OPENAI[0]]
    kept = build_colab_settings({"OPENAI_API_KEY": "k"}, existing={"OUROBOROS_MODEL_LIGHT": ""})
    assert kept["OUROBOROS_MODEL_LIGHT"] == ""


# --- continuation: the contract gaps the first candidate left open -------------------------
#
# D: the legacy compatible endpoint is the legacy PAIR — a dedicated key never rides it.

LEGACY_PAIR = {"OPENAI_BASE_URL": "https://legacy.example/v1", "OPENAI_API_KEY": "sk-legacy-official-key"}
DEDICATED_KEY = "sk-dedicated-compatible-key"


def _catalog_fetch(settings, monkeypatch):
    """The ``(api_key, base_url)`` the model-catalog probe of the compatible lane would send."""
    import asyncio

    from ouroboros.gateway import models as gw_models

    sent = []

    async def fetch(_client, provider, _label, api_key, base_url, *, record_image_input=False):
        assert record_image_input is False
        sent.append((provider, api_key, base_url))
        return []

    monkeypatch.setattr(gw_models, "_fetch_openai_compatible_model_catalog", fetch)
    for name, spec in gw_models._provider_specs(dict(settings)):
        if name == "openai-compatible":
            asyncio.run(spec(None))
    return [(key, base) for provider, key, base in sent if provider == "openai-compatible"]


@pytest.mark.parametrize("source", ["environment", "document"])
def test_a_dedicated_key_never_rides_the_legacy_endpoint(install, monkeypatch, source):
    """Both keys beside the legacy base URL: the legacy endpoint takes the legacy key — in the
    task-start capture, the ambient resolver, the explicit-mapping resolver, readiness and the
    model-catalog probe alike."""
    from ouroboros.llm import LLMClient
    from ouroboros.provider_models import provider_has_credentials, provider_has_credentials_in_settings
    from ouroboros.settings_integrity import task_settings_scope

    pair = {**LEGACY_PAIR, "OPENAI_COMPATIBLE_API_KEY": DEDICATED_KEY}
    if source == "environment":
        for key, value in pair.items():
            monkeypatch.setenv(key, value)
    else:
        _write(install, pair)
    snapshot = _task_start()
    legacy = (LEGACY_PAIR["OPENAI_API_KEY"], LEGACY_PAIR["OPENAI_BASE_URL"])
    with task_settings_scope(snapshot):
        target = LLMClient()._resolve_remote_target("openai-compatible::served-model")
        assert (target["api_key"], target["base_url"]) == legacy
        assert provider_has_credentials("openai-compatible") is True
    target = LLMClient()._resolve_remote_target("openai-compatible::served-model", settings=dict(snapshot.settings))
    assert (target["api_key"], target["base_url"]) == legacy
    assert provider_has_credentials_in_settings("openai-compatible", dict(snapshot.settings)) is True
    assert _catalog_fetch(snapshot.settings, monkeypatch) == [legacy]


def test_a_dedicated_pair_keeps_its_blank_key_and_a_key_alone_is_no_endpoint(install, monkeypatch):
    from ouroboros.llm import LLMClient
    from ouroboros.provider_models import provider_has_credentials, provider_has_credentials_in_settings

    dedicated = {**LEGACY_PAIR, "OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:9/v1", "OPENAI_COMPATIBLE_API_KEY": ""}
    target = LLMClient()._resolve_remote_target("openai-compatible::m", settings=dedicated)
    assert (target["api_key"], target["base_url"]) == ("", "http://127.0.0.1:9/v1"), "blank = no auth, no borrowing"
    assert _catalog_fetch(dedicated, monkeypatch) == [("", "http://127.0.0.1:9/v1")]
    key_only = {"OPENAI_COMPATIBLE_API_KEY": DEDICATED_KEY, "OPENAI_API_KEY": "sk-official-only"}
    assert provider_has_credentials_in_settings("openai-compatible", key_only) is False
    for key, value in key_only.items():
        monkeypatch.setenv(key, value)
    assert provider_has_credentials("openai-compatible") is False
    assert _catalog_fetch(key_only, monkeypatch) == []


# 4A: an unrelated router key does not take the factory panel away from the chosen Main.


def _review_rows(settings):
    return [row for row in _items(settings) if row.get("review_eligible")]


def test_a_router_key_beside_a_chosen_direct_main_keeps_the_mains_provider_panel(install, monkeypatch):
    from ouroboros.provider_models import compute_direct_review_models_fallback, provider_for_model

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-unrelated-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-present-key")
    _write(install, {"OUROBOROS_MODEL": "anthropic::claude-opus-5"})
    settings = cfg.load_settings()
    panel = compute_direct_review_models_fallback("anthropic", "anthropic::claude-opus-5",
                                                  settings["OUROBOROS_MODEL_LIGHT"])
    assert _pool_targets(settings) == panel and {provider_for_model(model) for model in panel} == {"anthropic"}
    assert [row["effort"] for row in _review_rows(settings)] == ["high"]
    assert _task_start().settings["OUROBOROS_MODEL"] == "anthropic::claude-opus-5"


def test_a_router_key_beside_a_local_main_keeps_the_local_rows_local_for_review_and_delegation(install, monkeypatch):
    from ouroboros.reviewer_slot_config import review_pool_slots
    from ouroboros.settings_integrity import task_settings_scope
    from ouroboros.subagent_runtime import resolve_configured_actor_dispatch, select_subagent_snapshot

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-unrelated-key")
    _write(install, {"USE_LOCAL_MAIN": True, "LOCAL_MODEL_SOURCE": "/models/m.gguf", "OUROBOROS_MODEL": "qwen-local"})
    snapshot = _task_start()
    with task_settings_scope(snapshot):
        settings = dict(snapshot.settings)
        assert _pool_targets(settings) == ["qwen-local (local)"]
        assert all(slot.use_local for slot in review_pool_slots(settings))
        row = select_subagent_snapshot(settings, subagent_id="review-1")[0]
        dispatch = resolve_configured_actor_dispatch(
            {"id": "child", "configured_subagent": row, "task_constraint": {}}, task_type="task")
    assert not dispatch.blocked and dispatch.lane.use_local_model is True and dispatch.lane.model == "qwen-local"


def test_a_router_key_beside_a_managed_main_keeps_its_account_pin(install, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-unrelated-key")
    _write(install, {"OUROBOROS_MODEL": "claudexor::codex=gpt-5.6-sol",
                     "OUROBOROS_MODEL_ACCOUNTS": {"main": "account-a"}})
    rows = _review_rows(cfg.load_settings())
    assert [row["route"] for row in rows] == [{"kind": "api_model", "target_id": "claudexor::codex=gpt-5.6-sol",
                                               "credential_profile_id": "account-a"}]
    assert [row["effort"] for row in rows] == ["high"]


def test_a_router_main_keeps_the_diverse_router_panel(install, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-present-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-present-key")
    _write(install, {"OUROBOROS_MODEL": "openai/gpt-5.6"})
    assert _pool_targets(cfg.load_settings()) == list(OPENROUTER_REVIEW_DEFAULTS["triad"])


@pytest.mark.parametrize("env, main", [
    ({"ANTHROPIC_API_KEY": "sk-ant-present-key"}, "openai::gpt-5.6"),
    ({"OPENROUTER_API_KEY": "sk-or-present-key"}, "anthropic::claude-opus-5"),
], ids=["exclusive-provider", "router-key"])
def test_an_explicit_main_without_access_is_named_never_replaced_by_another_panel(install, monkeypatch, caplog,
                                                                                  env, main):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    _write(install, {"OUROBOROS_MODEL": main})
    with caplog.at_level("WARNING"):
        settings = cfg.load_settings()
    assert _pool_targets(settings) == [] and _items(settings) == []
    assert "OUROBOROS_MODEL" in caplog.text
    assert _task_start().settings["OUROBOROS_MODEL"] == main


# 4A: the compiler's other consumers — the wizard and the Settings draft — never build a helper
# from Light for a Main without access; a session keeps its Light companion.

UNREACHABLE_MAIN = {"ANTHROPIC_API_KEY": "sk-ant-present-key", "OUROBOROS_MODEL": "openai::gpt-5.6",
                    "OUROBOROS_MODEL_LIGHT": ANTHROPIC[1]}


def test_no_light_helper_stands_in_for_a_main_without_access_in_the_wizard_or_the_settings_draft(install, monkeypatch):
    from ouroboros.gateway import onboarding as gw_onboarding
    from ouroboros.gateway import settings as settings_mod

    app = _settings_app(monkeypatch)
    app.router.routes += [Route("/api/settings", endpoint=settings_mod.api_settings_get, methods=["GET"]),
                          Route("/api/onboarding/subagents/preview",
                                endpoint=gw_onboarding.api_onboarding_subagents_preview, methods=["POST"])]
    client = TestClient(app)
    preview = client.post("/api/onboarding/subagents/preview",
                          json={**UNREACHABLE_MAIN, "subscriptionsConnected": False})
    assert preview.status_code == 503 and preview.json()["code"] == "main_without_access", preview.text
    assert "OUROBOROS_MODEL" in preview.json()["detail"] and preview.json()["can_skip"] is True
    _write(install, UNREACHABLE_MAIN)
    draft = client.get("/api/settings").json()["_meta"]["available_subagents"]
    assert draft["candidate"] is None and [row["code"] for row in draft["diagnostics"]] == ["main_without_access"]


def test_a_session_keeps_its_light_companion_and_a_reachable_main_its_helpers():
    from ouroboros.configured_subagents import configured_subagents_dict, resolve_settings_subagent_candidate
    from ouroboros.subscription_install_presets import HarnessDiscovery, compile_install_preset

    preset = compile_install_preset([HarnessDiscovery("codex", ("gpt-5.6-sol",))], settings=UNREACHABLE_MAIN)
    assert preset.ok, preset.refusal
    assert [row["route"]["target_id"] for row in preset.receipt["available_subagents"]["items"]
            if not row.get("review_eligible")] == ["codex=gpt-5.6-sol", ANTHROPIC[1]]
    reachable = {**SETTINGS_DEFAULTS, **UNREACHABLE_MAIN, "OUROBOROS_MODEL": ANTHROPIC[0]}
    resolution, diagnostics = resolve_settings_subagent_candidate(reachable)
    assert [row["route"]["target_id"] for row in configured_subagents_dict(resolution.config)["items"]] == [
        ANTHROPIC[0], ANTHROPIC[1]] and not diagnostics
    # A saved blank Main runs as the shipped Main, beside a legacy session too (``model_slots._main_model``).
    blank = {**SETTINGS_DEFAULTS, "OPENROUTER_API_KEY": "sk-or-present-key", "OUROBOROS_MODEL": "",
             "OUROBOROS_MODEL_LIGHT": "", "OUROBOROS_SUBAGENT_HARNESS": "codex=gpt-5.6-sol"}
    resolution, _diagnostics = resolve_settings_subagent_candidate(blank)
    assert [row.route.target_id for row in resolution.config.items] == ["codex=gpt-5.6-sol", SHIPPED[0]]


# 2A: a saved blank Main is the owner's value — kept by GET, unrelated saves and reloads.


def test_a_saved_blank_main_survives_get_unrelated_saves_and_reload(install, monkeypatch):
    from ouroboros.gateway.owner_settings import _owner_update_settings
    from ouroboros.model_slots import _main_model
    from ouroboros.settings_integrity import task_settings_scope

    monkeypatch.setenv("OUROBOROS_MODEL", "anthropic::claude-operator-main")  # the saved blank beats it
    _write(install, {"ANTHROPIC_API_KEY": "sk-ant-present-key", "OUROBOROS_MODEL": ""})
    client = _settings_client(monkeypatch)
    shown = client.get("/api/settings").json()
    assert shown["OUROBOROS_MODEL"] == ""
    saved = client.post("/api/settings", json={**{key: shown[key] for key in SLOTS}, "TOTAL_BUDGET": "25"})
    assert saved.status_code == 200, saved.text
    assert json.loads(install.read_text(encoding="utf-8"))["OUROBOROS_MODEL"] == ""
    _owner_update_settings(lambda current: {**current, "OUROBOROS_UI_LANGUAGE": "ru"})
    assert json.loads(install.read_text(encoding="utf-8"))["OUROBOROS_MODEL"] == ""
    snapshot = _task_start()
    assert snapshot.settings["OUROBOROS_MODEL"] == ""
    with task_settings_scope(snapshot):
        # Read through the declared interface (``model_slots._main_model``), as the runtime reads it.
        assert _main_model() == SETTINGS_DEFAULTS["OUROBOROS_MODEL"]


# 4A: a local-only setup proposes Light/Fallbacks that follow the local Main, everywhere.


LOCAL_ALL = {"LOCAL_MODEL_SOURCE": "/models/m.gguf", "USE_LOCAL_MAIN": True, "USE_LOCAL_LIGHT": True,
             "USE_LOCAL_CONSCIOUSNESS": True, "USE_LOCAL_FALLBACK": True}


def test_a_local_only_setup_proposes_blank_light_and_fallbacks_at_boot_wizard_and_colab(install, monkeypatch):
    from ouroboros.colab_bootstrap import build_colab_settings
    from ouroboros.settings_setup_contract import build_setup_bootstrap, derive_provider_profile

    _write(install, LOCAL_ALL)
    boot = _slots(_task_start())
    assert derive_provider_profile(LOCAL_ALL) == "local"
    wizard = build_setup_bootstrap(LOCAL_ALL, fresh_install=True)["modelDefaults"]["local"]
    colab = build_colab_settings({}, existing=dict(LOCAL_ALL))
    assert boot == (SETTINGS_DEFAULTS["OUROBOROS_MODEL"], "", "")
    assert (wizard["main"], wizard["light"], wizard["fallback"]) == boot
    assert tuple(colab[key] for key in SLOTS) == boot
    # A saved choice stays; a router key makes the shipped Light reachable again (mixed setup).
    _write(install, {**LOCAL_ALL, "OUROBOROS_MODEL_LIGHT": "my-light (local)"})
    assert _task_start().settings["OUROBOROS_MODEL_LIGHT"] == "my-light (local)"
    _write(install, LOCAL_ALL)
    for key in SLOTS:  # a new process: the last read's projection is not this document's (conftest restores)
        os.environ.pop(key, None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-present-key")
    assert _slots(_task_start()) == SHIPPED


# Parent relay: the second read of a running process — load → project → change → reload.


def test_a_saved_blank_light_and_fallbacks_beat_the_processs_earlier_projection(install):
    """The task start projects this read's slots into the process environment; a later saved
    blank is the owner's value and wins over that projection and over an operator variable alike."""
    from ouroboros.gateway.owner_settings import _owner_update_settings

    _write(install, {"OPENROUTER_API_KEY": "sk-or-present-key"})
    assert _slots(_task_start()) == SHIPPED
    _owner_update_settings(lambda current: {**current, "OUROBOROS_MODEL_LIGHT": "", "OUROBOROS_MODEL_FALLBACKS": ""},
                           allow_safety_lowering=True)
    stored = json.loads(install.read_text(encoding="utf-8"))
    assert (stored["OUROBOROS_MODEL_LIGHT"], stored["OUROBOROS_MODEL_FALLBACKS"]) == ("", "")
    assert _slots(_task_start()) == (SHIPPED[0], "", "")
    assert _slots(_task_start()) == (SHIPPED[0], "", ""), "and the next projection keeps them"
    os.environ["OUROBOROS_MODEL_LIGHT"] = "operator/light"  # an operator variable (conftest restores the env)
    assert _slots(_task_start()) == (SHIPPED[0], "", "")


def test_unauthored_slots_follow_an_access_change_after_the_process_projected_them(install):
    """A projected proposal is this process's echo, not a choice: when an owner write swaps the
    access, the absent slots move to the new access's proposals on the next task start. An
    operator's variable for an absent slot still counts."""
    from ouroboros.gateway.owner_settings import _owner_update_settings

    _write(install, {"ANTHROPIC_API_KEY": "sk-ant-present-key"})
    assert _slots(_task_start()) == ANTHROPIC
    _owner_update_settings(lambda current: {**current, "OPENROUTER_API_KEY": "sk-or-present-key"},
                           allow_safety_lowering=True)
    assert not set(SLOTS) & set(json.loads(install.read_text(encoding="utf-8")))
    assert _slots(_task_start()) == SHIPPED
    assert _slots(_task_start()) == SHIPPED
    os.environ["OUROBOROS_MODEL"] = "anthropic::claude-operator-main"  # an operator variable (conftest restores)
    assert _task_start().settings["OUROBOROS_MODEL"] == "anthropic::claude-operator-main"


def test_a_projected_factory_catalog_follows_an_access_change_and_an_operator_catalog_still_wins(install, monkeypatch):
    """The task start projects the never-configured factory catalog into the process environment (and so
    into every child); that echo is no operator catalog, so an owner access change moves the panel on the
    next task start. An operator's catalog in the environment still wins, and a saved empty one stays."""
    from ouroboros.gateway.owner_settings import _owner_update_settings, fields_on_saved_document

    _write(install, {"ANTHROPIC_API_KEY": "sk-ant-present-key"})
    assert _pool_targets(_task_start().settings) == [ANTHROPIC[0]]
    assert os.environ.get(SUBAGENTS), "projected (conftest restores the environment)"
    _owner_update_settings(fields_on_saved_document({"OPENROUTER_API_KEY": "sk-or-present-key"}),
                           allow_safety_lowering=True)
    assert _pool_targets(_task_start().settings) == list(OPENROUTER_REVIEW_DEFAULTS["triad"])
    _write(install, {"ANTHROPIC_API_KEY": "sk-ant-present-key", SUBAGENTS: json.dumps({"enabled": True, "items": []})})
    assert _items(_task_start().settings) == [] and _items(_task_start().settings) == []
    operator = json.dumps({"enabled": True, "items": [{
        "subagent_id": "operator", "recommended_use": "Operator row.", "review_eligible": True, "effort": "high",
        "route": {"kind": "api_model", "target_id": "anthropic::claude-operator"}}]})
    _write(install, {"ANTHROPIC_API_KEY": "sk-ant-present-key"})
    monkeypatch.setenv(SUBAGENTS, operator)
    for _task in range(2):
        assert _pool_targets(_task_start().settings) == ["anthropic::claude-operator"]


def test_the_wizards_scout_keeps_its_own_lights_binding(install, monkeypatch):
    """The Auto helpers each run in their own role's whole binding: a distinct managed Light keeps
    its own account pin and processing, never Main's; a local Light keeps its local label."""
    _write(install, {"OUROBOROS_MODEL": "claudexor::codex=gpt-5.6-sol",
                     "OUROBOROS_MODEL_LIGHT": "claudexor::claude=claude-opus-5-5",
                     "OUROBOROS_MODEL_ACCOUNTS": {"main": "account-a", "light": "account-b"},
                     "OUROBOROS_MODEL_PROCESSING_PREFERENCES": {"light": "fast"}})
    actors = {row["subagent_id"]: row for row in _items(cfg.load_settings()) if not row.get("review_eligible")}
    assert actors["primary-builder"]["route"] == {"kind": "api_model", "target_id": "claudexor::codex=gpt-5.6-sol",
                                                  "credential_profile_id": "account-a"}
    assert actors["fast-scout"]["route"] == {"kind": "api_model", "target_id": "claudexor::claude=claude-opus-5-5",
                                             "credential_profile_id": "account-b"}
    assert actors["fast-scout"].get("processing_preference") == "fast"
    assert not actors["primary-builder"].get("processing_preference")
    m._MIGRATIONS_SEEN.clear()
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-present-key")
    _write(install, {"OUROBOROS_MODEL_LIGHT": "qwen-light", "USE_LOCAL_LIGHT": True, "LOCAL_MODEL_SOURCE": "/m.gguf"})
    actors = {row["subagent_id"]: row for row in _items(cfg.load_settings()) if not row.get("review_eligible")}
    assert actors["fast-scout"]["route"] == {"kind": "api_model", "target_id": "qwen-light (local)"}
    assert actors["primary-builder"]["route"] == {"kind": "api_model", "target_id": SETTINGS_DEFAULTS["OUROBOROS_MODEL"]}


# 3A: rows the same in EVERY setting are not duplicated — the whole binding, not the model name —
# and the wizard's preview builds the very helpers the headless catalog does.

MANAGED = "claudexor::codex=gpt-5.6-sol"


def _helpers(items):
    """Each helper's binding; a managed route spells "no pin" as an empty pin field or none."""
    return {row["subagent_id"]: {"route": {key: value for key, value in row["route"].items() if value},
                                 **({"processing_preference": row["processing_preference"]}
                                    if row.get("processing_preference") else {})}
            for row in items if not row.get("review_eligible")}


def _on(pin="", processing=""):
    return {"route": {"kind": "api_model", "target_id": MANAGED, **({"credential_profile_id": pin} if pin else {})},
            **({"processing_preference": processing} if processing else {})}


@pytest.mark.parametrize("difference, expected", [
    ({"OUROBOROS_MODEL_ACCOUNTS": {"main": "account-a", "light": "account-b"}},
     {"primary-builder": _on("account-a"), "fast-scout": _on("account-b")}),
    ({"OUROBOROS_MODEL_PROCESSING_PREFERENCES": {"light": "fast"}},
     {"primary-builder": _on(), "fast-scout": _on(processing="fast")}),
    ({"OUROBOROS_MODEL_ACCOUNTS": {"main": "account-a", "light": "account-a"}},
     {"primary-builder": _on("account-a")}),
], ids=["other-account", "other-processing", "same-binding"])
def test_a_light_on_mains_model_is_a_second_helper_only_when_its_binding_differs(install, difference, expected):
    from ouroboros.gateway import onboarding as gw_onboarding

    draft = {"OPENROUTER_API_KEY": "sk-or-present-key", "OUROBOROS_MODEL": MANAGED,
             "OUROBOROS_MODEL_LIGHT": MANAGED, **difference}
    _write(install, draft)
    assert _helpers(_items(cfg.load_settings())) == expected, "the headless never-configured catalog"
    app = Starlette(routes=[Route("/api/onboarding/subagents/preview",
                                  endpoint=gw_onboarding.api_onboarding_subagents_preview, methods=["POST"])])
    preview = TestClient(app).post("/api/onboarding/subagents/preview",
                                   json={**draft, "subscriptionsConnected": False})
    assert preview.status_code == 200, preview.text
    assert _helpers(preview.json()["available_subagents"]["items"]) == expected, "the wizard's preset"


def test_the_settings_draft_keeps_the_compiled_helpers_without_an_unpinned_twin():
    """The Settings draft for a configuration without a catalog (``GET /api/settings``) composes
    the same helpers; the legacy Light stand-in adds no unpinned twin of a row that runs it."""
    from ouroboros.configured_subagents import configured_subagents_dict, resolve_settings_subagent_candidate

    settings = {**SETTINGS_DEFAULTS, "OPENROUTER_API_KEY": "sk-or-present-key", "OUROBOROS_MODEL": MANAGED,
                "OUROBOROS_MODEL_LIGHT": MANAGED, "OUROBOROS_MODEL_ACCOUNTS": {"main": "account-a", "light": "account-b"},
                "OUROBOROS_MODEL_PROCESSING_PREFERENCES": {"light": "fast"}}
    resolution, _diagnostics = resolve_settings_subagent_candidate(settings)
    assert _helpers(configured_subagents_dict(resolution.config)["items"]) == {
        "primary-builder": _on("account-a"), "fast-scout": _on("account-b", "fast")}
