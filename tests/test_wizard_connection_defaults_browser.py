"""The setup wizard proposes, previews and saves models on a key the install has (#2969).

The page runs on the real bootstrap (``GET /api/onboarding/bootstrap``, the derivation the served page embeds),
the real preview and the real ``/api/onboarding/complete`` over a temp settings file. Keys are an operator's
environment (``OUROBOROS_CONNECTIONS``, ``OPENAI_API_KEY``; the wizard adds named keys for compatible servers only)
or a stored document; they are fixtures and nothing is sent anywhere. Main, Light and Fallbacks take the built-in
key when it serves, else the only named one; with several named keys and no built-in one the owner picks Main's,
and only the lanes the wizard proposed follow it. A saved blank lane stays blank whatever the environment says.
"""
from __future__ import annotations

import json

import pytest

from ouroboros.provider_models import CONNECTIONS_KEY, DIRECT_PROVIDER_DEFAULTS
from tests import test_subscription_setup_browser as setup_browser
from tests.test_connection_aware_defaults import HOME, WORK

pytestmark = [pytest.mark.ui_browser]
subscription_ui = setup_browser.subscription_ui
onboarding = setup_browser.onboarding
capture = setup_browser.capture
QUESTION = "Main is not set yet. Choose the key it uses: OpenAI · work or OpenAI · home."
SLOTS = ("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS")
CORP = {"kind": "gigachat", "name": "corp", "credentials": "fixture-gigachat-credentials"}


def _open(ui, onboarding, monkeypatch, document=None, *, subscriptions=False, **environ):
    """The wizard on the real bootstrap of an install whose environment holds ``environ`` (fresh unless a
    ``document`` is stored); every preview answer is kept in ``ui["previews"]`` as ``(status, body)``. API keys only,
    unless ``subscriptions``: then the fixture's Codex account is connected and the engine snapshot serves it."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import config as cfg
    from ouroboros.gateway import onboarding_host

    for key in (*SLOTS, CONNECTIONS_KEY, "OUROBOROS_MODEL_VISION", "OUROBOROS_MODEL_SAFETY", "OUROBOROS_SUBAGENTS",
                "GIGACHAT_CREDENTIALS", "GIGACHAT_USER", "GIGACHAT_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    for key, value in environ.items():
        monkeypatch.setenv(key, value)
    if document is not None:
        onboarding.settings_path.write_text(json.dumps(document), encoding="utf-8")
    app = Starlette(routes=[Route("/api/onboarding/bootstrap", onboarding_host.api_onboarding_bootstrap)])
    app.state.drive_root = cfg.DATA_DIR
    with TestClient(app) as client:
        bootstrap = client.get("/api/onboarding/bootstrap").json()["bootstrap"]
    assert bootstrap["freshInstall"] is (document is None)
    ui["bootstrap"].clear()
    ui["bootstrap"].update(bootstrap)
    if subscriptions:
        from tests.test_onboarding_complete_endpoint import LIVE_SNAPSHOT
        onboarding.calls["snapshot_payload"] = {**LIVE_SNAPSHOT, "model_catalog": ui["fixture"]["catalog"]["items"]}
    else:
        ui["fixture"]["status"]["profiles"]["profiles"] = []  # API keys only: no subscription is connected
    ui["backend"].update(client=onboarding.client, preview_client=onboarding.client)
    ui["previews"] = previews = []
    page = ui["page"]
    page.on("response", lambda response: previews.append((response.status, response.json()))
            if response.url.endswith("/api/onboarding/subagents/preview") else None)
    page.goto(ui["url"] + "/onboarding")
    return page


def _models(page):
    page.click("#next-btn")
    page.wait_for_selector('[data-model-role="main"]')
    return page.locator('[data-model-role-group="main"]')


def _row(page, slot):
    return page.locator(f'[data-model-role-group="{slot}"]')


def _finish(ui, page):
    for _ in range(3):
        page.click("#next-btn")
    page.wait_for_selector(".summary-card")
    page.click("#next-btn")
    page.wait_for_url(ui["url"] + "/")
    return [body for path, body in ui["posts"] if path == "/api/onboarding/complete"][-1]


def _on(key, value):
    return bool(value) and all(entry.strip().startswith(f"{key}::") for entry in value.split(",") if entry.strip())


def _reviewers(ui, *, every=True):
    """The reviewers the last preview showed; ``every``: each preview of the draft was answered."""
    answers = ui["previews"] if every else ui["previews"][-1:]
    assert answers and all(status == 200 for status, _body in answers), ui["previews"]
    return [row["route"]["target_id"] for row in ui["previews"][-1][1]["available_subagents"]["items"]
            if row.get("review_eligible")]


def _runs_on(key, reviewers):
    """What the install runs after completion: its slots and reviewers on ``key``, whose credential resolves."""
    from ouroboros import config as cfg
    from ouroboros.llm_routing import resolve_target

    loaded = cfg.load_settings()
    marked = [row["route"]["target_id"] for row in json.loads(loaded["OUROBOROS_SUBAGENTS"])["items"]
              if row.get("review_eligible")]
    assert marked == reviewers and all(_on(key, target) for target in marked), (marked, reviewers)
    assert all({"api_key", "credentials"} & {k for k, v in resolve_target(target, loaded).items() if v} for target in marked)
    return loaded


@pytest.mark.parametrize(("environ", "document", "key"), [
    ({CONNECTIONS_KEY: json.dumps([WORK])}, None, "openai@work"),  # the reviewed case: one named key, no built-in
    ({"OPENAI_API_KEY": "fixture-built-in-key"}, None, "openai"),
    ({CONNECTIONS_KEY: json.dumps([WORK]), "OPENAI_API_KEY": "fixture-built-in-key"}, None, "openai"),  # built-in first
    ({}, {CONNECTIONS_KEY: [WORK]}, "openai@work"),  # control: the same key stored, not in the environment
    ({CONNECTIONS_KEY: json.dumps([CORP])}, None, "gigachat@corp"),  # a direct provider of its own defaults
    ({"GIGACHAT_CREDENTIALS": "fixture-gigachat-authorization"}, None, "gigachat"),
    ({"GIGACHAT_CREDENTIALS": "fixture-gigachat-authorization", CONNECTIONS_KEY: json.dumps([CORP])}, None, "gigachat"),
], ids=["env-named", "env-built-in", "env-built-in-first", "stored-named", "env-named-gigachat", "env-built-in-gigachat",
        "env-built-in-gigachat-first"])
def test_the_wizard_proposes_previews_and_saves_on_the_key_the_install_has(subscription_ui, onboarding, monkeypatch,
                                                                          environ, document, key):
    from playwright.sync_api import expect

    ui = subscription_ui
    page = _open(ui, onboarding, monkeypatch, document, **environ)
    main = _models(page)
    expect(main.locator("select[data-model-role-source]")).to_have_value(f"api:{key}")
    for slot in ("main", "light", "fallback"):
        assert "no key" not in _row(page, slot).locator("[data-model-role-status]").first.inner_text()
    expect(page.locator("#next-btn")).to_be_enabled()
    capture(page, f"wizard-connection-defaults-{key.replace('@', '-')}-{'stored' if document else 'environment'}")
    body = _finish(ui, page)
    reviewers = _reviewers(ui)
    assert reviewers, ui["previews"][-1]
    assert all(_on(key, body[slot]) for slot in SLOTS), {slot: body[slot] for slot in SLOTS}
    kind = key.split("@")[0]
    assert onboarding.saved()["OUROBOROS_MODEL"] == DIRECT_PROVIDER_DEFAULTS[kind]["main"].replace(f"{kind}::", f"{key}::")
    loaded = _runs_on(key, reviewers)
    assert all(_on(key, loaded[slot]) for slot in SLOTS), {slot: loaded.get(slot) for slot in SLOTS}


def test_a_saved_blank_light_stays_blank_over_the_environment(subscription_ui, onboarding, monkeypatch):
    ui = subscription_ui
    stored = {"OPENAI_API_KEY": "fixture-stored-key-0001", "OUROBOROS_MODEL_LIGHT": ""}  # Light follows Main
    page = _open(ui, onboarding, monkeypatch, stored, OUROBOROS_MODEL_LIGHT="openai::env-light-model")
    _models(page)
    assert _row(page, "light").locator("[data-model-role-model]").input_value() == ""
    body = _finish(ui, page)
    assert body["OUROBOROS_MODEL_LIGHT"] == "" and _reviewers(ui)
    assert onboarding.saved()["OUROBOROS_MODEL_LIGHT"] == ""
    from ouroboros import config as cfg

    assert cfg.load_settings()["OUROBOROS_MODEL_LIGHT"] == ""


@pytest.mark.parametrize("stored", [False, True], ids=["environment", "stored"])
def test_several_named_keys_wait_for_the_owner_and_only_our_proposals_follow(subscription_ui, onboarding, monkeypatch,
                                                                            stored):
    """Neither the document nor a variable sets a model: every slot is the wizard's, also on an existing install."""
    from playwright.sync_api import expect

    ui = subscription_ui
    keys = {CONNECTIONS_KEY: [WORK, HOME]}
    page = (_open(ui, onboarding, monkeypatch, keys) if stored
            else _open(ui, onboarding, monkeypatch, **{CONNECTIONS_KEY: json.dumps(keys[CONNECTIONS_KEY])}))
    main = _models(page)
    status = main.locator("[data-model-role-status]")
    expect(status).to_have_text(QUESTION)
    expect(status).to_have_attribute("data-tone", "warn")
    expect(page.locator("#next-btn")).to_be_disabled()  # nothing is guessed: Main has no key yet
    for width in (1360, 390):
        page.set_viewport_size({"width": width, "height": 900})
        for scheme in ("light", "dark"):
            page.emulate_media(color_scheme=scheme)
            main.scroll_into_view_if_needed()
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            capture(page, f"wizard-main-connection-choice-{'stored' if stored else 'env'}-{scheme}-{width}")
    page.set_viewport_size({"width": 1360, "height": 900})
    page.emulate_media(color_scheme="light")
    light = _row(page, "light").locator("[data-model-role-model]").input_value()

    # The owner's own Fallback stays theirs; Light, still the wizard's proposal, follows Main's key.
    fallback = _row(page, "fallback")
    fallback.locator("select[data-model-role-source]").first.select_option("api:openai@work")
    fallback.locator("[data-model-role-model]").first.fill("owner-fallback")
    main.locator("select[data-model-role-source]").select_option("api:openai@home")
    main.locator("[data-model-role-model]").fill("gpt-5.6-terra")
    main.locator("h4").click()
    expect(status).not_to_have_text(QUESTION)
    expect(page.locator("#next-btn")).to_be_enabled()
    expected = {"OUROBOROS_MODEL": "openai@home::gpt-5.6-terra", "OUROBOROS_MODEL_LIGHT": f"openai@home::{light}",
                "OUROBOROS_MODEL_FALLBACKS": "openai@work::owner-fallback"}
    body = _finish(ui, page)
    assert {slot: body[slot] for slot in SLOTS} == expected
    assert {slot: onboarding.saved()[slot] for slot in SLOTS} == expected
    # Before the choice the preview has no key to mint reviewers on (503 no_available_subagents); after it, it does.
    _runs_on("openai@home", _reviewers(ui, every=False))


@pytest.mark.parametrize("case", ["environment-choice", "saved-all-blank"])
def test_an_operators_and_a_saved_choice_stay_while_the_wizards_own_slots_follow(subscription_ui, onboarding,
                                                                                monkeypatch, case):
    """A model an operator's variable sets and lanes the document saves blank are choices: Finish saves them as they
    are, while the slots neither sets still take the wizard's proposals."""
    ui = subscription_ui
    if case == "environment-choice":
        page = _open(ui, onboarding, monkeypatch, OPENAI_API_KEY="fixture-built-in-key",
                     OUROBOROS_MODEL_LIGHT="openai::owner-env-light")
        main = _models(page)
        expected = {"OUROBOROS_MODEL": DIRECT_PROVIDER_DEFAULTS["openai"]["main"],
                    "OUROBOROS_MODEL_LIGHT": "openai::owner-env-light",
                    "OUROBOROS_MODEL_FALLBACKS": DIRECT_PROVIDER_DEFAULTS["openai"]["fallback"]}
    else:
        blanks = {key: "" for key in (*SLOTS, "OUROBOROS_MODEL_VISION", "OUROBOROS_MODEL_SAFETY",
                                      "OUROBOROS_MODEL_CONSCIOUSNESS")}
        page = _open(ui, onboarding, monkeypatch, {"OPENAI_API_KEY": "fixture-stored-key-0001", **blanks})
        main = _models(page)
        assert main.locator("[data-model-role-model]").input_value() == ""  # Main is asked for, nothing is put there
        main.locator("select[data-model-role-source]").select_option("api:openai")
        main.locator("[data-model-role-model]").fill("gpt-5.6-terra")
        main.locator("h4").click()
        expected = {"OUROBOROS_MODEL": "openai::gpt-5.6-terra", "OUROBOROS_MODEL_LIGHT": "",
                    "OUROBOROS_MODEL_FALLBACKS": ""}
    capture(page, f"wizard-kept-{case}")
    body = _finish(ui, page)
    assert {slot: body[slot] for slot in SLOTS} == expected
    assert {slot: onboarding.saved()[slot] for slot in SLOTS} == expected
    if case == "saved-all-blank":
        assert onboarding.saved()["OUROBOROS_MODEL_VISION"] == ""


REVIEWER = {"subagent_id": "owner-reader", "recommended_use": "Owner's own reader.", "effort": "low",
            "review_eligible": True, "delivery": "packet", "route": {"kind": "api_model", "target_id": "openai::gpt-5.6-luna"}}
HELPER = {"subagent_id": "owner-helper", "recommended_use": "Owner's helper, turned off.", "enabled": False,
          "route": {"kind": "api_model", "target_id": "openai::gpt-5.6-sol"}}


@pytest.mark.parametrize("saved", [
    {"enabled": True, "items": [REVIEWER, HELPER]},  # a custom roster, row by row
    {"enabled": True, "items": []},  # explicitly empty
    {"enabled": False, "items": [REVIEWER]},  # Off
    None,  # never configured: the wizard's to generate
], ids=["custom", "empty", "off", "automatic"])
def test_a_reopened_wizard_keeps_the_saved_catalog_and_generates_only_an_automatic_one(subscription_ui, onboarding,
                                                                                      monkeypatch, saved):
    """Previews and Finish leave a catalog the install has exactly as saved; only an absent one is generated."""
    from ouroboros.configured_subagents import configured_subagents_dict as shown
    from ouroboros.configured_subagents import normalize_configured_subagents as parse
    from ouroboros.configured_subagents import serialize_configured_subagents as serialize

    def canonical(value):
        return shown(parse(value)[0])

    ui = subscription_ui
    stored = serialize(parse(saved)[0]) if saved else None
    document = {"OPENAI_API_KEY": "fixture-stored-key-0001", **({"OUROBOROS_SUBAGENTS": stored} if stored else {})}
    page = _open(ui, onboarding, monkeypatch, document)
    _models(page)
    capture(page, f"wizard-reopened-catalog-{'automatic' if saved is None else 'saved'}")
    body = _finish(ui, page)
    after = canonical(onboarding.saved()["OUROBOROS_SUBAGENTS"])
    if saved is not None:
        assert canonical(body["OUROBOROS_SUBAGENTS"]) == canonical(saved) == after
        assert onboarding.saved()["OUROBOROS_SUBAGENTS"] == stored  # byte for byte
    else:
        assert [row["route"]["target_id"] for row in after["items"] if row.get("review_eligible")] == _reviewers(ui)
        assert _reviewers(ui) and all(_on("openai", target) for target in _reviewers(ui))


def _quick_finish(ui, page, label):
    """Accounts → the Summary through the connected subscription's quick start, then Finish: ``(posted body, the
    error the wizard shows)``; the body is ``None`` when nothing was posted, the error ``""`` once it finished."""
    page.wait_for_selector("#quick-start-btn:not([hidden])")
    page.wait_for_function("() => !document.querySelector('#onboarding-access-retry')?.disabled")
    page.click("#quick-start-btn")
    page.wait_for_selector(".summary-card")
    return _quick_finish_from_summary(ui, page, label)


def _quick_finish_from_summary(ui, page, label):
    capture(page, label)
    page.click("#next-btn")
    page.wait_for_function("(home) => location.href === home || document.querySelector('.wizard-error')?.textContent",
                           arg=ui["url"] + "/")
    error = "" if page.url == ui["url"] + "/" else page.locator(".wizard-error").inner_text()
    if error:
        capture(page, f"{label}-refused")
    posted = [body for path, body in ui["posts"] if path == "/api/onboarding/complete"]
    return (posted[-1] if posted else None), error


@pytest.mark.parametrize("saved", [
    {"enabled": True, "items": []},  # explicitly empty
    {"enabled": False, "items": []},  # Off
    {"enabled": True, "items": [HELPER]},  # rows, none marked Reviewer
], ids=["empty", "off", "unmarked"])
def test_a_connected_subscription_keeps_the_saved_catalog_through_preview_and_finish(subscription_ui, onboarding,
                                                                                    monkeypatch, saved):
    """Reopening setup with Codex connected: the subscription preset proposes reviewers, and a catalog the install
    saved is still shown and saved as it is. A fresh install's generated one gains them (the next test)."""
    from ouroboros.configured_subagents import configured_subagents_dict as shown
    from ouroboros.configured_subagents import normalize_configured_subagents as parse

    ui = subscription_ui
    page = _open(ui, onboarding, monkeypatch, {"OPENAI_API_KEY": "fixture-openai-key-00001",
                                               "OUROBOROS_MODEL": "openai::gpt-5.6-terra",
                                               "OUROBOROS_SUBAGENTS": json.dumps(saved)}, subscriptions=True)
    label = f"wizard-subscription-saved-catalog-{'on' if saved['enabled'] else 'off'}-{len(saved['items'])}"
    body, error = _quick_finish(ui, page, label)
    if saved["items"]:
        # Rows and no reviewer: the wizard asks the owner to confirm the empty pool, and posts nothing until then.
        assert body is None and error.startswith("No row is marked Reviewer."), error
        assert onboarding.saved()["OUROBOROS_SUBAGENTS"] == json.dumps(saved)
        for _ in range(3):
            page.click("#back-btn")
        page.click('details[data-collapse="subagents"] > summary')
        page.check("[data-review-pool-allow-empty]")
        for _ in range(3):
            page.click("#next-btn")
        page.wait_for_selector(".summary-card")
        body, error = _quick_finish_from_summary(ui, page, f"{label}-confirmed")
    assert not error and body is not None, error
    previews = [answer["available_subagents"] for status, answer in ui["previews"] if status == 200]
    assert previews and all(catalog == shown(parse(saved)[0]) for catalog in previews), previews
    assert shown(parse(body["OUROBOROS_SUBAGENTS"])[0]) == shown(parse(saved)[0])
    assert shown(parse(onboarding.saved()["OUROBOROS_SUBAGENTS"])[0]) == shown(parse(saved)[0])


SAVED_MODELS = {"OUROBOROS_MODEL": "google/gemini-3.8-flash", "OUROBOROS_MODEL_LIGHT": "",
                "OUROBOROS_MODEL_FALLBACKS": "", "OUROBOROS_MODEL_VISION": ""}


@pytest.mark.parametrize("document", [{**SAVED_MODELS, "OPENROUTER_API_KEY": "sk-or-v1-fixture-000001"}, SAVED_MODELS,
                                      None], ids=["saved-with-key", "saved-without-key", "fresh"])
def test_a_connected_subscription_proposes_only_the_model_slots_the_wizard_owns(subscription_ui, onboarding,
                                                                               monkeypatch, document):
    """The preview proposes the subscription's model for Main and Light. A fresh install takes it; a saved slot,
    the shipped Main and a blank included, stays as saved. Without a key that serves the saved Main the wizard
    says so and writes nothing, rather than moving Main onto the subscription."""
    from ouroboros.settings_defaults import SETTINGS_DEFAULTS

    assert SAVED_MODELS["OUROBOROS_MODEL"] == SETTINGS_DEFAULTS["OUROBOROS_MODEL"]  # the shipped Main, saved
    ui = subscription_ui
    page = _open(ui, onboarding, monkeypatch, document, subscriptions=True)
    body, error = _quick_finish(ui, page, f"wizard-subscription-models-{'fresh' if document is None else 'saved'}"
                                          f"{'-key' if document and 'OPENROUTER_API_KEY' in document else ''}")
    proposed = [answer["model_settings"] for status, answer in ui["previews"] if status == 200]
    keyed = bool(document) and "OPENROUTER_API_KEY" in document  # Main has access: the server proposes nothing
    assert keyed is not any(settings.get("OUROBOROS_MODEL") == "claudexor::codex=gpt-test" for settings in proposed)
    saved = onboarding.saved()
    if document is not None and not keyed:  # the saved Main has no key here: the wizard says so, writes nothing
        assert body is None and error.startswith("Main uses OpenRouter"), error
        summary = page.locator(".summary-card").inner_text()
        assert "OpenRouter · google/gemini-3.8-flash" in summary
        assert "Uses Main · OpenRouter · google/gemini-3.8-flash" in summary  # Light: the saved blank
        assert saved == document
        return
    assert body is not None and not error, error
    if document is None:
        assert body["OUROBOROS_MODEL"] == saved["OUROBOROS_MODEL"] == "claudexor::codex=gpt-test"
        assert any(row.get("review_eligible") for row in json.loads(saved["OUROBOROS_SUBAGENTS"])["items"])
        return
    assert {key: body[key] for key in SAVED_MODELS} == SAVED_MODELS  # posted as saved, blanks included
    assert {key: saved[key] for key in SAVED_MODELS} == SAVED_MODELS


@pytest.mark.parametrize("main", ["openai::owner-saved-model", "openai::gpt-5.6-terra"])
def test_saved_main_with_a_missing_key_is_not_described_as_unset(subscription_ui, onboarding, monkeypatch, main):
    """A saved Main with an unavailable key stays chosen; two other keys are no automatic choice request."""
    document = {"OUROBOROS_MODEL": main, CONNECTIONS_KEY: [WORK, HOME]}
    page = _open(subscription_ui, onboarding, monkeypatch, document)
    row = _models(page)
    capture(page, "wizard-saved-main-missing-key-" + main.split("::")[1])
    assert row.locator('[data-model-role-model]').input_value() == main.split("::")[1]
    assert "Main is not set yet" not in row.inner_text()
    assert "This source has no key in Accounts." in row.inner_text()
    assert onboarding.saved() == document
