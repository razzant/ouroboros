"""Modern per-account discovery in real editors, using controlled HTTP responses."""
from __future__ import annotations

import copy
import json
from urllib.parse import parse_qs, urlparse

import pytest

from tests import test_subscription_role_routes_browser as roles

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui
# The API actor row is marked Reviewer: a reviewer is edited where its catalog row is.
CONSUMERS = ("Models", "API actor", "Native actor")
SAVED_MODEL = "catalog-owner-saved"
DRAFT_MODEL = "catalog-owner-unsaved"
SOURCE = "opaque-source"
# A model suggestion names the model alone; account and capability facts never ride along.
ACCOUNT_CLAIMS = ("personal", "work", "available", "unavailable", "date unknown",
                  "272000", "1,000,000", "1000000", "Fast", "free")


@pytest.fixture
def account_catalog_ui(role_ui):
    ui = role_ui
    roles.configure_mixed(ui)
    stamp = "2026-09-12T00:00:00Z"
    raw_accounts, native_accounts, items, models = [], [], [], []
    for pin, unique, window, modes, availability in [
        ("personal", "a", 272000, ["standard"], "available"),
        ("work", "b", 1000000, ["standard", "fast"], "unavailable"),
    ]:
        native_models = [{"id": name, "name": name, "contextWindow": window,
                          "processing": {"modes": modes, "eligible": None}}
                         for name in ("catalog-shared", f"catalog-only-{unique}")]
        catalog = {"credentialProfileId": pin, "models": native_models,
                   "provenance": "fixture", "observedAt": stamp if pin == "personal" else None}
        account = {"credentialProfileId": pin, "availability": availability, "problem": None}
        raw_accounts.append({**account, "catalog": {**catalog, "source": SOURCE}})
        native_accounts.append({**account, "catalog": {**catalog, "harnessId": "codex",
                                                       "source": "manifest", "verifiedAgainst": "fixture-v1"}})
        for model in native_models:
            facts = {"credential_profile_id": pin, "availability": availability,
                     "problem": None, "observed_at": catalog["observedAt"], "provenance": "fixture"}
            items.append({**facts, "id": model["id"], "name": model["id"],
                          "value": f"claudexor::{SOURCE}={model['id']}", "source_id": SOURCE,
                          "context_window": window, "processing": model["processing"]})
            models.append({**model, **facts, "catalog_source": "manifest"})
    ui["fixture"]["catalog"] = {
        "items": items, "model_sources": [{"id": SOURCE, "label": "Codex", "credentialHarness": "codex",
                                           "processingPreferences": ["standard", "fast", "economy"]}],
        "account_catalogs": [{"source": SOURCE, "accounts": raw_accounts, "partial": False}],
        "errors": [], "partial": False,
    }
    ui["fixture"]["status"]["harnesses"] = [{
        "id": "codex", "display_name": "Codex", "enabled": True, "status": "ok", "models": models,
        "model_catalog": {"harnessId": "codex", "accounts": native_accounts, "partial": False},
    }]
    ui["fixture"]["status"]["quota"] = [
        {"subject": {"harness": "codex", "subject_id": pin}, "freshness": "fresh", "constraints": []}
        for pin in ("personal", "work")
    ]
    target = f"claudexor::{SOURCE}={SAVED_MODEL}"
    ui["settings"].update(OUROBOROS_MODEL=target, OUROBOROS_MODEL_ACCOUNTS={"main": "personal"})
    actors = ui["settings"]["OUROBOROS_SUBAGENTS"]["items"]
    actors[0]["route"]["target_id"] = target
    actors[2]["route"]["target_id"] = f"codex={SAVED_MODEL}"

    def catalog_response(route):
        ui["reads"].append(route.request.url)
        body = copy.deepcopy(ui["fixture"]["catalog"])
        pin = parse_qs(urlparse(route.request.url).query).get("credential_profile_id", [""])[0]
        if pin:
            body["items"] = [item for item in body["items"] if item["credential_profile_id"] == pin]
            envelope = body["account_catalogs"][0]
            envelope["accounts"] = [account for account in envelope["accounts"] if account["credentialProfileId"] == pin]
            body["errors"] = [error for error in body["errors"] if error["credential_profile_id"] == pin]
            envelope["partial"] = body["partial"] = any(account["catalog"] is None for account in envelope["accounts"])
        route.fulfill(content_type="application/json", body=json.dumps(body))

    ui["page"].route("**/api/model-catalog*", catalog_response)
    return ui


def editor(ui, consumer):
    page = roles.open_agents(ui)
    if consumer == "Models":
        page.locator('[data-settings-tab="models"]').click()
        row = page.locator('[data-model-role="main"]')
        selectors = ('[data-model-role-source]', '[data-model-role-model]', '[data-model-role-account]')
    else:
        row = page.locator('[data-subagent-row]').nth(0 if consumer == "API actor" else 2)
        selectors = tuple(f'[data-subagent-field="{field}"]' for field in ("route", "model", "account"))
    return page, row, *(row.locator(selector) for selector in selectors)


def suggestions(page, field, expected=None):
    """Inspect the real popup, restoring the owner's value after the search."""
    previous = field.input_value()
    field.fill("catalog-")
    popup = page.locator('[id="' + field.get_attribute("aria-controls") + '"]')
    page.wait_for_function("""id => document.getElementById(id)?.querySelector('[data-model-value]')""",
                           arg=field.get_attribute("aria-controls"))
    if expected is not None:
        page.wait_for_function("""({id, expected, custom}) => {
            const actual = [...document.getElementById(id).querySelectorAll('[data-model-value]')]
                .map(e => e.dataset.modelValue).filter(value => !custom.includes(value)).sort();
            return JSON.stringify(actual) === JSON.stringify(expected.sort());
        }""", arg={"id": field.get_attribute("aria-controls"), "expected": sorted(expected),
                    "custom": [SAVED_MODEL, DRAFT_MODEL]})
    options = popup.locator('[data-model-value]').evaluate_all(
        "els => Object.fromEntries(els.map(e => [e.dataset.modelValue, e.textContent]))")
    field.fill(previous)
    field.press("Escape")
    return options


def refresh(ui):
    ui["page"].evaluate("""async () => {
        const {claudexorStatus} = await import('/static/modules/claudexor_status_store.js');
        await claudexorStatus.refresh({includeModels:true});
        await (await import('/static/modules/settings_catalog.js')).refreshModelCatalog();
    }""")


def assert_saved(ui, consumer, model, pin):
    page = ui["page"]
    with page.expect_response("**/api/settings"):
        page.locator("#btn-save-settings").click()
    page.wait_for_function("() => !document.querySelector('#btn-save-settings').disabled")
    saved = [body for path, body in ui["posts"] if path == "/api/settings"][-1]
    target = f"claudexor::{SOURCE}={model}"
    if consumer == "Models":
        assert saved["OUROBOROS_MODEL"] == target
        assert saved["OUROBOROS_MODEL_ACCOUNTS"]["main"] == pin
    else:
        route = saved["OUROBOROS_SUBAGENTS"]["items"][0 if consumer == "API actor" else 2]["route"]
        assert route == {"kind": "api_model" if consumer == "API actor" else "agent_session",
                         "target_id": target if consumer == "API actor" else f"codex={model}",
                         **({"credential_profile_id": pin} if pin else {})}
    assert saved["OUROBOROS_SUBAGENTS"]["items"][1]["route"]["target_id"] == "openai::gpt-api"
    assert not any("login" in path or "wake" in path for path, _ in ui["posts"])


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_account_pin_auto_roundtrip_keeps_saved_custom_model(account_catalog_ui, consumer):
    ui = account_catalog_ui
    page, row, source, field, account = editor(ui, consumer)
    expected_source = "session:codex" if consumer == "Native actor" else f"subscription:{SOURCE}"
    for pin, known in [("personal", {"catalog-shared", "catalog-only-a"}),
                       ("", {"catalog-shared", "catalog-only-a", "catalog-only-b"}),
                       ("work", {"catalog-shared", "catalog-only-b"})]:
        account.select_option(pin)
        assert field.input_value() == SAVED_MODEL
        assert source.input_value() == expected_source
        options = suggestions(page, field, known)
        assert set(options) - {SAVED_MODEL} == known
        shared = options["catalog-shared"]
        assert shared == "catalog-shared"
        assert not any(claim in shared for claim in ACCOUNT_CLAIMS)
    roles.capture(page, "account-catalog-pin-" + consumer.lower().replace(" ", "-"))
    assert_saved(ui, consumer, SAVED_MODEL, "work")
    _, _, _, reopened, reopened_account = editor(ui, consumer)
    assert reopened.input_value() == SAVED_MODEL
    assert reopened_account.input_value() == "work"


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_partial_account_refresh_keeps_draft_and_only_failed_account_history(account_catalog_ui, consumer):
    ui = account_catalog_ui
    page, row, source, field, account = editor(ui, consumer)
    account.select_option("")
    assert "catalog-only-a" in suggestions(page, field, {"catalog-shared", "catalog-only-a", "catalog-only-b"})
    field.fill(DRAFT_MODEL)
    field.evaluate("e => { window.accountDraftNode = e; e.setSelectionRange(4, 9); }")
    source.evaluate("e => { window.accountSourceNode = e; }")
    account.evaluate("e => { window.accountPinNode = e; }")
    problem = {"code": "model_catalog_unavailable", "message": "Work catalog could not be read"}
    raw = ui["fixture"]["catalog"]
    native = ui["fixture"]["status"]["harnesses"][0]
    for envelope in (raw["account_catalogs"][0], native["model_catalog"]):
        envelope["partial"] = True
        envelope["accounts"][0]["catalog"]["models"] = []
        envelope["accounts"][1].update(catalog=None, availability="unknown", problem=problem)
    raw.update(items=[], partial=True, errors=[{"credential_profile_id": "work", "error": problem["message"]}])
    native["models"] = []
    refresh(ui)
    if consumer == "Models":
        page.wait_for_function("""message => document.querySelector(
            '[data-model-role=main] [data-model-role-status]').textContent.includes(message)""", arg=problem["message"])
    assert field.evaluate("e => e === window.accountDraftNode && document.activeElement === e")
    assert field.evaluate("e => [e.selectionStart, e.selectionEnd]") == [4, 9]
    assert field.input_value() == DRAFT_MODEL
    assert source.evaluate("e => e === window.accountSourceNode")
    assert account.evaluate("e => e === window.accountPinNode")
    assert account.input_value() == ""
    assert problem["message"] in page.locator("#settings-model-catalog-status").text_content()
    options = suggestions(page, field, {"catalog-shared", "catalog-only-b"})
    assert set(options) - {DRAFT_MODEL} == {"catalog-shared", "catalog-only-b"}
    shared = options["catalog-shared"]
    assert shared == "catalog-shared"
    assert not any(claim in shared for claim in ACCOUNT_CLAIMS)
    if consumer == "Native actor":
        account.select_option("personal")
        assert "not in discovery" in suggestions(page, field)[DRAFT_MODEL]
        account.select_option("work")
        assert "not checked" in suggestions(page, field)[DRAFT_MODEL]
        status = row.locator("[data-subagent-status]")
        assert status.inner_text() == "Not checked"
        # The unread account catalog is disclosed on the row's own status sentence.
        assert "model list could not be read" in status.get_attribute("title")
        account.select_option("")
    roles.capture(page, "account-catalog-partial-" + consumer.lower().replace(" ", "-"))
    assert_saved(ui, consumer, DRAFT_MODEL, "")


def test_models_auto_does_not_borrow_largest_context_or_processing_modes(account_catalog_ui):
    ui = account_catalog_ui
    ui["settings"]["OUROBOROS_MODEL"] = f"claudexor::{SOURCE}=catalog-shared"
    ui["settings"]["OUROBOROS_PROCESSING_PREFERENCE"] = "fast"
    page, row, _, field, account = editor(ui, "Models")
    row.locator("summary").click()
    note = row.locator("[data-model-context-note]")
    page.wait_for_function("() => document.querySelector('[data-model-role=main] [data-model-context-note]').textContent.includes('272,000')")
    assert "Fast is not advertised" in row.locator("[data-processing-capability]").inner_text()
    account.select_option("")
    assert {"catalog-only-a", "catalog-only-b"} <= set(suggestions(page, field, {"catalog-shared", "catalog-only-a", "catalog-only-b"}))
    assert "not known" in note.inner_text()
    assert "1,000,000" not in note.inner_text()
    account.select_option("work")
    page.wait_for_function("() => document.querySelector('[data-model-role=main] [data-model-context-note]').textContent.includes('1,000,000')")
    assert row.locator("[data-processing-capability]").inner_text() == ""
    assert field.input_value() == "catalog-shared"
    roles.capture(page, "account-catalog-exact-context-processing")
