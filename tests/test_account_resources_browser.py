"""Actual Settings resources against synthetic API responses; no engine startup."""
from __future__ import annotations

import copy
import json
import math
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.settings_browser_helpers import SETTLED, open_settings

subscription_ui = setup_browser.subscription_ui
capture = setup_browser.capture
pytestmark = [pytest.mark.ui_browser]
TARGET = {"harness": "claude", "profile_id": "claude-default"}
ROW = '.harness-account-row[data-harness="claude"][data-profile="claude-default"]'


@pytest.fixture
def resources_ui(subscription_ui):
    ui = subscription_ui
    wire = json.loads((setup_browser.WEB / "tests/fixtures/account_resources.json").read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    for constraint, hours in zip(wire["quota"]["snapshots"][0]["constraints"], [2, 120]):
        constraint["resets_at"] = (now + timedelta(hours=hours)).isoformat()
    grant = wire["quota"]["resources"][0]["resets"]["value"][0]["grants"][0]
    grant.update(starts_at=None, expires_at=(now + timedelta(days=14)).isoformat())
    wire["quota"]["resources"][0]["spending"]["value"][0]["resets_at"] = (now + timedelta(days=20)).isoformat()
    status = ui["fixture"]["status"]
    status.update(quota=[], quota_absences=[], resources=[],
                  resource_capabilities_read="ok",
                  resource_capabilities={"read": True, "refresh": True, "reset": True, "inspect_reset": True})
    profiles = status["profiles"]["profiles"] = []
    for index in range(20):
        target = TARGET if index == 0 else {"harness": "claude",
                                          "profile_id": f"account-{index:02}"}
        profiles.append({"profile": {"harness_id": target["harness"], "profile_id": target["profile_id"],
                                     "display_name": "Personal default" if index == 0 else f"Workspace {index:02}",
                                     "enabled": index != 0},
                         "status": {"verification": "passed", "verification_source": "vendor", "availability": "available"},
                         "identity": {"label": "Personal default" if index == 0 else f"Workspace {index:02}"}})
        quota = copy.deepcopy(wire["quota"]["snapshots"][0])
        quota["subject"].update(harness=target["harness"], subject_id=target["profile_id"])
        if index:
            quota["constraints"][0]["used_ratio"] = None if index == 1 else 0 if index == 2 else 0.23
            quota["constraints"][1]["used_ratio"] = None if index == 1 else 0.42
        resource = copy.deepcopy(wire["quota"]["resources"][0])
        resource["target"] = target
        status["quota"].append(quota)
        status["resources"].append(resource)
    ui.update(wire=wire, reset_calls=[], refresh_calls=[], reset_replies=[], hold_refresh=[], pending_refresh=[])

    def reset(route):
        ui["reset_calls"].append((route.request.headers.get("idempotency-key"), route.request.post_data_json))
        reply = ui["reset_replies"].pop(0)
        if reply == "timeout":
            route.abort("timedout")
        else:
            route.fulfill(json=reply)

    def refresh(route):
        ui["refresh_calls"].append(route.request.post_data_json)
        if ui["hold_refresh"]:
            ui["pending_refresh"].append(route)
        else:
            route.fulfill(json=ui["wire"]["quota"])

    def bind_resources(page):
        page.route("**/api/claudexor/account-resets", reset)
        page.route("**/api/claudexor/quota/refresh", refresh)

    ui["bind_resources"] = bind_resources
    bind_resources(ui["page"])
    return ui


def open_accounts(ui, width=390, theme="dark"):
    from playwright.sync_api import expect

    page = ui["page"]
    page.set_viewport_size({"width": width, "height": 960})
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
    page.goto(ui["url"] + "/#settings")
    open_settings(page, "accounts/claude")
    expect(page.locator(".harness-account-row[data-harness]")).to_have_count(20)
    return page, page.locator(ROW)


def passive_read(page):
    page.evaluate("""async () => {
        const { claudexorStatus } = await import('/static/modules/claudexor_status_store.js');
        await claudexorStatus.refresh();
    }""")


def frame_section(locator):
    """Position evidence below Settings' existing sticky fade, without product scrolling."""
    locator.evaluate("""element => {
        const host = element.closest('[data-settings-page]');
        if (host) host.scrollTop += element.getBoundingClientRect().top - host.getBoundingClientRect().top - 48;
        else element.scrollIntoView({block: 'center'});
    }""")


@pytest.mark.parametrize("width", [360, 390, 1440])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_twenty_accounts_layout_and_scoped_refresh(resources_ui, width, theme):
    from playwright.sync_api import expect

    ui = resources_ui
    page, row = open_accounts(ui, width, theme)
    expect(page.locator(".account-resource-panel")).to_have_count(0)
    row.scroll_into_view_if_needed()
    capture(page, f"resources-{width}-{theme}-collapsed")
    sibling = page.locator('.harness-account-row[data-profile="account-01"]')
    assert "0%" not in sibling.locator(".harness-account-meta").inner_text()
    sibling_before = sibling.inner_text()
    row.locator("[data-resources]").click()
    panel = row.locator(".account-resource-panel")
    expect(panel).to_contain_text("24.60 USD")
    expect(panel).to_contain_text("0.00 USD")
    expect(panel).to_contain_text("balance_read_failed")
    expect(panel).to_contain_text("Disabled for automatic routing")
    expect(panel).to_contain_text("Weekly limits still apply")
    expect(panel).to_contain_text("Available now · count not reported")
    expect(panel.locator('[data-resource-reset="opaque-grant"]')).to_be_enabled()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    assert panel.evaluate("e => e.scrollWidth <= e.clientWidth + 1")
    if width < 980:
        # A desk pointer: the one control height at every width; a finger's 44 px comes from the coarse-pointer scope.
        control_h = page.evaluate("parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--control-h'))")
        assert control_h >= 34
        assert panel.locator("[data-refresh-account]").bounding_box()["height"] == pytest.approx(control_h, abs=0.5)
    panel.scroll_into_view_if_needed()
    capture(page, f"resources-{width}-{theme}-expanded")
    frame_section(panel.locator('.resource-reset-section'))
    capture(page, f"resources-{width}-{theme}-reset-options")
    details = panel.locator('.resource-provenance')
    details.locator('summary').focus()
    details.locator('summary').press('Enter')
    passive_read(page)
    expect(details).to_have_attribute('open', '')
    expect(details.locator('summary')).to_be_focused()
    details.locator('summary').click()
    refresh = row.locator("[data-refresh-account]")
    refresh.focus()
    passive_read(page)
    expect(refresh).to_be_focused()
    expect(row.locator("[data-resources]")).to_have_attribute("aria-expanded", "true")
    ui["wire"]["quota"]["snapshots"][0]["constraints"][0]["used_ratio"] = 0
    ui["hold_refresh"].append(True)
    refresh.click()
    expect(refresh).to_have_text("Refreshing…")
    expect(refresh).to_have_attribute("aria-disabled", "true")
    page.wait_for_function("document.querySelector('[data-refresh-account][aria-busy=true]') !== null")
    assert ui["refresh_calls"] == [{"target": TARGET}]
    ui["pending_refresh"][0].fulfill(json=ui["wire"]["quota"])
    expect(row.locator(".resource-window").first).to_contain_text("0% used")
    expect(refresh).to_be_focused()
    assert sibling.inner_text() == sibling_before
    assert not ui["reset_calls"]


def test_confirm_reset_failed_readback_and_read_only_refresh(resources_ui):
    from playwright.sync_api import expect

    ui = resources_ui
    page, row = open_accounts(ui, width=1440)
    row.locator("[data-resources]").click()
    reset = row.locator('[data-resource-reset="opaque-grant"]')
    reset.focus()
    reset.click()
    dialog = page.get_by_role("dialog")
    expect(dialog).to_have_count(1)
    expect(dialog).to_contain_text("Restores the five-hour and weekly included limits")
    passive_read(page)  # Trigger replaced while the shared dialog owns focus.
    page.keyboard.press("Escape")
    expect(reset).to_be_focused()
    assert not ui["reset_calls"]
    ui["reset_replies"].append(ui["wire"]["receipt"])
    reset.click()
    page.locator("[data-confirm-ok]").click()
    expect(row).to_contain_text("Reset applied")
    expect(row).to_contain_text("Usage and remaining resets have not been updated")
    expect(row.locator(".resource-window").first).to_contain_text("Last known · 100% used")
    expect(row).to_contain_text("Last reported: 1 available · current availability unknown")
    expect(row).not_to_have_class("harness-account-row harness-exhausted")
    expect(reset).to_be_focused()
    first_key, first_body = ui["reset_calls"][0]
    assert first_key and first_body == ui["wire"]["receipt"]["request"]
    frame_section(row.locator('.account-resource-panel'))
    capture(page, "resources-reset-applied-readback-failed")
    ui["wire"]["quota"]["snapshots"][0]["constraints"][0]["used_ratio"] = 0
    ui["wire"]["quota"]["resources"][0]["resets"]["value"][0]["available_count"] = 0
    row.locator("[data-refresh-account]").click()
    expect(row.locator(".resource-window").first).to_contain_text("0% used")
    expect(row.locator(".resource-window").first).not_to_contain_text("Last known")
    assert len(ui["reset_calls"]) == 1
    expect(reset).to_be_disabled()
    expect(row.locator('[data-resource-reset="opaque-refill"]')).to_be_enabled()
    ui["wire"]["quota"]["resources"][0]["resets"]["value"][0]["available_count"] = 2
    row.locator("[data-refresh-account]").click()
    expect(reset).to_be_enabled()
    reset.click()
    expect(dialog).to_contain_text("separate request and may use an additional reset")
    assert len(ui["reset_calls"]) == 1
    ui["reset_replies"].append(ui["wire"]["receipt"])
    page.locator("[data-confirm-ok]").click()
    expect(row).to_contain_text("Reset applied")
    assert ui["reset_calls"][1][0] != first_key


def test_timeout_before_receipt_survives_session_storage_loss_and_recovers_same_request(resources_ui):
    from playwright.sync_api import expect

    ui = resources_ui
    page, row = open_accounts(ui, width=1440)
    row.locator("[data-resources]").click()
    ui["reset_replies"].append("timeout")
    row.locator('[data-resource-reset="opaque-refill"]').click()
    expect(page.get_by_role("dialog")).to_contain_text("session window only. Weekly limits still apply")
    page.locator("[data-confirm-ok]").click()
    expect(row).to_contain_text("Reset outcome is unconfirmed")
    original = ui["reset_calls"][0]
    # The recovery result is multiline and its action is optional. Narrow the
    # real row inside a wide window: zero overflow alone missed text starvation.
    result_row = row.locator(".resource-message").filter(has=page.locator("[data-recover-reset]"))
    result_row.evaluate("e => { e.style.width = '320px'; e.style.maxWidth = '100%'; }")
    result_row.scroll_into_view_if_needed()
    copy_box = result_row.locator(".settings-action-copy").bounding_box()
    recovery = row.locator("[data-recover-reset]")
    action_box = recovery.bounding_box()
    available = result_row.evaluate("e => { const s = getComputedStyle(e); return e.clientWidth - parseFloat(s.paddingLeft) - parseFloat(s.paddingRight); }")
    assert copy_box["width"] >= available - 2
    assert action_box["y"] >= copy_box["y"] + copy_box["height"] - 1
    assert result_row.evaluate("e => e.scrollWidth <= e.clientWidth + 1")
    recovery.focus()
    expect(recovery).to_be_focused()
    capture(page, "resources-recovery-container-reflow")
    # Closing an embedded client loses session storage; durable references must survive.
    page.evaluate("sessionStorage.clear()")
    ui["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    open_settings(page, "accounts/claude")
    row.locator("[data-resources]").click()
    expect(row).to_contain_text("Reset outcome is unconfirmed")
    receipt = copy.deepcopy(ui["wire"]["receipt"])
    receipt.update(request=original[1], outcome="already_used")
    ui["reset_replies"].append(receipt)
    row.locator("[data-recover-reset]").click()
    expect(row).to_contain_text("does not confirm that this request applied it")
    assert ui["reset_calls"] == [original, original]
    expect(row).not_to_contain_text("Reset applied")
    frame_section(row.locator('.account-resource-panel'))
    capture(page, "resources-same-request-already-used")
    receipt = {**receipt, "outcome": "already_redeemed"}
    ui["reset_replies"].append(receipt)
    row.locator("[data-recover-reset]").click()
    expect(row).to_contain_text("Reset applied")
    assert ui["reset_calls"] == [original] * 3


@pytest.mark.parametrize("scheme", ["http", "https"])
@pytest.mark.parametrize("offer_id", ["opaque-grant", "opaque-refill"])
def test_non_loopback_reset_keeps_original_request_after_reload(resources_ui, scheme, offer_id):
    from playwright.sync_api import expect

    ui = resources_ui
    page = ui["page"]
    fixture_url = ui["url"]
    origin = f"{scheme}://ouroboros-browser.invalid"

    def serve_fixture(route):
        path = urlparse(route.request.url).path
        if path.startswith("/api/"):
            route.fallback()  # Existing synthetic API routes own every account action.
        else:
            route.fulfill(response=route.fetch(url=fixture_url + route.request.url[len(origin):]))

    page.route(origin + "/**", serve_fixture)
    ui["url"] = origin
    page, row = open_accounts(ui, width=390)
    assert page.evaluate("isSecureContext") is (scheme == "https")
    assert page.evaluate("typeof crypto.randomUUID") == ("function" if scheme == "https" else "undefined")
    row.locator("[data-resources]").click()
    ui["reset_replies"].append("timeout")
    row.locator(f'[data-resource-reset="{offer_id}"]').click()
    expect(page.get_by_role("dialog")).to_have_count(1)
    page.locator("[data-confirm-ok]").click()
    expect(row).to_contain_text("Reset outcome is unconfirmed")
    assert len(ui["reset_calls"]) == 1
    original = ui["reset_calls"][0]
    assert original[0] and original[1]["offer_id"] == offer_id and original[1]["target"] == TARGET
    ui["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    open_settings(page, "accounts/claude")
    row.locator("[data-resources]").click()
    expect(row).to_contain_text("Reset outcome is unconfirmed")
    assert len(ui["reset_calls"]) == 1, "reload must not resend an unresolved request"
    reply = {**ui["wire"]["receipt"], "request": original[1], "outcome": "no_credit", "resources": None}
    ui["reset_replies"].append(reply)
    row.locator("[data-recover-reset]").click()
    expect(row).to_contain_text("No reset available")
    assert ui["reset_calls"] == [original, original]
    expect(page.get_by_role("dialog")).to_have_count(0)
    frame_section(row.locator(".account-resource-panel"))
    capture(page, f"resources-{scheme}-{offer_id}-same-request-recovered")


def test_deliberate_refill_no_effect_keeps_lost_first_request_after_client_reopen(resources_ui):
    from playwright.sync_api import expect

    ui = resources_ui
    page, row = open_accounts(ui, width=390, theme="light")
    row.locator("[data-resources]").click()
    ui["reset_replies"].append("timeout")
    row.locator('[data-resource-reset="opaque-grant"]').click()
    page.locator("[data-confirm-ok]").click()
    expect(row).to_contain_text("Reset outcome is unconfirmed")
    original = ui["reset_calls"][0]
    refill = row.locator('[data-resource-reset="opaque-refill"]')
    expect(refill).to_have_text("Refill session again")
    refill.click()
    dialog = page.get_by_role("dialog")
    expect(dialog).to_have_count(1)
    expect(dialog).to_contain_text("Refill session again for Personal default?")
    expect(dialog).to_contain_text("separate request and may use an additional reset")
    reply = {**ui["wire"]["receipt"], "id": "new-no-effect", "outcome": "no_credit", "resources": None,
             "request": {"target": TARGET, "offer_id": "opaque-refill"}}
    ui["reset_replies"].append(reply)
    expect(page.locator("[data-confirm-ok]")).to_have_text("Refill session again")
    page.locator("[data-confirm-ok]").click()
    expect(row).to_contain_text("No reset available")
    expect(row).to_contain_text("Earlier unresolved requests")
    expect(row.locator('[data-recover-reset]')).to_have_count(1)
    assert ui["reset_calls"][1][0] != original[0]
    recover = row.locator(f'[data-recover-reset="{original[0]}"]')
    recover.focus()
    passive_read(page)
    expect(recover).to_be_focused()
    assert row.locator(".account-resource-panel").evaluate("e => e.scrollWidth <= e.clientWidth + 1")
    frame_section(row.locator('.account-resource-panel'))
    capture(page, "correction-earlier-unresolved-mobile")
    saved = page.context.storage_state()
    page.close()
    ui["page"] = ui["new_page"](storage_state=saved)
    ui["bind_resources"](ui["page"])
    page, row = open_accounts(ui, width=390, theme="light")
    row.locator("[data-resources]").click()
    expect(row).to_contain_text("Reset outcome is unconfirmed")
    assert len(ui["reset_calls"]) == 2, "opening the client must not recover automatically"
    recovered = {**ui["wire"]["receipt"], "request": original[1], "outcome": "already_used", "resources": None}
    ui["reset_replies"].append(recovered)
    recover = row.locator(f'[data-recover-reset="{original[0]}"]')
    recover.click()
    expect(row).to_contain_text("does not confirm that this request applied it")
    expect(recover).to_be_focused()
    assert ui["reset_calls"][2] == original
    expect(page.get_by_role("dialog")).to_have_count(0)
    ui["reset_replies"].append({**recovered, "outcome": "already_redeemed"})
    recover.click()
    expect(row).to_contain_text("Reset applied")
    expect(row.locator("[data-resources]")).to_be_focused()
    assert ui["reset_calls"][3] == original


def test_old_receipt_preserves_new_refresh_and_newer_recovery_updates_display(resources_ui):
    from playwright.sync_api import expect

    ui = resources_ui
    page, row = open_accounts(ui, width=1440)
    row.locator("[data-resources]").click()
    old = copy.deepcopy(ui["wire"]["quota"])
    old["snapshots"][0].update(observed_at="2026-10-09T10:00:00Z")
    old["snapshots"][0]["constraints"][0]["used_ratio"] = 0.9
    for facet in ("balances", "spending", "resets", "diagnostics"):
        old["resources"][0][facet].update(observed_at="2026-10-09T10:00:00Z", last_attempt_at="2026-10-09T10:00:00Z")
    receipt = {**ui["wire"]["receipt"], "outcome": "already_used", "resources": old}
    ui["reset_replies"].append(receipt)
    row.locator('[data-resource-reset="opaque-grant"]').click()
    page.locator("[data-confirm-ok]").click()
    expect(row).to_contain_text("does not confirm that this request applied it")
    newer = ui["wire"]["quota"]
    newer["snapshots"][0]["observed_at"] = "2026-10-09T13:00:00Z"
    newer["snapshots"][0]["constraints"][0]["used_ratio"] = 0.4
    newer["resources"][0]["resets"].update(observed_at="2026-10-09T13:00:00Z", last_attempt_at="2026-10-09T13:00:00Z")
    newer["resources"][0]["resets"]["value"][0]["available_count"] = 0
    ui["fixture"]["status"]["quota"][0] = copy.deepcopy(newer["snapshots"][0])
    ui["fixture"]["status"]["resources"][0] = copy.deepcopy(newer["resources"][0])
    row.locator("[data-refresh-account]").click()
    expect(row.locator(".resource-window").first).to_contain_text("40% used")
    ui["reset_replies"].append(receipt)
    row.locator("[data-recover-reset]").click()
    expect(row.locator(".resource-window").first).to_contain_text("40% used")
    expect(row.locator(".resource-window").first).not_to_contain_text("Last known")
    expect(row).to_contain_text("Refresh completed")
    expect(row.locator('[data-resource-reset="opaque-grant"]')).to_be_disabled()
    expect(row.locator("[data-recover-reset]")).to_be_focused()
    frame_section(row.locator('.account-resource-panel'))
    capture(page, "correction-old-receipt-new-refresh")
    latest = copy.deepcopy(newer)
    latest["snapshots"][0]["observed_at"] = "2026-10-09T11:30:00-03:00"
    latest["snapshots"][0]["constraints"][0]["used_ratio"] = 0.2
    latest["resources"][0]["resets"].update(observed_at="2026-10-09T14:30:00Z", last_attempt_at="2026-10-09T14:30:00Z")
    latest["resources"][0]["resets"]["value"][0]["available_count"] = 2
    ui["reset_replies"].append({**receipt, "resources": latest})
    row.locator("[data-recover-reset]").click()
    expect(row.locator(".resource-window").first).to_contain_text("20% used")
    expect(row.locator('[data-resource-reset="opaque-grant"]')).to_be_enabled()
    assert ui["reset_calls"] == [ui["reset_calls"][0]] * 3


def test_catalog_gap_retains_last_known_resources_and_valid_account_quota_siblings(resources_ui):
    from playwright.sync_api import expect

    ui = resources_ui
    page, row = open_accounts(ui, width=390)
    row.locator("[data-resources]").click()
    status = ui["fixture"]["status"]
    rich = copy.deepcopy(status)
    status.update(resource_capabilities_read="failed", resource_capabilities={"read": False, "reset": False})
    status.pop("resources")
    status["quota"][0]["constraints"][0]["used_ratio"] = 0.3
    passive_read(page)
    expect(row).to_contain_text("capabilities could not be checked")
    expect(row).to_contain_text("24.60 USD")
    expect(row).to_contain_text("Last reported: 1 available")
    expect(row.locator(".resource-window").first).to_contain_text("30% used")
    expect(row.locator(".resource-window").first).not_to_contain_text("Last known")
    expect(row).not_to_contain_text("does not expose account resources")
    expect(row.locator('[data-resource-reset="opaque-grant"]')).to_be_enabled()
    expect(page.locator(".harness-account-row[data-harness]")).to_have_count(20)
    status["resource_capabilities_read"] = "ok"
    passive_read(page)
    expect(row).to_contain_text("This engine does not expose account resources")
    expect(row).not_to_contain_text("24.60 USD")
    expect(row.locator('[data-resource-reset]')).to_have_count(0)
    status.update(rich)
    passive_read(page)
    expect(row).to_contain_text("24.60 USD")
    expect(row).not_to_contain_text("capabilities could not be checked")
    expect(row.locator('[data-resource-reset="opaque-grant"]')).to_be_enabled()
    assert not ui["reset_calls"]


@pytest.mark.parametrize("width,theme", [(1440, "light"), (390, "dark")])
def test_maintenance_and_resources_share_refresh_without_losing_pending_reset(resources_ui, width, theme):
    from urllib.parse import parse_qs, urlparse

    from playwright.sync_api import expect
    from tests.test_harness_maintenance_browser import _inventory_row

    ui = resources_ui
    entry = _inventory_row("claude")
    maintenance_only = _inventory_row("agy", exact=False)
    inventory = {"harnesses": [entry, maintenance_only]}
    operation = {"id": "program-update", "harness": "claude", "state": "running",
                 "phase": "installing", "target": {"kind": "latest", "version": "3.0.0"},
                 "mutation": "unknown", "termination": "not_applicable"}
    inspections, held, updates, cancellations = [], [], [], []
    hold_inventory = False

    def maintenance(route):
        path = urlparse(route.request.url).path
        if path.endswith("/harnesses"):
            inspections.append(parse_qs(urlparse(route.request.url).query))
            if hold_inventory:
                held.append(route)
            else:
                route.fulfill(json=inventory)
        elif path.endswith("/cancel"):
            cancellations.append(path)
            operation.update(state="cancelled", phase="settled", termination="unconfirmed")
            route.fulfill(json=operation)
        elif route.request.method == "POST":
            updates.append((route.request.headers.get("idempotency-key"), route.request.post_data_json))
            entry["operation"] = {key: operation[key] for key in ("id", "state", "phase")}
            route.fulfill(status=202, json=operation)
        else:
            route.fulfill(json=operation)

    ui["page"].route("**/api/claudexor/maintenance/**", maintenance)
    # A full resource refresh returns all 20 targets; the single-target fixture
    # used by the other tests must not manufacture account absence here.
    ui["wire"]["quota"]["snapshots"] = copy.deepcopy(ui["fixture"]["status"]["quota"])
    ui["wire"]["quota"]["resources"] = copy.deepcopy(ui["fixture"]["status"]["resources"])
    page, row = open_accounts(ui, width, theme)
    family = page.locator('.agent-family-card[data-family="claude"]')
    expect(family.locator('[data-family-maintenance]')).to_have_count(1)
    expect(family.get_by_role("button", name="Update", exact=True)).to_be_enabled()
    row.locator("[data-resources]").click()
    ui["reset_replies"].append("timeout")
    row.locator('[data-resource-reset="opaque-grant"]').click()
    page.locator("[data-confirm-ok]").click()
    expect(row).to_contain_text("Reset outcome is unconfirmed")
    original = ui["reset_calls"][0]
    details = row.locator(".resource-provenance")
    details.locator("summary").click()

    hold_inventory = True
    family.get_by_role("button", name="Details", exact=True).click()
    family.get_by_role("button", name="Check latest", exact=True).click()
    expect(family.get_by_role("button", name="Checking…", exact=True)).to_be_disabled()
    expect(details).to_have_attribute("open", "")
    assert held
    details.locator("summary").focus()
    hold_inventory = False
    entry["newest"] = {"status": "available", "version": "3.0.0", "observedAt": "2026-10-09T12:00:00Z"}
    entry["updateAvailable"] = "yes"
    for route in held:
        route.fulfill(json=inventory)
    expect(family.locator(".harness-maintenance-line")).to_contain_text("3.0.0 available")
    expect(details).to_have_attribute("open", "")
    expect(details.locator("summary")).to_be_focused()
    expect(row.locator("[data-resources]")).to_have_attribute("aria-expanded", "true")
    expect(row).to_contain_text("Reset outcome is unconfirmed")
    assert ui["reset_calls"] == [original]

    family.get_by_role("button", name="Update", exact=True).click()
    expect(family).to_contain_text("Updating…")
    expect(row.locator("[data-refresh-account]")).to_have_attribute("aria-disabled", "false")
    before_inspections = len(inspections)
    ui["hold_refresh"].append(True)
    page.locator("#btn-harness-refresh").click()
    expect(page.locator("#btn-harness-refresh")).to_have_text("Refreshing…")
    expect(page.locator("#btn-harness-refresh")).to_be_disabled()
    assert ui["refresh_calls"] == [{}]
    assert any(query.get("fresh") == ["true"] for query in inspections[before_inspections:])
    status_reads = lambda: sum(urlparse(url).path == "/api/claudexor/status" for url in ui["reads"])
    before_status = status_reads()
    family.get_by_role("button", name="Cancel update", exact=True).click()
    expect(family).to_contain_text("Installer may still be running")
    assert status_reads() == before_status, "maintenance settlement must join the held resource writer"
    expect(row).to_contain_text("Reset outcome is unconfirmed")
    ui["pending_refresh"][0].fulfill(json=ui["wire"]["quota"])
    expect(page.locator("#btn-harness-refresh")).to_be_enabled()
    expect(page.locator(".harness-account-row[data-harness]")).to_have_count(20)
    expect(row).to_contain_text("Reset outcome is unconfirmed")
    expect(details).to_have_attribute("open", "")
    expect(family).to_contain_text("Update cancelled")
    assert len(updates) == len(cancellations) == 1
    assert updates[0][0] and updates[0][1] == {"harness": "claude", "target": {"kind": "latest"}}
    expect(page.locator('.agent-family-card[data-family="agy"]')).to_have_count(0)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    frame_section(family.locator(".agent-family-head"))
    capture(page, f"combined-maintenance-resources-{width}-{theme}")

    ui["reset_replies"].append({**ui["wire"]["receipt"], "request": original[1],
                               "outcome": "already_used", "resources": None})
    row.locator("[data-recover-reset]").click()
    expect(row).to_contain_text("does not confirm that this request applied it")
    assert ui["reset_calls"] == [original, original]
    assert ui["refresh_calls"] == [{}]
    frame_section(row.locator(".account-resource-panel"))
    capture(page, f"combined-maintenance-recovered-{width}-{theme}")
    assert not any("/login" in path or path == "/api/settings" for path, _ in ui["posts"])


class ObservationTransport:
    """Delay real SPA transports independently, capturing each reply at dispatch."""

    def __init__(self, ui):
        self.ui = ui
        self.hold = set()
        self.pending = {name: [] for name in ("full", "passive", "patch")}
        self.calls = []
        self.full_status = 200
        self.passive_accounts = "ok"
        self.passive_quota = "ok"
        self.include_resources = True
        ui["page"].route("**/api/claudexor/status**", self.status)
        ui["page"].route("**/api/claudexor/credential-profiles/**", self.patch)

    def status(self, route):
        from urllib.parse import parse_qs

        passive = parse_qs(urlparse(route.request.url).query).get("view") == ["quota"]
        kind = "passive" if passive else "full"
        self.calls.append(kind)
        body = copy.deepcopy(self.ui["fixture"]["status"])
        code = 200 if passive else self.full_status
        if passive:
            body = {key: body[key] for key in ("profiles", "quota", "resources", "quota_absences") if key in body}
            body.update(view="quota", reads={"accounts": self.passive_accounts, "quota": self.passive_quota})
            if not self.include_resources:
                body.pop("resources", None)
        elif code != 200:
            body = {"error": "fixture full discovery unavailable"}
        if kind in self.hold:
            self.pending[kind].append((route, body, code))
        else:
            route.fulfill(status=code, json=body)

    def patch(self, route):
        self.calls.append("patch")
        assert route.request.method == "PATCH"
        wanted = route.request.post_data_json
        self.ui["posts"].append((urlparse(route.request.url).path, wanted))
        body = {"ok": True, **TARGET, "enabled": wanted["enabled"]}
        if "patch" in self.hold:
            self.pending["patch"].append((route, body, 200))
        else:
            route.fulfill(json=body)

    def release(self, kind):
        route, body, code = self.pending[kind].pop(0)
        route.fulfill(status=code, json=body)


def store_call(page, expression, *, wait=True):
    """Use the actual singleton mounted by Accounts; never replace its implementation."""
    return page.evaluate("async () => { const {claudexorStatus: s} = await import("
                         "'/static/modules/claudexor_status_store.js'); "
                         + (f"return await ({expression});" if wait else f"void ({expression});") + " }")


def wait_pending(page, transport, kind):
    # Sync Playwright dispatches route callbacks while the browser makes progress.
    for _ in range(100):
        if transport.pending[kind]:
            return
        page.wait_for_timeout(10)
    assert transport.pending[kind], f"No {kind} request; observed {transport.calls}"


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_requested_switch_survives_rerender_patch_and_older_full(resources_ui, theme):
    from playwright.sync_api import expect

    ui = resources_ui
    transport = ObservationTransport(ui)
    page, row = open_accounts(ui, width=1440, theme=theme)
    switch = row.locator("[data-harness-toggle]")
    expect(switch).not_to_be_checked()
    row.locator("[data-resources]").click()
    expect(row).to_contain_text("24.60 USD")
    transport.hold.update({"full", "patch"})
    store_call(page, "s.refresh()", wait=False)
    wait_pending(page, transport, "full")
    row.locator(".ui-switch-track").click()
    wait_pending(page, transport, "patch")
    expect(switch).to_be_checked()
    expect(switch).to_be_disabled()
    expect(switch).to_have_attribute("aria-busy", "true")
    # A separate quota observation repaints the actual Accounts controls while PATCH waits.
    store_call(page, "s.observe()")
    expect(switch).to_be_checked()
    expect(switch).to_be_disabled()
    expect(row.locator("[data-resources]")).to_have_attribute("aria-expanded", "true")
    frame_section(row)
    capture(page, f"staged-accounts-{theme}-pending-switch")
    # The engine has saved enabled=true; the held full reply still contains false.
    ui["fixture"]["status"]["profiles"]["profiles"][0]["profile"]["enabled"] = True
    transport.release("patch")
    expect(switch).to_be_enabled()
    expect(switch).to_be_checked()
    transport.release("full")
    page.wait_for_function("async () => !(await import('/static/modules/claudexor_status_store.js')).claudexorStatus.loading")
    expect(switch).to_be_checked()
    assert transport.calls.count("patch") == 1
    capture(page, f"staged-accounts-{theme}-confirmed-switch")


def test_fast_observation_recovers_facets_without_certifying_failed_catalog(resources_ui):
    from playwright.sync_api import expect

    ui = resources_ui
    transport = ObservationTransport(ui)
    page, row = open_accounts(ui, width=1440)
    transport.full_status = 503
    store_call(page, "s.refresh()")
    assert store_call(page, "s.reads") == {"catalog": "transport", "accounts": "transport", "quota": "transport"}
    full_count = transport.calls.count("full")
    store_call(page, "s.observe()")
    assert store_call(page, "s.reads") == {"catalog": "transport", "accounts": "ok", "quota": "ok"}
    expect(row.locator("[data-harness-toggle]")).to_be_enabled()
    assert transport.calls.count("full") == full_count
    assert "fixture full discovery unavailable" in store_call(page, "s.unavailableNote('catalog').text")
    assert store_call(page, "s.unavailableNote('accounts')") is None
    transport.passive_accounts = "failed"
    store_call(page, "s.observe()")
    assert store_call(page, "s.reads") == {"catalog": "transport", "accounts": "failed", "quota": "ok"}


def test_provider_refresh_does_not_block_or_fence_passive_roster(resources_ui):
    from playwright.sync_api import expect

    ui = resources_ui
    transport = ObservationTransport(ui)
    page, row = open_accounts(ui, width=1440)
    row.locator("[data-resources]").click()
    ui["hold_refresh"].append(True)
    row.locator("[data-refresh-account]").click()
    expect(row.locator("[data-refresh-account]")).to_have_attribute("aria-busy", "true")
    transport.hold.add("passive")
    sibling = ui["fixture"]["status"]["profiles"]["profiles"][1]
    sibling["profile"]["display_name"] = "Roster changed during provider refresh"
    sibling["identity"]["label"] = "Roster changed during provider refresh"
    store_call(page, "s.observe()", wait=False)
    wait_pending(page, transport, "passive")
    # Provider completion must merge into current state without dating the roster/catalog.
    ui["wire"]["quota"]["snapshots"][0]["constraints"][0]["used_ratio"] = 0.17
    ui["pending_refresh"][0].fulfill(json=ui["wire"]["quota"])
    expect(row.locator(".resource-window").first).to_contain_text("17% used")
    transport.release("passive")
    expect(page.locator('.harness-account-row[data-profile="account-01"]')).to_contain_text("Roster changed during provider refresh")
    expect(row.locator(".resource-window").first).to_contain_text("17% used")
    assert ui["refresh_calls"] == [{"target": TARGET}]


def test_passive_omission_retains_resources_but_explicit_empty_replaces_them(resources_ui):
    from playwright.sync_api import expect

    ui = resources_ui
    transport = ObservationTransport(ui)
    page, row = open_accounts(ui, width=1440)
    row.locator("[data-resources]").click()
    transport.include_resources = False
    store_call(page, "s.observe()")
    expect(row).to_contain_text("24.60 USD")
    transport.include_resources = True
    ui["fixture"]["status"]["resources"] = []
    store_call(page, "s.observe()")
    expect(row).not_to_contain_text("24.60 USD")


def test_account_polling_stops_when_hidden_unmounted_and_disposed(resources_ui):
    ui = resources_ui
    transport = ObservationTransport(ui)
    page = ui["page"]
    page.clock.install()
    page, _ = open_accounts(ui, width=1440)
    store_call(page, "s.refresh()")  # Let initial full discovery settle and arm its timer first.
    page.clock.fast_forward(5100)
    page.wait_for_timeout(50)
    count = transport.calls.count("passive")
    assert count >= 2  # Initial observation and the visible-page poll.
    # Browser visibility is controlled, but the real document listener/predicate/timer runs.
    page.evaluate("Object.defineProperty(document, 'hidden', {configurable: true, get: () => true}); document.dispatchEvent(new Event('visibilitychange'))")
    page.clock.fast_forward(15000)
    assert transport.calls.count("passive") == count
    page.evaluate("Object.defineProperty(document, 'hidden', {configurable: true, get: () => false}); document.dispatchEvent(new Event('visibilitychange'))")
    page.wait_for_timeout(20)
    assert transport.calls.count("passive") > count
    # A wide Settings home restores its last page; explicitly choose Ouroboros
    # so Accounts really unmounts instead of remounting beside the home rail.
    page.locator('[data-settings-overview-row]').click()
    page.locator('[data-settings-page="accounts/claude"]').wait_for(state="detached")
    assert store_call(page, "s.polling") is False
    count = transport.calls.count("passive")
    page.clock.fast_forward(15000)
    assert transport.calls.count("passive") == count
    store_call(page, "s.dispose()")
    assert store_call(page, "s.polling") is False
    page.clock.fast_forward(15000)
    assert transport.calls.count("passive") == count
    assert not any("/wake" in path or "/login" in path for path, _ in ui["posts"])


@pytest.mark.parametrize("full_code", [200, 503])
def test_activation_waits_for_fresh_full_before_any_daemon_write(resources_ui, full_code):
    ui = resources_ui
    transport = ObservationTransport(ui)
    page, _ = open_accounts(ui, width=1440)
    # Remember an installed-but-idle daemon without triggering an activation.
    ui["fixture"]["status"]["daemon"] = {"state": "stale", "runtime": {"state": "ready"}}
    store_call(page, "s.refresh()")
    ui["fixture"]["status"]["daemon"] = {"state": "running", "runtime": {"state": "ready"}}
    transport.full_status = full_code
    transport.hold.add("full")
    page.evaluate("window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', {detail: {path: 'accounts/claude'}}))")
    wait_pending(page, transport, "full")
    assert not any(path == "/api/claudexor/wake" for path, _ in ui["posts"])
    transport.release("full")
    page.wait_for_function("async () => !(await import('/static/modules/claudexor_status_store.js')).claudexorStatus.loading")
    assert not any(path == "/api/claudexor/wake" for path, _ in ui["posts"])
    assert store_call(page, "s.snapshot.daemon.state") == ("running" if full_code == 200 else "stale")


def test_failed_switch_restores_saved_state_and_keeps_its_error(resources_ui):
    from playwright.sync_api import expect

    ui = resources_ui
    transport = ObservationTransport(ui)
    page, row = open_accounts(ui, width=1440)
    transport.hold.add("patch")
    row.locator(".ui-switch-track").click()
    wait_pending(page, transport, "patch")
    switch = row.locator("[data-harness-toggle]")
    expect(switch).to_be_checked()
    expect(switch).to_be_disabled()
    route, _, _ = transport.pending["patch"].pop()
    route.fulfill(status=503, json={"error": "Fixture write refused"})
    expect(switch).not_to_be_checked()
    expect(switch).to_be_enabled()
    expect(page.locator("#page-settings")).to_contain_text("Fixture write refused")
    assert transport.calls.count("patch") == 1


def test_dispose_ignores_already_dispatched_passive_reply(resources_ui):
    ui = resources_ui
    transport = ObservationTransport(ui)
    page, _ = open_accounts(ui, width=1440)
    store_call(page, "s.refresh()")
    before = store_call(page, "({generation: s.generation, snapshot: s.snapshot})")
    transport.hold.add("passive")
    ui["fixture"]["status"]["profiles"]["profiles"] = []
    page.evaluate("async () => { const {claudexorStatus: s} = await import('/static/modules/claudexor_status_store.js'); "
                  "s.observe().then(() => window.__lateObservationSettled = true); }")
    wait_pending(page, transport, "passive")
    store_call(page, "s.dispose()")
    transport.release("passive")
    page.wait_for_function("window.__lateObservationSettled === true")
    assert store_call(page, "({generation: s.generation, snapshot: s.snapshot})") == before
    assert store_call(page, "s.polling") is False


def test_visible_program_inspection_runs_while_full_discovery_waits(resources_ui):
    ui = resources_ui
    transport = ObservationTransport(ui)
    page = ui['page']
    page.clock.install()
    page, _ = open_accounts(ui, width=1440)
    store_call(page, 's.refresh()')
    inventory_path = '/api/claudexor/maintenance/harnesses'

    def inspections():
        return sum(urlparse(url).path == inventory_path for url in ui['reads'])

    before = inspections()
    transport.hold.add('full')
    # The owner's showing inspects the programs at once, while its full discovery is held.
    page.evaluate("window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', {detail: {path: 'accounts/claude'}}))")
    wait_pending(page, transport, 'full')
    page.wait_for_timeout(50)
    shown = inspections()
    assert shown > before, 'visible program maintenance must not await slow full discovery'
    # Owner choice 2026-11-14: ticks never inspect, neither the 5 s observation nor the 120 s discovery.
    page.clock.fast_forward(121000)
    store_call(page, 's.observe()')
    page.wait_for_timeout(50)
    assert inspections() == shown, 'a status tick inspected the programs'
    transport.release('full')


@pytest.mark.parametrize("answer", ["lost", "refused"])
def test_lost_switch_answer_is_unconfirmed_and_a_refusal_stays_an_error(resources_ui, answer):
    import re

    from playwright.sync_api import expect

    ui = resources_ui
    transport = ObservationTransport(ui)
    page, row = open_accounts(ui, width=1440)
    switch, note = row.locator("[data-harness-toggle]"), page.locator("#harness-accounts-error")
    transport.hold.add("patch")
    row.locator(".ui-switch-track").click()
    wait_pending(page, transport, "patch")
    route, _, _ = transport.pending["patch"].pop()
    if answer == "lost":
        ui["fixture"]["status"]["profiles"]["profiles"][0]["profile"]["enabled"] = True  # the engine saved it
        route.abort("failed")
        expect(switch).to_be_checked()  # the follow-up read shows the saved state
        expect(note).to_have_text(re.compile(r'^Enabling "Personal default" is not confirmed: (Failed to fetch|Load failed)\.$'))
        expect(note).to_have_attribute("data-tone", "warn")
    else:
        # A 4xx is the one refusal proven unapplied: here the host's own gate answers before any handler
        # (ouroboros/server_auth.py); a 503 also follows an engine that kept the write (below).
        route.fulfill(status=401, json={"error": "Authentication required."})
        expect(switch).not_to_be_checked()
        expect(note).to_have_text('Could not enable "Personal default": Authentication required.')
        expect(note).to_have_attribute("data-tone", "error")
    expect(switch).to_be_enabled()
    frame_section(note)
    capture(page, f"switch-answer-{answer}")
    if answer == "lost":
        # The owner's Refresh is a new action: the line about the earlier request ends with it.
        page.locator("#btn-harness-refresh").click()
        expect(note).to_be_hidden()
        expect(switch).to_be_checked()  # what the engine holds, never inferred from the lost answer
    assert transport.calls.count("patch") == 1


# Evidence order: a reading is superseded only by a later event about the same thing, and an
# event dates only what it read. Passive reads are held so no poll tick settles the state shown.
SIBLING = '.harness-account-row[data-harness="claude"][data-profile="account-03"]'
STORE_ERROR = ("async () => (await import('/static/modules/claudexor_status_store.js'))"
               ".claudexorStatus.error !== ''")


def lose_status(page, transport):
    transport.full_status = 503
    transport.hold.add("passive")
    store_call(page, "s.refresh()")


def frame_rows(page, row, name, *, panel=True):
    """The open panel, then the rows beside it: a sibling's words are evidence too."""
    if panel:
        frame_section(row)
        capture(page, f"{name}-panel")
        row.locator("[data-resources]").click()
    frame_section(row)
    capture(page, f"{name}-rows")


def test_a_status_request_lost_before_a_refresh_completed_keeps_that_reading(resources_ui):
    from playwright.sync_api import expect

    ui = resources_ui
    transport = ObservationTransport(ui)
    page, row = open_accounts(ui, width=1440)
    meta, sibling = row.locator(".harness-account-meta"), page.locator(SIBLING).locator(".harness-account-meta")
    transport.passive_quota = "failed"  # an earlier failed quota facet
    store_call(page, "s.observe()")
    expect(meta).to_contain_text("Limits not checked")
    transport.full_status = 503
    transport.hold.update({"full", "passive"})
    store_call(page, "s.refresh()", wait=False)  # dispatched before the Refresh, lost after it
    wait_pending(page, transport, "full")
    row.locator("[data-resources]").click()
    row.locator("[data-refresh-account]").click()
    expect(meta).to_contain_text("Limit reached")
    transport.release("full")
    page.wait_for_function(STORE_ERROR)
    expect(meta).to_contain_text("Limit reached")
    expect(row.locator(".resource-window").first).not_to_contain_text("Last known")
    expect(sibling).to_contain_text("Limits not checked")
    frame_rows(page, row, "refresh-reading-outlives-older-lost-status")
    transport.hold.discard("full")
    store_call(page, "s.refresh()")  # a request lost after the reading ages it
    expect(meta).to_contain_text("Limits not checked")
    frame_rows(page, row, "refresh-reading-retired-by-later-lost-status", panel=False)
    row.locator("[data-resources]").click()
    expect(row.locator(".resource-window").first).to_contain_text("Last known")
    assert ui["refresh_calls"] == [{"target": TARGET}]


def test_one_accounts_refresh_dates_no_sibling_after_a_lost_status_request(resources_ui):
    from playwright.sync_api import expect

    ui = resources_ui
    transport = ObservationTransport(ui)
    page, row = open_accounts(ui, width=1440)
    meta, sibling = row.locator(".harness-account-meta"), page.locator(SIBLING).locator(".harness-account-meta")
    expect(sibling).to_contain_text("42% used")
    lose_status(page, transport)
    expect(sibling).to_contain_text("Limits not checked")
    row.locator("[data-resources]").click()
    row.locator("[data-refresh-account]").click()
    expect(meta).to_contain_text("Limit reached")
    expect(sibling).to_contain_text("Limits not checked")
    expect(sibling).not_to_contain_text("42% used")
    assert store_call(page, "s.reads") == {"catalog": "transport", "accounts": "transport", "quota": "transport"}
    frame_rows(page, row, "target-refresh-leaves-sibling-last-known")
    # Every account's Refresh is a quota read; it still reads no roster.
    store_call(page, "s.refreshResources()")
    expect(sibling).to_contain_text("42% used")
    assert store_call(page, "s.reads") == {"catalog": "transport", "accounts": "transport", "quota": "ok"}
    frame_rows(page, row, "every-account-refresh-dates-quota-not-roster", panel=False)
    assert ui["refresh_calls"] == [{"target": TARGET}, {}]


def test_a_refused_switch_is_no_roster_read_after_a_lost_status_request(resources_ui):
    from playwright.sync_api import expect

    ui = resources_ui
    transport = ObservationTransport(ui)
    page, row = open_accounts(ui, width=1440)
    badge, family = row.locator(".harness-account-main .ui-status"), page.locator(
        '[data-family="claude"] .agent-family-id .ui-status')
    expect(badge).to_have_count(0)  # a verified login is the quiet case
    lose_status(page, transport)
    expect(badge).to_have_text("Login verified — last known")
    expect(family).to_contain_text("last known")
    transport.hold.add("patch")
    row.locator(".ui-switch-track").click()
    wait_pending(page, transport, "patch")
    route, _, _ = transport.pending["patch"].pop()
    route.fulfill(status=401, json={"error": "Authentication required."})
    expect(page.locator("#harness-accounts-error")).to_have_text(
        'Could not enable "Personal default": Authentication required.')
    expect(row.locator("[data-harness-toggle]")).to_be_enabled()
    expect(badge).to_have_text("Login verified — last known")
    expect(family).to_contain_text("last known")
    assert store_call(page, "s.reads")["accounts"] == "transport"
    frame_section(page.locator("#harness-accounts-error"))
    capture(page, "refused-switch-keeps-roster-last-known")
    # The read begun after the write is the observation that ends it.
    transport.hold.discard("passive")
    wait_pending(page, transport, "passive")
    while transport.pending["passive"]:
        transport.release("passive")
    expect(badge).to_have_count(0)
    expect(family).not_to_contain_text("last known")
    assert store_call(page, "s.reads")["accounts"] == "ok"
    frame_section(row)
    capture(page, "read-after-switch-observes-roster")


@pytest.fixture
def engine_keeps_then_drops(resources_ui, monkeypatch, tmp_path):
    """The actual host proxy and gateway over a loopback engine that applies a row write and closes unanswered."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import claudexor_daemon as owned
    from ouroboros.config import CLAUDEXOR_MIN_VERSION, CLAUDEXOR_PROTOCOL_MAJOR
    from ouroboros.gateway.claudexor_accounts import api_claudexor_credential_profile

    rows, writes, answers = resources_ui["fixture"]["status"]["profiles"]["profiles"], [], []

    class Engine(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def body(self):
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length)) if length else None

        def do_POST(self):
            assert (self.path, self.body()["protocolMajor"]) == ("/v2/handshake", CLAUDEXOR_PROTOCOL_MAJOR)
            reply = json.dumps({"compatible": True, "protocolMajor": CLAUDEXOR_PROTOCOL_MAJOR,
                                "engine": {"version": CLAUDEXOR_MIN_VERSION}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def write(self):
            wanted, key = self.body(), tuple(self.path.removeprefix("/v2/credential-profiles/").split("/"))
            writes.append((self.command, *key, wanted))
            for row in [row for row in rows if (row["profile"]["harness_id"], row["profile"]["profile_id"]) == key]:
                if self.command == "DELETE":
                    rows.remove(row)
                else:
                    row["profile"]["enabled"] = wanted["enabled"]
            self.close_connection = True  # the engine kept it; its answer never leaves

        do_PATCH = do_DELETE = write

    engine = ThreadingHTTPServer(("127.0.0.1", 0), Engine)
    Thread(target=engine.serve_forever, daemon=True).start()
    home = tmp_path / "owned"
    (home / "daemon").mkdir(parents=True)
    (home / "token").write_text("fixture-token", encoding="utf-8")
    (home / "daemon" / "control-api.json").write_text(json.dumps(
        {"host": "127.0.0.1", "port": engine.server_address[1], "tokenPath": str(home / "token")}), encoding="utf-8")
    monkeypatch.setattr(owned, "owned_config_dir", lambda: home)
    client = TestClient(Starlette(routes=[Route("/api/claudexor/credential-profiles/{harness}/{profile_id}",
                                                api_claudexor_credential_profile, methods=["DELETE", "PATCH"])]))

    def forward(route):
        request = route.request
        result = client.request(request.method, urlparse(request.url).path, content=request.post_data_buffer,
                                headers={"Content-Type": "application/json"} if request.post_data_buffer else {})
        answers.append((request.method, result.status_code, result.json()))
        route.fulfill(status=result.status_code, content_type="application/json", body=result.text)

    # Bound after ObservationTransport, so this route answers the row writes.
    yield lambda page: page.route("**/api/claudexor/credential-profiles/**", forward), writes, answers
    client.close()
    engine.shutdown()
    engine.server_close()


@pytest.mark.parametrize("write", ["switch", "remove"])
def test_host_answer_after_the_engine_kept_a_write_claims_no_refusal(resources_ui, engine_keeps_then_drops, write):
    import re

    from playwright.sync_api import expect

    bind, writes, answers = engine_keeps_then_drops
    ui = resources_ui
    ObservationTransport(ui)
    page, row = open_accounts(ui, width=1440)
    bind(page)
    note, rows = page.locator("#harness-accounts-error"), page.locator(".harness-account-row[data-harness]")
    if write == "switch":
        row.locator(".ui-switch-track").click()
        expect(note).to_have_text(re.compile(r'^Enabling "Personal default" is not confirmed: daemon_unreachable: '
                                             r'Claudexor control PATCH /v2/credential-profiles/\.\.\. failed after '
                                             r'[0-9.]+s: RemoteProtocolError\.$'))
        expect(row.locator("[data-harness-toggle]")).to_be_checked()  # the read after it shows what the engine kept
        expect(row.locator("[data-harness-toggle]")).to_be_enabled()
        assert writes == [("PATCH", "claude", "claude-default", {"enabled": True})]
    else:
        row.locator("[data-resources]").click()
        row.locator("[data-harness-remove]").click()
        page.locator(".confirm-dialog [data-confirm-ok]").click()
        expect(note).to_have_text(re.compile(r'^Removing "Personal default" is not confirmed: daemon_unreachable: '
                                             r'Claudexor control DELETE /v2/credential-profiles/\.\.\. failed after '
                                             r'[0-9.]+s: RemoteProtocolError\.$'))
        expect(rows).to_have_count(19)  # the account the engine forgot is gone after the read, not "unchanged"
        assert writes == [("DELETE", "claude", "claude-default", None)]
    expect(note).to_have_attribute("data-tone", "warn")
    # The real proxy's 503: what the browser read was the host's own answer, sent once and never retried.
    assert [(method, status) for method, status, _ in answers] == [("PATCH" if write == "switch" else "DELETE", 503)]
    assert answers[0][2]["error"].startswith("daemon_unreachable: ")
    frame_section(note)
    capture(page, f"engine-kept-{write}")


def test_refused_removal_keeps_the_row_and_says_unchanged(resources_ui):
    from playwright.sync_api import expect

    ui = resources_ui
    ObservationTransport(ui)
    page, row = open_accounts(ui, width=1440)
    deletes = []

    def gate(route):
        deletes.append(route.request.method)
        route.fulfill(status=401, json={"error": "Authentication required."})  # the host's gate, before any handler

    page.route("**/api/claudexor/credential-profiles/**", gate)
    row.locator("[data-resources]").click()
    row.locator("[data-harness-remove]").click()
    page.locator(".confirm-dialog [data-confirm-ok]").click()
    note = page.locator("#harness-accounts-error")
    expect(note).to_have_text('Could not remove "Personal default": Authentication required. The account is unchanged.')
    expect(note).to_have_attribute("data-tone", "error")
    expect(page.locator(".harness-account-row[data-harness]")).to_have_count(20)
    assert deletes == ["DELETE"]
    frame_section(note)
    capture(page, "remove-refused")


def test_program_inspection_left_mid_read_is_replaced_on_return(resources_ui):
    from urllib.parse import parse_qs

    from playwright.sync_api import expect
    from tests.test_harness_maintenance_browser import _inventory_row

    ui = resources_ui
    page = ui["page"]
    entry = {**_inventory_row("claude"), "updateAvailable": "yes",
             "newest": {"status": "available", "version": "2.1.0", "observedAt": "2026-11-14T12:00:00Z"}}
    inspections, held = [], []

    def maintenance(route):
        if not urlparse(route.request.url).path.endswith("/harnesses"):
            return route.fulfill(status=404, json={"error": "fixture: no operation"})
        inspections.append(parse_qs(urlparse(route.request.url).query))
        if len(inspections) == 1:
            held.append(route)  # still inspecting when the owner leaves
        else:
            route.fulfill(json={"harnesses": [entry]})

    page.route("**/api/claudexor/maintenance/**", maintenance)
    page, _ = open_accounts(ui, width=1440)
    family = page.locator('.agent-family-card[data-family="claude"]')
    for _ in range(300):
        if held:
            break
        page.wait_for_timeout(10)
    assert inspections == [{"fresh": ["true"], "checkLatest": ["true"]}]

    def leave():
        # A wide Settings home restores its last page: choose Ouroboros so Accounts really unmounts.
        page.locator("[data-settings-overview-row]").click()
        page.locator('[data-settings-page="accounts/claude"]').wait_for(state="detached")

    leave()
    open_settings(page, "accounts/claude")
    expect(family.locator(".harness-maintenance-line")).to_contain_text("2.1.0 available")
    expect(family.get_by_role("button", name="Update", exact=True)).to_be_enabled()
    assert inspections[1] == {"fresh": ["true"], "checkLatest": ["true"]}, inspections
    held.pop().fulfill(json={"harnesses": [{**entry, "updateAvailable": "no"}]})  # the released view's late answer
    page.wait_for_timeout(200)
    expect(family.get_by_role("button", name="Update", exact=True)).to_be_enabled()
    frame_section(family.locator("[data-family-maintenance]"))
    capture(page, "program-inspection-replaced-on-return")
    # Each showing is the owner's act and asks again; the late answer above changed nothing.
    leave()
    open_settings(page, "accounts/claude")
    expect(family.get_by_role("button", name="Update", exact=True)).to_be_enabled()
    page.wait_for_timeout(500)
    assert inspections[2:] == [{"fresh": ["true"], "checkLatest": ["true"]}], inspections


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_first_screen_keeps_checking_until_the_first_full_read(resources_ui, theme):
    from playwright.sync_api import expect

    ui = resources_ui
    transport = ObservationTransport(ui)
    transport.hold.add("full")  # the first discovery probes every agent CLI
    page = ui["page"]
    page.set_viewport_size({"width": 1440, "height": 960})
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
    page.goto(ui["url"] + "/#settings")
    open_settings(page, "accounts/claude")
    wait_pending(page, transport, "full")
    for _ in range(300):
        if "passive" in transport.calls:
            break
        page.wait_for_timeout(10)
    page.wait_for_timeout(200)  # the passive answer has landed and repainted
    banner = page.locator("#agents-service-banner")
    checking = "Checking Claudexor… the first read probes each agent CLI and can take a minute or more."
    expect(banner).to_have_text(checking)
    expect(page.locator('.agent-family-card[data-family="claude"]')).to_contain_text("Checking…")
    expect(page.locator("#page-settings")).not_to_contain_text("were not read")
    capture(page, f"first-screen-{theme}-checking")
    while transport.pending["full"]:
        transport.release("full")
    expect(page.locator(".harness-account-row[data-harness]")).to_have_count(20)
    expect(banner).to_have_text("Claudexor ready")


ENGINE = os.environ.get("OUROBOROS_UI_BROWSER_ENGINE")
PROVIDER = '[data-settings-page="accounts/claude"]'


@pytest.fixture(params=[ENGINE] if ENGINE else ["webkit", "chromium"])
def more_ui(request, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    ui = request.getfixturevalue("resources_ui")
    page = ui["new_page"](viewport={"width": 1100, "height": 750}, device_scale_factor=2,
                          color_scheme="light", has_touch=False)
    ui["bind_resources"](page)
    origin = urlparse(ui["url"]).netloc
    page.route("**/*", lambda r: r.fallback() if urlparse(r.request.url).netloc == origin else r.abort())
    page.add_init_script("localStorage.setItem('ouroboros.theme', 'light')")
    ui["fixture"]["status"]["resources"][15]["resets"]["value"] = []
    budgets(ui, 0)
    return page, ui


def budgets(ui, count, *, number=15, start=0):
    budget = ui["wire"]["quota"]["resources"][0]["spending"]["value"][0]
    ui["fixture"]["status"]["resources"][number]["spending"]["value"] = [
        {**budget, "id": f"budget-{i}", "label": f"Workspace budget {i}"} for i in range(start, start + count)
    ]


def row(number):
    return f'{PROVIDER} [data-profile="account-{number:02}"]'


def center(button):
    box = button.bounding_box()
    return (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


def assert_at(button, point):
    assert button.evaluate("(b, [x, y]) => b.contains(document.elementFromPoint(x, y))", point), point
    assert center(button) == pytest.approx(point, abs=2)


def wait_at_y(page, selector, y):
    page.wait_for_function("""([selector, y]) => {
        const r = document.querySelector(selector).getBoundingClientRect();
        return Math.abs(r.top + r.height / 2 - y) <= 2;
    }""", arg=[selector, y])
    page.evaluate(SETTLED)


def wheel_to(page, selector, *, bottom_edge=False, inset=None):
    """Wheel inside the real scroller. The last row can only travel as far as its scroll range."""
    button = page.locator(selector)
    button.wait_for()
    view = button.evaluate("""(b, [edge, inset]) => {
        let box = b.parentElement;
        while (box && !(box.scrollHeight > box.clientHeight && /auto|scroll/.test(getComputedStyle(box).overflowY)))
            box = box.parentElement;
        box ||= document.scrollingElement;
        const r = box.getBoundingClientRect(), bRect = b.getBoundingClientRect();
        const y = (r.top + r.bottom) / 2, start = bRect.top + bRect.height / 2;
        const aim = edge ? Math.min(innerHeight - 48, r.bottom - 24) : inset == null ? y : r.top + inset;
        const delta = Math.round(start - aim);
        const end = Math.max(0, Math.min(box.scrollHeight - box.clientHeight, box.scrollTop + delta));
        return {x: (r.left + r.right) / 2, y, aim, delta, targetY: start - (end - box.scrollTop)};
    }""", [bottom_edge, inset])
    page.mouse.move(view["x"], view["y"])
    page.mouse.wheel(0, view["delta"])
    try:
        wait_at_y(page, selector, view["targetY"])
    except Exception as error:
        after = button.evaluate("""b => {
            const box = b.closest('[data-settings-page]');
            return [box?.scrollTop, box?.scrollHeight, box?.clientHeight];
        }""")
        raise AssertionError((selector, view, center(button), after)) from error
    point = center(button)
    assert_at(button, point)
    if bottom_edge:
        assert point[1] == pytest.approx(view["aim"], abs=2)
    return point


def observe(page, ui, percent, *, number=15):
    from playwright.sync_api import expect

    ui["fixture"]["status"]["quota"][number]["constraints"][0]["used_ratio"] = percent / 100
    store_call(page, "s.observe({afterCurrent: true})")
    expect(page.locator(row(number) + " .resource-window").first).to_contain_text(f"{percent}% used")
    page.evaluate(SETTLED)


@pytest.mark.parametrize("initial_budgets", [0, 12], ids=["grows-on-poll", "tall-at-open"])
def test_more_stays_under_pointer_through_poll_and_close(more_ui, initial_budgets):
    from playwright.sync_api import expect

    page, ui = more_ui
    budgets(ui, initial_budgets)
    page.goto(ui["url"] + "/")
    for door in ('#primary-sidebar [data-nav-page="settings"]', '[data-settings-home] [data-settings-open="accounts"]',
                 '[data-settings-page="accounts"] [data-entry-id="claude"] [data-entry-open]'):
        button = page.locator(door)
        button.wait_for()
        point = center(button)
        if not button.evaluate("(b, [x, y]) => b.contains(document.elementFromPoint(x, y))", point):
            point = wheel_to(page, door)
        page.mouse.click(*point)
    level = page.locator(PROVIDER)
    expect(level.locator(".harness-account-row")).to_have_count(20)
    page.locator(PROVIDER + "[data-settings-arrive]").wait_for(state="detached")
    selector = row(15) + " [data-resources]"
    more = page.locator(selector)
    point = wheel_to(page, selector, bottom_edge=True)
    viewport = level.evaluate("b => b.clientHeight")
    assert level.evaluate("b => b.scrollTop") > viewport
    # Deliver an actual scheduled answer after the press. No automation scrolls the click target into view.
    poll = "**/api/claudexor/status?view=quota"
    pending = []
    with page.expect_request(poll, timeout=15_000):
        page.route(poll, lambda route: pending.append(route), times=1)
    closed = level.evaluate("b => b.scrollHeight")
    page.mouse.move(*point)
    assert_at(more, point)
    page.mouse.click(*point)
    expect(more).to_have_attribute("aria-expanded", "true")
    expect(page.locator(row(16) + " [data-resources]")).to_have_attribute("aria-expanded", "false")
    page.evaluate(SETTLED)
    assert_at(more, point)
    opened = level.evaluate("b => b.scrollHeight")
    assert 0 < opened - closed and (opened - closed > viewport) == bool(initial_budgets)
    page.wait_for_timeout(1000)  # The reported jump happens when a late response arrives.
    assert_at(more, point)
    budgets(ui, 12)
    ui["fixture"]["status"]["quota"][15]["constraints"][0]["used_ratio"] = 0.61
    pending[0].fallback()
    expect(page.locator(row(15) + " .resource-window").first).to_contain_text("61% used")
    expect(page.locator(row(15) + " .account-resource-panel")).to_contain_text("Workspace budget 11")
    page.evaluate(SETTLED)
    if not initial_budgets:
        assert level.evaluate("b => b.scrollHeight") - opened > viewport
    assert_at(more, point)

    # The owner's next wheel chooses a new place, including for the following poll and close.
    moved = wheel_to(page, selector)
    assert moved[1] < point[1] - 40
    observe(page, ui, 37)
    assert_at(more, moved)
    page.mouse.click(*moved)
    expect(more).to_have_attribute("aria-expanded", "false")
    expect(page.locator(row(15) + " .account-resource-panel")).to_have_count(0)
    page.evaluate(SETTLED)
    assert_at(more, moved)


def test_poll_keeps_reading_row_and_offscreen_keyboard_focus(more_ui):
    from playwright.sync_api import expect

    page, ui = more_ui
    page.goto(ui["url"] + "/#settings")
    level = open_settings(page, "accounts/claude")
    page.locator(PROVIDER + "[data-settings-arrive]").wait_for(state="detached")
    more = page.locator(row(15) + " [data-resources]")
    tab = "Alt+Tab" if page.context.browser.browser_type.name == "webkit" else "Tab"
    page.locator(row(15)).get_by_role("switch").focus()
    page.keyboard.press(tab)
    expect(more).to_be_focused()
    for key, expanded in (("Space", "true"), ("Enter", "false"), ("Enter", "true")):
        point = center(more)
        page.keyboard.press(key)
        expect(more).to_have_attribute("aria-expanded", expanded)
        page.evaluate(SETTLED)
        assert_at(more, point)
        expect(more).to_be_focused()

    for number, percent in ((4, 37), (19, 61)):
        selector = row(number) + " [data-resources]"
        point = wheel_to(page, selector)
        top = level.evaluate("b => b.scrollTop")
        assert top > 0
        expect(more).not_to_be_in_viewport()
        expect(more).to_be_focused()
        if number == 19:
            assert level.evaluate("b => b.scrollHeight - b.clientHeight - b.scrollTop") <= 1
            budgets(ui, 12)  # The open panel now grows ABOVE the account being read.
        observe(page, ui, percent)
        assert_at(page.locator(selector), point)
        if number == 19:
            expect(page.locator(row(15))).to_contain_text("Workspace budget 11")
            assert level.evaluate("b => b.scrollTop - b.clientHeight") > top
        expect(more).not_to_be_in_viewport()
        expect(more).to_be_focused()

    page.keyboard.press(tab)
    login = page.locator(row(15) + " [data-harness-login]")
    expect(login).to_be_focused()
    expect(login).to_be_in_viewport()


def show_more_page(page, ui):
    page.goto(ui["url"] + "/#settings")
    level = open_settings(page, "accounts/claude")
    page.locator(PROVIDER + "[data-settings-arrive]").wait_for(state="detached")
    page.locator(row(19)).wait_for()
    return level


@pytest.mark.parametrize("gesture", ["click", "tab-space", "held-space"])
def test_poll_grows_upper_panel_without_moving_lower_more(more_ui, gesture):
    from playwright.sync_api import expect

    page, ui = more_ui
    page.set_viewport_size({"width": 1100, "height": 1180})
    level = show_more_page(page, ui)
    page.mouse.click(*wheel_to(page, row(15) + " [data-resources]"))
    lower = row(16) + " [data-resources]"
    point = wheel_to(page, lower, bottom_edge=True)
    more = page.locator(lower)
    if gesture == "click":
        page.mouse.click(*point)
    else:
        page.locator(row(16)).get_by_role("switch").focus()
        page.keyboard.press("Alt+Tab" if page.context.browser.browser_type.name == "webkit" else "Tab")
        expect(more).to_be_focused()
        point = center(more)
        assert_at(more, point)
        if gesture == "held-space":
            # Give up the tabbed reading position. The new press must choose the still-visible button itself.
            page.mouse.wheel(0, 36)
            wait_at_y(page, lower, point[1] - 36)
            point = center(more)
            page.keyboard.down("Space")
        expect(more).to_have_attribute("aria-expanded", "false")
    expect(page.locator(row(15) + " [data-resources]")).to_be_in_viewport()
    before = level.evaluate("b => b.scrollHeight")
    budgets(ui, 12)
    observe(page, ui, 61)
    assert level.evaluate("b => b.scrollHeight - b.clientHeight") > before
    print(f"{gesture} during growth: More y={point[1]:.1f} -> {center(more)[1]:.1f}")
    assert_at(more, point)
    if gesture != "click":
        if gesture == "tab-space":
            page.keyboard.down("Space")
        page.keyboard.up("Space")
        expect(more).to_be_focused()
    expect(level.locator(".account-resource-panel")).to_have_count(2)
    assert_at(more, point)
    page.mouse.click(*point)
    expect(page.locator(lower)).to_have_attribute("aria-expanded", "false")


def test_poll_wraps_metadata_above_pressed_more_on_phone(more_ui):
    page, ui = more_ui
    page.set_viewport_size({"width": 360, "height": 750})
    show_more_page(page, ui)
    selector = row(15) + " [data-resources]"
    point = wheel_to(page, selector)
    page.mouse.click(*point)
    meta = page.locator(row(15) + " .harness-account-meta")
    height = meta.bounding_box()["height"]
    offer = ui["wire"]["quota"]["resources"][0]["resets"]["value"][0]
    ui["fixture"]["status"]["resources"][15]["resets"]["value"] = [
        {**offer, "label": "Workspace session reset"}
    ]
    observe(page, ui, 37)
    assert meta.bounding_box()["height"] > height + 8
    assert_at(page.locator(selector), point)


def fit_closed_list(page, level, first):
    """Make the closed list fit the Settings scroller whatever the system's fonts (Ubuntu draws it taller than macOS).

    Measured in a short window where it overflows: the scroller must hold the whole list, and centring the first
    More must need no scrolling, so that choosing it later lands inside the natural range. The window is 1020 high
    unless the fonts need more.
    """
    content, first_center, height = level.evaluate("""(b, first) => {
        const r = b.getBoundingClientRect(), m = b.querySelector(first).getBoundingClientRect();
        return [b.scrollHeight, m.top + m.height / 2 - r.top + b.scrollTop, b.clientHeight];
    }""", first)
    assert content > height
    wanted = math.ceil(max(content, 2 * first_center + 2))
    window = max(1020, 700 + wanted - height)
    page.set_viewport_size({"width": 1100, "height": window})
    page.wait_for_function("([b, h]) => b.clientHeight === h", arg=[level.element_handle(), height + window - 700])
    page.evaluate(SETTLED)


@pytest.mark.parametrize("short_list", [False, True], ids=["long-list", "fits-when-closed"])
def test_last_more_closes_at_wheeled_position_until_reader_moves(more_ui, short_list):
    from playwright.sync_api import expect

    page, ui = more_ui
    if short_list:
        profiles = ui["fixture"]["status"]["profiles"]["profiles"]
        profiles[:] = [profiles[i] for i in (12, 14, 15, 17, 19)]
        page.set_viewport_size({"width": 1100, "height": 700})
    level = show_more_page(page, ui)
    if short_list:
        fit_closed_list(page, level, row(12) + " [data-resources]")
    selector = row(19) + " [data-resources]"
    last = page.locator(selector)
    natural = level.evaluate("b => b.scrollHeight")
    if short_list:
        assert natural == level.evaluate("b => b.clientHeight")
    start = center(last) if short_list else wheel_to(page, selector)
    assert_at(last, start)
    page.mouse.click(*start)
    expect(last).to_have_attribute("aria-expanded", "true")
    point = wheel_to(page, selector)
    assert level.evaluate("b => b.scrollTop + b.clientHeight") > natural + 60
    page.mouse.click(*point)
    expect(last).to_have_attribute("aria-expanded", "false")
    page.evaluate(SETTLED)
    assert_at(last, point)
    store_call(page, "s.observe({afterCurrent: true})")
    page.evaluate(SETTLED)
    assert_at(last, point)

    # A small wheel chooses a new place inside the extra room. Dropping all of it here would turn
    # this single step into a jump to the natural end.
    page.mouse.wheel(0, -32)
    moved = (point[0], point[1] + 32)
    wait_at_y(page, selector, moved[1])
    store_call(page, "s.observe({afterCurrent: true})")
    page.evaluate(SETTLED)
    assert_at(last, moved)
    # Once the reader chooses another account above, there is no extra room to keep.
    other = row(12) + " [data-resources]"
    chosen = wheel_to(page, other)
    assert level.evaluate("b => b.scrollHeight") == pytest.approx(natural, abs=2)
    store_call(page, "s.observe({afterCurrent: true})")
    page.evaluate(SETTLED)
    assert_at(page.locator(other), chosen)


@pytest.mark.parametrize("change", ["quota", "reorder", "remove"])
def test_more_press_survives_quota_answer_between_down_and_up(more_ui, change):
    from playwright.sync_api import expect

    page, ui = more_ui
    level = show_more_page(page, ui)
    page.mouse.click(*wheel_to(page, row(15) + " [data-resources]"))
    selector = row(16) + " [data-resources]"
    more = page.locator(selector)
    point = wheel_to(page, selector)
    page.mouse.move(*point)
    page.mouse.down()
    profiles = ui["fixture"]["status"]["profiles"]["profiles"]
    if change == "reorder":
        profiles.insert(14, profiles.pop(16))  # Move the pressed account past the expanded one.
    elif change == "remove":
        profiles.pop(16)
    observe(page, ui, 61)
    if change == "remove":
        expect(more).to_have_count(0)
        page.mouse.up()
        page.evaluate(SETTLED)
        expect(level.locator("[data-resources][aria-expanded=true]")).to_have_count(1)
        expect(page.locator(row(15) + " [data-resources]")).to_have_attribute("aria-expanded", "true")
        return
    if change == "reorder":
        expect(level.locator(".harness-account-row").nth(14)).to_have_attribute("data-profile", "account-16")
        assert_at(more, point)
    expect(more).to_have_attribute("aria-expanded", "false")
    page.mouse.up()
    expect(more).to_have_attribute("aria-expanded", "true")
    expect(level.locator(".account-resource-panel")).to_have_count(2)
    expect(page.locator(row(15) + " [data-resources]")).to_have_attribute("aria-expanded", "true")


def test_quota_repaint_keeps_rotation_switch_and_next_tab(more_ui):
    from playwright.sync_api import expect

    page, ui = more_ui
    show_more_page(page, ui)
    page.mouse.click(*wheel_to(page, row(15) + " [data-resources]"))
    account = page.locator(row(16))
    switch = account.get_by_role("switch")
    switch.focus()
    observe(page, ui, 37)
    expect(switch).to_be_focused()
    page.keyboard.press("Alt+Tab" if page.context.browser.browser_type.name == "webkit" else "Tab")
    expect(account.locator("[data-resources]")).to_be_focused()


@pytest.mark.parametrize("prior_focus", ["switch", "refresh", "more"])
def test_more_press_keeps_point_when_previous_focus_yields(more_ui, prior_focus):
    from playwright.sync_api import expect

    page, ui = more_ui
    page.set_viewport_size({"width": 1100, "height": 1180})
    level = show_more_page(page, ui)
    first = row(15) + " [data-resources]"
    page.mouse.click(*wheel_to(page, first))
    before = page.locator(row(15)).locator({
        "switch": "[data-harness-toggle]", "refresh": "[data-refresh-account]", "more": "[data-resources]",
    }[prior_focus])
    before.focus()
    selector = row(16) + " [data-resources]"
    point = wheel_to(page, selector, bottom_edge=True)
    expect(before).to_be_focused()
    expect(page.locator(first)).to_be_in_viewport()
    page.mouse.move(*point)
    page.mouse.down()
    # The previous focus yields to the press. Where it goes is the engine's platform convention: macOS WebKit
    # focuses the scroller (a button takes no focus from a pointer there), Chromium and Linux WebKit the pressed More.
    expect(before).not_to_be_focused()
    taken = """([level, more]) => [level, level.querySelector(more)].includes(document.activeElement)"""
    try:
        page.wait_for_function(taken, arg=[level.element_handle(), selector], timeout=5000)
    except Exception as error:
        active = page.evaluate("() => document.activeElement?.outerHTML.slice(0, 200)")
        raise AssertionError(f"the press left focus on {active}") from error
    budgets(ui, 12)
    observe(page, ui, 61)
    more = page.locator(selector)
    assert_at(more, point)
    page.mouse.up()
    expect(more).to_have_attribute("aria-expanded", "true")
    assert_at(more, point)


def test_more_press_survives_sign_in_reorder_to_first(more_ui):
    from playwright.sync_api import expect

    page, ui = more_ui
    level = show_more_page(page, ui)
    selector = row(16) + " [data-resources]"
    more = page.locator(selector)
    point = wheel_to(page, selector, bottom_edge=True)
    page.mouse.move(*point)
    page.mouse.down()
    ui["fixture"]["status"]["profiles"]["profiles"][16]["status"]["verification"] = "failed"
    store_call(page, "s.observe({afterCurrent: true})")
    expect(level.locator(".harness-account-row").first).to_have_attribute("data-profile", "account-16")
    expect(page.locator(row(16))).to_have_attribute("data-tone", "warn")
    page.evaluate(SETTLED)
    assert_at(more, point)
    page.mouse.up()
    expect(more).to_have_attribute("aria-expanded", "true")
    assert_at(more, point)
    # Move towards the natural top. If range was borrowed for the press, it is not a new landing position.
    capture(page, f"more-reorder-{page.context.browser.browser_type.name}-held")
    page.mouse.wheel(0, -32)
    page.wait_for_function("s => document.querySelector(s).scrollTop <= 1", arg=PROVIDER)
    page.evaluate(SETTLED)
    assert level.evaluate("b => b.scrollTop") == pytest.approx(0, abs=1)
    assert center(more)[1] < point[1] - 60
    capture(page, f"more-reorder-{page.context.browser.browser_type.name}-released")
    moved = wheel_to(page, row(12) + " [data-resources]")
    store_call(page, "s.observe({afterCurrent: true})")
    page.evaluate(SETTLED)
    assert_at(page.locator(row(12) + " [data-resources]"), moved)


def test_passive_shrink_keeps_last_reading_row_past_natural_end(more_ui):
    from playwright.sync_api import expect

    page, ui = more_ui
    page.set_viewport_size({"width": 1100, "height": 1080})
    ui["fixture"]["status"]["resources"][19]["resets"]["value"] = []
    budgets(ui, 18, number=19)
    level = show_more_page(page, ui)
    selector = row(19) + " [data-resources]"
    more = page.locator(selector)
    page.mouse.click(*wheel_to(page, selector))
    expect(more).to_have_attribute("aria-expanded", "true")
    point = wheel_to(page, selector, inset=160)
    budgets(ui, 0, number=19)
    observe(page, ui, 61, number=19)
    expect(page.locator(row(19))).not_to_contain_text("Workspace budget")
    assert_at(more, point)
    page.mouse.wheel(0, -32)
    wait_at_y(page, selector, point[1] + 32)
    # Leave the borrowed range and measure that the formerly short panel has its natural end again.
    other = row(9) + " [data-resources]"
    wheel_to(page, other)
    natural = level.evaluate("b => b.scrollHeight")
    end = wheel_to(page, selector, inset=160)
    assert level.evaluate("b => b.scrollHeight") == natural
    assert level.evaluate("b => b.scrollHeight - b.clientHeight - b.scrollTop") <= 1
    assert end[1] > point[1] + 32


def test_passive_growth_keeps_visible_budget_inside_resources(more_ui):
    from playwright.sync_api import expect

    page, ui = more_ui
    budgets(ui, 18)
    level = show_more_page(page, ui)
    page.mouse.click(*wheel_to(page, row(15) + " [data-resources]"))
    panel = page.locator(row(15) + " .account-resource-panel")
    selector = row(15) + ' [data-resource-reading="spending:budget-10"] dl > :first-child dt'
    point = wheel_to(page, selector)
    budget = panel.get_by_text("Workspace budget 10", exact=True)
    assert_at(budget, point)
    assert level.locator("[data-resources]").evaluate_all("""buttons => buttons.every(b => {
        const view = b.closest('[data-settings-page]').getBoundingClientRect();
        const r = b.getBoundingClientRect();
        return r.bottom <= view.top || r.top >= view.bottom;
    })""")
    budgets(ui, 23, start=-5)
    observe(page, ui, 61)
    expect(panel).to_contain_text("Workspace budget -5")
    assert_at(budget, point)
    capture(page, f"reading-budget-{page.context.browser.browser_type.name}")


def test_pointed_balance_survives_another_balance_arriving_above(more_ui):
    from playwright.sync_api import expect

    page, ui = more_ui
    fixture = setup_browser.WEB.parent / "tests/fixtures/claudexor_resources_freshness_cursor.json"
    balances = json.loads(fixture.read_text(encoding="utf-8"))["resources"][0]["balances"]["value"]
    resource = ui["fixture"]["status"]["resources"][15]
    resource["balances"]["value"] = balances[1:]
    show_more_page(page, ui)
    page.mouse.click(*wheel_to(page, row(15) + " [data-resources]"))
    point = wheel_to(page, row(15) + " .resource-section:nth-child(2) > dl:first-of-type > :first-child dt")
    balance = page.locator(row(15)).get_by_text("Personal on-demand remaining", exact=True)
    assert_at(balance, point)
    page.mouse.move(*point)
    resource["balances"]["value"] = balances
    observe(page, ui, 61)
    expect(page.locator(row(15))).to_contain_text("Personal included remaining")
    print(f"balance y={point[1]:.1f} -> {center(balance)[1]:.1f}")
    assert_at(balance, point)


def test_quota_glance_change_does_not_move_open_more(more_ui):
    from playwright.sync_api import expect

    page, ui = more_ui
    transport = ObservationTransport({**ui, "page": page})
    level = show_more_page(page, ui)
    glance = level.locator("[data-provider-quota]")
    expect(glance).to_be_visible()
    selector = row(15) + " [data-resources]"
    more = page.locator(selector)
    point = wheel_to(page, selector, bottom_edge=True)
    page.mouse.click(*point)
    expect(more).to_have_attribute("aria-expanded", "true")
    page.evaluate(SETTLED)
    assert_at(more, point)
    capture(page, "quota-glance-before")

    positions = []
    for quota in ("failed", "ok") * 24:
        transport.passive_quota = quota
        store_call(page, "s.observe({afterCurrent: true})")
        if quota == "failed":
            expect(glance).to_be_hidden()
        else:
            expect(glance).to_be_visible()
        page.evaluate(SETTLED)
        positions.append(center(more)[1])
    print(f"Repeated quota reads: More y={point[1]:.2f} -> {positions[-1]:.2f}")
    assert positions == pytest.approx([point[1]] * len(positions), abs=2)
    assert_at(more, point)
    page.mouse.click(*point)
    expect(more).to_have_attribute("aria-expanded", "false")


@pytest.mark.parametrize("intent", ["reading-edge", "pointed-more", "wheel-during-space", "restored-focus"])
def test_resource_before_next_more(more_ui, intent):
    from playwright.sync_api import expect

    page, ui = more_ui
    page.set_viewport_size({"width": 1100, "height": 900})
    budgets(ui, 12)
    spending = ui["fixture"]["status"]["resources"][15]["spending"]
    spending["value"][0]["label"] = spending["value"][9]["label"]
    level = show_more_page(page, ui)
    page.mouse.click(*wheel_to(page, row(15) + " [data-resources]"))
    next_more = page.locator(row(16) + " [data-resources]")
    if intent == "wheel-during-space":
        next_more.focus()
        page.keyboard.down("Space")
    selector = row(15) + ' [data-resource-reading="spending:budget-9"] dl > :first-child dt'
    point = wheel_to(page, selector, inset=36)
    assert level.evaluate("b => b.scrollHeight - b.clientHeight - b.scrollTop") > 1
    budget = page.locator(selector)
    expect(next_more).to_be_in_viewport()
    next_point = center(next_more)
    before_next = next_point[1]
    assert before_next > point[1] + 200
    expect(page.locator(row(15)).get_by_text("Workspace budget 9", exact=True)).to_have_count(2)
    if intent == "pointed-more":
        assert_at(next_more, next_point)
        page.mouse.move(*next_point)
    elif intent == "wheel-during-space":
        page.keyboard.up("Space")
        expect(next_more).to_have_attribute("aria-expanded", "true")
        expect(next_more).to_be_focused()
    elif intent == "restored-focus":
        next_more.evaluate("async el => (await import('/static/modules/scroll_fade.js')).restoreFocusWithoutReveal(el)")
        expect(next_more).to_be_focused()

    old = spending["value"]
    budgets(ui, 6, start=len(old))
    spending["value"] = old + spending["value"]
    observe(page, ui, 61)
    expect(page.locator(row(15))).to_contain_text("Workspace budget 17")
    print(f"growth below budget: budget y={point[1]:.1f} -> {center(budget)[1]:.1f}; "
          f"next More y={before_next:.1f} -> {center(next_more)[1]:.1f}")
    if intent == "pointed-more":
        assert_at(next_more, next_point)
        page.mouse.click(*next_point)
        expect(next_more).to_have_attribute("aria-expanded", "true")
    else:
        assert_at(budget, point)
        expect(next_more).not_to_be_in_viewport()


@pytest.mark.parametrize("part", ["label", "renewal"])
def test_pointed_resource_wins_over_natural_end(more_ui, part):
    from playwright.sync_api import expect

    page, ui = more_ui
    # At the natural end a lower budget and the last More share the view,
    # but the expanded panel's own More is above it.
    budgets(ui, 8, number=18)
    ui["fixture"]["status"]["resources"][18]["resets"]["value"] = []
    level = show_more_page(page, ui)
    more = row(18) + " [data-resources]"
    page.mouse.click(*wheel_to(page, more))
    last = row(19) + " [data-resources]"
    last_point = wheel_to(page, last)
    assert level.evaluate("b => b.scrollHeight - b.clientHeight - b.scrollTop") <= 1
    expect(page.locator(more)).not_to_be_in_viewport()
    # Use the exact stable key; a sibling has the same displayed name.
    spending = ui["fixture"]["status"]["resources"][18]["spending"]["value"]
    spending[6]["label"] = spending[7]["label"]
    observe(page, ui, 37, number=18)
    item = row(18) + ' [data-resource-reading="spending:budget-7"]'
    budget = page.locator(item + " dt").first
    if part == "renewal":
        budget = budget.locator("xpath=ancestor::dl/following-sibling::*[1]")
    point = center(budget)
    assert_at(budget, point)
    page.mouse.move(*point)
    budgets(ui, 6, number=18, start=len(spending))
    resource = ui["fixture"]["status"]["resources"][18]["spending"]
    resource["value"] = spending + resource["value"]
    observe(page, ui, 61, number=18)
    expect(page.locator(row(18))).to_contain_text("Workspace budget 11")
    print(f"growth below pointed budget: budget y={point[1]:.1f} -> {center(budget)[1]:.1f}")
    assert_at(budget, point)
    assert center(page.locator(last))[1] > last_point[1] + 200


def test_offscreen_more_activation_keeps_visible_reader_and_focus(more_ui):
    from playwright.sync_api import expect

    page, ui = more_ui
    budgets(ui, 18)
    level = show_more_page(page, ui)
    more = page.locator(row(15) + " [data-resources]")
    more.focus()
    page.keyboard.press("Space")
    expect(more).to_have_attribute("aria-expanded", "true")
    last = row(19) + " [data-resources]"
    point = wheel_to(page, last)
    assert level.evaluate("b => b.scrollHeight - b.clientHeight - b.scrollTop") <= 1
    expect(more).to_be_focused()
    expect(more).not_to_be_in_viewport()
    page.keyboard.press("Space")
    expect(more).to_have_attribute("aria-expanded", "false")
    page.evaluate(SETTLED)
    print(f"offscreen Space: last More y={point[1]:.1f} -> {center(page.locator(last))[1]:.1f}")
    assert_at(page.locator(last), point)
    expect(more).to_be_focused()
    page.keyboard.press("Alt+Tab" if page.context.browser.browser_type.name == "webkit" else "Tab")
    expect(page.locator(row(15) + " [data-harness-toggle]")).not_to_be_focused()
    expect(page.locator(row(16) + " [data-harness-toggle]")).to_be_focused()
    expect(page.locator(row(16) + " [data-harness-toggle]")).to_be_in_viewport()


SIGN_IN = row(15) + " [data-harness-login]"
LOGIN_CARD = PROVIDER + ' [data-family-login="claude"] [data-login-card]'


REFUSED_CANCEL = {"status": 409, "json": {"error": "Synthetic release not proven"}}


def sign_in_job(page, *, hold_create=False, cancels=None):
    """A synthetic sign-in job. Its cancel is refused, as for a process whose stop is not proven.

    Given a `cancels` list, each cancel waits there for the test to refuse it."""
    answer = {"job_id": "fixture", "job": {"state": "waiting_for_input", "harness": "claude"},
              "deviceCode": {"flow": "oauth_url_input", "verificationUrl": "https://example.test/login"}}
    creates, polls = [], []

    def create(route):
        creates.append(route)
        if not hold_create:
            route.fulfill(json=answer)

    def job(route):
        if route.request.method == "DELETE":
            if cancels is None:
                route.fulfill(**REFUSED_CANCEL)
            else:
                cancels.append(route)
        elif polls:
            route.fulfill(json=answer)
        else:
            polls.append(route)  # The first poll waits for the test; later ones answer.

    page.route("**/api/claudexor/login", create)
    page.route("**/api/claudexor/login/fixture", job)

    def polled_code():
        """The provider's next answer adds a code, so the open card grows by a line."""
        for _ in range(300):
            if polls:
                break
            page.wait_for_timeout(10)
        answer["deviceCode"]["userCode"] = "TEST-CODE"
        polls[0].fulfill(json=answer)
        page.locator(LOGIN_CARD + " [data-device-code]").wait_for()
        page.evaluate(SETTLED)

    return answer, creates, polled_code


def press_sign_in_again(page):
    """Open the late account's More and press its Sign in again with the mouse; nothing else is touched."""
    from playwright.sync_api import expect

    page.mouse.click(*wheel_to(page, row(15) + " [data-resources]"))
    button = page.locator(SIGN_IN)
    expect(button).to_have_text("Sign in again")
    point = center(button)
    assert_at(button, point)
    page.mouse.click(*point)
    return point


def card_in_view(page, level):
    """The whole card lies inside the list's box; returns its top for later comparison."""
    from playwright.sync_api import expect

    card = page.locator(LOGIN_CARD)
    expect(card).to_be_visible()
    page.evaluate(SETTLED)
    box, view = card.bounding_box(), level.bounding_box()
    # A browser rounds the shortest scroll to a whole pixel.
    assert view["y"] - 1 <= box["y"] and box["y"] + box["height"] <= view["y"] + view["height"] + 1, (box, view)
    expect(card.locator("[data-login-dismiss]")).to_be_in_viewport(ratio=1)
    return box["y"]


@pytest.mark.parametrize("reader", ["waits", "wheels-away", "tabs-back"])
def test_sign_in_again_shows_its_card_unless_the_reader_wheeled_away(more_ui, reader):
    from playwright.sync_api import expect

    page, ui = more_ui
    answer, creates, polled_code = sign_in_job(page, hold_create=reader == "wheels-away")
    level = show_more_page(page, ui)
    card = page.locator(LOGIN_CARD)
    if reader == "tabs-back":
        # Keyboard alone: Space opens More, Tab reaches Sign in again, Space commands it.
        tab = "Alt+Tab" if page.context.browser.browser_type.name == "webkit" else "Tab"
        button = page.locator(SIGN_IN)
        page.locator(row(15) + " [data-resources]").focus()
        page.keyboard.press("Space")
        page.keyboard.press(tab)
        expect(button).to_be_focused()
        page.keyboard.press("Space")
        expect(card.locator("[data-login-code-input]")).to_be_visible()
        top = card_in_view(page, level)
        expect(button).to_be_focused()
        expect(button).not_to_be_in_viewport()
        polled_code()
        assert card_in_view(page, level) == pytest.approx(top, abs=1)
        # The next Tab leaves the form for the account's own row; ordinary answers keep the owner there.
        page.keyboard.press(tab)
        remove = page.locator(row(15) + " [data-harness-remove]")
        expect(remove).to_be_focused()
        expect(remove).to_be_in_viewport()
        page.evaluate(SETTLED)
        point = center(remove)
        observe(page, ui, 61)
        print(f"tabbed back: Remove y={point[1]:.1f} -> {center(remove)[1]:.1f}")
        assert center(remove) == pytest.approx(point, abs=1)
        expect(remove).to_be_focused()
        expect(card).not_to_be_in_viewport()
        return
    pressed = press_sign_in_again(page)
    if reader == "waits":
        # The create answers at once and the owner touches nothing more: the form and its actions are in view.
        expect(card.locator("[data-login-code-input]")).to_be_visible()
        top = card_in_view(page, level)
        expect(card.locator("[data-login-code-input]")).to_be_in_viewport(ratio=1)
        print(f"prompt sign-in: pressed y={pressed[1]:.1f}, card y={top:.1f}")
        height = card.bounding_box()["height"]
        polled_code()
        assert card.bounding_box()["height"] > height + 20
        observe(page, ui, 61)
        assert card_in_view(page, level) == pytest.approx(top, abs=1)
        expect(card.locator("[data-login-code-submit]")).to_be_in_viewport(ratio=1)
        return
    # The opening card is shown; the owner then wheels to another account before the create answers.
    expect(card).to_be_visible()
    expect(card.locator("[data-login-code-input]")).to_have_count(0)
    card_in_view(page, level)
    other = row(9) + " [data-resources]"
    point = wheel_to(page, other)
    expect(card).not_to_be_in_viewport()
    creates[0].fulfill(json=answer)
    expect(card.locator("[data-login-code-input]")).to_be_attached()
    page.evaluate(SETTLED)
    assert_at(page.locator(other), point)
    polled_code()
    observe(page, ui, 61)
    print(f"wheeled away: More y={point[1]:.1f} -> {center(page.locator(other))[1]:.1f}")
    assert_at(page.locator(other), point)
    expect(card).not_to_be_in_viewport()


def test_refused_cancel_shows_the_open_sign_in_again(more_ui):
    from playwright.sync_api import expect

    page, ui = more_ui
    _, creates, polled_code = sign_in_job(page)
    level = show_more_page(page, ui)
    press_sign_in_again(page)
    card = page.locator(LOGIN_CARD)
    expect(card.locator("[data-login-code-input]")).to_be_visible()
    # Back at the account, an ordinary answer leaves the owner there.
    button = page.locator(SIGN_IN)
    point = wheel_to(page, SIGN_IN)
    expect(card).not_to_be_in_viewport()
    polled_code()
    observe(page, ui, 61)
    assert_at(button, point)
    # The second press cannot stop the open sign-in. Its card says so where the owner can read it.
    page.mouse.click(*point)
    expect(card).to_contain_text("Could not cancel the active sign-in")
    top = card_in_view(page, level)
    print(f"refused cancel: pressed y={point[1]:.1f}, card y={top:.1f}")
    observe(page, ui, 37)
    assert card_in_view(page, level) == pytest.approx(top, abs=1)
    assert len(creates) == 1


def test_add_account_form_grows_into_view_below_the_fold(more_ui):
    from playwright.sync_api import expect

    page, ui = more_ui
    answer, creates, _ = sign_in_job(page, hold_create=True)
    level = show_more_page(page, ui)
    add = page.locator(PROVIDER + ' [data-family="claude"] [data-family-add]')
    # A short window ends just under the Accounts head: the card will open below the fold.
    head = add.bounding_box()
    page.set_viewport_size({"width": 1100, "height": round(head["y"] + head["height"] + 16)})
    page.evaluate(SETTLED)
    point = center(add)
    assert_at(add, point)
    assert level.evaluate("b => b.scrollTop") == 0 and level.bounding_box()["height"] > 300
    page.mouse.click(*point)
    dialog = page.get_by_role("dialog")
    dialog.get_by_role("textbox").fill("fixture-account")
    dialog.locator("[data-confirm-ok]").click()
    card = page.locator(LOGIN_CARD)
    opening = card_in_view(page, level)
    expect(card.locator("[data-login-code-input]")).to_have_count(0)
    height = card.bounding_box()["height"]
    creates[0].fulfill(json=answer)
    expect(card.locator("[data-login-code-input]")).to_be_visible()
    grown = card_in_view(page, level)
    assert card.bounding_box()["height"] > height + 60
    expect(card.locator("[data-login-code-submit]")).to_be_in_viewport(ratio=1)
    print(f"add below the fold: card y={opening:.1f} -> {grown:.1f}, scrolled {level.evaluate('b => b.scrollTop')}")
    assert creates[0].request.post_data_json == {"harness": "claude", "login_flow": "device_auth",
                                                "profile_id": "fixture-account"}



def wait_for_count(page, items, count):
    for _ in range(300):
        if len(items) >= count:
            break
        page.wait_for_timeout(10)
    assert len(items) == count, items


@pytest.mark.parametrize("held", ["cancel", "create"])
def test_repeated_sign_in_again_shows_the_one_sign_in(more_ui, held):
    """The owner presses Sign in again, wheels back to it or waits, and presses it again before the
    first answer: one operation, and the latest press still brings its card into view."""
    from playwright.sync_api import expect

    page, ui = more_ui
    cancels = []
    answer, creates, _ = sign_in_job(page, hold_create=held == "create", cancels=cancels)
    level = show_more_page(page, ui)
    card = page.locator(LOGIN_CARD)
    press_sign_in_again(page)
    card_in_view(page, level)
    point = wheel_to(page, SIGN_IN)
    expect(card).not_to_be_in_viewport()
    if held == "cancel":
        # The open sign-in must stop before another starts; that stop is still unanswered.
        expect(card.locator("[data-login-code-input]")).to_be_visible()
        page.mouse.click(*point)
        wait_for_count(page, cancels, 1)
        page.evaluate(SETTLED)
        assert_at(page.locator(SIGN_IN), point)
    page.mouse.click(*point)
    if held == "cancel":
        page.wait_for_timeout(100)
        assert len(cancels) == 1
        cancels[0].fulfill(**REFUSED_CANCEL)
        expect(card).to_contain_text("Could not cancel the active sign-in")
    else:
        expect(card.locator("[data-login-code-input]")).to_have_count(0)
    top = card_in_view(page, level)
    print(f"repeated press ({held}): pressed y={point[1]:.1f}, card y={top:.1f}")
    if held == "create":
        creates[0].fulfill(json=answer)
        expect(card.locator("[data-login-code-input]")).to_be_visible()
        card_in_view(page, level)
    assert len(creates) == 1 and len(cancels) == (1 if held == "cancel" else 0)


@pytest.mark.parametrize("part", ["[data-resources]", ".ui-switch"], ids=["more", "rotation-switch"])
def test_pointing_at_another_account_outranks_a_pending_sign_in(more_ui, part):
    """While the open sign-in is being stopped, the owner moves to the next account up: to its More or to
    its rotation switch. The refusal that arrives later is shown in the card where it stands, and the page
    stays with the control they are about to press."""
    from playwright.sync_api import expect

    page, ui = more_ui
    cancels = []
    sign_in_job(page, cancels=cancels)
    level = show_more_page(page, ui)
    card = page.locator(LOGIN_CARD)
    press_sign_in_again(page)
    expect(card.locator("[data-login-code-input]")).to_be_visible()
    point = wheel_to(page, SIGN_IN)
    page.mouse.click(*point)
    wait_for_count(page, cancels, 1)
    other = page.locator(f"{row(14)} {part}")
    expect(other).to_be_in_viewport(ratio=1)
    target = center(other)
    page.mouse.move(*target, steps=4)
    page.evaluate(SETTLED)
    assert_at(other, target)
    before = level.evaluate("b => b.scrollTop")
    cancels[0].fulfill(**REFUSED_CANCEL)
    expect(card).to_contain_text("Could not cancel the active sign-in")
    page.evaluate(SETTLED)
    print(f"pointed away: scrollTop {before} -> {level.evaluate('b => b.scrollTop')}")
    assert_at(other, target)
    expect(card).not_to_be_in_viewport()


def test_moving_to_another_more_after_the_card_is_shown_ends_the_reveal(more_ui):
    """The sign-in card is already shown. The owner's pointer then really moves to another visible More; the
    card growing later is an ordinary repaint around what they chose."""
    page, ui = more_ui
    _, _, polled_code = sign_in_job(page)
    level = show_more_page(page, ui)
    press_sign_in_again(page)
    card = page.locator(LOGIN_CARD)
    card_in_view(page, level)
    profile = level.evaluate("""b => {
        const box = b.getBoundingClientRect();
        const full = n => n.getBoundingClientRect().top >= box.top && n.getBoundingClientRect().bottom <= box.bottom;
        return [...b.querySelectorAll('[data-resources]')].filter(full)
            .map(n => n.closest('[data-profile]').dataset.profile).find(id => id !== 'account-15');
    }""")
    other = page.locator(f'[data-profile="{profile}"] [data-resources]')
    target = center(other)
    page.mouse.move(*target, steps=4)
    page.evaluate(SETTLED)
    assert_at(other, target)
    height = card.bounding_box()["height"]
    polled_code()
    assert card.bounding_box()["height"] > height + 20
    page.evaluate(SETTLED)
    assert_at(other, target)


@pytest.mark.parametrize("drift", ["tremor", "row-gap"])
def test_shown_sign_in_card_outlasts_a_pointer_that_only_drifts(more_ui, drift):
    """Showing the card scrolled the page under a pointer that has not moved: another account lies there now.
    A two-pixel tremor, or a drift from the space between two rows into the next row, presses nothing and
    reaches no control, so the create's answer still grows the card where its actions can be read."""
    from playwright.sync_api import expect

    page, ui = more_ui
    answer, creates, _ = sign_in_job(page, hold_create=True)
    level = show_more_page(page, ui)
    x, y = press_sign_in_again(page)
    card = page.locator(LOGIN_CARD)
    card_in_view(page, level)
    expect(card.locator("[data-login-code-input]")).to_have_count(0)
    under = "([x, y]) => document.elementFromPoint(x, y)?.closest('[data-profile]')?.dataset.profile || ''"
    if drift == "row-gap":
        gap, y = level.evaluate("""(b, y) => {
            const rows = [...b.querySelectorAll('.harness-account-row')].map(r => r.getBoundingClientRect());
            const next = rows.findIndex((r, i) => i && r.top > y && rows[i - 1].bottom < r.top);
            return [(rows[next - 1].bottom + rows[next].top) / 2, rows[next].top];
        }""", y)
        page.mouse.move(x, gap)
        assert page.evaluate(under, [x, gap]) == ""
    page.mouse.move(x + 2, y + 2)
    rested = page.evaluate(under, [x + 2, y + 2])
    assert rested not in ("", "account-15"), rested
    page.evaluate(SETTLED)
    height = card.bounding_box()["height"]
    creates[0].fulfill(json=answer)
    expect(card.locator("[data-login-code-input]")).to_be_visible()
    top = card_in_view(page, level)
    print(f"{drift} over {rested}: card y={top:.1f}, height {height:.1f} -> {card.bounding_box()['height']:.1f}")
    assert card.bounding_box()["height"] > height + 60
    expect(card.locator("[data-login-code-submit]")).to_be_in_viewport(ratio=1)


def test_pending_refusal_and_status_promotion_keep_the_pointed_rotation_switch(more_ui):
    """During a pending stop the owner points at the neighbouring account's rotation switch. A status read
    that promotes that same account to the first row, and then the refusal, leave the switch under the pointer."""
    from playwright.sync_api import expect

    page, ui = more_ui
    cancels = []
    sign_in_job(page, cancels=cancels)
    level = show_more_page(page, ui)
    card = page.locator(LOGIN_CARD)
    press_sign_in_again(page)
    expect(card.locator("[data-login-code-input]")).to_be_visible()
    page.mouse.click(*wheel_to(page, SIGN_IN))
    wait_for_count(page, cancels, 1)
    switch = page.locator(f"{row(14)} .ui-switch")
    expect(switch).to_be_in_viewport(ratio=1)
    target = center(switch)
    page.mouse.move(*target, steps=4)
    page.evaluate(SETTLED)
    ui["fixture"]["status"]["profiles"]["profiles"][14]["status"]["verification"] = "failed"
    store_call(page, "s.observe({afterCurrent: true})")
    expect(level.locator(".harness-account-row").first).to_have_attribute("data-profile", "account-14")
    page.evaluate(SETTLED)
    assert_at(switch, target)
    cancels[0].fulfill(**REFUSED_CANCEL)
    expect(card).to_contain_text("Could not cancel the active sign-in")
    page.evaluate(SETTLED)
    assert_at(switch, target)


@pytest.mark.parametrize("change", ["wrap", "promotion"])
def test_rotation_switch_press_survives_an_answer_between_down_and_up(more_ui, change):
    """A deep account's rotation switch is pressed as its More is. Between mouse down and up an answer rewraps
    that row's own facts above the switch (a phone), or promotes the account to the first row. The switch
    stays under the pointer and the release is its one real click."""
    from playwright.sync_api import expect

    page, ui = more_ui
    patches = []
    page.route("**/api/claudexor/credential-profiles/**",
               lambda route: patches.append((urlparse(route.request.url).path, route.request.post_data_json)))
    if change == "wrap":
        page.set_viewport_size({"width": 360, "height": 750})
        ui["fixture"]["status"]["resources"][16]["resets"]["value"] = []
    level = show_more_page(page, ui)
    selector = row(16) + " .ui-switch"
    switch = page.locator(selector)
    point = wheel_to(page, selector, bottom_edge=True)
    assert level.evaluate("b => b.scrollTop") > 0
    page.mouse.move(*point)
    page.mouse.down()
    if change == "wrap":
        meta = row(16) + " .harness-account-meta"
        height = page.locator(meta).bounding_box()["height"]
        offer = ui["wire"]["quota"]["resources"][0]["resets"]["value"][0]
        ui["fixture"]["status"]["resources"][16]["resets"]["value"] = [{**offer, "label": "Workspace session reset"}]
        store_call(page, "s.observe({afterCurrent: true})")
        page.wait_for_function("([s, h]) => document.querySelector(s).getBoundingClientRect().height > h + 8",
                               arg=[meta, height])
    else:
        ui["fixture"]["status"]["profiles"]["profiles"][16]["status"]["verification"] = "failed"
        store_call(page, "s.observe({afterCurrent: true})")
        expect(level.locator(".harness-account-row").first).to_have_attribute("data-profile", "account-16")
    page.evaluate(SETTLED)
    print(f"switch held through {change}: y={point[1]:.1f} -> {center(switch)[1]:.1f}")
    assert_at(switch, point)
    assert patches == []
    page.mouse.up()
    wait_for_count(page, patches, 1)
    page.wait_for_timeout(200)
    assert patches == [("/api/claudexor/credential-profiles/claude/account-16", {"enabled": False})]
    assert_at(switch, point)


def test_pointing_at_the_more_a_repaint_kept_outranks_a_pending_sign_in(more_ui):
    """An ordinary answer during the held stop keeps the first More in view as the reading place. The owner
    then moves onto that very More in one step: their own choice, which the later refusal must not undo."""
    from playwright.sync_api import expect

    page, ui = more_ui
    cancels = []
    sign_in_job(page, cancels=cancels)
    level = show_more_page(page, ui)
    card = page.locator(LOGIN_CARD)
    press_sign_in_again(page)
    expect(card.locator("[data-login-code-input]")).to_be_visible()
    page.mouse.click(*wheel_to(page, SIGN_IN))
    wait_for_count(page, cancels, 1)
    observe(page, ui, 61)
    expect(card).not_to_be_in_viewport()
    # The first More with any part in view; the pointer aims at the middle of that part.
    profile, x, y = level.evaluate("""b => {
        const box = b.getBoundingClientRect();
        const more = [...b.querySelectorAll('[data-resources]')].find(n => n.getBoundingClientRect().bottom > box.top);
        const r = more.getBoundingClientRect(), top = Math.max(r.top, box.top);
        return [more.closest('[data-profile]').dataset.profile, r.x + r.width / 2, (top + r.bottom) / 2];
    }""")
    other = level.locator(f'[data-profile="{profile}"] [data-resources]')
    hit = "(b, [x, y]) => b.contains(document.elementFromPoint(x, y))"
    assert other.evaluate(hit, [x, y])
    top = other.bounding_box()["y"]
    page.mouse.move(x, y)
    page.evaluate(SETTLED)
    before = level.evaluate("b => b.scrollTop")
    cancels[0].fulfill(**REFUSED_CANCEL)
    expect(card).to_contain_text("Could not cancel the active sign-in")
    page.evaluate(SETTLED)
    print(f"pointed at kept {profile}: y={y:.1f}, More top {top:.1f} -> {other.bounding_box()['y']:.1f}, "
          f"scrollTop {before} -> {level.evaluate('b => b.scrollTop')}")
    assert other.evaluate(hit, [x, y])
    assert other.bounding_box()["y"] == pytest.approx(top, abs=1)
    expect(card).not_to_be_in_viewport()


def next_answer(page, polls):
    """Wait for the job's next poll (one every three seconds) and for its answer to be drawn."""
    seen = len(polls)
    for _ in range(800):
        if len(polls) > seen:
            break
        page.wait_for_timeout(10)
    assert len(polls) > seen
    page.wait_for_timeout(200)
    page.evaluate(SETTLED)


def test_sign_in_card_keeps_focus_and_press_across_job_polls(more_ui):
    """Tab from the typed code to Submit, then press it with the mouse, while the job keeps answering.
    An unchanged answer leaves the card alone; a changed one gives focus back to the same command."""
    from playwright.sync_api import expect

    page, ui = more_ui
    answer, _, polled_code = sign_in_job(page)
    polls, inputs = [], []
    page.route("**/api/claudexor/login/fixture", lambda r: (polls.append(1), r.fallback()))
    page.route("**/api/claudexor/login/fixture/input", lambda r: (inputs.append(r.request.post_data_json),
                                                                 r.fulfill(json={"ok": True})))
    show_more_page(page, ui)
    press_sign_in_again(page)
    card = page.locator(LOGIN_CARD)
    code, submit = card.locator("[data-login-code-input]"), card.locator("[data-login-code-submit]")
    polled_code()
    code.click()
    page.keyboard.type("AB-12")
    next_answer(page, polls)  # the typed value redraws the field once
    expect(code).to_be_focused()
    tab = "Alt+Tab" if page.context.browser.browser_type.name == "webkit" else "Tab"
    page.keyboard.press(tab)
    expect(submit).to_be_focused()
    next_answer(page, polls)
    expect(submit).to_be_focused()  # unchanged answer
    answer["deviceCode"]["userCode"] = "NEW-CODE"
    next_answer(page, polls)
    expect(card.locator("[data-device-code]")).to_have_text("NEW-CODE")
    expect(submit).to_be_focused()  # changed answer, same job and command
    point = center(submit)
    page.mouse.move(*point)
    page.mouse.down()
    next_answer(page, polls)
    page.mouse.up()
    wait_for_count(page, inputs, 1)
    assert inputs == [{"value": "AB-12"}]


def test_first_unchanged_poll_after_typing_cannot_land_inside_a_press_on_submit(more_ui):
    """Typing is no new answer: the card is not redrawn for it, so the first identical poll that follows
    (no render in between) leaves the held Submit where it is and the release is one real click."""
    page, ui = more_ui
    answer, _, polled_code = sign_in_job(page)
    polls, inputs = [], []
    page.route("**/api/claudexor/login/fixture", lambda r: (polls.append(1), r.fallback()))
    page.route("**/api/claudexor/login/fixture/input", lambda r: (inputs.append(r.request.post_data_json),
                                                                 r.fulfill(json={"ok": True})))
    show_more_page(page, ui)
    press_sign_in_again(page)
    card = page.locator(LOGIN_CARD)
    code, submit = card.locator("[data-login-code-input]"), card.locator("[data-login-code-submit]")
    polled_code()
    page.wait_for_timeout(300)
    seen = len(polls)
    code.click()
    page.keyboard.type("AB-12")
    node = submit.evaluate_handle("b => (window.__held = b)")
    page.mouse.move(*center(submit))
    page.mouse.down()
    for _ in range(800):
        if len(polls) > seen:
            break
        page.wait_for_timeout(10)
    assert len(polls) > seen, "the first poll after typing never came"
    page.wait_for_timeout(300)
    assert submit.evaluate("b => b === window.__held"), "the poll replaced the held Submit"
    page.mouse.up()
    wait_for_count(page, inputs, 1)
    assert inputs == [{"value": "AB-12"}]
    assert code.input_value() == "AB-12" and node


@pytest.mark.parametrize("advanced", [False, True], ids=["form", "advanced-just-opened"])
def test_changed_poll_during_a_press_on_submit_still_releases_one_real_click(more_ui, advanced):
    """Same job, same command, new words (a changed code line) arrive between mouse down and up. Opening
    Advanced first is the owner's own act on a disclosure the browser keeps: the first poll after it is still
    no new card, so the held Submit is the node that receives the release."""
    page, ui = more_ui
    answer, _, polled_code = sign_in_job(page)
    if advanced:
        answer.update(attach_command="fixture-attach --job fixture", disclosure_native=False)
    polls, inputs = [], []
    page.route("**/api/claudexor/login/fixture", lambda r: (polls.append(1), r.fallback()))
    page.route("**/api/claudexor/login/fixture/input", lambda r: (inputs.append(r.request.post_data_json),
                                                                 r.fulfill(json={"ok": True})))
    show_more_page(page, ui)
    press_sign_in_again(page)
    card = page.locator(LOGIN_CARD)
    code, submit = card.locator("[data-login-code-input]"), card.locator("[data-login-code-submit]")
    polled_code()
    page.wait_for_timeout(300)
    seen = len(polls)
    code.click()
    page.keyboard.type("AB-12")
    if advanced:
        details = card.locator("[data-login-advanced]")
        page.mouse.click(*center(details.locator("> summary")))
        page.wait_for_function("d => d.open", arg=details.element_handle())
        page.evaluate(SETTLED)
    point = center(submit)
    submit.evaluate("b => (window.__held = b)")
    page.mouse.move(*point)
    page.mouse.down()
    answer["deviceCode"]["userCode"] = "NEW-CODE"
    for _ in range(800):
        if len(polls) > seen:
            break
        page.wait_for_timeout(10)
    assert len(polls) == seen + 1, "the press must hold through exactly the first poll"
    page.wait_for_timeout(300)
    page.evaluate(SETTLED)
    assert card.locator("[data-device-code]").inner_text() == "NEW-CODE"
    assert_at(submit, point)
    assert submit.evaluate("b => b === window.__held"), "the poll replaced the held Submit"
    page.mouse.up()
    wait_for_count(page, inputs, 1)
    assert inputs == [{"value": "AB-12"}]
    if advanced:
        # The sent code is a new face: the form is rebuilt, and what the owner opened is given back to it.
        page.wait_for_function("""s => {
            const button = document.querySelector(s);
            return button && button !== window.__held && button.disabled;
        }""", arg=LOGIN_CARD + " [data-login-code-submit]")
        assert details.evaluate("d => d.open")


def test_job_settling_during_a_press_on_submit_never_sends_the_obsolete_code(more_ui):
    """The counterexample: the job itself settles between mouse down and up. The verdict replaces the
    answered form, so the pressed Submit is gone with the job that owned it and the release sends nothing."""
    from playwright.sync_api import expect

    page, ui = more_ui
    answer, _, polled_code = sign_in_job(page)
    polls, inputs = [], []
    page.route("**/api/claudexor/login/fixture", lambda r: (polls.append(1), r.fallback()))
    page.route("**/api/claudexor/login/fixture/input", lambda r: (inputs.append(r.request.post_data_json),
                                                                 r.fulfill(json={"ok": True})))
    show_more_page(page, ui)
    press_sign_in_again(page)
    card = page.locator(LOGIN_CARD)
    code, submit = card.locator("[data-login-code-input]"), card.locator("[data-login-code-submit]")
    polled_code()
    page.wait_for_timeout(300)
    seen = len(polls)
    code.click()
    page.keyboard.type("AB-12")
    page.mouse.move(*center(submit))
    page.mouse.down()
    answer["job"] = {"state": "succeeded", "harness": "claude"}
    for _ in range(800):
        if len(polls) > seen:
            break
        page.wait_for_timeout(10)
    assert len(polls) > seen
    page.wait_for_timeout(300)
    page.evaluate(SETTLED)
    expect(card.locator("[data-login-verdict]")).to_have_text("Signed in.")
    expect(card.locator("[data-login-code-submit]")).to_have_count(0)
    page.mouse.up()
    page.wait_for_timeout(300)
    page.evaluate(SETTLED)
    assert inputs == []


@pytest.mark.parametrize("answered", ["checking", "settled"])
def test_shorter_state_line_keeps_the_held_submit_under_the_pointer_on_a_phone(more_ui, answered):
    """A phone wraps the state line above the code field onto two lines. Submit is held three pixels above its
    lower edge when the same job starts checking the sign-in, and that line gets a line shorter. The same
    command stays under the pointer and the release is its one real click; a job that settled took the command
    with it, so there the release sends nothing."""
    from playwright.sync_api import expect

    page, ui = more_ui
    page.set_viewport_size({"width": 360, "height": 900})
    answer, _, polled_code = sign_in_job(page)
    polls, inputs = [], []
    page.route("**/api/claudexor/login/fixture", lambda r: (polls.append(1), r.fallback()))
    page.route("**/api/claudexor/login/fixture/input", lambda r: (inputs.append(r.request.post_data_json),
                                                                 r.fulfill(json={"ok": True})))
    show_more_page(page, ui)
    press_sign_in_again(page)
    card = page.locator(LOGIN_CARD)
    code, submit = card.locator("[data-login-code-input]"), card.locator("[data-login-code-submit]")
    state = card.locator("[data-login-state]")
    polled_code()
    page.wait_for_timeout(300)
    seen = len(polls)
    code.click()
    page.keyboard.type("AB-12")
    page.evaluate(SETTLED)
    box, line = submit.bounding_box(), state.bounding_box()["height"]
    point = (box["x"] + box["width"] / 2, box["y"] + box["height"] - 3)
    hit = "(b, [x, y]) => b.contains(document.elementFromPoint(x, y))"
    assert submit.evaluate(hit, point)
    submit.evaluate("b => (window.__held = b)")
    page.mouse.move(*point)
    page.mouse.down()
    answer["job"] = ({"state": "running", "phase": "verifying", "harness": "claude"} if answered == "checking"
                     else {"state": "succeeded", "harness": "claude"})
    for _ in range(800):
        if len(polls) > seen:
            break
        page.wait_for_timeout(10)
    assert len(polls) > seen
    page.wait_for_timeout(300)
    page.evaluate(SETTLED)
    if answered == "checking":
        expect(state).to_have_text("Checking the sign-in…")
        shorter, held = line - state.bounding_box()["height"], submit.bounding_box()["y"]
        same = submit.evaluate("b => b === window.__held && !b.disabled")
    else:
        expect(card.locator("[data-login-verdict]")).to_have_text("Signed in.")
        expect(submit).to_have_count(0)
    page.mouse.up()
    page.wait_for_timeout(300)
    page.evaluate(SETTLED)
    print(f"held Submit, job {answered}: code posts {len(inputs)}")
    if answered == "checking":
        print(f"  state line {shorter:.2f}px shorter, same live node {same}, y={box['y']:.2f} -> {held:.2f}")
        assert shorter > 8, "the state line must have lost a line for this to be the case"
        assert same, "the same live command"
        assert held == pytest.approx(box["y"], abs=1)
    assert inputs == ([{"value": "AB-12"}] if answered == "checking" else [])


def test_sign_in_card_keeps_open_focused_advanced_across_changed_answer(more_ui):
    """An engine that wants an outside terminal shows Advanced at once. The owner opens it from the keyboard;
    the job's next answer changes the card. Advanced stays open, focused and where it was."""
    from playwright.sync_api import expect

    page, ui = more_ui
    answer, _, polled_code = sign_in_job(page)
    answer.update(attach_command="fixture-attach --job fixture", disclosure_native=False)
    polls = []
    page.route("**/api/claudexor/login/fixture", lambda r: (polls.append(1), r.fallback()))
    level = show_more_page(page, ui)
    press_sign_in_again(page)
    card = page.locator(LOGIN_CARD)
    details = card.locator("[data-login-advanced]")
    summary = details.locator("> summary")
    polled_code()
    summary.focus()
    page.keyboard.press("Enter")
    expect(details).to_have_attribute("open", "")
    expect(summary).to_be_focused()
    next_answer(page, polls)  # the opened state is drawn once
    expect(summary).to_be_focused()
    top, scrolled = summary.bounding_box()["y"], level.evaluate("b => b.scrollTop")
    answer["deviceCode"]["userCode"] = "NEW-CODE"
    next_answer(page, polls)
    expect(card.locator("[data-device-code]")).to_have_text("NEW-CODE")
    print(f"advanced: y {top:.1f} -> {summary.bounding_box()['y']:.1f}, "
          f"scrollTop {scrolled} -> {level.evaluate('b => b.scrollTop')}")
    expect(details).to_have_attribute("open", "")
    expect(summary).to_be_focused()
    assert summary.bounding_box()["y"] == pytest.approx(top, abs=1)
    assert level.evaluate("b => b.scrollTop") == pytest.approx(scrolled, abs=1)


def test_rebuilt_sign_in_card_gives_focus_back_inside_the_advanced_the_owner_opened(more_ui):
    """The owner opens Advanced and tabs to Copy command inside it. The same job's first one-time code then
    rebuilds the card: Advanced is open again before focus is given back, so the keyboard stays on that
    command and does not fall to the page."""
    from playwright.sync_api import expect

    page, ui = more_ui
    answer, _, polled_code = sign_in_job(page)
    answer.update(attach_command="fixture-attach --job fixture", disclosure_native=False)
    show_more_page(page, ui)
    press_sign_in_again(page)
    card = page.locator(LOGIN_CARD)
    details = card.locator("[data-login-advanced]")
    copy = details.locator("[data-copy-attach]")
    details.locator("> summary").focus()
    page.keyboard.press("Enter")
    expect(details).to_have_attribute("open", "")
    page.keyboard.press("Alt+Tab" if page.context.browser.browser_type.name == "webkit" else "Tab")
    expect(copy).to_be_focused()
    copy.evaluate("b => (window.__held = b)")
    polled_code()
    expect(card.locator("[data-device-code]")).to_have_text("TEST-CODE")
    assert copy.evaluate("b => b !== window.__held"), "the first code must have rebuilt the card"
    expect(details).to_have_attribute("open", "")
    print(f"rebuilt card: focus on {page.evaluate('document.activeElement.tagName')}")
    expect(copy).to_be_focused()


def test_double_click_on_sign_in_again_starts_one_sign_in(more_ui):
    """A physical double-click: the first press brings the card into view, and its Close now stands where
    the second press lands. That press was aimed at Sign in again; a deliberate later press closes."""
    from playwright.sync_api import expect

    page, ui = more_ui
    cancels = []
    _, creates, _ = sign_in_job(page, cancels=cancels)
    level = show_more_page(page, ui)
    page.mouse.click(*wheel_to(page, row(15) + " [data-resources]"))
    button = page.locator(SIGN_IN)
    expect(button).to_have_text("Sign in again")
    # Measured in both engines: the opened card's Close spans y 147-181, x 590-662 at this size,
    # so the left part of Sign in again held near y 163 sits under it once the card is shown.
    _, y = wheel_to(page, SIGN_IN, inset=97)
    x = button.bounding_box()["x"] + 12
    page.evaluate("""() => {
        window.__presses = [];
        document.addEventListener('mousedown', event => window.__presses.push([event.detail,
            Boolean(event.target.closest('[data-harness-login]')), Boolean(event.target.closest('[data-login-dismiss]'))]),
            true);
    }""")
    page.mouse.dblclick(x, y)
    card = page.locator(LOGIN_CARD)
    expect(card.locator("[data-login-code-input]")).to_be_visible()
    card_in_view(page, level)
    # The second press really reached the new Close, as the second click of the same gesture.
    assert page.evaluate("window.__presses") == [[1, True, False], [2, False, True]]
    page.wait_for_timeout(300)
    assert len(creates) == 1 and cancels == []
    close = card.locator("[data-login-dismiss]")
    expect(close).to_have_text("Close")
    # A new press on Close, wherever the answered form has put it, is the owner's own and stops the sign-in.
    point = center(close)
    assert_at(close, point)
    page.mouse.click(*point)
    wait_for_count(page, cancels, 1)
    cancels[0].fulfill(**REFUSED_CANCEL)
    expect(card).to_contain_text("Could not cancel this sign-in")
    assert len(creates) == 1
