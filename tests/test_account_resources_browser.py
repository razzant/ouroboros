"""Actual Settings resources against synthetic API responses; no engine startup."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.settings_browser_helpers import open_settings

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
