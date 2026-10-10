"""Actual Settings resources against synthetic API responses; no engine startup."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
capture = setup_browser.capture
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
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
        target = TARGET if index == 0 else {"harness": "codex" if index % 2 else "claude",
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
    page.locator('[data-settings-tab="providers"]').click()
    expect(page.locator(".harness-account-row")).to_have_count(20)
    return page, page.locator(ROW)


def passive_read(page):
    page.evaluate("""async () => {
        const { claudexorStatus } = await import('/static/modules/claudexor_status_store.js');
        await claudexorStatus.refresh();
    }""")


def frame_section(locator):
    """Position evidence below Settings' existing sticky fade, without product scrolling."""
    locator.evaluate("""element => {
        const host = element.closest('.settings-scroll');
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
        assert panel.locator("[data-refresh-account]").bounding_box()["height"] >= 36
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
    # Closing an embedded client loses session storage; durable references must survive.
    page.evaluate("sessionStorage.clear()")
    page.reload()
    page.locator('[data-settings-tab="providers"]').click()
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
    page.reload()
    page.locator('[data-settings-tab="providers"]').click()
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
    expect(page.locator(".harness-account-row")).to_have_count(20)
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
    maintenance_only = _inventory_row("agy", targets=["latest"])
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
    family.get_by_role("button", name="Check latest", exact=True).click()
    expect(family.get_by_role("button", name="Checking…", exact=True)).to_be_disabled()
    expect(details).to_have_attribute("open", "")
    assert held
    details.locator("summary").focus()
    hold_inventory = False
    entry["available"] = {"version": "3.0.0", "observedAt": "2026-10-09T12:00:00Z"}
    for route in held:
        route.fulfill(json=inventory)
    expect(family.locator(".harness-maintenance-line")).to_contain_text("Latest 3.0.0")
    expect(details).to_have_attribute("open", "")
    expect(details.locator("summary")).to_be_focused()
    expect(row.locator("[data-resources]")).to_have_attribute("aria-expanded", "true")
    expect(row).to_contain_text("Reset outcome is unconfirmed")
    assert ui["reset_calls"] == [original]

    family.get_by_role("button", name="Update", exact=True).click()
    expect(family).to_contain_text("Installing 3.0.0")
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
    expect(page.locator(".harness-account-row")).to_have_count(20)
    expect(row).to_contain_text("Reset outcome is unconfirmed")
    expect(details).to_have_attribute("open", "")
    expect(family).to_contain_text("Update cancelled")
    assert len(updates) == len(cancellations) == 1
    assert updates[0][0] and updates[0][1] == {"harness": "claude", "target": {"kind": "latest"}}
    orphan = page.locator('.agent-family-card[data-family="agy"]')
    expect(orphan.locator("[data-family-add], [data-resources]")).to_have_count(0)
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
