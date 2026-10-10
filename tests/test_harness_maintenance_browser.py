"""Real Settings maintenance controls against fixture APIs, without runtime startup."""
from __future__ import annotations

import copy
from urllib.parse import parse_qs, urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
capture = setup_browser.capture
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def _inventory_row(harness, *, targets=None):
    return {
        "harness": harness, "maintainable": True,
        "canCheckLatest": targets is None,
        "targets": targets if targets is not None else ["latest", "version", "previous", "baseline"],
        "selection": {"kind": "managed", "binary": f"/managed/{harness}/bin/program", "version": "2.0.0"},
        "installed": {"version": "2.0.0", "proved": True},
        "releaseTested": {"version": "1.0.0", "verification": "deterministic_only"},
        "available": None, "previous": {"version": "1.8.0", "operationId": "earlier"},
        "operation": None,
    }


@pytest.mark.parametrize("width", [1440, 390])
def test_maintenance_lost_reply_cancel_and_drafts(subscription_ui, width):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    page.set_viewport_size({"width": width, "height": 950})
    profiles = ui["fixture"]["status"]["profiles"]["profiles"]
    seed = copy.deepcopy(profiles[0])
    profiles[:] = []
    for number in range(18):
        profile = copy.deepcopy(seed)
        profile["profile"]["profile_id"] = f"account-{number}"
        profile["identity"]["label"] = f"Account {number + 1}"
        profiles.append(profile)

    row = _inventory_row("codex")
    native = _inventory_row("agy", targets=["latest"])
    operation = {
        "id": "fixture-operation", "harness": "codex", "state": "running", "phase": "installing",
        "target": {"kind": "latest", "version": "3.0.0"}, "mutation": "unknown",
        "termination": "not_applicable", "progress": ["Downloading the selected version"],
        "limitations": ["in_place_replacement", "new_starts_may_fail"],
    }
    requests, cancels = [], []

    def maintenance(route):
        path = urlparse(route.request.url).path
        if path.endswith("/harnesses"):
            if parse_qs(urlparse(route.request.url).query).get("checkLatest") == ["true"]:
                row["available"] = {"version": "3.0.0", "observedAt": "2026-10-09T12:00:00Z"}
            route.fulfill(json={"harnesses": [row, native]})
        elif path.endswith("/cancel"):
            cancels.append(path)
            # Acknowledged but not stopped yet.
            route.fulfill(json=operation)
        elif route.request.method == "POST":
            requests.append((route.request.post_data_json, route.request.headers.get("idempotency-key")))
            row["operation"] = {"id": operation["id"], "state": "running", "phase": "installing"}
            if len(requests) == 1:
                route.abort("failed")
            else:
                route.fulfill(status=202, json=operation)
        else:
            route.fulfill(json=operation)

    page.route("**/api/claudexor/maintenance/**", maintenance)
    page.goto(ui["url"] + "/#settings")
    page.locator('[data-settings-tab="models"]').click()
    model = page.locator('[data-model-role="main"] [data-model-role-model]')
    model.fill("owner-unsaved-model")
    page.locator('[data-settings-tab="providers"]').click()
    family = page.locator('.agent-family-card[data-family="codex"]')
    expect(family.locator('[data-family-maintenance]')).to_have_count(1)
    expect(family.locator('.harness-account-row')).to_have_count(18)
    expect(family.locator('.harness-maintenance-line')).to_have_text("Program 2.0.0 · Managed · Latest not checked")
    family.locator('summary').click()
    expect(family.locator('dl > div').filter(has_text="Bundled baseline").locator('dd')).to_have_text("1.0.0")
    family.get_by_role('button', name='Install exact version…').click()
    modal = page.locator('[role="dialog"]')
    modal.locator('input').fill("1.9.0")
    page.evaluate("""async () => {
        const {claudexorStatus} = await import('/static/modules/claudexor_status_store.js');
        await claudexorStatus.refresh();
    }""")
    expect(modal.locator('input')).to_have_value("1.9.0")
    expect(modal.locator('input')).to_be_focused()
    modal.get_by_role('button', name='Cancel', exact=True).click()
    family.get_by_role('button', name='Check latest', exact=True).click()
    expect(family.locator('.harness-maintenance-line')).to_contain_text("Latest 3.0.0")
    family.get_by_role('button', name='Update', exact=True).click()
    expect(family).to_contain_text("Update acceptance unconfirmed")
    family.get_by_role('button', name='Retry same request', exact=True).click()
    expect(family).to_contain_text("Installing 3.0.0")
    assert len(requests) == 2 and requests[0] == requests[1]
    assert requests[0][0] == {"harness": "codex", "target": {"kind": "latest"}}
    assert requests[0][1]
    family.get_by_role('button', name='Cancel update', exact=True).click()
    expect(family.get_by_role('button', name='Cancelling…', exact=True)).to_be_disabled()
    expect(family).to_contain_text("Installing 3.0.0")
    assert len(cancels) == 1

    operation.update(state="cancelled", phase="settled", termination="confirmed")
    row["installed"] = {"version": None, "proved": False}
    row["selection"]["version"] = None
    row["operation"].update(state="cancelled", phase="settled")
    page.evaluate("""async () => {
        const {claudexorStatus} = await import('/static/modules/claudexor_status_store.js');
        await claudexorStatus.refresh();
    }""")
    expect(family).to_contain_text("Update cancelled · Installed files may have changed")
    expect(family.locator('.harness-maintenance-line')).to_contain_text("Program version unknown")
    expect(family.get_by_role('button', name='Update', exact=True)).to_be_enabled()
    family.locator('.agent-family-head').scroll_into_view_if_needed()
    capture(page, f"maintenance-{width}-cancelled")
    native_family = page.locator('.agent-family-card[data-family="agy"]')
    expect(native_family.get_by_role('button', name='Update', exact=True)).to_be_visible()
    expect(native_family.get_by_role('button', name='Check version', exact=True)).to_be_visible()
    assert native_family.get_by_role('button', name='Check latest', exact=True).count() == 0
    assert native_family.locator('[data-maintenance-action="version"]').count() == 0
    assert native_family.locator('[data-maintenance-action="previous"]').count() == 0
    assert native_family.locator('[data-family-add]').count() == 0
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.locator('[data-settings-tab="models"]').click()
    expect(model).to_have_value("owner-unsaved-model")
    assert not any("/login" in path or path == "/api/settings" for path, _ in ui["posts"])


def test_old_engine_explains_missing_capability(subscription_ui):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    page.route("**/api/claudexor/maintenance/**", lambda route: route.fulfill(status=503, json={
        "error": {"code": "capability_unavailable", "message": "This engine does not support program maintenance."},
    }))
    page.goto(ui["url"] + "/#settings")
    page.locator('[data-settings-tab="providers"]').click()
    family = page.locator('.agent-family-card[data-family="codex"]')
    expect(family).to_contain_text("This engine does not support program maintenance")
    expect(family.get_by_role('button', name='Check version', exact=True)).to_be_enabled()
    assert family.locator('[data-maintenance-action="latest"]').count() == 0
    assert not ui["posts"]


@pytest.mark.parametrize("termination", ["confirmed", "unconfirmed"])
def test_historical_unknown_effect_preserves_newer_version(subscription_ui, termination):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    row = _inventory_row("codex")
    row.update(observedAt="2026-10-09T12:01:00Z")
    row["selection"]["version"] = row["installed"]["version"] = "3.1.0"
    operation = {
        "id": "historical-operation", "harness": "codex", "state": "failed", "phase": "settled",
        "finishedAt": "2026-10-09T12:00:00Z", "mutation": "unknown", "termination": termination,
    }
    row["operation"] = {key: operation[key] for key in ("id", "state", "phase", "finishedAt")}
    held_details, requests = [], []

    def maintenance(route):
        requests.append(route.request.method)
        if urlparse(route.request.url).path.endswith("/harnesses"):
            route.fulfill(json={"harnesses": [row]})
        else:
            held_details.append(route)

    page.route("**/api/claudexor/maintenance/**", maintenance)
    page.goto(ui["url"] + "/#settings")
    page.locator('[data-settings-tab="providers"]').click()
    family = page.locator('.agent-family-card[data-family="codex"]')
    expect(family.locator('.harness-maintenance-line')).to_contain_text("Program 3.1.0")
    assert held_details
    for route in held_details:
        route.fulfill(json=operation)
    expect(family).to_contain_text("Installed files may have changed")
    expect(family.locator('.harness-maintenance-line')).to_contain_text("Program 3.1.0")
    expect(family).to_contain_text("Update failed")
    if termination == "unconfirmed":
        expect(family).to_contain_text("Installer may still be running")
        expect(family.get_by_role('button', name='Cancel update', exact=True)).to_be_enabled()
        assert family.locator('[data-maintenance-action="latest"]').count() == 0
    else:
        expect(family.get_by_role('button', name='Update', exact=True)).to_be_enabled()
    assert set(requests) == {"GET"}
    family.locator('.agent-family-head').scroll_into_view_if_needed()
    capture(page, f"maintenance-historical-{termination}")
