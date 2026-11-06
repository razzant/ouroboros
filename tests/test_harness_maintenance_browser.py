"""Real Settings maintenance controls against fixture APIs, without runtime startup."""
from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
capture = setup_browser.capture
pytestmark = [pytest.mark.ui_browser]


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
    family.get_by_role('button', name='Details', exact=True).click()
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


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_packaged_update_preparation_requires_an_explicit_restart_without_fake_operation(subscription_ui, theme):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    page.add_init_script("localStorage.setItem('ouroboros.theme', %s)" % repr(theme))
    row = _inventory_row("codex")
    posts = []
    message = ("Environment preparation complete; CLI update not started. "
               "Use Restart Claudexor in Accounts, then repeat Update. It ends every run on the engine; "
               "its confirmation says how many and whose.")

    def maintenance(route):
        if route.request.method == "GET":
            route.fulfill(json={"harnesses": [row]})
        else:
            posts.append(route.request.post_data_json)
            route.fulfill(status=409, json={"error": {"code": "maintenance_restart_required", "message": message,
                "context": {"acceptance": "not_submitted", "scope": "this_attempt", "preparation": "ready"}}})

    page.route("**/api/claudexor/maintenance/**", maintenance)
    page.goto(ui["url"] + "/#settings")
    page.locator('[data-settings-tab="providers"]').click()
    family = page.locator('.agent-family-card[data-family="codex"]')
    family.get_by_role("button", name="Update", exact=True).click()
    expect(family).to_contain_text(message)
    expect(family.get_by_role("button", name="Update", exact=True)).to_be_enabled()
    expect(family.locator(".harness-maintenance-line")).to_contain_text("Program 2.0.0")
    assert family.get_by_role("button", name="Cancel update", exact=True).count() == 0
    assert "Operation completed" not in family.inner_text() and "acceptance unconfirmed" not in family.inner_text()
    assert posts == [{"harness": "codex", "target": {"kind": "latest"}}]
    assert not any("restart" in path or "login" in path for path, _ in ui["posts"])
    family.locator(".harness-maintenance-row").scroll_into_view_if_needed()
    capture(page, f"maintenance-preparation-{theme}")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert not ui["errors"]


PINNED_SPAWN_FAILURE = json.loads((Path(__file__).parent / "fixtures/claudexor_maintenance_spawn_enoent.json").read_text(encoding="utf-8"))
ROW_OBSERVED = "2026-10-09T12:00:00Z"


def _host_reply_for_missing_executable(path, monkeypatch):
    """What the real host service answers to the pinned engine's captured wire, for this path."""
    from ouroboros import harness_maintenance as service
    from ouroboros import review_run_isolation
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    monkeypatch.setattr(review_run_isolation, "attach_home", lambda: None)
    monkeypatch.delenv("OUROBOROS_CLAUDEXOR_BIN", raising=False)
    received = {**PINNED_SPAWN_FAILURE["body"], "message": f"spawn {path} ENOENT"}

    def fail(*_args, **_kwargs):
        error = ClaudexorUnavailable(received["code"], received["message"], status_code=PINNED_SPAWN_FAILURE["status"])
        error.problem = copy.deepcopy(received)
        raise error

    class Gateway:
        engine_version = PINNED_SPAWN_FAILURE["provenance"]["engine_version"]
        engine_build_sha = PINNED_SPAWN_FAILURE["provenance"]["engine_build_sha"]
        maintenance_harnesses = staticmethod(fail)

    @contextmanager
    def connection():
        yield Gateway()

    monkeypatch.setattr(service, "read_owned_gateway", connection)
    with pytest.raises(ClaudexorUnavailable) as caught:
        service.inspect_harnesses([], fresh=True)
    return service.maintenance_problem(caught.value)


def _remount(page):
    page.evaluate("""async () => {
        const accounts = await import('/static/modules/harness_accounts.js');
        const {claudexorStatus} = await import('/static/modules/claudexor_status_store.js');
        accounts.destroyHarnessAccounts();
        await accounts.initHarnessAccounts({store: claudexorStatus});
    }""")


@pytest.mark.parametrize("width,theme", [(1440, "light"), (1440, "dark"), (390, "light")])
def test_missing_executable_is_one_explained_section_fact_until_a_real_fresh_read(subscription_ui, tmp_path, monkeypatch, width, theme):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    page.add_init_script("localStorage.setItem('ouroboros.theme', %s)" % repr(theme))
    page.set_viewport_size({"width": width, "height": 950})
    executable = tmp_path / "Moved Ouroboros.app" / "Contents" / "Frameworks" / "node-standalone" / "bin" / "node"
    status, failure = _host_reply_for_missing_executable(executable, monkeypatch)
    assert failure["error"]["message"] == f"spawn {executable} ENOENT"
    rows = [_inventory_row("codex"), _inventory_row("agy", targets=["latest"])]
    for row in rows:
        row["observedAt"] = ROW_OBSERVED
    world = {"broken": False}
    reads, held = [], []

    def maintenance(route):
        query = parse_qs(urlparse(route.request.url).query)
        current = query.get("fresh") == ["true"] or query.get("checkLatest") == ["true"]
        reads.append(current)
        assert route.request.method == "GET", "Inspecting never posts"
        if world.get("hold") and query.get("harness") == ["agy"]:
            held.append(route)
        elif current and world["broken"]:
            route.fulfill(status=status, json=failure)
        else:  # Every read mints a new top-level time, the cached ones included.
            route.fulfill(json={"observedAt": "2026-11-04T01:31:00Z", "harnesses": rows})

    page.route("**/api/claudexor/maintenance/**", maintenance)
    page.goto(ui["url"] + "/#settings")
    page.locator('[data-settings-tab="providers"]').click()
    banner = page.locator("#agents-service-banner")
    codex = page.locator('.agent-family-card[data-family="codex"]')
    agy = page.locator('.agent-family-card[data-family="agy"]')
    expect(codex.locator(".harness-maintenance-line")).to_have_text("Program 2.0.0 · Managed · Latest not checked")
    expect(banner).not_to_contain_text("no longer present")

    # Press the family's own check, the path an owner takes; the failure reaches the whole section.
    world["broken"] = True
    codex.get_by_role("button", name="Check latest", exact=True).click()
    expect(banner).to_contain_text("The executable reported by the last program inspection is no longer present.")
    expect(banner).to_contain_text("If the app was moved or replaced, Restart Claudexor in Accounts may help")
    expect(banner).to_contain_text("it ends every run on the engine, and success is not guaranteed.")
    expect(banner).to_have_attribute("data-tone", "warn")
    expect(banner).not_to_contain_text("ENOENT")
    for family in (codex, agy):
        line = family.locator(".harness-maintenance-line")
        expect(line).to_contain_text("Last known")
        expect(line).to_contain_text("2026")
        expect(line).not_to_contain_text("2026-11")
        assert "T12:00:00Z" not in line.inner_text()
        text = family.inner_text()
        assert "no longer present" not in text and "ENOENT" not in text and "Program inspection unavailable" not in text
        assert family.locator('[data-maintenance-action="latest"]').count() == 0
    expect(codex.get_by_role("button", name="Check latest", exact=True)).to_be_enabled()

    # The exact words have one home.
    toggle = page.locator("#agents-service-diagnostics-toggle")
    expect(toggle).to_be_visible()
    toggle.click()
    fact = page.locator('#agents-service-diagnostics [data-fact="Last program inspection"] dd')
    expect(fact).to_contain_text(f"ENOENT: spawn {executable} ENOENT")
    expect(page.locator('#agents-service-diagnostics [data-fact="Inspected by engine"] dd')).to_have_text(
        f"{PINNED_SPAWN_FAILURE['provenance']['engine_version']} · {PINNED_SPAWN_FAILURE['provenance']['engine_build_sha']}")
    assert page.get_by_text(f"spawn {executable} ENOENT", exact=False).count() == 1
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.locator(".agents-service").scroll_into_view_if_needed()
    capture(page, f"maintenance-missing-executable-{width}-{theme}")

    # A failed read lands without rebuilding the rows: another family's Details keeps its focus and open state.
    details = codex.get_by_role("button", name="Details", exact=True)
    details.click()
    world["hold"] = True
    agy.get_by_role("button", name="Check version", exact=True).click()
    expect(agy.get_by_role("button", name="Checking…", exact=True)).to_be_disabled()
    details.focus()
    assert held
    world["hold"] = False
    held.pop().fulfill(status=status, json=failure)
    expect(agy.get_by_role("button", name="Check version", exact=True)).to_be_enabled()
    expect(details).to_be_focused()
    expect(details).to_have_attribute("aria-expanded", "true")

    # The status tick's own rebuild keeps the disclosure open too.
    page.evaluate("""async () => {
        const {refreshHarnessStatus} = await import('/static/modules/harness_accounts.js');
        await refreshHarnessStatus();
    }""")
    expect(codex.get_by_role("button", name="Details", exact=True)).to_have_attribute("aria-expanded", "true")
    expect(codex.locator(".harness-maintenance-details")).to_be_visible()
    expect(page.locator("#agents-service-diagnostics-toggle")).to_have_attribute("aria-expanded", "true")
    assert codex.locator(".harness-maintenance-details").inner_text().count("ENOENT") == 0
    codex.locator(".agent-family-head").scroll_into_view_if_needed()
    capture(page, f"maintenance-missing-executable-details-{width}-{theme}")

    # A remount's cached read answers 200: not proof of anything current.
    before = len(reads)
    _remount(page)
    expect(banner).to_contain_text("no longer present")
    expect(codex.locator(".harness-maintenance-line")).to_contain_text("Last known")
    assert False in reads[before:]

    # A true fresh success removes the current instruction and the last-known mark.
    world["broken"] = False
    codex.get_by_role("button", name="Check latest", exact=True).click()
    expect(banner).not_to_contain_text("no longer present")
    expect(banner).not_to_contain_text("Restart")
    expect(codex.locator(".harness-maintenance-line")).not_to_contain_text("Last known")
    expect(agy.locator(".harness-maintenance-line")).not_to_contain_text("Last known")
    expect(page.locator('#agents-service-diagnostics [data-fact="Last program inspection"]')).to_have_count(0)
    expect(codex.get_by_role("button", name="Update", exact=True)).to_be_visible()
    assert not any("restart" in path or "login" in path or "maintenance" in path for path, _ in ui["posts"])
    page.locator(".agents-service").scroll_into_view_if_needed()
    capture(page, f"maintenance-missing-executable-recovered-{width}-{theme}")
    assert not ui["errors"]


def test_retained_failed_operation_on_a_healthy_replacement_advises_nothing(subscription_ui, tmp_path):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    gone = tmp_path / "old-app" / "node"
    operation = {
        "id": "old-operation", "harness": "codex", "state": "failed", "phase": "settled",
        "finishedAt": "2026-10-09T12:00:00Z", "mutation": "unknown", "termination": "confirmed",
        "problem": {**PINNED_SPAWN_FAILURE["body"], "message": f"spawn {gone} ENOENT"},
    }
    row = _inventory_row("codex")
    row.update(observedAt="2026-10-09T12:05:00Z", operation={key: operation[key] for key in ("id", "state", "phase", "finishedAt")})
    methods = []

    def maintenance(route):
        methods.append(route.request.method)
        route.fulfill(json={"harnesses": [row]} if urlparse(route.request.url).path.endswith("/harnesses") else operation)

    page.route("**/api/claudexor/maintenance/**", maintenance)
    page.goto(ui["url"] + "/#settings")
    page.locator('[data-settings-tab="providers"]').click()
    family = page.locator('.agent-family-card[data-family="codex"]')
    expect(family).to_contain_text("Update failed · Installed files may have changed")
    family.get_by_role("button", name="Details", exact=True).click()
    expect(family.locator(".harness-maintenance-exact code")).to_have_text(f"spawn {gone} ENOENT")
    expect(page.locator("#agents-service-banner")).not_to_contain_text("Restart")
    expect(page.locator("#agents-service-banner")).not_to_contain_text("no longer present")
    assert "Restart" not in family.inner_text()
    assert set(methods) == {"GET"}
    family.locator(".agent-family-head").scroll_into_view_if_needed()
    capture(page, "maintenance-missing-executable-history")
