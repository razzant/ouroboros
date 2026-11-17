"""Real Settings maintenance controls against fixture APIs, without runtime startup."""
from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.settings_browser_helpers import open_settings

subscription_ui = setup_browser.subscription_ui
capture = setup_browser.capture
pytestmark = [pytest.mark.ui_browser]


def _inventory_row(harness, *, exact=True):
    allowed = {"available": True, "reason": None, "commands": [{"kind": "exec", "command": f"/vendor/{harness}", "args": ["update"]}]}
    return {
        "harness": harness, "actions": {"install": allowed, "update": allowed,
            "exactVersion": allowed if exact else {"available": False, "reason": "No exact verb.", "commands": []}},
        "selection": {"path": f"/vendor/{harness}", "realPath": f"/vendor/{harness}", "version": "2.0.0",
            "versionOutput": "2.0.0", "overrideEnv": None, "overrideValue": None},
        "newest": {"status": "not_read", "source": None}, "updateAvailable": "cannot_tell",
        "releaseNotesUrl": None, "operation": None,
    }


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("account_count", [1, 18])
def test_maintenance_lost_reply_cancel_and_saved_assignments(subscription_ui, width, account_count):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    page.set_viewport_size({"width": width, "height": 950})
    profiles = ui["fixture"]["status"]["profiles"]["profiles"]
    seed = copy.deepcopy(profiles[0])
    profiles[:] = []
    for number in range(account_count):
        profile = copy.deepcopy(seed)
        profile["profile"]["profile_id"] = f"account-{number}"
        profile["identity"]["label"] = f"Account {number + 1}"
        profiles.append(profile)

    row = _inventory_row("codex")
    native = _inventory_row("agy", exact=False)
    operation = {
        "id": "fixture-operation", "harness": "codex", "state": "running", "phase": "installing",
        "target": {"kind": "latest", "version": "3.0.0"}, "mutation": "unknown",
        "termination": "not_applicable", "progress": ["Downloading the selected version"],
        "limitations": ["Program files were replaced in place."],
    }
    requests, cancels = [], []

    def maintenance(route):
        path = urlparse(route.request.url).path
        if path.endswith("/harnesses"):
            if parse_qs(urlparse(route.request.url).query).get("checkLatest") == ["true"]:
                row["newest"] = {"status": "available", "version": "3.0.0", "source": "https://vendor.test/latest", "observedAt": "2026-10-09T12:00:00Z"}
                row["updateAvailable"] = "yes"
            route.fulfill(json={"harnesses": [row, native]})
        elif path.endswith("/cancel"):
            cancels.append(path)
            # Acknowledged but not stopped yet.
            route.fulfill(json=operation)
        elif route.request.method == "POST":
            requests.append((route.request.post_data_json, route.request.headers.get("idempotency-key")))
            row["operation"] = copy.deepcopy(operation)
            if len(requests) == 1:
                route.abort("failed")
            else:
                route.fulfill(status=202, json=operation)
        else:
            route.fulfill(json=operation)

    page.route("**/api/claudexor/maintenance/**", maintenance)
    page.goto(ui["url"] + "/#settings")
    open_settings(page, "models/role/main")
    model = page.locator('[data-model-role="main"] [data-model-role-model]')
    expect(model).to_have_value("gpt-test")
    # Separate pages dispose unsaved editors; maintenance must leave the saved assignment alone.
    open_settings(page, "accounts/codex")
    family = page.locator('.agent-family-card[data-family="codex"]')
    expect(family.locator('[data-family-maintenance]')).to_have_count(1)
    expect(family.locator('.harness-account-row')).to_have_count(account_count)
    expect(family.locator('.harness-maintenance-line')).to_contain_text("Program 2.0.0")
    family.get_by_role('button', name='Details', exact=True).click()
    expect(family.locator('dl > div').filter(has_text="Program path").locator('dd')).to_have_text("/vendor/codex")
    family.get_by_role('button', name='Check latest', exact=True).click()
    expect(family.locator('.harness-maintenance-line')).to_contain_text("3.0.0 available")
    family.get_by_role('button', name='Update', exact=True).click()
    expect(family).to_contain_text("Update acceptance unconfirmed")
    family.get_by_role('button', name='Retry same request', exact=True).click()
    expect(family).to_contain_text("Updating…")
    assert len(requests) == 2 and requests[0] == requests[1]
    assert requests[0][0] == {"harness": "codex", "target": {"kind": "latest"}}
    assert requests[0][1]
    family.get_by_role('button', name='Cancel update', exact=True).click()
    expect(family.get_by_role('button', name='Cancelling…', exact=True)).to_be_disabled()
    expect(family).to_contain_text("Updating…")
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
    open_settings(page, "accounts")
    open_settings(page, "accounts/agy")
    native_family = page.locator('.agent-family-card[data-family="agy"]')
    expect(native_family.get_by_role('button', name='Update', exact=True)).to_be_visible()
    native_family.get_by_role('button', name='Details', exact=True).click()
    expect(native_family.get_by_role('button', name='Check latest', exact=True)).to_be_visible()
    assert native_family.locator('[data-maintenance-action="version"]').count() == 0
    assert native_family.locator('[data-maintenance-action="previous"]').count() == 0
    assert native_family.locator('[data-family-add]').count() == 0
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    open_settings(page, "models/role/main")
    expect(model).to_have_value("gpt-test")
    assert not any("/login" in path or path == "/api/settings" for path, _ in ui["posts"])


@pytest.mark.parametrize("reason", [
    "Update the app or connect an engine at version 6.0.0 or newer to install or update harness programs.",
    "This engine does not support program maintenance.",
])
def test_old_engine_explains_missing_capability(subscription_ui, reason):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    page.route("**/api/claudexor/maintenance/**", lambda route: route.fulfill(status=503, json={
        "error": {"code": "maintenance_unavailable", "message": reason},
    }))
    page.goto(ui["url"] + "/#settings")
    open_settings(page, "accounts/codex")
    family = page.locator('.agent-family-card[data-family="codex"]')
    expect(family).to_contain_text(f"Program inspection unavailable: {reason}")
    assert family.inner_text().count("Program inspection unavailable") == 1
    family.get_by_role('button', name='Details', exact=True).click()
    expect(family.get_by_role('button', name='Check latest', exact=True)).to_be_enabled()
    assert family.locator('[data-maintenance-action="latest"]').count() == 0
    assert not ui["posts"]


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("maintenance", [False, True])
def test_connect_card_shows_only_maintenance_problem_actions(subscription_ui, width, maintenance):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    page.set_viewport_size({"width": width, "height": 950})
    message = "The vendor installer could not write its program." if maintenance else "Disable another profile before connecting."
    action = "Read the vendor log in Settings, Accounts" if maintenance else "disable_extra_profiles"
    page.route("**/api/claudexor/login", lambda route: route.fulfill(status=503 if maintenance else 409, json={
        "error": message, "code": "installer_failed" if maintenance else "credential_profile_limit",
        "required_actions": [action], "retryable": False,
        "context": {"operationId": "accepted-install"} if maintenance else {},
    }))
    page.goto(ui["url"] + "/#settings")
    open_settings(page, "accounts/codex")
    family = page.locator('.agent-family-card[data-family="codex"]')
    family.locator('[data-resources]').first.click()
    family.locator('[data-harness-login]').first.click()
    card = family.locator('[data-login-card]')
    expect(card).to_contain_text(message)
    expect(card.locator('[data-login-required-action]')).to_have_count(int(maintenance))
    if maintenance:
        expect(card.locator('[data-login-required-action]')).to_have_text(action)
    else:
        expect(card).not_to_contain_text(action)
    expect(card).not_to_contain_text("accepted-install")
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


@pytest.mark.parametrize("termination", ["confirmed", "unconfirmed"])
def test_historical_unknown_effect_preserves_newer_version(subscription_ui, termination):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    row = _inventory_row("codex")
    row.update(observedAt="2026-10-09T12:01:00Z")
    row["selection"]["version"] = "3.1.0"
    operation = {
        "id": "historical-operation", "harness": "codex", "state": "failed", "phase": "settled",
        "finishedAt": "2026-10-09T12:00:00Z", "mutation": "unknown", "termination": termination,
    }
    row["operation"] = copy.deepcopy(operation)
    details, requests = [], []

    def maintenance(route):
        requests.append(route.request.method)
        if urlparse(route.request.url).path.endswith("/harnesses"):
            route.fulfill(json={"harnesses": [row]})
        else:
            details.append(route.request.url)
            route.fulfill(json=operation)

    page.route("**/api/claudexor/maintenance/**", maintenance)
    page.goto(ui["url"] + "/#settings")
    open_settings(page, "accounts/codex")
    family = page.locator('.agent-family-card[data-family="codex"]')
    expect(family.locator('.harness-maintenance-line')).to_contain_text("Program 3.1.0")
    if termination == "confirmed":
        assert not details  # Full settled receipts need no detail fetch; unresolved custody still polls.
    expect(family).to_contain_text("Installed files may have changed")
    expect(family.locator('.harness-maintenance-line')).to_contain_text("Program 3.1.0")
    expect(family).to_contain_text("Update failed")
    if termination == "unconfirmed":
        expect(family).to_contain_text("Installer may still be running")
        family.get_by_role('button', name='Details', exact=True).click()
        expect(family.get_by_role('button', name='Cancel update', exact=True)).to_be_enabled()
        expect(family.locator('[data-maintenance-action="latest"]')).to_be_disabled()
    else:
        expect(family.get_by_role('button', name='Update', exact=True)).to_be_enabled()
    assert set(requests) == {"GET"}
    family.locator('.agent-family-head').scroll_into_view_if_needed()
    capture(page, f"maintenance-historical-{termination}")


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
        await accounts.initHarnessAccounts({store: claudexorStatus, family: 'codex', path: 'accounts/codex'});
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
    rows = [_inventory_row("codex"), _inventory_row("agy", exact=False)]
    for row in rows:
        row["observedAt"] = ROW_OBSERVED
    world = {"broken": False}
    reads, held = [], []

    def maintenance(route):
        query = parse_qs(urlparse(route.request.url).query)
        current = query.get("fresh") == ["true"] or query.get("checkLatest") == ["true"]
        reads.append(current)
        assert route.request.method == "GET", "Inspecting never posts"
        if world.get("hold") and query.get("harness") == ["codex"]:
            held.append(route)
        elif current and world["broken"]:
            route.fulfill(status=status, json=failure)
        else:  # Every read mints a new top-level time, the cached ones included.
            route.fulfill(json={"observedAt": "2026-11-04T01:31:00Z", "harnesses": rows})

    page.route("**/api/claudexor/maintenance/**", maintenance)
    page.goto(ui["url"] + "/#settings")
    open_settings(page, "accounts/codex")
    banner = page.locator("#agents-service-banner")
    codex = page.locator('.agent-family-card[data-family="codex"]')
    expect(codex.locator(".harness-maintenance-line")).to_contain_text("Program 2.0.0")
    expect(banner).not_to_contain_text("no longer present")

    # Press the family's own check, the path an owner takes; the failure reaches the whole section.
    world["broken"] = True
    codex.get_by_role("button", name="Details", exact=True).click()
    codex.get_by_role("button", name="Check latest", exact=True).click()
    expect(banner).to_contain_text("The executable reported by the last program inspection is no longer present.")
    expect(banner).to_contain_text("If the app was moved or replaced, Restart Claudexor in Accounts may help")
    expect(banner).to_contain_text("it ends every run on the engine, and success is not guaranteed.")
    expect(banner).to_have_attribute("data-tone", "warn")
    expect(banner).not_to_contain_text("ENOENT")
    for family in (codex,):
        line = family.locator(".harness-maintenance-line")
        expect(line).to_contain_text("Last known")
        expect(line).to_contain_text("2026")
        expect(line).not_to_contain_text("2026-11")
        assert "T12:00:00Z" not in line.inner_text()
        text = family.inner_text()
        assert "no longer present" not in text and "ENOENT" not in text and "Program inspection unavailable" not in text
        expect(family.locator('[data-maintenance-action="latest"]')).to_have_count(0)
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

    # A failed read lands without rebuilding the row: its Details keeps focus and open state.
    details = codex.get_by_role("button", name="Details", exact=True)
    world["hold"] = True
    codex.get_by_role("button", name="Check latest", exact=True).click()
    expect(codex.get_by_role("button", name="Checking…", exact=True)).to_be_disabled()
    details.focus()
    assert held
    world["hold"] = False
    held.pop().fulfill(status=status, json=failure)
    expect(codex.get_by_role("button", name="Check latest", exact=True)).to_be_enabled()
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

    # A remount keeps the dated observation without an eager read or clearing the failure.
    before = len(reads)
    _remount(page)
    expect(banner).to_contain_text("no longer present")
    expect(codex.locator(".harness-maintenance-line")).to_contain_text("Last known")
    assert not reads[before:]

    # A true fresh success removes the current instruction and the last-known mark.
    world["broken"] = False
    codex.get_by_role("button", name="Check latest", exact=True).click()
    expect(banner).not_to_contain_text("no longer present")
    expect(banner).not_to_contain_text("Restart")
    expect(codex.locator(".harness-maintenance-line")).not_to_contain_text("Last known")
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
    row.update(observedAt="2026-10-09T12:05:00Z", operation=copy.deepcopy(operation))
    methods = []

    def maintenance(route):
        methods.append(route.request.method)
        route.fulfill(json={"harnesses": [row]} if urlparse(route.request.url).path.endswith("/harnesses") else operation)

    page.route("**/api/claudexor/maintenance/**", maintenance)
    page.goto(ui["url"] + "/#settings")
    open_settings(page, "accounts/codex")
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


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("account_count", [1, 18])
def test_program_facts_and_outcome_stay_once_in_their_places(subscription_ui, width, account_count):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    page.set_viewport_size({"width": width, "height": 950})
    profiles = ui["fixture"]["status"]["profiles"]["profiles"]
    seed = copy.deepcopy(profiles[0])
    profiles[:] = []
    for index in range(account_count):
        profile = copy.deepcopy(seed)
        profile["profile"]["profile_id"] = f"p{index}"
        profiles.append(profile)
    row = _inventory_row("codex")
    row["observedAt"] = "2026-11-10T12:00:00Z"
    reads = []

    def reply(route):
        assert route.request.method == "GET"
        reads.append(route.request.url)
        route.fulfill(json={"harnesses": [row]} if urlparse(route.request.url).path.endswith("/harnesses") else row["operation"])

    page.route("**/api/claudexor/maintenance/**", reply)
    page.goto(ui["url"] + "/#settings")
    open_settings(page, "accounts/codex")
    family = page.locator('[data-family="codex"]')
    expect(family.locator(".harness-account-row")).to_have_count(account_count)
    family.locator("[data-maintenance-details]").click()
    offsets = []
    for verdict, text in [("no", "Up to date"), ("yes", "3.0.0 available"), ("cannot_tell", "Latest unknown")]:
        row["updateAvailable"] = verdict
        row["newest"] = {"status": "available", "version": "3.0.0", "source": "https://vendor.test/latest", "observedAt": "2026-11-10T10:00:00Z"}
        family.get_by_role("button", name="Check latest", exact=True).click()
        expect(family.locator(".harness-maintenance-cause")).to_have_text(text)
        expect(family.locator(".harness-maintenance-line")).to_have_count(1)
        expect(family.locator('[data-maintenance-action="latest"]')).to_have_count(0 if verdict == "no" else 1)
        offsets.append(family.locator(".harness-maintenance-row").evaluate("e => ['.harness-maintenance-line','.harness-maintenance-state'].map(s => e.querySelector(s).getBoundingClientRect().top - e.getBoundingClientRect().top)"))
    row["operation"] = {"id": "nonzero-success", "harness": "codex", "state": "succeeded", "phase": "settled", "outcome": "changed",
        "mutation": "applied", "termination": "confirmed", "before": {"version": "1.0.0"}, "after": {"version": "2.0.0"},
        "exitCode": 7, "finishedAt": "2026-11-10T11:00:00Z", "commands": [{"binary": "/vendor/codex", "args": ["update"], "env": {}, "script": None}],
        "limitations": ["Installer exited 7; replacement verified."], "progress": ["Vendor finished."]}
    family.get_by_role("button", name="Check latest", exact=True).click()
    expect(family).to_contain_text("Updated from 1.0.0")
    expect(family.get_by_text("Installer exited 7; replacement verified.", exact=True)).to_have_count(1)
    expect(family.get_by_role("button", name="Return to 1.0.0", exact=True)).to_be_visible()
    assert "nonzero-success" not in family.inner_text()
    assert "Ran" not in family.locator("dl").inner_text()
    offsets.append(family.locator(".harness-maintenance-row").evaluate("e => ['.harness-maintenance-line','.harness-maintenance-state'].map(s => e.querySelector(s).getBoundingClientRect().top - e.getBoundingClientRect().top)"))
    assert all(abs(before - after) <= 1 for sample in offsets for before, after in zip(offsets[0], sample)), offsets
    row["actions"]["update"] = {"available": False, "reason": "Clear the pinned override before Update.", "commands": []}
    family.get_by_role("button", name="Check latest", exact=True).click()
    expect(family.get_by_text("Clear the pinned override before Update.", exact=True)).to_have_count(1)
    expect(family.locator('[data-maintenance-action="latest"]')).to_have_count(0)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    family.locator(".agent-family-head").scroll_into_view_if_needed()
    capture(page, f"program-facts-{account_count}-{width}")


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("host", ["desktop", "telegram"])
def test_release_notes_uses_host_opener_and_keeps_the_app(subscription_ui, width, host):
    from playwright.sync_api import expect
    ui, page = subscription_ui, subscription_ui["page"]
    row = _inventory_row("codex")
    page.route("**/api/claudexor/maintenance/harnesses*", lambda route: route.fulfill(json={"harnesses": [row]}))
    page.set_viewport_size({"width": width, "height": 950})
    page.goto(ui["url"] + "/#settings")
    open_settings(page, "accounts/codex")
    family = page.locator('.agent-family-card[data-family="codex"]')
    family.get_by_role("button", name="Details", exact=True).click()
    expect(family.get_by_role("link", name="Release notes")).to_have_count(0)
    row["releaseNotesUrl"] = "https://vendor.example/releases"
    family.get_by_role("button", name="Check latest", exact=True).click()
    link = family.get_by_role("link", name="Release notes")
    expect(link).to_have_attribute("target", "_blank")
    page.evaluate("""host => {
        window.__external = [];
        if (host === 'desktop') window.pywebview = {api: {open_external_url: async url => {
            window.__external.push(url); return {ok: true};
        }}};
        else window.Telegram = {WebApp: {openLink: url => window.__external.push(url)}};
    }""", host)
    before = page.url
    link.click()
    page.wait_for_function("window.__external.length === 1")
    assert page.evaluate("window.__external") == [row["releaseNotesUrl"]]
    assert page.url == before and len(page.context.pages) == 1
