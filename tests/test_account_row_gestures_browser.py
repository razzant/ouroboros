"""Native gestures on the Accounts renderer, with only synthetic transport."""
from urllib.parse import urlsplit

import pytest

from tests import test_account_resources_browser as accounts
from tests.settings_browser_helpers import SETTLED
from tests.test_account_resources_browser import (
    ObservationTransport, assert_at, center, open_accounts, store_call,
)

# Pytest discovers the two dependent fixtures in this module.
resources_ui = accounts.resources_ui
subscription_ui = accounts.subscription_ui

pytestmark = pytest.mark.ui_browser


def fixture_page(resources_ui):
    page = resources_ui["page"]
    origin = urlsplit(resources_ui["url"]).netloc
    page.route("**/*", lambda r: r.fallback() if urlsplit(r.request.url).netloc == origin else r.abort())
    return page


def begin(page, button, modality, *, hit=None):
    if modality == "space":
        button.focus()
        page.keyboard.down("Space")
        return lambda: page.keyboard.up("Space")
    target = button if hit is None else hit
    target.scroll_into_view_if_needed()
    point = center(target)
    assert_at(target, point)
    page.mouse.move(*point)
    page.mouse.down()

    def release():
        # Wording may change the button width during the read. The original point must
        # still hit the same node, or no native click could diagnose command retargeting.
        assert target.evaluate("(el, [x, y]) => el.contains(document.elementFromPoint(x, y))", point)
        page.mouse.up()

    return release


@pytest.mark.parametrize("modality", ["pointer", "space"])
@pytest.mark.parametrize("changed", [False, True], ids=["same-command", "new-command"])
def test_status_button_gesture_keeps_its_command(resources_ui, modality, changed):
    from playwright.sync_api import expect

    ui = resources_ui
    page = fixture_page(ui)
    status = ui["fixture"]["status"]["profiles"]["profiles"][0]["status"]
    status.update(verification="not_run", availability="unknown")
    transport = ObservationTransport(ui)
    logins = []

    def login(route):
        logins.append(route.request.post_data_json)
        route.fulfill(status=409, json={"code": "fixture_refusal", "error": "Synthetic request only"})

    page.route("**/api/claudexor/login", login)
    page, row = open_accounts(ui, width=1440)
    store_call(page, "s.refresh({afterCurrent: true})")
    button = row.locator("[data-harness-login]")
    expect(button).to_have_text("Check status")
    release = begin(page, button, modality)
    if changed:
        status.update(verification="failed", availability="unavailable")
    ui["fixture"]["status"]["profiles"]["profiles"][0]["profile"]["display_name"] = "Fresh account identity"
    ui["fixture"]["status"]["profiles"]["profiles"][0]["identity"]["label"] = "Fresh account identity"
    store_call(page, "s.observe({afterCurrent: true})")
    expect(button).to_have_text("Sign in" if changed else "Check status")
    expect(row.locator(".harness-account-main strong")).to_have_text("Fresh account identity")
    if modality == "space":
        expect(button).to_be_focused()
    full_before = transport.calls.count("full")
    release()
    if not changed:
        page.wait_for_function("async () => !(await import('/static/modules/claudexor_status_store.js')).claudexorStatus.loading")
    page.evaluate(SETTLED)
    assert not logins, "The held Check status gesture must never start Sign in"
    assert transport.calls.count("full") == full_before + (not changed)
    if changed:
        expect(button).to_be_enabled()
        button.press("Enter")
        expect(page.locator("[data-login-card]")).to_contain_text("Synthetic request only")
        assert [(body["harness"], body["profile_id"]) for body in logins] == [("claude", "claude-default")]


@pytest.mark.parametrize("modality", ["pointer", "space"])
@pytest.mark.parametrize("changed", [False, True], ids=["same-command", "new-command"])
def test_rotation_switch_gesture_keeps_its_command(resources_ui, modality, changed):
    from playwright.sync_api import expect

    ui = resources_ui
    page = fixture_page(ui)
    transport = ObservationTransport(ui)
    page, row = open_accounts(ui, width=1440)
    store_call(page, "s.refresh({afterCurrent: true})")
    button = row.get_by_role("switch")
    expect(button).not_to_be_checked()
    release = begin(page, button, modality, hit=row.locator(".ui-switch"))
    ui["fixture"]["status"]["profiles"]["profiles"][0]["profile"]["enabled"] = changed
    ui["fixture"]["status"]["profiles"]["profiles"][0]["profile"]["display_name"] = "Fresh account identity"
    ui["fixture"]["status"]["profiles"]["profiles"][0]["identity"]["label"] = "Fresh account identity"
    store_call(page, "s.observe({afterCurrent: true})")
    expect(row.locator(".harness-account-main strong")).to_have_text("Fresh account identity")
    expect(button).to_have_attribute("aria-checked", str(changed).lower())
    if modality == "space":
        expect(button).to_be_focused()
    transport.hold.add("patch")
    release()
    page.evaluate(SETTLED)
    if changed:
        assert not transport.pending["patch"], "The held enable gesture must never disable the account"
        expect(button).to_be_checked()  # native checkbox activation must also be cancelled
        if modality == "space":
            expect(button).to_be_focused()
        button.press("Space")  # a deliberate new gesture may now disable it
    assert transport.calls.count("patch") == 1
    _, body, _ = transport.pending["patch"][0]
    assert body["enabled"] is not changed
    assert ui["posts"][-1] == ("/api/claudexor/credential-profiles/claude/claude-default", {"enabled": not changed})
    ui["fixture"]["status"]["profiles"]["profiles"][0]["profile"]["enabled"] = not changed
    transport.release("patch")
    expect(button).to_have_attribute("aria-checked", str(not changed).lower())
    expect(button).to_be_enabled()


@pytest.mark.parametrize("transition", ["named-to-default", "default-to-named", "named-to-named"])
def test_family_add_does_not_retarget_on_release(resources_ui, transition):
    from playwright.sync_api import expect

    ui = resources_ui
    page = fixture_page(ui)
    profiles = ui["fixture"]["status"]["profiles"]["profiles"]
    original = profiles[0]
    profiles[:] = [] if transition == "default-to-named" else [original]
    ObservationTransport(ui)
    logins = []

    def login(route):
        logins.append(route.request.post_data_json)
        route.fulfill(status=409, json={"code": "fixture_refusal", "error": "Synthetic request only"})

    page.route("**/api/claudexor/login", login)
    page.goto(ui["url"] + "/#settings")
    accounts.open_settings(page, "accounts/claude")
    store_call(page, "s.refresh({afterCurrent: true})")
    button = page.locator('[data-family="claude"] [data-family-add]')
    expect(button).to_have_text("Connect" if not profiles else "Add account")
    release = begin(page, button, "space")
    if transition == "named-to-default":
        profiles.clear()
    else:
        original["profile"]["display_name"] = "Fresh account identity"
        original["identity"]["label"] = "Fresh account identity"
        profiles[:] = [original]
    store_call(page, "s.observe({afterCurrent: true})")
    expect(button).to_have_text("Connect" if not profiles else "Add account")
    if profiles:
        expect(page.locator(".harness-account-main strong")).to_have_text("Fresh account identity")
    expect(button).to_be_focused()

    release()
    dialog = page.get_by_role("dialog")
    page.evaluate(SETTLED)
    if transition != "named-to-named":
        print(f"held gesture {transition}: dialogs={dialog.count()}, login requests={len(logins)}")
        assert not logins, "A held Add must not become Connect"
        expect(dialog).to_have_count(0)
        button.press("Enter")
    if profiles:
        expect(dialog).to_contain_text("Name for the additional")
        expect(dialog).to_have_count(1)
        assert not logins
        dialog.get_by_role("textbox").fill("fixture-account")
        dialog.locator("[data-confirm-ok]").click()
    expect(page.locator("[data-login-card]")).to_contain_text("Synthetic request only")
    assert [(body["harness"], body.get("profile_id", "")) for body in logins] == [
        ("claude", "fixture-account" if profiles else "")
    ]


def test_multi_family_login_mount_follows_the_active_job(resources_ui):
    from playwright.sync_api import expect

    ui = resources_ui
    page = fixture_page(ui)
    profiles = ui["fixture"]["status"]["profiles"]["profiles"]
    profiles.append({"profile": {"harness_id": "codex", "profile_id": "codex-default", "enabled": True},
                     "status": {"verification": "failed", "availability": "unavailable"}})
    login_calls = []
    refuse_cancel = True

    def login(route):
        body = route.request.post_data_json
        family = body["harness"]
        login_calls.append((family, body["profile_id"]))
        route.fulfill(json={"job_id": family, "job": {"harness": family, "state": "waiting_for_input"},
                            "deviceCode": {"flow": "oauth_url_input", "verificationUrl": "https://example.test/login"}})

    def prior(route):
        if route.request.method == "DELETE":
            if refuse_cancel:
                return route.fulfill(status=409, json={"error": "Synthetic release not proven"})
            return route.fulfill(json={"job": {"state": "cancelled", "outcome": {"reason": "owner_cancelled"}}})
        return route.fulfill(json={"job": {"state": "waiting_for_input"}})

    page.route("**/api/claudexor/login", login)
    page.route("**/api/claudexor/login/claude", prior)
    page, row = open_accounts(ui, width=1440)
    # The production provider door scopes to one family. Reuse its actual mounted view and the
    # exported default initializer to exercise the supported all-families renderer.
    expect(row).to_have_count(1)
    page.evaluate("""async () => {
        const {initHarnessAccounts} = await import('/static/modules/harness_accounts.js');
        await initHarnessAccounts({path: 'accounts/claude'});
    }""")
    first = page.locator('[data-family-login="claude"]')
    second = page.locator('[data-family-login="codex"]')
    a = row.locator('[data-harness-login]')
    b = page.locator('[data-profile="codex-default"] [data-harness-login]')
    # Verified sign-in-again is in More, not a permanently visible button.
    row.locator('[data-resources]').click()
    a.click()
    expect(first.locator('[data-login-code-input]')).to_be_visible()
    b.click()
    expect(first).to_contain_text('Could not cancel the active sign-in')
    expect(second.locator('[data-login-card]')).to_have_count(0)
    assert login_calls == [("claude", "claude-default")]
    refuse_cancel = False
    b.click()
    expect(second.locator('[data-login-card]')).to_be_visible()
    expect(first.locator('[data-login-card]')).to_have_count(0)
    assert login_calls == [("claude", "claude-default"), ("codex", "codex-default")]
