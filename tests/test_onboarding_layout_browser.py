"""The wizard's selected layout (docs/DESIGN.md §7) in real engines, over the real modules and CSS.

An ordinary Accounts state fits the selected 980x752 client viewport (not a native-window measurement) with
every connection row, the limits warning and Continue in view. A narrow or short window, expanded keys, a sign-in or
a long error may scroll, and Continue, Back and the reason stay on the bottom edge, covering nothing focus reaches.
Down to 320px the top bar wraps within the window, every Appearance choice on the page.
"""
from __future__ import annotations

import json

import pytest

from ouroboros.test_evidence import diagnostic_directory
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui

pytestmark = [pytest.mark.ui_browser]

SETUP_WINDOW = (980, 752)
PHONE = (390, 844)
LOGIN = {"codex": {"flow": "chatgptDeviceCode", "verificationUrl": "https://auth.example/device", "userCode": "ABCD-1234"},
         "claude": {"flow": "oauth_url_input", "verificationUrl": "https://claude.example/oauth/authorize?code=true&state=fixture"}}
WARNING = "Subscription limits still apply. This connection does not enable paid credits or change your provider's spending limits."

FACTS = r"""() => {
  const se = document.scrollingElement;
  const box = (el) => { if (!el || el.closest('[hidden]')) return null; const b = el.getBoundingClientRect();
    if (!b.width || !b.height) return null;
    const hit = document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2);
    return {x: b.left, y: b.top, w: b.width, h: b.height, bottom: b.bottom,
      inView: b.top >= 0 && b.bottom <= innerHeight + 0.5 && b.left >= 0 && b.right <= innerWidth + 0.5,
      uncovered: Boolean(hit && (hit === el || el.contains(hit) || hit.contains(el)))}; };
  const one = (sel) => box(document.querySelector(sel));
  const size = (sel) => getComputedStyle(document.querySelector(sel)).fontSize;
  const rows = [...document.querySelectorAll('[data-agent-family]')].map(box);
  const footer = document.querySelector('.wizard-footer');
  const rail = document.querySelector('.wizard-steps'), current = rail?.querySelector('.wizard-step.active');
  const railBox = rail?.getBoundingClientRect(), currentBox = current?.getBoundingClientRect();
  return {overflow: se.scrollHeight - se.clientHeight, sideways: se.scrollWidth - se.clientWidth, rows,
    columns: new Set(rows.map((row) => row.x)).size, widths: new Set(rows.map((row) => row.w)).size,
    connects: [...document.querySelectorAll('[data-agent-connect]')].map(box),
    next: one('#next-btn'), back: one('#back-btn'), reason: one('.wizard-error'), limits: one('.agent-limits-note'), footer: footer && box(footer),
    reasonInFooter: Boolean(document.querySelector('.wizard-footer .wizard-error')),
    footerVar: parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--wizard-footer-height')),
    padding: parseFloat(getComputedStyle(document.documentElement).scrollPaddingBottom),
    measured: footer ? footer.getBoundingClientRect().height : 0,
    railLabel: rail?.getAttribute('aria-label'), currentMarked: current?.getAttribute('aria-current'),
    currentInRail: Boolean(currentBox && currentBox.left >= railBox.left - 0.5 && currentBox.right <= railBox.right + 0.5),
    badge: Boolean(document.querySelector('.wizard-badge')), nudge: document.body.innerText.includes('Connect Codex to start without'),
    limitsText: document.querySelector('.agent-limits-note')?.textContent.trim(), reasonText: document.querySelector('.wizard-error')?.textContent.trim(),
    sizes: {stepTitle: size('.step-title'), reason: size('.wizard-error'), next: size('#next-btn')}};
}"""


# The top bar above the wizard: the product name and the Appearance choices, read where they are painted.
TOP_BAR = r"""() => {
  const se = document.scrollingElement;
  const box = (el) => { const b = el.getBoundingClientRect();
    const hit = document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2);
    return {x: b.left, y: b.top, w: b.width, h: b.height, right: b.right, bottom: b.bottom,
      inView: b.top >= 0 && b.bottom <= innerHeight + 0.5 && b.left >= 0 && b.right <= innerWidth + 0.5,
      uncovered: Boolean(hit && (hit === el || el.contains(hit)))}; };
  const own = (el) => ({...box(el), scroll: el.scrollWidth, client: el.clientWidth});
  const bar = document.querySelector('.onboarding-top'), appearance = document.querySelector('.onboarding-appearance');
  const size = (sel) => getComputedStyle(document.querySelector(sel)).fontSize;
  return {viewport: innerWidth, page: se.scrollWidth, sideways: se.scrollWidth - se.clientWidth,
    clips: [document.documentElement, document.body, bar, appearance].map((el) => getComputedStyle(el).overflowX),
    bar: own(bar), appearance: own(appearance), brand: box(document.querySelector('.onboarding-brand')),
    label: box(document.querySelector('#onboarding-appearance-label')),
    choices: [...document.querySelectorAll('.onboarding-appearance [data-theme-choice]')].map((el) => ({name: el.dataset.themeChoice, ...box(el)})),
    sizes: {title: size('.wizard-title'), subtitle: size('.wizard-subtitle'), label: size('#onboarding-appearance-label'),
      choice: size('.onboarding-appearance [data-theme-choice]')}};
}"""


@pytest.fixture(params=["chromium", "webkit"])
def wizard(request, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    return request.getfixturevalue("subscription_ui")


def _serve_login(page, harness):
    """The two sign-in disclosures the shared card renders, for a fictitious account."""
    answer = {"job_id": "fixture-login", "job": {"state": "waiting_for_input", "phase": "awaiting_user", "harness": harness},
              "sequence": 1, "cursor": "fixture", "deviceCode": LOGIN[harness]}
    page.route("**/api/claudexor/login**", lambda route: route.fulfill(json=answer))


def _open(ui, size, theme="light", connected=False):
    page = ui["page"]
    if not connected:
        ui["fixture"]["status"]["profiles"]["profiles"] = []
    page.set_viewport_size(dict(zip(("width", "height"), size)))
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
    page.goto(ui["url"] + "/onboarding")
    page.wait_for_function("() => document.querySelectorAll('[data-agent-family]').length === 4")
    page.wait_for_selector("#quick-start-btn:not([hidden])" if connected else '[data-agent-family="codex"] [data-tone="muted"], [data-agent-family="codex"] .agent-family-status')
    if not connected:
        page.wait_for_function("() => document.querySelector('.wizard-error').textContent.includes('Connect Codex')")
    return page


def _facts(page):
    return page.evaluate(FACTS)


def _assert_way_forward(facts):
    """Back and Continue are in view and nothing lies over them, wherever the page is scrolled."""
    assert facts["next"] and facts["next"]["inView"] and facts["next"]["uncovered"], facts["next"]
    assert facts["back"] and facts["back"]["inView"] and facts["back"]["uncovered"], facts["back"]
    assert facts["sideways"] <= 1, facts["sideways"]


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("connected", [False, True], ids=["blank", "connected"])
def test_an_ordinary_accounts_state_fits_the_setup_window(wizard, connected, theme, tmp_path):
    page = _open(wizard, SETUP_WINDOW, theme, connected)
    facts = _facts(page)
    page.screenshot(path=str(diagnostic_directory(tmp_path / "frames", "fit", publish=True) / f"{'connected' if connected else 'blank'}-{theme}.png"), animations="disabled")
    assert facts["overflow"] == 0, facts
    # Four equal subscriptions in two columns, each with its action, all in view without a scroll.
    assert (len(facts["rows"]), facts["columns"], facts["widths"]) == (4, 2, 1), facts["rows"]
    assert all(row["inView"] and row["uncovered"] for row in facts["rows"] + facts["connects"])
    assert facts["limits"]["inView"] and facts["limitsText"] == WARNING
    _assert_way_forward(facts)
    # What the compact layout dropped is the repetition, never the size of the text or what it says.
    assert (facts["badge"], facts["nudge"]) == (False, False)
    assert (facts["railLabel"], facts["currentMarked"]) == ("Step 1 of 5", "step")
    assert facts["sizes"] == {"stepTitle": "24px", "reason": "12px", "next": facts["sizes"]["next"]}
    assert facts["reasonInFooter"]
    if connected:
        page.locator("#quick-start-btn").wait_for()
        assert "Signed in to Codex" in page.locator("#agents-outcome").inner_text()
    else:
        assert facts["reason"]["inView"] and facts["reasonText"].startswith("Connect Codex, enter an API key")
        assert page.locator("#agents-outcome").is_hidden()


@pytest.mark.parametrize("scene, size, harness, keys", [
    ("phone", (390, 640), None, False),
    ("short", (980, 500), None, False),
    ("keys-open", SETUP_WINDOW, None, True),
    ("codex-sign-in", SETUP_WINDOW, "codex", False),
    ("claude-sign-in-phone", PHONE, "claude", False),
])
def test_a_taller_state_scrolls_under_the_way_forward(wizard, scene, size, harness, keys, tmp_path):
    evidence = diagnostic_directory(tmp_path / "frames", "scrolls", publish=True)
    page = wizard["page"]
    if harness:
        _serve_login(page, harness)
    page = _open(wizard, size)
    if keys:
        page.locator('[data-collapse="api-access"] > summary').click()
        page.locator("#openrouter-key").wait_for()
    if harness:
        page.click(f'[data-agent-connect="{harness}"]')
        page.wait_for_selector("[data-open-signin]")
    page.evaluate("() => { document.activeElement?.blur?.(); window.scrollTo(0, 0); }")
    facts = _facts(page)
    page.screenshot(path=str(evidence / f"{scene}-top.png"), animations="disabled")
    assert facts["overflow"] > 0, "this state is taller than the window"
    _assert_way_forward(facts)
    assert facts["reason"]["inView"] and facts["reasonInFooter"]
    # The warning stands before any connection, above the sign-in that a Connect opens.
    limits = page.locator(".agent-limits-note")
    assert limits.inner_text() == WARNING
    if harness:
        assert limits.bounding_box()["y"] < page.locator("[data-open-signin]").bounding_box()["y"]
    # A real wheel to the end: the actions stay, and the whole sign-in card can be read and used.
    page.mouse.move(size[0] / 2, size[1] / 2)
    page.mouse.wheel(0, facts["overflow"] + 200)
    page.wait_for_function("() => { const se = document.scrollingElement; return se.scrollTop + se.clientHeight >= se.scrollHeight - 1; }")
    page.screenshot(path=str(evidence / f"{scene}-end.png"), animations="disabled")
    _assert_way_forward(_facts(page))
    controls = ["[data-open-signin]", "[data-copy-signin-link]", "[data-login-dismiss]"] if harness else []
    controls += ["[data-device-code]", "[data-copy-device-code]"] if harness == "codex" else ["[data-login-code-input]"] if harness == "claude" else []
    for selector in controls:
        control = page.locator(selector)
        control.scroll_into_view_if_needed()
        box, footer = control.bounding_box(), page.locator(".wizard-footer").bounding_box()
        assert box["y"] + box["height"] <= footer["y"] + 1, (selector, box, footer)
    _assert_way_forward(_facts(page))
    assert not page.locator(".wizard-badge").count()


@pytest.mark.parametrize("width", [320, 360, 390])
def test_a_narrow_window_wraps_the_top_bar_and_keeps_every_appearance_choice(wizard, width, tmp_path):
    """Appearance wraps under the product name, and its own label and choices wrap inside it, so the bar never widens
    the page: each choice is on the page, uncovered and usable, at the type size the setup window shows. Nothing
    clips an overflow away; the setup window keeps the name and Appearance on one row."""
    from playwright.sync_api import expect

    evidence = diagnostic_directory(tmp_path / "frames", "top-bar", publish=True)
    page = _open(wizard, SETUP_WINDOW)
    wide = page.evaluate(TOP_BAR)
    page.set_viewport_size({"width": width, "height": PHONE[1]})
    page.evaluate("window.scrollTo(0, 0)")
    narrow = page.evaluate(TOP_BAR)
    page.screenshot(path=str(evidence / f"{width}x{PHONE[1]}.png"), animations="disabled")
    page.locator(".onboarding-top").screenshot(path=str(evidence / f"{width}-top-bar.png"), animations="disabled")
    (evidence / f"{width}.json").write_text(json.dumps({"wide": wide, "narrow": narrow}, indent=1), encoding="utf-8")
    assert wide["appearance"]["y"] < wide["brand"]["bottom"] and wide["appearance"]["x"] >= wide["brand"]["right"], wide
    assert narrow["sideways"] == 0, narrow
    assert narrow["bar"]["scroll"] <= narrow["bar"]["client"] and narrow["appearance"]["scroll"] <= narrow["appearance"]["client"], narrow
    assert not {"hidden", "clip"} & set(narrow["clips"]), narrow["clips"]
    assert [choice["name"] for choice in narrow["choices"]] == ["light", "dark", "system"]
    assert all(choice["inView"] and choice["uncovered"] for choice in narrow["choices"] + [narrow["label"]]), narrow
    assert narrow["sizes"] == wide["sizes"], (narrow["sizes"], wide["sizes"])
    for choice in ("system", "dark", "light"):
        button = page.locator(f'.onboarding-appearance [data-theme-choice="{choice}"]')
        button.click()
        expect(button).to_have_attribute("aria-checked", "true")
    assert page.evaluate(TOP_BAR)["sideways"] == 0


def _settled_clearance(page):
    page.wait_for_function("() => Math.abs(parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--wizard-footer-height')) - document.querySelector('.wizard-footer').getBoundingClientRect().height) < 1")
    facts = _facts(page)
    assert abs(facts["padding"] - (facts["measured"] + 8)) < 1, facts
    return facts["measured"]


def _walk_focus(page, tabs=12):
    """Forward through what the keyboard reaches (WebKit's Tab skips buttons): no field stops under the footer.
    Read after the engine's own reveal, which WebKit runs a frame later."""
    stops = []
    for _ in range(tabs):
        page.keyboard.press("Tab")
        page.wait_for_timeout(120)
        stop = page.evaluate("""() => { const el = document.activeElement, b = el.getBoundingClientRect(), f = document.querySelector('.wizard-footer').getBoundingClientRect();
            return {name: el.id || el.tagName, inFooter: Boolean(el.closest('.wizard-footer')), top: b.top, bottom: b.bottom, footerTop: f.top}; }""")
        stops.append(stop)
        if not stop["inFooter"] and stop["name"] != "BODY":
            assert stop["bottom"] <= stop["footerTop"] + 1, stop
    assert len({stop["name"] for stop in stops}) > 2, stops
    return stops


def test_focus_stops_clear_of_the_footer_by_its_measured_height(wizard, tmp_path):
    """A guessed clearance breaks when the reason wraps: the padding is the footer's real height, whatever it says."""
    evidence = diagnostic_directory(tmp_path / "frames", "focus", publish=True)
    page = _open(wizard, (390, 640))
    page.locator('[data-collapse="api-access"] > summary').click()
    report = {"short_reason": {"footer": _settled_clearance(page)}}
    page.locator("#openrouter-key").focus()
    report["short_reason"]["stops"] = _walk_focus(page)
    page.screenshot(path=str(evidence / "short-reason.png"), animations="disabled")

    # A key with a long name and no address yet: the reason beside Continue wraps to more lines, and the padding follows.
    page.evaluate("window.scrollTo(0, 0)")
    page.locator('[data-connection-add="openai-compatible"]').click()
    page.locator('[data-connection-row][data-draft="1"] [data-connection-label]').fill("A compatible server with a very long name that the reason repeats in full")
    page.wait_for_function("() => document.querySelector('.wizard-error').textContent.includes('enter the Base URL')")
    report["long_reason"] = {"footer": _settled_clearance(page), "reason": page.locator(".wizard-error").inner_text()}
    assert report["long_reason"]["footer"] > report["short_reason"]["footer"], report
    page.locator("#openrouter-key").focus()
    report["long_reason"]["stops"] = _walk_focus(page)
    page.screenshot(path=str(evidence / "long-reason.png"), animations="disabled")
    _assert_way_forward(_facts(page))
    (evidence / "focus.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


@pytest.mark.parametrize("size", [(980, 500), (390, 640)], ids=["short", "narrow"])
def test_revealing_a_pinned_control_never_moves_the_page(wizard, size):
    """It is already in view; the clearance that keeps fields out from under the footer does not apply to the footer."""
    page = _open(wizard, size, connected=True)
    page.click("#next-btn")
    page.wait_for_selector('.wizard-steps[aria-label="Step 2 of 5"]')
    page.evaluate("window.scrollTo(0, 120)")
    page.wait_for_function("window.scrollY === 120")
    for control in ("#back-btn", "#next-btn"):
        page.locator(control).evaluate("el => el.scrollIntoView({block: 'nearest'})")
        assert page.evaluate("window.scrollY") == 120, control
    # The keyboard's own stops on them, wherever that page's Tab order reaches (WebKit's plain Tab skips buttons).
    page.locator("#back-btn").focus()
    assert page.evaluate("window.scrollY") == 120
    page.keyboard.press("Alt+Tab" if page.context.browser.browser_type.name == "webkit" else "Tab")
    page.wait_for_timeout(120)
    assert page.evaluate("document.activeElement.closest('.wizard-footer') !== null"), "Continue follows Back"
    assert page.evaluate("window.scrollY") == 120


@pytest.mark.parametrize("size", [SETUP_WINDOW, PHONE], ids=["setup-window", "phone"])
def test_every_step_keeps_the_way_forward_and_says_where_you_are(wizard, size, tmp_path):
    evidence = diagnostic_directory(tmp_path / "frames", "steps", publish=True)
    page = _open(wizard, size, connected=True)
    page.locator('[data-collapse="api-access"] > summary').click()
    page.locator("#openrouter-key").fill("sk-or-draft-fixture-0123456789")
    for index, step in enumerate(("accounts", "models", "review", "budget", "summary"), start=1):
        page.wait_for_selector(f'.wizard-steps[aria-label="Step {index} of 5"]')
        facts = _facts(page)
        page.screenshot(path=str(evidence / f"{size[0]}x{size[1]}-{index}-{step}.png"), animations="disabled")
        _assert_way_forward(facts)
        assert (facts["railLabel"], facts["currentMarked"], facts["currentInRail"]) == (f"Step {index} of 5", "step", True), facts
        assert abs(facts["footerVar"] - facts["measured"]) < 1, facts
        if index < 5:
            page.click("#next-btn")
    # Back walks the same footer, and the draft is still there.
    for _ in range(4):
        page.click("#back-btn")
    page.wait_for_selector("#openrouter-key")
    assert page.locator("#openrouter-key").input_value() == "sk-or-draft-fixture-0123456789"


@pytest.mark.parametrize("size", [SETUP_WINDOW, PHONE], ids=["setup-window", "phone"])
def test_late_save_states_keep_their_actions_beside_a_wrapped_reason(wizard, size, tmp_path):
    from playwright.sync_api import expect

    evidence = diagnostic_directory(tmp_path / "frames", "late-save", publish=True)
    page = wizard["page"]
    answers = [{"status": 503, "json": {"ok": False, "code": "settings_save_timeout", "saved": None,
                                        "error": "Saving settings is still running after the settings writer's bound."}},
               {"status": 400, "json": {"ok": False, "saved": False, "error": "This save was refused: " + "a long reason from the server that wraps " * 60}},
               {"status": 200, "json": {"ok": True, "runtime_mode": "pro", "restart_required": True}}]
    page.route("**/api/onboarding/complete", lambda route: route.fulfill(**answers.pop(0)) if route.request.method == "POST" else route.fallback())
    page = _open(wizard, size, connected=True)
    page.click("#quick-start-btn")
    page.wait_for_selector(".summary-card")

    page.click("#next-btn")  # the first save's answer says nothing certain: read back, or retry explicitly
    expect(page.locator("#check-save-btn")).to_be_visible()
    facts = _facts(page)
    page.screenshot(path=str(evidence / f"{size[0]}-unknown.png"), animations="disabled")
    _assert_way_forward(facts)
    assert page.locator("#next-btn").inner_text() == "Retry save"
    assert facts["reason"]["inView"] and "unknown" in facts["reasonText"]
    assert page.locator("#check-save-btn").bounding_box()["y"] + 1 >= facts["reason"]["y"]

    page.click("#next-btn")  # refused, with a long error: all of it readable above Back and Retry
    page.wait_for_function("() => document.querySelector('.wizard-error').textContent.includes('refused')")
    page.wait_for_function("() => Math.abs(parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--wizard-footer-height')) - document.querySelector('.wizard-footer').getBoundingClientRect().height) < 1")
    facts = _facts(page)
    page.screenshot(path=str(evidence / f"{size[0]}-refused.png"), animations="disabled")
    _assert_way_forward(facts)
    # However long it is, the reason scrolls in place: the actions and the step keep the window.
    assert facts["reason"]["inView"] and 16 < facts["reason"]["h"] <= size[1] * 0.4 + 1, facts["reason"]
    assert facts["footer"]["h"] < size[1] * 0.7, facts["footer"]

    page.click("#next-btn")  # saved, and this tab cannot restart: no footer, no clearance left behind
    expect(page.locator(".step-title")).to_have_text("Setup saved — restart to apply it")
    expect(page.locator("#open-app-btn")).to_be_visible()
    facts = page.evaluate("""() => ({footer: document.querySelectorAll('.wizard-footer').length,
        clearance: getComputedStyle(document.documentElement).getPropertyValue('--wizard-footer-height').trim(),
        sideways: document.scrollingElement.scrollWidth - document.scrollingElement.clientWidth})""")
    page.screenshot(path=str(evidence / f"{size[0]}-restart.png"), animations="disabled")
    assert facts == {"footer": 0, "clearance": "0px", "sideways": 0}
