"""The chat model picker in the real composer (DESIGN "Composer owner controls"; owner 29.10, the approved mockup).

On a real server from this checkout (candidate copy, isolated data, mock LLM) in WebKit at the owner's 1512×949 (DPR 2)
and on a 390×844 phone, dark and light: the quiet trigger stands first on the row and names the room's model; it opens
one sheet, anchored above it on a desk and rising from the bottom on a phone: search, Favorites · Subscription · API,
the source rail, the routes with their stars, Main and Fallback in the foot. A row chosen saves through the owner
writer and survives a reload; the foot's Use returns to Main's reference; a star survives a reload; a refused save
returns the trigger to the saved choice. Routes read newer first by the version their names state (owner 30.10);
OpenRouter and a connected direct OpenAI key draw their own marks, and only sources the catalog names are offered.
Only `/api/model-catalog` is answered by the test, in the shape the installed catalog answers (no provider is
contacted); the room's choice, its readback and the stars are the server's own.
"""
from __future__ import annotations

import json

import pytest

from tests.test_effort_range_browser import _evidence_dir, _launch
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture import

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

SOL = "claudexor::codex=gpt-6.1-sol"
ASTRA = "claudexor::codex=gpt-6-astra"
FLASH = "openai-compatible::giga-osa-ds-v41-flash/deepseek-v4.1-flash"
PICKER = "#page-chat .chat-model-picker"
TRIGGER = f"{PICKER} .chat-model-trigger"
SHEET = ".chat-model-sheet"


def _codex(model_id: str, name: str, profile: str = "codex-default") -> dict:
    return {"provider_id": "claudexor", "provider": "Codex", "source": "Claudexor", "id": model_id, "name": name,
            "value": f"claudexor::codex={model_id}", "label": f"Claudexor · {name}", "source_id": "codex",
            "credential_profile_id": profile, "availability": "available"}


def _api(provider_id: str, label: str, model_id: str, name: str) -> dict:
    value = model_id if provider_id == "openrouter" else f"{provider_id}::{model_id}"
    return {"provider_id": provider_id, "provider": label, "source": label, "id": model_id, "name": name,
            "value": value, "label": f"{label} · {name}"}


# The installed catalog's shape (gateway/models.py): its Codex rows and one account it could not read; API rows as the
# configured connections list them (names are the test's, not an account inventory).
CATALOG = {
    "items": [
        _codex("gpt-6.1-sol", "GPT-6.1-Sol"), _codex("gpt-6-astra", "GPT-6-Astra"),
        _codex("gpt-6-chat-thinking", "GPT-6 Chat Thinking"), _codex("gpt-6.1-terra", "GPT-6.1 Terra"),
        _codex("gpt-6-sol", "GPT-6-Sol"), _codex("gpt-6-luna", "GPT-6-Luna"), _codex("gpt-5.6-sol", "GPT-5.6-Sol"),
        _api("anthropic", "Anthropic", "claude-opus-5-5", "Claude Opus 5.5"),
        _api("anthropic", "Anthropic", "claude-sonnet-5-5", "Claude Sonnet 5.5"),
        _api("openrouter", "OpenRouter", "anthropic/claude-opus-5.5", "Anthropic: Claude Opus 5.5"),
        _api("openrouter", "OpenRouter", "deepseek/deepseek-v4.1-flash", "DeepSeek: DeepSeek V4.1 Flash"),
        _api("openai-compatible", "OpenAI-compatible endpoint", "giga-osa-ds-v41-flash/deepseek-v4.1-flash",
             "deepseek-v4.1-flash"),
        # A direct OpenAI key the owner connected: its /models rows carry no display name, so the id is the name.
        _api("openai", "OpenAI", "gpt-6", "gpt-6"), _api("openai", "OpenAI", "gpt-6.1", "gpt-6.1"),
    ],
    "errors": [{"provider_id": "claudexor", "source_id": "codex", "credential_profile_id": "work1",
                "code": "auth_unavailable", "error": "The account has no verified current catalog credential"}],
    "model_sources": [{"id": "codex", "label": "Codex", "credentialHarness": "codex", "accountCatalog": True}],
    "partial": True,
}

FACTS = """() => {
    const picker = document.querySelector('#page-chat .chat-model-picker');
    const trigger = picker.querySelector('.chat-model-trigger');
    const sheet = document.querySelector('.chat-model-sheet');
    const box = (node) => { if (!node) return null; const r = node.getBoundingClientRect();
        return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width, height: r.height }; };
    const row = picker.parentElement;
    return {
        known: picker.dataset.known, moved: picker.dataset.moved, saving: picker.dataset.saving,
        name: picker.querySelector('.chat-model-name').textContent,
        scope: picker.querySelector('.chat-model-scope').hidden ? '' : picker.querySelector('.chat-model-scope').textContent,
        // The closed line's fit (DESIGN "Composer owner controls"): what each capsule value shows against what its words
        // need, and whether the model's name is still drawn (it yields first, to a few letters, then to the mark).
        values: [...row.querySelectorAll('.chat-capsule')].filter((capsule) => capsule.getClientRects().length).map((capsule) => {
            const value = capsule.querySelector('.chat-capsule-value');
            const words = document.createRange();
            words.selectNodeContents(value);
            return { text: value.textContent.trim(), shown: value.getBoundingClientRect().width, needs: words.getBoundingClientRect().width };
        }),
        nameWidth: picker.querySelector('.chat-model-name').getClientRects().length
            ? picker.querySelector('.chat-model-name').getBoundingClientRect().width : 0,
        scopeShown: picker.querySelector('.chat-model-scope').getClientRects().length > 0,
        label: trigger.getAttribute('aria-label') || '',
        firstControl: [...row.children].find((kid) => kid.getClientRects().length && !kid.classList.contains('composer-menu'))
            ?.className || '',
        trigger: box(trigger), sheet: box(sheet), viewport: { width: innerWidth, height: innerHeight },
        tabs: sheet ? [...sheet.querySelectorAll('.chat-model-tab')].map((tab) => [tab.textContent, tab.getAttribute('aria-selected')]) : [],
        rail: sheet ? [...sheet.querySelectorAll('.chat-model-source')].map((b) => [b.getAttribute('aria-label'), b.getAttribute('aria-pressed')]) : [],
        initials: sheet ? [...sheet.querySelectorAll('.chat-model-source .chat-model-initial')].map((n) => n.closest('button').getAttribute('aria-label')) : [],
        rows: sheet ? [...sheet.querySelectorAll('.chat-model-option')].map((o) => o.getAttribute('aria-label') || o.textContent) : [],
        selected: sheet ? [...sheet.querySelectorAll('.chat-model-option[aria-selected="true"]')].map((o) => o.getAttribute('aria-label')) : [],
        foot: sheet ? [...sheet.querySelectorAll('.chat-model-ref')].map((ref) => ref.textContent.replace(/\\s+/g, ' ').trim()) : [],
        focused: document.activeElement?.className || '',
        docScroll: document.documentElement.scrollWidth > innerWidth + 1,
    };
}"""


def _seed(server) -> None:
    """The owner's routes: Main on a Codex subscription model, the first Fallback on an OpenAI-compatible key."""
    server["stop_server"]()
    path = server["data_dir"] / "settings.json"
    settings = json.loads(path.read_text(encoding="utf-8"))
    settings.update(OUROBOROS_MODEL=SOL, OUROBOROS_MODEL_FALLBACKS=FLASH)
    path.write_text(json.dumps(settings), encoding="utf-8")
    server["start_server"]()


def _open_chat(page, url: str, theme: str) -> None:
    page.add_init_script(f"try {{ localStorage.setItem('ouroboros.theme', '{theme}'); }} catch (e) {{}}")
    page.route("**/api/model-catalog", lambda route: route.fulfill(content_type="application/json", body=json.dumps(CATALOG)))
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("#chat-input", timeout=30_000)
    page.wait_for_function(f"() => document.querySelector('{PICKER}')?.dataset.known === 'true'", timeout=30_000)


def _open_sheet(page) -> dict:
    page.click(TRIGGER)
    page.wait_for_selector(f"{SHEET} .chat-model-option", timeout=15_000)
    page.wait_for_timeout(250)  # the entrance (ui-popup-in) ends before anything is measured or drawn
    return page.evaluate(FACTS)


def _shot(page, tmp_path, name: str) -> None:
    page.screenshot(path=str(_evidence_dir(tmp_path) / f"chat-model-{name}.png"), animations="disabled")


def _whole_values(facts: dict) -> bool:
    return bool(facts["values"]) and all(value["shown"] + 0.5 >= value["needs"] for value in facts["values"])


def _assert_phone_line(facts: dict, name: str) -> None:
    """At 390 px the model's name yields first (DESIGN "Composer owner controls", 30.10): Subagents, Context and Effort
    keep whole, distinct values; the name is a few letters or none (and then `this chat` goes with it: the mark alone),
    never a letter beside the mark; the trigger's label keeps the whole name."""
    texts = [value["text"] for value in facts["values"]]
    assert len(texts) == 3 and len(set(texts)) == 3 and _whole_values(facts), ("the capsules keep their values", facts)
    assert facts["nameWidth"] == 0 or facts["nameWidth"] >= 43, ("never a letter and an ellipsis beside the mark", facts)
    assert facts["nameWidth"] > 0 or not facts["scopeShown"], ("the mark alone, then", facts)
    assert name.lower() in facts["label"].lower(), facts  # before the sheet reads the catalog the name is the route's id


def _state(page, url: str) -> dict:
    return page.evaluate("async (url) => (await (await fetch(url + '/api/state', {cache: 'no-store'})).json())", url)


def test_desktop_sheet_saves_reloads_stars_and_returns_to_main(direct_server_with_data, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    _seed(server)
    url = server["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, "webkit", viewport={"width": 1512, "height": 949}, device_scale_factor=2)
        try:
            _open_chat(page, url, "dark")
            closed = page.evaluate(FACTS)
            assert closed["firstControl"].startswith("chat-model-picker"), closed  # first after ☰, before the capsules
            assert closed["moved"] == "false" and closed["scope"] == "", closed
            assert closed["nameWidth"] > 0 and _whole_values(closed), ("a line with room cuts nothing", closed)
            _shot(page, tmp_path, "desktop-dark-closed")

            facts = _open_sheet(page)
            assert facts["tabs"] == [["Favorites", "false"], ["Subscription", "true"], ["API", "false"]], facts
            assert facts["rail"] == [["Codex", "true"]], facts
            assert [row.split(",")[0] for row in facts["rows"]] == [
                "GPT-6.1-Sol", "GPT-6.1 Terra", "GPT-6-Astra", "GPT-6 Chat Thinking", "GPT-6-Luna", "GPT-6-Sol", "GPT-5.6-Sol"], \
                "newer first by the version each name states, never the catalog's alphabetical order"
            assert facts["foot"][0].startswith("Main") and "In use" in facts["foot"][0], facts
            assert facts["foot"][1].startswith("Fallback") and facts["foot"][1].endswith("Use"), facts
            sheet, trigger = facts["sheet"], facts["trigger"]
            assert sheet["bottom"] <= trigger["top"] and sheet["top"] >= 0 and sheet["right"] <= facts["viewport"]["width"], facts
            assert "chat-model-query" in facts["focused"] and not facts["docScroll"], facts
            assert page.locator(f"{SHEET} .chat-model-note").first.inner_text().startswith("Could not read every list: work1"), \
                "a partial catalog says which account it could not read, beside the rows it kept"
            _shot(page, tmp_path, "desktop-dark-subscription")

            page.click(f"{SHEET} .chat-model-tab[data-lane='api']")
            api = page.evaluate(FACTS)
            assert [label for label, _pressed in api["rail"]] == [
                "All API sources", "OpenRouter", "OpenAI", "Anthropic", "OpenAI-compatible endpoint"], api
            assert api["initials"] == [], ("every connected source here has a sourced mark; none is a boxed initial", api)
            page.click(f"{SHEET} .chat-model-source[data-source='openrouter']")
            _shot(page, tmp_path, "desktop-dark-api-openrouter")
            page.click(f"{SHEET} .chat-model-source[data-source='openai']")
            assert page.evaluate(FACTS)["rows"] == ["gpt-6.1, OpenAI · API", "gpt-6, OpenAI · API"]
            _shot(page, tmp_path, "desktop-dark-api-openai")
            page.click(f"{SHEET} .chat-model-source[data-source='anthropic']")
            assert page.evaluate(FACTS)["rows"][:2] == ["Claude Opus 5.5, Anthropic · API", "Claude Sonnet 5.5, Anthropic · API"]
            _shot(page, tmp_path, "desktop-dark-api-anthropic")

            page.fill(f"{SHEET} .chat-model-query", "opus")
            searched = page.evaluate(FACTS)
            assert [tab[1] for tab in searched["tabs"]] == ["false", "false", "false"], searched
            # Results keep the rail's order of sources (OpenRouter before Anthropic), each source's routes newer first.
            assert searched["rows"][:2] == ["Anthropic: Claude Opus 5.5, OpenRouter · API", "Claude Opus 5.5, Anthropic · API"], searched
            _shot(page, tmp_path, "desktop-dark-search")
            page.keyboard.press("Escape")
            assert page.locator(f"{SHEET} .chat-model-query").input_value() == ""
            page.keyboard.press("Escape")
            page.wait_for_selector(SHEET, state="detached")
            assert "chat-model-trigger" in page.evaluate(FACTS)["focused"], "focus returns to the trigger"

            # A row is the room's own route: saved by the owner writer, read back, and kept across a reload.
            _open_sheet(page)
            page.click(f"{SHEET} .chat-model-tab[data-lane='subscription']")
            page.locator(f"{SHEET} .chat-model-option", has_text="GPT-6-Astra").click()
            page.wait_for_function(f"() => document.querySelector('{PICKER}').dataset.saving === 'false'"
                                   f" && document.querySelector('{PICKER}').dataset.moved === 'true'", timeout=10_000)
            assert _state(page, url)["model_choice"] == {"kind": "model", "model": ASTRA, "use_local": False}
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function(f"() => document.querySelector('{PICKER}')?.dataset.moved === 'true'", timeout=30_000)
            after = page.evaluate(FACTS)
            assert (after["name"], after["scope"]) == ("GPT-6-Astra", "this chat"), after
            _shot(page, tmp_path, "desktop-dark-closed-chosen")

            # Stars are per route and live in the install's UI preferences.
            facts = _open_sheet(page)
            assert facts["selected"] and facts["selected"][0].startswith("GPT-6-Astra"), facts
            page.locator(f"{SHEET} .chat-model-option", has_text="GPT-6.1-Sol").hover()
            page.click(f"{SHEET} .chat-model-star[data-star='{SOL}']")
            page.click(f"{SHEET} .chat-model-tab[data-lane='api']")
            page.click(f"{SHEET} .chat-model-source[data-source='anthropic']")
            page.locator(f"{SHEET} .chat-model-option", has_text="Claude Opus 5.5").hover()
            page.click(f"{SHEET} .chat-model-star[data-star='anthropic::claude-opus-5-5']")
            page.click(f"{SHEET} .chat-model-tab[data-lane='favorites']")
            page.wait_for_function("() => document.querySelectorAll('.chat-model-sheet .chat-model-option').length === 2")
            favorites = page.evaluate(FACTS)
            assert favorites["rows"] == ["GPT-6.1-Sol, Codex · Subscription", "Claude Opus 5.5, Anthropic · API"], favorites
            page.wait_for_function("async () => ((await (await fetch('/api/ui/preferences', {cache: 'no-store'})).json())"
                                   ".model_favorites || []).length === 2", timeout=10_000)
            _shot(page, tmp_path, "desktop-dark-favorites")

            page.evaluate("() => window.ouroTheme.set('light')")
            page.click(f"{SHEET} .chat-model-tab[data-lane='subscription']")
            page.wait_for_timeout(250)
            _shot(page, tmp_path, "desktop-light-subscription")

            # The foot's Use is Main's reference again; a refused save keeps the saved choice and says why.
            page.click(f"{SHEET} .chat-model-use[data-use='main']")
            page.wait_for_function(f"() => document.querySelector('{PICKER}').dataset.moved === 'false'"
                                   f" && document.querySelector('{PICKER}').dataset.saving === 'false'", timeout=10_000)
            assert _state(page, url)["model_choice"] == {"kind": "main"}
            page.route("**/api/owner/chat-model", lambda route: route.fulfill(status=503, content_type="application/json",
                       body=json.dumps({"error": "another settings write is still in progress", "saved": False, "code": "settings_busy"})))
            _open_sheet(page)
            page.locator(f"{SHEET} .chat-model-option", has_text="GPT-6-Luna").click()
            page.wait_for_function(f"() => document.querySelector('{PICKER}').dataset.saving === 'false'", timeout=10_000)
            refused = page.evaluate(FACTS)
            assert (refused["moved"], refused["name"]) == ("false", "GPT-6.1-Sol"), refused
            assert page.locator(".toast", has_text="Could not change this chat's model").count() >= 1
            assert _state(page, url)["model_choice"] == {"kind": "main"}
        finally:
            browser.close()


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_phone_sheet_rises_from_the_bottom_with_the_rail_above_the_routes(direct_server_with_data, tmp_path, theme):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    _seed(server)
    with sync_playwright() as pw:
        browser, page = _launch(pw, "webkit", viewport={"width": 390, "height": 844}, device_scale_factor=3,
                                is_mobile=True, has_touch=True)
        try:
            _open_chat(page, server["url"], theme)
            closed = page.evaluate(FACTS)
            assert not closed["docScroll"] and closed["trigger"]["right"] <= 390, closed
            _assert_phone_line(closed, "GPT-6.1-Sol")
            _shot(page, tmp_path, f"phone-{theme}-closed")
            page.tap(TRIGGER)
            page.wait_for_selector(f"{SHEET} .chat-model-option", timeout=15_000)
            page.wait_for_timeout(300)
            facts = page.evaluate(FACTS)
            sheet = facts["sheet"]
            assert abs(sheet["left"]) <= 1 and abs(sheet["right"] - 390) <= 1 and abs(sheet["bottom"] - 844) <= 1, facts
            assert sheet["top"] > 844 * 0.1, ("a sheet over a dim, the room still above it", facts)
            assert "chat-model-query" not in facts["focused"], "a touch opening leaves the keyboard down"
            rail = page.locator(f"{SHEET} .chat-model-rail").bounding_box()
            listing = page.locator(f"{SHEET} .chat-model-list").bounding_box()
            assert rail["y"] + rail["height"] <= listing["y"] + 1, "the source rail is a row above the routes"
            assert not facts["docScroll"], facts
            _shot(page, tmp_path, f"phone-{theme}-sheet")
            # The room's own choice adds `this chat` to the line; the capsules still keep their values.
            page.locator(f"{SHEET} .chat-model-option", has_text="GPT-6-Astra").tap()
            page.wait_for_function(f"() => document.querySelector('{PICKER}').dataset.saving === 'false'"
                                   f" && document.querySelector('{PICKER}').dataset.moved === 'true'", timeout=10_000)
            page.wait_for_selector(SHEET, state="detached", timeout=10_000)
            page.wait_for_timeout(300)
            chosen = page.evaluate(FACTS)
            assert chosen["scope"] == "this chat" and not chosen["docScroll"], chosen
            _assert_phone_line(chosen, "GPT-6-Astra")
            _shot(page, tmp_path, f"phone-{theme}-closed-chosen")
        finally:
            browser.close()


@pytest.mark.parametrize("accept", [(False, False), (False, True), (True, False)])
def test_rapid_stars_rebase_on_the_last_acknowledged_list(direct_server_with_data, accept):  # noqa: F811
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    _seed(server)
    with sync_playwright() as pw:
        browser, page = _launch(pw, "webkit", viewport={"width": 1512, "height": 949})
        try:
            _open_chat(page, server["url"], "dark")
            _open_sheet(page)
            attempts = []

            def save(route):
                attempts.append(route.request.post_data_json["model_favorites"])
                if accept[len(attempts) - 1]:
                    route.continue_()
                else:
                    route.fulfill(status=503, content_type="application/json", body=json.dumps({"error": "refused before write"}))

            page.route("**/api/ui/preferences", lambda route: save(route) if route.request.method == "POST" else route.continue_())
            page.wait_for_function("() => !document.querySelector('.chat-model-star[aria-disabled=\"true\"]')")
            page.evaluate("routes => routes.forEach(route => [...document.querySelectorAll('.chat-model-star')].find(node => node.dataset.star === route).click())", [SOL, ASTRA])
            expected = ([SOL] if accept[0] else []) + ([ASTRA] if accept[1] else [])
            page.wait_for_function("expected => JSON.stringify([...document.querySelectorAll('.chat-model-star[aria-pressed=\"true\"]')].map(node => node.dataset.star)) === JSON.stringify(expected)", arg=expected)
            assert attempts == [[SOL], ([SOL, ASTRA] if accept[0] else [ASTRA])]
            assert page.evaluate("async () => (await (await fetch('/api/ui/preferences')).json()).model_favorites") == expected
        finally:
            browser.close()


def test_subscription_discovery_error_is_not_absence_and_api_choices_stay_usable(direct_server_with_data):  # noqa: F811
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    _seed(server)
    with sync_playwright() as pw:
        browser, page = _launch(pw, "webkit", viewport={"width": 1512, "height": 949})
        try:
            _open_chat(page, server["url"], "dark")
            page.route("**/api/model-catalog", lambda route: route.fulfill(content_type="application/json", body=json.dumps({
                "items": [CATALOG["items"][7]], "model_sources": [],
                "errors": [{"provider_id": "claudexor", "source_id": "", "error": "engine unavailable"}], "partial": True})))
            _open_sheet(page)
            assert "engine unavailable" in page.locator(SHEET).inner_text()
            assert "No subscription sources" not in page.locator(SHEET).inner_text()
            page.click(f"{SHEET} .chat-model-tab[data-lane='api']")
            page.locator(f"{SHEET} .chat-model-option", has_text="Claude Opus 5.5").click()
            page.wait_for_function(f"() => document.querySelector('{PICKER}').dataset.saving === 'false'")
            assert _state(page, server["url"])["model_choice"]["model"] == "anthropic::claude-opus-5-5"
        finally:
            browser.close()


def test_only_connected_sources_are_offered_and_a_direct_openai_route_saves(direct_server_with_data, tmp_path):  # noqa: F811
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    _seed(server)
    with sync_playwright() as pw:
        browser, page = _launch(pw, "webkit", viewport={"width": 1512, "height": 949}, device_scale_factor=2)
        try:
            _open_chat(page, server["url"], "light")
            # OpenRouter and OpenAI keys connected; no Anthropic, DeepSeek or other key: the catalog names no such source.
            page.route("**/api/model-catalog", lambda route: route.fulfill(content_type="application/json", body=json.dumps({
                "items": [CATALOG["items"][9], *CATALOG["items"][-2:]], "errors": [], "model_sources": []})))
            _open_sheet(page)
            page.click(f"{SHEET} .chat-model-tab[data-lane='api']")
            facts = page.evaluate(FACTS)
            # The compatible endpoint stays only because the configured Fallback names it (a reference's source is kept).
            assert [label for label, _pressed in facts["rail"]] == [
                "All API sources", "OpenRouter", "OpenAI", "OpenAI-compatible endpoint"], facts
            page.click(f"{SHEET} .chat-model-source[data-source='openai']")
            _shot(page, tmp_path, "desktop-light-api-openai")
            page.locator(f"{SHEET} .chat-model-option", has_text="gpt-6.1").click()
            page.wait_for_function(f"() => document.querySelector('{PICKER}').dataset.saving === 'false'"
                                   f" && document.querySelector('{PICKER}').dataset.moved === 'true'", timeout=10_000)
            assert _state(page, server["url"])["model_choice"] == {"kind": "model", "model": "openai::gpt-6.1", "use_local": False}
            closed = page.evaluate(FACTS)
            assert (closed["name"], closed["scope"]) == ("gpt-6.1", "this chat"), closed
            assert page.locator(f"{TRIGGER} .chat-model-icon svg.chat-model-mark").count() == 1, "the trigger names OpenAI by its mark"
            _shot(page, tmp_path, "desktop-light-closed-openai")
        finally:
            browser.close()
