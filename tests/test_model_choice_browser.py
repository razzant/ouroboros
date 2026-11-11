"""The model capsule on a large catalog, and the whole model choice on the real server, at the owner's widths.

Two looks, both on the real page of a real server on its own port and data root.

``test_the_capsule_holds_on_a_large_catalog``: the Model capsule at a pane of 282 px and 581 px, on a 390 px phone and
on a wide desktop, in Chromium and WebKit. Its catalog is answered by the test (several hundred model ids, twenty-two
subscription accounts): the smoke server has no engine and no accounts to list. The composer does not move when the
strip, the search or the list opens; Model keeps its value while its neighbours fold; nothing is wider than the screen.

``test_model_choice_on_the_real_server``: nothing of the server is answered. The loopback provider lists its models,
holds marked calls and refuses one model, so the page shows what the server itself publishes: a room following Main
and with its own model, the open strip, Search, the Main sheet plain and with its lists, a running card, a queued
card, the card's popover, a work on a fallback in amber with its three exits, and the arrow of a follower after Main
changed. Each state is read from the page and pictured at the four widths in both themes for the owner
(``OUROBOROS_UI_EVIDENCE_DIR``). The circle editor and the place sheet's Model field close the set; a place needs a
connector the smoke server does not have, so the places list is answered by the test there, as in
``test_circle_model_browser``.
"""
from __future__ import annotations

import json
import os
import pathlib
import uuid

import pytest

from tests import fixtures_mock_llm
from tests._model_choice_browser import (
    _CAPSULE, _CARD, MAIN, NEXT_STEP, OTHER, PICKED, SOURCE, Provider, _announce, _capsule_ready, _close_project,
    _open_project, _post, _state, _work_of,
)
from tests.test_circle_model_browser import _catalog as _places_catalog, _circle, _place
from tests.test_effort_range_browser import _launch
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 -- fixture

pytestmark = [pytest.mark.ui_browser]

BROKEN, FALLBACK = f"{SOURCE}::mock-broken", f"{SOURCE}::mock-fallback"
# (tag, transcript width the desk viewport is sized to; None: the viewport as given)
DESK = [("282", 282), ("581", 581), ("wide", None)]
WIDE = {"width": 1512, "height": 949}
PHONE = {"viewport": {"width": 390, "height": 844}, "is_mobile": True, "has_touch": True}


def _fit(page, transcript: int | None) -> None:
    """A desk viewport sized until the Main transcript is exactly as wide as asked; ``None`` is the wide desktop."""
    if transcript is None:
        page.set_viewport_size(WIDE)
        page.wait_for_timeout(400)
        return
    page.set_viewport_size({"width": 1000 if transcript > 400 else 360, "height": 900})
    for _ in range(5):
        page.wait_for_timeout(250)
        now = page.evaluate("() => document.querySelector('#chat-messages').clientWidth")
        if now == transcript:
            return
        size = page.viewport_size
        page.set_viewport_size({"width": size["width"] + transcript - now, "height": size["height"]})
    assert page.evaluate("() => document.querySelector('#chat-messages').clientWidth") == transcript


def _theme(page, theme: str) -> None:
    page.evaluate(f"() => window.ouroTheme.set('{theme}')")
    page.wait_for_function(f"() => document.documentElement.dataset.theme === '{theme}'")
    page.wait_for_timeout(250)  # inks settle: a control's colour eases


def _quiet(page) -> None:
    """No toast left over the page. The pointer goes to a corner first: a toast under it does not expire."""
    page.mouse.move(1, 1)
    page.wait_for_function("() => !document.querySelector('.toast')", timeout=20_000)


_WIDE = "() => document.documentElement.scrollWidth - window.innerWidth"
_FACTS_OVER = """() => [...document.querySelectorAll('.chat-model-row-facts')]
    .filter((node) => node.getClientRects().length && node.scrollWidth > node.clientWidth + 0.5)
    .map((node) => node.textContent)"""
_LIST_GROUPS = """() => {
    const list = document.querySelector('.ui-popup.chat-model-list .chat-model-rows, .card-model-pop .chat-model-rows');
    const groups = [{heading: '', models: []}];
    for (const node of list.children) {
        if (node.classList.contains('chat-model-section')) groups.push({heading: node.textContent, models: []});
        else if (node.classList.contains('chat-model-row')) groups.at(-1).models.push(node.querySelector('.chat-model-row-id').textContent);
    }
    return groups;
}"""
_FACT_LINES = """() => [...document.querySelectorAll('.ui-popup.chat-model-list .chat-model-row')].map(row => {
    const facts = row.querySelector('.chat-model-row-facts');
    const name = row.querySelector('.chat-model-row-id').getBoundingClientRect();
    const range = document.createRange();
    range.selectNodeContents(facts);
    return {text: facts.textContent, nameLeft: name.left, nameBottom: name.bottom,
        lines: [...range.getClientRects()].map(r => ({left: r.left, top: r.top, right: r.right}))};
})"""

# ---------------------------------------------------------------------------------------------------------------------
# 1. The capsule on a large catalog
# ---------------------------------------------------------------------------------------------------------------------

ASTRA = "claudexor::codex=gpt-6-astra"


def _large_catalog() -> dict:
    """Several hundred model ids over five API providers and two subscription sources with twenty-two accounts."""
    items = []
    for provider, label, count in (("openrouter", "OpenRouter", 240), ("openai", "OpenAI", 60), ("anthropic", "Anthropic", 30),
                                   ("deepseek", "DeepSeek", 20), ("gemini", "Google", 40)):
        for n in range(count):
            model = f"{provider}-vendor-{n % 12}/model-{n:03d}-long-name-preview"
            items.append({"provider_id": provider, "provider": label, "id": model, "name": model,
                          "value": model if provider == "openrouter" else f"{provider}::{model}"})
    for source, label, accounts, models in (("codex", "Codex", 14, ("gpt-6-astra", "gpt-6.1-sol", "gpt-6-mini", "gpt-5.5-codex")),
                                            ("claude", "Claude", 8, ("claude-opus-5.5", "claude-sonnet-5.5", "claude-haiku-5.5"))):
        for account in range(accounts):
            for model in models:
                items.append({"provider_id": "claudexor", "provider": label, "source_id": source, "id": model, "name": model,
                              "value": f"claudexor::{source}={model}", "credential_profile_id": f"{source}-account-{account:02d}",
                              "availability": "available", "max_context_window": 872000})
    assert len({row["id"] for row in items}) > 390 and len({row.get("credential_profile_id") for row in items} - {None}) == 22
    return {"items": items, "errors": [], "model_sources": [{"id": "codex", "label": "Codex", "credentialHarness": "codex"},
                                                             {"id": "claude", "label": "Claude", "credentialHarness": "claude"}]}


_ROW = """() => {
    const box = (node) => { const r = node.getBoundingClientRect(); return [r.left, r.top, r.width, r.height].map((v) => Math.round(v * 10) / 10); };
    const row = document.querySelector('#page-chat .chat-toolbar-row');
    const model = document.querySelector('#chat-model-mode');
    const shown = (node) => Boolean(node) && node.getClientRects().length > 0 && node.getBoundingClientRect().width > 1;
    const list = document.querySelector('.ui-popup.chat-model-list');
    const listBox = list ? list.getBoundingClientRect() : null;
    return {
        area: box(document.querySelector('#chat-input-area')), field: box(document.querySelector('#chat-input')),
        messages: box(document.querySelector('#chat-messages')), row: box(row),
        value: model.querySelector('.chat-capsule-value').textContent, valueShown: shown(model.querySelector('.chat-capsule-value')),
        glyphs: row.dataset.ownerGlyphs || '', open: model.dataset.open || 'false', search: model.dataset.search || '',
        neighbours: [...row.querySelectorAll('.chat-capsule:not(.chat-model-mode)')].map((el) => shown(el.querySelector('.chat-capsule-value, .chat-subagents-value'))),
        wide: document.documentElement.scrollWidth - window.innerWidth,
        out: [...row.querySelectorAll('.chat-capsule')].filter((el) => el.getClientRects().length && el.getBoundingClientRect().right > window.innerWidth + 0.5).map((el) => el.className),
        rows: list ? list.querySelectorAll('.chat-model-row').length : 0,
        factsOver: list ? [...list.querySelectorAll('.chat-model-row-facts')]
            .filter((node) => node.scrollWidth > node.clientWidth + 0.5).map((node) => node.textContent) : [],
        listInside: listBox ? listBox.left >= -0.5 && listBox.right <= window.innerWidth + 0.5 && listBox.top >= -0.5 && listBox.bottom <= window.innerHeight + 0.5 : null,
        listOverRow: listBox ? listBox.bottom <= row.getBoundingClientRect().top + 1 : null,
    };
}"""


def _capsule_walk(page, tag: str) -> None:
    def still(now: dict, was: dict, what: str) -> None:
        assert now["area"] == was["area"] and now["field"] == was["field"] and now["messages"] == was["messages"], \
            (tag, what, "the composer and the transcript stay where they were", was, now)
        assert now["wide"] <= 0 and not now["out"] and not now["factsOver"], (tag, what, now)
        assert now["value"] == "gpt-6-astra", (tag, what, "the capsule still holds Main's model", now)

    closed = page.evaluate(_ROW)
    assert closed["wide"] <= 0 and not closed["out"], (tag, closed)
    assert closed["value"] == "gpt-6-astra" and closed["valueShown"], (tag, "Model shows its value closed", closed)
    if closed["glyphs"] == "true":
        assert not any(closed["neighbours"]), (tag, "the neighbours folded to their glyphs; Model did not", closed)

    page.click("#chat-model-mode .chat-model-head")
    page.wait_for_selector('#chat-model-mode[data-open="true"] .chat-model-find', timeout=10_000)
    page.wait_for_timeout(500)  # the strip's own opening
    opened = page.evaluate(_ROW)
    still(opened, closed, "the strip open")

    page.click("#chat-model-mode .chat-model-find")
    page.wait_for_selector(".ui-popup.chat-model-list .chat-model-row", timeout=20_000)
    page.wait_for_timeout(400)
    listed = page.evaluate(_ROW)
    still(listed, closed, "the search and its list open")
    assert listed["search"] == "true" and listed["listInside"] and listed["listOverRow"], (tag, "the list floats above the row, on screen", listed)
    assert listed["rows"] >= 5, (tag, "Main, then the sources", listed)

    page.fill("#chat-model-mode .chat-model-query", "model-1")
    page.wait_for_function("() => document.querySelectorAll('.ui-popup.chat-model-list .chat-model-row').length >= 60", timeout=10_000)
    page.wait_for_timeout(300)
    typed = page.evaluate(_ROW)
    still(typed, closed, "sixty suggestions listed")
    assert typed["listInside"] and typed["listOverRow"], (tag, typed)
    assert page.evaluate("() => document.querySelector('.ui-popup.chat-model-list').innerText.includes('keep typing')"), "the rest wait for another letter"

    # Search's former position can be over a neighbour once the wide strip closes. Move away before Escape
    # so the neighbour's legitimate hover opening does not become the closed-row comparison.
    page.mouse.move(1, 1)
    for _ in range(3):  # clears the field, leaves the search, closes the capsule
        page.keyboard.press("Escape")
        page.wait_for_timeout(150)
    page.wait_for_function("() => document.querySelector('#chat-model-mode').dataset.open !== 'true' && !document.querySelector('.ui-popup.chat-model-list')",
                           timeout=10_000)
    page.wait_for_timeout(500)
    again = page.evaluate(_ROW)
    assert page.locator('.chat-toolbar-row .chat-capsule[data-open="true"]').count() == 0
    assert again["area"] == closed["area"] and again["row"] == closed["row"], (tag, "closed again, as it was", closed, again)
    assert again["valueShown"] and again["neighbours"] == closed["neighbours"], (tag, "Model shows its value again", again)


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_the_capsule_holds_on_a_large_catalog(direct_server_with_data, engine):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    catalog = json.dumps(_large_catalog())

    def serve(page) -> None:
        # Main is a subscription model of that catalog, as the owner's is: the state's `main_route` alone is replaced.
        page.route("**/api/model-catalog*", lambda route: route.fulfill(content_type="application/json", body=catalog))
        page.route("**/api/state*", lambda route: route.fulfill(json={**route.fetch().json(), "main_route": {"model": ASTRA, "use_local": False}}))

    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport=WIDE)
        try:
            serve(page)
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            _capsule_ready(page, "#chat-model-mode")
            for tag, transcript in DESK:
                _fit(page, transcript)
                _capsule_walk(page, f"{engine}-{tag}")
        finally:
            browser.close()
        browser, page = _launch(pw, engine, **PHONE)
        try:
            serve(page)
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            _capsule_ready(page, "#chat-model-mode")
            _capsule_walk(page, f"{engine}-phone")
        finally:
            browser.close()


# ---------------------------------------------------------------------------------------------------------------------
# 2. The whole choice on the real server, pictured for the owner
# ---------------------------------------------------------------------------------------------------------------------

def _task(page, project: dict, marker: str, title: str) -> str:
    """A task of a Project room, admitted by the server (work launched over the API lives in a project's thread)."""
    return _post(page, "/api/tasks", {"description": f"{marker} {title}", "title": title, "project_id": project["id"]})["task_id"]


def _fit_room(page, project: dict, transcript: int | None) -> str:
    """The Project room open with its own transcript as wide as asked. A narrow window shows the room as the page
    itself; on a desk it is the panel beside Main, whose width is the owner's (`--project-panel-width`, the splitter)."""
    if transcript is not None and transcript < 320:
        _fit(page, transcript)
        prefix = _open_project(page, project)
        page.wait_for_timeout(300)
        return prefix
    page.set_viewport_size(WIDE)
    page.evaluate("() => document.documentElement.style.removeProperty('--project-panel-width')")
    prefix = _open_project(page, project)
    page.wait_for_timeout(300)
    if transcript is None:
        return prefix
    width = "(prefix) => document.querySelector('#' + prefix + '-messages').clientWidth"
    panel = "() => document.getElementById('project-panel').getBoundingClientRect().width"
    for _ in range(5):
        now = page.evaluate(width, prefix)
        if now == transcript:
            return prefix
        page.evaluate("(px) => document.documentElement.style.setProperty('--project-panel-width', px + 'px')", page.evaluate(panel) + transcript - now)
        page.wait_for_timeout(300)
    assert page.evaluate(width, prefix) == transcript
    return prefix


def _activity(page, task: str, activity: str) -> None:
    page.wait_for_function("async ([id, activity]) => { const s = await (await fetch('/api/state', { cache: 'no-store' })).json();"
                           " return s.works.some((w) => w.episode.task_id === id && w.activity === activity); }", arg=[task, activity], timeout=90_000)


def _card(task: str) -> str:
    return f'#project-panel .chat-live-card[data-task-id="{task}"]'


def _show(page, task: str) -> None:
    page.wait_for_selector(_card(task), timeout=60_000)
    page.evaluate("(sel) => document.querySelector(sel).scrollIntoView({ block: 'center' })", _card(task))
    page.wait_for_timeout(300)


def test_model_choice_on_the_real_server(direct_server_with_data, monkeypatch, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    server["stop_server"]()
    settings_path = server["data_dir"] / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings.update(OUROBOROS_MODEL=MAIN, OUROBOROS_MODEL_LIGHT=f"{SOURCE}::mock-light", OUROBOROS_MODEL_FALLBACKS=FALLBACK,
                    OUROBOROS_TASK_REVIEW_MODE="off", OUROBOROS_SAFETY_MODE="off", OUROBOROS_MAX_WORKERS=2,
                    OUROBOROS_SUBAGENTS=json.dumps({"enabled": False, "items": []}))
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    provider = Provider(models=("mock-main", "mock-light", "mock-fallback", "mock-picked", "mock-other", "mock-broken"), broken={"mock-broken"})
    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", lambda handler: provider.post(handler))
    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_GET", lambda handler: provider.get(handler), raising=False)
    server["start_server"]()
    shots = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path)))
    shots.mkdir(parents=True, exist_ok=True)
    seen: dict = {}

    with sync_playwright() as pw:
        browser, desk = _launch(pw, "chromium", viewport=WIDE)
        phone_browser, phone = _launch(pw, "chromium", **PHONE)
        pages = [(tag, desk, transcript) for tag, transcript in DESK] + [("phone", phone, None)]

        def picture(page, number: str, name: str, tag: str) -> None:
            _quiet(page)
            page.wait_for_timeout(350)  # a transient hover hint has left before the frame
            for theme in ("dark", "light"):
                _theme(page, theme)
                page.screenshot(path=str(shots / f"{number}-{name}-{tag}-{theme}.png"), animations="disabled")
                assert page.evaluate(_WIDE) <= 0, (number, name, tag, "nothing is wider than the screen")
                assert page.evaluate(_FACTS_OVER) == [], (number, name, tag, "the route's facts fit inside its row")
                if number in {"04", "09", "10"}:
                    groups = page.evaluate(_LIST_GROUPS)
                    assert [group["heading"] for group in groups] == ["", "Recent", "OpenAI-compatible endpoint"], groups
                    for group in groups:
                        assert len(group["models"]) == len(set(group["models"])), (number, tag, "each section lists a model once", group)
                    assert sum(group["models"].count("mock-main") for group in groups) == 1, groups
                    seen[f"list-{number}-{tag}-{theme}"] = groups
                if number == "04":
                    facts = page.evaluate(_FACT_LINES)
                    seen[f"facts-{tag}-{theme}"] = facts
                    if tag == "282":
                        wrapped = [fact for fact in facts if len(fact["lines"]) > 1]
                        assert wrapped, (tag, "the long source facts wrap at this width", facts)
                        for fact in wrapped:
                            first = fact["lines"][0]
                            assert first["top"] >= fact["nameBottom"] - 0.5, fact
                            assert abs(first["left"] - fact["nameLeft"]) <= 0.5, fact
                            assert all(abs(line["left"] - first["left"]) <= 0.5 for line in fact["lines"][1:]), fact
            _theme(page, "dark")

        def everywhere(number: str, name: str, stage, leave=lambda page: None) -> None:
            """One state of the Main chat at every width in both themes: `stage(page)` shows it, `leave(page)` puts it away."""
            for tag, page, transcript in pages:
                if page is desk:
                    _fit(page, transcript)
                stage(page)
                picture(page, number, name, tag)
                leave(page)

        def in_room(project: dict, number: str, name: str, stage=lambda page: None, leave=lambda page: None) -> None:
            """One state of a Project room, the room's own transcript at every width, in both themes."""
            for tag, page, transcript in pages:
                if page is desk:
                    _fit_room(page, project, transcript)
                else:
                    _open_project(page, project)
                page.wait_for_timeout(300)
                stage(page)
                picture(page, number, name, tag)
                leave(page)
                _close_project(page)

        def load(page) -> None:
            page.goto(server["url"], wait_until="domcontentloaded", timeout=30_000)
            _capsule_ready(page, "#chat-model-mode")

        def open_strip(page, capsule: str = "#chat-model-mode") -> None:
            page.click(f"{capsule} .chat-model-head")
            page.wait_for_selector(f'{capsule}[data-open="true"] .chat-model-find', timeout=10_000)
            page.wait_for_timeout(500)

        def close_capsule(page) -> None:
            for _ in range(3):
                page.keyboard.press("Escape")
                page.wait_for_timeout(120)
            page.wait_for_function("() => !document.querySelector('.chat-model-mode[data-open=\"true\"]') && !document.querySelector('.ui-popup.chat-model-list')")

        def ask_main(page, route: str) -> None:
            """The Main chat's pick up to its confirming sheet."""
            open_strip(page)
            page.click("#chat-model-mode .chat-model-find")
            row = f'.ui-popup.chat-model-list .chat-model-row[data-row="route|{route}"]'
            page.wait_for_selector(row, timeout=20_000)
            page.click(row)
            page.wait_for_selector(".confirm-dialog [data-main-followers]", timeout=20_000)
            page.wait_for_timeout(300)
            proposed = page.evaluate("""() => {
                const model = document.querySelector('.chat-model-sheet-row b');
                return { text: model.textContent, whole: model.scrollWidth <= model.clientWidth };
            }""")
            assert proposed == {"text": route.split("::", 1)[-1], "whole": True}, proposed

        def cancel_sheet(page) -> None:
            page.click(".confirm-dialog [data-confirm-cancel]")
            page.wait_for_function("() => !document.querySelector('.confirm-dialog')")

        try:
            load(desk)
            load(phone)
            alpha = _post(desk, "/api/projects", {"name": "Research"})["project"]
            site = _post(desk, "/api/projects", {"name": "Site"})["project"]
            logs = _post(desk, "/api/projects", {"name": "Logs"})["project"]
            capsule = f"#pchat-{alpha['id']}-model-mode"

            # --- 01: a room follows Main until it chooses. 05: the Main sheet with nothing to list.
            in_room(alpha, "01", "capsule-follows-main")
            _open_project(desk, alpha)
            follows = desk.evaluate(_CAPSULE, capsule)
            assert (follows["name"], follows["value"], follows["own"]) == ("Main", "mock-main", "false"), follows
            _close_project(desk)

            def plain_sheet(page) -> None:
                ask_main(page, OTHER)

            everywhere("05", "main-sheet-plain", plain_sheet, cancel_sheet)
            ask_main(desk, OTHER)
            plain = desk.evaluate("""() => ({ count: document.querySelector('[data-main-followers]').innerText,
                rows: document.querySelectorAll('.confirm-dialog .chat-model-off').length, ok: document.querySelector('[data-confirm-ok]').innerText })""")
            assert plain == {"count": "No running work follows Main now.", "rows": 0, "ok": "Change Main"}, plain
            cancel_sheet(desk)
            assert _state(desk)["main_route"]["model"] == MAIN, "a cancelled sheet writes nothing"

            # --- The room chooses its own model. 02: its closed capsule. 03: Main's open strip. 04: Search with its list.
            _open_project(desk, alpha)
            open_strip(desk, capsule)
            desk.click(f"{capsule} .chat-model-find")
            desk.click(f'.ui-popup.chat-model-list .chat-model-row[data-row="route|{PICKED}"]')
            desk.wait_for_function("(sel) => document.querySelector(sel).dataset.own === 'true'", arg=capsule, timeout=20_000)
            own = desk.evaluate(_CAPSULE, capsule)
            assert own["value"] == "mock-picked" and own["name"] != "Main", own
            _close_project(desk)
            phone.reload(wait_until="domcontentloaded")
            _capsule_ready(phone, "#chat-model-mode")

            def room_own(page) -> None:
                page.wait_for_function("(sel) => document.querySelector(sel).dataset.own === 'true'", arg=capsule, timeout=20_000)
                page.wait_for_timeout(300)

            in_room(alpha, "02", "capsule-room-model", room_own)
            everywhere("03", "strip-open", open_strip, close_capsule)
            open_strip(desk)
            strip = desk.evaluate(_CAPSULE, "#chat-model-mode")
            assert strip["strip"][0] == "mock-main" and "mock-picked" in strip["strip"] and strip["strip"][-1] == "Search", strip
            close_capsule(desk)

            def search(page) -> None:
                open_strip(page)
                page.click("#chat-model-mode .chat-model-find")
                page.wait_for_selector(f'.ui-popup.chat-model-list .chat-model-row[data-row="route|{OTHER}"]', timeout=20_000)
                page.wait_for_timeout(300)

            everywhere("04", "search-list", search, close_capsule)

            # --- Two workers in separate Projects (each Project admits one root): the next two wait behind them.
            for marker in ("ROOM_WIRE_BLOCK", "ROOM_WIRE_FOLLOW", "ROOM_WIRE_FALL"):
                provider.hold(marker)
            _task(desk, logs, "ROOM_WIRE_BLOCK", "Sort the inbox")
            follower = _task(desk, site, "ROOM_WIRE_FOLLOW", "Compare the two parsers")
            assert provider.held("ROOM_WIRE_BLOCK") and provider.held("ROOM_WIRE_FOLLOW"), "the two running works never reached their held calls"
            falling = _task(desk, logs, "ROOM_WIRE_FALL", "Read last night's logs")
            _activity(desk, falling, "queued")
            # The owner gives the waiting work a model before it starts; that model will refuse it.
            given = _post(desk, f"/api/tasks/{falling}/model", {"request_id": uuid.uuid4().hex, "action": "set", "model": BROKEN, "use_local": False})
            assert given["applied"] == "at_start" and given["lead"] == {"model": BROKEN, "use_local": False, "basis": "owner"}, given
            queued = _task(desk, logs, "ROOM_WIRE_QUEUED", "Draft the release letter")
            _activity(desk, queued, "queued")
            state = _state(desk)
            _announce(desk, server["data_dir"], state, falling, "Queued: waiting for a worker")
            _announce(desk, server["data_dir"], state, queued, "Queued: waiting for a worker")
            for page in (desk, phone):
                page.reload(wait_until="domcontentloaded")
                _capsule_ready(page, "#chat-model-mode")

            # --- 07: a running card. 08: a queued card. 09: the card's popover.
            def fact_of(page, task: str) -> dict:
                page.wait_for_function("(sel) => Boolean(document.querySelector(sel + ' .card-model-fact'))", arg=_card(task), timeout=60_000)
                return page.evaluate(_CARD, _card(task))

            def short_card(page, task: str) -> None:
                _show(page, task)
                fact = fact_of(page, task)
                assert fact["modelWhole"], ("the source gives way before the short model name", fact)
                assert fact["separatorAttached"], ("the separator stays with the model or its pending route", fact)

            in_room(site, "07", "card-running", lambda page: short_card(page, follower))
            _fit_room(desk, site, 581)
            running = fact_of(desk, follower)
            seen["running"] = running
            assert (running["chip"], running["tag"], running["why"], running["basis"]) == ("Working", "BUTTON", "", "main"), running
            assert running["fact"].endswith("· mock-main"), running
            _close_project(desk)
            in_room(logs, "08", "card-queued", lambda page: short_card(page, queued))
            _fit_room(desk, logs, 581)
            waiting = fact_of(desk, queued)
            seen["queued"] = waiting
            assert (waiting["chip"], waiting["tag"], waiting["basis"]) == ("Queued", "BUTTON", "main"), waiting
            given_card = fact_of(desk, falling)
            assert given_card["fact"].endswith("· mock-broken") and given_card["why"] == "you chose it", given_card

            def popover(page) -> None:
                _show(page, follower)
                page.click(f"{_card(follower)} button.card-model-fact")
                page.wait_for_selector(".card-model-pop .chat-model-row", timeout=20_000)
                page.wait_for_timeout(350)

            def close_popover(page) -> None:
                page.keyboard.press("Escape")
                page.wait_for_function("() => !document.querySelector('.card-model-pop')")

            _close_project(desk)
            in_room(site, "09", "card-popover", popover, close_popover)
            _fit_room(desk, site, 581)
            popover(desk)
            said = desk.evaluate("""() => ({ title: document.querySelector('.card-model-pop-title').innerText,
                acts: document.querySelectorAll('.card-model-pop .card-model-act').length,
                main: document.querySelector('.card-model-pop .chat-model-row').innerText.replace(/\\s+/g, ' ').trim(),
                recent: document.querySelector('.card-model-pop .chat-model-section')?.innerText, rule: document.querySelector('.card-model-pop .chat-model-rule').innerText })""")
            seen["popover"] = said
            assert said == {"title": "Model for this work · follows Main", "acts": 0, "main": "mock-main Main this work follows Main",
                            "recent": "Recent", "rule": NEXT_STEP}, said
            close_popover(desk)
            _close_project(desk)

            # --- 10: the blocker ends, the waiting work starts on the model it was given, is refused and goes to the
            #         fallback from Settings: amber, with its three exits.
            provider.release("ROOM_WIRE_BLOCK")
            assert provider.held("ROOM_WIRE_FALL"), "the refused work never reached its held call on the fallback"
            assert provider.models_of("ROOM_WIRE_FALL")[:2] == ["mock-broken", "mock-fallback"], provider.models_of("ROOM_WIRE_FALL")
            _fit_room(desk, logs, 581)
            desk.wait_for_function("(sel) => Boolean(document.querySelector(sel + ' .card-model-fact.model-fallback'))", arg=_card(falling), timeout=60_000)
            fell_work = _work_of(_state(desk), falling)
            lead = fell_work["lead"]
            seen["fallback_lead"] = lead
            assert (lead["model"], lead["basis"], lead["role"]) == (FALLBACK, "fallback", "fallback:0"), lead
            assert lead["primary"] == {"model": BROKEN, "use_local": False, "basis": "owner"} and lead["cause"] and "T" in lead["since"], lead
            fell = fact_of(desk, falling)
            seen["fallback"] = fell
            assert fell["fact"].startswith("fallback from Settings · mock-fallback · since ") and fell["why"] == "left mock-broken", fell

            def exits(page) -> None:
                page.wait_for_function("(sel) => Boolean(document.querySelector(sel + ' .card-model-fact.model-fallback'))", arg=_card(falling), timeout=60_000)
                _show(page, falling)
                page.click(f"{_card(falling)} button.card-model-fact")
                page.wait_for_selector(".card-model-pop .card-model-act", timeout=20_000)
                page.wait_for_timeout(350)

            _close_project(desk)
            in_room(logs, "10", "card-fallback", exits, close_popover)
            _fit_room(desk, logs, 581)
            exits(desk)
            offered = desk.evaluate("""() => ({ title: document.querySelector('.card-model-pop-title').innerText,
                acts: [...document.querySelectorAll('.card-model-pop .card-model-act')].map((node) => [node.dataset.act, node.querySelector('.card-model-act-label').innerText]) })""")
            seen["exits"] = offered
            assert offered["acts"] == [["return", "Return to mock-broken"], ["wait", "Wait for mock-broken"], ["keep", "Keep on mock-fallback"]], offered
            assert offered["title"].startswith("Left mock-broken at "), offered
            close_popover(desk)
            _close_project(desk)

            # --- 06: the Main sheet with its lists; then Main changes. 11: the follower's arrow.
            def lists_sheet(page) -> None:
                ask_main(page, OTHER)

            everywhere("06", "main-sheet-lists", lists_sheet, cancel_sheet)
            ask_main(desk, OTHER)
            sheet = desk.evaluate("""() => { const d = document.querySelector('.confirm-dialog'); const text = (n) => (n ? n.innerText.replace(/\\s+/g, ' ').trim() : null);
                return { count: text(d.querySelector('[data-main-followers]')), heads: [...d.querySelectorAll('.chat-model-sheet-sub')].map(text),
                    rows: [...d.querySelectorAll('.chat-model-off')].map((row) => [text(row.querySelector('.chat-model-off-name')),
                        row.querySelector('.ui-checkbox') ? row.querySelector('.ui-checkbox').checked : null, text(row.querySelector('.model-fallback'))]) }; }""")
            seen["sheet"] = sheet
            assert sheet["count"] == "1 running work follows Main now.", sheet
            assert sheet["heads"] == ["Not on Main now · tick to move too", "Rooms that start new work on their own model · 1"], sheet
            rows = {row[0]: row for row in sheet["rows"]}
            assert {name: row[1] for name, row in rows.items()} == {"Read last night's logs": False, "Research": False}, sheet
            assert rows["Read last night's logs"][2].startswith("fallback since ") and rows["Research"][2] is None, sheet
            desk.click(".confirm-dialog [data-confirm-ok]")
            desk.wait_for_function("() => document.querySelector('#chat-model-mode .chat-capsule-value').textContent === 'mock-other'", timeout=20_000)
            assert _state(desk)["main_route"] == {"model": OTHER, "use_local": False}

            def arrow(page) -> None:
                page.wait_for_function("(sel) => document.querySelector(sel + ' .card-model-pending')?.innerText.includes('mock-other')",
                                       arg=_card(follower), timeout=60_000)
                short_card(page, follower)

            _quiet(desk)
            in_room(site, "11", "card-pending", arrow)
            _fit_room(desk, site, 581)
            moving = fact_of(desk, follower)
            seen["pending"] = moving
            assert moving["fact"].endswith("· mock-main") and moving["pending"] == "→ mock-other at the next step", moving
            assert _work_of(_state(desk), follower)["lead"] == {"model": MAIN, "use_local": False, "basis": "main", "pending": {"model": OTHER, "use_local": False}}
            assert _work_of(_state(desk), queued)["lead"] == {"model": OTHER, "use_local": False, "basis": "main"}, "a queued follower simply starts on the new Main"

            # --- One exit, on the real server: Keep makes the fallback the work's own choice; amber ends with the fact.
            _close_project(desk)
            _fit_room(desk, logs, 581)
            exits(desk)
            with desk.expect_response(lambda r: r.url.endswith(f"/api/tasks/{falling}/model") and r.request.method == "POST") as kept_reply:
                desk.click('.card-model-pop [data-act="keep"]')
            accepted = kept_reply.value.json()
            assert kept_reply.value.status == 200 and accepted["applied"] == "next_step", accepted
            assert accepted["lead"]["basis"] == "fallback", "the call in flight still runs on its fallback"
            provider.advance("ROOM_WIRE_FALL")
            assert provider.held("ROOM_WIRE_FALL"), "Keep never reached its next ordinary call"
            desk.wait_for_function("(sel) => !document.querySelector(sel + ' .card-model-fact').classList.contains('model-fallback')", arg=_card(falling), timeout=30_000)
            kept = fact_of(desk, falling)
            seen["kept"] = kept
            assert provider.models_of("ROOM_WIRE_FALL")[-1] == "mock-fallback"
            assert kept["fact"].endswith("· mock-fallback") and kept["why"] == "you chose it" and kept["basis"] == "own", kept
            assert _work_of(_state(desk), falling)["lead"] == {"model": FALLBACK, "use_local": False, "basis": "owner"}
            _close_project(desk)
            desk.set_viewport_size(WIDE)

            # --- 12, 13: the circle editor and the place sheet's Model field (the places list is answered here).
            glm, gpt = "z-ai/glm-5.3", "openai::gpt-6"
            circles = [_circle("team", "Team", glm, 3), _circle("home", "Home", "", 4)]
            places = [_place("alpha", "Chat · alpha", "team", "", {"value": glm, "layer": "circle"}),
                      _place("beta", "Chat · beta", "team", gpt, {"value": gpt, "layer": "place"})]
            for page in (desk, phone):
                page.route("**/api/places", lambda route: route.fulfill(json={
                    "ok": True, "places": places, "circles": circles, "migration_problems": [],
                    "connectors": [{"connector": "chat-bridge", "label": "Chat", "destination": "vendor:chat",
                                    "readiness": {"text": "Connector on", "tone": "ok"}}]}) if route.request.method == "GET" else route.continue_())
                page.route("**/api/places/*/rooms", lambda route: route.fulfill(json={
                    "place_id": route.request.url.split("/")[-2], "rooms": [], "coverage": {"discovery": "arrivals", "importing": False, "gaps": False}}))
                page.route("**/api/model-catalog*", lambda route: route.fulfill(json=_places_catalog()))

            def back_to_chat(page) -> None:
                page.reload(wait_until="domcontentloaded")
                page.wait_for_selector("#chat-input", timeout=30_000)
                page.wait_for_timeout(300)

            def circle_editor(page) -> None:
                page.evaluate("() => window.dispatchEvent(new CustomEvent('ouro:open-circles', { detail: { circle: 'team' } }))")
                page.wait_for_selector('.circle-fold[data-circle="team"][open] [data-place-open="beta"]', timeout=30_000)
                page.click('.circle-fold[data-circle="team"] [data-circles-act="edit"]')
                page.wait_for_selector('[data-circles-draft="team:model"]', timeout=10_000)
                page.wait_for_timeout(300)

            back_to_chat(desk)
            back_to_chat(phone)
            everywhere("12", "circle-editor", circle_editor, back_to_chat)

            def place_sheet(page) -> None:
                page.evaluate("() => window.dispatchEvent(new CustomEvent('ouro:open-circles', { detail: { circle: 'team' } }))")
                page.wait_for_selector('.circle-fold[data-circle="team"][open] [data-place-open="alpha"]', timeout=30_000)
                page.click('[data-place-open="alpha"]')
                page.wait_for_selector("dialog[open] .place-more", timeout=10_000)
                page.click("dialog[open] .place-more > summary")
                page.evaluate("() => { const body = document.querySelector('dialog[open] [data-place-body]'); body.scrollTop = body.scrollHeight; }")
                page.wait_for_timeout(300)
                assert page.locator('dialog[open] [data-place-draft="model"]').get_attribute("placeholder") == "Uses Team's model: glm-5.3"

            everywhere("13", "place-sheet-model", place_sheet, back_to_chat)
        finally:
            provider.release_all()
            (shots / "real-server-seen.json").write_text(json.dumps(seen, indent=2, default=str), encoding="utf-8")
            (shots / "real-server-models.json").write_text(json.dumps(
                [{"model": body.get("model"), "marker": provider.marker_of(body)} for body in provider.received]), encoding="utf-8")
            try:
                desk.unroute_all()
                (shots / "real-server-state.json").write_text(json.dumps(_state(desk), indent=1, default=str), encoding="utf-8")
                desk.screenshot(path=str(shots / "zz-last-desk.png"))
                phone.screenshot(path=str(shots / "zz-last-phone.png"))
            except Exception:  # noqa: BLE001 - diagnostics only: a page may be gone
                pass
            phone_browser.close()
            browser.close()
