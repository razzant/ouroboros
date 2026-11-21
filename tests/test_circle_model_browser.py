"""A circle's model and what a place inherits, on a phone (docs/DESIGN.md "Circles and Add", "Place sheet").

Owner, 10.11.2026: a circle may have a model; a place inherits it and may override it; the default is the
consciousness model. The real page at 390 px, with the places list and the two writes answered by the test in the
contract's shapes: the head names a circle's model only when it has one, a place row a model only when the place
overrides, and the open circle says how many places keep their own; Edit holds the model in the place sheet's chooser
(the local model is not offered there) and saves it with the circle under its revision; the place sheet's empty
Model field says whose model the place uses, and clearing an override returns the place to it. Nothing is wider
than the screen.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from tests.test_effort_range_browser import _launch
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture import

pytestmark = [pytest.mark.ui_browser]

GLM, GPT, ASTRA = "z-ai/glm-5.3", "openai::gpt-6", "claudexor::codex=gpt-6-astra"


def _place(place_id: str, label: str, circle: str, model: str, route: dict, **extra) -> dict:
    return {
        "place_id": place_id, "connector": "chat-bridge", "provider": "chat", "account_id": "acme", "conversation_id": place_id,
        "label": label, "state": "active", "wakes": [], "notes": [], "owner_accounts": [], "working_folder": "", "model": model,
        "model_route": route, "context_topics": [], "legacy_binding_ids": [], "revision": 2, "thread_id": "", "destination": "vendor:chat",
        "scope": "conversation", "contribution_id": "", "reply_destination": None, "icon": {}, "readiness": {"text": "Connector on", "tone": "ok"},
        "wake_suggestions": [], "rooms": [], "unanswered": 0, "last_activity_at": "", "activity_coverage": "", "spent_24h_usd": None,
        "account_candidates": [], "shared_owner_accounts": [], "circle": {"circle_id": circle, "placed_by": "owner", "placed_at": ""},
        "description": None, **extra,
    }


def _circle(circle_id: str, name: str, model: str, revision: int) -> dict:
    return {"circle_id": circle_id, "name": name, "text": "Help the team by the facts.", "text_author": "owner", "text_at": "",
            "revision": revision, "model": model}


def _catalog() -> dict:
    row = lambda provider, source, model, value, **extra: {  # noqa: E731
        "provider_id": provider, "provider": source, "source": source, "id": model, "name": model, "value": value,
        "label": f"{source} · {model}", **extra}
    return {"items": [row("openrouter", "OpenRouter", "z-ai/glm-5.3", GLM), row("openai", "OpenAI", "gpt-6", GPT),
                      row("local", "Local", "qwen3.gguf", "local::qwen3.gguf", use_local=True)], "errors": [], "model_sources": []}


_FACTS = """() => {
    const text = (node) => (node ? node.innerText.replace(/\\s+/g, ' ').trim() : null);
    const fold = (id) => document.querySelector(`.circle-fold[data-circle="${id}"]`);
    return {
        heads: Object.fromEntries(['team', 'home'].map((id) => [id, text(fold(id)?.querySelector('summary .circle-head-count'))])),
        rows: Object.fromEntries([...document.querySelectorAll('[data-place-open]')].map((row) => [row.dataset.placeOpen, text(row.querySelector('.consciousness-place-facts'))])),
        own: text(document.querySelector('[data-circle-own-models]')),
        wide: document.documentElement.scrollWidth - window.innerWidth,
        clipped: [...document.querySelectorAll('.circle-head-count, .place-row, .circle-editor, dialog[open] .ui-control')]
            .filter((node) => node.getClientRects().length && node.getBoundingClientRect().right > window.innerWidth + 0.5).map((node) => node.className),
    };
}"""


# Whatever the names, nothing of the directory leaves the screen and no head is wider than itself.
_HOLDS = """() => {
    const text = (node) => (node ? node.innerText.replace(/\\s+/g, ' ').trim() : null);
    const box = (node) => { const r = node.getBoundingClientRect(); return { left: r.left, right: r.right, width: r.width }; };
    const head = document.querySelector('.circle-fold[data-circle="team"] > summary');
    return {
        wide: document.documentElement.scrollWidth - window.innerWidth,
        out: [...document.querySelectorAll('.circles-directory *')].filter((node) => node.getClientRects().length
            && (node.getBoundingClientRect().right > window.innerWidth + 0.5 || node.getBoundingClientRect().left < -0.5)).map((node) => node.className || node.tagName),
        head: head.scrollWidth - head.clientWidth,
        count: text(head.querySelector('.circle-head-count')), countBox: box(head.querySelector('.circle-head-count')),
        name: text(head.querySelector('.circle-head-name')), nameBox: box(head.querySelector('.circle-head-name')),
        rows: Object.fromEntries([...document.querySelectorAll('[data-place-open]')].map((row) => {
            const facts = row.querySelector('.consciousness-place-facts');
            return [row.dataset.placeOpen, { facts: text(facts), cut: facts.scrollWidth - facts.clientWidth, over: row.scrollWidth - row.clientWidth }];
        })),
    };
}"""

# The sheet read to its end, where More and its Model field are.
_TO_END = "() => { const body = document.querySelector('dialog[open] [data-place-body]'); body.scrollTop = body.scrollHeight; }"


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_circles_model_and_what_a_place_inherits_on_a_phone(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    shots = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path)))
    shots.mkdir(parents=True, exist_ok=True)
    circles = {"team": _circle("team", "Team", GLM, 3), "home": _circle("home", "Home", "", 4)}
    places = {
        "alpha": _place("alpha", "Chat · alpha", "team", "", {"value": GLM, "layer": "circle"}),
        "beta": _place("beta", "Chat · beta", "team", GPT, {"value": GPT, "layer": "place"}),
        "gamma": _place("gamma", "Chat · gamma", "home", "", {"value": ASTRA, "layer": "consciousness"}),
    }
    writes: list = []

    def listing(route):
        if route.request.method != "GET":
            return route.continue_()
        route.fulfill(json={"ok": True, "places": list(places.values()), "circles": list(circles.values()), "migration_problems": [],
                            "connectors": [{"connector": "chat-bridge", "label": "Chat", "destination": "vendor:chat",
                                            "readiness": {"text": "Connector on", "tone": "ok"}}]})

    def circle_write(route):
        circle_id = route.request.url.rsplit("/", 1)[-1]
        body = json.loads(route.request.post_data or "{}")
        writes.append(("circle", circle_id, body))
        row = circles[circle_id]
        row.update({key: body[key] for key in ("name", "text", "model") if key in body}, revision=row["revision"] + 1)
        # A circle's model reaches the places that inherit it at their next read, their own rows unrevised.
        for place in places.values():
            if place["circle"]["circle_id"] == circle_id and not place["model"]:
                place["model_route"] = {"value": row["model"] or ASTRA, "layer": "circle" if row["model"] else "consciousness"}
        route.fulfill(json={"ok": True, "circle": row})

    def place_write(route):
        place_id = route.request.url.rsplit("/", 1)[-1]
        body = json.loads(route.request.post_data or "{}")
        writes.append(("place", place_id, body))
        row = places[place_id]
        if "model" in body:
            circle = circles[row["circle"]["circle_id"]]
            row["model"] = body["model"]
            row["model_route"] = ({"value": body["model"], "layer": "place"} if body["model"]
                                  else {"value": circle["model"] or ASTRA, "layer": "circle" if circle["model"] else "consciousness"})
        row["revision"] += 1
        route.fulfill(json={"ok": True, "place": row})

    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 390, "height": 844})
        try:
            page.route("**/api/places", listing)
            page.route("**/api/places/*/rooms", lambda route: route.fulfill(json={
                "place_id": route.request.url.split("/")[-2], "rooms": [], "coverage": {"discovery": "arrivals", "importing": False, "gaps": False}}))
            page.route("**/api/circles/*", circle_write)
            page.route("**/api/places/*", lambda route: place_write(route) if route.request.method == "POST" else route.continue_())
            page.route("**/api/model-catalog*", lambda route: route.fulfill(json=_catalog()))
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_selector("#chat-input", timeout=30_000)
            page.evaluate("() => window.dispatchEvent(new CustomEvent('ouro:open-circles', {detail: {circle: 'team'}}))")
            page.wait_for_selector('.circle-fold[data-circle="team"][open] [data-place-open="beta"]', timeout=30_000)
            page.wait_for_timeout(400)

            seen = page.evaluate(_FACTS)
            assert seen["heads"] == {"team": "2 places · model glm-5.3", "home": "1 place"}, seen
            assert seen["rows"]["beta"].endswith("· model gpt-6") and "model" not in seen["rows"]["alpha"], seen
            assert "model" not in seen["rows"]["gamma"], seen
            assert seen["own"] == "1 place here keeps a model of its own and does not follow the circle's.", seen
            assert seen["wide"] <= 0 and not seen["clipped"], seen
            page.screenshot(path=str(shots / f"{engine}-1-circles.png"), animations="disabled")

            # Edit: the circle's model in the place sheet's chooser; its suggestions are routes, the local model is not one.
            page.click('.circle-fold[data-circle="team"] [data-circles-act="edit"]')
            field = page.locator('[data-circles-draft="team:model"]')
            assert field.input_value() == GLM
            assert "1 place here keeps a model of its own" in page.locator("#circle-model-help-team").inner_text()
            field.click()
            field.fill("")
            page.wait_for_selector(".ui-model-options:not([hidden]) .ui-model-option", timeout=10_000)
            offered = page.locator(".ui-model-options:not([hidden]) .ui-model-option").evaluate_all("nodes => nodes.map(n => n.dataset.modelValue)")
            assert offered == [GLM, GPT], offered
            seen = page.evaluate(_FACTS)
            assert seen["wide"] <= 0 and not seen["clipped"], seen
            page.screenshot(path=str(shots / f"{engine}-2-circle-editor.png"), animations="disabled")
            page.locator(f'.ui-model-options:not([hidden]) .ui-model-option[data-model-value="{GPT}"]').click()
            assert field.input_value() == GPT
            page.click('.circle-fold[data-circle="team"] [data-circles-act="save"]')
            page.wait_for_function("""() => document.querySelector('.circle-fold[data-circle="team"] summary .circle-head-count')?.innerText.includes('model gpt-6')""", timeout=10_000)
            assert writes == [("circle", "team", {"model": GPT, "expected_revision": 3})], writes

            # The place sheet of a place that inherits: the empty field says whose model it uses.
            page.click('[data-place-open="alpha"]')
            page.wait_for_selector("dialog[open] .place-more", timeout=10_000)
            page.click("dialog[open] .place-more > summary")
            model = page.locator('dialog[open] [data-place-draft="model"]')
            assert model.input_value() == "" and model.get_attribute("placeholder") == "Uses Team's model: gpt-6"
            assert page.locator("#place-model-help").inner_text() == "The model it thinks with when it wakes here; empty uses Team's model, gpt-6."
            page.evaluate(_TO_END)
            seen = page.evaluate(_FACTS)
            assert seen["wide"] <= 0 and not seen["clipped"], seen
            page.screenshot(path=str(shots / f"{engine}-3-place-inherits.png"), animations="disabled")
            page.click("dialog[open] [data-place-close]")

            # A place with its own model: clearing the field returns it to the circle's.
            page.click('[data-place-open="beta"]')
            page.wait_for_selector("dialog[open] .place-more", timeout=10_000)
            page.click("dialog[open] .place-more > summary")
            model = page.locator('dialog[open] [data-place-draft="model"]')
            assert model.input_value() == GPT
            model.fill("")
            page.click('dialog[open] [data-place-act="save-model"]')
            page.wait_for_function("() => document.querySelector('dialog[open]')?.innerText.includes(\"Now uses Team's model: gpt-6.\")", timeout=10_000)
            assert writes[-1] == ("place", "beta", {"model": "", "expected_revision": 2}), writes
            page.evaluate(_TO_END)
            page.screenshot(path=str(shots / f"{engine}-4-place-cleared.png"), animations="disabled")
            page.click("dialog[open] [data-place-close]")
            page.wait_for_function("""() => !document.querySelector('[data-place-open="beta"] .consciousness-place-facts').innerText.includes('model')""", timeout=10_000)
            assert page.evaluate(_FACTS)["own"] is None, "no place keeps its own model now: the circle says nothing"

            # In a circle without a model the default is the consciousness model, named.
            page.click('.circle-fold[data-circle="home"] > summary')
            page.click('[data-place-open="gamma"]')
            page.wait_for_selector("dialog[open] .place-more", timeout=10_000)
            page.click("dialog[open] .place-more > summary")
            assert page.locator('dialog[open] [data-place-draft="model"]').get_attribute("placeholder") == "Uses the consciousness model: gpt-6-astra"
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_long_circle_and_model_names_hold_on_a_phone(direct_server_with_data, engine, tmp_path):  # noqa: F811
    """Real model names are long and so are circle names: at 390 px the head's count goes under the name and wraps
    there, and a place row that shows its overriding model wraps its facts; nothing is wider than the screen."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    shots = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path)))
    shots.mkdir(parents=True, exist_ok=True)
    kimi, argon, opus = "moonshotai/kimi-k3-thinking-preview", "google/gemini-4-argon-high", "anthropic::claude-opus-5.5"
    circles = [_circle("team", "Customer research and partner support desk", kimi, 3), _circle("home", "Home and family errands", argon, 4)]
    places = [
        _place("alpha", "Chat · the partners' long-named working channel", "team", "", {"value": kimi, "layer": "circle"}),
        _place("beta", "Chat · beta", "team", argon, {"value": argon, "layer": "place"}, state="paused"),
        _place("delta", "Chat · delta", "team", opus, {"value": opus, "layer": "place"}),
        _place("gamma", "Chat · gamma", "home", "", {"value": argon, "layer": "circle"}),
    ]
    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 390, "height": 844})
        try:
            page.route("**/api/places", lambda route: route.fulfill(json={
                "ok": True, "places": places, "circles": circles, "migration_problems": [],
                "connectors": [{"connector": "chat-bridge", "label": "Chat", "destination": "vendor:chat",
                                "readiness": {"text": "Connector on", "tone": "ok"}}]}) if route.request.method == "GET" else route.continue_())
            page.route("**/api/model-catalog*", lambda route: route.fulfill(json=_catalog()))
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_selector("#chat-input", timeout=30_000)
            page.evaluate("() => window.dispatchEvent(new CustomEvent('ouro:open-circles', {detail: {circle: 'team'}}))")
            page.wait_for_selector('.circle-fold[data-circle="team"][open] [data-place-open="delta"]', timeout=30_000)
            page.wait_for_timeout(400)
            seen = page.evaluate(_HOLDS)
            assert seen["wide"] <= 0 and not seen["out"] and seen["head"] <= 0, seen
            assert seen["count"] == "3 places · 1 paused · model kimi-k3-thinking-preview", seen
            assert seen["name"] == "Customer research and partner support desk" and seen["nameBox"]["width"] >= 150, \
                ("the count went under the name: the name keeps its line", seen)
            assert seen["countBox"]["right"] <= 390 and seen["countBox"]["left"] >= 0, seen
            assert seen["rows"]["beta"]["facts"].endswith("· model gemini-4-argon-high"), seen
            assert seen["rows"]["delta"]["facts"].endswith("· model claude-opus-5.5") and "model" not in seen["rows"]["alpha"]["facts"], seen
            assert all(row["cut"] <= 0 and row["over"] <= 0 for row in seen["rows"].values()), ("a row's facts wrap, never cut", seen)
            page.screenshot(path=str(shots / f"{engine}-circles-long-names.png"), animations="disabled")
            # Closed heads too: the other circle's count beside or under its name.
            page.click('.circle-fold[data-circle="team"] > summary')
            page.wait_for_timeout(300)
            closed = page.evaluate(_HOLDS)
            assert closed["wide"] <= 0 and not closed["out"] and closed["head"] <= 0, closed
            page.screenshot(path=str(shots / f"{engine}-circles-long-names-closed.png"), animations="disabled")
        finally:
            browser.close()
