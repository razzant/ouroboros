"""What the limits and the role lines say about a price counted only under a condition, in the real candidate shell.

DeepSeek's published entries price a token call only wholly off-peak on a USD key, and a search on a USD key
(``pricing.spend_coverage``, ``pricing_catalog.price_condition``). Dashboard › Costs keeps one quiet line under the
limits, `Not counted: …` then `Partly counted: …`; a role line and an agent's tariff say `Estimated for text,
off-peak · USD only`, never a figure, and Vision (its calls carry images) says `Price unknown`. Synthetic data only: the server holds a dated copy of the catalog entries
(tests/fixtures/pricing), a made-up key, and every outbound proxy-aware request lands on a local trap that answers
403, so the made-up key never leaves the host; DeepSeek's own model list is the real gateway function over an
in-memory transport. Desktop and phone, light and dark; frames go to ``OUROBOROS_UI_EVIDENCE_DIR`` when it is set.
The setup wizard's limits step says the same line for its draft: the real bootstrap and preview, in this process.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import os
import socket
import threading
from pathlib import Path

import pytest

from tests._tariff_guard import _no_unscripted_tariff_reads  # noqa: F401  (autouse: no tariff read in this process)
from tests.test_subscription_setup_browser import capture as _capture_onboarding
from tests.test_subscription_setup_browser import onboarding as onboarding  # noqa: F401  (the real preview gateway)
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui  # noqa: F401

pytest_plugins = ("tests.test_ui_smoke_playwright",)
pytestmark = [pytest.mark.ui_browser]

FIXTURE = Path(__file__).parent / "fixtures" / "pricing" / "litellm-deepseek-20261115.json"
COMBOS = [(1440, 900, "light"), (1440, 900, "dark"), (390, 844, "light"), (390, 844, "dark")]
AGENT = {"subagent_id": "deepseek-pro", "recommended_use": "Second opinion on maths, by API key.",
         "route": {"kind": "api_model", "target_id": "deepseek::deepseek-v4-pro"}, "review_eligible": True}
LINE = ("Not counted: all model roles (price unknown). "
        "Partly counted: Web search (USD only); Agents (text only, off-peak, USD only).")
CONDITION = "Estimated for text, off-peak · USD only"
# Sources this fixture cannot read: DeepSeek's goes to the trap, the mock LLM serves no /models, no engine is started.
UNREAD = {"deepseek", "openai-compatible", "claudexor"}


class _Trap:
    """A proxy that refuses every tunnel and records where it was asked to go."""

    def __init__(self):
        self.asked: list[str] = []
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        # Closing a listener ends an accept() blocked on it on macOS, never on Linux, where the thread would wait for
        # one more connection and outlive the test (conftest thread hygiene). The loop looks at a stop flag between
        # short waits instead, and close() joins it.
        self.sock.settimeout(0.1)
        self.port = self.sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with conn:
                conn.settimeout(5)
                try:
                    self.asked.append(conn.recv(4096).split(b"\r\n", 1)[0].decode("latin-1"))
                    conn.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                except OSError:
                    pass

    def close(self):
        self._stop.set()
        self._thread.join(timeout=10)  # a tunnel being answered takes at most its 5 s read
        self.sock.close()
        assert not self._thread.is_alive(), "the trap's thread outlived its close"


def _evidence(name: str) -> Path | None:
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if not root:
        return None
    path = Path(root) / "spend-coverage"
    path.mkdir(parents=True, exist_ok=True)
    return path / name


def _deepseek_items(data_dir: Path) -> list[dict]:
    """The gateway's own DeepSeek rows for a synthetic /models answer, marked from the server's dated record."""
    import httpx

    from ouroboros import pricing_catalog
    from ouroboros.gateway import models

    listed = {"object": "list", "data": [{"id": "deepseek-v4-pro"}, {"id": "deepseek-v4-flash"}, {"id": "deepseek-chat"}]}

    async def read():
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=listed))
        async with httpx.AsyncClient(transport=transport) as client:
            return await models._fetch_openai_compatible_model_catalog(
                client, "deepseek", "DeepSeek", "synthetic", "https://api.deepseek.com/v1")

    kept = os.environ.get("OUROBOROS_DATA_DIR")
    os.environ["OUROBOROS_DATA_DIR"] = str(data_dir)
    pricing_catalog._cache.clear()
    try:  # its own thread: a browser fixture may already run an event loop in this one
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            items = pool.submit(asyncio.run, read()).result()
    finally:
        pricing_catalog._cache.clear()
        if kept is None:
            os.environ.pop("OUROBOROS_DATA_DIR", None)
        else:
            os.environ["OUROBOROS_DATA_DIR"] = kept
    return [{**item, "context_window": 1000000} for item in items]


def _seed(server, trap: _Trap) -> None:
    """While stopped: DeepSeek on Vision, Web search and one agent; the catalog's dated record; the trap as proxy."""
    from ouroboros.utils import utc_now_iso

    data_dir, env = server["data_dir"], server["env"]
    settings = json.loads((data_dir / "settings.json").read_text(encoding="utf-8"))
    settings.update(DEEPSEEK_API_KEY="synthetic-deepseek-key-not-a-secret",
                    OUROBOROS_MODEL_VISION="deepseek::deepseek-v4-flash", OUROBOROS_WEBSEARCH_BACKEND="deepseek",
                    OUROBOROS_SUBAGENTS=json.dumps({"enabled": True, "items": [AGENT]}))
    (data_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    raw = FIXTURE.read_text(encoding="utf-8")
    record = data_dir / "state" / "pricing" / "litellm.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"url": "https://raw.githubusercontent.com/BerriAI/litellm/main/"
                                         "model_prices_and_context_window.json", "fetched_at": utc_now_iso(),
                                  "sha256": hashlib.sha256(raw.encode()).hexdigest(), "content": raw}), encoding="utf-8")
    proxy = f"http://127.0.0.1:{trap.port}"
    env.update(HTTPS_PROXY=proxy, HTTP_PROXY=proxy, ALL_PROXY=proxy, https_proxy=proxy, http_proxy=proxy,
               all_proxy=proxy, NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")


def _page(browser, engine, width, height, theme, catalog_items):
    from tests.ui_console_errors import PageErrors

    context = browser.new_context(viewport={"width": width, "height": height}, color_scheme=theme, locale="en-US",
                                  has_touch=width < 720, device_scale_factor=2)
    page = context.new_page()
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
    page.errors = PageErrors(page, engine)

    def catalog(route):
        response = route.fetch()
        body = response.json()
        if isinstance(body, dict) and isinstance(body.get("items"), list) and "source_id" not in route.request.url:
            if (frame := _evidence(f"{engine}-served-catalog-errors.json")) is not None:
                frame.write_text(json.dumps(body.get("errors"), indent=1), encoding="utf-8")
            body["items"] = [item for item in body["items"] if not str(item.get("provider_id", "")).startswith("deepseek")]
            body["items"] += catalog_items
            body["errors"] = [row for row in body.get("errors") or [] if row.get("provider_id") not in UNREAD]
        route.fulfill(response=response, json=body)

    page.route("**/api/model-catalog*", catalog)
    return page


def _nav(page, target: str) -> None:
    if page.evaluate("document.documentElement.dataset.layout === 'stack'"):
        back = page.locator('.page.active [data-dashboard-list-back]:visible')
        if back.count():
            back.click()
    toggle = page.locator(".page.active [data-mobile-nav-toggle]:visible")
    if toggle.count():
        toggle.first.click()
    page.click(f'[data-nav-page="{target}"]')


@pytest.mark.parametrize("engine", ["webkit", "chromium"])
def test_limits_roles_and_agents_say_a_conditional_price_apart_from_an_unknown_one(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright

    from tests.settings_browser_helpers import open_entry, open_settings

    server, trap = direct_server_with_data, _Trap()
    try:
        server["stop_server"]()
        _seed(server, trap)
        server["start_server"]()
        url, items = server["url"], _deepseek_items(server["data_dir"])
        with sync_playwright() as pw:
            browser = getattr(pw, engine).launch()
            try:
                for width, height, theme in COMBOS:
                    scene = f"{engine}-{width}-{theme}"
                    page = _page(browser, engine, width, height, theme, items)
                    page.goto(url + "/", wait_until="domcontentloaded")
                    page.wait_for_selector("#page-chat", timeout=30_000)
                    _nav(page, "dashboard")
                    page.click('[data-dashboard-tab="costs"]')
                    line = page.locator("#costs-gaps")
                    expect(line).to_have_text(LINE, timeout=30_000)
                    line.scroll_into_view_if_needed()
                    if (frame := _evidence(f"{scene}-costs-limits.png")) is not None:
                        page.locator(".costs-budget").screenshot(path=str(frame))
                        page.screenshot(path=str(frame.with_name(f"{scene}-costs-screen.png")))
                    page.goto(url + "/#settings", wait_until="domcontentloaded")
                    level = open_settings(page, "models/role/vision", timeout=30_000)
                    vision = level.locator("[data-model-role-status]").first
                    expect(vision).to_have_text("Price unknown · 1M window", timeout=30_000)  # it sends images
                    vision.scroll_into_view_if_needed()
                    if (frame := _evidence(f"{scene}-models-vision.png")) is not None:
                        level.screenshot(path=str(frame))
                        page.screenshot(path=str(frame.with_name(f"{scene}-models-screen.png")))
                    open_settings(page, "agents", timeout=30_000).locator("[data-entry-id]").first.wait_for()
                    editor = open_entry(page, AGENT["subagent_id"], timeout=30_000)
                    money = editor.locator("[data-subagent-delivery-cost]")  # a marked reviewer's money line
                    expect(money).to_have_text(CONDITION, timeout=30_000)
                    editor.locator("[data-subagent-details] > summary").click()
                    expect(editor.locator("[data-subagent-tariff] dd")).to_have_text(CONDITION)
                    if (frame := _evidence(f"{scene}-agents-editor.png")) is not None:
                        editor.screenshot(path=str(frame))
                    assert page.errors.unexpected() == [], page.errors.report()
                    page.context.close()
            finally:
                browser.close()
    finally:
        trap.close()
    reached = [line for line in trap.asked if "deepseek" in line.lower()]
    assert all(line.startswith("CONNECT ") for line in trap.asked), trap.asked
    if (log := _evidence(f"{engine}-trapped-egress.json")) is not None:
        log.write_text(json.dumps({"asked": trap.asked, "deepseek": reached}, indent=1), encoding="utf-8")


def _record_catalog(data_dir: Path) -> None:
    from ouroboros.utils import utc_now_iso

    raw = FIXTURE.read_text(encoding="utf-8")
    record = data_dir / "state" / "pricing" / "litellm.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"url": "https://raw.githubusercontent.com/BerriAI/litellm/main/"
                                         "model_prices_and_context_window.json", "fetched_at": utc_now_iso(),
                                  "sha256": hashlib.sha256(raw.encode()).hexdigest(), "content": raw}), encoding="utf-8")


def test_the_setup_limits_step_says_the_same_line_for_its_draft(subscription_ui, onboarding, monkeypatch,  # noqa: F811
                                                                  tmp_path):
    """A fresh install whose environment holds a DeepSeek key: the wizard proposes DeepSeek's own models, the real
    preview counts the text roles only off-peak in USD and Vision, which sends images, not at all, and its limits step
    says so in Costs' one line, desktop and phone."""
    from playwright.sync_api import expect
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import config as cfg
    from ouroboros import pricing_catalog
    from ouroboros.gateway import onboarding_host

    ui, page = subscription_ui, subscription_ui["page"]
    for key in ("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS", "OUROBOROS_CONNECTIONS",
                "OUROBOROS_MODEL_VISION", "OUROBOROS_MODEL_SAFETY", "OUROBOROS_SUBAGENTS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-deepseek-key-not-a-secret")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))  # the dated catalog record the preview reads
    _record_catalog(tmp_path)
    monkeypatch.setattr(pricing_catalog, "_cache", {})
    app = Starlette(routes=[Route("/api/onboarding/bootstrap", onboarding_host.api_onboarding_bootstrap)])
    app.state.drive_root = cfg.DATA_DIR
    with TestClient(app) as client:
        bootstrap = client.get("/api/onboarding/bootstrap").json()["bootstrap"]
    ui["bootstrap"].clear()
    ui["bootstrap"].update(bootstrap)
    ui["fixture"]["status"]["profiles"]["profiles"] = []  # API key only: no subscription is connected
    ui["backend"].update(client=onboarding.client, preview_client=onboarding.client,
                         catalog_response={"items": _deepseek_items(tmp_path), "model_sources": [], "errors": []})
    previews = []
    page.on("response", lambda response: previews.append(response.json())
            if response.url.endswith("/api/onboarding/subagents/preview") else None)
    page.goto(ui["url"] + "/onboarding")
    page.click("#next-btn")
    main = page.locator('[data-model-role-group="main"] [data-model-role-status]').first
    expect(main).to_have_text(f"{CONDITION} · 1M window", timeout=30_000)
    page.evaluate("document.documentElement.dataset.theme = 'light'")
    _capture_onboarding(page, "spend-coverage-onboarding-models-1360-light")
    for _ in range(4):
        if page.locator("#budget-gaps").count():
            break
        page.click("#next-btn")
    gaps = page.locator("#budget-gaps")
    expect(gaps).to_have_text("Not counted: Vision (price unknown). Partly counted: Main, Light, Safety Supervisor, "
                              "Consciousness, Fallback, Agents (text only, off-peak, USD only).", timeout=30_000)
    assert previews and previews[-1]["partly_counted_spend"] == [{"name": name, "condition": "off_peak_usd"} for name in (
        "Main", "Light", "Safety Supervisor", "Consciousness", "Fallback", "Agents")]
    assert previews[-1]["known_spend_gaps"] == ["Vision"]
    for width, height, theme in ((1360, 900, "light"), (390, 844, "light"), (390, 844, "dark")):
        page.set_viewport_size({"width": width, "height": height})
        page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
        gaps.scroll_into_view_if_needed()
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), (width, theme)
        expect(gaps).to_be_visible()
        _capture_onboarding(page, f"spend-coverage-onboarding-limits-{width}-{theme}")
    assert not any(path == "/api/onboarding/complete" for path, _ in ui["posts"]), "a look, never a save"
