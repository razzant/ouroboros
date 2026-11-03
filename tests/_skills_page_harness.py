"""The real Skills page mounted from web/ with every HTTP answer stubbed in the browser.

No backend runs: the page, its modules and stylesheets are served from the checkout, and each
`/api/` request is answered from the test's own facts (the skill-source fixtures in
web/tests/fixtures/skills_v4 by default). Requests can be held and released to observe the page
between a request and its answer. Nothing is written anywhere.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
FIXTURES = WEB / "tests" / "fixtures" / "skills_v4"
URL = "http://skills.test/"


def fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _document() -> str:
    sheets = "".join(f'<link rel="stylesheet" href="/static/{name}">' for name in ("ui.css", "style.css")
                     if (WEB / name).exists())
    return (
        f'<!doctype html><html><head>{sheets}</head><body><main id="content"></main>'
        '<script type="module">'
        'import {initSkills} from "/static/modules/skills.js";'
        'window.skillsView=initSkills({showPage:(page)=>{window.shownPage=page;},'
        'openSettingsTab:(tab)=>{window.settingsTab=tab;}});'
        'document.querySelector("#page-skills").classList.add("active");'
        'window.dispatchEvent(new CustomEvent("ouro:page-shown",{detail:{page:"skills"}}));'
        '</script></body></html>'
    )


@pytest.fixture
def skills_browser():
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("set OUROBOROS_RUN_UI_SMOKE=1 to run browser UI smoke")
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        yield playwright


class SkillsPage:
    """Routes one Playwright page. `answers[path]` is a JSON body, `(status, body)` or a callable
    `request -> (status, body)`; a path in `hold` waits until `release(path, ...)`."""

    def __init__(self, page, *, extensions=None, sources=None, catalog=None):
        self.page = page
        self.requests: list[tuple[str, str]] = []
        self.posts: list[tuple[str, object]] = []
        self.hold: set[str] = set()
        self.pending: list[tuple[str, object]] = []
        self.answers = {
            "/api/state": {"github_token_configured": True},
            "/api/extensions": {"skills": copy.deepcopy(extensions or []), "live": {}},
            "/api/skills/lifecycle-queue": {"events": [], "active": None},
            "/api/skill-sources": copy.deepcopy(sources) if sources is not None
            else {"sources": [], "materialized": False, "problem": ""},
            "/api/skill-catalog": copy.deepcopy(catalog) if catalog is not None
            else {"items": [], "categories": [], "sources": [], "starter": {}},
        }
        page.route("**/*", self._route)

    # -- the page's facts -------------------------------------------------------------------------
    def set_skills(self, skills):
        self.answers["/api/extensions"] = {"skills": copy.deepcopy(skills), "live": {}}

    def fail(self, path: str, error: str, status: int = 503):
        self.answers[path] = (status, {"error": error})

    def posted(self, path: str):
        return [body for posted_path, body in self.posts if posted_path == path]

    # -- routing ----------------------------------------------------------------------------------
    def _respond(self, route, answer):
        status, body = answer if isinstance(answer, tuple) else (200, answer)
        route.fulfill(status=status, content_type="application/json", body=json.dumps(body))

    def _route(self, route):
        request = route.request
        url = urlsplit(request.url)
        path = url.path
        if path == "/":
            route.fulfill(content_type="text/html", body=_document())
            return
        if path.startswith("/static/"):
            source = WEB / path.removeprefix("/static/")
            if source.is_file() and source.resolve().is_relative_to(WEB.resolve()):
                mime = "text/css" if source.suffix == ".css" else "text/javascript"
                route.fulfill(content_type=mime, body=source.read_bytes())
            else:
                route.fulfill(status=404, body="Missing fixture asset")
            return
        if not path.startswith("/api/"):
            route.fulfill(status=404, body="off the harness")
            return
        self.requests.append((request.method, path + (f"?{url.query}" if url.query else "")))
        if request.method != "GET":
            self.posts.append((path, request.post_data_json if request.post_data else None))
        if path in self.hold:
            self.pending.append((path, route))
            return
        answer = self.answers.get(path)
        if answer is None:
            answer = {"ok": True} if request.method != "GET" else (404, {"error": f"Unexpected fixture request: {path}"})
        if callable(answer):
            answer = answer(request)
        self._respond(route, answer)

    def wait_for_request(self, path: str, timeout_ms: int = 5000):
        """Let the page issue a held request before the test answers it."""
        for _ in range(max(1, timeout_ms // 25)):
            if any(pending == path for pending, _ in self.pending):
                return
            self.page.wait_for_timeout(25)
        raise AssertionError(f"the page never requested {path}; pending: {[p for p, _ in self.pending]}")

    def wait_for_count(self, request: tuple[str, str], count: int, timeout_ms: int = 5000):
        for _ in range(max(1, timeout_ms // 25)):
            if self.requests.count(request) >= count:
                return
            self.page.wait_for_timeout(25)
        raise AssertionError(f"{request} seen {self.requests.count(request)} times, expected {count}")

    def release(self, path: str, body=None, status: int = 200):
        self.wait_for_request(path)
        index = next(i for i, (pending, _) in enumerate(self.pending) if pending == path)
        _, route = self.pending.pop(index)
        if body is None:
            body = self.answers.get(path, {"ok": True})
            if callable(body):
                body = body(route.request)
            if isinstance(body, tuple):
                status, body = body
        route.fulfill(status=status, content_type="application/json", body=json.dumps(body))

    def close(self, browser):
        try:
            for _, route in self.pending:
                route.abort()
            self.pending.clear()
        finally:
            browser.close()


def open_skills(page, **facts) -> SkillsPage:
    return SkillsPage(page, **facts)


def refresh(page):
    page.locator("#skills-refresh").click()
    page.wait_for_function("() => !document.querySelector('#skills-refresh').disabled")


def capture(page, browser_name, width, state):
    if location := os.environ.get("OUROBOROS_UI_EVIDENCE_OUT"):
        target = Path(location)
        target.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(target / f"skills-{browser_name}-{width}-{state}.png"))


def no_overflow(page) -> bool:
    return page.evaluate("""() => {
        const scroll = document.querySelector('.skills-scroll');
        return document.documentElement.scrollWidth <= window.innerWidth
            && (!scroll || scroll.scrollWidth - scroll.clientWidth <= 1);
    }""")
