"""Settings › Agents catalogue Access through the actual served candidate (#1609).

The review pool is the catalogue's marked rows (PR1639); a session row's Access is
Full by default, Working files, or an explicit Read only. A byte-faithful candidate
checkout serves its own Python and JS (origin proof) over an isolated data root
seeded with a catalogue: a session row with no saved Access, Working files, Read
only, an API reviewer and an unmarked Read-only row. Every choice goes through the
real agent's page (its form's Save, its Reviewer switch), the real owner transaction,
the pool read, a fresh page, a server restart and the pool builder the review core
runs; a seat runs with its own row's rights, and an API row never carries any.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

import pytest

from tests.settings_browser_helpers import open_entry, open_settings, press_editor_save, settings_page

pytest_plugins = ("tests.test_ui_smoke_playwright",)
pytestmark = [pytest.mark.ui_browser]

REPO = Path(__file__).resolve().parents[1]
SERVED = ("modules/subagents_settings.js", "modules/settings.js")
CHOICES = [["full", "Full system access"], ["workspace_write", "Working files"], ["readonly", "Read only"]]
CAVEAT = ("Full system access (the default) can reach outside the working folder. The selected agent must support it. "
          "Explicit task restrictions still apply.")


def _session(row_id, target, access=None, *, marked=True):
    row = {"subagent_id": row_id, "recommended_use": f"{row_id} reviewer.",
           "route": {"kind": "agent_session", "target_id": target}}
    if access:
        row["access"] = access
    if marked:
        row["review_eligible"] = True
    return row


CATALOG = {"enabled": True, "items": [
    _session("full-default", "claude=claude-fable-5-1"),
    _session("working-files", "codex=gpt-5.6-sol", "workspace_write"),
    _session("read-only", "claude=claude-opus-5-5", "readonly"),
    {"subagent_id": "api-reviewer", "recommended_use": "API reviewer.", "review_eligible": True,
     "route": {"kind": "api_model", "target_id": "openai-compatible::mock-model"}},
    _session("read-only-helper", "codex=gpt-5.6-luna", "readonly", marked=False),
]}
IDS = [row["subagent_id"] for row in CATALOG["items"]]
SESSION_ROWS = [index for index, row in enumerate(CATALOG["items"]) if row["route"]["kind"] == "agent_session"]
API_ROW = IDS.index("api-reviewer")


def _evidence(name, payload=None, target=None):
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if not root:
        return
    Path(root).mkdir(parents=True, exist_ok=True)
    if target is not None:
        target.screenshot(path=str(Path(root, f"{name}.png")), animations="disabled")
    if payload is not None:
        Path(root, f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _get(url, route):
    with urllib.request.urlopen(url + route, timeout=15) as response:  # noqa: S310 - loopback test server
        return response.read()


def _seed(server):
    server["stop_server"]()
    path = server["data_dir"] / "settings.json"
    settings = json.loads(path.read_text(encoding="utf-8"))
    settings["OUROBOROS_SUBAGENTS"] = json.dumps(CATALOG)
    path.write_text(json.dumps(settings), encoding="utf-8")
    server["start_server"]()


def _stored(server):
    """The persisted document and, per row, (Reviewer mark, saved Access)."""
    settings = json.loads((server["data_dir"] / "settings.json").read_text(encoding="utf-8"))
    raw = settings["OUROBOROS_SUBAGENTS"]
    catalog = json.loads(raw) if isinstance(raw, str) else raw
    rows = {row["subagent_id"]: [row.get("review_eligible") is True, row.get("access")] for row in catalog["items"]}
    return settings, rows


def _core(settings):
    """The seats the review core builds from the persisted bytes, and the run shape each
    gets: ``review_execution`` hands ``slot.session_access`` to ``delegated_run_shape``."""
    from ouroboros.reviewer_slot_config import review_pool_slots
    from ouroboros.subagents import delegated_run_shape

    raw = settings["OUROBOROS_SUBAGENTS"]
    snapshot = {**settings, "OUROBOROS_SUBAGENTS": raw if isinstance(raw, str) else json.dumps(raw)}
    seats = {}
    for slot in review_pool_slots(snapshot):
        route = str(getattr(slot.route, "value", slot.route))
        seat = {"route": route, "session_access": slot.session_access}
        if route == "agent_session":
            shape = delegated_run_shape(False, slot.session_access or "readonly")
            seat["shape"] = {"access": shape.access, "mode": shape.mode, "isolation": shape.isolation,
                             "investigation": shape.investigation, "delegated": shape.delegated}
        seats[slot.slot_id] = seat
    return seats


def _expect_core(seats, expected):
    """``expected``: pool id -> Access ('' for the API row), in catalogue order."""
    assert list(seats) == list(expected), seats
    for row_id, access in expected.items():
        seat = seats[row_id]
        assert seat["session_access"] == access, (row_id, seat)
        if not access:
            assert seat["route"] == "api_chat" and "shape" not in seat, seat
        elif access == "readonly":
            assert seat["shape"] == {"access": "readonly", "mode": "ask", "isolation": "",
                                     "investigation": False, "delegated": False}, seat
        else:  # a writable reviewer works in a private checking copy whose results are evidence only
            assert seat["shape"] == {"access": access, "mode": "agent", "isolation": "live",
                                     "investigation": True, "delegated": True}, seat


def _expect_pool(url, expected):
    pool = json.loads(_get(url, "/api/review-pool"))
    assert not pool["config_error"], pool["config_error"]
    rows = {row["subagent_id"]: row for row in pool["pool"]}
    assert {row_id: row["access"] for row_id, row in rows.items()} == expected, pool["pool"]
    for row_id, access in expected.items():
        facet = re.search(r"/(readonly|workspace_write)\b", rows[row_id]["handle"])
        assert (facet.group(1) if facet else "") == ("" if access in ("", "full") else access), rows[row_id]
    return pool


def _open(page, url):
    from playwright.sync_api import expect

    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("#page-chat", timeout=30_000)
    page.click('[data-nav-page="settings"]')
    level = open_settings(page, "agents", timeout=30_000)
    expect(level.locator("[data-entry-id]")).to_have_count(len(IDS), timeout=30_000)
    return level


def _agent(page, index):
    """One agent's page, opened from the list, once the pool read reached it (it is enrichment)."""
    from playwright.sync_api import expect

    open_settings(page, "agents", timeout=30_000)
    form = open_entry(page, IDS[index], timeout=30_000).locator("[data-subagent-row]")
    expect(form.locator("[data-subagent-review-facts]")).not_to_contain_text("price appears after saving", timeout=30_000)
    return form


def _shown(page):
    """Per catalogue row, read on its own page: [Reviewer mark, Access value or None when it has no select]."""
    shown = []
    for index in range(len(IDS)):
        form = _agent(page, index)
        access = form.locator('[data-subagent-field="access"]')
        shown.append([form.locator('[data-subagent-field="review_eligible"]').is_checked(),
                      access.input_value() if access.count() else None])
    return shown


def _set_access(page, index, value):
    """Choose an Access on the agent's page and press its Save."""
    _agent(page, index).locator('[data-subagent-field="access"]').select_option(value)
    assert press_editor_save(page, timeout=30_000).startswith("Saved")


def _set_mark(page, index, on):
    """The Reviewer switch of the agent's page writes at once."""
    form = _agent(page, index)
    if form.locator('[data-subagent-field="review_eligible"]').is_checked() is not on:
        form.locator('.agent-use:has([data-subagent-field="review_eligible"]) .ui-switch').click()
    page.wait_for_function("""([path, on]) => {
        const level = [...document.querySelectorAll('#page-settings [data-settings-page]')].find((el) => el.dataset.settingsPage === path);
        const input = level?.querySelector('[data-subagent-field="review_eligible"]');
        return input && input.checked === on && input.getAttribute('aria-busy') !== 'true'; }""",
                           arg=[f"agents/{IDS[index]}", on], timeout=30_000)


def _posted(posts):
    """The last roster the owner transaction carried, per row: [Reviewer mark, Access]."""
    written = [body.get("settings", body) for body in posts]
    catalog = next(body["OUROBOROS_SUBAGENTS"] for body in reversed(written) if "OUROBOROS_SUBAGENTS" in body)
    catalog = json.loads(catalog) if isinstance(catalog, str) else catalog
    return {row["subagent_id"]: [row.get("review_eligible") is True, row.get("access")] for row in catalog["items"]}


def _center(locator):
    locator.evaluate("el => el.scrollIntoView({block: 'center'})")


def _assert_caveat(form, index, width):
    """The agent's Details disclosure exposes readable Access help, linked to the control."""
    details = form.locator('[data-subagent-details]')
    if details.get_attribute('open') is None:
        details.locator('summary').click()
    select, caveat = form.locator('[data-subagent-field="access"]'), form.locator('[id$="-access-help"]')
    _center(caveat)
    assert select.is_visible() and caveat.is_visible()
    covered = caveat.evaluate("""el => { const box = el.getBoundingClientRect();
        const hit = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
        return el === hit || el.contains(hit) ? '' : [JSON.stringify(box), hit && hit.outerHTML.slice(0, 300)]; }""")
    assert not covered, f"the caveat is covered: {covered}"
    assert caveat.inner_text().strip() == CAVEAT
    assert caveat.get_attribute("id") in (select.get_attribute("aria-describedby") or "").split()
    assert select.get_attribute("aria-label") == f"Access for Subagent {index + 1}"
    assert select.locator("option").evaluate_all("els => els.map(e => [e.value, e.textContent.trim()])") == CHOICES
    for box in (caveat.bounding_box(), select.bounding_box()):
        assert box and box["x"] >= 0 and box["x"] + box["width"] <= width + 0.5, box


def _assert_labels(page, width, engine="", shots=""):
    """Every agent's page: a session's Access with its help, an API row without any, the Reviewer row in words."""
    for index, row_id in enumerate(IDS):
        form = _agent(page, index)
        facts = form.locator("[data-subagent-review-facts]")
        if index in SESSION_ROWS:
            _assert_caveat(form, index, width)
            assert "uses a session seat and time" in facts.text_content(), facts.text_content()
        else:
            assert form.locator('[data-subagent-field="access"], [id$="-access-help"]').count() == 0
            assert form.locator('[data-subagent-field="delivery"]').input_value() == "native"
        label = form.locator('.agent-use:has([data-subagent-field="review_eligible"]) .ui-row-title')
        assert label.inner_text().strip() == "Reviewer" and label.is_visible()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        if shots:
            _evidence(f"{engine}-{shots}-row-{index + 1}-{row_id}", target=page)


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_catalogue_access_saves_reloads_restarts_and_reaches_the_pool(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright

    server = direct_server_with_data
    _seed(server)
    url, settings_path = server["url"], server["data_dir"] / "settings.json"
    served = {}
    for name in SERVED:
        body = _get(url, f"/static/{name}")
        assert body == (REPO / "web" / name).read_bytes() == (Path(server["repo_dir"]) / "web" / name).read_bytes()
        served[name] = hashlib.sha256(body).hexdigest()
    facts = {"engine": engine, "candidate_identity": server["candidate_identity"],
             "runtime_version": json.loads(_get(url, "/api/health"))["runtime_version"],
             "served_sha256": served, "steps": {}}

    def record(step, shown):
        settings, stored = _stored(server)
        facts["steps"][step] = {"shown": shown, "stored": stored, "core": _core(settings),
                                "pool": [{key: row[key] for key in ("subagent_id", "handle", "access", "delivery")}
                                         for row in json.loads(_get(url, "/api/review-pool"))["pool"]]}
        return settings, stored

    def reviewers(page):
        return settings_page(page, "agents").locator('[data-agents-filter="reviewers"]')

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            context = browser.new_context(viewport={"width": 1360, "height": 900})
            errors, posts, engine_calls = [], [], []
            context.on("weberror", lambda error: errors.append(str(error.error)))
            context.on("request", lambda r: posts.append(r.post_data_json)
                       if r.method == "POST" and r.url.endswith(("/api/settings", "/api/owner/safety-model")) else None)
            # Configuration only: nothing here may wake, log in or refresh an agent engine.
            context.route(re.compile(r".*/api/claudexor/(wake|login|quota|credential-profiles).*"),
                          lambda route: (engine_calls.append(route.request.url), route.abort()))
            # This scene proves the real roster transaction and Access, not provider discovery. The isolated
            # server has no provider catalog; serve a successful empty read so Circles' unrelated catalog listener
            # does not reject it. Settings, the owner write, and the review-pool read remain the real endpoints.
            context.route("**/api/model-catalog*", lambda route: route.fulfill(json={"items": [], "model_sources": [], "errors": []}))
            page = context.new_page()
            _open(page, url)

            # As seeded: Full is the unsaved default, both lower choices kept, the API row has none.
            seeded = [[True, "full"], [True, "workspace_write"], [True, "readonly"], [True, None], [False, "readonly"]]
            expect(reviewers(page)).to_have_text("Reviewers 4")
            _evidence(f"{engine}-01-desktop-list", target=page)
            assert _shown(page) == seeded
            _assert_labels(page, 1360, engine, "01-desktop")
            settings, stored = record("seeded", seeded)
            assert stored["full-default"] == [True, None], "the seeded row carries no Access of its own"
            _expect_pool(url, {"full-default": "full", "working-files": "workspace_write", "read-only": "readonly",
                               "api-reviewer": ""})
            _expect_core(facts["steps"]["seeded"]["core"], {"full-default": "full", "working-files": "workspace_write",
                                                             "read-only": "readonly", "api-reviewer": ""})

            # An unsaved choice stays on its page, says so, and never reaches the bytes; Discard drops it.
            untouched = settings_path.read_bytes()
            form = _agent(page, 2)
            form.locator('[data-subagent-field="access"]').select_option("full")
            discard = form.locator("[data-editor-discard]")
            expect(discard).to_be_visible()
            assert _agent(page, 2).locator('[data-subagent-field="access"]').input_value() == "full", "kept while unsaved"
            _agent(page, 2).locator("[data-editor-discard]").click()
            expect(settings_page(page, f"agents/{IDS[2]}").locator('[data-subagent-field="access"]')).to_have_value("readonly")
            assert settings_path.read_bytes() == untouched and not posts

            # Lower the Full row to Read only, raise the Read-only row to Full, take the
            # Working-files row out of the pool and put the Read-only helper in it.
            _set_access(page, 0, "readonly")
            _set_access(page, 2, "full")
            _set_mark(page, 1, False)
            _set_mark(page, 4, True)
            expected = {"full-default": [True, "readonly"], "working-files": [False, "workspace_write"],
                        "read-only": [True, "full"], "api-reviewer": [True, None], "read-only-helper": [True, "readonly"]}
            assert _posted(posts) == expected
            assert all(body.get("settings") for body in posts), "every roster write is the owner transaction"
            edited = [[True, "readonly"], [False, "workspace_write"], [True, "full"], [True, None], [True, "readonly"]]
            assert _shown(page) == edited, "a mark never touches the row's Access"
            open_settings(page, "agents", timeout=30_000)
            expect(reviewers(page)).to_have_text("Reviewers 4")
            settings, stored = record("edited", edited)
            assert stored == expected
            pool = {"full-default": "readonly", "read-only": "full", "api-reviewer": "", "read-only-helper": "readonly"}
            _expect_pool(url, pool)
            _expect_core(facts["steps"]["edited"]["core"], pool)
            fresh = context.new_page()
            _open(fresh, url)
            assert _shown(fresh) == edited
            _assert_labels(fresh, 1360)
            _evidence(f"{engine}-02-saved-row-1-full-to-readonly", target=_agent(fresh, 0))
            _evidence(f"{engine}-02-saved-row-3-readonly-to-full", target=_agent(fresh, 2))
            fresh.close()

            # Working files chosen on the page; the unmarked row returns to the pool unchanged.
            _set_access(page, 0, "workspace_write")
            _set_mark(page, 1, True)
            final = [[True, "workspace_write"], [True, "workspace_write"], [True, "full"], [True, None], [True, "readonly"]]
            assert _shown(page) == final
            open_settings(page, "agents", timeout=30_000)
            expect(reviewers(page)).to_have_text("Reviewers 5")
            _, before_restart = _stored(server)

            # Every choice survives a server restart and a new page, byte for byte.
            server["restart_server"]()
            page.close()
            page = context.new_page()
            _open(page, url)
            assert _shown(page) == final
            settings, stored = record("after_restart", final)
            assert stored == before_restart == {
                "full-default": [True, "workspace_write"], "working-files": [True, "workspace_write"],
                "read-only": [True, "full"], "api-reviewer": [True, None], "read-only-helper": [True, "readonly"]}
            pool = {"full-default": "workspace_write", "working-files": "workspace_write", "read-only": "full",
                    "api-reviewer": "", "read-only-helper": "readonly"}
            _expect_pool(url, pool)
            _expect_core(facts["steps"]["after_restart"]["core"], pool)
            open_settings(page, "agents", timeout=30_000)
            facts["review_pool_note"] = settings_page(page, "agents").locator("[data-review-pool-note]").text_content()

            # The caveat and labels stay readable text at a phone width, with no sideways scroll.
            page.set_viewport_size({"width": 390, "height": 844})
            # The off-canvas sidebar slides away first; only then is the layout the phone sees.
            page.wait_for_function("() => document.getAnimations().every(a => a.playState !== 'running')")
            _assert_labels(page, 390, engine, "03-narrow")
            assert not engine_calls and not errors, (engine_calls, errors)
        finally:
            browser.close()
    _evidence(f"{engine}-flow", facts)
