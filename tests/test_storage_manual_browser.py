"""Storage: finished copies not verified saved, discarded by hand, in a real browser.

The page, its controller and the shared confirm dialog are the production ones. The Storage endpoints are answered
with bodies shaped as backend seam v2 documents them (`pool.manual`, `manual_count`, `availability`, `view=manual`,
`manual` + `accept_loss`), so this proves what the screen draws and sends, not what a server removes: the removal itself
is the backend's own tests and the combined run over a real data folder.
"""
from __future__ import annotations

import json
import re
from urllib.parse import parse_qs, urlparse

import pytest

from tests.test_emoji_free_browser import _browser, _open, _server

direct_server_with_data = _server
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

VOLUME = {"free_bytes": 170_000_000_000, "total_bytes": 4_000_000_000_000, "observed_at": "2026-10-30T10:00:00Z"}
STORAGE_API = re.compile(r"/api/storage(/|\?|$)")
LOST = "The answer to Discard was lost; nothing is sent again."


def _unsaved(count, total_bytes, paths, *, omitted=0, keeps_commits=False):
    return {"not_proven_saved_count": count, "not_proven_saved_bytes": total_bytes, "omitted_count": omitted,
            "keeps_commits": keeps_commits, "not_proven_saved": [{"path": path, "bytes": size} for path, size in paths]}


def _copy(ident, name, manual, size=2_400_000_000):
    return {"id": ident, "group": "working_copies", "kind": "working_copy", "name": name, "path": f"/data/worktrees/copies.noindex/{ident}",
            "task_id": ident, "bytes": size, "bytes_complete": True, "eligible_after": None, "reason": "capture_unproven",
            "outcome": None, "availability": "manual_discard", "manual": manual}


COPIES = [
    _copy("small", "Finished work, one note left", _unsaved(1, 18, [("only-here.txt", 18)])),
    _copy("many", "A copy holding many unsaved files with a very long name that has to wrap inside its row on a phone without clipping anything",
          _unsaved(61, 1_900_000, [(f"build/cache/chunk-{n:02}.tmp", 40_000 - n) for n in range(50)], omitted=11)),
    _copy("committed", "A copy with commits on its branch", _unsaved(1, 1, [("scratch.tmp", 1)], keeps_commits=True)),
]
POOL = [{"id": f"p{n}", "group": "work_folders", "kind": "work_folder", "name": f"Task {n}", "path": f"/data/state/headless_tasks/p{n}",
         "task_id": f"p{n}", "bytes": 1_000_000_000, "bytes_complete": True, "eligible_after": None, "reason": None, "outcome": None,
         "availability": "ordinary"} for n in range(2)]
KEPT = {"id": "kept", "group": "working_copies", "kind": "working_copy", "name": "A copy you asked to keep", "path": "/data/worktrees/copies.noindex/kept",
        "task_id": "kept", "bytes": 5, "bytes_complete": True, "eligible_after": None, "reason": "owner_keep", "outcome": None, "availability": "none"}


class Stub:
    """The Storage endpoints of one plan: preview, the three lists, one free request and its answer."""

    def __init__(self):
        self.phase, self.plan, self.mode = "idle", 0, "answer"
        self.posts, self.reads = [], []

    def free(self):
        if self.phase == "idle":
            return {"state": "idle", "plan": None, "started_at": None, "finished_at": None, "done": 0, "total": 0, "groups": [], "pool": None, "run": None, "error": None}
        pool = {"count": 2, "bytes": 2_000_000_000, "bytes_complete": True, "kinds": {"work_folder": 2},
                "manual": {"count": 3, "bytes": 7_200_000_000, "bytes_complete": True},
                "excluded": [{"reason": "capture_unproven", "count": 3, "manual_count": 3}, {"reason": "owner_keep", "count": 1}]}
        run = None
        if self.phase == "done":
            run = {"groups": [], "selected": 1, "done": 1, "removed": 1, "removed_bytes": 2_400_000_000, "deleting": 0, "deleting_bytes": 0, "stuck": 0,
                   "failed": 0, "not_started": 0, "kept": [], "free_before": 140e9, "free_after": 142e9, "stopped": False, "error": None, "outcomes_ready": True}
        return {"state": "done" if self.phase == "done" else "ready", "plan": f"p{self.plan}", "started_at": "2026-10-30T10:00:00Z", "finished_at": None,
                "done": 0, "total": 6, "groups": [], "pool": pool, "run": run, "error": None}

    def envelope(self, **extra):
        return {"count": None, "counting": None, "error": None, "volume": VOLUME, "retention_days": 7, "free": self.free(), "removers": {}, **extra}

    def answer(self, route):
        request = route.request
        url = urlparse(request.url)

        def send(body, status=200):
            route.fulfill(status=status, content_type="application/json", body=json.dumps(body))

        if url.path.endswith("/free/items"):
            query = {key: values[0] for key, values in parse_qs(url.query).items()}
            self.reads.append(query["view"])
            items = {"pool": POOL, "manual": COPIES, "excluded": [*COPIES, KEPT], "outcomes": []}[query["view"]]
            return send({"plan": query["plan"], "view": query["view"], "version": query["plan"], "total": len(items), "counts": {}, "next_cursor": None, "items": items})
        if url.path.endswith("/free/preview") and request.method == "POST":
            self.phase, self.plan = "ready", self.plan + 1
            return send(self.envelope(started=True), 202)
        if url.path.endswith("/api/storage/free") and request.method == "POST":
            self.posts.append(json.loads(request.post_data))
            if self.mode == "lose":
                self.mode = "answer"
                return route.abort("connectionreset")
            self.phase = "done"
            return send(self.envelope(started=True), 202)
        if request.method == "GET" and url.path.endswith("/api/storage"):
            return send(self.envelope())
        return route.continue_()


def _no_overflow(page, label):
    """Nothing in the Storage page is wider than its scroll area and no sentence is clipped by its own box."""
    over = page.locator("#page-storage .costs-scroll").evaluate("node => node.scrollWidth - node.clientWidth")
    assert over <= 2, f"{label}: the page overflows by {over}px"
    clipped = page.locator("[data-storage-tray]").evaluate(
        "tray => [...tray.querySelectorAll('.storage-free-why, .storage-free-label, .storage-tray-text, .storage-status, .storage-meta')]"
        ".filter(node => node.scrollWidth > node.clientWidth + 1).map(node => node.textContent.slice(0, 60))")
    assert not clipped, f"{label}: clipped text {clipped}"


def drive(page, url, stub, evidence, errors):
    """The whole owner flow on `page` against a server serving the SPA at `url`: the Storage endpoints are `stub`'s."""
    from playwright.sync_api import expect

    page.route(STORAGE_API, stub.answer)
    _open(page, url)
    page.locator('[data-nav-page="dashboard"]').click()
    page.wait_for_selector("#page-dashboard")
    page.locator('[data-dashboard-tab="storage"]').click()
    tray = page.locator("[data-storage-tray]")
    door = page.locator("[data-storage-door]")
    expect(door).to_be_visible(timeout=10_000)
    door.click()
    expect(tray).to_contain_text("Ready to remove", timeout=10_000)
    tray.locator('summary[data-key="left-open"]').click()
    expect(tray).to_contain_text("Finished copies not verified saved")
    # What the pool leaves out: only the copy the owner kept cannot go; the three are offered by hand, not listed as blocked.
    expect(tray.locator("p.storage-meta", has_text="you asked to keep it")).to_have_text("1: you asked to keep it")
    expect(tray).to_contain_text("3 copies of finished work stay: some files in them are not verified saved. 7.2 GB counted.")
    assert tray.locator("[data-free-manual-open]").count() == 1
    assert tray.locator(".btn-danger").count() == 0 and tray.locator("[data-free-manual-go]").count() == 0
    assert "manual" not in stub.reads, "closed: the copies are not read"
    page.screenshot(path=str(evidence / "review-door.png"))

    tray.locator("[data-free-manual-open]").click()
    expect(tray.locator("[data-free-manual-pick]")).to_have_count(3, timeout=10_000)
    assert stub.reads.count("manual") == 1
    assert tray.locator("[data-free-manual-pick]:checked").count() == 0, "nothing starts chosen"
    expect(tray).to_contain_text("Nothing here starts chosen.")
    expect(tray).to_contain_text("Nothing selected")
    assert tray.locator(".btn-danger").count() == 0 and "Choose all" not in tray.inner_text()
    expect(tray).to_contain_text("61 files not verified saved")
    expect(tray).to_contain_text("1 file not verified saved · 1 byte. Its branch and the commits on it stay.")

    # The same screen at the owner's window and at a phone, in both themes.
    for width, height, scheme in ((1232, 880, "dark"), (1232, 880, "light"), (390, 844, "dark"), (390, 844, "light")):
        page.set_viewport_size({"width": width, "height": height})
        page.emulate_media(color_scheme=scheme)
        page.wait_for_timeout(150)
        _no_overflow(page, f"{width}x{height} {scheme}")
        tray.scroll_into_view_if_needed()
        page.screenshot(path=str(evidence / f"manual-list-{width}-{scheme}.png"))
    page.set_viewport_size({"width": 1232, "height": 880})
    page.emulate_media(color_scheme="dark")

    # One choice, its facts, the sample of its paths said as a sample.
    tray.locator("[data-free-manual-pick]").nth(1).check()
    capsule = tray.locator("[data-free-manual-go]")
    expect(capsule).to_have_text("Discard 1 copy")
    assert "btn-danger" in capsule.get_attribute("class")
    tray.locator('details[data-key="manual-paths-1"] > summary').click()
    expect(tray.locator('details[data-key="manual-paths-1"] .storage-largest-item')).to_have_count(50)
    expect(tray.locator('details[data-key="manual-paths-1"]')).to_contain_text("11 more are not listed.")
    _no_overflow(page, "paths open")
    page.screenshot(path=str(evidence / "manual-chosen-paths-open.png"))

    # Declining sends nothing; the acknowledgement is the dialog's confirm, naming the whole copy's loss.
    capsule.click()
    dialog = page.get_by_role("dialog")
    expect(dialog).to_contain_text("This deletes 1 copy whole. 61 files in them are not verified saved (1.9 MB) and can be lost for good.")
    page.screenshot(path=str(evidence / "manual-dialog.png"))
    dialog.get_by_role("button", name="Cancel").click()
    expect(dialog).to_have_count(0)
    assert stub.posts == []

    # The answer is lost: the request is not repeated, the screen offers only a new list.
    stub.mode = "lose"
    capsule.click()
    page.get_by_role("dialog").get_by_role("button", name="Discard 1 copy").click()
    expect(tray).to_contain_text("The answer to Discard was lost. It may still start, so nothing is sent again from this list.")
    expect(page.locator(".toast")).to_contain_text(LOST)
    assert stub.posts == [{"plan": "p1", "manual": ["many"], "accept_loss": True}]
    assert tray.locator("[data-free-manual-go]").count() == 0 and tray.locator(".btn-danger").count() == 0
    page.wait_for_timeout(1500)
    assert len(stub.posts) == 1, "nothing is sent again"
    page.screenshot(path=str(evidence / "manual-lost.png"))

    # A new list: nothing carries over; one more choice goes through, exactly its id.
    tray.locator("[data-free-preview]").click()
    expect(tray).to_contain_text("Ready to remove", timeout=10_000)
    tray.locator('summary[data-key="left-open"]').click()
    tray.locator("[data-free-manual-open]").click()
    expect(tray.locator("[data-free-manual-pick]")).to_have_count(3, timeout=10_000)
    assert tray.locator("[data-free-manual-pick]:checked").count() == 0
    tray.locator("[data-free-manual-pick]").nth(0).check()
    tray.locator("[data-free-manual-go]").click()
    page.get_by_role("dialog").get_by_role("button", name="Discard 1 copy").click()
    expect(tray).to_contain_text("Removed 1 item.", timeout=10_000)
    assert stub.posts[-1] == {"plan": "p2", "manual": ["small"], "accept_loss": True}
    assert sorted(stub.posts[-1]) == ["accept_loss", "manual", "plan"], "its own form"
    page.screenshot(path=str(evidence / "manual-result.png"))
    assert not errors, errors


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_manual_discard_flow_in_a_real_browser(direct_server_with_data, request, engine):
    server = direct_server_with_data
    with _browser(server, request, engine) as (page, evidence, errors):
        drive(page, server["url"], Stub(), evidence, errors)
