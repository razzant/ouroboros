"""A real browser consumer for Dashboard -> Storage over the candidate server.

No route stub: the isolated server owns the synthetic data folder, so the assertions cover the
HTTP contract, the count child, the SPA's lifecycle wiring and the rendered gauge together.
"""

from __future__ import annotations

import json

import pytest

from tests.test_emoji_free_browser import _browser, _open, _server
from tests.ui_chat_viewport_smoke import wait_drawer_closed

direct_server_with_data = _server
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

AVOIDED = ("At least", "Excluded", "System data", "Junk", "Clean ", "Prune", "%")


def _file_snapshots(paths):
    return {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in paths}


def _assert_bar_fills_its_card(page):
    """The five group shares sum to the headline: the segments fill the bar, none is clipped."""
    bar = page.locator(".storage-bar").bounding_box()
    segments = page.locator(".storage-bar-seg")
    assert bar and segments.count() >= 1, bar
    boxes = [segments.nth(index).bounding_box() for index in range(segments.count())]
    assert all(box and bar["x"] - 0.5 <= box["x"] and box["x"] + box["width"] <= bar["x"] + bar["width"] + 0.5 for box in boxes), boxes
    assert sum(box["width"] for box in boxes) >= bar["width"] * 0.9, boxes


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_storage_dashboard_real_server_consumer(direct_server_with_data, request, engine):
    """GET is passive; an explicit click counts in a child and renders the grouped gauge."""
    from playwright.sync_api import expect

    server = direct_server_with_data
    data_dir = server["data_dir"]
    # Distinct known markers; the server writes runtime files of its own, so a place only has to
    # include its marker, while the response arithmetic must still reconcile exactly.
    markers = {
        "model_calls": (data_dir / "observability" / "storage-consumer-call.json", 127),
        "work_folders": (data_dir / "state" / "headless_tasks" / "storage-consumer-task.bin", 223),
        "task_results": (data_dir / "task_results" / "storage-consumer-result.bin", 347),
        "archives": (data_dir / "archive" / "storage-consumer-archive.bin", 431),
        "memory_skills": (data_dir / "memory" / "storage-consumer-memory.bin", 509),
        "app_state": (data_dir / "state" / "storage-consumer-state.bin", 613),
        "data_other": (data_dir / "storage-consumer-other.bin", 751),
    }
    for path, size in markers.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([size % 251]) * size)
    posts: list[str] = []

    def storage_get(response):
        return response.url.split("?")[0].endswith("/api/storage") and response.request.method == "GET"

    with _browser(server, request, engine) as (page, evidence, errors):
        page.on("request", lambda req: posts.append(req.url.rsplit("/", 1)[-1])
                if "/api/storage/" in req.url and req.method == "POST" else None)
        _open(page, server["url"])
        page.locator('[data-nav-page="dashboard"]').click()
        page.wait_for_selector("#page-dashboard")
        with page.expect_response(storage_get, timeout=10_000) as first_read:
            page.locator('[data-dashboard-tab="storage"]').click()
        expect(page.get_by_role("region", name="Where it is", exact=True)).to_contain_text("Not counted yet", timeout=10_000)
        expect(page.get_by_role("region", name="Storage summary", exact=True)).to_contain_text("Free on this disk")
        expect(page.get_by_role("region", name="Storage summary", exact=True)).to_contain_text("Ouroboros not counted yet")
        assert posts == [], "opening the tab must not start a count"
        first = first_read.value.json()
        assert first["count"] is None and first["counting"] is None and first["error"] is None
        assert isinstance(first["retention_days"], int)
        assert page.locator(".storage-bar").count() == 0, "no bar before a finished count"

        measure = page.locator('[data-storage-action="measure"]')
        expect(measure).to_have_text("Measure now")
        with page.expect_response(lambda response: response.url.endswith("/api/storage/measure"),
                                  timeout=10_000) as posted:
            measure.click()
        started = posted.value.json()
        assert posted.value.status == 202 and started["started"] is True
        # The child counts a folder this small in moments; the finished count or the running one are
        # both legitimate answers to the POST, and the gauge settles on the finished one.
        assert started["counting"] or started["count"], started
        page.wait_for_selector('[data-storage-group="records"]', timeout=60_000)
        expect(measure).to_have_text("Measure again", timeout=60_000)
        expect(measure).to_have_attribute("aria-disabled", "false")
        expect(page.locator('[data-storage-action="stop"]')).to_be_hidden()

        report = page.evaluate("async () => (await fetch('/api/storage', {cache: 'no-store'})).json()")
        count = report["count"]
        places = {row["id"]: row for row in count["places"]}
        for place_id, (_path, size) in markers.items():
            assert places[place_id]["logical_bytes"] >= size, (place_id, places[place_id])
        assert count["bytes"] == sum(row["bytes"] for row in count["places"] if row["bytes"] is not None)
        assert count["io_priority"] and report["counting"] is None and report["error"] is None
        assert (data_dir / "state" / "storage_measurement.json").is_file()
        assert posts == ["measure"]

        # Every group with a present place, a bar of their shares, the places with their words.
        for group in ("records", "work", "temporary", "other"):
            expect(page.locator(f'[data-storage-group="{group}"]')).to_be_visible()
        expect(page.get_by_role("region", name="Storage summary", exact=True)).to_contain_text("Ouroboros uses")
        expect(page.get_by_role("region", name="Where it is", exact=True)).to_contain_text("Last count:")
        expect(page.locator('[data-storage-place="model_calls"]')).to_contain_text("Model call records")
        # Each group folds under its head; opening one shows its places with their words.
        expect(page.locator('[data-storage-place="data_other"]')).to_be_hidden()
        for group in ("records", "other"):
            page.locator(f'[data-storage-group="{group}"] > .storage-group-fold > summary').click()
        expect(page.locator('[data-storage-place="data_other"]')).to_contain_text("Everything else in the data folder")
        expect(page.locator('[data-storage-place="program"]')).to_contain_text("Ouroboros program")
        rendered = page.locator("#page-storage").inner_text()
        assert not any(word in rendered for word in AVOIDED), rendered
        assert page.locator(".storage-bar-seg").count() <= 5
        assert page.locator("#page-storage [title]").count() == 0, "no hover-only meaning"
        _assert_bar_fills_its_card(page)
        page.locator('[data-storage-place="model_calls"] .storage-parts > summary').click()
        expect(page.locator('[data-storage-place="model_calls"] .storage-parts-list')).to_be_visible()

        # Free temporary space: the door previews without side effects over the real server; with no
        # finished work in this install the list is empty and the door stays.
        # The server and its worker write housekeeping even while idle (including memory bootstrap).
        # Compare the test-owned data and saved count here. test_preview_is_pure separately checks the
        # entire isolated tree and spies on every settlement/write seam without concurrent writers.
        protected = [path for path, _size in markers.values()] + [data_dir / "state" / "storage_measurement.json"]
        before = _file_snapshots(protected)
        tray = page.locator("[data-storage-tray]")
        assert page.locator("[data-storage-tray] .btn-danger").count() == 0
        with page.expect_response(lambda response: response.url.endswith("/api/storage/free/preview"), timeout=10_000) as previewed:
            page.locator("[data-storage-door]").click()
        assert previewed.value.status == 202 and previewed.value.json()["started"] is True
        expect(tray).to_contain_text("Nothing temporary is waiting", timeout=30_000)
        expect(page.locator("[data-storage-door]")).to_be_visible()
        after = _file_snapshots(protected)
        assert after == before, "the preview changed no task data or saved count"
        supervisor_rows = [json.loads(line) for line in (data_dir / "logs" / "supervisor.jsonl").read_text().splitlines()]
        assert not any(row.get("type") == "storage_free" for row in supervisor_rows), "preview must not record a free run"
        assert posts == ["measure", "preview"]

        # A transient read failure keeps the last count beside the failure.
        failed = {"remaining": 1}

        def fail_one_read(route):
            if route.request.method == "GET" and failed["remaining"]:
                failed["remaining"] -= 1
                route.fulfill(status=503, content_type="application/json",
                              body=json.dumps({"error": "synthetic storage read refusal"}))
                return
            route.continue_()

        page.route("**/api/storage*", fail_one_read)
        page.locator('[data-dashboard-tab="logs"]').click()
        page.locator('[data-dashboard-tab="storage"]').click()
        expect(page.locator(".storage-status")).to_contain_text("could not be read", timeout=10_000)
        expect(page.get_by_role("region", name="Where it is", exact=True)).to_contain_text("Last count:")
        expect(page.locator('[data-storage-group="records"]')).to_be_visible()
        page.unroute("**/api/storage*", fail_one_read)
        page.screenshot(path=str(evidence / "storage-desktop.png"), full_page=True)

        page.set_viewport_size({"width": 390, "height": 844})
        if page.locator("body").evaluate("node => node.classList.contains('nav-drawer-open')"):
            if page.locator("#nav-drawer-backdrop").is_visible():
                page.locator("#nav-drawer-backdrop").click()  # the room's quarter, all the scrim covers
            page.wait_for_timeout(300)
        wait_drawer_closed(page)
        page.screenshot(path=str(evidence / "storage-narrow.png"), full_page=True)
        body = page.locator("#page-storage .costs-scroll")
        overflow = body.evaluate("node => node.scrollWidth - node.clientWidth")
        assert overflow <= 2, f"storage dashboard overflows the narrow viewport by {overflow}px"
        _assert_bar_fills_its_card(page)

        # The narrowest phone with working copies whose default folder could not be chosen: the server's
        # reason names unbroken absolute paths. Negative control on the staged CSS (review R3): at 360 px
        # the line overflowed its row and widened the Storage panel in both engines.
        lab = "/Users/ouroboros_owner/Library/Application_Support/Ouroboros_lab_installation_2026"
        reason = ("the default subagent_worktrees folder is undecided: this data root's work-root basis is unknown "
                  "(state/state.initialized.json is unreadable) and the candidates differ: "
                  "/Users/ouroboros_owner/Ouroboros/subagent_worktrees (earlier releases) and "
                  f"{lab}/subagent_worktrees (OUROBOROS_APP_ROOT). Set OUROBOROS_SUBAGENT_WORKTREE_ROOT to the "
                  "folder to use; nothing was created or moved.")
        copies = places["working_copies"]
        undecided = {**report, "count": {
            **count, "bytes": count["bytes"] - (copies["bytes"] or 0), "unreadable": count["unreadable"] + 1,
            "places": [{**row, "present": True, "bytes": None, "unreadable": row["unreadable"] + 1, "undecided": reason}
                       if row["id"] == "working_copies" else row for row in count["places"]]}}
        page.route("**/api/storage*", lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(undecided))
            if route.request.method == "GET" else route.continue_())
        page.set_viewport_size({"width": 360, "height": 780})
        page.locator('[data-dashboard-tab="logs"]').click()
        page.locator('[data-dashboard-tab="storage"]').click()
        page.locator('[data-storage-group="temporary"] > .storage-group-fold > summary').click()
        why = page.locator('[data-storage-place="working_copies"] .storage-place-why')
        expect(why).to_contain_text(f"Not counted: {reason[:40]}", timeout=10_000)
        why.scroll_into_view_if_needed()
        page.screenshot(path=str(evidence / "storage-undecided-360.png"))
        widths = why.evaluate("node => [node.scrollWidth, node.clientWidth]")
        assert widths[0] <= widths[1] + 1, f"the undecided reason overflows its row: {widths}"
        overflow = body.evaluate("node => node.scrollWidth - node.clientWidth")
        assert overflow <= 2, f"the undecided reason widens the storage dashboard by {overflow}px"
        assert not errors, errors
