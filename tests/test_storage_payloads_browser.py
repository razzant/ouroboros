"""Storage: the mind's proposals to release heavy files of finished work, chosen by the owner in a real browser.

The page, its controller and the shared confirm dialog are the production ones; the Storage endpoints answer with
bodies shaped as ``payload_release`` and ``storage_free_view`` send them (``pool.manual.payloads``, ``payload_offer``,
``view=manual`` rows with ``payload_release``, ``manual`` + ``accept_loss``). This proves what the screen draws and
sends at the owner's window and a phone, in both themes. A separate case below drives the same UI against the real release API and result store.
"""
from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import pytest

from tests.test_emoji_free_browser import _browser, _open, _server
from tests.test_storage_manual_browser import COPIES, POOL, STORAGE_API, VOLUME, _no_overflow

direct_server_with_data = _server
pytestmark = [pytest.mark.ui_browser]

LONG = ("Compare sample catalog snapshots from two benchmark runs using a detailed experiment title long enough to "
        "wrap on a phone")


def _proposal(ident, name, files):
    return {"id": f"payload:{ident}+payload-p1", "group": "stored_payloads", "kind": "payload", "name": name,
            "path": f"/data/task_results/artifacts/{ident}", "task_id": ident, "bytes": sum(size for _p, size, _r in files),
            "bytes_complete": True, "eligible_after": None, "reason": "payload_offer", "outcome": None,
            "availability": "manual_discard", "manual": {
                "not_proven_saved_count": 0, "not_proven_saved_bytes": 0, "not_proven_saved": [], "omitted_count": 0,
                "keeps_commits": False, "payload_release": True, "proposed_at": "2024-02-03T12:00:00Z", "root_task_id": ident,
                "consequence": "The original sample dataset cannot be measured again; a fresh database is a different case.",
                "files": [{"path": path, "bytes": size, "role": role, "loss": "the measurement cannot be rerun on these bytes"}
                          for path, size, role in files],
                "retained": [{"role": "report", "path": f"delegated_runs/r/evidence-{ident}/files/REPORT.md", "bytes": 9_000},
                             {"role": "method", "path": f"delegated_runs/r/evidence-{ident}/files/compare_catalogs.py", "bytes": 4_000},
                             {"role": "result", "path": "task_result", "bytes": None}]}}


PROPOSALS = [
    _proposal("fixture-task-a", LONG, [("delegated_runs/fixture-run-a/evidence-fixture-generation-a/files/catalog/before.sqlite", 12_300_000_000,
                                  "frozen copy of before.sqlite taken for the measurement"),
                                 ("delegated_runs/fixture-run-a/evidence-fixture-generation-a/files/catalog/after.sqlite", 14_100_000_000,
                                  "frozen copy of after.sqlite taken for the measurement")]),
    _proposal("fixture-task-b", "VM smoke of the macOS lab image", [("source_handles/test-evidence/run-7/disk.img", 9_900_000_000,
                                                                 "the lab VM's disk after the smoke run")]),
]


class Stub:
    """One plan whose by-hand form holds three copies and two proposals; one free request and its answer."""

    def __init__(self):
        self.phase, self.plan, self.posts, self.reads = "idle", 0, [], []

    def free(self):
        if self.phase == "idle":
            return {"state": "idle", "plan": None, "started_at": None, "finished_at": None, "done": 0, "total": 0, "groups": [],
                    "pool": None, "run": None, "error": None}
        pool = {"count": 2, "bytes": 2_000_000_000, "bytes_complete": True, "kinds": {"work_folder": 2},
                "manual": {"count": 5, "bytes": 7_200_000_000 + sum(p["bytes"] for p in PROPOSALS), "bytes_complete": True, "retained": 0, "payloads": 2},
                "excluded": [{"reason": "capture_unproven", "count": 3, "manual_count": 3}, {"reason": "payload_offer", "count": 2, "manual_count": 2}]}
        run = None
        if self.phase == "done":
            run = {"groups": [], "selected": 1, "done": 1, "removed": 0, "removed_bytes": 0, "deleting": 1, "deleting_bytes": PROPOSALS[0]["bytes"],
                   "stuck": 0, "failed": 0, "not_started": 0, "kept": [], "free_before": 140e9, "free_after": 140e9, "stopped": False,
                   "error": None, "outcomes_ready": True}
        return {"state": "done" if self.phase == "done" else "ready", "plan": f"p{self.plan}", "started_at": "2024-02-03T12:00:00Z",
                "finished_at": None, "done": 0, "total": 7, "groups": [], "pool": pool, "run": run, "error": None}

    def envelope(self, **extra):
        return {"count": None, "counting": None, "error": None, "volume": VOLUME, "retention_days": 7, "free": self.free(), "removers": {}, **extra}

    def answer(self, route):
        request, url = route.request, urlparse(route.request.url)

        def send(body, status=200):
            route.fulfill(status=status, content_type="application/json", body=json.dumps(body))

        if url.path.endswith("/free/items"):
            query = {key: values[0] for key, values in parse_qs(url.query).items()}
            self.reads.append(query["view"])
            items = {"pool": POOL, "manual": [*COPIES, *PROPOSALS], "excluded": [*COPIES, *PROPOSALS], "outcomes": []}[query["view"]]
            return send({"plan": query["plan"], "view": query["view"], "version": query["plan"], "total": len(items), "counts": {},
                         "next_cursor": None, "items": items})
        if url.path.endswith("/free/preview") and request.method == "POST":
            self.phase, self.plan = "ready", self.plan + 1
            return send(self.envelope(started=True), 202)
        if url.path.endswith("/api/storage/free") and request.method == "POST":
            self.posts.append(json.loads(request.post_data))
            self.phase = "done"
            return send(self.envelope(started=True), 202)
        if request.method == "GET" and url.path.endswith("/api/storage"):
            return send(self.envelope())
        return route.continue_()


def drive(page, url, stub, evidence, errors):
    from playwright.sync_api import expect

    page.route(STORAGE_API, stub.answer)
    _open(page, url)
    page.locator('[data-nav-page="dashboard"]').click()
    page.wait_for_selector("#page-dashboard")
    page.locator('[data-dashboard-tab="storage"]').click()
    tray = page.locator("[data-storage-tray]")
    page.locator("[data-storage-door]").click()
    proposals = tray.locator(".storage-decision-row", has=page.locator("[data-free-payload-open]"))
    expect(proposals.locator(".storage-decision-label")).to_contain_text("2 proposals to release heavy files of finished work", timeout=10_000)
    expect(tray.locator(".storage-decision-row", has=page.locator("[data-free-manual-open]"))).to_contain_text("3 finished copies not verified saved")
    page.screenshot(path=str(evidence / "payload-door.png"))

    tray.locator("[data-free-payload-open]").click()
    expect(tray.locator("[data-free-payload-item]")).to_have_count(2, timeout=10_000)
    assert tray.locator("[data-free-manual-pick]:checked").count() == 0, "nothing starts chosen"
    assert "Finished work, one note left" not in tray.inner_text(), "copies stay on their own screen"
    expect(tray).to_contain_text("cannot be measured again")
    for index in (0, 1):
        item = tray.locator("[data-free-payload-item]").nth(index)
        item.locator('details[data-key^="payload-files-"] > summary').click()
        item.locator('details[data-key^="payload-stays-"] > summary').click()
    expect(tray).to_contain_text("frozen copy of before.sqlite taken for the measurement")
    expect(tray).to_contain_text("compare_catalogs.py")
    for width, height, scheme in ((1232, 880, "dark"), (1232, 880, "light"), (390, 844, "dark"), (390, 844, "light")):
        page.set_viewport_size({"width": width, "height": height})
        page.emulate_media(color_scheme=scheme)
        page.wait_for_timeout(150)
        _no_overflow(page, f"{width}x{height} {scheme}")
        tray.scroll_into_view_if_needed()
        page.screenshot(path=str(evidence / f"payload-list-{width}-{scheme}.png"), full_page=True)
    page.set_viewport_size({"width": 1232, "height": 880})

    tray.locator("[data-free-manual-pick]").first.check()  # the first proposal (its index is the whole list's)
    go = tray.locator("[data-free-payload-go]")
    expect(go).to_have_text("Release 2 files")
    go.click()
    dialog = page.get_by_role("dialog")
    expect(dialog).to_contain_text("This deletes 2 files (26 GB) of 1 finished task for good")
    page.screenshot(path=str(evidence / "payload-dialog.png"))
    dialog.get_by_role("button", name="Cancel").click()
    assert stub.posts == []
    go.click()
    page.get_by_role("dialog").get_by_role("button", name="Release 2 files").click()
    expect(tray).to_contain_text("still being deleted", timeout=10_000)
    assert stub.posts == [{"plan": "p1", "manual": ["payload:fixture-task-a+payload-p1"], "accept_loss": True}]
    assert not errors, errors


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_payload_release_flow_in_a_real_browser(direct_server_with_data, request, engine):
    server = direct_server_with_data
    with _browser(server, request, engine) as (page, evidence, errors):
        drive(page, server["url"], Stub(), evidence, errors)


def _seed_saved_files(root):
    """The synthetic server owns this result store. Register exact bytes as a finished task would."""
    import hashlib

    from ouroboros import artifact_storage
    from tests._store_shared import as_install, seed_result

    task = "sample-count"
    directory = "delegated_runs/sample/evidence-final/files/"
    snapshot, report, method, control = [directory + name for name in
                                         ("rows.bin", "report.md", "measure.py", "control.bin")]
    files = {snapshot: b"r\n" * 24000, report: b"Counted 24000 rows.\n",
             method: b"print(len(rows))\n", control: b"another sample"}
    # Keep startup compression from changing the identities recorded by the offer.
    assert all(len(data) < artifact_storage.FLOOR_BYTES for data in files.values())
    result = "The sample contained 24000 rows."
    store = root / "task_results" / "artifacts" / task
    (store / directory).mkdir(parents=True)
    artifacts = []
    for path, data in files.items():
        address = store / path
        address.write_bytes(data)
        artifacts.append({"name": address.name, "path": str(address), "size": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
    with as_install(root):
        seed_result(root, task, {"task_id": task, "root_task_id": task, "status": "completed", "task_type": "task",
                                 "title": "Count sample rows", "result": result, "artifacts": artifacts})
    retained = {"report": [report], "method": [method], "result": ["task_result"]}
    return task, store, files, snapshot, result, retained


def _offer_saved_file(root, task, snapshot, retained):
    from ouroboros.payload_release import propose
    from tests._store_shared import as_install

    with as_install(root):
        offer = propose(root, task, [{"path": snapshot, "role": "frozen input", "loss": "Cannot recount these rows."}],
                        retained, "The original rows cannot be examined again.", proposer="sample-author")
        assert offer["ok"], offer


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_owner_releases_one_saved_file_through_storage(direct_server_with_data, request, engine):
    from pathlib import PurePosixPath

    from playwright.sync_api import expect

    from tests.separate_ui_support import wait_until

    server = direct_server_with_data
    server["stop_server"]()
    task, store, files, snapshot, result, retained = _seed_saved_files(server["data_dir"])
    server["start_server"]()
    _offer_saved_file(server["data_dir"], task, snapshot, retained)
    base = server["url"]
    detail = f"{base}/api/tasks/{task}"
    free = f"{base}/api/storage/free"
    with _browser(server, request, engine) as (page, _evidence, errors):
        def assert_readable(paths):
            for path in paths:
                response = page.request.get(f"{detail}/artifacts/{PurePosixPath(path).name}", params={"relpath": path})
                assert response.status == 200 and response.body() == (store / path).read_bytes() == files[path]
            assert page.request.get(detail).json()["result"] == result

        def release_post(req):
            return req.method == "POST" and req.url == free

        sent = []
        page.on("request", lambda req: sent.append(req.post_data_json) if release_post(req) else None)
        _open(page, base)
        assert_readable(files)
        page.locator('[data-nav-page="dashboard"]').click()
        page.locator('[data-dashboard-tab="storage"]').click()
        page.locator("[data-storage-door]").click()
        tray = page.locator("[data-storage-tray]")
        tray.locator("[data-free-payload-open]").click(timeout=60000)
        row = tray.locator("[data-free-payload-item]")
        expect(row).to_have_count(1)
        expect(row).to_contain_text("The original rows cannot be examined again.")
        pick = row.locator("[data-free-manual-pick]")
        expect(pick).not_to_be_checked()
        row.locator('details[data-key^="payload-files-"] > summary').click()
        row.locator('details[data-key^="payload-stays-"] > summary').click()
        for path in files:
            if path.endswith("control.bin"):
                expect(row).not_to_contain_text(path)
            else:
                expect(row).to_contain_text(path)
        expect(row).to_contain_text("Cannot recount these rows.")
        expect(row).to_contain_text("the task’s own answer")
        key = row.get_attribute("data-free-payload-item")
        plan = page.request.get(base + "/api/storage").json()["free"]["plan"]
        selection = {"plan": plan, "manual": [key]}
        pick.check()
        go = tray.locator("[data-free-payload-go]")
        go.click()
        dialog = page.get_by_role("dialog")
        dialog.get_by_role("button", name="Cancel", exact=True).click()
        expect(dialog).not_to_be_visible()
        assert sent == []
        refused = page.request.post(free, data=selection).json()
        assert refused["started"] is False and refused["reason"] == "loss_not_accepted"
        assert_readable(files)

        go.click()
        with page.expect_response(lambda response: release_post(response.request)) as receipt:
            dialog.get_by_role("button", name="Release 1 file", exact=True).click()
        assert receipt.value.status == 202 and receipt.value.json()["started"]
        assert sent == [{**selection, "accept_loss": True}]
        def poll_free_settled():
            response = page.request.get(base + "/api/storage")
            assert response.status == 200, f"GET /api/storage returned {response.status}"
            body = response.json()
            assert isinstance(body, dict) and isinstance(body.get("free"), dict), body
            free_state = body["free"]
            if free_state.get("plan") == plan and free_state.get("state") in {"done", "failed", "stopped"}:
                return free_state
            return None

        settled = wait_until(
            poll_free_settled,
            timeout=60,
            message=f"storage free plan {plan!r} did not reach a terminal state within 60s",
            interval=0.25,
        )
        assert settled["state"] == "done" and settled["run"]["done"] == 1, settled
        assert not (store / snapshot).exists()
        lost = page.request.get(f"{detail}/artifacts/{PurePosixPath(snapshot).name}", params={"relpath": snapshot})
        assert lost.status == 410 and lost.json()["reason_code"] == "artifact_released"
        assert_readable(files.keys() - {snapshot})
        assert not errors, errors
