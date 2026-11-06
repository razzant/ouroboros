"""A real browser consumer of the folders deleted projects leave behind, over the candidate server.

No route stub: projects are created and deleted through the real API, the real count child walks real folders, and the SPA reads
what it saved. A count saved before the list, a deletion between two counts and a real Stop during the outside walk are the same
screen's other answers. Stop is exact: the count child is the one the server's own custody ledger names, proven by its recorded
birth and command before every signal, and it runs in short slices so it is caught inside the outside walk on any machine.
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import stat
import time

import pytest

from ouroboros import platform_layer as pl
from ouroboros import projects_registry as registry
from ouroboros import storage
from ouroboros.process_custody import process_fingerprint_matches, process_stop_snapshot
from tests.test_emoji_free_browser import _browser, _open, _server
from tests.ui_chat_viewport_smoke import wait_drawer_closed

direct_server_with_data = _server
pytestmark = [pytest.mark.ui_browser, pytest.mark.skipif(os.name == "nt", reason="drives the child with POSIX signals")]

MB = 1_000_000
LONG_NAME = "Example: refactoring pass, fifty tasks for next sprints, second try after rollback"[:80]  # the longest a project takes
LONG_FOLDER = "example-refactoring-pass-fifty-tasks-for-the-next-sprints-second-attempt-after-a-rollback-and-a-very-long-discussion"


def _write(path: pathlib.Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _created(page, url, name, *, path=None, **body):
    answer = page.request.post(f"{url}/api/projects", data={"name": name, **({"path": str(path)} if path else {}), **body})
    assert answer.ok, answer.text()
    return answer.json()["project"]


def _deleted(page, url, data_dir, project):
    answer = page.request.post(f"{url}/api/projects/{project['id']}/delete")
    assert answer.ok, answer.text()
    deadline = time.monotonic() + 60
    while registry.get_reserved_project(data_dir, project["id"])["lifecycle"] != registry.PROJECT_TOMBSTONED:
        assert time.monotonic() < deadline, f"{project['id']} was not tombstoned"
        time.sleep(0.05)


def _read(page):
    return page.evaluate("async () => (await fetch('/api/storage', {cache: 'no-store'})).json()")


def _rows(report):
    return {row["path"]: row for row in report["count"]["deleted_projects"]["folders"]}


def _real(path):
    return os.path.realpath(str(path))


def _tree(root):
    return sorted((str(path), path.lstat().st_size) for path in pathlib.Path(root).rglob("*"))


def _reopen(page):
    page.locator('[data-dashboard-tab="logs"]').click()
    page.locator('[data-dashboard-tab="storage"]').click()


def _measure(page):
    measure = page.locator('[data-storage-action="measure"]')
    with page.expect_response(lambda response: response.url.endswith("/api/storage/measure"), timeout=30_000) as posted:
        measure.click()
    assert posted.value.status == 202 and posted.value.json()["started"] is True
    return measure


def _measured(page):
    from playwright.sync_api import expect

    measure = _measure(page)
    expect(page.locator('[data-storage-action="stop"]')).to_be_hidden(timeout=120_000)
    expect(measure).to_have_attribute("aria-disabled", "false", timeout=30_000)
    return _read(page)


def _count_children(data_dir, deadline):
    """The live count children the server's custody ledger names in its data folder, each row's recorded birth and command
    still matching (``process_custody``'s own reader); a row appended while it reads is read again. The free-space sizing
    child records the same purpose; this test starts none, and a second new row fails it rather than picking one."""
    while True:
        try:
            return process_stop_snapshot(data_dir, {storage.PURPOSE})
        except OSError:
            assert time.monotonic() < deadline, "the custody ledger stayed unreadable"
            time.sleep(0.01)


def _identity(entry):
    return int(entry["pid"]), json.dumps(entry.get("fingerprint"), sort_keys=True)


class _CountChild:
    """The count child the server spawned for one Measure, by its custody row. Its recorded birth and command are proven again
    before every signal (``process_fingerprint_matches``), so a pid the system has since given to another process is never
    signalled; signals go through the platform layer. ``release`` resumes whatever this test froze, whatever it met."""

    def __init__(self, entry):
        self.entry, self.pid, self.frozen = entry, int(entry["pid"]), False

    @classmethod
    def spawned_since(cls, data_dir, before, deadline):
        while time.monotonic() < deadline:
            fresh = [entry for entry in _count_children(data_dir, deadline) if _identity(entry) not in before]
            if fresh:
                assert len(fresh) == 1, f"more than one new count child: {fresh}"
                return cls(fresh[0])
            time.sleep(0.005)
        raise AssertionError("the server's custody ledger named no new count child")

    def ours(self):
        return process_fingerprint_matches(self.entry, require_measured=True)

    def freeze(self):
        assert self.ours(), f"the custody row no longer proves pid {self.pid}: the count ended before it was caught"
        assert pl.signal_pid(self.pid, signal.SIGSTOP), f"pid {self.pid} could not be stopped"
        self.frozen = True

    def release(self):
        if self.frozen and self.ours():
            pl.signal_pid(self.pid, signal.SIGCONT)
        self.frozen = False


def _freeze_in_the_outside_walk(page, child, outside, deadline):
    """Catch the child inside the walk of OUTSIDE and leave it frozen there. No signal until the server says the last place
    has begun; from then the child runs in short slices and the server is asked while it is frozen, so the walk cannot pass
    unseen. A frozen walk is what a slow listing looks like to the owner; a disk that hangs on ``stat`` is not this."""
    while time.monotonic() < deadline:
        counting = _read(page).get("counting") or {}
        if counting.get("place") == "deleted_projects" or counting.get("places_done", 0) >= counting.get("places_total", 1) - 1:
            break
        time.sleep(0.01)
    child.freeze()
    while time.monotonic() < deadline:
        answer = _read(page)  # asked while frozen: the walk cannot end between two looks
        rows = ((answer.get("count") or {}).get("deleted_projects") or {}).get("folders", [])
        if ((answer.get("counting") or {}).get("place") == "deleted_projects"
                and any(row["path"] == outside and row["state"] == "not_measured" for row in rows)):
            return
        child.release()
        time.sleep(0.002)
        child.freeze()
    raise AssertionError("the child never reached the outside walk")


# Short name and ids: this case's folder is in the server's manager socket path, which macOS caps (long scratch roots exist).
@pytest.mark.parametrize("engine", ["webkit", "chromium"], ids=["wk", "cr"])
def test_kept(direct_server_with_data, request, engine):
    from playwright.sync_api import expect

    server = direct_server_with_data
    url, data_dir = server["url"], server["data_dir"]
    base = data_dir.parent / "owner-folders"
    export, monorepo, gone, locked = base / "example-export", base / "example-monorepo", base / "example-gone", base / "example-locked"
    _write(export / "exports" / "e.bin", 3 * MB)
    _write(export / "top.txt", 1000)
    _write(monorepo / "packages" / "p.bin", 2 * MB)
    _write(monorepo / "own.bin", MB)
    _write(gone / "a.bin", 1000)
    _write(locked / "hidden.bin", 1000)
    _write(base / LONG_FOLDER / "deep" / "d.bin", 500_000)

    with _browser(server, request, engine) as (page, _evidence, errors):
        posts: list[str] = []
        page.on("request", lambda req: posts.append(req.url.rsplit("/", 1)[-1])
                if "/api/storage/" in req.url and req.method == "POST" else None)
        _open(page, url)
        page.locator('[data-nav-page="dashboard"]').click()
        page.wait_for_selector("#page-dashboard")
        fold = page.locator("[data-storage-deleted]")

        # A count saved before the list existed, as an update finds it: its places stand and the list says not counted.
        first = {"started_at": "2026-10-01T10:00:00+00:00", "finished_at": "2026-10-01T10:05:00+00:00", "duration_seconds": 300.0,
                 "io_priority": "utility", "basis": "allocated", "bytes": 4096, "logical_bytes": 4096, "files": 1, "unreadable": 0,
                 "volume": {"free_bytes": None, "total_bytes": None, "observed_at": "2026-10-01T10:05:00+00:00"},
                 "places": [{"id": "task_results", "group": "work", "roots": [], "present": True, "same_volume": True, "bytes": 4096,
                             "logical_bytes": 4096, "files": 1, "unreadable": 0, "parts": [], "rest": {"count": 0, "bytes": 0}}]}
        saved_file = data_dir / "state" / "storage_measurement.json"
        saved_file.write_text(json.dumps({"_schema_version": 1, "latest": first, "previous": None}), encoding="utf-8")
        page.locator('[data-dashboard-tab="storage"]').click()
        expect(fold).to_contain_text("Not counted yet", timeout=15_000)
        expect(fold).to_contain_text("Refresh sizes to list them.")
        assert posts == [], "opening the tab starts no count"

        # Projects through the real API, each folder kind once; the two attached to one folder, and one still alive on it.
        attached = _created(page, url, "Example: client export", path=export)
        managed = _created(page, url, "Example: video pipeline", with_workspace=True)
        managed_dir = pathlib.Path(managed["working_dir"])
        _write(managed_dir / "cache" / "c.bin", 5 * MB)
        mobile = _created(page, url, "Example: mobile app", path=monorepo)
        payments = _created(page, url, "Example: payments spike", path=monorepo)
        nested = _created(page, url, "Example: packages review", path=monorepo / "packages")
        lost = _created(page, url, "Example: photo sorter", path=gone)
        closed = _created(page, url, "Example: locked archive", path=locked)
        fileless = _created(page, url, "Example: research chat")
        long_named = _created(page, url, LONG_NAME, path=base / LONG_FOLDER)
        _created(page, url, "Example: monorepo maintenance", path=monorepo)
        for project in (attached, managed, mobile, payments, nested, lost, closed, fileless, long_named):
            _deleted(page, url, data_dir, project)
        before = _tree(export), _tree(monorepo), _tree(managed_dir)
        (gone / "a.bin").unlink()
        gone.rmdir()
        locked.chmod(0)
        try:
            report = _measured(page)
        finally:
            locked.chmod(stat.S_IRWXU)
        assert (_tree(export), _tree(monorepo), _tree(managed_dir)) == before, "measuring changes no folder"

        # The saved count answers GET, and the screen says what it answers.
        count = report["count"]
        rows = _rows(report)
        assert len(rows) == 7 and count["deleted_projects"]["no_folder"] == 1
        assert count["bytes"] == sum(place["bytes"] for place in count["places"] if place["bytes"] is not None), "the headline is the places"
        inside = rows[_real(managed_dir)]
        assert inside["counted_in"] == "project_folders" and inside["state"] == "measured" and inside["bytes"] >= 5 * MB
        projects_place = next(place for place in count["places"] if place["id"] == "project_folders")
        assert inside["bytes"] == next(part["bytes"] for part in projects_place["parts"] if part["name"] == managed_dir.name)
        assert inside["projects"][0]["provenance"] == "genesis" and inside["projects"][0]["deleted_at"]
        shared = rows[_real(monorepo)]
        assert {project["name"] for project in shared["projects"]} == {"Example: mobile app", "Example: payments spike"}
        assert shared["counted_in"] is None and shared["bytes"] >= 3 * MB
        assert shared["active_projects"] == [{"name": "Example: monorepo maintenance", "relation": "same"}]
        assert rows[_real(monorepo / "packages")]["within"] == _real(monorepo)
        assert rows[_real(monorepo / "packages")]["bytes"] < shared["bytes"], "part of it, never added"
        assert rows[_real(export)]["bytes"] >= 3 * MB and rows[_real(export)]["projects"][0]["provenance"] == "attached"
        assert rows[_real(gone)]["state"] == "absent" and rows[_real(gone)]["bytes"] is None
        if os.geteuid() != 0:
            assert rows[_real(locked)]["state"] == "unreadable" and rows[_real(locked)]["reason"] == "permission_denied"

        expect(fold).to_contain_text("is already in Project folders above", timeout=15_000)
        expect(fold).to_contain_text("is outside the total")
        expect(fold).to_contain_text("1 not found")
        assert fold.locator("button").count() == 0, "nothing on this list acts on a folder"
        assert 1 <= page.locator(".storage-legend li").count() <= 5, "no sixth group"
        assert fold.locator(".storage-key").count() == 0
        fold.locator("summary").first.click()
        opened = fold.locator(".storage-deleted-open")
        expect(opened).to_contain_text("In Project folders, already in the total")
        expect(opened).to_contain_text("Outside the total")
        expect(opened).to_contain_text("Not found · 1 folder")
        expect(opened).to_contain_text("Moved, removed, or on a disk that is not connected.")
        expect(opened).to_contain_text("Managed folder")
        expect(opened).to_contain_text("Attached folder")
        expect(opened).to_contain_text("shared by 2 deleted projects, counted once")
        expect(opened).to_contain_text("Still the folder of project: Example: monorepo maintenance.")
        expect(opened).to_contain_text("Inside the folder of Example: mobile app · Example: payments spike, so its size is part of that one.")
        expect(opened).to_contain_text("1 deleted project never had a folder.")
        if os.geteuid() != 0:
            expect(opened).to_contain_text("Could not be read: permission denied.")
        export_row = opened.locator('[data-deleted-folder="measured"]', has_text="Example: client export")
        export_row.locator(".storage-parts > summary").click()
        expect(export_row.locator(".storage-parts-list")).to_contain_text("exports")
        expect(export_row.locator(".storage-parts-list")).to_contain_text("(files)")

        # The longest name and path stay inside the card, at the widest and the narrowest shipped window.
        body = page.locator("#page-storage")  # the tab body is its own scroller
        long_row = opened.locator('[data-deleted-folder="measured"]', has_text=LONG_NAME)
        expect(long_row).to_contain_text(LONG_FOLDER)
        assert body.evaluate("node => node.scrollWidth - node.clientWidth") <= 2
        page.set_viewport_size({"width": 390, "height": 844})
        if page.locator("body").evaluate("node => node.classList.contains('nav-drawer-open')"):
            if page.locator("#nav-drawer-backdrop").is_visible():
                page.locator("#nav-drawer-backdrop").click()
            page.wait_for_timeout(300)
        wait_drawer_closed(page)
        assert body.evaluate("node => node.scrollWidth - node.clientWidth") <= 2, "no sideways scroll on a phone"
        page.set_viewport_size({"width": 1280, "height": 900})

        # A deletion between two counts: the saved answer is dated and stays until the next count.
        token = count["finished_at"]
        later_dir = base / "example-later"
        _write(later_dir / "l.bin", 500_000)
        later = _created(page, url, "Example: deleted later", path=later_dir)
        _deleted(page, url, data_dir, later)
        between = _read(page)
        assert between["count"]["finished_at"] == token and _real(later_dir) not in _rows(between)
        _reopen(page)
        expect(page.locator("[data-storage-places]")).not_to_contain_text("Example: deleted later")
        report = _measured(page)
        assert _rows(report)[_real(later_dir)]["state"] == "measured"
        assert report["count"]["growth_since"] == token, "the count before it is the baseline"
        expect(fold).to_contain_text("Example: deleted later")

        # Stop while the walk is outside every place: the places stand, what it did not reach says so.
        held = base / "example-held"  # the shortest outside path: walked first after the places
        held.mkdir()
        for index in range(20_000):
            (held / f"f{index:05}").touch()
        held_project = _created(page, url, "Example: held walk", path=held)
        _deleted(page, url, data_dir, held_project)
        token = report["count"]["finished_at"]
        before = {_identity(entry) for entry in _count_children(data_dir, time.monotonic() + 30)}
        measure = _measure(page)
        child = _CountChild.spawned_since(data_dir, before, time.monotonic() + 60)
        try:
            _freeze_in_the_outside_walk(page, child, _real(held), time.monotonic() + 120)
            expect(page.locator('[data-storage-action="stop"]')).to_be_visible(timeout=15_000)
            expect(page.locator("[data-storage-status]")).to_contain_text("Measuring folders of deleted projects outside these places", timeout=15_000)
            expect(fold).to_contain_text("not measured yet", timeout=15_000)
            running = _read(page)
            assert running["counting"] is not None and running["count"]["finished_at"] != token, "the finished places are the count already"
            page.locator('[data-storage-action="stop"]').click()
            expect(page.locator("[data-storage-status]")).to_contain_text("Stopped before it finished. Showing the count from", timeout=30_000)
        finally:
            child.release()  # whatever happened above, nothing this test froze stays frozen
        expect(page.locator('[data-storage-action="stop"]')).to_be_hidden()
        expect(measure).to_have_attribute("aria-disabled", "false")
        stopped = _read(page)
        assert stopped["counting"] is None and stopped["error"]["stopped"] is True
        assert _rows(stopped)[_real(held)]["state"] == "not_measured" and _rows(stopped)[_real(held)]["bytes"] is None
        assert stopped["count"]["bytes"] == sum(place["bytes"] for place in stopped["count"]["places"] if place["bytes"] is not None)
        assert not child.ours(), "the Stop ended the very child the ledger named"
        unreached = sum(row["state"] == "not_measured" for row in stopped["count"]["deleted_projects"]["folders"])
        assert unreached >= 1
        expect(fold).to_contain_text(f"{unreached} not measured yet")

        finished = _measured(page)
        assert _rows(finished)[_real(held)]["state"] == "measured"
        assert finished["error"] is None

        # A place's row opens the same "Largest parts" from the same renderer, under its own keys.
        page.locator('[data-storage-place="app_state"] summary[data-key="place-app_state-open"]').click()
        place_parts = page.locator('[data-storage-place="app_state"] details[data-key="parts-app_state"]')
        place_parts.locator('summary[data-key="parts-app_state-open"]').click()
        expect(place_parts.locator(".storage-parts-list")).to_be_visible()
        assert not errors, errors
