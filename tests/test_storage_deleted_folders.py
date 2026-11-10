"""The folders deleted Projects left behind, as a count saves them and `GET /api/storage` answers them.

Registry rows go through the registry's own lifecycle, folders are real, and the reader is the count itself: a place's walk
sizes what lies inside it, and only what no place holds is walked again. The real child and a real Stop close the file.
"""

from __future__ import annotations

import json
import os
import pathlib
import stat
import subprocess
import sys
import time

import pytest

from ouroboros import projects_registry as registry
from ouroboros import storage
from ouroboros.gateway.schema import validate_ingress
from ouroboros.platform_layer import subprocess_new_group_kwargs
from ouroboros.storage import StorageCounter, load_saved, measure, save_count
from tests.test_storage_dashboard_backend import REPO, _join, _table, _write

UNLISTABLE = pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                                reason="needs a folder this user cannot list")


def _deleted(data, project_id, folder="", *, name="", provenance="attached"):
    entry = registry.create_project(data, project_id, name=name or f"Example: {project_id}", working_dir=str(folder))
    registry.update_project(data, entry["id"], provenance=provenance)
    registry.begin_project_deletion(data, entry["id"])
    registry.complete_project_deletion(data, entry["id"])


def _active(data, project_id, folder, *, name=""):
    return registry.create_project(data, project_id, name=name or f"Example: {project_id}", working_dir=str(folder))


@pytest.fixture
def roots(tmp_path):
    """A data folder, a projects root and an empty program, so a count walks only what a case made."""
    data, projects = tmp_path / "data", tmp_path / "work" / "projects"
    (tmp_path / "program").mkdir()
    data.mkdir()
    projects.mkdir(parents=True)
    return data, projects, tmp_path


def _count(roots, **kwargs):
    data, projects, base = roots
    table = _table(data, get_subagent_projects_root=str(projects), REPO_DIR=str(base / "program"))
    return measure(data, places=table, sleep=lambda _s: None, **kwargs)


def _folders(record):
    assert record["deleted_projects"]["state"] == "counted"
    return {row["path"]: row for row in record["deleted_projects"]["folders"]}


def _real(path):
    return os.path.realpath(str(path))


def _place(record, place_id):
    return next(row for row in record["places"] if row["id"] == place_id)


def _roots_walked(monkeypatch):
    """Every root a walk was started on: nothing inside a counted place may be started again."""
    started, real = [], storage._walk_roots

    def spy(roots, walk, named=()):
        started.extend(os.path.join(base, *parts) for base, parts in roots)
        return real(roots, walk, named)

    monkeypatch.setattr(storage, "_walk_roots", spy)
    return started


def test_a_folder_inside_a_place_is_a_slice_of_that_places_walk_with_no_second_walk(roots, monkeypatch):
    data, projects, _base = roots
    folder = projects / "example-video"
    _write(folder / "cache" / "a.bin", 9000)
    _write(folder / "src" / "b.bin", 5000)
    _write(folder / "top.txt", 100)
    _write(projects / "alive" / "c.bin", 7000)
    _deleted(data, "video", folder, provenance="genesis")
    started = _roots_walked(monkeypatch)
    record = _count(roots)
    row = _folders(record)[_real(folder)]
    place = _place(record, "project_folders")
    assert row["state"] == "measured" and row["counted_in"] == "project_folders" and row["unreadable"] == 0
    assert row["bytes"] == next(part["bytes"] for part in place["parts"] if part["name"] == "example-video"), \
        "the same walk's figure for the same folder"
    assert {part["name"] for part in row["parts"]} == {"cache", "src", "(files)"}
    assert sum(part["bytes"] for part in row["parts"]) == row["bytes"]
    assert row["projects"][0]["name"] == "Example: video" and row["projects"][0]["provenance"] == "genesis"
    assert row["projects"][0]["deleted_at"]
    assert _real(folder) not in started
    assert record["bytes"] == sum(row["bytes"] for row in record["places"] if row["bytes"] is not None), "the headline is the places"


def test_an_outside_folder_stays_out_of_the_total_and_the_places_are_kept_first(roots):
    data, _projects, base = roots
    _write(data / "task_results" / "r.bin", 3000)
    outside = base / "elsewhere" / "example-export"
    _write(outside / "exports" / "e.bin", 40_000)
    _deleted(data, "export", outside)
    kept, progress = [], []
    record = _count(roots, keep=lambda count: kept.append(json.loads(json.dumps(count))), progress=progress.append,
                    clock=iter(float(step) for step in range(100_000)).__next__)
    row = _folders(record)[_real(outside)]
    assert row["state"] == "measured" and row["counted_in"] is None and row["bytes"] >= 40_000 and row["same_volume"] is True
    assert record["bytes"] == sum(place["bytes"] for place in record["places"]), "outside bytes join no headline"
    assert record["bytes"] < row["bytes"], "these few files could not hold the folder"
    # Saved once the places are done, unreached, then again as the folder ends.
    assert [_folders(count)[_real(outside)]["state"] for count in kept] == ["not_measured", "measured"]
    assert _folders(kept[0])[_real(outside)]["bytes"] is None and kept[0]["bytes"] == record["bytes"]
    outside_ticks = [tick for tick in progress if tick["place"] == storage.OUTSIDE_PHASE]
    assert outside_ticks and {tick["current"] for tick in outside_ticks} == {"Example: export"}
    assert (outside_ticks[0]["bytes"], outside_ticks[0]["places_done"]) == (0, outside_ticks[0]["places_total"]), "said before its first listing"
    assert all(tick["bytes"] <= row["bytes"] for tick in outside_ticks), "the phase counts those folders alone"


def test_folders_on_this_disk_are_walked_before_another_disks(roots, monkeypatch):
    """Walk order only. A disk that hangs on `stat` is met earlier, while the folders are read, and this does not cover it."""
    data, _projects, base = roots
    near, far = base / "a" / "example-near-with-a-longer-path", base / "b"
    _write(near / "a.bin", 100)
    _write(far / "b.bin", 100)
    _deleted(data, "near", near)
    _deleted(data, "far", far)
    other_disk = os.stat(far).st_ino
    real, order = storage.read_folders, []

    def far_on_another_disk(root, device):
        state, no_folder, marks = real(root, device)
        for mark in marks:
            mark.same_volume = mark.identity[1] != other_disk
        return state, no_folder, marks

    monkeypatch.setattr(storage, "read_folders", far_on_another_disk)
    _count(roots, keep=lambda count: order.append([row["path"] for row in count["deleted_projects"]["folders"] if row["state"] == "measured"]))
    assert order[1] == [_real(near)], "the longer path on this disk, before the shorter one on another"
    assert set(order[2]) == {_real(near), _real(far)}


def test_a_nested_folder_on_this_disk_still_waits_for_its_enclosing_folder_on_another(roots, monkeypatch):
    """This disk first never puts a folder before the outside folder that holds it (a mount inside a folder on another disk):
    walked first, it would be counted again in that walk, in the progress total and in the rows the screen sums."""
    data, _projects, base = roots
    outer, independent = base / "elsewhere" / "outer", base / "elsewhere" / "example-independent"
    _write(outer / "own.bin", 30_000)
    _write(outer / "mnt" / "frame.bin", 50_000)
    _write(independent / "a.bin", 7_000)
    _deleted(data, "outer", outer)
    _deleted(data, "inner", outer / "mnt")
    _deleted(data, "independent", independent)
    real = storage.read_folders

    def outer_on_another_disk(root, device):
        state, no_folder, marks = real(root, device)
        for mark in marks:
            mark.same_volume = mark.path != _real(outer)
        return state, no_folder, marks

    monkeypatch.setattr(storage, "read_folders", outer_on_another_disk)
    progress = []
    rows = _folders(_count(roots, progress=progress.append, clock=iter(float(step) for step in range(100_000)).__next__))
    walks = {}
    for tick in progress:
        if tick["place"] == storage.OUTSIDE_PHASE:
            walks.setdefault(tick["current"], []).append(tick["bytes"])
    assert list(walks) == ["Example: independent", "Example: outer"], "this disk first; the inner folder is met on outer's walk"
    assert rows[_real(outer / "mnt")]["within"] == _real(outer)
    total = rows[_real(outer)]["bytes"] + rows[_real(independent)]["bytes"]
    assert sum(row["bytes"] for row in rows.values() if not row.get("within")) == total, "what the screen sums, each byte once"
    assert max(max(found) for found in walks.values()) <= total, "the progress total too"


def test_a_count_with_nothing_outside_saves_once_as_it_always_did(roots):
    data, projects, _base = roots
    _write(projects / "inside" / "a.bin", 100)
    _deleted(data, "inside", projects / "inside")
    kept = []
    _count(roots, keep=kept.append)
    assert kept == [], "the child's one final save is the only write"


def test_one_folder_of_two_deleted_projects_is_one_row_however_it_is_spelled(roots, monkeypatch):
    data, _projects, base = roots
    folder = base / "shared" / "example-monorepo"
    _write(folder / "pkg" / "a.bin", 20_000)
    spelled = base / "shared" / "link-to-monorepo"
    spelled.symlink_to(folder, target_is_directory=True)
    _deleted(data, "first", folder)
    _deleted(data, "second", spelled)
    started = _roots_walked(monkeypatch)
    record = _count(roots)
    rows = record["deleted_projects"]["folders"]
    assert len(rows) == 1 and rows[0]["path"] == _real(folder)
    assert {project["name"] for project in rows[0]["projects"]} == {"Example: first", "Example: second"}
    assert started.count(_real(folder)) == 1, "walked once, for both"
    assert rows[0]["bytes"] >= 20_000 > rows[0]["bytes"] / 2


@pytest.mark.parametrize("where", ["in_place", "outside"])
def test_a_folder_inside_another_is_part_of_it_and_is_never_walked_or_summed_again(roots, monkeypatch, where):
    data, projects, base = roots
    outer = (projects if where == "in_place" else base / "elsewhere") / "example-pipeline"
    inner = outer / "renders"
    _write(inner / "frame.bin", 30_000)
    _write(outer / "own.bin", 10_000)
    _deleted(data, "inner", inner)  # registered first: the walk's order, not the registry's, decides
    _deleted(data, "outer", outer)
    started = _roots_walked(monkeypatch)
    rows = _folders(_count(roots))
    assert rows[_real(inner)]["within"] == _real(outer) and "within" not in rows[_real(outer)]
    assert rows[_real(outer)]["bytes"] > rows[_real(inner)]["bytes"] >= 30_000
    assert rows[_real(inner)]["counted_in"] == rows[_real(outer)]["counted_in"]
    assert _real(inner) not in started, "met on the enclosing folder's walk"
    assert {part["name"] for part in rows[_real(outer)]["parts"]} == {"renders", "(files)"}


def test_a_place_root_inside_an_outside_folder_stays_that_places(roots):
    data, projects, base = roots
    _write(projects / "alive" / "a.bin", 50_000)
    _write(base / "work" / "notes" / "n.bin", 2000)
    _deleted(data, "wide", base / "work")  # holds the projects root
    record = _count(roots)
    row = _folders(record)[_real(base / "work")]
    assert row["holds"] == ["project_folders"]
    assert 2000 <= row["bytes"] < 50_000, "the place's bytes are the place's"
    assert _place(record, "project_folders")["bytes"] >= 50_000


def test_a_folder_that_is_a_place_root_reports_that_place(roots):
    data, projects, _base = roots
    _write(projects / "alive" / "a.bin", 10_000)
    _deleted(data, "the-root", projects)
    record = _count(roots)
    row = _folders(record)[_real(projects)]
    assert row["counted_in"] == "project_folders" and row["bytes"] == _place(record, "project_folders")["bytes"]


def test_projects_that_remain_are_linked_whichever_way_the_folders_nest(roots):
    data, _projects, base = roots
    same, holding, inside = base / "x" / "same", base / "x" / "holding", base / "y" / "live" / "inside"
    for folder in (same, holding / "live-part", inside):
        _write(folder / "a.bin", 100)
    _deleted(data, "gone-same", same)
    _deleted(data, "gone-holding", holding)
    _deleted(data, "gone-inside", inside)
    _active(data, "live-same", same)
    _active(data, "live-part", holding / "live-part")
    _active(data, "live-outer", base / "y" / "live")
    closed = _active(data, "live-closed", same)
    registry.update_project(data, closed["id"], closed={"at": "2026-11-02T00:00:00Z", "by": "owner", "note": ""})
    unrelated = _active(data, "unrelated", base / "x" / "same-prefix-not-inside")
    assert unrelated["lifecycle"] == registry.PROJECT_ACTIVE
    deleting = registry.create_project(data, "still-deleting", name="Example: still deleting", working_dir=str(same))
    registry.begin_project_deletion(data, deleting["id"])
    rows = _folders(_count(roots))
    links = {name: rows[_real(folder)]["active_projects"] for name, folder in (("same", same), ("holding", holding), ("inside", inside))}
    assert links["same"] == [{"name": "Example: live-closed", "relation": "same"}, {"name": "Example: live-same", "relation": "same"}]
    assert links["holding"] == [{"name": "Example: live-part", "relation": "holds"}]
    assert links["inside"] == [{"name": "Example: live-outer", "relation": "inside"}]
    assert len(rows) == 3, "a Project still deleting is not deleted yet"


def test_with_no_folder_listed_no_remaining_projects_folder_is_looked_up(roots, monkeypatch):
    """Relations are found for a listed folder only. With none, a remaining Project's folder outside every place is never
    resolved or stat'ed, so a disk that does not answer there cannot hold the count; once listed, the lookup finds it."""
    data, _projects, base = roots
    live = base / "elsewhere" / "example-live"
    _write(live / "a.bin", 100)
    _active(data, "live", live)
    _deleted(data, "never-had-one", provenance="none")
    elsewhere, real_live = (str(base / "elsewhere"), _real(base / "elsewhere")), _real(live)  # resolved before the spy
    looked = []
    for name in ("stat", "lstat"):
        def spy(path, *args, _call=getattr(os, name), **kwargs):
            seen = os.fspath(path) if isinstance(path, (str, os.PathLike)) else None
            if isinstance(seen, str):
                looked.append(seen)
            return _call(path, *args, **kwargs)
        monkeypatch.setattr(os, name, spy)
    assert _count(roots)["deleted_projects"] == {"state": "counted", "no_folder": 1, "folders": []}
    assert not [path for path in looked if path.startswith(elsewhere)], "nothing listed to relate it to: never looked up"
    _deleted(data, "gone", live)
    looked.clear()
    rows = _folders(_count(roots))
    assert [path for path in looked if path.startswith(elsewhere)], "the same spy sees the lookups once a folder is listed"
    assert rows[real_live]["active_projects"] == [{"name": "Example: live", "relation": "same"}]


def test_absent_never_assigned_and_a_file_are_different_answers(roots):
    data, _projects, base = roots
    _deleted(data, "gone", base / "Old Drive" / "example-sorter")
    _deleted(data, "no-folder-one", provenance="none")
    _deleted(data, "no-folder-two", provenance="none")
    (base / "a-file").write_text("not a folder")
    _deleted(data, "a-file", base / "a-file")
    record = _count(roots)
    deleted, rows = record["deleted_projects"], _folders(record)
    assert deleted["no_folder"] == 2 and len(rows) == 2
    absent = rows[_real(base / "Old Drive" / "example-sorter")]
    assert (absent["state"], absent["bytes"], absent["counted_in"]) == ("absent", None, None)
    a_file = rows[_real(base / "a-file")]
    assert (a_file["state"], a_file["bytes"], a_file["reason"]) == ("unreadable", None, "not_a_folder")


def test_a_relative_folder_on_record_is_never_resolved_against_the_childs_cwd(roots):
    data, _projects, _base = roots
    _deleted(data, "relative", "example/relative")
    [row] = _count(roots)["deleted_projects"]["folders"]
    assert (row["path"], row["state"], row["reason"], row["bytes"]) == ("example/relative", "unreadable", "not_a_folder", None)


@UNLISTABLE
@pytest.mark.parametrize("where", ["in_place", "outside"])
def test_unreadable_and_partial_are_said_and_never_read_as_small(roots, where):
    data, projects, base = roots
    parent = projects if where == "in_place" else base / "elsewhere"
    closed, partly = parent / "example-locked", parent / "example-partly"
    _write(closed / "hidden.bin", 500)
    _write(partly / "open" / "a.bin", 7000)
    _write(partly / "shut" / "hidden.bin", 500)
    _deleted(data, "locked", closed)
    _deleted(data, "partly", partly)
    closed.chmod(0)
    (partly / "shut").chmod(0)
    try:
        rows = _folders(_count(roots))
    finally:
        closed.chmod(stat.S_IRWXU)
        (partly / "shut").chmod(stat.S_IRWXU)
    expected_place = "project_folders" if where == "in_place" else None
    locked, partial = rows[_real(closed)], rows[_real(partly)]
    assert (locked["state"], locked["bytes"], locked["reason"], locked["counted_in"]) == ("unreadable", None, "permission_denied", expected_place)
    assert (partial["state"], partial["unreadable"], partial["counted_in"]) == ("partial", 1, expected_place)
    assert 7000 <= partial["bytes"], "what could be read, at least"


def test_a_folder_that_vanishes_before_its_walk_is_absent(roots, monkeypatch):
    data, _projects, base = roots
    folder = base / "elsewhere" / "example-brief"
    _write(folder / "a.bin", 10)
    _deleted(data, "brief", folder)
    real = storage._walk_roots

    def vanish(roots_, walk, named=()):
        if walk.place is None:
            (folder / "a.bin").unlink()
            folder.rmdir()
        return real(roots_, walk, named)

    monkeypatch.setattr(storage, "_walk_roots", vanish)
    [row] = _count(roots)["deleted_projects"]["folders"]
    assert (row["state"], row["bytes"]) == ("absent", None)


def test_a_hard_link_counts_where_the_walk_met_it_first(roots):
    data, _projects, base = roots
    linked = _write(data / "task_results" / "shared.bin", 80_000)
    outside = base / "elsewhere" / "example-links"
    outside.mkdir(parents=True)
    os.link(linked, outside / "shared.bin")
    _write(outside / "own.bin", 8000)
    _deleted(data, "links", outside)
    record = _count(roots)
    row = _folders(record)[_real(outside)]
    assert 8000 <= row["bytes"] < 80_000, "bytes already in the total are not outside it"


def test_more_parts_than_a_row_lists_are_summed_as_the_rest(roots):
    data, _projects, base = roots
    folder = base / "elsewhere" / "example-many"
    sizes = {f"part{index}": 5000 + index * 5000 for index in range(11)}
    for name, size in sizes.items():
        _write(folder / name / "a.bin", size)
    _deleted(data, "many", folder)
    row = _folders(_count(roots))[_real(folder)]
    assert [part["name"] for part in row["parts"]] == [f"part{index}" for index in range(10, 2, -1)]
    assert row["rest"]["count"] == 3 and sum(part["bytes"] for part in row["parts"]) + row["rest"]["bytes"] == row["bytes"]


def test_a_part_name_no_text_can_hold_cannot_stop_the_save(roots, monkeypatch):
    """A folder outside every place can hold any bytes in a name (Linux); listing it must not lose the count."""
    data, _projects, base = roots
    folder = base / "elsewhere" / "example-bytes"
    _write(folder / "plain" / "a.bin", 100)
    _deleted(data, "bytes", folder)
    real = storage.saved_folders

    def with_undecodable_part(state, no_folder, marks):
        for mark in marks:
            if mark.parts:
                mark.parts["caf\udcc3"] = 1
        return real(state, no_folder, marks)

    monkeypatch.setattr(storage, "saved_folders", with_undecodable_part)
    record = _count(roots)
    save_count(data, record)
    [row] = load_saved(data)["latest"]["deleted_projects"]["folders"]
    assert {part["name"] for part in row["parts"]} == {"plain", "caf?"}


@pytest.mark.skipif(os.name == "nt", reason="needs a POSIX link to a name no text can hold")
def test_a_never_named_project_is_named_by_its_folder_in_text_the_save_can_hold(roots):
    """A never-named Project (a minted id is its name) is listed by its folder's name, and a folder on record can resolve to
    bytes no text holds (a re-pointed link here; on Linux a real folder): the save keeps the count, in the same replaced text
    as its path, while every name text can hold, a Project's own or a folder's, reads exactly as it is."""
    data, _projects, base = roots
    elsewhere = base / "elsewhere"
    elsewhere.mkdir()
    os.symlink(b"caf\xc3", os.fsencode(elsewhere / "example-link"))
    plain = elsewhere / "example-café"
    _write(plain / "a.bin", 100)
    for project_id, folder in (("proj_deadbeef1234", elsewhere / "example-link"), ("proj_cafe5678", plain)):
        row = registry.create_project(data, project_id, working_dir=str(folder))  # no name: the minted id stands for one
        registry.begin_project_deletion(data, row["id"])
        registry.complete_project_deletion(data, row["id"])
    _deleted(data, "named", elsewhere / "example-named")
    save_count(data, _count(roots))
    rows = {row["path"]: row for row in load_saved(data)["latest"]["deleted_projects"]["folders"]}
    resolved = _real(elsewhere) + os.sep
    assert {path: [project["name"] for project in row["projects"]] for path, row in rows.items()} == {
        resolved + "caf?": ["caf?"], resolved + "example-café": ["example-café"], resolved + "example-named": ["Example: named"]}
    assert rows[resolved + "example-café"]["state"] == "measured" and rows[resolved + "caf?"]["state"] == "absent"


@pytest.mark.parametrize("damage", ["malformed", "lost_beside_its_witness"])
def test_a_registry_that_cannot_be_read_is_unavailable_and_the_count_still_finishes(roots, damage):
    data, projects, _base = roots
    _write(projects / "kept" / "a.bin", 100)
    _deleted(data, "kept", projects / "kept")
    path = data / "state" / "projects.json"
    if damage == "malformed":
        path.write_text("{not json", encoding="utf-8")
    else:
        path.unlink()
    record = _count(roots)
    assert record["deleted_projects"] == {"state": "unavailable", "no_folder": 0, "folders": []}
    assert _place(record, "project_folders")["bytes"] >= 100, "the places are counted all the same"


def test_a_fresh_installation_has_no_deleted_projects(roots):
    assert _count(roots)["deleted_projects"] == {"state": "counted", "no_folder": 0, "folders": []}


def test_an_old_count_has_no_field_and_a_new_deletion_waits_for_the_next_count(roots):
    """GET answers what the saved count saw: a count from before the list says nothing, a deletion after a count changes no
    saved answer, and the next count finds it."""
    data, projects, _base = roots
    old = _count(roots)
    del old["deleted_projects"]
    save_count(data, old)
    counter = StorageCounter()
    assert "deleted_projects" not in counter.projection(data)["count"]

    first = projects / "example-first"
    _write(first / "a.bin", 100)
    _deleted(data, "first", first)
    save_count(data, _count(roots))
    answer = counter.projection(data)["count"]
    assert [row["path"] for row in answer["deleted_projects"]["folders"]] == [_real(first)]
    token = answer["finished_at"]

    second = projects / "example-second"
    _write(second / "b.bin", 100)
    _deleted(data, "second", second)
    between = counter.projection(data)["count"]
    assert between["finished_at"] == token and [row["path"] for row in between["deleted_projects"]["folders"]] == [_real(first)]

    save_count(data, _count(roots))
    assert {row["path"] for row in counter.projection(data)["count"]["deleted_projects"]["folders"]} == {_real(first), _real(second)}


def test_saving_the_same_count_again_keeps_the_growth_baseline(roots):
    data, _projects, base = roots
    _write(data / "task_results" / "r.bin", 4096)
    save_count(data, _count(roots))
    baseline = load_saved(data)["latest"]["finished_at"]
    _write(base / "elsewhere" / "example" / "a.bin", 100)
    _deleted(data, "example", base / "elsewhere" / "example")
    kept = []

    def keep(record):
        kept.append(record["finished_at"])
        save_count(data, record)

    latest = _count(roots, keep=keep)
    save_count(data, latest)
    assert len(kept) == 2
    assert load_saved(data)["previous"]["finished_at"] == baseline, "not the same count's own earlier save"
    assert load_saved(data)["latest"]["finished_at"] == latest["finished_at"] != kept[0], "a new token: the client reads it again"


def test_a_real_record_matches_the_contract(roots):
    from ouroboros.gateway import contracts

    data, projects, base = roots
    _write(projects / "inside" / "a.bin", 100)
    _write(base / "elsewhere" / "outer" / "inner" / "b.bin", 100)
    _deleted(data, "inside", projects / "inside")
    _deleted(data, "outer", base / "elsewhere" / "outer")
    _deleted(data, "inner", base / "elsewhere" / "outer" / "inner")
    _deleted(data, "no-folder", provenance="none")
    _deleted(data, "gone", base / "gone")
    _active(data, "live", projects / "inside")
    record = _count(roots)
    assert validate_ingress(record, contracts.StorageCount) == []
    required = {key for key, kind in contracts.StorageDeletedFolder.__annotations__.items() if "NotRequired" not in str(kind)}
    assert all(required <= set(row) <= set(contracts.StorageDeletedFolder.__annotations__) for row in record["deleted_projects"]["folders"])
    assert all(set(part) == set(contracts.StorageFolderPart.__annotations__) for row in record["deleted_projects"]["folders"] for part in row["parts"])


# ---------------------------------------------------------------- the real child and a real Stop

_HELD_CHILD = """
import os, pathlib, sys, time
from ouroboros import storage

real, gate = storage._walk_roots, pathlib.Path(os.environ["DELETED_FOLDERS_GATE"])

def held(roots, walk, named=()):
    if walk.place is None:  # the first outside folder, held where its listing would begin
        gate.touch()
        time.sleep(600)
    return real(roots, walk, named)

storage._walk_roots = held
sys.exit(storage.main(sys.argv[1:]))
"""


def _child_env(data, base, **extra):
    return {**os.environ, "OUROBOROS_DATA_DIR": str(data), "OUROBOROS_REPO_DIR": str(base / "program"), "PYTHONPATH": str(REPO),
            "OUROBOROS_SETTINGS_PATH": str(base / "settings.json"), "OUROBOROS_SUBAGENT_WORKTREE_ROOT": str(base / "wt"),
            "OUROBOROS_SUBAGENT_PROJECTS_ROOT": str(base / "work" / "projects"), "OUROBOROS_DELIVERABLES_ROOT": str(base / "deliverables"), **extra}


def test_stop_during_an_outside_walk_keeps_the_places_and_says_what_it_did_not_reach(roots, monkeypatch):
    """The real count child and owner; only the outside walk is held, before its first listing (a stand-in for a slow walk;
    a disk that hangs on `stat` while the folders are read is earlier and not covered). What the owner was told: Stop
    cancels it and what was not read stays explicitly incomplete."""
    data, projects, base = roots
    _write(projects / "inside" / "a.bin", 9000)
    _write(data / "task_results" / "r.bin", 5000)
    outside = base / "elsewhere" / "example-nas"
    _write(outside / "a.bin", 7000)
    _deleted(data, "inside", projects / "inside")
    _deleted(data, "nas", outside)
    gate = base / "outside-walk-reached"
    env = _child_env(data, base, DELETED_FOLDERS_GATE=str(gate))

    def spawn(data_root, _repo_dir, _purpose, **_kwargs):
        return subprocess.Popen([sys.executable, "-c", _HELD_CHILD, "--data-root", str(data_root)], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, env=env, cwd=str(REPO), **subprocess_new_group_kwargs())

    monkeypatch.setattr("ouroboros.startup_historical_audit.spawn_maintenance_job", spawn)
    counter = StorageCounter()
    try:
        assert counter.start(data, REPO) is True
        deadline = time.monotonic() + 90
        while not gate.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert gate.exists(), "the child reached the outside walk"

        running = counter.projection(data)
        assert running["counting"] is not None, "still counting"
        saved = running["count"]
        assert saved is not None, "the finished places are the screen's count already"
        rows = _folders(saved)
        assert rows[_real(outside)]["state"] == "not_measured" and rows[_real(projects / "inside")]["state"] == "measured"
        assert counter.cancel() is True
    finally:
        counter.cancel()  # a failure above never leaves the held child sleeping
        _join(counter)
    stopped = counter.projection(data)
    assert stopped["counting"] is None and stopped["error"]["stopped"] is True
    assert stopped["count"]["finished_at"] == saved["finished_at"], "the places stand"
    assert _folders(stopped["count"])[_real(outside)]["state"] == "not_measured"
    assert stopped["count"]["bytes"] == sum(place["bytes"] for place in stopped["count"]["places"])


def test_outside_progress_counts_a_nested_folder_once_as_the_server_holds_it(roots, monkeypatch):
    """Each outside walk starts from what the outside walks before it counted. A folder inside another is met on that one's
    walk, so the independent folder after them starts from the outer folder alone, never outer plus inner. Observed where
    the screen reads it: every running state the server's own count owner holds, merged line by line from the real child."""
    data, _projects, base = roots
    outer, independent = base / "elsewhere" / "outer", base / "elsewhere" / "example-independent"  # longer: walked after outer
    _write(outer / "own.bin", 30_000)
    _write(outer / "inner" / "frame.bin", 50_000)
    _write(independent / "a.bin", 7_000)
    _deleted(data, "outer", outer)
    _deleted(data, "inner", outer / "inner")
    _deleted(data, "independent", independent)
    held = []

    class Watched(StorageCounter):
        """The real owner; every running state it holds, what GET serves at that moment, is also kept here."""

        @property
        def _counting(self):
            return self.__dict__.get("_held")

        @_counting.setter
        def _counting(self, value):
            self.__dict__["_held"] = value
            if value:
                held.append(dict(value))

    env = _child_env(data, base)

    def spawn(data_root, _repo_dir, _purpose, **_kwargs):
        return subprocess.Popen([sys.executable, "-m", "ouroboros.storage", "--data-root", str(data_root)], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, env=env, cwd=str(REPO), **subprocess_new_group_kwargs())

    monkeypatch.setattr("ouroboros.startup_historical_audit.spawn_maintenance_job", spawn)
    counter = Watched()
    try:
        assert counter.start(data, REPO) is True
        _join(counter)
    finally:
        counter.cancel()  # nothing runs on success; on a failure above, the child goes with the test
        _join(counter)
    saved = counter.projection(data)["count"]
    rows = _folders(saved)
    walks, files = {}, {}
    for state in held:
        if state["place"] == storage.OUTSIDE_PHASE:
            walks.setdefault(state["current"], []).append(state["bytes"])
            files.setdefault(state["current"], []).append(state["files"])
    assert list(walks) == ["Example: outer", "Example: independent"], "the inner folder is met on the outer one's walk"
    assert rows[_real(outer / "inner")]["within"] == _real(outer) and rows[_real(outer / "inner")]["bytes"] >= 50_000
    assert walks["Example: outer"][0] == 0
    assert walks["Example: independent"][0] == rows[_real(outer)]["bytes"], "outer alone: its inner folder is already in it"
    assert max(max(found) for found in walks.values()) <= rows[_real(outer)]["bytes"] + rows[_real(independent)]["bytes"]
    assert files["Example: outer"][0] == 0 and files["Example: independent"][0] == 2, "files run on as bytes do: outer's two, once"
    assert max(max(found) for found in files.values()) <= 3
    assert (saved["bytes"], saved["files"]) == (sum(place["bytes"] for place in saved["places"]), sum(place["files"] for place in saved["places"])), \
        "the headline is the places"


def test_the_real_child_measures_outside_folders_and_saves_one_finished_count(roots):
    """No hold: the child as the server starts it, one complete run, then the file it leaves."""
    data, projects, base = roots
    _write(projects / "inside" / "a.bin", 9000)
    outside = base / "elsewhere" / "example-export"
    _write(outside / "a.bin", 7000)
    _deleted(data, "inside", projects / "inside")
    _deleted(data, "export", outside)
    run = subprocess.run([sys.executable, "-m", "ouroboros.storage", "--data-root", str(data)], capture_output=True, text=True,
                         env=_child_env(data, base), cwd=str(REPO), timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    lines = [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
    assert lines[-1]["status"] == "finished"
    saved = load_saved(data)
    assert saved["previous"] is None and saved["latest"]["finished_at"] == lines[-1]["finished_at"]
    rows = _folders(saved["latest"])
    assert rows[_real(outside)]["state"] == rows[_real(projects / "inside")]["state"] == "measured"
    assert rows[_real(outside)]["counted_in"] is None and rows[_real(projects / "inside")]["counted_in"] == "project_folders"


def test_the_mounted_read_never_lists_a_registered_folder(roots, monkeypatch):
    """GET serves the saved count: no folder of a deleted Project is listed, opened or sized by a read."""
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros.gateway.router import collect_routes

    data, _projects, base = roots
    outside = base / "elsewhere" / "example-read"
    _write(outside / "a.bin", 100)
    _deleted(data, "read", outside)
    save_count(data, _count(roots))
    monkeypatch.setattr(storage, "count", StorageCounter())
    routes = [route for route in collect_routes(data_dir=data) if getattr(route, "path", "").startswith("/api/storage")]
    listed, real = [], os.scandir
    monkeypatch.setattr(os, "scandir", lambda path=".": listed.append(pathlib.Path(str(path))) or real(path))
    with TestClient(Starlette(routes=routes)) as client:
        body = client.get("/api/storage").json()
    assert [row["path"] for row in body["count"]["deleted_projects"]["folders"]] == [_real(outside)]
    assert not any(str(path).startswith(str(outside)) for path in listed)
