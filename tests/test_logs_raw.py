"""Actual HTTP consumers of compact legacy logs and their verified Raw references.

Synthetic bodies only. No installed data, producer, browser or sidecar is exercised.
"""
from __future__ import annotations

import base64
import hashlib
import json

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros.jsonl_tail import JsonlChainSnapshot
from tests._logs_raw_shared import client, preview, read, write
from tests._store_shared import seed_result


def edited_ref(row, **changes):
    source = {**row["_raw"]["source"], **changes}
    return {"_raw": {"ref": base64.urlsafe_b64encode(json.dumps(source).encode()).decode().rstrip("=")}}


def tree_bytes(root, *, sqlite_read_marks=False):
    # SQLite readers may change the existing SHM reader marks. Keep its name in
    # the snapshot; main DB, WAL, source bytes and every other file stay exact.
    return {str(p.relative_to(root)): ("SQLite reader marks" if sqlite_read_marks and p.name == "current.sqlite-shm"
                                      else hashlib.sha256(p.read_bytes()).hexdigest())
            for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("name", ["events", "tools", "supervisor"])
def test_legacy_oversized_backfill_compact_page_download_and_unchanged_default(tmp_path, name):
    # The real failure was above the sidecar's 8 MiB frame budget. Match that
    # scale without copying any private event or human words into a fixture.
    body = "synthetic-diagnostic-" * 700_000
    row = {"ts": "2026-11-04T00:00:00Z", "type": "task_checkpoint", "task_id": "task",
           "context_view": {"messages": [{"content": body}], "duplicate": body}}
    raw = write(tmp_path, row, name)
    before = tree_bytes(tmp_path)
    http = client(tmp_path)
    full = http.get(f"/api/logs/{name}")
    assert full.json()["entries"] == [{**row, "_source_root": str(tmp_path), "_line": 1}]
    assert http.get(f"/api/logs/{name}?view=full").content == full.content
    response = http.get(f"/api/logs/{name}?view=compact")
    assert response.status_code == 200
    assert len(response.content) < 3000
    entry = response.json()["entries"][0]
    assert "context_view" not in entry
    assert entry["_raw"]["omitted_fields"] == ["context_view"]
    assert entry["_raw"]["source"] == {"v": 1, "kind": "log_row", "placement": "canonical", "task_id": "",
        "log": name, "start": 0, "length": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    page = read(http, entry, name).json()
    assert len(page["text"].encode()) == page["next_offset"] == 65536
    assert page["has_more"] and page["total_bytes"] == len(raw)
    assert page["text"].encode() == raw[:65536]
    assert page["sha256"] == entry["_raw"]["source"]["sha256"]
    downloaded = read(http, entry, name, download=1)
    assert downloaded.status_code == 200 and downloaded.content == raw
    assert downloaded.headers["x-ouroboros-raw-representation"] == "stored"
    assert downloaded.headers["content-disposition"].startswith("attachment;")
    assert tree_bytes(tmp_path) == before, "neither passive preview nor explicit Raw creates an archive/cache"


def test_reference_survives_append_multiple_rotations_and_a_fresh_reader(tmp_path):
    raw = write(tmp_path, {"type": "task_checkpoint", "context_view": "x" * (9 * 1024 * 1024)})
    row = preview(client(tmp_path))[0]
    archive = tmp_path / "archive"
    archive.mkdir()
    for i in range(4):
        write(tmp_path, {"type": "sentinel", "round": i}, append=True)
        (tmp_path / "logs/events.jsonl").rename(archive / f"events_20261104T00000{i}.jsonl")
        write(tmp_path, {"type": "after_rotation", "round": i})
    # A new request/reader has no retained snapshot or ref registry. The first
    # source now lies beyond the preview tail's three-archive backfill budget.
    assert read(client(tmp_path), row, download=1).content == raw
    assert all(item.get("type") != "task_checkpoint" for item in preview(client(tmp_path)))


def test_rotation_between_capture_and_payload_open_rebinds_original_inode(tmp_path, monkeypatch):
    raw = write(tmp_path, {"type": "old", "value": "unchanged"})
    row = preview(client(tmp_path))[0]
    original = JsonlChainSnapshot._read
    rotated = []

    def rotate(snapshot, start, end):
        if not rotated:
            rotated.append(True)
            (tmp_path / "archive").mkdir()
            (tmp_path / "logs/events.jsonl").rename(tmp_path / "archive/events_20261104T000000.jsonl")
            write(tmp_path, {"type": "new"})
        return original(snapshot, start, end)

    monkeypatch.setattr(JsonlChainSnapshot, "_read", rotate)
    assert read(client(tmp_path), row, download=1).content == raw


@pytest.mark.parametrize("change", ["missing", "truncated", "changed", "symlink"])
def test_source_failures_never_fall_back_to_newer_rows(tmp_path, change):
    path = tmp_path / "logs/events.jsonl"
    raw = write(tmp_path, {"type": "old", "value": "unchanged"})
    row = preview(client(tmp_path))[0]
    if change == "missing":
        path.unlink()
    elif change == "truncated":
        path.write_bytes(raw[:-1])
    elif change == "changed":
        path.write_bytes(raw.replace(b"old", b"new"))
    else:
        external = tmp_path / "other.jsonl"
        external.write_bytes(raw)
        path.unlink()
        path.symlink_to(external)
    response = read(client(tmp_path), row)
    assert response.status_code == (409 if change == "changed" else 404)
    assert response.json()["reason_code"] == ("log_source_changed" if change == "changed" else "log_source_unavailable")
    assert "text" not in response.json()


@pytest.mark.parametrize("target", ["logs", "archive", "archive_member"])
def test_compact_refuses_links_even_when_target_bytes_match(tmp_path, target):
    data = tmp_path / "data"
    raw = write(data, {"type": "old"})
    row = preview(client(data))[0]
    external = tmp_path / "external"
    external.mkdir()
    if target == "logs":
        (data / "logs/events.jsonl").unlink()
        (data / "logs").rmdir()
        (external / "events.jsonl").write_bytes(raw)
        (data / "logs").symlink_to(external, target_is_directory=True)
    else:
        (data / "logs/events.jsonl").unlink()
        (external / "events_20261104.jsonl").write_bytes(raw)
        if target == "archive":
            (data / "archive").symlink_to(external, target_is_directory=True)
        else:
            (data / "archive").mkdir()
            (data / "archive/events_20261104.jsonl").symlink_to(external / "events_20261104.jsonl")
    assert read(client(data), row).status_code == 404
    assert client(data).get("/api/logs/events?view=compact").status_code == 503


def test_public_projection_is_reapplied_and_has_a_separate_identity(tmp_path):
    row = {"ts": "2026-11-04Z", "type": "task_checkpoint", "status": "completed", "result_status": "failed",
           "swarm_fanouts": [{"private_plane": "not in public contract"}], "nested": {"compat_result_status": "old"},
           "body": "x" * 20000}
    raw = write(tmp_path, row)
    http = client(tmp_path)
    full = http.get("/api/logs/events").json()["entries"][0]
    full.pop("_line")
    full.pop("_source_root")
    entry = preview(http)[0]
    response = read(http, entry, download=1)
    assert json.loads(response.content) == full
    assert response.content != raw
    assert response.headers["x-ouroboros-raw-representation"] == "public_task_result"
    assert response.headers["x-ouroboros-source-sha256"] == hashlib.sha256(raw).hexdigest()
    assert response.headers["x-ouroboros-payload-sha256"] == hashlib.sha256(response.content).hexdigest()
    assert entry["_raw"]["public_bytes"] == len(response.content)
    assert "result_status" not in full and "swarm_fanouts" not in full
    assert full["nested"] == {} and "outcome_axes" in full


def test_raw_refuses_overwritten_json_members_instead_of_leaking_unprojected_bytes(tmp_path):
    write(tmp_path, {"type": "row"})
    http = client(tmp_path)
    row = preview(http)[0]
    raw = b'{"type":"row","nested":{"result_status":"withheld"},"nested":{"ok":true}}\n'
    (tmp_path / "logs/events.jsonl").write_bytes(raw)
    forged = edited_ref(row, length=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    assert read(http, forged, download=1).status_code == 409
    compact = http.get("/api/logs/events?view=compact")
    assert compact.status_code == 200
    assert compact.json() == {"name": "events", "entries": [], "view": "compact",
                              "gaps": ["malformed_jsonl"], "skipped_rows": 1}
    assert http.get("/api/logs/events").json()["entries"][0]["nested"] == {"ok": True}


@pytest.mark.parametrize("damage", [
    b'{"type":"torn"' + b'{"type":"glued_append"}\n', b'[1,2]\n',
    b'{"type":"duplicate","nested":{"result_status":"withheld"},"nested":{}}\n',
], ids=["torn_glued_append", "non_object", "duplicate_member"])
@pytest.mark.parametrize("name", ["events", "tools", "supervisor"])
def test_compact_keeps_addressable_rows_around_damage(tmp_path, damage, name):
    first = write(tmp_path, {"type": "before", "task_id": "wanted"}, name)
    with (tmp_path / "logs" / f"{name}.jsonl").open("ab") as handle:
        handle.write(damage * 2)
    last = write(tmp_path, {"type": "after", "task_id": "wanted", "text": "аб🦊"}, name, append=True)
    before = tree_bytes(tmp_path)
    http = client(tmp_path)
    response = http.get(f"/api/logs/{name}", params={"view": "compact", "task_id": "wanted"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["gaps"] == ["malformed_jsonl"] and payload["skipped_rows"] == 2
    rows = payload["entries"]
    assert [row["type"] for row in rows] == ["before", "after"]
    for row, raw, start in zip(rows, [first, last], [0, len(first) + 2 * len(damage)]):
        assert row["_raw"]["source"]["start"] == start
        assert row["_raw"]["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert read(http, row, name, download=1).content == raw
    assert tree_bytes(tmp_path) == before


def test_compact_gap_count_is_measured_once_per_selected_window(tmp_path):
    first = write(tmp_path, {"type": "first"})
    path = tmp_path / "logs/events.jsonl"
    with path.open("ab") as handle:
        handle.write(b'[1,2]\n')
    write(tmp_path, {"type": "large", "body": "x" * 1_100_000}, append=True)
    with path.open("ab") as handle:
        handle.write(b'{bad}\n')
    write(tmp_path, {"type": "last"}, append=True)
    http = client(tmp_path)
    narrow = http.get("/api/logs/events?view=compact&limit=1").json()
    broad = http.get("/api/logs/events?view=compact&limit=100").json()
    assert narrow["skipped_rows"] == 1 and broad["skipped_rows"] == 2
    assert narrow["gaps"] == broad["gaps"] == ["malformed_jsonl"]
    assert narrow["entries"][0]["_raw"] == broad["entries"][-1]["_raw"]
    assert read(http, broad["entries"][0], download=1).content == first


@pytest.mark.parametrize("archived", [False, True], ids=["live", "archive"])
def test_compact_discloses_unterminated_record_without_losing_valid_rows(tmp_path, archived):
    first = write(tmp_path, {"type": "before"})
    path = tmp_path / "logs/events.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"type":"unfinished"}')
    if archived:
        (tmp_path / "archive").mkdir()
        path.rename(tmp_path / "archive/events_20261104.jsonl")
        last = write(tmp_path, {"type": "after"})
    http = client(tmp_path)
    response = http.get("/api/logs/events?view=compact")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["skipped_rows"] == 1
    assert ("torn_archive_line" if archived else "incomplete_live_line") in payload["gaps"]
    assert [row["type"] for row in payload["entries"]] == (["before", "after"] if archived else ["before"])
    assert read(http, payload["entries"][0], download=1).content == first
    if archived:
        assert read(http, payload["entries"][-1], download=1).content == last


def test_raw_refuses_a_forged_row_joined_across_archive_and_live(tmp_path):
    first = write(tmp_path, {"type": "before"})
    http = client(tmp_path)
    selector = preview(http)[0]
    prefix, suffix = b'{"type":"joined",', b'"task_id":"allowed"}\n'
    with (tmp_path / "logs/events.jsonl").open("ab") as handle:
        handle.write(prefix)
    (tmp_path / "archive").mkdir()
    (tmp_path / "logs/events.jsonl").rename(tmp_path / "archive/events_000000.jsonl")
    (tmp_path / "logs/events.jsonl").write_bytes(suffix)
    invented_row = prefix + suffix
    forged = edited_ref(selector, start=len(first), length=len(invented_row),
                        sha256=hashlib.sha256(invented_row).hexdigest())
    response = read(http, forged, download=1)
    assert response.status_code == 409 and response.json()["reason_code"] == "log_source_changed"


def test_damaged_rotating_chain_keeps_exact_refs_then_refuses_source_conflict(tmp_path):
    raw = write(tmp_path, {"type": "wanted", "text": "аб🦊"})
    with (tmp_path / "logs/events.jsonl").open("ab") as handle:
        handle.write(b'{torn\rstill_one_record}\n')
    write(tmp_path, {"type": "after"}, append=True)
    http = client(tmp_path)
    response = http.get("/api/logs/events?view=compact")
    assert response.status_code == 200, response.text
    assert response.json()["skipped_rows"] == 1
    row = response.json()["entries"][0]
    archive = tmp_path / "archive"
    archive.mkdir()
    for index in range(4):
        (tmp_path / "logs/events.jsonl").rename(archive / f"events_{index:06d}.jsonl")
        write(tmp_path, {"type": "new", "round": index})
        assert read(client(tmp_path), row, download=1).content == raw
        current = http.get("/api/logs/events?view=compact").json()
        assert current["skipped_rows"] == (1 if index < 3 else 0)
    old = archive / "events_000000.jsonl"
    old.write_bytes(old.read_bytes().replace(b"wanted", b"forged", 1))
    conflict = read(client(tmp_path), row, download=1)
    assert conflict.status_code == 409 and conflict.json()["reason_code"] == "log_source_changed"
    assert "text" not in conflict.json()


@pytest.mark.parametrize("archived", [False, True])
def test_unreadable_captured_source_never_claims_zero_skipped_rows(tmp_path, monkeypatch, archived):
    write(tmp_path, {"type": "captured"})
    if archived:
        (tmp_path / "archive").mkdir()
        (tmp_path / "logs/events.jsonl").rename(tmp_path / "archive/events_000000.jsonl")

    def unavailable(*_args):
        raise OSError("captured source disappeared before read")

    monkeypatch.setattr(JsonlChainSnapshot, "_read", unavailable)
    response = client(tmp_path).get("/api/logs/events?view=compact")
    assert response.status_code == 503
    assert response.json()["reason_code"] == "log_source_unavailable"
    assert "skipped_rows" not in response.json() and "entries" not in response.json()


def test_raw_uses_existing_network_auth_and_browser_origin_boundary(tmp_path, monkeypatch):
    from ouroboros import config, server_auth

    write(tmp_path, {"type": "owner_log"})
    inner = client(tmp_path)
    row = preview(inner)[0]
    monkeypatch.setenv("OUROBOROS_NETWORK_PASSWORD", "test-log-password")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_auth, "_auth_secret_cache", None)
    gate = server_auth.NetworkAuthGate(inner.app)

    async def remote(scope, receive, send):
        await gate({**scope, "client": ("198.51.100.7", 40000)}, receive, send)

    http = TestClient(remote)
    assert http.get("/api/logs/events").status_code == 401
    assert read(http, row).status_code == 401, "knowing the source reference does not authenticate"
    assert http.post("/auth/login", json={"password": "test-log-password", "next": "/"}).status_code == 200
    assert read(http, row).status_code == 200
    assert http.get("/api/logs/events/raw", params={"ref": row["_raw"]["ref"]},
                    headers={"Origin": "https://another.example"}).status_code == 403


def test_pages_reassemble_utf8_and_do_not_mistake_end_for_missing(tmp_path):
    raw = write(tmp_path, {"type": "unicode", "text": "аб🦊" * 40})
    http = client(tmp_path)
    row = preview(http)[0]
    parts, offset = [], 0
    while True:
        response = read(http, row, offset=offset, limit=17)
        assert response.status_code == 200
        page = response.json()
        assert len(page["text"].encode()) <= 17
        parts.append(page["text"])
        offset = page["next_offset"]
        if not page["has_more"]:
            break
    assert "".join(parts).encode() == raw
    eof = read(http, row, offset=len(raw)).json()
    assert eof["text"] == "" and eof["has_more"] is False
    mid_codepoint = raw.index("🦊".encode()) + 1
    assert read(http, row, offset=mid_codepoint).status_code == 416
    assert read(http, row, offset=len(raw) + 1).status_code == 416


@pytest.mark.parametrize("params", [{"offset": -1}, {"limit": 1048577}, {"limit": 0}, {"limit": "x"},
                                  {"download": "yes"}, {"download": 1, "limit": 100}])
def test_invalid_page_bounds_are_typed(tmp_path, params):
    write(tmp_path, {"type": "row"})
    http = client(tmp_path)
    assert read(http, preview(http)[0], **params).status_code == 400


def test_forged_reference_cannot_choose_path_log_task_or_partial_row(tmp_path):
    raw = write(tmp_path, {"task_id": "a", "type": "row", "value": {"x": "nested"}})
    http = client(tmp_path)
    row = preview(http)[0]
    for placement in ("../other", str(tmp_path), "task_drives/unknown", "logs/events.jsonl"):
        assert read(http, edited_ref(row, placement=placement)).status_code == 404
    assert read(http, edited_ref(row, log="tools")).status_code == 400
    assert read(http, edited_ref(row, task_id="b")).status_code == 404
    assert read(http, edited_ref(row, task_id="../a")).status_code == 400
    start = raw.index(b'{"x"')
    partial = raw[start:-2] + b"\n"
    # A digest is not a licence to reinterpret bytes from the middle of a row.
    assert read(http, edited_ref(row, start=start, length=len(partial), sha256=hashlib.sha256(partial).hexdigest())).status_code == 409
    assert read(http, edited_ref(row, v=True)).status_code == 400
    assert read(http, edited_ref(row, path="/arbitrary")).status_code == 400


@pytest.mark.parametrize("layout", ["task_drives/child", "state/headless_tasks/child/data"])
def test_admitted_child_drive_scope_and_revocation(tmp_path, layout):
    child = tmp_path / layout
    seed_result(tmp_path, "parent", {"status": "running"})
    admission = {"parent_task_id": "parent", "root_task_id": "parent", "delegation_role": "subagent",
                 "child_drive_root": str(child), "status": "running"}
    seed_result(tmp_path, "child", admission)
    seed_result(child, "child", {**admission, "task_id": "child"})
    write(child, {"ts": "2026-11-04Z", "task_id": "sibling", "type": "noise"})
    raw = write(child, {"ts": "2026-11-04Z", "task_id": "child", "type": "child", "body": "x" * 20000}, append=True)
    http = client(tmp_path)
    rows = preview(http, task_id="parent")
    assert len(rows) == 1 and rows[0]["type"] == "child"
    row = rows[0]
    assert row["_raw"]["source"]["placement"] == layout
    assert read(http, row, download=1).content == raw
    assert read(http, edited_ref(row, task_id="unrelated")).status_code == 404
    seed_result(tmp_path, "child", {**admission, "child_drive_root": ""})
    assert read(http, row).status_code == 404


@pytest.mark.parametrize("link", ["superseded_by", "retry_task_id"])
@pytest.mark.parametrize("layout", ["task_drives/retry", "state/headless_tasks/retry/data"])
def test_original_task_scope_follows_effective_retry_drive(tmp_path, link, layout):
    """The original HTTP filter must find the same retry event in both views."""
    child = tmp_path / layout
    seed_result(tmp_path, "old", {"status": "failed", link: "retry"})
    seed_result(tmp_path, "retry", {"status": "completed", "root_task_id": "old", "child_drive_root": str(child)})
    seed_result(child, "retry", {"status": "completed", "root_task_id": "old"})
    # The query's row predicate is unchanged: following a root's placement does
    # not make unlinked rows belonging to its physical successor visible.
    write(child, {"task_id": "retry", "type": "outside_original_scope"})
    raw = write(child, {"task_id": "retry", "root_task_id": "old", "type": "retry_event"}, append=True)
    http = client(tmp_path)
    full = http.get("/api/logs/events?task_id=old").json()["entries"]
    rows = preview(http, task_id="old")
    assert [row["type"] for row in full] == [row["type"] for row in rows] == ["retry_event"]
    assert {key: value for key, value in rows[0].items() if key != "_raw"} == full[0]
    assert rows[0]["_raw"]["source"]["placement"] == layout
    assert rows[0]["_raw"]["source"]["task_id"] == "old"
    assert read(http, rows[0], download=1).content == raw


@pytest.mark.parametrize("layout", ["task_drives/child", "state/headless_tasks/child/data"])
@pytest.mark.parametrize("field", ["child_drive_root", "headless_child_drive_root", "drive_root"])
@pytest.mark.parametrize("nested", [False, True], ids=["row", "metadata"])
@pytest.mark.parametrize("scope", ["task", "parent", "retry", "child_retry", "queue"])
@pytest.mark.parametrize("replica", [False, True], ids=["canonical_only", "with_child_record"])
def test_full_and_compact_share_all_recorded_placement_forms(tmp_path, layout, field, nested, scope, replica):
    child = tmp_path / layout
    placement = {field: str(child)}
    admission = {"status": "completed", **({"metadata": placement} if nested else placement)}
    query = "parent" if scope in {"parent", "child_retry", "queue"} else "old" if scope == "retry" else "child"
    if scope in {"parent", "child_retry"}:
        admission.update(parent_task_id="parent", root_task_id="parent", delegation_role="subagent")
    if scope in {"retry", "child_retry"}:
        seed_result(tmp_path, "old", {**admission, "status": "failed", "retry_task_id": "child"})
    seed_result(tmp_path, "child", admission)
    if replica:
        seed_result(child, "child", {"status": "completed"})
    if scope == "queue":
        from ouroboros.utils import utc_now_iso

        (tmp_path / "state/queue_snapshot.json").write_text(json.dumps({"ts": utc_now_iso(), "pending": [],
            "running": [{"id": "child", "task": {"id": "child", "parent_task_id": "parent",
                         "root_task_id": "parent", "delegation_role": "subagent"}}]}))
    raw = write(child, {"task_id": "child", "root_task_id": query, "type": "supported", "body": "x" * 17000})
    http = client(tmp_path)
    before = tree_bytes(tmp_path, sqlite_read_marks=True)
    full = http.get("/api/logs/events", params={"task_id": query})
    compact = http.get("/api/logs/events", params={"task_id": query, "view": "compact"})
    assert full.status_code == compact.status_code == 200
    assert set(full.json()) == {"name", "entries"}
    assert [row["type"] for row in full.json()["entries"]] == ["supported"], full.text
    assert [row["type"] for row in compact.json()["entries"]] == ["supported"], compact.text
    assert full.json()["entries"][0]["body"] == "x" * 17000
    assert read(http, compact.json()["entries"][0], download=1).content == raw
    assert tree_bytes(tmp_path, sqlite_read_marks=True) == before


@pytest.mark.parametrize("view", ["full", "compact"])
def test_scope_uses_canonical_placement_even_when_child_claims_another(tmp_path, view):
    child, other = tmp_path / "task_drives/child", tmp_path / "task_drives/other"
    seed_result(tmp_path, "child", {"status": "completed", "child_drive_root": str(child)})
    seed_result(child, "child", {"status": "completed", "child_drive_root": str(other)})
    write(child, {"task_id": "child", "type": "admitted"})
    write(other, {"task_id": "child", "type": "unadmitted"})
    response = client(tmp_path).get("/api/logs/events", params={"task_id": "child", "view": view})
    assert response.status_code == 200
    assert [row["type"] for row in response.json()["entries"]] == ["admitted"]


def test_full_and_compact_keep_multiple_recorded_placements(tmp_path):
    pool, headless = tmp_path / "task_drives/child", tmp_path / "state/headless_tasks/child/data"
    seed_result(tmp_path, "child", {"status": "completed", "child_drive_root": str(pool),
                                   "metadata": {"drive_root": str(headless)}})
    write(pool, {"ts": "1", "task_id": "child", "type": "pool"})
    write(headless, {"ts": "2", "task_id": "child", "type": "headless"})
    http = client(tmp_path)
    for view in ("full", "compact"):
        response = http.get("/api/logs/events", params={"task_id": "child", "view": view})
        assert response.status_code == 200
        assert [row["type"] for row in response.json()["entries"]] == ["pool", "headless"]


@pytest.mark.parametrize("links, expected", [
    ({"old": {"superseded_by": "middle"}, "middle": {"retry_task_id": "retry"}}, "retry"),
    ({"old": {"retry_task_id": "middle"}, "middle": {"superseded_by": "retry"},
      "retry": {"retry_task_id": "old"}}, "retry"),
    ({"old": {"retry_task_id": "old"}}, "old"),
    ({"old": {"superseded_by": "absent"}}, "old"),
    ({"old": {"superseded_by": "retry", "retry_task_id": "middle"}}, "retry"),
], ids=["multihop", "cycle", "self", "missing", "supersession_precedence"])
def test_retry_scope_keeps_existing_effective_leaf_semantics(tmp_path, links, expected):
    sources = {}
    for identity in ("old", "middle", "retry"):
        child = tmp_path / "task_drives" / identity
        row = {"status": "completed", "child_drive_root": str(child), **links.get(identity, {})}
        seed_result(tmp_path, identity, row)
        seed_result(child, identity, {"status": "completed"})
        sources[identity] = write(child, {"task_id": identity, "root_task_id": "old", "type": identity})
    http = client(tmp_path)
    full = http.get("/api/logs/events?task_id=old").json()["entries"]
    rows = preview(http, task_id="old")
    assert [row["type"] for row in full] == [row["type"] for row in rows] == [expected]
    assert read(http, rows[0], download=1).content == sources[expected]
    assert not (tmp_path / "task_drives/absent").exists()


def test_parent_scope_uses_effective_child_retry_identifier(tmp_path):
    child = tmp_path / "task_drives/retry"
    seed_result(tmp_path, "parent", {"status": "completed"})
    lineage = {"parent_task_id": "parent", "root_task_id": "parent", "delegation_role": "subagent"}
    seed_result(tmp_path, "old", {**lineage, "status": "failed", "retry_task_id": "retry"})
    seed_result(tmp_path, "retry", {**lineage, "status": "completed", "child_drive_root": str(child)})
    seed_result(child, "retry", {**lineage, "status": "completed"})
    raw = write(child, {"task_id": "retry", "type": "effective_child"})
    http = client(tmp_path)
    full = http.get("/api/logs/events?task_id=parent").json()["entries"]
    rows = preview(http, task_id="parent")
    assert [row["type"] for row in full] == [row["type"] for row in rows] == ["effective_child"]
    assert read(http, rows[0], download=1).content == raw
    assert read(http, edited_ref(rows[0], task_id="unrelated")).status_code == 404


@pytest.mark.parametrize("layout", ["task_drives/old", "state/headless_tasks/old/data"])
@pytest.mark.parametrize("alias", [False, True], ids=["physical_root", "configured_alias"])
def test_retry_reuses_recorded_predecessor_drive_without_source_writes(tmp_path, layout, alias):
    root = tmp_path / "physical"
    root.mkdir()
    configured = tmp_path / "configured" if alias else root
    if alias:
        configured.symlink_to(root, target_is_directory=True)
    child = configured / layout
    seed_result(root, "old", {"status": "failed", "superseded_by": "retry"})
    seed_result(root, "retry", {"status": "completed", "root_task_id": "old", "child_drive_root": str(child)})
    seed_result(child, "retry", {"status": "completed", "root_task_id": "old"})
    stored = write(child, {"task_id": "retry", "root_task_id": "old", "type": "retry_event",
                          "swarm_fanouts": [{"private_plane": "withheld"}], "body": "x" * 20000})
    http = client(configured)
    full = http.get("/api/logs/events?task_id=old").json()["entries"][0]
    full.pop("_source_root")
    full.pop("_line")
    before = tree_bytes(root, sqlite_read_marks=True)
    row = preview(http, task_id="old")[0]
    assert row["_raw"]["source"]["placement"] == layout
    response = read(http, row, download=1)
    assert response.status_code == 200 and json.loads(response.content) == full
    assert response.content != stored and b"withheld" not in response.content
    assert response.headers["x-ouroboros-raw-representation"] == "public_task_result"
    assert read(http, row, limit=64).json()["text"].encode() == response.content[:64]
    assert tree_bytes(root, sqlite_read_marks=True) == before, "retry discovery/Raw reads publish no source, archive or store changes"


@pytest.mark.parametrize("loss", ["link", "canonical_record", "placement", "occupant", "drive_cleanup"])
def test_retry_raw_rechecks_admission_and_cleanup_after_preview(tmp_path, loss):
    import shutil

    from ouroboros.store import close_connections_under
    from tests._store_shared import delete_result

    child = tmp_path / "task_drives/old"
    seed_result(tmp_path, "old", {"status": "failed", "retry_task_id": "retry"})
    record = {"status": "completed", "root_task_id": "old", "child_drive_root": str(child)}
    seed_result(tmp_path, "retry", record)
    seed_result(child, "retry", {"status": "completed"})
    write(child, {"task_id": "retry", "root_task_id": "old", "type": "retry_event"})
    http = client(tmp_path)
    row = preview(http, task_id="old")[0]
    if loss == "link":
        seed_result(tmp_path, "old", {"status": "failed"})
    elif loss == "canonical_record":
        delete_result(tmp_path, "retry")
    elif loss == "placement":
        seed_result(tmp_path, "retry", {**record, "child_drive_root": ""})
    elif loss == "occupant":
        delete_result(child, "retry")
        seed_result(child, "another", {"status": "completed"})
    else:
        close_connections_under(child)
        shutil.rmtree(child)
    before = tree_bytes(tmp_path, sqlite_read_marks=True)
    refused = read(http, row, download=1)
    assert refused.status_code == 404 and refused.json()["reason_code"] == "log_source_unavailable"
    assert "text" not in refused.json()
    assert preview(http, task_id="old") == []
    assert tree_bytes(tmp_path, sqlite_read_marks=True) == before


@pytest.mark.parametrize("placement", ["arbitrary", "symlink", "unrecorded", "wrong_owner"])
@pytest.mark.parametrize("view", ["full", "compact"])
def test_retry_child_metadata_requires_canonical_placement_admission(tmp_path, monkeypatch, placement, view):
    from ouroboros import task_store

    root = tmp_path / "canonical"
    child = root / ("not_a_child_drive" if placement == "arbitrary" else "task_drives/old")
    if placement == "symlink":
        external = tmp_path / "external"
        external.mkdir()
        child.parent.mkdir(parents=True)
        child.symlink_to(external, target_is_directory=True)
    seed_result(root, "old", {"status": "failed", "superseded_by": "retry"})
    seed_result(root, "retry", {"status": "completed", "root_task_id": "old",
        "child_drive_root": "" if placement == "unrecorded" else str(child)})
    seed_result(child, "someone_else" if placement == "wrong_owner" else "retry", {"status": "completed"})
    event = {"task_id": "retry", "root_task_id": "old", "type": "retry_event"}
    write(child, event)
    write(root, event)
    http = client(root)
    selector = edited_ref(preview(http)[0], task_id="old", placement=child.relative_to(root).as_posix())
    (root / "logs/events.jsonl").unlink()
    touched = []
    # Observe both current-fact and display-view loaders: an assertion thrown
    # inside the custody helper would be caught as an unavailable source.
    for name in ("read_document", "display_documents"):
        original = getattr(task_store, name)

        def watched(source, *args, _original=original, **kwargs):
            touched.append(source)
            return _original(source, *args, **kwargs)

        monkeypatch.setattr(task_store, name, watched)
    before = tree_bytes(root, sqlite_read_marks=True)
    response = http.get("/api/logs/events", params={"task_id": "old", "view": view})
    assert response.status_code == 200 and response.json()["entries"] == []
    assert read(http, selector).status_code == 404
    assert tree_bytes(root, sqlite_read_marks=True) == before
    assert touched
    if placement == "wrong_owner":
        assert child in touched, "a canonically admitted reused drive must still prove its occupant"
    else:
        assert all(source == root for source in touched), "no unadmitted child metadata may be opened"


def test_child_symlink_is_not_an_admitted_drive(tmp_path):
    external = tmp_path / "external"
    write(external, {"task_id": "child", "type": "no"})
    child = tmp_path / "task_drives/child"
    child.parent.mkdir()
    child.symlink_to(external, target_is_directory=True)
    seed_result(tmp_path, "child", {"child_drive_root": str(child), "status": "running"})
    assert preview(client(tmp_path), task_id="child") == []


def test_configured_root_alias_keeps_its_recorded_child_placement(tmp_path):
    root = tmp_path / "physical"
    root.mkdir()
    alias = tmp_path / "configured"
    alias.symlink_to(root, target_is_directory=True)
    child = alias / "task_drives/child"
    seed_result(root, "child", {"child_drive_root": str(child), "status": "completed"})
    raw = write(child, {"task_id": "child", "type": "through_configured_root"})
    http = client(alias)
    row = preview(http, task_id="child")[0]
    assert row["_raw"]["source"]["placement"] == "task_drives/child"
    assert read(http, row, download=1).content == raw


def test_child_metadata_is_not_read_before_placement_admission(tmp_path, monkeypatch):
    from ouroboros import task_status

    outside = tmp_path / "not_a_child_drive"
    seed_result(tmp_path, "parent", {"status": "completed"})
    seed_result(tmp_path, "child", {"parent_task_id": "parent", "root_task_id": "parent",
        "delegation_role": "subagent", "child_drive_root": str(outside), "status": "completed"})
    write(outside, {"task_id": "child", "body": "not admitted"})
    original = task_status.load_task_result

    def load(root, *args, **kwargs):
        assert root != outside, "placement must be admitted before even metadata is read"
        return original(root, *args, **kwargs)

    monkeypatch.setattr(task_status, "load_task_result", load)
    assert preview(client(tmp_path), task_id="parent") == []


def test_replica_filter_is_applied_before_compaction(tmp_path):
    child = tmp_path / "task_drives/child"
    seed_result(tmp_path, "parent", {"status": "completed"})
    seed_result(tmp_path, "child", {"parent_task_id": "parent", "root_task_id": "parent",
        "delegation_role": "subagent", "child_drive_root": str(child), "status": "completed"})
    row = {"ts": "2026-11-04Z", "task_id": "child", "type": "llm_api_error", "llm_call_id": "same",
           "error": "x" * 20000}
    write(tmp_path, row)
    write(child, row)
    http = client(tmp_path)
    assert len(preview(http, task_id="parent")) == 1
    assert len(http.get("/api/logs/events?task_id=parent").json()["entries"]) == 1


@pytest.mark.parametrize("stream", ["chat", "progress"])
def test_dialogue_compact_request_keeps_existing_projection_and_positions(tmp_path, stream):
    from ouroboros import dialogue_import

    write(tmp_path, {"ts": "2026-11-04Z", "chat_id": 1, "content": "retained dialogue"}, stream)
    assert dialogue_import.run_dialogue_import(tmp_path)["state"] == "done"
    http = client(tmp_path)
    default = http.get(f"/api/logs/{stream}").json()["entries"]
    assert preview(http, stream) == default
    assert http.get(f"/api/logs/{stream}/raw?ref=anything").status_code == 404


def test_window_line_number_is_not_used_as_source_identity(tmp_path):
    first = write(tmp_path, {"type": "a", "pad": "x" * 600000})
    second = write(tmp_path, {"type": "b"}, append=True)
    http = client(tmp_path)
    narrow = preview(http, limit=1)[0]
    broad = preview(http, limit=100)[-1]
    assert narrow["_line"] != broad["_line"]
    assert narrow["_raw"]["ref"] == broad["_raw"]["ref"]
    assert broad["_raw"]["source"]["start"] == len(first)
    assert read(http, narrow, download=1).content == second


def test_changed_earlier_segment_does_not_silently_retarget_chain_address(tmp_path):
    first = write(tmp_path, {"type": "before"})
    (tmp_path / "archive").mkdir()
    old = tmp_path / "archive/events_20261104.jsonl"
    (tmp_path / "logs/events.jsonl").rename(old)
    write(tmp_path, {"type": "wanted"})
    row = preview(client(tmp_path))[-1]
    assert row["_raw"]["source"]["start"] == len(first)
    old.unlink()
    assert read(client(tmp_path), row).status_code in {404, 409}


@pytest.mark.parametrize("scale", [20, 2000])
def test_hot_tail_growth_reads_a_fixed_window_and_reuses_segment_metadata(tmp_path, monkeypatch, scale):
    """Archive count and live row count grow together, 100x; hot IO stays bounded."""
    from ouroboros.confined_files import ConfinedDirectory

    archives = tmp_path / "archive"
    archives.mkdir()
    archive_raw = b'{"type":"archived"}\n'
    for index in range(scale):
        (archives / f"events_{index:06d}.jsonl").write_bytes(archive_raw)
    live = tmp_path / "logs/events.jsonl"
    live.parent.mkdir()
    live_row = (json.dumps({"type": "live", "body": "x" * 32768}) + "\n").encode()
    with live.open("wb") as handle:
        for _ in range(scale):
            handle.write(live_row)
    counts = {"opens": 0, "listings": 0, "bytes": 0}
    original_open, original_entries, original_read = ConfinedDirectory.open, ConfinedDirectory.entries, JsonlChainSnapshot._read

    def opened(directory, name):
        counts["opens"] += 1
        return original_open(directory, name)

    def listed(directory, **kwargs):
        counts["listings"] += 1
        return original_entries(directory, **kwargs)

    def read_bytes(snapshot, start, end):
        counts["bytes"] += end - start
        return original_read(snapshot, start, end)

    monkeypatch.setattr(ConfinedDirectory, "open", opened)
    monkeypatch.setattr(ConfinedDirectory, "entries", listed)
    monkeypatch.setattr(JsonlChainSnapshot, "_read", read_bytes)
    http = client(tmp_path)
    cold = preview(http, limit=1)
    assert counts["listings"] == 1 and counts["opens"] >= scale
    assert counts["bytes"] <= 512 * 1024 + 1, "cold capture stats old archives, never parses their bodies"
    cold_counts = dict(counts)
    JsonlChainSnapshot(live)  # another consumer's ordinary capture must not evict the confined metadata
    counts.update(opens=0, listings=0, bytes=0)
    hot = preview(http, limit=1)
    assert hot == cold
    assert counts["listings"] == 0 and counts["opens"] <= 6
    assert counts["bytes"] <= 512 * 1024 + 1
    hot_counts = dict(counts)
    write(tmp_path, {"type": "sentinel"}, append=True)
    counts.update(opens=0, listings=0, bytes=0)
    assert preview(http, limit=1)[0]["type"] == "sentinel"
    assert counts["listings"] == 0 and counts["opens"] <= 6 and counts["bytes"] <= 512 * 1024 + 1
    print(json.dumps({"growth_scale": scale, "cold": cold_counts, "hot": hot_counts, "after_append": counts}))


def test_published_contract_includes_raw_route_and_distinct_source_identity(tmp_path):
    from ouroboros.gateway.logs import LogRawResponse
    from ouroboros.gateway.router import collect_routes
    from ouroboros.gateway.schema import contract_document, json_schema_for

    assert any(route.path == "/api/logs/{name}/raw" for route in collect_routes(data_dir=tmp_path))
    assert "LogRawResponse" in json.dumps(contract_document())
    schema = json_schema_for(LogRawResponse)
    assert schema["properties"]["source"]["properties"]["kind"] == {"enum": ["log_row"]}
    assert "sha256" in schema["properties"] and "sha256" in schema["properties"]["source"]["properties"]


@pytest.fixture
def tcp_logs(tmp_path):
    """Actual route table on an owned ephemeral port; always stop the server."""
    import socket
    import threading
    import time

    import httpx
    import uvicorn

    from ouroboros.gateway.router import collect_routes

    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, lifespan="off", loop="asyncio", log_level="warning"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started
            with httpx.Client(base_url=f"http://127.0.0.1:{sock.getsockname()[1]}", timeout=10) as http:
                yield http
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive(), "the owned gateway server must stop"


def test_real_tcp_gateway_large_raw_download_after_rotation(tmp_path, tcp_logs):
    """Actual route table and TCP, with a streaming client; no installed server or WS claim."""
    http = tcp_logs
    raw = write(tmp_path, {"type": "task_checkpoint", "context_view": {"diagnostic": "x" * 13_677_952}})
    compact = http.get("/api/logs/events?view=compact")
    assert compact.status_code == 200 and len(compact.content) < 3000
    row = compact.json()["entries"][0]
    (tmp_path / "archive").mkdir()
    (tmp_path / "logs/events.jsonl").rename(tmp_path / "archive/events_20261104.jsonl")
    write(tmp_path, {"type": "sentinel"})
    page = read(http, row).json()
    assert page["has_more"] and page["next_offset"] == 65536
    digest, length = hashlib.sha256(), 0
    with http.stream("GET", "/api/logs/events/raw", params={"ref": row["_raw"]["ref"], "download": 1}) as download:
        assert download.status_code == 200
        for chunk in download.iter_bytes(chunk_size=65536):
            digest.update(chunk)
            length += len(chunk)
    assert digest.hexdigest() == hashlib.sha256(raw).hexdigest() and length == len(raw)
    full = http.get("/api/logs/events").json()["entries"]
    assert full[0]["context_view"]["diagnostic"] == "x" * 13_677_952
    assert full[-1]["type"] == "sentinel"
    print(json.dumps({"transport": "TCP", "source_bytes": length, "compact_bytes": len(compact.content),
                      "page_bytes": page["next_offset"], "sha256": digest.hexdigest()}))


@pytest.mark.parametrize("layout", ["task_drives/retry", "state/headless_tasks/retry/data"])
@pytest.mark.parametrize("field", ["child_drive_root", "headless_child_drive_root", "drive_root"])
@pytest.mark.parametrize("nested", [False, True], ids=["row", "metadata"])
def test_real_tcp_cli_keeps_full_rows_for_all_supported_placements(tmp_path, tcp_logs, capsys, layout, field, nested):
    from ouroboros import cli

    child = tmp_path / layout
    placement = {field: str(child)}
    seed_result(tmp_path, "old", {"status": "failed", "superseded_by": "retry"})
    seed_result(tmp_path, "retry", {"status": "completed", **({"metadata": placement} if nested else placement)})
    raw = write(child, {"task_id": "retry", "root_task_id": "old", "type": "retry", "body": "аб🦊" * 5000})
    full = tcp_logs.get("/api/logs/events?task_id=old").json()
    compact = tcp_logs.get("/api/logs/events?view=compact&task_id=old").json()
    args = ["--url", str(tcp_logs.base_url), "logs", "tail", "events", "--task-id", "old"]
    assert cli.main([*args, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == full
    assert set(full) == {"name", "entries"} and len(full["entries"]) == 1
    entry = full["entries"][0]
    assert entry["body"] == "аб🦊" * 5000 and "_raw" not in entry
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out) == entry
    assert read(tcp_logs, compact["entries"][0], download=1).content == raw
