"""Typed sources, bounded reads, exact artifacts and real registry delivery."""
from __future__ import annotations

import errno
import gzip
import hashlib
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest

from ouroboros import artifacts, observability, store, store_import, tool_capabilities as caps
from ouroboros.tools.query_data import _query_data, get_tools
from ouroboros.tools.registry import ToolContext, ToolRegistry


@pytest.fixture
def ctx(tmp_path):
    store_import.ensure_current_store(tmp_path, with_children=False)
    store.update_task(tmp_path, "caller", lambda _: {"status": "completed", "result": "first"})
    store.update_task(tmp_path, "second", lambda _: {"status": "completed", "result": "second"})
    repo = tmp_path / "repo"
    repo.mkdir()
    return ToolContext(repo_dir=repo, drive_root=tmp_path, task_id="caller")


def write(ctx, relative, raw):
    path = ctx.drive_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def call(ctx, **kwargs):
    text = _query_data(ctx, **kwargs)
    assert isinstance(text, str)
    return json.loads(text)


def saved(ctx, result):
    receipt = result["saved"]
    path = ctx.drive_root / "task_results" / "artifacts" / ctx.task_id / receipt["path"]
    raw = path.read_bytes()
    assert receipt["sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["size"] == len(raw)
    assert receipt["source"] == result["source"]
    return raw


def test_task_batches_use_store_and_canonical_projection(ctx):
    # A stale file must never override the public store document.
    write(ctx, "task_results/caller.json", b'{"result":"stale"}')
    result = call(ctx, op="json", kind="task_result", ids=["caller", "second"], select="result", save_as="batch.json")
    assert result["status"] == "ok" and result["data"] == {"caller": "first", "second": "second"}
    assert result["reconstructed"] is True
    assert saved(ctx, result) == b'{"caller":"first","second":"second"}'
    single = call(ctx, op="json", kind="task_result", ids=["caller"], save_as="whole.json")
    assert json.loads(saved(ctx, single))["result"] == "first"
    assert single["saved"]["reconstructed"] is True


def test_exact_whole_document_and_string_projection(ctx):
    raw = b'{ "z": 2, "primary_output": {"text":"line\\r\\n\\u03bb\\n"}, "a": 1 }\n'
    write(ctx, "state/review_ledger/review-one.json", raw)
    whole = call(ctx, op="json", kind="review_ledger", ids=["review-one"], save_as="record.json")
    assert saved(ctx, whole) == raw and not whole["reconstructed"]
    text = call(ctx, op="json", kind="review_ledger", ids=["review-one"], select="primary_output.text",
                format="text", save_as="output.txt")
    assert saved(ctx, text) == "line\r\nλ\n".encode("utf-8")
    projection = call(ctx, op="json", kind="review_ledger", ids=["review-one"], select="primary_output", save_as="projection.json")
    assert saved(ctx, projection) == '{"text":"line\\r\\nλ\\n"}'.encode("utf-8")
    assert call(ctx, op="json", kind="review_ledger", ids=["review-one"], format="text")["status"] == "error"
    collision = call(ctx, op="json", kind="review_ledger", ids=["review-one"], select="a", save_as="record.json")
    assert collision["status"] == "error" and "collision" in collision["message"]
    assert saved(ctx, whole) == raw


@pytest.mark.parametrize("same_bytes", [False, True])
@pytest.mark.parametrize("link_error", [None, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EPERM,
                                       errno.EXDEV, errno.ENOSYS, "win32"])
def test_concurrent_exports_publish_exclusively(ctx, monkeypatch, same_bytes, link_error):
    if link_error is not None:
        def unsupported(*_args, **_kwargs):
            exc = OSError(errno.EINVAL if link_error == "win32" else link_error, "fixture link unsupported")
            if link_error == "win32":
                exc.winerror = 50  # ERROR_NOT_SUPPORTED
            raise exc
        monkeypatch.setattr(artifacts.os, "link", unsupported)
    payloads = [b'{"value":"first"}', b'{"value":"first"}' if same_bytes else b'{"value":"second"}']
    for index, raw in enumerate(payloads):
        write(ctx, f"state/review_ledger/source-{index}.json", raw)
    ready, first_done, worker = threading.Barrier(2), threading.Event(), threading.local()
    original_write = artifacts.write_bytes_atomic

    def publish(path, data):
        ready.wait(timeout=10)  # Both writers reach publication before either publishes.
        if worker.index == 1:
            assert first_done.wait(timeout=10)
        return original_write(path, data)

    def export(index):
        worker.index = index
        try:
            result = call(ctx, op="json", kind="review_ledger", ids=[f"source-{index}"], save_as="shared.json")
            if index == 0:
                assert result["status"] == "ok", result
                assert saved(ctx, result) == payloads[0]
            return result
        finally:
            if index == 0:
                first_done.set()

    monkeypatch.setattr(artifacts, "write_bytes_atomic", publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(export, range(2)))
    if same_bytes:
        assert second["status"] == "ok", second
        assert saved(ctx, second) == payloads[0]
    else:
        assert second["status"] == "error" and "collision" in second["message"], second
    assert saved(ctx, first) == payloads[0]
    artifact_dir = artifacts.task_artifact_dir_path(ctx.drive_root, ctx.task_id)
    assert not list(artifact_dir.glob(".*.tmp"))


def test_export_link_io_failure_preserves_absence(ctx, monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError(errno.EIO, "fixture link IO failure")
    monkeypatch.setattr(artifacts.os, "link", broken)
    with pytest.raises(OSError, match="fixture link IO failure"):
        artifacts.store_task_artifact_bytes(ctx.drive_root, ctx.task_id, "io.json", b"{}")
    artifact_dir = artifacts.task_artifact_dir_path(ctx.drive_root, ctx.task_id)
    assert not (artifact_dir / "io.json").exists()
    assert not list(artifact_dir.glob(".*.tmp"))


def test_artifact_publication_failure_removes_temporary(ctx, monkeypatch):
    def fail_link(*args):
        raise OSError("publication failed")

    monkeypatch.setattr(artifacts.os, "link", fail_link)
    with pytest.raises(OSError, match="publication failed"):
        artifacts.store_task_artifact_bytes(ctx.drive_root, ctx.task_id, "failed.json", b"{}")
    artifact_dir = artifacts.task_artifact_dir_path(ctx.drive_root, ctx.task_id)
    assert not (artifact_dir / "failed.json").exists()
    assert not list(artifact_dir.glob(".*.tmp"))


@pytest.mark.parametrize("kind,path", [("queue_snapshot", "state/queue_snapshot.json"),
                                      ("queue_dispatch", "state/queue_dispatch.json"),
                                      ("advisory_review", "state/advisory_review.json")])
def test_fixed_names_and_selectors(ctx, kind, path):
    write(ctx, path, b'{"rows":[{"items":[1,2]},{"items":[3,4]}]}')
    value = call(ctx, op="json", kind=kind, select="rows[*].items[1]")
    assert value["data"] == [2, 4]
    for selector in ("rows..items", "rows[-1]", "rows[0]items", "rows[8]", "rows[*].missing"):
        assert call(ctx, op="json", kind=kind, select=selector)["status"] == "error"


@pytest.mark.parametrize("name", ["supervisor", "events", "tools", "chat"])
def test_jsonl_filter_tail_and_exact_rows(ctx, name):
    lines = [b'{ "type": "read", "n": 1 }\r\n', b'{"type":"write","n":2}\n', b'{"type":"read","n":3}\n']
    write(ctx, f"logs/{name}.jsonl", b"".join(lines))
    result = call(ctx, op="jsonl", kind=name + "_log", filter={"field": "type", "op": "eq", "value": "read"}, tail=1, save_as="row.jsonl")
    assert result["data"] == [{"type": "read", "n": 3}]
    assert saved(ctx, result) == lines[-1]
    contains = call(ctx, op="jsonl", kind=name + "_log", filter={"field": "type", "op": "contains", "value": "rea"}, select="[*].n")
    assert contains["data"] == [1, 3]
    assert call(ctx, op="jsonl", kind=name + "_log", filter={"field": "missing", "op": "eq", "value": 1})["data"] == []
    assert call(ctx, op="jsonl", kind=name + "_log", filter={"bad": 1})["status"] == "error"


def test_whole_jsonl_blank_lines_and_receipts(ctx):
    raw = b'\n {"pass":true}\n\n'
    write(ctx, "task_results/artifacts/caller/verification_receipts.jsonl", raw)
    result = call(ctx, op="jsonl", kind="verification_receipts", tail=0, save_as="receipt.jsonl")
    assert saved(ctx, result) == raw
    assert result["data"] == [{"pass": True}]


def refused(result, *fragments):
    """A pre-existing typed refusal whose message carries each fragment."""
    assert (result["status"], result["code"]) == ("error", "query_failed"), result
    for fragment in fragments:
        assert fragment in result["message"], (fragment, result["message"])
    return result["message"]


def test_stream_and_document_byte_limits(ctx, monkeypatch):
    """#1801: the fixed input budget is spent before select and max_bytes apply; the output
    budget is the caller's max_bytes. Each refusal names its budget, the measured size where it
    is known (a bounded read proves only a lower bound) and a repair valid for that stage."""
    from ouroboros import query_data_sql
    monkeypatch.setattr(query_data_sql, "MAX_BYTES", 512)
    write(ctx, "logs/tools.jsonl", b'{"n":1}\n' * 200)
    tail = call(ctx, op="jsonl", kind="tools_log", tail=2, max_bytes=512)
    assert tail["data"] == [{"n": 1}, {"n": 1}] and tail["source"]["scan_truncated"]
    scan = "tools_log scan (tail=0) is larger than the fixed 512-byte input budget"
    repair = "a nonzero tail scans only the newest bytes and reports scan_truncated"
    message = refused(call(ctx, op="jsonl", kind="tools_log", tail=0, max_bytes=512), scan, repair)
    assert "read at least 513 bytes" in message
    # A filter still reads every row, so it is not offered as a repair for the scan.
    refused(call(ctx, op="jsonl", kind="tools_log", tail=0, max_bytes=512,
                 filter={"field": "n", "op": "eq", "value": 2}), scan)
    assert "filter" not in message and "max_bytes" not in message
    write(ctx, "state/queue_snapshot.json", json.dumps({"large": "x" * 600}).encode("utf-8"))
    for narrowed in ({}, {"select": "large"}):  # select applies after the read and cannot admit it
        refused(call(ctx, op="json", kind="queue_snapshot", max_bytes=512, **narrowed),
                "queue_snapshot is larger than the fixed 512-byte input budget",
                "select and max_bytes apply after this read and cannot admit it")
    write(ctx, "logs/tools.jsonl", b'{"value":"' + b"x" * 600 + b'"}\n')
    refused(call(ctx, op="jsonl", kind="tools_log", tail=0, max_bytes=512), scan)
    raw = json.dumps({"large": "x" * 300}).encode("utf-8")
    write(ctx, "state/queue_snapshot.json", raw)
    output = refused(call(ctx, op="json", kind="queue_snapshot", max_bytes=256),
                     f"selected result is {len(raw)} bytes, over max_bytes=256",
                     f"raise max_bytes to at least {len(raw)} (at most 512), or narrow it with select")
    assert "input budget" not in output
    assert call(ctx, op="json", kind="queue_snapshot", max_bytes=len(raw))["status"] == "ok"
    # Canonical JSON can outgrow the bytes read: 1e5 is reprojected as 100000.0. Past the
    # ceiling no max_bytes admits it, and only the controls of this operation are offered.
    write(ctx, "logs/tools.jsonl", b'{"n":1e5}\n' * 40)
    rows = call(ctx, op="jsonl", kind="tools_log", tail=0, max_bytes=512)
    refused(rows, "selected result is 400 bytes (601 as canonical JSON), over max_bytes=512",
            "no max_bytes admits it (at most 512), or narrow it with select, filter or tail")
    assert len(call(ctx, op="jsonl", kind="tools_log", tail=30, max_bytes=512)["data"]) == 30


def test_positive_tail_input_refusal_after_controlled_growth(ctx, monkeypatch):
    from pathlib import Path

    from ouroboros import query_data_sql

    monkeypatch.setattr(query_data_sql, "MAX_BYTES", 512)
    path = write(ctx, "logs/tools.jsonl", b'{"n":1}\n')
    original_stat = Path.stat
    original_open = Path.open
    observed = []
    reading = False

    def open_for_scan(self, mode="r", *args, **kwargs):
        nonlocal reading
        stream = original_open(self, mode, *args, **kwargs)
        if self == path and mode == "rb":
            reading = True
        return stream

    def grow_after_stat(self, *args, **kwargs):
        stat = original_stat(self, *args, **kwargs)
        if self == path and reading and not observed:
            observed.append(stat.st_size)
            with path.open("ab") as stream:
                stream.write(b'{"n":2}\n' * 200)
        return stat

    with monkeypatch.context() as growing:
        growing.setattr(Path, "open", open_for_scan)
        growing.setattr(Path, "stat", grow_after_stat)
        result = call(ctx, op="jsonl", kind="tools_log", tail=2, max_bytes=512)
    print(result)
    assert observed == [8]  # No seek: the size check saw an input below the cap.
    message = refused(result, "tools_log scan (tail=2)", "at least 513 bytes",
                      "fixed 512-byte input budget", "retry")
    assert "tail=0" not in message and "a nonzero tail" not in message
    assert "max_bytes" not in message and "filter" not in message
    # With the append finished, the same request selects a bounded newest window.
    retried = call(ctx, op="jsonl", kind="tools_log", tail=2, max_bytes=512)
    assert retried["data"] == [{"n": 2}, {"n": 2}] and retried["source"]["scan_truncated"]


def test_ids_share_one_input_budget_and_a_single_document_is_bound_by_the_fixed_one(ctx, monkeypatch):
    from ouroboros import query_data_sql

    store.update_task(ctx.drive_root, "caller", lambda _: {"status": "completed", "result": "a" * 300})
    store.update_task(ctx.drive_root, "second", lambda _: {"status": "completed", "result": "b" * 300})
    with store.open_store(ctx.drive_root, write=False) as txn:
        stored = dict(txn.conn.execute("SELECT task_id, length(CAST(body AS BLOB)) FROM task_bodies").fetchall())
    first = len(json.dumps(call(ctx, op="json", kind="task_result", ids=["caller"])["data"], ensure_ascii=False,
                           sort_keys=True, separators=(",", ":")).encode("utf-8"))
    budget = max(stored["caller"], stored["second"], first) + 10
    assert first + stored["second"] > budget  # each fits alone, stored and reconstructed; together they do not
    monkeypatch.setattr(query_data_sql, "MAX_BYTES", budget)
    refused(call(ctx, op="json", kind="task_result", ids=["caller", "second"], max_bytes=budget),
            f"task_result second as stored is {stored['second']} bytes, over the {budget - first} bytes "
            f"earlier ids left of the {budget}-byte input budget; query fewer ids per call")
    assert call(ctx, op="json", kind="task_result", ids=["second"], select="result", max_bytes=budget)["data"] == "b" * 300
    monkeypatch.setattr(query_data_sql, "MAX_BYTES", 256)
    for narrowed in ({}, {"select": "status"}):
        refused(call(ctx, op="json", kind="task_result", ids=["second"], max_bytes=256, **narrowed),
                f"task_result second as stored is {stored['second']} bytes, over the fixed 256-byte input budget",
                "select and max_bytes apply after this read and cannot admit it")


def test_text_output_counts_its_exact_bytes_and_their_canonical_json(ctx):
    text = "é" * 150  # 300 UTF-8 bytes; 302 as a canonical JSON string
    write(ctx, "state/queue_snapshot.json", json.dumps({"note": text}, ensure_ascii=False).encode("utf-8"))
    refused(call(ctx, op="json", kind="queue_snapshot", select="note", format="text", max_bytes=301),
            "selected result is 300 bytes (302 as canonical JSON), over max_bytes=301",
            "raise max_bytes to at least 302")
    result = call(ctx, op="json", kind="queue_snapshot", select="note", format="text", max_bytes=302, save_as="note.txt")
    assert result["status"] == "ok" and saved(ctx, result) == text.encode("utf-8")


def child(ctx, identity, parent="caller"):
    from ouroboros.task_custody import HEADLESS_TASKS_DIR
    store.update_task(ctx.drive_root, identity, lambda _: {"parent_task_id": parent, "root_task_id": "caller", "status": "completed"})
    root = ctx.drive_root / HEADLESS_TASKS_DIR / identity / "data"
    root.mkdir(parents=True)
    path = root / "task_drives" / identity / "delegated_runs" / "run-one.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"primary_output":{"text":"exact child output"}}', encoding="utf-8")
    return root


def test_owned_child_grandchild_and_nonowned_drive(ctx):
    owned = child(ctx, "child")
    child(ctx, "grandchild", parent="child")
    child(ctx, "stranger", parent="unrelated")
    for identity in ("child", "grandchild"):
        result = call(ctx, op="json", kind="run", child_task_id=identity, ids=["run-one"],
                      select="primary_output.text", format="text", save_as=identity + ".txt")
        assert result["status"] == "ok", result
        assert saved(ctx, result) == b"exact child output"
    assert not (owned / "task_results/artifacts/caller/child.txt").exists()
    blocked = call(ctx, op="json", kind="run", child_task_id="stranger", ids=["run-one"])
    assert blocked["status"] == "blocked" and blocked["code"] == "access_denied"
    # A child's shared root_task_id never grants a sibling's drive.
    child_ctx = ToolContext(repo_dir=ctx.repo_dir, drive_root=owned, task_id="child", budget_drive_root=str(ctx.drive_root))
    assert call(child_ctx, op="json", kind="run", child_task_id="stranger", ids=["run-one"])["status"] == "blocked"
    assert call(child_ctx, op="json", kind="task_result", ids=["caller"], select="result")["data"] == "first"


def test_shared_drive_run_uses_the_delegate_writer_location(ctx):
    """#1573: the writer retains a run's terminal JSON once in the staging task's canonical
    store and the run's custody row names it; a pre-retention run keeps its task-drive file."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_output import _stage_full_output
    from ouroboros.task_custody import HEADLESS_TASKS_DIR

    store.update_task(ctx.drive_root, "child", lambda _: {"parent_task_id": "caller", "status": "completed"})
    child_ctx = ToolContext(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root, task_id="child")
    raw = '{ "primary_output": {"text": "shared child output"} }\n'
    staged = _stage_full_output(child_ctx, "run-one", raw)
    retained = ctx.drive_root / "task_results" / "artifacts" / "child" / staged["source"]["path"]
    assert staged["abs_path"] == str(retained) and retained.read_bytes() == raw.encode("utf-8")
    for run_id, owner in (("run-one", "child"), ("run-legacy", "child"), ("run-other", "second")):
        assert custody.record_started(ctx.drive_root, custody.RunCustody(
            run_id=run_id, task_id=owner, route_id="r", ledger_root=str(ctx.drive_root)))
    assert custody.emit(ctx.drive_root, custody.OUTPUT_SPILLED, {
        "run_id": "run-one", "task_id": "child", "artifact": staged["source"]["path"], "source": staged["source"],
        "staged": True, "full_content": True, "sha256": staged["sha256"]})
    assert not (ctx.drive_root / HEADLESS_TASKS_DIR / "child").exists()
    result = call(ctx, op="json", kind="run", child_task_id="child", ids=["run-one"], save_as="shared-run.json")
    assert result["status"] == "ok", result
    assert saved(ctx, result) == raw.encode("utf-8")
    legacy = write(ctx, "task_drives/child/delegated_runs/run-legacy.json", b'{"legacy": true}\n')
    assert call(ctx, op="json", kind="run", child_task_id="child", ids=["run-legacy"])["data"] == {"legacy": True}
    # Another task's retained run is not the selected child's: no path reaches it here.
    other = _stage_full_output(ToolContext(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root, task_id="second"),
                               "run-other", raw)
    assert custody.emit(ctx.drive_root, custody.OUTPUT_SPILLED, {
        "run_id": "run-other", "task_id": "second", "artifact": other["source"]["path"], "source": other["source"],
        "staged": True, "full_content": True, "sha256": other["sha256"]})
    assert call(ctx, op="json", kind="run", child_task_id="child", ids=["run-other"])["status"] != "ok"
    # Retained bytes that no longer match the custody ref are refused, never served.
    retained.write_bytes(raw.replace("shared", "SHARED").encode("utf-8"))
    assert call(ctx, op="json", kind="run", child_task_id="child", ids=["run-one"])["status"] != "ok"
    assert legacy.exists()


@pytest.mark.parametrize("identity", ["../caller", "/tmp/caller", "a/b", "a\\b", "..", "C:caller"])
def test_names_refuse_paths(ctx, identity):
    assert call(ctx, op="json", kind="task_result", ids=[identity])["status"] == "error"
    assert call(ctx, op="json", kind="task_result", ids=["caller"], save_as=identity)["status"] == "error"
    assert call(ctx, op="sql", db=identity, query="SELECT 1")["status"] == "error"


def test_symlink_escapes_and_secrets_but_skill_reads_work(ctx, tmp_path):
    from ouroboros.tools.core_secret_paths import is_runtime_secret_target
    state = write(ctx, "state/skills/example/review_job.json", b'{"status":"done"}')
    assert is_runtime_secret_target(state, ctx.repo_dir, data_root=ctx.drive_root, ctx=ctx)
    assert call(ctx, op="json", kind="skill_state", ids=["example"])["data"] == {"status": "done"}
    secret = write(ctx, "settings.json", b'{"value":"fake-key-for-tests"}')
    state.unlink()
    state.symlink_to(secret)
    assert call(ctx, op="json", kind="skill_state", ids=["example"])["status"] == "blocked"
    outside = tmp_path.parent / (tmp_path.name + "-outside.json")
    outside.write_text('{"ordinary":true}', encoding="utf-8")
    target = ctx.drive_root / "state" / "queue_snapshot.json"
    target.symlink_to(outside)
    assert call(ctx, op="json", kind="queue_snapshot")["status"] == "blocked"
    target.unlink()
    target.hardlink_to(secret)
    assert call(ctx, op="json", kind="queue_snapshot")["status"] == "blocked"


def test_blob_exact_bytes_projection_limits_and_packed_resolver(ctx, monkeypatch):
    raw = '{ "text": "λ\\r\\n", "z":2 }\n'.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    path = write(ctx, f"observability/blobs/{digest}.json.gz", gzip.compress(raw))
    result = call(ctx, op="blob", sha256=digest, save_as="blob.json")
    assert saved(ctx, result) == raw
    projected = call(ctx, op="blob", sha256=digest, select="text", format="text", save_as="blob.txt")
    assert saved(ctx, projected) == "λ\r\n".encode("utf-8")
    ref = observability.write_blob(ctx.drive_root, {"big": "x" * 10000})
    assert observability.read_blob_ref(ctx.drive_root, ref) == {"big": "x" * 10000}
    with pytest.raises(ValueError, match="byte limit"):
        observability.read_blob_ref(ctx.drive_root, ref, max_bytes=100)
    with pytest.raises(ValueError, match="size"):
        observability.read_blob_ref(ctx.drive_root, {key: value for key, value in ref.items() if key != "size"})
    path.write_bytes(gzip.compress(b'{"changed":true}'))
    assert call(ctx, op="blob", sha256=digest)["status"] == "error"
    path.unlink()
    seen = []

    def packed(root, ref, **kwargs):
        seen.append((root, ref, kwargs))
        return raw

    monkeypatch.setattr(observability, "read_blob_ref", packed)
    assert call(ctx, op="blob", sha256=digest)["data"]["z"] == 2
    assert seen[0][2]["raw_bytes"] and seen[0][2]["max_bytes"] > 0


def test_money_defaults_direct_and_missing_packs_are_typed(ctx, monkeypatch):
    from ouroboros import query_data_sql, usage_store
    usage_store.migrate_from_journal(ctx.drive_root)
    requests = []
    monkeypatch.setattr(query_data_sql, "run_sql", lambda *a, **kw: requests.append(kw) or {"status": "ok", "data": 1})
    for snapshot in (None, False, True):
        assert call(ctx, op="sql", db="money", query="SELECT 1", snapshot=snapshot)["status"] == "ok"
    assert [request["snapshot"] for request in requests] == [False, False, True]
    no_packs = call(ctx, op="sql", db="packs", query="SELECT 1")
    assert no_packs["status"] == "unavailable" and no_packs["code"] == "no_packs"
    assert not (ctx.drive_root / "observability/packs").exists()


@pytest.mark.parametrize("cache", ["valid", "missing", "old_version", "quick_check"])
def test_pack_index_sql_recovers_from_sidecars(ctx, cache):
    from ouroboros import evidence_packs as packs
    from tests.test_evidence_packs import _task

    raw = b'{"same": true}'
    _task(ctx.drive_root, "packed", blobs=1, payloads=[raw], handle=raw)
    pack_path, sidecar = packs.write_pack(ctx.drive_root, "packed", 1, packs.loose_members(ctx.drive_root, "packed"))
    packs.index_upsert(ctx.drive_root, pack_path, sidecar)
    path = packs.packs_dir(ctx.drive_root) / packs.INDEX_NAME
    authority = {p: p.read_bytes() for p in (pack_path, path.parent / "packed.members.json")}
    if cache == "missing":
        path.unlink()
    elif cache == "old_version":
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.executescript("""
                ALTER TABLE members RENAME TO previous_members;
                CREATE TABLE members (sha256 TEXT, task_id TEXT, generation INTEGER, offset INTEGER,
                    size INTEGER, kind TEXT, PRIMARY KEY (sha256, task_id, generation));
                INSERT OR REPLACE INTO members SELECT * FROM previous_members;
                DROP TABLE previous_members;
                UPDATE meta SET value='1' WHERE key='schema_version';
            """)
            assert conn.execute("SELECT count(*) FROM members").fetchone() == (1,)
    elif cache == "quick_check":
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.executescript("""
                CREATE TABLE damaged(value INTEGER CHECK(value=0));
                PRAGMA ignore_check_constraints=ON;
                INSERT INTO damaged VALUES (1);
                DELETE FROM members WHERE kind='checkpoint';
                PRAGMA ignore_check_constraints=OFF;
            """)
            assert conn.execute("PRAGMA quick_check").fetchall() != [("ok",)]
    # Cold cache recovery is the subject, not a one-second completion SLA.
    result = call(ctx, op="sql", db="packs", query="SELECT kind, size FROM members ORDER BY kind", timeout_sec=5)
    assert result["status"] == "ok", result
    assert result["data"]["rows"] == [["checkpoint", len(raw)], ["json", len(raw)]], result
    assert authority == {p: p.read_bytes() for p in authority}
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA quick_check").fetchall() == [("ok",)]
        assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() == (str(packs.INDEX_SCHEMA_VERSION),)


def test_no_packs_depends_on_sidecars_not_an_empty_cache(ctx):
    from ouroboros import evidence_packs as packs
    with packs.open_index(ctx.drive_root):
        pass
    result = call(ctx, op="sql", db="packs", query="SELECT count(*) FROM members")
    assert result["status"] == "unavailable" and result["code"] == "no_packs", result


@pytest.mark.parametrize("kwargs,message", [
    ({"select": "rows..value"}, "select/filter fields accept dotted keys"),
    ({"filter": {"field": "a..b", "op": "eq", "value": 1}}, "select/filter fields accept dotted keys"),
    ({"max_bytes": 0}, "max_bytes=0: use an integer from 256 to 8388608"),
    ({"max_bytes": "4096"}, 'max_bytes="4096": use an integer from 256 to 8388608'),
    ({"tail": -1}, "tail=-1: use an integer from 0 to 10000"),
    ({"max_rows": 2}, "max_rows=2 applies only to op=sql; omit it; tail bounds rows for op=jsonl"),
    ({"snapshot": False}, "snapshot=false applies only to op=sql; omit it"),
    ({"snapshot": True}, "snapshot=true applies only to op=sql; omit it"),
    ({"timeout_sec": 2}, "timeout_sec=2 applies only to op=sql; omit it"),
    ({"db": "money", "query": "SELECT 1"}, 'db="money", query="SELECT 1" apply only to op=sql; omit them'),
    ({"sha256": "a" * 64}, f'sha256="{"a" * 59}... applies only to op=blob; omit it'),
    ({"state_file": "health"}, 'state_file="health" applies only to kind=skill_state; omit it'),
])
def test_invalid_requests_are_rejected_before_reading(ctx, kwargs, message):
    assert call(ctx, op="jsonl", kind="events_log")["status"] == "unavailable"
    refused(call(ctx, op="jsonl", kind="events_log", **kwargs), message)


@pytest.mark.parametrize("kwargs,message", [
    ({"op": "sql", "query": "SELECT 1", "kind": "tools_log"}, 'kind="tools_log" applies only to op=json or op=jsonl'),
    ({"op": "sql", "query": "SELECT 1", "tail": 5}, "tail=5 applies only to op=jsonl; omit it; max_rows bounds rows for op=sql"),
    ({"op": "json", "kind": "queue_snapshot", "tail": 5}, "tail=5 applies only to op=jsonl; omit it"),
    ({"op": "json", "kind": "queue_snapshot", "filter": {"field": "a", "op": "eq", "value": 1}},
     'filter={"field": "a", "op": "eq", "value": 1} applies only to op=jsonl; omit it'),
    ({"op": "blob", "sha256": "0" * 64, "ids": ["caller"]}, 'ids=["caller"] applies only to op=json or op=jsonl'),
    ({"op": "json", "kind": "queue_snapshot", "blob_kind": "txt"}, 'blob_kind="txt" applies only to op=blob'),
    ({"op": "json"}, 'kind="": op=json needs one of task_result, review_ledger'),
    ({"op": "jsonl", "kind": "queue_snapshot"}, "kind=queue_snapshot requires op=json"),
    ({"op": "lines"}, 'op="lines": use sql, json, jsonl or blob'),
])
def test_another_operations_parameter_is_named_with_its_owner(ctx, kwargs, message):
    """#1801: applicability is answered before any read, with the operation's own alternative."""
    write(ctx, "state/queue_snapshot.json", b'{"a":1}')
    message_text = refused(call(ctx, **kwargs), message)
    if kwargs["op"] in {"json", "blob"}:
        assert "bounds rows" not in message_text  # these operations have no row control to offer


def test_a_default_valued_parameter_of_another_operation_takes_the_omitted_path(ctx):
    """Naming a documented default is never a different request (DEVELOPMENT 02)."""
    write(ctx, "logs/events.jsonl", b'{"n":1}\n')
    write(ctx, "state/queue_snapshot.json", b'{"a":1}')
    defaults = {"db": "current", "query": "", "max_rows": 100, "timeout_sec": 1, "sha256": "",
                "blob_kind": "json", "state_file": "review_job"}
    assert call(ctx, op="jsonl", kind="events_log", **defaults)["data"] == [{"n": 1}]
    assert call(ctx, op="json", kind="queue_snapshot", tail=100, timeout_sec=1.0)["data"] == {"a": 1}
    assert call(ctx, op="sql", query="SELECT 1 AS one", kind="", tail=100)["data"]["rows"] == [[1]]


def test_schema_states_where_each_limit_and_parameter_applies():
    properties = get_tools()[0].schema["parameters"]["properties"]
    max_bytes = properties["max_bytes"]["description"]
    assert "Output budget after select/filter/tail (canonical JSON counts)" in max_bytes
    assert "op=sql stops fetching rows at it (truncated=true) before select" in max_bytes
    assert "fixed 8 MiB input budget, shared by ids, is separate and precedes select" in max_bytes
    assert "8 MiB scan window shared by ids (256 MiB for knowledge_history)" in properties["tail"]["description"]
    assert properties["filter"]["description"] == "op=jsonl only; keeps whole rows before tail."
    assert properties["state_file"]["description"] == "kind=skill_state only."
    assert properties["blob_kind"]["description"] == "op=blob only."


def test_real_registry_light_children_and_typed_errors(ctx, monkeypatch):
    from ouroboros import config, safety
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "light")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    assert registry._module_load_omissions == []
    assert "query_data" in registry._FROZEN_TOOL_MODULES
    assert safety.TOOL_POLICY["query_data"] == safety.POLICY_SKIP
    for names in (caps.CORE_TOOL_NAMES, caps.LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
                  caps.ACTING_SUBAGENT_TOOL_NAMES, caps.READ_ONLY_PARALLEL_TOOLS, caps.PAGEABLE_TOOL_RESULTS):
        assert "query_data" in names
    for constraint in ({}, {"mode": "local_readonly_subagent"},
                       {"mode": "acting_subagent", "surface": "self_worktree", "root_task_id": "caller"}):
        ctx.task_constraint = constraint
        registry.set_context(ctx)
        assert registry.get_schema_by_name("query_data") is not None
        result = registry.execute_result("query_data", {"op": "json", "kind": "task_result", "ids": ["caller"]})
        assert result.status == "ok", result.text
        denied = registry.execute_result("query_data", {"op": "sql", "query": "DELETE FROM tasks"})
        assert denied.status == "error", denied.text
        # A schema-valid argument of another operation is refused typed, with this operation's repair.
        misplaced = registry.execute_result("query_data", {"op": "jsonl", "kind": "events_log", "max_rows": 20})
        assert (misplaced.status, misplaced.code) == ("error", "TOOL_ERROR"), misplaced.text
        assert json.loads(misplaced.text)["message"] == "max_rows=20 applies only to op=sql; omit it; tail bounds rows for op=jsonl"
    schema = get_tools()[0].schema["parameters"]
    assert "path" not in schema["properties"] and "root" not in schema["properties"]


@pytest.mark.parametrize("retained_store", [True, False])
def test_native_inspection_queries_stay_on_selected_data_plane(ctx, retained_store):
    from ouroboros.review_native_episode import inspection_registry
    from ouroboros.task_results import write_task_result
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.usage_accounting import UsageScope, usage_scope

    write_task_result(ctx.drive_root, "caller", "running", root_task_id="caller")
    retained = ctx.drive_root / "retained-review"
    retained.mkdir()
    if retained_store:
        store_import.ensure_current_store(retained, with_children=False)
        store.update_task(retained, "caller", lambda _: {"status": "failed", "result": "retained"})
    retained_ctx = ToolContext(repo_dir=ctx.repo_dir, drive_root=retained, task_id="caller")
    for relative in ("state/review_ledger/record.json", "state/skills/example/review_job.json"):
        write(ctx, relative, b'{"plane":"live"}')
        write(retained_ctx, relative, b'{"plane":"retained"}')
    with usage_scope(UsageScope(drive_root=ctx.drive_root, task_id="caller", root_task_id="caller")):
        registry, inspection, _ = inspection_registry(str(ctx.repo_dir), retained, "reviewer")
        assert inspection.drive_root == retained
        assert inspection.task_id == "caller" and inspection.task_lifecycle_bound
        assert canonical_data_root(inspection) == ctx.drive_root  # Budget/Pause keep their owner.
        task = registry.execute_result("query_data", {"op": "json", "kind": "task_result", "ids": ["caller"], "select": "result"})
        facts = registry.execute_result("query_data", {"op": "sql", "query": "SELECT status FROM tasks"})
        if retained_store:
            assert task.status == "ok" and json.loads(task.text)["data"] == "retained", task.text
            assert facts.status == "ok" and json.loads(facts.text)["data"]["rows"] == [["failed"]], facts.text
        else:
            for result in (task, facts):
                assert result.status == "unavailable", result.text
                assert json.loads(result.text)["code"] == "not_found"
            assert not store.store_path(retained).exists()
        file_read = registry.execute_result("read_file", {"root": "runtime_data", "path": "state/review_ledger/record.json"})
        assert file_read.status == "ok" and '"plane":"retained"' in file_read.text, file_read.text
        for kind, identity in (("review_ledger", "record"), ("skill_state", "example")):
            result = registry.execute_result("query_data", {"op": "json", "kind": kind, "ids": [identity]})
            assert result.status == "ok" and json.loads(result.text)["data"] == {"plane": "retained"}, result.text


def test_delivery_retains_the_complete_query_before_cut(ctx):
    from tests.test_tool_result_delivery import _deliver, _row, _source_ref, _view_json
    from tests._tool_result_delivery_shared import measured_fit
    write(ctx, "state/queue_snapshot.json", json.dumps({"text": "large data " * 12000}).encode("utf-8"))
    full = _query_data(ctx, op="json", kind="queue_snapshot", max_bytes=200000)
    assert json.loads(full)["status"] == "ok"
    messages, trace = _deliver(ctx, [_row("query", "query_data", full)], measured_fit(window=12000, reserve=2000))
    shown = messages[0]["content"]
    assert _view_json(shown)["complete_chars"] == len(full)
    ref = _source_ref(shown)
    assert artifacts.read_actor_source_bytes(ctx.drive_root, ctx.task_id, ref) == full.encode("utf-8")
    assert trace["tool_calls"][0]["result_partial"]


def test_direct_current_reads_stop_within_half_the_writers_wait(ctx):
    # A direct current-store read holds the lock task writes wait on; a longer query goes to a snapshot.
    from ouroboros import query_data_sql as sql

    refused = call(ctx, op="sql", query="SELECT 1", timeout_sec=sql.CURRENT_HOLD_SEC + 1)
    assert refused["status"] == "error" and "snapshot=true" in refused["message"]
    assert call(ctx, op="sql", query="SELECT 1", timeout_sec=sql.CURRENT_HOLD_SEC)["status"] == "ok"
    assert call(ctx, op="sql", query="SELECT 1", timeout_sec=sql.CURRENT_HOLD_SEC + 1, snapshot=True)["status"] == "ok"


def test_storage_kind_reads_the_saved_count_from_the_installation_root_and_refuses_a_child(ctx, monkeypatch):
    """``kind="storage"`` is installation-wide: the saved count comes from the canonical root
    through the existing admission; a child selector is refused before any read."""
    from ouroboros import storage

    (ctx.drive_root / "observability").mkdir()
    (ctx.drive_root / "observability" / "a.bin").write_bytes(b"x" * 4096)
    table = [{"id": "model_calls", "group": "records", "roots": [(str(ctx.drive_root), ("observability",))]}]
    record = storage.measure(ctx.drive_root, places=table, sleep=lambda _s: None)
    storage.save_count(ctx.drive_root, record)
    from ouroboros import maintenance_jobs
    from ouroboros.storage_removers import place_removers

    maintenance_jobs.record(ctx.drive_root, "runtime_retirement", state="done", last_error={"class": "OSError", "at": "T"})
    maintenance_jobs.record(ctx.drive_root, "record_compression", state="pending", totals={
        "evidence": {"compressed": 12, "saved_bytes": 4096}, "seats": {"compressed": 3, "saved_bytes": None}})
    result = call(ctx, op="json", kind="storage")
    assert result["status"] == "ok" and result["source"]["kind"] == "storage" and result["reconstructed"] is True
    assert result["data"]["latest"]["finished_at"] == record["finished_at"] and result["data"]["previous"] is None
    removers = result["data"]["removers"]
    assert removers == place_removers(ctx.drive_root) == storage.count.projection(ctx.drive_root)["removers"], \
        "the per-place facts GET /api/storage serves"
    assert removers["app_state"][0]["last_error"] == {"class": "OSError", "at": "T"}
    assert {place: [fact["compressed"] for fact in removers[place] if fact["id"] == "record_compression"]
            for place in ("task_results", "app_state")} == {"task_results": [12], "app_state": [3]}
    assert call(ctx, op="json", kind="storage", select="latest.places[*].id")["data"] == ["model_calls"]
    monkeypatch.setattr("ouroboros.tools.query_data._child_root", lambda *_a: pytest.fail("storage must not resolve a child"))
    refused = call(ctx, op="json", kind="storage", child_task_id="second")
    assert refused["status"] == "error" and "installation-wide" in refused["message"]
    assert call(ctx, op="json", kind="storage", ids=["x"])["status"] == "error"
    (ctx.drive_root / "state" / "storage_measurement.json").unlink()
    uncounted = call(ctx, op="json", kind="storage")
    assert uncounted["status"] == "ok" and set(uncounted["data"]) == {"removers"}, "the removers come before any count"


def test_knowledge_history_lists_writes_without_bodies_from_the_installation_root(ctx, monkeypatch):
    """``kind="knowledge_history"``: global knowledge writes, newest first-in-window, bodies left
    out; a fact an old row lacks stays absent; a child selector is refused; a window past the
    read budget is said as truncated."""
    from ouroboros.tools import query_data

    rows = [{"ts": "2026-10-20T00:00:00Z", "topic": "patterns", "mode": "overwrite", "writer": "turn",
             "old_content": "OLD_BODY", "new_content": "NEW_BODY", "new_chars": 8, "summary": "Classes."},
            {"ts": "2026-10-21T00:00:00Z", "topic": "lessons/tests", "mode": "edit", "publication": "source_capture",
             "old_content": "A", "new_content": "B", "new_chars": 1, "summary": "Exit codes."}]
    write(ctx, "memory/knowledge_history.jsonl", b"".join(json.dumps(r).encode() + b"\n" for r in rows))
    result = call(ctx, op="jsonl", kind="knowledge_history", tail=10)
    assert result["status"] == "ok" and result["source"]["kind"] == "knowledge_history"
    assert result["reconstructed"] is True and result["source"]["scan_truncated"] is False
    assert [r["topic"] for r in result["data"]] == ["patterns", "lessons/tests"]
    assert "OLD_BODY" not in json.dumps(result) and "NEW_BODY" not in json.dumps(result)
    assert "publication" not in result["data"][0] and result["data"][1]["publication"] == "source_capture"
    narrowed = call(ctx, op="jsonl", kind="knowledge_history", filter={"field": "topic", "op": "contains", "value": "lessons"})
    assert [r["summary"] for r in narrowed["data"]] == ["Exit codes."]
    assert call(ctx, op="jsonl", kind="knowledge_history", tail=1, select="[0].topic")["data"] == "lessons/tests"
    assert call(ctx, op="json", kind="knowledge_history")["status"] == "error"  # a log reads as JSONL
    monkeypatch.setattr("ouroboros.tools.query_data._child_root", lambda *_a: pytest.fail("history has no child"))
    refused = call(ctx, op="jsonl", kind="knowledge_history", child_task_id="second")
    assert refused["status"] == "error" and "global knowledge" in refused["message"]
    monkeypatch.setattr(query_data, "_HISTORY_SCAN_BYTES", 120)
    clipped = call(ctx, op="jsonl", kind="knowledge_history")
    assert clipped["status"] == "ok" and clipped["source"]["scan_truncated"] is True


def test_knowledge_history_input_budget_counts_rows_without_their_bodies(ctx, monkeypatch):
    """The 256 MiB scan window reads whole rows; what spends the input budget is the rows kept
    after bodies are left out, measured before select. Bodies far past the budget are no refusal."""
    from ouroboros import query_data_sql

    row = {"topic": "lessons/tests", "mode": "edit", "summary": "Exit codes.", "old_content": "B" * 4000,
           "new_content": "B" * 4000}
    write(ctx, "memory/knowledge_history.jsonl", b"".join(json.dumps(dict(row, n=n)).encode() + b"\n" for n in range(12)))
    monkeypatch.setattr(query_data_sql, "MAX_BYTES", 512)
    kept = call(ctx, op="jsonl", kind="knowledge_history", tail=5, max_bytes=512)
    assert kept["status"] == "ok" and [r["n"] for r in kept["data"]] == [7, 8, 9, 10, 11]
    stripped = len(json.dumps([{k: v for k, v in dict(row, n=n).items() if not k.endswith("_content")} for n in range(12)],
                              ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for narrowed in ({}, {"select": "[0].topic"}):  # select applies only after the budget check
        refused(call(ctx, op="jsonl", kind="knowledge_history", tail=0, max_bytes=512, **narrowed),
                f"knowledge_history without note bodies is {stripped} bytes, over the fixed 512-byte input budget",
                "keep fewer rows with tail or filter; select applies only after this check")
    narrowed = call(ctx, op="jsonl", kind="knowledge_history", tail=0, max_bytes=512,
                    filter={"field": "n", "op": "eq", "value": 3})
    assert [r["n"] for r in narrowed["data"]] == [3]
