"""Public child lookup retains full-scan semantics while avoiding warm body reads."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
from types import SimpleNamespace

import pytest

import ouroboros.task_results as task_results
import ouroboros.task_result_scan as task_result_scan
import ouroboros.task_status as task_status
from ouroboros.task_result_schema import TASK_RESULT_SCHEMA_VERSION


def _put(root: pathlib.Path, name: str, **fields) -> pathlib.Path:
    path = root / "task_results" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"_schema_version": TASK_RESULT_SCHEMA_VERSION, "task_id": name,
           "status": "completed", "ts": "2026-10-04T00:00:00Z", **fields}
    path.write_text(json.dumps(row), encoding="utf-8")
    return path


def _ids(rows):
    return [row["task_id"] for row in rows]


def _root_normalized_json(rows, root):
    # Replace the JSON representation, including escaped Windows separators.
    return json.dumps(rows).replace(json.dumps(str(root))[1:-1], "<root>")


@pytest.mark.parametrize("old,new", [
    ("/fixture/old", "/fixture/new"),
    (r"C:\fixture\old", r"C:\fixture\new"),
])
def test_root_comparison_preserves_nested_payload_and_windows_paths(old, new):
    def rows(root):
        return [{"metadata": {"child_drive_root": root + "/split"},
                 "artifacts": [{"path": root + "/recorded.txt"}],
                 "result": "full payload", "status": "completed"}]

    assert _root_normalized_json(rows(old), old) == _root_normalized_json(rows(new), new)
    changed = rows(new)
    changed[0]["result"] = "different payload"
    assert _root_normalized_json(rows(old), old) != _root_normalized_json(changed, new)


def _old_lookup(monkeypatch, root, **kwargs):
    """Exercise the same public function with its original full-list call."""
    materialize = kwargs.pop("materialize_artifacts", False)
    with monkeypatch.context() as patch:
        patch.setattr(task_status, "list_task_results",
                      lambda drive, **_ignored: task_results.list_task_results(drive))
        patch.setattr(task_status, "raw_result_facts", lambda _directory: ({}, []))
        return task_status.find_child_tasks(root, materialize_artifacts=materialize, **kwargs)


def _pair(tmp_path, monkeypatch, source, **kwargs):
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    shutil.copytree(source, old_root)
    shutil.copytree(source, new_root)
    old = _old_lookup(monkeypatch, old_root, **kwargs)
    materialize = kwargs.pop("materialize_artifacts", False)
    new = task_status.find_child_tasks(new_root, materialize_artifacts=materialize, **kwargs)
    assert old == new
    return old_root, new_root, new


@pytest.mark.parametrize("scope, expected", [
    ("direct", ["child", "retry"]),
    ("subtree", ["child", "grandchild", "retry"]),
])
def test_differential_lineage_retry_queue_and_refusals(tmp_path, monkeypatch, scope, expected):
    source = tmp_path / "source"
    _put(source, "child", delegation_role="subagent", parent_task_id="parent",
         root_task_id="parent", result="child full body")
    _put(source, "sibling", delegation_role="subagent", parent_task_id="other",
         root_task_id="other")
    _put(source, "grandchild", delegation_role="subagent", parent_task_id="child",
         root_task_id="parent")
    _put(source, "retry", delegation_role="subagent", parent_task_id="parent",
         root_task_id="parent", result="retry full body")
    _put(source, "old", delegation_role="subagent", parent_task_id="elsewhere",
         root_task_id="elsewhere", superseded_by="retry")
    _put(source, "unrelated", delegation_role="", result="x" * 200_000)
    _put(source, "refused", _schema_version=TASK_RESULT_SCHEMA_VERSION + 1)
    (source / "task_results" / "broken.json").write_text("{", encoding="utf-8")

    old_root, new_root = tmp_path / "old", tmp_path / "new"
    shutil.copytree(source, old_root)
    shutil.copytree(source, new_root)
    args = {"parent_task_id": "parent", "root_task_id": "parent", "scope": scope}
    old = _old_lookup(monkeypatch, old_root, **args)
    new = task_status.find_child_tasks(new_root, materialize_artifacts=False, **args)
    assert old == new
    assert _ids(new) == sorted(expected)  # fixture timestamps tie; ID breaks ties
    assert next(row for row in new if row["task_id"] == "retry")["result"] == "retry full body"
    assert "retry_lineage" not in next(row for row in new if row["task_id"] == "retry")
    assert (new_root / "task_results" / "quarantine" / "broken.json").exists()
    assert (new_root / "task_results" / "quarantine" / "refused.json").exists()
    events = [json.loads(line) for line in (new_root / "logs" / "events.jsonl").read_text().splitlines()]
    batches = [row for row in events if row.get("type") == "task_results_quarantined"]
    assert len(batches) == 1 and batches[0]["count"] == 2


def test_warm_public_lookup_reads_no_unrelated_large_bodies(tmp_path, monkeypatch):
    _put(tmp_path, "child", delegation_role="subagent", parent_task_id="parent",
         root_task_id="parent", result="selected")
    for i in range(12):
        _put(tmp_path, f"unrelated{i:02d}", delegation_role="subagent",
             parent_task_id="elsewhere", root_task_id="elsewhere", result="X" * 200_000)
    reads = []
    decodes = []
    real_read = pathlib.Path.read_text
    real_loads = json.loads

    def read_spy(path, *args, **kwargs):
        value = real_read(path, *args, **kwargs)
        if path.parent.name == "task_results" and path.suffix == ".json":
            reads.append((path.name, len(value.encode("utf-8"))))
        return value

    def loads_spy(value, *args, **kwargs):
        if len(value) > 100_000:
            decodes.append(len(value))
        return real_loads(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pathlib.Path, "read_text", read_spy)
        patch.setattr(json, "loads", loads_spy)
        assert _ids(task_status.find_child_tasks(tmp_path, parent_task_id="parent",
                                                  scope="direct", materialize_artifacts=False)) == ["child"]
        assert len(decodes) == 12
        reads.clear()
        decodes.clear()
        for _ in range(3):
            assert _ids(task_status.find_child_tasks(tmp_path, parent_task_id="parent",
                                                      scope="direct", materialize_artifacts=False)) == ["child"]
        assert not any(name.startswith("unrelated") for name, _size in reads)
        assert decodes == []
        assert sum(size for _name, size in reads) < 10_000


def test_changed_appearance_removal_and_root_isolation(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    _put(left, "same", delegation_role="subagent", parent_task_id="other", root_task_id="other")
    _put(right, "same", delegation_role="subagent", parent_task_id="parent", root_task_id="parent")

    def lookup(root):
        return _ids(task_status.find_child_tasks(root, parent_task_id="parent", scope="direct",
                                                  materialize_artifacts=False))

    assert lookup(left) == [] and lookup(right) == ["same"]
    _put(left, "same", delegation_role="subagent", parent_task_id="parent", root_task_id="parent")
    assert lookup(left) == ["same"]
    (left / "task_results" / "same.json").unlink()
    assert lookup(left) == [] and lookup(right) == ["same"]
    _put(left, "same", delegation_role="subagent", parent_task_id="parent", root_task_id="parent")
    assert lookup(left) == ["same"]
    _put(left, "same", delegation_role="subagent", parent_task_id="other", root_task_id="other")
    assert lookup(left) == []


def test_private_paths_empty_and_strict_always_full(tmp_path):
    _put(tmp_path, "good", delegation_role="subagent")
    assert task_results.list_task_results(tmp_path, _paths=[]) == []
    assert _ids(task_results.list_task_results(tmp_path, _paths=[], strict=True)) == ["good"]


@pytest.mark.parametrize("suffix", [".json", ".JSON", ".JsOn"])
@pytest.mark.parametrize("windows_glob", [False, True])
def test_mixed_case_names_follow_canonical_glob(tmp_path, monkeypatch, suffix, windows_glob):
    source = tmp_path / "source"
    path = _put(source, "child", delegation_role="subagent", parent_task_id="parent",
                status="completed", result="completed disk child")
    path.rename(path.with_suffix(suffix))
    if windows_glob:
        # Emulate only the Windows canonical filename selection on POSIX too.
        # Real Windows runs also exercise its native glob in the other branch.
        original_glob = pathlib.Path.glob

        def folded_glob(directory, pattern):
            if directory.name == "task_results" and pattern == "*.json":
                return (p for p in directory.iterdir() if p.suffix.lower() == ".json")
            return original_glob(directory, pattern)

        monkeypatch.setattr(pathlib.Path, "glob", folded_glob)
    _, _, rows = _pair(tmp_path, monkeypatch, source,
                       parent_task_id="parent", scope="direct")
    if windows_glob or suffix == ".json" or os.name == "nt":
        assert _ids(rows) == ["child"]
        assert rows[0]["result"] == "completed disk child"
        assert rows[0]["status"] == "completed"
    else:
        assert rows == []


def test_atomic_replacement_during_metadata_read_is_admitted_fresh(tmp_path, monkeypatch):
    path = _put(tmp_path, "changing", delegation_role="subagent",
                parent_task_id="other", root_task_id="other", result="old")
    real_reader = task_result_scan.read_json_dict
    replaced = []

    def reader(candidate):
        data = real_reader(candidate)
        if candidate == path and not replaced:
            replacement = path.with_suffix(".replacement")
            replacement.write_text(json.dumps({**data, "parent_task_id": "parent",
                                               "root_task_id": "parent", "result": "new"}))
            os.replace(replacement, path)
            replaced.append(True)
        return data

    monkeypatch.setattr(task_status, "raw_result_facts",
                        lambda directory: task_result_scan.raw_result_facts(directory, reader=reader))
    rows = task_status.find_child_tasks(tmp_path, parent_task_id="parent", scope="direct",
                                        materialize_artifacts=False)
    assert _ids(rows) == ["changing"]
    assert rows[0]["result"] == "new"
    assert task_result_scan._RAW_TS_MEMO.get((str(path.parent), path.name)) is None


@pytest.mark.parametrize("pointer,first,second", [
    ("superseded_by", "a_old", "z_old"),
    ("retry_task_id", "z_old", "a_old"),
])
def test_retry_collision_uses_last_filename_whole_body_and_effective_exclusion(
    tmp_path, monkeypatch, pointer, first, second,
):
    source = tmp_path / "source"
    _put(source, "middle", delegation_role="subagent", parent_task_id="parent",
         root_task_id="parent", result="winner body", cost_usd=6.25)
    for name in (first, second):
        _put(source, name, delegation_role="subagent", parent_task_id="elsewhere",
             root_task_id="elsewhere", result=f"original {name}", **{pointer: "middle"})
    _, _, rows = _pair(tmp_path, monkeypatch, source, parent_task_id="parent", scope="direct")
    assert _ids(rows) == ["middle"]
    assert rows[0]["result"] == "winner body" and rows[0]["cost_usd"] == 6.25
    assert rows[0]["original_task_id"] == max(first, second)
    assert rows[0]["retry_lineage"][0]["task_id"] == max(first, second)
    assert task_status.find_child_tasks(tmp_path / "new", parent_task_id="parent",
                                        scope="direct", exclude_task_id="middle",
                                        materialize_artifacts=False) == []


def test_roles_subtree_exclusion_and_queue_terminal_truth(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _put(source, "direct", delegation_role="subagent", parent_task_id="parent",
         root_task_id="parent", status="completed", result="terminal disk")
    _put(source, "deep", delegation_role="subagent", parent_task_id="direct",
         root_task_id="parent")
    _put(source, "wrong_role", delegation_role="root", parent_task_id="parent",
         root_task_id="parent")
    _put(source, "wrong_tree", delegation_role="subagent", parent_task_id="other",
         root_task_id="other")
    _put(source, "lineage_less", delegation_role="", parent_task_id="", root_task_id="",
         status="running", result="disk payload", cost_usd=2.5,
         artifacts=[{"path": "one.txt"}])
    state = source / "state"
    state.mkdir(parents=True)
    (state / "queue_snapshot.json").write_text(json.dumps({"pending": [], "running": [
        {"id": "direct", "task": {"id": "direct", "delegation_role": "subagent",
         "parent_task_id": "parent", "root_task_id": "parent"}},
        {"id": "lineage_less", "task": {"id": "lineage_less",
         "delegation_role": "subagent", "parent_task_id": "parent", "root_task_id": "parent"}},
    ]}), encoding="utf-8")
    for scope, excluded, expected in (
        ("direct", "", ["direct", "lineage_less"]),
        ("subtree", "direct", ["deep", "lineage_less"]),
    ):
        case = tmp_path / f"case-{scope}"
        case.mkdir()
        _, _, rows = _pair(case, monkeypatch, source, parent_task_id="parent",
                           root_task_id="parent", scope=scope, exclude_task_id=excluded)
        assert _ids(rows) == expected
        by_id = {row["task_id"]: row for row in rows}
        assert by_id["lineage_less"]["result"] == "disk payload"
        assert by_id["lineage_less"]["cost_usd"] == 2.5
        assert by_id["lineage_less"]["artifacts"] == [{"path": "one.txt"}]
        assert by_id["lineage_less"]["status"] == "running"
        if "direct" in by_id:
            assert by_id["direct"]["status"] == "completed"


def test_malformed_future_legacy_one_batch_and_strict_no_quarantine(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _put(source, "child", delegation_role="subagent", parent_task_id="parent")
    _put(source, "future", _schema_version=TASK_RESULT_SCHEMA_VERSION + 1)
    legacy = _put(source, "legacy")
    legacy.write_text(json.dumps({"task_id": "legacy", "status": "completed"}))
    (source / "task_results" / "broken.json").write_text("{", encoding="utf-8")
    strict_root = tmp_path / "strict"
    shutil.copytree(source, strict_root)
    with pytest.raises(ValueError):
        task_results.list_task_results(strict_root, strict=True, _paths=[])
    assert sorted(path.name for path in (strict_root / "task_results").glob("*.json")) == [
        "broken.json", "child.json", "future.json", "legacy.json"]
    _, new_root, rows = _pair(tmp_path, monkeypatch, source,
                              parent_task_id="parent", scope="direct")
    assert _ids(rows) == ["child"]
    assert sorted(path.name for path in (new_root / "task_results" / "quarantine").glob("*.json")) == [
        "broken.json", "future.json", "legacy.json"]
    events = [json.loads(line) for line in (new_root / "logs" / "events.jsonl").read_text().splitlines()]
    assert [event["count"] for event in events if event.get("type") == "task_results_quarantined"] == [3]


def test_concurrent_repair_kept_then_fresh_predicate(tmp_path, monkeypatch):
    path = _put(tmp_path, "repair", _schema_version=TASK_RESULT_SCHEMA_VERSION + 1,
                delegation_role="subagent", parent_task_id="other")
    real = task_results._quarantine_task_result
    repaired = []

    def repair_before_quarantine(candidate, reason):
        if candidate == path and not repaired:
            repaired.append(True)
            _put(tmp_path, "repair", delegation_role="subagent", parent_task_id="other")
        return real(candidate, reason)

    monkeypatch.setattr(task_results, "_quarantine_task_result", repair_before_quarantine)
    assert task_status.find_child_tasks(tmp_path, parent_task_id="parent",
                                        scope="direct", materialize_artifacts=False) == []
    assert path.exists() and not (path.parent / "quarantine" / path.name).exists()
    assert repaired


def test_admitted_body_replaces_memo_fact_before_predicate(tmp_path, monkeypatch):
    path = _put(tmp_path, "selected", delegation_role="subagent",
                parent_task_id="parent", root_task_id="parent")
    original = task_status.list_task_results
    replaced = []

    def replace_before_admission(root, **kwargs):
        if not replaced:
            replaced.append(True)
            _put(tmp_path, "selected", delegation_role="subagent",
                 parent_task_id="other", root_task_id="other")
        return original(root, **kwargs)

    monkeypatch.setattr(task_status, "list_task_results", replace_before_admission)
    assert task_status.find_child_tasks(tmp_path, parent_task_id="parent", scope="direct",
                                        materialize_artifacts=False) == []
    assert replaced and path.exists()


def test_same_size_atomic_replace_restored_mtime_invalidates_navigation(tmp_path):
    path = _put(tmp_path, "equal", delegation_role="subagent", parent_task_id="other",
                root_task_id="other", result="same")
    lookup = lambda: _ids(task_status.find_child_tasks(
        tmp_path, parent_task_id="other", scope="direct", materialize_artifacts=False))
    assert lookup() == ["equal"]
    previous = path.stat()
    content = path.read_text().replace('"parent_task_id": "other"', '"parent_task_id": "alien"')
    # Keep byte length exactly fixed while replacing the inode.
    content = content.replace('"root_task_id": "other"', '"root_task_id": "alien"')
    assert len(content.encode()) == previous.st_size
    replacement = path.with_suffix(".replacement")
    replacement.write_text(content, encoding="utf-8")
    os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    os.replace(replacement, path)
    assert path.stat().st_mtime_ns == previous.st_mtime_ns
    assert lookup() == []


@pytest.mark.parametrize("kind", ["missing", "corrupt", "inaccessible"])
def test_navigation_directory_failure_keeps_queue_overlay(tmp_path, monkeypatch, kind):
    state = tmp_path / "state"
    state.mkdir()
    (state / "queue_snapshot.json").write_text(json.dumps({"pending": [{
        "id": "queued", "task": {"id": "queued", "delegation_role": "subagent",
        "parent_task_id": "parent"}}], "running": []}), encoding="utf-8")
    if kind == "corrupt":
        (tmp_path / "task_results").write_text("not a directory")
    elif kind == "inaccessible":
        def denied(directory):
            raise PermissionError("simulated directory refusal")

        monkeypatch.setattr(task_status, "raw_result_facts", denied)
        original_glob = pathlib.Path.glob

        def denied_glob(directory, pattern):
            if directory == tmp_path / "task_results":
                raise PermissionError("simulated canonical directory refusal")
            return original_glob(directory, pattern)

        monkeypatch.setattr(pathlib.Path, "glob", denied_glob)
    # Match canonical glob behavior on this platform: a non-directory can
    # yield no files on Windows but raise on POSIX. Do not change that policy.
    try:
        old = _old_lookup(monkeypatch, tmp_path, parent_task_id="parent", scope="direct")
    except OSError as error:
        with pytest.raises(type(error)):
            task_status.find_child_tasks(tmp_path, parent_task_id="parent", scope="direct",
                                         materialize_artifacts=False)
        if kind == "inaccessible":
            assert isinstance(error, PermissionError)
        return
    assert kind != "inaccessible", "injected canonical PermissionError must propagate"
    rows = task_status.find_child_tasks(tmp_path, parent_task_id="parent", scope="direct",
                                        materialize_artifacts=False)
    assert rows == old
    assert _ids(rows) == ["queued"] and rows[0]["status"] == "scheduled"


def test_gateway_wrapper_and_shared_memo_injected_reader(tmp_path):
    import ouroboros.gateway.task_list_scan as gateway_scan

    path = _put(tmp_path, "gateway", ts="2026-10-04T01:00:00Z")
    directory = path.parent
    calls = []

    def reader(candidate):
        calls.append(candidate)
        return json.loads(candidate.read_text())

    first, bad = gateway_scan.raw_result_facts(directory, reader=reader)
    assert not bad and first["gateway.json"]["task_id"] == "gateway"
    assert calls == [path]
    second, bad = gateway_scan.raw_result_facts(directory, reader=reader)
    assert second == first and not bad and calls == [path]
    shared, bad = task_result_scan.raw_result_facts(directory, reader=reader)
    assert shared == first and not bad and calls == [path]


def test_gateway_sorted_names_uses_shared_memo_and_wrapper_reader(tmp_path, monkeypatch):
    import ouroboros.gateway.task_list_scan as gateway_scan

    path = _put(tmp_path, "shown", ts="2026-10-04T01:00:00Z")
    calls = []
    original = gateway_scan.read_json_dict

    def injected(candidate):
        calls.append(candidate)
        return original(candidate)

    monkeypatch.setattr(gateway_scan, "read_json_dict", injected)
    assert gateway_scan._raw_sorted_result_names(path.parent) == (["shown.json"], [])
    assert calls == [path]
    assert gateway_scan._raw_sorted_result_names(path.parent) == (["shown.json"], [])
    assert calls == [path]


@pytest.mark.parametrize("root_field", ["child_drive_root", "headless_child_drive_root"])
@pytest.mark.parametrize("materialize", [False, True])
def test_split_child_metadata_and_artifact_modes(tmp_path, monkeypatch, root_field, materialize):
    source = tmp_path / "source"
    _put(source, "child", delegation_role="subagent", parent_task_id="parent",
         root_task_id="parent", status="running", result="canonical",
         metadata={root_field: "placeholder"}, artifacts=[{"path": "recorded.txt"}])
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    shutil.copytree(source, old_root)
    shutil.copytree(source, new_root)
    values = []
    for root in (old_root, new_root):
        child_drive = root / "split"
        _put(child_drive, "child", status="completed", result="replica terminal",
             metadata={"replica_fact": "yes"})
        canonical = root / "task_results" / "child.json"
        row = json.loads(canonical.read_text())
        row["metadata"][root_field] = str(child_drive)
        canonical.write_text(json.dumps(row))
        if root == old_root:
            values.append(_old_lookup(monkeypatch, root, parent_task_id="parent", scope="direct",
                                      materialize_artifacts=materialize))
        else:
            values.append(task_status.find_child_tasks(
                root, parent_task_id="parent", scope="direct",
                materialize_artifacts=materialize))
    assert _root_normalized_json(values[0], old_root) == _root_normalized_json(values[1], new_root)
    assert _ids(values[1]) == ["child"]
    assert values[1][0]["result"] == "replica terminal"
    assert values[1][0]["status"] == "completed"
    assert values[1][0]["metadata"]["replica_fact"] == "yes"


def test_public_coordination_terminal_wake_after_child_settles(tmp_path, monkeypatch):
    import ouroboros.delegate_supervision as supervision

    source = tmp_path / "source"
    _put(source, "child", delegation_role="subagent", parent_task_id="parent",
         root_task_id="parent", status="running")
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    shutil.copytree(source, old_root)
    shutil.copytree(source, new_root)
    observed = []
    for root in (old_root, new_root):
        ctx = SimpleNamespace(drive_root=root, task_id="parent", task_metadata={})
        if root == old_root:
            with monkeypatch.context() as patch:
                patch.setattr(task_status, "list_task_results",
                              lambda drive, **_kw: task_results.list_task_results(drive))
                patch.setattr(task_status, "raw_result_facts", lambda _dir: ({}, []))
                before, cursor = supervision._coordination_wakes(ctx, {})
                _put(root, "child", delegation_role="subagent", parent_task_id="parent",
                     root_task_id="parent", status="completed")
                after, _ = supervision._coordination_wakes(
                    ctx, {"coordination_cursor": cursor})
        else:
            before, cursor = supervision._coordination_wakes(ctx, {})
            _put(root, "child", delegation_role="subagent", parent_task_id="parent",
                 root_task_id="parent", status="completed")
            after, _ = supervision._coordination_wakes(ctx, {"coordination_cursor": cursor})
        observed.append((before, after))
    assert observed[0] == observed[1]
    assert observed[0][0] == []
    assert [(event["child_task_id"], event["status"]) for event in observed[0][1]
            if event["type"] == "child_terminal"] == [("child", "completed")]
