"""The cutover's two structural guards.

1. No production code reads, patches, unlinks or stats ``task_results/*.json`` outside the
   importer/exporter's directory work, the quarantine and the sidecar locks: the allowlist
   names CALLS, not files (two-sided: a real pre-cutover reader restored into an owner module
   is caught).
2. Every store read the runtime makes is answered from an index: ``EXPLAIN QUERY PLAN`` over the
   statements CAPTURED while the runtime API runs against a store of 3,000 rows shows no table
   scan of ``tasks`` or ``task_bodies`` (two-sided: a broken WHERE and a dropped index are caught).
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import sqlite3
import subprocess
import sys

import pytest

from ouroboros import obligations, store, task_results, task_store

REPO = pathlib.Path(__file__).resolve().parents[1]

# The only CALLS that may touch a result FILE, per module: the importer/exporter owns the
# directory; the result module defines the two path helpers; the rest name the quarantine
# directory or a sidecar lock beside the result. Any other mention in these files is a violation.
_LOCK = r"\.lock[\"']"
_QUARANTINE = r"TASK_RESULT_QUARANTINE_DIR"
RESULT_FILE_CALLS = {
    "ouroboros/store_import.py": r".",
    "ouroboros/task_results.py": r"^def task_results?_(dir|path)\(|^    return task_results_dir\(drive_root, create=create\) / f\"\{validate_task_id\(task_id\)\}\.json\"$",
    "ouroboros/terminal_projection.py": _LOCK,
    "ouroboros/owner_pause.py": f"{_LOCK}|{_QUARANTINE}",
    "ouroboros/task_custody.py": _LOCK,
    "ouroboros/presence_runner.py": _QUARANTINE,
}
# Directory-layout mentions that are not result files: the artifact tree under task_results/,
# data-root layout lists, lineage labels.
_ALLOWED_LINE = re.compile(
    r"task_results[\"'/)]*\s*/\s*[\"']?(artifacts|_ARTIFACT|versions)|ARTIFACTS_DIR|"
    r"lineage=\"task_results\"|\"task_results\",\s*\"uploads\"|\(\"memory\", \"logs\", \"state\", \"task_results\"\)|"
    r"_EVIDENCE_DIRS|\"task_results\",$|skill_payload_policy|task_results/artifacts|task_results\" / _ARTIFACT"
)
_VIOLATION = re.compile(
    r"task_result_path\(|task_results_dir\(|[\"']task_results[\"']\s*/\s*f?[\"'][^\"']*\.json|"
    r"/\s*[\"']task_results[\"']\s*/\s*f[\"']\{|task_results\.json|\"task_results\"\s*\)\s*\.(glob|iterdir)"
)


def _production_files():
    for folder in ("ouroboros", "supervisor", "skills"):
        yield from sorted((REPO / folder).rglob("*.py"))
    yield REPO / "server.py"


def result_file_consumers(files, base) -> list:
    hits = []
    for path in files:
        rel = path.relative_to(base).as_posix()
        allowed = re.compile(RESULT_FILE_CALLS[rel]) if rel in RESULT_FILE_CALLS else None
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not _VIOLATION.search(line) or _ALLOWED_LINE.search(line) or line.lstrip().startswith("#"):
                continue
            if allowed is not None and allowed.search(line):
                continue
            hits.append(f"{rel}:{number}: {line.strip()}")
    return hits


def test_no_production_code_touches_a_result_file_outside_its_owners():
    assert result_file_consumers(_production_files(), REPO) == []


def test_the_consumer_guard_catches_a_planted_file_read(tmp_path):
    planted = tmp_path / "ouroboros" / "planted.py"
    planted.parent.mkdir(parents=True)
    planted.write_text(
        'import json\n'
        'def read(root, tid):\n'
        '    return json.loads((root / "task_results" / f"{tid}.json").read_text(encoding="utf-8"))\n',
        encoding="utf-8")
    hits = result_file_consumers([planted], tmp_path)
    assert hits and hits[0].startswith("ouroboros/planted.py:3")


_OLD_READER = (
    "def load_task_result(drive_root, task_id, *, strict=False):\n"
    "    path = task_result_path(drive_root, tid, create=False)\n"
    "    data = json.loads(read_text_across_replace(path))\n")


@pytest.mark.parametrize("owner", ["ouroboros/task_custody.py", "ouroboros/owner_pause.py", "ouroboros/task_results.py"])
def test_the_consumer_guard_catches_the_old_reader_restored_into_an_owner_module(tmp_path, owner):
    """The pre-cutover ``load_task_result`` (its file read, verbatim shape) appended to a module
    whose own lock/quarantine/helper calls are allowed is still a violation at exactly its line."""
    copy = tmp_path / owner
    copy.parent.mkdir(parents=True, exist_ok=True)
    original = (REPO / owner).read_text(encoding="utf-8")
    copy.write_text(original + "\n\n" + _OLD_READER, encoding="utf-8")
    hits = result_file_consumers([copy], tmp_path)
    line = original.count("\n") + 4
    assert hits == [f"{owner}:{line}: path = task_result_path(drive_root, tid, create=False)"], hits


@pytest.fixture(scope="module")
def populated(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    with store.open_store(root, write=True) as txn:
        for i in range(3000):
            lineage = ({"delegation_role": "subagent", "parent_task_id": f"t{i - 1}", "root_task_id": "t0"}
                       if i % 7 else {"project_id": f"p{i % 5}"})
            txn.store_task({"task_id": f"t{i}", "status": "completed" if i % 3 else "running", "_schema_version": 1,
                            "ts": f"2026-01-01T00:{i // 60 % 60:02d}:{i % 60:02d}Z", **lineage})
    return root


def _plan(root, sql, params=()):
    conn = sqlite3.connect(store.store_path(root))
    try:
        return [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()]
    finally:
        conn.close()


def _scans(plan):
    return [step for step in plan if re.match(r"SCAN (tasks|task_bodies|t|b)\b", step)
            and "USING COVERING INDEX tasks_ts" not in step and "USING INDEX tasks_updated" not in step]


class _Recording(sqlite3.Connection):
    """The connection the runtime opens during a capture: every statement it executes is kept."""

    captured: list = []

    def execute(self, sql, params=()):
        type(self).captured.append((str(sql), tuple(params) if not isinstance(params, dict) else params))
        return super().execute(sql, params)


_TABLES = re.compile(r"\bFROM\s+(tasks|task_bodies|obligations|custody_runs|handoffs)\b", re.IGNORECASE)


def captured_statements(monkeypatch, call) -> list:
    """``(sql, params)`` of every read over the store's tables that ``call`` made."""
    store.close_connections()  # install the recorder before opening the persistent connection
    real = sqlite3.connect

    def recording(*args, factory=sqlite3.Connection, **kwargs):  # over the store's own connection class
        return real(*args, factory=type("Recording", (_Recording, factory), {}), **kwargs)

    monkeypatch.setattr(sqlite3, "connect", recording)
    _Recording.captured = []
    try:
        call()
    finally:
        monkeypatch.setattr(sqlite3, "connect", real)
    return [(sql, params) for sql, params in _Recording.captured
            if sql.lstrip().upper().startswith("SELECT") and _TABLES.search(sql)]


def runtime_reads(root):
    """``(name, call)``: the store reads the runtime makes, through its own API (the bounded
    ordered scans ``newest_ids``/``inventory`` read an index in order and are allowed by ``_scans``)."""
    from ouroboros.task_status import find_child_tasks
    from ouroboros.tools.control_delegation import durable_direct_child_count

    ids = [f"t{i}" for i in range(20)]
    return [
        ("find_child_tasks subtree", lambda: find_child_tasks(root, parent_task_id="t5", root_task_id="t0", materialize_artifacts=False)),
        ("find_child_tasks direct", lambda: find_child_tasks(root, parent_task_id="t5", scope="direct", materialize_artifacts=False)),
        ("list_task_results(statuses)", lambda: task_results.list_task_results(root, statuses=["running", "interrupted"])),
        ("documents_by_ids", lambda: task_store.documents_by_ids(root, ids)),
        ("durable_direct_child_count", lambda: durable_direct_child_count(root, "t5")),
        ("nonterminal members", lambda: obligations.members(root, "nonterminal")),
        *[(f"{name} members", lambda name=name: obligations.members(root, name)) for name in store.FLAG_COLUMNS],
        ("task_facts(lineage)", lambda: task_store.task_facts(
            root, where="delegation_role='subagent' AND (parent_task_id=? OR root_task_id=?)", params=("t5", "t0"))),
        ("task_facts(tree census)", lambda: task_store.task_facts(root, where="root_task_id=? OR root_task_id IS NULL", params=("t0",))),
        ("task_facts(project)", lambda: task_store.task_facts(root, where="project_id=?", params=("p1",))),
        ("newest_ids(project roots)", lambda: task_store.newest_ids(
            root, where=f"project_id=? AND {task_store.ROOT_RESULT_WHERE}", params=("p1",))),
        ("newest_ids(limit)", lambda: task_store.newest_ids(root, 50)),
        ("inventory", lambda: task_store.inventory(root)),
        ("read_document", lambda: task_store.read_document(root, "t5")),
        ("signature", lambda: task_store.signature(root, "t5")),
        ("owner-held members", lambda: obligations.members(root, "custody_open")),
    ]


def test_the_runtime_reads_are_answered_from_an_index_never_the_table(populated, monkeypatch):
    scans = {}
    for name, call in runtime_reads(populated):
        statements = captured_statements(monkeypatch, call)
        assert statements, f"{name}: no store read was captured"
        for sql, params in statements:
            plan = _plan(populated, sql, params)
            if _scans(plan):
                scans[name] = (sql, plan)
    assert scans == {}


def test_the_captured_plan_guard_catches_a_broken_where_and_a_dropped_index(populated, monkeypatch, tmp_path):
    """The control: the lineage statement the runtime really issues, with its WHERE made
    unindexable, and unchanged over a copy of the store whose lineage indexes are dropped."""
    [(sql, params)] = [row for row in captured_statements(  # the statement that carries the lineage WHERE
        monkeypatch, lambda: task_store.select_lineage(populated, "t5", "t0"))
        if "task_bodies" in row[0] and "parent_task_id" in row[0]]
    assert "parent_task_id" in sql and not _scans(_plan(populated, sql, params))
    assert _scans(_plan(populated, sql.replace("parent_task_id=?", "parent_task_id||''=?"), params))
    copy = tmp_path / "copy"
    shutil.copytree(store.store_path(populated).parent, copy / "state")
    conn = sqlite3.connect(store.store_path(copy))
    conn.executescript("DROP INDEX tasks_parent; DROP INDEX tasks_root; DROP INDEX tasks_superseded; DROP INDEX tasks_retry;")
    conn.close()
    assert _scans(_plan(copy, sql, params))


def test_the_plan_guard_catches_a_table_scan(populated):
    plan = _plan(populated, "SELECT task_id FROM tasks WHERE child_drive_root=?", ("/x",))
    assert _scans(plan)


def test_select_lineage_matches_the_python_prefilter(populated):
    """Without retry lineage the closure selects exactly the retired Python prefilter's rows; with it,
    ``test_child_lineage_closure`` holds the full-projection oracle (unrelated lineage is never read)."""
    def may_match(item, parent, root_id, direct_only):
        if str(item.get("superseded_by") or item.get("retry_task_id") or "").strip():
            return True
        if str(item.get("delegation_role") or "") != "subagent":
            return False
        if parent and str(item.get("parent_task_id") or "") == parent:
            return True
        return bool(not direct_only and root_id and str(item.get("root_task_id") or "") == root_id)

    everything = task_store.list_documents(populated)
    for parent, root_id, direct_only in (("t5", "t0", False), ("t5", "", True), ("", "t0", False), ("t9", "t0", True)):
        expected = sorted((row["task_id"] for row in everything if may_match(row, parent, root_id, direct_only)))
        selected = [row["task_id"] for row in task_store.select_lineage(populated, parent, root_id, direct_only=direct_only)]
        assert selected == expected, (parent, root_id, direct_only)


def test_the_store_opens_in_a_fresh_interpreter_without_the_runtime(tmp_path):
    """``task_store`` depends on ``store`` alone (an importer/exporter and the P8 lane run it bare)."""
    code = ("import pathlib, sys; from ouroboros import task_store; "
            "print(task_store.exists(pathlib.Path(sys.argv[1]), 'none'))")
    child_env = {"PYTHONPATH": str(REPO), "PATH": os.defpath, "OUROBOROS_DATA_DIR": str(tmp_path)}
    if os.environ.get("SystemRoot"):  # Windows Python needs this even before importing the store.
        child_env["SystemRoot"] = os.environ["SystemRoot"]
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, cwd=REPO,
                         env=child_env)
    assert out.returncode == 0 and out.stdout.strip() == "False", out.stderr
