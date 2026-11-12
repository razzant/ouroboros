import pathlib
import re
from types import SimpleNamespace

from ouroboros.task_tree_ledger import tree_ledger_append
from ouroboros.tools.project_journal import (
    _journal_read,
    journal_tail_digest,
)
from ouroboros.tools.task_tree import _tree_read
from ouroboros.utils import append_jsonl


def _snapshot(text: str) -> str:
    match = re.search(r"snapshot=['\"]?([0-9a-f]{64})", text)
    assert match is not None, text
    return match.group(1)


def _journal_rows(root, project_id: str, count: int) -> None:
    path = root / "projects" / project_id / "journal.jsonl"
    for index in range(count):
        append_jsonl(path, {
            "ts": f"2026-08-21T00:{index // 60:02d}:{index % 60:02d}Z",
            "kind": "note",
            "text": f"journal-{index:03d}",
            "task_id": f"task-{index:03d}",
        })


def test_journal_read_pages_205_rows_and_digest_pointer_is_executable(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    _journal_rows(tmp_path, "mine", 205)
    _journal_rows(tmp_path, "other", 205)
    ctx = SimpleNamespace(
        drive_root=tmp_path / "fork",
        budget_drive_root=str(tmp_path),
        project_id="mine",
        task_metadata={"budget_drive_root": str(tmp_path), "root_task_id": "root"},
    )

    first = _journal_read(ctx, "other", limit=200)
    assert "total=205" in first and "remaining=5" in first
    assert "journal-005" in first and "journal-204" in first
    assert "journal-004" not in first
    second = _journal_read(
        ctx, "other", limit=200, offset=200, snapshot=_snapshot(first),
    )
    assert "total=205" in second and "remaining=0" in second
    assert "journal-000" in second and "journal-004" in second
    assert "journal-005" not in second

    digest = journal_tail_digest("mine", limit=40)
    assert "journal_read(project_id='mine', limit=40, offset=40, snapshot=" in digest
    pointed = _journal_read(
        ctx, "", limit=40, offset=40, snapshot=_snapshot(digest),
    )
    assert "journal-164" in pointed and "journal-165" not in pointed


def test_journal_read_refuses_append_between_pages(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    _journal_rows(tmp_path, "mine", 3)
    ctx = SimpleNamespace(drive_root=tmp_path / "fork", project_id="mine")
    first = _journal_read(ctx, limit=2)
    _journal_rows(tmp_path, "mine", 1)

    changed = _journal_read(ctx, limit=2, offset=2, snapshot=_snapshot(first))
    assert changed.startswith("JOURNAL_READ_SNAPSHOT_CHANGED:")
    assert "journal-000" not in changed


def test_journal_read_preserves_valid_rows_but_marks_malformed_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    _journal_rows(tmp_path, "mine", 3)
    path = tmp_path / "projects" / "mine" / "journal.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"broken":\n')
        handle.write(b'["legacy", "array"]\n')
    ctx = SimpleNamespace(drive_root=tmp_path / "fork", project_id="mine")

    first = _journal_read(ctx, limit=2)
    assert "total=5" in first
    assert "valid_total=3" in first
    assert "unreadable=2" in first
    assert "coverage=partial" in first
    second = _journal_read(ctx, limit=2, offset=2, snapshot=_snapshot(first))
    joined = first + second
    assert all(f"journal-{index:03d}" in joined for index in range(3))
    assert "remaining=0" in second and "coverage=partial" in second

    digest = journal_tail_digest("mine", limit=40)
    assert "coverage partial" in digest
    assert "2 unreadable" in digest


def test_journal_read_preserves_valid_rows_with_invalid_utf8(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    _journal_rows(tmp_path, "mine", 1)
    path = tmp_path / "projects" / "mine" / "journal.jsonl"
    with path.open("ab") as handle:
        handle.write(b"\xfflegacy-row\n")
    ctx = SimpleNamespace(drive_root=tmp_path / "fork", project_id="mine")

    page = _journal_read(ctx, limit=2)

    assert "journal-000" in page
    assert "unreadable=1" in page
    assert "coverage=partial" in page


def _tree_rows(root, root_id: str, count: int) -> None:
    for index in range(count):
        result = tree_ledger_append(
            root_id,
            "note",
            f"tree-{index:03d}",
            task_id=f"child-{index:03d}",
            data_root=root,
        )
        assert result.startswith("OK:")


def test_tree_read_pages_500_rows_from_fork_and_digest_pointer_is_executable(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    _tree_rows(tmp_path, "root-task", 500)
    assert tree_ledger_append(
        "other-root", "note", "foreign-tree", task_id="foreign", data_root=tmp_path,
    ).startswith("OK:")
    ctx = SimpleNamespace(
        task_id="child-task",
        drive_root=tmp_path / "fork",
        budget_drive_root=str(tmp_path),
        task_metadata={
            "root_task_id": "root-task",
            "budget_drive_root": str(tmp_path),
        },
    )

    pages = []
    offset = 0
    snapshot = ""
    while True:
        page = _tree_read(ctx, limit=200, offset=offset, snapshot=snapshot)
        pages.append(page)
        snapshot = snapshot or _snapshot(page)
        remaining = int(re.search(r"remaining=(\d+)", page).group(1))
        if remaining == 0:
            break
        offset += 200
    joined = "\n".join(pages)
    assert len(pages) == 3
    assert "tree-300" in pages[0] and "tree-499" in pages[0]
    assert "tree-299" not in pages[0]
    assert "foreign-tree" not in joined
    assert all(f"tree-{index:03d}" in joined for index in range(500))

    from ouroboros.task_tree_ledger import tree_ledger_tail_digest

    digest = tree_ledger_tail_digest("root-task", limit=40, data_root=tmp_path)
    assert "tree_read(limit=40, offset=40, snapshot=" in digest
    pointed = _tree_read(ctx, limit=40, offset=40, snapshot=_snapshot(digest))
    assert "tree-420" in pointed and "tree-419" not in pointed


def test_tree_read_refuses_append_between_pages(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    _tree_rows(tmp_path, "root-task", 3)
    ctx = SimpleNamespace(
        task_id="child",
        drive_root=tmp_path / "fork",
        budget_drive_root=str(tmp_path),
        task_metadata={"root_task_id": "root-task"},
    )
    first = _tree_read(ctx, limit=2)
    _tree_rows(tmp_path, "root-task", 1)

    changed = _tree_read(ctx, limit=2, offset=2, snapshot=_snapshot(first))
    assert changed.startswith("TREE_READ_SNAPSHOT_CHANGED:")
    assert "tree-000" not in changed


def test_tree_read_preserves_valid_rows_but_marks_malformed_coverage(tmp_path):
    _tree_rows(tmp_path, "root-task", 3)
    path = tmp_path / "task_trees" / "root-task" / "blackboard.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"broken":\n')
        handle.write(b'"legacy scalar"\n')
    ctx = SimpleNamespace(
        task_id="child",
        drive_root=tmp_path / "fork",
        budget_drive_root=str(tmp_path),
        task_metadata={"root_task_id": "root-task"},
    )

    first = _tree_read(ctx, limit=2)
    assert "total=5" in first
    assert "valid_total=3" in first
    assert "unreadable=2" in first
    assert "coverage=partial" in first
    second = _tree_read(ctx, limit=2, offset=2, snapshot=_snapshot(first))
    joined = first + second
    assert all(f"tree-{index:03d}" in joined for index in range(3))
    assert "remaining=0" in second and "coverage=partial" in second


def test_tree_read_preserves_valid_rows_with_invalid_utf8(tmp_path):
    _tree_rows(tmp_path, "root-task", 1)
    path = tmp_path / "task_trees" / "root-task" / "blackboard.jsonl"
    with path.open("ab") as handle:
        handle.write(b"\xfflegacy-row\n")
    ctx = SimpleNamespace(
        task_id="child",
        drive_root=tmp_path / "fork",
        budget_drive_root=str(tmp_path),
        task_metadata={"root_task_id": "root-task"},
    )

    page = _tree_read(ctx, limit=2)

    assert "tree-000" in page
    assert "unreadable=1" in page
    assert "coverage=partial" in page


def test_runtime_tree_digest_uses_same_canonical_root_as_tree_read(tmp_path):
    from ouroboros.context import build_runtime_section

    canonical = tmp_path / "canonical"
    fork = tmp_path / "fork"
    fork.mkdir()
    assert tree_ledger_append(
        "root-task", "note", "canonical-passive-row", task_id="child",
        data_root=canonical,
    ).startswith("OK:")
    path = canonical / "task_trees" / "root-task" / "blackboard.jsonl"
    with path.open("ab") as handle:
        handle.write(b"\xfflegacy-row\n")
    env = SimpleNamespace(
        repo_dir=pathlib.Path(__file__).parents[1],
        drive_root=fork,
        budget_drive_root=fork,
    )
    task = {
        "id": "child",
        "root_task_id": "root-task",
        "drive_root": str(fork),
    }
    ctx = SimpleNamespace(
        drive_root=fork,
        budget_drive_root="",
        task_metadata={"budget_drive_root": str(canonical)},
    )

    rendered = build_runtime_section(env, task, ctx=ctx)

    assert "canonical-passive-row" in rendered
    assert "coverage partial: 1 unreadable task-tree row" in rendered
