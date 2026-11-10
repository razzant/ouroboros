"""Effective-result reads are memoized only inside one state projection."""

from types import SimpleNamespace

from ouroboros.gateway import work_projection as wp


def _reader(value, calls, stamps=None):
    def effective(task_id):
        calls.append(task_id)
        return value

    return SimpleNamespace(versions={"task": 1}, _stamps=stamps or {}, effective=effective)


def test_effective_result_is_reused_within_one_snapshot(tmp_path):
    calls = []
    obs = wp._Observations(tmp_path)
    obs.task_reads = _reader({"status": "completed"}, calls)
    assert obs.effective("task")["status"] == "completed"
    assert obs.effective("task")["status"] == "completed"
    assert calls == ["task"]


def test_effective_read_exports_child_dependency_even_when_unavailable(tmp_path):
    child = str(tmp_path / "child-drive")
    calls = []
    obs = wp._Observations(tmp_path)
    obs.task_reads = _reader({}, calls, stamps={child: None})
    assert obs.effective("task") == {}
    assert child in obs.dependency_roots
    assert calls == ["task"]
