"""Focused acceptance-fence owner identity regression coverage."""

from tests.test_acceptance_fence import _isolated_queue


def test_begin_cannot_readopt_different_live_owners_fence(monkeypatch, tmp_path):
    queue_mod, _pending = _isolated_queue(monkeypatch, tmp_path)
    queue_mod.RUNNING["owner-1"] = {"task": {"id": "owner-1", "root_task_id": "root-1"}}
    queue_mod.transition_acceptance_fence(
        action="begin", token="a" * 32, root_task_id="root-1", task_id="owner-1",
    )

    rejected = queue_mod.transition_acceptance_fence(
        action="begin", token="b" * 32, root_task_id="root-1", task_id="owner-2",
    )

    assert rejected["ok"] is False
    assert "different live owner" in rejected["error"]
    assert queue_mod.ACCEPTANCE_FENCES["root-1"]["token"] == "a" * 32
    assert queue_mod.ACCEPTANCE_FENCES["root-1"]["task_id"] == "owner-1"
