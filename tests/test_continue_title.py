"""#1717: a chain of Continues keeps the work's own name under one prefix.

The link between links is ``metadata.continuation``; the title carries no
second copy of it, so "Continue: Continue: …" never stacks and an 80-character
name is not pushed out by its own prefixes.
"""

from __future__ import annotations

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_owner_continue import NONCE, _interrupted

SOURCES = {"original": {"content": "Write the Friday report"}, "later": []}


def _successor(predecessor: str, result: dict, successor: str) -> dict:
    from supervisor.continuation_admission import _successor_task

    return _successor_task(None, predecessor, result, {"successor_task_id": successor, "chat_id": 7},
                           {"cause": "context_overflow"}, SOURCES, {})


@pytest.mark.parametrize("name", [
    pytest.param("Friday report", id="short"),
    pytest.param("x" * 80, id="at-limit"),
    pytest.param("y" * 120, id="over-limit"),
])
def test_a_chain_of_continues_keeps_one_prefix_and_its_provenance(name):
    result, predecessor = {"title": name}, "pred-0"
    for link in range(1, 5):
        task = _successor(predecessor, result, f"succ-{link}")
        assert task["title"] == task["suggested_name"] == f"Continue: {name[:80]}"
        assert task["metadata"]["continuation"]["predecessor_task_id"] == predecessor
        result, predecessor = task, task["id"]


def test_a_title_that_is_only_prefixes_falls_back_to_the_predecessor_id():
    task = _successor("pred-0", {"title": "Continue: Continue: "}, "succ-1")
    assert task["title"] == "Continue: pred-0"


def test_admission_does_not_stack_a_predecessor_already_carrying_prefixes(tmp_path, monkeypatch):
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, title="Continue: Continue: Friday report")
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    assert ack["ok"] is True, ack
    row = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
    assert row["title"] == row["suggested_name"] == "Continue: Friday report"
    assert row["metadata"]["continuation"]["predecessor_task_id"] == "pred-1"
