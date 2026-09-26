"""The original turn owns whether autobiography awaits provider receipts."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ouroboros.presence_runner import PresenceTurnGate, run_presence_turn
from ouroboros.task_results import write_task_result
from ouroboros.utils import atomic_write_json
from tests.test_presence_runner import _admission, _event


@pytest.mark.parametrize("outcome", ["message", "deferred"])
@pytest.mark.parametrize("version", [0, 1])
def test_receipt_mode_defers_only_outgoing_log_until_transport_confirmation(tmp_path, outcome, version):
    captured = {}

    class Agent:
        def handle_task(self, task):
            captured.update(task)
            # The durable terminal is the authority the Host reads back; the envelope alone never answers.
            write_task_result(tmp_path / "data", task["id"], "completed", metadata=task["metadata"], result="The reply")
            return [{"type": "presence_result", "outcome": outcome, "text": "The reply", "work_ref": "work-1"}]

    result = run_presence_turn(
        admission=_admission(), event=replace(_event(), delivery_reporting_version=version),
        repo_dir=tmp_path / "repo", drive_root=tmp_path / "data",
        agent_factory=lambda **_kwargs: Agent(), gate=PresenceTurnGate(2),
    )
    rows = [json.loads(line) for line in (tmp_path / "data/logs/chat.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result.text == "The reply" and result.delivery_reporting_version == version
    assert captured["metadata"]["presence"]["delivery_reporting_version"] == version
    assert [row["direction"] for row in rows] == (["in", "out"] if version == 0 else ["in"])
    if version == 0:
        assert rows[1]["transport"]["delivery"]["state"] == "authored"


@pytest.mark.parametrize("original", [None, 0, 1])
def test_retry_echoes_original_mode_instead_of_new_request(tmp_path, original):
    import ouroboros.presence_runner as runner

    event = replace(_event(), delivery_reporting_version=1)
    task_id = runner._task_id(_admission(), event)
    presence = {} if original is None else {"delivery_reporting_version": original}
    atomic_write_json(tmp_path / "task_results" / f"{task_id}.json", {
        "_schema_version": 1, "task_id": task_id, "status": "completed", "result": "Old reply",
        "metadata": {"presence": {
            **presence, "binding_id": _admission().binding_id, "observed_text": event.text,
            "event": {"source_event_id": event.source_event_id, "provider": event.provider,
                      "account_id": event.account_id, "conversation_id": event.conversation_id,
                      "thread_id": event.thread_id, "actor": dict(event.actor)},
        }, "presence_outcome": "message", "presence_result_text": "Old reply"},
    })

    def no_reexecution(**_kwargs):
        raise AssertionError("A retry must not run the model again")

    result = run_presence_turn(admission=_admission(), event=event, repo_dir=tmp_path / "repo",
                               drive_root=tmp_path, agent_factory=no_reexecution, gate=PresenceTurnGate(2))
    assert result.delivery_reporting_version == (original or 0)
    assert result.text == "Old reply"
    assert not (tmp_path / "logs/chat.jsonl").exists()
