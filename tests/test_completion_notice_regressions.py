"""Real producer boundaries for the review findings and Telegram's second status path; an older
task's author closure still renders by the host trigger its record carries."""
import asyncio
import copy

import pytest

from ouroboros import project_dialogue, terminal_projection
from tests.test_completion_notices import _record
from tests.test_project_completion_mirror import _enqueue_row
from tests.test_telegram_subagent_cards import _setup, _Rec


@pytest.mark.parametrize("mode", ["auto", "required"])
@pytest.mark.parametrize("advisory", [False, True])
def test_an_old_author_closure_renders_by_its_recorded_host_trigger(mode, advisory):
    # The trigger the host acceptance request (gone with task acceptance) recorded on older review axes.
    trigger = f"{mode}_agent_request_after_agent_{'advisory' if advisory else 'tool'}"
    record = _record("author-close")
    axes = record["outcome_axes"]
    axes["objective"] = {"status": "pass", "source": "author_acceptance"}
    axes["review"] = {"status": "degraded", "trigger": trigger, "acceptance_decision": {
        "status": "finalized_unaccepted", "reason": "author_finish", "enforcement": "advisory"}}
    assert terminal_projection.completion_notice(record)["neutral"] is (mode == "auto")
    assert "had not signed it off" in project_dialogue._completion_verdict(record, {})


def test_a_rail_keeps_its_cause_and_the_notice_explains_current_debt():
    """#1583: the debt is the row's current list beside the rail, never its reason."""
    result = _record("rail")
    result["reason_code"] = "round_limit"
    result["outcome_axes"]["execution"].update(status="best_effort", reason_code="round_limit")
    result["delegated_runs_unreconciled"] = ["run-1"]
    before = copy.deepcopy(result)
    row = terminal_projection._project_row("rail", result, {}, {"chat_id": 1})
    assert "round limit" in row["completion_notice"]["detail"]
    assert "delegated work was never reconciled" in row["completion_notice"]["detail"]
    assert result == before
    result["delegated_runs_unreconciled"] = []
    assert "delegated work" not in terminal_projection.completion_notice(result)["detail"]


@pytest.mark.parametrize("notify", ["off", "on"])
@pytest.mark.parametrize("case", ["receipt", "verification_output"])
@pytest.mark.parametrize("silent", ["off", "on"])
def test_project_status_producer_through_bus_to_telegram(notify, case, silent, tmp_path, monkeypatch):
    from supervisor import message_bus

    plugin, api = _setup(tmp_path, monkeypatch, TELEGRAM_NOTIFY_TASKS=notify, TELEGRAM_SILENT_MODE=silent)
    handle = plugin._make_outbound(api)
    record = _record(case)
    record.update(terminal_origin="model_final", result="The complete authored answer.")
    event = _enqueue_row(tmp_path / "host", monkeypatch, record)
    original = copy.deepcopy(event)
    published, frames = [], []
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path / "host")

    def publish(_topic, row):
        published.append(row)
        asyncio.run(handle(row))

    monkeypatch.setattr(message_bus, "publish_event", publish)
    # The original Project answer is an independent, existing outbound event.
    bridge.send_message(1, "The complete authored answer.", role="assistant", task_id="root-project")
    bridge.send_message(1, event["text"], task_id="root-project", role=event["role"],
                        system_type=event["system_type"], progress_meta=event["progress_meta"])
    assert event == original and "earlier version" in event["text"]
    assert len(_Rec.sent) == (1 if notify == "off" else 2)
    assert not _Rec.edited  # a System summary never joins the reply-edit chain
    assert frames[-1]["content"] == event["text"]  # Main is not quieted by a Telegram preference
    assert published[-1]["completion_notice"] == event["progress_meta"]["completion_notice"]
    assert _Rec.sent[0][1] == "The complete authored answer."
    if notify == "on":
        notice = _Rec.sent[-1][1]
        assert "earlier version" not in notice and "No host verification receipt" not in notice
        if case == "receipt":
            assert "finished" in notice and "⚠" not in notice and "✅" not in notice
        else:
            assert "A tool this task used failed" in notice


def test_disabled_status_mirror_keeps_preserved_intermediate_content(tmp_path, monkeypatch):
    plugin, api = _setup(tmp_path, monkeypatch, TELEGRAM_NOTIFY_TASKS="off")
    record = _record("receipt")
    record.update(terminal_origin="host_salvage", result="Preserved intermediate output (not a final answer): useful draft")
    event = _enqueue_row(tmp_path / "host", monkeypatch, record)
    salvage = event["progress_meta"]["completion_notice"]["salvage"]
    assert "useful draft" in salvage
    asyncio.run(plugin._make_outbound(api)({**event, **event["progress_meta"]}))
    assert [row[1] for row in _Rec.sent] == [salvage]
