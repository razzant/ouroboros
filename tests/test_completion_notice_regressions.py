"""Real producer boundaries for the review findings and Telegram's second status path; an older
task's author closure still renders by the host trigger its record carries."""
import asyncio
import copy


from ouroboros import terminal_projection
from tests.test_completion_notices import _record
from tests.test_project_completion_mirror import _enqueue_row
from tests.test_telegram_subagent_cards import _setup, _Rec


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


def test_disabled_status_mirror_keeps_preserved_intermediate_content(tmp_path, monkeypatch):
    plugin, api = _setup(tmp_path, monkeypatch, TELEGRAM_NOTIFY_TASKS="off")
    record = _record("receipt")
    record.update(terminal_origin="host_salvage", result="Preserved intermediate output (not a final answer): useful draft")
    event = _enqueue_row(tmp_path / "host", monkeypatch, record)
    salvage = event["progress_meta"]["completion_notice"]["salvage"]
    assert "useful draft" in salvage
    asyncio.run(plugin._make_outbound(api)({**event, **event["progress_meta"]}))
    assert [row[1] for row in _Rec.sent] == [salvage]
