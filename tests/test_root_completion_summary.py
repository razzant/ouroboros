"""A root's closing sentence survives the real completion pipeline without renaming its work."""
import pytest

from ouroboros.task_finalization import terminal_result_fields
from ouroboros.tools.control_runtime import _finish_task
from tests import test_finish_silent_wake as quiet

wake = quiet.wake
_no_post_task_model = quiet._no_post_task_model


@pytest.mark.parametrize("metadata,direct", [({}, True), ({}, False),
    ({**quiet.SCHEDULE, "presence": {"binding_id": "b1"}}, False)])
@pytest.mark.parametrize("action", ["finish", "stop"])
def test_root_may_close_with_a_sentence_without_gaining_silent_authority(wake, metadata, direct, action):
    wake.ctx.task_id, wake.ctx.is_direct_chat, wake.ctx.task_metadata = quiet.TASK_ID, direct, metadata
    assert "COMPLETION_ARGUMENT" in _finish_task(wake.ctx, action="finish", silent=True)
    result = _finish_task(wake.ctx, action=action, answer="Full answer.", rationale="Remaining work.", summary="  Итог\n готов. ")
    assert "completion_requested" in result
    assert wake.ctx._completion_request["wake_summary"] == "Итог готов."


def test_an_unnamed_ordinary_root_is_not_renamed_by_its_sentence():
    fields = terminal_result_fields({"wake_summary": "The export now works."})
    assert fields["wake_summary"] == "The export now works."
    assert "suggested_name" not in fields


@pytest.mark.parametrize("title", ["", "Fix the export"])
def test_owner_root_sentence_reaches_storage_without_replacing_its_answer_or_title(wake, title):
    answer, summary = "Full answer.\n\nThe test results.", "Экспорт исправлен."
    text, usage, trace, bus = wake.run([quiet.finish(answer, summary=summary)], metadata={})
    task = {"id": quiet.TASK_ID, "type": "task", "chat_id": 1, "text": "Fix the export", "metadata": {}, "title": title}
    _, stored = quiet.emit(wake, text, usage, trace, bus=bus, task=task)
    assert stored["wake_summary"] == summary and stored["result"] == answer
    assert stored.get("suggested_name") != summary
