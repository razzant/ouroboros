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
@pytest.mark.parametrize("metadata,contract", [({}, {}), (quiet.SCHEDULE, {}),
    ({**quiet.SCHEDULE, "presence": {"binding_id": "b1"}}, {}), (quiet.SCHEDULE, {"capability_ceiling": {}})],
                         ids=["owner", "schedule", "bound-schedule", "presence-ceiling"])
def test_root_sentence_names_only_an_unnamed_self_started_run(wake, title, metadata, contract):
    answer, summary = "Full answer.\n\nThe test results.", "Экспорт исправлен."
    wake.ctx.is_direct_chat = False
    if contract:
        from ouroboros.presence_authority import presence_ceiling_payload
        from tests.test_presence_runner import _admission
        contract = {"capability_ceiling": presence_ceiling_payload(_admission().capability_ceiling)}
    wake.ctx.task_contract = contract
    text, usage, trace, bus = wake.run([quiet.finish(answer, summary=summary)], metadata=metadata)
    task = {"id": quiet.TASK_ID, "type": "task", "chat_id": 1, "text": "Fix the export", "metadata": metadata, "title": title, "task_contract": contract}
    _, stored = quiet.emit(wake, text, usage, trace, bus=bus, task=task)
    assert stored["wake_summary"] == summary and stored["result"] == answer
    assert (stored.get("suggested_name") == summary) == (metadata == quiet.SCHEDULE and not contract and not title)
