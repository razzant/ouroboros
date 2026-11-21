"""A root's closing sentence survives the real completion pipeline without renaming its work."""
import pytest

from ouroboros.task_finalization import terminal_result_fields
from ouroboros.tools.control_runtime import _finish_task
from tests import test_finish_silent_wake as quiet

wake = quiet.wake
_no_post_task_model = quiet._no_post_task_model


@pytest.mark.parametrize(("action", "metadata", "direct"), [
    pytest.param("finish", {}, True, id='finish-metadata0-True'),
    pytest.param("finish", {**quiet.SCHEDULE, "presence": {"binding_id": "b1"}}, False, id='finish-metadata2-False'),
    pytest.param("stop", {}, False, id='stop-metadata1-False'),
])
def test_root_may_close_with_a_sentence_without_gaining_silent_authority(wake, metadata, direct, action):
    wake.ctx.task_id, wake.ctx.is_direct_chat, wake.ctx.task_metadata = quiet.TASK_ID, direct, metadata
    assert "COMPLETION_ARGUMENT" in _finish_task(wake.ctx, action="finish", silent=True)
    result = _finish_task(wake.ctx, action=action, answer="Full answer.", rationale="Remaining work.", summary="  Итог\n готов. ")
    assert "completion_requested" in result
    assert wake.ctx._completion_request["wake_summary"] == "Итог\n готов."


def test_an_unnamed_ordinary_root_is_not_renamed_by_its_sentence():
    fields = terminal_result_fields({"wake_summary": "The export now works."})
    assert fields["wake_summary"] == "The export now works."
    assert "suggested_name" not in fields


@pytest.mark.parametrize("title", ["", "Fix the export"])
@pytest.mark.parametrize("summary", ["Экспорт исправлен.", "Экспорт исправлен.\n\nПуть: /tmp/export.csv; 42 строки. Проверка за владельцем."])
@pytest.mark.parametrize("metadata,contract", [({}, {}), (quiet.SCHEDULE, {}),
    ({**quiet.SCHEDULE, "presence": {"binding_id": "b1"}}, {}), (quiet.SCHEDULE, {"capability_ceiling": {}})],
                         ids=["owner", "schedule", "bound-schedule", "presence-ceiling"])
def test_root_sentence_names_only_an_unnamed_self_started_run(wake, title, metadata, contract, summary):
    answer = "Full answer.\n\nThe test results."
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
    assert (stored.get("suggested_name") == "Экспорт исправлен.") == (metadata == quiet.SCHEDULE and not contract and not title)
    if title:
        assert stored["title"] == title


@pytest.mark.parametrize("fold", [True, False, None], ids=["fold", "false", "absent"])
def test_a_finish_may_fold_what_it_read_into_its_sentence_and_the_flag_rides_the_sentence(wake, fold):
    """S5: ``fold_read=true`` rides the closing sentence's own path into the stored result; false or absent leaves
    no flag, and the sentence stands either way."""
    wake.ctx.is_direct_chat = False
    answer, summary = "Read the thread; nothing is owed there.", "Прочитал тред: ничего не должен."
    extra = {} if fold is None else {"fold_read": fold}
    text, usage, trace, bus = wake.run([quiet.finish(answer, summary=summary, **extra)])
    _, stored = quiet.emit(wake, text, usage, trace, bus=bus)
    assert stored["wake_summary"] == summary and stored["result"] == answer
    assert stored.get("fold_read") is (True if fold else None)
    assert terminal_result_fields({"wake_summary": summary, "fold_read": True})["fold_read"] is True
    assert "fold_read" not in terminal_result_fields({"fold_read": True})  # no sentence, nothing to stand for


def test_a_fold_without_its_sentence_is_refused_and_stages_nothing(wake):
    wake.ctx.task_id, wake.ctx.is_direct_chat, wake.ctx.task_metadata = quiet.TASK_ID, False, {}
    refused = _finish_task(wake.ctx, action="finish", answer="Full answer.", fold_read=True)
    assert "COMPLETION_ARGUMENT" in refused and "give the gist" in refused, refused
    assert not getattr(wake.ctx, "_completion_request", None)
    staged = _finish_task(wake.ctx, action="finish", answer="Full answer.", summary="Read it.", fold_read=True)
    assert "completion_requested" in staged and wake.ctx._completion_request["fold_read"] is True


@pytest.mark.parametrize("text,opening", [
    ("One. Two. Three.", "One."), ("Что? Потом!", "Что?"),
    ("好。 下一句！", "好。"), ("Да！ Потом", "Да！"),
    ("Да！？ Потом", "Да！？"), ("No punctuation\nStill the opening", "No punctuation\nStill the opening"),
    ("No.space!Still one", "No.space!Still one"), ("", ""),
    ("x" * 5000 + ". Detail.", "x" * 5000 + "."),
])
def test_first_sentence_keeps_the_page_boundary_without_a_length_cut(text, opening):
    from ouroboros.text_sentences import first_sentence
    from ouroboros.memory_view_work import first_sentence as memory_first_sentence

    assert first_sentence(text) == opening
    assert memory_first_sentence is first_sentence
