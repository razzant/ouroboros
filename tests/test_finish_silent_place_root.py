"""A root answering its own place conversation may finish silently; binding authority alone may not."""

import pytest

from tests.test_finish_silent_wake import SCHEDULE, TASK_ID, wake  # noqa: F401 -- the shared wake context

PLACE_SPEAKER = {"binding_id": "b1", "place_id": "place-1", "transport_skill": "bridge",
                 "event": {"conversation_key": "chat:a:r", "source_event_id": "event1"}}


@pytest.mark.parametrize("metadata", [
    {"presence": PLACE_SPEAKER, **SCHEDULE},  # a follow-up root that kept its conversation's speaker carrier
    {"presence": PLACE_SPEAKER},  # a root promoted from a place conversation
], ids=["scheduled-place-root", "promoted-place-root"])
def test_a_root_answering_its_own_place_conversation_may_finish_silently(wake, metadata):  # noqa: F811
    from ouroboros.tools.control_runtime import _finish_task

    wake.ctx.is_direct_chat = False
    wake.ctx.task_id, wake.ctx.parent_task_id = TASK_ID, None
    wake.ctx.task_metadata = dict(metadata)
    assert "completion_requested" in _finish_task(wake.ctx, action="finish", answer="Sent the file.", silent=True)
    assert wake.ctx._completion_request["wake_silent"] is True
    assert wake.ctx._completion_request["answer"] == "Sent the file."
    wake.ctx._completion_request = None
    # The one sentence for a human stays a self-started run's; silence never needs it.
    assert "COMPLETION_ARGUMENT" in _finish_task(wake.ctx, action="finish", silent=True, summary="A line")
    assert getattr(wake.ctx, "_completion_request", None) is None


@pytest.mark.parametrize("metadata", [
    {"presence_binding_authority": {"binding_id": "b1"}, **SCHEDULE},  # related work acting for a binding
    {"presence_binding_authority": {"binding_id": "b1"}},
    {"presence": {"binding_id": "b1", "event": {"conversation_key": "chat:a:r"}}},  # legacy carrier, no place
], ids=["authority-schedule", "authority-root", "legacy-binding"])
def test_binding_authority_or_a_placeless_carrier_names_no_speaker(wake, metadata):  # noqa: F811
    from ouroboros.tools.control_runtime import _finish_task

    wake.ctx.is_direct_chat = False
    wake.ctx.task_id, wake.ctx.parent_task_id = TASK_ID, None
    wake.ctx.task_metadata = dict(metadata)
    assert "COMPLETION_ARGUMENT" in _finish_task(wake.ctx, action="finish", silent=True)
    assert getattr(wake.ctx, "_completion_request", None) is None
