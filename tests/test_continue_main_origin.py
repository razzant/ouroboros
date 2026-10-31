"""Continue retains canonical owner origin without changing Main-notice policy."""

import copy
import queue as stdqueue

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_owner_continue import NONCE, _interrupted, _owner_mail

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("case", ["main-owner", "main-no-ref", "child", "project-no-ref"])
def test_continued_origin_reaches_main_notice_consumer_without_new_authority(tmp_path, monkeypatch, case):
    from ouroboros import usage_accounting as ua
    from ouroboros.dialogue_provenance import run_origin
    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.owner_continue import owner_sources
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control_runtime import _main_notice_refusal, _send_user_message
    from ouroboros.tools.tool_context import ToolContext
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    project_id, chat_id = "", 1
    if case == "project-no-ref":
        from ouroboros.projects_registry import create_project

        project = create_project(tmp_path, "continued-project", name="Continued project")
        project_id, chat_id = project["id"], int(project["chat_id"])
    has_ref = case in {"main-owner", "child"}
    ref = {"chat_id": chat_id, "client_message_id": "owner-original"} if has_ref else {}
    _interrupted(tmp_path, chat_id=chat_id, project_id=project_id, origin_message_ref=ref)
    _owner_mail(tmp_path)
    before = load_task_result(tmp_path, "pred-1")
    sources = owner_sources(tmp_path, before, "pred-1")
    money = ua.read_usage_records(tmp_path)

    ack = admit_continuation("pred-1", action_nonce=NONCE)

    assert ack["ok"] and not ack["held"], ack
    successor = ack["successor_task_id"]
    row = next(task for task in workers.PENDING if task["id"] == successor)
    stored = load_task_result(tmp_path, successor)
    metadata = copy.deepcopy(stored["metadata"])
    assert metadata == row["metadata"]
    assert metadata["objective_author"] == {"kind": "continuation", "predecessor_task_id": "pred-1"}
    # Held quote handles and open relay return addresses survive Continue (K3/K4).
    assert metadata["continuation"]["owner_sources"] == {
        "original": sources["original"], "schedule": None, "later": sources["later"],
        "held_owner_messages": sources["held_owner_messages"], "open_relays": sources["open_relays"]}
    assert metadata["continuation"]["peer_context"] == sources["peer_context"]
    for key, value in before["billing_group"].items():
        assert metadata["continuation"][key] == value
    assert row["deadline_at"] == before["deadline_at"]
    assert ua.read_usage_records(tmp_path) == money

    if case == "child":
        metadata.update(parent_task_id=successor, delegation_role="subagent")
    events = stdqueue.Queue()
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, current_chat_id=chat_id,
                      task_id="child-of-continue" if case == "child" else successor,
                      task_metadata=metadata, event_queue=events)
    allowed = case in {"main-owner", "project-no-ref"}
    refusal = _main_notice_refusal(ctx, chat_id)
    assert bool(refusal) is not allowed, refusal
    result = _send_user_message(ctx, "A continued-task notice.", destination="main")
    if allowed:
        assert result == "OK: notice sent to the main chat."
        notice = events.get_nowait()
        assert notice["chat_id"] == 1 and notice["system_type"] == "main_notice"
    else:
        assert result.startswith("⚠️ MAIN_NOTICE_BLOCKED:")
        assert events.empty() and not ctx.pending_events
    assert _send_user_message(ctx, "An ordinary update.", destination="current").startswith("OK:")
    current = events.get_nowait()
    assert current["chat_id"] == chat_id and current["system_type"] == "proactive_message"
    if case == "main-owner":
        assert metadata["origin_message_ref"] == ref
        assert metadata["origin_message_text"] == sources["original"]["content"]
        assert run_origin({"metadata": metadata})["owner_ingress"]
    elif not has_ref:
        assert not metadata.get("origin_message_ref") and not metadata.get("origin_suppressed")
        assert not run_origin({"metadata": metadata})["owner_ingress"]
    _initialize_owner_directives(ctx, [{"role": "user", "content": row["text"]}])
    assert [item["content"] for item in ctx._owner_directives] == [
        "Write the Friday report", "Also add the charts", "Use last week's numbers"]
    assert all(item["content"] != row["text"] for item in ctx._owner_directives)
