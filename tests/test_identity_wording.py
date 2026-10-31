def test_live_task_message_marker_uses_my_human_wording(tmp_path):
    # The drained mailbox text (plus its optional surface note) must still go
    # through the owner-marking wrapper before injection.
    import queue
    from types import SimpleNamespace
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.loop_round_limits import _drain_incoming_messages

    messages = []
    assert write_owner_message(tmp_path, "look here", "wording-task", msg_id="owner-wording", client_surface={"channel": "telegram"})
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, "wording-task", None, set(), SimpleNamespace(task_attempt=1))
    assert messages[0]["content"].startswith("[Message from my human]: look here\n[note: sent from client surface:")
    assert "telegram" in messages[0]["content"]
    # Addressed task-tree messages are peer/ancestor/peer-root communication,
    # not owner dialogue, and must never borrow the owner's priority marker.
    # Ask the render ladder itself: every provenance it can frame — including
    # the independent-root prefix a task may now be addressed by — names a
    # TASK, so none of them can be mistaken for my human.
    from ouroboros.owner_mailbox import PROVENANCE_INDEPENDENT_TASK, deliver_task_message

    rendered = {}
    for provenance in (
        "ancestor_task", "descendant_task", "peer_via_ancestor", "system",
        PROVENANCE_INDEPENDENT_TASK, "",
    ):
        lines: list[str] = []
        deliver_task_message(
            {
                "provenance": provenance, "source_task_id": "t-source",
                "relayed_from_task_id": "t-relayed", "text": "body",
            },
            "t-recipient", None, lines.append,
        )
        rendered[provenance] = lines[0]
    assert all("human" not in text for text in rendered.values()), rendered
    assert rendered[PROVENANCE_INDEPENDENT_TASK].startswith(
        "[Message from independent task t-source]")
    assert rendered["ancestor_task"].startswith("[Message from ancestor task t-source]")
    assert rendered[""] == rendered["ancestor_task"], "unknown provenance keeps the tree fallback"
