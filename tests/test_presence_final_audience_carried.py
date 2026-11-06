"""#2634's audience section for a root promoted from a CARRIED Presence turn (round-2 item 9).

A carried author's task is the one ``run_presence_turn`` builds in its carrier from the Host's admitted facts (the same
``_build_task``), and its ``start_task`` reaches the supervisor through the carrier's own task endpoint (the
conversation lane's ``log_addressing.task_event_queue``, which stamps a lane turn's payloads by value) instead of the
in-server author's queue. The root it promotes keeps the place conversation it answers: its first request names that
audience, and so does its cold rebuild from the persisted (JSON, underscore-free) task.
"""
from __future__ import annotations

import json
import types


from tests.test_presence_final_audience import THREAD, _section, _source_turn
from tests.test_presence_promises import fired_context as fired_context



def test_a_root_promoted_from_a_carried_turn_names_its_conversation_and_keeps_it_cold(fired_context, monkeypatch):
    import supervisor.queue as queue
    import supervisor.workers as workers
    from ouroboros.place_reply import reply_carrier
    from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION
    from ouroboros.tools.control_routing import _start_task
    from supervisor.events import _handle_promote_chat_to_task
    from supervisor.log_addressing import task_event_queue
    from tests.test_promote_chat_flow import _confirm_promote

    root, _pending, first_request = fired_context
    registry, turn = _source_turn(root, THREAD)
    monkeypatch.setattr(workers, "DRIVE_ROOT", root)
    monkeypatch.setattr(queue, "DRIVE_ROOT", str(root))
    _confirm_promote(monkeypatch)
    pending, crossed = [], []
    handler = types.SimpleNamespace(
        DRIVE_ROOT=root, WORKERS={0: types.SimpleNamespace()}, PENDING=pending, bridge=None,
        append_jsonl=lambda *_a, **_k: None, persist_queue_snapshot=lambda **_k: True,
        enqueue_task=lambda task: pending.append(dict(task)) or pending[-1], load_state=lambda: {"owner_chat_id": 1})

    def supervisor_side(event):  # what reaches the server from the carrier, by value
        crossed.append(json.loads(json.dumps(event)))
        _handle_promote_chat_to_task(crossed[-1], handler)

    carried = {**turn, "_execution_owner": {"kind": "presence", "lane": LANE_CONVERSATION, "executor": EXECUTOR_WORKER}}
    registry._ctx.event_queue = task_event_queue(types.SimpleNamespace(put=supervisor_side, put_nowait=supervisor_side),
                                                 carried)
    out = _start_task(registry._ctx, "Compile the requested report", title="Report", destination="place")
    assert out.startswith("OK: task"), out
    [promoted] = pending
    assert promoted["metadata"]["presence"]["event"] == turn["metadata"]["presence"]["event"]
    assert not promoted.get("_presence_turn") and reply_carrier(promoted["metadata"])
    assert _section(first_request(promoted)) == 1
    cold = {key: value for key, value in json.loads(json.dumps(promoted)).items() if not key.startswith("_")}
    assert _section(first_request(cold)) == 1
