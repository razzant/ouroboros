"""Peer provenance and the flat continuation facts survive the actual outer delivery projections."""
from types import SimpleNamespace
import json
import queue

import pytest

from ouroboros.delegate_interactions import SAME_SESSION_CONTINUATION


@pytest.mark.parametrize("budget", [15000, 1800, 700])
def test_outer_wake_spill_preserves_continuation(tmp_path, monkeypatch, budget):
    from ouroboros.delegate_supervision import _render_wake_payload
    import ouroboros.tool_capabilities as caps

    monkeypatch.setattr(caps, "tool_result_limit", lambda name: budget)
    ctx = SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path,
                          task_id="participant", task_metadata={}, event_queue=queue.Queue())
    payload = {"status": "waiting_on_user", "run_id": "run-peer", "state": "running",
               "supervision_wake_id": "wake-peer", **SAME_SESSION_CONTINUATION,
               "coordination_context": {"observed_at": "now", "root_task_id": "root",
                                        "large": "a" * 25000},
               "wake_events": [{"type": "addressed_message", "text": "x" * 25000}]}
    out = json.loads(_render_wake_payload(ctx, payload).text)
    assert out["wake_delivery"]["complete"] is False
    assert {key: out[key] for key in SAME_SESSION_CONTINUATION} == SAME_SESSION_CONTINUATION
    payload.update(status="terminal", continuation="new_physical_run")
    payload.pop("continuation_note")
    out = json.loads(_render_wake_payload(ctx, payload).text)
    assert out["continuation"] == "new_physical_run"
    assert "continuation_note" not in out


@pytest.mark.parametrize("advanced", [False, True])
def test_rewait_window_preserves_same_session_fact(advanced):
    from ouroboros.delegate_progress import WindowObservations, window_payload

    seen = WindowObservations()
    if advanced:
        seen.record({}, seq=1, at_sec=0)
    kwargs = dict(run_id="run-peer", state="running", last_seq=1, window=2,
                  elapsed_seconds=2, max_seconds=30, detail={}, seen=seen, budget=15000,
                  pending_interactions=[])
    out = window_payload(waiting_on_user=True, **kwargs)
    assert out["continuation"] == SAME_SESSION_CONTINUATION["continuation"] == "same_session"
    out = window_payload(waiting_on_user=False, **kwargs)
    assert "continuation" not in out and "continuation_note" not in out


def test_supervisor_preserves_peer_relation_for_live_and_durable_events(tmp_path, monkeypatch):
    from supervisor.telemetry_events import _handle_task_message_injected
    import supervisor.log_addressing as addressing

    recorded, pushed = [], []
    monkeypatch.setattr(addressing, "address_ctx_event", lambda ctx, row: row)
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path,
                          append_jsonl=lambda path, row: recorded.append(row),
                          bridge=SimpleNamespace(push_log=pushed.append))
    _handle_task_message_injected({"task_id": "b", "source_task_id": "a",
                                  "provenance": "peer_task", "relation": "sibling",
                                  "text_preview": "interim"}, ctx)
    assert recorded[0] == pushed[0]
    assert recorded[0]["relation"] == "sibling"
    assert recorded[0]["source_task_id"] == "a"


def test_addressed_wakes_carry_the_peer_relation_the_drain_projected(tmp_path):
    """A child's contribution to its parent (S29: C → P, relation ``parent``) and a
    sibling's across-message wake the parked supervised wait WITH the sender's typed
    place beside provenance and id, so a wake never signs a peer as an ancestor or
    owner; an ancestor's steering carries no relation, as the drain wrote none."""
    from ouroboros.delegate_supervision import _addressed_wakes
    from ouroboros.owner_mailbox import PROVENANCE_PEER_TASK, write_task_message

    assert write_task_message(tmp_path, "C's original", "parent-p", source_task_id="child-c",
                              provenance=PROVENANCE_PEER_TASK, relation="parent", msg_id="c-1", attention_kind="question")
    assert write_task_message(tmp_path, "across", "parent-p", source_task_id="sib-s",
                              provenance=PROVENANCE_PEER_TASK, relation="sibling", msg_id="s-1", attention_kind="question")
    assert write_task_message(tmp_path, "steer", "parent-p", source_task_id="root-r",
                              provenance="ancestor_task", msg_id="a-1", attention_kind="question")
    ctx = SimpleNamespace(task_id="parent-p", task_attempt=1, drive_root=tmp_path,
                          budget_drive_root=str(tmp_path), task_metadata={})

    wakes = {row["msg_id"]: row for row in _addressed_wakes(ctx, {})}

    assert wakes["c-1"] == {
        "type": "addressed_message", "msg_id": "c-1", "kind": "task_message",
        "provenance": "peer_task", "source_task_id": "child-c", "relayed_from_task_id": "",
        "relation": "parent", "text": "C's original", "ts": wakes["c-1"]["ts"],
    }
    assert (wakes["s-1"]["provenance"], wakes["s-1"]["relation"]) == ("peer_task", "sibling")
    assert wakes["a-1"]["provenance"] == "ancestor_task" and "relation" not in wakes["a-1"]


def test_outer_wake_spill_keeps_sender_attribution_in_the_reduced_projection(tmp_path, monkeypatch):
    """When the exact wake spills to an artifact, the fitted summaries still name WHO
    wrote each addressed message and in what place (provenance, source, relayed-from
    identity, peer relation) beside the cut text; empty facts stay out."""
    from ouroboros.delegate_supervision import _render_wake_payload, _wake_event_summary
    import ouroboros.tool_capabilities as caps

    monkeypatch.setattr(caps, "tool_result_limit", lambda name: 3000)
    ctx = SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path,
                          task_id="parent-p", task_metadata={}, event_queue=queue.Queue())
    child = {"type": "addressed_message", "msg_id": "c-1", "kind": "task_message",
             "provenance": "peer_task", "source_task_id": "child-c", "relayed_from_task_id": "",
             "relation": "parent", "text": "C_ORIGINAL " + "x" * 5000, "ts": "t"}
    relayed = {"type": "addressed_message", "msg_id": "r-1", "kind": "task_message",
               "provenance": "peer_via_ancestor", "source_task_id": "root-r",
               "relayed_from_task_id": "cousin-q", "text": "relayed", "ts": "t"}
    payload = {"status": "no_progress", "run_id": "run-peer", "state": "running",
               "supervision_wake_id": "wake-peer", "wake_events": [child, relayed]}

    out = json.loads(_render_wake_payload(ctx, payload).text)

    assert out["wake_delivery"]["complete"] is False and out["wake_delivery"]["wake_events_omitted"] == 0
    [c, r] = out["wake_events"]
    assert {key: c[key] for key in ("msg_id", "provenance", "source_task_id", "relation")} == {
        "msg_id": "c-1", "provenance": "peer_task", "source_task_id": "child-c", "relation": "parent"}
    assert c["text"] == child["text"][:600] and c["text_omitted_chars"] == len(child["text"]) - 600
    assert (r["provenance"], r["source_task_id"], r["relayed_from_task_id"]) == (
        "peer_via_ancestor", "root-r", "cousin-q")
    assert "relation" not in r and "relayed_from_task_id" not in c
    assert _wake_event_summary(child)["relation"] == "parent"
