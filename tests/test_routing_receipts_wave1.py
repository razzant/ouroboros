"""The routing rail tells the truth about what it did (14.09 incident).

A Project root steered the Main root; the supervisor wrote a refusal receipt and
the model was told `STEER_UNCONFIRMED`. Four separate lies produced that one
sentence, and each is pinned here against the PRODUCER's own output:

* the tool waited on an EMPTY receipt id, so a task-authored steer could never be
  confirmed — refused or delivered;
* the wait's accept-set had no `rejected`, so a refusal the host had already
  written durably was polled to the deadline and reported as a timeout;
* the promote/route refusal sentences were not in the identifier register (and
  the promotion ones carried no marker at all), so a routing act that scheduled
  nothing counted as a SUCCESSFUL tool call in the outcome classifier;
* an admitted promote named the project it ASKED for rather than the one
  admission put the task in.

Plus the sibling projection (#902): after an ADOPTING conversion the message's
other live cards still offered a second "turn into project".
"""

from __future__ import annotations

import types



def _tool_ctx(tmp_path, metadata=None, **kwargs):
    """A routing turn with no supervisor: events land in ``pending_events``."""
    return types.SimpleNamespace(
        pending_events=[],
        event_queue=None,
        current_chat_id=1,
        drive_root=tmp_path,
        task_id="turn-1",
        task_metadata=dict(metadata or {}),
        last_owner_delivery=None,
        **kwargs,
    )


# --- (b) a task-authored steer has a receipt of its own -----------------------


def test_a_resumed_turn_keeps_its_own_unconfirmed_acts(tmp_path):
    """A cold continuation (an owner wait across a restart) restores the exact ids, so the
    resumed turn still reports a late outcome; a state saved before the key restores none."""
    import json

    from ouroboros.owner_wait import continuation_state, restore_continuation_state

    act = {"action": "steer_task", "client_message_id": "agent-steer:tok", "routing_token": "tok", "target": "t-1"}
    state = json.loads(json.dumps(continuation_state(
        types.SimpleNamespace(task_id="turn-1", unconfirmed_routing_acts=[act]), [], {}, {}, 1, [], set())))
    restored = types.SimpleNamespace(task_id="turn-1", unconfirmed_routing_acts=[])  # the resumed turn's own context
    restore_continuation_state(types.SimpleNamespace(_ctx=restored), state, [], {}, {}, set())
    assert restored.unconfirmed_routing_acts == [act]
    state.pop("unconfirmed_routing_acts")
    restore_continuation_state(types.SimpleNamespace(_ctx=restored), state, [], {}, {}, set())
    assert restored.unconfirmed_routing_acts == []


# --- (d) an admitted promote names where the task actually landed -------------


def test_an_unconfirmed_promote_names_what_was_requested_as_a_request(tmp_path, monkeypatch):
    """No receipt means no known destination, so the sentence says what was ASKED
    for and that the outcome is unknown — never a room the task may not be in."""
    from ouroboros.tools import control_events
    from ouroboros.tools.control import _start_task

    monkeypatch.setattr(
        control_events, "_wait_for_promotion_admission",
        lambda *_a, **_k: {"status": "unconfirmed", "reason": "confirmation_timeout"},
    )

    out = _start_task(
        _tool_ctx(tmp_path), "Audit the issue", title="Issue audit", project_name="Dinosaurs",

    )

    assert out.startswith("Not confirmed yet:")
    assert "admission to new project 'Dinosaurs'" in out
    assert "is pending" in out


# --- (e) a relayed owner steer refused by a pending cancel: receipt, no bubble ---
