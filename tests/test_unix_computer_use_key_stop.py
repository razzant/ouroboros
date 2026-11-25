"""unix_computer_use ``key``: the host's Stop of the call (PluginAPI 2.6 ``get_stop_state``) before every chord.

The observed failure: after the owner's Stop, a two-part call waited for its first part to answer and then sent the
second. The skill asks before a command that may raise an app and separately before every chord, after preparation.
Only ``continue`` permits the next action. A held key with its release and a drag are one action each and never
ask. A host without the method keeps the earlier contract. Device input is faked; test_extension_call_stop integrates
the actual key with PluginAPI, its host-owned Stop latch and the dispatcher.
"""
from __future__ import annotations

import contextvars

import pytest

from tests._unix_computer_use import CONTINUE, DOWN, KEYNOTE, RETURN, SAFARI, STOPPED, StopAPI, app_dict, inputs, is_input, mac, receipt

NO_TASK = {"status": "continue", "reason": "no_task", "note": "this call was made for no task, which no Stop governs"}
UNREADABLE = {"status": "stop", "reason": "stop_unreadable", "note": "this task's Stop could not be read"}
UNBOUND = {"status": "unbound", "reason": "unbound", "note": "no tool call of this skill runs in this context"}


def stop_host(module, tmp_path, state, *answers) -> StopAPI:
    api = StopAPI(tmp_path / "state", state["calls"], answers)
    module.register(api)
    return api


def stop_reads(state) -> list[int]:
    return [index for index, argv in enumerate(state["calls"]) if argv == ["get_stop_state"]]


def is_helper(argv: list[str]) -> bool:
    return any(part.endswith("macos_native.py") for part in argv)


@pytest.mark.parametrize("app", ["Keynote", ""])
def test_every_chord_asks_last_the_first_one_included(tmp_path, monkeypatch, app):
    """The first chord asks after the switch (targeted) or the front read (untargeted); a later chord asks after its
    own front re-read. Each chord starts right after its reading, never before the first one."""
    module, _plain, state = mac(tmp_path, monkeypatch)
    api = stop_host(module, tmp_path, state)  # every reading: continue
    out = receipt(api.tools["key"](keys="down down return", app=app))
    assert out["ok"] is True and out["steps"] == 3 and len(inputs(state)) == 3, out
    calls, reads = state["calls"], stop_reads(state)
    if app:
        switch, *reads = reads
        assert is_helper(calls[switch + 1]) and "activate" in calls[switch + 1]
    assert len(reads) == 3, "each chord has its own check"
    assert all(is_input(calls[index + 1]) for index in reads), calls
    assert is_helper(calls[reads[0] - 1]) if app else calls[reads[0] - 1][0] == "lsappinfo", calls
    assert all(calls[index - 1][0] == "lsappinfo" for index in reads[1:]), "a later chord asks after its front re-read"
    assert next(i for i, argv in enumerate(calls) if is_input(argv)) > reads[0], "no chord before the first reading"


@pytest.mark.parametrize("answer", [STOPPED, UNREADABLE, UNBOUND])
def test_key_refused_before_the_mac_activation_helper_starts(tmp_path, monkeypatch, answer):
    module, _, state = mac(tmp_path, monkeypatch)
    api = stop_host(module, tmp_path, state, answer)
    out = receipt(api.tools["key"](keys="down return", app=KEYNOTE[0]))
    assert out["status"] == "stopped" and out["dispatched"] is False
    assert (out["steps_done"], out["steps_total"]) == (0, 2)
    assert out["target"]["activated"] is False and answer["reason"] in out["note"]
    assert "observed_effect" not in out and "activation_done" not in out
    assert state["helper"] == [] and inputs(state) == [] and state["front"] == SAFARI
    assert len(stop_reads(state)) == 1


def test_never_policy_only_checks_at_the_chord(tmp_path, monkeypatch):
    module, _, state = mac(tmp_path, monkeypatch, front=KEYNOTE)
    api = stop_host(module, tmp_path, state, STOPPED)
    out = receipt(api.tools["key"](keys="return", app=KEYNOTE[0], raise_policy="never"))
    assert out["status"] == "stopped" and out["target"]["already_front"] is True
    assert [op for op, *_ in state["helper"]] == ["resolve"] and inputs(state) == []
    assert len(stop_reads(state)) == 1


@pytest.mark.parametrize("answer", [STOPPED, UNREADABLE, UNBOUND])
def test_a_stop_before_the_first_chord_sends_nothing_and_a_switch_before_it_keeps_its_mark(tmp_path, monkeypatch, answer):
    """Anything but ``continue`` (``unbound`` included: no call context was proven) withholds the first chord too.
    Untargeted, that is a plain "not sent" with no mark; targeted, the switch that ran before it keeps its own mark."""
    module, _plain, state = mac(tmp_path, monkeypatch)
    api = stop_host(module, tmp_path, state, answer)
    plain = receipt(api.tools["key"](keys="right"))
    assert plain.get("status") == plain["reason"] == "stopped" and "ok" not in plain and "error" not in plain, plain
    assert plain["dispatched"] is False
    assert "observed_effect" not in plain and "activation_done" not in plain and plain["note"].startswith("the input was not sent; ")
    assert answer["reason"] in plain["note"] and answer["note"] in plain["note"] and plain["front"] == app_dict(SAFARI)
    api.answers.extend([CONTINUE, answer])
    switched = receipt(api.tools["key"](keys="right", app="Keynote"))
    assert switched.get("status") == switched["reason"] == "stopped" and "ok" not in switched and "error" not in switched, switched
    assert switched["dispatched"] is False
    assert switched["observed_effect"] is True and switched["activation_done"] is True and switched["target"]["activated"] is True
    assert switched["note"].startswith("the input was not sent; the switch to Keynote ran before it; ")
    assert switched["front"] == app_dict(KEYNOTE) and switched["front_before"] == app_dict(SAFARI)
    assert inputs(state) == [] and len(stop_reads(state)) == 3


def test_a_stop_before_the_first_chord_of_a_sequence_sends_none_of_it(tmp_path, monkeypatch):
    module, _plain, state = mac(tmp_path, monkeypatch)
    api = stop_host(module, tmp_path, state, CONTINUE, STOPPED)
    out = receipt(api.tools["key"](keys="down down return", app="Keynote"))
    assert out.get("status") == out["reason"] == "stopped" and "ok" not in out and "error" not in out, out
    assert out["dispatched"] is False
    assert out["steps_done"] == 0 and out["steps_total"] == 3 and "'down' not sent" in out["note"]
    assert out["observed_effect"] is True and out["activation_done"] is True, "the switch ran before the first chord"
    assert "the input was not sent" in out["note"] and "the switch to Keynote ran before it" in out["note"]
    assert inputs(state) == [] and len(stop_reads(state)) == 2


@pytest.mark.parametrize("answer", [STOPPED, UNBOUND])
def test_a_stop_between_chords_sends_no_further_chord_and_keeps_the_sent_ones_receipt(tmp_path, monkeypatch, answer):
    """The observed failure: a call kept sending its next part after Stop. The chord already sent keeps its mark."""
    module, _plain, state = mac(tmp_path, monkeypatch)
    api = stop_host(module, tmp_path, state, CONTINUE, CONTINUE, answer)
    out = receipt(api.tools["key"](keys="down down return", app="Keynote"))
    assert out.get("status") == out["reason"] == "stopped" and "ok" not in out and "error" not in out, out
    assert out["dispatched"] is False
    assert out["steps_done"] == 1 and out["steps_total"] == 3 and out["observed_effect"] is True
    assert answer["reason"] in out["note"] and answer["note"] in out["note"] and "'down' not sent" in out["note"]
    assert inputs(state) == [[DOWN]], "only the chord before the refusal was sent"
    assert len(stop_reads(state)) == 3 and out["activation_done"] is True
    assert "the next step was not sent" in out["note"] and "Keynote" in out["note"]


def test_a_stop_of_one_task_leaves_another_tasks_call_of_the_same_skill_whole(tmp_path, monkeypatch):
    """One registered skill serves the calls of several tasks and the host answers each call for its own task. The
    skill keeps no Stop of its own between calls, so task A's refusal never withholds a chord of task B."""
    task: contextvars.ContextVar[str] = contextvars.ContextVar("task")
    module, _plain, state = mac(tmp_path, monkeypatch)

    class PerCallHost(StopAPI):
        def get_stop_state(self) -> dict:
            self.calls.append(["get_stop_state", task.get()])
            return dict(STOPPED if task.get() == "A" else CONTINUE)

    api = PerCallHost(tmp_path / "state", state["calls"], ())
    module.register(api)

    def call(name: str) -> dict:
        task.set(name)
        return receipt(api.tools["key"](keys="down return", app="Keynote"))

    stopped = contextvars.copy_context().run(call, "A")
    whole = contextvars.copy_context().run(call, "B")
    assert stopped.get("status") == stopped["reason"] == "stopped" and "ok" not in stopped and "error" not in stopped, stopped
    assert stopped["steps_done"] == 0
    assert whole["ok"] is True and whole["steps"] == 2, whole
    assert inputs(state) == [[DOWN], [RETURN]], "only B's chords were sent"
    assert [argv for argv in state["calls"] if argv[0] == "get_stop_state"] == [
        ["get_stop_state", "A"], ["get_stop_state", "B"], ["get_stop_state", "B"], ["get_stop_state", "B"]]


def test_a_call_for_no_task_and_an_older_host_keep_the_whole_sequence(tmp_path, monkeypatch):
    """``continue`` with ``no_task`` (a legitimate call no task's Stop governs) lets every chord start. A host without
    the method (this skill declares plugin_api 2.0) keeps the earlier contract: no reading and no refusal."""
    module, plain, state = mac(tmp_path, monkeypatch)
    assert receipt(plain.tools["key"](keys="down down"))["steps"] == 2 and stop_reads(state) == []
    api = stop_host(module, tmp_path, state, NO_TASK, NO_TASK, NO_TASK)
    out = receipt(api.tools["key"](keys="down return", app="Keynote"))
    assert out["ok"] is True and out["steps"] == 2 and len(inputs(state)) == 4 and len(stop_reads(state)) == 3, out


def test_one_action_is_never_split(tmp_path, monkeypatch):
    """A held key with its release and a drag are one action each: they never ask, so a Stop never cuts one."""
    module, _plain, state = mac(tmp_path, monkeypatch)
    api = stop_host(module, tmp_path, state, STOPPED, STOPPED)
    assert receipt(api.tools["hold_key"](keys="shift", duration_ms=50))["ok"] is True, "a held key keeps its release"
    assert receipt(api.tools["left_click_drag"](start_x=1, start_y=2, end_x=30, end_y=40, raw=True))["ok"] is True
    assert stop_reads(state) == [] and api.answers == [STOPPED, STOPPED], "press/release and a drag asked nothing"
