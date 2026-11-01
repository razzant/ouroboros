"""Stop now reaches an inline Presence turn's running tool wait (R4).

An inline Presence turn's Stop keeps only the durable intent, written for the turn to read (#2012), so an
in-slot ``await_messages`` and a tracked ``run_command`` kept waiting their whole requested time after the
press: 21-24 s of a 25 s wait, while Wrap up (mail) ended the await at once. The owner presses through the real
public route (``gateway.tasks.api_task_cancel``) while a real Place focus waits in each tool; only inference is
scripted (``tests/test_presence_owner_stop.py``). Stop now ends the wait within one poll and kills the command's
whole process group; the next dispatch gate ends the turn with no model call. Wrap up keeps its own semantics:
it ends the await as mail, and a running command finishes with its output before the one final turn.
"""
from __future__ import annotations

import json
import os
import pathlib
import threading
import time

import pytest

from ouroboros import cancel_intents as ci
from ouroboros.platform_layer import pid_provably_gone
from ouroboros.task_results import load_task_result
from tests.test_presence_continuation_bootstrap import install_bootstrap_harness, tool_call
from tests.test_presence_contact_wait import start
from tests.test_presence_owner_contact import contact_runtime as _contact_runtime, focus  # noqa: F401
from tests.test_presence_owner_stop import SUMMARY, owner_stop as _owner_stop

contact_runtime, owner_stop = _contact_runtime, _owner_stop
pytestmark = pytest.mark.serial

WAIT_SEC = 25
PROMPT_SEC = 8  # the press-to-end bound: one poll and the dispatch gate, far below the 25 s wait


# The command is an `sh -c` script reporting its own and a background child's pid ($$, $!): POSIX pids only.
KINDS = ["await", pytest.param("command", marks=pytest.mark.skipif(os.name != "posix", reason="uses sh and its POSIX pids"))]


def _alive(pid):
    return not pid_provably_gone(pid)


def _tool(kind, marks, seconds):
    """The round's one tool call. The command records its shell and a background grandchild, so a kill that
    misses the process group leaves the grandchild alive."""
    if kind == "await":
        return tool_call("await_messages", {"timeout_sec": seconds}, "wait")
    script = (f"sleep {seconds} & echo $! > {marks / 'child.pid'}; echo $$ > {marks / 'shell.pid'}; "
              "echo started; wait; echo finished")
    return tool_call("run_command", {"cmd": ["sh", "-c", script], "timeout_sec": seconds + 30}, "command")


def _in_flight(root, kind, marks, deadline):
    """Wait until the tool is actually waiting: the command's own pid files, or the durable start row of the
    await (``tool_call_started`` is written before submission) plus a moment for its first poll."""
    while time.monotonic() < deadline:
        if kind == "command":
            if (marks / "shell.pid").exists() and (marks / "child.pid").exists():
                return True
        else:
            for path in root.rglob("tools.jsonl"):
                text = path.read_text(encoding="utf-8", errors="replace")
                if '"tool_call_started"' in text and '"await_messages"' in text:
                    time.sleep(1.0)
                    return True
        time.sleep(0.05)
    return False


def _marks(tmp_path):
    """The command's pid files, in the turn's own writable user_files root (the isolated HOME)."""
    marks = pathlib.Path(os.path.expanduser("~")) / f"stop-marks-{tmp_path.name}"
    marks.mkdir(parents=True)
    return marks


def _scripted(monkeypatch, r, kind, marks, press_while_waiting, seconds=WAIT_SEC):
    """Round one asks for the tool and starts the owner's press, which waits until that tool is waiting."""
    from ouroboros.llm_attempt import require_physical_dispatch_window

    h = install_bootstrap_harness(monkeypatch, r.root, review_mode="off")
    generated, inputs, presses = [], [], []

    def model(messages):
        require_physical_dispatch_window()  # the production gate before any physical send
        inputs.append(messages)
        generated.append(len(generated) + 1)
        if len(generated) == 1:
            presses.append(threading.Thread(target=press_while_waiting, args=(h.agents[0].tools._ctx.task_id,),
                                            name="owner-press", daemon=True))
            presses[-1].start()
            return {"content": None, "tool_calls": [_tool(kind, marks, seconds)]}
        return {"content": SUMMARY, "tool_calls": []}

    h.script = model
    h.presses = presses
    return h, generated, inputs


def _tool_rows(root, name):
    rows = []
    for path in root.rglob("tools.jsonl"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            row = json.loads(line)
            if row.get("tool") == name:
                rows.append(row)
    return rows


@pytest.mark.parametrize("kind", KINDS)
def test_stop_now_ends_a_running_tool_wait_promptly_and_kills_the_command_group(owner_stop, monkeypatch,
                                                                                 tmp_path, kind):
    r, press = owner_stop
    marks = _marks(tmp_path)
    pressed = {}

    def stop_now_while_waiting(task_id):
        assert _in_flight(r.root, kind, marks, time.monotonic() + 30), "the tool never started waiting"
        pressed["at"] = time.monotonic()
        pressed["response"] = press(task_id, cascade=True)  # the app's Stop now

    h, generated, _inputs = _scripted(monkeypatch, r, kind, marks, stop_now_while_waiting)
    stopped = start(r, h).result.exception(timeout=WAIT_SEC + 60)
    ended = time.monotonic()
    for thread in h.presses:
        thread.join(35)

    assert pressed["response"].status_code == 202, pressed["response"].text
    assert pressed["response"].json()["stop_policy"] == "immediate"
    assert ended - pressed["at"] < PROMPT_SEC, f"the turn ended {ended - pressed['at']:.1f}s after Stop now"
    assert generated == [1], "no model call after the owner's Stop"
    tid = h.agents[0].tools._ctx.task_id
    row = load_task_result(r.root, tid)
    assert (row["status"], row["reason_code"]) == ("cancelled", "owner_requested_finalization")
    assert row["cancel_origin"]["stop_policy"] == "immediate" and stopped is not None
    if kind == "command":
        shell, child = (int((marks / name).read_text()) for name in ("shell.pid", "child.pid"))
        deadline = time.monotonic() + 5
        while (_alive(shell) or _alive(child)) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(shell) and not _alive(child), "the command's whole process group is gone"
    assert ci.active_intent(r.root, tid), "the runner's terminal settles the intent later, as before"
    assert not h.executions.live()


@pytest.mark.parametrize("kind", KINDS)
def test_wrap_up_keeps_its_semantics_for_a_running_tool_wait(owner_stop, monkeypatch, tmp_path, kind):
    """Wrap up is mail: it ends an in-slot wait at once, while a running command finishes with its output and only
    then the one final turn answers. The command here is short, so its natural end is what the turn waits for."""
    r, press = owner_stop
    marks = _marks(tmp_path)
    pressed = {}

    def wrap_up_while_waiting(task_id):
        assert _in_flight(r.root, kind, marks, time.monotonic() + 30), "the tool never started waiting"
        pressed["at"] = time.monotonic()
        pressed["response"] = press(task_id, cascade=True, policy="finalize_then_cancel", action="owner-wrap-up")

    h, generated, inputs = _scripted(monkeypatch, r, kind, marks, wrap_up_while_waiting,
                                     seconds=WAIT_SEC if kind == "await" else 4)
    result = start(r, h).result.result(timeout=90)
    ended = time.monotonic()
    for thread in h.presses:
        thread.join(35)

    assert pressed["response"].status_code in {200, 202}, pressed["response"].text
    assert generated == [1, 2] and "[OWNER_STOP]" in json.dumps(inputs[1], ensure_ascii=False)
    assert result.text == SUMMARY
    if kind == "await":
        assert ended - pressed["at"] < PROMPT_SEC
    else:
        [finished] = [row for row in _tool_rows(r.root, "run_command") if row.get("type") == "tool_call"]
        assert "finished" in finished["result_preview"], "Wrap up let the running command finish with its output"
