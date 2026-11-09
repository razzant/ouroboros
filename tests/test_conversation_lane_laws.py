"""Acceptance laws of the conversation lane that no single module owns (E1, E6, E9, one author).

E1 — placement: no server-side module starts cognition (``handle_task`` / ``make_agent``) outside
the two named carriers of the worker body; the admission doors construct no agent.
E6 — the owner's wait is flat: an owner turn's admission does the same work with 0 and 5 other
lane turns live (counted, not timed).
E9 — Panic kills a lane carrier by measured identity and its row is settled; a project fence does
not; a lane turn handed off at a planned restart returns as the same id's next attempt with its
saved source, never a replay; a lane row's Stop is the pooled custody (no inline arming).
One author — my human's next ordinary Main message reaches the one confirmed owner turn reading that
chat, through the real bridge intake, routing, lane admission and carrier intake fact (a second process
plays the carrier), from its admission to its final close; every other reader set, and every refusal,
leaves the words their own turn.
"""

import hashlib
import json
import os
import pathlib
import queue as stdqueue
import re
import subprocess
import sys
import threading
import time
import uuid
from types import SimpleNamespace

import pytest

from ouroboros import store_timing
from ouroboros.task_carrier import LANE_CONVERSATION
from supervisor import conversation_lane, worker_chat_lane, workers
from tests._main_followup import _delivered, _say, _turns, main as main
from tests.test_owner_wait_pool import InputQueue, Process, pool as pool  # noqa: F401 -- fixture
from tests.worker_pool_support import affirm_ready


REPO = pathlib.Path(__file__).resolve().parents[1]
# The processes that may START cognition: the worker body and its inline twin, the Presence body
# and its carrier factory, and the fresh native actor the inline executors borrow.
COGNITION_STARTERS = {
    "supervisor/worker_process.py", "supervisor/inline_carrier.py",
    "ouroboros/presence_runner.py", "ouroboros/presence_carrier.py",
    "supervisor/workers.py",  # _get_chat_agent: the inline executors' actor, called only from inline_carrier
}


def _server_side_modules():
    roots = [REPO / "supervisor", REPO / "ouroboros" / "gateway"]
    files = [path for root in roots for path in sorted(root.glob("*.py"))]
    files += sorted((REPO / "ouroboros").glob("server*.py")) + sorted((REPO / "ouroboros").glob("consciousness*.py"))
    files.append(REPO / "server.py")
    return files


def test_e1_no_server_side_module_starts_cognition_outside_the_named_carriers():
    offenders = []
    pattern = re.compile(r"(\.handle_task\(|\bmake_agent\()")
    for path in _server_side_modules():
        rel = path.relative_to(REPO).as_posix()  # the carriers are spelled with "/" on every OS
        if rel in COGNITION_STARTERS:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or "def make_agent" in line or "def handle_task" in line:
                continue
            if pattern.search(line) and "import" not in line:
                offenders.append(f"{rel}:{number}: {stripped[:100]}")
    assert offenders == [], "cognition started in a server-side module:\n" + "\n".join(offenders)
    # The inline executors' actor is borrowed only by the inline carrier.
    callers = [p.relative_to(REPO).as_posix() for p in _server_side_modules()
               if "_get_chat_agent()" in p.read_text(encoding="utf-8")
               and p.relative_to(REPO).as_posix() != "supervisor/workers.py"]
    assert callers == ["supervisor/inline_carrier.py"], callers


@pytest.fixture
def lane(pool, monkeypatch):
    from supervisor import message_bus

    monkeypatch.setattr(message_bus, "_BRIDGE", SimpleNamespace(send_chat_action=lambda *a, **k: None,
                                                                 broadcast=lambda m: None, push_log=lambda *a, **k: None))
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: pytest.fail("the server constructed an agent"))
    monkeypatch.setattr("ouroboros.agent.make_agent", lambda **k: pytest.fail("the server constructed an agent"))
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", id(workers.WORKERS))
    return pool


def _spare(wid):
    slot = workers.Worker(wid, Process(6000 + wid), InputQueue(), lane=LANE_CONVERSATION)
    affirm_ready(slot)
    workers.WORKERS[wid] = slot
    return slot


def _admit(lane_spare, text):
    with store_timing.metered() as meter:
        worker_chat_lane.handle_chat_direct(1, text, None, task_metadata={"client_message_id": f"cm-{text}"})
    snapshot = store_timing.snapshot(meter)
    return snapshot


def test_e6_admission_work_is_equal_with_zero_and_five_other_lane_turns_live(lane):
    spare = _spare(10)
    alone = _admit(spare, "alone")
    spare.busy_task_id = None
    spare.in_q.get_nowait()
    for i in range(5):  # five other owner turns live on their carriers
        other = _spare(20 + i)
        other.busy_task_id = f"turn-{i}"
        workers.RUNNING[f"turn-{i}"] = {"task": {"id": f"turn-{i}", "chat_id": 1, "_is_direct_chat": True},
                                        "worker_id": other.wid, "lane": LANE_CONVERSATION, "attempt": 1,
                                        "started_at": time.time(), "last_heartbeat_at": time.time()}
    crowded = _admit(spare, "crowded")
    counted = {key: value for key, value in alone.items() if isinstance(value, (int, float)) and key not in {"wall_sec", "ms"}}
    assert counted == {key: value for key, value in crowded.items() if key in counted}, (alone, crowded)
    assert len(workers.PENDING) == 0


def test_e9_panic_kills_a_lane_carrier_by_identity_and_settles_its_row(lane, monkeypatch):
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda _pid: None)
    monkeypatch.setattr(workers, "_audit_delegate_terminal_custody", lambda *a, **k: None)
    monkeypatch.setattr(workers, "reconstruct_task_cost", lambda *a, **k: {"cost_accounting_status": "unavailable",
                                                                             "cost_final": False})
    monkeypatch.setattr(workers, "append_jsonl", lambda *a, **k: True)
    killed = []
    from supervisor import worker_pool_lifecycle as lifecycle

    def kill(pid, **kwargs):
        killed.append(pid)
        for worker in workers.WORKERS.values():
            if worker.proc.pid == pid:
                worker.proc.alive = False
                worker.proc.exitcode = -9

    monkeypatch.setattr(lifecycle, "kill_worker_tree", kill)
    monkeypatch.setattr(workers, "kill_worker_tree", kill)
    spare = _spare(10)
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-p"})
    turn = spare.busy_task_id
    assert turn and spare.proc.pid in {w.proc.pid for w in workers.WORKERS.values()}
    assert workers.kill_workers(stop_source="panic")  # Panic: every slot, by measured identity
    assert spare.proc.pid in killed and 42 in killed  # the lane carrier AND the project worker
    assert turn not in workers.RUNNING and not workers.WORKERS
    from ouroboros.task_results import load_task_result
    assert load_task_result(lane.root, turn)["status"] in {"failed", "cancelled", "interrupted"}


def test_e9_a_lane_turn_survives_a_planned_restart_as_the_same_ids_next_attempt(lane, monkeypatch):
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda _pid: None)
    monkeypatch.setattr(workers, "append_jsonl", lambda *a, **k: True)
    from supervisor import worker_pool_lifecycle as lifecycle

    def kill(pid, **kwargs):
        for worker in workers.WORKERS.values():
            if worker.proc.pid == pid:
                worker.proc.alive = False
                worker.proc.exitcode = 0

    monkeypatch.setattr(lifecycle, "kill_worker_tree", kill)
    monkeypatch.setattr(workers, "kill_worker_tree", kill)
    spare = _spare(10)
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-r"})
    turn = spare.busy_task_id
    workers.RUNNING.pop("owner")  # the pooled fixture row is not this law's subject
    assert workers.kill_workers(preserve_pending=True, preserve_running_task_ids={turn},
                                terminal_status="interrupted", retain_saved_work=True, stop_source="owner_restart")
    successors = [row for row in workers.PENDING if row.get("id") == turn]
    assert len(successors) == 1, workers.PENDING
    successor = successors[0]
    assert successor["_attempt"] == 2 and successor["_is_direct_chat"]
    assert turn not in workers.RUNNING


def test_e9_a_lane_rows_stop_is_the_pooled_custody_not_the_inline_arming(lane, monkeypatch):
    armed = []
    monkeypatch.setattr(workers, "arm_direct_chat_turn", lambda *a, **k: armed.append(a) or None)
    spare = _spare(10)
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-s"})
    turn = spare.busy_task_id
    assert workers.direct_chat_turn(turn) is None  # no in-process actor owns a lane row
    from supervisor import cancel_publication, queue
    assert cancel_publication.finish_in_process_turn(queue, turn, None, {}, intent=None) is None
    assert armed == []


# --- One author: my human's next Main words reach the turn already answering them ---------------------

def _intake(root, task_id):
    from ouroboros.conversation_lease import _intake_path

    path = _intake_path(root, task_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _upload(root, name, data):
    path = pathlib.Path(root) / "uploads" / f"{uuid.uuid4().hex}_{name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(path), "label": name, "mime": "image/png", "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


@pytest.mark.parametrize("carrier", ["awaiting", "dispatched"])
def test_a_main_follow_up_joins_the_admitted_owner_turn_with_its_words_files_and_reply(main, carrier):
    """From confirmed admission, while it awaits a carrier or its carrier is preparing it, the turn takes them."""
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.project_dialogue import latest_chat_annotations

    spare = _spare(10) if carrier == "dispatched" else None
    _say(main.bridge, main.ctx, "Send it to our team channel", "cm-first")
    (first,) = _turns()
    turn = first["task"]["id"]
    assert first["worker_id"] == (spare.wid if spare else None) and not first.get("_unpublished")
    assert _intake(main.root, turn) == {"accepting": True, "conversation_key": "owner:1", "pid": os.getpid(),
                                        "attempt": 1}  # the host's own fact: nothing has read yet
    shot = _upload(main.root, "screenshot.png", b"\x89PNG\r\n\x1a\n" + b"pixels" * 40)
    reply = {"address": "row:1@2025-01-02T12:00:00+00:00#aabbccddeeff", "author": "assistant"}
    _say(main.bridge, main.ctx, "This chat", "cm-second",
         task_metadata={"chat_attachment_uploads": [shot], "reply_reference": reply})

    assert [meta["task"]["id"] for meta in _turns()] == [turn]  # no second author
    (ack,) = _delivered(main.bridge, "cm-second")
    assert ack["status"] == "delivered" and ack["target"] == turn
    assert [row["label"] for row in ack["attachment_manifest"]] == ["screenshot.png"]
    row = latest_chat_annotations(main.root)["cm-second"]
    assert (row["action"], row["status"], row["target"]) == ("mailbox_delivery", "delivered", turn)
    (entry,) = drain_owner_entries(main.root, turn, set(), 1)
    assert entry["text"].startswith("This chat") and entry["client_message_id"] == "cm-second"
    assert entry["msg_id"] == f"cm-second:{turn}" and entry["reply_reference"] == reply
    (staged,) = entry["attachment_manifest"]
    assert staged["status"] == "staged" and staged["is_image"] and staged["sha256"] == shot["sha256"]
    assert pathlib.Path(staged["abs_path"]).read_bytes() == pathlib.Path(shot["path"]).read_bytes()
    if spare is not None:
        assert spare.in_q.get_nowait()["id"] == turn and spare.in_q.empty()  # one dispatch: one author


@pytest.mark.parametrize("named", [False, True])
def test_the_saved_receipt_names_the_turn_by_its_own_name_or_by_none(main, named):
    """The receipt's event-time label is the turn's own name when it has one; an unnamed turn gets no placeholder
    ("Task" named nothing), so the line says only that the words were saved."""
    from ouroboros.project_dialogue import latest_chat_annotations

    _say(main.bridge, main.ctx, "Send it to our chat", "cm-first")
    (first,) = _turns()
    if named:  # as the turn's namer leaves it on its row
        first["task"]["title"] = "Release note for the team channel"
    _say(main.bridge, main.ctx, "This chat", "cm-second")
    (ack,) = _delivered(main.bridge, "cm-second")
    row = latest_chat_annotations(main.root)["cm-second"]
    expected = "Release note for the team channel" if named else ""
    assert (ack["target"], ack.get("target_label") or "", row.get("target_label") or "") == (
        first["task"]["id"], expected, expected)


_CARRIER = r"""
import json, os, sys
from contextlib import contextmanager
from types import SimpleNamespace
from ouroboros import conversation_lease, task_custody
from ouroboros.owner_mailbox import drain_owner_entries
from supervisor.worker_chat_lane import CarrierTurn

root, task = sys.argv[1], json.load(open(sys.argv[2], encoding="utf-8"))
say = lambda line: print(line, flush=True)
turn = None
say(f"ready {os.getpid()}")  # the interpreter's own pid: on Windows a venv's python.exe launches it as a child
while True:
    line = sys.stdin.readline()
    if not line:
        say("eof")
        break
    command = line.strip()
    if command.startswith("take"):  # "take write" / "take lock": its one write fails, or its lock is refused, once
        real_write, real_lock = conversation_lease._write_intake, task_custody.task_mail_lock
        def broken(*args, **kwargs):
            conversation_lease._write_intake = real_write
            raise OSError("intake storage unavailable")
        @contextmanager
        def refused(*args, **kwargs):
            task_custody.task_mail_lock = real_lock
            yield False
        if command == "take write":
            conversation_lease._write_intake = broken
        elif command == "take lock":
            task_custody.task_mail_lock = refused
        turn = CarrierTurn(task, root, SimpleNamespace(put=lambda *_a, **_k: None))
        say("taken " + turn.intake)
    elif command in ("close", "final"):  # the final drain's close ("close": held inside its mail lock until released)
        real = conversation_lease._write_intake
        def held(*args, **kwargs):
            say("in-lock")
            assert sys.stdin.readline().strip() == "release"
            return real(*args, **kwargs)
        conversation_lease._write_intake = held if command == "close" else real
        loop_ctx = SimpleNamespace(task_metadata={}, drive_root=root, task_id=task["id"],
                                   task_attempt=task.get("_attempt"))
        conversation_lease.follow_loop_admission(loop_ctx, False)
        conversation_lease._write_intake = real
        drained = drain_owner_entries(root, task["id"], set(), task.get("_attempt") or 1)
        say("drained " + json.dumps([entry.get("client_message_id") for entry in drained]))
    elif command == "hold":  # another mailbox effect of this turn: its mail lock held until released
        with task_custody.task_mail_lock(root, task["id"]) as locked:
            say("holding " + str(bool(locked)))
            assert sys.stdin.readline().strip() == "release"
        say("released")
    elif command == "end":
        turn.ended()
        say("ended")
        break
"""


_THREAD = threading.Thread  # the pool fixture stubs ``threading.Thread`` process-wide; these helpers need real ones


class _Carrier:
    """A second process that takes the dispatched turn as its carrier does (``CarrierTurn``) and runs its loop's
    final close (``follow_loop_admission``) and end, one stdin command at a time."""

    def __init__(self, root, task, scratch):
        task_path = scratch / "dispatched.json"
        task_path.write_text(json.dumps(task), encoding="utf-8")
        self.errors = scratch / "carrier.stderr"
        with self.errors.open("w", encoding="utf-8") as errors:
            self.proc = subprocess.Popen([sys.executable, "-c", _CARRIER, str(root), str(task_path)],
                                         cwd=str(pathlib.Path(__file__).resolve().parents[1]), stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=errors, text=True)
        self.lines = stdqueue.Queue()
        _THREAD(target=lambda: [self.lines.put(line.strip()) for line in self.proc.stdout], daemon=True).start()
        self.pid = int(self.send("", "ready"))  # the pid an intake fact it writes names, not the launcher's

    def send(self, command, expect=None):
        if command:
            self.proc.stdin.write(command + "\n")
            self.proc.stdin.flush()
        if expect is None:
            return ""
        try:
            line = self.lines.get(timeout=60)
        except stdqueue.Empty:
            raise AssertionError(f"the carrier said nothing after {command!r}: {self.errors.read_text()[-4000:]}")
        assert line.startswith(expect), (line, self.errors.read_text()[-4000:])
        return line[len(expect):].strip()

    def stop(self):
        if self.proc.poll() is None:
            self.proc.kill()  # this test's own child, by its exact handle
        self.proc.wait(timeout=30)


def test_the_carrier_takes_the_fact_and_its_locked_final_close_orders_a_racing_arrival_out(main, tmp_path):
    """A real second process: words landed before the close are its final drain's; one racing the close waits for
    its mail lock, then finds the intake closed and starts its own turn. Its end leaves no fact behind."""
    from ouroboros.owner_mailbox import drain_owner_entries

    spare = _spare(10)
    _say(main.bridge, main.ctx, "Send the draft to the group", "cm-first")
    dispatched = spare.in_q.get_nowait()
    turn = dispatched["id"]
    carrier = _Carrier(main.root, dispatched, tmp_path)
    try:
        assert carrier.send("take", "taken") == "owner:1"
        taken = _intake(main.root, turn)
        assert taken["pid"] == carrier.pid and taken["pid"] != os.getpid() and taken["accepting"] is True
        _say(main.bridge, main.ctx, "The workshop group", "cm-early")
        assert _delivered(main.bridge, "cm-early") and [meta["task"]["id"] for meta in _turns()] == [turn]

        carrier.send("close", "in-lock")  # the carrier holds the mail lock, about to close
        racing = _THREAD(target=_say, args=(main.bridge, main.ctx, "And mention the deadline", "cm-late"))
        racing.start()
        racing.join(0.7)
        assert racing.is_alive(), "the arrival must wait for the carrier's mail lock"
        assert not _turns("And mention")
        assert carrier.send("release", "drained") == json.dumps(["cm-early"])
        racing.join(30)
        assert not racing.is_alive()
        assert not _delivered(main.bridge, "cm-late")
        assert [entry["client_message_id"] for entry in drain_owner_entries(main.root, turn, set(), 1)] == ["cm-early"]
        (own,) = _turns("And mention the deadline")  # the refused words are their own turn, nothing lost
        assert own["task"]["id"] != turn and _intake(main.root, own["task"]["id"])["accepting"] is True
        assert _intake(main.root, turn)["accepting"] is False

        assert carrier.send("end", "ended") == ""
        assert _intake(main.root, turn) is None
        assert carrier.proc.wait(timeout=30) == 0
    finally:
        carrier.stop()


@pytest.mark.parametrize("failing", ["write", "lock"])
def test_a_failed_take_retires_the_hosts_fact_so_no_word_lands_after_the_final_drain(main, tmp_path, failing):
    """A real second process whose take fails once (its one write, or its mail lock refused): the host's fact, which
    its loop's final close cannot move (another process wrote it), is retired at once. Words and files admitted before
    stay for its drain; words after its final drain are their own turn, never saved where no round reads them."""
    from ouroboros.owner_mailbox import drain_owner_entries

    spare = _spare(10)
    _say(main.bridge, main.ctx, "Send the draft to the group", "cm-first")
    dispatched = spare.in_q.get_nowait()
    turn = dispatched["id"]
    shot = _upload(main.root, "group.png", b"\x89PNG\r\n\x1a\n" + b"group" * 30)
    _say(main.bridge, main.ctx, "The workshop group", "cm-early", task_metadata={"chat_attachment_uploads": [shot]})
    assert [ack["target"] for ack in _delivered(main.bridge, "cm-early")] == [turn]  # the host's fact admitted it
    carrier = _Carrier(main.root, dispatched, tmp_path)
    try:
        assert carrier.send(f"take {failing}", "taken") == ""
        assert carrier.send("final", "drained") == json.dumps(["cm-early"])
        _say(main.bridge, main.ctx, "Thanks", "cm-late")
        mailbox = [entry["client_message_id"] for entry in drain_owner_entries(main.root, turn, set(), 1)]
        assert not _delivered(main.bridge, "cm-late"), (mailbox, _intake(main.root, turn))
        assert mailbox == ["cm-early"] and len(_turns("Thanks")) == 1
        assert _intake(main.root, turn) is None
        (early,) = drain_owner_entries(main.root, turn, set(), 1)[0]["attachment_manifest"]
        assert pathlib.Path(early["abs_path"]).read_bytes() == pathlib.Path(shot["path"]).read_bytes()
        assert carrier.send("end", "ended") == ""
        assert carrier.proc.wait(timeout=30) == 0
    finally:
        carrier.stop()


@pytest.mark.parametrize("standing", ["host", "successor_attempt", "other_conversation"])
def test_a_failed_take_retires_only_its_conversations_fact_for_its_own_attempt(main, monkeypatch, standing):
    """Only the fact naming the turn's conversation and the attempt taken goes; a later attempt's or another
    conversation's stays as its writer left it, and the failed carrier's end removes nothing."""
    from ouroboros import conversation_lease

    spare = _spare(10)
    _say(main.bridge, main.ctx, "Draft the reply", "cm-first")
    dispatched = spare.in_q.get_nowait()
    turn = dispatched["id"]
    host = _intake(main.root, turn)
    fact = {"host": host, "successor_attempt": {**host, "attempt": 2},
            "other_conversation": {**host, "conversation_key": "owner:4242"}}[standing]
    conversation_lease._intake_path(main.root, turn).write_text(json.dumps(fact), encoding="utf-8")
    real = conversation_lease._write_intake

    def broken(*_args, **_kwargs):
        monkeypatch.setattr(conversation_lease, "_write_intake", real)
        raise OSError("intake storage unavailable")

    monkeypatch.setattr(conversation_lease, "_write_intake", broken)
    carrier = worker_chat_lane.CarrierTurn(dispatched, main.root, SimpleNamespace(put=lambda *_a, **_k: None))
    assert carrier.intake == ""
    expected = None if standing == "host" else fact
    assert _intake(main.root, turn) == expected
    carrier.ended()
    assert _intake(main.root, turn) == expected


def _seed(text, metadata=None, **fields):
    return {**worker_chat_lane._seed_task(1, text, None, None, metadata or {}), **fields}


def _admitted(task, *, reply_expected, reserved=False, publish=True):
    from supervisor import queue

    with queue._queue_lock:
        meta = conversation_lane.admit_turn(task, chat_id=1, quiet=not reply_expected,
                                            reply_expected=reply_expected, reserved=reserved)
    if publish:
        assert conversation_lane.publish_awaiting_rows()
    return meta


def _owner_turn(main, text="First request"):
    _say(main.bridge, main.ctx, text, "cm-" + uuid.uuid4().hex[:6])
    return next(meta for meta in _turns(text))


@pytest.mark.parametrize("case", [
    "pooled_root_beside", "wake_beside", "sole_wake", "reserved_call", "sole_call", "unpublished", "no_intake",
    "superseded_attempt", "readmitted_while_staging", "settled", "settles_before_append", "stop_pending",
    "project_bound", "detached", "other_chat", "constrained_message",
])
def test_a_main_message_is_its_own_turn_unless_exactly_one_owner_turn_reads(main, monkeypatch, case):
    """Only owner turns are candidates, none is chosen among several; absence of the fact is never preparation."""
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.conversation_lease import _intake_path, open_author_intake
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_results import write_task_result

    message = {}
    reader = None
    if case == "pooled_root_beside":
        reader = _owner_turn(main)
        workers.RUNNING["pooled"] = {"task": {"id": "pooled", "type": "task", "chat_id": 1, "text": "background"},
                                     "worker_id": 0, "attempt": 1, "started_at": time.time()}
        write_task_result(main.root, "pooled", "running")
    elif case in {"wake_beside", "sole_wake"}:
        if case == "wake_beside":
            reader = _owner_turn(main)
        _admitted(_seed("Own clock", {"wake_cause": "own_clock", "initiator": "consciousness"}), reply_expected=False)
    elif case in {"reserved_call", "sole_call"}:  # a live call's row: reserved for its executor, then bound to it
        call = _seed("Spoken words", {"origin_suppressed": True})
        meta = _admitted(call, reply_expected=True, reserved=True)
        if case == "sole_call":  # as ``inline_carrier.start_inline_turn`` binds it
            meta["task"]["_execution_owner"]["executor"] = "inline_call"
            meta.pop("_inline_bind")
        open_author_intake(main.root, call["id"], "owner:1", attempt=1)  # even a fact written for it admits nothing
    elif case == "unpublished":  # admitted under Q, its barrier not landed: not yet a confirmed admission
        held = _seed("Not on disk yet", {"origin_suppressed": True})
        reader = _admitted(held, reply_expected=True, publish=False)
        open_author_intake(main.root, held["id"], "owner:1", attempt=1)
    else:
        reader = _owner_turn(main)
    turn = reader["task"]["id"] if reader else ""
    if case == "no_intake":
        _intake_path(main.root, turn).unlink()  # awaiting a carrier, yet no positive fact: not preparation
    elif case == "superseded_attempt":  # a crash's readmission: the attempt-1 fact is not attempt 2's
        from supervisor import queue

        with queue._queue_lock:
            task = {**workers.RUNNING.pop(turn)["task"], "_attempt": 2}
            conversation_lane.readmit_after_crash(task, reader)
        assert conversation_lane.publish_awaiting_rows()
    elif case == "readmitted_while_staging":  # the row judged is replaced before the delivery takes the queue lock
        from ouroboros import server_owner_routing
        from supervisor import queue

        real_stage = server_owner_routing._stage_mailbox_attachments

        def stage(*args, **kwargs):
            with queue._queue_lock:
                conversation_lane.readmit_after_crash({**workers.RUNNING.pop(turn)["task"], "_attempt": 2}, reader)
            assert conversation_lane.publish_awaiting_rows()  # a published row: only its identity tells it apart
            return real_stage(*args, **kwargs)

        monkeypatch.setattr(server_owner_routing, "_stage_mailbox_attachments", stage)
    elif case == "settled":
        write_task_result(main.root, turn, "completed", result="answered")
    elif case == "settles_before_append":  # its exception terminal lands between the queue-lock checks and the append
        from ouroboros import owner_mailbox

        real_append = owner_mailbox._append_mail

        def append(drive_root, task_id, *args, **kwargs):
            if task_id == turn:
                write_task_result(main.root, turn, "failed", result="the turn's own exception terminal")
            return real_append(drive_root, task_id, *args, **kwargs)

        monkeypatch.setattr(owner_mailbox, "_append_mail", append)
    elif case == "stop_pending":
        request_cancel(main.root, turn, source="http_single")
    elif case == "project_bound":
        project = create_project(main.root, "moved-work")
        bind_task_to_project(main.root, turn, project["id"], origin={"absent": "system"})
    elif case == "detached":
        reader["task"]["detached_at"] = "2025-01-02T12:10:00+00:00"
    elif case == "other_chat":
        message = {"chat_id": 4242, "source": "telegram"}
    elif case == "constrained_message":
        message = {"task_constraint": {"mode": "normal", "extra_allowlist": ["web_search"]}}
    before = {meta["task"]["id"] for meta in _turns()}
    _say(main.bridge, main.ctx, "One more thing", "cm-follow", **message)
    if case in {"pooled_root_beside", "wake_beside"}:  # no other candidate beside the one owner turn: the words join it
        joined = [ack["target"] for ack in _delivered(main.bridge, "cm-follow")]
        assert joined == [turn] and not _turns("One more thing")
        return
    assert not _delivered(main.bridge, "cm-follow")
    started = [meta for meta in _turns("One more thing") if meta["task"]["id"] not in before]
    assert len(started) == 1, (case, [meta["task"].get("text") for meta in _turns()])
    for meta in _turns():
        if meta["task"]["id"] != started[0]["task"]["id"]:
            assert not any(entry.get("client_message_id") == "cm-follow"
                           for entry in drain_owner_entries(main.root, meta["task"]["id"], set(), 1))


def test_after_a_turns_locked_final_close_the_next_words_reach_the_one_turn_still_reading(main, tmp_path):
    """A real second process carries A to its final close; A still finishes (its result not stored). The words its
    closed intake refused began B; the next words reach B, the one turn still reading, never a third author."""
    from ouroboros.owner_mailbox import drain_owner_entries

    spare = _spare(10)
    _say(main.bridge, main.ctx, "Send the draft to the group", "cm-first")
    dispatched = spare.in_q.get_nowait()
    first = dispatched["id"]
    carrier = _Carrier(main.root, dispatched, tmp_path)
    try:
        assert carrier.send("take", "taken") == "owner:1"
        assert carrier.send("final", "drained") == "[]"
        assert _intake(main.root, first)["accepting"] is False and first in workers.RUNNING
        _say(main.bridge, main.ctx, "The workshop group", "cm-second")
        assert not _delivered(main.bridge, "cm-second")
        (second,) = _turns("The workshop group")
        _say(main.bridge, main.ctx, "And mention the deadline", "cm-third")
        assert [ack["target"] for ack in _delivered(main.bridge, "cm-third")] == [second["task"]["id"]], [
            meta["task"]["text"] for meta in _turns()]
        assert not _turns("And mention the deadline")
        assert [entry["client_message_id"] for entry in drain_owner_entries(
            main.root, second["task"]["id"], set(), 1)] == ["cm-third"]
        assert carrier.send("end", "ended") == ""
    finally:
        carrier.stop()


@pytest.mark.parametrize("meanwhile", ["nothing", "final_close", "stop", "readmitted", "pause", "owner_hold",
                                        "project_hold", "task_budget", "root_budget", "acceptance",
                                        "project_bound", "detached", "voice"])
def test_a_follow_up_waits_for_the_turns_mail_lock_outside_q_and_refuses_what_moved_meanwhile(
        main, tmp_path, monkeypatch, meanwhile):
    """A real second process holds the turn's mail lock (its final close, or another mailbox effect). The words wait
    for it without the queue lock, which this thread takes meanwhile; what moved during that wait (the close itself,
    a Stop/Pause/hold, a budget fence, or the row's readmission) still refuses them, and they are their own turn."""
    from ouroboros import task_custody
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.owner_mailbox import drain_owner_entries
    from supervisor import queue

    spare = _spare(10)
    _say(main.bridge, main.ctx, "Send the draft to the group", "cm-first")
    dispatched = spare.in_q.get_nowait()
    turn = dispatched["id"]
    reader = workers.RUNNING[turn]
    if meanwhile == "pause":
        # The scripted carrier takes intake but never starts Agent.handle_task. Supply its ordinary running
        # result so this case exercises the real durable Pause, not install_fence's missing-result refusal.
        from ouroboros.task_results import load_task_result, write_task_result
        write_task_result(main.root, turn, "running", root_task_id=turn)
        assert load_task_result(main.root, turn)["status"] == "running"
    real_lock, waiting, failures = task_custody.task_mail_lock, threading.Event(), []

    def watched(drive_root, task_id, **kwargs):
        if task_id == turn and threading.current_thread() is sender:
            waiting.set()
        return real_lock(drive_root, task_id, **kwargs)

    def send():
        try:
            _say(main.bridge, main.ctx, "This chat", "cm-second")
        except BaseException as exc:  # noqa: BLE001 -- reported by the asserting thread
            failures.append(exc)

    monkeypatch.setattr(task_custody, "task_mail_lock", watched)
    sender = _THREAD(target=send, daemon=True)
    carrier = _Carrier(main.root, dispatched, tmp_path)
    try:
        assert carrier.send("take", "taken") == "owner:1"
        closing = meanwhile == "final_close"
        assert carrier.send("close" if closing else "hold", "in-lock" if closing else "holding") in ("", "True")
        sender.start()
        assert waiting.wait(30), "the words never reached the turn's mail lock"
        time.sleep(0.3)
        assert sender.is_alive() and not _delivered(main.bridge, "cm-second"), "the carrier's lock is not held"
        started = time.monotonic()
        assert queue._queue_lock.acquire(timeout=2.0), "Q is held across the mail lock's wait"
        try:
            q_wait = time.monotonic() - started
            if meanwhile == "readmitted":  # a crash's readmission while the words wait: a new attempt's row
                conversation_lane.readmit_after_crash({**workers.RUNNING.pop(turn)["task"], "_attempt": 2}, reader)
        finally:
            queue._queue_lock.release()
        assert q_wait < 1.0 and sender.is_alive(), (q_wait, "the words still wait for the mail lock")
        if meanwhile == "readmitted":
            assert conversation_lane.publish_awaiting_rows()
        elif meanwhile == "stop":
            request_cancel(main.root, turn, source="http_single")
        elif meanwhile == "pause":
            from ouroboros.owner_pause import fence_closed, install_fence
            fence, created = install_fence(main.root, turn, request_id="pause-while-followup-waits")
            assert created and fence_closed(fence)
        elif meanwhile == "project_bound":
            from ouroboros.projects_registry import bind_task_to_project, create_project, project_chat_for_task
            project = create_project(main.root, "moved-while-mail-waits")
            bind_task_to_project(main.root, turn, project["id"], origin={"absent": "system"})
            assert project_chat_for_task(main.root, turn) == project["chat_id"]
        elif meanwhile in {"detached", "voice"}:
            with queue._queue_lock:
                owner = reader["task"]["_execution_owner"]
                if meanwhile == "detached":
                    owner["lane"] = "project"
                else:
                    owner["executor"] = "inline_call"
        elif meanwhile in {"owner_hold", "project_hold", "task_budget", "root_budget", "acceptance"}:
            with queue._queue_lock:
                if meanwhile == "root_budget":
                    main.ctx.BUDGET_ROOT_FENCES[turn] = {"status": "active", "root_task_id": turn}
                elif meanwhile == "acceptance":
                    queue.ACCEPTANCE_FENCES[turn] = {"status": "sealed"}
                else:
                    field = {"owner_hold": "_owner_hold", "project_hold": "_project_admission_hold",
                             "task_budget": "_budget_pause"}[meanwhile]
                    reader["task"][field] = {"reason": "test control while mail waits"}
        assert carrier.send("release", "drained" if closing else "released") in ("[]", "")
        sender.join(60)
        assert not sender.is_alive() and not failures, failures
        held = [entry.get("client_message_id") for entry in drain_owner_entries(main.root, turn, set(), 1)]
        if meanwhile == "nothing":
            assert [ack["target"] for ack in _delivered(main.bridge, "cm-second")] == [turn]
            assert held == ["cm-second"] and not _turns("This chat")
        else:
            assert not _delivered(main.bridge, "cm-second") and "cm-second" not in held
            assert len(_turns("This chat")) == 1, [meta["task"].get("text") for meta in _turns()]
    finally:
        carrier.stop()
        sender.join(30) if sender.is_alive() else None


@pytest.mark.parametrize("fact", ["closed", "open", "missing", "unreadable", "not_a_dict", "stale_attempt",
                                  "other_conversation", "closed_beside_wake"])
def test_only_its_own_closed_fact_for_its_attempt_takes_a_finishing_owner_turn_out_of_the_count(main, fact):
    """Proof that A stopped reading is its own fact for this chat and its current attempt, closed. No fact, an
    unreadable or foreign one, or an open one leaves A and B two owner turns (the words are their own turn); a wake
    beside B is no owner turn, so once A is closed B alone takes the words."""
    from ouroboros.conversation_lease import _intake_path, follow_loop_admission

    first = _owner_turn(main, "Send it to the group")["task"]["id"]
    path = _intake_path(main.root, first)
    host = json.loads(path.read_text(encoding="utf-8"))
    if fact in {"closed", "closed_beside_wake"}:  # its final close, as its loop runs it
        follow_loop_admission(SimpleNamespace(task_metadata={}, drive_root=main.root, task_id=first,
                                              task_attempt=None), False)
        assert json.loads(path.read_text(encoding="utf-8"))["accepting"] is False
    elif fact == "missing":
        path.unlink()
    elif fact == "unreadable":
        path.write_text("{", encoding="utf-8")
    elif fact != "open":
        closed = {**host, "accepting": False}
        path.write_text(json.dumps({"not_a_dict": [closed], "stale_attempt": {**closed, "attempt": 2},
                                    "other_conversation": {**closed, "conversation_key": "owner:4242"}}[fact]),
                        encoding="utf-8")
    second = worker_chat_lane.admit_owner_turn(1, "The workshop group",
                                               task_metadata={"origin_suppressed": True})["task"]["id"]
    if fact == "closed_beside_wake":
        _admitted(_seed("Own clock", {"wake_cause": "own_clock", "initiator": "consciousness"}), reply_expected=False)
    _say(main.bridge, main.ctx, "And mention the deadline", "cm-third")

    if fact in {"closed", "closed_beside_wake"}:  # a wake beside the one reading owner turn is no candidate
        assert [ack["target"] for ack in _delivered(main.bridge, "cm-third")] == [second]
        assert not _turns("And mention the deadline")
    else:
        assert not _delivered(main.bridge, "cm-third") and len(_turns("And mention the deadline")) == 1


@pytest.mark.parametrize("refusal", ["gate_closed", "restart_draining", "admission_error"])
def test_admission_writes_the_fact_before_its_row_and_a_refused_one_leaves_none(main, monkeypatch, refusal):
    real_admit, real_seed, seen, minted = conversation_lane.admit_turn, worker_chat_lane._seed_task, [], []

    def admit(task, **kwargs):
        seen.append(_intake(main.root, task["id"]))  # its row does not exist yet: no carrier can precede the fact
        return real_admit(task, **kwargs)

    monkeypatch.setattr(conversation_lane, "admit_turn", admit)
    monkeypatch.setattr(worker_chat_lane, "_seed_task", lambda *a, **k: minted.append(real_seed(*a, **k)) or minted[-1])
    _say(main.bridge, main.ctx, "First request", "cm-a")
    assert seen == [{"accepting": True, "conversation_key": "owner:1", "pid": os.getpid(), "attempt": 1}]
    assert _intake(main.root, minted[0]["id"]) == seen[0]

    if refusal == "gate_closed":
        monkeypatch.setattr(worker_chat_lane, "owner_conversation_admitted", lambda *_a: False)
    elif refusal == "restart_draining":
        monkeypatch.setattr("ouroboros.server_restart.agent_restart_draining", lambda: True)
        monkeypatch.setattr(worker_chat_lane, "_defer_chat_for_restart", lambda *_a, **_k: None)
    else:
        monkeypatch.setattr(conversation_lane, "admit_turn",
                            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("admission refused")))
        monkeypatch.setattr(worker_chat_lane, "_report_direct_chat_error", lambda *_a, **_k: None)
    assert worker_chat_lane.admit_owner_turn(1, "Not admitted", task_metadata={"origin_suppressed": True}) is None
    refused = minted[-1]["id"]
    assert refused != minted[0]["id"] and refused not in workers.RUNNING and _intake(main.root, refused) is None


def test_a_stop_takes_the_joined_words_with_the_turn_unread_and_continue_still_carries_them(main, monkeypatch):
    """Shared Stop: one custody ends the work both messages reached; a later message is its own turn; the joined
    words and their file stay with the stopped work, unread, never answered or replayed, for Continue to read."""
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.owner_continue import owner_sources
    from ouroboros.owner_mailbox import mail_read_state
    from ouroboros.task_results import load_task_result
    from supervisor import queue, task_lifecycle

    monkeypatch.setattr(queue, "_emit_cancel_task_done", lambda *_a, **_kw: None)
    _say(main.bridge, main.ctx, "Send it to our chat", "cm-first")
    (first,) = _turns()
    turn = first["task"]["id"]
    shot = _upload(main.root, "chat.png", b"\x89PNG\r\n\x1a\n" + b"shot" * 30)
    _say(main.bridge, main.ctx, "This chat", "cm-second", task_metadata={"chat_attachment_uploads": [shot]})
    assert _delivered(main.bridge, "cm-second")

    request_cancel(main.root, turn, source="http_single")
    _say(main.bridge, main.ctx, "Never mind, wait", "cm-third")
    assert not _delivered(main.bridge, "cm-third") and len(_turns("Never mind")) == 1
    assert task_lifecycle.cancel_task_custody(turn) == task_lifecycle.CANCEL_CANCELLED
    stopped = load_task_result(main.root, turn)
    assert stopped["status"] == "cancelled" and turn not in workers.RUNNING
    assert mail_read_state(main.root, turn, f"cm-second:{turn}") is False  # saved, never read or answered
    later = owner_sources(main.root, stopped, turn)
    assert [row["content"].split("\n")[0] for row in later["later"]] == ["This chat"]
    assert [row["attachment_manifest"][0]["label"] for row in later["attached"]] == ["chat.png"]


def test_preparation_failure_keeps_an_accepted_clarification_and_file_for_continue(main, monkeypatch):
    """An I/O failure in the real first-step staging happens before the agent binds; its terminal still keeps
    the clarification admitted while preparing. This uses the carrier's exception publisher, not a fabricated result."""
    from ouroboros import artifacts
    from ouroboros.owner_continue import owner_sources
    from ouroboros.owner_mailbox import mail_read_state
    from ouroboros.task_results import load_task_result

    spare = _spare(10)
    initial = _upload(main.root, "first.png", b"\x89PNG\r\n\x1a\n" + b"first" * 30)
    follow = _upload(main.root, "follow.png", b"\x89PNG\r\n\x1a\n" + b"follow" * 30)
    _say(main.bridge, main.ctx, "Use the attached example", "cm-first",
         task_metadata={"chat_attachment_uploads": [initial]})
    task = spare.in_q.get_nowait()
    turn = task["id"]
    frames = stdqueue.Queue()
    actor = SimpleNamespace(env=SimpleNamespace(drive_root=main.root, drive_path=lambda rel: main.root / rel))
    carrier = worker_chat_lane.CarrierTurn(task, main.root, frames)
    real_stage = artifacts.stage_task_attachments

    def fail_initial(root, task_id, uploads, *args, **kwargs):
        if uploads == [initial]:
            _say(main.bridge, main.ctx, "Actually use this picture", "cm-second",
                 task_metadata={"chat_attachment_uploads": [follow]})
            assert [row["target"] for row in _delivered(main.bridge, "cm-second")] == [turn]
            raise OSError("first attachment storage unavailable")
        return real_stage(root, task_id, uploads, *args, **kwargs)

    monkeypatch.setattr(artifacts, "stage_task_attachments", fail_initial)
    try:
        with pytest.raises(OSError, match="first attachment") as failed:
            worker_chat_lane.prepare_conversation_task(actor, task, main.root)
        worker_chat_lane.conversation_exception_terminal(actor, task, failed.value, frames, main.root)
    finally:
        carrier.ended()
    result = load_task_result(main.root, turn)
    assert result["status"] == "failed" and _intake(main.root, turn) is None
    assert mail_read_state(main.root, turn, f"cm-second:{turn}") is False
    later = owner_sources(main.root, result, turn)
    assert [row["content"].split("\n")[0] for row in later["later"]] == ["Actually use this picture"]
    attachment = later["attached"][0]["attachment_manifest"][0]
    assert attachment["label"] == "follow.png"
    assert pathlib.Path(attachment["abs_path"]).read_bytes() == pathlib.Path(follow["path"]).read_bytes()
    _say(main.bridge, main.ctx, "A new question", "cm-third")
    assert not _delivered(main.bridge, "cm-third") and len(_turns("A new question")) == 1


def test_a_dead_carrier_takes_its_fact_along_and_the_retry_reads_again_what_its_attempt_had_read(main, monkeypatch):
    """The carrier's death removes its fact (``worker_health``). The crash retry is the same id's next attempt: no
    words join it before its own carrier takes it, and its first drain reads again what the dead attempt read
    (owner-text acknowledgements belong to their attempt), beside what joined attempt 2."""
    from ouroboros.owner_mailbox import acknowledge_transcript_entry, drain_owner_entries
    from ouroboros.task_results import write_task_result
    from supervisor import queue, task_reaper
    from supervisor.worker_health import recover_confirmed_dead_worker

    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda *_a: None)
    monkeypatch.setattr(task_reaper, "_respawn_after_reap", lambda *_a, **_k: None)
    silent = SimpleNamespace(put=lambda *_a, **_k: None)
    spare = _spare(10)
    _say(main.bridge, main.ctx, "Draft the reply", "cm-first")
    dispatched = spare.in_q.get_nowait()
    turn = dispatched["id"]
    first_carrier = worker_chat_lane.CarrierTurn(dispatched, main.root, silent)
    _say(main.bridge, main.ctx, "Keep it short", "cm-early")
    (read,) = drain_owner_entries(main.root, turn, set(), 1)
    assert acknowledge_transcript_entry(main.root, turn, read)  # attempt 1 read it; then its carrier died
    spare.proc.alive, spare.proc.exitcode = False, -9
    dead = workers.RUNNING[turn]
    recover_confirmed_dead_worker({"kind": "confirmed_dead_worker", "worker_id": spare.wid, "worker": spare,
                                   "task_id": turn, "meta": dead, "task": dict(dead["task"]), "attempt": 1,
                                   "exitcode": -9, "drive_root": str(main.root)})
    assert _intake(main.root, turn) is None
    retry = workers.RUNNING[turn]
    assert retry is not dead and retry["attempt"] == 2 and retry["worker_id"] is None

    _say(main.bridge, main.ctx, "And sign it", "cm-gap")  # no carrier holds attempt 2 yet: no fact, their own turn
    assert not _delivered(main.bridge, "cm-gap")
    (gap,) = _turns("And sign it")
    write_task_result(main.root, gap["task"]["id"], "completed", result="answered on its own")
    workers.RUNNING.pop(gap["task"]["id"])
    spare2 = _spare(11)
    with queue._queue_lock:
        assert conversation_lane.dispatch_awaiting_turns() == 1
    second = spare2.in_q.get_nowait()
    assert second["id"] == turn and second["_attempt"] == 2
    retry_carrier = worker_chat_lane.CarrierTurn(second, main.root, silent)
    assert _intake(main.root, turn)["attempt"] == 2
    _say(main.bridge, main.ctx, "Add today's date", "cm-late")
    assert [ack["target"] for ack in _delivered(main.bridge, "cm-late")] == [turn]
    again = [entry["client_message_id"] for entry in drain_owner_entries(main.root, turn, set(), 2)]
    assert again == ["cm-early", "cm-late"]
    first_carrier.ended()  # the dead attempt's late end removes nothing of attempt 2's
    assert _intake(main.root, turn)["attempt"] == 2
    retry_carrier.ended()
    assert _intake(main.root, turn) is None


class _InlineOwner:
    """The degraded executor's actor: it holds its turn under its own admission lock (as ``Agent._bind_turn``),
    then, when told, closes as its final drain does (its local flag and durable fact in one step) or as a forced
    rail does (``_close_forced_intake``: the fact only, the flag stays its Stop door)."""

    def __init__(self, forced=False):
        self.forced = forced
        self._owner_message_admission_lock = threading.Lock()
        self._busy = self._accepting_owner_messages = False
        self._current_task_id, self._current_chat_id = None, None
        self._current_task_metadata, self._current_task_text = {}, ""
        self.owner_wait_callback = self._event_queue = None
        self.env = SimpleNamespace(drive_root=None, drive_path=lambda rel: None)
        self.bound, self.close, self.closed, self.release = (threading.Event() for _ in range(4))

    def handle_task(self, task):
        from ouroboros.conversation_lease import follow_loop_admission

        if self.bound.is_set():  # a second turn borrowing this actor: it answers at once
            return [{"type": "task_done", "task_id": task["id"], "chat_id": task["chat_id"], "status": "completed"}]
        with self._owner_message_admission_lock:
            self._busy, self._accepting_owner_messages = True, True
            self._current_task_id, self._current_chat_id = task["id"], task["chat_id"]
            self._current_task_metadata, self._current_task_text = dict(task.get("metadata") or {}), task["text"]
        self.bound.set()
        assert self.close.wait(30)
        tools_ctx = SimpleNamespace(task_metadata=dict(task.get("metadata") or {}), task_id=task["id"],
                                    drive_root=workers.DRIVE_ROOT, task_attempt=None,
                                    owner_message_admission_agent=self,
                                    owner_message_admission_lock=self._owner_message_admission_lock)
        if self.forced:
            from ouroboros.loop_forced_finalization import _close_forced_intake

            _close_forced_intake(SimpleNamespace(tools=SimpleNamespace(_ctx=tools_ctx)))
        else:
            with self._owner_message_admission_lock:
                self._accepting_owner_messages = False
                follow_loop_admission(tools_ctx, False)
        self.closed.set()
        assert self.release.wait(30)
        with self._owner_message_admission_lock:
            self._busy, self._current_task_id = False, None
        return [{"type": "task_done", "task_id": task["id"], "chat_id": task["chat_id"], "status": "completed"}]


@pytest.mark.parametrize("close", ["final", "forced"])
def test_the_degraded_executors_turn_takes_follow_ups_under_its_own_lock_until_its_final_close(
        main, monkeypatch, close):
    """On a physical fact the turn runs on a server thread: its actor's own admission lock admits the words while it
    reads and its fact accepts, and its closed final, or a forced rail that leaves its flag open as its Stop door,
    refuses the next ones whichever route they take."""
    from ouroboros.server_routing_context import addressable_roots
    from supervisor import inline_carrier

    actor = _InlineOwner(forced=close == "forced")
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: actor)
    monkeypatch.setattr(inline_carrier, "threading", SimpleNamespace(Thread=_THREAD))
    monkeypatch.setattr(conversation_lane, "_SPAWN_REFUSED", ["carrier_spawn_refused:OSError"])
    _say(main.bridge, main.ctx, "Rename the files", "cm-first")
    try:
        assert actor.bound.wait(30)
        turn = actor._current_task_id
        slot = next(w for w in list(workers.WORKERS.values()) if w.busy_task_id == turn)
        assert workers.RUNNING[turn]["executor"] == "inline_degraded"
        assert _intake(main.root, turn)["pid"] == os.getpid()  # its carrier is this process: the fact is its own
        assert [(row["task_id"], bool(row.get("direct_chat"))) for row in addressable_roots(main.ctx).rows(1)] == [
            (turn, True)]  # its actor's own row stands for its queue row
        _say(main.bridge, main.ctx, "Only the PDFs", "cm-pdf")
        assert [ack["target"] for ack in _delivered(main.bridge, "cm-pdf")] == [turn]
        actor.close.set()
        assert actor.closed.wait(30)
        assert _intake(main.root, turn)["accepting"] is False
        assert (workers.direct_chat_turn(turn) is not None) is actor.forced  # a forced rail's Stop door stays
        _say(main.bridge, main.ctx, "Thanks!", "cm-thanks")
        assert not _delivered(main.bridge, "cm-thanks") and len(_turns("Thanks!")) == 1
        actor.release.set()
        slot.proc.join(30)
        assert not slot.proc.is_alive() and _intake(main.root, turn) is None
    finally:  # never leave the inline thread waiting on a failed assertion
        actor.close.set()
        actor.release.set()


@pytest.mark.parametrize("task,key", [
    ({"chat_id": 1, "_is_direct_chat": True, "metadata": {"origin_suppressed": True}}, "owner:1"),
    ({"chat_id": 4242, "_is_direct_chat": True, "metadata": {"origin_message_ref": {"client_message_id": "m"}},
      "_execution_owner": {"kind": "direct", "lane": "conversation", "executor": "worker"}}, "owner:4242"),
    ({"chat_id": 1, "_is_direct_chat": True, "metadata": {"origin_suppressed": True},
      "_execution_owner": {"kind": "direct", "lane": "conversation", "executor": "inline_degraded"}}, "owner:1"),
    ({"chat_id": 1, "_is_direct_chat": True, "metadata": {"origin_suppressed": True},
      "_execution_owner": {"kind": "direct", "lane": "conversation", "executor": "inline_call"}}, ""),
    ({"chat_id": 1, "_is_direct_chat": True,
      "metadata": {"wake_cause": "own_clock", "initiator": "consciousness"}}, ""),  # a wake: no owner door
    ({"chat_id": 1, "_is_direct_chat": True, "metadata": {}}, ""),  # an auto-resume or recovery turn: no owner door
    ({"chat_id": 1, "_presence_turn": True, "metadata": {"origin_suppressed": True}}, ""),
    ({"chat_id": 1, "_is_direct_chat": True, "project_id": "room", "metadata": {"origin_suppressed": True}}, ""),
    ({"chat_id": 1, "_is_direct_chat": True, "detached_at": "t", "metadata": {"origin_suppressed": True}}, ""),
    ({"chat_id": 1, "_is_direct_chat": True, "metadata": {"origin_suppressed": True},
      "_execution_owner": {"kind": "direct", "lane": "project", "executor": "worker"}}, ""),
    ({"chat_id": 1, "metadata": {"origin_suppressed": True}}, ""),  # pooled work
    ({"chat_id": 0, "_is_direct_chat": True, "metadata": {"origin_suppressed": True}}, ""),
])
def test_only_my_humans_own_conversation_turn_reads_its_chat(task, key):
    from ouroboros.conversation_lease import owner_conversation_key

    assert owner_conversation_key(task) == key
