"""The executor inventory: every place runtime code can start cognition, and whose thread runs it.

The structural twin of tests/test_duty_inventory.py. The class it keeps closed (the 02.11 saturation): the
interpreter that answers the owner runs only bounded work, and every focus of the mind executes in a custodied
process the server admits, routes, observes and can kill. Two rows stay on a server thread by the owner's decision:
``inline_call`` (3A: a live voice call's native session) and ``inline_degraded`` (2A: a carrier physically cannot
start, the same worker task protocol on a server thread, disclosed). "Busy" selects neither.

A *door* (``DOORS``) is a name whose reference starts or carries cognition: an agent's task, its loop or its
construction, a context build, a wake's preparation, a Presence turn's body, the direct lane, the worker's own entry.
``INVENTORY`` names every reference to a door in the runtime tree by (file, enclosing function, door) with the
executor it runs on, so a reference that exists without a row fails by name here, and a row that names a reference
the tree no longer has fails too: both directions, as the duty inventory holds the tick and its table. Line numbers
are no part of a row. A name is what is scanned, not a resolved call: a door reached through a string or a ``getattr``
is not seen, which is why the dynamic drives below and the stand (tests/system_e2e/test_stand_server.py, E1, a real
server with real workers and a spy at each door) hold the law at the doors themselves.

A start that breaks the class could only be a row with the executor ``SERVER_THREAD``, naming the focus it starts, and
the laws below hold that none exists. A row reclassified by hand is still held by the drives and the stand. This table
named 25 such starts on the base (1d88f843b); the conversation lane removed them one focus at a time.
"""

from __future__ import annotations

import ast
import pathlib
import threading
from types import SimpleNamespace
from typing import Dict, Iterable, List, NamedTuple, Set, Tuple

import pytest

from tests._source_scan import parse
from tests.test_conversation_lane_presence import lane as lane  # noqa: F401 -- fixture
from tests.test_owner_wait_pool import pool as pool  # noqa: F401 -- fixture

REPO = pathlib.Path(__file__).resolve().parents[1]
ROOTS = ("ouroboros", "supervisor", "server.py")

# Door -> the one file its name means it in (``None``: anywhere); a review episode has a ``_context_facts`` of its own.
DOORS: Dict[str, str | None] = {
    "worker_main": None,  # a worker process's own body; on a thread it is the degraded carrier
    "handle_task": None, "run_llm_loop": None, "build_llm_messages": None,  # an agent's task, its loop, its context
    "make_agent": None, "OuroborosAgent": None,  # an agent comes into being
    "handle_chat_direct": None, "_run_chat_task": None, "_execute_chat_task": None, "handle_wake_direct": None,  # the lane
    "observe_wake": None, "previous_wake_text": None, "render_wake_message": None, "bind_wake_observation": None,
    "_context_facts": "ouroboros/consciousness.py",  # a wake's preparation
    "run_worker_task": None, "start_inline_turn": None, "run_inline_turn": None,  # the one task body, and the inline rows' doors
    "prepare_conversation_task": None, "prepare_wake": None,  # a turn's first step in its carrier
    "run_presence_turn": None, "carry_presence_turn": None,  # a Presence turn's body
    "_realtime_turn": None,  # a live call's native session, bound to an agent
    "_run_post_task_processing_async": None,  # paid model calls after a task, on their own thread
    "load_extension": None,  # foreign Python imported into this process
}

WORKER = "worker"  # a worker process's own body
INSIDE = "inside_handle_task"  # reached only from a task's body: the row that started the task is its executor
ADMISSION = "admission"  # the server's door: it writes the row and hands the turn to a carrier; the body is not here
INLINE_CALL = "inline_call"  # owner decision 3A
INLINE_DEGRADED = "inline_degraded"  # owner decision 2A
SERVER_THREAD = "server_thread"  # a focus on a server thread: the class this file closes
DISPLAY = "display"  # the same facts read for a panel; nothing is rendered for a model
RESIDUAL = "residual"  # disclosed: not a focus, a class of its own with its own trigger

# Executors a server thread may run cognition on: exactly the owner's two rows.
ALLOWED_ON_A_SERVER_THREAD = {INLINE_CALL, INLINE_DEGRADED}
# Every focus of the mind (owner decision 1A): none may start on a server thread.
OWNER_TURN = "owner_turn"
WAKE = "wake"
PRESENCE = "presence"
AUTO_RESUME = "auto_resume"
FOCI = (AUTO_RESUME, OWNER_TURN, PRESENCE, WAKE)


class Row(NamedTuple):
    executor: str
    why: str
    focus: str = ""


Site = Tuple[str, str, str]  # (file, enclosing function, door)

INVENTORY: Dict[Site, Row] = {
    # The worker process: the home of every focus.
    ("supervisor/worker_process.py", "spawn_worker_process", "worker_main"): Row(
        WORKER, "the pool's one spawn (ctx.Process), ledgered by _record_worker_processes"),
    ("supervisor/worker_process.py", "worker_main", "make_agent"): Row(WORKER, "the worker's own agents"),
    ("supervisor/worker_process.py", "worker_main", "run_worker_task"): Row(WORKER, "the worker's loop, one task at a time"),
    # The server's doors: they admit, and the body runs in a carrier.
    ("ouroboros/server_owner_routing.py", "_route_owner_message", "handle_chat_direct"): Row(
        ADMISSION, "from the intake's own stack; a live call keeps a thread of its own, whose door is run_inline_turn below"),
    ("server.py", "_run_supervisor", "handle_chat_direct"): Row(ADMISSION, "the routing context's door"),
    ("supervisor/worker_chat_lane.py", "auto_resume_after_restart", "handle_chat_direct"): Row(ADMISSION, "the turn after a planned restart"),
    ("ouroboros/consciousness.py", "BackgroundConsciousness._launch", "handle_wake_direct"): Row(
        ADMISSION, "the tick decides when and admits; the wake's preparation is its carrier's first step"),
    # A task's body and what only it reaches: their executor is the row that started the task.
    ("supervisor/worker_process.py", "run_worker_task", "make_agent"): Row(INSIDE, "a task with a drive of its own"),
    ("supervisor/worker_process.py", "_run_task_body", "handle_task"): Row(INSIDE, "the one task body"),
    ("supervisor/worker_process.py", "_run_task_body", "prepare_conversation_task"): Row(INSIDE, "a conversation turn's first step"),
    ("supervisor/worker_process.py", "_run_task_body", "carry_presence_turn"): Row(INSIDE, "a Host-admitted Presence turn"),
    ("supervisor/worker_chat_lane.py", "prepare_conversation_task", "prepare_wake"): Row(INSIDE, "a wake's first step"),
    ("ouroboros/consciousness_wake.py", "prepare_wake", "observe_wake"): Row(INSIDE, "a wake's observation"),
    ("ouroboros/consciousness_wake.py", "prepare_wake", "previous_wake_text"): Row(INSIDE, "the last wake's text"),
    ("ouroboros/consciousness_wake.py", "prepare_wake", "bind_wake_observation"): Row(INSIDE, "the source the wake accepts"),
    ("ouroboros/consciousness_wake.py", "prepare_wake.render", "render_wake_message"): Row(INSIDE, "the wake's message"),
    ("ouroboros/presence_carrier.py", "_runner", "run_presence_turn"): Row(INSIDE, "the carrier's runner"),
    ("ouroboros/presence_carrier.py", "carry_presence_turn", "run_presence_turn"): Row(INSIDE, "the carrier's body for a Host turn"),
    ("ouroboros/presence_carrier.py", "carry_presence_turn.factory", "make_agent"): Row(INSIDE, "its actor"),
    ("ouroboros/presence_carrier.py", "carry_saved_turn", "handle_task"): Row(
        INSIDE, "a moved turn's saved work, in the pool worker the queue dispatched it to"),
    ("ouroboros/presence_carrier.py", "carry_saved_turn.factory", "make_agent"): Row(INSIDE, "its actor"),
    ("ouroboros/presence_runner.py", "run_presence_turn._run_current", "handle_task"): Row(INSIDE, "a Presence turn's body"),
    ("ouroboros/presence_runner.py", "run_presence_turn._run_current", "make_agent"): Row(INSIDE, "its actor"),
    ("ouroboros/agent.py", "OuroborosAgent._prepare_task_context", "build_llm_messages"): Row(INSIDE, "the task's first step"),
    ("ouroboros/agent.py", "OuroborosAgent._handle_task_scoped", "run_llm_loop"): Row(INSIDE, "the task's loop"),
    ("ouroboros/agent.py", "make_agent", "OuroborosAgent"): Row(INSIDE, "the one constructor"),
    ("ouroboros/agent_task_pipeline.py", "_dispatch_root_post_task", "_run_post_task_processing_async"): Row(
        INSIDE, "the root's post-task phase, off the reply path of the process that ran the task"),
    # A live voice call: the loop body is the native session, with a process-local hub and an in-process feature handler.
    ("supervisor/worker_chat_lane.py", "handle_chat_direct", "run_inline_turn"): Row(INLINE_CALL, "an owner's call"),
    ("supervisor/inline_carrier.py", "run_inline_turn", "start_inline_turn"): Row(INLINE_CALL, "its thread"),
    ("supervisor/inline_carrier.py", "start_inline_turn", "_realtime_turn"): Row(INLINE_CALL, "an owner's call binds its turn"),
    ("ouroboros/agent.py", "OuroborosAgent._handle_task_scoped", "_realtime_turn"): Row(INLINE_CALL, "the native session"),
    ("ouroboros/agent.py", "OuroborosAgent._prepare_task_context", "_realtime_turn"): Row(INLINE_CALL, "its context"),
    ("ouroboros/presence_runner.py", "run_presence_turn._run_current", "_realtime_turn"): Row(
        INLINE_CALL, "a Presence call binds its turn"),
    ("ouroboros/presence_call.py", "open_presence_call.run", "run_presence_turn"): Row(
        INLINE_CALL, "a Presence call's body is a live call"),
    # No carrier can start (spawn refused, a marker window, no pool generation; never "busy"): the same body, disclosed.
    ("supervisor/worker_chat_lane.py", "_degrade_if_no_carrier_can_start", "start_inline_turn"): Row(
        INLINE_DEGRADED, "at admission, on inline_carrier.refused_carrier_reason, a physical fact"),
    ("supervisor/conversation_lane.py", "dispatch_awaiting_turns", "start_inline_turn"): Row(
        INLINE_DEGRADED, "a turn already waiting, on the same physical fact"),
    ("supervisor/inline_carrier.py", "start_inline_turn.body", "run_worker_task"): Row(
        INLINE_DEGRADED, "the worker's task body on this process's thread, the pseudo-slot's (a live call's too)"),
    ("supervisor/workers.py", "_get_chat_agent", "make_agent"): Row(INLINE_DEGRADED, "an inline row's fresh actor (a live call's too)"),
    # Not cognition, or not a focus: named so that they stay what they are.
    ("ouroboros/consciousness.py", "BackgroundConsciousness.status_snapshot", "_context_facts"): Row(
        DISPLAY, "the state endpoint's small panel reads the same places, causes and roots; no wake is rendered"),
    ("ouroboros/agent_task_pipeline.py", "recover_pending_root_post_task_synthesis", "_run_post_task_processing_async"): Row(
        RESIDUAL, "a saved root synthesis owed after its worker died: paid calls on a server thread, recovery only, no focus"),
    ("ouroboros/extension_loader.py", "reconcile_extension", "load_extension"): Row(
        RESIDUAL, "pure-Python extensions import in-process (FINAL_PLAN §3.6): a separate class with its own trigger"),
    ("ouroboros/extension_process_runner.py", "_load_child_extension", "load_extension"): Row(WORKER, "the extension's own child"),
}


# ---- the scan ----------------------------------------------------------------------------------------------------

class _References(ast.NodeVisitor):
    """Every ``Name`` or ``Attribute`` whose identifier is a door, with the function that holds it. An import, a
    definition, a string and a comment start nothing."""

    def __init__(self, path: str) -> None:
        self.path, self.scope, self.found = path, [], set()

    def _scoped(self, node: ast.AST, name: str) -> None:
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scoped(node, node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scoped(node, node.name)

    def _reference(self, door: str) -> None:
        if door in DOORS and DOORS[door] in (None, self.path):
            self.found.add((self.path, ".".join(self.scope) or "<module>", door))

    def visit_Name(self, node: ast.Name) -> None:
        self._reference(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._reference(node.attr)
        self.generic_visit(node)


def references_in(source: str, path: str) -> Set[Site]:
    visitor = _References(path)
    visitor.visit(parse(source))
    return visitor.found


def runtime_sources() -> Iterable[pathlib.Path]:
    for name in ROOTS:
        path = REPO / name
        yield from (path.rglob("*.py") if path.is_dir() else (path,))


def scanned_references() -> Set[Site]:
    found: Set[Site] = set()
    for path in runtime_sources():
        found |= references_in(path.read_text(encoding="utf-8"), path.relative_to(REPO).as_posix())
    return found


def _unnamed_or_gone(found: Set[Site], inventory: Iterable[Site]) -> Dict[str, List[Site]]:
    named = set(inventory)
    return {key: rows for key, rows in (("unnamed", sorted(found - named)), ("gone", sorted(named - found))) if rows}


def _on_server_threads(focus: str = "") -> List[Site]:
    return sorted(site for site, row in INVENTORY.items()
                  if row.executor == SERVER_THREAD and focus in ("", row.focus))


# ---- the inventory, both directions ------------------------------------------------------------------------------

def test_every_reference_to_a_door_is_named_and_every_row_still_exists():
    """A new start without a row fails by name (add the row, and it must then justify its executor); a row whose start
    left the tree fails too (delete the row). A site that moved to another function is one of each."""
    assert not (drift := _unnamed_or_gone(scanned_references(), INVENTORY)), drift


def test_the_inventory_check_fails_in_both_directions():
    """A thread started with a door as its target, a direct call and a stored reference each show as a row nobody
    named; a row whose start is gone shows as gone; an import, a definition and a string do not start anything."""
    found = references_in("""
from ouroboros.context import build_llm_messages
def build_llm_messages_note():
    '''handle_task is only mentioned here'''
def starts(worker, ctx):
    threading.Thread(target=handle_chat_direct).start()
    worker.agent.handle_task({})
    ctx.runner = render_wake_message
class Lane:
    def carry(self):
        return run_presence_turn(event=None)
""", "ouroboros/new_module.py")
    assert found == {("ouroboros/new_module.py", "starts", "handle_chat_direct"),
                     ("ouroboros/new_module.py", "starts", "handle_task"),
                     ("ouroboros/new_module.py", "starts", "render_wake_message"),
                     ("ouroboros/new_module.py", "Lane.carry", "run_presence_turn")}
    site = ("ouroboros/new_module.py", "starts", "handle_task")
    assert _unnamed_or_gone(found, [*found, ("gone.py", "f", "handle_task")]) == {"gone": [("gone.py", "f", "handle_task")]}
    assert site in _unnamed_or_gone(found, found - {site})["unnamed"]
    assert not _unnamed_or_gone(found, found)


def test_a_door_named_for_one_file_means_only_that_file():
    source = "def f(self):\n    return self._context_facts()\n"
    assert not references_in(source, "ouroboros/review_native_episode.py")
    assert references_in(source, "ouroboros/consciousness.py") == {("ouroboros/consciousness.py", "f", "_context_facts")}


def test_every_row_names_a_known_executor_and_a_reason():
    executors = {WORKER, INSIDE, ADMISSION, DISPLAY, RESIDUAL, SERVER_THREAD, *ALLOWED_ON_A_SERVER_THREAD}
    for site, row in INVENTORY.items():
        assert row.executor in executors and row.why, site
        assert (row.executor == SERVER_THREAD) is bool(row.focus), site
        assert row.focus in ("", *FOCI), site


# ---- the law over the rows: exactly the owner's two ---------------------------------------------------------------

def test_no_focus_starts_on_a_server_thread(subtests):
    for focus in FOCI:
        with subtests.test(focus=focus):
            assert not (rows := _on_server_threads(focus)), rows


def test_a_server_thread_runs_cognition_only_on_the_two_named_rows():
    """The allowlist is exactly the owner's two rows, 3A and 2A: both exist, and nothing else runs on a server thread."""
    on_threads = {row.executor for row in INVENTORY.values() if row.executor in (SERVER_THREAD, *ALLOWED_ON_A_SERVER_THREAD)}
    assert on_threads == ALLOWED_ON_A_SERVER_THREAD, on_threads


# ---- the doors, driven: what one intake starts in this process ----------------------------------------------------

class CognitionInThisProcess(AssertionError):
    """A door ran in the process that answers the owner."""


class _ImmediateThread:
    """``threading.Thread``, its target run at ``start``: a turn's thread is in the count when the intake returns."""

    depth = 0  # above zero while a thread this intake started is running

    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, daemon=None) -> None:
        self.target, self.args, self.kwargs = target, args, kwargs or {}

    def start(self) -> None:
        _ImmediateThread.depth += 1
        try:
            self.target(*self.args, **self.kwargs)
        finally:
            _ImmediateThread.depth -= 1


def _no_door(started: List[str]) -> None:
    if started:
        raise CognitionInThisProcess(f"this process started {started}")


def _spy(started: List[str], door: str, answer=None):
    def called(*_args, **_kwargs):
        started.append(door)
        return answer
    return called


def _spy_on_a_started_thread(started: List[str], door: str):
    """The lane's door on a thread the intake itself started: the turn is this process's. The same door called from the
    intake's own stack only admits: the turn is a carrier's."""
    def called(*_args, **_kwargs):
        if _ImmediateThread.depth:
            started.append(f"{door} on a thread of this process")
    return called


def test_an_owner_message_starts_no_turn_in_this_process(tmp_path, monkeypatch):
    """The owner's message, routed: nothing in this process runs the turn."""
    from ouroboros import server_owner_routing as routing
    from ouroboros.server_routing_context import RootSnapshot

    started: List[str] = []
    monkeypatch.setattr(threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(routing, "_reserved_project_for_chat", lambda *_a: {})
    monkeypatch.setattr(routing, "addressable_roots", lambda *_a, **_k: RootSnapshot(()))
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, handle_chat_direct=_spy_on_a_started_thread(started, "handle_chat_direct"))
    routing._route_owner_message(SimpleNamespace(), ctx, {"chat_id": 1, "text": "hello", "client_message_id": "m-1"})
    _no_door(started)


def test_auto_resume_starts_no_turn_in_this_process(tmp_path, monkeypatch):
    """The planned restart's verify record is the one trigger of the resume turn: the turn itself runs elsewhere."""
    from supervisor import worker_chat_lane as lane

    started: List[str] = []
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "pending_restart_verify.json").write_text("{}", encoding="utf-8")
    (tmp_path / "logs").mkdir()
    monkeypatch.setattr(threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(lane, "handle_chat_direct", _spy_on_a_started_thread(started, "handle_chat_direct"))
    monkeypatch.setattr(lane, "_pool", lambda: SimpleNamespace(
        DRIVE_ROOT=tmp_path, load_state=lambda: {"owner_chat_id": 7}, chat_turn_liveness=lambda: False))
    monkeypatch.setattr("ouroboros.delegate_recovery.import_legacy_restart", lambda _root: None)
    monkeypatch.setattr("ouroboros.delegate_recovery.pending_owner_restart", lambda _root: None)
    monkeypatch.setattr("ouroboros.installation_stop.panic_input", lambda _root: "")
    lane.auto_resume_after_restart()
    _no_door(started)


@pytest.fixture
def alarm(monkeypatch, tmp_path):
    """A due alarm clock over an in-memory runtime state (tests/test_consciousness.py), its doors spied."""
    from ouroboros import consciousness as clock_module
    from ouroboros import consciousness_wake
    from ouroboros.consciousness import BackgroundConsciousness
    from supervisor import state, workers

    store = {"bg_consciousness_enabled": True, "owner_chat_id": 7}
    monkeypatch.setattr(state, "load_state", lambda: dict(store))
    monkeypatch.setattr(state, "update_state", lambda mutator, **_k: (mutator(store), dict(store))[1])
    monkeypatch.setenv("OUROBOROS_BG_WAKEUP_MIN", "900")
    monkeypatch.setenv("OUROBOROS_BG_WAKEUP_MAX", "14400")
    (tmp_path / "logs").mkdir()
    (tmp_path / "repo" / "prompts").mkdir(parents=True)
    (tmp_path / "repo" / "prompts" / "CONSCIOUSNESS.md").write_text("[Wake-up · {reason}] {events}", encoding="utf-8")
    started: List[str] = []
    monkeypatch.setattr(clock_module, "allowance_window", lambda *_a, **_k: {
        "status": "available", "limit_usd": 20.0, "settled_usd": 2.5, "accounted_usd": 2.5, "remaining_usd": 17.5,
        "resets_at": "", "unknown_unmetered": 0})
    monkeypatch.setattr(workers, "handle_wake_direct", lambda *_a, **_k: {"admitted": True, "task_id": "wake0001", "reason": ""})
    clock = BackgroundConsciousness(tmp_path, tmp_path / "repo", lambda: 7, now=1_800_000_000.0)
    monkeypatch.setattr(clock, "_context_facts", _spy(started, "_context_facts", {"counted_roots": ()}))
    for door in ("observe_wake", "previous_wake_text", "render_wake_message"):
        spy = _spy(started, door, SimpleNamespace(full_text=str))
        monkeypatch.setattr(consciousness_wake, door, spy)
        monkeypatch.setattr(clock_module, door, spy, raising=False)  # the name it took at import, while it still takes it
    return SimpleNamespace(clock=clock, started=started)


def test_a_due_wake_prepares_nothing_on_the_tick(alarm):
    """The tick decides when and admits; the wake's observation, context and render are its carrier's first step
    (one such pass held the supervisor loop 163.8 s on 01.11: 0.009 s of CPU behind other threads)."""
    alarm.clock.tick(now=alarm.clock.next_wake_at + 1)
    _no_door(alarm.started)


def test_a_presence_event_runs_no_turn_body_in_this_process(lane, monkeypatch):
    """The Host's door over a started lane: the body reaches a carrier's queue, and its future resolves from that carrier's
    frames (tests/test_conversation_lane_presence.py); nothing here runs it."""
    from ouroboros import presence_carrier, presence_runner
    from ouroboros.presence_runner import PresenceTurnExecutions
    from tests.test_conversation_lane_presence import _payload
    from tests.test_presence_runner import _admission, _event

    started: List[str] = []
    monkeypatch.setattr(presence_carrier, "run_presence_turn", _spy(started, "run_presence_turn"))
    monkeypatch.setattr(presence_carrier, "carry_presence_turn", _spy(started, "carry_presence_turn"))
    admission, event = _admission(), _event()
    turn_id = presence_runner.presence_turn_task_id(admission.binding_id, event.source_event_id)
    _execution, began = PresenceTurnExecutions().start_or_dispatch(
        turn_id, identity="id-1", reserve=lambda: True, release=lambda: None,
        dispatch=lambda execution: presence_carrier.dispatch_presence_turn(
            execution, drive_root=lane.root, skill_name="telegram-bot", admission=admission, event=event,
            event_payload=_payload(event), staged_files=[], gate_wait=True))
    assert began and lane.spare.in_q.get_nowait()["id"] == turn_id, "the body went to the carrier"
    _no_door(started)


@pytest.mark.parametrize("started, breached", [(["handle_task"], True), ([], False)])
def test_the_drives_fail_on_a_start_and_stay_quiet_without_one(started, breached):
    if breached:
        with pytest.raises(CognitionInThisProcess, match="handle_task"):
            _no_door(started)
    else:
        _no_door(started)
