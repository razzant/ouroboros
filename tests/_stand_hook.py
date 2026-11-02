"""The stand's spy, loaded as ``sitecustomize`` into the isolated server tree only (tests/system_e2e/test_stand_server.py).

It wraps the doors of cognition where they are defined, before any importer takes a name from them, and records every call with
its process and thread: E1 reads which of them ran in the server's own process. It imports nothing of the tree. Knobs, each off
unless its variable is set: ``STAND_DOOR_LOG``: the rows' file; ``STAND_REFUSE_SPAWN``: while this file exists a worker's spawn is
refused, the physical fact the degraded executor answers to; ``STAND_MARKER_WINDOW``: while this file exists the repo-writer gate
reads as an assisted update's, the other such fact: a child started now would import the tree's conflict markers, so its start
is refused and counted (``spawn_in_marker_window``); ``STAND_JOURNAL``: the journal mode every store takes;
``STAND_SAMPLE_LOG``: every quarter second the supervisor's and the event loop's top frames in the server process, counted, and how
late this sampler itself woke, the wait of any thread that wants the interpreter now (reported facts: E5's look at what the door
waits behind).
"""
import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time

_DOORS = {
    "ouroboros.agent": [("OuroborosAgent", "handle_task")],
    "ouroboros.context": [("", "build_llm_messages")],
    "ouroboros.consciousness": [("BackgroundConsciousness", "_context_facts"), ("BackgroundConsciousness", "_launch")],
    "ouroboros.consciousness_wake": [("", "render_wake_message"), ("", "observe_wake")],
    "ouroboros.presence_runner": [("", "run_presence_turn")],
}
_LOG = os.environ.get("STAND_DOOR_LOG", "")
_REFUSE = os.environ.get("STAND_REFUSE_SPAWN", "")
_WINDOW = os.environ.get("STAND_MARKER_WINDOW", "")
_WINDOW_REASON = "stand_marker_window"
_JOURNAL = os.environ.get("STAND_JOURNAL", "")
_SAMPLES = os.environ.get("STAND_SAMPLE_LOG", "")
_WRITE = threading.Lock()


def _task_of(args, kwargs):
    for value in (*args, *kwargs.values()):
        if isinstance(value, dict) and value.get("id"):
            return str(value["id"])
    return ""


def _row(door, task_id, caller):
    row = {"pid": os.getpid(), "door": door, "thread": threading.current_thread().name, "task_id": task_id, "caller": caller, "ts": time.time()}
    try:
        with _WRITE, open(_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    except OSError:
        pass


def _spied(door, call):
    def spy(*args, **kwargs):
        _row(door, _task_of(args, kwargs), sys._getframe(1).f_code.co_name)
        return call(*args, **kwargs)
    return spy


def _patch(module):
    for owner_name, name in _DOORS.get(module.__name__, ()):
        owner = getattr(module, owner_name) if owner_name else module
        setattr(owner, name, _spied(name, getattr(owner, name)))
    if _JOURNAL and module.__name__ == "ouroboros.store":
        module._journal_mode = lambda *_args: _JOURNAL
    if (_REFUSE or _WINDOW) and module.__name__ == "supervisor.worker_process":
        real = module.spawn_worker_process

        def spawn_worker_process(*args, **kwargs):
            if _WINDOW and os.path.exists(_WINDOW):
                _row("spawn_in_marker_window", "", sys._getframe(1).f_code.co_name)
                raise OSError("a child started now would import the tree's conflict markers")
            if _REFUSE and os.path.exists(_REFUSE):
                raise OSError("the stand refuses this spawn")
            return real(*args, **kwargs)

        module.spawn_worker_process = spawn_worker_process
    if _WINDOW and module.__name__ == "supervisor.workers":
        closed = module.repo_writer_admission_closed
        module.repo_writer_admission_closed = lambda: _WINDOW_REASON if os.path.exists(_WINDOW) else closed()
    if _WINDOW and module.__name__ == "supervisor.worker_chat_lane":
        admitted = module.conversation_admitted_during_update
        module.conversation_admitted_during_update = lambda reason: reason == _WINDOW_REASON or admitted(reason)


class _Finder(importlib.abc.MetaPathFinder):
    """After a named module executes, before its importer takes anything from it."""

    NAMES = {*_DOORS, "ouroboros.store", "supervisor.worker_process", "supervisor.workers", "supervisor.worker_chat_lane"}

    def find_spec(self, name, path=None, target=None):
        if name not in self.NAMES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        execute = spec.loader.exec_module

        def patched(module):
            execute(module)
            _patch(module)

        spec.loader.exec_module = patched
        return spec


def _sample():
    """The server process alone (its main module is server.py): top frames of its two threads that matter."""
    counted = {}
    late = {"samples": 0, "total_sec": 0.0, "max_sec": 0.0}
    time.sleep(1.0)
    if os.path.basename(str(getattr(sys.modules.get("__main__"), "__file__", ""))) != "server.py":
        return
    last = time.monotonic()
    while True:
        before = time.monotonic()
        time.sleep(0.25)
        waited = max(0.0, time.monotonic() - before - 0.25)
        late.update(samples=late["samples"] + 1, total_sec=late["total_sec"] + waited, max_sec=max(late["max_sec"], waited))
        frames = sys._current_frames()
        for thread in threading.enumerate():
            if thread.name == "supervisor-main" or thread is threading.main_thread():
                frame = frames.get(thread.ident)
                if frame is not None:
                    key = f"{thread.name}: {frame.f_globals.get('__name__', '?')}:{frame.f_code.co_name}"
                    counted[key] = counted.get(key, 0) + 1
        if time.monotonic() - last > 5.0:
            last = time.monotonic()
            try:
                with open(_SAMPLES, "w", encoding="utf-8") as handle:
                    json.dump({**counted, "sampler_woke_late": late}, handle)
            except OSError:
                pass


if _LOG or _REFUSE or _WINDOW or _JOURNAL:
    sys.meta_path.insert(0, _Finder())
if _SAMPLES:
    threading.Thread(target=_sample, name="stand-sampler", daemon=True).start()
