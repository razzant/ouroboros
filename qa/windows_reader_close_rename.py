"""QA-branch-only Windows witness for the production reader stop boundary.

The standalone reader is real. Observe the exact Win32 termination call made
by reader.stop(), then the same process's exit and released SQLite paths. The
in-process endpoint exercises production history assembly and real child IPC;
it does not claim a Windows HTTP-server/tray Panic or frozen launcher test.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

from ouroboros import dialogue_store, usage_store
from ouroboros.gateway import history_reader
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.store_import import ensure_current_store
from tests._dialogue import confirm_first_use, import_all
from tests._store_shared import release_store_handles
from tests._task_settings_helpers import isolated_runtime  # noqa: F401

QA_TEXT = "QA_4287_RENAME_PROBE"
DATABASES = ("current.sqlite", "dialog.sqlite", "usage.sqlite")


def rename_known_stores(root, phase):
    state = root / "state"
    paths = [state / name for name in DATABASES]
    missing = [path.name for path in paths if not path.is_file()]
    assert not missing, f"WINDOWS_STORE_MISSING_{phase}: {missing}"
    observed = []
    for source in paths:
        moved = source.with_name(source.name + ".qa-4287-rename")
        assert not moved.exists(), f"WINDOWS_RENAME_FIXTURE_OCCUPIED_{phase}: {moved.name}"
        try:
            os.replace(source, moved)
        except OSError as exc:
            raise AssertionError(f"WINDOWS_RENAME_FAILED_{phase}: {source.name} out") from exc
        try:
            os.replace(moved, source)
        except OSError as exc:
            raise AssertionError(f"WINDOWS_RENAME_FAILED_{phase}: {source.name} back") from exc
        assert source.is_file() and not moved.exists(), f"WINDOWS_RENAME_UNVERIFIED_{phase}: {source.name}"
        observed.append(source.name)
    return tuple(observed)


def test_reader_normal_close_releases_store_files_for_rename(isolated_runtime, monkeypatch):  # noqa: F811
    if os.name != "nt" or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted":
        raise RuntimeError("WINDOWS_NATIVE_WITNESS_UNVERIFIED: Windows hosted runner required")
    import _winapi

    root = isolated_runtime.resolve()
    instance = history_reader.HistoryReader()
    monkeypatch.setattr(history_reader, "reader", instance)
    try:
        confirm_first_use(root)
        import_all(root)
        ensure_current_store(root)
        usage_store.migrate_from_journal(root, initialize_journal=True)
        dialogue_store.append(root, "chat", {"direction": "in", "text": QA_TEXT, "chat_id": 1,
                                            "ts": "2026-11-25T19:00:00Z"})
        # The fixture's writes may cache handles in *this* interpreter. Prove
        # they are gone before launching the independent reader. No cross-process cleanup.
        release_store_handles(root)
        before = rename_known_stores(root, "BASELINE_BEFORE_SPAWN")
        endpoint = make_chat_history_endpoint(root)

        async def exercise():
            instance.start(root, Path(__file__).resolve().parents[1])
            assert await asyncio.to_thread(instance.ready.wait, 8), f"WINDOWS_CHILD_NOT_READY: {instance._failure}"
            child = instance._process
            assert child is not None and instance.pid == child.pid != os.getpid(), "WINDOWS_CHILD_PID_UNVERIFIED"
            assert Path(child.args[0]).resolve() == Path(sys._base_executable).resolve(), "WINDOWS_CHILD_INTERPRETER_UNVERIFIED"
            used = []
            token = history_reader.READ_PATH.set(used.append)
            try:
                response = await endpoint(SimpleNamespace(query_params={}))
            finally:
                history_reader.READ_PATH.reset(token)
            assert (response.status_code, used) == (200, ["child"]), ("WINDOWS_CHILD_GET_UNVERIFIED", response.status_code, used)
            assert any(row.get("text") == QA_TEXT for row in json.loads(response.body)["messages"]), "WINDOWS_CHILD_GET_SEED_MISSING"
            # Serving-side collection in GET can reopen test-process handles.
            # Release just those while the actual reader remains alive. stop()
            # must then target its own live native process handle, not an EOF
            # or an unrelated teardown thread's later kill.
            release_store_handles(root)
            main_thread = threading.get_ident()
            child_handle = int(child._handle)
            original_terminate = _winapi.TerminateProcess
            termination = []

            def observed_terminate(handle, code):
                if int(handle) != child_handle or threading.get_ident() != main_thread:
                    return original_terminate(handle, code)
                prior = _winapi.GetExitCodeProcess(handle)
                result = original_terminate(handle, code)
                termination.append((prior, code))
                return result

            with monkeypatch.context() as probe:
                probe.setattr(_winapi, "TerminateProcess", observed_terminate)
                assert child.poll() is None and not child.stdin.closed, "WINDOWS_CHILD_EXITED_BEFORE_STOP"
                instance.stop()
            assert termination == [(_winapi.STILL_ACTIVE, 1)], ("WINDOWS_STOP_SIGNAL_UNVERIFIED", termination)
            instance._thread.join(8)
            assert not instance.live_children(), "WINDOWS_CHILD_STOP_UNCONFIRMED"
            assert child.poll() == 1, ("WINDOWS_CHILD_EXIT_UNVERIFIED", child.returncode)
            assert child.stdin.closed and child.stdout.closed, "WINDOWS_CHILD_PIPES_UNCLOSED"
            return child.pid, child.returncode

        pid, exit_code = asyncio.run(exercise())
        after = rename_known_stores(root, "AFTER_CHILD_STOP")
        assert before == after == DATABASES
        print(json.dumps({"door": "WINDOWS_OWN_HANDLE_STOP_AND_RENAME_VERIFIED", "pid": pid,
                          "child_served_seed": True, "terminate_exit_code": exit_code,
                          "before_spawn": before, "after_stop": after}, sort_keys=True))
    finally:
        instance.stop()
        if instance._thread is not None:
            instance._thread.join(5)
            assert not instance.live_children(), "WINDOWS_TEST_TEARDOWN_UNCONFIRMED"
