"""A registry tool keeps its unsent claim through physical store contention."""
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import wraps
import threading
import queue

import pytest

from ouroboros import owner_pause, queue_observation, store, task_store
from ouroboros.llm_attempt import PhysicalDispatchInterrupted
from ouroboros.model_wait import task_model_wait_scope
from ouroboros.review_native_episode import inspection_registry
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_admission_contention import database_reader
from tests._store_journal_shared import use_journal

pytestmark = pytest.mark.serial


@pytest.fixture
def reader(tmp_path):
    repo, root = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    (repo / "answer.txt").write_text("Complete source evidence", encoding="utf-8")
    write_task_result(root, "root", "running", root_task_id="root")
    registry, _ctx, _schemas = inspection_registry(str(repo), root, "root")
    calls = []
    original = registry._entries["read_file"].handler

    @wraps(original)
    def read(*args, **kwargs):
        calls.append(threading.get_ident())
        return original(*args, **kwargs)

    registry.override_handler("read_file", read)
    return root, registry, calls


def hold_claim(monkeypatch, stack, root, *, exclusive=False):
    """Acquire an independent SQLite lock only AFTER launch validation passed: on every access
    (``exclusive``), else on the claim's write as the journal mode contends one (a reader holding
    a rollback COMMIT, a writer holding a WAL one)."""
    original = task_store.patch_handoff
    release, attempts, busy = [], [], []

    def contend(*args, **kwargs):
        attempts.append(args[3])
        if not release:
            release.append(stack.enter_context(database_reader(root, exclusive=exclusive, write=not exclusive)))
        try:
            return original(*args, **kwargs)
        except store.StoreBusy as exc:
            busy.append(exc)
            raise

    monkeypatch.setattr(task_store, "patch_handoff", contend)
    return release, attempts, busy


@pytest.mark.parametrize("exclusive", [False, True], ids=["read-blocks-commit", "exclusive-blocks-open"])
def test_real_sql_claim_contention_preserves_one_registry_read(reader, monkeypatch, exclusive):
    if not exclusive:  # a reader holds only a rollback-journal COMMIT; a WAL COMMIT waits for no reader
        use_journal(monkeypatch, "PERSIST")
    root, registry, calls = reader
    waited = []
    with ExitStack() as stack:
        release, attempts, busy = hold_claim(monkeypatch, stack, root, exclusive=exclusive)

        def wait(_seconds):
            assert calls == [] and busy[-1].reason == "contention"
            assert getattr(busy[-1], "rollback_confirmed", None) is not False
            # The whole launch hold has unwound before waiting for the store.
            with owner_pause.launch_lock(root, "root", wait=False):
                waited.append(True)
            release[0]()

        monkeypatch.setattr(queue_observation, "_sleep", wait)
        result = registry.execute_result("read_file", {"path": "answer.txt"})
    assert result.status == "ok" and "Complete source evidence" in result.text
    assert waited == [True] and len(calls) == 1
    assert not load_task_result(root, "root").get("launch_handoffs")
    assert len(set(attempts)) == 1, "claim retry and settlement retain one operation ID"


@pytest.mark.parametrize("handed, control", [
    (False, "pause"), (False, "stop"), (True, "pause"), (True, "stop"),
    (True, "deadline"), (True, "closed"),
])
@pytest.mark.parametrize("journal", ["PERSIST", "WAL"])
def test_wait_rechecks_current_controls_without_revoking_handed_pause(
    reader, monkeypatch, handed, control, journal,
):
    use_journal(monkeypatch, journal)  # a held write fails its COMMIT (rollback journal), its BEGIN (WAL)
    root, registry, calls = reader
    stopped, waited = [], []
    with task_model_wait_scope(task={"id": "root", "_attempt": 1}, drive_root=root,
            event_queue=None, worker_slot_held=False,
            owner_control=lambda: "cancelled" if stopped else None) as owner, ExitStack() as stack:
        release, _attempts, busy = hold_claim(monkeypatch, stack, root)

        def wait(_seconds):
            assert calls == [] and busy[-1].reason == "contention"
            with owner_pause.launch_lock(root, "root", wait=False):
                waited.append(True)
            release[0]()
            # The control arrives AFTER retry_presend's pre-sleep control read.
            if control == "pause":
                owner_pause.install_fence(root, "root", request_id="during-claim-wait")
            elif control == "stop":
                stopped.append(True)
            elif control == "deadline":
                owner.task["deadline_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
            else:
                owner.closed = True

        monkeypatch.setattr(queue_observation, "_sleep", wait)
        with ThreadPoolExecutor(max_workers=1) as executor:
            identity = executor.submit(threading.get_ident).result()
            if handed:
                launch = threading.Event()
                blocker = executor.submit(lambda: launch.wait(5))
                future = owner_pause.submit_tool(registry._ctx, "read_file", executor.submit,
                    registry.execute_result, "read_file", {"path": "answer.txt"})
                launch.set()
                assert blocker.result()
                invoke = future.result
            else:
                invoke = lambda: registry.execute_result("read_file", {"path": "answer.txt"})
            if control in {"stop", "deadline"}:
                with pytest.raises(PhysicalDispatchInterrupted) as caught:
                    invoke()
                assert caught.value.control_reason == ("cancelled" if control == "stop" else "deadline")
            else:
                result = invoke()
                if handed and control == "pause":
                    assert result.status == "ok" and "Complete source evidence" in result.text
                else:
                    assert result.meta["owner_pause_not_started"]
    assert waited == [True]
    assert calls == ([identity] if handed and control == "pause" else [])
    assert not load_task_result(root, "root").get("launch_handoffs")


@pytest.mark.parametrize("error", [store.StoreBusy("unknown", reason="unknown"),
    store.StoreBusy("uninspectable", reason="uninspectable"), store.StoreLost("lost"), PermissionError("denied")])
def test_hard_or_unknown_claim_failure_never_retries_or_enters_handler(reader, monkeypatch, error):
    _root, registry, calls = reader
    attempts = []

    def refuse(*_args):
        attempts.append(True)
        raise error

    monkeypatch.setattr(task_store, "exists", refuse)
    monkeypatch.setattr(queue_observation, "_sleep", lambda _: pytest.fail("hard claim retried"))
    if isinstance(error, store.StoreError):
        result = registry.execute_result("read_file", {"path": "answer.txt"})
        assert result.code == "TOOL_ADMISSION_UNAVAILABLE"
        assert result.meta["owner_pause_not_started"] is True
        physical = getattr(error, "reason", None) or f"{type(error).__name__}: {error}"
        assert result.meta["launch_reason"] == physical and physical in result.text
    else:
        with pytest.raises(type(error)) as caught:
            registry.execute_result("read_file", {"path": "answer.txt"})
        assert caught.value is error
    assert attempts == [True] and calls == []


def test_unconfirmed_rollback_does_not_replay_claim(reader, monkeypatch):
    error = store.StoreBusy("transaction outcome uncertain", reason="contention")
    error.rollback_confirmed = False
    test_hard_or_unknown_claim_failure_never_retries_or_enters_handler(reader, monkeypatch, error)


def test_store_busy_after_handler_entry_never_replays_handler(reader, monkeypatch):
    root, registry, calls = reader

    def entered(_ctx, _resolved_binding=None, **_kwargs):
        calls.append(True)
        raise store.StoreBusy("after the tool's effect", reason="contention")

    registry.override_handler("read_file", entered)
    monkeypatch.setattr(queue_observation, "_sleep", lambda _: pytest.fail("tool body retried"))
    result = registry.execute_result("read_file", {"path": "answer.txt"})
    assert result.status == "error" and "after the tool's effect" in result.text
    assert calls == [True]
    assert not load_task_result(root, "root").get("launch_handoffs")


@pytest.mark.parametrize("tool", ["read_file", "query_code", "send_user_message"])
@pytest.mark.parametrize("contended", [False, True], ids=["uncontended", "read-blocks-commit"])
def test_ordinary_registry_tools_keep_one_real_effect(reader, monkeypatch, tool, contended):
    from ouroboros.tools.registry import ToolRegistry

    if contended:  # the reader-held COMMIT is the rollback journal's (a WAL COMMIT waits for no reader)
        use_journal(monkeypatch, "PERSIST")
    root, inspection, _calls = reader
    registry = ToolRegistry(repo_dir=inspection._ctx.repo_dir, drive_root=root)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    registry._ctx.current_chat_id = 7
    registry._ctx.event_queue = queue.Queue()
    (registry._ctx.repo_dir / "feature.py").write_text("def source_function():\n    return 42\n", encoding="utf-8")
    original, calls, waits = registry._entries[tool].handler, [], []

    @wraps(original)
    def entered(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    registry.override_handler(tool, entered)
    args = {"read_file": {"path": "answer.txt"},
            "query_code": {"op": "symbols", "path": "feature.py"},
            "send_user_message": {"text": "Complete source evidence"}}[tool]
    with ExitStack() as stack:
        if contended:
            release, attempts, busy = hold_claim(monkeypatch, stack, root)

        def wait(_seconds):
            assert contended and calls == [] and busy[-1].reason == "contention"
            assert busy[-1].rollback_confirmed is True
            assert not load_task_result(root, "root").get("launch_handoffs"), "failed COMMIT rolled the claim back"
            with owner_pause.launch_lock(root, "root", wait=False):
                waits.append(True)
            release[0]()

        monkeypatch.setattr(queue_observation, "_sleep", wait)
        result = registry.execute_result(tool, args)
    assert result.status == "ok", result.text
    assert calls == [True] and waits == ([True] if contended else [])
    assert not load_task_result(root, "root").get("launch_handoffs")
    if contended:
        assert len(set(attempts)) == 1
    if tool == "query_code":
        assert "source_function" in result.text
    elif tool == "read_file":
        assert "Complete source evidence" in result.text
    else:
        event = registry._ctx.event_queue.get_nowait()
        assert (event["type"], event["text"], event["task_id"]) == (
            "send_message", "Complete source evidence", "root")
        assert registry._ctx.event_queue.empty() and registry._ctx.pending_events == []
