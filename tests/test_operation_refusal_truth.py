"""Refusal truth belongs to a whole invocation, not its last nested start."""
import asyncio
import threading
from contextlib import asynccontextmanager, contextmanager
from contextvars import copy_context

import pytest

from ouroboros import mcp_client, owner_pause
from tests.test_batch4_producer_custody import _registry
from tests.test_invocation_ownership import claims, owner, run



@pytest.mark.parametrize("code,status", [("OWNER_OPERATION_REFUSED", "blocked"),
                                        ("TOOL_ADMISSION_UNAVAILABLE", "unavailable")])
def test_text_refusal_roundtrip_keeps_execution_failure(code, status):
    from ouroboros.loop_tool_execution import _extract_result_metadata, _is_tool_execution_failure
    from ouroboros.store import StoreBusy
    from ouroboros.tools.tool_result import LegacyTextResultAdapter, launch_refusal_result

    refusal = (owner_pause.OwnerPauseRefused("owner_pause") if status == "blocked" else
               StoreBusy("admission read failed", reason="permission_denied"))
    original = launch_refusal_result(refusal, not_started=status != "blocked")
    assert original.code == code and original.status == status
    for tool in ("knowledge_read", "mcp_demo__effect"):
        restored = LegacyTextResultAdapter.from_text(tool, original.text)
        assert restored.code == code and restored.status == status
        assert _is_tool_execution_failure(True, original.text, fn_name=tool) is True
        assert _extract_result_metadata(tool, original.text, True)["status"] == status


@pytest.mark.parametrize("bound", [False, True])
def test_post_effect_refusal_preserves_whole_call_phase(tmp_path, monkeypatch, bound):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    marker, later = tmp_path / "effect", []

    def body(*_args, **_kwargs):
        marker.write_text("already happened")
        owner_pause.install_fence(tmp_path, "root", request_id="after-effect")
        with owner_pause.operation_start(registry._ctx):
            later.append(True)

    registry.override_handler("knowledge_read", body)
    if bound:
        with owner(registry):
            result = run(registry)["tool_result"]
    else:
        result = registry.execute_result("knowledge_read", {"topic": "probe"})
    assert marker.read_text() == "already happened" and later == []
    assert result.code == "OWNER_OPERATION_REFUSED"
    assert result.meta["owner_pause_not_started"] is False
    assert "Nothing ran" not in result.text
    assert not claims(tmp_path)


@pytest.mark.parametrize("route", ["builtin", "mcp"])
def test_refusal_keeps_physical_cause_and_positive_no_entry(tmp_path, monkeypatch, route):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    from tests.test_mcp_call_completion import _wire
    calls = []
    if route == "mcp":
        calls = _wire(monkeypatch)

        def refused(*_args, **_kwargs):
            raise owner_pause.LaunchBusy("permission_denied")
        monkeypatch.setattr(owner_pause, "submit_async_preparation", refused)
        name, args = "mcp_demo__effect", {}
    else:
        @contextmanager
        def refused(*_args, **_kwargs):
            raise owner_pause.LaunchBusy("permission_denied")
            yield
        monkeypatch.setattr(owner_pause, "launch_lock", refused)
        registry.override_handler("knowledge_read", lambda *_args, **_kwargs: calls.append(True))
        name, args = "knowledge_read", {"topic": "probe"}
    try:
        result = registry.execute_result(name, args)
        assert not calls
        assert "permission_denied" in result.text and "temporarily" not in result.text
        assert result.meta["launch_reason"] == "permission_denied"
        assert result.meta["control_reason"] == "owner_launch_authority_unavailable"
        assert result.meta["owner_pause_not_started"] is True
        assert not claims(tmp_path)
    finally:
        mcp_client.reset_manager_for_tests()


def test_closed_copied_context_cannot_enter_mcp_transport(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    contexts, observed = [], []

    def body(*_args, **_kwargs):
        contexts.append(copy_context())
        return "joined"
    registry.override_handler("knowledge_read", body)
    assert registry.execute_result("knowledge_read", {"topic": "probe"}).text == "joined"

    @asynccontextmanager
    async def transport(_cfg):
        observed.append("entered")
        yield None, None
    monkeypatch.setattr(mcp_client, "_MCP_SDK_AVAILABLE", True)
    monkeypatch.setattr(mcp_client, "_transport_factory", transport)
    with pytest.raises(owner_pause.OwnerPauseRefused, match="operation_already_returned"):
        contexts[0].run(asyncio.run, mcp_client._call_tool_async(None, "probe", {}, timeout_sec=1))
    assert observed == []


def test_phase_admission_does_not_wait_on_durable_lock(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    held, release, started = (threading.Event() for _ in range(3))
    errors = []
    with owner_pause.tool_handoff(registry._ctx, "probe") as outcome:
        def writer():
            with outcome["lock"]:
                held.set()
                assert release.wait(5)
        thread = threading.Thread(target=writer)
        thread.start()
        assert held.wait(2)
        context = copy_context()

        def start():
            try:
                with owner_pause.operation_start(registry._ctx):
                    started.set()
            except BaseException as exc:
                errors.append(exc)
        starter = threading.Thread(target=lambda: context.run(start))
        starter.start()
        try:
            assert started.wait(2), "durable I/O must not hold the phase admission"
            # The root gate is free while the independent durable writer waits.
            owner_pause.install_fence(tmp_path, "root", request_id="during-durable-write")
        finally:
            release.set()
            starter.join(5)
            thread.join(5)
        assert not errors and not starter.is_alive() and not thread.is_alive()
        outcome["settled"] = True


@pytest.mark.parametrize("close_first", [False, True])
def test_close_and_start_share_one_atomic_phase_boundary(tmp_path, monkeypatch, close_first):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    contexts, outcomes, submitted, errors = [], [], [], []
    ready, release = threading.Event(), threading.Event()

    def own():
        with owner_pause.tool_handoff(registry._ctx, "probe") as outcome:
            contexts.append(copy_context())
            outcomes.append(outcome)
            ready.set()
            assert release.wait(5)
            outcome["settled"] = True
    thread = threading.Thread(target=own)
    thread.start()
    assert ready.wait(2)
    outcome = outcomes[0]
    if close_first:
        release.set()
        thread.join(5)
    else:
        # Keep the admitted action open while the invocation closes: admission
        # is ordered, not a promise that all independent effects have finished.
        outcome["phase_lock"].acquire()

    def start():
        try:
            with owner_pause.operation_start(registry._ctx):
                submitted.append(True)
        except owner_pause.OwnerPauseRefused as exc:
            errors.append(str(exc))
    if close_first:
        contexts[0].run(start)
        assert submitted == [] and errors == ["operation_already_returned"]
    else:
        outcome["phase_lock"].release()
        contexts[0].run(start)
        release.set()
        thread.join(5)
        assert submitted == [True] and errors == []
    assert not thread.is_alive() and outcome["closed"] is True


@pytest.mark.parametrize("failure", ["pause", "pre_popen_error", "candidate_environment"])
def test_retired_service_is_not_whole_call_no_effect(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace

    from ouroboros.tools import services
    registry, _, _ = _registry(tmp_path, monkeypatch)
    ctx = registry._ctx
    marker = tmp_path / "retired"
    key = services._service_key(ctx, "replace")
    monkeypatch.setitem(services._SERVICES, key, SimpleNamespace(proc=SimpleNamespace(poll=lambda: 0)))

    def retire(*_args, **_kwargs):
        marker.write_text("retired previous record")
        services._SERVICES.pop(key)  # a stop that returned frees the name
        return "stopped"
    monkeypatch.setattr(services, "_stop_service", retire)
    monkeypatch.setattr(services, "_executor_can_run_cwd", lambda *_args: True)

    def refused(*_args, **_kwargs):
        if failure == "pause":
            raise owner_pause.OwnerPauseRefused("owner_pause")
        if failure == "candidate_environment":
            from ouroboros.body_candidate import CandidateRefused
            raise CandidateRefused("CANDIDATE_ENVIRONMENT_UNAVAILABLE", "isolation unavailable")
        error = OSError("before new process")
        error.process_not_started = True
        raise error
    monkeypatch.setattr(services, "executor_start_service", refused)
    result = registry.execute_result("start_service", {"cmd": ["ignored"], "name": "replace"})
    assert marker.exists()
    assert result.meta.get("operation_outcome") != "completed_no_effect"
    assert result.meta.get("owner_pause_not_started") is not True
    assert "Nothing ran" not in result.text
    if failure == "pre_popen_error":
        assert result.meta["process_not_started"] is True
        assert result.meta["previous_service_retired"] is True


def test_executor_retirement_is_not_denied_by_missing_host_record(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ouroboros import body_candidate, workspace_executor
    from ouroboros.tools import services

    registry, _, _ = _registry(tmp_path, monkeypatch, backend="local")
    ctx = registry._ctx
    retired = []
    key = workspace_executor.service_key(ctx, "replace")
    old = SimpleNamespace(service_id=key)
    monkeypatch.setattr(services, "_SERVICES", {})
    monkeypatch.setitem(workspace_executor._SERVICES, key, old)
    monkeypatch.setattr(workspace_executor, "_service_state", lambda _record: "exited")

    def retire(record):
        retired.append(record)
        workspace_executor._SERVICES.pop(key)
        return {"state": "exited"}

    def refuse(*_args, **_kwargs):
        error = OSError("environment failed before replacement Popen")
        error.process_not_started = True
        raise error

    monkeypatch.setattr(workspace_executor, "_stop_service_record", retire)
    monkeypatch.setattr(body_candidate, "executor_environment", refuse)
    result = registry.execute_result("start_service", {"cmd": ["ignored"], "name": "replace"})
    assert retired == [old]
    assert result.meta["process_not_started"] is True
    assert result.meta.get("previous_service_retired") is not False
    assert result.meta.get("operation_outcome") != "completed_no_effect"


def test_close_cannot_split_a_live_closed_check_and_start_mark(tmp_path, monkeypatch):
    """Pause exactly after the real start's closed read, before its mark."""
    registry, _, _ = _registry(tmp_path, monkeypatch)
    checked, release, closing, done = (threading.Event() for _ in range(4))
    threads, outcomes, submissions, errors = [], [], [], []
    def own():
        try:
            with owner_pause.tool_handoff(registry._ctx, "probe") as outcome:
                outcomes.append(outcome)
                class Observed:
                    def get(self, key, default=None):
                        value = outcome.get(key, default)
                        if key == "closed" and threading.current_thread().name == "phase-starter":
                            assert not value
                            checked.set()
                            assert release.wait(5)
                        return value
                    def __getitem__(self, key):
                        return outcome[key]
                    def __setitem__(self, key, value):
                        outcome[key] = value
                context = copy_context()
                context.run(owner_pause._TOOL_OPERATION.set, (registry._ctx, "probe", Observed()))
                def start():
                    try:
                        with owner_pause.operation_start(registry._ctx):
                            submissions.append(True)
                    except BaseException as exc:
                        errors.append(exc)
                starter = threading.Thread(target=lambda: context.run(start), name="phase-starter")
                threads.append(starter)
                starter.start()
                assert checked.wait(3)
                outcome["settled"] = True
                closing.set()
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()
    owner_thread = threading.Thread(target=own)
    owner_thread.start()
    try:
        assert closing.wait(5)
        assert not done.wait(.05), "closure must not overtake the checked-but-unmarked start"
        assert outcomes[0].get("closed") is not True
    finally:
        release.set()
        owner_thread.join(5)
        for thread in threads:
            thread.join(5)
    assert not owner_thread.is_alive() and not any(t.is_alive() for t in threads)
    assert errors == [] and submissions == [True]
    assert outcomes[0]["closed"] is True and outcomes[0]["not_started"] is False
