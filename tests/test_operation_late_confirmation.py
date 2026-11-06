"""Consumer regression: exact external confirmation cannot release a live runner."""
import threading

import pytest

from ouroboros import budget_pause, owner_pause, tool_custody
from ouroboros.presence_continuation import _custody_blockers
from tests.test_batch4_producer_custody import _registry
from tests.test_invocation_ownership import claims, owner, run



@pytest.mark.parametrize("bound", [False, True])
def test_late_confirmation_waits_for_actual_runner_hold(tmp_path, monkeypatch, bound):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    confirm_now, confirmed, finish = threading.Event(), threading.Event(), threading.Event()
    runners, outcomes, errors, calls = [], [], [], []

    def body(*_args, **_kwargs):
        calls.append(True)
        confirm = tool_custody.external_handoff("nested transport")
        release_hold = tool_custody.hold_invocation()
        outcomes.append(owner_pause._TOOL_OPERATION.get()[2])

        def runner():
            try:
                assert confirm_now.wait(5)
                confirm()
                confirmed.set()
                assert finish.wait(5)
            except BaseException as exc:
                errors.append(exc)
            finally:
                release_hold()

        thread = threading.Thread(target=runner, name="held-operation-runner")
        runners.append(thread)
        thread.start()
        return "caller returned; runner still owned"

    registry.override_handler("knowledge_read", body)
    try:
        if bound:
            with owner(registry):
                result = run(registry)["tool_result"]
        else:
            result = registry.execute_result("knowledge_read", {"topic": "probe"})
        assert result.text == "caller returned; runner still owned" and calls == [True]
        assert outcomes[0]["closed"] and len(claims(tmp_path)) == 1
        if bound:
            row = outcomes[0]["future_row"]
            assert row["future"].done() and row["holds"] == 1
            assert not row["settled"].is_set()
        confirm_now.set()
        assert confirmed.wait(5) and not errors
        assert runners[0].is_alive()
        assert len(claims(tmp_path)) == 1, "confirmation must not retire the runner's invocation"
        assert not next(iter(claims(tmp_path).values()))["external_pending"]
        assert any(x["kind"] == "tool_handoff" for x in _custody_blockers(tmp_path, "root"))
        if bound:
            assert row["holds"] == 1 and not row["settled"].is_set()
            assert not budget_pause.drain_local_tool_futures(registry._ctx, timeout_sec=0)["drained"]
    finally:
        confirm_now.set()
        finish.set()
        for thread in runners:
            thread.join(5)
            assert not thread.is_alive()
    assert not errors and not claims(tmp_path)
    assert not _custody_blockers(tmp_path, "root")
    if bound:
        assert row["holds"] == 0 and row["settled"].is_set()


def test_async_extension_mcp_join_keeps_enclosing_live_runner(tmp_path, monkeypatch):
    """Real extension timeout and MCP SDK unwind share one Future-owned call."""
    import asyncio

    from ouroboros import extension_process_runner, mcp_client
    from ouroboros.tools import extension_dispatch
    from tests.test_mcp_call_completion import _wire

    registry, _, _ = _registry(tmp_path, monkeypatch)
    method_calls = _wire(monkeypatch, timeout=10)
    initialized, fail_init, mcp_joined, finish = [threading.Event() for _ in range(4)]
    runners, results = [], []
    initialize = mcp_client.ClientSession.initialize

    async def failing_init(session):
        await initialize(session)  # real SDK exchange, before tools/call
        initialized.set()
        assert await asyncio.to_thread(fail_init.wait, 8)
        raise RuntimeError("controlled initialization failure")

    monkeypatch.setattr(mcp_client.ClientSession, "initialize", failing_init)

    async def handler(**_kwargs):
        runners.append(threading.current_thread())
        results.append(mcp_client.get_manager()._call_tool_result("mcp_demo__effect", {}))
        mcp_joined.set()
        assert finish.wait(8)
        return "runner finished"

    descriptor = {"name": "ext_probe_nested", "handler": handler, "wants_ctx": False,
                  "content_hash": "c" * 64, "skill": "", "timeout_sec": 1}
    monkeypatch.setattr(extension_dispatch, "_extension_dispatch_candidate", lambda *_a: (descriptor, False))
    monkeypatch.setattr(extension_dispatch, "_generation_digest_for", lambda *_a: "d" * 64)
    monkeypatch.setattr(extension_process_runner, "disclose_inprocess_extension_dispatch", lambda *_a, **_k: None)
    try:
        with owner(registry):
            result = run(registry, name="ext_probe_nested", args={}, timeout=6)["tool_result"]
            assert result.code == "EXTENSION_TIMEOUT" and initialized.is_set(), result.text
            assert len(claims(tmp_path)) == 1 and runners[0].is_alive()
            fail_init.set()
            assert mcp_joined.wait(5)
            assert results[0].meta["mcp_call_not_sent"] is True and not method_calls
            assert runners[0].is_alive()
            assert len(claims(tmp_path)) == 1
            assert any(x["kind"] == "tool_handoff" for x in _custody_blockers(tmp_path, "root"))
            assert not budget_pause.drain_local_tool_futures(registry._ctx, timeout_sec=0)["drained"]
    finally:
        fail_init.set()
        finish.set()
        for thread in runners:
            thread.join(8)
            assert not thread.is_alive()
        mcp_client.reset_manager_for_tests()
    assert not claims(tmp_path)
    assert not _custody_blockers(tmp_path, "root")
