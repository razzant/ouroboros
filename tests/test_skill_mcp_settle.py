"""Lifecycle completion owns its work even with a slow first MCP import."""
import asyncio
import builtins
import threading
import time


def test_lifecycle_without_bound_mcp_never_starts_sdk_settle(tmp_path, monkeypatch):
    from ouroboros import config, skill_lifecycle_queue

    monkeypatch.setattr(config, "load_settings", lambda: {"MCP_SERVERS": []})
    entered = threading.Event()
    imported, threads = [], []
    original_import, original_start = builtins.__import__, threading.Thread.start

    def slow_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ouroboros" and "mcp_client" in (fromlist or ()):
            imported.append(name)
            entered.set()
            time.sleep(2.5)  # a cold SDK import longer than the thread-hygiene grace
        return original_import(name, globals, locals, fromlist, level)

    def start(thread):
        if thread.name.startswith("mcp-"):
            threads.append(thread)
        return original_start(thread)

    monkeypatch.setattr(builtins, "__import__", slow_import)
    monkeypatch.setattr(threading.Thread, "start", start)

    async def done():
        return "done"

    try:
        result = asyncio.run(skill_lifecycle_queue.run_lifecycle_job(
            kind="review", target="no-mcp", runner=done,
            options=skill_lifecycle_queue.LifecycleJobOptions(drive_root=tmp_path)))
        assert result == "done"
        entered.wait(0.1)
        assert not threads and not imported, "no bound server: no settle thread or MCP SDK import"
    finally:
        for thread in threads:
            thread.join(10)
            assert not thread.is_alive()
