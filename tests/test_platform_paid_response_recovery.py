"""A money-store refusal must preserve the response already paid for."""

import asyncio
import errno
import pathlib

import pytest

from ouroboros import platform_layer, usage_accounting, usage_ledger, usage_store
from tests._usage_store_testing import ledger_rows


@pytest.mark.parametrize("tier", ["name", "enforced"])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_paid_response_survives_kernel_lock_refusal(
    tmp_path, monkeypatch, caplog, asynchronous, tier,
):
    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    usage_accounting._reset_task_cache_splits()
    request = usage_accounting.AttemptRequest(
        model="openai/gpt-5.2", provider="openai", reservation_usd=1.0,
        drive_root=root, task_id="child", root_task_id="root", source="test",
    )
    assert platform_layer.kernel_file_locks_enforced(root / usage_ledger.LOCK_REL)
    if tier == "name":
        # A mount whose kernel locks are not enforced: every store access runs
        # under the money name lock, which this platform then refuses.
        monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda _path: False)
    usage_store.migrate_from_journal(root)
    response = {"content": "useful result", "usage": {
        "prompt_tokens": 3, "completion_tokens": 2,
    }}
    sends = 0
    refused = []
    actual_acquire = platform_layer.acquire_exclusive_file_lock
    actual_connect = usage_store._connect

    def name_lock_refused_after_response(path, **kwargs):
        if sends and pathlib.Path(path).name == usage_ledger.LOCK_REL.name:
            refused.append(path)
            if kwargs.get("outcome") is not None:
                kwargs["outcome"].update(reason="kernel_refused", errno=errno.EIO)
            return None
        return actual_acquire(path, **kwargs)

    def store_refused_after_response(*args, **kwargs):
        if sends:
            refused.append(args[0])
            raise usage_ledger.UsageAccountingError("usage store unavailable after response")
        return actual_connect(*args, **kwargs)

    def send():
        nonlocal sends
        sends += 1
        return response

    async def send_async():
        return send()

    with monkeypatch.context() as failure:
        if tier == "name":
            failure.setattr(platform_layer, "acquire_exclusive_file_lock", name_lock_refused_after_response)
        else:
            failure.setattr(usage_store, "_connect", store_refused_after_response)
        if asynchronous:
            actual = asyncio.run(usage_accounting.execute_physical_attempt_async(
                request, send_async,
            ))
        else:
            actual = usage_accounting.execute_physical_attempt(request, send)

    assert actual is response
    assert sends == 1
    assert len(refused) >= 2  # Settlement and the fallback unresolved write both failed.
    assert "Failed to mark post-response accounting failure unresolved" in caplog.text
    rows = ledger_rows(root)
    assert [row["state"] for row in rows] == ["dispatched"]
    projection = usage_accounting.usage_projection(root)
    assert projection["unresolved_upper_bound_usd"] == 1.0
    assert projection["cost_final"] is False
    assert not (root / usage_ledger.LOCK_REL).exists()
    assert len({row["attempt_id"] for row in rows}) == 1
    usage_accounting._reset_task_cache_splits()
    usage_store.forget(root)
