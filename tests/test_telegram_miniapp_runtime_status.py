from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).parents[1] / "skills" / "telegram" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import runtime_status  # noqa: E402


def test_status_transition_is_atomic_bounded_and_hides_stale_url(tmp_path: Path) -> None:
    status = runtime_status.RuntimeStatus(tmp_path, cloudflared_version="2026.7.2")
    status.transition(
        "ready",
        "ready",
        reason_code="healthy",
        public_url="https://abc.trycloudflare.com/",
    )
    ready = json.loads((tmp_path / runtime_status.STATUS_NAME).read_text(encoding="utf-8"))
    assert ready["schema"] == 2
    assert ready["public_url"] == "https://abc.trycloudflare.com/"
    assert ready["last_ready_at_epoch"] > 0
    status.transition("degraded", "x" * 500, reason_code="core_unavailable")
    degraded = json.loads((tmp_path / runtime_status.STATUS_NAME).read_text(encoding="utf-8"))
    assert "public_url" not in degraded
    assert len(degraded["message"]) == 300
    assert degraded["last_ready_at_epoch"] == ready["last_ready_at_epoch"]
    status.transition("disabled", "Mini App is off.", reason_code="miniapp_disabled")
    disabled = json.loads((tmp_path / runtime_status.STATUS_NAME).read_text(encoding="utf-8"))
    assert disabled["state"] == "disabled"


def test_status_exports_only_aggregate_security_metrics(tmp_path: Path) -> None:
    status = runtime_status.RuntimeStatus(
        tmp_path,
        cloudflared_version="2026.7.2",
        metrics=lambda: {"auth_success": 3, "secret": 99},
    )
    status.transition("starting", "ok", reason_code="init")
    payload = json.loads((tmp_path / runtime_status.STATUS_NAME).read_text(encoding="utf-8"))
    assert payload["security"]["auth_success"] == 3
    assert "secret" not in payload["security"]


def test_heartbeat_failure_sets_fatal_event(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def scenario() -> None:
        status = runtime_status.RuntimeStatus(tmp_path, cloudflared_version="2026.7.2")
        stop = asyncio.Event()
        failed = asyncio.Event()
        monkeypatch.setattr(runtime_status, "HEARTBEAT_INTERVAL_SEC", 0.01)
        monkeypatch.setattr(status, "publish", lambda: (_ for _ in ()).throw(
            runtime_status.RuntimeStatusError("disk failed")
        ))
        with pytest.raises(runtime_status.RuntimeStatusError, match="disk failed"):
            await status.heartbeat(stop, failure_event=failed)
        assert failed.is_set()

    asyncio.run(scenario())


def test_heartbeat_republishes_a_waits_deadline_unchanged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A long wait transitions once; each heartbeat refreshes its age and keeps its state, reason, attempt and
    next attempt exactly as published, so a reader sees one deadline for the whole wait."""
    async def scenario() -> list[dict]:
        ticks = iter(range(1_800_000_000, 1_800_000_100))
        monkeypatch.setattr(runtime_status, "time", type("Clock", (), {"time": staticmethod(lambda: next(ticks))}))
        monkeypatch.setattr(runtime_status, "HEARTBEAT_INTERVAL_SEC", 0.01)
        status = runtime_status.RuntimeStatus(tmp_path, cloudflared_version="2026.7.2")
        status.transition("reconnecting", "waiting", reason_code="cloudflare_rate_limited", attempt=2,
                          next_retry_at_epoch=1_800_005_400)
        publish, published, stop = status.publish, [], asyncio.Event()

        def recorded() -> None:
            publish()
            published.append(json.loads((tmp_path / runtime_status.STATUS_NAME).read_text(encoding="utf-8")))
            if len(published) == 3:
                stop.set()

        monkeypatch.setattr(status, "publish", recorded)
        await status.heartbeat(stop)
        return published

    published = asyncio.run(scenario())
    assert [payload["updated_at_epoch"] for payload in published] == [1_800_000_001, 1_800_000_002, 1_800_000_003]
    assert {(payload["state"], payload["reason_code"], payload["attempt"], payload["next_retry_at_epoch"])
            for payload in published} == {("reconnecting", "cloudflare_rate_limited", 2, 1_800_005_400)}
