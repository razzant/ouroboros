from __future__ import annotations

import asyncio
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any

import pytest


SCRIPTS_DIR = Path(__file__).parents[1] / "skills" / "telegram" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import companion  # noqa: E402
import public_gateway as gateway  # noqa: E402


class FakeStream:
    def __init__(self, response: Any) -> None:
        self.response = response

    async def __aenter__(self) -> Any:
        return self.response

    async def __aexit__(self, *_args: Any) -> None:
        return None


class FakeStreamResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.body = body

    def aiter_bytes(self):
        body = self.body

        class Iterator:
            done = False

            def __aiter__(self):
                return self

            async def __anext__(self) -> bytes:
                if self.done:
                    raise StopAsyncIteration
                self.done = True
                return body

        return Iterator()


def write_config(path: Path, **updates: Any) -> None:
    payload: dict[str, Any] = {
        "schema": 2,
        "core_port": 8765,
        "owner_chat_id": 12345,
        "button_text": "Ouroboros",
        "tunnel": "cloudflare_quick",
    }
    payload.update(updates)
    (path / companion._CONFIG_NAME).write_text(json.dumps(payload), encoding="utf-8")


def test_runtime_config_is_exact_and_has_no_user_origin(tmp_path: Path) -> None:
    state = companion._safe_state_dir(tmp_path / "state")
    write_config(state)
    assert companion.load_runtime_config(state) == {
        "core_port": 8765,
        "owner_chat_id": 12345,
        "button_text": "Ouroboros",
    }
    write_config(state, origin="http://127.0.0.1:8765")
    with pytest.raises(companion.CompanionError, match="unexpected shape"):
        companion.load_runtime_config(state)


def test_runtime_config_rejects_symlink(tmp_path: Path) -> None:
    state = companion._safe_state_dir(tmp_path / "state")
    target = tmp_path / "elsewhere.json"
    target.write_text("{}", encoding="utf-8")
    (state / companion._CONFIG_NAME).symlink_to(target)
    with pytest.raises(companion.CompanionError, match="unsafe|escaped"):
        companion.load_runtime_config(state)


def test_status_is_bounded_and_private(tmp_path: Path) -> None:
    state = companion._safe_state_dir(tmp_path / "state")
    status = companion.RuntimeStatus(state, cloudflared_version="2026.7.2")
    status.transition(
        "ready",
        "x" * 500,
        reason_code="healthy",
        public_url="https://abc.trycloudflare.com/",
    )
    path = state / companion._STATUS_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["state"] == "ready"
    assert len(payload["message"]) == 300
    assert payload["reason_code"] == "healthy"
    assert payload["updated_at_epoch"] > 0
    if sys.platform != "win32":
        assert path.stat().st_mode & 0o777 == 0o600


def test_singleton_waits_boundedly_and_releases(tmp_path: Path) -> None:
    state = companion._safe_state_dir(tmp_path / "state")
    first = companion.acquire_singleton(state, timeout_sec=0)
    try:
        started = time.monotonic()
        with pytest.raises(companion.CompanionError, match="still shutting down"):
            companion.acquire_singleton(state, timeout_sec=0.05)
        assert time.monotonic() - started < 0.5
    finally:
        companion.release_singleton(first)
    second = companion.acquire_singleton(state, timeout_sec=0)
    companion.release_singleton(second)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group contract")
def test_parent_lifeline_requests_stop_then_cancels_hard_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        parent = 4242
        parent_alive = [True]
        killed: list[tuple[int, int]] = []
        monkeypatch.setattr(companion.os, "getppid", lambda: parent if parent_alive[0] else 1)
        monkeypatch.setattr(companion, "_pid_alive", lambda _pid: parent_alive[0])
        monkeypatch.setattr(companion.os, "getpgrp", lambda: companion.os.getpid())
        monkeypatch.setattr(companion.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
        stop = asyncio.Event()
        cancel, thread = companion.start_parent_lifeline(
            parent,
            asyncio.get_running_loop(),
            stop,
            hard_kill_after_sec=0.25,
        )
        parent_alive[0] = False
        await asyncio.wait_for(stop.wait(), timeout=1)
        cancel.set()
        thread.join(timeout=1)
        assert killed == []

    asyncio.run(scenario())


def test_core_health_retries_cold_boot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = [0]

    class Bridge:
        state_dir = tmp_path / "observer-state"
        def owner_chat_id(self) -> int:
            return 12345

        def safe_for_exposure(self, owner: int) -> bool:
            return owner == 12345

    class Status:
        def transition(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    async def fake_probe_core(_port: int) -> bool:
        attempts[0] += 1
        return attempts[0] >= 3

    async def fake_wait_or_stop(_delay: float, _stop: asyncio.Event, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(companion, "probe_core", fake_probe_core)
    monkeypatch.setattr(companion, "_wait_or_stop", fake_wait_or_stop)
    ready = asyncio.run(
        companion._wait_for_core_and_owner(
            Bridge(),  # type: ignore[arg-type]
            12345,
            8765,
            Status(),  # type: ignore[arg-type]
            asyncio.Event(),
        )
    )
    assert ready is True
    assert attempts[0] == 3


def test_doh_parser_accepts_public_ipv4_and_rejects_bad_answers() -> None:
    host = "fresh-name.trycloudflare.com"
    valid = {
        "Status": 0,
        "TC": False,
        "Question": [{"name": f"{host}.", "type": 1}],
        "Answer": [
            {"name": f"{host}.", "type": 5, "data": "edge.example.net."},
            {"name": "edge.example.net.", "type": 1, "data": "104.16.230.132"},
            {"name": "edge.example.net.", "type": 1, "data": "104.16.230.132"},
            {"name": "edge.example.net.", "type": 28, "data": "2606:4700::1"},
        ],
    }
    assert gateway._parse_doh_ipv4_response(json.dumps(valid).encode(), host) == (
        "104.16.230.132",
    )

    invalid_payloads = [
        {**valid, "Status": 3},
        {**valid, "TC": True},
        {**valid, "Question": [{"name": "wrong.example", "type": 1}]},
        {**valid, "Answer": [{"name": host, "type": 1, "data": "127.0.0.1"}]},
        {**valid, "Answer": [{"name": host, "type": 1, "data": "224.0.0.1"}]},
        {**valid, "Answer": [{"name": host, "type": 1, "data": "not-an-ip"}]},
    ]
    for payload in invalid_payloads:
        with pytest.raises(gateway._PublicProbeError):
            gateway._parse_doh_ipv4_response(json.dumps(payload).encode(), host)
    with pytest.raises(gateway._PublicProbeError, match="oversized"):
        gateway._parse_doh_ipv4_response(
            b"{" + b" " * gateway._MAX_DOH_RESPONSE_BYTES + b"}",
            host,
        )


def test_doh_request_uses_literal_ip_with_cloudflare_host_and_sni(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = "fresh-name.trycloudflare.com"
    calls: list[tuple[Any, ...]] = []
    payload = json.dumps(
        {
            "Status": 0,
            "TC": False,
            "Question": [{"name": host, "type": 1}],
            "Answer": [{"name": host, "type": 1, "data": "104.16.230.132"}],
        }
    ).encode()

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        def stream(self, method: str, url: str, **kwargs: Any) -> FakeStream:
            calls.append((method, url, kwargs))
            return FakeStream(
                FakeStreamResponse(
                    headers={"content-type": "application/dns-json"},
                    body=payload,
                )
            )

    monkeypatch.setattr(gateway.httpx, "AsyncClient", lambda **_kwargs: Client())
    assert asyncio.run(gateway._resolve_public_ipv4_via_doh(host, timeout_sec=1)) == (
        "104.16.230.132",
    )
    method, url, kwargs = calls[0]
    assert method == "GET"
    assert url == "https://1.1.1.1/dns-query"
    assert kwargs["params"] == {"name": host, "type": "A"}
    assert kwargs["headers"]["Host"] == "cloudflare-dns.com"
    assert kwargs["extensions"] == {"sni_hostname": "cloudflare-dns.com"}


def test_direct_gateway_probe_preserves_quick_host_and_tls_sni(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        def stream(self, method: str, url: str, **kwargs: Any) -> FakeStream:
            calls.append((method, url, kwargs))
            return FakeStream(
                FakeStreamResponse(
                    headers={gateway.GATEWAY_MARKER_HEADER: "1"},
                    body=b"bootstrap",
                )
            )

    monkeypatch.setattr(gateway.httpx, "AsyncClient", lambda **_kwargs: Client())
    host = "fresh-name.trycloudflare.com"
    assert asyncio.run(
        gateway._probe_gateway_via_ipv4(host, "104.16.230.132", timeout_sec=1)
    )
    method, url, kwargs = calls[0]
    assert method == "GET"
    assert url == "https://104.16.230.132/"
    assert kwargs["headers"]["Host"] == host
    assert kwargs["extensions"] == {"sni_hostname": host}


def test_gateway_response_requires_exact_marker_status_and_bounded_body() -> None:
    async def scenario() -> None:
        marker = {gateway.GATEWAY_MARKER_HEADER: "1"}
        assert await gateway._gateway_response_ready(
            FakeStreamResponse(headers=marker, body=b"ok")
        )
        assert not await gateway._gateway_response_ready(
            FakeStreamResponse(status_code=302, headers=marker, body=b"redirect")
        )
        assert not await gateway._gateway_response_ready(
            FakeStreamResponse(headers={gateway.GATEWAY_MARKER_HEADER: "0"}, body=b"wrong")
        )
        assert not await gateway._gateway_response_ready(
            FakeStreamResponse(
                headers=marker,
                body=b"x" * (gateway._MAX_GATEWAY_RESPONSE_BYTES + 1),
            )
        )

    asyncio.run(scenario())


def test_public_gateway_normal_success_never_calls_doh(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

    class Tunnel:
        returncode = None

    async def success(*_args: Any, **_kwargs: Any) -> bool:
        return True

    async def forbidden(*_args: Any, **_kwargs: Any) -> tuple[str, ...]:
        raise AssertionError("DoH must not run after a normal successful probe")

    monkeypatch.setattr(gateway.httpx, "AsyncClient", lambda **_kwargs: Client())
    monkeypatch.setattr(gateway, "_probe_gateway_normally", success)
    monkeypatch.setattr(gateway, "_resolve_public_ipv4_via_doh", forbidden)
    asyncio.run(
        gateway.verify_public_gateway(
            "https://fresh-name.trycloudflare.com/",
            Tunnel(),
            timeout_sec=1,
        )
    )


def test_public_gateway_dns_failure_uses_doh_but_marker_failure_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

    class Tunnel:
        returncode = None

    doh_calls: list[str] = []
    direct_calls: list[str] = []

    async def dns_failure(*_args: Any, **_kwargs: Any) -> bool:
        request = gateway.httpx.Request("GET", "https://fresh-name.trycloudflare.com/")
        try:
            raise socket.gaierror(8, "name unavailable")
        except socket.gaierror as exc:
            raise gateway.httpx.ConnectError("dns failed", request=request) from exc

    async def resolve(host: str, **_kwargs: Any) -> tuple[str, ...]:
        doh_calls.append(host)
        return "104.16.230.132", "104.16.231.132"

    async def direct(_host: str, address: str, **_kwargs: Any) -> bool:
        direct_calls.append(address)
        return address.endswith("231.132")

    monkeypatch.setattr(gateway.httpx, "AsyncClient", lambda **_kwargs: Client())
    monkeypatch.setattr(gateway, "_probe_gateway_normally", dns_failure)
    monkeypatch.setattr(gateway, "_resolve_public_ipv4_via_doh", resolve)
    monkeypatch.setattr(gateway, "_probe_gateway_via_ipv4", direct)
    asyncio.run(
        gateway.verify_public_gateway(
            "https://fresh-name.trycloudflare.com/",
            Tunnel(),
            timeout_sec=1,
        )
    )
    assert doh_calls == ["fresh-name.trycloudflare.com"]
    assert direct_calls == ["104.16.230.132", "104.16.231.132"]

    doh_calls.clear()

    async def marker_failure(*_args: Any, **_kwargs: Any) -> bool:
        return False

    async def forbidden_resolve(*_args: Any, **_kwargs: Any) -> tuple[str, ...]:
        doh_calls.append("unexpected")
        return ("104.16.230.132",)

    monkeypatch.setattr(gateway, "_probe_gateway_normally", marker_failure)
    monkeypatch.setattr(gateway, "_resolve_public_ipv4_via_doh", forbidden_resolve)
    with pytest.raises(gateway.PublicGatewayError, match="did not reach"):
        asyncio.run(
            gateway.verify_public_gateway(
                "https://fresh-name.trycloudflare.com/",
                Tunnel(),
                timeout_sec=0.02,
            )
        )
    assert doh_calls == []


def test_public_gateway_retries_doh_after_initial_dns_propagation_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

    class Tunnel:
        returncode = None

    attempts = 0

    async def dns_failure(*_args: Any, **_kwargs: Any) -> bool:
        request = gateway.httpx.Request("GET", "https://fresh-name.trycloudflare.com/")
        try:
            raise socket.gaierror(8, "name unavailable")
        except socket.gaierror as exc:
            raise gateway.httpx.ConnectError("dns failed", request=request) from exc

    async def resolve(_host: str, **_kwargs: Any) -> tuple[str, ...]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise gateway._PublicProbeError("record not propagated")
        return ("104.16.230.132",)

    async def direct(*_args: Any, **_kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(gateway.httpx, "AsyncClient", lambda **_kwargs: Client())
    monkeypatch.setattr(gateway, "_probe_gateway_normally", dns_failure)
    monkeypatch.setattr(gateway, "_resolve_public_ipv4_via_doh", resolve)
    monkeypatch.setattr(gateway, "_probe_gateway_via_ipv4", direct)
    monkeypatch.setattr(gateway, "_DOH_RETRY_INTERVAL_SEC", 0.01)
    asyncio.run(
        gateway.verify_public_gateway(
            "https://fresh-name.trycloudflare.com/",
            Tunnel(),
            timeout_sec=2,
        )
    )
    assert attempts == 3


def test_public_gateway_stop_cancels_active_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

    class Tunnel:
        returncode = None

    async def scenario() -> None:
        stop = asyncio.Event()
        cancelled = asyncio.Event()

        async def slow_probe(*_args: Any, **_kwargs: Any) -> bool:
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.set()
            return False

        monkeypatch.setattr(gateway.httpx, "AsyncClient", lambda **_kwargs: Client())
        monkeypatch.setattr(gateway, "_probe_gateway_normally", slow_probe)
        task = asyncio.create_task(
            gateway.verify_public_gateway(
                "https://fresh-name.trycloudflare.com/",
                Tunnel(),
                timeout_sec=5,
                stop_event=stop,
            )
        )
        await asyncio.sleep(0.01)
        stop.set()
        with pytest.raises(gateway.PublicGatewayError, match="shutdown was requested"):
            await asyncio.wait_for(task, timeout=0.5)
        await asyncio.wait_for(cancelled.wait(), timeout=0.5)

    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group contract")
def test_companion_requires_host_isolated_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(companion.os, "getpid", lambda: 100)
    monkeypatch.setattr(companion.os, "getpgrp", lambda: 200)
    with pytest.raises(companion.CompanionError, match="isolated process group"):
        companion.require_isolated_process_group()


def test_legacy_lifecycle_ordering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = companion._safe_state_dir(tmp_path / "state")
    write_config(state, core_port=9012)
    events: list[Any] = []

    async def fake_probe_core(port: int, **_kwargs: Any) -> bool:
        events.append(("core", port))
        return True

    class FakeBridge:
        def __init__(self, state_dir: Path, core_port: int) -> None:
            assert state_dir == state and core_port == 9012
            self.state_dir = state_dir

        def owner_chat_id(self) -> int:
            return 12345

        def owned_owner_chat_id(self) -> None:
            return None

        async def activate(self, owner: int) -> bool:
            events.append(("bridge_activate", owner))
            return True

        def safe_for_exposure(self, owner: int) -> bool:
            return owner == 12345

        async def restore(self) -> bool:
            events.append("bridge_restore")
            return True

    class FakeTelegramClient:
        async def aclose(self) -> None:
            events.append("telegram_client_closed")

    class FakeMenu:
        def __init__(self, _token: str, chat_id: int, _state: Path, *, client: Any) -> None:
            assert chat_id == 12345 and isinstance(client, FakeTelegramClient)

        async def verify_private_owner(self) -> str:
            events.append("owner_verified")
            return "@bot"

        async def install(self, url: str, text: str) -> bool:
            events.append(("menu_install", url, text))
            return True

        async def restore(self, **_kwargs: Any) -> bool:
            events.append("menu_restore")
            return True

    class FakeSidecar:
        def __init__(self, _token: str, _owner: int, core_port: int, **_kwargs: Any) -> None:
            assert core_port == 9012
            self.public_url: str | None = None

        def set_public_url(self, url: str) -> None:
            self.public_url = url
            events.append(("sidecar_public", url))

        def clear_public_url(self) -> None:
            self.public_url = None
            events.append("sidecar_closed")

        def diagnostics(self) -> dict[str, int]:
            return {}

    class FakeServer:
        should_exit = False

    server_task: asyncio.Task[None] | None = None

    async def fake_start_sidecar(_sidecar: FakeSidecar):
        nonlocal server_task

        async def forever() -> None:
            await asyncio.Event().wait()

        server_task = asyncio.create_task(forever())
        events.append(("sidecar_port", 45678))
        return FakeServer(), server_task, 45678

    async def fake_ensure(state_dir: Path) -> Path:
        assert state_dir == state
        events.append("binary_verified")
        return state / "verified-cloudflared"

    class FakeTunnel:
        def __init__(self, _binary: Path, _state: Path, sidecar_port: int) -> None:
            # The only origin input is the random sidecar port, never core 9012.
            assert sidecar_port == 45678
            self.returncode: int | None = None

        async def start(self) -> None:
            events.append("tunnel_started")

        async def wait_url(self, timeout_sec: float) -> str:
            assert timeout_sec == 30
            return "https://abc.trycloudflare.com/"

        async def stop(self) -> None:
            events.append("tunnel_stopped")

    async def fake_public(url: str, _tunnel: FakeTunnel, **_kwargs: Any):
        events.append(("public_verified", url))
        return companion.PublicProbeOutcome.READY

    async def fake_monitor(
        _url: str,
        _tunnel: FakeTunnel,
        context: companion._GenerationContext,
    ) -> tuple[Any, bool]:
        events.append("ready_monitor")
        context.stop_event.set()
        return companion._GenerationResult.RECONNECT, False

    async def fake_stop_sidecar(*_args: Any) -> None:
        assert server_task is not None
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)
        events.append("sidecar_stopped")

    monkeypatch.setenv("OUROBOROS_SKILL_STATE_DIR", str(state))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setattr(companion, "require_isolated_process_group", lambda: None)
    monkeypatch.setattr(companion, "probe_core", fake_probe_core)
    monkeypatch.setattr(companion, "TelegramSettingsObserver", FakeBridge)
    monkeypatch.setattr(companion.httpx, "AsyncClient", lambda **_kwargs: FakeTelegramClient())
    monkeypatch.setattr(companion, "TelegramMenuManager", FakeMenu)
    monkeypatch.setattr(companion, "TelegramProxySidecar", FakeSidecar)
    monkeypatch.setattr(companion, "start_sidecar", fake_start_sidecar)
    monkeypatch.setattr(companion, "ensure_cloudflared", fake_ensure)
    monkeypatch.setattr(companion, "QuickTunnel", FakeTunnel)
    monkeypatch.setattr(companion, "probe_public_gateway_once", fake_public)
    monkeypatch.setattr(companion, "_monitor_generation", fake_monitor)
    monkeypatch.setattr(companion, "stop_sidecar", fake_stop_sidecar)

    assert asyncio.run(companion.run()) == 0
    assert ("core", 9012) in events
    assert events.index(("public_verified", "https://abc.trycloudflare.com/")) < events.index(
        ("menu_install", "https://abc.trycloudflare.com/", "Ouroboros")
    )
    assert "menu_restore" in events
    assert "tunnel_stopped" in events
    assert "sidecar_stopped" in events


def test_disable_stops_sidecar_restores_menu_once_and_returns_to_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = companion._safe_state_dir(tmp_path / "state-disabled")
    write_config(state, core_port=9012)
    calls = {"owner": 0, "sidecar_stop": 0, "menu_restore": 0}

    class SettingsObserver:
        def __init__(self, _state: Path, _port: int) -> None: pass

    class Client:
        async def aclose(self) -> None: pass

    class Menu:
        async def restore(self, **_kwargs: Any) -> bool:
            calls["menu_restore"] += 1
            return True

    class Sidecar:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None: pass
        def clear_public_url(self) -> None: pass
        def diagnostics(self) -> dict[str, int]: return {}

    class Server:
        should_exit = False

    server_task: asyncio.Task[None] | None = None

    async def wait_owner(*_args: Any, **_kwargs: Any):
        calls["owner"] += 1
        if calls["owner"] == 1:
            return 12345, Menu()
        raise companion.ShutdownRequested("idle after disable")

    async def start_sidecar(_sidecar: Sidecar):
        nonlocal server_task
        server_task = asyncio.create_task(asyncio.Event().wait())
        return Server(), server_task, 45678

    async def stop_sidecar(*_args: Any) -> None:
        if len(_args) >= 3 and _args[2] is None:
            return
        calls["sidecar_stop"] += 1
        assert server_task is not None
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)

    async def disabled_generation(*_args: Any, **_kwargs: Any):
        return companion._GenerationResult.DISABLED

    monkeypatch.setenv("OUROBOROS_SKILL_STATE_DIR", str(state))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setattr(companion, "require_isolated_process_group", lambda: None)
    monkeypatch.setattr(companion, "TelegramSettingsObserver", SettingsObserver)
    monkeypatch.setattr(companion.httpx, "AsyncClient", lambda **_kwargs: Client())
    monkeypatch.setattr(companion, "_wait_for_owner", wait_owner)
    monkeypatch.setattr(companion, "_prior_owner_reconciliation", lambda *_args: (False, None))
    monkeypatch.setattr(companion, "_wait_for_cloudflared", lambda *_args, **_kwargs: _async_value(state / "cloudflared"))
    monkeypatch.setattr(companion, "_wait_for_core_and_owner", lambda *_args, **_kwargs: _async_value(True))
    monkeypatch.setattr(companion, "TelegramProxySidecar", Sidecar)
    monkeypatch.setattr(companion, "start_sidecar", start_sidecar)
    monkeypatch.setattr(companion, "_run_tunnel_generations", disabled_generation)
    monkeypatch.setattr(companion, "stop_sidecar", stop_sidecar)

    assert asyncio.run(companion.run()) == 0
    assert calls == {"owner": 2, "sidecar_stop": 1, "menu_restore": 1}


async def _async_value(value: Any) -> Any:
    return value
