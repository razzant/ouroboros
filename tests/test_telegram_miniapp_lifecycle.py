from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest


SCRIPTS_DIR = Path(__file__).parents[1] / "skills" / "telegram" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import companion  # noqa: E402


class _Status:
    def __init__(self) -> None:
        self.state = "starting"
        self.reason_code = "initializing"
        self.transitions: list[tuple[str, str]] = []
        self.messages: list[str] = []

    def transition(self, state: str, _message: str, *, reason_code: str, **_kwargs: Any) -> None:
        self.state = state
        self.reason_code = reason_code
        self.transitions.append((state, reason_code))
        self.messages.append(_message)


def test_owner_change_revokes_sidecar_before_monitor_tick(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    settings = state_dir / "settings.json"
    settings.write_text('{"TELEGRAM_CHAT_ID":"12345"}', encoding="utf-8")
    observer = companion.TelegramSettingsObserver(state_dir, 8765)
    gateway = companion.TelegramProxySidecar(
        "12345678:" + "A" * 35,
        12345,
        8765,
        seams=companion.SidecarSeams(exposure_guard=lambda: observer.safe_for_exposure(12345)),
    )
    gateway.set_public_url("https://owner-change.trycloudflare.com/")
    token, _session = gateway._issue_session()

    settings.write_text('{"TELEGRAM_CHAT_ID":"54321"}', encoding="utf-8")

    assert gateway._exposure_is_authorized() is False
    assert gateway.public_url is None
    assert gateway._lookup_session(token) is None


def test_disable_propagates_through_all_long_lifecycle_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        (state_dir / "settings.json").write_text(
            '{"TELEGRAM_CHAT_ID":"12345","TELEGRAM_MINIAPP_ENABLED":"off"}',
            encoding="utf-8",
        )
        observer = companion.TelegramSettingsObserver(state_dir, 8765)
        status = _Status()

        async def forbidden(*_args: Any, **_kwargs: Any):
            raise AssertionError("disabled Mini App reached external work")

        class Tunnel:
            returncode = None
            async def wait(self) -> None:
                await asyncio.Event().wait()

        class Menu:
            install = forbidden
            check_snapshot_owned = forbidden
            check_owned = forbidden

        monkeypatch.setattr(companion, "probe_core", forbidden)
        monkeypatch.setattr(companion, "ensure_cloudflared", forbidden)
        monkeypatch.setattr(companion, "probe_public_gateway_once", forbidden)
        stop = asyncio.Event()
        server_task = asyncio.create_task(asyncio.Event().wait())
        context = companion._GenerationContext(
            menu=Menu(),  # type: ignore[arg-type]
            settings_observer=observer,
            owner=12345,
            button_text="Ouroboros",
            core_port=8765,
            status=status,  # type: ignore[arg-type]
            stop_event=stop,
            server_task=server_task,
        )
        assert await companion._wait_for_core_and_owner(observer, 12345, 8765, status, stop) is companion._GenerationResult.DISABLED
        assert await companion._wait_for_cloudflared(state_dir, status, stop) is companion._GenerationResult.DISABLED
        assert await companion._wait_for_public_verification(
            "https://abc.trycloudflare.com/", Tunnel(), observer, 12345, status, stop, server_task
        ) is companion._GenerationResult.DISABLED
        assert await companion._sync_menu_until_confirmed(
            "https://abc.trycloudflare.com/", Tunnel(), context,
        ) is companion._GenerationResult.DISABLED
        assert await companion._wait_blocked_state(
            companion._GenerationResult.MENU_BLOCKED, Menu(), observer, 12345,
            status, stop, server_task,
        ) is companion._GenerationResult.DISABLED
        assert await companion._monitor_generation(
            "https://abc.trycloudflare.com/", Tunnel(), context,
        ) == (companion._GenerationResult.DISABLED, False)
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)

    asyncio.run(scenario())


class _RunningTunnel:
    returncode = None

    async def wait(self) -> int:
        await asyncio.Event().wait()
        return 0


def test_public_observer_outage_keeps_same_tunnel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        outcomes = iter(
            [
                companion.PublicProbeOutcome.OBSERVER_UNAVAILABLE,
                companion.PublicProbeOutcome.OBSERVER_UNAVAILABLE,
                companion.PublicProbeOutcome.READY,
            ]
        )

        async def probe(*_args: Any, **_kwargs: Any) -> companion.PublicProbeOutcome:
            return next(outcomes)

        async def no_wait(*_args: Any, **_kwargs: Any) -> None:
            return None

        monkeypatch.setattr(companion, "probe_public_gateway_once", probe)
        monkeypatch.setattr(companion, "_wait_or_stop", no_wait)
        status = _Status()
        settings_observer = type(
            "Bridge",
            (),
            {
                "state_dir": tmp_path / "observer-state",
                "owner_chat_id": lambda self: 12345,
                "safe_for_exposure": lambda self, _owner=None: True,
            },
        )()
        assert await companion._wait_for_public_verification(
            "https://abc.trycloudflare.com/",
            _RunningTunnel(),
            settings_observer,  # type: ignore[arg-type]
            12345,
            status,  # type: ignore[arg-type]
            asyncio.Event(),
            asyncio.create_task(asyncio.Event().wait()),
        ) is None
        assert status.transitions.count(("degraded", "public_observer_unavailable")) == 2

    asyncio.run(scenario())


def test_three_confirmed_bad_markers_rotate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        calls = 0

        async def probe(*_args: Any, **_kwargs: Any) -> companion.PublicProbeOutcome:
            nonlocal calls
            calls += 1
            return companion.PublicProbeOutcome.UNHEALTHY_RESPONSE

        async def no_wait(*_args: Any, **_kwargs: Any) -> None:
            return None

        monkeypatch.setattr(companion, "probe_public_gateway_once", probe)
        monkeypatch.setattr(companion, "_wait_or_stop", no_wait)
        settings_observer = type(
            "Bridge",
            (),
            {
                "state_dir": tmp_path / "observer-state",
                "owner_chat_id": lambda self: 12345,
                "safe_for_exposure": lambda self, _owner=None: True,
            },
        )()
        result = await companion._wait_for_public_verification(
            "https://abc.trycloudflare.com/",
            _RunningTunnel(),
            settings_observer,  # type: ignore[arg-type]
            12345,
            _Status(),  # type: ignore[arg-type]
            asyncio.Event(),
            asyncio.create_task(asyncio.Event().wait()),
        )
        assert result is companion._GenerationResult.RECONNECT
        assert calls == 3

    asyncio.run(scenario())


def test_public_verification_fails_closed_when_owner_binding_is_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        class Bridge:
            state_dir = tmp_path / "observer-state"
            def owner_chat_id(self) -> int:
                raise companion.TelegramSettingsError("owner unreadable")

        async def forbidden_probe(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("public probing must not outrun owner revocation checks")

        monkeypatch.setattr(companion, "probe_public_gateway_once", forbidden_probe)
        result = await companion._wait_for_public_verification(
            "https://abc.trycloudflare.com/",
            _RunningTunnel(),
            Bridge(),  # type: ignore[arg-type]
            12345,
            _Status(),  # type: ignore[arg-type]
            asyncio.Event(),
            asyncio.create_task(asyncio.Event().wait()),
        )
        assert result is companion._GenerationResult.SETTINGS_BLOCKED

    asyncio.run(scenario())


def test_owner_unobservable_fails_closed_before_menu_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        class Bridge:
            state_dir = tmp_path / "observer-state"
            def owner_chat_id(self) -> int:
                raise companion.TelegramSettingsError("unreadable")

        class Menu:
            async def install(self, *_args: Any) -> bool:
                raise AssertionError("menu must not be installed without fresh owner binding")

        async def no_wait(*_args: Any, **_kwargs: Any) -> None:
            return None

        monkeypatch.setattr(companion, "_wait_or_stop", no_wait)
        monkeypatch.setattr(companion, "_retry_delay", lambda _count: 0.0)
        server_task = asyncio.create_task(asyncio.Event().wait())
        context = companion._GenerationContext(
            menu=Menu(),  # type: ignore[arg-type]
            settings_observer=Bridge(),  # type: ignore[arg-type]
            owner=12345,
            button_text="Ouroboros",
            core_port=8765,
            status=_Status(),  # type: ignore[arg-type]
            stop_event=asyncio.Event(),
            server_task=server_task,
        )
        result = await companion._sync_menu_until_confirmed(
            "https://abc.trycloudflare.com/",
            _RunningTunnel(),
            context,
        )
        assert result is companion._GenerationResult.SETTINGS_BLOCKED
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)

    asyncio.run(scenario())


def test_sidecar_task_exception_is_typed_for_internal_restart() -> None:
    async def scenario() -> None:
        async def fail() -> None:
            raise RuntimeError("uvicorn failed")

        task = asyncio.create_task(fail())
        await asyncio.sleep(0)
        with pytest.raises(companion.SidecarStoppedError, match="failed unexpectedly"):
            await companion._wait_or_stop(1.0, asyncio.Event(), server_task=task)

    asyncio.run(scenario())


def test_stop_sidecar_consumes_failed_server_task_and_closes_proxy() -> None:
    async def scenario() -> None:
        class Server:
            should_exit = False

        class Sidecar:
            def __init__(self) -> None:
                self.cleared = False
                self.closed = False

            def clear_public_url(self) -> None:
                self.cleared = True

            async def aclose(self) -> None:
                self.closed = True

        async def fail() -> None:
            raise RuntimeError("uvicorn runtime failure")

        server = Server()
        sidecar = Sidecar()
        task = asyncio.create_task(fail())
        await asyncio.sleep(0)
        await companion.stop_sidecar(
            server,  # type: ignore[arg-type]
            task,
            sidecar,  # type: ignore[arg-type]
        )
        assert server.should_exit
        assert sidecar.cleared
        assert sidecar.closed

    asyncio.run(scenario())


def test_stop_sidecar_preserves_caller_cancellation_on_bundled_python() -> None:
    async def scenario() -> None:
        class Server:
            should_exit = False

        class Sidecar:
            def __init__(self) -> None:
                self.closed = False

            def clear_public_url(self) -> None:
                return None

            async def aclose(self) -> None:
                self.closed = True

        async def forever() -> None:
            await asyncio.Event().wait()

        server_task = asyncio.create_task(forever())
        sidecar = Sidecar()
        cleanup = asyncio.create_task(
            companion.stop_sidecar(
                Server(),  # type: ignore[arg-type]
                server_task,
                sidecar,  # type: ignore[arg-type]
            )
        )
        await asyncio.sleep(0)
        cleanup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cleanup
        assert server_task.done()
        assert server_task.cancelled()
        assert sidecar.closed

    asyncio.run(scenario())


def test_stop_sidecar_late_cancellation_still_closes_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        child_finalizing = asyncio.Event()
        release_child = asyncio.Event()

        class Server:
            should_exit = False
            force_exit = False

        class Sidecar:
            def __init__(self) -> None:
                self.closed = False

            def clear_public_url(self) -> None:
                return None

            async def aclose(self) -> None:
                self.closed = True

        async def stubborn_child() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                child_finalizing.set()
                await release_child.wait()

        hard_exits: list[bool] = []
        monkeypatch.setattr(companion, "_SIDECAR_STOP_GRACE_SEC", 0.01)
        monkeypatch.setattr(
            companion,
            "_hard_exit_stuck_cleanup",
            lambda: hard_exits.append(True),
        )
        server = Server()
        sidecar = Sidecar()
        server_task = asyncio.create_task(stubborn_child())
        cleanup = asyncio.create_task(
            companion.stop_sidecar(
                server,  # type: ignore[arg-type]
                server_task,
                sidecar,  # type: ignore[arg-type]
            )
        )
        await asyncio.wait_for(child_finalizing.wait(), timeout=0.2)
        cleanup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cleanup, timeout=0.2)
        assert server.should_exit
        assert server.force_exit
        assert sidecar.closed
        assert hard_exits == [True]
        release_child.set()
        await asyncio.gather(server_task, return_exceptions=True)

    asyncio.run(scenario())


def test_stop_sidecar_bounds_cancellation_resistant_proxy_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        close_started = asyncio.Event()
        release_close = asyncio.Event()
        close_finished = asyncio.Event()

        class Sidecar:
            def clear_public_url(self) -> None:
                return None

            async def aclose(self) -> None:
                close_started.set()
                try:
                    await release_close.wait()
                except asyncio.CancelledError:
                    await release_close.wait()
                finally:
                    close_finished.set()

        hard_exits: list[bool] = []
        monkeypatch.setattr(companion, "_SIDECAR_STOP_GRACE_SEC", 0.01)
        monkeypatch.setattr(
            companion,
            "_hard_exit_stuck_cleanup",
            lambda: hard_exits.append(True),
        )
        cleanup = asyncio.create_task(
            companion.stop_sidecar(
                None,
                None,
                Sidecar(),  # type: ignore[arg-type]
            )
        )
        await asyncio.wait_for(close_started.wait(), timeout=0.2)
        await asyncio.wait_for(cleanup, timeout=0.2)
        assert not close_finished.is_set()
        assert hard_exits == [True]
        release_close.set()
        await asyncio.wait_for(close_finished.wait(), timeout=0.2)

    asyncio.run(scenario())


@pytest.mark.serial
def test_stuck_proxy_close_hard_exits_companion_process() -> None:
    code = """
import asyncio
import companion

companion._SIDECAR_STOP_GRACE_SEC = 0.01

class Sidecar:
    def clear_public_url(self):
        pass

    async def aclose(self):
        event = asyncio.Event()
        try:
            await event.wait()
        except asyncio.CancelledError:
            await event.wait()

asyncio.run(companion.stop_sidecar(None, None, Sidecar()))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(SCRIPTS_DIR)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # Patience for a real interpreter to boot, not the behaviour under test:
        # `_SIDECAR_STOP_GRACE_SEC = 0.01` is what keeps the stuck-close fast, and
        # the assertion below is what proves the hard exit. A 1s budget could not
        # cover Python startup on a loaded Windows runner, so this test failed CI
        # for a reason unrelated to the contract it guards.
        timeout=60.0,
        check=False,
    )
    assert result.returncode == companion._STUCK_CLEANUP_EXIT_CODE


def test_generation_loop_survives_more_than_five_tunnel_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        attempts = 0
        stopped = 0

        class Tunnel:
            returncode = None

            def __init__(self, _binary: Path, _state: Path, sidecar_port: int) -> None:
                assert sidecar_port == 45678

            async def start(self) -> None:
                nonlocal attempts
                attempts += 1
                if attempts <= 6:
                    raise companion.CloudflaredError("transient")

            async def wait_url(self, timeout_sec: float) -> str:
                assert timeout_sec == 30
                return "https://abc.trycloudflare.com/"

            async def stop(self) -> None:
                nonlocal stopped
                stopped += 1

        async def binary(*_args: Any, **_kwargs: Any) -> Path:
            return tmp_path / "cloudflared"

        async def verified(*_args: Any, **_kwargs: Any) -> None:
            return None

        async def synced(*_args: Any, **_kwargs: Any) -> None:
            return None

        async def monitored(*_args: Any, **_kwargs: Any) -> tuple[Any, bool]:
            raise companion.SidecarStoppedError("test complete")

        async def no_wait(*_args: Any, **_kwargs: Any) -> None:
            return None

        monkeypatch.setattr(companion, "QuickTunnel", Tunnel)
        monkeypatch.setattr(companion, "_wait_for_cloudflared", binary)
        monkeypatch.setattr(companion, "_wait_for_public_verification", verified)
        monkeypatch.setattr(companion, "_sync_menu_until_confirmed", synced)
        monkeypatch.setattr(companion, "_monitor_generation", monitored)
        monkeypatch.setattr(companion, "_wait_or_stop", no_wait)
        monkeypatch.setattr(companion, "_retry_delay", lambda _count: 0.0)
        sidecar = type(
            "Sidecar",
            (),
            {"set_public_url": lambda self, _url: None, "clear_public_url": lambda self: None},
        )()
        server_task = asyncio.create_task(asyncio.Event().wait())
        context = companion._GenerationContext(
            menu=object(),  # type: ignore[arg-type]
            settings_observer=type("Settings", (), {"state_dir": tmp_path})(),  # type: ignore[arg-type]
            owner=12345,
            button_text="Ouroboros",
            core_port=9012,
            status=_Status(),  # type: ignore[arg-type]
            stop_event=asyncio.Event(),
            server_task=server_task,
        )
        with pytest.raises(companion.SidecarStoppedError, match="test complete"):
            await companion._run_tunnel_generations(
                tmp_path / "cloudflared",
                tmp_path,
                45678,
                sidecar,  # type: ignore[arg-type]
                context,
            )
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)
        assert attempts == 7
        assert stopped == 7

    asyncio.run(scenario())


def test_generation_backoff_preserves_tunnel_failure_detail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        class Tunnel:
            returncode = 17

            def __init__(self, *_args: Any) -> None:
                pass

            async def start(self) -> None:
                raise companion.CloudflaredError(
                    "Cloudflared exited before publishing a URL (exit 17). Last output: allocation refused"
                )

            async def stop(self) -> None:
                return None

        status = _Status()

        async def inspect_backoff(*_args: Any, **_kwargs: Any) -> None:
            assert status.reason_code == "tunnel_generation_failed"
            assert status.messages[-1].endswith("allocation refused")
            raise companion.ShutdownRequested("test complete")

        monkeypatch.setattr(companion, "QuickTunnel", Tunnel)
        monkeypatch.setattr(companion, "_wait_or_stop", inspect_backoff)
        monkeypatch.setattr(companion, "_retry_delay", lambda _count: 1.0)
        server_task = asyncio.create_task(asyncio.Event().wait())
        context = companion._GenerationContext(
            menu=object(),  # type: ignore[arg-type]
            settings_observer=type("Settings", (), {"state_dir": tmp_path})(),  # type: ignore[arg-type]
            owner=12345,
            button_text="Ouroboros",
            core_port=9012,
            status=status,  # type: ignore[arg-type]
            stop_event=asyncio.Event(),
            server_task=server_task,
        )
        sidecar = type("Sidecar", (), {"clear_public_url": lambda self: None})()
        with pytest.raises(companion.ShutdownRequested, match="test complete"):
            await companion._run_tunnel_generations(
                tmp_path / "cloudflared",
                tmp_path,
                45678,
                sidecar,  # type: ignore[arg-type]
                context,
            )
        assert status.transitions[-1] == ("reconnecting", "tunnel_generation_failed")
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)

    asyncio.run(scenario())


def test_shutdown_cancels_hung_tunnel_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        stopped = asyncio.Event()

        class Tunnel:
            returncode = None

            def __init__(self, *_args: Any) -> None:
                pass

            async def start(self) -> None:
                entered.set()
                await asyncio.Event().wait()

            async def stop(self) -> None:
                stopped.set()

        monkeypatch.setattr(companion, "QuickTunnel", Tunnel)
        stop = asyncio.Event()
        server_task = asyncio.create_task(asyncio.Event().wait())
        context = companion._GenerationContext(
            menu=object(),  # type: ignore[arg-type]
            settings_observer=type("Settings", (), {"state_dir": tmp_path})(),  # type: ignore[arg-type]
            owner=12345,
            button_text="Ouroboros",
            core_port=9012,
            status=_Status(),  # type: ignore[arg-type]
            stop_event=stop,
            server_task=server_task,
        )
        task = asyncio.create_task(
            companion._run_tunnel_generations(
                tmp_path / "cloudflared",
                tmp_path,
                45678,
                type("Sidecar", (), {"clear_public_url": lambda self: None})(),  # type: ignore[arg-type]
                context,
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=0.5)
        stop.set()
        with pytest.raises(companion.ShutdownRequested):
            await asyncio.wait_for(task, timeout=0.5)
        assert stopped.is_set()
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)

    asyncio.run(scenario())


def test_owner_change_retries_menu_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        menu_calls = 0

        class Menu:
            async def restore(self) -> bool:
                nonlocal menu_calls
                menu_calls += 1
                if menu_calls < 3:
                    raise companion.TelegramMenuTransportError("offline")
                return True

        class SettingsObserver:
            pass

        async def no_wait(*_args: Any, **_kwargs: Any) -> None:
            return None

        monkeypatch.setattr(companion, "_wait_or_stop", no_wait)
        monkeypatch.setattr(companion, "_retry_delay", lambda _count: 0.0)
        await companion._reconcile_owner_change(
            Menu(),  # type: ignore[arg-type]
            SettingsObserver(),  # type: ignore[arg-type]
            _Status(),  # type: ignore[arg-type]
            asyncio.Event(),
        )
        assert menu_calls == 3

    asyncio.run(scenario())


def test_cold_start_reconciles_durable_old_owner_before_new_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        observed: list[int] = []

        class Menu:
            def __init__(
                self,
                _token: str,
                owner: int,
                _state: Path,
                *,
                client: Any,
            ) -> None:
                assert client is not None
                observed.append(owner)

        monkeypatch.setattr(companion, "snapshot_owner_chat_id", lambda _state: 111)
        monkeypatch.setattr(companion, "TelegramMenuManager", Menu)

        class Bridge:
            state_dir = tmp_path / "observer-state"
            @staticmethod
            def owned_owner_chat_id() -> int:
                return 111

        needs_reconcile, prior = companion._prior_owner_reconciliation(
            "token",
            222,
            tmp_path,
            object(),  # type: ignore[arg-type]
            Bridge(),  # type: ignore[arg-type]
        )
        assert needs_reconcile
        assert isinstance(prior, Menu)
        assert observed == [111]

    asyncio.run(scenario())


def test_shutdown_during_prior_owner_reconcile_cleans_up_prior_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        state = tmp_path / "state"
        state.mkdir()
        (state / companion._CONFIG_NAME).write_text(
            '{"schema":2,"core_port":8765,"owner_chat_id":0,"button_text":"Ouroboros","tunnel":"cloudflare_quick"}',
            encoding="utf-8",
        )
        restored: list[int] = []

        class Menu:
            def __init__(self, owner: int) -> None:
                self.owner = owner

            async def restore(self, **_kwargs: Any) -> bool:
                restored.append(self.owner)
                return True

        candidate = Menu(222)

        class Client:
            async def aclose(self) -> None:
                return None

        class Bridge:
            state_dir = state
            @staticmethod
            def owned_owner_chat_id() -> int:
                return 111

            async def restore(self) -> bool:
                return True

        class Thread:
            def join(self, timeout: float) -> None:
                assert timeout == 0.5

        async def owner(*_args: Any, **_kwargs: Any) -> tuple[int, Menu]:
            return 222, candidate

        async def stop_during_reconcile(menu: Menu, *_args: Any) -> None:
            assert menu.owner == 111
            raise companion.ShutdownRequested("disable during old-owner rollback")

        monkeypatch.setenv("OUROBOROS_SKILL_STATE_DIR", str(state))
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setattr(companion, "require_isolated_process_group", lambda: None)
        monkeypatch.setattr(companion, "_pid_alive", lambda _pid: True)
        monkeypatch.setattr(
            companion,
            "start_parent_lifeline",
            lambda *_args, **_kwargs: (__import__("threading").Event(), Thread()),
        )
        monkeypatch.setattr(companion.httpx, "AsyncClient", lambda **_kwargs: Client())
        monkeypatch.setattr(companion, "TelegramSettingsObserver", lambda *_args: Bridge())
        monkeypatch.setattr(companion, "_wait_for_owner", owner)
        monkeypatch.setattr(companion, "snapshot_owner_chat_id", lambda _state: 111)
        monkeypatch.setattr(
            companion,
            "TelegramMenuManager",
            lambda _token, old_owner, _state, *, client: Menu(old_owner),
        )
        monkeypatch.setattr(companion, "_reconcile_owner_change", stop_during_reconcile)
        assert await companion.run() == 0
        assert restored == [111]

    asyncio.run(scenario())


def test_sidecar_start_is_cancel_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        instances: list[Any] = []

        class Listener:
            def setsockopt(self, *_args: Any) -> None:
                return None

            def bind(self, _address: Any) -> None:
                return None

            def listen(self, _backlog: int) -> None:
                return None

            def setblocking(self, _value: bool) -> None:
                return None

            def getsockname(self) -> tuple[str, int]:
                return "127.0.0.1", 43123

            def close(self) -> None:
                return None

        class Server:
            def __init__(self, _config: Any) -> None:
                self.started = False
                self.should_exit = False
                self.install_signal_handlers = None
                instances.append(self)

            async def serve(self, *, sockets: list[Any]) -> None:
                assert sockets
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        monkeypatch.setattr(companion.uvicorn, "Server", Server)
        monkeypatch.setattr(
            companion,
            "socket",
            types.SimpleNamespace(
                socket=lambda *_args: Listener(),
                AF_INET=companion.socket.AF_INET,
                SOCK_STREAM=companion.socket.SOCK_STREAM,
                SOL_SOCKET=companion.socket.SOL_SOCKET,
                SO_REUSEADDR=companion.socket.SO_REUSEADDR,
            ),
        )
        sidecar = type("Sidecar", (), {"app": object()})()
        stop = asyncio.Event()
        task = asyncio.create_task(
            companion._run_startup_operation(
                companion.start_sidecar(sidecar),  # type: ignore[arg-type]
                stop_event=stop,
                timeout_sec=6.0,
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=0.5)
        stop.set()
        with pytest.raises(companion.ShutdownRequested):
            await asyncio.wait_for(task, timeout=0.5)
        assert cancelled.is_set()
        assert instances[0].should_exit is True

    asyncio.run(scenario())


def test_sidecar_start_wraps_server_task_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        class Listener:
            def setsockopt(self, *_args: Any) -> None:
                return None

            def bind(self, _address: Any) -> None:
                return None

            def listen(self, _backlog: int) -> None:
                return None

            def setblocking(self, _value: bool) -> None:
                return None

            def getsockname(self) -> tuple[str, int]:
                return "127.0.0.1", 43123

            def close(self) -> None:
                return None

        class Server:
            started = False
            should_exit = False

            def __init__(self, _config: Any) -> None:
                self.install_signal_handlers = None

            async def serve(self, *, sockets: list[Any]) -> None:
                assert sockets
                raise RuntimeError("uvicorn boom")

        monkeypatch.setattr(companion.uvicorn, "Server", Server)
        monkeypatch.setattr(
            companion,
            "socket",
            types.SimpleNamespace(
                socket=lambda *_args: Listener(),
                AF_INET=companion.socket.AF_INET,
                SOCK_STREAM=companion.socket.SOCK_STREAM,
                SOL_SOCKET=companion.socket.SOL_SOCKET,
                SO_REUSEADDR=companion.socket.SO_REUSEADDR,
            ),
        )
        sidecar = type("Sidecar", (), {"app": object()})()
        with pytest.raises(companion.CompanionError, match="failed during startup"):
            await companion.start_sidecar(sidecar)  # type: ignore[arg-type]

    asyncio.run(scenario())


def test_missing_token_publishes_error_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILL_STATE_DIR", str(state))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setattr(companion, "require_isolated_process_group", lambda: None)
    assert asyncio.run(companion.run()) == 1
    payload = __import__("json").loads((state / companion._STATUS_NAME).read_text(encoding="utf-8"))
    assert payload["state"] == "error"
    assert payload["reason_code"] in {"fatal_error", "fatal_error_rollback_pending"}


def test_sidecar_start_retries_beyond_host_restart_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        state = tmp_path / "state"
        state.mkdir()
        (state / companion._CONFIG_NAME).write_text(
            '{"schema":2,"core_port":8765,"owner_chat_id":0,"button_text":"Ouroboros","tunnel":"cloudflare_quick"}',
            encoding="utf-8",
        )
        attempts = 0

        class Client:
            async def aclose(self) -> None:
                return None

        class Sidecar:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                pass

            def diagnostics(self) -> dict[str, int]:
                return {}

            def clear_public_url(self) -> None:
                return None

            async def aclose(self) -> None:
                return None

        class Thread:
            def join(self, timeout: float) -> None:
                assert timeout == 0.5

        class Bridge:
            state_dir = state
            @staticmethod
            def owned_owner_chat_id() -> int:
                return 0

            async def restore(self) -> bool:
                return True

        async def owner(*_args: Any, **_kwargs: Any) -> tuple[int, object]:
            return 12345, type("Menu", (), {"restore": lambda self: _true()})()

        async def _true() -> bool:
            return True

        async def binary(*_args: Any, **_kwargs: Any) -> Path:
            return state / "cloudflared"

        async def mirror(*_args: Any, **_kwargs: Any) -> bool:
            return True

        async def failing_start(_sidecar: Any) -> Any:
            nonlocal attempts
            attempts += 1
            if attempts >= 7:
                # The retry count must already exceed the host's five-restart cap.
                raise companion.ShutdownRequested("test complete")
            raise companion.CompanionError("transient sidecar startup")

        monkeypatch.setenv("OUROBOROS_SKILL_STATE_DIR", str(state))
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setattr(companion, "require_isolated_process_group", lambda: None)
        monkeypatch.setattr(companion, "_pid_alive", lambda _pid: True)
        monkeypatch.setattr(
            companion,
            "start_parent_lifeline",
            lambda *_args, **_kwargs: (__import__("threading").Event(), Thread()),
        )
        monkeypatch.setattr(companion.httpx, "AsyncClient", lambda **_kwargs: Client())
        monkeypatch.setattr(companion, "TelegramSettingsObserver", lambda *_args: Bridge())
        monkeypatch.setattr(companion, "_wait_for_owner", owner)
        monkeypatch.setattr(companion, "_wait_for_cloudflared", binary)
        monkeypatch.setattr(companion, "_wait_for_core_and_owner", mirror)
        monkeypatch.setattr(companion, "TelegramProxySidecar", Sidecar)
        monkeypatch.setattr(companion, "start_sidecar", failing_start)
        monkeypatch.setattr(companion, "_retry_delay", lambda _count: 0.0)
        assert await companion.run() == 0
        assert attempts == 7

    asyncio.run(scenario())


def test_sidecar_runtime_crash_retries_beyond_host_restart_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        state = tmp_path / "state"
        state.mkdir()
        (state / companion._CONFIG_NAME).write_text(
            '{"schema":2,"core_port":8765,"owner_chat_id":0,"button_text":"Ouroboros","tunnel":"cloudflare_quick"}',
            encoding="utf-8",
        )
        attempts = 0
        closed = 0

        class Client:
            async def aclose(self) -> None:
                return None

        class Sidecar:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                pass

            def diagnostics(self) -> dict[str, int]:
                return {}

            def clear_public_url(self) -> None:
                return None

            async def aclose(self) -> None:
                nonlocal closed
                closed += 1

        class Server:
            should_exit = False

        class Thread:
            def join(self, timeout: float) -> None:
                assert timeout == 0.5

        class Bridge:
            state_dir = state
            @staticmethod
            def owned_owner_chat_id() -> int:
                return 0

            async def restore(self) -> bool:
                return True

        async def _true() -> bool:
            return True

        async def owner(*_args: Any, **_kwargs: Any) -> tuple[int, object]:
            return 12345, type("Menu", (), {"restore": lambda self: _true()})()

        async def binary(*_args: Any, **_kwargs: Any) -> Path:
            return state / "cloudflared"

        async def mirror(*_args: Any, **_kwargs: Any) -> bool:
            return True

        async def failed_server() -> None:
            raise RuntimeError("uvicorn runtime failure")

        async def start(_sidecar: Any) -> tuple[Server, asyncio.Task[None], int]:
            nonlocal attempts
            attempts += 1
            return Server(), asyncio.create_task(failed_server()), 43123

        async def generation(*_args: Any) -> Any:
            server_task = _args[-1].server_task
            assert isinstance(server_task, asyncio.Task)
            if attempts >= 7:
                # The internal runtime-restart count already exceeds the host's
                # five-restart cap; terminate the test through normal shutdown.
                raise companion.ShutdownRequested("test complete")
            try:
                await server_task
            except Exception as exc:
                raise companion.SidecarStoppedError("runtime crash") from exc
            raise AssertionError("failed server task unexpectedly succeeded")

        monkeypatch.setenv("OUROBOROS_SKILL_STATE_DIR", str(state))
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setattr(companion, "require_isolated_process_group", lambda: None)
        monkeypatch.setattr(companion, "_pid_alive", lambda _pid: True)
        monkeypatch.setattr(
            companion,
            "start_parent_lifeline",
            lambda *_args, **_kwargs: (__import__("threading").Event(), Thread()),
        )
        monkeypatch.setattr(companion.httpx, "AsyncClient", lambda **_kwargs: Client())
        monkeypatch.setattr(companion, "TelegramSettingsObserver", lambda *_args: Bridge())
        monkeypatch.setattr(companion, "_wait_for_owner", owner)
        monkeypatch.setattr(companion, "_wait_for_cloudflared", binary)
        monkeypatch.setattr(companion, "_wait_for_core_and_owner", mirror)
        monkeypatch.setattr(companion, "TelegramProxySidecar", Sidecar)
        monkeypatch.setattr(companion, "start_sidecar", start)
        monkeypatch.setattr(companion, "_run_tunnel_generations", generation)
        monkeypatch.setattr(companion, "_retry_delay", lambda _count: 0.0)
        assert await companion.run() == 0
        assert attempts == 7
        assert closed == 7

    asyncio.run(scenario())
