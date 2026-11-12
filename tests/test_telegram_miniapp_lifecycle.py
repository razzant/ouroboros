from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import types
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest


def _nonexistent_state_dir() -> Path:
    """A state_dir that provably does not exist, unique per call.

    A FIXED literal (/tmp/nonexistent-telegram-test) broke on a shared host:
    another user's run had squatted the exact name with mode 700, so stat()
    raised PermissionError where the scenario needs FileNotFoundError. A
    per-call unique name under the system tmp dir cannot be squatted and
    creates nothing.
    """
    return Path(tempfile.gettempdir()) / f"nonexistent-telegram-test-{uuid.uuid4().hex}"



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


def test_public_observer_outage_keeps_same_tunnel(monkeypatch: pytest.MonkeyPatch) -> None:
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
                "state_dir": _nonexistent_state_dir(),
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


def test_three_confirmed_bad_markers_rotate(monkeypatch: pytest.MonkeyPatch) -> None:
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
                "state_dir": _nonexistent_state_dir(),
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        class Bridge:
            state_dir = _nonexistent_state_dir()
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        class Bridge:
            state_dir = _nonexistent_state_dir()
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
                api_root: str | None = None,
                bot: str = "",
            ) -> None:
                assert client is not None and bot == "7"
                observed.append(owner)

        monkeypatch.setattr(companion, "snapshot_owner_chat_id", lambda _state, _bot: 111)
        monkeypatch.setattr(companion, "TelegramMenuManager", Menu)

        class Bridge:
            state_dir = _nonexistent_state_dir()
            @staticmethod
            def owned_owner_chat_id() -> int:
                return 111

        needs_reconcile, prior = await companion._prior_owner_reconciliation(
            "token",
            222,
            tmp_path,
            object(),  # type: ignore[arg-type]
            Bridge(),  # type: ignore[arg-type]
            "7",
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
            bot = "7"

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
            state_dir = _nonexistent_state_dir()
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
        monkeypatch.setattr(companion, "snapshot_owner_chat_id", lambda _state, _bot: 111)
        monkeypatch.setattr(
            companion,
            "TelegramMenuManager",
            lambda _token, old_owner, _state, *, client, api_root=None, bot="": Menu(old_owner),
        )
        monkeypatch.setattr(companion, "_reconcile_owner_change", stop_during_reconcile)
        assert await companion.run() == 0
        assert restored == [111]

    asyncio.run(scenario())


def test_cold_start_reconciles_only_this_bots_previous_owner(tmp_path: Path) -> None:
    """Real per-bot ledgers: another bot's record names no previous owner of this one."""
    for bot, owner in (("111", 41), ("222", 42)):
        (tmp_path / f"menu_button_snapshot.bot-{bot}.json").write_text(json.dumps({
            "schema": 3, "bot": bot, "chat_id": owner, "phase": "installed",
            "original": {"type": "default"}, "owned": {"type": "commands"}}), encoding="utf-8")

    def prior(token: str, owner: int, bot: str) -> tuple[bool, Any]:
        return asyncio.run(companion._prior_owner_reconciliation(token, owner, tmp_path, object(), object(), bot))

    needs, menu = prior("222:t", 43, "222")
    assert needs and menu is not None and menu.chat_id == 42 and menu.bot == "222"
    assert prior("333:t", 43, "333") == (False, None)
    assert prior("222:t", 42, "222") == (False, None)


# The menu writer of the last public release before ledgers named their bot (chat-only, schema 2), read from
# history by commit and checked by digest, so these tests run the very code that wrote such files.
_SCHEMA2_WRITER = ("8ee5aaadc1a56e724d62dceb163d3bf8216ec3a7", "skills/telegram/scripts/telegram_menu.py",
                   "5d00cabe2e0e7afd1ce19de2f7cc78ac1d1d79871e02001a5123e02ab41b0654")


def _older_release_left(tmp_path: Path, state: Path, bots: Any, *, interrupted: bool) -> bytes:
    """The older release's companion for bot 111 and owner 42 installs its Mini App button (commands before it)
    and is killed: its chat-only record stays, ``interrupted`` mid-install (the set reached Telegram, its reply
    was lost). Returns the record's bytes."""
    import hashlib
    import importlib.util

    from tests.test_telegram_miniapp_telegram_menu import OLD_URL

    commit, path, digest = _SCHEMA2_WRITER
    try:
        source = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "show", f"{commit}:{path}"],
                                capture_output=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip("the schema-2 menu writer's commit is not in this checkout's history")
    assert hashlib.sha256(source).hexdigest() == digest
    (tmp_path / "schema2_telegram_menu.py").write_bytes(source)
    spec = importlib.util.spec_from_file_location("schema2_telegram_menu", tmp_path / "schema2_telegram_menu.py")
    assert spec is not None and spec.loader is not None
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    bots.fail_after_set = interrupted

    async def install() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(bots.handler)) as http:
            writer = release.TelegramMenuManager("111:current", 42, state, api_root="https://api.test", client=http)
            if not interrupted:
                assert await writer.install(OLD_URL) is True
                return
            with pytest.raises(release.TelegramMenuTransportError):
                await writer.install(OLD_URL)

    asyncio.run(install())
    record = json.loads((state / "menu_button_snapshot.json").read_text(encoding="utf-8"))
    assert (record["schema"], record["chat_id"], record["phase"]) == (2, 42, "mutating" if interrupted else "installed")
    return (state / "menu_button_snapshot.json").read_bytes()


@pytest.mark.parametrize("interrupted", [False, True])
def test_a_cold_start_restores_an_older_releases_owner_before_exposing_the_new_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupted: bool,
) -> None:
    """The real companion after an upgrade, with the owner changed to 43 while the older release's chat-only
    record names 42: one bounded read of 42's menu (retried after a transport failure, exposing nothing) derives
    the bot's own ledger, 42 gets its commands button back, and only then is the new owner exposed. The retained
    chat-only file no longer matches, so the next pass reads 42 once and moves on; its bytes never change."""
    import threading

    from tests.test_telegram_miniapp_telegram_menu import Bots

    state = tmp_path / "state"
    state.mkdir()
    bots = Bots({"111:current": 111}, {(111, 42): {"type": "commands"}})
    legacy = _older_release_left(tmp_path, state, bots, interrupted=interrupted)
    reads: list[int] = []
    sets: list[int] = []
    unavailable = [1]  # the first read of 42's menu fails in transport

    def handler(request: httpx.Request) -> httpx.Response:
        method, chat = request.url.path.rsplit("/", 1)[-1], json.loads(request.content or b"{}").get("chat_id")
        if method == "getChatMenuButton":
            reads.append(chat)
            if chat == 42 and unavailable[0]:
                unavailable[0] -= 1
                return httpx.Response(502, json={"ok": False, "error_code": 502, "description": "Bad Gateway"})
        if method == "setChatMenuButton":
            sets.append(chat)
        return bots.handler(request)

    real_client = httpx.AsyncClient
    (state / companion._CONFIG_NAME).write_text(
        '{"schema":2,"core_port":8765,"owner_chat_id":0,"button_text":"Ouroboros","tunnel":"cloudflare_quick"}',
        encoding="utf-8")
    (state / "settings.json").write_text('{"TELEGRAM_CHAT_ID":"43"}', encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILL_STATE_DIR", str(state))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "111:current")
    monkeypatch.setattr(companion, "require_isolated_process_group", lambda: None)
    monkeypatch.setattr(companion, "_server_parent_pid", lambda: os.getpid())
    monkeypatch.setattr(companion, "start_parent_lifeline", lambda *_a, **_k: (threading.Event(), None))
    monkeypatch.setattr(companion, "_retry_delay", lambda _count: 0.01)
    monkeypatch.setattr(companion.httpx, "AsyncClient",
                        lambda **_kwargs: real_client(transport=httpx.MockTransport(handler)))
    seen: list[tuple[str, str]] = []
    transition = companion.RuntimeStatus.transition

    def record(self: Any, state_name: str, message: str, *, reason_code: str, **kwargs: Any) -> None:
        seen.append((state_name, reason_code))
        transition(self, state_name, message, reason_code=reason_code, **kwargs)

    exposed: list[dict[str, Any]] = []

    async def expose(*_args: Any) -> Any:  # the first step toward the new owner's tunnel and button
        exposed.append(dict(bots.buttons))
        raise companion.ShutdownRequested("the new owner is about to be exposed")

    monkeypatch.setattr(companion.RuntimeStatus, "transition", record)
    monkeypatch.setattr(companion, "_wait_for_cloudflared", expose)
    assert asyncio.run(asyncio.wait_for(companion.run(), timeout=20)) == 0

    assert len(exposed) == 1 and exposed[0][(111, 42)] == {"type": "commands"}
    assert ("reconciling", "owner_change_rollback_pending") in seen[:seen.index(("stopping", "shutdown"))]
    assert sets == [42]  # the one restore of 42; nothing is set for 43 before it is exposed
    # The failed read, the read that derived, the restore's read and confirmation, and the next pass's one read.
    assert reads.count(42) == 5
    assert (state / "menu_button_snapshot.json").read_bytes() == legacy
    assert not (state / "menu_button_snapshot.bot-111.json").exists()


@pytest.mark.parametrize("case", ["other_bot", "refused", "external_change", "current_owner"])
def test_a_chat_only_record_without_this_bots_button_restores_nothing(tmp_path: Path, case: str) -> None:
    """Another bot (its menu there shows none of the record's buttons, or Telegram refuses that chat), a menu
    changed outside the skill, or a record naming the current owner: ownership there stays unestablished, so no
    previous owner is reconciled, no ledger is derived, the record and every button stay as they were."""
    from tests.test_telegram_miniapp_telegram_menu import OLD_URL, Bots, web_app

    state = tmp_path / "state"
    state.mkdir()
    bots = Bots({"111:current": 111, "222:other": 222}, {(111, 42): {"type": "commands"}})
    legacy = _older_release_left(tmp_path, state, bots, interrupted=False)
    token, owner = ("222:other", 43) if case in {"other_bot", "refused"} else ("111:current", 43)
    if case == "current_owner":
        owner = 42
    if case == "external_change":
        bots.buttons[(111, 42)] = web_app("https://elsewhere.trycloudflare.com/")
    calls: list[tuple[str, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        method, chat = request.url.path.rsplit("/", 1)[-1], json.loads(request.content or b"{}").get("chat_id")
        calls.append((method, chat))
        if case == "refused" and chat == 42:
            return httpx.Response(400, json={"ok": False, "error_code": 400, "description": "Bad Request"})
        return bots.handler(request)

    before = dict(bots.buttons)

    async def scenario() -> tuple[bool, Any]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await companion._prior_owner_reconciliation(token, owner, state, client, object(),
                                                               token.split(":")[0])

    assert asyncio.run(scenario()) == (False, None)
    assert calls == ([] if case == "current_owner" else [("getChatMenuButton", 42)])
    assert bots.buttons == before and (case == "external_change" or before[(111, 42)] == web_app(OLD_URL))
    assert (state / "menu_button_snapshot.json").read_bytes() == legacy
    assert not list(state.glob("menu_button_snapshot.bot-*.json"))


def test_a_restore_interrupted_after_the_derivation_resumes_from_its_ledger(tmp_path: Path) -> None:
    """The derived ledger carries the chat-only source and digest through an interrupted restore; the next start
    finds it as this bot's own previous owner and completes, and the chat-only file stays as it was."""
    import hashlib

    from tests.test_telegram_miniapp_telegram_menu import Bots

    state = tmp_path / "state"
    state.mkdir()
    bots = Bots({"111:current": 111}, {(111, 42): {"type": "commands"}})
    legacy = _older_release_left(tmp_path, state, bots, interrupted=False)
    ledger = state / "menu_button_snapshot.bot-111.json"

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(bots.handler)) as http:
            async def prior_owner() -> tuple[bool, Any]:
                return await companion._prior_owner_reconciliation("111:current", 43, state, http, object(), "111")

            needs, prior = await prior_owner()
            assert needs and prior.chat_id == 42
            bots.fail_after_set = True  # the restore reaches Telegram; its reply is lost
            with pytest.raises(companion.TelegramMenuTransportError):
                await prior.restore()
            saved = json.loads(ledger.read_text(encoding="utf-8"))
            assert (saved["phase"], saved["chat_id"], saved["bot"]) == ("mutating", 42, "111")
            assert saved["adopted_from"] == {"ledger": "menu_button_snapshot.json", "schema": 2,
                                             "basis": "current_menu_match", "sha256": hashlib.sha256(legacy).hexdigest()}
            needs, prior = await prior_owner()
            assert needs and prior.chat_id == 42  # this bot's own ledger now names it
            await prior.restore()
            assert not ledger.exists() and bots.buttons[(111, 42)] == {"type": "commands"}
            assert await prior_owner() == (False, None)

    asyncio.run(scenario())
    assert (state / "menu_button_snapshot.json").read_bytes() == legacy


@pytest.mark.parametrize("old_original,new_original", [("default", "commands"), ("commands", "default")])
def test_a_replacement_bot_syncs_its_menu_past_a_record_its_predecessor_could_not_restore(
    tmp_path: Path, old_original: str, new_original: str,
) -> None:
    """The companion's real menu sync for bot B after bot A's revoked token left A's record: B is not blocked."""
    buttons = {"111:old": {"type": old_original}, "222:new": {"type": new_original}}
    revoked: set[str] = set()

    def respond(request: httpx.Request) -> httpx.Response:
        token, method = request.url.path.removeprefix("/bot").split("/", 1)
        if token in revoked:
            return httpx.Response(401, json={"ok": False, "error_code": 401, "description": "Unauthorized"})
        body = json.loads(request.content or b"{}")
        result: Any = {"id": int(token.split(":")[0]), "is_bot": True} if method == "getMe" else True
        if method == "getChatMenuButton":
            result = buttons[token]
        elif method == "setChatMenuButton":
            buttons[token] = body["menu_button"]
        return httpx.Response(200, json={"ok": True, "result": result})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            old = companion.TelegramMenuManager("111:old", 42, tmp_path, client=client, api_root="https://api.test")
            await old.install("https://old-menu.trycloudflare.com/")
            revoked.add("111:old")
            with pytest.raises(companion.TelegramMenuRejectedError):
                await old.restore()
            new = companion.TelegramMenuManager("222:new", 42, tmp_path, client=client, api_root="https://api.test")
            observer = types.SimpleNamespace(state_dir=tmp_path, owner_chat_id=lambda: 42,
                                             safe_for_exposure=lambda _owner: True)
            transitions: list[str] = []
            status = types.SimpleNamespace(transition=lambda state, *_args, **_kwargs: transitions.append(state))
            server_task = asyncio.create_task(asyncio.Event().wait())
            context = companion._GenerationContext(new, observer, 42, "Ouroboros", 8765, status, asyncio.Event(),
                                                   server_task)
            try:
                outcome = await companion._sync_menu_until_confirmed(
                    "https://new-menu.trycloudflare.com/", types.SimpleNamespace(returncode=None), context)
            finally:
                server_task.cancel()
                await asyncio.gather(server_task, return_exceptions=True)
            assert outcome is None and "blocked" not in transitions
            assert buttons["222:new"]["web_app"]["url"] == "https://new-menu.trycloudflare.com/"
            assert old.snapshot_path.exists()  # A's record waits for A

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


@pytest.mark.parametrize("enabled", ["on", "off"])
def test_a_missing_token_is_a_visible_wait_not_a_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enabled: str,
) -> None:
    """No token in the spawn environment: the companion stays alive with a fresh heartbeat and a truthful
    state (Off is told first), reads no runtime config, opens no Telegram client or tunnel and spends none
    of the host's restart budget. A host stop ends it cleanly; the host respawns it with a saved token."""
    import json
    import threading

    state = tmp_path / "state"
    state.mkdir()
    (state / "settings.json").write_text(json.dumps({"TELEGRAM_MINIAPP_ENABLED": enabled}), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILL_STATE_DIR", str(state))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setattr(companion, "require_isolated_process_group", lambda: None)
    monkeypatch.setattr(companion, "_server_parent_pid", lambda: os.getpid())
    monkeypatch.setattr(companion, "start_parent_lifeline", lambda *_a, **_k: (threading.Event(), None))

    def forbidden(*_args: Any, **_kwargs: Any):
        raise AssertionError("a companion without a token reached Telegram, the tunnel or its config")

    monkeypatch.setattr(companion, "load_runtime_config", forbidden)
    monkeypatch.setattr(companion.httpx, "AsyncClient", forbidden)

    async def scenario() -> dict:
        task = asyncio.create_task(companion.run())
        for _ in range(200):
            await asyncio.sleep(0.01)
            payload = json.loads((state / companion._STATUS_NAME).read_text(encoding="utf-8"))
            if payload.get("state") in {"waiting_token", "disabled"}:
                break
        await asyncio.sleep(0.2)
        assert not task.done(), "a missing token must not end the companion"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return payload

    payload = asyncio.run(scenario())
    if enabled == "on":
        assert (payload["state"], payload["reason_code"]) == ("waiting_token", "token_missing")
    else:
        assert (payload["state"], payload["reason_code"]) == ("disabled", "miniapp_disabled")
    assert payload["pid"] == os.getpid()


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
            state_dir = _nonexistent_state_dir()
            @staticmethod
            def owned_owner_chat_id() -> int:
                return 0

            async def restore(self) -> bool:
                return True

        async def owner(*_args: Any, **_kwargs: Any) -> tuple[int, object]:
            return 12345, type("Menu", (), {"bot": "7", "restore": lambda self: _true()})()

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
            state_dir = _nonexistent_state_dir()
            @staticmethod
            def owned_owner_chat_id() -> int:
                return 0

            async def restore(self) -> bool:
                return True

        async def _true() -> bool:
            return True

        async def owner(*_args: Any, **_kwargs: Any) -> tuple[int, object]:
            return 12345, type("Menu", (), {"bot": "7", "restore": lambda self: _true()})()

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
