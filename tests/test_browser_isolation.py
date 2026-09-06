"""Tests for browser state isolation and infrastructure error detection."""
import pathlib
import socket
import sys
import types

import pytest

import ouroboros.tools.browser as browser_mod
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools.browser import (
    _is_infrastructure_error,
    cleanup_browser,
    cleanup_browser_handles,
)


class TestInfrastructureErrorDetection:
    """_is_infrastructure_error should detect structural Playwright failures.

    Parametrized in v5.15.x — 7 single-case detection tests collapsed
    into one table (5 truthy infrastructure errors + 2 falsy
    application errors).
    """

    @pytest.mark.parametrize("exc,expected", [
        (RuntimeError("cannot switch to a different green thread"), True),
        (RuntimeError("different thread"), True),
        (Exception("browser has been closed"), True),
        (Exception("page has been closed"), True),
        (Exception("Connection closed"), True),
        (ValueError("invalid selector"), False),
        (TimeoutError("navigation timeout"), False),
    ])
    def test_classification(self, exc, expected):
        assert _is_infrastructure_error(exc) is expected


class TestBrowserModuleState:
    """Module-level state should be properly initialized."""

    # test_is_infrastructure_error_is_function removed in v5.15.x —
    # `assert callable(...)` on a function imported in this module's
    # imports is trivially true.

    def test_ensure_browser_tolerates_missing_thread_id(self, monkeypatch):
        routes = []
        contexts = []
        fake_page = types.SimpleNamespace(
            set_default_timeout=lambda timeout: None,
        )

        def _new_context(**kwargs):
            context = types.SimpleNamespace(
                kwargs=kwargs,
                route=lambda pattern, handler: routes.append((pattern, handler)),
                new_page=lambda: fake_page,
                close=lambda: None,
            )
            contexts.append(context)
            return context

        fake_browser = types.SimpleNamespace(
            new_context=_new_context,
            is_connected=lambda: True,
            close=lambda: None,
        )
        fake_playwright = types.SimpleNamespace(
            chromium=types.SimpleNamespace(launch=lambda **kwargs: fake_browser),
            webkit=types.SimpleNamespace(launch=lambda **kwargs: fake_browser),
            devices={
                "iPhone 13": {
                    "viewport": {"width": 390, "height": 844},
                    "user_agent": "Mobile Safari",
                    "device_scale_factor": 3,
                    "is_mobile": True,
                    "has_touch": True,
                }
            },
        )
        fake_sync_api = types.SimpleNamespace(
            sync_playwright=lambda: types.SimpleNamespace(start=lambda: fake_playwright)
        )
        monkeypatch.setattr(browser_mod, "_HAS_STEALTH", False)
        monkeypatch.setattr(browser_mod, "_ensure_playwright_installed", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            browser_mod.socket,
            "getaddrinfo",
            lambda host, *args, **kwargs: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
            ] if host == "example.com" else [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
            ],
        )
        monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_sync_api)

        ctx = types.SimpleNamespace(
            browser_state=types.SimpleNamespace(
                page=None,
                browser=None,
                pw_instance=None,
                last_screenshot_b64=None,
            )
        )

        page, _generation = browser_mod._ensure_browser(ctx)

        assert page is fake_page
        assert contexts[-1].kwargs["viewport"] == {"width": 1920, "height": 1080}
        assert "Chrome/131.0.0.0" in contexts[-1].kwargs["user_agent"]
        assert getattr(ctx.browser_state, "_thread_id", None) is not None
        assert getattr(ctx.browser_state, "_browser_engine", None) == "chromium"
        assert routes[:4] == [
            ("**/api/owner/context-mode", browser_mod._block_context_mode_owner_post),
            # v6.54.3: the owner-only LLM-safety coverage endpoint is route-blocked too
            # (broad glob + decoding handler so percent-encoding cannot slip it).
            ("**/api/owner/**", browser_mod._block_safety_mode_owner_post),
            # C1, v6.39: the owner-only skill attestation endpoint is route-blocked too
            # (broad glob so a percent-encoded path still reaches the decoding handler).
            ("**/api/owner/skills/**", browser_mod._block_owner_skill_attest_post),
            ("**/api/settings", browser_mod._block_owner_settings_post),
        ]
        # v6.26.0: the main agent gets a metadata-only SSRF route guard too.
        assert len(routes) == 5 and routes[4][0] == "**/*"

        browser_mod._ensure_browser(ctx, engine="webkit", device="iphone 13")[0]
        assert contexts[-1].kwargs["viewport"] == {"width": 390, "height": 844}
        assert contexts[-1].kwargs["is_mobile"] is True
        assert getattr(ctx.browser_state, "_browser_engine", None) == "webkit"
        assert getattr(ctx.browser_state, "_browser_device", None) == "iPhone 13"

        subagent_ctx = types.SimpleNamespace(
            task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
            browser_state=types.SimpleNamespace(
                page=None,
                browser=None,
                pw_instance=None,
                last_screenshot_b64=None,
            )
        )
        assert browser_mod._ensure_browser(subagent_ctx)[0] is fake_page
        assert routes and routes[-1][0] == "**/*"
        events = []
        route = types.SimpleNamespace(
            request=types.SimpleNamespace(url="http://127.0.0.1:8765/api/settings"),
            abort=lambda: events.append("abort"),
            continue_=lambda: events.append("continue"),
            # v6.54.3 (review round 8): guard handlers defer non-matching requests
            # DOWN the chain via fallback so earlier-registered blocks stay live.
            fallback=lambda: events.append("fallback"),
        )
        routes[-1][1](route)
        route.request.url = "http://192.168.1.1/admin"
        routes[-1][1](route)
        route.request.url = "http://169.254.1.1/"
        routes[-1][1](route)
        route.request.url = "http://[::]/"
        routes[-1][1](route)
        route.request.url = "https://example.com/"
        routes[-1][1](route)
        assert events == ["abort", "abort", "abort", "abort", "fallback"]

    def test_local_readonly_browser_url_guard_resolves_dns_fail_closed(self, monkeypatch):
        def fake_getaddrinfo(host, *args, **kwargs):
            if host == "public.example":
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
            if host == "internal.example":
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]
            raise socket.gaierror("no such host")

        monkeypatch.setattr(browser_mod.socket, "getaddrinfo", fake_getaddrinfo)

        assert browser_mod._is_subagent_blocked_browser_url("ftp://public.example/file") is True
        assert browser_mod._is_subagent_blocked_browser_url("http://127.0.0.1:8765") is True
        assert browser_mod._is_subagent_blocked_browser_url("http://0177.0.0.1/") is True
        assert browser_mod._is_subagent_blocked_browser_url("http://0x7f.0.0.1/") is True
        assert browser_mod._is_subagent_blocked_browser_url("http://2130706433/") is True
        assert browser_mod._is_subagent_blocked_browser_url("http://012.0.0.1/") is True
        assert browser_mod._is_subagent_blocked_browser_url("http://internal.example") is True
        assert browser_mod._is_subagent_blocked_browser_url("http://missing.example") is True
        assert browser_mod._is_subagent_blocked_browser_url("https://public.example/path") is False

    def test_subagent_screenshot_text_does_not_reference_blocked_send_photo(self):
        fake_page = types.SimpleNamespace(screenshot=lambda **_kwargs: b"png")
        ctx = types.SimpleNamespace(
            task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
            browser_state=types.SimpleNamespace(last_screenshot_b64=None),
        )

        result = browser_mod._extract_page_output(fake_page, "screenshot", ctx)

        assert "send_photo" not in result
        assert "analyze_screenshot" in result
        assert ctx.browser_state.last_screenshot_b64

    def test_aliases_arm64_browser_cache_for_missing_x64_binary(self, monkeypatch, tmp_path):
        monkeypatch.setattr(browser_mod.sys, "platform", "darwin", raising=False)
        root = tmp_path / "playwright" / "chromium_headless_shell-1208"
        arm_dir = root / "chrome-headless-shell-mac-arm64"
        arm_dir.mkdir(parents=True)
        arm_binary = arm_dir / "chrome-headless-shell"
        arm_binary.write_text("stub", encoding="utf-8")

        missing_binary = root / "chrome-headless-shell-mac-x64" / "chrome-headless-shell"
        err = RuntimeError(f"BrowserType.launch: Executable doesn't exist at {missing_binary}")

        assert browser_mod._maybe_alias_playwright_binary(err) is True
        alias_dir = missing_binary.parent
        assert alias_dir.is_symlink()
        assert pathlib.Path(alias_dir.resolve()) == arm_dir.resolve()

        arm_dir_2 = root / "chrome-headless-shell-mac-arm64-copy"
        arm_dir_2.mkdir(parents=True)
        (arm_dir_2 / "chrome-headless-shell").write_text("stub", encoding="utf-8")
        missing_binary_2 = root / "chrome-headless-shell-mac-x64-copy" / "chrome-headless-shell"
        err2 = RuntimeError(f"BrowserType.launch: Executable doesn't exist at {missing_binary_2}")
        fake_pw = types.SimpleNamespace(chromium=types.SimpleNamespace(launch=lambda **_kwargs: (_ for _ in ()).throw(err2)))
        with pytest.raises(RuntimeError):
            browser_mod._launch_browser_with_fallback(fake_pw, allow_cache_write=False)
        assert not missing_binary_2.parent.exists()

    def test_launches_selected_webkit_engine_without_chromium_args(self):
        launch_calls = []
        fake_pw = types.SimpleNamespace(
            chromium=types.SimpleNamespace(launch=lambda **kwargs: launch_calls.append(("chromium", kwargs)) or "chromium-browser"),
            webkit=types.SimpleNamespace(launch=lambda **kwargs: launch_calls.append(("webkit", kwargs)) or "webkit-browser"),
        )

        assert browser_mod._launch_browser_with_fallback(fake_pw, engine="webkit") == "webkit-browser"

        assert launch_calls == [("webkit", {"headless": True})]


class TestHasPlatformChromium:
    """_has_platform_chromium: two-level check — chromium-* dir + platform-matching subdir.

    Parametrized in v5.15.x — 7 tests collapsed into 2 (one for the
    "not found" matrix, one for the "found via real executable" matrix).
    Each subcase builds its filesystem skeleton inside tmp_path via a
    small builder kwarg.
    """

    def _build_fixture(self, tmp_path, kind: str):
        """kind values:
        - missing       : nothing exists
        - empty         : tmp_path has no chromium-* subdir
        - non_chromium  : a firefox-* dir but no chromium-*
        - wrong_platform: chromium-X/chrome-linux-x64 (wrong platform on darwin)
        - metadata_only : chromium-X/chrome-mac-x64/metadata.json (no exe)
        - real_app      : chromium-X/chrome-mac-x64/Chromium.app/.../Chromium
        - headless_shell: chromium_headless_shell-X/chrome-headless-shell-mac-arm64/chrome-headless-shell
        """
        if kind == "missing":
            return tmp_path / "nonexistent"
        if kind == "empty":
            return tmp_path
        if kind == "non_chromium":
            (tmp_path / "firefox-1234").mkdir()
            return tmp_path
        if kind == "wrong_platform":
            cdir = tmp_path / "chromium-1234"
            cdir.mkdir()
            (cdir / "chrome-linux-x64").mkdir()
            return tmp_path
        if kind == "metadata_only":
            cdir = tmp_path / "chromium-1234"
            cdir.mkdir()
            pdir = cdir / "chrome-mac-x64"
            pdir.mkdir()
            (pdir / "metadata.json").write_text("{}", encoding="utf-8")
            return tmp_path
        if kind == "real_app":
            cdir = tmp_path / "chromium-1234"
            cdir.mkdir()
            exe = cdir / "chrome-mac-x64" / "Chromium.app" / "Contents" / "MacOS" / "Chromium"
            exe.parent.mkdir(parents=True)
            exe.write_text("stub", encoding="utf-8")
            return tmp_path
        if kind == "headless_shell":
            cdir = tmp_path / "chromium_headless_shell-1234"
            cdir.mkdir()
            exe = cdir / "chrome-headless-shell-mac-arm64" / "chrome-headless-shell"
            exe.parent.mkdir(parents=True)
            exe.write_text("stub", encoding="utf-8")
            return tmp_path
        raise ValueError(f"unknown kind: {kind}")

    @pytest.mark.parametrize("kind,expected", [
        ("missing",        False),
        ("empty",          False),
        ("non_chromium",   False),
        ("wrong_platform", False),
        ("metadata_only",  False),
        ("real_app",       True),
        ("headless_shell", True),
    ])
    def test_classification(self, kind, expected, tmp_path, monkeypatch):
        from ouroboros.tools import browser as bmod
        monkeypatch.setattr(bmod.sys, "platform", "darwin", raising=False)
        from ouroboros.tools.browser import _has_platform_chromium

        root = self._build_fixture(tmp_path, kind)
        assert _has_platform_chromium(root) is expected


class TestHasPlatformWebKit:
    @pytest.mark.parametrize("kind,expected", [
        ("missing", False),
        ("wrong_engine", False),
        ("metadata_only", False),
        ("pw_run", True),
        ("minibrowser", True),
    ])
    def test_classification(self, kind, expected, tmp_path):
        if kind == "missing":
            root = tmp_path / "missing"
        else:
            root = tmp_path
            if kind == "wrong_engine":
                (root / "chromium-1234").mkdir()
            elif kind == "metadata_only":
                meta = root / "webkit-1234" / "metadata.json"
                meta.parent.mkdir(parents=True)
                meta.write_text("{}", encoding="utf-8")
            elif kind == "pw_run":
                exe = root / "webkit-1234" / "pw_run.sh"
                exe.parent.mkdir(parents=True)
                exe.write_text("stub", encoding="utf-8")
            elif kind == "minibrowser":
                exe = root / "webkit-1234" / "MiniBrowser.app" / "Contents" / "MacOS" / "MiniBrowser"
                exe.parent.mkdir(parents=True)
                exe.write_text("stub", encoding="utf-8")
        assert browser_mod._has_platform_webkit(root) is expected


class TestSetPlaywrightBrowsersPathIfBundled:
    """_set_playwright_browsers_path_if_bundled: sets env var when a bundled engine is found."""

    def test_no_op_when_env_already_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "/some/custom/path")
        import ouroboros.tools.browser as bmod
        monkeypatch.setattr(bmod.sys, "platform", "darwin", raising=False)
        # Should not overwrite existing env var
        bmod._set_playwright_browsers_path_if_bundled()
        import os
        assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == "/some/custom/path"

    def test_sets_zero_when_chromium_dir_matches(self, monkeypatch, tmp_path):
        import os
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        import ouroboros.tools.browser as bmod
        monkeypatch.setattr(bmod.sys, "platform", "darwin", raising=False)
        # Build fake playwright package structure
        local_browsers = tmp_path / "driver" / "package" / ".local-browsers"
        chromium_dir = local_browsers / "chromium-9999"
        chromium_dir.mkdir(parents=True)
        platform_dir = chromium_dir / "chrome-mac-x64"
        exe = platform_dir / "Chromium.app" / "Contents" / "MacOS" / "Chromium"
        exe.parent.mkdir(parents=True)
        exe.write_text("stub", encoding="utf-8")  # real macOS executable path
        fake_pw = types.SimpleNamespace(__file__=str(tmp_path / "__init__.py"))
        monkeypatch.setitem(sys.modules, "playwright", fake_pw)
        bmod._set_playwright_browsers_path_if_bundled()
        assert os.environ.get("PLAYWRIGHT_BROWSERS_PATH") == "0"

    def test_sets_zero_when_headless_shell_dir_matches(self, monkeypatch, tmp_path):
        import os
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        import ouroboros.tools.browser as bmod
        monkeypatch.setattr(bmod.sys, "platform", "darwin", raising=False)
        local_browsers = tmp_path / "driver" / "package" / ".local-browsers"
        chromium_dir = local_browsers / "chromium_headless_shell-9999"
        chromium_dir.mkdir(parents=True)
        platform_dir = chromium_dir / "chrome-headless-shell-mac-arm64"
        exe = platform_dir / "chrome-headless-shell"
        exe.parent.mkdir(parents=True)
        exe.write_text("stub", encoding="utf-8")
        fake_pw = types.SimpleNamespace(__file__=str(tmp_path / "__init__.py"))
        monkeypatch.setitem(sys.modules, "playwright", fake_pw)
        bmod._set_playwright_browsers_path_if_bundled()
        assert os.environ.get("PLAYWRIGHT_BROWSERS_PATH") == "0"

    def test_no_change_when_no_matching_chromium(self, monkeypatch, tmp_path):
        import os
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        import ouroboros.tools.browser as bmod
        monkeypatch.setattr(bmod.sys, "platform", "darwin", raising=False)
        local_browsers = tmp_path / "driver" / "package" / ".local-browsers"
        local_browsers.mkdir(parents=True)
        fake_pw = types.SimpleNamespace(__file__=str(tmp_path / "__init__.py"))
        monkeypatch.setitem(sys.modules, "playwright", fake_pw)
        bmod._set_playwright_browsers_path_if_bundled()
        assert "PLAYWRIGHT_BROWSERS_PATH" not in os.environ

    def test_sets_zero_when_webkit_dir_matches(self, monkeypatch, tmp_path):
        import os
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        import ouroboros.tools.browser as bmod
        local_browsers = tmp_path / "driver" / "package" / ".local-browsers"
        exe = local_browsers / "webkit-9999" / "pw_run.sh"
        exe.parent.mkdir(parents=True)
        exe.write_text("stub", encoding="utf-8")
        fake_pw = types.SimpleNamespace(__file__=str(tmp_path / "__init__.py"))
        monkeypatch.setitem(sys.modules, "playwright", fake_pw)
        bmod._set_playwright_browsers_path_if_bundled()
        assert os.environ.get("PLAYWRIGHT_BROWSERS_PATH") == "0"

    def test_import_time_side_effect_sets_env_when_bundled(self, monkeypatch, tmp_path):
        """Module-import calls _set_playwright_browsers_path_if_bundled(); reloading the
        module with a fake bundled Chromium present must set PLAYWRIGHT_BROWSERS_PATH=0."""
        import os
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        # Build fake playwright package with a non-empty platform dir
        local_browsers = tmp_path / "driver" / "package" / ".local-browsers"
        exe = local_browsers / "chromium-9999" / "chrome-mac-x64" / "Chromium.app" / "Contents" / "MacOS" / "Chromium"
        exe.parent.mkdir(parents=True)
        exe.write_text("stub", encoding="utf-8")  # real macOS executable path
        fake_pw = types.SimpleNamespace(__file__=str(tmp_path / "__init__.py"))
        monkeypatch.setitem(sys.modules, "playwright", fake_pw)
        import ouroboros.tools.browser as bmod
        monkeypatch.setattr(bmod.sys, "platform", "darwin", raising=False)
        # Simulate a fresh module import by calling the module-level init directly
        # (importlib.reload would re-run the side-effect but also re-register tools;
        # calling the function directly tests the same code path without side effects)
        bmod._set_playwright_browsers_path_if_bundled()
        assert os.environ.get("PLAYWRIGHT_BROWSERS_PATH") == "0"

    def test_browser_install_uses_data_cache_when_zero_has_no_browser(self, monkeypatch, tmp_path):
        import os

        import ouroboros.tools.browser as bmod

        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "0")
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr(bmod, "_playwright_ready", False)
        monkeypatch.setattr(bmod, "_playwright_ready_engines", set())
        monkeypatch.setattr(bmod, "_playwright_browsers_path_managed", False)
        calls = []
        monkeypatch.setattr(bmod.subprocess, "check_call", lambda cmd: calls.append(cmd))
        fake_pw = types.SimpleNamespace(__file__=str(tmp_path / "playwright" / "__init__.py"))
        monkeypatch.setitem(sys.modules, "playwright", fake_pw)
        fake_sync = types.ModuleType("playwright.sync_api")

        class FakeSyncPlaywright:
            def __enter__(self):
                chromium = types.SimpleNamespace(executable_path=str(tmp_path / "missing-chromium"))
                webkit = types.SimpleNamespace(executable_path=str(tmp_path / "missing-webkit"))
                return types.SimpleNamespace(chromium=chromium, webkit=webkit)

            def __exit__(self, *_args):
                return False

        fake_sync.sync_playwright = lambda: FakeSyncPlaywright()
        monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_sync)

        bmod._ensure_playwright_installed()

        assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == str(tmp_path / "data" / "playwright-browsers")
        assert calls[-1][-3:] == ["playwright", "install", "chromium"]

    def test_webkit_install_uses_data_cache_when_zero_has_no_bundled_webkit(self, monkeypatch, tmp_path):
        import os

        import ouroboros.tools.browser as bmod

        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "0")
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr(bmod, "_playwright_ready", False)
        monkeypatch.setattr(bmod, "_playwright_ready_engines", set())
        monkeypatch.setattr(bmod, "_playwright_browsers_path_managed", False)
        calls = []
        monkeypatch.setattr(bmod.subprocess, "check_call", lambda cmd: calls.append(cmd))
        fake_pw = types.SimpleNamespace(__file__=str(tmp_path / "playwright" / "__init__.py"))
        monkeypatch.setitem(sys.modules, "playwright", fake_pw)
        fake_sync = types.ModuleType("playwright.sync_api")

        class FakeSyncPlaywright:
            def __enter__(self):
                chromium = types.SimpleNamespace(executable_path=str(tmp_path / "bundled-chromium"))
                webkit = types.SimpleNamespace(executable_path=str(tmp_path / "missing-webkit"))
                return types.SimpleNamespace(chromium=chromium, webkit=webkit)

            def __exit__(self, *_args):
                return False

        fake_sync.sync_playwright = lambda: FakeSyncPlaywright()
        monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_sync)

        bmod._ensure_playwright_installed(engine="webkit")

        assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == str(tmp_path / "data" / "playwright-browsers")
        assert calls[-1][-3:] == ["playwright", "install", "webkit"]

    def test_webkit_cache_fallback_does_not_strand_bundled_chromium(self, monkeypatch, tmp_path):
        import os

        import ouroboros.tools.browser as bmod

        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "0")
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr(bmod, "_playwright_ready", False)
        monkeypatch.setattr(bmod, "_playwright_ready_engines", set())
        monkeypatch.setattr(bmod, "_playwright_browsers_path_managed", False)
        monkeypatch.setattr(bmod.sys, "platform", "darwin", raising=False)
        local_browsers = tmp_path / "playwright" / "driver" / "package" / ".local-browsers"
        chrome = (
            local_browsers
            / "chromium_headless_shell-9999"
            / "chrome-headless-shell-mac-arm64"
            / "chrome-headless-shell"
        )
        chrome.parent.mkdir(parents=True)
        chrome.write_text("stub", encoding="utf-8")
        calls = []
        monkeypatch.setattr(bmod.subprocess, "check_call", lambda cmd: calls.append(cmd))
        fake_pw = types.SimpleNamespace(__file__=str(tmp_path / "playwright" / "__init__.py"))
        monkeypatch.setitem(sys.modules, "playwright", fake_pw)
        fake_sync = types.ModuleType("playwright.sync_api")

        class FakeSyncPlaywright:
            def __enter__(self):
                chromium = types.SimpleNamespace(executable_path=str(chrome))
                webkit = types.SimpleNamespace(executable_path=str(tmp_path / "data" / "playwright-browsers" / "missing-webkit"))
                return types.SimpleNamespace(chromium=chromium, webkit=webkit)

            def __exit__(self, *_args):
                return False

        fake_sync.sync_playwright = lambda: FakeSyncPlaywright()
        monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_sync)

        bmod._ensure_playwright_installed(engine="webkit")
        assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == str(tmp_path / "data" / "playwright-browsers")

        bmod._ensure_playwright_installed(engine="chromium")

        assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == "0"
        assert any(cmd[-3:] == ["playwright", "install", "webkit"] for cmd in calls)
        assert not any(cmd[-3:] == ["playwright", "install", "chromium"] for cmd in calls)

    def test_frozen_missing_bundled_engine_installs_with_embedded_python(self, monkeypatch, tmp_path):
        import os

        import ouroboros.tools.browser as bmod

        embedded_python_posix = tmp_path / "python-standalone" / "bin" / "python3"
        embedded_python_win = tmp_path / "python-standalone" / "python.exe"
        embedded_python_posix.parent.mkdir(parents=True)
        embedded_python_win.parent.mkdir(parents=True, exist_ok=True)
        embedded_python_posix.write_text("#!/bin/sh\n", encoding="utf-8")
        embedded_python_win.write_text("@echo off\r\n", encoding="utf-8")
        monkeypatch.setattr(bmod.sys, "frozen", True, raising=False)
        monkeypatch.setattr(bmod.sys, "_MEIPASS", str(tmp_path), raising=False)
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "0")
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr(bmod, "_playwright_ready", False)
        monkeypatch.setattr(bmod, "_playwright_ready_engines", set())
        monkeypatch.setattr(bmod, "_playwright_browsers_path_managed", False)
        calls = []
        monkeypatch.setattr(bmod.subprocess, "check_call", lambda cmd: calls.append(cmd))
        fake_pw = types.SimpleNamespace(__file__=str(tmp_path / "playwright" / "__init__.py"))
        monkeypatch.setitem(sys.modules, "playwright", fake_pw)
        fake_sync = types.ModuleType("playwright.sync_api")

        class FakeSyncPlaywright:
            def __enter__(self):
                return types.SimpleNamespace(
                    chromium=types.SimpleNamespace(executable_path=str(tmp_path / "missing-chromium")),
                    webkit=types.SimpleNamespace(executable_path=str(tmp_path / "missing-webkit")),
                )

            def __exit__(self, *_args):
                return False

        fake_sync.sync_playwright = lambda: FakeSyncPlaywright()
        monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_sync)

        bmod._ensure_playwright_installed(engine="webkit")

        assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == str(tmp_path / "data" / "playwright-browsers")
        assert calls[-1][0] in {str(embedded_python_posix), str(embedded_python_win)}
        assert calls[-1][-3:] == ["playwright", "install", "webkit"]

    def test_readonly_missing_bundled_engine_does_not_create_cache(self, monkeypatch, tmp_path):
        import os

        import ouroboros.tools.browser as bmod

        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "0")
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr(bmod, "_playwright_ready", False)
        monkeypatch.setattr(bmod, "_playwright_ready_engines", set())
        monkeypatch.setattr(bmod, "_playwright_browsers_path_managed", False)
        calls = []
        monkeypatch.setattr(bmod.subprocess, "check_call", lambda cmd: calls.append(cmd))
        fake_pw = types.SimpleNamespace(__file__=str(tmp_path / "playwright" / "__init__.py"))
        monkeypatch.setitem(sys.modules, "playwright", fake_pw)
        fake_sync = types.ModuleType("playwright.sync_api")

        class FakeSyncPlaywright:
            def __enter__(self):
                return types.SimpleNamespace(
                    chromium=types.SimpleNamespace(executable_path=str(tmp_path / "missing-chromium")),
                    webkit=types.SimpleNamespace(executable_path=str(tmp_path / "missing-webkit")),
                )

            def __exit__(self, *_args):
                return False

        fake_sync.sync_playwright = lambda: FakeSyncPlaywright()
        monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_sync)

        with pytest.raises(RuntimeError, match="local_readonly_subagent"):
            bmod._ensure_playwright_installed(engine="webkit", allow_install=False)

        assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == "0"
        assert not (tmp_path / "data" / "playwright-browsers").exists()
        assert calls == []

    def test_readonly_existing_webkit_cache_is_reused_without_install(self, monkeypatch, tmp_path):
        import os

        import ouroboros.tools.browser as bmod

        cache = tmp_path / "data" / "playwright-browsers"
        webkit = cache / "webkit-9999" / "pw_run.sh"
        webkit.parent.mkdir(parents=True)
        webkit.write_text("stub", encoding="utf-8")
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "0")
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr(bmod, "_playwright_ready", False)
        monkeypatch.setattr(bmod, "_playwright_ready_engines", set())
        monkeypatch.setattr(bmod, "_playwright_browsers_path_managed", False)
        calls = []
        monkeypatch.setattr(bmod.subprocess, "check_call", lambda cmd: calls.append(cmd))
        fake_pw = types.SimpleNamespace(__file__=str(tmp_path / "playwright" / "__init__.py"))
        monkeypatch.setitem(sys.modules, "playwright", fake_pw)
        fake_sync = types.ModuleType("playwright.sync_api")

        class FakeSyncPlaywright:
            def __enter__(self):
                return types.SimpleNamespace(
                    chromium=types.SimpleNamespace(executable_path=str(tmp_path / "missing-chromium")),
                    webkit=types.SimpleNamespace(executable_path=str(webkit)),
                )

            def __exit__(self, *_args):
                return False

        fake_sync.sync_playwright = lambda: FakeSyncPlaywright()
        monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_sync)

        bmod._ensure_playwright_installed(engine="webkit", allow_install=False)

        assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == str(cache)
        assert calls == []


class TestCleanupBrowser:
    """cleanup_browser should null out all browser_state references."""

    def test_cleanup_replaces_the_generation_and_nulls_the_retired_one(self):
        from ouroboros.tools.registry import BrowserState

        retired = BrowserState()
        ctx = types.SimpleNamespace(browser_state=retired)
        cleanup_browser(ctx)
        # The shared slot holds a FRESH generation; the retired one is inert.
        assert ctx.browser_state is not retired
        assert ctx.browser_state.page is None
        assert retired.page is None and retired.browser is None
        assert retired.pw_instance is None

    def test_cleanup_detached_handles_closes_in_order_on_the_calling_thread(self):
        # Thread-affinity is the CALLER's contract (the settlement callback
        # runs on the worker thread that owns the handles); this test pins
        # the close order and records the executing thread honestly.
        import threading

        calls = []
        threads = set()

        def _mark(name):
            def _inner():
                threads.add(threading.get_ident())
                calls.append(name)
            return _inner

        from ouroboros.tools.registry import BrowserState

        retired = BrowserState()
        retired.page = types.SimpleNamespace(close=_mark("page"))
        retired._browser_context = types.SimpleNamespace(close=_mark("context"))
        retired.browser = types.SimpleNamespace(close=_mark("browser"))
        retired.pw_instance = types.SimpleNamespace(stop=_mark("playwright"))
        worker = threading.Thread(target=cleanup_browser_handles, args=(retired,))
        worker.start()
        worker.join()
        assert calls == ["page", "context", "browser", "playwright"]
        # Every close ran on the SAME (calling) thread — never hopped back
        # to the main thread.
        assert threads == {worker.ident}
        # Idempotence: the retired generation was nulled while closing, so a
        # second sweep closes nothing twice.
        cleanup_browser_handles(retired)
        assert calls == ["page", "context", "browser", "playwright"]


class TestRetiredGenerationBacklog:
    def test_backlog_cap_refuses_a_new_session(self, monkeypatch):
        """In-process bound (post-merge audit): with the cap's worth of
        abandoned-but-unclosed generations, opening ANOTHER session is a typed
        refusal instead of unbounded Chromium accumulation; closing one frees
        capacity."""
        from ouroboros.tools import browser as browser_mod
        from ouroboros.tools.registry import BrowserState

        monkeypatch.setattr(browser_mod, "_ensure_playwright_installed",
                            lambda *a, **k: None)
        hung = []
        for _ in range(browser_mod._RETIRED_GENERATIONS_MAX):
            g = BrowserState()
            g.pw_instance = types.SimpleNamespace(stop=lambda: None)
            hung.append(g)
        ctx = types.SimpleNamespace(browser_state=BrowserState(),
                                    _retired_browser_generations=list(hung))
        with pytest.raises(RuntimeError) as err:
            browser_mod._ensure_browser(ctx)
        assert "BROWSER_BACKLOG_RETIRED_SESSIONS" in str(err.value)
        # One hung generation SETTLES (cleanup stamps _cleanup_done after its
        # closes ran): the prune frees capacity — asserted directly on the
        # prune seam so this default-lane test never launches real Playwright.
        browser_mod.cleanup_browser_handles(hung[0])
        assert len(browser_mod._live_retired_generations(ctx)) \
            == browser_mod._RETIRED_GENERATIONS_MAX - 1
        # A close that HANGS mid-way keeps its generation counted: nulled
        # fields alone are not settlement (sol MAJOR-3).
        half = types.SimpleNamespace(pw_instance=None, _cleanup_done=False)
        ctx._retired_browser_generations.append(half)
        assert half in browser_mod._live_retired_generations(ctx)

    def test_detach_registers_the_retired_generation(self):
        from ouroboros.tools import browser as browser_mod
        from ouroboros.tools.registry import BrowserState

        first = BrowserState()
        ctx = types.SimpleNamespace(browser_state=first)
        retired, fresh = browser_mod._detach_browser(ctx)
        assert retired is first
        assert ctx.browser_state is fresh and fresh is not first
        assert ctx._retired_browser_generations == [first]


class TestReplacementGenerationGuard:
    def test_late_exception_closes_only_the_call_generation(self, monkeypatch):
        """sol MAJOR pin: a call whose generation was retired mid-flight closes
        ITS OWN state on its own thread and never touches the replacement."""
        from ouroboros.tools import browser as browser_mod
        from ouroboros.tools.registry import BrowserState

        closed = []
        mine = BrowserState()
        mine.pw_instance = types.SimpleNamespace(stop=lambda: closed.append("mine"))
        replacement = BrowserState()
        replacement.pw_instance = types.SimpleNamespace(
            stop=lambda: closed.append("replacement"))
        ctx = types.SimpleNamespace(browser_state=mine, task_constraint=None)

        def _ensure_and_retire(c, engine="chromium", device=""):
            # Simulate the timeout detach landing mid-call: the shared slot
            # now holds the NEXT command's generation.
            c.browser_state = replacement
            raise RuntimeError("boom after retirement")

        monkeypatch.setattr(browser_mod, "_ensure_browser", _ensure_and_retire)
        monkeypatch.setattr(browser_mod, "_readonly_subagent", lambda c: False)
        out = browser_mod._browse_page(ctx, "https://example.com")
        assert "BROWSER_SESSION_RETIRED" in out
        assert closed == ["mine"]
        assert replacement.pw_instance is not None


class TestMidEnsureRetirement:
    def test_call_works_with_the_generation_that_owns_its_page(self, monkeypatch):
        """sol delta finding: a timeout retirement DURING _ensure_browser must
        not rebind the call onto the replacement — the call's generation is the
        tuple _ensure_browser returns, and the screenshot lands there."""
        from ouroboros.tools import browser as browser_mod
        from ouroboros.tools.registry import BrowserState

        own = BrowserState()
        replacement = BrowserState()
        page = types.SimpleNamespace(set_default_timeout=lambda ms: None,
                                     goto=lambda *a, **k: None)
        ctx = types.SimpleNamespace(browser_state=own, task_constraint=None)

        def _ensure_and_retire(c, engine="chromium", device=""):
            c.browser_state = replacement  # the mid-ensure timeout detach
            return page, own               # the call's OWN generation

        received = []
        monkeypatch.setattr(browser_mod, "_ensure_browser", _ensure_and_retire)
        monkeypatch.setattr(browser_mod, "_readonly_subagent", lambda c: False)
        monkeypatch.setattr(
            browser_mod, "_extract_page_output",
            lambda p, output, c, bs=None: received.append(bs) or "ok")
        out = browser_mod._browse_page(ctx, "https://example.com")
        assert out == "ok"
        assert received == [own]  # never the replacement


class TestGenerationExpectation:
    def test_ensure_refuses_when_the_pinned_generation_was_replaced(self):
        """The check-then-run window (sol delta 3): _ensure_browser re-checks
        the wrapper's pinned expectation at the exact capture point, so a call
        whose timeout retired it during the pre-tool safety check refuses
        instead of building a session in the replacement."""
        from ouroboros.tools import browser as browser_mod
        from ouroboros.tools.registry import BrowserState

        pinned, replacement = BrowserState(), BrowserState()
        ctx = types.SimpleNamespace(browser_state=replacement,
                                    _active_browser_generation=pinned)
        with pytest.raises(RuntimeError) as err:
            browser_mod._ensure_browser(ctx)
        assert "BROWSER_SESSION_RETIRED" in str(err.value)
        assert replacement.pw_instance is None  # untouched

    def test_legitimate_ensure_replacement_failure_is_not_a_false_retired(self, monkeypatch):
        """sol delta 3: _ensure_browser legitimately replaced the generation
        (engine change) and THEN failed — the guard sees the slot matching the
        pinned expectation and treats it as an ordinary error, never RETIRED."""
        from ouroboros.tools import browser as browser_mod
        from ouroboros.tools.registry import BrowserState

        own = BrowserState()
        ctx = types.SimpleNamespace(browser_state=own, task_constraint=None)

        def _replace_legitimately_and_fail(c, engine="chromium", device=""):
            _retired, fresh = browser_mod._detach_browser(c)
            setattr(c, "_active_browser_generation", fresh)
            raise RuntimeError("recreate failed")

        monkeypatch.setattr(browser_mod, "_ensure_browser", _replace_legitimately_and_fail)
        monkeypatch.setattr(browser_mod, "_readonly_subagent", lambda c: False)
        monkeypatch.setattr(browser_mod, "_is_infrastructure_error", lambda c: False)
        with pytest.raises(RuntimeError) as err:
            browser_mod._browse_page(ctx, "https://example.com")
        assert "recreate failed" in str(err.value)
