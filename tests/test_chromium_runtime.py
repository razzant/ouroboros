"""The browser tools' pinned Chromium: exact supply, sandbox by default, typed refusals.

``ouroboros/chromium_runtime.py`` owns the pin and its installation; ``tools/browser.py``
launches nothing else. Real launches of the pinned build are in the ``browser`` lane
(``tests/test_browser_tools_smoke.py``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import shutil
import types
import zipfile

import pytest

import ouroboros.tools.browser as browser_mod
from ouroboros import chromium_runtime, verified_download
from ouroboros.tools.registry import BrowserState

PINNED = chromium_runtime.CHROMIUM_VERSION
FOLDER = "chrome-headless-shell-test"
GOOD_MEMBERS = [
    (f"{FOLDER}/", b"", 0o40755),
    (f"{FOLDER}/chrome-headless-shell", b"#!/bin/sh\nexit 0\n", 0o100755),
    (f"{FOLDER}/resources/data.pak", b"pak", 0o100644),
]
# Playwright 1.62 registry/index.ts ``allDownloadableDirectoriesThatEverExisted``: its
# browser garbage collector deletes unreferenced directories named "<name with _>-...".
PLAYWRIGHT_GC_PREFIXES = ("android-", "chromium-", "firefox-", "webkit-", "ffmpeg-", "firefox_beta-",
                          "chromium_tip_of_tree-", "chromium_headless_shell-",
                          "chromium_tip_of_tree_headless_shell-", "winldd-")


def _archive(tmp_path: pathlib.Path, members=GOOD_MEMBERS) -> pathlib.Path:
    path = tmp_path / "archive" / "cft.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data, mode in members:
            info = zipfile.ZipInfo(name)
            info.external_attr = (mode & 0xFFFF) << 16
            bundle.writestr(info, data)
    return path


def _pin_for(archive: pathlib.Path, **overrides) -> chromium_runtime.ChromiumPin:
    data = archive.read_bytes()
    fields = {"platform": "test", "archive_size": len(data), "archive_sha256": hashlib.sha256(data).hexdigest()}
    return chromium_runtime.ChromiumPin(**{**fields, **overrides})


def _leftovers(root: pathlib.Path) -> list:
    return sorted(path.name for path in root.iterdir() if path.name.startswith("."))


# --- the pin and the host -------------------------------------------------------------


def test_pins_name_official_archives_outside_playwright_garbage_collection():
    assert set(chromium_runtime.PINS) == {"linux-arm64", "linux64", "mac-arm64", "mac-x64", "win32", "win64"}
    for key, pin in chromium_runtime.PINS.items():
        assert pin.platform == key and pin.archive_size > 0 and len(pin.archive_sha256) == 64
        assert pin.url == ("https://storage.googleapis.com/chrome-for-testing-public/"
                           f"{PINNED}/{key}/chrome-headless-shell-{key}.zip")
        assert not pin.directory.startswith(PLAYWRIGHT_GC_PREFIXES), pin.directory
        assert pin.macos_minimum == ((13, 0) if key.startswith("mac-") else ())


@pytest.mark.parametrize("system,machine,platform", [
    ("darwin", "arm64", "mac-arm64"), ("darwin", "x86_64", "mac-x64"),
    ("linux", "x86_64", "linux64"), ("win32", "AMD64", "win64"),
])
def test_each_supported_host_has_its_pin(system, machine, platform):
    assert chromium_runtime.host_pin(system=system, machine=machine).platform == platform


def test_an_unpinned_host_is_refused():
    with pytest.raises(chromium_runtime.ChromiumRefusal) as refused:
        chromium_runtime.host_pin(system="freebsd14", machine="amd64")
    assert refused.value.code == chromium_runtime.UNAVAILABLE and "webkit" in str(refused.value)


@pytest.mark.parametrize("release,refused", [
    ("12.7.6", True), ("12", True), ("13", False), ("13.0", False), ("26.4.1", False), ("", False),
])
def test_a_mac_below_the_builds_own_minimum_is_refused(release, refused):
    pin = chromium_runtime.PINS["mac-arm64"]
    if not refused:
        chromium_runtime.require_supported_host(pin, mac_release=release)
        return
    with pytest.raises(chromium_runtime.ChromiumRefusal) as refusal:
        chromium_runtime.require_supported_host(pin, mac_release=release)
    text = refusal.value.tool_text()
    assert text.startswith(f"⚠️ {chromium_runtime.UNAVAILABLE}: ") and "macOS 13.0 or later" in text
    assert release in text and "engine='webkit'" in text and "older" in text


def test_other_platforms_have_no_macos_floor():
    chromium_runtime.require_supported_host(chromium_runtime.PINS["linux64"], mac_release="12.0")


# --- installation ---------------------------------------------------------------------


def test_a_verified_archive_publishes_the_exact_pin_once(tmp_path):
    archive = _archive(tmp_path)
    pin = _pin_for(archive)
    root = tmp_path / "browsers"

    executable = chromium_runtime.install(root, pin, archive=archive)

    assert executable == root / pin.directory / "chrome-headless-shell"
    assert (root / pin.directory / "resources" / "data.pak").read_bytes() == b"pak"
    assert json.loads((root / pin.directory / "install.json").read_text()) == pin.receipt()
    if os.name != "nt":
        assert executable.stat().st_mode & 0o111
    assert chromium_runtime.find_installed([tmp_path / "elsewhere", root], pin) == executable
    archive.unlink()  # an existing exact pin needs no archive again
    assert chromium_runtime.install(root, pin) == executable
    assert _leftovers(root) == []


def test_a_receipt_for_anything_else_is_not_the_pin(tmp_path):
    archive = _archive(tmp_path)
    pin = _pin_for(archive)
    executable = chromium_runtime.install(tmp_path, pin, archive=archive)
    receipt = executable.parent / "install.json"
    receipt.write_text(json.dumps({**pin.receipt(), "version": "151.0.7922.34"}))
    assert chromium_runtime.installed_executable(tmp_path, pin) is None
    receipt.write_text(json.dumps(pin.receipt()))
    executable.unlink()
    assert chromium_runtime.installed_executable(tmp_path, pin) is None


def test_a_digest_mismatch_installs_nothing(tmp_path):
    archive = _archive(tmp_path)
    pin = _pin_for(archive, archive_sha256="0" * 64)
    root = tmp_path / "browsers"

    with pytest.raises(chromium_runtime.ChromiumRefusal) as refused:
        chromium_runtime.install(root, pin, archive=archive)

    assert refused.value.code == chromium_runtime.UNAVAILABLE and "chromium_archive_digest_mismatch" in str(refused.value)
    assert not (root / pin.directory).exists() and _leftovers(root) == []


@pytest.mark.parametrize("member", [
    (f"{FOLDER}/../escape", b"x", 0o100644),
    ("elsewhere/chrome-headless-shell", b"x", 0o100755),
    (f"{FOLDER}/link", b"/etc/passwd", 0o120777),
])
def test_an_unexpected_member_installs_nothing(tmp_path, member):
    archive = _archive(tmp_path, GOOD_MEMBERS + [member])
    pin = _pin_for(archive)
    root = tmp_path / "browsers"

    with pytest.raises(chromium_runtime.ChromiumRefusal) as refused:
        chromium_runtime.install(root, pin, archive=archive)

    assert "unexpected member" in str(refused.value)
    assert not (root / pin.directory).exists() and _leftovers(root) == []
    assert not (tmp_path / "escape").exists()


def test_an_incomplete_directory_under_the_pin_name_is_replaced(tmp_path):
    archive = _archive(tmp_path)
    pin = _pin_for(archive)
    partial = tmp_path / pin.directory
    partial.mkdir()
    (partial / "chrome-headless-shell").write_text("truncated")  # no receipt: never usable

    executable = chromium_runtime.install(tmp_path, pin, archive=archive)

    assert executable.read_bytes() == b"#!/bin/sh\nexit 0\n" and _leftovers(tmp_path) == []


def test_a_download_is_verified_through_verified_download_and_not_kept(tmp_path, monkeypatch):
    archive = _archive(tmp_path)
    pin = _pin_for(archive)
    root = tmp_path / "browsers"
    seen = []

    def fetch(*, url, destination, verify, size_bytes, **_kw):
        seen.append((url, size_bytes))
        shutil.copyfile(archive, destination)
        return verify(pathlib.Path(destination))

    monkeypatch.setattr(verified_download, "fetch_exact_file", fetch)

    executable = chromium_runtime.install(root, pin)

    assert seen == [(pin.url, pin.archive_size)] and executable.is_file()
    assert _leftovers(root) == []


def test_the_cli_prints_the_url_and_installs_from_an_archive(tmp_path, monkeypatch, capsys):
    archive = _archive(tmp_path)
    pin = _pin_for(archive)
    monkeypatch.setattr(chromium_runtime, "host_pin", lambda **_kw: pin)

    assert chromium_runtime.main(["url"]) == 0
    assert capsys.readouterr().out.strip() == pin.url
    root = tmp_path / "browsers"
    assert chromium_runtime.main(["install", "--browsers-path", str(root), "--archive", str(archive)]) == 0
    assert json.loads(capsys.readouterr().out)["executable"] == str(root / pin.directory / "chrome-headless-shell")
    bad = _archive(tmp_path / "bad", GOOD_MEMBERS[:2])
    assert chromium_runtime.main(["install", "--browsers-path", str(tmp_path / "b2"), "--archive", str(bad)]) == 1
    assert capsys.readouterr().err.startswith(f"⚠️ {chromium_runtime.UNAVAILABLE}: ")


# --- the sandbox decision -------------------------------------------------------------


@pytest.mark.parametrize("value,enabled", [
    (None, True), ("off", False), ("0", False), ("false", False), ("No", False),
])
def test_the_sandbox_is_on_unless_explicitly_opted_out(value, enabled):
    environ = {} if value is None else {chromium_runtime.SANDBOX_ENV: value}
    assert chromium_runtime.sandbox_enabled(environ) is enabled


def test_an_unrecognized_sandbox_setting_refuses_instead_of_guessing():
    with pytest.raises(chromium_runtime.ChromiumRefusal) as refused:
        chromium_runtime.sandbox_enabled({chromium_runtime.SANDBOX_ENV: "maybe"})
    assert refused.value.code == chromium_runtime.SANDBOX_UNAVAILABLE


@pytest.mark.parametrize("log_line", [
    "[FATAL:zygote_host_impl_linux.cc(132)] Running as root without --no-sandbox is not supported. "
    "See https://crbug.com/638180.",
    "[FATAL:zygote_host_impl_linux.cc(117)] No usable sandbox! If you are running on Ubuntu 23.10+ ...",
    "[FATAL:credentials.cc(127)] Check failed: . : Permission denied (13)",
])
def test_a_sandbox_start_failure_is_an_actionable_refusal(log_line):
    error = RuntimeError(f"BrowserType.launch: Target page, context or browser has been closed\n"
                         f"Browser logs:\n\n{log_line}\n")
    refusal = chromium_runtime.sandbox_refusal(error)
    assert refusal is not None and refusal.code == chromium_runtime.SANDBOX_UNAVAILABLE
    text = refusal.tool_text()
    assert log_line[:60] in text and f"{chromium_runtime.SANDBOX_ENV}=off" in text and "non-root" in text


def test_an_unrelated_launch_failure_is_not_called_a_sandbox_failure():
    error = RuntimeError("chrome-headless-shell: error while loading shared libraries: libnss3.so: "
                         "cannot open shared object file")
    assert chromium_runtime.sandbox_refusal(error) is None


# --- the browser tools launch only the pin --------------------------------------------


class _Chromium:
    def __init__(self, *, error=None, version=PINNED):
        self.calls, self.error, self.version, self.closed = [], error, version, []

    def launch(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return types.SimpleNamespace(version=self.version, close=lambda: self.closed.append(True))


def test_no_pinned_executable_means_no_launch():
    chromium = _Chromium()
    with pytest.raises(chromium_runtime.ChromiumRefusal):
        browser_mod._launch_browser_with_fallback(types.SimpleNamespace(chromium=chromium), executable_path=None)
    assert chromium.calls == []


def test_the_pin_launches_inside_its_sandbox_by_default(monkeypatch):
    monkeypatch.delenv(chromium_runtime.SANDBOX_ENV, raising=False)
    chromium = _Chromium()

    browser = browser_mod._launch_browser_with_fallback(types.SimpleNamespace(chromium=chromium),
                                                        executable_path="/pinned/chrome-headless-shell")

    assert browser.version == PINNED
    (kwargs,) = chromium.calls
    assert kwargs["executable_path"] == "/pinned/chrome-headless-shell" and kwargs["chromium_sandbox"] is True
    assert kwargs["headless"] is True and "--use-angle=swiftshader" in kwargs["args"]
    assert not any("no-sandbox" in arg or "site-per-process" in arg for arg in kwargs["args"])


def test_the_administrator_opt_out_launches_unsandboxed_and_warns(monkeypatch, caplog):
    monkeypatch.setenv(chromium_runtime.SANDBOX_ENV, "off")
    chromium = _Chromium()
    with caplog.at_level(logging.WARNING, logger=browser_mod.log.name):
        browser_mod._launch_browser_with_fallback(types.SimpleNamespace(chromium=chromium),
                                                  executable_path="/pinned/chrome-headless-shell")
    assert chromium.calls[0]["chromium_sandbox"] is False
    assert any("sandbox is OFF" in record.getMessage() for record in caplog.records)


def test_a_sandbox_that_cannot_start_refuses_with_no_second_launch():
    chromium = _Chromium(error=RuntimeError("Browser logs:\nRunning as root without --no-sandbox is not supported."))
    with pytest.raises(chromium_runtime.ChromiumRefusal) as refused:
        browser_mod._launch_browser_with_fallback(types.SimpleNamespace(chromium=chromium),
                                                  executable_path="/pinned/x", sandbox=True)
    assert refused.value.code == chromium_runtime.SANDBOX_UNAVAILABLE and len(chromium.calls) == 1
    assert chromium.calls[0]["chromium_sandbox"] is True


def test_other_launch_failures_keep_their_own_error():
    error = RuntimeError("libnss3.so: cannot open shared object file")
    chromium = _Chromium(error=error)
    with pytest.raises(RuntimeError) as raised:
        browser_mod._launch_browser_with_fallback(types.SimpleNamespace(chromium=chromium),
                                                  executable_path="/pinned/x", sandbox=True)
    assert raised.value is error and len(chromium.calls) == 1


def test_another_build_answering_is_closed_and_refused():
    chromium = _Chromium(version="151.0.7922.34")
    with pytest.raises(chromium_runtime.ChromiumRefusal) as refused:
        browser_mod._launch_browser_with_fallback(types.SimpleNamespace(chromium=chromium),
                                                  executable_path="/pinned/x", sandbox=True)
    assert "151.0.7922.34" in str(refused.value) and chromium.closed == [True]


def _fake_session(monkeypatch, chromium):
    page = types.SimpleNamespace(set_default_timeout=lambda _ms: None, on=lambda *_a: None)
    context = types.SimpleNamespace(new_page=lambda: page, route=lambda *_a: None, close=lambda: None)
    starts = []

    def launch(**kwargs):
        chromium.calls.append(kwargs)
        return types.SimpleNamespace(new_context=lambda **_kw: context, is_connected=lambda: True,
                                     close=lambda: None, version=PINNED)

    playwright = types.SimpleNamespace(chromium=types.SimpleNamespace(launch=launch), devices={}, stop=lambda: None)
    sync_api = types.SimpleNamespace(sync_playwright=lambda: types.SimpleNamespace(
        start=lambda: starts.append(True) or playwright))
    monkeypatch.setitem(__import__("sys").modules, "playwright.sync_api", sync_api)
    monkeypatch.setattr(browser_mod, "_HAS_STEALTH", False)
    monkeypatch.setattr(browser_mod, "_ensure_playwright_installed",
                        lambda *, engine, **_kw: "/pinned/chrome-headless-shell" if engine == "chromium" else None)
    return starts


def test_the_opt_out_is_disclosed_by_every_observation_of_that_session(monkeypatch):
    monkeypatch.setenv(chromium_runtime.SANDBOX_ENV, "off")
    chromium = _Chromium()
    _fake_session(monkeypatch, chromium)
    ctx = types.SimpleNamespace(browser_state=BrowserState(), task_constraint=None)

    _page, generation = browser_mod._ensure_browser(ctx)

    assert chromium.calls[0]["chromium_sandbox"] is False
    assert generation.browser_conditions.startswith(
        f"Browser conditions: chromium {PINNED} (Playwright; pinned Chrome for Testing headless shell, "
        f"Chromium sandbox disabled by administrator opt-out {chromium_runtime.SANDBOX_ENV}=off); ")


def test_an_unrecognized_sandbox_setting_refuses_before_playwright_starts(monkeypatch):
    monkeypatch.setenv(chromium_runtime.SANDBOX_ENV, "sometimes")
    chromium = _Chromium()
    starts = _fake_session(monkeypatch, chromium)
    ctx = types.SimpleNamespace(browser_state=BrowserState(), task_constraint=None)

    with pytest.raises(chromium_runtime.ChromiumRefusal):
        browser_mod._ensure_browser(ctx)
    assert starts == [] and chromium.calls == []


@pytest.mark.parametrize("cause", ["sandbox", "cache-mkdir"])
def test_tools_report_a_refusal_with_no_retry_and_no_unknown_outcome(tmp_path, monkeypatch, cause):
    attempts = []
    archive = _archive(tmp_path)
    pin = _pin_for(archive)
    root = tmp_path / "denied-cache"
    mkdir = pathlib.Path.mkdir

    def deny_cache(path, *args, **kwargs):
        if path == root:
            raise PermissionError("cache creation denied")
        return mkdir(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "mkdir", deny_cache)
    code = chromium_runtime.UNAVAILABLE if cause == "cache-mkdir" else chromium_runtime.SANDBOX_UNAVAILABLE

    def refuse(*, engine, allow_install):
        attempts.append(engine)
        if cause == "cache-mkdir":
            return chromium_runtime.install(root, pin, archive=archive)
        raise chromium_runtime.ChromiumRefusal(code,
                                               "Chromium could not start inside its sandbox (as root)")

    monkeypatch.setattr(browser_mod, "_ensure_playwright_installed", refuse)
    monkeypatch.setattr(browser_mod.browser_policy, "browser_url_block_reason", lambda *a, **k: "")
    ctx = types.SimpleNamespace(browser_state=BrowserState(), task_constraint=None)
    first = ctx.browser_state

    page = browser_mod._browse_page(ctx, url="https://example.test/")
    action = browser_mod._browser_action(ctx, action="click", selector="#go")

    assert page.startswith(f"⚠️ {code}: ")
    assert action.startswith(f"⚠️ {code}: ") and "OUTCOME_UNKNOWN" not in action
    assert attempts == ["chromium", "chromium"]  # one attempt per call: a refusal is not retried
    assert ctx.browser_state is not first and ctx.browser_state.pw_instance is None
