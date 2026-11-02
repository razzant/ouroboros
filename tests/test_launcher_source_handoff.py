"""The launcher's handoff to the checkout's own launcher: origin, cold start, transport, legacy console.

The handoff's operating-system effects are stand-ins: ``Popen``, ``execve`` and ``os._exit`` are recorded, Windows is
the platform layer's flag (``merge_hidden_kwargs``), and ``main()`` runs with its boot steps stubbed around the
real origin and change detection over temporary Git repositories. Git runs real child processes; no launcher,
window or installed app starts. Native proofs: ``tests/test_windows_launcher_handoff_native.py`` (Windows only).
"""
from __future__ import annotations

import ast
import ctypes
import ctypes.wintypes
import json
import os
import pathlib
import subprocess
import sys
import threading
import types

import pytest

import ouroboros.launcher_bootstrap as bootstrap_module
from ouroboros import platform_layer
from tests.test_launcher_handoff_stop import native_guard as native_guard

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MARK = bootstrap_module.LAUNCHER_HANDOFF_ENV
NO_WINDOW, NEW_GROUP, BREAKAWAY, DETACHED, NEW_CONSOLE = 0x08000000, 0x200, 0x01000000, 0x8, 0x10


class Handover(BaseException):
    pass


class Continued(BaseException):
    pass


class Exited(BaseException):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True).stdout.strip()


def checkout(tmp_path):
    """A managed checkout and a seed whose manifest names the commit it was cut from."""
    repo, seed = tmp_path / "repo", tmp_path / "seed"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.invalid")
    git(repo, "config", "user.name", "t")
    for rel in ("launcher.py", "server.py", "ouroboros/launcher_tray.py", "ouroboros/not_imported_yet.py",
                "supervisor/queue.py"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("GEN = 1\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "seed")
    seed.mkdir()
    origin = git(repo, "rev-parse", "HEAD")
    (seed / bootstrap_module.BUNDLE_MANIFEST_NAME).write_text(json.dumps({"source_sha": origin}), encoding="utf-8")
    return repo, seed, origin


def land(repo, rel):
    (repo / rel).write_text("GEN = 2\n", encoding="utf-8")
    git(repo, "commit", "-q", "-am", f"update {rel}")


# ------------------------------------------------------------------------------------- origin ---

def test_the_origin_is_what_a_frozen_or_seed_launcher_imported_never_the_newer_head(tmp_path, monkeypatch):
    repo, seed, origin = checkout(tmp_path)
    land(repo, "ouroboros/launcher_tray.py")
    monkeypatch.setattr(bootstrap_module, "_LOADED_CHECKOUT_SHA", "")
    assert bootstrap_module.remember_loaded_checkout(repo, bundle_dir=seed) == origin != git(repo, "rev-parse", "HEAD")
    assert bootstrap_module.remember_loaded_checkout(repo) == origin, "recorded once"
    monkeypatch.setattr(bootstrap_module, "_LOADED_CHECKOUT_SHA", "")
    assert bootstrap_module.remember_loaded_checkout(repo) == git(repo, "rev-parse", "HEAD"), "the checkout's own"
    monkeypatch.setattr(bootstrap_module, "_LOADED_CHECKOUT_SHA", "")
    assert bootstrap_module.remember_loaded_checkout(repo, bundle_dir=tmp_path / "no-manifest") == ""
    assert bootstrap_module.launcher_sources_changed(repo) is False, "an unknown origin never relaunches"


@pytest.mark.parametrize("rel,loaded_only,packaged", [
    (None, False, False), ("server.py", False, False), ("ouroboros/not_imported_yet.py", False, True),
    ("supervisor/queue.py", False, True), ("launcher.py", True, True)])
def test_the_cold_check_counts_the_packaged_trees_the_warm_check_only_what_is_loaded(tmp_path, rel, loaded_only,
                                                                                    packaged):
    """A frozen or seed launcher imports later helpers from its own bundle too, so its cold check counts the
    packaged trees; the warm (exit-42) check counts what is loaded. A server-only update changes neither, and
    a different commit, never an older or newer one, is what counts."""
    repo, seed, origin = checkout(tmp_path)
    if rel:
        land(repo, rel)
    assert bootstrap_module.launcher_sources_changed(repo, origin, bundle_dir=seed) is loaded_only
    assert bootstrap_module.launcher_sources_changed(repo, origin, bundle_dir=seed, packaged=True) is packaged


def test_the_launchers_git_asks_for_a_hidden_window(tmp_path, monkeypatch):
    """The packaged GUI launcher is the one Ouroboros process without a console: its own Git calls would open a
    console window each (one flash per start, two per restart) unless they ask Windows for none."""
    repo, seed, origin = checkout(tmp_path)
    land(repo, "launcher.py")
    head, seen, run = git(repo, "rev-parse", "HEAD"), [], subprocess.run

    def recorded(args, **kwargs):
        seen.append((args[:2], kwargs.pop("creationflags", 0)))
        return run(args, **kwargs)

    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    monkeypatch.setattr(bootstrap_module.subprocess, "run", recorded)
    assert bootstrap_module.checkout_sha(repo) == head
    assert bootstrap_module.launcher_sources_changed(repo, origin, bundle_dir=seed) is True
    assert [args for args, _ in seen] == [["git", "rev-parse"], ["git", "rev-parse"], ["git", "diff"]]
    assert all(flags & NO_WINDOW for _, flags in seen)


@pytest.mark.parametrize("changed", ["ouroboros/launcher_bootstrap.py", "ouroboros/launcher_server_reaper.py",
                                     "ouroboros/process_scope.py", "server.py"])
@pytest.mark.parametrize("compiled", [False, True])
def test_container_warm_handoff_counts_helpers_loaded_from_the_image(tmp_path, monkeypatch, changed, compiled):
    """Docker imports helpers from /app, with body and seed in two other directories."""
    repo, seed, app = (tmp_path / name for name in ("body/repo", "opt/ouroboros/seed", "app"))
    helper = changed if changed != "server.py" else "ouroboros/launcher_bootstrap.py"
    source = app / helper
    if compiled:
        source = source.with_suffix(".pyc")
    monkeypatch.setattr(bootstrap_module, "checkout_sha", lambda root: "updated")
    monkeypatch.setattr(bootstrap_module, "sys", types.SimpleNamespace(platform=sys.platform, modules={
        "loaded": types.SimpleNamespace(__file__=str(source)),
        "unrelated": types.SimpleNamespace(__file__=str(tmp_path / "elsewhere/server.py"))}))
    probes = []

    def diff(command, **kwargs):
        paths = command[command.index("--") + 1:]
        probes.append(paths)
        return types.SimpleNamespace(returncode=int(changed in paths))

    monkeypatch.setattr(bootstrap_module.subprocess, "run", diff)
    result = bootstrap_module.launcher_sources_changed(repo, "image", bundle_dir=seed, source_root=app)
    assert result is (changed != "server.py")
    assert probes == [["launcher.py", helper]], "only the launcher and its imported helpers count"


# ---------------------------------------------------------------------------------- transport ---

class Log:
    def __init__(self):
        self.lines = []

    def info(self, *args, **kwargs):
        self.lines.append(("info", args[0]))

    def warning(self, *args, **kwargs):
        self.lines.append(("warning", args[0]))

    def error(self, *args, **kwargs):
        self.lines.append(("error", args[0], bool(kwargs.get("exc_info"))))

    def exception(self, *args, **kwargs):
        self.error(*args, **kwargs, exc_info=True)


def context(tmp_path):
    return bootstrap_module.BootstrapContext(
        bundle_dir=tmp_path / "app dir" / "_internal", repo_dir=tmp_path / "Ouroboros data" / "repo",
        data_dir=tmp_path / "Ouroboros data", settings_path=tmp_path / "settings.json",
        embedded_python=str(tmp_path / "app dir" / "_internal" / "python-standalone" / "python.exe"),
        app_version="0.0.0", hidden_run=subprocess.run, save_settings=lambda settings: None, log=Log())


@pytest.fixture
def hand_over(native_guard, tmp_path, monkeypatch):
    """``exec_source_launcher`` with every effect recorded in ``calls``; ``os._exit`` raises ``Exited``."""
    calls = []
    import contextlib

    # Keep the console/argv contract tests independent of native handle creation.
    monkeypatch.setattr(platform_layer, "inheritable_self_handle", lambda: contextlib.nullcontext((
        501, {"startupinfo": object(), "close_fds": True})))
    monkeypatch.setattr(bootstrap_module, "source_launcher_importable", lambda *a, **kw: True)

    def run(argv, *, windows, frozen=False, acquired=True, spawn_error=None, console_clients=0,
            cancelled=lambda: False):
        def popen(args, **kwargs):
            if spawn_error:
                calls.append("refused")
                raise spawn_error
            calls.append(("popen", args, kwargs))
            return types.SimpleNamespace(pid=9990001)

        def execve(executable, args, env):
            if spawn_error:
                calls.append("refused")
                raise spawn_error
            calls.append(("exec", executable, args, env))
            raise Handover()  # an image replacement never returns

        def leave(code):
            calls.append(("exit", code))
            raise Exited(code)

        monkeypatch.setattr(platform_layer, "IS_WINDOWS", windows)
        # GetConsoleProcessList as the shell answers it: 0 for a GUI-subsystem start, its clients for a console.
        monkeypatch.setattr(platform_layer, "_kernel32", types.SimpleNamespace(
            GetConsoleProcessList=lambda buffer, size: console_clients), raising=False)
        monkeypatch.setattr(platform_layer, "ctypes", ctypes, raising=False)
        monkeypatch.setattr(bootstrap_module.subprocess, "Popen", popen)
        monkeypatch.setattr(bootstrap_module.os, "execve", execve)
        monkeypatch.setattr(bootstrap_module.os, "_exit", leave)
        monkeypatch.setattr(bootstrap_module.sys, "frozen", frozen, raising=False)
        ctx = context(tmp_path)
        run.log = ctx.log
        bootstrap_module.exec_source_launcher(
            ctx, argv, release_lock=lambda: calls.append("release"),
            acquire_lock=lambda: calls.append("acquire") or acquired,
            before_release=lambda: calls.append("remove icon"), on_failure=lambda: calls.append("restore shell"),
            desktop="--no-ui" not in argv, macos=False, cancelled=cancelled)
        return ctx

    run.calls = calls
    return run


@pytest.mark.parametrize("console_clients", [0, 2])  # the packaged app (no console), a terminal run (attached)
def test_windows_starts_the_successor_quoted_after_the_lock_then_leaves(hand_over, console_clients):
    """No console window for a console-less launcher's successor; a terminal run's keeps that terminal."""
    with pytest.raises(Exited) as exited:
        hand_over(["--launch-intent", "automatic", "--runtime-venv", "C:\\body venv"], windows=True, frozen=True,
                  console_clients=console_clients)
    calls = hand_over.calls
    assert exited.value.code == 0 and [call if isinstance(call, str) else call[0] for call in calls] == [
        "remove icon", "release", "popen", "exit"], "remove the old icon while still holding the lock"
    _, args, kwargs = calls[2]
    assert isinstance(args, list) and args[1].endswith(os.path.join("repo", "launcher.py"))
    assert args[2:4] == ["--launch-intent", "automatic"] and "--runtime-venv" not in args, "consumed: not handed on"
    assert args[-2] == "--seed-bundle" and args[-1].endswith("_internal"), "a frozen app hands over its bundle"
    assert '"' in subprocess.list2cmdline(args), "a list: Windows gets each spaced path quoted"
    flags = kwargs.get("creationflags", 0)
    assert flags == (0 if console_clients else NO_WINDOW), "hidden only without a console to inherit"
    assert not flags & (NEW_GROUP | BREAKAWAY | DETACHED | NEW_CONSOLE), "never a group, breakaway or console change"
    assert kwargs["close_fds"] is True and kwargs["env"][MARK] == "1" and kwargs["env"]["PATH"] == os.environ["PATH"]


def test_unavailable_parent_handle_keeps_compatible_source_handoff_but_discloses_gap(hand_over, monkeypatch):
    def unavailable():
        raise OSError("fake handle inheritance unavailable")

    monkeypatch.setattr(platform_layer, "inheritable_self_handle", unavailable)
    with pytest.raises(Exited) as exited:
        hand_over(["--launch-intent", "owner"], windows=True)
    _, _, (_, args, kwargs), _ = hand_over.calls
    assert exited.value.code == 0 and args[2:] == ["--launch-intent", "owner"]
    assert "startupinfo" not in kwargs
    assert "OUROBOROS_PLANNED_RESTART_PARENT_HANDLE" not in kwargs["env"]
    assert ("error", "Restart parent handle unavailable; the successor cannot prove its continuation",
            True) in hand_over.log.lines


@pytest.mark.parametrize("windows", [True, False])
def test_a_refused_successor_leaves_this_launcher_locked_and_running(hand_over, windows):
    ctx = hand_over(["--launch-intent", "owner"], windows=windows, spawn_error=PermissionError(13, "Access is denied"))
    assert hand_over.calls == ["remove icon", "release", "refused", "acquire", "restore shell"], "restore only under the retaken lock"
    assert ("error", "The checkout's launcher did not start", True) in ctx.log.lines


@pytest.mark.parametrize("windows", [True, False])
def test_a_refused_successor_whose_lock_another_launcher_took_leaves(hand_over, windows):
    with pytest.raises(Exited) as exited:
        hand_over(["--launch-intent", "owner"], windows=windows, acquired=False, spawn_error=OSError("blocked"))
    assert exited.value.code == 1, "never an unlocked launcher beside the one that took the lock"
    assert hand_over.calls == ["remove icon", "release", "refused", "acquire", ("exit", 1)]


def test_posix_replaces_the_process_image_with_the_marked_environment(hand_over):
    with pytest.raises(Handover):
        hand_over(["--no-ui", "--seed-bundle", "/opt/ouroboros/seed"], windows=False)
    (cleanup, release, (kind, executable, args, env)) = hand_over.calls
    assert cleanup == "remove icon"
    assert release == "release" and kind == "exec" and executable == args[0]
    assert args[2:] == ["--no-ui", "--seed-bundle", "/opt/ouroboros/seed"], "a seed copy's argv, unchanged"
    assert env[MARK] == "1"


@pytest.mark.parametrize("windows", [True, False])
@pytest.mark.parametrize("when", ["before teardown", "after unlock"])
def test_cancelled_handoff_never_starts_a_successor(hand_over, windows, when):
    cancelled = lambda: when == "before teardown" or "release" in hand_over.calls
    try:
        hand_over([], windows=windows, cancelled=cancelled)
    except (Handover, Exited):
        pass
    assert hand_over.calls == ([] if when == "before teardown" else [
        "remove icon", "release", "acquire", "restore shell"]), "a stop cannot spawn/exec over its Panic owner"


def test_cancelled_handoff_keeps_panic_alive_even_if_the_lock_was_taken(hand_over):
    try:
        hand_over([], windows=True, acquired=False, cancelled=lambda: "release" in hand_over.calls)
    except Exited:
        pass
    assert hand_over.calls == ["remove icon", "release", "acquire"], "never hard-exit over pending Panic persistence"


# --------------------------------------------------------------------------------- cold start ---


@pytest.mark.parametrize("windows", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_import_broken_successor_keeps_frozen_or_seed_shell(hand_over, monkeypatch, windows, frozen):
    monkeypatch.setattr(bootstrap_module, "source_launcher_importable", lambda *a, **kw: False)
    hand_over([], windows=windows, frozen=frozen)
    assert hand_over.calls == [], "probe refusal precedes icon removal, unlock and process replacement"


def test_released_pid_lock_never_unlinks_a_new_owners_file(tmp_path, monkeypatch):
    lock = tmp_path / "successor.pid"
    lock.write_text("999999")
    monkeypatch.setattr(platform_layer, "_lock_fd", None)
    platform_layer.pid_lock_release(str(lock))
    assert lock.read_text() == "999999"

@pytest.fixture
def boot(monkeypatch):
    """``launcher.main`` up to the first-run settings (``Continued``) or a handoff (``Handover``), around the real
    origin recording and change detection; ``events`` records the boot and the handoff's argv."""
    import atexit
    import launcher

    monkeypatch.setattr(atexit, "register", lambda callback: None)
    monkeypatch.setattr(launcher, "_shutdown_event", threading.Event())
    for name in ("_headless", "_external_ui", "_external_host_update", "_external_seed_bundle", "_launch_argv",
                 "_launch_intent", "_runtime_venv", "EMBEDDED_PYTHON"):
        monkeypatch.setattr(launcher, name, getattr(launcher, name))  # main() sets them; restored afterwards
    events = []

    def run(repo, argv, *, frozen_bundle=None, macos=False, launcher_file=None, open_pending=False, stop_at=None):
        class Listener:
            def __init__(self, exit_launcher, read_port, shutdown_event, **kwargs):
                self._open_pending = open_pending
                self.panic = threading.Event()

            def listen(self, lock_path):
                if stop_at == "listen":
                    launcher._shutdown_event.set()
                return self

        def bootstrap():
            events.append("bootstrap")
            if stop_at == "bootstrap":
                launcher._shutdown_event.set()
            return True

        def handover():
            events.append(("handover", list(launcher._launch_argv)))
            raise Handover()

        def continued():
            events.append("first-run settings")
            raise Continued()

        monkeypatch.setitem(sys.modules, "webview", types.ModuleType("webview"))
        for name, value in (("IS_WINDOWS", False), ("IS_MACOS", macos), ("_detect_headless", lambda: None),
                            ("acquire_pid_lock", lambda: True), ("release_pid_lock", lambda: None),
                            ("automatic_launch_allowed", lambda *args: True), ("Background", Listener),
                            ("check_git", lambda: True), ("bootstrap_repo", bootstrap),
                            ("_prepare_first_run_settings", continued), ("_exec_source_launcher", handover),
                            ("REPO_DIR", repo), ("_headless", False)):
            monkeypatch.setattr(launcher, name, value)
        monkeypatch.setattr(bootstrap_module, "_LOADED_CHECKOUT_SHA", "")
        monkeypatch.setattr(launcher.sys, "frozen", frozen_bundle is not None, raising=False)
        if frozen_bundle is not None:
            monkeypatch.setattr(launcher.sys, "_MEIPASS", str(frozen_bundle), raising=False)
        source_root = frozen_bundle or bootstrap_module.parse_launch_options(argv).seed_bundle or repo
        monkeypatch.setattr(launcher, "__file__", str(launcher_file or source_root / "launcher.py"))
        if stop_at:
            launcher.main(argv)
        else:
            with pytest.raises((Handover, Continued)):
                launcher.main(argv)
        return events

    return run


@pytest.mark.parametrize("stop_at", ["listen", "bootstrap"])
def test_startup_panic_fences_bootstrap_handoff_and_server_admission(tmp_path, boot, stop_at):
    repo, seed, _origin = checkout(tmp_path)
    land(repo, "ouroboros/launcher_tray.py")  # would force cold handoff without the Panic fence
    events = boot(repo, ["--launch-intent", "owner"], frozen_bundle=seed, stop_at=stop_at)
    assert events == ([] if stop_at == "listen" else ["bootstrap"])


@pytest.mark.parametrize("rel,hands_over", [(None, False), ("server.py", False),
                                            ("ouroboros/not_imported_yet.py", True), ("supervisor/queue.py", True)])
def test_a_frozen_app_loads_the_checkouts_launcher_at_cold_start_only_when_its_code_differs(tmp_path, boot, rel,
                                                                                         hands_over):
    """Owner answer 2A: after a full exit too, the updated shell, before any server or main window. Same code
    and a server-only update keep the frozen launcher (no forced handoff on every start)."""
    repo, seed, origin = checkout(tmp_path)
    if rel:
        land(repo, rel)
    events = boot(repo, ["--launch-intent", "owner"], frozen_bundle=seed)
    assert bootstrap_module._LOADED_CHECKOUT_SHA == origin, "the bundle's commit, read before bootstrap"
    assert events == ["bootstrap", ("handover", ["--launch-intent", "owner"])] if hands_over \
        else events == ["bootstrap", "first-run settings"]


def test_a_frozen_macos_app_without_a_signed_entry_keeps_its_launcher_at_cold_start(tmp_path, boot):
    """A legacy macOS bundle cannot preserve its identity/profile in bare Python at cold start.
    Current bundles hand off through the signed entry (test_macos_source_entry)."""
    repo, seed, _origin = checkout(tmp_path)
    land(repo, "ouroboros/not_imported_yet.py")
    assert boot(repo, ["--launch-intent", "owner"], frozen_bundle=seed, macos=True) == ["bootstrap", "first-run settings"]


@pytest.mark.parametrize("rel,hands_over", [(None, False), ("server.py", False), ("ouroboros/launcher_tray.py", True)])
def test_a_headless_seed_copy_hands_over_at_cold_start_only_when_its_code_differs(tmp_path, boot, rel, hands_over):
    """Android and the managed container start a seed copy with ``--no-ui``: its argv reaches the checkout's
    launcher unchanged, and same code or a server-only update runs the seed copy as before."""
    repo, seed, origin = checkout(tmp_path)
    if rel:
        land(repo, rel)
    argv = ["--no-ui", "--host-update", str(tmp_path / "hook.py"), "--seed-bundle", str(seed),
            "--launch-intent", "automatic"]
    events = boot(repo, argv)
    assert bootstrap_module._LOADED_CHECKOUT_SHA == origin
    assert events == ["bootstrap", ("handover", argv)] if hands_over else events == ["bootstrap", "first-run settings"]


def test_the_checkouts_own_launcher_never_hands_over_at_cold_start(tmp_path, boot):
    """Its modules are the checkout's at load time, even with ``--seed-bundle`` still in its argv."""
    repo, seed, _origin = checkout(tmp_path)
    land(repo, "ouroboros/launcher_tray.py")
    events = boot(repo, ["--no-ui", "--seed-bundle", str(seed)], launcher_file=repo / "launcher.py")
    assert bootstrap_module._LOADED_CHECKOUT_SHA == git(repo, "rev-parse", "HEAD")
    assert events == ["bootstrap", "first-run settings"]


def test_an_open_during_the_boot_shows_the_successors_window(tmp_path, boot):
    repo, seed, _origin = checkout(tmp_path)
    land(repo, "ouroboros/launcher_tray.py")
    events = boot(repo, ["--launch-intent", "automatic"], frozen_bundle=seed, open_pending=True)
    assert events[-1] == ("handover", ["--launch-intent", "automatic", "--launch-intent", "owner"])
    assert bootstrap_module.parse_launch_options(events[-1][1]).launch_intent == "owner", "a quiet start is cancelled"


# ---------------------------------------------------------------------------- legacy console ---

@pytest.fixture
def prologue(monkeypatch, capsys, tmp_path):
    """``launcher._rehome_legacy_handoff`` on a simulated Windows; ``calls`` holds its Popen and exit."""
    import launcher

    calls = []

    def run(argv, *, platform="win32", frozen=False, marked=False, spawn_error=None, provenance=True,
            clients=1, own_client=True, visible=True, window=123, subsystem=2, malformed=False):
        def popen(args, **kwargs):
            if spawn_error:
                raise spawn_error
            calls.append(("popen", args, kwargs))

        def leave(code):
            calls.append(("exit", code))
            raise Exited(code)

        class NativeCall:
            def __init__(self, fn):
                self.fn = fn

            def __call__(self, *args):
                return self.fn(*args)

        def console_list(buffer, size):
            buffer[0] = os.getpid() if own_client else os.getpid() + 1
            return clients

        kernel = types.SimpleNamespace(GetConsoleWindow=NativeCall(lambda: window),
                                       GetConsoleProcessList=NativeCall(console_list))
        user = types.SimpleNamespace(IsWindowVisible=NativeCall(lambda hwnd: visible))
        monkeypatch.setattr(ctypes, "WinDLL", lambda name: kernel if name == "kernel32" else user, raising=False)
        if provenance:
            image = tmp_path / "Ouroboros.exe"
            pe = bytearray(222)
            pe[:2], pe[60:64], pe[128:132] = b"MZ", (128).to_bytes(4, "little"), b"PE\0\0"
            pe[152:154], pe[220:222] = b"\x0b\x02", subsystem.to_bytes(2, "little")
            image.write_bytes(b"broken" if malformed else pe)
            monkeypatch.setenv("_PYI_ARCHIVE_FILE", str(image))
        else:
            monkeypatch.delenv("_PYI_ARCHIVE_FILE", raising=False)
        monkeypatch.setattr(launcher.sys, "platform", platform)
        monkeypatch.setattr(launcher.sys, "frozen", frozen, raising=False)
        monkeypatch.setattr(launcher.sys, "argv", ["launcher.py", *argv])
        monkeypatch.setattr(launcher.sys, "orig_argv", ["python.exe", "-X", "utf8", "launcher.py", *argv])
        monkeypatch.setattr(launcher.subprocess, "CREATE_NO_WINDOW", NO_WINDOW, raising=False)
        monkeypatch.setattr(launcher.subprocess, "Popen", popen)
        monkeypatch.setattr(launcher.os, "_exit", leave)
        if marked:
            monkeypatch.setenv(MARK, "1")
        else:
            monkeypatch.delenv(MARK, raising=False)
        try:
            launcher._rehome_legacy_handoff()
        except Exited:
            pass
        assert MARK not in os.environ, "the mark is consumed at entry, so it never reaches the server or a child"
        return calls

    run.seed = str(tmp_path / "_internal")
    run.capsys = capsys
    return run


@pytest.mark.parametrize("equals", [False, True])

def test_an_older_apps_handoff_restarts_once_without_its_console_window(prologue, equals):
    """A packaged 7.7.0 hands over through the C runtime's exec, which opens a console window for this source
    launcher; it restarts itself hidden and marked, with the same interpreter options and arguments."""
    seed = [f"--seed-bundle={prologue.seed}"] if equals else ["--seed-bundle", prologue.seed]
    calls = prologue([*seed, "--launch-intent", "owner"])
    (_, args, kwargs), leave = calls
    assert args == [sys.executable, "-X", "utf8", "launcher.py", *seed, "--launch-intent", "owner"]
    assert kwargs["creationflags"] == NO_WINDOW and kwargs["close_fds"] is True and kwargs["env"][MARK] == "1"
    assert leave == ("exit", 0), "never FreeConsole-and-continue: this process leaves at once"


@pytest.mark.parametrize("argv,options", [
    (["--seed-bundle", "C:\\seed"], {"platform": "darwin"}),
    (["--seed-bundle", "C:\\seed"], {"platform": "linux"}),
    (["--seed-bundle", "C:\\seed"], {"frozen": True}),
    (["--seed-bundle", "C:\\seed"], {"marked": True}),
    (["--no-ui", "--seed-bundle", "C:\\seed"], {}),
    (["--launch-intent", "owner"], {}),  # a source launcher someone started from a terminal: its console is theirs
])
def test_every_other_start_keeps_its_console(prologue, argv, options):
    assert prologue(argv, **options) == []


@pytest.mark.parametrize("facts", [
    {"provenance": False},  # manual source CLI, even with seed argv and a console of its own
    {"clients": 2},         # an intentional shared terminal, even with inherited GUI provenance
    {"clients": 0}, {"own_client": False}, {"window": 0}, {"visible": False},
    {"subsystem": 3}, {"malformed": True},  # a CLI executable or unproven GUI origin
])
def test_legacy_repair_requires_gui_origin_and_its_own_visible_console(prologue, facts):
    assert prologue(["--seed-bundle", prologue.seed], **facts) == []


def test_unrelated_packaged_provenance_does_not_detach_source_cli(prologue):
    assert prologue(["--seed-bundle", str(pathlib.Path(prologue.seed).parent / "other" / "_internal")]) == []


def test_a_refused_restart_keeps_the_process_in_its_console(prologue):
    assert prologue(["--seed-bundle", prologue.seed], spawn_error=OSError("blocked by policy")) == []
    assert "could not leave this console window: blocked by policy" in prologue.capsys.readouterr().err


def test_the_prologue_runs_before_any_package_import_and_only_as_the_entry():
    """``ouroboros/__init__.py`` may run an armed body adoption that would wait in the visible console, so the
    prologue precedes the first package import; a multiprocessing child re-imports launcher.py under another
    name and must never respawn itself."""
    tree = ast.parse((REPO_ROOT / "launcher.py").read_text(encoding="utf-8"))
    body = tree.body

    def is_package_import(node):
        return (isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "ouroboros") or (
            isinstance(node, ast.Import) and any(alias.name.split(".")[0] == "ouroboros" for alias in node.names))

    first_package = next(index for index, node in enumerate(body) if is_package_import(node))
    guard = next(index for index, node in enumerate(body) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "__name__ == '__main__'"
                 and ast.unparse(node.body[0]) == "_rehome_legacy_handoff()")
    assert guard < first_package
    for node in body[:guard]:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
            assert all(name.split(".")[0] in sys.stdlib_module_names or name == "__future__" for name in names)
    prologue = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == "_rehome_legacy_handoff")
    assert [alias.name for node in ast.walk(prologue) if isinstance(node, ast.Import)
            for alias in node.names] == ["runpy"], "the Windows leaf is loaded by path, before package adoption"
    leaf = (REPO_ROOT / "ouroboros" / "launcher_windows_runtime.py").read_text(encoding="utf-8")
    assert not any(is_package_import(node) for node in ast.walk(ast.parse(leaf)))
    assert ast.unparse(prologue).count(repr(MARK)) == 1 and leaf.count('"' + MARK + '"') == 1


@pytest.mark.parametrize("windows,clients,flags", [(True, 0, NO_WINDOW), (True, 1, None), (False, 0, None)])
def test_a_successor_keeps_an_attached_console_and_is_hidden_only_without_one(monkeypatch, windows, clients, flags):
    """``GetConsoleProcessList``, not a window's visibility or a sole-client count: any attached console is kept."""
    asked = []
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", windows)
    monkeypatch.setattr(platform_layer, "_kernel32", types.SimpleNamespace(
        GetConsoleProcessList=lambda buffer, size: asked.append(size) or clients), raising=False)
    monkeypatch.setattr(platform_layer, "ctypes", ctypes, raising=False)
    assert platform_layer.inherited_console_kwargs() == ({"creationflags": flags} if flags else {})
    assert asked == ([1] if windows else []), "elsewhere nothing is asked"


def test_linux_source_handoff_keeps_native_bundle_environment(hand_over, monkeypatch, tmp_path):
    from ouroboros.desktop_media import LINUX_DESKTOP_KEYS, host_environment

    import dataclasses
    embedded_python = str(tmp_path / 'app dir/_internal/python-standalone/bin/python3')
    original_context = context
    monkeypatch.setattr(sys.modules[__name__], 'context', lambda path: dataclasses.replace(
        original_context(path), embedded_python=embedded_python))
    monkeypatch.setattr(bootstrap_module.sys, 'executable', '/seed/Ouroboros.bin')
    native = {'LD_LIBRARY_PATH': '/seed/_internal:/seed/_internal/linux-gtk/lib/x86_64-linux-gnu',
              'GI_TYPELIB_PATH': '/seed/_internal/linux-gtk/lib/x86_64-linux-gnu/girepository-1.0',
              'GIO_MODULE_DIR': '/seed/_internal/linux-gtk/lib/x86_64-linux-gnu/gio/modules',
              'PYWEBVIEW_GUI': 'gtk', 'OUROBOROS_DESKTOP_ENV_SAVED': '1',
              **{'OUROBOROS_DESKTOP_HOST_' + key: '' for key in LINUX_DESKTOP_KEYS},
              'OUROBOROS_DESKTOP_HOST_LD_LIBRARY_PATH': '1',
              'OUROBOROS_DESKTOP_HOST_XDG_DATA_DIRS': '1/caller/share'}
    for key, value in native.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(Handover):
        hand_over([], windows=False, frozen=True)
    executed = next(call for call in hand_over.calls if isinstance(call, tuple) and call[0] == 'exec')
    assert all(executed[3][key] == value for key, value in native.items())
    host = host_environment(executed[3])
    assert host['LD_LIBRARY_PATH'] == '' and host['XDG_DATA_DIRS'] == '/caller/share'
    assert 'GI_TYPELIB_PATH' not in host
    assert '--seed-bundle' in executed[2]
    assert executed[1] == executed[2][0] == embedded_python
    assert executed[1] != bootstrap_module.sys.executable
