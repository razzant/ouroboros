"""Runtime Git selection (ouroboros/git_selection.py) and its Update/entry consumers.

Selection decisions and Update call sites are verified with inert fixture paths and in-memory
process interception (no executable Git stand-ins). Real host Git reuse and macOS xcode-select
are verified separately against the system's own binaries."""

from __future__ import annotations

import atexit
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ouroboros import git_selection


@pytest.fixture
def _restore_git_selection_env(monkeypatch):
    """Anchor Git-selection env keys at the bottom of monkeypatch's undo stack so keys that
    start unset, get written by select_git(), and are later delenv'd never leak across tests."""
    for key in (git_selection.SELECTED_GIT_ENV, "GIT_EXEC_PATH", "GIT_TEMPLATE_DIR", "PATH"):
        original = os.environ.get(key)
        monkeypatch.setenv(key, "" if original is None else original)
        if original is None:
            os.environ.pop(key, None)


def _inert(path: pathlib.Path) -> pathlib.Path:
    """Create a non-executable placeholder file at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("inert git fixture", encoding="utf-8")
    return path


def _patch_posix_which(monkeypatch) -> None:
    """In-memory POSIX ``shutil.which`` for ``git`` over inert fixture files (no chmod +x)."""
    def fake_which(cmd: str, path: str | None = None) -> str | None:
        if cmd != "git" or not path:
            return None
        for folder in path.split(os.pathsep):
            if not folder:
                continue
            candidate = os.path.join(folder, "git")
            if os.path.isfile(candidate):
                return candidate
        return None

    monkeypatch.setattr(git_selection.shutil, "which", fake_which)


def _fake_runner(responses: dict, *, calls: list | None = None, xcode_select=None):
    """In-memory replacement for ``subprocess.run`` used by ``select_git``."""
    resolved = {os.path.realpath(str(key)): value for key, value in responses.items()}

    def run(argv, **kwargs):
        if calls is not None:
            calls.append((list(argv), kwargs))
        if list(argv) == ["/usr/bin/xcode-select", "-p"]:
            if callable(xcode_select):
                return xcode_select(argv, kwargs)
            return xcode_select if xcode_select is not None else SimpleNamespace(returncode=2, stdout=b"")
        target = resolved.get(os.path.realpath(str(argv[0])))
        if callable(target):
            return target(argv, kwargs)
        if isinstance(target, BaseException):
            raise target
        if isinstance(target, bytes):
            if kwargs.get("stdout") is not None:
                kwargs["stdout"].write(target)
            return SimpleNamespace(returncode=0)
        if isinstance(target, int):
            return SimpleNamespace(returncode=target)
        return SimpleNamespace(returncode=1)

    return run


def test_a_working_system_git_keeps_the_environment_and_none_selects_the_bundled_copy(tmp_path, monkeypatch):
    _patch_posix_which(monkeypatch)
    sys_git = _inert(tmp_path / "sys" / "bin" / "git")
    bundle = tmp_path / "bundle"
    bundled_git = _inert(bundle / "git-standalone" / "bin" / "git")
    run = _fake_runner({sys_git: b"git version 2.47.0\n", bundled_git: b"git version 2.56.1\n"})

    # Working system Git first on PATH: PATH and the person's Git config stay untouched.
    env = {"PATH": str(sys_git.parent), "GIT_CONFIG_GLOBAL": "/owner/gitconfig"}
    assert git_selection.select_git(bundle, env, platform="linux", run=run) == "system"
    assert env == {
        "PATH": str(sys_git.parent),
        "GIT_CONFIG_GLOBAL": "/owner/gitconfig",
        git_selection.SELECTED_GIT_ENV: str(sys_git),
    }

    # No system Git on PATH: the bundled copy goes first with its relocation variables.
    empty = tmp_path / "empty"
    empty.mkdir()
    env = {"PATH": str(empty)}
    assert git_selection.select_git(bundle, env, platform="linux", run=run) == "bundled"
    assert env["PATH"].split(os.pathsep) == [str(bundled_git.parent), str(empty)]
    assert env["GIT_EXEC_PATH"] == str(bundle / "git-standalone" / "libexec" / "git-core")
    assert env["GIT_TEMPLATE_DIR"] == str(bundle / "git-standalone" / "share" / "git-core" / "templates")
    assert env[git_selection.SELECTED_GIT_ENV] == str(bundled_git)


def test_a_git_that_cannot_answer_is_passed_over_for_the_next_then_the_bundle_then_the_page(tmp_path, monkeypatch):
    _patch_posix_which(monkeypatch)
    broken_exit = _inert(tmp_path / "broken-exit" / "git")
    broken_os = _inert(tmp_path / "broken-os" / "git")
    broken_noise = _inert(tmp_path / "broken-noise" / "git")
    working_sys = _inert(tmp_path / "good-sys" / "git")
    working_extra = _inert(tmp_path / "good-extra" / "git")
    broken_extra = _inert(tmp_path / "broken-extra" / "git")
    bundle = tmp_path / "bundle"
    bundled_git = _inert(bundle / "git-standalone" / "bin" / "git")

    calls: list = []
    run = _fake_runner(
        {
            broken_exit: 127,
            broken_os: OSError("exec format error"),
            broken_noise: b"not a git version banner\n",
            broken_extra: 1,
            working_sys: b"git version 2.45.0\n",
            working_extra: b"git version 2.46.0\n",
            bundled_git: b"git version 2.56.1\n",
        },
        calls=calls,
    )

    # Broken candidates before a working system Git: working Git wins and its folder moves first on PATH.
    # Repeated identical folder entries are probed once per distinct realpath.
    env = {"PATH": os.pathsep.join([str(broken_exit.parent), str(broken_exit.parent),
                                    str(broken_os.parent), str(broken_noise.parent), str(working_sys.parent)])}
    assert git_selection.select_git(bundle, env, platform="linux", run=run) == "system"
    assert env[git_selection.SELECTED_GIT_ENV] == str(working_sys)
    assert env["PATH"].split(os.pathsep)[0] == str(working_sys.parent)
    assert sum(1 for argv, _ in calls if os.path.realpath(argv[0]) == os.path.realpath(broken_exit)) == 1

    # Broken on PATH, working in extra_dirs: extra Git wins and moves first on PATH.
    env = {"PATH": str(broken_exit.parent)}
    assert git_selection.select_git(bundle, env, extra_dirs=[str(working_extra.parent)],
                                    platform="linux", run=run) == "system"
    assert env[git_selection.SELECTED_GIT_ENV] == str(working_extra)
    assert env["PATH"].split(os.pathsep)[0] == str(working_extra.parent)

    # Broken on PATH and in extra_dirs: the bundled copy goes first; an extra folder never joins PATH for it.
    env = {"PATH": str(broken_exit.parent)}
    assert git_selection.select_git(bundle, env, extra_dirs=[str(broken_extra.parent)],
                                    platform="linux", run=run) == "bundled"
    assert env[git_selection.SELECTED_GIT_ENV] == str(bundled_git)
    assert env["PATH"].split(os.pathsep) == [str(bundled_git.parent), str(broken_exit.parent)]

    # Neither system nor bundled works: "missing" (caller shows the setup page) and stale selection cleared.
    failing_run = _fake_runner({broken_exit: 1, bundled_git: 1})
    env = {"PATH": str(broken_exit.parent), git_selection.SELECTED_GIT_ENV: "/stale/git"}
    assert git_selection.select_git(bundle, env, platform="linux", run=failing_run) == "missing"
    assert git_selection.SELECTED_GIT_ENV not in env
    assert "GIT_EXEC_PATH" not in env and "GIT_TEMPLATE_DIR" not in env


def test_a_git_is_probed_with_the_path_and_loader_values_it_will_run_with(tmp_path, monkeypatch):
    _patch_posix_which(monkeypatch)
    sys_git = _inert(tmp_path / "sys" / "bin" / "git")
    bundle = tmp_path / "bundle"
    bundled_git = _inert(bundle / "git-standalone" / "bin" / "git")
    seen_envs: dict[str, dict] = {}

    def restore_host(candidate_env):
        restored = dict(candidate_env)
        restored.pop("LD_LIBRARY_PATH", None)
        restored["HOST_RESTORED"] = "1"
        return restored

    def probe_requires_restored_loader(argv, **kwargs):
        probe_env = kwargs["env"]
        seen_envs[os.path.realpath(argv[0])] = probe_env
        if "LD_LIBRARY_PATH" in probe_env or probe_env.get("HOST_RESTORED") != "1":
            return SimpleNamespace(returncode=127)
        if os.path.realpath(argv[0]) == os.path.realpath(sys_git):
            kwargs["stdout"].write(b"git version 2.48.0\n")
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=1)

    env = {"PATH": str(sys_git.parent), "LD_LIBRARY_PATH": "/bundle/linux-gtk/lib"}
    assert git_selection.select_git(
        bundle, env, platform="linux", run=probe_requires_restored_loader, host=restore_host
    ) == "system"
    probed = seen_envs[os.path.realpath(sys_git)]
    assert probed["PATH"].split(os.pathsep)[0] == str(sys_git.parent)
    assert "LD_LIBRARY_PATH" not in probed and probed["HOST_RESTORED"] == "1"
    assert env["LD_LIBRARY_PATH"] == "/bundle/linux-gtk/lib"  # GUI shell keeps its own loader path

    # Same host-environment restoration applies when probing the bundled copy.
    seen_envs.clear()

    def probe_bundled_only(argv, **kwargs):
        probe_env = kwargs["env"]
        seen_envs[os.path.realpath(argv[0])] = probe_env
        if os.path.realpath(argv[0]) == os.path.realpath(bundled_git) and "LD_LIBRARY_PATH" not in probe_env:
            kwargs["stdout"].write(b"git version 2.56.1\n")
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=1)

    env = {"PATH": str(sys_git.parent), "LD_LIBRARY_PATH": "/bundle/linux-gtk/lib"}
    assert git_selection.select_git(
        bundle, env, platform="linux", run=probe_bundled_only, host=restore_host
    ) == "bundled"
    probed = seen_envs[os.path.realpath(bundled_git)]
    assert probed["PATH"].split(os.pathsep)[0] == str(bundled_git.parent)
    assert probed["GIT_EXEC_PATH"] == str(bundle / "git-standalone" / "libexec" / "git-core")
    assert "LD_LIBRARY_PATH" not in probed and probed["HOST_RESTORED"] == "1"


def test_one_bounded_search_gives_the_next_git_its_share_and_never_takes_the_unjudged_for_a_refusal(
        tmp_path, monkeypatch):
    """One budget covers every candidate and the bundled copy. A Git that hangs spends one share, so the next is still
    judged; a Git the search's time left unjudged raises ``GitUnverified`` with nothing applied, never "missing"."""
    _patch_posix_which(monkeypatch)
    share, budget = git_selection._PROBE_SECONDS, git_selection._SEARCH_SECONDS
    gits = {name: _inert(tmp_path / name / "git") for name in ("hangs", "hangs-too", "fails", "late", "answers")}
    bundle = tmp_path / "bundle"
    bundled = _inert(bundle / "git-standalone" / "bin" / "git")
    behaviour = {os.path.realpath(path): spent_and_answer for path, spent_and_answer in (
        (gits["hangs"], (float("inf"), False)), (gits["hangs-too"], (float("inf"), False)),
        (gits["fails"], (6.0, False)), (gits["late"], (5.0, True)), (gits["answers"], (0.5, True)),
        (bundled, (0.5, True)))}
    now, seen = [100.0], []
    monkeypatch.setattr(git_selection.time, "monotonic", lambda: now[0])

    def timed_probe(argv, **kwargs):
        target, timeout = os.path.realpath(argv[0]), kwargs["timeout"]
        seen.append((target, timeout))
        assert kwargs["stdin"] is subprocess.DEVNULL and kwargs["stderr"] is subprocess.DEVNULL
        assert hasattr(kwargs["stdout"], "fileno")  # TemporaryFile, never PIPE
        spent, answers = behaviour[target]
        if spent > timeout:
            now[0] += timeout
            raise subprocess.TimeoutExpired(argv, timeout)
        now[0] += spent
        if answers:  # Non-UTF-8 bytes after the header must not fail decoding; only the first bytes are read.
            kwargs["stdout"].write(b"git version 2.56.1\n\xff\xfe" + b"z" * 4096)
        return SimpleNamespace(returncode=0 if answers else 1)

    def select(*names):
        seen.clear()
        env = {"PATH": os.pathsep.join(str(gits[name].parent) for name in names),
               git_selection.SELECTED_GIT_ENV: "/stale/git"}
        before = dict(env)
        try:
            return git_selection.select_git(bundle, env, platform="linux", run=timed_probe), env
        except git_selection.GitUnverified as unverified:
            assert env == before  # nothing applied, the earlier selection included
            return unverified, env

    def probed():
        return [target for target, _ in seen]

    # A Git that hangs spends one share; the next is judged within the same search, and so is the bundled copy.
    choice, env = select("hangs", "answers")
    assert choice == "system" and env[git_selection.SELECTED_GIT_ENV] == str(gits["answers"])
    assert seen == [(os.path.realpath(gits["hangs"]), share),
                    (os.path.realpath(gits["answers"]), min(share, budget - share))]
    choice, env = select("hangs")
    assert choice == "bundled" and env[git_selection.SELECTED_GIT_ENV] == str(bundled)
    assert seen[-1] == (os.path.realpath(bundled), min(share, budget - share))
    # A probe the search's own time cut short leaves its Git unverified: no choice, the bundled copy never tried.
    unverified, _ = select("hangs", "fails", "late")
    assert isinstance(unverified, git_selection.GitUnverified) and str(unverified) == str(gits["late"])
    assert seen[-1] == (os.path.realpath(gits["late"]), pytest.approx(budget - share - 6.0))
    assert os.path.realpath(bundled) not in probed()
    # With no time left the next Git is not run at all, and is still not refused; nor is the bundled copy.
    unverified, _ = select("hangs", "hangs-too", "answers")
    assert str(unverified) == str(gits["answers"]) and os.path.realpath(gits["answers"]) not in probed()
    unverified, _ = select("hangs", "hangs-too")
    assert str(unverified) == str(bundled) and os.path.realpath(bundled) not in probed()


def test_a_quit_or_panic_stops_the_search_at_its_first_probe_and_applies_nothing(tmp_path, monkeypatch):
    """The launcher's bootstrap runner (``run_bootstrap_command`` under its shutdown event) refuses a probe once Quit
    or Panic latched, before any process exists: the first probe on macOS is ``xcode-select``, elsewhere a Git's
    ``--version``, and the bundled copy's probe stops the same way."""
    import launcher
    from ouroboros import launcher_bootstrap

    _patch_posix_which(monkeypatch)
    stub = _inert(tmp_path / "usr_bin" / "git")
    first = _inert(tmp_path / "first" / "git")
    _inert(first.with_name("git.exe"))  # what the launcher's own search finds there on Windows
    bundle = tmp_path / "bundle"
    _inert(bundle / "git-standalone" / "bin" / "git")
    _inert(bundle / "git-standalone" / "cmd" / "git.exe")
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(git_selection, "APPLE_GIT_STUB", os.path.realpath(stub))
    latched = threading.Event()
    latched.set()
    calls: list[list[str]] = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        return launcher_bootstrap.run_bootstrap_command(argv, latched, **kwargs)

    for platform, folder in (("darwin", stub.parent), ("linux", first.parent), ("linux", empty)):
        env = {"PATH": str(folder), git_selection.SELECTED_GIT_ENV: "/stale/git"}
        with pytest.raises(InterruptedError):
            git_selection.select_git(bundle, env, platform=platform, run=runner)
        assert env == {"PATH": str(folder), git_selection.SELECTED_GIT_ENV: "/stale/git"}
    assert calls == [["/usr/bin/xcode-select", "-p"], [str(first), "--version"],
                     [str(bundle / "git-standalone" / "bin" / "git"), "--version"]]

    # The launcher hands that runner to its own search: a latched Quit or Panic reaches main (which returns on the
    # latched event before any page) and leaves no choice or prepended PATH behind.
    common_dir = tmp_path / "common-without-git"
    common_dir.mkdir()
    monkeypatch.setattr(launcher, "_shutdown_event", latched)
    monkeypatch.setattr(launcher, "_bundle_dir", lambda: tmp_path / "no-bundle")
    monkeypatch.setattr(launcher, "common_tool_directories", lambda path: [str(common_dir)])
    monkeypatch.setenv("PATH", str(first.parent))
    monkeypatch.delenv(git_selection.SELECTED_GIT_ENV, raising=False)
    with pytest.raises(InterruptedError):
        launcher.check_git()
    assert git_selection.SELECTED_GIT_ENV not in os.environ and os.environ["PATH"] == str(first.parent)


def test_the_apple_stub_hands_over_to_the_developer_git_and_is_never_executed(tmp_path, monkeypatch):
    _patch_posix_which(monkeypatch)
    stub = _inert(tmp_path / "usr_bin" / "git")
    dev_root = tmp_path / "CommandLineTools"
    dev_git = _inert(dev_root / "usr" / "bin" / "git")
    bundle = tmp_path / "bundle"
    bundled_git = _inert(bundle / "git-standalone" / "bin" / "git")
    monkeypatch.setattr(git_selection, "APPLE_GIT_STUB", os.path.realpath(stub))

    calls: list = []
    run = _fake_runner(
        {dev_git: b"git version 2.39.5 (Apple Git-154)\n", bundled_git: b"git version 2.56.1\n"},
        calls=calls,
        xcode_select=SimpleNamespace(returncode=0, stdout=f"{dev_root}\n".encode("utf-8")),
    )
    tools = tmp_path / "tools-without-git"  # e.g. Homebrew's folder, put on PATH ahead of /usr/bin
    tools.mkdir()
    env = {"PATH": os.pathsep.join([str(tools), str(stub.parent)])}
    assert git_selection.select_git(bundle, env, platform="darwin", run=run) == "system"
    assert env[git_selection.SELECTED_GIT_ENV] == str(dev_git)
    # PATH stays: the stub forwards a bare git to that developer Git, and cc/clang stay where they were.
    assert env["PATH"].split(os.pathsep) == [str(tools), str(stub.parent)]
    assert all(os.path.realpath(argv[0]) != os.path.realpath(stub) for argv, _ in calls)
    asked = [kwargs for argv, kwargs in calls if argv == ["/usr/bin/xcode-select", "-p"]]
    assert len(asked) == 1 and asked[0]["timeout"] == git_selection._PROBE_SECONDS  # a share of the one search

    # Without Command Line Tools installed, xcode-select exits non-zero; the stub is still never run.
    calls.clear()
    no_clt = _fake_runner(
        {bundled_git: b"git version 2.56.1\n"},
        calls=calls,
        xcode_select=SimpleNamespace(returncode=2, stdout=b""),
    )
    env = {"PATH": str(stub.parent)}
    assert git_selection.select_git(bundle, env, platform="darwin", run=no_clt) == "bundled"
    assert env[git_selection.SELECTED_GIT_ENV] == str(bundled_git)
    assert all(os.path.realpath(argv[0]) != os.path.realpath(stub) for argv, _ in calls)
    # Without them, another Git after the stub goes just before it, so no bare git ever reaches the stub.
    later = _inert(tmp_path / "later" / "git")
    env = {"PATH": os.pathsep.join([str(stub.parent), str(later.parent)])}
    assert git_selection.select_git(bundle, env, platform="darwin", run=_fake_runner(
        {later: b"git version 2.47.0\n"}, xcode_select=SimpleNamespace(returncode=2, stdout=b""))) == "system"
    assert env["PATH"].split(os.pathsep) == [str(later.parent), str(stub.parent), str(later.parent)]

    # xcode-select whose share the search's own time cut short leaves the stub's Git unverified, not absent:
    # the bundled copy is not taken over it and nothing is applied.
    def slow_xcode_select(argv, kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(git_selection, "_SEARCH_SECONDS", git_selection._PROBE_SECONDS / 2)
    env = {"PATH": str(stub.parent)}
    late = _fake_runner({bundled_git: b"git version 2.56.1\n"}, xcode_select=slow_xcode_select)
    with pytest.raises(git_selection.GitUnverified, match=re.escape(os.path.realpath(stub))):
        git_selection.select_git(bundle, env, platform="darwin", run=late)
    assert env == {"PATH": str(stub.parent)}


def test_the_choice_survives_worker_path_setup_and_bootstrap_dispatch(
        tmp_path, monkeypatch, _restore_git_selection_env):
    """The launcher selects Git on its normal augmented PATH (common tool folders ahead of inherited entries, skipping
    broken ones). Adding a folder that appears later, in any process inheriting that choice (the once-only flag is each
    process's own), never puts another Git ahead of it: the files decide there, no Git is run again."""
    import launcher
    from ouroboros import launcher_bootstrap, platform_layer

    _patch_posix_which(monkeypatch)
    exe = "git.exe" if sys.platform == "win32" else "git"
    bundle = tmp_path / "bundle"
    if sys.platform == "win32":
        bundled_git = _inert(bundle / "git-standalone" / "cmd" / "git.exe")
        common = [tmp_path / "ProgramFiles" / "Git" / "cmd", tmp_path / "LocalAppData" / "Programs" / "Git" / "cmd",
                  tmp_path / "Profile" / ".cargo" / "bin"]
        for name, value in (("ProgramFiles", "ProgramFiles"), ("LOCALAPPDATA", "LocalAppData"), ("APPDATA", "AppData"),
                            ("USERPROFILE", "Profile")):
            monkeypatch.setenv(name, str(tmp_path / value))
    else:
        bundled_git = _inert(bundle / "git-standalone" / "bin" / "git")
        home = tmp_path / "home"
        common = [home / ".local" / "bin", home / ".cargo" / "bin", home / "go" / "bin"]
        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: home))
    cannot_run, other_working = _inert(common[0] / exe), _inert(common[1] / exe)
    person_git = _inert(tmp_path / "person" / exe)
    calls: list = []
    monkeypatch.setattr(launcher, "_hidden_run", _fake_runner({
        cannot_run: 127, other_working: b"git version 2.48.0\n", person_git: b"git version 2.40.0\n",
        bundled_git: b"git version 2.56.1\n"}, calls=calls))
    monkeypatch.setenv("PATH", str(person_git.parent))
    for name in (git_selection.SELECTED_GIT_ENV, "GIT_EXEC_PATH", "GIT_TEMPLATE_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(launcher, "_bundle_dir", lambda: bundle)
    assert launcher.check_git() is True
    assert os.environ[git_selection.SELECTED_GIT_ENV] == str(other_working)

    def first_git():
        return next(git for folder in os.environ["PATH"].split(os.pathsep)
                    if (git := git_selection._git_in(folder, sys.platform)))

    assert first_git() == str(other_working)
    probes = len(calls)
    for appeared in (None, common[2]):  # a worker, then a descendant that finds a folder made since
        if appeared is not None:
            _inert(appeared / exe)
        monkeypatch.setattr(platform_layer, "_PATH_BOOTSTRAPPED", False)
        added = platform_layer.bootstrap_process_path()
        if appeared is not None:
            assert str(appeared) in added
        assert first_git() == str(other_working)
    assert len(calls) == probes  # adding folders ran no Git

    # Bootstrap dispatch (checkout_sha, launcher_sources_changed, _run_git) uses OUROBOROS_GIT across chdir.
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    chosen = os.environ[git_selection.SELECTED_GIT_ENV]
    assert git_selection.selected_git() == chosen

    seen_sub: list[list[str]] = []

    def fake_sub_run(argv, **kwargs):
        seen_sub.append(list(argv))
        if argv[1:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="b" * 40 + "\n")
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(launcher_bootstrap.subprocess, "run", fake_sub_run)
    repo = tmp_path / "repo"
    repo.mkdir()
    assert launcher_bootstrap.checkout_sha(repo) == "b" * 40
    assert launcher_bootstrap.launcher_sources_changed(repo, loaded_sha="a" * 40) is True
    assert seen_sub[0] == [chosen, "rev-parse", "HEAD"]
    assert seen_sub[2][:4] == [chosen, "diff", "--quiet", "a" * 40]

    seen_hidden: list[list[str]] = []
    context = SimpleNamespace(hidden_run=lambda argv, **kw: seen_hidden.append(list(argv)) or SimpleNamespace(returncode=0))
    launcher_bootstrap._run_git(context, ["git", "status", "--short"], cwd=repo)
    launcher_bootstrap._run_git(context, ["custom-tool", "arg"], cwd=repo)
    assert seen_hidden == [[chosen, "status", "--short"], ["custom-tool", "arg"]]

    monkeypatch.delenv(git_selection.SELECTED_GIT_ENV, raising=False)
    assert git_selection.selected_git() == "git"


def test_finder_gui_launch_preserves_homebrew_non_git_tools_and_agrees_on_git(
        tmp_path, monkeypatch, _restore_git_selection_env):
    """R4 regression guard (noexec, inert fixtures): a macOS Finder launch inherits ``/usr/bin`` before common tool
    folders. Choosing Git on the normal augmented PATH keeps Homebrew's non-Git tools (``python3``, ``openssl``) ahead
    of ``/usr/bin`` both with valid Git and with bundled fallback, while terminal CLI keeps its inherited PATH first."""
    import launcher
    from ouroboros import packaged_cli, platform_layer

    _patch_posix_which(monkeypatch)
    usr_bin = tmp_path / "usr_bin"
    stub = _inert(usr_bin / "git")
    _inert(usr_bin / "python3")
    _inert(usr_bin / "openssl")
    dev_root = tmp_path / "CommandLineTools"
    dev_git = _inert(dev_root / "usr" / "bin" / "git")
    homebrew_with_git = tmp_path / "homebrew_with_git"
    brew_git = _inert(homebrew_with_git / "git")
    _inert(homebrew_with_git / "python3")
    _inert(homebrew_with_git / "openssl")
    homebrew_no_git = tmp_path / "homebrew_no_git"
    _inert(homebrew_no_git / "python3")
    _inert(homebrew_no_git / "openssl")
    bundle = tmp_path / "bundle"
    bundled_git = _inert(bundle / "git-standalone" / "bin" / "git")

    monkeypatch.setattr(git_selection, "APPLE_GIT_STUB", os.path.realpath(stub))
    monkeypatch.setattr(launcher, "_bundle_dir", lambda: bundle)
    monkeypatch.setattr(platform_layer, "PATH_SEP", os.pathsep)

    def first_in_path(name: str, path_value: str) -> str:
        return next(os.path.join(folder, name) for folder in path_value.split(os.pathsep)
                    if folder and os.path.isfile(os.path.join(folder, name)))

    def stub_common(folders: list[str]):
        def _common(search_path: str) -> list[str]:
            present = {part for part in search_path.split(os.pathsep) if part}
            return [folder for folder in folders if folder not in present]

        monkeypatch.setattr(launcher, "common_tool_directories", _common)
        monkeypatch.setattr(platform_layer, "common_tool_directories", _common)

    def run_launcher_and_worker(common_dirs: list[str], runner):
        stub_common(common_dirs)
        monkeypatch.setattr(launcher, "_hidden_run", runner)
        monkeypatch.setenv("PATH", str(usr_bin))
        for key in (git_selection.SELECTED_GIT_ENV, "GIT_EXEC_PATH", "GIT_TEMPLATE_DIR"):
            monkeypatch.delenv(key, raising=False)
        real_select = git_selection.select_git
        monkeypatch.setattr(launcher, "select_git", lambda *a, **kw: real_select(*a, platform="darwin", **kw))
        assert launcher.check_git() is True
        monkeypatch.setattr(platform_layer, "_PATH_BOOTSTRAPPED", False)
        platform_layer.bootstrap_process_path()

    # 1. Valid Homebrew Git + non-Git tools + Apple CLT on GUI PATH: Homebrew wins for Git and non-Git tools.
    calls: list = []
    clt_ok = SimpleNamespace(returncode=0, stdout=f"{dev_root}\n".encode("utf-8"))
    run_launcher_and_worker([str(homebrew_with_git)], _fake_runner(
        {brew_git: b"git version 2.47.0\n", dev_git: b"git version 2.39.5\n", bundled_git: b"git version 2.56.1\n"},
        calls=calls, xcode_select=clt_ok))
    assert os.environ[git_selection.SELECTED_GIT_ENV] == str(brew_git)
    assert os.environ["PATH"].split(os.pathsep) == [str(homebrew_with_git), str(usr_bin)]
    assert first_in_path("git", os.environ["PATH"]) == str(brew_git) == git_selection.selected_git()
    assert first_in_path("python3", os.environ["PATH"]) == str(homebrew_with_git / "python3")
    assert first_in_path("openssl", os.environ["PATH"]) == str(homebrew_with_git / "openssl")
    assert all(os.path.realpath(argv[0]) != os.path.realpath(stub) for argv, _ in calls)

    # 2. Homebrew has non-Git tools only + Apple CLT on GUI PATH: CLT Git chosen via stub; Homebrew stays ahead.
    calls.clear()
    run_launcher_and_worker([str(homebrew_no_git)], _fake_runner(
        {dev_git: b"git version 2.39.5\n", bundled_git: b"git version 2.56.1\n"},
        calls=calls, xcode_select=clt_ok))
    assert os.environ[git_selection.SELECTED_GIT_ENV] == str(dev_git)
    assert os.environ["PATH"].split(os.pathsep) == [str(homebrew_no_git), str(usr_bin)]
    assert first_in_path("git", os.environ["PATH"]) == str(stub)
    assert first_in_path("python3", os.environ["PATH"]) == str(homebrew_no_git / "python3")
    assert first_in_path("openssl", os.environ["PATH"]) == str(homebrew_no_git / "openssl")
    assert all(os.path.realpath(argv[0]) != os.path.realpath(stub) for argv, _ in calls)

    # 3. No working system Git (unconfigured Apple stub) + packaged bundled Git: bundled Git first, Homebrew ahead of /usr/bin.
    calls.clear()
    no_clt = SimpleNamespace(returncode=2, stdout=b"")
    run_launcher_and_worker([str(homebrew_no_git)], _fake_runner(
        {bundled_git: b"git version 2.56.1\n"}, calls=calls, xcode_select=no_clt))
    assert os.environ[git_selection.SELECTED_GIT_ENV] == str(bundled_git)
    assert os.environ["PATH"].split(os.pathsep) == [str(bundled_git.parent), str(homebrew_no_git), str(usr_bin)]
    assert first_in_path("git", os.environ["PATH"]) == str(bundled_git) == git_selection.selected_git()
    assert first_in_path("python3", os.environ["PATH"]) == str(homebrew_no_git / "python3")
    assert first_in_path("openssl", os.environ["PATH"]) == str(homebrew_no_git / "openssl")
    assert all(os.path.realpath(argv[0]) != os.path.realpath(stub) for argv, _ in calls)

    # 4. Terminal CLI (packaged_cli) keeps its inherited PATH ahead of common tool folders.
    terminal_bin = tmp_path / "terminal_bin"
    terminal_git = _inert(terminal_bin / "git")
    _inert(terminal_bin / "python3")
    monkeypatch.setenv("PATH", str(terminal_bin))
    for key in (git_selection.SELECTED_GIT_ENV, "GIT_EXEC_PATH", "GIT_TEMPLATE_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(packaged_cli, "common_tool_directories", lambda path: [str(homebrew_with_git)])
    monkeypatch.setattr(packaged_cli, "bootstrap_repo", lambda ctx: None)
    cli_run = _fake_runner({terminal_git: b"git version 2.44.0\n", brew_git: b"git version 2.47.0\n"})
    monkeypatch.setattr(packaged_cli, "select_git", lambda *a, **kw: git_selection.select_git(
        *a, platform="darwin", run=cli_run, **kw))
    runtime = packaged_cli.PackagedRuntime(
        bundle_root=bundle, embedded_python=tmp_path / "python", app_root=tmp_path / "app",
        repo_dir=tmp_path / "app" / "repo", data_dir=tmp_path / "app" / "data", app_version="0.0.0",
    )
    packaged_cli._bootstrap_runtime(runtime)
    assert os.environ[git_selection.SELECTED_GIT_ENV] == str(terminal_git)
    assert os.environ["PATH"] == str(terminal_bin)
    assert first_in_path("python3", os.environ["PATH"]) == str(terminal_bin / "python3")


def test_terminal_cli_without_git_on_path_appends_extra_git_without_shadowing_terminal_tools(
        tmp_path, monkeypatch, _restore_git_selection_env):
    """R5 regression guard (noexec, inert fixtures): when a terminal CLI's inherited PATH holds no Git and
    select_git finds a working system Git in extra_dirs, that folder is appended instead of prepended so
    terminal non-Git tools keep priority while bare ``git`` still reaches the chosen Git."""
    from ouroboros import packaged_cli, platform_layer

    _patch_posix_which(monkeypatch)
    exe = "git.exe" if sys.platform == "win32" else "git"
    venv_bin, custom_tls = tmp_path / "venv" / "bin", tmp_path / "custom-tls" / "bin"
    homebrew, later_common = tmp_path / "homebrew" / "bin", tmp_path / "later-common" / "bin"
    broken_dir, bundle = tmp_path / "broken" / "bin", tmp_path / "bundle"
    _inert(venv_bin / "python3")
    _inert(custom_tls / "openssl")
    brew_git = _inert(homebrew / exe)
    _inert(homebrew / "python3")
    _inert(homebrew / "openssl")
    _inert(later_common / exe)
    broken_git = _inert(broken_dir / exe)
    _inert(bundle / "git-standalone" / ("cmd" if sys.platform == "win32" else "bin") / exe)

    def first_in_path(name: str, path_value: str) -> str:
        return next(os.path.join(folder, name) for folder in path_value.split(os.pathsep)
                    if folder and os.path.isfile(os.path.join(folder, name)))

    cli_run = _fake_runner({brew_git: b"git version 2.47.0\n", broken_git: 127})
    monkeypatch.setenv("PATH", os.pathsep.join([str(venv_bin), str(custom_tls)]))
    for key in (git_selection.SELECTED_GIT_ENV, "GIT_EXEC_PATH", "GIT_TEMPLATE_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(packaged_cli, "common_tool_directories", lambda path: [str(homebrew)])
    monkeypatch.setattr(packaged_cli, "bootstrap_repo", lambda ctx: None)
    monkeypatch.setattr(packaged_cli, "select_git", lambda *a, **kw: git_selection.select_git(
        *a, platform=sys.platform, run=cli_run, **kw))
    runtime = packaged_cli.PackagedRuntime(
        bundle_root=bundle, embedded_python=tmp_path / "python", app_root=tmp_path / "app",
        repo_dir=tmp_path / "app" / "repo", data_dir=tmp_path / "app" / "data", app_version="0.0.0",
    )
    packaged_cli._bootstrap_runtime(runtime)
    assert os.environ[git_selection.SELECTED_GIT_ENV] == str(brew_git)
    assert os.environ["PATH"].split(os.pathsep) == [str(venv_bin), str(custom_tls), str(homebrew)]
    assert first_in_path("python3", os.environ["PATH"]) == str(venv_bin / "python3")
    assert first_in_path("openssl", os.environ["PATH"]) == str(custom_tls / "openssl")
    assert first_in_path(exe, os.environ["PATH"]) == str(brew_git) == git_selection.selected_git()

    # Later common-tool bootstrap keeps bare git on the appended folder without shadowing terminal tools.
    monkeypatch.setattr(platform_layer, "PATH_SEP", os.pathsep)
    monkeypatch.setattr(platform_layer, "common_tool_directories", lambda path: [str(later_common)])
    monkeypatch.setattr(platform_layer, "_PATH_BOOTSTRAPPED", False)
    platform_layer.bootstrap_process_path()
    assert first_in_path(exe, os.environ["PATH"]) == str(brew_git)
    assert first_in_path("python3", os.environ["PATH"]) == str(venv_bin / "python3")

    # Empty PATH yields only the chosen folder; a broken Git after a non-Git dir inserts right before it.
    empty_env = {"PATH": ""}
    assert git_selection.select_git(bundle, empty_env, extra_dirs=[str(homebrew)],
                                    platform=sys.platform, run=cli_run) == "system"
    assert empty_env["PATH"] == str(homebrew)

    broken_env = {"PATH": os.pathsep.join([str(venv_bin), str(broken_dir)])}
    assert git_selection.select_git(bundle, broken_env, extra_dirs=[str(homebrew)],
                                    platform=sys.platform, run=cli_run) == "system"
    assert broken_env["PATH"].split(os.pathsep) == [str(venv_bin), str(homebrew), str(broken_dir)]


def test_platform_layer_retains_node_runtime_compat_reexports():
    from ouroboros import node_runtime, platform_layer

    for name in (
        "NodeRuntimeHealth",
        "node_runtime_health",
        "probe_node_version",
        "prepend_skill_node_emergency_path",
        "select_skill_node_runtime",
        "skill_manifest_owns_path",
        "skill_node_argv",
        "skill_node_emergency_path_dir",
    ):
        assert getattr(platform_layer, name) is getattr(node_runtime, name)


def test_common_tool_folders_keep_their_old_order_where_no_git_was_chosen(tmp_path, monkeypatch):
    """Only a process tree with a chosen Git (``OUROBOROS_GIT``) puts a common tool folder holding another Git after the
    Git its PATH reaches; anywhere else (a source run, a test's own environment) every folder goes ahead as before."""
    from ouroboros import platform_layer

    _patch_posix_which(monkeypatch)
    exe = "git.exe" if sys.platform == "win32" else "git"
    person = _inert(tmp_path / "person" / exe)
    if sys.platform == "win32":
        monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
        common = tmp_path / "AppData" / "npm"
    else:
        home = tmp_path / "home"
        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: home))
        common = home / ".npm-global" / "bin"
    _inert(common / exe)
    for chosen in ("", str(person)):
        if chosen:
            monkeypatch.setenv(git_selection.SELECTED_GIT_ENV, chosen)
        else:
            monkeypatch.delenv(git_selection.SELECTED_GIT_ENV, raising=False)
        monkeypatch.setenv("PATH", str(person.parent))
        monkeypatch.setattr(platform_layer, "_PATH_BOOTSTRAPPED", False)
        added = platform_layer.bootstrap_process_path()
        parts = os.environ["PATH"].split(os.pathsep)
        assert str(common) in added
        if chosen:
            assert parts.index(str(person.parent)) < parts.index(str(common))
        else:
            assert parts == [*added, str(person.parent)]


@pytest.fixture(params=["selected", "unset"])
def update_git_executable(request, monkeypatch):
    """An inert argv value; all update calls below replace the process boundary."""
    key = git_selection.SELECTED_GIT_ENV
    if request.param == "selected":
        executable = r"C:\Example Git\cmd\git.exe"
        monkeypatch.setenv(key, executable)
        return executable
    monkeypatch.delenv(key, raising=False)
    return "git"


def test_update_constitution_uses_selected_git(monkeypatch, update_git_executable):
    from supervisor import update_source

    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=f"100644 blob {'a' * 40} 5\tBIBLE.md\n"))
    monkeypatch.setattr(update_source, "subprocess", SimpleNamespace(run=run))
    assert update_source.official_ref_has_constitution("managed/target", repo_dir="test repo")
    run.assert_called_once_with(
        [update_git_executable, "ls-tree", "-l", "managed/target", "--", "BIBLE.md"],
        cwd="test repo", capture_output=True, text=True, check=False,
    )


def test_update_carrier_runner_uses_selected_git(monkeypatch, update_git_executable):
    from ouroboros.platform_layer import subprocess_new_group_kwargs
    from supervisor import update_carriers

    output = b"7.12.0\r\n"
    proc = SimpleNamespace(returncode=0, communicate=Mock(return_value=(output, b"")))
    popen = Mock(return_value=proc)
    monkeypatch.setattr(update_carriers, "subprocess", SimpleNamespace(
        Popen=popen, PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired,
    ))
    assert update_carriers._run_git("test repo", ["show", ":2:VERSION"]) == (0, output, b"")
    popen.assert_called_once_with(
        [update_git_executable, "-C", "test repo", "show", ":2:VERSION"],
        stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **subprocess_new_group_kwargs(),
    )
    proc.communicate.assert_called_once_with(input=None, timeout=300.0)


def test_update_store_compatibility_uses_selected_git(monkeypatch, update_git_executable):
    from ouroboros import store_switch
    from ouroboros.dialogue_store import THREAD_FORMAT

    seen = []

    def run(argv, *, cwd, capture_output, env):
        assert cwd == "test repo" and capture_output is True
        assert env["LC_ALL"] == env["LANG"] == "C"
        seen.append(argv)
        return SimpleNamespace(returncode=0, stdout=THREAD_FORMAT.encode("ascii"))

    monkeypatch.setattr(store_switch, "subprocess", SimpleNamespace(run=run))
    sha = "b" * 40
    assert store_switch.switch_needs("test repo", sha) == dict.fromkeys(
        ("facts", "money", "handles", "dialogue", "threads"), False,
    )
    files = ("dialogue_store.py", "store.py", "usage_store.py", "evidence_packs.py")
    assert sorted(seen) == sorted(
        [update_git_executable, "cat-file", "-p", f"{sha}:ouroboros/{name}"] for name in files
    )


def test_update_store_head_uses_selected_git(monkeypatch, update_git_executable):
    from ouroboros import store_switch

    sha = "c" * 40

    def run(argv, *, cwd, capture_output, text, env):
        assert argv == [update_git_executable, "rev-parse", "--verify", "HEAD^{commit}"]
        assert cwd == "test repo" and capture_output is True and text is True
        assert env["LC_ALL"] == env["LANG"] == "C"
        return SimpleNamespace(returncode=0, stdout=f"{sha}\n")

    monkeypatch.setattr(store_switch, "subprocess", SimpleNamespace(run=run))
    assert store_switch.rev_parse("test repo", "HEAD") == sha


def test_windows_lookup_never_takes_a_git_exe_from_the_current_folder(tmp_path, monkeypatch):
    cwd_dir = tmp_path / "working-dir"
    cwd_git = _inert(cwd_dir / "git.exe")
    empty_dir = tmp_path / "empty-dir"
    empty_dir.mkdir()
    bundle = tmp_path / "bundle"
    mingit = _inert(bundle / "git-standalone" / "cmd" / "git.exe")

    monkeypatch.chdir(cwd_dir)
    assert git_selection._git_in("", "win32") is None
    assert git_selection._git_in(str(empty_dir), "win32") is None

    calls: list = []
    run = _fake_runner({cwd_git: b"git version 9.9.9\n", mingit: b"git version 2.56.1.windows.1\n"}, calls=calls)
    env = {"PATH": os.pathsep.join([str(empty_dir), ""])}
    assert git_selection.select_git(bundle, env, platform="win32", run=run) == "bundled"
    assert env[git_selection.SELECTED_GIT_ENV] == str(mingit)
    assert all(os.path.realpath(argv[0]) != os.path.realpath(cwd_git) for argv, _ in calls)

    # After changing cwd to another folder containing a git.exe placeholder, the selected absolute path stands.
    other_dir = tmp_path / "second-cwd"
    _inert(other_dir / "git.exe")
    monkeypatch.chdir(other_dir)
    assert git_selection.selected_git(env) == str(mingit)


def test_a_windows_program_files_git_is_left_alone_and_without_it_mingit_goes_first(tmp_path):
    program_git = _inert(tmp_path / "Program Files" / "Git" / "cmd" / "git.exe")
    bundle = tmp_path / "bundle"
    mingit = _inert(bundle / "git-standalone" / "cmd" / "git.exe")
    other = tmp_path / "Windows" / "System32"
    other.mkdir(parents=True)
    run = _fake_runner({program_git: b"git version 2.47.1.windows.1\n", mingit: b"git version 2.56.1.windows.1\n"})

    # System Git already on PATH: PATH stays untouched and no POSIX relocation vars are added.
    original_path = os.pathsep.join([str(other), str(program_git.parent)])
    env = {"PATH": original_path}
    assert git_selection.select_git(bundle, env, platform="win32", run=run) == "system"
    assert env == {"PATH": original_path, git_selection.SELECTED_GIT_ENV: str(program_git)}

    # System Git only in extra_dirs (common_tool_directories): appended to PATH so inherited entries keep priority.
    env = {"PATH": str(other)}
    assert git_selection.select_git(bundle, env, extra_dirs=[str(program_git.parent)],
                                    platform="win32", run=run) == "system"
    assert env["PATH"].split(os.pathsep) == [str(other), str(program_git.parent)]
    assert env[git_selection.SELECTED_GIT_ENV] == str(program_git)
    assert "GIT_EXEC_PATH" not in env and "GIT_TEMPLATE_DIR" not in env

    # Without a working system Git, MinGit goes first on PATH and sets no POSIX relocation vars.
    broken_run = _fake_runner({program_git: 1, mingit: b"git version 2.56.1.windows.1\n"})
    env = {"PATH": str(other)}
    assert git_selection.select_git(bundle, env, extra_dirs=[str(program_git.parent)],
                                    platform="win32", run=broken_run) == "bundled"
    assert env["PATH"].split(os.pathsep) == [str(mingit.parent), str(other)]
    assert env[git_selection.SELECTED_GIT_ENV] == str(mingit)
    assert "GIT_EXEC_PATH" not in env and "GIT_TEMPLATE_DIR" not in env


def test_windows_relative_path_entry_selects_absolute_system_git_across_cwd_change(tmp_path, monkeypatch):
    initial_cwd = tmp_path / "initial-cwd"
    rel_dir = pathlib.Path("tools") / "git" / "cmd"
    rel_git = _inert(initial_cwd / rel_dir / "git.exe")
    bundle = tmp_path / "bundle"
    mingit = _inert(bundle / "git-standalone" / "cmd" / "git.exe")
    empty_cwd = tmp_path / "empty-cwd"
    empty_cwd.mkdir()

    monkeypatch.chdir(initial_cwd)
    run = _fake_runner({rel_git: b"git version 2.47.1.windows.1\n", mingit: b"git version 2.56.1.windows.1\n"})
    env = {"PATH": str(rel_dir)}

    assert git_selection.select_git(bundle, env, platform="win32", run=run) == "system"
    chosen = git_selection.selected_git(env)
    assert pathlib.Path(chosen).is_absolute()
    assert os.path.realpath(chosen) == os.path.realpath(rel_git)

    monkeypatch.chdir(empty_cwd)
    assert git_selection.selected_git(env) == chosen
    assert pathlib.Path(chosen).is_file()


def test_packaged_paths_resolve_under_the_bundle_and_a_parents_selection_is_recognized(tmp_path, monkeypatch):
    _patch_posix_which(monkeypatch)
    bundle = tmp_path / "bundle"
    assert git_selection.bundled_git_root(None, "linux") is None
    assert git_selection.bundled_git_root(bundle, "linux") is None
    assert git_selection.bundled_git_root(bundle, "win32") is None

    bundled_posix = _inert(bundle / "git-standalone" / "bin" / "git")
    bundled_win = _inert(bundle / "git-standalone" / "cmd" / "git.exe")
    assert git_selection.bundled_git_root(bundle, "linux") == bundle / "git-standalone"
    assert git_selection.bundled_git_root(bundle, "win32") == bundle / "git-standalone"

    # A parent process previously fell back to our bundled Git; later a system Git exists on PATH.
    # Re-selecting strips our own bundled dir and relocation vars so the person's Git wins.
    sys_git = _inert(tmp_path / "sys" / "bin" / "git")
    run = _fake_runner({sys_git: b"git version 2.49.0\n", bundled_posix: b"git version 2.56.1\n"})
    env = {
        "PATH": os.pathsep.join([str(bundled_posix.parent), str(sys_git.parent)]),
        "GIT_EXEC_PATH": str(bundle / "git-standalone" / "libexec" / "git-core"),
        "GIT_TEMPLATE_DIR": str(bundle / "git-standalone" / "share" / "git-core" / "templates"),
        git_selection.SELECTED_GIT_ENV: str(bundled_posix),
    }
    assert git_selection.select_git(bundle, env, platform="linux", run=run) == "system"
    assert env == {"PATH": str(sys_git.parent), git_selection.SELECTED_GIT_ENV: str(sys_git)}

    # Relocation vars pointing outside our bundle belong to the owner and are never removed.
    owner_env = {
        "PATH": os.pathsep.join([str(bundled_posix.parent), str(sys_git.parent)]),
        "GIT_EXEC_PATH": "/owner/custom-git-exec",
        "GIT_TEMPLATE_DIR": "/owner/custom-templates",
    }
    assert git_selection.select_git(bundle, owner_env, platform="linux", run=run) == "system"
    assert owner_env["GIT_EXEC_PATH"] == "/owner/custom-git-exec"
    assert owner_env["GIT_TEMPLATE_DIR"] == "/owner/custom-templates"
    assert bundled_win.is_file()


def test_one_decision_gates_both_entries_before_anything_runs_git(tmp_path, monkeypatch, capsys):
    import launcher
    from ouroboros import desktop_media, packaged_cli

    answer = {"choice": "bundled"}

    def decided():
        if isinstance(answer["choice"], BaseException):
            raise answer["choice"]
        return answer["choice"]

    # 1. launcher.check_git chooses with the bundle dir, the common tool dirs prepended to PATH, its bootstrap runner
    #    (which Quit and Panic stop) and the Linux host hook; a Git not judged in time goes to main, never as missing.
    seen: list = []

    def fake_select(bundle_dir, *args, **kwargs):
        seen.append((bundle_dir, kwargs))
        return decided()

    monkeypatch.setattr(launcher, "select_git", fake_select)
    monkeypatch.setattr(launcher, "_bundle_dir", lambda: tmp_path / "bundle")
    monkeypatch.setattr(launcher, "common_tool_directories", lambda path: ["/common/bin"])
    assert launcher.check_git() is True
    assert seen[-1][0] == tmp_path / "bundle" and seen[-1][1]["prepend_dirs"] == ["/common/bin"]
    assert "extra_dirs" not in seen[-1][1]
    assert seen[-1][1]["run"] is launcher._hidden_run
    assert seen[-1][1]["host"] is (desktop_media.host_environment if launcher.IS_LINUX else None)
    answer["choice"] = "system"
    assert launcher.check_git() is True
    answer["choice"] = "missing"
    assert launcher.check_git() is False
    answer["choice"] = git_selection.GitUnverified("/slow/git")
    with pytest.raises(git_selection.GitUnverified, match="/slow/git"):
        launcher.check_git()

    # 2. launcher.main gates on check_git() BEFORE remember_loaded_checkout and bootstrap_repo.
    order: list[str] = []

    class _HaltAtBootstrap(Exception):
        pass

    def select_in_main(*args, **kwargs):
        order.append("select_git")
        if isinstance(answer["choice"], InterruptedError):
            launcher._shutdown_event.set()  # the bootstrap runner stops only under a latched Quit or Panic
        return decided()

    for name in ("_external_ui", "_external_host_update", "_external_seed_bundle", "_launch_argv", "_launch_intent",
                 "_headless", "_runtime_venv", "EMBEDDED_PYTHON"):
        monkeypatch.setattr(launcher, name, getattr(launcher, name))
    monkeypatch.setattr(atexit, "register", lambda *a: None)
    monkeypatch.setattr(atexit, "unregister", lambda *a: None)
    for name, value in {
        "DATA_DIR": tmp_path / "data",
        "REPO_DIR": tmp_path / "repo",
        "_shutdown_event": threading.Event(),
        "acquire_pid_lock": lambda: True,
        "release_pid_lock": lambda: None,
        "stop_tray_before_exit": lambda release: release(),
        "automatic_launch_allowed": lambda *a: True,
        "remember_loaded_checkout": lambda *a, **kw: order.append("remember_loaded_checkout") or "",
        "bootstrap_repo": lambda: (order.append("bootstrap_repo"), (_ for _ in ()).throw(_HaltAtBootstrap()))[0],
        "select_git": select_in_main,
    }.items():
        monkeypatch.setattr(launcher, name, value)

    answer["choice"] = "missing"
    with pytest.raises(SystemExit) as exited:
        launcher.main(["--no-ui"])
    assert exited.value.code == 1
    assert order == ["select_git"]
    assert "Git is required to run Ouroboros." in capsys.readouterr().err

    order.clear()
    answer["choice"] = git_selection.GitUnverified("/slow/git")
    with pytest.raises(SystemExit) as exited:
        launcher.main(["--no-ui"])
    assert exited.value.code == 1 and order == ["select_git"]
    said = capsys.readouterr().err
    assert "Git did not answer in time" in said and "/slow/git" in said and "Git is required" not in said

    order.clear()
    answer["choice"] = InterruptedError("Launcher shutdown stopped bootstrap admission")
    assert launcher.main(["--no-ui"]) is None  # Quit or Panic during the search: no page, no exit code, no bootstrap
    said = capsys.readouterr().err
    assert order == ["select_git"] and "Git is required" not in said
    assert "did not answer" not in said
    launcher._shutdown_event.clear()

    order.clear()
    answer["choice"] = "bundled"
    with pytest.raises(_HaltAtBootstrap):
        launcher.main(["--no-ui"])
    assert order == ["select_git", "remember_loaded_checkout", "bootstrap_repo"]

    # 3. packaged_cli._bootstrap_runtime gates on select_git before bootstrap_repo.
    cli_calls: list[str] = []
    runtime = packaged_cli.PackagedRuntime(
        bundle_root=tmp_path / "bundle",
        embedded_python=tmp_path / "python",
        app_root=tmp_path / "app",
        repo_dir=tmp_path / "app" / "repo",
        data_dir=tmp_path / "app" / "data",
        app_version="0.0.0",
    )
    monkeypatch.setattr(packaged_cli, "common_tool_directories", lambda path: ["/common/bin"])
    monkeypatch.setattr(packaged_cli, "bootstrap_repo", lambda ctx: cli_calls.append("bootstrap_repo"))

    def cli_select(bundle_dir, **kwargs):
        assert bundle_dir == runtime.bundle_root
        assert kwargs["extra_dirs"] == ["/common/bin"]
        assert kwargs["host"] is (desktop_media.host_environment if sys.platform == "linux" else None)
        cli_calls.append("select_git")
        return decided()

    monkeypatch.setattr(packaged_cli, "select_git", cli_select)
    answer["choice"] = "missing"
    with pytest.raises(packaged_cli.PackagedCLIError, match="Git is required before first CLI use"):
        packaged_cli._bootstrap_runtime(runtime)
    assert cli_calls == ["select_git"]

    cli_calls.clear()
    answer["choice"] = git_selection.GitUnverified("/slow/git")
    with pytest.raises(packaged_cli.PackagedCLIError, match="Git did not answer in time: .*/slow/git") as refused:
        packaged_cli._bootstrap_runtime(runtime)
    assert "Git is required" not in str(refused.value) and cli_calls == ["select_git"]

    cli_calls.clear()
    answer["choice"] = "bundled"
    packaged_cli._bootstrap_runtime(runtime)
    assert cli_calls == ["select_git", "bootstrap_repo"]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS xcode-select check")
def test_real_xcode_select_answers_without_a_dialog():
    developer_git = git_selection._apple_developer_git(
        os.environ, subprocess.run, git_selection.time.monotonic() + git_selection._SEARCH_SECONDS)
    if developer_git is not None:
        assert developer_git != git_selection.APPLE_GIT_STUB
        assert os.path.isfile(developer_git)
        assert pathlib.Path(developer_git).parts[-3:] == ("usr", "bin", "git")


def test_real_installed_git_selection_survives_cwd_change(tmp_path, monkeypatch):
    """Ordinary reuse of the host's real installed Git: select_git picks it, stores its absolute
    path in OUROBOROS_GIT, and running that chosen path succeeds after changing cwd."""
    if shutil.which("git") is None:
        pytest.skip("host has no installed git on PATH")
    env = {k: v for k, v in os.environ.items() if k != git_selection.SELECTED_GIT_ENV}
    choice = git_selection.select_git(None, env)
    if choice == "missing":
        pytest.skip("host git did not answer --version")
    assert choice == "system"
    chosen = git_selection.selected_git(env)
    assert pathlib.Path(chosen).is_absolute() and pathlib.Path(chosen).is_file()

    other_cwd = tmp_path / "other-cwd"
    _inert(other_cwd / "git")
    _inert(other_cwd / "git.exe")
    monkeypatch.chdir(other_cwd)
    assert git_selection.selected_git(env) == chosen
    done = subprocess.run([chosen, "--version"], cwd=other_cwd, env=env, capture_output=True, timeout=10)
    assert done.returncode == 0 and done.stdout.startswith(b"git version")
