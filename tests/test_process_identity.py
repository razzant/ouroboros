"""Validated kernel birth identity for death proof, and its macOS reader.

Portable: the token grammar and the one comparison death proof may use, two
validated tokens of one kind that differ. Native macOS (skipped elsewhere): the
``sysctl`` ABI against real processes this module spawns and reaps itself, since
no fixture can prove a struct layout.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import time

import pytest

from ouroboros import platform_darwin_abi, platform_layer
from ouroboros.process_identity import birth_identity, different_birth, process_kernel_birth, same_birth

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_BOOT = "0123456789abcdef0123456789abcdef"
_DARWIN_TOKEN = re.compile(r"darwin-start:[0-9]+\.[0-9]{6}")
# One token per validated kind, and an unequal token of the same kind.
_KINDS = {
    "darwin": ("darwin-start:1793143271.251840", "darwin-start:1793143271.251841"),
    "windows": ("win-filetime:134058912345678901", "win-filetime:134058912345678902"),
    "linux_boot": (f"812.{_BOOT}", f"811.{_BOOT}"),
    "ticks": ("812", "811."),
}


@pytest.mark.parametrize("token,identity", [
    ("darwin-start:1793143271.251840", ("darwin", 1793143271, 251840)),
    ("darwin-start:1793143271.000005", ("darwin", 1793143271, 5)),
    ("win-filetime:134058912345678901", ("windows", 134058912345678901)),
    (f"812.{_BOOT}", ("linux_boot", 812, _BOOT)),
    ("812", ("ticks", 812)),
    ("812.", ("ticks", 812)),
])
def test_each_validated_kind_parses(token, identity):
    assert birth_identity(token) == identity


@pytest.mark.parametrize("text", [
    "", " ", "darwin-start:", "darwin-start:1793143271", "darwin-start:1793143271.",
    "darwin-start:1793143271.25184", "darwin-start:1793143271.2518400",  # five and seven microsecond digits
    "darwin-start:-1793143271.251840", "darwin-start:1793143271.251840\n", " darwin-start:1793143271.251840",
    "DARWIN-START:1793143271.251840", "darwin:1793143271.251840",
    "darwin-start:١٧٩٣.251840",  # non-ASCII digits are no kernel spelling
    "win-filetime:", "win-filetime:-5", "win-filetime:0x1F",
    "812.0123", "812." + _BOOT.upper(), f"812.{_BOOT}0", "-812",
    "Tue Oct 27 14:57:17 2026",  # ps -o lstart= in the C locale
    "Thu Oct  8 17:18:45 2026",
    "вторник, 27 октября 2026 г. 11:57:40",  # the same field under a Russian locale
    None, 812, b"812",
])
def test_unvalidated_text_is_no_identity_and_proves_nothing(text):
    assert birth_identity(text) is None
    for token, other in _KINDS.values():
        assert different_birth(text, token) is False and different_birth(token, text) is False
        assert different_birth(other, token) is True  # the same comparison of validated tokens still proves


@pytest.mark.parametrize("live,recorded,different", [
    ("darwin-start:1793143271.251840", "darwin-start:1793143272.251840", True),   # another second
    ("darwin-start:1793143271.251840", "darwin-start:1793143271.251841", True),   # one microsecond apart
    ("darwin-start:1793143271.251840", "darwin-start:1793143271.251840", False),
    ("win-filetime:1300", "win-filetime:1200", True),
    ("win-filetime:1300", "win-filetime:1300", False),
    (f"812.{_BOOT}", f"811.{_BOOT}", True),
    (f"812.{_BOOT}", "812." + "f" * 32, True),  # the same ticks in another boot
    (f"812.{_BOOT}", f"812.{_BOOT}", False),
    ("812.", "811", True),
    ("812.", "812", False),  # one identity, two spellings
    ("Wed Oct 28 14:38:55 2026", "Wed Oct 28 03:53:55 2026", False),  # one process, two ps zones
])
def test_only_unequal_validated_tokens_of_one_kind_are_a_different_birth(live, recorded, different):
    assert different_birth(live, recorded) is different
    assert different_birth(recorded, live) is different


@pytest.mark.parametrize("first", sorted(_KINDS))
@pytest.mark.parametrize("second", sorted(_KINDS))
def test_two_kinds_never_prove_a_different_birth(first, second):
    live, recorded = _KINDS[first][0], _KINDS[second][1]
    assert different_birth(live, recorded) is (first == second)
    assert different_birth(recorded, live) is (first == second)
    assert same_birth(live, recorded) is same_birth(recorded, live) is (False if first == second else None)


@pytest.mark.parametrize("first,second,same", [
    ("darwin-start:1793143271.251840", "darwin-start:1793143271.251840", True),
    ("darwin-start:1793143271.251840", "darwin-start:1793143271.251841", False),
    (f"812.{_BOOT}", f"812.{_BOOT}", True),  # already validated kinds keep both answers
    (f"812.{_BOOT}", f"811.{_BOOT}", False),
    ("812.", "812", True),
    ("win-filetime:1300", "win-filetime:1300", True),
    ("win-filetime:1300", "win-filetime:1200", False),
    ("Wed Oct 28 14:38:55 2026", "Wed Oct 28 03:53:55 2026", None),  # one process, two ps zones
    ("Wed Oct 28 14:38:55 2026", "Wed Oct 28 14:38:55 2026", None),  # an equal spelling proves nothing either
    ("darwin-start:1793143271.25184", "darwin-start:1793143271.25184", None),  # malformed twins
    ("darwin-start:1793143271.251840", "win-filetime:1300", None),
    ("darwin-start:1793143271.251840", "", None),
    ("darwin-start:1793143271.251840", None, None),
])
def test_same_birth_answers_only_for_two_validated_tokens_of_one_kind(first, second, same):
    assert same_birth(first, second) is same and same_birth(second, first) is same
    assert different_birth(first, second) is (same is False)


def test_macos_mints_only_from_the_kernel_record(monkeypatch):
    def no_fallback(_pid):
        raise AssertionError("macOS must not fall back to process_start_time")

    monkeypatch.setattr(platform_darwin_abi, "IS_DARWIN", True)
    monkeypatch.setattr(platform_layer, "process_start_time", no_fallback)
    for birth, token in (((1793143271, 251840, False), "darwin-start:1793143271.251840"),
                         ((1793143271, 5, False), "darwin-start:1793143271.000005"),
                         ((1793143271, 5, True), "darwin-start:1793143271.000005")):  # a zombie keeps its birth
        monkeypatch.setattr(platform_darwin_abi, "kernel_birth", lambda _pid, birth=birth: birth)
        assert process_kernel_birth(4242) == token and birth_identity(token)[0] == "darwin"
    monkeypatch.setattr(platform_darwin_abi, "kernel_birth", lambda _pid: None)
    assert process_kernel_birth(4242) is None


@pytest.mark.parametrize("minted,expected", [
    (f"812.{_BOOT}", f"812.{_BOOT}"),
    ("812.", "812."),
    ("win-filetime:134058912345678901", "win-filetime:134058912345678901"),
    ("Tue Oct 27 14:57:17 2026", None),  # the ps wall-clock fallback is no kernel birth
    ("", None),
])
def test_other_platforms_pass_only_a_validated_process_start_time(monkeypatch, minted, expected):
    monkeypatch.setattr(platform_darwin_abi, "IS_DARWIN", False)
    monkeypatch.setattr(platform_layer, "process_start_time", lambda _pid: minted)
    assert process_kernel_birth(4242) == expected


def test_an_unreadable_start_time_is_no_birth_and_no_exception(monkeypatch):
    def broken(_pid):
        raise OSError("no /proc")

    monkeypatch.setattr(platform_darwin_abi, "IS_DARWIN", False)
    monkeypatch.setattr(platform_layer, "process_start_time", broken)
    assert process_kernel_birth(4242) is None


def test_the_darwin_reader_answers_nothing_off_macos(monkeypatch):
    assert platform_darwin_abi.IS_DARWIN is platform_layer.IS_MACOS is (sys.platform == "darwin")
    if sys.platform == "darwin":
        assert platform_darwin_abi.kernel_birth(os.getpid()) is not None  # the real platform still reads
    monkeypatch.setattr(platform_darwin_abi, "IS_DARWIN", False)
    assert platform_darwin_abi.kernel_birth(os.getpid()) is None
    assert platform_darwin_abi.kernel_birth(1) is None


def _native(test):
    """Real processes on this Mac (serial lane: real subprocesses)."""
    return pytest.mark.serial(pytest.mark.skipif(sys.platform != "darwin", reason="native macOS sysctl ABI")(test))


def _spawn(argv, **kwargs):
    return subprocess.Popen(argv, stdin=kwargs.pop("stdin", subprocess.DEVNULL),
                            stdout=kwargs.pop("stdout", subprocess.DEVNULL), stderr=subprocess.DEVNULL, **kwargs)


def _stop(child):
    """Kill and reap exactly this child (Popen never signals a pid it already reaped)."""
    for stream in (child.stdin, child.stdout):
        if stream is not None:
            stream.close()
    child.kill()
    child.wait(10)


def _child_env(**extra):
    path = os.pathsep.join(part for part in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if part)
    return {**os.environ, "PYTHONPATH": path, **extra}


_SELF_TOKEN = ("import os, sys; from ouroboros.process_identity import process_kernel_birth; "
               "print(process_kernel_birth(os.getpid()), flush=True); ")


@_native
def test_a_spawned_child_carries_a_stable_kernel_birth_inside_its_spawn_window():
    own = process_kernel_birth(os.getpid())
    assert _DARWIN_TOKEN.fullmatch(own) and birth_identity(own)[0] == "darwin"
    before = time.time()
    child = _spawn(["/bin/sleep", "30"])
    after = time.time()
    try:
        token = process_kernel_birth(child.pid)
        sec, usec = birth_identity(token)[1:]
        # The kernel truncates to whole microseconds; a millisecond also covers float rounding.
        assert before - 0.001 <= sec + usec / 1e6 <= after + 0.001
        assert {process_kernel_birth(child.pid) for _ in range(50)} == {token}
        second = _spawn(["/bin/sleep", "30"])
        try:
            assert different_birth(process_kernel_birth(second.pid), token) is True
        finally:
            _stop(second)
    finally:
        _stop(child)


@_native
def test_a_child_spells_its_own_birth_as_the_parent_reads_it_from_another_zone_and_locale():
    child = _spawn([sys.executable, "-c", _SELF_TOKEN + "sys.stdin.read()"], stdin=subprocess.PIPE,
                   stdout=subprocess.PIPE, text=True, cwd=REPO_ROOT,
                   env=_child_env(TZ="Pacific/Chatham", LC_ALL="ru_RU.UTF-8"))
    try:
        own = child.stdout.readline().strip()
        assert _DARWIN_TOKEN.fullmatch(own), own
        assert process_kernel_birth(child.pid) == own
    finally:
        _stop(child)


@_native
def test_exec_keeps_the_birth_of_the_same_process():
    script = _SELF_TOKEN + "os.execv('/bin/sh', ['/bin/sh', '-c', 'echo exec-done; exec /bin/sleep 30'])"
    child = _spawn([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, cwd=REPO_ROOT,
                   env=_child_env())
    try:
        before_exec = child.stdout.readline().strip()
        assert child.stdout.readline().strip() == "exec-done"
        assert _DARWIN_TOKEN.fullmatch(before_exec) and process_kernel_birth(child.pid) == before_exec
    finally:
        _stop(child)


@_native
def test_another_users_process_is_readable_without_privilege():
    birth = platform_darwin_abi.kernel_birth(1)  # launchd, owned by root
    assert birth is not None and birth[0] > 0 and birth[2] is False
    assert _DARWIN_TOKEN.fullmatch(process_kernel_birth(1))
    # A C int truncates 2**32 + 1 onto pid 1; pid 0 would answer kernel_task: neither is that process.
    for pid in (2**32 + 1, 0, -1):
        assert platform_darwin_abi.kernel_birth(pid) is None and process_kernel_birth(pid) is None


@_native
def test_a_reaped_child_has_no_kernel_birth_or_another_one():
    child = _spawn(["/bin/sleep", "30"])
    try:
        recorded = platform_darwin_abi.kernel_birth(child.pid)
        assert recorded is not None and recorded[2] is False  # live: readable, not a zombie
    finally:
        _stop(child)
    after = platform_darwin_abi.kernel_birth(child.pid)
    assert after is None or after[:2] != recorded[:2]  # absent, or the pid was reused by now
    token = process_kernel_birth(child.pid)
    assert token is None or different_birth(token, f"darwin-start:{recorded[0]}.{recorded[1]:06d}")


@_native
def test_an_exited_unreaped_child_reads_as_a_zombie_with_its_birth():
    child = _spawn(["/bin/sh", "-c", "read _; exit 3"], stdin=subprocess.PIPE)
    try:
        recorded = platform_darwin_abi.kernel_birth(child.pid)
        assert recorded is not None and recorded[2] is False
        child.stdin.close()  # the child exits; nothing reaps it until wait()
        deadline = time.monotonic() + 10
        state = recorded
        while state is not None and not state[2] and time.monotonic() < deadline:
            time.sleep(0.01)
            state = platform_darwin_abi.kernel_birth(child.pid)
        assert state == (recorded[0], recorded[1], True)
    finally:
        _stop(child)
    assert child.returncode == 3
