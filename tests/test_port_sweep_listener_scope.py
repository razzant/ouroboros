"""Port sweeps must target the LISTENER, never connected clients.

The POSIX sweep helper (``platform_layer.kill_process_on_port``, kept for the
disclosed companion reserved-port exception) selects processes with ``lsof``. A bare
``tcp:PORT`` selector also matches ESTABLISHED client sockets — on a
browser-mode install the owner's own browser holds one to the UI port, so an
unscoped sweep SIGKILLs the whole browser at panic/startup. These tests pin
the ``-sTCP:LISTEN`` scoping and that only lsof-listed pids are signalled. The
Windows branch reads the socket table through psutil: netstat translates its
state column (no ``LISTENING`` on a Russian Windows) and prints in a code page
the ANSI decoder cannot read, so a text filter never matched there.

Each branch is forced via ``IS_WINDOWS`` monkeypatching so the pins hold on every
CI platform; ``subprocess.run``, psutil and the kill primitives are faked, so no
real process is touched.
"""

import sys
import types

import pytest

import ouroboros.platform_layer as platform_layer


class _FakeCompleted:
    def __init__(self, stdout: str):
        self.stdout = stdout


def test_kill_process_on_port_scopes_lsof_to_listeners(monkeypatch):
    seen: dict = {"argv": None, "killed": []}

    def fake_run(argv, **kwargs):
        seen["argv"] = list(argv)
        return _FakeCompleted("101\n202\n")

    monkeypatch.setattr(platform_layer, "IS_WINDOWS", False)
    monkeypatch.setattr(platform_layer.subprocess, "run", fake_run)
    monkeypatch.setattr(
        platform_layer.os, "kill", lambda pid, sig: seen["killed"].append((pid, sig))
    )

    platform_layer.kill_process_on_port(8765)

    assert seen["argv"] is not None, "POSIX branch must consult lsof"
    assert "-sTCP:LISTEN" in seen["argv"], (
        "unscoped tcp:PORT also matches ESTABLISHED client sockets "
        "(the owner's browser) — the sweep must stay a listener sweep"
    )
    assert "tcp:8765" in seen["argv"]
    assert "-nP" in seen["argv"], "name resolution could eat the 5s timeout"
    assert seen["killed"] == [(101, 9), (202, 9)]


def test_kill_process_on_port_spares_self(monkeypatch):
    killed: list = []

    monkeypatch.setattr(platform_layer, "IS_WINDOWS", False)
    monkeypatch.setattr(
        platform_layer.subprocess,
        "run",
        lambda argv, **kwargs: _FakeCompleted(f"{platform_layer.os.getpid()}\n303\n"),
    )
    monkeypatch.setattr(
        platform_layer.os, "kill", lambda pid, sig: killed.append(pid)
    )

    platform_layer.kill_process_on_port(8765)

    assert killed == [303]


def _connection(port, status, pid, ip="127.0.0.1"):
    laddr = types.SimpleNamespace(ip=ip, port=port) if port else ()  # psutil gives () for an unbound socket
    return types.SimpleNamespace(laddr=laddr, status=status, pid=pid)


def test_windows_sweep_reads_the_socket_table_not_netstat_text(monkeypatch):
    killed: list = []
    table = [
        _connection(8765, "LISTEN", 101),
        _connection(8765, "LISTEN", 101, ip="::1"),  # the same listener's IPv6 row
        _connection(8765, "LISTEN", 505, ip="::"),
        _connection(8765, "ESTABLISHED", 202),  # a client of the port, such as the owner's browser
        _connection(9999, "LISTEN", 303),
        _connection(87650, "LISTEN", 404),  # the old text filter matched this port as a substring
        _connection(8765, "LISTEN", None),  # a listener whose owner psutil may not read
        _connection(0, "LISTEN", 606),
        _connection(8765, "LISTEN", platform_layer.os.getpid()),
    ]
    psutil = types.SimpleNamespace(CONN_LISTEN="LISTEN", net_connections=lambda kind: table)
    monkeypatch.setitem(sys.modules, "psutil", psutil)
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    monkeypatch.setattr(platform_layer, "_hidden_run", lambda *a, **k: pytest.fail("netstat text is not parsed"))
    monkeypatch.setattr(platform_layer, "force_kill_pid", killed.append)

    platform_layer.kill_process_on_port(8765)

    assert killed == [101, 505]
