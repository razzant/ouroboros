"""Launcher unit tests may replace HTTP readers, never contact a real installation.

Block below urlopen, independently of its per-test stub. Record even attempts that
production catches, so swallowing a network error cannot make the test pass.
"""
from contextlib import contextmanager
import http.client
import socket

from _pytest.monkeypatch import MonkeyPatch


@contextmanager
def deny_real_network():
    attempts = []

    def denied(*args, **kwargs):
        attempts.append(repr(args[1:]))
        raise AssertionError("launcher unit test attempted real HTTP/socket I/O")

    with MonkeyPatch.context() as patch:
        patch.setattr(http.client.HTTPConnection, "connect", denied)
        patch.setattr(http.client.HTTPSConnection, "connect", denied)
        patch.setattr(socket.socket, "connect", denied)
        patch.setattr(socket.socket, "connect_ex", denied)
        yield attempts


def launcher_unit_module(name):
    return name.startswith("test_launcher_") or name in {
        "test_macos_source_entry.py", "test_desktop_notifications.py",
    }
