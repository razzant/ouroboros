"""The server's open-file limit rises at start (pool-capacity WP3/WP6): the body follows each live delegated run on its
own transport, and a GUI-launched macOS process starts at 256 files, so hundreds of followed runs would meet EMFILE."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(("platform", "hard", "set_to"), [
    ("linux", 524288, 524288), ("darwin", 2 ** 63 - 1, 10240), ("darwin", 256, None)])
def test_the_servers_open_file_limit_rises_to_its_hard_limit_at_start(monkeypatch, platform, hard, set_to):
    """Linux takes the hard limit; macOS takes ``min(OPEN_MAX, hard)`` (its setrlimit refuses an unlimited soft
    limit for files); a limit already at its ceiling is left alone. ``server.main`` raises it once, at start."""
    import inspect

    import server
    from ouroboros import server_process

    calls = []
    fake = SimpleNamespace(RLIMIT_NOFILE=8, RLIM_INFINITY=2 ** 63 - 1, getrlimit=lambda kind: (256, hard),
                           setrlimit=lambda kind, limits: calls.append(limits))
    monkeypatch.setitem(sys.modules, "resource", fake)
    monkeypatch.setattr(server_process, "sys", SimpleNamespace(platform=platform))
    server_process.raise_open_file_limit()
    assert calls == ([] if set_to is None else [(set_to, hard)]), "already at its ceiling: nothing to raise"
    assert "raise_open_file_limit()" in inspect.getsource(server.main)
