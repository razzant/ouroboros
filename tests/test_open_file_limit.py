"""What the server sets for its own process at start. The open-file limit rises (pool-capacity WP3/WP6): the body
follows each live delegated run on its own transport, and a GUI-launched macOS process starts at 256 files, so hundreds
of followed runs would meet EMFILE. The interpreter's switch interval shortens and its young-generation collection
threshold rises (#3399)."""

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


def test_the_server_tunes_its_interpreter_at_start(monkeypatch, tmp_path):
    """A read takes the interpreter lock back after each store step only when a computing thread yields, once per
    switch interval: the server asks for 1 ms instead of CPython's 5 ms, in ``main`` before it serves (#3399), and
    collects its young generation after 20,000 new objects instead of 700. The conftest fixture puts the session's own
    values back."""
    import gc

    import server
    from ouroboros import server_entrypoint

    class Serving(Exception):
        pass

    def serving():
        raise Serving()

    sys.setswitchinterval(0.005)
    server_entrypoint.shorten_switch_interval()
    assert sys.getswitchinterval() == pytest.approx(0.001)
    sys.setswitchinterval(0.005)
    gc.set_threshold(700, 7, 3)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.delenv("OUROBOROS_LAUNCH_INTENT", raising=False)
    monkeypatch.setattr(server, "verify_settings_integrity", serving)  # the first step past the process setup
    with pytest.raises(Serving):
        server.main()
    assert sys.getswitchinterval() == pytest.approx(0.001)
    assert gc.get_threshold() == (20_000, 7, 3), "the young generation's threshold rises with it, the older keep theirs"
