"""Append lock budgets preserve authority refusal and live-holder ownership."""
import json
import time
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("required", [False, True])
def test_zero_budget_keeps_required_lock_and_live_holder_semantics(tmp_path, monkeypatch, required):
    from ouroboros import platform_layer as pl
    from ouroboros.utils import append_jsonl, jsonl_append_lock_path

    path = tmp_path / "events.jsonl"
    lock = jsonl_append_lock_path(path)
    fd = pl.acquire_exclusive_file_lock(lock, timeout_sec=1.0)
    assert fd is not None
    original = pl._lock_identity(lock)
    sleeps = []
    clock = SimpleNamespace(**{name: getattr(time, name) for name in dir(time) if not name.startswith("__")})
    clock.sleep = lambda seconds: sleeps.append(seconds)
    monkeypatch.setattr(pl, "time", clock)
    try:
        answer = append_jsonl(path, {"event": "probe"}, require_lock=required, lock_timeout_sec=0.0)
        assert answer is (not required)
        assert sleeps == []
        assert pl._lock_identity(lock) == original
        if required:
            assert not path.exists()
        else:
            assert json.loads(path.read_text()) == {"event": "probe"}
    finally:
        pl.release_exclusive_file_lock(lock, fd)


def test_default_append_retains_its_lock_budget(tmp_path, monkeypatch):
    from ouroboros import platform_layer as pl
    from ouroboros.utils import append_jsonl

    original = pl.acquire_exclusive_file_lock
    budgets = []

    def observed(path, **kwargs):
        budgets.append(kwargs["timeout_sec"])
        return original(path, **kwargs)

    monkeypatch.setattr(pl, "acquire_exclusive_file_lock", observed)
    path = tmp_path / "events.jsonl"
    assert append_jsonl(path, {"event": "ordinary"})
    assert budgets == [2.0]
    assert json.loads(path.read_text()) == {"event": "ordinary"}


@pytest.mark.parametrize("fallback", [False, True])
def test_an_append_lands_exact_bytes_on_every_platform(tmp_path, monkeypatch, fallback):
    """Windows opens a descriptor in text mode unless ``O_BINARY`` is set, and ``os.write`` then turns each newline
    into CRLF: every ``tools.jsonl`` line of the 7.8.1 qualification grew a byte on Windows. POSIX has no translation
    to observe, so the bit is simulated as in ``test_atomic_write_v639``: the descriptor asks for it, and the
    fallback after a failed open writes the same bytes."""
    import os as _os
    import pathlib

    from ouroboros import utils
    from ouroboros.utils import append_jsonl

    real_o_binary = getattr(_os, "O_BINARY", None)
    fake_o_binary = real_o_binary if real_o_binary is not None else 1 << 26
    flags_seen, real_open = [], _os.open

    def spy_open(path, flags, *args, **kwargs):
        flags_seen.append(flags)
        if fallback:
            raise OSError("no descriptor")
        return real_open(path, flags if real_o_binary is not None else flags & ~fake_o_binary, *args, **kwargs)

    modes, real_path_open = [], pathlib.Path.open

    def spy_path_open(self, mode="r", *args, **kwargs):
        if self.name == "tools.jsonl":
            modes.append(mode)
        return real_path_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(_os, "O_BINARY", fake_o_binary, raising=False)
    monkeypatch.setattr(utils.os, "open", spy_open)
    monkeypatch.setattr(pathlib.Path, "open", spy_path_open)
    monkeypatch.setattr(utils.time, "sleep", lambda _seconds: None)
    path = tmp_path / "tools.jsonl"
    assert append_jsonl(path, {"event": "probe", "text": "a\nb"})
    monkeypatch.undo()
    assert path.read_bytes() == b'{"event": "probe", "text": "a\\nb"}\n'
    appends = [flags for flags in flags_seen if flags & _os.O_APPEND]
    assert appends and all(flags & fake_o_binary for flags in appends)
    assert all("b" in mode for mode in modes) and any(mode.startswith("a") for mode in modes) == fallback


def test_every_byte_append_asks_for_a_binary_descriptor():
    """The same class over every writer: an ``os.open`` that appends asks for ``O_BINARY``, so the bytes a reader
    counts and the offsets it records match on Windows."""
    import ast
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted([*repo.joinpath("ouroboros").rglob("*.py"), *repo.joinpath("supervisor").rglob("*.py")]):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            func = getattr(node, "func", None)
            if (isinstance(node, ast.Call) and isinstance(func, ast.Attribute) and func.attr == "open"
                    and getattr(func.value, "id", None) == "os"):
                text = ast.get_source_segment(source, node) or ""
                if "O_APPEND" in text and "O_BINARY" not in text:
                    offenders.append(f"{path.relative_to(repo).as_posix()}:{node.lineno}")
    assert not offenders, offenders

