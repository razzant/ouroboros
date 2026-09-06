"""Single-element command mistakes get a repair hint without changing argv.

The contribution's examples remain regression cases. Shell syntax is executed
only when the caller explicitly chooses a shell or run_script.
"""
from __future__ import annotations

import pathlib
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

from ouroboros.tools.shell import _run_shell


def _ctx(tmp_path):
    return SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path,
                           drive_logs=lambda: pathlib.Path(str(tmp_path)))


@pytest.mark.parametrize('raw', [
    'git && status && --porcelain', 'cd foo && make', 'grep -rn foo && bar',
    'echo $HOME && ls', 'git status --porcelain', 'grep -rn foo . | head -20',
    "grep 'a|b' file.txt", 'make || true', 'ls -la', "sh -c 'a | b'",
])
def test_single_element_is_not_split_or_wrapped(tmp_path, monkeypatch, raw):
    calls = []
    def missing(cmd, **kwargs):
        calls.append(cmd)
        raise FileNotFoundError('executable not found')
    monkeypatch.setattr('ouroboros.tools.shell._tracked_subprocess_run', missing)
    result = _run_shell(_ctx(tmp_path), [raw])
    assert calls == [[raw]]
    assert 'SHELL_ARG_ERROR' in str(result)
    assert 'ONE executable name' in str(result)
    assert 'run_script' in str(result)
    assert 'AUTO_SPLIT' not in str(result) and 'AUTO_WRAP' not in str(result)


@pytest.mark.parametrize('cmd', [
    ['git', 'status'], ['printf', '%s', 'a | b'],
    ['sh', '-c', 'printf one | cat'], ['printf', '%s', '&&'],
])
def test_explicit_argv_reaches_process_unchanged(tmp_path, monkeypatch, cmd):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return CompletedProcess(argv, 0, 'ok', '')
    monkeypatch.setattr('ouroboros.tools.shell._tracked_subprocess_run', run)
    result = _run_shell(_ctx(tmp_path), cmd)
    assert calls == [cmd]
    assert 'SHELL_ARG_ERROR' not in str(result)
    assert 'AUTO_SPLIT' not in str(result) and 'AUTO_WRAP' not in str(result)
