"""A synthetic gh process with separate startup and response callbacks."""

from subprocess import PIPE
from unittest.mock import MagicMock

from ouroboros.tools import github


def stub_gh(monkeypatch, reply, *, start=None):
    def popen(argv, *, cwd, env, stdout, stderr, encoding, **_):
        assert stdout == stderr == PIPE and encoding == "utf-8"
        if start:
            start(argv, cwd=cwd, env=env)
        proc = MagicMock()
        proc.__enter__.return_value = proc

        def communicate(input=None, timeout=None):
            if proc.kill.called:  # Windows drains the same pipes after a timeout.
                return "", ""
            result = reply(argv, cwd=cwd, env=env, input=input, timeout=timeout)
            proc.returncode = result.returncode
            return result.stdout, result.stderr

        proc.communicate.side_effect = communicate
        return proc

    monkeypatch.setattr(github, "Popen", popen)
