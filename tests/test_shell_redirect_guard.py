"""Glued-redirect detection (["find", ..., "2>/dev/null"] as one argv element).

Since #447 A5 the detection DISCLOSES instead of refusing: the element is literal
data to subprocess (no shell interprets it), so the command runs and the result
carries the actionable [sh,-c,...] hint explaining a cryptic program error.
A '>' inside a sed/awk/grep expression must NOT be flagged at all."""

import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.tools.shell import _GLUED_REDIRECT_RE, _run_shell


def _ctx(tmp_path):
    return SimpleNamespace(repo_dir=tmp_path, drive_logs=lambda: pathlib.Path(str(tmp_path)))


@pytest.mark.parametrize(
    "arg",
    # output redirects (permissive glued tail) + UNAMBIGUOUS input-redirect shapes
    ["2>/dev/null", "2>&1", ">out.log", ">>app.log", "&>all.log", ">&2", "1>x", "2>>err",
     "<<EOF", "<<<word", "0<in.txt", "2<&1", "<"],
)
def test_glued_redirect_detected(arg):
    assert _GLUED_REDIRECT_RE.match(arg)


@pytest.mark.parametrize(
    "arg",
    # A bare "<word" is NOT flagged: it is indistinguishable from a literal angle-
    # bracket arg (grep "<div>"), and false-flagging those is worse than missing a
    # rare glued "<file" input redirect (the output side stays fully guarded).
    ["s/a>b/c/g", "find", "-name", "*.txt", "foo|bar", "x>y", "> hi", "report2024", "-->flag", "2", ".",
     "<div>", "<stdin>", "<html>", "<in.txt"],
)
def test_legit_args_not_flagged(arg):
    assert not _GLUED_REDIRECT_RE.match(arg)


def test_run_shell_discloses_glued_redirect(tmp_path, monkeypatch):
    # #447 A5: the redirect-looking element is literal data to subprocess — the
    # command runs and the literal pass-through is disclosed with the
    # ["sh","-c",...] escape hatch, so the program's own error stays explainable.
    from subprocess import CompletedProcess

    monkeypatch.setattr("ouroboros.tools.shell.load_settings", lambda: {})
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr("ouroboros.tools.shell._tracked_subprocess_run", fake_run)
    ctx = _ctx(tmp_path)
    ctx.drive_root = tmp_path  # the command now really runs, so binding resolves
    out = _run_shell(ctx, cmd=["find", ".", "-name", "*.py", "2>/dev/null"])
    assert "SHELL_LITERAL_ARGV_NOTE" in out
    assert "2>/dev/null" in out
    assert "sh" in out  # points to the ["sh","-c",...] escape hatch
    assert seen["cmd"][-1] == "2>/dev/null"  # reached the program literally
