"""Glued-redirect detection (["find", ..., "2>/dev/null"] as one argv element).

Since #447 A5 the detection DISCLOSES instead of refusing: the element is literal
data to subprocess (no shell interprets it), so the command runs and the result
carries the actionable [sh,-c,...] hint explaining a cryptic program error.
A '>' inside a sed/awk/grep expression must NOT be flagged at all."""

import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.tools.shell import _GLUED_REDIRECT_RE, _literal_argv_notes, _run_shell


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


@pytest.mark.parametrize("cmd", [
    ["cmd.exe", "/d", "/c", "type", "input.txt", ">", "output.txt"],
    ["cmd", "/c", "dir", "2>nul"],
    ["CMD.EXE", "/K", "echo a", "&&", "echo b"],
    ["C:\\Windows\\System32\\cmd.exe", "/c", "type", "a", ">", "b"],
    ["pwsh", "-Command", "Get-Content", "a", ">", "b"],
    ["powershell.exe", "-NoProfile", "-c", "Get-Content", "a", "|", "Out-File", "b"],
])
def test_an_explicit_interpreter_operand_is_not_called_an_ordinary_argument(cmd):
    """#927: cmd /c|/k and PowerShell -c|-Command parse their operand themselves."""
    assert _literal_argv_notes(cmd) == ""


@pytest.mark.parametrize(("cmd", "subject"), [
    (["git", "log", "|", "head"], 'shell operator "|"'),
    (["find", ".", "-name", "*.py", "2>/dev/null"], 'redirect-looking argument "2>/dev/null"'),
    (["echo", "$HOME"], 'literal env reference "$HOME"'),
    (["sh", "-c", "cat", ">", "x"], 'shell operator ">"'),  # a positional parameter, not script text
    (["powershell.exe", "-File", "s.ps1", ">", "b"], 'shell operator ">"'),
    (["pwsh", "-File", "s.ps1", "-c", ">", "b"], 'shell operator ">"'),  # script arguments
    (["pwsh", "-CommandWithArgs", "$args", "a", ">", "b"], 'shell operator ">"'),
    (["cmd.exe", "type", "a", ">", "b"], 'shell operator ">"'),  # no /c: cmd gets no command text
])
def test_ordinary_arguments_keep_exactly_one_note_each(cmd, subject):
    notes = _literal_argv_notes(cmd)
    assert notes.count("SHELL_LITERAL_ARGV_NOTE") == 1 and subject in notes
    assert "passed to the program as an ordinary argument; run_command starts no shell" in notes


@pytest.mark.parametrize(("cmd", "subjects"), [
    (["echo", ">$HOME"], ['literal env reference "$HOME"']),  # one element: one note, never two
    (["echo", "$HOME", "2>/dev/null"], ['literal env reference "$HOME"', 'redirect-looking argument "2>/dev/null"']),
    (["echo", ">$HOME", "2>/dev/null"], ['literal env reference "$HOME"', 'redirect-looking argument "2>/dev/null"']),
    (["sh", "-c", "cat", ">$HOME"], ['redirect-looking argument ">$HOME"']),  # no env note under a shell head
])
def test_each_argv_element_gets_at_most_one_note(cmd, subjects):
    notes = _literal_argv_notes(cmd)
    assert notes.count("SHELL_LITERAL_ARGV_NOTE") == len(subjects)
    assert all(subject in notes for subject in subjects)


def test_the_guard_and_the_note_read_one_inline_operand():
    from ouroboros.shell_parse import inline_operand_start, inline_shell_command

    assert inline_shell_command(["cmd.exe", "/d", "/c", "type", "a", ">", "b"], "cmd") == "type a > b"
    assert inline_shell_command(["pwsh", "-Command", "Get-Content", "a"], "pwsh") == "Get-Content a"
    assert inline_shell_command(["bash", "-c", "echo a > b"], "bash") == "echo a > b"
    assert inline_shell_command(["cmd", "/c"], "cmd") == "" and inline_shell_command(["cmd", "dir"], "cmd") == ""
    # The guard keeps its wider view of a later -c; only the disclosure stops at -File.
    argv = ["pwsh", "-File", "s.ps1", "-c", "rm", "x"]
    assert inline_shell_command(argv, "pwsh") == "rm x"
    assert (inline_operand_start(argv, "pwsh"), inline_operand_start(argv, "pwsh", strict=True)) == (4, 0)
