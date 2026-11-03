"""Retained CLI error tests for the GitHub tools' transport; publication uses HTTPS separately."""

import subprocess
import types

import pytest

from ouroboros.tools import github as transport
from ouroboros.tools.github_read import run_read
from ouroboros.tools.github_write import run_write


@pytest.fixture(autouse=True)
def synthetic_credentials(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_SYNTHETIC1234567890")


@pytest.mark.parametrize(
    "stderr,expected_status,expected_text",
    [
        # the FIRST line-terminal gh marker wins; the bounded head keeps three non-empty lines
        ("\n first\n\n second (HTTP 401)\n third\n ignored (HTTP 403)", 401,
         "⚠️ GH_ERROR: first | second (HTTP 401) | third"),
        # a marker quoted mid-sentence is prose; gh's own marker ends its line
        ("warning: quoted (HTTP 403) is not a status\ngh: Not Found (HTTP 404)", 404,
         "⚠️ GH_ERROR: warning: quoted (HTTP 403) is not a status | gh: Not Found (HTTP 404)"),
        ("a marker (HTTP 403) inside a sentence", None, "⚠️ GH_ERROR: a marker (HTTP 403) inside a sentence"),
        ("gh: Not Found (HTTP 404)\r\n", 404, "⚠️ GH_ERROR: gh: Not Found (HTTP 404)"),
        # gh api without a message; go-gh HTTPError from every other command, wrapped or bare
        ("gh: HTTP 401", 401, "⚠️ GH_ERROR: gh: HTTP 401"),
        ("failed to fork: HTTP 403: Resource not accessible by personal access token (https://api.github.com/repos/hub/project/forks)",
         403, None),
        ("HTTP 422: Validation Failed (https://api.github.com/repos/hub/project/pulls)", 422, None),
        ("note that HTTP 403 is not what happened here", None, "⚠️ GH_ERROR: note that HTTP 403 is not what happened here"),
        # a bare number is prose, never a status
        ("permission denied, status 403", None, "⚠️ GH_ERROR: permission denied, status 403"),
        # the status is read from the WHOLE redacted stderr, before the head is cut
        ("first\nsecond\nthird\nlater line (HTTP 403)", 403, "⚠️ GH_ERROR: first | second | third"),
        ("x" * 700 + " (HTTP 403)", 403, None),
        ("", None, "⚠️ GH_ERROR: "),
    ],
)
def test_transport_bounds_stderr_and_only_reads_gh_http_marker(
    monkeypatch, tmp_path, stderr, expected_status, expected_text,
):
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 7, "ignored stdout", stderr),
    )
    result = transport._gh_run(["api", "/user"], types.SimpleNamespace(repo_dir=tmp_path))
    assert (result.ok, result.exit_code, result.http_status, result.failure) == (False, 7, expected_status, "exit")
    assert result.text.startswith("⚠️ GH_ERROR: ")
    assert len(result.text.removeprefix("⚠️ GH_ERROR: ")) <= 600
    assert "ignored" not in result.text
    if expected_text is not None:
        assert result.text == expected_text


@pytest.mark.parametrize("filename,failure,fragment", [
    (None, "cli_missing", "`gh` CLI not found"),
    ("/usr/local/bin/gh", "cli_missing", "`gh` CLI not found"),
    # a vanished working directory is a launch failure, not a missing CLI
    ("/vanished/project", "exception", "/vanished/project"),
])
def test_launch_failure_names_what_is_missing(monkeypatch, tmp_path, filename, failure, fragment):
    def run(cmd, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory", filename)

    monkeypatch.setattr(subprocess, "run", run)
    result = transport._gh_run(["api", "/user"], types.SimpleNamespace(repo_dir=tmp_path))
    assert (result.ok, result.exit_code, result.http_status, result.failure) == (False, None, None, failure)
    assert fragment in result.text


@pytest.mark.parametrize("failure", ["cli_missing", "timeout", "exit", "exception"])
def test_transport_never_publishes_a_sidecar(monkeypatch, tmp_path, failure):
    sentinel = object()
    ctx = types.SimpleNamespace(repo_dir=tmp_path, _active_builtin_tool_result=sentinel)

    def run(cmd, **kwargs):
        if failure == "cli_missing":
            raise FileNotFoundError()
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        if failure == "exception":
            raise RuntimeError("synthetic ghp_SYNTHETIC1234567890")
        return subprocess.CompletedProcess(cmd, 1, "", "synthetic")

    monkeypatch.setattr(subprocess, "run", run)
    result = transport._gh_run(["api", "/user"], ctx)
    assert result.failure == failure
    assert ctx._active_builtin_tool_result is sentinel
    assert transport._gh_cmd(["api", "/user"], ctx) == result.text
    assert ctx._active_builtin_tool_result is sentinel
    assert "ghp_SYNTHETIC1234567890" not in result.text



@pytest.mark.parametrize("handler,kwargs,expected", [
    (run_read, {"method": "get", "kind": "issue", "target": 0}, "⚠️ TOOL_ARG_ERROR: issue number must be positive."),
    (run_write, {"method": "comment", "target": 0, "body": "comment"},
     "⚠️ TOOL_ARG_ERROR: target must be the positive issue, pull request or discussion number."),
    (run_write, {"method": "update", "target": 0, "state": "closed"},
     "⚠️ TOOL_ARG_ERROR: target must be the positive issue or pull request number."),
    (run_read, {"method": "get", "kind": "pr", "target": 0}, "⚠️ TOOL_ARG_ERROR: pr number must be positive."),
    (run_write, {"method": "review", "target": 0, "body": "comment"}, "⚠️ TOOL_ARG_ERROR: target must be the positive pull request number."),
    (run_write, {"method": "comment", "target": 7, "body": "  "}, "⚠️ TOOL_ARG_ERROR: comment body cannot be empty."),
    (run_write, {"method": "create_issue", "title": " "}, "⚠️ TOOL_ARG_ERROR: issue title cannot be empty."),
])
def test_argument_refusals_publish_typed_argument_errors(handler, kwargs, expected):
    ctx = types.SimpleNamespace(_active_builtin_tool_result=None)
    text = handler(ctx, **kwargs)
    result = ctx._active_builtin_tool_result
    assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR")
    assert result.text == text == expected
