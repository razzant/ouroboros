"""Retained CLI error tests for the GitHub tools' transport; publication uses HTTPS separately.

An installed gh outside the inherited PATH (a desktop launch) is found in the platform's common tool folders,
appended after every inherited entry in gh's own environment only; finding it proves no login.
"""

import json
import os
import pathlib
import subprocess
import sys
import types

import pytest

from ouroboros import platform_layer
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
        ("failed to fork: HTTP 403: Resource not accessible by personal access token (https://api.github.com/repos/hub/project/forks)",
         403, None),
        ("HTTP 422: Validation Failed (https://api.github.com/repos/hub/project/pulls)", 422, None),
        ("x" * 700 + " (HTTP 403)", 403, None),
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


@pytest.mark.parametrize("failure", ["cli_missing", "timeout", "exception"])
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
])
def test_argument_refusals_publish_typed_argument_errors(handler, kwargs, expected):
    ctx = types.SimpleNamespace(_active_builtin_tool_result=None)
    text = handler(ctx, **kwargs)
    result = ctx._active_builtin_tool_result
    assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR")
    assert result.text == text == expected


# ---- an installed gh outside the inherited PATH ---------------------------------------------------------------
posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX resolves the program on the child's PATH")
ISSUE = {"number": 7, "title": "Issue 7", "state": "open", "user": {"login": "dev", "type": "User"}, "labels": [],
         "assignees": [], "created_at": "2026-10-01T00:00:00Z", "updated_at": "2026-10-07T00:00:00Z", "closed_at": None,
         "comments": 0, "html_url": "https://github.com/owner/name/issues/7", "body": "b"}


def _stub_gh(directory: pathlib.Path, copy: str) -> pathlib.Path:
    """A real ``gh`` executable answering like ``gh api --include``: the copy that ran, its argv and its PATH ride in
    the body; ``STUB_GH_AUTH=fail`` answers GitHub's 401 instead, echoing the token it was given."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "gh"
    script.write_text(f"""#!{sys.executable}
import json, os, sys
if os.environ.get("STUB_GH_AUTH") == "fail":
    sys.stderr.write("gh: " + os.environ.get("GH_TOKEN", "") + " rejected\\nHTTP 401: Bad credentials (https://api.github.com/user)\\n")
    sys.exit(1)
seen = {{"copy": {copy!r}, "argv": sys.argv[1:], "path": os.environ.get("PATH", "")}}
sys.stdout.write("HTTP/2.0 200 OK\\r\\nContent-Type: application/json\\r\\n\\r\\n"
                 + json.dumps({{"total_count": 1, "incomplete_results": False, "items": [{{**{ISSUE!r}, "title": json.dumps(seen)}}]}}))
""", encoding="utf-8")
    script.chmod(0o755)
    return directory


@pytest.fixture
def gui_path(tmp_path, monkeypatch):
    """A desktop-style inherited PATH holding no gh at all, and a stub gh in a folder the policy names."""
    inherited, common = tmp_path / "inherited", _stub_gh(tmp_path / "common", "common folder")
    inherited.mkdir()
    monkeypatch.setenv("PATH", str(inherited))
    monkeypatch.setattr(transport, "common_tool_directories", lambda path: [str(common)])
    return types.SimpleNamespace(inherited=inherited, common=common)


def _search(tmp_path):
    from tests.test_github_project_target import _context

    ctx, _project = _context(tmp_path, "queued")
    ctx._active_builtin_tool_result = None
    handler = next(entry.handler for entry in transport.get_tools() if entry.name == "github_read")
    return handler(ctx, method="search", kind="issue", query="is:open", repo="owner/name")


def _seen(text):
    """What the stub that ran saw, out of the github_read search answer."""
    line, _nl, body = text.partition("\n")
    assert line.startswith("github_read search "), text
    return json.loads(json.loads(body)["items"][0]["title"])


@posix_only
def test_github_read_runs_an_installed_gh_from_a_common_tool_folder(gui_path, tmp_path):
    """The reported failure: a real github_read search under a PATH without gh. The gh in a common tool folder runs,
    behind every inherited entry of its own environment; this process's PATH stays what it was."""
    seen = _seen(_search(tmp_path))

    assert seen["copy"] == "common folder" and seen["argv"][:2] == ["api", "--include"]
    assert seen["argv"][2].startswith("search/issues?") and "repo%3Aowner%2Fname" in seen["argv"][2]
    assert seen["path"] == os.pathsep.join([str(gui_path.inherited), str(gui_path.common)])
    assert os.environ["PATH"] == str(gui_path.inherited)


@posix_only
def test_a_gh_on_the_inherited_path_keeps_its_precedence(gui_path, tmp_path):
    """The negative precedence control: a gh the inherited PATH already holds runs, not the common folder's."""
    _stub_gh(gui_path.inherited, "inherited PATH")

    assert _seen(_search(tmp_path))["copy"] == "inherited PATH"


@posix_only
def test_no_gh_anywhere_is_still_the_missing_cli_with_nothing_launched(gui_path, tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(transport, "common_tool_directories", lambda path: [str(empty)])

    result = transport._gh_run(["api", "user"], types.SimpleNamespace(repo_dir=tmp_path))

    assert (result.ok, result.exit_code, result.failure) == (False, None, "cli_missing")
    assert "`gh` CLI not found" in result.text


@posix_only
def test_a_found_gh_whose_login_fails_reports_the_401_without_the_token(gui_path, tmp_path, monkeypatch):
    """Finding the binary proves no login: GitHub's refusal stays an exit failure with its status, token redacted."""
    monkeypatch.setenv("STUB_GH_AUTH", "fail")

    result = transport._gh_run(["api", "user"], types.SimpleNamespace(repo_dir=tmp_path))

    assert (result.ok, result.exit_code, result.http_status, result.failure) == (False, 1, 401, "exit")
    assert "ghp_SYNTHETIC1234567890" not in result.text and "Bad credentials" in result.text


@posix_only
def test_the_common_tool_folders_are_existing_ones_the_path_lacks_in_policy_order(tmp_path, monkeypatch):
    """The one policy, read without the stub: existing folders only, none the given PATH already holds, in order."""
    home = tmp_path / "home"
    for folder in (".local/bin", ".cargo/bin", "go/bin"):
        (home / folder).mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    inherited = os.pathsep.join(["/nowhere", str(home / ".cargo" / "bin")])
    before = dict(os.environ)

    found = platform_layer.common_tool_directories(inherited)

    assert [folder for folder in found if folder.startswith(str(home))] == [str(home / ".local" / "bin"),
                                                                             str(home / "go" / "bin")]
    assert all(pathlib.Path(folder).is_dir() for folder in found) and not set(found) & set(inherited.split(os.pathsep))
    assert dict(os.environ) == before


@posix_only
def test_the_process_bootstrap_still_prepends_the_same_folders_once(tmp_path, monkeypatch):
    """The other consumer of that policy keeps its contract: once per process, ahead of the inherited entries."""
    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(platform_layer, "_PATH_BOOTSTRAPPED", False)
    expected = platform_layer.common_tool_directories(str(tmp_path))

    added = platform_layer.bootstrap_process_path()

    assert added == expected and str(home / ".local" / "bin") in added
    assert os.environ["PATH"] == os.pathsep.join([*added, str(tmp_path)])
    assert platform_layer.bootstrap_process_path() == [] and os.environ["PATH"] == os.pathsep.join([*added, str(tmp_path)])
