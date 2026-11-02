"""``_gh_run``'s exact-capture branch over a real fake ``gh`` executable.

Paired with the text transport it extends: the same bytes through both branches show what the exact branch
adds (bytes as written, a hash, a cap that stops the process) and what the one-command baseline loses.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import types

import pytest

from ouroboros.tools import github as transport
from tests.test_github_project_target import _context

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the fake gh is a POSIX shell wrapper")

_FAKE_GH = r'''
import os, sys, time
mode = os.environ.get("FAKE_GH_MODE", "bytes")
out = sys.stdout.buffer
if mode == "bytes":
    out.write(bytes.fromhex(os.environ["FAKE_GH_STDOUT_HEX"]))
    out.flush()
elif mode == "flood":
    chunk = b"x" * 65536
    for _ in range(1024):  # 64 MiB unless the reader stops the process first.
        out.write(chunk)
        out.flush()
    open(os.environ["FAKE_GH_DONE_FILE"], "w").close()
elif mode == "fail":
    sys.stderr.write(os.environ.get("FAKE_GH_STDERR", "gh: Not Found (HTTP 404)\n"))
    sys.stderr.flush()
    out.write(b'{"message":"Not Found"}')
    sys.exit(1)
elif mode == "sleep":
    out.write(b"partial\n")
    out.flush()
    time.sleep(float(os.environ.get("FAKE_GH_SLEEP", "30")))
elif mode == "env":
    out.write(("GH_TOKEN set" if os.environ.get("GH_TOKEN") else "GH_TOKEN unset").encode() + b"\n")
elif mode == "argv":
    import json
    out.write(json.dumps(sys.argv[1:]).encode() + b"\n")
elif mode == "stdin":
    body = sys.stdin.buffer.read()
    with open(os.environ["FAKE_GH_STDIN_FILE"], "wb") as received:
        received.write(body)
    out.write(body)
elif mode == "stderr_flood":
    token = os.environ["GH_TOKEN"].encode()
    at = int(os.environ["FAKE_GH_TOKEN_AT"])  # The token starts here; the stream always exceeds the 64 KiB cap.
    sys.stderr.buffer.write(b"e" * at + token + b"\n" + b"e" * max(4096, 65536 + 4096 - at) + b"\n")
    sys.stderr.buffer.flush()
    sys.exit(1)
'''


@pytest.fixture
def fake_gh(tmp_path, monkeypatch):
    """A ``gh`` on PATH whose behaviour the test selects through FAKE_GH_* variables; every token is synthetic."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = tmp_path / "fake_gh.py"
    script.write_text(_FAKE_GH, encoding="utf-8")
    wrapper = bin_dir / "gh"
    wrapper.write_text('#!/bin/sh\nexec "$FAKE_GH_PYTHON" -I "$FAKE_GH_SCRIPT" "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    monkeypatch.setenv("FAKE_GH_PYTHON", sys.executable)
    monkeypatch.setenv("FAKE_GH_SCRIPT", str(script))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_SYNTHETIC1234567890")
    for key in ("FAKE_GH_MODE", "FAKE_GH_STDOUT_HEX", "FAKE_GH_STDERR", "FAKE_GH_DONE_FILE", "FAKE_GH_SLEEP", "FAKE_GH_TOKEN_AT"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def _ctx(tmp_path):
    return types.SimpleNamespace(repo_dir=tmp_path, _active_builtin_tool_result=None)


@pytest.mark.parametrize("body", ["ASCII body", "An em dash —", "Owner's code → 🤖"])
@pytest.mark.parametrize("method", ["merge", "squash"])
def test_text_stdin_is_utf8_under_a_non_utf8_locale(fake_gh, tmp_path, body, method):
    """Real parent and gh child pipes, with UTF-8 mode and locale coercion disabled."""
    source, received = tmp_path / "body.txt", tmp_path / "received.bin"
    source.write_text(body, encoding="utf-8")
    fake_gh.setenv("FAKE_GH_MODE", "stdin")
    fake_gh.setenv("FAKE_GH_STDIN_FILE", str(received))
    script = '''
import codecs, json, locale, pathlib, sys, types
from ouroboros.tools.github import _gh_run
locale.setlocale(locale.LC_CTYPE, "C")
assert not sys.flags.utf8_mode
assert codecs.lookup(locale.getpreferredencoding(False)).name != "utf-8"
source = pathlib.Path(sys.argv[1])
body = source.read_text(encoding="utf-8")
result = _gh_run(["pr", "merge", "7", "--" + sys.argv[2], "--match-head-commit", "a" * 40,
                  "--body-file", "-"], types.SimpleNamespace(repo_dir=source.parent), input_data=body)
print(json.dumps({"ok": result.ok, "text": result.text, "failure": result.failure}))
'''
    child = subprocess.run([sys.executable, "-X", "utf8=0", "-c", script, str(source), method],
                           env={**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONCOERCECLOCALE": "0"},
                           capture_output=True, encoding="utf-8", timeout=30)
    assert child.returncode == 0, child.stderr
    result = json.loads(child.stdout)
    assert result == {"ok": True, "text": body, "failure": ""}
    assert received.read_bytes() == body.encode("utf-8")


def test_exact_capture_keeps_crlf_and_the_trailing_newline_the_text_branch_folds(fake_gh, tmp_path):
    """Paired: one command, two branches. The baseline is what a shell read of ``gh run view --log`` yields."""
    source = b"job\tstep\tline one\r\njob\tstep\tline two\r\n"
    fake_gh.setenv("FAKE_GH_STDOUT_HEX", source.hex())
    target = tmp_path / "capture" / "log.bin"
    target.parent.mkdir()

    exact = transport._gh_run(["run", "view", "11", "--log"], _ctx(tmp_path), capture_to=target, max_bytes=1 << 20)
    baseline = transport._gh_run(["run", "view", "11", "--log"], _ctx(tmp_path))

    assert exact.ok and exact.text == "" and exact.exit_code == 0 and exact.failure == ""
    assert target.read_bytes() == source
    assert (exact.capture.size, exact.capture.truncated) == (len(source), False)
    assert exact.capture.sha256 == hashlib.sha256(source).hexdigest()
    assert exact.capture.path == target
    # The text branch is the baseline: universal newlines fold CRLF and strip() removes the final newline.
    assert baseline.ok and baseline.text == "job\tstep\tline one\njob\tstep\tline two"
    assert baseline.capture is None


def test_exact_capture_retains_a_non_utf8_byte_the_text_branch_cannot_decode(fake_gh, tmp_path):
    source = b"step\tbinary \xff tail\n"
    fake_gh.setenv("FAKE_GH_STDOUT_HEX", source.hex())
    target = tmp_path / "log.bin"

    exact = transport._gh_run(["run", "view", "11", "--log"], _ctx(tmp_path), capture_to=target, max_bytes=4096)
    baseline = transport._gh_run(["run", "view", "11", "--log"], _ctx(tmp_path))

    assert exact.ok and target.read_bytes() == source
    assert not baseline.ok and baseline.failure == "exception" and "decode" in baseline.text


def test_cap_bounds_the_file_and_stops_the_process_before_it_finishes(fake_gh, tmp_path):
    """The cap bounds acquisition: the file holds exactly the cap, hashed as written, and the flood never completes."""
    done = tmp_path / "flood-finished"
    fake_gh.setenv("FAKE_GH_MODE", "flood")
    fake_gh.setenv("FAKE_GH_DONE_FILE", str(done))
    target = tmp_path / "flood.bin"
    cap = 300_000

    res = transport._gh_run(["api", "repos/o/r/actions/artifacts/5/zip"], _ctx(tmp_path), timeout=60,
                            capture_to=target, max_bytes=cap)

    assert res.ok and res.exit_code is None and res.failure == ""
    assert res.capture.truncated is True and res.capture.size == cap
    assert target.stat().st_size == cap
    assert res.capture.sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    assert not done.exists(), "the process was stopped at the cap instead of being read to the end"


def test_capture_exit_failure_is_classified_like_the_text_branch_and_redacted(fake_gh, tmp_path):
    fake_gh.setenv("FAKE_GH_MODE", "fail")
    fake_gh.setenv("FAKE_GH_STDERR", "gh: token ghp_SYNTHETIC1234567890 rejected\ngh: Not Found (HTTP 404)\n")
    target = tmp_path / "out.bin"

    exact = transport._gh_run(["api", "repos/o/r/actions/artifacts/5/zip"], _ctx(tmp_path), capture_to=target, max_bytes=4096)
    baseline = transport._gh_run(["api", "repos/o/r/actions/artifacts/5/zip"], _ctx(tmp_path))

    for res in (exact, baseline):
        assert (res.ok, res.exit_code, res.http_status, res.failure, res.capture) == (False, 1, 404, "exit", None)
        assert res.text == "⚠️ GH_ERROR: gh: token *** rejected | gh: Not Found (HTTP 404)"


def test_capture_timeout_kills_the_process_and_reports_it(fake_gh, tmp_path):
    fake_gh.setenv("FAKE_GH_MODE", "sleep")
    fake_gh.setenv("FAKE_GH_SLEEP", "30")
    target = tmp_path / "out.bin"
    started = time.monotonic()

    res = transport._gh_run(["run", "view", "11", "--log"], _ctx(tmp_path), timeout=1, capture_to=target, max_bytes=4096)

    assert (res.ok, res.failure, res.text, res.capture) == (False, "timeout", "⚠️ GH_TIMEOUT: exceeded 1s.", None)
    assert time.monotonic() - started < 20


@pytest.mark.parametrize("kwargs", [{"max_bytes": 0}, {"max_bytes": -1}, {"max_bytes": 10.5}, {"max_bytes": True},
                                    {"max_bytes": 10, "input_data": "body"}])
def test_capture_refuses_a_missing_cap_and_stdin_before_any_launch(fake_gh, tmp_path, kwargs):
    fake_gh.setenv("FAKE_GH_MODE", "flood")
    fake_gh.setenv("FAKE_GH_DONE_FILE", str(tmp_path / "never"))
    with pytest.raises(ValueError):
        transport._gh_run(["api", "x"], _ctx(tmp_path), capture_to=tmp_path / "out.bin", **kwargs)
    assert not (tmp_path / "out.bin").exists()


def test_capture_launch_sets_the_invocation_custody_fact(fake_gh, tmp_path):
    fake_gh.setenv("FAKE_GH_STDOUT_HEX", b"x".hex())
    token = transport._SUBMITTED.set([False])
    try:
        transport._gh_run(["api", "x"], _ctx(tmp_path), capture_to=tmp_path / "out.bin", max_bytes=16)
        assert transport._SUBMITTED.get() == [True]
    finally:
        transport._SUBMITTED.reset(token)


def test_configured_cli_auth_runs_without_a_saved_token(fake_gh, tmp_path, monkeypatch):
    """No GITHUB_TOKEN anywhere: the launch carries no GH_TOKEN and still runs the configured CLI."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(transport, "github_token_from_env_or_settings", lambda: "")
    fake_gh.setenv("FAKE_GH_MODE", "env")

    exact = transport._gh_run(["api", "user"], _ctx(tmp_path), capture_to=tmp_path / "out.bin", max_bytes=64)
    baseline = transport._gh_run(["api", "user"], _ctx(tmp_path))

    assert (tmp_path / "out.bin").read_bytes() == b"GH_TOKEN unset\n"
    assert baseline.text == "GH_TOKEN unset"
    assert exact.ok and baseline.ok


def test_capture_never_publishes_a_sidecar(fake_gh, tmp_path):
    sentinel = object()
    ctx = types.SimpleNamespace(repo_dir=tmp_path, _active_builtin_tool_result=sentinel)
    fake_gh.setenv("FAKE_GH_MODE", "fail")
    transport._gh_run(["api", "x"], ctx, capture_to=tmp_path / "out.bin", max_bytes=16)
    assert ctx._active_builtin_tool_result is sentinel


def test_repo_view_names_its_target_positionally_and_every_other_command_takes_repo_flag(fake_gh, tmp_path):
    """Real argv through a real process: `gh repo view [<repository>]` has no --repo flag (gh 2.101 INHERITED FLAGS list
    only --help), so the transport inserts the explicit target positionally there and appends `--repo` elsewhere."""
    fake_gh.setenv("FAKE_GH_MODE", "argv")
    ctx, _project = _context(tmp_path, "queued")
    target = "github.example/owner/selected"

    view = transport._gh_run(["repo", "view", "--json", "nameWithOwner,url"], ctx, repo=target)
    bound = transport._gh_run(["repo", "view", "--json", "nameWithOwner,url"], ctx, repo="")
    other = transport._gh_run(["pr", "view", "7", "--json", "number"], ctx, repo=target)

    assert json.loads(view.text) == ["repo", "view", target, "--json", "nameWithOwner,url"]
    assert json.loads(bound.text) == ["repo", "view", "--json", "nameWithOwner,url"]
    assert json.loads(other.text) == ["pr", "view", "7", "--json", "number", "--repo", target]


@pytest.mark.skipif(shutil.which("gh") is None, reason="the installed GitHub CLI is the evidence")
def test_installed_gh_help_confirms_the_repo_view_parser_shape():
    """Parser evidence from the real CLI: repo view takes a positional repository and lists no --repo flag."""
    env = {**os.environ, "GH_NO_UPDATE_NOTIFIER": "1", "GH_PROMPT_DISABLED": "1", "NO_COLOR": "1"}
    repo_help = subprocess.run(["gh", "repo", "view", "--help"], capture_output=True, text=True, env=env, timeout=30).stdout
    pr_help = subprocess.run(["gh", "pr", "view", "--help"], capture_output=True, text=True, env=env, timeout=30).stdout
    assert "gh repo view [<repository>] [flags]" in repo_help
    inherited = repo_help.split("INHERITED FLAGS", 1)[1].split("JSON FIELDS", 1)[0]
    assert "--repo" not in inherited and "--help" in inherited
    assert "-R, --repo [HOST/]OWNER/REPO" in pr_help


@pytest.mark.parametrize("token_at", [65536 - 10, 65536 - 1024 - 10, 65536 - 1024 + 2, 100])
def test_overflowing_stderr_is_withheld_whole_so_no_token_crosses_any_cut(fake_gh, tmp_path, token_at):
    """Any cut of an over-cap stderr leaves a boundary a token can straddle (the 64 KiB cap, a 1 KiB guard before
    it, or anywhere else), and whole-value redaction cannot match a fragment; so the diagnostics are withheld whole,
    the exit code stays and the HTTP status is unknown."""
    fake_gh.setenv("FAKE_GH_MODE", "stderr_flood")
    fake_gh.setenv("FAKE_GH_TOKEN_AT", str(token_at))
    env = transport._gh_env(types.SimpleNamespace())
    capture, returncode, stderr, timed_out = transport._capture_stdout(
        ["gh"], tmp_path, env, 30, tmp_path / "out.bin", 4096)

    assert (returncode, timed_out, capture.size) == (1, False, 0)
    assert stderr == "[stderr exceeded 64 KiB; diagnostics withheld]"
    assert "ghp_" not in stderr and "SYNTHETIC" not in stderr
    res = transport._gh_run(["api", "x"], types.SimpleNamespace(repo_dir=tmp_path), capture_to=tmp_path / "o.bin", max_bytes=16)
    assert (res.ok, res.exit_code, res.http_status, res.failure) == (False, 1, None, "exit")
    assert res.text == "⚠️ GH_ERROR: [stderr exceeded 64 KiB; diagnostics withheld]"


def test_stderr_within_the_cap_is_kept_and_redacted_whole(fake_gh, tmp_path):
    """Negative control: an in-cap stderr keeps its lines, with the token redacted as a whole value."""
    fake_gh.setenv("FAKE_GH_MODE", "fail")
    raw_stderr = "gh: ghp_SYNTHETIC1234567890 rejected (HTTP 401)\n" + "e" * 60000 + "\n"
    fake_gh.setenv("FAKE_GH_STDERR", raw_stderr)
    env = transport._gh_env(types.SimpleNamespace())
    _capture, returncode, stderr, _timed_out = transport._capture_stdout(["gh"], tmp_path, env, 30, tmp_path / "out.bin", 4096)
    assert returncode == 1 and stderr == raw_stderr, "kept whole and raw before classification: under the cap, nothing is withheld"
    res = transport._gh_run(["api", "x"], types.SimpleNamespace(repo_dir=tmp_path), capture_to=tmp_path / "o.bin", max_bytes=4096)
    assert res.http_status == 401 and "SYNTHETIC" not in res.text and "gh: *** rejected (HTTP 401)" in res.text
