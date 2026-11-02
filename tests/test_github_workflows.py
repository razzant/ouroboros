"""The run-log and artifact methods of ``github_read``.

Exercise exact captured bytes, source identity, retained reads and negative controls. A valid shell
command can preserve bytes or fetch an artifact by id too; these tests make no claim that a shell
alternative cannot provide the same facts, or that model latency has improved.
"""

import hashlib
import io
import json
import pathlib
import subprocess
import zipfile
from types import SimpleNamespace

import pytest

from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.tool_access import canonical_data_root
from ouroboros.tools import github, github_workflows
from tests.test_github_project_target import BASE, SHA, _context

LOG = b"test\tRun pytest\t2026-10-20T10:00:00Z FAILED tests/test_x.py::test_y\r\ntest\tRun pytest\t=== 1 failed ===\r\n"
OTHER = "https://github.example/other/thing"


def _answer(payload):
    return github.GhResult(True, json.dumps(payload), 0, None, "")


def _zip(members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _capture(path, data, cap):
    """What the transport's exact branch leaves behind: the first ``cap`` bytes, hashed as written."""
    kept = data[:cap]
    pathlib.Path(path).write_bytes(kept)
    truncated = len(data) > cap
    return github.GhResult(True, "", None if truncated else 0, None, "",
                           github.GhCapture(pathlib.Path(path), len(kept), hashlib.sha256(kept).hexdigest(), truncated))


class FakeGh:
    """GitHub's answers about runs and artifacts; every call is recorded with its transport binding and cap."""

    def __init__(self, *, run=None, log=LOG, artifacts=(), artifact=None, archive=b"", repo_url=BASE, failing=None, total=None):
        self.run = run if run is not None else {"databaseId": 11, "url": f"{BASE}/actions/runs/11", "status": "completed",
                                                "conclusion": "failure", "attempt": 1, "headSha": SHA, "workflowName": "CI",
                                                "jobs": [{"databaseId": 55, "name": "test", "status": "completed",
                                                          "conclusion": "failure", "url": f"{BASE}/actions/runs/11/job/55"}]}
        self.log, self.artifacts, self.artifact, self.archive, self.repo_url = log, list(artifacts), artifact, archive, repo_url
        self.failing, self.calls, self.total = failing or {}, [], total

    def __call__(self, args, ctx, timeout=30, input_data=None, *, repo=github._GENERIC_TRANSPORT, capture_to=None, max_bytes=0):
        self.calls.append(SimpleNamespace(args=list(args), repo=repo, timeout=timeout, capture_to=capture_to, max_bytes=max_bytes))
        kind = self.kind(args, capture_to)
        if kind in self.failing:
            return self.failing[kind]
        if kind == "run_json":
            return _answer(self.run)
        if kind == "log":
            return _capture(capture_to, self.log, max_bytes)
        if kind == "repo":
            return _answer({"nameWithOwner": "/".join(self.repo_url.split("/")[-2:]), "url": self.repo_url})
        if kind == "artifacts":
            return _answer({"total_count": len(self.artifacts) if self.total is None else self.total, "artifacts": self.artifacts})
        if kind == "artifact":
            return _answer(self.artifact)
        return _capture(capture_to, self.archive, max_bytes)

    @staticmethod
    def kind(args, capture_to):
        if args[:2] == ["run", "view"]:
            return "log" if capture_to is not None else "run_json"
        if args[:2] == ["repo", "view"]:
            return "repo"
        if args[0] == "api":
            return "zip" if args[1].endswith("/zip") else "artifacts" if "/runs/" in args[1] else "artifact"
        raise AssertionError(args)

    def kinds(self):
        return [self.kind(call.args, call.capture_to) for call in self.calls]


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "fixture-token")

    def install(**kwargs):
        gh = FakeGh(**kwargs)
        monkeypatch.setattr(github_workflows, "_gh_run", gh)
        return gh
    return install


@pytest.fixture
def ctx(tmp_path):
    context, _project = _context(tmp_path, "queued")
    context._active_builtin_tool_result = None  # Handler-level calls publish their typed result here.
    return context


def _read(ctx, method, kind, **args):
    args = dict(args)
    for old in ("run_id", "artifact_id"):
        if old in args:
            args["target"] = args.pop(old)
    if method == "get":
        args.setdefault("include", ["log"])
    if method == "list":
        args.setdefault("limit", 100)  # The original fixtures exercise 100-record pages.
    if "page" in args:
        args["cursor"] = str(args.pop("page") or "")
    entry = next(entry for entry in github.get_tools() if entry.name == "github_read")
    text = entry.handler(ctx, method=method, kind=kind, **args)
    if "\n{" not in text:
        return text
    line, _, body = text.partition("\n")
    payload = json.loads(body)
    if "saved" in payload:
        payload = json.loads(pathlib.Path(payload["saved"]["absolute_path"]).read_text())
        line = line.split(" — saved whole as task artifact", 1)[0]
    assert payload["method"] == method and payload["kind"] == kind
    assert "complete" in payload and "next_cursor" in payload
    return line + "\n" + payload["view"]


def _retained(ctx, text):
    """The retained file a result names, read from this task's artifact store."""
    path = text.split("Retained: artifact_store/", 1)[1].split(" ", 1)[0]
    return task_artifact_dir_path(canonical_data_root(ctx), "task-fixture") / path


def _typed(ctx):
    return ctx._active_builtin_tool_result


# --- github_read get run with log -----------------------------------------------------------------------------------

def test_run_log_resolves_identity_retains_exact_bytes_and_shows_the_tail(fake, ctx):
    """Paired benefit over ``gh run view 11 --log`` read as text: one call names the repository, attempt, source command
    and exact byte count, retains the bytes with their sha256 at a read_file address, and shows the tail."""
    gh = fake()

    text = _read(ctx, "get", "run", run_id=11, repo="")

    assert gh.kinds() == ["run_json", "log"]
    assert gh.calls[1].args == ["run", "view", "11", "--log", "--attempt", "1"], "pinned to the attempt the metadata read observed"
    assert gh.calls[1].repo == "" and gh.calls[1].max_bytes == github_workflows.LOG_MAX_BYTES_DEFAULT
    assert text.startswith("GitHub Actions log — owner/selected run 11 attempt 1\n")
    assert f"Source: gh run view 11 --log --attempt 1; captured {len(LOG)} bytes exactly\n" in text
    assert hashlib.sha256(LOG).hexdigest() in text
    assert _retained(ctx, text).read_bytes() == LOG, "the retained file holds the bytes as captured, CRLF and final newline included"
    assert "Lines: 2; showing the last 2" in text
    assert text.endswith("---\n" + LOG.decode().replace("\r\n", "\n").rstrip("\n"))
    assert (_typed(ctx).status, _typed(ctx).code, _typed(ctx).meta["operation_outcome"]) == ("ok", "OK", "completed_no_effect")


def test_run_log_job_validates_membership_and_reads_by_job_id(fake, ctx):
    gh = fake()
    text = _read(ctx, "get", "run", run_id=11, job_id=55, failed_only=True, repo="github.example/owner/selected")
    assert gh.calls[1].args == ["run", "view", "--log-failed", "--job", "55", "--attempt", "1"]
    assert gh.calls[1].repo == "github.example/owner/selected"
    assert text.startswith("GitHub Actions log — owner/selected run 11 attempt 1, job 55 test (failure)\n")


def test_run_log_retention_failure_is_typed_incomplete_with_capture_facts(fake, ctx, monkeypatch):
    gh = fake()
    monkeypatch.setattr(github_workflows, "_retain", lambda *_a, **_k: "OSError: store unavailable")
    text = _read(ctx, "get", "run", run_id=11)
    result = _typed(ctx)
    assert gh.kinds() == ["run_json", "log"]
    assert (result.status, result.code, result.meta["operation_outcome"]) == ("error", "TOOL_ERROR", "incomplete")
    assert "run 11 attempt 1" in text and "Source: gh run view 11 --log --attempt 1" in text
    assert f"captured {len(LOG)} bytes exactly, sha256 {hashlib.sha256(LOG).hexdigest()}" in text
    assert "NOT retained (OSError: store unavailable)" in text


def test_run_log_refuses_a_job_outside_the_run_before_any_download(fake, ctx):
    gh = fake()
    text = _read(ctx, "get", "run", run_id=11, job_id=99)
    assert gh.kinds() == ["run_json"]
    assert text == "⚠️ TOOL_ARG_ERROR: job 99 is not a job of run 11 attempt 1 in owner/selected; its jobs: 55 test."
    assert (_typed(ctx).status, _typed(ctx).code) == ("error", "TOOL_ARG_ERROR")


def test_run_log_of_an_unfinished_run_is_refused_without_a_download(fake, ctx):
    gh = fake(run={"databaseId": 11, "url": f"{BASE}/actions/runs/11", "status": "in_progress", "attempt": 1, "jobs": []})
    text = _read(ctx, "get", "run", run_id=11)
    assert gh.kinds() == ["run_json"]
    assert text == ("⚠️ TOOL_ERROR: run 11 attempt 1 in owner/selected is in_progress; GitHub serves the log of a completed run "
                    "only, so no log was requested.")


def test_run_log_attempt_is_read_and_passed_to_the_capture(fake, ctx):
    gh = fake(run={"databaseId": 11, "url": f"{BASE}/actions/runs/11", "status": "completed", "conclusion": "success",
                   "attempt": 2, "headSha": SHA, "workflowName": "CI", "jobs": []})
    text = _read(ctx, "get", "run", run_id=11, attempt=2)
    assert gh.calls[0].args == ["run", "view", "11", "--attempt", "2", "--json", github_workflows._RUN_FIELDS + ",jobs"]
    assert gh.calls[1].args == ["run", "view", "11", "--log", "--attempt", "2"]
    assert "run 11 attempt 2" in text


@pytest.mark.parametrize("run,attempt,fragment", [
    ({"databaseId": 12, "attempt": 1}, 0, "GitHub's answer for run 11 names run 12 attempt 1; the run was not read."),
    ({"databaseId": 11}, 0, "GitHub's answer for run 11 names run 11 attempt None; the run was not read."),
    ({"databaseId": 11, "attempt": 0}, 0, "names run 11 attempt 0; the run was not read."),
    ({"databaseId": 11, "attempt": 1}, 2, "GitHub's answer for run 11 attempt 2 names run 11 attempt 1; the run was not read."),
])
def test_run_log_never_invents_the_run_or_attempt_it_reads(fake, ctx, run, attempt, fragment):
    """A later re-run must not relabel bytes: the observed id and attempt are validated, never defaulted to 1."""
    gh = fake(run={"url": f"{BASE}/actions/runs/11", "status": "completed", "jobs": [], **run})
    text = _read(ctx, "get", "run", run_id=11, attempt=attempt)
    assert gh.kinds() == ["run_json"] and text.startswith("⚠️ TOOL_ERROR: ") and fragment in text
    assert (_typed(ctx).status, _typed(ctx).code) == ("error", "TOOL_ERROR")


def test_run_log_cap_retains_the_first_bytes_and_says_the_log_is_longer(fake, ctx):
    gh = fake(log=b"line\n" * 1000)
    text = _read(ctx, "get", "run", run_id=11, max_bytes=1234)
    assert gh.calls[1].max_bytes == 1234
    assert "captured 1234 bytes exactly; the 1234-byte cap ended the capture: the log is longer and this is its first 1234 bytes" in text
    assert _retained(ctx, text).read_bytes() == b"line\n" * 246 + b"line"


def test_run_log_empty_failed_steps_log_retains_nothing(fake, ctx):
    fake(log=b"")
    text = _read(ctx, "get", "run", run_id=11, failed_only=True)
    assert "The log is empty: GitHub lists no failed step here, so --log-failed prints nothing. Nothing was retained." in text
    assert "Retained:" not in text


def test_run_log_tail_is_bounded_by_the_result_limit_and_clips_long_lines(fake, ctx, monkeypatch):
    monkeypatch.setattr(github_workflows, "tool_result_limit", lambda name: 2000)
    fake(log=(b"x" * 900 + b"\n") * 50)
    text = _read(ctx, "get", "run", run_id=11, tail_lines=50)
    assert len(text) <= 2000
    shown = int(text.split("showing the last ", 1)[1].split(" ", 1)[0])
    assert 1 <= shown < 50 and f"Lines: 50; showing the last {shown} (each clipped to 400 chars)." in text
    assert text.endswith("x" * 399 + "…") and text.count("…") == shown


@pytest.mark.parametrize("kwargs,fragment", [
    ({"run_id": 0}, "target names the workflow run"),
    ({"run_id": "eleven"}, "whole numbers"),
    ({"run_id": 11, "failed_only": "yes"}, "failed_only is a boolean"),
    ({"run_id": 11, "max_bytes": 0}, "max_bytes is 1.."),
    ({"run_id": 11, "max_bytes": github_workflows.LOG_MAX_BYTES_CEILING + 1}, "max_bytes is 1.."),
    ({"run_id": 11, "tail_lines": 0}, "tail_lines is 1.."),
    ({"run_id": 11, "job_id": -1}, "job_id and attempt are 0"),
])
def test_run_log_argument_refusals_launch_nothing(fake, ctx, kwargs, fragment):
    gh = fake()
    text = _read(ctx, "get", "run", **kwargs)
    assert fragment in text and gh.calls == []
    assert (_typed(ctx).code, _typed(ctx).meta["operation_outcome"]) == ("TOOL_ARG_ERROR", "completed_no_effect")


def test_run_log_malformed_json_and_transport_failures_are_reported_not_guessed(fake, ctx):
    gh = fake(failing={"run_json": github.GhResult(True, "not json", 0, None, "")})
    assert _read(ctx, "get", "run", run_id=11).startswith("⚠️ TOOL_ERROR: failed to parse the workflow run JSON")
    forbidden = github.GhResult(False, "⚠️ GH_ERROR: gh: Resource not accessible (HTTP 403)", 1, 403, "exit")
    gh = fake(failing={"log": forbidden})
    assert _read(ctx, "get", "run", run_id=11) == forbidden.text
    assert gh.kinds() == ["run_json", "log"]


# --- github_read list artifact -------------------------------------------------------------------------------------

_DIST = {"id": 501, "name": "dist", "size_in_bytes": 2048, "expired": False, "created_at": "2026-10-20T10:00:00Z",
         "expires_at": "2027-01-18T10:00:00Z", "digest": "sha256:" + "0" * 64, "workflow_run": {"id": 11, "head_sha": SHA}}


def test_artifact_list_identifies_each_artifact_by_repository_and_id(fake, ctx):
    """Paired benefit over ``gh api …/artifacts``: the raw dump has names and ids but no repository; this call reads the
    run first, names the repository GitHub returned, and gives the exact id-addressed download call per artifact."""
    gh = fake(artifacts=[_DIST, {**_DIST, "id": 502, "name": "dist", "expired": True}])
    text = _read(ctx, "list", "artifact", run_id=11)
    assert gh.kinds() == ["run_json", "artifacts"]
    assert gh.calls[1].args == ["api", "repos/owner/selected/actions/runs/11/artifacts?per_page=100&page=1", "--hostname", "github.example"]
    assert gh.calls[1].repo is github._GENERIC_TRANSPORT, "the literal API path carries the repository GitHub named"
    assert text.startswith("GitHub Actions artifacts — owner/selected run 11: 2; page 1 of 1 (2 listed)\n")
    assert "Identity is repository + artifact id: the same name may exist in other runs and repositories." in text
    assert f"- id 501: dist — 2048 bytes; created 2026-10-20T10:00:00Z; expires 2027-01-18T10:00:00Z; head {SHA}; digest sha256:" in text
    assert "    github_read(method='download', kind='artifact', target=501, max_bytes=2048)" in text
    assert "- id 502: dist — 2048 bytes, expired;" in text and "    expired: GitHub no longer serves it" in text


def test_artifact_list_of_the_same_name_in_two_repositories_gives_two_addresses(fake, ctx):
    first = fake(artifacts=[_DIST])
    one = _read(ctx, "list", "artifact", run_id=11)
    second = fake(run={"databaseId": 12, "url": f"{OTHER}/actions/runs/12", "status": "completed", "attempt": 1},
                  artifacts=[{**_DIST, "id": 777}])
    two = _read(ctx, "list", "artifact", run_id=12, repo="github.example/other/thing")
    assert first.calls[1].args[1] == "repos/owner/selected/actions/runs/11/artifacts?per_page=100&page=1"
    assert second.calls[1].args[1] == "repos/other/thing/actions/runs/12/artifacts?per_page=100&page=1"
    assert "github_read(method='download', kind='artifact', target=501, max_bytes=2048)" in one
    assert "github_read(method='download', kind='artifact', target=777, max_bytes=2048, repo='github.example/other/thing')" in two


def test_artifact_list_is_run_scoped_even_when_latest_attempt_changes(fake, ctx):
    gh = fake(artifacts=[_DIST])
    first = _read(ctx, "list", "artifact", run_id=11)
    gh.run = {**gh.run, "attempt": 2}
    second = _read(ctx, "list", "artifact", run_id=11)
    assert gh.calls[1].args == gh.calls[3].args
    assert "owner/selected run 11:" in first and "owner/selected run 11:" in second
    assert "attempt 1" not in first and "attempt 2" not in second
    assert "Identity is repository + artifact id" in second


def test_artifact_list_needs_run_identity_without_attempt_metadata(fake, ctx):
    gh = fake(run={"databaseId": 11, "url": f"{BASE}/actions/runs/11", "status": "completed"}, artifacts=[_DIST])
    text = _read(ctx, "list", "artifact", run_id=11)
    assert gh.calls[0].args == ["run", "view", "11", "--json", github_workflows._RUN_ARTIFACT_FIELDS]
    assert gh.kinds() == ["run_json", "artifacts"]
    assert "owner/selected run 11: 1" in text and "- id 501: dist" in text
    assert (_typed(ctx).status, _typed(ctx).meta["operation_outcome"]) == ("ok", "completed_no_effect")


def test_artifact_list_pages_explicitly_and_names_the_next_page(fake, ctx):
    """Paired benefit over a raw first page: every remaining record is reachable by an explicit page."""
    gh = fake(artifacts=[_DIST], total=150)
    assert "run 11: 150; page 1 of 2 (1 listed; page 2 holds the next)" in _read(ctx, "list", "artifact", run_id=11)
    text = _read(ctx, "list", "artifact", run_id=11, page=2)
    assert gh.calls[-1].args[1] == "repos/owner/selected/actions/runs/11/artifacts?per_page=100&page=2"
    assert "150; page 2 of 2 (1 listed)" in text and "page 3" not in text
    gh = fake(artifacts=[], total=1)
    assert _read(ctx, "list", "artifact", run_id=11, page=2).endswith("Page 2 lists no artifact (GitHub holds 1).")
    assert "page=2" in gh.calls[-1].args[1]
    fake(artifacts=[])
    assert _read(ctx, "list", "artifact", run_id=11).endswith("No artifact is registered for this run.")
    gh = fake(artifacts=[_DIST])
    assert "cursor starts at page 1" in _read(ctx, "list", "artifact", run_id=11, page=-1) and gh.calls == []
    assert "whole page number" in _read(ctx, "list", "artifact", run_id=11, page="two") and gh.calls == []
    assert "page 1 of 1" in _read(ctx, "list", "artifact", run_id=11, page=0), "0 reads as omitted: the first page"


def test_artifact_list_refusals(fake, ctx):
    gh = fake(failing={"artifacts": github.GhResult(True, "[]", 0, None, "")})
    assert _read(ctx, "list", "artifact", run_id=11).startswith("⚠️ TOOL_ERROR: failed to parse the artifacts JSON")
    assert _read(ctx, "list", "artifact", run_id=0) == ("⚠️ TOOL_ARG_ERROR: target is the positive workflow run number; "
                                                                  "cursor is a whole page number.")
    assert gh.kinds() == ["run_json", "artifacts"]


# --- github_read download artifact ---------------------------------------------------------------------------------

ARCHIVE = _zip({"report.xml": b"<testsuite/>", "logs/pytest.txt": b"1 failed\n"})


def test_artifact_download_retains_the_archive_unextracted_with_hash_and_listing(fake, ctx, tmp_path):
    """Paired benefit over ``gh run download -n dist``: the baseline selects by NAME and EXTRACTS into the working
    directory; this call selects by repository + id, bounds the download, keeps the archive as served with its sha256
    and lists its members without writing one of them anywhere."""
    gh = fake(artifact={**_DIST, "digest": "sha256:" + hashlib.sha256(ARCHIVE).hexdigest()}, archive=ARCHIVE)

    text = _read(ctx, "download", "artifact", artifact_id=501, max_bytes=1 << 20, repo="")

    assert gh.kinds() == ["repo", "artifact", "zip"]
    assert gh.calls[0].repo == "" and gh.calls[1].repo is gh.calls[2].repo is github._GENERIC_TRANSPORT
    assert gh.calls[2].args == ["api", "repos/owner/selected/actions/artifacts/501/zip", "--hostname", "github.example"]
    assert gh.calls[2].max_bytes == 1 << 20
    assert text.startswith(f"GitHub Actions artifact — owner/selected artifact 501 'dist' (run 11, head {SHA}); created ")
    assert f"Archive: {len(ARCHIVE)} bytes, sha256 {hashlib.sha256(ARCHIVE).hexdigest()}; GitHub's digest sha256:" in text and " matches these bytes" in text
    retained = task_artifact_dir_path(canonical_data_root(ctx), "task-fixture") / text.split("not extracted: artifact_store/", 1)[1].split(" ", 1)[0]
    assert retained.read_bytes() == ARCHIVE and retained.suffix == ".zip"
    assert "Members (2, 21 bytes uncompressed; names and sizes are read from the archive directory and are untrusted):" in text
    assert "  report.xml (12 bytes)\n  logs/pytest.txt (9 bytes)" in text
    extracted = [p for p in tmp_path.rglob("report.xml")] + [p for p in tmp_path.rglob("pytest.txt")]
    assert extracted == [], "no member was extracted anywhere"
    assert _typed(ctx).meta["operation_outcome"] == "completed_no_effect"


def test_artifact_download_digest_mismatch_is_stated_not_hidden(fake, ctx):
    fake(artifact={**_DIST, "digest": "sha256:" + "f" * 64}, archive=ARCHIVE)
    text = _read(ctx, "download", "artifact", artifact_id=501, max_bytes=4096)
    assert "GitHub's digest sha256:" + "f" * 64 + " differs from these bytes" in text


def test_artifact_download_retention_failure_is_typed_incomplete_with_archive_facts(fake, ctx, monkeypatch):
    gh = fake(artifact=_DIST, archive=ARCHIVE)
    monkeypatch.setattr(github_workflows, "_retain", lambda *_a, **_k: "OSError: store unavailable")
    text = _read(ctx, "download", "artifact", artifact_id=501, max_bytes=4096)
    result = _typed(ctx)
    assert gh.kinds() == ["repo", "artifact", "zip"]
    assert (result.status, result.code, result.meta["operation_outcome"]) == ("error", "TOOL_ERROR", "incomplete")
    assert "owner/selected artifact 501 'dist' (run 11" in text
    assert f"Archive: {len(ARCHIVE)} bytes, sha256 {hashlib.sha256(ARCHIVE).hexdigest()}" in text
    assert "NOT retained (OSError: store unavailable)" in text


def test_artifact_download_refuses_before_downloading_when_too_large_or_expired(fake, ctx):
    gh = fake(artifact={**_DIST, "size_in_bytes": 5000})
    text = _read(ctx, "download", "artifact", artifact_id=501, max_bytes=4096)
    assert gh.kinds() == ["repo", "artifact"], "no zip request was sent"
    assert text == ("⚠️ TOOL_ARG_ERROR: artifact 501 'dist' of owner/selected is 5000 bytes, above max_bytes=4096; nothing was "
                    f"downloaded. Raise max_bytes deliberately (ceiling {github_workflows.ARTIFACT_MAX_BYTES_CEILING}).")
    assert _typed(ctx).meta["operation_outcome"] == "completed_no_effect"
    gh = fake(artifact={**_DIST, "expired": True})
    text = _read(ctx, "download", "artifact", artifact_id=501, max_bytes=4096)
    assert gh.kinds() == ["repo", "artifact"]
    assert text.startswith("⚠️ TOOL_ERROR: artifact 501 'dist' of owner/selected expired (2027-01-18T10:00:00Z); GitHub no longer serves it")


def test_artifact_download_over_the_cap_discards_the_partial_archive(fake, ctx):
    """GitHub stated a size under the cap but served more: the stream stopped at the cap and nothing is retained."""
    gh = fake(artifact={**_DIST, "size_in_bytes": 10}, archive=ARCHIVE)
    text = _read(ctx, "download", "artifact", artifact_id=501, max_bytes=100)
    assert gh.calls[2].max_bytes == 100
    assert text == ("⚠️ TOOL_ERROR: the download of artifact 501 'dist' of owner/selected exceeded max_bytes=100 (GitHub stated "
                    "10 bytes); the partial archive was discarded and nothing was retained.")
    store = task_artifact_dir_path(canonical_data_root(ctx), "task-fixture")
    assert not store.exists() or not list(store.rglob("*.zip"))


def test_artifact_download_non_zip_bytes_are_discarded(fake, ctx):
    fake(artifact=_DIST, archive=b'{"message":"Gone"}')
    text = _read(ctx, "download", "artifact", artifact_id=501, max_bytes=4096)
    assert text == "⚠️ TOOL_ERROR: the 18 bytes GitHub served for artifact 501 'dist' of owner/selected are not a zip archive; they were discarded."


def test_artifact_download_sends_the_explicit_repo_positionally_to_repo_view(ctx, monkeypatch):
    """Actual argv through the real transport: `gh repo view <repo>` (no --repo flag exists there), then the literal
    API path from the URL gh returned; the expired artifact stops the call before any download."""
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "fixture-token")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[1:3] == ["repo", "view"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"nameWithOwner": "other/thing", "url": OTHER}), stderr="")
        return SimpleNamespace(returncode=0, stdout=json.dumps({**_DIST, "expired": True}), stderr="")
    monkeypatch.setattr(subprocess, "run", run)

    text = _read(ctx, "download", "artifact", artifact_id=501, max_bytes=4096, repo="github.example/other/thing")

    assert calls[0] == ["gh", "repo", "view", "github.example/other/thing", "--json", "nameWithOwner,url"]
    assert calls[1] == ["gh", "api", "repos/other/thing/actions/artifacts/501", "--hostname", "github.example"]
    assert len(calls) == 2 and text.startswith("⚠️ TOOL_ERROR: artifact 501 'dist' of other/thing expired")


def test_artifact_download_targets_the_repository_the_cli_resolved(fake, ctx):
    gh = fake(repo_url=OTHER, artifact=_DIST, archive=ARCHIVE)
    text = _read(ctx, "download", "artifact", artifact_id=501, max_bytes=4096, repo="github.example/other/thing")
    assert gh.calls[0].args == ["repo", "view", "--json", "nameWithOwner,url"] and gh.calls[0].repo == "github.example/other/thing"
    assert gh.calls[1].args[1] == "repos/other/thing/actions/artifacts/501"
    assert text.startswith("GitHub Actions artifact — other/thing artifact 501 'dist'")


@pytest.mark.parametrize("kwargs,fragment", [
    ({"artifact_id": 501}, "max_bytes is required"),
    ({"artifact_id": 501, "max_bytes": 0}, "max_bytes is required"),
    ({"artifact_id": 501, "max_bytes": github_workflows.ARTIFACT_MAX_BYTES_CEILING + 1}, "max_bytes is required"),
    ({"artifact_id": 0, "max_bytes": 10}, "target is the id"),
    ({"artifact_id": "x", "max_bytes": 10}, "whole numbers"),
])
def test_artifact_download_argument_refusals_launch_nothing(fake, ctx, kwargs, fragment):
    gh = fake()
    assert fragment in _read(ctx, "download", "artifact", **kwargs)
    assert gh.calls == [] and _typed(ctx).meta["operation_outcome"] == "completed_no_effect"


def test_artifact_download_malformed_or_mismatched_metadata_is_refused(fake, ctx):
    gh = fake(artifact={**_DIST, "id": 999})
    assert _read(ctx, "download", "artifact", artifact_id=501, max_bytes=10).startswith("⚠️ TOOL_ERROR: the artifact answer is malformed or names another id")
    assert gh.kinds() == ["repo", "artifact"]


# --- registration -----------------------------------------------------------------------------------------------------

def test_readers_register_through_github_get_tools_with_repo_and_skip_policy():
    from ouroboros.safety import TOOL_POLICY, POLICY_SKIP

    entries = {entry.name: entry for entry in github.get_tools()}
    assert set(entries) == {"github_read", "github_write", "pr_merge"}
    assert TOOL_POLICY["github_read"] == POLICY_SKIP and not entries["github_read"].mutates_worktree
    props = entries["github_read"].schema["parameters"]["properties"]
    assert props["repo"]["type"] == "string" and "default" not in props["max_bytes"]
    assert not hasattr(github_workflows, "tool_entries")
    assert not any(hasattr(github_workflows, name) for name in ("get_run_log", "list_run_artifacts", "download_artifact"))


def test_registered_reader_publishes_a_typed_completed_read(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "fixture-token")
    gh = FakeGh()
    monkeypatch.setattr(github_workflows, "_gh_run", gh)
    ctx, _ = _context(tmp_path, "queued")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    result = registry.execute_result("github_read", {"method": "get", "kind": "run", "target": 11, "include": ["log"], "repo": "github.example/owner/selected"})
    assert (result.status, result.code, result.meta["operation_outcome"]) == ("ok", "OK", "completed_no_effect")
    assert all(call.repo == "github.example/owner/selected" for call in gh.calls)
    refused = registry.execute_result("github_read", {"method": "get", "kind": "run", "target": 0, "include": ["log"], "repo": "github.example/owner/selected"})
    assert (refused.status, refused.code, refused.meta["operation_outcome"]) == ("error", "TOOL_ARG_ERROR", "completed_no_effect")


@pytest.mark.parametrize("name,args", [
    ("github_read", {"method": "get", "kind": "run", "target": 11, "include": ["log"]}),
    ("github_read", {"method": "download", "kind": "artifact", "target": 501, "max_bytes": 4096}),
])
def test_registered_reader_reports_retention_failure_as_incomplete(tmp_path, monkeypatch, name, args):
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "fixture-token")
    gh = FakeGh(artifact=_DIST, archive=ARCHIVE)
    monkeypatch.setattr(github_workflows, "_gh_run", gh)
    monkeypatch.setattr(github_workflows, "_retain", lambda *_a, **_k: "OSError: store unavailable")
    ctx, _ = _context(tmp_path, "queued")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    result = registry.execute_result(name, args)
    assert (result.status, result.code, result.meta["operation_outcome"]) == ("error", "TOOL_ERROR", "incomplete")
    assert "NOT retained (OSError: store unavailable)" in result.text
    assert gh.kinds() == (["run_json", "log"] if args["method"] == "get" else ["repo", "artifact", "zip"])


def test_artifact_page_preview_keeps_every_omitted_record_addressable(fake, ctx, monkeypatch):
    monkeypatch.setattr(github_workflows, "tool_result_limit", lambda _name: 2000)
    rows = [{**_DIST, "id": 500 + i, "name": "long " * 40} for i in range(100)]
    fake(artifacts=rows)
    text = _read(ctx, "list", "artifact", run_id=11)
    assert len(text) <= 2000 and "omitted from this preview" in text
    import ast
    quoted = text.split("path=", 1)[1].split("); sha256", 1)[0]
    ref = ast.literal_eval(quoted)
    retained = task_artifact_dir_path(canonical_data_root(ctx), "task-fixture") / ref
    saved = json.loads(retained.read_text())
    assert saved["artifacts"] == rows and saved["total_count"] == 100


def test_artifact_page_unknown_total_never_means_complete(fake, ctx):
    fake(failing={"artifacts": _answer({"artifacts": [_DIST]})})
    text = _read(ctx, "list", "artifact", run_id=11)
    assert "no valid total_count; completeness is unknown" in text
    assert _typed(ctx).status == "error"


@pytest.mark.parametrize("size,expired,usable", [
    (github_workflows.ARTIFACT_MAX_BYTES_CEILING, False, True),
    (github_workflows.ARTIFACT_MAX_BYTES_CEILING + 1, False, False),
    (42, True, False),
])
def test_artifact_page_only_suggests_supported_downloads(fake, ctx, size, expired, usable):
    fake(artifacts=[{"id": 99, "name": "evidence", "size_in_bytes": size, "expired": expired}])
    text = _read(ctx, "list", "artifact", run_id=11)
    assert ("github_read(method='download', kind='artifact', target=99" in text) is usable
    if expired:
        assert "expired: GitHub no longer serves it" in text
    elif not usable:
        assert "exceeds this tool's supported" in text


def test_old_job_requires_its_explicit_attempt(fake, ctx):
    gh = fake()
    first = {**gh.run, "attempt": 1}
    second = {**gh.run, "attempt": 2, "jobs": []}
    gh.run = second
    refused = _read(ctx, "get", "run", run_id=11, job_id=55)
    assert "not a job of run 11 attempt 2" in refused
    assert gh.kinds() == ["run_json"]
    gh = fake(run=first)
    text = _read(ctx, "get", "run", run_id=11, job_id=55, attempt=1)
    assert text.startswith("GitHub Actions log — owner/selected run 11 attempt 1, job 55")
    assert gh.calls[0].args[3:5] == ["--attempt", "1"]
    assert gh.calls[-1].args[-2:] == ["--attempt", "1"]


def test_run_metadata_without_log_uses_shared_projection_and_never_captures(fake, ctx):
    gh = fake()
    entry = next(entry for entry in github.get_tools() if entry.name == "github_read")
    text = entry.handler(ctx, method="get", kind="run", target=11, fields=["databaseId", "headSha"])
    payload = json.loads(text.partition("\n")[2])
    assert payload["item"] == {"databaseId": 11, "headSha": SHA}
    assert payload["complete"] is True and payload["next_cursor"] == "" and "capture" not in payload
    assert gh.kinds() == ["run_json"]


def test_artifact_page_uses_shared_cursor_limit_projection_and_explicit_filter_note(fake, ctx):
    gh = fake(artifacts=[_DIST], total=7)
    entry = next(entry for entry in github.get_tools() if entry.name == "github_read")
    text = entry.handler(ctx, method="list", kind="artifact", target=11, cursor="2", limit=2,
                         fields=["id"], since="2026-01-01T00:00:00Z")
    payload = json.loads(text.partition("\n")[2])
    assert payload["items"] == [{"id": _DIST["id"]}]
    assert payload["complete"] is False and payload["next_cursor"] == "3"
    assert "per_page=2&page=2" in gh.calls[-1].args[1]
    assert payload["omitted"] == ["artifact listing has no since filter; this page is unfiltered"]


def test_shared_cut_manifest_preserves_entire_artifact_page_and_cursor(fake, ctx, monkeypatch):
    from ouroboros.tools import github_read
    monkeypatch.setattr(github_read, "tool_result_limit", lambda _: 2000)
    rows = [{**_DIST, "id": 500 + i, "name": "long " * 40} for i in range(100)]
    fake(artifacts=rows, total=150)
    entry = next(entry for entry in github.get_tools() if entry.name == "github_read")
    text = entry.handler(ctx, method="list", kind="artifact", target=11, limit=100, detail="full")
    manifest = json.loads(text.partition("\n")[2])
    data = pathlib.Path(manifest["saved"]["absolute_path"]).read_bytes()
    saved = json.loads(data)
    assert saved["items"] == rows and saved["next_cursor"] == "2" and not saved["complete"]
    assert manifest["summary"]["next_cursor"] == "2" and manifest["summary"]["items"] == 100
    assert len(data) == manifest["saved"]["bytes"] and hashlib.sha256(data).hexdigest() == manifest["saved"]["sha256"]


def test_gh_run_readers_add_run_identity_exact_capture_and_artifact_id(tmp_path, monkeypatch):
    from tests.test_github_project_target import _context

    ctx, _project = _context(tmp_path, "queued")
    ctx._active_builtin_tool_result = None
    driver = FakeGh(artifacts=[_DIST], artifact=_DIST, archive=_zip({"out.txt": b"ok"}))
    monkeypatch.setattr(github_workflows, "_gh_run", driver)
    listing = _read(ctx, "list", "artifact", run_id=11)
    assert "owner/selected run 11:" in listing and "github_read(method='download', kind='artifact', target=501" in listing
    assert "attempt 1" not in listing  # artifact API is run-scoped, not attempt-scoped
    log = _read(ctx, "get", "run", run_id=11)
    assert "run 11 attempt 1" in log and hashlib.sha256(LOG).hexdigest() in log
    assert driver.calls[-1].args == ["run", "view", "11", "--log", "--attempt", "1"]
    assert _retained(ctx, log).read_bytes() == LOG
    downloaded = _read(ctx, "download", "artifact", artifact_id=501, max_bytes=4096)
    assert "artifact 501" in downloaded and "out.txt" in downloaded
    assert driver.kinds()[-3:] == ["repo", "artifact", "zip"]
    assert "Retained as the archive" in downloaded


def test_gh_readers_reach_readonly_child_that_cannot_use_shell(tmp_path, monkeypatch):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_github_project_target import _context

    ctx, _project = _context(tmp_path, "queued")
    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent", allow_enable=False)
    registry = ToolRegistry(ctx.repo_dir, ctx.drive_root)
    registry.set_context(ctx)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: (True, ""))
    monkeypatch.setattr(github, "github_cli_configured", lambda: True)
    driver = FakeGh(artifacts=[_DIST], artifact=_DIST, archive=_zip({"out.txt": b"ok"}))
    monkeypatch.setattr(github_workflows, "_gh_run", driver)
    assert registry.get_schema_by_name("run_command") is None
    for name, args, content in (
        ("github_read", {"method": "get", "kind": "run", "target": 11, "include": ["log"]}, "run 11 attempt 1"),
        ("github_read", {"method": "list", "kind": "artifact", "target": 11, "limit": 100}, "target=501"),
        ("github_read", {"method": "download", "kind": "artifact", "target": 501, "max_bytes": 4096}, "out.txt"),
    ):
        assert registry.get_schema_by_name(name) is not None
        result = registry.execute_result(name, args)
        assert result.status == "ok" and content in result.text, (name, result.text)
