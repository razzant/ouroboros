"""Owner decision 1A (2026-10-21): no tag waives a release gate, and the notes and the
proof disclose the run's FLAKY and ACCEPTED tests from the `test-verdict` table."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import pytest
import yaml

from tests.test_release_proof import GREEN_GATES, REPO, _fixture_release, _gate_env, _verdict, release_proof

RED = {"full-test": "failure", "system-e2e-mock": "cancelled", "ui-smoke": "failure"}


def _assemble_args(tmp_path, version, results, verdict):
    release_dir, version_file, readme = _fixture_release(tmp_path, version=version)
    return release_dir, argparse.Namespace(
        directory=release_dir, version_file=version_file, readme=readme, repository="razzant/ouroboros",
        tag=f"v{version}", commit="a" * 40, run_url="https://example.test/run/7", previous_tag=None,
        generated_at=None, notes_output=tmp_path / "notes.md", android_build_result="skipped",
        android_attestation_result="not_run", windows_signer_thumbprint="", github_output=None,
        test_verdict=verdict,
        gate_result=[f"{name}={results.get(name, 'success')}" for name in release_proof.RELEASE_GATES])


@pytest.mark.parametrize(("version", "results"), [
    # No tag, however official, may ship a red lane any more: v7.7.1's one-off exception is gone.
    ("7.7.1", RED),
    ("7.7.2", {"full-test": "failure"}),
    ("7.7.2", {"lint": "failure"}),
    ("7.7.2", {"test-verdict": "failure"}),
    ("7.7.2", {"windows-proof": "cancelled"}),
    ("7.7.2", {"ui-smoke": "skipped"}),
])
def test_every_release_gate_must_succeed(tmp_path, version, results):
    release_dir, args = _assemble_args(tmp_path, version, results, _verdict(tmp_path))
    name, result = next(iter(results.items()))
    with pytest.raises(ValueError, match=f"{name} concluded {result}"):
        release_proof.command_assemble(args)
    assert not (release_dir / "SHA256SUMS").exists() and not args.notes_output.exists()


def test_green_gates_record_the_run_verdict_and_disclose_it_in_the_notes(tmp_path):
    flaky = [{"nodeid": "tests/test_a.py::test_once", "label": "full-test-windows-latest-shard-3/parallel"}]
    accepted = [{"nodeid": "tests/test_b.py::test_known", "issue": 2083, "label": "journal-persist-macos-latest/serial"}]
    release_dir, args = _assemble_args(tmp_path, "7.7.2", {}, _verdict(tmp_path, flaky=flaky, accepted=accepted))
    release_proof.command_assemble(args)
    workflow = json.loads((release_dir / "release-evidence.json").read_text(encoding="utf-8"))["workflow"]
    assert workflow["gates"][:len(GREEN_GATES)] == [
        {"name": name, "status": "success"} for name in release_proof.RELEASE_GATES]
    assert workflow["testVerdict"] == {"flaky": flaky, "accepted": accepted}
    assert "ownerException" not in workflow
    text = args.notes_output.read_text(encoding="utf-8")
    assert "passed the full test matrix" in text
    assert "`tests/test_b.py::test_known` — [#2083](https://github.com/razzant/ouroboros/issues/2083)" in text
    assert "1 test(s) failed once and passed on their retry in the same job (flaky):" in text
    assert "`tests/test_a.py::test_once` (full-test-windows-latest-shard-3/parallel)" in text


def test_a_quiet_run_adds_no_test_section_and_long_flaky_lists_are_capped(tmp_path):
    release_dir, args = _assemble_args(tmp_path, "7.7.2", {}, _verdict(tmp_path))
    release_proof.command_assemble(args)
    text = args.notes_output.read_text(encoding="utf-8")
    assert "flaky" not in text and "accepted as not product bugs" not in text
    many = [{"nodeid": f"tests/test_m.py::test_{index}"} for index in range(release_proof.NOTES_FLAKY_SHOWN + 5)]
    second = tmp_path / "second"
    second.mkdir()
    release_dir, args = _assemble_args(second, "7.7.2", {}, _verdict(second, flaky=many))
    release_proof.command_assemble(args)
    text = args.notes_output.read_text(encoding="utf-8")
    assert f"test_{release_proof.NOTES_FLAKY_SHOWN - 1}`" in text
    assert f"test_{release_proof.NOTES_FLAKY_SHOWN}`" not in text
    assert "… and 5 more, listed in `release-evidence.json`." in text


@pytest.mark.parametrize(("table", "message"), [
    ({"failed": [{"nodeid": "tests/test_c.py::test_red"}]}, "1 failed and 0 never-run tests"),
    ({"not_run": [{"nodeid": "tests/test_c.py::test_lost"}]}, "0 failed and 1 never-run tests"),
    ({"flaky": "tests/test_c.py::test_red"}, "no valid 'flaky' list"),
    ({"accepted": None}, "no valid 'accepted' list"),
])
def test_a_red_or_malformed_verdict_refuses_the_release(tmp_path, table, message):
    rows = {"flaky": [], "accepted": [], "failed": [], "not_run": [], **table}
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    release_dir, args = _assemble_args(tmp_path, "7.7.2", {}, path)
    with pytest.raises(ValueError, match=re.escape(message)):
        release_proof.command_assemble(args)
    assert not (release_dir / "SHA256SUMS").exists()


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda gates: gates[:-1], "missing release gate results"),
    (lambda gates: [], "missing release gate results"),
    (lambda gates: [*gates, gates[0]], "invalid release gate result"),
    (lambda gates: ["full-test=", *gates[1:]], "invalid release gate result"),
    (lambda gates: ["full-test=passed", *gates[1:]], "invalid release gate result"),
    (lambda gates: [*gates, "packaged-artifact-smoke=success"], "invalid release gate result"),
])
def test_missing_or_unrecognised_gate_results_never_become_success(mutate, message):
    with pytest.raises(ValueError, match=message):
        release_proof._gate_results(mutate(list(GREEN_GATES)))


@pytest.mark.parametrize("results", [RED, {}])
def test_release_step_passes_actual_results_and_the_run_verdict(tmp_path, results):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("workflow shell execution requires bash")
    jobs = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
    step = next(step for step in jobs["release"]["steps"]
                if step.get("name") == "Assemble release proof capsule and notes")
    _fixture_release(tmp_path, version="7.7.1")[0].rename(tmp_path / "release-artifacts")
    _verdict(tmp_path, flaky=[{"nodeid": "tests/test_a.py::test_once"}])
    script = step["run"].replace("python scripts/release_proof.py", " ".join(map(shlex.quote, [
        Path(sys.executable).as_posix(), (REPO / "scripts/release_proof.py").as_posix()])))
    env = dict(os.environ, GITHUB_OUTPUT=str(tmp_path / "output"), WINDOWS_SIGNER_THUMBPRINT="",
               GITHUB_REF_NAME="v7.7.1", GITHUB_SHA="a" * 40, GITHUB_REPOSITORY="razzant/ouroboros",
               GITHUB_SERVER_URL="https://example.test", GITHUB_RUN_ID="123",
               ANDROID_BUILD_RESULT="skipped", ANDROID_ATTESTATION_RESULT="not_run",
               RELEASE_GATE_RESULTS=_gate_env(jobs, results))
    done = subprocess.run([bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c",
                           "git() { echo v7.7.0; }\n" + script], cwd=tmp_path, env=env,
                          capture_output=True, text=True)
    evidence = tmp_path / "release-artifacts" / "release-evidence.json"
    if results:
        assert done.returncode != 0 and "full-test concluded failure" in done.stderr
        assert not evidence.exists()
        return
    assert done.returncode == 0, done.stderr
    workflow = json.loads(evidence.read_text(encoding="utf-8"))["workflow"]
    assert workflow["testVerdict"]["flaky"] == [{"nodeid": "tests/test_a.py::test_once"}]
    assert "`tests/test_a.py::test_once`" in (tmp_path / "release-notes.md").read_text(encoding="utf-8")


def test_the_release_reads_the_newest_attempts_verdict(tmp_path):
    """A re-run that turned a red lane green uploads a new table; the first attempt's red one stays
    among the run's artifacts and must not be what the release reads."""
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("workflow shell execution requires bash")
    jobs = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
    step = next(step for step in jobs["release"]["steps"] if step.get("name") == "Keep the newest attempt's verdict")
    for attempt, rows in (("2", {"failed": []}), ("10", {"failed": []}), ("1", {"failed": [{"nodeid": "t::red"}]})):
        table = tmp_path / "release-verdict" / "attempts" / f"test-verdict-123-{attempt}" / "test-verdict.json"
        table.parent.mkdir(parents=True)
        table.write_text(json.dumps({**rows, "attempt": attempt}), encoding="utf-8")
    done = subprocess.run([bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
                          cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads((tmp_path / "release-verdict" / "test-verdict.json").read_text(encoding="utf-8"))[
        "attempt"] == "10"
    # Without any table the step fails rather than letting the release read nothing.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert subprocess.run([bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
                          cwd=empty, capture_output=True, text=True).returncode != 0
