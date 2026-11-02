"""The selected CLI runs explicit modules through both real pytest lanes."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ouroboros import test_selection
from tests.test_preflight_runner import _make_repo


@pytest.mark.serial
def test_selected_cli_skips_unrelated_collection_and_records_both_lanes(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("selected_cli", Path(__file__).resolve().parents[1] / "scripts/run_tests.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    repo = _make_repo(tmp_path, {
        ".gitignore": "__pycache__/\n.pytest_cache/\n",
        "tests/conftest.py": "def pytest_addoption(parser): parser.addoption('--ci-evidence-dir')\n",
        "tests/test_floor.py": "def test_floor(): assert True\n",
        "leaf.py": "value = 1\n",
        "tests/test_leaf.py": "from leaf import value\ndef test_leaf(): assert value == 2\n",
        "tests/test_unrelated.py": "raise AssertionError('must not collect')\n",
    })
    monkeypatch.setattr(cli, "REPO", repo)
    monkeypatch.setattr(test_selection, "FLOOR_TESTS", {"tests/test_floor.py": "minimal witness"})
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "2")
    monkeypatch.setenv("OURO_TEST_WORKER_BUDGET", "2")  # nested slots: the selected passes run two real workers
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    node = []
    monkeypatch.setattr(cli, "_node_lane", lambda: node.append(True) or 0)
    (repo / "leaf.py").write_text("value = 2\n", encoding="utf-8")
    evidence = tmp_path / "evidence"
    github_output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    args = ["--selected", "--base", "HEAD", "--evidence-dir", str(evidence)]
    assert cli.main(args) == 0
    row = json.loads((evidence / "selection.json").read_text(encoding="utf-8"))
    assert row["tier"] == "selected" and row["expected_files"] == ["tests/test_floor.py", "tests/test_leaf.py"]
    assert row["omitted_files"] == 1 and set(row["inventory"]) == {"parallel", "serial"}
    assert sorted(row["inventory"]["parallel"]["executed"]) == sorted(row["inventory"]["parallel"]["expected"])
    assert "selected: 2/3 Python modules" in capsys.readouterr().out
    assert node == [True]
    assert row["outcomes"] == {"parallel": "success", "serial": "success"}
    assert github_output.read_text(encoding="utf-8").splitlines() == ["parallel_outcome=success", "serial_outcome=success"]
    # The same entry point escalates a verifier change and exposes the hidden failure.
    with (repo / "pytest.ini").open("a", encoding="utf-8") as stream:
        stream.write("\n# changed verifier\n")
    assert cli.main(args) != 0
    assert json.loads((evidence / "selection.json").read_text(encoding="utf-8"))["tier"] == "full"
    assert github_output.read_text(encoding="utf-8").splitlines()[-2:] == ["parallel_outcome=failure", "serial_outcome=skipped"]
    assert node == [True, True]


@pytest.mark.serial
@pytest.mark.skipif(os.name == "nt", reason="the fixture links the real package; Windows symlinks need privileges")
def test_cli_wrapper_owns_a_selected_run_and_keeps_its_full_inventory(tmp_path):
    """The real `python scripts/run_tests.py --selected` path: safe_test owns it, CI's REQUIRE
    controls survive its scrub while a credential does not, and both lanes' complete native
    inventories stay in durable evidence after the scratch is released."""
    from ouroboros.test_evidence import diagnostic_directory

    real = Path(__file__).resolve().parents[1]
    files = {path: "def test_floor():\n    assert True\n" for path in test_selection.FLOOR_TESTS}
    files.update({
        ".gitignore": "__pycache__/\n.pytest_cache/\nouroboros\n",
        "scripts/run_tests.py": (real / "scripts/run_tests.py").read_text(encoding="utf-8"),
        "scripts/safe_test.py": (real / "scripts/safe_test.py").read_text(encoding="utf-8"),
        "tests/conftest.py": "def pytest_addoption(parser): parser.addoption('--ci-evidence-dir')\n",
        "leaf.py": "value = 1\n",
        "tests/test_leaf.py": "import os\nfrom leaf import value\ndef test_leaf():\n    assert value == 2\n"
                              "    assert os.environ['OUROBOROS_PREFLIGHT_REQUIRE_PLUGINS'] == '1'\n"
                              "    assert os.environ['OUROBOROS_PREFLIGHT_REQUIRE_NODE'] == '1'\n"
                              "    assert 'FIXTURE_API_KEY' not in os.environ\n",
        "tests/test_unrelated.py": "raise AssertionError('must not collect')\n",
    })
    repo = _make_repo(tmp_path, files)
    (repo / "ouroboros").symlink_to(real / "ouroboros", target_is_directory=True)
    (repo / "leaf.py").write_text("value = 2\n", encoding="utf-8")
    evidence = diagnostic_directory(tmp_path / "evidence", "selected-wrapper-" + tmp_path.name)
    env = {**os.environ, "OUROBOROS_PREFLIGHT_REQUIRE_PLUGINS": "1", "OUROBOROS_PREFLIGHT_REQUIRE_NODE": "1",
           "FIXTURE_API_KEY": "must-not-cross", "OUROBOROS_TEST_EVIDENCE_ROOT": str(evidence),
           "OUROBOROS_PREFLIGHT_TEST_WORKERS": "2", "OURO_TEST_WORKER_BUDGET": "2"}
    result = subprocess.run([sys.executable, str(repo / "scripts/run_tests.py"), "--selected", "--base", "HEAD"],
                            cwd=repo, env=env, capture_output=True, text=True, timeout=300)
    (evidence / "launcher.stdout").write_text(result.stdout, encoding="utf-8")
    (evidence / "launcher.stderr").write_text(result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    run = Path(next(line.split(" ", 1)[1] for line in result.stderr.splitlines() if line.startswith("SAFE_TEST_EVIDENCE ")))
    outcome = json.loads((run / "result.json").read_text(encoding="utf-8"))
    assert (outcome["returncode"], outcome["containment_error"], outcome["evidence_error"], outcome["retention_reason"]) == (0, "", "", "")
    # This fixture lacks the battery conftest that declares a basetemp, so pytest's own
    # empty default stays as undeclared command content: retained, never guessed away.
    root, last = Path(outcome["scratch"]), result.stderr.splitlines()[-1]
    if last.startswith("SAFE_TEST_RETAINED "):
        left = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
        assert all(item == "tmp" or item.startswith("tmp/pytest-of-") for item in left), left
        assert not [path for path in root.rglob("*") if path.is_file() and not path.is_symlink()], left
        import shutil
        shutil.rmtree(root)  # this test's own ended root: no file, nothing undeclared but pytest's empty default
    else:
        assert last == f"SAFE_TEST_RELEASED {root}", result.stderr
    row = json.loads((run / "pytest/selection.json").read_text(encoding="utf-8"))
    assert row["tier"] == "selected" and "tests/test_unrelated.py" not in row["expected_files"]
    assert row["outcomes"] == {"parallel": "success", "serial": "success"} and row["outcome"] == "success"
    for lane in ("parallel", "serial"):
        events = [json.loads(line)["event"] for line in
                  (run / f"pytest/inventory-{lane}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert events[0] == "start" and events[-1] == "finish", events
    executed = row["inventory"]["parallel"]["executed"]
    assert "tests/test_leaf.py::test_leaf" in executed and sorted(executed) == sorted(row["inventory"]["parallel"]["expected"])


def test_merges_run_the_selected_scope_and_qualification_runs_remain_full():
    """The affected selection runs after merges (post-merge.yml, owner decision 3A); manual and
    release runs of ci.yml keep the full default suite and no selected step."""
    import yaml
    from tests.test_platform_ci_events import ROOT, WORKFLOW

    post = yaml.safe_load((ROOT / ".github/workflows/post-merge.yml").read_text(encoding="utf-8"))
    steps = {step.get("id"): step for step in post["jobs"]["post-merge"]["steps"] if step.get("id")}
    selected = steps["selected"]
    assert selected["run"].startswith('python scripts/run_tests.py --selected --base "$BASE" --sequential --keep-going')
    # The run's exit belongs to classify and summary, which retry, bisect and decide the job.
    assert selected["timeout-minutes"] > 0 and selected["continue-on-error"] is True
    assert "continue-on-error" not in steps["classify"]
    for job in ("quick-test", "full-test-runs"):
        runs = [step.get("run", "") for step in WORKFLOW["jobs"][job]["steps"]]
        assert not any("--selected" in run for run in runs), job
        assert sum("python scripts/run_tests.py tests/ -m " in run for run in runs) == 2, job


@pytest.mark.parametrize("tier", ["selected", "full"])
def test_ci_summary_discloses_the_executed_tier(tmp_path, tier):
    from tests.ci_evidence import render_summary

    (tmp_path / "selection.json").write_text(json.dumps({"tier": tier}), encoding="utf-8")
    summary = render_summary(tmp_path / "parallel", producer_outcome="success", artifact_outcome="success")
    assert f"Selection: **{tier}**" in summary
