"""The selected CLI runs explicit modules through both real pytest lanes."""
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from ouroboros import test_selection
from ouroboros.preflight_inventory import MODULE
from ouroboros.preflight_runner import LANE_EXCLUSION_EXPR
from tests.ci_verdicts import SESSION_UNAVAILABLE
from tests.test_preflight_runner import _make_repo

WINDOWS_COMMAND_LINE = 32_767  # CreateProcess takes a command line shorter than this, terminator included
# The fields the two plugins a run writes are filled from (``preflight_probe.WORKER_PROBE_SOURCE``,
# ``preflight_inventory.SOURCE``). A path or a module name may spell any of them, and that is never a field.
TEMPLATE_FIELDS = ("__WORKER_LOG__", "__RESULTS_POINTER__", "__SCOPE__", "__RESULT_PATH__")


def plugin_values(argv, directory):
    """The module-level values (``_path``, ``_scope``) of the inventory plugin that ``argv`` loads, read by running
    the source the run generated."""
    module = next(arg for arg in argv if arg.startswith(MODULE + "_"))
    return runpy.run_path(str(Path(directory) / f"{module}.py"))


def plugin_scope(argv, directory):
    """The scope that plugin carries: what each of its pytest processes collects."""
    return plugin_values(argv, directory)["_scope"]


def refuse_long_pytest_launches(monkeypatch, limit=WINDOWS_COMMAND_LINE):
    """Refuse a pytest launch whose command line reaches ``limit`` characters, as Windows does (WinError 206)
    before the program starts: this platform takes such a line, so the guard is what makes a long scope fail
    here. Returns the argv of every pytest launch that was accepted."""
    real, accepted = subprocess.Popen, []

    class Bounded(real):
        def __init__(self, args, *rest, **options):
            if list(args[1:3]) == ["-m", "pytest"]:
                if len(subprocess.list2cmdline(args)) >= limit:
                    raise FileNotFoundError(2, "[WinError 206] The filename or extension is too long")
                accepted.append(list(args))
            super().__init__(args, *rest, **options)

    monkeypatch.setattr(subprocess, "Popen", Bounded)
    return accepted


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
    monkeypatch.setattr(cli, "_node_lane", lambda: node.append(True) or (0, ""))
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
    empty = row["inventory"]["serial"]
    assert empty["collection_reports"] == 1 and empty["expected"] == empty["executed"] == []
    assert empty["exitcode"] in (0, 5)
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


def test_selected_cli_reuse_keeps_raw_inventory_and_projections_by_attempt(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("selected_cli", Path(__file__).resolve().parents[1] / "scripts/run_tests.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    module = "tests/test_fixture.py"
    chosen = test_selection.TestSelection("selected", "base", (module,), (), (), "policy", 1)
    monkeypatch.setattr(test_selection, "select_tests", lambda *a, **k: chosen)
    monkeypatch.setattr(test_selection, "FLOOR_TESTS", {module: "fixture floor"})
    monkeypatch.setattr(cli, "_node_lane", lambda: (0, ""))
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "base.json").write_bytes(b"another producer's evidence")
    public = f"{module}::test_param[masked]"
    lanes = ("parallel", "serial")
    paths = [evidence / "selection.json", *(evidence / f"inventory-{lane}.jsonl" for lane in lanes),
             *(evidence / lane / name for lane in lanes for name in ("results.json", "events.jsonl"))]

    def run(label, argv, cwd):
        if label == "warm":
            return 0
        identity = f"{label}-{attempt}"
        raw = [f"{module}::test_param[{identity}-{suffix}]" for suffix in ("a", "b")]
        code = int(attempt == 3)
        reports = [{"nodeid": public, "phase": phase, "outcome": "failed" if code and phase == "call" else "passed"}
                   for _ in raw for phase in ("setup", "call", "teardown")]
        lane = evidence / label
        lane.mkdir(exist_ok=True)
        assert not (lane / "events.jsonl").exists() and not (lane / "results.json").exists()
        (lane / "results.json").write_text(json.dumps({"session": identity, "session_exit_code": code,
                                                      "reports": reports, "tests_collected": len(raw)}), encoding="utf-8")
        logs = [{"event": "session", "session": identity}, *({"event": "report", **r} for r in reports)]
        (lane / "events.jsonl").write_text("\n".join(map(json.dumps, logs)) + "\n", encoding="utf-8")
        rows = [{"event": "start"}, {"event": "evidence_session", "session": identity},
                {"event": "collection", "ids": raw},
                *({"event": "report", "when": "teardown", "nodeid": node, "outcome": "passed"} for node in raw),
                {"event": "finish", "exitcode": code}]
        (evidence / f"inventory-{label}.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
        if (label, attempt) == ("parallel", 2):
            (lane / SESSION_UNAVAILABLE).touch()  # a producer that never confirmed its session in this directory
        return code

    monkeypatch.setattr(cli, "_run", run)
    history = {}
    for attempt in (1, 2, 3):
        assert cli.main(["--selected", "--evidence-dir", str(evidence)]) == int(attempt == 3)
        assert len(list(evidence.glob("selection.previous-*.json"))) == attempt - 1
        selection = paths[0].read_bytes()
        if attempt < 3:
            history[selection] = [path.read_bytes() for path in paths]
    for archive in evidence.glob("selection.previous-*.json"):
        suffix = archive.stem.removeprefix("selection")
        assert [path.with_stem(path.stem + suffix).read_bytes() for path in paths] == history.pop(archive.read_bytes())
    assert not history
    current = json.loads(selection)
    assert current["outcomes"] == {"parallel": "failure", "serial": "skipped"}
    assert current["inventory"]["parallel"]["session"] == "parallel-3"
    assert (evidence / "inventory-serial.jsonl").read_bytes() == b""
    assert not (evidence / "serial" / "results.json").exists() and not (evidence / "serial" / "events.jsonl").exists()
    assert (evidence / "base.json").read_bytes() == b"another producer's evidence"
    # The second attempt's producer never confirmed its session: its marker is set aside with that attempt's files
    # (one suffix), and never stands at the current name over a later attempt.
    [parked] = (evidence / "parallel").glob(SESSION_UNAVAILABLE + ".previous-*")
    assert not (evidence / "parallel" / SESSION_UNAVAILABLE).exists()
    assert (evidence / f"selection{parked.suffix}.json").is_file()


def test_selected_cli_runs_a_changed_test_modules_importers_and_only_selects_opt_in_ones(tmp_path, monkeypatch, capsys):
    """Selection names modules: an importer whose only test is in an excluded lane is scope, not execution."""
    spec = importlib.util.spec_from_file_location("selected_cli", Path(__file__).resolve().parents[1] / "scripts/run_tests.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    hub = "def helper():\n    return 1\n\n\ndef test_hub():\n    assert helper() == 1\n"
    repo = _make_repo(tmp_path, {
        ".gitignore": "__pycache__/\n.pytest_cache/\n",
        "pytest.ini": "[pytest]\nmarkers =\n    serial: serial lane\n    ui_browser: opt-in lane\n",
        "tests/__init__.py": "",
        "tests/conftest.py": "def pytest_addoption(parser): parser.addoption('--ci-evidence-dir')\n",
        "tests/test_floor.py": "def test_floor(): assert True\n",
        "tests/test_hub.py": hub,
        "tests/test_importer.py": "from tests.test_hub import helper\n\n\ndef test_importer():\n    assert helper() == 1\n",
        "tests/test_opt_in.py": "import pytest\nfrom tests.test_hub import helper\n\n\n@pytest.mark.ui_browser\n"
                                "def test_opt_in():\n    raise AssertionError('an excluded lane must not run')\n",
        "tests/test_unrelated.py": "raise AssertionError('must not collect')\n",
    })
    monkeypatch.setattr(cli, "REPO", repo)
    monkeypatch.setattr(test_selection, "FLOOR_TESTS", {"tests/test_floor.py": "minimal witness"})
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "2")
    monkeypatch.setenv("OURO_TEST_WORKER_BUDGET", "2")  # nested slots: the selected passes run two real workers
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    monkeypatch.setattr(cli, "_node_lane", lambda: (0, ""))
    (repo / "tests/test_hub.py").write_text(hub + "# changed\n", encoding="utf-8")
    evidence = tmp_path / "evidence"
    assert cli.main(["--selected", "--base", "HEAD", "--evidence-dir", str(evidence)]) == 0
    row = json.loads((evidence / "selection.json").read_text(encoding="utf-8"))
    assert row["expected_files"] == ["tests/test_floor.py", "tests/test_hub.py", "tests/test_importer.py", "tests/test_opt_in.py"]
    assert ["tests/test_opt_in.py", "direct import: tests.test_hub"] in row["reasons"]
    executed = [node for lane in row["inventory"].values() for node in lane["executed"]]
    collected = [node for lane in row["inventory"].values() for node in lane["expected"]]
    assert {"tests/test_hub.py::test_hub", "tests/test_importer.py::test_importer"} <= set(executed)
    assert not [node for node in collected if node.startswith("tests/test_opt_in.py")]  # selected, never collected
    assert row["outcomes"] == {"parallel": "success", "serial": "success"} and row["outcome"] == "success"
    assert "selected: 4/5 Python modules" in capsys.readouterr().out


def test_selected_cli_starts_pytest_with_a_scope_longer_than_one_windows_command_line(tmp_path, monkeypatch, capsys):
    """Windows CI refused the post-merge run before pytest started: 662 ``--select-test`` options made a
    command line past 32,767 characters (WinError 206). The scope now travels in the run's inventory plugin, off
    the command line, and pytest still collects exactly it: a scope of non-ASCII names (with a space and a tab),
    lane markers, the policy, and an unrelated module that is never collected."""
    from ouroboros.preflight_inventory import SELECT, selection_passes

    spec = importlib.util.spec_from_file_location("selected_cli", Path(__file__).resolve().parents[1] / "scripts/run_tests.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    importer = "from leaf import value\n\n\ndef test_it():\n    assert value == 2\n"
    lanes = ("import pytest\nfrom leaf import value\n\n\ndef test_plain():\n    assert value == 2\n\n\n"
             "@pytest.mark.serial('exclusive')\ndef test_serial():\n    assert value == 2\n\n\n"
             "@pytest.mark.ui_browser\ndef test_opt_in():\n    raise AssertionError('an excluded lane must not run')\n")
    folder = "tests/проверки_выбранной_области"  # kept short: the longest path stays under a Windows MAX_PATH
    long_scope = {f"{folder}/test_проверка_{n:03d}_имя_модуля_для_заполнения_вектора.py": importer for n in range(340)}
    odd = {"tests/папка с пробелом/test_пробел.py": importer, "tests/test_ü_unicode.py": importer,
           **({"tests/папка\tс табом/test_\tтаб.py": importer} if os.name != "nt" else {})}  # no tab in a Windows name
    repo = _make_repo(tmp_path, {
        ".gitignore": "__pycache__/\n.pytest_cache/\n",
        "pytest.ini": "[pytest]\nmarkers =\n    serial: serial lane\n    ui_browser: opt-in lane\n",
        "tests/conftest.py": "def pytest_addoption(parser): parser.addoption('--ci-evidence-dir')\n",
        "tests/test_floor.py": "def test_floor(): assert True\n",
        "leaf.py": "value = 1\n",
        "tests/test_lanes.py": lanes,
        "tests/test_unrelated.py": "raise AssertionError('must not collect')\n",
        **long_scope, **odd,
    })
    monkeypatch.setattr(cli, "REPO", repo)
    monkeypatch.setattr(test_selection, "FLOOR_TESTS", {"tests/test_floor.py": "minimal witness"})
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "2")
    monkeypatch.setenv("OURO_TEST_WORKER_BUDGET", "2")  # nested slots: the selected passes run two real workers
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    monkeypatch.setattr(cli, "_node_lane", lambda: (0, ""))
    (repo / "leaf.py").write_text("value = 2\n", encoding="utf-8")
    # The control: spelled out inline, as its spec names it, this scope is past what Windows launches.
    inline = selection_passes(test_selection.select_tests(repo))[0].args
    assert len(subprocess.list2cmdline([sys.executable, "-m", "pytest", *inline])) > WINDOWS_COMMAND_LINE
    launches = refuse_long_pytest_launches(monkeypatch)
    evidence = tmp_path / "evidence"
    assert cli.main(["--selected", "--base", "HEAD", "--evidence-dir", str(evidence)]) == 0

    row = json.loads((evidence / "selection.json").read_text(encoding="utf-8"))
    selected = row["expected_files"]
    assert sorted(selected) == sorted(["tests/test_floor.py", "tests/test_lanes.py", *long_scope, *odd])
    assert "tests/test_unrelated.py" not in selected and sum(not path.isascii() for path in selected) > 300
    assert len(launches) == 2  # one launch per lane, each accepted under the bound
    for launch in launches:
        assert not [arg for arg in launch if arg.startswith(SELECT)] and len(subprocess.list2cmdline(launch)) < 2_000
        assert plugin_scope(launch, evidence / "probe") == selected  # exactly the selection, in order
    assert [launch[launch.index("tests/") + 2] for launch in launches] == [
        f"not exclusive and {LANE_EXCLUSION_EXPR}", f"exclusive and {LANE_EXCLUSION_EXPR}"]  # the gate's pass split
    parallel, serial = row["inventory"]["parallel"], row["inventory"]["serial"]
    assert sorted(parallel["executed"]) == sorted(parallel["expected"])
    assert {node.split("::")[0] for node in parallel["executed"]} == set(selected)  # every selected module ran
    assert all(f"{path}::test_it" in parallel["executed"] for path in odd)
    assert "tests/test_lanes.py::test_plain" in parallel["executed"]
    assert serial["executed"] == serial["expected"] == ["tests/test_lanes.py::test_serial"]
    assert not [node for lane in row["inventory"].values() for node in lane["expected"]
                if "test_opt_in" in node or "test_unrelated" in node]
    assert row["outcomes"] == {"parallel": "success", "serial": "success"} and row["outcome"] == "success"
    assert SELECT not in capsys.readouterr().out  # the command the CLI prints no longer lists the scope


def test_selected_cli_runs_where_the_evidence_directory_and_a_module_spell_template_fields(tmp_path, monkeypatch):
    """``--evidence-dir`` is the caller's path and the scope is the selection's: spelling a field of either plugin's
    template (``TEMPLATE_FIELDS``) changes neither, so the worker probe and the inventory plugin the run writes
    there report to the directories it reads."""
    spec = importlib.util.spec_from_file_location("selected_cli", Path(__file__).resolve().parents[1] / "scripts/run_tests.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    named = "tests/test_" + "_and_".join(TEMPLATE_FIELDS) + ".py"  # a module that spells every field
    repo = _make_repo(tmp_path, {
        ".gitignore": "__pycache__/\n.pytest_cache/\n",
        "tests/conftest.py": "def pytest_addoption(parser): parser.addoption('--ci-evidence-dir')\n",
        "tests/test_floor.py": "def test_floor(): assert True\n",
        named: "def test_named(): assert True\n",
        "tests/test_unrelated.py": "raise AssertionError('must not collect')\n",
    })
    monkeypatch.setattr(cli, "REPO", repo)
    monkeypatch.setattr(test_selection, "FLOOR_TESTS", {"tests/test_floor.py": "minimal witness"})
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "2")
    monkeypatch.setenv("OURO_TEST_WORKER_BUDGET", "2")  # nested slots: the selected passes run two real workers
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    monkeypatch.setattr(cli, "_node_lane", lambda: (0, ""))
    (repo / named).write_text("def test_named(): assert True\n# changed\n", encoding="utf-8")
    evidence = tmp_path.joinpath(*TEMPLATE_FIELDS)  # an evidence directory that spells every field
    assert cli.main(["--selected", "--base", "HEAD", "--evidence-dir", str(evidence)]) == 0
    row = json.loads((evidence / "selection.json").read_text(encoding="utf-8"))
    assert row["expected_files"] == sorted(["tests/test_floor.py", named])
    parallel = row["inventory"]["parallel"]
    assert sorted(parallel["executed"]) == sorted(parallel["expected"]) == sorted([
        "tests/test_floor.py::test_floor", f"{named}::test_named"])
    assert row["outcomes"] == {"parallel": "success", "serial": "success"} and row["outcome"] == "success"
    # The gate's plugin reported where its readers look (observed_worker_ids, recorded_session): a file per worker of
    # the parallel pass, and the pointer each pass's controller leaves at its own results directory (the serial pass last).
    workers = [path.name for path in (evidence / "workers").iterdir()]
    assert workers and all(name.startswith("gw") for name in workers), workers
    assert (evidence / "ci-evidence-dir").read_text(encoding="utf-8") == str((evidence / "serial").resolve())


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
    # The tier only: the step's exact command (--each-commit, --keep-going, its evidence directory) has one owner,
    # tests/test_post_merge_workflow.py::test_the_steps_check_retry_attribute_hand_over_and_report_in_order.
    assert selected["run"].startswith("python scripts/run_tests.py --selected ")
    # The run's exit belongs to classify and summary, which retry, bisect and decide the job.
    assert selected["timeout-minutes"] > 0 and selected["continue-on-error"] is True
    assert "continue-on-error" not in steps["classify"]
    for job in ("quick-test", "full-test-runs"):
        runs = [step.get("run", "") for step in WORKFLOW["jobs"][job]["steps"]]
        assert not any("--selected" in run for run in runs), job
        assert sum("python scripts/run_tests.py --lane-pass " in run for run in runs) == 2, job


@pytest.mark.parametrize("tier", ["selected", "full"])
def test_ci_summary_discloses_the_executed_tier(tmp_path, tier):
    from tests.ci_evidence import render_summary

    (tmp_path / "selection.json").write_text(json.dumps({"tier": tier}), encoding="utf-8")
    summary = render_summary(tmp_path / "parallel", producer_outcome="success", artifact_outcome="success")
    assert f"Selection: **{tier}**" in summary


def test_a_deferred_record_names_the_origin_that_wrote_it(tmp_path, monkeypatch):
    """A deferred leg's selection.json names the commit, run, attempt and system that wrote it, spelled as
    tests/ci_evidence.py `_identity` writes them (empty strings outside Actions), and the platforms map to
    the systems the record spells. The deferral's fallback cases — a scan that fails, an escalation this leg
    does not name, each carrying the fixed `selection_problem` — live with the runner's own deferral tests
    (tests/test_run_tests_script.py::test_a_deferred_full_escalation_keeps_what_the_other_changes_select)."""
    import dataclasses

    spec = importlib.util.spec_from_file_location("selected_cli", Path(__file__).resolve().parents[1] / "scripts/run_tests.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    chosen = test_selection.TestSelection(
        "full", "b" * 40, ("tests/test_everything.py",),
        (("tests/conftest.py", "verification apparatus changed"), ("ouroboros/utils.py", "changed code")),
        ("tests/conftest.py", "ouroboros/utils.py"), "policy", 3)
    monkeypatch.setattr(test_selection, "select_tests", lambda *a, **k: chosen)
    monkeypatch.setattr(cli, "_node_lane", lambda: (0, ""))
    monkeypatch.setattr(cli, "_run", lambda label, argv, cwd: 0)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    github = {"sha": "a" * 40, "run_id": "7", "run_attempt": "2"}
    for name, value in github.items():
        monkeypatch.setenv("GITHUB_" + name.upper(), value)

    def selection(name):
        cli.main(["--selected", "--defer-full", "--evidence-dir", str(tmp_path / name)])
        return json.loads((tmp_path / name / "selection.json").read_text(encoding="utf-8"))

    # A genuine deferral selects what the other changes select and carries no problem; its record
    # still names the commit, run, attempt and system that wrote it.
    monkeypatch.setattr(test_selection, "select_from_paths", lambda *a, **k: dataclasses.replace(
        chosen, tier="selected", files=("tests/test_importer.py",)))
    for name in github:  # outside Actions the identity is empty, never missing
        monkeypatch.delenv("GITHUB_" + name.upper())
    assert selection("local")["github"] == {"sha": "", "run_id": "", "run_attempt": ""}
    for host, system in (("Linux", "linux"), ("Windows", "windows"), ("Darwin", "darwin")):
        monkeypatch.setattr(cli.platform, "system", lambda host=host: host)
        assert cli._system() == system


def test_a_warm_interrupted_before_its_passes_leaves_the_node_outcome_on_disk(tmp_path, monkeypatch):
    """The typed Node outcome reaches selection.json before the warm runs. A warm interrupted on its way (its
    memory wait, its walk) ends the run before any pass writes the record again, and the red the lane observed
    stays on disk for the reader that picks the run up — the post-merge leg's --keep-going can meet exactly
    this interruption between its Node lane and its passes."""
    spec = importlib.util.spec_from_file_location("selected_cli", Path(__file__).resolve().parents[1] / "scripts/run_tests.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    module = "tests/test_fixture.py"
    chosen = test_selection.TestSelection("selected", "base", (module,), (), (), "policy", 1)
    monkeypatch.setattr(test_selection, "select_tests", lambda *a, **k: chosen)
    monkeypatch.setattr(test_selection, "FLOOR_TESTS", {module: "fixture floor"})
    monkeypatch.setattr(cli, "_node_lane", lambda: (2, "node"))  # a red lane that named the Node it ran

    def interrupted_warm(files=(), evidence=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_warm_bytecode", interrupted_warm)
    monkeypatch.setattr(cli, "_run", lambda label, argv, cwd, node="": pytest.fail("a pass must not start"))
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    evidence = tmp_path / "evidence"
    with pytest.raises(KeyboardInterrupt):
        cli.main(["--selected", "--keep-going", "--evidence-dir", str(evidence)])
    row = json.loads((evidence / "selection.json").read_text(encoding="utf-8"))
    assert row["node"] == "failure"
    assert row["node_problem"] == "the complete Node lane did not run to completion (runner exit 2)"
