"""Whole-file shards of every OS preserve both passes and cannot publish incomplete proof."""
from collections import Counter
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from ouroboros.preflight_runner import LANE_EXCLUSION_EXPR, PARALLEL_PASS_FLAGS, SERIAL_PASS_FLAGS
from tests.ci_evidence import file_shards, reconcile_shards
from ouroboros.test_parallelism import NESTED_WORKERS_ENV

ROOT = Path(__file__).resolve().parents[1]
SHA, RUN = "a" * 40, "42"
EXPRESSIONS = {name: f"{prefix}serial and {LANE_EXCLUSION_EXPR}"
               for name, prefix in (("parallel", "not "), ("serial", ""))}
# The full-test matrix: runner image, `runner.os` (the artifacts' OS name) and its shard count.
SHARDED = (("ubuntu-latest", "Linux", 4), ("macos-latest", "macOS", 4), ("windows-latest", "Windows", 8))


def test_file_assignment_keeps_modules_and_balances_independent_of_input_order():
    weights = {"tests/b.py": 9, "tests/a.py": 9, "tests/c.py": 5, "tests/d.py": 4}
    expected = {"tests/a.py": 1, "tests/b.py": 2, "tests/c.py": 1, "tests/d.py": 2}
    assert file_shards(weights, 2) == expected
    assert file_shards(dict(reversed(list(weights.items()))), 2) == expected
    assert set(file_shards(weights, 1).values()) == {1}


@pytest.mark.serial
def test_real_pytest_manifests_and_xdist_prove_both_passes_by_whole_file(tmp_path):
    suite = tmp_path / "suite"
    (suite / "tests").mkdir(parents=True)
    for name, parallel, serial in (("mixed", 4, 3), ("credential_shapes", 3, 2), ("services_tool_v2", 2, 2)):
        (suite / "tests" / f"test_{name}.py").write_text(
            "import pytest\n"
            "calls = 0\n"
            "@pytest.fixture(scope='module')\n"
            "def shared():\n"
            "    global calls\n"
            "    calls += 1\n"
            "    return calls\n"
            f"@pytest.mark.parametrize('n', range({parallel}))\n"
            "def test_parallel(n, shared): assert shared == 1\n"
            "@pytest.mark.serial\n"
            f"@pytest.mark.parametrize('n', range({serial}))\n"
            "def test_serial(n, shared): assert shared == 1\n"
            "@pytest.mark.ui_browser\n"
            "def test_excluded(): raise AssertionError('opt-in lane ran')\n", encoding="utf-8")
    entry = tmp_path / "run.py"
    entry.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        f"os.chdir({str(suite)!r})\n"
        f"os.environ.update(GITHUB_SHA={SHA!r}, GITHUB_RUN_ID={RUN!r}, GITHUB_RUN_ATTEMPT='1')\n"
        "import pytest\n"
        "raise SystemExit(pytest.main(sys.argv[1:]))\n", encoding="utf-8")
    assignments, selections = {}, {}
    for lane in EXPRESSIONS:
        for shard in (None, 1, 2):
            public = tmp_path / ("manifest/attempt1" if shard is None else f"shards/shard{shard}-attempt1") / lane
            args = ["tests/", "-p", "tests.conftest", "--rootdir", str(suite), "--confcutdir", str(suite),
                    "-o", "addopts=", "-m", EXPRESSIONS[lane], "-q", "--ci-evidence-dir", str(public)]
            args += (["--default-suite-manifest", "--collect-only"] if shard is None
                     else [f"--default-suite-shard={shard}/2"])
            if lane == "parallel" and shard is not None:
                args += ["-n", "2", "--dist", "loadscope", "--max-worker-restart=0"]
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(ROOT / "scripts/safe_test.py"),
                 "--temp-parent", str(tmp_path), "--", sys.executable, str(entry), *args],
                env={**os.environ, NESTED_WORKERS_ENV: "2"},
                capture_output=True, text=True, encoding="utf-8", timeout=90)
            assert result.returncode == 0, result.stdout + result.stderr
            data = json.loads((public / "results.json").read_text(encoding="utf-8"))
            proof = data[f"default_{lane}"]
            assert data["session_exit_code"] == 0
            if shard is None:
                selections[lane] = set(proof["full"])
                assert not data["reports"] and "shard" not in proof
            else:
                assignments[lane, shard] = set(proof["assigned"])
                assert {row["nodeid"] for row in data["reports"]} == assignments[lane, shard]
        gaps, _ = reconcile_shards(tmp_path / "manifest", tmp_path / "shards", count=2,
                                   sha=SHA, run_id=RUN, lane_key=f"default_{lane}")
        assert gaps == []
        assert assignments[lane, 1].isdisjoint(assignments[lane, 2])
        assert assignments[lane, 1] | assignments[lane, 2] == selections[lane]
    assert len(selections["parallel"]) == 7 and len(selections["serial"]) == 9
    assert selections["parallel"].isdisjoint(selections["serial"])
    owners = {}
    for (_, shard), nodes in assignments.items():
        for node in nodes:
            path = node.split("::", 1)[0]
            assert owners.setdefault(path, shard) == shard  # Even the mixed-marker file stays together.
    # The full-selection guard also refuses an explicit narrowing control.
    narrowed = subprocess.run(
        [sys.executable, "-I", "-S", str(ROOT / "scripts/safe_test.py"),
         "--temp-parent", str(tmp_path), "--", sys.executable, str(entry),
         *args, "-k", "test_parallel"], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert narrowed.returncode != 0 and "DEFAULT_SUITE_INCOMPLETE" in narrowed.stderr


def _proofs(tmp_path, lane):
    full = [f"tests/{name}.py::test_case[{index}]" for name in "abcd" for index in range(3)]
    weights = dict(Counter(node.split("::", 1)[0] for node in full))
    owners = file_shards(weights, 2)
    paths = {}
    for shard in (None, 1, 2):
        proof = {"full": full, "file_weights": [[path, weight] for path, weight in weights.items()]}
        if shard:
            proof.update(shard=[shard, 2], assigned=[node for node in full if owners[node.split("::", 1)[0]] == shard])
        path = tmp_path / ("manifest/attempt1" if shard is None else f"shards/shard{shard}-attempt1") / "results.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            lane: proof, "session_exit_code": 0,
            "github": {"sha": SHA, "run_id": RUN, "run_attempt": "1"},
            "reports": [{"nodeid": node, "phase": "call", "outcome": "passed"}
                        for node in proof.get("assigned", [])]}), encoding="utf-8")
        paths[shard] = path
    return paths


@pytest.mark.parametrize("lane", ["default_parallel", "default_serial"])
@pytest.mark.parametrize("fault", ["none", "missing node", "duplicate assignment", "duplicate execution",
    "wrong file", "weights", "missing projection", "missing lane", "red shard", "red manifest",
    "duplicate proof", "foreign run", "newer red", "newer green"])
def test_existing_reconciler_accepts_only_complete_default_proofs(tmp_path, lane, fault):
    paths = _proofs(tmp_path, lane)
    path = paths[None if fault == "red manifest" else 1]
    data = json.loads(path.read_text(encoding="utf-8"))
    if fault == "missing node":
        data["reports"].pop()
    elif fault == "duplicate assignment":
        data[lane]["assigned"].append(data[lane]["assigned"][0])
    elif fault == "duplicate execution":
        data["reports"].append(data["reports"][0])
    elif fault == "wrong file":
        data[lane]["assigned"][0] = "tests/b.py::test_case[0]"
    elif fault == "weights":
        data[lane]["file_weights"][0][1] += 1
    elif fault == "missing lane":
        data.pop(lane)
    elif fault in {"red shard", "red manifest"}:
        data["session_exit_code"] = 1
    elif fault == "foreign run":
        data["github"]["run_id"] = "43"
    elif fault == "duplicate proof":
        other = path.parent.with_name("duplicate")
        other.mkdir()
        shutil.copyfile(path, other / "results.json")
    elif fault.startswith("newer"):
        older = dict(data, session_exit_code=1)
        path.write_text(json.dumps(older), encoding="utf-8")
        path = path.parent.with_name("shard1-attempt2") / "results.json"
        path.parent.mkdir()
        data["github"]["run_attempt"] = "2"
        data["session_exit_code"] = int(fault == "newer red")
    path.write_text(json.dumps(data), encoding="utf-8")
    if fault == "missing projection":
        path.unlink()
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(ROOT / "tests/ci_evidence.py"), "reconcile-shards",
         "--lane", lane, "--manifest", str(tmp_path / "manifest"), "--shards", str(tmp_path / "shards"),
         "--count", "2", "--sha", SHA, "--run-id", RUN, "--summary", str(tmp_path / "summary.md")],
        capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == (0 if fault in {"none", "newer green"} else 1), result.stdout + result.stderr
    assert ("GAP " in result.stdout) == (result.returncode == 1)


@pytest.mark.parametrize("value", ["0/4", "5/4", "1/0", "x/4", "1", "1/2/3"])
def test_invalid_file_shard_is_refused(value):
    from tests.default_lane import _shard

    with pytest.raises(pytest.UsageError, match="DEFAULT_SUITE_SHARD"):
        _shard(SimpleNamespace(getoption=lambda name: value))


def _jobs():
    return yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]


def test_every_os_runs_whole_file_shards_of_both_passes_and_its_checks_once(tmp_path):
    job = _jobs()["full-test-runs"]
    matrix = job["strategy"]["matrix"]
    assert matrix == {"include": [{"os": image, "shard": index, "shards": count}
                                  for image, _, count in SHARDED for index in range(1, count + 1)]}
    assert job["strategy"]["fail-fast"] is False and "continue-on-error" not in job
    steps = {step["id"]: step for step in job["steps"] if "id" in step}
    for label in EXPRESSIONS:
        run, collect = steps[f"tests_{label}"], steps[f"manifest_{label}"]
        assert f'-m "{EXPRESSIONS[label]}"' in run["run"] and f'-m "{EXPRESSIONS[label]}"' in collect["run"]
        # scripts/ci_retry.py wraps each pass; the shard option stays its last argument.
        assert run["run"].startswith(
            f'python scripts/ci_retry.py --evidence-dir "${{{{ runner.temp }}}}/ci-evidence/{label}" '
            "--accepted tests/accepted_failures.json -- python scripts/run_tests.py tests/ ")
        assert run["run"].endswith("--default-suite-shard=${{ matrix.shard }}/${{ matrix.shards }}")
        assert run["env"] == {"GH_TOKEN": "${{ github.token }}"}  # The accepted-failure check reads issues.
        assert "matrix.shard" not in run["if"]
        # Every OS witnesses its own two passes on its first shard.
        assert "matrix.shard == 1" in collect["if"] and "runner.os == 'Windows'" not in collect["if"]
        assert "--default-suite-shard" not in collect["run"] and "ci_retry" not in collect["run"]
        assert "--collect-only -q --default-suite-manifest" in collect["run"]
        assert "continue-on-error" not in run and "continue-on-error" not in collect
        assert run["timeout-minutes"] == 90
    assert " ".join(PARALLEL_PASS_FLAGS) in steps["tests_parallel"]["run"]
    assert " ".join(SERIAL_PASS_FLAGS) in steps["tests_serial"]["run"]
    assert all(flag not in steps["tests_serial"]["run"].split()
               for flag in PARALLEL_PASS_FLAGS if flag not in SERIAL_PASS_FLAGS)
    assert {"issues": "read"}.items() <= job["permissions"].items()
    for name in ("setup_node", "posix_tools", "select_posix_tools"):
        assert "matrix.shard" not in steps[name]["if"]  # Every runner needs the utilities and node.
    for name in ("book_mode", "tests_size", "attachment_browser_install", "attachment_browser", "composer_browser"):
        assert "matrix.shard == 1" in steps[name]["if"]
    for step in job["steps"]:
        if "run_tests.py --node" in step.get("run", "") or "test_no_core_a2a_telegram_imports.py" in step.get("run", ""):
            assert "matrix.shard == 1" in step["if"]
    mandatory = {step["name"]: step for step in job["steps"] if step.get("name") in (
        "Require both shard projections", "Upload the shard proof", "Require both manifest projections",
        "Upload the manifests")}
    assert len(mandatory) == 4
    for step in mandatory.values():
        assert "continue-on-error" not in step and "!cancelled()" in step["if"]
        assert ("matrix.shard == 1" in step["if"]) is ("manifest" in step["name"])
        if "Upload" in step["name"]:
            assert step["with"]["if-no-files-found"] == "error"
            # The reconciliation sorts the artifacts by OS, then judges this commit, run and attempt.
            assert all(f"{key}" in step["with"]["name"] for key in (
                "runner.os", "github.sha", "github.run_id", "github.run_attempt"))
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("workflow shell unavailable on this host")
    # A pass proves itself by its export, or, killed before it, by its journal and its retry verdict.
    cases = [({}, True), ({"parallel": ("events.jsonl", "verdict.json")}, True),
             ({"serial": ("events.jsonl",)}, False), ({"parallel": ("verdict.json",)}, False),
             ({"serial": ()}, False), ({"parallel": ("results.json:empty",)}, False)]
    for number, (changes, accepted) in enumerate(cases):
        temp = tmp_path / f"shard-case-{number}"
        for label in EXPRESSIONS:
            for name in changes.get(label, ("results.json",)):
                (temp / "ci-evidence" / label).mkdir(parents=True, exist_ok=True)
                (temp / "ci-evidence" / label / name.split(":")[0]).write_text(
                    "" if name.endswith(":empty") else "{}", encoding="utf-8")
        command = mandatory["Require both shard projections"]["run"].replace("${{ runner.temp }}", str(temp))
        # As `shell: bash` runs it: errexit stops the loop at the first pass without a proof.
        result = subprocess.run([bash, "--noprofile", "--norc", "-eo", "pipefail", "-c", command],
                                capture_output=True, timeout=10)
        assert (result.returncode == 0) is accepted, changes
    command = mandatory["Require both manifest projections"]["run"]
    assert command.count("test -s ") == 2  # Manifests are not retried: only their export proves them.
    command = command.replace("${{ runner.temp }}", str(tmp_path))
    files = [Path(line.split('"')[1]) for line in command.splitlines()]
    for missing in (0, 1, None):
        for index, path in enumerate(files):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("" if index == missing else "{}", encoding="utf-8")
        result = subprocess.run([bash, "-e", "-c", command], capture_output=True, timeout=10)
        assert (result.returncode == 0) == (missing is None)


def _artifacts(proof, os_name, count):
    """One OS's two passes as full-test downloads them: its manifest artifact and a proof per shard."""
    files = [f"tests/test_{index:02}.py" for index in range(count + 1)]
    owners = file_shards({path: 2 for path in files}, count)
    identity = {"sha": SHA, "run_id": RUN, "run_attempt": "1"}
    for label in EXPRESSIONS:
        full = [f"{path}::test_{label}" for path in files]
        block = {"full": full, "file_weights": [[path, 2] for path in files], "pass": label}
        for shard in range(count + 1):
            directory = (proof / "manifest" / f"default-ci-manifest-{os_name}-{SHA}-{RUN}-1" if shard == 0
                         else proof / "shards" / f"default-ci-shard-{os_name}-{shard}-{SHA}-{RUN}-1")
            lane = dict(block, **({"shard": [shard, count], "assigned": [
                node for node in full if owners[node.split("::", 1)[0]] == shard]} if shard else {}))
            (directory / label).mkdir(parents=True, exist_ok=True)
            (directory / label / "results.json").write_text(json.dumps({
                f"default_{label}": lane, "session_exit_code": 0, "github": identity,
                "reports": [{"nodeid": node, "phase": "call", "outcome": "passed"} for node in lane.get("assigned", [])]}),
                encoding="utf-8")


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled", "skipped", ""])
def test_full_test_result_requires_the_matrix_and_every_os_reconciliation(tmp_path, outcome):
    jobs = _jobs()
    job = jobs["full-test"]
    assert job["needs"] == ["full-test-runs", "journal-persist"]
    assert job["if"] == "${{ !cancelled() && needs.full-test-runs.result != 'skipped' }}"
    assert "continue-on-error" not in job
    gate, *steps = job["steps"]
    assert gate["env"] == {"MATRIX_RESULT": "${{ needs.full-test-runs.result }}",
                           "PERSIST_RESULT": "${{ needs.journal-persist.result }}"}
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("workflow shell unavailable on this host")
    # The matrix (WAL) and the PERSIST journal legs each have to succeed on their own.
    for matrix, persist in ((outcome, "success"), ("success", outcome)):
        result = subprocess.run([bash, "-c", gate["run"]],
                                env={**os.environ, "MATRIX_RESULT": matrix, "PERSIST_RESULT": persist},
                                capture_output=True, text=True, timeout=10)
        assert (result.returncode == 0) == (outcome == "success"), (matrix, persist)
    for step in steps:
        assert step["if"] == "${{ !cancelled() }}" and "continue-on-error" not in step
    (reconcile,) = [step for step in steps if "run" in step]
    assert "full-test" in jobs["release"]["needs"], "nothing is published without the reconciled matrix"
    if outcome != "success":
        return  # The loop below is the same for every outcome of the matrix.
    # Every OS of the matrix, with its own shard count, is reconciled in both passes: run the step itself.
    assert f"for spec in {' '.join(f'{os_name}:{count}' for _, os_name, count in SHARDED)}; do" in reconcile["run"]
    script = reconcile["run"].replace("python3 -I -S", f'"{sys.executable}" -I -S')
    for missing in (None, ("macOS", 3)):
        proof = tmp_path / f"proof-{missing is None}"
        for _, os_name, count in SHARDED:
            _artifacts(proof, os_name, count)
        if missing:
            shutil.rmtree(proof / "shards" / f"default-ci-shard-{missing[0]}-{missing[1]}-{SHA}-{RUN}-1")
        summary = tmp_path / f"summary-{missing is None}.md"
        result = subprocess.run([bash, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=120,
                                env={**os.environ, "PROOF": str(proof), "SHA": SHA, "RUN_ID": RUN,
                                     "GITHUB_STEP_SUMMARY": str(summary)})
        text = summary.read_text(encoding="utf-8")
        if missing is None:
            assert result.returncode == 0, result.stdout + result.stderr
            assert text.count("reconciliation — complete") == 2 * len(SHARDED)
        else:
            assert result.returncode == 1 and text.count("**GAP** shard 3/4: no proof") == 2, text


def test_each_journal_mode_is_proven_by_its_own_lane():
    """Fresh installs run the shipped interpreter's WAL, installs updated in place keep an older SQLite's
    PERSIST: the matrix probes WAL, while quick-test and journal-persist (all three OSes) probe PERSIST."""
    jobs = _jobs()
    persist = jobs["journal-persist"]
    assert persist["strategy"]["matrix"]["os"] == ["ubuntu-latest", "macos-latest", "windows-latest"]
    for name in ("journal-persist", "quick-test"):
        setup = next(step for step in jobs[name]["steps"] if step.get("uses") == "./.github/actions/setup-python-env")
        assert "interpreter" not in (setup.get("with") or {}), name  # setup-python's SQLite

    def probe(name):
        return next(step["run"] for step in jobs[name]["steps"] if "store._journal_mode" in step.get("run", ""))

    assert probe("journal-persist").endswith(" PERSIST") and probe("quick-test").endswith(" PERSIST")
    assert probe("full-test-runs").endswith(" WAL")
    assert persist["if"] == jobs["full-test-runs"]["if"], "full-test needs it wherever the matrix runs"


def test_the_persist_legs_select_real_modules_and_keep_the_gate_split(tmp_path):
    """The selection is read back through $(...): no trailing newline, since Git Bash keeps a Windows CR there
    and the last path would not be found. An empty selection fails before any pass runs the whole tree, and the
    passes keep the gate's split: parallel flags in the parallel pass only, evidence from both."""
    repo = Path(__file__).resolve().parents[1]
    job_steps = _jobs()["journal-persist"]["steps"]
    steps = {step["id"]: step for step in job_steps if "id" in step}
    select, parallel, serial = steps["select"], steps["tests_parallel"], steps["tests_serial"]
    assert "end='')" in select["run"] and 'test "$count" -ge 8' in select["run"]
    # A misspelt or renamed name, or a renamed shared helper, would drop modules and still clear the floor:
    # every named module exists, and the module holding the journal helpers is the one the selector greps for.
    named = re.findall(r"'(test_[\w]+\.py)'", re.search(r"core = \{([^}]*)\}", select["run"])[1])
    assert len(named) == 11 and all((repo / "tests" / name).is_file() for name in named), named
    helpers = [path.stem for path in (repo / "tests").glob("_*.py")
               if re.search(r"^def (?:exclusive_hold|write_hold|use_journal)\b", path.read_text(encoding="utf-8"), re.M)]
    assert all(f"re.compile(r'{stem}')" in select["run"] for stem in helpers), helpers
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("workflow shell unavailable on this host")
    # `python` is this interpreter's own folder first on PATH: a link to a Windows venv launcher loses its pyvenv.cfg.
    path = f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    result = subprocess.run([bash, "-c", select["run"]], cwd=repo, capture_output=True, text=True, timeout=60,
                            env={**os.environ, "RUNNER_TEMP": str(tmp_path), "PATH": path})
    assert result.returncode == 0, result.stderr
    listed = (tmp_path / "journal_modules.txt").read_bytes()
    assert listed and not listed.endswith((b"\n", b"\r")), listed[-20:]
    modules = listed.decode().split()
    assert len(modules) >= 8 and all((repo / module).is_file() for module in modules), modules
    assert serial["if"] == "${{ !cancelled() && steps.select.outcome == 'success' }}"
    assert " ".join(PARALLEL_PASS_FLAGS) in parallel["run"]  # the gate's own split, as the matrix pins it
    assert " ".join(SERIAL_PASS_FLAGS) in serial["run"]
    assert all(flag not in serial["run"].split() for flag in PARALLEL_PASS_FLAGS if flag not in SERIAL_PASS_FLAGS)
    for step, lane in ((parallel, "parallel"), (serial, "serial")):
        assert f'-m "{EXPRESSIONS[lane]}"' in step["run"], lane
        assert f'--ci-evidence-dir="${{{{ runner.temp }}}}/ci-evidence/{lane}"' in step["run"], lane
        # Named modules, no shard: scripts/ci_retry.py retries and continues from the crash record.
        assert step["run"].startswith(f'python scripts/ci_retry.py --evidence-dir "${{{{ runner.temp }}}}/ci-evidence/'
                                      f'{lane}" --accepted tests/accepted_failures.json -- python scripts/run_tests.py ')
    publish = next(step for step in job_steps if step.get("uses") == "./.github/actions/test-evidence")
    assert publish["with"]["passes"] == ("parallel=${{ steps.tests_parallel.outcome }} "
                                         "serial=${{ steps.tests_serial.outcome }}")
