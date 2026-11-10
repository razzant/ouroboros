"""editbench's fixed v2 workload: seeds carried as pinned data, validated
publication, and a runner that records a refusal of a missing, partial or stale
fixture before anything is paid for.

Nothing here needs git history: a shallow clone or a source archive runs every
test. The tools under test are today's: the smoke at the end drives the current
edit_batch / apply_patch / edit_text / write_file through the real registry on the
materialized workspaces, with no model, and grades them with the benchmark's checker.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import pytest

from devtools.benchmarks.editbench import run_editbench
from tests._cases import case_folders

# The very module objects the runner imported (it puts the editbench dir on sys.path),
# so a pin patched here is the pin the runner checks.
checker = sys.modules["checker"]
mf = sys.modules["make_fixtures_v2"]

V2_NAMES = list(mf.V2_TASKS)


def _drifted_review_state() -> bytes:
    """What the pre-v2.1 generator wrote once the product moved on: not the pinned seed."""
    return mf.seed_path("ouroboros/review_state.py").read_bytes() + b"# drifted since August\n"


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    """One real materialization, shared read-only; tests that mutate copy it."""
    out = tmp_path_factory.mktemp("editbench") / "fixtures_v2"
    assert mf.materialize(out) == []
    return out


def _copy_published(published, tmp_path) -> pathlib.Path:
    return pathlib.Path(shutil.copytree(published, tmp_path / "fixtures_v2"))


# ---------------------------------------------------------------------------
# Generator mechanics and seed provenance
# ---------------------------------------------------------------------------

def test_quote_flip_reads_an_fstring_whole_on_every_python():
    """Python 3.12+ splits f"..." into FSTRING_* tokens; the oracle must not lose it."""
    source = 'a = f"{x}::"\nb = f"{y[\'k\']}"\nc = "s"\nd = """doc"""\ne = r"p"\n'
    flipped, count = mf._flip_quotes(source)
    assert count == 3
    assert flipped == 'a = f\'{x}::\'\nb = f"{y[\'k\']}"\nc = \'s\'\nd = """doc"""\ne = r\'p\'\n'


def test_publish_swaps_in_and_keeps_exactly_one_previous_set(tmp_path):
    out = tmp_path / "fixtures_v2"
    for generation in ("first", "second", "third"):
        staging = tmp_path / f"staging-{generation}"
        staging.mkdir()
        (staging / "marker").write_text(generation, encoding="utf-8")
        mf._publish(staging, out)
        assert not staging.exists()
    assert (out / "marker").read_text(encoding="utf-8") == "third"
    assert (tmp_path / "fixtures_v2.previous" / "marker").read_text(encoding="utf-8") == "second"


class _Interrupted(BaseException):
    """A crash stand-in: unlike OSError, nothing in _publish handles it."""


@pytest.mark.parametrize("failure", [OSError, _Interrupted])
def test_a_failure_between_the_two_renames_keeps_the_replaced_set(failure, tmp_path, monkeypatch):
    """Publication is not atomic; its documented limit is what these two paths pin."""
    out, previous = tmp_path / "fixtures_v2", tmp_path / "fixtures_v2.previous"
    out.mkdir()
    (out / "marker").write_text("replaced", encoding="utf-8")
    real_replace = os.replace

    def second_rename_fails(src, dst):
        if pathlib.Path(src).name.startswith("fixtures_v2.staging-"):
            raise failure("injected between the renames")
        real_replace(src, dst)

    with monkeypatch.context() as patched:
        patched.setattr(mf.os, "replace", second_rename_fails)
        with pytest.raises(failure):
            mf.materialize(out)

    handled = failure is OSError  # an OSError puts it back; a crash leaves no canonical set
    assert out.exists() == handled
    assert ((out if handled else previous) / "marker").read_text(encoding="utf-8") == "replaced"

    assert mf.materialize(out) == []  # rerunning publishes and keeps the replaced set
    assert mf.verify_task("t2_surgical", out) == []
    assert (previous / "marker").read_text(encoding="utf-8") == "replaced"


def test_the_carried_seeds_are_the_recorded_historical_blobs():
    """Provenance without history: each data file hashes to its recorded git blob id."""
    for rel, pin in mf.SEED_FILES.items():
        path = mf.seed_path(rel)
        assert path.name.endswith(".py.txt"), path  # data, never an importable module
        data = path.read_bytes()
        assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == pin["blob"], rel
        assert hashlib.sha256(data).hexdigest() == pin["sha256"], rel


@pytest.mark.parametrize("damage", ["altered", "missing"])
def test_a_damaged_seed_fails_closed_and_keeps_the_published_set(damage, tmp_path):
    seeds = pathlib.Path(shutil.copytree(mf.SEEDS, tmp_path / "seeds_v2"))
    seed = mf.seed_path("ouroboros/review_state.py", seeds)
    if damage == "altered":
        seed.write_bytes(seed.read_bytes().replace(b"_MAX_RUN_HISTORY = 10", b"_MAX_RUN_HISTORY = 11"))
    else:
        seed.unlink()
    out = tmp_path / "fixtures_v2"
    (out / "t2_surgical" / "workspace").mkdir(parents=True)
    (out / "t2_surgical" / "workspace" / "review_state.py").write_text("operator data\n", encoding="utf-8")
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))

    problems = mf.materialize(out, seeds)

    assert len(problems) == 1 and problems[0].startswith(f"seed ouroboros/review_state.py: {seed} sha256 "), problems
    assert problems[0].endswith(f"is not the pinned {mf.SEED_FILES['ouroboros/review_state.py']['sha256']}")
    assert ("sha256 missing " in problems[0]) == (damage == "missing")
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == before
    assert (out / "t2_surgical" / "workspace" / "review_state.py").read_text(encoding="utf-8") == "operator data\n"


def test_oracle_refuses_incomplete_failing_and_vacuous_fixtures(tmp_path):
    workspace, expected = tmp_path / "workspace", tmp_path / "expected"
    workspace.mkdir()
    expected.mkdir()
    (workspace / "m.py").write_text("X = 1\n", encoding="utf-8")
    assert checker.oracle_problems(workspace, expected, ["m.py"]) == ["expected tree lacks graded file m.py"]

    (expected / "m.py").write_text("X = 1\n", encoding="utf-8")
    assert checker.oracle_problems(workspace, expected, ["m.py"]) == [
        "the untouched workspace already passes the oracle"
    ]

    (expected / "m.py").write_text("X = 2\n", encoding="utf-8")
    failing_check = tmp_path / "check.py"
    failing_check.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
    problems = checker.oracle_problems(workspace, expected, ["m.py"], [sys.executable, str(failing_check)], True)
    assert len(problems) == 1 and problems[0].startswith("expected tree fails its own oracle"), problems
    assert sorted(p.name for p in expected.iterdir()) == ["m.py"]  # graded on copies


# ---------------------------------------------------------------------------
# The real pinned workload
# ---------------------------------------------------------------------------

def test_the_pinned_workload_reproduces_the_documented_august_shapes(published):
    for name in V2_NAMES:
        assert mf.task_problems(name, published) == [], name
        identity = mf.task_identity(name)
        assert identity["version"] == mf.WORKLOAD_VERSION and len(identity["digest"]) == 64

    def changed(task: str, rel: str) -> tuple[int, int]:
        seed = (published / task / "workspace" / rel).read_text(encoding="utf-8").splitlines()
        oracle = (published / task / "expected" / rel).read_text(encoding="utf-8").splitlines()
        return sum(a != b for a, b in zip(seed, oracle)), len(seed)

    assert changed("t2_surgical", "review_state.py") == (11, 1691)
    assert changed("t5_overhaul", "provider_models.py") == (119, 423)
    assert "provider_prefix = f'{provider}::'" in (
        published / "t5_overhaul" / "expected" / "provider_models.py"
    ).read_text(encoding="utf-8")


def test_a_failing_validation_publishes_nothing(published, tmp_path, monkeypatch):
    out = _copy_published(published, tmp_path)
    (out / "t3_blocks" / "workspace" / "stray.txt").write_text("stale\n", encoding="utf-8")
    monkeypatch.setitem(mf.PINS["t3_blocks"]["files"], "expected/shell_parse.py", "0" * 64)

    problems = mf.materialize(out)

    assert any(p.startswith("t3_blocks: expected/shell_parse.py sha256") for p in problems), problems
    assert (out / "t3_blocks" / "workspace" / "stray.txt").exists()
    assert not list(tmp_path.glob("fixtures_v2.staging-*"))
    assert not (tmp_path / "fixtures_v2.previous").exists()


# ---------------------------------------------------------------------------
# Runner admission
# ---------------------------------------------------------------------------

def _point_v2_tasks_at(monkeypatch, root: pathlib.Path) -> None:
    for name in V2_NAMES:
        task = dict(run_editbench.TASKS[name])
        task["workspace"] = root / name / "workspace"
        task["expected"] = root / name / "expected"
        monkeypatch.setitem(run_editbench.TASKS, name, task)


def _forbid_paid_steps(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a refused fixture must stop the run before this step")

    for name in ("_clone_working_tree", "_seed_settings", "IsolatedServer", "_one_run"):
        monkeypatch.setattr(run_editbench, name, forbidden)


def _run_main(monkeypatch, out: pathlib.Path, task: str) -> int:
    monkeypatch.setattr(sys, "argv", ["run_editbench.py", "--tasks", task, "--runs", "1",
                                      "--out", str(out), "--allow-dirty-seed"])
    return run_editbench.main()


@pytest.mark.parametrize("shape", ["empty_workspace", "old_generator_leftover", "never_materialized"])
def test_a_missing_or_half_built_fixture_is_a_recorded_refusal_before_any_paid_step(shape, tmp_path, monkeypatch):
    """No expected tree: the build that died half-way, or no build at all."""
    root = tmp_path / "fixtures_v2"
    workspace = root / "t2_surgical" / "workspace"
    if shape != "never_materialized":
        workspace.mkdir(parents=True)
    if shape == "old_generator_leftover":  # what the pre-v2.1 generator left: a drifted file, no expected/
        (workspace / "review_state.py").write_bytes(_drifted_review_state())
    _point_v2_tasks_at(monkeypatch, root)
    _forbid_paid_steps(monkeypatch)

    assert _run_main(monkeypatch, tmp_path / "out", "t2_surgical") == 2

    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text(encoding="utf-8"))
    extra = manifest["extra"]
    assert (extra["outcome"], extra["exit_code"]) == ("refused", 2)
    assert extra["refusal"]["stage"] == "fixture_preflight"
    problems = extra["refusal"]["problems"]
    assert problems[0] == "t2_surgical: missing expected/review_state.py"
    if shape == "old_generator_leftover":
        assert len(problems) == 2 and problems[1].startswith("t2_surgical: workspace/review_state.py sha256 ")
    else:
        assert problems[1:] == ["t2_surgical: missing workspace/review_state.py"]
    assert "workloads" not in manifest and "recovery_checkpoint" not in extra


def test_stale_fixture_content_is_a_recorded_refusal_before_any_paid_step(tmp_path, monkeypatch):
    root = tmp_path / "fixtures_v2"
    (root / "t2_surgical" / "expected").mkdir(parents=True)
    (root / "t2_surgical" / "workspace").mkdir()
    (root / "t2_surgical" / "workspace" / "review_state.py").write_bytes(_drifted_review_state())
    _point_v2_tasks_at(monkeypatch, root)
    _forbid_paid_steps(monkeypatch)

    assert _run_main(monkeypatch, tmp_path / "out", "t2_surgical") == 2

    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text(encoding="utf-8"))
    extra = manifest["extra"]
    assert (extra["outcome"], extra["exit_code"]) == ("refused", 2)
    assert extra["refusal"]["stage"] == "fixture_preflight"
    problems = extra["refusal"]["problems"]
    assert "t2_surgical: missing expected/review_state.py" in problems
    assert any(p.startswith("t2_surgical: workspace/review_state.py sha256") for p in problems)
    assert "workloads" not in manifest


def test_preflight_admits_the_pinned_workload_and_names_its_version(published, tmp_path, monkeypatch):
    _point_v2_tasks_at(monkeypatch, published)

    workloads, problems = run_editbench._fixture_preflight(["t1_rename", *V2_NAMES])

    assert problems == []
    assert workloads["t1_rename"]["version"] == "v1"
    assert {workloads[name]["version"] for name in V2_NAMES} == {mf.WORKLOAD_VERSION}
    assert workloads["t3_blocks"]["digest"] == mf.task_identity("t3_blocks")["digest"]
    grader = hashlib.sha256(mf.CHECKER.read_bytes()).hexdigest()
    assert {workloads[name]["checker_sha256"] for name in V2_NAMES} == {grader}

    # The shared grader is part of the digest: an edited checker.py keeps the label, not the digest.
    edited = tmp_path / "checker.py"
    edited.write_bytes(mf.CHECKER.read_bytes() + b"# edited grading rules\n")
    monkeypatch.setattr(mf, "CHECKER", edited)
    regraded = mf.task_identity("t3_blocks")
    assert regraded["version"] == mf.WORKLOAD_VERSION
    assert regraded["digest"] != workloads["t3_blocks"]["digest"]


class _StopAtClone(Exception):
    pass


def test_an_admitted_run_has_its_workload_identities_on_disk_before_the_clone(published, tmp_path, monkeypatch):
    """A run killed during execution must still name its tasks: they are persisted first."""
    _point_v2_tasks_at(monkeypatch, published)
    _forbid_paid_steps(monkeypatch)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # where main() puts its run root
    out = tmp_path / "out"
    at_clone: dict = {}

    def clone_boundary(_run_root):
        at_clone.update(json.loads((out / "run_manifest.json").read_text(encoding="utf-8")))
        raise _StopAtClone

    monkeypatch.setattr(run_editbench, "_clone_working_tree", clone_boundary)

    with pytest.raises(_StopAtClone):
        _run_main(monkeypatch, out, "t1_rename,t2_surgical")

    assert (at_clone["extra"]["outcome"], at_clone["extra"]["exit_code"]) == ("running", None)
    assert at_clone["extra"]["recovery_checkpoint"]["phase"] == "fixtures_validated"
    assert at_clone["workloads"] == {
        "t1_rename": {"version": "v1", "source": "tracked fixtures/"},
        "t2_surgical": mf.task_identity("t2_surgical"),
    }
    final = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert final["extra"]["outcome"] == "crashed" and final["workloads"] == at_clone["workloads"]


# ---------------------------------------------------------------------------
# Today's tools on the fixed workload (deterministic, no model)
# ---------------------------------------------------------------------------

def _registry_on(tmp_path, monkeypatch, workspace: pathlib.Path):
    import ouroboros.safety as safety
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    system, drive = tmp_path / "system", tmp_path / "drive"
    system.mkdir()
    drive.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=system, check=True)
    registry = ToolRegistry(repo_dir=system, drive_root=drive)
    registry.set_context(ToolContext(repo_dir=system, system_repo_dir=system, drive_root=drive,
                                     workspace_root=workspace, workspace_mode="external"))
    monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))
    return registry


def _scripted_calls(name: str, seed_root: pathlib.Path, oracle_root: pathlib.Path) -> list[tuple[str, dict]]:
    if name == "t2_surgical":
        return [("edit_batch", {"edits": [
            {"path": "review_state.py", "old_str": "_stable_digest", "new_str": "_content_digest", "count": 6},
            {"path": "review_state.py", "old_str": "_max_iso_ts", "new_str": "_latest_iso_ts", "count": 3},
            {"path": "review_state.py", "old_str": "_MAX_RUN_HISTORY = 10", "new_str": "_MAX_RUN_HISTORY = 25", "count": 1},
            {"path": "review_state.py", "old_str": "_REVIEW_ATTEMPT_TTL_SEC = 1800",
             "new_str": "_REVIEW_ATTEMPT_TTL_SEC = 2400", "count": 1},
        ]})]
    if name == "t3_blocks":
        seed = (seed_root / "shell_parse.py").read_text(encoding="utf-8")
        hunks = []
        for function, block in mf.T3_REPLACEMENTS.items():
            hunks += ["@@"] + [f"-{line}" for line in mf._function_block(seed, function).splitlines()]
            hunks += [f"+{line}" for line in block.splitlines()]
        patch = "\n".join(["*** Begin Patch", "*** Update File: shell_parse.py", *hunks, "*** End Patch"])
        return [("apply_patch", {"patch": patch})]
    if name == "t4_move":
        block = mf._function_block((seed_root / "ouroboros" / "shell_parse.py").read_text(encoding="utf-8"),
                                   "collect_leading_env")
        policy = "ouroboros/git_shell_policy.py"
        private = block.replace("def collect_leading_env(", "def _collect_leading_env(", 1)
        return [
            ("edit_text", {"path": "ouroboros/shell_parse.py", "old_str": block + "\n\n\n", "new_str": ""}),
            ("edit_text", {"path": policy, "old_str": "    collect_leading_env,\n", "new_str": ""}),
            ("edit_text", {"path": policy, "old_str": "def _git_subcommand_and_args(",
                           "new_str": private + "\n\n\ndef _git_subcommand_and_args("}),
            ("edit_text", {"path": policy, "old_str": "env_assigns, command = collect_leading_env(segment)",
                           "new_str": "env_assigns, command = _collect_leading_env(segment)"}),
        ]
    oracle = (oracle_root / "provider_models.py").read_text(encoding="utf-8")
    return [("write_file", {"path": "provider_models.py", "content": oracle})]


def test_todays_edit_tools_solve_the_pinned_task_and_the_checker_agrees(subtests, published, tmp_path, monkeypatch):
    case_paths = case_folders(tmp_path)
    for name in V2_NAMES:
        tmp_path = next(case_paths)
        with subtests.test(name=name), monkeypatch.context() as monkeypatch:
            task = run_editbench.TASKS[name]
            seed_root, oracle_root = published / name / "workspace", published / name / "expected"
            workspace = run_editbench._make_workspace(tmp_path, name, seed_root)
            registry = _registry_on(tmp_path, monkeypatch, workspace)

            for tool, args in _scripted_calls(name, seed_root, oracle_root):
                result = str(registry.execute(tool, args))
                assert result.startswith("✅"), f"{tool}: {result[:400]}"

            graded = checker.grade_generic(workspace, oracle_root, task["files"],
                                           check_argv=task["check"], check_pythonpath=task["check_pythonpath"])
            assert graded["pass"], json.dumps(graded, indent=2)[:2000]


def test_the_checker_rejects_an_overreaching_rename(published, tmp_path, monkeypatch):
    seed_root = published / "t2_surgical" / "workspace"
    workspace = run_editbench._make_workspace(tmp_path, "t2_trap", seed_root)
    registry = _registry_on(tmp_path, monkeypatch, workspace)
    calls = _scripted_calls("t2_surgical", seed_root, published / "t2_surgical" / "expected")
    calls[0][1]["edits"].append({"path": "review_state.py", "old_str": "_min_iso_ts",
                                 "new_str": "_earliest_iso_ts", "count": 2})
    assert str(registry.execute(*calls[0])).startswith("✅")

    graded = checker.grade_generic(workspace, published / "t2_surgical" / "expected", ["review_state.py"])
    assert not graded["pass"]
    assert graded["files"]["review_state.py"]["status"] == "mismatch"
