"""The real provider selections partition the matrix without forgiving producer exits."""
from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.provider_contract_ci import provider_canary_matrix
from tests.provider_release_floor import required_canary_ids
from tests.test_ci_diagnostic_workflows import CANARY, ROOT, _run_shell, _steps, _workflow


@pytest.mark.serial
def test_real_canary_collection_is_nonempty_disjoint_and_complete():
    # Collection executes pytest's real parametrization and marker selection, not a
    # second Python filter. Fixtures (and therefore provider calls) never execute.
    prefix = "tests/test_provider_integration.py::test_full_registry_provider_contract["
    matrix = provider_canary_matrix()
    expected = {prefix + canary.canary_id + "]" for canary in matrix}
    required = {prefix + canary_id + "]" for canary_id in required_canary_ids()}
    selections = {
        "integration": expected,
        "integration and required_provider": required,
        "integration and optional_provider": expected - required,
        "integration and provider_typo": set(),
    }
    assert required and expected - required
    for selection, wanted in selections.items():
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_provider_integration.py",
             "--collect-only", "-q", "-o", "addopts=", "-m", selection],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=120,
        )
        assert done.returncode == (0 if wanted else 5), (selection, done.stdout, done.stderr)
        rows = [line for line in done.stdout.splitlines() if line.startswith(prefix)]
        assert len(rows) == len(wanted) and set(rows) == wanted, (selection, rows)


@pytest.mark.parametrize("scope,selection", [
    ("all", "integration"),
    ("required", "integration and required_provider"),
    ("optional", "integration and optional_provider"),
])
@pytest.mark.parametrize("result", [0, 1, 2, 3, 4, 5])
def test_actual_scoped_shell_preserves_each_pytest_exit(tmp_path, scope, selection, result):
    step = _steps(CANARY, "integration-test")["provider_tests"]
    assert step["env"]["SCOPE"] == "${{ inputs.scope }}"
    assert "inputs.scope" not in step["run"]  # Input is data, never shell syntax.
    done, args = _run_shell(step, tmp_path, scope=scope, result=result)
    assert done.returncode == result, done.stderr
    assert args[args.index("-m", 2) + 1] == selection
    assert args[:3] == ["-m", "pytest", "tests/test_provider_integration.py"]


@pytest.mark.parametrize("scope", ["", "REQUIRED", "unrecognized", "all; touch injected"])
def test_unknown_scope_refuses_before_launch(tmp_path, scope):
    done, args = _run_shell(_steps(CANARY, "integration-test")["provider_tests"], tmp_path, scope=scope)
    assert done.returncode == 2 and args == []
    assert not (tmp_path / "injected").exists()


@pytest.mark.parametrize("scope", ["all", "required", "optional"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_optional_job_never_misreports_missing_required_results(scope, cancelled):
    floor = _workflow(CANARY)["jobs"]["integration-test"]["steps"][-1]
    expression = floor["if"].strip()[3:-2].replace("&&", " and ").replace("!cancelled()", "not cancelled()")
    actual = eval(expression, {"__builtins__": {}}, {
        "inputs": SimpleNamespace(scope=scope), "cancelled": lambda: cancelled,
    })
    assert actual is (not cancelled and scope != "optional")


def test_optional_failure_cannot_enter_the_release_dependency_graph():
    jobs = _workflow("ci.yml")["jobs"]
    visited = set()
    pending = ["release"]
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        needs = jobs[name].get("needs", [])
        pending.extend([needs] if isinstance(needs, str) else needs)
    assert "integration-test" in visited and "integration-optional" not in visited
    # The release job reads the required result itself; GitHub's implicit success() must not
    # suppress that explicit consumer merely because an independent optional job failed.
    release = jobs["release"]
    assert "always()" in release["if"] and "!cancelled()" in release["if"]
    assert "needs.integration-test.result == 'success'" in release["if"]
    assert "integration-optional" not in release["if"]


@pytest.mark.parametrize("required", ["success", "failure", "cancelled", "skipped", ""])
@pytest.mark.parametrize("optional", ["success", "failure", "cancelled", "skipped"])
def test_actual_release_bar_uses_required_job_result_not_optional(required, optional):
    from tests.test_windows_signing_release import expression_value

    release = _workflow("ci.yml")["jobs"]["release"]
    context = {"github.ref": "refs/tags/v7.7.2", "github.repository": "razzant/ouroboros", "cancelled": False,
               **{f"needs.{name}.result": "success" for name in release["needs"]},
               "needs.integration-test.result": required, "needs.integration-optional.result": optional}
    assert bool(expression_value(release["if"], context)) is (required == "success")
