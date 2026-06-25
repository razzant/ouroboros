"""The repo-root mock-pollution sweep in conftest.pytest_sessionfinish must run ONLY on the
xdist controller. Under -n, the hook fires on the controller AND every worker against the shared
repo root; without the guard, workers race the same shutil.rmtree and each set session.exitstatus,
manufacturing a non-deterministic failed-shaped run. A worker is identified by config.workerinput.
"""
from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

# The rootdir conftest is already imported by pytest; grab that live module object (importing
# "conftest" by name is not reliable, and re-executing the file would re-run its mkdtemp/env setup).
conftest = next(
    m for m in list(sys.modules.values())
    if getattr(m, "__file__", "") and m.__file__.endswith("tests/conftest.py")
    and hasattr(m, "pytest_sessionfinish") and hasattr(m, "_mock_pollution_files")
)

_REPO_ROOT = pathlib.Path(conftest.__file__).resolve().parents[1]
# A path lexically UNDER the repo root (so the hook's relative_to() works) but NON-EXISTENT, so the
# sweep's is_dir()/unlink(missing_ok=True) touch nothing on disk. The controller-vs-worker decision
# is observed purely through session.exitstatus (set to 1 only when the sweep actually runs).
_FAKE_LEAK = _REPO_ROOT / "__xdist_guard_test__" / "<MagicMock pollution>"


def _session(is_worker: bool) -> SimpleNamespace:
    config = SimpleNamespace(_ouroboros_initial_mock_pollution=set())
    if is_worker:
        config.workerinput = {"workerid": "gw0"}  # xdist sets this on every worker
    return SimpleNamespace(config=config, exitstatus=0)


def test_pollution_sweep_runs_on_controller(monkeypatch):
    monkeypatch.setattr(conftest, "_mock_pollution_files", lambda root: {_FAKE_LEAK})
    monkeypatch.setattr(conftest, "_PYTEST_DATA_DIR", None)  # don't touch the live session data dir

    session = _session(is_worker=False)
    conftest.pytest_sessionfinish(session, 0)

    assert session.exitstatus == 1, "controller must run the sweep and fail the run on pollution"
    assert not _FAKE_LEAK.exists(), "sweep must not have created anything on disk"


def test_pollution_sweep_is_skipped_on_worker(monkeypatch):
    monkeypatch.setattr(conftest, "_mock_pollution_files", lambda root: {_FAKE_LEAK})
    monkeypatch.setattr(conftest, "_PYTEST_DATA_DIR", None)

    session = _session(is_worker=True)
    conftest.pytest_sessionfinish(session, 0)

    # A worker must NOT run the sweep nor mutate exitstatus — the controller owns the repo-root
    # authority, so workers cannot race the rmtree or each independently fail the run.
    assert session.exitstatus == 0
