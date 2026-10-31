"""Shared fixtures of the system_e2e scenario package.

One throwaway clone serves every scenario server of a session (cloning the
checkout is the expensive step); scenarios that MOVE the clone's HEAD or add
remotes (the managed-update wave) must build their own private clone instead —
sharing a mutated clone would couple scenario outcomes to execution order.

Every mock-lane scenario test (a ``test_s<N>_*`` test of this package whose
``SCENARIOS`` row names the mock lane) also gets one per-test ceiling,
``SCENARIO_TIMEOUT_SEC``, unless it carries its own ``timeout`` marker: a hung
scenario then fails BY NAME and the lane runs on, instead of one blind wait
starving the job until the runner cancels it with no result for any later
scenario (S37 before the 7.7.1 repair). On POSIX the ceiling fails only the
running test (pytest-timeout's SIGALRM method), so the scenario's own
``finally`` still stops its server. The ceiling counts the test's setup too, so
the first scenario also carries the session clone. The operator-run container
lane (S39 on real Docker) keeps its own pace.
"""

from __future__ import annotations

import pathlib
import re
import signal

import pytest

from tests.system_e2e.harness import LANE_MOCK, SCENARIOS, clone_repo, require_lane

_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
_SCENARIO_TEST_RE = re.compile(r"^test_(s\d+)_")
# The slowest healthy scenario, measured on a loaded developer Mac, with room for a slow runner.
SCENARIO_TIMEOUT_SEC = 900
_TIMEOUT_METHOD = "signal" if hasattr(signal, "SIGALRM") else "thread"


def _mock_lane_scenario(item) -> bool:
    match = _SCENARIO_TEST_RE.match(str(getattr(item, "originalname", "") or item.name))
    return (match is not None and SCENARIOS.get(match[1].upper(), ("", ""))[1] == LANE_MOCK
            and _PACKAGE_DIR in pathlib.Path(str(item.path)).resolve().parents)


def pytest_collection_modifyitems(config, items):  # noqa: ARG001 - pytest hook signature
    for item in items:
        if item.get_closest_marker("timeout") is None and _mock_lane_scenario(item):
            item.add_marker(pytest.mark.timeout(SCENARIO_TIMEOUT_SEC, method=_TIMEOUT_METHOD))


@pytest.fixture(scope="session")
def e2e_clone(tmp_path_factory):
    """One throwaway clone of the checkout under test, shared by every scenario
    server that leaves HEAD alone."""
    require_lane(LANE_MOCK)
    return clone_repo(tmp_path_factory.mktemp("system_e2e_clone"))
