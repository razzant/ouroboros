"""Shared fixtures of the system_e2e scenario package.

One throwaway clone serves every scenario server of a session (cloning the
checkout is the expensive step); scenarios that MOVE the clone's HEAD or add
remotes (the managed-update wave) must build their own private clone instead —
sharing a mutated clone would couple scenario outcomes to execution order.
"""

from __future__ import annotations

import pytest

from tests.system_e2e.harness import LANE_MOCK, clone_repo, require_lane


@pytest.fixture(scope="session")
def e2e_clone(tmp_path_factory):
    """One throwaway clone of the checkout under test, shared by every scenario
    server that leaves HEAD alone."""
    require_lane(LANE_MOCK)
    return clone_repo(tmp_path_factory.mktemp("system_e2e_clone"))
