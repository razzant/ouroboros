"""Keeps the distro's pip policy on while the subject's own tests run.

``tests/test_update_environment.py::offline_pip`` sets ``PIP_BREAK_SYSTEM_PACKAGES=1`` for its user-site case, and
``_pip_env`` passes that on, so a plain run of those fixtures would switch off the very protection this qualification
exists to observe. Fixtures have finished by the call phase, so the variable goes here, before the test body. This only
removes a permission; no assertion, fixture or test is changed. ``PIP_NO_INDEX`` keeps the toy pins away from any index.
"""
from __future__ import annotations

import os
import shutil
import sys

import pytest


def pytest_sessionstart(session):  # noqa: ARG001
    problems = [name for name in ("PYTHONPATH", "PIP_BREAK_SYSTEM_PACKAGES", "VIRTUAL_ENV") if name in os.environ]
    if sys.prefix != sys.base_prefix:
        problems.append(f"sys.prefix={sys.prefix} is a venv")
    if problems:
        pytest.exit(f"not the bare interpreter under test: {problems}", returncode=3)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):  # noqa: ARG001
    os.environ.pop("PIP_BREAK_SYSTEM_PACKAGES", None)
    # interpreter_pip: the runtime lock's pip beside the toy wheels, as test 279 copies one, from the product's own
    # digest-checked fetch instead of ensurepip's bundle (which this interpreter may not have).
    wheel, links = os.environ.get("Q_SEED_PIP_WHEEL"), os.environ.get("PIP_FIND_LINKS")
    if wheel and links:
        shutil.copy(wheel, links)
