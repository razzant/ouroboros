"""Run the owner-send browser reducer journeys as a focused, supervised test."""
from pathlib import Path
import shutil

import pytest

from tests.group_run import run_in_group


def test_owner_send_confirmation_journeys():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for owner-send DOM journeys")
    result = run_in_group(
        [node, "--test", "--test-force-exit", "web/tests/owner_input_saved.test.js"],
        cwd=Path(__file__).resolve().parents[1], timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
