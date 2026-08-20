"""``supervisor.git_ops`` pre-``init`` roots follow the environment, never a hard-coded home path.

A process that imports git_ops without calling ``init`` (isolated tests, smokes) must
resolve its supervisor log and repo roots under the configured ``OUROBOROS_*`` roots —
otherwise a test-context write lands in the live ``~/Ouroboros/data`` drive.

The property is about the PRE-``init`` state, so it is proved in a fresh subprocess:
in-process, any earlier test that legitimately called ``state.init``/``git_ops.init``
on a scratch root leaves the module globals rebound, and asserting the default there
is an order-dependent lie (caught as a cross-worker flake by the S7b split).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

_PROBE = """
import json, pathlib
from ouroboros import config
from supervisor import git_ops, state
print(json.dumps({
    "git_ops_drive": str(git_ops.DRIVE_ROOT),
    "state_drive": str(state.DRIVE_ROOT),
    "config_data": str(config.DATA_DIR),
    "git_ops_repo": str(git_ops.REPO_DIR),
    "home_data": str(pathlib.Path.home() / "Ouroboros" / "data"),
    "home_repo": str(pathlib.Path.home() / "Ouroboros" / "repo"),
}))
"""


def test_git_ops_default_drive_root_follows_config_not_home(tmp_path) -> None:
    env = dict(os.environ)
    env.update({
        "OUROBOROS_APP_ROOT": str(tmp_path),
        "OUROBOROS_REPO_DIR": str(tmp_path / "repo"),
        "OUROBOROS_DATA_DIR": str(tmp_path / "data"),
        "OUROBOROS_SETTINGS_PATH": str(tmp_path / "data" / "settings.json"),
    })
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, check=True,
    )
    roots = json.loads(out.stdout)
    # The drive root is the invariant that decides where supervisor rows land.
    assert roots["git_ops_drive"] == roots["config_data"] == roots["state_drive"]
    assert roots["git_ops_drive"] == str(tmp_path / "data")
    assert roots["git_ops_drive"] != roots["home_data"]
    assert roots["git_ops_repo"] != roots["home_repo"]
