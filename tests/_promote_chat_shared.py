"""The projects-root isolation fixture shared by the promote/project chat suites.

Split out of ``tests/test_promote_chat_flow.py`` when that module was divided by
theme; the fixture is verbatim and autouse, so importing it into a test module
re-applies it there.
"""

from __future__ import annotations


import pytest


@pytest.fixture(autouse=True)
def _isolated_projects_root(tmp_path_factory, monkeypatch):
    """Q10=A auto-provisions a genesis workspace for file-less project promotes;
    keep it out of the real ~/Ouroboros/projects."""
    monkeypatch.setenv(
        "OUROBOROS_SUBAGENT_PROJECTS_ROOT",
        str(tmp_path_factory.mktemp("projects_root")),
    )
