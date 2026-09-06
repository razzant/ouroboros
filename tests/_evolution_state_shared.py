"""Builders and stubs shared by the evolution state-integrity suites.

Split out of ``tests/test_evolution_state_integrity_v3.py`` when that module was divided by
theme; every definition is verbatim, so each sibling suite keeps the exact campaign and
transaction shape it was written against.
"""

from __future__ import annotations

import pathlib


def _patch_commit_seam(monkeypatch, name, value):
    """Stub one reviewed-commit seam on every module that resolves it.

    ``_repo_commit_push`` spans ``tools/git.py`` and, on trees where the v7
    split landed, its extracted owners (``git_review_cycle``, ``git_evolution``),
    so a seam stub has to reach whichever module resolves the name at call time.
    On the upstream layout only ``tools/git.py`` exists; the extracted owners
    are patched when importable.
    """
    from ouroboros.tools import git as git_tools

    modules = [git_tools]
    try:
        from ouroboros.tools import git_evolution, git_review_cycle
        modules.extend((git_review_cycle, git_evolution))
    except ImportError:
        pass

    for module in modules:
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)


def _active_transaction(tmp_path: pathlib.Path, task_id: str = "evo-task"):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    queue.init(tmp_path)
    queue.init_queue_refs([], {}, {"value": 0})
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    live = state.load_state()
    live.update({
        "owner_chat_id": 1,
        "evolution_mode_enabled": True,
        "evolution_owner_stopped": False,
    })
    state.save_state(live)
    tx = evolution_lifecycle.begin_evolution_transaction(task_id, cycle=1, campaign=campaign)
    return campaign, tx


class _CaptureQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)
