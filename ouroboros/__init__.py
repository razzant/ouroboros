"""
Ouroboros — self-modifying AI agent.

Philosophy: BIBLE.md
Architecture: agent.py (orchestrator), tools/ (plugin tools),
              llm.py (LLM client), memory.py (memory), review.py (deep review),
              utils.py (shared utilities).
"""

# IMPORTANT: Do NOT import agent/loop/llm/etc here!
# Eager imports here persist in worker processes as stale code,
# preventing hot-reload. Workers import make_agent directly.

__all__ = ['agent', 'tools', 'llm', 'memory', 'review', 'utils']


def _body_adoption_hook() -> None:
    """Finish an ARMED body adoption before any other body module is imported.

    Standard library only, and the first thing every entry reaches (server.py,
    the module CLI, the source/Android launcher, a packaged launcher starting
    repo/server.py). An ordinary boot pays one failed ``open``. When the pointer
    ``body_adoption.arm`` published in this checkout's Git dir exists, the switch
    helper CAPTURED outside the tree at authorize time runs — never the tree's own
    copy, which may be mid-switch. A pending transition whose helper is unreadable
    stops the boot: a possibly mixed tree is not imported. This text must stay
    compatible across versions; ``body_adoption`` refuses a body without it.
    """
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git_dir = os.path.join(root, ".git")
    try:
        if os.path.isfile(git_dir):  # a linked worktree: ".git" is a pointer file
            with open(git_dir, encoding="utf-8") as fh:
                line = fh.readline().strip()
            git_dir = line[len("gitdir:"):].strip() if line.startswith("gitdir:") else ""
            if git_dir and not os.path.isabs(git_dir):
                git_dir = os.path.normpath(os.path.join(root, git_dir))
        with open(os.path.join(git_dir, "ouroboros-body-adoption"), encoding="utf-8") as fh:
            helper = os.path.join(fh.read().strip(), "switch.py")
    except OSError:
        return
    try:
        with open(helper, "rb") as fh:
            code = compile(fh.read(), helper, "exec")
    except (OSError, SyntaxError, ValueError) as exc:
        sys.stderr.write("[ouroboros] a body adoption is pending in %s but its helper %s is unusable (%s); "
                         "refusing to import a possibly mixed tree.\n" % (git_dir, helper, exc))
        sys.exit(3)
    exec(code, {"__name__": "__ouroboros_body_switch__", "__file__": helper, "SERVING_ROOT": root})


_body_adoption_hook()
del _body_adoption_hook

from .version import get_version

__version__ = get_version()
