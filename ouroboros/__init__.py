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


def _private_environment_hook() -> None:
    """Run a bare interpreter's entry in the private packages an Update selected for it.

    Standard library only, before the adoption hook, so a redirected entry is never
    recorded as an entry of a switched tree. ``<data>/state/python-private``
    (``update_prepare.private_environment_root``) exists while an Update keeps a
    generation's own packages selected there; ``ouroboros-private-environment.json``
    names the checkout they were made for. An entry of that checkout on exactly the
    interpreter that environment runs (the same real file; on Windows the Python its
    ``pyvenv.cfg`` names, as its interpreter is a copied redirector), outside any venv,
    executes itself there with its own arguments (``exec``: the same process, its parent's
    custody unchanged). Windows has no exec: the entry runs as the waited-on child of this
    process (``body_switch._wait_as_parent``: its base Python in that venv, quoted argv, the
    console's interrupts its own, its exit status this process's). Any other entry runs as it is.
    """
    import json
    import os
    import sys

    if os.environ.pop("OUROBOROS_PRIVATE_ENVIRONMENT_ENTRY", "") or getattr(sys, "frozen", False) \
            or sys.prefix != sys.base_prefix:
        return
    windows = sys.platform == "win32"
    data = os.environ.get("OUROBOROS_DATA_DIR", os.path.join(  # config.resolve_data_dir, before any import
        os.environ.get("OUROBOROS_APP_ROOT", os.path.join(os.path.expanduser("~"), "Ouroboros")), "data"))
    root = os.path.join(data, "state", "python-private")
    python = os.path.join(root, "Scripts", "python.exe") if windows else os.path.join(root, "bin", "python")
    try:
        with open(os.path.join(root, "ouroboros-private-environment.json"), encoding="utf-8") as fh:
            checkout = json.load(fh)["repo_dir"]
        runs = python
        if windows:  # runtime_venv.made_from
            with open(os.path.join(root, "pyvenv.cfg"), encoding="utf-8") as fh:
                config = {key.strip(): value.strip() for key, _, value in (line.partition("=") for line in fh)}
            runs = config.get("executable") or os.path.join(config["home"], "python.exe")
    except (OSError, ValueError, KeyError, TypeError):
        return
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    same = os.path.normcase(os.path.realpath(runs)) == os.path.normcase(os.path.realpath(sys.executable))
    if checkout != os.path.realpath(here) or not same:
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass
    os.environ["OUROBOROS_PRIVATE_ENVIRONMENT_ENTRY"] = "1"  # read once by the entry it starts: never a loop
    try:
        if windows:
            from ouroboros import body_switch

            os._exit(body_switch._wait_as_parent([sys.executable, *sys.orig_argv[1:]], python))
        os.execv(python, [python, *sys.orig_argv[1:]])
    except OSError as exc:
        sys.stderr.write("[ouroboros] the private environment %s could not start (%s); refusing to run this checkout "
                         "on the interpreter's own packages. Moving that directory aside runs them.\n" % (root, exc))
        sys.exit(3)


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


_private_environment_hook()
del _private_environment_hook
_body_adoption_hook()
del _body_adoption_hook

from .version import get_version

__version__ = get_version()
