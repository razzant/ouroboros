"""The diff pulse's code-shape facts over one Python source (``ouroboros/code_shape_facts.py``):
what a module handle, a call through one, a re-export, a patched product-module name and a
``git init`` are, and how a change's two sides become one fact line (never a gate)."""

from __future__ import annotations

import pathlib
import textwrap

from ouroboros.code_shape_facts import module_names, pulse_line, source_shape

REPO = pathlib.Path(__file__).resolve().parents[1]
MODULES = module_names(["ouroboros/__init__.py", "ouroboros/loop.py", "ouroboros/tools/__init__.py",
                        "ouroboros/tools/git.py", "ouroboros/config.py", "supervisor/queue.py", "tests/_helpers.py"])


def _shape(source: str, path: str = "ouroboros/leaf.py", modules: frozenset[str] = MODULES) -> dict:
    return dict(source_shape(textwrap.dedent(source), path, modules))


def test_a_module_handle_is_a_zero_argument_function_returning_the_module_it_imports():
    shape = _shape('''
        def _loop():
            """The parent, read at call time."""
            from ouroboros import loop

            return loop


        def _git():
            from . import git
            return git


        def _queue():
            import supervisor.queue as queue
            return queue


        def _const():  # a lazy import of a value returns no module
            from ouroboros.loop import LIMIT
            return LIMIT


        def _with(arg=None):
            from ouroboros import loop
            return loop


        async def _later():
            from ouroboros import loop
            return loop


        def _busy():
            from ouroboros import loop
            loop.touch()
            return loop


        def run():
            _loop().step(); _loop().LIMIT = 3; _git().run_cmd(); _queue().PENDING; _const().real; _loop()()
            q = _queue()  # a local holding the module is not followed (the extraction table's unit)
            q.RUNNING


        def shadowed(_loop):  # a parameter hides the module-level handle
            return _loop().step
    ''', path="ouroboros/tools/leaf.py")
    assert shape == {("handle", "_loop"): 1, ("handle", "_git"): 1, ("handle", "_queue"): 1,
                     ("call", "_loop", "step"): 1, ("call", "_loop", "LIMIT"): 1, ("call", "_git", "run_cmd"): 1,
                     ("call", "_queue", "PENDING"): 1}


def test_every_leaf_the_extraction_table_pins_reads_as_its_handle_and_its_declared_calls():
    """The shape is the repository's own: each leaf of ``test_module_handle_extraction.LEAVES`` holds its
    handle once (a leaf may hold another, such as ``dispatch_choice._assignment``), and the calls through
    it are exactly the names the table declares."""
    from tests.test_module_handle_extraction import LEAVES

    tracked = module_names(path.relative_to(REPO).as_posix() for root in ("ouroboros", "supervisor")
                           for path in (REPO / root).rglob("*.py"))
    for leaf, (_parent, handle, declared) in LEAVES.items():
        shape = source_shape((REPO / leaf).read_text(encoding="utf-8"), leaf, tracked)
        assert shape[("handle", handle)] == 1, leaf
        assert {item[2] for item in shape if item[:2] == ("call", handle)} == set(declared), leaf


def test_a_reexport_is_a_name_a_marked_top_level_from_import_binds():
    shape = _shape('''
        from ouroboros.loop import (  # noqa: F401 - compatibility re-export
            run_loop,
            step as one_step,
        )
        from ouroboros.config import DATA_DIR  # noqa: E501
        from . import git  # noqa
        import ouroboros.config  # noqa: F401


        def inner():
            from ouroboros.loop import hidden  # noqa: F401
    ''', path="ouroboros/tools/facade.py")
    # Each name of a marked statement, by its resolved source and the name it binds (the facade
    # inventory's unit and reader).
    assert shape == {("re-export", "ouroboros.loop", "run_loop", "run_loop"): 1,
                     ("re-export", "ouroboros.loop", "step", "one_step"): 1,
                     ("re-export", "ouroboros.tools", "git", "git"): 1}
    for spelling in ("git", "git as git"):  # absolute and relative spellings, a redundant alias: one name
        assert _shape(f"from ouroboros.tools import {spelling}  # noqa: F401\n", path="ouroboros/tools/facade.py") == {
            ("re-export", "ouroboros.tools", "git", "git"): 1}


def test_a_test_source_counts_patched_product_names_and_git_init_call_sites():
    shape = _shape('''
        import importlib
        import subprocess
        from unittest import mock
        from unittest.mock import patch

        from ouroboros import config
        from ouroboros.tools import git as git_tool
        from ouroboros.loop import Runner
        import supervisor.queue
        from tests import _helpers as shared

        real_run = subprocess.run


        def _git(repo, *args):  # a Git helper this module defines
            return subprocess.run(["git", "-C", str(repo), *args], check=True)


        def run(cmd, cwd=None):  # a command runner this module defines
            return subprocess.run(cmd, cwd=cwd, check=True)


        @mock.patch("ouroboros.loop._step")
        def test_x(monkeypatch, tmp_path):
            monkeypatch.setattr("ouroboros.loop.LIMIT", 3)
            monkeypatch.setattr(config, "DATA_DIR", tmp_path)
            monkeypatch.delattr(git_tool, "run_cmd")
            monkeypatch.setattr(supervisor.queue, "PENDING", [])
            monkeypatch.setattr(Runner, "step", None)      # a class a product module defines
            monkeypatch.setattr(config.os, "environ", {})  # a name reached through a product module
            monkeypatch.setattr("os.environ", {})          # not a product module
            monkeypatch.setattr(subprocess, "run", None)   # nor this one
            setattr(config, "OTHER", 1)                    # the builtin is no monkeypatch
            with patch.object(config, "LOADED", True), patch("ouroboros.config.load"):
                pass
            client = object()
            client.patch("/api/x")                         # an HTTP PATCH, not a patch target
            monkeypatch.setattr(client, "x", 1)            # a local object, not a product name
            subprocess.run(["git", "init", "-q"], cwd=tmp_path)
            subprocess.run(["git", "-C", str(tmp_path), "-c", "init.defaultBranch=main", "init"])
            subprocess.run(["git", "-c", "user.name=T", "commit", "-m", "init"])  # a commit named init
            _git(tmp_path, "init", "-q")
            shared.git(tmp_path, "init")
            run(["git", "init"], cwd=tmp_path)
            real_run(["git", "init", "--bare", "remote"])
            subprocess.check_call("cd x && git init -q", shell=True)
            for args in (("init", "-q"), ("add", "-A")):
                subprocess.run(["git", *args], cwd=tmp_path)
            for cmd in (["git", "init"], ["git", "status"]):
                subprocess.run(cmd, cwd=tmp_path)
            pytest.param("git init /tmp/proj", id="a command as data, never run")
            CASES = [pytest.param(["git", "init"]), ("git", "init")]  # data too
            fake_run(["git", "init"])                      # nothing here runs it


        def test_scopes(monkeypatch, config):
            monkeypatch.setattr(config, "DATA_DIR", None)  # the parameter hides the module's import
            review = importlib.import_module("ouroboros.review")
            monkeypatch.setattr(review, "subprocess", None)

            def inner():
                monkeypatch.setattr(review, "tempfile", None)  # an enclosing function's binding

            from supervisor import queue as config_q
            monkeypatch.setattr(config_q, "RUNNING", {})


        def unrelated():
            import os as git_tool  # rebinds the name in this function only
            return git_tool
    ''', path="tests/test_leaf.py")
    assert shape == {("patch", "ouroboros.loop._step"): 1, ("patch", "ouroboros.loop.LIMIT"): 1,
                     ("patch", "ouroboros.config.DATA_DIR"): 1, ("patch", "ouroboros.tools.git.run_cmd"): 1,
                     ("patch", "supervisor.queue.PENDING"): 1, ("patch", "ouroboros.loop.Runner.step"): 1,
                     ("patch", "ouroboros.config.os.environ"): 1, ("patch", "ouroboros.config.LOADED"): 1,
                     ("patch", "ouroboros.config.load"): 1, ("patch", "ouroboros.review.subprocess"): 1,
                     ("patch", "ouroboros.review.tempfile"): 1, ("patch", "supervisor.queue.RUNNING"): 1,
                     ("git init",): 9}
    # The groups follow the path: a product module holds no test facts, a test source no handles.
    assert _shape('subprocess.run(["git", "init"])\nmonkeypatch.setattr("ouroboros.loop.X", 1)\n') == {}
    assert _shape("def _loop():\n    from ouroboros import loop\n    return loop\n", path="tests/test_a.py") == {}


def test_a_git_init_site_runs_git_init_and_a_helper_counts_by_what_its_body_runs():
    """A runner given ``["git", "init"]``, as its first argument or ``args=``; a helper the module defines
    counts by its body (it passes its varargs to git as the subcommand), a callable whose body is elsewhere
    by its name; building an argument list, or running another subcommand, is no ``git init``."""
    shape = _shape('''
        import subprocess

        from tests import _helpers as shared

        IDENT = ("-c", "user.name=t")
        GIT = ("git", "-c", "user.name=t")


        def argv(*parts):  # builds a list and runs nothing
            return ["git", *parts]


        def _git_argv(*parts):  # named for git, and still runs nothing
            return ["git", *parts]


        def status(repo, *paths):  # runs git status with the paths it is given
            return subprocess.run(["git", "status", "--", *paths], cwd=repo)


        def git_status(repo, *paths):
            return subprocess.run(["git", "status", "--", *paths], cwd=repo)


        def _git(repo, *args):  # its own import, identity options before the varargs
            import subprocess as sp
            return sp.run(["git", *IDENT, *args], cwd=repo, check=True)


        def git(repo, *args):  # named for git, behind a constant git prefix
            return subprocess.run([*GIT, *args], cwd=repo)


        def run(cmd, **kwargs):
            return subprocess.run(args=cmd, **kwargs)


        def test_y(tmp_path, git_fixture):
            CASES = [argv("init"), _git_argv("init")]
            status(tmp_path, "init")
            git_status(tmp_path, "init")
            _git(tmp_path, "init", "-q")
            git(tmp_path, "init")
            subprocess.run(args=["git", "init"], cwd=tmp_path)
            run(["git", "init"])
            shared.run_git(tmp_path, "init")  # imported: its name says what it runs
            git_fixture("init")  # a parameter: so does this one's
    ''', path="tests/test_y.py")
    assert shape == {("git init",): 6}


def test_a_name_resolves_in_its_scope_in_source_order():
    """A comprehension binds its targets inside itself alone; a scope's bindings take effect in source order,
    and a rebinding to a value the module cannot name (an assignment, a match capture) hides an import."""
    shape = _shape('''
        import importlib
        import subprocess

        from ouroboros import config
        from ouroboros import loop

        real_run = subprocess.run


        def run(cmd):  # a command runner through the alias bound above it
            return real_run(cmd)


        def test_x(monkeypatch):
            monkeypatch.setattr(config, "DATA_DIR", None)
            [config for config in ()]  # the comprehension's own target
            [monkeypatch.setattr(config, "LOADED", n) for n in range(2)]  # the enclosing binding, inside one
            [monkeypatch.setattr(item, "x", 1) for item in (loop,)]  # its target is no module
            m = importlib.import_module("ouroboros.loop")
            m2 = m
            monkeypatch.setattr(m2, "LIMIT", 3)
            run(["git", "init"])


        def test_rebound(monkeypatch, value):
            from ouroboros import loop
            loop = object()
            monkeypatch.setattr(loop, "STEP", 3)
            match value:
                case config:
                    monkeypatch.setattr(config, "OTHER", 1)
    ''', path="tests/test_scope.py")
    assert shape == {("patch", "ouroboros.config.DATA_DIR"): 1, ("patch", "ouroboros.config.LOADED"): 1,
                     ("patch", "ouroboros.loop.LIMIT"): 1, ("git init",): 1}


def test_the_line_states_what_the_change_added_and_removed_and_names_what_it_could_not_read():
    leaf_before = "def _loop():\n    from ouroboros import loop\n    return loop\n\n\ndef f():\n    _loop().a\n"
    leaf_after = leaf_before + "    _loop().b\n    _loop().b\n"
    facade = "from ouroboros.loop import a, b  # noqa: F401\n"
    rows = [("ouroboros/leaf.py", leaf_before, leaf_after), ("ouroboros/facade.py", facade, None),
            ("tests/test_a.py", None, 'monkeypatch.setattr("ouroboros.loop.a", 1)\n')]
    assert pulse_line(rows, MODULES, MODULES) == (
        "Code shape: module handles +0/-0, calls through them +2/-0, re-exported names +0/-2; "
        "tests: patched product-module names +1/-0, git init call sites +0/-0.")
    # The same call renamed is one added and one removed, not nothing.
    assert "calls through them +1/-1," in pulse_line(
        [("ouroboros/leaf.py", leaf_before, leaf_before.replace("().a", "().c"))], MODULES, MODULES)
    # Unknown is never zero: a side that does not parse, or was not read, is named instead of counted.
    broken = [("ouroboros/leaf.py", leaf_before, "def broken(:\n"), ("ouroboros/facade.py", facade, None)]
    assert pulse_line(broken, MODULES, MODULES) == (
        "Code shape: module handles +0/-0, calls through them +0/-0, re-exported names +0/-2, "
        "1 unreadable not counted: ouroboros/leaf.py.")
    assert pulse_line(broken[:1], MODULES, MODULES) == (
        "Code shape: module handles, calls through them, re-exported names unknown (1 unreadable: ouroboros/leaf.py).")
    assert pulse_line(rows[2:], MODULES, MODULES, unreadable=["tests/test_a.py"]) == (
        "Code shape: tests: patched product-module names, git init call sites unknown (1 unreadable: tests/test_a.py).")
    assert pulse_line(rows[:1], None, MODULES).endswith("unknown (1 unreadable: ouroboros/leaf.py).")
    assert pulse_line([], MODULES, MODULES) == ""
    # Ordinary valid sources yield known zero counts; real syntax errors leave both groups unknown.
    ordinary = [("ouroboros/ordinary.py", None, "X = 1\n"),
                ("tests/test_ordinary.py", None, "x = 1\n")]
    assert pulse_line(ordinary, MODULES, MODULES) == (
        "Code shape: module handles +0/-0, calls through them +0/-0, re-exported names +0/-0; "
        "tests: patched product-module names +0/-0, git init call sites +0/-0.")
    unparsed = [("ouroboros/broken.py", None, "def broken(:\n"),
                ("tests/test_broken.py", None, "def broken(:\n")]
    assert pulse_line(unparsed, MODULES, MODULES) == (
        "Code shape: module handles, calls through them, re-exported names unknown (1 unreadable: ouroboros/broken.py); "
        "tests: patched product-module names, git init call sites unknown (1 unreadable: tests/test_broken.py).")
