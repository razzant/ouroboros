"""Helpers shared by the workspace-executor suites.

The git repository was split out of ``tests/test_workspace_executor.py`` when that
module was divided by theme; it is now one world of the run, so every sibling suite
keeps the tree, commit and configuration it was written against (the ``.git`` folder
holds no template files). ``fake_docker_cli`` is the one Docker CLI boundary those
suites fake.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from tests._workspace_git import git_repository, world_git
from tests._worlds import clone_repository, world


def fake_docker_cli(monkeypatch, script, *, docker_popen=None, on_docker_stop=None) -> None:
    """Script the workspace executor's Docker CLI, including starts under the Pause gate.

    Short calls and process submissions take their result from the same script.
    A timeout test supplies its fake process and observes its stop without using host process control.
    Other commands and other modules use their normal subprocess implementation.
    """
    from ouroboros import workspace_executor

    class DockerCli:
        pid = None  # No OS process exists for a scripted call.

        def __init__(self, cmd, *args, **kwargs):
            self.args = cmd
            answer = script(cmd, *args, **kwargs)
            self.returncode, self._output = answer.returncode, (answer.stdout, answer.stderr)

        def communicate(self, timeout=None):
            return self._output

    def is_docker(args):
        return isinstance(args, (list, tuple)) and bool(args) and args[0] == "docker"

    scripted = []
    native_stop = workspace_executor.kill_process_tree

    def popen(args, *pos, **kwargs):
        if is_docker(args):
            proc = (docker_popen or DockerCli)(args, *pos, **kwargs)
            scripted.append(proc)
            return proc
        return subprocess.Popen(args, *pos, **kwargs)

    def stop(proc, **kwargs):
        if any(proc is obj for obj in scripted):
            if on_docker_stop is None:
                raise AssertionError("unexpected stop of a simulated Docker CLI")
            return on_docker_stop(proc)
        return native_stop(proc, **kwargs)

    def run(args, *pos, **kwargs):
        if is_docker(args):
            return script(args, *pos, **kwargs)
        return subprocess.run(args, *pos, **kwargs)

    # shell_process uses the identity of the stdlib Popen to retain a group or Job.
    cli = SimpleNamespace(**vars(subprocess))
    cli.run, cli.Popen = run, popen
    monkeypatch.setattr(workspace_executor, "subprocess", cli)
    monkeypatch.setattr(workspace_executor, "kill_process_tree", stop)


def _lay_readme_repo(folder: Path) -> None:
    repo = git_repository(folder / "repo", unborn=True)
    world_git(repo, "config", "user.email", "test@example.invalid")
    world_git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    world_git(repo, "add", "README.md")
    world_git(repo, "commit", "-m", "init")


def _init_repo(path: Path) -> None:
    clone_repository(world("workspace-executor-readme", _lay_readme_repo) / "repo", path)
