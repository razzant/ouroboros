"""Work-domain effects of a managed deployment, at their real consumers (``ouroboros.work_domain``).

The tools, Git, the private copies and a native run's material flow run over a work tree through the fake
container of ``test_execution_domain`` (its ``sh`` is dash, so the reference container's POSIX stdin rule
holds). An audit hook records what the service process itself does: any pathname open, folder or file effect,
or process started in a work folder is a violation; a held-descriptor descent shows only plain names. What the
container's own resolution reaches is the OS's isolation, which this host cannot stand in for: that is the
parent's Linux qualification. No Docker, daemon, network or engine is used.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import subprocess
import sys

import pytest

from ouroboros import work_domain
from tests._safety_fixtures import disable_safety
from tests.test_execution_domain import _registry, attached, fake_docker, section
from tests.test_execution_domain import env as env  # fixture

_EFFECTS = frozenset({"open", "os.mkdir", "os.remove", "os.rename", "os.replace", "os.rmdir", "os.chmod",
                      "shutil.rmtree", "os.listdir", "os.scandir", "os.symlink", "os.truncate", "os.utime"})


@contextlib.contextmanager
def service_effects_on(*roots, allowed_opens=()):
    """What the service process itself does to a path under ``roots`` while active: pathname opens, folder and
    file effects and process starts in such a folder (``sys.addaudithook``; a hook cannot be removed, so it goes
    inert after). ``allowed_opens`` names the mapping roots a confined descent opens by their configured spelling."""
    seen, active = [], [True]
    prefixes = [os.fspath(root) for root in roots]

    def under(value):
        text = os.fspath(value) if isinstance(value, (str, bytes, os.PathLike)) else ""
        text = text.decode("utf-8", "replace") if isinstance(text, bytes) else text
        return any(text == prefix or text.startswith(prefix + "/") for prefix in prefixes)

    def hook(event, args):
        if not active[0]:
            return
        if event == "subprocess.Popen":
            if args[2] and under(args[2]):
                seen.append((event, [str(part) for part in (args[1] or [])][:3], os.fspath(args[2])))
        elif event in _EFFECTS and args:
            named = [arg for arg in args[:2] if isinstance(arg, (str, bytes, os.PathLike)) and under(arg)]
            if named and not (event == "open" and os.fspath(named[0]) in allowed_opens):
                seen.append((event, os.fspath(named[0])))

    sys.addaudithook(hook)
    try:
        yield seen
    finally:
        active[0] = False


def _git(repo, *args):
    env_git = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
               "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env_git).stdout


def _work_repo(path, files):
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    for rel, text in files.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(text, encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "base")
    return path


def _managed(env, monkeypatch, **extra):
    disable_safety(monkeypatch)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    raw = section(env)
    raw.update(extra)
    env.manage(raw)
    return fake_docker(env, monkeypatch)


def test_the_service_performs_no_pathname_effect_on_a_work_tree(env, monkeypatch):
    """Every tool over a managed work repository — write, edit, patch (add, update, delete with its recovery
    copy), read, list (flat and tree), search, Git status, diff and history — works, and the service itself
    opens, writes, removes or starts nothing in that tree by its own pathname: the container does it, and the
    service's reads are held-descriptor descents from the mapping's root."""
    docker = _managed(env, monkeypatch)
    repo = _work_repo(env.work / "app", {"src/main.py": "print('a')\n", "README.md": "readme\n", "old.txt": "gone\n"})
    registry = _registry(env, repo)
    calls = []
    with service_effects_on(env.work, allowed_opens=(str(env.work),)) as seen:
        calls.append(registry.execute("write_file", {"root": "active_workspace", "path": "src/new.py",
                                                     "content": "VALUE = 1\n"}))
        calls.append(registry.execute("edit_text", {"root": "active_workspace", "path": "src/main.py",
                                                    "old_str": "print('a')", "new_str": "print('b')"}))
        calls.append(registry.execute("apply_patch", {"patch": "*** Begin Patch\n*** Add File: added.txt\n+added\n"
                                                      "*** Update File: README.md\n@@\n-readme\n+readme v2\n"
                                                      "*** Delete File: old.txt\n*** End Patch\n"}))
        read = registry.execute("read_file", {"root": "active_workspace", "path": "src/main.py"})
        flat = registry.execute("list_files", {"root": "active_workspace", "path": "src"})
        tree = registry.execute("list_files", {"root": "active_workspace", "path": ".", "depth": 3})
        found = registry.execute("search_code", {"query": "VALUE", "root": "active_workspace"})
        status = registry.execute("vcs_read", {"action": "status"})
        diff = registry.execute("vcs_read", {"action": "diff"})
        log = registry.execute("vcs_read", {"action": "log", "max_count": 1})
        shown = registry.execute("vcs_read", {"action": "file", "path": "README.md"})
    assert seen == [], seen  # nothing the service did named a path inside the work tree
    assert all("⚠️" not in text for text in calls), calls
    assert (repo / "src" / "new.py").read_text() == "VALUE = 1\n" and "print('b')" in (repo / "src" / "main.py").read_text()
    assert (repo / "added.txt").read_text() == "added\n" and not (repo / "old.txt").exists()
    assert "print('b')" in read and "src/new.py" in flat and "src/main.py" in tree and "src/new.py:1" in found
    assert "new.py" in status and "readme v2" in diff and "base" in log and shown.startswith("readme")
    assert attached(docker, "git") and attached(docker, "mv -f") and not docker.records()


def test_a_work_root_swapped_for_a_link_reaches_nothing_of_the_service(env, monkeypatch):
    """The employee makes the workspace's folder a link to a service folder after the task began. Which domain an
    operation belongs to is read from its spelling, so the write goes to the container, the service's own read
    refuses to follow the link (the read becomes the work domain's), and the service never opens, writes or lists
    its folder through the link. (In the reference deployment the container cannot see that folder at all.)"""
    docker = _managed(env, monkeypatch)
    secret = env.root / "service-data"
    secret.mkdir()
    (secret / "settings.json").write_text("SERVICE SECRET\n", encoding="utf-8")
    repo = _work_repo(env.work / "app", {"a.txt": "work\n"})
    registry = _registry(env, repo)
    (env.work / "app").rename(env.work / "app-moved")
    (env.work / "app").symlink_to(secret, target_is_directory=True)
    monkeypatch.setattr(work_domain, "read_in_work", lambda path, limit=None: b"WORK-DOMAIN VIEW\n")
    with service_effects_on(secret, env.work, allowed_opens=(str(env.work),)) as seen:
        read = registry.execute("read_file", {"root": "active_workspace", "path": "settings.json"})
        listed = registry.execute("list_files", {"root": "active_workspace", "path": "."})
        found = registry.execute("search_code", {"query": "SECRET", "root": "active_workspace"})
        wrote = registry.execute("write_file", {"root": "active_workspace", "path": "planted.txt", "content": "x\n"})
    assert seen == [], seen
    assert "SERVICE SECRET" not in read + listed + found and "WORK-DOMAIN VIEW" in read
    assert "No matches" in found or "SEARCH_" in found
    assert attached(docker, "mv -f"), wrote  # the write was the container's, resolved in its own namespace


def test_a_failing_work_domain_never_falls_back_to_the_service(env, monkeypatch):
    """When the container refuses or the service has no Docker CLI, writes, Git and reads fail typed; nothing is
    done on the service instead, and the work tree is unchanged."""
    _managed(env, monkeypatch)
    repo = _work_repo(env.work / "app", {"a.txt": "one\n"})
    registry = _registry(env, repo)
    monkeypatch.setenv("PATH", str(env.root / "no-cli"))  # the CLI is gone: positive proof nothing can start
    with service_effects_on(env.work, allowed_opens=(str(env.work),)) as seen:
        wrote = registry.execute("write_file", {"root": "active_workspace", "path": "a.txt", "content": "two\n"})
        status = registry.execute("vcs_read", {"action": "status"})
        with pytest.raises(FileNotFoundError) as missing:
            work_domain.run_in_work(["true"], cwd=repo)
    assert seen == [] and missing.value.process_not_started is True
    assert "⚠️" in wrote and "GIT_ERROR" in status and (repo / "a.txt").read_text() == "one\n"


def test_work_domain_stdin_and_stdout_cross_byte_exact(env, monkeypatch):
    """A real pipe consumer in the container (dash, whose asynchronous list's stdin is /dev/null before explicit
    redirections): every byte the service sends arrives and returns unchanged, NUL and invalid UTF-8 included."""
    docker = _managed(env, monkeypatch)
    payload = bytes(range(256)) * 64 + b"\x00tail\xff"
    echoed = work_domain.run_in_work([sys.executable, "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"],
                                     cwd=env.work, stdin=payload, timeout_sec=30)
    assert echoed.returncode == 0 and echoed.stdout == payload and echoed.backend_settled is True
    assert not docker.records()


def test_a_native_run_receives_its_inputs_and_returns_verified_outputs_through_the_work_domain(env, monkeypatch, tmp_path):
    """The material flow of a native run on a work repository: its selected inputs are delivered into the staging
    root as bytes the work domain unpacks; its private copy is made there by the work domain's Git (the
    repository's clean filter running in the container); the engine would read only work paths; the session's
    edits are captured into a service patch and applied to the target as the work domain, kept across a release
    and a continuation's recreate, and removed. The service never touches the tree by its own pathname."""
    from ouroboros import delegate_readonly_inputs, execution_domain, subagent_worktrees
    from ouroboros.tools.patch_staging import git_apply

    staging = env.work / "staging"
    docker = _managed(env, monkeypatch, staging_root=str(staging))
    data = env.root / "data"
    repo = _work_repo(env.work / "app", {"src/main.py": "print('a')\n", "notes.txt": "keep\n"})
    witness, script = env.root / "filter-ran", env.root / "mark-clean.sh"
    script.write_text(f"#!/bin/sh\nprintf '%s\\n' \"${{IN_FAKE_CONTAINER:-service}}\" >> {witness}\ncat\n", encoding="utf-8")
    script.chmod(0o755)
    _git(repo, "config", "filter.mark.clean", str(script))
    (repo / ".gitattributes").write_text("*.txt filter=mark\n", encoding="utf-8")
    (repo / "input.txt").write_text("selected input\n", encoding="utf-8")
    registry = _registry(env, repo)
    with service_effects_on(env.work, allowed_opens=(str(env.work),)) as seen:
        inputs = delegate_readonly_inputs.prepare_readonly_inputs(
            registry._ctx, "inv-1", [{"root": "active_workspace", "path": "input.txt"}])
        handle = subagent_worktrees.provision_execution_snapshot(target_root=repo, task_id="task-w", snapshot_id="snap-w",
                                                                data_dir=data)
    assert seen == [], seen
    delivered = pathlib.Path(inputs.selected[0]["local"])
    held = delivered.parents[3]  # <staging_root>/scratch/obs-*, owned by the task's drive (session_scratch_owners)
    assert held.parent == staging / "scratch" and held.name.startswith("obs-") and delivered.read_text() == "selected input\n"
    assert (held / "inputs.json").is_file()
    copy = pathlib.Path(handle.path)
    assert copy.is_relative_to(staging / "copies") and (copy / "src" / "main.py").read_text() == "print('a')\n"
    assert set(witness.read_text().split()) == {"1"}  # the repository's clean filter ran only in the container
    assert execution_domain.request_paths_refusal({"scope": {"root": str(copy)},
                                                   "execution": {"workspaceRoot": str(copy),
                                                                 "auxiliaryWritableRoots": [str(delivered.parent)]}}) == ""
    (copy / "src" / "main.py").write_text("print('session')\n", encoding="utf-8")  # the native session's own edits
    (copy / "src" / "extra.py").write_text("EXTRA = 2\n", encoding="utf-8")
    patch = tmp_path / "service-artifacts" / "result.patch"
    with service_effects_on(env.work, allowed_opens=(str(env.work),)) as seen:
        captured = subagent_worktrees.capture_result_commit("snap-w", patch, data_dir=data)
        applied = git_apply(repo, patch, [])
        released = subagent_worktrees.release_snapshot_checkout("snap-w", data_dir=data)
        recreated = subagent_worktrees.recreate_snapshot_checkout("snap-w", data_dir=data)
    restored = (copy / "src" / "extra.py").read_text()  # this test's own look, outside the audited window
    with service_effects_on(env.work, allowed_opens=(str(env.work),)) as removing:
        removed = subagent_worktrees.remove_execution_snapshot("snap-w", data_dir=data)
    assert seen == [] and removing == [], (seen, removing)
    body = patch.read_bytes()
    assert captured["patch_size"] == len(body) and b"print('session')" in body and b"EXTRA = 2" in body
    assert applied.returncode == 0, applied.stderr
    assert (repo / "src" / "main.py").read_text() == "print('session')\n" and (repo / "src" / "extra.py").is_file()
    assert released is True and recreated["snapshot_id"] == "snap-w" and restored == "EXTRA = 2\n"
    assert removed is True and not copy.exists() and subagent_worktrees.find_execution_snapshot("snap-w", data) is None
    assert attached(docker, "worktree add") and attached(docker, "diff-tree") and attached(docker, "apply")
    assert not docker.records()  # every launch's group was confirmed ended and its record retired


def test_work_outputs_return_as_checked_bytes_and_a_planted_link_returns_nothing_of_the_service(env, monkeypatch):
    """A run's file outputs come back as bytes the service checks: a work file through the confined intake into the
    task's artifacts, a work folder as the container's tar stream (each member a requested regular file, hashed as
    received). A member the employee made a link to a service file is refused, never followed or packaged."""
    import hashlib
    import zipfile

    from ouroboros.artifact_directories import copy_directory_to_task_artifacts
    from ouroboros.artifacts import copy_file_to_task_artifacts

    docker = _managed(env, monkeypatch, staging_root=str(env.work / "staging"))
    secret = env.root / "service-secret.txt"
    secret.write_text("SERVICE SECRET\n", encoding="utf-8")
    out = env.work / "app" / "out"
    (out / "sub").mkdir(parents=True)
    payload = bytes(range(256)) * 4096  # 1 MiB: streamed in chunks, never held whole
    (out / "model.bin").write_bytes(payload)
    (out / "sub" / "notes.txt").write_text("notes\n", encoding="utf-8")
    (out / "leak.txt").symlink_to(secret)
    registry = _registry(env, env.work / "app")
    with service_effects_on(env.work, allowed_opens=(str(env.work),)) as seen:
        record = copy_file_to_task_artifacts(registry._ctx, out / "model.bin")
        received = work_domain.export_tree(out, ["model.bin", "sub/notes.txt"], env.root / "intake")
        with pytest.raises(work_domain.WorkDomainRefused):
            work_domain.export_tree(out, ["leak.txt"], env.root / "refused")
        folder = copy_directory_to_task_artifacts(registry._ctx, out)
    assert seen == [], seen
    assert pathlib.Path(record["path"]).read_bytes() == payload and record["source_path"] == str(out / "model.bin")
    assert {row["path"]: row["sha256"] for row in received} == {
        "model.bin": hashlib.sha256(payload).hexdigest(), "sub/notes.txt": hashlib.sha256(b"notes\n").hexdigest()}
    assert not (env.root / "refused" / "leak.txt").exists()
    with zipfile.ZipFile(folder[1]["path"]) as packaged:
        assert sorted(packaged.namelist()) == ["model.bin", "sub/notes.txt"]
        assert packaged.read("model.bin") == payload and b"SERVICE SECRET" not in packaged.read("sub/notes.txt")
    (out / "model.bin").write_bytes(b"retrained\n")  # the run's next output under the same name (outside the window)
    with service_effects_on(env.work, allowed_opens=(str(env.work),)) as again:
        second = copy_file_to_task_artifacts(registry._ctx, out / "model.bin")
    assert again == [] and second["path"] == record["path"] and second["source_path"] == str(out / "model.bin")
    assert pathlib.Path(second["path"]).read_bytes() == b"retrained\n"  # one artifact, versioned, not a new copy
    assert attached(docker, "tar -c") and not docker.records()
