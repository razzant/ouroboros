"""Real Git: observations preserve the index; writers still acquire its lock."""

import json
import os
import subprocess

import pytest

from supervisor import git_ops
from tests.test_git_lock_owners import _git, _repo


def _index(repo):
    path = repo / ".git" / "index"
    stat = path.stat()
    return path.read_bytes(), stat.st_ino, stat.st_mtime_ns


@pytest.mark.parametrize("changed", [False, True], ids=["stat-dirty-same-content", "real-content"])
@pytest.mark.parametrize("reader", ["capture", "bounded", "shell", "attribution", "review", "tool",
                                    "preflight", "checkpoint", "patch", "streamed_patch", "subject", "pr",
                                    "vcs_names", "selection"])
def test_host_observations_never_write_the_index(tmp_path, monkeypatch, changed, reader):
    from ouroboros import mutation_attribution, utils
    from ouroboros.tools import review_file_pack, shell_effects, vcs_read

    monkeypatch.setattr(vcs_read, "_CONFIG_LOCATIONS", vcs_read._CONFIG_LOCATIONS | {"GIT_TRACE2_EVENT"})
    repo = _repo(tmp_path)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "1")  # defeat the test launcher's protective default
    trace = tmp_path / "git-trace.jsonl"
    _git(repo, "config", "trace2.eventTarget", str(trace))
    monkeypatch.setenv("GIT_TRACE2_EVENT", str(trace))
    seed = repo / "seed.txt"
    if changed:
        seed.write_text("different content\n", encoding="utf-8")
    stat = seed.stat()
    os.utime(seed, ns=(stat.st_atime_ns, stat.st_mtime_ns - 10_000_000_000))
    before = _index(repo)
    if reader in {"capture", "bounded"}:
        results = [git_ops.git_capture(["git", *args], timeout=10 if reader == "bounded" else None)
                   for args in (("status", "--porcelain"), ("diff", "--stat"))]
        assert all(row[0] == 0 for row in results)
        output = "".join(row[1] for row in results)
    elif reader == "shell":
        output = "".join(shell_effects._get_changed_files(repo)) + shell_effects._get_diff_stat(repo)
    elif reader == "attribution":
        output = mutation_attribution._run_git(repo, "status", "--porcelain")
    elif reader == "review":
        output = "".join(review_file_pack.list_changed_paths_from_git_status(repo))
    elif reader == "tool":
        output = vcs_read._capture_read_git(repo, ["diff", "--stat"]).stdout.decode("utf-8")
    elif reader == "preflight":
        from ouroboros.workspace_preflight import _git_stdout
        output = _git_stdout(repo, ["git", "status", "--porcelain"], timeout=10)
    elif reader == "checkpoint":
        from ouroboros.coop_checkpoint import _run_git
        output = _run_git(["git", "status", "--porcelain"], repo).stdout
    elif reader in {"patch", "streamed_patch"}:
        from ouroboros.workspace_patch_capture import _git_bytes, _append_git_output
        cmd = ["git", "diff", "--stat"]
        if reader == "patch":
            output = _git_bytes(cmd, repo).decode("utf-8")
        else:
            captured = tmp_path / "captured.diff"
            errors = []
            with captured.open("wb") as stream:
                _append_git_output(cmd, repo, stream, allow_rc={0}, errors=errors, diagnostics=[])
            assert errors == []
            output = captured.read_text(encoding="utf-8")
    elif reader == "subject":
        from ouroboros.tools.review_subject import _git_bytes
        output = _git_bytes(repo, ["diff", "--stat"])[1].decode("utf-8")
    elif reader == "pr":
        from ouroboros.tools.git_pr import _g
        output = _g(["status", "--porcelain"], repo).stdout
    elif reader == "vcs_names":
        from ouroboros.tools.registry import ToolContext
        output = vcs_read._vcs_read(ToolContext(repo_dir=repo, drive_root=tmp_path / "data"),
                                    action="diff", root="system_repo", name_only=True)
        assert "VCS target:" in output and "ERROR" not in output, output
    else:
        assert reader == "selection"
        from ouroboros.test_selection import changed_paths
        output = "".join(changed_paths(repo, "HEAD")[1])
    assert ("seed.txt" in output) is changed
    assert _index(repo) == before
    assert not (repo / ".git" / "index.lock").exists()
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert not [row for row in rows if row.get("label") == "do_write_index"]
    # The ordinary tool command seam shares the same convention.
    assert bool(utils.run_cmd(["git", "diff", "--stat"], cwd=repo)) is changed
    assert _index(repo) == before


@pytest.mark.parametrize("changed", [False, True], ids=["stat-only", "content"])
@pytest.mark.parametrize("operation", ["cherry_pick", "stage_merge"])
@pytest.mark.parametrize("binary", [False, True], ids=["text", "binary"])
@pytest.mark.parametrize("tight_budget", [False, True], ids=["one-batch", "chunked"])
def test_pr_integration_checks_content_before_writing(tmp_path, monkeypatch, changed, operation, binary, tight_budget):
    from ouroboros import argv_budget
    from ouroboros.git_observation import host_git_env
    from ouroboros.tools import git_pr
    from ouroboros.tools.registry import ToolContext

    repo = _repo(tmp_path)
    if binary:
        (repo / "seed.txt").write_bytes(b"\0original\n")
    # Put unconfirmed rows before a text edit, after a binary edit.
    others = [repo / f"{'z' if binary else 'a'}-other-{i:03}.dat" for i in range(12)]
    for path in others:
        path.write_bytes(b"\0unchanged")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "binary fixtures")
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "data", branch_dev="ouroboros")
    _git(repo, "checkout", "-qb", "integrate/pr-1")
    (repo / "feature.txt").write_text("incoming work\n", encoding="utf-8")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-qm", "feature")
    feature = _git(repo, "rev-parse", "HEAD")
    if operation == "cherry_pick":
        _git(repo, "reset", "--hard", "HEAD~1")
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "1")
    if tight_budget:
        limit = (argv_budget.argv_env_bytes([], host_git_env())
                 + argv_budget.DEFAULT_HEADROOM_BYTES + 300)
        monkeypatch.setattr(argv_budget, "system_argv_limit_bytes", lambda: limit)

    def dirty_stat():
        seed = repo / "seed.txt"
        if changed:
            seed.write_bytes(b"\0owner work\n" if binary else b"owner work\n")
        for path in [seed, *others]:
            info = path.stat()
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns - 10_000_000_000))

    observed = []
    original = git_pr._g

    def capture(args, cwd, **kwargs):
        before = _index(repo)
        result = original(args, cwd, **kwargs)
        if args[0] in {"status", "diff"}:
            observed.append(args)
            assert _index(repo) == before, args
            assert not (repo / ".git" / "index.lock").exists()
        if operation == "stage_merge" and args == ["checkout", "ouroboros"]:
            # Checkout may refresh stat data; exercise the target-tree reader itself.
            assert result.returncode == 0
            dirty_stat()
        return result

    monkeypatch.setattr(git_pr, "_g", capture)
    if operation == "cherry_pick":
        dirty_stat()
        result = git_pr._cherry_pick_pr_commits(ctx, [feature])
    else:
        # The initial integration-branch status reader also sees the stat-only case.
        if not changed:
            dirty_stat()
        result = git_pr._stage_pr_merge(ctx, "integrate/pr-1")
    assert observed
    probes = [args for args in observed if "--patch" in args]
    if operation == "stage_merge":
        if changed and not binary:
            assert probes == [], "a text row already proves the tree changed"
        elif changed or not tight_budget:
            assert len(probes) == 1, "stop after the first confirmed binary batch"
        else:
            assert len(probes) > 1, "an unchanged tree must examine every argv chunk"
    if changed:
        assert "Working tree" in result and "changes" in result, result
        assert (repo / "seed.txt").read_bytes() == (b"\0owner work\n" if binary else b"owner work\n")
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "integrate/pr-1"
    else:
        assert result.startswith("✅"), result
        assert (repo / "feature.txt").read_text(encoding="utf-8") == "incoming work\n"
        if operation == "stage_merge":
            assert _git(repo, "rev-parse", "MERGE_HEAD") == feature
        else:
            assert _git(repo, "log", "-1", "--format=%s") == "feature"


@pytest.mark.parametrize("comparison", ["worktree", "staged", "tree"])
@pytest.mark.parametrize("binary_metadata", [False, True], ids=["text-metadata", "binary-metadata"])
@pytest.mark.parametrize("rename", [
    "renamed file.txt",  # binary, mode and rename changes on every host
    # The quoting case: a tab and a double quote reach the output quoted, but Windows refuses such a name.
    pytest.param('renamed\t"file.txt', marks=pytest.mark.skipif(os.name == "nt", reason="a POSIX-only file name")),
], ids=["portable-name", "quoted-name"])
def test_vcs_names_keep_binary_mode_and_rename_changes(tmp_path, monkeypatch, comparison, binary_metadata, rename):
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.vcs_read import _vcs_read

    repo = _repo(tmp_path)
    binary = repo / "binary.dat"
    binary.write_bytes(b"\0before")
    (repo / "mode.sh").write_text("exit 0\n", encoding="utf-8")
    if binary_metadata:
        (repo / "mode.sh").write_bytes(b"\0mode only")
        (repo / "seed.txt").write_bytes(b"\0rename only")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixtures")
    binary.write_bytes(b"\0after")
    _git(repo, "mv", "seed.txt", rename)
    _git(repo, "update-index", "--chmod=+x", "mode.sh")
    _git(repo, "add", "binary.dat")
    # Preserve filename quoting and rename destinations, including zero-line changes.
    if comparison == "worktree":
        options, refs = {"base": "HEAD"}, ["HEAD"]
    elif comparison == "staged":
        options, refs = {"staged": True}, ["--cached"]
    else:
        _git(repo, "commit", "-qm", "changes")
        options, refs = {"base": "HEAD~1", "head": "HEAD"}, ["HEAD~1", "HEAD"]
    expected = _git(repo, "diff", "--name-only", *refs)
    assert "binary.dat" in expected and "renamed" in expected
    if comparison != "worktree":
        assert "mode.sh" in expected
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "1")
    before = _index(repo)
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "data")
    output = _vcs_read(ctx, action="diff", root="system_repo", name_only=True, **options)
    assert output.split("\n\nVCS target:", 1)[0] == expected
    assert _index(repo) == before


@pytest.mark.parametrize("command", [("reset", "--hard", "HEAD"), ("checkout", "-B", "other"),
                                   ("commit", "--allow-empty", "-qm", "writer"),
                                   ("merge", "--no-edit", "other")])
def test_writer_locks_remain_mandatory(tmp_path, monkeypatch, command):
    repo = _repo(tmp_path)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    if command[0] == "merge":
        _git(repo, "checkout", "-b", "other")
        (repo / "other.txt").write_text("merge me\n", encoding="utf-8")
        _git(repo, "add", "other.txt")
        _git(repo, "commit", "-qm", "other")
        _git(repo, "checkout", "ouroboros")
    lock = repo / ".git" / "index.lock"
    lock.write_bytes(b"external writer")
    inode = lock.stat().st_ino
    rc, _, error = git_ops.git_capture(["git", *command])
    assert rc != 0 and "index.lock" in error
    assert (lock.read_bytes(), lock.stat().st_ino) == (b"external writer", inode)
    lock.unlink()  # fixture owner releases it
    assert git_ops.git_capture(["git", *command])[0] == 0


def test_observation_does_not_repair_corruption_but_a_writer_can(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    index = repo / ".git" / "index"
    index.write_bytes(b"broken")
    before = _index(repo)
    assert git_ops.git_capture(["git", "status", "--porcelain"])[0] != 0
    assert _index(repo) == before and not list(index.parent.glob("index.corrupt.*"))
    assert git_ops.git_capture(["git", "reset", "--mixed", "HEAD"], repair_index=True)[0] == 0
    assert _git(repo, "ls-files") == "seed.txt"


def test_read_policy_beats_inherited_config_without_losing_it(tmp_path, monkeypatch):
    from ouroboros.git_observation import host_git_env

    assert git_ops.host_git_env is host_git_env
    repo = _repo(tmp_path)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.autocrlf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "false")
    env = git_ops.host_git_env()
    for key in ("core.autocrlf", "diff.autoRefreshIndex"):
        result = subprocess.run(["git", "config", "--get", key], cwd=repo, env=env,
                                capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "false"


@pytest.mark.parametrize("count,valid", [
    ("not-a-number", False), ("-1", False), ("1_0", False), ("１", False), ("1 ", False),
    ("", True), ("0", True), ("1", True), ("+1", True), (" 1", True), ("-0", True),
])
def test_inherited_config_count_keeps_gits_own_result(tmp_path, monkeypatch, count, valid):
    """Bad configuration returns Git's diagnostic, while valid counts still work."""
    repo = _repo(tmp_path)
    _git(repo, "config", "core.autocrlf", "false")
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setenv("GIT_CONFIG_COUNT", count)
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.autocrlf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "false")
    command = ["git", "config", "--get", "core.autocrlf"]
    expected = subprocess.run(command, cwd=repo, capture_output=True, text=True,
                              env={**os.environ, "LC_ALL": "C", "LANG": "C"})
    assert (expected.returncode == 0) is valid
    assert git_ops.git_capture(command) == (
        expected.returncode, expected.stdout.strip(), expected.stderr.strip())


@pytest.mark.parametrize("changed", [False, True], ids=["stat-only", "content"])
@pytest.mark.parametrize("file_reference", [False, True], ids=["patch", "file-reference"])
@pytest.mark.parametrize("binary", [False, True], ids=["text", "binary"])
@pytest.mark.parametrize("count", [1, 30])
def test_workspace_capture_lists_only_real_tracked_changes(tmp_path, monkeypatch, changed, file_reference, binary, count):
    from ouroboros import workspace_patch_capture as capture

    repo = _repo(tmp_path)
    others = [repo / f"unchanged-{i:03}.dat" for i in range(count - 1)]
    if binary:
        (repo / "seed.txt").write_bytes(b"\0original payload\n")
    for path in others:
        path.write_bytes(b"\0original payload\n" if binary else b"original payload\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "--allow-empty", "-qm", "fixtures")
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "1")
    trace = tmp_path / "git-trace.jsonl"
    monkeypatch.setenv("GIT_TRACE2_EVENT", str(trace))
    if file_reference:
        monkeypatch.setattr(capture, "_PATCH_FILE_REFERENCE_BYTES", 8)
    seed = repo / "seed.txt"
    if changed:
        seed.write_bytes(b"\0changed payload!\n" if binary else b"changed payload\n")
    for path in [seed, *others]:
        info = path.stat()
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns - 10_000_000_000))
    before = _index(repo)
    artifacts, manifest = capture.write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task={})
    assert manifest["errors"] == []
    assert manifest["status"] == ("ready_with_changes" if changed else "ready_no_changes")
    assert manifest["tracked_changed"] == (["seed.txt"] if changed else [])
    assert manifest["counts"]["tracked_changed"] == int(changed)
    assert bool(manifest["file_output_changes"]) is (changed and file_reference)
    assert any(row["kind"] == "workspace_file_outputs" for row in artifacts) is (changed and file_reference)
    assert any(row["kind"] == "workspace_patch" for row in artifacts) is (changed and not file_reference)
    assert _index(repo) == before
    assert not (repo / ".git" / "index.lock").exists()
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    probes = [row["argv"] for row in rows if row.get("event") == "start" and "--patch" in row["argv"]]
    assert len(probes) == int(binary), probes
    assert not [row for row in rows if row.get("label") == "do_write_index"]


@pytest.mark.parametrize("changed", [False, True], ids=["stat-only", "content"])
@pytest.mark.parametrize("reader", ["vcs_names", "selection"])
@pytest.mark.parametrize("name", ["binary.dat", "binary[1].dat"])
@pytest.mark.parametrize("count", [1, 30, 100])
def test_binary_path_observers_compare_content(tmp_path, monkeypatch, changed, reader, name, count):
    """Binary numstat rows alone prove neither edits nor an unchanged tree."""
    from ouroboros.git_observation import host_git_env

    repo = _repo(tmp_path)
    seed = repo / name
    seed.write_bytes(b"\0before")
    others = [repo / f"unchanged-{i:03}.dat" for i in range(count - 1)]
    for path in others:
        path.write_bytes(b"\0before")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "binary seed")
    trace = tmp_path / "git-trace.jsonl"
    monkeypatch.setenv("GIT_TRACE2_EVENT", str(trace))
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "1")
    if changed:
        seed.write_bytes(b"\0after!")  # same size, different content
    for path in [seed, *others]:
        info = path.stat()
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns - 10_000_000_000))
    before = _index(repo)
    if reader == "selection":
        from ouroboros.test_selection import changed_paths
        paths = list(changed_paths(repo, "HEAD")[1])
    else:
        from ouroboros.tools.registry import ToolContext
        from ouroboros.tools.vcs_read import _vcs_read
        ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "data")
        output = _vcs_read(ctx, action="diff", root="system_repo", name_only=True)
        assert "VCS target:" in output and "ERROR" not in output, output
        names = output.split("VCS target:", 1)[0].rstrip("\n")
        paths = [json.loads(path) if path.startswith('"') else path for path in names.splitlines()]
    assert paths == ([name] if changed else [])
    assert _index(repo) == before
    assert not (repo / ".git" / "index.lock").exists()
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert not [row for row in rows if row.get("label") == "do_write_index"]
    starts = [row["argv"] for row in rows if row.get("event") == "start"]
    probes = [args for args in starts if "--patch" in args]
    assert len(probes) == 1, probes
    assert len(starts) == {"selection": 4, "vcs_names": 2}[reader], starts
    # The content oracle also reads without refreshing the index.
    result = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=repo, env=host_git_env())
    assert result.returncode == int(changed)
    assert _index(repo) == before


@pytest.mark.parametrize("changed", [False, True], ids=["stat-only", "content"])
@pytest.mark.parametrize("quoted", [False, True])
@pytest.mark.parametrize("text_reader", [False, True], ids=["bytes", "text"])
def test_binary_batches_keep_literal_and_git_quoted_paths(tmp_path, monkeypatch, changed, quoted, text_reader):
    from ouroboros import argv_budget
    from ouroboros.git_observation import host_git_env, numstat_paths

    repo = _repo(tmp_path)
    names = ["space file.dat", "binary[1].dat", "binary1.dat", "é猫.dat"]
    if os.name != "nt":
        names += ['quote".dat', "tab\t.dat", "line\n.dat", "slash\\.dat", ":(glob)*.dat"]
    for name in names:
        (repo / name).write_bytes(b"\0before")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "literal fixtures")
    _git(repo, "config", "core.quotePath", str(quoted).lower())
    _git(repo, "config", "diff.mnemonicPrefix", "true")
    _git(repo, "config", "color.ui", "always")
    for i, name in enumerate(names):
        path = repo / name
        if changed and i % 2 == 0:
            path.write_bytes(b"\0after!")
        info = path.stat()
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns - 10_000_000_000))
    env = host_git_env()
    limit = argv_budget.argv_env_bytes([], env) + argv_budget.DEFAULT_HEADROOM_BYTES + 300
    monkeypatch.setattr(argv_budget, "system_argv_limit_bytes", lambda: limit)
    calls = []

    def read(args):
        calls.append(args)
        output = subprocess.run(args, cwd=repo, env=env, capture_output=True, check=True).stdout
        return output.decode("utf-8", "surrogateescape") if text_reader else output

    command = ["git", "diff", "--numstat", "-z", "HEAD", "--"]
    before = _index(repo)
    paths = numstat_paths(read(command), command, read)
    assert set(paths) == (set(names[::2]) if changed else set())
    assert len(calls) > 2, "exercise multiple transport chunks"
    for args in calls[1:]:
        assert argv_budget.argv_budget_excess(args, env=env) == ""
        assert all(spec.startswith(":(literal)") for spec in args[args.index("--") + 1:])
    assert _index(repo) == before
    assert not (repo / ".git" / "index.lock").exists()
