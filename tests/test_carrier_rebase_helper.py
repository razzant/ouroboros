"""Unit test for scripts/carrier_rebase_helper.py — the tactical-rebase side
of the carrier engine (spec §1.9-10): span-substitution 'ours' for the
declared version carriers, ordinary 3-way for everything else, untouched
non-carrier conflicts, and honest exit codes.

The helper reads only the unmerged index stages, so a real `git merge`
conflict stands in for the rebase stop: during a rebase, stage 2 ('ours') is
the side being rebased ONTO, which is exactly the side the default preference
keeps inside the spans.
"""

import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts" / "carrier_rebase_helper.py"


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def _conflicted_repo(tmp_path, *, break_ours_anchor=False):
    """ours = upstream at 7.0.1 (+ a code edit), theirs = replayed work at 7.1.0."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "VERSION").write_text("7.0.0\n", newline="\n")
    (repo / "a.txt").write_text("base\n", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base 7.0.0")
    _git(repo, "checkout", "-q", "-b", "replayed")
    (repo / "VERSION").write_text("7.1.0\n", newline="\n")
    (repo / "a.txt").write_text("replayed code\n", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "replayed 7.1.0")
    _git(repo, "checkout", "-q", "-")
    (repo / "VERSION").write_text(
        "broken anchor\n" if break_ours_anchor else "7.0.1\n", newline="\n"
    )
    (repo / "a.txt").write_text("upstream code\n", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "upstream 7.0.1")
    merge = _git(repo, "merge", "--no-commit", "--no-ff", "replayed")
    assert merge.returncode == 1, merge.stderr  # both files conflict
    return repo


def _run_helper(repo, *extra):
    return subprocess.run(
        [sys.executable, str(HELPER), "--worktree", str(repo), *extra],
        capture_output=True, text=True,
    )


def test_helper_keeps_ours_inside_the_span_and_leaves_the_rest(tmp_path):
    repo = _conflicted_repo(tmp_path)

    result = _run_helper(repo)

    assert result.returncode == 0, result.stderr or result.stdout
    assert "VERSION" in result.stdout
    unmerged = _git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split()
    assert "VERSION" not in unmerged  # resolved and staged
    assert "a.txt" in unmerged  # non-carrier conflict untouched
    assert (repo / "VERSION").read_text() == "7.0.1\n"  # 'ours' won the span
    assert "not carrier files" in result.stdout and "a.txt" in result.stdout


def test_helper_prefer_theirs_flips_the_span_side(tmp_path):
    repo = _conflicted_repo(tmp_path)

    result = _run_helper(repo, "--prefer", "theirs")

    assert result.returncode == 0, result.stderr or result.stdout
    assert (repo / "VERSION").read_text() == "7.1.0\n"


def test_helper_degrades_a_broken_anchor_and_reports_failure(tmp_path):
    repo = _conflicted_repo(tmp_path, break_ours_anchor=True)

    result = _run_helper(repo)

    assert result.returncode == 1, result.stderr or result.stdout
    assert "left for manual resolution: VERSION" in result.stdout
    unmerged = _git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split()
    assert "VERSION" in unmerged  # untouched, exactly as git left it
    body = (repo / "VERSION").read_text()
    assert "<<<<<<<" in body and ">>>>>>>" in body


def test_the_helper_resolves_standalone_and_loads_none_of_the_servers_owners(tmp_path):
    """The operator helper keeps the carrier engine's own runner: resolving a real conflict loads neither the
    server's Git owner nor its shell admission nor the tool registry (``tracked_run_git`` is imported only by the
    server's callers, when they call it)."""
    import json

    repo = _conflicted_repo(tmp_path)
    probe = ("import json, runpy, sys\n"
             f"sys.argv = [{str(HELPER)!r}, '--worktree', {str(repo)!r}]\n"
             "code = None\n"
             "try:\n    runpy.run_path(sys.argv[0], run_name='__main__')\n"
             "except SystemExit as exc:\n    code = exc.code\n"
             "print(json.dumps({'code': code, 'loaded': sorted(name for name in sys.modules if name in ("
             "'supervisor.git_ops', 'ouroboros.tools.shell_process', 'ouroboros.tools.registry'))}))\n")

    # Isolated, and writing no bytecode beside the sources it imports.
    result = subprocess.run([sys.executable, "-I", "-B", "-c", probe], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"code": 0, "loaded": []}
    assert (repo / "VERSION").read_text() == "7.0.1\n"


def test_helper_is_quiet_on_a_clean_tree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "VERSION").write_text("7.0.0\n", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")

    result = _run_helper(repo)

    assert result.returncode == 0, result.stderr or result.stdout
    assert "nothing to do" in result.stdout
