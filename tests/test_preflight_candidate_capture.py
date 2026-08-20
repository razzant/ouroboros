"""The candidate tree the gate assembles out of the live working state.

Split verbatim out of ``tests/test_preflight_runner.py`` by theme. This module owns the
conflicted and mixed-unmerged index, the staged change reverted in the worktree, the
disposable index that matches the source, the CRLF, binary, chmod and non-UTF-8 content
that has to survive byte for byte, and the untracked names decoded with the filesystem
codec.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import textwrap

import pytest


from tests._preflight_runner_shared import (
    _commit_all,
    _git,
    _make_repo,
)
from tests._preflight_runner_shared import stub_passes as _stub_passes
from tests._preflight_runner_shared import two_pass_env as _two_pass_env

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
stub_passes = _stub_passes
two_pass_env = _two_pass_env


def _start_conflicted_merge(
    repo: pathlib.Path, incoming: dict[str, str], ours: dict[str, str]
) -> None:
    """Drive `repo` into an in-progress merge whose index holds unmerged entries.

    Two real branches, a real `git merge` that stops on the conflict — no mocked
    git anywhere, because the subject under test is git's own rendering of an
    unmerged index. Asserts the fixture really produced unmerged entries so a
    test can never silently pin the ordinary merged path instead.
    """
    _git(repo, "checkout", "-b", "incoming")
    for rel, body in incoming.items():
        (repo / rel).write_text(textwrap.dedent(body), encoding="utf-8")
    _commit_all(repo)
    _git(repo, "checkout", "ouroboros")
    for rel, body in ours.items():
        (repo / rel).write_text(textwrap.dedent(body), encoding="utf-8")
    _commit_all(repo)
    merge = subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "merge", "incoming"],
        cwd=str(repo), capture_output=True, text=True,
    )
    assert merge.returncode != 0, (
        f"fixture precondition: the merge must conflict, got rc=0:\n{merge.stdout}{merge.stderr}"
    )
    unmerged = subprocess.run(
        ["git", "ls-files", "-u"], cwd=str(repo),
        capture_output=True, text=True, check=True,
    )
    assert unmerged.stdout.strip(), "fixture precondition: no unmerged index entries"

def _spy_on_candidate(monkeypatch, rel_paths):
    """Replace the pytest spawn with a spy that records candidate-file contents.

    Returns the dict the spy fills: relative path -> file text, or None when the
    path is absent from the candidate worktree. Complements ``stub_passes``
    (whose fixture setup still neutralises the plugin/worker seams): the
    recorder it installs is replaced, because these tests need the WORKTREE
    argument — the one thing the recorder drops.
    """
    from ouroboros import preflight_runner

    seen: dict[str, object] = {}

    def _spy(agent_python, worktree, temp_root, args, timeout):
        wt = pathlib.Path(worktree)
        for rel in rel_paths:
            target = wt / rel
            seen[rel] = target.read_text(encoding="utf-8") if target.is_file() else None
        return (0, "", "")

    monkeypatch.setattr(preflight_runner, "_execute_pytest_pass", _spy)
    return seen

def test_a_purely_conflicted_merge_runs_against_the_worktree_resolution(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """A merge whose ONLY change is the conflicted file used to kill the gate
    outright: the staged diff is nothing but the "* Unmerged path" stub and the
    unstaged diff nothing but the `--cc` hunk, so `git apply` returned rc=128
    ("No valid patches in input") and the whole preflight died as "hermetic
    preflight failed" before running a single test. The one-diff capture has no
    such rendering — the conflicted path arrives as plain worktree content — so
    the gate runs, and runs against the RESOLUTION the resolver typed, not
    against HEAD."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "conflict.txt": "base\n",
    })
    _start_conflicted_merge(
        repo, incoming={"conflict.txt": "incoming\n"}, ours={"conflict.txt": "ours\n"}
    )
    (repo / "conflict.txt").write_text("resolved\n", encoding="utf-8")  # no `git add`

    stub_passes([])  # seam neutralisation only; the spy below replaces the recorder
    seen = _spy_on_candidate(monkeypatch, ["conflict.txt"])

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["conflict.txt"] == "resolved\n", (
        f"candidate does not carry the worktree resolution: {seen!r}"
    )

def test_a_mixed_unmerged_index_drops_neither_staged_nor_conflicted_changes(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """The SILENT failure mode, worse than the rc=128 one: with an ordinary hunk
    in each diff stream (an auto-merged staged file, an unstaged edit) alongside
    the conflict, `git apply` exits 0 and just DROPS the `--cc` hunk. The
    candidate then carried the ordinary changes but NOT the resolution — a
    chimera tree nobody has, whose green or red verdict is equally meaningless.
    On the unfixed base this test fails on the conflict-file assertion."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "conflict.txt": "base\n",
        "auto.txt": "base auto\n",
        "notes.txt": "base notes\n",
    })
    _start_conflicted_merge(
        repo,
        incoming={"conflict.txt": "incoming\n", "auto.txt": "incoming auto\n"},
        ours={"conflict.txt": "ours\n"},
    )
    # Resolve the conflict and touch an unrelated tracked file — both WITHOUT
    # `git add`, exactly how a resolver's tree looks mid-work.
    (repo / "conflict.txt").write_text("resolved\n", encoding="utf-8")
    (repo / "notes.txt").write_text("edited notes\n", encoding="utf-8")

    stub_passes([])
    seen = _spy_on_candidate(monkeypatch, ["conflict.txt", "auto.txt", "notes.txt"])

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["auto.txt"] == "incoming auto\n", "staged auto-merged change lost"
    assert seen["notes.txt"] == "edited notes\n", "unstaged ordinary change lost"
    assert seen["conflict.txt"] == "resolved\n", (
        "the conflicted file's resolution was silently dropped — the candidate "
        f"is a chimera: {seen!r}"
    )

def test_an_unmerged_resolution_by_deletion_is_absent_from_the_candidate(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """Deleting the conflicted file (plain `rm`, no `git rm`) is a legitimate
    resolution. `git diff --binary HEAD` renders it as an ordinary deletion
    hunk, so the candidate must NOT carry the file — a candidate that resurrects
    it from HEAD would test a tree the resolver explicitly deleted from."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "conflict.txt": "base\n",
    })
    _start_conflicted_merge(
        repo, incoming={"conflict.txt": "incoming\n"}, ours={"conflict.txt": "ours\n"}
    )
    (repo / "conflict.txt").unlink()  # resolution by deletion, no `git rm`

    stub_passes([])
    seen = _spy_on_candidate(monkeypatch, ["conflict.txt"])

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["conflict.txt"] is None, (
        f"a file deleted as the conflict resolution reappeared in the candidate: {seen!r}"
    )

def test_a_staged_delete_with_a_recreated_untracked_file_mirrors_the_live_worktree(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """Classification edge: `git rm` stages a deletion, and a NEW same-named file
    written afterwards is untracked (`ls-files --others` lists it). The one-diff
    capture deletes the path from the candidate and the untracked copy then
    restores the reborn content — net effect, the candidate equals the live
    worktree, which is the whole equivalence the one-diff capture promises."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "conflict.txt": "base\n",
        "victim.txt": "victim base\n",
    })
    _start_conflicted_merge(
        repo, incoming={"conflict.txt": "incoming\n"}, ours={"conflict.txt": "ours\n"}
    )
    (repo / "conflict.txt").write_text("resolved\n", encoding="utf-8")
    _git(repo, "rm", "-q", "victim.txt")
    (repo / "victim.txt").write_text("reborn\n", encoding="utf-8")  # untracked now

    stub_passes([])
    seen = _spy_on_candidate(monkeypatch, ["conflict.txt", "victim.txt"])

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["victim.txt"] == "reborn\n", (
        f"candidate diverged from the live worktree on the recreated path: {seen!r}"
    )
    assert seen["conflict.txt"] == "resolved\n"

def test_a_failed_capture_is_a_named_hard_block_not_a_test_failure(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """When the one-diff capture itself fails, the verdict must say so in the
    gate's named-hard-block vocabulary — PREFLIGHT_CANDIDATE_ASSEMBLY, with a
    remediation that owns the failure itself and does NOT blame the merge in
    progress (an unmerged index is a supported source state for this capture,
    per the function's own docstring) — and no pass may run, because there is
    no candidate worth running it against. A bare "hermetic preflight failed"
    here reads as an infrastructure flake and invites a retry that cannot
    succeed. The interception below pins the EXACT capture argv, the whole
    config-pinning tail included (`--no-ext-diff --no-textconv --no-color
    --src-prefix=a/ --dst-prefix=b/`): dropping any of those flags re-opens
    the door to an operator git config — a diff driver, textconv filter,
    colour escapes, or diff.noprefix/srcPrefix — that reshapes the payload
    into something `git apply` cannot re-apply."""
    from ouroboros import preflight_runner
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "conflict.txt": "base\n",
    })
    _start_conflicted_merge(
        repo, incoming={"conflict.txt": "incoming\n"}, ours={"conflict.txt": "ours\n"}
    )

    events = stub_passes([])
    real_run_git = preflight_runner._run_git
    capture_argv = [
        "diff", "--binary", "--no-ext-diff", "--no-textconv", "--no-color",
        "--src-prefix=a/", "--dst-prefix=b/", "HEAD",
    ]

    def _broken_capture(repo_dir, args, **kwargs):
        if list(args) == capture_argv:
            return subprocess.CompletedProcess(
                ["git", *args], 1, "", "synthetic capture failure"
            )
        return real_run_git(repo_dir, args, **kwargs)

    monkeypatch.setattr(preflight_runner, "_run_git", _broken_capture)

    result = run_hermetic_pytest(repo, timeout=120)

    assert result is not None
    assert "PREFLIGHT_CANDIDATE_ASSEMBLY" in result, result
    assert "hard block" in result, result
    assert "is not a test failure" in result, result
    assert "synthetic capture failure" in result, result
    # The remediation must not send the operator off to "finish the merge":
    # an unmerged index is a state this capture supports, so the block means
    # the capture/apply ITSELF failed and the text says exactly that.
    assert "supported source state" in result, result
    assert "mid-merge" not in result, result
    assert [event[0] for event in events].count("pass") == 0, (
        "a pass ran against a candidate whose capture failed"
    )

@pytest.mark.parametrize(
    "failure_mode, misread",
    [
        pytest.param(
            "capture_timeout", "pytest timed out",
            id="git-diff-timeout-is-not-a-pytest-timeout",
        ),
        pytest.param(
            "untracked_permission", "hermetic preflight failed",
            id="untracked-copy-permission-error-is-not-a-generic-failure",
        ),
    ],
)
def test_a_raised_assembly_exception_is_owned_by_the_assembly_block(
    tmp_path, two_pass_env, stub_passes, monkeypatch, failure_mode, misread
):
    """The assembly block must own RAISED exceptions, not only the rc!=0 path
    the failed-capture test above pins. `_run_git` raises
    subprocess.TimeoutExpired when the diff capture outruns its budget, and
    `_copy_untracked` raises OSErrors (PermissionError, FileNotFoundError) from
    the filesystem copy. On the unfixed base the block caught RuntimeError
    alone, so these flew past it into the OUTER handlers and were misread as a
    pytest timeout ("pytest timed out after N seconds") or a generic "hermetic
    preflight failed" — retryable-looking verdicts for a candidate that was
    never assembled. Both cases raise REAL exceptions through the real code
    path; neither returns a CompletedProcess(rc=1)."""
    from ouroboros import preflight_runner
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
    })
    events = stub_passes([])

    if failure_mode == "capture_timeout":
        real_run_git = preflight_runner._run_git
        capture_argv = [
            "diff", "--binary", "--no-ext-diff", "--no-textconv", "--no-color",
            "--src-prefix=a/", "--dst-prefix=b/", "HEAD",
        ]

        def _timing_out_capture(repo_dir, args, **kwargs):
            if list(args) == capture_argv:
                raise subprocess.TimeoutExpired(cmd=["git", *args], timeout=30)
            return real_run_git(repo_dir, args, **kwargs)

        monkeypatch.setattr(preflight_runner, "_run_git", _timing_out_capture)
    else:

        def _denied_copy(repo_dir, worktree):
            raise PermissionError(13, "Permission denied", str(worktree))

        monkeypatch.setattr(preflight_runner, "_copy_untracked", _denied_copy)

    result = run_hermetic_pytest(repo, timeout=120)

    assert result is not None
    assert "PREFLIGHT_CANDIDATE_ASSEMBLY" in result, result
    assert "hard block" in result, result
    assert "is not a test failure" in result, result
    assert "supported source state" in result, result
    assert misread not in result, result
    assert [event[0] for event in events].count("pass") == 0, (
        "a pass ran against a candidate whose assembly raised"
    )

def test_a_zero_context_diff_config_still_assembles_the_candidate(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """Hunk WIDTH is the config axis the capture's flag tail cannot pin: a user
    `diff.context=0` (equivalently `GIT_DIFF_OPTS=--unified=0` in the
    environment) makes `git diff` emit zero-context hunks, which `git apply`
    REJECTS by default — so on the unfixed base an ORDINARY tracked edit died
    as PREFLIGHT_CANDIDATE_ASSEMBLY before any test ran. `--unidiff-zero` on
    the apply accepts zero-context hunks and is a no-op for hunks that carry
    context, so one flag covers both the config and the env route. The
    repo-local config below is the real reviewer reproduction, not a mock."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "victim.txt": "a\nb\nc\nd\n",
    })
    _git(repo, "config", "diff.context", "0")
    (repo / "victim.txt").write_text("a\nb\nEDITED\nd\n", encoding="utf-8")

    stub_passes([])
    seen = _spy_on_candidate(monkeypatch, ["victim.txt"])

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["victim.txt"] == "a\nb\nEDITED\nd\n", (
        f"zero-context capture blocked or corrupted the candidate: {seen!r}"
    )

def test_a_staged_change_reverted_in_the_worktree_lands_as_head_content(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """Merged-path regression for the one-diff capture: a change that is staged
    but reverted in the worktree must land as HEAD content. Both schemes model
    the WORKTREE, not the index — the old pair replayed stage(A→B) then
    unstage(B→A) and netted out, the one-diff capture simply emits no hunk —
    so this pins that dropping the two-step replay did not silently start
    honouring the index's intermediate bookkeeping."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "reverted.txt": "base\n",
    })
    (repo / "reverted.txt").write_text("changed\n", encoding="utf-8")
    _git(repo, "add", "reverted.txt")
    (repo / "reverted.txt").write_text("base\n", encoding="utf-8")  # back to HEAD

    stub_passes([])
    seen = _spy_on_candidate(monkeypatch, ["reverted.txt"])

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["reverted.txt"] == "base\n", (
        f"candidate honoured the staged intermediate, not the worktree: {seen!r}"
    )

def test_disposable_index_matches_source_while_files_match_live_worktree(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """Tests see both projections: live bytes on disk and staged bytes in Git."""
    from ouroboros import preflight_runner
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "dual.txt": "head\n",
    })
    (repo / "dual.txt").write_text("staged\n", encoding="utf-8")
    _git(repo, "add", "dual.txt")
    (repo / "dual.txt").write_text("live\n", encoding="utf-8")

    stub_passes([])
    seen: dict[str, str] = {}

    def _spy(agent_python, worktree, temp_root, args, timeout):
        candidate = pathlib.Path(worktree)
        seen["live"] = (candidate / "dual.txt").read_text(encoding="utf-8")
        seen["staged"] = subprocess.run(
            ["git", "show", ":dual.txt"], cwd=candidate, check=True,
            capture_output=True, text=True,
        ).stdout
        seen["tree"] = subprocess.run(
            ["git", "write-tree"], cwd=candidate, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        return (0, "", "")

    monkeypatch.setattr(preflight_runner, "_execute_pytest_pass", _spy)

    source_tree = subprocess.run(
        ["git", "write-tree"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen == {"live": "live\n", "staged": "staged\n", "tree": source_tree}

def test_non_unmerged_source_write_tree_failure_is_a_hard_block(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    from ouroboros import preflight_runner
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    events = stub_passes([])
    real_run_git = preflight_runner._run_git

    def _broken_write_tree(repo_dir, args, **kwargs):
        if pathlib.Path(repo_dir).resolve() == repo.resolve() and list(args) == ["write-tree"]:
            return subprocess.CompletedProcess(
                ["git", *args], 128, "", "synthetic index corruption"
            )
        return real_run_git(repo_dir, args, **kwargs)

    monkeypatch.setattr(preflight_runner, "_run_git", _broken_write_tree)

    result = run_hermetic_pytest(repo, timeout=120)

    assert result is not None and "PREFLIGHT_SOURCE_INDEX" in result
    assert "synthetic index corruption" in result
    assert [event[0] for event in events].count("pass") == 0

def test_a_chmod_only_change_reaches_the_candidate(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """A mode flip with identical content is a real change (a script that lost
    its executable bit fails differently under test). The capture must carry the
    old mode/new mode header and `git apply` must apply it in the candidate.

    Gated on what `git init` actually PROBED for this filesystem (core.filemode)
    rather than on the OS name: an `os.name` skip is wrong in both directions —
    a FAT/exFAT volume on POSIX cannot track the bit either, and the probe is
    the same signal git itself trusts when deciding whether to emit mode
    hunks."""
    from ouroboros import preflight_runner
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "tool.sh": "#!/bin/sh\necho hi\n",
    })
    filemode = subprocess.run(
        ["git", "config", "--get", "core.filemode"], cwd=str(repo),
        capture_output=True, text=True,
    ).stdout.strip().lower()
    if filemode != "true":
        pytest.skip(f"this filesystem does not track the executable bit (core.filemode={filemode or 'unset'})")
    os.chmod(repo / "tool.sh", 0o755)  # unstaged mode-only change

    stub_passes([])
    seen: dict[str, int] = {}

    def _spy(agent_python, worktree, temp_root, args, timeout):
        seen["mode"] = (pathlib.Path(worktree) / "tool.sh").stat().st_mode & 0o111
        return (0, "", "")

    monkeypatch.setattr(preflight_runner, "_execute_pytest_pass", _spy)

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["mode"], "the executable bit never reached the candidate"

def test_crlf_content_survives_the_capture_byte_for_byte(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """The capture travels through `_run_git`'s binary pipes and `_apply_diff`'s
    UTF-8 re-encode; CRLF line endings are the classic casualty of a text-mode
    hop (a translated diff stops matching the LF worktree and `git apply`
    rejects it wholesale). Pinned as raw bytes — `read_text` would translate the
    very characters under test."""
    from ouroboros import preflight_runner
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "crlf.txt": "one\n",
    })
    # Pinned, not assumed: an operator/global autocrlf=true would rewrite the
    # very bytes this test is about at add/checkout time and test nothing.
    _git(repo, "config", "core.autocrlf", "false")
    (repo / "crlf.txt").write_bytes(b"one\r\ntwo\r\n")  # unstaged CRLF edit

    stub_passes([])
    seen: dict[str, bytes] = {}

    def _spy(agent_python, worktree, temp_root, args, timeout):
        seen["crlf.txt"] = (pathlib.Path(worktree) / "crlf.txt").read_bytes()
        return (0, "", "")

    monkeypatch.setattr(preflight_runner, "_execute_pytest_pass", _spy)

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["crlf.txt"] == b"one\r\ntwo\r\n", (
        f"CRLF bytes were translated in transit: {seen!r}"
    )

def test_a_staged_binary_change_reaches_the_candidate(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """Non-UTF-8 binary content travels as a base85 "GIT binary patch" section —
    which only exists because the capture passes `--binary`. Dropping the flag
    would degrade the hunk to "Binary files differ", which `git apply` cannot
    replay, so a staged icon/fixture change would kill the whole gate."""
    from ouroboros import preflight_runner
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
    })
    (repo / "blob.bin").write_bytes(b"\x00\x01\x02")
    _commit_all(repo)
    (repo / "blob.bin").write_bytes(b"\x00\xff\xfe\x00")
    _git(repo, "add", "blob.bin")

    stub_passes([])
    seen: dict[str, bytes] = {}

    def _spy(agent_python, worktree, temp_root, args, timeout):
        seen["blob.bin"] = (pathlib.Path(worktree) / "blob.bin").read_bytes()
        return (0, "", "")

    monkeypatch.setattr(preflight_runner, "_execute_pytest_pass", _spy)

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["blob.bin"] == b"\x00\xff\xfe\x00", (
        f"staged binary content did not reach the candidate: {seen!r}"
    )

def test_non_utf8_text_content_survives_the_capture_byte_for_byte(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """Git classifies NUL-free content as TEXT even when its bytes are not
    valid UTF-8 (latin-1 logs, cp1251 fixtures), so those bytes travel on plain
    diff lines — never inside a base85 binary section that the previous test
    already covers. The capture→apply hop used to decode the payload with
    errors="replace" and re-encode it: every non-UTF-8 byte on an added line
    became U+FFFD, the apply still succeeded, and the candidate SILENTLY
    diverged from the worktree while the gate stayed green. The payload now
    travels as raw bytes end to end; pinned with `read_bytes`, since a text
    read would mask the very substitution under test."""
    from ouroboros import preflight_runner
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
        "latin.txt": "plain\n",
    })
    (repo / "latin.txt").write_bytes(b"plain\ncaf\xe9 au lait\n")  # unstaged latin-1 edit

    stub_passes([])
    seen: dict[str, bytes] = {}

    def _spy(agent_python, worktree, temp_root, args, timeout):
        seen["latin.txt"] = (pathlib.Path(worktree) / "latin.txt").read_bytes()
        return (0, "", "")

    monkeypatch.setattr(preflight_runner, "_execute_pytest_pass", _spy)

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["latin.txt"] == b"plain\ncaf\xe9 au lait\n", (
        f"non-UTF-8 text bytes were substituted in transit: {seen!r}"
    )

def test_a_staged_add_removed_from_the_worktree_is_absent_from_the_candidate(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """A file `git add`ed and then deleted from the worktree exists only in the
    index. The worktree-vs-HEAD capture emits no hunk for it (absent on both
    sides) and the untracked copy cannot see it (it is IN the index, so
    `ls-files --others` skips it) — the candidate must not resurrect it. The
    old pair reached the same absence the long way round: staged add, then
    unstaged delete."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    repo = _make_repo(tmp_path, {
        "tests/test_plain.py": "def test_ok():\n    assert True\n",
    })
    (repo / "ghost.txt").write_text("ghost\n", encoding="utf-8")
    _git(repo, "add", "ghost.txt")
    (repo / "ghost.txt").unlink()

    stub_passes([])
    seen = _spy_on_candidate(monkeypatch, ["ghost.txt"])

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen["ghost.txt"] is None, (
        f"an index-only file was resurrected in the candidate: {seen!r}"
    )

def test_an_untracked_file_with_a_non_utf8_name_reaches_the_candidate(
    tmp_path, two_pass_env, stub_passes, monkeypatch
):
    """POSIX filenames are BYTES, and Git lists them as such. Decoding that list
    as UTF-8 with `errors="replace"` turned a raw non-UTF-8 byte into U+FFFD, so
    the reconstructed path did not exist, `is_file()` said False, and the file was
    skipped in silence — an inexact candidate with no PREFLIGHT_CANDIDATE_ASSEMBLY
    block. The names are now decoded with the filesystem codec (surrogateescape),
    which round-trips the original bytes."""
    from ouroboros.preflight_runner import run_hermetic_pytest

    if os.name != "posix":
        pytest.skip("byte filenames are a POSIX property")
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    raw_name = b"fixture_\xff.dat"
    try:
        with open(os.path.join(os.fsencode(str(repo)), raw_name), "wb") as handle:
            handle.write(b"untracked payload\n")
    except (OSError, UnicodeError):  # APFS and friends enforce UTF-8 names
        pytest.skip("this filesystem rejects non-UTF-8 filenames")
    decoded_name = os.fsdecode(raw_name)

    stub_passes([])
    seen = _spy_on_candidate(monkeypatch, [decoded_name])

    assert run_hermetic_pytest(repo, timeout=120) is None
    assert seen[decoded_name] == "untracked payload\n", (
        f"an untracked file vanished from the candidate on a byte filename: {seen!r}"
    )

@pytest.mark.skipif(
    os.name == "nt",
    reason=(
        "POSIX-only invariant: the guarantee is that a raw non-UTF-8 filename byte "
        "survives to the copy via os.fsdecode's surrogateescape round-trip. Windows "
        "uses a UTF-16 filesystem where such a name cannot exist, and its fs codec "
        "(utf-8/surrogatepass) raises on the synthetic 0xff byte this test injects — "
        "git never emits such a name on Windows, so the production path is unaffected."
    ),
)
def test_untracked_listing_is_decoded_with_the_filesystem_codec(tmp_path, monkeypatch):
    """Filesystem-independent pin for the same defect (the test above can only run
    where non-UTF-8 names are creatable): the listing is read as BYTES and each
    name goes through `os.fsdecode`, so the original bytes reach the copy instead
    of a U+FFFD name that matches no file on disk. POSIX-only: os.fsdecode is only
    byte-transparent under surrogateescape (POSIX); see the skip marker."""
    from ouroboros import preflight_runner

    seen_kwargs = {}

    def fake_run_git(_repo_dir, args, **kwargs):
        # Mirrors the real seam: bytes only when the caller asks for them, the old
        # utf-8/replace decode otherwise — so this pins the decision, not the stub.
        seen_kwargs.update(kwargs)
        raw = b"fixture_\xff.dat\x00"
        out = raw if kwargs.get("binary_stdout") else raw.decode("utf-8", "replace")
        return subprocess.CompletedProcess(list(args), 0, out, "")

    copied = []
    monkeypatch.setattr(preflight_runner, "_run_git", fake_run_git)
    monkeypatch.setattr(preflight_runner.shutil, "copy2", lambda src, dst: copied.append(src))
    monkeypatch.setattr(pathlib.Path, "is_file", lambda _self: True)

    preflight_runner._copy_untracked(tmp_path, tmp_path / "candidate")

    assert seen_kwargs.get("binary_stdout") is True, "the names must not be decoded by _run_git"
    assert os.fsencode(str(copied[0])).endswith(b"fixture_\xff.dat"), copied
