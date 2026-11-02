"""Explicit black-box operand contracts and ordinary execution capabilities.

Actual file/script operands and shell redirections retain task authority.
Interpreter bodies, quoted patterns and data mentions are not physical targets.
Positive registry execution coverage lives in test_tool_capabilities_black_box_policy.
"""
from types import SimpleNamespace

import pytest

import ouroboros.tools  # noqa: F401  — normal package init order (registry first)
from ouroboros.protected_artifacts import shell_block_reason


REF = "reference_executable"


def _ctx(tmp_path, *, allow=("execute",), deny=None):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / REF).write_bytes(b"\x7fELF-blackbox")
    record = {"id": "programbench_reference", "role": "black_box_reference", "paths": [f"./{REF}"]}
    if allow is not None:
        record["allow"] = list(allow)
    if deny is not None:
        record["deny"] = list(deny)
        record.pop("allow", None)
    ctx = SimpleNamespace(
        workspace_root=str(workspace),
        workspace_mode="external",
        repo_dir=str(workspace),
        drive_root=str(tmp_path / "data"),
        task_metadata={"task_contract": {"resource_policy": {"protected_artifacts": [record]}}},
        task_contract={"resource_policy": {"protected_artifacts": [record]}},
    )
    return ctx, workspace


def _block(ctx, workspace, cmd):
    return shell_block_reason(ctx, cmd, cwd="", default_cwd=workspace)


# ---------------------------------------------------------------------------
# Core denials (the fa59893 harness cases): direct content access stays blocked.

@pytest.mark.parametrize("cmd", [
    ["cat", "./" + REF],
    ["cp", "./" + REF, "/tmp/stolen"],
    ["xxd", "./" + REF],
    ["dd", "if=./" + REF, "of=/tmp/out"],
    ["strings", "./" + REF],
    ["sha256sum", "./" + REF],
])
def test_direct_content_access_blocked(tmp_path, cmd):
    ctx, ws = _ctx(tmp_path)
    assert "RESOURCE_POLICY_BLOCKED" in _block(ctx, ws, cmd)


def test_interpreter_reading_artifact_as_script_blocked(tmp_path):
    ctx, ws = _ctx(tmp_path)
    assert "RESOURCE_POLICY_BLOCKED" in _block(ctx, ws, ["python3", "./" + REF])


def test_write_and_delete_of_artifact_blocked(tmp_path):
    ctx, ws = _ctx(tmp_path)
    assert "RESOURCE_POLICY_BLOCKED" in _block(ctx, ws, ["rm", "-f", "./" + REF])
    assert "RESOURCE_POLICY_BLOCKED" in _block(ctx, ws, ["sh", "-c", "echo x > ./" + REF])


# ---------------------------------------------------------------------------
# Sanctioned differential-testing workflows pass.

@pytest.mark.parametrize("cmd", [
    ["./" + REF, "--help"],
    "./reference_executable --version > ref.out 2>&1",
    "./reference_executable input.wav out.wav && ./executable input.wav out2.wav && diff out.wav out2.wav",
    "python3 -c \"import subprocess; print(subprocess.run(['./reference_executable','-h'], capture_output=True).stdout[:64])\"",
])
def test_execute_and_output_capture_allowed(tmp_path, cmd):
    ctx, ws = _ctx(tmp_path)
    assert _block(ctx, ws, cmd) == ""


def test_write_targets_only_mention_with_scratch_redirect_allowed(tmp_path):
    """fa59893 pin: redirecting to a scratch file while MENTIONING the artifact
    is the differential-loop shape, not a write to the artifact."""
    ctx, ws = _ctx(tmp_path)
    assert _block(ctx, ws, "./reference_executable probe > probes/ref.txt") == ""


# ---------------------------------------------------------------------------
# Round-2 structural exceptions.

def test_compound_segmentation_touch_then_execute_allowed(tmp_path):
    """Live smoke2 FP: `touch /tmp/x && ./reference_executable /tmp/x` read the
    execute mention as a touch WRITE target."""
    ctx, ws = _ctx(tmp_path)
    assert _block(ctx, ws, ["sh", "-c", "touch /tmp/hxprobe/empty && ./reference_executable /tmp/hxprobe/empty"]) == ""


def test_ln_source_is_not_a_write_target(tmp_path):
    ctx, ws = _ctx(tmp_path)
    assert _block(ctx, ws, ["ln", "-sf", str(ws / REF), "/tmp/hxprobe/executable"]) == ""


def test_git_diff_output_limited_shapes_allowed_content_diff_blocked(tmp_path):
    ctx, ws = _ctx(tmp_path)
    # Output-LIMITED diffs (names/stat only — the vcs_read(action=diff, stat=true) FP shape) pass.
    assert _block(ctx, ws, ["git", "diff", "--stat"]) == ""
    assert _block(ctx, ws, ["git", "diff", "--staged", "--name-only"]) == ""
    assert _block(ctx, ws, ["git", "diff", "--stat", "HEAD"]) == ""  # limited output wins over rev
    # A BARE worktree diff can dump a modified text file's content → still blocked
    # via the whole-work_dir fallback (conservative: the guard can't know the
    # protected artifact is binary); a rev/content-flag diff likewise.
    assert "RESOURCE_POLICY_BLOCKED" in _block(ctx, ws, ["git", "diff"])
    assert "RESOURCE_POLICY_BLOCKED" in _block(ctx, ws, ["git", "diff", "4b825dc642cb6eb9a060e54bf8d69288fbee4904"])
    assert "RESOURCE_POLICY_BLOCKED" in _block(ctx, ws, ["git", "diff", "--binary"])
    # A pathspec naming the protected file blocks even under --stat.
    assert "RESOURCE_POLICY_BLOCKED" in _block(ctx, ws, ["git", "diff", "--stat", "--", REF])
    assert "RESOURCE_POLICY_BLOCKED" in _block(ctx, ws, ["git", "show", "HEAD:" + REF])


def test_heredoc_quoted_mentions_in_verify_check_allowed(tmp_path):
    """Live smoke2 FP: a verify check whose python heredoc merely QUOTES the
    protected filename in asserts got its tokens treated as read candidates."""
    ctx, ws = _ctx(tmp_path)
    check = (
        "python3 - <<'PY'\n"
        "import subprocess\n"
        "ignored = subprocess.check_output(['git','status','--short','--ignored'], text=True)\n"
        "assert '!! executable' in ignored\n"
        "assert '!! reference_executable' in ignored\n"
        "print('OK')\n"
        "PY"
    )
    assert _block(ctx, ws, ["sh", "-c", check]) == ""


def test_pty_spawn_with_output_reads_allowed(tmp_path):
    """Live smoke2 FP: pty differential probes spawn the artifact and read the
    pty STREAM (`os.read(fd)`) — execute + output capture, not a byte read."""
    ctx, ws = _ctx(tmp_path)
    script = (
        "import os, pty, select\n"
        "pid, fd = pty.fork()\n"
        "if pid == 0:\n"
        "    os.execv('./reference_executable', ['./reference_executable', '--help'])\n"
        "out = b''\n"
        "while True:\n"
        "    r, _, _ = select.select([fd], [], [], 1)\n"
        "    if not r: break\n"
        "    out += os.read(fd, 4096)\n"
        "print(len(out))\n"
    )
    assert _block(ctx, ws, ["python3", "-c", script]) == ""
    # pexpect.spawn shape too.
    script2 = "import pexpect\nchild = pexpect.spawn('./reference_executable --i x.wav')\nprint(child.read())\n"
    assert _block(ctx, ws, ["python3", "-c", script2]) == ""


def test_open_near_mention_is_not_a_direct_target(tmp_path):
    """An inline body is assessed as a whole by the configured Supervisor."""
    ctx, ws = _ctx(tmp_path)
    script = (
        "import subprocess\n"
        "subprocess.run(['./reference_executable', '-h'])\n"
        "data = open('./reference_executable', 'rb').read()\n"
    )
    assert _block(ctx, ws, ["python3", "-c", script]) == ""


def test_execute_denied_artifact_requires_actual_execution_operand(tmp_path):
    """A concrete execute operand is distinct from a quoted program body."""
    ctx, ws = _ctx(tmp_path, allow=None, deny=["execute", "read_bytes", "copy", "hash",
                                               "static_introspection", "dynamic_trace", "debug",
                                               "write", "delete"])
    script = "import subprocess\nsubprocess.run(['./reference_executable', '-h'])\n"
    assert _block(ctx, ws, ["python3", "-c", script]) == ""
    assert "RESOURCE_POLICY_BLOCKED" in _block(ctx, ws, ["./" + REF, "--help"])


def test_proximity_does_not_change_program_body_admission(tmp_path):
    """Source proximity never establishes a physical target."""
    ctx, ws = _ctx(tmp_path)
    far = (
        "import subprocess, hashlib\n"
        "out = subprocess.run(['./reference_executable', 'x'], capture_output=True).stdout\n"
        + "pad = 1\n" * 40 +
        "print(hashlib.sha256(out).hexdigest())\n"
    )
    assert _block(ctx, ws, ["python3", "-c", far]) == ""
    near = "import hashlib\nprint(hashlib.sha256(open('./reference_executable','rb').read()).hexdigest())\n"
    assert _block(ctx, ws, ["python3", "-c", near]) == ""


# --- v6.56.0 review regressions: round-2 exception BYPASSES (adversarial) ------


def test_program_body_spawn_arguments_are_not_direct_operands(tmp_path):
    """Program-internal argv remains opaque to the direct operand policy."""
    ctx, ws = _ctx(tmp_path)
    for prog in ("cat", "cp", "sha256sum", "xxd", "od", "install"):
        arg = "'/tmp/x'" if prog in ("cp", "install") else ""
        sep = ", " if arg else ""
        script = f"import subprocess\nsubprocess.run(['{prog}', './reference_executable'{sep}{arg}])\n"
        assert _block(ctx, ws, ["python3", "-c", script]) == "", prog
    # os.execvp with a read tool as the program, artifact as an argv element.
    ex = "import os\nos.execvp('cat', ['cat', './reference_executable'])\n"
    assert _block(ctx, ws, ["python3", "-c", ex]) == ""
    # But the genuine execute (artifact IS argv[0], incl. the exec(prog,[prog,...])
    # echo) stays exempt.
    assert _block(ctx, ws, ["python3", "-c",
                            "import subprocess\nsubprocess.run(['./reference_executable', '-h'])\n"]) == ""
    assert _block(ctx, ws, ["python3", "-c",
                            "import os\nos.execv('./reference_executable', ['./reference_executable', '-h'])\n"]) == ""


def test_git_diff_stat_with_patch_flag_dumps_content_blocks(tmp_path):
    """`--stat`/`--name-only` are content-free ONLY without a patch flag: `-p`/
    `-u`/`--patch` re-enable hunk output alongside the stat and can dump the
    protected file's content, so the whole-work_dir fallback must NOT be skipped."""
    ctx, ws = _ctx(tmp_path)
    for cmd in (["git", "diff", "--stat", "-p"],
                ["git", "diff", "--name-only", "-p"],
                ["git", "diff", "-p", "--stat"],
                ["git", "diff", "--stat", "--patch"],
                ["git", "diff", "--numstat", "-U5"]):
        assert "RESOURCE_POLICY_BLOCKED" in _block(ctx, ws, cmd), cmd
    # content-free stat diffs (the real vcs_read diff FP) still pass.
    assert _block(ctx, ws, ["git", "diff", "--stat"]) == ""
    assert _block(ctx, ws, ["git", "diff", "--name-only"]) == ""


def test_python_dash_m_file_operand_read_blocks(tmp_path):
    """`python -m <module> <file>` OPENS the file operand (pdb/py_compile/zipfile/
    trace); the artifact passed as that operand is a read/copy and must block."""
    ctx, ws = _ctx(tmp_path)
    for cmd in (["python3", "-m", "pdb", "./reference_executable"],
                ["python3", "-m", "py_compile", "./reference_executable"],
                ["python3", "-m", "zipfile", "-c", "/tmp/o.zip", "./reference_executable"]):
        assert "RESOURCE_POLICY_BLOCKED" in _block(ctx, ws, cmd), cmd


def test_alias_source_is_not_a_direct_read_target(tmp_path):
    """Variable binding inside source does not create a direct operand."""
    ctx, ws = _ctx(tmp_path)
    pad = "\n".join(f"pad{i} = {i}" for i in range(60))
    for read in ("print(open(p, 'rb').read())",
                 "import pathlib; q = pathlib.Path(p); data = q.read_bytes()",
                 "import shutil; shutil.copy(p, '/tmp/leak')"):
        script = f"p = './reference_executable'\n{pad}\n{read}"
        assert _block(ctx, ws, ["python3", "-c", script]) == "", read


def test_alias_bound_execute_not_a_read_allowed(tmp_path):
    """A program body may perform useful differential execution."""
    ctx, ws = _ctx(tmp_path)
    pad = "\n".join(f"pad{i} = {i}" for i in range(60))
    script = f"import subprocess\np = './reference_executable'\n{pad}\nsubprocess.run([p, '--version'], capture_output=True)"
    assert _block(ctx, ws, ["python3", "-c", script]) == ""


# --- v6.57.0 (1.6): glob carve-out — a scratch-cleaning glob beside the ref -----


def test_scratch_cleaning_glob_beside_protected_allowed(tmp_path):
    """`rm -f *.out` must NOT block just because the black-box binary sits in the
    same directory — the glob pattern `*.out` cannot match `reference_executable`.
    Now the pattern is matched against the protected basename, not the whole dir.
    (A compound `rm ... && ./ref` remains a separate flat-parse limitation.)"""
    ctx, ws = _ctx(tmp_path)
    assert _block(ctx, ws, ["rm", "-f", "*.out"]) == ""
    assert _block(ctx, ws, ["rm", "-f", "*.txt"]) == ""
    # A glob subdir'd away from the artifact is likewise fine.
    assert _block(ctx, ws, ["rm", "-f", "probes/*.out"]) == ""


def test_glob_matching_protected_name_still_blocks(tmp_path):
    """A glob whose pattern CAN match the protected artifact's basename still blocks
    (the carve-out is precise, not a blanket un-blocking of globs)."""
    ctx, ws = _ctx(tmp_path)
    assert "RESOURCE_POLICY_BLOCKED" in _block(ctx, ws, ["sh", "-c", "rm -f reference_*"])
    assert "RESOURCE_POLICY_BLOCKED" in _block(ctx, ws, ["sh", "-c", "rm -f reference_executabl?"])


def test_refusal_names_nearest_allowed_action(tmp_path):
    """The refusal message names WHAT IS allowed (execute + capture its output) so
    the agent redirects in one move (1.6)."""
    ctx, ws = _ctx(tmp_path)
    reason = _block(ctx, ws, ["cat", "./" + REF])
    assert "RESOURCE_POLICY_BLOCKED" in reason
    assert "EXECUTE" in reason and "stdout" in reason
