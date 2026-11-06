"""Write-target observations, concrete utility boundaries, and ordinary interpreter capability."""

from __future__ import annotations

import os
import pathlib
import shlex

import pytest

# Shares the real-subprocess-adjacent registry harness with
# test_external_workspace_access.py; keep it in the serial lane.

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.tools.shell_guards import interpreter_write_shape, shell_has_write_indicator
from tests._typed_guard_shared import _shell_guard_text




@pytest.fixture(autouse=True)
def _home_outside_tmp(tmp_path, monkeypatch):
    # Same premise as test_external_workspace_access.py: keep tmp scratch
    # outside $HOME on every platform so host-scratch reads stay non-runtime.
    fake_home = tmp_path / "_home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setattr(pathlib.Path, "home", lambda: fake_home)


def _registry(tmp_path: pathlib.Path, *, mode: str = "external", acting: bool = False) -> ToolRegistry:
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for p in (system, workspace, data):
        p.mkdir(exist_ok=True)
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    reg.set_context(
        ToolContext(
            repo_dir=system,
            drive_root=data,
            workspace_root=workspace,
            workspace_mode=mode,
            task_id="task-write-shape",
            task_constraint=(TaskConstraint(mode="acting_subagent", surface="external_workspace",
                                            write_root=str(workspace)) if acting else None),
        )
    )
    return reg


READ_ONLY_HASH_SCRIPT = (
    "import hashlib\n"
    "def h(p):\n"
    "    with open(p, 'rb') as f:\n"
    "        return hashlib.sha256(f.read()).hexdigest()\n"
    "print(h({target!r}))\n"
)


# Outside-workspace write refusals below use an ACTING child. Ordinary roots
# have the same user_files write authority regardless of their selected cwd.

# --- unit layer: the classifier itself -------------------------------------


def test_read_only_open_is_not_interpreter_write_shape():
    cmd = ["python3", "-c", "with open('f.bin', 'rb') as f:\n    print(len(f.read()))"]
    assert interpreter_write_shape(cmd) is False
    # The legacy coarse classifier keeps its pinned behavior for its other
    # consumers (_protected_shell_block, ws5 carryover).
    assert shell_has_write_indicator(cmd) is True


def test_write_mode_open_and_pathlib_open_stay_write_shaped():
    assert interpreter_write_shape(["python3", "-c", "open('/d/x', 'w').write('hi')"]) is True
    assert (
        interpreter_write_shape(
            ["python3", "-c", "from pathlib import Path; Path('/d/x').open('w')"]
        )
        is True
    )


def test_opaque_subprocess_and_library_saves_stay_write_shaped():
    assert (
        interpreter_write_shape(
            ["python3", "-c", "import subprocess; subprocess.run(['rm', '-rf', '/d/x'])"]
        )
        is True
    )
    assert interpreter_write_shape(["python3", "-c", "df.to_csv('out.csv')"]) is True
    assert interpreter_write_shape(["python3", "-c", "fh.writelines(rows)"]) is True
    assert (
        interpreter_write_shape(
            ["node", "-e", "const {writeFileSync} = require('fs'); writeFileSync('x', 'y')"]
        )
        is True
    )


def test_shell_level_signals_still_write_shaped_for_interpreters():
    assert interpreter_write_shape(["sh", "-c", "python3 -c 'print(1)' && rm -rf /tmp/x"]) is True
    assert interpreter_write_shape("python3 gen.py > out.txt") is True
    assert interpreter_write_shape(["sh", "-c", "python3 gen.py && cp out.txt /tmp/y"]) is True


def test_ruby_perl_pure_reads_are_not_write_shaped():
    """LIGHT_SHELL_WRITER_COMMANDS membership (a coarse shell-writer role) must not
    re-add the write shape the mode-aware classifier just re-judged."""
    assert interpreter_write_shape(["ruby", "-e", "puts File.read('/tmp/f.txt')"]) is False
    # perl 3-arg READ open: the filename's own letters (the 'x' in f.txt) must not
    # classify the mode.
    assert (
        interpreter_write_shape(["perl", "-e", "open(my $fh, '<', '/tmp/f.txt'); print <$fh>"])
        is False
    )
    assert interpreter_write_shape(["ruby", "-e", "File.write('/tmp/x', 'y')"]) is True


def test_perl_ruby_native_write_idioms_stay_write_shaped():
    """fable-5 review: dropping the membership floor demands the vocabulary
    actually SEE perl/ruby write spellings — '>'-mode opens, File.delete,
    FileUtils with a variable argument, IO.binwrite."""
    assert interpreter_write_shape(["perl", "-e", "open(FH,'>','/outside/x'); print FH 'data'"]) is True
    assert interpreter_write_shape(["perl", "-e", "open(FH, '>>', $log); print FH $line"]) is True
    assert interpreter_write_shape(["ruby", "-e", "File.delete('/outside/x')"]) is True
    assert interpreter_write_shape(["ruby", "-e", "f='/x'; FileUtils.rm_rf(f)"]) is True
    assert interpreter_write_shape(["ruby", "-e", "IO.binwrite('a.bin', d)"]) is True


def test_ruby_file_open_is_mode_aware(tmp_path):
    """sol review: File.open must be a write target only with a write-mode 2nd arg —
    File.open('/x','r') is a READ and must not be reported as a write, while
    File.open('/x','w') and File.new('/x','w') stay writes."""
    from ouroboros.tools.shell_guards import writer_target_tokens
    read = ["ruby", "-e", "File.open('/tmp/r','r') { |f| puts f.read }"]
    # The literal path is NOT emitted as a write target for a read-mode open.
    assert "/tmp/r" not in writer_target_tokens(read)
    assert interpreter_write_shape(read) is False
    assert interpreter_write_shape(["ruby", "-e", "File.open('/tmp/o','w') { |f| f.puts 'x' }"]) is True
    assert "/tmp/o" in writer_target_tokens(["ruby", "-e", "File.open('/tmp/o','w') { |f| f.puts 'x' }"])
    assert interpreter_write_shape(["ruby", "-e", "File.new('/tmp/o','w')"]) is True
    assert interpreter_write_shape(["ruby", "-e", "File.new('/tmp/o','r')"]) is False


def test_keyword_mode_open_is_write_shaped():
    """sol review: open('/x', mode='w') with a statically-known keyword mode is a
    real write (distinct from the disclosed variable-mode residual)."""
    assert interpreter_write_shape(["python3", "-c", "open('/tmp/o', mode='w')"]) is True
    assert interpreter_write_shape(["python3", "-c", "open('/tmp/o', mode='rb')"]) is False


def test_unspaced_posix_redirect_is_write_shaped():
    """fable-5 review: `python3 gen.py>out.txt` is ONE shlex token; the redirect
    shape must be recognized mid-token, while located inline-code bodies keep
    their '>' comparisons/filehandles as reads."""
    assert interpreter_write_shape("python3 gen.py>out.txt") is True
    assert interpreter_write_shape("python3 gen.py>>log.txt") is True
    assert interpreter_write_shape(["python3", "-c", "print(1 if a > b else 2)"]) is False
    assert interpreter_write_shape(["python3", "-c", "x = {'k': 'v => w'}; print(x)"]) is False


def test_prose_words_are_not_write_shapes_for_interpreters():
    """Natural-language words in code text ('scp done', 'count deleted rows') are
    not write evidence; structural spellings (os.remove, rm in a compound) are."""
    assert interpreter_write_shape(["python3", "-c", "print(open('/tmp/f').read()); print('scp done')"]) is False
    assert interpreter_write_shape(["python3", "-c", "print('count deleted rows'); print(open('/tmp/f').read())"]) is False
    assert interpreter_write_shape(["python3", "-c", "print('results truncated')"]) is False
    assert interpreter_write_shape(["python3", "-c", "import os; os.remove('/tmp/x')"]) is True
    assert interpreter_write_shape(["python3", "-c", "f.truncate(0)"]) is True


def test_python_inline_strings_and_comments_do_not_create_write_shape():
    """Writer vocabulary is structural: API names in output text/comments are
    not filesystem channels, while an actual redirect outside the body remains
    visible to the shell lane."""
    assert interpreter_write_shape(
        ["python3", "-c", "print('BIBLE.md write_text'); # os.remove('/tmp/x')"]
    ) is False
    assert interpreter_write_shape(
        "sh -c \"python3 -c 'print(\\\"BIBLE.md write_text\\\")'\""
    ) is False
    assert interpreter_write_shape(
        "python3 -c \"print('BIBLE.md write_text')\" > report.txt"
    ) is True


def test_python_collection_constructor_remove_is_not_a_path_write():
    """A list/tuple/set/dict constructor produces a collection receiver; an item
    named like a protected file is not a filesystem target."""
    from ouroboros.tools.shell_guards import writer_target_rows

    command = [
        "python3", "-c",
        "xs = list(('BIBLE.md',)); xs.remove('BIBLE.md'); print(xs)",
    ]
    assert interpreter_write_shape(command) is False
    assert writer_target_rows(command)[0][1] == []


# --- guard layer: workspace lanes ------------------------------------------


def test_external_pure_read_outside_runtime_is_allowed(tmp_path):
    """The census class (rows 1-5): a read-only interpreter hash/inspect over
    host scratch was refused as 'write-like'; it is a plain allowed read."""
    reg = _registry(tmp_path, mode="external")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    target = scratch / "artifact.bin"
    target.write_bytes(b"payload")
    cmd = ["python3", "-c", READ_ONLY_HASH_SCRIPT.format(target=str(target))]
    assert _shell_guard_text(reg, {"cmd": cmd, "cwd": str(tmp_path / "workspace")}, "advanced") is None


def test_workspace_mode_pure_read_outside_root_is_allowed(tmp_path):
    reg = _registry(tmp_path, mode="workspace")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    target = scratch / "report.txt"
    target.write_text("data", encoding="utf-8")
    cmd = ["python3", "-c", f"print(open({str(target)!r}, 'r').read())"]
    assert _shell_guard_text(reg, {"cmd": cmd, "cwd": str(tmp_path / "workspace")}, "advanced") is None


def test_external_sh_wrapped_pure_read_is_allowed(tmp_path):
    reg = _registry(tmp_path, mode="external")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    target = scratch / "blob.bin"
    target.write_bytes(b"x" * 16)
    inner = f"python3 -c \"print(open({str(target)!r}, 'rb').read(8))\""
    assert _shell_guard_text(reg, {"cmd": ["sh", "-c", inner], "cwd": str(tmp_path / "workspace")}, "advanced") is None


def test_pure_filter_reads_outside_root_are_allowed(tmp_path):
    """Scope-C: sort/uniq/sed -n/tar -tf/gzip -l READ invocations must not be
    'write-like' — membership alone is not a write channel."""
    reg = _registry(tmp_path, mode="external")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    data_file = scratch / "data.csv"
    data_file.write_text("b\na\n", encoding="utf-8")
    archive = scratch / "a.tar"
    archive.write_bytes(b"x" * 16)
    for cmd in (
        ["sort", str(data_file)],
        ["uniq", str(data_file)],
        ["sed", "-n", "1p", str(data_file)],
        ["tar", "-tf", str(archive)],
        ["gzip", "-l", str(archive)],
    ):
        out = _shell_guard_text(reg, {"cmd": cmd, "cwd": str(tmp_path / "workspace")}, "advanced")
        assert out is None, (cmd, out)


def test_sed_script_write_channels_stay_write_shaped():
    """fable-5 round-2: sed writes WITHOUT -i too — the POSIX in-script `w FILE`
    command, the `s///w` flag, GNU `s///e` execute, a -f script file (unprovable),
    and the GNU attached `-ibak` suffix. All must keep writer targets; plain
    filters and patterns containing prose words stay reads."""
    from ouroboros.tools.shell_guards import writer_target_tokens
    for cmd in (
        ["sed", "w out.py", "f"],
        ["sed", "-n", "s/a/b/w out.txt", "f"],
        ["sed", "s/x/y/e", "f"],
        ["sed", "-f", "script.sed", "f"],
        ["sed", "-e", "w dump.txt", "f"],
        ["sed", "-ibak", "s/x/y/", "f"],
    ):
        assert writer_target_tokens(cmd), cmd
    for cmd in (
        ["sed", "-n", "1,40p", "f"],
        ["sed", "-n", "/delete/p", "f"],
        ["sed", "s/hello/world/g", "f"],
    ):
        assert writer_target_tokens(cmd) == [], cmd

def test_sol_r2_channel_grammar_writes_stay_write_shaped():
    """sol-max round-2: option/operand grammar gaps — every one of these is a
    real write and must report a target (or write shape)."""
    from ouroboros.tools.shell_guards import writer_target_tokens
    for cmd in (
        ["sort", "--output", "/o", "/etc/hosts"],
        ["uniq", "-", "/o"],
        ["sed", "-nibak", "s/a/b/", "/f"],
        ["sed", "-n", "1w /o", "/etc/hosts"],
        ["tar", "-cf/o.tar", "/etc/hosts"],
        ["tar", "--extract", "--file=/i.tar", "--directory=/od"],
        ["tar", "xf", "/a.tar"],
        ["gzip", "-S.tgz", "/f"],
    ):
        assert writer_target_tokens(cmd), cmd
    # Old-style/list/read spellings stay reads.
    for cmd in (
        ["tar", "tf", "/a.tar"],
        ["tar", "-tf", "/a.tar"],
        ["gzip", "-l", "/a.gz"],
        ["sed", "s/e/x/", "/f"],
        ["sed", "-n", "/e/p", "/f"],
    ):
        assert writer_target_tokens(cmd) == [], cmd


def test_sol_r2_compound_and_carve_holes_closed(tmp_path):
    """sol-max round-2: a pure-filter HEAD speaks only for its own segment, and
    uniq's '-' stdin operand cannot hide its output operand from the carve."""
    from ouroboros.tools.registry import _is_pure_read_inspection
    from ouroboros.tools.write_shape import non_interpreter_write_shape
    assert (
        non_interpreter_write_shape(
            "sort /etc/hosts && find /d -name x -delete",
            ["sort"], "sort", is_pure_read=_is_pure_read_inspection,
        )
        is True
    )
    assert _is_pure_read_inspection("printf x | uniq - data/settings.json") is False
    assert _is_pure_read_inspection("uniq /var/log/a.txt") is True


def test_inline_body_isolation_for_joined_and_wrapped_forms():
    """sol-max round-2: a '>' comparison inside a located body is not a redirect
    even when the body rides a joined flag or an sh -c wrap; real redirects
    outside the body still classify."""
    assert interpreter_write_shape(["python3", "-cprint(2 > 1)"]) is False
    assert (
        interpreter_write_shape(
            ["sh", "-c", "python3 -c 'print(2 > 1); print(open(\"/etc/hosts\").read())'"]
        )
        is False
    )
    assert interpreter_write_shape(["sh", "-c", "python3 gen.py > out.txt"]) is True


def test_pure_filter_write_channels_still_blocked(tmp_path):
    """The real channels stay writes: sort -o, sed -i, uniq's second operand,
    tar extract, gzip default all still take guard B."""
    reg = _registry(tmp_path, mode="external", acting=True)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    src = scratch / "in.txt"
    src.write_text("x\n", encoding="utf-8")
    missing = []
    for cmd in (
        ["sort", "-o", str(scratch / "out.txt"), str(src)],
        ["sed", "-i", "s/a/b/", str(src)],
        ["uniq", str(src), str(scratch / "out.txt")],
        ["tar", "-xf", str(scratch / "a.tar"), "-C", str(scratch)],
        ["gzip", str(src)],
    ):
        out = _shell_guard_text(reg, {"cmd": cmd, "cwd": str(tmp_path / "workspace")}, "advanced") or ""
        if "WORKSPACE_SHELL_BLOCKED" not in out:
            missing.append((cmd, out))
    assert missing == []


@pytest.mark.parametrize(
    "cmd",
    (
        ["/usr/bin/python3", "-c", 'import os; os.write(1, b"x")'],
        ["/opt/homebrew/bin/node", "-e", "console.log(1)"],
    ),
    ids=("python-os-write-fd", "node-console-log"),
)
def test_unprovable_row_never_promotes_absolute_executable_to_write(tmp_path, cmd):
    from ouroboros.tools.registry import _workspace_write_candidates

    forced_uncertain_row = [(cmd, [], (cmd[-1],), True)]
    assert not any(
        token == cmd[0] and is_write
        for token, is_write, _row in _workspace_write_candidates(forced_uncertain_row, [], cmd)
    )
    reg = _registry(tmp_path, mode="external")
    assert _shell_guard_text(reg,
        {"cmd": cmd, "cwd": str(tmp_path / "workspace")}, "advanced"
    ) is None


def test_os_write_through_literal_os_open_retains_target_observation():
    from ouroboros.tools.shell_guards import _python_write_targets_and_unknown

    code = (
        'import os; fd=os.open("/Users/Shared/out", os.O_WRONLY|os.O_CREAT); '
        'os.write(fd, b"x")'
    )
    targets, unknown = _python_write_targets_and_unknown(code)
    assert targets == ["/Users/Shared/out"] and unknown is False

def test_split_redirections_grammar():
    """One redirect grammar: both the glued and the split spellings, reads and
    descriptor duplication yield no target, and an operator away from a token's
    start is left alone."""
    from ouroboros.shell_parse import shell_argv, split_redirections

    assert split_redirections(shell_argv("cp a b 2>/dev/null")) == (["cp", "a", "b"], [])
    assert split_redirections(shell_argv("cp x y >> log.txt")) == (["cp", "x", "y"], ["log.txt"])
    assert split_redirections(shell_argv("node t.js > out.log 2>&1")) == (
        ["node", "t.js"],
        ["out.log"],
    )
    assert split_redirections(shell_argv("printf x >&2")) == (["printf", "x"], [])
    assert split_redirections(shell_argv("tee out.txt < in.txt")) == (["tee", "out.txt"], [])
    assert split_redirections(shell_argv("cat <<'EOF' > out.txt")) == (["cat"], ["out.txt"])
    # The split spelling the operator-aware lexer emits: a bare descriptor token
    # belongs to the operator that follows it.
    assert split_redirections(["cp", "a", "b", "2", ">", "/dev/null"]) == (["cp", "a", "b"], [])
    assert split_redirections(shell_argv("echo hi >|out")) == (["echo", "hi"], ["out"])
    assert split_redirections(shell_argv("echo hi >& /outside/log")) == (["echo", "hi"], ["/outside/log"])
    assert split_redirections(shell_argv("echo hi >1")) == (["echo", "hi"], ["1"])
    assert split_redirections(shell_argv("echo hi >2")) == (["echo", "hi"], ["2"])
    assert split_redirections(shell_argv("echo hi >&word")) == (["echo", "hi"], ["word"])
    for descriptor_redirect in ("2>&1", ">&-", ">&2"):
        assert split_redirections(shell_argv(f"echo hi {descriptor_redirect}")) == (["echo", "hi"], [])
    # A `>`/`<` away from position 0 is a literal byte, not an operator.
    assert split_redirections(["sed", "s/>/x/", "f"]) == (["sed", "s/>/x/", "f"], [])
    assert split_redirections(["git", "log", "--pretty=<x>"]) == (
        ["git", "log", "--pretty=<x>"],
        [],
    )


def test_writer_targets_recover_the_destination_behind_a_redirect():
    """The writer-target lane reads the operator-aware view, so a redirect no
    longer displaces the command's own destination."""
    from ouroboros.tools.shell_guards import writer_target_rows, writer_target_tokens

    assert writer_target_tokens(["cp", "x", "y", ">>", "log.txt"]) == ["y", "log.txt"]
    # A row is (segment_argv, targets, inline_code, unprovable); a `cd` operand
    # takes the target policy because it is how a later relative write escapes.
    assert writer_target_rows("cd . && cp src.txt /D/.env") == [
        (["cd", "."], ["."], (), False),
        (["cp", "src.txt", "/D/.env"], ["/D/.env"], (), False),
    ]


def test_workspace_rows_filter_bodies_without_weakening_the_light_fence():
    from ouroboros.tools.shell_guards import writer_target_rows, writer_target_tokens

    body = "print '/outside/mentioned.txt'"
    # The light fence keeps the historical unfiltered signal. The workspace
    # lane removes the body from its path targets; its mode-aware classifier
    # proves this body read-only, so it carries no row uncertainty.
    assert writer_target_tokens(["perl", "-e", body]) == [body]
    assert writer_target_rows(["perl", "-e", body]) == [
        (["perl", "-e", body], [], (body,), False),
    ]


def test_cp_source_outside_root_is_a_read_and_destination_still_blocked(tmp_path):
    """A writer's SOURCE operand is a read: copying an outside file INTO the
    process root is the sanctioned transfer route, while an outside DESTINATION
    stays refused."""
    reg = _registry(tmp_path, mode="external", acting=True)
    workspace = str(tmp_path / "workspace")
    outside = tmp_path / "outside"
    outside.mkdir()

    def check(cmd):
        return _shell_guard_text(reg, {"cmd": cmd, "cwd": workspace}, "advanced")

    assert check(["cp", str(outside / "widget.js"), "widget.js"]) is None
    assert check(["ln", "-s", str(outside / "src"), "link.js"]) is None
    assert check(["cat", str(outside / "widget.js")]) is None
    destination_block = check(["cp", "widget.js", str(outside / "copy.js")]) or ""
    assert "WORKSPACE_SHELL_BLOCKED" in destination_block
    assert str(outside / "copy.js") in destination_block


def test_direct_copy_sources_are_read_operands(tmp_path):
    """The direct write destination does not turn a source into a mutation."""
    reg = _registry(tmp_path, mode="external")
    workspace = str(tmp_path / "workspace")
    (tmp_path / "data" / "settings.json").write_text("{}", encoding="utf-8")
    (tmp_path / "system" / "ouroboros").mkdir(parents=True, exist_ok=True)
    (tmp_path / "system" / "ouroboros" / "safety.py").write_text("x = 1\n", encoding="utf-8")

    for cmd, _source in (
        (["cp", "../data/settings.json", "./x"], tmp_path / "data" / "settings.json"),
        (
            ["cp", "../system/ouroboros/safety.py", "./x"],
            tmp_path / "system" / "ouroboros" / "safety.py",
        ),
    ):
        out = _shell_guard_text(reg, {"cmd": cmd, "cwd": workspace}, "advanced") or ""
        assert out == "", cmd


def test_glued_operator_is_not_a_path_candidate(tmp_path):
    """A redirection glued to the following separator is not a path: `2>/dev/null;`
    was forged into the target `/dev/null;` and refused a pure read."""
    reg = _registry(tmp_path, mode="external", acting=True)
    workspace = str(tmp_path / "workspace")
    outside = tmp_path / "outside"
    outside.mkdir()

    def check(cmd):
        return _shell_guard_text(reg, {"cmd": cmd, "cwd": workspace}, "advanced")

    assert check(["sh", "-c", "git reset HEAD scratch/ 2>/dev/null; rm -rf scratch/"]) is None
    assert check(["sh", "-c", "node build.js 2>/dev/null; echo ok"]) is None
    redirect_block = check(["sh", "-c", f"node t.js > {shlex.quote(str(outside / 'out.log'))} 2>&1"]) or ""
    assert "WORKSPACE_SHELL_BLOCKED" in redirect_block
    assert str(outside / "out.log") in redirect_block


@pytest.mark.parametrize(
    "body",
    ("(cd sub && make) 2>&1", "echo $(date) 2>/dev/null"),
    ids=("subshell-stderr-dup", "command-substitution-dev-null"),
)
def test_round5_redirect_only_segments_are_allowed_without_crash(tmp_path, body):
    reg = _registry(tmp_path, mode="external")
    workspace = tmp_path / "workspace"
    (workspace / "sub").mkdir()
    assert _shell_guard_text(reg,
        {"cmd": ["sh", "-c", body], "cwd": str(workspace)}, "advanced"
    ) is None


def test_f7_python_literal_heredoc_read_stays_allowed(tmp_path):
    reg = _registry(tmp_path, mode="external")
    outside = tmp_path / "outside"
    outside.mkdir()
    command = f"python3 - <<'EOF'\nprint(open(\"{outside / 'read'}\").read())\nEOF"
    assert _shell_guard_text(reg,
        {"cmd": command, "cwd": str(tmp_path / "workspace")}, "advanced"
    ) is None


def test_round5_sequential_effective_cwd_blocks_nested_escape(tmp_path):
    reg = _registry(tmp_path, mode="external", acting=True)
    workspace = tmp_path / "workspace"
    fixtures = workspace / "fixtures"
    fixtures.mkdir()
    for body in (
        "cd .. && echo x > ../outside",
        "cd .. && touch ../docs/X",
        "pushd .. && echo x > ../outside",
    ):
        out = _shell_guard_text(reg,
            {"cmd": ["sh", "-c", body], "cwd": str(fixtures)}, "advanced"
        ) or ""
        assert "WORKSPACE_SHELL_BLOCKED" in out, (body, out)


def test_round5_sequential_effective_cwd_keeps_in_workspace_write_allowed(tmp_path):
    reg = _registry(tmp_path, mode="external")
    workspace = tmp_path / "workspace"
    (workspace / "sub").mkdir()
    (workspace / "tools").mkdir()
    for body in ("cd sub && echo x > f", "cd sub && touch ../file", "cd tools && touch ../shell_parse.py"):
        args = {"cmd": ["sh", "-c", body], "cwd": str(workspace)}
        assert _shell_guard_text(reg, args, "advanced") is None, body


def test_round6_redirect_file_targets_outside_workspace_are_blocked(tmp_path):
    reg = _registry(tmp_path, mode="external", acting=True)
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "1").symlink_to(outside / "numeric-log")
    for body in (f"echo x >& {shlex.quote(str(outside / 'redirect-log'))}", "echo x >1"):
        args = {"cmd": ["sh", "-c", body], "cwd": str(workspace)}
        out = _shell_guard_text(reg, args, "advanced") or ""
        assert "WORKSPACE_SHELL_BLOCKED" in out, (body, out)


def test_round5_python_stdin_heredoc_read_without_dash_is_allowed(tmp_path):
    reg = _registry(tmp_path, mode="external")
    outside = tmp_path / "outside"
    outside.mkdir()
    command = f"python3 <<'EOF'\nprint(open({str(outside / 'read')!r}).read())\nEOF"
    assert _shell_guard_text(reg,
        {"cmd": command, "cwd": str(tmp_path / "workspace")}, "advanced"
    ) is None


def test_f8_uncertain_perl_row_does_not_widen_independent_cat(tmp_path):
    reg = _registry(tmp_path, mode="external")
    outside = tmp_path / "outside"
    outside.mkdir()
    command = f"perl -e 'print 1' && cat {outside / 'read'}"
    assert _shell_guard_text(reg,
        {"cmd": command, "cwd": str(tmp_path / "workspace")}, "advanced"
    ) is None


def test_round3_old_block_coverage_stays_blocked(tmp_path):
    reg = _registry(tmp_path, mode="external", acting=True)
    workspace = str(tmp_path / "workspace")
    outside = tmp_path / "outside"
    outside.mkdir()

    commands = (
        ("cd_relative_write", f"cd {shlex.quote(str(outside))} && echo x > rel"),
        ("pushd_relative_write", f"pushd {shlex.quote(str(outside))} && echo x > rel"),
        ("awk_redirect", f"awk '{{print $1}}' input > {shlex.quote(str(outside / 'awk'))}"),
        ("rsync_destination", ["rsync", "src", pathlib.Path(os.path.relpath(outside / "rsync", workspace)).as_posix()]),
        ("tar_chdir_extract", ["tar", "-C", str(outside), "-xf", "a.tar"]),
        ("append_redirect", f"echo x >> {shlex.quote(str(outside / 'append'))}"),
        ("stderr_redirect", f"awk '{{print $1}}' input 2> {shlex.quote(str(outside / 'stderr'))}"),
        ("combined_redirect", f"awk '{{print $1}}' input &> {shlex.quote(str(outside / 'combined'))}"),
    )
    if pathlib.Path(r"C:\outside\drive.txt").is_absolute():
        # These are physical absolute targets on Windows; on POSIX the
        # backslashes are ordinary filename characters, not an outside root.
        commands += (
            ("windows_drive", ["cp", "x", r"C:\outside\drive.txt"]),
            ("windows_unc", ["cp", "x", r"\\server\share\unc.txt"]),
        )
    missing = []
    for name, command in commands:
        # Shell syntax is executable only through an explicit shell. A legacy
        # command string is normalized as argv and keeps operators literal.
        process_command = ["sh", "-c", command] if isinstance(command, str) else command
        out = _shell_guard_text(reg, {"cmd": process_command, "cwd": workspace}, "advanced") or ""
        if "WORKSPACE_SHELL_BLOCKED" not in out:
            missing.append((name, process_command, out))
    assert missing == []


def test_round3_body_uncertainty_variants_are_row_scoped():
    from ouroboros.tools.shell_guards import writer_target_rows

    find_row = writer_target_rows(["find", "/tree", "-exec", "tee", "copy-{}", ";"])[0]
    assert find_row[1] == []
    assert find_row[3] is True
    for method in ("exec", "spawn", "execSync", "spawnSync"):
        row = writer_target_rows(
            ["node", "-e", f'child_process.{method}("rm /outside")']
        )[0]
        assert row[3] is True, method
    ruby_row = writer_target_rows(
        ["ruby", "-e", 'FileUtils.mv("/outside/source", destination)']
    )[0]
    assert ruby_row[3] is True


def test_round3_ordinary_outside_reads_stay_allowed(tmp_path):
    reg = _registry(tmp_path, mode="external")
    workspace = str(tmp_path / "workspace")
    outside = tmp_path / "outside"
    outside.mkdir()
    commands = (
        ("grep", ["grep", "x", str(outside / "read")]),
        ("rg", ["rg", "x", str(outside / "read")]),
        ("cat", ["cat", str(outside / "read")]),
        ("ls", ["ls", str(outside)]),
        ("git_status", ["git", "-C", str(outside), "status"]),
        ("pytest", ["pytest", str(outside / "test.py")]),
        (
            "python_open_read",
            ["python3", "-c", f'print(open("{outside / "read"}").read())'],
        ),
        (
            "python_heredoc_read",
            f"python3 - <<'EOF'\nprint(open(\"{outside / 'read'}\").read())\nEOF",
        ),
        ("mixed_perl_cat", f"perl -e 'print 1' && cat {outside / 'read'}"),
        (
            "perl_open_read",
            ["perl", "-e", f'open(my $fh, "<", "{outside / "read"}"); print <$fh>'],
        ),
    )
    for name, command in commands:
        out = _shell_guard_text(reg, {"cmd": command, "cwd": workspace}, "advanced")
        assert out is None, (name, command, out)


# --- Windows spellings: the first windows-latest execution of this serial suite ---
#
# Five failures, two root causes, neither in tokenization (shlex posix mode strips
# only UNQUOTED backslashes, and the raw-text harvest lane already recovers such a
# spelling): (1) a Windows path typed into a plain Python literal ("C:\Users\x") is
# not a valid Python string (\U opens a unicode escape), so the AST lane reported
# UNKNOWN and the row fail-closed into an outside-root WRITE for a pure read;
# (2) the block named only the natively RESOLVED path (C:\Users\Shared\x for a
# command that wrote /Users/Shared/x). The tests below push the exact Windows
# spellings through the host-independent seams. The one branch that cannot run
# here — pathlib resolving a drive path natively (`os.name == "nt"` in
# _workspace_shell_write_block) — is exercised through the message builder with
# PureWindowsPath inputs: PosixPath cannot resolve a drive path, and patching
# os.name breaks pathlib's flavour selection rather than simulating Windows.

WINDOWS_TMP = r"C:\Users\runneradmin\AppData\Local\Temp\pytest-of-runneradmin\pytest-0"


def test_windows_spelled_python_literals_are_read_verbatim():
    from ouroboros.tools.shell_guards import _python_write_targets_and_unknown
    from ouroboros.tools.write_shape import python_body_ast

    read = f'print(open("{WINDOWS_TMP}\\outside\\read").read())'
    assert python_body_ast(read) is not None
    assert _python_write_targets_and_unknown(read) == ([], False)
    assert _python_write_targets_and_unknown(
        f"print(open('{WINDOWS_TMP}\\scratch\\blob.bin', 'rb').read(8))"
    ) == ([], False)
    # Writes keep their targets in the spelling the model typed, through every
    # modelled channel: mode-open, os.open, pathlib join.
    out = f"{WINDOWS_TMP}\\outside\\out"
    assert _python_write_targets_and_unknown(f'open("{out}", "w").write("x")') == ([out], False)
    assert _python_write_targets_and_unknown(
        f'import os; fd=os.open("{out}", os.O_WRONLY|os.O_CREAT); os.write(fd, b"x")'
    ) == ([out], False)
    assert _python_write_targets_and_unknown(
        f'from pathlib import Path; (Path("{WINDOWS_TMP}") / "outside" / "out").write_text("y")'
    ) == ([out], False)
    # An already-escaped spelling parses first time and keeps the same value.
    assert _python_write_targets_and_unknown(f"open({out!r}, 'w')") == ([out], False)
    # Uncertainty is untouched: an opaque body stays UNKNOWN, and so does a body
    # unparseable for any reason other than its literals (python2 print).
    assert _python_write_targets_and_unknown(
        f'import subprocess; subprocess.run(["rm", "{out}"])'
    )[1] is True
    assert _python_write_targets_and_unknown(f'print "{out}"') == ([], True)


def test_verbatim_string_source_respells_only_unescaped_backslashes():
    import ast

    from ouroboros.tools.write_shape import _verbatim_string_source

    for source, value in (
        (r'"C:\Users\x"', "C:\\Users\\x"),
        (r"b'C:\Users\x'", b"C:\\Users\\x"),
        (r'"""C:\Users\x"""', "C:\\Users\\x"),
        (r'"a\"b\\c"', 'a"b\\c'),  # quote and backslash escapes are kept
        (r"'it\'s'", "it's"),
        ('"\\n"', "\\n"),  # verbatim, not a newline
    ):
        assert ast.literal_eval(_verbatim_string_source(source)) == value, source
    assert _verbatim_string_source(r'r"C:\Users\x"') == r'r"C:\Users\x"'
    assert _verbatim_string_source('x = 1\ny = "C:\\Users\\x"\n') == 'x = 1\ny = "C:\\\\Users\\\\x"\n'
    assert _verbatim_string_source('open("unterminated') is None


def test_windows_spelled_pure_reads_stay_allowed_on_every_host(tmp_path):
    """The three windows-latest shapes, spelled exactly as the Windows run spelled
    them, allowed on a POSIX host too — the classification is host-independent."""
    reg = _registry(tmp_path, mode="external")
    workspace = str(tmp_path / "workspace")
    read = f"{WINDOWS_TMP}\\outside\\read"
    for name, command in (
        ("python_c", ["python3", "-c", f'print(open("{read}").read())']),
        ("python_heredoc", f"python3 - <<'EOF'\nprint(open(\"{read}\").read())\nEOF"),
        ("sh_wrapped_repr", ["sh", "-c", f"python3 -c \"print(open({read!r}, 'rb').read(8))\""]),
    ):
        out = _shell_guard_text(reg, {"cmd": command, "cwd": workspace}, "advanced")
        assert out is None, (name, command, out)


def test_outside_root_block_names_the_spelling_the_model_used(tmp_path):
    """When the resolved path differs from the operand (a relative spelling, a
    symlink alias — or, on Windows, a POSIX-rooted spelling resolved onto the
    cwd drive), the block names both, resolved first so earlier pins hold."""
    reg = _registry(tmp_path, mode="external", acting=True)
    workspace = tmp_path / "workspace"
    real = tmp_path / "outside_real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    for spelled in (f"{alias / 'f'}", "../outside_real/f"):
        text = _shell_guard_text(reg,
            {"cmd": ["sh", "-c", f"echo x > {shlex.quote(spelled)}"], "cwd": str(workspace)}, "advanced",
        ) or ""
        assert "WORKSPACE_SHELL_BLOCKED" in text and str(workspace) in text, spelled
        assert spelled in text or str(real / "f") in text
    # Identical spellings are named once.
    text = _shell_guard_text(reg,
        {"cmd": ["sh", "-c", f"echo x > {shlex.quote(str(real / 'g'))}"], "cwd": str(workspace)}, "advanced",
    ) or ""
    assert "WORKSPACE_SHELL_BLOCKED" in text and str(real / "g") in text


def test_block_messages_carry_a_windows_resolution_beside_the_spelling():
    """The `os.name == "nt"` resolving branch hands the builders a natively
    resolved WindowsPath for a POSIX-rooted operand; the message must still
    show the operand the model typed."""
    from ouroboros.tools.registry_guards import (
        _workspace_write_block_outside_root_message,
        _workspace_write_block_runtime_message,
    )

    resolved = pathlib.PureWindowsPath(r"C:\Users\Shared\x")
    root = pathlib.PureWindowsPath(r"C:\Users\runneradmin\ws")
    text = _workspace_write_block_outside_root_message(resolved, root, spelled="/Users/Shared/x")
    assert r"Blocked path: C:\Users\Shared\x (as written: /Users/Shared/x)." in text
    assert r"Selected process root: C:\Users\runneradmin\ws." in text
    runtime = _workspace_write_block_runtime_message(resolved, spelled=str(resolved))
    assert r"Blocked path: C:\Users\Shared\x." in runtime
    assert "(as written" not in runtime
    assert "Blocked path" not in _workspace_write_block_runtime_message("")
