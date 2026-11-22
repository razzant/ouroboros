"""Interpreter-family observations and execution through the selected Supervisor.

Parser observations retain their existing shape without becoming permission
proof. Actual run_script cases verify useful source execution and one effect."""
from __future__ import annotations

import pathlib
import sys

import pytest

from ouroboros.tools.shell_guards import (
    _python_write_targets_and_unknown,
    interpreter_family,
    interpreter_inline_code,
    writer_target_tokens,
)
from tests._cases import case_folders


# ---- the classifier itself: one structural owner ---------------------------

FAMILY_SPELLINGS = [
    # (spelling, family) — bare, versioned, pathed, Windows and ABI forms.
    ("python", "python"), ("python3", "python"), ("python3.11", "python"),
    ("python3.11.exe", "python"), ("pythonw", "python"), ("python3.7m", "python"),
    ("pypy", "python"), ("pypy3.9", "python"),
    ("/opt/homebrew/bin/python3.11", "python"),
    ("node", "node"), ("node18", "node"), ("nodejs", "node"),
    ("ruby", "ruby"), ("ruby3.2", "ruby"), ("/usr/bin/ruby3.2", "ruby"),
    ("perl", "perl"), ("perl5.38", "perl"), ("perl5.38.2", "perl"),
    ("php", "php"), ("php8.3", "php"), ("PHP8.3.EXE", "php"),
]

NON_INTERPRETERS = [
    # Same stems, different programs: name-shaped lookalikes must NOT classify —
    # the classifier is structural (stem + dotted version), not a prefix match.
    "python-config", "perldoc", "phpunit", "php-fpm", "ruby-build", "node-gyp",
    "nodew", "sh", "bash", "zsh", "rm", "sed", "",
]


def test_interpreter_family_recognizes_every_family_and_versioned_form():
    for spelling, family in FAMILY_SPELLINGS:
        assert interpreter_family(spelling) == family, spelling


def test_interpreter_family_rejects_stem_lookalikes():
    for spelling in NON_INTERPRETERS:
        assert interpreter_family(spelling) == "", spelling


# ---- surface 1: inline write targets per interpreter family ----------------

# (unversioned, versioned, inline flag, write payload) — each payload carries a
# write form for that family.
FENCE_CASES = [
    ("python3", "python3.11", "-c", "open('probe.txt','w').write('x')"),
    ("ruby", "ruby3.2", "-e", "File.write('probe.txt', 'x')"),
    ("perl", "perl5.38", "-e", "unlink('probe.txt')"),
    ("node", "node18", "-e", "fs.writeFileSync('probe.txt','x')"),
]


def test_versioned_spellings_read_inline_write_targets_like_unversioned():
    """The versioned spelling of an interpreter yields the same inline write targets
    as its family name; the python/ruby/node literal readers name the probe path."""
    # Parsing only, so a fixed path: a native temp path (D:\a\_temp\...) inside a
    # Python literal would decode \a and \t and never read back equal.
    probe = "/srv/probe/repo/probe.txt"
    for unversioned, versioned, flag, code in FENCE_CASES:
        named = code.replace("probe.txt", probe)
        results = {
            spelling: writer_target_tokens([spelling, flag, named])
            for spelling in (unversioned, versioned)
        }
        # Parity is the regression: pre-fix the versioned spelling fell through
        # while its family name was read.
        assert results[versioned] == results[unversioned], results
        if unversioned in {"python3", "ruby", "node"}:
            assert probe in results[versioned], results


def test_versioned_writer_spelling_reads_writer_operands_like_the_family():
    # ruby/perl are LIGHT_SHELL_WRITER_COMMANDS members; membership must
    # canonicalize the versioned spelling to the family name so a versioned
    # spelling reports the same operands as write targets.
    for exe in ("ruby", "ruby3.2", "perl", "perl5.38"):
        assert "BIBLE.md" in writer_target_tokens([exe, "rewrite.rb", "BIBLE.md"]), exe


def _light_registry(tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    reg = ToolRegistry(repo_dir=repo, drive_root=tmp_path / "drive")
    reg._ctx.task_id = "t1"
    return reg


# ---- surface 2: protected-artifact high-risk interpreter check --------------


def test_versioned_spellings_are_high_risk_interpreters():
    from ouroboros.protected_artifacts import _is_high_risk_interpreter

    for spelling, _family in FAMILY_SPELLINGS:
        name = pathlib.PurePath(spelling).name.lower().removesuffix(".exe")
        assert _is_high_risk_interpreter(name) is True, spelling
    # Shells stay covered by the explicit set, not the family classifier.
    assert _is_high_risk_interpreter("bash") is True
    # Lookalikes stay out.
    assert _is_high_risk_interpreter("perldoc") is False


# =========================================================================
# XG-7B3.1: the inline-flag / write-primitive vocabularies are per-family
# and complete.
# =========================================================================


# (family, flag spelling, code payload writing PATH) — every flag here was verified
# by EXECUTION against the real interpreter (php's from the upstream CLI manual).
INLINE_FLAG_PAYLOADS = [
    ("python3.11", ["-c", "open('{p}','w').write('x')"]),
    ("python3.11", ["-copen('{p}','w').write('x')"]),                  # joined short
    ("node18", ["-e", "require('fs').writeFileSync('{p}','x')"]),
    ("node", ["--eval=require('node:fs').writeFileSync('{p}','x')"]),  # long =
    ("node", ["-p", "fs.appendFileSync('{p}','x')"]),
    ("ruby3.2", ["-e", "File.write('{p}', 'x')"]),
    ("ruby", ["-eFile.write('{p}', 'x')"]),                            # joined short
    ("perl5.38", ["-e", "open(F,'>','{p}');"]),
    ("perl5.38", ["-E", "unlink('{p}');"]),                            # perl -E IS eval
    ("php8.3", ["-r", "file_put_contents('{p}','x');"]),
    ("php8.3", ["-R", "file_put_contents('{p}','x');"]),               # per-line code
]

# Forms carrying no inline code at all: the flag is not an inline-code flag for that
# family, or there is no flag. A table that guessed these in would read a FILENAME as
# code.
NON_INLINE_FORMS = [
    ["ruby", "-E", "utf-8", "s.rb"],    # encoding selector, not eval
    ["php", "-F", "file.php"],          # per-line FILE, not code
    ["node", "script.js"],              # plain script
    ["python3", "-m", "pdb", "x.py"],   # module form
]

# `-c` is accepted for EVERY family on purpose: `process_shell_guard_args` normalizes
# a run_script call into the synthetic argv `[interpreter, "-c", script]` whatever the
# interpreter is. So a real `ruby -c file` / `perl -c file` compile check DOES yield a
# located "body".


def test_inline_code_extraction_is_per_family_and_shape_complete():
    """Every verified flag, in all three spellings that really execute — separate
    token, joined short flag, long flag with `=` — yields the CODE and not the flag."""
    for exe, rest in INLINE_FLAG_PAYLOADS:
        argv = [exe, *[part.format(p="/tmp/probe") for part in rest]]
        bodies = interpreter_inline_code(argv)
        assert len(bodies) == 1, (argv, bodies)
        body = bodies[0]
        assert "/tmp/probe" in body, (argv, body)
        assert not body.startswith("-"), (argv, body)


def test_non_inline_forms_yield_no_code_body():
    for argv in NON_INLINE_FORMS:
        assert interpreter_inline_code(argv) == [], argv


def test_the_synthetic_run_script_argv_is_located_for_every_family():
    """`process_shell_guard_args` hands the guards `[interpreter, "-c", script]` for
    EVERY interpreter, so `-c` must locate a body in every family — reading it
    per-family (node wants `-e`) silently stopped extracting run_script write targets,
    and a run_script writing through a symlink out of the workspace stopped being
    blocked (tests/test_headless_cli.py's symlink-escape test caught it)."""
    for interpreter in ("node", "ruby", "perl", "php", "python3"):
        argv = [interpreter, "-c", "require('fs').writeFileSync('filelink','x')"]
        assert interpreter_inline_code(argv) == [argv[2]], interpreter
    # node's literal target is extracted from that synthetic shape, which is what the
    # workspace guard needs.
    from ouroboros.tools.shell_guards import writer_target_tokens

    assert "filelink" in writer_target_tokens(
        ["node", "-c", "require('fs').writeFileSync('filelink','x')"]
    )


def test_php_carries_several_inline_bodies_at_once():
    # -B/-R/-E are three separate code arguments in one command.
    assert interpreter_inline_code(
        ["php8.3", "-B", "BEGIN", "-R", "EACH", "-E", "END"]
    ) == ["BEGIN", "EACH", "END"]


def test_unparseable_python_code_is_unknown_not_a_proven_read():
    """No token list can prove the ABSENCE of a write in arbitrary code, so the
    python walker answers UNKNOWN once execution leaves what it models: computed
    payloads and execution handed to another process."""
    for code in (
        "exec(open('/dev/stdin').read())",
        "import subprocess;subprocess.run(['rm','x.py'])",
        "import os;os.system('rm x.py')",
        "eval(__import__('os').environ['C'])",
    ):
        _targets, unknown = _python_write_targets_and_unknown(code)
        assert unknown is True, code


def test_only_a_real_parse_earns_a_proven_read():
    """A fully-resolved AST with no write targets is a proven read; the same shape
    whose execution leaves the parse is UNKNOWN."""
    assert _python_write_targets_and_unknown("print(open('README.md').read())") == ([], False)
    _targets, unknown = _python_write_targets_and_unknown("print(eval(open('README.md').read()))")
    assert unknown is True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shim script")
def test_run_script_follows_the_configured_supervisor(tmp_path, monkeypatch):
    """The script body executes once when the chosen Supervisor permits it."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    reg = _light_registry(tmp_path)
    repo = pathlib.Path(reg._ctx.repo_dir)
    target = repo / "via_script.py"
    target.write_text("original\n")
    result = reg.execute("run_script", {
        "script": f"open({str(target)!r},'w').write('MUTATED')",
    })
    assert "exit_code=0" in result, result
    assert target.read_text() == "MUTATED"


@pytest.mark.parametrize("cwd", ["", "task_drive"])
def test_readonly_python_string_method_executes_without_a_false_write(tmp_path, monkeypatch, cwd):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _light_registry(tmp_path)
    target = pathlib.Path(reg._ctx.repo_dir) / "probe.txt"
    target.write_text("probe text\n")
    script = f"text = open({target.as_posix()!r}).read(); print(text.replace('probe', 'read'))"
    # A string method is ordinary code, not proof of a filesystem mutation.
    import subprocess

    direct = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                            capture_output=True, text=True, timeout=10)
    assert direct.returncode == 0 and direct.stdout == "read text\n\n"
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    result = reg.execute("run_script", {"script": script, "cwd": cwd})
    assert "exit_code=0" in result and "read text" in result, result
    assert target.read_text() == "probe text\n"
    read = reg.execute("read_file", {"path": "probe.txt", "root": "system_repo"})
    assert "probe text" in read and "LIGHT_MODE_BLOCKED" not in read


@pytest.mark.parametrize("body", [
    "from pathlib import Path\np = Path('.')\nprint((p / '.git').exists())\n"
    "with (p / 'result.txt').open('a', newline='') as f: f.write('once\\n')\n",
    "from pathlib import Path\np = Path('nested')\np.mkdir(exist_ok=True)\n"
    "with (p / 'result.txt').open('a', newline='') as f: f.write('once\\n')\n",
])
def test_path_join_code_executes_once_without_guessing_a_control_write(tmp_path, monkeypatch, body):
    reg = _light_registry(tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    root = pathlib.Path(reg._ctx.task_drive_root())
    result = reg.execute("run_script", {"script": body, "cwd": "task_drive"})
    assert "exit_code=0" in result, result
    outputs = list(root.rglob("result.txt"))
    assert len(outputs) == 1 and outputs[0].read_bytes() == b"once\n"


def test_literal_deliverable_writes_name_their_target(tmp_path):
    """The per-family literal readers report a literal write target outside the
    repo; a python read is a proven read wherever it points."""
    out = tmp_path / "drive" / "task_drives" / "t1" / "deliverable.txt"
    for argv in (
        ["node", "-e", f"require('fs').writeFileSync('{out}','x')"],
        ["ruby3.2", "-e", f"File.write('{out}','x')"],
        ["python3.11", "-c", f"open({str(out)!r},'w').write('x')"],
    ):
        assert str(out) in writer_target_tokens(argv), argv
    # as_posix() keeps the source valid python on Windows too (raw C:\Users would
    # be a \U escape).
    read = f"print(open('{pathlib.PurePath(tmp_path).as_posix()}/x.py').read())"
    assert _python_write_targets_and_unknown(read) == ([], False)
    assert writer_target_tokens(["python3.11", "-c", read]) == []


def test_a_template_literal_backtick_is_an_ordinary_path_byte():
    """The POSIX absolute-path harvest does not treat a closing backtick as a
    delimiter, so ``String.raw`<root>` `` yields the root plus the backtick. A real
    sibling path with a literal backtick in its name stays harvestable as it is."""
    from ouroboros.shell_parse import embedded_absolute_path_tokens

    posix_repo = "/srv/host/repo"
    code = 'require("fs").rmSync(String.raw`' + posix_repo + '`, {recursive:true})'
    assert embedded_absolute_path_tokens(code) == [posix_repo + "`"]


# ---------------------------------------------------------------------------
# ONE write vocabulary: the AST walker never reads a known write as a proven read.
# ---------------------------------------------------------------------------

_ORDINARY_STDLIB_WRITES = [
    ("shutil.copy", "import shutil; shutil.copy('/tmp/src', {t})"),
    ("shutil.move", "import shutil; shutil.move('/tmp/src', {t})"),
    ("shutil.copytree", "import shutil; shutil.copytree('/tmp/src', {t})"),
    ("Path.touch", "import pathlib; pathlib.Path({t}).touch()"),
    ("os.symlink", "import os; os.symlink('/tmp/src', {t})"),
    ("os.truncate", "import os; os.truncate({t}, 0)"),
    ("os.chmod", "import os; os.chmod({t}, 0o600)"),
    ("zipfile.extractall", "import zipfile; zipfile.ZipFile('/tmp/a.zip').extractall({t})"),
    ("shutil.unpack_archive", "import shutil; shutil.unpack_archive('/tmp/a.zip', {t})"),
]


def test_an_ordinary_stdlib_write_is_never_read_as_a_proven_read(subtests, tmp_path):
    """Nine ordinary stdlib write APIs parsed to a fully-resolved AST with ZERO write
    targets, so `targets == [] and not unknown` was handed out for a payload that
    writes. The walker now either resolves the target or says UNKNOWN (BIBLE P2).
    """
    case_paths = case_folders(tmp_path)
    for label, template in _ORDINARY_STDLIB_WRITES:
        tmp_path = next(case_paths)
        with subtests.test(label=label, template=template):
            victim = str(tmp_path / "victim.txt")
            code = template.format(t=repr(victim))
            # The DESTINATION is the target, never the source operand.
            assert _python_write_targets_and_unknown(code) == ([victim], False), label


def test_a_write_through_an_untraceable_handle_is_unknown_not_a_proven_read(tmp_path):
    """The same defect one spelling over: `.write` on a receiver the walker cannot
    trace to a path contributed no target AND no unknown, so an unlocatable write
    read as proof of a read. Locatable handles must keep resolving."""
    for code in (
        'f = get_handle(); f.write("x")',
        'import tempfile; f = tempfile.NamedTemporaryFile(); f.write(b"x")',
    ):
        targets, unknown = _python_write_targets_and_unknown(code)
        assert unknown and not targets, code

    # ...while the shapes it CAN trace still resolve to the written file.
    out = repr(str(tmp_path / "out.txt"))
    for code in (
        f'open({out}, "w").write("x")',
        f'import pathlib; pathlib.Path({out}).open("w").write("x")',
        f'f = open({out}, "w"); f.write("x")',
        f'import json; f = open({out}, "w"); json.dump({{}}, f)',
    ):
        targets, unknown = _python_write_targets_and_unknown(code)
        assert targets and not unknown, code
        assert str(tmp_path / "out.txt") in writer_target_tokens(["python3.11", "-c", code]), code


def test_the_vocabulary_repair_does_not_widen_to_name_collisions():
    """`.copy()` on a dict and `dump()` on an arbitrary object are not writes, so the
    walker must not call them write-capable either."""
    for code in ("d = {}; e = d.copy()", "import yaml; print(yaml.dump({}))", "logger.dump()"):
        assert _python_write_targets_and_unknown(code) == ([], False), code


def test_builtin_replace_and_remove_are_not_filesystem_writes():
    """`.replace`/`.rename`/`.remove` were credited as writes on ANY receiver.

    `.replace` is the most common string method in Python; there is no such thing
    as a filesystem write via `'a,b'.replace`. Same receiver discrimination the
    module already applies to `shutil.copy` vs `some_dict.copy()`.
    """
    for code in (
        "print('a,b'.replace(',', ';'))",
        "d = [1, 2]; d.remove(1); print(d)",
        "print('a-b'.replace('-', '_').upper())",
    ):
        assert _python_write_targets_and_unknown(code) == ([], False), code

    # ...and the GENUINE path writes those same spellings also name are never a
    # proven read: the receiver-is-a-path forms resolve to a literal target, the
    # `os.replace`/`os.rename` forms answer UNKNOWN.
    for code in (
        "import pathlib; pathlib.Path('pkg/mod.py').replace('other.py')",
        "import pathlib; pathlib.Path('pkg/mod.py').rename('other.py')",
        "import os; os.replace('pkg/mod.py', 'other.py')",
        "import os; os.rename('pkg/mod.py', 'other.py')",
        "import os; os.remove('pkg/mod.py')",
    ):
        targets, unknown = _python_write_targets_and_unknown(code)
        assert targets or unknown, code  # never `([], False)` — a proven read
