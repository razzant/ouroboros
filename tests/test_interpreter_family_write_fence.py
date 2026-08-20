"""XG-2R.2: interpreter recognition is STRUCTURAL, owned once, consumed by every guard.

INFRA-1 fixed the versioned-basename bypass for python alone (`startswith("python")`
beside an exact set), so the versioned spellings every OTHER interpreter family
ships under — ruby3.2, php8.3, perl5.38, node18 — still bypassed the light-mode
inline-write fence (`shell_guards.light_shell_repo_mutation`), the registry
runtime_data scan trigger (`registry_guard_process._run_shell_safety_check`), and the
protected-artifact high-risk-interpreter check. That patched one interpreter, not
the failure class (BIBLE P2).

The class fix is ONE structural classifier — `shell_guards.interpreter_family` —
recognizing family stem + optional dotted version, which all three surfaces consume.
These tests are table-driven across families and versioned forms and assert PARITY:
a versioned spelling must classify and guard exactly like its unversioned family
name. They fail on the pre-fix tree (exact-set + python-only startswith).

XG-7B3.1 (second half of the same class, two reviewers converged): classification was
fixed, REACHABILITY was not. `registry_guard_process._run_shell_safety_check` passed
`detect_interpreter_inline=False` for `run_command`, and inline code was parsed only
behind `-c`, so in light mode `run_command(["node18", "-e", "...writeFileSync(...)"])`
mutated an ordinary repo file BEFORE the post-execution tripwire — which reports
without rolling back. Three things were wrong at once and all three are covered
below: the fence did not reach `run_command`; the inline-FLAG vocabulary was
`-c`-only; and the write-INDICATOR vocabulary was python-shaped, so the reviewers'
own `require('fs').writeFileSync` / `file_put_contents` payloads did not even trip
it. The e2e tests put a real executable SHIM on PATH, so a guard that lets the
command through is caught by the mutated file, not merely by a missing block string.
"""
from __future__ import annotations

import inspect
import os
import pathlib
import stat
import sys

import pytest

from ouroboros.tools.shell_guards import (
    _INTERPRETER_ANY_WRITE_RE,
    _python_write_targets_and_unknown,
    interpreter_family,
    interpreter_inline_code,
    light_shell_repo_mutation,
    shell_writer_targets_protected,
)


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


# ---- guard surface 1: the light-mode inline-write fence --------------------

# (unversioned, versioned, inline flag, write payload) — each payload carries a
# write form the fence models for that family.
FENCE_CASES = [
    ("python3", "python3.11", "-c", "open('probe.txt','w').write('x')"),
    ("ruby", "ruby3.2", "-e", "File.write('probe.txt', 'x')"),
    ("perl", "perl5.38", "-e", "unlink('probe.txt')"),
    ("php", "php8.3", "-r", "unlink('probe.txt');"),
    ("node", "node18", "-e", "fs.writeFileSync('probe.txt','x')"),
]


def test_versioned_spellings_engage_the_inline_write_fence_like_unversioned(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for unversioned, versioned, flag, code in FENCE_CASES:
        # The payload NAMES the repo, which is the owner-approved trigger for the
        # non-python families (a bare relative filename is not a repo-path spelling
        # and no longer blocks — see the disclosure in CHECKLISTS item 21).
        named = code.replace("probe.txt", f"{repo}/probe.txt")
        results = {
            spelling: light_shell_repo_mutation(
                [spelling, flag, named],
                repo_dir=repo, cwd=str(repo), work_dir=repo,
                detect_interpreter_inline=True,
            )
            for spelling in (unversioned, versioned)
        }
        # Parity is the regression: pre-fix the versioned spelling fell through
        # to False while its family name blocked.
        assert results[versioned] == results[unversioned] == True, results  # noqa: E712


def test_versioned_writer_spelling_engages_the_protected_path_guard():
    # ruby/perl are LIGHT_SHELL_WRITER_COMMANDS members; membership must
    # canonicalize the versioned spelling to the family name so a versioned
    # spelling touching a protected/frozen-contract file fires like the family.
    for exe in ("ruby", "ruby3.2", "perl", "perl5.38"):
        assert shell_writer_targets_protected(
            [exe, "rewrite.rb", "BIBLE.md"]
        ) is True, exe


# ---- guard surface 2: the registry runtime_data scan trigger ---------------


def _light_registry(tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    reg = ToolRegistry(repo_dir=repo, drive_root=tmp_path / "drive")
    reg._ctx.task_id = "t1"
    return reg


@pytest.mark.serial
def test_versioned_interpreter_runtime_data_write_is_registry_blocked(tmp_path, monkeypatch):
    """End to end through ToolRegistry.execute in light mode: a versioned
    NON-python interpreter whose inline code writes a runtime_data path outside
    the task's own roots must be refused before execution, exactly like the
    unversioned spelling. Host-independent: the guard refuses pre-exec, so the
    binary need not exist; where it does, the no-file assertion still holds.
    The payload's write call (`FileUtils.copyfile`) deliberately carries no
    shell-level write token, so pre-fix nothing triggered the scan at all."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _light_registry(tmp_path)
    target = tmp_path / "drive" / "state" / "probe.json"
    for exe in ("php8.3", "node18", "ruby3.2"):
        result = reg.execute("run_command", {
            "cmd": [exe, "-e", f"FileUtils.copyfile('src', '{target}')"],
        })
        assert "LIGHT_MODE_BLOCKED" in result, (exe, result[:300])
    assert not target.exists()


# ---- guard surface 3: protected-artifact high-risk interpreter check -------


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
# XG-7B3.1: REACHABILITY — the fence reaches both tool surfaces, and the
# inline-flag / write-primitive vocabularies are per-family and complete.
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
# located "body" — and is separated from actual code by its SHAPE, not by the flag.
COMPILE_CHECK_FORMS = [
    ["ruby", "-c", "script.rb"],
    ["perl", "-c", "script.pl"],
]


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


def test_a_compile_check_is_told_apart_from_code_by_shape(tmp_path):
    """The cost of accepting `-c` everywhere: `ruby -c file` locates a "body" that is a
    FILENAME. Shape — not the flag — decides whether the inline fence judges it, so a
    compile check is not treated as an unprovable write."""
    from ouroboros.tools.shell_guards import _carries_inline_code

    for argv in COMPILE_CHECK_FORMS:
        bodies = interpreter_inline_code(argv)
        assert bodies and not _carries_inline_code(argv, bodies), argv
    # Real code behind the same flag IS judged.
    code_argv = ["ruby", "-c", "File.write('a.py','x')"]
    assert _carries_inline_code(code_argv, interpreter_inline_code(code_argv)) is True


def test_php_carries_several_inline_bodies_at_once():
    # -B/-R/-E are three separate code arguments in one command.
    assert interpreter_inline_code(
        ["php8.3", "-B", "BEGIN", "-R", "EACH", "-E", "END"]
    ) == ["BEGIN", "EACH", "END"]


def test_unparseable_interpreter_code_is_treated_as_write_capable(tmp_path):
    """THE inversion (XG-7B3.1 r2). The first fix chased write vocabularies; that is
    whack-a-mole by construction, because no token list can prove the ABSENCE of a
    write in arbitrary interpreted code. Two holes survived the enumeration and are
    pinned here, alongside the shapes a vocabulary can never cover: an inline FLAG the
    table does not contain, a primitive no table lists, and code whose payload is
    computed at runtime.

    For PYTHON the fence asks whether it can PROVE the code cannot write into the
    repo and refuses when it cannot. For every other family the owner approved a
    NARROWER rule — refused only when the code names the repo by an ABSOLUTE path or
    a `./`/`../`-prefixed relative one, even for reading — so an unprovable non-python
    payload that does not name the repo in one of those spellings RUNS. A PLAIN
    relative spelling (`ouroboros/safety.py`) is not one of them: `EMBEDDED_RELATIVE_
    PATH_RE` anchors on `./`/`../`, so it is invisible to the scan. Both halves are
    pinned here, because the second one is a disclosed hole, not an oversight."""
    repo = tmp_path / "repo"
    repo.mkdir()
    unprovable_python = [
        ["python3.11", "-c", "exec(open('/dev/stdin').read())"],
        # Execution handed to another process: the AST models nothing past this.
        ["python3.11", "-c", "import subprocess;subprocess.run(['rm','x.py'])"],
        ["python3.11", "-c", "import os;os.system('rm x.py')"],
        ["python3.11", "-c", "eval(__import__('os').environ['C'])"],
    ]
    for argv in unprovable_python:
        assert light_shell_repo_mutation(
            argv, repo_dir=repo, cwd=str(repo), work_dir=repo,
        ) is True, argv

    # Non-python, NAMING the repo: refused, whatever the write spelling — this is
    # the half the vocabulary chase could never cover.
    unprovable_named = [
        # An inline flag the table does NOT contain, carrying a write.
        ["node18", "--experimental-foo", f"require('fs').writeFileSync('{repo}/a.py','x')"],
        ["ruby3.2", "--some-new-flag", f"File.write('{repo}/a.py','x')"],
        # A write primitive listed in no vocabulary anywhere.
        ["ruby3.2", "-e", f"IO.binwrite('{repo}/a.py','x')"],
        ["php8.3", "-r", f"$f=new SplFileObject('{repo}/a.py','w');"],
        # Nested quoting the parser cannot resolve.
        ["node18", "-e", f"require('fs')['write'+'FileSync']('{repo}/a.py','x')"],
        # Dynamic/computed target that still names the repo.
        ["perl5.38", "-e", f"open(F,'>',$ENV{{X}}.'{repo}/a.py');"],
    ]
    for argv in unprovable_named:
        assert light_shell_repo_mutation(
            argv, repo_dir=repo, cwd=str(repo), work_dir=repo,
        ) is True, argv

    # DISCLOSED HOLE, pinned so it cannot be re-tightened by accident: a non-python
    # payload the parser cannot read, naming no repo path, RUNS even with the cwd in
    # the repo. Refusing these refused every ordinary `node -e` in an ordinary chat,
    # which is more than the owner approved.
    disclosed_open = [
        ["node18", "-e", "eval(process.env.CODE)"],
        ["node18", "-e", "require('fs')['write'+'FileSync']('a.py','x')"],
        ["php8.3", "-r", "$f=new SplFileObject('a.py','w');"],
    ]
    for argv in disclosed_open:
        assert light_shell_repo_mutation(
            argv, repo_dir=repo, cwd=str(repo), work_dir=repo,
        ) is False, argv
    # ruby/perl are NOT in this list because they are LIGHT_SHELL_WRITER_COMMANDS
    # members and block on the earlier writer-command branch — public-head behaviour
    # this range never touched, not the inline fence.
    for argv in (["ruby3.2", "-e", "IO.binwrite('a.py','x')"],
                 ["perl5.38", "-e", "open(F,'>',$ENV{X});"]):
        assert light_shell_repo_mutation(
            argv, repo_dir=repo, cwd=str(repo), work_dir=repo,
        ) is True, argv


def test_a_script_or_module_invocation_is_not_judged_by_the_inline_fence(tmp_path):
    """The inversion is scoped to code handed IN THE ARGV. A script or module
    invocation hands a FILE, which this fence does not read, so `python -m pytest -q`
    and `node build.js` must still run with the cwd inside the repo — an early
    version of the inversion refused exactly those, which would have refused running
    the repo's own tooling in light mode."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "tool.py").write_text("print(1)\n")
    outside = tmp_path / "outside"
    outside.mkdir()

    for argv, wd in [
        (["python3.11", "-m", "pytest", "-q"], repo),
        (["python3.11", "tool.py"], repo),
        (["python3.11", str(repo / "tool.py")], outside),
        (["node", "build.js"], repo),
        (["node18", "-m", "something"], repo),
    ]:
        assert light_shell_repo_mutation(
            argv, repo_dir=repo, cwd=str(wd), work_dir=wd,
        ) is False, argv
    # But an in-place edit expressed inline IS inline code, and is refused.
    assert light_shell_repo_mutation(
        ["perl5.38", "-pi", "-e", "s/a/b/", "tool.py"],
        repo_dir=repo, cwd=str(repo), work_dir=repo,
    ) is True


def test_only_a_real_parse_earns_the_read_allowance(tmp_path):
    """The inversion must not become a blanket refusal: python is the one family whose
    code this module genuinely parses, and a fully-resolved AST with no write targets
    is a real proof of read-only. That proof is what preserves the v6.54.3 allowance
    (python scripts opening their own staged attachment), so it stays — and it stops
    being granted the moment the parse cannot account for the code."""
    repo = tmp_path / "repo"
    repo.mkdir()
    # Proven read-only, even with the cwd inside the repo.
    assert light_shell_repo_mutation(
        ["python3.11", "-c", "print(open('README.md').read())"],
        repo_dir=repo, cwd=str(repo), work_dir=repo,
    ) is False
    # Same shape, but execution leaves the parse -> no proof, so it blocks.
    assert light_shell_repo_mutation(
        ["python3.11", "-c", "print(eval(open('README.md').read()))"],
        repo_dir=repo, cwd=str(repo), work_dir=repo,
    ) is True


def test_the_fence_inspects_inline_code_by_default():
    """The reachability defect in one assertion: the parameter that decides whether
    inline code is read at all defaulted to OFF, and only run_script opted in."""
    default = inspect.signature(light_shell_repo_mutation).parameters["detect_interpreter_inline"].default
    assert default is True


def _shim(bin_dir: pathlib.Path, name: str, target: pathlib.Path) -> None:
    """A REAL executable standing in for the interpreter: if the guard lets the
    command run, this writes the target and the test fails on the file, not on a
    missing message."""
    path = bin_dir / name
    path.write_text(f"#!/bin/sh\nprintf MUTATED > '{target}'\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@pytest.mark.serial
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shim script")
def test_run_command_inline_repo_write_is_blocked_before_the_file_changes(tmp_path, monkeypatch):
    """THE regression for XG-7B3.1, driving the REAL run_command path per family.

    Pre-fix this asserted-on file came back 'MUTATED' for node/php (and python),
    because `run_command` disabled inline inspection: the write landed and only the
    post-execution tripwire spoke, which does not roll back."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _light_registry(tmp_path)
    repo = pathlib.Path(reg._ctx.repo_dir)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    for index, (exe, rest) in enumerate(INLINE_FLAG_PAYLOADS):
        target = repo / f"ordinary{index}.py"
        target.write_text("original\n")
        _shim(bin_dir, exe, target)
        cmd = [exe, *[part.format(p=str(target)) for part in rest]]
        result = reg.execute("run_command", {"cmd": cmd})
        # CONTAINMENT FIRST: the file must be UNCHANGED. The shim on PATH writes
        # "MUTATED" the moment the command is allowed to run, so this assertion —
        # not the message below — is what the finding is about: the post-execution
        # tripwire reports a mutation it cannot roll back.
        assert target.read_text() == "original\n", f"{cmd} MUTATED the repo file"
        assert "LIGHT_MODE_BLOCKED" in result, (cmd, result[:200])


@pytest.mark.serial
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shim script")
def test_run_script_keeps_the_same_fence(tmp_path, monkeypatch):
    """The surface that already had it must not regress while the other gains it."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _light_registry(tmp_path)
    repo = pathlib.Path(reg._ctx.repo_dir)
    target = repo / "via_script.py"
    target.write_text("original\n")
    result = reg.execute("run_script", {
        "script": f"open({str(target)!r},'w').write('MUTATED')",
    })
    assert "LIGHT_MODE_BLOCKED" in result, result[:200]
    assert target.read_text() == "original\n"


def test_legitimate_deliverable_writes_stay_allowed(tmp_path):
    """Reaching every surface must not turn the fence into a blanket refusal.

    A literal write OUTSIDE the repo is allowed for every family (this is what the
    per-family literal extractors buy), and a write whose path the scan cannot
    resolve is refused only where it could actually land in the repo — code that
    NAMES a repo path, or a cwd inside the repo. Without that distinction,
    extending inspection to run_command refused an ordinary user_files deliverable
    whose filename the code computes."""
    repo = tmp_path / "repo"
    repo.mkdir()
    out_dir = tmp_path / "drive" / "task_drives" / "t1"
    out_dir.mkdir(parents=True)
    out = out_dir / "deliverable.txt"

    allowed = [
        ["node", "-e", f"require('fs').writeFileSync('{out}','x')"],
        ["php8.3", "-r", f"file_put_contents('{out}','x');"],
        ["ruby3.2", "-e", f"File.write('{out}','x')"],
        ["perl5.38", "-e", f"open(F,'>','{out}');"],
        ["python3.11", "-c", f"open('{out}','w').write('x')"],
        # Computed filename, cwd outside the repo: cannot reach the repo.
        ["python3", "-c", "open(chr(100)+'eliverable.dat','w').write('x')"],
        ["node", "-e", "require('fs').writeFileSync(process.env.OUT,'x')"],
        # A python READ is provably a read, wherever it points. as_posix() keeps the
        # source valid python on Windows too (raw C:\Users would be a \U escape,
        # the parse would fail, and the fail-closed mention-scan would refuse).
        ["python3.11", "-c", f"print(open('{pathlib.PurePath(repo).as_posix()}/x.py').read())"],
    ]
    for argv in allowed:
        assert light_shell_repo_mutation(
            argv, repo_dir=repo, cwd=str(out_dir), work_dir=out_dir,
        ) is False, argv

    blocked = [
        # Dynamic target, but the code NAMES the repo.
        ["python3.11", "-c", f"import os,pathlib;p=pathlib.Path({str(repo)!r});(p/os.environ['N']).write_text('x')"],
        ["node", "-e", f"require('fs').writeFileSync(process.env.N.replace('X','{repo}'),'y')"],
    ]
    for argv in blocked:
        assert light_shell_repo_mutation(
            argv, repo_dir=repo, cwd=str(out_dir), work_dir=out_dir,
        ) is True, argv
    # Dynamic target with the cwd INSIDE the repo: for a NON-python family this now
    # RUNS. The resolved-cwd test is python-only, because the default shell cwd IS
    # the repository and applying it to every family refused ordinary node/ruby work
    # outright — more than the owner approved. Disclosed in CHECKLISTS item 21.
    assert light_shell_repo_mutation(
        ["node", "-e", "require('fs').writeFileSync(process.env.N,'y')"],
        repo_dir=repo, cwd=str(repo), work_dir=repo,
    ) is False
    # Python keeps the cwd test: an unresolvable target with the cwd in the repo is
    # still refused, which is what keeps run_script at or above the public head.
    assert light_shell_repo_mutation(
        ["python3", "-c", "import os;open(os.environ['N'],'w').write('y')"],
        repo_dir=repo, cwd=str(repo), work_dir=repo,
    ) is True


def test_the_price_of_the_inversion_is_named(tmp_path):
    """The narrowing the inverted default costs, pinned so it is a DECISION and not a
    surprise: a NON-python inline invocation that names the repo by an ABSOLUTE path
    (or a `./`/`../`-prefixed relative one) is refused in light mode even when it only
    reads, because nothing here can prove a regex-described read is a read. Python is
    unaffected (it is really parsed), the drive/user_files roots are unaffected,
    `read_file` remains the gated read path for every family, and advanced/pro mode is
    untouched — this fence only exists in light mode.

    The cost does NOT extend to a plain relative spelling: `EMBEDDED_RELATIVE_PATH_RE`
    anchors on `./`/`../`, so `node -e "…('pkg/mod.py')"` is invisible to the scan and
    runs. That is pinned below as a disclosed hole, not corrected — widening the regex
    would be a strengthening.

    If this assertion ever needs to flip, the honest fix is a real parser for that
    family, not another write-token list."""
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    # The price: an unprovable READ that names the repo blocks.
    assert light_shell_repo_mutation(
        ["node", "-e", f"require('fs').readFileSync('{repo}/x.py')"],
        repo_dir=repo, cwd=str(outside), work_dir=outside,
    ) is True
    # What it does NOT cost: the same read outside the repo, and python anywhere.
    assert light_shell_repo_mutation(
        ["node", "-e", f"require('fs').readFileSync('{outside}/x.py')"],
        repo_dir=repo, cwd=str(outside), work_dir=outside,
    ) is False
    assert light_shell_repo_mutation(
        ["python3.11", "-c",
         f"print(open('{pathlib.PurePath(repo).as_posix()}/x.py').read())"],
        repo_dir=repo, cwd=str(outside), work_dir=outside,
    ) is False  # as_posix(): raw backslashes are escapes in python source (see above)

    # DISCLOSED HOLE, pinned so the claim above stays honest: "names a repo path"
    # means ABSOLUTE or `./`/`../`-prefixed. `EMBEDDED_RELATIVE_PATH_RE` anchors on
    # those prefixes, so a PLAIN relative spelling is invisible to the scan and runs
    # even with the cwd in the repo — for a write as much as for a read. Every other
    # pin in this file happens to use an absolute path, which is why the generalized
    # wording went unmeasured. Widening the regex would be a strengthening; the text
    # was corrected instead.
    for code in ("require('fs').writeFileSync('pkg/mod.py','x')",
                 "require('fs').readFileSync('pkg/mod.py')"):
        assert light_shell_repo_mutation(
            ["node", "-e", code], repo_dir=repo, cwd=str(repo), work_dir=repo,
        ) is False, code
    # ...while the `./`-prefixed spelling of the SAME path is seen and refused.
    for code in ("require('fs').writeFileSync('./pkg/mod.py','x')",
                 "require('fs').readFileSync('./pkg/mod.py')"):
        assert light_shell_repo_mutation(
            ["node", "-e", code], repo_dir=repo, cwd=str(repo), work_dir=repo,
        ) is True, code


def test_a_windows_spelled_repo_path_is_a_repo_mention_on_every_platform(tmp_path, monkeypatch):
    """"An ABSOLUTE path" is a platform-neutral promise, and it was a POSIX-only one.

    Both mention scans in this fence — the non-python branch and the python
    unresolved-target branch — harvested `embedded_absolute_path_tokens` (which requires
    a leading `/`) plus the `./`-anchored relative regex, so `C:\\Users\\...\\repo` and a
    UNC share named nothing to either. On a Windows host the fence therefore engaged for
    NO non-python inline payload however it spelled the repo, and no python payload whose
    target the AST could not resolve. `runtime_data_write_targets` in the same module had
    always harvested all three spellings; the two scanners simply disagreed.

    Pinned on BOTH platforms, in the only two ways that ARE platform-neutral:

      * the SCAN. The Windows spellings must reach `repo_target_mentioned` — that is
        what the harvest does and what the pre-fix scan did not. Asserting a `True`
        VERDICT on a hardcoded foreign path (`C:\\Users\\runneradmin\\work\\repo`) instead
        would pass on POSIX for the WRONG reason — the token is not absolute there, so
        it lands under the tmp repo — and fail on Windows, where the same token is
        absolute and outside the tmp repo. That token is foreign on both platforms, so
        its verdict is `False` on both; only the harvest distinguishes the fix.
      * the VERDICT on the repo named in the HOST's own spelling, which is the shape a
        Windows CI run actually produces. There the string starts with a drive letter,
        so it is refused only because of this harvest; on POSIX the pre-existing POSIX
        harvest answers it.
    """
    import ouroboros.tools.shell_guards as shell_guards

    repo = tmp_path / "repo"
    repo.mkdir()
    windows_repo = r"C:\Users\runneradmin\work\repo"
    unc_repo = r"\\build-share\ouroboros\repo"
    scanned: list[str] = []
    real_mentioned = shell_guards.repo_target_mentioned

    def _spy(argv, **kwargs):
        scanned.extend(str(token) for token in argv)
        return real_mentioned(argv, **kwargs)

    monkeypatch.setattr(shell_guards, "repo_target_mentioned", _spy)

    # 1. THE SCAN. Non-python inline code naming a Windows path — drive letter or UNC
    #    share — hands that token to the containment check on every platform. The
    #    verdict is False because this particular root is foreign to the tmp repo
    #    everywhere, which is also the non-over-block half: a Windows path mentioned on
    #    a POSIX host is not a repo path and must not be treated as one.
    for spelling in (windows_repo, unc_repo):
        for code in (f"require('fs').writeFileSync('{spelling}\\\\a.py','x')",
                     f"require('fs').readFileSync('{spelling}\\\\a.py')"):
            scanned.clear()
            assert shell_guards.light_shell_repo_mutation(
                ["node18", "-e", code], repo_dir=repo, cwd=str(repo), work_dir=repo,
            ) is False, code
            assert any(token.startswith(spelling) for token in scanned), \
                f"the scan never saw the Windows spelling: {code} -> {scanned}"
    # The same for the python unresolved-target site, which is the second call site of
    # the one defect: no cwd to fall back on, so the decision rests on the scan alone.
    scanned.clear()
    dynamic = (f"import os,pathlib;p=pathlib.Path(r'{windows_repo}');"
               "(p/os.environ['N']).write_text('x')")
    assert shell_guards.light_shell_repo_mutation(
        ["python3.11", "-c", dynamic], repo_dir=repo, cwd="", work_dir=None,
    ) is False
    assert any(token.startswith("C:") for token in scanned), scanned

    # 2. THE VERDICT, on the repo spelled the way THIS host spells it — the CI shape.
    #    On Windows this is a drive-letter token and it is this harvest that refuses it.
    for code in (f"require('fs').writeFileSync('{repo}/probe.txt','x')",
                 f"require('fs').readFileSync('{repo}/probe.txt')"):
        assert shell_guards.light_shell_repo_mutation(
            ["node18", "-e", code], repo_dir=repo, cwd=str(repo), work_dir=repo,
        ) is True, code
    # The controls that make the lines above mean something: with the SAME cwd, a
    # payload naming no path at all still runs (the disclosed hole, unchanged).
    assert shell_guards.light_shell_repo_mutation(
        ["node18", "-e", "eval(process.env.CODE)"],
        repo_dir=repo, cwd=str(repo), work_dir=repo,
    ) is False
    assert shell_guards.light_shell_repo_mutation(
        ["python3.11", "-c", "import os;open(os.environ['N'],'w').write('x')"],
        repo_dir=repo, cwd="", work_dir=None,
    ) is False


def test_a_windows_spelled_token_is_judged_where_it_comes_from(tmp_path):
    """The containment half of the same fix, and the reason the harvest costs nothing.

    A `C:\\…` token is not absolute to POSIX `pathlib`, so joining it onto the work dir
    — which IS the system repository by default — made every Windows path merely
    MENTIONED in a payload read as repo-internal. Judged with Windows grammar instead,
    the answer is the true one on both hosts: inside a Windows-spelled repo root, and
    outside every other. Runnable anywhere: the root is passed in, so POSIX exercises
    exactly the arithmetic Windows performs natively.
    """
    windows_repo = pathlib.Path(r"C:\Users\runneradmin\work\repo")
    for code, verdict in (
        (r"require('fs').writeFileSync('C:\Users\runneradmin\work\repo\a.py','x')", True),
        (r"require('fs').writeFileSync('C:\Users\runneradmin\work\repo\\a.py','x')", True),
        (r"require('fs').writeFileSync('C:/Users/runneradmin/work/repo/a.py','x')", True),
        (r"require('fs').writeFileSync('C:\Users\runneradmin\work\other\a.py','x')", False),
        (r"require('fs').writeFileSync('D:\Users\runneradmin\work\repo\a.py','x')", False),
    ):
        assert light_shell_repo_mutation(
            ["node18", "-e", code],
            repo_dir=windows_repo, cwd=str(tmp_path), work_dir=tmp_path,
        ) is verdict, code


def test_a_backslash_idiom_is_not_a_windows_path(tmp_path):
    """The price the Windows harvest must NOT charge: POSIX capability.

    Harvesting Windows spellings arrived with a UNC alternative that matched any doubled
    backslash after a non-alphanumeric — i.e. the commonest escape idiom in every
    language that is not Python — and with a drive-letter token that POSIX `pathlib`
    joins onto the cwd. Measured together, four ordinary `node -e` one-liners that name
    no repo path flipped from allowed to REFUSED in light mode. A restriction has to
    earn its keep: this one bought nothing (none of these is a repo path anywhere) and
    cost working invocations, so the UNC alternative now requires a real `\\\\server\\share`
    and the drive-letter token is judged with Windows grammar.

    The last payload is the sharp one: it NAMES a Windows path, and is still allowed
    because that path is not the repo. Windows-native behaviour is the same for all
    four — `C:\\tmp\\out.txt` resolves outside the tmp repo, and the `/g` of the first
    payload joins to `C:\\g`, also outside.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    for code in (r"s.replace(/\\/g,'/')",
                 r'const s = "a\\\\b"; console.log(s)',
                 r"console.log('\\b')",
                 r"console.log('C:\\tmp\\out.txt')"):
        assert light_shell_repo_mutation(
            ["node", "-e", code], repo_dir=repo, cwd=str(repo), work_dir=repo,
        ) is False, code


def test_the_template_literal_exact_root_hole_is_disclosed_not_fenced(tmp_path):
    """DELIBERATELY REVERSED (2026-08-05): this test used to pin a backtick DELIMITER that
    closed the hole below; the delimiter was removed and the hole is now pinned as a
    DISCLOSED residual instead.

    The hole: JavaScript's template literal is the everyday way to spell a Windows path
    without doubling separators — ``String.raw`C:\\repo` `` — and neither harvest
    treats the closing backtick as a delimiter, so the token is the root plus a trailing
    backtick: one character past the real root, a SIBLING of it. Light mode therefore
    ALLOWS ``rmSync(String.raw`<root>`, {recursive:true})`` — deleting the EXACT repo
    root. Only the root itself escapes: any path UNDER it still carries a separator
    before the stray backtick and resolves inside, so the fence still refuses it.

    Why disclosed rather than fenced: the round-3 delimiter fix closed this but (a) it
    was found by adversarial review, not by the Windows failures this commit exists to
    fix — an adjacent hole, which owner policy says to disclose, not fence («не уходить
    в погоню за edge cases»); and (b) it NARROWED capability the base had: a write to a
    real sibling whose name contains a literal backtick was newly refused («Защиту можно
    ослаблять но не усиливать»). Both grammars behave identically, which is the parity
    the Windows work was asked to deliver. A future change that turns the backtick back
    into a delimiter must fail this pin and re-open that deliberation with the owner.
    """
    from ouroboros.shell_parse import (
        EMBEDDED_WINDOWS_ABSOLUTE_PATH_RE,
        embedded_absolute_path_tokens,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    windows_repo = pathlib.Path(r"C:\Users\runneradmin\work\repo")

    def _rm_root(root: str) -> str:
        return 'require("fs").rmSync(String.raw`' + root + '`, {recursive:true})'

    # 1. THE HARVEST: the backtick is an ordinary path byte, so the token is the root
    #    plus the closing backtick — a sibling name, not the root. Each grammar is
    #    probed on a synthetic literal of its own spelling (NOT the host's tmp path,
    #    whose shape follows the host and starved the POSIX regex on Windows).
    posix_repo = "/srv/host/repo"
    win_tokens = EMBEDDED_WINDOWS_ABSOLUTE_PATH_RE.findall(_rm_root(str(windows_repo)))
    posix_tokens = embedded_absolute_path_tokens(_rm_root(posix_repo))
    assert win_tokens == [str(windows_repo) + "`"], win_tokens
    assert posix_tokens == [posix_repo + "`"], posix_tokens

    # 2. THE DISCLOSED HOLE: deleting the exact root through a template literal is
    #    ALLOWED in both spellings, because the harvested token reads as a sibling.
    assert light_shell_repo_mutation(
        ["node18", "-e", _rm_root(str(windows_repo))],
        repo_dir=windows_repo, cwd=str(tmp_path), work_dir=tmp_path,
    ) is False
    assert light_shell_repo_mutation(
        ["node18", "-e", _rm_root(str(repo))],
        repo_dir=repo, cwd=str(tmp_path), work_dir=tmp_path,
    ) is False

    # 3. THE FENCE STILL ENGAGES under the root: a template-literal write INSIDE the
    #    repo keeps its separator before the stray backtick and is refused, both
    #    spellings — the hole is exactly one path wide.
    assert light_shell_repo_mutation(
        ["node18", "-e",
         'require("fs").writeFileSync(String.raw`' + str(repo) + '/a.py`, "x")'],
        repo_dir=repo, cwd=str(tmp_path), work_dir=tmp_path,
    ) is True
    assert light_shell_repo_mutation(
        ["node18", "-e",
         'require("fs").writeFileSync(String.raw`' + str(windows_repo) + '\\a.py`, "x")'],
        repo_dir=windows_repo, cwd=str(tmp_path), work_dir=tmp_path,
    ) is True

    # 4. THE RESTORED CAPABILITY: a real sibling path with a literal backtick in its
    #    name runs, as it did at base — the delimiter refused it.
    assert light_shell_repo_mutation(
        ["node18", "-e",
         'require("fs").writeFileSync("' + str(tmp_path / "repo`backup" / "a") + '", "x")'],
        repo_dir=repo, cwd=str(tmp_path), work_dir=tmp_path,
    ) is False


# ---------------------------------------------------------------------------
# ONE write vocabulary: the AST walker knows everything the regex calls a write.
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


@pytest.mark.parametrize("label, template", _ORDINARY_STDLIB_WRITES,
                         ids=[label for label, _ in _ORDINARY_STDLIB_WRITES])
def test_an_ordinary_stdlib_write_is_never_read_as_a_proven_read(tmp_path, label, template):
    """Nine ordinary stdlib write APIs parsed to a fully-resolved AST with ZERO write
    targets, so `targets == [] and not unknown` — the callers' PROOF of read-only —
    was handed out for a payload that writes. `_INTERPRETER_ANY_WRITE_RE`, in the
    SAME module, recognised all nine: two write vocabularies, and the weaker one
    signed the proof. The fence's claim marked this failure class closed, so the
    next reviewer would not have come back to it (BIBLE P2).
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    code = template.format(t=repr(str(repo / "victim.txt")))

    # The regex half of the vocabulary has always seen these as writes...
    assert _INTERPRETER_ANY_WRITE_RE.search(code), label
    # ...and now the AST half agrees: it either resolves the target or says UNKNOWN.
    targets, unknown = _python_write_targets_and_unknown(code)
    assert targets or unknown, f"{label} parsed as a proven read"
    # End to end, with the cwd OUTSIDE the repo so only the target can block it.
    assert light_shell_repo_mutation(
        ["python3.11", "-c", code],
        repo_dir=repo, cwd=str(tmp_path), work_dir=tmp_path,
    ) is True, label


def test_the_same_stdlib_writes_still_run_when_they_target_elsewhere(tmp_path):
    """The vocabulary repair must not become a blanket refusal: a resolved
    destination OUTSIDE the repo is a proven-elsewhere write and still runs."""
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "elsewhere"
    for label, template in _ORDINARY_STDLIB_WRITES:
        code = template.format(t=repr(str(outside / "file.txt")))
        assert light_shell_repo_mutation(
            ["python3.11", "-c", code],
            repo_dir=repo, cwd=str(tmp_path), work_dir=tmp_path,
        ) is False, label


def test_a_write_through_an_untraceable_handle_is_unknown_not_a_proven_read(tmp_path):
    """The same defect one spelling over: `.write` on a receiver the walker cannot
    trace to a path contributed no target AND no unknown, so an unlocatable write
    read as proof of a read. Locatable handles must keep resolving, or the repair
    would just turn every deliverable write into a refusal."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for code in (
        'f = get_handle(); f.write("x")',
        'import tempfile; f = tempfile.NamedTemporaryFile(); f.write(b"x")',
    ):
        targets, unknown = _python_write_targets_and_unknown(code)
        assert unknown and not targets, code

    # ...while the shapes it CAN trace still resolve, so an ordinary out-of-repo
    # deliverable keeps running even with the cwd inside the repo.
    out = repr(str(tmp_path / "out.txt"))
    for code in (
        f'open({out}, "w").write("x")',
        f'import pathlib; pathlib.Path({out}).open("w").write("x")',
        f'f = open({out}, "w"); f.write("x")',
        f'import json; f = open({out}, "w"); json.dump({{}}, f)',
    ):
        targets, unknown = _python_write_targets_and_unknown(code)
        assert targets and not unknown, code
        assert light_shell_repo_mutation(
            ["python3.11", "-c", code], repo_dir=repo, cwd=str(repo), work_dir=repo,
        ) is False, code


def test_the_vocabulary_repair_does_not_widen_past_the_regex(tmp_path):
    """`.copy()` on a dict and `dump()` on an arbitrary object are NOT writes to the
    module's regex, so the walker must not call them write-capable either — a wider
    walker would refuse ordinary read-only payloads for a name collision."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for code in ("d = {}; e = d.copy()", "import yaml; print(yaml.dump({}))", "logger.dump()"):
        assert not _INTERPRETER_ANY_WRITE_RE.search(code), code
        assert _python_write_targets_and_unknown(code) == ([], False), code
        assert light_shell_repo_mutation(
            ["python3.11", "-c", code], repo_dir=repo, cwd=str(repo), work_dir=repo,
        ) is False, code


def test_builtin_replace_and_remove_are_not_filesystem_writes(tmp_path):
    """`.replace`/`.rename`/`.remove` were credited as writes on ANY receiver.

    `.replace` is the most common string method in Python, so in light mode an
    ordinary text one-liner was refused with a security message telling the agent
    to move its cwd — advice it cannot connect to the cause. There is no such
    thing as a repo write via `'a,b'.replace`, so this was a pure over-block with
    no security gain. Same receiver discrimination the module already applies to
    `shutil.copy` vs `some_dict.copy()`.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    for code in (
        "print('a,b'.replace(',', ';'))",
        "d = [1, 2]; d.remove(1); print(d)",
        "print('a-b'.replace('-', '_').upper())",
    ):
        assert _python_write_targets_and_unknown(code) == ([], False), code
        assert light_shell_repo_mutation(
            ["python3.11", "-c", code], repo_dir=repo, cwd=str(repo), work_dir=repo,
        ) is False, code

    # ...and the GENUINE path writes those same spellings also name are still refused.
    # The walker resolves the receiver-is-a-path forms to a literal target; the
    # `os.replace`/`os.rename` forms read their receiver (`os`) and answer UNKNOWN,
    # which the fence resolves through `_dynamic_write_could_hit_repo`. Either way the
    # one invariant that matters is that NEITHER is ever a proven read.
    for code in (
        "import pathlib; pathlib.Path('pkg/mod.py').replace('other.py')",
        "import pathlib; pathlib.Path('pkg/mod.py').rename('other.py')",
        "import os; os.replace('pkg/mod.py', 'other.py')",
        "import os; os.rename('pkg/mod.py', 'other.py')",
        "import os; os.remove('pkg/mod.py')",
    ):
        targets, unknown = _python_write_targets_and_unknown(code)
        assert targets or unknown, code  # never `([], False)` — a proven read
        assert light_shell_repo_mutation(
            ["python3.11", "-c", code], repo_dir=repo, cwd=str(repo), work_dir=repo,
        ) is True, code
