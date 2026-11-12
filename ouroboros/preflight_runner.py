"""Hermetic pytest preflight for reviewed repository changes."""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from typing import NamedTuple, Optional, Sequence

from ouroboros.preflight_probe import (  # the gate's plugin in the candidate run and the admission beside it: its public contract
    MIN_PREFLIGHT_WORKERS, WORKER_PROBE_MODULE, allocated_worker_count, install_worker_probe, observed_worker_ids,
    probe_dir, recorded_admission, recorded_session,
)

# Node lane of the gate (`run_hermetic_pytest` calls it once the candidate is
# assembled, as the first consumer of the shared budget): content-keyed on the
# candidate's web/tests/*.test.js — see ouroboros/preflight_node.py. Imported
# by name so tests/operators can stub `preflight_runner.run_node_tests`.
from ouroboros.preflight_node import run_node_tests
from ouroboros.test_environment import gate_root


DEFAULT_PYTEST_ARGS = ["tests/", "-q", "--tb=line", "--no-header"]

# SSOT for the costly marker lanes that never belong in a local/gate run. It is
# byte-identical to the ``-m`` expression in pyproject.toml ``addopts`` and is
# the tail of BOTH ci.yml markexprs. A command-line ``-m`` REPLACES the addopts
# ``-m`` entirely, so every pass below must re-state this conjunction or the
# excluded lanes silently come back. Kept as a constant (not a runtime
# tomllib parse): README pins Python 3.10+ and tomllib is 3.11+.
LANE_EXCLUSION_EXPR = (
    "not integration and not browser and not ui_browser and not ui_browser_docker "
    "and not portable_detail and not skill_smoke and not size_ratchet"
)

# Shared flags for ci.yml `quick-test`/`full-test-runs`, including the hang timeout.
# CI appends `--default-suite-shard` on every OS; the local full run is unsharded.
# No `--timeout-method`: pytest-timeout's default is SIGALRM where the platform has
# it, which fails only the hung test, and its thread method elsewhere (Windows),
# which stops the worker. `--dist loadgroup`: the one xdist pass in which a
# `serial("<resource>")` test takes turns per resource beside everything else
# (tests/conftest.py pins the groups, longest first by recorded time).
PARALLEL_PASS_FLAGS = [
    "-n", "auto",
    "--dist", "loadgroup",
    "--max-worker-restart=0",
    "--timeout=300",
]
# The serial pass (`serial("exclusive")` alone, one process) carries the same per-test hang bound and none of
# the xdist flags.
SERIAL_PASS_FLAGS = ["--timeout=300"]
# The marker that splits the two passes: tests/conftest.py derives it from `serial("exclusive")`.
EXCLUSIVE_MARKER = "exclusive"

# Verifying the INTERPRETER (see `_verify_preflight_plugins`) proves the plugins
# are installed; it does not prove the candidate's own pytest configuration lets
# them LOAD. A candidate `pytest.ini`/`pyproject.toml` `addopts` carrying
# `-p no:xdist -p no:timeout` — together with a conftest that declares
# `-n`/`--dist`/`--timeout` via `pytest_addoption` and ignores them — would
# otherwise run the nominal parallel lane serially and exit 0. So the parallel
# pass loads them explicitly.
#
# ini `addopts` are PREPENDED to the command line, so these entries are the LAST
# `-p` pytest processes, and `consider_pluginarg` unblocks exactly the names a
# `no:` entry blocked ("xdist" also unblocks "pytest_xdist", "timeout" also
# unblocks "pytest_timeout"). ENTRY-POINT names, not module paths, on purpose:
# entry-point autoload skips a name that is already registered, so forcing
# `xdist` cannot double-register the same module under a second name — which
# `-p xdist.plugin` would, turning an ordinary green run into a pluggy
# "Plugin already registered" error.
#
# NOT part of `PARALLEL_PASS_FLAGS`: that constant is the CI mirror and is
# compared token-for-token against ci.yml, where the environment is provisioned
# and no candidate configuration is in play.
_FORCED_PLUGIN_FLAGS = ["-p", "xdist", "-p", "timeout"]
# The serial pass carries `--timeout` too, so it forces the timeout plugin the same way.
_FORCED_SERIAL_PLUGIN_FLAGS = ["-p", "timeout"]

# Option NAMES that only a parallel pass carries, derived from the SSOT above so
# the classifier can never disagree with the flags actually passed. Compared as
# WHOLE tokens: a substring test would match "-n" inside "--no-header", which
# DEFAULT_PYTEST_ARGS passes on every single invocation.
_PARALLEL_FLAG_NAMES = frozenset(
    flag.split("=", 1)[0] for flag in PARALLEL_PASS_FLAGS if flag.startswith("-")
)

# Flags that mean "xdist is distributing this run". Read off the argv rather
# than the pass label so a caller-supplied `pytest_args` carrying its own `-n`
# still gets the xdist-specific diagnoses.
_XDIST_ARGV_FLAGS = frozenset({"-n", "--numprocesses", "--dist"})

# xdist also accepts pytest's combined short-option form as ONE token (`-n4`,
# `-nauto`, `-nlogical`). The gate's own passes always use the separated form,
# but a caller-supplied argv using the combined one really does hand the run to
# xdist, so it must reach the xdist-specific diagnoses too. `--no-header` (in
# DEFAULT_PYTEST_ARGS, on EVERY invocation) cannot match: its second character
# is `-`, not `n`.
_XDIST_COMBINED_NUMPROCESSES = re.compile(r"^-n\S+$")

# pytest's EXIT_NOTESTSCOLLECTED. Green-empty per pass: a candidate repo may
# legitimately have zero serial tests. Only ALL passes empty is a block.
_PYTEST_EXIT_NO_TESTS = 5
# pytest's EXIT_USAGEERROR — how a missing xdist/timeout plugin surfaces.
_PYTEST_EXIT_USAGE_ERROR = 4

# Terminal decoration (SGR colours, cursor moves) that xdist/pytest may wrap a
# reported line in. Stripped before matching so a coloured controller line is not
# a miss and a decorated assertion line is not a surprise match.
_ANSI_DECORATION = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _strip_decoration(line: str) -> str:
    return _ANSI_DECORATION.sub("", line).strip()


# xdist signatures for a worker that DIED rather than reported a failure. A
# crashed worker is a hard block, never a retryable flake.
#
# Both error directions are handled, and they are NOT symmetric:
#   * a MISS only degrades the label — the nonzero exit still blocks;
#   * a FALSE POSITIVE would be lossy if the matched lines replaced the report,
#     so they are only ever a highlighted PREFIX in front of the full pytest
#     output (see ``_classify_pass_result``). A mislabelled ordinary failure
#     therefore still costs the reader nothing but a wrong remediation line.
#
# Each pattern matches a COMPLETE xdist controller line shape — the full phrase
# together with the worker id (or the numeric operand) that xdist itself always
# prints — never a free substring of it. Bare `node down:`, `crashed while
# running`, `replacing crashed worker` and `maximum crashed workers reached` all
# occur in ordinary assertion text and captured logs of tests that reason about
# worker pools, and labelling those `PARALLEL_WORKER_CRASH` handed the author a
# mark-it-serial instruction for a failure the marker cannot fix.
#
# Shapes, from xdist/dsession.py: `handle_crashitem` -> "worker 'gwN' crashed
# while running 'nodeid'"; `TerminalDistReporter.pytest_testnodedown` -> "[gwN]
# node down: ..."; and the restart branch, which under our
# `--max-worker-restart=0` says "worker gwN crashed and worker restarting
# disabled".
#
# Every pattern is anchored to the start of the DECORATION-STRIPPED line
# (`_strip_decoration` removes ANSI and surrounding whitespace before matching),
# because xdist emits all of these as whole lines of its own. Without the anchor
# the same phrases matched anywhere they appear — assertion text, captured logs,
# and the diff of this very module — so a candidate whose ordinary test failure
# merely MENTIONS a crashed worker was labelled `PARALLEL_WORKER_CRASH` and
# handed a mark-it-serial instruction for a failure the marker cannot fix.
#
# `handle_crashitem` reports the crash as a TestReport longrepr, so under `-q`
# pytest ALSO re-emits it mid-line in the short summary. That form is matched by
# its own anchored pattern rather than by unanchoring the phrase: only pytest's
# own `FAILED`/`ERROR` line prefix opens the line, which candidate text does not.
_WORKER_CRASH_PATTERNS = (
    re.compile(r"^\[gw\d+\] node down:", re.IGNORECASE),
    re.compile(r"^worker '?gw\d+'? crashed while running\b", re.IGNORECASE),
    re.compile(r"^(?:FAILED|ERROR) \S+ - worker '?gw\d+'? crashed while running\b", re.IGNORECASE),
    re.compile(r"^worker '?gw\d+'? crashed and worker restarting disabled", re.IGNORECASE),
    re.compile(r"^replacing crashed worker '?gw\d+", re.IGNORECASE),
    re.compile(r"^maximum crashed workers reached: \d+", re.IGNORECASE),
)

# Why a worker died. pytest-timeout's thread method (the default where SIGALRM does
# not exist, i.e. Windows) does NOT fail the test: it dumps stacks and `os._exit(1)`s
# the whole worker, which xdist reports with the crash phrasing above, and the serial
# mark would only move that hang (the serial pass carries the same per-test timeout).
# Its banner and stacks go to the worker's stdout, which execnet discards (#1949); a
# signal-method `Failed: Timeout` fails its test while the worker lives on. So the cause
# comes from this pass's own results.json (tests/ci_evidence.py), which the probe in this
# pass's controller locates: a crash row names its worker and, only when the controller
# matched the record that worker's callback left for exactly that test of this run, the
# timeout. Printed text is no authority, since any test can print it; this pattern keeps
# the controller's readable copy of such a row in the display only.
_TIMEOUT_CALLBACK_LINE = re.compile(r"^\[gw\d+\] pytest-timeout callback invoked\b")


class PreflightPass(NamedTuple):
    """One labeled pytest invocation inside the shared hermetic worktree.

    ``parallel`` records whether this argv actually hands the run to xdist. Both
    hard-block diagnoses below describe xdist-specific failures, so they must
    only be reachable for a pass that really carried those flags.
    """

    label: str
    args: list[str]
    parallel: bool


def _make_pass(label: str, args: list[str]) -> PreflightPass:
    parallel = any(
        arg.split("=", 1)[0] in _XDIST_ARGV_FLAGS
        or _XDIST_COMBINED_NUMPROCESSES.match(arg)
        for arg in args
    )
    return PreflightPass(label, args, parallel)


def _serial_escape_hatch_enabled() -> bool:
    """`OUROBOROS_PREFLIGHT_SERIAL=1` forces the legacy single serial pass.

    Operator rollback lever for the two-pass gate. The value never reaches the
    candidate suite: `_preflight_env` scrubs every `OUROBOROS_*` key.
    """
    raw = os.environ.get("OUROBOROS_PREFLIGHT_SERIAL", "")
    return raw.strip().lower() in {"1", "true", "yes"}


def _preflight_pass_specs(
    pytest_args: Optional[Sequence[str]] = None,
    *,
    probe_module: str = "",
) -> list[PreflightPass]:
    """The gate mirrors CI: one xdist pass of everything but `serial("exclusive")`, then those alone.

    An explicit ``pytest_args=`` (or the escape hatch) collapses to the single
    legacy pass, forwarding that argv unchanged, so callers that pin their own
    argv keep running exactly the tests they asked for. (The VERDICTS are also
    unchanged, but the failure TEXT now carries this pass's header — see
    ``_classify_pass_result``.)

    The test is ``is not None``, not truthiness, so the two-pass default is
    reserved for "no argv supplied": silently upgrading an explicit argv to the
    two-pass suite would run tests the caller never asked for under xdist
    requirements they never opted into. An explicitly EMPTY sequence is the one
    exception — it selects ``DEFAULT_PYTEST_ARGS`` exactly as the pre-two-pass
    runner's truthiness test did, because forwarding a bare ``pytest`` with no
    argv at all changes the discovery target and the output flags rather than
    preserving any caller contract.

    ``probe_module`` is the NONCE name ``install_worker_probe`` actually wrote.
    It must be threaded through: ``-p name`` is an ordinary import, so a parallel
    pass built against the bare stem loads a module no file is named after and
    pytest exits ``EXIT_USAGEERROR`` before collecting a single test. The stem is
    the fallback purely so inspecting the spec shape without installing a probe
    never emits an empty ``-p``.
    """
    if pytest_args is not None:
        explicit = list(pytest_args)
        return [_make_pass("single", explicit or list(DEFAULT_PYTEST_ARGS))]
    if _serial_escape_hatch_enabled():
        return [_make_pass("single", list(DEFAULT_PYTEST_ARGS))]
    target, *output_flags = DEFAULT_PYTEST_ARGS
    return [
        _make_pass(
            "parallel",
            [
                target, "-m", f"not {EXCLUSIVE_MARKER} and {LANE_EXCLUSION_EXPR}",
                *PARALLEL_PASS_FLAGS,
                # Neither of these mirrors CI: they defend the parallel lane
                # against candidate-controlled pytest configuration, which CI
                # does not have. `-p xdist -p timeout` overrides a `no:` block in
                # the candidate's addopts; the probe module reports the worker
                # ids the run actually started. An explicit `pytest_args=` gets
                # neither — the caller's argv is forwarded verbatim, and the
                # worker-count check keys on the probe flag being present.
                *_FORCED_PLUGIN_FLAGS,
                "-p", probe_module or WORKER_PROBE_MODULE,
                *output_flags,
            ],
        ),
        _make_pass(
            "serial",
            [target, "-m", f"{EXCLUSIVE_MARKER} and {LANE_EXCLUSION_EXPR}", *SERIAL_PASS_FLAGS,
             *_FORCED_SERIAL_PLUGIN_FLAGS, "-p", probe_module or WORKER_PROBE_MODULE, *output_flags],
        ),
    ]


def _run_git(
    repo_dir: pathlib.Path,
    args: Sequence[str],
    *,
    input_text: "str | bytes" = "",
    timeout: int = 30,
    binary_stdout: bool = False,
) -> subprocess.CompletedProcess:
    # BINARY pipes, decoded here so callers keep the str contract: text-mode pipes
    # translate \n to os.linesep, and on Windows a CRLF-mangled stdin corrupts a
    # multi-line git payload (a replayed diff's context lines stop matching the
    # LF worktree, so `git apply` rejects the candidate diff wholesale).
    #
    # ``binary_stdout`` skips the decode for the ONE payload that must survive
    # byte-for-byte — the candidate capture that is fed straight back into
    # `git apply`. Git classifies NUL-free non-UTF-8 content (latin-1 logs,
    # cp1251 fixtures) as TEXT, so its bytes travel on plain diff lines, and
    # decode(errors="replace") would substitute U+FFFD for each of them —
    # silently corrupting the candidate while the gate stays green.
    # ``input_text`` symmetrically accepts those captured bytes unmodified.
    payload = input_text.encode("utf-8") if isinstance(input_text, str) else input_text
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo_dir),
        input=payload or None,
        capture_output=True,
        timeout=timeout,
    )
    return subprocess.CompletedProcess(
        proc.args,
        proc.returncode,
        (proc.stdout or b"") if binary_stdout else (proc.stdout or b"").decode("utf-8", "replace"),
        (proc.stderr or b"").decode("utf-8", "replace"),
    )


# The refs consulted for "did this repository have a test suite?" — and they are
# NOT the same for both entry points, because the two run on opposite sides of
# the commit.
#
# POST-commit (`tools/git.py::_run_pre_push_tests`): `_post_commit_result` is
# reached only once `commit_sha` exists, which is why its failure text says "the
# commit was already created and preserved". By then a whole-suite deletion is
# already IN HEAD, so a HEAD-only check answers False and the hard block below is
# unreachable from the entry point that most needs it. Hence HEAD~1. Only the
# IMMEDIATELY preceding commit is consulted, so a suite deliberately removed
# several commits ago stays out of scope rather than blocking forever.
#
# PRE-commit (`review_helpers::_run_review_preflight_tests`): the candidate is
# still a working-tree change, so HEAD alone already answers the question — and
# consulting HEAD~1 there is actively WRONG. Immediately after a deliberate
# removal commit, HEAD legitimately carries no suite while HEAD~1 still does, so
# an `any()` over both rejected the next unrelated staged change as "removes the
# entire tests/ tree". The one-commit horizon does expire, but only once that
# next commit exists — which is after the pre-commit gate has already refused it.
_TESTS_BASELINE_REFS = ("HEAD", "HEAD~1")
_PRE_COMMIT_BASELINE_REFS = ("HEAD",)

# Phase label the pre-commit review passes; anything else keeps the wider
# post-commit baseline, so an unrecognised value fails toward MORE blocking.
PRE_COMMIT_PHASE = "pre_commit"


def _baseline_refs(phase: str) -> tuple:
    return _PRE_COMMIT_BASELINE_REFS if phase == PRE_COMMIT_PHASE else _TESTS_BASELINE_REFS


def _baseline_commit_oids(
    repo: pathlib.Path, refs: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    """Resolve the phase baselines without mistaking unreadable history for absence."""
    # Deliberately do not peel here: rev-parse returns the ref's recorded OID
    # even when the commit object itself is missing. A quiet rc=1 with no output
    # can mean either an unborn HEAD or a broken ref, so only a successfully read
    # symbolic HEAD proves the former; every other failure is an operational
    # error the caller must hard-block.
    resolved = _run_git(repo, ["rev-parse", "--verify", "--quiet", "HEAD"])
    if resolved.returncode != 0:
        if (
            resolved.returncode == 1
            and not (resolved.stdout or "").strip()
            and not (resolved.stderr or "").strip()
        ):
            symbolic = _run_git(repo, ["symbolic-ref", "--quiet", "HEAD"])
            if symbolic.returncode == 0 and (symbolic.stdout or "").strip():
                return ()
            detail = (
                symbolic.stderr.strip()
                or symbolic.stdout.strip()
                or f"exit {symbolic.returncode}"
            )
            raise RuntimeError(f"git could not prove HEAD is unborn: {detail}")
        detail = (
            resolved.stderr.strip()
            or resolved.stdout.strip()
            or f"exit {resolved.returncode}"
        )
        raise RuntimeError(f"git could not resolve HEAD: {detail}")

    head_oid = (resolved.stdout or "").strip()
    head_commit = _run_git(repo, ["cat-file", "commit", head_oid])
    if head_commit.returncode != 0:
        detail = (
            head_commit.stderr.strip()
            or head_commit.stdout.strip()
            or f"exit {head_commit.returncode}"
        )
        raise RuntimeError(f"git could not read HEAD commit {head_oid}: {detail}")

    headers = (head_commit.stdout or "").partition("\n\n")[0].splitlines()
    first_parent = next(
        (line.removeprefix("parent ") for line in headers if line.startswith("parent ")),
        None,
    )
    by_ref = {"HEAD": head_oid}
    if first_parent is not None:
        by_ref["HEAD~1"] = first_parent
    return tuple((ref, by_ref[ref]) for ref in refs if ref in by_ref)


def _head_tracks_tests(repo: pathlib.Path, refs: Sequence[str] = _TESTS_BASELINE_REFS) -> bool:
    """Whether the committed history already carries a ``tests/`` directory.

    Distinguishes "this repository has no test suite" (out of scope for the
    gate) from "this candidate deleted the test suite" (a hard block). A
    repository with no commits at all answers False. ``refs`` is the caller's
    phase baseline — see above; it defaults to the post-commit pair, which is the
    conservative direction for a caller that does not say.
    """
    for ref, oid in _baseline_commit_oids(repo, refs):
        listed = _run_git(repo, ["ls-tree", "-r", "--name-only", oid, "--", "tests"])
        if listed.returncode != 0:
            detail = (
                listed.stderr.strip()
                or listed.stdout.strip()
                or f"exit {listed.returncode}"
            )
            raise RuntimeError(
                f"git could not read tests/ from baseline {ref} ({oid}): {detail}"
            )
        if (listed.stdout or "").strip():
            return True
    return False


def _apply_diff(worktree: pathlib.Path, diff_text: "str | bytes") -> None:
    # Accepts bytes so the candidate capture reaches `git apply` undecoded —
    # see ``binary_stdout`` on `_run_git` for why the round-trip must not
    # pass through UTF-8.
    #
    # `--unidiff-zero`: the capture's flag tail pins away every operator config
    # that reshapes diff CONTENT, but hunk WIDTH still leaks through — a user
    # `diff.context=0` (or `GIT_DIFF_OPTS=--unified=0` in the environment)
    # makes `git diff` emit zero-context hunks, which `git apply` REJECTS by
    # default, hard-blocking an ordinary textual candidate before any test
    # runs. The flag accepts zero-context hunks and is a no-op for hunks that
    # carry context, so it covers both the config and the env route without
    # scrubbing either.
    if not diff_text.strip():
        return
    proc = _run_git(
        worktree,
        # `-c core.autocrlf=false -c core.eol=lf`: the candidate capture is
        # byte-exact (`--binary`, no textconv), so the apply must not re-run
        # end-of-line conversion either. On a Windows runner (`core.autocrlf=true`
        # by default) an LF payload applied against a CRLF-converted checkout
        # otherwise mangles every line ending, breaking the byte-faithful
        # guarantee this whole capture exists to hold. Where `.gitattributes`
        # names a line ending (this repository's `* text=auto eol=lf`), the
        # attribute decides and agrees; elsewhere this override is authoritative.
        ["-c", "core.autocrlf=false", "-c", "core.eol=lf",
         "apply", "--whitespace=nowarn", "--binary", "--unidiff-zero"],
        input_text=diff_text,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git apply failed")


def _copy_untracked(repo_dir: pathlib.Path, worktree: pathlib.Path) -> None:
    # The NAMES arrive as bytes and are decoded with the filesystem's own codec
    # (surrogateescape on POSIX), not utf-8/replace: a filename carrying a raw
    # non-UTF-8 byte would otherwise become a U+FFFD name that no longer exists on
    # disk, `is_file()` would answer False, and the file would drop out of the
    # candidate silently — an inexact candidate with no assembly failure raised.
    listed = _run_git(
        repo_dir, ["ls-files", "--others", "--exclude-standard", "-z"], binary_stdout=True
    )
    if listed.returncode != 0:
        raise RuntimeError(listed.stderr.strip() or "git ls-files failed")
    for rel in [os.fsdecode(part) for part in (listed.stdout or b"").split(b"\0") if part]:
        src = (repo_dir / rel).resolve()
        dst = (worktree / rel).resolve()
        try:
            dst.relative_to(worktree.resolve())
            src.relative_to(repo_dir.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Unsafe untracked path: {rel}") from exc
        if not src.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _preflight_env(temp_root: pathlib.Path, repo_worktree: pathlib.Path, *, create=True, cwd=None,
                   workers: Optional[int] = None) -> dict:
    from ouroboros.test_environment import (
        DIAGNOSTIC_OUTPUT_KEYS, isolated_environment, retain_diagnostic_outputs, without_launch_descriptors,
    )
    from ouroboros.preflight_node import node_environment
    from ouroboros.test_parallelism import NESTED_WORKERS_ENV

    temp_root = pathlib.Path(temp_root).resolve(strict=False)
    # The scrub drops settings keys (`test_environment.settings_keys`), secret suffixes,
    # `NODE_OPTIONS`, `VIRTUAL_ENV`, the OUROBOROS_/GH_/GIT_/PYTEST_/PYTHON/UV_/PIP_
    # prefixes and launch descriptors; writable paths are re-rooted under `temp_root`.
    env = node_environment(without_launch_descriptors(isolated_environment(
        temp_root, repo_worktree, create=create, keep=DIAGNOSTIC_OUTPUT_KEYS)))
    # PREPENDED, so `-p ouroboros_preflight_probe` resolves to the gate's own
    # worker-count plugin and not to anything the candidate tree or the
    # operator's PYTHONPATH happens to shadow it with.
    env["PYTHONPATH"] = str(probe_dir(temp_root))
    # Re-injected AFTER the scrub, so `-n auto` resolves to a count this process
    # chose rather than one the operator environment happened to carry. Inert for
    # the serial/legacy passes, which pass no `-n` at all.
    env["PYTEST_XDIST_AUTO_NUM_WORKERS"] = str(allocated_worker_count(temp_root) if workers is None else workers)
    env[NESTED_WORKERS_ENV] = env["PYTEST_XDIST_AUTO_NUM_WORKERS"]
    from ouroboros.test_evidence import DURATIONS_POINTER, EVIDENCE_ENV, EVIDENCE_POINTER, TEST_DURATIONS_ENV
    pointer = temp_root / EVIDENCE_POINTER
    evidence = pointer.read_text(encoding="utf-8").strip() if create and pointer.is_file() else str(temp_root / "evidence")
    env.update({"OUROBOROS_TEST_TEMP_ROOT": str(temp_root), EVIDENCE_ENV: evidence})
    durations = temp_root / DURATIONS_POINTER  # the install's table (`_durations_target`): read and brought up to date
    if create and durations.is_file():
        env[TEST_DURATIONS_ENV] = durations.read_text(encoding="utf-8").strip()
    if create:
        retain_diagnostic_outputs(temp_root, env, cwd=cwd or repo_worktree)
    env.setdefault("OUROBOROS_UI_EVIDENCE_DIR", str(pathlib.Path(evidence) / "ui"))
    return env


# (import name, distribution name, minimum version) for the plugins the parallel
# pass cannot work without. Kept in step with pyproject.toml.
_REQUIRED_PREFLIGHT_PLUGINS = (
    ("xdist", "pytest-xdist", (3, 5)),
    ("pytest_timeout", "pytest-timeout", (2, 1)),
)

# Runs under `-I` in a non-candidate directory; `sys.argv[1]` carries the spec
# above as JSON and stdout carries the problem list as JSON. Kept stdlib-only and
# free of f-strings/interpolation so nothing candidate-controlled can reach it.
_PLUGIN_PROBE_SOURCE = """
import importlib, json, sys
try:
    from importlib import metadata as md
except Exception:
    md = None
problems = []
for module, dist, minimum in json.loads(sys.argv[1]):
    try:
        importlib.import_module(module)
    except Exception as exc:
        problems.append("%s: %s is not importable (%s: %s)" % (dist, module, type(exc).__name__, exc))
        continue
    version = ""
    if md is not None:
        try:
            version = md.version(dist)
        except Exception:
            version = ""
    if not version:
        continue
    found = []
    for part in (version.split(".") + ["0"] * len(minimum))[:len(minimum)]:
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        found.append(int(digits or 0))
    if found < list(minimum):
        problems.append("%s %s is older than the required %s" % (
            dist, version, ".".join(str(p) for p in minimum)))
print(json.dumps(problems))
"""


def _probe_plugins(
    agent_python: str,
    probe_root: pathlib.Path,
    *,
    isolated: bool,
    spec: Optional[Sequence[tuple]] = None,
) -> list[str]:
    """One plugin probe run. Returns the problem list; empty means all present.

    ``spec`` is resolved at CALL time rather than bound as a default argument, so
    the required-plugin table stays one substitutable module attribute.
    """
    if spec is None:
        spec = _REQUIRED_PREFLIGHT_PLUGINS
    payload = json.dumps([[module, dist, list(minimum)] for module, dist, minimum in spec])
    argv = [agent_python] + (["-I"] if isolated else []) + ["-c", _PLUGIN_PROBE_SOURCE, payload]
    try:
        proc = subprocess.run(
            argv, cwd=str(probe_root), capture_output=True, text=True, timeout=60
        )
    except FileNotFoundError:
        # The caller already renders a precise "no such interpreter" message.
        raise
    except (subprocess.SubprocessError, OSError) as exc:
        return [f"could not probe {agent_python} for the preflight plugins: {exc}"]
    if proc.returncode != 0:
        detail = ((proc.stderr or "") + (proc.stdout or "")).strip()
        return [f"{agent_python} could not run the plugin probe: {detail}"]
    try:
        problems = json.loads((proc.stdout or "").strip() or "[]")
    except ValueError:
        return [f"unreadable plugin probe output from {agent_python}"]
    return [str(problem) for problem in problems]


def _verify_preflight_plugins(agent_python: str, probe_root: pathlib.Path) -> list[str]:
    """Prove the SELECTED interpreter really carries the parallel-pass plugins.

    Independent of the candidate on purpose. Inferring "the plugins are there"
    from the candidate's own pytest accepting ``-n``/``--dist``/``--timeout`` is
    not a check: a conftest can declare those exact option names with
    ``pytest_addoption`` and ignore them, so with xdist or pytest-timeout absent
    the nominal parallel lane would run serially, exit 0, and return green while
    proving nothing.

    The candidate-controlled import surface is the working directory, so the
    probe runs from ``probe_root`` — the disposable temp root, before the
    worktree even exists — and never from the diff-applied tree. No candidate
    conftest, sitecustomize or plugin can run before the answer is known.

    Returns the list of problems; an empty list means verified.
    """
    problems = _probe_plugins(agent_python, probe_root, isolated=True)
    if not problems:
        return []
    # `-I` additionally drops PYTHONPATH and the user site directory, which the
    # real pass DOES honour. Neither is candidate-controlled — both come from the
    # operator's own environment — so a strict miss is re-confirmed without them
    # rather than hard-blocking every commit on a legitimate `pip install --user`.
    return _probe_plugins(agent_python, probe_root, isolated=False)


def _plugin_missing_remediation(agent_python: str, rejected: str = "") -> str:
    """Shared instruction line for both routes to PREFLIGHT_PLUGIN_MISSING."""
    lead = (
        f"pytest under {agent_python} rejected {rejected}, so the two-pass "
        if rejected
        else f"the interpreter {agent_python} could not provide them, so the two-pass "
    )
    return (
        lead
        + "gate needs pytest-xdist and pytest-timeout installed into THAT interpreter. "
        "They are declared in pyproject.toml (pytest-xdist>=3.5, pytest-timeout>=2.1). "
        "The gate does NOT silently fall back to a serial run — a degraded gate is "
        "indistinguishable from a passing one; set OUROBOROS_PREFLIGHT_SERIAL=1 to "
        "take the legacy single serial pass deliberately while you provision."
    )


_DEFAULT_PREFLIGHT_TIMEOUT_SEC = 1800  # a selected run's total; the full suite has none


def _full_tier(selection, pytest_args) -> bool:
    """The whole default suite: a selection says so itself; without one, a run with no argv of its caller's."""
    return selection.tier == "full" if selection is not None else pytest_args is None


def _resolve_preflight_timeout(timeout: Optional[int] = None, *, full: bool = False) -> Optional[int]:
    """The total budget, in seconds, of the lanes that decide a run (Node, the passes); None is no budget.

    A bounded run (a selected tier, an explicit pytest argv) takes the env override
    (`OUROBOROS_PREFLIGHT_TIMEOUT_SEC`, one SSOT across callers, the Android entry's slower default), then the
    caller's own, then `_DEFAULT_PREFLIGHT_TIMEOUT_SEC`. The full suite takes only a caller's own: how long the
    whole suite honestly takes is this install's to find, and a constant below it makes every full proof red, so
    a full run ends by its tests' own 300 s bound, the containment proof and the owner's Stop, and records how
    long it took. The residual of that choice (owner 5A): nothing bounds a full run's collection or its bytecode
    warm, so a candidate whose test import blocks holds a full gate until Stop."""
    raw = None if full else os.environ.get("OUROBOROS_PREFLIGHT_TIMEOUT_SEC")
    if raw:
        try:
            parsed = int(float(raw))
            if parsed > 0:
                return parsed
        except (TypeError, ValueError, OverflowError):
            pass
    if timeout is not None:
        return timeout
    return None if full else _DEFAULT_PREFLIGHT_TIMEOUT_SEC


def _terminate_preflight_tree(proc: "subprocess.Popen", temp_root: pathlib.Path) -> None:
    """Reap pytest and its WHOLE tree on timeout/crash.

    ``killpg`` alone reaps only pytest's own process group; descendants that
    started their own session/group (Ouroboros spawns servers/browsers/extension
    children with new sessions) or double-forked to init survive it. So collect
    the live descendant PIDs and their group ids FIRST (the ``pgrep -P`` chain
    breaks once pytest dies and children reparent), then kill pytest's group, the
    recursive PID tree and each captured escapee group. The caller's existing
    ProcessContainer then reaps detached/reparented members by its unique token
    or Windows job, and reports any unconfirmed cleanup. A path substring is
    not process ownership. Platform-specific operations stay in platform_layer.
    """
    from ouroboros.platform_layer import (
        IS_WINDOWS,
        collect_descendant_pids,
        kill_pid_tree,
        kill_process_group_id,
        kill_process_tree,
        process_group_id,
    )

    pid = getattr(proc, "pid", 0) or 0
    descendant_pgids: set[int] = set()
    if pid and not IS_WINDOWS:
        for dpid in collect_descendant_pids(pid):
            gid = process_group_id(dpid)
            if gid and gid != pid:
                descendant_pgids.add(gid)
    try:
        kill_process_tree(proc)
    except Exception:
        pass
    if pid:
        try:
            kill_pid_tree(pid)
        except Exception:
            pass
    for gid in descendant_pgids:
        kill_process_group_id(gid)


def _partial_stream_text(chunk) -> str:
    """Decode one `TimeoutExpired.output`/`.stderr` payload.

    `Popen.communicate` joins the RAW partial reads into the exception before it
    ever applies the text-mode decoding, so these attributes are bytes even when
    the process was opened with ``text=True``. Both forms are accepted rather
    than assumed, because the whole point of reading them is that this is the
    path where nothing else survived.
    """
    if not chunk:
        return ""
    return chunk.decode("utf-8", "replace") if isinstance(chunk, bytes) else str(chunk)


def _controller_diagnostics(output_root) -> str:
    """Keep the xdist crash label and the timeout line's display independent of a long preview."""
    patterns = (*_WORKER_CRASH_PATTERNS, _TIMEOUT_CALLBACK_LINE)
    found = {}
    for name in ("stdout.bin", "stderr.bin"):
        with (pathlib.Path(output_root) / name).open("rb") as stream:
            start = True
            while chunk := stream.readline(8192):
                if start:
                    text = _strip_decoration(chunk.decode("utf-8", "replace"))
                    for index, pattern in enumerate(patterns):
                        if index not in found and pattern.search(text):
                            found[index] = text
                start = chunk.endswith(b"\n")
    return "\n".join(found.values()) + ("\n" if found else "")


def _execute_lane(argv: Sequence[str], worktree: pathlib.Path, temp_root: pathlib.Path,
                  timeout: Optional[float], lane: str):
    """Run ONE contained lane of the gate (a pytest pass, the bytecode warm) in the prepared worktree.

    ``timeout`` is the caller's exact remaining share of the TOTAL budget and stays a float — rounding it up
    would let the last lane outlive the total; None when the run has no total budget (the full suite).

    Returns a ``CapturedPass``: ``(returncode, combined_output, containment_error)`` plus the lane's own
    diagnostic errors and container token. ``returncode is None`` means the lane exhausted its slice of the
    budget and its whole process tree was reaped; its output is whatever the child had flushed before the
    kill, the only evidence of WHICH test was running.

    A NON-EMPTY ``containment_error`` is a hard block the caller must not accept as green, whatever the return
    code says: the container's post-lane SCAN found a process the lane spawned still alive, or could not
    determine whether one is (which counts the same), so a green lane may be sitting on top of a live tree
    that survives into the next lane and past teardown. The reap therefore happens before the return rather
    than in the `finally` below, which cannot alter an already-computed tuple; only the handle release stays
    unconditional. The lane's tree runs behind the owner's work (``background=True``).
    """
    from ouroboros.process_containment import ProcessContainer

    from ouroboros.preflight_inventory import note_lane_process
    from ouroboros.test_evidence import CapturedPass, FileOutput

    argv = list(argv)
    container = ProcessContainer(lifeline=True, overload=True)
    output = FileOutput(temp_root, lane)
    proc = None
    try:
        proc = container.spawn(argv, background=True, cwd=str(worktree), env=_preflight_env(temp_root, worktree),
                               stdout=output.stdout, stderr=output.stderr)
        try:
            returncode: Optional[int] = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_preflight_tree(proc, temp_root)
            returncode = None
        # Regular files do not wait for a detached child's pipe EOF. Reap even
        # after green, before reading or releasing anything the lane wrote.
        reap_error = container.reap()
        if proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                reap_error = reap_error or f"direct {lane} process did not exit after reap"
        preview = output.finish(returncode=returncode, containment_error=reap_error, argv=argv)
        if lane == "pytest":
            try:
                preview = _controller_diagnostics(output.root) + preview
            except OSError as exc:
                output.failed("controller diagnostics", exc)
        return CapturedPass(returncode, preview, reap_error, output.errors, container.token)
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            _terminate_preflight_tree(proc, temp_root)
        note_lane_process(exc, container, proc)
        raise
    finally:
        if proc is not None and proc.poll() is None:
            _terminate_preflight_tree(proc, temp_root)
        container.reap()
        container.close()
        output.close()


def _execute_pytest_pass(
    agent_python: str,
    worktree: pathlib.Path,
    temp_root: pathlib.Path,
    args: Sequence[str],
    timeout: Optional[float],
) -> tuple[Optional[int], str, str]:
    """Run ONE pytest invocation in the prepared worktree: ``_execute_lane`` with pytest's argv (a selected scope
    travels in its inventory plugin, not on the command line) and the controller's diagnostics ahead of the
    output. Both passes carry the same 300 s per-test timeout; a kill of the whole pass budget leaves the
    flushed output as the only evidence of which test was running."""
    from ouroboros.preflight_inventory import launch_args

    return _execute_lane([agent_python, "-m", "pytest", *launch_args(args)], worktree, temp_root, timeout, "pytest")

_TRUNCATION_MARKER = "\n...(truncated)..."


def _bounded(output: str, max_output: int) -> str:
    """Bound pytest output by cutting the MIDDLE, never the tail.

    pytest prints the FAILURES section and the short test summary at the END —
    a head-only cut (the old shape) discarded exactly the lines that name the
    failing tests once the run was long enough to need cutting. The head is
    kept too (collection errors and the session header live there): one third
    head, two thirds tail."""
    if len(output) <= max_output:
        return output
    head = max_output // 3
    tail = max_output - head
    return output[:head] + _TRUNCATION_MARKER + "\n" + output[-tail:]


def _diagnosis(header: str, remediation: str, body: str, max_output: int) -> str:
    """Header, then remediation, THEN the bounded pytest body.

    The caller re-truncates this string from the TAIL at the same limit
    (ouroboros/tools/review_helpers.py (_run_review_preflight_tests) passes ``limit=MAX_OUTPUT=8000``),
    so a remediation printed after a full-budget body is the first thing lost —
    precisely when the output is long enough to need it. The prefix, the
    truncation marker, and its newline are reserved out of the body's budget
    too, so the whole string stays inside the caller's declared limit instead
    of overrunning it (an overrun would cost the tail — the failure summary)."""
    prefix = f"{header}\n{remediation}\n" if remediation else f"{header}\n"
    body = body.strip()
    room = max_output - len(prefix) - len(_TRUNCATION_MARKER) - 1
    if not body or room <= 0:
        # A limit too small to hold header+remediation+marker still may not be
        # overrun: the invariant is unconditional, so the prefix is cut too.
        return prefix.rstrip("\n")[:max_output]
    return prefix + _bounded(body, room)


_TIMEOUT_EXCERPT_HEADER = (
    "\nlast output before the kill (the hung test is usually the last one started):\n"
)


def _with_timeout_excerpt(message: str, output: str, max_output: int) -> str:
    """Append what the killed pass had already flushed to its timeout message.

    A pass killed at its budget produces no exit code to classify, and the 300 s
    per-test timeout both passes carry did not end its hang first, so without this
    the operator is told only which pass timed out — with nothing naming the test
    that hung. The TAIL is kept
    (progress output ends at the test that never finished) and the whole result
    stays inside the caller's declared ``max_output`` — unconditionally, exactly
    as ``_diagnosis`` does, including when the message alone already fills the
    budget (unreachable at the production 8000, but an invariant with an
    exception is not one).
    """
    body = (output or "").strip()
    if not body:
        return message[:max_output]
    head_marker = _TRUNCATION_MARKER.strip() + "\n"
    room = max_output - len(message) - len(_TIMEOUT_EXCERPT_HEADER) - len(head_marker)
    if room <= 0:
        return message[:max_output]
    if len(body) > room:
        body = head_marker + body[-room:]
    return message + _TIMEOUT_EXCERPT_HEADER + body


def _rejected_parallel_flags(output: str) -> list[str]:
    """Parallel-pass option names that pytest reported as unrecognized.

    Matches WHOLE tokens against ``_PARALLEL_FLAG_NAMES``: the old substring
    test saw "-n" inside "--no-header" and blamed a missing pytest-xdist for
    any usage error at all.
    """
    rejected: list[str] = []
    for line in output.splitlines():
        _, sep, tail = line.partition("unrecognized arguments:")
        if not sep:
            continue
        for token in tail.split():
            name = token.split("=", 1)[0]
            if name in _PARALLEL_FLAG_NAMES and name not in rejected:
                rejected.append(name)
    return rejected


def _crash_remediation(recorded: Optional[dict] = None) -> str:
    """The instruction line for a dead worker — it depends on WHY it died.

    The label and the hard block are identical either way; only this line changes.
    ``recorded`` is this pass's own results.json (``recorded_session``). The timeout
    instruction needs the worker the session recorded dead among its crash rows and
    every crash row naming its worker and a matched timeout; anything less, a missing
    or unreadable record included, is diagnosed before any fix is chosen.
    """
    record = recorded if isinstance(recorded, dict) else {}
    dead = record.get("worker_crash")
    reports = record.get("reports")
    rows = [row for row in reports if isinstance(row, dict) and row.get("phase") == "crash"] if isinstance(reports, list) else []
    timed = [row["worker"] for row in rows
             if isinstance(row.get("worker"), str) and type(row.get("timeout_callback_seconds")) in (int, float)]
    if rows and len(timed) == len(rows) and isinstance(dead, dict) and dead.get("worker") in timed:
        return (
            "pytest-timeout's callback was invoked for the test its worker was running: the per-test "
            "timer fired before that test finished, and the worker was lost afterwards. Make that test "
            "faster or split it. Do NOT mark it @pytest.mark.serial: a resource group or the exclusive "
            "pass carries the same per-test timeout."
        )
    return (
        "A pytest-xdist worker DIED instead of reporting a failure, and nothing recorded why: a native "
        "crash, an OS kill, an os._exit in a test and a timeout that left no callback record look alike. "
        "Rerun the test it was running on its own and read that output before choosing a fix; name the host "
        "resource it cannot isolate (@pytest.mark.serial(\"<resource>\"): a fixed port, a daemon, a process-global "
        "its fixture cannot restore) only once it proves to share one, serial(\"exclusive\") only when it harms "
        "or needs no live sibling; never a flake/retry."
    )


def _classify_pass_result(
    label: str,
    returncode: Optional[int],
    output: str,
    max_output: int,
    *,
    parallel: bool,
    agent_python: str = "python3",
    elapsed: float = 0.0,
    recorded: Optional[dict] = None,
) -> Optional[str]:
    """Turn one pass's exit code + output into ``None`` or a bounded diagnosis.

    Pure function: no process, no filesystem. Exit 0 and exit 5
    (nothing collected for this lane) are green here; the orchestrator alone
    decides that ALL passes being empty is a block. ``recorded`` (this pass's
    own results.json, when its controller left one) only selects a dead
    worker's remediation.
    """
    if returncode in (0, _PYTEST_EXIT_NO_TESTS):
        return None
    head = f"{label} pass, exit {returncode}, {elapsed:.0f}s"

    # Both hard-block labels describe xdist failures, so they are only reachable
    # for a pass that actually handed the run to xdist. Firing them on the
    # serial or legacy single pass would not change the fail-closed verdict (a
    # nonzero exit still blocks) — it would hand the reader a confidently wrong
    # remediation, which is the expensive part for an autonomous agent.
    if parallel:
        if returncode == _PYTEST_EXIT_USAGE_ERROR:
            rejected = _rejected_parallel_flags(output)
            if rejected:
                return _diagnosis(
                    f"⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_PLUGIN_MISSING (hard block): {head}",
                    _plugin_missing_remediation(agent_python, " ".join(rejected)),
                    output, max_output,
                )
        matched = [
            line for line in output.splitlines()
            if any(p.search(_strip_decoration(line)) for p in _WORKER_CRASH_PATTERNS)
        ]
        if matched:
            # The crash lines lead, but the FULL pytest output still follows:
            # a crash usually happens alongside real failures, and a pattern
            # false positive must never cost the reader those lines. The body
            # budget in `_diagnosis` trims the tail, not the highlight.
            return _diagnosis(
                f"⚠️ PRE_PUSH_TEST_ERROR: PARALLEL_WORKER_CRASH (hard block): {head}",
                _crash_remediation(recorded),
                "\n".join(matched) + "\n\n" + output, max_output,
            )

    return _diagnosis(f"⚠️ PRE_PUSH_TEST_ERROR: {head}", "", output, max_output)


_SOURCE_INDEX_BLOCK = "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_SOURCE_INDEX (hard block): "


def _capture_source_index_tree(
    repo: pathlib.Path,
    max_output: int,
) -> tuple[str | None, str | None]:
    """Return a merged index tree, or ``None`` only for proven unmerged entries."""
    try:
        source_index = _run_git(repo, ["write-tree"])
    except (subprocess.SubprocessError, OSError) as exc:
        return None, _diagnosis(
            _SOURCE_INDEX_BLOCK + "the source index could not be snapshotted",
            "The disposable preflight cannot reproduce an unreadable source index, "
            "so no test pass was run. Repair the Git/filesystem error shown below.",
            str(exc),
            max_output,
        )
    if source_index.returncode == 0:
        return str(source_index.stdout).strip(), None
    try:
        unmerged = _run_git(repo, ["ls-files", "-u"])
    except (subprocess.SubprocessError, OSError) as exc:
        return None, _diagnosis(
            _SOURCE_INDEX_BLOCK + "the failed source index could not be classified",
            "The preflight may preserve live resolution semantics only for a proven "
            "unmerged index. No test pass was run.",
            str(exc),
            max_output,
        )
    if unmerged.returncode == 0 and str(unmerged.stdout).strip():
        return None, None
    return None, _diagnosis(
        _SOURCE_INDEX_BLOCK + "the source index could not be snapshotted",
        "The source `git write-tree` failed without real unmerged entries. "
        "The disposable preflight cannot reproduce the staged candidate, so no "
        "test pass was run. Repair the index error shown below.",
        str(source_index.stderr).strip() or "git write-tree failed",
        max_output,
    )


def _install_source_index_tree(
    worktree: pathlib.Path,
    source_index_tree: str,
    max_output: int,
) -> str | None:
    """Install and verify the immutable source index without updating files.

    A Git call that raised and one that exited nonzero are the same typed block.
    """
    try:
        read_tree = _run_git(worktree, ["read-tree", source_index_tree])
        failure = None if read_tree.returncode == 0 else read_tree.stderr.strip() or "git read-tree failed"
    except (subprocess.SubprocessError, OSError) as exc:
        failure = str(exc)  # may be empty: an exception is a block whatever its message
    if failure is not None:
        return _diagnosis(
            _SOURCE_INDEX_BLOCK + "the source index tree could not be installed",
            "The disposable index must exactly reproduce the source index before "
            "tests run. No candidate test pass was started.",
            failure,
            max_output,
        )
    try:
        projected_index = _run_git(worktree, ["write-tree"])
    except (subprocess.SubprocessError, OSError) as exc:
        problem, detail = "the disposable index could not be verified", str(exc)
    else:
        if projected_index.returncode == 0 and str(projected_index.stdout).strip() == source_index_tree:
            return None
        problem = "the disposable index does not match the source index tree"
        detail = projected_index.stderr.strip() or (
            f"expected {source_index_tree}, got {str(projected_index.stdout).strip() or '<empty>'}"
        )
    return _diagnosis(
        _SOURCE_INDEX_BLOCK + problem,
        "The source and disposable indexes must have identical `write-tree` "
        "identities before tests run. No candidate test pass was started.",
        detail,
        max_output,
    )


def _with_retention_notice(verdict: str, unproven: str, worktree: pathlib.Path, max_output: int) -> str:
    """Name the kept tree (``worktree.parent``) right after the verdict's header.

    Ahead of any bounded body, which is what yields when the budget is tight —
    the priority ``_diagnosis`` keeps — and never past ``max_output``."""
    if not unproven:
        return verdict
    header, _, rest = verdict.partition("\n")
    notice = (
        f"RETAINED (not deleted): {worktree.parent}, with the git worktree registration of "
        f"{worktree}, because processes of this run were not proven gone; remove both "
        f"(`git worktree remove --force {worktree}`) only once they are."
    )
    return "\n".join(part for part in (header, notice, rest) if part)[:max_output]


def _warm_bytecode(agent_python, worktree, temp_root, evidence, timeout, files=()) -> str:
    """The gate's warm before its passes (ouroboros/test_bytecode.py): one contained root compiles the candidate
    and collects its tests, or a selection's modules, in as many shards as the gate admitted workers; its facts
    join the run's lanes. Returns a hard-block diagnosis only when the warm's tree outlived it (containment),
    never for a warm that failed or ran out of ``timeout`` (what the run's budget still held, None for none;
    the passes are charged nothing for it): those leave the passes cold, recorded as such."""
    from ouroboros.test_bytecode import command

    files_from = None
    if files:
        files_from = probe_dir(temp_root) / "warm-files.txt"
        files_from.write_text("\n".join(files) + "\n", encoding="utf-8")
    argv = command(agent_python, worktree, workers=allocated_worker_count(temp_root), files_from=files_from)
    started = time.monotonic()
    returncode, preview, reap_error = result = _execute_lane(argv, worktree, temp_root, timeout, "warm")
    facts = next((line.removeprefix("WARM_BYTECODE ").strip() for line in preview.splitlines()
                  if line.startswith("WARM_BYTECODE ")), "")
    _record_lane(evidence, "warm", {"evidence_errors": result.evidence_errors}, returncode=returncode,
                 containment_error=reap_error, seconds=round(time.monotonic() - started, 1),
                 facts=json.loads(facts) if facts else None)
    if reap_error:
        return _diagnosis(
            "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_CONTAINMENT_FAILED (hard block): the warm ran, but processes it "
            "spawned were still alive afterwards (or could not be determined to be gone)",
            "The bytecode warm collects the candidate's test tree once; a tree it leaves alive survives into the "
            "passes and past teardown, so its verdict is not accepted. Find what the candidate's collection starts "
            "and does not wait for.", reap_error, 8000)
    return ""


def _durations_target(ctx) -> Optional[pathlib.Path]:
    """The install's durations table for a caller with a canonical data root (the receipts' root); None otherwise."""
    from ouroboros.preflight_receipts import canonical_data_root
    from ouroboros.test_evidence import durations_path

    root = canonical_data_root(ctx)
    return durations_path(root) if root is not None else None


def _release_hermetic_tree(repo, temp_root, worktree, worktree_added: bool, unproven: str) -> None:
    """Delete the disposable tree, unless a contained lane left its teardown unproven.

    ``unproven`` is armed before each contained lane starts and cleared only by
    that lane's own proven reap, returned or carried by its exception, so a lane
    that RAISED without proof keeps the tree exactly like one whose container
    reported a live or undeterminable leftover: a surviving process never loses
    the worktree it runs in, and the source repository keeps the registration
    for whoever proves those processes gone.
    The marker is the enclosing layers' retention protocol; this decision does
    not depend on the marker write succeeding. A marker beneath the proven root
    keeps its own folder and everything else goes (``retained_folders``), the
    worktree's registration with it unless the marker sits in that worktree."""
    from ouroboros.test_environment import release_except_retained, retain_tree, retained_folders
    from ouroboros.test_evidence import remove_scratch_tree
    kept = {} if unproven else retained_folders(temp_root)
    if unproven or temp_root in kept:
        retain_tree(temp_root, f"hermetic preflight: {unproven or 'retention marker or unreadable subtree'}")
        return
    if worktree_added and worktree not in kept:
        removed = subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if removed.returncode:
            retain_tree(temp_root, f"git worktree removal failed: {removed.stderr}")
            return
    if kept:
        release_except_retained(temp_root, kept, remove_scratch_tree)
    else:
        remove_scratch_tree(temp_root)


def _assemble_live_source(repo, worktree, evidence):
    """Apply the admitted live projection and retain the actual tested recipe once.

    Source evidence is exported AFTER the disposable index/files agree with their
    respective source projections. Exporting twice paid twice and left ambiguous
    earlier inputs if the second export only partly completed.
    """
    from ouroboros.test_evidence import describe_gaps, source_evidence, write_record

    combined = _run_git(repo, ["diff", "--binary", "--no-ext-diff", "--no-textconv", "--no-color",
                             "--src-prefix=a/", "--dst-prefix=b/", "HEAD"], binary_stdout=True)
    if combined.returncode != 0:
        raise RuntimeError(combined.stderr.strip() or "git diff HEAD failed")
    _apply_diff(worktree, combined.stdout or b"")
    _copy_untracked(repo, worktree)
    try:
        source = source_evidence(worktree, evidence["path"] / "source")
        if source.get("gaps"):
            # A proof covers its whole recorded candidate: a gap makes this run ineligible.
            raise RuntimeError("untracked inputs not recorded: " + describe_gaps(source["gaps"]))
        source["source_repository"] = str(repo)
        write_record(evidence["path"] / "source" / "source.json", source)
    except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
        evidence["retain_reason"] = f"candidate evidence incomplete: {exc}"
        from ouroboros.test_evidence import diagnostics_incomplete
        diagnostics_incomplete(str(exc))  # Preserve an enclosing disposable Git owner too.
        raise


def _record_lane(evidence, lane, result, **facts) -> None:
    """One lane's observed exit/containment and its separate diagnostic errors."""
    evidence.setdefault("lanes", []).append({"lane": lane, **facts})
    fact = result.get if isinstance(result, dict) else lambda name, default: getattr(result, name, default)
    evidence.setdefault("evidence_errors", []).extend(fact("evidence_errors", ()))
    evidence.setdefault("containment_tokens", []).append(fact("containment_token", ""))


def _missing_tests_tree(repo: pathlib.Path, phase: str, max_output: int) -> Optional[str]:
    """No `tests/` in the candidate: out of scope (None) when the baseline never had one, a hard block when it did.

    Git does not track empty directories, so staging the removal of every test file removes `tests/` with
    them. An early return here once skipped the passes entirely, so the all-passes-empty hard block never
    ran and the change that deleted the gate sailed through it.
    """
    baseline_refs = _baseline_refs(phase)
    try:
        baseline_tracks_tests = _head_tracks_tests(repo, baseline_refs)
    except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
        return _diagnosis(
            "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_TESTS_BASELINE_UNREADABLE (hard block): "
            f"could not read {' or '.join(baseline_refs)}",
            "The working tree just lost its tests/ directory and git could not read "
            "one of the baseline refs well enough to say whether that loss is real. "
            "Treating an unreadable ref as 'never tracked tests' would let this "
            "candidate through the gate it exists to trip. This is not a test failure.",
            str(exc), max_output,
        )
    if baseline_tracks_tests:
        return (
            "⚠️ PRE_PUSH_TEST_ERROR: the candidate change removes the entire tests/ tree "
            f"that {' or '.join(baseline_refs)} carries. The preflight cannot verify "
            "a change that deletes its own gate — restore tests/ (git does not track empty "
            "directories, so deleting every test file deletes the directory with them)."
        )
    return None


def _collected_nothing(recorded: Optional[dict]) -> bool:
    """A green pass whose own results.json (the probe's pointer) counts no collected test."""
    return isinstance(recorded, dict) and recorded.get("tests_collected") == 0


def _containment_failed(label: str, reap_error: str, max_output: int) -> str:
    """The pass ran, but the scan says processes it spawned are still alive (or cannot tell): a live tree
    outlives this whole run, so the pass verdict is not accepted, whatever it was."""
    return _diagnosis(
        "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_CONTAINMENT_FAILED (hard block): "
        f"the {label} pass ran, but processes it spawned were still "
        "alive afterwards (or could not be determined to be gone)",
        "The container DETECTS leaked processes; it does not promise to kill "
        "them, and a process it cannot inspect counts as leaked. The pass "
        "verdict is not accepted, whatever it was, because a leaked tree "
        "survives into the next pass and past teardown. Find the test that "
        "spawns a real process, binds a real port, or daemonises a helper "
        "and does not wait for it — make it clean up (an owned child it waits "
        "for marks nothing; a shared host resource is named with "
        "@pytest.mark.serial(\"<resource>\")); never a "
        "flake/retry. Kill any pid named below before re-running; when the "
        "report names none it states the reason instead — read that line, "
        "since more than one cause leaves no pid to name.",
        reap_error, max_output,
    )


def _run_hermetic_pytest(
    repo_dir: pathlib.Path | str,
    *,
    timeout: Optional[int] = None,
    pytest_args: Optional[Sequence[str]] = None,
    max_output: int = 8000,
    phase: str = "post_commit",
    ctx=None,
    selected: bool = False, base: str | None = None, extra_tests=(), full_tests=False, test_reason="",
    evidence=None,
) -> Optional[str]:
    """Run Node and both pytest lanes in one disposable candidate checkout.

    A full run is the default; reviewed ordinary commits request ``selected`` against
    ``base`` (default HEAD before commit, HEAD~1 after) plus authored widening. Green
    runs mint receipts: full ones as full, selected ones only of an exact commit.
    Skip/no suite returns None without proof; failures return bounded diagnostics.
    Node and the passes share one total budget (``_resolve_preflight_timeout``; the
    full suite has none). The bytecode warm decides nothing, so it takes none of it:
    it is bounded by what the budget still held when it started. The run records
    its budget and how long it took.

    Assembly captures one hardened, raw-byte worktree-vs-HEAD diff (including
    live merge resolutions), applies it with LF preserved, and copies non-ignored
    untracked files. Untracked symlinks are dereferenced, non-files skipped.
    A merged source index is installed and verified without changing files; an
    unmerged index retains the live-resolution projection. The phase also selects
    the deleted-suite baseline: HEAD before commit, HEAD plus HEAD~1 afterwards.
    Unconfirmed teardown retains the checkout and its Git registration.
    """
    previous_proof = getattr(ctx, "_preflight_test_proof", None)
    previous_inventory = getattr(ctx, "_preflight_test_inventory", None)
    if ctx is not None:
        ctx._preflight_test_proof, ctx._preflight_tests_passed, ctx._preflight_test_inventory = None, False, None
    began = time.monotonic()
    # Check before execution: an unrenderable failure used to become an empty
    # string, which the caller reads as success. Never silently green a red run.
    if max_output <= 0:
        return (
            f"⚠️ PRE_PUSH_TEST_ERROR: max_output must be positive (got {max_output}); "
            "refusing to run the preflight with an output budget that cannot render a "
            "failure, because an unrenderable diagnosis reads as success"
        )
    repo = pathlib.Path(repo_dir).resolve()
    if not (repo / ".git").exists():
        return None
    if not (repo / "tests").exists():
        return _missing_tests_tree(repo, phase, max_output)
    agent_python = os.environ.get("OUROBOROS_AGENT_PYTHON") or sys.executable or "python3"
    source_index_tree, source_index_error = _capture_source_index_tree(repo, max_output)
    if source_index_error is not None:
        return source_index_error
    temp_root = gate_root()
    worktree = temp_root / "repo"
    worktree_added = False
    unproven_teardown = ""
    selection, subject, inventory, lease = None, None, {}, None
    try:
        from ouroboros.test_evidence import DURATIONS_POINTER, EVIDENCE_POINTER
        from ouroboros.test_parallelism import reserve
        (temp_root / EVIDENCE_POINTER).write_text(str(evidence["path"]), encoding="utf-8")
        if (durations := _durations_target(ctx)) is not None:  # every session of this gate keeps it current
            (temp_root / DURATIONS_POINTER).write_text(str(durations), encoding="utf-8")
        # One reservation under the ledger's lock: the admission and this run's row, which stays leased in the
        # host ledger until the run ends, whatever the verdict.
        admitted, lease = reserve("hermetic gate")
        # Install before compiling specs: -p must name the actual nonce module.
        probe_module = install_worker_probe(temp_root, admitted)
        evidence["workers"] = recorded_admission(temp_root)
        passes = _preflight_pass_specs(pytest_args, probe_module=probe_module)
        # Before ANY candidate code is on disk, let alone imported.
        if any(spec.parallel for spec in passes):
            problems = _verify_preflight_plugins(agent_python, temp_root)
            if problems:
                return _diagnosis(
                    "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_PLUGIN_MISSING (hard block): "
                    "interpreter verification",
                    _plugin_missing_remediation(agent_python),
                    "\n".join(problems), max_output,
                )

        # Preserve HEAD's LF bytes on Windows too, paired with _apply_diff.
        add = _run_git(
            repo,
            ["-c", "core.autocrlf=false", "-c", "core.eol=lf",
             "worktree", "add", "--detach", str(worktree), "HEAD"],
            timeout=60,
        )
        if add.returncode != 0:
            return f"⚠️ PRE_PUSH_TEST_ERROR: could not create hermetic worktree: {add.stderr.strip()}"
        worktree_added = True

        if source_index_tree is not None:
            if (source_index_error := _install_source_index_tree(worktree, source_index_tree, max_output)) is not None:
                return source_index_error

        # One worktree-vs-HEAD capture includes live resolutions. A staged +
        # unstaged pair emits unmerged stubs/combined hunks that git apply can
        # reject or silently drop. Pin content-changing diff config and retain
        # raw bytes end to end (the byte/EOL contracts live in _run_git/_apply_diff).
        try:
            _assemble_live_source(repo, worktree, evidence)
        # Capture/apply timeouts and filesystem faults are assembly failures,
        # not pytest timeouts or missing interpreters from the outer handlers.
        except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
            return _diagnosis(
                "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_CANDIDATE_ASSEMBLY (hard block): "
                "the candidate tree could not be assembled from the live worktree",
                "The worktree-vs-HEAD capture could not be built or applied, so "
                "there is no candidate worth testing and no pass was run. An "
                "unmerged index is NOT the cause — a merge in progress is a "
                "supported source state for this capture — so this block means "
                "the capture or apply itself failed: read the git/filesystem "
                "error in the body below. This is not a test failure.",
                str(exc), max_output,
            )
        from ouroboros.commit_admission import (
            PreflightTestProof, capture_preflight_test_subject, log_preflight_test_proof,
            preflight_test_workload_unchanged, selected_workload_unchanged,
        )
        from ouroboros.preflight_receipts import GateReceipt
        from ouroboros.preflight_inventory import launching, selection_passes, read_inventory, inventory_problem
        if selected:
            from ouroboros.test_selection import select_tests
            selection = select_tests(worktree, base=base or ("HEAD" if phase == PRE_COMMIT_PHASE else "HEAD~1"),
                                     extra_tests=extra_tests, full_tests=full_tests, test_reason=test_reason)
            if pytest_args is not None:
                raise ValueError("selected scope cannot be combined with custom pytest arguments")
            passes = selection_passes(selection, probe_module=probe_module, temp_root=temp_root,
                                      inventory_dir=evidence["path"])
        # The tier is known only now: a selection may have widened to the whole suite.
        timeout = evidence["budget_seconds"] = _resolve_preflight_timeout(timeout, full=_full_tier(selection, pytest_args))

        subject = capture_preflight_test_subject(
            worktree, timeout=timeout, pytest_args=pytest_args,
            passes=passes, agent_python=agent_python, probe_module=probe_module, selection=selection,
        ) if ctx is not None else None
        if selection is not None and ctx is not None and subject is None:
            raise ValueError("selected test subject could not be bound")
        can_reuse = isinstance(previous_proof, PreflightTestProof) and previous_proof.covers(subject)
        receipt = None if can_reuse else GateReceipt.prepare(ctx, subject, passes, probe_module, agent_python, temp_root, worktree)
        if can_reuse or receipt.covered():
            evidence.update(reuse="process_proof" if can_reuse else "receipt", reused_from=getattr(
                ctx, "_preflight_previous_evidence", None) if can_reuse else receipt.evidence)
            ctx._preflight_test_proof = previous_proof if can_reuse else subject
            ctx._preflight_test_inventory = previous_inventory if can_reuse else None
            ctx._preflight_tests_passed = True
            log_preflight_test_proof(ctx, ctx._preflight_test_proof, reused=True, phase=phase, receipt=receipt and receipt.disclosure(),
                                     empty_lanes=receipt and receipt.empty_lanes)
            return None
        started = time.monotonic()
        unproven_teardown = "the node lane did not return, so its teardown is unconfirmed"
        with launching(evidence):
            node_result = run_node_tests(worktree, temp_root, timeout, max_output)
        evidence["execution"] = (node_result or {}).get("execution", "not_run")
        unproven_teardown = (node_result or {}).get("reap_error") or ""
        if node_result is not None:
            _record_lane(evidence, "node", node_result, returncode=node_result.get("returncode"), containment_error=unproven_teardown)
        if node_error := (node_result or {}).get("error"):
            return _with_retention_notice(node_error, unproven_teardown, worktree, max_output)
        unproven_teardown = "the bytecode warm did not return, so its teardown is unconfirmed"
        warm_started = time.monotonic()
        with launching(evidence):  # the prefix every pass reads; a warm out of budget or red leaves them cold
            warm_error = _warm_bytecode(agent_python, worktree, temp_root, evidence,
                                        None if timeout is None else max(0.0, timeout - (warm_started - started)),
                                        selection.files if selection is not None else ())
        started += time.monotonic() - warm_started  # the passes keep the budget the warm found: it decides nothing
        unproven_teardown = ""  # it returned: its own reap decided, and a leak is the block below
        if warm_error:
            unproven_teardown = next((lane.get("containment_error") for lane in reversed(evidence.get("lanes", []))
                                      if lane.get("lane") == "warm"), "") or warm_error
            return _with_retention_notice(warm_error, unproven_teardown, worktree, max_output)
        empty_passes: list[str] = []
        timings: list[tuple[str, float]] = []
        for spec in passes:
            # Do not round the remaining shared budget up.
            remaining = None if timeout is None else timeout - (time.monotonic() - started)
            if remaining is not None and remaining <= 0:
                return (
                    f"⚠️ PRE_PUSH_TEST_ERROR: the {spec.label} pass never started — the "
                    f"total budget of {timeout} seconds was exhausted by the earlier pass(es)"
                )
            if lease is not None and not spec.parallel:
                lease.hold(1)  # a one-process pass holds one slot: the admitted count went with the xdist pass
            # Headers report this pass, not cumulative shared-budget time.
            pass_started = time.monotonic()
            unproven_teardown = f"the {spec.label} pass did not return, so its teardown is unconfirmed"
            with launching(evidence):
                lane_result = _execute_pytest_pass(
                    agent_python, worktree, temp_root, spec.args, remaining
                )
            returncode, output, reap_error = lane_result
            _record_lane(evidence, "pytest", lane_result, returncode=returncode, containment_error=reap_error, **{"pass": spec.label})
            unproven_teardown = reap_error
            inventory_error = ""
            if selection is not None:  # the full native inventory is run evidence; ledgers stay compact
                inventory[spec.label], inventory_error = read_inventory(evidence["path"], spec.label)
                if ctx is not None:
                    ctx._preflight_test_inventory = inventory
            elapsed = time.monotonic() - pass_started
            timings.append((spec.label, round(elapsed, 1)))
            if reap_error:
                # Checked BEFORE the exit code, including before the green path: an exit 0 taken on top of a
                # live tree is exactly the fail-open the container exists to close, and the more urgent report.
                return _with_retention_notice(_containment_failed(spec.label, reap_error, max_output),
                                              unproven_teardown, worktree, max_output)
            if returncode is None:
                return _with_timeout_excerpt(
                    f"⚠️ PRE_PUSH_TEST_ERROR: pytest timed out after {remaining:.0f} seconds "
                    f"in the {spec.label} pass (total budget {timeout} seconds)",
                    output, max_output,
                )
            if inventory_error and returncode in (0, _PYTEST_EXIT_NO_TESTS):
                return f"⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_INVENTORY: {spec.label}: {inventory_error}"
            recorded = recorded_session(temp_root) if probe_module in spec.args else None
            # A candidate's own default-lane plugin (tests/default_lane.py) ends a pass that holds none of the
            # selection's tests with 0 and says so; its record still shows what it collected: nothing.
            if returncode == _PYTEST_EXIT_NO_TESTS or (returncode == 0 and _collected_nothing(recorded)):
                empty_passes.append(spec.label)
                continue
            if returncode == 0 and spec.parallel and probe_module in spec.args:
                # Use the actual nonce flag, not the stem or spec.parallel:
                # custom argv gets no host probe; red/empty lanes need no count.
                observed = observed_worker_ids(temp_root)
                if len(observed) < (required := min(MIN_PREFLIGHT_WORKERS, allocated_worker_count(temp_root))):
                    return _diagnosis(
                        "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_PARALLELISM_LOST (hard block): "
                        f"the {spec.label} pass returned green on {len(observed)} xdist "
                        f"worker(s), fewer than the {required} its admitted budget requires",
                        "A pass that never ran concurrently proves nothing about the "
                        "parallel-only defects this gate exists to catch, so it is not "
                        "accepted as green. Check the repository's pytest configuration for "
                        "an addopts entry that disables xdist, and the interpreter for a "
                        "pytest-xdist that cannot start workers. Set "
                        "OUROBOROS_PREFLIGHT_SERIAL=1 to take the legacy single serial pass "
                        "deliberately instead of getting a silently serial one.",
                        "workers observed: " + (", ".join(sorted(observed)) or "none"),
                        max_output,
                    )
            if returncode != 0:
                # The EXIT CODE decides that this pass failed; the rendered text
                # only decides how it reads. Gating the return on the diagnosis
                # being truthy made a budget too small to render one turn a red
                # pass into a green gate.
                return _classify_pass_result(
                    spec.label, returncode, output, max_output,
                    parallel=spec.parallel, agent_python=agent_python, elapsed=elapsed, recorded=recorded,
                ) or (
                    f"⚠️ PRE_PUSH_TEST_ERROR: {spec.label} pass, exit {returncode}, {elapsed:.0f}s"
                )
        if len(empty_passes) == len(passes):
            return (
                "⚠️ PRE_PUSH_TEST_ERROR: no tests were collected in any preflight pass — "
                "a repository with a tests/ directory must yield at least one runnable test"
            )
        if selection and (problem := inventory_problem(selection, inventory)):
            return f"⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_INVENTORY: {problem}"
        if ctx is not None:
            ctx._preflight_tests_passed = True
            # Missing node-lane execution cannot mint a complete proof. The
            # normal producer returns rc=0 only after its containment check.
            if (subject
                    and (selected_workload_unchanged(subject, worktree, timeout=timeout, selection=selection) if selection
                         else preflight_test_workload_unchanged(subject, worktree, timeout=timeout, pytest_args=pytest_args))
                    and (not subject.workload[4] or (node_result or {}).get("returncode") == 0)):
                ctx._preflight_test_proof = subject
                receipt.publish(ctx, temp_root, worktree, agent_python, empty_passes, phase, timings)
                log_preflight_test_proof(ctx, subject, reused=False, phase=phase, passes=timings, receipt=receipt.disclosure(),
                                         empty_lanes=empty_passes)
            if selection and ctx._preflight_test_proof is None:
                ctx._preflight_tests_passed = False
                raise ValueError("selected test scope changed during execution; no proof was minted")
        return None
    except subprocess.TimeoutExpired as exc:  # a command of the gate's own; the lanes report their budget themselves
        failure = f"⚠️ PRE_PUSH_TEST_ERROR: pytest timed out after {exc.timeout:g} seconds"
    except FileNotFoundError:
        failure = f"⚠️ PRE_PUSH_TEST_ERROR: pytest not available via interpreter: {agent_python}"
    except Exception as exc:
        failure = f"⚠️ PRE_PUSH_TEST_ERROR: hermetic preflight failed: {exc}"
    finally:
        if lease is not None:
            lease.close()
        evidence["elapsed_seconds"] = round(time.monotonic() - began, 1)
        unproven_teardown = evidence.pop("lane_teardown", unproven_teardown)
        evidence["cleanup"] = (repo, temp_root, worktree, worktree_added, unproven_teardown)
        from ouroboros.preflight_inventory import record_failed_attempt
        record_failed_attempt(ctx, selection, subject, inventory, evidence)
    return _with_retention_notice(failure, unproven_teardown, worktree, max_output)


def run_hermetic_pytest(repo_dir, *, timeout=None, pytest_args=None,
                       max_output=8000, phase="post_commit", ctx=None, selected=False, base=None,
                       extra_tests=(), full_tests=False, test_reason="") -> Optional[str]:
    """Existing gate, with durable raw evidence also on green and ctx-less calls.

    Evidence location is diagnostic, never part of the reusable test identity.
    The internal runner still owns assembly, selection, lane verdicts and green receipts.
    """
    from ouroboros.test_evidence import evidence_directory, finish_gate

    repo = pathlib.Path(repo_dir).resolve()
    try:
        evidence = {"path": evidence_directory(repo, ctx=ctx,
            parent=getattr(ctx, "_preflight_evidence_parent", None))} if (repo / ".git").exists() else None
    except OSError as exc:
        evidence = f"⚠️ PRE_PUSH_TEST_ERROR: preflight evidence destination unavailable: {exc}"
    if not isinstance(evidence, dict):
        if ctx is not None:
            ctx._preflight_tests_passed, ctx._preflight_test_proof, ctx._preflight_test_inventory = False, None, None
        return evidence
    if ctx is not None:
        ctx._preflight_previous_evidence = getattr(ctx, "_preflight_test_evidence", None)
        ctx._preflight_test_evidence = str(evidence["path"])
    print(f"PREFLIGHT_EVIDENCE {evidence['path']}", file=sys.stderr, flush=True)
    verdict = "preflight interrupted before a verdict"
    try:
        verdict = _run_hermetic_pytest(repo, timeout=timeout, pytest_args=pytest_args,
                                       max_output=max_output, phase=phase, ctx=ctx, evidence=evidence,
                                       selected=selected, base=base, extra_tests=extra_tests,
                                       full_tests=full_tests, test_reason=test_reason)
        return verdict
    finally:
        finish_gate(evidence, verdict, phase, _release_hermetic_tree, ctx=ctx)
