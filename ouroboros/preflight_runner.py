"""Hermetic pytest preflight for reviewed repository changes."""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import NamedTuple, Optional, Sequence

# Node lane of the gate (`run_hermetic_pytest` calls it once the candidate is
# assembled, as the first consumer of the shared budget): content-keyed on the
# candidate's web/tests/*.test.js — see ouroboros/preflight_node.py. Imported
# by name so tests/operators can stub `preflight_runner.run_node_tests`.
from ouroboros.preflight_node import run_node_tests
from ouroboros.settings_defaults import settings_env_keys


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

# Mirrors ci.yml `quick-test`/`full-test` exactly, including the per-test timeout
# that converts a silent hang into a named failure.
PARALLEL_PASS_FLAGS = [
    "-n", "auto",
    "--dist", "loadscope",
    "--max-worker-restart=0",
    "--timeout=300",
    "--timeout-method=thread",
]

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

# pytest-timeout's own banner. `--timeout-method=thread` does NOT fail the test:
# it dumps stacks and `os._exit(1)`s the whole worker, which xdist then reports
# with the crash phrasing above. So a plain hang is indistinguishable from a real
# crash by exit code alone, and the generic "mark it @pytest.mark.serial"
# remediation would be actively harmful — the serial pass carries no per-test
# timeout, so obeying it relocates the hang into the pass that has no defence
# against it. These patterns split the two cases apart.
#   * thread method (pytest_timeout.timeout_timer -> write_title("Timeout", sep="+"))
#     prints a "+++++ Timeout +++++" rule; the fill width follows the terminal, and
#     xdist may prefix the forwarded line, so neither end is anchored. This is the
#     banner the configured `--timeout-method=thread` actually emits, and only
#     pytest-timeout emits it.
#   * signal method raises Failed("Timeout >300.0s") — kept so the diagnosis is
#     right if the method is ever changed. The bare phrase alone is NOT enough:
#     `Timeout >30s` is ordinary text a test can print or assert on, and matching
#     it against the whole pass output would INVERT the remediation for a genuine
#     crash that happens to run alongside such a test. So the phrase only counts
#     as pytest's own `Failed:` exception line, which is how both `--tb=line`
#     ("path:5: Failed: Timeout >300.0s") and `--tb=short` ("E   Failed: Timeout
#     >300.0s") render the signal method's exception.
_TIMEOUT_BANNER_PATTERNS = (
    re.compile(r"\+{3,} Timeout \+{3,}"),
    re.compile(r"\bFailed:\s*Timeout >\d+(?:\.\d+)?s"),
)


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
    """The gate mirrors CI: a parallel non-serial pass, then a serial pass.

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

    ``probe_module`` is the NONCE name ``_install_worker_probe`` actually wrote.
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
                target, "-m", f"not serial and {LANE_EXCLUSION_EXPR}",
                *PARALLEL_PASS_FLAGS,
                # Neither of these mirrors CI: they defend the parallel lane
                # against candidate-controlled pytest configuration, which CI
                # does not have. `-p xdist -p timeout` overrides a `no:` block in
                # the candidate's addopts; the probe module reports the worker
                # ids the run actually started. An explicit `pytest_args=` gets
                # neither — the caller's argv is forwarded verbatim, and the
                # worker-count check keys on the probe flag being present.
                *_FORCED_PLUGIN_FLAGS,
                "-p", probe_module or _WORKER_PROBE_MODULE,
                *output_flags,
            ],
        ),
        _make_pass(
            "serial",
            [target, "-m", f"serial and {LANE_EXCLUSION_EXPR}", *output_flags],
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
        # guarantee this whole capture exists to hold. No `.gitattributes text`
        # directive governs the affected paths, so the config override is
        # authoritative.
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


# The parallel lane is only a parallel lane if xdist really starts more than one
# worker. `-n auto` resolves through PYTEST_XDIST_AUTO_NUM_WORKERS, which the
# operator environment can carry: an inherited "1" runs the whole pass on a
# single worker while the argv still says `-n`, so `PreflightPass.parallel` stays
# True, the pass returns green, and the gate has proved nothing about the
# parallel-only defects it exists to catch. That value is therefore never
# inherited — `_preflight_env` scrubs the whole PYTEST_* namespace and re-injects
# the count resolved here, which is clamped so it can never fall below two.
_MIN_PREFLIGHT_WORKERS = 2
# Private test-only seam. Read from the OPERATOR environment and never forwarded
# to the candidate (the scrub removes every OUROBOROS_* key first): the nested
# fixture repos in tests/test_preflight_runner.py hold 1-3 probe tests, so a full
# `-n auto` fan-out would spend minutes on worker startup for nothing. It can
# only lower the count TO the floor, never below it.
_PREFLIGHT_WORKERS_ENV = "OUROBOROS_PREFLIGHT_TEST_WORKERS"


def _preflight_worker_count() -> int:
    """How many xdist workers the parallel pass must start — always >= 2."""
    raw = os.environ.get(_PREFLIGHT_WORKERS_ENV, "")
    if raw.strip():
        try:
            return max(_MIN_PREFLIGHT_WORKERS, int(raw.strip()))
        except ValueError:
            pass
    return max(_MIN_PREFLIGHT_WORKERS, os.cpu_count() or _MIN_PREFLIGHT_WORKERS)


# Forcing the flags and the plugins still only proves what the gate ASKED for.
# The pass is accepted as parallel only if xdist really started >= 2 workers, and
# that is observed from inside the run: a tiny plugin the gate writes into its own
# disposable temp root (never into the candidate worktree) records one file per
# worker id. Reading it back is the difference between "the argv said -n" and
# "concurrency actually happened".
#
# Scope of the guarantee: this closes the SILENT-DOWNGRADE window (an inherited
# worker count, a blocked plugin, a conftest swallowing the flags), all of which
# return green today. It is not a defence against a candidate that forges the
# evidence by writing worker-named files into the gate's temp root — nothing
# in-process can be, since the candidate suite runs with the same privileges.
#
# The name is only a STEM: `_install_worker_probe` appends a per-run nonce. `-p
# name` resolves through `sys.path`, and `python -m pytest` puts the candidate
# worktree at `sys.path[0]`, AHEAD of the PYTHONPATH entry the gate prepends. A
# repository that happens to contain a top-level module of the fixed name would
# therefore shadow the probe, report zero workers, and hard-block every green
# parallel run as PREFLIGHT_PARALLELISM_LOST. The nonce cannot be predicted by a
# committed file, so the gate's own module always wins.
_WORKER_PROBE_MODULE = "ouroboros_preflight_probe"

_WORKER_PROBE_SOURCE = '''"""Written by the Ouroboros preflight gate; not part of the repository under test."""
import os
import pathlib

_LOG = pathlib.Path(__WORKER_LOG__)


def pytest_configure(config):
    # Runs in the controller too, where the variable is unset — only workers
    # report, so the file count IS the worker count.
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker:
        return
    try:
        _LOG.mkdir(parents=True, exist_ok=True)
        (_LOG / worker).write_text(worker, encoding="utf-8")
    except OSError:
        pass
'''


def _probe_dir(temp_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(temp_root) / "probe"


def _worker_log_dir(temp_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(temp_root) / "workers"


def _install_worker_probe(temp_root: pathlib.Path) -> str:
    """Write the worker-count plugin and return the nonce module name to load.

    The nonce is what makes the probe candidate-independent: see the note on
    ``_WORKER_PROBE_MODULE``. Callers must pass the RETURNED name to `-p`.
    """
    module = f"{_WORKER_PROBE_MODULE}_{uuid.uuid4().hex}"
    probe = _probe_dir(temp_root)
    probe.mkdir(parents=True, exist_ok=True)
    (probe / f"{module}.py").write_text(
        _WORKER_PROBE_SOURCE.replace("__WORKER_LOG__", repr(str(_worker_log_dir(temp_root)))),
        encoding="utf-8",
    )
    return module


def _observed_worker_ids(temp_root: pathlib.Path) -> set:
    """Distinct xdist worker ids that reported during the parallel pass."""
    try:
        return {entry.name for entry in _worker_log_dir(temp_root).iterdir() if entry.is_file()}
    except OSError:
        return set()


def _preflight_env(temp_root: pathlib.Path, repo_worktree: pathlib.Path) -> dict:
    env = dict(os.environ)
    # The candidate suite must not inherit live runtime behaviour or credentials.
    # A disposable data/settings/repo triple is injected below; every other
    # OUROBOROS_* value is owner/runtime state, not test wiring. Keeping those
    # values made a supposedly hermetic preflight depend on the operator's live
    # safety/review/mode settings and could also expose prefixed secrets to a
    # self-written test.
    secret_suffixes = ("_API_KEY", "_TOKEN", "_PASSWORD", "_CREDENTIALS", "_SECRET")
    # Every key config.apply_settings_to_env projects from settings.json is the
    # same owner state under a name the prefix/suffix rules miss (provider base
    # URLs, USE_LOCAL_*, LOCAL_MODEL_*, MCP_*, GITHUB_REPO, TOTAL_BUDGET): the
    # suite routes on OPENAI_COMPATIBLE_BASE_URL alone. Derived, not hand-listed.
    projected = frozenset(settings_env_keys())
    for key in list(env):
        if (
            key.startswith("OUROBOROS_")
            or key.endswith(secret_suffixes)
            or key in projected
            or key.startswith("GH_")
            # Externally supplied pytest/xdist controls are dropped WHOLESALE
            # rather than by name, because every one of them can weaken the pass
            # while the argv still reads like a full parallel run:
            # PYTEST_XDIST_AUTO_NUM_WORKERS decides what `-n auto` resolves to,
            # PYTEST_ADDOPTS can append `-p no:xdist` or its own `-m`,
            # PYTEST_PLUGINS / PYTEST_DISABLE_PLUGIN_AUTOLOAD decide whether the
            # verified plugins load at all, and PYTEST_XDIST_WORKER /
            # PYTEST_XDIST_TESTRUNUID / PYTEST_CURRENT_TEST leak the OUTER run's
            # identity into the nested one. A green pass under any of those is
            # indistinguishable from a green pass under the real gate.
            or key.startswith("PYTEST_")
        ):
            env.pop(key, None)
    temp_root = pathlib.Path(temp_root).resolve(strict=False)
    repo_worktree = pathlib.Path(repo_worktree).resolve(strict=False)
    data_dir = (temp_root / "data").resolve(strict=False)
    env["OUROBOROS_DATA_DIR"] = str(data_dir)
    env["OUROBOROS_SETTINGS_PATH"] = str(data_dir / "settings.json")
    env["OUROBOROS_REPO_DIR"] = str(repo_worktree)
    env["PYTHONPYCACHEPREFIX"] = str((temp_root / "pycache").resolve(strict=False))
    # PREPENDED, so `-p ouroboros_preflight_probe` resolves to the gate's own
    # worker-count plugin and not to anything the candidate tree or the
    # operator's PYTHONPATH happens to shadow it with.
    inherited_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_probe_dir(temp_root)) + (
        os.pathsep + inherited_path if inherited_path else ""
    )
    # Re-injected AFTER the scrub, so `-n auto` resolves to a count this process
    # chose rather than one the operator environment happened to carry. Inert for
    # the serial/legacy passes, which pass no `-n` at all.
    env["PYTEST_XDIST_AUTO_NUM_WORKERS"] = str(_preflight_worker_count())
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
    probe_dir: pathlib.Path,
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
            argv, cwd=str(probe_dir), capture_output=True, text=True, timeout=60
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


def _verify_preflight_plugins(agent_python: str, probe_dir: pathlib.Path) -> list[str]:
    """Prove the SELECTED interpreter really carries the parallel-pass plugins.

    Independent of the candidate on purpose. Inferring "the plugins are there"
    from the candidate's own pytest accepting ``-n``/``--dist``/``--timeout`` is
    not a check: a conftest can declare those exact option names with
    ``pytest_addoption`` and ignore them, so with xdist or pytest-timeout absent
    the nominal parallel lane would run serially, exit 0, and return green while
    proving nothing.

    The candidate-controlled import surface is the working directory, so the
    probe runs from ``probe_dir`` — the disposable temp root, before the
    worktree even exists — and never from the diff-applied tree. No candidate
    conftest, sitecustomize or plugin can run before the answer is known.

    Returns the list of problems; an empty list means verified.
    """
    problems = _probe_plugins(agent_python, probe_dir, isolated=True)
    if not problems:
        return []
    # `-I` additionally drops PYTHONPATH and the user site directory, which the
    # real pass DOES honour. Neither is candidate-controlled — both come from the
    # operator's own environment — so a strict miss is re-confirmed without them
    # rather than hard-blocking every commit on a legitimate `pip install --user`.
    return _probe_plugins(agent_python, probe_dir, isolated=False)


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


_DEFAULT_PREFLIGHT_TIMEOUT_SEC = 900


def _resolve_preflight_timeout(timeout: int) -> int:
    """Env override (`OUROBOROS_PREFLIGHT_TIMEOUT_SEC`) takes precedence so the
    timeout is one SSOT across callers without editing each call site."""
    raw = os.environ.get("OUROBOROS_PREFLIGHT_TIMEOUT_SEC")
    if raw:
        try:
            parsed = int(float(raw))
            if parsed > 0:
                return parsed
        except (TypeError, ValueError, OverflowError):
            pass
    return timeout


def _terminate_preflight_tree(proc: "subprocess.Popen", temp_root: pathlib.Path) -> None:
    """Reap pytest and its WHOLE tree on timeout/crash.

    ``killpg`` alone reaps only pytest's own process group; descendants that
    started their own session/group (Ouroboros spawns servers/browsers/extension
    children with new sessions) or double-forked to init survive it. So collect
    the live descendant PIDs and their group ids FIRST (the ``pgrep -P`` chain
    breaks once pytest dies and children reparent), then kill pytest's group, the
    recursive PID tree, each captured escapee group, and finally sweep any
    straggler still rooted under the disposable temp root. All platform-specific
    process discovery/termination lives behind platform_layer helpers."""
    from ouroboros.platform_layer import (
        IS_WINDOWS,
        collect_descendant_pids,
        kill_pid_tree,
        kill_process_group_id,
        kill_process_tree,
        kill_processes_referencing,
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
    kill_processes_referencing(str(temp_root))


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


def _execute_pytest_pass(
    agent_python: str,
    worktree: pathlib.Path,
    temp_root: pathlib.Path,
    args: Sequence[str],
    timeout: float,
) -> tuple[Optional[int], str, str]:
    """Run ONE pytest invocation in the prepared worktree.

    ``timeout`` is the caller's exact remaining share of the TOTAL budget and
    stays a float — rounding it up would let the last pass outlive the total.

    Returns ``(returncode, combined_output, containment_error)``.
    ``returncode is None`` means the pass exhausted its slice of the budget and
    its whole process tree was reaped. The output of a timed-out pass is whatever
    the child had flushed before the kill — pass 2 carries no per-test timeout,
    so a serial hang is exactly the case with no other evidence of WHICH test
    hung.

    A NON-EMPTY ``containment_error`` is a hard block the caller must not accept
    as green, whatever the return code says: the container's post-pass SCAN found
    a process the pass spawned still alive, or could not determine whether one is
    (which counts the same), so a green pass may be sitting on top of a live tree
    that survives into pass 2 and past teardown. The reap therefore happens before
    the return rather than in the `finally` below, which cannot alter an
    already-computed tuple; only the handle release stays unconditional.
    """
    from ouroboros.process_containment import ProcessContainer

    container = ProcessContainer()
    # `container.spawn`, not `Popen` + `adopt`: it applies the process-group/session
    # kwargs AND leaves no window between the process existing and being contained.
    # On Windows job membership only takes effect at assignment, so a descendant
    # started in that window is outside the job and survives terminate/close.
    proc = container.spawn(
        [agent_python, "-m", "pytest", *args],
        cwd=str(worktree),
        env=_preflight_env(temp_root, worktree),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    reap_error = ""
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            returncode: Optional[int] = proc.returncode
            output = (stdout or "") + (stderr or "")
        except subprocess.TimeoutExpired as first_timeout:
            _terminate_preflight_tree(proc, temp_root)
            # Seeded from the FIRST timeout, which already carries everything
            # pytest had flushed. The retry below exists to collect more, not to
            # be the only source: if it times out too — the exact case this path
            # anticipates — an empty seed threw away the one excerpt naming the
            # last test pytest started.
            partial = (
                _partial_stream_text(first_timeout.output)
                + _partial_stream_text(first_timeout.stderr)
            )
            # Reaped BEFORE the retry, not only after it: the retry blocks on an
            # escaped grandchild still holding the inherited stdout/stderr pipe,
            # and that grandchild is precisely what the container can kill.
            reap_error = container.reap()
            try:
                stdout, stderr = proc.communicate(timeout=10)
                # Only a RICHER excerpt replaces the seed; a retry that returns
                # nothing must not erase what the first timeout captured.
                partial = ((stdout or "") + (stderr or "")) or partial
            except subprocess.TimeoutExpired:
                for stream in (proc.stdout, proc.stderr):
                    try:
                        if stream is not None:
                            stream.close()
                    except Exception:
                        pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
            returncode, output = None, partial
        # UNCONDITIONAL, including after a green pass, and BEFORE the return so
        # the verdict can carry it. `communicate` returning only proves the pytest
        # CONTROLLER exited: a test that spawned a detached child with redirected
        # stdio and a temp-path-free argv is invisible to both the `pgrep -P` walk
        # (its parent is gone, so the ppid links are gone with it) and the
        # command-line sweep. The container still names it, through the membership
        # token the kernel copied into every descendant's environment — including a
        # `setsid()` escapee's — resolved from the live process table HERE rather
        # than sampled earlier, so a child born and orphaned in the same instant is
        # not missed. The scan is what keeps pass 1 out of pass 2: it kills what it
        # can and BLOCKS on whatever is left. A second call is inert (the handles
        # are consumed by the first), so the timeout path's earlier reap stands.
        if not reap_error:
            reap_error = container.reap()
        return returncode, output, reap_error
    finally:
        # A crash/exception path (not only timeout) can leak a detached child;
        # never hand a live tree back to the orchestrator.
        try:
            if proc.poll() is None:
                _terminate_preflight_tree(proc, temp_root)
        except Exception:
            pass
        # Reached with the reap already done on every returning path; it still
        # runs on the raising one, where no verdict can carry the reason anyway.
        container.reap()
        container.close()


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

    A timed-out pass produces no exit code to classify, and pass 2 carries no
    per-test timeout, so without this the operator is told only that "the serial
    pass timed out" — with nothing naming the test that hung. The TAIL is kept
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


def _crash_remediation(output: str) -> str:
    """The instruction line for a dead worker — it depends on WHY it died.

    A worker killed by ``--timeout=300 --timeout-method=thread`` reaches the
    controller with the same crash phrasing as a genuine crash, but the fix is
    the opposite one: the serial pass runs flag-free (no per-test timeout), so
    marking a merely-slow test ``@pytest.mark.serial`` moves the hang into the
    pass that cannot bound it, where it burns the whole remaining total budget
    and resurfaces as a pass-2 timeout. The label and the hard block are
    identical either way; only this line changes.
    """
    if any(pattern.search(output) for pattern in _TIMEOUT_BANNER_PATTERNS):
        return (
            "A test exceeded the 300s per-test limit and pytest-timeout killed its worker "
            "(--timeout-method=thread terminates the process instead of failing the test). "
            "Make that test faster or split it. Do NOT mark it @pytest.mark.serial: the "
            "serial pass carries no per-test timeout, so the hang would simply move there "
            "and consume the rest of the total budget."
        )
    return (
        "A pytest-xdist worker DIED instead of reporting a failure. Find the test that "
        "spawns a real process, binds a real port, or mutates a module global and "
        "mark it @pytest.mark.serial; never a flake/retry."
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
) -> Optional[str]:
    """Turn one pass's exit code + output into ``None`` or a bounded diagnosis.

    Pure function: no process, no filesystem. Exit 0 and exit 5
    (nothing collected for this lane) are green here; the orchestrator alone
    decides that ALL passes being empty is a block.
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
                _crash_remediation(output),
                "\n".join(matched) + "\n\n" + output, max_output,
            )

    return _diagnosis(f"⚠️ PRE_PUSH_TEST_ERROR: {head}", "", output, max_output)


def _capture_source_index_tree(
    repo: pathlib.Path,
    max_output: int,
) -> tuple[str | None, str | None]:
    """Return a merged index tree, or ``None`` only for proven unmerged entries."""
    try:
        source_index = _run_git(repo, ["write-tree"])
    except (subprocess.SubprocessError, OSError) as exc:
        return None, _diagnosis(
            "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_SOURCE_INDEX (hard block): "
            "the source index could not be snapshotted",
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
            "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_SOURCE_INDEX (hard block): "
            "the failed source index could not be classified",
            "The preflight may preserve live resolution semantics only for a proven "
            "unmerged index. No test pass was run.",
            str(exc),
            max_output,
        )
    if unmerged.returncode == 0 and str(unmerged.stdout).strip():
        return None, None
    return None, _diagnosis(
        "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_SOURCE_INDEX (hard block): "
        "the source index could not be snapshotted",
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
    """Install and verify the immutable source index without updating files."""
    try:
        read_tree = _run_git(worktree, ["read-tree", source_index_tree])
    except (subprocess.SubprocessError, OSError) as exc:
        return _diagnosis(
            "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_SOURCE_INDEX (hard block): "
            "the source index tree could not be installed",
            "The disposable index must exactly reproduce the source index before "
            "tests run. No candidate test pass was started.",
            str(exc),
            max_output,
        )
    if read_tree.returncode != 0:
        return _diagnosis(
            "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_SOURCE_INDEX (hard block): "
            "the source index tree could not be installed",
            "The disposable index must exactly reproduce the source index before "
            "tests run. No candidate test pass was started.",
            read_tree.stderr.strip() or "git read-tree failed",
            max_output,
        )
    try:
        projected_index = _run_git(worktree, ["write-tree"])
    except (subprocess.SubprocessError, OSError) as exc:
        return _diagnosis(
            "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_SOURCE_INDEX (hard block): "
            "the disposable index could not be verified",
            "The source and disposable indexes must have identical `write-tree` "
            "identities before tests run. No candidate test pass was started.",
            str(exc),
            max_output,
        )
    if projected_index.returncode == 0 and str(projected_index.stdout).strip() == source_index_tree:
        return None
    detail = projected_index.stderr.strip() or (
        f"expected {source_index_tree}, got {str(projected_index.stdout).strip() or '<empty>'}"
    )
    return _diagnosis(
        "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_SOURCE_INDEX (hard block): "
        "the disposable index does not match the source index tree",
        "The source and disposable indexes must have identical `write-tree` "
        "identities before tests run. No candidate test pass was started.",
        detail,
        max_output,
    )


def run_hermetic_pytest(
    repo_dir: pathlib.Path | str,
    *,
    timeout: int = _DEFAULT_PREFLIGHT_TIMEOUT_SEC,
    pytest_args: Optional[Sequence[str]] = None,
    max_output: int = 8000,
    phase: str = "post_commit",
) -> Optional[str]:
    """Run pytest against the candidate diff in a disposable worktree.

    Mirrors CI: the web node lane (``preflight_node``, on candidates carrying
    ``web/tests/*.test.js``), then a parallel ``not serial`` pass, then a
    ``serial`` pass — one worktree/env, ONE shared total budget. Fails fast on
    the first red lane, so the output never truncates the failing section away.

    Returns ``None`` on success, otherwise a bounded human-readable error.
    ``OUROBOROS_PREFLIGHT_TIMEOUT_SEC`` overrides ``timeout`` for all callers.

    ``phase`` selects the deleted-suite baseline (see ``_TESTS_BASELINE_REFS``):
    the pre-commit review passes ``PRE_COMMIT_PHASE`` so it compares against HEAD
    only, while the default post-commit verification also consults HEAD~1,
    because by then the deletion it is looking for is already in HEAD.

    The candidate is assembled as ONE hardened ``git diff --binary … HEAD``
    capture (external drivers, textconv, colour and operator prefix configs
    pinned off; payload kept as raw bytes end to end) applied to a clean
    worktree at HEAD, plus a copy of the untracked files: an exact tracked
    projection of the live worktree plus its safe non-ignored untracked
    entries — for every source-index state, including a merge in progress,
    whose unmerged entries the former staged+unstaged diff pair rendered as
    stubs and ``--cc`` hunks that ``git apply`` dropped or rejected. The
    untracked side keeps ``_copy_untracked``'s long-standing boundaries:
    ignored files are absent, and untracked symlinks are dereferenced to
    regular files (non-file entries skipped), so the candidate is not
    literally byte-equal to the worktree in those corners. When the source index
    is merged, its exact ``write-tree`` snapshot is also installed into the
    disposable index (without updating files) and verified before tests run.
    A genuinely unmerged source index retains the live-resolution projection.
    """
    timeout = _resolve_preflight_timeout(timeout)
    # Checked BEFORE anything runs. `_diagnosis` renders inside this budget, so a
    # non-positive one produced an empty string for a real failure — which the
    # loop below reads as "no diagnosis", i.e. as success. A budget that cannot
    # render a failure must stop the gate, never silently pass it.
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
        # A repo that never had tests is out of scope; a candidate that DELETES
        # the whole suite is not. Git does not track empty directories, so
        # staging the removal of every test file removes `tests/` with them and
        # reached this early return — skipping the passes entirely, so the
        # all-passes-empty hard block below never ran and the change that
        # deleted the gate sailed through it.
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
    agent_python = os.environ.get("OUROBOROS_AGENT_PYTHON") or sys.executable or "python3"
    source_index_tree, source_index_error = _capture_source_index_tree(repo, max_output)
    if source_index_error is not None:
        return source_index_error
    temp_root_path = tempfile.mkdtemp(prefix="ouroboros-preflight-")
    temp_root = pathlib.Path(temp_root_path).resolve(strict=False)
    worktree = temp_root / "repo"
    worktree_added = False
    try:
        # The probe is written BEFORE the specs are built, and the nonce name it
        # returns is what the parallel pass loads. Building the specs first and
        # discarding the name left `-p` pointing at a stem no file carries, which
        # is a pytest usage error, not a run.
        probe_module = _install_worker_probe(temp_root)
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

        # `-c core.autocrlf=false -c core.eol=lf`: check HEAD out byte-for-byte so
        # the hermetic worktree matches the bytes the `--binary` candidate diff was
        # captured against. Without it a Windows runner (`core.autocrlf=true`)
        # materializes HEAD with CRLF, and the LF candidate then applies onto a
        # CRLF base — the capture is byte-faithful but the checkout it lands on is
        # not. Paired with the same override on `_apply_diff`.
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
            source_index_error = _install_source_index_tree(worktree, source_index_tree, max_output)
            if source_index_error is not None:
                return source_index_error

        # ONE capture for every repository state: the tracked delta between
        # HEAD and the live worktree, assembled identically whether the source
        # index is clean, dirty, or mid-merge. The staged+unstaged diff pair
        # this replaces could not represent an unmerged index at all: `git diff
        # --cached` renders each conflicted path as a literal "* Unmerged path"
        # stub and `git diff` as a combined `--cc` hunk — which `git apply`
        # REJECTS when the payload holds nothing else (rc=128, the gate died
        # before running a test) and silently DROPS when ordinary hunks
        # accompany it (the gate then ran against a candidate MISSING the
        # resolutions, so its verdict described a tree nobody has). The two-way
        # HEAD form has no such rendering: staged-only files, resolutions and
        # conflict markers all arrive as plain content.
        #
        # The flag tail pins away every operator config that reshapes diff
        # output into something `git apply` cannot re-apply: external diff
        # drivers (`--no-ext-diff`), textconv filters (`--no-textconv`), colour
        # escapes (`--no-color`), and prefix rewrites (`--src-prefix=a/
        # --dst-prefix=b/` — the explicit CLI prefixes win over diff.noprefix
        # AND diff.srcPrefix/dstPrefix, which `-c diff.noprefix=false` alone
        # would not). Captured and applied as BYTES end to end (see
        # ``binary_stdout``) so NUL-free non-UTF-8 text content is not
        # U+FFFD-substituted in transit.
        try:
            combined_proc = _run_git(
                repo,
                ["diff", "--binary", "--no-ext-diff", "--no-textconv", "--no-color",
                 "--src-prefix=a/", "--dst-prefix=b/", "HEAD"],
                binary_stdout=True,
            )
            if combined_proc.returncode != 0:
                raise RuntimeError(combined_proc.stderr.strip() or "git diff HEAD failed")
            _apply_diff(worktree, combined_proc.stdout or b"")
            _copy_untracked(repo, worktree)
        # The assembly block owns EVERY way its own capture can fail, not only
        # the RuntimeErrors it raises itself: `_run_git` can raise
        # subprocess.TimeoutExpired (a SubprocessError subclass) and
        # `_copy_untracked` can raise FileNotFoundError/PermissionError (OSError
        # subclasses). Let through, those land in the OUTER handlers below and
        # are misread as a pytest timeout, a missing pytest interpreter, or a
        # generic preflight failure — all of which invite a retry against a
        # candidate that was never assembled.
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
        from ouroboros.platform_layer import kill_processes_referencing
        started = time.monotonic()
        if node_error := (run_node_tests(worktree, temp_root, timeout, max_output) or {}).get("error"):
            return node_error
        empty_passes = 0
        for spec in passes:
            # ONE total budget across passes; the later pass gets the exact
            # float remainder. Never clamped up to a whole second: `max(1, ...)`
            # would hand an already-exhausted budget another second, and int()
            # truncation would round a 0.9s remainder up to 1s — both let the
            # gate outrun the total it advertises.
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                return (
                    f"⚠️ PRE_PUSH_TEST_ERROR: the {spec.label} pass never started — the "
                    f"total budget of {timeout} seconds was exhausted by the earlier pass(es)"
                )
            # `started` sizes the shared budget; the header reports THIS pass's
            # own duration, or a fast serial pass would be blamed for the whole
            # gate's wall-clock after a slow parallel one.
            pass_started = time.monotonic()
            returncode, output, reap_error = _execute_pytest_pass(
                agent_python, worktree, temp_root, spec.args, remaining
            )
            elapsed = time.monotonic() - pass_started
            # Sweep between passes so a pass-1 escapee cannot touch pass 2.
            kill_processes_referencing(str(temp_root))
            if reap_error:
                # Checked BEFORE the exit code, including before the green path:
                # the scan says processes the pass spawned are still running (or
                # that it cannot tell), and an exit 0 taken on top of that is
                # exactly the fail-open the container exists to close. It is also
                # the more urgent report — a live tree outlives this whole run.
                return _diagnosis(
                    "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_CONTAINMENT_FAILED (hard block): "
                    f"the {spec.label} pass ran, but processes it spawned were still "
                    "alive afterwards (or could not be determined to be gone)",
                    "The container DETECTS leaked processes; it does not promise to kill "
                    "them, and a process it cannot inspect counts as leaked. The pass "
                    "verdict is not accepted, whatever it was, because a leaked tree "
                    "survives into the next pass and past teardown. Find the test that "
                    "spawns a real process, binds a real port, or daemonises a helper "
                    "and does not wait for it — make it clean up, and mark it "
                    "@pytest.mark.serial if it must own real processes; never a "
                    "flake/retry. Kill any pid named below before re-running; when the "
                    "report names none it states the reason instead — read that line, "
                    "since more than one cause leaves no pid to name.",
                    reap_error, max_output,
                )
            if returncode is None:
                return _with_timeout_excerpt(
                    f"⚠️ PRE_PUSH_TEST_ERROR: pytest timed out after {remaining:.0f} seconds "
                    f"in the {spec.label} pass (total budget {timeout} seconds)",
                    output, max_output,
                )
            if returncode == _PYTEST_EXIT_NO_TESTS:
                empty_passes += 1
                continue
            if returncode == 0 and probe_module in spec.args:
                # A GREEN parallel pass is the only place this can go unnoticed:
                # a red one blocks anyway, and an empty one had nothing to
                # distribute. The probe flag is the key (not `spec.parallel`) so
                # a caller-supplied argv carrying its own `-n` — which never gets
                # the probe — is not blocked for evidence it was never asked for.
                # Keyed on the NONCE name that was actually threaded into the
                # argv: testing the bare stem here would never match again, and
                # would silently disable the very block the nonce protects.
                observed = _observed_worker_ids(temp_root)
                if len(observed) < _MIN_PREFLIGHT_WORKERS:
                    return _diagnosis(
                        "⚠️ PRE_PUSH_TEST_ERROR: PREFLIGHT_PARALLELISM_LOST (hard block): "
                        f"the {spec.label} pass returned green on {len(observed)} xdist "
                        f"worker(s), fewer than the {_MIN_PREFLIGHT_WORKERS} it requires",
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
                failure = _classify_pass_result(
                    spec.label, returncode, output, max_output,
                    parallel=spec.parallel, agent_python=agent_python, elapsed=elapsed,
                )
                return failure or (
                    f"⚠️ PRE_PUSH_TEST_ERROR: {spec.label} pass, exit {returncode}, {elapsed:.0f}s"
                )
        if empty_passes == len(passes):
            return (
                "⚠️ PRE_PUSH_TEST_ERROR: no tests were collected in any preflight pass — "
                "a repository with a tests/ directory must yield at least one runnable test"
            )
        return None
    except subprocess.TimeoutExpired:
        return f"⚠️ PRE_PUSH_TEST_ERROR: pytest timed out after {timeout} seconds"
    except FileNotFoundError:
        return f"⚠️ PRE_PUSH_TEST_ERROR: pytest not available via interpreter: {agent_python}"
    except Exception as exc:
        return f"⚠️ PRE_PUSH_TEST_ERROR: hermetic preflight failed: {exc}"
    finally:
        from ouroboros.platform_layer import kill_processes_referencing
        kill_processes_referencing(str(temp_root))
        if worktree_added:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=30,
            )
        shutil.rmtree(temp_root, ignore_errors=True)
