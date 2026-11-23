"""Format existing preflight blockers without changing their classification."""
from __future__ import annotations

import re
from typing import Optional


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


def _prepend_reading(message: str, lines: str, max_output: int) -> str:
    """Keep the classifying first line and the assertion tail around the bound reading."""
    header, sep, rest = message.partition("\n")
    if not lines or len(header) + len(lines) + 1 > max_output:
        return message
    return _diagnosis(header, lines, rest if sep else "", max_output)


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


def _rejected_parallel_flags(output: str, parallel_flags) -> list[str]:
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
            if name in parallel_flags and name not in rejected:
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
    parallel_flags=(),
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
            rejected = _rejected_parallel_flags(output, parallel_flags)
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
