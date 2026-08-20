"""What the gate concludes from a red pass, and what it refuses to blame.

Split verbatim out of ``tests/test_preflight_runner.py`` by theme. This module owns the
plugin-missing and xdist diagnoses, the crash patterns and the terminal decoration they
survive, the remediation a per-test timeout may not suggest, and the output budget a
diagnosis may never overrun.
"""

from __future__ import annotations

import inspect

import pytest



def test_classify_plugin_missing():
    from ouroboros.preflight_runner import _classify_pass_result

    output = (
        "ERROR: usage: pytest [options] [file_or_dir]\n"
        "pytest: error: unrecognized arguments: -n auto --dist loadscope\n"
    )
    result = _classify_pass_result(
        "parallel", 4, output, 8000, parallel=True, agent_python="/opt/py/bin/python3"
    )
    assert result is not None
    assert "PREFLIGHT_PLUGIN_MISSING" in result
    assert "/opt/py/bin/python3" in result
    assert "pytest-xdist" in result and "pytest-timeout" in result

def test_classify_does_not_blame_plugins_for_an_unrelated_usage_error():
    """`-n` is a SUBSTRING of `--no-header`, which `DEFAULT_PYTEST_ARGS` passes on
    every invocation, so a substring test blamed missing xdist for any usage
    error at all. Only a whole-token match against the parallel flags counts."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = "pytest: error: unrecognized arguments: --no-header\n"
    result = _classify_pass_result("parallel", 4, output, 8000, parallel=True)
    assert result is not None, "a usage error still blocks"
    assert "PREFLIGHT_PLUGIN_MISSING" not in result
    assert "pytest-xdist" not in result

def test_classify_skips_xdist_diagnoses_on_a_non_parallel_pass():
    """The serial and legacy passes carry no `-n`/`--dist`, so they have no
    workers to crash and no xdist plugin to miss. Both labels would still block
    (nonzero exit), but with a remediation that is wrong by construction."""
    from ouroboros.preflight_runner import _classify_pass_result

    crash_text = "[gw0] node down: Not properly terminated\nworker gw0 crashed while running 'x'\n"
    result = _classify_pass_result("serial", 1, crash_text, 8000, parallel=False)
    assert result is not None
    assert "PARALLEL_WORKER_CRASH" not in result
    assert "@pytest.mark.serial" not in result, "cannot ask a serial-lane test to be marked serial"
    assert "node down" in result, "the raw pytest output is still reported"

    usage = "pytest: error: unrecognized arguments: -n auto\n"
    assert "PREFLIGHT_PLUGIN_MISSING" not in _classify_pass_result("single", 4, usage, 8000, parallel=False)

@pytest.mark.parametrize("returncode,output,label,remediation_marker,body_marker", [
    (4, "pytest: error: unrecognized arguments: -n auto --dist loadscope\n",
     "PREFLIGHT_PLUGIN_MISSING", "pyproject.toml", "unrecognized arguments"),
    (1, "worker gw3 crashed while running 'tests/test_x.py::test_y'\n",
     "PARALLEL_WORKER_CRASH", "@pytest.mark.serial", "worker gw3 crashed"),
])
def test_hard_block_remediation_survives_caller_truncation(
    returncode, output, label, remediation_marker, body_marker
):
    """`review_helpers._run_review_preflight_tests` re-truncates this string from
    the TAIL at the same 8000 limit (ouroboros/utils.py::truncate_review_artifact),
    so a remediation emitted AFTER a full-budget body is the first thing
    destroyed — exactly when the output is long enough to need it."""
    from ouroboros.preflight_runner import _classify_pass_result

    # Comfortably larger than the longest remediation (~430 chars) so this test
    # keeps pinning ORDER, not the exact prose length; the too-small-budget
    # extreme is pinned separately by
    # `test_diagnosis_never_overruns_a_declared_max_output`.
    max_output = 1200
    noisy = output + ("E   assert 0 == 1  # routine failing-suite noise\n" * 200)
    result = _classify_pass_result("parallel", returncode, noisy, max_output, parallel=True)

    assert result is not None
    assert label in result
    assert len(result) <= max_output, f"diagnosis overran the caller's {max_output}-char budget"
    assert result.index(remediation_marker) < result.index(body_marker), (
        "the remediation must precede the pytest body, or a tail cut removes it first"
    )

def test_per_test_timeout_kill_is_not_told_to_mark_the_test_serial():
    """`--timeout-method=thread` does not FAIL a slow test — it `os._exit`s the
    whole worker, which xdist reports with its crash phrasing. The generic
    remediation would then tell the author to mark a merely-slow test
    `@pytest.mark.serial`, and the serial pass carries NO per-test timeout, so
    obeying it moves the hang into the one pass that cannot bound it. Same hard
    block, different instruction."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = (
        "+++++++++++++++++++++++++++ Timeout +++++++++++++++++++++++++++\n"
        "~~~~~~~~~~~~~~ Stack of MainThread (123) ~~~~~~~~~~~~~~\n"
        '  File "tests/test_slow.py", line 4, in test_slow\n'
        "    time.sleep(600)\n"
        "+++++++++++++++++++++++++++ Timeout +++++++++++++++++++++++++++\n"
        "[gw0] node down: Not properly terminated\n"
        "worker gw0 crashed and worker restarting disabled\n"
    )
    result = _classify_pass_result("parallel", 1, output, 8000, parallel=True)

    assert result is not None
    assert "PARALLEL_WORKER_CRASH" in result, "a killed worker is still a hard block"
    assert "300s per-test limit" in result
    assert "faster or split" in result
    assert "Do NOT mark it @pytest.mark.serial" in result, (
        "the serial pass has no per-test timeout, so the wrong instruction relocates the hang"
    )
    assert "Find the test that spawns a real process" not in result, (
        "the generic crash remediation must not be emitted for a timeout kill"
    )
    assert "never a flake/retry" not in result
    # The evidence the author needs is still there.
    assert "time.sleep(600)" in result

def test_signal_method_timeout_banner_also_avoids_the_serial_remediation():
    """The signal method spells the same event `Failed: Timeout >300.0s`. Pinned
    so the diagnosis stays right if `--timeout-method` is ever changed."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = "E   Failed: Timeout >300.0s\nworker gw2 crashed while running 'tests/test_slow.py::test_slow'\n"
    result = _classify_pass_result("parallel", 1, output, 8000, parallel=True)

    assert "PARALLEL_WORKER_CRASH" in result
    assert "Find the test that spawns a real process" not in result
    assert "300s per-test limit" in result

def test_genuine_crash_still_gets_the_mark_it_serial_remediation():
    """The timeout branch must not swallow the ordinary case: a worker that dies
    with no pytest-timeout banner is the real-process/port/global-state class,
    and `@pytest.mark.serial` IS the fix for it."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = "worker gw0 crashed and worker restarting disabled\n"
    result = _classify_pass_result("parallel", 1, output, 8000, parallel=True)

    assert "PARALLEL_WORKER_CRASH" in result
    assert "@pytest.mark.serial" in result
    assert "never a flake/retry" in result
    assert "300s per-test limit" not in result

def test_crash_diagnosis_keeps_the_full_pytest_output():
    """A crash label must never COST the reader the report. The matched xdist
    lines are a highlighted prefix, not a replacement: a worker usually dies
    alongside ordinary failures, and a pattern false positive would otherwise
    delete every real failure line the author needs."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = (
        "tests/test_a.py::test_one FAILED\n"
        "[gw0] node down: Not properly terminated\n"
        "tests/test_b.py::test_two FAILED\n"
        "E   assert 0 == 1  # the failure that actually explains the crash\n"
    )
    result = _classify_pass_result("parallel", 1, output, 8000, parallel=True)

    assert result is not None
    assert "PARALLEL_WORKER_CRASH" in result
    assert "node down" in result
    for survivor in ("test_one FAILED", "test_two FAILED", "the failure that actually explains"):
        assert survivor in result, f"crash diagnosis discarded {survivor!r}"

def test_crash_patterns_ignore_a_bare_worker_id_in_test_text():
    """A bare `worker gwN` appears in ordinary assertion text and captured logs.
    Only xdist's controller phrasing counts, so a routine failure keeps its
    ordinary report and its ordinary (absent) remediation."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = (
        "tests/test_pool.py::test_scheduling FAILED\n"
        "E   AssertionError: assert 'worker gw1' in queue_label\n"
    )
    result = _classify_pass_result("parallel", 1, output, 8000, parallel=True)

    assert result is not None, "an ordinary failure still blocks"
    assert "PARALLEL_WORKER_CRASH" not in result
    assert "@pytest.mark.serial" not in result
    assert "test_scheduling FAILED" in result

@pytest.mark.parametrize("crash_line", [
    "worker 'gw0' crashed while running 'tests/test_x.py::test_y'",
    "[gw0] node down: Not properly terminated",
    # The parallel pass runs with `--max-worker-restart=0`, so THIS is the
    # phrasing xdist actually emits for the configuration the gate uses.
    "worker gw0 crashed and worker restarting disabled",
    "replacing crashed worker gw1",
    "Maximum crashed workers reached: 0",
])
def test_crash_patterns_cover_xdist_controller_phrasing(crash_line):
    """Tightening the patterns away from a bare worker id must not lose the
    lines xdist really prints — especially the restart-disabled variant that
    `--max-worker-restart=0` selects."""
    from ouroboros.preflight_runner import _classify_pass_result

    result = _classify_pass_result("parallel", 1, crash_line + "\n", 8000, parallel=True)
    assert result is not None
    assert "PARALLEL_WORKER_CRASH" in result, f"unrecognised xdist crash line: {crash_line!r}"

def test_crash_patterns_survive_terminal_decoration():
    """pytest/xdist colour their output. A pattern matched against the raw line
    would miss a controller line wrapped in SGR escapes — a silent downgrade to
    the generic diagnosis for exactly the coloured terminals humans use."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = "\x1b[31m[gw0] node down: Not properly terminated\x1b[0m\n"
    result = _classify_pass_result("parallel", 1, output, 8000, parallel=True)
    assert "PARALLEL_WORKER_CRASH" in result

@pytest.mark.parametrize("innocent_line", [
    # Every crash PHRASE, in text a passing/failing test can legitimately emit.
    "E   AssertionError: assert 'node down: pool drained' in status_banner",
    "E   AssertionError: assert log == 'crashed while running the migration'",
    "INFO     scheduler:pool.py:88 replacing crashed worker in the pool",
    "WARNING  scheduler:pool.py:91 maximum crashed workers reached; giving up",
    "E   ValueError: worker gw1 crashed",
])
def test_crash_patterns_need_the_whole_controller_line_shape(innocent_line):
    """The patterns are UNANCHORED (xdist re-emits `handle_crashitem` mid-line in
    the `-q` short summary), so a free-substring match would label any test that
    reasons about worker pools a `PARALLEL_WORKER_CRASH` and hand its author a
    mark-it-serial instruction the marker cannot satisfy. Only the complete
    shape — phrase plus the worker id or numeric operand xdist always prints —
    counts."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = f"tests/test_pool.py::test_scheduling FAILED\n{innocent_line}\n"
    result = _classify_pass_result("parallel", 1, output, 8000, parallel=True)

    assert result is not None, "an ordinary failure still blocks"
    assert "PARALLEL_WORKER_CRASH" not in result, f"false crash label from: {innocent_line!r}"
    assert "@pytest.mark.serial" not in result
    assert "test_scheduling FAILED" in result

def test_crash_pattern_still_matches_the_mid_line_short_summary_form():
    """`handle_crashitem` reports the crash as a TestReport longrepr, so under
    `-q` pytest re-emits it inside the short-summary line. Pinned because it is
    the reason the patterns may not be `^`-anchored — the real-spawn regression
    `test_worker_crash_is_hard_block` depends on this exact shape."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = "FAILED tests/test_x.py::test_y - worker 'gw0' crashed while running 'tests/test_x.py::test_y'\n"
    result = _classify_pass_result("parallel", 1, output, 8000, parallel=True)
    assert "PARALLEL_WORKER_CRASH" in result

def test_a_genuine_crash_is_not_reclassified_by_timeout_text_elsewhere_in_the_output():
    """`_crash_remediation` INVERTS the instruction, so its patterns must be as
    tight as the crash patterns. Matching a bare `Timeout >30s` anywhere in the
    pass output let one unrelated test's assertion text rewrite a real crash's
    remediation into "make it faster, do NOT mark it serial" — the one
    instruction that leaves the real-process test breaking every parallel run."""
    from ouroboros.preflight_runner import _classify_pass_result

    output = (
        "tests/test_banner.py::test_render FAILED\n"
        "E   AssertionError: assert 'Timeout >30s' in banner\n"
        "worker gw0 crashed and worker restarting disabled\n"
    )
    result = _classify_pass_result("parallel", 1, output, 8000, parallel=True)

    assert "PARALLEL_WORKER_CRASH" in result
    assert "@pytest.mark.serial" in result, "a real crash lost its mark-it-serial fix"
    assert "never a flake/retry" in result
    assert "300s per-test limit" not in result

@pytest.mark.parametrize("max_output", [1, 40, 200])
def test_diagnosis_never_overruns_a_declared_max_output(max_output):
    """`_diagnosis` promises the returned string stays inside the caller's
    limit. The `PREFLIGHT_PLUGIN_MISSING` remediation alone is ~380 chars, so
    the header+remediation prefix must be cut too when the budget is smaller
    than it — not returned whole."""
    from ouroboros.preflight_runner import _classify_pass_result

    usage = "pytest: error: unrecognized arguments: -n auto --dist loadscope\n"
    result = _classify_pass_result("parallel", 4, usage, max_output, parallel=True)
    assert result is not None
    assert len(result) <= max_output

def test_pass_header_reports_that_pass_s_own_duration():
    """`elapsed` is per-pass, not cumulative: a 40s serial pass after a 140s
    parallel one must print 40s, or the header blames the wrong pass for the
    budget it burned."""
    from ouroboros import preflight_runner

    assert "parallel pass, exit 1, 140s" in preflight_runner._classify_pass_result(
        "parallel", 1, "boom", 8000, parallel=False, elapsed=140.0
    )
    src = inspect.getsource(preflight_runner.run_hermetic_pytest)
    assert "pass_started = time.monotonic()" in src
    assert "elapsed = time.monotonic() - pass_started" in src

def test_classify_green_and_empty_pass():
    from ouroboros.preflight_runner import _classify_pass_result

    assert _classify_pass_result("parallel", 0, "", 8000, parallel=True) is None
    # Exit 5 is green PER PASS: a candidate repo may have zero serial tests.
    assert _classify_pass_result("serial", 5, "no tests ran", 8000, parallel=False) is None
