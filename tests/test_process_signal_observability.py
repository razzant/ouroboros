"""Stream B (node-runtime sprint): typed process meta, verify-receipt
disclosure keys, and the signal-death classification contract.

R5 — run_command/run_script results carry ``exit_code`` / ``signal`` /
``duration_ms`` as TYPED result_meta fields published STRUCTURALLY by the
handler (thread-local channel in ``tools/process_facts.py``); the regex harvest over
the rendered text remains as the read-fallback for records that lack typed
meta, and the rendered prose of healthy runs stays byte-identical (no duration
in prose — roast finding R5).

D6/R4 — verify_and_record receipts gain ``duration_ms`` (always, for run-kind
checks), ``signal`` (killed checks, POSIX name), and ``resolved_runtime``
(present ONLY when the interpreter resolver substituted the physical
executable — the Stream-A seam ``ctx._process_resolved_runtime``). Both FIXED
projections (``_outcome_receipts.verification_receipt_ledger_row`` and
``review_evidence._accept_verification_summary``) carry the new keys; receipt
IDENTITY and reconciliation are untouched.

D7/R6 (Q2-2=A) — a run_command/run_script SIGNAL DEATH (typed meta:
exit_code < 0 or a signal name) is no longer cosmetic, symmetric with the
timeout exclusion; exit_code=1 stays cosmetic; a Windows kill (large POSITIVE
exit code) is a declared residual that stays cosmetic; the cancel/panic pin
lives in test_observability_outcomes_v2.py.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

import ouroboros.tools.shell as shell
from ouroboros.tools.process_facts import consume_last_process_facts, signal_name_for_returncode
from ouroboros.tools.shell import _run_shell

# The SSOT (``signal_name_for_returncode``) names -9 from the host signal table:
# SIGKILL on POSIX; Windows has no such signal, so the name there is the
# disclosed numeric fallback. The flow tests below pin that the typed meta
# CARRIES the SSOT-derived name end to end; the POSIX vocabulary itself is
# pinned by the posix-only real-process tests further down.
_KILL_NAME = signal_name_for_returncode(-9)


@pytest.fixture(autouse=True)
def _clean_facts_slot():
    """Tests in one pytest worker share a thread; keep the TLS slot clean."""
    consume_last_process_facts()
    yield
    consume_last_process_facts()


def _ctx(tmp_path):
    return SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        drive_logs=lambda: pathlib.Path(str(tmp_path)),
    )


@pytest.fixture
def fake_subprocess(monkeypatch):
    """Patch _tracked_subprocess_run with a closure returning a queued result
    (same shape as tests/test_shell_run_shell.py)."""
    monkeypatch.setattr("ouroboros.tools.shell.load_settings", lambda: {})

    def _install(*, returncode: int = 0, stdout: str = "", stderr: str = "", raise_timeout: bool = False):
        calls: list[dict] = []

        def fake_run(cmd, **kwargs):
            calls.append({"cmd": cmd, "kwargs": kwargs})
            if raise_timeout:
                raise subprocess.TimeoutExpired(cmd, 1)
            return CompletedProcess(cmd, returncode, stdout, stderr)

        monkeypatch.setattr("ouroboros.tools.shell._tracked_subprocess_run", fake_run)
        return calls

    return _install


# ---------------------------------------------------------------------------
# R5 — handler publishes typed process facts; prose is unchanged
# ---------------------------------------------------------------------------


def test_run_shell_publishes_typed_facts_on_healthy_run(tmp_path, fake_subprocess):
    fake_subprocess(stdout="ok", returncode=0)
    result = _run_shell(_ctx(tmp_path), ["echo", "x"])
    facts = consume_last_process_facts()
    assert facts is not None
    assert facts["exit_code"] == 0
    assert "signal" not in facts
    assert isinstance(facts["duration_ms"], int) and facts["duration_ms"] >= 0
    assert "resolved_runtime" not in facts
    # The rendered prose of a healthy run is byte-shape unchanged (roast R5):
    # no duration and no new fields appear in the text the agent reads.
    assert "duration_ms" not in result
    assert result.startswith("exit_code=0")


def test_run_shell_publishes_signal_death_facts(tmp_path, fake_subprocess):
    fake_subprocess(returncode=-9, stderr="")
    result = _run_shell(_ctx(tmp_path), ["node", "--version"])
    facts = consume_last_process_facts()
    assert facts["exit_code"] == -9
    assert facts["signal"] == _KILL_NAME
    # The existing rendered shape stays: signal named in prose as before.
    assert "⚠️ SHELL_EXIT_ERROR" in result and f"signal={_KILL_NAME}" in result
    assert "duration_ms" not in result


def test_run_shell_timeout_publishes_typed_timeout_facts(tmp_path, fake_subprocess):
    fake_subprocess(raise_timeout=True)
    result = _run_shell(_ctx(tmp_path), ["sleep", "999"])
    assert result.startswith("⚠️ TOOL_TIMEOUT")
    facts = consume_last_process_facts()
    # A timed-out child has NO returncode — structurally different from signal
    # death (timeout classification is unchanged by this sprint). The absence
    # used to be the WHOLE record; the deadline kill is now typed beside it, on
    # every platform (typed-process-facts lane, owner 10=B).
    assert "exit_code" not in facts and "signal" not in facts
    assert facts["timed_out"] is True
    assert facts["killed_by_host"] is True
    assert isinstance(facts["duration_ms"], int)


def test_run_shell_missing_binary_publishes_duration_only(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.tools.shell.load_settings", lambda: {})

    def raise_missing(cmd, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", cmd[0])

    monkeypatch.setattr("ouroboros.tools.shell._tracked_subprocess_run", raise_missing)
    result = _run_shell(_ctx(tmp_path), ["definitely-not-a-binary"])
    assert result.startswith("⚠️ SHELL_ARG_ERROR")
    facts = consume_last_process_facts()
    assert "exit_code" not in facts and "signal" not in facts
    # The spawn never happened, and the platform's exception class is the typed
    # cause — not a synthesized exit code, and not silence.
    assert facts["pre_exec_failure"] == "FileNotFoundError"
    assert not {"pid", "pgid", "process_start_identity"}.intersection(facts)
    assert "timed_out" not in facts and "killed_by_host" not in facts
    assert isinstance(facts["duration_ms"], int)


def test_run_shell_pre_exec_refusal_publishes_no_facts(tmp_path):
    # No process ran (shell builtin refusal) -> the channel stays empty.
    result = _run_shell(_ctx(tmp_path), ["cd", "/tmp"])
    assert result.startswith("⚠️ SHELL_CMD_ERROR")
    assert consume_last_process_facts() is None


def test_run_shell_facts_carry_resolved_runtime_seam(tmp_path, fake_subprocess):
    """Stream-A seam pin: when the registry dispatch attests a substituted
    physical executable in ``ctx._process_resolved_runtime``, the typed facts
    disclose it; absent attestation -> absent key (the healthy default)."""
    fake_subprocess(returncode=0, stdout="v24.16.0")
    ctx = _ctx(tmp_path)
    ctx._process_resolved_runtime = "/opt/ouroboros/bundled/node"
    _run_shell(ctx, ["node", "--version"])
    facts = consume_last_process_facts()
    assert facts["resolved_runtime"] == "/opt/ouroboros/bundled/node"


# ---------------------------------------------------------------------------
# R5 — executor wrapper merges typed facts into result_meta (typed > regex)
# ---------------------------------------------------------------------------


def _run_single(tmp_path, execute, tool="run_command"):
    from ouroboros.loop_tool_execution import _execute_single_tool

    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    from ouroboros.tools.tool_result import LegacyTextResultAdapter

    tools = SimpleNamespace(
        CODE_TOOLS={tool},
        _ctx=SimpleNamespace(task_metadata={}, drive_root=tmp_path),
        execute=execute,
        # Campaign dispatch is typed: the loop reads the dispatcher's published
        # ToolResult; a text-only fake goes through the one legacy adapter.
        execute_result=lambda name, args: LegacyTextResultAdapter.from_text(
            name, str(execute(name, args))
        ),
    )
    return _execute_single_tool(
        tools,
        {"id": "call-1", "function": {"name": tool, "arguments": "{}"}},
        logs,
        "task-1",
    )


def test_execute_single_tool_merges_typed_meta_with_precedence(tmp_path):
    def execute(_name, _args):
        # The handler's typed fact (exit_code=-9) beats the prose (exit_code=1):
        # typed fields take precedence; the regex path stays only as fallback.
        shell._publish_process_facts(returncode=-9, started_ts=time.time() - 0.005)
        return "⚠️ SHELL_EXIT_ERROR: command exited with exit_code=1."

    out = _run_single(tmp_path, execute)
    meta = out["result_meta"]
    assert meta["status"] == "non_zero_exit"
    assert meta["exit_code"] == -9
    assert meta["signal"] == _KILL_NAME
    assert isinstance(meta["duration_ms"], int)


def test_execute_single_tool_legacy_prose_forges_no_process_facts(tmp_path):
    """Campaign D02 contract (supersedes the upstream regex fallback): process
    facts come ONLY from typed producer publications — prose spelling
    exit_code/signal inside the result text is producer-controlled and forges
    nothing. Absence is the honest reading for an untyped legacy record."""
    def execute(_name, _args):
        # No typed publication (legacy path): nothing is harvested from prose.
        return "⚠️ SHELL_EXIT_ERROR: command exited with exit_code=-9 (signal=SIGKILL, cwd=/x)."

    out = _run_single(tmp_path, execute)
    meta = out["result_meta"]
    assert "exit_code" not in meta
    assert "signal" not in meta
    assert "duration_ms" not in meta  # nothing typed was measured
    assert meta["status"] == "non_zero_exit"  # the typed CODE still classifies


def test_execute_single_tool_drops_stale_thread_facts(tmp_path):
    # Leftover facts on this thread from an unrelated earlier call must never
    # leak into a process-tool call whose path ran NO process.
    shell._publish_process_facts(returncode=0, started_ts=time.time())

    def execute(_name, _args):
        return '⚠️ SHELL_CMD_ERROR: "cd" is a shell builtin, not an executable.'

    out = _run_single(tmp_path, execute)
    meta = out["result_meta"]
    assert "duration_ms" not in meta and "exit_code" not in meta


def test_tools_that_run_no_process_merge_no_facts(tmp_path):
    """The channel is PUBLISHER-scoped, not tool-name-scoped (typed-process-
    facts lane): a call that published nothing carries no process facts, and a
    stale publication left on the thread is DROPPED before dispatch rather than
    merely ignored — the same no-contamination contract the earlier name gate
    provided, now without a name table that dynamic surfaces (ext_*) and the
    skill/verify handlers could never be listed in."""
    shell._publish_process_facts(returncode=-9, started_ts=time.time())
    out = _run_single(tmp_path, lambda _n, _a: "file contents", tool="read_file")
    meta = out["result_meta"]
    assert "duration_ms" not in meta and "exit_code" not in meta
    assert consume_last_process_facts() is None


def test_any_publishing_handler_reaches_result_meta(tmp_path):
    """A handler that is not run_command/run_script — here the skill/verify/
    extension shape, a tool whose name no longer gates the channel — reaches
    the call's result_meta with its typed facts."""
    def execute(_name, _args):
        shell._publish_process_facts(
            returncode=None, started_ts=time.time() - 0.004,
            timed_out=True, killed_by_host=True,
        )
        return "verify_and_record [visible_verifier] FAIL: check timed out after 60s."

    out = _run_single(tmp_path, execute, tool="verify_and_record")
    meta = out["result_meta"]
    assert meta["timed_out"] is True and meta["killed_by_host"] is True
    assert "exit_code" not in meta and "signal" not in meta


def test_typed_meta_flows_handler_to_trace_item_to_error_record(tmp_path):
    """END-TO-END pin of the R5 chain: handler-published typed facts reach the
    durable TRACE ITEM (``process_tool_results`` spreads result_meta into
    ``llm_trace["tool_calls"]``) and from there the execution-axis ERROR RECORD
    — no regex over prose anywhere on the path (the prose deliberately lies
    with exit_code=1 to prove the typed channel is what flowed through)."""
    from ouroboros.loop_tool_execution import process_tool_results
    from ouroboros.outcomes import _classify_tool_errors

    def execute(_name, _args):
        shell._publish_process_facts(returncode=-9, started_ts=time.time() - 0.009)
        return "⚠️ SHELL_EXIT_ERROR: command exited with exit_code=1."

    exec_result = _run_single(tmp_path, execute)
    llm_trace: dict = {"tool_calls": []}
    errors = process_tool_results([exec_result], [], llm_trace, lambda _msg: None)
    assert errors == 1
    item = llm_trace["tool_calls"][0]
    assert item["exit_code"] == -9
    assert item["signal"] == _KILL_NAME
    assert isinstance(item["duration_ms"], int)
    buckets = _classify_tool_errors(llm_trace)
    assert buckets["unresolved"] and not buckets["cosmetic"]
    assert buckets["unresolved"][0]["signal"] == _KILL_NAME


def test_typed_absence_beats_regex_signal_from_stdout(tmp_path):
    """ABSENCE-precedence pin (adversarial finding B-1): when the handler
    publishes honest typed facts WITHOUT a signal (exit_code=1, a plain
    failure), a ``signal=SIGKILL`` string inside the child's own stdout (e.g.
    the agent grepping logs for this very incident) must NOT survive the merge
    as a regex-harvested fact — the typed publication owns the whole fact
    family, including the absence of a signal, so the call stays cosmetic."""
    from ouroboros.loop_tool_execution import process_tool_results
    from ouroboros.outcomes import _classify_tool_errors

    def execute(_name, _args):
        shell._publish_process_facts(returncode=1, started_ts=time.time() - 0.05)
        return (
            "⚠️ SHELL_EXIT_ERROR: command exited with exit_code=1.\n\n"
            "STDOUT:\nkernel log: rejecting invalid page signal=SIGKILL observed\n"
        )

    exec_result = _run_single(tmp_path, execute)
    llm_trace: dict = {"tool_calls": []}
    errors = process_tool_results([exec_result], [], llm_trace, lambda _msg: None)
    assert errors == 1
    item = llm_trace["tool_calls"][0]
    assert item["exit_code"] == 1
    assert "signal" not in item
    buckets = _classify_tool_errors(llm_trace)
    assert buckets["cosmetic"] and not buckets["unresolved"]


# ---------------------------------------------------------------------------
# D6/R4 — verify_and_record receipt disclosure keys
# ---------------------------------------------------------------------------


def _verify_ctx(tmp_path):
    from ouroboros.tools.registry import ToolContext

    work = tmp_path / "ws"
    work.mkdir(exist_ok=True)
    drive = tmp_path / "drive"
    drive.mkdir(exist_ok=True)
    return ToolContext(repo_dir=work, drive_root=drive, task_id="t"), drive


def _read_receipts(drive):
    path = drive / "task_results" / "artifacts" / "t" / "verification_receipts.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.serial
@pytest.mark.skipif(os.name != "posix", reason="signal-death semantics are POSIX-only (declared residual)")
def test_verify_receipt_records_duration_and_signal_for_killed_check(tmp_path):
    from ouroboros.tools.verify import _verify_and_record

    ctx, drive = _verify_ctx(tmp_path)
    out_text = _verify_and_record(
        ctx, contract_kind="explicit_command", check=["sh", "-c", "kill -9 $$"],
    )
    row = _read_receipts(drive)[-1]
    assert row["status"] == "fail"
    assert row["returncode"] == -9
    assert row["signal"] == "SIGKILL"
    assert isinstance(row["duration_ms"], int) and row["duration_ms"] >= 0
    assert "resolved_runtime" not in row
    # Agent-visible text is UNCHANGED (legacy `exit=-9` shape, no new fields).
    assert "FAIL: exit=-9" in out_text
    assert "duration_ms" not in out_text and "SIGKILL" not in out_text


@pytest.mark.serial
@pytest.mark.skipif(os.name != "posix", reason="uses sh")
def test_verify_receipt_healthy_default_has_duration_and_no_other_new_keys(tmp_path):
    from ouroboros.tools.verify import _verify_and_record

    ctx, drive = _verify_ctx(tmp_path)
    out_text = _verify_and_record(
        ctx, contract_kind="explicit_command", check=["sh", "-c", "true"],
    )
    row = _read_receipts(drive)[-1]
    assert row["status"] == "pass"
    assert isinstance(row["duration_ms"], int)
    assert "signal" not in row
    assert "resolved_runtime" not in row
    assert "PASS: exit=0" in out_text and "duration_ms" not in out_text


@pytest.mark.serial
@pytest.mark.skipif(os.name != "posix", reason="uses sh")
def test_verify_receipt_stores_resolved_runtime_when_attested(tmp_path):
    """The receipt writer accepts and stores the Stream-A attested runtime; the
    recorded ``check`` text stays the ORIGINAL argv rendering (identity, R4)."""
    from ouroboros.tools.verify import _verify_and_record

    ctx, drive = _verify_ctx(tmp_path)
    ctx._process_resolved_runtime = "/opt/ouroboros/bundled/node"
    _verify_and_record(ctx, contract_kind="explicit_command", check=["sh", "-c", "true"])
    row = _read_receipts(drive)[-1]
    assert row["resolved_runtime"] == "/opt/ouroboros/bundled/node"
    assert row["check"] == "sh -c true"  # identity text untouched by disclosure


def test_verify_timeout_receipt_carries_duration(tmp_path, monkeypatch):
    from ouroboros.tools.verify import _verify_and_record

    ctx, drive = _verify_ctx(tmp_path)

    def raise_timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr("ouroboros.tools.shell._tracked_subprocess_run", raise_timeout)
    _verify_and_record(ctx, contract_kind="explicit_command", check=["sh", "-c", "sleep 999"])
    row = _read_receipts(drive)[-1]
    assert row["status"] == "fail" and row["returncode"] is None
    assert isinstance(row["duration_ms"], int)
    assert "signal" not in row


# ---------------------------------------------------------------------------
# D6 — both FIXED projections carry the new keys; identity is untouched
# ---------------------------------------------------------------------------


def test_ledger_row_carries_disclosure_keys_only_when_present():
    from ouroboros._outcome_receipts import verification_receipt_ledger_row

    row = verification_receipt_ledger_row({
        "status": "fail", "contract_kind": "explicit_command", "check": "node x.js",
        "returncode": -9, "duration_ms": 9, "signal": "SIGKILL",
        "resolved_runtime": "/opt/ouroboros/bundled/node",
    })
    assert row["duration_ms"] == 9
    assert row["signal"] == "SIGKILL"
    assert row["resolved_runtime"] == "/opt/ouroboros/bundled/node"

    bare = verification_receipt_ledger_row({"status": "pass", "check": "true"})
    assert "duration_ms" not in bare
    assert "signal" not in bare
    assert "resolved_runtime" not in bare


def test_build_verification_ledger_entry_carries_new_keys():
    from ouroboros.outcomes import build_verification_ledger

    led = build_verification_ledger(
        task={"id": "t", "task_contract": {}},
        loop_outcome={"outcome_axes": {"execution": {"status": "ok"}, "objective": {"status": "not_evaluated"}}},
        llm_trace={"tool_calls": [], "verification_receipts": [{
            "status": "fail", "contract_kind": "explicit_command", "check": "node --test",
            "returncode": -9, "duration_ms": 9, "signal": "SIGKILL",
            "resolved_runtime": "/opt/node",
        }]},
        artifact_bundle={},
    )
    entry = next(e for e in led["entries"] if e.get("kind") == "verification_receipt")
    assert entry["duration_ms"] == 9
    assert entry["signal"] == "SIGKILL"
    assert entry["resolved_runtime"] == "/opt/node"


def test_accept_summary_carries_latest_disclosure_keys():
    from ouroboros.review_evidence import _accept_verification_summary

    summary = _accept_verification_summary([{
        "status": "fail", "check": "node --test", "returncode": -9,
        "duration_ms": 9, "signal": "SIGKILL", "resolved_runtime": "/opt/node",
    }])
    assert summary["latest_duration_ms"] == 9
    assert summary["latest_signal"] == "SIGKILL"
    assert "/opt/node" in summary["latest_resolved_runtime"]

    plain = _accept_verification_summary([{"status": "pass", "check": "true"}])
    assert "latest_duration_ms" not in plain
    assert "latest_signal" not in plain
    assert "latest_resolved_runtime" not in plain


def test_reconciliation_and_identity_unchanged_by_disclosure_keys():
    """Regression (R4): a red receipt carrying duration/signal/resolved_runtime
    reconciles EXACTLY as before — the new keys never enter the identity."""
    from ouroboros._outcome_receipts import receipt_identity
    from ouroboros.review_evidence import _accept_verification_summary

    red = {
        "status": "fail", "check": "pytest -q", "check_rendering": "shlex_join",
        "returncode": -9, "duration_ms": 9, "signal": "SIGKILL",
        "resolved_runtime": "/opt/ouroboros/bundled/node",
    }
    green = {"status": "pass", "check": "pytest -q", "check_rendering": "shlex_join"}
    assert receipt_identity(red) == receipt_identity(green)
    summary = _accept_verification_summary([red, green])
    assert summary["unreconciled_red"] is False
    assert summary["unreconciled_red_count"] == 0
    # And an UNRECONCILED red keyed the same way still reads red.
    alone = _accept_verification_summary([red])
    assert alone["unreconciled_red"] is True


# ---------------------------------------------------------------------------
# D7/R6 — classification matrix (typed meta decides; regex stays fallback)
# ---------------------------------------------------------------------------


def _shell_item(**overrides):
    item = {
        "tool": "run_command",
        "is_error": True,
        "status": "non_zero_exit",
        "args": {"cmd": ["node", "--version"]},
        "result": "⚠️ SHELL_EXIT_ERROR: command exited with exit_code=1.",
    }
    item.update(overrides)
    return item


def test_classifier_signal_death_matrix():
    from ouroboros.outcomes import _classify_tool_errors

    def classify(item):
        return _classify_tool_errors({"tool_calls": [item]})

    # exit_code=1: STAYS cosmetic (the T4 contract is untouched for plain exits).
    plain = classify(_shell_item(exit_code=1))
    assert plain["cosmetic"] and not plain["unresolved"]

    # Typed -9: signal death is a REAL execution-degrading error (D7).
    killed = classify(_shell_item(exit_code=-9, signal="SIGKILL"))
    assert killed["unresolved"] and not killed["cosmetic"]
    assert killed["unresolved"][0]["signal"] == "SIGKILL"

    # Negative exit_code alone (no signal name) is enough.
    assert classify(_shell_item(exit_code=-11))["unresolved"]

    # Signal name alone (legacy regex-harvested record) is enough too.
    assert classify(_shell_item(signal="SIGKILL"))["unresolved"]

    # No typed/harvested meta at all: cannot prove signal death -> cosmetic.
    assert classify(_shell_item())["cosmetic"]

    # Windows residual (DECLARED, not faked): a killed process there reports a
    # large POSITIVE code and no signal — the partition cannot see it, and the
    # record stays cosmetic. Contract pin of the honest gap.
    assert classify(_shell_item(exit_code=3221225477))["cosmetic"]

    # run_script is covered symmetrically.
    script_kill = classify(_shell_item(tool="run_script", exit_code=-9, signal="SIGKILL"))
    assert script_kill["unresolved"]

    # timeout: unchanged — was never cosmetic, stays a real failure.
    timeout = classify(_shell_item(status="timeout", result="⚠️ TOOL_TIMEOUT (run_command): exceeded 360s"))
    assert timeout["unresolved"]


def test_classifier_recovered_signal_death_is_recovered():
    """A -9 followed by an IDENTICAL successful rerun is RECOVERED (the recovery
    scan runs before the cosmetic/unresolved split) — the agent proved the
    command works, so no residual red."""
    from ouroboros.outcomes import _classify_tool_errors

    buckets = _classify_tool_errors({"tool_calls": [
        _shell_item(exit_code=-9, signal="SIGKILL"),
        {
            "tool": "run_command", "is_error": False, "status": "ok",
            "args": {"cmd": ["node", "--version"]}, "result": "exit_code=0\nSTDOUT:\nv24",
        },
    ]})
    assert buckets["recovered"] and not buckets["unresolved"] and not buckets["cosmetic"]


def test_freshness_audit_receives_epoch_not_monotonic(tmp_path, fake_subprocess, monkeypatch):
    """D2-1 regression (delta review): the undeclared-output freshness audit
    compares its floor against st_mtime (EPOCH seconds). The duration stamps
    moved to time.monotonic(); the audit must keep receiving an epoch stamp,
    or every pre-existing file looks newer than the command."""
    import time as _time

    from ouroboros.tools import shell as shell_mod

    captured = {}

    def fake_audit(ctx, cmd, outputs, scratch_abs=None, command_start_ts=None, cwd=None):
        captured["start_ts"] = command_start_ts
        return []

    monkeypatch.setattr(
        shell_mod, "_mentioned_user_file_outputs_without_declaration", fake_audit
    )
    fake_subprocess(returncode=0, stdout="done")
    ctx = _ctx(tmp_path)
    shell_mod._run_shell(ctx, ["echo", "hi"])
    # An epoch stamp is ~1.7e9; a monotonic stamp is host uptime (< 1e9 for
    # any host younger than ~31 years). The audit must get the epoch one.
    assert captured["start_ts"] is not None
    assert captured["start_ts"] > 1_000_000_000
    assert abs(captured["start_ts"] - _time.time()) < 300


# ---------------------------------------------------------------------------
# Typed process facts for the five remaining surfaces (owner batch 13, item
# 10 = B). Each surface publishes at the point the truth is known; none of them
# derives a fact from prose, and none synthesizes a code the platform did not
# give.
# ---------------------------------------------------------------------------


@pytest.mark.serial
@pytest.mark.skipif(os.name != "posix", reason="uses sh")
def test_verify_check_publishes_typed_exit_and_signal(tmp_path):
    """verify_and_record renders ``exit=`` for the agent but never stamped the
    exit into the call's typed meta; the check is a child of this call and now
    publishes like run_command's."""
    from ouroboros.tools.verify import _verify_and_record

    ctx, _drive = _verify_ctx(tmp_path)
    _verify_and_record(ctx, contract_kind="explicit_command", check=["sh", "-c", "exit 3"])
    facts = consume_last_process_facts()
    assert facts["exit_code"] == 3
    assert "signal" not in facts
    assert isinstance(facts["duration_ms"], int)


@pytest.mark.serial
@pytest.mark.skipif(os.name != "posix", reason="uses sh")
def test_verify_killed_check_publishes_signal_fact(tmp_path):
    from ouroboros.tools.verify import _verify_and_record

    ctx, _drive = _verify_ctx(tmp_path)
    _verify_and_record(ctx, contract_kind="explicit_command", check=["sh", "-c", "kill -9 $$"])
    facts = consume_last_process_facts()
    assert facts["exit_code"] == -9
    assert facts["signal"] == "SIGKILL"


@pytest.mark.serial
@pytest.mark.skipif(os.name != "posix", reason="uses sh")
def test_verify_timeout_publishes_typed_kill_facts(tmp_path, monkeypatch):
    from ouroboros.tools import verify as verify_mod

    def _boom(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="sleep", timeout=1)

    monkeypatch.setattr("ouroboros.tools.shell._tracked_subprocess_run", _boom)
    ctx, _drive = _verify_ctx(tmp_path)
    out = verify_mod._verify_and_record(
        ctx, contract_kind="explicit_command", check=["sh", "-c", "sleep 999"],
    )
    assert "timed out" in out
    facts = consume_last_process_facts()
    assert facts["timed_out"] is True and facts["killed_by_host"] is True
    assert "exit_code" not in facts


def _skill_child_facts(monkeypatch, *, returncode=0, timed_out=False, overflow=False):
    """Drive ``_run_skill_subprocess`` over a fake child and return the facts."""
    import ouroboros.tools.skill_exec as skill_exec

    class _FakeProc:
        pid = 5150

        def __init__(self):
            self.stdout = None
            self.stderr = None
            self.returncode = None if (timed_out or overflow) else returncode

        def poll(self):
            return None if (timed_out or overflow) else returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(skill_exec, "Popen", lambda *_a, **_k: _FakeProc())
    monkeypatch.setattr(skill_exec, "_kill_process_group", lambda _p: None)

    def _fake_drain(_pipe, _cap, _buf, flag, label):
        if overflow:
            flag[label] = True

    monkeypatch.setattr(skill_exec, "_drain_pipe_with_cap", _fake_drain)
    try:
        skill_exec._run_skill_subprocess(
            ["python", "s.py"], cwd=".", env={}, timeout_sec=(0 if timed_out else 5),
            stdout_cap=10, stderr_cap=10,
        )
    except subprocess.TimeoutExpired:
        pass
    return consume_last_process_facts()


def test_skill_exec_child_publishes_typed_exit_code(monkeypatch):
    facts = _skill_child_facts(monkeypatch, returncode=7)
    assert facts["exit_code"] == 7
    assert "timed_out" not in facts and "killed_by_host" not in facts


def test_skill_exec_signal_death_survives_the_or_zero_flattening(monkeypatch):
    """``_run_skill_subprocess`` returns ``returncode or 0`` for the legacy text
    renderer; the typed channel carries the real negative code and its signal."""
    facts = _skill_child_facts(monkeypatch, returncode=-9)
    assert facts["exit_code"] == -9
    assert facts["signal"] == _KILL_NAME


def test_skill_exec_timeout_publishes_host_kill_without_exit_code(monkeypatch):
    facts = _skill_child_facts(monkeypatch, timed_out=True)
    assert facts["timed_out"] is True and facts["killed_by_host"] is True
    assert "exit_code" not in facts


def test_skill_exec_output_cap_kill_is_a_host_kill_not_a_timeout(monkeypatch):
    facts = _skill_child_facts(monkeypatch, overflow=True)
    assert facts["killed_by_host"] is True and "timed_out" not in facts


def test_skill_preflight_timeout_reports_no_returncode_instead_of_fake_minus_nine(
    tmp_path, monkeypatch,
):
    """The synthesized ``-9`` read downstream as a real POSIX SIGKILL — on
    Windows too, where a host kill produces no signal at all. The honest record
    is no returncode plus the typed kill facts."""
    import ouroboros.tools.skill_preflight as preflight

    class _FakeProc:
        returncode = None
        stdout = None
        stderr = None

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="node", timeout=timeout or 1)

    monkeypatch.setattr(preflight, "Popen", lambda *_a, **_k: _FakeProc())
    monkeypatch.setattr(preflight, "_kill_process_group", lambda _p: None)
    result = preflight._run_check(["node", "--check", "x.js"], cwd=tmp_path)
    assert result["returncode"] is None
    assert result["timeout"] is True and result["killed_by_host"] is True
    facts = consume_last_process_facts()
    assert facts["timed_out"] is True and facts["killed_by_host"] is True
    assert "exit_code" not in facts and "signal" not in facts


def test_skill_exec_spawn_failure_publishes_typed_pre_exec_cause(monkeypatch):
    """No child existed: the typed cause replaces silence, and no exit code is
    invented for a process that never ran."""
    import ouroboros.tools.skill_exec as skill_exec

    def _boom(*_a, **_k):
        raise FileNotFoundError(2, "No such file or directory", "python")

    monkeypatch.setattr(skill_exec, "Popen", _boom)
    with pytest.raises(FileNotFoundError):
        skill_exec._run_skill_subprocess(
            ["python", "s.py"], cwd=".", env={}, timeout_sec=5,
            stdout_cap=10, stderr_cap=10,
        )
    facts = consume_last_process_facts()
    assert facts["pre_exec_failure"] == "FileNotFoundError"
    assert "exit_code" not in facts


def test_skill_preflight_missing_runtime_reports_typed_pre_exec_cause(tmp_path, monkeypatch):
    """The synthesized ``-1`` claimed an exit code for a process that never
    existed; the typed cause replaces it."""
    import ouroboros.tools.skill_preflight as preflight

    def _boom(*_a, **_k):
        raise FileNotFoundError(2, "No such file or directory", "node")

    monkeypatch.setattr(preflight, "Popen", _boom)
    result = preflight._run_check(["node", "--check", "x.js"], cwd=tmp_path)
    assert result["returncode"] is None
    assert result["pre_exec_failure"] == "FileNotFoundError"
    facts = consume_last_process_facts()
    assert facts["pre_exec_failure"] == "FileNotFoundError"
    assert "exit_code" not in facts


def _run_extension_child(tmp_path, monkeypatch, proc):
    from ouroboros import extension_process_runner

    monkeypatch.setattr(
        extension_process_runner.subprocess, "Popen", lambda *_a, **_k: proc
    )
    monkeypatch.setattr(extension_process_runner, "_kill_process_group", lambda _p: None)
    try:
        extension_process_runner._run_child(
            {"skill_name": "facts-probe"},
            skill_dir=tmp_path, drive_root=tmp_path, repo_dir=tmp_path,
            env={}, timeout_sec=1,
        )
    except extension_process_runner.ExtensionProcessError:
        pass
    return consume_last_process_facts()


def _extension_proc(returncode):
    import io

    class _FakeProc:
        pid = 9001

        def __init__(self):
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"")
            self.returncode = returncode

        def poll(self):
            return returncode

        def wait(self, timeout=None):
            return returncode

    return _FakeProc()


def test_extension_child_death_publishes_typed_exit_and_signal(tmp_path, monkeypatch):
    """Extension children were closed by DELETION, not by typing: the dispatcher
    stamped EXTENSION_ERROR but the child's exit code and signal existed only
    inside the error prose."""
    facts = _run_extension_child(tmp_path, monkeypatch, _extension_proc(-9))
    assert facts["exit_code"] == -9
    assert facts["signal"] == _KILL_NAME
    assert isinstance(facts["duration_ms"], int)


def test_extension_child_clean_exit_publishes_zero(tmp_path, monkeypatch):
    facts = _run_extension_child(tmp_path, monkeypatch, _extension_proc(0))
    assert facts["exit_code"] == 0
    assert "signal" not in facts


def test_extension_child_timeout_publishes_host_kill(tmp_path, monkeypatch):
    import io

    class _NeverExits:
        pid = 9002

        def __init__(self):
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"")
            self.returncode = None

        def poll(self):
            return None

        def wait(self, timeout=None):
            return None

    facts = _run_extension_child(tmp_path, monkeypatch, _NeverExits())
    assert facts["timed_out"] is True and facts["killed_by_host"] is True


def test_windows_host_kill_carries_no_forged_signal(tmp_path):
    """Windows residual, pinned with the platform honest: ``TerminateProcess``
    leaves a large POSITIVE exit status and no POSIX signal number, so the
    signal partition can name nothing. ``killed_by_host`` beside that exit
    status is the whole fact the platform gives — and the POSIX name is never
    fabricated from it. Windows-executed proof pending the matrix; this pins the
    platform-independent derivation the Windows path relies on."""
    from ouroboros.tools.process_facts import (
        publish_process_facts,
        signal_name_for_returncode,
    )

    windows_terminate_status = 0xC000013A  # STATUS_CONTROL_C_EXIT
    assert signal_name_for_returncode(windows_terminate_status) == ""
    publish_process_facts(
        returncode=windows_terminate_status, started_ts=time.monotonic(),
        killed_by_host=True,
    )
    facts = consume_last_process_facts()
    assert facts["exit_code"] == windows_terminate_status
    assert facts["killed_by_host"] is True
    assert "signal" not in facts


def test_tools_jsonl_row_carries_the_typed_process_facts(tmp_path):
    """Consumer pin: the row an operator greps carries the facts of the call,
    not just its status."""
    def execute(_name, _args):
        shell._publish_process_facts(
            returncode=-9, started_ts=time.monotonic(), killed_by_host=True,
        )
        return "⚠️ SHELL_EXIT_ERROR: command exited."

    logs = tmp_path / "logs"
    _run_single(tmp_path, execute)
    rows = [
        json.loads(line)
        for line in (logs / "tools.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    row = rows[-1]
    assert row["exit_code"] == -9
    assert row["signal"] == _KILL_NAME
    assert row["killed_by_host"] is True
