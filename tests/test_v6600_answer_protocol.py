"""v6.60.0 (Phase 4) — answer protocol (contract-gated FINAL ANSWER), blocking
widening, and the bytes_equal verification mode.
"""
from __future__ import annotations

import pathlib

from ouroboros.contracts.task_contract import (
    answer_protocol_active,
    build_task_contract,
    normalize_answer_protocol,
)


# --- 4.1 contract field ---------------------------------------------------------

def test_normalize_answer_protocol_closed_enum():
    assert normalize_answer_protocol("final_answer_line") == "final_answer_line"
    assert normalize_answer_protocol("FINAL_ANSWER_LINE") == "final_answer_line"
    assert normalize_answer_protocol("") == ""
    assert normalize_answer_protocol(None) == ""
    assert normalize_answer_protocol("bogus") == ""  # unknown -> no protocol, never an instruction


def test_contract_carries_answer_protocol_and_inherits_via_metadata():
    contract = build_task_contract({"description": "x", "answer_protocol": "final_answer_line"})
    assert contract["answer_protocol"] == "final_answer_line"
    # metadata path (the subagent/CLI --task-metadata-json route)
    contract_meta = build_task_contract({"description": "x", "metadata": {"answer_protocol": "final_answer_line"}})
    assert contract_meta["answer_protocol"] == "final_answer_line"
    # default: no protocol
    assert build_task_contract({"description": "x"})["answer_protocol"] == ""


def test_answer_protocol_active_gate_reads_ctx_and_dicts():
    from types import SimpleNamespace

    assert answer_protocol_active({"answer_protocol": "final_answer_line"}) is True
    assert answer_protocol_active({"answer_protocol": ""}) is False
    ctx = SimpleNamespace(task_contract={"answer_protocol": "final_answer_line"}, task_metadata={})
    assert answer_protocol_active(ctx) is True
    ctx2 = SimpleNamespace(task_contract={}, task_metadata={"task_contract": {"answer_protocol": "final_answer_line"}})
    assert answer_protocol_active(ctx2) is True
    assert answer_protocol_active(SimpleNamespace(task_contract={}, task_metadata={})) is False


def test_context_injects_protocol_rule_only_when_declared(tmp_path):
    from ouroboros.context import build_runtime_section

    class _Env:
        repo_dir = str(tmp_path)
        drive_root = tmp_path

    task_with = {"id": "t1", "task_contract": {"answer_protocol": "final_answer_line"}}
    task_without = {"id": "t2", "task_contract": {}}
    with_rule = build_runtime_section(_Env(), task_with)
    without_rule = build_runtime_section(_Env(), task_without)
    assert "FINAL ANSWER" in with_rule
    assert "answer_protocol" in with_rule
    assert "FINAL ANSWER" not in without_rule


def test_system_prompt_carries_no_marker_doctrine():
    """The SYSTEM.md marker rule moved to the per-task contract: the default prompt
    must NOT instruct every task to emit FINAL ANSWER lines. Whitespace-NORMALIZED
    check (adversarial r1: a line-wrapped 'FINAL\\nANSWER' slipped past the plain
    substring assert — the guard must be wrap-insensitive)."""
    text = (pathlib.Path(__file__).resolve().parents[1] / "prompts" / "SYSTEM.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "FINAL ANSWER" not in normalized
    assert "CANDIDATES:" not in normalized


def test_pacing_marker_phrases_are_protocol_gated():
    from types import SimpleNamespace

    from ouroboros.task_pacing import build_intrinsic_pacing_note
    from ouroboros.deadline_utils import utc_now
    import datetime

    created = utc_now() - datetime.timedelta(minutes=90)

    def _note(ctx):
        return build_intrinsic_pacing_note(
            ctx, created=created, now=utc_now(), round_idx=5, accumulated_usage={"cost": 1.0},
        )

    plain = SimpleNamespace(task_contract={}, task_metadata={})
    note_plain = _note(plain)
    assert note_plain is not None and "FINAL ANSWER" not in note_plain.text

    protocol = SimpleNamespace(task_contract={"answer_protocol": "final_answer_line"}, task_metadata={})
    note_protocol = _note(protocol)
    assert note_protocol is not None and "FINAL ANSWER" in note_protocol.text


# --- 4.2 blocking widening --------------------------------------------------------

def _result(signal, findings, tiers=("solved",)):
    from types import SimpleNamespace

    actors = [
        {"slot_id": f"s{i}", "signal": signal, "parsed": {"outcome_tier": tier}}
        for i, tier in enumerate(tiers)
    ]
    return SimpleNamespace(aggregate_signal=signal, actors=actors, parsed_findings=findings)


def test_obligations_widen_to_high_only_on_failing_aggregate():
    from ouroboros.loop import _collect_acceptance_obligations

    high_finding = {"severity": "high", "slot_id": "s0", "item": "missed requirement", "recommendation": "implement X"}
    critical_finding = {"severity": "critical", "slot_id": "s0", "item": "broken", "recommendation": "fix Y"}

    # PASS aggregate: high findings do NOT become obligations (critical-only bar).
    trace_pass = {}
    _collect_acceptance_obligations(trace_pass, _result("PASS", [high_finding, critical_finding]))
    items = {o["item"] for o in trace_pass["acceptance_obligations"]}
    assert items == {"broken"}

    # FAIL aggregate: high + critical both become obligations.
    trace_fail = {}
    _collect_acceptance_obligations(trace_fail, _result("FAIL", [high_finding, critical_finding]))
    items_fail = {o["item"] for o in trace_fail["acceptance_obligations"]}
    assert items_fail == {"missed requirement", "broken"}

    # blocked_with_evidence tier (even on non-FAIL signal): widened too.
    trace_blocked = {}
    _collect_acceptance_obligations(
        trace_blocked, _result("PASS", [high_finding], tiers=("blocked_with_evidence",))
    )
    assert {o["item"] for o in trace_blocked["acceptance_obligations"]} == {"missed requirement"}


def test_verdict_is_advisory_flag_is_gone():
    """The dead policy KEY is removed from every ReviewRequest (comments may still
    name it historically); enforcement semantics live in OUROBOROS_REVIEW_ENFORCEMENT."""
    hits = []
    for path in (pathlib.Path(__file__).resolve().parents[1] / "ouroboros").rglob("*.py"):
        if '"verdict_is_advisory":' in path.read_text(encoding="utf-8", errors="replace"):
            hits.append(str(path))
    assert hits == [], f"dead policy key still set in: {hits}"


# --- 4.3 bytes_equal ---------------------------------------------------------------

def _bytes_equal_ctx(tmp_path):
    """A real ToolContext whose active workspace holds the compared files —
    bytes_equal operands are CONFINED (adversarial r1), so the mechanics tests
    run against workspace-resident files like real callers do."""
    from ouroboros.tools.registry import ToolContext

    work = tmp_path / "ws"
    work.mkdir(exist_ok=True)
    drive = tmp_path / "drive"
    drive.mkdir(exist_ok=True)
    return ToolContext(repo_dir=work, drive_root=drive, task_id="t"), work


def test_bytes_equal_compare(tmp_path):
    from ouroboros.tools.verify import _compare_files_bytes_equal

    ctx, work = _bytes_equal_ctx(tmp_path)
    a = work / "a.bin"
    b = work / "b.bin"
    a.write_bytes(b"hello world" * 100)
    b.write_bytes(b"hello world" * 100)
    equal, detail = _compare_files_bytes_equal(ctx, [str(a), str(b)], work, use_executor=False)
    assert equal is True and "==" in detail

    # Introduce a one-byte divergence mid-file; the detail names the offset + hexdump.
    data = bytearray(b"hello world" * 100)
    data[500] = 0x00
    b.write_bytes(bytes(data))
    equal2, detail2 = _compare_files_bytes_equal(ctx, [str(a), str(b)], work, use_executor=False)
    assert equal2 is False
    assert "offset 500" in detail2 and "@" in detail2

    # Size mismatch (prefix case).
    b.write_bytes(b"hello world")
    equal3, detail3 = _compare_files_bytes_equal(ctx, [str(a), str(b)], work, use_executor=False)
    assert equal3 is False and "sizes" in detail3

    # Missing file is a fail, not an exception.
    equal4, detail4 = _compare_files_bytes_equal(ctx, [str(a), str(work / "nope")], work, use_executor=False)
    assert equal4 is False and "not found" in detail4

    # Relative paths resolve against the workspace.
    b.write_bytes(a.read_bytes())
    equal5, _ = _compare_files_bytes_equal(ctx, ["a.bin", "b.bin"], work, use_executor=False)
    assert equal5 is True


def test_bytes_equal_confines_operands(tmp_path, monkeypatch):
    """Adversarial r1 hard blocker: the comparison is a byte-read oracle (sizes +
    divergence hexdump), so BOTH operands must clear the same confinement every
    other artifact-path surface enforces — no control-plane, no arbitrary host
    files, no protected black-box references, no absolute/traversal paths in-executor.
    The user_files root is JAILED into the workspace so the refusals are
    deterministic on every OS (Windows runners put pytest tmp INSIDE the user
    home, where the deliberate user_files lane would otherwise admit tmp siblings)."""
    from ouroboros.tools.verify import _compare_files_bytes_equal

    ctx, work = _bytes_equal_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(work))
    inside = work / "golden.bin"
    inside.write_bytes(b"data")

    # Arbitrary host file (outside workspace + home): refused, no size/hexdump leak.
    equal, detail = _compare_files_bytes_equal(
        ctx, ["/etc/passwd", str(inside)], work, use_executor=False
    )
    assert equal is False and "refused" in detail
    assert "sizes" not in detail and "@" not in detail

    # Control-plane (the data drive) is refused even though it exists on disk.
    secret = pathlib.Path(ctx.drive_root) / "settings.json"
    secret.write_text("{}", encoding="utf-8")
    equal2, detail2 = _compare_files_bytes_equal(
        ctx, [str(secret), str(inside)], work, use_executor=False
    )
    assert equal2 is False and "refused" in detail2

    # Relative traversal out of the workspace: refused.
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"data")
    equal3, detail3 = _compare_files_bytes_equal(
        ctx, ["../outside.bin", str(inside)], work, use_executor=False
    )
    assert equal3 is False and "refused" in detail3

    # Executor surface: absolute and traversing operands are refused BEFORE any cmp
    # runs (same rule as _probe_artifact_lifecycle — no hidden-grader oracle).
    equal4, detail4 = _compare_files_bytes_equal(
        ctx, ["/hidden/tests/expected.bin", "golden.bin"], work, use_executor=True
    )
    assert equal4 is False and "workspace-relative" in detail4
    equal5, detail5 = _compare_files_bytes_equal(
        ctx, ["golden.bin", "../peer/graded.bin"], work, use_executor=True
    )
    assert equal5 is False and "workspace-relative" in detail5


def test_bytes_equal_user_files_lane_is_deliberate(tmp_path, monkeypatch):
    """Claudexor final-review adjudication pinned as a contract: an in-home file
    that clears the user_files guard IS a valid bytes_equal operand (the agent's
    profiles already grant full user_files READ — a size/hexdump is weaker), while
    a bench-style OUROBOROS_USER_FILES_ROOT jail confines the lane: the same
    outside-jail path refuses."""
    from ouroboros.tools.verify import _compare_files_bytes_equal

    ctx, work = _bytes_equal_ctx(tmp_path)
    inside = work / "golden.bin"
    inside.write_bytes(b"data")
    # Simulate "in-home, non-secret" by pointing the user_files root at tmp_path:
    # the sibling file clears the guard → comparable (deliberate lane).
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(tmp_path))
    owner_file = tmp_path / "owner_notes.bin"
    owner_file.write_bytes(b"data")
    equal, detail = _compare_files_bytes_equal(
        ctx, [str(owner_file), str(inside)], work, use_executor=False
    )
    assert equal is True and "refused" not in detail
    # Jail the user_files root elsewhere: the SAME operand now refuses (bench shape).
    jail = tmp_path / "jail"
    jail.mkdir()
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(jail))
    equal2, detail2 = _compare_files_bytes_equal(
        ctx, [str(owner_file), str(inside)], work, use_executor=False
    )
    assert equal2 is False and "refused" in detail2


def test_bytes_equal_refuses_protected_black_box_reference(tmp_path):
    """A black-box reference binary must stay byte-opaque: comparing it IS reading
    its bytes, so the protected-artifacts read_bytes denial applies."""
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.verify import _compare_files_bytes_equal

    work = tmp_path / "ws"
    work.mkdir()
    drive = tmp_path / "drive"
    drive.mkdir()
    ref = work / "reference_bin"
    ref.write_bytes(b"opaque")
    probe = work / "probe.bin"
    probe.write_bytes(b"opaque")
    record = {"id": "ref", "role": "black_box_reference", "paths": ["./reference_bin"]}
    ctx = ToolContext(
        repo_dir=work, drive_root=drive, task_id="t",
        task_metadata={"task_contract": {"resource_policy": {"protected_artifacts": [record]}}},
    )
    equal, detail = _compare_files_bytes_equal(
        ctx, [str(ref), str(probe)], work, use_executor=False
    )
    assert equal is False and "refused" in detail


def test_review_output_budget_knob_lowers_never_raises(monkeypatch):
    """OUROBOROS_REVIEW_MAX_TOKENS lets an operator shrink the reviewer response
    reservation so a mega-diff input pack + output fits a reviewer endpoint's
    context cap (triad r3: 999K input + 65K default output overflowed ALL triad
    endpoints). Floored at 8192; can never exceed the 65536 default."""
    from ouroboros.tools.review import _review_output_budget

    monkeypatch.delenv("OUROBOROS_REVIEW_MAX_TOKENS", raising=False)
    assert _review_output_budget() == 65536
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_TOKENS", "32768")
    assert _review_output_budget() == 32768
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_TOKENS", "128")
    assert _review_output_budget() == 8192
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_TOKENS", "999999")
    assert _review_output_budget() == 65536
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_TOKENS", "bogus")
    assert _review_output_budget() == 65536


def test_bytes_equal_executor_infra_exit_is_not_a_byte_verdict(tmp_path, monkeypatch):
    """cmp exit >1 (missing binary =127, missing file =2) is an INFRA failure —
    still FAIL, but the receipt must not claim the bytes differ (triad r3)."""
    from types import SimpleNamespace

    import ouroboros.workspace_executor as wx
    from ouroboros.tools.verify import _compare_files_bytes_equal

    ctx, work = _bytes_equal_ctx(tmp_path)
    (work / "a.bin").write_bytes(b"x")
    (work / "b.bin").write_bytes(b"x")
    monkeypatch.setattr(
        wx, "execute",
        lambda *a, **k: SimpleNamespace(returncode=127, stdout="", stderr="cmp: not found"),
    )
    equal, detail = _compare_files_bytes_equal(ctx, ["a.bin", "b.bin"], work, use_executor=True)
    assert equal is False
    assert "infra error" in detail and "bytes differ" not in detail


def test_bytes_equal_rejected_for_non_run_kinds(tmp_path):
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.verify import _verify_and_record

    work = tmp_path / "ws"
    work.mkdir()
    drive = tmp_path / "drive"
    drive.mkdir()
    ctx = ToolContext(repo_dir=work, drive_root=drive, task_id="t")
    out = _verify_and_record(
        ctx, contract_kind="artifact_observation",
        expected_match="bytes_equal", artifact_paths=["a.bin", "b.bin"],
    )
    assert "TOOL_ARG_ERROR" in out and "run-kind" in out


def test_verify_and_record_bytes_equal_requires_two_paths(tmp_path):
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.verify import _verify_and_record

    work = tmp_path / "ws"
    work.mkdir()
    drive = tmp_path / "drive"
    drive.mkdir()
    ctx = ToolContext(repo_dir=work, drive_root=drive, task_id="t")
    out = _verify_and_record(
        ctx, contract_kind="explicit_command", check=["true"],
        expected_match="bytes_equal", artifact_paths=["only-one.txt"],
    )
    assert "TOOL_ARG_ERROR" in out and "exactly two files" in out
