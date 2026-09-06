"""v6.50.2 capability changes: A1 expected_match, A3 no-op-nudge gate reader, B1 swarm rollup."""

import json

from ouroboros.tools.verify import _expected_matches
from ouroboros.loop_nudges import _contract_expected_output
from ouroboros.agent_task_pipeline import _build_swarm_efficiency


# --- A1: verify_and_record expected_match modes ---

def test_expected_match_substring_is_default_legacy_behavior():
    # substring == the historical `expected in out`
    assert _expected_matches("xx1179403647xx", "1179403647", "substring") is True
    assert _expected_matches("nope", "1179403647", "substring") is False


def test_expected_match_exact():
    assert _expected_matches("  42 \n", "42", "exact") is True       # stripped both sides
    assert _expected_matches("42 extra", "42", "exact") is False     # substring is NOT exact


def test_expected_match_exact_line():
    out = "header\n  answer: 42  \nfooter"
    assert _expected_matches(out, "answer: 42", "exact_line") is True
    assert _expected_matches(out, "42", "exact_line") is False       # must equal a whole line


def test_expected_match_json_equals_key_order_tolerant():
    assert _expected_matches('{"a":1,"b":2}', '{"b":2,"a":1}', "json_equals") is True
    assert _expected_matches('{"a":1}', '{"a":2}', "json_equals") is False


def test_expected_match_json_equals_non_json_is_false_not_raise():
    # malformed output / empty must return False, never raise
    assert _expected_matches("", "{}", "json_equals") is False
    assert _expected_matches("not json", '{"a":1}', "json_equals") is False


# --- A3: the no-op-nudge gate reads the declared expected_output ---

class _Ctx:
    def __init__(self, task_contract=None, task_metadata=None):
        self.task_contract = task_contract or {}
        self.task_metadata = task_metadata or {}


def test_contract_expected_output_reads_contract_then_metadata():
    assert _contract_expected_output(_Ctx(task_contract={"expected_output": "a CSV"})) == "a CSV"
    # falls back to metadata when contract empty
    assert _contract_expected_output(_Ctx(task_metadata={"expected_output": "from meta"})) == "from meta"
    # falls back to nested metadata.task_contract
    assert _contract_expected_output(_Ctx(task_metadata={"task_contract": {"expected_output": "nested"}})) == "nested"
    # no declared output (e.g. a direct-chat turn) -> "" so the nudge never fires
    assert _contract_expected_output(_Ctx()) == ""


# --- B1: swarm-efficiency rollup ---

class _Env:
    def __init__(self, drive_root):
        self.drive_root = str(drive_root)


def test_build_swarm_efficiency_none_for_plain_task(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "events.jsonl").write_text("", encoding="utf-8")
    assert _build_swarm_efficiency(_Env(tmp_path), {"id": "t1"}) is None


def test_build_swarm_efficiency_rolls_up_fanout_events(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    events = [
        {"type": "swarm_fanout", "parent_task_id": "root1", "task_ids": ["c1", "c2"],
         "inter_wave_latency_sec": 0.0, "requested_model_lane": "heavy"},
        {"type": "swarm_fanout", "parent_task_id": "root1", "task_ids": ["c3"],
         "inter_wave_latency_sec": 194.0, "requested_model_lane": "light"},
        {"type": "swarm_fanout", "parent_task_id": "other", "task_ids": ["x1"],
         "inter_wave_latency_sec": 5.0, "requested_model_lane": "heavy"},
        {"type": "llm_round", "parent_task_id": "root1"},
    ]
    (logs / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    roll = _build_swarm_efficiency(_Env(tmp_path), {"id": "root1"})
    assert roll is not None
    assert roll["subagent_count"] == 3            # c1,c2,c3 (other-parent x1 excluded)
    assert roll["wave_count"] == 2
    assert roll["inter_wave_latency_sec_total"] == 194.0
    assert sorted(roll["lanes_requested"]) == ["heavy", "light"]
    # A plain task's rollup carries no intent annotation (rc-phaseC).
    assert "intent_source" not in roll and "requested_count" not in roll


def _swarm_task(task_id):
    return {"id": task_id, "metadata": {"force_plan": True, "force_plan_source": "swarm"}}


def test_build_swarm_efficiency_zero_fanout_swarm_intent_returns_minimal_block(tmp_path):
    # A Swarm-button task that spawned zero children must not vanish into None:
    # it returns the minimal no_fanout_observed block. `planned` is null — never
    # inferred as 0 from the absence of events.
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "events.jsonl").write_text("", encoding="utf-8")
    block = _build_swarm_efficiency(_Env(tmp_path), _swarm_task("t-swarm"))
    assert block == {
        "intent_source": "swarm",
        "planned": None,
        "observed_started": 0,
        "status": "no_fanout_observed",
    }


def test_build_swarm_efficiency_rollup_carries_intent_source_and_requested_count(tmp_path):
    # With real fanout events present, the existing rollup keys stay unchanged and
    # the swarm-intent task additionally carries intent_source plus the waves'
    # requested_count sum under its existing event name (no synonyms).
    logs = tmp_path / "logs"
    logs.mkdir()
    events = [
        {"type": "swarm_fanout", "parent_task_id": "t-swarm", "task_ids": ["c1", "c2"],
         "requested_count": 2, "inter_wave_latency_sec": 0.0, "requested_model_lane": "heavy"},
        {"type": "swarm_fanout", "parent_task_id": "t-swarm", "task_ids": ["run-1"],
         "requested_count": 1, "inter_wave_latency_sec": 3.5, "requested_model_lane": "codex-route"},
    ]
    (logs / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    roll = _build_swarm_efficiency(_Env(tmp_path), _swarm_task("t-swarm"))
    assert roll is not None
    assert roll["subagent_count"] == 3
    assert roll["wave_count"] == 2
    assert roll["inter_wave_latency_sec_total"] == 3.5
    assert roll["lanes_requested"] == ["heavy", "codex-route"]
    assert roll["intent_source"] == "swarm"
    assert roll["requested_count"] == 3


# --- B2: burst/absorb advisory fires when >=1 OTHER child is still live ---

def test_count_live_sibling_children_counts_other_live(tmp_path):
    import sys as _sys
    _sys.path.insert(0, ".")
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.control import _count_live_sibling_children
    from ouroboros.task_results import write_task_result, STATUS_RUNNING, STATUS_COMPLETED

    drive = tmp_path / "drive"
    (drive).mkdir()
    parent = "aabbccdd"
    write_task_result(drive, "11112222", STATUS_RUNNING, parent_task_id=parent)
    write_task_result(drive, "33334444", STATUS_RUNNING, parent_task_id=parent)
    write_task_result(drive, "55556666", STATUS_RUNNING, parent_task_id=parent)   # the one being waited on
    write_task_result(drive, "77778888", STATUS_COMPLETED, parent_task_id=parent)  # terminal — not counted
    write_task_result(drive, "99990000", STATUS_RUNNING, parent_task_id="ffffffff")  # other parent — not counted
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=drive, task_id=parent)
    # waiting on 5555 with 1111+3333 still live -> 2 other live (advisory fires, threshold >=1)
    assert _count_live_sibling_children(ctx, drive, exclude_task_id="55556666") == 2
    # waiting on 3333 -> 1111 and 5555 still live -> 2 other live (advisory fires under the >=1 rule)
    assert _count_live_sibling_children(ctx, drive, exclude_task_id="33334444") == 2


# --- A1: a run-kind FAILURE (timeout) receipt still carries expected_match + matched ---

def test_verify_timeout_receipt_carries_matched_and_expected_match(tmp_path):
    import sys as _sys
    _sys.path.insert(0, ".")
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.verify import _verify_and_record
    from ouroboros.outcomes import read_verification_receipts

    drive = tmp_path / "drive"
    drive.mkdir()
    (tmp_path / "repo").mkdir()
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=drive, task_id="deadbeef")
    out = _verify_and_record(
        ctx, contract_kind="explicit_command",
        check=[_sys.executable, "-c", "import time; time.sleep(5)"],
        timeout_sec=1,
    )
    assert "FAIL" in out and "timed out" in out
    receipts = read_verification_receipts(drive, "deadbeef")
    assert receipts, "timeout must still record a receipt"
    r = receipts[-1]
    assert r["status"] == "fail"
    assert r["expected_match"] == "substring"   # default carried even on the failure path
    assert r["matched"] is False                # explicit False, not null (ledger projects this)


# --- A2: ignored read-only access-policy block must NOT count as a ledger failure ---

def test_verification_ledger_ignores_readonly_access_policy_block():
    from ouroboros.outcomes import build_verification_ledger

    def _ledger(tool, status):
        return build_verification_ledger(
            task={"id": "t1"},
            loop_outcome={"outcome_axes": {"execution": {"status": "ok"}}, "reason_code": "final_message"},
            llm_trace={"tool_calls": [{"tool": tool, "is_error": True, "status": status}]},
            artifact_bundle={"status": "not_applicable"},
        )
    # read-only exploratory tool blocked by policy -> recorded as 'ignored', NOT a ledger failure
    led = _ledger("search_code", "resource_policy_blocked")
    tc = [e for e in led["entries"] if e.get("kind") == "tool_call"][0]
    assert tc["status"] == "ignored" and tc["blocked_status"] == "resource_policy_blocked"
    assert led["summary"]["has_failures"] is False
    # boundary: the SAME block on a non-read-only effect tool stays a real ledger failure
    led2 = _ledger("run_command", "resource_policy_blocked")
    assert led2["summary"]["has_failures"] is True


# --- B1: swarm_efficiency is actually accepted + persisted by write_task_result ---

def test_write_task_result_persists_swarm_efficiency(tmp_path):
    from ouroboros.task_results import write_task_result, load_task_result, STATUS_COMPLETED

    drive = tmp_path / "drive"
    drive.mkdir()
    roll = {"subagent_count": 3, "wave_count": 2, "inter_wave_latency_sec_total": 194.0, "lanes_requested": ["heavy", "light"]}
    rec = write_task_result(drive, "fa11fa11", STATUS_COMPLETED, swarm_efficiency=roll)
    assert rec.get("swarm_efficiency") == roll          # accepted as a field (no TypeError)
    persisted = load_task_result(drive, "fa11fa11")
    assert persisted.get("swarm_efficiency") == roll    # durably persisted + read back
