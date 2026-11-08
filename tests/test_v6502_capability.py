"""v6.50.2 capability changes: A1 expected_match and B1 swarm rollup (the A3 no-op nudge is gone, #1748)."""


from ouroboros.tools.verify import _expected_matches


# --- A1: verify_and_record expected_match modes ---


def test_expected_match_json_equals_non_json_is_false_not_raise():
    # malformed output / empty must return False, never raise
    assert _expected_matches("", "{}", "json_equals") is False
    assert _expected_matches("not json", '{"a":1}', "json_equals") is False


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
