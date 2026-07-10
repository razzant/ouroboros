import gzip
import json
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from ouroboros.observability import (
    persist_call,
    posix_private_modes_supported,
    redact_projection,
    write_blob,
)
from ouroboros.outcomes import (
    EXECUTION_DEGRADED,
    EXECUTION_FAILED,
    EXECUTION_INFRA_FAILED,
    EXECUTION_OK,
    OBJECTIVE_NOT_EVALUATED,
    RESULT_INFRA_FAILED,
    artifact_bundle_from_result,
    build_verification_ledger,
    derive_loop_outcome,
    maybe_write_verification_artifact,
    normalize_outcome_axes,
    refresh_verification_ledger_artifacts,
)
from ouroboros.utils import sanitize_tool_args_for_log


def _read_gzip_json(path):
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def test_redactor_records_key_and_value_rules_without_secret_leak():
    payload = {
        "OPENAI_API_KEY": "sk-testsecretvalue000000000000",
        "log": "MY_API_KEY=thisisaverylongsecretvalue123456 github_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
        "prompt_tokens": 123,
        "completion_tokens": 45,
        "cached_tokens": 6,
        "token_estimate": 789,
        "reasoning_tokens": 10,
        "nested": {
            "authorization": "Bearer verylongbearertokenvalue123456",
            "access_token": "verylongaccesstokenvalue123456",
            "refreshToken": "verylongrefreshtokenvalue123456",
            "secret": "plainsecretvalue1234567890",
            "secret_key": "secretkeyvalue1234567890",
            "apiKey": "apikeyvalue1234567890",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            "PRIVATE_KEY_PEM": "-----BEGINPRIVATEKEY-----abc1234567890",
            "STRIPE_SECRET_KEY": "stripescretvalue1234567890",
            "bearer_token": "verylongbearertokenvalueabcdef",
            "anthropic_secret": "sk-ant-verylongsecretvalue123456",
            "url": "https://user:pass@example.com/path",
        },
    }

    redacted = redact_projection(payload)

    rendered = json.dumps(redacted.value)
    assert "sk-testsecretvalue" not in rendered
    assert "thisisaverylongsecretvalue" not in rendered
    assert "github_pat_" not in rendered
    assert "verylongbearertokenvalue" not in rendered
    assert "verylongaccesstokenvalue" not in rendered
    assert "verylongrefreshtokenvalue" not in rendered
    assert "plainsecretvalue" not in rendered
    assert "secretkeyvalue" not in rendered
    assert "apikeyvalue" not in rendered
    assert "wJalrXUtnFEMI" not in rendered
    assert "BEGINPRIVATEKEY" not in rendered
    assert "stripescretvalue" not in rendered
    assert "verylongsecretvalue" not in rendered
    assert "user:pass" not in rendered
    assert redacted.value["prompt_tokens"] == 123
    assert redacted.value["completion_tokens"] == 45
    assert redacted.value["cached_tokens"] == 6
    assert redacted.value["token_estimate"] == 789
    assert redacted.value["reasoning_tokens"] == 10
    assert redacted.manifest()["redacted"] is True
    rules = {item["rule"] for item in redacted.manifest()["rules"]}
    assert {"secret_key_name", "url_credentials"} <= rules


def test_persist_call_writes_private_full_and_redacted_refs(tmp_path):
    # Built by concatenation so the staged source never contains a literal PAT pattern.
    payload = {"tool": "run_command", "args": {"token": "ghp_" + "abcdefghijklmnopqrstuvwxyz123456"}}

    refs = persist_call(
        tmp_path,
        task_id="task-1",
        call_id="call-1",
        call_type="tool_call",
        payload=payload,
        manifest={"model": "test/model"},
    )

    manifest_path = tmp_path / "observability" / "calls" / "task-1" / "call-1.json"
    assert manifest_path.exists()
    if posix_private_modes_supported():
        assert os.stat(tmp_path / "observability").st_mode & 0o777 == 0o700
        assert os.stat(tmp_path / "observability" / "blobs").st_mode & 0o777 == 0o700
        assert os.stat(manifest_path.parent).st_mode & 0o777 == 0o700
        assert os.stat(manifest_path).st_mode & 0o777 == 0o600

    redacted_path = refs["redacted_projection_ref"]["path"]
    if posix_private_modes_supported():
        assert os.stat(redacted_path).st_mode & 0o777 == 0o600
    assert "full_payload_ref" not in refs
    assert "redacted_projection" not in refs
    assert _read_gzip_json(redacted_path)["args"]["token"] == "***REDACTED***"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Default: the authoritative blob is REDACTED so no raw secret lands on disk;
    # structure is preserved and the redaction is declared honestly.
    assert manifest["full_payload_redacted"] is True
    assert refs["full_payload_redacted"] is True
    full_path = manifest["full_payload_ref"]["path"]
    if posix_private_modes_supported():
        assert os.stat(full_path).st_mode & 0o777 == 0o600
    assert _read_gzip_json(full_path)["args"]["token"] == "***REDACTED***"
    assert manifest["call_type"] == "tool_call"
    assert manifest["redaction"]["redacted"] is True
    assert manifest["full_payload_ref"]["sha256"]
    assert refs["manifest_ref"]["sha256"] == __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest()


def test_persist_call_keep_raw_env_persists_unredacted(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_OBSERVABILITY_KEEP_RAW", "1")
    payload = {"tool": "run_command", "args": {"token": "ghp_" + "abcdefghijklmnopqrstuvwxyz123456"}}
    refs = persist_call(
        tmp_path, task_id="task-1", call_id="call-1", call_type="tool_call",
        payload=payload, manifest={"model": "test/model"},
    )
    manifest_path = tmp_path / "observability" / "calls" / "task-1" / "call-1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # KEEP_RAW: authoritative blob holds the raw payload; the redacted projection is separate.
    assert manifest["full_payload_redacted"] is False
    assert refs["full_payload_redacted"] is False
    assert _read_gzip_json(manifest["full_payload_ref"]["path"])["args"]["token"].startswith("ghp_")
    assert _read_gzip_json(refs["redacted_projection_ref"]["path"])["args"]["token"] == "***REDACTED***"
    assert manifest["full_payload_ref"]["path"] != refs["redacted_projection_ref"]["path"]


def test_write_blob_accepts_concurrent_same_payload_publish(tmp_path):
    payload = {"message": "same reviewer response", "usage": {"prompt_tokens": 1}}

    with ThreadPoolExecutor(max_workers=8) as pool:
        refs = list(pool.map(lambda _: write_blob(tmp_path, payload), range(16)))

    assert len({ref["sha256"] for ref in refs}) == 1
    assert all(os.path.exists(ref["path"]) for ref in refs)


def test_loop_outcome_distinguishes_success_empty_and_provider_failure():
    ok = derive_loop_outcome("done", {"rounds": 1}, {"tool_calls": []})
    assert ok["outcome_axes"]["execution"]["status"] == EXECUTION_OK
    assert ok["outcome_axes"]["objective"]["status"] == OBJECTIVE_NOT_EVALUATED
    assert ok["failure"] is None

    empty = derive_loop_outcome("", {"rounds": 1}, {"tool_calls": []})
    assert empty["outcome_axes"]["execution"]["status"] == EXECUTION_FAILED
    assert empty["reason_code"] == "empty_final_text"

    infra = derive_loop_outcome(
        "ignored",
        {"result_status": RESULT_INFRA_FAILED, "reason_code": "llm_api_error"},
        {"tool_calls": []},
    )
    assert infra["outcome_axes"]["execution"]["status"] == EXECUTION_INFRA_FAILED
    assert infra["failure"]["kind"] == "provider"

    runtime_error = derive_loop_outcome(
        "⚠️ Error during processing: RuntimeError: boom",
        {"rounds": 1},
        {"tool_calls": []},
    )
    assert runtime_error["outcome_axes"]["execution"]["status"] == EXECUTION_INFRA_FAILED
    assert runtime_error["reason_code"] == "task_exception"

    deep_unavailable = derive_loop_outcome(
        "❌ Deep self-review unavailable: no key",
        {},
        {"tool_calls": []},
    )
    assert deep_unavailable["outcome_axes"]["execution"]["status"] == EXECUTION_INFRA_FAILED
    assert deep_unavailable["reason_code"] == "deep_self_review_unavailable"

    tool_failure = derive_loop_outcome(
        "Created the file.",
        {"rounds": 2},
        {"tool_calls": [{
            "tool": "run_command",
            "is_error": True,
            "status": "artifact_output_error",
            "result": "⚠️ ARTIFACT_OUTPUT_ERROR: undeclared output",
        }]},
    )
    assert tool_failure["outcome_axes"]["execution"]["status"] == EXECUTION_DEGRADED
    assert tool_failure["reason_code"] == "tool_failure"
    assert tool_failure["failure"]["kind"] == "tool"
    assert tool_failure["failure"]["tool_errors"][0]["status"] == "artifact_output_error"

    answer_with_tool_error = derive_loop_outcome(
        "FINAL ANSWER: 42",
        {"rounds": 2},
        {"tool_calls": [{
            "tool": "run_command",
            "is_error": True,
            "status": "tool_failure",
            "result": "bad probe",
        }]},
    )
    assert answer_with_tool_error["outcome_axes"]["execution"]["status"] == EXECUTION_DEGRADED
    assert answer_with_tool_error["outcome_axes"]["execution"]["reason_code"] == "tool_failure"
    assert answer_with_tool_error["reason_code"] == "final_message"
    assert answer_with_tool_error["failure"] is None
    assert answer_with_tool_error["final_answer"] == "42"

    stale_latch = derive_loop_outcome(
        "I kept working and hit a tool issue.",
        {"rounds": 2},
        {
            "best_valid_final_answer": "draft",
            "best_valid_final_answer_tools": 0,
            "tool_calls": [{
                "tool": "run_command",
                "is_error": True,
                "status": "tool_failure",
                "result": "bad probe",
            }],
        },
    )
    assert stale_latch["final_answer"] == ""
    assert stale_latch["reason_code"] == "tool_failure"

    # A2 (v6.50.2): an access-policy block on a READ-ONLY exploratory tool is honest
    # telemetry, not a degraded execution — the agent simply could not look there. It is
    # routed to a fully-ignored bucket (recorded as ignored_tool_errors) and never sets
    # tool_failure or a residual warning.
    for tool_name, status in (
        ("web_search", "resource_constraint_blocked"),
        ("read_file", "resource_policy_blocked"),
    ):
        policy_block = derive_loop_outcome(
            "Done.",
            {"rounds": 1},
            {"tool_calls": [{
                "tool": tool_name,
                "is_error": True,
                "status": status,
                "result": f"⚠️ {status.upper()}: blocked",
            }]},
        )
        execution = policy_block["outcome_axes"]["execution"]
        assert execution["status"] == EXECUTION_OK
        assert policy_block["reason_code"] == "final_message"
        assert policy_block["failure"] is None
        assert execution["ignored_tool_errors"][0]["status"] == status

    # v6.57.0 (1.3): a POLICY refusal on a write/shell tool (run_command) is telemetry,
    # NOT a degraded execution and NOT a tool_failure headline — the runtime said "no" to
    # this action; the agent's work is judged on the objective/review axis. It lands in the
    # dedicated policy_denials bucket (owner-approved: the site-presentation incident where
    # integration_blocked/LIST_FILES forced degraded/tool_failure over a shipped site).
    policy_write_block = derive_loop_outcome(
        "Done.",
        {"rounds": 1},
        {"tool_calls": [{
            "tool": "run_command",
            "is_error": True,
            "status": "resource_policy_blocked",
            "result": "⚠️ RESOURCE_POLICY_BLOCKED: blocked",
        }]},
    )
    execution = policy_write_block["outcome_axes"]["execution"]
    assert execution["status"] == EXECUTION_OK
    assert policy_write_block["reason_code"] == "final_message"
    assert policy_write_block["failure"] is None
    assert execution["policy_denials"][0]["status"] == "resource_policy_blocked"

    # Boundary: a GENUINE tool/exec error (not a policy refusal) on a write/shell tool
    # STILL degrades — the policy_denials demotion is scoped to `*_blocked` refusals.
    real_error = derive_loop_outcome(
        "Done.",
        {"rounds": 1},
        {"tool_calls": [{
            "tool": "run_command",
            "is_error": True,
            "status": "error",
            "result": "⚠️ unexpected error",
        }]},
    )
    assert real_error["outcome_axes"]["execution"]["status"] == EXECUTION_DEGRADED
    assert real_error["reason_code"] == "tool_failure"


def test_forced_finalization_with_answer_is_best_effort():
    """Deadline/budget/round-limit forced finalization with a REAL extracted
    model answer is the typed best_effort shelf, not a failure. Structural
    gate: forced reason code + the loop's typed _best_effort_extracted fact +
    non-empty non-error text."""
    from ouroboros.outcomes import EXECUTION_BEST_EFFORT, derive_loop_outcome

    for reason in ("budget_exhausted", "round_limit", "finalization_grace", "deadline_local"):
        outcome = derive_loop_outcome(
            "Partial result: 3 of 5 modules fixed. Unverified: integration tests.",
            {"execution_status": "failed", "reason_code": reason, "_best_effort_extracted": True},
            {"tool_calls": [], "reasoning_notes": []},
        )
        execution = outcome["outcome_axes"]["execution"]
        assert execution["status"] == EXECUTION_BEST_EFFORT, reason
        assert execution["reason_code"] == reason
        assert execution["failure"] is None

    # Host fallback text WITHOUT the typed extraction fact stays failed —
    # "🚫 Task rejected..." / "Task spent $..." strings are not model answers.
    fallback = derive_loop_outcome(
        "🚫 Task rejected. Total budget exhausted. Please increase TOTAL_BUDGET in settings.",
        {"execution_status": "failed", "reason_code": "budget_exhausted"},
        {"tool_calls": [], "reasoning_notes": []},
    )
    assert fallback["outcome_axes"]["execution"]["status"] == "failed"
    unmarked = derive_loop_outcome(
        "Task spent $4.5 (>50% of remaining $8.0). Budget exhausted.",
        {"execution_status": "failed", "reason_code": "budget_exhausted"},
        {"tool_calls": [], "reasoning_notes": []},
    )
    assert unmarked["outcome_axes"]["execution"]["status"] == "failed"

    # Empty text stays a real failure even with the fact set.
    empty = derive_loop_outcome(
        "", {"execution_status": "failed", "reason_code": "round_limit", "_best_effort_extracted": True},
        {"tool_calls": [], "reasoning_notes": []},
    )
    assert empty["outcome_axes"]["execution"]["status"] == "failed"

    # Error-marker text stays a failure even with the fact set — BOTH marker
    # families (⚠️ and ❌, e.g. deep-self-review error strings) are rejected.
    for marker_text in (
        "⚠️ Task exceeded MAX_ROUNDS (200).",
        "❌ Deep self-review unavailable: configure OUROBOROS_MODEL_DEEP_SELF_REVIEW.",
    ):
        marker = derive_loop_outcome(
            marker_text,
            {"execution_status": "failed", "reason_code": "round_limit", "_best_effort_extracted": True},
            {"tool_calls": [], "reasoning_notes": []},
        )
        assert marker["outcome_axes"]["execution"]["status"] == "failed", marker_text

    # Non-forced failure reasons keep failing regardless of text/fact.
    other = derive_loop_outcome(
        "some text",
        {"execution_status": "failed", "reason_code": "llm_empty_response", "_best_effort_extracted": True},
        {"tool_calls": [], "reasoning_notes": []},
    )
    assert other["outcome_axes"]["execution"]["status"] == "failed"


def test_outcome_tier_aggregation_and_objective_mapping():
    """Reviewer outcome_tier drives the objective axis: solved->pass (with the
    false-solved veto), best_effort->best_effort, blocked_with_evidence->fail."""
    from ouroboros.outcomes import OBJECTIVE_BEST_EFFORT, derive_loop_outcome

    def _trace(tier: str, signal: str = "PASS"):
        return {
            "tool_calls": [], "reasoning_notes": [],
            "review_runs": [{
                "aggregate_signal": signal,
                "actors": [{"status": "ok", "parsed": {"verdict": signal, "outcome_tier": tier}}],
            }],
        }

    solved = derive_loop_outcome("done", {}, _trace("solved", "PASS"))
    assert solved["outcome_axes"]["objective"]["status"] == "pass"
    assert solved["outcome_axes"]["objective"]["outcome_tier"] == "solved"

    best = derive_loop_outcome("partial", {}, _trace("best_effort", "DEGRADED"))
    assert best["outcome_axes"]["objective"]["status"] == OBJECTIVE_BEST_EFFORT

    blocked = derive_loop_outcome("blocked", {}, _trace("blocked_with_evidence", "FAIL"))
    assert blocked["outcome_axes"]["objective"]["status"] == "fail"

    # False-solved veto: reviewer verdict FAIL blocks a solved tier claim.
    veto = derive_loop_outcome("claims done", {}, _trace("solved", "FAIL"))
    assert veto["outcome_axes"]["objective"]["status"] == "fail"

    # Conservative: a solved claim inside a DEGRADED review (quorum not met)
    # keeps the pre-feature degraded objective, never upgrades to pass.
    degraded_solved = derive_loop_outcome("claims done", {}, _trace("solved", "DEGRADED"))
    assert degraded_solved["outcome_axes"]["objective"]["status"] == "degraded"

    # Worst-tier-wins across actors.
    multi = {
        "tool_calls": [], "reasoning_notes": [],
        "review_runs": [{
            "aggregate_signal": "DEGRADED",
            "actors": [
                {"status": "ok", "parsed": {"outcome_tier": "solved"}},
                {"status": "ok", "parsed": {"outcome_tier": "best_effort"}},
            ],
        }],
    }
    agg = derive_loop_outcome("x", {}, multi)
    assert agg["outcome_axes"]["review"]["outcome_tier"] == "best_effort"

    # No tier -> legacy review-status mapping unchanged.
    legacy = derive_loop_outcome("x", {}, {
        "tool_calls": [], "reasoning_notes": [],
        "review_runs": [{"aggregate_signal": "PASS", "actors": [{"status": "ok", "parsed": {"verdict": "PASS"}}]}],
    })
    assert legacy["outcome_axes"]["objective"]["status"] == "pass"
    assert "outcome_tier" not in legacy["outcome_axes"]["objective"]


def test_outcome_tier_quorum_pass_not_poisoned_by_dissenting_slot():
    """A quorum PASS must take its tier from the CONTRIBUTING PASS actors only; a
    single dissenting/degraded slot's pessimistic tier must not drag the objective
    to fail (same non-surrender rule as the aggregate-signal quorum)."""
    from ouroboros.outcomes import derive_loop_outcome

    trace = {
        "tool_calls": [], "reasoning_notes": [],
        "review_runs": [{
            "aggregate_signal": "PASS",  # 2-of-3 PASS quorum
            "actors": [
                {"status": "ok", "signal": "PASS", "parsed": {"verdict": "PASS", "outcome_tier": "solved"}},
                {"status": "ok", "signal": "PASS", "parsed": {"verdict": "PASS", "outcome_tier": "solved"}},
                # Dissenting slot: degraded, pessimistic tier — must be ignored on a PASS run.
                {"status": "ok", "signal": "DEGRADED", "parsed": {"outcome_tier": "blocked_with_evidence"}},
            ],
        }],
    }
    out = derive_loop_outcome("done", {}, trace)
    assert out["outcome_axes"]["objective"]["status"] == "pass"
    assert out["outcome_axes"]["objective"]["outcome_tier"] == "solved"


def test_final_answer_extraction():
    from ouroboros.outcomes import derive_loop_outcome, extract_final_answer

    assert extract_final_answer("reasoning...\nFINAL ANSWER: 1.456") == "1.456"
    # Last marker wins; surrounding whitespace trimmed.
    assert extract_final_answer("FINAL ANSWER: draft\nmore...\n  FINAL ANSWER: cornstarch  ") == "cornstarch"
    assert extract_final_answer("no marker here") == ""

    outcome = derive_loop_outcome(
        "Computed the value.\nFINAL ANSWER: 42", {}, {"tool_calls": [], "reasoning_notes": []},
    )
    assert outcome["final_answer"] == "42"


def test_final_answer_rejects_internal_outcome_tier_tokens():
    """Structural invariant: snake_case outcome-tier ledger identifiers are never a
    deliverable (a v6.56.0 GAIA run shipped `FINAL ANSWER: blocked_with_evidence`
    verbatim after an acceptance downgrade). They count as missing so the marker
    nudge / salvage path can recover a real answer. `solved` stays extractable —
    it is an ordinary English word that can be a legitimate answer."""
    from ouroboros.outcomes import derive_loop_outcome, extract_final_answer

    assert extract_final_answer("FINAL ANSWER: blocked_with_evidence") == ""
    assert extract_final_answer("FINAL ANSWER:  Best_Effort ") == ""
    assert extract_final_answer("FINAL ANSWER: solved") == "solved"
    # A real answer that merely CONTAINS a tier word is untouched.
    assert extract_final_answer("FINAL ANSWER: best effort estimate") == "best effort estimate"

    outcome = derive_loop_outcome(
        "downgraded.\nFINAL ANSWER: blocked_with_evidence", {},
        {"tool_calls": [], "reasoning_notes": []},
    )
    assert outcome["final_answer"] == ""
    assert outcome["final_answer_missing_sentinel"] is True


def test_normalize_outcome_axes_preserves_best_effort_legacy():
    from ouroboros.outcomes import EXECUTION_BEST_EFFORT, normalize_outcome_axes

    axes = normalize_outcome_axes({"result_status": "best_effort", "reason_code": "round_limit"})
    assert axes["execution"]["status"] == EXECUTION_BEST_EFFORT


def test_normalize_outcome_axes_canonicalizes_partial_and_unknown_legacy():
    axes = normalize_outcome_axes({
        "status": "running",
        "artifact_status": "finalizing",
        "outcome_axes": {
            "objective": {"status": "pass", "source": "task_acceptance_review"},
        },
    })

    assert axes["lifecycle"]["status"] == "running"
    assert axes["artifacts"]["status"] == "finalizing"
    assert axes["execution"]["status"] == EXECUTION_OK
    assert axes["objective"]["status"] == "pass"
    assert axes["review"]["status"] == "skipped"

    preserved_artifacts = normalize_outcome_axes({
        "status": "completed",
        "outcome_axes": {"artifacts": {"status": "failed", "error_count": 1}},
    })
    assert preserved_artifacts["artifacts"]["status"] == "failed"
    assert preserved_artifacts["artifacts"]["error_count"] == 1

    explicit_artifacts = normalize_outcome_axes({
        "status": "completed",
        "artifact_bundle": {"status": "ready_no_changes"},
        "outcome_axes": {"artifacts": {"status": "failed"}},
    })
    assert explicit_artifacts["artifacts"]["status"] == "ready_no_changes"

    legacy = normalize_outcome_axes({"status": "completed", "result_status": "mystery"})
    assert legacy["execution"]["status"] == EXECUTION_DEGRADED
    assert legacy["execution"]["legacy_status"] == "mystery"

    cancelled = normalize_outcome_axes({"status": "cancelled"})
    assert cancelled["execution"]["status"] == "cancelled"
    assert cancelled["execution"]["reason_code"] == "cancelled"
    cancel_requested = normalize_outcome_axes({"status": "cancel_requested"})
    assert cancel_requested["execution"]["status"] == "cancelled"
    assert cancel_requested["execution"]["reason_code"] == "cancel_requested"
    duplicate = normalize_outcome_axes({"status": "rejected_duplicate"})
    assert duplicate["execution"]["status"] == EXECUTION_OK
    assert duplicate["execution"]["reason_code"] == "scheduler_duplicate_rejection"
    legacy_cancelled = normalize_outcome_axes({"status": "completed", "result_status": "cancelled"})
    assert legacy_cancelled["execution"]["status"] == "cancelled"

    forged = normalize_outcome_axes({
        "status": "completed",
        "outcome_axes": {"objective": {"status": "pass", "source": "manual"}},
    })
    assert forged["objective"]["status"] == OBJECTIVE_NOT_EVALUATED
    assert forged["objective"]["source"] == "none"
    assert forged["objective"]["ignored_status"] == "pass"

    recovered = derive_loop_outcome(
        "Created the file.",
        {"rounds": 3},
        {"tool_calls": [
            {
                "tool": "edit_text",
                "args": {"root": "user_files", "path": "Desktop/report.html"},
                "is_error": True,
                "status": "edit_text_blocked",
                "result": "⚠️ EDIT_TEXT_ERROR: old_str matched 0 times",
            },
            {
                "tool": "write_file",
                "args": {"root": "user_files", "path": "Desktop/report.html"},
                "is_error": False,
                "status": "ok",
                "artifact_registered": True,
                "result": "OK: wrote user_files:Desktop/report.html\nARTIFACT_OUTPUTS: registered user file -> artifact_store:report.html",
            },
        ]},
    )
    assert recovered["outcome_axes"]["execution"]["status"] == EXECUTION_OK
    assert recovered["failure"] is None
    assert recovered["outcome_axes"]["execution"]["recoveries"][0]["status"] == "edit_text_blocked"
    assert recovered["outcome_axes"]["execution"]["recoveries"][0]["recovered_by_call_index"] == 2

    # T4 (v6.35.0): an unrecovered one-shot run_command non-zero exit is COSMETIC,
    # not a degraded execution. The error is preserved on the execution axis for
    # monitoring; because no acceptance review ran, the objective carries a
    # structural warning so the default-`auto` path still flags a possible
    # overclaim (honesty moves to the LLM review axis — Bible P5).
    cosmetic_shell = derive_loop_outcome(
        "Created another file.",
        {"rounds": 3},
        {"tool_calls": [
            {
                "tool": "run_command",
                "args": {"cmd": "python3 build_report.py", "outputs": ["report.html"]},
                "is_error": True,
                "status": "non_zero_exit",
                "result": "⚠️ SHELL_EXIT_ERROR: command exited with exit_code=1.",
            },
            {
                "tool": "write_file",
                "args": {"root": "user_files", "path": "Desktop/other.html"},
                "is_error": False,
                "status": "ok",
                "artifact_registered": True,
                "result": "OK: wrote user_files:Desktop/other.html\nARTIFACT_OUTPUTS: registered user file -> artifact_store:other.html",
            },
        ]},
    )
    assert cosmetic_shell["outcome_axes"]["execution"]["status"] == EXECUTION_OK
    assert cosmetic_shell["failure"] is None
    assert cosmetic_shell["outcome_axes"]["execution"]["cosmetic_tool_errors"][0]["status"] == "non_zero_exit"
    assert cosmetic_shell["outcome_axes"]["objective"]["warning"] == "residual_tool_errors_without_review"

    cleanup_failure = derive_loop_outcome(
        "Done",
        {"rounds": 2},
        {
            "tool_calls": [],
            "verification_events": [{
                "kind": "services_stopped",
                "services": [{
                    "name": "devserver",
                    "artifact_output_failed": True,
                    "artifact_outputs": "⚠️ ARTIFACT_OUTPUT_ERROR:\n- missing output: report.html",
                }],
            }],
        },
    )
    assert cleanup_failure["outcome_axes"]["execution"]["status"] == EXECUTION_DEGRADED
    assert cleanup_failure["failure"]["kind"] == "verification"


def test_t4_cosmetic_partition_guards():
    # A GENUINE tool error STILL degrades (partition is structural).
    blocking = derive_loop_outcome(
        "Done",
        {"rounds": 2},
        {"tool_calls": [{
            "tool": "write_file",
            "args": {"root": "active_workspace", "path": "x.py"},
            "is_error": True,
            "status": "error",
            "result": "⚠️ unexpected write error",
        }]},
    )
    assert blocking["outcome_axes"]["execution"]["status"] == EXECUTION_DEGRADED

    # v6.57.0 (1.3): a POLICY refusal (`write_file_blocked`) is NOT degrading — it lands in
    # policy_denials telemetry (the runtime declined the write; the deliverable, if any, is
    # judged on the objective/review axis).
    policy_blocked = derive_loop_outcome(
        "Done",
        {"rounds": 2},
        {"tool_calls": [{
            "tool": "write_file",
            "args": {"root": "active_workspace", "path": "x.py"},
            "is_error": True,
            "status": "write_file_blocked",
            "result": "⚠️ WRITE_FILE_BLOCKED: protected path",
        }]},
    )
    assert policy_blocked["outcome_axes"]["execution"]["status"] == EXECUTION_OK
    assert policy_blocked["outcome_axes"]["execution"]["policy_denials"][0]["status"] == "write_file_blocked"

    # A failed-then-rerun-IDENTICAL run_command is RECOVERED, not cosmetic.
    recovered = derive_loop_outcome(
        "Built it.",
        {"rounds": 2},
        {"tool_calls": [
            {"tool": "run_command", "args": {"cmd": "make build"}, "is_error": True,
             "status": "non_zero_exit", "result": "⚠️ SHELL_EXIT_ERROR: exit_code=1."},
            {"tool": "run_command", "args": {"cmd": "make build"}, "is_error": False,
             "status": "ok", "result": "ok"},
        ]},
    )
    assert recovered["outcome_axes"]["execution"]["status"] == EXECUTION_OK
    assert not recovered["outcome_axes"]["execution"]["cosmetic_tool_errors"]
    assert recovered["outcome_axes"]["execution"]["recoveries"]
    assert "warning" not in recovered["outcome_axes"]["objective"]

    # When an acceptance review DID run, the residual warning is suppressed
    # (the review axis already judged "did it actually work?").
    reviewed = derive_loop_outcome(
        "Done",
        {"rounds": 2},
        {
            "tool_calls": [{
                "tool": "run_command", "args": {"cmd": "find /nope"}, "is_error": True,
                "status": "non_zero_exit", "result": "⚠️ SHELL_EXIT_ERROR: exit_code=1.",
            }],
            "review_runs": [{"aggregate_signal": "PASS"}],  # a review ran -> objective judged
            "review_decision": {"eligibility": "eligible", "trigger": "agent_called_tool_result"},
        },
    )
    assert reviewed["outcome_axes"]["execution"]["status"] == EXECUTION_OK
    assert reviewed["outcome_axes"]["execution"]["cosmetic_tool_errors"]
    assert "warning" not in reviewed["outcome_axes"]["objective"]


def test_tool_arg_sanitizer_uses_value_pattern_redactor():
    args = {
        "cmd": "curl -H 'Authorization: Bearer verylongbearertokenvalue1234567890' https://x",
        "script": "OPENROUTER_API_KEY=sk-or-thisisaverylongsecretvalue1234567890",
    }

    rendered = json.dumps(sanitize_tool_args_for_log("run_command", args))

    assert "verylongbearertokenvalue" not in rendered
    assert "sk-or-thisisaverylongsecret" not in rendered
    assert "***REDACTED***" in rendered


def test_latest_llm_response_text_finds_real_producer_manifests(tmp_path):
    """Glob<->producer coupling (scope-review finding): a REAL observability
    producer run must yield a manifest the kill-path salvage glob matches —
    if the producer naming ever changes, this breaks loudly instead of the
    salvage silently returning ''."""
    from ouroboros.llm_observability import chat_observed
    from ouroboros.observability import latest_llm_response_text, new_call_id, persist_call

    class _StubLLM:
        def chat(self, **_kwargs):
            return {"content": "real producer text"}, {"prompt_tokens": 1}

    # Producer 1: chat_observed (consciousness/deep-review/etc. lane).
    chat_observed(
        _StubLLM(), drive_root=tmp_path, task_id="t-real", call_type="llm_call",
        messages=[{"role": "user", "content": "hi"}], model="stub",
    )
    assert latest_llm_response_text(tmp_path, "t-real") == "real producer text"

    # Producer 2: the main-loop naming scheme (loop_llm_call.py builds
    # new_call_id("llm") then persists f"{llm_call_id}_response").
    llm_call_id = new_call_id("llm")
    persist_call(
        tmp_path, task_id="t-loop", call_id=f"{llm_call_id}_response",
        call_type="llm_response",
        payload={"message": {"content": "loop producer text"}, "usage": {}},
    )
    assert latest_llm_response_text(tmp_path, "t-loop") == "loop producer text"


def test_latest_llm_response_text_salvages_newest_assistant_text(tmp_path):
    """The supervisor kill path salvages the most recent persisted assistant
    content; empty responses are skipped, missing dirs return ""."""
    import time as _time

    from ouroboros.observability import latest_llm_response_text, persist_call

    assert latest_llm_response_text(tmp_path, "nope") == ""

    persist_call(
        tmp_path, task_id="t-salvage", call_id="llm_a_response", call_type="llm_response",
        payload={"message": {"content": "older progress"}, "usage": {}},
    )
    _time.sleep(0.02)
    persist_call(
        tmp_path, task_id="t-salvage", call_id="llm_b_response", call_type="llm_response",
        payload={"message": {"content": ""}, "usage": {}},  # empty: skipped
    )
    _time.sleep(0.02)
    persist_call(
        tmp_path, task_id="t-salvage", call_id="llm_c_response", call_type="llm_response",
        payload={"message": {"content": "newest real progress"}, "usage": {}},
    )

    assert latest_llm_response_text(tmp_path, "t-salvage") == "newest real progress"


def test_latest_llm_response_text_scans_past_many_empty_tool_rounds(tmp_path):
    """Long tool-driven tasks have dozens of newest empty-content responses;
    the salvage must scan past ALL of them to the older real text."""
    import os as _os
    import pathlib

    from ouroboros.observability import latest_llm_response_text, persist_call

    ref = persist_call(
        tmp_path, task_id="t-deep", call_id="llm_000_response", call_type="llm_response",
        payload={"message": {"content": "the real early answer"}, "usage": {}},
    )
    base = pathlib.Path(str(ref["manifest_ref"]["path"])).parent / "llm_000_response.json"
    _os.utime(base, (1_000_000, 1_000_000))  # oldest
    for idx in range(30):
        r = persist_call(
            tmp_path, task_id="t-deep", call_id=f"llm_{idx + 1:03d}_response", call_type="llm_response",
            payload={"message": {"content": "", "tool_calls": [{"id": f"c{idx}"}]}, "usage": {}},
        )
        p = pathlib.Path(str(r["manifest_ref"]["path"]))
        _os.utime(p, (2_000_000 + idx, 2_000_000 + idx))  # all newer

    assert latest_llm_response_text(tmp_path, "t-deep") == "the real early answer"


def test_supervisor_task_drive_resolution_prefers_child_drive(tmp_path, monkeypatch):
    """finalize_now controls and kill-salvage must target the task's ACTIVE
    drive: child drive for forked/workspace tasks, canonical otherwise."""
    from supervisor import queue as queue_mod

    monkeypatch.setattr(queue_mod, "DRIVE_ROOT", str(tmp_path))

    child = tmp_path / "state" / "headless_tasks" / "t1" / "data"
    assert queue_mod._task_drive_for_task({"child_drive_root": str(child)}, "t1") == child
    assert queue_mod._task_drive_for_task({"drive_root": str(child)}, "t1") == child

    # No in-memory fields: falls back to the durable task result record.
    from ouroboros.task_results import write_task_result
    write_task_result(tmp_path, "t2", "running", child_drive_root=str(child))
    assert queue_mod._task_drive_for_task({}, "t2") == child

    # Nothing anywhere: canonical drive.
    assert queue_mod._task_drive_for_task({}, "t3") == tmp_path


def test_kill_salvage_truncation_carries_omission_note():
    """Salvaged text entering the durable terminal result must use the
    explicit OMISSION NOTE helper, never silent clipping."""
    from ouroboros.utils import truncate_review_artifact

    long_text = "x" * 5000
    out = truncate_review_artifact(long_text, 4000)
    assert "OMISSION NOTE" in out
    assert "original length 5000" in out


def test_loop_outcome_trace_refs_include_llm_and_tool_refs():
    outcome = derive_loop_outcome(
        "done",
        {
            "execution_id": "exec_1",
            "rounds": 1,
            "llm_call_refs": [{
                "llm_call_id": "llm_1",
                "round_id": "exec_1:round:1",
                "request_ref": {"path": "req"},
                "response_ref": {"path": "resp"},
            }],
        },
        {
            "tool_calls": [{
                "trace_ref": {
                    "call_id": "tool_1",
                    "manifest_ref": {"path": "tool"},
                    "redacted_projection_ref": {"path": "redacted"},
                }
            }]
        },
    )

    refs = outcome["trace_refs"]
    assert refs["execution_id"] == "exec_1"
    assert refs["llm_call_refs"][0]["llm_call_id"] == "llm_1"
    assert refs["tool_call_refs"][0]["call_id"] == "tool_1"


def test_artifact_bundle_and_large_verification_ledger_artifact(tmp_path):
    patch_path = tmp_path / "fix.patch"
    patch_path.write_text("diff", encoding="utf-8")
    bundle = artifact_bundle_from_result({
        "artifact_status": "ready",
        "artifacts": [{"kind": "patch", "name": "fix.patch", "path": str(patch_path), "size": 4, "sha256": "abcd"}],
    })
    assert bundle["status"] == "ready"
    assert bundle["artifacts"][0]["kind"] == "patch"

    mixed = artifact_bundle_from_result({
        "artifact_status": "failed",
        "artifact_error": "patch failed",
        "artifacts": [{"kind": "verification_ledger", "name": "verification_ledger.json", "path": "/tmp/ledger"}],
    })
    assert mixed["status"] == "failed"
    assert mixed["artifacts"][0]["status"] == "missing"

    preserved_axis = artifact_bundle_from_result({
        "outcome_axes": {"artifacts": {"status": "failed", "error_count": 1}},
    })
    assert preserved_axis["status"] == "failed"

    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.task_results import load_task_result, write_task_result

    write_task_result(
        tmp_path,
        "task-preserve-axis",
        "running",
        outcome_axes={"artifacts": {"status": "failed", "error_count": 1}},
    )
    _store_task_result(
        SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path),
        {"id": "task-preserve-axis", "type": "task", "text": "store result"},
        "done",
        {"rounds": 1, "cost": 0},
        {"tool_calls": [], "reasoning_notes": []},
        review_evidence={},
    )
    stored = load_task_result(tmp_path, "task-preserve-axis")
    assert stored["artifact_bundle"]["status"] == "failed"
    assert stored["outcome_axes"]["artifacts"]["status"] == "failed"
    assert stored["outcome_axes"]["artifacts"]["error_count"] == 1

    ledger = {"schema_version": 1, "created_at": "now", "task_id": "task-1", "entries": [{"x": "y" * 200}]}
    refs = maybe_write_verification_artifact(tmp_path, "task-1", ledger, threshold_chars=20)
    assert refs["inline"]["omitted_to_artifact"] is True
    artifact = refs["artifact"]
    assert artifact["status"] == "ready"
    assert artifact["path"].endswith("verification_ledger.json")
    assert os.path.exists(artifact["path"])

    refreshed = refresh_verification_ledger_artifacts(
        {
            "schema_version": 2,
            "outcome_axes": {"objective": {"status": OBJECTIVE_NOT_EVALUATED, "source": "none"}},
            "entries": [
                {"kind": "objective_outcome", "status": OBJECTIVE_NOT_EVALUATED},
                {"kind": "task_contract", "status": "draft"},
            ],
        },
        {"status": "ready_with_changes", "artifacts": [], "errors": []},
    )
    assert refreshed["outcome_axes"]["artifacts"]["status"] == "ready_with_changes"
    assert refreshed["summary"]["has_failures"] is False

    long_objective = "preserve " + ("full objective " * 80)
    ledger = build_verification_ledger(
        task={
            "id": "task-contract",
            "task_contract": {
                "status": "draft",
                "objective": long_objective,
                "expected_output": "full expected output",
            },
        },
        loop_outcome={"outcome_axes": normalize_outcome_axes({})},
        llm_trace={},
        artifact_bundle={"status": "not_applicable", "artifacts": [], "errors": []},
    )
    contract_entry = next(item for item in ledger["entries"] if item.get("kind") == "task_contract")
    assert contract_entry["status"] == "recorded"
    assert contract_entry["contract_status"] == "draft"
    assert contract_entry["objective"] == long_objective
    assert ledger["summary"]["has_failures"] is False


def _ledger_with_receipt_status(status: str) -> dict:
    """A minimal v2 ledger whose only entry is a verification_receipt of the given status."""
    return {
        "schema_version": 2,
        "outcome_axes": {"objective": {"status": OBJECTIVE_NOT_EVALUATED, "source": "none"}},
        "entries": [{"kind": "verification_receipt", "status": status}],
    }


def test_has_failures_false_for_observed_receipt():
    # A successful artifact_observation grounding (status=observed) must NOT read as a
    # ledger failure (regression: observed was missing from the success allow-list).
    refreshed = refresh_verification_ledger_artifacts(
        _ledger_with_receipt_status("observed"),
        {"status": "ready", "artifacts": [], "errors": []},
    )
    assert refreshed["summary"]["has_failures"] is False


def test_has_failures_false_for_declared_receipt():
    # An honest no_visible_machine_contract declaration (status=declared) is a SUCCESS
    # grounding and must NOT read as a ledger failure.
    refreshed = refresh_verification_ledger_artifacts(
        _ledger_with_receipt_status("declared"),
        {"status": "ready", "artifacts": [], "errors": []},
    )
    assert refreshed["summary"]["has_failures"] is False


def test_has_failures_true_for_failed_receipt():
    # Sanity: the fix must NOT over-suppress — a real verify_and_record fail still flags.
    refreshed = refresh_verification_ledger_artifacts(
        _ledger_with_receipt_status("fail"),
        {"status": "ready", "artifacts": [], "errors": []},
    )
    assert refreshed["summary"]["has_failures"] is True
