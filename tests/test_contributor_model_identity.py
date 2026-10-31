"""Persisted final-attempt identity reaches contributor packets without changing verdicts."""

import copy
import hashlib
import json
import zipfile

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.gateways.claudexor import final_attempt_facts
from ouroboros.observability import persist_call
from ouroboros.review_session_usage import session_usage_facts
from scripts.contributor_review_evidence import (
    bind_execution_receipts,
    finalize_contributor_outcome,
    write_contributor_packet,
)

MODEL = "cursor-grok-4.6-xhigh"
LABEL = "Cursor Grok 4.6 Extra High"
RUN = "run-identity"
PROFILE = "cursor-profile"
TRANSCRIPT = "[]\nREVIEW_EOF_SENTINEL"


def _attempt(model=LABEL, harness="cursor"):
    return {"attempt_id": "a02", "harness_id": harness,
            "observed_model": model, "profile_id": PROFILE}


def _proved_attempt():
    # Engine/parent agreed wire: the same spawn's unique exact catalog label,
    # captured with the first observed_model, including its profile at that time.
    return {**_attempt(), "observed_model_catalog_match": {
        "label": LABEL, "id": MODEL, "source": "account_catalog", "profile_id": PROFILE}}


def _telemetry(root, attempt, *, run_id=RUN, extra=()):
    path = root / "engine-run" / "final" / "telemetry.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 2, "run_id": run_id, "final_attempt_id": "a02",
        "attempts": [*extra, attempt],
    }), encoding="utf-8")
    return {"summary": {"runDir": str(path.parent.parent), "model": MODEL,
                        "route": {"observedModel": MODEL, "verified": True}}}


def _usage(root, attempt, **telemetry):
    detail = _telemetry(root, attempt, **telemetry)
    observed = final_attempt_facts(detail, RUN)
    return {**session_usage_facts(RUN, requested_route="cursor", observed_attempt=observed),
            "applied_profile": observed.get("profile_id", ""),
            "applied_access": "readonly", "custody_durable": True,
            "verdict_provenance": {"raw_transcript_chars": len(TRANSCRIPT),
                "raw_transcript_sha256": hashlib.sha256(TRANSCRIPT.encode()).hexdigest()}}


def _persist(root, usage, *, requested=MODEL, harness="cursor", transcript=TRANSCRIPT,
             settle=True):
    target = f"{harness}={requested}"
    config = {"pool_slots": [{"slot_id": "s1", "effort": "high", "route": {
        "kind": "agent_session", "target_id": target, "profile_id": PROFILE}}]}
    prompt = persist_call(root, task_id="review", call_id="prompt", call_type="prompt",
        payload={"slot": {"slot_id": "s1", "model": target, "effort": "high",
            "route": "agent_session", "session_target": target, "session_profile": PROFILE}})
    response = persist_call(root, task_id="review", call_id="response", call_type="response",
        payload={"message": {"content": "[]", "session_transcript": transcript}, "usage": usage})
    run_id = usage["delegated_run_id"]
    custody.record_started(root, custody.RunCustody(
        run_id=run_id, task_id="review", project_id="identity-project",
        project_owned=True, ledger_root=str(root)))
    for event in (custody.LEDGER_RECORDED, custody.PROJECT_RETIRED):
        custody.emit(root, event, {"run_id": run_id})
    if settle:
        custody.emit(root, custody.SETTLED, {"run_id": run_id})
    custody._CUSTODY.clear()  # Exercise the persisted custody reader.
    return [("pool", {"slot_id": "s1", "status": "responded",
                       "prompt_ref": prompt, "response_ref": response})], config


def _packet(root, actors, config, *, outcome=None, exit_code=0):
    receipts, mismatches, transcripts = bind_execution_receipts(
        actors=actors, resolved_config=config, drive_root=root)
    code, final = finalize_contributor_outcome(
        outcome=outcome or {"status": "passed"}, exit_code=exit_code, mismatches=mismatches)
    output = root / "packet"
    output.mkdir()
    packet = write_contributor_packet(
        output_dir=output, snapshot={"base_sha": "a" * 40, "head_sha": "b" * 40},
        resolved_config=config, outcome=final, exit_code=code,
        evidence_refs=[actor["response_ref"] for _, actor in actors], cost_report={},
        elapsed_sec=0, seats=[], review_record={}, execution_receipts=receipts,
        execution_mismatches=mismatches, session_transcripts=transcripts,
        degraded_reasons=[], replacements=[(str(root), "$REVIEW_DRIVE")])
    with zipfile.ZipFile(packet) as archive:
        evidence = json.loads(archive.read("review-evidence.json"))
        assert json.loads(archive.read("outcome.json"))["exit_code"] == code
        text = archive.read("full-output.txt").decode().replace("\\n", "\n")
        for transcript in transcripts:
            assert transcript["transcript"] in text
    return code, final, evidence


@pytest.mark.parametrize("harness,requested,observed,verification,result", [
    ("cursor", MODEL, MODEL, "exact", "READY_FOR_INTEGRATION"),
    ("cursor", MODEL, LABEL, "observed_display_label", "INCOMPLETE"),
    ("codex", "gpt-5.6-sol", "GPT-5.6 Terra 300K High", "observed_display_label", "INCOMPLETE"),
    ("cursor", MODEL, "different-model", "mismatch", "INCOMPLETE"),
])
def test_legacy_identity_reaches_persisted_packet(tmp_path, harness, requested, observed, verification, result):
    usage = _usage(tmp_path, _attempt(observed, harness))
    actors, config = _persist(tmp_path, usage, requested=requested, harness=harness)
    _code, _final, packet = _packet(tmp_path, actors, config)

    assert packet["result"] == result
    receipt = packet["review_execution"]["receipts"][0]
    assert receipt["observed"]["model"] == observed
    assert receipt["model_verification"] == verification
    assert packet["review_execution"]["consistent"] is (verification == "exact")


@pytest.mark.parametrize("reason,code", [("critical_findings", 1), ("review_late_result_pending", 3)])
def test_unverified_identity_preserves_original_block_in_packet(tmp_path, reason, code):
    actors, config = _persist(tmp_path, _usage(tmp_path, _attempt()))
    original = {"status": "blocked", "block_reason": reason, "message": "Original review cause.",
                "combined_findings": [{"severity": "critical", "text": "Retain this finding."}]}
    returned, final, packet = _packet(tmp_path, actors, config, outcome=copy.deepcopy(original), exit_code=code)

    assert returned == code
    assert final["block_reason"] == reason
    assert final["message"].startswith(original["message"])
    assert final["combined_findings"] == original["combined_findings"]
    assert packet["result"] == ("BLOCKED" if code == 1 else "INCOMPLETE")
    assert any(item.startswith("model_identity_unverified:")
               for item in final["execution_receipt_mismatches"])


@pytest.mark.parametrize("fault,expected", [
    ("profile", "profile_mismatch:"),
    ("access", "readonly_access_unproven:"),
    ("custody", "custody_unproven:"),
    ("settlement", "session_settlement_unproven:"),
    ("transcript_absent", "session_transcript_absent:"),
    ("transcript_changed", "session_transcript_mismatch:"),
    ("response_tampered", "unreadable_response_receipt:"),
])
def test_identity_does_not_override_other_receipt_refusals(tmp_path, fault, expected):
    usage = _usage(tmp_path, _proved_attempt())
    if fault == "profile":
        usage["applied_profile"] = "other-profile"
    elif fault == "access":
        usage["applied_access"] = "workspace_write"
    elif fault == "custody":
        usage["custody_durable"] = False
    transcript = "" if fault == "transcript_absent" else "altered" if fault == "transcript_changed" else TRANSCRIPT
    actors, config = _persist(tmp_path, usage, transcript=transcript, settle=fault != "settlement")
    if fault == "response_tampered":
        actors[0][1]["response_ref"]["redacted_projection_ref"]["sha256"] = "0" * 64
    code, _final, packet = _packet(tmp_path, actors, config)

    assert code == 3 and packet["result"] == "INCOMPLETE"
    assert any(item.startswith(expected) for item in packet["review_execution"]["mismatches"])


@pytest.mark.parametrize("catalog_id,verification,result", [
    (MODEL, "catalog_match", "READY_FOR_INTEGRATION"),
    ("different-model", "mismatch", "INCOMPLETE"),
])
def test_catalog_identity_reaches_packet_and_final_outcome(tmp_path, catalog_id, verification, result):
    attempt = _proved_attempt()
    attempt["observed_model_catalog_match"]["id"] = catalog_id
    usage = _usage(tmp_path, attempt)
    usage["capability_delta"] = [{"reason": "session_route_resolves_its_own_model"}]
    assert usage["resolved_model"] == LABEL  # Raw native observation is never rewritten.
    assert usage["observed_attempt"]["observed_model_catalog_match"] == attempt["observed_model_catalog_match"]
    actors, config = _persist(tmp_path, usage)
    _code, _final, packet = _packet(tmp_path, actors, config)

    assert packet["result"] == result
    receipt = packet["review_execution"]["receipts"][0]
    assert receipt["observed"]["model"] == LABEL
    assert receipt["observed"]["observed_model_catalog_match"] == attempt["observed_model_catalog_match"]
    assert receipt["model_verification"] == verification
    assert packet["review_execution"]["telemetry_limitations"] == []
    if catalog_id != MODEL:
        assert packet["review_execution"]["mismatches"] == [f"model_mismatch:pool:s1:{MODEL}->{catalog_id}"]


@pytest.mark.parametrize("edit", [
    {"source": "requested"}, {"source": "summary"}, {"source": True},
    {"label": "Other label"}, {"label": LABEL + " "}, {"label": 123},
    {"id": ""}, {"id": "  "}, {"id": 123},
    {"profile_id": "prior-profile"}, {"profile_id": ""}, {"profile_id": False},
])
def test_invalid_or_unbound_catalog_match_remains_unverified(tmp_path, edit):
    attempt = _proved_attempt()
    attempt["observed_model_catalog_match"].update(edit)
    actors, config = _persist(tmp_path, _usage(tmp_path, attempt))
    code, _final, packet = _packet(tmp_path, actors, config)

    assert code == 3 and packet["result"] == "INCOMPLETE"
    assert packet["review_execution"]["receipts"][0]["model_verification"] == "observed_display_label"
    assert any(item.startswith("model_identity_unverified:") for item in packet["review_execution"]["mismatches"])


@pytest.mark.parametrize("match", [None, [], "account_catalog", {}, {"id": MODEL}])
def test_missing_or_malformed_catalog_match_keeps_legacy_display_label(tmp_path, match):
    attempt = {**_attempt(), "observed_model_catalog_match": match}
    actors, config = _persist(tmp_path, _usage(tmp_path, attempt))
    code, _final, packet = _packet(tmp_path, actors, config)
    assert code == 3
    assert packet["review_execution"]["receipts"][0]["model_verification"] == "observed_display_label"


@pytest.mark.parametrize("binding", ["wrong_run", "duplicate_final", "earlier_attempt", "summary", "non_cursor"])
def test_catalog_proof_never_escapes_the_selected_run_and_attempt(tmp_path, binding):
    attempt = _proved_attempt()
    extra = []
    if binding == "duplicate_final":
        extra = [copy.deepcopy(attempt)]
    elif binding == "earlier_attempt":
        extra = [{**copy.deepcopy(attempt), "attempt_id": "a01"}]
        attempt.pop("observed_model_catalog_match")
    elif binding == "non_cursor":
        attempt["harness_id"] = "codex"
    detail = _telemetry(tmp_path, attempt, run_id="other-run" if binding == "wrong_run" else RUN, extra=extra)
    if binding == "summary":
        # The accepted schema lives only on the final attempt, never summary.route.
        proof = attempt.pop("observed_model_catalog_match")
        detail = _telemetry(tmp_path, attempt)
        detail["summary"]["route"]["observed_model_catalog_match"] = proof
    observed = final_attempt_facts(detail, RUN)
    assert not observed.get("observed_model_catalog_match")
    usage = {**session_usage_facts(RUN, observed_attempt=observed),
             "applied_profile": PROFILE, "applied_access": "readonly", "custody_durable": True,
             "verdict_provenance": {"raw_transcript_chars": len(TRANSCRIPT),
                 "raw_transcript_sha256": hashlib.sha256(TRANSCRIPT.encode()).hexdigest()}}
    actors, config = _persist(tmp_path, usage, harness="codex" if binding == "non_cursor" else "cursor")
    code, _final, packet = _packet(tmp_path, actors, config)
    assert code == 3 and packet["result"] == "INCOMPLETE"


def test_job_handle_keeps_the_engine_proven_run_binding(tmp_path):
    detail = _telemetry(tmp_path, _proved_attempt())
    detail["summary"].update(runId=RUN, jobId="job-handle")
    observed = final_attempt_facts(detail, "job-handle")
    usage = {**session_usage_facts("job-handle", observed_attempt=observed),
             "applied_profile": PROFILE, "applied_access": "readonly", "custody_durable": True,
             "verdict_provenance": {"raw_transcript_chars": len(TRANSCRIPT),
                 "raw_transcript_sha256": hashlib.sha256(TRANSCRIPT.encode()).hexdigest()}}
    actors, config = _persist(tmp_path, usage)
    code, _final, packet = _packet(tmp_path, actors, config)
    assert code == 0 and packet["result"] == "READY_FOR_INTEGRATION"


@pytest.mark.parametrize("fault", ["resolved_model", "applied_profile", "delegated_route"])
def test_top_level_receipt_drift_cannot_borrow_nested_proof(tmp_path, fault):
    usage = _usage(tmp_path, _proved_attempt())
    usage[fault] = "Other Display Label" if fault == "resolved_model" else "other"
    actors, config = _persist(tmp_path, usage)
    code, _final, packet = _packet(tmp_path, actors, config)
    receipt = packet["review_execution"]["receipts"][0]
    assert code == 3 and packet["result"] == "INCOMPLETE"
    assert receipt["model_verification"] == "observed_display_label"
    assert "observed_model_catalog_match" not in receipt["observed"]


@pytest.mark.parametrize("fault", ["model", "profile_id", "harness_id"])
def test_persisted_match_must_agree_with_the_receipts_observed_identity(tmp_path, fault):
    usage = _usage(tmp_path, _proved_attempt())
    usage["observed_attempt"][fault] = "other"
    actors, config = _persist(tmp_path, usage)
    code, _final, packet = _packet(tmp_path, actors, config)
    assert code == 3 and packet["result"] == "INCOMPLETE"
    assert packet["review_execution"]["receipts"][0]["model_verification"] == "observed_display_label"
