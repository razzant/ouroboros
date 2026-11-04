"""Plan-review continuity seams that outlive the retired engine: exact evidence selectors,
Claudexor thread receipts and turns, profile pins and rotation receipts (telemetry, never a
gate: owner decision 1=B), the agent-session executor's paid transcript and model attestation,
the parser of the planning organ's reply, and the earlier-schema wave compaction's counts."""
from __future__ import annotations

import json

import httpx
import pytest


def _finding(index: int, klass: str = "note") -> dict:
    return {
        "id": f"f{index}",
        "class": klass,
        "breaks": "goal" if klass == "blocking" else "",
        "locator": "",
        "summary": f"finding {index}",
        "recommendation": "repair it",
    }


def test_exact_evidence_selectors_return_the_requested_slice(tmp_path) -> None:
    from ouroboros.tools.plan_evidence import resolve_evidence

    source = tmp_path / "evidence.txt"
    source.write_bytes(b"one\ntwo\nthree\nfour\nfive\n")
    manifest = resolve_evidence(
        ["evidence.txt::lines=3-4", "evidence.txt::tail=5"],
        active_root=tmp_path,
        allowed_roots=[tmp_path],
    )

    assert manifest["omissions"] == []
    assert manifest["attached"][0]["text"] == "three\nfour\n"
    assert manifest["attached"][0]["selector"] == {
        "kind": "line_range", "start": 3, "end": 4,
    }
    assert manifest["attached"][1]["text"] == "five\n"
    assert manifest["attached"][1]["selector"] == {"kind": "tail", "bytes": 5}


def test_symbol_selector_uses_the_qualified_definition(tmp_path) -> None:
    from ouroboros.tools.plan_evidence import resolve_evidence

    source = tmp_path / "subject.py"
    source.write_text(
        "class First:\n    def decide(self):\n        return 'wrong'\n\n"
        "class Second:\n    def decide(self):\n        return 'exact'\n",
        encoding="utf-8",
    )
    manifest = resolve_evidence(
        ["subject.py::symbol=Second.decide"], active_root=tmp_path,
        allowed_roots=[tmp_path],
    )

    assert manifest["omissions"] == []
    assert "return 'exact'" in manifest["attached"][0]["text"]
    assert "return 'wrong'" not in manifest["attached"][0]["text"]


def test_compacted_wave_keeps_the_full_blocking_count() -> None:
    from ouroboros.task_results import _compact_plan_review_wave

    wave = {
        "cycle_index": 1,
        "request_fingerprint": "a" * 64,
        "aggregate": "REVISE_PLAN",
        "findings": [_finding(index) for index in range(1, 33)],
        "findings_total": 33,
        "counts": {"blocking": 1, "note": 32, "need_evidence": 0},
        "closed": False,
        "paid": True,
    }

    assert _compact_plan_review_wave(wave)["counts"]["blocking"] == 1


def test_api_chat_continuation_uses_exact_slot_transcript() -> None:
    from ouroboros.review_execution import ApiChatReviewExecutor, ReviewAssignment
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot

    prior = [
        {"role": "system", "content": "system-v1"},
        {"role": "user", "content": "plan-v1"},
        {"role": "assistant", "content": "need exact tail"},
        {"role": "user", "content": "plan-v1 + exact tail"},
    ]
    request = ReviewRequest(
        surface="plan_review",
        goal="review",
        task_id="task-1",
        messages=[{"role": "user", "content": "wrong common transcript"}],
        slot_messages={"slot_a": prior},
    )
    slot = ReviewSlot(slot_id="slot_a", model="m/a")
    executor = ApiChatReviewExecutor(ReviewAssignment(request=request, slot=slot))

    assert executor.messages == prior
    assert executor._kwargs()["model"] == "m/a"


def test_review_thread_receipt_requires_run_and_turn_to_match() -> None:
    from ouroboros.review_thread_continuity import review_thread_receipt
    from ouroboros.review_execution import ReviewRouteUnavailable

    class Gateway:
        def get_thread(self, _thread_id):
            return {
                "thread": {"headRunId": "other-run"},
                "turns": [{"id": "turn-1", "runId": "other-run"}],
                "sessions": [],
            }

        def get_run_artifact(self, _run_id, _path):
            return b""

    with pytest.raises(ReviewRouteUnavailable) as excinfo:
        review_thread_receipt(Gateway(), "thread-1", "target-run", "turn-1")

    assert excinfo.value.code == "review_thread_receipt_missing"


def test_claudexor_gateway_thread_turn_contract(monkeypatch) -> None:
    from ouroboros.gateways.claudexor import ClaudexorGateway, DaemonEndpoint

    seen: list[tuple[str, str, dict, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        seen.append((request.method, request.url.path, body, request.headers.get("Idempotency-Key", "")))
        if request.url.path == "/v2/threads":
            return httpx.Response(200, json={"id": "thread-1"})
        if request.url.path == "/v2/threads/thread-1/turns":
            return httpx.Response(200, json={
                "jobId": "job-2", "runId": "run-2", "runDir": "/tmp/run-2",
                "threadId": "thread-1", "turnId": "turn-2",
            })
        if request.url.path == "/v2/threads/thread-1":
            return httpx.Response(200, json={
                "thread": {"id": "thread-1", "headRunId": "run-2"},
                "sessions": [{
                    "id": "session-1", "threadId": "thread-1", "harnessId": "claude",
                    "profileId": "profile-b", "state": "live",
                }],
                "turns": [{
                    "id": "turn-2", "threadId": "thread-1", "runId": "run-2",
                    "continuity": {
                        "kind": "packet", "packetTurns": 1, "summarized": False,
                        "laneSwitchedFrom": {"harness": "claude", "profileId": "profile-a"},
                    },
                }],
            })
        return httpx.Response(404, json={"code": "not_found", "message": "no"})

    gateway = ClaudexorGateway(DaemonEndpoint(host="127.0.0.1", port=1, token="token"))
    gateway._client.close()
    gateway._client = httpx.Client(
        base_url="http://127.0.0.1:1", transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer token"},
    )
    try:
        thread = gateway.create_thread({
            "scope": {"kind": "project", "root": "/repo"},
            "mode": "ask", "authPreference": "subscription",
            "primaryHarness": "claude", "eligibleHarnesses": ["claude"],
            "credentialProfileId": "profile-a", "access": "readonly",
        }, idempotency_key="thread-key")
        turn = gateway.start_thread_turn(
            thread["id"], {"prompt": "exact evidence", "mode": "ask"},
            idempotency_key="turn-key",
        )
        detail = gateway.get_thread(thread["id"])
    finally:
        gateway.close()

    assert turn["threadId"] == "thread-1" and turn["runId"] == "run-2"
    assert detail["turns"][0]["continuity"]["kind"] == "packet"
    assert detail["sessions"][0]["profileId"] == "profile-b"
    assert seen[0][3] == "thread-key" and seen[1][3] == "turn-key"


def test_continued_thread_explicitly_repins_the_expected_profile() -> None:
    from ouroboros.review_thread_continuity import start_review_thread_turn

    captured = {}

    class Gateway:
        def start_thread_turn(self, thread_id, request, *, idempotency_key):
            captured.update({
                "thread_id": thread_id, "request": request,
                "idempotency_key": idempotency_key,
            })
            return {"runId": "run-2", "threadId": thread_id, "turnId": "turn-2"}

    start_review_thread_turn(Gateway(), "thread-1", {
        "prompt": "continue", "model": "fable", "harnesses": ["claude"],
        "credentialProfileId": "profile-a", "_thread_id": "thread-1",
        "scope": {"kind": "project", "root": "/repo"},
    }, idempotency_key="turn-key")

    assert captured["request"]["model"] == "fable"
    assert captured["request"]["harnesses"] == ["claude"]
    assert captured["request"]["credentialProfileId"] == "profile-a"


def test_initial_unpinned_thread_omits_profile_but_its_turn_sends_explicit_null() -> None:
    from types import SimpleNamespace

    from ouroboros.gateways.claudexor import ClaudexorGateway, DaemonEndpoint
    from ouroboros.review_thread_continuity import ensure_review_thread, start_review_thread_turn

    captured = {}

    class Custody:
        @staticmethod
        def idempotency_key(*parts):
            return ":".join(str(part) for part in parts)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        if request.url.path == "/v2/threads":
            # Installed ControlThreadCreateRequest: nonblank string or omission.
            assert "credentialProfileId" not in body
            captured["create"] = body
            return httpx.Response(200, json={"id": "thread-1"})
        if request.url.path == "/v2/threads/thread-1/turns":
            # Installed ControlThreadTurnRequest: explicit null clears a sticky pin.
            assert "credentialProfileId" in body and body["credentialProfileId"] is None
            captured["turn"] = body
            return httpx.Response(200, json={
                "threadId": "thread-1", "turnId": "turn-1", "runId": "run-1",
            })
        return httpx.Response(404, json={"code": "not_found", "message": "no"})

    gateway = ClaudexorGateway(DaemonEndpoint(host="127.0.0.1", port=1, token="token"))
    gateway._client.close()
    gateway._client = httpx.Client(
        base_url="http://127.0.0.1:1", transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer token"},
    )
    try:
        thread_id = ensure_review_thread(
            gateway, Custody(), "", route=SimpleNamespace(route_id="claude", profile_id=""),
            root="/repo", surface="plan_review", slot_id="s1", task_id="task-1",
        )
        start_review_thread_turn(
            gateway, thread_id,
            {"prompt": "review", "credentialProfileId": None, "_thread_id": thread_id},
            idempotency_key="turn-key",
        )
    finally:
        gateway.close()

    assert thread_id == "thread-1"
    assert captured["turn"]["credentialProfileId"] is None


def test_profile_rotation_receipt_is_read_from_the_settled_run_events() -> None:
    from ouroboros.review_thread_continuity import profile_rotation_receipts

    class Gateway:
        def get_run_artifact(self, run_id, path):
            assert (run_id, path) == ("run-2", "events.jsonl")
            return (json.dumps({
                "seq": 7,
                "type": "route.profile.rotated",
                "payload": {
                    "from_profile_id": "profile-a", "to_profile_id": "profile-b",
                    "reason": "profile_headroom_preflight", "resets_at": "later",
                },
            }) + "\n").encode()

    assert profile_rotation_receipts(Gateway(), "run-2") == [{
        "seq": 7,
        "type": "route.profile.rotated",
        "from_profile_id": "profile-a",
        "to_profile_id": "profile-b",
        "reason": "profile_headroom_preflight",
        "attempt_id": "",
        "resets_at": "later",
    }]


def _rotation(seq, source, target, *, reason="vendor_limit_rejected"):
    return {
        "seq": seq, "type": "route.profile.rotated",
        "from_profile_id": source, "to_profile_id": target,
        "reason": reason, "attempt_id": f"attempt-{seq}", "resets_at": "later",
    }


@pytest.mark.parametrize(
    ("expected", "applied", "rotations", "expected_status", "expected_reason"),
    [
        ("profile-a", "profile-b", [_rotation(10, "profile-a", "profile-b")],
         "typed_rotation", "typed_rotation_chain"),
        ("profile-a", "profile-c",
         [_rotation(10, "profile-a", "profile-b"), _rotation(20, "profile-b", "profile-c")],
         "typed_rotation", "typed_rotation_chain"),
        # FIX 2: a typed engine rotation on an UNPINNED slot (no recorded
        # expectation) walks from the chain's own first hop — never chain-gap noise.
        ("", "profile-b", [_rotation(10, "profile-a", "profile-b")],
         "typed_rotation", "typed_rotation_chain"),
        ("profile-a", "profile-c",
         [_rotation(10, "profile-a", "profile-b"), _rotation(20, "profile-x", "profile-c")],
         "cannot_verify", "rotation_chain_gap"),
        ("profile-a", "profile-c", [_rotation(10, "profile-a", "profile-b")],
         "cannot_verify", "rotation_terminal_mismatch"),
        ("profile-a", "profile-c",
         [_rotation(20, "profile-a", "profile-b"), _rotation(10, "profile-b", "profile-c")],
         "cannot_verify", "rotation_event_order_invalid"),
        ("profile-a", "profile-b", [{**_rotation(10, "profile-a", "profile-b"), "reason": ""}],
         "cannot_verify", "rotation_event_malformed"),
    ],
    ids=("one-hop", "multi-hop", "unpinned-chain", "broken-gap", "terminal-mismatch",
         "reordered", "malformed"),
)
def test_profile_continuity_folds_only_one_ordered_engine_rotation_chain(
    expected, applied, rotations, expected_status, expected_reason,
) -> None:
    from ouroboros.review_thread_continuity import profile_continuity_receipt

    receipt = profile_continuity_receipt(expected, applied, rotations)

    assert receipt["status"] == expected_status
    assert receipt["verification_reason"] == expected_reason
    if expected_status == "typed_rotation":
        assert receipt["rotation_receipts"] == rotations
        assert receipt["rotation_receipt"] == rotations[-1]
    else:
        assert receipt["rotation_receipt"] == {}


def _session_run(**overrides):
    """One settled delegated-session payload; each test overrides its deltas."""
    return {
        "run_id": "run-2", "thread_id": "thread-1", "turn_id": "turn-2",
        "thread_receipt": {"continuity": {"kind": "native_resume"}},
        "text": "[]\nNO_FINDINGS", "conformance": "passed", "schema_asked": True,
        "custody_durable": True, "settlement": "settled", "route_id": "claude",
        "effective_route_ids": ["claude"], "model": "fable", "spend": 0.0,
        "spend_estimated": False, "applied_profile": "profile-a", "applied_access": "readonly",
        "auth_route_receipt": {
            "requested": "subscription", "effective": "subscription",
            "reason": "subscription_preferred", "profileId": "profile-a",
        },
        **overrides,
    }


def test_agent_session_continuation_passes_the_real_thread_id(monkeypatch, tmp_path) -> None:
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot

    captured = {}

    def fake_run(*, prompt, root, custody_drive, invocation):
        captured.update({"prompt": prompt, "root": root, "invocation": invocation})
        return _session_run(auth_route_receipt={
            "requested": "subscription", "effective": "subscription",
            "reason": "quota_exhausted", "profileId": "profile-a",
        })

    monkeypatch.setattr("ouroboros.review_execution.run_delegated_review_session", fake_run)
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="task-1",
        session_root=str(tmp_path), session_task="review exact evidence",
        session_threads={"slot_a": "thread-1"},
        policy={"output_contract": "return findings"},
    )
    slot = ReviewSlot(
        slot_id="slot_a", model="fable", route="agent_session",
        session_target="claude=fable", session_profile="profile-a",
    )
    result = AgentSessionReviewExecutor(
        ReviewAssignment(request=request, slot=slot, custody_root=tmp_path)
    ).execute()

    assert captured["invocation"].thread_id == "thread-1"
    assert captured["invocation"].use_thread is True
    assert result.usage["review_thread_id"] == "thread-1"
    assert result.usage["review_thread_receipt"]["continuity"]["kind"] == "native_resume"
    assert result.usage["auth_route_receipt"]["reason"] == "quota_exhausted"


DRIFT_FINDINGS = (
    '[{"id": "f4", "class": "blocking", "breaks": "goal", "locator": "",'
    ' "summary": "swarm delegation dropped", "recommendation": "restore it"}]'
)


def test_agent_session_preserves_paid_findings_under_profile_drift(monkeypatch, tmp_path) -> None:
    """Profile continuity is PURE TELEMETRY: a real pinned mismatch keeps the paid
    transcript whole and parseable — the receipt rides the usage record and the
    plan row built from it, never a blanked artifact or a fabricated parse cause."""
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot
    from ouroboros.tools.plan_review_runtime import _plan_row_from_actor

    receipt = {
        "expected_profile": "profile-a", "applied_profile": "profile-b",
        "status": "cannot_verify", "verification_reason": "unexplained_profile_drift",
        "rotation_receipt": {},
    }

    def fake_run(*, prompt, root, custody_drive, invocation):
        return _session_run(
            profile_continuity_receipt=dict(receipt), text=DRIFT_FINDINGS,
            applied_profile="profile-b",
        )

    monkeypatch.setattr("ouroboros.review_execution.run_delegated_review_session", fake_run)
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="task-1",
        session_root=str(tmp_path), session_task="review exact evidence",
        session_threads={"slot_a": "thread-1"},
        policy={"output_contract": "return findings"},
    )
    slot = ReviewSlot(
        slot_id="slot_a", model="fable", route="agent_session",
        session_target="claude=fable", session_profile="profile-a",
    )

    result = AgentSessionReviewExecutor(
        ReviewAssignment(request=request, slot=slot, custody_root=tmp_path)
    ).execute()

    assert result.message["session_transcript"] == DRIFT_FINDINGS
    assert json.loads(result.raw_text)[0]["summary"] == "swarm delegation dropped"
    assert result.usage["profile_continuity_receipt"] == receipt
    row = _plan_row_from_actor({"slot_id": "slot_a", "status": "ok", "raw_text": result.raw_text,
                                "usage": result.usage, "prompt_ref": {}, "response_ref": {}}, slot)
    assert row["profile_continuity_receipt"] == receipt and row["text"] == DRIFT_FINDINGS and not row["error"]


def test_empty_profile_expectation_is_matched_telemetry_without_disclosure() -> None:
    """An expectation that was never recorded is not drift: an unpinned seat stays
    `matched`/`no_expectation_recorded` — a durable receipt, never a mismatch."""
    from ouroboros.review_thread_continuity import profile_continuity_receipt

    for applied in ("profile-b", ""):  # FIX 4: ""/"" is no-expectation, not a false pin match
        receipt = profile_continuity_receipt("", applied, [])
        assert receipt["status"] == "matched"
        assert receipt["verification_reason"] == "no_expectation_recorded"
    matched = profile_continuity_receipt("profile-a", "profile-a", [])
    assert (matched["status"], matched["verification_reason"]) == ("matched", "profile_matched")


def test_agent_session_executor_is_structurally_unable_to_blank_the_transcript(tmp_path) -> None:
    """The blanking field is gone: one transcript attribute feeds parser, provenance
    and the durable message alike — an emptied copy can no longer exist."""
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot

    executor = AgentSessionReviewExecutor(ReviewAssignment(
        request=ReviewRequest(surface="plan_review", goal="review", task_id="task-1"),
        slot=ReviewSlot(slot_id="s1", model="fable", route="agent_session"),
        custody_root=tmp_path,
    ))

    assert "_transcript" not in vars(executor)
    assert executor._raw_transcript is None


def test_session_model_mismatch_is_disclosed_on_the_actor_row(monkeypatch, tmp_path) -> None:
    """The protection the owner relies on: an effective session model that differs
    from the slot's requested model records `session_route_resolves_its_own_model`
    and the delta travels through actor usage into the wave actor record."""
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot
    from ouroboros.tools.plan_review_runtime import _plan_row_from_actor, plan_row_typed_facts

    def fake_run(*, prompt, root, custody_drive, invocation):
        return _session_run(model="grok-4.6")

    monkeypatch.setattr("ouroboros.review_execution.run_delegated_review_session", fake_run)
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="task-1",
        session_root=str(tmp_path), session_task="review exact evidence",
        session_threads={"slot_a": "thread-1"},
        policy={"output_contract": "return findings"},
    )
    slot = ReviewSlot(
        slot_id="slot_a", model="fable", route="agent_session",
        session_target="claude=fable", session_profile="profile-a",
    )

    result = AgentSessionReviewExecutor(
        ReviewAssignment(request=request, slot=slot, custody_root=tmp_path)
    ).execute()

    mismatch = [d for d in result.usage["capability_delta"]
                if d.get("reason") == "session_route_resolves_its_own_model"]
    assert mismatch == [{
        "kind": "capability_delta",
        "requested": "model fable",
        "effective": "model grok-4.6",
        "reason": "session_route_resolves_its_own_model",
        "note": "The engine names the model it ran by its own alias; the names differ, capabilities were "
                "not compared. A disclosure, not a failure.",
    }]
    assert result.raw_text == "[]\nNO_FINDINGS"  # disclosed, never discarded
    row = _plan_row_from_actor({
        "slot_id": "slot_a", "status": "ok", "raw_text": result.raw_text,
        "usage": result.usage, "prompt_ref": {}, "response_ref": {},
    }, slot)
    assert mismatch[0] in row["capability_delta"]
    assert mismatch[0] in plan_row_typed_facts(row)["capability_delta"]


def test_agent_session_accepts_only_a_typed_profile_rotation_receipt(monkeypatch, tmp_path) -> None:
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot

    rotation = {
        "type": "route.profile.rotated", "from_profile_id": "profile-a",
        "to_profile_id": "profile-b", "reason": "vendor_limit_rejected",
        "attempt_id": "a01", "resets_at": "2026-08-22T00:00:00Z",
    }

    def fake_run(*, prompt, root, custody_drive, invocation):
        return _session_run(
            thread_receipt={"continuity": {"kind": "packet", "laneSwitchedFrom": {
                "harness": "claude", "profileId": "profile-a"}}},
            profile_continuity_receipt={
                "expected_profile": "profile-a", "applied_profile": "profile-b",
                "status": "typed_rotation", "rotation_receipt": rotation,
            },
            applied_profile="profile-b",
        )

    monkeypatch.setattr("ouroboros.review_execution.run_delegated_review_session", fake_run)
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="task-1",
        session_root=str(tmp_path), session_task="review exact evidence",
        session_threads={"slot_a": "thread-1"},
        policy={"output_contract": "return findings"},
    )
    slot = ReviewSlot(
        slot_id="slot_a", model="fable", route="agent_session",
        session_target="claude=fable", session_profile="profile-a",
    )

    result = AgentSessionReviewExecutor(
        ReviewAssignment(request=request, slot=slot, custody_root=tmp_path)
    ).execute()

    assert result.raw_text == "[]\nNO_FINDINGS"
    assert result.usage["profile_continuity_receipt"] == {
        "expected_profile": "profile-a",
        "applied_profile": "profile-b",
        "status": "typed_rotation",
        "rotation_receipt": rotation,
    }


def test_a_reply_with_null_optional_lists_reads_as_absent() -> None:
    """§1.4 (incident f0a6fb64c92d491f): a model that emits null for an absent optional
    field must still be read. The planning organ's reply parser takes a null prose,
    objections or withdrawn as absent; a wrongly typed field still refuses the whole reply."""
    from ouroboros.plan_record import parse_reply

    reply, error = parse_reply('{"prose": null, "objections": null, "withdrawn": null}')
    assert error == "" and reply == {"prose": "", "objections": [], "withdrawn": []}
    reply, error = parse_reply('{"prose": "fine", "objections": [{"id": "o1", "text": "x"}], "withdrawn": null}')
    assert error == "" and reply["objections"] == [{"id": "o1", "text": "x"}] and reply["withdrawn"] == []
    reply, error = parse_reply('{"prose": "fine", "objections": "none"}')
    assert reply is None and error
