from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from tests import test_plan_review_engine as plan_review_engine


plan_review_harness_fixture = pytest.fixture(name="_harness")(
    plan_review_engine.harness.__wrapped__
)


def _finding(index: int, klass: str = "note") -> dict:
    return {
        "id": f"f{index}",
        "class": klass,
        "breaks": "goal" if klass == "blocking" else "",
        "locator": "",
        "summary": f"finding {index}",
        "recommendation": "repair it",
    }


def test_33rd_blocking_finding_is_aggregated() -> None:
    from ouroboros.tools.plan_spec import aggregate, validate_findings

    raw = [_finding(i) for i in range(1, 33)] + [_finding(33, "blocking")]
    findings, disclosures, _seen = validate_findings(
        raw, spec_ids={"goal"}, seen_locators=(), slot="slot_a",
    )
    result = aggregate([
        {"slot": "slot_a", "model": "m/a", "ok": True, "findings": findings},
    ], quorum=1)

    assert disclosures == []
    assert len(findings) == 33
    assert result["aggregate"] == "REVISE_PLAN"
    assert result["counts"]["blocking"] == 1
    assert result["findings"][-1]["finding_id"] == "slot_a:f33"


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


def test_requested_tail_preempts_a_full_120k_declared_pack(_harness) -> None:
    from tests.test_plan_review_engine import CLEAN, DECK_SPEC, _call, _finding, _user_text

    for index, char in enumerate("abc", start=1):
        (_harness.workspace / f"bulk-{index}.txt").write_text(char * 40_000, encoding="utf-8")
    decisive = _harness.workspace / "decisive.txt"
    decisive.write_text("x" * 50_000 + "DECISIVE_TAIL\n", encoding="utf-8")
    spec = {**DECK_SPEC, "evidence": [f"bulk-{index}.txt" for index in range(1, 4)]}
    ask = json.dumps([
        _finding(
            "tail", "need_evidence", breaks="goal", locator="decisive.txt::tail=64",
            summary="need the decisive tail",
        )
    ])
    _harness.install({"s1": ask, "s2": CLEAN, "s3": CLEAN})
    _call(_harness.make_ctx(), spec=spec)

    substrate = _harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    _call(_harness.make_ctx(), spec=spec)

    current_user = _user_text(substrate.calls[0]["request"].slot_messages["s1"][-1]["content"])
    assert "DECISIVE_TAIL" in current_user


def test_missing_requested_evidence_dispatches_with_the_absence_named(_harness) -> None:
    from tests.test_plan_review_engine import CLEAN, _call, _control, _finding, _user_text

    ask = json.dumps([
        _finding(
            "f1", "need_evidence", breaks="goal", locator="missing.md::lines=1-2",
            summary="read the exact lines",
        )
    ])
    _harness.install({"s1": ask, "s2": CLEAN, "s3": CLEAN})
    assert _control(_call(_harness.make_ctx())) == {
        "outcome": "REVIEW_REQUIRED", "closed": False,
    }

    substrate = _harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    out = _call(_harness.make_ctx())

    assert "cannot_verify" not in out
    assert len(substrate.calls) == 1  # the panel is dispatched, never refused for free
    sent = _user_text(substrate.calls[0]["request"].slot_messages["s1"][-1]["content"])
    assert "[reviewer-requested]" in sent and "missing.md::lines=1-2" in sent
    assert _control(out) == {"outcome": "GREEN", "closed": True}  # the panel's own call


def test_wave_9_and_65_remain_exactly_readable_after_hot_trimming(tmp_path) -> None:
    from ouroboros.task_results import load_plan_review_state, record_plan_review_wave
    from ouroboros.tools.plan_review_runtime import (
        persist_plan_review_wave_artifact,
        read_plan_review_wave_artifact,
    )

    task_id = "task-review"
    refs = {}
    for index in range(1, 66):
        fingerprint = f"{index:064x}"
        exact = {
            "schema_version": 1, "cycle_index": index, "request_fingerprint": fingerprint,
            "findings": [{"id": f"tail-{index}", "summary": "x" * 5000}],
            "reviewer_outputs": [{"slot_id": "s1", "text": f"exact-wave-{index}"}],
        }
        ref = persist_plan_review_wave_artifact(tmp_path, task_id, exact)
        refs[index] = ref
        record_plan_review_wave(tmp_path, task_id, {
            "schema_version": 2,
            "cycle_index": index,
            "request_fingerprint": fingerprint,
            "spec": {"goal": "g"},
            "findings": exact["findings"],
            "aggregate": "GREEN",
            "closed": True,
            "paid": True,
            "dispositions": [],
            "wave_artifact": ref,
        })

    state = load_plan_review_state(tmp_path, task_id)
    assert len(state["waves"]) == 64
    assert sum(1 for wave in state["waves"] if not wave.get("compact")) == 8
    wave9 = next(w for w in state["waves"] if w["cycle_index"] == 9)
    assert wave9["compact"] is True
    assert wave9["wave_artifact"] == refs[9]
    assert read_plan_review_wave_artifact(tmp_path, task_id, refs[9])["reviewer_outputs"][0]["text"] == "exact-wave-9"
    assert read_plan_review_wave_artifact(tmp_path, task_id, refs[65])["reviewer_outputs"][0]["text"] == "exact-wave-65"


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


@pytest.mark.parametrize("current_ids", [("s1", "s2"), ("s1", "s2", "s3", "s4")])
def test_continuation_roster_change_degrades_to_fresh_dispatch(tmp_path, current_ids) -> None:
    """A changed reviewer roster is a cache miss, not a validity event: the wave
    re-dispatches fresh with the self-contained packet and the typed cause."""
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.tools.plan_review_artifacts import continuation_inputs, persist_wave

    prior_slots = [ReviewSlot(slot_id=sid, model=f"model-{sid}") for sid in ("s1", "s2", "s3")]
    exact = {
        "schema_version": 1, "cycle_index": 1, "request_fingerprint": "b" * 64,
        "slots": [
            {
                "slot_id": slot.slot_id, "model": slot.model, "effort": slot.effort,
                "route": "api_chat", "session_target": "", "session_profile": "",
            }
            for slot in prior_slots
        ],
        "reviewer_outputs": [
            {
                "slot_id": slot.slot_id,
                "request_messages": [{"role": "user", "content": "prior"}],
                "text": "[]",
            }
            for slot in prior_slots
        ],
    }
    ref = persist_wave(tmp_path, "task-1", exact)
    current = [ReviewSlot(slot_id=sid, model=f"model-{sid}") for sid in current_ids]

    slots_out, messages, threads, restarted = continuation_inputs(
        tmp_path, "task-1", {"wave_artifact": ref}, current, user_content="continue",
    )

    assert restarted == "prior_reviewer_assignment_set_changed"
    assert messages == {} and threads == {}
    assert slots_out == current  # exactly as configured, never rebound to prior rows


def test_continuation_uses_the_currently_configured_slot_pin(tmp_path) -> None:
    """The prior wave's applied account is telemetry, not a sticky repin: the
    currently configured slot pin (or empty = rotation) always applies."""
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.tools.plan_review_artifacts import continuation_inputs, persist_wave, slot_row

    prior = ReviewSlot(
        slot_id="s1", model="claude=fable", route="agent_session",
        session_target="claude=fable", session_profile="profile-a",
    )
    exact = {
        "schema_version": 1, "cycle_index": 1, "request_fingerprint": "c" * 64,
        "slots": [slot_row(prior)],
        "reviewer_outputs": [{
            "slot_id": "s1", "review_thread_id": "thread-1",
            "applied_profile": "profile-b", "text": "need evidence",
        }],
    }
    ref = persist_wave(tmp_path, "task-1", exact)

    rebound, _messages, threads, restarted = continuation_inputs(
        tmp_path, "task-1", {"wave_artifact": ref}, [prior], user_content="continue",
    )

    assert restarted == ""
    assert threads == {"s1": "thread-1"}
    assert rebound[0].session_profile == "profile-a"


def test_agent_session_continuation_restarts_fresh_when_prior_thread_is_missing(tmp_path) -> None:
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.tools.plan_review_artifacts import continuation_inputs, persist_wave, slot_row

    slot = ReviewSlot(
        slot_id="s1", model="claude=fable", route="agent_session",
        session_target="claude=fable", session_profile="",
    )
    exact = {
        "schema_version": 1, "cycle_index": 1, "request_fingerprint": "e" * 64,
        "slots": [slot_row(slot)],
        "reviewer_outputs": [{"slot_id": "s1", "text": "need evidence"}],
    }
    ref = persist_wave(tmp_path, "task-1", exact)

    slots_out, messages, threads, restarted = continuation_inputs(
        tmp_path, "task-1", {"wave_artifact": ref}, [slot], user_content="continue",
    )

    assert restarted == "prior_review_thread_missing:s1"
    assert messages == {} and threads == {}
    assert slots_out == [slot]


def test_exact_wave_custody_gaps_degrade_to_a_named_fresh_dispatch(tmp_path) -> None:
    """An absent or unreadable prior exact wave is a cache miss here: custody is
    enforced one level up, so each gap names a typed fresh-restart cause."""
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.tools.plan_review_artifacts import continuation_inputs

    slot = ReviewSlot(slot_id="s1", model="model-s1")

    for previous, expected in (
        (None, "prior_exact_wave_missing"),
        ({"paid": True}, "prior_exact_wave_ref_missing"),
    ):
        _slots, messages, threads, restarted = continuation_inputs(
            tmp_path, "task-1", previous, [slot], user_content="continue",
        )
        assert restarted == expected
        assert messages == {} and threads == {}

    ref = {"root": "artifact_store", "path": "missing-wave.json", "bytes": 3, "sha256": "0" * 64}
    _slots, _messages, _threads, restarted = continuation_inputs(
        tmp_path, "task-1", {"wave_artifact": ref}, [slot], user_content="continue",
    )
    assert restarted.startswith("prior_exact_wave_unreadable:")


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


@pytest.mark.parametrize(
    "prior_messages",
    [[], [{}], [{"role": "user"}], [{"role": "", "content": "prior"}]],
)
def test_api_chat_continuation_restarts_fresh_on_invalid_transcript(
    tmp_path, prior_messages,
) -> None:
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.tools.plan_review_artifacts import continuation_inputs, persist_wave, slot_row

    slot = ReviewSlot(slot_id="s1", model="model-s1")
    exact = {
        "schema_version": 1, "cycle_index": 1, "request_fingerprint": "d" * 64,
        "slots": [slot_row(slot)],
        "reviewer_outputs": [{
            "slot_id": "s1", "request_messages": prior_messages, "text": "need evidence",
        }],
    }
    ref = persist_wave(tmp_path, "task-1", exact)

    _slots, messages, threads, restarted = continuation_inputs(
        tmp_path, "task-1", {"wave_artifact": ref}, [slot], user_content="continue",
    )

    assert restarted == "prior_api_transcript_invalid:s1"
    assert messages == {} and threads == {}


def test_exact_wave_preserves_an_explicit_empty_slot_transcript() -> None:
    from ouroboros.tools.plan_review_artifacts import exact_wave

    result = exact_wave(
        {"cycle_index": 1}, plan_prose="plan", manifest={}, slots=[], rows=[{
            "slot_id": "s1", "route": "api_chat", "model": "m",
        }], system_prompt="system", user_content="user", session_task="",
        slot_messages={"s1": []},
    )

    assert result["reviewer_outputs"][0]["request_messages"] == []


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
    actor row's disclosure, never a blanked artifact or a fabricated parse cause."""
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot
    from ouroboros.tools.plan_review_runtime import plan_row_disclosures

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
    assert plan_row_disclosures({"profile_continuity_receipt": result.usage[
        "profile_continuity_receipt"]}) == [
        "profile_continuity: cannot_verify (unexplained_profile_drift)"]


def test_empty_profile_expectation_is_matched_telemetry_without_disclosure() -> None:
    """An expectation that was never recorded is not drift: unpinned waves stay
    `matched`/`no_expectation_recorded` — durable receipt only, no actor noise."""
    from ouroboros.review_thread_continuity import profile_continuity_receipt
    from ouroboros.tools.plan_review_runtime import plan_row_disclosures

    for applied in ("profile-b", ""):  # FIX 4: ""/"" is no-expectation, not a false pin match
        receipt = profile_continuity_receipt("", applied, [])
        assert receipt["status"] == "matched"
        assert receipt["verification_reason"] == "no_expectation_recorded"
        assert plan_row_disclosures({"profile_continuity_receipt": receipt}) == []
    matched = profile_continuity_receipt("profile-a", "profile-a", [])
    assert (matched["status"], matched["verification_reason"]) == ("matched", "profile_matched")


def test_profile_drift_disclosure_lands_in_the_wave_actor_record(_harness, monkeypatch) -> None:
    """FIX 3 / owner decision 1=B: a cannot_verify continuity receipt riding one
    reviewer's usage lands in the recorded wave actor's `disclosures`, while the
    wave aggregate counts that reviewer's findings completely — telemetry only."""
    import ouroboros.review_substrate as rs
    from ouroboros.task_results import load_plan_review_state
    from tests.test_plan_review_engine import CLEAN, _call, _finding

    drift = json.dumps([_finding("f1", "blocking", breaks="goal", summary="drops swarm")])

    def _sub(request, *, slots, drive_root, llm, usage_ctx=None):
        actors = []
        for slot in slots:
            usage = {"prompt_tokens": 10, "completion_tokens": 5}
            if slot.slot_id == "s1":
                usage["profile_continuity_receipt"] = {
                    "expected_profile": "profile-a", "applied_profile": "profile-b",
                    "status": "cannot_verify",
                    "verification_reason": "unexplained_profile_drift",
                }
            actors.append({
                "slot_id": slot.slot_id, "model": slot.model, "status": "ok",
                "raw_text": drift if slot.slot_id == "s1" else CLEAN, "error": "",
                "usage": usage, "prompt_ref": {}, "response_ref": {},
            })
        return SimpleNamespace(actors=actors)

    monkeypatch.setattr(rs, "run_review_request", _sub)
    _call(_harness.make_ctx())

    wave = load_plan_review_state(_harness.drive, "task-1")["waves"][-1]
    actor = next(a for a in wave["actors"] if a["slot_id"] == "s1")
    assert "profile_continuity: cannot_verify (unexplained_profile_drift)" in actor["disclosures"]
    assert actor["ok"] is True  # the paid transcript stayed parseable
    assert wave["aggregate"] == "REVIEW_REQUIRED"  # computed normally, never gated
    assert wave["counts"] == {**wave["counts"], "parseable": 3, "blocking": 1}


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


def test_disposition_supersedes_the_exact_wave_before_hot_state(_harness) -> None:
    from ouroboros.task_results import load_plan_review_state
    from ouroboros.tools import plan_review as pr
    from ouroboros.tools.plan_review_artifacts import read_wave
    from tests.test_plan_review_engine import CLEAN, _call, _finding

    note = json.dumps([_finding("n1", "note")])
    _harness.install({"s1": note, "s2": CLEAN, "s3": CLEAN})
    _call(_harness.make_ctx())
    prior = load_plan_review_state(_harness.drive, "task-1")["waves"][-1]
    prior_ref = prior["wave_artifact"]

    pr._handle_plan_task(_harness.make_ctx(), review_disposition={
        "review_fingerprint": prior["request_fingerprint"],
        "items": [{"finding_id": "s1:n1", "decision": "accept", "rationale": "will do"}],
    })

    stored = load_plan_review_state(_harness.drive, "task-1")["waves"][-1]
    assert stored["wave_artifact"] != prior_ref
    exact = read_wave(_harness.drive, "task-1", stored["wave_artifact"])
    assert exact["dispositions"][0]["finding_id"] == "s1:n1"
    assert exact["supersedes_wave_artifact"] == prior_ref
    assert exact["artifact_meta"]["retention_owner"] == "task_artifact_store"


def test_33rd_blocker_is_dispositionable(_harness) -> None:
    from ouroboros.task_results import load_plan_review_state
    from ouroboros.tools import plan_review as pr
    from tests.test_plan_review_engine import CLEAN, _call

    findings = json.dumps([_finding(index) for index in range(1, 33)] + [_finding(33, "blocking")])
    _harness.install({"s1": findings, "s2": CLEAN, "s3": CLEAN})
    _call(_harness.make_ctx())
    wave = load_plan_review_state(_harness.drive, "task-1")["waves"][-1]

    out = pr._handle_plan_task(_harness.make_ctx(), review_disposition={
        "review_fingerprint": wave["request_fingerprint"],
        "items": [{"finding_id": "s1:f33", "decision": "reject", "rationale": "not valid"}],
    })

    assert "unknown finding ids" not in out
    assert "blocking_finding_below_quorum_stays_open" in out


def test_roster_change_preserves_undisposed_prior_blocking_findings(_harness) -> None:
    """Owner-decided contract: after a roster change the wave restarts FRESH (no
    refusal), the new packet still carries the prior cycle's undisposed blocking
    findings via prior_cycles, the recorded prior wave keeps them open, and every
    actor row carries the typed continuation-restart capability delta."""
    from ouroboros.task_results import load_plan_review_state
    from tests.test_plan_review_engine import CLEAN, _call, _finding, _slots, _user_text

    wave1 = json.dumps([
        _finding("b1", "blocking", breaks="goal", summary="the plan drops swarm delegation"),
        _finding("e1", "need_evidence", locator="notes.md::tail=16", summary="show the notes tail"),
    ])
    _harness.install({"s1": wave1, "s2": CLEAN, "s3": CLEAN})
    _call(_harness.make_ctx())
    prior = load_plan_review_state(_harness.drive, "task-1")["waves"][-1]
    assert prior["closed"] is False

    _harness.state["slots"] = _slots(("s1", "m/a"), ("s2", "m/b"), ("s3", "m/new"))
    substrate = _harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    out = _call(_harness.make_ctx())

    assert "cannot_verify" not in out
    assert len(substrate.calls) == 1
    user = _user_text(substrate.calls[0]["request"].messages[1]["content"])
    assert "the plan drops swarm delegation" in user
    state = load_plan_review_state(_harness.drive, "task-1")
    wave2 = state["waves"][-1]
    assert wave2["cycle_index"] == 2
    for actor in wave2["actors"]:
        assert {
            "kind": "capability_delta",
            "requested": "continuation of prior thread",
            "effective": "fresh session, full packet",
            "reason": "prior_reviewer_assignment_set_changed",
        } in actor["capability_delta"]
    recorded_prior = next(w for w in state["waves"] if w["cycle_index"] == 1)
    assert recorded_prior["dispositions"] == [] and recorded_prior["closed"] is False


def test_degraded_progress_line_discloses_untrusted_counts(_harness) -> None:
    from tests.test_plan_review_engine import _call

    _harness.install({"s1": "no json here", "s2": "no json here", "s3": "no json here"})
    _harness.progress.clear()
    _call(_harness.make_ctx())

    assert (
        "📐 plan_task: DEGRADED (0/3 parseable reviewers; counts are untrusted) — "
        "0 blocking / 0 note / 0 need_evidence; cycles paid 1/2"
    ) in _harness.progress


def test_clean_progress_line_stays_byte_identical(_harness) -> None:
    from tests.test_plan_review_engine import CLEAN, _call

    _harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    _harness.progress.clear()
    _call(_harness.make_ctx())

    assert (
        "📐 plan_task: GREEN — 0 blocking / 0 note / 0 need_evidence; cycles paid 1/2"
    ) in _harness.progress


def test_session_output_schema_admits_null_in_its_own_optional_fields() -> None:
    """§1.4: a model that emits null for an absent optional field must conform.

    The schema previously declared plain "string" for breaks/locator/recommendation,
    so every schema-bearing route landed outputConformance=failed on legitimate
    nulls (incident f0a6fb64c92d491f). Required fields stay strict strings.
    """
    from ouroboros.tools.plan_spec import PLAN_REVIEW_SESSION_OUTPUT_SCHEMA

    props = PLAN_REVIEW_SESSION_OUTPUT_SCHEMA["properties"]["findings"]["items"]["properties"]
    for optional in ("breaks", "locator", "recommendation"):
        assert props[optional]["type"] == ["string", "null"], optional
    for required in ("id", "summary"):
        assert props[required]["type"] == "string", required


def test_validate_findings_treats_null_optionals_as_absent() -> None:
    from ouroboros.tools import plan_spec

    rows, disclosures, _seen = plan_spec.validate_findings(
        [{"id": "f1", "class": "note", "summary": "s",
          "breaks": None, "locator": None, "recommendation": None}],
        spec_ids=["goal", "claim_1"], seen_locators=[],
    )
    assert len(rows) == 1 and rows[0]["class"] == "note"
    assert not rows[0].get("breaks")
