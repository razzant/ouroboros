"""Plan-review continuity seams that outlive the retired engine: exact evidence selectors,
Claudexor thread receipts and turns, profile pins and rotation receipts (telemetry, never a
gate: owner decision 1=B), the agent-session executor's paid transcript and model attestation,
the parser of the planning organ's reply, and the earlier-schema wave compaction's counts."""
from __future__ import annotations

import json

import httpx
import pytest


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
