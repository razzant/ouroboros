"""Owner model switches reprepare frozen review assignments, never their tools."""

from dataclasses import replace
import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence as ce
from ouroboros.llm import LLMClient
from ouroboros.review_records import ReviewSlot, apply_review_model_override
from ouroboros.reviewer_slot_config import ConfiguredReviewerSlot
from tests.test_llm_claudexor import MODEL, ROUTE, result, ledger, setup as gateway_fixture
from tests.test_model_wait import live_wait as wait_fixture

setup = gateway_fixture
live_wait = wait_fixture


def test_pure_projection_preserves_native_delivery_and_changes_only_one_role():
    # F8: the delivery class is the slot's own fact; the subagent id is identity only.
    native = ReviewSlot(slot_id="critic", model=MODEL, effort="high", subagent_id="frozen-actor", session_profile="a",
                        native_retrieval_override=True)
    other = replace(native, slot_id="other", session_profile="other-pin")
    override = {"reviewer:critic": {"model": "local-review", "model_account_override": "", "use_local": True},
                "main": {"model": "unrelated", "model_account_override": "main", "use_local": False}}
    applied = apply_review_model_override(native, override)
    assert (applied.slot_id, applied.effort, applied.subagent_id, applied.native_retrieval) == ("critic", "high", "frozen-actor", True)
    assert (applied.model, applied.session_profile, applied.use_local) == ("local-review", "", True)
    assert native.model == MODEL and native.session_profile == "a"
    assert apply_review_model_override(other, override) is other
    configured = ConfiguredReviewerSlot("critic", "api_chat", MODEL, subagent_id="actor")
    assert apply_review_model_override(configured, override).native_retrieval
    # The override is keyed by the row's identity alone: a caller-named identity
    # (the deep review's Main row, a preflight seat) takes the same projection.
    assert apply_review_model_override(configured, {"reviewer:main": override["reviewer:critic"]},
                                       slot_id="main").use_local is True


@pytest.mark.parametrize("narrow", [False, True])
def test_native_wait_switch_rechecks_bound_before_send_and_never_replays_read(live_wait, monkeypatch, narrow):
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
    from ouroboros.review_execution import ReviewAssignment, ReviewRouteUnavailable
    from ouroboros.review_records import ReviewRequest

    root, gateway, client, controller, events, decide = live_wait
    subject = root / "subject"
    subject.mkdir(parents=True)
    (subject / "file.txt").write_text("completed original read", encoding="utf-8")
    route_b = {**ROUTE, "credentialProfileId": "account-b", "accountFingerprint": "fingerprint-b"}
    first = result()
    first["message"] = {"role": "assistant", "content": "Inspecting", "tool_calls": [
        {"id": "read-one", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"file.txt"}'}}]}
    quota = result(outcome="failed", problem={"code": "subscription_window_exhausted", "message": "quota"})
    final = result(route=route_b)
    final["message"] = {"role": "assistant", "content": '[{"severity":"advisory","item":"x","evidence":"e","recommendation":"r"}]'}
    gateway.results, gateway.dispatch = [first, quota, final], ["response_received", "not_started", "response_received"]
    monkeypatch.setattr(ce, "resolve_review_token_density", lambda *_a, **_kw: (1.0, "measured"))

    def catalog(source, profile=None, *, requested_model=None):
        profile = profile or "account-a"
        return {"source": source, "credentialProfileId": profile,
                "accountFingerprint": "fingerprint-b" if profile == "account-b" else "fingerprint-a",
                "observedAt": ce.utc_now_iso(), "provenance": "fixture",
                "models": [{"id": "exact-model", "contextWindow": 4_000 if narrow and profile == "account-b" else 800_000}]}

    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(catalog))
    from ouroboros import config
    monkeypatch.setattr(config, "DATA_DIR", root)
    decisions = []

    def choose(*_args, **_kwargs):
        event = next(event for event in reversed(list(events.queue)) if event.get("type") == "task_model_wait")
        response = decide({"request_id": "switch", "decision_id": f"model_wait:task-one:{event['wait_id']}",
                           "revision": event["revision"], "action": "switch", "model": MODEL,
                           "credential_profile_id": "account-b", "use_local": False, "persist_role": False})
        assert response.status_code == 202
        decisions.append(response)
        return {}

    monkeypatch.setattr(client, "claudexor_model_catalog", choose)
    request = ReviewRequest(surface="multi_model_review", goal="Review", task_id="task-one",
                            session_root=str(subject), session_task="Read file.txt and review it.",
                            policy={"output_contract": "JSON array of findings"})
    slot = ReviewSlot("critic", MODEL, session_profile="account-a", subagent_id="frozen-actor")
    executor = NativeToolRoundReviewExecutor(ReviewAssignment(request, slot, call_id="episode"), llm=client)
    reads = []
    real_read = executor._execute_inspection_call

    def read_once(*args, **kwargs):
        reads.append(1)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(executor, "_execute_inspection_call", read_once)
    if narrow:
        with pytest.raises(ReviewRouteUnavailable) as raised:
            executor.execute()
        assert raised.value.code == "native_transcript_cap_exceeded"
        assert len(gateway.accepted_operations) == 2
        assert executor.failure_custody()["native_transcript_bound"] == 0
        with pytest.raises(ReviewRouteUnavailable):
            executor.execute()
        assert len(gateway.accepted_operations) == 2
    else:
        answer = executor.execute()
        assert answer.raw_text == final["message"]["content"]
        assert answer.usage["model_role_route"]["credential_profile_id"] == "account-b"
        assert answer.usage["claudexor"]["route"] == route_b
        sent = gateway.uploads[-1][0]
        assert sent["account"] == {"mode": "pin", "profileId": "account-b"}
        assert any(message.get("role") == "tool" and "completed original read" in message["content"] for message in sent["messages"])
        assert answer.usage["native_rounds"] == 2 and len(gateway.accepted_operations) == 3
    assert len(reads) == len(decisions) == 1
    assert controller.overrides.keys() == {"reviewer:critic"}
    assert [row["state"] for row in ledger(root)].count("settled") == (1 if narrow else 2)


def test_plan_and_acceptance_prepare_effective_slot_before_capacity(live_wait, monkeypatch):
    from ouroboros.review_evidence_sections import acceptance_packet_budget_chars
    from ouroboros.tools.plan_review_runtime import plan_slot_fit
    from ouroboros import reviewer_window

    _root, _gateway, _client, controller, _events, _decide = live_wait
    slot = ReviewSlot("critic", MODEL, session_profile="account-a")
    controller.overrides["reviewer:critic"] = {"model": "local-review", "use_local": True, "model_account_override": ""}
    calls = []

    def capacity(model, **binding):
        calls.append((model, binding))
        return 600_000

    monkeypatch.setattr(reviewer_window, "reviewer_context_window", capacity)
    accepted, _rows, _error = plan_slot_fit([slot], prompt_chars=200, quorum=1)
    budget = acceptance_packet_budget_chars([slot])
    assert accepted[0].model == "local-review" and accepted[0].use_local
    assert "critic" in budget.slot_input_caps
    assert all(model == "local-review" and binding["use_local"] is True
               and binding["credential_profile_id"] == "" for model, binding in calls)
    assert slot.model == MODEL and slot.session_profile == "account-a"


def test_triad_density_switch_reprepares_cap_and_mutates_only_the_frozen_slot(live_wait, monkeypatch):
    from ouroboros.tools import review, review_admission

    root, _gateway, _client, controller, _events, _decide = live_wait
    slots = [ReviewSlot("critic", MODEL, session_profile="account-a")]
    models = [MODEL]
    caps = []

    def window(model, **kwargs):
        caps.append((model, kwargs["credential_profile_id"]))
        return 40_000 if model == MODEL else 800_000

    def switched(*_args, **_kwargs):
        controller.overrides["reviewer:critic"] = {"model": "changed-model", "model_account_override": "account-b", "use_local": False}
        return "unrecorded"

    monkeypatch.setattr(review, "reviewer_context_window", window)
    monkeypatch.setattr(review_admission, "density_probe_before_size_refusal", switched)
    _prompt, _stable, error = review_admission.fit_triad_prompt(
        models, lambda *_args: ("x" * 80_000, 1), "files", "diff", "file.py", root,
        ctx=SimpleNamespace(drive_root=root), slots=slots)
    assert not error and models == ["changed-model"]
    assert slots[0].model == "changed-model" and slots[0].session_profile == "account-b"
    assert caps == [(MODEL, "account-a"), ("changed-model", "account-b")]


def test_review_substrate_dispatches_the_effective_profile_with_real_ledger(live_wait, monkeypatch):
    from ouroboros.review_substrate import run_review_request
    from ouroboros.review_records import ReviewRequest

    root, gateway, client, controller, _events, _decide = live_wait
    controller.overrides["reviewer:critic"] = {"model": MODEL, "model_account_override": "account-b", "use_local": False}
    final = result()
    final["message"] = {"content": '[{"severity":"advisory","item":"x","evidence":"e","recommendation":"r"}]'}
    gateway.results = [final]
    request = ReviewRequest(surface="multi_model_review", goal="Review", task_id="task-one",
                            messages=[{"role":"user","content":"Review this"}])
    outcome = run_review_request(request, slots=[ReviewSlot("critic", MODEL, session_profile="account-a")],
                                 drive_root=root, llm=client)
    assert outcome.actors[0]["status"] == "ok"
    assert gateway.uploads[0][0]["account"] == {"mode":"pin", "profileId":"account-b"}
    assert ledger(root)[-1]["state"] == "settled"


def test_raw_triad_query_keeps_frozen_profile_without_wait_override_rescuing_it(setup):
    from ouroboros.tools.review_multi_model import _query_model
    from ouroboros.tools.registry import ToolContext

    root, gateway, client = setup
    completed = result(route={**ROUTE, "credentialProfileId": "account-b", "accountFingerprint": "fingerprint-b"})
    completed["message"] = {"content": '[{"severity":"advisory","item":"x","evidence":"e","recommendation":"r"}]'}
    gateway.results = [completed]
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id="task-one")

    async def run():
        return await _query_model(client, MODEL, [{"role": "user", "content": "Review"}], asyncio.Semaphore(1),
                                  ctx, slot_id="critic", session_profile="account-b", effort="high")

    _model, payload, _extra = asyncio.run(run())
    assert "error" not in payload
    assert gateway.uploads[0][0]["account"] == {"mode": "pin", "profileId": "account-b"}
    assert gateway.uploads[0][0]["options"]["reasoningEffort"] == "high"
    assert ledger(root)[-1]["state"] == "settled" and ledger(root)[-1]["cost_usd"] is None


def test_retrieving_seat_sends_under_the_row_plan_profile_not_the_original_slot(setup, monkeypatch):
    """The one wave dispatches each retrieving seat with the profile its row plan
    carries (`session_profiles`), never the configured slot object's original
    profile: the reservation, the catalog lookup and the pinned send all name
    that account, and the seat's output reserve is scaled by THAT account's window."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_multi_model import _query_model, _review_output_budget
    from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS
    from ouroboros import config

    root, gateway, client = setup
    monkeypatch.setattr(config, "DATA_DIR", root)
    catalog_profiles = []

    def catalog(source, profile=None, *, requested_model=None):
        catalog_profiles.append(profile)
        return {"source": source, "credentialProfileId": profile, "accountFingerprint": "fingerprint-b",
                "observedAt": ce.utc_now_iso(), "provenance": "fixture",
                "models": [{"id": "exact-model", "contextWindow": 200_000 if profile == "account-b" else 800_000}]}

    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(catalog))
    original = ReviewSlot("scope-one", MODEL, session_profile="account-a")
    completed = result(route={**ROUTE, "credentialProfileId": "account-b", "accountFingerprint": "fingerprint-b"})
    matrix = [{"item": item, "verdict": "PASS", "severity": "advisory", "reason": "ok"} for item in sorted(SCOPE_REQUIRED_ITEMS)]
    completed["message"] = {"content": json.dumps({"change": [], "change_clean": True, "coupling": matrix})}
    gateway.results = [completed]

    async def seat():
        return await _query_model(client, MODEL, [], asyncio.Semaphore(1),
                                  ToolContext(repo_dir=root, drive_root=root, task_id="task-one"),
                                  slot_id=original.slot_id, route=ReviewRouteKind.API_CHAT, effort="high",
                                  session_profile="account-b", native_retrieval=True,
                                  session_task="Review the staged change", session_root=str(root))

    _model, payload, _extra = asyncio.run(seat())
    # Window size no longer decides authority: the seat answers on the account
    # its row was PREPARED with, and that account's profile is what gets pinned.
    assert "error" not in payload and payload["choices"][0]["message"]["content"]
    assert gateway.uploads[0][0]["account"] == {"mode": "pin", "profileId": "account-b"}
    assert catalog_profiles and set(catalog_profiles) == {"account-b"}
    assert payload["usage"]["prompt_tokens"] == 20 and ledger(root)[-1]["state"] == "settled"
    assert payload["prompt_ref"]  # The real substrate persisted its actual request.
    # Every seat of the one wave reserves the one review output budget
    # (review_multi_model._review_output_budget), the retrieving seat included.
    assert payload["usage"]["claudexor"]["output_reserve_tokens"] == _review_output_budget()
    assert original.session_profile == "account-a"
