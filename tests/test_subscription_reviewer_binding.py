"""Reviewer sizing and explicit owner authority belong to a frozen account row."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence as ce
from ouroboros.llm import LLMClient
from ouroboros.llm_claudexor import ClaudexorModelError
from ouroboros.review_records import ReviewSlot
from ouroboros.reviewer_window import resolve_reviewer_window
from ouroboros.tools.review_synthesis import per_slot_input_token_limits

MODEL = "claudexor::test-source=exact-model"


@pytest.fixture
def accounts(tmp_path, monkeypatch):
    from ouroboros import config

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", json.dumps({"main": "not-a-reviewer"}))
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ce, "resolve_review_token_density", lambda *_a, **_kw: (1.0, "measured"))
    current = {"auto": "account-a", "calls": [], "windows": {"account-a": 200_000, "account-b": 800_000}}

    def catalog(source, credential_profile_id=None, *, requested_model=None):
        account = credential_profile_id or current["auto"]
        current["calls"].append((source, credential_profile_id))
        return {"source": source, "credentialProfileId": account, "accountFingerprint": f"identity-{account}",
                "observedAt": ce.utc_now_iso(), "provenance": "owned exact transport metadata",
                "models": [{"id": "exact-model", "contextWindow": current["windows"][account]}]}

    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(catalog))
    monkeypatch.setattr(ce, "_generative_probe_window", lambda *_a, **_kw: pytest.fail("No generation probe"))
    return current


def _slots():
    return [ReviewSlot(slot_id="review-a", model=MODEL, session_profile="account-a", use_local=False, max_tokens=16_000),
            ReviewSlot(slot_id="review-b", model=MODEL, session_profile="account-b", use_local=False, max_tokens=16_000)]


def _options(account):
    return {"source_id": "test-source", "credential_profile_id": account, "account_fingerprint": f"identity-{account}"}


def test_same_model_slots_keep_their_actual_distinct_capacity_and_acceptance_cap(accounts):
    from ouroboros.review_dispatch import acceptance_slot_fit
    from ouroboros.review_evidence_sections import acceptance_packet_budget_chars

    slots = _slots()
    caps = per_slot_input_token_limits([MODEL, MODEL], slots=slots, output_reserve=16_000, tokenizer_margin=50_000)
    assert caps == {"review-a": 159_000, "review-b": 734_000}
    packet = acceptance_packet_budget_chars(slots)
    assert packet.slot_input_caps == caps
    executor = SimpleNamespace(prompt_chars=lambda: 100_000)
    assert acceptance_slot_fit(slots[0], executor, slot_input_caps=caps)[0] == 159_000
    assert acceptance_slot_fit(slots[1], executor, slot_input_caps=caps)[0] == 734_000
    assert {account for _source, account in accounts["calls"]} == {"account-a", "account-b"}


def test_plan_fit_excludes_only_the_small_account_of_same_model(accounts):
    from ouroboros.tools.plan_review_runtime import plan_slot_fit

    slots = _slots()
    accepted, rejected, error = plan_slot_fit(slots, prompt_chars=1_200_000, quorum=1)
    assert [slot.slot_id for slot in accepted] == ["review-b"]
    assert [row["slot_id"] for row in rejected] == ["review-a"]
    assert not error


def test_retrieving_seat_sizing_uses_frozen_role_pin_and_not_main(accounts):
    """Each retrieving seat's window — and therefore its output reserve —
    resolves under the seat's OWN frozen role pin and credential profile."""
    from ouroboros.reviewer_window import window_scaled_reserves
    from ouroboros.tools.review_multi_model import _review_output_budget
    from ouroboros.tools.scope_window import scope_window

    first = scope_window(MODEL, model_role="reviewer:scope-a", credential_profile_id="account-a")
    second = scope_window(MODEL, model_role="reviewer:scope-b", credential_profile_id="account-b")
    assert (first.window_tokens, second.window_tokens) == (200_000, 800_000)
    # The brief's first-send bound scales the one review output budget by the
    # seat's own window (review_brief_coupling.first_send_bound): a quarter of
    # the small window, the full budget where the window has room for it.
    scaled = [window_scaled_reserves(w.window_tokens, output_reserve=_review_output_budget(), tokenizer_margin=50_000)[0]
              for w in (first, second)]
    assert scaled == [50_000, _review_output_budget()]


def test_triad_fit_uses_the_already_frozen_rows(tmp_path, accounts, monkeypatch):
    from ouroboros.tools import review
    from ouroboros.tools.review_admission import fit_triad_prompt

    captured = []
    quorum = review._quorum_input_token_limit
    monkeypatch.setattr(review, "_quorum_input_token_limit", lambda keys, caps: captured.append(dict(caps)) or quorum(keys, caps))
    slots = _slots()
    prompt, _stable, refused = fit_triad_prompt([MODEL, MODEL], lambda *_args: ("tiny prompt", 1),
                                               "files", "diff", "file.py", tmp_path, slots=slots)
    assert prompt == "tiny prompt" and not refused
    assert captured[0]["review-a"] < captured[0]["review-b"]
    assert {account for _source, account in accounts["calls"]} == {"account-a", "account-b"}


def test_bound_cap_map_with_missing_slot_is_never_an_unlimited_dispatch():
    from ouroboros.review_dispatch import acceptance_slot_fit

    with pytest.raises(ValueError, match="frozen reviewer slot"):
        acceptance_slot_fit(_slots()[0], SimpleNamespace(prompt_chars=lambda: 100_000),
                            slot_input_caps={"different-slot": 800_000})


@pytest.mark.parametrize("mode", ["pin", "auto"])
def test_ack_is_reachable_after_fresh_binding_without_an_extra_fetch_and_stays_on_a(tmp_path, accounts, mode):
    ce.record_owner_ack(tmp_path, provider="claudexor", model=MODEL, window_tokens=1_200_000, options=_options("account-a"))
    calls = len(accounts["calls"])
    options = {"source_id": "test-source", "credential_profile_id": "account-a" if mode == "pin" else ""}
    a = ce.probe(tmp_path, provider="claudexor", model=MODEL, options=options)
    assert a.source == "owner_ack" and a.window_tokens == 1_200_000
    assert len(accounts["calls"]) == calls + 1
    accounts["auto"] = "account-b"
    b = ce.probe(tmp_path, provider="claudexor", model=MODEL,
                 options={"source_id": "test-source", "credential_profile_id": "account-b" if mode == "pin" else ""})
    assert b.source == "provider_metadata" and b.window_tokens == 800_000
    assert b.route_fp != a.route_fp and not ce.confirms_at_least(b)
    assert len(accounts["calls"]) == calls + 2


def test_explicit_capability_ack_passes_exact_options_through_existing_gateway(tmp_path, accounts, monkeypatch):
    """The ack endpoint binds one exact route and one exact account.

    It is the write path for owner window evidence on any route the host sizes
    against — the main model's own working window included — so an ack minted for
    account A must stay unusable for the same model on account B.
    """
    from ouroboros.gateway import settings as gateway

    route = {"provider": "claudexor", "model": MODEL, "base_url": "", "options": _options("account-a")}
    payload = {key: deepcopy(value) for key, value in route.items()}
    payload["route_fp"] = ce.route_fingerprint(provider=route["provider"], base_url=route["base_url"],
                                               model=route["model"], options=route["options"])
    payload["window_tokens"] = 1_200_000

    async def read_body():
        return payload

    monkeypatch.setattr(gateway, "_owner_audit", lambda *_a, **_kw: None)
    request = SimpleNamespace(json=read_body, app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)))
    response = asyncio.run(gateway.api_acknowledge_capability(request))
    assert response.status_code == 200
    ack = json.loads(response.body)["ack"]
    assert ack["route_fp"] == payload["route_fp"]
    assert ack["binding_evidence"]["credential_profile_id"] == "account-a"
    acked = ce.probe(tmp_path, provider="claudexor", model=MODEL,
                     options={"source_id": "test-source", "credential_profile_id": "account-a"})
    assert acked.source == "owner_ack" and ce.confirms_at_least(acked, 1_200_000)
    other = ce.probe(tmp_path, provider="claudexor", model=MODEL,
                     options={"source_id": "test-source", "credential_profile_id": "account-b"})
    assert other.source != "owner_ack" and not ce.confirms_at_least(other, 1_200_000)


@pytest.mark.parametrize("change", [{}, {"account_fingerprint": "old-identity"}, {"source_id": "other-source"}])
def test_unknown_or_changed_binding_cannot_mint_owner_ack(tmp_path, accounts, change):
    options = None if not change else {**_options("account-a"), **change}
    with pytest.raises(ValueError):
        ce.record_owner_ack(tmp_path, provider="claudexor", model=MODEL, window_tokens=1_200_000, options=options)
    assert ce.list_owner_acks(tmp_path) == []


def test_manual_sizing_stays_separate_from_explicit_scope_ack(tmp_path, accounts, monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL_CONTEXT_WINDOWS", json.dumps({"deep_review": 1_200_000}))
    window = resolve_reviewer_window(MODEL, model_role="deep_review", credential_profile_id="account-a")
    assert window.sizing_window() == 1_200_000 and window.window_tokens == 200_000
    assert window.sizing_source == "user_setting" and ce.list_owner_acks(tmp_path) == []


@pytest.mark.parametrize("code", ["subscription_window_exhausted", "auth_required", "model_outcome_unknown"])
def test_existing_density_rung_keeps_frozen_binding_and_does_not_swallow_typed_error(tmp_path, accounts, monkeypatch, code):
    from ouroboros import llm_observability

    monkeypatch.setattr(ce, "resolve_review_token_density", lambda *_a, **_kw: (1.65, "cold"))
    seen = []
    error = ClaudexorModelError({"code": code, "message": "typed fixture"})

    def refuse(_client, **kwargs):
        seen.append(kwargs)
        raise error

    monkeypatch.setattr(llm_observability, "chat_observed", refuse)
    with pytest.raises(ClaudexorModelError) as raised:
        ce.cold_start_density_probe(tmp_path, object(), lambda _text: None, MODEL, "sample",
                                   task_id="test", call_type="test", source="test",
                                   model_role="reviewer:frozen", model_account_override="account-b")
    assert raised.value is error and len(seen) == 1
    assert seen[0]["model_role"] == "reviewer:frozen" and seen[0]["model_account_override"] == "account-b"


def test_density_after_owner_model_switch_is_never_written_under_the_old_model(tmp_path, monkeypatch):
    from ouroboros import llm_observability

    monkeypatch.setattr(ce, "resolve_review_token_density", lambda *_a, **_kw: (1.65, "cold"))
    writes = []
    monkeypatch.setattr(ce, "record_token_density", lambda _root, model, **kwargs: writes.append((model, kwargs)))
    monkeypatch.setattr(llm_observability, "chat_observed", lambda *_a, **_kw: ({"content": "OK"}, {
        "prompt_tokens": 42, "resolved_model": "stale-display", "model_role_route": {"model": "openai::different"}}))
    outcome = ce.cold_start_density_probe(tmp_path, object(), lambda _text: None, MODEL, "sample",
                                          task_id="test", call_type="test", source="test")
    assert outcome == "unrecorded" and len(writes) == 1
    assert writes[0][0] == ce._normalized_density_model("openai::different")
