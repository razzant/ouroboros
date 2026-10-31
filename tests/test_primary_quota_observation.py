"""#1628: quota-only facts about the primary pool, read by the existing non-generating observation.

The observation reads the owned engine's passive quota envelope for the primary's exact
model, request kind and pin (or Auto) inside the one shared metadata budget, and the
acting model reads the engine's own aggregate beside bounded, dated rows. A failed read,
an unread envelope and an empty one stay distinct; a pin never borrows a sibling; the
default alias needs a positive roster; nothing refreshes, starts, selects or generates.
"""

from __future__ import annotations

import copy
import json
import re
from types import SimpleNamespace

import pytest

from ouroboros import claudexor_daemon, llm_capability_policy, llm_probe
from ouroboros import primary_route_observation as observation
from ouroboros import usage_accounting as ua
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
from tests import test_unknown_fallback_first as fallback_fixtures
from tests.test_completion_selection import finish
from tests.test_llm_claudexor import setup as gateway_fixture
from tests.test_model_wait import live_wait as wait_fixture
from tests.test_route_switch_facts import FALLBACK, PRIMARY, SOL, _facts, _Routes, _tools
from tests.test_subscription_main_wait import main_call as main_call_fixture
from tests.test_unknown_fallback_first import _death, _run

data_root = fallback_fixtures.data_root
setup, live_wait, main_call = gateway_fixture, wait_fixture, main_call_fixture
RESET = "2099-01-01T00:00:00Z"


def _snapshot(subject_id, *, state="available", freshness="fresh", resets_at=None, source="codex_app_server",
              observed_at="2026-10-21T01:00:00Z", harness="codex"):
    return {"subject": {"harness": harness, "credential_route": "vendor_native", "plan_label": "Pro plan",
                        "subject_id": subject_id},
            "constraints": [{"id": "5h", "label": "five hour window", "used_ratio": 0.42, "window_seconds": 18000,
                             "resets_at": resets_at, "cooldown_until": None, "freshness": freshness}],
            "source": source, "observed_at": observed_at, "freshness": freshness, "snapshot_id": f"s-{subject_id}",
            "availability": {"state": state, "blocking_constraints": [] if state == "available" else ["5h"],
                             "resets_at": resets_at, "model_scoped_exhaustions": []}}


def _absence(subject_id, reason="not_logged_in"):
    return {"subject": {"harness": "codex", "credential_route": "vendor_native", "plan_label": None,
                        "subject_id": subject_id},
            "reason": reason, "detail": "owner@example.com token at /secret/path", "observed_at": "2026-10-21T01:00:00Z"}


def _envelope(*snapshots, absences=(), model="gpt-test", scoped=True):
    value = {"snapshots": list(snapshots), "absences": list(absences), "refreshed_at": None,
             "resources": [{"target": {"harness": "codex", "profile_id": "acct-a"}, "balances": [{"usd": 99.0}]}]}
    if scoped:
        value["availability_scope"] = {"model": model, "requestKind": "text", "basis": "quota_block"}
    return value


ROSTER = {"profiles": [{"profile": {"harness_id": "codex", "profile_id": "codex-default", "enabled": True,
                                    "display_name": "Owner Name", "email": "owner@example.com"}}],
          "harnessAccounts": [{"harness_id": "codex"}], "accountPools": []}


class _Engine:
    """The owned engine's passive reads only: a refresh, a start or a generation fails the test."""

    def __init__(self, *quota, roster=None, clock=None, step=0.0):
        self.quota, self.roster, self.calls, self.clock, self.step = list(quota), roster, [], clock, step

    def _tick(self):
        if self.clock is not None:
            self.clock[0] += self.step

    def quota_state(self, **kwargs):
        self.calls.append(("quota", kwargs))
        self._tick()
        reply = self.quota.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return copy.deepcopy(reply)

    def credential_profiles(self, **kwargs):
        self.calls.append(("roster", kwargs))
        self._tick()
        if isinstance(self.roster, Exception):
            raise self.roster
        return copy.deepcopy(self.roster)

    def close(self):
        self.calls.append(("close", {}))

    def __getattr__(self, name):
        pytest.fail(f"a quota observation used {name}")


def _connect(monkeypatch, engine, connects=None, clock=None, step=0.0):
    def read_owned_gateway(**kwargs):
        if connects is not None:
            connects.append(kwargs)
        if clock is not None:
            clock[0] += step
        if isinstance(engine, Exception):
            raise engine
        return engine

    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", read_owned_gateway)


def _observed(messages) -> list:
    rows = [match for row in messages
            for match in re.findall(r"\[PRIMARY ROUTE OBSERVATION\] (\{.*?\})\. Earlier", str(row.get("content")))]
    return [json.loads(row) for row in rows]


# -- the real loop: cadence, ageing, catalog failure and no attempts ----------------------------


def test_the_loop_reads_quota_at_boundaries_reports_ageing_and_never_a_newer_date_alone(data_root, tmp_path, monkeypatch):
    from tests._usage_store_testing import ledger_rows

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", FALLBACK)
    clock, probes, connects = [0], [], []
    monkeypatch.setattr(observation, "time", SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1))
    # The catalog never confirms the primary: that must not hide the quota facts.
    monkeypatch.setattr(llm_probe, "upstream_transport_reachable", lambda *a, **kw: probes.append(kw) or {})
    engine = _Engine(_envelope(_snapshot("acct-a", observed_at="2026-10-21T01:00:00Z")),
                     _envelope(_snapshot("acct-a", observed_at="2026-10-21T01:02:00Z")),  # newer date only
                     _envelope(_snapshot("acct-a", freshness="stale", observed_at="2026-10-21T01:02:00Z")))
    _connect(monkeypatch, engine, connects)
    llm = _Routes(data_root, **{PRIMARY: [_death], FALLBACK: [(_tools(), dict(SOL)) for _ in range(4)]
                                + [(finish("done on the fallback"), dict(SOL))]})
    original = llm.chat

    def chat(**kwargs):
        reply = original(**kwargs)
        own = [model for model, _ in llm.sent if model == FALLBACK]
        clock[0] = [0, 30, 150, 300, 301][len(own) - 1]
        return reply

    llm.chat = chat
    text, usage, trace, _registry = _run(tmp_path, llm)
    assert text == "done on the fallback" and len(ledger_rows(data_root)) == 6  # observations made no attempt
    quota_reads = [kwargs for name, kwargs in engine.calls if name == "quota"]
    assert len(quota_reads) == len(connects) == len(probes) == 3  # rounds at 0, 150 and 300 s, not at 30 or 301
    assert all(read["view"] == "resources_freshness" and read["model"] == "gpt-test" and read["request_kind"] == "text"
               and 0 < read["timeout_sec"] <= 3 for read in quota_reads)
    assert all(0 < kw["timeout_sec"] <= 3 for kw in connects) and all(0 < kw["timeout"] <= 3 for kw in probes)
    notes = trace["primary_route_observations"]
    assert [note["facts"]["quota"]["availability"]["state"] for note in notes] == ["available", "unknown"]
    assert notes[1]["facts"]["quota"]["availability"]["reason"] == "quota_evidence_stale_or_incomplete"
    assert all(note["facts"]["kind"] == "unconfirmed" and note["generation_tested"] is False for note in notes)
    seen = _observed(llm.sent[-1][1])
    assert [row["facts"]["quota"]["snapshots"][0]["freshness"] for row in seen] == ["fresh", "stale"]
    assert seen[0]["facts"]["quota"]["requested_scope"] == {"model": "gpt-test", "requestKind": "text", "account": "Auto"}
    assert "unknown is not exhaustion and available is not admission" in str(llm.sent[-1][1])
    assert not usage.get("primary_return_results")  # facts, never a host return


# -- one real switch: engine refusal, the API fallback's own money, the next input -------------


def test_a_real_engine_refusal_reaches_the_next_input_with_its_reset_the_prior_engine_call_and_pool_facts(
        main_call, monkeypatch):
    from copy import deepcopy

    from ouroboros import fallback_cooldown
    from ouroboros.llm_attempt import _attempt_request, _candidate_before_dispatch
    from tests.test_llm_claudexor import result
    from tests.test_model_wait_controls import _loop_tools
    from tests.test_quota_rotation_fallback import FALLBACK as API_FALLBACK
    from tests.test_quota_rotation_fallback import _pool_quota
    from tests.test_quota_rotation_fallback import _run as run_main

    from ouroboros import context

    ctx, gateway, owner, events, _decide, _observations = main_call
    shared_route = context._context_fit_route

    def fallback_holds_its_input(task, **kwargs):
        # The fallback's reported 512,206-token input is this test's subject; its route must hold it and a reply.
        route, evidence = shared_route(task, **kwargs)
        if evidence.route_fp == "other-route":
            evidence = SimpleNamespace(**{**vars(evidence), "window_tokens": 1_050_000})
        return route, evidence

    monkeypatch.setattr(context, "_context_fit_route", fallback_holds_its_input)
    tools = _loop_tools(ctx, owner)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", API_FALLBACK)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_args: False)
    gateway.results, gateway.dispatch = [result(), _pool_quota()], ["response_received", "not_started"]
    engine = _Engine(_envelope(_snapshot("account-a", state="exhausted", resets_at=RESET), _snapshot(
        "account-b", state="exhausted", resets_at="2099-01-02T00:00:00Z"), model="exact-model"))
    _connect(monkeypatch, engine)

    def no_engine(**_kwargs):
        raise ClaudexorUnavailable("daemon_not_discovered", "no owned engine for the catalog check")

    monkeypatch.setattr(llm_capability_policy, "read_owned_gateway", no_engine)
    sent, remote = [], ctx.llm._chat_remote

    def send(target, messages, schemas, *args, **kwargs):
        if target["provider"] == "claudexor":
            return remote(target, messages, schemas, *args, **kwargs)
        sent.append([dict(row) for row in messages])
        request_body = {"messages": deepcopy(messages), "tools": schemas or [], "model": "alternate"}
        request = _attempt_request(target, request_body)
        reply = _tools() if len(sent) == 1 else {"role": "assistant", "content": "Finished by the fallback"}
        usage = {"prompt_tokens": 512_206, "cached_tokens": 0, "cache_write_tokens": 512_203, "completion_tokens": 2,
                 "cost": 2.568527, "cost_final": True, "provider": "openai", "resolved_model": "alternate"}
        return ua.execute_physical_attempt(request, lambda: (reply, usage), extractor=lambda value: (value[1], 2.568527, True),
                                           before_dispatch=_candidate_before_dispatch(request_body, request))

    monkeypatch.setattr(ctx.llm, "_chat_remote", send)
    text, _usage, _trace = run_main(ctx, tools, events)
    assert text == "Finished by the fallback" and len(sent) == 2 and len(gateway.accepted_operations) == 2
    [facts] = _facts(sent[1])
    assert f"{API_FALLBACK} (role fallback:0) via openai answered this round after claudexor::codex=exact-model " \
           f"failed with subscription_window_exhausted (resource refusal quota) at " in facts
    assert f"(its engine dated a reset at {RESET})." in facts and "Recorded error text" in facts
    assert ("That answering call: cost $2.5685 reported, billing basis not reported; tokens input 512,206, "
            "cache read 0, cache write 512,203; tariff unknown.") in facts
    # The engine's Codex result: cash unknown and billing unknown, its $12.50 valuation is no charge.
    assert "The primary's own last accepted call in this task (round 1): cost unknown, engine billing unknown;" in facts
    assert "$12.5" not in facts and "(role main; account Auto)" in facts
    [seen] = _observed(sent[1])
    quota = seen["facts"]["quota"]
    assert quota["read"] == "ok" and quota["scope_honored"] is True
    assert quota["requested_scope"] == {"model": "exact-model", "requestKind": "text", "account": "Auto"}
    assert quota["availability"] == {"state": "exhausted", "resets_at": RESET, "execution_guaranteed": False,
                                     "scope": "quota_only"}
    assert [row["subject_id"] for row in quota["snapshots"]] == ["account-a", "account-b"]
    for private in ("Pro plan", "five hour window", "0.42", "99.0", "balances"):
        assert private not in json.dumps(seen), private
    [read] = [kwargs for name, kwargs in engine.calls if name == "quota"]
    assert read["model"] == "exact-model" and 0 < read["timeout_sec"] <= 3


# -- the round consumer: exact binding, aliases, failures and the shared budget ----------------


def _observe(monkeypatch, *, pin="", primary=PRIMARY):
    """Run the real round consumer once; return the facts row it appended (or None) and the catalog probes."""
    from ouroboros import loop, loop_model_call

    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": pin}))
    probes = []
    monkeypatch.setattr(llm_probe, "upstream_transport_reachable", lambda *a, **kw: probes.append(kw) or {})
    owner = SimpleNamespace(primary_route={"model": primary, "use_local": False, "role": "main"},
                            exact_model_route=False, task_metadata={}, _execution_trace={})
    monkeypatch.setattr(loop, "_dispatch_round_model", lambda *_a, **_k: ({"content": "accepted fallback"}, 0))
    monkeypatch.setattr(loop, "_measure_round_main_fit", lambda *_a, **_k: None)
    monkeypatch.setattr(loop_model_call, "_append_routing_receipts", lambda _ctx: False)
    monkeypatch.setattr(loop_model_call, "_project_wake_input", lambda _ctx: False)
    ctx = SimpleNamespace(tools=SimpleNamespace(_ctx=owner), llm=object(), context_fit_plan=None,
                          messages=[{"role": "user", "content": "go"}], defer_resource_wait=None, attempt_cap=None,
                          accumulated_usage={}, active_context_mode="max", active_model=FALLBACK,
                          active_use_local=False, tool_schemas=[])
    loop_model_call._call_round_model(ctx)
    rows = _observed(ctx.messages)
    return (rows[0]["facts"] if rows else None), probes, ctx


@pytest.mark.parametrize("pin,quota,roster,expected,subjects", [
    ("acct-a", _envelope(_snapshot("acct-a", state="exhausted", resets_at=RESET), _snapshot("acct-b")), None,
     {"state": "exhausted", "resets_at": RESET}, ["acct-a"]),
    ("", _envelope(_snapshot("acct-a", state="exhausted", resets_at=RESET), _snapshot("acct-b")), None,
     {"state": "available", "resets_at": None}, ["acct-a", "acct-b"]),
    # A fresh record beside an older stale same-subject source keeps the shared verdict unknown.
    ("", _envelope(_snapshot("acct-a"), _snapshot("acct-a", freshness="stale", source="codex_rollout")), None,
     {"state": "unknown", "reason": "quota_evidence_stale_or_incomplete"}, ["acct-a", "acct-a"]),
    # #244: a null subject is an unprofiled route's window, never `codex-default`'s, roster or not.
    ("codex-default", _envelope(_snapshot(None)), ROSTER, {"state": "unknown", "reason": "quota_evidence_missing"}, []),
    ("codex-default", _envelope(_snapshot(None), _snapshot("codex-default")), ROSTER,
     {"state": "available", "resets_at": None}, ["codex-default"]),
    ("", _envelope(), None, {"state": "unknown", "reason": "quota_evidence_missing"}, []),
    ("", _envelope(_snapshot("acct-a"), scoped=False), None,
     {"state": "unknown", "reason": "availability_scope_unhonored"}, ["acct-a"]),
], ids=["pin-never-borrows-sibling", "auto-sees-every-subject", "stale-source-keeps-unknown",
        "default-pin-never-borrows-unprofiled", "default-pin-exact", "empty-envelope", "scope-ignored"])
def test_the_observation_projects_the_shared_verdict_for_the_exact_binding(monkeypatch, pin, quota, roster,
                                                                           expected, subjects):
    engine = _Engine(quota, roster=roster)
    _connect(monkeypatch, engine)
    facts, probes, _ctx = _observe(monkeypatch, pin=pin)
    reported = facts["quota"]
    assert reported["read"] == "ok" and reported["requested_scope"]["account"] == (pin or "Auto")
    assert {key: reported["availability"].get(key) for key in expected} == expected
    assert reported["availability"]["scope"] == "quota_only" and reported["availability"].get("execution_guaranteed") is not True
    assert [row["subject_id"] for row in reported["snapshots"]] == subjects
    assert reported["scope_honored"] is ("availability_scope" in quota)
    assert [name for name, _ in engine.calls] == ["quota", "close"]  # no roster read guesses a subject
    assert len(probes) == 1  # the catalog check still ran, after the quota read, inside the same budget
    assert "Pro plan" not in json.dumps(facts) and "used_ratio" not in json.dumps(facts)


@pytest.mark.parametrize("pin", ["", "acct-07"], ids=["auto", "pin"])
def test_every_scoped_row_reaches_the_facts_beside_the_unchanged_shared_verdict(monkeypatch, pin):
    """#1628 r2: 43 matching rows, the size one real engine returned, all reach the facts compact and dated,
    none dropped behind a count; the verdict is the shared reducer's own, the requested scope and the one
    passive read are unchanged, and a pin still keeps only its own row."""
    from ouroboros.claudexor_observations import quota_availability

    rows = [_snapshot(f"acct-{index:02d}") for index in range(42)] + [_snapshot("acct-42", freshness="stale")]
    envelope = _envelope(*rows, absences=[_absence(f"gone-{index}") for index in range(7)])
    engine = _Engine(envelope)
    _connect(monkeypatch, engine)
    facts, probes, _ctx = _observe(monkeypatch, pin=pin)
    quota = facts["quota"]
    subjects = [f"acct-{index:02d}" for index in range(43)]
    assert [row["subject_id"] for row in quota["snapshots"]] == ([pin] if pin else subjects)
    assert [row["subject_id"] for row in quota["absences"]] == ([] if pin else [f"gone-{index}" for index in range(7)])
    assert "omitted" not in quota and all(set(row) == {"subject_id", "observed_at", "source", "freshness", "fresh_windows",
                                                      "windows", "state", "resets_at"} for row in quota["snapshots"])
    assert quota["availability"] == {**quota_availability(envelope, "codex", model="gpt-test", profile=pin,
                                                          request_kind="text"), "scope": "quota_only"}
    assert quota["requested_scope"] == {"model": "gpt-test", "requestKind": "text", "account": pin or "Auto"}
    assert quota["scope_honored"] is True and [name for name, _ in engine.calls] == ["quota", "close"]
    assert len(probes) == 1 and "Pro plan" not in json.dumps(facts) and "/secret/path" not in json.dumps(facts)


@pytest.mark.parametrize("connect,quota,expected", [
    (ClaudexorUnavailable("daemon_not_discovered", "none"), None, {"code": "daemon_not_discovered"}),
    (None, ClaudexorUnavailable("daemon_unreachable", "socket lost"), {"code": "daemon_unreachable"}),
    (None, ClaudexorUnavailable("daemon_busy", "busy", status_code=503), {"code": "read_refused", "status_code": 503}),
], ids=["no-engine", "quota-unreachable", "quota-503"])
def test_a_failed_read_stays_failed_never_an_empty_envelope(monkeypatch, connect, quota, expected):
    _connect(monkeypatch, connect or _Engine(quota))
    facts, probes, _ctx = _observe(monkeypatch, pin="acct-a")
    assert facts["quota"] == {"requested_scope": {"model": "gpt-test", "requestKind": "text", "account": "acct-a"},
                              "read": "failed", "error": expected}
    assert len(probes) == 1 and facts["kind"] == "unconfirmed"


def test_a_default_pin_keeps_its_exact_absence_and_never_asks_the_roster(monkeypatch):
    engine = _Engine(_envelope(_snapshot(None), absences=[_absence("codex-default")]),
                     roster=ClaudexorUnavailable("daemon_unreachable", "roster lost"))
    _connect(monkeypatch, engine)
    facts, probes, _ctx = _observe(monkeypatch, pin="codex-default")
    reported = facts["quota"]
    assert reported["read"] == "ok" and "roster_error" not in reported
    assert [name for name, _ in engine.calls] == ["quota", "close"]
    # The exact pin's own absence still counts; the null-subject row is not the pin's.
    assert reported["snapshots"] == [] and reported["subjects"] == "the exact pin"
    assert reported["absences"] == [{"subject_id": "codex-default", "observed_at": "2026-10-21T01:00:00Z",
                                     "reason": "not_logged_in"}]
    assert reported["availability"] == {"state": "unknown", "resets_at": None, "reason": "quota_evidence_missing",
                                        "scope": "quota_only"}
    assert "owner@example.com" not in json.dumps(facts) and "/secret/path" not in json.dumps(facts)
    assert len(probes) == 1


def test_legacy_views_and_the_catalog_share_one_three_second_budget(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    legacy = ClaudexorUnavailable("http_400", "unknown view", status_code=400)
    engine = _Engine(legacy, legacy, _envelope(_snapshot("acct-a"), scoped=False), clock=clock, step=0.5)
    connects = []
    _connect(monkeypatch, engine, connects, clock=clock, step=0.25)
    facts, probes, _ctx = _observe(monkeypatch)
    reads = [kwargs for name, kwargs in engine.calls if name == "quota"]
    assert [read.get("view") for read in reads] == ["resources_freshness", "constraint_freshness", None]
    assert connects[0]["timeout_sec"] == pytest.approx(3.0)
    assert [read["timeout_sec"] for read in reads] == pytest.approx([2.75, 2.25, 1.75])  # one shrinking budget
    assert probes[0]["timeout"] == pytest.approx(1.25)  # the catalog gets only what is left
    assert facts["quota"]["read"] == "ok" and facts["quota"]["scope_honored"] is False


def test_a_spent_budget_reads_no_more_and_never_calls_the_catalog(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    legacy = ClaudexorUnavailable("http_400", "unknown view", status_code=400)
    engine = _Engine(legacy, _envelope(_snapshot("acct-a")), clock=clock, step=1.6)
    _connect(monkeypatch, engine, clock=clock, step=1.6)
    facts, probes, _ctx = _observe(monkeypatch)
    assert [name for name, _ in engine.calls] == ["quota", "close"]  # no legacy view after the budget is spent
    assert facts["quota"] == {"requested_scope": {"model": "gpt-test", "requestKind": "text", "account": "Auto"},
                              "read": "not_read", "reason": "observation_deadline_elapsed"}
    assert probes == [] and facts["kind"] == "unconfirmed"


def test_a_stop_after_the_quota_read_ends_the_observation_before_the_catalog(monkeypatch):
    from ouroboros import loop_transport

    engine = _Engine(_envelope(_snapshot("acct-a")))
    _connect(monkeypatch, engine)
    asked = []

    def stop_requested(*_args, **_kwargs):
        asked.append(1)
        return len(asked) > 1  # the owner's Stop lands while the quota read is in flight

    monkeypatch.setattr(loop_transport, "transport_repeat_stop_requested", stop_requested)
    facts, probes, ctx = _observe(monkeypatch)
    assert facts is None and probes == [] and [name for name, _ in engine.calls] == ["quota", "close"]
    assert ctx.tools._ctx._execution_trace == {} and len(ctx.messages) == 1  # nothing appended or recorded


def test_a_default_pin_observes_only_its_exact_subject_without_a_roster_read(monkeypatch):
    """#244: a fresh null-subject window (an unprofiled API-key route) is not `codex-default`'s,
    even when the roster lists that profile; nothing reads the roster to guess it."""
    engine = _Engine(_envelope(_snapshot(None, state="exhausted", resets_at=RESET)), roster=ROSTER)
    _connect(monkeypatch, engine)
    facts, _probes, _ctx = _observe(monkeypatch, pin="codex-default")
    reported = facts["quota"]
    assert reported["read"] == "ok" and reported["snapshots"] == []
    assert reported["availability"] == {"state": "unknown", "resets_at": None,
                                        "reason": "quota_evidence_missing", "scope": "quota_only"}
    assert reported["subjects"] == "the exact pin" and "roster_error" not in reported
    assert [name for name, _ in engine.calls] == ["quota", "close"]
