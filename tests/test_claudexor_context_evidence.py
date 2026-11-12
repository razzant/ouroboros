"""Context capacity belongs to an exact subscription account and role."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from ouroboros import capability_evidence as ce


MODEL = "claudexor::test-source=exact-model"


def _catalog(profile="account-a", fingerprint="identity-a", window=872_000):
    return {
        "source": "test-source", "credentialProfileId": profile,
        "accountFingerprint": fingerprint, "observedAt": ce.utc_now_iso(),
        "provenance": "exact raw transport catalog",
        "models": [{"id": "exact-model", "contextWindow": 272_000,
                    "maxContextWindow": window, "maxOutputTokens": None,
                    "effectiveContextWindow": 250_240, "compactionThreshold": 95}],
    }


@pytest.fixture
def catalog(monkeypatch):
    from ouroboros.llm import LLMClient

    current = {"value": _catalog(), "calls": []}

    def fetch(source, credential_profile_id=None, *, requested_model=None):
        current["calls"].append((source, credential_profile_id, requested_model))
        return copy.deepcopy(current["value"])

    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(fetch), raising=False)
    return current


def _probe(root, **kwargs):
    return ce.probe(root, provider="claudexor", model=MODEL, **kwargs)


def _binding(evidence):
    return {"source": evidence.source_id, "model": "exact-model",
            "credentialProfileId": evidence.credential_profile_id,
            "accountFingerprint": evidence.account_fingerprint}


@pytest.mark.parametrize("changed", [
    {"source_id": "other-source"},
    {"credential_profile_id": "account-b"},
    {"account_fingerprint": "new-account-identity"},
])
def test_subscription_capacity_identity_includes_account_binding(changed):
    options = {
        "source_id": "test-source",
        "credential_profile_id": "account-a",
        "account_fingerprint": "account-identity-a",
    }
    route = {"provider": "claudexor", "model": "claudexor::test-source=exact-model"}
    first = ce.route_fingerprint(**route, options=options)
    assert first != ce.route_fingerprint(**route, options={**options, **changed})
    # A renewed bearer credential does not change the account's capacity identity.
    assert first == ce.route_fingerprint(**route, options=options,
                                         headers={"Authorization": "Bearer test-placeholder"})


def test_auto_uses_largest_advertised_capacity_and_retains_provenance(tmp_path, catalog):
    evidence = _probe(tmp_path)
    assert evidence.window_tokens == 872_000
    assert evidence.status == ce.STATUS_CONFIRMED
    assert evidence.source == ce.SOURCE_PROVIDER_METADATA
    assert evidence.credential_profile_id == "account-a"
    assert evidence.account_fingerprint == "identity-a"
    assert evidence.provenance == catalog["value"]["provenance"]
    assert evidence.ts == catalog["value"]["observedAt"]
    assert catalog["calls"] == [("test-source", None, "exact-model")]


def test_auto_capacity_uses_model_compatible_account_and_keeps_exact_binding(tmp_path, monkeypatch):
    from ouroboros.llm import LLMClient

    def catalog(source, credential_profile_id=None, *, requested_model=None):
        assert (source, credential_profile_id, requested_model) == ("test-source", None, "exact-model")
        return _catalog("account-b", "identity-b", 500_000)

    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(catalog))
    evidence = _probe(tmp_path)
    assert evidence.window_tokens == 500_000
    assert evidence.credential_profile_id == "account-b"
    assert evidence.account_fingerprint == "identity-b"
    wrong = _probe(tmp_path, allow_fetch=False, options={"source_id": "test-source",
                   "credential_profile_id": "account-a", "account_fingerprint": "identity-a"})
    assert wrong.window_tokens == 0


def test_auto_does_not_reuse_last_discovered_accounts_capacity(tmp_path, catalog):
    first = _probe(tmp_path)
    catalog["value"] = _catalog("account-b", "identity-b", 500_000)
    second = _probe(tmp_path)
    assert second.window_tokens == 500_000
    assert first.route_fp != second.route_fp
    assert second.credential_profile_id == "account-b"
    assert len(catalog["calls"]) == 2


def test_an_auto_start_reads_the_catalog_once_and_rewrites_nothing_it_already_holds(tmp_path, catalog):
    """Every Auto start reads the engine's catalog live, exactly once: that read answers the window and the output
    limit even when the output record is past its ten minutes, the same statement read again is not written again,
    and another advertisement is what the very next start plans by."""
    from ouroboros import context_timing

    def start(**kwargs):
        with context_timing.measured_sections():
            return _probe(tmp_path, **kwargs), context_timing.section_spans()["work"]

    hours_ago = (ce.utc_now() - __import__("datetime").timedelta(hours=2)).isoformat()
    catalog["value"]["observedAt"] = hours_ago  # no maximum and an output record already past its ten minutes
    first, work = start()
    assert (first.window_tokens, first.credential_profile_id, len(catalog["calls"])) == (872_000, "account-a", 1)
    assert work == {"catalog_calls": 1, "evidence_writes": 2}  # the probe and the output limit; no second read
    again, work = start()
    assert (again.to_json(), len(catalog["calls"]), work) == (first.to_json(), 2, {"catalog_calls": 1})
    catalog["value"] = _catalog("account-b", "identity-b", 500_000)
    changed, work = start()
    assert (changed.window_tokens, changed.credential_profile_id, len(catalog["calls"])) == (500_000, "account-b", 3)
    assert work["catalog_calls"] == 1 and "catalog_calls_again" not in work


def test_one_catalog_read_serves_window_and_output_and_a_failed_read_is_not_repeated(tmp_path, monkeypatch):
    """A probe reads the engine's catalog once. When that read fails (now, or its ten-minute record), the output limit
    names that failed read instead of waiting on a second read of the same daemon; a read that answered without an
    account's window answers the output limit too, so a nameless answer is read once per probe."""
    from ouroboros.llm import LLMClient

    reads = []

    def unreachable(source, credential_profile_id=None, *, requested_model=None):
        reads.append("timed out")
        raise TimeoutError("controlled daemon timeout")
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(unreachable))
    failed = _probe(tmp_path)
    assert failed.status == ce.STATUS_FAILED and reads == ["timed out"]
    assert failed.response_limit["source"] == "claudexor metadata: metadata read failed"  # its provenance, not none
    assert _probe(tmp_path).status == ce.STATUS_FAILED and reads == ["timed out"]  # the failure's record: no read
    nameless = {**_catalog(), "accountFingerprint": ""}  # answered, but names no account to hold a window
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog",
                        staticmethod(lambda *_a, **_kw: reads.append("answered") or copy.deepcopy(nameless)))
    (tmp_path / "fresh").mkdir()
    assert _probe(tmp_path / "fresh").status == ce.STATUS_UNPROBEABLE and reads[1:] == ["answered"]


@pytest.mark.parametrize("output", ["expired", "own_fresh"])
def test_a_stale_record_kept_after_a_failed_read_starts_no_second_read_for_output(tmp_path, catalog, monkeypatch, output):
    """H3 (W2, 12.11.2026): an exact account record past its 24 h, kept stale because its refresh failed, carries that
    failure as a fact of the attempt: the output limit does not read the catalog a second time. Window and output keep
    their own dates: an expired maximum cools down without a renewed date, an output observed fresh on its own stays."""
    import datetime
    from ouroboros import response_limits
    from ouroboros.llm import LLMClient

    exact = {"source_id": "test-source", "credential_profile_id": "account-a", "account_fingerprint": "identity-a"}
    day_ago = (ce.utc_now() - datetime.timedelta(hours=30)).isoformat()
    catalog["value"].update(observedAt=day_ago, models=[{**catalog["value"]["models"][0], "maxOutputTokens": 64_000}])
    assert _probe(tmp_path).stale  # the account's window and maximum, both recorded 30 h ago
    if output == "own_fresh":
        response_limits.record_metadata_limit(tmp_path, provider="claudexor", model=MODEL, options=exact,
                                              maximum=32_000, observed_at=ce.utc_now_iso())
    reads = []
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(
        lambda *_a, **_kw: reads.append(1) or (_ for _ in ()).throw(TimeoutError("controlled daemon timeout"))))
    kept = _probe(tmp_path, options=exact)
    assert reads == [1] and kept.stale and kept.window_tokens == 872_000 and kept.ts == day_ago  # one read, kept as it was
    assert ce._read(tmp_path)["probes"][kept.route_fp]["ts"] == day_ago  # no new FAILED row with a new date
    limit = kept.response_limit
    if output == "own_fresh":
        assert (limit["max_output_tokens"], limit["stale"]) == (32_000, False)
    else:
        assert (limit["max_output_tokens"], limit["stale"], limit["observed_at"]) == (64_000, True, day_ago)
        assert kept.route_fp and ce._read(tmp_path)["response_limit_refresh_failures"]  # cools down, date kept


def test_a_failed_auto_catalog_read_plans_by_the_pools_smallest_fresh_statement(tmp_path, catalog, monkeypatch):
    """Window B (owner, 07.11.2026): when the engine's catalog does not answer, an unpinned Auto plan is bounded by the
    smallest window this model's accounts stated within the confirmed TTL, labelled ``pool`` and dated: a bound of one
    plan, no account witness and no server statement, while the store keeps the failure. A pin (even without its
    fingerprint), a stale or another model's statement, a catalog that answered without a window and a reader that
    may not fetch leave the window unknown."""
    from ouroboros.context_window import resolve_window
    from ouroboros.llm import LLMClient

    for account, window in (("account-a", 872_000), ("account-b", 500_000)):  # two Auto reads advertised two accounts
        catalog["value"] = _catalog(account, f"identity-{account[-1]}", window)
        assert _probe(tmp_path).window_tokens == window
    reads = []
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(
        lambda *_a, **_kw: reads.append(1) or (_ for _ in ()).throw(TimeoutError("controlled daemon timeout"))))
    bound = _probe(tmp_path)
    assert (bound.window_tokens, bound.status, bound.source) == (500_000, ce.STATUS_CONFIRMED, "pool") and reads == [1]
    assert not (bound.account_fingerprint or bound.credential_profile_id or bound.server_window)
    assert bound.ts == max(row["ts"] for row in ce._read(tmp_path)["probes"].values() if row["status"] == "confirmed")
    assert bound.detail.startswith("smallest advertised capacity of 2 accounts, newest stated 20")
    assert ce._read(tmp_path)["probes"][bound.route_fp]["status"] == ce.STATUS_FAILED  # only the failure is stored
    assert resolve_window(bound, 65_536, bound=False).usable == 500_000 - 65_536
    assert _probe(tmp_path).to_json() == bound.to_json() and reads == [1]  # its ten-minute record reads nothing
    assert _probe(tmp_path, allow_fetch=False).status == ce.STATUS_FAILED  # a reader that may not fetch: unknown
    pinned = _probe(tmp_path, options={"source_id": "test-source", "credential_profile_id": "account-a"})
    # A pin is no Auto: never the pool's bound, only its own account's fresh record (H6, the fingerprint test below).
    assert (pinned.window_tokens, pinned.account_fingerprint, pinned.source) == (872_000, "identity-a", "provider_metadata")
    other = ce.probe(tmp_path, provider="claudexor", model=MODEL + "-other")
    assert other.window_tokens == 0 and other.status == ce.STATUS_FAILED  # another model's accounts state nothing here
    data = ce._read(tmp_path)
    for row in data["probes"].values():
        row["ts"] = "2020-01-01T00:00:00Z"
    monkeypatch.setattr(ce, "_read", lambda _root: data)
    assert not ce._pool_bound(tmp_path, bound, {"provider": "claudexor", "model": MODEL})  # stale statements: unknown
    monkeypatch.undo()
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    assert _probe(fresh, allow_fetch=False).window_tokens == 0  # not probed: unknown, no pool


def _resolve(monkeypatch, root, *, model_route=None, pin=None):
    """The fit's and every rebind's route question, through the production route builder."""
    from ouroboros import config, context_fit as cf, server_runtime

    monkeypatch.setattr(config, "runtime_settings", lambda: {"OUROBOROS_MODEL": MODEL})
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(server_runtime, "apply_runtime_provider_defaults", lambda s: (s, False, []))
    return cf.resolve_context_fit_route({"model": MODEL, "model_role": "main", "task_metadata": {},
                                         "model_route": model_route or {}, "credential_profile_id": pin}, allow_fetch=True)


def _answered(route):
    """Restored messages whose last answer of this model names ``route`` (the conversation's account)."""
    return [{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "nativeContinuation": {"route": route}}]


@pytest.mark.parametrize("pin", [None, "account-a"])
def test_a_resumed_turn_reads_the_conversations_fresh_record_and_asks_only_its_own_question(tmp_path, monkeypatch,
                                                                                         catalog, pin):
    """H2 (W2, 12.11.2026): a resumed turn binds the account its conversation names (``owner_wait._conversation_account``).
    That account's fresh record answers with zero questions; a missing or expired record, or an identity the record does
    not match, asks exactly one: the route's own question (Auto stays Auto, a pin the pin), never the preference as a pin."""
    import datetime
    from types import SimpleNamespace
    from ouroboros.owner_wait import _conversation_account

    seeded = _resolve(monkeypatch, tmp_path, pin=pin)[1]  # the build's own question recorded account-a
    assert len(catalog["calls"]) == 1 and seeded.account_fingerprint == "identity-a"
    ctx = SimpleNamespace(active_model=MODEL, active_use_local=False, context_fit_plan=None)

    def resumed(route):
        identity = _conversation_account(ctx, _answered(route), "main", pin)
        return _resolve(monkeypatch, tmp_path, model_route=identity, pin=pin)[1]

    same = resumed({**_binding(seeded), "credentialProfileId": "account-a"})
    assert len(catalog["calls"]) == 1 and same.to_json() == seeded.to_json()  # fresh record: no question
    relogged = resumed({**_binding(seeded), "accountFingerprint": "identity-new"})
    assert catalog["calls"][1:] == [("test-source", pin, "exact-model")]  # changed identity: one, the route's own
    assert relogged.account_fingerprint == "identity-a"  # the engine's answer, not a name matched to a record
    later = ce.utc_now() + datetime.timedelta(hours=25)
    monkeypatch.setattr(ce, "utc_now", lambda: later)
    catalog["value"]["observedAt"] = later.isoformat()
    expired = resumed(_binding(seeded))
    assert catalog["calls"][2:] == [("test-source", pin, "exact-model")] and not expired.stale  # expired: exactly one


def test_the_rebind_after_tool_reconciliation_reuses_the_first_answer(tmp_path, monkeypatch, catalog):
    """A restore rebinds twice when its tools changed. The first binds the conversation's account (account-a, no record
    here, so the route's own question is asked, and the engine answers account-b); the second binds that first answer,
    so it reads account-b's record instead of asking again."""
    from types import SimpleNamespace
    from ouroboros import continuation_tools, loop, owner_wait

    catalog["value"] = _catalog("account-b", "identity-b")
    conversation = {"source": "test-source", "model": "exact-model", "credentialProfileId": "account-a",
                    "accountFingerprint": "identity-a"}
    calls = []

    def rebind(plan, _tools, _messages, **route):
        calls.append(route["model_route"])
        evidence = _resolve(monkeypatch, tmp_path, model_route=route["model_route"])[1]
        return SimpleNamespace(model_route=_binding(evidence) if evidence.account_fingerprint else {}), "max"

    monkeypatch.setattr(loop, "_rebind_context_fit_plan", rebind)
    monkeypatch.setattr(continuation_tools, "reconcile_restored_tools", lambda *_a: True)
    monkeypatch.setattr(owner_wait, "sanction_rewrite", lambda *_a, **_kw: None)
    ctx = SimpleNamespace(context_fit_plan=None, active_model=MODEL, active_use_local=False, task_metadata={},
                          model_wait_context=None, active_context_mode="max")
    owner_wait.rebind_restored_route(SimpleNamespace(_ctx=ctx), {"tool_schemas": []}, _answered(conversation))
    assert calls[0] == conversation and calls[1]["accountFingerprint"] == "identity-b"
    assert catalog["calls"] == [("test-source", None, "exact-model")]  # once (Auto, no hidden pin), not twice


def test_a_pinned_route_with_a_silent_engine_keeps_its_window_only_with_the_same_fingerprint(tmp_path, monkeypatch,
                                                                                             catalog):
    """H6 (W2, 12.11.2026): a pinned route whose catalog read failed plans by the fresh record of the same profile AND
    the same account fingerprint (its own date, nothing stored), never by another account behind that profile name."""
    from ouroboros.llm import LLMClient

    pinned = {"source_id": "test-source", "credential_profile_id": "account-a"}
    seeded = _probe(tmp_path, options=pinned)
    answered = dict(catalog["value"])
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(
        lambda *_a, **_kw: (_ for _ in ()).throw(TimeoutError("controlled daemon timeout"))))
    kept = _probe(tmp_path, options=pinned)
    assert (kept.window_tokens, kept.account_fingerprint, kept.ts, kept.stale) == (872_000, "identity-a", seeded.ts, False)
    assert ce._read(tmp_path)["probes"][ce.route_fingerprint(provider="claudexor", model=MODEL, options=pinned)][
        "status"] == ce.STATUS_FAILED  # the failure is what the store keeps for the pin
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(  # the profile now logs into another account
        lambda *_a, **_kw: {**answered, "accountFingerprint": "identity-b", "models": [], "observedAt": ce.utc_now_iso()}))
    _probe(tmp_path, options={**pinned, "account_fingerprint": "identity-b"})  # observed: identity-b, no window stated
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(
        lambda *_a, **_kw: (_ for _ in ()).throw(TimeoutError("controlled daemon timeout"))))
    other = _probe(tmp_path, options=pinned)
    assert other.window_tokens == 0 and not ce.is_known(other)  # account-a's window is not this profile's any more


TASK = {"model": MODEL, "model_role": "main", "task_metadata": {}}


def _built(monkeypatch, root, resolver_for, task=TASK):
    """One start fit of a small core on the production route builder, its receipt window open as in a build; the
    resolver (``resolver_for(task)``) is made inside that window, as the agent makes its handle."""
    from ouroboros import config, context_fit as cf, context_timing, server_runtime

    monkeypatch.setattr(config, "runtime_settings", lambda: {"OUROBOROS_MODEL": MODEL})
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(server_runtime, "apply_runtime_provider_defaults", lambda s: (s, False, []))
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: root)
    core = cf.ContextCore("Core governance. " * 4000, "", "", "", "", "", '"Assignment"', False)
    with context_timing.measured_sections():
        plan = cf.build_context_fit_plan(None, core, task, preferred_mode="max", route_resolver=resolver_for(task))
        return plan, context_timing.section_spans()


@pytest.mark.parametrize("window", [872_000, 70_000])
def test_the_fits_route_question_runs_beside_the_build_and_repeats_only_before_a_narrowing(tmp_path, monkeypatch,
                                                                                         catalog, window):
    """H4 (W2, 12.11.2026): the fit's route question starts with the build and runs beside it, the same question
    through the same resolver; with or without that handle the plan is the same. An early answer that would show less
    than the selected view is asked again at the usual point and that answer is used; one that fits is not repeated."""
    from ouroboros import context_fit as cf

    catalog["value"]["models"][0].update(contextWindow=window, maxContextWindow=window)
    plain, _ = _built(monkeypatch, tmp_path / "plain", lambda _task: cf.resolve_context_fit_route)
    narrowed = plain.initial_mode != "max"
    assert narrowed is (window == 70_000) and len(catalog["calls"]) == 1
    early, spans = _built(monkeypatch, tmp_path / "early", lambda task: cf.RouteQuestion(task, cf.resolve_context_fit_route))
    assert len(catalog["calls"]) == (3 if narrowed else 2)  # the early question, and its repeat only before a narrowing
    assert spans["work"]["catalog_calls"] == (2 if narrowed else 1)  # counted on the build's own receipt
    assert ("fit.route_wait" in spans["sections_ms"]) and (("fit.route" in spans["sections_ms"]) is narrowed)
    for name in ("initial_mode", "window_tokens", "route_fp", "model_route", "start", "core_sha256"):
        assert getattr(early, name) == getattr(plain, name), name
    assert early.max_projection.system_content_json == plain.max_projection.system_content_json


def test_an_early_answer_that_would_narrow_memory_yields_to_the_usual_points_answer(tmp_path, monkeypatch):
    """The horizon guard: the pool answered the early question with a smaller account (Low here); the question asked
    again at the usual point names a larger one, and the plan is that answer's: the mind sees no less than today."""
    from ouroboros import context_fit as cf
    from ouroboros.llm import LLMClient

    answers = [_catalog("account-a", "identity-a", 70_000), _catalog("account-b", "identity-b", 872_000)]
    for value in answers:
        value["models"][0]["contextWindow"] = value["models"][0]["maxContextWindow"]
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(lambda *_a, **_kw: copy.deepcopy(answers.pop(0))))
    plan, spans = _built(monkeypatch, tmp_path, lambda task: cf.RouteQuestion(task, cf.resolve_context_fit_route))
    assert not answers and spans["work"]["catalog_calls"] == 2
    assert (plan.initial_mode, plan.model_route["credentialProfileId"]) == ("max", "account-b")


def test_the_memory_view_receipt_names_the_kind_and_age_of_its_window_evidence(tmp_path, monkeypatch, catalog):
    """W2 H7: ``context_memory_view`` says whether the start's window was read live, recorded, a failed Auto read's pool
    bound or unknown, and how old that statement is."""
    import json
    from types import SimpleNamespace
    from ouroboros import context, context_fit as cf
    from ouroboros.llm import LLMClient

    live, _ = _built(monkeypatch, tmp_path, lambda _task: cf.resolve_context_fit_route)
    identity = _binding(ce.probe(tmp_path, provider="claudexor", model=MODEL, allow_fetch=False, options={
        "source_id": "test-source", "credential_profile_id": "account-a", "account_fingerprint": "identity-a"}))
    recorded, _ = _built(monkeypatch, tmp_path, lambda _task: cf.resolve_context_fit_route, task={**TASK, "model_route": identity})
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(
        lambda *_a, **_kw: (_ for _ in ()).throw(TimeoutError("controlled daemon timeout"))))
    pool, _ = _built(monkeypatch, tmp_path, lambda _task: cf.resolve_context_fit_route)
    unknown, _ = _built(monkeypatch, tmp_path / "cold", lambda _task: cf.resolve_context_fit_route)
    assert [plan.window_evidence[0] for plan in (live, recorded, pool, unknown)] == ["live", "recorded", "pool", "unknown"]
    plan = SimpleNamespace(initial_mode="max", window_evidence=live.window_evidence,
                           projection=lambda _mode: SimpleNamespace(memory_facts={"role": "integrator"}))
    context._record_memory_view(None, SimpleNamespace(drive_root=tmp_path), {"id": "t", "budget_drive_root": str(tmp_path)},
                                None, plan)
    row = json.loads((tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["window_evidence"]["kind"] == "live" and 0 <= row["window_evidence"]["age_s"] < 3600


def test_a_changed_route_voids_the_early_answer_and_asks_its_own_question(tmp_path, monkeypatch, catalog):
    """One handle per route: the build whose route changed (here the role's pin) after the question started asks its
    own question; the early answer is never applied to it, and the account that just failed is never spent by it."""
    from ouroboros import context_fit as cf, llm_substitution

    token = llm_substitution._FAILED_PROFILE.set(("cache", "test-source", "exact-model", "account-a"))
    try:
        plan, _ = _built(monkeypatch, tmp_path, lambda task: cf.RouteQuestion(TASK, cf.resolve_context_fit_route),
                         task={**TASK, "credential_profile_id": "account-a"})
        assert llm_substitution._FAILED_PROFILE.get() == ("cache", "test-source", "exact-model", "account-a")
    finally:
        llm_substitution._FAILED_PROFILE.reset(token)
    assert sorted(catalog["calls"], key=str) == [("test-source", "account-a", "exact-model"),
                                                  ("test-source", None, "exact-model")]
    assert plan.account_pin == "account-a" and plan.model_route["credentialProfileId"] == "account-a"


def test_exact_observed_binding_reuses_only_its_cache(tmp_path, catalog):
    first = _probe(tmp_path)
    options = ce.model_account_options(MODEL, model_route=_binding(first))
    cached = _probe(tmp_path, options=options, allow_fetch=False)
    assert cached.to_json() == first.to_json()
    assert len(catalog["calls"]) == 1
    for change in ({"credential_profile_id": "account-b"},
                   {"account_fingerprint": "reauthorized-identity"}):
        other = _probe(tmp_path, options={**options, **change}, allow_fetch=False)
        assert other.window_tokens == 0
        assert not ce.is_known(other)


@pytest.mark.parametrize("change", [
    {"source": "wrong-source"}, {"credentialProfileId": "wrong-profile"},
    {"accountFingerprint": "wrong-identity"},
])
def test_mismatched_catalog_cannot_supply_requested_account_capacity(tmp_path, catalog, change):
    catalog["value"].update(change)
    evidence = _probe(tmp_path, options={"source_id": "test-source",
                      "credential_profile_id": "account-a", "account_fingerprint": "identity-a"})
    assert evidence.window_tokens == 0
    assert evidence.status == ce.STATUS_FAILED


@pytest.mark.parametrize("change", [
    {"accountFingerprint": None}, {"models": []},
    {"models": [{"id": "exact-model", "contextWindow": None, "maxContextWindow": None,
                 "effectiveContextWindow": 250_240}]},
])
def test_missing_capacity_or_binding_remains_unknown_without_generation(tmp_path, catalog, change):
    catalog["value"].update(change)
    evidence = _probe(tmp_path)
    assert evidence.window_tokens == 0
    assert not ce.is_known(evidence, require_fresh=True)


def test_catalog_observation_time_is_not_refreshed_by_local_cache(tmp_path, catalog):
    catalog["value"]["observedAt"] = "2020-01-01T00:00:00Z"
    evidence = _probe(tmp_path)
    assert evidence.window_tokens == 872_000
    assert evidence.stale
    assert not ce.is_known(evidence, require_fresh=True)


def test_an_older_answer_never_replaces_a_later_statement_and_a_future_date_is_not_fresh(tmp_path, catalog):
    """H8 (W2, 12.11.2026): two reads of one account that finish out of order keep the later statement in the
    store; a statement dated after now proves no freshness (every TTL reads it as expired) and decides no race."""
    import datetime

    now = ce.utc_now()

    def read(hours, window):
        catalog["value"].update(observedAt=(now + datetime.timedelta(hours=hours)).isoformat(),
                                models=[{**catalog["value"]["models"][0], "maxContextWindow": window}])
        return _probe(tmp_path)

    def stored():  # one account, one key
        return ce._read(tmp_path)["probes"][later.route_fp]

    later = read(-1, 500_000)
    assert read(-2, 872_000).window_tokens == 872_000  # the late answer is its own read's evidence ...
    assert (stored()["window_tokens"], stored()["ts"]) == (500_000, later.ts)  # ... but never replaces the later one
    assert read(0, 872_000).window_tokens == stored()["window_tokens"] == 872_000  # a later statement does
    exact = {"source_id": "test-source", "credential_profile_id": "account-a", "account_fingerprint": "identity-a"}
    assert ce.is_known(_probe(tmp_path, options=exact, allow_fetch=False), require_fresh=True)
    future = read(1, 990_000)
    assert future.stale and not ce.is_known(future, require_fresh=True)  # observed "in an hour": not fresh now
    assert ce._age_seconds(future.ts) == float("inf") and ce._age_seconds(later.ts) < 2 * 3600
    assert not ce.is_known(_probe(tmp_path, options=exact, allow_fetch=False), require_fresh=True)
    assert read(-0.5, 872_000).window_tokens == stored()["window_tokens"] == 872_000  # a future row wins no race


def test_observed_route_must_match_model_and_current_role_pin():
    observed = _binding(ce.CapabilityEvidence(
        872_000, "confirmed", "provider_metadata", "fp", source_id="test-source",
        credential_profile_id="account-a", account_fingerprint="identity-a"))
    settings = {"OUROBOROS_MODEL_ACCOUNTS": {"main": "account-b", "light": "account-a"}}
    main = ce.model_account_options(MODEL, role="main", settings=settings, model_route=observed)
    light = ce.model_account_options(MODEL, role="light", settings=settings, model_route=observed)
    assert main == {"source_id": "test-source", "credential_profile_id": "account-b"}
    assert light["account_fingerprint"] == "identity-a"
    changed = ce.model_account_options(MODEL + "-new", role="light", settings=settings,
                                       model_route=observed)
    assert "account_fingerprint" not in changed
    nameless = {**observed, "accountFingerprint": None}
    assert "account_fingerprint" not in ce.model_account_options(MODEL, model_route=nameless)


def test_role_manual_window_above_catalog_is_asserted_not_scope_ack(tmp_path, monkeypatch, catalog):
    from ouroboros import config
    from ouroboros.context_fit import resolve_context_fit_route

    settings = {"OUROBOROS_MODEL": MODEL,
                "OUROBOROS_MODEL_ACCOUNTS": {"main": "account-a", "light": "account-b"},
                "OUROBOROS_MODEL_CONTEXT_WINDOWS": {"main": 1_200_000, "fallback": [900_000]}}
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    main_route, main = resolve_context_fit_route({"model": MODEL, "model_role": "main"}, allow_fetch=True)
    assert main.window_tokens == 1_200_000 and main.status == "asserted"
    assert main.source == "user_setting"
    assert ce.is_known(main)  # sizing assertion, not governance proof
    assert main_route["options"]["credential_profile_id"] == "account-a"
    assert ce.list_owner_acks(tmp_path) == []
    catalog["value"] = _catalog("account-b", "identity-b", 500_000)
    light_route, light = resolve_context_fit_route({"model": MODEL, "model_role": "light"}, allow_fetch=True)
    assert light.window_tokens == 500_000 and light.source == "provider_metadata"
    assert light_route["options"]["credential_profile_id"] == "account-b"
    assert light.route_fp != main.route_fp
    catalog["value"] = _catalog()
    _, fallback = resolve_context_fit_route({"model": MODEL, "model_role": "fallback:0"}, allow_fetch=True)
    assert fallback.window_tokens == 900_000 and fallback.source == "user_setting"


def test_manual_reviewer_sizing_does_not_change_scope_authority(tmp_path, monkeypatch, catalog):
    from ouroboros import config
    from ouroboros.reviewer_window import resolve_reviewer_window

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setenv("OUROBOROS_MODEL_CONTEXT_WINDOWS", json.dumps({"deep_review": 1_200_000}))
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", json.dumps({"deep_review": "account-a"}))
    window = resolve_reviewer_window(MODEL, model_role="deep_review", use_local=False)
    assert window.window_tokens == 1_200_000 and window.window_record.contradicted_by.tokens == 872_000
    assert window.sizing_window() == 1_200_000
    assert window.sizing_source == "user_setting"
    assert window.model_route["credentialProfileId"] == "account-a"
    assert ce.list_owner_acks(tmp_path) == []


def test_main_input_reserve_and_account_capacity_remain_distinct(tmp_path, monkeypatch, catalog):
    from ouroboros import config
    from ouroboros.context_fit import measure_main_fit
    from tests.test_context_fit_v664 import _plan

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    evidence = _probe(tmp_path)
    plan = replace(_plan(), model=MODEL, provider="claudexor", route_fp=evidence.route_fp,
                   window_tokens=evidence.window_tokens, status=evidence.status,
                   model_route=_binding(evidence))
    measured = measure_main_fit(plan, plan.messages_for("max"), [],
                                profile="owner_max", rendered_mode="max", round_id="r1").measurement
    assert measured.capacity_total_tokens == 872_000
    assert measured.response_reserve_tokens == 65_536
    assert 0 < measured.estimated_input_tokens < measured.response_reserve_tokens
    assert measured.target_total_tokens is None


def test_unknown_subscription_reviewer_keeps_the_assembler_input_budget(tmp_path, monkeypatch, catalog):
    from ouroboros import config
    from ouroboros.reviewer_window import resolve_reviewer_window, window_scaled_reserves
    from ouroboros.tools.review_helpers import REVIEW_PROMPT_TOKEN_BUDGET, calibrated_input_token_limit
    from ouroboros.tools.review_synthesis import per_slot_input_token_limits

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    catalog["value"]["models"] = []
    window = resolve_reviewer_window(MODEL, use_local=False)
    assert window.window_tokens == 0
    assert window.sizing_window() == 0
    assert window.window_record.usable is None and not ce.is_known(window)
    assert window_scaled_reserves(0, output_reserve=65_536, tokenizer_margin=50_000) == (65_536, 50_000)
    assert calibrated_input_token_limit(MODEL, context_window=window.sizing_window(), output_reserve=65_536,
                                         tokenizer_margin=50_000, budget_cap=456_789) == 456_789
    caps = per_slot_input_token_limits([MODEL], output_reserve=65_536, tokenizer_margin=50_000)
    assert caps == {MODEL: REVIEW_PROMPT_TOKEN_BUDGET}


def test_unknown_subscription_native_layout_ignores_retired_character_cap(tmp_path, monkeypatch, catalog):
    from ouroboros import config
    from ouroboros.review_native_episode import review_native_transcript_bound
    from ouroboros.reviewer_window import resolve_reviewer_window

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    catalog["value"]["models"] = []
    monkeypatch.delenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", raising=False)
    hint = review_native_transcript_bound(MODEL, output_reserve=65_536, use_local=False)
    for stored in ("900000", "765432", "1000"):
        monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", stored)
        assert review_native_transcript_bound(MODEL, output_reserve=65_536, use_local=False) == hint
    assert resolve_reviewer_window(MODEL, use_local=False).window_tokens == 0


def test_unknown_subscription_scope_stays_unknown_without_authority(tmp_path, monkeypatch, catalog):
    from ouroboros import config
    from ouroboros.tools.scope_window import scope_window, scope_window_provenance, window_provenance_phrase

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    catalog["value"]["models"] = []
    window = scope_window(MODEL)
    assert window.window_tokens == 0
    assert window_provenance_phrase(window.window_tokens, scope_window_provenance(window)) == "unknown window"


def test_model_only_ack_and_cache_cannot_certify_subscription_account(tmp_path, catalog):
    fp = ce.route_fingerprint(provider="claudexor", model=MODEL)
    with pytest.raises(ValueError, match="exact account binding"):
        ce.record_owner_ack(tmp_path, provider="claudexor", model=MODEL, window_tokens=1_200_000)
    # A legacy unbound record remains readable, but cannot certify an account.
    ce._store_evidence(tmp_path, "owner_acks", fp, {"route_fp": fp, "window_tokens": 1_200_000})
    ce._store_evidence(tmp_path, "probes", fp, ce.CapabilityEvidence(
        1_200_000, "confirmed", "provider_metadata", fp, MODEL, "claudexor",
        ts=ce.utc_now_iso(),
    ).to_json())
    cold = _probe(tmp_path, allow_fetch=False)
    assert cold.window_tokens == 0
    actual = _probe(tmp_path)
    assert actual.window_tokens == 872_000


def test_exact_account_ack_retains_its_binding_and_expires_on_account_change(tmp_path, catalog):
    options = {"source_id": "test-source", "credential_profile_id": "account-a",
               "account_fingerprint": "identity-a"}
    ce.record_owner_ack(tmp_path, provider="claudexor", model=MODEL,
                        window_tokens=1_200_000, options=options)
    acknowledged = _probe(tmp_path, options=options, allow_fetch=False)
    assert acknowledged.credential_profile_id == "account-a"
    assert acknowledged.account_fingerprint == "identity-a"
    replaced_account = _probe(tmp_path, options={**options, "account_fingerprint": "identity-b"},
                              allow_fetch=False)
    assert replaced_account.window_tokens == 0
