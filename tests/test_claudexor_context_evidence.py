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


def _resolve(monkeypatch, root, *, model_route=None, pin=None, preferred=False):
    """The fit's and every rebind's route question, through the production route builder. ``preferred``: the carried
    route is the account a restore's send will prefer, not one a receipt observed serving."""
    from ouroboros import config, context_fit as cf, server_runtime

    monkeypatch.setattr(config, "runtime_settings", lambda: {"OUROBOROS_MODEL": MODEL})
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(server_runtime, "apply_runtime_provider_defaults", lambda s: (s, False, []))
    return cf.resolve_context_fit_route({"model": MODEL, "model_role": "main", "task_metadata": {},
                                         "model_route": model_route or {}, "credential_profile_id": pin,
                                         **({"model_route_preferred": True} if preferred else {})}, allow_fetch=True)


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
        return _resolve(monkeypatch, tmp_path, model_route=identity, pin=pin, preferred=True)[1]

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


@pytest.mark.parametrize("record", ["none", "stale", "fresh"])
def test_a_receipt_rebind_reads_the_account_that_served_and_settles_on_it(tmp_path, monkeypatch, catalog, record):
    """Fix round 1 (Opus B1): a receipt names the account that just answered an Auto request: an observation, no
    preference. Without its fresh record the rebind reads that account's own catalog (one question, its fingerprint
    checked) and plans on it, so the next round, answered by it again, binds the same route and rebinds no more; with
    a fresh record it asks nothing. The engine's Auto pick (another account here) never sizes that plan."""
    import datetime
    from ouroboros.llm import LLMClient

    served = {"source": "test-source", "model": "exact-model", "credentialProfileId": "account-a",
              "accountFingerprint": "identity-a"}
    if record != "none":  # a statement of account-a read earlier: 30 h ago, or a minute ago
        hours = 30 if record == "stale" else 0
        catalog["value"]["observedAt"] = (ce.utc_now() - datetime.timedelta(hours=hours, minutes=1)).isoformat()
        _resolve(monkeypatch, tmp_path, pin="account-a")
        catalog["value"]["observedAt"] = ce.utc_now_iso()
    calls = []

    def fetch(source, credential_profile_id=None, *, requested_model=None):
        calls.append(credential_profile_id)
        return (_catalog("account-b", "identity-b", 400_000) if not credential_profile_id
                else _catalog(credential_profile_id, "identity-a", 1_000_000))
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(fetch))
    for _round in range(2):  # each round answered by account-a: its receipt rebinds the plan
        route, evidence = _resolve(monkeypatch, tmp_path, model_route=served)
        assert ce.evidence_model_route(evidence, MODEL) == served  # the plan binds the served account: settled
    assert calls == ([] if record == "fresh" else ["account-a"])  # one read of the served account, never Auto's pick
    assert evidence.window_tokens == (872_000 if record == "fresh" else 1_000_000)


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
        assert route["route_is_preference"] is True  # both restore rebinds bind an account a send will prefer
        evidence = _resolve(monkeypatch, tmp_path, model_route=route["model_route"], preferred=True)[1]
        return SimpleNamespace(model_route=_binding(evidence) if evidence.account_fingerprint else {}), "max"

    monkeypatch.setattr(loop, "_rebind_context_fit_plan", rebind)
    monkeypatch.setattr(continuation_tools, "reconcile_restored_tools", lambda *_a: True)
    monkeypatch.setattr(owner_wait, "sanction_rewrite", lambda *_a, **_kw: None)
    ctx = SimpleNamespace(context_fit_plan=None, active_model=MODEL, active_use_local=False, task_metadata={},
                          model_wait_context=None, active_context_mode="max")
    owner_wait.rebind_restored_route(SimpleNamespace(_ctx=ctx), {"tool_schemas": []}, _answered(conversation))
    assert calls[0] == conversation and calls[1]["accountFingerprint"] == "identity-b"
    assert catalog["calls"] == [("test-source", None, "exact-model")]  # once (Auto, no hidden pin), not twice


@pytest.mark.parametrize("preference", [True, False])
def test_only_a_restores_rebind_marks_its_account_as_the_one_a_send_prefers(monkeypatch, preference):
    """Fix round 1 (Opus B1): the rebind hands the fit its account as the one a send will prefer only for a restore
    (``route_is_preference``); a receipt's rebind hands it the account observed serving, unmarked."""
    from types import SimpleNamespace
    from ouroboros import context, loop_model_call

    class Asked(BaseException):
        """Stops the rebind at its route question, holding the task it was asked with."""

    def resolver(task, *, allow_fetch):
        raise Asked(task)

    monkeypatch.setattr(context, "_context_fit_route", resolver)
    plan = SimpleNamespace(max_projection=None, low_projection=None, core_sha256="", model_role="main")
    with pytest.raises(Asked) as asked:
        loop_model_call._rebind_context_fit_plan(plan, SimpleNamespace(_ctx=SimpleNamespace()), [], model=MODEL,
                                                 use_local=False, tool_schemas=[], route_is_preference=preference,
                                                 model_route={"credentialProfileId": "account-a"})
    task = asked.value.args[0]
    assert task["model_route"] == {"credentialProfileId": "account-a"}
    assert task.get("model_route_preferred", False) is preference


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


@pytest.mark.parametrize("repeat", ["answers", "fails"])
def test_the_guards_repeat_after_an_early_failed_read_asks_the_engine_again(tmp_path, monkeypatch, repeat):
    """Fix round 1 (Opus B2): the early read failed (an engine stalled at build start) and the pool's fresh smaller
    statement narrows the view, so the guard asks again at the usual point. That repeat reads the engine anew: the
    failure the early read wrote in this build answers nothing. An engine back by then sizes the plan; one still
    silent leaves today's failed-read path (the pool bound) after exactly one more question."""
    from ouroboros import context_fit as cf
    from ouroboros.llm import LLMClient

    small = _catalog("account-s", "identity-s", 70_000)
    small["models"][0]["contextWindow"] = 70_000
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(lambda *_a, **_kw: copy.deepcopy(small)))
    ce.probe(tmp_path, provider="claudexor", model=MODEL)  # the pool holds a fresh 70K statement of account-s
    big, calls = _catalog("account-b", "identity-b", 872_000), []

    def stalled(*_a, **_kw):
        calls.append(1)
        if len(calls) == 1 or repeat == "fails":
            raise TimeoutError("engine stalled")
        return copy.deepcopy(big)

    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(stalled))
    plan, spans = _built(monkeypatch, tmp_path, lambda task: cf.RouteQuestion(task, cf.resolve_context_fit_route))
    assert len(calls) == 2 and spans["work"]["catalog_calls"] == 2  # the early read and one repeat, no more
    if repeat == "answers":
        assert (plan.initial_mode, plan.window_tokens, plan.window_evidence[0]) == ("max", 872_000, "live")
    else:
        assert (plan.window_evidence[0], plan.window_tokens) == ("pool", 70_000) and plan.initial_mode != "max"


@pytest.mark.parametrize("late", [872_000, 1])
def test_an_early_window_that_cannot_hold_the_core_yields_to_the_usual_points_answer(tmp_path, monkeypatch, late):
    """Fix round 1 (Grok B1): an early answer whose window cannot hold the fixed core (a refusal of the start walk) is
    asked again at the usual point, like one that would narrow the view; only that answer's refusal is final, and it
    is today's refusal."""
    from ouroboros import context_fit as cf
    from ouroboros.llm import LLMClient

    answers = [_catalog("account-a", "identity-a", 1), _catalog("account-b", "identity-b", late)]
    for value in answers:
        value["models"][0]["contextWindow"] = value["models"][0]["maxContextWindow"]
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(lambda *_a, **_kw: copy.deepcopy(answers.pop(0))))
    if late == 1:
        with pytest.raises(cf.MinimumViewUnfit):
            _built(monkeypatch, tmp_path, lambda task: cf.RouteQuestion(task, cf.resolve_context_fit_route))
        assert not answers  # both asked; the usual point's refusal stands
        return
    plan, _spans = _built(monkeypatch, tmp_path, lambda task: cf.RouteQuestion(task, cf.resolve_context_fit_route))
    assert not answers and (plan.window_tokens, plan.initial_mode) == (872_000, "max")
    assert plan.model_route["credentialProfileId"] == "account-b"


@pytest.mark.parametrize("flip", [True, False])
def test_a_route_changed_during_the_wait_voids_the_early_answer(monkeypatch, flip):
    """Fix round 1 (Grok B2): the fit compares its route with the one asked about again once its wait on the handle
    ends: a route changed while the early read was in flight (a model or pin edited in Settings) asks its own
    question; an unchanged one takes the early answer."""
    import threading
    from ouroboros import context_fit as cf

    state, asked, release, waiting = {"key": "K1"}, [], threading.Event(), threading.Event()

    def key(_task):
        if threading.current_thread().name == "fit-caller":
            waiting.set()  # the fit has made its first comparison and waits on the handle
        return state["key"]

    def resolver(_task, *, allow_fetch):
        asked.append(state["key"])
        if len(asked) == 1:
            release.wait(5)  # the early read in flight
        return {"model": state["key"]}, object()

    monkeypatch.setattr(cf, "_route_key", key)
    question, box = cf.RouteQuestion({"id": "t"}, resolver), {}
    caller = threading.Thread(target=lambda: box.update(out=question({"id": "t"}, allow_fetch=True)), name="fit-caller")
    caller.start()
    assert waiting.wait(5)
    if flip:
        state["key"] = "K2"
    release.set()
    caller.join(5)
    assert (box["out"][0]["model"], question.early) == (("K2", False) if flip else ("K1", True))
    assert asked == (["K1", "K2"] if flip else ["K1"])


@pytest.mark.parametrize("used", [True, False])
def test_the_early_questions_work_stays_its_own_until_the_fit_takes_its_answer(monkeypatch, used):
    """Fix round 1 (Fable 1): the early question's thread counts in a window of its own, so the build's receipt can be
    read while it runs (no dictionary changes size under the reader, no count lost to a shared read-modify-write);
    the build adopts that work, exactly, when the fit takes the answer, and a voided question's work is not counted."""
    import threading
    from ouroboros import context_fit as cf, context_timing

    release, counted_all, keys, calls = threading.Event(), threading.Event(), {"key": "K1"}, []

    def resolver(_task, *, allow_fetch):
        calls.append(1)
        for n in range(2_000 if len(calls) == 1 else 3):  # the early read inserts keys while the build reads its receipt
            context_timing.counted("catalog_calls", key=("source", n))
        counted_all.set()
        release.wait(5)
        return {"model": "m"}, object()

    monkeypatch.setattr(cf, "_route_key", lambda _task: keys["key"])
    with context_timing.measured_sections():
        question = cf.RouteQuestion({}, resolver)
        assert counted_all.wait(5)
        for _read in range(200):
            assert "catalog_calls" not in context_timing.section_spans()["work"]  # nothing of the thread's yet
        if not used:
            keys["key"] = "K2"  # the route changed: the early answer is void
        release.set()
        question({}, allow_fetch=True)
        work = context_timing.section_spans()["work"]
    assert work.get("catalog_calls") == (2_000 if used else 3)  # adopted exactly, or only the fit's own question
    assert "catalog_calls_again" not in work


@pytest.mark.parametrize("provider", ["claudexor", "openrouter"])
def test_only_a_claudexor_route_is_asked_beside_the_build(monkeypatch, provider):
    """Fix round 1 (Fable 5): the early question exists for the engine's catalog; another route starts no thread and
    its fit asks its own question at the usual point, as before."""
    from ouroboros import context_fit as cf

    asked = []
    monkeypatch.setattr(cf, "_context_route", lambda _task: ({"provider": provider, "model": "m"}, {}))
    question = cf.RouteQuestion({}, lambda _task, *, allow_fetch: asked.append("early") or ({"model": "m"}, object()))
    assert question.asked.wait(5)
    assert (asked, question.key is None) == ((["early"], False) if provider == "claudexor" else ([], True))


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


@pytest.mark.parametrize("held", ["failed-read", "later-statement"])
def test_an_output_limit_dated_earlier_replaces_a_failed_read_never_a_later_statement(tmp_path, monkeypatch, held):
    """Fix round 2 (Carbon B1): a failed read of the output limit states nothing, so the engine's answer dated before it
    (its catalog's ``observedAt``) is stored and sizes the reserve; a statement dated later still keeps an answer read
    earlier out, as for windows."""
    import datetime
    from ouroboros import response_limits
    from ouroboros.llm import LLMClient

    exact = {"source_id": "test-source", "credential_profile_id": "account-a", "account_fingerprint": "identity-a"}

    def answers(minutes_ago, maximum):
        value = _catalog()
        value["observedAt"] = (ce.utc_now() - datetime.timedelta(minutes=minutes_ago)).isoformat()
        value["models"][0]["maxOutputTokens"] = maximum
        monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(lambda *_a, **_kw: copy.deepcopy(value)))

    if held == "failed-read":
        monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(
            lambda *_a, **_kw: (_ for _ in ()).throw(TimeoutError("controlled daemon timeout"))))
        response_limits.resolve_response_limit(tmp_path, provider="claudexor", model=MODEL, options=exact, allow_fetch=True)
    else:
        answers(1, 32_000)
        _probe(tmp_path)
    answers(5, 16_384)  # the engine's catalog, observed before the held row was written
    _probe(tmp_path)
    fp = response_limits._response_fingerprint(provider="claudexor", model=MODEL, options=exact)
    assert ce._read(tmp_path)["response_limits"][fp]["max_output_tokens"] == (16_384 if held == "failed-read" else 32_000)


@pytest.mark.parametrize("relogin", ["sibling-model", "unprobeable-11-minutes", "none"])
def test_a_silent_engine_keeps_a_pins_record_only_while_its_account_is_the_newest_observed(tmp_path, monkeypatch, relogin):
    """Fix round 2 (Carbon B2): the account behind a pinned profile is the newest one observed on any model of its
    source, and an observation naming it outlives its ten-minute retry throttle (kept for the confirmed lifetime), so a
    silent engine never plans by an account the host saw replaced: unknown window, Max, as the target. Without a
    relogin the pin keeps its own fresh record."""
    import datetime
    from ouroboros import context_fit as cf
    from ouroboros.llm import LLMClient

    start = ce.utc_now()
    clock = {"now": start}
    monkeypatch.setattr(ce, "utc_now", lambda: clock["now"])
    monkeypatch.setattr(ce, "utc_now_iso", lambda: clock["now"].isoformat())

    def answers(fingerprint, window, model="exact-model"):
        value = {**_catalog("account-a", fingerprint, window), "observedAt": clock["now"].isoformat()}
        value["models"] = [{**value["models"][0], "id": model, "contextWindow": window}] if window else []
        monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(lambda *_a, **_kw: copy.deepcopy(value)))

    pinned = {"source_id": "test-source", "credential_profile_id": "account-a"}
    answers("identity-a", 70_000)
    _probe(tmp_path, options=pinned)  # the pinned profile holds identity-a: its record, 70K
    clock["now"] = start + datetime.timedelta(minutes=1)
    if relogin == "sibling-model":  # identity-b observed behind the profile, on another model of the source
        answers("identity-b", 1_000_000, model="sibling-model")
        ce.probe(tmp_path, provider="claudexor", model="claudexor::test-source=sibling-model", options=pinned)
    elif relogin == "unprobeable-11-minutes":  # identity-b observed with no window stated; the engine falls silent later
        answers("identity-b", 0)
        _probe(tmp_path, options={**pinned, "account_fingerprint": "identity-b"})
        clock["now"] = start + datetime.timedelta(minutes=12)
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(
        lambda *_a, **_kw: (_ for _ in ()).throw(TimeoutError("controlled daemon timeout"))))
    plan, _spans = _built(monkeypatch, tmp_path, lambda _task: cf.resolve_context_fit_route,
                          task={**TASK, "credential_profile_id": "account-a"})
    if relogin == "none":
        assert (plan.window_tokens, plan.model_route.get("accountFingerprint")) == (70_000, "identity-a")
    else:
        assert (plan.window_tokens, plan.initial_mode, plan.window_witness_bound) == (0, "max", False)


@pytest.mark.parametrize("repeat", ["guard", "within-one-pass"])
def test_the_guards_repeat_is_a_pass_of_its_own_not_work_done_again(tmp_path, monkeypatch, catalog, repeat):
    """Fix round 2 (Carbon S1): the horizon guard asks and walks again by design: its receipt names exactly the work done
    again that the same build asked once names (the target's single pass), so the health block says nothing more. A
    read repeated within one pass still counts as done again, and the health block names it."""
    from ouroboros import context_fit as cf
    from ouroboros.context_health import _host_work_line
    from ouroboros.transcript_prefix import _running_code

    guard = repeat == "guard"
    window = 70_000 if guard else 872_000
    catalog["value"]["models"][0].update(contextWindow=window, maxContextWindow=window)

    def twice(task, *, allow_fetch):  # one pass that reads the same catalog twice
        cf.resolve_context_fit_route(task, allow_fetch=allow_fetch)
        return cf.resolve_context_fit_route(task, allow_fetch=allow_fetch)

    def receipt(resolver_for, root):
        plan, spans = _built(monkeypatch, root, resolver_for)
        again = {verb: n for verb, n in spans["work"].items() if verb.endswith("_again")}
        return plan, spans["work"], again, _host_work_line([{"code": _running_code(), "work": spans["work"]}])

    _plan, _work, once, once_line = receipt(lambda _task: cf.resolve_context_fit_route, tmp_path / "once")
    plan, work, again, line = receipt(lambda task: cf.RouteQuestion(task, cf.resolve_context_fit_route if guard else twice),
                                      tmp_path / "early")
    assert work["catalog_calls"] == 2 and (plan.initial_mode != "max") is guard  # asked twice: by the guard, or in one pass
    if guard:
        assert again == once and line == once_line  # the guard's repeat adds no work done again, and no health fact
    else:
        assert again == {**once, "catalog_calls_again": 1} and "catalog_calls 1" in line


@pytest.mark.parametrize("early", ["takes-F1", "takes-no-step"])
def test_an_early_view_that_only_collapses_task_lines_counts_as_narrowed(early):
    """Fix round 2 (Fable 5.2 finding 1, the reviewer's seam test): a walk that took any step, collapsed task lines (F1)
    included, shows less than the selected view, so the guard asks again; a walk that took none stays quiet."""
    from types import SimpleNamespace
    from ouroboros import context_fit as cf, memory_floor as mf
    from tests import _memory_view_synthetic as syn

    snapshot = syn.snapshot(room_facts=syn.room("1", lane2=6))
    args = dict(mode="max", owner_mode="max", known_window=True, output_reserve=8192, ratio=1, non_memory_tokens=10_000)
    low, high = 1, 872_000  # the largest window whose walk takes any step
    while high - low > 1:
        mid = (low + high) // 2
        low, high = (mid, high) if dict(mf.render_view_for_mode(snapshot, window_tokens=mid, **args).level.steps) else (low, mid)
    window = low if early == "takes-F1" else 872_000
    view = mf.render_view_for_mode(snapshot, window_tokens=window, **args)
    assert set(dict(view.level.steps)) == ({"F1"} if early == "takes-F1" else set()) and not view.facts["floor"]["by_budget"]
    state = SimpleNamespace(known_window=True, output_reserve=8192, window_record=SimpleNamespace(total_equivalent=window))
    assert cf.narrowed_by_window({"max": (None, view.facts)}, "max", "max", state, {}) is (early == "takes-F1")


@pytest.mark.parametrize("sized_by", ["refusal", "live-read"])
def test_the_receipt_names_what_sized_the_window(tmp_path, sized_by):
    """Fix round 2 (Fable 5.2 finding 4): the memory view's receipt names the evidence that sized the window: ``live``
    only for a window the read itself stated; a window a refusal sized beside a read that stated none is ``recorded``,
    with the refusal's date, not the read's."""
    from ouroboros.context_window import record_observation, resolve_window

    stated = 872_000 if sized_by == "live-read" else 0
    evidence = ce.CapabilityEvidence(stated, ce.STATUS_CONFIRMED if stated else ce.STATUS_UNPROBEABLE,
                                     ce.SOURCE_PROVIDER_METADATA, "scope", MODEL, "claudexor", ts=ce.utc_now_iso())
    evidence.read = "answered"
    if not stated:
        record_observation(tmp_path, "scope", refused=90_000, kind="input")
    record = resolve_window(evidence, 16_384, root=tmp_path, observation_fp="scope")
    refused = ((ce._read(tmp_path).get("main_windows") or {}).get("scope") or {}).get("observations", [{}])[-1].get("ts")
    assert ce.window_evidence(evidence, record) == (("live", evidence.ts) if stated else ("recorded", refused))


@pytest.mark.parametrize("output", ["stated", "expired"])
def test_a_pins_lent_record_answers_its_output_only_from_that_accounts_records(tmp_path, monkeypatch, output):
    """Fix round 2 (Carbon B1, reproduction 2b): after a failed read the pin plans by its account's own record, and that
    account's output limit comes only from its own records: a fresh stated maximum answers; an expired one stays
    unknown, and the failed read writes nothing under that account, so its next read of the output limit is not held
    back for ten minutes."""
    import datetime
    from ouroboros import response_limits
    from ouroboros.llm import LLMClient

    start = ce.utc_now()
    clock = {"now": start}
    monkeypatch.setattr(ce, "utc_now", lambda: clock["now"])
    monkeypatch.setattr(ce, "utc_now_iso", lambda: clock["now"].isoformat())

    def answers(maximum, observed):
        value = {**_catalog(), "observedAt": observed.isoformat()}
        value["models"][0].update(contextWindow=110_000, maxContextWindow=110_000, maxOutputTokens=maximum)
        monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(lambda *_a, **_kw: copy.deepcopy(value)))

    pinned = {"source_id": "test-source", "credential_profile_id": "account-a"}
    exact = {**pinned, "account_fingerprint": "identity-a"}
    answers(16_384 if output == "stated" else None, start)
    _probe(tmp_path, options=pinned)  # identity-a: window 110K, its output stated or not
    clock["now"] = start + datetime.timedelta(minutes=15)
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(
        lambda *_a, **_kw: (_ for _ in ()).throw(TimeoutError("controlled daemon timeout"))))
    kept = _probe(tmp_path, options=pinned)
    fp = response_limits._response_fingerprint(provider="claudexor", model=MODEL, options=exact)
    held = ce._read(tmp_path)["response_limits"][fp]
    assert kept.window_tokens == 110_000 and not held["source"].endswith(response_limits.FAILED_READ)
    if output == "stated":
        assert kept.response_limit["max_output_tokens"] == 16_384  # that account's own fresh statement
        return
    assert not kept.response_limit["max_output_tokens"]  # expired: unknown, and nothing written under identity-a
    answers(16_384, start + datetime.timedelta(minutes=10))  # the engine answers again
    assert _probe(tmp_path, options=exact).response_limit["max_output_tokens"] == 16_384  # read at once, not held back


@pytest.mark.parametrize("case", ["relogin", "relogin-then-11-minutes", "no-relogin"])
def test_a_failed_read_of_an_observed_account_keeps_it_the_profiles_newest(tmp_path, monkeypatch, case):
    """Fix round 3 (Grok r2, Opus r2 finding 1): a failed read of a route that names its whole account keeps naming that
    account (no window, nothing bound), and lives as long as the observation it replaces (24 h), so a silent pin never
    lends the account the host saw replaced, at once or 11 minutes later. Without a relogin a failed read of the pin's
    own account keeps its record, which the silent pin still lends."""
    import datetime
    from ouroboros import context_fit as cf
    from ouroboros.llm import LLMClient

    start = ce.utc_now()
    clock = {"now": start}
    monkeypatch.setattr(ce, "utc_now", lambda: clock["now"])
    monkeypatch.setattr(ce, "utc_now_iso", lambda: clock["now"].isoformat())

    def answers(fingerprint, window):
        value = {**_catalog("account-a", fingerprint, window), "observedAt": clock["now"].isoformat()}
        value["models"] = [{**value["models"][0], "contextWindow": window}] if window else []
        monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(lambda *_a, **_kw: copy.deepcopy(value)))

    pinned = {"source_id": "test-source", "credential_profile_id": "account-a"}
    answers("identity-a", 70_000)
    _probe(tmp_path, options=pinned)  # the pinned profile holds identity-a: its record, 70K
    relogin = case != "no-relogin"
    if relogin:  # identity-b observed behind the profile, no window stated for this model
        clock["now"] = start + datetime.timedelta(minutes=1)
        answers("identity-b", 0)
        _probe(tmp_path, options={**pinned, "account_fingerprint": "identity-b"})
    clock["now"] = start + datetime.timedelta(minutes=12)  # past the read throttle: the exact route is read again
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(
        lambda *_a, **_kw: (_ for _ in ()).throw(TimeoutError("controlled daemon timeout"))))
    named = {**pinned, "account_fingerprint": "identity-b" if relogin else "identity-a"}
    failed = _probe(tmp_path, options=named, force=not relogin)  # a receipt rebind's read of that account fails
    assert (failed.window_tokens, failed.status) == ((0, ce.STATUS_FAILED) if relogin else (70_000, ce.STATUS_CONFIRMED))
    if relogin:  # the failure names no account to its caller; its row keeps identity-b's observation, with its date
        assert failed.account_fingerprint == ""
        kept = ce._read(tmp_path)["probes"][ce.route_fingerprint(provider="claudexor", model=MODEL, options=named)]
        assert kept["kept_observation"]["account_fingerprint"] == "identity-b" and kept["kept_observation"][
            "observed_at"] == (start + datetime.timedelta(minutes=1)).isoformat()
    if case == "relogin-then-11-minutes":
        clock["now"] = start + datetime.timedelta(minutes=23)
    plan, _spans = _built(monkeypatch, tmp_path, lambda _task: cf.resolve_context_fit_route,
                          task={**TASK, "credential_profile_id": "account-a"})
    if relogin:
        assert (plan.window_tokens, plan.initial_mode, plan.window_witness_bound) == (0, "max", False)
    else:
        assert (plan.window_tokens, plan.model_route.get("accountFingerprint")) == (70_000, "identity-a")


@pytest.mark.parametrize("b_window", [0, 872_000])
@pytest.mark.parametrize("failure", ["binding-mismatch", "timeout", "timeout-over-its-old-observation"])
def test_a_failed_read_of_a_replaced_account_creates_no_observation(tmp_path, monkeypatch, b_window, failure):
    """Fix round 4 (Fable 5.2 r3 finding 1, the reviewer's four cases): a route that still names the replaced account
    (a receipt from before the relogin), read on a model with no record of it, fails (it times out, or the catalog
    answers that the profile is now another account). A failure neither erases, creates nor re-dates an observation,
    so identity-b stays the profile's newest: a silent pin plans b's own window or an unknown one, never identity-a's.
    Over a row that already observed identity-a on that model, the failure keeps that observation with its old date."""
    import datetime
    from ouroboros import context_fit as cf
    from ouroboros.llm import LLMClient

    start = ce.utc_now()
    clock = {"now": start}
    monkeypatch.setattr(ce, "utc_now", lambda: clock["now"])
    monkeypatch.setattr(ce, "utc_now_iso", lambda: clock["now"].isoformat())

    def answers(fingerprint, window):
        value = {**_catalog("account-a", fingerprint, window), "observedAt": clock["now"].isoformat()}
        value["models"] = [{**value["models"][0], "contextWindow": window, "maxContextWindow": window}] if window else []
        monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(lambda *_a, **_kw: copy.deepcopy(value)))

    def silent():
        monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(
            lambda *_a, **_kw: (_ for _ in ()).throw(TimeoutError("controlled daemon timeout"))))

    pinned = {"source_id": "test-source", "credential_profile_id": "account-a"}
    answers("identity-a", 70_000)
    _probe(tmp_path, options=pinned)  # identity-a: 70K
    other = "claudexor::test-source=other-model"
    if failure == "timeout-over-its-old-observation":  # identity-a seen on the other model too, with no window there
        answers("identity-a", 0)
        ce.probe(tmp_path, provider="claudexor", model=other, options={**pinned, "account_fingerprint": "identity-a"})
    clock["now"] = start + datetime.timedelta(minutes=1)
    answers("identity-b", b_window)
    _probe(tmp_path, options={**pinned, "account_fingerprint": "identity-b"})  # the relogin, observed
    clock["now"] = start + datetime.timedelta(minutes=12)  # past the read throttle of identity-a's row there
    if failure != "binding-mismatch":
        silent()  # else the catalog answers identity-b to a question naming identity-a: a binding mismatch
    stale = ce.probe(tmp_path, provider="claudexor", model=other,
                     options={**pinned, "account_fingerprint": "identity-a"})
    assert (stale.status, stale.account_fingerprint) == (ce.STATUS_FAILED, "")  # names no account, as on the target
    clock["now"] = start + datetime.timedelta(minutes=13)
    silent()
    plan, _spans = _built(monkeypatch, tmp_path, lambda _task: cf.resolve_context_fit_route,
                          task={**TASK, "credential_profile_id": "account-a"})
    assert (plan.window_tokens, plan.model_route.get("accountFingerprint") or "") == (
        (872_000, "identity-b") if b_window else (0, ""))


@pytest.mark.parametrize("engine", ["silent", "answers-after-the-throttle"])
def test_a_failed_read_of_the_served_account_keeps_the_receipt_rebinding(tmp_path, monkeypatch, engine):
    """Fix round 4 (Opus r3 finding 1): a receipt rebind whose read of the served account (no record) fails names no
    account, so the plan does not settle on an unknown window: the next round rebinds again, and once the 10-minute
    throttle is past the account's read answers and the plan settles on it, as on the target."""
    import datetime
    from ouroboros.llm import LLMClient

    served = {"source": "test-source", "model": "exact-model", "credentialProfileId": "account-a",
              "accountFingerprint": "identity-a"}
    up = {"now": False}

    def fetch(source, credential_profile_id=None, *, requested_model=None):
        if not up["now"]:
            raise TimeoutError("controlled daemon timeout")
        return _catalog("account-a", "identity-a", 872_000)
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(fetch))
    _route, evidence = _resolve(monkeypatch, tmp_path, model_route=served)
    assert evidence.window_tokens == 0 and ce.evidence_model_route(evidence, MODEL) != served  # rebinds again
    if engine == "answers-after-the-throttle":
        later = ce.utc_now() + datetime.timedelta(minutes=11)
        monkeypatch.setattr(ce, "utc_now", lambda: later)
        up["now"] = True
        _route, evidence = _resolve(monkeypatch, tmp_path, model_route=served)
        assert evidence.window_tokens == 872_000 and ce.evidence_model_route(evidence, MODEL) == served  # settled


def _owner_mode_build(monkeypatch, root, mode, windows, snapshot, *, early=True):
    """One start in the owner's ``mode`` with the given catalog windows in order (the early question's, then the usual
    point's; a 0 times out), and the catalog calls it made."""
    from ouroboros import config, context_fit as cf, server_runtime
    from ouroboros.llm import LLMClient
    from ouroboros.memory_view import snapshot_json

    monkeypatch.setattr(config, "runtime_settings", lambda: {"OUROBOROS_MODEL": MODEL, "OUROBOROS_CONTEXT_MODE": mode})
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(server_runtime, "apply_runtime_provider_defaults", lambda s: (s, False, []))
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: root)
    answers = list(windows)

    def fetch(*_a, **_kw):
        window = answers.pop(0) if len(answers) > 1 else answers[0]
        if not window:
            raise TimeoutError("controlled daemon timeout")
        value = _catalog("account-b", "identity-b", window)
        value["models"][0].update(contextWindow=window, maxOutputTokens=16_384)
        return value
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(fetch))
    calls = []
    resolver = cf.RouteQuestion if early else (lambda _task, _resolver: _resolver)

    def counted(task, *, allow_fetch):
        calls.append(1)
        return cf.resolve_context_fit_route(task, allow_fetch=allow_fetch)
    core = cf.ContextCore("Core governance. " * 1500, "BIBLE text. " * 300, "ARCH text. " * 800, "DEV text. " * 300,
                          "Semi stable.", "Dynamic.", '"Assignment"', False, memory_view_json=snapshot_json(snapshot))
    task = {"model": MODEL, "model_role": "main", "task_metadata": {}}
    plan = cf.build_context_fit_plan(None, core, task, preferred_mode=mode, route_resolver=resolver(task, counted))
    return plan, len(calls)


def _memory(scale=1, owner_rows=0):
    from tests import _memory_view_synthetic as syn

    return syn.snapshot(story=syn.pages(18 * scale, 2500), conversations=[
        syn.conversation(i, words=6, word_chars=800) for i in range(4 * scale)],
        room_facts=syn.room("1", lane2=12, lane1=[syn.spoken(i, "human", 20_000) for i in range(owner_rows)]))


@pytest.mark.parametrize(("mode", "early", "pool"), [("low", 0, 95_000), ("low", 45_000, 0), ("nano", 30_000, 0)])
def test_a_window_below_the_owners_mode_target_narrows_and_is_asked_again(tmp_path, monkeypatch, mode, early, pool):
    """Fix round 5 (Carbon r34 B1, its three cases): in the owner's Low or Nano, an early window at or below the mode's
    target sets the start bound itself, so its cuts are the window's: the guard asks again and the plan shows the
    usual point's memory (872,000), the target's view. Here the early read times out onto a 95,000 pool bound, or
    answers 45,000 (Low) or 30,000 (Nano) before the usual point answers 872,000."""
    if pool:  # a fresh 95K statement of another account: the failed early read's pool bound
        from ouroboros.llm import LLMClient
        monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(
            lambda *_a, **_kw: {**_catalog("account-a", "identity-a", pool), "models": [
                {**_catalog()["models"][0], "contextWindow": pool, "maxContextWindow": pool}]}))
        _probe(tmp_path / "early", options={"source_id": "test-source", "credential_profile_id": "account-a"})
    target, _calls = _owner_mode_build(monkeypatch, tmp_path / "target", mode, [872_000], _memory(), early=False)
    plan, calls = _owner_mode_build(monkeypatch, tmp_path / "early", mode, [early, 872_000], _memory())
    assert calls == 2 and plan.window_tokens == 872_000  # asked again; the usual point's answer is used
    assert plan.projection(mode).system_content_json == target.projection(mode).system_content_json


def test_the_same_small_window_twice_is_asked_again_and_walked_once(tmp_path, monkeypatch):
    """Fix round 5: owner Low with the same 45,000 window early and at the usual point: the guard asks once more, and
    the plan is the target's (the second walk is skipped on the same frame)."""
    target, _calls = _owner_mode_build(monkeypatch, tmp_path / "target", "low", [45_000], _memory(), early=False)
    plan, calls = _owner_mode_build(monkeypatch, tmp_path / "early", "low", [45_000, 45_000], _memory())
    assert calls == 2 and plan.window_tokens == 45_000
    assert plan.projection("low").system_content_json == target.projection("low").system_content_json


@pytest.mark.parametrize("case", ["owner-budget-only", "physical-fold"])
def test_above_the_mode_target_only_the_windows_physical_fold_narrows(tmp_path, monkeypatch, case):
    """Fix round 5: with the window above the owner's mode target, cuts the budget took are the owner's own: owner Low
    at 400,000 over a memory its 250,000 budget cuts asks no extra question. The window's physical fold of this room's
    uncovered owner rows (owner Nano, rows larger than a 200,000 window's room) is the window's: it is asked again,
    and the plan shows the usual point's (872,000) memory, the target's."""
    if case == "owner-budget-only":
        plan, calls = _owner_mode_build(monkeypatch, tmp_path, "low", [400_000], _memory(scale=30))
        floor = plan.projection("low").memory_facts["floor"]
        assert floor["by_budget"] and calls == 1 and plan.window_tokens == 400_000  # owner's cuts: quiet
        return
    memory = _memory(owner_rows=60)
    target, _calls = _owner_mode_build(monkeypatch, tmp_path / "target", "nano", [872_000], memory, early=False)
    early, _calls = _owner_mode_build(monkeypatch, tmp_path / "alone", "nano", [200_000], memory, early=False)
    assert early.projection("nano").memory_facts["floor"]["steps"].get("F7")  # 200K folds owner rows, 872K does not
    assert not target.projection("nano").memory_facts["floor"]["steps"].get("F7")
    plan, calls = _owner_mode_build(monkeypatch, tmp_path / "early", "nano", [200_000, 872_000], memory)
    assert calls == 2 and plan.projection("nano").system_content_json == target.projection("nano").system_content_json
