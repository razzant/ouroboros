"""Planning evidence is exact-route, and I/T/O/R are independent facts."""
from dataclasses import replace

import pytest

from ouroboros import capability_evidence as ce
from ouroboros.context_window import WindowLimit, WindowRecord, record_observation, resolve_window


@pytest.mark.parametrize("input_limit,total,expected", [(200_000, None, 200_000),
    (None, 200_000, 134_464), (150_000, 200_000, 134_464), (100_000, 200_000, 100_000)])
def test_reserve_is_charged_once_only_to_a_total(input_limit, total, expected):
    record = WindowRecord("r", WindowLimit(input_limit, "input", "metadata") if input_limit else None,
                          WindowLimit(total, "total", "metadata") if total else None, reserve=65_536)
    assert record.usable == expected
    assert record.total_equivalent - record.reserve == expected
    assert WindowRecord("unknown", reserve=65_536).usable is None


def test_refusal_expires_or_is_superseded_only_by_a_later_larger_acceptance(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    monkeypatch.setattr(ce, "utc_now_iso", lambda: now.isoformat())
    evidence = ce.CapabilityEvidence(872_000, "confirmed", "metadata", "r")
    record_observation(tmp_path, "r", accepted=921_227, reserve=65_536, transmitted=65_536)
    accepted = resolve_window(evidence, 65_536, root=tmp_path)
    assert accepted.input.tokens == 921_227 and accepted.total is None and accepted.usable == 921_227
    assert accepted.source.startswith("accepted")
    record_observation(tmp_path, "r", refused=200_000)
    assert resolve_window(evidence, 65_536, root=tmp_path).total.tokens == 200_000
    record_observation(tmp_path, "r", accepted=199_999)
    assert resolve_window(evidence, 65_536, root=tmp_path).source.startswith("refusal")
    record_observation(tmp_path, "r", accepted=210_000)
    assert resolve_window(evidence, 65_536, root=tmp_path).usable == 921_227
    record_observation(tmp_path, "r", refused=200_000)
    monkeypatch.setattr(ce, "utc_now", lambda: now + timedelta(seconds=ce._CONFIRMED_TTL_SEC + 1))
    assert resolve_window(evidence, 65_536, root=tmp_path).usable == 921_227
    assert resolve_window(replace(evidence, route_fp="other"), 65_536, root=tmp_path).total.tokens == 872_000
    assert resolve_window(evidence, 65_536, root=tmp_path, bound=False).total.tokens == 872_000


@pytest.mark.parametrize("message,usable", [
    ("prompt is too long: 210000 tokens > 200000 maximum", 200_000),
    ("The input token count (210000) exceeds the maximum number of tokens allowed (200000).", 200_000),
    ("This model's maximum context length is 200000 tokens.", 134_464),
])
def test_numberless_refusal_adds_no_ceiling(tmp_path, monkeypatch, message, usable):
    from types import SimpleNamespace
    from ouroboros.context_window import record_refusal

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    plan = SimpleNamespace(route_fp="r", window_witness_bound=True)
    evidence = ce.CapabilityEvidence(0, "unprobeable", "none", "r")
    record_refusal({"_last_llm_provider_message": "context length exceeded"}, plan)
    assert resolve_window(evidence, 65_536, root=tmp_path).usable is None
    record_refusal({"_last_llm_provider_message": message}, plan)
    assert resolve_window(evidence, 65_536, root=tmp_path).usable == usable


@pytest.mark.parametrize("main", [True, False])
def test_settlement_binds_main_density_and_acceptance_but_auxiliary_calls_do_not(tmp_path, main):
    from types import SimpleNamespace

    request = SimpleNamespace(provider="openai", model="openai::model", prompt_tokens_bounded_estimate=100_000,
        drive_root=tmp_path, max_completion_tokens=1234, physical_context=(SimpleNamespace(view_kind="own_wake",
        route_fp="own-route", window_observation_fp="own-account", window_witness_bound=True,
        response_reserve_tokens=65_536) if main else None))
    ce.observe_token_density(request, {"prompt_tokens": 110_000}, drive_root_resolver=lambda root: root)
    density, source = ce.resolve_main_token_density(tmp_path, "own-route", "openai::model",
        estimate_tokens=100_000, view_kind="own_wake")
    assert (density, source) == ((1.1, "fresh_route_usage") if main else (1.0, "cold_estimate"))
    assert ce.resolve_main_token_density(tmp_path, "other-route", "openai::model",
        estimate_tokens=100_000, view_kind="own_wake") == (1.0, "cold_estimate")
    evidence = ce.CapabilityEvidence(0, "unprobeable", "none", "own-account")
    record = resolve_window(evidence, 65_536, root=tmp_path)
    assert record.usable is None
    assert record.accepted == (110_000 if main else 0)
    if main:
        [observation] = ce._load(tmp_path)["main_windows"]["own-account"]["observations"]
        assert observation["reserve"] == 65_536 and observation["transmitted"] == 1234
        assert observation["kind"] == "input"


@pytest.mark.parametrize("metadata", [0, 1_000_000])
def test_a_plan_releases_its_refusal_ceiling_when_the_observation_expires(tmp_path, monkeypatch, metadata):
    from datetime import datetime, timedelta, timezone
    from ouroboros import context_fit as cf

    now = datetime.now(timezone.utc)
    monkeypatch.setattr(ce, "utc_now_iso", lambda: now.isoformat())
    monkeypatch.setattr(ce, "utc_now", lambda: now)
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    record_observation(tmp_path, "expiry-route", refused=200_000)
    core = cf.ContextCore("Core", "", "", "", "", "", '"Assignment"', False)
    plan = cf.build_context_fit_plan(None, core, {"type": "chat"}, preferred_mode="low",
        route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "test"},
            ce.CapabilityEvidence(metadata, "confirmed" if metadata else "unprobeable", "metadata", "expiry-route")))
    def capacity():
        return cf.measure_main_fit(plan, plan.messages_for("low"), [], profile="owner_low",
            rendered_mode="low", round_id="expiry").measurement.capacity_total_tokens

    assert capacity() == 200_000
    now += timedelta(seconds=ce._CONFIRMED_TTL_SEC - 1)
    assert capacity() == 200_000
    now += timedelta(seconds=2)
    assert capacity() == (metadata or None)


@pytest.mark.parametrize("stated,accepted,usable", [
    (872_000, 921_227, 921_227), (1_000_000, 200_000, 934_464),
    (1_000_000, 950_000, 934_464), (0, 921_227, None),
])
def test_start_and_in_task_admission_share_the_input_bound(tmp_path, monkeypatch, stated, accepted, usable):
    from ouroboros import context_fit as cf
    from ouroboros.context_view_facts import window_fact
    from ouroboros.llm_attempt import main_context_fit_refusal
    from ouroboros.usage_accounting import AttemptRequest
    from tests.test_memory_window_acceptance import _core

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr(ce, "resolve_main_token_density", lambda *_a, **_kw: (1.0, "cold_estimate"))
    monkeypatch.setattr(ce, "utc_now_iso", lambda: "2026-10-19T14:56:16+00:00")
    record_observation(tmp_path, "incident", accepted=accepted)
    core, _snap = _core(True)
    plan = cf.build_context_fit_plan(None, core, {"usage_category": "consciousness"}, preferred_mode="max",
        route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "test"},
            ce.CapabilityEvidence(stated, "confirmed" if stated else "unprobeable", "metadata", "incident")))
    assert plan.window_record.usable == usable
    line = window_fact(plan, plan.output_reserve_tokens)
    if usable is None:
        assert plan.initial_mode == "max"
        assert not plan.projection("max").memory_facts["floor"]["steps"]
        assert core.architecture_md in plan.projection("max").system_content_json
        assert "window unknown" in line and "921,227" in line and "2026-10-19" in line
    elif accepted > stated:
        assert "input: accepted 2026-10-19" in line
    for size in ([880_000, 921_227, 921_228] if stated == 872_000 else [900_000, 950_000]):
        monkeypatch.setattr(cf, "estimate_context_prompt_tokens", lambda *_a, **_kw: size)
        disposition = cf.measure_main_fit(plan, [], [], profile="owner_max", rendered_mode="max", round_id="r",
                                          accepted_input_tokens=accepted)
        physical = cf._physical_context_for_fit(disposition)
        refusal = main_context_fit_refusal(AttemptRequest(model="test", provider="test",
            prompt_tokens_bounded_estimate=size, physical_context=physical))
        assert disposition.predicted_capacity_miss is (usable is not None and size > usable)
        assert (refusal is not None) is (usable is not None and size > usable)


@pytest.mark.parametrize("model,key", [
    ("openai::model", "OPENAI_API_KEY"), ("anthropic::model", "ANTHROPIC_API_KEY"),
    ("deepseek::model", "DEEPSEEK_API_KEY"), ("zai::model", "ZAI_API_KEY"),
    ("minimax::model", "MINIMAX_API_KEY"), ("cloudru::model", "CLOUDRU_FOUNDATION_MODELS_API_KEY"),
    ("gigachat::model", "GIGACHAT_CREDENTIALS"), ("openrouter::model", "OPENROUTER_API_KEY"),
    ("openai-compatible::model", "OPENAI_COMPATIBLE_API_KEY"),
])
def test_direct_account_scope_survives_same_key_but_not_key_rotation(tmp_path, monkeypatch, model, key):
    import json
    from ouroboros import config, context_fit as cf, server_runtime
    from ouroboros.context_anchor import record_input_anchor, main_input_anchor
    from ouroboros.context_window import record_refusal
    from ouroboros.usage_accounting import AttemptRequest

    settings = {"OUROBOROS_MODEL": model, key: "fixture-account-A",
                "OPENAI_COMPATIBLE_BASE_URL": "https://fixture.invalid/v1"}
    monkeypatch.setattr(config, "runtime_settings", lambda: dict(settings))
    monkeypatch.setattr(server_runtime, "apply_runtime_provider_defaults", lambda s: (s, False, []))
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr(ce, "probe", lambda _root, **kw: ce.CapabilityEvidence(600_000, "confirmed", "metadata",
        ce.route_fingerprint(**{k: kw[k] for k in ("provider", "model", "base_url", "options")})))
    core = cf.ContextCore("Core", "", "", "", "", "", '"Assignment"', False)
    def start():
        return cf.build_context_fit_plan(None, core, {}, preferred_mode="max", route_resolver=cf.resolve_context_fit_route)

    first = start()
    measured = cf.measure_main_fit(first, first.messages_for("max"), [], profile="owner_max", rendered_mode="max", round_id="r")
    ce.observe_token_density(AttemptRequest(model=model, provider=first.provider, drive_root=tmp_path,
        prompt_tokens_bounded_estimate=700_000, physical_context=cf._physical_context_for_fit(measured)),
        {"prompt_tokens": 700_000}, drive_root_resolver=lambda root: root)
    usage = {"_last_round_cache_usage": {"prompt_tokens": 700_000, "provider": first.provider},
             "_context_raw_input_tokens": 700_000}
    record_input_anchor(usage, first, round_idx=1)
    same = start()
    assert same.window_record.usable == 700_000
    assert main_input_anchor(usage, same)[1] == 700_000
    record_refusal({"_last_llm_provider_message": "maximum context length is 200000 tokens"}, first)
    assert start().window_record.usable == 134_464
    settings[key] = "fixture-account-B"
    other = start()
    assert other.route_fp == first.route_fp  # capability acknowledgement contract is unchanged
    assert other.window_record.route_fp != first.window_record.route_fp
    assert other.window_record.accepted == 0 and other.window_record.usable == 534_464
    assert main_input_anchor(usage, other) == (None, None)
    settings[key] = "fixture-account-A"
    assert start().window_record.usable == 134_464
    assert "fixture-account-" not in json.dumps(ce._load(tmp_path))


def test_a_claudexor_account_reads_the_acceptances_its_receipts_wrote(tmp_path, monkeypatch):
    """6A through the production route builder: the catalog binds the account after the route is resolved; the
    next plan of that account reads what its receipts wrote, another account does not, and N <= T keeps T - R."""
    from ouroboros import config, context_fit as cf, server_runtime
    from ouroboros.context_anchor import record_input_anchor
    from ouroboros.llm_attempt import main_context_fit_refusal
    from ouroboros.usage_accounting import AttemptRequest

    model, catalog = "claudexor::codex=gpt-fixture", {"account": "a", "window": 872_000}
    monkeypatch.setattr(config, "runtime_settings", lambda: {"OUROBOROS_MODEL": model})
    monkeypatch.setattr(server_runtime, "apply_runtime_provider_defaults", lambda s: (s, False, []))
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr(ce, "resolve_main_token_density", lambda *_a, **_kw: (1.0, "cold_estimate"))

    def probe(_root, **kw):  # the catalog names the account it advertises; the request carried none
        assert "account_fingerprint" not in (kw["options"] or {})
        bound = {"source_id": "codex", "credential_profile_id": f"profile-{catalog['account']}",
                 "account_fingerprint": f"fp-{catalog['account']}"}
        return ce.CapabilityEvidence(catalog["window"], "confirmed", "metadata", ce.route_fingerprint(
            provider=kw["provider"], model=kw["model"], base_url=kw["base_url"], options={**kw["options"], **bound}), **bound)
    monkeypatch.setattr(ce, "probe", probe)
    core = cf.ContextCore("Core", "", "", "", "", "", '"Assignment"', False)

    def start():
        return cf.build_context_fit_plan(None, core, {}, preferred_mode="max", route_resolver=cf.resolve_context_fit_route)

    def accept(plan, tokens):  # the settled response's receipt names the serving account
        record_input_anchor({"_model_route": {"source": "codex", "model": "gpt-fixture", "accountFingerprint": f"fp-{catalog['account']}",
                                              "credentialProfileId": f"profile-{catalog['account']}"},
                             "_last_round_cache_usage": {"prompt_tokens": tokens, "provider": "claudexor"},
                             "_context_raw_input_tokens": tokens}, plan, round_idx=1)

    estimate = cf.estimate_context_prompt_tokens

    def refused(plan, size):  # the pre-send gate on a request of ``size`` estimated tokens
        monkeypatch.setattr(cf, "estimate_context_prompt_tokens", lambda *_a, **_kw: size)
        measured = cf.measure_main_fit(plan, [], [], profile="owner_max", rendered_mode="max", round_id="r")
        monkeypatch.setattr(cf, "estimate_context_prompt_tokens", estimate)
        return main_context_fit_refusal(AttemptRequest(model=model, provider="claudexor",
            prompt_tokens_bounded_estimate=size, physical_context=cf._physical_context_for_fit(measured))) is not None

    first = start()
    assert first.window_record.usable == 806_464
    accept(first, 921_227)
    same = start()
    assert same.window_record.usable == 921_227 and same.window_record.accepted == 921_227
    assert (refused(same, 921_227), refused(same, 921_228)) == (False, True)
    assert first.window_witness_bound and same.model_route["accountFingerprint"] == "fp-a"
    catalog["account"] = "b"  # a second account of the same model inherits nothing
    other = start()
    assert other.window_record.accepted == 0 and other.window_record.usable == 806_464 and refused(other, 921_227)
    catalog.update(account="c", window=1_000_000)  # an acceptance under a stated total keeps T - R
    accept(start(), 200_000)
    assert start().window_record.usable == 934_464 and start().window_record.accepted == 200_000


def test_observation_writes_bound_the_frontier_without_changing_resolution(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    monkeypatch.setattr(ce, "utc_now_iso", lambda: now.isoformat())
    monkeypatch.setattr(ce, "utc_now", lambda: now)
    evidence = ce.CapabilityEvidence(1_000_000, "asserted", "user_setting", "r")
    # Simulate an old unbounded store. Reading preserves it; the next write compacts it.
    rows = [{"accepted": n, "refused": 0, "kind": "input", "ts": now.isoformat()}
            for n in range(100_000, 1_000_000, 100_000)]
    ce._save(tmp_path, {"main_windows": {"r": {"observations": rows}}})
    assert resolve_window(evidence, 65_536, root=tmp_path).accepted == 900_000
    assert len(ce._load(tmp_path)["main_windows"]["r"]["observations"]) == 9
    for refusal in range(200_000, 500_000, 10_000):
        record_observation(tmp_path, "r", refused=refusal)
        assert resolve_window(evidence, 65_536, root=tmp_path).usable == refusal - 65_536
        record_observation(tmp_path, "r", accepted=refusal - 1)
        assert resolve_window(evidence, 65_536, root=tmp_path).source.startswith("refusal")
        record_observation(tmp_path, "r", accepted=refusal + 1)
        result = resolve_window(evidence, 65_536, root=tmp_path)
        assert result.usable == 934_464 and result.accepted == 900_000
        frontier = ce._load(tmp_path)["main_windows"]["r"]["observations"]
        assert len(frontier) == 3 and frontier[0] == rows[-1]
    record_observation(tmp_path, "r", refused=400_000)
    assert resolve_window(evidence, 65_536, root=tmp_path).usable == 334_464
    assert resolve_window(replace(evidence, route_fp="other"), 65_536, root=tmp_path).usable == 934_464
    now += timedelta(seconds=ce._CONFIRMED_TTL_SEC + 1)
    assert resolve_window(evidence, 65_536, root=tmp_path).usable == 934_464
