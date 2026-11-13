"""Planning evidence is exact-route, and I/T/O/R are independent facts."""
from dataclasses import replace

import pytest

from ouroboros import capability_evidence as ce
from ouroboros.context_window import WindowLimit, WindowRecord, record_observation, resolve_window


@pytest.mark.parametrize("input_limit,total,expected", [(200_000, None, 200_000),
    (None, 200_000, 134_464), (100_000, 200_000, 100_000)])
def test_reserve_is_charged_once_only_to_a_total(input_limit, total, expected):
    record = WindowRecord("r", WindowLimit(input_limit, "input", "metadata") if input_limit else None,
                          WindowLimit(total, "total", "metadata") if total else None, reserve=65_536)
    assert record.usable == expected
    assert record.total_equivalent - record.reserve == expected
    assert WindowRecord("unknown", reserve=65_536).usable is None


@pytest.mark.parametrize("source,refusal,admitted", [
    ("metadata", None, 991_808),           # an unconfirmed total: the stated number less the reply floor
    ("metadata", "total", 934_464),        # the provider refused a request as a total: the whole reserve, for its life
    ("metadata", "total+", 934_464),       # its own total is a little above the stated one: the reply is proved inside
    ("metadata", "input", 991_808),        # «prompt is too long» at the stated number: the favourable reading stands
    ("metadata", "input+", 991_808),       # an input refusal above the stated number names no limit under it
    ("metadata", "total,input", 934_464),  # a later refusal of the other kind leaves the total proved for its life
    ("local_health", None, 934_464),       # a local server's one buffer holds the reply too
])
def test_a_request_under_way_reads_a_total_favourably_until_a_refusal_confirms_it(tmp_path, source, refusal, admitted):
    """Two readings of one record: ``usable`` sizes a work's first start (a total always leaves the whole reserve);
    ``admitted`` is the line every question about a request that exists reads, in a round's frame as
    ``admitted_equivalent``. They differ only for a stated total nothing has proved to hold the reply."""
    evidence = ce.CapabilityEvidence(1_000_000, "confirmed", source, "r")
    for kind in (refusal or "").split(",") if refusal else ():  # at the stated number itself, or (+) at the server's
        record_observation(tmp_path, "r", refused=1_048_576 if kind.endswith("+") else 1_000_000,  # own a little above
                           kind=kind.rstrip("+"))
    record = resolve_window(evidence, 65_536, root=tmp_path)
    assert record.usable == 934_464 and record.total_equivalent == 1_000_000  # a start keeps the reserve
    assert record.admitted == admitted and record.admitted_equivalent == admitted + 65_536
    # A refusal at the stated number replaces its source; one above it leaves the stated limit as it was declared.
    assert (record.total.source == "refusal") is (refusal in ("total", "total,input")) and record.total.tokens == 1_000_000
    assert resolve_window(evidence, 8_192, root=tmp_path).admitted == 991_808  # Nano's reserve is the floor: one line
    record_observation(tmp_path, "r", accepted=1_000_001)  # a later larger acceptance disproves the stated total
    # (the server's own larger total, where a refusal named one, still holds the reply: 1,048,576 less the reserve)
    assert resolve_window(evidence, 65_536, root=tmp_path).admitted == (983_040 if refusal == "total+" else 1_000_001)
    bare = WindowRecord("unknown", reserve=65_536)
    assert bare.admitted is None and bare.admitted_equivalent == 0
    # A record built without a floor (an input limit's twin in tests, a projection) charges a total the whole reserve.
    assert WindowRecord("r", total=WindowLimit(300_000, "total", "metadata"), reserve=65_536).admitted == 234_464


def test_a_refusal_of_the_other_kind_stays_only_while_no_larger_acceptance_answered_it(tmp_path):
    """The record keeps the newest refusal of each kind, each with the first later acceptance above it. A refusal so
    answered binds nothing: a later refusal of the other kind does not bring its limit back, and an acceptance answers
    the refusal it exceeds whichever kind was refused last."""
    evidence = ce.CapabilityEvidence(1_000_000, "confirmed", "metadata", "r")
    record_observation(tmp_path, "r", accepted=900_000)  # the largest acceptance is older than every refusal below
    record_observation(tmp_path, "r", refused=200_000, kind="input")  # a numberless refusal at its measured size
    assert resolve_window(evidence, 65_536, root=tmp_path).usable == 200_000
    record_observation(tmp_path, "r", accepted=250_000)  # answered: the route took a larger input since
    assert resolve_window(evidence, 65_536, root=tmp_path).usable == 934_464
    record_observation(tmp_path, "r", refused=1_000_000, kind="total")
    record = resolve_window(evidence, 65_536, root=tmp_path)
    assert (record.usable, record.admitted, record.input) == (934_464, 934_464, None)
    record_observation(tmp_path, "r", refused=300_000, kind="input")  # unanswered: it stands beside the proved total
    record = resolve_window(evidence, 65_536, root=tmp_path)
    assert record.usable == 300_000 and record.total.source == "refusal"
    # An acceptance that is neither a new maximum nor above the newest refusal still answers the refusal it exceeds.
    record_observation(tmp_path, "r", refused=1_000_000, kind="total")  # the total's refusal is the newest again
    record_observation(tmp_path, "r", accepted=350_000)
    record = resolve_window(evidence, 65_536, root=tmp_path)
    assert (record.usable, record.input) == (934_464, None) and record.total.source == "refusal"
    # The other order: a total refused below the stated number, then an input refusal above it, then an acceptance
    # between the two. The total's refusal is answered; the input's stands.
    other = replace(evidence, route_fp="o")
    record_observation(tmp_path, "o", accepted=900_000)
    record_observation(tmp_path, "o", refused=300_000, kind="total")
    record_observation(tmp_path, "o", refused=600_000, kind="input")
    assert resolve_window(other, 65_536, root=tmp_path).admitted == 234_464
    record_observation(tmp_path, "o", accepted=350_000)
    record = resolve_window(other, 65_536, root=tmp_path)
    assert (record.admitted, record.input.tokens, record.total.source) == (600_000, 600_000, "metadata")


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


def test_numberless_refusal_bounds_the_window_only_at_a_measured_size(tmp_path, monkeypatch):
    """Codex refuses by size without a number: a measured refused size lowers a window an acceptance raised. A refusal
    of one input string's length is cured by a smaller input too, but bounds no token window at any size."""
    from types import SimpleNamespace
    from ouroboros.context_anchor import record_refused_input

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    plan = SimpleNamespace(route_fp="r", window_witness_bound=True, provider="openai", model="model")
    evidence = ce.CapabilityEvidence(872_000, "confirmed", "metadata", "r")
    record_observation(tmp_path, "r", accepted=919_379, reserve=65_536, transmitted=65_536)
    assert resolve_window(evidence, 65_536, root=tmp_path).usable == 919_379
    usage = {"_last_llm_provider_message": "Codex reported a failed response."}
    record_refused_input(usage, plan, 911_000)  # a cold estimate measures nothing beyond this task
    assert resolve_window(evidence, 65_536, root=tmp_path).usable == 919_379
    record_refused_input({**usage, "_last_llm_provider_code": "string_above_max_length"}, plan, 911_000, measured=True)
    assert resolve_window(evidence, 65_536, root=tmp_path).usable == 919_379
    record_refused_input(usage, plan, 911_000, measured=True)
    window = resolve_window(evidence, 65_536, root=tmp_path)
    assert (window.usable, window.source.split()[0]) == (911_000, "refusal")
    assert resolve_window(replace(evidence, route_fp="other"), 65_536, root=tmp_path).total.tokens == 872_000


@pytest.mark.parametrize("size,kind,admitted", [
    (240_000, "total", 234_464),  # sent inside the reply reservation and refused: the stated total holds the reply too
    (230_000, "input", 230_000),  # refused although it left the whole reserve: its size caps the input, as before
])
def test_a_numberless_refusal_inside_a_stated_totals_reservation_confirms_the_total(tmp_path, monkeypatch, size, kind, admitted):
    """The favourable reading sends a request inside the reply reservation of a stated total. A provider that counts
    the reply refuses it, and Codex names no number: the refusal still tells the kind, so every reading charges the
    reserve for that record's life and the route is not refused again at every wall."""
    from types import SimpleNamespace
    from ouroboros.context_anchor import record_refused_input

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    evidence = ce.CapabilityEvidence(300_000, "confirmed", "metadata", "r")
    stated = resolve_window(evidence, 65_536, root=tmp_path)
    assert stated.admitted == 291_808 and stated.total.source == "metadata"
    plan = SimpleNamespace(route_fp="r", window_witness_bound=True, provider="openai", model="model", window_record=stated)
    record_refused_input({"_last_llm_provider_message": "Codex reported a failed response."}, plan, size, measured=True)
    window = resolve_window(evidence, 65_536, root=tmp_path)
    assert window.total.tokens == 300_000 and (window.total.source == "refusal") is (kind == "total")
    assert (window.input.tokens if window.input else None) == (None if kind == "total" else size)
    assert window.admitted == admitted and window.usable == min(admitted, 234_464)


@pytest.mark.parametrize("source,anchored,size,probes", [
    ("metadata", True, 240_000, True),       # inside the reply reservation of a total nothing has proved
    ("metadata", True, 291_808, True),       # up to the admitted line itself
    ("metadata", True, 234_464, False),      # at the line a start is sized to: no reading is being tried
    ("metadata", False, 240_000, False),     # a cold estimate measures nothing: its refusal is read by its words
    ("local_health", True, 240_000, False),  # a total that holds the reply has no such band
])
def test_a_request_inside_the_reply_reservation_is_known_as_one(tmp_path, monkeypatch, source, anchored, size, probes):
    """What the recovery gate asks of a refused round (``loop_model_call._call_round_model``): a provider's size
    refusal of a request only the favourable reading admitted is that stated total's own answer, whatever its words
    (owner, 30 Sep 2026: the host sends once and records the provider's answer). Any other request's size refusal
    keeps the kind its words gave it."""
    from ouroboros import context_fit as cf
    from ouroboros.context_anchor import InputAnchor
    from tests.test_memory_window_acceptance import _core

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr(ce, "resolve_main_token_density", lambda *_a, **_kw: (1.0, "cold_estimate"))
    core, _snap = _core(True)
    plan = cf.build_context_fit_plan(None, core, {"usage_category": "consciousness"}, preferred_mode="max",
        route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "test"},
            ce.CapabilityEvidence(300_000, "confirmed", source, "probe-route")))
    monkeypatch.setattr(cf, "estimate_context_prompt_tokens", lambda *_a, **_kw: size)
    anchor = InputAnchor(input_tokens=size, estimate_tokens=size, transcript_tokens=size, round=1) if anchored else None
    fit = cf.measure_main_fit(plan, [], [], profile="owner_max", rendered_mode="max", round_id="r", anchor=anchor).measurement
    assert fit.estimated_input_tokens == size and fit.probes_reply_reservation is probes
    # The gate itself: a size-worded refusal of the request this route was sent, never another route's or another kind.
    from types import SimpleNamespace
    from ouroboros.context_anchor import reservation_refusal

    sent = lambda route: SimpleNamespace(physical_context=SimpleNamespace(route_fp=route))
    assert reservation_refusal({"_last_llm_error_kind": "request_too_large"}, sent(fit.route_fp), fit) is probes
    assert reservation_refusal({"_last_llm_error_kind": "request_too_large"}, sent("another-route"), fit) is False
    assert reservation_refusal({"_last_llm_error_kind": "bad_request"}, sent(fit.route_fp), fit) is False


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
    now += timedelta(seconds=2)  # the stated total returns, not refused: a round reads it less the reply floor
    assert capacity() == (metadata - 8_192 + 65_536 if metadata else None)


@pytest.mark.parametrize("stated,accepted,usable,admitted", [
    (872_000, 921_227, 921_227, 921_227), (1_000_000, 200_000, 934_464, 991_808),
    (1_000_000, 950_000, 934_464, 991_808), (0, 921_227, None, None),
])
def test_start_and_in_task_admission_share_the_input_bound(tmp_path, monkeypatch, stated, accepted, usable, admitted):
    """A work's first start keeps ``usable`` (a stated total less the reply reserve). A round shares an input bound
    whole and reads an unconfirmed total most favourably: its frame, its predicted miss and the admission of the final
    candidate are one line, the stated number less the reply floor (owner, 30 Sep 2026: the host acts before a send
    only when the request does not fit under the most favourable reading)."""
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
    for size in ([880_000, 921_227, 921_228] if stated == 872_000 else [900_000, 950_000, 991_808, 991_809]):
        monkeypatch.setattr(cf, "estimate_context_prompt_tokens", lambda *_a, **_kw: size)
        disposition = cf.measure_main_fit(plan, [], [], profile="owner_max", rendered_mode="max", round_id="r",
                                          accepted_input_tokens=accepted)
        physical = cf._physical_context_for_fit(disposition)
        refusal = main_context_fit_refusal(AttemptRequest(model="test", provider="test",
            prompt_tokens_bounded_estimate=size, physical_context=physical))
        fit = disposition.measurement
        assert physical.capacity_total_tokens == fit.capacity_total_tokens == (admitted + 65_536 if admitted else None)
        assert disposition.predicted_capacity_miss is (refusal is not None) is (admitted is not None and size > admitted)
        if refusal is not None:  # one line: what the check refuses is what the passes then measure
            assert refusal["predicted_free_tokens"] == refusal["free_tokens"] == admitted - size
            assert refusal["response_reserve_tokens"] == 65_536


@pytest.mark.parametrize(("mode", "kind", "account_refusal"), [
    ("nano", "input", False),
    ("nano", "unknown", False),
    ("low", "total", True),
])
def test_a_rebind_keeps_the_route_window_for_the_final_admission(tmp_path, monkeypatch, mode, kind, account_refusal):
    """3 Nov 2026 witness: a same-route rebind under Nano dropped an input-only 240K route's record and admitted
    297,344 input tokens until a provider refusal; Low kept it only because its reserve equals the ceiling.
    The record and its own frame survive in every mode: an input bound stays, the account scope's refusal still
    binds and an unknown window stays unknown. The round's frame and the admission's edge are the admitted line: the
    input bound whole, here the account's own refused 230,000 beside an unconfirmed 240,000 total."""
    from types import SimpleNamespace
    from ouroboros import context, context_fit as cf, loop
    from ouroboros.llm_attempt import main_context_fit_refusal
    from ouroboros.usage_accounting import AttemptRequest
    from tests.test_context_fit_integration import _plan, _projection

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr(ce, "resolve_main_token_density", lambda *_a, **_kw: (1.0, "cold_estimate"))
    evidence = (ce.CapabilityEvidence(0, "unprobeable", "none", "rebind-route") if kind == "unknown" else
                ce.CapabilityEvidence(240_000, "confirmed", "metadata", "rebind-route", window_kind=kind))
    if account_refusal:  # the serving account's own numbered input refusal, under its observation scope
        record_observation(tmp_path, "account-scope", refused=230_000, kind="input")
    model = "openai::rebind-fixture"
    monkeypatch.setattr(context, "_context_fit_route", lambda *_a, **_kw: (
        {"provider": "openai", "model": model, "window_observation_fp": "account-scope"}, evidence))
    start = resolve_window(evidence, 8_192 if mode == "nano" else 65_536, observation_fp="account-scope")
    plan = replace(_plan(preferred=mode, window=start.total_equivalent), model=model, route_fp="rebind-route",
                   status=evidence.status, window_record=start, nano_projection=_projection("nano"))
    tools = SimpleNamespace(_ctx=SimpleNamespace(task_metadata={}, _accumulated_usage={}))
    messages = plan.messages_for(mode)
    rebound, effective = loop._rebind_context_fit_plan(plan, tools, messages, model=model, use_local=False,
                                                       preferred_mode=mode, start_mode=mode, tool_schemas=[])
    ctx = SimpleNamespace(context_fit_plan=rebound, active_context_mode=effective, active_model=model, messages=messages,
                          tool_schemas=[], active_effort="", accumulated_usage={}, tools=tools, round_idx=1)
    disposition = loop._measure_round_main_fit(ctx, automatic_pass_used=False)
    fit, physical = disposition.measurement, loop._physical_context_for_fit(disposition)

    def admitted(size):
        return main_context_fit_refusal(AttemptRequest(model=model, provider="openai", prompt_tokens_bounded_estimate=size,
                                                       physical_context=physical)) is None

    edge = {"input": 240_000, "total": 240_000 - 8_192, "unknown": None}[kind]
    if account_refusal:
        edge = min(230_000, edge or 230_000)
    assert effective == mode and fit.response_reserve_tokens == (8_192 if mode == "nano" else 65_536)
    if edge is None:
        assert fit.capacity_total_tokens is None and admitted(10 ** 6)  # nothing invents a number
    else:
        assert rebound.window_record is not None and rebound.window_record.route_fp == "account-scope"
        assert rebound.window_record.usable == min(edge, 240_000 - fit.response_reserve_tokens if kind == "total" else edge)
        assert fit.capacity_total_tokens - fit.response_reserve_tokens == edge
        assert (admitted(edge), admitted(edge + 1)) == (True, False)


@pytest.mark.parametrize("model,key", [
    ("openai::model", "OPENAI_API_KEY"), ("anthropic::model", "ANTHROPIC_API_KEY"),
    ("deepseek::model", "DEEPSEEK_API_KEY"),
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
    record_input_anchor(usage, first, round_idx=1, estimate_tokens=700_000)
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
                             "_context_raw_input_tokens": tokens}, plan, round_idx=1, estimate_tokens=tokens)

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


@pytest.mark.parametrize("pin", ["", "profile-a"])
def test_an_answer_naming_no_route_writes_only_to_a_pinned_account(tmp_path, monkeypatch, pin):
    """H8 (W2, 12.11.2026): a receipt naming no route at all (an older engine) does not say which account served an
    Auto request, so no acceptance or refusal lands on the account the plan was built on; a pin names its account."""
    from types import SimpleNamespace
    from ouroboros.context_window import observed_plan, record_refusal

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    plan = SimpleNamespace(provider="claudexor", model="claudexor::codex=gpt-fixture", route_fp="account-a",
                           window_witness_bound=True, account_pin=pin)
    usage = {"_model_route": {}, "_last_llm_provider_message": "maximum context length is 200000 tokens"}
    assert (observed_plan(plan, usage) is plan) is bool(pin)
    record_refusal(usage, plan)
    assert bool(ce._load(tmp_path).get("main_windows", {}).get("account-a")) is bool(pin)


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
