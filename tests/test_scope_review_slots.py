"""The scope reviewer slots: window evidence, routes and identities.

Split by theme out of the original ``tests/test_scope_review.py`` giant. This
module owns the reviewer slot itself: the fail-closed reviewer window and its
five-way provenance wording, the scope slot route, the owner-only context mode,
per-row slot identities from the one mint, and the sourced capability evidence
that alone carries blocking authority.
"""

import json
import threading


def test_scope_reviewer_window_fail_closed_on_absent_evidence(monkeypatch, tmp_path):
    """claudexor B4 + v6.46.0 false-1M fix: with NO capability evidence an OFF-DEFAULT
    reviewer (e.g. an OUROBOROS_SCOPE_REVIEW_MODEL pin) fails closed to the conservative
    sub-floor SIZE, instead of silently treating a 200K model as 1M and overflowing its
    real window into a provider 400. The SHIPPED designated reviewer keeps the 1M
    sentinel as a SIZE so the review is still dispatched — but NEITHER carries blocking
    authority, because a model acquires no authority from its name (BIBLE P3: a window
    that cannot be established by sourced Capability Evidence is treated as too small)."""
    from ouroboros.tools import scope_review as sr
    from ouroboros import capability_evidence
    from types import SimpleNamespace

    # Isolated, empty evidence -> no model gets Capability Evidence.
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        capability_evidence,
        "probe",
        lambda *a, **k: SimpleNamespace(window_tokens=0),
    )

    # An OFF-DEFAULT reviewer with no evidence fails closed to the sub-floor...
    w_adv = sr._scope_window("gigachat::GigaChat-3-Ultra")
    assert 0 < w_adv.window_tokens < sr._SCOPE_MODEL_CONTEXT_WINDOW, w_adv

    # ...as does a pinned off-default 200K model (the v6.46.0 bug: it used to be
    # wrongly trusted as 1M and overflowed).
    w_offdefault = sr._scope_window("anthropic/claude-sonnet-4.5")
    assert w_offdefault.window_tokens == sr._SCOPE_FAILCLOSED_WINDOW, w_offdefault

    # The SHIPPED designated reviewer keeps the 1M sentinel as a SIZING number...
    w_designated = sr._scope_window(sr._SCOPE_MODEL_DEFAULT)
    assert w_designated.window_tokens == sr._SCOPE_MODEL_CONTEXT_WINDOW, w_designated

    # Direct-provider and explicit OpenRouter spellings of the same shipped reviewer
    # are also the designated default. Regression guard for a provider spelling
    # (openai::/openrouter::) being misclassified as off-default.
    for spelling in ("openai::gpt-5.6-terra", "openrouter::openai/gpt-5.6-terra"):
        assert sr._scope_window(spelling).window_tokens == sr._SCOPE_MODEL_CONTEXT_WINDOW

    # ...and NONE of them — the designated default least of all — may block a commit
    # on that invented number. Authority is computed from the evidence, not the name.
    for model in (
        "gigachat::GigaChat-3-Ultra", "anthropic/claude-sonnet-4.5",
        sr._SCOPE_MODEL_DEFAULT, "openai::gpt-5.6-terra",
        "openrouter::openai/gpt-5.6-terra",
    ):
        assert sr._scope_window(model).blocking_authority_allowed is False, model


def test_scope_reviewer_window_uses_scope_slot_route_not_main(monkeypatch, tmp_path):
    """Capability Evidence for scope review must use the scope slot's route.

    A local-routed main lane (`USE_LOCAL_MAIN=true`) must not turn a remote direct
    OpenAI scope reviewer into a local route lookup.
    """
    from types import SimpleNamespace
    from ouroboros import capability_evidence, config
    from ouroboros.tools import scope_review as sr

    captured = {}

    def fake_probe(drive_root, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(window_tokens=333_333)

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        config,
        "load_settings",
        lambda: {
            "USE_LOCAL_MAIN": True,
            "OPENAI_BASE_URL": "https://api.openai.test/v1",
        },
    )
    monkeypatch.setattr(capability_evidence, "probe", fake_probe)

    assert sr._scope_window("openai::gpt-5.5").window_tokens == 333_333
    assert captured["provider"] == "openai"
    assert captured["model"] == "openai::gpt-5.5"
    assert captured["base_url"] == "https://api.openai.test/v1"
    assert captured["use_local"] is False


def test_parallel_commit_scope_is_one_substantive_call(monkeypatch, tmp_path):
    """P3 wrapper must not fan a budget result into a second degraded call."""
    from types import SimpleNamespace

    from ouroboros import config
    from ouroboros.tools import parallel_review, review
    from ouroboros.tools.scope_review import ScopeReviewResult

    calls = []

    def fake_scope(_ctx, _message, **kwargs):
        calls.append((kwargs.get("scope_model"), kwargs.get("degraded", False)))
        return ScopeReviewResult(
            blocked=False,
            status="budget_exceeded",
            model_id=str(kwargs.get("scope_model") or ""),
        )

    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        task_id="one-pass-scope",
        _review_history=[],
        _review_advisory=[],
        _scope_review_history={},
    )
    monkeypatch.setattr(parallel_review, "run_cmd", lambda *_a, **_k: "staged diff")
    monkeypatch.setattr(parallel_review, "run_scope_review", fake_scope)
    monkeypatch.setattr(config, "get_scope_review_models", lambda: ["scope/model"])
    monkeypatch.setattr(review, "_run_unified_review", lambda *_a, **_k: None)

    parallel_review.run_parallel_review(ctx, "test commit")

    assert calls == [("scope/model", False)]


# --- v6.80.0: scope review follows the owner-only context mode -----------------

def test_low_context_mode_skips_scope_review_with_a_typed_evidence_row(monkeypatch, tmp_path):
    """RS2: in owner-selected `low` mode no reviewer is called, the commit is not
    gated on scope, and the skip leaves a TYPED durable row on the same
    review-evidence surface that carries fail-closed results — so a low-mode commit
    is never forensically confusable with "scope review silently failed" (BIBLE P1).

    The one-window provenance tombstone must be explicit `false` here: bare env
    Low remains effective sizing Low but resolves owner intent fail-closed to Max."""
    from ouroboros import config
    from ouroboros.tools import review_helpers
    from ouroboros.tools import scope_review as sr

    class _Ctx:
        repo_dir = str(tmp_path)
        task_id = "low-mode-skip"
        pending_events = []

        def drive_logs(self):
            return tmp_path

    called = []
    monkeypatch.setattr(sr, "_call_scope_llm", lambda *a, **k: called.append(1) or ("", None, ""))
    monkeypatch.setattr(sr, "_build_scope_prompt", lambda *a, **k: called.append(1) or ("p", None))
    monkeypatch.setattr(config, "get_context_mode", lambda: "low")
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE_AUTO_LOW", "false")

    result = sr.run_scope_review(_Ctx(), "test commit", scope_model="anthropic/claude-fable-5")

    assert called == [], "low mode must not call the reviewer or even assemble a prompt"
    assert result.blocked is False
    assert result.status == "skipped_low_context_mode"
    assert any(
        f.get("item") == "scope_review_skipped_low_context_mode"
        for f in result.advisory_findings
    )
    record = review_helpers.build_scope_actor_record(result, fallback_model_id="x")
    assert record["status"] == "skipped_low_context_mode"
    assert record["prompt_chars_source"] == "not_assembled"

    # max mode (the unchanged DEFAULT) still assembles and calls.
    monkeypatch.setattr(config, "get_context_mode", lambda: "max")
    sr.run_scope_review(_Ctx(), "test commit", scope_model="anthropic/claude-fable-5")
    assert called, "max mode must still run scope review"


def test_default_context_mode_is_max_and_agent_cannot_lower_it(monkeypatch):
    """RS2 anti-regression: the DEFAULT behaviour is unchanged (max ⇒ blocking scope
    gate), and the agent still cannot reach the setting that now also switches scope
    review off — on the settings merge, the shell guard, or the browser guard."""
    from ouroboros import config
    from ouroboros.gateway.settings import _merge_settings_payload
    from ouroboros.tools.browser import _blocks_context_mode_self_lowering_js
    from ouroboros.tools.registry_guard_process import _detect_context_mode_self_lowering

    assert config.SETTINGS_DEFAULTS["OUROBOROS_CONTEXT_MODE"] == "max"
    monkeypatch.delenv("OUROBOROS_CONTEXT_MODE", raising=False)
    assert config.get_context_mode() == "max"

    merged = _merge_settings_payload({"OUROBOROS_CONTEXT_MODE": "max"},
                                     {"OUROBOROS_CONTEXT_MODE": "low"})
    assert merged["OUROBOROS_CONTEXT_MODE"] == "max"
    assert _detect_context_mode_self_lowering(
        "save_settings({'ouroboros_context_mode': 'low'})"
    ) is True
    assert _blocks_context_mode_self_lowering_js(
        "fetch('/api/owner/context-mode', {body: JSON.stringify({mode: 'low'})})"
    ) is True


def test_window_provenance_wording_is_five_way():
    """RS5: the cases must read differently — a conservative fallback must not be
    reported with the same words as a confirmed measurement, and an EXPIRED record
    must not be reported with the same words as a live one."""
    from ouroboros.tools import scope_review as sr

    phrases = {
        sr._window_provenance_phrase(200_000, sr._WINDOW_CONFIRMED),
        sr._window_provenance_phrase(200_000, sr._WINDOW_ASSERTED),
        sr._window_provenance_phrase(200_000, sr._WINDOW_UNKNOWN),
        sr._window_provenance_phrase(1_000_000, sr._WINDOW_STALE),
        sr._window_provenance_phrase(1_000_000, sr._WINDOW_SENTINEL),
    }
    assert len(phrases) == 5
    assert "confirmed" in sr._window_provenance_phrase(200_000, sr._WINDOW_CONFIRMED)
    assert "owner-asserted" in sr._window_provenance_phrase(200_000, sr._WINDOW_ASSERTED)
    assert "unknown window" in sr._window_provenance_phrase(200_000, sr._WINDOW_UNKNOWN)
    assert "designated-default" in sr._window_provenance_phrase(1_000_000, sr._WINDOW_SENTINEL)
    assert "EXPIRED" in sr._window_provenance_phrase(1_000_000, sr._WINDOW_STALE)

    # The label is read off the EVIDENCE, so a stale 1M record can never be labelled
    # (or worded) as a confirmed one just because its number clears the floor.
    stale = sr.ReviewerWindow(1_000_000, "confirmed", stale=True)
    assert sr._scope_window_provenance(stale) == sr._WINDOW_STALE
    assert sr._scope_window_provenance(sr.ReviewerWindow(250_000)) == sr._WINDOW_UNKNOWN

# --- scope-slot identity: one owner, one id per configured row ----------------


def _run_scope_fanout(monkeypatch, tmp_path, models):
    """Run the parallel scope fan-out over ``models`` and collect every id surface.

    Returns (substrate_ids, actor_record_ids, manifest_ids): the ids the review
    substrate physically ran the rows under (sorted — the rows run concurrently,
    so completion order is not meaningful), the ids stamped on the durable actor
    records, and the ids in the scope context manifest.
    """
    from types import SimpleNamespace

    from ouroboros import config, review_substrate
    from ouroboros.tools import parallel_review, review
    from ouroboros.tools import scope_review as sr

    rows = [
        {
            "item": item,
            "verdict": "PASS",
            "severity": "advisory",
            "reason": "Concrete scope artifact was checked and passes.",
        }
        for item in sorted(sr._SCOPE_REQUIRED_ITEMS)
    ]
    substrate_ids: list = []
    lock = threading.Lock()

    def fake_run_review_request(request, *, slots, drive_root, llm, usage_ctx=None):
        with lock:
            substrate_ids.extend(slot.slot_id for slot in slots)
        return SimpleNamespace(actors=[{
            "slot_id": slots[0].slot_id,
            "model": slots[0].model,
            "status": "ok",
            "raw_text": json.dumps(rows),
            "usage": {},
            "prompt_ref": {},
            "response_ref": {},
        }])

    monkeypatch.setattr(config, "get_scope_review_models", lambda: list(models))
    monkeypatch.setattr(review_substrate, "run_review_request", fake_run_review_request)
    monkeypatch.setattr(sr, "_build_scope_prompt", lambda *a, **k: ("scope prompt", None))
    monkeypatch.setattr(sr, "_scope_window",
                        lambda _model, **_k: sr.ReviewerWindow(window_tokens=1_000_000, status="confirmed"))
    monkeypatch.setattr(parallel_review, "run_cmd", lambda *_a, **_k: "staged diff")
    monkeypatch.setattr(review, "_run_unified_review", lambda *_a, **_k: None)

    ctx = SimpleNamespace(
        repo_dir=tmp_path, drive_root=tmp_path, task_id="scope-slot-identity",
        pending_events=[], _review_history=[], _review_advisory=[], _scope_review_history={},
    )
    parallel_review.run_parallel_review(ctx, "identity commit")
    actor_ids = [str(r.get("slot_id") or "") for r in (ctx._last_scope_raw_results or [])]
    manifest = (ctx._last_scope_raw_result or {}).get("context_manifest") or {}
    manifest_ids = [str(a.get("slot_id") or "") for a in (manifest.get("actors") or [])]
    return sorted(substrate_ids), actor_ids, manifest_ids


def test_scope_rows_sharing_a_model_keep_distinct_identities(tmp_path, monkeypatch):
    """Duplicate model ids are valid independent slots (review_substrate contract,
    and get_scope_review_models preserves them on purpose). Naming a row after its
    model collapsed both rows onto one receipt id."""
    substrate_ids, actor_ids, manifest_ids = _run_scope_fanout(
        monkeypatch, tmp_path, ["model/a", "model/a"]
    )
    assert len(set(substrate_ids)) == 2, substrate_ids
    assert len(set(actor_ids)) == 2, actor_ids
    assert len(set(manifest_ids)) == 2, manifest_ids


def test_scope_rows_whose_models_sanitize_alike_keep_distinct_identities(tmp_path, monkeypatch):
    """Two DIFFERENT models can normalize to the same token (``openai::gpt-5`` and
    ``openai/gpt/5`` both sanitize to ``openai_gpt_5``), which merged two rows."""
    substrate_ids, actor_ids, manifest_ids = _run_scope_fanout(
        monkeypatch, tmp_path, ["openai::gpt-5", "openai/gpt/5"]
    )
    assert len(set(substrate_ids)) == 2, substrate_ids
    assert len(set(actor_ids)) == 2, actor_ids
    assert len(set(manifest_ids)) == 2, manifest_ids


def test_scope_row_identity_survives_editing_that_row_model(tmp_path, monkeypatch):
    """Editing a slot's model in the settings UI must not re-identify the slot:
    its receipts have to keep lining up with its own history."""
    before_substrate, before_actors, _ = _run_scope_fanout(
        monkeypatch, tmp_path, ["model/a", "model/b"]
    )
    after_substrate, after_actors, _ = _run_scope_fanout(
        monkeypatch, tmp_path, ["model/a", "model/EDITED"]
    )
    assert before_substrate == after_substrate, (before_substrate, after_substrate)
    assert before_actors == after_actors, (before_actors, after_actors)


def test_scope_actor_records_and_substrate_agree_on_one_identity(tmp_path, monkeypatch):
    """The durable actor record, the context manifest, and the substrate call that
    produced the prompt/response refs must name the SAME row. They were derived
    independently — positionally in the coordinator, from the model in the reviewer —
    so one row carried two disagreeing identities."""
    substrate_ids, actor_ids, manifest_ids = _run_scope_fanout(
        monkeypatch, tmp_path, ["model/a", "model/b"]
    )
    assert sorted(substrate_ids) == sorted(actor_ids) == sorted(manifest_ids), (
        substrate_ids, actor_ids, manifest_ids
    )
    # Pinned spelling: durable records written before v6.87.21 already carry these
    # ids, so historical receipts line up with new ones without a translation table.
    assert actor_ids == ["scope_slot_1", "scope_slot_2"], actor_ids


def test_scope_row_ids_come_from_the_one_mint(tmp_path, monkeypatch):
    """The coordinator must READ the row's id, not re-derive an identical string.

    parallel_review stamped ``scope_slot_{idx + 1}`` on the actor record and the
    manifest — byte-identical to the mint's output today, so nothing could tell
    the two apart. Repointing the ONE mint separates them: a surface that reads it
    follows, a surface that spells its own literal does not.
    """
    from ouroboros import review_substrate

    monkeypatch.setattr(
        review_substrate, "slot_id_for_row",
        lambda index, *, prefix=review_substrate.SLOT_ID_PREFIX: f"{prefix}_row{int(index)}",
    )
    substrate_ids, actor_ids, manifest_ids = _run_scope_fanout(
        monkeypatch, tmp_path, ["model/a", "model/b"]
    )
    expected = ["scope_slot_row1", "scope_slot_row2"]
    assert substrate_ids == expected, substrate_ids
    assert actor_ids == expected, actor_ids
    assert manifest_ids == expected, manifest_ids

# --- Blocking scope authority is a property of the EVIDENCE (v6.87.44) ----------

def _seed_scope_evidence(monkeypatch, tmp_path, model, *, window, status, ts, use_ack=False):
    """Write one Capability-Evidence record for ``model``'s real scope route."""
    import json as _json
    from ouroboros import capability_evidence as ce
    from ouroboros.reviewer_window import reviewer_route

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    provider, base_url = reviewer_route(model)
    fp = ce.route_fingerprint(provider=provider, base_url=base_url, model=model)
    store = tmp_path / "state" / "capability_evidence.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    key = "owner_acks" if use_ack else "probes"
    store.write_text(_json.dumps({key: {fp: {
        "window_tokens": window, "status": status, "source": "provider_metadata",
        "route_fp": fp, "model": model, "provider": provider, "ts": ts,
    }}}), encoding="utf-8")
    return fp


def test_stale_evidence_cannot_authorize_a_blocking_scope_verdict(monkeypatch, tmp_path):
    """BIBLE P3: blocking authority turns on SOURCED Capability Evidence, and an
    EXPIRED record that the probe could not re-verify is a dated impression, not a
    source. Before the typed result, `(window, status)` dropped `stale` on the floor,
    so a five-day-old 1M record kept across a provider outage read as `confirmed 1M`
    and signed the blocking verdict."""
    import datetime

    from ouroboros import capability_evidence as ce
    from ouroboros.reviewer_window import resolve_reviewer_window
    from ouroboros.tools import scope_review as sr

    model = "anthropic/claude-fable-5"
    old = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=5)).isoformat()
    _seed_scope_evidence(monkeypatch, tmp_path, model, window=1_000_000,
                         status="confirmed", ts=old)
    # The provider is unreachable now, so `probe` keeps the prior record — as STALE.
    monkeypatch.setattr(ce, "_provider_metadata_window", lambda *a, **k: 0)
    monkeypatch.setattr(ce, "_metadata_fetch_transport_failed", lambda *a, **k: True)

    resolved = resolve_reviewer_window(model)
    assert resolved.window_tokens == 1_000_000 and resolved.status == "confirmed"
    assert resolved.stale is True, "the outage-carried record must arrive marked stale"
    assert resolved.observed_at == old, "the observation time must survive the hand-off"
    assert resolved.blocking_authority_allowed is False

    # ...and the scope gate acts on it: criticals are preserved but demoted.
    critical = [{"item": "architecture_fit", "verdict": "FAIL",
                 "severity": "critical", "reason": "r"}]
    crit_out, adv_out, result = sr._apply_scope_authority(
        critical, [], scope_model_id=model, result_kwargs={},
    )
    assert crit_out == [] and result is not None and result.blocked is True
    assert result.status == "sub_floor"
    # The owner is told the window EXPIRED, not that it was "confirmed" — and WHEN it
    # was last confirmed, which is the difference between a blip and a dead route.
    assert "EXPIRED" in result.block_message
    assert f"last confirmed {old}" in result.block_message
    assert any("EXPIRED" in str(f.get("reason", "")) for f in adv_out)

    # A CURRENT record for the same route authorises normally — the fix rejects
    # staleness, not the route.
    fresh = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _seed_scope_evidence(monkeypatch, tmp_path, model, window=1_000_000,
                         status="confirmed", ts=fresh)
    assert resolve_reviewer_window(model).blocking_authority_allowed is True


def test_designated_default_gets_no_authority_from_its_name(monkeypatch, tmp_path):
    """A designated model does not acquire blocking authority from being designated.

    The sentinel still SIZES an unevidenced default at 1M (so the review is dispatched
    rather than declined before it starts), but sizing is not signing: with no sourced
    evidence the scope verdict is advisory, exactly as for any other unevidenced route.
    The same name-check used to disable the ONE lazy probe that could source the
    default's window, which is why it could never stop being invented."""
    from types import SimpleNamespace

    from ouroboros import capability_evidence as ce
    from ouroboros.tools import scope_review as sr

    fetches = []

    def fake_probe(_drive_root, **kw):
        fetches.append(bool(kw.get("allow_fetch")))
        return SimpleNamespace(window_tokens=0, status="unprobeable", source="none",
                               route_fp="fp", stale=False, ts="")

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    monkeypatch.setattr(ce, "probe", fake_probe)

    resolved = sr._scope_window(sr._SCOPE_MODEL_DEFAULT)
    assert resolved.window_tokens == sr._SCOPE_MODEL_CONTEXT_WINDOW  # sizing survives
    assert sr._scope_window_provenance(resolved) == sr._WINDOW_SENTINEL
    assert resolved.blocking_authority_allowed is False
    assert fetches == [True], "the default route must get the lazy probe like any other"

    critical = [{"item": "architecture_fit", "verdict": "FAIL",
                 "severity": "critical", "reason": "r"}]
    crit_out, _adv, result = sr._apply_scope_authority(
        critical, [], scope_model_id=sr._SCOPE_MODEL_DEFAULT, result_kwargs={},
    )
    assert crit_out == [] and result is not None and result.blocked is True

    # Owner-acking that exact route is what restores authority — evidence, not name.
    ce.record_owner_ack(tmp_path, provider="openrouter", model=sr._SCOPE_MODEL_DEFAULT,
                        window_tokens=1_050_000, note="test")
    monkeypatch.undo()
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    assert sr._scope_window(sr._SCOPE_MODEL_DEFAULT).blocking_authority_allowed is True


def test_concurrent_resolution_of_one_route_shares_one_probe(monkeypatch, tmp_path):
    """parallel_review runs the triad and the scope slots concurrently. Without the
    per-route lock two slots on the SAME route both reach the provider for a window
    the first one is already fetching; with it the second enters after the evidence
    has been stored and reads it back, so one route costs one metadata fetch."""
    import threading
    from types import SimpleNamespace

    from ouroboros import capability_evidence as ce
    from ouroboros.tools import scope_review as sr

    model = "anthropic/claude-fable-5"
    in_probe, release = threading.Event(), threading.Event()
    store: dict = {}   # stands in for capability_evidence.json, which the real probe writes
    fetches: list = []

    def fake_probe(_drive_root, **kw):
        # `probe` serves a CURRENT record straight from its cache without touching the
        # network whatever `allow_fetch` says; only an absent/expired one goes out.
        if "ev" in store:
            return store["ev"]
        if not kw.get("allow_fetch"):
            return SimpleNamespace(window_tokens=0, status="unprobeable", stale=False, ts="")
        fetches.append(str(kw.get("model") or ""))
        in_probe.set()
        release.wait(10)              # the network probe is still in flight
        store["ev"] = SimpleNamespace(
            window_tokens=1_000_000, status="confirmed", stale=False,
            ts="2026-08-02T00:00:00+00:00")
        return store["ev"]

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    monkeypatch.setattr(ce, "probe", fake_probe)
    monkeypatch.setattr("ouroboros.reviewer_window._LAZY_ROUTE_LOCKS", {})

    out = {}
    threads = [
        threading.Thread(target=lambda k=k: out.__setitem__(k, sr._scope_window(model)))
        for k in ("a", "b")
    ]
    threads[0].start()
    assert in_probe.wait(10), "the first thread never reached the probe"
    threads[1].start()
    threads[1].join(0.5)
    assert threads[1].is_alive(), (
        "the second thread must WAIT for the in-flight probe on its route"
    )
    release.set()
    for thread in threads:
        thread.join(10)

    assert fetches == [model], (
        f"one route must cost ONE metadata fetch; got {len(fetches)}"
    )
    assert out["a"].window_tokens == out["b"].window_tokens == 1_000_000
    assert out["a"].blocking_authority_allowed is out["b"].blocking_authority_allowed is True


def test_expired_evidence_is_re_sourced_instead_of_wedging_the_process(monkeypatch, tmp_path):
    """A long-lived process must be able to RE-confirm its scope reviewer.

    The lazy probe used to be memoised for the lifetime of the process while the
    evidence it produced expired after 24h, so a healthy, connected install that
    stayed up past the TTL read its own reviewer as EXPIRED on every later
    resolution: `blocking_authority_allowed` went False and stayed False, and
    `_apply_scope_authority` blocked EVERY commit for the rest of the process's
    life. How often a route may be re-probed is `capability_evidence.probe`'s TTL to
    decide — a second, never-expiring rate limit here could only ever wedge."""
    import datetime

    from ouroboros import capability_evidence as ce
    from ouroboros.reviewer_window import resolve_reviewer_window
    from ouroboros.tools import scope_review as sr

    model = "openai/gpt-5.6-terra"
    now = datetime.datetime.now(datetime.timezone.utc)
    _seed_scope_evidence(monkeypatch, tmp_path, model, window=1_050_000,
                         status="confirmed", ts=now.isoformat())
    # The provider is up the whole time: a metadata read returns the real window.
    monkeypatch.setattr(ce, "_provider_metadata_window", lambda *a, **k: 1_050_000)
    monkeypatch.setattr(ce, "_metadata_fetch_transport_failed", lambda *a, **k: False)

    assert resolve_reviewer_window(model).blocking_authority_allowed is True

    # ...25 hours later, in the SAME process: the one stored record has aged past the
    # 24h confirmed TTL. Nothing about the install changed.
    _seed_scope_evidence(monkeypatch, tmp_path, model, window=1_050_000, status="confirmed",
                         ts=(now - datetime.timedelta(hours=25)).isoformat())

    resolved = resolve_reviewer_window(model)
    assert resolved.stale is False, "an expired record must be RE-SOURCED, not read as expired"
    assert resolved.blocking_authority_allowed is True
    _crit, _adv, result = sr._apply_scope_authority(
        [{"item": "architecture_fit", "verdict": "FAIL",
          "severity": "critical", "reason": "r"}],
        [], scope_model_id=model, result_kwargs={},
    )
    assert result is None, "a healthy install must not block its own commits after 24h"
