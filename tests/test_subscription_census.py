"""The subscription census (``ouroboros/subscription_census.py``): the engine's verdicts on the install's subscription
accounts counted per family, the writer that keeps them as one dated file, and the requests that wake it.

Each guard is pinned both ways at its own seam. The round's part (who receives it, change-only against the facts
rows in view) is pinned in tests/test_round_facts_line.py, the round facts row's home test.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

from ouroboros import claudexor_daemon
from ouroboros import subscription_census as census
from ouroboros.gateway import claudexor_accounts, claudexor_passive

# The owner's install read at 21.11.2026 13:05:21Z (spec/census_v2_live.json), account ids anonymized.
NOW = 1795266321.546805
SCOPE = {"model": None, "requestKind": "text", "basis": "quota_block"}
WEEK, FIVE_HOURS = 604_800, 18_000
CLAUDE_RESETS = (1795320000.368309, 1795330799.802446, 1795489199.919261, 1795271400.337925, 1795798800.396706,
                 1795449599.976513, 1795528799.537178, 1795719600.209477, 1795838400.311705, 1795320000.023521,
                 1795813200.12108, 1795708799.889787, 1795586400.435241, 1795276799.928865, 1795392000.348527,
                 1795680000.069161)
CODEX_RESETS = (1795608236.0, 1795284968.0, 1795339282.0, 1795414748.0, 1795338217.0, 1795414820.0, 1795414981.0,
                1795338697.0, 1795550368.0, 1795601827.0, 1795555210.0, 1795338223.0, 1795546274.0, 1795546565.0)
LIVE_LINE = (
    "(subscription accounts, engine verdicts, unchanged since 21.11 13:05Z): agy 1 available (no measured general "
    "window), claude-sonnet-4-6,claude-opus-4-6-thinking,gpt-oss-120b-medium spent on 1 (earliest reset 25.11 16:36Z); "
    "claude 1 with room (7d 88% used, 1 of 1 measured), 16 at limit (earliest reset 21.11 14:30Z), 2 verification "
    "failed; codex 2 with room (7d 65–76% used, 2 of 2 measured), 14 at limit (earliest reset 21.11 18:16Z), 6 not "
    "freshly read; cursor 1 no quota source, 1 not freshly read, gemini-4-carbon-high spent on 1 (earliest reset "
    "02.12 00:00Z, seen 21.11 09:18Z)")


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def _profile(harness, profile_id, *, kind="config_dir_login", enabled=True, **status):
    return {"profile": {"harness_id": harness, "profile_id": profile_id, "enabled": enabled, "credential_kind": kind},
            "status": status}


def _row(harness, profile_id, *, state="available", freshness="fresh", windows=(), resets_at=None, scoped=(),
         route="vendor_native", observed=NOW - 60):
    """One quota snapshot as the engine serves it; ``windows`` are ``(ratio, seconds[, extra fields])``."""
    constraints = [{"id": f"w{index}", "used_ratio": window[0], "window_seconds": window[1], "freshness": freshness,
                    **(window[2] if len(window) > 2 else {})} for index, window in enumerate(windows)]
    return {"subject": {"harness": harness, "credential_route": route, "plan_label": None, "subject_id": profile_id},
            "source": "fixture", "observed_at": _iso(observed), "freshness": freshness, "constraints": constraints,
            "availability": {"state": state, "blocking_constraints": [], "resets_at": resets_at,
                             "model_scoped_exhaustions": list(scoped)}}


def _absence(harness, profile_id, reason):
    return {"subject": {"harness": harness, "credential_route": "vendor_native", "subject_id": profile_id},
            "reason": reason}


def _payload(profiles, rows=(), absences=(), scope=SCOPE, reads=("ok", "ok")):
    payload = {"view": "quota", "profiles": {"profiles": list(profiles), "accountPools": []}, "quota": list(rows),
               "quota_absences": list(absences), "reads": {"catalog": "not_read", "accounts": reads[0], "quota": reads[1]},
               "read_errors": {}, "timings_ms": {}}
    if scope is not None:
        payload["availability_scope"] = scope
    return payload


def _live_payload():
    """The facts of the owner's 21.11 13:05Z read, rebuilt row by row with anonymous ids."""
    profiles, rows, absences = [], [], []

    def add(harness, profile_id, *row_args, status=None, **row_fields):
        profiles.append(_profile(harness, profile_id, **(status or {})))
        if row_args or row_fields:
            rows.append(_row(harness, profile_id, *row_args, **row_fields))

    add("claude", "claude-room", windows=[(0.88, WEEK)])
    for index, reset in enumerate(CLAUDE_RESETS):
        add("claude", f"claude-{index}", state="exhausted", resets_at=_iso(reset), windows=[(1.0, WEEK)])
    for index in range(2):  # the engine's vendor verdict says these logins failed, whatever their quota says
        add("claude", f"claude-login-{index}", status={"verification": "failed", "verification_source": "vendor"},
            windows=[(0.2, WEEK)])
    for index, share in enumerate((0.65, 0.76)):
        add("codex", f"codex-room-{index}", windows=[(share, WEEK)])
    for index, reset in enumerate(CODEX_RESETS):
        add("codex", f"codex-{index}", state="exhausted", resets_at=_iso(reset), windows=[(1.0, WEEK)])
    for index in range(6):
        add("codex", f"codex-stale-{index}", freshness="stale", windows=[(0.37, WEEK)])
    add("agy", "agy-1", windows=[(0.5, FIVE_HOURS, {"applies_to_models": ["gemini-3-pro"]})], scoped=[{
        "applies_to_models": ["claude-sonnet-4-6", "claude-opus-4-6-thinking", "gpt-oss-120b-medium"],
        "resets_at": "2026-11-25T16:36:00.000000000Z"}])  # a nine-digit fraction, parsed the core's way
    add("cursor", "cursor-1", freshness="stale", observed=1795252680.0, scoped=[{
        "applies_to_models": ["gemini-4-carbon-high"], "resets_at": "2026-12-02T00:00:00Z"}])
    absences.append(_absence("cursor", "cursor-1", "no_source"))
    add("cursor", "cursor-2")
    profiles.append(_profile("openrouter", "paid-key", kind="api_key"))
    return _payload(profiles, rows, absences)


def _text(families, **fields):
    return census.census_text({"type": census.CENSUS_TYPE, "read": "ok", "code": None, "families": families,
                               "since": "2026-11-21T13:05:21+00:00", "confirmed_at": "2026-11-21T13:05:21+00:00",
                               **fields})


def _body(payload, now=NOW):
    return census.body_text(census.build_census(payload, now))


# ---- the count -----------------------------------------------------------------------------------------------------

def test_only_the_cores_subscription_kinds_are_counted():
    from ouroboros.gateway.onboarding import _SUBSCRIPTION_CREDENTIAL_KINDS

    assert _SUBSCRIPTION_CREDENTIAL_KINDS == {"config_dir_login", "oauth_token"}  # the engine's enum minus api_key
    profiles = [_profile("claude", "login", kind="config_dir_login"), _profile("claude", "token", kind="oauth_token"),
                _profile("claude", "key", kind="api_key"), _profile("claude", "unknown", kind="future_kind")]
    rows = [_row("claude", name, windows=[(0.5, WEEK)]) for name in ("login", "token", "key", "unknown")]
    assert _body(_payload(profiles, rows)) == "claude 2 with room (7d 50% used, 2 of 2 measured)"
    assert census.build_census(_payload([_profile("openrouter", "key", kind="api_key")]), NOW) == {}


def test_a_paid_key_route_of_the_same_id_never_gives_a_subscription_account_room():
    """The matrix's ``same_id_different_credential_route``: the core alone calls the subject available through its
    paid route; counted on the subscription route only, the account is at the limit."""
    rows = [_row("codex", "a", state="exhausted", resets_at="2026-11-22T12:00:00Z", windows=[(1.0, FIVE_HOURS)]),
            _row("codex", "a", route="managed_api_key", windows=[(0.2, FIVE_HOURS)])]
    assert _body(_payload([_profile("codex", "a")], rows)) == "codex 1 at limit (earliest reset 22.11 12:00Z)"
    assert _body(_payload([_profile("codex", "a")], rows[1:])) == "codex 1 not freshly read"


@pytest.mark.parametrize("verification,failed", [("failed", True), ("expired", True), ("passed", False),
                                                 ("not_run", False), ("", False), (None, False)])
def test_verification_failed_is_the_accounts_rule_and_beats_a_quota_with_room(verification, failed):
    assert census.verification_failed({"verification": verification}) is failed
    payload = _payload([_profile("claude", "a", verification=verification)], [_row("claude", "a", windows=[(0.3, WEEK)])])
    assert _body(payload) == ("claude 1 verification failed" if failed
                              else "claude 1 with room (7d 30% used, 1 of 1 measured)")


@pytest.mark.parametrize("rows,absences,expected", [
    # one subject's simultaneous rows are constraints: a fresh block beside a stale free row is a block
    ([_row("claude", "a", freshness="stale", windows=[(0.1, WEEK)]),
      _row("claude", "a", state="exhausted", resets_at="2026-11-22T09:00:00Z", windows=[(1.0, FIVE_HOURS)])],
     (), "claude 1 at limit (earliest reset 22.11 09:00Z)"),
    # a stale row whose block is still ahead stays at the limit: the engine keeps that cap
    ([_row("claude", "a", state="exhausted", freshness="stale", resets_at="2026-11-22T09:00:00Z")], (),
     "claude 1 at limit (earliest reset 22.11 09:00Z)"),
    # a block whose reset has passed is no live block
    ([_row("claude", "a", state="exhausted", resets_at="2026-11-21T09:00:00Z")], (), "claude 1 not freshly read"),
    ([_row("claude", "a", freshness="stale", windows=[(0.1, WEEK)])], (), "claude 1 not freshly read"),
    ((), (), "claude 1 not freshly read"),
    # the engine says available, but no finite general text window measured it: model-scoped, image-only, no ratio
    ([_row("claude", "a", windows=[(0.4, WEEK, {"applies_to_models": ["fable"]})])], (),
     "claude 1 available (no measured general window)"),
    ([_row("claude", "a", windows=[(0.4, WEEK, {"applies_to_request_kinds": ["image"]})])], (),
     "claude 1 available (no measured general window)"),
    ([_row("claude", "a", windows=[(None, WEEK)])], (), "claude 1 available (no measured general window)"),
    # an engine absence beside a fresh available row keeps the verdict unknown
    ([_row("claude", "a", windows=[(0.4, WEEK)])], [_absence("claude", "a", "poll_paced")], "claude 1 not freshly read"),
    # no quota source: the engine's no_source absence and no fresh row (Cursor)
    ((), [_absence("cursor", "a", "no_source")], "cursor 1 no quota source"),
    ([_row("cursor", "a", freshness="stale")], [_absence("cursor", "a", "no_source")], "cursor 1 no quota source"),
    ([_row("cursor", "a")], [_absence("cursor", "a", "no_source")], "cursor 1 not freshly read"),
    ((), [_absence("cursor", "a", "refresh_failed")], "cursor 1 not freshly read"),
])
def test_each_profile_takes_the_engines_verdict_then_its_unknown_by_cause(rows, absences, expected):
    harness = expected.split(" ", 1)[0]
    assert _body(_payload([_profile(harness, "a")], rows, absences)) == expected


def test_disabled_profiles_and_rows_of_no_named_subject_are_not_counted():
    rows = [_row("codex", "off", windows=[(0.2, WEEK)]), _row("codex", "on", windows=[(0.3, WEEK)])]
    null_subject = {**_row("codex", "x", windows=[(0.1, WEEK)]), "subject": {"harness": "codex",
                                                                              "credential_route": "vendor_native",
                                                                              "subject_id": None}}
    profiles = [_profile("codex", "off", enabled=False), _profile("codex", "on")]
    assert _body(_payload(profiles, [*rows, null_subject])) == "codex 1 with room (7d 30% used, 1 of 1 measured)"
    assert _body(_payload([_profile("codex", "on", enabled=True)], [null_subject])) == "codex 1 not freshly read"


def test_an_unhonored_scope_is_no_census():
    profiles, rows = [_profile("claude", "a")], [_row("claude", "a", windows=[(0.3, WEEK)])]
    assert census.build_census(_payload(profiles, rows), NOW) is not None
    for scope in (None, {"model": "gpt", "requestKind": "text", "basis": "quota_block"},
                  {"model": None, "requestKind": "image", "basis": "quota_block"}):
        assert census.build_census(_payload(profiles, rows, scope=scope), NOW) is None


def test_model_scoped_blocks_count_once_per_profile_with_their_earliest_reset_and_stale_ones_say_when_seen():
    fable = {"applies_to_models": ["fable"], "applies_to_model_prefixes": ["claude-fable-"]}
    rows = [
        _row("claude", "a", windows=[(0.3, WEEK)], scoped=[{**fable, "resets_at": "2026-11-23T03:00:00.5Z"},
                                                           {**fable, "resets_at": "2026-11-22T03:00:00Z"}]),
        _row("claude", "a", windows=[(0.4, WEEK)], scoped=[{**fable, "resets_at": "2026-11-24T03:00:00Z"}]),
        _row("claude", "b", state="exhausted", resets_at="2026-11-25T00:00:00Z",
             scoped=[{**fable, "resets_at": "2026-11-21T10:00:00Z"},  # passed: dropped
                     {"applies_to_models": ["opus"], "resets_at": "not an instant"}]),  # malformed: dropped
        _row("claude", "c", freshness="stale", observed=NOW - 7200, scoped=[{**fable, "resets_at": "2026-11-21T20:00:00.1Z"}]),
    ]
    profiles = [_profile("claude", name) for name in "abc"]
    body = _body(_payload(profiles, rows))
    assert body.endswith("fable,claude-fable-* spent on 2 (earliest reset 21.11 20:00Z, seen 21.11 11:05Z)"), body
    assert body.count("spent on") == 1
    fresh_only = _body(_payload(profiles[:1], rows[:2]))
    assert fresh_only.endswith("fable,claude-fable-* spent on 1 (earliest reset 22.11 03:00Z)"), fresh_only
    assert "seen" not in fresh_only


def test_a_fresh_reading_of_a_block_wins_over_a_stale_one_of_the_same_models():
    block = {"applies_to_models": ["fable"]}
    rows = [_row("claude", "a", freshness="stale", observed=NOW - 3600, scoped=[{**block, "resets_at": "2026-11-22T01:00:00Z"}]),
            _row("claude", "a", windows=[(0.2, WEEK)], scoped=[{**block, "resets_at": "2026-11-22T05:00:00Z"}])]
    assert _body(_payload([_profile("claude", "a")], rows)).endswith("fable spent on 1 (earliest reset 22.11 05:00Z)")


@pytest.mark.parametrize("ratio,shown", [(0.9999, "99.99"), (0.996, "99.6"), (0.65, "65"), (0.57, "57"),
                                         (0.123456, "12.34"), (1.0, "100"), (0.0, "0"), (0.29, "29")])
def test_a_share_is_cut_to_two_decimals_never_rounded_up(ratio, shown):
    assert census._percent(ratio) == shown


def test_a_window_range_is_over_profiles_with_their_coverage():
    """One value per profile and window length (its highest fresh measured share): one account with two seven-day
    windows prints one share, and the coverage counts the accounts with room that measured that length."""
    rows = [_row("claude", "a", windows=[(0.162, WEEK), (0.824, WEEK), (0.78, FIVE_HOURS)]),
            _row("claude", "b", windows=[(0.5, WEEK)]),
            _row("claude", "c", windows=[(0.9, WEEK, {"freshness": "stale"}), (0.3, WEEK)])]
    profiles = [_profile("claude", name) for name in "abc"]
    assert _body(_payload(profiles, rows)) == "claude 3 with room (5h 78% used, 1 of 3 measured; 7d 30–82.4% used, 3 of 3 measured)"
    assert _body(_payload(profiles[:1], rows[:1])) == "claude 1 with room (5h 78% used, 1 of 1 measured; 7d 82.4% used, 1 of 1 measured)"


def test_the_census_is_json_safe_and_its_text_drops_empty_families_and_classes():
    families = census.build_census(_payload([_profile("codex", "a")], [_row("codex", "a", windows=[(0.5, WEEK)])]), NOW)
    assert json.loads(json.dumps(families)) == families
    assert families["codex"]["windows"] == [{"window_seconds": WEEK, "least": 0.5, "greatest": 0.5, "profiles": 1}]
    assert census.body_text(families) == "codex 1 with room (7d 50% used, 1 of 1 measured)"
    assert "at limit" not in census.body_text(families) and "not freshly read" not in census.body_text(families)


def test_the_live_shaped_read_renders_its_line():
    families = census.build_census(_live_payload(), NOW)
    assert set(families) == {"agy", "claude", "codex", "cursor"}  # the paid key's family is no subscription
    assert _text(families) == LIVE_LINE
    failed = _text(families, read="failed", code="daemon_unreachable", confirmed_at="2026-11-21T14:09:30+00:00")
    assert failed.startswith("(subscription accounts, engine not read since 21.11 14:09Z (daemon_unreachable); "
                             "values from 21.11 13:05Z): agy 1 available")
    assert failed.split("): ", 1)[1] == LIVE_LINE.split("): ", 1)[1]


# ---- the writer ----------------------------------------------------------------------------------------------------

@pytest.fixture
def engine(monkeypatch, tmp_path):
    """A provisioned engine behind stubs: the passive read answers ``state["payload"]``; nothing else is reached."""
    from types import SimpleNamespace

    from ouroboros import config

    state = {"payload": _live_payload(), "reads": 0, "revision": 0, "now": NOW}

    def read():
        state["reads"] += 1
        return state["payload"]

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(census, "time", SimpleNamespace(time=lambda: state["now"]))  # the read's instant: the fixture's
    monkeypatch.setattr(census, "_PASS", {"confirmed_at": None, "failures": 0})
    monkeypatch.setattr(claudexor_daemon, "owned_daemon_provisioned", lambda: True)
    monkeypatch.setattr(claudexor_passive, "_quota_payload", read)
    monkeypatch.setattr(claudexor_accounts, "account_read_revision", lambda: state["revision"])
    state["path"] = tmp_path / census.CENSUS_FILE
    return state


def _failed_payload(code):
    payload = _payload([], reads=("not_read", "not_read"))
    payload["read_errors"] = {"discovery": {"code": code}}
    return payload


def test_an_unchanged_pass_writes_nothing_and_a_changed_one_writes(engine):
    assert census.refresh() == "written"
    stored = census.read_file()
    assert census.census_text(stored).split("): ", 1)[1] == LIVE_LINE.split("): ", 1)[1]
    assert stored["since"] == stored["confirmed_at"] and stored["digest"] == census.census_signature(stored)
    before = (engine["path"].read_bytes(), engine["path"].stat().st_mtime_ns)
    assert census.refresh() == "unchanged" and engine["reads"] == 2
    assert (engine["path"].read_bytes(), engine["path"].stat().st_mtime_ns) == before
    engine["payload"]["quota"][0]["constraints"][0]["used_ratio"] = 0.9
    assert census.refresh() == "written" and engine["path"].read_bytes() != before[0]


def test_a_value_change_moves_since_and_a_recovery_keeps_it(engine, monkeypatch):
    assert census.refresh() == "written"
    first = census.read_file()
    engine["now"] += 600
    engine["payload"]["quota"][0]["constraints"][0]["used_ratio"] = 0.91  # the account with room spent more
    assert census.refresh() == "written"
    second = census.read_file()
    assert second["since"] != first["since"] and "7d 91% used" in census.census_text(second)
    engine["now"] += 600
    good = engine["payload"]
    engine["payload"] = _failed_payload("daemon_unreachable")
    assert [census.refresh(), census.refresh()] == ["failure_debounced", "failure_recorded"]
    failed = census.read_file()
    assert (failed["read"], failed["code"], failed["since"]) == ("failed", "daemon_unreachable", second["since"])
    assert failed["families"] == second["families"]  # the values stay
    engine["payload"] = _failed_payload("observation_read_timeout")
    assert census.refresh() == "failure_unchanged"  # the code alternates; it is printed, never compared
    assert census.read_file()["code"] == "daemon_unreachable"
    engine["now"] += 600
    engine["payload"] = good
    assert census.refresh() == "recovered"
    recovered = census.read_file()
    assert recovered["read"] == "ok" and recovered["since"] == second["since"] and recovered["confirmed_at"] != second["confirmed_at"]


def test_a_failed_read_is_recorded_only_on_the_second_failed_pass_in_a_row(engine):
    assert census.refresh() == "written"
    engine["payload"] = _failed_payload("daemon_unreachable")
    good = _live_payload()
    assert census.refresh() == "failure_debounced"
    engine["payload"] = good  # the engine was warming up: one failure records nothing
    assert census.refresh() == "unchanged"
    engine["payload"] = _failed_payload("daemon_unreachable")
    assert census.refresh() == "failure_debounced"
    assert census.read_file()["read"] == "ok"
    assert census.refresh() == "failure_recorded"
    assert census.read_file()["read"] == "failed"


def test_a_missing_descriptor_is_a_failed_read_that_keeps_the_values(engine, monkeypatch):
    assert census.refresh() == "written"
    values = census.read_file()["families"]
    monkeypatch.setattr(claudexor_daemon, "owned_daemon_provisioned", lambda: False)
    engine["payload"] = _failed_payload("daemon_not_discovered")
    assert [census.refresh(), census.refresh()] == ["failure_debounced", "failure_recorded"]
    stored = census.read_file()
    assert engine["path"].exists() and stored["families"] == values
    assert "engine not read since" in census.census_text(stored) and "(daemon_not_discovered)" in census.census_text(stored)


def test_a_never_provisioned_engine_is_never_read_and_never_gets_a_file(engine, monkeypatch):
    monkeypatch.setattr(claudexor_daemon, "owned_daemon_provisioned", lambda: False)
    monkeypatch.setattr(claudexor_passive, "_quota_payload", lambda: pytest.fail("no engine was ever provisioned"))
    assert census.refresh() == "no_engine" and not engine["path"].exists()
    monkeypatch.setattr(claudexor_daemon, "owned_daemon_provisioned", lambda: True)
    engine["payload"] = _failed_payload("daemon_unreachable")
    monkeypatch.setattr(claudexor_passive, "_quota_payload", lambda: engine["payload"])
    assert [census.refresh(), census.refresh()] == ["failure_debounced", "failure_without_values"]
    assert not engine["path"].exists()


def test_an_empty_subscription_roster_writes_a_none_record(engine):
    assert census.refresh() == "written"
    engine["payload"] = _payload([_profile("openrouter", "key", kind="api_key")])
    assert census.refresh() == "none"
    stored = census.read_file()
    assert engine["path"].exists() and stored["families"] == {} and stored["read"] == "ok"
    assert census.refresh() == "unchanged"


def test_an_unhonored_scope_writes_nothing(engine):
    engine["payload"] = _payload([_profile("claude", "a")], [_row("claude", "a", windows=[(0.3, WEEK)])], scope=None)
    assert census.refresh() == "scope_unhonored" and not engine["path"].exists()


def test_a_confirmed_account_change_makes_the_file_pending_until_its_read_lands(engine):
    assert census.refresh() == "written"
    census.account_changed(True)  # an enable only asks for a read
    assert not census.read_file().get("pending") and census.request_mtime() is not None
    census.account_changed(False)  # a disable marks the census pending at once
    pending = census.read_file()
    assert pending["pending"] is True and pending["families"]
    assert census.refresh() == "confirmed"  # the same values: the pending mark goes, since stays
    assert not census.read_file().get("pending") and census.read_file()["since"] == pending["since"]
    census.account_changed(None)  # a removal too
    assert census.read_file()["pending"] is True


def test_a_read_crossed_by_a_confirmed_account_change_never_publishes(engine, monkeypatch):
    def crossed():
        engine["revision"] += 1  # a PATCH or DELETE confirmed while this read was in flight
        return engine["payload"]

    monkeypatch.setattr(claudexor_passive, "_quota_payload", crossed)
    assert census.refresh() == "roster_changed" and not engine["path"].exists()
    monkeypatch.setattr(claudexor_passive, "_quota_payload", lambda: engine["payload"])
    assert census.refresh() == "written"


def test_limit_failures_and_spent_windows_ask_for_a_read_and_other_failures_do_not(engine):
    for code in ("internal_error", "", None, "run_failed"):
        census.note_limit_failure(code)
    assert census.request_mtime() is None
    census.note_limit_failure("vendor_limit_rejected")
    assert census.request_mtime() is not None
    os.unlink(engine["path"].parent / "subscription_census.request")
    from ouroboros.gateways.claudexor import _window_exhausted_refusal

    assert _window_exhausted_refusal("credential_pool_exhausted", "pool", None) is None  # undated pool: structural
    assert census.request_mtime() is None
    assert _window_exhausted_refusal("subscription_window_exhausted", "spent", "2026-11-22T00:00:00Z") is not None
    request = json.loads((engine["path"].parent / "subscription_census.request").read_text(encoding="utf-8"))
    assert request["reason"] == "subscription_window_exhausted"


def test_the_duty_is_due_on_its_cadence_with_an_actor_or_soon_after_a_request(engine):
    start = 1_800_000_000.0
    assert census.duty_due(start, 0.0, has_actors=True)
    assert not census.duty_due(start, 0.0, has_actors=False)  # an idle install reads nothing
    assert not census.duty_due(start + 599, start, has_actors=True)
    assert census.duty_due(start + 600, start, has_actors=True)
    census.request_refresh("test")
    marked = census.request_mtime()
    assert not census.duty_due(marked + 5, marked - 1, has_actors=False)  # at most one pass per REQUEST_MIN_SEC
    assert census.duty_due(marked + 15, marked - 1, has_actors=False)
    assert not census.duty_due(marked + 30, marked + 1, has_actors=False)  # a pass after the request answered it


def test_the_census_duty_runs_one_pass_and_releases_its_latch(engine):
    import threading

    from ouroboros import server_maintenance as sm

    latch = threading.Lock()
    latch.acquire()
    sm._run_census_duty(None, latch)
    assert not latch.locked() and engine["reads"] == 1 and engine["path"].exists()
    stop = threading.Event()
    stop.set()
    latch.acquire()
    sm._run_census_duty(stop, latch)
    assert not latch.locked() and engine["reads"] == 1  # a closed generation reads nothing
    assert sm.host_duty("subscription-census").run is sm._run_census_duty


def test_an_unreadable_or_foreign_file_reads_as_no_census(engine):
    engine["path"].parent.mkdir(parents=True, exist_ok=True)
    for body in (b"{not json", b'{"type": "other", "read": "ok", "families": {}}', b"\xff\xfe", b"[]"):
        engine["path"].write_bytes(body)
        assert census._file_state()[0] == "unreadable" and census.read_file() is None
    engine["path"].unlink()
    assert census._file_state() == ("absent", None)
