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

# A live-shaped read: the owner's install at 21.11.2026 13:05:21Z, anonymized and kept whole in this file.
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
    "(subscription accounts, engine verdicts; counts unchanged since 21.11 13:05:21Z; shares read 21.11 13:05Z): agy 1 "
    "available (no measured general window), claude-opus-4-6-thinking,claude-sonnet-4-6,gpt-oss-120b-medium spent on "
    "1 (earliest reset 25.11 16:36Z); claude 1 with room (7d 88% used, 1 of 1 measured), 16 at limit (earliest reset "
    "21.11 14:30Z), 2 verification failed; codex 2 with room (7d 65–76% used, 2 of 2 measured), 14 at limit (earliest "
    "reset 21.11 18:16Z), 6 not freshly read; cursor 1 no quota source, 1 not freshly read, gemini-4-carbon-high spent "
    "on 1 (earliest reset 02.12 00:00Z, seen 21.11 09:18Z)")


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
    """The facts of the live-shaped read, rebuilt row by row with anonymous ids."""
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
                               "since": "2026-11-21T13:05:21+00:00", "read_at": "2026-11-21T13:05:21+00:00", **fields})


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


def test_model_scoped_blocks_count_once_per_profile_with_the_earliest_reset_and_the_seen_of_its_own_row():
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
    # The engine's prefixes name the list when it gives any (shorter than its model names); the earliest reset is a
    # stale row's, so its own observation time goes with it.
    assert body.endswith("claude-fable-* spent on 2 (earliest reset 21.11 20:00Z, seen 21.11 11:05Z)"), body
    assert body.count("spent on") == 1
    later_stale = [rows[0], _row("claude", "c", freshness="stale", observed=NOW - 7200,
                                 scoped=[{**fable, "resets_at": "2026-11-24T00:00:00Z"}])]
    body = _body(_payload([profiles[0], profiles[2]], later_stale))
    assert body.endswith("claude-fable-* spent on 2 (earliest reset 22.11 03:00Z)"), body  # a fresh reading: no seen
    named = [_row("claude", "a", scoped=[{"applies_to_models": ["opus", "fable"], "resets_at": "2026-11-22T03:00:00Z"}]),
             _row("claude", "b", scoped=[{"applies_to_models": ["fable", "opus"], "resets_at": "2026-11-22T04:00:00Z"}])]
    body = _body(_payload(profiles[:2], named))
    assert body.endswith("fable,opus spent on 2 (earliest reset 22.11 03:00Z)") and body.count("spent on") == 1


@pytest.mark.parametrize("stale_reset,fresh_reset,expected", [
    ("2026-11-22T01:00:00Z", "2026-11-22T05:00:00Z", "fable spent on 1 (earliest reset 22.11 01:00Z, seen 21.11 12:05Z)"),
    ("2026-11-22T05:00:00Z", "2026-11-22T01:00:00Z", "fable spent on 1 (earliest reset 22.11 01:00Z)"),
    ("2026-11-22T03:00:00Z", "2026-11-22T03:00:00Z", "fable spent on 1 (earliest reset 22.11 03:00Z)"),  # a tie: fresh
    ("2026-11-21T09:00:00Z", "2026-11-22T05:00:00Z", "fable spent on 1 (earliest reset 22.11 05:00Z)"),  # stale passed
])
@pytest.mark.parametrize("order", [1, -1])
def test_one_accounts_earliest_live_block_counts_whichever_row_reads_it(stale_reset, fresh_reset, expected, order):
    """Two rows of one account report the same model list: the earliest reset still ahead is the account's, whatever
    the rows' order; a fresh reading breaks a tie only."""
    block = {"applies_to_models": ["fable"]}
    rows = [_row("claude", "a", freshness="stale", observed=NOW - 3600, scoped=[{**block, "resets_at": stale_reset}]),
            _row("claude", "a", windows=[(0.2, WEEK)], scoped=[{**block, "resets_at": fresh_reset}])][::order]
    assert _body(_payload([_profile("claude", "a")], rows)).endswith(expected)


def test_a_reset_is_shown_to_the_nearest_minute_so_jitter_at_the_minute_is_no_change():
    def at_limit(reset):
        rows = [_row("claude", "a", state="exhausted", resets_at=reset, windows=[(1.0, WEEK)])]
        return _body(_payload([_profile("claude", "a")], rows))

    assert at_limit("2026-11-21T23:59:59.8Z") == at_limit("2026-11-22T00:00:00.2Z") == (
        "claude 1 at limit (earliest reset 22.11 00:00Z)")
    assert at_limit("2026-11-22T00:01:00Z") != at_limit("2026-11-22T00:00:00Z")


def test_room_without_a_fresh_measured_window_says_so():
    """The engine's verdict stands (with room); the text says no fresh window measured it."""
    rows = [_row("claude", "a", windows=[(0.4, WEEK, {"freshness": "stale"})])]
    assert _body(_payload([_profile("claude", "a")], rows)) == "claude 1 with room (no fresh measured window)"
    rows.append(_row("claude", "b", windows=[(0.2, WEEK)]))
    assert _body(_payload([_profile("claude", "a"), _profile("claude", "b")], rows)) == (
        "claude 2 with room (7d 20% used, 1 of 2 measured)")


def test_an_engine_name_with_a_row_separator_or_a_line_break_stays_one_part():
    name = "odd | name\nhere"
    rows = [_row(name, "a", scoped=[{"applies_to_models": ["m | x\ny"], "resets_at": "2026-11-22T03:00:00Z"}])]
    body = _body(_payload([_profile(name, "a")], rows))
    assert " | " not in body and "\n" not in body
    assert body == ("odd / name here 1 available (no measured general window), m / x y spent on 1 "
                    "(earliest reset 22.11 03:00Z)")


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
    assert failed.startswith("(subscription accounts, engine not read since 21.11 14:09Z (daemon_unreachable); counts "
                             "unchanged since 21.11 13:05:21Z; shares read 21.11 13:05Z): agy 1 available")
    assert failed.split("): ", 1)[1] == LIVE_LINE.split("): ", 1)[1]


# ---- the writer ----------------------------------------------------------------------------------------------------

def _fresh_pass():
    return {"confirmed_at": None, "failing_since": None, "dispatched": 0.0}


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
    monkeypatch.setattr(census, "_PASS", _fresh_pass())
    monkeypatch.setattr(claudexor_daemon, "owned_daemon_provisioned", lambda: True)
    monkeypatch.setattr(claudexor_passive, "_quota_payload", read)
    monkeypatch.setattr(claudexor_accounts, "account_read_revision", lambda: state["revision"])
    state["path"] = tmp_path / census.CENSUS_FILE
    state["request"] = tmp_path / census.REQUEST_FILE
    return state


def _failed_payload(code):
    payload = _payload([], reads=("not_read", "not_read"))
    payload["read_errors"] = {"discovery": {"code": code}}
    return payload


def _spend_the_room(payload):
    """The live read's one Claude account with room hits its limit: a count changes."""
    row = payload["quota"][0]
    row["constraints"][0]["used_ratio"] = 1.0
    row["availability"] = {"state": "exhausted", "blocking_constraints": [], "resets_at": _iso(NOW + 7200),
                           "model_scoped_exhaustions": []}


def test_an_unchanged_pass_writes_nothing_and_a_changed_one_writes(engine):
    assert census.refresh() == "written"
    stored = census.read_file()
    assert census.census_text(stored).split("): ", 1)[1] == LIVE_LINE.split("): ", 1)[1]
    assert stored["since"] == stored["read_at"] and set(stored) == {"type", "read", "code", "families", "since", "read_at"}
    before = (engine["path"].read_bytes(), engine["path"].stat().st_mtime_ns)
    assert census.refresh() == "unchanged" and engine["reads"] == 2
    assert (engine["path"].read_bytes(), engine["path"].stat().st_mtime_ns) == before
    _spend_the_room(engine["payload"])
    assert census.refresh() == "written" and engine["path"].read_bytes() != before[0]


def test_shares_alone_refresh_the_file_and_keep_since_and_the_signature(engine):
    """The owner's 1A: the part is due when the counts change, an account hit its limit or came back; the file still
    carries the latest shares and their read time for the next showing."""
    assert census.refresh() == "written"
    first = census.read_file()
    engine["now"] += 600
    engine["payload"]["quota"][0]["constraints"][0]["used_ratio"] = 0.91  # the account with room spent more
    assert census.refresh() == "refreshed"
    second = census.read_file()
    assert second["since"] == first["since"] and second["read_at"] != first["read_at"]
    assert "7d 91% used" in census.census_text(second) and "shares read 21.11 13:15Z" in census.census_text(second)
    assert census.census_signature(second) == census.census_signature(first)
    engine["now"] += 600
    _spend_the_room(engine["payload"])
    assert census.refresh() == "written"
    third = census.read_file()
    assert third["since"] == third["read_at"] != first["since"]
    assert census.census_signature(third) != census.census_signature(first)


def test_a_failed_read_names_the_last_confirmed_read_the_counts_epoch_and_the_shares_read_time(engine):
    """The counts began at 13:05:21 (88 %), the shares moved to 91 % at 13:15, an unchanged read confirmed them at 13:25:
    after the failure the text names each of the three times for what it dates."""
    assert census.refresh() == "written"
    engine["now"] += 600
    engine["payload"]["quota"][0]["constraints"][0]["used_ratio"] = 0.91
    assert census.refresh() == "refreshed"
    engine["now"] += 600
    assert census.refresh() == "unchanged"
    engine["payload"] = _failed_payload("daemon_unreachable")
    engine["now"] += 600
    assert census.refresh() == "failure_debounced"
    engine["now"] += 600
    assert census.refresh() == "failure_recorded"
    head, _sep, body = census.census_text(census.read_file()).partition("): ")
    assert head == ("(subscription accounts, engine not read since 21.11 13:25Z (daemon_unreachable); counts unchanged "
                    "since 21.11 13:05:21Z; shares read 21.11 13:15Z")
    assert "claude 1 with room (7d 91% used, 1 of 1 measured)" in body


def test_a_stale_blocks_moving_seen_refreshes_the_file_and_is_no_change(engine):
    """A stale observation time of an unchanged block (same profile, models and reset) is shown, never compared; a
    reset that moves is a change."""
    def stale_block(observed, reset="2026-12-02T00:00:00Z"):
        return _payload([_profile("cursor", "c")], [_row("cursor", "c", freshness="stale", observed=observed, scoped=[
            {"applies_to_models": ["gemini-4-carbon-high"], "resets_at": reset}])], [_absence("cursor", "c", "no_source")])

    engine["payload"] = stale_block(NOW - 7200)
    assert census.refresh() == "written"
    first = census.read_file()
    assert "seen 21.11 11:05Z" in census.census_text(first)
    engine["now"] += 600
    engine["payload"] = stale_block(NOW - 6600)
    assert census.refresh() == "refreshed"
    second = census.read_file()
    assert "seen 21.11 11:15Z" in census.census_text(second) and second["since"] == first["since"]
    assert census.census_signature(second) == census.census_signature(first)
    engine["now"] += 600
    engine["payload"] = stale_block(NOW - 6600, reset="2026-12-03T00:00:00Z")
    assert census.refresh() == "written"
    assert census.census_signature(census.read_file()) != census.census_signature(first)


def test_a_failed_replace_confirms_nothing(engine, monkeypatch):
    """A changed read whose file replace raised leaves the stored record and the last confirmation as they were: the
    failure recorded later dates the engine by the read that was stored."""
    assert census.refresh() == "written"
    first = census.read_file()
    engine["now"] += 600
    _spend_the_room(engine["payload"])

    def refuse(_record, _root):
        raise OSError("disk full")

    with monkeypatch.context() as patched:
        patched.setattr(census, "_write", refuse)
        assert census.refresh() == "error"
    assert census.read_file() == first and census._PASS["confirmed_at"] == NOW
    engine["payload"] = _failed_payload("daemon_unreachable")
    engine["now"] += 600
    assert census.refresh() == "failure_debounced"
    engine["now"] += 600
    assert census.refresh() == "failure_recorded"
    assert census.read_file()["confirmed_at"] == first["read_at"]


def test_a_count_change_moves_since_and_a_recovery_keeps_it(engine):
    assert census.refresh() == "written"
    first = census.read_file()
    engine["now"] += 600
    _spend_the_room(engine["payload"])
    assert census.refresh() == "written"
    second = census.read_file()
    assert second["since"] != first["since"] and "claude 17 at limit" in census.census_text(second)
    engine["now"] += 600
    good = engine["payload"]
    engine["payload"] = _failed_payload("daemon_unreachable")
    assert census.refresh() == "failure_debounced"
    engine["now"] += 600
    assert census.refresh() == "failure_recorded"
    failed = census.read_file()
    assert (failed["read"], failed["code"], failed["since"]) == ("failed", "daemon_unreachable", second["since"])
    assert failed["families"] == second["families"]  # the values stay
    assert failed["confirmed_at"] == second["read_at"]  # the last accepted read
    engine["payload"] = _failed_payload("observation_read_timeout")
    assert census.refresh() == "failure_unchanged"  # the code alternates; it is printed, never compared
    assert census.read_file()["code"] == "daemon_unreachable"
    engine["now"] += 600
    engine["payload"] = good
    assert census.refresh() == "recovered"
    recovered = census.read_file()
    assert recovered["read"] == "ok" and recovered["since"] == second["since"] and recovered["read_at"] != second["read_at"]


def test_a_failed_read_is_recorded_once_its_streak_has_lasted_the_duty_cadence(engine):
    """Two passes 15 s apart (a request during an engine's warm-up) record nothing; a streak of DUTY_SEC does."""
    assert census.refresh() == "written"
    engine["payload"] = _failed_payload("daemon_unreachable")
    assert census.refresh() == "failure_debounced"
    engine["now"] += census.REQUEST_MIN_SEC
    assert census.refresh() == "failure_debounced"
    engine["payload"] = _live_payload()
    assert census.refresh() == "unchanged"  # an accepted read ends the streak
    engine["payload"] = _failed_payload("daemon_unreachable")
    engine["now"] += 600
    assert census.refresh() == "failure_debounced"  # a new streak starts here
    engine["now"] += census.DUTY_SEC - 1
    assert census.refresh() == "failure_debounced"
    assert census.read_file()["read"] == "ok"
    engine["now"] += 1
    assert census.refresh() == "failure_recorded"
    assert census.read_file()["read"] == "failed"


def test_a_missing_descriptor_is_a_failed_read_that_keeps_the_values(engine, monkeypatch):
    assert census.refresh() == "written"
    values = census.read_file()["families"]
    monkeypatch.setattr(claudexor_daemon, "owned_daemon_provisioned", lambda: False)
    engine["payload"] = _failed_payload("daemon_not_discovered")
    assert census.refresh() == "failure_debounced"
    engine["now"] += 600
    assert census.refresh() == "failure_recorded"
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
    assert census.refresh() == "failure_debounced"
    engine["now"] += 600
    assert census.refresh() == "failure_without_values"
    assert not engine["path"].exists()


def test_an_empty_subscription_roster_is_a_stored_none_even_beside_a_failed_quota_read(engine):
    """The roster answered the membership question: no subscription account means none, whatever the quota read
    says; a roster with one still needs the quota read, and an unknown roster is a failed read."""
    assert census.refresh() == "written"
    keys_only = [_profile("openrouter", "key", kind="api_key")]
    engine["payload"] = _payload(keys_only, reads=("ok", "failed"))
    assert census.refresh() == "none"
    stored = census.read_file()
    assert engine["path"].exists() and stored["families"] == {} and stored["read"] == "ok"
    engine["payload"] = _payload(keys_only)
    assert census.refresh() == "unchanged"
    engine["payload"] = _payload([_profile("claude", "a")], reads=("ok", "failed"))
    assert census.refresh() == "failure_debounced"
    engine["now"] += 600
    engine["payload"] = _payload(keys_only, reads=("failed", "ok"))
    assert census.refresh() == "failure_recorded"
    assert census.read_file()["read"] == "failed"


def test_an_unhonored_scope_never_creates_a_file_and_an_existing_one_says_the_engine_was_not_read(engine):
    """An engine that no longer certifies the text scope (an older one) is no census: a file it finds is recorded as
    a failed read with its own code; that pass confirms nothing and resets no failed streak."""
    unhonored = _payload([_profile("claude", "a")], [_row("claude", "a", windows=[(0.3, WEEK)])], scope=None)
    engine["payload"] = unhonored
    assert census.refresh() == "failure_debounced"
    engine["now"] += 600
    assert census.refresh() == "failure_without_values" and not engine["path"].exists()
    engine["now"] += 600
    engine["payload"] = _live_payload()
    assert census.refresh() == "written"
    good = census.read_file()
    engine["now"] += 600
    engine["payload"] = unhonored
    assert census.refresh() == "failure_debounced"
    engine["now"] += 600
    assert census.refresh() == "failure_recorded"
    stored = census.read_file()
    assert (stored["read"], stored["code"], stored["confirmed_at"]) == ("failed", "availability_scope_unhonored",
                                                                        good["read_at"])
    assert "engine not read since 21.11 13:25Z (availability_scope_unhonored)" in census.census_text(stored)
    engine["now"] += 600
    engine["payload"] = _live_payload()
    assert census.refresh() == "recovered"
    engine["now"] += 600
    engine["payload"] = _failed_payload("daemon_unreachable")
    assert census.refresh() == "failure_debounced"
    engine["now"] += 300
    engine["payload"] = unhonored
    assert census.refresh() == "failure_debounced"  # inside the streak, never its end
    engine["now"] += 300
    engine["payload"] = _failed_payload("daemon_unreachable")
    assert census.refresh() == "failure_recorded"


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


def test_a_pending_file_whose_read_fails_stays_pending_with_the_failure(engine):
    assert census.refresh() == "written"
    census.account_changed(False)
    engine["payload"] = _failed_payload("daemon_unreachable")
    assert census.refresh() == "failure_debounced"
    engine["now"] += 600
    assert census.refresh() == "failure_recorded"
    stored = census.read_file()
    assert stored["pending"] is True and stored["read"] == "failed"


def test_a_read_crossed_by_a_confirmed_account_change_never_publishes_or_confirms(engine, monkeypatch):
    def crossed():
        engine["revision"] += 1  # a PATCH or DELETE confirmed while this read was in flight
        return _payload([_profile("claude", "a")], [_row("claude", "a", windows=[(0.3, WEEK)])])

    monkeypatch.setattr(claudexor_passive, "_quota_payload", crossed)
    assert census.refresh() == "roster_changed" and not engine["path"].exists()
    monkeypatch.setattr(claudexor_passive, "_quota_payload", lambda: engine["payload"])
    assert census.refresh() == "written"
    first = census.read_file()
    monkeypatch.setattr(claudexor_passive, "_quota_payload", crossed)
    engine["now"] += 600
    assert census.refresh() == "roster_changed" and census.read_file() == first
    monkeypatch.setattr(claudexor_passive, "_quota_payload", lambda: engine["payload"])
    engine["payload"] = _failed_payload("daemon_unreachable")
    engine["now"] += 600
    assert census.refresh() == "failure_debounced"
    engine["now"] += 600
    assert census.refresh() == "failure_recorded"
    assert census.read_file()["confirmed_at"] == first["read_at"]  # the discarded read confirmed nothing


def test_a_stop_during_the_read_writes_and_confirms_nothing(engine, monkeypatch):
    assert census.refresh() == "written"
    first, stop = census.read_file(), {"now": False}

    def read_then_stop():
        stop["now"] = True
        payload = _live_payload()
        _spend_the_room(payload)
        return payload

    monkeypatch.setattr(claudexor_passive, "_quota_payload", read_then_stop)
    engine["now"] += 600
    assert census.refresh(stop_requested=lambda: stop["now"]) == "stopped"
    assert census.read_file() == first and census._PASS["confirmed_at"] == NOW


def test_limit_codes_ask_for_a_read_unless_the_pool_is_structural_or_no_engine_exists(engine, monkeypatch):
    for code, problem in (("internal_error", None), ("", None), (None, None), ("run_failed", None),
                          ("rate_limited", None), (["rate_limited"], None),  # a plain rate limit: no read
                          ("credential_pool_exhausted", {"context": {"poolCause": "structural"}})):
        census.note_limit_failure(code, problem)
    assert census.request_mtime() is None
    for code in sorted(census.LIMIT_FAILURE_CODES):
        census.note_limit_failure(code, {"context": {"poolCause": "quota"}})
        assert json.loads(engine["request"].read_text(encoding="utf-8"))["reason"] == code
        engine["request"].unlink()
    monkeypatch.setattr(claudexor_daemon, "owned_daemon_provisioned", lambda: False)
    census.note_limit_failure("subscription_window_exhausted")
    census.account_changed(True)
    assert census.request_mtime() is None  # no engine was ever provisioned: nothing to refresh, nothing written


def test_a_refused_run_start_asks_for_a_read_and_a_classified_read_never_does(engine):
    """The transport's problem classifier stays free of writes (a catalog GET answers a dated pool the same way);
    the run start the engine refused on a limit is the event."""
    import httpx

    from ouroboros.gateways import claudexor as cx
    from tests._delegated_transport_shared import _gateway

    def answering(code, context, status):
        return lambda _request: httpx.Response(status, json={"code": code, "message": f"{code} refusal",
                                                             "retryable": False, "context": context})

    dated = ("credential_pool_exhausted", {"resetsAt": "2030-01-01T00:00:00Z"}, 409)
    with _gateway(answering(*dated)) as gateway:
        with pytest.raises(cx.ClaudexorSubscriptionWindowExhausted):
            gateway.get_run("run-1")
    for refusal in (("invalid_request", {}, 400), ("credential_pool_exhausted", {"poolCause": "structural"}, 409)):
        with _gateway(answering(*refusal)) as gateway:
            with pytest.raises(cx.ClaudexorUnavailable):
                gateway.start_run({"mode": "agent"})
    assert census.request_mtime() is None
    with _gateway(answering(*dated)) as gateway:
        with pytest.raises(cx.ClaudexorSubscriptionWindowExhausted):
            gateway.start_run({"mode": "agent"})
    assert json.loads(engine["request"].read_text(encoding="utf-8"))["reason"] == "credential_pool_exhausted"


def test_a_review_thread_turn_refused_on_a_limit_asks_for_a_read(engine):
    """A continuing review appends a thread turn instead of starting a run: its typed refusal asks, a structural or an
    unrelated one does not (no run exists to settle)."""
    import httpx

    from ouroboros.gateways import claudexor as cx
    from tests._delegated_transport_shared import _gateway

    def answering(code, context, status):
        return lambda _request: httpx.Response(status, json={"code": code, "message": f"{code} refusal",
                                                             "retryable": False, "context": context})

    turn = {"_thread_id": "thread-1", "mode": "agent"}
    for refusal in (("credential_pool_exhausted", {"poolCause": "structural"}, 409), ("invalid_request", {}, 400)):
        with _gateway(answering(*refusal)) as gateway:
            with pytest.raises(cx.ClaudexorUnavailable):
                gateway.start_run(turn, idempotency_key="turn-key")
    assert census.request_mtime() is None
    with _gateway(answering("subscription_window_exhausted", {"resetsAt": "2030-01-01T00:00:00Z"}, 429)) as gateway:
        with pytest.raises(cx.ClaudexorSubscriptionWindowExhausted):
            gateway.start_run(turn, idempotency_key="turn-key")
    assert json.loads(engine["request"].read_text(encoding="utf-8"))["reason"] == "subscription_window_exhausted"


def test_the_duty_is_due_on_its_cadence_while_someone_works_or_after_a_request(engine, monkeypatch):
    start = 1_800_000_000.0
    assert census.duty_due(start, working=True)
    assert not census.duty_due(start, working=False)  # an idle install, or a pending-only queue, reads nothing
    census._PASS["dispatched"] = start
    assert not census.duty_due(start + 599, working=True)
    assert census.duty_due(start + 600, working=True)
    census.request_refresh("test")
    assert not census.duty_due(start + 5, working=False)  # at most one pass per REQUEST_MIN_SEC
    assert census.duty_due(start + 15, working=False)  # a request on disk is answered, even on an idle install
    monkeypatch.setattr(claudexor_daemon, "owned_daemon_provisioned", lambda: False)
    assert not census.duty_due(start + 15, working=False)  # no engine and no file: nothing to read
    engine["path"].write_text("{}", encoding="utf-8")
    assert census.duty_due(start + 15, working=False)  # a census file to keep is read again


def test_a_dispatched_pass_takes_its_request_so_a_restart_never_answers_it_again(engine, monkeypatch):
    """The answered request leaves the disk with the dispatch, so a new process (its stamps at zero) reads nothing for
    it; a request left before that start, or written during a read, is still answered."""
    import threading

    latch, tick = threading.Lock(), 1_800_000_000.0

    def tick_at(seconds):
        census.dispatch(tick + seconds, False, latch, None, lambda: census.duty_pass(lambda: False, latch),
                        lambda *_args: pytest.fail("the duty's latch was held"))

    census.request_refresh("test")
    tick_at(0)
    assert engine["reads"] == 1 and census.request_mtime() is None
    monkeypatch.setattr(census, "_PASS", _fresh_pass())  # the next process
    tick_at(1)
    assert engine["reads"] == 1  # an answered request wakes nothing
    census.request_refresh("left before this start")
    monkeypatch.setattr(census, "_PASS", _fresh_pass())
    tick_at(2)
    assert engine["reads"] == 2

    def read_and_request():
        engine["reads"] += 1
        census.request_refresh("written during the read")
        return engine["payload"]

    monkeypatch.setattr(claudexor_passive, "_quota_payload", read_and_request)
    census.request_refresh("test")
    tick_at(20)
    assert engine["reads"] == 3 and census.request_mtime() is not None  # the newer request waits for its own pass


def test_any_attempted_read_answers_the_requests_before_it_and_only_a_newer_one_asks_again(engine):
    """A failed read consumes its request too, so an engine that does not answer (on 21.11 the managed one stalled
    with a full accept queue) is not asked every 15 s: after a failure the next read comes on a request newer than
    that attempt, or at the ordinary cadence."""
    import threading

    latch, tick = threading.Lock(), 1_800_000_000.0
    engine["payload"] = _failed_payload("daemon_unreachable")

    def tick_at(seconds):
        census.dispatch(tick + seconds, True, latch, None, lambda: census.duty_pass(lambda: False, latch),
                        lambda *_args: pytest.fail("the duty's latch was held"))

    def request_at(seconds):
        census.request_refresh("test")
        os.utime(engine["request"], (tick + seconds, tick + seconds))

    tick_at(0)
    assert engine["reads"] == 1  # the cadence: one read, which failed
    request_at(1)  # a request after that failure
    tick_at(16)
    assert engine["reads"] == 2  # one read answers it, and fails again
    for seconds in (31, 46, 90):
        tick_at(seconds)
    assert engine["reads"] == 2  # the failed read consumed the request: none 15 s later, none after
    request_at(95)  # a newer request
    tick_at(96)
    assert engine["reads"] == 3
    tick_at(111)
    assert engine["reads"] == 3
    tick_at(96 + census.DUTY_SEC)
    assert engine["reads"] == 4  # the ordinary cadence


def test_dispatch_stamps_and_starts_a_due_pass_or_reports_a_held_latch(engine):
    import threading

    latch, started, stalls = threading.Lock(), [], []
    tick = 1_800_000_000.0
    census.dispatch(tick, False, latch, None, lambda: started.append(1), lambda *args: stalls.append(args))
    assert not started and not latch.locked()  # not due: nothing
    census.dispatch(tick, True, latch, None, lambda: started.append(1), lambda *args: stalls.append(args))
    assert started == [1] and latch.locked() and census._PASS["dispatched"] == tick
    census._PASS["dispatched"] = 0.0
    census.dispatch(tick, True, latch, None, lambda: started.append(2), lambda *args: stalls.append(args))
    assert started == [1] and stalls == [(latch, census.DUTY_SEC)]  # its latch still held: the stall is reported


def test_the_census_duty_runs_one_pass_on_its_modules_root_and_releases_its_latch(engine, monkeypatch, tmp_path):
    import threading

    from ouroboros import server_maintenance as sm

    server_root = tmp_path / "server-root"
    monkeypatch.setattr(sm, "DATA_DIR", server_root)
    latch = threading.Lock()
    latch.acquire()
    sm._run_census_duty(None, latch)
    assert not latch.locked() and engine["reads"] == 1
    assert (server_root / census.CENSUS_FILE).exists() and not engine["path"].exists()  # the module's own root
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
